"""Target company profile: validated, with explicit assumptions."""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, Field, field_validator

from . import config as C
from .errors import ConfigError

if TYPE_CHECKING:
    from .data import Dataset


class ProfileError(ConfigError):
    """The target profile is invalid for this dataset."""


def _match(value: str, choices: list[str], what: str) -> str:
    lookup = {c.casefold(): c for c in choices}
    if value.casefold() in lookup:
        return lookup[value.casefold()]
    close = difflib.get_close_matches(value, choices, n=3, cutoff=0.5)
    raise ProfileError(f"Unknown {what} '{value}'. Did you mean: {close or choices}?")


DEFAULT_NAME = "Project Atlas (Healthcare Services target)"
DEFAULT_SECTOR = "Healthcare Services"


class TargetProfile(BaseModel):
    """The company being sold. Categorical values are checked against the data by validate_against()."""

    name: str = DEFAULT_NAME
    sector: str = DEFAULT_SECTOR
    ev_mm: float = Field(200.0, gt=0)
    geography: str = "Regional"  # "Regional" = a single, unspecified US region
    ownership: str = "Private"
    ebitda_margin_pct: float | None = None  # None -> "strong" = dataset 75th percentile
    notes: str = "Mid-market, private, regional, strong EBITDA margins"

    @field_validator("sector", "geography", "ownership", "name", "notes")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    @property
    def slug(self) -> str:
        """Filesystem-safe id of the target, used for the output folder and the feedback file."""
        geo = self.geography.lower().replace(" ", "-")
        base = f"{self.sector}-{self.ev_mm:g}m-{geo}-{self.ownership}"
        return re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")


def validate_against(profile: TargetProfile, ds: Dataset) -> TargetProfile:
    """Check the profile's categorical values against THIS file's vocabulary (case-insensitive).

    Raises ProfileError with close matches. A geography or ownership the file cannot express (column
    missing) is reset to a neutral value rather than rejected.
    """
    v = ds.vocab
    if v is None:  # a Dataset built without vocabulary (tests/tooling): nothing to validate against
        return profile
    updates: dict[str, str] = {"sector": _match(profile.sector, list(v.sectors), "sector")}
    if profile.geography.casefold() == "regional" or not v.geographies:
        updates["geography"] = "Regional"
    else:
        updates["geography"] = _match(profile.geography, list(v.geographies), "geography")
    if profile.name == DEFAULT_NAME and updates["sector"] != DEFAULT_SECTOR:  # don't keep a healthcare title
        updates["name"] = f"{updates['sector']} target (~${profile.ev_mm:g}M)"
    choices = list(v.ownership) or list(C.OWNERSHIP_W)
    updates["ownership"] = _match(profile.ownership, choices, "ownership") if v.ownership else profile.ownership
    return profile.model_copy(update=updates)


def load_yaml(path: Path | str) -> TargetProfile:
    """Read a target profile YAML. Raises ProfileError if the file is missing or not a mapping."""
    path = Path(path)
    if not path.is_file():
        raise ProfileError(f"Target file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ProfileError(f"Target file must be a mapping: {path}")
    return TargetProfile(**data)
