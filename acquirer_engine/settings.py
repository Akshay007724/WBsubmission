"""Validated overrides for the tunable constants in config.py.

Precedence: defaults (config.py) < --config FILE.yaml < environment (ACQ_<NAME>) < --set NAME=VALUE.
Only names in TUNABLE can be overridden; values are type-checked against the default's type with
pydantic, so 'MIN_COMPS=abc' fails with a clear message instead of crashing mid-run.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError, create_model

from . import config as C
from .errors import ConfigError

TUNABLE = (
    # deal relevance
    "SIZE_SIGMA",
    "RECENCY_HALF_LIFE_Y",
    "STALE_PENDING_YEARS",
    "STALE_PENDING_W",
    "MARGIN_SIGMA_PP",
    "GEO_SAME",
    "GEO_BROAD",
    "GEO_OTHER",
    "DEAL_TYPE_EXPONENT",
    "STRONG_MARGIN_PCTL",
    # acquirer score and selection
    "SHRINKAGE_K",
    "W_MASS",
    "W_FIT",
    "W_BEST",
    "W_EXEC_PENALTY",
    "MIN_DEAL_RELEVANCE",
    "MIN_PER_TYPE",
    # evidence and comps
    "ADJACENT_SECTOR_MIN",
    "COMP_SIZE_BAND",
    "COMP_WIDE_BAND",
    "MIN_COMPS",
    "TOP_PRECEDENTS",
    "RELEVANT_DEAL_W",
    "RECENT_YEARS",
    "PRICING_IN_LINE_TURNS",
    # size role
    "ADD_ON_PCTL",
    "ADD_ON_MEDIAN_FRACTION",
    "STRETCH_P90_MULTIPLE",
    "PLATFORM_EVIDENCE_MIN",
    "PLATFORM_MATCH_MULTIPLE",
    # conviction and checker
    "CONV_HIGH",
    "CONV_MEDIUM",
    "DISTINCT_JACCARD_MAX",
    "REPEATED_RISK_MIN_PAGES",
    # LLM
    "ANTHROPIC_EFFORT",
    "ANTHROPIC_WORKERS",
    "LLM_TIMEOUT_S",
    "LLM_MAX_RETRIES",
)
DEFAULTS: dict[str, Any] = {k: getattr(C, k) for k in TUNABLE}
_ACTIVE: dict[str, Any] = {}
_Model = create_model(  # type: ignore[call-overload]  # dynamic model: one optional field per tunable
    "Settings",
    **{k: (type(v) | None if v is not None else str | None, None) for k, v in DEFAULTS.items()},
    __config__={"extra": "forbid"},
)


def _from_file(path: Path | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"cannot read --config {path}: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"--config {path} must be a mapping of NAME: value")
    return {str(k).upper(): v for k, v in raw.items()}


def _from_env() -> dict[str, Any]:
    return {k[4:]: v for k, v in os.environ.items() if k.startswith("ACQ_") and k[4:] in DEFAULTS}


def _from_sets(sets: list[str]) -> dict[str, Any]:
    out = {}
    for item in sets:
        if "=" not in item:
            raise ConfigError(f"--set expects NAME=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip().upper()] = v.strip()
    return out


def apply(config_file: Path | None = None, sets: list[str] | None = None) -> dict[str, Any]:
    """Reset to defaults, merge file < env < --set, validate, and write the result into config.py.

    Returns the overrides that differ from the defaults. Raises ConfigError on unknown names or bad types.
    """
    merged = {**_from_file(config_file), **_from_env(), **_from_sets(sets or [])}
    unknown = sorted(set(merged) - set(DEFAULTS))
    if unknown:
        raise ConfigError(f"unknown setting(s) {unknown}; tunable: {', '.join(TUNABLE)}")
    try:
        validated = _Model(**merged).model_dump(exclude_unset=True)
    except ValidationError as e:
        err = e.errors()[0]
        raise ConfigError(f"setting {err['loc'][0]}: {err['msg']}") from e
    if (w := {**DEFAULTS, **validated}) and abs(w["W_MASS"] + w["W_FIT"] + w["W_BEST"] - 1) > 1e-6:
        raise ConfigError("W_MASS + W_FIT + W_BEST must sum to 1")
    reset()
    for k, v in validated.items():
        setattr(C, k, v)
    _ACTIVE.clear()
    _ACTIVE.update({k: v for k, v in validated.items() if v != DEFAULTS[k]})
    return dict(_ACTIVE)


def reset() -> None:
    """Restore every tunable to its config.py default (used between runs and in tests)."""
    for k, v in DEFAULTS.items():
        setattr(C, k, v)
    _ACTIVE.clear()


def effective() -> dict[str, Any]:
    """Run-stamp view: the overrides in force, plus a fingerprint of every tunable value."""
    values = {k: getattr(C, k) for k in TUNABLE}
    fp = hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return {"overrides": dict(_ACTIVE), "fingerprint": fp}
