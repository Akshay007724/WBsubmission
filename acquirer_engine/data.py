"""Load any M&A transactions CSV: map headers, normalise units, degrade gracefully, build vocabulary.

Never trust the file. Every drop, conversion and switched-off feature is recorded as a note. Missing
values stay missing (NaN / "unknown"); nothing is filled with invented numbers.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import units
from .errors import DataError
from .schema import BY_NAME, CORE, FIELDS, Match, SchemaError, closest_header, map_headers

__all__ = ["DataError", "Dataset", "Vocab", "load"]

OUTCOME_SYNONYMS = {
    "closed": "Closed",
    "completed": "Closed",
    "complete": "Closed",
    "done": "Closed",
    "effective": "Closed",
    "pending": "Pending",
    "announced": "Pending",
    "signed": "Pending",
    "in progress": "Pending",
    "withdrawn": "Withdrawn",
    "abandoned": "Withdrawn",
    "cancelled": "Withdrawn",
    "canceled": "Withdrawn",
    "terminated": "Terminated",
    "rumored": "Rumored",
    "rumoured": "Rumored",
    "rumor": "Rumored",
}
ACQUIRER_TYPE_SYNONYMS = {
    "financial sponsor": "Financial Sponsor",
    "sponsor": "Financial Sponsor",
    "pe": "Financial Sponsor",
    "private equity": "Financial Sponsor",
    "financial": "Financial Sponsor",
    "financial buyer": "Financial Sponsor",
    "strategic": "Strategic",
    "corporate": "Strategic",
    "strategic buyer": "Strategic",
    "corporate buyer": "Strategic",
}
SPONSOR_NAME_RE = re.compile(
    r"\b(?:capital|partners|equity|ventures|investments?|holdings|advisors|fund|l\.?p\.?)\b", re.IGNORECASE
)
BROAD_GEO_RE = re.compile(r"national|nationwide|multi|global|international|various|cross[- ]border", re.IGNORECASE)
UNKNOWN = "unknown"


@dataclass(frozen=True)
class Vocab:
    """Categorical vocabulary of THIS file, plus the sector adjacency used for ranking."""

    sectors: tuple[str, ...]
    geographies: tuple[str, ...]
    broad_geographies: frozenset[str]
    ownership: tuple[str, ...]
    deal_types: tuple[str, ...]
    sector_preset: str | None  # "healthcare" when every sector is a preset sector
    adjacency: dict[tuple[str, str], float]
    adjacency_derived: bool

    def adj(self, a: str, b: str) -> float:
        """Sector similarity in [0, 1]; 1 on the diagonal."""
        if a == b:
            return 1.0
        return self.adjacency.get((a, b), C.SECTOR_ADJ_DEFAULT)


@dataclass(frozen=True)
class Dataset:
    """A loaded, validated file plus everything learned while loading it (notes, features, schema, vocabulary)."""

    df: pd.DataFrame
    sha256: str
    ref_year: float
    notes: tuple[str, ...] = field(default_factory=tuple)
    vocab: Vocab | None = None
    features: dict[str, str] = field(default_factory=dict)  # feature -> "on" | "computed" | "inferred" | "off: why"
    schema: dict[str, Match] = field(default_factory=dict)
    unmapped: tuple[str, ...] = ()
    load_stats: dict[str, int] = field(default_factory=dict)

    def on(self, feature: str) -> bool:
        """True if the feature is available (natively, computed or inferred)."""
        return not self.features.get(feature, "on").startswith("off")


def _clean_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True).fillna("")


def _read(path: Path) -> tuple[pd.DataFrame, bytes]:
    if not path.is_file():
        raise DataError(f"Data file not found: {path}")
    raw = path.read_bytes()
    if not raw.strip():
        raise DataError(f"Data file is empty: {path}")
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False, skip_blank_lines=True)
    except (pd.errors.ParserError, UnicodeDecodeError) as e:
        raise DataError(f"Could not parse {path.name} as CSV: {e}") from e
    if df.empty:
        raise DataError("CSV has a header but no rows")
    blank_rows = df.apply(lambda r: all(not str(v).strip() for v in r), axis=1)
    df = df[~blank_rows].reset_index(drop=True)
    df.attrs["blank_rows"] = int(blank_rows.sum())
    return df, raw


def _apply_schema(
    df: pd.DataFrame, overrides: dict[str, str] | None
) -> tuple[pd.DataFrame, dict[str, Match], list[str]]:
    headers = list(df.columns)
    try:
        matches, unmapped = map_headers(headers, overrides)
    except SchemaError as e:
        raise DataError(str(e)) from e
    missing_core = [f.name for f in FIELDS if f.tier == CORE and f.name not in matches]
    if missing_core:
        hints = "; ".join(
            f"'{m}' (closest unused header: '{closest_header(m, unmapped) or 'none'}')" for m in missing_core
        )
        raise DataError(
            f"CSV is missing core column(s) needed to rank: {hints}. "
            "Rename the column or pass --schema-map mapping.yaml"
        )
    out = pd.DataFrame({name: df[m.source] for name, m in matches.items()})
    return out, matches, unmapped


def _convert(df: pd.DataFrame, matches: dict[str, Match]) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    notes: list[str] = []
    failures: dict[str, int] = {}
    for name in list(df.columns):
        f, src = BY_NAME[name], matches[name].source
        if f.kind == "text":
            df[name] = _clean_str(df[name])
        elif f.kind == "number":
            df[name], failures[name] = units.parse_number(df[name])
        elif f.kind == "pct":
            vals, failures[name] = units.parse_number(df[name])
            df[name], n = units.normalise_pct(vals, src)
            notes += n
        elif f.kind == "money":
            df[name], failures[name], n = units.parse_money(df[name], src)
            notes += n
        elif f.kind == "year":
            year, quarter, failures[name], n = units.parse_year_quarter(df[name], src)
            df[name] = year
            notes += n
            if quarter is not None and "deal_quarter" not in df.columns:
                df["deal_quarter"] = quarter.fillna("")
    for name, count in failures.items():
        if count:
            notes.append(
                f"{matches[name].source}: {count} value(s) could not be read as numbers/dates and were left blank"
            )
    return df, notes, {k: v for k, v in failures.items() if v}


def _features(df: pd.DataFrame, infer_types: bool) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    """Fill absent columns with 'unknown'/NaN (never invented numbers) and record what that switches off."""
    feats: dict[str, str] = {}
    notes: list[str] = []
    for f in FIELDS:
        if f.name in df.columns:
            continue
        df[f.name] = np.nan if f.kind in ("number", "pct", "money", "year") else ""
        if f.feature:
            feats.setdefault(f.feature, f"off: no '{f.name}' column")

    # Multiples: compute from EV and EBITDA / revenue when the file has those instead.
    for mult, base in (("ev_ebitda_multiple", "target_ebitda_mm"), ("ev_revenue_multiple", "target_revenue_mm")):
        if df[mult].isna().all() and df[base].notna().any():
            denom = df[base].where(df[base] > 0)
            df[mult] = (df["deal_size_mm"] / denom).round(1)
            notes.append(
                f"{mult} computed as deal size / {base} ({int(df[mult].notna().sum())} rows); labelled computed"
            )
            feats["valuation"] = "computed"
    if df["ev_ebitda_multiple"].isna().all() and df["ev_revenue_multiple"].isna().all():
        feats["valuation"] = "off: no multiples, and no EBITDA/revenue to compute them"

    raw_types = df["acquirer_type"].str.lower().map(ACQUIRER_TYPE_SYNONYMS)
    df["acquirer_type"] = raw_types.fillna(df["acquirer_type"]).where(df["acquirer_type"] != "", UNKNOWN)
    if (df["acquirer_type"] == UNKNOWN).all():
        if infer_types:
            df["acquirer_type"] = np.where(
                df["acquirer"].str.contains(SPONSOR_NAME_RE), "Financial Sponsor", "Strategic"
            )
            feats["buyer_mix"] = "inferred"
            notes.append(
                "acquirer_type inferred from acquirer names (Capital/Partners/Equity/... -> Financial Sponsor); "
                "labelled inferred"
            )
        else:
            feats["buyer_mix"] = "off: no acquirer_type column (use --infer-acquirer-type to infer from names)"

    mapped = df["outcome"].str.lower().map(OUTCOME_SYNONYMS)
    df["outcome"] = mapped.fillna(df["outcome"]).where(df["outcome"] != "", UNKNOWN)
    if "outcome" in feats:
        notes.append(
            "No outcome column: every deal is treated as completed for comps and eligibility "
            "(stated, not assumed silently)"
        )
    if feats.get("margin", "").startswith("off"):
        notes.append("No margin column: margin similarity and the high-margin valuation anchor are switched off")
    return df, feats, notes


def _derived_adjacency(df: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Sector similarity from co-investment: cosine of sector vectors over acquirer deal counts.

    Two sectors are close when the same acquirers buy in both. Mapped to [ADJ_DERIVED_MIN, ADJ_DERIVED_MAX]
    so derived values sit on the same scale as the hand-set healthcare preset.
    """
    m = pd.crosstab(df["sector"], df["acquirer_key"]).astype(float)
    norms = np.sqrt((m**2).sum(axis=1)).replace(0, np.nan)
    cos = (m @ m.T) / np.outer(norms, norms)
    lo, hi = C.ADJ_DERIVED_MIN, C.ADJ_DERIVED_MAX
    out = {}
    for a in cos.index:
        for b in cos.columns:
            if a != b and pd.notna(cos.loc[a, b]):
                out[(a, b)] = round(lo + (hi - lo) * float(cos.loc[a, b]), 2)
    return out


def _vocab(df: pd.DataFrame) -> tuple[Vocab, list[str]]:
    notes: list[str] = []
    sectors = tuple(sorted(s for s in df["sector"].unique() if s))
    preset = "healthcare" if set(sectors) <= set(C.SECTORS) else None
    if preset:
        adjacency, derived = dict(C.SECTOR_ADJ_PAIRS), False
    else:
        adjacency, derived = _derived_adjacency(df), True
        notes.append(
            f"Sectors are not the healthcare preset ({len(sectors)} sectors); adjacency derived from which "
            "acquirers buy in which sectors (see EDA)"
        )
    geos = tuple(sorted(g for g in df["geography"].unique() if g))
    broad = frozenset(g for g in geos if g in C.BROAD_GEOS or BROAD_GEO_RE.search(g))
    for col, known, what in (
        ("target_ownership_pre", C.OWNERSHIP_W, "ownership"),
        ("deal_type", C.DEAL_TYPE_W, "deal type"),
        ("outcome", C.OUTCOME_W, "outcome"),
    ):
        unknown = sorted({v for v in df[col].unique() if v and v != UNKNOWN} - set(known))
        if unknown:
            notes.append(f"Unrecognised {what} values get a neutral weight: {unknown[:8]}")
    vocab = Vocab(
        sectors=sectors,
        geographies=geos,
        broad_geographies=broad,
        ownership=tuple(sorted(o for o in df["target_ownership_pre"].unique() if o)),
        deal_types=tuple(sorted(t for t in df["deal_type"].unique() if t)),
        sector_preset=preset,
        adjacency=adjacency,
        adjacency_derived=derived,
    )
    return vocab, notes


def load(
    path: Path | str = C.DEFAULT_DATA, schema_map: dict[str, str] | None = None, infer_acquirer_type: bool = False
) -> Dataset:
    """Read, map, convert and validate a transactions CSV.

    Raises DataError for an unreadable/empty file, a missing core column (naming the closest header)
    or no usable rows. Everything else degrades with a note.
    """
    path = Path(path)
    df, raw = _read(path)
    rows_read = len(df)
    blank = df.attrs.get("blank_rows", 0)
    df, matches, unmapped = _apply_schema(df, schema_map)
    df, notes, failures = _convert(df, matches)
    df, feats, more = _features(df, infer_acquirer_type)
    notes += more
    if blank:
        notes.insert(0, f"Skipped {blank} blank row(s)")

    n0 = len(df)
    df = df.drop_duplicates()
    dups = n0 - len(df)
    if dups:
        notes.append(f"Dropped {dups} exact duplicate rows")
    if (df["transaction_id"] == "").all():
        df["transaction_id"] = [f"ROW-{i:05d}" for i in range(1, len(df) + 1)]
        notes.append("No transaction_id column: generated ROW-00001... ids so deals can still be cited")
    dup_ids = df["transaction_id"].duplicated(keep="first") & (df["transaction_id"] != "")
    if dup_ids.any():
        notes.append(f"Dropped {int(dup_ids.sum())} rows with duplicate transaction_id")
        df = df[~dup_ids]

    bad = (
        df["deal_size_mm"].isna()
        | (df["deal_size_mm"] <= 0)
        | (df["acquirer"] == "")
        | df["deal_year"].isna()
        | (df["sector"] == "")
    )
    if bad.any():
        notes.append(
            f"Dropped {int(bad.sum())} rows with missing/non-positive deal size, or missing acquirer, sector or year"
        )
        df = df[~bad]
    if df.empty:
        raise DataError("No usable rows after validation (every row lacked a deal size, acquirer, sector or year)")
    bad_mult = df["ev_ebitda_multiple"].isna() | df["ev_revenue_multiple"].isna()
    if bad_mult.any() and not feats.get("valuation", "").startswith("off"):
        notes.append(f"{int(bad_mult.sum())} rows lack multiples; excluded from valuation comps only")
    if "currency" in matches:
        foreign = df["currency"].str.upper().isin(["", "USD", "US$", "$"]).__invert__()
        if foreign.any():
            notes.append(
                f"{int(foreign.sum())} rows are in non-USD currencies "
                f"({sorted(set(df.loc[foreign, 'currency']))[:5]}); "
                "not FX-converted, so their sizes are not comparable"
            )

    # Canonical acquirer key merges casing variants; keep the most common display spelling.
    df["acquirer_key"] = df["acquirer"].str.casefold()
    variants = int(df.groupby("acquirer_key")["acquirer"].nunique().sub(1).sum())
    if variants:
        notes.append(f"Merged {variants} acquirer name variant(s) that differ only in case/whitespace")
    display = df.groupby("acquirer_key")["acquirer"].agg(lambda s: s.mode().iat[0])
    df["acquirer"] = df["acquirer_key"].map(display)
    q = df["deal_quarter"].astype("string").str.extract(r"(\d)")[0].astype(float).fillna(2.5)
    df["deal_time"] = df["deal_year"] + (q - 1) / 4
    df["tags"] = df["strategic_rationale_tags"].apply(
        lambda s: tuple(t.strip() for t in str(s).split("|") if t.strip())
    )
    df = df.reset_index(drop=True)
    vocab, vnotes = _vocab(df)
    return Dataset(
        df=df,
        sha256=hashlib.sha256(raw).hexdigest(),
        ref_year=float(df["deal_year"].max()),
        notes=tuple(notes + vnotes),
        vocab=vocab,
        features=feats,
        schema=matches,
        unmapped=tuple(unmapped),
        load_stats={
            "rows_read": rows_read,
            "rows_used": len(df),
            "duplicates_dropped": dups,
            **{f"coercion_failures_{k}": v for k, v in failures.items()},
        },
    )
