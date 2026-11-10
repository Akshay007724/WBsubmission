"""Canonical schema for M&A transaction files, and mapping of arbitrary headers onto it.

Each canonical field has a tier:
  core       - needed to rank at all; missing -> the run stops with a clear message
  important  - missing switches off a feature (e.g. no valuation without multiples)
  optional   - missing loses a detail (e.g. no competitive-process risk without bidder counts)

Headers are matched in order: --schema-map override, exact canonical name, alias table, then fuzzy
match (difflib ratio >= FUZZY_MIN). Every mapping records its method and confidence for the EDA.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

CORE, IMPORTANT, OPTIONAL = "core", "important", "optional"
FUZZY_MIN = 0.80


@dataclass(frozen=True)
class Field:
    """One canonical column: name, tier (core/important/optional), kind, aliases, and the feature it enables."""

    name: str
    tier: str
    kind: str  # text | number | money | pct | date | year
    aliases: tuple[str, ...] = ()
    feature: str = ""  # feature switched off when this field is missing


FIELDS: tuple[Field, ...] = (
    Field("acquirer", CORE, "text", ("buyer", "acquiror", "acquirer_name", "buyer_name", "purchaser", "bidder")),
    Field("target_company", CORE, "text", ("target", "target_name", "company", "seller", "acquired_company")),
    Field("sector", CORE, "text", ("industry", "target_sector", "sector_name", "vertical", "segment")),
    Field(
        "deal_size_mm",
        CORE,
        "money",
        (
            "deal_size",
            "ev_usd_m",
            "ev_mm",
            "enterprise_value",
            "enterprise_value_mm",
            "deal_value",
            "deal_value_mm",
            "deal_value_usd",
            "deal_value_bn",
            "transaction_value",
            "transaction_value_usd_m",
            "ev",
            "size_mm",
            "ev_usd",
            "value_usd_m",
            "price_mm",
        ),
    ),
    Field(
        "deal_year",
        CORE,
        "year",
        (
            "year",
            "announce_year",
            "announced_year",
            "deal_date",
            "announce_date",
            "announcement_date",
            "date_announced",
            "date",
            "close_date",
        ),
    ),
    Field(
        "ev_ebitda_multiple",
        IMPORTANT,
        "number",
        ("ev_ebitda", "ev_to_ebitda", "ebitda_multiple", "ev_ebitda_x"),
        "valuation",
    ),
    Field(
        "ev_revenue_multiple",
        IMPORTANT,
        "number",
        ("ev_revenue", "ev_to_revenue", "ev_sales", "revenue_multiple", "ev_rev"),
        "valuation",
    ),
    Field("outcome", IMPORTANT, "text", ("status", "deal_status", "result", "deal_outcome"), "outcome"),
    Field(
        "acquirer_type",
        IMPORTANT,
        "text",
        ("buyer_type", "acquiror_type", "investor_type", "acquirer_category"),
        "buyer_mix",
    ),
    Field("geography", IMPORTANT, "text", ("region", "geo", "location", "target_region", "state"), "geography"),
    Field(
        "ebitda_margin_pct",
        IMPORTANT,
        "pct",
        ("ebitda_margin", "margin", "margin_pct", "ebitda_margin_percent"),
        "margin",
    ),
    Field("transaction_id", OPTIONAL, "text", ("deal_id", "id", "txn_id", "transaction", "deal_number")),
    Field("deal_quarter", OPTIONAL, "text", ("quarter", "qtr", "announce_quarter")),
    Field("deal_type", OPTIONAL, "text", ("type", "transaction_type", "deal_structure"), "deal_type"),
    Field("financing_type", OPTIONAL, "text", ("financing", "consideration", "payment_type"), "financing"),
    Field(
        "target_revenue_mm", OPTIONAL, "money", ("revenue", "target_revenue", "revenue_mm", "sales", "revenue_usd_m")
    ),
    Field("target_ebitda_mm", OPTIONAL, "money", ("ebitda", "target_ebitda", "ebitda_mm", "ebitda_usd_m")),
    Field("revenue_growth_pct", OPTIONAL, "pct", ("revenue_growth", "growth", "growth_pct", "yoy_growth")),
    Field("synergy_pct_of_deal", OPTIONAL, "pct", ("synergy_pct", "synergies_pct", "synergy")),
    Field("num_bidders", OPTIONAL, "number", ("bidders", "number_of_bidders", "n_bidders"), "competitive_process"),
    Field("days_to_close", OPTIONAL, "number", ("time_to_close", "days_close", "closing_days")),
    Field(
        "strategic_rationale_tags", OPTIONAL, "text", ("tags", "rationale", "rationale_tags", "deal_rationale"), "tags"
    ),
    Field(
        "target_ownership_pre",
        OPTIONAL,
        "text",
        ("ownership", "seller_type", "target_ownership", "ownership_type"),
        "ownership",
    ),
    Field("sub_sector", OPTIONAL, "text", ("subsector", "sub_industry", "subindustry")),
    Field("currency", OPTIONAL, "text", ("ccy", "deal_currency", "currency_code")),
)
BY_NAME = {f.name: f for f in FIELDS}


class SchemaError(ValueError):
    """Raised when a core field cannot be found, or a --schema-map is invalid."""


@dataclass(frozen=True)
class Match:
    """How a canonical field was found in the CSV: source header, method and confidence."""

    canonical: str
    source: str
    method: str  # override | exact | alias | fuzzy
    confidence: float


def norm(header: str) -> str:
    """Lower-case, non-alphanumerics to '_' (the form every comparison uses)."""
    return re.sub(r"[^a-z0-9]+", "_", str(header).strip().lower()).strip("_")


def load_schema_map(path: Path | str) -> dict[str, str]:
    """Read a YAML mapping of canonical_name: source_header. Unknown canonical names are rejected."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SchemaError(f"--schema-map must be a mapping of canonical_name: csv_header ({path})")
    unknown = sorted(set(data) - set(BY_NAME))
    if unknown:
        raise SchemaError(f"--schema-map has unknown canonical names {unknown}; valid: {sorted(BY_NAME)}")
    return {str(k): str(v) for k, v in data.items()}


def map_headers(headers: list[str], overrides: dict[str, str] | None = None) -> tuple[dict[str, Match], list[str]]:
    """Map source headers onto canonical fields. Returns (matches by canonical name, unused headers).

    Each source header is used at most once; earlier fields in FIELDS win ties, so 'deal_size' is not
    stolen by a fuzzy match from a later field.
    """
    by_norm = {norm(h): h for h in headers}
    used: set[str] = set()
    out: dict[str, Match] = {}

    def take(f: Field, src: str, method: str, conf: float) -> None:
        out[f.name] = Match(f.name, src, method, round(conf, 2))
        used.add(src)

    for name, src in (overrides or {}).items():
        if src not in headers and norm(src) not in by_norm:
            raise SchemaError(f"--schema-map: column '{src}' for {name} not in the CSV; headers are {headers}")
        take(BY_NAME[name], src if src in headers else by_norm[norm(src)], "override", 1.0)
    for f in FIELDS:  # exact, then alias
        if f.name in out:
            continue
        if f.name in by_norm and by_norm[f.name] not in used:
            take(f, by_norm[f.name], "exact", 1.0)
            continue
        alias = next((by_norm[a] for a in f.aliases if a in by_norm and by_norm[a] not in used), None)
        if alias:
            take(f, alias, "alias", 0.95)
    for f in FIELDS:  # fuzzy last, so exact/alias claims win
        if f.name in out:
            continue
        best, score = None, 0.0
        for n, src in by_norm.items():
            if src in used:
                continue
            s = max(difflib.SequenceMatcher(None, n, cand).ratio() for cand in (f.name, *f.aliases))
            if s > score:
                best, score = src, s
        if best is not None and score >= FUZZY_MIN:
            take(f, best, "fuzzy", score)
    return out, [h for h in headers if h not in used]


def closest_header(field: str, headers: list[str]) -> str | None:
    """Best guess for an error message: the header most similar to a missing field."""
    f = BY_NAME[field]
    scored = [
        (max(difflib.SequenceMatcher(None, norm(h), c).ratio() for c in (f.name, *f.aliases)), h) for h in headers
    ]
    return max(scored)[1] if scored else None
