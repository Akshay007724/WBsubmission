"""Unit and format normalisation. Every conversion returns a human-readable note; nothing is guessed silently.

Deal sizes end up in USD millions, percentages in 0-100, dates as a year plus quarter.
"""

from __future__ import annotations

import re

import pandas as pd

MONEY_SUFFIX = {
    "k": 1e-3,
    "thousand": 1e-3,
    "m": 1.0,
    "mm": 1.0,
    "mn": 1.0,
    "million": 1.0,
    "b": 1e3,
    "bn": 1e3,
    "billion": 1e3,
}
_MONEY_RE = re.compile(r"^\s*[$€£]?\s*(-?[\d,]*\.?\d+)\s*([a-zA-Z]*)\s*$")
DOLLARS_MEDIAN_MIN = 1e5  # median >= 100,000 can only be raw dollars when the unit is millions
BILLIONS_MEDIAN_MAX = 1.0  # median < 1 with no header hint: deals are in billions (a $0.2M median is implausible)


def parse_number(s: pd.Series) -> tuple[pd.Series, int]:
    """Strings like '1,234.5', '21%', '$350M' -> float. Returns (values, count of failed coercions).

    Suffixes are kept in a side channel by parse_money; here they are stripped. Blank stays NaN.
    """
    raw = s.astype("string").str.strip()
    blank = raw.isna() | (raw == "")
    cleaned = raw.str.replace(r"[,$€£%\s]", "", regex=True)
    vals = pd.to_numeric(cleaned, errors="coerce")
    failed = int((vals.isna() & ~blank).sum())
    return vals.astype(float), failed


def parse_money(s: pd.Series, header: str) -> tuple[pd.Series, int, list[str]]:
    """Money column -> USD millions. Uses per-cell suffixes ($1.2B), then header hints, then magnitude."""
    notes: list[str] = []
    raw = s.astype("string").str.strip()
    blank = raw.isna() | (raw == "")
    parts = raw.str.extract(_MONEY_RE)
    num = pd.to_numeric(parts[0].str.replace(",", "", regex=False), errors="coerce")
    suffix = parts[1].fillna("").str.lower()
    unknown_suffix = suffix.ne("") & ~suffix.isin(MONEY_SUFFIX)
    num = num.where(~unknown_suffix)
    failed = int((num.isna() & ~blank).sum())
    has_suffix = suffix.isin(MONEY_SUFFIX) & suffix.ne("")
    if has_suffix.any():
        num = num.where(~has_suffix, num * suffix.map(MONEY_SUFFIX).astype(float))
        notes.append(f"{header}: {int(has_suffix.sum())} values had unit suffixes (K/M/B) and were converted to $M")
    plain = ~has_suffix & num.notna()
    if plain.any():
        h = header.lower()
        med = float(num[plain].median())
        if re.search(r"(^|_)(bn|b|billions?)($|_)", h):
            num = num.where(~plain, num * 1e3)
            notes.append(f"{header}: header says billions; multiplied by 1,000 to get $M")
        elif re.search(r"(^|_)(k|thousands?)($|_)", h):
            num = num.where(~plain, num / 1e3)
            notes.append(f"{header}: header says thousands; divided by 1,000 to get $M")
        elif med >= DOLLARS_MEDIAN_MIN:
            num = num.where(~plain, num / 1e6)
            notes.append(f"{header}: median {med:,.0f} looks like raw dollars; divided by 1,000,000 to get $M")
        elif med < BILLIONS_MEDIAN_MAX and not re.search(r"(^|_)(mm|m|millions?)($|_)", h):
            num = num.where(~plain, num * 1e3)
            notes.append(
                f"{header}: median {med:g} with no unit in the header; assumed billions (x1,000). "
                "Pass --schema-map or rename the column if wrong"
            )
    return num.astype(float), failed, notes


def normalise_pct(vals: pd.Series, header: str) -> tuple[pd.Series, list[str]]:
    """0.21 -> 21. Decided per column: if every value is within [-1.5, 1.5] it is a fraction."""
    v = vals.dropna()
    if not v.empty and v.abs().max() <= 1.5:
        return vals * 100, [f"{header}: values look like fractions (max {v.abs().max():g}); multiplied by 100"]
    return vals, []


def parse_year_quarter(s: pd.Series, header: str) -> tuple[pd.Series, pd.Series | None, int, list[str]]:
    """A year column ('2021') or any common date format -> (year, quarter or None, failures, notes)."""
    raw = s.astype("string").str.strip()
    blank = raw.isna() | (raw == "")
    as_num = pd.to_numeric(raw, errors="coerce")
    if as_num.notna().sum() >= 0.9 * (~blank).sum() and as_num.dropna().between(1900, 2100).all():
        return as_num.astype(float), None, int((as_num.isna() & ~blank).sum()), []
    dates = pd.to_datetime(raw, errors="coerce", format="mixed")
    failed = int((dates.isna() & ~blank).sum())
    quarter = ("Q" + dates.dt.quarter.astype("Int64").astype("string")).where(dates.notna())
    return (
        dates.dt.year.astype(float),
        quarter,
        failed,
        [f"{header}: parsed as dates; derived deal_year and deal_quarter ({failed} unparseable)"],
    )
