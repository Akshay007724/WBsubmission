"""Stage 0: exploratory data analysis. Runs before scoring on every invocation, on any valid M&A CSV.

Each check measures something about the loaded file and, when it trips, records the finding with its
severity and how the engine responds. A check that needs a missing column reports "not checked" and why,
instead of failing. Checks re-run on any CSV, so a cleaner file shows 'not observed'.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from . import charts
from . import config as C
from .profile import TargetProfile

if TYPE_CHECKING:
    from .data import Dataset, Vocab

NUMERIC_SUMMARY = [
    "deal_size_mm",
    "ev_ebitda_multiple",
    "ev_revenue_multiple",
    "ebitda_margin_pct",
    "revenue_growth_pct",
    "synergy_pct_of_deal",
    "num_bidders",
    "days_to_close",
]
CATEGORICALS = [
    "sector",
    "deal_type",
    "outcome",
    "acquirer_type",
    "geography",
    "target_ownership_pre",
    "financing_type",
]
EV_RECONCILE_TOL = 0.10  # |EV - multiple x EBITDA| / EV
UBIQUITOUS_TAG_SHARE = 0.75
NOISY_LABEL_CRAMERS_V = 0.2  # below this, deal type is ~independent of buyer type
CONCENTRATION_RATIO = 3.0  # median sponsor deal count / median strategic deal count
OUTLIER_IQR_K = 3.0  # Tukey fence on log(size) and on multiples: beyond Q1-3*IQR / Q3+3*IQR
SKIP = "not checked"


def _r(x: Any, nd: int = 1) -> float | None:
    return None if x is None or pd.isna(x) else round(float(x), nd)


def _summary(s: pd.Series) -> dict:
    s = s.dropna()
    if s.empty:
        return {"n": 0}
    return {
        "n": len(s),
        "min": _r(s.min()),
        "p25": _r(s.quantile(0.25)),
        "median": _r(s.median()),
        "p75": _r(s.quantile(0.75)),
        "max": _r(s.max()),
        "mean": _r(s.mean()),
    }


def _has(df: pd.DataFrame, col: str) -> bool:
    if col not in df:
        return False
    s = df[col]
    return bool(s.notna().any() if s.dtype.kind in "fiu" else (~s.astype(str).str.strip().isin(["", "unknown"])).any())


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Association between two categoricals (0 = independent, 1 = perfectly linked)."""
    table = pd.crosstab(a, b).to_numpy(dtype=float)
    n = table.sum()
    if n == 0 or min(table.shape) < 2:
        return 0.0
    expected = table.sum(1, keepdims=True) @ table.sum(0, keepdims=True) / n
    chi2 = float(((table - expected) ** 2 / np.where(expected == 0, 1, expected)).sum())
    return round(float(np.sqrt(chi2 / (n * (min(table.shape) - 1)))), 3)


def _outliers(s: pd.Series, log: bool = False) -> int:
    v = s.dropna()
    v = np.log(v[v > 0]) if log else v
    if len(v) < 4:
        return 0
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    iqr = q3 - q1
    return int(((v < q1 - OUTLIER_IQR_K * iqr) | (v > q3 + OUTLIER_IQR_K * iqr)).sum())


def overview(ds: Dataset) -> dict:
    """Size, schema mapping, missingness, coercion failures, duplicates, distributions and mixes."""
    df = ds.df
    raw_cols = [c for c in df.columns if c not in ("acquirer_key", "deal_time", "tags")]
    missing = {}
    for c in raw_cols:
        blank = df[c].isna() | df[c].astype(str).str.strip().isin(["", "unknown", "nan"])
        if blank.any():
            missing[c] = round(100 * float(blank.mean()), 1)
    return {
        "rows": len(df),
        "rows_read": ds.load_stats.get("rows_read", len(df)),
        "duplicates_dropped": ds.load_stats.get("duplicates_dropped", 0),
        "coercion_failures": {
            k.removeprefix("coercion_failures_"): v
            for k, v in ds.load_stats.items()
            if k.startswith("coercion_failures_")
        },
        "columns_mapped": len(ds.schema),
        "unmapped_columns": list(ds.unmapped),
        "schema": [
            {"canonical": k, "source": m.source, "method": m.method, "confidence": m.confidence}
            for k, m in ds.schema.items()
        ],
        "features": dict(ds.features),
        "years": [int(df["deal_year"].min()), int(df["deal_year"].max())],
        "deals_per_year": {
            int(str(k)): int(v) for k, v in df["deal_year"].dropna().astype(int).value_counts().sort_index().items()
        },
        "unique_acquirers": int(df["acquirer"].nunique()),
        "missing_pct_by_column": missing,
        "numeric": {c: _summary(df[c]) for c in NUMERIC_SUMMARY if _has(df, c)},
        "categorical": {
            c: dict(Counter(v for v in df[c] if v and v != "unknown").most_common())
            for c in CATEGORICALS
            if _has(df, c)
        },
        "outliers": {
            "deal_size_log_iqr": _outliers(df["deal_size_mm"], log=True),
            "ev_ebitda_iqr": _outliers(df["ev_ebitda_multiple"]),
            "ev_revenue_iqr": _outliers(df["ev_revenue_multiple"]),
        },
        "correlations": _correlations(df),
    }


def _correlations(df: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    cols = [
        c
        for c in ("ebitda_margin_pct", "revenue_growth_pct", "ev_ebitda_multiple", "ev_revenue_multiple")
        if _has(df, c)
    ]
    if len(cols) < 2:
        return {}
    corr = df[cols].corr()
    return {a: {b: _r(corr.loc[a, b], 2) for b in cols if b != a} for a in cols}


def target_coverage(df: pd.DataFrame, profile: TargetProfile, vocab: Vocab | None = None) -> dict:
    """How many deals sit near the target's size, in its sector and in adjacent sectors."""
    from .scoring import sector_adjacency  # local import: scoring imports profile, not eda

    lo, hi = profile.ev_mm / C.COMP_SIZE_BAND, profile.ev_mm * C.COMP_SIZE_BAND
    band = df[df["deal_size_mm"].between(lo, hi)]
    adj = band[band["sector"].map(lambda s: sector_adjacency(profile.sector, s, vocab)) >= C.ADJACENT_SECTOR_MIN]
    closed = adj["outcome"].isin(["Closed", "unknown"])
    return {
        "size_band_mm": [_r(lo), _r(hi)],
        "deals_in_band": len(band),
        "target_sector_total": int((df["sector"] == profile.sector).sum()),
        "target_sector_in_band": int((band["sector"] == profile.sector).sum()),
        "adjacent_sectors_in_band": len(adj),
        "adjacent_closed_in_band": int(closed.sum()),
        "in_band_by_sector": dict(Counter(band["sector"]).most_common()),
    }


def buyer_structure(df: pd.DataFrame) -> dict:
    """Deals per acquirer by buyer type: concentration drives the shrinkage and mix-floor choices."""
    per = df.groupby(["acquirer_type", "acquirer"]).size()
    out = {}
    for typ, s in per.groupby(level=0):
        out[typ] = {
            "acquirers": len(s),
            "deals": int(s.sum()),
            "median_deals_per_acquirer": _r(s.median()),
            "max_deals": int(s.max()),
            "top": {str(k[1]) if isinstance(k, tuple) else str(k): int(v) for k, v in s.nlargest(5).items()},
        }
    return out


def _flag(fid: str, observed: bool | str, severity: str, finding: str, response: str, **metrics: Any) -> dict:
    return {
        "id": fid,
        "observed": observed if isinstance(observed, str) else bool(observed),
        "severity": severity,
        "finding": finding,
        "engine_response": response,
        "metrics": metrics,
    }


IMPLIED_MULTIPLE_RATIO = 3.0  # deal size / target EBITDA more than 3x the stated EV/EBITDA
MARGIN_GAP_PP = 2.0  # stated margin vs EBITDA / revenue, in percentage points


def financial_consistency_flags(df: pd.DataFrame) -> list[dict]:
    """Row-level contradictions between deal size, EBITDA, revenue, margin and the stated multiple.

    The engine never 'fixes' these (it cannot know which field is wrong); it reports the rows and uses
    only the stated multiples and margin, not the revenue / EBITDA fields.
    """
    if not (_has(df, "target_ebitda_mm") and _has(df, "target_revenue_mm")):
        return [
            _flag(
                "financials_consistency",
                SKIP,
                "medium",
                "Deal size vs EBITDA/revenue consistency",
                "Needs target revenue and EBITDA columns",
            )
        ]
    out = []
    ebitda = df["target_ebitda_mm"].where(df["target_ebitda_mm"] > 0)
    implied = df["deal_size_mm"] / ebitda
    if _has(df, "ev_ebitda_multiple"):
        bad = df[(implied / df["ev_ebitda_multiple"]) > IMPLIED_MULTIPLE_RATIO]
        ids = bad.assign(x=implied[bad.index]).sort_values("x", ascending=False)
        out.append(
            _flag(
                "ebitda_too_small_for_deal_size",
                len(bad) > 0,
                "high",
                f"{len(bad)} deals imply EV/EBITDA more than {IMPLIED_MULTIPLE_RATIO:g}x their stated multiple "
                f"(deal size / target EBITDA up to {implied.max():.0f}x), e.g. "
                + ", ".join(
                    f"{r.transaction_id} ${r.deal_size_mm:,.1f}M on ${r.target_ebitda_mm:,.1f}M EBITDA "
                    f"(stated {r.ev_ebitda_multiple:g}x)"
                    for r in ids.head(3).itertuples()
                ),
                "Stated multiples are used for comps; target EBITDA is never used to compute value or rank",
                rows=len(bad),
                transaction_ids=ids["transaction_id"].tolist(),
            )
        )
    if _has(df, "ebitda_margin_pct"):
        gap = (
            100 * ebitda / df["target_revenue_mm"].where(df["target_revenue_mm"] > 0) - df["ebitda_margin_pct"]
        ).abs()
        off = df[gap > MARGIN_GAP_PP]
        worst = off.assign(g=gap[off.index]).sort_values("g", ascending=False).head(3)
        out.append(
            _flag(
                "margin_disagrees_with_financials",
                len(off) > 0,
                "medium",
                f"Stated EBITDA margin differs from EBITDA / revenue by more than {MARGIN_GAP_PP:g}pp on {len(off)} of "
                f"{len(df)} rows (median gap {gap.median():.1f}pp, max {gap.max():.1f}pp), e.g. "
                + ", ".join(f"{r.transaction_id} ({r.g:.1f}pp)" for r in worst.itertuples()),
                "The stated margin column is used for margin similarity and the anchor; revenue/EBITDA are not used",
                rows=len(off),
                median_gap_pp=_r(gap.median()),
                max_gap_pp=_r(gap.max()),
            )
        )
    return out


def quality_flags(
    df: pd.DataFrame, profile: TargetProfile, ref_year: float, vocab: Vocab | None = None, features: dict | None = None
) -> list[dict]:
    """Every data-quality check, each with what was observed and how the engine responds."""
    features = features or {}
    flags = []

    if _has(df, "sub_sector"):
        same = float((df["sub_sector"] == df["sector"]).mean())
        flags.append(
            _flag(
                "sub_sector_redundant",
                same > 0.95,
                "medium",
                f"sub_sector equals sector on {same:.0%} of rows",
                "Similarity modelled with a sector adjacency matrix, not sub-sectors",
                share=_r(same, 3),
            )
        )

    if _has(df, "target_ebitda_mm") and _has(df, "ev_ebitda_multiple") and features.get("valuation") != "computed":
        implied = df["ev_ebitda_multiple"] * df["target_ebitda_mm"]
        off = (implied - df["deal_size_mm"]).abs() / df["deal_size_mm"] > EV_RECONCILE_TOL
        flags.append(
            _flag(
                "ev_does_not_reconcile",
                off.mean() > 0.2,
                "high",
                f"EV differs from EV/EBITDA x EBITDA by >{EV_RECONCILE_TOL:.0%} on {int(off.sum())} of {len(df)} rows",
                "Stated multiples used as given and never recomputed; synthetic-data caveat in README",
                rows=int(off.sum()),
                share=_r(off.mean(), 3),
            )
        )
    else:
        flags.append(
            _flag(
                "ev_does_not_reconcile",
                SKIP,
                "high",
                "Consistency of EV vs multiple x EBITDA",
                "Needs stated multiples and target EBITDA in the file",
            )
        )

    flags += financial_consistency_flags(df)

    cov = target_coverage(df, profile, vocab)
    flags.append(
        _flag(
            "thin_target_sector",
            cov["target_sector_in_band"] < C.THIN_SECTOR_BAND_DEALS,
            "high",
            f"Only {cov['target_sector_in_band']} {profile.sector} deals in the "
            f"${cov['size_band_mm'][0]}-{cov['size_band_mm'][1]}M band; "
            f"{cov['adjacent_sectors_in_band']} including adjacent sectors",
            "Adjacent sectors weighted in via the adjacency matrix; comp ladder widens when thin",
            same_sector=cov["target_sector_in_band"],
            with_adjacent=cov["adjacent_sectors_in_band"],
        )
    )

    bs = buyer_structure(df)
    sp, st = bs.get("Financial Sponsor", {}), bs.get("Strategic", {})
    if sp and st:
        ratio = (sp.get("median_deals_per_acquirer") or 0) / max(st.get("median_deals_per_acquirer") or 1, 1)
        flags.append(
            _flag(
                "buyer_concentration",
                ratio >= CONCENTRATION_RATIO,
                "high",
                f"Median sponsor has {sp.get('median_deals_per_acquirer')} deals vs "
                f"{st.get('median_deals_per_acquirer')} for the median strategic",
                "Empirical-Bayes shrinkage on fit and a strategic/sponsor mix floor; "
                "displaced buyers listed under 'Also considered'",
                ratio=_r(ratio),
            )
        )
    else:
        flags.append(
            _flag(
                "buyer_concentration",
                SKIP,
                "high",
                "Sponsor vs strategic concentration",
                "Needs both buyer types in acquirer_type; mix floor is off without them",
            )
        )

    s = df["deal_size_mm"]
    skew = float(s.mean() / s.median())
    flags.append(
        _flag(
            "size_skew",
            skew > 1.5,
            "medium",
            f"Deal size mean ${s.mean():,.0f}M is {skew:.1f}x the median ${s.median():,.0f}M",
            "Log-space size kernel; medians not means",
            mean_over_median=_r(skew),
        )
    )
    n_out = _outliers(s, log=True) + _outliers(df["ev_ebitda_multiple"])
    flags.append(
        _flag(
            "outliers",
            n_out > 0,
            "low",
            f"{n_out} extreme values (beyond {OUTLIER_IQR_K:g}x IQR) in log deal size or EV/EBITDA",
            "Kept (they are real deals); the log size kernel and medians limit their influence",
            rows=n_out,
        )
    )

    if (df["outcome"] == "unknown").all():
        flags.append(_flag("stale_pending", SKIP, "medium", "Stale pending deals", "Needs an outcome column"))
        flags.append(
            _flag(
                "non_closed_outcomes",
                SKIP,
                "medium",
                "Non-closed deals",
                "No outcome column: every deal is treated as completed (stated in the report)",
            )
        )
    else:
        stale = df["outcome"].isin(["Pending", "Rumored"]) & (ref_year - df["deal_year"] > C.STALE_PENDING_YEARS)
        flags.append(
            _flag(
                "stale_pending",
                bool(stale.any()),
                "medium",
                f"{int(stale.sum())} Pending/Rumored deals are more than {C.STALE_PENDING_YEARS} years old",
                "Down-weighted to Withdrawn level; excluded from comps; listed as risk facts",
                rows=int(stale.sum()),
            )
        )
        non_closed = int((df["outcome"] != "Closed").sum())
        flags.append(
            _flag(
                "non_closed_outcomes",
                non_closed > 0,
                "medium",
                f"{non_closed} deals did not close",
                "Comps use Closed control deals only",
                rows=non_closed,
            )
        )

    if _has(df, "deal_type") and sp and st:
        v = cramers_v(df["deal_type"], df["acquirer_type"])
        flags.append(
            _flag(
                "noisy_deal_type",
                v < NOISY_LABEL_CRAMERS_V,
                "medium",
                f"deal_type is nearly independent of acquirer_type (Cramér's V = {v}); real-world "
                "LBOs/recaps are sponsor-dominated",
                "Deal-type weight softened (exponent 0.5); prompt forbids theses built on one label",
                cramers_v=v,
            )
        )

    shares = {t: c / len(df) for t, c in Counter(t for ts in df["tags"] for t in ts).items()}
    if shares:
        ubiq = {t: _r(sh, 2) for t, sh in shares.items() if sh >= UBIQUITOUS_TAG_SHARE}
        flags.append(
            _flag(
                "ubiquitous_tags",
                bool(ubiq),
                "low",
                f"Tags on >= {UBIQUITOUS_TAG_SHARE:.0%} of deals carry no signal: {ubiq}",
                "Tags reported by lift vs. market base rate",
                tags=ubiq,
            )
        )

    if _has(df, "synergy_pct_of_deal") and sp and st:
        syn = df.groupby("acquirer_type")["synergy_pct_of_deal"].median()
        backwards = bool(syn.get("Financial Sponsor", 0) > syn.get("Strategic", 0))
        flags.append(
            _flag(
                "synergy_backwards",
                backwards,
                "medium",
                f"Across all deals, median synergy % is higher for sponsors ({_r(syn.get('Financial Sponsor'))}) "
                f"than strategics ({_r(syn.get('Strategic'))}), which is economically backwards",
                "synergy_pct_of_deal not used for ability-to-pay",
                **{str(k): _r(x) for k, x in syn.items()},
            )
        )

    closed = df[df["outcome"].isin(["Closed", "unknown"]) & ~df["deal_type"].isin(C.COMP_EXCLUDE_DEAL_TYPES)]
    by_type = closed.groupby("acquirer_type")["ev_ebitda_multiple"].median().dropna()
    if len(by_type) == 2:
        gap = float(by_type.max() - by_type.min())
        flags.append(
            _flag(
                "multiples_differ_by_buyer",
                gap >= 0.5,
                "medium",
                "All closed control deals (every sector and size): median EV/EBITDA "
                + ", ".join(f"{k} {x:.1f}x" for k, x in by_type.items()),
                "Each acquirer benchmarked against its own buyer-type peer set",
                gap_turns=_r(gap),
                **{str(k): _r(x) for k, x in by_type.items()},
            )
        )

    if _has(df, "deal_type"):
        minority = df["deal_type"].isin(C.COMP_EXCLUDE_DEAL_TYPES)
        flags.append(
            _flag(
                "non_control_deal_types",
                bool(minority.any()),
                "medium",
                f"{int(minority.sum())} deals are {sorted(C.COMP_EXCLUDE_DEAL_TYPES)} (not control acquisitions)",
                "Excluded from valuation comps; Minority Investment heavily down-weighted in ranking",
                rows=int(minority.sum()),
            )
        )

    corr = _correlations(df).get("ebitda_margin_pct", {}).get("ev_ebitda_multiple")
    if corr is not None:
        flags.append(
            _flag(
                "margin_drives_multiple",
                corr >= 0.3,
                "medium",
                f"Correlation of EBITDA margin with EV/EBITDA is {corr}",
                "Pricing compared like-for-like (any margin); the high-margin set is only the target value anchor",
                correlation=corr,
            )
        )

    for feature, why in features.items():
        if why.startswith("off"):
            flags.append(
                _flag(
                    f"feature_off_{feature}",
                    True,
                    "medium",
                    f"Feature '{feature}' switched off: {why.removeprefix('off: ')}",
                    "Dependent sections are omitted and say so",
                )
            )
        elif why in ("computed", "inferred"):
            flags.append(
                _flag(
                    f"feature_{why}_{feature}",
                    True,
                    "low",
                    f"Feature '{feature}' is {why}",
                    f"Values are labelled {why} wherever they appear",
                )
            )
    return flags


def adjacency_table(vocab: Vocab | None) -> dict:
    """The sector adjacency in use (preset or derived), as a nested dict for the EDA appendix."""
    if vocab is None:
        return {}
    secs = list(vocab.sectors)[:14]
    return {
        "source": "derived from co-investment" if vocab.adjacency_derived else f"preset ({vocab.sector_preset})",
        "matrix": {a: {b: vocab.adj(a, b) for b in secs} for a in secs},
    }


def run(ds: Dataset, profile: TargetProfile) -> dict:
    """Stage 0 for one dataset and target."""
    flags = quality_flags(ds.df, profile, ds.ref_year, ds.vocab, ds.features)
    return {
        "overview": overview(ds),
        "target_coverage": target_coverage(ds.df, profile, ds.vocab),
        "buyer_structure": buyer_structure(ds.df),
        "adjacency": adjacency_table(ds.vocab),
        "flags": flags,
        "observed_flags": [f["id"] for f in flags if f["observed"] is True],
        "load_notes": list(ds.notes),
        "charts": charts.eda_charts(ds.df),
    }
