"""Deterministic, explainable acquirer ranking.

Per-deal relevance  w = sector_adjacency * size_kernel * recency * outcome * profile_fit
Acquirer score      = 0.45*norm(sum w) + 0.30*norm(shrunk mean w) + 0.25*max w - 0.15*exec_penalty
then a buyer-mix floor (>= MIN_PER_TYPE strategics and sponsors) and a deterministic tie-break.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import config as C
from .data import Dataset, Vocab
from .profile import TargetProfile


def sector_adjacency(a: str, b: str, vocab: Vocab | None = None) -> float:
    """Similarity of two sectors: the dataset's vocabulary (preset or derived) if given, else the preset."""
    if vocab is not None:
        return vocab.adj(a, b)
    if a == b:
        return 1.0
    return C.SECTOR_ADJ_PAIRS.get((a, b), C.SECTOR_ADJ_DEFAULT)


def size_kernel(size_mm: pd.Series | float, target_mm: float) -> np.ndarray:
    """Log-normal size similarity in (0, 1]: 1 at the target's size, ~0.61 at 2.5x or 1/2.5x."""
    return np.exp(-(np.log(np.asarray(size_mm, dtype=float) / target_mm) ** 2) / (2 * C.SIZE_SIGMA**2))


def target_margin(profile: TargetProfile, df: pd.DataFrame) -> float | None:
    """The target's EBITDA margin: given, else the dataset's 75th percentile; None if the file has no margins."""
    if profile.ebitda_margin_pct is not None:
        return profile.ebitda_margin_pct
    q = df["ebitda_margin_pct"].quantile(C.STRONG_MARGIN_PCTL)
    return None if pd.isna(q) else round(float(q), 1)


def _geo_fit(geo: str, target_geo: str, broad: frozenset[str] = frozenset(C.BROAD_GEOS)) -> float:
    if not geo or target_geo == "Regional":
        return C.GEO_BROAD if geo in broad else C.GEO_SAME
    if geo == target_geo:
        return C.GEO_SAME
    return C.GEO_BROAD if geo in broad or target_geo in broad else C.GEO_OTHER


def closed_mask(df: pd.DataFrame) -> pd.Series:
    """Completed deals. If the file has no outcome column at all, every deal counts (stated in the notes)."""
    if (df["outcome"] == "unknown").all():
        return pd.Series(True, index=df.index)
    return df["outcome"] == "Closed"


def _outcome_w(row: pd.Series, ref_year: float) -> float:
    if row["outcome"] == "unknown":
        return 1.0
    w = C.OUTCOME_W.get(row["outcome"], C.UNKNOWN_OUTCOME_W)
    if row["outcome"] in ("Pending", "Rumored") and ref_year - row["deal_year"] > C.STALE_PENDING_YEARS:
        w = min(w, C.STALE_PENDING_W)
    return w


def score_deals(ds: Dataset, profile: TargetProfile) -> pd.DataFrame:
    """Return a copy of the data with each relevance factor and the combined weight `w`."""
    df = ds.df.copy()
    margin_t = target_margin(profile, df)
    vocab = ds.vocab
    df["f_sector"] = df["sector"].map(lambda s: sector_adjacency(profile.sector, s, vocab))
    df["f_size"] = size_kernel(df["deal_size_mm"], profile.ev_mm)
    df["f_recency"] = 0.5 ** ((ds.ref_year - df["deal_time"]).clip(lower=0) / C.RECENCY_HALF_LIFE_Y)
    df["f_outcome"] = df.apply(_outcome_w, axis=1, ref_year=ds.ref_year)
    df["is_closed"] = closed_mask(df)  # decided once, dataset-wide (never per acquirer group)
    own = df["target_ownership_pre"].map(lambda o: C.OWNERSHIP_W.get(o, C.UNKNOWN_OWNERSHIP_W) if o else 1.0)
    if margin_t is None:
        margin = pd.Series(1.0, index=df.index)
    else:
        margin = np.exp(-((df["ebitda_margin_pct"].fillna(margin_t) - margin_t) ** 2) / (2 * C.MARGIN_SIGMA_PP**2))
    broad = vocab.broad_geographies if vocab else frozenset(C.BROAD_GEOS)
    geo = df["geography"].map(lambda g: _geo_fit(g, profile.geography, broad))
    dtype = (
        df["deal_type"].map(lambda t: C.DEAL_TYPE_W.get(t, C.UNKNOWN_DEAL_TYPE_W) if t else 1.0) ** C.DEAL_TYPE_EXPONENT
    )
    df["f_profile"] = (own * margin * geo) ** (1 / 3) * dtype
    df["w"] = df["f_sector"] * df["f_size"] * df["f_recency"] * df["f_outcome"] * df["f_profile"]
    return df


@dataclass(frozen=True)
class Candidate:
    """One ranked acquirer with every component of its score (shown as 'Why ranked #k')."""

    acquirer: str
    acquirer_type: str
    score: float
    relevance_mass: float
    fit_quality: float
    best_match: float
    exec_penalty: float
    n_deals: int
    n_relevant: int
    last_year: int
    mix_fill: bool = False  # True if included to satisfy the buyer-mix floor

    def as_dict(self) -> dict:
        """Rounded plain dict for the report and the prompt."""
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def aggregate(scored: pd.DataFrame, excluded: frozenset[str] = frozenset()) -> list[Candidate]:
    """One Candidate per eligible acquirer, scored and sorted (deterministic tie-break).

    Eligible = at least one deal with relevance >= MIN_DEAL_RELEVANCE and at least one closed deal.
    score = W_MASS*mass/max(mass) + W_FIT*fit/max(fit) + W_BEST*best - W_EXEC_PENALTY*failed share.
    """
    mu = float(scored["w"].mean())
    found: list[Candidate] = []  # score filled in once the normalisers are known
    for key, g in scored.groupby("acquirer_key"):
        name = str(g["acquirer"].iat[0])
        if str(key) in excluded or name.casefold() in excluded:
            continue
        relevant = g[g["w"] >= C.MIN_DEAL_RELEVANCE]
        if relevant.empty or not g["is_closed"].any():
            continue  # eligibility: one meaningful deal and at least one closed deal ever
        mass = float(g["w"].sum())
        found.append(
            Candidate(
                acquirer=name,
                acquirer_type=str(g["acquirer_type"].iat[0]),
                score=0.0,
                relevance_mass=mass,
                fit_quality=(mass + C.SHRINKAGE_K * mu) / (len(g) + C.SHRINKAGE_K),
                best_match=float(g["w"].max()),
                exec_penalty=float(relevant["outcome"].isin(["Withdrawn", "Terminated"]).mean()),
                n_deals=len(g),
                n_relevant=len(relevant),
                last_year=int(g["deal_year"].max()),
            )
        )
    if not found:
        return []
    max_mass = max(c.relevance_mass for c in found)
    max_fit = max(c.fit_quality for c in found)
    out = [
        replace(
            c,
            score=C.W_MASS * c.relevance_mass / max_mass
            + C.W_FIT * c.fit_quality / max_fit
            + C.W_BEST * c.best_match
            - C.W_EXEC_PENALTY * c.exec_penalty,
        )
        for c in found
    ]
    out.sort(key=_order)  # deterministic tie-break: score, mass, recency, name
    return out


def floor_qualifies(c: Candidate, scored: pd.DataFrame | None, profile: TargetProfile | None) -> bool:
    """A buyer may be pulled in by the mix floor only with direct evidence: at least one closed deal in the
    target's size band (±COMP_SIZE_BAND) or in the target's own sector. No data to check -> qualifies."""
    if scored is None or profile is None:
        return True
    g = scored[(scored["acquirer"] == c.acquirer) & scored["is_closed"]]
    lo, hi = profile.ev_mm / C.COMP_SIZE_BAND, profile.ev_mm * C.COMP_SIZE_BAND
    return bool((g["deal_size_mm"].between(lo, hi) | (g["sector"] == profile.sector)).any())


def _order(c: Candidate) -> tuple:
    return (-round(c.score, 9), -c.relevance_mass, -c.last_year, c.acquirer)


def select(
    cands: list[Candidate],
    top_n: int,
    min_per_type: int | None = None,
    scored: pd.DataFrame | None = None,
    profile: TargetProfile | None = None,
) -> tuple[list[Candidate], list[str], list[dict]]:
    """Pick top_n by score, then enforce a strategic/sponsor floor of ``min_per_type`` each.

    A floor pick replaces the lowest-ranked pick of the other type, and must pass ``floor_qualifies``;
    if no candidate qualifies the slot stays with the higher-scoring buyer and a warning says so.
    Returns (picks, warnings, displaced) where displaced pairs each removed buyer with its replacement.
    """
    min_per_type = C.MIN_PER_TYPE if min_per_type is None else min_per_type
    warnings: list[str] = []
    displaced: list[dict] = []
    if len(cands) < top_n:
        warnings.append(f"Only {len(cands)} eligible acquirers (requested {top_n}).")
    picks = list(cands[:top_n])
    floor = min(min_per_type, top_n // 2)
    for typ in ("Strategic", "Financial Sponsor"):
        have = sum(c.acquirer_type == typ for c in picks)
        pool = [c for c in cands if c.acquirer_type == typ and c not in picks]
        skipped = [c.acquirer for c in pool if not floor_qualifies(c, scored, profile)]
        pool = [c for c in pool if floor_qualifies(c, scored, profile)]
        while have < floor and pool:
            others = [c for c in picks if c.acquirer_type != typ and not c.mix_fill]
            if not others:
                break
            out = others[-1]
            picks.remove(out)
            add = pool.pop(0)
            picks.append(replace(add, mix_fill=True))
            displaced.append(
                {
                    "displaced": out.acquirer,
                    "displaced_type": out.acquirer_type,
                    "displaced_score": round(out.score, 3),
                    "replaced_by": add.acquirer,
                    "replacement_score": round(add.score, 3),
                }
            )
            have += 1
        if have < floor:
            why = f"; {len(skipped)} lacked a closed deal in the size band or target sector" if skipped else ""
            warnings.append(
                f"Only {have} qualifying '{typ}' acquirers; mix floor of {floor} not met{why}. "
                "Remaining slots stay with the higher-scoring buyers."
            )
    picks.sort(key=_order)
    return picks, warnings, displaced


def rank(
    ds: Dataset, profile: TargetProfile, top_n: int = 10, excluded: frozenset[str] = frozenset()
) -> tuple[pd.DataFrame, list[Candidate], list[str]]:
    """Convenience: score deals, aggregate and select (default floor). Used by tests and tooling."""
    scored = score_deals(ds, profile)
    picks, warnings, _ = select(aggregate(scored, excluded), top_n, None, scored, profile)
    return scored, picks, warnings
