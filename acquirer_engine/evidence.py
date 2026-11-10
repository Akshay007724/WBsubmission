"""Deterministic fact packs. The LLM may only cite numbers that appear here."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import pandas as pd

from . import config as C
from .profile import TargetProfile
from .scoring import Candidate, target_margin


def _r(x: Any, nd: int = 1) -> Any:
    return None if pd.isna(x) else round(float(x), nd)


def multiple_stats(deals: pd.DataFrame) -> dict:
    """n, median and (n >= 3) IQR of EV/EBITDA and EV/Revenue, plus the transaction ids used.
    Empty input -> {'n': 0}.
    """
    d = deals.dropna(subset=["ev_ebitda_multiple", "ev_revenue_multiple"])
    if d.empty:
        return {"n": 0}
    e, r = d["ev_ebitda_multiple"], d["ev_revenue_multiple"]
    out = {"n": len(d), "ev_ebitda_median": _r(e.median()), "ev_revenue_median": _r(r.median(), 2)}
    if len(d) >= 3:  # an IQR of 1-2 deals is not a range
        out.update(
            ev_ebitda_p25=_r(e.quantile(0.25)),
            ev_ebitda_p75=_r(e.quantile(0.75)),
            ev_revenue_p25=_r(r.quantile(0.25), 2),
            ev_revenue_p75=_r(r.quantile(0.75), 2),
        )
    out["transaction_ids"] = d["transaction_id"].tolist()
    return out


def control_comps(scored: pd.DataFrame) -> pd.DataFrame:
    """Closed, control transactions only: valuation comps must carry a control premium."""
    closed = scored["is_closed"] if "is_closed" in scored else scored["outcome"] == "Closed"
    return scored[closed & ~scored["deal_type"].isin(C.COMP_EXCLUDE_DEAL_TYPES)]


def _deal_row(r: pd.Series) -> dict:
    return {
        "transaction_id": r["transaction_id"],
        "target": r["target_company"],
        "year": int(r["deal_year"]),
        "sector": r["sector"],
        "geography": r["geography"],
        "deal_size_mm": _r(r["deal_size_mm"]),
        "deal_type": r["deal_type"],
        "ev_ebitda": _r(r["ev_ebitda_multiple"]),
        "ev_revenue": _r(r["ev_revenue_multiple"], 2),
        "ebitda_margin_pct": _r(r["ebitda_margin_pct"]),
        "revenue_growth_pct": _r(r["revenue_growth_pct"]),
        "outcome": r["outcome"],
        "target_ownership": r["target_ownership_pre"],
        "financing": r["financing_type"],
        "num_bidders": None if pd.isna(r["num_bidders"]) else int(r["num_bidders"]),
        "tags": list(r["tags"]),
        "relevance": round(float(r["w"]), 2),
        "usable_as_comp": bool(
            r.get("is_closed", r["outcome"] == "Closed") and r["deal_type"] not in C.COMP_EXCLUDE_DEAL_TYPES
        ),
    }


def market_comps(scored: pd.DataFrame, profile: TargetProfile, ref_year: float | None = None) -> dict:
    """Closed control comps, widening the net until n >= MIN_COMPS. Every step is recorded."""
    closed = control_comps(scored)
    ref_year = ref_year or float(scored["deal_year"].max())
    margin_t = target_margin(profile, scored)
    med_margin = float(scored["ebitda_margin_pct"].median()) if margin_t is not None else float("nan")
    ladder = (
        [
            ("adjacent sectors, ±2.5x size, margin >= median", C.COMP_SIZE_BAND, True),
            ("adjacent sectors, ±4x size, margin >= median", C.COMP_WIDE_BAND, True),
            ("adjacent sectors, ±4x size, any margin", C.COMP_WIDE_BAND, False),
        ]
        if margin_t is not None
        else [
            ("adjacent sectors, ±2.5x size (no margin data)", C.COMP_SIZE_BAND, False),
            ("adjacent sectors, ±4x size (no margin data)", C.COMP_WIDE_BAND, False),
        ]
    )
    steps, chosen, peers = [], {}, closed.iloc[0:0]
    for label, band, use_margin in ladder:
        lo, hi = profile.ev_mm / band, profile.ev_mm * band
        m = (closed["f_sector"] >= C.ADJACENT_SECTOR_MIN) & closed["deal_size_mm"].between(lo, hi)
        if use_margin:
            m &= closed["ebitda_margin_pct"] >= med_margin
        peers = closed[m]
        stats = multiple_stats(peers)
        steps.append(f"{label}: n={stats['n']}")
        chosen = {"definition": label, "size_range_mm": [_r(lo), _r(hi)], **stats}
        if stats["n"] >= C.MIN_COMPS:
            break
    lo, hi = profile.ev_mm / C.COMP_SIZE_BAND, profile.ev_mm * C.COMP_SIZE_BAND
    in_band = closed[(closed["f_sector"] >= C.ADJACENT_SECTOR_MIN) & closed["deal_size_mm"].between(lo, hi)]
    same = closed[(closed["sector"] == profile.sector) & closed["deal_size_mm"].between(lo, hi)]
    implied_ebitda = _r(profile.ev_mm / chosen["ev_ebitda_median"]) if chosen.get("n") else None
    implied_rev = _r(implied_ebitda / (margin_t / 100)) if implied_ebitda and margin_t else None
    return {
        # Anchor: margin-filtered, because margin and EV/EBITDA correlate (~0.56 here). Answers
        # "what should a target with this margin command" - NOT "how does a buyer usually pay".
        "peer_set": {"role": TARGET_ANCHOR_ROLE.format(margin=margin_t if margin_t is not None else "n/a"), **chosen},
        "widening_steps": steps,
        "excluded_deal_types": sorted(C.COMP_EXCLUDE_DEAL_TYPES),
        # Like-for-like pricing benchmark: same filters as each acquirer's own comps (adjacent
        # sectors, ±2.5x size, ANY margin), split by buyer type because sponsors and strategics pay
        # differently. Used for every "above / in line with / below peers" statement.
        "pricing_peers_by_buyer_type": {
            t: {
                "definition": f"{t} closed control deals, adjacent sectors, ±2.5x size, any margin",
                **multiple_stats(g),
            }
            for t, g in in_band.groupby("acquirer_type")
        },
        "peer_recent_3y": multiple_stats(peers[peers["deal_year"] > ref_year - C.RECENT_YEARS]),
        "same_sector_band": {"definition": f"{profile.sector} only, ±2.5x size", **multiple_stats(same)},
        "target_margin_assumed_pct": margin_t,
        "dataset_median_margin_pct": _r(med_margin),
        "implied_target_ebitda_mm": implied_ebitda,
        "implied_target_revenue_mm": implied_rev,
    }


TARGET_ANCHOR_ROLE = (
    "target value anchor: what a {margin}%-margin target should command "
    "(high-margin deals only; never use it to say a buyer pays below/above peers)"
)


def pricing_vs_peers(acq: dict, peer: dict, acquirer_type: str) -> dict | None:
    """Like-for-like: buyer's own comps vs same-type peers under identical filters (any margin).

    ``position`` is a deterministic label the prompt must use verbatim, so "below peers" can only
    appear when the like-for-like gap exceeds ``PRICING_IN_LINE_TURNS``.
    """
    if not acq.get("n") or not peer.get("n"):
        return None
    gap = round(acq["ev_ebitda_median"] - peer["ev_ebitda_median"], 1)
    position = "in line with" if abs(gap) <= C.PRICING_IN_LINE_TURNS else "above" if gap > 0 else "below"
    return {
        "benchmark": f"{acquirer_type} peers, same filters, any margin: "
        f"median {peer['ev_ebitda_median']}x (n={peer['n']})",
        "acquirer_median": acq["ev_ebitda_median"],
        "peer_median": peer["ev_ebitda_median"],
        "ev_ebitda_gap_turns": gap,
        "position": position,
    }


def tag_lift(g: pd.DataFrame, base: dict[str, float]) -> dict[str, float]:
    """Tags this buyer uses more than the market does. 'Geographic Expansion' is on 90% of deals."""
    counts = Counter(t for ts in g["tags"] for t in ts)
    lifts = {t: round((n / len(g)) / base[t], 1) for t, n in counts.items() if base.get(t)}
    kept = sorted(((t, lift) for t, lift in lifts.items() if lift >= C.TAG_MIN_LIFT), key=lambda x: -x[1])
    return dict(kept[:4])


def size_position(g: pd.DataFrame, ev: float, acquirer_type: str = "Financial Sponsor") -> dict:
    """Where the target's EV sits in this buyer's own deal sizes, as a deterministic role.

    Sponsors: add-on / platform / upper end of range / stretch. Strategics: tuck-in / core-sized /
    stretch. If the buyer has >= PLATFORM_EVIDENCE_MIN deals labelled Platform Investment, the
    platform-vs-add-on call compares against those deals' sizes instead of generic percentiles.
    """
    sizes = g["deal_size_mm"].dropna()
    # Capacity (largest deal, 90th percentile) is judged on deals that actually CLOSED: a withdrawn
    # $6.9B carve-out is appetite, not proof the buyer can finance that size. Percentile and IQR still
    # use every deal, because they describe what the buyer goes after.
    closed = g["is_closed"] if "is_closed" in g else (g["outcome"] == "Closed" if "outcome" in g else None)
    done = g.loc[closed, "deal_size_mm"].dropna() if closed is not None else sizes
    capacity_basis = "closed deals"
    if done.empty:
        done, capacity_basis = sizes, "all deals (none closed)"
    add_on_q = C.ADD_ON_PCTL / 100
    q = sizes.quantile([0.25, add_on_q, 0.75, 0.9])
    median = float(sizes.median())
    pct = round(100 * float((sizes < ev).mean()))
    sponsor = acquirer_type == "Financial Sponsor"
    small, mid = {"Financial Sponsor": ("add-on", "platform"), "Strategic": ("tuck-in", "core-sized")}.get(
        acquirer_type, ("smaller than usual", "typical size")
    )  # buyer type unknown: neutral labels
    platform_sizes = g.loc[g["deal_type"] == "Platform Investment", "deal_size_mm"].dropna()
    facts = {
        "target_size_percentile_in_buyer_history": pct,
        "buyer_size_iqr_mm": [_r(q[0.25]), _r(q[0.75])],
        "largest_closed_deal_mm": _r(done.max()),
        "closed_deal_p90_mm": _r(done.quantile(0.9)),
        "capacity_basis": capacity_basis,
    }

    pmed = float(platform_sizes.median()) if len(platform_sizes) >= C.PLATFORM_EVIDENCE_MIN else None
    if pmed is not None:
        facts["platform_deals"] = {"n": len(platform_sizes), "median_mm": _r(pmed)}
    # Labelled platform deals decide only when the target is comparable to them (<= 2x their median);
    # a target far above them says nothing about platform vs add-on, so percentiles decide instead.
    platform_evidence = sponsor and pmed is not None and ev <= C.PLATFORM_MATCH_MULTIPLE * pmed

    if ev > done.max() or ev > C.STRETCH_P90_MULTIPLE * done.quantile(0.9):
        role, why = (
            "stretch",
            (f"above its largest closed deal (${_r(done.max())}M) or 2x its closed-deal 90th percentile"),
        )
    elif platform_evidence and pmed is not None:
        role = small if ev < C.ADD_ON_MEDIAN_FRACTION * pmed else mid
        why = f"vs its {len(platform_sizes)} Platform Investment deals (median ${_r(pmed)}M)"
    elif ev < q[add_on_q] or ev < C.ADD_ON_MEDIAN_FRACTION * median:
        role, why = small, f"below its {C.ADD_ON_PCTL}th percentile or half its ${_r(median)}M median"
    elif ev <= q[0.75]:
        role, why = mid, f"inside its interquartile range ${_r(q[0.25])}-{_r(q[0.75])}M"
    else:
        role = "upper end of range" if sponsor else mid
        why = f"above its 75th percentile (${_r(q[0.75])}M) but not a stretch"
    return {**facts, "size_role": role, "size_role_basis": why}


def _risk_facts(g: pd.DataFrame, cand: Candidate, profile: TargetProfile, ref_year: float) -> dict:
    failed = g[g["outcome"].isin(["Withdrawn", "Terminated"])]
    stale = g[g["outcome"].isin(["Pending", "Rumored"]) & (ref_year - g["deal_year"] > C.STALE_PENDING_YEARS)]
    same_sector = g[g["sector"] == profile.sector]
    recent = g[g["deal_year"] > ref_year - C.RECENT_YEARS]
    big_same = same_sector[same_sector["deal_size_mm"] >= C.ANTITRUST_LARGE_DEAL_MM]
    same_region = None if profile.geography == "Regional" else int((g["geography"] == profile.geography).sum())
    closed_rows = g["is_closed"] if "is_closed" in g else g["outcome"] == "Closed"
    closed_sizes = g.loc[closed_rows, "deal_size_mm"].dropna()
    return {
        "withdrawn_or_terminated": [
            {
                "transaction_id": r.transaction_id,
                "outcome": r.outcome,
                "year": int(r.deal_year),
                "deal_size_mm": _r(r.deal_size_mm),
            }
            for r in failed.itertuples()
        ],
        "stale_pending_or_rumored": stale["transaction_id"].tolist(),
        "deals_in_target_sector": len(same_sector),
        "large_deals_in_target_sector_mm": [_r(x) for x in big_same["deal_size_mm"]],
        "deals_in_target_region": same_region if same_region is not None else "target region not specified",
        "antitrust_signal": bool(
            cand.acquirer_type == "Strategic"
            and (len(big_same) > 0 or len(same_sector) >= C.ANTITRUST_MIN_SECTOR_DEALS or (same_region or 0) > 0)
        ),
        "deals_last_3y": len(recent),
        "largest_closed_deal_mm": _r(closed_sizes.max()) if not closed_sizes.empty else None,
        "largest_withdrawn_or_pending_mm": _r(g.loc[~closed_rows, "deal_size_mm"].max())
        if (~closed_rows).any()
        else None,
        "avg_bidders": _r(g["num_bidders"].mean()),
        "high_competition": bool(g["num_bidders"].mean() >= C.HIGH_BIDDERS),
        "leveraged_or_seller_financed_share_pct": _r(
            100 * g["financing_type"].isin(["Leveraged", "Seller Financing"]).mean()
        ),
    }


def suggested_conviction(pack: dict, profile: TargetProfile, market: dict) -> tuple[str, list[str], float]:
    """Weighted evidence points -> High/Medium/Low. Each signal is a readable '+n reason' string."""
    pts, signals = 0.0, []

    def add(n: float, why: str) -> None:
        nonlocal pts
        pts += n
        signals.append(f"{n:+g} {why}")

    pr, risk = pack["profile"], pack["risk"]
    band = pr["closed_in_size_band"]
    if band >= C.CONV_BAND_STRONG:
        add(2, f"{band} closed adjacent-sector deals in the size band")
    elif band >= C.CONV_BAND_SOME:
        add(1, f"{band} closed adjacent-sector deals in the size band")
    elif band == 0:
        add(-1, "no closed deals in the size band")
    recent = pr["recent_in_band_closed"]
    if recent >= 2:
        add(1, f"{recent} of those closed in the last {C.RECENT_YEARS} years")
    elif recent == 1:
        add(0.5, f"1 of those closed in the last {C.RECENT_YEARS} years")
    a = pack["acquirer_comps"]
    p = market["pricing_peers_by_buyer_type"].get(pack["acquirer_type"], {})
    if (
        a.get("n", 0) >= C.MIN_ACQ_COMPS_FOR_SIGNAL
        and p.get("ev_ebitda_p25") is not None
        and p["ev_ebitda_p25"] <= a["ev_ebitda_median"] <= p["ev_ebitda_p75"]
    ):
        add(
            1,
            f"median {a['ev_ebitda_median']}x sits inside {pack['acquirer_type']} peer IQR "
            f"{p['ev_ebitda_p25']}-{p['ev_ebitda_p75']}x",
        )
    sec = risk["deals_in_target_sector"]
    if sec >= 3:
        add(2, f"{sec} prior {profile.sector} deals")
    elif sec >= 1:
        add(1, f"{sec} prior {profile.sector} deal(s)")
    failed = len(risk["withdrawn_or_terminated"])
    if failed == 0:
        add(1, "no withdrawn/terminated deals")
    elif failed / pr["total_deals"] > C.CONV_FAIL_SHARE:
        add(-1, f"{failed} of {pr['total_deals']} deals withdrawn/terminated")
    if pack["size_position"]["size_role"] == "stretch":
        add(-1, "target would be its largest deal")
    level = "High" if pts >= C.CONV_HIGH else "Medium" if pts >= C.CONV_MEDIUM else "Low"
    return level, signals, pts


def build(scored: pd.DataFrame, cand: Candidate, profile: TargetProfile, market: dict, ref_year: float) -> dict:
    """The deterministic fact pack for one acquirer.

    Contains profile, size role, precedents, comps, like-for-like pricing, risks and the conviction prior.
    Never raises for thin data; missing figures are None.
    """
    g = scored[scored["acquirer"] == cand.acquirer].sort_values("w", ascending=False)
    closed = g[g["is_closed"]] if "is_closed" in g else g[g["outcome"] == "Closed"]
    ctrl = control_comps(g)
    lo, hi = profile.ev_mm / C.COMP_SIZE_BAND, profile.ev_mm * C.COMP_SIZE_BAND
    in_band = closed[(closed["f_sector"] >= C.ADJACENT_SECTOR_MIN) & closed["deal_size_mm"].between(lo, hi)]
    comps = ctrl[(ctrl["f_sector"] >= C.ADJACENT_SECTOR_MIN) & ctrl["deal_size_mm"].between(lo, hi)]
    base = {t: n / len(scored) for t, n in Counter(t for ts in scored["tags"] for t in ts).items()}
    acq = multiple_stats(comps)
    pack = {
        "acquirer": cand.acquirer,
        "acquirer_type": cand.acquirer_type,
        "profile": {
            "total_deals": len(g),
            "closed_deals": len(closed),
            "deals_by_sector": dict(Counter(g["sector"]).most_common()),
            "deal_size_median_mm": _r(g["deal_size_mm"].median()),
            "deal_size_min_mm": _r(g["deal_size_mm"].min()),
            "largest_closed_deal_mm": _r(closed["deal_size_mm"].max()) if not closed.empty else None,
            "first_year": int(g["deal_year"].min()),
            "last_year": int(g["deal_year"].max()),
            "deal_types": dict(Counter(g["deal_type"]).most_common()),
            "distinctive_tags_lift_vs_market": tag_lift(g, base),
            "geographies": dict(Counter(g["geography"]).most_common()),
            "target_ownership": dict(Counter(g["target_ownership_pre"]).most_common()),
            "financing": dict(Counter(g["financing_type"]).most_common(3)),
            "median_target_margin_pct": _r(g["ebitda_margin_pct"].median()),
            "median_target_growth_pct": _r(g["revenue_growth_pct"].median()),
            "median_days_to_close": _r(closed["days_to_close"].median()),
            "relevant_closed_deals": int(((g["w"] >= C.RELEVANT_DEAL_W) & g.index.isin(closed.index)).sum()),
            "closed_in_size_band": len(in_band),
            "recent_in_band_closed": int((in_band["deal_year"] > ref_year - C.RECENT_YEARS).sum()),
        },
        "size_position": size_position(g, profile.ev_mm, cand.acquirer_type),
        "precedents": [_deal_row(r) for _, r in g.head(C.TOP_PRECEDENTS).iterrows()],
        "acquirer_comps": {"definition": "acquirer's closed control deals, adjacent sectors, ±2.5x size", **acq},
        "pricing_vs_peers": None,
        "risk": _risk_facts(g, cand, profile, ref_year),
        "score": cand.as_dict(),
    }
    pack["pricing_vs_peers"] = pricing_vs_peers(
        acq, market["pricing_peers_by_buyer_type"].get(cand.acquirer_type, {}), cand.acquirer_type
    )
    pack["descriptor"] = descriptor(pack)
    level, signals, points = suggested_conviction(pack, profile, market)
    pack["suggested_conviction"] = {"level": level, "signals": signals, "points": points}
    return pack


def _top(counts: dict[str, int], total: int) -> str:
    """'Private (9 of 24)', or 'Private and PE-Backed (9 each of 24)' on a tie. Never invents a single winner."""
    counts = {k: v for k, v in counts.items() if k and k != "unknown"}
    if not counts:
        return "n/a"
    items = list(counts.items())
    best = items[0][1]
    tied = [k for k, v in items if v == best]
    if len(tied) == 1:
        return f"{tied[0]} ({best} of {total})"
    return f"{' and '.join(tied[:3])} ({best} each of {total})"


def descriptor(pack: dict) -> str:
    """Deterministic one-line 'who they are' built only from pack facts (no outside knowledge)."""
    p, sp = pack["profile"], pack["size_position"]
    n = p["total_deals"]
    sectors = list(p["deals_by_sector"].items())
    sec = ", ".join(f"{k} ({v})" for k, v in sectors[:2])
    years = str(p["first_year"]) if p["first_year"] == p["last_year"] else f"{p['first_year']}-{p['last_year']}"
    iqr = sp.get("buyer_size_iqr_mm") or [None, None]
    kind = {"Financial Sponsor": "Financial sponsor", "Strategic": "Strategic acquirer"}.get(
        pack["acquirer_type"], "Acquirer"
    )
    parts = [
        f"{kind}: {n} deals in {years}, {pack['risk']['deals_last_3y']} in the last {C.RECENT_YEARS} years",
        f"most deals in {sec}",
        f"size sweet spot ${iqr[0]}-{iqr[1]}M (IQR)",
    ]
    if _top(p["target_ownership"], n) != "n/a":
        parts.append(f"most often buys from {_top(p['target_ownership'], n)} sellers")
    if _top(p["financing"], n) != "n/a":
        parts.append(f"most common financing {_top(p['financing'], n)}")
    return "; ".join(parts) + "."


def _sector_cosine(a: dict[str, int], b: dict[str, int]) -> float:
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def competitor_distance(a: dict, b: dict) -> float:
    """Distance between two packs: |log ratio of median deal size| + (1 - sector-mix cosine)."""
    ma, mb = a["profile"]["deal_size_median_mm"], b["profile"]["deal_size_median_mm"]
    size = abs(math.log(ma / mb)) if ma and mb else 1.0
    return size + (1 - _sector_cosine(a["profile"]["deals_by_sector"], b["profile"]["deals_by_sector"]))


def add_closest_competitors(packs: list[dict]) -> None:
    """Attach the nearest other buyer on the list (deterministic; ties broken by list order) and the
    figures on which they differ, so the page can name one data-backed difference."""
    for a in packs:
        others = [b for b in packs if b is not a]
        if not others:
            a["closest_competitor"] = None
            continue
        b = min(others, key=lambda o: competitor_distance(a, o))

        def facts(p: dict) -> dict:
            return {
                "deals": p["profile"]["total_deals"],
                "closed_in_size_band": p["profile"]["closed_in_size_band"],
                "deals_in_target_sector": p["risk"]["deals_in_target_sector"],
                "deals_last_3y": p["risk"]["deals_last_3y"],
                "deal_size_median_mm": p["profile"]["deal_size_median_mm"],
                "size_role": p["size_position"]["size_role"],
                "pricing_position": (p.get("pricing_vs_peers") or {}).get("position", "n/a"),
                "withdrawn_or_terminated": len(p["risk"]["withdrawn_or_terminated"]),
            }

        fa, fb = facts(a), facts(b)
        a["closest_competitor"] = {
            "acquirer": b["acquirer"],
            "acquirer_type": b["acquirer_type"],
            "distance": round(competitor_distance(a, b), 2),
            "this_vs_them": {k: [fa[k], fb[k]] for k in fa if fa[k] != fb[k]},
        }


def differentiator(pack: dict) -> str:
    """One-line summary of a pack for the other pages' <other_candidates> roster."""
    p = pack["profile"]
    top_sector = next(iter(p["deals_by_sector"]))
    return (
        f"{pack['acquirer']} ({pack['acquirer_type']}): {p['total_deals']} deals, most in {top_sector}; "
        f"median size ${p['deal_size_median_mm']}M; {p['closed_in_size_band']} closed in size band; "
        f"target is {pack['size_position']['size_role']} for it; "
        f"suggested conviction {pack['suggested_conviction']['level']}"
    )
