"""Pipeline: rank -> evidence -> LLM (parallel) -> validate -> distinctiveness pass -> report dict."""

from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from . import config as C
from . import eda, evidence, prompts, scoring, validate
from .data import Dataset
from .llm import Cache, LLMClient, OfflineClient, Result, generate
from .profile import TargetProfile

log = logging.getLogger(__name__)


def assumptions(profile: TargetProfile, market: dict, ds: Dataset) -> list[str]:
    """Plain-language assumptions shown at the top of every report."""
    out = []
    if profile.geography == "Regional":
        out.append(
            "Target region not specified: every named region scores as a footprint match; "
            "national / multi-regional deals score slightly lower."
        )
    if profile.ebitda_margin_pct is None and market["target_margin_assumed_pct"] is not None:
        out.append(
            f"'Strong EBITDA margin' taken as the dataset 75th percentile: {market['target_margin_assumed_pct']}%."
        )
    if market["implied_target_ebitda_mm"]:
        out.append(
            f"Implied target EBITDA ~${market['implied_target_ebitda_mm']}M = ${profile.ev_mm:g}M EV / "
            f"anchor median {market['peer_set']['ev_ebitda_median']}x (illustrative, not given)."
        )
    out.append(f"Recency measured from the latest year in the data ({int(ds.ref_year)}), not today's date.")
    if ds.features.get("valuation") == "computed":
        out.append("Multiples were computed from deal size and EBITDA/revenue (the file has none).")
    elif ds.on("valuation"):
        out.append("Stated multiples are used as given and never recomputed.")
    return out


def also_considered(ranked: list, picks: list, n: int | None = None) -> list[dict]:
    """Next-best buyers, so an MD can see who was left off and why (e.g. displaced by the mix floor)."""
    n = C.ALSO_CONSIDERED_N if n is None else n
    chosen = {c.acquirer for c in picks}
    cutoff = min(c.score for c in picks) if picks else 0
    out = []
    for overall, c in enumerate(ranked, 1):
        if c.acquirer in chosen:
            continue
        why = (
            "outscored a mix-floor pick; displaced to keep strategic/sponsor balance"
            if c.score > cutoff
            else "below cutoff"
        )
        out.append(
            {
                "overall_rank": overall,
                "acquirer": c.acquirer,
                "acquirer_type": c.acquirer_type,
                "score": round(c.score, 3),
                "n_deals": c.n_deals,
                "n_relevant": c.n_relevant,
                "note": why,
            }
        )
        if len(out) >= n:
            break
    return out


LEVEL_ORDER = {"Low": 0, "Medium": 1, "High": 2}


def top_signals(signals: list[str], n: int = 2) -> list[str]:
    """The n conviction signals with the largest absolute points ("+2 ...", "-1 ...")."""
    return sorted(signals, key=lambda s: -abs(float(s.split()[0])))[:n]


def rank_conviction_notes(pages: list[dict]) -> None:
    """Explain, on each page where it happens, why score rank and conviction tier disagree.

    Rank (score) = how much comparable deal activity the buyer has overall; conviction = direct
    evidence for this specific deal. A page diverges when a higher-ranked page has a lower tier, or
    a lower-ranked page a higher tier. Writes ``page["rank_vs_conviction"]`` (None when consistent).
    """
    lvl = [LEVEL_ORDER.get((p.get("rationale") or {}).get("conviction", {}).get("level", "")) for p in pages]
    for i, p in enumerate(pages):
        p["rank_vs_conviction"] = None
        if lvl[i] is None:
            continue

        def tag(q: dict) -> str:
            return (
                f"{q['candidate']['acquirer']} ({q['evidence']['suggested_conviction'].get('points', '?'):g} pts)"
                if isinstance(q["evidence"]["suggested_conviction"].get("points"), (int, float))
                else q["candidate"]["acquirer"]
            )

        mine = lvl[i]
        assert mine is not None  # checked above; narrows the type for the comparisons below
        above = [tag(q) for q, lv in zip(pages[:i], lvl[:i], strict=True) if lv is not None and lv < mine]
        below = [tag(q) for q, lv in zip(pages[i + 1 :], lvl[i + 1 :], strict=True) if lv is not None and lv > mine]
        if not above and not below:
            continue
        level = p["rationale"]["conviction"]["level"]
        sc = p["evidence"]["suggested_conviction"]
        pts = f"{sc['points']:g} evidence pts; " if isinstance(sc.get("points"), (int, float)) else ""
        why = pts + "; ".join(top_signals(sc["signals"]))
        detail = (
            f"it outranks {', '.join(below)} on overall activity, but they have more direct evidence for this deal"
            if below
            else f"{', '.join(above)} rank higher on overall activity but have less direct evidence for this deal"
        )
        p["rank_vs_conviction"] = (
            f"Ranked #{p['rank']} on score (overall comparable-deal activity) with {level} "
            f"conviction (direct evidence: {why}): {detail}."
        )


def cross_page_feedback(packs: list[dict], results: list[Result]) -> dict[str, str]:
    """Pages that need one regeneration because of what OTHER pages say, with the feedback to send."""
    order = [p["acquirer"] for p in packs]
    notes: dict[str, list[str]] = {}
    theses = {r.acquirer: r.rationale.strategic_fit_thesis for r in results if r.rationale}
    for a, b, j in validate.similar_pairs(theses):
        later, other = (b, a) if order.index(b) > order.index(a) else (a, b)
        notes.setdefault(later, []).append(
            f"Your fit thesis overlapped {j:.0%} with {other}'s. Make it specific to {later}'s own deals."
        )
    flags = {r.acquirer: [f.detail for f in r.rationale.risk_flags] for r in results if r.rationale}
    for name, dup in validate.repeated_risks(flags).items():
        notes.setdefault(name, []).append(
            "These risk flags are near-identical on 3+ other pages, so they are true of every buyer and already "
            f"covered by the report's Market context: {dup}. Replace them with risks specific to {name}'s data."
        )
    return {name: "\n\n" + " ".join(msgs) for name, msgs in notes.items()}


def coverage_warnings(ds: Dataset, profile: TargetProfile, market: dict) -> list[str]:
    """Up-front warnings when the file has little data near the requested target."""
    cov = eda.target_coverage(ds.df, profile, ds.vocab)
    out = []
    if cov["target_sector_in_band"] < C.THIN_SECTOR_BAND_DEALS:
        out.append(
            f"Thin coverage: only {cov['target_sector_in_band']} {profile.sector} deals within "
            f"${cov['size_band_mm'][0]}-{cov['size_band_mm'][1]}M; search widened to adjacent sectors "
            f"({cov['adjacent_sectors_in_band']} deals in band)."
        )
    if len(market["widening_steps"]) > 1:
        out.append(f"Valuation anchor widened: {' -> '.join(market['widening_steps'])}.")
    if market["peer_set"].get("n", 0) < C.MIN_COMPS:
        out.append(f"Thin comp set even after widening (n={market['peer_set'].get('n', 0)}).")
    return out


def run_identity(pages: list[dict]) -> str:
    """Stable id of a run's content: hash of its pages' cache keys (same pages -> same id, live or replay)."""
    keys = "|".join(sorted(p["cache_key"] for p in pages if p.get("cache_key")))
    return hashlib.sha256(keys.encode()).hexdigest()[:16]


def run_stamp(report: dict) -> dict:
    """What a reviewer needs to know about the live run behind a report: when, how long, cost, results."""
    m = report["meta"]
    return {
        "generated_at": m["generated_at"],
        "elapsed_s": m["elapsed_s"],
        "provider": m["provider"],
        "model": m["model"],
        "prompt_version": m["prompt_version"],
        "tokens_in": m["tokens_in"],
        "tokens_out": m["tokens_out"],
        "cost_usd": m["cost_usd"],
        "statuses": {
            s: sum(p["status"] == s for p in report["pages"]) for s in ("verified", "repaired", "unverified", "failed")
        },
    }


def run(
    ds: Dataset,
    profile: TargetProfile,
    client: LLMClient,
    cache: Cache,
    top_n: int = 10,
    excluded: frozenset[str] = frozenset(),
    workers: int | None = None,
    min_per_type: int | None = None,
) -> dict:
    """Stage 0 (EDA) -> rank -> evidence -> LLM pages (parallel) -> cross-page repair -> report dict.

    Never raises for a single page's failure; every warning ends up in report["warnings"].
    """
    t0 = time.perf_counter()
    workers = C.DEFAULT_WORKERS if workers is None else workers
    replay = client.provider == "replay"
    data_profile = eda.run(ds, profile)  # stage 0: know the data before scoring it
    scored = scoring.score_deals(ds, profile)
    ranked = scoring.aggregate(scored, excluded)
    floor = C.MIN_PER_TYPE if min_per_type is None else min_per_type
    pre: list[str] = []
    if floor and not ds.on("buyer_mix"):
        pre.append(f"Buyer-mix floor disabled: {ds.features['buyer_mix'].removeprefix('off: ')}.")
        floor = 0
    picks, warnings, displaced = scoring.select(ranked, top_n, floor, scored, profile)
    market = evidence.market_comps(scored, profile, ds.ref_year)
    also = also_considered(ranked, picks)
    warnings = pre + coverage_warnings(ds, profile, market) + warnings
    packs = [evidence.build(scored, c, profile, market, ds.ref_year) for c in picks]
    evidence.add_closest_competitors(packs)
    roster = {p["acquirer"]: evidence.differentiator(p) for p in packs}
    assume = assumptions(profile, market, ds)
    target = {**profile.model_dump(), "ebitda_margin_pct": market["target_margin_assumed_pct"]}
    system = prompts.SYSTEM

    if isinstance(client, OfflineClient):
        for p in packs:
            client.register(p, market)

    def user_for(pack: dict, extra: str = "") -> str:
        others = [v for k, v in roster.items() if k != pack["acquirer"]]
        return prompts.build_user(profile, pack, market, others, assume) + extra

    def job(pack: dict, extra: str = "") -> Result:
        # Everything the model was shown is a legitimate source of numbers (roster, assumptions too).
        shown = {
            **target,
            "assumptions": assume,
            "other_candidates": [v for k, v in roster.items() if k != pack["acquirer"]],
        }
        return generate(client, cache, system, user_for(pack, extra), pack, market, shown)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(job, packs))
        # One cross-page repair pass (only for live models; the offline template is deliberately uniform):
        #  - near-duplicate fit theses: regenerate the lower-ranked page of each pair
        #  - risk flags near-identical on 3+ other pages: market-wide, so move them off the page
        if not isinstance(client, OfflineClient):
            fixes = cross_page_feedback(packs, results)
            idx = {p["acquirer"]: i for i, p in enumerate(packs)}
            futures = {name: pool.submit(job, packs[idx[name]], note) for name, note in fixes.items()}
            for name, fut in futures.items():
                new = fut.result()
                if new.rationale:
                    results[idx[name]] = new
            if fixes:
                warnings.append(f"Cross-page repair pass regenerated: {', '.join(fixes)}")

    pages: list[dict[str, Any]] = []
    for c, pack, res in zip(picks, packs, results, strict=True):
        pages.append(
            {
                "rank": len(pages) + 1,
                "candidate": c.as_dict(),
                "evidence": pack,
                "rationale": res.rationale.model_dump() if res.rationale else None,
                "status": res.status,
                "issues": res.issues,
                "cached": res.cached,
                "cache_key": res.cache_key,
                "timing": {
                    "elapsed_s": res.elapsed_s,
                    "attempts": res.attempts,
                    "tokens_in": res.tokens_in,
                    "tokens_out": res.tokens_out,
                },
            }
        )
    rank_conviction_notes(pages)
    for p in pages:
        p["top_signals"] = top_signals(p["evidence"]["suggested_conviction"]["signals"])
    convictions = {p["rationale"]["conviction"]["level"] for p in pages if p["rationale"]}
    if len(pages) >= 3 and len(convictions) == 1:
        warnings.append(f"All acquirers received the same conviction ({convictions.pop()}); review signals.")
    failed = [p["candidate"]["acquirer"] for p in pages if p["status"] == "failed"]
    if failed:
        warnings.append(f"Narrative generation failed for: {failed}; evidence shown without narrative.")

    run_id = run_identity(pages)
    tin = sum(r.tokens_in for r in results)
    tout = sum(r.tokens_out for r in results)
    price = C.PRICES_PER_MTOK.get(client.model)
    report_meta_extra: dict[str, Any] = {}
    if replay:
        report_meta_extra["source_run"] = cache.get(f"run-{run_id}")
    report: dict[str, Any] = {
        "target": target,
        "assumptions": assume,
        "market": market,
        "pages": pages,
        "warnings": warnings,
        "excluded": sorted(excluded),
        "also_considered": also,
        "mix_floor": {"min_per_type": floor, "displaced": displaced},
        "data_profile": data_profile,
        "meta": {
            "provider": client.provider,
            "model": client.model,
            "prompt_version": C.PROMPT_VERSION,
            "data_sha256": ds.sha256,
            "data_rows": len(ds.df),
            "data_notes": list(ds.notes),
            # Replay output is byte-identical by design: its own stamps are fixed, and the live run it
            # reproduces is shown from the stored run stamp (source_run) instead.
            "generated_at": "replay" if replay else datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "elapsed_s": None if replay else round(time.perf_counter() - t0, 1),
            "run_id": run_id,
            "cache_hits": sum(p["cached"] for p in pages),
            "features": dict(ds.features),
            "tokens_in": tin,
            "tokens_out": tout,
            "cost_usd": round((tin * price[0] + tout * price[1]) / 1e6, 4) if price and not replay else None,
            **report_meta_extra,
        },
    }
    if not replay and not isinstance(client, OfflineClient):
        cache.put(f"run-{run_id}", run_stamp(report))
    return report
