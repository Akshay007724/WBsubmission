"""Deterministic one-pager drafts built from the evidence pack (used when no LLM is available).

Not the product: it exists so the pipeline is testable and reviewable without an API key.
Every page it produces is labelled 'LLM disabled'.
"""

from __future__ import annotations


def _ids(rows: list[dict], n: int = 2) -> str:
    return "".join(f"[{r['transaction_id']}]" for r in rows[:n])


def _rival(pack: dict) -> str:
    cc = pack.get("closest_competitor")
    if not cc:
        return ""
    if not cc["this_vs_them"]:
        return f"Closest competitor on the list: {cc['acquirer']}."
    k, (mine, theirs) = next(iter(cc["this_vs_them"].items()))
    return f"Versus closest competitor {cc['acquirer']}: {k.replace('_', ' ')} {mine} vs {theirs}."


def write(pack: dict, market: dict) -> dict:
    """Deterministic template one-pager built only from the pack (labelled 'LLM disabled' in the report)."""
    p, risk, prec = pack["profile"], pack["risk"], pack["precedents"]
    top_sector, top_n = next(iter(p["deals_by_sector"].items()))
    acq = pack["acquirer_comps"]
    sug = pack["suggested_conviction"]

    if acq.get("n"):
        val = (
            f"{acq['n']} of its closed comparable deals price at a median {acq['ev_ebitda_median']}x EV/EBITDA "
            f"and {acq['ev_revenue_median']}x EV/Revenue, "
            f"{(pack['pricing_vs_peers'] or {}).get('position', 'not comparable with')} "
            f"its peers ({(pack['pricing_vs_peers'] or {}).get('benchmark', 'no same-type peers')}) "
            f"{_ids(prec, 1)}."
        )
    else:
        val = (
            f"No closed control deals by this acquirer inside the size band; see the report's Market context "
            f"for what the target should command. Closest precedent: {_ids(prec, 1)}."
        )

    risks = []
    if risk["withdrawn_or_terminated"]:
        w = risk["withdrawn_or_terminated"][0]
        risks.append(
            {"category": "execution", "detail": f"{w['outcome']} deal [{w['transaction_id']}] in {w['year']}."}
        )
    if risk["antitrust_signal"]:
        risks.append(
            {"category": "antitrust", "detail": f"{risk['deals_in_target_sector']} prior deals in the target sector."}
        )
    if risk["high_competition"]:
        risks.append(
            {"category": "competitive_process", "detail": f"Average {risk['avg_bidders']} bidders on its deals."}
        )
    risks.append(
        {"category": "integration", "detail": f"Most activity is in {top_sector} ({top_n} deals), not a direct match."}
    )
    risks.append(
        {"category": "valuation", "detail": f"Median deal size ${p['deal_size_median_mm']}M versus target EV."}
    )
    seen, uniq = set(), []
    for r in risks:
        if r["category"] not in seen:
            seen.add(r["category"])
            uniq.append(r)

    return {
        "headline": f"{p['closed_in_size_band']} closed deals in the size band; most activity in {top_sector}.",
        "acquirer_overview": (
            f"{pack['descriptor']} "
            f"Over-indexes on: {', '.join(p['distinctive_tags_lift_vs_market']) or 'no tag above market rate'}. "
            f"Target sizing: {pack['size_position']['size_role']}."
        ),
        "strategic_fit_thesis": (
            f"{p['relevant_closed_deals']} closed deals score as relevant to this target, led by "
            f"{_ids(prec)}. {p['closed_in_size_band']} closed deals sit in the size band. " + _rival(pack)
        ),
        "precedent_activity": [
            {
                "transaction_id": d["transaction_id"],
                "why_relevant": (
                    f"{d['sector']}, ${d['deal_size_mm']}M, {d['deal_type']}, {d['ev_ebitda']}x, {d['outcome']}"
                ),
            }
            for d in prec[:3]
        ],
        "valuation_context": val,
        "risk_flags": uniq[:4],
        "conviction": {"level": sug["level"], "rationale": "Signals: " + "; ".join(sug["signals"] or ["none"]) + "."},
    }
