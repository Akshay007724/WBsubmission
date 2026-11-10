"""Deterministic scoring of a finished report. No LLM judge: every metric is computed from report.json.

Usage: python -m acquirer_engine.eval out/<slug>/report.json      (writes eval.json next to it)
       python -m acquirer_engine.eval BEFORE.json AFTER.json      (prints a before/after table)
"""

from __future__ import annotations

import json
import re
import sys
from itertools import combinations
from pathlib import Path
from statistics import mean

from . import validate

# "or below peers" is a hedged list ("above, in line with or below peers"), not a claim.
BELOW_PEERS_RE = re.compile(r"(?<!\bor )below (?:the )?(?:[\w-]+ ){0,3}peers?\b", re.IGNORECASE)
LEVEL = {"Low": 0, "Medium": 1, "High": 2}
METRICS = [  # (key, label, direction): +1 higher is better, -1 lower is better, 0 descriptive
    ("pages_with_narrative", "Pages with narrative", +1),
    ("citation_validity", "Cited ids that exist in the evidence", +1),
    ("number_validity", "Numbers found in the evidence", +1),
    ("mean_thesis_jaccard", "Mean pairwise thesis similarity (trigram Jaccard)", -1),
    ("repeated_risk_flags", "Risk flags repeated on 3+ other pages", -1),
    ("below_peers_pages", "Pages saying 'below peers'", 0),
    ("conviction_levels_used", "Distinct conviction levels used", +1),
    ("rank_conviction_inversions", "Rank/conviction inversions (pairs)", 0),
    ("words_per_page", "Words per page (mean)", 0),
]


def page_text(page: dict) -> str:
    """All narrative text on one page (empty string when the narrative failed)."""
    x = page.get("rationale") or {}
    if not x:
        return ""
    parts = [
        x["headline"],
        x["acquirer_overview"],
        x["strategic_fit_thesis"],
        x["valuation_context"],
        x["conviction"]["rationale"],
        *(f["detail"] for f in x["risk_flags"]),
        *(p["why_relevant"] for p in x["precedent_activity"]),
    ]
    return " ".join(parts)


def below_peers_count(report: dict) -> int:
    """Pages whose narrative says the buyer prices 'below ... peers'."""
    return sum(bool(BELOW_PEERS_RE.search(page_text(p))) for p in report["pages"])


def _page_validity(page: dict, report: dict) -> tuple[int, int, int, int]:
    """(valid ids, cited ids, valid numbers, numbers) for one page."""
    nums, ids = validate.allowed(page["evidence"], report["market"], report["target"], report.get("assumptions", []))
    text = page_text(page)
    cited = validate.cited_ids(text)
    ctx = validate.numbers_with_context(validate.strip_ids(text, ids))
    ok_nums = sum(validate._num_ok(n, nums, counted) for n, counted in ctx)
    return len(cited & ids), len(cited), ok_nums, len(ctx)


def score(report: dict) -> dict:
    """Compute every metric in METRICS for one report dict."""
    pages = [p for p in report["pages"] if p.get("rationale")]
    v = [_page_validity(p, report) for p in pages]
    theses = {p["candidate"]["acquirer"]: p["rationale"]["strategic_fit_thesis"] for p in pages}
    grams = {k: validate._trigrams(t) for k, t in theses.items()}
    jac = [validate._jaccard(grams[a], grams[b]) for a, b in combinations(grams, 2)]
    flags = {p["candidate"]["acquirer"]: [f["detail"] for f in p["rationale"]["risk_flags"]] for p in pages}
    levels = [LEVEL[p["rationale"]["conviction"]["level"]] for p in pages]
    inversions = sum(1 for i, j in combinations(range(len(levels)), 2) if levels[i] < levels[j])
    return {
        "pages_with_narrative": f"{len(pages)}/{len(report['pages'])}",
        "citation_validity": round(sum(a for a, *_ in v) / max(1, sum(b for _, b, *_ in v)), 3),
        "number_validity": round(sum(c for *_, c, _ in v) / max(1, sum(d for *_, d in v)), 3),
        "mean_thesis_jaccard": round(mean(jac), 3) if jac else 0.0,
        "repeated_risk_flags": sum(len(x) for x in validate.repeated_risks(flags).values()),
        "below_peers_pages": below_peers_count(report),
        "conviction_levels_used": len(set(levels)),
        "conviction_spread": {k: levels.count(n) for k, n in LEVEL.items()},
        "rank_conviction_inversions": inversions,
        "words_per_page": round(mean(len(page_text(p).split()) for p in pages), 1) if pages else 0.0,
        "status": {
            s: sum(p["status"] == s for p in report["pages"]) for s in ("verified", "repaired", "unverified", "failed")
        },
    }


def table(before: dict, after: dict) -> str:
    """Markdown before/after table comparing two reports."""
    rows = ["| Metric | Before | After | Better is |", "|---|---|---|---|"]
    for key, label, direction in METRICS:
        rows.append(
            f"| {label} | {before.get(key, 'n/a')} | {after.get(key, 'n/a')} | "
            f"{'higher' if direction > 0 else 'lower' if direction < 0 else 'context'} |"
        )
    rows.append(f"| Conviction spread (H/M/L) | {_spread(before)} | {_spread(after)} | varied |")
    return "\n".join(rows)


def _spread(s: dict) -> str:
    c = s.get("conviction_spread", {})
    return f"{c.get('High', 0)}/{c.get('Medium', 0)}/{c.get('Low', 0)}"


def main(argv: list[str] | None = None) -> int:
    """Score one report (writes eval.json beside it) or print a before/after table for two reports."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) not in (1, 2):
        print(__doc__, file=sys.stderr)
        return 2
    reports = [json.loads(Path(a).read_text(encoding="utf-8")) for a in args]
    scores = [score(r) for r in reports]
    if len(args) == 1:
        out = Path(args[0]).with_name("eval.json")
        out.write_text(json.dumps(scores[0], indent=1), encoding="utf-8")
        for k, v in scores[0].items():
            print(f"{k:28} {v}")
        print(f"wrote {out}")
    else:
        print(table(*scores))
    return 0


if __name__ == "__main__":
    sys.exit(main())
