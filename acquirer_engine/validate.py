"""Checks LLM output against the evidence it was given. This is the guard against hallucinated numbers.

Deal ids are not assumed to have any particular format: a citation is any [bracketed] token, and it
must be one of the ids present in the evidence. Known ids are stripped before numbers are extracted.
"""

from __future__ import annotations

import re
from itertools import combinations
from typing import Any

from . import config as C
from .schemas import Rationale

TXN_RE = re.compile(r"MA-\d{4}-\d{4}")  # this dataset's id format (used only as a fallback)
CITE_RE = re.compile(r"\[([^\[\]\s,;]{3,40})\]")  # a citation: [ID]
NUM_RE = re.compile(r"(?<![A-Za-z\d.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?")
LEVELS = ["Low", "Medium", "High"]
SMALL_INT_OK = 10  # small integers pass unchecked ("two risks", "one of") ...
# ... unless they count something; then they must be real evidence figures ("3 deals", "5 bidders").
COUNTED_NOUN_RE = re.compile(
    r"\s+(?:[a-z-]+\s+){0,2}?(deals?|bidders?|years?|withdrawn|terminated|closed|transactions?|acquisitions?|"
    r"platforms?|add-ons?|precedents?)\b",
    re.IGNORECASE,
)
# A dollar amount must carry "$": flags "168.1-391.6M" but not "$168.1-391.6M".
BARE_DOLLAR_RE = re.compile(r"(?<![$\d.,\-])\d[\d,]*(?:\.\d+)?(?:\s?-\s?\d[\d,]*(?:\.\d+)?)?\s?(?:M|B|bn|mm)\b")
SPECIALIST_RE = re.compile(
    r"\b(speciali[sz]\w*|focus(?:es|ed)?\s+on|focused|concentrat\w*\s+(?:in|on)|"
    r"dedicated\s+to)\b",
    re.IGNORECASE,
)
SPECIALIST_MIN_SHARE = 0.30
ATTRIB_WINDOW = 60  # "[ID] at 21.4x": the first multiple within this many characters, same clause
MULT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*x\b")
BENCHMARK_RE = re.compile(r"\b(?:peers?|median|benchmark|anchor|market)\b", re.I)
BEFORE_ID_MULT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*x(?:\s+EV/(?:EBITDA|Revenue))?\s*[(\[]?\s*$")


# ---------------------------------------------------------------- ids and numbers


def id_pattern(ids: set[str]) -> re.Pattern[str] | None:
    """Regex matching any of the given ids as a whole token (longest first)."""
    if not ids:
        return None
    alts = "|".join(re.escape(i) for i in sorted(ids, key=len, reverse=True))
    return re.compile(rf"(?<![\w-])(?:{alts})(?![\w-])")


def strip_ids(text: str, ids: set[str]) -> str:
    """Remove known ids (and MA-style ids) so their digits are not read as numbers."""
    rx = id_pattern(ids)
    text = rx.sub(" ", text) if rx else text
    return TXN_RE.sub(" ", text)


def cited_ids(text: str) -> set[str]:
    """Ids the text cites: [bracketed] tokens plus bare MA-style ids."""
    return set(CITE_RE.findall(text)) | set(TXN_RE.findall(text))


def _walk(obj: Any, nums: set[float], ids: set[str], key: str = "") -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if key == "transaction_id" and isinstance(obj, str):
        ids.add(obj)
        return
    if key == "transaction_ids" and isinstance(obj, (list, tuple)):
        ids.update(str(i) for i in obj)
        return
    if isinstance(obj, (int, float)):
        nums.add(round(float(obj), 2))
    elif isinstance(obj, str):
        ids.update(TXN_RE.findall(obj))
        nums.update(extract_numbers(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _walk(k, nums, ids)
            _walk(v, nums, ids, str(k))
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            _walk(v, nums, ids, key)


def extract_numbers(text: str) -> list[float]:
    """Every number in the text (MA-style ids removed first)."""
    return [n for n, _ in numbers_with_context(text)]


def numbers_with_context(text: str) -> list[tuple[float, bool]]:
    """(value, counts_something) per number; counts_something = followed by a counted noun ("3 deals")."""
    text = TXN_RE.sub(" ", text)
    out = []
    for m in NUM_RE.finditer(text):
        whole, frac = m.group(1), m.group(2)
        val = round(float(whole.replace(",", "") + (f".{frac}" if frac else "")), 2)
        out.append((val, bool(COUNTED_NOUN_RE.match(text, m.end()))))
    return out


def allowed(*sources: Any) -> tuple[set[float], set[str]]:
    """All numbers and ids that appear anywhere in the given evidence structures."""
    nums: set[float] = set()
    ids: set[str] = set()
    for s in sources:
        _walk(s, nums, ids)
    # Text numbers are parsed unsigned ("a gap of -4.9 turns" -> 4.9), so signed evidence counts by magnitude.
    return nums | {abs(n) for n in nums}, ids


def _num_ok(n: float, nums: set[float], counts_something: bool = False) -> bool:
    if n <= SMALL_INT_OK and float(n).is_integer() and not counts_something:
        return True
    if 1900 <= n <= 2100 and float(n).is_integer():
        return n in nums  # a year must be a year in the data, exactly (no rounding tolerance)
    return any(abs(n - a) <= 0.051 or (a and abs(n - a) / abs(a) <= 0.01) for a in nums)


# ---------------------------------------------------------------- individual rules


def misattributed(text: str, deals: dict[str, dict]) -> list[str]:
    """Flag a multiple written right after - or immediately before - a deal id that is not that deal's
    EV/EBITDA or EV/Revenue. A range before a cluster of ids ("19.4x to 25.7x [A][B]") is checked
    against the whole cluster."""
    rx = id_pattern(set(deals))
    if rx is None:
        return []
    out = []
    for m in rx.finditer(text):
        deal = deals[m.group()]
        actual = [v for v in (deal.get("ev_ebitda"), deal.get("ev_revenue")) if v is not None]
        window = text[m.end() : m.end() + ATTRIB_WINDOW]
        tail = re.split(r"[;,]|\.\s|\[", window)[0]
        after = MULT_RE.search(tail)
        # "11.6x [A] to 16.4x [B]" / "9.3x [A] against 20.3x on X [B]": a multiple followed by another id
        # in the same window belongs to that next id (its before-check covers it), not to this one.
        if after and rx.search(window, after.end()):
            after = None
        # "[A] against a strategic peer median of 14.2x": the multiple belongs to the named benchmark.
        if after and BENCHMARK_RE.search(tail[: after.start()]):
            after = None
        # "[A] and [B] cleared at 15.6x and 15.3x" (respectively): fine if ANY multiple in the clause is this deal's.
        clause_mults = [float(x) for x in MULT_RE.findall(tail)]
        matches_any = any(abs(v - a) <= 0.051 for v in clause_mults for a in actual)
        if after and actual and not matches_any:
            out.append(f"{m.group()} quoted at {float(after.group(1))}x but the deal's multiples are {actual}")
        start = m.start() - 1 if text[max(0, m.start() - 1) : m.start()] == "[" else m.start()
        if text[max(0, start - 1) : start] == "]":
            continue  # not the first id of a cluster
        before = BEFORE_ID_MULT_RE.search(text[max(0, start - 30) : m.start()])
        run = re.match(r"(?:\[?[^\[\]\s,;]+\]?)+", text[start:])
        cluster = rx.findall(run.group()) if run else [m.group()]
        pool = [v for i in cluster for v in (deals[i].get("ev_ebitda"), deals[i].get("ev_revenue")) if v is not None]
        if before and pool and all(abs(float(before.group(1)) - a) > 0.051 for a in pool):
            out.append(f"{m.group()} quoted at {float(before.group(1))}x but the deal's multiples are {actual}")
    return out


def specialist_claims(text: str, pack: dict, other_names: list[str]) -> list[str]:
    """A sentence calling THIS buyer a specialist in / focused on a sector must name its single top sector,
    holding >= 30% of its deals. Sentences that mention another candidate are about them, so skipped."""
    counts = pack["profile"]["deals_by_sector"]
    total = sum(counts.values()) or 1
    best = max(counts.values(), default=0)
    top = [s for s, v in counts.items() if v == best]
    issues = []
    for sentence in re.split(r"(?<=[.;!?])\s+", text):
        if not SPECIALIST_RE.search(sentence) or any(o and o.lower() in sentence.lower() for o in other_names):
            continue
        low = sentence.lower()
        claimed = []
        for s in {*counts, *C.SECTORS}:
            sec = re.escape(s.lower())
            # the sector must be the object of the claim: "focused on X" / "concentrated in X" / "an X specialist"
            if re.search(SPECIALIST_RE.pattern + rf"\s+(?:the\s+|a\s+)?{sec}", low, re.IGNORECASE) or re.search(
                rf"{sec}\s+(?:specialist|focused)", low
            ):
                claimed.append(s)
        for sector in claimed:
            n = counts.get(sector, 0)
            if not (top == [sector] and n / total >= SPECIALIST_MIN_SHARE):
                issues.append(
                    f"calls the buyer focused on {sector} ({n} of {total} deals) but its top sector is "
                    f"{' / '.join(top)} ({best} of {total}); a specialist claim needs the single top "
                    f"sector with >= {SPECIALIST_MIN_SHARE:.0%} of deals"
                )
    return issues


def market_only_figures(pack: dict, market: dict) -> set[float]:
    """Market-wide figures printed once in the report's Market context section. A page may not restate
    them unless the same number is also one of this buyer's own facts."""
    own, _ = allowed(pack)
    figs = {
        market["peer_set"].get("ev_ebitda_median"),
        market["peer_recent_3y"].get("ev_ebitda_median"),
        market["same_sector_band"].get("ev_ebitda_median"),
        market.get("implied_target_ebitda_mm"),
        market.get("implied_target_revenue_mm"),
    }
    return {round(float(f), 2) for f in figs if f is not None} - own


def _text_fields(r: Rationale) -> dict[str, str]:
    fields = {
        "headline": r.headline,
        "acquirer_overview": r.acquirer_overview,
        "strategic_fit_thesis": r.strategic_fit_thesis,
        "valuation_context": r.valuation_context,
        "conviction_rationale": r.conviction.rationale,
    }
    for i, p in enumerate(r.precedent_activity):
        fields[f"precedent[{i}]"] = f"[{p.transaction_id}] {p.why_relevant}"
    for i, f in enumerate(r.risk_flags):
        fields[f"risk[{i}]"] = f.detail
    return fields


# ---------------------------------------------------------------- the checker


def check(r: Rationale, pack: dict, market: dict, target: dict) -> list[str]:
    """Every rule the prompt states, applied to one page. Returns human-readable issues (empty = verified)."""
    nums, own_ids = allowed(pack, target)
    m_nums, m_ids = allowed(market)
    nums |= m_nums
    ok_ids = own_ids | m_ids
    issues: list[str] = []
    deals = {d["transaction_id"]: d for d in pack.get("precedents", [])}
    market_wide = market_only_figures(pack, market)
    others = [str(o).split(" (")[0] for o in target.get("other_candidates", [])]
    if pack.get("closest_competitor"):
        others.append(pack["closest_competitor"]["acquirer"].split(" (")[0])
    for name, text in _text_fields(r).items():
        issues += [f"{name}: {bad}" for bad in misattributed(text, deals)]
        bad_ids = sorted(cited_ids(text) - ok_ids)
        if bad_ids:
            issues.append(f"{name}: cites transaction ids not in the evidence: {bad_ids}")
        plain = strip_ids(text, ok_ids)
        found_ctx = numbers_with_context(plain)
        bad_nums = sorted({n for n, counted in found_ctx if not _num_ok(n, nums, counted)})
        if bad_nums:
            issues.append(f"{name}: numbers not found in evidence: {bad_nums}")
        shared = sorted({n for n, _ in found_ctx if n in market_wide})
        if shared:
            issues.append(
                f"{name}: restates market-wide figures {shared}; they are printed once in Market "
                "context, so cite only this buyer's own figures"
            )
        bare = BARE_DOLLAR_RE.findall(plain)
        if bare:
            issues.append(f"{name}: dollar amounts missing '$': {bare}")
        issues += [f"{name}: {claim}" for claim in specialist_claims(text, pack, others)]
        low = text.lower()
        hits = [p for p in C.BANNED_PHRASES if re.search(rf"\b{re.escape(p)}\b", low)]
        if hits:
            issues.append(f"{name}: banned generic phrases used: {hits}")
        limit = C.WORD_LIMITS.get(name)
        if limit and len(text.split()) > limit * 1.15:
            issues.append(f"{name}: {len(text.split())} words exceeds limit {limit}")
    if len({f.category for f in r.risk_flags}) < 2:
        issues.append("risk_flags: need at least two different categories")
    if not any(cited_ids(t) & ok_ids for t in (r.strategic_fit_thesis, r.valuation_context)):
        issues.append("strategic_fit_thesis/valuation_context: cite at least one transaction_id")
    rival = (pack.get("closest_competitor") or {}).get("acquirer")
    if rival and rival.split(" (")[0].lower() not in r.strategic_fit_thesis.lower():
        issues.append(f"strategic_fit_thesis: name the closest competitor {rival} and one data-backed difference")
    sug = pack["suggested_conviction"]["level"]
    if abs(LEVELS.index(r.conviction.level) - LEVELS.index(sug)) > 1:
        issues.append(f"conviction: moved more than one notch from suggested '{sug}'")
    return issues


# ---------------------------------------------------------------- cross-page rules


def _trigrams(text: str) -> set[tuple[str, ...]]:
    w = re.findall(r"[a-z]+", CITE_RE.sub(" ", TXN_RE.sub(" ", text.lower())))
    return {tuple(w[i : i + 3]) for i in range(len(w) - 2)}


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def similar_pairs(theses: dict[str, str], threshold: float | None = None) -> list[tuple[str, str, float]]:
    """Pairs of pages whose fit theses overlap more than the distinctiveness threshold."""
    threshold = C.DISTINCT_JACCARD_MAX if threshold is None else threshold
    grams = {k: _trigrams(v) for k, v in theses.items()}
    return [(a, b, round(j, 2)) for a, b in combinations(grams, 2) if (j := _jaccard(grams[a], grams[b])) > threshold]


def repeated_risks(
    flags_by_page: dict[str, list[str]], min_pages: int | None = None, threshold: float | None = None
) -> dict[str, list[str]]:
    """Risk flags near-identical (trigram Jaccard > threshold) to a flag on >= min_pages OTHER pages.
    Those are true of every buyer and belong in the report's Market context, not on a page."""
    min_pages = C.REPEATED_RISK_MIN_PAGES if min_pages is None else min_pages
    threshold = C.DISTINCT_JACCARD_MAX if threshold is None else threshold
    grams = {(page, i): _trigrams(t) for page, fl in flags_by_page.items() for i, t in enumerate(fl)}
    out: dict[str, list[str]] = {}
    for (page, i), g in grams.items():
        twins = {p2 for (p2, _), g2 in grams.items() if p2 != page and _jaccard(g, g2) > threshold}
        if len(twins) >= min_pages:
            out.setdefault(page, []).append(flags_by_page[page][i])
    return out
