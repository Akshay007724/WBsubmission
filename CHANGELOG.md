# Changelog

## 1.0.0

### Engine
- **Ranking:** a deterministic acquirer score (sector adjacency × log-size similarity × recency × outcome × profile fit, with shrinkage), plus a buyer-mix floor of at least 3 strategics and 3 sponsors.
  - A floor pick needs a closed deal in the size band or the target's sector.
  - Every swap is shown in the report with both scores; `--min-per-type 0` gives a pure score ranking.
- **Valuation:** two benchmarks.
  - *How a buyer pays*: its own comps against same-type peers under identical filters, at any margin.
  - *What the target should command*: a high-margin anchor.

  EBITDA margin and EV/EBITDA correlate at 0.56 in this data, so a single margin-filtered peer set made most buyers look cheap.
- **Size role:** data-driven. Add-on / platform / upper end / stretch for sponsors; tuck-in / core-sized / stretch for strategics. Labelled Platform Investment deals are used only when the target is comparable to them.
- **Conviction:** an evidence-points prior, with a note on every page where rank and conviction disagree.
- **Fact pack per buyer:** a "who they are" descriptor and the closest competitor on the list (by size and sector mix).

### LLM and checker
- **Every fact comes from the CSV.**
  - The output schema has no field for outside knowledge, and the prompt forbids it.
  - The checker rejects numbers, years and cited deal IDs that are not in the data.
  - It also rejects multiples attributed to the wrong deal, uncounted integers, `$`-less amounts, unsupported "specialist" claims, restated market-wide figures, risks repeated across pages, and pages that don't name their closest competitor.
- **Repair:** each failing page is regenerated once with the specific errors, then one cross-page pass. Anything still failing is shown as *unverified* with its issues listed.
- **Checker false positives fixed** (found in live runs, each with a regression test): a benchmark multiple written after a deal ID ("[ID] against a strategic peer median of 14.2x") is no longer attributed to that deal, and a signed fact-pack value ("a gap of -4.9 turns") is matched by magnitude. A fresh live run after the fix: 8 verified, 2 repaired, 0 unverified.
- **Providers:** Anthropic (default), OpenAI, Gemini, local OpenAI-compatible servers, offline (template text) and replay (no key; reproduces a cached run byte for byte).

### Data
- Works with any M&A deals CSV:
  - header mapping (exact name, then aliases, then fuzzy match with a confidence score; `--schema-map` to override);
  - unit conversion ($/K/M/B, fractions vs percentages, dates);
  - graceful degradation: missing columns switch off only the features that need them, and values are never invented;
  - sector adjacency derived from co-investment when the sectors are not the healthcare preset.
- **Stage-0 EDA** on every run: data-quality findings, each with the engine's response, plus four charts.

### Engineering
- Packaging (`pyproject.toml`, `acquirer-engine` CLI), ruff, mypy, 151 tests at 94% coverage, CI (Python 3.11 and 3.12).
- Error classes with exit codes; structured logs (`--log-json`); validated settings overrides (`--set`, `--config`).
- Deterministic evaluation (`python -m acquirer_engine.eval`).
- A secret-scanned submission script, a Dockerfile, and `run.sh` / `run.bat`.

### Sample run (`sample_output/`)
- **Live run:** 2026-11-09 13:19 UTC, Claude Sonnet 5.5 (prompt v1.6), 26.3 s engine time (27.6 s wall clock), about $0.37.
- **Pages:** 8 verified, 2 repaired, 0 unverified.
- **Grounding:** 100% of numbers found in the data; 57 cited deal IDs, all in the CSV and attributed to the right buyer or the market peer set; no page cites a deal flagged by the financial-consistency checks.
- **Reproducing it:** `./run.sh` (or `python -m acquirer_engine --provider replay`) reproduces the report byte for byte. The report header shows the live run's timestamp, timing and cost.

| Metric (deterministic eval) | Initial version | 1.0.0 |
|---|---|---|
| Numbers found in the evidence | 0.982 | 1.0 |
| Cited IDs that exist in the evidence | 1.0 | 1.0 |
| Pages saying "below peers" | 7 | 2 (both backed by the like-for-like labels) |
| Rank/conviction inversions (pairs, each explained on its page) | 8 | 6 |
| Words per page | 513 | 476 |
| Conviction spread (High/Medium/Low) | 3/4/3 | 4/3/3 |

### Data quality
The CSV is used as given. Contradictory values are reported, never "corrected", because the engine cannot know which field is wrong. The EDA flags:
- **EV vs stated multiple × EBITDA:** inconsistent on 403 of 500 rows.
- **EBITDA far too small for the deal size:** 17 deals imply EV/EBITDA more than 3x their stated multiple (e.g. MA-2022-0272, a $220.7M deal on $1.0M EBITDA).
- **Stated margin vs EBITDA ÷ revenue:** off by more than 2 pp on 322 rows.

Ranking and valuation use only the stated multiples and margin, never the revenue or EBITDA fields. A buyer's **capacity** (largest deal, stretch check) counts closed deals only. For example, New Mountain's largest closed deal is $590.2M; its $6.9B insurer carve-out was withdrawn and is listed as such.

### Known limitations
- **Synthetic data:** the deals are synthetic (with real acquirer names), so don't quote the multiples as market data.
- **CSV-only:** pages cannot mention events the data doesn't hold (later acquisitions, bankruptcies). Check entity status before contacting a buyer.
- **Hand-set weights:** weights and thresholds are judgment calls, overridable but not fitted.
- **Checker limits:** it catches invented and misattributed figures and copied boilerplate, not paraphrased boilerplate or a true number used misleadingly.
