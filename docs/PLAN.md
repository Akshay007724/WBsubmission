# Implementation Plan

## Problem

Banker ask: *the 10 most likely acquirers for a ~$200M EV, private, regional, high-margin Healthcare Services company, each with a one-page rationale.* The constraints from the brief:
- CSV is the only data source;
- an LLM must write the rationale;
- nothing is hardcoded;
- one command, under 60s;
- an honest README.

### What the banker actually needs
- A ranked list that is **defensible line by line**: for any name, why it's on the list and why above the next one.
- **Specific, cited** evidence: deal ids, sizes, multiples. Not boilerplate.
- **Valuation framing** from closed comps.
- **Risks** that will come up in the MD conversation (antitrust, financing, competitive process).
- A **conviction** call that actually differentiates buyers.

## Data findings that shaped the design

| Finding | Consequence |
|---|---|
| `sub_sector == sector` on all 500 rows | Adjacency must be modelled at sector level |
| Healthcare Services: 46 deals, 12 in $100–400M | Use a sector adjacency matrix instead of a narrow filter |
| 15 PE firms have 14–25 deals each; most strategics have 1–6 | Empirical-Bayes shrinkage + strategic/sponsor mix floor |
| Deal size $6M–$14.4B, median $154M | Log-space size kernel, medians instead of means |
| EV ≠ EV/EBITDA × EBITDA in 403/500 rows | Use stated multiples as given; never recompute; document it |
| 94 non-closed deals; some "Pending" from 2016–2020 | Comps are Closed only; failed deals become risk signals; stale pending deals are down-weighted |
| Odd label pairings (hospital "LBO", PE "Merger of Equals") | Deal type carries a light weight; the prompt says not to build a thesis on one label |
| `days_to_close` blank in 94 rows | Medians skip NaN |

## Phases & status

| # | Phase | Files | Status |
|---|---|---|---|
| 1 | Data load and validation; target profile | `data.py`, `profile.py` | ✅ |
| 2 | Scoring, shrinkage, mix floor, tie-break | `scoring.py`, `config.py` | ✅ |
| 3 | Evidence packs, market comps ladder, conviction points | `evidence.py` | ✅ |
| 4 | Prompts, schema, provider adapter, cache | `prompts.py`, `schemas.py`, `llm.py` | ✅ (offline + fake-client tested; real-key run pending) |
| 5 | Validator and repair loop; distinctiveness pass | `validate.py`, `engine.py` | ✅ |
| 6 | HTML/MD/JSON rendering | `render.py`, `templates/` | ✅ |
| 7 | Stretch goals: arbitrary targets, exclude/feedback, compare | `profile.py`, `feedback.py`, `cli.py` | ✅ |
| 0 | EDA as pipeline stage 0 (12 checks, `--eda-only`, `docs/EDA.md`) | `eda.py` | ✅ |
| 8 | Tests (42) | `tests/` | ✅ |
| 8b | Senior M&A practitioner review and fixes | `docs/MA_PRACTITIONER_REVIEW.md`, `evidence.py`, `prompts.py` | ✅ |
| 9 | README, sample output from a real LLM run, demo video | `README.md`, `sample_output/` | README ✅; sample output and video need an API key |

## Edge cases

| Area | Case | Handling |
|---|---|---|
| Input | BOM, header case/whitespace | `utf-8-sig`, header normalization |
| Input | Missing or renamed column | `DataError` naming the missing columns, exit 2 |
| Input | Non-numeric/NaN/≤0 size, missing acquirer/year | Dropped and counted in `meta.data_notes` |
| Input | NaN multiples | Kept for activity, excluded from comps |
| Input | Duplicate rows / duplicate transaction_id | Deduped, with a note |
| Input | Acquirer name casing/whitespace variants | Canonical casefold key, most common display spelling |
| Input | Missing/empty CSV, missing target YAML | Clear error, exit 2 |
| Profile | Unknown/typo sector, geography, ownership | "Did you mean …", exit 2 |
| Profile | EV ≤ 0 | pydantic error, exit 2 |
| Profile | EV far outside the data range | Warning; comps widen |
| Profile | Region unspecified ("regional") | Documented assumption: any named region matches |
| Profile | "Strong margin" not quantified | Dataset 75th percentile (21.5%), stated as an assumption |
| Selection | Fewer than N eligible acquirers | Returns what exists; warning on the report |
| Selection | Too few of one buyer type for the floor | Floor relaxed; warning |
| Selection | Ties | Ordered by score, then mass, then recency, then name |
| Selection | Small-sample strategics vs. high-volume PE | Shrinkage (k=3) + mix floor |
| Selection | Acquirer with only withdrawn deals | Ineligible (needs ≥1 Closed) |
| Comps | Fewer than 5 comparable closed deals | Widening ladder (±2.5x → ±4x → drop margin filter), each step shown |
| Comps | Pending/withdrawn deals polluting multiples | Closed only |
| Comps | Target EBITDA not given | Implied EBITDA = EV / peer median, labelled illustrative |
| Risk | Large strategics in the target sector | `antitrust_signal` fact |
| Risk | Not-for-profit health systems | `nonprofit_health_system` fact (heuristic list, flagged) |
| Risk | Crowded processes | `avg_bidders`, `high_competition` |
| Risk | Failed or stale deals | Listed by transaction_id |
| LLM | Missing API key / unknown provider | Clear message suggesting `--provider offline`, exit 2 |
| LLM | 429 / 5xx / timeout | SDK exponential backoff (2 retries, 60s timeout) |
| LLM | Truncation / refusal / invalid JSON / schema mismatch | Raised; repair retry with the errors |
| LLM | Invented numbers or transaction ids | Validator; repair; otherwise an `unverified` badge with issues listed |
| LLM | Generic boilerplate | Banned phrases, the "vs. other candidates" rule, the roster |
| LLM | Near-identical theses | Trigram Jaccard > 0.35, regenerated once |
| LLM | Same conviction for every buyer | Points-based prior; ±1 notch rule; warning if it still happens |
| LLM | World knowledge vs. dataset (e.g. Steward's 2024 bankruptcy) | Isolated `external_context` field; `--dataset-only` forbids it |
| LLM | One acquirer fails entirely | Evidence still rendered with a "narrative unavailable" note; exit 0; exit 1 only if all fail |
| Repro | Run-to-run variance | Temperature 0, seed (OpenAI), validated-only cache, metadata stamp |
| Output | `&`, `'`, `<` in names | jinja autoescape (HTML); Markdown rendered raw |
| Output | Windows/macOS paths, encoding | `pathlib`, explicit UTF-8 |
| Output | Name containing parentheses ("Aspen Dental (ADMI)") | Offline client parses with `rsplit` |
| Feedback | Unknown or ambiguous name to exclude | Fuzzy and substring match; otherwise lists candidates, exit 2 |
| Feedback | Corrupt `feedback.json` | Moved to `.corrupt.json`; starts fresh with a warning |
| Feedback | Every candidate excluded | "Only k eligible" warning |
| Compare | Same file passed twice | Warning; run proceeds |
| Perf | Under 60s | 5 parallel workers; cache; `--workers` |

## Deferred, with reasons

- **Learned weights from feedback.** Needs labelled banker data; exclusion reasons are already captured.
- **Embedding similarity on tags and target descriptions.** Twelve fixed tags don't justify the cost.
- **PitchBook/CapIQ ingestion, fund dry-powder and vintage data.** Out of scope; this is where most real-world lift would come from.
- **LLM-as-judge quality eval with a banker-graded rubric.** The right next step for prompt iteration.
- **Web UI.** The brief favours CLI depth.
