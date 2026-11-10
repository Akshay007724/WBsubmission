# M&A Acquirer Identification Engine

Given a target profile and a deals CSV, this tool ranks the 10 most likely acquirers and writes a one-page, data-cited rationale for each. Each page covers: overview, strategic fit, precedents, valuation context, risk flags and conviction.

**Time to review: one command, no API key, under 60 seconds.** Reviewers: see [START_HERE.md](START_HERE.md) (`./run.sh` on Mac/Linux; on Windows rename `run.bat.txt` to `run.bat`, since Gmail blocks `.bat` attachments).

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m acquirer_engine --provider replay        # reproduces sample_output/ byte for byte, ~3s, no key
open out/healthcare-services-200m-regional-private/report.html
```

New here? [DEMO.md](DEMO.md) walks through every step. [CHANGELOG.md](CHANGELOG.md) summarises the release.

## Run

| Goal | Command |
|---|---|
| Reproduce the committed sample without a key | `python -m acquirer_engine --provider replay` |
| Full pipeline, no key (deterministic template text, labelled "LLM disabled") | `python -m acquirer_engine --provider offline` |
| Real run (~30s on Claude Sonnet 5.5) | put `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY` / `GEMINI_API_KEY`) in `.env`, then `python -m acquirer_engine` |
| Data profile only (stage 0) | `python -m acquirer_engine --eda-only` → `out/<slug>/eda.md` and charts |
| Another target | `--sector "Dental" --ev 150 --geography Southeast`, or `--target targets/behavioral_500m.yaml` |
| Another CSV | `--data file.csv` (headers mapped automatically; `--schema-map map.yaml` to override) |
| Banker feedback | `--exclude "Blackstone" --reason "conflicted"` (persists per target); `--clear-feedback` |
| Compare two targets | `--compare targets/default.yaml targets/behavioral_500m.yaml` |
| Pure score ranking (no buyer-mix floor) | `--min-per-type 0` |
| Tune a weight | `--set MIN_COMPS=8` or `--config settings.yaml` (also `ACQ_*` env vars) |
| Installed CLI | `pip install .` then `acquirer-engine ...` |
| Docker (replay, no key) | `make docker` (build, run replay in the container, compare with `sample_output/`). On a slow network, run `make wheelhouse` first so the build installs offline |

**Sample output:** [`sample_output/`](sample_output/) holds a real Claude Sonnet 5.5 run (`report.html`, `.md`, `.json`, `eval.json`), plus the EDA report and charts.
- **Live run:** 2026-11-09, 26.3 s engine time (27.6 s wall clock), about $0.37.
- **Pages:** 8 verified, 2 repaired.
- **Grounding:** 100% of numbers found in the data; 57 cited deal IDs, all in the CSV.
- **Run stamp:** the report header shows the live run's timestamp, timing, tokens and cost, even when rendered by replay.

**Demo video:** sent as an attachment with this submission (a recorded live run plus every mode). No key is needed to review: `--provider replay` reproduces the sample exactly, and `--provider offline` runs the whole pipeline end to end.

## How it works

```
CSV → schema + units → EDA (stage 0) → score every deal → rank acquirers (+ mix floor) → fact pack per acquirer
    → LLM writes from the pack (10 in parallel) → checker verifies each page → cross-page repair → report + eval
```

Code ranks and computes; the LLM writes; the checker verifies. Details are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

0. **Load any M&A CSV.**
   - Headers are mapped to a canonical schema (exact name, then aliases, then a fuzzy match with a confidence score).
   - Units are normalised: $/K/M/B, fractions vs percentages, dates.
   - Missing columns switch off only the features that need them, and the report says which and why. Values are never invented.
   - Vocabulary comes from the file. The healthcare sector adjacency is a preset; for other sectors, adjacency is derived from which acquirers buy in which sectors.
1. **EDA (stage 0)** runs first on every invocation. It reports 15+ findings (schema confidence, missing values, coercion failures, outliers, EV/multiple consistency, sector coverage near the target, buyer concentration, label noise, margin/multiple correlation and more). Each finding is paired with how the engine responds, alongside four charts.
2. **Deal relevance** = sector adjacency × log-size similarity × recency (4-year half-life) × outcome × profile fit.
3. **Acquirer score** = relevant volume + a *shrunk* average fit (one lucky deal can't win) + the best precedent − a penalty for failed deals.
   - A **buyer-mix floor** keeps at least 3 strategics and 3 sponsors.
   - A floor pick needs a closed deal in the size band or the target's sector, and every swap is shown with both scores.
4. **Fact pack per acquirer**, all deterministic:
   - a "who they are" descriptor (pace, top sectors, size sweet spot, seller and financing habits);
   - a **size role** (add-on / platform / stretch for sponsors; tuck-in / core-sized / stretch for strategics);
   - precedents with IDs;
   - **two valuation benchmarks**:
     - *how this buyer pays*: like for like against same-type peers at any margin;
     - *what this target should command*: a high-margin anchor.

     Margin and EV/EBITDA correlate (0.56) here, so mixing the two made buyers look cheap.
   - risk facts;
   - a conviction prior from evidence points;
   - the **closest competitor** on the list (by size and sector mix), with the figures where they differ.
5. **LLM** (Claude Sonnet 5.5 by default, structured output) writes the six sections from the pack only, citing `[ID]`s. **Every fact comes from the CSV**: the output schema has no field for outside knowledge, the prompt forbids it, and the checker rejects numbers and years that aren't in the data. Market-wide figures appear **once**, in the report's Market context section, not on every page.
6. **Checker** (`validate.py`):
   - every number and cited ID must exist in the evidence;
   - a multiple next to an ID must be that deal's;
   - counts ("3 deals") must be real;
   - dollar amounts need `$`;
   - "specialist in X" needs X to be the buyer's clear top sector (≥ 30% of its deals);
   - market-wide figures must not be restated;
   - the closest competitor must be named;
   - no banned filler; word limits apply;
   - conviction may move at most one notch from the prior.

   A failing page is regenerated once with the errors. Then one cross-page pass regenerates near-duplicate theses and risks repeated on 3+ pages. Anything still failing is shown as `unverified` with its issues listed.
7. **Report:**
   - summary table: rank, score, conviction and its top 2 signals;
   - a "why rank and conviction differ" note wherever they disagree (score = overall comparable-deal activity; conviction = direct evidence for this deal);
   - also-considered and mix-floor swaps;
   - data profile, deterministic evaluation and the run stamp (model, prompt version, data sha256, settings fingerprint, tokens, cost).

## Prompt design

- **Role:** a VP writing for an MD.
- **Rules:** 13 numbered rules, **every one machine-checked**; a rule the code can't check is a suggestion.
- **Input:** facts arrive as tagged JSON blocks. Text inside them is data from a CSV, never instructions.
- **Style:** a short exemplar about a made-up buyer sets the density bar.
- **Sponsors vs strategics:** they get different reasoning rules. Sponsors are judged on platform vs add-on, deployment pace and size; strategics on footprint, vertical integration, antitrust and financing.
- **Comparison:** each page must state one data-backed difference against its closest competitor.
- **CSV-only facts:** rule 1 forbids outside knowledge about any company or market (ownership, size, HQ, strategy, history, regulation, news). Risks must come from the buyer's own data (failed deals, size gaps, overlap counts, bidders, financing). There is no output field where outside facts could go.
- **Versioning:** `PROMPT_VERSION` is bumped on every prompt change, which invalidates the cache.

## Assumptions

- **Region:** "regional" with no region given means any named region counts as a footprint match.
- **Margin:** "strong EBITDA margin" = the dataset's 75th percentile (21.5%). Implied target EBITDA (EV ÷ anchor median) is illustrative.
- **Recency:** measured from the latest year in the data (2024), so results are reproducible.
- **Multiples and margins:** stated multiples and margins are used as given. The CSV's revenue/EBITDA fields contradict them on many rows (17 deals imply EV/EBITDA >3x their stated multiple; 322 margins differ from EBITDA ÷ revenue by >2pp), so those fields never drive ranking or valuation. The EDA lists the affected rows.
- **Buyer capacity:** the largest deal and the stretch check use closed deals only; withdrawn or pending deals are appetite, not capacity.
- **Multiples (detail):** stated multiples are used as given. EV ≠ multiple × EBITDA on 403 of 500 rows, so they are never recomputed (computed only when a file has no multiples at all, and then labelled).
- **Buyer-mix floor of 3 per type:** kept because the data supports it. Both floor picks (Advocate, Ardent) have closed Healthcare Services deals. The cost is visible in the report: KKR (0.466) and Vista (0.468) are displaced. `--min-per-type 0` gives the pure score ranking.
- **No outcome column:** every deal counts as completed, and the report says so.

## Non-determinism

- **Deterministic:** ranking, comps, size roles, conviction priors and every checker rule.
- **LLM:** there is no seed on Claude; thinking effort is set to medium.
- **Cache:** only answers that pass the checker are cached, keyed by provider, model, prompt version, full prompt and schema. Changing the data, weights or prompts invalidates it automatically.
- **Replay:** `--provider replay` serves the cache and fixes the run stamp, so its output is byte-identical (CI checks this).
- **Live runs:** without the cache, wording varies between runs; facts, ranking and checks don't.

## Known limitations

- **Synthetic data:** the deals are synthetic (with real acquirer names), so don't quote the multiples as market data. Because every fact must come from the CSV, the pages can't mention things the data doesn't hold, such as a buyer's later acquisition or bankruptcy. A banker should check entity status before calling.
- **Hand-set weights:** all weights and thresholds are judgment calls, not fitted (`config.py`, each with a rationale; overridable with `--set`).
- **Checker limits:** it catches invented numbers, misattributed multiples, false specialist claims and copied boilerplate. It can't catch a true number used in a misleading argument, or *paraphrased* boilerplate (trigram similarity only).
- **No parent roll-up or portfolio data:** buyers under the same parent (e.g. CenterWell / Kindred at Home) aren't rolled up, and there's no portfolio-conflict or fund-capacity (dry powder) data.
- **Name-based inference is weak:** `--infer-acquirer-type` misses names with no cue (e.g. "Blackstone").
- **Thin data for small buyers:** strategics with 1–2 deals have thin evidence. Shrinkage and the floor gate limit, but can't fix, that.

## With more time

- An eval set with banker-graded rubrics on top of the deterministic metrics, to tune prompts against human judgment.
- Learning adjacency and weights from `feedback.json` exclusion reasons.
- Embedding similarity to catch paraphrased boilerplate.
- Parent-company roll-up, fund vintage and dry-powder data, and footprint maps for real capacity and antitrust analysis.

## Development

```bash
make install     # .venv + pip install -e ".[dev]"
make lint        # ruff check, ruff format --check, mypy
make test        # pytest with coverage (fails under 85%; currently 94%, 151 tests)
make run         # offline end-to-end run
make replay      # reproduce sample_output/
make submission  # build submission.zip (secret-scanned, < 5 MB)
```

- **Tests** use no API key and no network. Provider failure paths are tested with fake SDK objects. Fixtures for other CSV shapes live in `tests/fixtures/` (regenerate with `python tests/fixtures/make_fixtures.py`).
- **CI** (`.github/workflows/ci.yml`) runs lint, types, tests, an offline run, a replay diff against `sample_output/`, EDA on the fixtures, and pip-audit, on Python 3.11 and 3.12.
- **Exit codes:** 0 ok · 1 unexpected error or empty result · 2 usage · 3 data · 4 config · 5 LLM · 6 output. Errors are one line; `--debug` shows the traceback. `--log-json` gives structured logs (per-acquirer timing, tokens, cache hits; never prompts or keys).
- **Adding a checker rule:** write it in `validate.py`, call it from `check()`, state it in `prompts.SYSTEM`, bump `PROMPT_VERSION`, and add a passing and a failing test.
