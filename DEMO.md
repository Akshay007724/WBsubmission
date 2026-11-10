# Demo: Running the M&A Acquirer Identification Engine

A step-by-step walkthrough, from a fresh clone to a banker-ready report. Steps 1–5 need no API key. Step 6 is the real LLM run and takes about 30 seconds.

Commands are for macOS/Linux. On Windows, use `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

---

## Step 0 — Prerequisites

| Need | Check |
|---|---|
| Python **3.10+** | `python3 --version` |
| The data file in the project root | `ls ma_transactions_500.csv` |
| An API key (step 6 only) | Anthropic recommended; OpenAI or Gemini also work |

---

## Step 1 — Set up the environment (about 1 minute)

```bash
cd acquirer-engine          # the folder unzipped from submission.zip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt      # runtime + test/lint tools (or: make install)
```

**Expected:** pip finishes without errors. Runtime packages: pandas, numpy, pydantic, jinja2, pyyaml, anthropic, openai. Dev tools: pytest, pytest-cov, ruff, mypy, pip-audit.

---

## Step 2 — Run the tests (about 1 second, no key)

```bash
pytest -q
```

**Expected:** `151 passed`, coverage 94% (on Python 3.10 one packaging check is skipped, because `tomllib` arrived in 3.11). The tests never read `.env` or call a paid API. They cover:
- data loading;
- scoring;
- the M&A domain rules;
- the checker that catches invented numbers and deal IDs;
- the repair loop;
- the EDA stage;
- the command-line interface end to end.

---

## Step 3 — Explore the data (stage 0, no key)

```bash
python -m acquirer_engine --eda-only
```

**Expected output:**
```
EDA: out/healthcare-services-200m-regional-private/eda.md
     out/healthcare-services-200m-regional-private/eda.json
```

Open `eda.md`. Section 2 lists 12 data-quality findings, each paired with how the engine responds. Points to call out in a demo:
- **EV doesn't reconcile** with EV/EBITDA × EBITDA on 403 of 500 rows. The engine uses the stated multiples and never recomputes them.
- **Only 16 Healthcare Services deals** fall in the $80–500M band, so the engine draws on adjacent sectors (126 deals).
- **The median sponsor has 18 deals, the median strategic 2.** The engine uses small-sample shrinkage and a buyer-mix floor.
- **"Geographic Expansion" appears on 90% of deals**, so tags are shown by lift against that base rate.

---

## Step 3b — Reproduce the committed sample exactly (no key)

```bash
python -m acquirer_engine --provider replay
```

Serves the validated answers shipped in `.cache/llm/`. The report is byte-identical to `sample_output/`.

## Step 4 — Dry run without an LLM (no key)

```bash
python -m acquirer_engine --provider offline
```

**Expected:**
```
Report: out/healthcare-services-200m-regional-private/report.html
...
Project Atlas (Healthcare Services target): 10 acquirers · {'verified': 10} · offline/offline-template · 0.1s
```

This runs the whole pipeline (data, EDA, ranking, evidence, checker, rendering), with deterministic template text in place of the LLM. Every page is labelled **"LLM disabled"**. It proves the pipeline works; it is not the product.

---

## Step 5 — Add your API key

```bash
cp .env.example .env
# then edit .env and fill in ONE of:
#   ANTHROPIC_API_KEY=sk-ant-...     (recommended: default model claude-sonnet-5-5)
#   OPENAI_API_KEY=sk-...
#   GEMINI_API_KEY=...
```

The engine reads `.env` automatically and uses the first key it finds, checking in the order Anthropic, OpenAI, Gemini. Variables already exported in your shell take precedence. `.env` is in `.gitignore`, so the key is never committed.

---

## Step 6 — The real run (about 30 seconds)

```bash
python -m acquirer_engine
```

**Expected (from a real run):**
```
Report: out/healthcare-services-200m-regional-private/report.html
        out/healthcare-services-200m-regional-private/report.md
        out/healthcare-services-200m-regional-private/report.json
Project Atlas (Healthcare Services target): 10 acquirers · {'verified': 8, 'repaired': 2} · anthropic/claude-sonnet-5-5 · 26.3s · cache hits 0 · est. cost $0.37
```

- `verified` means the page passed the checker on the first try.
- `repaired` means it failed a check, got the specific errors back, and passed on its one retry.
- `unverified` means it was still flagged after the retry. The page is shown with a badge and the issues listed, never hidden.

**Run it again.** It finishes in about 1 second with `cache hits 10` and produces the same output. Only outputs that passed the checker are cached. Use `--refresh` to force fresh calls.

---

## Step 7 — Walk through the report

```bash
open out/healthcare-services-200m-regional-private/report.html     # macOS; or double-click the file
```

| Section | What to point out |
|---|---|
| **Header and warnings** | The target profile, and assumptions stated up front (21.5% "strong margin" = the dataset's 75th percentile; region unspecified) |
| **Summary table** | Rank, buyer type, score, conviction (varied: 4 High / 3 Medium / 3 Low in the sample), a one-line headline, and checker status. `(mix)` marks a buyer included to meet the strategic/sponsor floor |
| **Valuation peer set** | Closed control deals only, split by buyer type, with a last-3-years subset; the widening steps are shown |
| **Each acquirer page** | The 6 required sections, each citing `[MA-YYYY-NNNN]` deal IDs. Also: **size positioning** (platform vs. add-on), **"Why ranked #k"** with the score breakdown, and the precedent table. Every fact on the page comes from the CSV |
| **Also considered** | Who just missed the cut and why. For example, sponsors that outscored the strategics added for the buyer-mix floor |
| **Data profile (stage 0)** | The EDA findings and the engine's responses |
| **Methodology and run details** | Model, prompt version, data sha256, timestamp, cache hits |

**Print to PDF:** Cmd/Ctrl+P. Each acquirer prints on its own page.

**Spot-check that it's grounded:** pick any cited ID and look it up.
```bash
grep "MA-2021-0025" ma_transactions_500.csv
```
The size and multiple in the CSV match what the page says. That is what the checker enforces.

---

## Step 8 — Stretch features

**Any target profile**, from flags or a YAML file:
```bash
python -m acquirer_engine --sector "Dental" --ev 150 --geography Southeast
#   → out/dental-150m-southeast-private/report.html
python -m acquirer_engine --target targets/behavioral_500m.yaml
#   → out/behavioral-health-500m-southeast-pe-backed/report.html
```
Typos get suggestions, e.g. `--sector "Helthcare"` → `Did you mean: ['Healthcare Services', ...]`.

**Banker feedback** (flag a buyer as irrelevant; the flag is remembered for that target):
```bash
python -m acquirer_engine --exclude "Blackstone" --reason "conflicted on a portfolio company"
python -m acquirer_engine                      # Blackstone stays excluded; the next-best buyer takes its slot
python -m acquirer_engine --clear-feedback     # undo
```

**Two targets side by side:**
```bash
python -m acquirer_engine --compare targets/default.yaml targets/behavioral_500m.yaml
#   → out/compare.html  (overlapping buyers highlighted, with rank and conviction changes)
```

**Other options:**
```bash
python -m acquirer_engine --provider openai --model gpt-4o
python -m acquirer_engine --provider gemini              # free-tier keys allow ~20 requests/model/day
python -m acquirer_engine --model claude-haiku-5-5       # cheaper, faster, less detailed
python -m acquirer_engine -v                             # log every API call and retry
```

---

## Step 9 (optional) — Local, no-cloud models

The engine also works with a local server that speaks the OpenAI API (Docker Model Runner or Ollama). Nothing leaves the machine.

```bash
# Docker Model Runner
docker desktop enable model-runner --tcp=12434
docker model pull ai/llama3.2
docker model configure --context-size 16384 ai/llama3.2      # the default 4k context is too small
python -m acquirer_engine --provider local

# Ollama (set the context when starting the server)
OLLAMA_CONTEXT_LENGTH=16384 ollama serve
LOCAL_LLM_BASE_URL=http://localhost:11434/v1 python -m acquirer_engine --provider local --model <model>
```

Set expectations: a 3B model ran the pipeline end to end in about 7 minutes, but only 1 of 10 pages passed the checker, and the prose was generic. The checker labelled every weak page correctly. Use a cloud model for banker-quality output.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `error: ANTHROPIC_API_KEY is not set` | Add the key to `.env` (step 5), or run with `--provider offline` |
| `error: Unknown sector '...'. Did you mean ...` | Use one of the 10 sectors listed in the message |
| Page marked `unverified` | Read the "Checker notes" on that page. Usually a number or deal ID that isn't in the evidence |
| `Gemini free-tier daily quota exhausted` | The free tier allows 20 requests per model per day. Wait, use `--model` for another model, or enable billing |
| Local model: `exceeds the available context size` | Raise the context to 16k (step 9) |
| Want fresh LLM output | `--refresh` (or delete `.cache/`) |

---

## Demo video outline (about 6 minutes)

| Time | Show | Point to make |
|---|---|---|
| 0:00 | Summary table in `report.html` | The ask (10 buyers, $200M Healthcare Services target) and the design: code ranks, the LLM writes, a checker verifies, and every fact comes from the CSV |
| 0:30 | `pytest -q` | 151 tests, 94% coverage, no API key needed |
| 0:50 | `python -m acquirer_engine --eda-only` | Data findings and the engine's response to each: thin sector band (16 deals), 403 inconsistent EV rows, sponsor/strategic activity gap (18 vs 2) |
| 1:40 | `python -m acquirer_engine --refresh` | Live run in about 30 s; verified/repaired status, cost |
| 2:20 | Report: header, Market context, Welsh Carson page | Rank vs conviction, two valuation benchmarks, size role, closest competitor, data-backed risks |
| 3:20 | `grep "MA-2021-0083" ma_transactions_500.csv` | A cited deal matches the CSV exactly |
| 3:35 | Blackstone and Ardent size roles; "Also considered" | Add-on vs stretch (capacity from closed deals only); mix-floor swaps shown with scores |
| 4:00 | `--provider offline --data tests/fixtures/tech_billions.csv --sector "Enterprise Software"` | Any CSV: header mapping, unit conversion, derived adjacency |
| 4:40 | `--exclude "Blackstone" --reason "conflicted"` | Banker feedback re-ranks and persists; `--compare` for two targets |
| 5:15 | `./run.sh` | Reviewer path: no key, byte-identical replay; offline mode, Docker |
| 5:40 | README limitations | Synthetic data, CSV-only facts, hand-set weights; next steps |
