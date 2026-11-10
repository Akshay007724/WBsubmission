# Start here

**M&A Acquirer Identification Engine.** It ranks the 10 most likely buyers for a ~$200M Healthcare Services target and writes a data-cited one-page rationale for each. Every fact comes from `ma_transactions_500.csv`.

## See the result right away (no install)

Open **`sample_output/report.html`** in a browser. It's a real run on Claude Sonnet 5.5.

## Run it yourself (about 1 minute, no API key needed)

You need **Python 3.10 or newer** ([python.org/downloads](https://www.python.org/downloads/)).

| | Command |
|---|---|
| **Mac / Linux** | open a terminal in this folder and run `./run.sh` |
| **Windows** | rename `run.bat.txt` to `run.bat`, then double-click it (or run it from a terminal in this folder) |

The first run creates a local virtual environment (`.venv`) and installs the dependencies. It then rebuilds the sample report from the validated answers shipped in `.cache/llm/`, with no API key and no LLM calls, and opens `out/healthcare-services-200m-regional-private/report.html`. The output is identical to `sample_output/`.

Other modes:

| Mode | Mac / Linux | Windows | Needs a key? |
|---|---|---|---|
| Reproduce the sample (default) | `./run.sh` | `run.bat` | no |
| Full pipeline with template text | `./run.sh offline` | `run.bat offline` | no |
| Real LLM run (~30 s) | `./run.sh live` | `run.bat live` | yes: put `ANTHROPIC_API_KEY=...` in a file named `.env` (see `.env.example`) |

**Why `run.bat.txt`?** Gmail blocks attachments that contain a `.bat` file, so the launcher ships with a `.txt` extension. Prefer not to rename it? From a terminal in this folder run:
`py -3 -m venv .venv && .venv\Scripts\python -m pip install -r requirements.txt && .venv\Scripts\python -m acquirer_engine --provider replay`

Docker instead of Python: `make wheelhouse && make docker` (see README).

## What to read

- **`sample_output/report.html`**: the deliverable. Summary table, one page per acquirer, market context, data profile, evaluation.
- **`README.md`**: how it works, prompt design, assumptions, limitations.
- **`DEMO.md`**: a step-by-step walkthrough of every feature.
- **`CHANGELOG.md`**: what the release contains, the sample run's numbers, and known limitations.
- **`docs/EDA.md`**: the data profile of the CSV, with each finding and how the engine responds to it.
