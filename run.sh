#!/usr/bin/env bash
# One-command run for reviewers (macOS / Linux).
#   ./run.sh            reproduce the committed sample report, no API key needed (default)
#   ./run.sh offline    run the full pipeline with template text, no API key needed
#   ./run.sh live       real LLM run (needs ANTHROPIC_API_KEY, OPENAI_API_KEY or GEMINI_API_KEY in .env or the environment)
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-replay}"

# Find Python 3.10+ (macOS often ships 3.9 as python3).
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "This project needs Python 3.10 or newer. Install it from https://www.python.org/downloads/ and re-run ./run.sh" >&2
  exit 1
fi
echo "Using $($PY --version) ($(command -v "$PY"))"

if [ ! -x .venv/bin/python ]; then
  echo "Creating a virtual environment and installing dependencies (first run only, ~1 minute)..."
  "$PY" -m venv .venv
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -q -r requirements.txt
fi

case "$MODE" in
  replay)  ARGS=(--provider replay) ;;
  offline) ARGS=(--provider offline) ;;
  live)    ARGS=() ;;
  *) echo "usage: ./run.sh [replay|offline|live]" >&2; exit 2 ;;
esac

.venv/bin/python -m acquirer_engine ${ARGS[@]+"${ARGS[@]}"}   # bash 3.2-safe empty-array expansion
REPORT="out/healthcare-services-200m-regional-private/report.html"
echo
echo "Report: $REPORT"
if [ -n "${NO_OPEN:-}" ]; then :; elif command -v open >/dev/null 2>&1; then open "$REPORT"; elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$REPORT" >/dev/null 2>&1 || true; fi
