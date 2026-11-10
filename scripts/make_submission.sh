#!/usr/bin/env bash
# Build submission.zip from a clean staged copy of the repo.
#   - excludes secrets (.env), local state and generated output
#   - includes sample_output/ and only the .cache/llm entries the committed sample needs,
#     so `python -m acquirer_engine --provider replay` reproduces it without any API key
#   - fails loudly if anything in the staged copy looks like an API key
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1   # the secret scan imports the package from the staging copy: no __pycache__ in the zip

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
case "$PY" in */*) PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")" ;; esac   # absolute: we cd later
if ! "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
  echo "ERROR: need Python >= 3.10 (set PYTHON=/path/to/python)" >&2
  exit 1
fi
MAX_BYTES=$((5 * 1024 * 1024))
STAGE_PARENT="$(mktemp -d)"
STAGE="$STAGE_PARENT/acquirer-engine"
trap 'rm -rf "$STAGE_PARENT"' EXIT

cd "$ROOT"
rsync -a ./ "$STAGE/" \
  --exclude '.env' --exclude '.venv/' --exclude '.omc/' --exclude 'out/' \
  --exclude '__pycache__/' --exclude '.pytest_cache/' --exclude '.mypy_cache/' --exclude '.ruff_cache/' \
  --exclude '.coverage' --exclude 'htmlcov/' --exclude '*.egg-info/' --exclude 'build/' --exclude 'dist/' --exclude 'wheelhouse/*.whl' --exclude 'interview_notes/' --exclude 'brag-output*/' \
  --exclude 'feedback*.json' --exclude '*.log' --exclude 'Agentic_Take_Home_Instructions*.docx' \
  --exclude '*.zip' --exclude '.git/' --exclude '.DS_Store' --exclude '.cache/'

# Copy only the cache entries the committed sample was built from (listed in its report.json).
mkdir -p "$STAGE/.cache/llm"
"$PY" - "$ROOT" "$STAGE" <<'PYEOF'
import json, shutil, sys
from pathlib import Path
root, stage = Path(sys.argv[1]), Path(sys.argv[2])
report = json.loads((root / "sample_output" / "report.json").read_text(encoding="utf-8"))
keys = [p["cache_key"] for p in report["pages"] if p.get("cache_key")]
if report["meta"].get("run_id"):
    keys.append("run-" + report["meta"]["run_id"])   # the live run's stamp, shown by replay
missing = [k for k in keys if not (root / ".cache" / "llm" / f"{k}.json").is_file()]
if missing:
    sys.exit(f"cache entries missing for sample_output: {missing[:3]} ... rerun the sample first")
for k in keys:
    shutil.copy2(root / ".cache" / "llm" / f"{k}.json", stage / ".cache" / "llm" / f"{k}.json")
print(f"included {len(keys)} cache entries for replay (pages + run stamp)")
PYEOF

# Gmail rejects any attachment that contains a .bat (even inside a zip): ship the Windows launcher as run.bat.txt.
mv "$STAGE/run.bat" "$STAGE/run.bat.txt"
BLOCKED=$(cd "$STAGE" && find . -type f | grep -iE '\.(ade|adp|apk|appx|bat|cab|chm|cmd|com|cpl|dll|dmg|exe|hta|img|ins|iso|isp|jar|jnlp|js|jse|lib|lnk|mde|mjs|msc|msi|msix|msp|mst|nsh|pif|ps1|scr|sct|shb|sys|vb|vbe|vbs|vhd|vxd|wsc|wsf|wsh|xll)$' || true)
if [ -n "$BLOCKED" ]; then
  echo "ERROR: file types Gmail blocks would be in the zip: $BLOCKED" >&2
  exit 1
fi

echo "replay check..."
CHECK_DIR="$(mktemp -d)"
if ! (cd "$STAGE" && "$PY" -B -m acquirer_engine --provider replay --out-dir "$CHECK_DIR" >/dev/null 2>&1); then
  echo "ERROR: replay failed in the staged copy - submission.zip NOT built" >&2
  exit 1
fi
for f in report.html report.md report.json eval.json; do
  if ! cmp -s "$CHECK_DIR"/*/"$f" "$ROOT/sample_output/$f"; then
    echo "ERROR: replay does not reproduce sample_output/$f (stale .cache/llm?) - submission.zip NOT built" >&2
    exit 1
  fi
done
rm -rf "$CHECK_DIR"
echo "replay reproduces sample_output/ byte for byte"

echo "secret scan..."
if ! (cd "$STAGE" && "$PY" -B -m acquirer_engine.secretscan .); then
  echo "ERROR: secret scan FAILED - submission.zip NOT built" >&2
  exit 1
fi

rm -f "$ROOT/submission.zip"
(cd "$STAGE_PARENT" && zip -qr "$ROOT/submission.zip" acquirer-engine)
SIZE=$(wc -c < "$ROOT/submission.zip" | tr -d ' ')
if [ "$SIZE" -gt "$MAX_BYTES" ]; then
  echo "ERROR: submission.zip is $SIZE bytes (> 5 MB)" >&2
  exit 1
fi
echo "OK: submission.zip ($SIZE bytes), secret scan clean"
