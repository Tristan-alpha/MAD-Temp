#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install textgrad==0.1.8

python - <<'PY'
import textgrad
print(f"textgrad import OK: {textgrad.__version__}")
PY

echo "If you plan to use --temp_selector textgrad, set a provider API key first (for example OPENAI_API_KEY)."
