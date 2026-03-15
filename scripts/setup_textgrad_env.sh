#!/usr/bin/env bash
set -euo pipefail

INSTALL_VLLM="${INSTALL_VLLM:-0}"

python -m pip install --upgrade pip
python -m pip install textgrad==0.1.8

if [[ "${INSTALL_VLLM}" == "1" ]]; then
    python -m pip install vllm
fi

python - <<'PY'
import textgrad
print(f"textgrad import OK: {textgrad.__version__}")
PY

if [[ "${INSTALL_VLLM}" == "1" ]]; then
python - <<'PY'
import vllm
print("vllm import OK")
PY
fi

echo "Set INSTALL_VLLM=1 when preparing the TG-MAD vLLM workflow."
echo "Set a provider API key first (for example OPENAI_API_KEY) before calling remote TextGrad engines."
