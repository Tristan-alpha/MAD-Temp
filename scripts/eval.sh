#!/bin/bash

# Simple hh_rlhf TG-MAD evaluation entrypoint.
# Edit the experiment block below; most runs should not need to touch the
# serving/workstation section at the bottom.

#SBATCH -p PA100q
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -o job-hh-rlhf-v2-eval-%j.output
#SBATCH --time=24:00:00
#SBATCH -J hh-rlhf-v2-eval

set -euo pipefail

REPO_ROOT="/export/home3/dazhou/debate-or-vote"
PYTHON_BIN="python"
cd "${REPO_ROOT}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

# =========================
# Experiment settings
# Change these lines first.
# Prefer the stage-specific names below. Legacy aliases such as MODEL,
# DATA_SIZE, NUM_AGENTS, DEBATE_ROUNDS, EXISTING_DATA, and SEED are still
# accepted.
# =========================
RUN_NAME="hh_rlhf_v2_eval"

# Experiment identity
DATASET="hh_rlhf"
EVAL_SEED="42"

# Models
DEBATER_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

# Evaluation inputs
PROMPT_HISTORY_PATH=""
EVAL_EXISTING_DATA="/export/home3/dazhou/debate-or-vote/out/history/hh_rlhf/hh_rlhf_60__Qwen_Qwen2.5-7B-Instruct_N=5_R=2.jsonl"
TRAIN_EXISTING_DATA=""
MAX_TEST_SAMPLES="60"
PROMPT_INDEX=""

# Debate topology
EVAL_N_AGENTS="5"
EVAL_N_ROUNDS="2"

# Generation budget
MAX_NEW_TOKENS="2048"

# Outputs
OUTPUT_DIR="${REPO_ROOT}/out/${RUN_NAME}_${RUN_STAMP}"
TEXT_HISTORY_DIR=""

# =========================
# Serving defaults
# Leave these alone unless the cluster setup changes.
# =========================
SERVING_PROFILE=""
START_DEBATER_SERVER="1"
SAVE_TEXT_HISTORY="1"
ALLOW_FAILED_GENERATIONS="0"

# Optional advanced overrides. Uncomment only when debugging local serving.
# export DEBATER_MAX_MODEL_LEN=16384
# export DEBATER_CUDA_VISIBLE_DEVICES=0

append_arg_if_set() {
    local flag="$1"
    local value="${2:-}"
    if [[ -n "${value}" ]]; then
        cmd+=("${flag}" "${value}")
    fi
}

append_flag_if_enabled() {
    local flag="$1"
    local enabled="${2:-0}"
    if [[ "${enabled}" == "1" ]]; then
        cmd+=("${flag}")
    fi
}

if [[ -z "${PROMPT_HISTORY_PATH}" ]]; then
    echo "PROMPT_HISTORY_PATH is required for eval.sh. Point it at a train output prompt_history.json." >&2
    exit 1
fi

# Avoid inheriting train-time GPU free-memory thresholds (for example,
# DEBATER_MIN_FREE_MIB=12000) that can make eval auto-pick fail on shared nodes.
# Set TG_MAD_PRESERVE_DEBATER_MIN_FREE_MIB=1 to keep the inherited value.
if [[ "${TG_MAD_PRESERVE_DEBATER_MIN_FREE_MIB:-0}" != "1" ]]; then
    unset DEBATER_MIN_FREE_MIB
fi

export REPO_ROOT
export START_DEBATER_SERVER

cmd=(
    "${PYTHON_BIN}" -m tg_mad.job_runner eval
    --dataset "${DATASET}"
    --model "${DEBATER_MODEL_NAME}"
    --prompt-history "${PROMPT_HISTORY_PATH}"
    --max-test-samples "${MAX_TEST_SAMPLES}"
    --num_agents "${EVAL_N_AGENTS}"
    --debate_rounds "${EVAL_N_ROUNDS}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --seed "${EVAL_SEED}"
    --output-dir "${OUTPUT_DIR}"
)

append_arg_if_set --serving-profile "${SERVING_PROFILE}"
append_arg_if_set --eval-existing-data "${EVAL_EXISTING_DATA}"
append_arg_if_set --train-existing-data "${TRAIN_EXISTING_DATA}"
append_arg_if_set --prompt-index "${PROMPT_INDEX}"
append_arg_if_set --text-history-dir "${TEXT_HISTORY_DIR}"
append_flag_if_enabled --allow-failed-generations "${ALLOW_FAILED_GENERATIONS}"

if [[ "${SAVE_TEXT_HISTORY}" == "0" ]]; then
    cmd+=(--no-save-text-history)
fi

exec "${cmd[@]}" "$@"
