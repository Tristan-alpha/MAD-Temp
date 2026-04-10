#!/bin/bash

# Simple hh_rlhf TG-MAD training entrypoint.
# Edit the experiment block below; most runs should not need to touch the
# serving/workstation section at the bottom.

#SBATCH -p RTXA6Kq
#SBATCH -w node09
#SBATCH --gres=gpu:3
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --exclude=node06
#SBATCH -o mmlu-logic-%j.output
#SBATCH --time=24:00:00
#SBATCH -J mmlu_logic

set -euo pipefail

REPO_ROOT="/export/home3/dazhou/debate-or-vote"
PYTHON_BIN="python"
cd "${REPO_ROOT}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

# =========================
# Experiment settings
# Change these lines first.
# Prefer the stage-specific names below. Legacy aliases such as MODEL,
# DATA_SIZE, NUM_AGENTS, DEBATE_ROUNDS, and SEED are still accepted.
# =========================
RUN_NAME="formal_logic"

# Experiment identity
DATASET="formal_logic"
TRAIN_SEED="42"

# Models
DEBATER_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
EVALUATOR_MODEL_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507"

# Data
TRAIN_SIZE="19"
TRAIN_EXISTING_DATA="out/history/formal_logic/formal_logic_19__Qwen_Qwen2.5-7B-Instruct_N=5_R=2_repro_20260328.jsonl"

# Debate topology
TRAIN_N_AGENTS="5"
TRAIN_N_ROUNDS="2"

# Optimization
TRAIN_BATCH_SIZE="1"
TRAIN_NUM_EPOCHS="1"

# Generation budgets
MAX_NEW_TOKENS="2048"
EVALUATOR_MAX_NEW_TOKENS="2048"

# Outputs
OUTPUT_DIR="${REPO_ROOT}/out/${RUN_NAME}_${RUN_STAMP}"
PROMPT_HISTORY_PATH=""
TEXT_HISTORY_DIR=""

# =========================
# Serving defaults
# Leave these alone unless the cluster setup changes.
# =========================
SERVING_PROFILE="local_30b_dual_gpu"
START_DEBATER_SERVER="1"
START_EVALUATOR_SERVER="1"
SAVE_TEXT_HISTORY="0"
ALLOW_FAILED_GENERATIONS="0"

# Optional advanced overrides. Uncomment only when debugging local serving.
# export DEBATER_MAX_MODEL_LEN=16384
# export EVALUATOR_MAX_MODEL_LEN=24576
# export DEBATER_CUDA_VISIBLE_DEVICES=0
# export EVALUATOR_CUDA_VISIBLE_DEVICES=1,2

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

export REPO_ROOT
export START_DEBATER_SERVER
export START_EVALUATOR_SERVER

cmd=(
    "${PYTHON_BIN}" -m tg_mad.job_runner train
    --dataset "${DATASET}"
    --model "${DEBATER_MODEL_NAME}"
    --evaluator-model "${EVALUATOR_MODEL_NAME}"
    --train-size "${TRAIN_SIZE}"
    --batch-size "${TRAIN_BATCH_SIZE}"
    --num-epochs "${TRAIN_NUM_EPOCHS}"
    --num_agents "${TRAIN_N_AGENTS}"
    --debate_rounds "${TRAIN_N_ROUNDS}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --evaluator-max-new-tokens "${EVALUATOR_MAX_NEW_TOKENS}"
    --seed "${TRAIN_SEED}"
    --output-dir "${OUTPUT_DIR}"
    --train-existing-data "${TRAIN_EXISTING_DATA}"
    --serving-profile "${SERVING_PROFILE}"
)

append_arg_if_set --prompt-history "${PROMPT_HISTORY_PATH}"
append_arg_if_set --text-history-dir "${TEXT_HISTORY_DIR}"
append_flag_if_enabled --allow-failed-generations "${ALLOW_FAILED_GENERATIONS}"

if [[ "${SAVE_TEXT_HISTORY}" == "0" ]]; then
    cmd+=(--no-save-text-history)
fi

exec "${cmd[@]}" "$@"
