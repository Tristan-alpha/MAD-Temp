"""Configuration constants for TG-MAD."""

# === Models ===
DEBATER_MODEL = "hosted_vllm/Qwen/Qwen3-4B-Instruct-2507"
EVALUATOR_MODEL = "hosted_vllm/Qwen/Qwen3-8B"
DEBATER_BASE_URL = "http://localhost:8000/v1"
EVALUATOR_BASE_URL = "http://localhost:8001/v1"

# === Debate ===
N_AGENTS = 3
N_ROUNDS = 3  # T=3
TEMPERATURE = 1.0  # Fixed per AGENTS.md
EVALUATOR_TEMPERATURE = TEMPERATURE  # Fixed per AGENTS.md
TOP_P = 0.9  # Match src/model/model_utils.py
MAX_NEW_TOKENS = 512  # Match src/main.py default and my_scripts GSM8K runs

# === Training ===
BATCH_SIZE = 5
NUM_EPOCHS = 2
TRAIN_SIZE = 10
SEED = 42

# === Data ===
EXISTING_DATA_PATH = "out/history/gsm8k/gsm8k_500__qwen3-4b_N=3_R=3.jsonl"
OUTPUT_DIR = "out/tg_mad/"
PROMPT_HISTORY_FILENAME = "prompt_history.json"
EVAL_RESULTS_FILENAME = "eval_results.json"
SPLIT_INFO_FILENAME = "split_info.json"
RUN_CONFIG_FILENAME = "run_config.json"
PROMPT_HISTORY_FILE = f"{OUTPUT_DIR}{PROMPT_HISTORY_FILENAME}"
EVAL_RESULTS_FILE = f"{OUTPUT_DIR}{EVAL_RESULTS_FILENAME}"
ARTIFACT_SCHEMA_VERSION = 1

# === Answer Format (matching existing codebase) ===
ANSWER_SUFFIX = ' Make sure to state your final answer in curly brackets at the very end of your response, just like: "{final answer: 123}".'
ANSWER_REGEX = r"\{(.*?)\}"

# === Initial Debater Prompt ===
INITIAL_DEBATER_PROMPT = """You are a mathematical reasoning agent participating in a multi-agent debate.
Solve the given math problem step by step. Show your work clearly.
Always conclude your response with your answer in curly brackets: "{final answer: <number>}".

During debate rounds:
- Carefully evaluate other agents' solutions for logical and arithmetic errors.
- If you find a specific error in another agent's reasoning, explain exactly what is wrong.
- Do NOT change your answer merely because other agents disagree. Only change if you are presented with a clear logical or mathematical proof that your reasoning contains an error.
- If your reasoning is correct, defend it with evidence."""

# === Optimizer Constraints ===
OPTIMIZER_CONSTRAINTS = [
    "The prompt must always instruct agents to end responses with their answer in curly brackets: '{final answer: <number>}'.",
    "The prompt must be a system prompt for math debate agents.",
    "The prompt must not exceed 500 words.",
]

# === Evaluator swap ===
# To switch to MiniMax M2.5:
# 1. Change EVALUATOR_MODEL to the appropriate litellm string
# 2. Set MINIMAX_API_KEY in environment
# 3. No other code changes needed
