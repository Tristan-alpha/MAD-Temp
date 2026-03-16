# TG-MAD Execution Plan: Detailed Implementation Specification for Coding Agent

## 0. Project Context & Goal

Build a **TextGrad-Optimized Multi-Agent Debate (TG-MAD)** system that uses TextGrad to optimize a shared debater system prompt, proving that TG-MAD consistently outperforms Majority Voting (MV) by addressing two theoretical flaws:

1. **Echo Chamber Effect** — a correct minority agent gets subverted by an incorrect majority.
2. **Martingale Property** — standard debate fails to improve expected accuracy across rounds.

---

## 1. Environment & Dependencies

### 1.1 Models (served via vLLM with OpenAI-compatible API)

| Role | Model | Notes |
|------|-------|-------|
| **Debater agents (×3)** | `Qwen3-4B-Instruct-2507` | Served locally via vLLM |
| **Evaluator / backward engine** | `Qwen3-8B` (initial) | Served locally via vLLM. Will be swapped to MiniMax M2.5 (external API) later. Code must abstract the evaluator engine so swapping is a config change. |

### 1.2 Python Dependencies

```
textgrad          # pip install textgrad
litellm           # for routing to vLLM OpenAI-compatible endpoints
```

### 1.3 TextGrad Engine Configuration

TextGrad supports litellm via experimental engines. Configure as follows:

```python
import textgrad as tg

# Debater engine — Qwen3-4B via vLLM
debater_engine = tg.get_engine(
    "experimental:hosted_vllm/Qwen3-4B-Instruct-2507",
    cache=False
)
# Set OPENAI_API_BASE or pass api_base to point to local vLLM server

# Backward/evaluator engine — Qwen3-8B via vLLM
tg.set_backward_engine(
    "experimental:hosted_vllm/Qwen3-8B",
    cache=False
)
```

**IMPORTANT**: The coding agent must verify the exact litellm model string required for local vLLM. The pattern is `hosted_vllm/<model_name>` with `api_base` pointing to the vLLM server URL (e.g., `http://localhost:8000`). If TextGrad's experimental engine doesn't work cleanly with vLLM, fall back to setting `OPENAI_API_BASE` and `OPENAI_API_KEY=EMPTY` environment variables and using `openai/<model_name>` as the engine string.

### 1.4 Existing Data

The file `out/history/gsm8k/gsm8k_500__qwen3-4b_N=3_R=3.jsonl` contains pre-run MAD results. **The coding agent must read and inspect this file first** to understand its schema (fields like question, agent answers per round, ground truth, etc.). This data can be reused to avoid redundant API calls during development. Specifically, the `t=0` independent answers from this file can serve as the frozen MV baseline for evaluation.

### 1.5 Dataset

- **Dataset**: GSM8K (grade school math, `question` and `answer` fields, final numeric answer after `####`).
- **Training set**: 10 samples for initial optimization. Select samples where MV at `t=0` disagrees (i.e., not all 3 agents agree) to maximize gradient signal. If fewer than 10 disagreement samples exist in the first 50 of the JSONL, use the first 10 regardless.
- **Test set**: Remaining samples from the 500-sample JSONL file (after excluding training samples).

---

## 2. System Architecture

### 2.1 Variable Definition

```python
debater_prompt = tg.Variable(
    value=INITIAL_DEBATER_PROMPT,  # see Section 2.2
    requires_grad=True,
    role_description="shared system prompt for all 3 debater agents in a multi-agent debate on math problems"
)
```

This is the **single globally shared prompt** optimized by TextGrad. All 3 agents use the same prompt.

### 2.2 Initial Debater Prompt

```
You are a mathematical reasoning agent participating in a multi-agent debate.
Solve the given math problem step by step. Show your work clearly.
Always conclude your response with: Answer: <number>

During debate rounds:
- Carefully evaluate other agents' solutions for logical and arithmetic errors.
- If you find a specific error in another agent's reasoning, explain exactly what is wrong.
- Do NOT change your answer merely because other agents disagree. Only change if you are presented with a clear logical or mathematical proof that your reasoning contains an error.
- If your reasoning is correct, defend it with evidence.
```

### 2.3 Forward Pass: `mad_forward_pass(question, debater_prompt, T=3)`

**Parameters:**
- `question`: the GSM8K question string
- `debater_prompt`: the `tg.Variable` being optimized
- `T=3`: number of debate rounds
- `N=3`: number of agents (fixed)

**Execution flow:**

```
Round t=0 (Independent):
  For each agent i in {1, 2, 3}:
    answer_i_t0 = LLM(system=debater_prompt.value, user=question)
    Parse numeric answer from answer_i_t0

Round t=1..T (Debate):
  For each round t:
    For each agent i:
      context = format_debate_context(question, all_answers_from_previous_round)
      answer_i_t = LLM(system=debater_prompt.value, user=context)
      Parse numeric answer from answer_i_t
```

**CRITICAL LOGGING — the forward pass MUST record and return:**

```python
{
    "question": str,
    "ground_truth": str,  # numeric answer
    "t0_answers": [str, str, str],  # raw responses at t=0
    "t0_parsed": [number, number, number],  # parsed numeric answers at t=0
    "t0_majority_vote": number,  # majority vote of t0_parsed
    "rounds": {
        0: {"answers": [...], "parsed": [...]},
        1: {"answers": [...], "parsed": [...]},
        2: {"answers": [...], "parsed": [...]},
        3: {"answers": [...], "parsed": [...]}
    },
    "final_majority_vote": number,  # majority vote at t=T
    "full_transcript": str  # concatenated debate for evaluator
}
```

**Debate context format** (for rounds t>=1):

```
Question: {question}

Here are the solutions from all agents in the previous round:

Agent 1's solution:
{agent_1_previous_answer}

Agent 2's solution:
{agent_2_previous_answer}

Agent 3's solution:
{agent_3_previous_answer}

Now provide your updated solution. Carefully check each agent's reasoning for errors. If you find errors, explain them. If your previous answer was correct, defend it.
```

**Consensus mechanism**: Final majority vote at round `t=T`. Take the most common parsed numeric answer among the 3 agents. If all 3 disagree (no majority), use Agent 1's answer as tiebreaker.

**Implementation detail — using BlackboxLLM vs raw calls**: The forward pass should use `tg.BlackboxLLM` with the `debater_prompt` as `system_prompt` so that TextGrad can build the computation graph. Each agent call is a `BlackboxLLM` call. The forward pass produces a `tg.Variable` containing the full transcript as its value.

```python
debater_model = tg.BlackboxLLM(debater_engine, system_prompt=debater_prompt)

# At t=0, call debater_model(question_var) for each agent
# At t=1..T, call debater_model(context_var) for each agent
# Collect all outputs
```

**IMPORTANT**: All 3 agents share the same `debater_prompt` Variable. They produce different outputs due to LLM sampling stochasticity (set `temperature=0.7` for diversity). The `debater_prompt` is the same at `t=0` and during debate rounds — there is no separate pre/post prompt. The prompt is updated only between optimization steps (between training samples), never mid-debate.



---

## 3. Custom TextLoss Evaluator

### 3.1 Evaluator Design

Create a custom loss function that wraps `tg.TextLoss`. The evaluator LLM receives the ground truth, `t=0` independent answers, and the full debate transcript, then generates a textual gradient targeting the `debater_prompt`.

```python
def create_evaluator_loss(ground_truth: str, t0_answers: list, t0_majority: str):
    """Create a TextLoss that evaluates the debate transcript against ground truth."""

    evaluator_prompt = f"""You are evaluating a multi-agent math debate system. Your job is to generate feedback on the SYSTEM PROMPT used by the debater agents, so it can be improved.

GROUND TRUTH ANSWER: {ground_truth}

INDEPENDENT ANSWERS AT t=0 (before debate):
Agent 1: {t0_answers[0]}
Agent 2: {t0_answers[1]}
Agent 3: {t0_answers[2]}
Majority Vote at t=0: {t0_majority}

You will receive the full debate transcript. Analyze it and provide feedback on the debater system prompt.

YOUR ANALYSIS MUST CHECK FOR:

1. ECHO CHAMBER EFFECT: Did any agent have the CORRECT answer at t=0 but then change to an INCORRECT answer during debate because the majority pressured them? If so, generate a HARSH critique: the system prompt must be updated to forbid changing answers based on peer consensus alone — require explicit mathematical/logical proof before conceding.

2. MARTINGALE STAGNATION: Did the debate rounds fail to improve accuracy? Did agents just repeat their positions without meaningful engagement? If so, critique the prompt for failing to encourage productive mathematical reasoning exchange.

3. SUCCESSFUL CORRECTION (POSITIVE FEEDBACK): Did the debate successfully correct an initially wrong majority? Did a minority agent with the correct answer successfully convince others through rigorous reasoning? If so, generate POSITIVE feedback: identify what aspects of the prompt encouraged this good behavior and recommend reinforcing them.

4. SUCCESSFUL DEFENSE (POSITIVE FEEDBACK): Did an agent with the correct answer successfully resist pressure from an incorrect majority? If so, generate positive feedback praising the prompt's encouragement of evidence-based reasoning.

Be specific. Reference exact moments in the transcript. Your feedback will be used to update the system prompt via gradient descent."""

    return tg.TextLoss(evaluator_prompt)
```

### 3.2 Loss Computation (Scalar + Textual)

The loss function is applied per sample. It works in two stages:

**Stage 1 — Scalar signal** (for logging/metrics):
```python
scalar_loss = 0.0 if final_answer == ground_truth else 1.0
```

**Stage 2 — Textual gradient** (via TextLoss): The evaluator LLM generates the textual critique regardless of whether the answer is correct or not (because we want positive gradients too). The TextLoss evaluates the full transcript Variable and produces gradients.

```python
# In the training loop:
transcript_var = tg.Variable(
    value=full_transcript_string,
    requires_grad=False,
    role_description="full debate transcript including t=0 answers and all rounds"
)

loss_fn = create_evaluator_loss(ground_truth, t0_parsed, t0_majority)
loss = loss_fn(transcript_var)  # returns a Variable with textual evaluation

# The loss.backward() will propagate gradients back through the computation graph
# to debater_prompt since debater_prompt was used in BlackboxLLM calls
```

**IMPORTANT — Positive gradients**: Unlike the original plan that set loss=0 for correct answers, we DO generate gradients for correct answers too. The evaluator prompt includes instructions to produce positive feedback when the debate works well (successful correction, successful defense). This reinforces good prompt behaviors.

---

## 4. Training Loop

### 4.1 Batch Optimization with `tg.sum`

TextGrad natively supports batch gradient accumulation. The mechanism:

1. Run forward pass on each sample in the mini-batch.
2. Compute individual loss for each sample.
3. Use `tg.sum(losses)` to aggregate — this concatenates textual gradients from all samples.
4. Call `total_loss.backward()` — gradients propagate back to `debater_prompt`.
5. Call `optimizer.step()` — the optimizer LLM reads all concatenated gradients and produces a single updated prompt.

```python
# Configuration
BATCH_SIZE = 5
NUM_EPOCHS = 2  # 10 samples / batch_size=5 = 2 steps per epoch; run 2 epochs = 4 total steps
T = 3  # debate rounds
N = 3  # agents

# Setup
debater_prompt = tg.Variable(
    value=INITIAL_DEBATER_PROMPT,
    requires_grad=True,
    role_description="shared system prompt for all 3 debater agents in a multi-agent debate on math problems"
)

optimizer = tg.TGD(parameters=[debater_prompt])

# Training data: 10 GSM8K samples
train_data = load_train_data()  # list of {"question": str, "ground_truth": str}

for epoch in range(NUM_EPOCHS):
    # Shuffle training data each epoch
    random.shuffle(train_data)

    for batch_start in range(0, len(train_data), BATCH_SIZE):
        batch = train_data[batch_start:batch_start + BATCH_SIZE]
        losses = []

        for sample in batch:
            # Forward pass — runs full T-round debate
            result = mad_forward_pass(
                question=sample["question"],
                debater_prompt=debater_prompt,
                T=T, N=N
            )

            # Create per-sample loss
            loss_fn = create_evaluator_loss(
                ground_truth=sample["ground_truth"],
                t0_answers=result["t0_parsed"],
                t0_majority=result["t0_majority_vote"]
            )
            loss = loss_fn(result["transcript_variable"])
            losses.append(loss)

            # Log scalar metrics
            log_sample_result(result, sample["ground_truth"])

        # Aggregate and step
        total_loss = tg.sum(losses)
        total_loss.backward()
        optimizer.step()

        # Log prompt evolution
        print(f"[Epoch {epoch}, Batch {batch_start//BATCH_SIZE}] Updated prompt:")
        print(debater_prompt.value[:200] + "...")

        # Zero gradients for next batch
        optimizer.zero_grad()
```

### 4.2 Computation Graph Structure

The computation graph that TextGrad builds per sample:

```
debater_prompt (requires_grad=True)
    |
    v
BlackboxLLM(agent1_t0) --> response_a1_t0
BlackboxLLM(agent2_t0) --> response_a2_t0
BlackboxLLM(agent3_t0) --> response_a3_t0
    |
    v  (responses assembled into debate context)
BlackboxLLM(agent1_t1) --> response_a1_t1
... (T rounds × N agents)
    |
    v  (full transcript assembled)
TextLoss(evaluator) --> loss_i
```

All `BlackboxLLM` calls use the same `debater_prompt` as system_prompt, so `loss.backward()` accumulates gradients onto `debater_prompt`.

### 4.3 Training Data Selection Strategy

From the 500-sample JSONL:

1. **Read the existing data** to identify samples where agents disagreed at `t=0` (not all 3 had the same answer). These are the most informative for optimization.
2. **Select 10 such samples** as the training set. If fewer than 10 disagreement samples exist in the file, supplement with random samples.
3. **Reserve the rest** (490 samples) as the test set.
4. **Record the indices/IDs** of training samples so they are excluded from evaluation.

### 4.4 Prompt Versioning

Save every version of the prompt after each `optimizer.step()`:

```python
prompt_history = []
# After each step:
prompt_history.append({
    "epoch": epoch,
    "batch": batch_idx,
    "prompt": debater_prompt.value,
    "train_batch_accuracy": batch_accuracy
})
# Save to out/tg_mad/prompt_history.json
```

---

## 5. Evaluation Script

### 5.1 What to Evaluate

Run evaluation on the **test set** using:
1. **The optimized prompt** (from the final `optimizer.step()`).
2. **The initial unoptimized prompt** (frozen, for the standard MAD baseline).
3. **The `t=0` answers from the existing JSONL** (for the frozen MV baseline — these used whatever prompt was originally configured, before any optimization).

### 5.2 Required Metrics

For each of the following, compute on the test set:

#### Metric 0: Single Agent Accuracy

- Use the first agent's answer in the fisrt round to compute the accuracy.

#### Metric 1: Majority Voting (MV) Accuracy
- Use `t=0` answers from the existing JSONL data (frozen baseline, unoptimized prompt).
- Compute majority vote across 3 agents per question.
- `MV_accuracy = correct_majority_votes / total_questions`

#### Metric 2: Standard MAD Accuracy (unoptimized)
- Use the full debate results from the existing JSONL (T=3 rounds, unoptimized prompt).
- `Standard_MAD_accuracy = correct_final_votes / total_questions`
- This is the "standard MAD" that should show the Martingale property.

#### Metric 3: TG-MAD Accuracy (optimized)
- Run new debates on the test set using the optimized `debater_prompt`.
- `TGMAD_accuracy = correct_final_votes / total_questions`

#### Metric 4: Round-by-Round Mean Accuracy
For BOTH standard MAD and TG-MAD, compute:

```python
for t in [0, 1, 2, 3]:
    # Mean accuracy = fraction of individual agent answers that are correct
    correct_agents = sum(1 for agent_answer in all_agent_answers_at_t if agent_answer == ground_truth)
    total_agents = N * num_questions
    mean_accuracy_t = correct_agents / total_agents
```

**Expected result**: Standard MAD shows a flat line (Martingale). TG-MAD shows monotonically increasing accuracy across rounds.

#### Metric 5: Subversion vs. Correction Rate

For each question in the test set:

| Event | Condition |
|-------|-----------|
| **Correction** | MV at `t=0` was WRONG, but final consensus at `t=T` is CORRECT |
| **Subversion** (Echo Chamber) | MV at `t=0` was CORRECT, but final consensus at `t=T` is WRONG |
| **Maintained Correct** | MV at `t=0` was CORRECT and final consensus is also CORRECT |
| **Maintained Wrong** | MV at `t=0` was WRONG and final consensus is also WRONG |

Compute these for BOTH standard MAD and TG-MAD. The key claim is that TG-MAD has a higher correction rate and lower subversion rate.

### 5.3 Output Format

Save to `out/tg_mad/eval_results.json`:

```json
{
    "mv_accuracy": 0.XX,
    "standard_mad_accuracy": 0.XX,
    "tgmad_accuracy": 0.XX,
    "round_by_round": {
        "standard_mad": {"t0": 0.XX, "t1": 0.XX, "t2": 0.XX, "t3": 0.XX},
        "tgmad": {"t0": 0.XX, "t1": 0.XX, "t2": 0.XX, "t3": 0.XX}
    },
    "correction_rate": {"standard_mad": 0.XX, "tgmad": 0.XX},
    "subversion_rate": {"standard_mad": 0.XX, "tgmad": 0.XX},
    "maintained_correct_rate": {"standard_mad": 0.XX, "tgmad": 0.XX},
    "maintained_wrong_rate": {"standard_mad": 0.XX, "tgmad": 0.XX},
    "num_test_samples": N,
    "optimized_prompt": "...",
    "initial_prompt": "..."
}
```

Also generate matplotlib plots saved as PNG:

1. **`round_accuracy_comparison.png`**: Line plot. X-axis = round (0,1,2,3). Y-axis = mean agent accuracy. Two lines: Standard MAD (should be flat) vs TG-MAD (should increase).
2. **`subversion_correction_bar.png`**: Grouped bar chart. X-axis = {Correction, Subversion}. Y-axis = rate. Two bars per group: Standard MAD vs TG-MAD.
3. **`overall_accuracy_bar.png`**: Bar chart comparing MV, Standard MAD, and TG-MAD accuracy.

---

## 6. File Structure

```
tg_mad/
├── config.py              # All hyperparameters, model endpoints, paths
├── data_loader.py         # Load & parse GSM8K JSONL, select train/test splits
├── forward_pass.py        # mad_forward_pass() using TextGrad BlackboxLLM
├── evaluator.py           # Custom TextLoss creation, answer parsing
├── train.py               # Training loop with batch optimization
├── evaluate.py            # Evaluation script computing all 5 metrics + plots
├── utils.py               # Answer parsing, logging, prompt I/O
└── out/
    └── tg_mad/
        ├── prompt_history.json
        ├── eval_results.json
        ├── round_accuracy_comparison.png
        ├── subversion_correction_bar.png
        └── overall_accuracy_bar.png
```

---

## 7. Configuration (config.py)

```python
# === Models ===
DEBATER_MODEL = "hosted_vllm/Qwen3-4B-Instruct-2507"
EVALUATOR_MODEL = "hosted_vllm/Qwen3-8B"  # Change to "minimax/M2.5" later
VLLM_BASE_URL = "http://localhost:8000/v1"  # Adjust to your vLLM server

# === Debate ===
N_AGENTS = 3
N_ROUNDS = 3  # T=3
TEMPERATURE = 0.7  # For agent diversity

# === Training ===
BATCH_SIZE = 5
NUM_EPOCHS = 2
TRAIN_SIZE = 10

# === Data ===
EXISTING_DATA_PATH = "out/history/gsm8k/gsm8k_500__qwen3-4b_N=3_R=3.jsonl"
OUTPUT_DIR = "out/tg_mad/"

# === Evaluator swap ===
# To switch to MiniMax M2.5:
# 1. Change EVALUATOR_MODEL to the appropriate litellm string
# 2. Set MINIMAX_API_KEY in environment
# 3. No other code changes needed
```

---

## 8. Key Implementation Notes for Coding Agent

1. **Inspect the existing JSONL first.** Before writing any code, read 2-3 lines of the existing data file to understand its exact schema. The field names and structure will determine how `data_loader.py` works.

2. **TextGrad computation graph is essential.** The forward pass MUST use `tg.BlackboxLLM` with `debater_prompt` as the system prompt parameter, not raw API calls. Otherwise `loss.backward()` won't propagate gradients to `debater_prompt`.

3. **`tg.sum` for batching is native.** Use `tg.sum(losses)` then `total_loss.backward()` then `optimizer.step()`. This is the standard TextGrad batch pattern. Gradients from individual losses are concatenated (not averaged) before being sent to the optimizer.

4. **Generate gradients for CORRECT answers too.** The evaluator should produce positive feedback when debate works well (correction, defense). Do NOT skip `loss.backward()` on correct answers.

5. **The t=0 baseline is frozen.** For the paper's MV baseline, use the `t=0` answers from the existing JSONL (which used the original unoptimized prompt). Do NOT re-run `t=0` with the optimized prompt for the baseline — that would conflate the two.

6. **Answer comparison must be numeric.** Compare parsed float values, not string equality. Use tolerance: `abs(predicted - ground_truth) < 0.01`.

7. **Rate limiting.** With 3 agents × 4 rounds × 10 training samples = 120 LLM calls per epoch just for forward passes, plus evaluator calls and backward pass calls. For local vLLM this should be fine, but add retry logic with exponential backoff.

8. **Prompt constraints.** When creating the `tg.TGD` optimizer, pass a constraint string to ensure the optimizer doesn't break the answer format requirement:

    ```python
    optimizer = tg.TGD(
        parameters=[debater_prompt],
        constraints=[
            "The prompt must always instruct agents to end responses with 'Answer: <number>'.",
            "The prompt must be a system prompt for math debate agents."
        ]
    )
    ```

9. **Evaluator model swap.** The evaluator model is used both as the TextLoss engine and as `set_backward_engine`. To swap to MiniMax M2.5, only the engine string and API key need to change. The `create_evaluator_loss` function and all other code remain identical. Design the code so the evaluator model is read from `config.py`.

10. **Error handling.** LLM calls can fail (timeout, malformed response, unparseable answer). Wrap each agent call in try/except. If an agent fails to produce a parseable answer, use `None` and treat it as incorrect in metrics. Log failures.
