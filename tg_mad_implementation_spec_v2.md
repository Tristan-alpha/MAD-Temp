# TG-MAD implementation spec v2

## For: coding agent
## Version: 2.0 (revised MVP — simplified signals, no IntrospecLOO, no scalar J)
## Base codebase: [debate-or-vote](https://github.com/deeplearning-wisc/debate-or-vote)

---

## 0. Objective

Extend the Debate-or-Vote codebase so it can use TextGrad to optimize the shared debate-update prompt in the original 5-agent simultaneous-talk setting.

The system must:

1. Preserve the original 5-agent simultaneous-talk debate protocol exactly.
2. Log the full debate trajectory as a structured trace.
3. Compute three simple process signals from the trajectory (FlipRate, DisagreementDrop, Stability).
4. Build a compact diagnostic trace summary per episode.
5. Feed trace summaries and signal averages into TextGrad to iteratively improve the shared revision prompt.

This version is deliberately minimal. It treats the Debate-or-Vote setting as memoryless: each agent update is context-conditioned resampling, not persistent stateful deliberation.

**There is no scalar reward (J) in this version.** TextGrad receives trace summaries and a signal vector directly. This avoids premature commitment to a weighting scheme before empirical signal behavior is understood.

---

## 1. Design rationale

### 1.1 Why no IntrospecLOO or causal signals in v1

In the Debate-or-Vote setting, agents do not carry persistent memory. Each round is:
- original question
- current peer responses
- update prompt
- new sampled answer

Asking an agent to "ignore agent X's response" (IntrospecLOO) is asking for a post-hoc counterfactual in a system that doesn't maintain internal state across rounds. The approximation is weak in this setting.

**Decision**: defer IntrospecLOO and all causal influence signals (C, E, F, O) to Phase 5 extensions. Use simple trajectory statistics in v1.

### 1.2 Why no scalar J in v1

A scalar objective requires choosing signal weights before knowing how the signals behave empirically. TextGrad can consume rich textual context, so passing trace summaries + a signal vector is both more informative and less brittle than a premature scalar.

**Decision**: TextGrad optimizes based on trace summaries and batch-level signal averages. If a scalar is needed for logging/checkpointing, use the optional auxiliary score `S_aux` (defined in section 4.4), but do not use it as an optimization target.

---

## 2. What stays unchanged

Do not modify any of the following in v1:

- Number of agents: **5**
- Debate protocol: simultaneous-talk (all peers visible each round)
- Final system answer: majority vote over final-round agent labels (existing repo logic)
- Evaluation logic: the repo's existing accuracy computation
- Entry point structure: `src/main.py` remains the experiment driver

The only additions are instrumentation (trace logging, perturbation replays), signal computation, and a prompt-optimization wrapper.

---

## 3. What to optimize

Optimize **one text variable only**: the shared agent revision / debate-update prompt.

This is the instruction each agent sees when revising its answer after reading peers' responses. It is shared across all 5 agents and all rounds.

Do **not** optimize in v1:
- The task prompt (question framing)
- The answer-extraction prompt
- The final aggregation rule
- Agent personas or per-agent strategy cards
- Multiple prompts simultaneously

---

## 4. Process signals

Use only **three** signals in v1. All are computed from the debate trajectory and perturbation replays. No ground-truth labels are used.

### 4.1 FlipRate

Measures how much answer-changing occurred during debate.

For each agent i, define:

```
Flip_i = 1 if agent i changed its answer at any round >= 1, else 0
```

Then:

```
FlipRate = (1/5) * sum(Flip_i for i in range(5))
```

**Scale**: [0.0, 1.0]

**Interpretation**:
- 0.0: debate was inert — no agent revised
- 0.2-0.4: limited revision
- 0.4-0.8: several agents revised — active debate
- 0.8-1.0: most/all agents revised — may be productive or noisy

**Known limitation**: FlipRate does not distinguish productive flips (convergence toward a better answer) from noisy flips (random oscillation). The diagnosis rules in section 12 cross-reference FlipRate with DisagreementDrop to partially address this. This is acceptable for v1 — the distinction requires causal signals (Phase 5).

### 4.2 DisagreementDrop

Measures how much pairwise disagreement decreased from the initial round to the final round.

Define round-level disagreement:

```
D(r) = (2 / (5 * 4)) * sum(1 for i < j if y_i(r) != y_j(r))
```

Then:

```
DisagreementDrop = D(0) - D(R)
```

**Scale**: [-1.0, 1.0] (typically [0.0, 0.6])

**Interpretation**:
- positive: debate produced convergence
- ~0: debate did not change collective disagreement
- negative: debate increased disagreement (rare, signals a problem)

### 4.3 Stability

Measures how robust the final majority answer is to small perturbations of the final round.

Run exactly 3 perturbations of the final update only (see section 10 for details). Let `y_hat` be the original final majority label and `y_hat(m)` the perturbed answers.

```
Stability = (1/3) * sum(1 for m in range(3) if y_hat(m) == y_hat)
```

**Scale**: {0.0, 0.33, 0.67, 1.0}

**Interpretation**:
- 1.0: rock-solid consensus
- 0.67: moderately stable
- 0.33: fragile
- 0.0: every perturbation flipped the answer

### 4.4 Optional auxiliary score (logging only)

If a single scalar is needed for checkpointing or trend logging, use:

```
S_aux = DisagreementDrop + Stability - 0.5 * abs(FlipRate - 0.5)
```

This rewards convergence and stability while penalizing both inert debates (FlipRate too low) and noisy debates (FlipRate too high). The "ideal" FlipRate is centered at 0.5 — this is a tunable assumption, not a ground truth.

**This is NOT the optimization target.** It is for logging only. TextGrad does not see S_aux.

---

## 5. What is explicitly excluded from v1

Do not implement any of the following:

- IntrospecLOO (soft or hard leave-one-out)
- Causal influence signal (C)
- Earned consensus signal (E)
- Fragility signal (F) — replaced by Stability
- Overconfidence signal (O)
- Scalar objective (J)
- Semantic clustering or JSD
- Claim extraction or claim graphs
- Verbalized confidence extraction
- Per-agent strategy cards
- Cross-debate normalization or replay buffers
- Redundancy signal (R)

These may return in Phase 5 extensions once the MVP is validated.

---

## 6. New modules

| Module | Responsibility |
|---|---|
| `trace_logger.py` | Initialize, populate, and serialize per-question EpisodeTrace objects |
| `answer_parser.py` | Parse raw LLM output into canonical discrete labels, dataset-aware |
| `perturbation_runner.py` | Run the 3 final-round perturbation replays |
| `signal_engine.py` | Compute FlipRate, DisagreementDrop, Stability |
| `trace_summary.py` | Build compact diagnostic text summaries for TextGrad |
| `tg_optimizer.py` | Build TextGrad optimization context and propose prompt revisions |

No `introspec_loo.py` in v1.

---

## 7. Data structures

### 7.1 Core types

```python
@dataclass
class AgentRoundRecord:
    agent_id: int
    raw_output: str
    parsed_label: str | None   # None if parse failed
    prompt_used: str

@dataclass
class RoundRecord:
    round_idx: int
    agents: list[AgentRoundRecord]  # length 5
    majority_label: str | None
    label_histogram: dict[str, int]

@dataclass
class PerturbationResult:
    name: str                    # "shuffle_order" | "paraphrase_update" | "resample_once"
    final_majority_label: str | None
    agent_labels: dict[int, str | None]  # agent_id -> label

@dataclass
class SignalValues:
    flip_rate: float
    disagreement_drop: float
    stability: float

@dataclass
class EpisodeTrace:
    question_id: str
    question_text: str
    task_type: str               # "mmlu", "gsm8k", "arithmetics", "hellaswag", "csqa"
    num_agents: int              # always 5
    num_rounds: int              # R (start with 2)
    rounds: list[RoundRecord]
    final_majority_label: str | None
    perturbations: list[PerturbationResult]
    signals: SignalValues | None # computed after perturbations complete
```

### 7.2 Parse failure handling

- If `parse_label()` returns None, set `parsed_label = None` and increment `"PARSE_FAIL"` in the histogram.
- For perturbation replays: if an agent's perturbed answer is unparseable, treat that agent's answer as **different from original** (conservative — parse failure counts as instability).
- Skip episodes from the TextGrad optimization batch if more than 2 agents fail parsing at round 0. Still save the trace for debugging.

---

## 8. Label parsing

### 8.1 Implementation

```python
def parse_label(raw_output: str, task_type: str) -> str | None:
    """
    Extract canonical label from raw LLM output.
    Returns None if parsing fails.
    """
```

### 8.2 Rules per task type

| Task type | Label space | Extraction rule |
|---|---|---|
| `mmlu` | `A`, `B`, `C`, `D` | Find last occurrence of a single letter A-D after "answer" keyword or at end of response |
| `hellaswag` | `A`, `B`, `C`, `D` | Same as MMLU |
| `csqa` | `A`, `B`, `C`, `D`, `E` | Same, extended to E |
| `gsm8k` | Numeric string | Extract number after `####` or last number in response; normalize (strip commas, trailing zeros) |
| `arithmetics` | Numeric string | Same as GSM8K |
| Binary tasks | `yes` / `no` or `A` / `B` | Match last yes/no or option letter |

### 8.3 Unit tests for parser

Must handle at minimum:
- `"The answer is B"` → `"B"`
- `"I think it's (C)"` → `"C"`
- `"#### 42"` → `"42"`
- `"The result is 1,234.50"` → `"1234.5"`
- `"I'm not sure"` → `None`
- `"Let me think... A. No wait, B."` → `"B"` (last occurrence wins)

---

## 9. Temperature policy

The main debate and perturbation replays require a clear temperature policy. Without this, the "resample" perturbation is meaningless.

### 9.1 Main debate

Run the main debate at **temperature > 0** (e.g., `temperature=0.7` or whatever the Debate-or-Vote repo defaults to). At `temperature=0`, all agents would produce identical outputs given identical inputs, making the resample perturbation a no-op.

### 9.2 Perturbation replays

- **Shuffle order**: same temperature as main debate, same seed. Only the message ordering changes.
- **Paraphrase update**: same temperature as main debate, same seed. Only the prompt text changes.
- **Resample**: same temperature as main debate, **different seed**. Everything else identical.

### 9.3 If the repo uses temperature=0

If the Debate-or-Vote codebase defaults to `temperature=0` for debate rounds:
- Keep `temperature=0` for the main debate (preserve comparability).
- For the resample perturbation specifically, use `temperature=0.3` to introduce controlled sampling noise. Document this in the experiment config.

---

## 10. Perturbation replays

Run exactly 3 perturbations per episode, each applied to the **final debate round only**.

### 10.1 Shuffle order

Randomly permute the order of peer messages in the final-round context.

```python
import random

def shuffle_perturbation(peer_messages: list, question_id: str) -> list:
    """Deterministic shuffle based on question_id."""
    shuffled = peer_messages.copy()
    random.Random(hash(question_id) & 0xFFFFFFFF).shuffle(shuffled)
    return shuffled
```

### 10.2 Paraphrase update

Replace the revision prompt with a pre-written semantically equivalent variant. Pre-write **exactly 2 variants** before the experiment and store them in the config file.

**Variant examples** (adapt to match your actual revision prompt):

```yaml
paraphrase_variants:
  - >
    Review the other participants' answers below. Weigh their evidence
    against your own reasoning. Revise your response only if their
    arguments are stronger; if not, maintain your position with
    justification.
  - >
    Other agents have provided their answers. Evaluate whether any
    of them present a more compelling case than yours. Change your
    answer if persuaded, or reaffirm it with your reasoning if not.
```

Select deterministically: `variant_index = hash(question_id) % len(variants)`.

### 10.3 Resample

Re-run the exact same final-round prompt with a different random seed (see temperature policy in section 9).

```python
def resample_seed(original_seed: int) -> int:
    return original_seed + 10007  # deterministic, large offset to avoid correlation
```

### 10.4 What each perturbation isolates

| Perturbation | What changes | What stays | Isolates |
|---|---|---|---|
| Shuffle order | Peer message ordering | Prompt, content, seed | Primacy/recency bias |
| Paraphrase | Revision prompt wording | Peer messages, order, seed | Instruction sensitivity |
| Resample | Random seed (+ temperature if base is 0) | Prompt, messages, order | Sampling noise |

### 10.5 Output

Each perturbation produces a `PerturbationResult` with the majority label and per-agent labels. Store all three in the EpisodeTrace.

---

## 11. Signal computation

### 11.1 FlipRate

```python
def compute_flip_rate(trace: EpisodeTrace) -> float:
    changed_agents = 0
    for agent_id in range(trace.num_agents):
        flipped = False
        for r in range(1, len(trace.rounds)):
            prev = trace.rounds[r - 1].agents[agent_id].parsed_label
            curr = trace.rounds[r].agents[agent_id].parsed_label
            if prev is not None and curr is not None and prev != curr:
                flipped = True
                break
        changed_agents += int(flipped)
    return changed_agents / trace.num_agents
```

### 11.2 DisagreementDrop

```python
def pairwise_disagreement(labels: list[str | None]) -> float:
    valid = [x for x in labels if x is not None]
    n = len(valid)
    if n < 2:
        return 0.0
    total = 0
    diff = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            diff += int(valid[i] != valid[j])
    return diff / total

def compute_disagreement_drop(trace: EpisodeTrace) -> float:
    d0_labels = [a.parsed_label for a in trace.rounds[0].agents]
    dR_labels = [a.parsed_label for a in trace.rounds[-1].agents]
    return pairwise_disagreement(d0_labels) - pairwise_disagreement(dR_labels)
```

### 11.3 Stability

```python
def compute_stability(trace: EpisodeTrace) -> float:
    if not trace.perturbations:
        return 0.0
    matches = sum(
        1 for p in trace.perturbations
        if p.final_majority_label is not None
        and p.final_majority_label == trace.final_majority_label
    )
    return matches / len(trace.perturbations)
```

### 11.4 Combined computation

```python
def compute_all_signals(trace: EpisodeTrace) -> SignalValues:
    return SignalValues(
        flip_rate=compute_flip_rate(trace),
        disagreement_drop=compute_disagreement_drop(trace),
        stability=compute_stability(trace),
    )
```

---

## 12. Deterministic diagnosis rules

Use fixed rules to generate the one-line diagnosis. Do not use an LLM to interpret signals.

### 12.1 Signal interpretation strings

**FlipRate**:
- `== 0.0`: `"debate was inert; no agent revised"`
- `<= 0.4`: `"limited revision occurred"`
- `<= 0.8`: `"several agents revised their answers"`
- `> 0.8`: `"most agents revised; debate may be noisy"`

**DisagreementDrop**:
- `<= 0.0`: `"debate did not reduce disagreement"`
- `<= 0.3`: `"modest convergence"`
- `<= 0.6`: `"substantial convergence"`
- `> 0.6`: `"very strong convergence"`

**Stability**:
- `< 0.34`: `"brittle final answer"`
- `< 0.67`: `"moderately stable"`
- `>= 0.67`: `"stable final answer"`

### 12.2 One-line diagnosis (combinatorial)

Use the first matching rule:

```python
def diagnose(flip_rate: float, disagreement_drop: float, stability: float) -> str:
    if flip_rate <= 0.1 and disagreement_drop <= 0.05:
        return "Debate had little effect; behavior resembled voting."
    if flip_rate > 0.6 and stability < 0.34:
        return "Debate changed many answers but produced a brittle conclusion."
    if flip_rate > 0.6 and disagreement_drop <= 0.05:
        return "Agents revised answers, but convergence did not improve."
    if disagreement_drop > 0.1 and stability >= 0.67:
        return "Debate reduced disagreement and produced a stable outcome."
    if disagreement_drop > 0.1 and stability < 0.67:
        return "Debate reduced disagreement, but the outcome has moderate fragility."
    if flip_rate > 0.3 and disagreement_drop > 0.0:
        return "Active debate with some convergence."
    return "Mixed signals; no clear pattern."
```

---

## 13. Trace summary

### 13.1 Builder

```python
def build_trace_summary(trace: EpisodeTrace) -> str:
    # Question (truncated to 200 chars if long)
    q = trace.question_text[:200] + "..." if len(trace.question_text) > 200 else trace.question_text

    # Initial and final agent labels
    r0 = trace.rounds[0]
    rR = trace.rounds[-1]
    init_labels = ", ".join(
        f"A{a.agent_id}={a.parsed_label or '?'}" for a in r0.agents
    )
    final_labels = ", ".join(
        f"A{a.agent_id}={a.parsed_label or '?'}" for a in rR.agents
    )

    # Round-by-round flips
    flip_lines = []
    for r in range(1, len(trace.rounds)):
        flips = []
        for a in trace.rounds[r].agents:
            prev = trace.rounds[r - 1].agents[a.agent_id].parsed_label
            curr = a.parsed_label
            if prev is not None and curr is not None and prev != curr:
                flips.append(f"A{a.agent_id} {prev}->{curr}")
        line = f"Round {r}: {', '.join(flips)}" if flips else f"Round {r}: no changes"
        flip_lines.append(line)

    # Perturbation outcomes
    pert_lines = [f"{p.name} -> {p.final_majority_label or '?'}" for p in trace.perturbations]

    # Signals
    s = trace.signals

    # Diagnosis
    diagnosis = diagnose(s.flip_rate, s.disagreement_drop, s.stability)

    return f"""Question:
{q}

Initial agent labels:
{init_labels}

Initial distribution:
{r0.label_histogram}

Round-by-round changes:
{chr(10).join(flip_lines)}

Final agent labels:
{final_labels}

Final distribution:
{rR.label_histogram}

Perturbation outcomes:
{chr(10).join(pert_lines)}

Process signals:
- FlipRate = {s.flip_rate:.2f}
- DisagreementDrop = {s.disagreement_drop:.2f}
- Stability = {s.stability:.2f}

Diagnosis:
{diagnosis}"""
```

### 13.2 Length constraint

Each trace summary must stay under **250 words** (~1000 characters). If the question text is very long, truncate to 200 characters. Do not include full agent rationales or raw outputs.

---

## 14. TextGrad integration

### 14.1 What TextGrad sees per optimization step

1. The current shared update prompt.
2. Batch-level signal averages (mean FlipRate, mean DisagreementDrop, mean Stability).
3. Three representative trace summaries from the batch.

### 14.2 Representative trace selection

Select 3 traces from the batch that give TextGrad diverse diagnostic context:

```python
def select_representative_traces(traces: list[EpisodeTrace]) -> list[EpisodeTrace]:
    valid = [t for t in traces if t.signals is not None]
    if len(valid) <= 3:
        return valid

    # Sort by each axis
    by_stability = sorted(valid, key=lambda t: t.signals.stability)
    by_ddrop = sorted(valid, key=lambda t: t.signals.disagreement_drop)

    worst_stab = by_stability[0]
    best_stab = by_stability[-1]

    # Median disagreement drop, but not the same trace as worst/best stability
    remaining = [t for t in valid if t is not worst_stab and t is not best_stab]
    if remaining:
        remaining_sorted = sorted(remaining, key=lambda t: t.signals.disagreement_drop)
        median_ddrop = remaining_sorted[len(remaining_sorted) // 2]
    else:
        median_ddrop = by_ddrop[len(by_ddrop) // 2]

    return [worst_stab, best_stab, median_ddrop]
```

This ensures:
- TextGrad sees the worst failure case (lowest stability).
- TextGrad sees the best success case (highest stability).
- TextGrad sees a representative middle case (median disagreement drop, deduplicated from the stability extremes).

### 14.3 TextGrad evaluator instruction

```text
You are improving the shared update prompt for a 5-agent simultaneous-talk debate system.

The process signals below are not correctness labels. They describe how the debate evolved.

Your goal is to revise the prompt so that:
1. Agents revise their answers when peers provide stronger reasoning (not just because the majority disagrees).
2. Debate reduces disagreement rather than causing random flips.
3. Final answers are stable under small perturbations (message reordering, prompt paraphrasing, resampling).
4. Agents avoid shallow conformity where they switch without justification.

Current prompt:
{current_prompt}

Batch statistics:
- mean FlipRate = {mean_flip_rate:.2f}
- mean DisagreementDrop = {mean_disagreement_drop:.2f}
- mean Stability = {mean_stability:.2f}

Representative traces:

--- Trace 1 (lowest stability) ---
{summary_1}

--- Trace 2 (highest stability) ---
{summary_2}

--- Trace 3 (median disagreement drop) ---
{summary_3}

Constraints:
- The revised prompt must be under 200 words.
- Do not include dataset-specific instructions (the prompt must be task-agnostic).
- Do not instruct agents to always agree or always disagree.
- Do not add complex formatting requirements that could interfere with answer parsing.

Return:
1. The revised update prompt between <prompt> tags.
2. A short rationale for the key changes you made.
```

### 14.4 Prompt length guard

TextGrad can produce increasingly verbose prompts across optimization steps. Enforce a hard cap of **200 words**.

```python
def enforce_prompt_length(prompt: str, max_words: int = 200) -> str:
    words = prompt.split()
    if len(words) <= max_words:
        return prompt
    return " ".join(words[:max_words])
```

If truncation is needed, log a warning. If it happens frequently (>30% of steps), consider adding an explicit length instruction to the TextGrad evaluator template.

---

## 15. Main loop integration

### 15.1 Episode runner

```python
def run_full_episode(
    question: dict,
    revision_prompt: str,
    config: dict
) -> EpisodeTrace:
    # 1. Initialize trace
    trace = trace_logger.init_trace(question, config)

    # 2. Run original debate (UNCHANGED from repo)
    debate_result = run_original_debate(question, revision_prompt, config)

    # 3. Log all rounds
    for r, round_data in enumerate(debate_result.rounds):
        trace_logger.add_round(trace, r, round_data)
    trace.final_majority_label = debate_result.final_majority

    # 4. Run perturbation replays (final round only)
    pert_records = perturbation_runner.run_all(trace, revision_prompt, config)
    trace.perturbations = pert_records

    # 5. Compute signals
    trace.signals = signal_engine.compute_all_signals(trace)

    # 6. Save trace to disk
    trace_logger.save(trace)

    return trace
```

### 15.2 TextGrad optimization loop

```python
def train_prompt(
    initial_prompt: str,
    dev_questions: list[dict],
    config: dict
) -> str:
    prompt = initial_prompt
    batch_size = config.get("textgrad_batch_size", 8)
    num_steps = config.get("textgrad_num_steps", 20)

    best_aux = float("-inf")
    best_prompt = prompt

    for step in range(num_steps):
        # 1. Sample batch
        batch = random.sample(dev_questions, min(batch_size, len(dev_questions)))

        # 2. Run episodes
        traces = [run_full_episode(q, prompt, config) for q in batch]

        # 3. Filter out episodes with too many parse failures
        valid = [t for t in traces if t.signals is not None]
        if len(valid) < 2:
            log(f"Step {step}: insufficient valid traces, skipping")
            continue

        # 4. Compute batch statistics
        mean_fr = mean([t.signals.flip_rate for t in valid])
        mean_dd = mean([t.signals.disagreement_drop for t in valid])
        mean_st = mean([t.signals.stability for t in valid])

        # 5. Select representative traces
        reps = select_representative_traces(valid)
        summaries = [build_trace_summary(t) for t in reps]

        # 6. Build TextGrad context and get revised prompt
        new_prompt = tg_optimizer.optimize(
            current_prompt=prompt,
            mean_flip_rate=mean_fr,
            mean_disagreement_drop=mean_dd,
            mean_stability=mean_st,
            trace_summaries=summaries,
        )

        # 7. Enforce length cap
        new_prompt = enforce_prompt_length(new_prompt, config.get("max_prompt_words", 200))

        # 8. Optional: compute S_aux for logging/checkpointing
        s_aux = mean_dd + mean_st - 0.5 * abs(mean_fr - 0.5)
        if s_aux > best_aux:
            best_aux = s_aux
            best_prompt = new_prompt

        # 9. Update prompt for next step
        prompt = new_prompt

        # 10. Log
        log_step(step, prompt, mean_fr, mean_dd, mean_st, s_aux)

    return best_prompt
```

---

## 16. Configuration

```yaml
# Debate settings (match the paper)
num_agents: 5
debate_rounds: 2              # start with 2; test 3 in Phase 5
debate_temperature: 0.7       # must be > 0 for resample perturbation to work

# Perturbations
enable_perturbation_eval: true
num_perturbations: 3          # always 3 in v1 (one of each type)
resample_temperature: null    # null = same as debate_temperature; set to 0.3 if debate uses 0.0
paraphrase_variants:
  - >
    Review the other participants' answers below. Weigh their evidence
    against your own reasoning. Revise your response only if their
    arguments are stronger; if not, maintain your position with
    justification.
  - >
    Other agents have provided their answers. Evaluate whether any
    of them present a more compelling case than yours. Change your
    answer if persuaded, or reaffirm it with your reasoning if not.

# TextGrad
textgrad_batch_size: 8
textgrad_num_steps: 20
trace_examples_per_step: 3    # worst stability, best stability, median ddrop
max_prompt_words: 200

# Logging
save_traces: true
trace_output_dir: "traces/"
log_s_aux: true               # log optional auxiliary score for trend monitoring
```

---

## 17. Acceptance tests

All of these must pass before the implementation is considered correct.

### Test 1: trace completeness

A valid EpisodeTrace must contain:
- Round 0 with 5 agent records and parsed labels
- Final round with 5 agent records and parsed labels
- Majority label and label histogram for every round
- Exactly 3 PerturbationResult entries
- A non-null SignalValues object

### Test 2: signal bounds

- `0.0 <= flip_rate <= 1.0`
- `-1.0 <= disagreement_drop <= 1.0`
- `stability in {0.0, 1/3, 2/3, 1.0}` (since exactly 3 perturbations)

### Test 3: parse failure handling

- `parse_label("I'm not sure", "mmlu")` returns `None`
- An episode where 3+ agents fail parsing at round 0 has `signals = None`
- A perturbation where an agent's output is unparseable is treated as "different from original"

### Test 4: perturbation determinism

- `shuffle_perturbation(messages, "q123")` called twice returns the same order
- `hash("q123") % 2` always selects the same paraphrase variant

### Test 5: summary length

Each trace summary is under 250 words.

### Test 6: prompt length guard

After TextGrad optimization, the resulting prompt is at most 200 words.

### Test 7: signal correctness on synthetic data

Given a synthetic trace where:
- All 5 agents answer A at round 0
- No agent changes at round 1

Then:
- `flip_rate == 0.0`
- `disagreement_drop == 0.0`
- `stability == 1.0` (if all perturbations also return A)

Given a synthetic trace where:
- Round 0: agents answer A, B, A, B, A (D0 = 0.6)
- Round 1: agents answer A, A, A, A, A (DR = 0.0)

Then:
- `flip_rate == 0.4` (2 of 5 agents flipped)
- `disagreement_drop == 0.6`

---

## 18. Rollout phases

### Phase 1: trace infrastructure

**Implement**: `trace_logger.py`, `answer_parser.py`, `perturbation_runner.py`, `compute_stability()`.

**Deliverable**: run the existing debate, log traces to disk, compute Stability for each episode.

**Gate**: saved traces have the correct structure; Stability values are in {0, 1/3, 2/3, 1}.

### Phase 2: full signal engine

**Implement**: `compute_flip_rate()`, `compute_disagreement_drop()`, diagnosis rules, `trace_summary.py`.

**Deliverable**: all three signals computed per episode. Summaries readable and under 250 words.

**Gate**: signals vary meaningfully across questions; summaries contain all required fields.

### Phase 3: TextGrad integration

**Implement**: `tg_optimizer.py`, batch stats, representative trace selection, training loop.

**Deliverable**: run 10-20 optimization steps on a dev split. Observe signal trends.

**Gate**: TextGrad produces valid prompts under 200 words; at least one signal shows improvement trend over steps.

### Phase 4: held-out evaluation

**Compare**: TG-MAD accuracy vs Majority Voting, vanilla MAD, optionally MAD-Conformist / MAD-Follower.

**Gate**: TG-MAD accuracy >= MV accuracy on at least one benchmark.

### Phase 5: extensions (only after MVP works)

Consider adding:
- IntrospecLOO probes
- Causal influence signal (C)
- Earned consensus (E), Fragility (F), Overconfidence (O)
- Normalized scalar objective (J)
- Hard LOO on an audit subset
- Additional benchmarks (3-round, 5-round, sparse MAD)
- Signal ablation experiments

---

## 19. Final instruction block for the coding agent

```text
Implement a minimal TextGrad optimization pipeline on top of the Debate-or-Vote codebase.

Requirements:
1. Keep the original 5-agent simultaneous-talk debate protocol unchanged.
2. Optimize only the shared debate-update prompt.
3. Add per-episode trace logging:
   - parsed agent labels for every round
   - majority label and histogram per round
   - final majority label
4. Add 3 final-round perturbation replays:
   - shuffle peer order (deterministic by question_id)
   - paraphrase update prompt (2 pre-written variants, selected by question_id)
   - resample once (different seed, same or slightly elevated temperature)
5. Compute only these process signals in v1:
   - FlipRate: fraction of agents that changed answer at any round
   - DisagreementDrop: D(round 0) - D(final round)
   - Stability: fraction of perturbations that preserved the majority answer
6. Build a compact trace summary per episode containing:
   - question (truncated to 200 chars)
   - initial and final agent labels and distributions
   - round-by-round flips
   - perturbation outcomes
   - the 3 signal values
   - one-line deterministic diagnosis
7. Feed TextGrad per optimization step:
   - the current shared prompt
   - batch averages of the 3 signals
   - 3 representative trace summaries (worst stability, best stability, median disagreement drop — deduplicated)
8. Enforce a 200-word cap on the optimized prompt.
9. Do NOT implement IntrospecLOO, hard LOO, scalar J, C/E/F/O signals,
   semantic clustering, JSD, claim graphs, or verbal confidence in v1.
10. Add tests for: trace completeness, signal bounds, parse failure handling,
    perturbation determinism, summary length, prompt length, and
    signal correctness on synthetic data.
```
