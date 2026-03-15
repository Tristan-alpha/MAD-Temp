# AGENTS.md

## Mission
Build and evaluate TextGrad-based optimization for multi-agent debate, with the goal of improving final debate accuracy over majority voting while preserving reproducibility and fair baseline comparisons.

## Project Scope
- Codebase focus:
`src/main.py`, `src/evaluator.py`, `src/model/*`, `scripts/*.sh`, `scripts/analyze_results.py`.
- Primary research question:
How TextGrad-based strategy optimization affects debate dynamics and final accuracy.
- Primary feature direction:
Self-supervised strategy updates that improve debate quality without leaking ground-truth labels at inference time.

## Non-Negotiable Rules
- Keep default behavior unchanged unless an explicit flag enables new behavior.
- Preserve reproducibility: deterministic seeds and logged config fields.
- Do not mix analysis outputs from different dataset/model settings without explicit grouping keys.
- Avoid label leakage:
No use of reference answers to choose inference-time strategy.

## Required Start-of-Task Workflow
For every coding task in this repository:
1. Read this `AGENTS.md` first.
2. State the active objective and assumptions.
3. Inspect relevant files before editing.
4. Propose minimal changes that preserve backward compatibility.
5. After changes, run a targeted validation command and report result.

## Strategy Optimization Contract
- Optimization inputs may include only signals available at decision time, such as:
prior-round responses, disagreement statistics, vote entropy, response length/format quality, and historical self-supervised reward estimates.
- Optimization must support:
fixed baseline behavior and optional learned behavior behind explicit flags.
- Self-supervised updates should rely on delayed/proxy feedback from debate outcomes, consistency, or agreement structure, not privileged labels during inference.
- Generation temperature is fixed to `1.0`.

## Interface and Configuration Guidelines
- Add new CLI args instead of overloading old ones.
- Keep existing args functional, including `--max_new_tokens`.
- If adding optimization logic, use explicit flags and log configuration into output metadata for each run.

## Evaluation Protocol
- Always compare against fixed-strategy baselines (including majority vote).
- Report at least:
final debate accuracy, per-round accuracy trajectory, and variance across seeds.
- Keep dataset-specific reporting separated (`gsm8k`, `arithmetics`, `csqa`, etc.).
- Ensure analysis scripts can still parse historical files and new files.

## Data and Artifact Conventions
- Keep outputs under `out/` and histories under `out/history`.
- Do not overwrite prior results silently.
- Any new output schema must remain JSONL-friendly and versioned with additive fields.

## Code Quality Expectations
- Small, reviewable diffs.
- Clear naming for optimization state and reward signals.
- Add short comments only for non-obvious logic.
- Add or update lightweight tests/check scripts when behavior changes.

## Definition of Done
- New behavior is flag-gated and backward compatible.
- A reproducible command is provided.
- Validation command has been run and result reported.
- Analysis path for comparing against baseline is documented in the task summary.
