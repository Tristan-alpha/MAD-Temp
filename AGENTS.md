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
6. Update the daily diary file with a short summary of the changes made that day.

## Daily Diary
- Record each coding-task change in a diary file under `diary/` using the current date as the filename, for example `diary/2026-03-16.md`.
- Append concise entries describing what changed, why, and any validation or run commands that were executed.
- Keep diary updates additive so the day-by-day history remains easy to audit.

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

## TG-MAD Operational Notes
- For local vLLM serving, `localhost` means the current machine or SLURM job node. Train/eval must reach servers on the same node unless an explicit cross-node host is configured.
- Training requires both the debater server and the evaluator/backward server at the same time. Evaluation only requires the debater server.
- Submit work through SLURM instead of trying to run it locally.
- Before submitting TG-MAD jobs, prefer checking cluster GPU availability with the local `gpu` helper (`python scripts/gpu_monitor.py`). If that helper is unavailable in a non-interactive shell, fall back to `python scripts/gpu_monitor.py --partitions PA100q` or `sinfo`.
- Prefer partition-only scheduling for TG-MAD jobs; do not pin specific nodes unless the user explicitly asks for it.
- For 30B local evaluator runs, prefer the `PA100q` partition, request 3 GPUs total, and run the evaluator with tensor parallel size `2` across 2 GPUs while leaving 1 GPU for the debater. When auto-pick is enabled, prefer the cleanest allocated GPUs for the evaluator.
- For every experiment run, use a sleep-based wait/check loop and do not report completion until training/evaluation is stable.
- If training/evaluation shows problems, diagnose and fix them, then continue monitoring until the run is stable before finishing the response.
- TextGrad failures can occur at `optimizer.step()` even when debate forward passes succeed. Treat the optimizer prompt length as a first-class constraint.
- The main context-overflow mitigation that worked in practice was: batch size `1`, `n_rounds=1` (`t0` plus one debate round), a shorter second-round debate prompt, and `max_model_len=16384`.
- `Qwen/Qwen3-30B-A3B-Instruct-2507` did not fit as a local vLLM evaluator on one 47 GB GPU, but it did start successfully with tensor parallel size `2` across two GPUs.
- For SLURM wrapper scripts that call other repo scripts, use an absolute repo path or `cd` into the repo root first. SLURM may execute a copied script from its spool directory, which breaks relative paths.
- Keep deterministic `400 Bad Request` context-length errors fail-fast. Retrying them only wastes time and hides the real bottleneck.
- Use separate `output_dir` values for materially different TG-MAD experiments so prompt histories and eval artifacts are not mixed silently.

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
