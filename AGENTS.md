# AGENTS.md

## Mission
Build and evaluate TextGrad-based optimization for multi-agent debate. The core goal is to improve final debate accuracy over fixed baselines while keeping runs reproducible and comparisons fair.

## Working Rules
- Read this file before making repo changes.
- State the active objective and assumptions before substantial work.
- Inspect relevant files before editing.
- Prefer small, backward-compatible changes behind explicit flags or new CLI args.
- Keep deterministic seeds and log all run-critical config fields.
- After changes, run a targeted validation command and report the result.
- Append a short entry to `diary/<YYYY-MM-DD>.md` for every coding-task change.

## Project Constraints
- Do not change default behavior unless a flag or explicit config enables it.
- Do not use ground-truth labels to choose inference-time behavior.
- Keep generation temperature fixed at `1.0`.
- Always compare TG-MAD against fixed baselines, including majority vote.
- Keep dataset/model settings separated in outputs and analysis.

## Artifacts
- Put experiment outputs under `out/`.
- Put saved debate/text histories under `out/history/`.
- Use a fresh `output_dir` for materially different runs.
- Keep JSON/JSONL schemas additive so older analysis still works.

## Canonical TG-MAD Entrypoints
- Training: [scripts/run_tg_mad_train.sh](/export/home3/dazhou/debate-or-vote/scripts/run_tg_mad_train.sh)
- Evaluation: [scripts/run_tg_mad_eval.sh](/export/home3/dazhou/debate-or-vote/scripts/run_tg_mad_eval.sh)
- Orchestration: [tg_mad/job_runner.py](/export/home3/dazhou/debate-or-vote/tg_mad/job_runner.py)

Configure experiments by exporting env vars before `sbatch`; do not add new wrapper scripts unless there is a strong reason. Common env vars:
- `OUTPUT_DIR`
- `DATASET`
- `TRAIN_EXISTING_DATA`
- `EVAL_EXISTING_DATA`
- `PROMPT_HISTORY_PATH`
- `DEBATER_MODEL_NAME`
- `EVALUATOR_MODEL_NAME`
- `TRAIN_N_AGENTS` / `EVAL_N_AGENTS`
- `TRAIN_N_ROUNDS` / `EVAL_N_ROUNDS`

## job_runner.py Notes
- `python -m tg_mad.job_runner train|eval --dry-run` is the fastest way to inspect the exact launch plan.
- `job_runner.py` handles local vLLM server startup, health checks, port conflicts, GPU preflight checks, and CLI/env plumbing for train/eval.
- `START_DEBATER_SERVER=0` and `START_EVALUATOR_SERVER=0` let a job reuse already-running servers.
- Prefer GPU auto-pick on shared nodes:
  - `DEBATER_AUTO_PICK_GPU=1`
  - `EVALUATOR_AUTO_PICK_GPU=1`
  - `EVALUATOR_AUTO_PICK_BEFORE_DEBATER=1`
  - `EVALUATOR_AUTO_PICK_PREFER_TOPOLOGY=1`
- Use `DEBATER_MIN_FREE_MIB` and `EVALUATOR_MIN_FREE_MIB` to reject dirty cards before startup.
- For long model startup, increase `MAX_WAIT_SECONDS`; `job_runner.py` propagates it to the vLLM engine-ready timeout.

## GPU / Partition Experience
- Submit TG-MAD jobs through SLURM. `localhost` means the current SLURM node, so train/eval and local servers must live on the same node unless a remote host is explicitly configured.
- Check availability first with `python scripts/gpu_monitor.py`. If needed, fall back to `python scripts/gpu_monitor.py --partitions PA100q` or `sinfo`.
- Prefer partition-only scheduling. Do not pin specific nodes unless the user explicitly asks.

### PA100q
- Good default partition for general TG-MAD runs and smaller local serving setups.
- Not reliable for the local `Qwen/Qwen3-30B-A3B-Instruct-2507` evaluator path in this repo. Repeated failures showed `pynccl`, `flashinfer_cutlass`, and `shm_broadcast` startup problems on 2xA100-40GB even when GPUs were otherwise idle.
- If a local 30B evaluator is required, do not assume PA100q will be stable. Prefer RTXA6Kq or use an API evaluator.

### RTXA6Kq
- This was the stable path for the successful local-30B evaluator experiment.
- Shared A6000 nodes often have dirty cards; a job may receive nominally idle GPUs that fail free-memory checks or behave inconsistently.
- For local 30B evaluator runs, it worked better to request a larger GPU pool and let `job_runner.py` auto-pick the clean subset. In practice, requesting 6 GPUs to use an actual 3-GPU layout (2 evaluator + 1 debater) was more reliable than requesting exactly 3.
- Historical note: node11 showed misleading CUDA state where some visible GPUs failed `torch.cuda.set_device()` even though `nvidia-smi` looked fine. Trust runtime preflight checks over static node assumptions.

## Local 30B Evaluator Notes
- `Qwen/Qwen3-30B-A3B-Instruct-2507` did not fit on one ~47 GB GPU.
- The workable layout was tensor parallel size `2` across two GPUs for the evaluator, plus one separate GPU for the debater.
- TextGrad can fail at `optimizer.step()` even after debate generation succeeds. Treat evaluator context length as a first-class constraint.
- Deterministic `400 Bad Request` context-length errors should fail fast; do not mask them with retries.
- Practical mitigations that helped:
  - `batch_size=1`
  - shorter debate depth when needed
  - reduced prompt verbosity
  - `max_model_len` tuning on the evaluator

## Monitoring Expectations
- Use a sleep-based wait/check loop for experiment runs; do not report success until the run is clearly stable.
- If a run fails, diagnose the cause, fix it, and continue monitoring the replacement run until it is stable.

## Definition of Done
- Behavior is backward compatible unless explicitly enabled otherwise.
- A reproducible command or launch pattern is available.
- Validation was run and reported.
- The result can still be compared cleanly against baseline outputs.
