# TG-MAD Implementation Notes

Date: 2026-03-16

Historical note: the repo later consolidated the TG-MAD SLURM entrypoints down
to the canonical `scripts/run_tg_mad_train.sh` and
`scripts/run_tg_mad_eval.sh` wrappers. Older wrapper-script names mentioned in
this note are kept here as historical context, not as current recommended
entrypoints.

## Scope

This note documents the TG-MAD implementation and runtime integration work added to this repository on 2026-03-16. The goal was to add a TextGrad-based multi-agent debate workflow under `tg_mad/`, provide SLURM entry points for training and evaluation with local vLLM servers, review the runtime failures, and converge on a working 2-GPU training setup.

## Files Added Or Updated

### New TG-MAD package

- `tg_mad/__init__.py`
- `tg_mad/config.py`
- `tg_mad/data_loader.py`
- `tg_mad/engine.py`
- `tg_mad/evaluate.py`
- `tg_mad/evaluator.py`
- `tg_mad/forward_pass.py`
- `tg_mad/train.py`
- `tg_mad/utils.py`

### New or updated scripts

- `scripts/check_tg_mad_smoke.py`
- `scripts/run_tg_mad_all_in_one.sh`
- `scripts/run_tg_mad_eval.sh`
- `scripts/run_tg_mad_eval_1gpu.sh`
- `scripts/run_tg_mad_train.sh`
- `scripts/run_tg_mad_train_2gpu.sh`
- `scripts/setup_textgrad_env.sh`
- `scripts/start_vllm_servers.sh`

## Functional Changes

### 1. Added a dedicated TG-MAD pipeline

The new `tg_mad/` package implements a prompt-optimization workflow built on TextGrad:

- `train.py` optimizes a shared debater system prompt by running full multi-agent debates and applying TextGrad updates.
- `evaluate.py` computes baseline metrics from historical JSONL outputs and compares them against fresh TG-MAD runs.
- `forward_pass.py` runs the multi-agent multi-round debate and keeps TextGrad gradient flow attached to the shared prompt.
- `evaluator.py` creates the TextGrad loss prompt used for prompt updates.
- `data_loader.py` rebuilds the GSM8K question ordering to match the existing baseline file and creates a train/test split.
- `utils.py` provides parsing, voting, correctness checks, artifact helpers, and logging helpers.

### 2. Added a custom vLLM-backed TextGrad engine

`tg_mad/engine.py` adds `VLLMEngine`, which uses LiteLLM with an explicit `api_base`. This was necessary because the built-in TextGrad LiteLLM engine did not route requests cleanly to different local vLLM endpoints.

The engine now uses source-aligned generation defaults:

- `temperature = 1.0`
- `top_p = 0.9`
- `max_new_tokens = 512` by default

Those values were chosen to match the existing debate path in:

- `src/model/model_utils.py`
- `scripts/my_scripts/run_q3_4B_Instruct_1_512.sh`

### 3. Preserved backward compatibility with explicit flags and artifact logging

The TG-MAD CLI now logs configuration into run metadata and keeps new behavior configurable through arguments such as:

- `--max_new_tokens`
- `--evaluator_max_new_tokens`
- `--allow_failed_generations`
- `--output_dir`

Artifacts are resolved from the requested output directory instead of hard-coded paths.

### 4. Added cluster-oriented SLURM entry points

The repo now includes:

- `scripts/run_tg_mad_train_2gpu.sh` for 2-GPU training
- `scripts/run_tg_mad_eval_1gpu.sh` for 1-GPU evaluation
- `scripts/run_tg_mad_all_in_one.sh` for a single-job combined workflow

The working training layout is:

- GPU 0: debater model `Qwen/Qwen3-4B-Instruct-2507`
- GPU 1: evaluator model `Qwen/Qwen3-8B`
- local endpoints:
  - `http://127.0.0.1:8000/v1`
  - `http://127.0.0.1:8001/v1`

### 5. Updated environment setup

`scripts/setup_textgrad_env.sh` now supports:

- `INSTALL_VLLM=1 bash scripts/setup_textgrad_env.sh`

This installs `vllm` in the user environment in addition to `textgrad`.

### 6. Added a smoke test

`scripts/check_tg_mad_smoke.py` provides a lightweight local check for the TG-MAD code path without requiring a full training run.

## Runtime Failures Investigated

### 1. Missing vLLM runtime

Early jobs failed immediately because `vllm` was not installed in the user environment. This was fixed by updating the setup script and installing `vllm` in the user space instead of assuming it was already present.

### 2. `localhost` mismatch across nodes

The original multi-step SLURM approach assumed the training process and the vLLM servers were on the same machine. That fails if the client and server jobs land on different nodes, because `localhost` only refers to the local node. The repo now includes single-job and same-node launch patterns to avoid that failure mode.

### 3. Single-GPU OOM and forced context reduction

The first all-in-one job tried to host both the 4B debater and 8B evaluator on one GPU. That required aggressive memory constraints and eventually caused context-length failures. This path was kept only as an auxiliary script; the stable path is the split-GPU train script.

### 4. Excessively large generation limits in TG-MAD

The early TG-MAD implementation used `max_new_tokens=2000`, which did not match the source debate runs and caused avoidable context pressure in both debate and evaluator prompts. This was corrected by aligning TG-MAD with the existing debate settings:

- `max_new_tokens = 512`
- `temperature = 1.0`
- `top_p = 0.9`

### 5. Over-aggressive attempts to preserve full default model context

I tested uncapped `max_model_len` and multi-GPU tensor-parallel launches. Those runs failed for practical reasons in this environment:

- the 4B model default context in vLLM was far larger than needed for this workload
- multi-GPU slot exports through SLURM command-line environment passing were fragile
- the stable workload did not need default max context to run successfully

The final stable configuration therefore uses `max_model_len=8192` for both local vLLM servers. That setting is large enough for the current TG-MAD prompts and was the first setting that consistently passed startup and sample execution on the available 2-GPU node.

## Code Quality And Review Fixes

During review and iteration, the following correctness and robustness fixes were also applied:

- backend generation failures now fail loudly by default instead of silently substituting fake text
- evaluator prompt construction now supports configurable agent counts instead of assuming exactly three hard-coded slots
- output artifacts now respect `output_dir`
- run metadata now includes configuration fields needed for reproducibility
- evaluator temperature remains fixed to `1.0` to match repository policy

## Validation Performed

The following targeted checks were run successfully:

```bash
python -m compileall tg_mad
bash -n scripts/run_tg_mad_train_2gpu.sh
bash -n scripts/run_tg_mad_eval_1gpu.sh
python scripts/check_tg_mad_smoke.py
```

## Submission History

Multiple SLURM submissions were used to isolate startup, memory, and context issues. The key successful configuration is the 2-GPU train job:

```bash
sbatch scripts/run_tg_mad_train_2gpu.sh
```

The train job submitted on 2026-03-16 as job `342841` reached the following confirmed state:

- both vLLM servers became healthy on `node15`
- Python training started successfully
- batch 0 started
- sample 1 completed successfully
- sample 2 and later samples began processing

This established that the source-aligned TG-MAD training path can run successfully with:

- one GPU for the 4B debater
- one GPU for the 8B evaluator
- `max_new_tokens = 512`
- `max_model_len = 8192`

## Recommended Commands

Prepare the environment:

```bash
INSTALL_VLLM=1 bash scripts/setup_textgrad_env.sh
```

Run training:

```bash
sbatch scripts/run_tg_mad_train_2gpu.sh
```

After `out/tg_mad/prompt_history.json` is written, run evaluation:

```bash
sbatch scripts/run_tg_mad_eval_1gpu.sh
```

## Current Repository State

At the end of this work:

- TG-MAD code is implemented under `tg_mad/`
- SLURM launchers are present for train, eval, and all-in-one execution
- the train path has been tested on the cluster and is running with the stable 2-GPU source-aligned configuration
- evaluation should be run after training finishes and writes the optimized prompt history

## Follow-up Rerun Note

Later on 2026-03-16, the 2-GPU training job `342841` progressed through the first batch of forward passes but failed during `optimizer.step()`. The evaluator request exceeded the current `8192` serving window by one input token:

- input tokens: `7681`
- requested output tokens: `512`
- effective maximum input length at `8192` context: `7680`

This failure happened during the aggregated TextGrad optimization step rather than during debate generation. The next corrective action was to reduce `TRAIN_BATCH_SIZE` in `scripts/run_tg_mad_train_2gpu.sh` from `5` to `2` so the optimizer prompt stays smaller while preserving the same debate and generation settings.

After rerunning with batch size `2`, the same failure still occurred during `optimizer.step()`. The useful timing split from job `342893` was:

- batch forward pass for 2 samples: about `2m39s`
- TextGrad backward pass before optimizer step: about `4m33s`

The root cause remained the same: the optimizer update prompt for the evaluator exceeded the `8192` serving window. To address that more directly, I tested a local evaluator swap from `Qwen3-8B` to `Qwen3-30B-A3B-Instruct-2507`, together with a larger evaluator serving window (`16384`) and lower evaluator concurrency (`max-num-seqs=1`). The TextGrad engine was also updated not to retry deterministic `400 Bad Request` context-length errors.

That 30B-A3B instruct evaluator did not start successfully under the current local vLLM setup. The server failed during model load with CUDA OOM on a single 47 GB GPU. So the practical local default was restored to `Qwen3-8B`, while keeping the useful non-retry behavior for deterministic `400` errors. The 30B-A3B instruct path remains a reasonable quality experiment if more GPU memory, tensor parallelism, quantization, or a remote API backend is available.

## 2026-03-16 30B / 1-Round Rerun

After reviewing job `342898`, the failure reason was still a context overflow during `optimizer.step()`, now at the larger evaluator window:

- input tokens: `15873`
- requested output tokens: `512`
- evaluator context length: `16384`
- maximum allowed input tokens at that setting: `15872`

To reduce optimizer prompt size without compressing the serving window, I made the following backward-compatible changes:

- added `--debate_prompt_mode` to `tg_mad/train.py` and `tg_mad/evaluate.py`
- added `compact_second_round` prompt handling in `tg_mad/forward_pass.py`
- exposed `TRAIN_N_AGENTS`, `TRAIN_N_ROUNDS`, `TRAIN_SIZE`, `DEBATE_PROMPT_MODE`, and `EVALUATOR_ENGINE_MODEL` in `scripts/run_tg_mad_train_2gpu.sh`
- exposed `EVAL_N_AGENTS`, `EVAL_N_ROUNDS`, and `DEBATE_PROMPT_MODE` in `scripts/run_tg_mad_eval_1gpu.sh`
- added a dedicated 3-GPU launcher: `scripts/run_tg_mad_train_30b_3gpu.sh`

The new 30B launcher is configured for:

- batch size `1`
- `t0` plus one debate round (`n_rounds=1`)
- compact second-round debate prompt
- `Qwen/Qwen3-30B-A3B-Instruct-2507` evaluator over 2 GPUs with tensor parallel size `2`
- `16384` max model length for both debater and evaluator
- separate output directory: `out/tg_mad_30b_r1_bs1`

- First submission of `run_tg_mad_train_30b_3gpu.sh` failed immediately because SLURM executed the wrapper from its spool directory, so the relative call to `run_tg_mad_train_2gpu.sh` resolved to a missing path. The wrapper now changes into the repo root and calls the generic launcher by absolute path.

The rerun submitted as job `342916` succeeded past the previous failure point:

- debater healthy on GPU `0`
- 30B evaluator healthy on GPUs `1,2` with tensor parallel size `2`
- batch size `1`
- `n_rounds=1` (`t0` + one debate round)
- `compact_second_round` debate prompt mode
- `16384` max model length for both servers

Most importantly, batch `0` completed its backward pass and `optimizer.step()` successfully. This confirms that the earlier optimizer-step context overflow was resolved by the combined change set rather than only delayed.
