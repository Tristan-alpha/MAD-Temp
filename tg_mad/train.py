"""TG-MAD training loop: optimize debater prompt via TextGrad."""

import argparse
import os
import random
import time

import textgrad as tg

from tg_mad.config import (
    ARTIFACT_SCHEMA_VERSION,
    BATCH_SIZE,
    DEBATER_BASE_URL,
    EVALUATOR_BASE_URL,
    EVALUATOR_TEMPERATURE,
    EXISTING_DATA_PATH,
    NUM_EPOCHS,
    INITIAL_DEBATER_PROMPT,
    MAX_NEW_TOKENS,
    N_AGENTS,
    N_ROUNDS,
    OPTIMIZER_CONSTRAINTS,
    OUTPUT_DIR,
    SEED,
    TEMPERATURE,
    TRAIN_SIZE,
)
from tg_mad.engine import create_debater_engine, create_evaluator_engine, create_api_evaluator_engine
from tg_mad.data_loader import load_existing_data, load_gsm8k_questions, select_train_test_split
from tg_mad.forward_pass import mad_forward_pass
from tg_mad.evaluator import create_evaluator_loss
from tg_mad.utils import (
    append_jsonl_record,
    answer_is_correct,
    build_text_history_manifest,
    init_text_history_file,
    render_transcript_text,
    resolve_artifact_paths,
    resolve_text_history_paths,
    save_json,
    serialize_rounds_for_history,
    set_seeds,
    setup_logging,
)


def parse_args():
    parser = argparse.ArgumentParser(description="TG-MAD Training")
    parser.add_argument("--debater_base_url", type=str, default=None)
    parser.add_argument("--evaluator_base_url", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--train_size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--n_agents", type=int, default=N_AGENTS)
    parser.add_argument("--n_rounds", type=int, default=N_ROUNDS)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--evaluator_max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--prompt_history_file", type=str, default=None)
    parser.add_argument("--split_info_file", type=str, default=None)
    parser.add_argument("--run_config_file", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--existing_data", type=str, default=EXISTING_DATA_PATH)
    parser.add_argument(
        "--save_text_history",
        action="store_true",
        help="Save per-sample debate text to JSONL files under out/history.",
    )
    parser.add_argument(
        "--text_history_dir",
        type=str,
        default=None,
        help="Optional directory for train/eval text history JSONL files.",
    )
    parser.add_argument(
        "--allow_failed_generations",
        action="store_true",
        help="Continue with placeholder text if the debater backend fails.",
    )
    # API evaluator options
    parser.add_argument(
        "--evaluator_type",
        type=str,
        default="local",
        choices=["local", "api"],
        help="Evaluator backend: 'local' for vLLM Qwen3-8B, 'api' for remote API (e.g. kimi-k2.5).",
    )
    parser.add_argument("--evaluator_api_key", type=str, default=None,
                        help="API key for remote evaluator (or set KIMI_API_KEY env var).")
    parser.add_argument("--evaluator_model", type=str, default=None,
                        help="Override evaluator model string (e.g. 'openai/kimi-k2.5').")
    parser.add_argument("--evaluator_api_base_url", type=str, default=None,
                        help="Override evaluator API base URL.")
    return parser.parse_args()


def build_run_config(args, artifact_paths, text_history_paths):
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "output_dir": artifact_paths["output_dir"],
        "prompt_history_file": artifact_paths["prompt_history"],
        "split_info_file": artifact_paths["split_info"],
        "run_config_file": artifact_paths["run_config"],
        "save_text_history": args.save_text_history,
        "text_history_dir": text_history_paths["text_history_dir"],
        "text_history_file": text_history_paths["text_history_file"],
        "debater_base_url": args.debater_base_url or DEBATER_BASE_URL,
        "evaluator_base_url": args.evaluator_base_url or EVALUATOR_BASE_URL,
        "debater_temperature": TEMPERATURE,
        "evaluator_temperature": EVALUATOR_TEMPERATURE,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "train_size": args.train_size,
        "n_agents": args.n_agents,
        "n_rounds": args.n_rounds,
        "max_new_tokens": args.max_new_tokens,
        "evaluator_max_new_tokens": args.evaluator_max_new_tokens,
        "seed": args.seed,
        "data_dir": args.data_dir,
        "existing_data": args.existing_data,
        "allow_failed_generations": args.allow_failed_generations,
        "evaluator_type": args.evaluator_type,
        "evaluator_model": args.evaluator_model,
    }


def build_train_text_history_record(
    *,
    sample,
    result,
    prompt_before_step: str,
    epoch: int,
    batch_idx: int,
    sample_position: int,
    evaluator_feedback: str,
):
    rounds_payload = serialize_rounds_for_history(result["rounds"])
    return {
        "record_type": "sample",
        "stage": "train",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_index": sample["index"],
        "sample_position_in_batch": sample_position,
        "epoch": epoch,
        "batch": batch_idx,
        "question": sample["question"],
        "ground_truth": sample["ground_truth"],
        "prompt_before_step": prompt_before_step,
        "rounds": rounds_payload,
        "t0_majority_vote": result["t0_majority_vote"],
        "final_majority_vote": result["final_majority_vote"],
        "final_correct": result["final_correct"],
        "transcript_text": render_transcript_text(sample["question"], result["rounds"]),
        "evaluator_feedback": evaluator_feedback,
    }


def train(args):
    if args.n_agents < 1:
        raise ValueError("--n_agents must be at least 1")
    if args.n_rounds < 0:
        raise ValueError("--n_rounds must be non-negative")

    artifact_paths = resolve_artifact_paths(
        output_dir=args.output_dir,
        prompt_history_file=args.prompt_history_file,
        split_info_file=args.split_info_file,
        run_config_file=args.run_config_file,
    )
    text_history_paths = resolve_text_history_paths(
        output_dir=args.output_dir,
        existing_data_path=args.existing_data,
        stage="train",
        save_text_history=args.save_text_history,
        text_history_dir=args.text_history_dir,
    )
    logger = setup_logging(artifact_paths["output_dir"], name="tg_mad_train")
    set_seeds(args.seed)
    run_config = build_run_config(args, artifact_paths, text_history_paths)
    if args.save_text_history:
        init_text_history_file(
            text_history_paths["text_history_file"],
            build_text_history_manifest(
                schema_version=ARTIFACT_SCHEMA_VERSION,
                stage="train",
                text_history_paths=text_history_paths,
                output_dir=artifact_paths["output_dir"],
                prompt_history_file=artifact_paths["prompt_history"],
                run_config_file=artifact_paths["run_config"],
                split_info_file=artifact_paths["split_info"],
                run_config=run_config,
            ),
        )
    save_json(run_config, artifact_paths["run_config"])
    if args.save_text_history:
        logger.info(
            "Saving train text history to %s",
            text_history_paths["text_history_file"],
        )

    # === Create engines ===
    logger.info("Creating engines...")
    debater_engine = create_debater_engine(
        base_url=args.debater_base_url,
        max_tokens=args.max_new_tokens,
    )
    if args.evaluator_type == "api":
        logger.info("Using API-based evaluator engine")
        evaluator_engine = create_api_evaluator_engine(
            model=args.evaluator_model,
            base_url=args.evaluator_api_base_url,
            api_key=args.evaluator_api_key,
            max_tokens=args.evaluator_max_new_tokens,
        )
    else:
        logger.info("Using local vLLM evaluator engine")
        evaluator_engine = create_evaluator_engine(
            model=args.evaluator_model,
            base_url=args.evaluator_base_url,
            max_tokens=args.evaluator_max_new_tokens,
        )

    # Set global backward engine for TextGrad
    tg.set_backward_engine(evaluator_engine)

    # === Load data ===
    logger.info("Loading data...")
    existing_data = load_existing_data(args.existing_data)
    questions, answers = load_gsm8k_questions(args.data_dir, data_size=len(existing_data))
    train_samples, test_samples, train_indices = select_train_test_split(
        existing_data, questions, answers, train_size=args.train_size
    )
    logger.info(
        f"Training on {len(train_samples)} samples, "
        f"test set has {len(test_samples)} samples"
    )

    # Save train/test split info
    save_json(
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "train_indices": train_indices,
            "num_test": len(test_samples),
            "run_config": run_config,
        },
        artifact_paths["split_info"],
    )

    # === Create optimizable variable ===
    debater_prompt = tg.Variable(
        value=INITIAL_DEBATER_PROMPT,
        requires_grad=True,
        role_description=(
            "shared system prompt for all 3 debater agents in a "
            "multi-agent debate on math problems"
        ),
    )

    # === Create optimizer ===
    optimizer = tg.TGD(
        parameters=[debater_prompt],
        engine=evaluator_engine,
        constraints=OPTIMIZER_CONSTRAINTS,
    )

    # === Training loop ===
    prompt_history = [
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_config": run_config,
            "epoch": -1,
            "batch": -1,
            "prompt": debater_prompt.value,
            "train_batch_accuracy": None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    ]

    total_steps = 0
    for epoch in range(args.num_epochs):
        random.shuffle(train_samples)

        for batch_start in range(0, len(train_samples), args.batch_size):
            batch = train_samples[batch_start : batch_start + args.batch_size]
            batch_idx = batch_start // args.batch_size
            losses = []
            batch_correct = 0
            prompt_before_step = debater_prompt.value

            logger.info(
                f"=== Epoch {epoch}, Batch {batch_idx} "
                f"({len(batch)} samples) ==="
            )
            logger.info(f"Current prompt (first 200 chars): {prompt_before_step[:200]}...")

            for si, sample in enumerate(batch):
                logger.info(
                    f"  Sample {si+1}/{len(batch)}: "
                    f"Q='{sample['question'][:60]}...' GT={sample['ground_truth']}"
                )

                # Forward pass — runs full T-round debate
                result = mad_forward_pass(
                    question=sample["question"],
                    ground_truth=sample["ground_truth"],
                    debater_prompt=debater_prompt,
                    debater_engine=debater_engine,
                    n_agents=args.n_agents,
                    n_rounds=args.n_rounds,
                    allow_failed_generations=args.allow_failed_generations,
                )

                # Log results
                t0_mv = result["t0_majority_vote"]
                final_mv = result["final_majority_vote"]
                gt = sample["ground_truth"]
                logger.info(
                    f"    t0_parsed={result['t0_parsed']}, "
                    f"t0_MV={t0_mv} ({'correct' if answer_is_correct(t0_mv, gt) else 'wrong'})"
                )
                logger.info(
                    f"    final_MV={final_mv} ({'correct' if result['final_correct'] else 'wrong'})"
                )

                if result["final_correct"]:
                    batch_correct += 1

                # Create per-sample loss
                loss_fn = create_evaluator_loss(
                    ground_truth=sample["ground_truth"],
                    t0_parsed=result["t0_parsed"],
                    t0_majority=result["t0_majority_vote"],
                    evaluator_engine=evaluator_engine,
                )
                loss = loss_fn(result["transcript_var"])
                losses.append(loss)
                if args.save_text_history:
                    append_jsonl_record(
                        text_history_paths["text_history_file"],
                        build_train_text_history_record(
                            sample=sample,
                            result=result,
                            prompt_before_step=prompt_before_step,
                            epoch=epoch,
                            batch_idx=batch_idx,
                            sample_position=si,
                            evaluator_feedback=loss.value,
                        ),
                    )

            # Aggregate and step
            batch_accuracy = batch_correct / len(batch)
            logger.info(
                f"  Batch accuracy: {batch_correct}/{len(batch)} = {batch_accuracy:.2%}"
            )

            logger.info("  Aggregating losses and computing backward pass...")
            total_loss = tg.sum(losses)
            total_loss.backward()

            logger.info("  Running optimizer step...")
            optimizer.step()
            optimizer.zero_grad()

            total_steps += 1
            logger.info(
                f"  Updated prompt (first 200 chars): {debater_prompt.value[:200]}..."
            )

            # Save prompt version
            prompt_history.append(
                {
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "epoch": epoch,
                    "batch": batch_idx,
                    "prompt": debater_prompt.value,
                    "train_batch_accuracy": batch_accuracy,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            save_json(prompt_history, artifact_paths["prompt_history"])

    logger.info(f"\nTraining complete. Total optimizer steps: {total_steps}")
    logger.info(f"Final optimized prompt:\n{debater_prompt.value}")
    logger.info(f"Prompt history saved to {artifact_paths['prompt_history']}")

    return debater_prompt.value, prompt_history


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    optimized_prompt, history = train(args)
