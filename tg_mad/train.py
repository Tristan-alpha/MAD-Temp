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
    MAX_NEW_TOKENS,
    N_AGENTS,
    N_ROUNDS,
    NUM_EPOCHS,
    OUTPUT_DIR,
    SEED,
    TEMPERATURE,
    TRAIN_SIZE,
)
from tg_mad.engine import create_debater_engine, create_evaluator_engine, create_api_evaluator_engine
from tg_mad.data_loader import (
    build_samples_from_history,
    load_existing_data,
    load_gsm8k_questions,
    select_train_test_split,
)
from tg_mad.forward_pass import mad_forward_pass
from tg_mad.evaluator import create_evaluator_loss, create_per_agent_evaluator
from tg_mad.task_spec import get_task_spec
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
    parser.add_argument("--debater_model", type=str, default=None)
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
    parser.add_argument("--dataset", type=str, default="hh_rlhf")
    parser.add_argument("--existing_data", type=str, default=EXISTING_DATA_PATH)
    parser.add_argument("--train_existing_data", type=str, default=None)
    parser.add_argument("--eval_existing_data", type=str, default=None)
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
    # Per-agent prompt optimization
    parser.add_argument(
        "--per_agent_prompts",
        action="store_true",
        default=False,
        help="Optimize a separate prompt per agent instead of one shared prompt.",
    )
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
        "debater_model": args.debater_model,
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
        "dataset": args.dataset,
        "existing_data": args.existing_data,
        "train_existing_data": args.train_existing_data,
        "eval_existing_data": args.eval_existing_data,
        "allow_failed_generations": args.allow_failed_generations,
        "evaluator_type": args.evaluator_type,
        "evaluator_model": args.evaluator_model,
        "per_agent_prompts": args.per_agent_prompts,
    }


def build_train_text_history_record(
    *,
    sample,
    result,
    prompt_before_step,
    epoch: int,
    batch_idx: int,
    sample_position: int,
    evaluator_feedback,
):
    """Build a per-sample text history record.

    ``prompt_before_step`` can be a single string (shared mode) or a list of
    strings (per-agent mode).  ``evaluator_feedback`` can be a string or a list.
    """
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


def _load_training_samples(args):
    if args.dataset == "gsm8k" and args.train_existing_data is None and args.eval_existing_data is None:
        existing_data = load_existing_data(args.existing_data)
        questions, answers = load_gsm8k_questions(args.data_dir, data_size=len(existing_data))
        train_samples, test_samples, train_indices = select_train_test_split(
            existing_data,
            questions,
            answers,
            train_size=args.train_size,
        )
        split_info = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "dataset": args.dataset,
            "split_strategy": "gsm8k_disagreement_split",
            "train_indices": train_indices,
            "num_test": len(test_samples),
        }
        return train_samples, split_info

    if args.dataset != "gsm8k" and args.train_existing_data is None:
        raise ValueError(
            f"{args.dataset} training requires --train_existing_data. "
            "New non-gsm8k datasets must use the split train/eval history interface."
        )

    train_history_path = args.train_existing_data or args.existing_data
    if train_history_path is None:
        raise ValueError(
            f"{args.dataset} training requires --train_existing_data (or legacy --existing_data)."
        )

    train_samples = build_samples_from_history(
        dataset=args.dataset,
        history_path=train_history_path,
        data_dir=args.data_dir,
        pool="train",
        seed=args.seed,
    )
    eval_pool_size = None
    if args.eval_existing_data:
        eval_pool_size = len(load_existing_data(args.eval_existing_data))
    split_info = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "dataset": args.dataset,
        "split_strategy": "explicit_train_eval_histories",
        "train_existing_data": os.path.abspath(train_history_path),
        "eval_existing_data": (
            os.path.abspath(args.eval_existing_data)
            if args.eval_existing_data is not None
            else None
        ),
        "train_pool_size": len(train_samples),
        "eval_pool_size": eval_pool_size,
    }
    return train_samples, split_info


def train(args):
    if args.n_agents < 1:
        raise ValueError("--n_agents must be at least 1")
    if args.n_rounds < 0:
        raise ValueError("--n_rounds must be non-negative")
    task_spec = get_task_spec(args.dataset, n_agents=args.n_agents)

    artifact_paths = resolve_artifact_paths(
        output_dir=args.output_dir,
        prompt_history_file=args.prompt_history_file,
        split_info_file=args.split_info_file,
        run_config_file=args.run_config_file,
    )
    text_history_paths = resolve_text_history_paths(
        output_dir=args.output_dir,
        existing_data_path=args.train_existing_data or args.existing_data,
        stage="train",
        save_text_history=args.save_text_history,
        text_history_dir=args.text_history_dir,
        dataset=args.dataset,
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
        model=args.debater_model,
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
    train_samples, split_info = _load_training_samples(args)
    logger.info(
        "Training on %d samples using dataset=%s (%s)",
        len(train_samples),
        args.dataset,
        split_info["split_strategy"],
    )

    # Save train/test split info
    split_info["run_config"] = run_config
    save_json(split_info, artifact_paths["split_info"])

    # === Create optimizable variable(s) and optimizer(s) ===
    if args.per_agent_prompts:
        logger.info("Per-agent prompt mode: creating %d independent prompts", args.n_agents)
        debater_prompts = [
            tg.Variable(
                value=task_spec.per_agent_initial_prompts[i],
                requires_grad=True,
                role_description=f"system prompt for debate agent {i + 1}",
            )
            for i in range(args.n_agents)
        ]
        optimizers = [
            tg.TGD(
                parameters=[p],
                engine=evaluator_engine,
                constraints=task_spec.per_agent_constraints,
            )
            for p in debater_prompts
        ]
        prompt_history = [
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "run_config": run_config,
                "epoch": -1,
                "batch": -1,
                "prompts": [p.value for p in debater_prompts],
                "train_batch_accuracy": None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        ]
    else:
        debater_prompt = tg.Variable(
            value=task_spec.shared_initial_prompt,
            requires_grad=True,
            role_description=(
                f"shared system prompt for all {args.n_agents} debater agents "
                f"in a multi-agent {args.dataset} debate"
            ),
        )
        optimizer = tg.TGD(
            parameters=[debater_prompt],
            engine=evaluator_engine,
            constraints=task_spec.shared_constraints,
        )
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

    # === Training loop ===
    total_steps = 0
    for epoch in range(args.num_epochs):
        random.shuffle(train_samples)

        for batch_start in range(0, len(train_samples), args.batch_size):
            batch = train_samples[batch_start : batch_start + args.batch_size]
            batch_idx = batch_start // args.batch_size
            batch_correct = 0

            if args.per_agent_prompts:
                prompts_before_step = [p.value for p in debater_prompts]
            else:
                prompt_before_step = debater_prompt.value

            logger.info(
                f"=== Epoch {epoch}, Batch {batch_idx} "
                f"({len(batch)} samples) ==="
            )
            if args.per_agent_prompts:
                for i, p in enumerate(debater_prompts):
                    logger.info(f"  Agent {i+1} prompt (first 120 chars): {p.value[:120]}...")
            else:
                logger.info(f"Current prompt (first 200 chars): {prompt_before_step[:200]}...")

            if args.per_agent_prompts:
                # --- Per-agent prompt training path ---
                for si, sample in enumerate(batch):
                    logger.info(
                        f"  Sample {si+1}/{len(batch)}: "
                        f"Q='{sample['question'][:60]}...' GT={sample['ground_truth']}"
                    )

                    result = mad_forward_pass(
                        question=sample["question"],
                        ground_truth=sample["ground_truth"],
                        debater_prompt=debater_prompts,
                        debater_engine=debater_engine,
                        n_agents=args.n_agents,
                        n_rounds=args.n_rounds,
                        dataset=args.dataset,
                        allow_failed_generations=args.allow_failed_generations,
                    )

                    t0_mv = result["t0_majority_vote"]
                    final_mv = result["final_majority_vote"]
                    gt = sample["ground_truth"]
                    logger.info(
                        f"    t0_parsed={result['t0_parsed']}, "
                        f"t0_MV={t0_mv} ({'correct' if answer_is_correct(t0_mv, gt, dataset=args.dataset) else 'wrong'})"
                    )
                    logger.info(
                        f"    final_MV={final_mv} ({'correct' if result['final_correct'] else 'wrong'})"
                    )
                    if result["final_correct"]:
                        batch_correct += 1

                    # Build transcript text for evaluator (plain string, not tg.Variable)
                    transcript_text = render_transcript_text(
                        sample["question"], result["rounds"]
                    )

                    # Get per-agent feedback (one evaluator call → N feedback strings)
                    evaluate_fn = create_per_agent_evaluator(
                        ground_truth=sample["ground_truth"],
                        t0_parsed=result["t0_parsed"],
                        t0_majority=result["t0_majority_vote"],
                        evaluator_engine=evaluator_engine,
                        n_agents=args.n_agents,
                        dataset=args.dataset,
                    )
                    agent_feedbacks = evaluate_fn(transcript_text)

                    # Accumulate per-agent gradients onto prompt Variables.
                    # We skip backward() entirely because:
                    #   1. backward(engine=) raises when global engine is set
                    #   2. backward() clears self.gradients on the root var
                    # TGD.step() reads parameter.get_gradient_text(), so injecting
                    # here is sufficient — the optimizer sees the routed feedback.
                    # Gradients accumulate across all samples in the batch;
                    # optimizer.step() runs once after the batch loop.
                    for i, (prompt, feedback) in enumerate(
                        zip(debater_prompts, agent_feedbacks)
                    ):
                        grad = tg.Variable(
                            feedback,
                            requires_grad=False,
                            role_description=(
                                f"evaluator feedback for agent {i + 1}'s system prompt"
                            ),
                        )
                        prompt.gradients.add(grad)
                        logger.info(
                            f"    Agent {i+1} feedback (first 100 chars): {feedback[:100]}..."
                        )

                    if args.save_text_history:
                        append_jsonl_record(
                            text_history_paths["text_history_file"],
                            build_train_text_history_record(
                                sample=sample,
                                result=result,
                                prompt_before_step=prompts_before_step,
                                epoch=epoch,
                                batch_idx=batch_idx,
                                sample_position=si,
                                evaluator_feedback=agent_feedbacks,
                            ),
                        )

                # Per-agent optimizer step after all samples in the batch
                batch_accuracy = batch_correct / len(batch)
                logger.info(
                    f"  Batch accuracy: {batch_correct}/{len(batch)} = {batch_accuracy:.2%}"
                )
                logger.info("  Running per-agent optimizer steps...")
                for i, opt in enumerate(optimizers):
                    opt.step()
                    opt.zero_grad()

                total_steps += 1
                for i, p in enumerate(debater_prompts):
                    logger.info(f"  Agent {i+1} updated prompt (first 120 chars): {p.value[:120]}...")

                prompt_history.append(
                    {
                        "schema_version": ARTIFACT_SCHEMA_VERSION,
                        "epoch": epoch,
                        "batch": batch_idx,
                        "prompts": [p.value for p in debater_prompts],
                        "train_batch_accuracy": batch_accuracy,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
                save_json(prompt_history, artifact_paths["prompt_history"])

            else:
                # --- Shared prompt training path (original) ---
                losses = []
                for si, sample in enumerate(batch):
                    logger.info(
                        f"  Sample {si+1}/{len(batch)}: "
                        f"Q='{sample['question'][:60]}...' GT={sample['ground_truth']}"
                    )

                    result = mad_forward_pass(
                        question=sample["question"],
                        ground_truth=sample["ground_truth"],
                        debater_prompt=debater_prompt,
                        debater_engine=debater_engine,
                        n_agents=args.n_agents,
                        n_rounds=args.n_rounds,
                        dataset=args.dataset,
                        allow_failed_generations=args.allow_failed_generations,
                    )

                    t0_mv = result["t0_majority_vote"]
                    final_mv = result["final_majority_vote"]
                    gt = sample["ground_truth"]
                    logger.info(
                        f"    t0_parsed={result['t0_parsed']}, "
                        f"t0_MV={t0_mv} ({'correct' if answer_is_correct(t0_mv, gt, dataset=args.dataset) else 'wrong'})"
                    )
                    logger.info(
                        f"    final_MV={final_mv} ({'correct' if result['final_correct'] else 'wrong'})"
                    )
                    if result["final_correct"]:
                        batch_correct += 1

                    loss_fn = create_evaluator_loss(
                        ground_truth=sample["ground_truth"],
                        t0_parsed=result["t0_parsed"],
                        t0_majority=result["t0_majority_vote"],
                        evaluator_engine=evaluator_engine,
                        dataset=args.dataset,
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
    if args.per_agent_prompts:
        for i, p in enumerate(debater_prompts):
            logger.info(f"Final agent {i+1} prompt:\n{p.value}")
    else:
        logger.info(f"Final optimized prompt:\n{debater_prompt.value}")
    logger.info(f"Prompt history saved to {artifact_paths['prompt_history']}")

    if args.per_agent_prompts:
        return [p.value for p in debater_prompts], prompt_history
    return debater_prompt.value, prompt_history


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    optimized_prompt, history = train(args)
