"""Generate legacy-format baseline debate histories for TG-MAD datasets."""

import argparse
import json
import os
from pathlib import Path

from tg_mad.data_loader import get_fixed_pool_size, load_task_questions
from tg_mad.engine import create_debater_engine
from tg_mad.task_spec import get_task_spec
from tg_mad.utils import save_json, set_seeds

try:
    from src.evaluator import evaluate_arithmetics, evaluate_mcq
    from src.legacy_debate import build_debate_round_messages
except ImportError:
    from evaluator import evaluate_arithmetics, evaluate_mcq
    from legacy_debate import build_debate_round_messages


def parse_args():
    parser = argparse.ArgumentParser(description="Generate legacy-format baseline history JSONL")
    parser.add_argument("--dataset", type=str, required=True, choices=["gsm8k", "formal_logic", "hh_rlhf"])
    parser.add_argument("--pool", type=str, required=True, choices=["train", "eval"])
    parser.add_argument("--debater_base_url", type=str, required=True)
    parser.add_argument("--debater_model", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--data_size", type=int, default=None)
    parser.add_argument("--n_agents", type=int, default=5)
    parser.add_argument("--n_rounds", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_failed_generations", action="store_true")
    return parser.parse_args()


def _sanitize_model_label(model_name: str) -> str:
    return model_name.replace("hosted_vllm/", "").replace("/", "_")


def _default_output_path(args, data_size: int) -> str:
    model_label = _sanitize_model_label(args.debater_model)
    return os.path.join(
        "out",
        "history",
        args.dataset,
        f"{args.dataset}_{data_size}__{model_label}_N={args.n_agents}_R={args.n_rounds}.jsonl",
    )


def _evaluate_round(dataset: str, agent_responses: dict, answer):
    if dataset == "gsm8k":
        final_resps, debate_resps, is_corr = evaluate_arithmetics(agent_responses, answer)
        return {
            "responses": agent_responses,
            "final_answers": final_resps,
            "final_answer_iscorr": [pred == answer for pred in final_resps],
            "debate_answer": debate_resps,
            "debate_answer_iscorr": is_corr,
            "answer": answer,
        }

    final_resps, debate_resps, is_corr = evaluate_mcq(agent_responses, answer)
    return {
        "responses": agent_responses,
        "final_answers": final_resps,
        "final_answer_iscorr": [pred == answer for pred in final_resps],
        "debate_answer": debate_resps,
        "debate_answer_iscorr": is_corr,
        "answer": answer,
    }


def _generate_one(
    *,
    engine,
    messages,
    agent_idx: int,
    sample_idx: int,
    allow_failed_generations: bool,
):
    try:
        return engine.generate_messages(messages)
    except Exception as exc:
        if not allow_failed_generations:
            raise RuntimeError(
                f"Agent {agent_idx + 1} failed on sample {sample_idx}: {exc}"
            ) from exc
        return f"Agent {agent_idx + 1} failed to respond."


def generate_history(args):
    pool_size = get_fixed_pool_size(args.dataset, args.pool)
    if args.dataset == "gsm8k":
        if args.data_size is None:
            raise ValueError("gsm8k history generation requires --data_size.")
        data_size = args.data_size
    else:
        data_size = pool_size

    questions, answers = load_task_questions(
        dataset=args.dataset,
        data_dir=args.data_dir,
        pool=args.pool,
        data_size=data_size,
        seed=args.seed,
    )
    if data_size is None:
        data_size = len(questions)
    output_path = os.path.abspath(args.output_path or _default_output_path(args, data_size))
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_path):
        raise FileExistsError(
            f"History file already exists: {output_path}. Choose a new output_path."
        )

    task_spec = get_task_spec(args.dataset, n_agents=args.n_agents)
    debater_engine = create_debater_engine(
        model=args.debater_model,
        base_url=args.debater_base_url,
        max_tokens=args.max_new_tokens,
    )

    model_label = _sanitize_model_label(args.debater_model)
    agent_names = [
        f"{args.dataset}_{data_size}__{model_label}__None__Agent{i+1}"
        for i in range(args.n_agents)
    ]

    histories = []
    for sample_idx, (question, answer) in enumerate(zip(questions, answers)):
        set_seeds(args.seed + sample_idx)

        round_zero_messages = [
            {"role": "user", "content": question + task_spec.answer_suffix}
        ]
        responses = [
            _generate_one(
                engine=debater_engine,
                messages=round_zero_messages,
                agent_idx=agent_idx,
                sample_idx=sample_idx,
                allow_failed_generations=args.allow_failed_generations,
            )
            for agent_idx in range(args.n_agents)
        ]
        agent_responses = dict(zip(agent_names, responses))
        rounds_data = {"0": _evaluate_round(args.dataset, agent_responses, answer)}

        prev_responses = agent_responses
        for round_idx in range(1, args.n_rounds + 1):
            round_messages = build_debate_round_messages(
                question,
                prev_responses,
                suffix=task_spec.answer_suffix,
            )
            responses = []
            for agent_idx, agent_name in enumerate(agent_names):
                responses.append(
                    _generate_one(
                        engine=debater_engine,
                        messages=[round_messages[agent_name]],
                        agent_idx=agent_idx,
                        sample_idx=sample_idx,
                        allow_failed_generations=args.allow_failed_generations,
                    )
                )
            agent_responses = dict(zip(agent_names, responses))
            rounds_data[str(round_idx)] = _evaluate_round(args.dataset, agent_responses, answer)
            prev_responses = agent_responses

        histories.append(rounds_data)

    with open(output_path, "w") as f:
        for record in histories:
            f.write(json.dumps(record) + "\n")

    metadata = {
        "dataset": args.dataset,
        "pool": args.pool,
        "output_path": output_path,
        "debater_model": args.debater_model,
        "debater_base_url": args.debater_base_url,
        "n_agents": args.n_agents,
        "n_rounds": args.n_rounds,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "num_samples": len(histories),
    }
    save_json(metadata, str(Path(output_path).with_suffix(".meta.json")))
    print(f"Wrote {len(histories)} records to {output_path}")
    print(f"Metadata: {Path(output_path).with_suffix('.meta.json')}")


if __name__ == "__main__":
    generate_history(parse_args())
