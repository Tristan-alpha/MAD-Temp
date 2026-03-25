"""Data loading and task-pool reconstruction for TG-MAD."""

import json
import os
import re
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset

from tg_mad.config import EXISTING_DATA_PATH, TRAIN_SIZE


# GSM8K answer extraction (mirrors src/data/gsm8k.py)
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

FIXED_POOL_SIZES = {
    "formal_logic": {"train": 19, "eval": 126},
    "hh_rlhf": {"train": 60, "eval": 300},
}


def extract_gsm8k_answer(full_ans_text: str) -> Optional[int]:
    """Extract numeric answer after #### from GSM8K answer field."""
    match = ANS_RE.search(full_ans_text)
    if match:
        match_str = match.group(1).strip().replace(",", "")
        return int(match_str.strip())
    return None


def load_existing_data(path: str = EXISTING_DATA_PATH) -> List[dict]:
    """Load one JSONL history file into a list of dicts."""
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_split_info(path: Optional[str]) -> Optional[dict]:
    if path is None or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_gsm8k_questions(
    data_dir: str = "./datasets",
    data_size: int = 500,
) -> Tuple[List[str], List[int]]:
    """Load GSM8K questions and ground truth answers."""
    dataset = load_dataset("openai/gsm8k", "main", cache_dir=data_dir)["test"]
    dataset = pd.DataFrame(dataset)
    dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True).head(data_size)

    questions, labels = [], []
    for question, answer in zip(dataset["question"], dataset["answer"]):
        questions.append(question)
        labels.append(extract_gsm8k_answer(answer))

    return questions, labels


def load_formal_logic_questions(
    data_dir: str = "./datasets",
    pool: str = "eval",
) -> Tuple[List[str], List[str]]:
    dataset = load_dataset("cais/mmlu", "formal_logic", cache_dir=data_dir)
    if pool == "train":
        split_names = ["validation", "dev"]
    elif pool == "eval":
        split_names = ["test"]
    else:
        raise ValueError(f"Unsupported formal_logic pool: {pool}")

    questions, labels = [], []
    choices = "ABCD"
    template = "{}\n(A) {}\n(B) {}\n(C) {}\n(D) {}\n\n"
    for split_name in split_names:
        split_df = pd.DataFrame(dataset[split_name])
        for query, options, answer in zip(split_df["question"], split_df["choices"], split_df["answer"]):
            if len(options) != 4:
                continue
            question = template.format(query, options[0], options[1], options[2], options[3])
            label = f"({choices[int(answer)]})"
            questions.append(question)
            labels.append(label)

    return questions, labels


def load_hh_rlhf_questions(
    data_dir: str = "./datasets",
    split: str = "test",
    data_size: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    dataset = load_dataset("Anthropic/hh-rlhf", cache_dir=data_dir)[split]
    dataset = pd.DataFrame(dataset)
    dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True)
    if data_size is not None:
        dataset = dataset.head(data_size)

    rng = np.random.RandomState(seed)
    questions, labels = [], []
    template = "Which of the following is relatively more helpful and/or harmless data?\n\n(A) {}\n\n\n(B) {}\n\n"
    for chosen, rejected in zip(dataset["chosen"], dataset["rejected"]):
        if rng.random_sample() > 0.5:
            question = template.format(chosen, rejected)
            label = "(A)"
        else:
            question = template.format(rejected, chosen)
            label = "(B)"
        questions.append(question)
        labels.append(label)

    return questions, labels


def get_fixed_pool_size(dataset: str, pool: str) -> Optional[int]:
    return FIXED_POOL_SIZES.get(dataset, {}).get(pool)


def load_task_questions(
    dataset: str,
    data_dir: str = "./datasets",
    pool: str = "eval",
    data_size: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], List[Any]]:
    if dataset == "gsm8k":
        if data_size is None:
            raise ValueError("gsm8k task loading requires data_size.")
        return load_gsm8k_questions(data_dir=data_dir, data_size=data_size)

    if dataset == "formal_logic":
        questions, labels = load_formal_logic_questions(data_dir=data_dir, pool=pool)
    elif dataset == "hh_rlhf":
        split = "train" if pool == "train" else "test"
        effective_size = data_size
        if effective_size is None:
            effective_size = get_fixed_pool_size(dataset, pool)
        questions, labels = load_hh_rlhf_questions(
            data_dir=data_dir,
            split=split,
            data_size=effective_size,
            seed=seed,
        )
    else:
        raise NotImplementedError(f"Unsupported TG-MAD dataset: {dataset}")

    if data_size is not None and len(questions) != data_size:
        raise ValueError(
            f"{dataset} {pool} pool size mismatch: expected {data_size}, got {len(questions)}"
        )

    expected_size = get_fixed_pool_size(dataset, pool)
    if expected_size is not None and len(questions) != expected_size:
        raise ValueError(
            f"{dataset} {pool} pool must be {expected_size}, got {len(questions)}"
        )
    return questions, labels


def build_samples(
    questions: List[str],
    answers: List[Any],
    existing_data: List[dict],
) -> List[dict]:
    if len(existing_data) != len(questions) or len(questions) != len(answers):
        raise ValueError(
            f"Length mismatch: history={len(existing_data)} questions={len(questions)} answers={len(answers)}"
        )
    return [
        {
            "question": questions[i],
            "ground_truth": answers[i],
            "existing_data": existing_data[i],
            "index": i,
        }
        for i in range(len(existing_data))
    ]


def build_samples_from_history(
    *,
    dataset: str,
    history_path: str,
    data_dir: str = "./datasets",
    pool: str = "eval",
    seed: int = 42,
) -> List[dict]:
    existing_data = load_existing_data(history_path)
    expected_size = get_fixed_pool_size(dataset, pool)
    if expected_size is not None and len(existing_data) != expected_size:
        raise ValueError(
            f"{dataset} {pool} history at {history_path} must contain {expected_size} records, got {len(existing_data)}"
        )
    questions, answers = load_task_questions(
        dataset=dataset,
        data_dir=data_dir,
        pool=pool,
        data_size=len(existing_data),
        seed=seed,
    )
    return build_samples(questions, answers, existing_data)


def build_icl_prompt(
    *,
    dataset: str,
    data_dir: str = "./datasets",
    seed: int = 42,
    icl_mode: str = "qa",
    history_path: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> str:
    """Build an ICL system prompt from the training pool.

    Args:
        dataset: Dataset name (hh_rlhf, formal_logic, gsm8k).
        icl_mode: "qa" for question+answer, "qra" for question+reasoning+answer.
        history_path: Path to train-pool JSONL (required for "qra" mode).
        max_examples: Cap the number of examples (None = use all).

    Returns:
        Formatted system prompt string containing ICL examples.
    """
    if dataset == "gsm8k":
        raise NotImplementedError(
            "gsm8k ICL requires --data_size for train pool; not yet supported."
        )

    questions, labels = load_task_questions(
        dataset=dataset,
        data_dir=data_dir,
        pool="train",
        seed=seed,
    )

    # For qra mode, load correct agent responses from history
    correct_responses: Optional[List[Optional[str]]] = None
    if icl_mode == "qra":
        if history_path is None:
            raise ValueError("--icl_history_path is required for icl_mode='qra'")
        history = load_existing_data(history_path)
        if len(history) != len(questions):
            raise ValueError(
                f"ICL history length ({len(history)}) != train pool size ({len(questions)})"
            )
        correct_responses = []
        for i, (record, label) in enumerate(zip(history, labels)):
            t0 = record.get("0", {})
            responses = t0.get("responses", {})
            iscorr = t0.get("final_answer_iscorr", [])
            agent_names = list(responses.keys())
            found = None
            for j, name in enumerate(agent_names):
                if j < len(iscorr) and iscorr[j]:
                    found = responses[name]
                    break
            correct_responses.append(found)

    # Optionally limit examples
    indices = list(range(len(questions)))
    if max_examples is not None and max_examples < len(indices):
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
        indices = sorted(indices[:max_examples])

    # Build the prompt
    from tg_mad.task_spec import get_task_prompt_label, is_mcq_dataset

    task_label = get_task_prompt_label(dataset)
    if is_mcq_dataset(dataset):
        answer_format = '{final answer: (X)}'
    else:
        answer_format = '{final answer: <number>}'

    header = (
        f"You are participating in a multi-agent {task_label}. "
        f"Below are examples of correctly answered questions from this task. "
        f"Use them to guide your reasoning.\n"
    )

    examples = []
    for idx, i in enumerate(indices):
        q = questions[i]
        label = labels[i]
        if icl_mode == "qra" and correct_responses[i] is not None:
            examples.append(
                f"--- Example {idx + 1} ---\n"
                f"Question:\n{q}\n"
                f"A correct agent's reasoning:\n{correct_responses[i]}\n"
                f"Correct Answer: {label}"
            )
        else:
            examples.append(
                f"--- Example {idx + 1} ---\n"
                f"Question:\n{q}\n"
                f"Correct Answer: {label}"
            )

    footer = (
        f"\nAlways end your response with your answer in the format: "
        f'"{answer_format}".'
    )

    return header + "\n\n".join(examples) + footer


def has_disagreement_at_t0(sample: dict) -> bool:
    """Check if agents disagreed at round 0."""
    answers = sample["0"]["final_answers"]
    valid = [a for a in answers if a != "" and a is not None]
    if len(valid) <= 1:
        return True
    return len(set(valid)) > 1


def select_train_test_split(
    existing_data: List[dict],
    questions: List[str],
    answers: List[Any],
    train_size: int = TRAIN_SIZE,
) -> Tuple[List[dict], List[dict], List[int]]:
    """Select training samples (preferring disagreement) and test samples."""
    if len(existing_data) != len(questions) or len(questions) != len(answers):
        raise ValueError(
            f"Length mismatch: history={len(existing_data)} questions={len(questions)} answers={len(answers)}"
        )

    disagreement_indices = [
        i for i, sample in enumerate(existing_data) if has_disagreement_at_t0(sample)
    ]

    if len(disagreement_indices) >= train_size:
        train_indices = disagreement_indices[:train_size]
    else:
        remaining = [i for i in range(len(existing_data)) if i not in disagreement_indices]
        supplement = remaining[: train_size - len(disagreement_indices)]
        train_indices = disagreement_indices + supplement

    train_indices_set = set(train_indices)
    samples = build_samples(questions, answers, existing_data)
    train_samples = [samples[i] for i in train_indices]
    test_samples = [samples[i] for i in range(len(samples)) if i not in train_indices_set]
    return train_samples, test_samples, train_indices
