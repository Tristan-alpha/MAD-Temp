"""Data loading: JSONL parsing, GSM8K question loading, train/test splitting."""

import json
import re
from typing import List, Tuple, Optional

import pandas as pd
from datasets import load_dataset

from tg_mad.config import EXISTING_DATA_PATH, TRAIN_SIZE


# GSM8K answer extraction (mirrors src/data/gsm8k.py)
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_gsm8k_answer(full_ans_text: str) -> Optional[int]:
    """Extract numeric answer after #### from GSM8K answer field."""
    match = ANS_RE.search(full_ans_text)
    if match:
        match_str = match.group(1).strip().replace(",", "")
        return int(match_str.strip())
    return None


def load_existing_data(path: str = EXISTING_DATA_PATH) -> List[dict]:
    """Load the 500-sample JSONL file into a list of dicts.

    Each dict has keys "0", "1", "2", "3" (round numbers as strings).
    Each round has: responses, final_answers, final_answer_iscorr,
    debate_answer, debate_answer_iscorr, answer.
    """
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_gsm8k_questions(
    data_dir: str = "./datasets", data_size: int = 500
) -> Tuple[List[str], List[int]]:
    """Load GSM8K questions and ground truth answers.

    Uses random_state=0 shuffle to match existing JSONL ordering
    (mirrors src/data/gsm8k.py lines 16-31).
    """
    dataset = load_dataset("openai/gsm8k", "main", cache_dir=data_dir)["test"]
    dataset = pd.DataFrame(dataset)
    dataset = (
        dataset.sample(frac=1, random_state=0).reset_index(drop=True).head(data_size)
    )

    questions, labels = [], []
    for question, answer in zip(dataset["question"], dataset["answer"]):
        label = extract_gsm8k_answer(answer)
        questions.append(question)
        labels.append(label)

    return questions, labels


def has_disagreement_at_t0(sample: dict) -> bool:
    """Check if agents disagreed at round 0.

    Returns True if not all agents produced the same valid answer.
    """
    answers = sample["0"]["final_answers"]
    valid = [a for a in answers if a != "" and a is not None]
    if len(valid) <= 1:
        # 0 or 1 valid answers = can't determine agreement
        return True
    return len(set(valid)) > 1


def select_train_test_split(
    existing_data: List[dict],
    questions: List[str],
    answers: List[int],
    train_size: int = TRAIN_SIZE,
) -> Tuple[List[dict], List[dict], List[int]]:
    """Select training samples (preferring disagreement) and test samples.

    Returns:
        train_samples: list of {"question": str, "ground_truth": int, "existing_data": dict, "index": int}
        test_samples: same format
        train_indices: list of indices used for training
    """
    assert len(existing_data) == len(questions) == len(answers)

    # Find disagreement samples
    disagreement_indices = [
        i for i, sample in enumerate(existing_data) if has_disagreement_at_t0(sample)
    ]

    # Select training indices: prefer disagreement, supplement if needed
    if len(disagreement_indices) >= train_size:
        train_indices = disagreement_indices[:train_size]
    else:
        remaining = [
            i for i in range(len(existing_data)) if i not in disagreement_indices
        ]
        supplement = remaining[: train_size - len(disagreement_indices)]
        train_indices = disagreement_indices + supplement

    train_indices_set = set(train_indices)

    def make_sample(i):
        return {
            "question": questions[i],
            "ground_truth": answers[i],
            "existing_data": existing_data[i],
            "index": i,
        }

    train_samples = [make_sample(i) for i in train_indices]
    test_samples = [
        make_sample(i) for i in range(len(existing_data)) if i not in train_indices_set
    ]

    return train_samples, test_samples, train_indices
