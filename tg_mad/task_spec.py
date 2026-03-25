"""Task-specific prompt, parsing, and optimization helpers for TG-MAD."""

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from tg_mad.config import (
    INITIAL_DEBATER_PROMPT,
    INITIAL_DEBATER_PROMPTS,
    OPTIMIZER_CONSTRAINTS,
    OPTIMIZER_CONSTRAINTS_PER_AGENT,
)

try:
    from src.evaluator import (
        MCQ_DATASETS,
        extract_mcq_final_answer,
        extract_numeric_final_answer,
        get_instruction_suffix_for_data,
    )
except ImportError:
    from evaluator import (
        MCQ_DATASETS,
        extract_mcq_final_answer,
        extract_numeric_final_answer,
        get_instruction_suffix_for_data,
    )


NUMERIC_DATASETS = {"gsm8k", "arithmetics"}


@dataclass(frozen=True)
class TaskSpec:
    dataset: str
    answer_mode: str
    answer_suffix: str
    shared_initial_prompt: str
    per_agent_initial_prompts: List[str]
    shared_constraints: List[str]
    per_agent_constraints: List[str]


def is_mcq_dataset(dataset: str) -> bool:
    return dataset in set(MCQ_DATASETS)


def get_task_prompt_label(dataset: str) -> str:
    if dataset == "formal_logic":
        return "formal logic debate"
    if dataset == "hh_rlhf":
        return "helpfulness and harmlessness preference debate"
    if dataset == "gsm8k":
        return "math debate"
    return f"{dataset} debate"


def _mcq_constraints(dataset: str) -> tuple[List[str], List[str]]:
    task_label = get_task_prompt_label(dataset)
    shared = [
        "The prompt must always instruct agents to end responses with: '{final answer: (X)}'.",
        f"The prompt must be a system prompt for a {task_label} agent.",
        "The prompt must not exceed 500 words.",
    ]
    per_agent = [
        "The prompt must always instruct the agent to end responses with: '{final answer: (X)}'.",
        f"The prompt must be a system prompt for a {task_label} agent.",
        "The prompt MUST NOT exceed 80 words. Be concise.",
    ]
    return shared, per_agent


def get_task_spec(dataset: str, n_agents: int) -> TaskSpec:
    answer_suffix = get_instruction_suffix_for_data(dataset, bae=False, cot=False)
    if dataset == "gsm8k":
        return TaskSpec(
            dataset=dataset,
            answer_mode="numeric",
            answer_suffix=answer_suffix,
            shared_initial_prompt=INITIAL_DEBATER_PROMPT,
            per_agent_initial_prompts=[
                INITIAL_DEBATER_PROMPTS[i % len(INITIAL_DEBATER_PROMPTS)]
                for i in range(n_agents)
            ],
            shared_constraints=list(OPTIMIZER_CONSTRAINTS),
            per_agent_constraints=list(OPTIMIZER_CONSTRAINTS_PER_AGENT),
        )

    if is_mcq_dataset(dataset):
        shared_constraints, per_agent_constraints = _mcq_constraints(dataset)
        return TaskSpec(
            dataset=dataset,
            answer_mode="mcq",
            answer_suffix=answer_suffix,
            shared_initial_prompt="",
            per_agent_initial_prompts=[""] * n_agents,
            shared_constraints=shared_constraints,
            per_agent_constraints=per_agent_constraints,
        )

    raise NotImplementedError(f"Unsupported TG-MAD dataset: {dataset}")


def parse_prediction(response: str, dataset: str) -> Optional[Any]:
    if dataset in NUMERIC_DATASETS:
        parsed = extract_numeric_final_answer(response)
        return None if parsed == "" else float(parsed)
    if is_mcq_dataset(dataset):
        parsed = extract_mcq_final_answer(response)
        return None if parsed == "" else parsed
    raise NotImplementedError(f"Unsupported TG-MAD dataset: {dataset}")


def normalize_stored_answer(answer: Any, dataset: str) -> Optional[Any]:
    if answer in ("", None):
        return None
    if dataset in NUMERIC_DATASETS:
        return float(np.round(float(answer), 1))
    if is_mcq_dataset(dataset):
        return str(answer)
    raise NotImplementedError(f"Unsupported TG-MAD dataset: {dataset}")


def answer_is_correct(predicted: Any, ground_truth: Any, dataset: str) -> bool:
    if predicted is None:
        return False
    if dataset in NUMERIC_DATASETS:
        try:
            return abs(float(predicted) - float(np.round(ground_truth, 1))) < 0.01
        except (ValueError, TypeError):
            return False
    if is_mcq_dataset(dataset):
        return str(predicted) == str(ground_truth)
    raise NotImplementedError(f"Unsupported TG-MAD dataset: {dataset}")
