"""MAD forward pass using TextGrad computation graph in per-agent mode."""

import logging
from typing import List

import textgrad as tg

from tg_mad.config import N_AGENTS, N_ROUNDS
from tg_mad.task_spec import get_task_spec
from tg_mad.utils import parse_answer, majority_vote, answer_is_correct

try:
    from src.legacy_debate import build_debate_round_messages
except ImportError:
    from legacy_debate import build_debate_round_messages

logger = logging.getLogger("tg_mad")


def _handle_generation_error(
    *,
    agent_idx: int,
    round_idx: int,
    question: str,
    error: Exception,
    allow_failed_generations: bool,
):
    context = (
        f"Agent {agent_idx + 1} failed at round {round_idx} "
        f"for question {question[:80]!r}: {error}"
    )
    if not allow_failed_generations:
        raise RuntimeError(
            context
            + ". This usually means the vLLM server is unavailable or returned an invalid response."
        ) from error

    logger.warning("%s. Continuing because allow_failed_generations=True.", context)
    return tg.Variable(
        f"Agent {agent_idx + 1} failed to respond.",
        requires_grad=False,
        role_description=f"failed response from agent {agent_idx + 1} at round {round_idx}",
    )


def mad_forward_pass(
    question: str,
    ground_truth,
    debater_prompt: List[tg.Variable],
    debater_engine,
    n_agents: int = N_AGENTS,
    n_rounds: int = N_ROUNDS,
    dataset: str = "gsm8k",
    allow_failed_generations: bool = False,
) -> dict:
    """Run a full multi-agent debate using one prompt variable per agent."""
    task = get_task_spec(dataset, n_agents=n_agents)
    answer_suffix = task.answer_suffix
    agent_names = [f"Agent {i + 1}" for i in range(n_agents)]

    if not isinstance(debater_prompt, list):
        raise TypeError(
            "TG-MAD now requires per-agent prompt mode. "
            "Expected a list of prompt Variables, one per agent."
        )
    assert len(debater_prompt) == n_agents, (
        f"Expected {n_agents} per-agent prompts, got {len(debater_prompt)}"
    )
    debater_models = [
        tg.BlackboxLLM(debater_engine, system_prompt=p)
        for p in debater_prompt
    ]

    all_response_vars = []
    # per_agent_response_vars[i] collects all response vars for agent i
    per_agent_response_vars: List[List[tg.Variable]] = [[] for _ in range(n_agents)]
    rounds_data = {}

    # === Round t=0: Independent answers ===
    t0_responses = []
    t0_parsed = []
    t0_vars = []
    for i in range(n_agents):
        q_var = tg.Variable(
            question + answer_suffix,
            requires_grad=False,
            role_description=f"{dataset} problem for agent {i+1} at round 0",
        )
        try:
            resp_var = debater_models[i](q_var)
            t0_responses.append(resp_var.value)
            t0_vars.append(resp_var)
            t0_parsed.append(parse_answer(resp_var.value, dataset=dataset))
        except Exception as e:
            resp_var = _handle_generation_error(
                agent_idx=i,
                round_idx=0,
                question=question,
                error=e,
                allow_failed_generations=allow_failed_generations,
            )
            t0_responses.append(resp_var.value)
            t0_vars.append(resp_var)
            t0_parsed.append(None)

    all_response_vars.extend(t0_vars)
    for i, v in enumerate(t0_vars):
        per_agent_response_vars[i].append(v)
    t0_majority = majority_vote(t0_parsed)

    rounds_data[0] = {
        "answers": t0_responses,
        "parsed": t0_parsed,
        "majority_vote": t0_majority,
        "individual_correct": [
            answer_is_correct(a, ground_truth, dataset=dataset) for a in t0_parsed
        ],
    }

    # === Rounds t=1..T: Debate ===
    prev_responses = t0_responses

    for t in range(1, n_rounds + 1):
        round_responses = []
        round_parsed = []
        round_vars = []
        round_messages = build_debate_round_messages(
            question,
            dict(zip(agent_names, prev_responses)),
            agent_names=agent_names,
            suffix=answer_suffix,
        )

        for i in range(n_agents):
            context_str = round_messages[agent_names[i]]["content"]
            context_var = tg.Variable(
                context_str,
                requires_grad=False,
                role_description=f"{dataset} debate context for agent {i+1} at round {t}",
            )
            try:
                resp_var = debater_models[i](context_var)
                round_responses.append(resp_var.value)
                round_vars.append(resp_var)
                round_parsed.append(parse_answer(resp_var.value, dataset=dataset))
            except Exception as e:
                resp_var = _handle_generation_error(
                    agent_idx=i,
                    round_idx=t,
                    question=question,
                    error=e,
                    allow_failed_generations=allow_failed_generations,
                )
                round_responses.append(resp_var.value)
                round_vars.append(resp_var)
                round_parsed.append(None)

        all_response_vars.extend(round_vars)
        for i, v in enumerate(round_vars):
            per_agent_response_vars[i].append(v)
        round_majority = majority_vote(round_parsed)

        rounds_data[t] = {
            "answers": round_responses,
            "parsed": round_parsed,
            "majority_vote": round_majority,
            "individual_correct": [
                answer_is_correct(a, ground_truth, dataset=dataset) for a in round_parsed
            ],
        }

        prev_responses = round_responses

    final_majority = majority_vote(rounds_data[n_rounds]["parsed"])

    transcript_var = None

    return {
        "question": question,
        "ground_truth": ground_truth,
        "t0_answers": t0_responses,
        "t0_parsed": t0_parsed,
        "t0_majority_vote": t0_majority,
        "rounds": rounds_data,
        "final_majority_vote": final_majority,
        "final_correct": answer_is_correct(final_majority, ground_truth, dataset=dataset),
        "all_response_vars": all_response_vars,
        "per_agent_response_vars": per_agent_response_vars,
        "transcript_var": transcript_var,
    }
