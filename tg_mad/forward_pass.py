"""MAD forward pass using TextGrad computation graph.

Each agent call goes through tg.BlackboxLLM so that the shared
debater_prompt Variable is registered as a predecessor, enabling
gradient flow via loss.backward().
"""

import logging
from typing import List, Optional

import textgrad as tg

from tg_mad.config import N_AGENTS, N_ROUNDS, ANSWER_SUFFIX
from tg_mad.utils import parse_answer, majority_vote, answer_is_correct

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


def format_debate_context(
    question: str,
    prev_responses: List[str],
    own_response: str,
    agent_idx: int,
    answer_suffix: str = ANSWER_SUFFIX,
) -> str:
    """Format debate context for rounds t>=1.

    Similar to src/main.py get_new_message() (lines 81-98).
    Each agent sees all OTHER agents' previous responses + their own.
    """
    msg = "These are the recent opinions from other agents: "
    for i, resp in enumerate(prev_responses):
        if i != agent_idx:
            msg += f"\n\nOne of the agents' response: \n{resp}\n"
    msg += f"\n\nThis was your most recent opinion:\n{own_response}\n"
    msg += (
        f"\n\nUse these opinions carefully as additional advice to revise "
        f"your recent opinion to give your final answer to the question:\n{question}"
    )
    msg += answer_suffix
    return msg


def mad_forward_pass(
    question: str,
    ground_truth,
    debater_prompt: tg.Variable,
    debater_engine,
    n_agents: int = N_AGENTS,
    n_rounds: int = N_ROUNDS,
    answer_suffix: str = ANSWER_SUFFIX,
    allow_failed_generations: bool = False,
) -> dict:
    """Run a full multi-agent debate using TextGrad's BlackboxLLM.

    The debater_prompt Variable (requires_grad=True) is shared as the
    system_prompt for all agent calls. BlackboxLLM → LLMCall.forward()
    registers it as a predecessor of each response Variable, enabling
    gradient flow during backward().

    Returns dict with:
        question, ground_truth,
        t0_answers (raw), t0_parsed, t0_majority_vote,
        rounds (per-round data),
        final_majority_vote,
        all_response_vars (list of tg.Variable for transcript),
        transcript_var (tg.sum of all responses)
    """
    # All agents share the same BlackboxLLM instance → same debater_prompt
    debater_model = tg.BlackboxLLM(debater_engine, system_prompt=debater_prompt)

    all_response_vars = []
    rounds_data = {}

    # === Round t=0: Independent answers ===
    t0_responses = []
    t0_parsed = []
    t0_vars = []
    for i in range(n_agents):
        q_var = tg.Variable(
            question + answer_suffix,
            requires_grad=False,
            role_description=f"math problem for agent {i+1} at round 0",
        )
        try:
            resp_var = debater_model(q_var)
            t0_responses.append(resp_var.value)
            t0_vars.append(resp_var)
            t0_parsed.append(parse_answer(resp_var.value))
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
    t0_majority = majority_vote(t0_parsed)

    rounds_data[0] = {
        "answers": t0_responses,
        "parsed": t0_parsed,
        "majority_vote": t0_majority,
        "individual_correct": [answer_is_correct(a, ground_truth) for a in t0_parsed],
    }

    # === Rounds t=1..T: Debate ===
    prev_responses = t0_responses

    for t in range(1, n_rounds + 1):
        round_responses = []
        round_parsed = []
        round_vars = []

        for i in range(n_agents):
            context_str = format_debate_context(
                question, prev_responses, prev_responses[i], i, answer_suffix
            )
            context_var = tg.Variable(
                context_str,
                requires_grad=False,
                role_description=f"debate context for agent {i+1} at round {t}",
            )
            try:
                resp_var = debater_model(context_var)
                round_responses.append(resp_var.value)
                round_vars.append(resp_var)
                round_parsed.append(parse_answer(resp_var.value))
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
        round_majority = majority_vote(round_parsed)

        rounds_data[t] = {
            "answers": round_responses,
            "parsed": round_parsed,
            "majority_vote": round_majority,
            "individual_correct": [
                answer_is_correct(a, ground_truth) for a in round_parsed
            ],
        }

        prev_responses = round_responses

    # === Build transcript Variable for TextLoss ===
    # tg.sum concatenates all response Variables and tracks predecessors
    # Each response_var has debater_prompt as predecessor via BlackboxLLM
    transcript_var = tg.sum(all_response_vars)

    final_majority = majority_vote(rounds_data[n_rounds]["parsed"])

    return {
        "question": question,
        "ground_truth": ground_truth,
        "t0_answers": t0_responses,
        "t0_parsed": t0_parsed,
        "t0_majority_vote": t0_majority,
        "rounds": rounds_data,
        "final_majority_vote": final_majority,
        "final_correct": answer_is_correct(final_majority, ground_truth),
        "all_response_vars": all_response_vars,
        "transcript_var": transcript_var,
    }
