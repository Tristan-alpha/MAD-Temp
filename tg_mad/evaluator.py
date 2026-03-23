"""TextLoss evaluator for TG-MAD.

Creates per-sample TextLoss functions that embed ground truth and t=0 context
in the evaluator system prompt. The evaluator LLM generates textual gradients
targeting the debater_prompt Variable.

Also provides a per-agent evaluator that returns structured, routed feedback
for independent per-agent prompt optimization.
"""

import logging
import re
from typing import Callable, List, Optional

import textgrad as tg

from tg_mad.config import N_AGENTS

logger = logging.getLogger("tg_mad")


def create_evaluator_loss(
    ground_truth,
    t0_parsed: List[Optional[float]],
    t0_majority: Optional[float],
    evaluator_engine,
) -> tg.TextLoss:
    """Create a TextLoss that evaluates the debate transcript against ground truth.

    The evaluator prompt is constructed per-sample with ground truth and t=0
    answers baked in. It instructs the evaluator to generate feedback on the
    debater SYSTEM PROMPT (not the responses themselves).

    Generates gradients for BOTH correct and incorrect outcomes:
    - Incorrect: harsh critique about echo chamber / martingale stagnation
    - Correct: positive feedback reinforcing good debate behaviors

    The eval_system_prompt has requires_grad=False, so gradients flow only
    through transcript_var → response_vars → debater_prompt.

    Label leakage note: ground truth is used ONLY during training for gradient
    computation, never at inference/evaluation time.
    """
    t0_strs = [str(a) if a is not None else "FAILED" for a in t0_parsed]
    agent_lines = "\n".join(
        f"Agent {idx + 1}: {answer}" for idx, answer in enumerate(t0_strs)
    )

    evaluator_prompt = f"""You are evaluating a multi-agent math debate system. Your job is to generate feedback on the SYSTEM PROMPT used by the debater agents, so it can be improved.

GROUND TRUTH ANSWER: {ground_truth}

INDEPENDENT ANSWERS AT t=0 (before debate):
{agent_lines}
Majority Vote at t=0: {t0_majority}

You will receive the full debate transcript. Analyze it and provide feedback on the debater system prompt.

YOUR ANALYSIS MUST CHECK FOR:

1. ECHO CHAMBER EFFECT: Did any agent have the CORRECT answer at t=0 but then change to an INCORRECT answer during debate because the majority pressured them? If so, generate a HARSH critique: the system prompt must be updated to forbid changing answers based on peer consensus alone — require explicit mathematical/logical proof before conceding.

2. MARTINGALE STAGNATION: Did the debate rounds fail to improve accuracy? Did agents just repeat their positions without meaningful engagement? If so, critique the prompt for failing to encourage productive mathematical reasoning exchange.

3. SUCCESSFUL CORRECTION (POSITIVE FEEDBACK): Did the debate successfully correct an initially wrong majority? Did a minority agent with the correct answer successfully convince others through rigorous reasoning? If so, generate POSITIVE feedback: identify what aspects of the prompt encouraged this good behavior and recommend reinforcing them.

4. SUCCESSFUL DEFENSE (POSITIVE FEEDBACK): Did an agent with the correct answer successfully resist pressure from an incorrect majority? If so, generate positive feedback praising the prompt's encouragement of evidence-based reasoning.

Be specific. Reference exact moments in the transcript. Your feedback will be used to update the system prompt via gradient descent."""

    return tg.TextLoss(evaluator_prompt, engine=evaluator_engine)


def _parse_per_agent_feedback(raw: str, n_agents: int) -> List[str]:
    """Parse evaluator output into per-agent feedback sections.

    Expects sections delimited by ``[AGENT_N_FEEDBACK]`` markers.
    Falls back to duplicating the full text if parsing fails.
    """
    pattern = r"\[AGENT_(\d+)_FEEDBACK\]"
    splits = re.split(pattern, raw)

    # splits should be: [preamble, "1", text1, "2", text2, "3", text3, ...]
    feedbacks: dict[int, str] = {}
    for i in range(1, len(splits) - 1, 2):
        try:
            idx = int(splits[i])
            feedbacks[idx] = splits[i + 1].strip()
        except (ValueError, IndexError):
            continue

    if all((i + 1) in feedbacks for i in range(n_agents)):
        return [feedbacks[i + 1] for i in range(n_agents)]

    # Fallback: duplicate full feedback to all agents
    logger.warning(
        "Could not parse %d agent sections (found %d). "
        "Duplicating full feedback to all agents.",
        n_agents,
        len(feedbacks),
    )
    return [raw.strip()] * n_agents


def create_per_agent_evaluator(
    ground_truth,
    t0_parsed: List[Optional[float]],
    t0_majority: Optional[float],
    evaluator_engine,
    n_agents: int = N_AGENTS,
) -> Callable[[str], List[str]]:
    """Create a callable that routes evaluator feedback to individual agents.

    Unlike ``create_evaluator_loss`` (which returns a ``tg.TextLoss`` for
    automatic backward), this returns a plain function that:
    1. Takes a debate transcript string.
    2. Calls the evaluator LLM once with a prompt requesting per-agent feedback.
    3. Returns ``list[str]`` — one feedback string per agent.

    The caller is responsible for injecting these as gradients.
    """
    t0_strs = [str(a) if a is not None else "FAILED" for a in t0_parsed]
    agent_lines = "\n".join(
        f"Agent {idx + 1}: {answer}" for idx, answer in enumerate(t0_strs)
    )

    system_prompt = f"""You are evaluating a multi-agent math debate. Each agent has its OWN independent system prompt that will be optimized separately. Your job is to provide SEPARATE feedback for each agent's system prompt.

GROUND TRUTH ANSWER: {ground_truth}

INDEPENDENT ANSWERS AT t=0 (before debate):
{agent_lines}
Majority Vote at t=0: {t0_majority}

Analyze the debate transcript below. For EACH agent, evaluate:

1. ECHO CHAMBER: Did this agent change a correct answer to an incorrect one under peer pressure? If so, harshly critique — their prompt must forbid conceding without mathematical proof.

2. STAGNATION: Did this agent just repeat their position without engaging? Critique their prompt for failing to encourage productive reasoning.

3. COMPLEMENTARITY: Did this agent's approach complement or duplicate the others? Suggest how their prompt should differentiate their role.

4. POSITIVE PATTERNS: Did this agent successfully defend a correct answer or convince others through rigorous reasoning? Reinforce these behaviors.

You MUST structure your output with these exact markers:

{chr(10).join(
    f"[AGENT_{i+1}_FEEDBACK]{chr(10)}<feedback for Agent {i+1}'s system prompt>{chr(10)}"
    for i in range(n_agents)
)}
Be specific. Reference exact moments in the transcript."""

    def evaluate_transcript(transcript_text: str) -> List[str]:
        raw = evaluator_engine.generate(transcript_text, system_prompt=system_prompt)
        return _parse_per_agent_feedback(raw, n_agents)

    return evaluate_transcript
