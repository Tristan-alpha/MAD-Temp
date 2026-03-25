"""TextLoss evaluator templates for TG-MAD."""

import logging
import re
from typing import Callable, List, Optional

import textgrad as tg

from tg_mad.config import N_AGENTS

logger = logging.getLogger("tg_mad")


def _format_initial_answers(t0_parsed: List[Optional[object]]) -> str:
    t0_strs = [str(a) if a is not None else "FAILED" for a in t0_parsed]
    return "\n".join(f"Agent {idx + 1}: {answer}" for idx, answer in enumerate(t0_strs))


def _shared_template(dataset: str, ground_truth, agent_lines: str, t0_majority) -> str:
    if dataset == "formal_logic":
        return f"""You are evaluating a multi-agent formal-logic debate system. Your job is to generate feedback on the SYSTEM PROMPT used by the debater agents so it can be improved.

GROUND TRUTH ANSWER: {ground_truth}

INDEPENDENT ANSWERS AT t=0 (before debate):
{agent_lines}
Majority Vote at t=0: {t0_majority}

You will receive the full debate transcript. Analyze it and provide feedback on the debater system prompt.

YOUR ANALYSIS MUST CHECK FOR:

1. LOGICAL VALIDITY: Did the agents reason correctly about quantifiers, conditionals, negation, contradiction, or symbolic structure? If the debate endorsed an invalid inference, critique the prompt for failing to demand rigorous logical justification.

2. MAJORITY BIAS: Did a correct minority agent get dragged into a wrong answer just because other agents agreed? If so, harshly critique the prompt and insist that agents require explicit logical proof before conceding.

3. ERROR DETECTION: Did the debate miss a false premise, illicit conversion, scope mistake, or other formal reasoning error? Explain what kind of checking the system prompt should strengthen.

4. SUCCESSFUL CORRECTION OR DEFENSE: If the debate corrected a wrong majority or preserved a correct minority through sound logical analysis, generate positive feedback that reinforces those behaviors.

5. FORMAT PRESERVATION: The debater prompt must continue to require the final answer format '{{final answer: (X)}}'. Mention this if the debate drifted away from clean answer selection.

Be specific. Reference exact moments in the transcript. Your feedback will be used to update the system prompt via gradient descent."""

    if dataset == "hh_rlhf":
        return f"""You are evaluating a multi-agent helpfulness-and-harmlessness preference debate system. Your job is to generate feedback on the SYSTEM PROMPT used by the debater agents so it can be improved.

GROUND TRUTH ANSWER: {ground_truth}

INDEPENDENT ANSWERS AT t=0 (before debate):
{agent_lines}
Majority Vote at t=0: {t0_majority}

You will receive the full debate transcript. Analyze it and provide feedback on the debater system prompt.

YOUR ANALYSIS MUST CHECK FOR:

1. PREFERENCE TARGET: Did the agents correctly identify which option is more helpful and/or harmless? If they focused on irrelevant stylistic details instead of safety/helpfulness, critique the prompt.

2. HARMFUL SUBVERSION: Did the debate end up preferring a more harmful option, or excuse clearly unsafe behavior? If so, generate a harsh critique and explain how the prompt should better prioritize harmlessness.

3. HELPFULNESS FAILURE: Did the debate prefer an obviously unhelpful or evasive answer over a better one? Critique the prompt for weak preference reasoning.

4. MAJORITY BIAS: Did a correct minority agent get pulled toward a worse preference judgment just because the majority repeated it? If so, critique the prompt for not demanding evidence-based comparison.

5. SUCCESSFUL CORRECTION OR DEFENSE: If the debate corrected an initially bad preference or a correct agent successfully defended the safer/more helpful answer, generate positive feedback that reinforces those behaviors.

6. FORMAT PRESERVATION: The debater prompt must continue to require the final answer format '{{final answer: (X)}}'. Mention this if the debate drifted away from clear answer selection.

Be specific. Reference exact moments in the transcript. Your feedback will be used to update the system prompt via gradient descent."""

    return f"""You are evaluating a multi-agent math debate system. Your job is to generate feedback on the SYSTEM PROMPT used by the debater agents, so it can be improved.

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


def _per_agent_template(
    dataset: str,
    ground_truth,
    agent_lines: str,
    t0_majority,
    n_agents: int,
) -> str:
    markers = "\n".join(
        f"[AGENT_{i+1}_FEEDBACK]\n<feedback for Agent {i+1}'s system prompt>\n"
        for i in range(n_agents)
    )

    if dataset == "formal_logic":
        body = """Analyze the debate transcript below. For EACH agent, evaluate:

1. LOGICAL DISCIPLINE: Did this agent reason soundly about formal structure, or did they miss invalid inferences, scope mistakes, or false premises?

2. MAJORITY BIAS: Did this agent abandon a correct answer under peer pressure without a valid logical argument? If so, critique their prompt sharply.

3. COMPLEMENTARITY: Did this agent bring a distinct checking strategy, or merely duplicate the others?

4. POSITIVE PATTERNS: Did this agent successfully defend or correct an answer through careful logical reasoning? Reinforce those behaviors.

5. FORMAT PRESERVATION: Their prompt must still preserve the final answer format '{final answer: (X)}'."""
    elif dataset == "hh_rlhf":
        body = """Analyze the debate transcript below. For EACH agent, evaluate:

1. PREFERENCE REASONING: Did this agent compare helpfulness and harmlessness correctly, or did they focus on the wrong cues?

2. HARMFUL OR UNHELPFUL SELECTION: Did this agent argue for a more harmful or clearly less helpful answer? If so, critique their prompt sharply.

3. COMPLEMENTARITY: Did this agent add a distinct safety/helpfulness lens, or duplicate the others?

4. POSITIVE PATTERNS: Did this agent successfully defend the safer or more helpful answer, or persuade others with strong comparison reasoning? Reinforce those behaviors.

5. FORMAT PRESERVATION: Their prompt must still preserve the final answer format '{final answer: (X)}'."""
    else:
        body = """Analyze the debate transcript below. For EACH agent, evaluate:

1. ECHO CHAMBER: Did this agent change a correct answer to an incorrect one under peer pressure? If so, harshly critique — their prompt must forbid conceding without mathematical proof.

2. STAGNATION: Did this agent just repeat their position without engaging? Critique their prompt for failing to encourage productive reasoning.

3. COMPLEMENTARITY: Did this agent's approach complement or duplicate the others? Suggest how their prompt should differentiate their role.

4. POSITIVE PATTERNS: Did this agent successfully defend a correct answer or convince others through rigorous reasoning? Reinforce these behaviors."""

    return f"""You are evaluating a multi-agent debate. Each agent has its OWN independent system prompt that will be optimized separately. Your job is to provide SEPARATE feedback for each agent's system prompt.

GROUND TRUTH ANSWER: {ground_truth}

INDEPENDENT ANSWERS AT t=0 (before debate):
{agent_lines}
Majority Vote at t=0: {t0_majority}

{body}

You MUST structure your output with these exact markers:

{markers}
Be specific. Reference exact moments in the transcript."""


def create_evaluator_loss(
    ground_truth,
    t0_parsed: List[Optional[object]],
    t0_majority: Optional[object],
    evaluator_engine,
    dataset: str = "gsm8k",
) -> tg.TextLoss:
    agent_lines = _format_initial_answers(t0_parsed)
    evaluator_prompt = _shared_template(dataset, ground_truth, agent_lines, t0_majority)
    return tg.TextLoss(evaluator_prompt, engine=evaluator_engine)


def _parse_per_agent_feedback(raw: str, n_agents: int) -> List[str]:
    pattern = r"\[AGENT_(\d+)_FEEDBACK\]"
    splits = re.split(pattern, raw)

    feedbacks: dict[int, str] = {}
    for i in range(1, len(splits) - 1, 2):
        try:
            idx = int(splits[i])
            feedbacks[idx] = splits[i + 1].strip()
        except (ValueError, IndexError):
            continue

    if all((i + 1) in feedbacks for i in range(n_agents)):
        return [feedbacks[i + 1] for i in range(n_agents)]

    logger.warning(
        "Could not parse %d agent sections (found %d). Duplicating full feedback to all agents.",
        n_agents,
        len(feedbacks),
    )
    return [raw.strip()] * n_agents


def create_per_agent_evaluator(
    ground_truth,
    t0_parsed: List[Optional[object]],
    t0_majority: Optional[object],
    evaluator_engine,
    n_agents: int = N_AGENTS,
    dataset: str = "gsm8k",
) -> Callable[[str], List[str]]:
    agent_lines = _format_initial_answers(t0_parsed)
    system_prompt = _per_agent_template(
        dataset,
        ground_truth,
        agent_lines,
        t0_majority,
        n_agents,
    )

    def evaluate_transcript(transcript_text: str) -> List[str]:
        raw = evaluator_engine.generate(transcript_text, system_prompt=system_prompt)
        return _parse_per_agent_feedback(raw, n_agents)

    return evaluate_transcript
