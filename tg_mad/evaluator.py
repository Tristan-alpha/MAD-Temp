"""TextLoss evaluator for TG-MAD.

Creates per-sample TextLoss functions that embed ground truth and t=0 context
in the evaluator system prompt. The evaluator LLM generates textual gradients
targeting the debater_prompt Variable.
"""

from typing import List, Optional

import textgrad as tg


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
