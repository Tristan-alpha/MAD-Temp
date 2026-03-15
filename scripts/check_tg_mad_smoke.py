"""Lightweight smoke checks for TG-MAD local behavior.

These checks do not require live vLLM servers. They verify:
- generation failures fail fast by default
- degraded mode remains available behind an explicit flag
- artifact paths resolve under the selected output directory
"""

import textgrad as tg

from tg_mad.forward_pass import mad_forward_pass
from tg_mad.utils import resolve_artifact_paths


class FailingEngine:
    def __call__(self, *args, **kwargs):
        raise RuntimeError("backend down")


def main():
    prompt = tg.Variable(
        "system prompt",
        requires_grad=False,
        role_description="shared system prompt",
    )

    try:
        mad_forward_pass(
            question="2+2?",
            ground_truth=4,
            debater_prompt=prompt,
            debater_engine=FailingEngine(),
            n_agents=1,
            n_rounds=0,
        )
    except RuntimeError as exc:
        assert "backend down" in str(exc)
    else:
        raise AssertionError("Expected mad_forward_pass to fail fast by default")

    degraded = mad_forward_pass(
        question="2+2?",
        ground_truth=4,
        debater_prompt=prompt,
        debater_engine=FailingEngine(),
        n_agents=1,
        n_rounds=0,
        allow_failed_generations=True,
    )
    assert degraded["t0_parsed"] == [None]
    assert degraded["final_majority_vote"] is None

    paths = resolve_artifact_paths("out/tg_mad_smoke")
    assert paths["prompt_history"].endswith("out/tg_mad_smoke/prompt_history.json")
    assert paths["split_info"].endswith("out/tg_mad_smoke/split_info.json")
    assert paths["eval_results"].endswith("out/tg_mad_smoke/eval_results.json")

    print("TG-MAD smoke checks passed.")


if __name__ == "__main__":
    main()
