"""Lightweight smoke checks for TG-MAD local behavior.

These checks do not require live vLLM servers. They verify:
- generation failures fail fast by default
- degraded mode remains available behind an explicit flag
- artifact paths resolve under the selected output directory
- text-history paths resolve and JSONL writing works
"""

import json
import os
import tempfile

import textgrad as tg

from tg_mad.forward_pass import mad_forward_pass
from tg_mad.utils import (
    append_jsonl_record,
    init_text_history_file,
    resolve_artifact_paths,
    resolve_text_history_paths,
)


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

    history_paths = resolve_text_history_paths(
        output_dir="out/tg_mad_smoke",
        existing_data_path="out/history/gsm8k/gsm8k_500__qwen3-4b_N=3_R=3.jsonl",
        stage="train",
        save_text_history=True,
    )
    assert history_paths["text_history_dir"].endswith("out/history/gsm8k/tg_mad_text/tg_mad_smoke")
    assert history_paths["text_history_file"].endswith(
        "out/history/gsm8k/tg_mad_text/tg_mad_smoke/train_text_history.jsonl"
    )

    os.makedirs("out/history", exist_ok=True)
    with tempfile.TemporaryDirectory(dir="out/history") as tmpdir:
        history_file = os.path.join(tmpdir, "train_text_history.jsonl")
        init_text_history_file(
            history_file,
            {"record_type": "manifest", "schema_version": 1, "stage": "train"},
        )
        append_jsonl_record(
            history_file,
            {"record_type": "sample", "schema_version": 1, "sample_index": 0},
        )
        with open(history_file, "r") as f:
            records = [json.loads(line) for line in f if line.strip()]
        assert [record["record_type"] for record in records] == ["manifest", "sample"]
        assert records[1]["sample_index"] == 0

    print("TG-MAD smoke checks passed.")


if __name__ == "__main__":
    main()
