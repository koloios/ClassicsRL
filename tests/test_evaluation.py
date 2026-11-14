import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from classicsrl.artifacts import append_json, create_run, save_checkpoint
from classicsrl.config import Config
from classicsrl.dqn import QNetwork
from classicsrl.evaluation import evaluate, select_validation


def test_validation_selection_respects_protocol_budget_and_latest_tie():
    def row(step, score, environment="full", policy="greedy"):
        return dict(agent_steps=step, mean_return=score, environment=environment, policy=policy)

    rows = [
        row(50, -10),
        row(100, -10),
        row(150, 20),
        row(100, 21, environment="capped"),
        row(100, 21, policy="random"),
        row(100, float("nan")),
        {**row(100, 21), "evaluation_role": "heldout"},
        {**row(100, 21), "evaluation_role": "demonstration"},
    ]
    assert select_validation(rows, "full", completed_steps=100) == rows[1]


def test_validation_selection_requires_an_eligible_checkpoint():
    with pytest.raises(ValueError, match="no matching"):
        select_validation([], "full", completed_steps=100)


@pytest.mark.parametrize(
    "interrupted,require_complete", [(False, True), (True, True), (True, False)]
)
def test_comparison_locks_both_selections_and_requires_completed_runs(
    tmp_path, monkeypatch, interrupted, require_complete
):
    script = Path(__file__).parents[1] / "scripts" / "compare_prefix.py"
    spec = importlib.util.spec_from_file_location("compare_prefix", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = replace(Config(), total_steps=100)
    runs = [tmp_path / f"run-{index}" for index in range(2)]
    for run in runs:
        metadata = create_run(run, config, "cpu")
        checkpoint = save_checkpoint(
            run, QNetwork(2), config, {"step": 100, "updates": 5}, metadata["run_id"]
        )
        append_json(
            run / "evaluations.jsonl",
            {
                "agent_steps": 100,
                "mean_return": 1.0,
                "policy": "greedy",
                "environment": metadata["environment"],
                "checkpoint": str(checkpoint.relative_to(run)),
            },
        )
        append_json(run / "segments.jsonl", {"event": "start", "config": config.to_dict()})
        append_json(
            run / "segments.jsonl",
            {
                "event": "stop",
                "agent_steps": 100,
                "interrupted": interrupted,
            },
        )
    output = tmp_path / "comparison"
    calls = []

    def check_locked(*args, **kwargs):
        locked = json.loads((output / "selection.json").read_text())
        assert len(locked["selections"]) == 2
        assert locked["test_seeds"] == [7000000, 7000001]
        calls.append(kwargs.get("policy", "greedy"))
        return {"mean_return": 1.0, "episodes": [{"terminated": True, "return": 1.0}]}

    monkeypatch.setattr(module, "evaluate", check_locked)
    argv = [
        str(script),
        *map(str, runs),
        "--budget",
        "100",
        "--output",
        str(output),
        "--episodes",
        "2",
    ]
    if require_complete:
        argv.append("--require-complete")
    monkeypatch.setattr(sys, "argv", argv)
    if interrupted and require_complete:
        with pytest.raises(ValueError, match="has not completed"):
            module.main()
        assert not output.exists() and not calls
    else:
        module.main()
        report = json.loads((output / "comparison.json").read_text())
        assert calls == ["greedy", "greedy", "random"]
        assert report["kind"] == (
            "prospective_completed_runs"
            if require_complete
            else "posthoc_common_budget_after_interruption"
        )
        assert report["evaluations"][0]["evaluation_role"] == (
            "heldout" if require_complete else "posthoc_heldout"
        )


def test_recording_has_one_frame_per_decision_and_explicit_final_frame(tmp_path, pixel_env_factory):
    imageio = pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    output = tmp_path / "recording"
    result = evaluate(
        QNetwork(2), Config(), "cpu", [17], record_dir=output, env_factory=pixel_env_factory
    )
    decisions = [
        json.loads(line) for line in (output / "episode-000.jsonl").read_text().splitlines()
    ]
    reader = imageio.get_reader(output / result["episodes"][0]["video"])
    try:
        assert sum(1 for _ in reader) == 4
        assert reader.get_meta_data()["fps"] == 15
    finally:
        reader.close()
    assert [row["frame_index"] for row in decisions] == [0, 1, 2]
    assert all(row["action_name"] in {"0", "1"} for row in decisions)
    assert all(len(row["action_values"]) == 2 for row in decisions)
    assert result["episodes"][0]["length"] == 3
    assert (output / "evaluation.json").is_file()
