import json

import pytest

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
