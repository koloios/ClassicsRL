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


@pytest.fixture
def heldout_run(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "evaluate_run.py"
    spec = importlib.util.spec_from_file_location("heldout_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = tmp_path / "run"
    config = replace(Config(), device="cpu", torch_threads=1, total_steps=100)
    metadata = create_run(run, config, "cpu")
    for step in (50, 100):
        checkpoint = save_checkpoint(run, QNetwork(2), config, {"step": step}, metadata["run_id"])
        append_json(
            run / "evaluations.jsonl",
            {
                "agent_steps": step,
                "mean_return": float(step),
                "policy": "greedy",
                "environment": metadata["environment"],
                "checkpoint": str(checkpoint.relative_to(run)),
            },
        )
    append_json(run / "segments.jsonl", {"event": "start", "config": config.to_dict()})
    append_json(run / "segments.jsonl", {"event": "stop", "agent_steps": 100})
    return module, run


@pytest.mark.parametrize("interrupt_on", [2, 5])
def test_heldout_resume_matches_uninterrupted_games(
    heldout_run, monkeypatch, pixel_env_factory, interrupt_on
):
    module, run = heldout_run
    calls = []
    interruption = interrupt_on
    label = "resumed"

    class InterruptibleEnv(pixel_env_factory):
        def reset(self, *, seed=None, options=None):
            lock = run / f"test-selection-{label}.json"
            assert json.loads(lock.read_text())["test_seeds"] == [90, 91, 92]
            calls.append(seed)
            self.resets = seed
            return super().reset(seed=seed, options=options)

        def step(self, action):
            if len(calls) == interruption and self.steps == 1:
                raise KeyboardInterrupt
            return super().step(action)

    monkeypatch.setattr(
        module, "evaluate", lambda *a, **kw: evaluate(*a, **kw, env_factory=InterruptibleEnv)
    )
    command = [str(run), "--device", "cpu", "--episodes", "3", "--seed", "90", "--label"]
    with pytest.raises(KeyboardInterrupt):
        module.main([*command, label])
    lock_path = run / "test-selection-resumed.json"
    lock = lock_path.read_bytes()
    selected = json.loads(lock)
    policy = "greedy" if interrupt_on == 2 else "random"
    progress_path = run / "episodes" / f"heldout-resumed-{policy}" / "progress.json"
    assert [row["seed"] for row in json.loads(progress_path.read_text())["episodes"]] == [90]

    with pytest.raises(ValueError, match="test seeds differ"):
        module.main([*command, label, "--seed", "91"])
    checkpoint = run / selected["checkpoint"]
    weights = checkpoint.read_bytes()
    checkpoint.write_bytes(weights + b"changed")
    with pytest.raises(ValueError, match="checksum differs"):
        module.main([*command, label])
    checkpoint.write_bytes(weights)
    assert lock_path.read_bytes() == lock

    interruption = None
    calls.clear()
    module.main([*command, label])
    assert calls == ([91, 92, 90, 91, 92] if interrupt_on == 2 else [91, 92])
    assert not progress_path.exists()
    assert lock_path.read_bytes() == lock
    calls.clear()
    module.main([*command, label])
    assert not calls

    label = "uninterrupted"
    module.main([*command, label])
    for policy in ("greedy", "random"):
        resumed = run / "episodes" / f"heldout-resumed-{policy}" / "evaluation.json"
        fresh = run / "episodes" / f"heldout-uninterrupted-{policy}" / "evaluation.json"
        left, right = json.loads(resumed.read_text()), json.loads(fresh.read_text())
        assert left.pop("evaluation_id") != right.pop("evaluation_id")
        assert left == right


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
