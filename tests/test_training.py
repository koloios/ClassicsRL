import copy
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from classicsrl.artifacts import (
    load_checkpoint,
    restore_snapshot,
    sync_run,
    verify_snapshot,
)
from classicsrl.config import Config
from classicsrl.dqn import QNetwork
from classicsrl.evaluation import evaluate
from classicsrl.training import train


def small_config():
    return Config(
        device="cpu",
        torch_threads=1,
        total_steps=12,
        replay_capacity=16,
        batch_size=2,
        learning_starts=4,
        train_every=2,
        target_update_every=6,
        eval_every=6,
        eval_episodes=2,
        checkpoint_every=6,
        snapshot_every=12,
        log_every=6,
    )


def test_training_checkpoint_snapshot_and_resume_round_trip(tmp_path, pixel_env_factory):
    run = tmp_path / "run"
    config = small_config()
    first = train(config, run, env_factory=pixel_env_factory)
    assert first["counters"]["updates"] == 5
    assert first["counters"]["raw_frames"] == 48
    assert first["counters"]["episodes"] == 4
    saved = load_checkpoint(first["checkpoint"], "cpu")
    network = QNetwork(2)
    target = copy.deepcopy(network)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    state, replay = restore_snapshot(
        first["snapshot"], network, target, optimizer, torch.device("cpu"), np.random.default_rng()
    )
    torch.testing.assert_close(network.state_dict(), saved["online"])
    assert state["counters"] == first["counters"]
    assert len(replay) > 0
    assert int(next(iter(optimizer.state.values()))["step"]) == 5
    resumed = train(
        replace(config, total_steps=20),
        run,
        resume=first["snapshot"],
        env_factory=pixel_env_factory,
    )
    assert resumed["run_id"] == first["run_id"]
    assert resumed["counters"]["step"] == 20
    assert resumed["counters"]["updates"] == 9
    assert resumed["counters"]["raw_frames"] == 80
    segments = [json.loads(line) for line in (run / "segments.jsonl").read_text().splitlines()]
    starts = [segment for segment in segments if segment["event"] == "start"]
    assert len(starts) == 2 and starts[0]["segment_id"] != starts[1]["segment_id"]
    evaluations = [
        json.loads(line) for line in (run / "evaluations.jsonl").read_text().splitlines()
    ]
    assert [item["agent_steps"] for item in evaluations] == [6, 12, 18, 20]
    assert evaluations[0]["episodes"][0]["seed"] == config.eval_seed


def test_evaluation_does_not_change_torch_or_training_action_rng(pixel_env_factory):
    network = QNetwork(2)
    before = torch.get_rng_state().clone()
    original = copy.deepcopy(network.state_dict())
    evaluate(network, small_config(), "cpu", [100, 101], env_factory=pixel_env_factory)
    torch.testing.assert_close(torch.get_rng_state(), before)
    torch.testing.assert_close(network.state_dict(), original)
    assert network.training


def test_bad_snapshots_and_copy_recovery(tmp_path, pixel_env_factory):
    run = tmp_path / "run"
    result = train(small_config(), run, env_factory=pixel_env_factory)
    (run / "snapshots" / ".interrupted.tmp").mkdir()
    durable = tmp_path / "durable"
    sync_run(run, durable)
    assert not (durable / "snapshots" / ".interrupted.tmp").exists()
    copied = durable / "snapshots" / result["snapshot"].name
    verify_snapshot(copied)
    # Simulate interruption before the completion marker was persisted.
    (copied / "manifest.json").unlink()
    (copied / "training.pt").write_bytes(b"incomplete")
    sync_run(run, durable)
    verify_snapshot(copied)
    (copied / "training.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="integrity"):
        verify_snapshot(copied)


def test_resume_rejects_learning_changes_and_foreign_run(tmp_path, pixel_env_factory):
    config = small_config()
    run = tmp_path / "run"
    result = train(config, run, env_factory=pixel_env_factory)
    with pytest.raises(ValueError, match="gamma"):
        train(
            replace(config, total_steps=20, gamma=0.5),
            run,
            resume=result["snapshot"],
            env_factory=pixel_env_factory,
        )
    meta = json.loads((run / "run.json").read_text())
    meta["run_id"] = "another-run"
    (run / "run.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="does not belong"):
        train(
            replace(config, total_steps=20),
            run,
            resume=result["snapshot"],
            env_factory=pixel_env_factory,
        )


def test_replaying_old_snapshot_preserves_checkpoints_from_both_continuations(
    tmp_path, pixel_env_factory
):
    from classicsrl.artifacts import digest

    run = tmp_path / "run"
    config = small_config()
    first = train(config, run, env_factory=pixel_env_factory)
    second = train(
        replace(config, total_steps=18),
        run,
        resume=first["snapshot"],
        env_factory=pixel_env_factory,
    )
    original_digest = digest(second["checkpoint"])
    third = train(
        replace(config, total_steps=18),
        run,
        resume=first["snapshot"],
        env_factory=pixel_env_factory,
    )
    assert third["checkpoint"] != second["checkpoint"]
    assert digest(second["checkpoint"]) == original_digest
    evaluations = [
        json.loads(line) for line in (run / "evaluations.jsonl").read_text().splitlines()
    ]
    assert (run / evaluations[-1]["checkpoint"]).exists()
    assert evaluations[-1]["checkpoint"] != evaluations[-2]["checkpoint"]


def test_diagnostic_logging_tolerates_empty_replay_after_reset(tmp_path, pixel_env_factory):
    class FourStepFixture(pixel_env_factory):
        def step(self, action):
            observation, reward, _, _, info = super().step(action)
            return observation, reward, self.steps == 4, False, info

    config = replace(
        small_config(),
        total_steps=8,
        replay_capacity=1,
        batch_size=1,
        learning_starts=1,
        train_every=1,
        target_update_every=2,
        log_every=1,
        checkpoint_every=4,
        snapshot_every=8,
        eval_every=0,
    )
    result = train(config, tmp_path / "run", env_factory=FourStepFixture)
    assert result["counters"]["step"] == 8
    rows = [
        json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    ]
    probes = [row for row in rows if row["metric"] == "train/features/conv3_active_fraction"]
    assert {row["agent_steps"] for row in probes} == {1, 2, 3, 5, 6, 7}
