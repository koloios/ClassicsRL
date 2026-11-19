"""Optional recovery checks against the pinned SB3 implementation; no game assets."""

import copy
import importlib.util
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from classicsrl.config import Config
from classicsrl.dqn import epsilon_at

pytest.importorskip("stable_baselines3", minversion="2.9.2")
from stable_baselines3.common.logger import configure  # noqa: E402


@pytest.fixture
def reference():
    path = Path(__file__).parents[1] / "scripts" / "train_sb3_reference.py"
    spec = importlib.util.spec_from_file_location("sb3_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config():
    return replace(
        Config(),
        device="cpu",
        torch_threads=1,
        total_steps=32,
        replay_capacity=32,
        batch_size=4,
        learning_starts=8,
        train_every=2,
        target_update_every=10,
        exploration_steps=24,
        checkpoint_every=8,
        snapshot_every=8,
        log_every=4,
        eval_every=8,
        eval_episodes=1,
    )


@pytest.fixture
def paused(reference, config, tmp_path, pixel_env_factory, monkeypatch):
    captured = []
    original_callback = reference.RecordControl

    class Pause(reference.RecordControl):
        def _on_rollout_start(self):
            super()._on_rollout_start()
            if self.num_timesteps == 16:
                captured.append(self.model)
                raise KeyboardInterrupt

    monkeypatch.setattr(reference, "RecordControl", Pause)
    run, durable = tmp_path / "run", tmp_path / "durable"
    result = reference.train_control(
        config,
        run,
        persist_dir=durable,
        env_factory=pixel_env_factory,
    )
    monkeypatch.setattr(reference, "RecordControl", original_callback)
    assert result["status"] == "paused" and result["agent_steps"] == 16
    metadata = json.loads((run / "run.json").read_text())
    copied = durable / metadata["run_id"]
    snapshot = copied / "reference-snapshots" / result["snapshot"].name
    reference.verify_reference_snapshot(snapshot)
    return captured[0], copied, snapshot, metadata


@pytest.mark.parametrize("legacy", [False, True])
def test_native_state_rng_and_next_update_survive_durable_recovery(
    reference,
    config,
    paused,
    pixel_env_factory,
    monkeypatch,
    legacy,
):
    original, run, snapshot, metadata = paused
    if legacy:
        path = snapshot / "state.json"
        state = json.loads(path.read_text())
        del state["config"]["double_dqn"]
        path.write_text(json.dumps(state))
        manifest_path = snapshot / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["state.json"] = {
            "bytes": path.stat().st_size,
            "sha256": reference.digest(path),
        }
        manifest_path.write_text(json.dumps(manifest))
    expected = {
        "policy": copy.deepcopy(original.policy.state_dict()),
        "optimizer": copy.deepcopy(original.policy.optimizer.state_dict()),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "actions": copy.deepcopy(original.action_space.np_random.bit_generator.state),
    }
    env = pixel_env_factory(config)
    restored, state = reference.restore_reference_snapshot(
        snapshot,
        config,
        metadata,
        env,
        "cpu",
    )
    try:
        torch.testing.assert_close(restored.policy.state_dict(), expected["policy"], rtol=0, atol=0)
        torch.testing.assert_close(
            restored.policy.optimizer.state_dict(),
            expected["optimizer"],
            rtol=0,
            atol=0,
        )
        assert restored.num_timesteps == original.num_timesteps == 16
        assert restored._n_updates == original._n_updates == 5
        assert restored._n_calls == original._n_calls == 16
        assert restored.exploration_rate == original.exploration_rate
        assert random.getstate() == expected["python"]
        np.testing.assert_equal(np.random.get_state(), expected["numpy"])
        assert torch.equal(torch.get_rng_state(), expected["torch"])
        assert restored.action_space.np_random.bit_generator.state == expected["actions"]
        assert state["train_seconds"] > 0
        assert restored._last_obs is None
        assert restored.replay_buffer.pos == original.replay_buffer.pos == 16
        for key in ("observations", "next_observations", "actions", "rewards", "dones", "timeouts"):
            np.testing.assert_array_equal(
                getattr(restored.replay_buffer, key),
                getattr(original.replay_buffer, key),
            )
        assert restored.replay_buffer.handle_timeout_termination
        assert restored.replay_buffer.timeouts[:16].any()
        assert not all(
            torch.equal(a, b)
            for a, b in zip(
                restored.q_net.parameters(), restored.q_net_target.parameters(), strict=True
            )
        )
        # Identical restored Adam/replay/RNG state must produce the same next update.
        restored.set_logger(configure(format_strings=[]))
        original.train(gradient_steps=1, batch_size=config.batch_size)
        np.random.set_state(expected["numpy"])
        restored.train(gradient_steps=1, batch_size=config.batch_size)
        torch.testing.assert_close(
            restored.policy.state_dict(), original.policy.state_dict(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            restored.policy.optimizer.state_dict(),
            original.policy.optimizer.state_dict(),
            rtol=0,
            atol=0,
        )
    finally:
        env.close()

    observations = []

    class Continue(reference.RecordControl):
        def _on_training_start(self):
            super()._on_training_start()
            observations.append(self.model._last_obs.copy())

        def _on_training_end(self):
            super()._on_training_end()
            assert self.model._n_updates == 13
            assert self.model._n_calls == config.total_steps
            assert self.model.exploration_rate == pytest.approx(config.epsilon_end)

    monkeypatch.setattr(reference, "RecordControl", Continue)
    result = reference.train_control(config, run, resume=snapshot, env_factory=pixel_env_factory)
    assert result["status"] == "complete" and result["agent_steps"] == 32
    assert np.all(observations[0] == config.seed + 16)
    final = reference.verify_reference_snapshot(result["snapshot"])
    assert final["updates"] == 13 and final["train_seconds"] > state["train_seconds"]
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    for row in rows:
        if row["metric"] == "train/epsilon":
            assert row["value"] == pytest.approx(
                epsilon_at(
                    row["agent_steps"],
                    config.epsilon_start,
                    config.epsilon_end,
                    config.exploration_steps,
                )
            )
    segments = [json.loads(line) for line in (run / "segments.jsonl").read_text().splitlines()]
    assert [(s["event"], s["agent_steps"]) for s in segments] == [
        ("start", 0),
        ("stop", 16),
        ("start", 16),
        ("stop", 32),
    ]
    evaluations = [
        json.loads(line) for line in (run / "evaluations.jsonl").read_text().splitlines()
    ]
    assert [row["agent_steps"] for row in evaluations] == [8, 16, 24, 32]


@pytest.mark.parametrize("change", [{"learning_rate": 0.001}, {"total_steps": 64}])
def test_resume_rejects_changed_experiment(reference, config, paused, pixel_env_factory, change):
    _, run, snapshot, _ = paused
    with pytest.raises(ValueError, match="resume would change"):
        reference.train_control(
            replace(config, **change),
            run,
            resume=snapshot,
            env_factory=pixel_env_factory,
        )


def test_reference_rejects_double_dqn_before_creating_a_run(
    reference, config, tmp_path, pixel_env_factory
):
    run = tmp_path / "run"
    with pytest.raises(ValueError, match="vanilla DQN only"):
        reference.train_control(
            replace(config, double_dqn=True), run, env_factory=pixel_env_factory
        )
    assert not run.exists()


def test_partial_copy_retries_and_corruption_is_rejected_before_loading(
    reference,
    config,
    paused,
    tmp_path,
    pixel_env_factory,
    monkeypatch,
):
    _, run, snapshot, metadata = paused
    output = tmp_path / "another-copy"
    temporary = output / "reference-snapshots" / f".{snapshot.name}.tmp"
    temporary.mkdir(parents=True)
    (temporary / "model.zip").write_bytes(b"interrupted copy")
    reference.sync_control(run, output)
    reference.verify_reference_snapshot(output / "reference-snapshots" / snapshot.name)
    assert not temporary.exists()
    foreign = {**metadata, "run_id": "unrelated"}
    with pytest.raises(ValueError, match="does not belong"):
        reference.restore_reference_snapshot(
            snapshot, config, foreign, pixel_env_factory(config), "cpu"
        )
    (snapshot / "replay.pkl.gz").write_bytes(b"corrupt")

    def unexpected_load(*args, **kwargs):
        pytest.fail("unverified pickle must never be loaded")

    monkeypatch.setattr(reference.DQN, "load", unexpected_load)
    with pytest.raises(ValueError, match="integrity"):
        reference.restore_reference_snapshot(
            snapshot, config, metadata, pixel_env_factory(config), "cpu"
        )


def test_interrupted_update_keeps_previous_coherent_snapshot(
    reference,
    config,
    tmp_path,
    pixel_env_factory,
    monkeypatch,
):
    native_train = reference.DQN.train

    def interrupted(model, *args, **kwargs):
        native_train(model, *args, **kwargs)
        if model.num_timesteps == 18:
            raise KeyboardInterrupt

    monkeypatch.setattr(reference.DQN, "train", interrupted)
    run = tmp_path / "run"
    result = reference.train_control(config, run, env_factory=pixel_env_factory)
    assert result["status"] == "paused" and result["agent_steps"] == 18
    state = reference.verify_reference_snapshot(result["snapshot"])
    assert state["agent_steps"] == 16 and state["updates"] == 5
    assert not list((run / "reference-snapshots").glob("step-000000000018-*"))
    last = json.loads((run / "segments.jsonl").read_text().splitlines()[-1])
    assert last["interrupted"] and last["resume_snapshot"] == result["snapshot"].name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_resume_preserves_native_adam_placement_and_next_update(
    reference, config, tmp_path, pixel_env_factory, monkeypatch
):
    config = replace(config, device="cuda")
    captured = []

    class Pause(reference.RecordControl):
        def _on_rollout_start(self):
            super()._on_rollout_start()
            if self.num_timesteps == 16:
                captured.append(self.model)
                raise KeyboardInterrupt

    monkeypatch.setattr(reference, "RecordControl", Pause)
    run = tmp_path / "run"
    result = reference.train_control(config, run, env_factory=pixel_env_factory)
    original = captured[0]
    metadata = json.loads((run / "run.json").read_text())
    expected_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
    env = pixel_env_factory(config)
    restored, _ = reference.restore_reference_snapshot(
        result["snapshot"], config, metadata, env, "cuda"
    )
    try:
        assert all(
            torch.equal(a, b)
            for a, b in zip(
                expected_rng,
                torch.cuda.get_rng_state_all(),
                strict=True,
            )
        )
        for model in (original, restored):
            assert all(
                s["step"].device.type == "cpu" for s in model.policy.optimizer.state.values()
            )
        torch.testing.assert_close(
            restored.policy.optimizer.state_dict(),
            original.policy.optimizer.state_dict(),
            rtol=0,
            atol=0,
        )
        numpy_rng = np.random.get_state()
        restored.set_logger(configure(format_strings=[]))
        original.train(gradient_steps=1, batch_size=config.batch_size)
        np.random.set_state(numpy_rng)
        restored.train(gradient_steps=1, batch_size=config.batch_size)
        torch.testing.assert_close(
            restored.policy.state_dict(), original.policy.state_dict(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            restored.policy.optimizer.state_dict(),
            original.policy.optimizer.state_dict(),
            rtol=0,
            atol=0,
        )
    finally:
        env.close()
