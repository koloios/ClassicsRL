import copy

import numpy as np
import pytest
import torch

from classicsrl.artifacts import digest, save_checkpoint
from classicsrl.config import Config
from classicsrl.diagnostics import activation_metrics, diagnose
from classicsrl.dqn import QNetwork
from classicsrl.replay import ReplayBuffer


def collapsed_network():
    network = QNetwork(2)
    with torch.no_grad():
        network.layers[4].weight.zero_()
        network.layers[4].bias.fill_(-1)
    return network


def test_activation_probe_detects_blocked_visual_features_without_mutation():
    network = collapsed_network()
    before = copy.deepcopy(network.state_dict())
    rng = torch.get_rng_state().clone()
    images = np.random.default_rng(3).integers(0, 256, (8, 4, 84, 84), dtype=np.uint8)
    result = activation_metrics(network, images, "cpu")
    assert result["features/conv3_active_fraction"] == 0
    assert result["features/conv3_zero_fraction"] == 1
    assert result["q/max_observation_range"] == 0
    torch.testing.assert_close(network.state_dict(), before)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    with pytest.raises(ValueError, match="uint8"):
        activation_metrics(network, images.astype(np.float32) / 255, "cpu")


def test_saved_replay_diagnosis_reports_zero_gradients_and_preserves_artifacts(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    checkpoint = save_checkpoint(tmp_path, collapsed_network(), Config(), {"step": 2}, "test-run")
    replay = ReplayBuffer(4)
    replay.begin_episode(np.zeros((84, 84), np.uint8))
    replay.add(1, 1.0, np.full((84, 84), 200, np.uint8), True, False)
    path = tmp_path / "replay.npz"
    replay.save(path)
    before = (digest(checkpoint), digest(path))
    report = diagnose(checkpoint, path, samples=4, seed=5)
    metrics = report["devices"]["cpu"]
    assert metrics["activations"]["features/conv3_zero_fraction"] == 1
    assert metrics["gradient_norms"]["layers.0.weight"] == 0
    assert metrics["parameter_change_norms"]["layers.0.weight"] == 0
    assert metrics["gradient_norms"]["layers.9.bias"] > 0
    assert before == (digest(checkpoint), digest(path))
    assert report["checkpoint_sha256"] == before[0]


def test_probe_sampling_does_not_advance_training_replay_rng():
    replay = ReplayBuffer(4)
    replay.begin_episode(np.zeros((84, 84), np.uint8))
    replay.add(0, 0.0, np.full((84, 84), 50, np.uint8), False, False)
    before = copy.deepcopy(replay.rng.bit_generator.state)
    replay.sample(4, rng=np.random.default_rng(77))
    assert replay.rng.bit_generator.state == before


def test_feature_probe_matches_network_forward_normalization():
    network = QNetwork(2)
    images = np.random.default_rng(2).integers(0, 256, (4, 4, 84, 84), dtype=np.uint8)
    result = activation_metrics(network, images, "cpu")
    with torch.no_grad():
        values = network(torch.tensor(images))
    assert result["q/observation_std"] == pytest.approx(values.std(dim=0, correction=0).mean())
    assert result["q/greedy_action_count"] == values.argmax(dim=1).unique().numel()
