import copy

import numpy as np
import pytest
import torch
from torch import nn

from classicsrl.dqn import QNetwork, epsilon_at, select_action, td_targets, update
from classicsrl.replay import Batch


def test_known_targets_distinguish_termination_from_truncation():
    # The third transition is time-limit truncated, which must keep its bootstrap.
    rewards = torch.tensor([1.0, -1.0, 2.0])
    terminated = torch.tensor([False, True, False])
    targets = td_targets(rewards, terminated, torch.tensor([4.0, 100.0, 8.0]), gamma=0.5)
    torch.testing.assert_close(targets, torch.tensor([3.0, -1.0, 6.0]))


@pytest.mark.parametrize("double_dqn, expected_loss", [(False, 2.0), (True, 2.25)])
def test_update_selects_and_evaluates_next_actions_without_mutating_target(
    double_dqn, expected_loss
):
    network = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    target = copy.deepcopy(network)
    with torch.no_grad():
        network[1].weight.zero_()
        network[1].bias.copy_(torch.tensor([7.0, 1.0]))
        target[1].weight.zero_()
        target[1].bias.copy_(torch.tensor([3.0, 4.0]))
    old_target = copy.deepcopy(target.state_dict())
    batch = Batch(
        observations=np.zeros((2, 1, 2, 2), np.float32),
        actions=np.array([0, 1]),
        rewards=np.array([1.0, 2.0], np.float32),
        next_observations=np.zeros((2, 1, 2, 2), np.float32),
        terminated=np.array([False, True]),
        truncated=np.array([True, False]),
    )
    result = update(
        network,
        target,
        torch.optim.SGD(network.parameters(), lr=0.1),
        batch,
        torch.device("cpu"),
        gamma=0.5,
        max_grad_norm=10,
        double_dqn=double_dqn,
    )
    # Online prefers action 0; target prefers 1. The truncated row still bootstraps.
    # Targets: vanilla [3, 2], double [2.5, 2]; predictions [7, 1].
    assert result["td_loss"] == pytest.approx(expected_loss)
    torch.testing.assert_close(target.state_dict(), old_target)
    assert all(parameter.grad is None for parameter in target.parameters())
    torch.testing.assert_close(network[1].bias, torch.tensor([6.95, 1.05]))


def test_dqn_update_learns_known_terminal_action_values():
    network = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    target = copy.deepcopy(network)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.05)
    batch = Batch(
        observations=np.zeros((16, 1, 2, 2), np.float32),
        actions=np.tile([0, 1], 8),
        rewards=np.tile([-1.0, 1.0], 8).astype(np.float32),
        next_observations=np.zeros((16, 1, 2, 2), np.float32),
        terminated=np.ones(16, bool),
        truncated=np.zeros(16, bool),
    )
    for _ in range(150):
        update(network, target, optimizer, batch, torch.device("cpu"), 0.99, 10)
    torch.testing.assert_close(
        network(torch.zeros(1, 1, 2, 2))[0], torch.tensor([-1.0, 1.0]), atol=0.01, rtol=0
    )


def test_cnn_handles_bytes_and_matches_explicit_float_input():
    network = QNetwork(6)
    image = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    assert network(image).shape == (2, 6)
    torch.testing.assert_close(network(image), network(image.float()))
    assert torch.isfinite(network(image)).all()


@pytest.mark.parametrize("seed", [0, 17])
def test_cnn_learns_opposite_actions_from_spatial_image_cues(seed):
    """Balanced labels prevent a state-independent policy from passing."""
    torch.manual_seed(seed)
    images = np.full((2, 4, 84, 84), 52, np.uint8)
    images[0, :, 28:44, 8:24] = 220
    images[1, :, 28:44, 60:76] = 220
    batch = Batch(
        observations=np.repeat(images, 2, axis=0),
        actions=np.array([0, 1, 0, 1]),
        rewards=np.array([1, -1, -1, 1], np.float32),
        next_observations=np.zeros((4, 4, 84, 84), np.uint8),
        terminated=np.ones(4, bool),
        truncated=np.zeros(4, bool),
    )
    network = QNetwork(2)
    target = copy.deepcopy(network).requires_grad_(False)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    for _ in range(100):
        update(network, target, optimizer, batch, torch.device("cpu"), 0.99, 10)
    with torch.no_grad():
        values = network(torch.tensor(images))
    torch.testing.assert_close(values, torch.tensor([[1.0, -1.0], [-1.0, 1.0]]), atol=0.1, rtol=0)


def test_exploration_does_not_need_network_inference():
    class MustNotRun(nn.Module):
        def forward(self, observation):
            raise AssertionError("random exploration ran the Q-network")

    rng = np.random.default_rng(5)
    actions = [select_action(MustNotRun(), None, 1.0, rng, "cpu", 3) for _ in range(100)]
    assert set(actions) == {0, 1, 2}


def test_epsilon_schedule_stops_at_end():
    assert [epsilon_at(step, 1.0, 0.1, 100) for step in (0, 50, 100, 200)] == pytest.approx(
        [1.0, 0.55, 0.1, 0.1]
    )
