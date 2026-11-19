"""The network, epsilon-greedy policy, and a DQN or Double DQN update."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from classicsrl.replay import Batch


class QNetwork(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        # Gymnasium's Discrete.n is a NumPy integer; checkpoint metadata stays primitive.
        num_actions = int(num_actions)
        self.layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, observations):
        return self.layers(observations.float() / 255.0)


def resolve_device(requested="auto") -> torch.device:
    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }
    if requested == "auto":
        requested = next(name for name in ("cuda", "mps", "cpu") if available[name])
    if requested not in available or not available[requested]:
        raise ValueError(f"device unavailable: {requested}")
    return torch.device(requested)


def epsilon_at(step, start, end, decay_steps):
    fraction = min(max(step / decay_steps, 0.0), 1.0)
    return start + fraction * (end - start)


@torch.no_grad()
def action_values(network, observation, device):
    tensor = torch.as_tensor(np.asarray(observation), device=device).unsqueeze(0)
    return network(tensor)[0].cpu().numpy()


def select_action(network, observation, epsilon, rng, device, num_actions):
    if rng.random() < epsilon:
        return int(rng.integers(num_actions))
    return int(action_values(network, observation, device).argmax())


def td_targets(rewards, terminated, next_values, gamma):
    """Time-limit truncations bootstrap; only true termination removes the next value."""
    return rewards + gamma * next_values * (~terminated).float()


def update(
    network, target, optimizer, batch: Batch, device, gamma, max_grad_norm, *, double_dqn=False
):
    observations = torch.as_tensor(batch.observations, device=device)
    next_observations = torch.as_tensor(batch.next_observations, device=device)
    actions = torch.as_tensor(batch.actions, device=device)
    rewards = torch.as_tensor(batch.rewards, device=device)
    terminated = torch.as_tensor(batch.terminated, device=device)
    with torch.no_grad():
        if double_dqn:
            next_actions = network(next_observations).argmax(dim=1, keepdim=True)
            next_values = target(next_observations).gather(1, next_actions).squeeze(1)
        else:
            next_values = target(next_observations).max(dim=1).values
        targets = td_targets(rewards, terminated, next_values, gamma)
    values = network(observations).gather(1, actions[:, None]).squeeze(1)
    loss = F.smooth_l1_loss(values, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)
    optimizer.step()
    return {
        "td_loss": loss.item(),
        "q_mean": values.detach().mean().item(),
        "grad_norm": float(grad_norm),
    }
