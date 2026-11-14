import gymnasium as gym
import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def predictable_torch():
    torch.set_num_threads(1)
    torch.manual_seed(17)


class PixelFixture(gym.Env):
    """Tiny deterministic image fixture; no game assets or Atari dependencies."""

    observation_space = gym.spaces.Box(0, 255, (4, 84, 84), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, config=None, render_mode=None):
        self.raw_frames = 0
        self.resets = 0
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.resets += 1
        self.steps = 0
        self.observation = np.full((4, 84, 84), seed % 128 if seed is not None else 0, np.uint8)
        return self.observation.copy(), {}

    def step(self, action):
        self.steps += 1
        self.raw_frames += 4
        frame = np.full((1, 84, 84), self.observation[-1, 0, 0] + 1, np.uint8)
        self.observation = np.concatenate((self.observation[1:], frame))
        ended = self.steps == 3
        return (
            self.observation.copy(),
            1.0 if action == 1 else -1.0,
            ended and self.resets % 2 == 1,
            ended and self.resets % 2 == 0,
            {},
        )

    def render(self):
        return np.repeat(self.observation[-1, :, :, None], 3, axis=2)


@pytest.fixture
def pixel_env_factory():
    return PixelFixture
