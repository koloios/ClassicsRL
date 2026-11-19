"""Validated settings shared by local and Colab training."""

import math
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Config:
    env_id: str = "ALE/Pong-v5"
    seed: int = 0
    device: str = "auto"
    torch_threads: int = 4
    total_steps: int = 1_000_000
    replay_capacity: int = 100_000
    batch_size: int = 32
    learning_starts: int = 20_000
    train_every: int = 4
    target_update_every: int = 10_000
    learning_rate: float = 0.00003
    gamma: float = 0.99
    double_dqn: bool = False
    max_grad_norm: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    exploration_steps: int = 200_000
    frame_skip: int = 4
    noop_max: int = 30
    sticky_action_probability: float = 0.25
    fire_on_reset: bool = True
    max_episode_steps: int = 27_000
    eval_every: int = 50_000
    eval_episodes: int = 5
    eval_seed: int = 1_000_000
    checkpoint_every: int = 25_000
    snapshot_every: int = 100_000
    log_every: int = 1_000

    def __post_init__(self):
        # Dataclasses do not enforce annotations when values come from TOML or checkpoints.
        for field in fields(self):
            value = getattr(self, field.name)
            expected = field.type
            if expected is float:
                valid = type(value) in (int, float) and math.isfinite(value)
            else:
                valid = type(value) is expected
            if not valid:
                raise ValueError(f"{field.name} must be {expected.__name__}")
        positive = (
            "torch_threads",
            "total_steps",
            "replay_capacity",
            "batch_size",
            "learning_starts",
            "train_every",
            "target_update_every",
            "learning_rate",
            "max_grad_norm",
            "exploration_steps",
            "frame_skip",
            "max_episode_steps",
            "eval_episodes",
            "checkpoint_every",
            "snapshot_every",
            "log_every",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("gamma", "epsilon_start", "epsilon_end", "sticky_action_probability"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.noop_max < 0 or self.eval_every < 0:
            raise ValueError("noop_max and eval_every must be nonnegative")
        if self.seed < 0 or self.eval_seed < 0:
            raise ValueError("seeds must be nonnegative")
        if self.epsilon_start < self.epsilon_end:
            raise ValueError("epsilon_start must be at least epsilon_end")
        if self.batch_size > self.replay_capacity:
            raise ValueError("batch_size must not exceed replay_capacity")
        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("device must be auto, cpu, mps, or cuda")

    def to_dict(self):
        return asdict(self)


def load_config(path: str | Path) -> Config:
    with Path(path).open("rb") as file:
        values = tomllib.load(file)
    unknown = values.keys() - Config.__dataclass_fields__.keys()
    if unknown:
        raise ValueError(f"unknown configuration keys: {', '.join(sorted(unknown))}")
    return Config(**values)
