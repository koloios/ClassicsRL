"""Optional fixed-batch audit against Stable-Baselines3's actual DQN.train method.

Run with: uv run --with stable-baselines3==2.9.2 python scripts/compare_dqn_reference.py
    CHECKPOINT --replay SNAPSHOT/replay.npz --output REPORT.json

This checks arithmetic with fresh Adam states, not full training trajectories.
"""

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from classicsrl.artifacts import digest, load_checkpoint, runtime_metadata, write_json
from classicsrl.config import Config
from classicsrl.dqn import QNetwork, update
from classicsrl.replay import ReplayBuffer


class ImageSpace(gym.Env):
    """Supply the image/action spaces without starting an emulator or collecting data."""

    def __init__(self, actions):
        self.observation_space = gym.spaces.Box(0, 255, (4, 84, 84), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(actions)


def compare(checkpoint, replay_path, steps=20, seed=20261107):
    import stable_baselines3 as sb3
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.type_aliases import ReplayBufferSamples

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    saved = load_checkpoint(checkpoint, "cpu")
    config = Config(**saved["config"])
    if config.double_dqn:
        raise ValueError("the SB3 arithmetic audit supports vanilla DQN only")
    if config.batch_size < 3:
        raise ValueError("the boundary fixture requires a batch size of at least three")
    online = QNetwork(saved["num_actions"])
    online.load_state_dict(saved["online"])
    target = copy.deepcopy(online).requires_grad_(False)
    optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer.load(replay_path)
    rng = np.random.default_rng(seed)
    reference = sb3.DQN(
        "CnnPolicy",
        ImageSpace(saved["num_actions"]),
        buffer_size=config.batch_size,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        max_grad_norm=config.max_grad_norm,
        device="cpu",
        seed=seed,
    )
    reference.set_logger(configure(format_strings=[]))
    layers = {
        0: "features_extractor.cnn.0",
        2: "features_extractor.cnn.2",
        4: "features_extractor.cnn.4",
        7: "features_extractor.linear.0",
        9: "q_net.0",
    }
    mapping = {
        f"layers.{index}.{part}": f"{name}.{part}"
        for index, name in layers.items()
        for part in ("weight", "bias")
    }
    assert set(mapping) == set(online.state_dict())
    assert set(mapping.values()) == set(reference.q_net.state_dict())
    initial = {name: online.state_dict()[key].clone() for key, name in mapping.items()}
    reference.q_net.load_state_dict(initial)
    reference.q_net_target.load_state_dict(initial)
    errors = dict.fromkeys(("q_values", "loss", "gradients", "weights", "adam_state"), 0.0)
    holder = [None]
    reference.replay_buffer.sample = lambda batch_size, **kwargs: holder[0]
    batch_digests = []
    for step in range(steps):
        batch = replay.sample(config.batch_size, rng=rng)
        # Cover true termination, time limits, and both flags on the same real images.
        batch.terminated[:3] = [True, False, True]
        batch.truncated[:3] = [False, True, True]
        holder[0] = ReplayBufferSamples(
            observations=torch.from_numpy(batch.observations),
            actions=torch.from_numpy(batch.actions[:, None]),
            next_observations=torch.from_numpy(batch.next_observations),
            dones=torch.from_numpy(batch.terminated[:, None].astype(np.float32)),
            rewards=torch.from_numpy(batch.rewards[:, None]),
        )
        batch_digests.append(hashlib.sha256(batch.observations.tobytes()).hexdigest())
        with torch.no_grad():
            error = online(holder[0].observations) - reference.q_net(holder[0].observations)
            errors["q_values"] = max(errors["q_values"], error.abs().max().item())
        actual = update(online, target, optimizer, batch, "cpu", config.gamma, config.max_grad_norm)
        reference.train(gradient_steps=1, batch_size=config.batch_size)
        errors["loss"] = max(
            errors["loss"], abs(actual["td_loss"] - reference.logger.name_to_value["train/loss"])
        )
        reference_parameters = dict(reference.q_net.named_parameters())
        for key, parameter in online.named_parameters():
            expected = reference_parameters[mapping[key]]
            for field, value in (("weights", parameter), ("gradients", parameter.grad)):
                other = expected if field == "weights" else expected.grad
                errors[field] = max(errors[field], (value - other).abs().max().item())
            for name, value in optimizer.state[parameter].items():
                other = reference.policy.optimizer.state[expected][name]
                errors["adam_state"] = max(errors["adam_state"], (value - other).abs().max().item())
        if (step + 1) % 5 == 0:
            target.load_state_dict(online.state_dict())
            reference.q_net_target.load_state_dict(reference.q_net.state_dict())
    return {
        "schema_version": 1,
        "scope": "fixed batches; fresh Adam; manually synchronized targets every five updates",
        "boundary_fixture": "first three masks: termination, truncation, both; real replay images",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "replay": str(replay_path),
        "replay_sha256": digest(replay_path),
        "harness_sha256": digest(Path(__file__)),
        "reference_version": sb3.__version__,
        "reference_dqn_sha256": digest(Path(inspect.getfile(sb3.DQN))),
        "runtime": runtime_metadata("cpu"),
        "updates": steps,
        "seed": seed,
        "batch_observation_sha256": batch_digests,
        "config": config.to_dict(),
        "max_absolute_errors": errors,
        "tolerance": 1e-6,
        "passed": all(value <= 1e-6 for value in errors.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    result = compare(args.checkpoint, args.replay, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("reference arithmetic comparison failed")


if __name__ == "__main__":
    main()
