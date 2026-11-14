"""Optional end-to-end SB3 DQN control using the ClassicsRL game protocol.

Install stable-baselines3==2.9.2 separately. This runner retains SB3's native
collection, replay, target synchronization, and optimization. Exported .pt files
are playback policies, not resumable ClassicsRL training snapshots.
"""

import argparse
import inspect
import time
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN, __version__
from stable_baselines3.common.callbacks import BaseCallback

from classicsrl.artifacts import (
    append_json,
    create_run,
    digest,
    runtime_metadata,
    save_checkpoint,
    sync_run,
    write_json,
)
from classicsrl.config import load_config
from classicsrl.dqn import QNetwork, resolve_device
from classicsrl.envs import make_env
from classicsrl.evaluation import evaluate


def parameter_mapping():
    layers = {
        0: "features_extractor.cnn.0",
        2: "features_extractor.cnn.2",
        4: "features_extractor.cnn.4",
        7: "features_extractor.linear.0",
        9: "q_net.0",
    }
    return {
        f"layers.{index}.{part}": f"{name}.{part}"
        for index, name in layers.items()
        for part in ("weight", "bias")
    }


def export_network(model):
    # Constructing a playback network must not advance the training Torch RNG.
    with torch.random.fork_rng(devices=[]):
        network = QNetwork(model.action_space.n).to(model.device)
    source = model.q_net.state_dict()
    network.load_state_dict({key: source[name] for key, name in parameter_mapping().items()})
    return network


class RecordControl(BaseCallback):
    def __init__(self, config, run, metadata, persist_dir=None):
        super().__init__()
        self.config, self.run, self.metadata = config, run, metadata
        self.segment = uuid.uuid4().hex
        self.persist = Path(persist_dir) / metadata["run_id"] if persist_dir else None
        self.started = time.perf_counter()
        self.excluded = 0.0
        self.last_logged, self.last_saved, self.last_evaluated = 0, 0, 0

    def _on_training_start(self):
        append_json(
            self.run / "segments.jsonl",
            {
                "event": "start",
                "segment_id": self.segment,
                "agent_steps": 0,
                "device": str(self.model.device),
                "config": self.config.to_dict(),
                "implementation": "stable_baselines3.DQN",
                "runtime": {
                    **runtime_metadata(self.model.device),
                    "reference_version": self.metadata["reference_version"],
                    "harness_sha256": self.metadata["harness_sha256"],
                },
                "comparison_limits": self.metadata["comparison_limits"],
            },
        )

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        # This hook follows the previous rollout's gradient update. The on-step
        # callback runs before storing its transition and would be one update early.
        self.record()

    def _on_training_end(self):
        self.record(final=True)
        append_json(
            self.run / "segments.jsonl",
            {"event": "stop", "segment_id": self.segment, "agent_steps": self.num_timesteps},
        )
        if self.persist:
            sync_run(self.run, self.persist)

    def record(self, final=False):
        step = self.num_timesteps
        if not step:
            return
        elapsed = time.perf_counter() - self.started - self.excluded
        if step != self.last_logged and (final or step % self.config.log_every == 0):
            metrics = {"train/epsilon": self.model.exploration_rate}
            if "train/loss" in self.model.logger.name_to_value:
                metrics["train/td_loss"] = self.model.logger.name_to_value["train/loss"]
            for metric, value in metrics.items():
                append_json(
                    self.run / "metrics.jsonl",
                    {
                        "segment_id": self.segment,
                        "agent_steps": step,
                        "updates": self.model._n_updates,
                        "train_seconds": elapsed,
                        "metric": metric,
                        "value": float(value),
                    },
                )
            self.last_logged = step
            print(
                f"step={step} updates={self.model._n_updates} "
                f"epsilon={self.model.exploration_rate:.3f}",
                flush=True,
            )
        needs_eval = final or (self.config.eval_every and step % self.config.eval_every == 0)
        needs_save = final or step % self.config.checkpoint_every == 0 or needs_eval
        if not needs_save or step == self.last_saved:
            return
        network = export_network(self.model)
        checkpoint = save_checkpoint(
            self.run,
            network,
            self.config,
            {"step": step, "updates": self.model._n_updates, "train_seconds": elapsed},
            self.metadata["run_id"],
            self.segment,
        )
        self.last_saved = step
        if needs_eval and step != self.last_evaluated:
            started = time.perf_counter()
            result = evaluate(
                network,
                self.config,
                self.model.device,
                range(self.config.eval_seed, self.config.eval_seed + self.config.eval_episodes),
            )
            self.excluded += time.perf_counter() - started
            result.update(
                checkpoint=checkpoint.relative_to(self.run).as_posix(),
                agent_steps=step,
                run_id=self.metadata["run_id"],
                implementation="stable_baselines3.DQN",
            )
            append_json(self.run / "evaluations.jsonl", result)
            self.last_evaluated = step
            print(f"validation={result['mean_return']:.2f} at step={step}", flush=True)
        if self.persist and (final or step % self.config.snapshot_every == 0):
            sync_run(self.run, self.persist)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--persist-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    changes = {
        key: value
        for key, value in {"total_steps": args.steps, "seed": args.seed}.items()
        if value is not None
    }
    config = replace(config, device=args.device, **changes)
    if config.total_steps % config.train_every:
        parser.error("steps must be divisible by train_every for a matched stopping budget")
    torch.set_num_threads(config.torch_threads)
    device = resolve_device(args.device)
    metadata = create_run(args.run, config, device)
    metadata.update(
        implementation="stable_baselines3.DQN",
        reference_version=__version__,
        reference_dqn_sha256=digest(Path(inspect.getfile(DQN))),
        harness_sha256=digest(Path(__file__)),
        replay_storage="SB3 stacked observations and next observations; timeout handling enabled",
        resume_mode="not_supported_by_this_reference_runner",
        comparison_limits=[
            "Shares ClassicsRL wrappers and evaluator; these require separate validation.",
            "SB3 uses uniform random actions during warm-up and its native reset and RNG behavior.",
            "SB3 synchronizes targets before a boundary update; ClassicsRL does so after it.",
            "Warm-up threshold is adjusted by one step to match the first optimizer update.",
            "Playback weights persist periodically; these are not full recovery snapshots.",
        ],
    )
    write_json(args.run / "run.json", metadata)
    env = make_env(config)
    try:
        model = DQN(
            "CnnPolicy",
            env,
            device=device,
            seed=config.seed,
            buffer_size=config.replay_capacity,
            learning_rate=config.learning_rate,
            learning_starts=max(0, config.learning_starts - 1),
            batch_size=config.batch_size,
            train_freq=config.train_every,
            gradient_steps=1,
            gamma=config.gamma,
            target_update_interval=config.target_update_every,
            max_grad_norm=config.max_grad_norm,
            exploration_fraction=config.exploration_steps / config.total_steps,
            exploration_initial_eps=config.epsilon_start,
            exploration_final_eps=config.epsilon_end,
        )
        # Match the original ClassicsRL random initialization while keeping all
        # subsequent data collection, replay sampling, and updates inside SB3.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            initial = QNetwork(env.action_space.n)
        state = {name: initial.state_dict()[key] for key, name in parameter_mapping().items()}
        model.q_net.load_state_dict(state)
        model.q_net_target.load_state_dict(state)
        probe = torch.as_tensor(
            np.random.default_rng(20261108).integers(0, 256, (4, 4, 84, 84), dtype=np.uint8),
            device=device,
        )
        with torch.no_grad():
            error = (export_network(model)(probe) - model.q_net(probe)).abs().max().item()
        if not np.isfinite(error) or error > 1e-5:
            raise ValueError(f"policy export mismatch: {error}")
        metadata["policy_export_max_absolute_error"] = error
        write_json(args.run / "run.json", metadata)
        model.learn(
            config.total_steps, callback=RecordControl(config, args.run, metadata, args.persist_dir)
        )
        print("Completed reference control:", args.run, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
