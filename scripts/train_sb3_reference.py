"""Optional end-to-end SB3 DQN control using the ClassicsRL game protocol.

Install stable-baselines3==2.9.2 separately. This runner retains SB3's native
collection, replay, target synchronization, and optimization. Recovery snapshots
include native SB3 model/optimizer state, replay, and RNG state. They start a fresh
emulator episode; exported .pt files remain playback-only.
"""

import argparse
import gzip
import inspect
import io
import json
import os
import random
import shutil
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
from classicsrl.config import Config, load_config
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


SNAPSHOT_FILES = {"state.json", "model.zip", "replay.pkl.gz", "rng.pt"}


def verify_reference_snapshot(path):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("kind") != "sb3-dqn-v1" or set(manifest.get("files", {})) != SNAPSHOT_FILES:
        raise ValueError("unsupported or incomplete SB3 snapshot")
    for name, entry in manifest["files"].items():
        file = path / name
        if file.stat().st_size != entry["bytes"] or digest(file) != entry["sha256"]:
            raise ValueError(f"SB3 snapshot integrity check failed: {name}")
    return json.loads((path / "state.json").read_text())


def save_reference_snapshot(run, model, config, run_id, train_seconds):
    name = f"step-{model.num_timesteps:012d}-{uuid.uuid4().hex[:8]}"
    parent = run / "reference-snapshots"
    parent.mkdir(exist_ok=True)
    temporary = parent / f".{name}.tmp"
    temporary.mkdir()
    numpy = np.random.get_state()
    state = {
        "run_id": run_id,
        "config": config.to_dict(),
        "reference_version": __version__,
        "device": str(model.device),
        "agent_steps": model.num_timesteps,
        "updates": model._n_updates,
        "train_seconds": train_seconds,
        "resume_mode": "fresh_episode",
        "python_rng": random.getstate(),
        "numpy_rng": [numpy[0], numpy[1].tolist(), *numpy[2:]],
        "action_rng": model.action_space.np_random.bit_generator.state,
    }
    rng = {"cpu": torch.get_rng_state()}
    if model.device.type == "cuda":
        rng["cuda"] = torch.cuda.get_rng_state_all()
    elif model.device.type == "mps":
        rng["mps"] = torch.mps.get_rng_state()
    write_json(temporary / "state.json", state)
    torch.save(rng, temporary / "rng.pt")
    model.save(temporary / "model.zip")
    with gzip.open(temporary / "replay.pkl.gz", "wb", compresslevel=1) as file:
        with io.BufferedWriter(file) as buffered:
            model.save_replay_buffer(buffered)
    files = {}
    for file_name in SNAPSHOT_FILES:
        file = temporary / file_name
        with file.open("rb") as stream:
            os.fsync(stream.fileno())
        files[file_name] = {"bytes": file.stat().st_size, "sha256": digest(file)}
    write_json(temporary / "manifest.json", {"kind": "sb3-dqn-v1", "files": files})
    destination = parent / name
    temporary.replace(destination)
    return destination


def restore_reference_snapshot(path, config, metadata, env, device):
    state = verify_reference_snapshot(path)
    if state["run_id"] != metadata["run_id"]:
        raise ValueError("snapshot does not belong to this run")
    if state["reference_version"] != __version__:
        raise ValueError("resume requires the same SB3 version")
    for key, value in config.to_dict().items():
        if key not in {"device", "torch_threads"} and value != state["config"][key]:
            raise ValueError(f"resume would change {key}; create a new experiment instead")
    if config.total_steps <= state["agent_steps"]:
        raise ValueError("the declared training budget is already complete")
    # SB3 model/replay loaders use pickle: load only snapshots from trusted runs.
    model = DQN.load(path / "model.zip", env=env, device=device, force_reset=True)
    with gzip.open(path / "replay.pkl.gz", "rb") as file:
        with io.BufferedReader(file) as buffered:
            model.load_replay_buffer(buffered)
    if (model.num_timesteps, model._n_updates) != (state["agent_steps"], state["updates"]):
        raise ValueError("SB3 snapshot counters disagree")
    model.get_env().seed(config.seed + model.num_timesteps)
    model.action_space.np_random.bit_generator.state = state["action_rng"]
    random.setstate((state["python_rng"][0], tuple(state["python_rng"][1]), state["python_rng"][2]))
    numpy = state["numpy_rng"]
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:]))
    rng = torch.load(path / "rng.pt", map_location="cpu", weights_only=True)
    torch.set_rng_state(rng["cpu"])
    if state["device"] == str(model.device):
        if model.device.type == "cuda":
            torch.cuda.set_rng_state_all(rng["cuda"])
        elif model.device.type == "mps":
            torch.mps.set_rng_state(rng["mps"])
    return model, state


def sync_control(run, destination):
    sync_run(run, destination)
    for snapshot in (run / "reference-snapshots").glob("step-*"):
        verify_reference_snapshot(snapshot)
        output = destination / "reference-snapshots" / snapshot.name
        if output.exists():
            verify_reference_snapshot(output)
            continue
        temporary = output.with_name(f".{output.name}.tmp")
        shutil.copytree(snapshot, temporary, dirs_exist_ok=True)
        verify_reference_snapshot(temporary)
        for file in temporary.iterdir():
            with file.open("rb") as stream:
                os.fsync(stream.fileno())
        temporary.replace(output)


class RecordControl(BaseCallback):
    def __init__(
        self,
        config,
        run,
        metadata,
        persist_dir=None,
        resume=None,
        train_seconds=0.0,
        env_factory=make_env,
    ):
        super().__init__()
        self.config, self.run, self.metadata = config, run, metadata
        self.segment = uuid.uuid4().hex
        self.persist = Path(persist_dir) / metadata["run_id"] if persist_dir else None
        self.started = time.perf_counter()
        self.excluded = 0.0
        self.last_logged, self.last_saved, self.last_evaluated = 0, 0, 0
        self.resume = resume
        self.env_factory = env_factory
        self.snapshot = resume
        self.base_seconds = train_seconds

    def _on_training_start(self):
        self.started = time.perf_counter()
        if self.resume is not None:
            self.last_logged = self.last_saved = self.last_evaluated = self.num_timesteps
        append_json(
            self.run / "segments.jsonl",
            {
                "event": "start",
                "segment_id": self.segment,
                "agent_steps": self.num_timesteps,
                "resume_snapshot": self.resume.name if self.resume else None,
                "reset_seed": self.config.seed + self.num_timesteps,
                "device": str(self.model.device),
                "config": self.config.to_dict(),
                "implementation": "stable_baselines3.DQN",
                "runtime": {
                    **runtime_metadata(self.model.device),
                    "reference_version": self.metadata["reference_version"],
                    "harness_sha256": digest(Path(__file__)),
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

    def record(self, final=False):
        started = time.perf_counter()
        try:
            self._record(final)
        finally:
            self.excluded += time.perf_counter() - started

    def _record(self, final):
        step = self.num_timesteps
        if not step:
            return
        elapsed = self.base_seconds + time.perf_counter() - self.started - self.excluded
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
        needs_snapshot = final or step % self.config.snapshot_every == 0
        needs_save = needs_snapshot or step % self.config.checkpoint_every == 0 or needs_eval
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
            result = evaluate(
                network,
                self.config,
                self.model.device,
                range(self.config.eval_seed, self.config.eval_seed + self.config.eval_episodes),
                env_factory=self.env_factory,
            )
            result.update(
                checkpoint=checkpoint.relative_to(self.run).as_posix(),
                agent_steps=step,
                run_id=self.metadata["run_id"],
                implementation="stable_baselines3.DQN",
            )
            append_json(self.run / "evaluations.jsonl", result)
            self.last_evaluated = step
            print(f"validation={result['mean_return']:.2f} at step={step}", flush=True)
        if needs_snapshot:
            self.snapshot = save_reference_snapshot(
                self.run, self.model, self.config, self.metadata["run_id"], elapsed
            )
            if self.persist:
                sync_control(self.run, self.persist)


def train_control(config, run, resume=None, persist_dir=None, env_factory=make_env):
    if __version__ != "2.9.2":
        raise ValueError("this reference runner requires stable-baselines3==2.9.2")
    if config.total_steps % config.train_every:
        raise ValueError("steps must be divisible by train_every for a matched stopping budget")
    for key in ("log_every", "checkpoint_every", "snapshot_every", "eval_every"):
        if getattr(config, key) % config.train_every:
            raise ValueError(f"{key} must be divisible by train_every")
    run = Path(run)
    if resume is not None:
        resume = Path(resume)
        metadata = json.loads((run / "run.json").read_text())
        if metadata.get("implementation") != "stable_baselines3.DQN":
            raise ValueError("not an SB3 reference run")
    else:
        metadata = None
    torch.set_num_threads(config.torch_threads)
    device = resolve_device(config.device)
    env = env_factory(config)
    try:
        if metadata is None:
            metadata = create_run(run, config, device)
            metadata.update(
                implementation="stable_baselines3.DQN",
                reference_version=__version__,
                reference_dqn_sha256=digest(Path(inspect.getfile(DQN))),
                harness_sha256=digest(Path(__file__)),
                replay_storage="SB3 stacked observations and next observations; timeouts enabled",
                resume_mode="fresh_episode",
                comparison_limits=[
                    "Shares ClassicsRL wrappers and evaluator; these require separate validation.",
                    "SB3 uses uniform warm-up actions and its native reset and RNG behavior.",
                    "SB3 synchronizes targets before a boundary update; ClassicsRL does so after.",
                    "Warm-up is adjusted by one step to match the first optimizer update.",
                    "Recovery restores learning state but starts a fresh emulator episode.",
                ],
            )
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
            train_seconds = 0.0
        else:
            model, state = restore_reference_snapshot(resume, config, metadata, env, device)
            train_seconds = state["train_seconds"]
        probe = torch.as_tensor(
            np.random.default_rng(20261108).integers(0, 256, (4, 4, 84, 84), dtype=np.uint8),
            device=device,
        )
        with torch.no_grad():
            error = (export_network(model)(probe) - model.q_net(probe)).abs().max().item()
        if not np.isfinite(error) or error > 1e-5:
            raise ValueError(f"policy export mismatch: {error}")
        metadata["policy_export_max_absolute_error"] = error
        write_json(run / "run.json", metadata)
        callback = RecordControl(
            config,
            run,
            metadata,
            persist_dir,
            resume=resume,
            train_seconds=train_seconds,
            env_factory=env_factory,
        )
        if callback.persist and (
            run.resolve() == callback.persist.resolve()
            or run.resolve() in callback.persist.resolve().parents
        ):
            raise ValueError("persistent storage must be outside the active run directory")
        interrupted = False
        try:
            model.learn(
                config.total_steps - model.num_timesteps,
                callback=callback,
                reset_num_timesteps=False,
            )
        except KeyboardInterrupt:
            # A partial rollout/update is unsafe to save. Keep the last verified boundary.
            interrupted = True
        append_json(
            run / "segments.jsonl",
            {
                "event": "stop",
                "segment_id": callback.segment,
                "agent_steps": model.num_timesteps,
                "interrupted": interrupted,
                "resume_snapshot": callback.snapshot.name if callback.snapshot else None,
            },
        )
        if callback.persist:
            sync_control(run, callback.persist)
        return {
            "status": "paused" if interrupted else "complete",
            "agent_steps": model.num_timesteps,
            "snapshot": callback.snapshot,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument(
        "--resume", type=Path, help="trusted reference-snapshots/<snapshot> directory"
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--persist-dir", type=Path)
    args = parser.parse_args()
    if args.resume:
        config = Config(**json.loads((args.resume / "state.json").read_text())["config"])
        run = args.run or args.resume.parent.parent
        if args.config:
            parser.error("--config cannot be combined with --resume")
    else:
        if args.config is None or args.run is None:
            parser.error("new runs require --config and --run")
        config, run = load_config(args.config), args.run
    changes = {
        key: value
        for key, value in {"total_steps": args.steps, "seed": args.seed}.items()
        if value is not None
    }
    config = replace(config, device=args.device, **changes)
    result = train_control(config, run, args.resume, args.persist_dir)
    print(f"Reference control {result['status']}: {run}", flush=True)
    if result["snapshot"]:
        print("Recovery snapshot:", result["snapshot"], flush=True)


if __name__ == "__main__":
    main()
