"""A single explicit DQN interaction loop, shared by Mac and Colab runs."""

import copy
import json
import random
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from classicsrl.artifacts import (
    append_json,
    create_run,
    restore_snapshot,
    runtime_metadata,
    save_checkpoint,
    save_snapshot,
    sync_run,
    verify_snapshot,
)
from classicsrl.config import Config
from classicsrl.diagnostics import activation_metrics
from classicsrl.dqn import QNetwork, epsilon_at, resolve_device, select_action, update
from classicsrl.envs import make_env
from classicsrl.evaluation import evaluate
from classicsrl.replay import ReplayBuffer


def train(config: Config, run_dir: Path, resume=None, persist_dir=None, env_factory=make_env):
    run_dir = Path(run_dir)
    if resume is not None:
        resume = Path(resume)
        verify_snapshot(resume)
        saved = json.loads((resume / "state.json").read_text())
        saved_config = Config(**saved["config"]).to_dict()
        allowed_changes = {"total_steps", "device", "torch_threads"}
        for key, value in config.to_dict().items():
            if key not in allowed_changes and value != saved_config[key]:
                raise ValueError(f"resume would change {key}; create a new experiment instead")
        if config.total_steps <= saved["counters"]["step"]:
            raise ValueError("total_steps must exceed the saved step counter")
        meta = json.loads((run_dir / "run.json").read_text())
        if meta["run_id"] != saved["run_id"]:
            raise ValueError("snapshot does not belong to this run")
    device = resolve_device(config.device)
    torch.set_num_threads(config.torch_threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    action_rng = np.random.default_rng(config.seed + 1)
    env = env_factory(config)
    try:
        if env.observation_space.shape != (4, 84, 84) or env.observation_space.dtype != np.uint8:
            raise ValueError("DQN expects four uint8 84x84 frames")
        network = QNetwork(env.action_space.n).to(device)
        target = copy.deepcopy(network).requires_grad_(False).eval()
        optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
        if resume is None:
            meta = create_run(run_dir, config, device)
            replay = ReplayBuffer(config.replay_capacity, seed=config.seed + 2)
            counters = {
                "step": 0,
                "updates": 0,
                "raw_frames": 0,
                "train_seconds": 0.0,
                "episodes": 0,
                "resets": 0,
            }
        else:
            state, replay = restore_snapshot(resume, network, target, optimizer, device, action_rng)
            counters = state["counters"].copy()
        segment_id = uuid.uuid4().hex
        append_json(
            run_dir / "segments.jsonl",
            {
                "schema_version": 1,
                "segment_id": segment_id,
                "event": "start",
                "config": config.to_dict(),
                "device": str(device),
                "runtime": runtime_metadata(device),
                "resume_snapshot": resume.name if resume else None,
                "agent_steps": counters["step"],
                "reset_seed": config.seed + counters["resets"],
            },
        )
        observation, _ = env.reset(seed=config.seed + counters["resets"])
        counters["resets"] += 1
        replay.begin_episode(observation[-1])
        base_seconds, base_frames = counters["train_seconds"], counters["raw_frames"]
        started = time.perf_counter()
        excluded_seconds = 0.0
        previous_seconds, previous_step = base_seconds, counters["step"]
        episode_return, episode_length = 0.0, 0
        last_update, last_evaluation = {}, None
        persistent = Path(persist_dir) / meta["run_id"] if persist_dir else None
        if persistent is not None and (
            run_dir.resolve() == persistent.resolve()
            or run_dir.resolve() in persistent.resolve().parents
        ):
            raise ValueError("persistent storage must be outside the active run directory")
        interrupted = False

        def refresh_counters():
            counters["train_seconds"] = (
                base_seconds + time.perf_counter() - started - excluded_seconds
            )
            counters["raw_frames"] = base_frames + env.get_wrapper_attr("raw_frames")

        def log(metric, value):
            append_json(
                run_dir / "metrics.jsonl",
                {
                    "schema_version": 1,
                    "segment_id": segment_id,
                    "agent_steps": counters["step"],
                    "updates": counters["updates"],
                    "raw_frames": counters["raw_frames"],
                    "train_seconds": counters["train_seconds"],
                    "metric": metric,
                    "value": float(value),
                },
            )

        def checkpoint():
            return save_checkpoint(run_dir, network, config, counters, meta["run_id"], segment_id)

        def snapshot():
            path = save_snapshot(
                run_dir,
                network,
                target,
                optimizer,
                replay,
                config,
                counters,
                meta["run_id"],
                action_rng,
            )
            if persistent is not None:
                sync_run(run_dir, persistent)
            return path

        def run_evaluation(checkpoint_path):
            nonlocal excluded_seconds, last_evaluation
            evaluation_start = time.perf_counter()
            result = evaluate(
                network,
                config,
                device,
                range(config.eval_seed, config.eval_seed + config.eval_episodes),
                env_factory=env_factory,
            )
            excluded_seconds += time.perf_counter() - evaluation_start
            result.update(
                {
                    "checkpoint": checkpoint_path.relative_to(run_dir).as_posix(),
                    "agent_steps": counters["step"],
                    "run_id": meta["run_id"],
                }
            )
            append_json(run_dir / "evaluations.jsonl", result)
            log("eval/mean_return", result["mean_return"])
            last_evaluation = counters["step"]

        try:
            while counters["step"] < config.total_steps:
                epsilon = epsilon_at(
                    counters["step"],
                    config.epsilon_start,
                    config.epsilon_end,
                    config.exploration_steps,
                )
                action = select_action(
                    network, observation, epsilon, action_rng, device, env.action_space.n
                )
                next_observation, reward, terminated, truncated, _ = env.step(action)
                replay.add(action, reward, next_observation[-1], terminated, truncated)
                observation = next_observation
                counters["step"] += 1
                episode_return += float(reward)
                episode_length += 1
                if (
                    counters["step"] >= config.learning_starts
                    and counters["step"] % config.train_every == 0
                    and len(replay) >= config.batch_size
                ):
                    last_update = update(
                        network,
                        target,
                        optimizer,
                        replay.sample(config.batch_size),
                        device,
                        config.gamma,
                        config.max_grad_norm,
                        double_dqn=config.double_dqn,
                    )
                    counters["updates"] += 1
                if counters["step"] % config.target_update_every == 0:
                    target.load_state_dict(network.state_dict())
                refresh_counters()
                if terminated or truncated:
                    counters["episodes"] += 1
                    log("train/episode_return", episode_return)
                    log("train/episode_length", episode_length)
                    episode_return, episode_length = 0.0, 0
                    observation, _ = env.reset(seed=config.seed + counters["resets"])
                    counters["resets"] += 1
                    replay.begin_episode(observation[-1])
                if (
                    counters["step"] % config.log_every == 0
                    or counters["step"] == config.total_steps
                ):
                    refresh_counters()
                    seconds = counters["train_seconds"] - previous_seconds
                    speed = (counters["step"] - previous_step) / max(seconds, 1e-9)
                    log("train/epsilon", epsilon)
                    log("train/steps_per_second", speed)
                    log("replay/transitions", len(replay))
                    for name, value in last_update.items():
                        log(f"train/{name}", value)
                    # A separate deterministic probe must not advance training replay RNG.
                    if len(replay):
                        probe_rng = np.random.default_rng([config.seed, counters["step"], 3])
                        probe = replay.sample(config.batch_size, rng=probe_rng)
                        metrics = activation_metrics(network, probe.observations, device)
                        for name, value in metrics.items():
                            log(f"train/{name}", value)
                    print(
                        f"step={counters['step']} updates={counters['updates']} "
                        f"epsilon={epsilon:.3f} steps/s={speed:.1f}",
                        flush=True,
                    )
                    previous_seconds, previous_step = counters["train_seconds"], counters["step"]
                if config.eval_every and counters["step"] % config.eval_every == 0:
                    run_evaluation(checkpoint())
                if counters["step"] % config.checkpoint_every == 0:
                    checkpoint()
                if (
                    counters["step"] % config.snapshot_every == 0
                    and counters["step"] < config.total_steps
                ):
                    snapshot()
        except KeyboardInterrupt:
            interrupted = True
            print("Stopping after interruption; saving recoverable training state.", flush=True)
        refresh_counters()
        checkpoint_path = checkpoint()
        if config.eval_every and last_evaluation != counters["step"] and not interrupted:
            run_evaluation(checkpoint_path)
        append_json(
            run_dir / "segments.jsonl",
            {
                "schema_version": 1,
                "segment_id": segment_id,
                "event": "stop",
                "agent_steps": counters["step"],
                "interrupted": interrupted,
                "unfinished_episode_length": episode_length,
                "unfinished_episode_return": episode_return,
            },
        )
        snapshot_path = snapshot()
        print(f"Checkpoint: {checkpoint_path}\nSnapshot: {snapshot_path}", flush=True)
        return {
            "run_id": meta["run_id"],
            "counters": counters.copy(),
            "checkpoint": checkpoint_path,
            "snapshot": snapshot_path,
        }
    finally:
        env.close()
