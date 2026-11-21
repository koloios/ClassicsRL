"""Independent evaluation and color recordings aligned with agent decisions."""

import math
import uuid
from pathlib import Path

import numpy as np

from classicsrl.artifacts import append_json, write_json
from classicsrl.dqn import action_values
from classicsrl.envs import make_env, protocol


def select_validation(rows, environment, completed_steps):
    """Select within the validation budget, preferring the latest tied checkpoint."""
    eligible = [
        row
        for row in rows
        if row.get("evaluation_role", "validation") == "validation"
        and row["policy"] == "greedy"
        and row["environment"] == environment
        and row["agent_steps"] <= completed_steps
        and math.isfinite(row["mean_return"])
    ]
    if not eligible:
        raise ValueError("no matching full-game validation checkpoint")
    return max(eligible, key=lambda row: (row["mean_return"], row["agent_steps"]))


def evaluate(
    network,
    config,
    device,
    seeds,
    policy="greedy",
    record_dir=None,
    env_factory=make_env,
    *,
    completed=(),
    on_episode=None,
):
    """Evaluate seeded games, optionally continuing a saved prefix without video."""
    if policy not in {"greedy", "random"}:
        raise ValueError("policy must be greedy or random")
    seeds = list(seeds)
    if not seeds:
        raise ValueError("evaluation requires at least one episode seed")
    results = list(completed)
    if [item["seed"] for item in results] != seeds[: len(results)]:
        raise ValueError("saved episodes must match the start of the evaluation seed list")
    if record_dir is not None:
        if results:
            raise ValueError("resume is only supported for evaluations without recording")
        try:
            import imageio.v2 as imageio
        except ImportError as error:
            raise RuntimeError("install video support with: uv sync --extra video") from error
        record_dir = Path(record_dir)
        if record_dir.exists():
            raise ValueError("recording directory already exists; use a new directory")
        record_dir.mkdir(parents=True, exist_ok=False)
        write_json(
            record_dir / "protocol.json",
            {
                "environment": protocol(config),
                "policy": policy,
                "video_fps": 60 / config.frame_skip,
                "frame_sampling": "one RGB observation before each decision, plus final frame",
            },
        )
    env = env_factory(config, render_mode="rgb_array" if record_dir is not None else None)
    action_names = (
        env.unwrapped.get_action_meanings()
        if hasattr(env.unwrapped, "get_action_meanings")
        else [str(index) for index in range(env.action_space.n)]
    )
    was_training = network.training
    network.eval()
    try:
        for number, seed in enumerate(seeds[len(results) :], start=len(results)):
            rng = np.random.default_rng(seed)
            observation, _ = env.reset(seed=seed)
            score, steps = 0.0, 0
            writer = None
            video_path = None
            if record_dir is not None:
                video_path = record_dir / f"episode-{number:03d}.mp4"
                writer = imageio.get_writer(
                    video_path, fps=60 / config.frame_skip, macro_block_size=1
                )
            try:
                while True:
                    values = None
                    if policy == "greedy" or record_dir is not None:
                        values = action_values(network, observation, device)
                    action = (
                        int(values.argmax())
                        if policy == "greedy"
                        else int(rng.integers(env.action_space.n))
                    )
                    if writer is not None:
                        writer.append_data(env.render())
                    observation, reward, terminated, truncated, _ = env.step(action)
                    if record_dir is not None:
                        append_json(
                            record_dir / f"episode-{number:03d}.jsonl",
                            {
                                "frame_index": steps,
                                "time_seconds": steps * config.frame_skip / 60,
                                "action": action,
                                "action_name": action_names[action],
                                "action_values": values.tolist(),
                                "reward": float(reward),
                                "terminated": bool(terminated),
                                "truncated": bool(truncated),
                            },
                        )
                    score += float(reward)
                    steps += 1
                    if terminated or truncated:
                        if writer is not None:
                            writer.append_data(env.render())
                        break
            finally:
                if writer is not None:
                    writer.close()
            results.append(
                {
                    "seed": int(seed),
                    "return": score,
                    "length": steps,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "video": video_path.name if video_path else None,
                }
            )
            if on_episode is not None:
                on_episode(results[-1])
    finally:
        env.close()
        network.train(was_training)
    summary = {
        "schema_version": 1,
        "evaluation_id": uuid.uuid4().hex,
        "policy": policy,
        "episodes": results,
        "mean_return": float(np.mean([item["return"] for item in results])),
        "environment": protocol(config),
        "action_names": action_names,
    }
    if record_dir is not None:
        write_json(record_dir / "evaluation.json", summary)
    return summary
