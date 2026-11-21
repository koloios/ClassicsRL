"""Lock a completed run's policy and resume its full-game tests after interruption."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from filelock import FileLock, Timeout

from classicsrl.artifacts import digest, load_checkpoint, runtime_metadata, write_json
from classicsrl.config import Config
from classicsrl.dqn import QNetwork, resolve_device
from classicsrl.envs import protocol
from classicsrl.evaluation import evaluate, select_validation


def read_json(path):
    return json.loads(path.read_text())


def lock_selection(run, lock, metadata, budget, seeds, folders):
    if lock.exists():
        selection = read_json(lock)
        if selection["test_seeds"] != seeds:
            raise ValueError(
                "test seeds differ from the locked selection; use the original command"
            )
        return selection, False
    if any(
        (folder / name).exists()
        for folder in folders
        for name in ("progress.json", "evaluation.json")
    ):
        raise ValueError("held-out results are missing their original selection lock")
    rows = [json.loads(line) for line in (run / "evaluations.jsonl").read_text().splitlines()]
    selected = select_validation(rows, metadata["environment"], budget)
    return {
        "rule": "highest matching validation mean, ties prefer latest checkpoint",
        "checkpoint": selected["checkpoint"],
        "checkpoint_sha256": digest(run / selected["checkpoint"]),
        "agent_steps": selected["agent_steps"],
        "validation_mean": selected["mean_return"],
        "completed_training_steps": budget,
        "test_seeds": seeds,
    }, True


def test_policy(network, config, device, policy, folder, identity, runtime):
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / "evaluation.json"
    progress = folder / "progress.json"
    seeds = identity["selection"]["test_seeds"]
    if final.exists():
        result = read_json(final)
        if (
            any(result.get(key) != value for key, value in identity.items())
            or result.get("policy") != policy
            or result.get("environment") != protocol(config)
            or [row["seed"] for row in result["episodes"]] != seeds
        ):
            raise ValueError(f"saved {policy} results do not match the locked evaluation")
        return result

    episodes = []
    if progress.exists():
        saved = read_json(progress)
        if (
            saved.get("identity") != identity
            or saved.get("policy") != policy
            or saved.get("environment") != protocol(config)
        ):
            raise ValueError(f"saved {policy} progress does not match the locked evaluation")
        for key in ("device", "python", "dependencies", "source_sha256"):
            if saved["runtime"][key] != runtime[key]:
                raise ValueError(f"cannot resume {policy} evaluation with changed {key}")
        runtime = saved["runtime"]
        episodes = saved["episodes"]

    def save_episode(episode):
        episodes.append(episode)
        write_json(
            progress,
            {
                "identity": identity,
                "policy": policy,
                "environment": protocol(config),
                "runtime": runtime,
                "episodes": episodes,
            },
        )

    result = evaluate(
        network, config, device, seeds, policy=policy, completed=episodes, on_episode=save_episode
    )
    result.update(
        identity,
        runtime=runtime,
        full_game_wins=sum(row["terminated"] and row["return"] > 0 for row in result["episodes"]),
    )
    write_json(final, result)
    progress.unlink(missing_ok=True)
    return result


def record_demo(network, config, device, seed, folder, identity):
    if folder.exists():
        result = read_json(folder / "evaluation.json")
        if [row["seed"] for row in result["episodes"]] != [seed] or any(
            result.get(key) != value for key, value in identity.items()
        ):
            raise ValueError("saved demonstration differs; use a new label")
        return result

    with TemporaryDirectory(dir=folder.parent, prefix=".recording-") as temporary:
        recording = Path(temporary) / "episode"
        result = evaluate(network, config, device, [seed], record_dir=recording)
        result.update(identity)
        write_json(recording / "evaluation.json", result)
        recording.rename(folder)
    return result


def evaluate_run(args):
    metadata = read_json(args.run / "run.json")
    segments = [json.loads(line) for line in (args.run / "segments.jsonl").read_text().splitlines()]
    target = [row for row in segments if row["event"] == "start"][-1]["config"]["total_steps"]
    last = segments[-1]
    if last["event"] != "stop" or last["agent_steps"] < target or last.get("interrupted"):
        raise ValueError("finish the declared training budget before final checkpoint selection")
    policies = ("greedy", "random")
    folders = [args.run / "episodes" / f"heldout-{args.label}-{policy}" for policy in policies]
    lock = args.run / f"test-selection-{args.label}.json"
    seeds = list(range(args.seed, args.seed + args.episodes))
    selection, is_new = lock_selection(
        args.run, lock, metadata, last["agent_steps"], seeds, folders
    )
    checkpoint = args.run / selection["checkpoint"]
    if not is_new and digest(checkpoint) != selection["checkpoint_sha256"]:
        raise ValueError("checkpoint checksum differs from the locked selection")
    device = resolve_device(args.device)
    saved = load_checkpoint(checkpoint, device)
    config = Config(**saved["config"])
    if (
        saved["run_id"] != metadata["run_id"]
        or protocol(config) != metadata["environment"]
        or saved["counters"]["step"] != selection["agent_steps"]
        or not selection["agent_steps"]
        <= selection["completed_training_steps"]
        <= last["agent_steps"]
    ):
        raise ValueError("selected checkpoint does not match the run and evaluation protocol")
    torch.set_num_threads(config.torch_threads)
    network = QNetwork(saved["num_actions"]).to(device)
    network.load_state_dict(saved["online"])
    if is_new:
        write_json(lock, selection)
    print("LOCKED", json.dumps(selection), flush=True)
    identity = {
        "evaluation_role": "heldout",
        "selection": selection,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": selection["checkpoint_sha256"],
        "agent_steps": saved["counters"]["step"],
        "run_id": saved["run_id"],
        "implementation": metadata.get("implementation", "classicsrl.DQN"),
    }
    runtime = runtime_metadata(device)
    for policy, folder in zip(policies, folders, strict=True):
        result = test_policy(network, config, device, policy, folder, identity, runtime)
        print(policy, result["mean_return"], "wins", result["full_game_wins"], flush=True)
    if args.record_seed is not None:
        demo_identity = {key: identity[key] for key in ("checkpoint", "run_id", "agent_steps")} | {
            "evaluation_role": "demonstration"
        }
        result = record_demo(
            network,
            config,
            device,
            args.record_seed,
            args.run / "episodes" / f"demo-{args.label}",
            demo_identity,
        )
        print("Demo", result["mean_return"], flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=5_000_000)
    parser.add_argument("--label", default="long-baseline")
    parser.add_argument("--record-seed", type=int)
    args = parser.parse_args(argv)
    if (
        args.episodes <= 0
        or args.seed < 0
        or (args.record_seed is not None and args.record_seed < 0)
        or not args.label.replace("-", "").isalnum()
    ):
        parser.error("use positive episodes, nonnegative seeds, and an alphanumeric label")
    if not (args.run / "run.json").is_file():
        parser.error("provide a run directory with run.json")
    try:
        with FileLock(args.run / f".test-{args.label}.lock", timeout=0):
            evaluate_run(args)
    except Timeout:
        parser.error("an evaluation for this label is already running")


if __name__ == "__main__":
    main()
