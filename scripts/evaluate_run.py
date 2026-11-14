"""Lock a completed run's validation-selected policy before fresh full-game tests."""

import argparse
import json
from pathlib import Path

import torch

from classicsrl.artifacts import digest, load_checkpoint, runtime_metadata, write_json
from classicsrl.config import Config
from classicsrl.dqn import QNetwork, resolve_device
from classicsrl.envs import protocol
from classicsrl.evaluation import evaluate, select_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=5_000_000)
    parser.add_argument("--label", default="long-baseline")
    parser.add_argument("--record-seed", type=int)
    args = parser.parse_args()
    if args.episodes <= 0 or args.seed < 0 or not args.label.replace("-", "").isalnum():
        parser.error("use positive episodes, a nonnegative seed, and an alphanumeric label")
    metadata = json.loads((args.run / "run.json").read_text())
    segments = [json.loads(s) for s in (args.run / "segments.jsonl").read_text().splitlines()]
    target = [row for row in segments if row["event"] == "start"][-1]["config"]["total_steps"]
    last = segments[-1]
    if last["event"] != "stop" or last["agent_steps"] < target or last.get("interrupted"):
        raise ValueError("finish the declared training budget before final checkpoint selection")
    rows = [json.loads(s) for s in (args.run / "evaluations.jsonl").read_text().splitlines()]
    selected = select_validation(rows, metadata["environment"], last["agent_steps"])
    checkpoint = args.run / selected["checkpoint"]
    lock = args.run / f"test-selection-{args.label}.json"
    if lock.exists():
        raise ValueError("this test is already locked; preserve its selection and results")
    selection = {
        "rule": "highest matching validation mean, ties prefer latest checkpoint",
        "checkpoint": selected["checkpoint"],
        "checkpoint_sha256": digest(checkpoint),
        "agent_steps": selected["agent_steps"],
        "validation_mean": selected["mean_return"],
        "completed_training_steps": last["agent_steps"],
        "test_seeds": list(range(args.seed, args.seed + args.episodes)),
    }
    device = resolve_device(args.device)
    saved = load_checkpoint(checkpoint, device)
    config = Config(**saved["config"])
    if saved["run_id"] != metadata["run_id"] or protocol(config) != metadata["environment"]:
        raise ValueError("selected checkpoint does not match the run and evaluation protocol")
    torch.set_num_threads(config.torch_threads)
    network = QNetwork(saved["num_actions"]).to(device)
    network.load_state_dict(saved["online"])
    write_json(lock, selection)
    print("LOCKED", json.dumps(selection), flush=True)
    for policy in ("greedy", "random"):
        result = evaluate(network, config, device, selection["test_seeds"], policy=policy)
        result.update(
            evaluation_role="heldout",
            selection=selection,
            checkpoint=checkpoint.name,
            checkpoint_sha256=selection["checkpoint_sha256"],
            agent_steps=saved["counters"]["step"],
            run_id=saved["run_id"],
            implementation=metadata.get("implementation", "classicsrl.DQN"),
            runtime=runtime_metadata(device),
            full_game_wins=sum(r["terminated"] and r["return"] > 0 for r in result["episodes"]),
        )
        folder = args.run / "episodes" / f"heldout-{args.label}-{policy}"
        folder.mkdir(parents=True, exist_ok=False)
        write_json(folder / "evaluation.json", result)
        print(policy, result["mean_return"], "wins", result["full_game_wins"], flush=True)
    if args.record_seed is not None:
        folder = args.run / "episodes" / f"demo-{args.label}"
        result = evaluate(network, config, device, [args.record_seed], record_dir=folder)
        result.update(
            evaluation_role="demonstration",
            checkpoint=checkpoint.name,
            run_id=saved["run_id"],
            agent_steps=saved["counters"]["step"],
        )
        write_json(folder / "evaluation.json", result)
        print("Demo", result["mean_return"], flush=True)


if __name__ == "__main__":
    main()
