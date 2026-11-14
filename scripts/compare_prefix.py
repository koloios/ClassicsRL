"""Evaluate saved policies at a common available budget after a run interruption.

This does not mark an incomplete training run complete or resume it without replay.
Both checkpoint selections are locked before any new test games are inspected.
"""

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
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", default=50, type=int)
    parser.add_argument("--seed", default=7_000_000, type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if len(args.runs) < 2 or args.budget <= 0 or args.episodes <= 0 or args.seed < 0:
        parser.error(
            "provide at least two runs, a positive budget/episode count, and nonnegative seed"
        )
    selections, environment = [], None
    for run in args.runs:
        metadata = json.loads((run / "run.json").read_text())
        rows = [json.loads(s) for s in (run / "evaluations.jsonl").read_text().splitlines()]
        matching_steps = [
            row["agent_steps"]
            for row in rows
            if row.get("evaluation_role", "validation") == "validation"
            and row["policy"] == "greedy"
            and row["environment"] == metadata["environment"]
        ]
        if not matching_steps or max(matching_steps) < args.budget:
            raise ValueError(f"Run has not reached the common budget: {run}")
        if environment is not None and environment != metadata["environment"]:
            raise ValueError("comparison requires the same game protocol")
        environment = metadata["environment"]
        selected = select_validation(rows, environment, args.budget)
        checkpoint = run / selected["checkpoint"]
        saved = load_checkpoint(checkpoint, "cpu")
        if (
            saved["run_id"] != metadata["run_id"]
            or protocol(Config(**saved["config"])) != environment
        ):
            raise ValueError("checkpoint does not match its source run and game protocol")
        segments = [json.loads(s) for s in (run / "segments.jsonl").read_text().splitlines()]
        starts = [row for row in segments if row["event"] == "start"]
        selections.append(
            {
                "source_run": str(run),
                "run_id": metadata["run_id"],
                "implementation": metadata.get("implementation", "classicsrl.DQN"),
                "declared_training_budget": starts[-1]["config"]["total_steps"],
                "source_max_validation_step": max(matching_steps),
                "source_training_segments": starts,
                "comparison_limits": metadata.get("comparison_limits", []),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest(checkpoint),
                "selected_steps": saved["counters"]["step"],
                "selected_updates": saved["counters"]["updates"],
                "validation_mean": selected["mean_return"],
            }
        )
    if len({row["run_id"] for row in selections}) != len(selections):
        raise ValueError("provide distinct source runs")
    device = resolve_device(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "kind": "posthoc_common_budget_after_interruption",
        "common_available_budget": args.budget,
        "selection_rule": "highest validation mean within the common budget; ties prefer latest",
        "test_seeds": list(range(args.seed, args.seed + args.episodes)),
        "environment": environment,
        "selections": selections,
        "harness_sha256": digest(Path(__file__)),
        "runtime": runtime_metadata(device),
        "limitations": [
            "The common budget was chosen after an external interruption, before these tests.",
            "This is not completion of either source run's larger declared training budget.",
            "One training seed and differing recovery/RNG behavior do not establish superiority.",
        ],
    }
    write_json(args.output / "selection.json", report)
    print("LOCKED", [(r["implementation"], r["selected_steps"]) for r in selections], flush=True)
    report["evaluations"] = []
    for index, selection in enumerate(selections):
        saved = load_checkpoint(Path(selection["checkpoint"]), device)
        config = Config(**saved["config"])
        torch.set_num_threads(config.torch_threads)
        network = QNetwork(saved["num_actions"]).to(device)
        network.load_state_dict(saved["online"])
        result = evaluate(network, config, device, report["test_seeds"])
        result.update(
            evaluation_role="posthoc_heldout",
            selection=selection,
            full_game_wins=sum(r["terminated"] and r["return"] > 0 for r in result["episodes"]),
        )
        write_json(args.output / f"policy-{index}.json", result)
        report["evaluations"].append(result)
        print(
            selection["implementation"], result["mean_return"], result["full_game_wins"], flush=True
        )
    baseline = evaluate(network, config, device, report["test_seeds"], policy="random")
    baseline["evaluation_role"] = "posthoc_heldout"
    baseline["full_game_wins"] = sum(
        r["terminated"] and r["return"] > 0 for r in baseline["episodes"]
    )
    report["random_baseline"] = baseline
    write_json(args.output / "comparison.json", report)
    print("COMPARISON SAVED", args.output, flush=True)


if __name__ == "__main__":
    main()
