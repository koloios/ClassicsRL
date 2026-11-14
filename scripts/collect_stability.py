"""Collect numeric stability evidence from local runs; never package game data."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from classicsrl.artifacts import digest, load_checkpoint, public_report, write_json
from classicsrl.diagnostics import activation_metrics
from classicsrl.dqn import QNetwork
from classicsrl.replay import ReplayBuffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--public", action="store_true", help="omit machine details and private paths"
    )
    parser.add_argument("--probe-label", help="description of the fixed replay probe's source")
    parser.add_argument("--diagnosis-report", type=Path, action="append", default=[])
    args = parser.parse_args()
    torch.set_num_threads(4)
    replay = ReplayBuffer.load(args.replay)
    sample = replay.sample(128, rng=np.random.default_rng(20261107))
    report = {
        "schema_version": 1,
        "probe": {
            "samples": 128,
            "seed": 20261107,
            "replay_sha256": digest(args.replay),
            "replay_source": args.probe_label or str(args.replay),
            "pixel_min": int(sample.observations.min()),
            "pixel_max": int(sample.observations.max()),
            "distinct_observations": len(np.unique(sample.observations, axis=0)),
        },
        "runs": [],
    }
    for run in args.runs:
        metadata = json.loads((run / "run.json").read_text())
        segments = [json.loads(line) for line in (run / "segments.jsonl").read_text().splitlines()]
        starts = [row for row in segments if row["event"] == "start"]
        config = starts[-1]["config"]
        if (
            segments[-1].get("event") != "stop"
            or segments[-1].get("agent_steps", 0) < config["total_steps"]
        ):
            raise ValueError(f"Run has not completed its target budget: {run}")
        entry = {
            "name": run.name,
            "run_id": metadata["run_id"],
            "implementation": metadata.get("implementation", "classicsrl.DQN"),
            "comparison_limits": metadata.get("comparison_limits", []),
            "config": config,
            "training_segments": starts,
            "checkpoints": [],
            "evaluations": [],
        }
        for path in sorted((run / "checkpoints").glob("*.pt")):
            saved = load_checkpoint(path, "cpu")
            model = QNetwork(saved["num_actions"])
            model.load_state_dict(saved["online"])
            entry["checkpoints"].append(
                {
                    "step": saved["counters"]["step"],
                    "checkpoint_sha256": digest(path),
                    "probe": activation_metrics(model, sample.observations, "cpu"),
                }
            )
        for row in map(json.loads, (run / "evaluations.jsonl").read_text().splitlines()):
            entry["evaluations"].append(
                {
                    key: row[key]
                    for key in ("agent_steps", "policy", "mean_return", "episodes", "environment")
                }
            )
        entry["heldout_evaluations"] = []
        for path in sorted((run / "episodes").glob("heldout-*/evaluation.json")):
            result = json.loads(path.read_text())
            if result.get("evaluation_role") != "heldout":
                raise ValueError(f"Expected a labeled held-out evaluation: {path}")
            if result["environment"] != metadata["environment"]:
                raise ValueError(f"Held-out evaluation changed the game protocol: {path}")
            entry["heldout_evaluations"].append(result)
        report["runs"].append(entry)
    report["diagnosis_reports"] = []
    for path in args.diagnosis_report:
        comparison = json.loads(path.read_text())
        checkpoints = {
            row["checkpoint_sha256"] for run in report["runs"] for row in run["checkpoints"]
        }
        if (
            comparison["replay_sha256"] != report["probe"]["replay_sha256"]
            or comparison["checkpoint_sha256"] not in checkpoints
        ):
            raise ValueError("Diagnosis does not match the collected artifacts")
        report["diagnosis_reports"].append(comparison)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, public_report(report) if args.public else report)


if __name__ == "__main__":
    main()
