"""Small command-line entry points for local and notebook use."""

import argparse
import json
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from classicsrl.config import Config, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="classicsrl")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="train or resume vanilla DQN")
    train_parser.add_argument("--config", type=Path)
    train_parser.add_argument("--run", type=Path)
    train_parser.add_argument("--resume", type=Path, help="complete training snapshot directory")
    train_parser.add_argument(
        "--steps", type=int, help="total target agent steps, including saved steps"
    )
    train_parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    train_parser.add_argument("--seed", type=int, help="independent training seed")
    train_parser.add_argument("--learning-rate", type=float)
    train_parser.add_argument(
        "--persist-dir", type=Path, help="periodic verified copies outside the run"
    )
    evaluate_parser = commands.add_parser("evaluate", help="evaluate a playback checkpoint")
    evaluate_parser.add_argument("checkpoint", type=Path)
    evaluate_parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    evaluate_parser.add_argument("--episodes", type=int, default=5)
    evaluate_parser.add_argument("--seed", type=int, default=2_000_000)
    evaluate_parser.add_argument("--policy", choices=("greedy", "random"), default="greedy")
    evaluate_parser.add_argument(
        "--max-steps", type=int, help="explicit evaluation episode cap override"
    )
    evaluate_parser.add_argument("--record-dir", type=Path)
    diagnose_parser = commands.add_parser("diagnose", help="inspect a checkpoint on saved replay")
    diagnose_parser.add_argument("checkpoint", type=Path)
    diagnose_parser.add_argument("--replay", required=True, type=Path)
    diagnose_parser.add_argument("--samples", default=128, type=int)
    diagnose_parser.add_argument("--seed", default=0, type=int)
    diagnose_parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    diagnose_parser.add_argument("--compare-device", choices=("cpu", "mps", "cuda"))
    diagnose_parser.add_argument("--output", type=Path)
    dashboard_parser = commands.add_parser(
        "dashboard", help="inspect local experiments in a browser"
    )
    dashboard_parser.add_argument("--runs", type=Path, default=Path("runs"))
    dashboard_parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            from classicsrl.training import train

            if args.config is not None:
                config = load_config(args.config)
            elif args.resume is not None:
                state = json.loads((args.resume / "state.json").read_text())
                config = Config(**state["config"])
            else:
                config = Config()
            changes = {}
            if args.steps is not None:
                changes["total_steps"] = args.steps
            if args.device is not None:
                changes["device"] = args.device
            if args.seed is not None:
                changes["seed"] = args.seed
            if args.learning_rate is not None:
                changes["learning_rate"] = args.learning_rate
            config = replace(config, **changes)
            if args.run is None:
                args.run = (
                    args.resume.parent.parent
                    if args.resume
                    else Path("runs") / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
                )
            train(config, args.run, resume=args.resume, persist_dir=args.persist_dir)
        elif args.command == "diagnose":
            from classicsrl.artifacts import write_json
            from classicsrl.diagnostics import diagnose

            devices = [args.device] + ([args.compare_device] if args.compare_device else [])
            result = diagnose(
                args.checkpoint, args.replay, samples=args.samples, seed=args.seed, devices=devices
            )
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                write_json(args.output, result)
            print(json.dumps(result, indent=2))
        elif args.command == "dashboard":
            try:
                import uvicorn

                from classicsrl.dashboard import create_app
            except ImportError as error:
                raise RuntimeError(
                    "install dashboard support with: uv sync --extra dashboard"
                ) from error
            if not 1 <= args.port <= 65535:
                raise ValueError("port must be between 1 and 65535")
            print(f"ClassicsRL dashboard: http://127.0.0.1:{args.port}", flush=True)
            uvicorn.run(
                create_app(args.runs), host="127.0.0.1", port=args.port, log_level="warning"
            )
        else:
            from classicsrl.artifacts import load_checkpoint
            from classicsrl.dqn import QNetwork, resolve_device
            from classicsrl.evaluation import evaluate

            if args.episodes <= 0 or args.seed < 0:
                raise ValueError("episodes must be positive and seed must be nonnegative")
            device = resolve_device(args.device)
            saved = load_checkpoint(args.checkpoint, device)
            config = Config(**saved["config"])
            if args.max_steps is not None:
                config = replace(config, max_episode_steps=args.max_steps)
            import torch

            torch.set_num_threads(config.torch_threads)
            network = QNetwork(saved["num_actions"]).to(device)
            network.load_state_dict(saved["online"])
            result = evaluate(
                network,
                config,
                device,
                range(args.seed, args.seed + args.episodes),
                policy=args.policy,
                record_dir=args.record_dir,
            )
            result.update(
                {
                    "run_id": saved["run_id"],
                    "checkpoint": args.checkpoint.name,
                    "agent_steps": saved["counters"]["step"],
                }
            )
            if args.record_dir:
                from classicsrl.artifacts import write_json

                write_json(args.record_dir / "evaluation.json", result)
            print(json.dumps(result, indent=2))
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        parser.exit(2, f"classicsrl: {error}\n")


if __name__ == "__main__":
    main()
