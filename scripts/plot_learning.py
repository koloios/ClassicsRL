"""Plot individual full-game validation curves from collected numeric reports."""

import argparse
import json
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    colors = cycle(("#9299a2", "#252b35", "#3868d9", "#2a8a7b", "#9b6caa"))
    for path in args.reports:
        report = json.loads(Path(path).read_text())
        for run in report["runs"]:
            config = run["config"]
            segments = run["training_segments"]
            device = segments[0]["device"].upper()
            recipe = (
                "SB3 control"
                if run.get("implementation") == "stable_baselines3.DQN"
                else "ClassicsRL DQN (reference-inspired)"
                if config["learning_starts"] == 80_000
                else f"Target {config['target_update_every']:,} · lr {config['learning_rate']:g}"
            )
            label = f"{recipe} · {device} · seed {config['seed']}"
            color = next(colors)
            for index, segment in enumerate(segments):
                start = segment["agent_steps"]
                end = (
                    segments[index + 1]["agent_steps"]
                    if index + 1 < len(segments)
                    else float("inf")
                )
                rows = [
                    row
                    for row in run["evaluations"]
                    if row["policy"] == "greedy" and start < row["agent_steps"] <= end
                ]
                ax.plot(
                    [row["agent_steps"] / 1e6 for row in rows],
                    [row["mean_return"] for row in rows],
                    color=color,
                    marker="o",
                    markersize=4,
                    linewidth=1.8,
                    linestyle="--" if index else "-",
                    label=label if index == 0 else None,
                )
    ax.axhline(0, color="#aab1ba", linewidth=0.8)
    ax.set(
        xlabel="Agent steps (millions)",
        ylabel="Mean Pong return · five greedy validation games",
        ylim=(-22, 22),
        title="DQN learning experiments",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.15)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.supxlabel(
        "Individual runs; no smoothing. Dashed segments resume from a fresh episode.", fontsize=8
    )
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
