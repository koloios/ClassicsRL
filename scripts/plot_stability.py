"""Plot the numeric report; requires matplotlib, never loads game assets."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = json.loads(args.report.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    colors = {1e-4: "#a14f4f", 3e-5: "#2171b5"}
    markers = {0: "o", 1: "s", 2: "^"}
    for run in data["runs"]:
        rate, seed = run["config"]["learning_rate"], run["config"]["seed"]
        checkpoints = sorted(run["checkpoints"], key=lambda row: row["step"])
        scores = run["evaluations"]
        continuation = run["config"]["total_steps"] > 100000
        if continuation:
            checkpoints = [row for row in checkpoints if row["step"] >= 100000]
            scores = [row for row in scores if row["agent_steps"] >= 100000]
        axes[0].plot(
            [row["step"] / 1000 for row in checkpoints],
            [row["probe"]["features/conv3_active_fraction"] for row in checkpoints],
            color=colors[rate],
            marker=markers[seed],
            alpha=0.8,
            lw=1.5,
            ms=5,
            linestyle="--" if continuation else "-",
        )
        axes[1].plot(
            [row["agent_steps"] / 1000 for row in scores],
            [row["mean_return"] for row in scores],
            color=colors[rate],
            marker=markers[seed],
            alpha=0.8,
            lw=1.5,
            ms=5,
            linestyle="--" if continuation else "-",
        )
    axes[0].set(
        title="Visual features on a fixed replay probe",
        ylabel="Conv3 positive activation fraction",
        xlabel="Agent steps (thousands)",
        ylim=(-0.03, 1.03),
    )
    axes[1].set(
        title="Full greedy validation returns",
        ylabel="Mean episode return (5 seeds)",
        xlabel="Agent steps (thousands)",
        ylim=(-22, 22),
        yticks=[-21, -10, 0, 10, 21],
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D([], [], color=color, label=f"Learning rate {rate:g}")
        for rate, color in colors.items()
    ]
    handles += [
        Line2D([], [], color="#555", marker=marker, linestyle="none", label=f"Training seed {seed}")
        for seed, marker in markers.items()
    ]
    handles.append(Line2D([], [], color="#555", linestyle="--", label="Seed 0 continuation"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(
        "ClassicsRL DQN stability investigation · individual runs, no smoothing", fontsize=12
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
