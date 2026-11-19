# ClassicsRL

[![CI](https://github.com/koloios/ClassicsRL/actions/workflows/ci.yml/badge.svg)](https://github.com/koloios/ClassicsRL/actions/workflows/ci.yml)

A lightweight toolkit for training, evaluating, and visualizing reinforcement
learning agents on classic games.

Apple Silicon and optional Colab GPU runs use the same Python package.

## Status

**Experimental v0.1.** Currently supports pixel-based Pong with DQN. Additional
games and algorithms are planned.

A prospective comparison trained ClassicsRL DQN and Stable-Baselines3 DQN for
2M agent steps each on the same CUDA runtime, using training seed 2. Both
validation-selected policies were locked before 50 fresh full-game tests:

| Policy | Mean return | Wins |
| --- | ---: | ---: |
| ClassicsRL DQN | +6.44 | 41/50 |
| SB3 DQN | -14.32 | 0/50 |
| Random actions | -20.36 | 0/50 |

All test games ended naturally. This demonstrates a winning ClassicsRL policy
for one seed, not consistent learning or general superiority over SB3. Earlier
ClassicsRL runs scored +8.88 and +0.42 under different seeds/backends; the earlier
1.3M-step comparison favored SB3. See the
[experiments](docs/experiments.md) for raw results, methods, and limitations.

## Run locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). Python 3.11+
is supported; the checkout defaults to 3.12. Tests and the dashboard need no Atari
installation:

```sh
git clone https://github.com/koloios/ClassicsRL.git
cd ClassicsRL
uv sync --locked --extra dashboard --extra video
uv run --locked --extra dashboard --extra video pytest -q -m "not atari"
uv run --locked --extra dashboard --extra video classicsrl dashboard --runs runs
```

Open `http://127.0.0.1:8765`. The workspace is empty until you train or import a run.

## Train Pong

Review [game content and dependencies](docs/game-content.md) before installing
Atari support. Begin with the short pipeline check:

```sh
uv sync --locked --all-extras
uv run --locked --extra atari classicsrl train --config configs/smoke.toml --run runs/smoke
```

This runs 256 steps and checks gradients and persistence. The longer experimental
recipe used for the results above is:

```sh
uv run --extra atari classicsrl train --config configs/pong-reference-inspired.toml \
  --seed 0 --steps 2000000 --run runs/pong-seed-0
```

`--device cpu`, `mps`, or `cuda` selects a backend. The default `auto` prefers CUDA,
then MPS, then CPU. All schedules count agent transitions; frame repetition is
applied once. Use a new directory for each experiment.

To continue from a complete snapshot, use the paths printed by training:

```sh
uv run --extra atari classicsrl train \
  --resume runs/smoke/snapshots/<snapshot> --steps 512
```

`--steps` is the total target. Recovery restores learning state and starts a fresh
emulator episode. Load only trusted artifacts. See [architecture](docs/architecture.md)
for configuration profiles, artifact contents, and recovery boundaries.

## Record and inspect

```sh
uv run --extra atari --extra video classicsrl evaluate \
  runs/smoke/checkpoints/<checkpoint>.pt --episodes 1 --max-steps 1000 \
  --record-dir runs/smoke/episodes/preview
uv run --extra dashboard classicsrl dashboard --runs runs
```

The dashboard discovers recordings under each run. Playback supports seeking,
decision stepping, synchronized action values, and minimal or original Pong colors.
Appearance changes affect only the viewer. The capped preview above is excluded
from validation statistics; omit `--max-steps` for full-game evaluation.

Training, validation, held-out tests, and recordings are labeled separately. A
positive time-limited lead is not a full-game win. Q-values estimate discounted
return, not action probabilities.

## Colab

The [notebook](notebooks/train_colab.ipynb) installs a source ZIP and calls the same
package. Active training stays on runtime-local storage; verified snapshots can
be copied to Drive. A runtime loss can discard work since the last persisted
snapshot. Free GPU allocation and duration vary.

Use `--persist-dir /path/to/durable/storage` for periodic verified copies. Copy the
complete run back to local storage before resuming in another session.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, checks, and experiment conventions.
CI tests Python 3.11 and 3.12 without game assets. Tests cover DQN targets, visual
learning on a controlled task, replay boundaries, recovery, and dashboard behavior.

- [Architecture and artifact format](docs/architecture.md)
- [Experiments and reproduction](docs/experiments.md)
- [Game content and dependencies](docs/game-content.md)

The next milestones are consistent learning across training seeds and a controlled
Double DQN comparison.

## License

[MIT](LICENSE). Third-party games and assets have separate
terms; no game content, weights, replay buffers, or gameplay recordings are bundled.
