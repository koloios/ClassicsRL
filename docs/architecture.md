# Architecture

ClassicsRL uses plain functions, a single training loop, and portable run folders.
Learning code does not depend on the dashboard.

| Module | Responsibility |
| --- | --- |
| `config.py`, `envs.py` | Resolved settings and explicit ALE preprocessing |
| `dqn.py` | CNN, epsilon-greedy actions, TD targets, and an optimizer update |
| `replay.py` | Uniform sampling with reconstructed frame stacks |
| `training.py` | Interaction, schedules, validation, and persistence |
| `evaluation.py` | Validation selection, independent games, and recordings |
| `artifacts.py` | Checkpoints, verified snapshots, and source provenance |
| `diagnostics.py` | Feature activity, gradients, and fixed-batch device checks |
| `dashboard.py`, `web/` | Read-only local API and buildless browser UI |

## Learning protocol

The CNN receives four uint8 84×84 grayscale frames and normalizes them internally.
DQN uses uniform replay, one-step targets, Smooth L1 loss, Adam, gradient clipping,
and hard target updates. True terminations remove bootstrapping; time-limit
truncations use the actual final observation.

ALE frame skip is one. The preprocessing wrapper applies action repetition and
max-pooling; defaults use four repeated frames, sticky-action probability 0.25,
startup no-ops, FIRE on reset, and a 27,000-agent-step episode cap. The ordered
wrapper protocol is saved with every run and evaluation.

The replay arena stores `capacity + stack_size` frames with absolute IDs. Sampling
reconstructs stacks without crossing resets or ring generations. Reset frames
also occupy the arena, so fewer than `capacity` transitions may be retained.

## Configuration

TOML files override the validated `Config` defaults. CLI overrides are recorded
in the resolved settings. Step schedules count agent transitions.

| Profile | Purpose |
| --- | --- |
| `smoke.toml` | 256-step CPU pipeline check |
| `benchmark.toml` | Short throughput check with gradients active |
| `pong.toml` | Conservative starting recipe with the lower-rate mitigation |
| `pong-reference-inspired.toml` | Longer experimental recipe used in the 2M trials |

The two Pong recipes differ in several learning settings. The lower rate preserved
features in the early study; the longer recipe produced one strong policy. Neither
is established as consistently competitive. See [experiments](experiments.md).

## Run format, version 1

| Artifact | Contents |
| --- | --- |
| `run.json` | Run identity, settings, environment protocol, versions, source hashes |
| `segments.jsonl` | Starts, stops, recovery boundaries, device and source metadata |
| `metrics.jsonl`, `evaluations.jsonl` | Raw training metrics and repeated validation |
| `checkpoints/*.pt` | Small CPU-portable playback weights and policy metadata |
| `snapshots/*/` | Online/target weights, Adam, replay, counters, RNGs, checksum manifest |
| `episodes/*/` | Evaluation metadata, RGB video, and per-decision JSONL |

Snapshot directories are promoted only after all files and checksums are written.
Persistent copies publish the manifest last, after verification. Incomplete
snapshots are ignored. Segment suffixes keep policies from separate continuations
distinct. Playback checkpoints alone cannot resume training.

Resume restores learning state, then starts a new emulator episode and frame stack.
It preserves the run ID, allows device/thread/target-budget changes, and rejects
learning-setting changes. Exact emulator continuation and cross-device equality
are not promised. Git provenance comes from tracked package source, independent
of the launch directory; source hashes are recorded even without a checkout.

## Visualization

The FastAPI server binds to loopback, reads artifacts without loading Torch models,
and restricts media to each run's episode directory. Native video, canvas, and SVG
need no JavaScript build chain or database.

Only matching validation protocols contribute to learning curves and best-return
summaries. Videos contain one frame before each decision and a final frame;
sidecars align actions, Q-values, rewards, and termination flags. Seeking inspects
recorded decisions. Minimal styling changes neither training observations nor saved
video. Complete run folders can be copied between Colab and the local workspace.
