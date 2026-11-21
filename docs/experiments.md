# Experiments

These are development results, not a claim of algorithmic novelty. The committed
reports retain raw returns, learning settings, source-file hashes, dependency
versions, and recovery boundaries. Exact machine details, absolute storage paths,
and private development revision links are withheld. Unredacted originals remain private.
Models, replay, and recordings stay outside the repository.

## DQN and Double DQN: training seed 3

Both agents completed 2,000,000 agent steps and 480,001 updates using
`configs/pong-reference-inspired.toml`, training seed 3, and validation seeds
14,000,000–14,000,004. The resolved learning settings differed only in
`double_dqn`. Double DQN separates online next-action selection from target-network
evaluation, following [van Hasselt et al.](https://arxiv.org/abs/1509.06461).

The source and protocol were fixed before training. DQN ran first and completed
without recovery. The Colab runtime was lost during Double DQN; its exact final
counter was not persisted. Double DQN resumed on a new CUDA runtime from a verified
1.1M-step snapshot, restoring replay, online/target weights, Adam, counters, and
RNG state, then starting a fresh emulator episode. Persisted validation rows
through 1.1M were retained; later rows came from the resumed segment. Source,
Python, and dependency versions matched across segments, with no learning-setting
changes. This recovery boundary is a comparison limitation.

Validation used five full games every 50k steps. The selection rule was fixed
before training: highest matching validation mean within 2M, latest checkpoint on
ties. Both selections were locked before 50 greedy final games on seeds
15,000,000–15,000,049. Random actions used the same test seeds.

| Policy | Selected step | Validation mean | Test mean | Wins / 50 | Truncations |
| --- | ---: | ---: | ---: | ---: | ---: |
| DQN | 1,850,000 | +12.0 | +12.42 | 49 | 0 |
| Double DQN | 2,000,000 | +14.6 | +11.76 | 50 | 0 |
| Random actions | — | — | -20.30 | 0 | 0 |

All 150 final games ended naturally. [Raw evidence](experiments/dqn-double-dqn-seed-3.json)
retains the test returns, all 40 validation evaluations per agent, selections,
resolved settings, source hashes, and recovery details. Its exported SHA-256 was
verified before publication.

![Seed-3 DQN and Double DQN validation curves](figures/dqn-double-dqn-seed-3.png)

Both agents met the per-seed target of mean return above +10 and at least 80%
wins. Confirmation across three training seeds remains outstanding. Double DQN
won one additional game but had a lower mean return; this single seed and the
unequal recovery histories do not establish an algorithm advantage. Vanilla DQN
remains the default. The current evidence does not establish overestimation as
the cause of earlier performance variability.

Training/evaluation source was `853395759224177bd55658694bd5fa898f85ae30`.
Both training entry points set `cudnn.deterministic=True` and
`cudnn.benchmark=False`. Recovery preflight passed 19 tests, including CUDA RNG
restoration. DQN wall time was about 116 minutes; the Double DQN recovery segment
took about 59 minutes. Its original process wall time was lost, so no total Double
DQN wall time or throughput comparison is reported.

The completed-run comparison used:

```sh
uv run --extra atari python scripts/compare_prefix.py runs/dqn runs/double-dqn \
  --require-complete --budget 2000000 --episodes 50 --seed 15000000 \
  --device cuda --output runs/seed-3-comparison
```

These test seeds have now been inspected; reserve fresh seeds for future final tests.

## Completed matched pair: training seed 2

ClassicsRL DQN and SB3 2.9.2 each completed 2,000,000 agent steps and 480,001
updates in separate processes on the same Colab CUDA runtime. SB3 ran first.
Both used `configs/pong-reference-inspired.toml`, with training seed 2, the 2M
budget, and validation seeds 12,000,000–12,000,004. Validation ran every 50k steps.
The selection rule was fixed before training: highest validation mean within the
budget, latest checkpoint on ties. Both policies were locked before testing on
seeds 13,000,000–13,000,049.

| Policy | Selected step | Validation mean | Test mean | Wins / 50 | Truncations |
| --- | ---: | ---: | ---: | ---: | ---: |
| ClassicsRL DQN | 1,900,000 | +8.4 | +6.44 | 41 | 0 |
| SB3 DQN | 1,200,000 | -13.4 | -14.32 | 0 | 0 |
| Random actions | — | — | -20.36 | 0 | 0 |

[Raw evidence](experiments/dqn-matched-pair.json) includes all test returns, 40
validation evaluations and 80 policy probes per implementation, source hashes,
and the persisted trial plan. The exported report's SHA-256 was verified before
adding it to this repository.

![Matched seed-2 validation curves](figures/dqn-matched-pair.png)

ClassicsRL won most test games in this pair. One training seed, native warm-up/RNG
and reset differences, different target synchronization order, and shared wrappers
and evaluation prevent a general superiority claim. The earlier common-budget
trial below favored SB3. The target of mean return above +10 with at least 80%
wins across three training seeds remains unmet.

Neither run showed complete Conv3 collapse on the shared 128-observation replay
probe at stored checkpoints. Minimum active fractions were 0.0669 for ClassicsRL
and 0.1739 for SB3. Higher feature activity did not imply stronger play. A fresh
20-update arithmetic audit on this software stack matched SB3 exactly for loss,
Q-values, gradients, weights, and Adam state. This checks fixed-batch arithmetic;
it does not explain the trajectory or playing-strength gap.

Both runs completed without training recovery; browser reconnections did not
restart the processes. Verified 1.9M-step snapshots from Drive were restored after
testing, including replay, optimizer counters, and CUDA RNG state. Replay arrays
occupied 709,428,224 bytes for ClassicsRL and 5,646,800,000 for SB3. The inspected
buffers retained 99,971 and 100,000 transitions respectively: the compact frame
arena also stores reset/history frames. These are array allocations, not peak
process RAM. Wall times were about 124 and 135 minutes, including validation,
persistence, startup, and up to 30 seconds of polling delay; this single trial is
not a throughput benchmark.

Training/evaluation source was `59e5b2f669cc885ec38d5572b02ad67a2e537b3a`.
Before training, CUDA preflight led to a loader fix preserving native Adam scalar
step placement on CPU; the original plan and amendment were retained privately.
Both processes used `cudnn.deterministic=True` and `cudnn.benchmark=False`, with a
ClassicsRL entry-point wrapper matching SB3's native CUDA setup. After testing,
`bf9f436d1012becd94961d3cd357b6f6a565773b` fixed accelerator RNG restoration for
unindexed device aliases and was used for the recovery checks and evidence
collection. This recovery-only change did not alter the completed fresh runs.

The completed-run comparison command locks both selections before either test:

```sh
uv run --extra atari python scripts/compare_prefix.py runs/classicsrl runs/sb3 \
  --require-complete --budget 2000000 --episodes 50 --seed 13000000 \
  --device cuda --output runs/matched-pair-comparison
```

These test seeds have now been inspected; reserve new seeds for future final tests.

## Earlier two-million-step Pong runs

Both ClassicsRL runs used `configs/pong-reference-inspired.toml`, completed
2,000,000 agent steps and 480,001 updates, and selected the highest greedy
validation mean, preferring the latest checkpoint on ties. Validation used five
full games, seeds 1,000,000–1,000,004. Selection was locked before the final tests
on seeds 5,000,000–5,000,049. All final games ended naturally.

| Training seed / device | Selected step | Validation mean | Test mean | Wins / 50 |
| --- | ---: | ---: | ---: | ---: |
| 0 / CUDA | 2,000,000 | +9.6 | +8.88 | 46 |
| 1 / MPS | 1,900,000 | +2.0 | +0.42 | 22 |
| Random actions | — | — | -20.28 | 0 |

The CUDA policy is a working winning agent. The MPS seed did not reproduce its
strength. Seed, backend, and recovery history differ, so the gap cannot be
attributed to hardware. The proposed target—mean return above +10 and at least
80% wins per seed, confirmed in three training seeds—is unmet.

CUDA recovered at 1M and 1.2M steps; MPS also recovered from a verified 1.7M
snapshot after an execution interruption. Recovery starts a fresh emulator
episode. Each segment's actual settings and source are retained in the reports:

- [CUDA seed 0](experiments/dqn-long-cuda.json)
- [MPS seed 1](experiments/dqn-long-mps.json)

![Individual validation curves](figures/dqn-long-seeds.png)

These are five-game validation means, not the final test means. Dashed segments
follow recovery. Separate demonstrations do not contribute to the test results.

Post-run inspection found no all-zero Conv3 collapse on either run's saved replay
probe: the minimum active fractions across stored policies were 0.082 and 0.092.
The probes differ between runs, so their activation magnitudes are not a controlled
backend comparison. The MPS validation score fell from -2.4 at 1.65M to -9.0 at
1.7M before that recovery, and later from +2.0 at 1.9M to -5.6 at 2M without another
recovery. These observations do not isolate a cause; seed-matched trials with
controlled recovery are needed before attributing the gap to the backend or a
specific hyperparameter.

## Common-budget SB3 control

The optional reference runner uses Stable-Baselines3 2.9.2's native collector,
replay, and updates with matched wrappers, architecture, initial weights, optimizer
settings, and exploration schedule. Remaining differences include warm-up RNG,
reset behavior, and target synchronization before a boundary update rather than
after it. The evaluator and wrappers are shared, so this is not an independent
validation of the environment.

The reference runtime ended with policies and validations persisted through 1.3M
steps. The runner used then saved playback policies, not replay/optimizer state;
the original 2M control could not be completed from those files. A post-hoc common
budget of 1.3M was chosen before new tests. Both selections were locked using only
validation within that prefix, then tested on seeds 7,000,000–7,000,049.

| Policy | Selected step | Test mean | Wins / 50 | Truncations |
| --- | ---: | ---: | ---: | ---: |
| ClassicsRL | 1,300,000 | -5.50 | 6 | 0 |
| SB3 | 1,200,000 | -1.94 | 22 | 0 |
| Random | — | -20.50 | 0 | 0 |

SB3 performed better in this trial. The later 2M ClassicsRL winner is a separate
result; one training seed and different recovery histories do not establish
implementation superiority. No SB3 training was restarted for this analysis.

- [Selections and raw tests](experiments/dqn-prefix-comparison-1300k.json)
- [All 26 validation evaluations per source](experiments/dqn-prefix-validation-1300k.json)

![Validation through the common available budget](figures/dqn-prefix-comparison.png)

## Correctness and early stability

The [SB3 arithmetic audit](experiments/dqn-reference-audit.json) matched 20
fixed-batch updates exactly for Q-values, loss, gradients, weights, and Adam state.
It used fresh Adam states, real replay images, explicit termination/truncation
masks, and manually synchronized targets. This checks update arithmetic, not full
training trajectories or general playing strength.

An earlier controlled study found all-zero Conv3 features on a fixed replay probe
at learning rate 1e-4. Changing only the rate to 3e-5 preserved features in three
100k-step seeds:

| Seed | Conv3 active fraction, 1e-4 | Conv3 active fraction, 3e-5 | Validation mean, 3e-5 |
| --- | ---: | ---: | ---: |
| 0 | 0 | 0.604 | -21.0 |
| 1 | 0 | 0.574 | -20.8 |
| 2 | 0 | 0.625 | -20.8 |

A 250k continuation retained active features but scored -21 on five held-out
games, versus -20.6 for random actions. Feature activity does not establish useful
play. The lower rate remains a starting mitigation, not a general recommendation.
The [stability report](experiments/dqn-stability.json) includes the raw data,
shared replay-probe identity, source versions, and CPU/MPS diagnostic comparisons.

![Feature activity and validation returns](figures/dqn-stability.png)

## Reproduction

Start each trial in a new run directory. Declare training seeds, budget, selection
rule, and test seeds before running the experiment. The test seeds above have
already been inspected; reserve new seeds for future final tests.

```sh
uv run --extra atari classicsrl train --config configs/pong-reference-inspired.toml \
  --seed 0 --steps 2000000 --device cuda --run runs/pong-seed-0
uv run --extra atari python scripts/evaluate_run.py runs/pong-seed-0 \
  --device cuda --episodes 50 --seed 9000000 --label new-test
```

`evaluate_run.py` requires the declared budget to finish and locks the selection
before testing. It saves each finished game atomically. After an interruption,
rerun the same command with the same label, seeds, and evaluation runtime. The
evaluator checks the locked checkpoint hash and continues with the next unfinished
game. On Colab, point the command at the completed run copied to Drive so the
selection and game results survive a runtime loss.

`compare_prefix.py` locks selections within a stated common budget without marking
interrupted source runs complete. Existing selections must not be overwritten
after scores are inspected.

The optional reference command is:

```sh
uv run --extra atari --with stable-baselines3==2.9.2 \
  python scripts/train_sb3_reference.py --config configs/pong-reference-inspired.toml \
  --run runs/sb3-seed-0 --seed 0 --steps 2000000 --device cuda
```

The current runner also saves verified native SB3 recovery snapshots under
`reference-snapshots/`. They include both networks, Adam state, replay, counters,
and Python/NumPy/Torch/action-space RNG states. Snapshots are taken after a complete
rollout and update; an interruption during an update falls back to the previous
snapshot. They cannot recover the original control's missing state.

To resume, copy the complete persisted run to runtime-local storage, then use:

```sh
uv run --extra atari --with stable-baselines3==2.9.2 \
  python scripts/train_sb3_reference.py \
  --resume runs/sb3-seed-0/reference-snapshots/<snapshot> --device cuda
```

The original training budget and learning settings stay fixed on resume so the
exploration schedule does not change. Recovery starts a fresh emulator episode;
it does not reproduce an uninterrupted trajectory. Load only trusted snapshots:
SB3 model and replay loading uses pickle. `--persist-dir` copies verified snapshots
to a dedicated run-ID directory, excluding unfinished temporary copies. Replay is
gzip-compressed, but SB3's stacked replay still needs substantial RAM and storage;
measure snapshot/copy time before starting a long session. Snapshot, evaluation,
and persistence time is excluded from reported training time.

Use `classicsrl diagnose --help` for fixed-replay feature and device checks, and
`scripts/collect_stability.py --public` to collect shareable reports from complete
local runs against a shared probe.
`scripts/compare_dqn_reference.py` reproduces the fixed-batch arithmetic audit.

Plots need only committed numeric reports and optional Matplotlib:

```sh
uv run --no-project --with matplotlib python scripts/plot_learning.py \
  docs/experiments/dqn-long-cuda.json docs/experiments/dqn-long-mps.json \
  --output docs/figures/dqn-long-seeds.png
uv run --no-project --with matplotlib python scripts/plot_stability.py \
  docs/experiments/dqn-stability.json --output docs/figures/dqn-stability.png
```

For local throughput checks, use `configs/benchmark.toml` in separate CPU/MPS/CUDA
runs and measure after learning starts. Warm-up collection is not learning
throughput; short checks do not predict sustained speed or time to useful play.
Exact training trajectories across versions and devices are not promised.

Unredacted development records are kept privately. Shared reports omit machine
identifiers; their numerical results are unchanged.
