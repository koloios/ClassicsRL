# Contributing

ClassicsRL is an experimental reinforcement learning toolkit for classic games.
Pong with DQN is the first supported workflow. See the
[architecture](docs/architecture.md) and [experiments](docs/experiments.md) before proposing
larger changes.

## Local development

Use Python 3.11 or newer and [uv](https://docs.astral.sh/uv/getting-started/installation/).
The checked-in default is Python 3.12.

```sh
git clone https://github.com/koloios/ClassicsRL.git
cd ClassicsRL
uv sync --locked --extra dashboard --extra video
uv run --locked --extra dashboard --extra video pytest -q -m "not atari"
uv run --locked --extra dashboard --extra video ruff check .
uv run --locked --extra dashboard --extra video ruff format --check .
```

These checks need no ROMs, GPU, network services, or optional Atari installation.
CI runs them on Python 3.11 and 3.12. Tests use deterministic image fixtures;
long training experiments are separate from ordinary tests.

For optional integration tests, first review [game content](docs/game-content.md),
then install the Atari extra and run:

```sh
CLASSICSRL_TEST_ATARI=1 uv run --locked --extra atari --extra dashboard \
  pytest -q tests/test_atari.py
```

## Changes and experiments

Keep pull requests focused. Explain the problem, resulting behavior, and relevant
validation. Add meaningful tests for correctness changes. Prefer small functions,
explicit configuration, and existing dependencies.
Comments should explain a constraint or a non-obvious decision. Avoid narrating
the code, speculative abstractions, and progress notes in source files. Keep current
experiment evidence; use Git history for superseded pilots and planning documents.

For algorithm comparisons, declare the training budget, seeds, checkpoint-selection
rule, and evaluation protocol before testing. Record the source revision, resolved
configuration, dependency versions, and recovery boundaries. Keep validation,
held-out tests, and demonstrations separate. Include raw numerical results and
report weaker runs alongside successful ones.

Bug reports should include the command, expected and observed behavior, Python and
dependency versions, generic backend, and configuration or training seed when relevant.
Keep exact machine specifications, OS details, private paths, account links, and
notebook outputs out of shared reports. Use the collector's `--public` option;
inspect any other artifact before committing it.

Keep credentials, private data, ROMs, game artwork, gameplay recordings, model
weights, and replay buffers out of issues and pull requests. Numeric reports and
plots of learning metrics can be included. See [game content](docs/game-content.md)
for the boundary between this project's license and third-party content.
