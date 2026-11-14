# Game content and dependencies

ClassicsRL's code is available under the [MIT license](../LICENSE). That license
does not grant rights to third-party games, ROMs, artwork, or trademarks.

The repository and its own distributions contain the learning code, configuration,
numeric experiment reports, and metric plots. They do not include ROMs, game
artwork, playback policies, replay buffers, or gameplay recordings. Generated runs
stay in the ignored `runs/` directory.

## Running Pong

Pong uses the Atari 2600 environment `ALE/Pong-v5` through the external
[Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment).
Its visuals can differ from the arcade version. The optional `atari` extra installs
`ale-py` and OpenCV; the ALE project states that its Python package includes Atari
ROMs. Those files come from the external dependency, not from ClassicsRL's source
or wheel. See the upstream [README](https://github.com/Farama-Foundation/Arcade-Learning-Environment#features)
and [installation instructions](https://ale.farama.org/getting-started/).

Review the applicable upstream terms and game-content permissions before installing
or using that extra, or sharing recordings. Installing an emulator or receiving a
ROM in a dependency does not establish permission to redistribute game content.
ClassicsRL does not provide an independent clearance of game-content rights.

After reviewing those requirements, the supported setup is:

```sh
uv sync --locked --extra atari --extra video --extra dashboard
uv run --locked --extra atari classicsrl train \
  --config configs/smoke.toml --run runs/smoke
```

The short smoke run checks the pipeline; it is not a trained Pong agent.

## Working without game content

The core package, ordinary tests, diagnostics on your own trusted artifacts, and
dashboard can run without the Atari extra. Start with the repository's
[quick start](../README.md#run-locally). The dashboard opens with an empty workspace
until you create or import a run.

Keep game content and recordings local. Do not attach ROMs or gameplay media to
issues, commits, release assets, or pull requests. The project's published evidence
uses numerical results and plots of learning metrics.
