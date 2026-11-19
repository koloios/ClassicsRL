import pytest

from classicsrl.config import Config, load_config


@pytest.mark.parametrize(
    "changes",
    [
        {"gamma": 1.1},
        {"learning_rate": -0.1},
        {"batch_size": True},
        {"double_dqn": 1},
        {"frame_skip": 0},
        {"seed": -1},
        {"device": "tpu"},
        {"epsilon_start": 0.1, "epsilon_end": 0.5},
        {"learning_rate": float("nan")},
    ],
)
def test_invalid_configuration_fails_early(changes):
    with pytest.raises(ValueError):
        Config(**changes)


def test_config_rejects_misspelled_keys(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("batch_szie = 16\n")
    with pytest.raises(ValueError, match="batch_szie"):
        load_config(path)


def test_default_values_are_resolved_when_loading_partial_toml(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("total_steps = 128\n")
    config = load_config(path)
    assert config.total_steps == 128
    assert config.to_dict()["frame_skip"] == 4


@pytest.mark.parametrize(
    "configured, flag, expected",
    [
        (False, None, False),
        (True, None, True),
        (False, "--double-dqn", True),
        (True, "--no-double-dqn", False),
    ],
)
def test_cli_double_dqn_override_respects_config(tmp_path, monkeypatch, configured, flag, expected):
    from classicsrl import training
    from classicsrl.cli import main

    path = tmp_path / "config.toml"
    path.write_text(f"double_dqn = {str(configured).lower()}\n")
    captured = []
    monkeypatch.setattr(training, "train", lambda config, *a, **kw: captured.append(config))
    main(
        ["train", "--config", str(path), "--run", str(tmp_path / "run")] + ([flag] if flag else [])
    )
    assert captured[0].double_dqn is expected
