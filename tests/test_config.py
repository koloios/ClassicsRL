import pytest

from classicsrl.config import Config, load_config


@pytest.mark.parametrize(
    "changes",
    [
        {"gamma": 1.1},
        {"learning_rate": -0.1},
        {"batch_size": True},
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
