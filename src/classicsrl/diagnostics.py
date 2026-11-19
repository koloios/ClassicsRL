"""Inspect feature activity and fixed-batch updates without running a game."""

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from classicsrl.artifacts import digest, load_checkpoint, runtime_metadata
from classicsrl.config import Config
from classicsrl.dqn import QNetwork, resolve_device, update
from classicsrl.replay import ReplayBuffer


@torch.no_grad()
def activation_metrics(network, observations, device):
    """Measure the actual CNN on a batch; never rescale an already normalized input."""
    if observations.dtype != np.uint8 or observations.shape[1:] != (4, 84, 84):
        raise ValueError("diagnostics require a batch of four uint8 84x84 frames")
    values = torch.as_tensor(observations, device=device).float() / 255
    metrics = {}
    names = iter(("conv1", "conv2", "conv3", "hidden"))
    for layer in network.layers:
        values = layer(values)
        if isinstance(layer, nn.ReLU):
            name = next(names)
            metrics[f"features/{name}_active_fraction"] = (values > 0).float().mean().item()
            metrics[f"features/{name}_zero_fraction"] = (
                (values.flatten(1) == 0).all(1).float().mean().item()
            )
    metrics["q/observation_std"] = values.std(dim=0, correction=0).mean().item()
    metrics["q/max_observation_range"] = (
        (values.max(dim=0).values - values.min(dim=0).values).max().item()
    )
    metrics["q/greedy_action_count"] = float(values.argmax(dim=1).unique().numel())
    return metrics


def diagnose(checkpoint: Path, replay_path: Path, *, samples=128, seed=0, devices=("cpu",)):
    """Compare devices on identical data/weights with a fresh Adam and copied target.

    This is an arithmetic check, not a continuation of the saved optimizer or a
    test of long-run cross-device reproducibility. Saved artifacts remain unchanged.
    """
    if samples <= 0 or seed < 0:
        raise ValueError("samples must be positive and seed must be nonnegative")
    selected = list(dict.fromkeys(str(resolve_device(device)) for device in devices))
    if not selected:
        raise ValueError("at least one device is required")
    saved = load_checkpoint(Path(checkpoint), "cpu")
    config = Config(**saved["config"])
    torch.set_num_threads(config.torch_threads)
    replay = ReplayBuffer.load(Path(replay_path))
    batch = replay.sample(samples, rng=np.random.default_rng(seed))
    reports, results = {}, {}
    for name in selected:
        device = torch.device(name)
        network = QNetwork(saved["num_actions"]).to(device)
        network.load_state_dict(saved["online"])
        target = copy.deepcopy(network).requires_grad_(False)
        before = {key: value.detach().cpu().clone() for key, value in network.named_parameters()}
        metrics = activation_metrics(network, batch.observations, device)
        with torch.no_grad():
            q_values = network(torch.as_tensor(batch.observations, device=device)).cpu()
        optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
        losses = update(
            network,
            target,
            optimizer,
            batch,
            device,
            config.gamma,
            config.max_grad_norm,
            double_dqn=config.double_dqn,
        )
        gradients = {key: value.grad.cpu().clone() for key, value in network.named_parameters()}
        weights = {key: value.detach().cpu().clone() for key, value in network.named_parameters()}
        reports[name] = {
            "activations": metrics,
            "update": losses,
            "gradient_norms": {key: value.norm().item() for key, value in gradients.items()},
            "parameter_change_norms": {
                key: (value - before[key]).norm().item() for key, value in weights.items()
            },
        }
        results[name] = {"q": q_values, "gradients": gradients, "weights": weights}
    comparisons = {}
    reference = selected[0]
    for name in selected[1:]:
        first, second = results[reference], results[name]
        comparisons[f"{reference}_vs_{name}"] = {
            "q_max_abs_error": (first["q"] - second["q"]).abs().max().item(),
            "td_loss_abs_error": abs(
                reports[reference]["update"]["td_loss"] - reports[name]["update"]["td_loss"]
            ),
            **{
                f"{field}_max_abs_error": max(
                    (value - second[field][key]).abs().max().item()
                    for key, value in first[field].items()
                )
                for field in ("gradients", "weights")
            },
        }
    return {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(Path(checkpoint)),
        "replay": str(replay_path),
        "replay_sha256": digest(Path(replay_path)),
        "sample_seed": seed,
        "samples": samples,
        "optimizer": "fresh Adam; copied online weights used as target",
        "config": config.to_dict(),
        "runtime": runtime_metadata(selected[0]),
        "data": {
            "replay_transitions": len(replay),
            "pixel_min": int(batch.observations.min()),
            "pixel_max": int(batch.observations.max()),
            "distinct_observations": len(np.unique(batch.observations, axis=0)),
            "rewards": {
                str(value): int(count)
                for value, count in zip(*np.unique(batch.rewards, return_counts=True), strict=True)
            },
        },
        "devices": reports,
        "comparisons": comparisons,
    }
