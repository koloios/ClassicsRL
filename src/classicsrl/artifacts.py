"""Portable experiment artifacts and verified, immutable training snapshots."""

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from classicsrl.config import Config
from classicsrl.envs import protocol
from classicsrl.replay import ReplayBuffer

SCHEMA_VERSION = 1


def public_report(value):
    """Copy a report without machine identifiers, private paths, or original Git refs."""
    private = {
        "platform",
        "hostname",
        "hardware",
        "machine",
        "cpu_model",
        "gpu_model",
        "gpu",
        "gpu_name",
        "gpu_memory_bytes",
        "gpu_memory_gib",
        "memory_gib",
        "ram_gb",
        "ram_gib",
        "revision",
    }
    if isinstance(value, dict):
        return {key: public_report(item) for key, item in value.items() if key not in private}
    if isinstance(value, list):
        return [public_report(item) for item in value]
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        if normalized.startswith("/") or (len(normalized) > 2 and normalized[1:3] == ":/"):
            for marker in ("/ClassicsRL/", "/runs/"):
                if marker in normalized:
                    return "runs/" + normalized.split(marker, 1)[1]
            return "local-artifact" + PurePosixPath(normalized).suffix
    return value


def write_json(path: Path, value):
    """Replace small metadata files only after the new file has been flushed."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def append_json(path: Path, value):
    with path.open("a") as file:
        file.write(json.dumps(value, allow_nan=False) + "\n")
        file.flush()


def digest(path: Path):
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def runtime_metadata(device):
    """Record versions and tracked source provenance independently of the launch directory."""
    source = Path(__file__).resolve().parent
    git = ["git", "-C", str(source)]
    try:
        # Installed packages may sit inside an unrelated checkout.
        subprocess.check_output(
            git + ["ls-files", "--error-unmatch", "--", Path(__file__).name],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        revision = subprocess.check_output(
            git + ["rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, timeout=5
        ).strip()
        dirty = bool(
            subprocess.check_output(
                git + ["status", "--porcelain"], stderr=subprocess.DEVNULL, text=True, timeout=5
            ).strip()
        )
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    versions = {}
    for name in ("classicsrl", "torch", "numpy", "gymnasium", "ale-py", "opencv-python-headless"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "revision": revision,
        "dirty": dirty,
        "source_sha256": {file.name: digest(file) for file in source.glob("*.py")},
        "dependencies": versions,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
    }


def create_run(path: Path, config: Config, device):
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"run directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    for directory in ("checkpoints", "snapshots", "episodes"):
        (path / directory).mkdir(exist_ok=True)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "run_id": uuid.uuid4().hex,
        "created_at": datetime.now(UTC).isoformat(),
        "config": config.to_dict(),
        "environment": protocol(config),
        **runtime_metadata(device),
        "loss": "smooth_l1",
        "network": "atari_cnn_v1",
        "evaluation_policy": "greedy",
        "resume_mode": "fresh_episode",
    }
    write_json(path / "run.json", meta)
    return meta


def save_checkpoint(run: Path, network, config: Config, counters, run_id, segment_id=None):
    suffix = f"-{segment_id[:8]}" if segment_id else ""
    destination = run / "checkpoints" / f"step-{counters['step']:012d}{suffix}.pt"
    if destination.exists():
        return destination
    temporary = destination.with_suffix(".tmp")
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "playback",
            "run_id": run_id,
            "segment_id": segment_id,
            "config": config.to_dict(),
            "num_actions": network.layers[-1].out_features,
            "counters": counters.copy(),
            "online": {key: value.detach().cpu() for key, value in network.state_dict().items()},
        },
        temporary,
    )
    temporary.replace(destination)
    return destination


def load_checkpoint(path: Path, device):
    data = torch.load(path, map_location=device, weights_only=True)
    if data.get("schema_version") != SCHEMA_VERSION or data.get("kind") != "playback":
        raise ValueError("unsupported playback checkpoint")
    return data


def _rng_state(action_rng):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "actions": action_rng.bit_generator.state,
    }


def save_snapshot(run, network, target, optimizer, replay, config, counters, run_id, action_rng):
    name = f"step-{counters['step']:012d}-{uuid.uuid4().hex[:8]}"
    temporary = run / "snapshots" / f".{name}.tmp"
    temporary.mkdir()
    state = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "config": config.to_dict(),
        "counters": counters.copy(),
        "rng": _rng_state(action_rng),
        "device": str(next(network.parameters()).device),
        "resume_mode": "fresh_episode",
    }
    write_json(temporary / "state.json", state)
    torch_state = {
        "online": network.state_dict(),
        "target": target.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cpu_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        torch_state["cuda_rng"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        torch_state["mps_rng"] = torch.mps.get_rng_state()
    torch.save(torch_state, temporary / "training.pt")
    replay.save(temporary / "replay.npz")
    files = {}
    for path in temporary.iterdir():
        with path.open("rb") as file:
            os.fsync(file.fileno())
        files[path.name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    write_json(temporary / "manifest.json", {"schema_version": SCHEMA_VERSION, "files": files})
    destination = run / "snapshots" / name
    temporary.replace(destination)
    return destination


def verify_snapshot(path: Path):
    manifest = json.loads((path / "manifest.json").read_text())
    expected = {"state.json", "training.pt", "replay.npz"}
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or set(manifest.get("files", {})) != expected
    ):
        raise ValueError("unsupported or incomplete snapshot manifest")
    for name, entry in manifest["files"].items():
        file = path / name
        if file.stat().st_size != entry["bytes"] or digest(file) != entry["sha256"]:
            raise ValueError(f"snapshot integrity check failed: {name}")


def restore_snapshot(path, network, target, optimizer, device, action_rng):
    verify_snapshot(path)
    state = json.loads((path / "state.json").read_text())
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported training snapshot")
    # Optimizer.load_state_dict moves parameter state to the model's device and
    # retains CPU step counters where required by non-capturable Adam.
    data = torch.load(path / "training.pt", map_location="cpu", weights_only=True)
    network.load_state_dict(data["online"])
    target.load_state_dict(data["target"])
    optimizer.load_state_dict(data["optimizer"])
    torch.set_rng_state(data["cpu_rng"].cpu())
    rng = state["rng"]
    random.setstate((rng["python"][0], tuple(rng["python"][1]), rng["python"][2]))
    numpy = rng["numpy"]
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:]))
    action_rng.bit_generator.state = rng["actions"]
    if state["device"] == str(device):
        if device.type == "cuda" and "cuda_rng" in data:
            torch.cuda.set_rng_state_all([value.cpu() for value in data["cuda_rng"]])
        if device.type == "mps" and "mps_rng" in data:
            torch.mps.set_rng_state(data["mps_rng"].cpu())
    replay = ReplayBuffer.load(path / "replay.npz")
    return state, replay


def sync_run(source: Path, destination: Path):
    """Copy a run to durable storage; publish snapshot manifests after verification.

    Use a dedicated destination directory, e.g. Drive/ClassicsRL/<run_id>.
    A failed copy leaves no completion marker and can be retried.
    """
    if (
        source.resolve() == destination.resolve()
        or source.resolve() in destination.resolve().parents
    ):
        raise ValueError("persistent storage must be outside the active run directory")
    destination.mkdir(parents=True, exist_ok=True)
    for file in source.iterdir():
        if file.is_file() and file.suffix in {".json", ".jsonl"}:
            _copy_file(file, destination / file.name)
    for folder in ("checkpoints", "episodes"):
        for file in (source / folder).rglob("*"):
            if file.is_file() and not file.name.endswith(".tmp"):
                _copy_file(file, destination / file.relative_to(source))
    for snapshot in (source / "snapshots").iterdir():
        if snapshot.name.startswith("."):
            continue
        output = destination / "snapshots" / snapshot.name
        if (output / "manifest.json").exists():
            verify_snapshot(output)
            continue
        verify_snapshot(snapshot)
        output.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((snapshot / "manifest.json").read_text())
        for name in manifest["files"]:
            _copy_file(snapshot / name, output / name)
            if digest(output / name) != manifest["files"][name]["sha256"]:
                raise ValueError(f"persistent copy failed verification: {name}")
        _copy_file(snapshot / "manifest.json", output / "manifest.json")


def _copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    with temporary.open("rb") as file:
        os.fsync(file.fileno())
    temporary.replace(destination)
