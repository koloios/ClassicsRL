import copy
import json
import re
import subprocess
from pathlib import Path

import pytest

from classicsrl import artifacts


def test_public_report_preserves_results_without_disclosing_machine_or_storage():
    original = {
        "runtime": {"platform": "private-os", "hardware": "private-machine", "revision": "old"},
        "source_run": "/content/drive/MyDrive/ClassicsRL/run-id",
        "checkpoint": "/content/drive/MyDrive/ClassicsRL/run-id/checkpoints/policy.pt",
        "episodes": [{"return": 12.0, "length": 100}],
        "device": "mps",
        "source_sha256": {"dqn.py": "digest"},
    }
    saved = copy.deepcopy(original)
    result = artifacts.public_report(original)
    assert result == {
        "runtime": {},
        "source_run": "runs/run-id",
        "checkpoint": "runs/run-id/checkpoints/policy.pt",
        "episodes": [{"return": 12.0, "length": 100}],
        "device": "mps",
        "source_sha256": {"dqn.py": "digest"},
    }
    assert artifacts.public_report(result) == result
    assert original == saved


def test_committed_reports_and_notebook_have_no_private_runtime_metadata():
    root = Path(__file__).parents[1]
    for path in (root / "docs" / "experiments").glob("*.json"):
        report = json.loads(path.read_text())
        assert artifacts.public_report(report) == report, path.name
    notebook = json.loads((root / "notebooks" / "train_colab.ipynb").read_text())
    assert not re.search(
        r"colab\.research\.google\.com/drive/|drive\.google\.com/(?:file|drive)/"
        r"|[\w.+-]+@(?:gmail|googlemail)\.com|ya29\.|1//",
        json.dumps(notebook),
    )
    assert set(notebook["metadata"]) <= {"kernelspec", "language_info"}
    for cell in notebook["cells"]:
        assert not cell.get("outputs")
        assert cell.get("execution_count") is None
        assert not cell.get("metadata")


def commit(root):
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


@pytest.mark.parametrize("tracked", [True, False])
def test_provenance_uses_tracked_source_not_launch_directory(tmp_path, monkeypatch, tracked):
    source = tmp_path / "source"
    package = source / "package"
    package.mkdir(parents=True)
    module = package / "artifacts.py"
    module.write_text("SOURCE = True\n")
    (source / ".gitignore").write_text("" if tracked else "package/\n")
    revision = commit(source)

    caller = tmp_path / "caller"
    caller.mkdir()
    (caller / "unrelated.txt").write_text("another project\n")
    commit(caller)
    monkeypatch.chdir(caller)
    monkeypatch.setattr(artifacts, "__file__", str(module))

    metadata = artifacts.runtime_metadata("cpu")
    assert metadata["revision"] == (revision if tracked else None)
    assert metadata["dirty"] is (False if tracked else None)
    assert metadata["source_sha256"] == {"artifacts.py": artifacts.digest(module)}

    if tracked:
        module.write_text("SOURCE = False\n")
        assert artifacts.runtime_metadata("cpu")["dirty"] is True
