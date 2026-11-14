import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from classicsrl.dashboard import create_app


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def workspace(tmp_path):
    run = tmp_path / "runs" / "pong"
    environment = {
        "env_id": "ALE/Pong-v5",
        "wrappers": [{"name": "TimeLimit", "max_episode_steps": 27000}],
    }
    write(
        run / "run.json",
        {
            "run_id": "one",
            "config": {"env_id": "ALE/Pong-v5", "seed": 0, "total_steps": 100},
            "environment": environment,
        },
    )
    (run / "metrics.jsonl").write_text(
        json.dumps({"metric": "train/td_loss", "value": 0.25, "agent_steps": 100})
        + '\n{"metric": "train/td_loss", "value": NaN, "agent_steps": 101}\n{"partial":'
    )
    full = {
        "policy": "greedy",
        "checkpoint": "step-000100.pt",
        "agent_steps": 100,
        "environment": environment,
        "episodes": [{"return": -21, "seed": 1}],
    }
    (run / "evaluations.jsonl").write_text(json.dumps(full) + "\n")
    preview = {
        **full,
        "environment": {"wrappers": [{"name": "TimeLimit", "max_episode_steps": 10}]},
        "episodes": [{"return": 0, "seed": 1, "video": "episode-000.mp4"}],
    }
    folder = run / "episodes" / "preview"
    write(folder / "evaluation.json", preview)
    (folder / "episode-000.mp4").write_bytes(b"0123456789")
    (folder / "episode-000.jsonl").write_text('{"frame_index": 0}\n')
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "step-000100.pt").write_bytes(b"never unpickle me")
    return run, TestClient(create_app(run.parent), base_url="http://127.0.0.1")


def test_dashboard_keeps_preview_scores_out_of_learning_summary(workspace):
    run, client = workspace
    response = client.get("/api/runs/pong")
    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["best_evaluation"] == -21
    assert data["summary"]["step"] == 100
    assert data["summary"]["seed"] == 0
    assert len(data["metrics"]["train/td_loss"]) == 1
    assert len(data["warnings"]) == 2
    assert [entry["compatible_for_learning"] for entry in data["evaluations"]] == [True, False]
    assert data["checkpoints"][0]["record_command"][8] == str(run / "checkpoints/step-000100.pt")
    assert client.get("/api/runs").json()["runs"][0]["id"] == "pong"


def test_media_ranges_and_sidecar_are_available_but_other_files_are_not(workspace):
    run, client = workspace
    base = "/api/runs/pong/"
    response = client.get(
        base + "media/episodes/preview/episode-000.mp4", headers={"Range": "bytes=2-5"}
    )
    assert response.status_code == 206 and response.content == b"2345"
    decisions = client.get(base + "decisions/episodes/preview/episode-000.jsonl")
    assert decisions.json()["decisions"] == [{"frame_index": 0}]
    assert client.get(base + "decisions/run.json").status_code == 404
    outside = run.parent.parent / "secret.mp4"
    outside.write_bytes(b"private")
    (run / "episodes" / "preview" / "outside.mp4").symlink_to(outside)
    assert client.get(base + "media/episodes/preview/outside.mp4").status_code == 404
    (run.parent / "external").symlink_to(run.parent.parent)
    assert client.get("/api/runs/external").status_code == 404
    # Absolute video references in artifact metadata must not escape the run or crash it.
    path = run / "episodes" / "preview" / "evaluation.json"
    data = json.loads(path.read_text())
    data["episodes"][0]["video"] = str(outside)
    write(path, data)
    assert client.get(base.rstrip("/")).status_code == 200


def test_foreign_run_evaluations_do_not_inflate_scores(workspace):
    run, client = workspace
    entry = json.loads((run / "evaluations.jsonl").read_text())
    entry.update({"run_id": "another-run", "episodes": [{"return": 21}]})
    with (run / "evaluations.jsonl").open("a") as file:
        file.write(json.dumps(entry) + "\n")
    data = client.get("/api/runs/pong").json()
    assert data["summary"]["best_evaluation"] == -21
    assert not data["evaluations"][1]["compatible_for_learning"]


def test_interruption_at_target_is_not_reported_as_complete(workspace):
    run, client = workspace
    (run / "segments.jsonl").write_text(
        json.dumps({"event": "stop", "agent_steps": 100, "interrupted": True}) + "\n"
    )
    summary = client.get("/api/runs/pong").json()["summary"]
    assert summary["step"] == 100
    assert summary["status"] == "paused"


def test_pong_win_rate_excludes_truncated_leads_and_labels_final_tests(workspace):
    run, client = workspace
    entry = json.loads((run / "evaluations.jsonl").read_text())
    entry.update(
        evaluation_role="heldout",
        episodes=[
            {"return": 21, "terminated": True, "truncated": False},
            {"return": 5, "terminated": False, "truncated": True},
            {"return": -1, "terminated": True, "truncated": False},
        ],
    )
    write(run / "episodes" / "final-test" / "evaluation.json", entry)
    data = client.get("/api/runs/pong").json()
    final = next(row for row in data["evaluations"] if row["evaluation_role"] == "heldout")
    assert final["full_game_wins"] == 1
    assert final["win_rate"] == pytest.approx(1 / 3)
    assert final["truncations"] == 1
    assert data["summary"]["best_evaluation"] == -21
    assert data["evaluations"][0]["win_rate"] is None  # Legacy outcomes are unknown.


def test_empty_workspace_static_assets_and_local_host_restriction(tmp_path):
    client = TestClient(create_app(tmp_path / "missing"), base_url="http://localhost")
    assert client.get("/api/runs").json() == {"runs": [], "warnings": []}
    assert "Experiment workspace" in client.get("/").text
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/api/runs/missing").status_code == 404
    assert client.get("/", headers={"host": "untrusted.example"}).status_code == 400
    assert "script-src 'self'" in client.get("/").headers["content-security-policy"]
