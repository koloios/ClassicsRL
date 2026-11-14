"""Read-only local API over portable experiment directories. No Torch dependency."""

import json
import math
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware


def _json(path, warnings):
    try:
        value = json.loads(path.read_text(), parse_constant=_invalid_constant)
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        return value
    except (OSError, ValueError):
        warnings.append(f"Could not read {path.name}.")
        return {}


def _lines(path, warnings):
    if not path.is_file():
        return []
    result = []
    try:
        with path.open() as file:
            for number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line, parse_constant=_invalid_constant)
                    if isinstance(value, dict):
                        result.append(value)
                    else:
                        raise ValueError("expected an object")
                except ValueError:
                    warnings.append(f"Skipped incomplete or invalid {path.name} line {number}.")
    except OSError:
        warnings.append(f"Could not read {path.name}.")
    return result


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def _object(value):
    return value if isinstance(value, dict) else {}


def _array(value):
    return value if isinstance(value, list) else []


def _artifact(run, relative, suffix):
    path = (run / relative).resolve()
    episodes = (run / "episodes").resolve()
    # Do not follow either the episodes root or a media file outside the run.
    if run not in episodes.parents or episodes not in path.parents or path.suffix != suffix:
        raise HTTPException(404, "Artifact not found")
    if not path.is_file():
        raise HTTPException(404, "Artifact not found")
    return path


def _evaluations(run, meta, warnings):
    entries = [(row, None) for row in _lines(run / "evaluations.jsonl", warnings)]
    for path in sorted((run / "episodes").glob("*/evaluation.json")):
        if run not in path.resolve().parents:
            continue
        entries.append((_json(path, warnings), path.parent))
    results = []
    for number, (entry, folder) in enumerate(entries):
        episodes = []
        for episode in _array(entry.get("episodes")):
            if not isinstance(episode, dict) or not _finite(episode.get("return")):
                continue
            item = {key: episode.get(key) for key in ("seed", "length", "terminated", "truncated")}
            item["return"] = float(episode["return"])
            item["video_url"], item["decisions_url"] = None, None
            if folder is not None and isinstance(episode.get("video"), str):
                try:
                    relative = (folder / episode["video"]).relative_to(run).as_posix()
                    _artifact(run, relative, ".mp4")
                    sidecar = str(Path(relative).with_suffix(".jsonl"))
                    _artifact(run, sidecar, ".jsonl")
                    base = f"/api/runs/{quote(run.name, safe='')}"
                    item["video_url"] = f"{base}/media/{quote(relative)}"
                    item["decisions_url"] = f"{base}/decisions/{quote(sidecar)}"
                except (HTTPException, ValueError):
                    warnings.append(f"Missing video or decisions in {folder.name}.")
            episodes.append(item)
        if not episodes:
            continue
        environment = _object(entry.get("environment"))
        wrappers = [_object(wrapper) for wrapper in _array(environment.get("wrappers"))]
        cap = next(
            (
                wrapper.get("max_episode_steps")
                for wrapper in wrappers
                if wrapper.get("name") == "TimeLimit"
            ),
            None,
        )
        frame_skip = next(
            (
                wrapper.get("frame_skip")
                for wrapper in wrappers
                if wrapper.get("name") == "AtariPreprocessing"
            ),
            4,
        )
        checkpoint = Path(str(entry.get("checkpoint", ""))).name
        step_match = re.search(r"step-(\d+)", checkpoint)
        known_outcomes = environment.get("env_id") == "ALE/Pong-v5" and all(
            type(episode.get("terminated")) is bool and type(episode.get("truncated")) is bool
            for episode in episodes
        )
        wins = (
            sum(episode["terminated"] and episode["return"] > 0 for episode in episodes)
            if known_outcomes
            else None
        )
        results.append(
            {
                "id": str(entry.get("evaluation_id", f"evaluation-{number}")),
                "checkpoint": checkpoint,
                "agent_steps": entry.get("agent_steps", int(step_match[1]) if step_match else None),
                "policy": entry.get("policy", "unknown"),
                "evaluation_role": entry.get(
                    "evaluation_role", "validation" if folder is None else "recording"
                ),
                "action_names": entry.get("action_names", []),
                "video_fps": 60 / frame_skip if _finite(frame_skip) and frame_skip > 0 else 15,
                "episodes": episodes,
                "mean_return": sum(episode["return"] for episode in episodes) / len(episodes),
                "full_game_wins": wins,
                "win_rate": wins / len(episodes) if wins is not None else None,
                "truncations": (
                    sum(episode["truncated"] for episode in episodes) if known_outcomes else None
                ),
                "compatible_for_learning": bool(meta.get("environment"))
                and environment == meta["environment"]
                and entry.get("run_id", meta.get("run_id")) == meta.get("run_id"),
                "episode_cap": cap,
                "has_video": any(episode["video_url"] for episode in episodes),
                "folder": folder.name if folder else None,
            }
        )
    return results


def read_run(run):
    warnings = []
    meta = _json(run / "run.json", warnings)
    if not meta:
        raise HTTPException(422, "Run metadata cannot be read")
    rows = _lines(run / "metrics.jsonl", warnings)
    metrics, valid_rows = {}, []
    for row in rows:
        if (
            not isinstance(row.get("metric"), str)
            or not _finite(row.get("value"))
            or not _finite(row.get("agent_steps"))
        ):
            warnings.append("Skipped an invalid metric record.")
            continue
        valid_rows.append(row)
        metrics.setdefault(row["metric"], []).append(
            {
                "step": row["agent_steps"],
                "value": row["value"],
                "segment": row.get("segment_id", "legacy"),
                "updates": row.get("updates"),
                "seconds": row.get("train_seconds"),
            }
        )
    segments = _lines(run / "segments.jsonl", warnings)
    starts = [segment for segment in segments if segment.get("event") == "start"]
    config = _object(starts[-1].get("config", meta.get("config")) if starts else meta.get("config"))
    last = segments[-1] if segments else {}
    active_id = last.get("segment_id")
    current_rows = [
        row for row in valid_rows if not active_id or row.get("segment_id") == active_id
    ]
    step = current_rows[-1].get("agent_steps", 0) if current_rows else last.get("agent_steps", 0)
    updated = max(
        (
            path.stat().st_mtime
            for path in (run / "metrics.jsonl", run / "segments.jsonl")
            if path.exists()
        ),
        default=(run / "run.json").stat().st_mtime,
    )
    if last.get("event") == "stop":
        step = last.get("agent_steps", step)
        step = step if _finite(step) else 0
        target = config.get("total_steps")
        status = (
            "complete"
            if _finite(target) and step >= target and not last.get("interrupted")
            else "paused"
        )
    else:
        status = (
            "active"
            if last.get("event") == "start" and time.time() - updated < 120
            else "incomplete"
        )
    evaluations = _evaluations(run, meta, warnings)
    checkpoints = []
    for path in (run / "checkpoints").glob("*.pt"):
        if run not in path.resolve().parents:
            continue
        match = re.search(r"step-(\d+)", path.name)
        checkpoints.append(
            {
                "name": path.name,
                "step": int(match[1]) if match else None,
                "modified": path.stat().st_mtime,
                "bytes": path.stat().st_size,
                "record_command": [
                    "uv",
                    "run",
                    "--extra",
                    "atari",
                    "--extra",
                    "video",
                    "classicsrl",
                    "evaluate",
                    str(path),
                    "--episodes",
                    "1",
                    "--max-steps",
                    "1000",
                    "--record-dir",
                    str(run / "episodes" / f"preview-{time.time_ns()}"),
                ],
            }
        )
    checkpoints.sort(key=lambda item: item["modified"])
    compatible = [
        entry["mean_return"]
        for entry in evaluations
        if entry["compatible_for_learning"]
        and entry["policy"] == "greedy"
        and entry["evaluation_role"] == "validation"
    ]
    summary = {
        "id": run.name,
        "run_id": meta.get("run_id"),
        "env_id": config.get("env_id", "Unknown"),
        "seed": config.get("seed"),
        "created_at": meta.get("created_at"),
        "status": status,
        "step": step,
        "target_steps": config.get("total_steps"),
        "device": starts[-1].get("device", meta.get("device")) if starts else meta.get("device"),
        "updates": current_rows[-1].get("updates") if current_rows else None,
        "seconds": current_rows[-1].get("train_seconds") if current_rows else None,
        "best_evaluation": max(compatible) if compatible else None,
        "checkpoint_count": len(checkpoints),
        "recording_count": sum(bool(entry["has_video"]) for entry in evaluations),
        "updated_at": datetime.fromtimestamp(updated, UTC).isoformat(),
    }
    return {
        "summary": summary,
        "metadata": meta,
        "config": config,
        "metrics": metrics,
        "segments": segments,
        "evaluations": evaluations,
        "checkpoints": checkpoints,
        "warnings": list(dict.fromkeys(warnings)),
    }


def create_app(runs_root: Path):
    root = Path(runs_root).expanduser().resolve()
    web = Path(__file__).parent / "web"
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

    @app.middleware("http")
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; "
            "frame-ancestors 'none'; base-uri 'self'"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def selected_run(identifier):
        run = (root / identifier).resolve()
        if run.parent != root or not (run / "run.json").is_file():
            raise HTTPException(404, "Run not found")
        return run

    @app.get("/")
    def index():
        return FileResponse(web / "index.html")

    @app.get("/api/runs")
    def runs():
        result, warnings = [], []
        if root.is_dir():
            for path in root.iterdir():
                if path.resolve().parent != root or not (path / "run.json").is_file():
                    continue
                try:
                    result.append(read_run(path.resolve())["summary"])
                except HTTPException:
                    warnings.append(f"Skipped unreadable run {path.name}.")
        result.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return {"runs": result, "warnings": warnings}

    @app.get("/api/runs/{identifier}")
    def detail(identifier: str):
        return read_run(selected_run(identifier))

    @app.get("/api/runs/{identifier}/decisions/{relative:path}")
    def decisions(identifier: str, relative: str):
        path = _artifact(selected_run(identifier), relative, ".jsonl")
        warnings = []
        return {"decisions": _lines(path, warnings), "warnings": warnings}

    @app.get("/api/runs/{identifier}/media/{relative:path}")
    def media(identifier: str, relative: str):
        return FileResponse(
            _artifact(selected_run(identifier), relative, ".mp4"), media_type="video/mp4"
        )

    app.mount("/assets", StaticFiles(directory=web), name="assets")
    return app
