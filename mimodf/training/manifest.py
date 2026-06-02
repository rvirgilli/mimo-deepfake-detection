"""Training manifest for future controlled runs."""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mimodf.config import ExperimentConfig


@dataclass(frozen=True)
class GitState:
    commit: str
    branch: str
    dirty: bool


@dataclass
class TrainingManifest:
    run_id: str
    status: str
    config: dict[str, Any]
    git: GitState
    command: list[str]
    python_version: str
    platform: str
    hostname: str
    working_dir: str
    started_at: str
    ended_at: str | None = None
    duration_seconds: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def start(
        cls,
        config: ExperimentConfig,
        *,
        command: list[str] | None = None,
        working_dir: str | Path | None = None,
    ) -> TrainingManifest:
        return cls(
            run_id=f"{_timestamp_compact()}_{uuid.uuid4().hex[:8]}",
            status="running",
            config=config.to_dict(),
            git=current_git_state(Path(working_dir or Path.cwd())),
            command=list(sys.argv if command is None else command),
            python_version=sys.version,
            platform=platform.platform(),
            hostname=socket.gethostname(),
            working_dir=str(Path(working_dir or Path.cwd())),
            started_at=_now_iso(),
        )

    def complete(
        self,
        *,
        metrics: dict[str, Any],
        artifacts: dict[str, str] | None = None,
        status: str = "completed",
        root: str | Path = ".",
    ) -> None:
        if status not in {"completed", "failed", "interrupted"}:
            raise ValueError(f"invalid manifest completion status: {status}")
        self.status = status
        self.ended_at = _now_iso()
        self.duration_seconds = _duration_seconds(self.started_at, self.ended_at)
        self.metrics = dict(metrics)
        if artifacts is not None:
            self.artifacts = dict(artifacts)
            self.artifact_hashes = hash_existing_artifacts(self.artifacts, root=root)

    def fail(self, error: BaseException | str, *, metrics: dict[str, Any] | None = None) -> None:
        self.error = str(error)
        self.complete(metrics=metrics or {}, status="failed")

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return out

    @classmethod
    def load(cls, path: str | Path) -> TrainingManifest:
        data = json.loads(Path(path).read_text())
        data["git"] = GitState(**data["git"])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def current_git_state(root: str | Path = ".") -> GitState:
    root_path = Path(root)
    return GitState(
        commit=_git(root_path, "rev-parse", "HEAD") or "unknown",
        branch=_git(root_path, "rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        dirty=bool(_git(root_path, "status", "--porcelain")),
    )


def hash_existing_artifacts(artifacts: dict[str, str], *, root: str | Path = ".") -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, value in artifacts.items():
        path = Path(value)
        if not path.is_absolute():
            path = Path(root) / path
        if path.is_file():
            hashes[name] = sha256_file(path)
    return hashes


def sha256_file(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _timestamp_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _duration_seconds(started_at: str, ended_at: str) -> float:
    start = datetime.fromisoformat(started_at)
    end = datetime.fromisoformat(ended_at)
    return (end - start).total_seconds()
