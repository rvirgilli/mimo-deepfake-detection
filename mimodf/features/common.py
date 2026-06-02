"""Shared helpers for cached feature extraction manifests."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FEATURE_MANIFEST_SCHEMA = "mimodf-feature-manifest/v1"
FEATURE_RECORD_SCHEMA = "mimodf-feature-record/v1"


@dataclass(frozen=True)
class FeatureExtractionResult:
    manifest_path: Path
    records_path: Path
    records: int
    output_dir: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest": str(self.manifest_path),
            "records_path": str(self.records_path),
            "records": self.records,
            "output_dir": str(self.output_dir),
        }


def load_audio_protocol(path: str | Path, *, max_items: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            audio_path = record.get("audio_path")
            if not audio_path:
                continue
            if not Path(str(audio_path)).is_file():
                continue
            records.append(record)
            if max_items is not None and len(records) >= max_items:
                break
    return records


def safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def batched(records: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def command_argv() -> list[str]:
    return list(sys.argv)
