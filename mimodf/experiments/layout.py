"""Run-layout helpers for immutable experiment directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunLayout:
    root: Path
    experiment_id: str
    spec_hash: str
    seed: int

    @property
    def spec_hash_dir(self) -> str:
        return self.spec_hash.removeprefix("sha256:")

    @property
    def run_dir(self) -> Path:
        return self.root / self.experiment_id / self.spec_hash_dir / f"seed_{self.seed}"

    @property
    def resolved_spec_path(self) -> Path:
        return self.run_dir / "resolved_spec.yaml"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def metrics_dir(self) -> Path:
        return self.run_dir / "metrics"

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def eval_dir(self) -> Path:
        return self.run_dir / "eval"

    @property
    def reports_dir(self) -> Path:
        return self.run_dir / "reports"

    def create(self, *, overwrite: bool = False) -> RunLayout:
        if self.run_dir.exists() and any(self.run_dir.iterdir()) and not overwrite:
            raise FileExistsError(f"run directory already exists: {self.run_dir}")
        for path in (
            self.logs_dir,
            self.metrics_dir,
            self.checkpoints_dir,
            self.eval_dir,
            self.reports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


def build_run_layout(
    *,
    root: str | Path,
    experiment_id: str,
    spec_hash: str,
    seed: int,
) -> RunLayout:
    if not spec_hash.startswith("sha256:"):
        raise ValueError("spec_hash must start with sha256:")
    return RunLayout(root=Path(root), experiment_id=experiment_id, spec_hash=spec_hash, seed=seed)
