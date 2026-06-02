"""Audit local external dependencies without importing them.

The project intentionally keeps heavy/nested external repositories out of Git.
This module makes their local state visible for release/review packages: what is
present, which Git revision it points at, whether it is dirty, and which files a
specific workflow expects.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_DEPENDENCY_SPEC = Path("docs/current/external_dependencies.yaml")
DEFAULT_LOCAL_DEPENDENCY_SPEC = Path("docs/current/external_dependencies.local.yaml")


@dataclass(frozen=True)
class RequiredPathSpec:
    path: str
    kind: str
    expected_sha256: str | None = None


@dataclass(frozen=True)
class ExternalDependencySpec:
    name: str
    path: str
    policy: str
    required_paths: tuple[RequiredPathSpec, ...]
    expected_git_head: str | None = None
    expected_git_remote: str | None = None
    setup: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    spec_source: str = "base"


@dataclass(frozen=True)
class RequiredPathCheck:
    path: str
    present: bool
    kind: str
    size_bytes: int | None = None
    sha256: str | None = None
    expected_sha256: str | None = None
    sha256_matches_expected: bool | None = None


@dataclass(frozen=True)
class ExternalDependencyCheck:
    name: str
    path: str
    present: bool
    policy: str
    spec_source: str
    git_head: str | None
    git_remote: str | None
    expected_git_head: str | None
    expected_git_remote: str | None
    git_head_matches_expected: bool | None
    git_remote_matches_expected: bool | None
    git_dirty: bool | None
    git_status_count: int | None
    git_status_lines: tuple[str, ...]
    required_paths: tuple[RequiredPathCheck, ...]
    setup: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


BUILTIN_DEPENDENCIES = (
    ExternalDependencySpec(
        name="SSL_Anti-spoofing",
        path="SSL_Anti-spoofing",
        policy="local external clone; required for official LA scorer and Tak wav2vec2 recipe",
        required_paths=(
            RequiredPathSpec("evaluate_2021_LA.py", "file"),
            RequiredPathSpec("eval_metric_LA.py", "file"),
            RequiredPathSpec("RawBoost.py", "file"),
            RequiredPathSpec("xlsr2_300m.pt", "file"),
        ),
        notes=(
            "Do not vendor blindly; clone/setup instructions or submodule pin required before external release.",
            "Official tDCF table values must come from evaluate_2021_LA.py output, not project result files.",
        ),
    ),
    ExternalDependencySpec(
        name="MiMo-Audio-Tokenizer",
        path="MiMo-Audio-Tokenizer",
        policy="local external clone; required for MiMo frontend construction",
        required_paths=(
            RequiredPathSpec("pyproject.toml", "file"),
            RequiredPathSpec("mimo_audio_tokenizer/__init__.py", "file"),
            RequiredPathSpec("mimo_audio_tokenizer/model.py", "file"),
        ),
        notes=(
            "MiMo weights remain local artifacts under models/ or an explicit model path.",
            "Document exact clone revision before claiming full rerun reproducibility.",
        ),
    ),
    ExternalDependencySpec(
        name="MiMo model weights",
        path="models/MiMo-Audio-Tokenizer",
        policy="local model artifact; not tracked in Git",
        required_paths=(),
        notes=(
            "Presence is environment-specific; publish configs should reference this through MIMO_TOKENIZER_MODEL or CLI overrides.",
        ),
    ),
)


def load_dependency_specs(
    path: str | Path = DEFAULT_DEPENDENCY_SPEC,
    local_path: str | Path | None = DEFAULT_LOCAL_DEPENDENCY_SPEC,
) -> tuple[ExternalDependencySpec, ...]:
    """Load base dependency specs, optionally overlaying machine-local overrides.

    Local overrides are intentionally separate from the tracked spec so a clean
    scorer clone can be used for release gates without mutating historical local
    dependencies. Override entries need only `name` plus the fields being
    replaced; extra dependency names must be complete specs.
    """

    spec_path = Path(path)
    if not spec_path.is_file():
        specs = BUILTIN_DEPENDENCIES
    else:
        specs = _load_dependency_spec_file(spec_path, source="base")

    if local_path is None:
        return specs
    local_spec_path = Path(local_path)
    if not local_spec_path.is_file():
        return specs
    return _apply_local_overrides(
        specs, _load_dependency_spec_dict(local_spec_path), local_spec_path
    )


def audit_external_dependencies(
    root: str | Path = ".",
    spec_path: str | Path = DEFAULT_DEPENDENCY_SPEC,
    *,
    local_spec_path: str | Path | None = DEFAULT_LOCAL_DEPENDENCY_SPEC,
    hash_files: bool = False,
) -> list[ExternalDependencyCheck]:
    base = Path(root)
    return [
        _audit_one(base, spec, hash_files=hash_files)
        for spec in load_dependency_specs(spec_path, local_path=local_spec_path)
    ]


def render_dependency_checks_json(checks: list[ExternalDependencyCheck]) -> str:
    return json.dumps([check.to_dict() for check in checks], indent=2) + "\n"


def render_dependency_checks_markdown(checks: list[ExternalDependencyCheck]) -> str:
    lines = ["# External dependency audit", ""]
    lines.extend(
        [
            "These directories are intentionally local/ignored, but their state must be visible before release.",
            "",
            "| Dependency | Path | Spec | Present | Git revision | Expected revision | Remote ok | Dirty | Missing required paths |",
            "|---|---|---|---:|---|---|---:|---:|---|",
        ]
    )
    for check in checks:
        missing = [item.path for item in check.required_paths if not item.present]
        lines.append(
            "| "
            + " | ".join(
                [
                    check.name,
                    f"`{check.path}`",
                    check.spec_source,
                    "yes" if check.present else "no",
                    f"`{_short(check.git_head)}`" if check.git_head else "n/a",
                    _format_expected_head(check),
                    _format_match(check.git_remote_matches_expected),
                    _format_dirty(check.git_dirty),
                    ", ".join(f"`{path}`" for path in missing) if missing else "none",
                ]
            )
            + " |"
        )
    lines.append("")

    for check in checks:
        lines.extend(
            [
                f"## {check.name}",
                "",
                f"Policy: {check.policy}",
                f"Spec source: {check.spec_source}",
                "",
            ]
        )
        if check.git_remote:
            lines.append(f"Remote: `{check.git_remote}`")
        if check.expected_git_remote:
            lines.append(f"Expected remote: `{check.expected_git_remote}`")
        if check.expected_git_head:
            lines.append(f"Expected revision: `{check.expected_git_head}`")
            lines.append(f"Revision match: {_format_match(check.git_head_matches_expected)}")
        if check.git_status_lines:
            count = check.git_status_count or len(check.git_status_lines)
            lines.append(
                f"Git status: showing {len(check.git_status_lines)} of {count} changed paths"
            )
            lines.extend(f"- `{line}`" for line in check.git_status_lines)
        if check.setup:
            lines.append("Setup notes:")
            lines.extend(f"- `{step}`" for step in check.setup)
        if check.required_paths:
            lines.append("Required paths:")
            for item in check.required_paths:
                marker = "present" if item.present else "missing"
                detail = f" ({item.kind})"
                if item.size_bytes is not None:
                    detail += f", {_format_bytes(item.size_bytes)}"
                if item.expected_sha256 is not None and item.sha256 is None:
                    detail += ", sha256 not checked"
                if item.expected_sha256 is not None and item.sha256 is not None:
                    detail += f", sha256 expected {_format_match(item.sha256_matches_expected)}"
                if item.sha256 is not None:
                    detail += f", sha256 `{item.sha256}`"
                lines.append(f"- {marker}: `{item.path}`{detail}")
        if check.notes:
            lines.append("Notes:")
            lines.extend(f"- {note}" for note in check.notes)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _load_dependency_spec_file(path: Path, *, source: str) -> tuple[ExternalDependencySpec, ...]:
    data = _load_dependency_spec_dict(path)
    return tuple(_spec_from_dict(item, source=source) for item in data["dependencies"])


def _load_dependency_spec_dict(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("dependencies"), list):
        raise ValueError("dependency spec must contain a dependencies list")
    return data


def _apply_local_overrides(
    specs: tuple[ExternalDependencySpec, ...],
    local_data: dict[str, Any],
    local_path: Path,
) -> tuple[ExternalDependencySpec, ...]:
    by_name = {spec.name: spec for spec in specs}
    order = [spec.name for spec in specs]
    for override in local_data["dependencies"]:
        if not isinstance(override, dict):
            raise ValueError("dependency override entries must be mappings")
        name = _required_string(override, "name")
        if name in by_name:
            by_name[name] = _overlay_spec(by_name[name], override, local_path)
        else:
            by_name[name] = _spec_from_dict(override, source=f"local:{local_path}")
            order.append(name)
    return tuple(by_name[name] for name in order)


def _overlay_spec(
    base: ExternalDependencySpec,
    override: dict[str, Any],
    local_path: Path,
) -> ExternalDependencySpec:
    required_paths_data = override.get("required_paths", None)
    if required_paths_data is None:
        required_paths = base.required_paths
    else:
        if not isinstance(required_paths_data, list):
            raise ValueError("dependency required_paths must be a list")
        required_paths = tuple(_required_path_spec(item) for item in required_paths_data)

    return ExternalDependencySpec(
        name=base.name,
        path=_optional_string(override.get("path")) or base.path,
        policy=_optional_string(override.get("policy")) or base.policy,
        expected_git_head=_optional_string(override.get("expected_git_head"))
        or base.expected_git_head,
        expected_git_remote=_optional_string(override.get("expected_git_remote"))
        or base.expected_git_remote,
        setup=_optional_string_tuple(override.get("setup"), base.setup),
        notes=_optional_string_tuple(override.get("notes"), base.notes),
        required_paths=required_paths,
        spec_source=f"local:{local_path}",
    )


def _spec_from_dict(data: Any, *, source: str) -> ExternalDependencySpec:
    if not isinstance(data, dict):
        raise ValueError("dependency entries must be mappings")
    required_paths = data.get("required_paths", [])
    if required_paths is None:
        required_paths = []
    if not isinstance(required_paths, list):
        raise ValueError("dependency required_paths must be a list")
    return ExternalDependencySpec(
        name=_required_string(data, "name"),
        path=_required_string(data, "path"),
        policy=_required_string(data, "policy"),
        expected_git_head=_optional_string(data.get("expected_git_head")),
        expected_git_remote=_optional_string(data.get("expected_git_remote")),
        setup=tuple(str(item) for item in data.get("setup", ()) or ()),
        notes=tuple(str(item) for item in data.get("notes", ()) or ()),
        required_paths=tuple(_required_path_spec(item) for item in required_paths),
        spec_source=source,
    )


def _required_path_spec(data: Any) -> RequiredPathSpec:
    if not isinstance(data, dict):
        raise ValueError("required path entries must be mappings")
    return RequiredPathSpec(
        path=_required_string(data, "path"),
        kind=_required_string(data, "kind"),
        expected_sha256=_optional_string(data.get("expected_sha256")),
    )


def _audit_one(
    root: Path,
    spec: ExternalDependencySpec,
    *,
    hash_files: bool,
) -> ExternalDependencyCheck:
    path = root / spec.path
    present = path.exists()
    required = tuple(
        _required_path_check(path, item, hash_files=hash_files) for item in spec.required_paths
    )
    git_head: str | None = None
    git_remote: str | None = None
    git_dirty: bool | None = None
    git_status_count: int | None = None
    git_status_lines: tuple[str, ...] = ()
    if present and (path / ".git").exists():
        git_head = _git(path, "rev-parse", "HEAD")
        git_remote = _git(path, "config", "--get", "remote.origin.url")
        status = _git(path, "status", "--porcelain")
        all_status_lines = tuple(line for line in (status or "").splitlines() if line)
        git_status_count = len(all_status_lines)
        git_status_lines = all_status_lines[:40]
        git_dirty = bool(all_status_lines)
    elif present:
        git_dirty = None

    return ExternalDependencyCheck(
        name=spec.name,
        path=spec.path,
        present=present,
        policy=spec.policy,
        spec_source=spec.spec_source,
        git_head=git_head,
        git_remote=git_remote,
        expected_git_head=spec.expected_git_head,
        expected_git_remote=spec.expected_git_remote,
        git_head_matches_expected=_matches_expected(git_head, spec.expected_git_head),
        git_remote_matches_expected=_matches_expected(git_remote, spec.expected_git_remote),
        git_dirty=git_dirty,
        git_status_count=git_status_count,
        git_status_lines=git_status_lines,
        required_paths=required,
        setup=spec.setup,
        notes=spec.notes,
    )


def _required_path_check(
    base: Path,
    spec: RequiredPathSpec,
    *,
    hash_files: bool,
) -> RequiredPathCheck:
    path = base / spec.path
    size_bytes: int | None = None
    sha256: str | None = None
    if spec.kind == "file":
        present = path.is_file()
        if present:
            size_bytes = path.stat().st_size
            if hash_files:
                sha256 = _sha256(path)
    elif spec.kind == "dir":
        present = path.is_dir()
    else:
        raise ValueError(f"unsupported required path kind: {spec.kind}")
    return RequiredPathCheck(
        path=spec.path,
        present=present,
        kind=spec.kind,
        size_bytes=size_bytes,
        sha256=sha256,
        expected_sha256=spec.expected_sha256,
        sha256_matches_expected=_matches_expected(sha256, spec.expected_sha256),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(cwd: Path, *args: str) -> str | None:
    result = subprocess.run(
        ("git", *args),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _matches_expected(actual: str | None, expected: str | None) -> bool | None:
    if expected is None or actual is None:
        return None
    return actual == expected


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"dependency {key} must be a non-empty string")
    return value


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("optional dependency string fields must be non-empty strings when set")
    return value


def _optional_string_tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, list):
        raise ValueError("dependency setup/notes overrides must be lists")
    return tuple(str(item) for item in value)


def _short(value: str | None) -> str:
    if value is None:
        return "n/a"
    return value[:12]


def _format_expected_head(check: ExternalDependencyCheck) -> str:
    if check.expected_git_head is None:
        return "n/a"
    return f"`{_short(check.expected_git_head)}` ({_format_match(check.git_head_matches_expected)})"


def _format_bytes(size: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024
    if unit == "B":
        return f"{size} B"
    return f"{value:.2f} {unit}"


def _format_match(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "yes" if value else "no"


def _format_dirty(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "yes" if value else "no"
