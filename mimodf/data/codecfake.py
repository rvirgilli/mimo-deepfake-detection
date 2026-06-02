"""CodecFake+ protocol indexing utilities.

The indexer intentionally handles labels/protocols only. Audio downloads and feature
extraction are separate steps so Wave 0 can validate splits before doing ML work.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Subset = Literal["CoSG", "CoRS"]

SCHEMA = "mimodf-protocol-record/v1"
DATASET_ID = "codecfake_plus"


@dataclass(frozen=True)
class CodecFakeRecord:
    schema: str
    dataset_id: str
    subset: Subset
    utterance_id: str
    clip_id: str
    audio_path: str | None
    archive_member: str | None
    label: str
    source_model: str | None
    quantizer_type: str | None
    auxiliary_objective: str | None
    decoder_type: str | None
    speaker_id: str | None
    codec_name: str | None
    split_hint: str | None
    caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "dataset_id": self.dataset_id,
            "subset": self.subset,
            "utterance_id": self.utterance_id,
            "clip_id": self.clip_id,
            "audio_path": self.audio_path,
            "archive_member": self.archive_member,
            "label": self.label,
            "source_model": self.source_model,
            "quantizer_type": self.quantizer_type,
            "auxiliary_objective": self.auxiliary_objective,
            "decoder_type": self.decoder_type,
            "speaker_id": self.speaker_id,
            "codec_name": self.codec_name,
            "split_hint": self.split_hint,
            "caveats": list(self.caveats),
        }


@dataclass(frozen=True)
class CodecFakeIndexSummary:
    dataset_id: str
    records: int
    subsets: dict[str, int]
    labels: dict[str, int]
    source_models: dict[str, int]
    quantizer_types: dict[str, int]
    auxiliary_objectives: dict[str, int]
    decoder_types: dict[str, int]
    codec_names: dict[str, int]
    missing_audio: int
    duplicate_utterance_ids: int
    inputs: dict[str, dict[str, object]]
    output_path: str
    caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "records": self.records,
            "subsets": self.subsets,
            "labels": self.labels,
            "source_models": self.source_models,
            "quantizer_types": self.quantizer_types,
            "auxiliary_objectives": self.auxiliary_objectives,
            "decoder_types": self.decoder_types,
            "codec_names": self.codec_names,
            "missing_audio": self.missing_audio,
            "duplicate_utterance_ids": self.duplicate_utterance_ids,
            "inputs": self.inputs,
            "output_path": self.output_path,
            "caveats": list(self.caveats),
        }


def load_cosg_labels(
    path: str | Path, audio_root: str | Path | None = None
) -> list[CodecFakeRecord]:
    records: list[CodecFakeRecord] = []
    audio_root_path = Path(audio_root) if audio_root is not None else None
    with Path(path).open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 6:
                raise ValueError(f"{path}:{line_number}: expected 6 fields, got {len(parts)}")
            model, clip_id, quantizer, auxiliary, decoder, label = parts
            label_norm = _normalize_label(label)
            audio_path = _cosg_audio_path(audio_root_path, clip_id)
            records.append(
                CodecFakeRecord(
                    schema=SCHEMA,
                    dataset_id=DATASET_ID,
                    subset="CoSG",
                    utterance_id=clip_id,
                    clip_id=clip_id,
                    audio_path=str(audio_path) if audio_path is not None else None,
                    archive_member=f"CoSG/{clip_id}.wav",
                    label=label_norm,
                    source_model=model,
                    quantizer_type=_none_if_real(quantizer),
                    auxiliary_objective=_none_if_real(auxiliary),
                    decoder_type=_none_if_real(decoder),
                    speaker_id=None,
                    codec_name=None,
                    split_hint=None,
                    caveats=(),
                )
            )
    return records


def load_cors_labels(path: str | Path) -> list[CodecFakeRecord]:
    records: list[CodecFakeRecord] = []
    with Path(path).open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 3:
                raise ValueError(f"{path}:{line_number}: expected 3 fields, got {len(parts)}")
            speaker_id, filename, label = parts
            label_norm = _normalize_label(label)
            stem = Path(filename).stem
            clip_id = _cors_clip_id(stem)
            codec_name = _cors_codec_name(stem, label_norm)
            records.append(
                CodecFakeRecord(
                    schema=SCHEMA,
                    dataset_id=DATASET_ID,
                    subset="CoRS",
                    utterance_id=stem,
                    clip_id=clip_id,
                    audio_path=None,
                    archive_member=None,
                    label=label_norm,
                    source_model=None,
                    quantizer_type=None,
                    auxiliary_objective=None,
                    decoder_type=None,
                    speaker_id=speaker_id,
                    codec_name=codec_name,
                    split_hint=_cors_split_hint(speaker_id),
                    caveats=("CoRS taxonomy fields require an explicit codec-name mapping",)
                    if codec_name is not None
                    else (),
                )
            )
    return records


def build_codecfake_plus_index(
    *,
    cosg_labels: str | Path | None = None,
    cors_labels: str | Path | None = None,
    cosg_audio_root: str | Path | None = None,
    out: str | Path,
) -> CodecFakeIndexSummary:
    records: list[CodecFakeRecord] = []
    inputs: dict[str, dict[str, object]] = {}
    if cosg_labels is not None:
        cosg_path = Path(cosg_labels)
        records.extend(load_cosg_labels(cosg_path, audio_root=cosg_audio_root))
        inputs["cosg_labels"] = _file_metadata(cosg_path)
    if cors_labels is not None:
        cors_path = Path(cors_labels)
        records.extend(load_cors_labels(cors_path))
        inputs["cors_labels"] = _file_metadata(cors_path)
    if not records:
        raise ValueError("at least one label file is required")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

    return summarize_codecfake_records(records, output_path=out_path, inputs=inputs)


def summarize_codecfake_records(
    records: Iterable[CodecFakeRecord],
    *,
    output_path: str | Path,
    inputs: dict[str, dict[str, object]] | None = None,
) -> CodecFakeIndexSummary:
    materialized = list(records)
    utterance_counts = Counter(record.utterance_id for record in materialized)
    missing_audio = sum(
        1
        for record in materialized
        if record.audio_path is not None and not Path(record.audio_path).is_file()
    )
    return CodecFakeIndexSummary(
        dataset_id=DATASET_ID,
        records=len(materialized),
        subsets=dict(sorted(Counter(record.subset for record in materialized).items())),
        labels=dict(sorted(Counter(record.label for record in materialized).items())),
        source_models=_counter_without_none(record.source_model for record in materialized),
        quantizer_types=_counter_without_none(record.quantizer_type for record in materialized),
        auxiliary_objectives=_counter_without_none(
            record.auxiliary_objective for record in materialized
        ),
        decoder_types=_counter_without_none(record.decoder_type for record in materialized),
        codec_names=_counter_without_none(record.codec_name for record in materialized),
        missing_audio=missing_audio,
        duplicate_utterance_ids=sum(1 for count in utterance_counts.values() if count > 1),
        inputs=inputs or {},
        output_path=str(output_path),
        caveats=(
            "CoRS rows are labels-only unless CoRS audio archives are staged separately",
            "CoRS-as-spoof is a labeling policy, not a universal deployment truth",
        ),
    )


def render_codecfake_summary_json(summary: CodecFakeIndexSummary) -> str:
    return json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n"


def render_codecfake_summary_markdown(summary: CodecFakeIndexSummary) -> str:
    lines = [
        "# CodecFake+ protocol index summary",
        "",
        f"- dataset: `{summary.dataset_id}`",
        f"- records: {summary.records}",
        f"- output: `{summary.output_path}`",
        f"- missing audio paths: {summary.missing_audio}",
        f"- duplicate utterance IDs: {summary.duplicate_utterance_ids}",
        "",
        "## Subsets",
        *_bullet_counts(summary.subsets),
        "",
        "## Labels",
        *_bullet_counts(summary.labels),
        "",
        "## Source models",
        *_bullet_counts(summary.source_models, limit=20),
        "",
        "## Taxonomy fields",
        "",
        "### Quantizer types",
        *_bullet_counts(summary.quantizer_types),
        "",
        "### Auxiliary objectives",
        *_bullet_counts(summary.auxiliary_objectives),
        "",
        "### Decoder types",
        *_bullet_counts(summary.decoder_types),
        "",
        "## Inputs",
    ]
    for name, metadata in summary.inputs.items():
        lines.extend(
            [
                f"- `{name}`: `{metadata['path']}`",
                f"  - size: {metadata['size_bytes']} bytes",
                f"  - sha256: `{metadata['sha256']}`",
            ]
        )
    lines.extend(["", "## Caveats"])
    lines.extend(f"- {caveat}" for caveat in summary.caveats)
    return "\n".join(lines) + "\n"


def _normalize_label(label: str) -> str:
    lower = label.lower()
    if lower in {"bonafide", "spoof"}:
        return lower
    if lower == "spoofing":
        return "spoof"
    raise ValueError(f"unsupported label: {label}")


def _none_if_real(value: str) -> str | None:
    return None if value.lower() == "real" else value


def _cosg_audio_path(audio_root: Path | None, clip_id: str) -> Path | None:
    if audio_root is None:
        return None
    return audio_root / f"{clip_id}.wav"


def _cors_clip_id(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return stem


def _cors_codec_name(stem: str, label: str) -> str | None:
    if label == "bonafide":
        return None
    parts = stem.split("_")
    if len(parts) <= 2:
        return None
    return "_".join(parts[2:])


def _cors_split_hint(speaker_id: str) -> str | None:
    if speaker_id in {"p226", "p229"}:
        return "validation"
    if speaker_id in {"p227", "p228"}:
        return "evaluation"
    return "train"


def _counter_without_none(values: Iterable[str | None]) -> dict[str, int]:
    return dict(sorted(Counter(value for value in values if value is not None).items()))


def _file_metadata(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bullet_counts(counts: dict[str, int], *, limit: int | None = None) -> list[str]:
    if not counts:
        return ["- none"]
    items = list(counts.items())
    if limit is not None:
        items = items[:limit]
    lines = [f"- `{key}`: {value}" for key, value in items]
    if limit is not None and len(counts) > limit:
        lines.append(f"- ... {len(counts) - limit} more")
    return lines
