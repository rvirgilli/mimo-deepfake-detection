"""Command line interface for the audited MiMo deepfake project."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from mimodf.audit.artifact_gaps import (
    DEFAULT_ARTIFACT_GAP_POLICY,
    load_artifact_gap_policy,
    render_artifact_gap_policy_json,
    render_artifact_gap_policy_markdown,
)
from mimodf.audit.artifacts import (
    check_artifacts_from_file,
    render_artifact_checks_json,
    render_artifact_checks_markdown,
)
from mimodf.audit.dependencies import (
    DEFAULT_DEPENDENCY_SPEC,
    DEFAULT_LOCAL_DEPENDENCY_SPEC,
    audit_external_dependencies,
    render_dependency_checks_json,
    render_dependency_checks_markdown,
)
from mimodf.audit.package import write_audit_package
from mimodf.audit.release_gate import (
    build_release_gate_report,
    render_release_gate_json,
    render_release_gate_markdown,
)
from mimodf.data.codecfake import (
    build_codecfake_plus_index,
    render_codecfake_summary_json,
    render_codecfake_summary_markdown,
)
from mimodf.data.codecfake_splits import (
    build_source_holdout_plan,
    render_source_holdout_plan_json,
    render_source_holdout_plan_markdown,
)
from mimodf.data.protocol import (
    ProtocolSampleSettings,
    render_protocol_sample_json,
    render_protocol_sample_markdown,
    sample_protocol,
)
from mimodf.evaluation.legacy_components import (
    LegacyEvaluationSettings,
    build_legacy_evaluation_components,
)
from mimodf.evaluation.plan import (
    build_evaluation_plan,
    render_evaluation_plan_json,
    render_evaluation_plan_markdown,
)
from mimodf.evaluation.run import EvaluationRunSettings, run_evaluation
from mimodf.experiments.execution import prepare_experiment_run
from mimodf.experiments.manifest import RunManifest
from mimodf.experiments.spec import SpecValidationError, load_experiment_spec
from mimodf.features.case_contrast import (
    CaseContrastSettings,
    parse_feature_source,
    run_case_contrast,
)
from mimodf.features.diagnostics import PredictionDiagnosticSettings, run_prediction_diagnostics
from mimodf.features.drift import PairedDriftSettings, summarize_paired_feature_drift
from mimodf.features.fusion import ProbeFusionSettings, run_probe_fusion
from mimodf.features.logmel import LogMelFeatureExtractionSettings, extract_logmel_features
from mimodf.features.mechanism import MechanismAnalysisSettings, run_mechanism_analysis
from mimodf.features.mimo import MimoFeatureExtractionSettings, extract_mimo_features
from mimodf.features.predictions import (
    PredictionComparisonSettings,
    compare_predictions,
    parse_prediction_source,
)
from mimodf.features.probe import ProbeSettings, run_feature_probe
from mimodf.features.wav2vec2 import Wav2Vec2FeatureExtractionSettings, extract_wav2vec2_features
from mimodf.features.wavlm import (
    WavLMFeatureExtractionSettings,
    WavLMSmokeExtractionSettings,
    extract_wavlm_features,
    extract_wavlm_smoke_features,
)
from mimodf.logs.execution import (
    DEFAULT_RESEARCH_LOG,
    render_summary_markdown,
    render_validation_text,
    summarize_log,
    validate_log,
)
from mimodf.report.aggregate import (
    aggregate_records,
    render_aggregates_json,
    render_aggregates_markdown,
)
from mimodf.report.compare import (
    compare_experiments,
    render_comparison_json,
    render_comparison_markdown,
)
from mimodf.report.index import (
    build_run_index,
    load_run_index_jsonl,
    render_run_index_jsonl,
    render_run_index_markdown,
)
from mimodf.research.matrix import MatrixValidationError, load_matrix, render_matrix_summary
from mimodf.scoring.compare import (
    compare_score_files,
    render_score_comparison_json,
    render_score_comparison_markdown,
)
from mimodf.scoring.official import run_official_la_scorer
from mimodf.tables.main_table import render_main_table_from_file
from mimodf.training.codecfake import (
    MODEL_CONDITIONS,
    CodecfakeXlsrModelSmokeSettings,
    CodecfakeXlsrPlanSettings,
    CodecfakeXlsrTrainSettings,
    build_codecfake_xlsr_dry_run_plan,
    check_codecfake_xlsr_loaders,
)
from mimodf.training.codecfake_runtime import (
    run_codecfake_xlsr_model_smoke,
    run_codecfake_xlsr_training,
)
from mimodf.training.run import build_legacy_asvspoof_plan, run_legacy_asvspoof_training
from mimodf.transforms.media import (
    AddNoiseSettings,
    MediaTransformSettings,
    add_noise,
    generate_media_transform_smoke,
)

DEFAULT_PROVENANCE = Path("docs/current/main_table_provenance.yaml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mimodf")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Audit/provenance commands")
    audit_sub = audit.add_subparsers(dest="audit_command", required=True)

    table = audit_sub.add_parser("main-table", help="Render audited main table")
    table.add_argument(
        "provenance",
        nargs="?",
        default=str(DEFAULT_PROVENANCE),
        help="Path to main_table_provenance.yaml",
    )

    artifacts = audit_sub.add_parser("check-artifacts", help="Check declared artifacts")
    artifacts.add_argument(
        "provenance",
        nargs="?",
        default=str(DEFAULT_PROVENANCE),
        help="Path to main_table_provenance.yaml",
    )
    artifacts.add_argument("--root", default=".", help="Repository/artifact root")
    artifacts.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format",
    )

    artifact_gaps = audit_sub.add_parser(
        "artifact-gaps",
        help="Render explicit decisions for known historical artifact gaps",
    )
    artifact_gaps.add_argument(
        "policy",
        nargs="?",
        default=str(DEFAULT_ARTIFACT_GAP_POLICY),
        help="Path to artifact_gap_decisions.yaml",
    )
    artifact_gaps.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format",
    )

    dependencies = audit_sub.add_parser(
        "dependencies",
        help="Audit local external dependency clones/artifacts",
    )
    dependencies.add_argument("--root", default=".", help="Repository root")
    dependencies.add_argument(
        "--spec",
        default=str(DEFAULT_DEPENDENCY_SPEC),
        help="Dependency spec YAML",
    )
    dependencies.add_argument(
        "--local-spec",
        default=str(DEFAULT_LOCAL_DEPENDENCY_SPEC),
        help="Optional local dependency override YAML; use 'none' to disable",
    )
    dependencies.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format",
    )
    dependencies.add_argument(
        "--hash-files",
        action="store_true",
        help="Compute SHA-256 for required files; can be slow for multi-GB weights",
    )

    release_gate = audit_sub.add_parser(
        "release-gate",
        help="Summarize release readiness from artifact/dependency audits",
    )
    release_gate.add_argument(
        "provenance",
        nargs="?",
        default=str(DEFAULT_PROVENANCE),
        help="Path to main_table_provenance.yaml",
    )
    release_gate.add_argument("--root", default=".", help="Repository/artifact root")
    release_gate.add_argument(
        "--dependency-spec",
        default=str(DEFAULT_DEPENDENCY_SPEC),
        help="Dependency spec YAML",
    )
    release_gate.add_argument(
        "--dependency-local-spec",
        default=str(DEFAULT_LOCAL_DEPENDENCY_SPEC),
        help="Optional local dependency override YAML; use 'none' to disable",
    )
    release_gate.add_argument(
        "--artifact-gap-policy",
        default=str(DEFAULT_ARTIFACT_GAP_POLICY),
        help="Known artifact gap policy YAML; use 'none' to disable",
    )
    release_gate.add_argument(
        "--allow-known-artifact-gaps",
        action="store_true",
        help="Downgrade missing artifacts listed in the gap policy to warnings",
    )
    release_gate.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format",
    )
    release_gate.add_argument(
        "--hash-files",
        action="store_true",
        help="Compute SHA-256 for dependency files; can be slow for multi-GB weights",
    )
    release_gate.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when blockers are present",
    )
    release_gate.add_argument(
        "--system-profile",
        action="store_true",
        help="Shortcut for --allow-known-artifact-gaps; useful for system/tooling release checks, not full reproduction claims",
    )

    package = audit_sub.add_parser("package", help="Write generated audit package")
    package.add_argument(
        "provenance",
        nargs="?",
        default=str(DEFAULT_PROVENANCE),
        help="Path to main_table_provenance.yaml",
    )
    package.add_argument("--out", required=True, help="Output directory")
    package.add_argument("--root", default=".", help="Repository/artifact root")
    package.add_argument(
        "--no-provenance-copy",
        action="store_true",
        help="Do not copy provenance YAML into output directory",
    )

    data = subparsers.add_parser("data", help="Dataset indexing and protocol utilities")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    codecfake_index = data_sub.add_parser(
        "codecfake-plus-index",
        help="Build a CodecFake+ protocol JSONL index from CoSG/CoRS label files",
    )
    codecfake_index.add_argument("--cosg-labels", help="Path to CoSG_labels.txt")
    codecfake_index.add_argument("--cors-labels", help="Path to CoRS_labels.txt")
    codecfake_index.add_argument(
        "--cosg-audio-root",
        help="Optional extracted CoSG WAV directory used to validate audio paths",
    )
    codecfake_index.add_argument("--out", required=True, help="Output protocol JSONL path")
    codecfake_index.add_argument(
        "--summary-out",
        help="Optional summary output path; format follows --format",
    )
    codecfake_index.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Summary output format",
    )

    sample_protocol_parser = data_sub.add_parser(
        "sample-protocol",
        help="Write a deterministic grouped JSONL protocol sample for smoke runs",
    )
    sample_protocol_parser.add_argument("--input", required=True, help="Input protocol JSONL")
    sample_protocol_parser.add_argument(
        "--out", required=True, help="Output sampled protocol JSONL"
    )
    sample_protocol_parser.add_argument(
        "--group-by",
        nargs="+",
        required=True,
        help="One or more JSON fields used as grouping keys",
    )
    sample_protocol_parser.add_argument("--max-per-group", type=int, required=True)
    sample_protocol_parser.add_argument("--max-records", type=int)
    sample_protocol_parser.add_argument("--seed", type=int, default=42)
    sample_protocol_parser.add_argument(
        "--allow-missing-audio",
        action="store_true",
        help="Keep rows even when audio_path is missing or not staged",
    )
    sample_protocol_parser.add_argument("--overwrite", action="store_true")
    sample_protocol_parser.add_argument(
        "--summary-out",
        help="Optional summary output path; format follows --format",
    )
    sample_protocol_parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Summary output format",
    )

    source_holdout_plan = data_sub.add_parser(
        "codecfake-source-holdout-plan",
        help="Plan deterministic CodecFake+ source-holdout folds without training",
    )
    source_holdout_plan.add_argument("--protocol", required=True, help="Input protocol JSONL")
    source_holdout_plan.add_argument("--subset", default="CoSG", help="Protocol subset to use")
    source_holdout_plan.add_argument("--min-per-label", type=int, default=10)
    source_holdout_plan.add_argument("--validation-source-count", type=int, default=1)
    source_holdout_plan.add_argument(
        "--validation-policy",
        choices=("source", "stratified-row"),
        default="source",
        help="How to choose validation data inside the non-held-out train pool",
    )
    source_holdout_plan.add_argument("--validation-fraction", type=float, default=0.15)
    source_holdout_plan.add_argument("--seed", type=int, default=42)
    source_holdout_plan.add_argument(
        "--allow-missing-audio",
        action="store_true",
        help="Keep rows even when audio_path is missing or not staged",
    )
    source_holdout_plan.add_argument(
        "--summary-out",
        help="Optional summary output path; format follows --format",
    )
    source_holdout_plan.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Summary output format",
    )

    transforms = subparsers.add_parser("transforms", help="Media transform smoke commands")
    transforms_sub = transforms.add_subparsers(dest="transforms_command", required=True)
    add_noise_parser = transforms_sub.add_parser(
        "add-noise",
        help="Add deterministic Gaussian noise after ffmpeg decode/resample",
    )
    add_noise_parser.add_argument("--input", required=True, help="Input audio path")
    add_noise_parser.add_argument("--output", required=True, help="Output WAV path")
    add_noise_parser.add_argument("--sample-rate", type=int, default=16000)
    add_noise_parser.add_argument("--snr-db", type=float, default=20.0)
    add_noise_parser.add_argument("--seed", type=int, default=42)
    add_noise_parser.add_argument("--overwrite", action="store_true")

    media_smoke = transforms_sub.add_parser(
        "media-smoke",
        help="Generate bounded media-transform smoke artifacts for a sampled protocol",
    )
    media_smoke.add_argument("--protocol", required=True, help="Sample protocol JSONL")
    media_smoke.add_argument("--out-root", required=True, help="Output root directory")
    media_smoke.add_argument(
        "--transforms",
        nargs="+",
        default=["resample_8k_16k", "mp3_64k_16k", "noise_snr20"],
        help="Transform ids to run",
    )
    media_smoke.add_argument("--sample-rate", type=int, default=16000)
    media_smoke.add_argument("--seed", type=int, default=42)
    media_smoke.add_argument("--overwrite", action="store_true")

    features = subparsers.add_parser("features", help="Feature extraction commands")
    features_sub = features.add_subparsers(dest="features_command", required=True)
    mimo_extract = features_sub.add_parser(
        "mimo-extract",
        help="Extract MiMo features/codes for a small protocol-indexed audio subset",
    )
    mimo_extract.add_argument(
        "--protocol", required=True, help="Protocol JSONL with audio_path rows"
    )
    mimo_extract.add_argument("--out-dir", required=True, help="Output feature directory")
    mimo_extract.add_argument("--model-path", required=True, help="MiMo model_weights directory")
    mimo_extract.add_argument(
        "--representation",
        required=True,
        choices=("continuous_25hz", "continuous_50hz_native", "rvq_codes"),
    )
    mimo_extract.add_argument(
        "--quantizer-group",
        choices=("all", "early", "late"),
        default="all",
        help="RVQ codebook group used when --representation rvq_codes",
    )
    mimo_extract.add_argument("--max-items", type=int)
    mimo_extract.add_argument("--batch-size", type=int, default=1)
    mimo_extract.add_argument("--device", default="cpu")
    mimo_extract.add_argument("--sample-rate", type=int, default=24000)
    mimo_extract.add_argument(
        "--no-bfloat16",
        action="store_true",
        help="Disable bf16 cast; MiMo FlashAttention usually requires bf16/fp16",
    )
    mimo_extract.add_argument("--overwrite", action="store_true")

    w2v_extract = features_sub.add_parser(
        "wav2vec2-extract",
        help="Extract frozen wav2vec2/XLSR continuous features for a protocol subset",
    )
    w2v_extract.add_argument(
        "--protocol", required=True, help="Protocol JSONL with audio_path rows"
    )
    w2v_extract.add_argument("--out-dir", required=True, help="Output feature directory")
    w2v_extract.add_argument(
        "--checkpoint", required=True, help="XLSR checkpoint path, e.g. xlsr2_300m.pt"
    )
    w2v_extract.add_argument("--max-items", type=int)
    w2v_extract.add_argument("--batch-size", type=int, default=1)
    w2v_extract.add_argument("--device", default="cpu")
    w2v_extract.add_argument("--sample-rate", type=int, default=16000)
    w2v_extract.add_argument("--overwrite", action="store_true")

    wavlm_smoke_extract = features_sub.add_parser(
        "wavlm-smoke-extract",
        help="Extract a bounded WavLM-Base+ smoke feature cache; max 16 items",
    )
    wavlm_smoke_extract.add_argument(
        "--protocol", required=True, help="Protocol JSONL with audio_path rows"
    )
    wavlm_smoke_extract.add_argument("--out-dir", required=True, help="Output feature directory")
    wavlm_smoke_extract.add_argument(
        "--model-id",
        default="microsoft/wavlm-base-plus",
        help="Hugging Face model id or local model directory",
    )
    wavlm_smoke_extract.add_argument(
        "--revision",
        default="b21194173c0af7e94822c1776d162e2659fd4761",
        help="Pinned Hugging Face revision",
    )
    wavlm_smoke_extract.add_argument(
        "--component-id",
        default="frontend:wavlm-base-plus/hf-b211941/v1",
        help="Component id written to the feature manifest",
    )
    wavlm_smoke_extract.add_argument("--max-items", type=int, default=8)
    wavlm_smoke_extract.add_argument("--batch-size", type=int, default=1)
    wavlm_smoke_extract.add_argument("--device", default="cpu")
    wavlm_smoke_extract.add_argument("--sample-rate", type=int, default=16000)
    wavlm_smoke_extract.add_argument("--cache-dir")
    wavlm_smoke_extract.add_argument("--local-files-only", action="store_true")
    wavlm_smoke_extract.add_argument("--overwrite", action="store_true")

    wavlm_extract = features_sub.add_parser(
        "wavlm-extract",
        help="Extract WavLM-Base+ feature cache for an approved protocol",
    )
    wavlm_extract.add_argument(
        "--protocol", required=True, help="Protocol JSONL with audio_path rows"
    )
    wavlm_extract.add_argument("--out-dir", required=True, help="Output feature directory")
    wavlm_extract.add_argument(
        "--model-id",
        default="microsoft/wavlm-base-plus",
        help="Hugging Face model id or local model directory",
    )
    wavlm_extract.add_argument(
        "--revision",
        default="b21194173c0af7e94822c1776d162e2659fd4761",
        help="Pinned Hugging Face revision",
    )
    wavlm_extract.add_argument(
        "--component-id",
        default="frontend:wavlm-base-plus/hf-b211941/v1",
        help="Component id written to the feature manifest",
    )
    wavlm_extract.add_argument("--max-items", type=int)
    wavlm_extract.add_argument("--batch-size", type=int, default=1)
    wavlm_extract.add_argument("--device", default="cpu")
    wavlm_extract.add_argument("--sample-rate", type=int, default=16000)
    wavlm_extract.add_argument("--cache-dir")
    wavlm_extract.add_argument("--local-files-only", action="store_true")
    wavlm_extract.add_argument("--overwrite", action="store_true")

    logmel_extract = features_sub.add_parser(
        "logmel-extract",
        help="Extract a boring log-mel baseline feature cache for a protocol",
    )
    logmel_extract.add_argument(
        "--protocol", required=True, help="Protocol JSONL with audio_path rows"
    )
    logmel_extract.add_argument("--out-dir", required=True, help="Output feature directory")
    logmel_extract.add_argument("--max-items", type=int)
    logmel_extract.add_argument("--sample-rate", type=int, default=16000)
    logmel_extract.add_argument("--n-mels", type=int, default=80)
    logmel_extract.add_argument("--n-fft", type=int, default=400)
    logmel_extract.add_argument("--hop-length", type=int, default=160)
    logmel_extract.add_argument("--win-length", type=int, default=400)
    logmel_extract.add_argument("--fmin", type=float, default=20.0)
    logmel_extract.add_argument("--fmax", type=float, default=7600.0)
    logmel_extract.add_argument("--overwrite", action="store_true")

    feature_probe = features_sub.add_parser(
        "probe",
        help="Run a frozen-feature linear probe over one cached feature directory",
    )
    feature_probe.add_argument("--feature-dir", required=True, help="Feature cache directory")
    feature_probe.add_argument("--out-dir", required=True, help="Probe output directory")
    feature_probe.add_argument(
        "--task",
        required=True,
        choices=("label", "source_model", "quantizer_type", "auxiliary_objective", "decoder_type"),
    )
    feature_probe.add_argument(
        "--split",
        choices=("random-stratified", "holdout-values"),
        default="random-stratified",
    )
    feature_probe.add_argument("--seed", type=int, default=42)
    feature_probe.add_argument("--test-fraction", type=float, default=0.2)
    feature_probe.add_argument("--holdout-field")
    feature_probe.add_argument("--holdout-values", nargs="*", default=[])
    feature_probe.add_argument(
        "--pooling",
        choices=("auto", "continuous_mean_std", "continuous_mean", "rvq_hist"),
        default="auto",
    )
    feature_probe.add_argument("--backend", choices=("auto", "sklearn", "numpy"), default="auto")
    feature_probe.add_argument("--l2", type=float, default=1.0)
    feature_probe.add_argument("--max-iter", type=int, default=500)
    feature_probe.add_argument(
        "--keep-missing-target",
        action="store_true",
        help="Fail instead of dropping rows whose target field is missing",
    )
    feature_probe.add_argument("--overwrite", action="store_true")

    paired_drift = features_sub.add_parser(
        "paired-drift",
        help="Summarize paired clean/transformed feature drift",
    )
    paired_drift.add_argument("--clean-feature-dir", required=True)
    paired_drift.add_argument("--transformed-feature-dir", required=True)
    paired_drift.add_argument("--transform-records", required=True)
    paired_drift.add_argument("--out-json", required=True)
    paired_drift.add_argument("--out-report", required=True)
    paired_drift.add_argument("--pooling", default="continuous_mean_std")
    paired_drift.add_argument("--overwrite", action="store_true")

    mechanism_analysis = features_sub.add_parser(
        "mechanism-analysis",
        help="Case-level mechanism analysis over named prediction files",
    )
    mechanism_analysis.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help="Named prediction file: NAME=path/to/predictions.jsonl",
    )
    mechanism_analysis.add_argument("--protocol", required=True)
    mechanism_analysis.add_argument("--out-dir", required=True)
    mechanism_analysis.add_argument("--reference", required=True)
    mechanism_analysis.add_argument("--positive-label", default="spoof")
    mechanism_analysis.add_argument("--source-model")
    mechanism_analysis.add_argument("--overwrite", action="store_true")

    case_contrast = features_sub.add_parser(
        "case-contrast",
        help="Contrast mechanism-analysis case groups against existing feature caches",
    )
    case_contrast.add_argument("--cases", required=True)
    case_contrast.add_argument("--protocol", required=True)
    case_contrast.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="Named feature directory: NAME=features/path",
    )
    case_contrast.add_argument("--out-dir", required=True)
    case_contrast.add_argument("--reference-system", required=True)
    case_contrast.add_argument("--contrast-system", required=True)
    case_contrast.add_argument("--positive-label", default="spoof")
    case_contrast.add_argument("--overwrite", action="store_true")

    feature_fuse = features_sub.add_parser(
        "fuse-probes",
        help="Average two feature-probe prediction files and report error overlap",
    )
    feature_fuse.add_argument("--left-predictions", required=True)
    feature_fuse.add_argument("--right-predictions", required=True)
    feature_fuse.add_argument("--out-dir", required=True)
    feature_fuse.add_argument("--left-weight", type=float, default=0.5)
    feature_fuse.add_argument("--right-weight", type=float, default=0.5)
    feature_fuse.add_argument("--positive-label", default="spoof")
    feature_fuse.add_argument("--overwrite", action="store_true")

    feature_compare_predictions = features_sub.add_parser(
        "compare-predictions",
        help="Compare two or more probe prediction files and summarize error cases",
    )
    feature_compare_predictions.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help="Named prediction file: NAME=path/to/predictions.jsonl",
    )
    feature_compare_predictions.add_argument("--out-dir", required=True)
    feature_compare_predictions.add_argument("--overwrite", action="store_true")

    feature_diagnose_predictions = features_sub.add_parser(
        "diagnose-predictions",
        help="Join held-out-source predictions to protocol metadata and summarize source diagnostics",
    )
    feature_diagnose_predictions.add_argument("--predictions-root", required=True)
    feature_diagnose_predictions.add_argument("--protocol", required=True)
    feature_diagnose_predictions.add_argument("--out-dir", required=True)
    feature_diagnose_predictions.add_argument("--sources", nargs="+", required=True)
    feature_diagnose_predictions.add_argument(
        "--systems",
        nargs="+",
        required=True,
        help="System directory names under each source, e.g. wav2vec2_xlsr mimo_continuous_25hz",
    )
    feature_diagnose_predictions.add_argument("--positive-label", default="spoof")
    feature_diagnose_predictions.add_argument(
        "--no-audio-metadata",
        action="store_true",
        help="Skip opportunistic WAV duration/header reads from protocol audio_path values",
    )
    feature_diagnose_predictions.add_argument("--overwrite", action="store_true")

    log_parser = subparsers.add_parser("log", help="Research execution log utilities")
    log_sub = log_parser.add_subparsers(dest="log_command", required=True)
    log_validate = log_sub.add_parser("validate", help="Validate the research execution JSONL log")
    log_validate.add_argument(
        "log",
        nargs="?",
        default=str(DEFAULT_RESEARCH_LOG),
        help="Research execution log JSONL path",
    )
    log_validate.add_argument("--strict", action="store_true")
    log_validate.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format",
    )
    log_summary = log_sub.add_parser("summary", help="Summarize the research execution JSONL log")
    log_summary.add_argument(
        "log",
        nargs="?",
        default=str(DEFAULT_RESEARCH_LOG),
        help="Research execution log JSONL path",
    )
    log_summary.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format",
    )

    experiment = subparsers.add_parser("experiment", help="Versioned experiment spec/run commands")
    experiment_sub = experiment.add_subparsers(dest="experiment_command", required=True)

    experiment_validate = experiment_sub.add_parser(
        "validate",
        help="Validate a versioned experiment spec without loading models",
    )
    experiment_validate.add_argument("spec", help="ExperimentSpec YAML")
    experiment_validate.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format",
    )

    experiment_resolve = experiment_sub.add_parser(
        "resolve",
        help="Validate and write a fully resolved experiment spec",
    )
    experiment_resolve.add_argument("spec", help="ExperimentSpec YAML")
    experiment_resolve.add_argument("--out", required=True, help="Resolved spec YAML output path")

    experiment_init = experiment_sub.add_parser(
        "init",
        help="Create a run-layout directory with resolved spec and planned manifest",
    )
    experiment_init.add_argument("spec", help="ExperimentSpec YAML")
    experiment_init.add_argument(
        "--seed", required=True, type=int, help="Declared seed to initialize"
    )
    experiment_init.add_argument("--root", help="Override artifacts.output_root")
    experiment_init.add_argument(
        "--overwrite", action="store_true", help="Overwrite an existing run directory"
    )

    experiment_inspect = experiment_sub.add_parser(
        "inspect",
        help="Inspect a run-layout directory containing manifest/resolved spec artifacts",
    )
    experiment_inspect.add_argument("run_dir", help="Run directory")

    research = subparsers.add_parser("research", help="Research planning utilities")
    research_sub = research.add_subparsers(dest="research_command", required=True)
    matrix_validate = research_sub.add_parser(
        "validate-matrix",
        help="Validate a representation-transfer matrix YAML file",
    )
    matrix_validate.add_argument("matrix", help="representation_transfer_matrix.yaml")
    matrix_validate.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format",
    )
    matrix_summary = research_sub.add_parser(
        "matrix-summary",
        help="Render a readable representation-transfer matrix summary",
    )
    matrix_summary.add_argument("matrix", help="representation_transfer_matrix.yaml")

    eval_parser = subparsers.add_parser("eval", help="Evaluation planning commands")
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)
    eval_plan = eval_sub.add_parser(
        "plan",
        help="Dry-run an evaluation launch without loading models or writing scores",
    )
    eval_plan.add_argument("--config", required=True, help="Publish experiment config YAML")
    eval_plan.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate")
    eval_plan.add_argument("--eval-root", required=True, help="Evaluation audio/protocol root")
    eval_plan.add_argument("--score-out", required=True, help="Planned score-file output path")
    eval_plan.add_argument("--track", required=True, choices=("LA", "DF"))
    eval_plan.add_argument("--scorer", help="Official scorer path; required by default for LA")
    eval_plan.add_argument("--phase", default="eval")
    eval_plan.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format",
    )
    eval_plan.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if the dry-run plan has missing/unsafe paths",
    )

    eval_run = eval_sub.add_parser(
        "run",
        help="Run controlled legacy ASVspoof evaluation from explicit paths",
    )
    eval_run.add_argument("--config", required=True, help="Publish experiment config YAML")
    eval_run.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate")
    eval_run.add_argument("--eval-root", required=True, help="ASVspoof2021 track eval directory")
    eval_run.add_argument("--protocols-path", default="SSL_Anti-spoofing/database")
    eval_run.add_argument("--score-out", required=True, help="Score-file output path")
    eval_run.add_argument("--track", required=True, choices=("LA", "DF"))
    eval_run.add_argument(
        "--frontend", required=True, help="Legacy frontend name, e.g. wav2vec2 or mimo"
    )
    eval_run.add_argument(
        "--legacy-run-config",
        help="Historical resolved run config used to reconstruct model/frontend architecture",
    )
    eval_run.add_argument("--scorer", help="Official LA scorer path")
    eval_run.add_argument("--phase", default="eval")
    eval_run.add_argument("--frontend-checkpoint")
    eval_run.add_argument("--frontend-model-path")
    eval_run.add_argument("--frontend-model-name")
    eval_run.add_argument("--feature-type", default="continuous")
    run_freeze = eval_run.add_mutually_exclusive_group()
    run_freeze.add_argument("--freeze-frontend", action="store_true")
    run_freeze.add_argument("--unfreeze-frontend", action="store_true")
    eval_run.add_argument("--sample-rate", type=int, default=16000)
    eval_run.add_argument("--cut", type=int, default=64600)
    eval_run.add_argument("--batch-size", type=int, default=14)
    eval_run.add_argument("--num-workers", type=int, default=4)
    eval_run.add_argument("--device", default="cpu")
    eval_run.add_argument("--max-items", type=int, help="Limit utterances for real smoke tests")
    eval_run.add_argument("--overwrite", action="store_true")
    eval_run.add_argument("--manifest-out")
    eval_run.add_argument("--score-official", action="store_true")
    eval_run.add_argument("--official-result-out")
    eval_run.add_argument("--python", default="python")
    eval_run.add_argument(
        "--experiment-spec",
        help="Optional ExperimentSpec YAML; writes run-layout v1 artifacts without changing model behavior",
    )
    eval_run.add_argument("--run-seed", type=int, help="Seed declared by --experiment-spec")
    eval_run.add_argument(
        "--run-root", help="Override artifacts.output_root from --experiment-spec"
    )
    eval_run.add_argument(
        "--run-overwrite",
        action="store_true",
        help="Overwrite existing run-layout directory for --experiment-spec",
    )

    report = subparsers.add_parser("report", help="Generated experiment reports")
    report_sub = report.add_subparsers(dest="report_command", required=True)
    report_index = report_sub.add_parser(
        "index",
        help="Build a run index from run-layout roots and optional historical provenance",
    )
    report_index.add_argument(
        "roots",
        nargs="*",
        help="Run-layout roots to scan for manifest.json files",
    )
    report_index.add_argument(
        "--provenance",
        help="Optional main_table_provenance.yaml to include as historical records",
    )
    report_index.add_argument(
        "--format",
        choices=("jsonl", "markdown"),
        default="jsonl",
        help="Output format",
    )
    report_index.add_argument("--out", help="Optional output file")

    report_aggregate = report_sub.add_parser(
        "aggregate",
        help="Aggregate numeric metrics from a run-index JSONL file",
    )
    report_aggregate.add_argument("--index", required=True, help="Run-index JSONL file")
    report_aggregate.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format",
    )
    report_aggregate.add_argument("--out", help="Optional output file")

    report_compare = report_sub.add_parser(
        "compare",
        help="Check whether indexed experiments are comparable",
    )
    report_compare.add_argument("--index", required=True, help="Run-index JSONL file")
    report_compare.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Experiment IDs to compare",
    )
    report_compare.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format",
    )
    report_compare.add_argument("--strict", action="store_true")
    report_compare.add_argument("--out", help="Optional output file")

    score = subparsers.add_parser("score", help="Official scoring commands")
    score_sub = score.add_subparsers(dest="score_command", required=True)
    official_la = score_sub.add_parser("official-la", help="Run official ASVspoof2021 LA scorer")
    official_la.add_argument("score_file")
    official_la.add_argument("--eval-root", required=True)
    official_la.add_argument("--scorer", default="SSL_Anti-spoofing/evaluate_2021_LA.py")
    official_la.add_argument("--phase", default="eval")
    official_la.add_argument("--python", default="python")

    compare_scores = score_sub.add_parser(
        "compare-files",
        help="Compare two ASVspoof score files for reproduction audits",
    )
    compare_scores.add_argument("candidate")
    compare_scores.add_argument("reference")
    compare_scores.add_argument("--tolerance", type=float)
    compare_scores.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format",
    )
    compare_scores.add_argument("--strict", action="store_true")

    train = subparsers.add_parser("train", help="Controlled training commands")
    train_sub = train.add_subparsers(dest="train_command", required=True)
    legacy = train_sub.add_parser(
        "legacy-asvspoof",
        help="Run or dry-run one explicit legacy ASVspoof training job",
    )
    legacy.add_argument("--config", required=True, help="Publish experiment config YAML")
    legacy.add_argument("--out", required=True, help="Training output directory")
    legacy.add_argument("--database-path", required=True, help="ASVspoof database root")
    legacy.add_argument("--protocols-path", required=True, help="ASVspoof protocol root")
    legacy.add_argument(
        "--validation-protocol",
        required=True,
        choices=("asvspoof2021_fast", "asvspoof2019_dev"),
        help="Explicit checkpoint-selection protocol",
    )
    legacy.add_argument(
        "--frontend", required=True, help="Legacy frontend name, e.g. wav2vec2 or mimo"
    )
    legacy.add_argument("--track", default="LA", choices=("LA", "DF"))
    legacy.add_argument("--sample-rate", type=int, default=16000)
    legacy.add_argument("--batch-size", type=int, default=14)
    legacy.add_argument("--eval-batch-size", type=int, default=14)
    legacy.add_argument("--num-workers", type=int, default=4)
    legacy.add_argument("--cut", type=int, default=64600)
    legacy.add_argument("--rawboost-algo", type=int, default=6)
    legacy.add_argument(
        "--rawboost-args-json",
        default="{}",
        help="JSON object passed to legacy RawBoost/data settings",
    )
    legacy.add_argument("--frontend-checkpoint")
    legacy.add_argument("--frontend-model-path")
    legacy.add_argument("--frontend-model-name")
    legacy.add_argument("--feature-type", default="continuous")
    freeze = legacy.add_mutually_exclusive_group()
    freeze.add_argument("--freeze-frontend", action="store_true")
    freeze.add_argument("--unfreeze-frontend", action="store_true")
    legacy.add_argument("--epochs", type=int, default=1)
    legacy.add_argument("--device", default="cpu")
    legacy.add_argument("--top-k-checkpoints", type=int, default=1)
    legacy.add_argument("--max-grad-norm", type=float, default=0.0)
    legacy.add_argument(
        "--max-train-batches", type=int, help="Bound training batches for smoke tests"
    )
    legacy.add_argument(
        "--max-val-batches", type=int, help="Bound validation batches for smoke tests"
    )
    legacy.add_argument("--frontend-prefix", default="frontend")
    legacy.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print required paths without importing Torch/legacy modules",
    )
    legacy.add_argument(
        "--experiment-spec",
        help="Optional ExperimentSpec YAML; writes run-layout v1 artifacts without changing model behavior",
    )
    legacy.add_argument("--run-seed", type=int, help="Seed declared by --experiment-spec")
    legacy.add_argument("--run-root", help="Override artifacts.output_root from --experiment-spec")
    legacy.add_argument(
        "--run-overwrite",
        action="store_true",
        help="Overwrite existing run-layout directory for --experiment-spec",
    )

    codecfake_xlsr = train_sub.add_parser(
        "codecfake-xlsr",
        help="Dry-run a CodecFake+ XLS-R training plan without loading Torch",
    )
    codecfake_xlsr.add_argument("--split-plan", required=True, help="Source-holdout plan JSON")
    codecfake_xlsr.add_argument("--protocol", required=True, help="CodecFake+ protocol JSONL")
    codecfake_xlsr.add_argument("--fold", required=True, help="Held-out source/fold name")
    codecfake_xlsr.add_argument(
        "--condition",
        required=True,
        choices=tuple(sorted(MODEL_CONDITIONS)),
        help="XLS-R training condition",
    )
    codecfake_xlsr.add_argument("--seed", type=int, required=True)
    codecfake_xlsr.add_argument("--out", required=True, help="Output root for planned run")
    codecfake_xlsr.add_argument(
        "--allow-missing-audio",
        action="store_true",
        help="Allow protocol rows with missing audio paths in the dry-run plan",
    )
    codecfake_xlsr.add_argument(
        "--train-subsample-total",
        type=int,
        help="Deterministically subsample train rows to this total (training-size control)",
    )
    codecfake_xlsr.add_argument(
        "--train-subsample-bonafide",
        type=int,
        help="Exact bonafide count within --train-subsample-total",
    )
    codecfake_xlsr.add_argument(
        "--val-subsample-total",
        type=int,
        help="Deterministically subsample validation rows to this total",
    )
    codecfake_xlsr.add_argument(
        "--val-subsample-bonafide",
        type=int,
        help="Exact bonafide count within --val-subsample-total",
    )
    codecfake_xlsr.add_argument(
        "--subsample-seed",
        type=int,
        help="Seed for subsampling (defaults to --seed)",
    )
    codecfake_xlsr.add_argument("--batch-size", type=int, default=2)
    codecfake_xlsr.add_argument("--eval-batch-size", type=int, default=2)
    codecfake_xlsr.add_argument("--num-workers", type=int, default=0)
    codecfake_xlsr.add_argument("--cut", type=int, default=64600)
    codecfake_xlsr.add_argument("--device", default="cpu")
    codecfake_xlsr.add_argument("--epochs", type=int, default=1)
    codecfake_xlsr.add_argument("--max-train-batches", type=int)
    codecfake_xlsr.add_argument("--max-val-batches", type=int)
    codecfake_xlsr.add_argument("--max-test-batches", type=int)
    codecfake_xlsr.add_argument(
        "--checkpoint-metric",
        choices=("val_loss", "val_auroc", "val_eer"),
        default="val_loss",
    )
    codecfake_xlsr.add_argument("--lr", type=float, default=1.0e-4)
    codecfake_xlsr.add_argument("--weight-decay", type=float, default=0.0)
    codecfake_xlsr.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic CuDNN settings and seeded DataLoader shuffling",
    )
    codecfake_xlsr.add_argument(
        "--xlsr-checkpoint",
        default="SSL_Anti-spoofing/xlsr2_300m.pt",
        help="XLS-R fairseq checkpoint for --model-smoke",
    )
    codecfake_xlsr.add_argument(
        "--check-loader",
        action="store_true",
        help="Also build DataLoaders and inspect one batch per split; imports Torch and reads audio",
    )
    codecfake_xlsr.add_argument(
        "--model-smoke",
        action="store_true",
        help="Run one optimizer step and one validation forward pass; no checkpoints/scores/claims",
    )
    codecfake_xlsr.add_argument(
        "--train-run",
        action="store_true",
        help="Run bounded training/eval and write manifest, history, optional checkpoint, scores, and metrics",
    )
    codecfake_xlsr.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Write checkpoints/best.pt during --train-run",
    )
    codecfake_xlsr.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan; required unless --model-smoke is set",
    )

    args = parser.parse_args(argv)

    if args.command == "audit" and args.audit_command == "main-table":
        print(render_main_table_from_file(args.provenance), end="")
        return 0

    if args.command == "audit" and args.audit_command == "check-artifacts":
        checks = check_artifacts_from_file(args.provenance, root=args.root)
        if args.format == "json":
            print(render_artifact_checks_json(checks), end="")
        else:
            print(render_artifact_checks_markdown(checks), end="")
        return 0

    if args.command == "audit" and args.audit_command == "artifact-gaps":
        policy = load_artifact_gap_policy(args.policy)
        if args.format == "json":
            print(render_artifact_gap_policy_json(policy), end="")
        else:
            print(render_artifact_gap_policy_markdown(policy), end="")
        return 0

    if args.command == "audit" and args.audit_command == "dependencies":
        checks = audit_external_dependencies(
            root=args.root,
            spec_path=args.spec,
            local_spec_path=_optional_path_arg(args.local_spec),
            hash_files=args.hash_files,
        )
        if args.format == "json":
            print(render_dependency_checks_json(checks), end="")
        else:
            print(render_dependency_checks_markdown(checks), end="")
        return 0

    if args.command == "audit" and args.audit_command == "release-gate":
        report = build_release_gate_report(
            args.provenance,
            root=args.root,
            dependency_spec_path=args.dependency_spec,
            dependency_local_spec_path=_optional_path_arg(args.dependency_local_spec),
            artifact_gap_policy_path=_optional_path_arg(args.artifact_gap_policy),
            allow_known_artifact_gaps=args.allow_known_artifact_gaps or args.system_profile,
            hash_files=args.hash_files,
        )
        if args.format == "json":
            print(render_release_gate_json(report), end="")
        else:
            print(render_release_gate_markdown(report), end="")
        if args.strict and not report.passed:
            return 1
        return 0

    if args.command == "audit" and args.audit_command == "package":
        package = write_audit_package(
            args.provenance,
            args.out,
            root=args.root,
            include_provenance_copy=not args.no_provenance_copy,
        )
        print(json.dumps(package.to_dict(), indent=2))
        return 0

    if args.command == "data" and args.data_command == "codecfake-plus-index":
        try:
            summary = build_codecfake_plus_index(
                cosg_labels=args.cosg_labels,
                cors_labels=args.cors_labels,
                cosg_audio_root=args.cosg_audio_root,
                out=args.out,
            )
        except (OSError, ValueError) as exc:
            print(f"invalid: {exc}")
            return 1
        rendered = (
            render_codecfake_summary_json(summary)
            if args.format == "json"
            else render_codecfake_summary_markdown(summary)
        )
        if args.summary_out:
            summary_out = Path(args.summary_out)
            summary_out.parent.mkdir(parents=True, exist_ok=True)
            summary_out.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0

    if args.command == "data" and args.data_command == "sample-protocol":
        try:
            sample_summary = sample_protocol(
                ProtocolSampleSettings(
                    input_path=Path(args.input),
                    out_path=Path(args.out),
                    group_by=tuple(args.group_by),
                    max_per_group=args.max_per_group,
                    max_records=args.max_records,
                    seed=args.seed,
                    require_audio=not args.allow_missing_audio,
                    overwrite=args.overwrite,
                )
            )
        except (OSError, ValueError, FileExistsError, json.JSONDecodeError) as exc:
            print(f"invalid: {exc}")
            return 1
        rendered = (
            render_protocol_sample_json(sample_summary)
            if args.format == "json"
            else render_protocol_sample_markdown(sample_summary)
        )
        if args.summary_out:
            summary_out = Path(args.summary_out)
            summary_out.parent.mkdir(parents=True, exist_ok=True)
            summary_out.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0

    if args.command == "data" and args.data_command == "codecfake-source-holdout-plan":
        try:
            plan = build_source_holdout_plan(
                protocol=Path(args.protocol),
                subset=args.subset,
                min_per_label=args.min_per_label,
                validation_source_count=args.validation_source_count,
                validation_policy=args.validation_policy,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
                require_audio=not args.allow_missing_audio,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"invalid: {exc}")
            return 1
        rendered = (
            render_source_holdout_plan_json(plan)
            if args.format == "json"
            else render_source_holdout_plan_markdown(plan)
        )
        if args.summary_out:
            summary_out = Path(args.summary_out)
            summary_out.parent.mkdir(parents=True, exist_ok=True)
            summary_out.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0

    if args.command == "transforms" and args.transforms_command == "add-noise":
        try:
            result = add_noise(
                AddNoiseSettings(
                    input_path=Path(args.input),
                    output_path=Path(args.output),
                    sample_rate=args.sample_rate,
                    snr_db=args.snr_db,
                    seed=args.seed,
                    overwrite=args.overwrite,
                )
            )
        except (OSError, RuntimeError, ValueError, FileExistsError) as exc:
            print(f"invalid: {exc}")
            return 1
        print(json.dumps(result.to_dict(), indent=2), end="\n")
        return 0

    if args.command == "transforms" and args.transforms_command == "media-smoke":
        try:
            result = generate_media_transform_smoke(
                MediaTransformSettings(
                    protocol=Path(args.protocol),
                    out_root=Path(args.out_root),
                    transforms=tuple(args.transforms),
                    sample_rate=args.sample_rate,
                    seed=args.seed,
                    overwrite=args.overwrite,
                )
            )
        except (
            OSError,
            RuntimeError,
            ValueError,
            FileExistsError,
            subprocess.CalledProcessError,
        ) as exc:
            print(f"invalid: {exc}")
            return 1
        print(json.dumps(result.to_dict(), indent=2), end="\n")
        return 0

    if args.command == "features" and args.features_command == "mimo-extract":
        try:
            result = extract_mimo_features(
                MimoFeatureExtractionSettings(
                    protocol=Path(args.protocol),
                    out_dir=Path(args.out_dir),
                    model_path=Path(args.model_path),
                    representation=args.representation,
                    quantizer_group=args.quantizer_group,
                    max_items=args.max_items,
                    batch_size=args.batch_size,
                    device=args.device,
                    use_bfloat16=not args.no_bfloat16,
                    sample_rate=args.sample_rate,
                    overwrite=args.overwrite,
                )
            )
        except (OSError, RuntimeError, ValueError, FileExistsError) as exc:
            print(f"invalid: {exc}")
            return 1
        print(json.dumps(result.to_dict(), indent=2), end="\n")
        return 0

    if args.command == "features" and args.features_command == "wav2vec2-extract":
        try:
            result = extract_wav2vec2_features(
                Wav2Vec2FeatureExtractionSettings(
                    protocol=Path(args.protocol),
                    out_dir=Path(args.out_dir),
                    checkpoint=Path(args.checkpoint),
                    max_items=args.max_items,
                    batch_size=args.batch_size,
                    device=args.device,
                    sample_rate=args.sample_rate,
                    overwrite=args.overwrite,
                )
            )
        except (OSError, RuntimeError, ValueError, FileExistsError) as exc:
            print(f"invalid: {exc}")
            return 1
        print(json.dumps(result.to_dict(), indent=2), end="\n")
        return 0

    if args.command == "features" and args.features_command == "wavlm-smoke-extract":
        try:
            result = extract_wavlm_smoke_features(
                WavLMSmokeExtractionSettings(
                    protocol=Path(args.protocol),
                    out_dir=Path(args.out_dir),
                    model_id=args.model_id,
                    revision=args.revision,
                    component_id=args.component_id,
                    max_items=args.max_items,
                    batch_size=args.batch_size,
                    device=args.device,
                    sample_rate=args.sample_rate,
                    cache_dir=Path(args.cache_dir) if args.cache_dir else None,
                    local_files_only=args.local_files_only,
                    overwrite=args.overwrite,
                )
            )
        except (OSError, RuntimeError, ValueError, FileExistsError) as exc:
            print(f"invalid: {exc}")
            return 1
        print(json.dumps(result.to_dict(), indent=2), end="\n")
        return 0

    if args.command == "features" and args.features_command == "wavlm-extract":
        try:
            result = extract_wavlm_features(
                WavLMFeatureExtractionSettings(
                    protocol=Path(args.protocol),
                    out_dir=Path(args.out_dir),
                    model_id=args.model_id,
                    revision=args.revision,
                    component_id=args.component_id,
                    max_items=args.max_items,
                    batch_size=args.batch_size,
                    device=args.device,
                    sample_rate=args.sample_rate,
                    cache_dir=Path(args.cache_dir) if args.cache_dir else None,
                    local_files_only=args.local_files_only,
                    overwrite=args.overwrite,
                )
            )
        except (OSError, RuntimeError, ValueError, FileExistsError) as exc:
            print(f"invalid: {exc}")
            return 1
        print(json.dumps(result.to_dict(), indent=2), end="\n")
        return 0

    if args.command == "features" and args.features_command == "logmel-extract":
        try:
            result = extract_logmel_features(
                LogMelFeatureExtractionSettings(
                    protocol=Path(args.protocol),
                    out_dir=Path(args.out_dir),
                    max_items=args.max_items,
                    sample_rate=args.sample_rate,
                    n_mels=args.n_mels,
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                    win_length=args.win_length,
                    fmin=args.fmin,
                    fmax=args.fmax,
                    overwrite=args.overwrite,
                )
            )
        except (OSError, RuntimeError, ValueError, FileExistsError) as exc:
            print(f"invalid: {exc}")
            return 1
        print(json.dumps(result.to_dict(), indent=2), end="\n")
        return 0

    if args.command == "features" and args.features_command == "probe":
        try:
            result = run_feature_probe(
                ProbeSettings(
                    feature_dir=Path(args.feature_dir),
                    out_dir=Path(args.out_dir),
                    task=args.task,
                    split=args.split,
                    seed=args.seed,
                    test_fraction=args.test_fraction,
                    holdout_field=args.holdout_field,
                    holdout_values=tuple(args.holdout_values),
                    pooling=args.pooling,
                    backend=args.backend,
                    l2=args.l2,
                    max_iter=args.max_iter,
                    drop_missing_target=not args.keep_missing_target,
                    overwrite=args.overwrite,
                )
            )
        except (OSError, RuntimeError, ValueError, FileExistsError, ImportError) as exc:
            print(f"invalid: {exc}")
            return 1
        print(json.dumps(result.to_dict(), indent=2), end="\n")
        return 0

    if args.command == "features" and args.features_command == "paired-drift":
        try:
            result = summarize_paired_feature_drift(
                PairedDriftSettings(
                    clean_feature_dir=Path(args.clean_feature_dir),
                    transformed_feature_dir=Path(args.transformed_feature_dir),
                    transform_records=Path(args.transform_records),
                    out_json=Path(args.out_json),
                    out_report=Path(args.out_report),
                    pooling=args.pooling,
                    overwrite=args.overwrite,
                )
            )
        except (OSError, RuntimeError, ValueError, FileExistsError, KeyError) as exc:
            print(f"invalid: {exc}")
            return 1
        print(json.dumps(result.to_dict(), indent=2), end="\n")
        return 0

    if args.command == "features" and args.features_command == "mechanism-analysis":
        try:
            result = run_mechanism_analysis(
                MechanismAnalysisSettings(
                    predictions=tuple(parse_prediction_source(item) for item in args.predictions),
                    protocol=Path(args.protocol),
                    out_dir=Path(args.out_dir),
                    reference=args.reference,
                    positive_label=args.positive_label,
                    source_model=args.source_model,
                    overwrite=args.overwrite,
                )
            )
        except (OSError, RuntimeError, ValueError, FileExistsError, KeyError) as exc:
            print(f"invalid: {exc}")
            return 1
        print(json.dumps(result.to_dict(), indent=2), end="\n")
        return 0

    if args.command == "features" and args.features_command == "case-contrast":
        try:
            result = run_case_contrast(
                CaseContrastSettings(
                    cases_path=Path(args.cases),
                    protocol=Path(args.protocol),
                    feature_sources=tuple(parse_feature_source(item) for item in args.features),
                    out_dir=Path(args.out_dir),
                    reference_system=args.reference_system,
                    contrast_system=args.contrast_system,
                    positive_label=args.positive_label,
                    overwrite=args.overwrite,
                )
            )
        except (OSError, RuntimeError, ValueError, FileExistsError, KeyError) as exc:
            print(f"invalid: {exc}")
            return 1
        print(json.dumps(result.to_dict(), indent=2), end="\n")
        return 0

    if args.command == "features" and args.features_command == "fuse-probes":
        try:
            result = run_probe_fusion(
                ProbeFusionSettings(
                    left_predictions=Path(args.left_predictions),
                    right_predictions=Path(args.right_predictions),
                    out_dir=Path(args.out_dir),
                    left_weight=args.left_weight,
                    right_weight=args.right_weight,
                    positive_label=args.positive_label,
                    overwrite=args.overwrite,
                )
            )
        except (OSError, RuntimeError, ValueError, FileExistsError) as exc:
            print(f"invalid: {exc}")
            return 1
        print(json.dumps(result.to_dict(), indent=2), end="\n")
        return 0

    if args.command == "features" and args.features_command == "compare-predictions":
        try:
            result = compare_predictions(
                PredictionComparisonSettings(
                    sources=tuple(parse_prediction_source(item) for item in args.predictions),
                    out_dir=Path(args.out_dir),
                    overwrite=args.overwrite,
                )
            )
        except (OSError, RuntimeError, ValueError, FileExistsError) as exc:
            print(f"invalid: {exc}")
            return 1
        print(json.dumps(result.to_dict(), indent=2), end="\n")
        return 0

    if args.command == "features" and args.features_command == "diagnose-predictions":
        try:
            result = run_prediction_diagnostics(
                PredictionDiagnosticSettings(
                    predictions_root=Path(args.predictions_root),
                    protocol=Path(args.protocol),
                    out_dir=Path(args.out_dir),
                    sources=tuple(args.sources),
                    systems=tuple(args.systems),
                    positive_label=args.positive_label,
                    include_audio_metadata=not args.no_audio_metadata,
                    overwrite=args.overwrite,
                )
            )
        except (OSError, RuntimeError, ValueError, FileExistsError) as exc:
            print(f"invalid: {exc}")
            return 1
        print(json.dumps(result.to_dict(), indent=2), end="\n")
        return 0

    if args.command == "log" and args.log_command == "validate":
        result = validate_log(args.log, strict=args.strict)
        if args.format == "json":
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(render_validation_text(result), end="")
        return 0 if result.passed else 1

    if args.command == "log" and args.log_command == "summary":
        try:
            summary = summarize_log(args.log)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"invalid: {exc}")
            return 1
        if args.format == "json":
            print(json.dumps(summary.to_dict(), indent=2))
        else:
            print(render_summary_markdown(summary), end="")
        return 0

    if args.command == "research" and args.research_command == "validate-matrix":
        try:
            matrix = load_matrix(args.matrix)
        except MatrixValidationError as exc:
            if args.format == "json":
                print(json.dumps({"passed": False, "error": str(exc)}, indent=2))
            else:
                print(f"invalid: {exc}")
            return 1
        payload = {
            "passed": True,
            "matrix_id": matrix["matrix_id"],
            "rows": len(matrix["rows"]),
        }
        if args.format == "json":
            print(json.dumps(payload, indent=2))
        else:
            print(f"valid: {payload['matrix_id']} rows={payload['rows']}")
        return 0

    if args.command == "research" and args.research_command == "matrix-summary":
        try:
            matrix = load_matrix(args.matrix)
        except MatrixValidationError as exc:
            print(f"invalid: {exc}")
            return 1
        print(render_matrix_summary(matrix), end="")
        return 0

    if args.command == "experiment" and args.experiment_command == "validate":
        try:
            spec = load_experiment_spec(args.spec)
        except SpecValidationError as exc:
            if args.format == "json":
                print(json.dumps({"passed": False, "error": str(exc)}, indent=2))
            else:
                print(f"invalid: {exc}")
            return 1
        payload = {"passed": True, "spec_hash": spec.spec_hash()}
        if args.format == "json":
            print(json.dumps(payload, indent=2))
        else:
            print(f"valid: {payload['spec_hash']}")
        return 0

    if args.command == "experiment" and args.experiment_command == "resolve":
        try:
            spec = load_experiment_spec(args.spec)
        except SpecValidationError as exc:
            print(f"invalid: {exc}")
            return 1
        out = spec.write_resolved(args.out)
        print(json.dumps({"resolved_spec": str(out), "spec_hash": spec.spec_hash()}, indent=2))
        return 0

    if args.command == "experiment" and args.experiment_command == "init":
        try:
            prepared = prepare_experiment_run(
                spec_path=args.spec,
                seed=args.seed,
                root=args.root,
                overwrite=args.overwrite,
                status="planned",
            )
        except (SpecValidationError, ValueError, FileExistsError) as exc:
            print(f"invalid: {exc}")
            return 1
        print(
            json.dumps(
                {
                    "run_dir": str(prepared.layout.run_dir),
                    "resolved_spec": str(prepared.layout.resolved_spec_path),
                    "manifest": str(prepared.layout.manifest_path),
                    "spec_hash": prepared.resolved_spec["spec_hash"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "experiment" and args.experiment_command == "inspect":
        run_dir = Path(args.run_dir)
        manifest_path = run_dir / "manifest.json"
        resolved_spec_path = run_dir / "resolved_spec.yaml"
        payload: dict[str, Any] = {
            "run_dir": str(run_dir),
            "manifest_present": manifest_path.is_file(),
            "resolved_spec_present": resolved_spec_path.is_file(),
        }
        if manifest_path.is_file():
            manifest = RunManifest.load(manifest_path)
            payload["manifest"] = manifest.to_dict()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "eval" and args.eval_command == "plan":
        plan = build_evaluation_plan(
            config_path=args.config,
            checkpoint=args.checkpoint,
            eval_root=args.eval_root,
            score_out=args.score_out,
            track=args.track,
            scorer=args.scorer,
            phase=args.phase,
        )
        if args.format == "json":
            print(render_evaluation_plan_json(plan), end="")
        else:
            print(render_evaluation_plan_markdown(plan), end="")
        if args.strict and not plan.passed:
            return 1
        return 0

    if args.command == "eval" and args.eval_command == "run":
        plan = build_evaluation_plan(
            config_path=args.config,
            checkpoint=args.checkpoint,
            eval_root=args.eval_root,
            score_out=args.score_out,
            track=args.track,
            scorer=args.scorer,
            phase=args.phase,
        )
        if not plan.passed:
            print(render_evaluation_plan_markdown(plan), end="")
            return 1
        freeze_frontend = None
        if args.freeze_frontend:
            freeze_frontend = True
        if args.unfreeze_frontend:
            freeze_frontend = False
        experiment_run = None
        if args.experiment_spec is not None:
            if args.run_seed is None:
                raise SystemExit("--run-seed is required with --experiment-spec")
            try:
                experiment_run = prepare_experiment_run(
                    spec_path=args.experiment_spec,
                    seed=args.run_seed,
                    root=args.run_root,
                    overwrite=args.run_overwrite,
                    status="running",
                )
            except (SpecValidationError, ValueError, FileExistsError) as exc:
                print(f"invalid: {exc}")
                return 1
        components = build_legacy_evaluation_components(
            LegacyEvaluationSettings(
                config_path=args.config,
                checkpoint=args.checkpoint,
                eval_root=args.eval_root,
                protocols_path=args.protocols_path,
                track=args.track,
                frontend=args.frontend,
                legacy_run_config=args.legacy_run_config,
                frontend_checkpoint=args.frontend_checkpoint,
                frontend_model_path=args.frontend_model_path,
                frontend_model_name=args.frontend_model_name,
                freeze_frontend=freeze_frontend,
                feature_type=args.feature_type,
                sample_rate=args.sample_rate,
                cut=args.cut,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device,
                max_items=args.max_items,
            )
        )
        result = run_evaluation(
            plan,
            components,
            EvaluationRunSettings(
                overwrite=args.overwrite,
                score_official=args.score_official,
                official_result_out=args.official_result_out,
                manifest_out=args.manifest_out,
                python=args.python,
                experiment_run=experiment_run,
            ),
        )
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    if args.command == "report" and args.report_command == "index":
        records = build_run_index(args.roots, provenance_path=args.provenance)
        if args.format == "markdown":
            rendered = render_run_index_markdown(records)
        else:
            rendered = render_run_index_jsonl(records)
        _write_or_print(rendered, args.out)
        return 0

    if args.command == "report" and args.report_command == "aggregate":
        aggregates = aggregate_records(load_run_index_jsonl(args.index))
        if args.format == "json":
            rendered = render_aggregates_json(aggregates)
        else:
            rendered = render_aggregates_markdown(aggregates)
        _write_or_print(rendered, args.out)
        return 0

    if args.command == "report" and args.report_command == "compare":
        report = compare_experiments(
            load_run_index_jsonl(args.index),
            args.experiments,
            strict=args.strict,
        )
        if args.format == "json":
            rendered = render_comparison_json(report)
        else:
            rendered = render_comparison_markdown(report)
        _write_or_print(rendered, args.out)
        if args.strict and not report.passed:
            return 1
        return 0

    if args.command == "score" and args.score_command == "official-la":
        result = run_official_la_scorer(
            args.score_file,
            args.eval_root,
            scorer_path=args.scorer,
            phase=args.phase,
            python=args.python,
        )
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    if args.command == "score" and args.score_command == "compare-files":
        comparison = compare_score_files(
            args.candidate,
            args.reference,
            tolerance=args.tolerance,
        )
        if args.format == "json":
            print(render_score_comparison_json(comparison), end="")
        else:
            print(render_score_comparison_markdown(comparison), end="")
        if args.strict and not comparison.passed:
            return 1
        return 0

    if args.command == "train" and args.train_command == "codecfake-xlsr":
        if not args.dry_run and not args.model_smoke and not args.train_run:
            print("invalid: set --dry-run, --model-smoke, and/or --train-run")
            return 1
        settings = CodecfakeXlsrPlanSettings(
            split_plan=Path(args.split_plan),
            protocol=Path(args.protocol),
            fold=args.fold,
            condition=args.condition,
            seed=args.seed,
            out_dir=Path(args.out),
            require_audio=not args.allow_missing_audio,
            train_subsample_total=args.train_subsample_total,
            train_subsample_bonafide=args.train_subsample_bonafide,
            validation_subsample_total=args.val_subsample_total,
            validation_subsample_bonafide=args.val_subsample_bonafide,
            subsample_seed=args.subsample_seed,
        )
        try:
            plan = build_codecfake_xlsr_dry_run_plan(settings)
            payload = plan.to_dict()
            if args.check_loader:
                payload["loader_check"] = check_codecfake_xlsr_loaders(
                    settings,
                    batch_size=args.batch_size,
                    eval_batch_size=args.eval_batch_size,
                    cut=args.cut,
                    num_workers=args.num_workers,
                )
            if args.model_smoke:
                payload["implementation_status"] = "model_smoke_only_no_full_training"
                payload["model_smoke"] = run_codecfake_xlsr_model_smoke(
                    CodecfakeXlsrModelSmokeSettings(
                        plan=settings,
                        checkpoint_path=Path(args.xlsr_checkpoint),
                        batch_size=args.batch_size,
                        eval_batch_size=args.eval_batch_size,
                        cut=args.cut,
                        num_workers=args.num_workers,
                        device=args.device,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        deterministic=args.deterministic,
                    )
                )
            if args.train_run:
                payload["implementation_status"] = "bounded_training_run"
                payload["train_run"] = run_codecfake_xlsr_training(
                    CodecfakeXlsrTrainSettings(
                        plan=settings,
                        checkpoint_path=Path(args.xlsr_checkpoint),
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        eval_batch_size=args.eval_batch_size,
                        cut=args.cut,
                        num_workers=args.num_workers,
                        device=args.device,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        max_train_batches=args.max_train_batches,
                        max_val_batches=args.max_val_batches,
                        max_test_batches=args.max_test_batches,
                        save_checkpoints=args.save_checkpoints,
                        checkpoint_metric=args.checkpoint_metric,
                        deterministic=args.deterministic,
                    )
                )
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            print(f"invalid: {exc}")
            return 1
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "train" and args.train_command == "legacy-asvspoof":
        rawboost_args = json.loads(args.rawboost_args_json)
        if not isinstance(rawboost_args, dict):
            raise SystemExit("--rawboost-args-json must decode to a JSON object")
        freeze_frontend = None
        if args.freeze_frontend:
            freeze_frontend = True
        if args.unfreeze_frontend:
            freeze_frontend = False
        plan = build_legacy_asvspoof_plan(
            config_path=args.config,
            database_path=args.database_path,
            protocols_path=args.protocols_path,
            frontend_name=args.frontend,
            validation_protocol=args.validation_protocol,
            track=args.track,
            output_sample_rate=args.sample_rate,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            cut=args.cut,
            rawboost_algo=args.rawboost_algo,
            rawboost_args=rawboost_args,
            frontend_checkpoint=args.frontend_checkpoint,
            frontend_model_path=args.frontend_model_path,
            frontend_model_name=args.frontend_model_name,
            freeze_frontend=freeze_frontend,
            feature_type=args.feature_type,
            epochs=args.epochs,
            device=args.device,
            top_k_checkpoints=args.top_k_checkpoints,
            max_grad_norm=args.max_grad_norm,
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
            frontend_prefix=args.frontend_prefix,
        )
        if args.dry_run:
            print(json.dumps(plan.to_dict(), indent=2))
            return 0
        experiment_run = None
        if args.experiment_spec is not None:
            if args.run_seed is None:
                raise SystemExit("--run-seed is required with --experiment-spec")
            try:
                experiment_run = prepare_experiment_run(
                    spec_path=args.experiment_spec,
                    seed=args.run_seed,
                    root=args.run_root,
                    overwrite=args.run_overwrite,
                    status="running",
                )
            except (SpecValidationError, ValueError, FileExistsError) as exc:
                print(f"invalid: {exc}")
                return 1
        result = run_legacy_asvspoof_training(
            plan,
            output_dir=args.out,
            experiment_run=experiment_run,
        )
        print(json.dumps(_training_result_dict(result), indent=2))
        return 0

    parser.error("unhandled command")
    return 2


def _write_or_print(rendered: str, out_path: str | None) -> None:
    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered)
    else:
        print(rendered, end="")


def _optional_path_arg(value: str | None) -> str | None:
    if value is None or value.lower() in {"", "none", "null"}:
        return None
    return value


def _training_result_dict(result: Any) -> dict[str, object]:
    return {
        "best_checkpoint": str(result.best_checkpoint),
        "manifest_path": str(result.manifest_path),
        "best_val_loss": result.best_val_loss,
        "final_train_loss": result.final_train_loss,
        "epochs_completed": result.epochs_completed,
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
