#!/usr/bin/env python3
"""Focused diagnostics for the Wave 3A PEFT MASKGCT failure."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from mimodf.data.codecfake_splits import build_source_holdout_rows

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs/current/wave3a_maskgct_failure_diagnostic_v1"
PROTOCOL = ROOT / "features/mimodf/wave0/codecfake_plus_protocol.jsonl"
FOLD = "MASKGCT"
PEFT_RUN = (
    ROOT
    / "experiments/runs/wave3a_xlsr_training_reference_peft_seed42_v1/xlsr_peft_adapter/seed_42/MASKGCT"
)
FROZEN42_RUN = (
    ROOT
    / "experiments/runs/wave3a_xlsr_training_reference_seed42_allfolds_deterministic_v1/xlsr_frozen_backend/seed_42/MASKGCT"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def split_composition(rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    labels = Counter(row["label"] for row in rows)
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_source[row["source_model"]][row["label"]] += 1
    sources = {
        source: {
            "records": sum(counts.values()),
            "labels": summarize_counter(counts),
        }
        for source, counts in sorted(by_source.items())
    }
    return {"records": len(rows), "labels": summarize_counter(labels), "sources": sources}


def score_thresholds(scores: list[dict[str, Any]], *, invert: bool = False) -> dict[str, Any]:
    y = np.array([1 if row["label"] == "spoof" else 0 for row in scores], dtype=np.int64)
    s = np.array([float(row["score"]) for row in scores], dtype=np.float64)
    if invert:
        s = 1.0 - s
    thresholds = np.unique(np.concatenate([[-math.inf], s, [math.inf]]))
    best = None
    eer = None
    for t in thresholds:
        pred = s >= t
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tpr = tp / (tp + fn) if tp + fn else float("nan")
        tnr = tn / (tn + fp) if tn + fp else float("nan")
        fpr = fp / (fp + tn) if fp + tn else float("nan")
        fnr = fn / (fn + tp) if fn + tp else float("nan")
        bacc = (tpr + tnr) / 2
        rec = {
            "threshold": float(t) if math.isfinite(float(t)) else str(float(t)),
            "balanced_accuracy": float(bacc),
            "accuracy": float((tp + tn) / len(y)),
            "true_positive_rate": float(tpr),
            "true_negative_rate": float(tnr),
            "false_positive_rate": float(fpr),
            "false_negative_rate": float(fnr),
            "predicted_spoof_rate": float(pred.mean()),
            "confusion_matrix": [[tn, fp], [fn, tp]],
        }
        if best is None or rec["balanced_accuracy"] > best["balanced_accuracy"]:
            best = rec
        diff = abs(fpr - fnr)
        if eer is None or diff < eer["abs_fpr_fnr_delta"]:
            eer = rec | {"abs_fpr_fnr_delta": float(diff), "eer_midpoint": float((fpr + fnr) / 2)}
    at_05 = min(
        (rec for rec in [threshold_record(scores, threshold=0.5, invert=invert)]),
        key=lambda rec: 0,
    )
    assert best is not None and eer is not None
    return {"at_0p5": at_05, "best_balanced_accuracy": best, "eer_operating_point": eer}


def threshold_record(
    scores: list[dict[str, Any]], *, threshold: float, invert: bool = False
) -> dict[str, Any]:
    y = np.array([1 if row["label"] == "spoof" else 0 for row in scores], dtype=np.int64)
    s = np.array([float(row["score"]) for row in scores], dtype=np.float64)
    if invert:
        s = 1.0 - s
    pred = s >= threshold
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    return {
        "threshold": threshold,
        "balanced_accuracy": float((tpr + tnr) / 2),
        "accuracy": float((tp + tn) / len(y)),
        "true_positive_rate": float(tpr),
        "true_negative_rate": float(tnr),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "predicted_spoof_rate": float(pred.mean()),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def quantiles(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "q05": float(np.quantile(arr, 0.05)),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "q95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def score_summary(scores: list[dict[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, list[float]] = defaultdict(list)
    for row in scores:
        by_label[row["label"]].append(float(row["score"]))
    return {
        label: quantiles(values) | {"records": len(values)}
        for label, values in sorted(by_label.items())
    }


def audio_summary(rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    by_label: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        audio, sr = sf.read(row["audio_path"], dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        abs_audio = np.abs(audio)
        label = row["label"]
        by_label[label]["duration_sec"].append(float(len(audio) / sr))
        by_label[label]["rms"].append(
            float(np.sqrt(np.mean(np.square(audio))) if len(audio) else 0.0)
        )
        by_label[label]["peak_abs"].append(float(np.max(abs_audio) if len(audio) else 0.0))
        by_label[label]["near_zero_fraction_1e-4"].append(
            float(np.mean(abs_audio < 1e-4) if len(audio) else 1.0)
        )
    return {
        label: {metric: quantiles(values) for metric, values in sorted(metrics.items())}
        for label, metrics in sorted(by_label.items())
    }


def history_summary(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    compact = []
    for row in rows:
        val = row["validation_metrics"]
        compact.append(
            {
                "epoch": row["epoch"],
                "train_loss": row["train_loss"],
                "validation_loss": row["validation_loss"],
                "validation_auroc": val["auroc"],
                "validation_eer": val["eer"],
                "validation_balanced_accuracy": val["balanced_accuracy"],
                "validation_score_gap": val["score_summary_by_label"]["spoof"]["mean"]
                - val["score_summary_by_label"]["bonafide"]["mean"],
            }
        )
    best_auroc = max(compact, key=lambda row: row["validation_auroc"])
    return {"epochs": compact, "best_validation_auroc_epoch": best_auroc}


def main() -> None:
    rows = build_source_holdout_rows(
        protocol=PROTOCOL,
        heldout_source=FOLD,
        validation_policy="stratified-row",
        validation_fraction=0.15,
        seed=42,
        require_audio=True,
    )
    peft_metrics = load_json(PEFT_RUN / "metrics.json")
    frozen_metrics = load_json(FROZEN42_RUN / "metrics.json")
    peft_scores = load_jsonl(PEFT_RUN / "scores.jsonl")
    frozen_scores = load_jsonl(FROZEN42_RUN / "scores.jsonl")

    summary = {
        "schema": "mimodf-wave3a-maskgct-failure-diagnostic/v1",
        "claim_scope": "diagnostic analysis over existing Wave 3A custom CoSG source-holdout artifacts; no new model scoring or training",
        "fold": FOLD,
        "split_composition": {
            "train": split_composition(rows.train_rows),
            "validation": split_composition(rows.validation_rows),
            "test": split_composition(rows.test_rows),
        },
        "peft_seed42": {
            "metrics": {
                "eer": peft_metrics["eer"],
                "auroc": peft_metrics["auroc"],
                "balanced_accuracy": peft_metrics["balanced_accuracy"],
                "confusion_matrix": peft_metrics["confusion_matrix"],
            },
            "score_summary": score_summary(peft_scores),
            "thresholds": score_thresholds(peft_scores),
            "thresholds_if_scores_inverted": score_thresholds(peft_scores, invert=True),
            "history": history_summary(PEFT_RUN / "train_history.jsonl"),
        },
        "frozen_seed42": {
            "metrics": {
                "eer": frozen_metrics["eer"],
                "auroc": frozen_metrics["auroc"],
                "balanced_accuracy": frozen_metrics["balanced_accuracy"],
                "confusion_matrix": frozen_metrics["confusion_matrix"],
            },
            "score_summary": score_summary(frozen_scores),
            "thresholds": score_thresholds(frozen_scores),
        },
        "audio_summary": {
            "train": audio_summary(rows.train_rows),
            "validation": audio_summary(rows.validation_rows),
            "test_maskgct": audio_summary(rows.test_rows),
        },
        "interpretation": [
            "The PEFT checkpoint is selected by strong validation AUROC on non-MASKGCT validation rows, but the selected checkpoint ranks MASKGCT bonafide above MASKGCT spoof on average.",
            "The default 0.5 threshold predicts almost all MASKGCT test utterances as spoof; this yields very low bonafide recall and poor balanced accuracy.",
            "MASKGCT is also the largest held-out source by far; the diagnostic fold trains on 558 non-MASKGCT rows and tests on 1152 MASKGCT rows.",
            "If score inversion improves threshold diagnostics, the failure is closer to source-specific ranking inversion than simple calibration.",
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    train = summary["split_composition"]["train"]
    val = summary["split_composition"]["validation"]
    test = summary["split_composition"]["test"]
    peft = summary["peft_seed42"]
    frozen = summary["frozen_seed42"]
    peft_score = peft["score_summary"]
    inv_best = peft["thresholds_if_scores_inverted"]["best_balanced_accuracy"]
    best = peft["thresholds"]["best_balanced_accuracy"]
    lines = [
        "# Wave 3A MASKGCT failure diagnostic",
        "",
        "Scope: CPU-only diagnostic over existing scored artifacts. No new training or model scoring.",
        "",
        "## Why MASKGCT matters",
        "",
        f"- Train rows excluding MASKGCT: `{train['records']}`",
        f"- Validation rows excluding MASKGCT: `{val['records']}`",
        f"- Held-out MASKGCT test rows: `{test['records']}`",
        "- MASKGCT is balanced: `576` bonafide / `576` spoof.",
        "",
        "## PEFT vs frozen on MASKGCT",
        "",
        "| Condition | EER | AUROC | Bal acc | Confusion [[TN,FP],[FN,TP]] |",
        "|---|---:|---:|---:|---|",
        f"| frozen seed42 | {frozen['metrics']['eer']:.4f} | {frozen['metrics']['auroc']:.4f} | {frozen['metrics']['balanced_accuracy']:.4f} | `{frozen['metrics']['confusion_matrix']}` |",
        f"| PEFT seed42 | {peft['metrics']['eer']:.4f} | {peft['metrics']['auroc']:.4f} | {peft['metrics']['balanced_accuracy']:.4f} | `{peft['metrics']['confusion_matrix']}` |",
        "",
        "## PEFT score distribution",
        "",
        "Scores are P(spoof). On MASKGCT, PEFT assigns slightly higher spoof probability to bonafide than spoof on average.",
        "",
        "| Label | Records | Mean | Median | q05 | q95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in ["bonafide", "spoof"]:
        s = peft_score[label]
        lines.append(
            f"| {label} | {s['records']} | {s['mean']:.4f} | {s['median']:.4f} | {s['q05']:.4f} | {s['q95']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Threshold diagnostics",
            "",
            f"- PEFT @0.5 predicted spoof rate: `{peft['thresholds']['at_0p5']['predicted_spoof_rate']:.4f}`",
            f"- PEFT best balanced accuracy threshold: `{best['threshold']}` with balanced acc `{best['balanced_accuracy']:.4f}`",
            f"- If PEFT scores are inverted, best balanced acc becomes `{inv_best['balanced_accuracy']:.4f}`",
            "",
            "## Validation-vs-test mismatch",
            "",
            f"- Selected PEFT epoch by validation AUROC: `{peft['history']['best_validation_auroc_epoch']['epoch']}`",
            f"- Best validation AUROC: `{peft['history']['best_validation_auroc_epoch']['validation_auroc']:.4f}`",
            f"- Test AUROC on MASKGCT: `{peft['metrics']['auroc']:.4f}`",
            "",
            "Interpretation: the validation split, drawn from non-MASKGCT sources, did not detect the MASKGCT-specific inversion. This is source-shift, not just undertraining.",
            "",
            "## Next decision",
            "",
            "Run PEFT seeds `123` and `2024` before claiming the MASKGCT failure is stable. If it repeats, MASKGCT becomes the primary mechanism case study for adaptation-induced source inversion.",
            "",
        ]
    )
    (OUT_DIR / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
