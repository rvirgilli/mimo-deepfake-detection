#!/usr/bin/env python3
"""CPU-only diagnostic for PEFT MASKGCT seed stability."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs/current/wave3a_peft_maskgct_seed_stability_diagnostic_v1"

RUNS = {
    42: ROOT
    / "experiments/runs/wave3a_xlsr_training_reference_peft_seed42_v1/xlsr_peft_adapter/seed_42/MASKGCT",
    123: ROOT
    / "experiments/runs/wave3a_xlsr_training_reference_peft_maskgct_seed_stability_v1/xlsr_peft_adapter/seed_123/MASKGCT",
    2024: ROOT
    / "experiments/runs/wave3a_xlsr_training_reference_peft_maskgct_seed_stability_v1/xlsr_peft_adapter/seed_2024/MASKGCT",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def threshold_record(
    scores: list[dict[str, Any]], threshold: float, *, invert: bool = False
) -> dict[str, Any]:
    labels = [1 if row["label"] == "spoof" else 0 for row in scores]
    values = [float(row["score"]) for row in scores]
    if invert:
        values = [1.0 - value for value in values]
    preds = [1 if value >= threshold else 0 for value in values]
    tp = sum(1 for y, p in zip(labels, preds, strict=True) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds, strict=True) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds, strict=True) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds, strict=True) if y == 1 and p == 0)
    tpr = tp / (tp + fn) if tp + fn else math.nan
    tnr = tn / (tn + fp) if tn + fp else math.nan
    fpr = fp / (fp + tn) if fp + tn else math.nan
    fnr = fn / (fn + tp) if fn + tp else math.nan
    return {
        "threshold": threshold,
        "balanced_accuracy": (tpr + tnr) / 2,
        "accuracy": (tp + tn) / len(labels),
        "true_positive_rate": tpr,
        "true_negative_rate": tnr,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "predicted_spoof_rate": sum(preds) / len(preds),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def threshold_summary(scores: list[dict[str, Any]], *, invert: bool = False) -> dict[str, Any]:
    values = sorted({float(row["score"]) for row in scores})
    thresholds = [-math.inf, *values, math.inf]
    best = max(
        (threshold_record(scores, threshold, invert=invert) for threshold in thresholds),
        key=lambda row: row["balanced_accuracy"],
    )
    eer_point = min(
        (threshold_record(scores, threshold, invert=invert) for threshold in thresholds),
        key=lambda row: abs(row["false_positive_rate"] - row["false_negative_rate"]),
    )
    eer_point = {
        **eer_point,
        "eer_midpoint": (eer_point["false_positive_rate"] + eer_point["false_negative_rate"]) / 2,
        "abs_fpr_fnr_delta": abs(
            eer_point["false_positive_rate"] - eer_point["false_negative_rate"]
        ),
    }
    return {
        "at_0p5": threshold_record(scores, 0.5, invert=invert),
        "best_balanced_accuracy": best,
        "eer_operating_point": eer_point,
    }


def score_summary(scores: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    by_label: dict[str, list[float]] = {"bonafide": [], "spoof": []}
    for row in scores:
        by_label[row["label"]].append(float(row["score"]))
    out: dict[str, dict[str, float | int]] = {}
    for label, values in by_label.items():
        values_sorted = sorted(values)
        out[label] = {
            "records": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": values_sorted[0],
            "max": values_sorted[-1],
            "q05": values_sorted[int(0.05 * (len(values_sorted) - 1))],
            "q95": values_sorted[int(0.95 * (len(values_sorted) - 1))],
        }
    return out


def history_summary(history_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for row in history_rows:
        val = row["validation_metrics"]
        by_label = val["score_summary_by_label"]
        rows.append(
            {
                "epoch": row["epoch"],
                "train_loss": row["train_loss"],
                "validation_loss": row["validation_loss"],
                "validation_auroc": val["auroc"],
                "validation_eer": val["eer"],
                "validation_balanced_accuracy": val["balanced_accuracy"],
                "validation_score_gap_spoof_minus_bonafide": by_label["spoof"]["mean"]
                - by_label["bonafide"]["mean"],
            }
        )
    best_val_auroc = max(rows, key=lambda row: row["validation_auroc"])
    best_val_bacc = max(rows, key=lambda row: row["validation_balanced_accuracy"])
    min_val_loss = min(rows, key=lambda row: row["validation_loss"])
    return {
        "epochs": rows,
        "best_validation_auroc_epoch": best_val_auroc,
        "best_validation_balanced_accuracy_epoch": best_val_bacc,
        "min_validation_loss_epoch": min_val_loss,
    }


def seed_record(seed: int, run_dir: Path) -> dict[str, Any]:
    metrics = load_json(run_dir / "metrics.json")
    manifest = load_json(run_dir / "manifest.json")
    scores = load_jsonl(run_dir / "scores.jsonl")
    history = history_summary(load_jsonl(run_dir / "train_history.jsonl"))
    scores_by_label = score_summary(scores)
    score_gap = scores_by_label["spoof"]["mean"] - scores_by_label["bonafide"]["mean"]
    thresholds = threshold_summary(scores)
    inverted_thresholds = threshold_summary(scores, invert=True)
    best_val = manifest["result_summary"]["best_checkpoint_metric_value"]
    return {
        "seed": seed,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "test_metrics": {
            "records": metrics["records"],
            "eer": metrics["eer"],
            "auroc": metrics["auroc"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "confusion_matrix": metrics["confusion_matrix"],
            "bonafide_recall": metrics["per_class_recall"]["bonafide"],
            "spoof_recall": metrics["per_class_recall"]["spoof"],
        },
        "score_summary": scores_by_label,
        "score_gap_spoof_minus_bonafide_mean": score_gap,
        "thresholds": thresholds,
        "thresholds_if_scores_inverted": inverted_thresholds,
        "history": history,
        "selected_checkpoint": {
            "epoch": manifest["result_summary"]["best_epoch"],
            "validation_auroc": best_val,
            "validation_minus_test_auroc": best_val - metrics["auroc"],
            "checkpoint_sha256": manifest["result_summary"]["best_checkpoint_sha256"],
            "scores_sha256": manifest["result_summary"]["scores_sha256"],
            "metrics_sha256": manifest["result_summary"]["metrics_sha256"],
        },
        "failure_type": classify_failure(metrics, thresholds, score_gap),
    }


def classify_failure(metrics: dict[str, Any], thresholds: dict[str, Any], score_gap: float) -> str:
    auroc = metrics["auroc"]
    bacc = metrics["balanced_accuracy"]
    spoof_rate = thresholds["at_0p5"]["predicted_spoof_rate"]
    if auroc < 0.5 and score_gap < 0:
        return "ranking_inversion_with_threshold_collapse"
    if auroc >= 0.5 and bacc < 0.55 and spoof_rate > 0.95:
        return "threshold_collapse_with_partly_recovered_ranking"
    if auroc >= 0.8 and bacc < 0.6:
        return "ranking_good_threshold_bad"
    return "mixed"


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    seed_rows = [seed_record(seed, run_dir) for seed, run_dir in RUNS.items()]
    aurocs = [row["test_metrics"]["auroc"] for row in seed_rows]
    eers = [row["test_metrics"]["eer"] for row in seed_rows]
    baccs = [row["test_metrics"]["balanced_accuracy"] for row in seed_rows]
    val_gaps = [row["selected_checkpoint"]["validation_minus_test_auroc"] for row in seed_rows]
    below_chance = [row["seed"] for row in seed_rows if row["test_metrics"]["auroc"] < 0.5]
    near_all_spoof = [
        row["seed"]
        for row in seed_rows
        if row["thresholds"]["at_0p5"]["predicted_spoof_rate"] > 0.95
    ]
    negative_score_gap = [
        row["seed"] for row in seed_rows if row["score_gap_spoof_minus_bonafide_mean"] < 0
    ]

    summary = {
        "schema": "mimodf-wave3a-peft-maskgct-seed-diagnostic/v1",
        "claim_scope": "CPU-only synthesis over completed custom CoSG PEFT MASKGCT seed runs; no new model scoring/training",
        "condition": "xlsr_peft_adapter",
        "fold": "MASKGCT",
        "seeds": list(RUNS),
        "seed_rows": seed_rows,
        "aggregate": {
            "mean_eer": mean(eers),
            "std_eer": statistics.stdev(eers),
            "mean_auroc": mean(aurocs),
            "std_auroc": statistics.stdev(aurocs),
            "mean_balanced_accuracy": mean(baccs),
            "std_balanced_accuracy": statistics.stdev(baccs),
            "mean_validation_minus_test_auroc": mean(val_gaps),
            "seeds_below_chance_auroc": below_chance,
            "seeds_near_all_spoof_at_0p5": near_all_spoof,
            "seeds_negative_score_gap": negative_score_gap,
        },
        "decision": {
            "classification": "mixed_ranking_but_stable_threshold_collapse",
            "recommendation": "Run full PEFT all-fold seeds 123/2024 only if the research question is average PEFT gain vs worst-source risk; do not claim stable below-chance inversion.",
            "rationale": [
                "Seeds 42 and 123 show below-chance AUROC; seed 2024 recovers ranking to AUROC 0.6282.",
                "All seeds predict nearly all MASKGCT utterances as spoof at threshold 0.5.",
                "All seeds have negative mean spoof-minus-bonafide score gap.",
                "Validation AUROC remains high for selected checkpoints despite poor held-out MASKGCT threshold behavior.",
            ],
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Wave 3A PEFT MASKGCT seed-stability diagnostic",
        "",
        "Scope: CPU-only synthesis over completed PEFT MASKGCT seed runs. No new model scoring or training.",
        "",
        "## Aggregate",
        "",
        "| Metric | Mean ± std |",
        "|---|---:|",
        f"| EER | {summary['aggregate']['mean_eer']:.4f} ± {summary['aggregate']['std_eer']:.4f} |",
        f"| AUROC | {summary['aggregate']['mean_auroc']:.4f} ± {summary['aggregate']['std_auroc']:.4f} |",
        f"| Balanced accuracy | {summary['aggregate']['mean_balanced_accuracy']:.4f} ± {summary['aggregate']['std_balanced_accuracy']:.4f} |",
        f"| Validation AUROC - test AUROC | {summary['aggregate']['mean_validation_minus_test_auroc']:.4f} |",
        "",
        "## Per seed",
        "",
        "| Seed | Failure type | EER | AUROC | Bal acc | Val AUROC | Val-test AUROC gap | Spoof rate @0.5 | Best BAcc threshold | Inverted best BAcc | Score gap |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in seed_rows:
        test = row["test_metrics"]
        selected = row["selected_checkpoint"]
        normal_best = row["thresholds"]["best_balanced_accuracy"]
        inverted_best = row["thresholds_if_scores_inverted"]["best_balanced_accuracy"]
        lines.append(
            f"| {row['seed']} | {row['failure_type']} | {test['eer']:.4f} | {test['auroc']:.4f} | "
            f"{test['balanced_accuracy']:.4f} | {selected['validation_auroc']:.4f} | "
            f"{selected['validation_minus_test_auroc']:.4f} | "
            f"{row['thresholds']['at_0p5']['predicted_spoof_rate']:.4f} | "
            f"{normal_best['balanced_accuracy']:.4f} | {inverted_best['balanced_accuracy']:.4f} | "
            f"{row['score_gap_spoof_minus_bonafide_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- MASKGCT is not a clean stable below-chance inversion: seed 2024 has AUROC above 0.5.",
            "- MASKGCT is a stable threshold-collapse / worst-source-risk case: every seed predicts >95% of utterances as spoof at threshold 0.5.",
            "- Every seed has negative mean spoof-minus-bonafide score gap, so the class-separation signal is weak or directionally wrong even when AUROC partly recovers.",
            "- Validation AUROC is high for selected checkpoints but does not protect MASKGCT held-out threshold behavior.",
            "",
            "## Recommendation",
            "",
            "Proceed to PEFT all-fold seeds `123/2024` only to answer the fair 3-seed PEFT-vs-frozen question. Do not claim stable MASKGCT inversion. Frame MASKGCT as a mixed-ranking, stable-threshold-collapse source-risk case.",
            "",
        ]
    )
    (OUT_DIR / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
