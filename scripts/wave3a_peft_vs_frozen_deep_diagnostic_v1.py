#!/usr/bin/env python3
"""Summarize Wave 3A frozen-backend vs PEFT source-holdout diagnostics."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs/current/wave3a_peft_vs_frozen_deep_diagnostic_v1"

FROZEN_SUMMARIES = {
    42: ROOT / "docs/current/wave3a_seed42_allfolds_xlsr_frozen_deterministic_v1/summary.json",
    123: ROOT / "docs/current/wave3a_overnight_gpu_queue_2026_05_31_v1/frozen_seed123/summary.json",
    2024: ROOT
    / "docs/current/wave3a_overnight_gpu_queue_2026_05_31_v1/frozen_seed2024/summary.json",
}
PEFT_SUMMARY = (
    ROOT / "docs/current/wave3a_overnight_gpu_queue_2026_05_31_v1/peft_seed42/summary.json"
)
PEFT_RUN_ROOT = (
    ROOT
    / "experiments/runs/wave3a_xlsr_training_reference_peft_seed42_v1/xlsr_peft_adapter/seed_42"
)
FROZEN_RUN_ROOTS = {
    42: ROOT
    / "experiments/runs/wave3a_xlsr_training_reference_seed42_allfolds_deterministic_v1/xlsr_frozen_backend/seed_42",
    123: ROOT
    / "experiments/runs/wave3a_xlsr_training_reference_frozen_confirmatory_seed123_v1/xlsr_frozen_backend/seed_123",
    2024: ROOT
    / "experiments/runs/wave3a_xlsr_training_reference_frozen_confirmatory_seed2024_v1/xlsr_frozen_backend/seed_2024",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def fold_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["fold"]: row for row in summary["folds"]}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            result[indexed[k][0]] = avg_rank
        i = j
    return result


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(ranks(xs), ranks(ys))


def metrics_for(run_root: Path, fold: str) -> dict[str, Any]:
    return load_json(run_root / fold / "metrics.json")


def diagnostic_fields(metrics: dict[str, Any]) -> dict[str, Any]:
    cm = metrics["confusion_matrix"]
    tn, fp = cm[0]
    fn, tp = cm[1]
    records = tn + fp + fn + tp
    by_label = metrics["score_summary_by_label"]
    bon = by_label["bonafide"]
    spoof = by_label["spoof"]
    score_gap = spoof["mean"] - bon["mean"]
    recall = metrics["per_class_recall"]
    return {
        "predicted_spoof_rate_at_0p5": (fp + tp) / records,
        "false_positive_rate_at_0p5": fp / (tn + fp),
        "false_negative_rate_at_0p5": fn / (fn + tp),
        "recall_gap_spoof_minus_bonafide": recall["spoof"] - recall["bonafide"],
        "score_gap_spoof_minus_bonafide_mean": score_gap,
        "bonafide_score_mean": bon["mean"],
        "spoof_score_mean": spoof["mean"],
        "bonafide_score_median": bon["median"],
        "spoof_score_median": spoof["median"],
    }


def main() -> None:
    frozen_summaries = {seed: load_json(path) for seed, path in FROZEN_SUMMARIES.items()}
    peft_summary = load_json(PEFT_SUMMARY)
    peft_by_fold = fold_map(peft_summary)
    frozen_by_seed = {seed: fold_map(summary) for seed, summary in frozen_summaries.items()}
    folds = list(peft_by_fold)

    rows: list[dict[str, Any]] = []
    for fold in folds:
        frozen_eers = [frozen_by_seed[seed][fold]["eer"] for seed in FROZEN_SUMMARIES]
        frozen_aurocs = [frozen_by_seed[seed][fold]["auroc"] for seed in FROZEN_SUMMARIES]
        frozen_baccs = [
            frozen_by_seed[seed][fold]["balanced_accuracy"] for seed in FROZEN_SUMMARIES
        ]
        peft = peft_by_fold[fold]
        peft_metrics = metrics_for(PEFT_RUN_ROOT, fold)
        frozen42_metrics = metrics_for(FROZEN_RUN_ROOTS[42], fold)
        row = {
            "fold": fold,
            "records": peft["records"],
            "frozen_eer_mean": mean(frozen_eers),
            "frozen_eer_std": stdev(frozen_eers),
            "frozen_eer_by_seed": dict(zip(map(str, FROZEN_SUMMARIES), frozen_eers, strict=True)),
            "frozen_auroc_mean": mean(frozen_aurocs),
            "frozen_balanced_accuracy_mean": mean(frozen_baccs),
            "peft_seed42_eer": peft["eer"],
            "peft_seed42_auroc": peft["auroc"],
            "peft_seed42_balanced_accuracy": peft["balanced_accuracy"],
            "peft_minus_frozen_mean_eer": peft["eer"] - mean(frozen_eers),
            "peft_minus_frozen_seed42_eer": peft["eer"] - frozen_by_seed[42][fold]["eer"],
            "peft_best_epoch": peft["best_epoch"],
            "peft_diagnostics": diagnostic_fields(peft_metrics),
            "frozen_seed42_diagnostics": diagnostic_fields(frozen42_metrics),
        }
        score_gap = row["peft_diagnostics"]["score_gap_spoof_minus_bonafide_mean"]
        auroc = row["peft_seed42_auroc"]
        bacc = row["peft_seed42_balanced_accuracy"]
        if score_gap < 0 and auroc < 0.5:
            interpretation = "inverted_score_ranking"
        elif auroc >= 0.8 and bacc < 0.6:
            interpretation = "ranking_good_threshold_bad"
        elif row["peft_minus_frozen_mean_eer"] < -0.1:
            interpretation = "peft_clear_improvement"
        elif row["peft_minus_frozen_mean_eer"] > 0.05:
            interpretation = "peft_clear_degradation"
        else:
            interpretation = "mixed_or_small_change"
        row["interpretation"] = interpretation
        rows.append(row)

    frozen_seed_means = {
        seed: {
            "mean_eer": summary["mean_eer"],
            "mean_auroc": summary["mean_auroc"],
            "mean_balanced_accuracy": summary["mean_balanced_accuracy"],
        }
        for seed, summary in frozen_summaries.items()
    }
    frozen_eer_means = [v["mean_eer"] for v in frozen_seed_means.values()]
    frozen_auroc_means = [v["mean_auroc"] for v in frozen_seed_means.values()]
    frozen_bacc_means = [v["mean_balanced_accuracy"] for v in frozen_seed_means.values()]

    frozen_fold_eer = [row["frozen_eer_mean"] for row in rows]
    peft_fold_eer = [row["peft_seed42_eer"] for row in rows]
    summary = {
        "schema": "mimodf-wave3a-peft-vs-frozen-deep-diagnostic/v1",
        "claim_scope": "custom CoSG diagnostic source-holdout only; PEFT has seed42 only; frozen backend has seeds 42/123/2024",
        "inputs": {
            "frozen_summaries": {
                str(seed): str(path.relative_to(ROOT)) for seed, path in FROZEN_SUMMARIES.items()
            },
            "peft_summary": str(PEFT_SUMMARY.relative_to(ROOT)),
        },
        "frozen_3seed_macro": {
            "seeds": list(FROZEN_SUMMARIES),
            "mean_eer_mean": mean(frozen_eer_means),
            "mean_eer_std": stdev(frozen_eer_means),
            "mean_auroc_mean": mean(frozen_auroc_means),
            "mean_auroc_std": stdev(frozen_auroc_means),
            "mean_balanced_accuracy_mean": mean(frozen_bacc_means),
            "mean_balanced_accuracy_std": stdev(frozen_bacc_means),
            "by_seed": frozen_seed_means,
        },
        "peft_seed42_macro": {
            "mean_eer": peft_summary["mean_eer"],
            "mean_auroc": peft_summary["mean_auroc"],
            "mean_balanced_accuracy": peft_summary["mean_balanced_accuracy"],
        },
        "peft_vs_frozen_mean": {
            "eer_delta_macro": peft_summary["mean_eer"] - mean(frozen_eer_means),
            "auroc_delta_macro": peft_summary["mean_auroc"] - mean(frozen_auroc_means),
            "balanced_accuracy_delta_macro": peft_summary["mean_balanced_accuracy"]
            - mean(frozen_bacc_means),
            "folds_improved_by_eer": [
                row["fold"] for row in rows if row["peft_minus_frozen_mean_eer"] < 0
            ],
            "folds_degraded_by_eer": [
                row["fold"] for row in rows if row["peft_minus_frozen_mean_eer"] > 0
            ],
            "eer_pearson_frozen_mean_vs_peft": pearson(frozen_fold_eer, peft_fold_eer),
            "eer_spearman_frozen_mean_vs_peft": spearman(frozen_fold_eer, peft_fold_eer),
        },
        "folds": rows,
        "decision_notes": [
            "PEFT seed42 improves average EER strongly relative to frozen 3-seed mean, but this is not yet a PEFT confirmatory claim.",
            "MASKGCT is the dominant PEFT failure: worse EER than frozen mean, AUROC below 0.5, and negative mean score separation.",
            "VALLE shows good PEFT ranking but poor 0.5-threshold balanced accuracy, so calibration/threshold diagnostics matter.",
            "Next GPU-efficient branch should either replicate PEFT with seeds 123/2024 or run MASKGCT-focused diagnostics before full fine-tuning.",
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    csv_lines = [
        "fold,records,frozen_eer_mean,frozen_eer_std,peft_eer,peft_minus_frozen_mean_eer,peft_auroc,peft_balanced_accuracy,peft_score_gap,peft_predicted_spoof_rate,interpretation"
    ]
    for row in rows:
        d = row["peft_diagnostics"]
        csv_lines.append(
            ",".join(
                [
                    row["fold"],
                    str(row["records"]),
                    f"{row['frozen_eer_mean']:.6f}",
                    f"{row['frozen_eer_std']:.6f}",
                    f"{row['peft_seed42_eer']:.6f}",
                    f"{row['peft_minus_frozen_mean_eer']:.6f}",
                    f"{row['peft_seed42_auroc']:.6f}",
                    f"{row['peft_seed42_balanced_accuracy']:.6f}",
                    f"{d['score_gap_spoof_minus_bonafide_mean']:.6f}",
                    f"{d['predicted_spoof_rate_at_0p5']:.6f}",
                    row["interpretation"],
                ]
            )
        )
    (OUT_DIR / "fold_table.csv").write_text("\n".join(csv_lines) + "\n")

    sorted_by_delta = sorted(rows, key=lambda row: row["peft_minus_frozen_mean_eer"])
    lines = [
        "# Wave 3A PEFT vs frozen deep diagnostic",
        "",
        "Scope: custom CoSG source-holdout diagnostic. Frozen backend has seeds `42,123,2024`; PEFT has seed `42` only.",
        "",
        "## Macro result",
        "",
        "| Condition | Seeds | Mean EER | Mean AUROC | Mean balanced acc |",
        "|---|---:|---:|---:|---:|",
        f"| XLS-R frozen backend | 3 | {summary['frozen_3seed_macro']['mean_eer_mean']:.4f} ± {summary['frozen_3seed_macro']['mean_eer_std']:.4f} | {summary['frozen_3seed_macro']['mean_auroc_mean']:.4f} ± {summary['frozen_3seed_macro']['mean_auroc_std']:.4f} | {summary['frozen_3seed_macro']['mean_balanced_accuracy_mean']:.4f} ± {summary['frozen_3seed_macro']['mean_balanced_accuracy_std']:.4f} |",
        f"| XLS-R PEFT adapter | 1 | {peft_summary['mean_eer']:.4f} | {peft_summary['mean_auroc']:.4f} | {peft_summary['mean_balanced_accuracy']:.4f} |",
        "",
        "PEFT seed42 vs frozen 3-seed mean:",
        "",
        f"- EER delta: `{summary['peft_vs_frozen_mean']['eer_delta_macro']:.4f}`",
        f"- AUROC delta: `{summary['peft_vs_frozen_mean']['auroc_delta_macro']:.4f}`",
        f"- balanced-accuracy delta: `{summary['peft_vs_frozen_mean']['balanced_accuracy_delta_macro']:.4f}`",
        "",
        "## Per-source result",
        "",
        "| Fold | Records | Frozen EER mean±std | PEFT EER | ΔEER | PEFT AUROC | PEFT bal acc | PEFT score gap | PEFT spoof-rate@0.5 | Interpretation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted_by_delta:
        d = row["peft_diagnostics"]
        lines.append(
            f"| {row['fold']} | {row['records']} | {row['frozen_eer_mean']:.4f}±{row['frozen_eer_std']:.4f} | "
            f"{row['peft_seed42_eer']:.4f} | {row['peft_minus_frozen_mean_eer']:.4f} | "
            f"{row['peft_seed42_auroc']:.4f} | {row['peft_seed42_balanced_accuracy']:.4f} | "
            f"{d['score_gap_spoof_minus_bonafide_mean']:.4f} | {d['predicted_spoof_rate_at_0p5']:.4f} | {row['interpretation']} |"
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- PEFT is a strong directional improvement: 8/9 folds improve by EER vs frozen 3-seed mean.",
            "- MASKGCT is the critical exception: PEFT EER worsens, AUROC falls below chance, and mean spoof score is lower than mean bonafide score.",
            "- VALLE is different: PEFT AUROC is high but default-threshold balanced accuracy is poor, so calibration/thresholding rather than representation ranking may dominate that fold.",
            "- Frozen-backend source ranking does not directly predict PEFT source ranking; adaptation changes the failure map.",
            "",
            "## Recommended next branch",
            "",
            "1. Run PEFT seeds `123` and `2024` if compute is available; this tests whether the PEFT improvement and MASKGCT collapse are seed-stable.",
            "2. In parallel/after, run MASKGCT-focused diagnostics: score distributions, train/val/test source composition, nearest-source/protocol checks, and calibration curves.",
            "3. Do not start full fine-tuning until PEFT seed stability and MASKGCT failure mode are understood.",
            "",
            "Caveat: this is not official CodecFake+ benchmark training; it is a custom CoSG diagnostic source-holdout protocol.",
            "",
        ]
    )
    (OUT_DIR / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
