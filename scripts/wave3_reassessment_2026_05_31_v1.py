#!/usr/bin/env python3
"""Generate the Wave 3 reassessment after deterministic frozen and PEFT diagnostics."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_MD = ROOT / "docs/current/WAVE3_REEVALUATION_2026_05_31.md"
OUT_JSON = ROOT / "docs/current/wave3_reevaluation_2026_05_31_v1.json"

FROZEN_42 = ROOT / "docs/current/wave3a_seed42_allfolds_xlsr_frozen_deterministic_v1/summary.json"
FROZEN_123 = (
    ROOT / "docs/current/wave3a_overnight_gpu_queue_2026_05_31_v1/frozen_seed123/summary.json"
)
FROZEN_2024 = (
    ROOT / "docs/current/wave3a_overnight_gpu_queue_2026_05_31_v1/frozen_seed2024/summary.json"
)
PEFT_42 = ROOT / "docs/current/wave3a_overnight_gpu_queue_2026_05_31_v1/peft_seed42/summary.json"
DEEP = ROOT / "docs/current/wave3a_peft_vs_frozen_deep_diagnostic_v1/summary.json"
MASKGCT = ROOT / "docs/current/wave3a_maskgct_failure_diagnostic_v1/summary.json"
PROBE = (
    ROOT
    / "docs/current/wave3a_probe_vs_trained_failure_map_xlsr_frozen_seed42_deterministic_v1.json"
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def r4(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    frozen = {42: load(FROZEN_42), 123: load(FROZEN_123), 2024: load(FROZEN_2024)}
    peft = load(PEFT_42)
    deep = load(DEEP)
    maskgct = load(MASKGCT)
    probe = load(PROBE)

    frozen_eer = [summary["mean_eer"] for summary in frozen.values()]
    frozen_auroc = [summary["mean_auroc"] for summary in frozen.values()]
    frozen_bacc = [summary["mean_balanced_accuracy"] for summary in frozen.values()]

    reassessment = {
        "schema": "mimodf-wave3-reevaluation/v1",
        "date": "2026-05-31",
        "scope": "post-deterministic Wave 3A CoSG diagnostic reassessment",
        "claim_boundary": [
            "custom CoSG source-holdout diagnostic only; not official CodecFake+ benchmark",
            "frozen XLS-R backend has seeds 42/123/2024",
            "PEFT XLS-R adapter has seed42 only",
            "ASVspoof5 and CoRS official/proxy training are not yet staged/executed",
        ],
        "current_thesis": (
            "Probe-guided trained validation shows that lightweight XLS-R adaptation improves "
            "average source-holdout transfer, but can induce a stable-looking source-specific "
            "failure mode that validation on other generators does not detect."
        ),
        "frozen_3seed": {
            "mean_eer": statistics.mean(frozen_eer),
            "std_eer": statistics.stdev(frozen_eer),
            "mean_auroc": statistics.mean(frozen_auroc),
            "std_auroc": statistics.stdev(frozen_auroc),
            "mean_balanced_accuracy": statistics.mean(frozen_bacc),
            "std_balanced_accuracy": statistics.stdev(frozen_bacc),
        },
        "peft_seed42": {
            "mean_eer": peft["mean_eer"],
            "mean_auroc": peft["mean_auroc"],
            "mean_balanced_accuracy": peft["mean_balanced_accuracy"],
        },
        "peft_vs_frozen": deep["peft_vs_frozen_mean"],
        "maskgct_failure": {
            "train_records_excluding_maskgct": maskgct["split_composition"]["train"]["records"],
            "validation_records_excluding_maskgct": maskgct["split_composition"]["validation"][
                "records"
            ],
            "test_records_maskgct": maskgct["split_composition"]["test"]["records"],
            "peft_eer": maskgct["peft_seed42"]["metrics"]["eer"],
            "peft_auroc": maskgct["peft_seed42"]["metrics"]["auroc"],
            "peft_predicted_spoof_rate_at_0p5": maskgct["peft_seed42"]["thresholds"]["at_0p5"][
                "predicted_spoof_rate"
            ],
            "peft_best_validation_auroc": maskgct["peft_seed42"]["history"][
                "best_validation_auroc_epoch"
            ]["validation_auroc"],
            "peft_selected_epoch": maskgct["peft_seed42"]["history"]["best_validation_auroc_epoch"][
                "epoch"
            ],
        },
        "probe_vs_trained": {
            "frozen_seed42_eer_spearman": probe["correlations"]["eer_spearman"],
            "frozen_seed42_auroc_spearman": probe["correlations"]["auroc_spearman"],
        },
        "updated_decisions": [
            "Stop treating frozen-feature probes as more than weak triage for trained behavior.",
            "Treat XLS-R frozen backend as a confirmed diagnostic baseline for CoSG source-holdout, not the target model.",
            "Treat PEFT as the leading model family, but do not claim PEFT robustness until seeds 123/2024 complete.",
            "Make MASKGCT the primary failure-mode case study if PEFT collapse repeats across seeds.",
            "Defer full fine-tuning until PEFT seed stability and MASKGCT mechanism are understood.",
            "Defer ASVspoof5 and CoRS execution until the local CoSG mechanism story is written cleanly and dataset setup is planned.",
        ],
        "next_actions_ranked": [
            {
                "rank": 1,
                "action": "Run targeted PEFT MASKGCT seeds 123 and 2024.",
                "why": "Cheapest test of whether the key failure mode is seed-stable.",
                "compute": "2 folds x 10 epochs; GPU; small artifact footprint",
            },
            {
                "rank": 2,
                "action": "If MASKGCT repeats, run PEFT all-fold seeds 123 and 2024.",
                "why": "Turns PEFT-vs-frozen into a fair 3-seed comparison.",
                "compute": "18 folds x 10 epochs; overnight-class GPU queue",
            },
            {
                "rank": 3,
                "action": "Write a mechanism note for adaptation-induced source inversion.",
                "why": "This is the emerging scientific contribution, not a benchmark chase.",
                "compute": "CPU/docs; may add score-distribution plots later",
            },
            {
                "rank": 4,
                "action": "Plan CoRS extraction/indexing and ASVspoof5 staging as external validation tracks.",
                "why": "Needed for official/full-dataset claims, but premature before PEFT stability.",
                "compute": "storage/data engineering first; no model training yet",
            },
        ],
    }

    OUT_JSON.write_text(json.dumps(reassessment, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Wave 3 reevaluation — 2026-05-31",
        "",
        "Status: post deterministic frozen-backend matrix, post overnight PEFT seed42 matrix, post MASKGCT failure diagnostic.",
        "",
        "## Bottom line",
        "",
        "This is scientifically good. Not because the detector is universally strong, but because the current evidence now exposes a clear trained-transfer phenomenon:",
        "",
        "> Lightweight XLS-R adaptation improves average CoSG source-holdout transfer, but can induce a held-out-generator ranking inversion that ordinary non-heldout validation does not detect.",
        "",
        "## Claim boundary",
        "",
        "- Custom CoSG source-holdout diagnostic only; not official CodecFake+ benchmark training.",
        "- Frozen XLS-R backend: seeds `42,123,2024` complete.",
        "- XLS-R PEFT adapter: seed `42` complete only.",
        "- CoRS official/proxy training: blocked until extraction/indexing/label policy.",
        "- ASVspoof5: relevant external validation, but not staged locally and not part of current evidence.",
        "",
        "## What changed",
        "",
        "### Frozen backend is now a confirmed diagnostic baseline",
        "",
        "| Metric | 3-seed mean ± std |",
        "|---|---:|",
        f"| EER | {r4(reassessment['frozen_3seed']['mean_eer'])} ± {r4(reassessment['frozen_3seed']['std_eer'])} |",
        f"| AUROC | {r4(reassessment['frozen_3seed']['mean_auroc'])} ± {r4(reassessment['frozen_3seed']['std_auroc'])} |",
        f"| Balanced accuracy | {r4(reassessment['frozen_3seed']['mean_balanced_accuracy'])} ± {r4(reassessment['frozen_3seed']['std_balanced_accuracy'])} |",
        "",
        "### PEFT is the leading model family, but still directional",
        "",
        "| Condition | Seeds | Mean EER | Mean AUROC | Mean balanced acc |",
        "|---|---:|---:|---:|---:|",
        f"| XLS-R frozen backend | 3 | {r4(reassessment['frozen_3seed']['mean_eer'])} | {r4(reassessment['frozen_3seed']['mean_auroc'])} | {r4(reassessment['frozen_3seed']['mean_balanced_accuracy'])} |",
        f"| XLS-R PEFT adapter | 1 | {r4(peft['mean_eer'])} | {r4(peft['mean_auroc'])} | {r4(peft['mean_balanced_accuracy'])} |",
        "",
        "PEFT seed42 improves EER on 8/9 folds. Macro deltas vs frozen 3-seed mean:",
        "",
        f"- EER: `{r4(deep['peft_vs_frozen_mean']['eer_delta_macro'])}`",
        f"- AUROC: `{r4(deep['peft_vs_frozen_mean']['auroc_delta_macro'])}`",
        f"- balanced accuracy: `{r4(deep['peft_vs_frozen_mean']['balanced_accuracy_delta_macro'])}`",
        "",
        "### MASKGCT is the critical anomaly",
        "",
        "| Fact | Value |",
        "|---|---:|",
        f"| train rows excluding MASKGCT | {maskgct['split_composition']['train']['records']} |",
        f"| validation rows excluding MASKGCT | {maskgct['split_composition']['validation']['records']} |",
        f"| held-out MASKGCT test rows | {maskgct['split_composition']['test']['records']} |",
        f"| selected PEFT validation AUROC | {r4(reassessment['maskgct_failure']['peft_best_validation_auroc'])} |",
        f"| PEFT MASKGCT test AUROC | {r4(reassessment['maskgct_failure']['peft_auroc'])} |",
        f"| PEFT MASKGCT EER | {r4(reassessment['maskgct_failure']['peft_eer'])} |",
        f"| PEFT predicted-spoof rate @0.5 | {r4(reassessment['maskgct_failure']['peft_predicted_spoof_rate_at_0p5'])} |",
        "",
        "Interpretation: validation on non-MASKGCT sources selected a checkpoint that looks healthy, yet the MASKGCT heldout ranking falls below chance. This is source-shift failure, not just ordinary threshold miscalibration.",
        "",
        "## Updated decisions",
        "",
    ]
    lines += [f"- {item}" for item in reassessment["updated_decisions"]]
    lines += [
        "",
        "## Ranked next actions",
        "",
    ]
    for item in reassessment["next_actions_ranked"]:
        lines += [
            f"### {item['rank']}. {item['action']}",
            "",
            f"Why: {item['why']}",
            "",
            f"Compute: {item['compute']}",
            "",
        ]
    lines += [
        "## Do not do yet",
        "",
        "- Do not start full fine-tuning.",
        "- Do not run broad broad frontend search.",
        "- Do not present CoSG source-holdout as official CodecFake+.",
        "- Do not jump to ASVspoof5 training before staging/protocol review.",
        "- Do not claim PEFT superiority until seeds `123` and `2024` are run or explicitly logged as unavailable.",
        "",
        "## New working paper angle",
        "",
        "A strong paper/story is now possible around trained transfer diagnostics rather than undirected benchmark optimization:",
        "",
        "> Frozen probes weakly triage source difficulty, but trained adaptation rewrites the failure map. PEFT improves most held-out generators yet can invert spoof/bonafide ranking on a large unseen generator family, exposing a validation blind spot under generator shift.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
