# Wave 3 reevaluation — 2026-05-31

Status: historical reassessment after deterministic frozen-backend matrix, overnight PEFT batch2 seed42 matrix, and MASKGCT failure diagnostic. For current planning after seed-stability and batch14 results, use `WAVE3_REVISED_PLAN_2026_05_31.md` and `WEEKLY_TEAM_UPDATE_2026_05_31.md`.

Current caveat: later evidence reclassified `MASKGCT` from a stable below-chance inversion candidate to a mixed-ranking, stable threshold-collapse / worst-source-risk case. PEFT batch14 seed42 is now the preferred directional protocol, with a matched frozen batch14 control pending.

## Bottom line

This is useful evidence. Not because the detector is universally strong, but because the current artifacts expose a trained-transfer failure mode:

> Lightweight XLS-R adaptation can improve average CoSG source-holdout transfer while ordinary non-heldout validation misses held-out-generator operating-point failures.

## Claim boundary

- Custom CoSG source-holdout diagnostic only; not official CodecFake+ benchmark training.
- Frozen XLS-R backend: seeds `42,123,2024` complete.
- XLS-R PEFT adapter: seed `42` complete only.
- CoRS official/proxy training: blocked until extraction/indexing/label policy.
- ASVspoof5: relevant external validation, but not staged locally and not part of current evidence.

## What changed

### Frozen backend is now a confirmed diagnostic baseline

| Metric | 3-seed mean ± std |
|---|---:|
| EER | 0.3401 ± 0.0331 |
| AUROC | 0.7146 ± 0.0448 |
| Balanced accuracy | 0.6057 ± 0.0650 |

### PEFT is the leading model family, but still directional

| Condition | Seeds | Mean EER | Mean AUROC | Mean balanced acc |
|---|---:|---:|---:|---:|
| XLS-R frozen backend | 3 | 0.3401 | 0.7146 | 0.6057 |
| XLS-R PEFT adapter | 1 | 0.2068 | 0.8486 | 0.7247 |

PEFT seed42 improves EER on 8/9 folds. Macro deltas vs frozen 3-seed mean:

- EER: `-0.1333`
- AUROC: `0.1340`
- balanced accuracy: `0.1190`

### MASKGCT is the critical anomaly

| Fact | Value |
|---|---:|
| train rows excluding MASKGCT | 558 |
| validation rows excluding MASKGCT | 87 |
| held-out MASKGCT test rows | 1152 |
| selected PEFT validation AUROC | 0.8291 |
| PEFT MASKGCT test AUROC | 0.4333 |
| PEFT MASKGCT EER | 0.5521 |
| PEFT predicted-spoof rate @0.5 | 0.9800 |

Interpretation: validation on non-MASKGCT sources selected a checkpoint that looks healthy, yet the MASKGCT heldout ranking falls below chance. This is source-shift failure, not just ordinary threshold miscalibration.

## Updated decisions

- Stop treating frozen-feature probes as more than weak triage for trained behavior.
- Treat XLS-R frozen backend as a confirmed diagnostic baseline for CoSG source-holdout, not the target model.
- Treat PEFT as the leading model family, but do not claim PEFT robustness until seeds 123/2024 complete.
- Make MASKGCT the primary failure-mode case study if PEFT collapse repeats across seeds.
- Defer full fine-tuning until PEFT seed stability and MASKGCT mechanism are understood.
- Defer ASVspoof5 and CoRS execution until the local CoSG mechanism story is written cleanly and dataset setup is planned.

## Ranked next actions

### 1. Run targeted PEFT MASKGCT seeds 123 and 2024.

Why: Cheapest test of whether the key failure mode is seed-stable.

Compute: 2 folds x 10 epochs; GPU; small artifact footprint

### 2. If MASKGCT repeats, run PEFT all-fold seeds 123 and 2024.

Why: Turns PEFT-vs-frozen into a fair 3-seed comparison.

Compute: 18 folds x 10 epochs; overnight-class GPU queue

### 3. Write a mechanism note for adaptation-induced source inversion.

Why: This is the emerging scientific contribution, not a benchmark chase.

Compute: CPU/docs; may add score-distribution plots later

### 4. Plan CoRS extraction/indexing and ASVspoof5 staging as external validation tracks.

Why: Needed for official/full-dataset claims, but premature before PEFT stability.

Compute: storage/data engineering first; no model training yet

## Do not do yet

- Do not start full fine-tuning.
- Do not run broad frontend search.
- Do not present CoSG source-holdout as official CodecFake+.
- Do not jump to ASVspoof5 training before staging/protocol review.
- Do not claim PEFT superiority until seeds `123` and `2024` are run or explicitly logged as unavailable.

## New working paper angle

A strong paper/story is now possible around trained transfer diagnostics rather than undirected benchmark optimization:

> Frozen probes weakly triage source difficulty, but trained adaptation rewrites the failure map. PEFT improves most held-out generators in seed42 pilots, while MASKGCT exposes a validation blind spot and source-specific operating-point failure under generator shift.
