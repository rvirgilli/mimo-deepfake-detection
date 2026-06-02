# Wave 3 revised plan — 2026-05-31

Status: current planning authority for Wave 3A as of the batch14 PEFT seed42 pilot.
Scope: custom CodecFake+ CoSG diagnostic source-holdout unless explicitly stated otherwise.
Compute authorization: this document does **not** authorize GPU work by itself; every run still needs a planned research-log row and explicit approval.

## Claim boundary

Current empirical claims are limited to custom CoSG source-holdout diagnostics. They are not official CodecFake+ benchmark results, CoRS proxy-training results, ASVspoof5 results, or media-robustness score claims.

## Current evidence boundary

| Track | Status | Claim allowed |
|---|---|---|
| CoSG source-holdout frozen XLS-R | complete, seeds `42/123/2024`, batch4 | confirmed custom diagnostic baseline |
| CoSG source-holdout PEFT XLS-R batch2 | complete, seed `42`; MASKGCT seeds `42/123/2024` | valid protocol-level evidence; not architecture-only vs frozen batch4 |
| CoSG source-holdout PEFT XLS-R batch14 | complete, seed `42` all folds | strongest directional PEFT protocol so far; seed42 only |
| MASKGCT failure diagnostic | complete across PEFT batch2 seeds; batch14 seed42 pilot complete | mixed-ranking, stable threshold-collapse / worst-source-risk case |
| CoRS official/proxy | downloaded, not extracted/indexed | planning only |
| ASVspoof5 | not staged locally | future external validation target only |
| media transforms | smoke/feature-drift only | no score/EER robustness claim |

## Revised thesis

Use this as the working thesis:

> Lightweight XLS-R adaptation can improve average held-out-generator transfer, while source shift can still produce hidden worst-source operating-point failures.

Do not frame the project as MiMo revival, undirected benchmark optimization, or broad frontend comparison.

## What changed the plan

Earlier PEFT batch2 results were promising but confounded against the frozen batch4 baseline. Batch-size review and local feasibility tests showed batch14 is a better declared PEFT protocol: literature-aligned and locally feasible. PEFT batch14 seed42 all-folds then improved mean EER/AUROC versus PEFT batch2 seed42, while `MASKGCT` remained the worst fold and still showed near-all-spoof default-threshold behavior.

Key batch14 seed42 facts:

| Metric | Value |
|---|---:|
| PEFT batch14 mean EER | `0.1908` |
| PEFT batch14 mean AUROC | `0.8856` |
| PEFT batch14 mean balanced accuracy | `0.7166` |
| Worst fold by EER | `MASKGCT` |
| MASKGCT EER | `0.3646` |
| MASKGCT AUROC | `0.6830` |
| MASKGCT spoof rate @0.5 | `0.9870` |

Interpretation:

- batch14 is a viable and cleaner protocol;
- PEFT remains promising, but seed42 only under batch14;
- `MASKGCT` is no longer a stable below-chance inversion claim; it is a worst-source threshold-collapse / operating-point failure case;
- architecture-only PEFT-vs-frozen claims remain blocked until batch policy is matched or explicitly modeled.

## Revised priorities

### P0 — matched frozen batch14 seed42 control

Proposed run ID:

```text
wave3a-frozen-batch14-seed42-allfolds-v1
```

Purpose:

- check whether batch14 also improves the frozen backend;
- avoid attributing batch/update-schedule effects to PEFT;
- establish a fair seed42 protocol-level comparison before spending on multi-seed matrices.

Protocol:

- condition: `xlsr_frozen_backend`;
- seed: `42`;
- all 9 CoSG source-holdout folds;
- batch size / eval batch size: `14`;
- epochs: `10`;
- checkpoint metric: `val_auroc`;
- deterministic: true.

### P1 — compare PEFT batch14 seed42 vs frozen batch14 seed42

Decision rule:

- if PEFT batch14 remains clearly better, run PEFT batch14 seeds `123/2024` and decide whether frozen batch14 seeds `123/2024` are needed for a matched 3-seed control;
- if frozen batch14 catches up, shift the thesis away from PEFT advantage and toward batch/update protocol plus source-threshold failure.

### P2 — mechanism note

Write:

```text
Adaptation and operating-point failure under held-out codec-generator shift
```

Must include:

- MASKGCT score distributions by label/source;
- threshold curves and default-threshold confusion;
- calibration-vs-ranking separation;
- validation-vs-heldout mismatch;
- seed stability;
- batch2-vs-batch14 comparison;
- artifact/confound audit.

### P3 — CoRS official/proxy preparation

No CoRS training yet. First do data engineering:

1. extraction/storage plan;
2. archive integrity/readability check;
3. CoRS audio index;
4. label-policy doc: CoRS-as-proxy, not literal fake speech;
5. official split materialization;
6. loader smoke.

Only then plan CoRS proxy training.

### P4 — ASVspoof5 staging

No ASVspoof5 training/evaluation yet. First:

1. stage official audio/protocols;
2. integrate Track 1 scorer/protocol reader;
3. decide closed/open condition;
4. dry-run index check;
5. bounded scoring smoke.

ASVspoof5 becomes external validation after the CoSG mechanism is stable.

### P5 — media robustness

Do after the batch14 control branch, not before:

- durable transformed CoSG artifacts;
- clean-vs-transform scoring from selected checkpoints;
- margin erosion and label-flip diagnostics;
- only then media-augmented PEFT.

## Explicitly deferred

- full XLS-R fine-tuning;
- MiMo or WavLM trained variants;
- Optuna/HPO;
- broad frontend search;
- ASVspoof5 official claims;
- CoRS official/proxy claims;
- robustness EER claims from temporary transform artifacts.

## Claim policy update

Allowed now:

> In a custom CoSG source-holdout diagnostic, frozen XLS-R is a stable 3-seed batch4 baseline. PEFT batch14 seed42 is the strongest current directional protocol, improving average transfer in this pilot while `MASKGCT` remains the worst-source threshold-collapse case.

Not allowed yet:

- PEFT architecture alone is better than frozen XLS-R.
- PEFT batch14 is seed-stable.
- MASKGCT is a stable below-chance inversion.
- This is official CodecFake+ performance.
- ASVspoof5 or CoRS supports the claim.
- Full fine-tuning is necessary or helpful.

## Stop rules

Stop and reassess before expanding compute if any of these happen:

- frozen batch14 seed42 matches or exceeds PEFT batch14 seed42;
- training logs show checkpoint/score determinism gaps;
- any source or seed is silently missing;
- CoRS extraction reveals label/readability ambiguity;
- ASVspoof5 staging cannot reproduce official protocol counts.

## Next concrete action

If approved, run the matched frozen batch14 seed42 all-fold control:

```text
wave3a-frozen-batch14-seed42-allfolds-v1
```

Then compare it directly to `wave3a-peft-batch14-seed42-allfolds-v1` before launching any full batch14 multi-seed matrix.
