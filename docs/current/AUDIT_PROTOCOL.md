# Audit protocol

Purpose: validate existing evidence while avoiding costly reruns.

## Validation states

Use these labels in `RESULTS_PROVENANCE.md`:

- `verified` — score file, config, checkpoint/log, and evaluator are identified; metric matches manuscript/table.
- `verified_external` — result comes from an external baseline or paper; source is cited and protocol differences are stated.
- `partial` — some evidence exists, but one required artifact is missing or the run is incomplete.
- `unverified` — number exists in prose/table only, without traceable artifacts.
- `invalid` — artifact contradicts the claimed number or protocol.
- `needs_rerun` — central result cannot be validated from existing artifacts.

## Required evidence for a result row

For each seed/result:

1. model and strategy;
2. seed;
3. training config path;
4. checkpoint path, or reason checkpoint is unnecessary/external;
5. score file path;
6. result file path;
7. evaluator script and whether it is official;
8. EER and min-tDCF where applicable;
9. notes on interruption, checkpoint selection, excluded seed, external baseline, or protocol mismatch.

## Preferred evidence order

1. Official evaluator output from existing score file.
2. Existing result file produced by project evaluator, cross-checked against score file.
3. Training logs/manifests for checkpoint selection and early stopping.
4. Archived docs only as pointers, never as final evidence.

## Rerun policy

Do not rerun training unless all are true:

- the result is central to a paper claim;
- no score/checkpoint/log evidence exists;
- changing/removing the claim would materially weaken the paper;
- we record expected GPU hours and get explicit approval.

Evaluation-only reruns are allowed if scores or checkpoints already exist and compute cost is acceptable, but still prefer existing official outputs.

## Outlier policy

Default: include all completed seeds.

Excluding a seed requires one of:

- documented run failure;
- wrong checkpoint/config;
- interrupted/incomplete run;
- corrupted score/eval artifact.

If excluding, report both all-seed and exclusion-sensitive stats in notes.

## Validation protocol policy

For every training run, record whether checkpoint selection used:

- ASVspoof2019 LA dev;
- ASVspoof2021 LA fast subset;
- validation loss only;
- external baseline procedure.

Do not mix these under the generic term `dev`.
