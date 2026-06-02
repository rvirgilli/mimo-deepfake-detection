# Shareable repo status

Date: 2026-05-27
Status: current collaborator/reviewer landing page

This repo is shareable as an audited research workspace and future-experiment framework.

It is not shareable as a turnkey reproduction of the earlier-manuscript experiment set.

## What this repo is now

The project studies which pretrained audio representations transfer for audio deepfake detection under distribution shift.

MiMo is no longer the central goal. It is one candidate frontend among others.

Current research conclusion:

- Exploratory Wave 1 is closed, including WavLM/log-mel/media-transform completion runs whose immutable IDs may have `wave2-*` prefixes.
- wav2vec2/XLSR transferred better than MiMo/WavLM/log-mel on most held-out sources.
- MiMo and WavLM have source-conditional signal, especially CLAMTTS.
- Wave 1 used frozen feature caches and sklearn linear probes, not full model training.
- Wave 2 found the CLAMTTS mechanism source-local: the low-energy/silence/RVQ-entropy hypothesis did not transfer to NS2/NS3.
- No broad MiMo training, Optuna, or MiMo-superiority claim is justified.
- Wave 3 is now planned as targeted trained validation: expensive training is allowed only for predeclared XLS-R/source/media/CoRS hypotheses with explicit specs and logs.
- Official split correction: CoRS has official train/validation/Eval CoRS speaker splits; local CoSG labels are evaluation/custom diagnostic source-holdout data, not an official training split.

Read:

```text
docs/current/RESEARCH_WAVE_1_CLOSEOUT.md
docs/current/RESEARCH_WAVE_1_NEGATIVE_NOTE.md
docs/current/wave1_exploratory_validation_matrix.yaml
docs/current/wave2_deepening_plan.yaml
docs/current/RESEARCH_WAVE_2_INTERIM_NOTE.md
docs/current/wave3_training_validation_plan.yaml
docs/current/wave3a_xlsr_training_reference_spec.yaml
docs/current/CODECFAKE_OFFICIAL_SPLITS.md
```

## What is trustworthy

Trust these as current source-of-truth layers:

1. machine-readable ledgers/specs;
2. generated or checked summaries;
3. current state docs;
4. appendices/historical notes.

Start here:

```text
docs/current/DOCS_INDEX.md
```

Central experiment ledger:

```text
docs/current/research_execution_log.jsonl
```

Check it:

```bash
python -m mimodf log validate --strict
python -m mimodf log summary
```

Latest checked state:

```text
research log rows: 157
planned runs: none
validation: pass, including --strict
```

## What works

Lightweight/system layer:

- package/import/test path;
- provenance rendering;
- artifact/dependency audits;
- release gate with explicit system profile;
- experiment spec validation/init/inspect;
- run indexing/aggregation/comparison;
- ASVspoof eval planning;
- controlled eval/training smoke command paths;
- feature extraction/probe/fusion/diagnostic CLIs;
- research execution log validation/summary.

Key command:

```bash
python -m mimodf audit release-gate --system-profile --strict
```

Latest result: pass.

## What is intentionally partial

Full historical reproducibility is not claimed.

The strict full gate still fails because 9 historical artifact paths are missing:

```bash
python -m mimodf audit release-gate --strict
```

Latest result: fail on `artifact_missing`.

Policy:

```text
docs/current/HISTORICAL_REPRO_SCOPE.md
```

Meaning:

- historical score files can guide research and support qualified audit context;
- missing-checkpoint rows are not full reproducibility evidence;
- if a historical metric becomes a requirement, recover the exact artifact or rerun as a new controlled experiment;
- a new rerun is not the missing historical run.

## Setup docs

External dependencies and weights:

```text
docs/current/EXTERNAL_DEPENDENCY_SETUP.md
```

ASVspoof data/protocol/key layout:

```text
docs/current/ASVSPOOF_DATA_LAYOUT.md
```

Public smoke command transcript:

```text
docs/current/CONTROLLED_SMOKE_TRANSCRIPT.md
```

Release checklist:

```text
docs/current/RELEASE_CHECKLIST.md
```

## Audit package

The current audit package command succeeds:

```bash
python -m mimodf audit package --out /tmp/mimodf-shareable-audit-package
```

Latest artifact status counts:

| Status | Count |
|---|---:|
| present | 132 |
| missing | 9 |
| declared_absent | 9 |

The missing/declared-absent counts are expected under the historical scope policy.

## Do not claim

Do not claim:

- full earlier-manuscript reproducibility;
- MiMo superiority;
- MiMo adapter n=5 evidence;
- random row split metrics as main evidence;
- exact MiMo historical reproduction without batch-size caveats;
- full LA/DF reruns through the new CLI unless explicitly run and logged.

## Safe first commands for a new reader

```bash
python -m ruff check .
python -m ruff format --check .
pytest -q
python -m mimodf log validate --strict
python -m mimodf log summary
python -m mimodf audit release-gate --system-profile --strict
python -m mimodf experiment validate docs/current/examples/experiment_spec_v1_minimal.yaml
```

Then read:

```text
docs/current/DOCS_INDEX.md
docs/current/RESEARCH_WAVE_1_NEGATIVE_NOTE.md
docs/current/HISTORICAL_REPRO_SCOPE.md
docs/current/RELEASE_CHECKLIST.md
```

## Current next choices

After this shareable checkpoint, useful next work is:

1. plan CoRS extraction/storage after `docs/current/CODECFAKE_CORS_AUDIT.md` found parts+labels present but no extracted audio;
2. build the CodecFake training/scoring path required by `wave3a_xlsr_training_reference_spec.yaml`;
3. add model/probe/backend persistence for clean/transformed score comparison;
4. create planned log rows for the first bounded Wave 3A training smoke;
5. run targeted XLS-R trained validation only after explicit approval.
