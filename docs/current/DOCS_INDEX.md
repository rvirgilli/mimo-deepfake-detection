# Docs index

Date: 2026-06-02
Status: public reader map

This directory keeps curated documentation and machine-readable summaries. Raw execution logs, stdout/stderr, PIDs, submitted PDFs, reviewer notes, and local run artifacts belong in ignored local storage such as `local_private/`.

## Read first

1. `README.md` — repo-level setup and commands.
2. `docs/README.md` — documentation map.
3. `docs/current/README.md` — current docs overview.
4. `docs/current/CODECFAKE_OFFICIAL_SPLITS.md` — CodecFake+ official/custom split framing.
5. `docs/current/WAVE3_REVISED_PLAN_2026_05_31.md` — current diagnostic research plan.
6. `docs/current/WAVE3_BATCH_SIZE_VALIDITY_2026_05_31.md` — batch-size protocol note.
7. `docs/current/RELEASE_CHECKLIST.md` — release/repro checklist.

## Machine-readable public summaries

- `main_table_provenance.yaml` — corrected table provenance.
- `official_tdcf_values.yaml` — official LA tDCF values and wrong-scale examples.
- `artifact_gap_decisions.yaml` — known historical artifact gaps.
- `external_dependencies.yaml` — external dependency pins and file hashes.
- `representation_transfer_matrix.yaml` — representation-transfer research matrix.
- `wave1_exploratory_validation_matrix.yaml` — Wave 1 validation design/summary.
- `media_transform_smoke_plan.yaml` — media-transform smoke plan/summary.
- `wave2_deepening_plan.yaml` — Wave 2 design/summary.
- `wave3_training_validation_plan.yaml` — targeted trained-validation plan.
- `wave3a_xlsr_training_reference_spec.yaml` — concrete CoSG XLS-R trained-reference spec.
- `wave3a_codecfake_cosg_source_holdout_plan_v3.json` — selected custom CoSG source-holdout/count plan.
- `codecfake_official_split_summary.yaml` — official CodecFake+ split summary.
- `codecfake_cors_audit.yaml` — local CoRS inventory summary.
- `examples/experiment_spec_v1_minimal.yaml` — example experiment spec.

## Research notes

Current/recent notes:

- `RESEARCH_PURPOSE_RESET.md`
- `RESEARCH_WAVE_1_CLOSEOUT.md`
- `RESEARCH_WAVE_2_INTERIM_NOTE.md`
- `REPRESENTATION_TRANSFER_MATRIX.md`
- `WAVE3_REVISED_PLAN_2026_05_31.md`
- `WAVE3_REEVALUATION_2026_05_31.md`
- `WAVE3_DEEP_RESEARCH_DIRECTIONAL_2026_05_31.md`
- `WAVE3_DEEP_RESEARCH_LANDSCAPE_2026_05_31.md`

Appendices/evidence notes:

- `CODECFAKE_OFFICIAL_SPLITS.md`
- `CODECFAKE_CORS_AUDIT.md`
- `HISTORICAL_REPRO_SCOPE.md`
- `CHECKPOINT_PROVENANCE_GAPS.md`
- `TDCF_RECONCILIATION.md`
- `MIMO_ADAPTER_DECISION.md`
- `MIMO_DRIFT_INVESTIGATION.md`
- `RESEARCH_SYSTEM_SPEC.md`
- `RESEARCH_FRAMEWORK_GUIDELINES.md`
- `RESEARCH_SYSTEM_MIGRATION_PLAN.md`
- `EVAL_EXECUTION_PLAN.md`
- `TRAINING_LOOP_HARVEST_PLAN.md`

If docs conflict, trust machine-readable summaries first, then this index, then prose notes. Raw local artifacts are not public documentation until summarized and sanitized.
