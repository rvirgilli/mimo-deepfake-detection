# Current docs

This directory contains curated documentation and machine-readable summaries.

Useful entry points:

- `CODECFAKE_OFFICIAL_SPLITS.md`
- `CODECFAKE_CORS_AUDIT.md`
- `WAVE3_REVISED_PLAN_2026_05_31.md`
- `WAVE3_BATCH_SIZE_VALIDITY_2026_05_31.md`
- `wave3_training_validation_plan.yaml`
- `wave3a_codecfake_cosg_source_holdout_plan_v3.json`
- `wave3a_xlsr_training_reference_spec.yaml`

Useful checks:

```bash
python -m ruff check mimodf tests
python -m ruff format --check mimodf tests
pytest -q
python -m mimodf research validate-matrix docs/current/representation_transfer_matrix.yaml
```
