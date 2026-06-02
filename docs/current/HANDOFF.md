# Handoff: current system/research-framework state

Date: 2026-05-26
Branch: `paper-rework-audit`
Last milestone tag: `system-audit-complete-2026-05-26`

## Current verdict

The repo has a usable audited system layer for:

- evidence-table generation from machine-readable provenance;
- artifact/dependency audits;
- official ASVspoof LA scoring guardrails;
- release-gate reporting;
- audit-package generation;
- safe training/evaluation dry-runs;
- first controlled real wav2vec2 eval smoke run;
- first controlled real wav2vec2 training smoke run with loadable checkpoint;
- first bounded 1000-utterance wav2vec2 score reproduction audit;
- first MiMo frozen historical eval smoke and MiMo frozen training smoke;
- Wave 0/1 CodecFake+ CoSG feature extraction for MiMo and wav2vec2/XLSR caches;
- completed Wave 1 frozen-feature probes, score-fusion/error comparisons, and held-out-source sweep, summarized in `docs/current/RESEARCH_WAVE_1_RESULTS.md`;
- post-Wave-1 purpose reset: no MiMo revival as organizing goal; representation-transfer/failure-mode framing in `docs/current/RESEARCH_PURPOSE_RESET.md`;
- representation-transfer matrix design in `docs/current/REPRESENTATION_TRANSFER_MATRIX.md` and `docs/current/representation_transfer_matrix.yaml`;
- targeted CLAMTTS-vs-NS metadata/error diagnostic in `experiments/runs/wave1_feature_source_diagnostics_v1/report.md` and summarized in `docs/current/RESEARCH_WAVE_1_RESULTS.md`;
- Wave 1 negative/conditional note in `docs/current/RESEARCH_WAVE_1_NEGATIVE_NOTE.md`;
- Wave 2 interim closure in `docs/current/RESEARCH_WAVE_2_INTERIM_NOTE.md`;
- Wave 3 targeted trained-validation plan in `docs/current/wave3_training_validation_plan.yaml`;
- Wave 3A concrete XLS-R trained-reference spec in `docs/current/wave3a_xlsr_training_reference_spec.yaml`;
- CodecFake+ official split summary in `docs/current/CODECFAKE_OFFICIAL_SPLITS.md`.

A versioned research execution system is now specified and partially implemented:

- `docs/current/RESEARCH_FRAMEWORK_GUIDELINES.md`
- `docs/current/RESEARCH_SYSTEM_SPEC.md`
- `docs/current/RESEARCH_SYSTEM_MIGRATION_PLAN.md`
- `docs/current/examples/experiment_spec_v1_minimal.yaml`
- `mimodf experiment validate/resolve/init/inspect`
- `mimodf report index/aggregate/compare`
- optional run-layout v1 manifest updates for `mimodf eval run` and `mimodf train legacy-asvspoof`

It is **not** a turnkey full-reproduction package. The default full gate still fails because 9 historical artifact paths are missing. Historical scope policy is documented in `docs/current/HISTORICAL_REPRO_SCOPE.md`: use score-backed gaps only as partial/directional evidence unless exact artifacts are recovered or rerun as new controlled experiments. External dependency setup is documented in `docs/current/EXTERNAL_DEPENDENCY_SETUP.md`; ASVspoof data/protocol/key layout is documented in `docs/current/ASVSPOOF_DATA_LAYOUT.md`; public smoke commands are documented in `docs/current/CONTROLLED_SMOKE_TRANSCRIPT.md`; shareable status is summarized in `docs/current/SHAREABLE_REPO_STATUS.md`.

## Do not claim

- Do not claim MiMo adapter n=5 results are supported.
- Do not claim full experiment reproducibility.
- Do not use project `min t-DCF:` result files as official tDCF.
- Do not silently exclude seeds.
- Do not run full training/eval/GPU jobs without explicit approval; targeted Wave 3 training also needs concrete specs and planned log rows first.
- Do not mutate dirty historical clones to make gates pass.
- Do not rewrite historical artifacts to fit new schemas; index them with source type and reproducibility tier.
- Do not add heavy experiment platforms before the plain-file spec/manifest/index system fails.

## Source of truth

Primary machine-readable sources:

- `docs/current/main_table_provenance.yaml`
- `docs/current/official_tdcf_values.yaml`
- `docs/current/external_dependencies.yaml`
- `docs/current/artifact_gap_decisions.yaml`
- `docs/current/examples/experiment_spec_v1_minimal.yaml`
- `docs/current/EXPERIMENT_LOGGING_PROTOCOL.md`
- `docs/current/research_execution_log.jsonl`

Important status docs:

- `docs/current/SYSTEM_STATUS.md`
- `docs/current/RELEASE_CHECKLIST.md`
- `docs/current/TASK_BOARD.md`
- `docs/current/DECISION_LOG.md`
- `docs/current/RESEARCH_EXECUTION_LOG.md`
- `docs/current/RESEARCH_WAVE_2_INTERIM_NOTE.md`
- `docs/current/wave3_training_validation_plan.yaml`

## Gates

### Full reproducibility gate

```bash
python -m mimodf audit release-gate --strict
```

Expected current result: nonzero. Known blocker:

- `artifact_missing` — 9 missing historical artifact paths.

### System/tooling gate

```bash
python -m mimodf audit release-gate --system-profile --strict
```

Expected current result: zero in this workspace. This only means the system/tooling layer is coherent; known historical artifact gaps are downgraded to warnings from `artifact_gap_decisions.yaml`.

### Dependency override caveat

This workspace uses ignored local override:

- `docs/current/external_dependencies.local.yaml`

It points the scorer dependency to:

- `local_dependencies/SSL_Anti-spoofing-clean/`

Disable it when auditing the historical clone directly:

```bash
python -m mimodf audit release-gate --dependency-local-spec none
```

Expected current result: dependency dirty blocker returns for `SSL_Anti-spoofing/`.

## Core commands

```bash
python -m mimodf audit main-table
python -m mimodf audit check-artifacts --format markdown
python -m mimodf audit artifact-gaps --format markdown
python -m mimodf audit dependencies --format markdown
python -m mimodf audit dependencies --format json --hash-files
python -m mimodf audit release-gate --format markdown
python -m mimodf audit release-gate --system-profile --strict
python -m mimodf audit package --out /tmp/mimodf-audit-package
python -m mimodf experiment validate docs/current/examples/experiment_spec_v1_minimal.yaml
python -m mimodf experiment init docs/current/examples/experiment_spec_v1_minimal.yaml --seed 42 --root /tmp/mimodf-runs
python -m mimodf report index /tmp/mimodf-runs --provenance docs/current/main_table_provenance.yaml --out /tmp/mimodf-runs/index.jsonl
python -m mimodf report aggregate --index /tmp/mimodf-runs/index.jsonl
python -m mimodf report compare --index /tmp/mimodf-runs/index.jsonl --experiments wav2vec2_frozen mimo_frozen
```

Safe dry-runs:

```bash
python -m mimodf eval plan \
  --config configs/publish/wav2vec2_adapter.yaml \
  --checkpoint experiments/paper_final/wav2vec2_adapter_multiseed/seed_123/models/w2v2_adapter_s123_seed123/epoch_12_eer_5.42.pth \
  --eval-root . \
  --score-out /tmp/mimodf-eval-plan/scores_LA_eval.txt \
  --track LA \
  --scorer local_dependencies/SSL_Anti-spoofing-clean/evaluate_2021_LA.py \
  --strict

python -m mimodf train legacy-asvspoof \
  --config configs/publish/mimo_full.yaml \
  --out /tmp/mimodf-train-dry-run \
  --database-path /data/asvspoof \
  --protocols-path /data/protocols \
  --validation-protocol asvspoof2021_fast \
  --frontend mimo \
  --dry-run
```

## Current verification baseline

Latest verified baseline after research-framework aggregation/comparison:

```text
pytest -q: 151 passed, 9 skipped
conda run -n mimo-df pytest -q: 307 passed, 2 skipped
python -m compileall -q mimodf src train.py: ok
python -m mimodf experiment validate docs/current/examples/experiment_spec_v1_minimal.yaml: ok
python -m mimodf experiment init ... --seed 42: ok
python -m mimodf report index ... --provenance docs/current/main_table_provenance.yaml: ok; 26 records in smoke output
python -m mimodf report aggregate --index ...: ok
python -m mimodf report compare --index ... --experiments wav2vec2_frozen mimo_frozen: ok in warning mode
python -m mimodf report compare --index ... --experiments wav2vec2_frozen mimo_frozen --strict: expected exit 1 for missing historical protocol IDs
python -m mimodf audit release-gate --system-profile --strict: ok
git diff --check: ok
```

## Recommended next work

1. Decide whether to prioritize official CoRS extraction or custom CoSG diagnostic training first.
2. If CoSG diagnostic first, extend source-holdout count planning into actual DataLoader row-index splits.
3. Build or select the CodecFake training/scoring path required by `wave3a_xlsr_training_reference_spec.yaml`.
4. Add CodecFake score/prediction writing and per-source metrics.
5. Run XLS-R trained-reference jobs only after explicit approval.
6. If pausing or sharing: use this handoff and current shareable status.
