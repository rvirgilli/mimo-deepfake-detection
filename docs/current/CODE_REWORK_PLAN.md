# Code rework plan

Purpose: build a lean, reliable, debuggable, maintainable, publishable codebase from the current research repo without losing evidence.

## Strategy

Use a **rebuild-from-harvest** strategy:

1. freeze the current assessment/evidence state;
2. create a new small package alongside old `src/`;
3. port only final-paper functionality;
4. build audit/scoring/table tooling before touching training;
5. retire old paths only after parity checks pass.

Do not do a big-bang rewrite. Do not delete historical artifacts. Do not run long training as part of the rework unless explicitly approved later.

## Progress

- Phase 0-1 started: new `mimodf` package skeleton exists.
- `docs/current/main_table_provenance.yaml` encodes the corrected main-table rows.
- `python -m mimodf audit main-table` renders the table and matches `CORRECTED_MAIN_TABLE_DRAFT.md` recommended rows.
- Unit tests cover sample-standard-deviation aggregation, tDCF seed-set mismatch guards, and exploratory MiMo adapter labeling.
- `python -m mimodf audit check-artifacts` now checks declared artifact paths, hashes present files, and detects current missing checkpoint gaps from the structured provenance.
- `python -m mimodf score official-la` wraps official ASVspoof LA scoring, emits normalized JSON, and refuses project wrong-scale `min t-DCF` files as official output.
- `mimodf.config` validates explicit protocol/scorer/optimizer facts; focused tests cover implicit validation failure and AdamW/`encoder_lr: null` rejection.
- Plain `pytest -q` now works in the default environment: optional Torch/SciPy legacy tests skip when dependencies are absent, while `mimodf` tests run.
- All planned `configs/publish/*.yaml` recipes exist and are covered by portability/schema tests.
- `mimodf.scoring.write_scores` defines the model-independent ASVspoof score-file contract with deterministic ordering, duplicate-id rejection, and roundtrip tests.
- `mimodf.scoring.evaluate` adds framework-agnostic eval batching: one score per utterance, duplicate-id checks across batches, deterministic score-file output, and fake-predictor tests.
- `mimodf.scoring.torch_eval` adapts Torch models to the same score contract; covered by Torch tests in the `mimo-df` env while lightweight default tests skip when Torch is absent.
- Training-loop migration is planned in `TRAINING_LOOP_HARVEST_PLAN.md`; foundation modules cover deterministic seeding, top-k checkpoint retention, JSON training manifests with artifact hashes, a minimal Torch `train_one_run` loop, a `TrainingComponents` seam, explicit ASVspoof data-loader planning, and lazy legacy frontend/model factories.
- `python -m mimodf audit package --out <dir>` packages generated audit outputs for review/release, including official tDCF summaries from `official_tdcf_values.yaml`.

## Non-negotiable constraints

- Existing assessment docs remain source of truth until generated tooling replaces them.
- Existing score/result artifacts must remain readable.
- No unsupported paper claim can be reintroduced.
- No GPU-heavy training during initial code rework.
- New code must make wrong tDCF/seed aggregation hard to do.
- Every new `mimodf` behavior must ship with focused tests and CLI smoke verification before commit.

## Target architecture

```text
mimodf/
  __init__.py
  config.py
  cli.py
  data/
    asvspoof.py
  augment/
    rawboost.py
  frontends/
    base.py
    wav2vec2.py
    mimo.py
  backends/
    aasist.py
  training/
    loop.py
    checkpoint.py
  scoring/
    write_scores.py
    official.py
  provenance.py
  tables/
    main_table.py
configs/
  publish/
    wav2vec2_frozen.yaml
    wav2vec2_adapter.yaml
    wav2vec2_full_external.yaml
    mimo_frozen.yaml
    mimo_adapter_exploratory.yaml
    mimo_full.yaml
tests/
  test_config_validation.py
  test_artifact_checks.py
  test_tdcf_aggregation.py
  test_main_table_generation.py
  test_scoring_wrapper.py
```

CLI target:

```bash
python -m mimodf audit check-artifacts docs/current/main_table_provenance.yaml
python -m mimodf audit main-table docs/current/main_table_provenance.yaml
python -m mimodf score official-la scores_LA_eval.txt --eval-root /path/to/ASVspoof2021_LA_eval
python -m mimodf train --config configs/publish/mimo_adapter_exploratory.yaml
```

## Phase 0 — freeze and guardrails

Goal: preserve the current state before code movement.

Tasks:

- Tag or branch the current assessment state.
- Add a short `docs/current/CODEBASE_AUDIT.md` and this plan.
- Decide new package name (`mimodf` recommended).
- Add rule: old `src/`, `train.py`, `evaluate.py`, `optuna_train.py` are legacy during migration.

Exit criteria:

- Worktree clean.
- Assessment docs point to code rework plan.
- No source behavior changed.

## Phase 1 — structured provenance and table generation

Goal: generate the corrected main table from data, not prose.

Create:

```text
docs/current/main_table_provenance.yaml
mimodf/provenance.py
mimodf/tables/main_table.py
```

The provenance YAML should list rows/seeds with:

- model;
- strategy;
- seed/source;
- track;
- EER;
- tDCF where available;
- score file;
- result file;
- config;
- checkpoint;
- status;
- notes.

Tasks:

- Encode current `RESULTS_PROVENANCE.md` main-table rows in YAML.
- Implement loader with schema validation.
- Generate Markdown matching `CORRECTED_MAIN_TABLE_DRAFT.md`.
- Fail if a row mixes EER and tDCF seed sets without explicit footnote.

Tests:

- aggregation uses sample std for EER;
- tDCF values match `TDCF_RECONCILIATION.md`;
- MiMo adapter n=2 is marked exploratory;
- missing artifact status does not silently become verified.

Exit criteria:

- Generated table reproduces corrected assessment table.
- No training/eval required.

## Phase 2 — artifact checker

Goal: make provenance gaps machine-checkable.

Create:

```text
mimodf/provenance.py
mimodf/audit/artifacts.py
```

Tasks:

- Check file existence for score/result/config/checkpoint paths.
- Compute SHA-256 for files that exist.
- Emit JSON and Markdown report.
- Reproduce `CHECKPOINT_PROVENANCE_GAPS.md` mechanically where possible.

Tests:

- missing checkpoint returns `missing`, not error;
- result-cited checkpoint mismatch detected;
- external/published rows are allowed only with explicit `verified_external`.

Exit criteria:

- One command prints current gap table.

## Phase 3 — official scoring wrapper

Goal: eliminate wrong-scale tDCF ambiguity.

Create:

```text
mimodf/scoring/official.py
mimodf/scoring/parse.py
```

Tasks:

- Wrap `SSL_Anti-spoofing/evaluate_2021_LA.py` or reimplement only if legally/technically safe.
- Normalize output to JSON:
  - `eer_percent`;
  - `min_tdcf`;
  - scorer command;
  - scorer source path;
  - score file hash.
- Parse existing score/result files.
- Add hard guard against project wrong-scale tDCF files being used as official values.

Tests:

- known wav2vec2 adapter seed42: project tDCF `0.0073` is excluded from official table use;
- official tDCF `0.2707` accepted;
- aggregation over four vs five seeds differs and is detected.

Exit criteria:

- `TDCF_RECONCILIATION.md` can be regenerated or verified.

## Phase 4 — config schema and protocol labels

Goal: make protocol claims explicit and validated.

Create:

```text
mimodf/config.py
mimodf/protocol.py
```

Config must include:

- train set;
- validation/checkpoint-selection set;
- eval set;
- optimizer;
- loss weights for train and validation;
- RawBoost settings;
- frontend strategy;
- trainable parameter policy;
- seed;
- scorer.

Tasks:

- Add typed dataclasses or Pydantic models. Prefer stdlib dataclasses + explicit validation unless Pydantic is already justified.
- Create final publish configs under `configs/publish/`.
- Validate no hardcoded `<home>` in tracked publish configs.

Tests:

- config with implicit validation set fails;
- config claiming AdamW with `encoder_lr: null` fails;
- config missing scorer fails.

Exit criteria:

- Protocol matrix can be represented by config/provenance data.

## Phase 5 — minimal scoring/eval path

Goal: replace stale `evaluate.py` for future use.

Create:

```text
mimodf/scoring/write_scores.py
```

Tasks:

- Load model from typed config + checkpoint.
- Write ASVspoof-compatible score files.
- Do not compute table metrics directly except by official scorer wrapper.
- Support only final frontends initially: wav2vec2 and MiMo continuous/adapter/full/frozen.

Tests:

- smoke test with mock frontend/backend writes valid score format;
- checkpoint load strictness configurable and logged;
- score file order stable.

Exit criteria:

- `evaluate.py` can be marked legacy.

## Phase 6 — training loop harvest

Goal: one future training path, not multiple script histories.

Tasks:

- Port minimal training loop from `train.py` into `mimodf/training/loop.py`.
- Make validation protocol explicit, not file-existence driven.
- Add deterministic DataLoader worker seeding.
- Save manifest before/after training with config/checkpoint hashes.
- Save best checkpoint with a stable manifest record.

Tests:

- mock one-epoch training path;
- checkpoint selection by configured metric;
- DataLoader seed function exists and is used;
- manifest includes protocol and optimizer.

Exit criteria:

- Future controlled MiMo adapter rerun could be launched through new path.

## Phase 7 — frontend/backend pruning

Goal: keep only final claim paths in the public core.

Keep initially:

- wav2vec2 frontend needed for evidence;
- MiMo continuous encoder path;
- MiMo adapter/full/frozen strategies actually used;
- AASIST backend and projection.

Defer/archive:

- HuBERT;
- EnCodec;
- LoRA;
- partial/gradual unfreezing;
- RVQ/layer-select/native 50Hz unless a later extension uses them.

Exit criteria:

- Public package has a small API surface.
- Legacy code remains accessible but noncanonical.

## Phase 8 — packaging and release

Goal: publishable repo.

Tasks:

- Fix `pyproject.toml` package target and URLs.
- Add dependency table for Python 3.10 vs 3.12 paths.
- Document external dependencies without vendoring giant clones.
- Add CI for non-GPU tests.
- Add `README_REPRODUCE.md` with exact commands for table generation.
- Add `DATA.md` for expected ASVspoof layout.

Exit criteria:

- Fresh clone can run tests and generate corrected table from provenance metadata without local model weights.

## Recommended immediate implementation order

Start with Phases 0-3 only.

Why:

- They improve reliability immediately.
- They do not risk model behavior.
- They directly support paper/review reproducibility.
- They reveal exactly what source code is still needed.

Do **not** start by rewriting training or MiMo internals. That is high-risk and gives no immediate paper-assessment benefit.

## Decisions needed from user before implementation

1. New package name: `mimodf` okay?
2. Keep old `src/` during migration? Recommended yes.
3. Do we want generated docs committed, or generated-on-demand only?
4. Is a future public repo expected to include checkpoints, score files, or only metadata + scripts?
5. Should the first implementation target table/provenance tooling only? Recommended yes.
