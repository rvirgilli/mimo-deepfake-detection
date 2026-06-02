# Research system migration plan

Date: 2026-05-26
Status: implementation plan

## Goal

Move from the current audited tooling plus historical artifact ledger to a versioned experiment system without breaking old evidence.

The migration must preserve three truths:

1. historical artifacts remain historical artifacts;
2. new runs follow the versioned spec/manifest/run-layout contracts;
3. reports can read both old and new evidence while labeling comparability and reproducibility honestly.

## Current state

Already available:

- corrected table provenance in `docs/current/main_table_provenance.yaml`;
- official LA tDCF provenance in `docs/current/official_tdcf_values.yaml`;
- artifact checks and known gap decisions;
- dependency checks and release gates;
- score-file writer/reader contract;
- official LA scoring wrapper;
- score comparison CLI;
- controlled `eval plan` and `eval run`;
- controlled `train legacy-asvspoof` dry-run and batch-limited real smoke path;
- training manifests/checkpointing primitives;
- MiMo batch-size sensitivity documented.

Main gaps:

- no first-class experiment spec schema;
- no stable component registry;
- no uniform immutable run layout;
- no run index over both historical and new results;
- Optuna remains a historical script path rather than spec-driven trials.

## Migration strategy

Do not move or edit historical experiment directories.

Instead:

1. add versioned readers for new run-layout v1;
2. add legacy normalizers that read current historical evidence and emit in-memory/index records;
3. write new experiments only under the new layout;
4. generate reports from a mixed index with explicit source type and reproducibility tier.

## Source categories

| Category | Example | Migration treatment |
|---|---|---|
| Historical metric/provenance row | `main_table_provenance.yaml` | Keep as source; index as `source_type: historical_provenance`. |
| Historical score file | `experiments/paper_final/.../results_LA_eval.txt` | Keep in place; index path/hash/tier. |
| Historical checkpoint/config | `experiments/paper_final/.../models/...` | Keep in place; index when present; record gaps. |
| New eval-only reproduction | `/tmp` or future run dir from `mimodf eval run` | Write into new run-layout v1. |
| New training run | future controlled experiments | Write into new run-layout v1. |
| Optuna trial | `experiments/optuna/...` | Do not rewrite old trials; future trials use new run layout plus SQLite study. |

## Phased implementation

### Phase 0 — freeze decisions

Status: complete. `RESEARCH_FRAMEWORK_GUIDELINES.md`, `RESEARCH_SYSTEM_SPEC.md`, and this migration plan define the lean contract-first approach.

Actions:

- accept `RESEARCH_SYSTEM_SPEC.md` as the target contract;
- record decision in `DECISION_LOG.md`;
- update `TASK_BOARD.md` so implementation starts from schema/layout, not model changes.

Exit criteria:

- docs committed;
- no code behavior changed.

### Phase 1 — schema, hash, manifest, layout

Status: first implementation slice complete for validation/hash/resolve, manifest roundtrip, run layout helper, minimal component registry, and CLI validation/initialization/inspection. Eval integration has started; train integration remains Phase 3.

Add modules:

```text
mimodf/experiments/spec.py
mimodf/experiments/manifest.py
mimodf/experiments/layout.py
mimodf/experiments/migration.py
```

Add CLI:

```bash
python -m mimodf experiment validate <spec.yaml>
python -m mimodf experiment resolve <spec.yaml> --out <resolved.yaml>
python -m mimodf experiment inspect <run_dir>
```

Implementation details:

- parse YAML into typed dataclasses;
- reject missing protocol-critical fields;
- compute deterministic spec hash from canonical JSON;
- create run paths without touching historical directories;
- write/read `manifest.json` with schema version and required fields;
- keep Torch imports out of validation and layout code.

Tests:

- valid minimal spec passes;
- missing checkpoint-selection fails;
- missing eval batch size fails;
- MiMo spec without explicit eval batch size fails;
- hash is deterministic across YAML key order;
- manifest roundtrips;
- completed run directory cannot be overwritten accidentally.

Exit criteria:

- `pytest -q` passes in lightweight environment;
- CLI validation works on an example spec under `configs/experiments/` or `docs/current/examples/`.

### Phase 2 — component registry metadata

Status: first metadata registry complete in `mimodf/components/registry.py`; factory migration remains deferred until a concrete component addition needs it.

Add modules:

```text
mimodf/components/registry.py
mimodf/components/frontends.py
mimodf/components/backends.py
mimodf/components/adaptation.py
```

Register initial components:

```text
frontend:wav2vec2-xlsr-300m/v1
frontend:wav2vec2-xlsr-300m-adapter/v1  # if adapter currently alters frontend construction
frontend:mimo-continuous-native50/v1
backend:aasist/v1
adaptation:frozen/v1
adaptation:houlsby-adapter-last8/v1
adaptation:full-finetune/v1
optimizer:adam/v1
optimizer:adamw-param-groups/v1
dataset:asvspoof2019-la-train/v1
dataset:asvspoof2019-la-dev/v1
dataset:asvspoof2021-la-eval/v1
dataset:asvspoof2021-df-eval/v1
scorer:asvspoof2021-la-official/v1
scorer:asvspoof-df-eer/v1
```

Implementation details:

- registry contains metadata only at first;
- factories can remain in current legacy modules until Phase 3;
- component metadata should include caveats like MiMo batch sensitivity.

Tests:

- unknown component ID fails validation;
- registered component metadata roundtrips;
- MiMo component exposes `bf16_flashattention_batch_size_sensitive` caveat.

Exit criteria:

- specs validate component IDs against the registry;
- no model loading required.

### Phase 3 — write new run-layout artifacts from existing CLIs

Status: implemented for `mimodf eval run` and `mimodf train legacy-asvspoof` via optional `--experiment-spec`, `--run-seed`, and `--run-root`; this writes/updates run-layout v1 resolved spec and manifest without changing model behavior.

Adapt:

```text
mimodf/evaluation/run.py
mimodf/training/run.py
mimodf/training/manifest.py
```

Behavior changes:

- `eval run` accepts either legacy flags or `--experiment-spec`;
- when given a spec, it writes run-layout v1;
- legacy flag mode can continue but should write a compatibility manifest if `--run-dir` is provided;
- `train legacy-asvspoof` writes resolved spec and run-manifest v1 alongside existing training manifest fields.

Rules:

- do not change model outputs;
- do not change historical score paths;
- preserve existing CLI behavior where possible;
- explicit overwrite guards remain.

Tests:

- fake eval writes `resolved_spec.yaml`, `manifest.json`, scores, and score manifest;
- fake train writes run manifest and checkpoint manifest;
- failed fake run records `status: failed` and error info;
- MiMo eval records batch size and caveat.

Exit criteria:

- current smoke-test commands still work;
- new spec-driven dry-run path works without GPU.

### Phase 4 — index and aggregate reports

Status: implemented for index, aggregate, and strict comparison. `mimodf report index` reads run-layout v1 manifests and optional `main_table_provenance.yaml` historical records into `run-index-record/v1` JSONL/Markdown. `mimodf report aggregate` summarizes numeric metrics. `mimodf report compare --strict` refuses unsafe comparisons such as missing protocol IDs, seed-set mismatches, exploratory intent, or missing eval batch-size policy for batch-sensitive records.

Add modules:

```text
mimodf/report/index.py
mimodf/report/aggregate.py
mimodf/report/compare.py
mimodf/experiments/compatibility.py
```

Add CLI:

```bash
python -m mimodf report index experiments/runs --out experiments/index/runs.jsonl
python -m mimodf report aggregate <experiment_id>
python -m mimodf report compare <group_id> --strict
```

Index sources:

- new run-layout v1 directories;
- `main_table_provenance.yaml` historical rows;
- official tDCF provenance;
- artifact gap decisions.

Output fields:

```json
{
  "record_schema": "run-index-record/v1",
  "source_type": "new_run|historical_provenance|historical_score",
  "experiment_id": "...",
  "run_id": "...",
  "seed": 42,
  "status": "completed",
  "intent": "confirmatory|exploratory|reproduction|diagnostic|historical",
  "reproducibility_tier": 1,
  "component_ids": {},
  "protocol_ids": {},
  "metrics": {},
  "artifact_paths": [],
  "warnings": []
}
```

Tests:

- indexer reads a fake new run;
- indexer reads a fake historical provenance row;
- aggregate computes mean/sample std and seed counts;
- strict compare refuses mismatched seed sets;
- strict compare refuses MiMo rows with mismatched eval batch-size policy;
- exploratory compare emits warnings instead of failing.

Exit criteria:

- current main-table values can be represented as historical index records without losing caveats;
- new runs can be aggregated from manifests and metrics.

### Phase 5 — Optuna under the spec system

Add:

```text
mimodf/optimization/optuna_runner.py
```

Behavior:

- study reads a validated spec with `search_space`;
- each trial resolves to a normal run spec and run directory;
- selected objective is declared before launch;
- study SQLite lives under `experiments/optuna/`;
- trial artifacts are normal run-layout records.

Rules:

- no ASVspoof2021 eval-set objective for confirmatory studies;
- trial failures are indexed;
- best trial selection is report-generated, not manually copied.

Exit criteria:

- fake objective test exercises trial manifest writing;
- no long GPU work in default tests.

### Phase 6 — research matrix execution

Only after Phases 1-4 are stable.

Candidate first matrix:

```text
wav2vec2 frozen
wav2vec2 adapter
MiMo frozen batch64
MiMo adapter batch64
```

Minimum rules:

- same seed set declared before execution;
- same train/validation/checkpoint protocol;
- same eval datasets/scorers;
- official LA scoring;
- MiMo eval batch size pinned and reported;
- failures indexed, not hidden.

## Mapping current artifacts to new records

### Corrected main table

Source:

```text
docs/current/main_table_provenance.yaml
```

New representation:

- `source_type: historical_provenance`;
- `intent: historical`;
- reproducibility tier based on available artifacts;
- warnings copied from provenance/caveat fields;
- no claim of run-layout v1 compliance.

### Official tDCF values

Source:

```text
docs/current/official_tdcf_values.yaml
```

New representation:

- linked metric records under scorer ID `scorer:asvspoof2021-la-official/v1`;
- reject project wrong-scale tDCF outputs as official metrics;
- preserve seed-set mismatch warnings.

### Current `mimodf eval run` outputs

Source:

```text
/tmp/mimodf-* or future explicit output dirs
```

New representation:

- if produced before run-layout v1, index as `source_type: reproduction_legacy_output` only when manifest/scores exist;
- future outputs should use run-layout v1.

### Current training smoke outputs

Source examples:

```text
/tmp/mimodf-real-train-smoke
/tmp/mimodf-real-train-mimo-frozen-smoke
```

New representation:

- diagnostic/reproduction records if still present;
- not paper evidence;
- future training runs should be run-layout v1.

### Historical `experiments/paper_final/*`

Treatment:

- do not move;
- do not rename;
- do not fill missing files with generated placeholders;
- index only paths/hashes/status/tier;
- keep artifact gaps in `artifact_gap_decisions.yaml`.

## Backward compatibility policy

- Old artifacts remain valid as historical evidence even when they fail new schema requirements.
- New schema strictness applies to new runs, not retroactively to old runs.
- Old rows may be compared only under historical/exploratory policies unless enough protocol fields are known.
- If an old field name maps to a new field, the mapping must be implemented in a normalizer and tested.
- If old meaning is ambiguous, store `unknown` plus warning rather than guessing.

## Data migration policy

There is no bulk data migration.

Allowed:

- generate indexes/reports from old artifacts;
- write sidecar generated reports under ignored or derived-output directories;
- write new run-layout artifacts for new executions.

Forbidden:

- editing historical score files;
- moving historical checkpoints;
- rewriting old configs to make them pass new validators;
- deleting interrupted/failed runs;
- changing metric values to fit new report formats.

## Risk register

| Risk | Mitigation |
|---|---|
| Schema becomes too large before research resumes. | Phase 1 only implements minimal required fields. |
| Component registry becomes abstract plugin sprawl. | Metadata-first; factories migrate only when needed. |
| Old evidence gets misrepresented as new reproducible runs. | `source_type` and reproducibility tiers are mandatory. |
| MiMo batch-size caveat is forgotten. | Component metadata and strict comparison policy enforce it. |
| Optuna reintroduces ad hoc protocol leakage. | Search space/objective must live in validated spec. |
| Reports compare mismatched protocols silently. | Strict compare refuses; exploratory compare warns. |

## First implementation slice

Implement only:

1. `ExperimentSpec` v1 validation/hash;
2. `RunManifest` v1 read/write;
3. `RunLayout` path creation/overwrite guard;
4. minimal component registry metadata;
5. CLI `experiment validate` and `experiment inspect`;
6. tests.

Do not yet modify training/eval behavior except to share validation helpers if harmless.

## Completion gate for migration foundation

Run before claiming Phase 1 complete:

```bash
pytest -q
python -m compileall -q mimodf src train.py
python -m mimodf experiment validate docs/current/examples/experiment_spec_v1_minimal.yaml
python -m mimodf audit release-gate --system-profile --strict
git diff --check
```

In the `mimo-df` environment, also run:

```bash
conda run -n mimo-df pytest -q
```
