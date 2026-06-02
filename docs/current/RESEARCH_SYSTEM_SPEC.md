# Research system specification

Date: 2026-05-26
Status: implementation spec

## Purpose

Build a lean experiment system that supports controlled deepfake-detection research while preserving historical evidence.

The system must allow us to:

- train and evaluate old and new models under explicit protocols;
- add new frontends, backends, adaptation methods, datasets, scorers, and parameters;
- keep old runs readable when schemas/components evolve;
- generate scientific tables from machine-readable artifacts, not hand-edited summaries;
- distinguish historical evidence, exploratory runs, confirmatory runs, and reproduction checks.

## Non-goals

- Do not introduce a heavy experiment platform unless the local file-based system fails.
- Do not mutate historical artifacts to fit new schemas.
- Do not make old experiments appear more reproducible than their artifacts allow.
- Do not hide protocol leakage behind convenience defaults.
- Do not make a plugin framework more abstract than the current research needs.

## Design principle

A run is interpreted by the schema and component versions it was created with.

New code may add readers, normalizers, and migrations, but must not silently rewrite historical meaning.

## Versioned contracts

Every stored experiment artifact that can be read later must include a contract/version field.

Required contract families:

| Contract | Example value | Owner |
|---|---|---|
| Experiment spec | `experiment-spec/v1` | `mimodf.experiments.spec` |
| Run manifest | `run-manifest/v1` | `mimodf.experiments.manifest` |
| Component API | `frontend/v1`, `backend/v1` | `mimodf.components` |
| Score file | `asvspoof-score/v1` | `mimodf.scoring.write_scores` |
| Metrics file | `metrics/v1` | `mimodf.scoring.evaluate` |
| Aggregate report | `aggregate-report/v1` | `mimodf.report` |

Version changes are required when stored meaning changes. Additive optional fields do not require a new major schema if old artifacts remain unambiguous.

## Stable component identity

Components must be identified by stable IDs, not Python class names or file paths.

Examples:

```text
frontend:wav2vec2-xlsr-300m/v1
frontend:mimo-continuous-native50/v1
frontend:mimo-rvq-sum-25hz/v1
backend:aasist/v1
backend:mlp-pool/v1
adaptation:frozen/v1
adaptation:houlsby-adapter-last8/v1
adaptation:full-finetune/v1
optimizer:adam/v1
optimizer:adamw-param-groups/v1
dataset:asvspoof2019-la-train/v1
dataset:asvspoof2019-la-dev/v1
dataset:asvspoof2021-la-eval/v1
scorer:asvspoof2021-la-official/v1
scorer:asvspoof-df-eer/v1
```

If behavior changes in a way that affects outputs, create a new component version. Do not silently change `/v1` behavior.

## ExperimentSpec v1

An experiment spec is the user-facing, reviewable description of an experiment family or one run. It must be validated before training/evaluation.

### Required top-level fields

```yaml
schema_version: experiment-spec/v1
experiment_id: controlled_mimo_wav2vec2_v1
intent: confirmatory  # exploratory | confirmatory | reproduction | diagnostic
hypothesis: >
  MiMo adapters improve LA relative to frozen MiMo, but wav2vec2 remains stronger on DF.
owner: local
created: 2026-05-26

protocol:
  train_dataset: dataset:asvspoof2019-la-train/v1
  validation_dataset: dataset:asvspoof2019-la-dev/v1
  checkpoint_selection: dev_eer  # dev_loss | dev_eer | fixed_epoch | historical
  eval_datasets:
    - dataset:asvspoof2021-la-eval/v1
    - dataset:asvspoof2021-df-eval/v1
  scorers:
    la: scorer:asvspoof2021-la-official/v1
    df: scorer:asvspoof-df-eer/v1
  leakage_policy: no_eval_selection  # no_eval_selection | exploratory_eval_selection

seeds: [42, 123, 456]

model:
  frontend: frontend:mimo-continuous-native50/v1
  backend: backend:aasist/v1
  projection:
    type: linear
    output_dim: 192
  adaptation: adaptation:houlsby-adapter-last8/v1

training:
  max_epochs: 30
  batch_size: 8
  optimizer: optimizer:adam/v1
  learning_rate: 0.000034
  weight_decay: 0.0001
  rawboost:
    enabled: true
    algorithm: 6
  early_stop: null

evaluation:
  batch_size: 64
  max_items: null
  write_scores: true
  official_scoring: true

comparability:
  group_id: controlled_mimo_wav2vec2_v1
  allowed_comparisons:
    - same_protocol
    - same_seed_set
    - same_eval_dataset
  caveats:
    - mimo_eval_batch_size_is_protocol_fact

artifacts:
  output_root: experiments/runs
  publishable: false
```

### Rules

- `intent` is mandatory. Exploratory results cannot be promoted to confirmatory without a new spec or explicit decision.
- `checkpoint_selection` must be explicit.
- `leakage_policy` must be explicit. ASVspoof2021 eval-subset selection is allowed only as `exploratory_eval_selection` or historical reproduction.
- `evaluation.batch_size` is mandatory for every run; MiMo reports must expose it.
- `seeds` are declared before execution. Missing/failed seeds remain in the run index with status.
- New fields must be either optional with stable defaults or require a new schema version.

## ResolvedSpec

Every run stores a fully resolved copy of the spec.

The resolved spec expands shorthand and records derived facts:

```yaml
schema_version: experiment-spec/v1
resolved_at: 2026-05-26T00:00:00Z
spec_hash: sha256:...
component_versions:
  frontend: frontend:mimo-continuous-native50/v1
  backend: backend:aasist/v1
  adaptation: adaptation:houlsby-adapter-last8/v1
frontend_facts:
  sample_rate: 24000
  feature_dim: 1280
  frame_rate_hz: 50
  precision: bf16
  known_caveats:
    - bf16_flashattention_batch_size_sensitive
```

User specs are editable plans. Resolved specs are immutable run evidence.

## Component contracts

### Frontend contract

A frontend implementation must expose metadata and a feature extractor.

Required metadata:

```yaml
component_id: frontend:mimo-continuous-native50/v1
api_version: frontend/v1
sample_rate: 24000
feature_dim: 1280
frame_rate_hz: 50
supports_lengths: true
supports_training: true
known_caveats:
  - bf16_flashattention_batch_size_sensitive
```

Required behavior:

- accept batched waveforms and optional lengths;
- return frame features and feature lengths when available;
- declare precision/device constraints;
- declare whether eval batch size can affect outputs if known.

### Backend contract

Required metadata:

```yaml
component_id: backend:aasist/v1
api_version: backend/v1
input_dim: 192
outputs: binary_logits
```

Required behavior:

- accept projected features and optional lengths;
- return exactly one score/logit pair per utterance;
- avoid hidden data-dependent filtering.

### Adaptation contract

Required metadata:

```yaml
component_id: adaptation:houlsby-adapter-last8/v1
api_version: adaptation/v1
trainable_scope: last_8_layers_adapters
```

Required behavior:

- declare trainable parameter groups;
- declare frozen modules;
- expose parameter counts in manifests.

## Run layout v1

Every run directory must be immutable after completion except for adding derived reports under `reports/`.

```text
experiments/runs/<experiment_id>/<spec_hash>/seed_<seed>/
  resolved_spec.yaml
  manifest.json
  git_state.json
  environment.json
  command.txt
  logs/
    train.log
    eval.log
  metrics/
    train_metrics.jsonl
    validation_metrics.jsonl
  checkpoints/
    checkpoint_epoch_001.pth
    best.pth
    checkpoint_manifest.json
  eval/
    asvspoof2021-la/
      scores.txt
      score_manifest.json
      official_metrics.json
    asvspoof2021-df/
      scores.txt
      score_manifest.json
      metrics.json
  reports/
    run_summary.json
```

For eval-only historical reproduction runs, `training/` artifacts may be absent, but `resolved_spec.yaml`, `manifest.json`, score outputs, and source checkpoint references are still required.

## RunManifest v1

The manifest is the run ledger. It records status, paths, hashes, environment, and important protocol facts.

Required fields:

```json
{
  "schema_version": "run-manifest/v1",
  "run_id": "controlled_mimo_wav2vec2_v1/sha256.../seed_42",
  "experiment_id": "controlled_mimo_wav2vec2_v1",
  "spec_hash": "sha256:...",
  "seed": 42,
  "intent": "confirmatory",
  "status": "completed",
  "started_at": "...",
  "ended_at": "...",
  "git": {"commit": "...", "dirty": false},
  "environment": {"python": "...", "torch": "...", "cuda": "..."},
  "protocol": {"checkpoint_selection": "dev_eer", "leakage_policy": "no_eval_selection"},
  "model": {"frontend": "frontend:mimo-continuous-native50/v1", "backend": "backend:aasist/v1"},
  "evaluation": {"batch_size": 64},
  "artifacts": [],
  "metrics": {},
  "warnings": [],
  "failures": []
}
```

Allowed statuses:

```text
planned
running
completed
failed
interrupted
superseded
retired
```

Failed and interrupted runs are first-class records. They are not deleted from the index.

## Comparability rules

Reports must distinguish:

- comparable confirmatory rows;
- exploratory rows;
- historical reproduction rows;
- diagnostic runs;
- rows with protocol caveats.

A report may compare runs only when the comparison policy allows it.

Default `same_protocol` policy requires:

- same train dataset ID;
- same validation/checkpoint-selection ID;
- same eval dataset/scorer IDs;
- same seed set or explicit missing-seed policy;
- same score contract;
- same leakage policy;
- same eval batch-size policy for batch-sensitive frontends.

If a comparison violates these conditions, the report must mark it as exploratory or refuse in strict mode.

## Reproducibility tiers

Every run or historical row gets a tier.

| Tier | Meaning |
|---:|---|
| 0 | Metric text only; no score file. |
| 1 | Score file and metric output available. |
| 2 | Checkpoint/config plus score file available. |
| 3 | Resolved spec, manifest, env/git state, checkpoint, scores, and metrics available. |
| 4 | Re-run reproduced within declared tolerance. |

The tier is not a quality score; it is an evidence completeness label.

## Reporting contracts

Reports are derived. Source artifacts remain specs, manifests, score files, official metrics, and provenance YAML.

Planned commands:

```bash
python -m mimodf experiment validate specs/foo.yaml
python -m mimodf experiment resolve specs/foo.yaml --out /tmp/resolved.yaml
python -m mimodf experiment inspect experiments/runs/.../seed_42
python -m mimodf report index experiments/runs --out experiments/index/runs.jsonl
python -m mimodf report aggregate controlled_mimo_wav2vec2_v1
python -m mimodf report compare controlled_mimo_wav2vec2_v1 --strict
```

## Optuna contract

Optuna is a runner over validated experiment specs, not a separate research path.

Search spaces are declared in spec files:

```yaml
search_space:
  learning_rate:
    type: loguniform
    low: 1.0e-6
    high: 1.0e-4
  projection.output_dim:
    type: categorical
    choices: [192, 384, 512]
  adapter.bottleneck_dim:
    type: categorical
    choices: [32, 64, 128]
objective:
  metric: validation.dev_eer
  direction: minimize
  max_epochs: 10
  leakage_policy: no_eval_selection
```

Rules:

- Optuna trials produce normal run directories and manifests.
- Trial selection metric must be declared before execution.
- ASVspoof2021 eval-set selection is forbidden for confirmatory studies.
- Study storage should be local SQLite under `experiments/optuna/`, with trial artifacts linked by run ID.

## Implementation modules

Target package layout:

```text
mimodf/experiments/
  __init__.py
  spec.py              # ExperimentSpec v1 parsing/validation/hash
  manifest.py          # RunManifest v1 read/write/validation
  layout.py            # immutable run paths
  compatibility.py     # comparison policies
  migration.py         # old artifact readers/normalizers

mimodf/components/
  __init__.py
  registry.py          # stable component IDs
  frontends.py         # metadata/factory seam
  backends.py
  adaptation.py

mimodf/report/
  __init__.py
  index.py             # run index builder
  aggregate.py         # seed aggregation/table rows
  compare.py           # strict/exploratory comparison checks
```

Existing modules such as `mimodf.evaluation.run`, `mimodf.training.run`, and `mimodf.scoring.*` should be adapted to consume/respect these contracts gradually.

## Acceptance criteria

Phase 1 is complete when:

- specs validate without importing Torch;
- stable spec hashes are deterministic;
- run layout creation is tested;
- manifests roundtrip and reject missing required protocol facts;
- legacy current outputs can be indexed as historical/reproduction records without moving them;
- default `pytest -q` remains lightweight.

Phase 2 is complete when:

- wav2vec2 and MiMo frontends have registered component metadata;
- AASIST backend and frozen/adapter/full adaptation strategies have registered IDs;
- `eval run` and `train legacy-asvspoof` write resolved specs/manifests in run-layout v1.

Phase 3 is complete when:

- report index/aggregate can produce a seed table from run manifests and score metrics;
- strict comparison refuses mismatched protocol/seed/batch-size cases;
- historical audited table rows can be represented with appropriate reproducibility tiers.

## Guardrails

- No silent schema inference for protocol-critical fields.
- No implicit seed exclusions.
- No overwriting completed run directories unless `--new-run-id` is used.
- No official LA tDCF unless generated by official scorer output.
- No MiMo fresh result without eval batch size recorded.
- No confirmatory claim from an exploratory/leaky spec.
