# Research framework guidelines

Date: 2026-05-26
Status: active management guidance

## Goal

Turn this repository into a lean, expandable, scientifically auditable research system for audio deepfake detection.

The system should let us:

- rerun historical evaluations where artifacts exist;
- run new controlled experiments without losing provenance;
- add frontends, backends, adaptation methods, datasets, scorers, and Optuna search spaces;
- compare old and new results honestly;
- keep old data/results readable after schemas evolve;
- generate paper/review outputs from machine-readable evidence.

## Philosophy

Build contracts, not platforms.

Prefer:

- plain YAML/JSON/JSONL files;
- typed validation;
- stable component IDs;
- immutable run directories;
- generated reports;
- small modules with clear ownership;
- explicit protocol fields;
- tests around scientific guardrails;
- a tracked execution ledger for every research-producing command.

Avoid:

- MLflow/W&B/database dependency by default;
- framework-shaped plugin systems before concrete need;
- broad config magic;
- mutable historical artifacts;
- silent defaults for protocol-critical choices;
- unlogged exploratory commands;
- optimizing or adding baselines before the research question is explicit.

The current abstractions are intentionally few:

| Abstraction | Purpose |
|---|---|
| `ExperimentSpec` | Intended protocol/model/training/eval plan. |
| `RunManifest` | What actually happened. |
| `ComponentRegistry` | Stable IDs and caveats for frontends/backends/etc. |
| `RunLayout` | Immutable directory shape for new runs. |
| `RunIndex` | Mixed index over new run-layout records and historical evidence. |
| Score/scorer contracts | Stable metric production and official LA tDCF guardrails. |

## Scientific rules

1. No paper number without machine-readable provenance.
2. No silent seed exclusions.
3. Failed/interrupted runs are records, not trash.
4. Historical artifacts are immutable evidence; do not rewrite them to satisfy new schemas.
5. New runs must store resolved specs and manifests.
6. Comparisons must state protocol, seed set, scorer, and eval batch-size compatibility.
7. MiMo eval batch size is a protocol fact.
8. Official LA tDCF comes only from official ASVspoof evaluator output.
9. Exploratory studies cannot be promoted to confirmatory without a new decision/spec.
10. Negative/diagnostic results are valid if the evidence is clean.
11. Every new research-producing command is logged in `docs/current/research_execution_log.jsonl` before/after execution; failed and interrupted runs are logged.

## Scope shift

The project has moved through three phases:

1. **Paper rescue / audit**: find what the earlier manuscript can honestly claim.
2. **System audit / harness**: create guarded scoring, artifact checks, eval/training smokes.
3. **Research framework**: create versioned specs/manifests/indexes so future experiments stay auditable.

We are now in phase 3. Paper prose should still lag evidence, but research-framework work is in scope because it prevents the next experiment wave from becoming another provenance problem.

## Current implementation status

Implemented:

- `ExperimentSpec v1` validation, deterministic hash, and resolved spec writing;
- `RunManifest v1` read/write/validation;
- immutable run-layout helper;
- component metadata registry with stable IDs;
- MiMo batch-size caveat encoded in component metadata;
- `mimodf experiment validate/resolve/init/inspect`;
- optional run-layout v1 manifest updates from `mimodf eval run` and `mimodf train legacy-asvspoof`;
- `mimodf report index` over new run-layout manifests and historical `main_table_provenance.yaml` records;
- JSONL/Markdown run-index rendering;
- `mimodf report aggregate` for numeric metric summaries;
- `mimodf report compare --strict` for seed/protocol/intent/batch-size compatibility checks;
- focused tests in lightweight and full environments;
- `docs/current/research_execution_log.jsonl` plus `docs/current/RESEARCH_EXECUTION_LOG.md` for Wave 0/1 command/artifact logging.

Pending:

- one stored new-layout eval/training smoke example;
- Optuna runner under the spec contract;
- public setup/download story for external dependencies and weights.

## Development order

Do next, in order unless a concrete research need changes priority:

1. controlled experiment matrix specs;
2. one stored new-layout smoke/reproduction example;
3. Optuna only after specs/results/indexing are stable;
4. new frontends/backends only when tied to a declared hypothesis.

## Adding new components

When adding a frontend/backend/adaptation/scorer:

1. assign a stable component ID, e.g. `frontend:wavlm-large/v1`;
2. record metadata and caveats in the registry;
3. add tests that validation accepts the ID and rejects unknown/mis-kind IDs;
4. do not change existing `/v1` semantics; create `/v2` if behavior changes;
5. ensure run manifests record the ID.

## Adding new parameters

New parameters must be additive or versioned.

Allowed:

```yaml
new_param: default_value
```

If old meaning changes, add a new field or schema version. Do not make old runs reinterpret themselves under new code.

## Execution logging policy

The execution log is a source artifact, not a generated afterthought.

Mandatory files:

- `docs/current/EXPERIMENT_LOGGING_PROTOCOL.md` — schema/workflow/rules;
- `docs/current/research_execution_log.jsonl` — machine-readable command ledger;
- `docs/current/RESEARCH_EXECUTION_LOG.md` — readable summary.

Every command that creates or changes research artifacts must have a log entry with exact command, environment, inputs, planned/actual outputs, git revision, status, timings when available, result summary, and caveats. This applies to protocol indexing, feature extraction, scoring, training, evaluation, downloads, probes, reports, and comparisons.

If a command fails, keep the entry and mark it `failed` or `interrupted`. Do not hide failed exploratory runs by omission.

## Reporting policy

Reports are generated outputs. Source of truth remains:

- experiment specs;
- resolved specs;
- manifests;
- score files;
- official metrics;
- provenance YAML for historical rows.

A report may be regenerated. A historical artifact must not be edited.

## Compute policy

No long GPU jobs by default.

Allowed without extra approval:

- lightweight tests;
- dry-run planning;
- bounded smoke tests already scoped by the user;
- indexing/report generation.

Requires explicit approval:

- full LA/DF evals;
- targeted Wave 3 training runs;
- any full training outside a predeclared Wave 3 spec;
- broad Optuna/HPO;
- new baseline matrix execution.

Wave 3 policy: training is no longer scientifically forbidden by default. Unmotivated broad training remains forbidden. Targeted expensive training is allowed only after the plan/spec/log contract is explicit.
