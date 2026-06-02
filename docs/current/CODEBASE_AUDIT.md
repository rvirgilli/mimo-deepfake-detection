# Codebase audit for publishable rework

Purpose: assess whether the current research repo should be refactored in place or rebuilt into a lean, reliable, debuggable, maintainable, publishable codebase.

## Verdict

Do a **clean rebuild-from-harvest**, not a broad in-place refactor.

Do not throw away the science or all code. Keep the verified pieces, evidence docs, and selected modules. But the public/reproducible code path should be rebuilt as a small package with one training path, one scoring path, typed configs, machine-readable provenance, and tests around failure-prone behavior.

Reason: the current repo is an effective exploration workspace, but its complexity now encodes the history of the earlier manuscript. In-place cleanup would preserve too many accidental interfaces and stale assumptions.

## Current architecture map

```text
train.py / evaluate.py / optuna_train.py
  -> Hydra configs under configs/
  -> src.data_utils + src.rawboost
  -> src.frontends.{wav2vec2,mimo,hubert,encodec,...}
  -> src.model.Model AASIST-style backend
  -> experiments/, outputs/, logs/, models/ local artifacts
  -> docs/current/* audit truth
```

Important source areas:

| Area | Current files | Assessment |
|---|---|---|
| Training | `train.py`, `optuna_train.py` | Duplicated logic, protocol surprises, noncanonical HPO path. |
| Evaluation | `evaluate.py`, `scripts/eval_asvspoof2021.py`, SSL scripts | Multiple paths; wrong-scale tDCF risk; `evaluate.py` stale for many checkpoints. |
| Model/backend | `src/model.py` | Valuable AASIST backend, but large and intertwined with frontend/projection assumptions. |
| Frontends | `src/frontends/*` | Useful abstractions, but too many speculative variants for final paper. |
| Data/augmentation | `src/data_utils.py`, `src/rawboost.py` | Useful, but needs deterministic seeding, explicit data layout, safer paths. |
| Provenance | `src/experiment.py`, `src/results.py`, docs | Manifest idea is good; SQLite/results DB is not yet source of truth. |
| Configs | `configs/` | Portable recipes now, but not typed or protocol-explicit enough. |
| Tests | `tests/` | Good start; mostly mocks/unit tests. Need scorer/provenance/protocol tests. |

## Key problems

### 1. Too many canonical-looking paths

There are several ways to train/evaluate:

- `train.py`
- `evaluate.py`
- `optuna_train.py`
- archived shell scripts
- `scripts/eval_asvspoof2021.py`
- `SSL_Anti-spoofing/main_SSL_LA.py`
- official ASVspoof scoring scripts

This makes it easy to produce a number without knowing which protocol/scorer created it.

### 2. Evaluation/scoring is dangerous

Known from the assessment:

- project `results_LA_eval.txt` often contains wrong-scale tDCF;
- official tDCF must be computed separately;
- `evaluate.py` tries to compute metrics through a path that is not the audit source of truth;
- tDCF aggregation drift caused paper inconsistencies.

A publishable repo needs one boring scorer wrapper and tests that prevent wrong-scale tDCF from entering tables.

### 3. Protocol is implicit

Important choices are hidden in code branches or old scripts:

- ASVspoof2021 fast subset vs ASVspoof2019 dev checkpoint selection;
- Adam vs AdamW depends on `encoder_lr` being null/non-null;
- validation loss weights differ from some training loss weights;
- wav2vec2 full FT uses a different Tak/SSL stack.

Protocol must become data, not prose.

### 4. Frontend scope is too broad

Current frontends include wav2vec2, MiMo, HuBERT, EnCodec, native 50Hz, RVQ strategies, weighted layers, adapters, LoRA, partial, gradual unfreezing.

For the assessed paper, most of this is not needed. Keeping it all in the public core makes the repo harder to debug and easier to misuse.

### 5. Reproducibility is partial

Good pieces exist:

- manifests;
- resolved configs in artifacts;
- score files;
- tests;
- active docs.

But gaps remain:

- missing checkpoints/configs for some rows;
- local ignored external clones;
- dependencies not fully pinned;
- package metadata has placeholder URLs and broad wheel packaging;
- no machine-readable main table provenance file.

### 6. Current docs are stronger than code

`docs/current/*` now contains better truth than the runnable scripts. That is good for assessment but bad for publication. The next codebase should generate or validate docs/tables from structured provenance.

## What to keep

Harvest these:

- `docs/current/*` as decision/evidence source of truth.
- `src/rawboost.py` scaling logic and tests.
- `src/frontends/base.py` interface idea.
- Minimal `wav2vec2` frontend path needed for final claims.
- Minimal `MiMo` continuous/adaptor/full/frozen path needed for final claims.
- AASIST backend behavior from `src/model.py`, after isolating it from exploration baggage.
- Existing score/result artifacts as evidence inputs.
- Unit tests for projection, RawBoost scaling, feature strategy, adapter wrappers.

## What to archive or exclude from the publishable core

Archive/private evidence only:

- historical shell scripts;
- HPO scripts and Optuna databases;
- historical outputs/logs/models;
- `ResultsDB` unless replaced by a tiny structured manifest index;
- HuBERT/EnCodec unless final paper uses them;
- LoRA/partial/gradual/RVQ/layer-select/native-50Hz variants unless explicitly retained as future work;
- local external clones (`SSL_Anti-spoofing/`, `MiMo-Audio-Tokenizer/`).

Do not delete evidence artifacts from the working repo without explicit approval. This is about the future public core, not destructive cleanup.

## Start-from-scratch recommendation

Start a new package inside this repo, then migrate only what we need:

```text
mimodf/
  config.py              typed config schema + validation
  data/asvspoof.py       protocols, datasets, file layout checks
  augment/rawboost.py    harvested RawBoost + deterministic RNG handling
  frontends/base.py      small stable interface
  frontends/wav2vec2.py  final wav2vec2 path only
  frontends/mimo.py      final MiMo path only
  backends/aasist.py     backend, no frontend side effects
  training/loop.py       one train_one_run function
  scoring/write_scores.py
  metrics/asvspoof.py    official scorer wrapper
  provenance.py          JSON manifests + hashes
  tables/main.py         table generation from provenance YAML/JSON
```

Keep old `src/` during migration. Do not rename/delete it until the new package reproduces core checks.

## Design principles for the rebuild

- One canonical path for each operation.
- Protocol is explicit in config and manifest.
- Tables are generated from structured provenance, not hand-edited values.
- Evaluation is score-first: score file in, official scorer out, JSON result out.
- Every result has hashes for config/checkpoint/score/result when files exist.
- Missing artifacts are first-class status, not comments hidden in prose.
- Small modules, but not fragmented abstractions.
- No training reruns required to build the new audit/scoring/table layer.

## Risks if we refactor in place instead

- Old scripts keep looking canonical.
- Stale `evaluate.py` may continue to produce untrusted metrics.
- HPO/frontends/experiments continue driving design.
- New code inherits implicit Hydra behavior and global paths.
- More time spent making historical decisions look clean than building a reliable interface.

## Publishability blockers to fix before release

- Replace placeholder package URLs.
- Pin/document external dependencies and Python versions.
- Remove or clearly exclude local clones from package build.
- Provide a small reproducibility README with exact commands.
- Provide structured provenance for the main table.
- Provide tests for scorer aggregation and artifact checks.
- Separate evidence archive from runnable public code.
