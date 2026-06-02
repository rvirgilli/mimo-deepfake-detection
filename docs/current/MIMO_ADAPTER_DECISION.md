# MiMo adapter path decision

Purpose: decide what to do with the unsupported MiMo adapter n=5 row before any manuscript rewrite or rerun.

## Current evidence

Only two full ASVspoof2021 MiMo adapter evaluations are locally traceable:

| Source | Seed/source | LA EER | LA tDCF | DF EER | Provenance status |
|---|---:|---:|---:|---:|---|
| `experiments/eval_2021_v2/` + `experiments/optuna/mimo_hpo_v2/trial_0039/` | trial39 / RNG seed 42-ish | 4.4053 | 0.2979 | 11.7145 | checkpoint/config/scores/results present; seed field inconsistent |
| `experiments/eval_seed2024/` | 2024 | 4.3700 | 0.2966 | 7.7014 | scores/results present; result-cited checkpoint/output dir missing |

Observed n=2 summary:

- LA EER: `4.39 ± 0.02`
- LA tDCF: `0.297 ± 0.001`
- DF EER: `9.71 ± 2.84`

Unsupported paper row:

- LA EER: `4.64 ± 0.26`
- DF EER: `10.15 ± 1.70`
- claimed/implied n=5

No local artifact family such as `mimo_adapter_multiseed/` was found. The archived statistical script explicitly used only two full-eval points for MiMo adapter while also showing the paper's claimed n=5 summary as a separate unsupported summary-stat calculation.

## Options

### Option A — recover missing artifacts first

Search likely external/local storage for:

- missing three full-eval score/result files;
- seed2024 output dir/checkpoint;
- any `mimo_adapter_multiseed` directory;
- old `outputs/2026-01-*` and `outputs/2026-02-*` snapshots;
- cloud/backup/anonymous repo package, if it existed.

Acceptance criteria for using n=5:

- five LA score files;
- five DF score files;
- five result files or official evaluator outputs;
- exact config per seed;
- checkpoint per seed, or explicit score-only caveat;
- seed list and aggregation command recorded.

Cost: low if backups exist. Scientific risk: lowest.

### Option B — report n=2 honestly

Use the two found full-eval points and label MiMo adapter as exploratory/underpowered.

Pros:

- no GPU cost;
- fully honest about evidence;
- preserves the observation that MiMo adapter can improve over all-seed MiMo frozen on LA, but without strong statistical claims.

Cons:

- cannot support significance tests or low-variance claims;
- weakens the central adaptation-trajectory narrative;
- DF variance is large (`±2.84`).

Required claim changes later:

- remove/avoid MiMo adapter significance claims;
- call adapter result preliminary;
- avoid saying adapters are definitively critical for MiMo unless supported by additional analyses.

### Option C — controlled rerun to complete n=5

Run exactly three missing MiMo adapter seeds with a locked protocol.

Preconditions:

- user approval for GPU time;
- write a run card before launching;
- no HPO;
- frozen config from trial39/seed2024 recipe;
- exact seeds chosen before run;
- final eval commands and scorer fixed;
- acceptance criteria defined before results are known.

Recommended seed set if rerunning:

- preserve existing sources: trial39/42-ish and seed2024;
- add three predetermined seeds, e.g. `123`, `456`, `789`, unless there is a stronger reason to match another seed set.

Risks:

- expensive relative to assessment phase;
- may produce worse or higher-variance results;
- if protocol differs from original, it becomes a new result, not provenance for the old n=5 claim.

## Decision

Assume the missing n=5 artifacts are unavailable.

Default for assessment: **Option B — report the two found full-eval seeds honestly as exploratory/underpowered.**

Do **not** start training during assessment. If later we decide MiMo adapter needs n=5 evidence, that becomes a new controlled experiment, not provenance for the historical n=5 claim. Before any GPU job, create a locked `MIMO_ADAPTER_RERUN_PLAN.md` with seeds, config, evaluator, and acceptance criteria fixed in advance.

## Search / recovery log

### 2026-05-25 local search

Commands searched the current repo parent, `<home>` to bounded depth, Claude/cache backup dirs, and the older `<legacy-repo>` project for:

- `mimo_adapter_multiseed`
- `trial39_seed*`
- `seed2024`
- `eval_seed*`
- `outputs/2026-01-27`

Result: no missing MiMo adapter artifacts were found beyond the already known current-repo files:

- `experiments/eval_seed2024/`
- `experiments/eval_seed2024_LA.log`
- `experiments/optuna/mimo_hpo_v2/trial_0039/`

An older project at `<legacy-repo>` exists, but bounded search found no `trial39`, `seed2024`, `eval_seed`, or MiMo adapter full-eval artifact matches there.

### 2026-05-25 expanded `~/projects` and home search

Follow-up targeted search covered all of `~/projects` with `.git`, `node_modules`, and `__pycache__` pruned, then targeted `~` with large/noisy roots pruned (`~/miniconda3`, `~/datasets`, `~/.cache`, `~/.local`, `.git`, `node_modules`, `__pycache__`). Patterns:

- `*mimo_adapter*`
- `*trial39*`
- `*seed2024*`
- `*eval_seed*`
- `*/outputs/2026-01-27*`
- `*/experiments/*multi_seed*` for `~/projects`

Result: still no missing adapter artifacts. Matches were only current-repo known files:

- `configs/train_seed2024.yaml`
- `experiments/eval_seed2024/`
- `experiments/eval_seed2024_LA.log`
- `experiments/multi_seed/` logs/placeholders
- `docs/current/MIMO_ADAPTER_DECISION.md`

### Remaining external search targets

Concrete names/patterns to search in backups or external storage:

```text
mimo_adapter_multiseed
trial39_seed123
trial39_seed456
trial39_seed789
trial39_seed1337
trial39_seed2024
experiments/eval_seed*
outputs/2026-01-27/18-01-37
outputs/2026-01-*/experiments/multi_seed
outputs/2026-02-*/experiments/*adapter*
```

Local evidence already found but incomplete:

- `experiments/eval_seed2024/`
- `experiments/multi_seed/logs/train_seed2024.log`
- `configs/train_seed2024.yaml`
- `experiments/optuna/mimo_hpo_v2/trial_0039/`

## Consequence for corrected table

Until Option A succeeds or Option C is approved/completed, the corrected main table should use:

```text
MiMo Adapter, n=2 found evals: LA 4.39 ± 0.02, LA tDCF 0.297, DF 9.71 ± 2.84
```

with a footnote that n=5 was unsupported in the historical artifacts.
