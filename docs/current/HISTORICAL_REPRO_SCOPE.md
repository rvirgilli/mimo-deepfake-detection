# Historical reproducibility scope

Date: 2026-05-27
Status: active policy for historical historical artifacts

Machine-readable inputs:

- `docs/current/main_table_provenance.yaml`
- `docs/current/artifact_gap_decisions.yaml`
- `docs/current/research_execution_log.jsonl`

Related evidence docs:

- `docs/current/CHECKPOINT_PROVENANCE_GAPS.md`
- `docs/current/RELEASE_CHECKLIST.md`
- `docs/current/SYSTEM_STATUS.md`

## Decision

Do not claim full historical reproduction of the earlier-manuscript experiment set.

The historical artifacts are useful for audit and for choosing research directions, but rows with missing exact checkpoints/configs/output directories are not complete reproducibility evidence.

If a historical metric becomes central to a new paper, release, requirement, or external claim, it must be one of:

1. verified from complete local provenance;
2. recovered exactly from the original artifact path/source;
3. rerun as a new controlled experiment under the current `ExperimentSpec`/manifest/log system.

A new rerun must not be presented as the missing historical artifact. It is a new run.

## Release-gate interpretation

The current split is intentional:

```bash
python -m mimodf audit release-gate --strict
```

should fail while historical artifacts are missing.

```bash
python -m mimodf audit release-gate --system-profile --strict
```

may pass for system/tooling release checks because known historical gaps are explicitly documented.

This means the repo can be released as an audited framework with partial historical evidence, not as a turnkey reproduction of every historical run.

## Reliability tiers

| Tier | Meaning | Allowed use |
|---|---|---|
| Tier 1: complete local provenance | Score/result files, config, checkpoint, and enough protocol facts are present. | Can support local reproduction/audit claims with caveats. |
| Tier 2: score-backed partial | Score/result files exist, but exact best checkpoint/config/log is missing or mismatched. | Can report metrics only as historical score-backed evidence; cannot claim exact rerunnable provenance. |
| Tier 3: exploratory | Underpowered, non-final, HPO/trial, or seed/source provenance is incomplete. | Can motivate future work; not a paper requirement or main claim. |
| Tier 4: retired / rerun-required | Original output dir/checkpoint/config is absent and score-only evidence is too weak for the intended claim. | Do not use for claims; recover or rerun if needed. |

## Deep revisit of the 9 known gaps

A new scan of local `experiments/` and `outputs/` artifacts found one recovery candidate and eight gaps that remain non-exact.

| Row | Seed | Missing exact artifact | Local evidence after revisit | Reliability tier | If this becomes required |
|---|---:|---|---|---|---|
| wav2vec2 adapter | 42 | `epoch_5_eer_5.85.pth` | LA/DF score/result files, config, manifest with best epoch 5; only local adjacent checkpoint is `epoch_10_eer_5.86.pth`. | Tier 2: score-backed partial | Recover exact epoch-5 checkpoint or rerun as new controlled seed. |
| wav2vec2 adapter | 789 | `epoch_17_eer_6.51.pth` | LA/DF score/result files, config, manifest with best epoch 17; only local adjacent checkpoint is `epoch_10_eer_6.83.pth`. | Tier 2: score-backed partial | Recover exact epoch-17 checkpoint or rerun as new controlled seed. |
| MiMo frozen | 123 | `epoch_6_eer_10.85.pth` | LA/DF score/result files, config, manifest with best epoch 6; only local adjacent checkpoint is `epoch_10_eer_11.17.pth`. | Tier 2: score-backed partial | Recover exact checkpoint or rerun under pinned MiMo batch-size protocol. |
| MiMo frozen | 789 | `epoch_5_eer_10.63.pth` | LA/DF score/result files, config, manifest with best epoch 5; only local adjacent checkpoint is `epoch_3_eer_11.61.pth`. | Tier 2: score-backed partial | Recover exact checkpoint or rerun under pinned MiMo batch-size protocol. |
| MiMo frozen | 1234 | `outputs/2026-02-08/02-05-01/.../epoch_1_eer_9.78.pth` | Original output dir absent; copied result evidence only; same-set official LA score/provenance incomplete. | Tier 4 for full reproduction; score-only directional evidence | Recover original output dir or rerun as a new controlled seed. |
| MiMo adapter | 2024 | `outputs/2026-01-27/18-01-37/.../epoch_4_eer_9.87.pth` | Original output dir absent; score/result files and `configs/train_seed2024.yaml`; adapter evidence remains n=2/exploratory. | Tier 3: exploratory | Rerun only if adapter becomes a new approved hypothesis; do not repair silently. |
| MiMo full | 42 | `epoch_12_eer_9.87.pth` under `mimo_full_multiseed` | Strong candidate exists under `mimo_full_no_earlystop/seed_42/models/mimo_full_noes_s42_seed42/epoch_12_eer_9.87.pth`; adjacent epoch-11 checkpoint is byte-identical between runs, but config differs in patience/experiment path/name. | Tier 2 with recovery candidate; not silently promoted | Decide explicitly whether candidate is acceptable, or rerun as controlled seed. |
| MiMo full | 789 | `epoch_1_eer_11.84.pth` | LA/DF score/result files, config, manifest says best epoch 1; local eval/checkpoint uses `epoch_0_eer_12.37.pth`. | Tier 2 with mismatch | Recover epoch-1 checkpoint or rerun; explain mismatch if metric is cited. |
| MiMo full | 1234 | `outputs/2026-02-08/05-20-18/.../epoch_1_eer_13.34.pth` | Original output dir absent; copied score/result evidence only. | Tier 2 for score-backed metric; Tier 4 for full reproduction | Recover original output dir or rerun as new controlled seed. |

## Interpretation for historical metrics

Historical main-table numbers can be used only with explicit tier labels and caveats.

Allowed phrasing:

> Historical score artifacts indicate X, but exact run provenance is partial for seeds A/B; the number is suitable for audit context and research planning, not as a standalone reproducibility guarantee.

Not allowed:

> This row is fully reproducible.

Not allowed:

> The missing checkpoint was effectively recovered by a nearby checkpoint.

Not allowed:

> A new rerun proves the old run existed.

## What to do if a metric becomes important

Use this decision tree:

1. Is the exact score/result/config/checkpoint/protocol present?
   - yes: audit and cite with caveats.
   - no: continue.
2. Is the metric only directional/background?
   - yes: mark Tier 2/3/4 and keep out of claims.
   - no: continue.
3. Can the exact artifact be recovered from backup/original output dir?
   - yes: recover and record provenance.
   - no: continue.
4. Rerun as a new controlled experiment:
   - create/reuse an `ExperimentSpec`;
   - log planned execution before running;
   - record seed, batch size, checkpoint selection, eval batch size, scorer, and git revision;
   - compare against historical score artifacts only as a reproduction audit, not as identity.

## Practical recommendation

Do not spend more time chasing these gaps unless a specific missing artifact is known to exist elsewhere.

For research direction, rely on the newer logged Wave 0/1 feature-probe evidence and any future controlled runs.

For paper/release language, treat historical rows as partial audit evidence, not hard requirements.
