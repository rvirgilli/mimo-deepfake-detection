# Checkpoint provenance gaps

Purpose: compact view of result-cited checkpoints, local availability, and manifest agreement for main-table artifacts. This is assessment evidence; it tells us what can be trusted as metric evidence vs what is incomplete as reproducible run evidence.

## Summary

Current policy: see `docs/current/HISTORICAL_REPRO_SCOPE.md`. Historical score/result files may guide research and support qualified audit context, but gaps are not full reproducibility evidence. If a metric becomes a requirement, recover the exact artifact or rerun as a new controlled experiment.

Metric score/result files are generally usable for EER/tDCF auditing, but several rows cannot yet be fully reproduced from local checkpoints/configs:

- Missing result-cited checkpoints: wav2vec2 adapter seeds 42/789; MiMo frozen seeds 123/789/1234; MiMo full seed1234; MiMo adapter seed2024. MiMo full seed42 has a strong non-exact recovery candidate under `mimo_full_no_earlystop`, but it must not be silently substituted because the run config differs.
- Interrupted/incomplete manifest: MiMo full seed456.
- Manifest/result checkpoint mismatch: MiMo full seed789.
- External/original runs outside `paper_final`: wav2vec2 adapter seed1234, MiMo frozen seed1234, MiMo full seed1234.

Default interpretation: keep score files as metric evidence, but keep affected rows `partial` until checkpoint/config provenance is recovered or the paper explicitly states that only score artifacts are available.

## Main-table checkpoint table

| Model | Strategy | Seed/source | Result-cited checkpoint | Exists locally? | Local checkpoint(s) / manifest | Status | Action |
|---|---|---:|---|---|---|---|---|
| wav2vec2 | frozen | 42 | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_42/best_model.pt` | yes | no config/manifest/log | partial | Locate training config/log if possible. |
| wav2vec2 | frozen | 123 | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_123/best_model.pt` | yes | no config/manifest/log | partial | Locate training config/log if possible. |
| wav2vec2 | frozen | 456 | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_456/best_model.pt` | yes | no config/manifest/log | partial | Locate training config/log if possible. |
| wav2vec2 | frozen | 789 | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_789/best_model.pt` | yes | no config/manifest/log | partial | Locate training config/log if possible. |
| wav2vec2 | frozen | 1234 | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_1234/best_model.pt` | yes | no config/manifest/log | partial | Locate training config/log if possible. |
| wav2vec2 | adapter | 42 | `.../w2v2_adapter_s42_seed42/epoch_5_eer_5.85.pth` | no | local only `epoch_10_eer_5.86.pth`; manifest completed, best_epoch 5 | gap | Recover cited checkpoint or mark score-only. |
| wav2vec2 | adapter | 123 | `.../w2v2_adapter_s123_seed123/epoch_12_eer_5.42.pth` | yes | manifest completed, best_epoch 12 | partial | OK except broader protocol caveats. |
| wav2vec2 | adapter | 456 | `.../w2v2_adapter_s456_seed456/epoch_12_eer_5.42.pth` | yes | manifest completed, best_epoch 12 | partial | OK except broader protocol caveats. |
| wav2vec2 | adapter | 789 | `.../w2v2_adapter_s789_seed789/epoch_17_eer_6.51.pth` | no | local only `epoch_10_eer_6.83.pth`; manifest completed, best_epoch 17 | gap | Recover cited checkpoint or mark score-only. |
| wav2vec2 | adapter | 1234 | `outputs/2026-02-07/13-13-09/.../epoch_14_eer_5.51.pth` | yes | config/manifest/checkpoint in outputs; eval in `experiments/eval_wav2vec2_adapter` | partial | External to `paper_final`; keep explicit. |
| wav2vec2 | full | 42 | `experiments/paper_final/wav2vec2_fullft_multiseed/seed_42/models/epoch_32.pth` | yes | no per-seed config/manifest; archived script documents protocol | partial | Accept as local metric seed; protocol differs from repo-native runs. |
| wav2vec2 | full | 123 | `experiments/paper_final/wav2vec2_fullft_multiseed/seed_123/models/epoch_38.pth` | yes | no per-seed config/manifest; archived script documents protocol | partial | Accept as local metric seed; protocol differs from repo-native runs. |
| wav2vec2 | full | 1234 | `experiments/paper_final/wav2vec2_fullft_multiseed/seed_1234/models/epoch_22.pth` | yes | README documents manual stop/best epoch; no manifest | partial | Accept as local metric seed; protocol differs from repo-native runs. |
| wav2vec2 | full | Tak2022 | external | no local artifact expected | published point only | verified_external | Do not call local seed. |
| MiMo | frozen | 42 | `.../mimo_frozen_s42_seed42/epoch_1_eer_11.17.pth` | yes | manifest completed, best_epoch 1 | partial | OK except broader protocol caveats. |
| MiMo | frozen | 123 | `.../mimo_frozen_s123_seed123/epoch_6_eer_10.85.pth` | no | local only `epoch_10_eer_11.17.pth`; manifest completed, best_epoch 6 | gap | Recover cited checkpoint or mark score-only. |
| MiMo | frozen | 456 | `.../mimo_frozen_s456_seed456/epoch_0_eer_13.67.pth` | yes | manifest completed, best_epoch 0 | partial | Valid completed outlier unless failure evidence appears. |
| MiMo | frozen | 789 | `.../mimo_frozen_s789_seed789/epoch_5_eer_10.63.pth` | no | local only `epoch_3_eer_11.61.pth`; manifest completed, best_epoch 5 | gap | Recover cited checkpoint or mark score-only. |
| MiMo | frozen | 1234 | `outputs/2026-02-08/02-05-01/.../epoch_1_eer_9.78.pth` | no | no local config/manifest/checkpoint in cited output; paper_final has result files only | gap | Recover output dir or keep result-only with no official tDCF. |
| MiMo | adapter | 42-ish/trial39 | `experiments/optuna/mimo_hpo_v2/trial_0039/best_model.pt` | yes | config/metrics/status present; seed provenance imperfect | partial | Keep as one found full eval. |
| MiMo | adapter | 2024 | `outputs/2026-01-27/18-01-37/.../epoch_4_eer_9.87.pth` | no | score/result files exist; portable recipe `configs/train_seed2024.yaml`; exact output dir missing | gap | Recover output dir or keep score-only n=2. |
| MiMo | full | 42 | `.../mimo_full_s42_seed42/epoch_12_eer_9.87.pth` | no | local only `epoch_11_eer_9.98.pth`; manifest completed, best_epoch 12 | gap | Recover cited checkpoint or mark score-only. |
| MiMo | full | 123 | `.../mimo_full_s123_seed123/epoch_1_eer_9.98.pth` | yes | manifest completed, best_epoch 1 | partial | OK except broader protocol caveats. |
| MiMo | full | 456 | `.../mimo_full_s456_seed456/epoch_0_eer_12.02.pth` | yes | manifest status `running`; no completed metrics; resume script says training was interrupted and eval-only used existing checkpoint | gap | Treat as interrupted/partial; decide whether acceptable or needs rerun. |
| MiMo | full | 789 | `.../mimo_full_s789_seed789/epoch_0_eer_12.37.pth` | yes | manifest completed, best_epoch 1, best_eer 11.84; result/local checkpoint use epoch 0 | mismatch | Explain mismatch or recover epoch 1 checkpoint. |
| MiMo | full | 1234 | `outputs/2026-02-08/05-20-18/.../epoch_1_eer_13.34.pth` | no | score/result files exist in `experiments/eval_mimo_full`; no local config/checkpoint | gap | Recover output dir or keep external score-only. |

## Priority recovery list

1. **MiMo adapter seed2024 output dir** — central if reporting adapter behavior beyond one HPO trial.
2. **MiMo frozen seed1234 score/config/checkpoint** — needed for same-set official tDCF and full provenance.
3. **MiMo full seed1234 config/checkpoint** — needed because paper EER mean uses this seed.
4. **MiMo full seed456 status** — decide whether interrupted eval-only seed is acceptable or requires controlled rerun.
5. **Missing best checkpoints for seeds with score files** — wav2vec2 adapter 42/789, MiMo frozen 123/789, MiMo full 42/789.

## Assessment consequence

The corrected table can still be built from score files, but reproducibility claims must be weaker:

- `metric artifact available` is not the same as `full run provenance available`.
- Rows with missing checkpoints/configs should stay `partial`.
- Any release package should either recover missing artifacts or provide score files plus explicit caveats.
