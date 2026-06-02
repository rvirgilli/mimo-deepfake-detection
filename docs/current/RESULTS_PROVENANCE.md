# Results provenance ledger

Status: starting audit. All manuscript numbers are untrusted until represented here.

Validation labels: `verified`, `verified_external`, `partial`, `unverified`, `invalid`, `needs_rerun`.

## Main manuscript table audit

| Claim/result | Paper value | Current status | Evidence needed | Notes |
|---|---:|---|---|---|
| wav2vec2 frozen LA/DF mean/std | LA 8.05±0.73, DF 6.76±0.61 | partial | missing historical configs/logs/manifests | seed rows populated; EER means match artifacts and paper when using sample std; summary file uses population std. LA tDCF mean 0.3844 from `tdcf_recomputed.txt`. |
| wav2vec2 adapter LA/DF mean/std | LA 2.77±0.81, DF 5.11±0.72 | partial | checkpoint mismatch for seeds 42/789; explain LA tDCF aggregation | seed rows populated. EER mean/std match paper when including external seed 1234 and using sample std. LA tDCF paper value 0.251 matches four `paper_final` seeds only; including external seed 1234 official tDCF gives mean 0.2553. |
| wav2vec2 full LA/DF mean/std | LA 1.07±0.06, DF 4.00±1.32 | invalid | revise n/stat definition; decide external Tak handling | paper value matches three local reproduced seeds plus Tak et al. published point (LA 1.01, DF 2.78), not four local seeds. Three local seeds alone give LA 1.09±0.06, DF 4.41±1.26. Do not call the current paper value a 4-seed mean. |
| MiMo frozen LA/DF mean/std | LA 6.09±0.58, DF 12.41±0.69 | invalid | revise to include seed 456 or document concrete failure | paper value exactly matches excluding seed 456 (sample std) and tDCF mean over seeds 42/123/789. No failure/corruption evidence found for seed 456; manifest status completed. All-five EER is LA 7.11±2.35, DF 12.86±1.18 (sample std). |
| MiMo adapter LA/DF mean/std | LA 4.64±0.26, DF 10.15±1.70 | invalid | replace with n=2 exploratory unless future controlled rerun is approved | Missing n=5 artifacts assumed unavailable. Only two full ASVspoof2021 evals found: trial 39/seed 42-ish and seed 2024. Observed n=2 stats are LA 4.39±0.02, DF 9.71±2.84 (sample std); paper's n=5 mean/std are unsupported. |
| MiMo full LA/DF mean/std | LA 6.94±2.02, DF 12.73±1.18 | partial | resolve seed 456 incomplete manifest, missing/checkpoint mismatches, tDCF aggregation | EER mean/std matches five seeds if including external/original seed 1234 (`experiments/eval_mimo_full`). Table footnote says 4 seeds, which is wrong for these EER values. LA tDCF 0.338 matches only four local seeds, excluding seed 1234; including seed 1234 official tDCF gives mean 0.3496. |
| MiMo LR sensitivity | table in paper | partial | configs/result files per LR, note differing epoch budgets | result files exist under `experiments/paper_final/lr_sensitivity`. |
| MiMo regularization sweep | table in paper | partial | result files, configs, single-seed caveat | result files exist for mild/moderate/strong; extreme under `mimo_fullft_regularized`. |
| LA min-tDCF values | table in paper | partial | resolve missing same-seed tDCF for MiMo frozen seed1234/Tak external point | Reconciled in `docs/current/TDCF_RECONCILIATION.md`. Valid same-set: wav2vec2 frozen 0.384, wav2vec2 adapter 0.255 for n=5, wav2vec2 full local n=3 0.215, MiMo adapter n=2 0.297, MiMo full n=5 0.350. Paper values 0.251 and 0.338 mix seed sets in some rows. |
| Validation protocol | ASVspoof2019 dev in paper | unverified | inspect configs/logs per run | `train.py` uses ASVspoof2021 fast subset when present. Must identify actual behavior. |

## Seed-level ledger

Fill one row per seed/result.

| Model | Strategy | Seed | Track | EER | min-tDCF | Status | Score file | Result file | Config | Checkpoint | Evaluator | Notes |
|---|---|---:|---|---:|---:|---|---|---|---|---|---|---|
| wav2vec2 | frozen | 42 | LA | 8.21 | 0.3867 | partial | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_42/eval/scores_LA_eval.txt` | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_42/eval/results_LA_eval.txt`; `experiments/paper_final/tdcf_recomputed.txt` | missing | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_42/best_model.pt` | official LA tDCF recompute + project result | config/log/manifest missing; metric artifacts present |
| wav2vec2 | frozen | 42 | DF | 6.79 | | partial | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_42/eval/scores_DF_eval.txt` | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_42/eval/results_DF_eval.txt` | missing | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_42/best_model.pt` | project EER result | config/log/manifest missing; DF EER only |
| wav2vec2 | frozen | 123 | LA | 8.40 | 0.3980 | partial | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_123/eval/scores_LA_eval.txt` | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_123/eval/results_LA_eval.txt`; `experiments/paper_final/tdcf_recomputed.txt` | missing | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_123/best_model.pt` | official LA tDCF recompute + project result | config/log/manifest missing; metric artifacts present |
| wav2vec2 | frozen | 123 | DF | 7.40 | | partial | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_123/eval/scores_DF_eval.txt` | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_123/eval/results_DF_eval.txt` | missing | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_123/best_model.pt` | project EER result | config/log/manifest missing; DF EER only |
| wav2vec2 | frozen | 456 | LA | 8.79 | 0.4037 | partial | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_456/eval/scores_LA_eval.txt` | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_456/eval/results_LA_eval.txt`; `experiments/paper_final/tdcf_recomputed.txt` | missing | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_456/best_model.pt` | official LA tDCF recompute + project result | config/log/manifest missing; metric artifacts present |
| wav2vec2 | frozen | 456 | DF | 6.60 | | partial | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_456/eval/scores_DF_eval.txt` | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_456/eval/results_DF_eval.txt` | missing | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_456/best_model.pt` | project EER result | config/log/manifest missing; DF EER only |
| wav2vec2 | frozen | 789 | LA | 6.85 | 0.3545 | partial | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_789/eval/scores_LA_eval.txt` | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_789/eval/results_LA_eval.txt`; `experiments/paper_final/tdcf_recomputed.txt` | missing | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_789/best_model.pt` | official LA tDCF recompute + project result | config/log/manifest missing; metric artifacts present |
| wav2vec2 | frozen | 789 | DF | 5.82 | | partial | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_789/eval/scores_DF_eval.txt` | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_789/eval/results_DF_eval.txt` | missing | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_789/best_model.pt` | project EER result | config/log/manifest missing; DF EER only |
| wav2vec2 | frozen | 1234 | LA | 7.98 | 0.3791 | partial | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_1234/eval/scores_LA_eval.txt` | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_1234/eval/results_LA_eval.txt`; `experiments/paper_final/tdcf_recomputed.txt` | missing | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_1234/best_model.pt` | official LA tDCF recompute + project result | config/log/manifest missing; metric artifacts present |
| wav2vec2 | frozen | 1234 | DF | 7.18 | | partial | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_1234/eval/scores_DF_eval.txt` | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_1234/eval/results_DF_eval.txt` | missing | `experiments/paper_final/wav2vec2_frozen_multiseed/seed_1234/best_model.pt` | project EER result | config/log/manifest missing; DF EER only |
| wav2vec2 | adapter | 42 | LA | 3.3217 | 0.2707 | partial | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_42/eval/scores_LA_eval.txt` | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_42/eval/results_LA_eval.txt`; `experiments/paper_final/tdcf_recomputed.txt` | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_42/models/w2v2_adapter_s42_seed42/config.yaml` | missing expected `epoch_5_eer_5.85.pth`; local dir has `epoch_10_eer_5.86.pth` | official LA tDCF recompute + project result | result/manifest says best epoch 5; checkpoint file missing locally; project tDCF 0.0073 is wrong-scale |
| wav2vec2 | adapter | 42 | DF | 3.9819 | | partial | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_42/eval/scores_DF_eval.txt` | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_42/eval/results_DF_eval.txt` | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_42/models/w2v2_adapter_s42_seed42/config.yaml` | missing expected `epoch_5_eer_5.85.pth`; local dir has `epoch_10_eer_5.86.pth` | project EER result | result/manifest says best epoch 5; checkpoint file missing locally; DF EER only |
| wav2vec2 | adapter | 123 | LA | 2.3284 | 0.2458 | partial | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_123/eval/scores_LA_eval.txt` | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_123/eval/results_LA_eval.txt`; `experiments/paper_final/tdcf_recomputed.txt` | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_123/models/w2v2_adapter_s123_seed123/config.yaml` | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_123/models/w2v2_adapter_s123_seed123/epoch_12_eer_5.42.pth` | official LA tDCF recompute + project result | config/manifest/checkpoint present; project tDCF 0.0052 is wrong-scale |
| wav2vec2 | adapter | 123 | DF | 5.8586 | | partial | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_123/eval/scores_DF_eval.txt` | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_123/eval/results_DF_eval.txt` | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_123/models/w2v2_adapter_s123_seed123/config.yaml` | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_123/models/w2v2_adapter_s123_seed123/epoch_12_eer_5.42.pth` | project EER result | config/manifest/checkpoint present; DF EER only |
| wav2vec2 | adapter | 456 | LA | 2.8833 | 0.2582 | partial | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_456/eval/scores_LA_eval.txt` | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_456/eval/results_LA_eval.txt`; `experiments/paper_final/tdcf_recomputed.txt` | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_456/models/w2v2_adapter_s456_seed456/config.yaml` | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_456/models/w2v2_adapter_s456_seed456/epoch_12_eer_5.42.pth` | official LA tDCF recompute + project result | config/manifest/checkpoint present; project tDCF 0.0064 is wrong-scale |
| wav2vec2 | adapter | 456 | DF | 5.5107 | | partial | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_456/eval/scores_DF_eval.txt` | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_456/eval/results_DF_eval.txt` | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_456/models/w2v2_adapter_s456_seed456/config.yaml` | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_456/models/w2v2_adapter_s456_seed456/epoch_12_eer_5.42.pth` | project EER result | config/manifest/checkpoint present; DF EER only |
| wav2vec2 | adapter | 789 | LA | 1.6333 | 0.2284 | partial | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_789/eval/scores_LA_eval.txt` | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_789/eval/results_LA_eval.txt`; `experiments/paper_final/tdcf_recomputed.txt` | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_789/models/w2v2_adapter_s789_seed789/config.yaml` | missing expected `epoch_17_eer_6.51.pth`; local dir has `epoch_10_eer_6.83.pth` | official LA tDCF recompute + project result | result/manifest says best epoch 17; checkpoint file missing locally; project tDCF 0.0035 is wrong-scale |
| wav2vec2 | adapter | 789 | DF | 4.8891 | | partial | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_789/eval/scores_DF_eval.txt` | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_789/eval/results_DF_eval.txt` | `experiments/paper_final/wav2vec2_adapter_multiseed/seed_789/models/w2v2_adapter_s789_seed789/config.yaml` | missing expected `epoch_17_eer_6.51.pth`; local dir has `epoch_10_eer_6.83.pth` | project EER result | result/manifest says best epoch 17; checkpoint file missing locally; DF EER only |
| wav2vec2 | adapter | 1234 | LA | 3.6857 | 0.2733 | partial | `experiments/eval_wav2vec2_adapter/scores_LA_eval.txt` | `experiments/eval_wav2vec2_adapter/results_LA_eval.txt`; command output from `SSL_Anti-spoofing/evaluate_2021_LA.py` | `outputs/2026-02-07/13-13-09/experiments/models/wav2vec2_adapter_fair_seed1234/config.yaml` | `outputs/2026-02-07/13-13-09/experiments/models/wav2vec2_adapter_fair_seed1234/epoch_14_eer_5.51.pth` | official LA tDCF command + project result | outside `paper_final`; project tDCF 0.0082 is wrong-scale; official tDCF computed during audit |
| wav2vec2 | adapter | 1234 | DF | 5.2858 | | partial | `experiments/eval_wav2vec2_adapter/scores_DF_eval.txt` | `experiments/eval_wav2vec2_adapter/results_DF_eval.txt` | `outputs/2026-02-07/13-13-09/experiments/models/wav2vec2_adapter_fair_seed1234/config.yaml` | `outputs/2026-02-07/13-13-09/experiments/models/wav2vec2_adapter_fair_seed1234/epoch_14_eer_5.51.pth` | project EER result | outside `paper_final`; DF EER only |
| wav2vec2 | full | 42 | LA | 1.10 | 0.2157 | partial | `experiments/paper_final/wav2vec2_fullft_multiseed/seed_42/eval/scores_LA_eval.txt` | `experiments/paper_final/wav2vec2_fullft_multiseed/seed_42/eval/results_LA_eval.txt`; `experiments/paper_final/tdcf_recomputed.txt` | missing; protocol in archived `run_w2v_fullft.sh` | `experiments/paper_final/wav2vec2_fullft_multiseed/seed_42/models/epoch_32.pth` | official LA evaluator via Tak/SSL script | no config/manifest; full FT protocol differs from repo train.py; best checkpoint selected by validation loss |
| wav2vec2 | full | 42 | DF | 4.17 | | partial | `experiments/paper_final/wav2vec2_fullft_multiseed/seed_42/eval/scores_DF_eval.txt` | `experiments/paper_final/wav2vec2_fullft_multiseed/seed_42/eval/results_DF_eval.txt` | missing; protocol in archived `run_w2v_fullft.sh` | `experiments/paper_final/wav2vec2_fullft_multiseed/seed_42/models/epoch_32.pth` | official/project DF EER via Tak/SSL script | no config/manifest; DF EER only |
| wav2vec2 | full | 123 | LA | 1.03 | 0.2127 | partial | `experiments/paper_final/wav2vec2_fullft_multiseed/seed_123/eval/scores_LA_eval.txt` | `experiments/paper_final/wav2vec2_fullft_multiseed/seed_123/eval/results_LA_eval.txt`; `experiments/paper_final/tdcf_recomputed.txt` | missing; protocol in archived `run_w2v_fullft.sh` | `experiments/paper_final/wav2vec2_fullft_multiseed/seed_123/models/epoch_38.pth` | official LA evaluator via Tak/SSL script | no config/manifest; full FT protocol differs from repo train.py; best checkpoint selected by validation loss |
| wav2vec2 | full | 123 | DF | 5.78 | | partial | `experiments/paper_final/wav2vec2_fullft_multiseed/seed_123/eval/scores_DF_eval.txt` | `experiments/paper_final/wav2vec2_fullft_multiseed/seed_123/eval/results_DF_eval.txt` | missing; protocol in archived `run_w2v_fullft.sh` | `experiments/paper_final/wav2vec2_fullft_multiseed/seed_123/models/epoch_38.pth` | official/project DF EER via Tak/SSL script | no config/manifest; DF EER only |
| wav2vec2 | full | 1234 | LA | 1.15 | 0.2164 | partial | `experiments/paper_final/wav2vec2_fullft_multiseed/seed_1234/eval/scores_LA_eval.txt` | `experiments/paper_final/wav2vec2_fullft_multiseed/seed_1234/eval/results_LA_eval.txt`; `experiments/paper_final/tdcf_recomputed.txt` | `experiments/paper_final/wav2vec2_fullft_multiseed/seed_1234/README.md` only | `experiments/paper_final/wav2vec2_fullft_multiseed/seed_1234/models/epoch_22.pth` | official LA evaluator via Tak/SSL script | README documents manual stop after epoch 32 and best epoch 22; no config/manifest |
| wav2vec2 | full | 1234 | DF | 3.29 | | partial | `experiments/paper_final/wav2vec2_fullft_multiseed/seed_1234/eval/scores_DF_eval.txt` | `experiments/paper_final/wav2vec2_fullft_multiseed/seed_1234/eval/results_DF_eval.txt` | `experiments/paper_final/wav2vec2_fullft_multiseed/seed_1234/README.md` only | `experiments/paper_final/wav2vec2_fullft_multiseed/seed_1234/models/epoch_22.pth` | official/project DF EER via Tak/SSL script | README documents manual stop after epoch 32 and best epoch 22; DF EER only |
| wav2vec2 | full | Tak2022 | LA | 1.01 | | verified_external | none local | Tak et al. published baseline cited in `docs/paper.tex` | external | external | published paper | Including this point with three local seeds reproduces the paper's LA 1.07±0.06, but it is not a local seed artifact. |
| wav2vec2 | full | Tak2022 | DF | 2.78 | | verified_external | none local | Tak et al. published baseline cited in `docs/paper.tex` | external | external | published paper | Including this point with three local seeds reproduces the paper's DF 4.00±1.32, but it is not a local seed artifact. |
| MiMo | frozen | 42 | LA | 6.9238 | 0.3550 | partial | `experiments/paper_final/mimo_frozen_multiseed/seed_42/eval/scores_LA_eval.txt` | `experiments/paper_final/mimo_frozen_multiseed/seed_42/eval/results_LA_eval.txt`; `experiments/paper_final/tdcf_recomputed.txt` | `experiments/paper_final/mimo_frozen_multiseed/seed_42/models/mimo_frozen_s42_seed42/config.yaml` | `experiments/paper_final/mimo_frozen_multiseed/seed_42/models/mimo_frozen_s42_seed42/epoch_1_eer_11.17.pth` | official LA tDCF recompute + project result | completed; project tDCF 0.0153 is wrong-scale |
| MiMo | frozen | 42 | DF | 13.3443 | | partial | `experiments/paper_final/mimo_frozen_multiseed/seed_42/eval/scores_DF_eval.txt` | `experiments/paper_final/mimo_frozen_multiseed/seed_42/eval/results_DF_eval.txt` | `experiments/paper_final/mimo_frozen_multiseed/seed_42/models/mimo_frozen_s42_seed42/config.yaml` | `experiments/paper_final/mimo_frozen_multiseed/seed_42/models/mimo_frozen_s42_seed42/epoch_1_eer_11.17.pth` | project EER result | completed; DF EER only |
| MiMo | frozen | 123 | LA | 5.7993 | 0.3282 | partial | `experiments/paper_final/mimo_frozen_multiseed/seed_123/eval/scores_LA_eval.txt` | `experiments/paper_final/mimo_frozen_multiseed/seed_123/eval/results_LA_eval.txt`; `experiments/paper_final/tdcf_recomputed.txt` | `experiments/paper_final/mimo_frozen_multiseed/seed_123/models/mimo_frozen_s123_seed123/config.yaml` | missing expected `epoch_6_eer_10.85.pth`; local dir has `epoch_10_eer_11.17.pth` | official LA tDCF recompute + project result | manifest completed and says best epoch 6; checkpoint file missing locally; project tDCF 0.0128 is wrong-scale |
| MiMo | frozen | 123 | DF | 12.3552 | | partial | `experiments/paper_final/mimo_frozen_multiseed/seed_123/eval/scores_DF_eval.txt` | `experiments/paper_final/mimo_frozen_multiseed/seed_123/eval/results_DF_eval.txt` | `experiments/paper_final/mimo_frozen_multiseed/seed_123/models/mimo_frozen_s123_seed123/config.yaml` | missing expected `epoch_6_eer_10.85.pth`; local dir has `epoch_10_eer_11.17.pth` | project EER result | manifest completed and says best epoch 6; checkpoint file missing locally; DF EER only |
| MiMo | frozen | 456 | LA | 11.2158 | 0.4294 | partial | `experiments/paper_final/mimo_frozen_multiseed/seed_456/eval/scores_LA_eval.txt` | `experiments/paper_final/mimo_frozen_multiseed/seed_456/eval/results_LA_eval.txt`; `experiments/paper_final/tdcf_recomputed.txt` | `experiments/paper_final/mimo_frozen_multiseed/seed_456/models/mimo_frozen_s456_seed456/config.yaml` | `experiments/paper_final/mimo_frozen_multiseed/seed_456/models/mimo_frozen_s456_seed456/epoch_0_eer_13.67.pth` | official LA tDCF recompute + project result | completed outlier; no failure evidence found; include by default; project tDCF 0.0244 is wrong-scale |
| MiMo | frozen | 456 | DF | 14.6808 | | partial | `experiments/paper_final/mimo_frozen_multiseed/seed_456/eval/scores_DF_eval.txt` | `experiments/paper_final/mimo_frozen_multiseed/seed_456/eval/results_DF_eval.txt` | `experiments/paper_final/mimo_frozen_multiseed/seed_456/models/mimo_frozen_s456_seed456/config.yaml` | `experiments/paper_final/mimo_frozen_multiseed/seed_456/models/mimo_frozen_s456_seed456/epoch_0_eer_13.67.pth` | project EER result | completed outlier; no failure evidence found; include by default; DF EER only |
| MiMo | frozen | 789 | LA | 6.0400 | 0.3296 | partial | `experiments/paper_final/mimo_frozen_multiseed/seed_789/eval/scores_LA_eval.txt` | `experiments/paper_final/mimo_frozen_multiseed/seed_789/eval/results_LA_eval.txt`; `experiments/paper_final/tdcf_recomputed.txt` | `experiments/paper_final/mimo_frozen_multiseed/seed_789/models/mimo_frozen_s789_seed789/config.yaml` | missing expected `epoch_5_eer_10.63.pth`; local dir has `epoch_3_eer_11.61.pth` | official LA tDCF recompute + project result | manifest completed and says best epoch 5; checkpoint file missing locally; project tDCF 0.0134 is wrong-scale |
| MiMo | frozen | 789 | DF | 12.2597 | | partial | `experiments/paper_final/mimo_frozen_multiseed/seed_789/eval/scores_DF_eval.txt` | `experiments/paper_final/mimo_frozen_multiseed/seed_789/eval/results_DF_eval.txt` | `experiments/paper_final/mimo_frozen_multiseed/seed_789/models/mimo_frozen_s789_seed789/config.yaml` | missing expected `epoch_5_eer_10.63.pth`; local dir has `epoch_3_eer_11.61.pth` | project EER result | manifest completed and says best epoch 5; checkpoint file missing locally; DF EER only |
| MiMo | frozen | 1234 | LA | 5.5946 | | partial | missing | `experiments/paper_final/mimo_frozen_multiseed/seed_1234/eval/results_LA_eval.txt` | missing | missing local checkpoint; result cites missing `outputs/2026-02-08/02-05-01/.../epoch_1_eer_9.78.pth` | project result only | no score file/config/manifest/checkpoint found; cannot recompute official tDCF |
| MiMo | frozen | 1234 | DF | 11.6748 | | partial | missing | `experiments/paper_final/mimo_frozen_multiseed/seed_1234/eval/results_DF_eval.txt` | missing | missing local checkpoint; result cites missing `outputs/2026-02-08/02-05-01/.../epoch_1_eer_9.78.pth` | project result only | no score file/config/manifest/checkpoint found; DF EER only |
| MiMo | adapter | 42 | LA | 4.4053 | 0.2979 | partial | `experiments/eval_2021_v2/scores_LA_eval.txt` | `experiments/eval_2021_v2/results_LA_eval.txt`; command output from `SSL_Anti-spoofing/evaluate_2021_LA.py` | `experiments/optuna/mimo_hpo_v2/trial_0039/config.yaml` | `experiments/optuna/mimo_hpo_v2/trial_0039/best_model.pt` | official LA tDCF command + project result | Optuna trial uses fixed RNG seed 42 in `optuna_train.py`; saved config says seed 1234, so seed provenance is imperfect; project tDCF 0.0098 is wrong-scale |
| MiMo | adapter | 42 | DF | 11.7145 | | partial | `experiments/eval_2021_v2/scores_DF_eval.txt` | `experiments/eval_2021_v2/results_DF_eval.txt` | `experiments/optuna/mimo_hpo_v2/trial_0039/config.yaml` | `experiments/optuna/mimo_hpo_v2/trial_0039/best_model.pt` | project EER result | Optuna trial 39; DF EER only |
| MiMo | adapter | 2024 | LA | 4.3700 | 0.2966 | partial | `experiments/eval_seed2024/scores_LA_eval.txt` | `experiments/eval_seed2024/results_LA_eval.txt`; command output from `SSL_Anti-spoofing/evaluate_2021_LA.py` | `configs/train_seed2024.yaml` portable recipe; exact resolved run config missing | missing local checkpoint; result cites missing `outputs/2026-01-27/18-01-37/.../epoch_4_eer_9.87.pth` | official LA tDCF command + project result | eval artifact exists; checkpoint/output run dir missing; project tDCF 0.0097 is wrong-scale |
| MiMo | adapter | 2024 | DF | 7.7014 | | partial | `experiments/eval_seed2024/scores_DF_eval.txt` | `experiments/eval_seed2024/results_DF_eval.txt` | `configs/train_seed2024.yaml` portable recipe; exact resolved run config missing | missing local checkpoint; result cites missing `outputs/2026-01-27/18-01-37/.../epoch_4_eer_9.87.pth` | project EER result | eval artifact exists; checkpoint/output run dir missing; DF EER only |
| MiMo | full | 42 | LA | 4.2797 | 0.2823 | partial | `experiments/paper_final/mimo_full_multiseed/seed_42/eval/scores_LA_eval.txt` | `experiments/paper_final/mimo_full_multiseed/seed_42/eval/results_LA_eval.txt`; `experiments/paper_final/tdcf_recomputed.txt` | `experiments/paper_final/mimo_full_multiseed/seed_42/models/mimo_full_s42_seed42/config.yaml` | missing expected `epoch_12_eer_9.87.pth`; local dir has `epoch_11_eer_9.98.pth` | official LA tDCF recompute + project result | manifest completed and says best epoch 12; checkpoint file missing locally; project tDCF 0.0093 is wrong-scale |
| MiMo | full | 42 | DF | 14.0420 | | partial | `experiments/paper_final/mimo_full_multiseed/seed_42/eval/scores_DF_eval.txt` | `experiments/paper_final/mimo_full_multiseed/seed_42/eval/results_DF_eval.txt` | `experiments/paper_final/mimo_full_multiseed/seed_42/models/mimo_full_s42_seed42/config.yaml` | missing expected `epoch_12_eer_9.87.pth`; local dir has `epoch_11_eer_9.98.pth` | project EER result | manifest completed and says best epoch 12; checkpoint file missing locally; DF EER only |
| MiMo | full | 123 | LA | 5.2646 | 0.3094 | partial | `experiments/paper_final/mimo_full_multiseed/seed_123/eval/scores_LA_eval.txt` | `experiments/paper_final/mimo_full_multiseed/seed_123/eval/results_LA_eval.txt`; `experiments/paper_final/tdcf_recomputed.txt` | `experiments/paper_final/mimo_full_multiseed/seed_123/models/mimo_full_s123_seed123/config.yaml` | `experiments/paper_final/mimo_full_multiseed/seed_123/models/mimo_full_s123_seed123/epoch_1_eer_9.98.pth` | official LA tDCF recompute + project result | manifest completed; project tDCF 0.0116 is wrong-scale |
| MiMo | full | 123 | DF | 12.4277 | | partial | `experiments/paper_final/mimo_full_multiseed/seed_123/eval/scores_DF_eval.txt` | `experiments/paper_final/mimo_full_multiseed/seed_123/eval/results_DF_eval.txt` | `experiments/paper_final/mimo_full_multiseed/seed_123/models/mimo_full_s123_seed123/config.yaml` | `experiments/paper_final/mimo_full_multiseed/seed_123/models/mimo_full_s123_seed123/epoch_1_eer_9.98.pth` | project EER result | manifest completed; DF EER only |
| MiMo | full | 456 | LA | 8.5484 | 0.3886 | partial | `experiments/paper_final/mimo_full_multiseed/seed_456/eval/scores_LA_eval.txt` | `experiments/paper_final/mimo_full_multiseed/seed_456/eval/results_LA_eval.txt`; `experiments/paper_final/tdcf_recomputed.txt` | `experiments/paper_final/mimo_full_multiseed/seed_456/models/mimo_full_s456_seed456/config.yaml` | `experiments/paper_final/mimo_full_multiseed/seed_456/models/mimo_full_s456_seed456/epoch_0_eer_12.02.pth` | official LA tDCF recompute + project result | manifest status `running`; train log stops after epoch 6; resume script says training was interrupted near early stop and eval-only used best existing checkpoint; project tDCF 0.0190 is wrong-scale |
| MiMo | full | 456 | DF | 13.4204 | | partial | `experiments/paper_final/mimo_full_multiseed/seed_456/eval/scores_DF_eval.txt` | `experiments/paper_final/mimo_full_multiseed/seed_456/eval/results_DF_eval.txt` | `experiments/paper_final/mimo_full_multiseed/seed_456/models/mimo_full_s456_seed456/config.yaml` | `experiments/paper_final/mimo_full_multiseed/seed_456/models/mimo_full_s456_seed456/epoch_0_eer_12.02.pth` | project EER result | manifest status `running`; train log stops after epoch 6; resume script says training was interrupted near early stop and eval-only used best existing checkpoint; DF EER only |
| MiMo | full | 789 | LA | 7.9519 | 0.3718 | partial | `experiments/paper_final/mimo_full_multiseed/seed_789/eval/scores_LA_eval.txt` | `experiments/paper_final/mimo_full_multiseed/seed_789/eval/results_LA_eval.txt`; `experiments/paper_final/tdcf_recomputed.txt` | `experiments/paper_final/mimo_full_multiseed/seed_789/models/mimo_full_s789_seed789/config.yaml` | missing expected `epoch_1_eer_11.84.pth`; local dir has `epoch_0_eer_12.37.pth` | official LA tDCF recompute + project result | manifest completed and says best epoch 1; result/local checkpoint use epoch 0; project tDCF 0.0173 is wrong-scale |
| MiMo | full | 789 | DF | 12.8647 | | partial | `experiments/paper_final/mimo_full_multiseed/seed_789/eval/scores_DF_eval.txt` | `experiments/paper_final/mimo_full_multiseed/seed_789/eval/results_DF_eval.txt` | `experiments/paper_final/mimo_full_multiseed/seed_789/models/mimo_full_s789_seed789/config.yaml` | missing expected `epoch_1_eer_11.84.pth`; local dir has `epoch_0_eer_12.37.pth` | project EER result | manifest completed and says best epoch 1; result/local checkpoint use epoch 0; DF EER only |
| MiMo | full | 1234 | LA | 8.6448 | 0.3957 | partial | `experiments/eval_mimo_full/scores_LA_eval.txt` | `experiments/eval_mimo_full/results_LA_eval.txt`; command output from `SSL_Anti-spoofing/evaluate_2021_LA.py` | missing | missing local checkpoint; result cites `outputs/2026-02-08/05-20-18/.../epoch_1_eer_13.34.pth` | official LA tDCF command + project result | outside `paper_final`; checkpoint/config not found; project tDCF 0.0191 is wrong-scale; official tDCF computed during audit |
| MiMo | full | 1234 | DF | 10.9228 | | partial | `experiments/eval_mimo_full/scores_DF_eval.txt` | `experiments/eval_mimo_full/results_DF_eval.txt` | missing | missing local checkpoint; result cites `outputs/2026-02-08/05-20-18/.../epoch_1_eer_13.34.pth` | project EER result | outside `paper_final`; checkpoint/config not found; DF EER only |

## Audit commands log

Record commands used to populate this file so the ledger is reproducible.

```bash
# 2026-05-25: first-pass main-table artifact inventory.
find experiments/paper_final -maxdepth 4 -type f \
  \( -iname '*score*' -o -iname '*result*' -o -name 'config.yaml' \
     -o -name 'manifest.json' -o -name '*.log' -o -name '*.pth' \
     -o -name '*.pt' -o -name '*.csv' -o -name '*.txt' \) | sort
find experiments/paper_final -maxdepth 2 -type d | sort
find experiments outputs -path '*wav2vec2_adapter*seed1234*' \
  -o -path '*w2v2_adapter*1234*' -o -path '*adapter*seed1234*' | sort
find experiments outputs logs -iname '*mimo*adapter*' -o -iname '*adapter*mimo*' | sort
grep -RIl "mimo.*adapter\|adapter.*mimo\|strategy: adapter" experiments outputs logs configs docs scripts 2>/dev/null
# Summary written to docs/current/ARTIFACT_INVENTORY.md.

# 2026-05-25: official LA tDCF for external wav2vec2 adapter seed 1234.
cd SSL_Anti-spoofing
python evaluate_2021_LA.py ../experiments/eval_wav2vec2_adapter/scores_LA_eval.txt \
  <datasets>/ASVspoof2021_LA_eval eval
# min_tDCF: 0.2733
# eer: 3.69

# 2026-05-25: wav2vec2 adapter aggregate check.
# LA EER mean/sample std over seeds 1234,42,123,456,789: 2.77048 / 0.81284.
# DF EER mean/sample std over seeds 1234,42,123,456,789: 5.10522 / 0.71992.
# LA tDCF mean over five official values: 0.25528; over four paper_final seeds: 0.25078.

# 2026-05-25: wav2vec2 full FT aggregate check.
# Local reproduced seeds 42,123,1234:
#   LA EER mean/sample std: 1.09333 / 0.06028.
#   DF EER mean/sample std: 4.41333 / 1.26271.
#   LA tDCF mean/sample std: 0.21493 / 0.00197.
# Adding Tak et al. published point LA=1.01, DF=2.78:
#   LA EER mean/sample std: 1.07250 / 0.06449.
#   DF EER mean/sample std: 4.00500 / 1.31526.

# 2026-05-25: MiMo frozen seed inclusion check.
# All five seeds 1234,42,123,456,789:
#   LA EER mean/sample std: 7.11470 / 2.34795.
#   DF EER mean/sample std: 12.86296 / 1.18001.
# Excluding seed 456:
#   LA EER mean/sample std: 6.08943 / 0.58528.
#   DF EER mean/sample std: 12.40850 / 0.69258.
# Official LA tDCF mean over seeds 42,123,456,789: 0.36055.
# Official LA tDCF mean excluding seed 456 over seeds 42,123,789: 0.33760.
# Search for seed-456 failure/corruption evidence:
# grep -RIn --exclude='scores_*' --exclude='*.tfevents.*' \
#   -E '456|outlier|fail|error|interrupt|corrupt|exclude|bad|wrong' \
#   experiments/paper_final/mimo_frozen_multiseed docs local_private/.../paper_final
# Result: no seed-456-specific failure/corruption note found; only routine audio loading warnings.

# 2026-05-25: MiMo full FT aggregate check.
# Local paper_final seeds 42,123,456,789:
#   LA EER mean/sample std: 6.51115 / 2.06231.
#   DF EER mean/sample std: 13.18870 / 0.69902.
#   LA tDCF mean/sample std: 0.33803 / 0.05041.
# Adding original/external seed 1234 from experiments/eval_mimo_full:
#   LA EER mean/sample std: 6.93788 / 2.02493.
#   DF EER mean/sample std: 12.73552 / 1.18040.
cd SSL_Anti-spoofing
python evaluate_2021_LA.py ../experiments/eval_mimo_full/scores_LA_eval.txt \
  <datasets>/ASVspoof2021_LA_eval eval
# min_tDCF: 0.3957
# eer: 8.64
# All-five LA tDCF mean including seed 1234: 0.34956.

# 2026-05-25: MiMo adapter blocker audit.
find experiments outputs logs -type f | grep -Ei 'seed2024|trial39|multi_seed|mimo.*adapter|adapter.*mimo'
python - <<'PY'
from pathlib import Path
import re
for f in list(Path('experiments').rglob('results_*_eval.txt')) + list(Path('outputs').rglob('results_*_eval.txt')):
    txt=f.read_text(errors='ignore')
    ck=re.search(r'Checkpoint:\\s*(.*)', txt)
    if ck and ('mimo' in ck.group(1).lower() or 'trial_0039' in ck.group(1).lower() or 'optuna' in ck.group(1).lower()):
        print(f, ck.group(1))
PY
cd SSL_Anti-spoofing
python evaluate_2021_LA.py ../experiments/eval_2021_v2/scores_LA_eval.txt \
  <datasets>/ASVspoof2021_LA_eval eval
# min_tDCF: 0.2979
# eer: 4.41
python evaluate_2021_LA.py ../experiments/eval_seed2024/scores_LA_eval.txt \
  <datasets>/ASVspoof2021_LA_eval eval
# min_tDCF: 0.2966
# eer: 4.37
# Observed MiMo adapter full-eval n=2 stats:
#   LA EER mean/sample std: 4.38765 / 0.02496.
#   DF EER mean/sample std: 9.70795 / 2.83769.
#   LA tDCF mean/sample std: 0.29725 / 0.00092.
# No local artifact set was found for the paper's claimed n=5 LA 4.64±0.26 / DF 10.15±1.70.
```
