# Corrected main-table draft

Purpose: replacement starting point for `docs/paper.tex` Table 1. This is evidence-first, not yet paper prose. Values use sample standard deviation for EER. LA min-tDCF values follow `TDCF_RECONCILIATION.md`; project-generated wrong-scale tDCF values are ignored.

## Recommended artifact-backed table

| Model | Strategy | n/source | LA EER (%) | LA min-tDCF | DF EER (%) | Status | Notes |
|---|---|---:|---:|---:|---:|---|---|
| wav2vec2 | Frozen | 5 local | 8.05 ± 0.73 | 0.384 | 6.76 ± 0.61 | partial | Metric artifacts complete; configs/manifests missing. |
| wav2vec2 | Adapter | 5 = 4 local + seed1234 external run | 2.77 ± 0.81 | 0.255 | 5.11 ± 0.72 | partial | EER matches paper; tDCF should be 0.255 if seed1234 is included. Checkpoint gaps for seeds 42/789. |
| wav2vec2 | Full FT | 3 local reproduced | 1.09 ± 0.06 | 0.215 | 4.41 ± 1.26 | partial | Do not describe as 4 local seeds. Paper value requires adding Tak et al. external point. |
| MiMo | Frozen | 5 found evals | 7.11 ± 2.35 | 0.361* | 12.86 ± 1.18 | invalid as paper row | Includes seed 456 by default. `*` tDCF is n=4 because seed1234 score file is missing. |
| MiMo | Adapter | 2 found evals | 4.39 ± 0.02 | 0.297 | 9.71 ± 2.84 | exploratory | Missing n=5 artifacts assumed unavailable. Paper n=5 mean/std unsupported; no significance claims. |
| MiMo | Full FT | 5 = 4 local + seed1234 external run | 6.94 ± 2.02 | 0.350 | 12.74 ± 1.18 | partial | EER matches paper only with seed1234; tDCF should be 0.350 if seed1234 is included. Seed/checkpoint caveats remain. |

## Optional external-reference variant

If the paper keeps Tak et al.'s published wav2vec2 full-FT result as an external reference point, state that explicitly instead of calling it a seed:

| Model | Strategy | n/source | LA EER (%) | LA min-tDCF | DF EER (%) | Notes |
|---|---|---:|---:|---:|---:|---|
| wav2vec2 | Full FT + Tak ref. | 3 local + 1 published external | 1.07 ± 0.06 | 0.215† | 4.00 ± 1.32 | `†` tDCF still local n=3 only; Tak tDCF not locally available. |

## Rows that must not be reused unchanged

- **MiMo frozen `6.09 ± 0.58`, `12.41 ± 0.69`**: this excludes seed 456 without failure evidence. It may be reported only as a sensitivity/exclusion analysis.
- **MiMo adapter `4.64 ± 0.26`, `10.15 ± 1.70`**: no local artifact set supports n=5 or these statistics.
- **MiMo full tDCF `0.338` with EER `6.94 ± 2.02`**: EER is all-five including seed1234; `0.338` is four local seeds only. Use `0.350` for all-five or make the n mismatch explicit.
- **wav2vec2 adapter tDCF `0.251` with EER n=5**: `0.251` is four `paper_final` seeds only. Use `0.255` if seed1234 is included.
- **wav2vec2 full as `4 seeds`**: local artifacts have three reproduced seeds; the fourth point is Tak et al. external.

## Scientific consequence

This table weakens the current narrative:

- wav2vec2 still improves monotonically in the artifact-backed rows.
- MiMo frozen is worse on LA when all completed seeds are included (`7.11`) than when seed 456 is excluded (`6.09`).
- MiMo adapter is now an exploratory n=2 result unless a future controlled rerun is approved.
- MiMo full is internally inconsistent until seed1234/checkpoint provenance and tDCF aggregation are stated clearly.
