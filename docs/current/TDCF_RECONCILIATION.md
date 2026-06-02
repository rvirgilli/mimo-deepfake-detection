# LA min-tDCF reconciliation

Purpose: decide which LA min-tDCF values are valid for the main table. This is assessment evidence, not manuscript prose.

Policy:

- Use official ASVspoof LA evaluator output only.
- Ignore project-generated wrong-scale `min t-DCF` values in many `results_LA_eval.txt` files.
- A table-row tDCF is clean only when computed over the same seed set as the row's EER.
- If same-seed tDCF is unavailable, mark the row with an explicit `n`/source footnote rather than silently mixing aggregates.

## Main-table reconciliation

| Row | EER seed/source set | Paper tDCF | Supported same-set tDCF | Status | Decision |
|---|---|---:|---:|---|---|
| wav2vec2 frozen | 5 local seeds: 42, 123, 456, 789, 1234 | 0.384 | 0.3844 | valid | Same seed set. Keep 0.384 if row remains n=5. |
| wav2vec2 adapter | 5 seeds: 42, 123, 456, 789 + external/local run 1234 | 0.251 | 0.2553 | mismatch | Paper 0.251 is four `paper_final` seeds only. Use 0.255 if EER stays n=5. |
| wav2vec2 full FT | 3 local reproduced seeds, or 3 local + Tak external point | 0.215 | 0.2149 for local n=3 | partial | 0.215 is valid for local n=3. If EER includes Tak external point, tDCF is not same-set because Tak tDCF/scores are not local. |
| MiMo frozen | 5 found evals if seed 456 included by policy | 0.338 | unavailable for all 5; 0.3606 for local score n=4 excluding seed1234 | mismatch | Paper 0.338 is seeds 42/123/789 only. Same-set all-five tDCF cannot be computed because seed1234 score file is missing. Use `0.361*` with n=4 footnote if reporting all completed EER seeds. |
| MiMo adapter | 2 found evals: trial39/42-ish, 2024 | --- | 0.2973 | valid for n=2 | If reporting n=2 adapter row, tDCF can be reported as 0.297. Paper n=5 row remains unsupported. |
| MiMo full FT | 5 seeds: 42, 123, 456, 789 + external/original 1234 | 0.338 | 0.3496 | mismatch | Paper 0.338 is four local seeds only. Use 0.350 if EER stays n=5. |

## Seed-level official tDCF values used

### wav2vec2 frozen

| Seed | tDCF | Source |
|---:|---:|---|
| 42 | 0.3867 | `experiments/paper_final/tdcf_recomputed.txt` |
| 123 | 0.3980 | `experiments/paper_final/tdcf_recomputed.txt` |
| 456 | 0.4037 | `experiments/paper_final/tdcf_recomputed.txt` |
| 789 | 0.3545 | `experiments/paper_final/tdcf_recomputed.txt` |
| 1234 | 0.3791 | `experiments/paper_final/tdcf_recomputed.txt` |

Mean: `0.3844`.

### wav2vec2 adapter

| Seed | tDCF | Source |
|---:|---:|---|
| 42 | 0.2707 | `experiments/paper_final/tdcf_recomputed.txt` |
| 123 | 0.2458 | `experiments/paper_final/tdcf_recomputed.txt` |
| 456 | 0.2582 | `experiments/paper_final/tdcf_recomputed.txt` |
| 789 | 0.2284 | `experiments/paper_final/tdcf_recomputed.txt` |
| 1234 | 0.2733 | audit command: `SSL_Anti-spoofing/evaluate_2021_LA.py ../experiments/eval_wav2vec2_adapter/scores_LA_eval.txt ...` |

Mean over four `paper_final` seeds: `0.2508` -> paper 0.251.
Mean over all five EER seeds: `0.2553`.

### wav2vec2 full FT

| Seed | tDCF | Source |
|---:|---:|---|
| 42 | 0.2157 | `experiments/paper_final/tdcf_recomputed.txt` |
| 123 | 0.2127 | `experiments/paper_final/tdcf_recomputed.txt` |
| 1234 | 0.2164 | `experiments/paper_final/tdcf_recomputed.txt` |
| Tak2022 | missing | external published EER point only |

Mean over local n=3: `0.2149` -> paper 0.215.

### MiMo frozen

| Seed | tDCF | Source |
|---:|---:|---|
| 42 | 0.3550 | `experiments/paper_final/tdcf_recomputed.txt` |
| 123 | 0.3282 | `experiments/paper_final/tdcf_recomputed.txt` |
| 456 | 0.4294 | `experiments/paper_final/tdcf_recomputed.txt` |
| 789 | 0.3296 | `experiments/paper_final/tdcf_recomputed.txt` |
| 1234 | missing | no local score file found for official recompute |

Mean over local score n=4: `0.3606`.
Mean excluding seed 456 and seed1234: `0.3376` -> paper 0.338.

### MiMo adapter

| Seed | tDCF | Source |
|---:|---:|---|
| 42-ish / trial39 | 0.2979 | audit command: `SSL_Anti-spoofing/evaluate_2021_LA.py ../experiments/eval_2021_v2/scores_LA_eval.txt ...` |
| 2024 | 0.2966 | audit command: `SSL_Anti-spoofing/evaluate_2021_LA.py ../experiments/eval_seed2024/scores_LA_eval.txt ...` |

Mean over found n=2: `0.2973`.

### MiMo full FT

| Seed | tDCF | Source |
|---:|---:|---|
| 42 | 0.2823 | `experiments/paper_final/tdcf_recomputed.txt` |
| 123 | 0.3094 | `experiments/paper_final/tdcf_recomputed.txt` |
| 456 | 0.3886 | `experiments/paper_final/tdcf_recomputed.txt` |
| 789 | 0.3718 | `experiments/paper_final/tdcf_recomputed.txt` |
| 1234 | 0.3957 | audit command: `SSL_Anti-spoofing/evaluate_2021_LA.py ../experiments/eval_mimo_full/scores_LA_eval.txt ...` |

Mean over four `paper_final` seeds: `0.3380` -> paper 0.338.
Mean over all five EER seeds: `0.3496`.

## Wrong-scale project tDCF examples

These project result files should not be used for table tDCF:

- `experiments/paper_final/wav2vec2_adapter_multiseed/seed_42/eval/results_LA_eval.txt`: project `0.0073`, official `0.2707`.
- `experiments/paper_final/mimo_frozen_multiseed/seed_42/eval/results_LA_eval.txt`: project `0.0153`, official `0.3550`.
- `experiments/paper_final/mimo_full_multiseed/seed_42/eval/results_LA_eval.txt`: project `0.0093`, official `0.2823`.
- `experiments/eval_2021_v2/results_LA_eval.txt`: project `0.0098`, official `0.2979`.

## Immediate fixes required

1. Replace wav2vec2 adapter tDCF `0.251` with `0.255` if row reports all five EER seeds.
2. Replace MiMo full tDCF `0.338` with `0.350` if row reports all five EER seeds.
3. Do not report MiMo frozen tDCF as all-five; use `0.361*` with `* n=4 score files only`, or recover seed1234 score file.
4. If wav2vec2 full FT keeps Tak external EER point, footnote that tDCF is local n=3 only.
5. If MiMo adapter is reported as n=2, add tDCF `0.297`; do not leave `---` unless intentionally omitting tDCF for underpowered rows.
