# Assessment conclusion

Purpose: freeze the current paper assessment before any manuscript rewrite or new training.

## Bottom line

The earlier manuscript is salvageable as a narrower, evidence-audited case study, but the old main table and several headline claims cannot be reused unchanged.

The assessment phase is now complete enough to make decisions:

- report only artifact-backed numbers;
- make seed/source sets explicit;
- mark MiMo adapter as exploratory n=2 unless a future controlled rerun is approved;
- include MiMo frozen seed 456 by default;
- report repo-native optimizer/protocol caveats honestly;
- avoid causal objective-class and feature-distortion claims unless extended analyses support them.

## Evidence package

Use these docs as the assessment source of truth:

| Doc | Role |
|---|---|
| `RESULTS_PROVENANCE.md` | Seed-level metric ledger. |
| `CORRECTED_MAIN_TABLE_DRAFT.md` | Corrected table values and rows not to reuse. |
| `TDCF_RECONCILIATION.md` | Official LA tDCF values and seed-set mismatches. |
| `VALIDATION_PROTOCOL_MATRIX.md` | Training/validation/checkpoint-selection protocol facts. |
| `CHECKPOINT_PROVENANCE_GAPS.md` | Missing/mismatched checkpoint and manifest gaps. |
| `MIMO_ADAPTER_DECISION.md` | Decision to treat missing adapter n=5 artifacts as unavailable for assessment. |
| `PAPER_ASSESSMENT_NEXT_STEPS.md` | Fix/improve/extend plan. |
| `DECISION_LOG.md` | Accepted assessment decisions. |

## Go / no-go by main table row

| Row | Assessment decision | Use in revised table? | Required caveat |
|---|---|---|---|
| wav2vec2 frozen | Mostly artifact-backed metrics; protocol artifacts incomplete. | Go, partial provenance. | Config/log/manifest missing; state metric artifacts are available. |
| wav2vec2 adapter | EER n=5 supported if including external seed1234; tDCF must be 0.255 not 0.251. | Go, partial provenance. | Mixed artifact locations; checkpoint gaps for seeds 42/789. |
| wav2vec2 full FT | Local n=3 supported; old n=4 claim only works by adding Tak external point. | Go as local n=3, or explicit external-reference variant. | Do not call Tak a local seed; full-FT protocol differs from repo-native runs. |
| MiMo frozen | Old row is invalid because it excludes seed 456. | Go only with all found seeds. | Use LA 7.11±2.35 / DF 12.86±1.18; tDCF is n=4 score files only unless seed1234 scores recovered. |
| MiMo adapter | Old n=5 row unsupported; missing artifacts assumed unavailable. | Go only as exploratory n=2. | No significance/low-variance claims; future n=5 requires controlled rerun. |
| MiMo full FT | EER n=5 matches only with external/original seed1234; tDCF must be 0.350 not 0.338. | Go with strong caveats, or demote. | Seed1234 config/checkpoint missing; seed456 interrupted; checkpoint gaps/mismatch. |

## Values to carry forward for assessment

| Model | Strategy | n/source | LA EER (%) | LA tDCF | DF EER (%) |
|---|---|---:|---:|---:|---:|
| wav2vec2 | Frozen | 5 local | 8.05 ± 0.73 | 0.384 | 6.76 ± 0.61 |
| wav2vec2 | Adapter | 5 = 4 local + seed1234 external run | 2.77 ± 0.81 | 0.255 | 5.11 ± 0.72 |
| wav2vec2 | Full FT | 3 local reproduced | 1.09 ± 0.06 | 0.215 | 4.41 ± 1.26 |
| MiMo | Frozen | 5 found evals | 7.11 ± 2.35 | 0.361* | 12.86 ± 1.18 |
| MiMo | Adapter | 2 found evals | 4.39 ± 0.02 | 0.297 | 9.71 ± 2.84 |
| MiMo | Full FT | 5 = 4 local + seed1234 external run | 6.94 ± 2.02 | 0.350 | 12.74 ± 1.18 |

`*` MiMo frozen tDCF is n=4 because seed1234 score file is missing.

## Claims to keep, weaken, or drop

### Keep, with caveats

- wav2vec2 improves strongly from frozen to adapter on LA and remains the cleanest adaptation trajectory.
- MiMo full fine-tuning is unstable/weak relative to wav2vec2 full, but full-FT protocols are confounded.
- DF remains difficult for MiMo across available rows.

### Weaken

- MiMo adapter benefit: present as exploratory n=2, not a statistically strong result.
- LA crossover: weakened because MiMo frozen all-seed LA is 7.11, not the selectively reported 6.09; still compare carefully if relevant.
- Regularization/learning-rate sweeps: single-seed controls, useful diagnostics only.

### Drop unless new evidence is added

- General contrastive-vs-reconstruction law.
- Feature distortion as proven fact.
- MiMo adapters are definitively critical.
- Any n=5 MiMo adapter claim.
- Any claim that repo-native runs used ASVspoof2019 dev checkpoint selection.
- Any claim that repo-native main-table runs used AdamW.

## Protocol decisions

- Repo-native main-table configs audited have `encoder_lr: null`; they used Adam, not AdamW.
- Repo-native paper-final logs/code show ASVspoof2021 fast eval subset for checkpoint selection, not ASVspoof2019 dev.
- wav2vec2 full FT uses the Tak/SSL stack and validation-loss checkpointing; it is not protocol-matched with MiMo full FT.
- LA tDCF must use official evaluator outputs and same seed set as EER, or carry an explicit footnote.

## Remaining gaps that do not block assessment

- wav2vec2 frozen configs/logs/manifests missing.
- MiMo frozen seed1234 score/config/checkpoint missing; tDCF all-five unavailable.
- MiMo full seed1234 config/checkpoint missing.
- MiMo full seed456 manifest incomplete/interrupted.
- Several best checkpoints cited by result files are missing locally.

These gaps block strong reproducibility claims, but not the assessment conclusion.

## Future work gates

### Allowed without training

- per-attack analysis from existing score files;
- validation-to-eval correlation from logs/manifests;
- representation drift/class-separation analysis if existing checkpoints load;
- artifact packaging and explicit score-file release plan.

### Requires explicit approval

- any MiMo adapter n=5 completion run;
- any repair run to avoid ASVspoof2021 fast-subset checkpoint selection;
- new encoders/baselines;
- broad HPO.

## Final assessment decision

Proceed to the next phase only with this framing:

> an audited, model-specific case study of wav2vec2/XLS-R and MiMo-Audio-Tokenizer under adaptation capacity, with explicit seed provenance, protocol caveats, and exploratory MiMo adapter evidence.

Do not proceed with the earlier manuscript's original headline narrative unchanged.
