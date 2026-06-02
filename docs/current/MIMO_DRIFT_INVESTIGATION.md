# MiMo drift investigation

Date: 2026-05-26
Scope: bounded reproduction checks for MiMo frozen seed 42.

## Question

Can the current `mimodf eval run` path reproduce historical MiMo frozen score files closely enough to trust fresh MiMo reruns as research evidence?

## Subject

Historical row:

- model: MiMo frozen
- seed: 42
- checkpoint: `experiments/paper_final/mimo_frozen_multiseed/seed_42/models/mimo_frozen_s42_seed42/epoch_1_eer_11.17.pth`
- resolved config: `experiments/paper_final/mimo_frozen_multiseed/seed_42/models/mimo_frozen_s42_seed42/config.yaml`
- historical score file: `experiments/paper_final/mimo_frozen_multiseed/seed_42/eval/scores_LA_eval.txt`

Key reconstructed frontend facts:

- sample rate: 24 kHz
- cut: 96000
- `use_bfloat16: true`
- `native_50hz: true`
- feature: continuous
- finetune strategy: frozen
- historical eval batch size in config: 64

## Results

### Stage localization

A diagnostic compared the first utterance (`LA_E_9332881`) evaluated alone vs as part of larger batches.

For native-50Hz historical reconstruction:

- raw mel spectrograms are identical: `max_abs_diff = 0.0`;
- bf16-cast mel spectrograms are identical: `max_abs_diff = 0.0`;
- drift first appears in MiMo encoder features;
- projection/backend logits inherit and amplify the feature drift.

Representative single-vs-batch-64 numbers for the same utterance:

| Stage | max abs diff | mean abs diff |
|---|---:|---:|
| mel raw | 0.0 | 0.0 |
| mel bf16 | 0.0 | 0.0 |
| MiMo features | 0.02344 | 1.02e-4 |
| projected features | 0.00642 | 4.97e-5 |
| logits | 0.02542 | 0.01860 |

The same drift pattern appears when the batch is made of repeated copies of the same utterance. Therefore the effect is batch-size/kernel-layout dependent, not caused by different neighboring utterance content.

### Determinism

Current MiMo eval is deterministic for fixed batch size and command:

- batch size 4, 100 utterances, repeat vs repeat: `max_abs_diff = 0.0`
- batch size 64, 100 utterances, repeat vs repeat: `max_abs_diff = 0.0`

So the drift is not random run-to-run noise.

### Batch-size sensitivity

Current MiMo scores change materially with eval batch size:

| Comparison | common | mean abs diff | max abs diff |
|---|---:|---:|---:|
| current batch 64 vs current batch 4 | 100 | 0.02637 | 0.11279 |

This is much larger than the wav2vec2 reproduction drift and suggests MiMo inference is batch-size sensitive. Because duplicated-utterance batches show the same behavior and mel inputs are identical, padding/audio loading is unlikely to be the root. The most likely cause is bf16 FlashAttention/varlen kernel numerics/layout in the MiMo encoder, with backend amplification.

### Historical reproduction

| Current command | common | mean abs diff vs historical | max abs diff vs historical |
|---|---:|---:|---:|
| batch size 4, first 100 | 100 | 0.02671 | 0.11260 |
| batch size 64, first 100 | 100 | 0.01447 | 0.06320 |

Batch size 64 is closer to historical, consistent with the historical config's `eval_batch_size: 64`, but drift remains non-trivial.

### Native-50Hz variant check

A diagnostic temporary config with `native_50hz: false` was tested. This is **not** a valid historical reproduction because the checkpoint/backend were trained with native-50Hz geometry, but it localizes the issue.

The standard MiMo 25Hz path also showed batch-size sensitivity, and much larger feature/logit changes. This argues against the local native-50Hz patch being the sole root cause. The issue likely lives in the MiMo encoder/FlashAttention path generally, while native-50Hz only changes the magnitude and downstream compatibility.

### Precision variant

A float32 MiMo run was attempted by changing `use_bfloat16: false` in a temporary config. It failed because the MiMo FlashAttention path requires fp16/bf16:

```text
RuntimeError: FlashAttention only support fp16 and bf16 data type
```

So fp32 is not a simple control without changing the MiMo attention implementation.

## Interpretation

MiMo is runnable through the new harness, but MiMo historical reproduction is not yet trustworthy at the score level.

The strongest finding is batch-size sensitivity in the MiMo encoder itself: the same current checkpoint/config/audio produces different features and scores when batch size changes. That can explain part of the historical drift and means future MiMo evaluations must pin batch size as a protocol fact.

This does not invalidate the historical MiMo table by itself, because historical metrics were produced by a consistent run. It does mean fresh MiMo reruns cannot be mixed with old MiMo scores until the batch-size/native-50Hz behavior is understood or explicitly controlled.

## Working hypotheses

1. FlashAttention varlen bf16 outputs vary with batch size/kernel layout enough to affect backend scores.
2. The backend amplifies small encoder feature differences into larger score differences.
3. Historical MiMo code/environment may have differed subtly from current local code despite matching high-level config.
4. Historical score files were likely generated with batch size 64, explaining why current batch 64 is closer than batch 4.

## Next checks

Recommended order:

1. If we need a fix, investigate whether MiMo can run with a deterministic/math SDPA attention path instead of FlashAttention; this is dependency-risky and should be isolated.
2. Pin eval batch size in all future MiMo manifests and research tables, especially `eval_batch_size: 64` when comparing to historical runs.
3. Do not run full MiMo evals for evidence until this drift is either fixed or accepted as a documented protocol caveat.
4. For any fresh MiMo result, report batch size and avoid mixing with historical score files generated under a different batch size.

## Current decision

MiMo fresh reruns are allowed for smoke/investigation, but not yet for paper claims or direct replacement of historical MiMo scores.
