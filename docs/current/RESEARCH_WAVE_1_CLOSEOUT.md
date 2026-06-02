# Research Wave 1 closeout

Date: 2026-05-30
Status: closed exploratory wave; diagnostic evidence, not paper-ready validation

## Wave-label clarification

Wave 1 is the broad exploratory map.

Some historical run IDs use a `wave2-...` prefix because the work originally started after the first Wave 1 source-holdout report. Do not rename those run IDs: they are immutable provenance. Conceptually, however, the WavLM comparator, log-mel baseline, and media-transform smoke complete the exploratory Wave 1 map.

That future **Wave 2** is now complete as the CLAMTTS deepening wave. **Wave 3** is planned targeted trained validation, not more broad broad frontend search.

## What Wave 1 did

Wave 1 mapped cheap frozen representations and explicit shifts on CodecFake+ CoSG:

- frontends:
  - wav2vec2/XLS-R;
  - WavLM-Base+;
  - MiMo continuous 25 Hz;
  - MiMo RVQ early/late/all histograms;
  - log-mel mean/std baseline;
- validations:
  - random row split diagnostics;
  - held-out-source probes;
  - taxonomy diagnostics;
  - RVQ early/late/all ablation;
  - score fusion/error overlap;
  - CLAMTTS-vs-NS/UNIAUDIO error diagnostics;
  - tiny media-transform generation + log-mel paired drift.

## What Wave 1 did not do

No full model training was run.

The sklearn logistic regressions in feature probes are linear diagnostic heads over cached frozen features. They count as feature-probe fitting, not frontend/backend training and not deployable model training.

No claim-bearing external evaluation was run:

- no full ASVspoof LA/DF eval;
- no end-to-end fine-tuning;
- no Optuna/HPO;
- no broad CoRS use;
- no robustness EER from transformed audio;
- no paper table update.

So if “real training/tests” means production training/eval, Wave 1 has none. If “tests” means scientific validation, Wave 1 has grouped source-holdout probes and transform drift diagnostics.

## Main evidence

### Source-holdout binary probes

Nine CoSG sources with enough bonafide/spoof support were held out.

| Frontend | Mean EER | Mean balanced acc | Mean AUROC | Best-by-EER count |
|---|---:|---:|---:|---:|
| wav2vec2/XLS-R | 0.193 | 0.776 | 0.873 | 7/9 in five-way comparison |
| MiMo continuous 25 Hz | 0.294 | 0.697 | 0.747 | 1/9 |
| MiMo RVQ all | 0.309 | 0.682 | 0.763 | 0/9 |
| WavLM-Base+ | 0.324 | 0.660 | 0.715 | 0/9 |
| log-mel mean/std | 0.362 | 0.637 | 0.682 | 1/9 |

Original three-way Wave 1 count was XLS-R `8/9`, MiMo continuous `1/9`, MiMo RVQ all `0/9`. Adding WavLM and log-mel did not change the broad conclusion: XLS-R remains the strongest transferring frontend in this slice.

### Conditional exception

CLAMTTS remains the important exception:

- MiMo continuous beats XLS-R on CLAMTTS;
- WavLM also improves over XLS-R on CLAMTTS;
- the effect does not generalize to NS2/NS3/UNIAUDIO.

Interpretation: CLAMTTS is a source-specific mechanism candidate, not evidence for a broad MiMo/tokenizer advantage.

### Media-transform smoke

Media-transform smoke generated 24 clean rows × 3 transforms:

- `resample_8k_16k`;
- `mp3_64k_16k`;
- `noise_snr20`.

Log-mel paired drift over 72 transformed rows:

| Transform | Pairs | Cosine mean | L2 mean | Mean abs delta |
|---|---:|---:|---:|---:|
| mp3_64k_16k | 24 | 0.99996 | 4.17 | 0.15 |
| noise_snr20 | 24 | 0.99155 | 125.73 | 8.89 |
| resample_8k_16k | 24 | 0.99024 | 73.58 | 2.62 |

This validates the transform pipeline and gives directional feature-drift evidence only. It is not a robustness metric and has no classifier scoring/EER.

## Decisions

Killed/deprioritized:

- MiMo-centered revival;
- MiMo superiority claim;
- broad MiMo training;
- Optuna/HPO;
- random-split headline metrics;
- naive taxonomy-causality claim;
- broad SSL-general claim from XLS-R alone.

Survives:

- representation-transfer/failure-mode framing;
- XLS-R as the current strong source-holdout reference;
- CLAMTTS as a real conditional case worth mechanistic analysis;
- media-transform axis as a valid next robustness/failure-mode direction;
- audit-first research system as a core contribution.

## Wave 2 was deep, not broad

Wave 2 did not add many more frontends. It picked the CLAMTTS mechanism and deepened it with sharper validation.

Wave 2 choices and closure are encoded in:

```text
docs/current/wave2_deepening_plan.yaml
```

Wave 2 result:

> The CLAMTTS reversal is real but source-local. The first low-energy/silence/RVQ-entropy mechanism did not predict NS2/NS3 behavior.

Wave 3 handoff:

> Use cheap probes as hypothesis selectors, then run targeted trained validation of XLS-R/source/media/CoRS hypotheses under `docs/current/wave3_training_validation_plan.yaml`.

## Claim-safe summary

Allowed:

> In a bounded CodecFake+ CoSG exploratory feature-probe wave, XLS-R transferred best across held-out sources. MiMo and WavLM showed source-conditional behavior, especially on CLAMTTS, but neither supported broad superiority claims. A tiny media-transform smoke validated the transform/provenance path and showed directional log-mel drift under noise/resampling, without classifier robustness metrics.

Not allowed:

- MiMo is better for deepfake detection.
- WavLM proves SSL generally fails or succeeds.
- Log-mel robustness metrics exist.
- Media transforms were evaluated with EER.
- Wave 1 trained a deployable detector.
