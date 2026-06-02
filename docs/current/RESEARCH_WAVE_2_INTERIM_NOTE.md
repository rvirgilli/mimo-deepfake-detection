# Research Wave 2 interim note

Date: 2026-05-31
Status: closed deepening wave; diagnostic evidence, not trained-model validation

## Purpose

Wave 2 tested whether the strongest Wave 1 conditional effect could become a predictive mechanism.

It did not. That is useful: it stops CLAMTTS-local curiosity from becoming an expensive training premise.

## What Wave 2 deepened

Wave 2 focused on the CLAMTTS reversal:

- XLS-R was the broad source-holdout reference in Wave 1.
- MiMo continuous and MiMo RVQ were much better than XLS-R on CLAMTTS.
- WavLM also improved over XLS-R on CLAMTTS, but not broadly.
- log-mel collapsed on CLAMTTS spoof recall.

Artifacts:

```text
experiments/runs/wave2_clamtts_mechanism_v1/report.md
experiments/runs/wave2_clamtts_case_contrast_v1/report.md
experiments/runs/wave2_clamtts_hypothesis_source_validation_v1/report.md
```

## Main result

CLAMTTS is real but source-local.

| Source | Records | XLS-R wrong | MiMo-cont wrong | MiMo fixes XLS-R | XLS-R fixes MiMo |
|---|---:|---:|---:|---:|---:|
| CLAMTTS | 119 | 32 | 13 | 26 | 7 |
| NS2 | 46 | 4 | 23 | 0 | 19 |
| NS3 | 64 | 13 | 33 | 4 | 24 |

The CLAMTTS-only contrast suggested that MiMo-cont fixes were often spoof cases with lower RMS, more silence, and higher RVQ entropy. NS2/NS3 validation broke that interpretation: similar high-entropy or low-energy patterns did not predict MiMo success and often appeared in MiMo failure groups.

## Media-transform side result

The exploratory media-transform path is technically validated but still metric-incomplete:

- 24 clean CoSG rows;
- 3 transforms: `mp3_64k_16k`, `noise_snr20`, `resample_8k_16k`;
- 72 transformed WAVs;
- log-mel and XLS-R paired feature drift computed.

XLS-R drift by transform:

| Transform | Pairs | Cosine mean | L2 mean |
|---|---:|---:|---:|
| mp3_64k_16k | 24 | 0.99798 | 0.265 |
| noise_snr20 | 24 | 0.98071 | 1.028 |
| resample_8k_16k | 24 | 0.99462 | 0.490 |

This is feature drift only. It is not robustness scoring, EER, or deployable-detector evidence.

## Decision

Stop CLAMTTS-only deepening unless a new mechanism predicts another source or transform before testing.

Replace the old default of “do not train” with:

> Do not run unmotivated broad training. Do run targeted confirmatory training once cheap probes have eliminated weak paths and selected a concrete hypothesis.

Wave 1 and Wave 2 have now selected the concrete hypotheses:

1. XLS-R is the trained reference to validate first.
2. Training may preserve, erase, or amplify frozen-probe failure maps.
3. Media augmentation may improve transform robustness.
4. CoRS proxy scale may help or hurt CoSG transfer depending on shortcut learning.

## Consequence

Open Wave 3 as a trained validation wave.

The next canonical plan is:

```text
docs/current/wave3_training_validation_plan.yaml
```

No GPU training is authorized by this note alone. Training requires planned log rows, explicit specs, and approval of the concrete run set.

## Claim-safe summary

Allowed:

> Wave 2 found that the CLAMTTS MiMo advantage is real but not currently predictive beyond CLAMTTS. The first proposed low-energy/silence/RVQ-entropy mechanism failed NS2/NS3 validation. Transform work remains feature-drift evidence only. These results justify moving from cheap elimination to targeted trained validation centered on XLS-R, media robustness, and CoRS/CoSG transfer.

Not allowed:

- MiMo has a general CLAMTTS-like mechanism.
- Feature drift is robustness accuracy.
- Existing probes prove trained detector behavior.
- Training is banned going forward.
