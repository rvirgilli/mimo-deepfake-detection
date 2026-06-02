# Research Wave 1 feature-probe results

Date: 2026-05-26
Status: first feature-only probes complete; diagnostic, not paper evidence

Parent plan: `docs/current/RESEARCH_WAVE_0_1_PLAN.md`

## Scope

These runs use cached CodecFake+ CoSG features only:

```text
features/mimodf/wave1/codecfake_cosg_mimo_continuous_25hz_b1/
features/mimodf/wave1/codecfake_cosg_mimo_rvq_early_b1/
features/mimodf/wave1/codecfake_cosg_mimo_rvq_late_b1/
features/mimodf/wave1/codecfake_cosg_mimo_rvq_all_b1/
features/mimodf/wave1/codecfake_cosg_wav2vec2_xlsr_b1/
```

All MiMo/SSL caches use extraction batch size 1. The precise command/artifact log is tracked in `docs/current/research_execution_log.jsonl` and summarized in `docs/current/RESEARCH_EXECUTION_LOG.md`.

Generated probe outputs are local/ignored under:

```text
experiments/runs/wave1_feature_probe_label_holdout_maskgct/
experiments/runs/wave1_feature_probe_label_random/
experiments/runs/wave1_feature_probe_source_random_seed42/
experiments/runs/wave1_feature_probe_taxonomy_random_seed42/
```

Probe type:

- frozen features;
- utterance-level pooling;
- logistic regression linear probe via sklearn in `mimo-df`;
- train-only standardization for continuous features;
- normalized per-codebook unigram histograms for RVQ codes.

This is not full model training, not ASVspoof evaluation, and not a paper result.

## Binary bonafide/spoof — held-out MASKGCT source

Split policy: train on all non-MASKGCT rows, test on all MASKGCT rows. This is more useful than random row split, but still only one held-out source probe.

| Representation | Balanced acc | AUROC | EER |
|---|---:|---:|---:|
| wav2vec2/XLSR continuous | 0.738 | 0.838 | 0.245 |
| MiMo continuous 25 Hz | 0.634 | 0.763 | 0.297 |
| MiMo RVQ all | 0.589 | 0.678 | 0.366 |
| MiMo RVQ late | 0.579 | 0.667 | 0.384 |
| MiMo RVQ early | 0.543 | 0.569 | 0.441 |

Initial read: SSL is strongest on this held-out source. MiMo continuous has signal. RVQ late/all beat early but do not beat SSL.

## Binary bonafide/spoof — random row split diagnostic

Seeds: 42, 123, 2024. Random row split is leakage-prone and diagnostic only.

| Representation | Balanced acc mean ± std | Mean EER |
|---|---:|---:|
| wav2vec2/XLSR continuous | 0.950 ± 0.005 | 0.053 |
| MiMo continuous 25 Hz | 0.900 ± 0.005 | 0.101 |
| MiMo RVQ all | 0.896 ± 0.005 | 0.107 |
| MiMo RVQ late | 0.872 ± 0.011 | 0.127 |
| MiMo RVQ early | 0.809 ± 0.009 | 0.188 |

Initial read: random splits show strong separability for all representations, but these numbers are not leakage-safe. Late/all RVQ clearly outperform early RVQ, supporting the early-vs-late diagnostic direction only weakly until stronger splits are run.

## Source-model classification — random row split diagnostic

Seed: 42. Random row split only.

| Representation | Balanced acc | Macro F1 |
|---|---:|---:|
| wav2vec2/XLSR continuous | 0.430 | 0.433 |
| MiMo continuous 25 Hz | 0.342 | 0.323 |
| MiMo RVQ early | 0.247 | 0.264 |
| MiMo RVQ all | 0.220 | 0.238 |
| MiMo RVQ late | 0.192 | 0.204 |

Initial read: this does not support a MiMo source-tracing advantage on the first diagnostic split. SSL is stronger.

## Taxonomy classification — random row split diagnostic

Seed: 42. Rows with missing taxonomy targets are dropped, so these are spoof-only taxonomy probes. Random row split only.

### Decoder type

| Representation | Balanced acc | Macro F1 |
|---|---:|---:|
| MiMo continuous 25 Hz | 0.961 | 0.961 |
| MiMo RVQ late | 0.957 | 0.954 |
| MiMo RVQ all | 0.953 | 0.948 |
| wav2vec2/XLSR continuous | 0.951 | 0.954 |
| MiMo RVQ early | 0.881 | 0.897 |

### Quantizer type

| Representation | Balanced acc | Macro F1 |
|---|---:|---:|
| MiMo continuous 25 Hz | 0.724 | 0.783 |
| wav2vec2/XLSR continuous | 0.697 | 0.744 |
| MiMo RVQ late | 0.564 | 0.598 |
| MiMo RVQ all | 0.537 | 0.566 |
| MiMo RVQ early | 0.408 | 0.437 |

Initial read: decoder type is highly predictable by nearly every representation, so this may reflect source/content confounding rather than codec-specific forensic structure. Quantizer type is harder; MiMo continuous is slightly above SSL, but RVQ histograms are weak.

## Held-out-source controls and fusion/error overlap

Second-pass held-out-source probes were run after adding per-utterance `predictions.jsonl` and score-fusion reporting.

Tracked outputs:

```text
experiments/runs/wave1_feature_probe_label_holdout_sources_v2/
experiments/runs/wave1_feature_probe_fusion_holdout_sources_v2/
```

All commands are logged in `docs/current/research_execution_log.jsonl` with run IDs prefixed by:

```text
wave1-holdout-...
wave1-fusion-holdout-...
```

### Held-out source binary probes

| Held-out source | Representation | Balanced acc | AUROC | EER | Test support |
|---|---|---:|---:|---:|---|
| MASKGCT | wav2vec2/XLSR | 0.738 | 0.838 | 0.245 | 576/576 |
| MASKGCT | MiMo continuous 25 Hz | 0.634 | 0.763 | 0.297 | 576/576 |
| MASKGCT | MiMo RVQ all | 0.589 | 0.678 | 0.366 | 576/576 |
| CLAMTTS | wav2vec2/XLSR | 0.737 | 0.863 | 0.244 | 41/78 |
| CLAMTTS | MiMo continuous 25 Hz | 0.894 | 0.929 | 0.100 | 41/78 |
| CLAMTTS | MiMo RVQ all | 0.856 | 0.945 | 0.144 | 41/78 |
| VALLE | wav2vec2/XLSR | 0.709 | 0.781 | 0.300 | 53/57 |
| VALLE | MiMo continuous 25 Hz | 0.712 | 0.696 | 0.355 | 53/57 |
| VALLE | MiMo RVQ all | 0.631 | 0.679 | 0.382 | 53/57 |

Initial read: MiMo is source-conditional. It is worse than SSL on MASKGCT, better on CLAMTTS, and roughly tied/worse on VALLE depending on metric. This weakens any broad MiMo claim but keeps a narrower conditional-forensic-cue hypothesis alive.

### Score fusion diagnostics

Simple equal-weight score averaging, not calibrated fusion.

| Held-out source | Fusion | Balanced acc | AUROC | EER | Score corr. |
|---|---|---:|---:|---:|---:|
| MASKGCT | SSL + MiMo continuous | 0.725 | 0.853 | 0.243 | 0.232 |
| MASKGCT | SSL + MiMo RVQ all | 0.708 | 0.826 | 0.262 | 0.207 |
| CLAMTTS | SSL + MiMo continuous | 0.851 | 0.917 | 0.169 | 0.566 |
| CLAMTTS | SSL + MiMo RVQ all | 0.826 | 0.925 | 0.175 | 0.508 |
| VALLE | SSL + MiMo continuous | 0.682 | 0.800 | 0.227 | 0.505 |
| VALLE | SSL + MiMo RVQ all | 0.672 | 0.781 | 0.282 | 0.263 |

Initial read: fusion sometimes improves ranking/EER over SSL (MASKGCT continuous, VALLE continuous), but often hurts balanced accuracy and does not beat the best single representation on CLAMTTS. Score correlations are low-to-moderate, so complementarity exists, but naive averaging is not consistently useful.

## Held-out-source sweep closure

To close Wave 1, held-out-source probes were expanded to every CoSG source with at least 10 bonafide and 10 spoof rows:

```text
MASKGCT, CLAMTTS, VALLE, SIMPLESPEECH1, NS3, SIMPLESPEECH2, NS2, GPST, UNIAUDIO
```

This is still small and source-support-limited, but it is a better elimination test than one-off random splits.

| Held-out source | Support bonafide/spoof | SSL EER | MiMo cont EER | MiMo RVQ all EER | Best single | SSL+cont EER | SSL+RVQ EER |
|---|---:|---:|---:|---:|---|---:|---:|
| MASKGCT | 576/576 | 0.245 | 0.297 | 0.366 | SSL | 0.243 | 0.262 |
| CLAMTTS | 41/78 | 0.244 | 0.100 | 0.144 | MiMo cont | 0.169 | 0.175 |
| VALLE | 53/57 | 0.300 | 0.355 | 0.382 | SSL | 0.227 | 0.282 |
| SIMPLESPEECH1 | 34/32 | 0.121 | 0.152 | 0.152 | SSL | 0.061 | 0.091 |
| NS3 | 32/32 | 0.188 | 0.562 | 0.438 | SSL | 0.375 | 0.312 |
| SIMPLESPEECH2 | 31/31 | 0.129 | 0.129 | 0.194 | SSL/tie | 0.129 | 0.129 |
| NS2 | 23/23 | 0.043 | 0.391 | 0.348 | SSL | 0.130 | 0.130 |
| GPST | 10/30 | 0.283 | 0.300 | 0.300 | SSL | 0.300 | 0.200 |
| UNIAUDIO | 11/11 | 0.182 | 0.364 | 0.455 | SSL | 0.273 | 0.182 |

Averaged over these nine held-out-source probes:

| Representation | Mean EER | Mean balanced acc | Mean AUROC |
|---|---:|---:|---:|
| wav2vec2/XLSR | 0.193 | 0.776 | 0.873 |
| MiMo continuous 25 Hz | 0.294 | 0.697 | 0.747 |
| MiMo RVQ all | 0.309 | 0.682 | 0.763 |

Best-single count by EER:

```text
wav2vec2/XLSR: 8/9 sources
MiMo continuous: 1/9 sources
MiMo RVQ all: 0/9 sources
```

Fusion beat SSL by EER on 4/9 sources for both SSL+MiMo-continuous and SSL+MiMo-RVQ-all, but not consistently and sometimes worsened EER. Treat this as evidence of partial complementarity, not a deployable fusion win.

## Log-mel baseline addendum

Date: 2026-05-30
Run IDs:

```text
wave1-codecfake-cosg-logmel-80-100hz-v1
wave1-holdout-*-logmel_80_100hz
wave1-logmel-source-holdout-summary-v1
```

A boring 80-bin log-mel, 100 Hz, mean/std-pooled baseline was added to complete the exploratory Wave 1 map. It used the same CoSG source-holdout protocol and the same linear-probe backend.

| Representation | Mean EER | Mean balanced acc | Mean AUROC | Best single by EER |
|---|---:|---:|---:|---:|
| wav2vec2/XLSR | 0.193 | 0.776 | 0.873 | 7/9 in five-way comparison |
| MiMo continuous 25 Hz | 0.294 | 0.697 | 0.747 | 1/9 |
| MiMo RVQ all | 0.309 | 0.682 | 0.763 | 0/9 |
| WavLM-Base+ | 0.324 | 0.660 | 0.715 | 0/9 |
| log-mel mean/std | 0.362 | 0.637 | 0.682 | 1/9 |

Log-mel wins only `SIMPLESPEECH1` by EER. It does not explain the broad XLS-R advantage and fails badly on `CLAMTTS`, `NS2`, `NS3`, and `UNIAUDIO`. This keeps log-mel as a useful sanity baseline, not a primary frontend.

Report:

```text
experiments/runs/wave1_logmel_source_holdout_v1/report.md
```

## Wave 1 final decision

Wave 1 does **not** justify broad MiMo training, Optuna, or a MiMo-superiority paper claim.

What survives:

- MiMo has real signal on CoSG, especially continuous features and late/all RVQ over early RVQ.
- MiMo behavior is source-conditional; CLAMTTS is the clearest positive case.
- SSL and MiMo errors are not identical, but equal-weight fusion is unstable.

What is weakened:

- broad codec/tokenizer-forensic-cue claim;
- source/taxonomy tracing claim as a MiMo advantage;
- naive SSL+MiMo fusion claim;
- RVQ histogram superiority.

Recommended post-Wave-1 direction:

1. Do not launch broad Wave 2 training.
2. If continuing, run only targeted diagnostics around the conditional cases: why CLAMTTS favors MiMo and why NS2/NS3/UNIAUDIO punish it.
3. Add protocol/feature joins and stronger grouping before any new metric table.
4. Keep Path E robustness deferred.

## Targeted source diagnostic addendum

Date: 2026-05-27
Run ID: `wave1-source-diagnostics-clamtts-ns-v1`
Output: `experiments/runs/wave1_feature_source_diagnostics_v1/report.md`

This command joined existing held-out-source predictions to CodecFake+ protocol/audio metadata for CLAMTTS, NS2, NS3, and UNIAUDIO. It did not train models or extract new features.

| Source | Support bonafide/spoof | SSL wrong | MiMo cont wrong | MiMo RVQ all wrong | Key SSL-vs-MiMo-cont pattern |
|---|---:|---:|---:|---:|---|
| CLAMTTS | 41/78 | 32 | 13 | 17 | MiMo-cont correct when SSL wrong: 26 cases; SSL correct when MiMo-cont wrong: 7 cases. |
| NS2 | 23/23 | 4 | 23 | 19 | SSL correct when MiMo-cont wrong: 19 cases; MiMo-cont adds 0 unique correct cases. |
| NS3 | 32/32 | 13 | 33 | 29 | SSL correct when MiMo-cont wrong: 24 cases; MiMo-cont correct when SSL wrong: 4 cases. |
| UNIAUDIO | 11/11 | 4 | 7 | 10 | SSL correct when MiMo-cont wrong: 4 cases; MiMo-cont correct when SSL wrong: 1 case. |

Audio duration does not explain the CLAMTTS win: mean durations are similar across these sources (`4.63s` CLAMTTS, `4.34s` NS2, `4.36s` NS3, `4.37s` UNIAUDIO), though CLAMTTS has larger support and a longer max-duration tail.

Interpretation: the CLAMTTS effect is not a generic tokenizer advantage. It is a source-specific error reversal: MiMo continuous fixes many SSL CLAMTTS mistakes, but the same representation collapses on NS2/NS3 and adds little on UNIAUDIO. This keeps only a narrow conditional-cue hypothesis alive.

Immediate consequence: do not add MiMo training. If research continues, inspect case-level CLAMTTS-vs-NS metadata/features or test a second SSL frontend; do not promote MiMo as a main frontend.

## Path-level update

| Path | Evidence observed | Decision | Next action |
|---|---|---|---|
| A Codec/tokenizer forensic cues | MiMo continuous/RVQ have signal, but SSL wins 8/9 held-out sources by EER. | deprioritize / narrow | Keep only source-conditional diagnostics; no broad Wave 2. |
| B MiMo + SSL complementarity | Fusion beats SSL EER on 4/9 sources but is unstable and often hurts balanced accuracy. | hold | Error-case analysis before any calibrated fusion. |
| C Source/taxonomy tracing | Source-model random split favors SSL; taxonomy random split likely confounded. | deprioritize | Do not promote without grouped/source-held-out taxonomy controls. |
| D Semantic vs acoustic explanation | Late/all RVQ outperform early RVQ in diagnostics, but RVQ all still loses to SSL on held-out sources. | hold / weak | Preserve as explanation diagnostic, not primary hypothesis. |
| E Robustness transforms | Not tested in Wave 1. | deferred | No action until ASVspoof5 or transform tooling is staged. |

## Do not claim

- Do not claim MiMo superiority.
- Do not use random row split metrics as main evidence.
- Do not treat taxonomy probe accuracy as codec causality without grouped/source-held-out controls.
- Do not mix MiMo feature caches across extraction batch sizes.

## Next slice

Wave 1 is closed. A first targeted CLAMTTS-vs-NS diagnostic is complete. The negative/conditional note is `docs/current/RESEARCH_WAVE_1_NEGATIVE_NOTE.md`. Next work should be one of:

1. test one additional SSL frontend with the existing source-holdout feature-probe protocol;
2. deepen CLAMTTS-vs-NS case analysis only if it names a concrete mechanism to test;
3. stage a separate robustness/transform probe for Path E;
4. move back to reproducibility/system-release blockers.
