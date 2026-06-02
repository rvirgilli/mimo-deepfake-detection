# Representation-transfer matrix design

Date: 2026-05-27
Status: planned matrix; no new experiment execution authorized

Machine-readable matrices/plans:

```text
docs/current/representation_transfer_matrix.yaml
docs/current/wave1_exploratory_validation_matrix.yaml
docs/current/wave2_deepening_plan.yaml
```

Validate:

```bash
python -m mimodf research validate-matrix docs/current/representation_transfer_matrix.yaml
python -m mimodf research matrix-summary docs/current/representation_transfer_matrix.yaml
```

## Purpose

Wave-label clarification: WavLM/log-mel/media-transform runs with some `wave2-*` IDs are immutable provenance, but conceptually complete the broad exploratory Wave 1 map. Wave 2 is now the completed CLAMTTS deepening wave. Wave 3 is targeted trained validation.

Post-Wave-1 research should not be organized around MiMo revival.

The purpose is now to map which frontend representation families transfer under explicit audio-deepfake distribution shifts.

Core question:

> Under which source, codec, media, and dataset shifts do different pretrained audio representation families help or fail, and which probe-discovered failures persist after real training?

## Representation families

Current matrix families:

| Family | Current candidates | Role |
|---|---|---|
| SSL speech encoders | wav2vec2/XLSR, planned WavLM-Base+ | Strong baseline family; tests whether source-transfer strength is SSL-general. |
| Tokenizer/reconstruction encoders | MiMo continuous/RVQ | Diagnostic contrast; no privileged status. |
| Codec-tokenizer features | future EnCodec/SoundStream-style baseline | Only if tied to codec/media shift. |
| Acoustic baselines | log-mel mean/std | Sanity check against simple acoustic statistics; completed baseline wins only SIMPLESPEECH1. |

## Shift axes

| Axis | Question |
|---|---|
| source holdout | Does a representation transfer to unseen generators/sources? |
| taxonomy holdout | Does a representation transfer across quantizer/decoder families? |
| media transform | Does a representation survive compression/resampling/noise/laundering? |
| dataset shift | Does a representation transfer across dataset families? |
| representation ablation | Which representation slice carries useful/fragile signal? |

## Current evidence anchor

Wave 1 is already represented as the completed row:

```text
wave1_closed_codecfake_cosg_source_holdout
```

Summary:

- wav2vec2/XLSR wins 8/9 held-out CoSG sources by EER;
- MiMo continuous wins CLAMTTS only;
- MiMo RVQ all wins no source;
- fusion is partially complementary but unstable;
- broad MiMo training/Optuna is not justified.

## Planned rows

The matrix intentionally separates planning from execution.

Most useful next rows:

1. `wave2_design_source_holdout_ssl_candidate`
   Completed WavLM-Base+ (`microsoft/wavlm-base-plus`, revision `b21194173c0af7e94822c1776d162e2659fd4761`) source-holdout probe. Result: WavLM-Base+ mean EER `0.324`; it beats XLS-R only on CLAMTTS and is best on `0/9` sources. Report: `experiments/runs/wave2_wavlm_source_holdout_v1/report.md`.

2. `wave1_source_diagnostics_clamtts_ns_v1`
   Completed first metadata/error join for CLAMTTS vs NS2/NS3/UNIAUDIO. Result: CLAMTTS is a source-specific error reversal, not a generic MiMo advantage.

3. `wave1_logmel_source_holdout_baseline_v1`
   Completed boring acoustic baseline. Result: log-mel mean EER `0.362`; best on `1/9` sources (`SIMPLESPEECH1` only), so it does not explain XLS-R's broad transfer strength. Report: `experiments/runs/wave1_logmel_source_holdout_v1/report.md`.

4. `wave2_clamtts_mechanism_v1`
   Completed first true Wave 2 deepening diagnostic. Result: on 119 CLAMTTS cases, wrong counts are XLS-R `32`, MiMo-cont `13`, MiMo-RVQ-all `17`, WavLM `30`, log-mel `68`. MiMo-cont fixes 26 XLS-R errors and loses 7, mainly reducing spoof false negatives; log-mel collapses on spoof recall. Report: `experiments/runs/wave2_clamtts_mechanism_v1/report.md`.

5. `wave2_clamtts_case_contrast_v1`
   Completed case contrast for MiMo-cont-fixes-XLS-R vs reverse cases. Result: fix group is mostly spoof (`19/26`), longer, quieter, more silent, and has higher RVQ entropy than the 7-case reverse group. This suggests a CLAMTTS-local low-energy/silence plus token-diversity artifact, not yet a validated general mechanism. Report: `experiments/runs/wave2_clamtts_case_contrast_v1/report.md`.

6. `wave2_clamtts_hypothesis_source_validation_v1`
   Completed NS2/NS3 validation of the CLAMTTS case pattern. Result: the low-energy/silence + high-RVQ-entropy cue does not transfer as a MiMo-success mechanism. NS2 has zero MiMo-cont fixes and 19 XLS-R-fixes-MiMo cases; NS3 has 4 MiMo fixes and 24 reverse cases. Report: `experiments/runs/wave2_clamtts_hypothesis_source_validation_v1/report.md`.

7. `wave2_design_media_transform_smoke`
   Completed through log-mel and XLS-R paired drift under `/tmp/mimodf-media-transform-smoke-v1`: 24 clean CoSG rows, three transforms (`resample_8k_16k`, `mp3_64k_16k`, `noise_snr20`), 72 transformed WAV files, explicit inherited-label caveat. Log-mel drift: mp3 near-identical by cosine (`0.99996`, L2 `4.17`), noise/resample larger (`0.99155`/`125.73`, `0.99024`/`73.58`). XLS-R drift: noise largest (`0.98071`/`1.028`), then resample (`0.99462`/`0.490`), then mp3 (`0.99798`/`0.265`). No classifier scoring/EER or robustness claim.

New Wave 3 rows are planned in `docs/current/wave3_training_validation_plan.yaml`:

- XLS-R frozen backend / PEFT / full fine-tune under CoSG source holdout;
- clean-only vs media-augmented XLS-R PEFT under transform shift;
- CoRS proxy/pretraining to CoSG evaluation after CoRS audit and label-policy pinning;
- MiMo/WavLM trained variants only as later diagnostic contrasts.

Blocked rows:

- codec-tokenizer baseline until it tests a declared shift;
- CoRS scale check until CoRS audio is audited/extracted/indexed and taxonomy/label policy is pinned.

## Decision policy

Promote a direction only if:

- it survives grouped controls;
- it explains a failure mode, not just a random-split metric;
- it improves transfer or robustness under a declared shift;
- or it creates useful complementarity without damaging basic error balance.

Kill/deprioritize if:

- gains appear only in random row splits;
- support is too low;
- a frontend loses under grouped controls and has no complementary errors;
- the proposed experiment does not test a declared shift/failure mode.

## Immediate recommendation

Do not run unmotivated broad training.

Do prepare targeted trained validation:

1. use `docs/current/RESEARCH_WAVE_2_INTERIM_NOTE.md` as the Wave 2 closeout;
2. use `docs/current/wave3_training_validation_plan.yaml` as the Wave 3 planning source;
3. audit CoRS status and label policy;
4. build CodecFake training/scoring and durable transform-score infrastructure;
5. run XLS-R trained reference jobs only with explicit planned log rows and approval.

Wave 1 closeout: `docs/current/RESEARCH_WAVE_1_CLOSEOUT.md`.

The negative/conditional Wave 1 note is complete: `docs/current/RESEARCH_WAVE_1_NEGATIVE_NOTE.md`.
