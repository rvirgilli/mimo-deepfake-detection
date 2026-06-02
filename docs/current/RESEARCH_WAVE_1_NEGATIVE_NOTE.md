# Wave 1 negative/conditional research note

Date: 2026-05-27
Status: accepted Wave 1 interpretation; diagnostic evidence, not paper-ready claim

Source evidence:

- `docs/current/RESEARCH_WAVE_1_RESULTS.md`
- `docs/current/RESEARCH_PURPOSE_RESET.md`
- `docs/current/representation_transfer_matrix.yaml`
- `docs/current/research_execution_log.jsonl`
- `experiments/runs/wave1_feature_source_diagnostics_v1/report.md`

## Short answer

Wave 1 did not find a reason to scale MiMo.

It found a useful negative result:

> On CodecFake+ CoSG held-out-source feature probes, wav2vec2/XLSR transfers better than MiMo overall. MiMo has real but source-conditional signal, strongest on CLAMTTS. That conditional signal is not broad enough to justify MiMo-centered training, Optuna, or a superiority claim.

## What was tested

Wave 1 used cached frozen features only:

- wav2vec2/XLSR continuous;
- MiMo continuous 25 Hz;
- MiMo RVQ early/late/all.

Probe design:

- CodecFake+ CoSG only;
- utterance-level pooled frozen features;
- linear logistic-regression probes;
- grouped held-out-source controls for the main evidence;
- random row splits only as leakage-prone diagnostics;
- all MiMo/SSL feature caches extracted with batch size 1.

This was not full training, not ASVspoof evaluation, and not paper-grade external validation.

## Main result

Held-out-source sweep over nine CoSG sources:

| Representation | Mean EER | Mean balanced acc | Mean AUROC | Best single by EER |
|---|---:|---:|---:|---:|
| wav2vec2/XLSR | 0.193 | 0.776 | 0.873 | 8/9 in original Wave 1, 7/9 after adding log-mel/WavLM comparators |
| MiMo continuous 25 Hz | 0.294 | 0.697 | 0.747 | 1/9 |
| MiMo RVQ all | 0.309 | 0.682 | 0.763 | 0/9 |

Interpretation:

- SSL/XLSR is the safer transferring representation in this evidence slice.
- MiMo continuous is not useless; it carries forensic signal.
- MiMo RVQ histograms are weaker than MiMo continuous and do not beat SSL on any held-out source.
- A later log-mel addendum found mean EER `0.362` and best-by-EER count `1/9` (`SIMPLESPEECH1` only), so the broad XLS-R effect is not explained by a boring acoustic baseline.
- The previous “MiMo revival” framing is not supported.

## Conditional positive case

CLAMTTS is the main exception.

| Source | SSL EER | MiMo cont EER | MiMo RVQ all EER | Best |
|---|---:|---:|---:|---|
| CLAMTTS | 0.244 | 0.100 | 0.144 | MiMo continuous |

The follow-up metadata/error diagnostic sharpened this:

| Source | SSL wrong | MiMo cont wrong | Key pattern |
|---|---:|---:|---|
| CLAMTTS | 32 | 13 | MiMo-cont fixes 26 SSL mistakes and loses 7 cases. |
| NS2 | 4 | 23 | SSL fixes 19 MiMo-cont mistakes; MiMo-cont adds 0. |
| NS3 | 13 | 33 | SSL fixes 24 MiMo-cont mistakes; MiMo-cont adds 4. |
| UNIAUDIO | 4 | 7 | SSL fixes 4 MiMo-cont mistakes; MiMo-cont adds 1. |

Mean audio durations are similar across these sources, so duration alone does not explain the CLAMTTS reversal.

Interpretation:

- CLAMTTS is a real conditional case worth explaining.
- It is not evidence for a generic tokenizer advantage.
- Any future MiMo use should be diagnostic: ask what source mechanism MiMo sees in CLAMTTS and why it fails on NS2/NS3/UNIAUDIO.

## What failed

### Broad MiMo claim

Not supported. MiMo loses most held-out-source probes and loses on averaged metrics.

### MiMo-centered Wave 2 training

Not justified. Training would be expensive relative to evidence quality and would likely chase a source-specific artifact.

### Optuna/HPO

Not justified. Hyperparameter search would optimize around a weak or unstable premise.

### Source/taxonomy tracing as a MiMo advantage

Not supported. Source-model random split favored SSL. Taxonomy random splits are likely confounded and are not claim-safe.

### Naive SSL+MiMo fusion

Not mature. Equal-weight fusion sometimes improves EER over SSL but is unstable and can worsen balanced accuracy. Treat as complementarity evidence only.

## What survived

1. **Representation-transfer framing**

   The project should compare frontend families under explicit shifts, not rescue one frontend.

2. **Conditional-cue hypothesis**

   MiMo may detect cues in specific generator/source mechanisms, especially CLAMTTS, but this is narrow.

3. **Audit-first framework contribution**

   The strongest current contribution is the controlled, logged, machine-readable research system: specs, run logs, feature caches, probe outputs, matrix rows, and decision records.

4. **Need for grouped controls**

   Random splits were flattering but unsafe. Held-out-source controls changed the conclusion.

## Decision

Do not run broad MiMo training.

Do not run MiMo Optuna.

Do not write a MiMo-superiority paper.

Use Wave 1 as a negative/conditional result and pivot to representation transfer under shifts.

## If continuing research

Best next options, in order:

1. **Add one SSL-family comparator** under the same feature-probe protocol, such as WavLM or HuBERT, to see whether XLSR strength is SSL-general.
2. **Deepen CLAMTTS-vs-NS analysis** only if there is a concrete mechanism to test, not just curiosity.
3. **Stage robustness/media-transform probes** as a separate Path E study.
4. **Return to system/release blockers** if the goal is a shareable repo.

Avoid:

- adding many frontends without a shift hypothesis;
- training before feature/probe evidence names a concrete reason;
- random-row metrics as headline evidence;
- any claim that MiMo is generally better.

## Claim-safe phrasing

Allowed:

> In a bounded CodecFake+ CoSG feature-probe study, wav2vec2/XLSR transferred better than MiMo on most held-out sources. MiMo showed source-conditional signal, especially on CLAMTTS, but this did not generalize across sources.

Not allowed:

> MiMo is better for deepfake detection.

Not allowed:

> Codec/tokenizer representations are superior forensic frontends.

Not allowed:

> Fusion solves the problem.

## Bottom line

Wave 1 was successful because it prevented a bad research bet from scaling.

The next defensible project is not “MiMo revival.” It is an audited study of representation transfer and failure modes under distribution shift.
