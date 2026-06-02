# Wave 3 deep research A — deepen the current finding

Date: 2026-05-31
Status: historical concept note written before the completed MASKGCT seed-stability diagnostic and PEFT batch14 seed42 pilot. For current claims and next actions, use `WAVE3_REVISED_PLAN_2026_05_31.md` and `WEEKLY_TEAM_UPDATE_2026_05_31.md`.
Scope: current Wave 3A custom CoSG source-holdout evidence plus literature/concept review.
Claim boundary: diagnostic CodecFake+ CoSG only; not official CodecFake+ benchmark training; the original PEFT evidence here is batch2 seed `42` only.
Current caveat: later evidence supports a mixed-ranking, stable threshold-collapse / worst-source-risk framing for `MASKGCT`, not a stable below-chance inversion claim. PEFT batch14 seed42 is promising but matched frozen batch14 control is pending.

## Executive thesis

The strongest current direction is not “better detector numbers”. It is:

> Lightweight XLS-R adaptation can improve average held-out-generator transfer, while exposing source-specific validation blind spots under generator shift.

This is scientifically useful because it exposes a concrete failure mode of trained adaptation under generator/source shift.

## Ground facts from our artifacts

Current logged evidence:

- Frozen XLS-R backend, deterministic, seeds `42/123/2024`:
  - mean EER `0.3401 ± 0.0331`
  - mean AUROC `0.7146 ± 0.0448`
  - mean balanced accuracy `0.6057 ± 0.0650`
- XLS-R PEFT adapter, deterministic, seed `42`:
  - mean EER `0.2068`
  - mean AUROC `0.8486`
  - mean balanced accuracy `0.7247`
- PEFT batch2 seed42 improves EER on `8/9` folds vs the frozen batch4 3-seed mean; this is protocol-level evidence, not an architecture-only comparison.
- Exception: `MASKGCT`:
  - frozen 3-seed mean EER `0.4265`
  - PEFT seed42 EER `0.5521`
  - PEFT seed42 AUROC `0.4333`
  - PEFT predicted-spoof rate at threshold `0.5`: `0.9800`
  - PEFT selected checkpoint validation AUROC on non-MASKGCT rows: `0.8291`
  - held-out MASKGCT support: `1152` rows, balanced `576/576`

The key observation is not just poor thresholding. PEFT assigns slightly higher mean spoof probability to MASKGCT bonafide than MASKGCT spoof:

| Label | mean P(spoof) |
|---|---:|
| MASKGCT bonafide | `0.7595` |
| MASKGCT spoof | `0.7419` |

That is a ranking failure / inversion candidate.

## Literature/concept anchors

These are the external ideas that best explain the finding.

1. **Generator/source shift in audio spoof detection**
   ASVspoof 2021 DF and related post-challenge analyses emphasize that channel, codec, and attack/source mismatch can dominate detector behavior.
   URL: https://arxiv.org/html/2210.02437

2. **Codec-based generation is its own regime**
   CodecFake/CodecFake+ explicitly targets codec-based speech generation and separates CoRS proxy data from CoSG generated/web-sourced material.
   URLs: https://arxiv.org/html/2406.07237v1 , https://arxiv.org/html/2501.08238v2

3. **MASKGCT is a modern codec-token generator family**
   If held out, it can expose artifacts absent from older or different codec-generation sources.
   URL: https://arxiv.org/abs/2409.00750

4. **Shortcut learning**
   Deep models can learn predictive-but-noncausal cues that flip under distribution shift. Below-chance AUROC is consistent with a learned shortcut whose label association reverses on a target domain.
   URLs: https://arxiv.org/abs/2004.07780 , https://www.isca-archive.org/interspeech_2023/shim23b_interspeech.html

5. **Negative transfer**
   Transfer/adaptation can improve average target behavior while harming a specific target domain.
   URLs: https://cse.hkust.edu.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf , https://arxiv.org/abs/2009.00909

6. **Fine-tuning can distort pretrained features**
   Fine-tuning/adaptation can move a representation away from broadly useful pretrained structure if the training validation objective rewards narrow source cues.
   URL: https://openreview.net/forum?id=UYneFzXSJWh

7. **Model selection under distribution shift is unreliable**
   DomainBed/WILDS-style results warn that validation drawn from the same source mix as training is weak evidence for unseen-domain performance.
   URLs: https://openreview.net/forum?id=lQdXeXDoWtI , https://arxiv.org/abs/2012.07421

8. **Calibration is not ranking**
   Calibration fixes monotone probability quality; it cannot repair below-chance class ranking without changing score direction or representation.
   URL: https://proceedings.mlr.press/v70/guo17a.html

9. **Worst-source risk matters**
   Average performance hides failures; source-macro and worst-source metrics are central when deployment domains shift.
   URL: https://openreview.net/forum?id=ryxGuJrFvS

## Working mechanism model

A compact mechanism model:

1. Frozen XLS-R contains broad acoustic/phonetic/codec cues.
2. PEFT updates are small but targeted enough to amplify correlations in the non-MASKGCT train/validation pool.
3. Those correlations improve many held-out sources.
4. MASKGCT violates the learned correlation: the cue PEFT maps toward “spoof” is present in MASKGCT bonafide, or absent/altered in MASKGCT spoof.
5. Non-MASKGCT validation selects the bad checkpoint because validation and training share the shortcut direction.
6. The result is not merely bad thresholding; the score ranking itself degrades below chance.

This should be called **adaptation-induced source inversion** until seed stability is verified.

## Hypotheses to test

### H1 — MASKGCT collapse is seed-stable

Prediction: PEFT seeds `123` and `2024` on MASKGCT also show poor AUROC/EER, ideally AUROC near or below `0.5`.

Minimum test:

```text
xlsr_peft_adapter, fold MASKGCT, seeds 123/2024, deterministic, 10 epochs, checkpoint val_auroc
```

Decision rule:

- If both repeat: treat MASKGCT as primary mechanism case.
- If one repeats and one improves: classify as unstable adaptation risk; still important.
- If neither repeats: reclassify seed42 as directional anomaly and investigate training stochasticity/checkpointing.

### H2 — Model selection is blind to MASKGCT

Prediction: epoch-wise non-MASKGCT validation AUROC is weakly or negatively correlated with MASKGCT test AUROC.

Test:

- Score every saved or reconstructable epoch on MASKGCT, or rerun MASKGCT PEFT with per-epoch test scoring.
- Plot validation AUROC vs MASKGCT AUROC/EER.

Interpretation:

- Positive correlation: current selected epoch unlucky but validation partially useful.
- Zero/negative correlation: validation protocol cannot select for this target source.

### H3 — PEFT increases source separability / source shortcut pressure

Prediction: source-model probes become easier after PEFT or logits encode source identity strongly.

Test:

- Fit source classifiers on frozen XLS-R embeddings vs PEFT embeddings/logits if accessible.
- Specifically test `MASKGCT` vs rest separability.

Interpretation:

- High source separability with poor label transfer supports shortcut learning.

### H4 — MASKGCT may separate ranking from calibration/thresholding

Prediction: monotone calibration cannot fix MASKGCT AUROC below chance; source-specific threshold may improve balanced accuracy slightly but not ranking.

Already partly supported:

- AUROC `0.4333`
- inverted scores give higher best balanced accuracy than normal scores in the diagnostic.

Next test:

- Reliability/ECE/Brier by source.
- ROC/score histograms by source and label.
- Source-specific threshold vs global threshold.

### H5 — The failure is tied to generator/codec source, not simple audio quality

Prediction: duration/RMS/silence summaries alone do not explain the inversion.

Test:

- Compare MASKGCT bonafide/spoof audio statistics and nearest-neighbor XLS-R features to train sources.
- Match subsets on duration/RMS/silence and rescore.

### H6 — A source-robust objective can reduce the inversion

Prediction: source-balanced validation, group DRO, leave-one-source validation, or source-adversarial regularization reduces worst-source collapse, possibly at average-cost.

Do not implement yet. First prove seed stability.

## Immediate experiment packet

### Packet A — cheap decisive GPU

Run only:

```text
PEFT MASKGCT seed 123
PEFT MASKGCT seed 2024
```

Same settings as seed42:

```text
--condition xlsr_peft_adapter
--fold MASKGCT
--epochs 10
--checkpoint-metric val_auroc
--deterministic
```

Expected cost: two folds, not overnight.

### Packet B — CPU mechanism extension

After Packet A:

1. Compare PEFT MASKGCT score distributions across seeds.
2. Compare selected epoch and validation AUROC across seeds.
3. Add source-size/support table to the mechanism note.
4. Add calibration-vs-ranking table:
   - AUROC
   - EER
   - best balanced accuracy threshold
   - inverted-score best balanced accuracy
   - predicted-spoof rate at `0.5`

### Packet C — if stable

Run PEFT seeds `123/2024` all folds. This gives a fair 3-seed PEFT-vs-frozen comparison.

## What not to do yet

- No full fine-tuning.
- No broad frontend sweep.
- No ASVspoof5 training.
- No CoRS training before extraction/indexing/label policy.
- No claim that PEFT is robust or superior until PEFT seeds `123/2024` exist.

## Claim language if H1 repeats

Allowed:

> In a custom CodecFake+ CoSG source-holdout diagnostic, XLS-R PEFT showed promising average source transfer in seed42 pilots, while held-out MASKGCT exposed poor operating-point behavior and mixed ranking under source shift.

Not allowed:

> PEFT fails on MASKGCT generally.

> XLS-R PEFT is better than frozen XLS-R.

> This is official CodecFake+ performance.

## Why this direction is worth deepening

This is a clean research path because it has:

- a positive average result;
- a large, balanced, interpretable failure source;
- a validation blind spot;
- a concrete mechanism candidate;
- cheap next experiments;
- clear claim boundaries.

That is stronger than another undirected benchmark run.
