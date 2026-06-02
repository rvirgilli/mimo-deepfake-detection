# Paper assessment and next steps

Purpose: decide what to fix, improve, and extend before any manuscript rewrite. This is not prose for the paper.

## Current assessment

The earlier manuscript has a worthwhile core question but is not yet defensible as a scientific report. The strongest problem is not wording; it is evidence control.

What still looks valuable:

- The adaptation-axis comparison is interesting: frozen vs adapter vs full fine-tuning.
- wav2vec2 evidence is comparatively strong and still supports monotonic improvement on LA.
- MiMo behavior is scientifically interesting, especially adapter usefulness vs unstable/full fine-tuning behavior.
- Existing artifacts are rich enough to salvage a paper without immediately starting a large training campaign.

What is currently unsafe:

- Main table rows mix seed sets, artifact families, and tDCF aggregation rules.
- MiMo adapter n=5 is unsupported locally; only two full evals were found.
- MiMo frozen silently excludes seed 456, with no failure evidence.
- wav2vec2 full FT is partly an external Tak et al. point, not four local seeds.
- MiMo full EER and tDCF use inconsistent seed sets.
- Feature-distortion and objective-class claims are stronger than the evidence.
- Validation/checkpoint-selection protocol is still not cleanly documented per experiment family.

## Fix first: must-do before any paper rewrite

1. **Freeze the corrected evidence table.**
   - Use `CORRECTED_MAIN_TABLE_DRAFT.md` as the working table.
   - Decide every row's `n/source` explicitly.
   - Never report a mean whose seed list is not named.

2. **Reconcile LA min-tDCF.**
   - Use official ASVspoof LA evaluator outputs only.
   - For each LA row, record whether tDCF is based on the same seeds as EER.
   - If not, either recompute from available scores or mark the mismatch.

3. **Audit validation protocol.**
   - For each experiment family, document:
     - train set;
     - validation/checkpoint-selection set;
     - eval set;
     - early stopping or fixed epoch rule;
     - optimizer/loss/RawBoost differences.
   - This is essential before making any causal comparison.

4. **Resolve MiMo adapter.**
   - Current local evidence supports only n=2.
   - Options, in order:
     1. recover missing artifacts from external storage;
     2. report n=2 honestly as exploratory;
     3. approve a small controlled rerun if adapter is central.
   - Do not keep n=5 unless all five score/result/config/checkpoint records exist.

5. **Resolve seed exclusion policy in the result table.**
   - Include all completed seeds by default.
   - Exclusions require concrete failure evidence.
   - Outlier-only exclusion belongs in sensitivity analysis, not the main result.

## Improve next: strengthen evidence without broad training

1. **Per-attack analysis from existing score files.**
   - Show whether MiMo/wav2vec2 differences are broad or attack-specific.
   - Especially useful for LA crossover and DF weakness.

2. **Validation-to-eval correlation.**
   - Compare dev/validation EER or loss to ASVspoof2021 LA/DF eval where logs exist.
   - This can expose whether checkpoint selection is reliable or leaking instability.

3. **Checkpoint provenance table.**
   - Several result files cite missing checkpoints or mismatched epochs.
   - Make a compact table: seed, cited checkpoint, local checkpoint, manifest best epoch, status.

4. **Training dynamics quality check.**
   - Separate real training curves from representative/selectively reported curves.
   - If curves are single-seed, label them as examples, not aggregate evidence.

5. **Parameter/protocol fairness table.**
   - Compare wav2vec2 and MiMo per strategy:
     - sample rate;
     - frontend trainable params;
     - backend/projection;
     - optimizer;
     - loss weights;
     - RawBoost;
     - checkpoint selection.
   - This will make limitations concrete rather than defensive.

## Extend only after fixes

Extensions should answer reviewer risks, not expand scope randomly.

High value / bounded:

1. **Representation drift analysis** from existing checkpoints.
   - Measure feature cosine/CKA drift frozen vs adapter/full on a small fixed subset.
   - Supports or weakens the feature-distortion hypothesis.
   - No new training required if checkpoints load.

2. **Class-separation analysis** on frozen/adapted features.
   - Simple metrics: bonafide/spoof centroid distance, silhouette, linear probe sanity.
   - Helps explain why adapter helps or full FT hurts.

3. **MiMo adapter controlled completion** only if approved.
   - If the paper's core claim depends on MiMo adapter, run missing seeds with a locked config.
   - Prefer 3 missing seeds to recover n=5 over any new HPO.
   - Before training, write the exact protocol and acceptance criteria.

Lower value / defer:

- Broad HPO.
- Many new encoders.
- New datasets before ASVspoof2021 evidence is clean.
- Architecture changes to chase better numbers.

## Recommended project direction

The revised study should become:

> A carefully audited case study of two pretrained audio encoders under increasing adaptation capacity, with explicit seed provenance and protocol caveats.

It should not claim:

- a general law about contrastive vs reconstruction objectives;
- feature distortion as proven without representation evidence;
- statistically strong MiMo adapter behavior from n=2;
- clean full-FT fairness while protocols differ.

## Assessment closure

Assessment is concluded in `ASSESSMENT_CONCLUSION.md`.

Immediate next phase choices:

1. rewrite scientific claims from the conclusion doc;
2. run no-training extensions from existing artifacts;
3. package/recover artifacts for reproducibility;
4. design, but do not launch, a controlled MiMo adapter rerun plan.
