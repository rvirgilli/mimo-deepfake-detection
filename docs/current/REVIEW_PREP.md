# Review preparation notes

This file is for likely reviewer concerns while the official reviews are unavailable. Update only with audited evidence. Final assessment decisions are in `ASSESSMENT_CONCLUSION.md`.

## Likely issue: result provenance

Risk: reviewer challenges whether multi-seed means are traceable.

Planned response:
- Provide seed-by-seed table with score files, configs, checkpoints, evaluator scripts from `RESULTS_PROVENANCE.md`.
- Do not report n=5 unless all five seed results are traceable.
- Current evidence: MiMo adapter n=5 is unsupported and missing artifacts are assumed unavailable; only two full evals were found. `CHECKPOINT_PROVENANCE_GAPS.md` records which metric rows are score-only vs fully checkpoint-backed.

## Likely issue: seed exclusions / outliers

Risk: excluding MiMo frozen seed 456 looks like selective reporting.

Planned response:
- Default to all completed seeds in main table.
- If seed is excluded, state concrete failure mode and report sensitivity with/without it.
- Current evidence: MiMo frozen seed 456 has no failure/corruption evidence and should be included by default.

## Likely issue: validation protocol

Risk: checkpoint selection may use ASVspoof2021 fast subset while text says ASVspoof2019 dev.

Planned response:
- State exact checkpoint-selection set for each experiment family.
- Avoid generic `dev` wording.

## Likely issue: asymmetric full fine-tuning protocols

Risk: wav2vec2 full FT uses Tak et al. script; MiMo full uses project training stack.

Known protocol differences to audit/report:
- loss weights;
- optimizer Adam vs AdamW;
- RawBoost algorithm 5 vs 6;
- AASIST projection/backend width;
- early stopping vs fixed epochs;
- sample rate.

Planned response:
- Treat frozen and adapter comparisons as cleanest between-encoder comparisons.
- Treat full FT comparison as informative but protocol-confounded.

## Likely issue: only two encoders

Risk: reviewer rejects objective-class interpretation.

Planned response:
- Frame as a case study of wav2vec2 XLS-R and MiMo-Audio-Tokenizer.
- Say results are consistent with, not proof of, objective-class hypotheses.
- Add HuBERT/WavLM only if core results are already clean.

## Likely issue: feature distortion claim

Risk: performance tables alone do not prove feature distortion.

Planned response:
- Weaken claim unless representation-drift evidence is added.
- Possible evidence: CKA/cosine drift between frozen and adapted encoders, class separation before/after adaptation.

## Likely issue: min-tDCF

Risk: EER-only reporting is insufficient for LA.

Planned response:
- Use official LA evaluator outputs for min-tDCF.
- State DF official evaluator reports EER only unless tDCF protocol is identified.
- Current evidence is in `TDCF_RECONCILIATION.md`: wav2vec2 adapter and MiMo full paper tDCF values use fewer seeds than their EER rows; MiMo frozen all-five tDCF cannot be computed because seed1234 score file is missing.

## Likely issue: MiMo objective characterization

Risk: MiMo is hybrid, not pure reconstruction.

Planned response:
- Describe MiMo as hybrid reconstruction/audio-to-text.
- Avoid `contrastive vs reconstruction` as the title-level claim.
