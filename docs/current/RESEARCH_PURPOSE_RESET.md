# Research purpose reset

Date: 2026-05-26
Status: accepted direction after Wave 1

## Decision

The project is no longer a MiMo revival project.

MiMo remains a candidate frontend/representation, but it has no privileged status. It can be compared, used as a diagnostic contrast, or dropped when evidence says it is not useful.

## Why

Wave 1 was useful because it killed the sunk-cost version of the hypothesis:

- MiMo has signal on CodecFake+ CoSG.
- MiMo is source-conditional.
- wav2vec2/XLSR wins most held-out-source probes.
- MiMo does not justify broad training, Optuna, or superiority claims.

That is progress, not failure. It means the research question should move up one level.

## New research purpose

Study **which pretrained audio representations transfer for audio deepfake detection under realistic distribution shifts**, and build an auditable framework for making those comparisons without hidden protocol drift.

The core question is not:

> Can MiMo beat wav2vec2?

The core question is:

> Under which data shifts, spoofing mechanisms, and media transformations do different frontend representation families help or fail?

Frontend families may include:

- SSL speech encoders: wav2vec2/XLSR, WavLM, HuBERT;
- tokenizer/codec/reconstruction encoders: MiMo, EnCodec-style representations;
- spectro-temporal baselines where useful;
- future representations only when tied to a declared shift/hypothesis.

## Candidate contribution

A defensible research contribution would be one of these:

1. **Representation transfer map**
   A controlled comparison showing how frontend families behave across generator/source, codec, and media-transformation shifts.

2. **Failure-mode taxonomy**
   Evidence that certain representation families fail predictably under specific spoofing mechanisms or transformations.

3. **Audit-first experimental framework**
   A reproducible, plain-file system for comparing audio deepfake frontends where every run has specs, manifests, logs, scoring contracts, and guarded comparisons.

4. **Conditional representation insight**
   A narrow finding such as: tokenizer features help for a specific generator/codec family but do not transfer broadly.

The current evidence most strongly supports (3), with evidence toward (2) and a narrow/source-local version of (4). It does not yet support (1) as a trained-model claim.

After Wave 2, the purpose has a second stage: use the audited cheap probes as hypothesis selectors, then run targeted trained validation where the evidence now justifies it.

## What changes now

- Stop treating MiMo as the central bet.
- Stop using “MiMo revival” language except as historical shorthand for Wave 0/1.
- Treat MiMo as one frontend in the component registry and experiment matrix.
- Future matrices should be organized around **shift types** and **representation families**, not around saving MiMo.
- New frontend additions need a declared purpose: what shift or failure mode they test.
- Wave 2 deepened one sharply scoped diagnostic and is now closed: the CLAMTTS mechanism did not transfer to NS2/NS3.
- Wave 3 may use expensive training, but only as targeted confirmatory validation tied to predeclared hypotheses, splits, seeds, budgets, and evaluation matrices.

## Near-term direction

Wave 1 and Wave 2 are complete enough to move from cheap elimination to trained validation.

Current next direction:

```text
docs/current/RESEARCH_WAVE_2_INTERIM_NOTE.md
docs/current/wave3_training_validation_plan.yaml
```

Wave 3 should test whether probe-discovered failure maps survive real training:

- XLS-R frozen backend vs PEFT/adapter vs full fine-tune;
- clean-only vs media-augmented XLS-R PEFT;
- CoRS proxy/pretraining vs CoSG evaluation, after CoRS audit and label policy;
- MiMo continuous only as a diagnostic contrast after the XLS-R reference is established.

The next non-compute work is to turn that plan into concrete specs, audit/index CoRS, and build the CodecFake training/scoring path.

## Guardrails

- No unmotivated broad training/Optuna; targeted expensive training is allowed only under the Wave 3 validation plan and explicit approval.
- No “MiMo superiority” language.
- No random-row metrics as main evidence.
- No new frontend unless it is tied to a shift/failure-mode question.
- Every experiment remains logged in `research_execution_log.jsonl`.
