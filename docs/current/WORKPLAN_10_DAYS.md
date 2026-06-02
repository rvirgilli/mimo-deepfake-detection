# 10-day workplan / phase history

## Objective

Make the paper and future experiments reviewable by replacing inherited claims with audited evidence and a lean research execution framework. Avoid costly GPU work unless a central result cannot be validated otherwise.

Current status: assessment phase concluded in `ASSESSMENT_CONCLUSION.md`. The system-audit milestone is complete. The active phase is versioned research-framework implementation, guided by `RESEARCH_FRAMEWORK_GUIDELINES.md`, `RESEARCH_SYSTEM_SPEC.md`, and `RESEARCH_SYSTEM_MIGRATION_PLAN.md`.

## Day 1-2: Evidence ledger — done

- Fill `RESULTS_PROVENANCE.md` for the current main table in `paper.tex`.
- Mark each row as `verified`, `partial`, `unverified`, `invalid`, or `verified_external`.
- Identify missing artifacts for MiMo adapter, MiMo frozen, MiMo full, wav2vec2 full.
- Decide which current paper claims are unsupported.

Deliverable: main-table provenance complete enough for assessment; remaining gaps are explicit.

## Day 3-4: Critical result fixes without training — partly done

- Recompute or locate official LA min-tDCF from existing score files. Done in `TDCF_RECONCILIATION.md`.
- Audit seed inclusion/exclusion, especially MiMo frozen seed 456. Done for current evidence: include by default.
- Audit MiMo adapter n=5 claim. Done for current evidence: unsupported locally; n=2 found.
- Validation protocol matrix started in `VALIDATION_PROTOCOL_MATRIX.md`; open questions remain.

Deliverable: corrected result table draft and tDCF reconciliation exist; protocol matrix needs gap closure.

## Day 5-6: Robustness analyses from existing artifacts

- Val-vs-eval correlation for MiMo full and adapter where possible.
- Per-attack summary from existing score files.
- Early-stopping/checkpoint-selection summary from logs/manifests.
- If feasible without training, representation-drift analysis using existing checkpoints and a bounded validation subset.

Deliverable: one robustness section outline plus figures/tables if evidence supports them.

## Day 7-8: Assessment-to-rewrite gate — done

Closed in `ASSESSMENT_CONCLUSION.md`.

Next rewrite, if chosen, must:

- replace causal objective-class claims with model-specific, track-specific claims;
- weaken feature-distortion language unless representation evidence supports it;
- state protocol asymmetries plainly;
- use MiMo adapter only as n=2 exploratory;
- use corrected tDCF values/footnotes.

Deliverable: rewrite-ready evidence package completed.

## Day 9 / current phase: research-framework execution

Current default work:

- controlled experiment specs before any new matrix;
- one stored new-layout smoke/reproduction example;
- eval-only reproduction only when stored under a versioned spec;
- no new training unless approved.

Potential training only if unavoidable and explicitly approved:

- controlled MiMo adapter seed completion;
- small baseline matrix tied to a declared hypothesis;
- Optuna study under `ExperimentSpec`, not ad hoc scripts.

## Day 10: Review response preparation

- Keep `REVIEW_PREP.md` updated with audited evidence.
- Prepare short responses for likely objections: provenance, seeds/outliers, validation subset, tDCF, asymmetry, two-model scope, MiMo hybrid objective, feature distortion.
- Freeze a clean paper TODO list for when reviews return.

## Non-goals

- Broad HPO.
- New long training campaigns.
- Adding many baselines before a controlled spec/matrix is approved.
- Heavy experiment platforms before the plain-file contract system is insufficient.
- Polishing prose before evidence is clean.
