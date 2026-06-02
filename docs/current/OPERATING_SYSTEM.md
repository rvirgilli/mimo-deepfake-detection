# Operating system for the paper rework

Purpose: keep humans and agents aligned while rebuilding the paper and research framework from evidence.

This is the control layer. If a future agent is unsure what to do, read this first, then `PROJECT_BRIEF.md`, then the task-specific doc.

## Source hierarchy

1. **Evidence artifacts**: score files, result files, configs, checkpoints, logs, manifests.
2. **Machine-readable contracts**: experiment specs, resolved specs, run manifests, score manifests, official metrics, run indexes.
3. **`RESULTS_PROVENANCE.md`**: audited interpretation of historical evidence.
4. **`EXPERIMENT_LOGGING_PROTOCOL.md` and `research_execution_log.jsonl`**: mandatory ledger for new research execution.
5. **`RESEARCH_FRAMEWORK_GUIDELINES.md` / `RESEARCH_SYSTEM_SPEC.md`**: active research-system philosophy and contracts.
6. **`PROJECT_BRIEF.md`**: current intent, scope, risks, and scientific framing.
7. **`DECISION_LOG.md`**: accepted project decisions.
8. **`TASK_BOARD.md`**: current execution plan.
9. **`docs/paper.tex`**: manuscript draft. It must lag the evidence, not lead it.
10. Archived docs/scripts: pointers only, not authority.

If two sources disagree, prefer the earlier item in this hierarchy.

## Scope

### In scope now

- Versioned research execution system: specs, manifests, run layout, component IDs, indexing, comparison/aggregation.
- Main-table provenance, seed by seed.
- Official/appropriate scoring from existing artifacts.
- Protocol audit: validation set, optimizer, loss weights, checkpoint selection.
- Conservative paper/research planning after provenance is complete.
- Small audit/eval/training tools if they reduce ambiguity without changing model behavior.

### Out of scope unless explicitly approved

- Long training runs.
- Broad HPO.
- Adding new baselines before a controlled spec/matrix is approved.
- Polishing prose before numbers are trusted.
- Deleting or rewriting historical artifacts.
- Heavy experiment platforms unless the file-based system is proven insufficient.

## Non-negotiable rules

1. No paper number without a provenance row.
2. No silent seed exclusions.
3. No GPU-expensive reruns without explicit approval and expected cost.
4. Existing artifacts first; eval-only reruns second; training last.
5. Main table before figures, robustness, or prose.
6. Claims must be narrower than evidence.
7. Archived docs can suggest where to look, but cannot validate a claim.
8. Prefer revising/removing weak claims over defending messy provenance.
9. Every new `mimodf` behavior needs focused tests before commit; untested code is not complete.
10. New research runs use versioned specs/manifests; historical runs are indexed with source type and reproducibility tier, not rewritten.
11. Prefer plain-file contracts over new infrastructure.
12. Every new research/experiment command must be logged in `docs/current/research_execution_log.jsonl`; failed and interrupted runs must be logged too.

## Session loop

At the start of a work session:

1. `git status -sb`
2. read this file if not already in context;
3. read `TASK_BOARD.md`;
4. pick one `Now` task only;
5. state or infer the evidence target before editing.

During work:

1. keep diffs small;
2. before research execution, follow `EXPERIMENT_LOGGING_PROTOCOL.md` and prepare/log the exact command, run ID, inputs, outputs, environment, and caveats;
3. after research execution, update `research_execution_log.jsonl` with status, timings, outputs, result summary, and failures if any;
4. record commands that matter in `RESULTS_PROVENANCE.md` or the relevant doc when they affect paper/research claims;
5. update `DECISION_LOG.md` when a policy/scientific decision is made;
6. stop and ask if evidence contradicts the intended claim;
7. when adding parameters/components, update the registry/spec docs rather than hiding behavior in code.

Before claiming completion:

1. run focused tests for changed behavior;
2. run relevant CLI smoke checks and `compileall` for touched Python packages;
3. run the broader suite when the environment supports required dependencies;
4. update the task status;
5. report what was verified and what remains uncertain.

## Stop-and-ask triggers

Ask before continuing if any of these happen:

- a central result is missing score files or checkpoints;
- seed inclusion/exclusion is ambiguous;
- a rerun would require meaningful GPU time;
- a manuscript claim requires a stronger causal statement than evidence supports;
- changing tracked source could alter historical reproducibility;
- a new field changes the meaning of old experiment records;
- an experiment/research command would run without a precise log entry;
- local artifacts appear corrupted or contradictory.

## Commit policy

Commit after coherent units:

- docs/control changes;
- provenance batch for one experiment family;
- scorer/audit tooling;
- paper rewrite section;
- research-system contract/migration slice.

Do not mix paper claim rewrites with provenance discovery unless the rewrite is directly caused by that batch.
