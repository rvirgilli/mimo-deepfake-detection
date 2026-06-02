# Weekly team update — 2026-05-31

Audience: research / engineering team
Scope: this week’s repo and Wave 3A evolution
Claim boundary: current empirical results are **custom CodecFake+ CoSG source-holdout diagnostics**, not official CodecFake+ benchmark or ASVspoof5 results.

## 1. Executive summary

This week we moved from paper-repair / MiMo-revival framing to an audited trained-transfer research program.

The new working thesis:

> Lightweight XLS-R adaptation can improve average held-out-generator transfer, while still exposing hidden worst-source failures under generator/source shift.

The most important scientific result is not just a better mean EER. It is the combination of:

- PEFT improves average CoSG source-holdout transfer in seed42 pilots;
- a matched frozen batch14 control is still pending before architecture-only PEFT claims;
- `MASKGCT` remains the critical worst-source case;
- larger, literature-aligned batch size improves ranking but does not remove threshold collapse;
- validation on non-heldout sources can miss source-specific deployment failures.

## 2. What changed this week

### Project framing

Old frame:

```text
MiMo revival / earlier-manuscript repair
```

New frame:

```text
audited representation-transfer and trained-validation diagnostics under source/media shift
```

Key decisions:

- MiMo is now a diagnostic contrast, not the project center.
- XLS-R is the current reference representation family.
- CoSG source-holdout is explicitly labeled custom diagnostic, not official CodecFake+.
- CoRS and ASVspoof5 are future validation tracks, not current evidence.

### Execution quality

We hardened the research workflow:

- every research-producing command is logged in `docs/current/research_execution_log.jsonl`;
- strict log validation passes;
- deterministic training uses seeded loaders and deterministic CuDNN settings;
- scoring uses the selected best checkpoint, not accidental final epoch;
- frozen XLS-R checkpoints are compact;
- score, metric, and checkpoint SHA-256 digests are logged;
- failed/interrupted runs are kept as records.

Current ledger state:

```text
research log rows: 190
strict validation: pass
```

## 3. Main empirical results

### Frozen XLS-R backend — deterministic 3-seed baseline

Custom CoSG source-holdout diagnostic, seeds `42/123/2024`:

| Metric | Mean ± std |
|---|---:|
| EER | `0.3401 ± 0.0331` |
| AUROC | `0.7146 ± 0.0448` |
| Balanced accuracy | `0.6057 ± 0.0650` |

Interpretation:

- frozen XLS-R is now a stable diagnostic baseline;
- it is not the target model, but a reference for source-transfer behavior.

### PEFT XLS-R seed42 — initial batch2 result

PEFT seed42, batch2, all 9 CoSG source-holdout folds:

| Metric | Value |
|---|---:|
| Mean EER | `0.2068` |
| Mean AUROC | `0.8486` |
| Mean balanced accuracy | `0.7247` |

PEFT batch2 improved EER on `8/9` folds vs frozen 3-seed mean.

Important caveat:

- PEFT batch2 is scientifically valid as a protocol;
- but batch2 vs frozen batch4 is not an architecture-only comparison.

### MASKGCT seed-stability diagnostic

PEFT MASKGCT seeds `42/123/2024`:

| Seed | EER | AUROC | Balanced acc | Spoof rate @0.5 |
|---:|---:|---:|---:|---:|
| 42 | `0.5521` | `0.4333` | `0.4887` | `0.9800` |
| 123 | `0.5226` | `0.4679` | `0.4983` | `0.9983` |
| 2024 | `0.3958` | `0.6282` | `0.4826` | `0.9774` |

Revised interpretation:

- not a clean stable below-chance inversion across every seed;
- yes a stable worst-source / threshold-collapse case;
- every seed predicts >95% of MASKGCT as spoof at threshold `0.5`;
- validation AUROC remains high while held-out MASKGCT threshold behavior is poor.

### Batch-size protocol check

We researched and tested batch-size feasibility.

Literature signal:

- XLS-R / SSL anti-spoofing recipes commonly use batch sizes around `5`, `8`, and `14`;
- EURECOM/Tak-style SSL anti-spoofing recipe uses batch size `14`.

Local model-smoke feasibility:

| Batch | Passed smoke | OOM |
|---:|---|---|
| 2 | yes | no |
| 4 | yes | no |
| 8 | yes | no |
| 10 | yes | no |
| 12 | yes | no |
| 14 | yes | no |
| 16 | yes | no |

Decision:

- use batch14 as the scientifically cleaner PEFT protocol going forward;
- treat earlier PEFT batch2 as valid directional evidence, not final model-comparison evidence.

### PEFT XLS-R seed42 — batch14 pilot

PEFT seed42, batch14, all 9 CoSG folds:

| Metric | Value |
|---|---:|
| Mean EER | `0.1908` |
| Mean AUROC | `0.8856` |
| Mean balanced accuracy | `0.7166` |
| Best fold by EER | `NS2` |
| Worst fold by EER | `MASKGCT` |

Batch14 vs prior PEFT batch2 seed42:

| Metric | Delta |
|---|---:|
| EER | `-0.0160` |
| AUROC | `+0.0370` |
| Balanced accuracy | `-0.0081` |

Interpretation:

- batch14 is viable and improves ranking quality;
- PEFT remains promising;
- threshold behavior still needs source-aware diagnostics.

## 4. The key scientific finding

`MASKGCT` is the clearest current failure-mode case.

Batch14 improves MASKGCT ranking substantially:

| PEFT protocol | MASKGCT EER | MASKGCT AUROC | Balanced acc | Spoof rate @0.5 |
|---|---:|---:|---:|---:|
| batch2 seed42 | `0.5521` | `0.4333` | `0.4887` | `0.9800` |
| batch14 seed42 | `0.3646` | `0.6830` | `0.5078` | `0.9870` |

So the story evolved:

Old interpretation:

```text
MASKGCT may be stable ranking inversion.
```

Current interpretation:

```text
MASKGCT is a mixed-ranking but stable threshold-collapse / worst-source-risk case.
```

That is still scientifically strong:

> Adaptation can improve average ranking but leave a hidden source-specific operating-point failure.

## 5. What we should not claim

Do **not** claim:

- official CodecFake+ benchmark performance;
- ASVspoof5 evidence;
- PEFT architecture alone beats frozen XLS-R;
- stable below-chance MASKGCT inversion;
- full fine-tuning is needed or beneficial;
- media robustness EER from temporary/smoke transform artifacts.

Allowed claim:

> In a custom CoSG source-holdout diagnostic, XLS-R PEFT with a literature-aligned batch size improves average seed42 transfer compared with earlier PEFT batch2 behavior, but MASKGCT remains a worst-source threshold-collapse case. A matched frozen batch14 control is still required before claiming a PEFT architecture advantage.

## 6. Proposed team-facing narrative

Slide 1 — Reset

```text
From MiMo revival to audited trained-transfer diagnostics
```

Slide 2 — Infrastructure

```text
strict execution log, deterministic runs, selected-checkpoint scoring, digest-tracked artifacts
```

Slide 3 — Baseline

```text
Frozen XLS-R 3-seed CoSG diagnostic baseline: EER 0.3401 ± 0.0331
```

Slide 4 — Adaptation is promising, control pending

```text
PEFT batch14 seed42: EER 0.1908, AUROC 0.8856; matched frozen batch14 next
```

Slide 5 — But failures remain

```text
MASKGCT: improved ranking, persistent threshold collapse
```

Slide 6 — Claim boundary

```text
custom diagnostic, not official CodecFake+ / ASVspoof5
```

Slide 7 — Next week

```text
matched batch14 frozen control, then batch14 multi-seed PEFT/frozen matrix
```

## 7. Recommended next-week plan

### P0 — matched frozen batch14 seed42 control

Run:

```text
wave3a-frozen-batch14-seed42-allfolds-v1
```

Purpose:

- check whether batch14 improves frozen baseline too;
- avoid claiming PEFT improvement when batch policy may explain part of the gain.

### P1 — decide after P0

If PEFT batch14 seed42 remains clearly better than frozen batch14 seed42:

```text
run batch14 seeds 123/2024 for PEFT and/or frozen
```

If frozen batch14 catches up:

```text
shift thesis toward batch/update protocol and source-threshold failure, not PEFT advantage
```

### P2 — mechanism note

Write:

```text
Adaptation and operating-point failure under held-out codec-generator shift
```

Include:

- MASKGCT score distributions;
- threshold curves;
- source-wise calibration;
- validation-vs-heldout mismatch;
- batch2 vs batch14 comparison.

### P3 — data-roadmap work, no training yet

Plan only:

- CoRS extraction/index/readability/label-policy;
- ASVspoof5 Track 1 staging and scorer integration;
- durable media-transform scoring infrastructure.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Batch size confounds model comparison | Match batch14 frozen and PEFT before architecture claims |
| CoSG custom split overclaimed | Label all CoSG source-holdout results as diagnostic |
| MASKGCT story overstated | Say threshold-collapse/worst-source risk, not stable inversion |
| Too much compute too soon | Use seed42 controls before 3-seed matrices |
| Official datasets distract | Defer CoRS/ASVspoof5 until local mechanism is clean |

## 9. One-line status

> This week we turned the repo into an audited research system and found a promising seed42 trained-transfer story: XLS-R PEFT batch14 improves average custom CoSG source-holdout performance in the pilot, while MASKGCT exposes a persistent source-specific operating-point failure that will drive next week’s matched control.
