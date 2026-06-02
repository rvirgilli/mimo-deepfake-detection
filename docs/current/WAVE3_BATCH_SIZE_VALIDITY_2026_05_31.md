# Wave 3 batch-size validity note — 2026-05-31

Status: validity note plus historical rationale. The current planning decision is in `WAVE3_REVISED_PLAN_2026_05_31.md`: batch14 is now the preferred PEFT protocol, and the next proposed control is frozen batch14 seed42.

Question: does PEFT batch size `2` have scientific validity?

Short answer: **yes, batch size 2 can be scientifically valid**, but only as an explicitly declared protocol. It is **not** a clean all-else-equal comparison against a batch-size-4 run unless we either match batch size or treat batch size as part of the condition. After this note, local feasibility and literature review motivated a batch14 PEFT pilot, so the old batch2-control recommendation below is superseded for next action.

## Facts at the time of the batch2 validity question

From the resolved specs that triggered this note:

| Run family | Condition | Batch size | Eval batch size | Epochs | LR | Deterministic |
|---|---|---:|---:|---:|---:|---|
| frozen all-fold deterministic | `xlsr_frozen_backend` | 4 | 4 | 10 | 0.0001 | true |
| PEFT seed42 all-fold | `xlsr_peft_adapter` | 2 | 2 | 10 | 0.0001 | true |
| PEFT MASKGCT seed-stability | `xlsr_peft_adapter` | 2 | 2 | 10 | 0.0001 | true |

So batch `2` is internally consistent for PEFT, but frozen-vs-PEFT currently has a batch-size confound.

## What the literature says

### Small batches are not unscientific

Masters & Luschi report strong performance with mini-batch sizes between `2` and `32`, and argue that small batches can generalize well.

- Masters & Luschi, *Revisiting Small Batch Training for Deep Neural Networks*: https://arxiv.org/abs/1804.07612

Keskar et al. report a large-batch generalization gap: large batches can converge to sharper minima, while smaller batches often find flatter minima.

- Keskar et al., *On Large-Batch Training for Deep Learning*: https://arxiv.org/abs/1609.04836

SGD noise is not merely nuisance; it can affect generalization and robustness.

- Smith et al., *On the Generalization Benefit of Noise in Stochastic Gradient Descent*: https://proceedings.mlr.press/v119/smith20a.html

Conclusion: there is no scientific rule that batch size must be at least `4`. Batch `2` is a real optimization regime.

### Batch size is a protocol variable

Batch size is a hyperparameter/control knob. Shallue et al. show batch-size effects are measurable and task-dependent.

- Shallue et al., *Measuring the Effects of Data Parallelism on Neural Network Training*: https://jmlr.org/beta/papers/v20/18-789.html

With fixed epochs and learning rate, batch `2` means roughly twice as many optimizer steps as batch `4`. That changes training dynamics. It is not a harmless memory setting.

### Very small batches can interact badly with BatchNorm

Our inherited backend code uses BatchNorm in `src/model.py` (`BatchNorm1d`, `BatchNorm2d`). Small batches can make BatchNorm statistics noisy. The GroupNorm paper highlights that BatchNorm accuracy degrades at very small batch sizes in vision settings, with batch size `2` being a known stress case.

- Wu & He, *Group Normalization*: https://openaccess.thecvf.com/content_ECCV_2018/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.html

This does not invalidate batch `2`, but it must be disclosed. It may even be part of the observed source-risk mechanism.

### Gradient accumulation is not a perfect escape hatch

Gradient accumulation can approximate a larger effective batch for gradients, but it does not fully reproduce larger-batch training when BatchNorm, dropout, data order, or variable sequence effects are present. If used, it becomes another protocol choice.

## Scientific validity classification

### Valid claims with batch 2

Allowed:

> Under the deterministic XLS-R PEFT protocol with batch size `2`, seed `42` improves average CoSG source-holdout performance but exhibits MASKGCT threshold collapse / mixed ranking risk.

Allowed:

> PEFT batch-size-2 behavior is a scientifically valid protocol observation.

Allowed:

> PEFT batch-size-2 results can be compared across PEFT seeds because seed42, MASKGCT seed-stability, and any future PEFT batch2 matrix share the same batch-size protocol.

### Not valid without caveat

Too strong:

> PEFT architecture alone beats frozen XLS-R.

Why: frozen used batch `4`, PEFT used batch `2`, so architecture/adaptation and batch/update schedule are confounded.

Too strong:

> Batch size 2 is equivalent to batch size 4.

Why: same epochs and LR imply different optimizer-step counts and gradient-noise scale.

### Valid with explicit boundary

Acceptable:

> In our current resource-bounded protocols, PEFT batch2 outperforms a frozen batch4 reference on average; batch size remains a protocol confound, so this is not an architecture-only comparison.

Better:

> We compare two trained detector protocols: frozen-backend batch4 and PEFT batch2. A matched-batch control is required before attributing all differences to adaptation.

## What should we do?

### Best scientific option

Run a matched control, not because batch `2` is invalid, but because comparability matters.

Most practical matched control:

```text
rerun frozen XLS-R backend at batch size 2, seeds 42/123/2024, all folds
```

Why:

- frozen is cheaper and compact;
- this matches PEFT batch2;
- then PEFT-vs-frozen comparison has matched batch size, epochs, LR, seeds, split, checkpoint metric.

### Alternative

Run PEFT at batch `4`:

```text
xlsr_peft_adapter batch4, seeds 42/123/2024, all folds
```

Why not preferred:

- may OOM;
- would require rerunning seed42 and MASKGCT seed-stability for consistency;
- more expensive.

### Minimal acceptable path if we do not rerun

Keep PEFT batch2 as a valid but condition-specific protocol:

- report batch size prominently;
- report optimizer steps per epoch or total train batches;
- avoid architecture-only claims;
- call frozen batch4 a reference, not a perfectly controlled baseline.

## Updated recommendation after batch14 feasibility/pilot

Do **not** discard PEFT batch2. It remains scientifically valid as an explicitly declared protocol and useful historical directional evidence.

However, batch14 is now the cleaner forward protocol because it is literature-aligned, locally feasible, and already completed for PEFT seed42 across all folds. Therefore the next best move is no longer a frozen batch2 matrix. It is the matched frozen batch14 seed42 control:

```text
wave3a-frozen-batch14-seed42-allfolds-v1
```

Then decide whether a full batch14 multi-seed matrix is worth the compute.

## Bottom line

Batch size `2` is not scientifically illegitimate. The problem is **comparability**, not validity. Batch14 is now preferred for the next PEFT/frozen comparison, but the same rule holds:

> Batch policy is part of the protocol. It cannot be hidden behind architecture claims.
