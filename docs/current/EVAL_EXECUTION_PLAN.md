# Evaluation execution plan

Purpose: extend `python -m mimodf eval plan` into a controlled `eval run` path without reintroducing opaque legacy execution.

Status: first implementation slice complete for wav2vec2 legacy LA smoke evaluation. Full eval reruns, MiMo eval, and training remain explicit-approval work.

## Goal

Produce ASVspoof-compatible score files from explicit configs and checkpoints, then optionally run official scoring.

The eval path should be boring and auditable:

1. validate plan;
2. construct dataset/model through narrow seams;
3. run inference under explicit device/batch settings;
4. write score file through `mimodf.scoring.write_scores`;
5. optionally call official scorer;
6. write a manifest recording all inputs/outputs.

## Non-goals

- No training.
- No HPO.
- No implicit checkpoint discovery.
- No silent protocol fallback by file existence.
- No hand-written score files outside the score-file contract.
- No paper-number update unless provenance is recorded.

## Proposed CLI

```bash
python -m mimodf eval run \
  --config configs/publish/wav2vec2_adapter.yaml \
  --checkpoint /path/to/checkpoint.pth \
  --database-path /data/asvspoof \
  --protocols-path /data/protocols \
  --eval-root /data/asvspoof2021 \
  --score-out /tmp/scores_LA_eval.txt \
  --track LA \
  --frontend wav2vec2 \
  --validation-protocol asvspoof2021_fast \
  --scorer local_dependencies/SSL_Anti-spoofing-clean/evaluate_2021_LA.py \
  --batch-size 14 \
  --num-workers 4 \
  --device cuda:0 \
  --manifest-out /tmp/eval_manifest.json
```

Safe defaults:

- `eval plan` remains the default dry-run path.
- `eval run` requires all paths explicitly.
- `--device` defaults to `cpu` unless user opts into CUDA.
- Existing `--score-out` must fail unless `--overwrite` is supplied.
- Official scoring is opt-in with `--score-official` or implicit only when `--scorer` is provided and `--track LA`; choose one policy and test it.

## Module shape

Add one narrow module:

- `mimodf/evaluation/run.py`

Suggested public types/functions:

```python
@dataclass(frozen=True)
class EvaluationRunSettings:
    batch_size: int
    num_workers: int
    device: str
    overwrite: bool = False
    score_official: bool = False

@dataclass(frozen=True)
class EvaluationRunResult:
    score_file: str
    manifest_file: str | None
    official_result_file: str | None

@dataclass(frozen=True)
class EvaluationComponents:
    batches: Iterable[EvaluationBatch[Any]]
    predict_batch: PredictBatch[Any]


def run_evaluation(
    plan: EvaluationPlan,
    components: EvaluationComponents,
    settings: EvaluationRunSettings,
) -> EvaluationRunResult:
    ...
```

Keep real legacy construction separate:

- `mimodf/evaluation/legacy_components.py`

That file may import Torch/legacy modules lazily. Tests for `run_evaluation` should use fake components and require no Torch.

## Reuse existing seams

Already implemented:

- `mimodf.evaluation.plan.build_evaluation_plan`
- `mimodf.scoring.evaluate.EvaluationBatch`
- `mimodf.scoring.evaluate.write_scores_from_batches`
- `mimodf.scoring.torch_eval.TorchBatchPredictor`
- `mimodf.scoring.official.run_official_la_scorer`
- `mimodf.data.asvspoof` path/protocol planning
- `mimodf.training.legacy_model` lazy model/frontend factories
- `mimodf.training.manifest` manifest primitives

Do not duplicate these.

## Execution flow

### 1. Plan first

`eval run` must call `build_evaluation_plan(...)` before importing Torch or legacy code.

If the plan fails, exit before side effects.

### 2. Build components

For MVP, support one legacy path:

- ASVspoof waveform dataset;
- legacy model factory;
- checkpoint load;
- `TorchBatchPredictor` wrapper.

The component builder owns all framework details:

- device movement;
- checkpoint state-dict shape handling;
- dataloader construction;
- waveform preprocessing;
- model output-to-spoof-score conversion.

The generic runner should not know those details.

### 3. Score file contract

All outputs go through:

```python
write_scores_from_batches(...)
```

This preserves:

- one score per utterance;
- duplicate-id rejection;
- deterministic ordering;
- ASVspoof two-column format.

### 4. Official scoring

For LA only, call:

```python
run_official_la_scorer(...)
```

Reject wrong-scale project result files exactly as current scorer wrapper does.

DF remains EER-only unless an official DF scoring protocol is pinned.

### 5. Manifest

Write an eval manifest with at least:

- git state;
- config path and parsed config hash;
- checkpoint path/hash;
- scorer path/hash;
- eval root/protocol roots;
- score file path/hash;
- command args;
- status: `completed` or `failed`;
- error message on failure;
- official scorer stdout/stderr path if run.

Reuse `mimodf.training.manifest` patterns if possible, but do not force training terminology into eval if it makes the file unclear.

## Tests before implementation is complete

Required lightweight tests:

1. `eval run` with fake components writes deterministic score file.
2. Duplicate utterance IDs fail before official scoring.
3. Predictor score-count mismatch fails and records failure if manifest is requested.
4. Existing `--score-out` fails without `--overwrite`.
5. `eval run` calls `build_evaluation_plan` before component construction.
6. LA official scoring command is invoked only after score file is written.
7. Default `pytest -q` does not import Torch.
8. CLI strict failures return nonzero.

Optional `mimo-df` env tests:

- Torch predictor uses `eval()` and `inference_mode()`.
- Tiny fake Torch model/checkpoint can run through legacy-compatible adapter.

## Implementation status

Completed:

- `mimodf/evaluation/run.py` generic fake-testable runner;
- `mimodf/evaluation/legacy_components.py` for legacy ASVspoof wav2vec2 evaluation;
- `eval run` CLI with explicit paths, overwrite guard, manifest output, `--max-items`, and optional official scoring;
- tests for score writing, overwrite guard, manifest failure, and historical model-architecture reconstruction.

First real smoke:

```bash
conda run -n mimo-df python -m mimodf eval run \
  --config configs/publish/wav2vec2_adapter.yaml \
  --checkpoint experiments/paper_final/wav2vec2_adapter_multiseed/seed_123/models/w2v2_adapter_s123_seed123/epoch_12_eer_5.42.pth \
  --legacy-run-config experiments/paper_final/wav2vec2_adapter_multiseed/seed_123/models/w2v2_adapter_s123_seed123/config.yaml \
  --eval-root <datasets>/ASVspoof2021_LA_eval \
  --protocols-path SSL_Anti-spoofing/database \
  --score-out /tmp/mimodf-real-eval-smoke/scores_LA_eval_smoke.txt \
  --track LA \
  --frontend wav2vec2 \
  --frontend-checkpoint SSL_Anti-spoofing/xlsr2_300m.pt \
  --scorer local_dependencies/SSL_Anti-spoofing-clean/evaluate_2021_LA.py \
  --batch-size 1 \
  --num-workers 0 \
  --device cuda:0 \
  --max-items 2 \
  --manifest-out /tmp/mimodf-real-eval-smoke/manifest.json
```

Result: score file and manifest written; generated scores matched the existing historical seed-123 score file for the same two utterances within about `3.5e-4` absolute difference.

Next slices:

- run a bounded larger LA subset if needed for throughput/stability;
- run one full LA eval only if the 30-minute budget is likely enough;
- add MiMo-specific eval reconstruction only after wav2vec2 path remains stable;
- then add training smoke tests.

## Approval needed before Slice 3/4

Real eval may touch GPUs, large datasets, legacy model code, and official scorer assumptions. Ask before running it.
