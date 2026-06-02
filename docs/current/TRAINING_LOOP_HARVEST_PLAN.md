# Training loop harvest plan

Purpose: migrate future controlled training paths out of legacy `train.py` without changing model science or historical artifacts.

Wave 3 update: the existing ASVspoof-oriented training seam is infrastructure evidence only. Targeted trained validation now needs a CodecFake training/scoring path before long GPU runs.

## Scope

This is for **future controlled runs only**. It does not reinterpret historical results and does not require rerunning training.

## Design target

Create one small training API:

```python
train_one_run(
    config: ExperimentConfig,
    model_factory: Callable[[], torch.nn.Module],
    loaders: TrainLoaders,
    output_dir: Path,
) -> TrainingRunResult
```

The API must make these facts explicit:

- seed;
- train set;
- validation/checkpoint-selection set;
- eval set;
- optimizer;
- scorer;
- checkpoint metric;
- manifest path;
- best checkpoint path.

## Migration order

1. Deterministic seeding utilities.
2. Checkpoint tracking/top-k retention.
3. Manifest writer with hashes and protocol/config summary.
4. Tiny `train_one_run` loop tested with fake Torch modules/data.
5. Only then connect real ASVspoof data/frontends/backend.

## What to harvest from legacy `train.py`

- Seed behavior: Python, NumPy, Torch, CUDA, cuDNN flags.
- Train epoch shape: model train, weighted cross entropy, optional grad clipping.
- Validation shape: model eval, explicit metric/loss choice.
- Optimizer split: Adam for `encoder_lr: null`; AdamW only with explicit encoder/backend groups.
- Top-k checkpoint retention by lower metric.
- Manifest start/complete lifecycle.

## What not to copy blindly

- File-existence-driven validation protocol fallback.
- Hidden validation loss weights.
- Ad hoc checkpoint naming as the only provenance record.
- SQLite results DB as source of truth.
- Mixed printing/logging side effects.
- Legacy Hydra object assumptions.

## Required tests before each training module is accepted

- seeding returns deterministic worker seeds;
- checkpoint tracker saves first k, replaces worst, deletes replaced checkpoints;
- manifest includes config/protocol/git/status and file hashes;
- fake one-epoch loop selects best checkpoint by configured metric;
- AdamW with `encoder_lr: null` remains impossible through config validation.

## Implementation progress

Completed foundation modules:

- `mimodf.training.seeding`
- `mimodf.training.checkpoint`
- `mimodf.training.manifest`
- `mimodf.training.loop` minimal `train_one_run`
- `mimodf.training.components` integration seam and optimizer builder
- `mimodf.data.asvspoof` explicit ASVspoof path planning/loader construction
- `mimodf.training.legacy_model` lazy legacy frontend/model factories

Real smoke status:

- `mimodf train legacy-asvspoof` supports `--max-train-batches` and `--max-val-batches` for bounded smoke tests.
- First real CUDA smoke completed for wav2vec2 adapter on ASVspoof LA with 1 train batch, 1 validation batch, batch size 1, RawBoost disabled.
- The smoke wrote a checkpoint and manifest under `/tmp/mimodf-real-train-smoke`.
- The smoke checkpoint was then loaded by `mimodf eval run` and scored on 2 LA eval utterances.

This is infrastructure validation only. It is not a publishable training result. Keep tests pure/default-env where possible; Torch-specific tests may run in `mimo-df`.

Wave 3 gap:

- CodecFake+ CoSG/CoRS train/dev/test loaders are not yet the main training path.
- CoRS label policy must be pinned before use as a proxy spoof-training source.
- Trained-model scoring must support clean/transformed paired comparisons and source-holdout reports.
