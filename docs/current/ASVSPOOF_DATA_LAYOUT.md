# ASVspoof data and protocol layout

Date: 2026-05-27
Status: public setup contract for current controlled CLI paths

This document defines the local directory layout expected by the current `mimodf` ASVspoof evaluation/training helpers.

It does not redistribute ASVspoof data, protocols, or keys. Users must obtain data from official ASVspoof sources and point the repo at local copies.

## Terms

| Term | Meaning | Used by |
|---|---|---|
| protocol root | Directory containing `ASVspoof_LA_cm_protocols/` and/or `ASVspoof_DF_cm_protocols/` text files | training/eval dataset construction |
| training database root | Parent directory containing ASVspoof2019 train/dev and optional ASVspoof2021 fast-eval audio dirs | `train legacy-asvspoof` |
| inference eval root | Direct ASVspoof2021 eval audio directory containing `flac/` | `eval run` |
| official LA key root | ASVspoof2021 LA key/scoring directory containing `ASV/` and `CM/` metadata | `score official-la` |

Do not assume these roots are the same path.

## Environment variables

Recommended local exports:

```bash
export ASVSPOOF_PROTOCOLS=/data/asvspoof/protocols
export ASVSPOOF_LA_DATABASE=/data/asvspoof/LA
export ASVSPOOF_DF_DATABASE=/data/asvspoof/DF
export ASVSPOOF2021_LA_EVAL=/data/asvspoof/LA/ASVspoof2021_LA_eval
export ASVSPOOF2021_DF_EVAL=/data/asvspoof/DF/ASVspoof2021_DF_eval
export ASVSPOOF2021_LA_KEYS=/data/asvspoof/keys/LA
```

`.env.example` contains placeholder versions of these names.

## Protocol root layout

Default protocol files usually come from the Tak `SSL_Anti-spoofing/database` tree or official ASVspoof protocol/key packages.

Expected LA protocol root:

```text
$ASVSPOOF_PROTOCOLS/
  ASVspoof_LA_cm_protocols/
    ASVspoof2019.LA.cm.train.trn.txt
    ASVspoof2019.LA.cm.dev.trl.txt
    ASVspoof2021.LA.cm.eval.trl.txt
    ASVspoof2021.LA.cm.eval.fast.trl.txt      # optional, for fast validation
    ASVspoof2021.LA.cm.eval.fast.key.txt      # optional, for fast validation
```

Expected DF protocol root for eval-only paths:

```text
$ASVSPOOF_PROTOCOLS/
  ASVspoof_DF_cm_protocols/
    ASVspoof2021.DF.cm.eval.trl.txt
```

Current controlled training helpers are LA-oriented because they build an ASVspoof2019 train protocol path. DF evaluation can be planned/run as eval-only when the DF trial file and audio root are present, but official DF scoring is not pinned in this repo.

## Audio layout for training helpers

`mimodf.data.asvspoof.build_path_plan` expects `--database-path` to be a parent directory with track-specific subdirectories.

For LA:

```text
$ASVSPOOF_LA_DATABASE/
  ASVspoof2019_LA_train/
    flac/
      LA_T_*.flac
  ASVspoof2019_LA_dev/
    flac/
      LA_D_*.flac
  ASVspoof2021_LA_eval/
    flac/
      LA_E_*.flac
```

Training command shape:

```bash
python -m mimodf train legacy-asvspoof \
  --config configs/publish/wav2vec2_adapter.yaml \
  --out /tmp/mimodf-train-smoke \
  --database-path "$ASVSPOOF_LA_DATABASE" \
  --protocols-path "$ASVSPOOF_PROTOCOLS" \
  --validation-protocol asvspoof2019_dev \
  --frontend wav2vec2 \
  --frontend-checkpoint "$WAV2VEC2_XLSR_CHECKPOINT" \
  --dry-run
```

For ASVspoof2021 fast validation during training, the same `--database-path` must also contain:

```text
$ASVSPOOF_LA_DATABASE/ASVspoof2021_LA_eval/flac/
```

and the protocol root must contain the `*.eval.fast.trl.txt` and `*.eval.fast.key.txt` files.

## Audio layout for evaluation inference

`mimodf eval run` expects `--eval-root` to be the direct eval audio directory containing `flac/`.

LA eval inference root:

```text
$ASVSPOOF2021_LA_EVAL/
  flac/
    LA_E_*.flac
```

DF eval inference root:

```text
$ASVSPOOF2021_DF_EVAL/
  flac/
    DF_E_*.flac
```

Dry-run planning example:

```bash
python -m mimodf eval plan \
  --config configs/publish/wav2vec2_adapter.yaml \
  --checkpoint experiments/paper_final/wav2vec2_adapter_multiseed/seed_123/models/w2v2_adapter_s123_seed123/epoch_12_eer_5.42.pth \
  --eval-root "$ASVSPOOF2021_LA_EVAL" \
  --score-out /tmp/mimodf-eval/scores_LA_eval.txt \
  --track LA \
  --scorer local_dependencies/SSL_Anti-spoofing-clean/evaluate_2021_LA.py \
  --strict
```

Bounded eval smoke shape:

```bash
python -m mimodf eval run \
  --config configs/publish/wav2vec2_adapter.yaml \
  --checkpoint experiments/paper_final/wav2vec2_adapter_multiseed/seed_123/models/w2v2_adapter_s123_seed123/epoch_12_eer_5.42.pth \
  --legacy-run-config experiments/paper_final/wav2vec2_adapter_multiseed/seed_123/models/w2v2_adapter_s123_seed123/config.yaml \
  --eval-root "$ASVSPOOF2021_LA_EVAL" \
  --protocols-path "$ASVSPOOF_PROTOCOLS" \
  --score-out /tmp/mimodf-eval/scores_LA_eval_smoke.txt \
  --track LA \
  --frontend wav2vec2 \
  --frontend-checkpoint "$WAV2VEC2_XLSR_CHECKPOINT" \
  --batch-size 1 \
  --num-workers 0 \
  --device cpu \
  --max-items 2 \
  --manifest-out /tmp/mimodf-eval/manifest.json
```

Use CUDA only when explicitly approved and logged for research-producing runs.

## Official ASVspoof2021 LA scoring layout

The official LA scorer wrapper expects a key root, not an audio root.

Expected key root:

```text
$ASVSPOOF2021_LA_KEYS/
  ASV/
    trial_metadata.txt
    ASVTorch_Kaldi/
      score.txt
  CM/
    trial_metadata.txt
```

Score a complete LA score file:

```bash
python -m mimodf score official-la /path/to/scores_LA_eval.txt \
  --eval-root "$ASVSPOOF2021_LA_KEYS" \
  --scorer local_dependencies/SSL_Anti-spoofing-clean/evaluate_2021_LA.py \
  --phase eval
```

Important: the official scorer checks the score-file row count against `CM/trial_metadata.txt`. Do not run official scoring on `--max-items` smoke outputs.

## Current CLI caveat

`mimodf eval run --score-official` currently passes `--eval-root` through to the official scorer. Because `--eval-root` is the audio root for inference, the safer public workflow is two-step:

1. run `mimodf eval run` to create a complete score file;
2. run `mimodf score official-la` with `--eval-root "$ASVSPOOF2021_LA_KEYS"`.

Do not use `--score-official` unless your local layout intentionally makes the official key root available at the same path used for scoring.

## Minimal preflight checks

Protocol root:

```bash
test -f "$ASVSPOOF_PROTOCOLS/ASVspoof_LA_cm_protocols/ASVspoof2021.LA.cm.eval.trl.txt"
```

LA eval audio root:

```bash
test -d "$ASVSPOOF2021_LA_EVAL/flac"
```

LA official key root:

```bash
test -f "$ASVSPOOF2021_LA_KEYS/CM/trial_metadata.txt"
test -f "$ASVSPOOF2021_LA_KEYS/ASV/trial_metadata.txt"
test -f "$ASVSPOOF2021_LA_KEYS/ASV/ASVTorch_Kaldi/score.txt"
```

Dry-run a plan before any real eval:

```bash
python -m mimodf eval plan \
  --config configs/publish/wav2vec2_adapter.yaml \
  --checkpoint /path/to/checkpoint.pth \
  --eval-root "$ASVSPOOF2021_LA_EVAL" \
  --score-out /tmp/mimodf-eval/scores.txt \
  --track LA \
  --scorer local_dependencies/SSL_Anti-spoofing-clean/evaluate_2021_LA.py \
  --strict
```

## What remains blocked

This document makes path expectations explicit. It does not solve:

- missing historical checkpoints/configs;
- ASVspoof data access/licensing;
- full LA/DF reruns through the new CLI;
- MiMo batch-size-sensitive score drift.
