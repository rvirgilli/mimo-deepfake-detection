# Controlled smoke transcript

Date: 2026-05-27
Status: public command transcript; no new full eval/training authorized

Purpose: give a public reader a small, copy/pasteable path through the controlled system without requiring GPU work, ASVspoof data, or model weights.

This is a transcript recipe, not a new result table. It demonstrates the plain-file experiment/run-index/release-gate machinery.

## Smoke A: no-data system smoke

Requires only the Python package dependencies needed for default tests.

```bash
cd mimo-deepfake-detection
RUN_ROOT=/tmp/mimodf-public-smoke-runs
rm -rf "$RUN_ROOT"

python -m mimodf experiment validate \
  docs/current/examples/experiment_spec_v1_minimal.yaml

python -m mimodf experiment init \
  docs/current/examples/experiment_spec_v1_minimal.yaml \
  --seed 42 \
  --root "$RUN_ROOT"

RUN_DIR=$(find "$RUN_ROOT" -path '*/seed_42' -type d | head -1)
python -m mimodf experiment inspect "$RUN_DIR"

python -m mimodf report index "$RUN_ROOT" \
  --provenance docs/current/main_table_provenance.yaml \
  --out "$RUN_ROOT/index.jsonl"

python -m mimodf report aggregate \
  --index "$RUN_ROOT/index.jsonl" \
  --out "$RUN_ROOT/aggregate.md"

python -m mimodf report compare \
  --index "$RUN_ROOT/index.jsonl" \
  --experiments wav2vec2_frozen mimo_frozen \
  --out "$RUN_ROOT/compare.md"

python -m mimodf audit release-gate --system-profile --strict
```

Expected behavior:

- spec validation prints a `sha256:` spec hash;
- run init writes `resolved_spec.yaml` and `manifest.json` under `$RUN_ROOT/.../seed_42/`;
- report index writes a JSONL index combining the planned run-layout record with historical provenance records;
- aggregate/compare write Markdown reports;
- system-profile release gate exits zero in a correctly configured workspace while explicitly allowing known historical artifact gaps as warnings.

This smoke does not load Torch, MiMo, wav2vec2, ASVspoof data, or model weights.

## Smoke B: dry-run eval plan

Requires local paths for a config, checkpoint, eval audio root, and LA scorer. It still does not load a model or run inference.

```bash
python -m mimodf eval plan \
  --config configs/publish/wav2vec2_adapter.yaml \
  --checkpoint experiments/paper_final/wav2vec2_adapter_multiseed/seed_123/models/w2v2_adapter_s123_seed123/epoch_12_eer_5.42.pth \
  --eval-root "$ASVSPOOF2021_LA_EVAL" \
  --score-out /tmp/mimodf-public-eval-plan/scores_LA_eval.txt \
  --track LA \
  --scorer local_dependencies/SSL_Anti-spoofing-clean/evaluate_2021_LA.py \
  --strict
```

Expected behavior:

- prints an evaluation dry-run plan;
- checks config/checkpoint/eval-root/scorer paths;
- checks that the score output path is available;
- exits nonzero if any required path is missing.

See `docs/current/ASVSPOOF_DATA_LAYOUT.md` for the required ASVspoof path contract.

## Smoke C: bounded wav2vec2 eval shape

Requires ASVspoof2021 LA audio, protocol files, the XLS-R checkpoint, and a compatible wav2vec2 adapter checkpoint.

This is the bounded command shape used for local smoke testing. It is intentionally limited by `--max-items 2` and should not be used for paper metrics.

```bash
conda run -n mimo-df python -m mimodf eval run \
  --config configs/publish/wav2vec2_adapter.yaml \
  --checkpoint experiments/paper_final/wav2vec2_adapter_multiseed/seed_123/models/w2v2_adapter_s123_seed123/epoch_12_eer_5.42.pth \
  --legacy-run-config experiments/paper_final/wav2vec2_adapter_multiseed/seed_123/models/w2v2_adapter_s123_seed123/config.yaml \
  --eval-root "$ASVSPOOF2021_LA_EVAL" \
  --protocols-path "$ASVSPOOF_PROTOCOLS" \
  --score-out /tmp/mimodf-public-eval-smoke/scores_LA_eval_smoke.txt \
  --track LA \
  --frontend wav2vec2 \
  --frontend-checkpoint "$WAV2VEC2_XLSR_CHECKPOINT" \
  --scorer local_dependencies/SSL_Anti-spoofing-clean/evaluate_2021_LA.py \
  --batch-size 1 \
  --num-workers 0 \
  --device cpu \
  --max-items 2 \
  --manifest-out /tmp/mimodf-public-eval-smoke/manifest.json
```

Expected behavior:

- writes an ASVspoof two-column score file;
- writes a manifest;
- does not run official scoring because `--score-official` is omitted;
- evaluates only two utterances.

Local prior smoke evidence, already summarized in `docs/current/SYSTEM_STATUS.md`:

- a two-utterance wav2vec2 adapter eval smoke wrote score/manifest artifacts;
- generated scores matched the corresponding historical score file within about `3.5e-4` absolute difference for those utterances;
- a 1000-utterance bounded wav2vec2 adapter reproduction had mean absolute score difference about `1.06e-4` and max absolute difference about `1.90e-2`.

Those are smoke/reproduction-audit facts only. They are not new paper metrics.

## Official LA scoring is separate

For a complete LA score file, run official scoring with the ASVspoof2021 LA key root, not the audio root:

```bash
python -m mimodf score official-la /path/to/complete_scores_LA_eval.txt \
  --eval-root "$ASVSPOOF2021_LA_KEYS" \
  --scorer local_dependencies/SSL_Anti-spoofing-clean/evaluate_2021_LA.py \
  --phase eval
```

Do not run official scoring on `--max-items` smoke score files. The official scorer checks score row count against `CM/trial_metadata.txt`.

## What this solves

This provides a public command transcript for:

- versioned spec validation;
- run-layout creation/inspection;
- run indexing/aggregation/comparison;
- system-profile release gate;
- dry-run eval planning;
- bounded eval smoke shape.

## What remains blocked

- Full reproducibility of historical paper rows remains blocked by 9 missing historical artifact paths.
- Full LA/DF eval reruns require explicit approval and a logged execution entry.
- MiMo exact score reproduction remains caveated by batch-size-sensitive encoder drift.
