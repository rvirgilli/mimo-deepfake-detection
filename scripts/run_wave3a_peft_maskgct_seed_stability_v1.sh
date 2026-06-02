#!/usr/bin/env bash
set -euo pipefail

FOLD="MASKGCT"
SEEDS=(123 2024)
PROTOCOL="features/mimodf/wave0/codecfake_plus_protocol.jsonl"
SPLIT_PLAN="docs/current/wave3a_codecfake_cosg_source_holdout_plan_v3.json"
XLSR="SSL_Anti-spoofing/xlsr2_300m.pt"
RUN_ROOT="experiments/runs/wave3a_xlsr_training_reference_peft_maskgct_seed_stability_v1"
DOC_ROOT="docs/current/wave3a_peft_maskgct_seed_stability_v1"
SEED42_RUN="experiments/runs/wave3a_xlsr_training_reference_peft_seed42_v1/xlsr_peft_adapter/seed_42/MASKGCT"
mkdir -p "$DOC_ROOT"

wait_for_gpu() {
  local threshold=20000
  while true; do
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 || echo 0)
    echo "GPU free MiB: $free (threshold $threshold)" >&2
    if [ "$free" -ge "$threshold" ]; then
      return 0
    fi
    sleep 60
  done
}

run_seed() {
  local seed="$1" batch="$2"
  local out_json="$DOC_ROOT/seed_${seed}.json"
  local run_dir="$RUN_ROOT/xlsr_peft_adapter/seed_${seed}/$FOLD"
  if [ -f "$run_dir/manifest.json" ] && grep -q '"status": "completed"' "$run_dir/manifest.json" && [ -s "$out_json" ]; then
    echo "skip completed seed=$seed" >&2
    return 0
  fi
  wait_for_gpu
  echo "=== xlsr_peft_adapter fold=$FOLD seed=$seed batch=$batch ===" >&2
  PYTHONPATH=. conda run -n mimo-df python -m mimodf train codecfake-xlsr \
    --split-plan "$SPLIT_PLAN" \
    --protocol "$PROTOCOL" \
    --fold "$FOLD" \
    --condition xlsr_peft_adapter \
    --seed "$seed" \
    --out "$RUN_ROOT" \
    --train-run \
    --epochs 10 \
    --save-checkpoints \
    --checkpoint-metric val_auroc \
    --batch-size "$batch" \
    --eval-batch-size "$batch" \
    --cut 64600 \
    --device cuda \
    --deterministic \
    --xlsr-checkpoint "$XLSR" > "$out_json"
}

for seed in "${SEEDS[@]}"; do
  if ! run_seed "$seed" 2; then
    echo "seed=$seed batch=2 failed; retry batch=1" >&2
    cp "$DOC_ROOT/seed_${seed}.json" "$DOC_ROOT/seed_${seed}.batch2_failed.json" 2>/dev/null || true
    run_seed "$seed" 1
  fi
done

python - <<'PY'
import json
from pathlib import Path

root = Path('experiments/runs/wave3a_xlsr_training_reference_peft_maskgct_seed_stability_v1/xlsr_peft_adapter')
doc_root = Path('docs/current/wave3a_peft_maskgct_seed_stability_v1')
seed42 = Path('experiments/runs/wave3a_xlsr_training_reference_peft_seed42_v1/xlsr_peft_adapter/seed_42/MASKGCT')
seeds = [42, 123, 2024]
rows = []
for seed in seeds:
    run_dir = seed42 if seed == 42 else root / f'seed_{seed}' / 'MASKGCT'
    metrics = json.loads((run_dir / 'metrics.json').read_text())
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    score_gap = metrics['score_summary_by_label']['spoof']['mean'] - metrics['score_summary_by_label']['bonafide']['mean']
    cm = metrics['confusion_matrix']
    predicted_spoof_rate = (cm[0][1] + cm[1][1]) / metrics['records']
    rows.append({
        'seed': seed,
        'condition': 'xlsr_peft_adapter',
        'fold': 'MASKGCT',
        'records': metrics['records'],
        'eer': metrics['eer'],
        'auroc': metrics['auroc'],
        'balanced_accuracy': metrics['balanced_accuracy'],
        'confusion_matrix': cm,
        'bonafide_recall': metrics['per_class_recall']['bonafide'],
        'spoof_recall': metrics['per_class_recall']['spoof'],
        'score_gap_spoof_minus_bonafide_mean': score_gap,
        'predicted_spoof_rate_at_0p5': predicted_spoof_rate,
        'best_epoch': manifest['result_summary']['best_epoch'],
        'best_validation_auroc': manifest['result_summary']['best_checkpoint_metric_value'],
        'checkpoint_sha256': manifest['result_summary']['best_checkpoint_sha256'],
        'scores_sha256': manifest['result_summary']['scores_sha256'],
        'metrics_sha256': manifest['result_summary']['metrics_sha256'],
    })
repeat_poor = [r for r in rows if r['seed'] in {123, 2024} and r['auroc'] < 0.5]
summary = {
    'schema': 'mimodf-wave3a-peft-maskgct-seed-stability/v1',
    'claim_scope': 'custom CoSG source-holdout diagnostic only; PEFT MASKGCT seed-stability test',
    'condition': 'xlsr_peft_adapter',
    'fold': 'MASKGCT',
    'seeds': seeds,
    'rows': rows,
    'seed_stability_result': {
        'new_seeds_below_chance_auroc': [r['seed'] for r in repeat_poor],
        'new_seed_count': 2,
        'all_new_seeds_repeat_below_chance': len(repeat_poor) == 2,
        'mean_eer_all_seeds': sum(r['eer'] for r in rows) / len(rows),
        'mean_auroc_all_seeds': sum(r['auroc'] for r in rows) / len(rows),
        'mean_balanced_accuracy_all_seeds': sum(r['balanced_accuracy'] for r in rows) / len(rows),
    },
    'decision_rule': {
        'both_new_seeds_below_chance': 'stable adaptation-induced source inversion candidate; proceed to full PEFT seeds 123/2024 and mechanism note',
        'mixed': 'unstable adaptation risk; inspect epoch/checkpoint variance before full matrix',
        'neither': 'seed42 anomaly; deprioritize MASKGCT mechanism and inspect stochasticity/checkpointing',
    },
}
doc_root.mkdir(parents=True, exist_ok=True)
(doc_root / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
lines = [
    '# Wave 3A PEFT MASKGCT seed-stability test',
    '',
    'Scope: custom CoSG source-holdout diagnostic only; not official CodecFake+ benchmark training.',
    '',
    '| Seed | EER | AUROC | Bal acc | Bon recall | Spoof recall | Score gap spoof-bon | Spoof rate @0.5 | Best epoch | Val AUROC |',
    '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
]
for r in rows:
    lines.append(
        f"| {r['seed']} | {r['eer']:.4f} | {r['auroc']:.4f} | {r['balanced_accuracy']:.4f} | "
        f"{r['bonafide_recall']:.4f} | {r['spoof_recall']:.4f} | "
        f"{r['score_gap_spoof_minus_bonafide_mean']:.4f} | {r['predicted_spoof_rate_at_0p5']:.4f} | "
        f"{r['best_epoch']} | {r['best_validation_auroc']:.4f} |"
    )
lines += [
    '',
    f"New seeds below-chance AUROC: `{summary['seed_stability_result']['new_seeds_below_chance_auroc']}`",
    f"All new seeds repeat below-chance: `{summary['seed_stability_result']['all_new_seeds_repeat_below_chance']}`",
    f"Mean EER all seeds: `{summary['seed_stability_result']['mean_eer_all_seeds']:.4f}`",
    f"Mean AUROC all seeds: `{summary['seed_stability_result']['mean_auroc_all_seeds']:.4f}`",
    f"Mean balanced accuracy all seeds: `{summary['seed_stability_result']['mean_balanced_accuracy_all_seeds']:.4f}`",
    '',
]
if summary['seed_stability_result']['all_new_seeds_repeat_below_chance']:
    lines += [
        '## Decision',
        '',
        'Both new seeds repeat below-chance MASKGCT AUROC. Treat MASKGCT as a stable adaptation-induced source-inversion candidate and proceed to the full PEFT 3-seed matrix plus mechanism note.',
        '',
    ]
elif repeat_poor:
    lines += [
        '## Decision',
        '',
        'Mixed seed result. Treat MASKGCT as an unstable adaptation risk and inspect epoch/checkpoint variance before full PEFT matrix.',
        '',
    ]
else:
    lines += [
        '## Decision',
        '',
        'New seeds do not repeat below-chance MASKGCT AUROC. Treat seed42 as a directional anomaly until stochasticity/checkpointing are understood.',
        '',
    ]
(doc_root / 'summary.md').write_text('\n'.join(lines))
PY
