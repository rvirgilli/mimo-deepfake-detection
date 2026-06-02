#!/usr/bin/env bash
set -uo pipefail

START_TS=$(date +%s)
DEADLINE=$((START_TS + 21600))
FOLDS=(CLAMTTS GPST MASKGCT NS2 NS3 SIMPLESPEECH1 SIMPLESPEECH2 UNIAUDIO VALLE)
SEED=42
BATCH=14
PROTOCOL="features/mimodf/wave0/codecfake_plus_protocol.jsonl"
SPLIT_PLAN="docs/current/wave3a_codecfake_cosg_source_holdout_plan_v3.json"
XLSR="SSL_Anti-spoofing/xlsr2_300m.pt"
RUN_ROOT="experiments/runs/wave3a_xlsr_peft_batch14_seed42_allfolds_v1"
DOC_ROOT="docs/current/wave3a_peft_batch14_seed42_allfolds_v1"
mkdir -p "$DOC_ROOT"

wait_for_gpu_stable() {
  local threshold=22000
  local needed=3
  local count=0
  while true; do
    local now free
    now=$(date +%s)
    if [ "$now" -ge "$DEADLINE" ]; then
      echo "deadline reached before next GPU job" >&2
      return 124
    fi
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 || echo 0)
    echo "GPU free MiB: $free (threshold $threshold, stable $count/$needed)" >&2
    if [ "$free" -ge "$threshold" ]; then
      count=$((count + 1))
      if [ "$count" -ge "$needed" ]; then
        return 0
      fi
      sleep 10
    else
      count=0
      sleep 60
    fi
  done
}

run_fold() {
  local fold="$1"
  local out_json="$DOC_ROOT/${fold}.json"
  local run_dir="$RUN_ROOT/xlsr_peft_adapter/seed_${SEED}/${fold}"
  if [ -f "$run_dir/manifest.json" ] && grep -q '"status": "completed"' "$run_dir/manifest.json" && [ -s "$out_json" ]; then
    echo "skip completed fold=$fold" >&2
    return 0
  fi
  wait_for_gpu_stable || return $?
  echo "=== xlsr_peft_adapter seed=$SEED fold=$fold batch=$BATCH ===" >&2
  PYTHONPATH=. conda run -n mimo-df python -m mimodf train codecfake-xlsr \
    --split-plan "$SPLIT_PLAN" \
    --protocol "$PROTOCOL" \
    --fold "$fold" \
    --condition xlsr_peft_adapter \
    --seed "$SEED" \
    --out "$RUN_ROOT" \
    --train-run \
    --epochs 10 \
    --save-checkpoints \
    --checkpoint-metric val_auroc \
    --batch-size "$BATCH" \
    --eval-batch-size "$BATCH" \
    --cut 64600 \
    --device cuda \
    --deterministic \
    --xlsr-checkpoint "$XLSR" > "$out_json"
}

summarize() {
  python - <<'PY'
import json
from pathlib import Path

folds = ['CLAMTTS','GPST','MASKGCT','NS2','NS3','SIMPLESPEECH1','SIMPLESPEECH2','UNIAUDIO','VALLE']
run_root = Path('experiments/runs/wave3a_xlsr_peft_batch14_seed42_allfolds_v1/xlsr_peft_adapter/seed_42')
doc_root = Path('docs/current/wave3a_peft_batch14_seed42_allfolds_v1')
peft_batch2_summary = Path('docs/current/wave3a_overnight_gpu_queue_2026_05_31_v1/peft_seed42/summary.json')
rows = []
for fold in folds:
    run_dir = run_root / fold
    metrics = json.loads((run_dir / 'metrics.json').read_text())
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    cm = metrics['confusion_matrix']
    score_gap = metrics['score_summary_by_label']['spoof']['mean'] - metrics['score_summary_by_label']['bonafide']['mean']
    rows.append({
        'fold': fold,
        'records': metrics['records'],
        'eer': metrics['eer'],
        'auroc': metrics['auroc'],
        'balanced_accuracy': metrics['balanced_accuracy'],
        'bonafide_recall': metrics['per_class_recall']['bonafide'],
        'spoof_recall': metrics['per_class_recall']['spoof'],
        'predicted_spoof_rate_at_0p5': (cm[0][1] + cm[1][1]) / metrics['records'],
        'score_gap_spoof_minus_bonafide_mean': score_gap,
        'confusion_matrix': cm,
        'best_epoch': manifest['result_summary']['best_epoch'],
        'best_validation_auroc': manifest['result_summary']['best_checkpoint_metric_value'],
        'checkpoint_sha256': manifest['result_summary']['best_checkpoint_sha256'],
        'scores_sha256': manifest['result_summary']['scores_sha256'],
        'metrics_sha256': manifest['result_summary']['metrics_sha256'],
    })
def mean(xs): return sum(xs)/len(xs)
batch2 = json.loads(peft_batch2_summary.read_text())
summary = {
    'schema': 'mimodf-wave3a-peft-batch14-seed42-allfolds/v1',
    'claim_scope': 'batch-policy pilot for custom CoSG source-holdout; PEFT batch14 seed42 only; not official CodecFake+ benchmark',
    'condition': 'xlsr_peft_adapter',
    'seed': 42,
    'batch_size': 14,
    'eval_batch_size': 14,
    'checkpoint_metric': 'val_auroc',
    'epochs': 10,
    'deterministic': True,
    'folds': rows,
    'mean_eer': mean([r['eer'] for r in rows]),
    'mean_auroc': mean([r['auroc'] for r in rows]),
    'mean_balanced_accuracy': mean([r['balanced_accuracy'] for r in rows]),
    'best_by_eer': min(rows, key=lambda r: r['eer'])['fold'],
    'worst_by_eer': max(rows, key=lambda r: r['eer'])['fold'],
    'batch14_minus_batch2_seed42': {
        'eer': mean([r['eer'] for r in rows]) - batch2['mean_eer'],
        'auroc': mean([r['auroc'] for r in rows]) - batch2['mean_auroc'],
        'balanced_accuracy': mean([r['balanced_accuracy'] for r in rows]) - batch2['mean_balanced_accuracy'],
    },
    'batch2_seed42_reference': {
        'summary_path': str(peft_batch2_summary),
        'mean_eer': batch2['mean_eer'],
        'mean_auroc': batch2['mean_auroc'],
        'mean_balanced_accuracy': batch2['mean_balanced_accuracy'],
    },
}
doc_root.mkdir(parents=True, exist_ok=True)
(doc_root / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
lines = [
    '# Wave 3A PEFT batch14 seed42 all-fold pilot',
    '',
    'Scope: batch-policy pilot over custom CoSG source-holdout; not official CodecFake+ benchmark.',
    '',
    '## Macro',
    '',
    f"Mean EER: `{summary['mean_eer']:.4f}`",
    f"Mean AUROC: `{summary['mean_auroc']:.4f}`",
    f"Mean balanced accuracy: `{summary['mean_balanced_accuracy']:.4f}`",
    f"Best fold by EER: `{summary['best_by_eer']}`",
    f"Worst fold by EER: `{summary['worst_by_eer']}`",
    '',
    'Batch14 minus prior PEFT batch2 seed42:',
    '',
    f"- EER: `{summary['batch14_minus_batch2_seed42']['eer']:.4f}`",
    f"- AUROC: `{summary['batch14_minus_batch2_seed42']['auroc']:.4f}`",
    f"- balanced accuracy: `{summary['batch14_minus_batch2_seed42']['balanced_accuracy']:.4f}`",
    '',
    '## Per fold',
    '',
    '| Fold | Records | EER | AUROC | Bal acc | Bon recall | Spoof recall | Spoof rate@0.5 | Score gap | Best epoch |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
]
for row in rows:
    lines.append(
        f"| {row['fold']} | {row['records']} | {row['eer']:.4f} | {row['auroc']:.4f} | {row['balanced_accuracy']:.4f} | "
        f"{row['bonafide_recall']:.4f} | {row['spoof_recall']:.4f} | {row['predicted_spoof_rate_at_0p5']:.4f} | "
        f"{row['score_gap_spoof_minus_bonafide_mean']:.4f} | {row['best_epoch']} |"
    )
lines += ['', '## Caveat', '', 'This is seed42 only. It tests whether batch14 is viable and materially changes PEFT behavior; it is not a final model-comparison matrix.', '']
(doc_root / 'summary.md').write_text('\n'.join(lines))
PY
}

for fold in "${FOLDS[@]}"; do
  run_fold "$fold" || exit $?
done
summarize
