#!/usr/bin/env bash
set -uo pipefail
RUN_ID="wave3a-overnight-gpu-queue-2026-05-31-v1"
START_TS=$(date +%s)
DEADLINE=$((START_TS + 21600))
FOLDS=(CLAMTTS GPST MASKGCT NS2 NS3 SIMPLESPEECH1 SIMPLESPEECH2 UNIAUDIO VALLE)
PROTOCOL="features/mimodf/wave0/codecfake_plus_protocol.jsonl"
SPLIT_PLAN="docs/current/wave3a_codecfake_cosg_source_holdout_plan_v3.json"
XLSR="SSL_Anti-spoofing/xlsr2_300m.pt"
QUEUE_ROOT="docs/current/wave3a_overnight_gpu_queue_2026_05_31_v1"
mkdir -p "$QUEUE_ROOT"

update_log() {
  local status="$1"
  local failure_msg="${2:-}"
  python - "$RUN_ID" "$status" "$failure_msg" <<'PY'
import json, subprocess, sys, time
from pathlib import Path
run_id, status, failure_msg = sys.argv[1], sys.argv[2], sys.argv[3]
log = Path('docs/current/research_execution_log.jsonl')
queue_root = Path('docs/current/wave3a_overnight_gpu_queue_2026_05_31_v1')
summary_paths = sorted(queue_root.glob('*/summary.json'))
summaries = {}
for path in summary_paths:
    try:
        summaries[path.parent.name] = json.loads(path.read_text())
    except Exception as exc:
        summaries[path.parent.name] = {'error': str(exc)}
outputs = [str(queue_root)] + [str(p) for p in summary_paths]
rows = []
for line in log.read_text().splitlines():
    rec = json.loads(line)
    if rec.get('run_id') == run_id:
        rec.update({
            'status': status,
            'git_revision_at_run': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
            'git_dirty_at_run': True,
            'finished_unix': time.time(),
            'outputs': outputs,
            'result_summary': {
                'matrices_with_summary': sorted(summaries),
                'summaries': {
                    name: {
                        'folds_completed': len(value.get('folds', [])) if isinstance(value, dict) else None,
                        'mean_eer': value.get('mean_eer') if isinstance(value, dict) else None,
                        'mean_auroc': value.get('mean_auroc') if isinstance(value, dict) else None,
                        'mean_balanced_accuracy': value.get('mean_balanced_accuracy') if isinstance(value, dict) else None,
                        'condition': value.get('condition') if isinstance(value, dict) else None,
                        'seed': value.get('seed') if isinstance(value, dict) else None,
                    }
                    for name, value in summaries.items()
                },
                'claim_scope': 'overnight queue artifacts; each matrix remains directional unless confirmatory seed policy is satisfied',
            },
        })
        if failure_msg:
            rec['failure'] = {'message': failure_msg}
    rows.append(json.dumps(rec, sort_keys=True))
log.write_text('\n'.join(rows) + '\n')
PY
}

on_exit() {
  code=$?
  if [ "$code" -eq 0 ]; then
    update_log completed ""
  elif [ "$code" -eq 124 ]; then
    update_log interrupted "overnight queue deadline reached"
  else
    update_log failed "overnight queue exited with code $code"
  fi
}
trap on_exit EXIT

wait_for_gpu() {
  local threshold="$1"
  while true; do
    local now
    now=$(date +%s)
    if [ "$now" -ge "$DEADLINE" ]; then
      echo "deadline reached before next GPU job" >&2
      exit 124
    fi
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 || echo 0)
    echo "GPU free MiB: $free (threshold $threshold)" >&2
    if [ "$free" -ge "$threshold" ]; then
      return 0
    fi
    sleep 60
  done
}

run_fold() {
  local condition="$1" seed="$2" fold="$3" runroot="$4" docroot="$5" batch="$6" threshold="$7"
  local out_json="$docroot/${fold}.json"
  local run_dir="$runroot/$condition/seed_${seed}/${fold}"
  mkdir -p "$docroot"
  if [ -f "$run_dir/manifest.json" ] && grep -q '"status": "completed"' "$run_dir/manifest.json" && [ -s "$out_json" ]; then
    echo "skip completed $condition seed=$seed fold=$fold" >&2
    return 0
  fi
  wait_for_gpu "$threshold"
  echo "=== $condition seed=$seed fold=$fold batch=$batch ===" >&2
  conda run -n mimo-df python -m mimodf train codecfake-xlsr \
    --split-plan "$SPLIT_PLAN" \
    --protocol "$PROTOCOL" \
    --fold "$fold" \
    --condition "$condition" \
    --seed "$seed" \
    --out "$runroot" \
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

summarize_matrix() {
  local condition="$1" seed="$2" runroot="$3" docroot="$4" schema="$5"
  python - "$condition" "$seed" "$runroot" "$docroot" "$schema" <<'PY'
import json, sys
from pathlib import Path
condition, seed, runroot, docroot, schema = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5]
folds=['CLAMTTS','GPST','MASKGCT','NS2','NS3','SIMPLESPEECH1','SIMPLESPEECH2','UNIAUDIO','VALLE']
rows=[]
for fold in folds:
    run_dir=runroot/condition/f'seed_{seed}'/fold
    metrics=json.loads((run_dir/'metrics.json').read_text())
    manifest=json.loads((run_dir/'manifest.json').read_text())
    rows.append({
        'fold': fold,
        'records': metrics['records'],
        'eer': metrics['eer'],
        'auroc': metrics['auroc'],
        'balanced_accuracy': metrics['balanced_accuracy'],
        'accuracy': metrics['accuracy'],
        'bonafide_recall': metrics['per_class_recall']['bonafide'],
        'spoof_recall': metrics['per_class_recall']['spoof'],
        'best_epoch': manifest['result_summary']['best_epoch'],
        'best_val_auroc': manifest['result_summary']['best_checkpoint_metric_value'],
        'checkpoint_sha256': manifest['result_summary']['best_checkpoint_sha256'],
        'scores_sha256': manifest['result_summary']['scores_sha256'],
        'metrics_sha256': manifest['result_summary']['metrics_sha256'],
        'checkpoint_bytes': (run_dir/'checkpoints'/'best.pt').stat().st_size,
    })
summary={
    'schema': schema,
    'condition': condition,
    'seed': seed,
    'checkpoint_metric': 'val_auroc',
    'epochs': 10,
    'deterministic': True,
    'folds': rows,
    'mean_eer': sum(r['eer'] for r in rows)/len(rows),
    'mean_auroc': sum(r['auroc'] for r in rows)/len(rows),
    'mean_balanced_accuracy': sum(r['balanced_accuracy'] for r in rows)/len(rows),
    'caveats': ['custom CoSG source-holdout split; not official CodecFake+ benchmark', 'no HPO', 'no seed-general claim unless all confirmatory seeds are present'],
}
docroot.mkdir(parents=True, exist_ok=True)
(docroot/'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
lines=[f'# Wave 3A deterministic {condition} seed {seed} all-fold summary','',f'Condition: `{condition}`',f'Seed: `{seed}`','Checkpoint metric: `val_auroc`','Deterministic: `true`','','| Fold | Records | EER | AUROC | Bal acc | Bon recall | Spoof recall | Best epoch |','|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['fold']} | {r['records']} | {r['eer']:.4f} | {r['auroc']:.4f} | {r['balanced_accuracy']:.4f} | {r['bonafide_recall']:.4f} | {r['spoof_recall']:.4f} | {r['best_epoch']} |")
lines += ['', f"Mean EER: `{summary['mean_eer']:.4f}`", f"Mean AUROC: `{summary['mean_auroc']:.4f}`", f"Mean balanced accuracy: `{summary['mean_balanced_accuracy']:.4f}`", '']
(docroot/'summary.md').write_text('\n'.join(lines))
PY
}

run_matrix() {
  local condition="$1" seed="$2" runroot="$3" docroot="$4" batch="$5" threshold="$6" schema="$7"
  mkdir -p "$docroot"
  for fold in "${FOLDS[@]}"; do
    run_fold "$condition" "$seed" "$fold" "$runroot" "$docroot" "$batch" "$threshold"
  done
  summarize_matrix "$condition" "$seed" "$runroot" "$docroot" "$schema"
}

# Confirmatory frozen backend seeds.
run_matrix xlsr_frozen_backend 123 \
  experiments/runs/wave3a_xlsr_training_reference_frozen_confirmatory_seed123_v1 \
  "$QUEUE_ROOT/frozen_seed123" 4 20000 mimodf-wave3a-frozen-confirmatory-seed123/v1

run_matrix xlsr_frozen_backend 2024 \
  experiments/runs/wave3a_xlsr_training_reference_frozen_confirmatory_seed2024_v1 \
  "$QUEUE_ROOT/frozen_seed2024" 4 20000 mimodf-wave3a-frozen-confirmatory-seed2024/v1

# PEFT adapter seed42. Try batch 2 first; retry failed fold once with batch 1.
PEFT_RUNROOT=experiments/runs/wave3a_xlsr_training_reference_peft_seed42_v1
PEFT_DOCROOT="$QUEUE_ROOT/peft_seed42"
mkdir -p "$PEFT_DOCROOT"
for fold in "${FOLDS[@]}"; do
  if ! run_fold xlsr_peft_adapter 42 "$fold" "$PEFT_RUNROOT" "$PEFT_DOCROOT" 2 20000; then
    echo "batch2 failed for PEFT $fold; retrying batch1" >&2
    cp "$PEFT_DOCROOT/${fold}.json" "$PEFT_DOCROOT/${fold}.batch2_failed.json" 2>/dev/null || true
    run_fold xlsr_peft_adapter 42 "$fold" "$PEFT_RUNROOT" "$PEFT_DOCROOT" 1 20000
  fi
  now=$(date +%s)
  if [ "$now" -ge "$DEADLINE" ]; then
    echo "deadline reached after $fold" >&2
    exit 124
  fi
done
summarize_matrix xlsr_peft_adapter 42 "$PEFT_RUNROOT" "$PEFT_DOCROOT" mimodf-wave3a-peft-seed42/v1

update_log completed ""
trap - EXIT
