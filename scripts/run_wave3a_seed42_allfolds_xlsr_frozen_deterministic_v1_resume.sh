#!/usr/bin/env bash
set -euo pipefail
FOLDS=(SIMPLESPEECH2 UNIAUDIO VALLE)
DOCROOT=docs/current/wave3a_seed42_allfolds_xlsr_frozen_deterministic_v1
RUNROOT=experiments/runs/wave3a_xlsr_training_reference_seed42_allfolds_deterministic_v1
DEADLINE=$((SECONDS+7200))
while true; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 || echo 0)
  echo "GPU free MiB: $FREE" >&2
  if [ "$FREE" -ge 20000 ]; then break; fi
  if [ "$SECONDS" -ge "$DEADLINE" ]; then echo "GPU wait timed out" >&2; exit 75; fi
  sleep 60
done
for FOLD in "${FOLDS[@]}"; do
  echo "=== deterministic resume fold ${FOLD} ===" >&2
  conda run -n mimo-df python -m mimodf train codecfake-xlsr \
    --split-plan docs/current/wave3a_codecfake_cosg_source_holdout_plan_v3.json \
    --protocol features/mimodf/wave0/codecfake_plus_protocol.jsonl \
    --fold "$FOLD" \
    --condition xlsr_frozen_backend \
    --seed 42 \
    --out "$RUNROOT" \
    --train-run \
    --epochs 10 \
    --save-checkpoints \
    --checkpoint-metric val_auroc \
    --batch-size 4 \
    --eval-batch-size 4 \
    --cut 64600 \
    --device cuda \
    --deterministic \
    --xlsr-checkpoint SSL_Anti-spoofing/xlsr2_300m.pt > "$DOCROOT/${FOLD}.json"
done
python - <<'PY'
import json
from pathlib import Path
folds=['CLAMTTS','GPST','MASKGCT','NS2','NS3','SIMPLESPEECH1','SIMPLESPEECH2','UNIAUDIO','VALLE']
docroot=Path('docs/current/wave3a_seed42_allfolds_xlsr_frozen_deterministic_v1')
runroot=Path('experiments/runs/wave3a_xlsr_training_reference_seed42_allfolds_deterministic_v1')
rows=[]
for fold in folds:
    run_dir=runroot/'xlsr_frozen_backend'/'seed_42'/fold
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
summary={'schema':'mimodf-wave3a-seed42-allfolds-summary/v2','condition':'xlsr_frozen_backend','seed':42,'checkpoint_metric':'val_auroc','epochs':10,'deterministic':True,'folds':rows,'mean_eer':sum(r['eer'] for r in rows)/len(rows),'mean_auroc':sum(r['auroc'] for r in rows)/len(rows),'mean_balanced_accuracy':sum(r['balanced_accuracy'] for r in rows)/len(rows),'caveats':['single seed directional diagnostic only','custom CoSG source-holdout split; not official CodecFake+ benchmark','no model-comparison or seed-general claim']}
(docroot/'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
lines=['# Wave 3A deterministic seed42 all-fold XLS-R frozen summary','','Condition: `xlsr_frozen_backend`','Seed: `42`','Checkpoint metric: `val_auroc`','Deterministic: `true`','','| Fold | Records | EER | AUROC | Bal acc | Bon recall | Spoof recall | Best epoch |','|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['fold']} | {r['records']} | {r['eer']:.4f} | {r['auroc']:.4f} | {r['balanced_accuracy']:.4f} | {r['bonafide_recall']:.4f} | {r['spoof_recall']:.4f} | {r['best_epoch']} |")
lines += ['',f"Mean EER: `{summary['mean_eer']:.4f}`",f"Mean AUROC: `{summary['mean_auroc']:.4f}`",f"Mean balanced accuracy: `{summary['mean_balanced_accuracy']:.4f}`",'', 'Caveats:', '- Single seed directional diagnostic only.', '- Custom CoSG source-holdout split; not official CodecFake+ benchmark.', '- No model-comparison or seed-general claim.', '']
(docroot/'summary.md').write_text('\n'.join(lines))
PY
