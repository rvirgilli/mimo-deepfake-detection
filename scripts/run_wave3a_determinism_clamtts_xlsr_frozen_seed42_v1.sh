#!/usr/bin/env bash
set -euo pipefail
DOCROOT=docs/current/wave3a_determinism_clamtts_xlsr_frozen_seed42_v1
RUNROOT=experiments/runs/wave3a_xlsr_training_reference_determinism_v1
mkdir -p "$DOCROOT"
for REP in repeat_a repeat_b; do
  echo "=== deterministic CLAMTTS ${REP} ===" >&2
  conda run -n mimo-df python -m mimodf train codecfake-xlsr \
    --split-plan docs/current/wave3a_codecfake_cosg_source_holdout_plan_v3.json \
    --protocol features/mimodf/wave0/codecfake_plus_protocol.jsonl \
    --fold CLAMTTS \
    --condition xlsr_frozen_backend \
    --seed 42 \
    --out "$RUNROOT/$REP" \
    --train-run \
    --epochs 10 \
    --save-checkpoints \
    --checkpoint-metric val_auroc \
    --batch-size 4 \
    --eval-batch-size 4 \
    --cut 64600 \
    --device cuda \
    --deterministic \
    --xlsr-checkpoint SSL_Anti-spoofing/xlsr2_300m.pt > "$DOCROOT/${REP}.json"
done
python - <<'PY'
import hashlib, json
from pathlib import Path
root=Path('experiments/runs/wave3a_xlsr_training_reference_determinism_v1')
doc=Path('docs/current/wave3a_determinism_clamtts_xlsr_frozen_seed42_v1')
rows=[]
for rep in ['repeat_a','repeat_b']:
    run=root/rep/'xlsr_frozen_backend'/'seed_42'/'CLAMTTS'
    manifest=json.loads((run/'manifest.json').read_text())
    metrics=json.loads((run/'metrics.json').read_text())
    rows.append({
        'repeat':rep,
        'run_dir':str(run),
        'scores_sha256':manifest['result_summary']['scores_sha256'],
        'metrics_sha256':manifest['result_summary']['metrics_sha256'],
        'checkpoint_sha256':manifest['result_summary']['best_checkpoint_sha256'],
        'best_epoch':manifest['result_summary']['best_epoch'],
        'best_val_auroc':manifest['result_summary']['best_checkpoint_metric_value'],
        'eer':metrics['eer'],
        'auroc':metrics['auroc'],
        'balanced_accuracy':metrics['balanced_accuracy'],
    })
summary={
    'schema':'mimodf-wave3a-repeatability-check/v1',
    'fold':'CLAMTTS','condition':'xlsr_frozen_backend','seed':42,'deterministic':True,'repeats':rows,
    'scores_match':rows[0]['scores_sha256']==rows[1]['scores_sha256'],
    'metrics_match':rows[0]['metrics_sha256']==rows[1]['metrics_sha256'],
    'checkpoints_match':rows[0]['checkpoint_sha256']==rows[1]['checkpoint_sha256'],
    'best_epoch_match':rows[0]['best_epoch']==rows[1]['best_epoch'],
}
(doc/'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
lines=['# Wave 3A CLAMTTS repeatability check','','| Repeat | Best epoch | EER | AUROC | Bal acc | Scores SHA-256 | Metrics SHA-256 | Checkpoint SHA-256 |','|---|---:|---:|---:|---:|---|---|---|']
for r in rows:
    lines.append(f"| {r['repeat']} | {r['best_epoch']} | {r['eer']:.4f} | {r['auroc']:.4f} | {r['balanced_accuracy']:.4f} | `{r['scores_sha256']}` | `{r['metrics_sha256']}` | `{r['checkpoint_sha256']}` |")
lines += ['',f"Scores match: `{summary['scores_match']}`",f"Metrics match: `{summary['metrics_match']}`",f"Checkpoints match: `{summary['checkpoints_match']}`",f"Best epoch match: `{summary['best_epoch_match']}`",'']
(doc/'summary.md').write_text('\n'.join(lines))
PY
