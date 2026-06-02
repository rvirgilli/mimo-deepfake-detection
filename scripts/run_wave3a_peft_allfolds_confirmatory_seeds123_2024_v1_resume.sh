#!/usr/bin/env bash
set -uo pipefail

RUN_ID="wave3a-peft-allfolds-confirmatory-seeds123-2024-v2-resume"
START_TS=$(date +%s)
DEADLINE=$((START_TS + 28800))
FOLDS=(CLAMTTS GPST MASKGCT NS2 NS3 SIMPLESPEECH1 SIMPLESPEECH2 UNIAUDIO VALLE)
SEEDS=(123 2024)
PROTOCOL="features/mimodf/wave0/codecfake_plus_protocol.jsonl"
SPLIT_PLAN="docs/current/wave3a_codecfake_cosg_source_holdout_plan_v3.json"
XLSR="SSL_Anti-spoofing/xlsr2_300m.pt"
RUN_ROOT="experiments/runs/wave3a_xlsr_training_reference_peft_confirmatory_seeds123_2024_v1"
DOC_ROOT="docs/current/wave3a_peft_allfolds_confirmatory_seeds123_2024_v1"
MASKGCT_RUN_ROOT="experiments/runs/wave3a_xlsr_training_reference_peft_maskgct_seed_stability_v1/xlsr_peft_adapter"
MASKGCT_DOC_ROOT="docs/current/wave3a_peft_maskgct_seed_stability_v1"
PEFT42_RUN_ROOT="experiments/runs/wave3a_xlsr_training_reference_peft_seed42_v1/xlsr_peft_adapter/seed_42"
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

run_dir_for() {
  local seed="$1" fold="$2"
  if [ "$fold" = "MASKGCT" ]; then
    echo "$MASKGCT_RUN_ROOT/seed_${seed}/MASKGCT"
  else
    echo "$RUN_ROOT/xlsr_peft_adapter/seed_${seed}/${fold}"
  fi
}

prepare_existing_maskgct_json() {
  local seed="$1"
  local out_dir="$DOC_ROOT/seed_${seed}"
  mkdir -p "$out_dir"
  if [ -s "$MASKGCT_DOC_ROOT/seed_${seed}.json" ] && [ ! -s "$out_dir/MASKGCT.json" ]; then
    cp "$MASKGCT_DOC_ROOT/seed_${seed}.json" "$out_dir/MASKGCT.json"
  fi
}

run_fold() {
  local seed="$1" fold="$2" batch="$3"
  local out_dir="$DOC_ROOT/seed_${seed}"
  local out_json="$out_dir/${fold}.json"
  local run_dir
  run_dir=$(run_dir_for "$seed" "$fold")
  mkdir -p "$out_dir"

  if [ "$fold" = "MASKGCT" ]; then
    prepare_existing_maskgct_json "$seed"
    if [ -f "$run_dir/manifest.json" ] && grep -q '"status": "completed"' "$run_dir/manifest.json"; then
      echo "skip precompleted seed=$seed fold=$fold" >&2
      return 0
    fi
  fi

  if [ -f "$run_dir/manifest.json" ] && grep -q '"status": "completed"' "$run_dir/manifest.json" && [ -s "$out_json" ]; then
    echo "skip completed seed=$seed fold=$fold" >&2
    return 0
  fi

  wait_for_gpu_stable || return $?
  echo "=== xlsr_peft_adapter seed=$seed fold=$fold batch=$batch ===" >&2
  PYTHONPATH=. conda run -n mimo-df python -m mimodf train codecfake-xlsr \
    --split-plan "$SPLIT_PLAN" \
    --protocol "$PROTOCOL" \
    --fold "$fold" \
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

summarize() {
  python - <<'PY'
import json, statistics
from pathlib import Path

folds = ['CLAMTTS','GPST','MASKGCT','NS2','NS3','SIMPLESPEECH1','SIMPLESPEECH2','UNIAUDIO','VALLE']
seeds_new = [123, 2024]
seeds_all = [42, 123, 2024]
root = Path('experiments/runs/wave3a_xlsr_training_reference_peft_confirmatory_seeds123_2024_v1/xlsr_peft_adapter')
mask_root = Path('experiments/runs/wave3a_xlsr_training_reference_peft_maskgct_seed_stability_v1/xlsr_peft_adapter')
seed42_root = Path('experiments/runs/wave3a_xlsr_training_reference_peft_seed42_v1/xlsr_peft_adapter/seed_42')
doc_root = Path('docs/current/wave3a_peft_allfolds_confirmatory_seeds123_2024_v1')
frozen_paths = {
    42: Path('docs/current/wave3a_seed42_allfolds_xlsr_frozen_deterministic_v1/summary.json'),
    123: Path('docs/current/wave3a_overnight_gpu_queue_2026_05_31_v1/frozen_seed123/summary.json'),
    2024: Path('docs/current/wave3a_overnight_gpu_queue_2026_05_31_v1/frozen_seed2024/summary.json'),
}

def run_dir(seed, fold):
    if seed == 42:
        return seed42_root / fold
    if fold == 'MASKGCT':
        return mask_root / f'seed_{seed}' / 'MASKGCT'
    return root / f'seed_{seed}' / fold

def load_metrics(seed, fold):
    rd = run_dir(seed, fold)
    metrics = json.loads((rd / 'metrics.json').read_text())
    manifest = json.loads((rd / 'manifest.json').read_text())
    cm = metrics['confusion_matrix']
    score_gap = metrics['score_summary_by_label']['spoof']['mean'] - metrics['score_summary_by_label']['bonafide']['mean']
    return {
        'seed': seed,
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
        'run_dir': str(rd),
    }

def mean(xs):
    return sum(xs) / len(xs)

def stdev(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0

rows = [load_metrics(seed, fold) for seed in seeds_all for fold in folds]
by_seed = {}
for seed in seeds_all:
    seed_rows = [r for r in rows if r['seed'] == seed]
    by_seed[str(seed)] = {
        'folds_completed': len(seed_rows),
        'mean_eer': mean([r['eer'] for r in seed_rows]),
        'mean_auroc': mean([r['auroc'] for r in seed_rows]),
        'mean_balanced_accuracy': mean([r['balanced_accuracy'] for r in seed_rows]),
        'worst_by_eer': max(seed_rows, key=lambda r: r['eer'])['fold'],
        'best_by_eer': min(seed_rows, key=lambda r: r['eer'])['fold'],
    }
by_fold = []
for fold in folds:
    fold_rows = [r for r in rows if r['fold'] == fold]
    by_fold.append({
        'fold': fold,
        'records': fold_rows[0]['records'],
        'eer_mean': mean([r['eer'] for r in fold_rows]),
        'eer_std': stdev([r['eer'] for r in fold_rows]),
        'auroc_mean': mean([r['auroc'] for r in fold_rows]),
        'auroc_std': stdev([r['auroc'] for r in fold_rows]),
        'balanced_accuracy_mean': mean([r['balanced_accuracy'] for r in fold_rows]),
        'balanced_accuracy_std': stdev([r['balanced_accuracy'] for r in fold_rows]),
        'predicted_spoof_rate_at_0p5_mean': mean([r['predicted_spoof_rate_at_0p5'] for r in fold_rows]),
        'score_gap_mean': mean([r['score_gap_spoof_minus_bonafide_mean'] for r in fold_rows]),
        'per_seed': fold_rows,
    })
frozen = {seed: json.loads(path.read_text()) for seed, path in frozen_paths.items()}
frozen_macro = {
    'mean_eer': mean([frozen[s]['mean_eer'] for s in seeds_all]),
    'std_eer': stdev([frozen[s]['mean_eer'] for s in seeds_all]),
    'mean_auroc': mean([frozen[s]['mean_auroc'] for s in seeds_all]),
    'std_auroc': stdev([frozen[s]['mean_auroc'] for s in seeds_all]),
    'mean_balanced_accuracy': mean([frozen[s]['mean_balanced_accuracy'] for s in seeds_all]),
    'std_balanced_accuracy': stdev([frozen[s]['mean_balanced_accuracy'] for s in seeds_all]),
}
peft_macro = {
    'mean_eer': mean([by_seed[str(s)]['mean_eer'] for s in seeds_all]),
    'std_eer': stdev([by_seed[str(s)]['mean_eer'] for s in seeds_all]),
    'mean_auroc': mean([by_seed[str(s)]['mean_auroc'] for s in seeds_all]),
    'std_auroc': stdev([by_seed[str(s)]['mean_auroc'] for s in seeds_all]),
    'mean_balanced_accuracy': mean([by_seed[str(s)]['mean_balanced_accuracy'] for s in seeds_all]),
    'std_balanced_accuracy': stdev([by_seed[str(s)]['mean_balanced_accuracy'] for s in seeds_all]),
}
summary = {
    'schema': 'mimodf-wave3a-peft-allfolds-confirmatory/v1',
    'claim_scope': 'custom CoSG source-holdout diagnostic only; not official CodecFake+ benchmark training',
    'condition': 'xlsr_peft_adapter',
    'seeds': seeds_all,
    'newly_run_seeds': seeds_new,
    'folds': folds,
    'by_seed': by_seed,
    'by_fold': by_fold,
    'peft_3seed_macro': peft_macro,
    'frozen_3seed_macro': frozen_macro,
    'peft_minus_frozen_macro': {
        'eer': peft_macro['mean_eer'] - frozen_macro['mean_eer'],
        'auroc': peft_macro['mean_auroc'] - frozen_macro['mean_auroc'],
        'balanced_accuracy': peft_macro['mean_balanced_accuracy'] - frozen_macro['mean_balanced_accuracy'],
    },
    'decision_notes': [
        'PEFT and frozen are now comparable over seeds 42/123/2024 in the custom CoSG source-holdout diagnostic.',
        'MASKGCT must be framed as mixed-ranking/stable-threshold-collapse risk, not stable below-chance inversion.',
        'No official CodecFake+ claim follows from this custom CoSG source-holdout matrix.',
    ],
}
doc_root.mkdir(parents=True, exist_ok=True)
(doc_root / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
lines = [
    '# Wave 3A PEFT all-fold confirmatory matrix',
    '',
    'Scope: custom CoSG source-holdout diagnostic only; not official CodecFake+ benchmark training.',
    '',
    '## Macro comparison',
    '',
    '| Condition | Seeds | Mean EER | Mean AUROC | Mean balanced acc |',
    '|---|---:|---:|---:|---:|',
    f"| XLS-R frozen backend | 3 | {frozen_macro['mean_eer']:.4f} ± {frozen_macro['std_eer']:.4f} | {frozen_macro['mean_auroc']:.4f} ± {frozen_macro['std_auroc']:.4f} | {frozen_macro['mean_balanced_accuracy']:.4f} ± {frozen_macro['std_balanced_accuracy']:.4f} |",
    f"| XLS-R PEFT adapter | 3 | {peft_macro['mean_eer']:.4f} ± {peft_macro['std_eer']:.4f} | {peft_macro['mean_auroc']:.4f} ± {peft_macro['std_auroc']:.4f} | {peft_macro['mean_balanced_accuracy']:.4f} ± {peft_macro['std_balanced_accuracy']:.4f} |",
    '',
    'PEFT minus frozen:',
    '',
    f"- EER: `{summary['peft_minus_frozen_macro']['eer']:.4f}`",
    f"- AUROC: `{summary['peft_minus_frozen_macro']['auroc']:.4f}`",
    f"- balanced accuracy: `{summary['peft_minus_frozen_macro']['balanced_accuracy']:.4f}`",
    '',
    '## PEFT by seed',
    '',
    '| Seed | Folds | Mean EER | Mean AUROC | Mean balanced acc | Best EER fold | Worst EER fold |',
    '|---:|---:|---:|---:|---:|---|---|',
]
for seed in seeds_all:
    s = by_seed[str(seed)]
    lines.append(f"| {seed} | {s['folds_completed']} | {s['mean_eer']:.4f} | {s['mean_auroc']:.4f} | {s['mean_balanced_accuracy']:.4f} | {s['best_by_eer']} | {s['worst_by_eer']} |")
lines += [
    '',
    '## PEFT by fold, 3 seeds',
    '',
    '| Fold | Records | EER mean±std | AUROC mean±std | Bal acc mean±std | Spoof rate@0.5 mean | Score gap mean |',
    '|---|---:|---:|---:|---:|---:|---:|',
]
for f in sorted(by_fold, key=lambda r: r['eer_mean']):
    lines.append(f"| {f['fold']} | {f['records']} | {f['eer_mean']:.4f}±{f['eer_std']:.4f} | {f['auroc_mean']:.4f}±{f['auroc_std']:.4f} | {f['balanced_accuracy_mean']:.4f}±{f['balanced_accuracy_std']:.4f} | {f['predicted_spoof_rate_at_0p5_mean']:.4f} | {f['score_gap_mean']:.4f} |")
lines += ['', '## Caveats', '', '- Custom CoSG source-holdout diagnostic, not official CodecFake+ benchmark.', '- No HPO.', '- MASKGCT is mixed-ranking/stable-threshold-collapse risk, not stable below-chance inversion.', '']
(doc_root / 'summary.md').write_text('\n'.join(lines))
PY
}

for seed in "${SEEDS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    if ! run_fold "$seed" "$fold" 1; then
      exit $?
    fi
  done
done
summarize
