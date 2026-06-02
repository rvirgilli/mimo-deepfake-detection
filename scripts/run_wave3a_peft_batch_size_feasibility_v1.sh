#!/usr/bin/env bash
set -u
BATCHES=(2 4 8)
FOLD="CLAMTTS"
SEED=123
DOC_ROOT="docs/current/wave3a_peft_batch_size_feasibility_v1"
OUT_ROOT="experiments/runs/wave3a_peft_batch_size_feasibility_v1"
mkdir -p "$DOC_ROOT"
for batch in "${BATCHES[@]}"; do
  echo "=== batch=$batch ===" | tee "$DOC_ROOT/batch_${batch}.status"
  PYTHONPATH=. conda run -n mimo-df python -m mimodf train codecfake-xlsr \
    --split-plan docs/current/wave3a_codecfake_cosg_source_holdout_plan_v3.json \
    --protocol features/mimodf/wave0/codecfake_plus_protocol.jsonl \
    --fold "$FOLD" \
    --condition xlsr_peft_adapter \
    --seed "$SEED" \
    --out "$OUT_ROOT" \
    --batch-size "$batch" \
    --eval-batch-size "$batch" \
    --cut 64600 \
    --device cuda \
    --deterministic \
    --xlsr-checkpoint SSL_Anti-spoofing/xlsr2_300m.pt \
    --model-smoke > "$DOC_ROOT/batch_${batch}.stdout" 2> "$DOC_ROOT/batch_${batch}.stderr"
  code=$?
  echo "$code" > "$DOC_ROOT/batch_${batch}.exit_code"
  if [ "$code" -ne 0 ]; then
    echo "batch=$batch failed with exit $code" >&2
  else
    echo "batch=$batch passed" >&2
  fi
  # Let CUDA memory settle.
  sleep 5
done
python - <<'PY'
import json
from pathlib import Path
root = Path('docs/current/wave3a_peft_batch_size_feasibility_v1')
rows = []
for batch in [2, 4, 8]:
    stdout = root / f'batch_{batch}.stdout'
    stderr = root / f'batch_{batch}.stderr'
    exit_code = int((root / f'batch_{batch}.exit_code').read_text().strip())
    out_text = stdout.read_text(errors='replace') if stdout.exists() else ''
    err_text = stderr.read_text(errors='replace') if stderr.exists() else ''
    parsed = None
    for start in [out_text.find('{'), out_text.rfind('{')]:
        if start >= 0:
            try:
                parsed = json.loads(out_text[start:])
                break
            except Exception:
                pass
    rows.append({
        'batch_size': batch,
        'eval_batch_size': batch,
        'exit_code': exit_code,
        'passed': exit_code == 0,
        'stdout_path': str(stdout),
        'stderr_path': str(stderr),
        'stdout_tail': out_text[-1200:],
        'stderr_tail': err_text[-1200:],
        'parsed_output': parsed,
        'oom': 'out of memory' in (out_text + err_text).lower(),
    })
summary = {
    'schema': 'mimodf-wave3a-peft-batch-size-feasibility/v1',
    'claim_scope': 'technical CUDA/model-smoke feasibility only; no training metric evidence',
    'condition': 'xlsr_peft_adapter',
    'fold': 'CLAMTTS',
    'seed': 123,
    'batches_tested': [2, 4, 8],
    'rows': rows,
    'max_passing_batch_size': max((r['batch_size'] for r in rows if r['passed']), default=None),
}
(root / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
lines = [
    '# Wave 3A PEFT batch-size feasibility',
    '',
    'Scope: model-smoke only; one optimizer step and one validation forward pass. No checkpoint/scores/metrics claims.',
    '',
    '| Batch | Exit | Passed | OOM |',
    '|---:|---:|---|---|',
]
for row in rows:
    lines.append(f"| {row['batch_size']} | {row['exit_code']} | {row['passed']} | {row['oom']} |")
lines += ['', f"Max passing batch size: `{summary['max_passing_batch_size']}`", '']
(root / 'summary.md').write_text('\n'.join(lines))
PY
