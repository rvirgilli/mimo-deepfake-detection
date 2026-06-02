# Release checklist

Purpose: separate what is currently reproducible from what is only available in this workstation's local ignored artifacts.

Paper work is in research-planning mode. This checklist is for system/repo readiness and controlled research execution readiness.

## Dependency policy

Machine-readable pins live in `docs/current/external_dependencies.yaml`. Machine-local path overrides may live in ignored `docs/current/external_dependencies.local.yaml`; copy `docs/current/external_dependencies.local.example.yaml` to start one.

External code and heavy weights are **local dependencies**, not tracked source:

| Dependency | Local path | Required for | Current policy |
|---|---|---|---|
| SSL_Anti-spoofing | `SSL_Anti-spoofing/` | official ASVspoof 2021 LA scorer; Tak wav2vec2 recipe | ignored local clone; audit revision before release |
| XLS-R checkpoint | `SSL_Anti-spoofing/xlsr2_300m.pt` | wav2vec2 frontend / Tak recipe | ignored local weight |
| MiMo-Audio-Tokenizer | `MiMo-Audio-Tokenizer/` | MiMo frontend code | ignored local clone; audit revision before release |
| MiMo weights | `models/MiMo-Audio-Tokenizer/` or `MIMO_TOKENIZER_MODEL` | MiMo frontend weights | ignored local artifact |

Do not vendor either external repository blindly. Before public release, choose one of:

1. documented setup commands with expected commit hashes;
2. Git submodules pinned to known commits;
3. small compatibility wrappers plus upstream installation instructions.

Current decision: documented local setup first. See `docs/current/EXTERNAL_DEPENDENCY_SETUP.md`. Submodules only if exact upstream revisions become more important than setup simplicity.

## Machine-readable dependency audit

Run:

```bash
python -m mimodf audit dependencies --format markdown
python -m mimodf audit dependencies --format json
```

The audit records:

- whether each local dependency path exists;
- Git remote and HEAD when it is a nested clone;
- whether the local remote/HEAD match `external_dependencies.yaml`;
- dirty status, capped to the first 40 changed paths plus a total count;
- required files for the official scorer / frontend seams;
- file sizes for required files;
- optional SHA-256 hashes with `--hash-files` for large weights.

The generated audit package includes:

- `dependency_checks.md`
- `dependency_checks.json`
- `external_dependencies.yaml`

Regenerate with:

```bash
python -m mimodf audit package --out /tmp/mimodf-audit-package
```

## Research-framework checks

These exercise the plain-file experiment system without model loading:

```bash
python -m mimodf experiment validate docs/current/examples/experiment_spec_v1_minimal.yaml
python -m mimodf experiment init docs/current/examples/experiment_spec_v1_minimal.yaml --seed 42 --root /tmp/mimodf-runs
python -m mimodf report index /tmp/mimodf-runs --provenance docs/current/main_table_provenance.yaml --out /tmp/mimodf-runs/index.jsonl
python -m mimodf report aggregate --index /tmp/mimodf-runs/index.jsonl
python -m mimodf report compare --index /tmp/mimodf-runs/index.jsonl --experiments wav2vec2_frozen mimo_frozen --strict
```

Current scope: specs/manifests/run-layout/indexing/aggregation/comparison are implemented; eval and training CLIs can optionally write run-layout v1 artifacts. The strict compare command above is expected to fail for historical rows with missing protocol IDs; that failure is the guardrail working.

## Lightweight release checks

These should pass in a non-GPU/lightweight environment:

```bash
pytest -q
python -m compileall -q mimodf src train.py
python -m mimodf audit main-table >/tmp/main_table.md
python -m mimodf audit dependencies --format json >/tmp/dependency_checks.json
# If using a clean local scorer clone, configure docs/current/external_dependencies.local.yaml first.
# Slow, but useful before release:
python -m mimodf audit dependencies --format json --hash-files >/tmp/dependency_checks_hashed.json
python -m mimodf audit release-gate --format markdown
python -m mimodf audit release-gate --format markdown --hash-files
python -m mimodf audit package --out /tmp/mimodf-audit-package
python -m mimodf experiment validate docs/current/examples/experiment_spec_v1_minimal.yaml
python -m mimodf experiment init docs/current/examples/experiment_spec_v1_minimal.yaml --seed 42 --root /tmp/mimodf-runs
python -m mimodf report index /tmp/mimodf-runs --provenance docs/current/main_table_provenance.yaml --out /tmp/mimodf-runs/index.jsonl
python -m mimodf report aggregate --index /tmp/mimodf-runs/index.jsonl >/tmp/mimodf-runs/aggregate.md
python -m mimodf report compare --index /tmp/mimodf-runs/index.jsonl --experiments wav2vec2_frozen mimo_frozen >/tmp/mimodf-runs/compare.md
python -m mimodf eval plan \
  --config configs/publish/wav2vec2_adapter.yaml \
  --checkpoint experiments/paper_final/wav2vec2_adapter_multiseed/seed_123/models/w2v2_adapter_s123_seed123/epoch_12_eer_5.42.pth \
  --eval-root . \
  --score-out /tmp/mimodf-eval-plan/scores_LA_eval.txt \
  --track LA \
  --scorer local_dependencies/SSL_Anti-spoofing-clean/evaluate_2021_LA.py \
  --strict
python -m mimodf train legacy-asvspoof \
  --config configs/publish/mimo_full.yaml \
  --out /tmp/mimodf-train-dry-run \
  --database-path /data/asvspoof \
  --protocols-path /data/protocols \
  --validation-protocol asvspoof2021_fast \
  --frontend mimo \
  --dry-run
```

Optional full-env check:

```bash
conda run -n mimo-df pytest -q
```

## Optional dependency checks

Run only when local external dependencies and data are available:

```bash
python -m mimodf score official-la <score-file> \
  --eval-root <ASVspoof2021-LA-root> \
  --scorer SSL_Anti-spoofing/evaluate_2021_LA.py
```

MiMo integration tests remain opt-in:

```bash
RUN_MIMO_INTEGRATION=1 conda run -n mimo-df pytest tests/test_native_50hz.py -q
```

## Release gate

Use this for the one-command truth summary:

```bash
python -m mimodf audit release-gate --format markdown
```

Use strict mode in CI/release scripts:

```bash
python -m mimodf audit release-gate --strict
```

`--strict` exits nonzero when blockers are present. Current expected blockers are missing historical artifacts and dirty `SSL_Anti-spoofing` state unless local overrides are configured. To clear the dependency-dirty blocker without mutating historical artifacts, point `SSL_Anti-spoofing` to a clean clone via ignored `docs/current/external_dependencies.local.yaml`.

For system/tooling release checks only, explicitly allow known historical artifact gaps:

```bash
python -m mimodf audit release-gate --system-profile --strict
```

This uses `docs/current/artifact_gap_decisions.yaml` to downgrade known missing historical artifacts to warnings. It is not a full-reproducibility gate.

Suggested clean scorer setup:

```bash
mkdir -p local_dependencies
git clone https://github.com/TakHemlata/SSL_Anti-spoofing.git local_dependencies/SSL_Anti-spoofing-clean
git -C local_dependencies/SSL_Anti-spoofing-clean checkout 4acaa61dcef5f7610f43aa4d0b29c4559b970cd2
cp docs/current/external_dependencies.local.example.yaml docs/current/external_dependencies.local.yaml
python -m mimodf audit release-gate --format markdown
```

## Release blockers

A public claim of full reproducibility remains blocked until:

- missing/mismatched historical artifacts are recovered exactly or rerun as new controlled experiments;
- full LA/DF rerun commands are executed only if explicitly approved and then logged.

Historical scope policy: `docs/current/HISTORICAL_REPRO_SCOPE.md`. Historical score artifacts can guide research, but rows with missing checkpoints/configs/output dirs are not full reproducibility evidence.

Resolved documentation blockers:

- `SSL_Anti-spoofing/` setup is pinned/documented in `EXTERNAL_DEPENDENCY_SETUP.md`;
- `MiMo-Audio-Tokenizer/` setup is pinned/documented in `EXTERNAL_DEPENDENCY_SETUP.md`;
- required model weights have source/checksum policy in `EXTERNAL_DEPENDENCY_SETUP.md` and `external_dependencies.yaml`;
- ASVspoof data/protocol/key layout is documented in `ASVSPOOF_DATA_LAYOUT.md`;
- public no-data and bounded smoke command transcript is documented in `CONTROLLED_SMOKE_TRANSCRIPT.md`.

The current system is suitable for audited evidence packaging, controlled dry-run planning, and beginning versioned research execution. It is not yet a turnkey public reproduction package.
