# System status

Date: 2026-05-26
Branch: `paper-rework-audit`
Paper work: research-planning resumed; manuscript claims still lag evidence.

## Verdict

The repo now has a usable audited system layer and the beginning of a lean scientific research framework.

The system is designed around plain-file contracts rather than heavy platforms: versioned experiment specs, resolved specs, run manifests, immutable run layouts, score contracts, component IDs, and JSONL/Markdown indexes.

Implemented research-framework slices cover spec validation/hash/resolve, run manifests, run-layout helpers, component metadata, `mimodf experiment validate/resolve/init/inspect`, optional run-layout v1 manifest updates for `mimodf eval run` and `mimodf train legacy-asvspoof`, `mimodf report index` over new run layouts plus historical main-table provenance, and `mimodf report aggregate/compare` over indexed records.

It is **not** yet a turnkey public reproduction package because historical artifact gaps remain and full LA/DF reruns still require explicit approval. Historical scope policy is in `docs/current/HISTORICAL_REPRO_SCOPE.md`: score-backed gaps can guide research but cannot support full reproducibility claims.

## Working system layer

- `mimodf audit main-table` renders the corrected main table from `docs/current/main_table_provenance.yaml`.
- `mimodf audit check-artifacts` audits declared evidence artifacts.
- `mimodf audit dependencies` audits local external clone/artifact presence, expected remotes/revisions, dirty state, required files, sizes, and optional hashes. It supports ignored local overrides at `docs/current/external_dependencies.local.yaml`.
- `mimodf audit release-gate` summarizes release readiness and exits nonzero in `--strict` mode when blockers are present. `--system-profile` allows explicitly documented historical artifact gaps as warnings, not full-reproducibility claims.
- `mimodf audit package` writes regenerated review/release evidence outputs.
- `mimodf score official-la` wraps official ASVspoof LA scoring and rejects project wrong-scale `min t-DCF` files.
- `mimodf score compare-files` compares score files for reproduction audits.
- `mimodf eval plan ...` validates evaluation config/checkpoint/eval-root/score-output/scorer paths without loading models, touching GPUs, or writing scores.
- `mimodf eval run ...` runs controlled legacy ASVspoof evaluation from explicit paths; current real smoke coverage is wav2vec2 adapter LA with a two-utterance CUDA run.
- `mimodf train legacy-asvspoof --dry-run ...` validates controlled-training launch inputs and required ASVspoof paths without starting a long run.
- `mimodf train legacy-asvspoof ... --max-train-batches N --max-val-batches M` supports bounded real training smoke tests and can optionally write run-layout v1 artifacts with `--experiment-spec`/`--run-seed`.
- `mimodf experiment validate/resolve/init/inspect` validates specs and creates/inspects versioned run-layout directories.
- `mimodf report index` indexes new run-layout manifests and historical main-table provenance into `run-index-record/v1` JSONL/Markdown.
- `mimodf report aggregate` computes numeric metric aggregates from run-index JSONL.
- `mimodf report compare` checks seed/protocol/intent/batch-size comparability and fails in strict mode when unsafe.
- `mimodf features mimo-extract` and `mimodf features wav2vec2-extract` write cached feature arrays plus manifests for feature-only research probes.
- `mimodf features probe` runs cached-feature linear probes and writes metrics/report/prediction artifacts.
- `mimodf features fuse-probes` averages two probe prediction files and writes fusion/error-overlap metrics.
- `docs/current/research_execution_log.jsonl` records research commands and generated artifact paths; new feature/probe artifacts also include `command_argv` going forward.

## Current generated-audit counts

From `python -m mimodf audit package --out /tmp/mimodf-status-package`:

| Artifact status | Count |
|---|---:|
| present | 132 |
| missing | 9 |
| declared_absent | 9 |

Audit package files now include:

- `main_table.md`
- `artifact_checks.md/json`
- `artifact_gaps.md/json`
- `dependency_checks.md/json`
- `external_dependencies.yaml`
- `tdcf_summary.md/json`
- `official_tdcf_values.yaml`
- `main_table_provenance.yaml`
- `summary.json`

## Current dependency state

Expected pins live in `docs/current/external_dependencies.yaml`. This workspace also has ignored `docs/current/external_dependencies.local.yaml` pointing `SSL_Anti-spoofing` at clean clone `local_dependencies/SSL_Anti-spoofing-clean`.

| Dependency | Audited path | Spec source | Present | Expected revision match | Dirty | Required files |
|---|---|---|---:|---:|---:|---:|
| `SSL_Anti-spoofing` | `local_dependencies/SSL_Anti-spoofing-clean` | local override | yes | yes (`4acaa61dcef5`) | no | present |
| `MiMo-Audio-Tokenizer` | `MiMo-Audio-Tokenizer/` | base | yes | yes (`b62b59922979`) | no | present |
| `MiMo model weights` | `models/MiMo-Audio-Tokenizer/` | base | yes | n/a | n/a | present |

The historical clone `SSL_Anti-spoofing/` remains dirty and untouched; the clean scorer clone is only for dependency/release-gate checks. Public setup policy is documented in `docs/current/EXTERNAL_DEPENDENCY_SETUP.md`: local clones/weights plus audited hashes, not submodules for now.

Large required local files:

| File | Size | SHA-256 status |
|---|---:|---|
| `SSL_Anti-spoofing/xlsr2_300m.pt` | 3,808,868,242 bytes | expected hash recorded in base spec and verified with `--hash-files` when local override is disabled |
| `models/MiMo-Audio-Tokenizer/model.safetensors` | 3,906,690,080 bytes | expected hash recorded and verified with `--hash-files` |

All required dependency file hashes currently match the spec when running:

```bash
python -m mimodf audit dependencies --format json --hash-files
```

## Corrected main-table status

| Row | Status |
|---|---|
| wav2vec2 frozen | partial; metrics present, configs/manifests missing |
| wav2vec2 adapter | partial; tDCF corrected to all-five `0.255` |
| wav2vec2 full FT | partial; local reproduced n=3 only |
| MiMo frozen | invalid as earlier-manuscript row; corrected all-completed-seed EER with n=4 tDCF footnote |
| MiMo adapter | exploratory n=2 only |
| MiMo full FT | partial; tDCF corrected to all-five `0.350` |

## Release gate

Current command:

```bash
python -m mimodf audit release-gate --format json
```

Current verdict: **fail**.

Current blocker codes with local clean-scorer override active:

- `artifact_missing` — 9 declared artifact paths are missing.

Those 9 gaps are documented in `docs/current/artifact_gap_decisions.yaml`. For system/tooling release checks only, run `python -m mimodf audit release-gate --system-profile --strict`; this downgrades the known gaps to warnings and passes in this workspace. It does **not** mean full experiment reproducibility.

The previous `dependency_dirty` blocker is cleared in this workspace by ignored `docs/current/external_dependencies.local.yaml`. Disable the override with `--dependency-local-spec none` to audit the historical local clone directly.

Strict mode is intentionally nonzero until those blockers are resolved:

```bash
python -m mimodf audit release-gate --strict
```

## Management scope update

Current work is no longer only paper repair. It is now a controlled research-framework build-out so future experiments can be expanded without losing auditability.

Still in scope by default:

- controlled experiment specs and smoke tests;
- paper/research planning tied to generated evidence.

Still out of scope without explicit approval:

- full training/eval runs;
- targeted Wave 3 training jobs before specs/log rows are written;
- broad HPO/Optuna studies;
- new baseline matrix execution;
- heavy experiment platforms.

Wave 3 update: targeted trained validation is now a planned research direction, but the system still needs a CodecFake training/scoring path before claim-bearing runs.

## Remaining blockers

- Historical missing/mismatched checkpoints/configs remain; do not silently repair them. If a historical metric becomes a requirement, recover the exact artifact or rerun as a new controlled experiment.
- MiMo adapter n=5 artifacts remain unavailable.
- No full controlled long training/evaluation has been run through the new CLI.
- No CodecFake+ CoSG/CoRS trained-validation path has been run; Wave 3 requires this before paper-grade trained claims.
- One real eval smoke was run through `mimodf eval run` on 2 LA utterances with wav2vec2 adapter seed 123; generated scores matched existing historical scores within ~3.5e-4 absolute difference for those utterances.
- One 1000-utterance bounded wav2vec2 adapter eval reproduction completed in about 21 seconds with batch size 16; compared with the historical score file, mean absolute score difference was about 1.06e-4 and max absolute difference about 1.90e-2.
- One real training smoke was run through `mimodf train legacy-asvspoof` on wav2vec2 adapter LA with 1 train batch and 1 validation batch; checkpoint + manifest were written under `/tmp/mimodf-real-train-smoke`, and the smoke checkpoint loaded successfully through `mimodf eval run`.
- MiMo frozen real eval smoke now runs through `mimodf eval run` with historical seed-42 checkpoint; 100-utterance comparison to historical score file has mean absolute drift about 2.67e-2 and max drift about 1.13e-1 at batch size 4, and about 1.45e-2 / 6.32e-2 at batch size 64.
- MiMo frozen real training smoke completed with 1 train batch and 1 validation batch; checkpoint + manifest were written under `/tmp/mimodf-real-train-mimo-frozen-smoke`, and the checkpoint loaded through `mimodf eval run`.
- `docs/current/MIMO_DRIFT_INVESTIGATION.md` records the current drift finding: MiMo scores are deterministic for fixed batch size but batch-size-sensitive; drift begins in MiMo encoder features while mel inputs are identical.

## Latest verification

Completed after research-framework indexing:

```text
pytest -q: 151 passed, 9 skipped
conda run -n mimo-df pytest -q: 307 passed, 2 skipped
python -m compileall -q mimodf src train.py: ok
python -m mimodf experiment validate docs/current/examples/experiment_spec_v1_minimal.yaml: ok
python -m mimodf experiment init docs/current/examples/experiment_spec_v1_minimal.yaml --seed 42 --root /tmp/mimodf-train-hook/runs: ok
python -m mimodf report index /tmp/mimodf-report-aggregate/runs --provenance docs/current/main_table_provenance.yaml --out /tmp/mimodf-report-aggregate/index.jsonl: ok; 26 records
python -m mimodf report aggregate --index /tmp/mimodf-report-aggregate/index.jsonl: ok
python -m mimodf report compare --index /tmp/mimodf-report-aggregate/index.jsonl --experiments wav2vec2_frozen mimo_frozen: ok, warning-mode report generated
python -m mimodf report compare --index /tmp/mimodf-report-aggregate/index.jsonl --experiments wav2vec2_frozen mimo_frozen --strict: expected exit 1 because historical rows lack protocol IDs
python -m mimodf audit release-gate --system-profile --strict: ok
git diff --check: ok
```

Earlier real-system smoke evidence remains valid:

```text
wav2vec2 adapter eval smoke: matched historical scores within ~3.5e-4 for 2 utterances
wav2vec2 adapter first-1000 reproduction: mean_abs_diff≈1.06e-4, max_abs_diff≈1.90e-2
existing wav2vec2 adapter seed-123 LA official score: EER 2.33, min_tDCF 0.2458
wav2vec2 adapter training smoke: checkpoint loadable through eval
MiMo frozen eval/training smokes: runnable, but historical score drift larger and batch-size-sensitive
```

## Lightweight verification commands

```bash
pytest -q
python -m compileall -q mimodf src train.py
python -m mimodf audit dependencies --format json
python -m mimodf audit dependencies --format json --hash-files
python -m mimodf audit release-gate --format markdown
python -m mimodf audit release-gate --format markdown --hash-files
python -m mimodf audit release-gate --system-profile --strict
python -m mimodf audit package --out /tmp/mimodf-audit-package
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
python -m mimodf experiment validate docs/current/examples/experiment_spec_v1_minimal.yaml
python -m mimodf experiment init docs/current/examples/experiment_spec_v1_minimal.yaml --seed 42 --root /tmp/mimodf-runs
python -m mimodf report index /tmp/mimodf-runs --provenance docs/current/main_table_provenance.yaml --out /tmp/mimodf-runs/index.jsonl
python -m mimodf report aggregate --index /tmp/mimodf-runs/index.jsonl
python -m mimodf report compare --index /tmp/mimodf-runs/index.jsonl --experiments wav2vec2_frozen mimo_frozen
```

Optional full environment:

```bash
conda run -n mimo-df pytest -q
```
