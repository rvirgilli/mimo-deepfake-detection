# MiMo Deepfake Detection

Research code for audio deepfake detection experiments with self-supervised speech representations and MiMo-style audio tokenizers.

This repository contains maintained code, tests, portable configuration templates, and curated documentation. Large generated artifacts such as checkpoints, cached features, run outputs, and machine-specific dataset/dependency paths are not tracked.

## Repository contents

- `mimodf/` — maintained audit, data, scoring, training, reporting, and experiment utilities.
- `tests/` — focused tests for maintained behavior.
- `configs/` — portable configuration templates.
- `src/` — legacy compatibility layer used by some controlled paths.
- `docs/current/` — curated notes and machine-readable summaries.
- `scripts/` — reusable scripts.

## Local setup

Copy `.env.example` or export equivalent variables:

```bash
export ASVSPOOF_LA_DATABASE=/path/to/ASVspoof_database/LA
export ASVSPOOF_PROTOCOLS=./SSL_Anti-spoofing/database/
export ASVSPOOF2021_LA_EVAL=/path/to/ASVspoof2021_LA_eval
export ASVSPOOF2021_LA_KEYS=/path/to/ASVspoof2021_LA_keys
export WAV2VEC2_XLSR_CHECKPOINT=./SSL_Anti-spoofing/xlsr2_300m.pt
export MIMO_TOKENIZER_MODEL=./models/MiMo-Audio-Tokenizer
```

Local external clones are intentionally ignored:

- `SSL_Anti-spoofing/`
- `MiMo-Audio-Tokenizer/`

Expected remotes/revisions and setup notes are documented in `docs/current/EXTERNAL_DEPENDENCY_SETUP.md`.

## Verification

Install development tooling:

```bash
python -m pip install -e ".[dev]"
```

Fast checks:

```bash
python -m ruff check mimodf tests
python -m ruff format --check mimodf tests
pytest -q
python -m compileall -q mimodf src train.py
```

Full local environment check, when the `mimo-df` environment exists:

```bash
conda run -n mimo-df pytest -q
```

MiMo model integration tests are skipped by default. To run them explicitly:

```bash
RUN_MIMO_INTEGRATION=1 conda run -n mimo-df python -m pytest tests/test_native_50hz.py -q
```
