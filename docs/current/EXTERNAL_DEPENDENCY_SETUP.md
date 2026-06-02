# External dependency setup

Date: 2026-05-27
Status: accepted documented-local-clone policy

This repo does not vendor external research code or multi-GB model weights.

Public/reproducible setup is based on:

- documented local clones;
- pinned upstream revisions;
- local heavyweight artifacts;
- SHA-256 verification through `mimodf audit dependencies`.

Machine-readable pins and hashes live in:

```text
docs/current/external_dependencies.yaml
```

Machine-local overrides may live in ignored:

```text
docs/current/external_dependencies.local.yaml
```

## Policy decision

Use documented local clones, not submodules, for now.

Reason:

- the repo is still a research audit/rework workspace;
- external projects are large and have their own dependency constraints;
- only a small scorer subset is needed for lightweight release checks;
- submodules would not solve model-weight or dataset licensing/download issues.

Submodules can be reconsidered only if exact upstream checkout reproducibility becomes more important than setup simplicity.

## Required local dependencies

| Dependency | Default path | Purpose | Required for lightweight tests |
|---|---|---|---:|
| SSL_Anti-spoofing | `SSL_Anti-spoofing/` | official ASVspoof LA scorer; Tak wav2vec2/XLSR recipe | no |
| XLS-R 300M checkpoint | `SSL_Anti-spoofing/xlsr2_300m.pt` | wav2vec2/XLSR frontend extraction/reruns | no |
| MiMo-Audio-Tokenizer code | `MiMo-Audio-Tokenizer/` | MiMo frontend code | no |
| MiMo tokenizer weights | `models/MiMo-Audio-Tokenizer/` | MiMo feature extraction/reruns | no |

The default `pytest -q` path skips optional heavy dependencies when absent.

## Setup: clean scorer clone for release checks

Use this when you only need dependency/release-gate checks and official LA scoring wrappers.

```bash
mkdir -p local_dependencies
git clone https://github.com/TakHemlata/SSL_Anti-spoofing.git \
  local_dependencies/SSL_Anti-spoofing-clean
git -C local_dependencies/SSL_Anti-spoofing-clean checkout \
  4acaa61dcef5f7610f43aa4d0b29c4559b970cd2
cp docs/current/external_dependencies.local.example.yaml \
  docs/current/external_dependencies.local.yaml
python -m mimodf audit dependencies --format markdown
python -m mimodf audit release-gate --system-profile --strict
```

This avoids mutating a historical dirty `SSL_Anti-spoofing/` clone.

## Setup: full SSL_Anti-spoofing / XLS-R dependency

Use this for wav2vec2/XLSR reruns or feature extraction.

```bash
git clone https://github.com/TakHemlata/SSL_Anti-spoofing.git SSL_Anti-spoofing
git -C SSL_Anti-spoofing checkout 4acaa61dcef5f7610f43aa4d0b29c4559b970cd2
curl -L \
  https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt \
  -o SSL_Anti-spoofing/xlsr2_300m.pt
python -m mimodf audit dependencies --format markdown --hash-files \
  --local-spec none
```

Expected XLS-R SHA-256 is pinned in `external_dependencies.yaml`.

Source notes:

- TakHemlata/SSL_Anti-spoofing README points users to fairseq XLSR models.
- fairseq lists `XLS-R 300M` at `https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt`.

## Setup: MiMo-Audio-Tokenizer code

Use this for MiMo frontend construction or feature extraction.

```bash
git clone https://github.com/XiaomiMiMo/MiMo-Audio-Tokenizer.git MiMo-Audio-Tokenizer
git -C MiMo-Audio-Tokenizer checkout b62b59922979bf9f389b373169298a251587653f
python -m pip install -e ./MiMo-Audio-Tokenizer
# Optional, GPU/FlashAttention environment only:
# python -m pip install -e './MiMo-Audio-Tokenizer[flash]'
python -m mimodf audit dependencies --format markdown
```

## Setup: MiMo tokenizer weights

Use Git LFS or Hugging Face tooling. The upstream MiMo README documents a Hugging Face model clone.

Preferred local layout for this repo:

```bash
git lfs install
mkdir -p models
git clone https://huggingface.co/XiaomiMiMo/MiMo-Audio-Tokenizer \
  models/MiMo-Audio-Tokenizer
python -m mimodf audit dependencies --format markdown --hash-files
```

Alternative with `huggingface_hub` if installed:

```bash
hf download XiaomiMiMo/MiMo-Audio-Tokenizer \
  --local-dir models/MiMo-Audio-Tokenizer
python -m mimodf audit dependencies --format markdown --hash-files
```

Expected files:

```text
models/MiMo-Audio-Tokenizer/config.json
models/MiMo-Audio-Tokenizer/model.safetensors
```

Expected SHA-256 values are pinned in `external_dependencies.yaml`.

## Environment variables

Recommended local exports:

```bash
export ASVSPOOF_LA_DATABASE=/path/to/ASVspoof_database/LA
export ASVSPOOF_PROTOCOLS=./SSL_Anti-spoofing/database
export WAV2VEC2_XLSR_CHECKPOINT=./SSL_Anti-spoofing/xlsr2_300m.pt
export MIMO_TOKENIZER_MODEL=./models/MiMo-Audio-Tokenizer
```

For scorer-only release checks with a clean local clone:

```bash
export ASVSPOOF_PROTOCOLS=./local_dependencies/SSL_Anti-spoofing-clean/database
```

## Verify local state

Fast dependency audit:

```bash
python -m mimodf audit dependencies --format markdown
```

Slow dependency audit with multi-GB hashes:

```bash
python -m mimodf audit dependencies --format markdown --hash-files
```

System/tooling gate, allowing documented historical artifact gaps:

```bash
python -m mimodf audit release-gate --system-profile --strict
```

Full reproducibility gate:

```bash
python -m mimodf audit release-gate --strict
```

The full gate is still expected to fail until historical artifact gaps are recovered or explicitly retired from full-reproduction claims.

## What this does not solve

This setup policy does not recover missing historical checkpoints/configs.

It also does not grant access to ASVspoof datasets. Users must obtain datasets from their official sources and provide local paths consistent with the evaluation/training commands.
