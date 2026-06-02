# Research Wave 0 status

Date: 2026-05-26  
Status: Wave 0 gates passed; Wave 1 CoSG feature caches extracted

Parent plan: `RESEARCH_WAVE_0_1_PLAN.md`

## Dataset staging

Local root:

```text
<datasets>/codecfake_plus/
```

Downloaded and verified:

| File/path | Size | Status |
|---|---:|---|
| `CoSG_labels.txt` | 74,650 bytes | downloaded |
| `CoRS_labels.txt` | 62,363,876 bytes | downloaded |
| `CodecFake_plus_CoSG.tar.xz` | 243,976,528 bytes | downloaded |
| `CoSG/` | 1,797 WAV files / about 336 MB | extracted |

Checksums:

```text
9b32a0b2a9dc5aa13277e7ec0e4b145b2e3941954cb903356ce6b86b099e2eb9  CoSG_labels.txt
78d2cc2784188fecd8c082c9a180593ff7b60cf89672e6b1a5b37aae4c26bd08  CoRS_labels.txt
1748998d84037dd621ea39355b65b9cfa76b4105fe34457902edc240d47508ed  CodecFake_plus_CoSG.tar.xz
```

CoSG audio facts from local WAV inspection:

```text
files: 1,797
total: about 3.02 hours
sample rate: 16 kHz
channels: mono
sample width: 16-bit PCM
mean duration: about 6.05 s
min/max duration: 0.85 s / 28.58 s
```

## Full CoRS download

Started in background by user request. This is **not required for Wave 0/early Wave 1**, but it will be useful if CodecFake+ probes show signal.

Files being downloaded:

```text
Codecfake_plus_CoRS.part0
Codecfake_plus_CoRS.part1
Codecfake_plus_CoRS.part2
Codecfake_plus_CoRS.part3
```

Each part is about 25,052,014,980 bytes; total compressed size is about 100 GB.

Monitor:

```bash
cat <datasets>/codecfake_plus/download_cors.pid
kill -0 $(cat <datasets>/codecfake_plus/download_cors.pid) && echo running
tail -f <datasets>/codecfake_plus/download_cors.log
ls -lh <datasets>/codecfake_plus/Codecfake_plus_CoRS.part*
```

The downloader uses `curl --continue-at -`, so it can resume partial files if interrupted.

## Protocol indexing

Implemented:

```bash
python -m mimodf data codecfake-plus-index \
  --cosg-labels <datasets>/codecfake_plus/CoSG_labels.txt \
  --cors-labels <datasets>/codecfake_plus/CoRS_labels.txt \
  --cosg-audio-root <datasets>/codecfake_plus/CoSG \
  --out features/mimodf/wave0/codecfake_plus_protocol.jsonl \
  --summary-out features/mimodf/wave0/codecfake_plus_protocol_summary.md
```

Generated outputs are ignored local artifacts:

```text
features/mimodf/wave0/codecfake_plus_protocol.jsonl
features/mimodf/wave0/codecfake_plus_protocol_summary.md
```

Current local index summary:

```text
records: 1,424,357
subsets:
  CoRS: 1,422,560
  CoSG: 1,797
labels:
  bonafide: 45,321
  spoof: 1,379,036
missing CoSG audio paths: 0
duplicate utterance IDs: 0
index size: about 647 MB
```

CoSG source model counts:

```text
CLAMTTS: 119
ELLAV: 16
GPST: 40
MASKGCT: 1152
NS2: 46
NS3: 64
RALLE: 10
SIMPLESPEECH1: 66
SIMPLESPEECH2: 62
SINGLECODEC: 22
SPEECHX: 16
TACOLM: 16
TI1G1R: 12
UNIAUDIO: 22
USLM: 10
VALLE: 110
VIOLA: 14
```

CoSG taxonomy counts:

```text
quantizer:
  Mvq: 848
  Scq: 63
  Svq: 20
auxiliary objective:
  Disent: 52
  None: 874
  Sem: 5
decoder:
  Freq: 668
  Time: 263
```

Bonafide rows use `Real` taxonomy fields in the source labels and are normalized to null taxonomy fields in the index.

## Current caveats

- CoSG is small and imbalanced by source model; held-out source tests may have low support for some models.
- CoRS rows are labels-only until full audio parts finish downloading and are extracted/indexed.
- CoRS taxonomy requires an explicit codec-name-to-taxonomy mapping before QUA/AUX/DEC claims.
- CoRS-as-spoof is a labeling policy, not a universal deployment truth.
- The generated full protocol JSONL is large because it includes all CoRS label rows; future commands may need a filtered CoSG-only index for speed.

## MiMo feature extraction smoke

Implemented:

```bash
python -m mimodf features mimo-extract \
  --protocol /tmp/codecfake_cosg_index.jsonl \
  --out-dir features/mimodf/wave0/mimo_continuous_25hz_smoke \
  --model-path <legacy-repo>/MiMo-Audio-Tokenizer/model_weights \
  --representation continuous_25hz \
  --max-items 2 \
  --batch-size 1 \
  --device cuda \
  --overwrite

python -m mimodf features mimo-extract \
  --protocol /tmp/codecfake_cosg_index.jsonl \
  --out-dir features/mimodf/wave0/mimo_rvq_late_smoke \
  --model-path <legacy-repo>/MiMo-Audio-Tokenizer/model_weights \
  --representation rvq_codes \
  --quantizer-group late \
  --max-items 2 \
  --batch-size 1 \
  --device cuda \
  --overwrite
```

Both commands were run through the `mimo-df` conda environment and completed on CUDA.

Generated ignored local artifacts:

```text
features/mimodf/wave0/mimo_continuous_25hz_smoke/manifest.json
features/mimodf/wave0/mimo_continuous_25hz_smoke/records.jsonl
features/mimodf/wave0/mimo_continuous_25hz_smoke/arrays/*.npz
features/mimodf/wave0/mimo_rvq_late_smoke/manifest.json
features/mimodf/wave0/mimo_rvq_late_smoke/records.jsonl
features/mimodf/wave0/mimo_rvq_late_smoke/arrays/*.npz
```

Smoke output shapes:

```text
continuous_25hz:
  CLAMTTS_1: [135, 1280], float32
  CLAMTTS_2: [386, 1280], float32
  output size: about 1.3 MB

rvq_codes late quantizers [2..19]:
  CLAMTTS_1: [135, 18], int16
  CLAMTTS_2: [386, 18], int16
  output size: about 32 KB
```

Recorded manifest facts:

```text
sample_rate: 24000
batch_size: 1
device: cuda
use_bfloat16: true
num_quantizers: 20
codebook_size: [1024, 1024, 128 x18]
```

Caveat: MiMo features are batch-size-sensitive; Wave 1 must pin extraction batch size in manifests and comparisons.

## Frozen SSL feature extraction smoke

Implemented:

```bash
python -m mimodf features wav2vec2-extract \
  --protocol /tmp/codecfake_cosg_index.jsonl \
  --out-dir features/mimodf/wave0/wav2vec2_xlsr_smoke \
  --checkpoint SSL_Anti-spoofing/xlsr2_300m.pt \
  --max-items 2 \
  --batch-size 1 \
  --device cuda \
  --overwrite
```

The command was run through the `mimo-df` conda environment and completed on CUDA.

Generated ignored local artifacts:

```text
features/mimodf/wave0/wav2vec2_xlsr_smoke/manifest.json
features/mimodf/wave0/wav2vec2_xlsr_smoke/records.jsonl
features/mimodf/wave0/wav2vec2_xlsr_smoke/arrays/*.npz
```

Smoke output shapes:

```text
continuous_50hz:
  CLAMTTS_1: [269, 1024], float32
  CLAMTTS_2: [770, 1024], float32
  output size: about 3.9 MB
```

Recorded manifest facts:

```text
component_id: frontend:wav2vec2-xlsr-300m/v1
checkpoint: SSL_Anti-spoofing/xlsr2_300m.pt
sample_rate: 16000
batch_size: 1
device: cuda
dtype: float32
```

## Wave 1 CoSG feature-extraction estimates

Based on the two-file smoke subset (`CLAMTTS_1`, `CLAMTTS_2`, about 20.8 s total) and CoSG total duration of about 3.02 h:

| Feature set | Smoke size | Estimated CoSG size | Notes |
|---|---:|---:|---|
| MiMo continuous 25 Hz `[T,1280]` | 1.3 MB | about 0.7 GB | compressed float32 `.npz`; extraction batch size 1 |
| MiMo RVQ late `[T,18]` | 32 KB | about 17 MB | compressed int16 `.npz`; extraction batch size 1 |
| wav2vec2/XLSR continuous 50 Hz `[T,1024]` | 3.9 MB | about 2.0 GB | compressed float32 `.npz`; extraction batch size 1 |

Wall-clock model loading dominates these tiny smokes. Post-load feature extraction took under 2 seconds for the two-file subset for each feature set, so full CoSG extraction is expected to be practical on the local RTX 3090 path. Treat timing as approximate until a 50-100 item pilot is run.

Wave 1 batch-size policy:

```text
MiMo extraction batch_size: 1
wav2vec2 extraction batch_size: 1
```

Rationale: MiMo features are known batch-size-sensitive in this repo. CoSG is small enough that batch size 1 is the safest protocol choice and keeps MiMo/SSL smoke IDs aligned. Do not mix MiMo feature caches across batch sizes.

## Wave 0 gates

- [x] Stage CodecFake+ CoSG labels/audio and CoRS labels.
- [x] Build protocol indexer and validate local label counts.
- [x] Add/validate MiMo feature extraction manifest on tiny CoSG subset.
- [x] Add/validate one frozen SSL feature extraction smoke on the same tiny IDs.
- [x] Estimate Wave 1 feature storage/time from smoke outputs.
- [x] Decide Wave 1 extraction batch size and pin it in manifests.

Wave 1 can start only as **feature-only CoSG probes**. No broad training, Optuna, or SOTA claims.

## Wave 1 CoSG feature cache extraction

After the 100-item pilot completed successfully, full CoSG feature caches were extracted with pinned batch size 1:

```bash
python -m mimodf features mimo-extract \
  --protocol features/mimodf/wave0/codecfake_plus_protocol.jsonl \
  --out-dir features/mimodf/wave1/codecfake_cosg_mimo_continuous_25hz_b1 \
  --model-path <legacy-repo>/MiMo-Audio-Tokenizer/model_weights \
  --representation continuous_25hz \
  --max-items 1797 \
  --batch-size 1 \
  --device cuda \
  --overwrite

python -m mimodf features mimo-extract \
  --protocol features/mimodf/wave0/codecfake_plus_protocol.jsonl \
  --out-dir features/mimodf/wave1/codecfake_cosg_mimo_rvq_late_b1 \
  --model-path <legacy-repo>/MiMo-Audio-Tokenizer/model_weights \
  --representation rvq_codes \
  --quantizer-group late \
  --max-items 1797 \
  --batch-size 1 \
  --device cuda \
  --overwrite

python -m mimodf features wav2vec2-extract \
  --protocol features/mimodf/wave0/codecfake_plus_protocol.jsonl \
  --out-dir features/mimodf/wave1/codecfake_cosg_wav2vec2_xlsr_b1 \
  --checkpoint SSL_Anti-spoofing/xlsr2_300m.pt \
  --max-items 1797 \
  --batch-size 1 \
  --device cuda \
  --overwrite
```

Generated ignored local artifacts:

```text
features/mimodf/wave1/codecfake_cosg_mimo_continuous_25hz_b1/
features/mimodf/wave1/codecfake_cosg_mimo_rvq_early_b1/
features/mimodf/wave1/codecfake_cosg_mimo_rvq_late_b1/
features/mimodf/wave1/codecfake_cosg_mimo_rvq_all_b1/
features/mimodf/wave1/codecfake_cosg_wav2vec2_xlsr_b1/
```

Extraction summary:

| Feature cache | Records | Frames | Manifest extraction time | Disk size |
|---|---:|---:|---:|---:|
| MiMo continuous 25 Hz | 1,797 | 272,785 | 200.93 s | 629 MB |
| MiMo RVQ early | 1,797 | 272,785 | 57.04 s | 2.53 MB |
| MiMo RVQ late | 1,797 | 272,785 | 57.45 s | 11 MB |
| MiMo RVQ all | 1,797 | 272,785 | 56.18 s | 8.35 MB |
| wav2vec2/XLSR continuous 50 Hz | 1,797 | 541,719 | 107.21 s | 2.0 GB |

Validation:

```text
first utterance: CLAMTTS_1
last utterance: UNIAUDIO_22
record order aligned across all three caches: true
```

Next step completed: `mimodf features probe` now runs frozen-feature linear probes. First results are summarized in `docs/current/RESEARCH_WAVE_1_RESULTS.md`. Next research slice is score fusion/error-overlap plus more held-out-source controls; tiny MLP remains deferred.
