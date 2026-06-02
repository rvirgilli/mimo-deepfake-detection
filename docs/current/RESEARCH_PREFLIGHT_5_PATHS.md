# Preflight: five-path MiMo/tokenizer research plan

Date: 2026-05-26  
Status: Exa-backed practical preflight; no experiments run  
Parent landscape: `RESEARCH_LANDSCAPE_MIMO_TOKENIZER_ADD.md`

## Purpose

Avoid repeating the historical failure mode: one attractive hypothesis, one path, weak fallback after disappointing results.

New strategy:

> Maintain five plausible research paths, run cheap waves that can confirm or kill paths, then converge only after evidence appears.

This document checks whether the five paths are practically testable now: data access, labels, protocols, MiMo extraction feasibility, and baseline availability.

## Verdict summary

| Path | Research question | Preflight verdict | First-wave readiness |
|---|---|---|---|
| A. Codec/tokenizer forensic cues | Do MiMo RVQ/codebook/residual features detect codec-based fake speech? | **Ready with storage caveat** | Ready on CodecFake+ small subsets |
| B. MiMo complements SSL | Does MiMo add independent signal to WavLM/wav2vec2/Whisper? | **Ready** | Ready after feature extraction adapter |
| C. Source/taxonomy tracing | Is MiMo better at identifying codec/generator family than binary spoofing? | **Ready with label join caveat** | Ready on CodecFake+ CoSG; CoRS needs mapping |
| D. Semantic vs acoustic explanation | Did semantic/audio-tokenizer abstraction suppress forensic cues? | **Ready with MiMo API caveats** | Ready if RVQ extraction is wrapped and batch size pinned |
| E. Robustness/media transformations | Does MiMo help under compression/resampling/noise/laundering? | **Feasible, external public data partly blocked** | Ready only via ASVspoof5 or self-applied transforms; RADAR/SAFE official data blocked |

Recommended first target:

> Run Wave 1 feature-only probes on **A + B + C + D together** using CodecFake+ small stratified subsets. Defer E to a second probe using ASVspoof5 codec/compression labels or self-applied transforms.

## Dataset/access preflight

### CodecFake+

Verdict: **ready with storage caveat**.

Sources:

- Dataset: <https://huggingface.co/datasets/CodecFake/CodecFake_Plus_Dataset>
- Paper: <https://arxiv.org/html/2501.08238>
- Source tracing repo/tasks: <https://github.com/responsiblegenai/codecfake-source-tracing>

Practical facts:

- Public Hugging Face dataset; not gated in current check.
- License shown on dataset card: MIT.
- Visible repo storage: about 101 GB / 94 GiB.
- CoRS archive split into four large parts, plus `CoRS_labels.txt`.
- CoSG archive is much smaller, about 244 MB, plus `CoSG_labels.txt`.
- CoRS labels format observed by scout: `speaker_id filename label`.
- CoSG labels format observed by scout: `Model ClipID QUA AUX DEC Label`.
- CoSG directly supports taxonomy/source tasks: model, quantizer type, auxiliary objective, decoder domain, label.
- CoRS taxonomy requires filename/codecs joined to paper taxonomy table before QUA/AUX/DEC tasks.

Risks:

- Full CoRS download is large.
- CoSG label counts may differ slightly from the paper/table; verify counts locally before claims.
- CoRS is heavily imbalanced; first probes must stratify.
- CoRS-as-spoof is a policy choice, not a universal truth. It can make models reject legitimate codec-compressed speech.

Use in waves:

- Best first dataset for paths A, B, C, D.
- Start with CoSG small archive if bandwidth/storage is constrained.
- Use CoRS only after a local protocol table is built and cached.

### Original CodecFake

Verdict: **usable fallback, weaker for taxonomy**.

Sources:

- Dataset: <https://huggingface.co/datasets/rogertseng/CodecFake>
- Paper: <https://arxiv.org/html/2406.07237v1>
- Code: <https://github.com/roger-tseng/CodecFake>

Practical facts:

- Public Hugging Face dataset; license CC-BY-4.0.
- `load_dataset("rogertseng/CodecFake")` documented.
- Features include `audio`, `label`, `speaker_id`, `codec_name`.
- Suitable for binary and codec-name source tasks.

Risks:

- Large full download.
- No direct QUA/AUX/DEC taxonomy fields.

Use in waves:

- Backup if CodecFake+ access is slow.
- Good for sanity-checking codec-name/source prediction.

### ASVspoof5

Verdict: **ready, large download**.

Sources:

- Zenodo data: <https://zenodo.org/records/14498691>
- Evaluation package: <https://github.com/asvspoof-challenge/asvspoof5/tree/main/evaluation-package>
- ASVspoof5 design paper: <https://arxiv.org/html/2502.08857>
- ASVspoof5 evaluation paper: <https://ar5iv.labs.arxiv.org/html/2601.03944>

Practical facts:

- Public Zenodo record, about 142.3 GB total.
- Includes train/dev/eval/protocol archives.
- Protocol fields reported by scout include `CODEC`, `CODEC_Q`, `CODEC_SEED`, `ATTACK_TAG`, `ATTACK_LABEL`, `KEY`.
- Official scoring package public.
- Includes codec/compression and adversarial conditions.

Risks:

- Large download.
- We need to verify protocol field availability after local extraction.
- ASVspoof-style labels still center bonafide/spoof, not codec-source taxonomy.

Use in waves:

- Best public path for E if full or partial data is available.
- Good for validating whether CodecFake findings transfer to a challenge-style setting.

### RADAR 2026

Verdict: **blocked for official first-wave probes**.

Sources:

- Challenge site: <https://radar-challenge.github.io/>
- Codabench: <https://www.codabench.org/competitions/15279/>
- Paper: <https://arxiv.org/html/2605.09568>
- Baseline: <https://github.com/radar-challenge/BASELINE-SSL_AASIST>

Practical facts:

- Registration closed.
- Development/evaluation data distributed by email to participants.
- Evaluation labels withheld.
- Paper says public release after conference.

Use in waves:

- Not usable as official first-wave data unless we already have access.
- Use as design inspiration for self-applied transformations.

### SAFE Audio

Verdict: **blocked for official-data probes**.

Sources:

- Repository: <https://github.com/stresearch/SAFE>
- Challenge paper: <https://arxiv.org/html/2510.03387>

Practical facts:

- Fully blind benchmark.
- Repository states no data release except small practice data.
- Metric is balanced accuracy.

Use in waves:

- Not usable for official probes.
- Use as transformation/laundering inspiration only.

## MiMo extraction preflight

Verdict: **feasible with explicit wrappers and protocol caveats**.

Sources:

- Public MiMo tokenizer repo: <https://github.com/XiaomiMiMo/MiMo-Audio-Tokenizer>
- Local config: `<legacy-repo>/MiMo-Audio-Tokenizer/model_weights/config.json`
- Local drift audit: `MIMO_DRIFT_INVESTIGATION.md`

Verified local config facts:

```text
sampling_rate: 24000
hop_length: 240
stride_size: 2
avg_pooler: 2
num_quantizers: 20
codebook_size: [1024, 1024, 128 x18]
```

Interpretation:

- Mel frame rate: 100 Hz.
- Encoder after stride: 50 Hz.
- Released/downsampled tokenizer representation: 25 Hz.
- 20 codebooks means 500 indices/sec at 25 Hz.
- Paper/model-card language often reports 8 RVQ layers / 200 tokens/sec for downstream MiMo-Audio usage; local released config exposes 20 quantizers.

Accessible slices:

| Slice | Feasibility | Notes |
|---|---|---|
| 25 Hz continuous encoder states | feasible | via encoder feature path/wrapper |
| RVQ indices per codebook | feasible | top-level encode returns codes; wrap with lengths/metadata |
| codebook embeddings | feasible with care | can use quantizer codebooks; API wrapper needed |
| native 50 Hz continuous | feasible but non-official | local wrapper skips final downsample; caveat required |
| all hidden layers | not public API | local upstream checkout has dirty modifications; do not rely on this as released behavior |

Protocol caveat:

- MiMo inference is deterministic at fixed batch size but materially batch-size-sensitive in our local harness.
- Future MiMo feature extraction must record and pin extraction/eval batch size.
- Do not compare MiMo features/scores across batch sizes without an explicit caveat.

Needed code before experiments:

- A feature extraction CLI or module that writes immutable feature manifests.
- Output metadata must include: sample rate, frame rate, representation slice, `n_q`, codebook indices, batch size, dtype, model revision/path, input file IDs, and extraction hash.
- No model-training changes required for Wave 1.

## Baseline preflight

Verdict: **ready**.

Sources:

- Generalizable/calibrated SSL baseline: <https://arxiv.org/html/2309.05384>
- Aletheia code: <https://github.com/danoneata/aletheia>
- Spoof-SUPERB: <https://arxiv.org/html/2603.01482>
- WavLM ASVspoof5 ensemble: <https://arxiv.org/html/2408.07414>
- Whisper ADD: <https://ar5iv.labs.arxiv.org/html/2306.01428>

Recommended lightweight baselines:

1. frozen encoder last-layer mean pool + logistic regression;
2. frozen encoder last-layer mean pool + single linear CE head;
3. tiny MLP only as a secondary row if linear probes show signal.

Baseline model choices:

| Model | Role | First-wave priority |
|---|---|---|
| wav2vec2 XLSR / existing repo path | continuity with old work | high |
| WavLM Base+ or Large | strong modern SSL baseline | high |
| Whisper tiny/base encoder | semantic/audio baseline | medium |
| Spoof-SUPERB weighted-layer setup | benchmark-style stronger baseline | wave 2, not first wave |

Reasoning:

- First wave should test representation value, not backend engineering.
- Frozen SSL + logistic regression is a hard low-parameter baseline in recent literature.
- Weighted all-layer SSL and PEFT baselines are important later but heavier.

## Path-by-path pre-validation

### Path A — Codec/tokenizer forensic cues

Question:

> Do MiMo RVQ/codebook/residual features detect codec-based fake speech better than generic SSL?

Ready because:

- CodecFake+ exists and directly targets codec-based fakes.
- CoSG labels include taxonomy fields.
- MiMo exposes RVQ codebook indices.

First probes:

- MiMo early codebooks vs late codebooks vs all codebooks.
- Continuous 25 Hz vs RVQ tokens.
- Binary bonafide/spoof on CodecFake+ CoSG and/or stratified CoRS.

Kill if:

- only seen codec families work;
- bonafide codec-compressed audio is falsely classified as spoof;
- MiMo never beats/helps a frozen SSL baseline.

Status: **ready**.

### Path B — MiMo complements SSL

Question:

> Does MiMo add independent signal to SSL frontends?

Ready because:

- frozen SSL baseline is easy and literature-supported;
- MiMo feature extraction is feasible;
- fusion can be score-level, no complex model needed.

First probes:

- WavLM/wav2vec2 logistic baseline;
- MiMo logistic baseline;
- late score fusion;
- score correlation/error overlap.

Kill if:

- fusion gains are tiny;
- MiMo errors duplicate SSL errors;
- gains disappear OOD.

Status: **ready**.

### Path C — Source/taxonomy tracing

Question:

> Is MiMo better at identifying codec/generator family than at binary spoofing?

Ready because:

- CodecFake+ CoSG has `Model`, `QUA`, `AUX`, `DEC`, `Label` fields.
- Codec/source tracing is an emerging ASVspoof5 future direction.

First probes:

- classify `Model`, `QUA`, `AUX`, `DEC` from MiMo token stats;
- compare SSL baseline;
- hold out one or more model/codec families if sample counts allow.

Kill if:

- taxonomy prediction collapses under held-out families;
- performance is just speaker/content leakage.

Status: **ready with label-count verification**.

### Path D — Semantic vs acoustic explanation

Question:

> Did MiMo fail because semantic abstraction suppresses forensic cues while acoustic residuals retain them?

Ready because:

- MiMo has a clear RVQ hierarchy.
- SpeechTokenizer and quantizer-aware detection literature support early/late codebook hypotheses.

First probes:

- early codebooks only;
- later residual codebooks only;
- continuous states;
- token histogram vs transition features;
- optional native 50 Hz, caveated.

Kill if:

- no meaningful difference across representation slices;
- all MiMo slices underperform trivial SSL baselines everywhere.

Status: **ready with batch-size caveat**.

### Path E — Robustness/media transformations

Question:

> Does MiMo help under compression, resampling, noise, laundering, or multilingual/media-transformed conditions?

Partly blocked because:

- RADAR official data is not publicly accessible right now.
- SAFE official data is blind/not released.

Available alternatives:

- ASVspoof5 codec/compression/adversarial labels, if downloaded;
- self-applied transforms on CodecFake+/ASVspoof subsets:
  - MP3/Opus/AAC/Vorbis;
  - resampling;
  - additive noise/music;
  - reverberation;
  - dynamic range / normalization;
  - neural codec encode-decode if tools are installed later.

Kill if:

- MiMo degrades faster than SSL;
- gains are codec shortcut artifacts;
- false rejects on benign transformed bonafide speech increase.

Status: **feasible-with-work, not first official path**.

## Wave design

### Wave 0 — local feasibility only

No ML claims.

Tasks:

1. Download/inspect small label/protocol files only:
   - CodecFake+ `CoSG_labels.txt`;
   - CodecFake+ `CoRS_labels.txt` if available without full audio;
   - ASVspoof5 protocol files if/when data is staged.
2. Build a canonical local protocol table:
   - utterance ID;
   - path/archive member;
   - label;
   - source/model;
   - QUA/AUX/DEC where available;
   - split;
   - speaker;
   - caveats.
3. Run MiMo extraction on a tiny local audio subset only to validate shapes and metadata.
4. Estimate storage/time for Wave 1.

Exit criteria:

- protocol table validates counts/splits;
- extraction manifests include required MiMo batch-size/model metadata;
- one tiny feature file roundtrips.

### Wave 1 — feature-only elimination

No full fine-tuning.

Dataset:

- CodecFake+ CoSG first, because it is small and taxonomy-rich.
- Add stratified CoRS subset if storage permits.

Representations:

- frozen SSL mean-pooled baseline;
- MiMo continuous 25 Hz;
- MiMo early RVQ token features;
- MiMo late RVQ token features;
- MiMo all-RVQ token features;
- SSL + MiMo late fusion.

Tasks:

- binary bonafide/spoof;
- source/model/taxonomy classification;
- held-out model/family if sample counts allow;
- error overlap/fusion analysis.

Allowed models:

- logistic regression;
- single linear head;
- tiny MLP only as secondary diagnostic.

Exit criteria:

- kill at least two weak paths, or identify no-signal state and stop;
- promote only paths with clear independent signal.

### Wave 2 — diagnosis for survivors

Possible diagnostics:

- codebook importance;
- token transition vs histogram;
- early/late masking;
- calibration;
- false-reject check for benign codec-compressed bonafide;
- transfer from CoSG to CoRS or vice versa;
- ASVspoof5 codec/compression subset if staged.

### Wave 3 — controlled experiment matrix

Only after Wave 1/2 show signal.

At this point write formal `ExperimentSpec` matrix rows and seed policy.

## Immediate next engineering slice

Do **not** start training.

Implement just enough to run Wave 0 cleanly:

1. dataset/protocol indexer for CodecFake+ labels;
2. MiMo feature extraction manifest format;
3. frozen feature probe runner for cached features;
4. report that aggregates by path/task/representation.

All outputs should be indexed under run-layout v1 or an explicit derived-feature layout. Old historical artifacts remain untouched.

## Recommended decision

Proceed with the five-path portfolio, but prioritize a single cheap first batch:

> **CodecFake+ CoSG feature-only probe over MiMo RVQ slices + SSL baselines, with binary and taxonomy tasks.**

This one batch tests paths A, B, C, and D at once, with limited storage and no full training. Path E should wait for ASVspoof5 staging or self-transform tooling.
