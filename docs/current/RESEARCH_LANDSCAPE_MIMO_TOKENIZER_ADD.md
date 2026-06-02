# Research landscape: MiMo/tokenizer features for audio deepfake detection

Date: 2026-05-26  
Status: Exa-backed literature map, not an experiment result  
Scope: recent audio deepfake detection (ADD), neural audio codecs/tokenizers, MiMo-style representations, and plausible revival paths for the original hypothesis.

## Executive take

The original idea is still interesting, but the publishable version cannot be "MiMo tokenizer beats SOTA" yet.

The stronger research direction is:

> **Do codec/tokenizer hierarchies expose forensic cues that SSL speech encoders suppress, especially for codec-based or media-transformed deepfakes?**

Recent literature moved toward exactly this neighborhood:

- codec-based fake speech is now treated as its own threat family (`CodecFake`, `CodecFake+`);
- ASVspoof 5 explicitly includes neural codec/compression conditions and warns that codec artifacts blur the bonafide/spoof boundary;
- new work models RVQ quantizer hierarchy directly for deepfake detection;
- PEFT/LoRA/MoE adaptation of SSL frontends is becoming a strong default;
- robustness benchmarks now stress multilingual, compressed, transformed, and laundered audio, not only clean ASVspoof19-style spoofing.

This supports reviving the idea, but with a narrower claim:

> **MiMo is a candidate codec-tokenizer representation whose semantic/acoustic/RVQ hierarchy may complement SSL frontends. The key question is where forensic cues live: continuous encoder states, coarse semantic codebooks, fine residual codebooks, or their fusion.**

Our previous negative result does not kill the idea. It mostly tells us that the specific tested configuration — pooled MiMo features in the old ASVspoof-style setup, with unresolved batch-size drift and weak adapter evidence — did not deliver.

## 2026-05-30 Wave 1 design clarifications

Follow-up Exa review before redefining Wave 1 sharpened the design:

1. **CodecFake+ is useful, but CoRS needs labeling discipline.** CodecFake+ provides CoSG/CoRS subsets and taxonomy fields (`QUA`, `AUX`, `DEC`), but CoRS is codec-resynthesized speech used as a proxy for CoSG detection, not literal fake speech. CoSG should be the primary binary-evidence dataset; CoRS requires an explicit policy.
2. **Source generalization is the core validation.** Recent CodecFake+ source-tracing work reports overfitting to non-speech regions and failure on unseen CoSG content. Random splits are therefore sanity checks only; source/group holdouts are the first claim-bearing tests.
3. **SSL is a frontend family baseline.** wav2vec2/XLS-R, WavLM, HuBERT, and Whisper-style representations are common strong baselines. MiMo should be tested as a tokenizer-family contrast against SSL, not as the default center of the study.
4. **Robustness belongs early, but tiny.** Robustness literature and challenges stress MP3/Opus/AAC compression, resampling, noise, reverberation, packet loss, and laundering. Exploratory Wave 1 should include a tiny transform smoke, not full robustness generation.
5. **Add a boring acoustic baseline.** LFCC/MFCC/CQCC/log-mel-style features remain useful for interpretability and robustness comparisons. A log-mel mean/std baseline is required before deepening complex frontend explanations.

The machine-readable corrected Wave 1 design is `docs/current/wave1_exploratory_validation_matrix.yaml`.

## Why the old framing failed

Likely failure modes:

1. **Wrong representation slice.** MiMo was used mostly as a feature extractor; we may have ignored the RVQ hierarchy where fine residual artifacts could live.
2. **Semantic pressure can erase forensic cues.** MiMo optimizes semantic/audio-to-text alignment plus reconstruction. Semantic abstraction is good for speech tasks but can suppress spoof artifacts.
3. **Temporal/frequency smoothing.** MiMo uses mel input and downsampling. Short high-frequency or phase artifacts may be attenuated before the classifier sees them.
4. **Dataset mismatch.** ASVspoof19/21 clean LA may reward different artifacts than modern codec-based generators, compression, or in-the-wild transforms.
5. **Backend too shallow or wrong.** A single pooling/classification backend may miss quantizer-level or temporal transition patterns.
6. **Protocol instability.** Our MiMo path is batch-size-sensitive. Fresh MiMo reruns remain caveated until eval batch size is pinned and drift is documented/handled.

## Landscape shift since the original idea

### 1. Detection moved from clean LA to robustness/generalization

ASVspoof 5 uses crowdsourced speech from almost 2,000 speakers, unlike older controlled ASVspoof databases, and includes contemporary TTS/VC plus adversarial attacks. Its evaluation also includes encoded/compressed conditions such as MP3, Opus, AMR, Speex, M4A, EnCodec, EnCodec+MP3, and telephony simulation. Source: [ASVspoof 5 design](https://arxiv.org/html/2502.08857), [ASVspoof 5 evaluation](https://ar5iv.labs.arxiv.org/html/2601.03944).

The ASVspoof 5 evaluation discussion explicitly flags neural codec ambiguity: bona fide speech processed by a neural codec can show artifacts resembling spoofs, so detecting "mere vocoding artifacts" is becoming unreliable. This is central for any tokenizer/codec-feature paper.

RADAR 2026 extends the target to multilingual/media-transformed detection, with 102,726 evaluation utterances across English, Singapore English, Mandarin, Taiwanese Mandarin, Japanese, and Vietnamese, plus heterogeneous sources, synthesis systems, and transformations. Source: [RADAR Challenge 2026](https://arxiv.org/html/2605.09568v1).

SAFE similarly frames synthetic audio forensics under raw, processed/compressed, and laundered conditions. Source: [SAFE Challenge](https://dl.acm.org/doi/10.1145/3733102.3736707).

Implication: a revived MiMo paper should not optimize only for ASVspoof19 LA. It needs at least one codec/robustness axis.

### 2. Codec-based fake speech is now a named threat

`CodecFake` argues that anti-spoofing models trained on common datasets struggle with speech from codec-based generation systems, and builds a codec re-synthesis dataset from 15 codec models. Source: [CodecFake](https://arxiv.org/html/2406.07237v1).

`CodecFake+` scales this up: training data from 31 open-source neural audio codec models, evaluation from 17 advanced codec-based speech generation models, plus a taxonomy by vector quantizer, auxiliary objective, and decoder type. It reports that codec re-synthesized speech (CoRS) is useful training data and that decoder-type balancing improves performance. Source: [CodecFake+](https://arxiv.org/html/2501.08238), [HF dataset](https://huggingface.co/datasets/CodecFake/CodecFake_Plus_Dataset).

`How to Label Resynthesized Audio` studies whether codec-resynthesized audio should be labeled bonafide or spoof, using an ASVspoof 5 extension. Source: [arXiv:2602.16343](https://arxiv.org/abs/2602.16343v1).

Implication: MiMo/tokenizer features are more relevant if the threat model is codec-based generation, codec compression, or source tracing — but the labeling policy must be explicit.

### 3. Tokenizer/RVQ hierarchy is becoming directly useful

`Quantizer-Aware Hierarchical Neural Codec Modeling for Speech Deepfake Detection` argues that RVQ quantizers form a coarse-to-fine hierarchy: early quantizers encode coarse structure, later quantizers refine residual details that may reveal synthesis artifacts. It keeps the speech encoder frozen, updates 4.4% additional parameters, and reports relative EER reductions of 46.2% on ASVspoof 2019 and 13.9% on ASVspoof5 over strong baselines. Source: [arXiv:2603.16914](https://arxiv.org/html/2603.16914v1).

SpeechTokenizer gives a useful conceptual model: first RVQ layer behaves more like semantic/content tokens; later quantizers carry acoustic/timbre residuals. Source: [SpeechTokenizer](https://arxiv.org/html/2308.16692), [implementation notes](https://github.com/zhangxinfd/speechtokenizer).

Implication: the promising MiMo variant is not "use the final pooled embedding." It is **per-quantizer analysis/fusion**, especially fine residual codebooks and token transitions.

### 4. SSL frontends remain the hard baseline

Frozen SSL representations plus logistic regression can already generalize surprisingly well: one study reports RawNet2 EER 30.9% vs 8.8% using frozen self-supervised representations across eight datasets, with fewer than 2k learned parameters and calibration analysis. Source: [generalizable/calibrated SSL ADD](https://arxiv.org/html/2309.05384).

ASVspoof5 representation benchmarking reports wav2vec2/WavLM among the strongest pretrained representations. Source: [WavLM model ensemble for ASVspoof5](https://arxiv.org/html/2408.07414).

Whisper features also improve in-the-wild deepfake detection in prior work. Source: [Whisper ADD](https://ar5iv.labs.arxiv.org/html/2306.01428).

A 2026 SUPERB-style benchmark targets SSL models for audio deepfake detection with a reproducible leaderboard. Source: [Spoof-SUPERB](https://arxiv.org/abs/2603.01482v1).

Implication: MiMo must be compared against frozen SSL linear/MLP baselines, not only against our old wav2vec2 implementation.

### 5. PEFT/MoE is now the default adaptation story

MoLEx combines LoRA with Mixture-of-Experts routing for SSL models and reports 5.56% EER on ASVspoof5 eval without augmentation. Source: [MoLEx](https://arxiv.org/html/2509.09175v1).

Another MoE-LoRA approach integrates multiple LoRA adapters into Wav2Vec2 attention layers and reports average OOD EER dropping from 8.55% to 6.08%. Source: [MoE-LoRA ADD](https://arxiv.org/html/2509.13878v1).

Meta-learned LoRA updates about 3.6M parameters / about 1.1% of full fine-tuning and reports OOD EER from 8.84% to 5.30%. Source: [meta-learned LoRA](https://arxiv.org/html/2502.10838).

Implication: a future MiMo adapter story should be PEFT-aware and compare against LoRA/adapter SSL baselines. But the first revival step should still be frozen probing, because it isolates representation value.

## What MiMo actually gives us

MiMo-Audio-Tokenizer is not just an audio codec and not just a semantic tokenizer. It is a large hybrid tokenizer trained for both reconstruction and audio-to-text alignment.

Reported architecture/facts:

- 1.2B-parameter Transformer tokenizer;
- encoder + discretization + decoder;
- 24 kHz mono audio;
- mel input at 100 Hz;
- continuous representations at 25 Hz;
- 8 RVQ layers / 200 tokens/sec in the paper-level description;
- released config/repo path indicates a 20-layer RVQ implementation with first two larger codebooks and later residual codebooks;
- trained from scratch on about 10M/11M+ hours;
- MiMo-Audio LLM patches four RVQ timesteps to 6.25 Hz for language modeling.

Sources: [MiMo-Audio paper](https://www.arxiv.org/pdf/2512.23808), [MiMo-Audio-Tokenizer HF](https://huggingface.co/XiaomiMiMo/MiMo-Audio-Tokenizer), [MiMo-Audio-Tokenizer GitHub](https://github.com/XiaomiMiMo/MiMo-Audio-Tokenizer).

Detection-relevant interpretation:

| MiMo slice | Why it may help | Why it may fail |
|---|---|---|
| Continuous encoder states | retains acoustic structure before hard quantization | still mel/downsampled; may smooth artifacts |
| Native 50 Hz states | higher temporal resolution; better for local artifacts | not the official released downstream representation; our repo needs caveat |
| Early RVQ codebooks | semantic/content invariance; may reduce speaker/content confounds | likely discards spoof residuals |
| Later RVQ codebooks | residual/timbre/fine codec artifacts | may overfit codec family or bitrate |
| Token histograms | captures codec/source fingerprints | weak for temporal/local manipulations |
| Token transitions | captures unnatural generation dynamics | needs more careful backend/data |
| MiMo LLM 6.25 Hz patches | useful for generative modeling | likely too coarse for forensic cues |

## Core hypotheses to test later

These are research hypotheses, not experiment specs yet.

### H1 — Residual codebooks carry forensic cues

Later MiMo RVQ layers should separate bonafide vs codec/generated speech better than early semantic layers, especially on codec-based fakes.

Cheap test:

- extract per-codebook token histograms and transition statistics;
- train logistic regression / small MLP;
- evaluate on CodecFake+ and ASVspoof5 codec/compression subsets.

Success signal:

- later codebooks outperform early codebooks on codec-based generation;
- performance remains above chance on unseen codec families.

Failure signal:

- only same-codec training/test works;
- bonafide codec-compressed audio is falsely classified as spoof.

### H2 — MiMo complements SSL, not replaces it

SSL encoders capture robust speech/acoustic context; MiMo residual tokens may add codec/source artifacts. Fusion may beat either alone under codec/media transformations.

Cheap test:

- frozen WavLM/wav2vec2 linear baseline;
- frozen MiMo continuous/token baseline;
- late score fusion or small gated fusion;
- same seeds/protocols.

Success signal:

- fusion improves OOD codec/transformation EER/calibration without hurting bonafide codec-compressed negatives.

### H3 — Semantic tokenizers are bad standalone forensic detectors

If first/semantic codebooks perform poorly while residual/acoustic codebooks help, that is publishable: it explains the negative result and gives design guidance.

Cheap test:

- early-vs-late codebook ablation;
- semantic/acoustic grouping;
- token dropout / codebook masking.

Success signal:

- clear codebook-level gradient of forensic utility.

### H4 — MiMo is useful for source tracing before binary detection

Codec/tokenizer representations may identify codec/generator family better than classify bonafide/spoof robustly.

Cheap test:

- multi-class codec/source classification on CodecFake+ taxonomy fields (`QUA`, `AUX`, `DEC`, generator family);
- open-set stress where one codec family is held out.

Success signal:

- source/taxonomy prediction works even where binary detection is ambiguous.

### H5 — The old ASVspoof19-only setup was the wrong proving ground

MiMo may underperform on old LA but become useful on codec-generation, compression, and media-transformation stress tests.

Cheap test:

- compare ASVspoof19 LA vs ASVspoof5 codec/compression vs CodecFake+;
- avoid training-heavy runs at first.

## Recommended research sequence

### Phase 0 — No GPU-heavy training

1. Finish literature map review and decide threat model:
   - binary spoof detection;
   - codec-based fake detection;
   - bonafide codec-compression robustness;
   - source tracing;
   - partial/localized manipulation.
2. Decide labeling policy for CoRS:
   - spoof proxy;
   - bonafide codec transmission;
   - separate third class.
3. Freeze claims:
   - no "MiMo beats SOTA" claim;
   - yes "tokenizer hierarchy as forensic representation" claim if evidence supports it.

### Phase 1 — Feature-only probes

No full fine-tuning.

Rows:

- WavLM/wav2vec2 frozen baseline + logistic/MLP head;
- MiMo continuous 25 Hz;
- MiMo native 50 Hz if caveated;
- MiMo early RVQ codebooks;
- MiMo late RVQ codebooks;
- MiMo token histogram/transitions;
- simple SSL + MiMo fusion.

Datasets/subsets:

- historical ASVspoof19 LA for continuity;
- ASVspoof5 codec/compression/adversarial subsets if available;
- CodecFake+ CoRS/CoSG;
- bonafide neural-codec-compressed negatives if available/constructible.

Metrics:

- EER;
- minDCF/actDCF where protocol supports it;
- calibration error or Cllr;
- per-condition EER by codec/generator/transform;
- false rejects on bonafide codec-compressed audio.

### Phase 2 — Representation diagnosis

Before training adapters:

- codebook ablation;
- layer/time resolution ablation;
- artifact localization or per-frame score heatmaps;
- score correlation between SSL and MiMo;
- failure clusters by codec taxonomy (`QUA`, `AUX`, `DEC`).

### Phase 3 — PEFT only if Phase 1/2 show signal

Candidate adaptation:

- LoRA/adapters on MiMo continuous encoder states;
- quantizer-aware gated backend;
- MoE over codebook groups;
- fusion with WavLM/wav2vec2 LoRA baseline.

Do not run broad PEFT until frozen probes show MiMo contains independent signal.

## Stop criteria

Stop/reframe away from MiMo as detector if:

- MiMo residual codebooks do not beat shallow SSL baselines on any codec/robustness slice;
- gains disappear when bonafide codec-compressed negatives are included;
- MiMo only learns codec-family shortcuts with poor held-out codec transfer;
- batch-size drift prevents trustworthy score reproduction and cannot be pinned as protocol.

Continue if:

- later RVQ/residual layers consistently outperform early/semantic layers for codec-based fakes;
- MiMo adds complementary signal to SSL under codec/media transformations;
- source/taxonomy tracing works even when binary detection is ambiguous;
- calibration or false-reject behavior improves under explicit codec-compression conditions.

## Paper angle if revived

Possible title direction:

> **Where do audio-tokenizer representations encode forensic evidence? A codebook-level study for codec-based speech deepfake detection**

Safer claims:

- not "new SOTA detector";
- not "MiMo universally better";
- yes "semantic/acoustic token hierarchy matters";
- yes "codec-tokenizer residuals expose/erase different forensic cues";
- yes "bonafide codec compression complicates spoof labels";
- yes "MiMo negative result is explained by representation/protocol choice."

## Key source map

| Area | Source | Why it matters |
|---|---|---|
| Codec-based fake detection | [CodecFake](https://arxiv.org/html/2406.07237v1) | Shows common anti-spoofing models struggle with codec-based generation; creates codec re-synthesis dataset. |
| Codec dataset/taxonomy | [CodecFake+](https://arxiv.org/html/2501.08238) | 31 codec training models, 17 CoSG eval systems, taxonomy by quantizer/objective/decoder. |
| CoRS label ambiguity | [Dual Role of Neural Audio Codecs](https://arxiv.org/abs/2602.16343v1) | Directly addresses whether codec-resynthesized audio should be bonafide or spoof. |
| RVQ hierarchy for ADD | [Quantizer-Aware Hierarchical Neural Codec Modeling](https://arxiv.org/html/2603.16914v1) | Strongest direct support for codebook-level forensic modeling. |
| Modern benchmark | [ASVspoof 5 design](https://arxiv.org/html/2502.08857) | Crowdsourced, adversarial, codec/compression conditions. |
| Challenge analysis | [ASVspoof 5 evaluation](https://ar5iv.labs.arxiv.org/html/2601.03944) | Notes neural codec ambiguity and architecture homogeneity. |
| Robust transformed benchmark | [RADAR 2026](https://arxiv.org/html/2605.09568v1) | Multilingual/media-transformed stress benchmark. |
| SSL baseline | [Generalizable/calibrated SSL ADD](https://arxiv.org/html/2309.05384) | Frozen SSL + logistic regression is a hard low-parameter baseline. |
| ASVspoof5 SSL baseline | [WavLM ensemble](https://arxiv.org/html/2408.07414) | wav2vec2/WavLM representations are strong on ASVspoof5. |
| SSL benchmark | [Spoof-SUPERB](https://arxiv.org/abs/2603.01482v1) | Emerging reproducible benchmark for SSL representations in ADD. |
| PEFT | [MoLEx](https://arxiv.org/html/2509.09175v1) | LoRA+MoE PEFT gives strong ASVspoof5 results. |
| PEFT/OOD | [MoE-LoRA ADD](https://arxiv.org/html/2509.13878v1) | LoRA experts improve out-of-domain EER. |
| PEFT/meta | [Meta-learned LoRA](https://arxiv.org/html/2502.10838) | Low-parameter adaptation improves OOD generalization. |
| MiMo tokenizer | [MiMo-Audio paper](https://www.arxiv.org/pdf/2512.23808) | Defines MiMo as hybrid semantic/acoustic tokenizer. |
| MiMo implementation | [MiMo-Audio-Tokenizer](https://github.com/XiaomiMiMo/MiMo-Audio-Tokenizer) | Released architecture/config facts. |
| Tokenizer hierarchy | [SpeechTokenizer](https://arxiv.org/html/2308.16692) | Conceptual support for semantic early codebooks and acoustic residual later codebooks. |

## Immediate decision needed

Before writing experiment specs, choose the primary revival target:

1. **CodecFake detection** — MiMo as codec/tokenizer forensic representation.
2. **Robustness under media transformations** — MiMo + SSL fusion for transformed/laundered audio.
3. **Source tracing** — MiMo token hierarchy predicts codec/generator taxonomy.
4. **Negative-result explanatory paper** — why semantic/audio tokenizers do or do not carry spoof cues.

My recommendation: start with **1 + 3** as cheap feature probes. They are closest to MiMo's actual structure and require less faith than trying to beat broad ADD SOTA immediately.
