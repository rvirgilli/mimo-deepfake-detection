# References kept locally

## MiMo-Audio technical report

File: `mimo-audio-technical-report.pdf`

Metadata:

- Title in PDF text: `MiMo-Audio: Audio Language Models are Few-Shot Learners`
- Pages: 31
- PDF creation date: 2025-09-18
- SHA-256: `fffe6ae7e7663b422b54b6b9286f46708fb09a0da5d1be889a80d97f396b8c5e`
- Size: 1,013,263 bytes
- URL printed in abstract: `https://github.com/XiaomiMiMo/MiMo-Audio`

Why this is tracked:

- It is the closest source document for MiMo-Audio-Tokenizer architecture/training facts used in our paper.
- Web/arXiv availability was not verified from this workspace, so relying only on an external URL would be brittle.
- It is small enough to version (~1 MB) and materially affects how we frame MiMo.

## Facts extracted for our paper audit

These facts should be verified against the final public MiMo paper before resubmission, but they are supported by this local technical report.

### Scope distinction

The report describes both:

1. **MiMo-Audio-Tokenizer** — tokenizer/codec system used by our project as an encoder frontend.
2. **MiMo-Audio-7B** — larger audio language model trained on tokenizer outputs.

Do not mix their scales:

- The tokenizer is described as a **1.2B-parameter model** comprising encoder, discretization, decoder, and vocoder.
- Our paper uses only the tokenizer **encoder**; encoder-only count is a separate code-derived fact (`638M`) and is not directly stated in this report.
- The report's **over 100M hours** claim refers to MiMo-Audio 7B pretraining, not the tokenizer.

### Tokenizer architecture

From Section 2.1.1:

- Input audio: single-channel waveform at **24 kHz**.
- Audio is converted to mel spectrogram at **100 Hz**.
- Audio encoder has input and output **2x downsampling** layers around a central bidirectional Transformer encoder.
- Central encoder: **32 layers**, **20 attention heads**, model dimension **1280**, FFN dimension **5120**, RoPE, GELU.
- Layer-3 hidden states are added to final-layer output by element-wise summation.
- Encoder continuous representations are at **25 Hz**.
- RVQ discretization has **20 layers**:
  - first two codebooks size **1024**;
  - remaining codebooks size **128**.
- The full MiMo-Audio language model uses the first **8** codebooks, yielding **200 audio tokens/s** (25 Hz × 8).

### Tokenizer training

From Section 2.1.2:

- The report is internally inconsistent on tokenizer data scale:
  - introduction says tokenizer trained on a **10-million-hour corpus**;
  - training section says **over 11 million hours**.
- Stage 1 trains MiMo-Audio-Tokenizer and an LLM from scratch with multi-task learning:
  - audio reconstruction;
  - audio-to-text (A2T);
  - commitment loss.
- A2T is **next-token prediction cross-entropy**, not contrastive/alignment loss:

```text
L_A2T = - sum_i log p(t_i | Q_tilde, t_1, ..., t_{i-1})
```

- Stage-1 loss weights:
  - `lambda_A2T = 10.0`
  - `lambda_recon = 1.0`
  - `lambda_commit = 1.0`
- Stage 2 freezes the audio encoder and discretization module, then trains decoder/vocoder with adversarial and feature-matching losses to improve waveform reconstruction.

Implication for our paper:

- It is accurate to call MiMo's tokenizer objective **hybrid reconstruction/audio-to-text**.
- It is not accurate to call it purely reconstruction-based.
- It is not contrastive in the wav2vec2 sense.
- For encoder behavior, adversarial stage-2 losses should be discussed carefully because the report says the encoder/discretizer are frozen during stage 2.

## Citation TODO

Current manuscript cites:

```bibtex
D. Zhang et al., "MiMo-Audio: Audio language models are few-shot learners,"
arXiv preprint arXiv:2512.23808, 2025.
```

Before resubmission, verify the final public citation metadata against the actual public paper/repository. The local PDF itself does not show the arXiv identifier in extracted text.
