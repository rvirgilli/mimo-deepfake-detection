# MiMo reference audit

Source: `docs/references/mimo-audio-technical-report.pdf`.

This audit exists because MiMo facts affect the paper framing and some older docs mixed tokenizer, encoder, and 7B model claims.

## What we can rely on from the local report

- MiMo-Audio-Tokenizer is a tokenizer/codec system, not just an encoder.
- The full tokenizer is described as 1.2B parameters.
- Our paper uses only the encoder; the encoder-only 638M count must remain a code-derived claim, not a direct quote from the report.
- Tokenizer training data is reported inconsistently as 10M hours in the intro and over 11M hours in the training section.
- A2T is next-token prediction loss over text conditioned on quantized audio representation.
- Stage 1 trains tokenizer + LLM from scratch with A2T, reconstruction, and commitment losses.
- Stage 2 freezes the encoder and discretizer and trains decoder/vocoder/discriminators for waveform detail.

## Consequences for our manuscript

1. Use `hybrid reconstruction/audio-to-text`, not `reconstruction` alone.
2. Do not imply MiMo uses contrastive learning.
3. Do not attribute the 100M-hour MiMo-Audio 7B pretraining scale to the tokenizer.
4. Do not call the encoder 1.2B parameters.
5. If discussing adversarial losses, state that the report says encoder/discretizer are frozen in stage 2.
6. Keep the 10M vs 11M discrepancy explicit, or avoid exact ratio claims if not central.

## Open validation items

- Verify final public citation metadata for `MiMo-Audio: Audio Language Models are Few-Shot Learners`.
- Verify whether arXiv ID `2512.23808` is correct for the final manuscript.
- Verify released repository README claims against the local PDF claims.
- Verify our encoder-only parameter count script and record it in the provenance ledger.
