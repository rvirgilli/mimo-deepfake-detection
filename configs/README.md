# Configuration templates

Tracked configs are portable templates for reproducible runs. They are not complete records of any specific historical run.

## Local paths

Configs use OmegaConf environment interpolation for machine-specific paths. Set variables in `.env.example` before running training/evaluation, or override values on the command line.

Important variables:

- `ASVSPOOF_LA_DATABASE`
- `ASVSPOOF_PROTOCOLS`
- `WAV2VEC2_XLSR_CHECKPOINT`
- `MIMO_TOKENIZER_MODEL`

Example:

```bash
export ASVSPOOF_LA_DATABASE=/data/ASVspoof_database/LA
export ASVSPOOF_PROTOCOLS=./SSL_Anti-spoofing/database/
export WAV2VEC2_XLSR_CHECKPOINT=./SSL_Anti-spoofing/xlsr2_300m.pt
export MIMO_TOKENIZER_MODEL=./models/MiMo-Audio-Tokenizer
```

## Run records

For claim-bearing experiments, keep the resolved config, command, environment, selected checkpoint, scores, metrics, and logs together in an ignored local run directory. Public docs should cite sanitized summaries rather than raw machine-local artifacts.

## Config groups

- `frontend/` — frontend definitions: wav2vec2, MiMo, HuBERT, EnCodec.
- `dataset/` — dataset/protocol locations and dataloader settings.
- `training/` — general training defaults.
- `rawboost/` — augmentation defaults and sample-rate-specific variants.
- `model/` — AASIST backend/projection architecture.
- `finetune_comparison/` — standalone experiment templates.
