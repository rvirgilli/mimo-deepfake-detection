# CodecFake+ official split summary

Date: 2026-05-31T05:17:00.379045-03:00
Status: split-policy summary; no training, scoring, or extraction

## Decision

CodecFake+ gives an official split for **CoRS**, not for the locally staged **CoSG** label file.

- CoRS official split: speaker-based train/validation/Eval CoRS.
- CoSG local labels: web-sourced evaluation set with source-model labels, no train/validation split field.
- Wave 3 should separate official/proxy training from custom CoSG diagnostics.

## CoRS split rule

From CodecFake+ paper: p226/p229 are validation, p227/p228 are Eval CoRS, remaining speakers are train.

| Split | Speakers | Bonafide | Spoof | Total |
|---|---:|---:|---:|---:|
| `evaluation` | 2 | 755 | 23405 | 24160 |
| `train` | 106 | 42965 | 1331915 | 1374880 |
| `validation` | 2 | 735 | 22785 | 23520 |

## CoSG status

The local `CoSG_labels.txt` has no split field. Treat CoSG source-holdout as a custom diagnostic protocol, not an official benchmark split.

## MaskGCT note

The paper mentions additional MaskGCT-VCTK train/validation/evaluation sets, but no local split/protocol files were found in the staged CodecFake+ directory. Do not assume those splits are available until explicitly staged/audited.

## Consequence for Wave 3

1. Official/proxy track: CoRS official train/validation/eval, then CoSG transfer/evaluation after CoRS extraction.
2. Diagnostic track: CoSG leave-one-source-out failure-map validation.
