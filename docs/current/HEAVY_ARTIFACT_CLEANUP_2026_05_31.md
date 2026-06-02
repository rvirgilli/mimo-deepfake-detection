# Heavy artifact cleanup 2026-05-31

Status: completed local cleanup/archive move

## Actions

1. Deleted reproducible local environment:

```text
<repo>/SSL_Anti-spoofing/.venv
```

2. Moved historical checkpoint directory to archive path:

```text
<repo>/experiments/paper_final/wav2vec2_fullft_matched_weights
-> <artifact-archive>/mimo-deepfake-detection/2026-05-31/experiments/paper_final/wav2vec2_fullft_matched_weights
```

## Sizes

| Item | Bytes | Approx GiB |
|---|---:|---:|
| archived directory | 102123967471 | 95.11 |
| deleted venv | 5173053073 | 4.82 |
| repo before | 237923681025 | 221.58 |
| repo after | 130626660481 | 121.66 |
| repo reduction | 107297020544 | 99.93 |
| /home free delta | 5245583360 | 4.89 |

## Caveat

The archive path is still on `/home`, so the move reduces repository size but does not free the full archived-directory size on the filesystem. Only the deleted venv materially frees disk.

## Restore

```bash
mv <artifact-archive>/mimo-deepfake-detection/2026-05-31/experiments/paper_final/wav2vec2_fullft_matched_weights <repo>/experiments/paper_final/wav2vec2_fullft_matched_weights
```

## Manifest

Full per-file SHA-256 manifest is in:

```text
<repo>/docs/current/heavy_artifact_cleanup_2026_05_31.yaml
```
