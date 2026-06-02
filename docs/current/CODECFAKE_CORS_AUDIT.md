# CodecFake+ CoRS local audit

Date: 2026-05-31T02:41:16.857079-03:00
Status: local inventory only; no extraction, training, or evaluation

## Result

CoRS archive parts and labels are present locally, but extracted CoRS audio is not ready.

- parts present: `True`
- download log reports all parts downloaded: `True`
- extracted directory present: `False`
- labels rows: `1422560`
- label counts: `{"bonafide": 44455, "spoof": 1378105}`

## Archive parts

| Part | Exists | Size bytes | SHA256 sidecar |
|---|---:|---:|---:|
| `Codecfake_plus_CoRS.part0` | True | 25052014980 | True |
| `Codecfake_plus_CoRS.part1` | True | 25052014980 | True |
| `Codecfake_plus_CoRS.part2` | True | 25052014980 | True |
| `Codecfake_plus_CoRS.part3` | True | 25052014980 | True |

## Label sample

```text
p225 p225_001.wav bonafide
p225 p225_001_audiodec_24k_320d.wav spoof
p225 p225_001_bigcodec.wav spoof
p225 p225_001_DAC24.wav spoof
p225 p225_001_Encodec_24b24k.wav spoof
```

## Download log tail

```text
 99 23.3G   99 23.2G    0     0  10.7M      0  0:37:11  0:37:07  0:00:04 10.5M
 99 23.3G   99 23.2G    0     0  10.7M      0  0:37:11  0:37:08  0:00:03 10.5M
 99 23.3G   99 23.3G    0     0  10.7M      0  0:37:11  0:37:09  0:00:02 10.5M
 99 23.3G   99 23.3G    0     0  10.7M      0  0:37:11  0:37:10  0:00:01 11.0M
 99 23.3G   99 23.3G    0     0  10.7M      0  0:37:11  0:37:11 --:--:-- 11.0M
100 23.3G  100 23.3G    0     0  10.7M      0  0:37:11  0:37:11 --:--:-- 11.1M
[2026-05-26T22:01:44-03:00] complete Codecfake_plus_CoRS.part3 size=25052014980
[2026-05-26T22:01:57-03:00] all CoRS parts downloaded
```

## Decision

Do not start CoRS training yet. CoRS needs an explicit extraction/index/readability step first.

Next required action:

1. estimate extraction storage/time;
2. run logged extraction or archive materialization if approved;
3. index extracted CoRS audio;
4. verify readable sampled audio against `CoRS_labels.txt`;
5. pin CoRS-as-spoof proxy label policy before training.

## Caveats

- This audit did not compute full SHA-256 over the 94GB archive parts.
- This audit did not extract archive contents.
- CoRS is codec-resynthesized proxy data, not literal generated fake speech.
