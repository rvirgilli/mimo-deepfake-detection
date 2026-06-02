# Wave 3 deep research B — broader exploration around the finding

Date: 2026-05-31
Status: historical landscape note written before the completed MASKGCT seed-stability diagnostic and PEFT batch14 seed42 pilot. For current claims and next actions, use `WAVE3_REVISED_PLAN_2026_05_31.md` and `WEEKLY_TEAM_UPDATE_2026_05_31.md`.
Scope: broader research landscape and experiment strategy around the current Wave 3A finding.
Current evidence: custom CodecFake+ CoSG source-holdout only. Later evidence supports a mixed-ranking, stable threshold-collapse / worst-source-risk framing for `MASKGCT`, and a matched frozen batch14 control is pending before PEFT architecture claims.

## Executive thesis

The broader program should not become “run every dataset”. The right frame is:

> Build an audited map of when trained audio deepfake detectors transfer, when adaptation helps, and when adaptation creates hidden source-specific failures.

The current MASKGCT finding is a seed for that program. External datasets and full protocols matter, but they should be used to validate this mechanism, not to restart benchmark chasing.

## Current evidence tier

| Evidence tier | Status | Role |
|---|---|---|
| CoSG custom source-holdout frozen XLS-R | complete, 3 seeds | confirmed diagnostic baseline |
| CoSG custom source-holdout PEFT XLS-R | seed42 complete | leading directional model family |
| MASKGCT failure diagnostic | complete for seed42 | primary mechanism candidate |
| CoRS official/proxy CodecFake+ | downloaded, not extracted/indexed | future official/proxy track |
| ASVspoof5 | not staged locally | future external validation |
| media-transform robustness | smoke/probes only | future robustness axis, not claim-bearing yet |

## Dataset and benchmark roles

### CodecFake+ CoSG

Source/value:

- CodecFake+: https://arxiv.org/html/2501.08238v2
- CodecFake+ dataset/project pages: https://huggingface.co/datasets/CodecFake/CodecFake_Plus_Dataset , https://responsiblegenai.github.io/CodecFake-Plus-Dataset/

Best role:

- custom diagnostic source-holdout;
- generator/source failure mapping;
- mechanism discovery.

Claim boundary:

- CoSG source-holdout is not official benchmark training in our current setup.
- It is still valuable because it isolates held-out generator behavior.

### CodecFake+ CoRS

Best role:

- official/proxy training track;
- tests whether proxy scale helps CoSG transfer or teaches shortcuts.

Current blocker:

- archives are present but audio is not extracted/indexed/readability-checked;
- label policy must be explicit because CoRS is codec-resynthesized/proxy data, not literal fake speech.

Next before compute:

1. storage/extraction plan;
2. archive integrity/readability check;
3. protocol index;
4. label policy doc;
5. small loader smoke;
6. only then training.

### ASVspoof5 Track 1

Sources/value:

- ASVspoof5 overview: https://arxiv.org/abs/2408.08739
- ASVspoof5 Zenodo: https://zenodo.org/records/14498691
- Evaluation plan: https://www.asvspoof.org/file/ASVspoof5___Evaluation_Plan_Phase2.pdf
- Baselines/scoring: https://github.com/asvspoof-challenge/asvspoof5

Best role:

- external validation after the CoSG/CoRS mechanism is stable;
- tests whether source-shift/adaptation conclusions generalize beyond CodecFake+.

Claim boundary:

- official ASVspoof5 claims require official Track 1 protocols/scorer;
- minDCF is primary in the challenge framing, EER secondary;
- no current local evidence because ASVspoof5 is not staged.

### ASVspoof2021 LA/DF

Source/value:

- ASVspoof 2021 challenge context: http://www.asvspoof.org/index2021.html
- DF analysis context: https://arxiv.org/html/2210.02437

Best role:

- historical comparison and channel/codec robustness precedent;
- not the next main compute target unless we need a low-friction external sanity check using already staged data.

### WaveFake / In-the-Wild / ADD-style datasets

Sources/value:

- WaveFake: https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/c74d97b01eae257e44aa9d5bade97baf-Paper-round2.pdf
- In-the-Wild audio deepfake dataset: https://arxiv.org/abs/2203.16263
- ADD challenge overview: https://arxiv.org/html/2408.04967v3

Best role:

- broader source/open-world generalization context;
- useful for writing related work and later external checks.

Not next:

- do not stage these before ASVspoof5/CoRS unless a specific hypothesis needs them.

### Media robustness

Sources/value:

- Robustness under compression/modification/neural codecs: https://arxiv.org/html/2503.17577
- RawBoost augmentation precedent: https://ar5iv.labs.arxiv.org/html/2111.04433

Best role:

- test whether PEFT gains are brittle under MP3/noise/resampling/codec transforms;
- compare clean-only vs media-augmented PEFT.

Claim boundary:

- current media-transform artifacts are smoke/feature-drift only;
- no robustness EER claim until transformed scoring infrastructure and durable artifacts exist.

## Broader concept map

### 1. Source shift is the main axis

The current result is about generator/source shift, not only average deepfake detection. Any paper should report:

- source-macro metrics;
- worst-source metrics;
- source-wise confidence/support;
- seed variability;
- validation-vs-test source mismatch.

### 2. Adaptation is both intervention and risk

PEFT is promising because it improves 8/9 folds, but the same adaptation may create a stronger shortcut. This makes PEFT the central object of study, not just a model variant.

### 3. Calibration and ranking must be separated

VALLE-like behavior: good ranking, bad threshold.
MASKGCT-like behavior: ranking breaks.
These are different failure modes and should not be collapsed into one EER number.

### 4. Official and diagnostic tracks must stay separate

Diagnostic tracks answer mechanism questions. Official tracks answer benchmark/protocol questions. Mixing them will corrupt the claims.

### 5. External validation should be hypothesis-driven

ASVspoof5 should not be “more data”. It should test:

- does PEFT improve average transfer externally?
- does PEFT create worst-source/worst-attack failures?
- does official validation detect those failures?

## Recommended roadmap

### P0 — lock the current finding

Run:

```text
PEFT MASKGCT seeds 123 and 2024
```

Goal:

- verify whether the source inversion is seed-stable.

Why first:

- cheapest decisive test;
- directly validates or weakens the emerging paper thesis.

### P1 — make PEFT vs frozen fair

If P0 repeats or remains concerning, run:

```text
PEFT all-fold seeds 123 and 2024
```

Goal:

- compare frozen vs PEFT as 3-seed source-holdout matrices;
- quantify whether PEFT average gain survives seeds;
- quantify worst-source risk.

### P2 — write the mechanism note

Produce a focused note:

```text
Adaptation-induced source inversion under codec-generator holdout
```

Include:

- score distributions;
- source-wise ROC/threshold diagnostics;
- validation-vs-test mismatch;
- seed stability;
- artifact/confound audit.

### P3 — CoRS official/proxy setup

Before any training:

1. extract CoRS;
2. index rows;
3. verify readability;
4. pin labels;
5. document official split use;
6. smoke loader/model.

Then test:

```text
CoRS proxy train -> CoSG evaluation
CoSG-only PEFT -> CoSG evaluation
CoRS pretrain + CoSG finetune -> CoSG evaluation
```

Core question:

> Does proxy scale improve transfer, or does it teach shortcuts?

### P4 — ASVspoof5 external validation

Before any training/eval:

1. stage ASVspoof5 audio/protocols;
2. integrate Track 1 scorer;
3. decide closed vs open condition;
4. run no-training dry-run/index check;
5. run a tiny scoring smoke;
6. only then evaluate/train.

Core question:

> Does the adaptation-vs-worst-source tradeoff generalize beyond CodecFake+?

### P5 — media robustness

After PEFT seed stability:

- generate durable transformed CoSG artifacts;
- score clean and transformed audio using selected checkpoints;
- compare margin erosion, AUROC/EER deltas, and label flips;
- optionally train media-augmented PEFT.

Core question:

> Does media augmentation reduce worst-source and transform brittleness, or just improve average transformed performance?

## What not to do

Do not run:

- full fine-tuning yet;
- broad frontend search;
- Optuna/HPO;
- ASVspoof5 training before protocol/staging;
- CoRS training before extraction/indexing/label policy;
- MiMo revival runs;
- CLAMTTS-only analysis without a new predictive mechanism.

## Possible paper framings

### Framing A — strongest current path

Title shape:

> Adaptation-induced source inversion in audio deepfake detection

Contribution:

- show frozen probes weakly triage trained failures;
- show whether PEFT improves average held-out generator transfer under matched protocol controls;
- show PEFT can invert ranking on a large held-out generator;
- propose source-wise validation/diagnostic protocol.

Risk:

- needs PEFT seed replication.

### Framing B — dataset/protocol audit paper

Title shape:

> Official vs diagnostic protocols for codec-generated speech deepfake detection

Contribution:

- separate CoRS official/proxy from CoSG diagnostic source-holdout;
- compare proxy training vs source-holdout diagnostics;
- show claim hygiene matters.

Risk:

- requires CoRS extraction and more engineering.

### Framing C — robustness paper

Title shape:

> Source and media robustness of SSL audio deepfake detectors

Contribution:

- combine source holdout and media transforms;
- compare clean-only and augmented PEFT;
- report margin erosion and worst-source risk.

Risk:

- needs durable transform scoring infrastructure.

Recommended framing now: **A**, with B/C as future validation tracks.

## Decision table

| Candidate next branch | Scientific value | Cost | Risk | Verdict |
|---|---:|---:|---:|---|
| PEFT MASKGCT seeds 123/2024 | very high | low | low | do now |
| PEFT all-fold seeds 123/2024 | high | medium | low | do after MASKGCT check |
| CoRS extraction/indexing | high | medium/storage | medium | plan next, no training yet |
| ASVspoof5 staging | high | medium/storage | medium | plan after mechanism lock |
| full fine-tuning | unclear | high | high | defer |
| HPO/Optuna | low | high | high | do not do |
| new frontend sweep | low now | medium/high | high | do not do |

## Claim hygiene template

Use this language:

> In a custom CoSG source-holdout diagnostic, frozen XLS-R establishes a stable baseline across three seeds. A PEFT adapter improves average seed42 performance across most held-out sources, but fails sharply on MASKGCT, where non-MASKGCT validation selects a checkpoint with below-chance held-out ranking. This suggests adaptation can improve average transfer while increasing worst-source risk.

Avoid:

- “official CodecFake+ result”;
- “PEFT is better” before matched controls and seeds;
- “MASKGCT proves detector failure” before seed replication;
- “ASVspoof5 relevance” before staging/scoring.

## Summary

The broader path is clear:

1. stabilize the MASKGCT finding;
2. make PEFT-vs-frozen a fair seed comparison;
3. write the adaptation-induced inversion mechanism;
4. then validate externally through CoRS and ASVspoof5.

This keeps the project scientific, bounded, and auditable.
