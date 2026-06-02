# Research Wave 0/1 execution plan

Date: 2026-05-26  
Status: historical plan plus 2026-05-30 design correction
Current machine-readable design reference: `docs/current/wave1_exploratory_validation_matrix.yaml`
Parents:

- `RESEARCH_LANDSCAPE_MIMO_TOKENIZER_ADD.md`
- `RESEARCH_PREFLIGHT_5_PATHS.md`

## 2026-05-30 design correction

Wave 1 should be broader than the original MiMo-revival framing.

Corrected Wave 1 intent:

> Build a broad, shallow, feature-only map of frontend families and shift axes on CodecFake+ CoSG before deepening any mechanism.

Early waves should explore; later waves should deepen only effects that survive grouped controls.
MiMo remains a candidate tokenizer-family contrast, not the organizing goal.

This correction is captured as a machine-readable matrix in:

```text
docs/current/wave1_exploratory_validation_matrix.yaml
```

Practical consequences:

- source-holdout remains the primary claim-bearing validation;
- random row splits stay sanity checks only;
- taxonomy probes are hypothesis generators until grouped/source controls exist;
- a boring acoustic baseline (`logmel_mean_std`) belongs in Wave 1;
- a tiny media-transform smoke belongs in exploratory coverage, but full transform generation needs separate logging/approval;
- no training, Optuna, broad eval, or paper claim follows from Wave 1 alone.

Coverage after the executed Wave 1/Wave 2 work:

- completed: XLS-R, MiMo continuous/RVQ, source-holdout, random diagnostics, taxonomy diagnostics, fusion/error overlap, CLAMTTS-vs-NS diagnostics, WavLM as a second SSL comparator, and log-mel mean/std baseline;
- missing before deepening: a tiny media-transform smoke plan/technical validation.

## Goal

Restart the MiMo/tokenizer idea without overcommitting to one path.

We will run cheap waves that can confirm or eliminate five candidate paths before any broad training, HPO, or paper claim.

Candidate paths:

1. **Codec/tokenizer forensic cues** — MiMo codebooks/residuals detect codec-based fake speech.
2. **MiMo + SSL complementarity** — MiMo adds independent signal to WavLM/wav2vec2/Whisper.
3. **Source/taxonomy tracing** — MiMo is better at codec/generator/source characterization than binary spoofing.
4. **Semantic vs acoustic explanation** — early semantic codebooks lose forensic cues; later acoustic/residual codebooks retain them.
5. **Robustness/media transformations** — MiMo helps under compression/resampling/noise/laundering.

Wave 0 and Wave 1 are deliberately small. They are not meant to produce paper numbers.

## Non-goals

Do not do these in Wave 0/1:

- no full training;
- no full ASVspoof5 download unless explicitly approved;
- no full CodecFake+ CoRS download unless explicitly approved;
- no full LA/DF eval;
- no Optuna;
- no SOTA claim;
- no paper table update;
- no mixing MiMo results across batch sizes;
- no silently treating codec-resynthesized bonafide speech as fake without marking that labeling policy.

## Data policy

Historical artifacts stay immutable.

New downloaded datasets and extracted features are local/generated artifacts. They must not be committed unless deliberately promoted as tiny fixtures.

Preferred local roots:

```text
<data>/mimodf/codecfake_plus/
<data>/mimodf/asvspoof5/
/features/mimodf/
/experiments/runs/
```

Repo-local `/features/` and `/experiments/` are already ignored.

Every generated artifact should have a manifest containing:

- source dataset URL/revision if available;
- local path;
- file size/hash where practical;
- protocol/index hash;
- extraction/model config;
- command;
- git revision;
- caveats.

## Wave 0 — feasibility and indexing

### Purpose

Validate that Wave 1 can run cleanly before doing ML.

Wave 0 answers:

1. Can we obtain small CodecFake+ labels and, ideally, CoSG audio?
2. Can we build a canonical protocol table without leakage-prone ambiguity?
3. Can MiMo extract the required representation slices on tiny audio?
4. Can frozen SSL baseline extraction run on the same tiny IDs?
5. How much storage/time will Wave 1 need?

### Inputs

Minimum:

- CodecFake+ `CoSG_labels.txt`;
- CodecFake+ CoSG audio archive, if explicitly downloaded;
- a tiny local audio subset for extraction smoke.

Optional:

- CodecFake+ `CoRS_labels.txt` only, no CoRS audio yet;
- ASVspoof5 protocol files only, no full audio yet.

### Canonical protocol table

Create one protocol table before any probes.

Recommended file:

```text
/features/mimodf/wave0/codecfake_plus_protocol.jsonl
```

Record schema:

```json
{
  "schema": "mimodf-protocol-record/v1",
  "dataset_id": "codecfake_plus",
  "subset": "CoSG",
  "utterance_id": "...",
  "clip_id": "...",
  "audio_path": "...",
  "archive_member": "...",
  "label": "bonafide|spoof",
  "source_model": "...",
  "quantizer_type": "...",
  "auxiliary_objective": "...",
  "decoder_type": "...",
  "speaker_id": null,
  "split_hint": null,
  "caveats": []
}
```

For CoRS, add codec-name extraction from filename and later join to taxonomy fields from the paper/table. Do not infer missing taxonomy silently.

### Leakage rules

CoSG may contain bonafide and generated versions of related clips. Therefore:

- split by `clip_id` where possible for binary tasks;
- for source/taxonomy tasks, report whether same content appears across train/test;
- hold-out model/family tests must hold out all rows from that source model/family;
- never use random row splits as the only result.

### MiMo extraction smoke

Validate these representation slices on a tiny subset:

| Slice | Required? | Notes |
|---|---:|---|
| `mimo_continuous_25hz` | yes | official-ish released path |
| `mimo_rvq_codes_all` | yes | all 20 codebooks from local config |
| `mimo_rvq_codes_early` | yes | codebooks 0-1 |
| `mimo_rvq_codes_late` | yes | codebooks 2-19 or a documented late group |
| `mimo_continuous_50hz_native` | optional | local/non-official; caveat required |

Required metadata:

```json
{
  "schema": "mimodf-feature-manifest/v1",
  "component_id": "frontend/mimo-audio-tokenizer@...",
  "representation_id": "mimo_rvq_codes_late",
  "model_path": "...",
  "model_revision": "...",
  "sample_rate": 24000,
  "frame_rate_hz": 25,
  "num_quantizers": 20,
  "selected_quantizers": [2, 3, 4],
  "batch_size": 1,
  "dtype": "bf16|fp16|...",
  "input_protocol_hash": "...",
  "output_path": "...",
  "caveats": ["MiMo batch-size-sensitive; compare only within pinned batch-size protocol"]
}
```

MiMo batch size is a protocol fact. Wave 0 should use batch size 1 for extraction smoke, then choose and pin a Wave 1 extraction batch size.

### SSL baseline smoke

Run at least one frozen SSL baseline smoke on the same tiny IDs:

- first choice: WavLM Base+ or existing wav2vec2/XLSR path;
- representation: last hidden state mean pooled;
- classifier is not needed in Wave 0.

### Wave 0 acceptance gates

Wave 0 passes only if:

- protocol table validates row counts and required fields;
- split/grouping policy is explicit;
- at least one MiMo continuous slice and one MiMo RVQ slice extract successfully;
- at least one SSL baseline feature extracts on same IDs;
- feature manifests record batch size, model path/revision, representation ID, and protocol hash;
- storage/time estimate for Wave 1 is documented.

If these fail, do not start Wave 1.

## Wave 1 — first feature-only elimination probe

### Purpose

Run one cheap experiment batch that tests paths A, B, C, and D together.

Wave 1 is a **feature-only probe**:

- frozen frontends;
- cached features;
- logistic regression / linear probes;
- no full fine-tuning;
- no HPO beyond fixed simple regularization choices documented beforehand.

### Primary dataset

Start with CodecFake+ CoSG because:

- small archive compared with CoRS;
- direct taxonomy labels: `Model`, `QUA`, `AUX`, `DEC`, `Label`;
- tests codec/generator-source questions directly.

If CoSG is too small for some tasks, add a small stratified CoRS subset only after Wave 0 validates its protocol mapping.

### Representations

Minimum rows:

| ID | Representation | Tests paths |
|---|---|---|
| `ssl_wavlm_or_w2v_mean` | frozen SSL mean-pooled continuous features | baseline for A/B |
| `mimo_cont_25hz_mean_std` | MiMo continuous 25 Hz pooled mean/std | A/B/D |
| `mimo_rvq_early_hist` | MiMo codebooks 0-1 unigram histograms | D |
| `mimo_rvq_late_hist` | MiMo later codebook unigram histograms | A/D |
| `mimo_rvq_all_hist` | all codebook unigram histograms | A/C/D |
| `ssl_plus_mimo_score_fusion` | late score fusion of best SSL + best MiMo | B |

Optional rows after minimum works:

- token transition features;
- native 50 Hz continuous;
- Whisper tiny/base mean-pooled features;
- small MLP.

### Feature definitions

Keep feature extraction boring and inspectable.

Continuous features:

- mean pool over valid frames;
- optional standard deviation pool;
- no attention pooling in Wave 1.

RVQ token features:

- per-codebook unigram histogram normalized by valid frame count;
- concatenate selected codebooks;
- store codebook grouping in manifest;
- transition histograms only after unigram features work.

Fusion:

- score-level average or logistic calibration on dev split;
- no learned deep fusion in Wave 1.

### Tasks

#### Task 1 — binary bonafide/spoof

Question:

> Is MiMo useful for codec-based fake detection?

Splits:

- content/clip-disjoint split where possible;
- random row split may be reported only as diagnostic, never as the main result.

Metrics:

- EER;
- AUROC;
- balanced accuracy;
- per-source-model EER where counts allow;
- false reject rate for bonafide rows.

Path decisions:

- supports A if MiMo late/all RVQ or continuous features are competitive with SSL or stronger on held-out source/model;
- weakens A if MiMo is worse than SSL everywhere and no fusion gain appears.

#### Task 2 — source/model classification

Question:

> Does MiMo encode generator/codec identity better than binary spoof status?

Labels:

- `Model` for CoSG;
- `codec_name` or joined codec source for CoRS if used.

Metrics:

- macro F1;
- balanced accuracy;
- confusion matrix by source model;
- held-out source test where feasible.

Path decisions:

- supports C if MiMo token features outperform SSL in source/taxonomy classification;
- weakens C if source classification is near chance or content leakage explains performance.

#### Task 3 — taxonomy classification

Question:

> Are MiMo features sensitive to codec architecture factors?

Labels:

- `QUA` — quantizer type;
- `AUX` — auxiliary objective;
- `DEC` — decoder type.

Metrics:

- macro F1;
- balanced accuracy;
- per-class support table.

Path decisions:

- supports C/D if codebook groups differ systematically by taxonomy;
- weakens C/D if no representation slice predicts taxonomy beyond chance.

#### Task 4 — early vs late MiMo ablation

Question:

> Are forensic cues in late/acoustic residual codebooks rather than early/semantic codebooks?

Comparison:

- early codebooks vs late codebooks vs all codebooks;
- continuous features as reference.

Path decisions:

- supports D if late codebooks consistently outperform early codebooks for binary/source/taxonomy tasks;
- weakens D if all codebook groups behave the same.

#### Task 5 — SSL complementarity

Question:

> Does MiMo add independent signal to SSL?

Analysis:

- compare SSL-only vs MiMo-only vs fusion;
- score correlation;
- overlap of errors;
- per-source cases where MiMo corrects SSL or vice versa.

Path decisions:

- supports B if fusion improves held-out performance and errors are meaningfully complementary;
- weakens B if MiMo errors duplicate SSL and fusion does not help.

### Model/probe rules

Allowed first-pass classifiers:

- scikit-learn logistic regression;
- linear SVM only if logistic has numerical issues;
- single linear CE head only if sklearn path is insufficient;
- tiny MLP as a secondary diagnostic, not primary evidence.

Fixed preprocessing:

- standardize continuous features using train split only;
- keep sparse histograms normalized;
- no feature selection using test labels;
- no seed selective reporting.

Seed policy:

- if split is randomized, predeclare 3 seeds maximum;
- report all seeds;
- no silent seed exclusions.

### Promotion/kill criteria

Wave 1 should force decisions.

Promote a path to Wave 2 if at least one robust signal appears:

| Path | Promote if | Kill/deprioritize if |
|---|---|---|
| A | MiMo late/all RVQ or continuous features beat/approach SSL on codec fake detection, especially held-out source/model | MiMo worse than SSL everywhere, no condition-specific gain |
| B | SSL+MiMo fusion improves over both single systems and errors are complementary | fusion flat or worse; score/errors highly redundant |
| C | MiMo token features predict source/taxonomy better than SSL or chance under sane splits | taxonomy/source results collapse under held-out grouping |
| D | late/acoustic codebooks clearly outperform early/semantic codebooks | early/late/all indistinguishable or all weak |
| E | not primary in Wave 1 | remains deferred unless ASVspoof5/self-transform probe is staged |

Do not overinterpret small in-domain gains. A path needs either held-out evidence, consistent codebook pattern, or useful complementarity.

### Wave 1 outputs

Expected generated artifacts:

```text
/features/mimodf/wave1/codecfake_plus_cosg_protocol.jsonl
/features/mimodf/wave1/<representation>/features.*
/features/mimodf/wave1/<representation>/manifest.json
/experiments/runs/<wave1-run-id>/resolved_spec.yaml
/experiments/runs/<wave1-run-id>/manifest.json
/experiments/runs/<wave1-run-id>/metrics.json
/experiments/runs/<wave1-run-id>/report.md
```

Expected committed docs after Wave 1:

```text
docs/current/RESEARCH_WAVE_1_RESULTS.md
docs/current/DECISION_LOG.md
docs/current/TASK_BOARD.md
```

## Re-evaluation meeting template

After Wave 1, fill this table before any next experiment:

| Path | Evidence observed | Decision | Next action |
|---|---|---|---|
| A Codec/tokenizer cues |  | promote / kill / hold |  |
| B MiMo+SSL complementarity |  | promote / kill / hold |  |
| C Source/taxonomy tracing |  | promote / kill / hold |  |
| D Semantic vs acoustic explanation |  | promote / kill / hold |  |
| E Robustness transforms |  | promote / kill / hold |  |

Then choose exactly one of:

1. stop MiMo revival;
2. run Wave 2 diagnostics for one or two surviving paths;
3. stage ASVspoof5/self-transform robustness probe for Path E;
4. write controlled matrix specs for a narrowed hypothesis.

## Immediate implementation checklist

Before Wave 0 starts:

- [ ] Add CodecFake+ protocol indexer.
- [ ] Add feature-manifest schema/helper or reuse experiment manifest with feature artifact metadata.
- [ ] Add MiMo feature extraction wrapper for continuous 25 Hz and RVQ indices.
- [ ] Add frozen-feature probe runner for cached features.
- [ ] Add report template for Wave 0/1 outputs.
- [ ] Add lightweight tests using tiny fake protocol/audio-feature fixtures.

No GPU-heavy command should be run until this checklist is implemented and reviewed.
