# Project brief: MiMo deepfake detection paper rework

This is the shared working memory for the rework. It should stay stricter than the manuscript and weaker than the evidence.

## Final goal

Rebuild the earlier manuscript into a defensible, reviewable study and evolve the repository into a lean scientific research system whose claims and future experiments are traceable to machine-readable evidence.

Success means:

1. every reported number has seed-level provenance;
2. unsupported claims are removed or weakened;
3. central tables use official/appropriate scoring;
4. exclusions and protocol asymmetries are explicit;
5. no costly reruns happen unless we agree they are unavoidable;
6. new experiments use versioned specs, resolved specs, manifests, score contracts, and generated reports;
7. old artifacts remain readable and auditable even as new parameters/components are added.

## Scientific purpose

The repo studies which pretrained audio representations transfer for audio deepfake detection under realistic distribution shifts, and how to compare them without hidden protocol drift.

MiMo is no longer the central bet. It is one frontend/representation family among others. It may be compared, used diagnostically, or dropped when evidence says it is not useful.

Relevant representation families include:

- **SSL speech encoders**: wav2vec2/XLS-R, WavLM, HuBERT;
- **tokenizer/codec/reconstruction encoders**: MiMo and related codec-tokenizer representations;
- **simple acoustic/spectro-temporal baselines** when they clarify failure modes.

The interesting question is not "can MiMo beat wav2vec2?" It is:

> Under which data shifts, spoofing mechanisms, and media transformations do different frontend representation families help or fail?

Current assessment: wav2vec2 still appears stronger in audited ASVspoof evidence and in Wave 1 CodecFake+ held-out-source probes. MiMo has signal and one source-conditional win, but Wave 1 does not justify broad MiMo training, Optuna, or superiority claims. The strongest current contribution is an audit-first experimental framework, with early evidence toward representation failure-mode mapping.

See `docs/current/RESEARCH_PURPOSE_RESET.md` for the accepted post-Wave-1 framing.

## Paper claims to keep in mind

Current `docs/paper.tex` claims:

- main table compares EER/min-tDCF on ASVspoof 2021 LA and EER on DF;
- wav2vec2: frozen → adapter → full improves on LA;
- MiMo: frozen → adapter improves, full regresses toward frozen;
- LA has a frozen-feature crossover where MiMo beats wav2vec2 frozen, but wav2vec2 wins after adaptation;
- DF has no crossover and MiMo underperforms wav2vec2;
- MiMo full fine-tuning sensitivity/regularization sweeps do not recover both LA and DF;
- interpretation invokes feature distortion, but this is currently too strong unless representation evidence is added.

Treat all of these as hypotheses until entered in `RESULTS_PROVENANCE.md`.

## Research-system architecture

Current `mimodf` research-system contracts:

```text
ExperimentSpec v1
  -> resolved_spec.yaml + stable spec hash
  -> RunLayout v1 / RunManifest v1
  -> eval/train/scoring artifacts
  -> RunIndex records
  -> aggregate/compare/report outputs
```

Implemented pieces:

- `mimodf experiment validate/resolve/init/inspect`;
- stable component registry metadata;
- MiMo batch-size caveat in component metadata;
- optional run-layout v1 manifest updates for `mimodf eval run` and `mimodf train legacy-asvspoof`;
- `mimodf report index` over new run manifests and historical table provenance.

Pending pieces:

- Optuna under the spec contract;
- public dependency/download setup.

## Legacy/model code architecture

Training/evaluation stack:

```text
Hydra configs
  -> train.py / evaluate.py
  -> src.frontends.get_frontend(...)
  -> frontend extracts frame features
  -> projection to AASIST input dimension
  -> src.model.Model AASIST-style graph backend
  -> scores/result files under ignored experiment dirs
```

Important modules:

- `src/frontends/base.py` — frontend contract.
- `src/frontends/wav2vec2.py` — XLS-R frontend plus adapter hooks.
- `src/frontends/mimo.py` — MiMo tokenizer encoder wrapper, feature extraction, 25/50 Hz handling.
- `src/frontends/mimo_features.py` — continuous/RVQ/dual/layer-select feature strategies.
- `src/frontends/mimo_finetune.py` — adapter, LoRA, partial, gradual unfreezing wrappers.
- `src/model.py` — AASIST-like backend and projection integration.
- `src/data_utils.py` — ASVspoof data/protocol loading and RawBoost integration.
- `src/rawboost.py` — RawBoost with sample-rate-aware scaling.
- `src/experiment.py`, `src/results.py` — manifests/results DB, useful but not sufficient provenance by themselves.

## Good points

- Clear frontend abstraction makes encoder swaps local.
- MiMo integration is isolated enough to test feature strategies, upsampling, and fine-tuning wrappers.
- Configs are now portable portable templates rather than machine-specific run records.
- Tests cover projection, RawBoost scaling, MiMo feature strategies, fine-tuning wrappers, and opt-in native MiMo integration.
- Active docs now separate audit truth (`docs/current`) from historical history (`docs/archive`, `scripts/archive`).
- The repo now preserves old scripts/artifacts without pretending they are canonical.
- New research contracts are plain YAML/JSON/JSONL rather than heavy experiment infrastructure.
- New and historical records can now be indexed together with source type and reproducibility tier.

## Main flaws and risks

### Evidence/provenance

- Main-table seed rows are now populated for wav2vec2 frozen/adapter/full, MiMo frozen/full, and the two found MiMo adapter evals.
- Current paper numbers remain unsafe where they mix seed sets or artifact families.
- Summary files are not enough; each seed still needs score/result/config/checkpoint/evaluator evidence.
- MiMo adapter n=5 is invalid unless missing artifacts are recovered; local full-eval evidence is n=2.
- MiMo frozen seed 456 appears excluded in the paper without sufficient documented failure evidence and must be included by default.
- LA min-tDCF has been reconciled in `TDCF_RECONCILIATION.md`; several paper tDCF values use different seed sets than their EER rows.

### Protocol mismatch

- Paper says ASVspoof2019 dev validation, but `train.py` uses a fast ASVspoof2021 eval subset if files exist.
- `train.py` uses Adam unless `training.encoder_lr` enables AdamW param groups; paper says AdamW generally.
- `evaluate_accuracy` hardcodes validation loss weights `[0.1, 0.9]`, while some paper-era configs used different training loss weights.
- Full fine-tuning comparisons may be protocol-confounded, especially if wav2vec2 full uses Tak/SSL scripts while MiMo uses this repo's stack.

### Scientific framing

- MiMo is not pure reconstruction; use hybrid reconstruction/audio-to-text.
- The full tokenizer is reported as ~1.2B params, but this repo uses encoder-only MiMo; 638M must be code-derived.
- Two encoders cannot establish an objective-class law. The paper should say "case study" and "consistent with", not causal proof.
- Feature distortion is plausible but not proven by result tables alone.

### Engineering hygiene

- Archived scripts contain hardcoded local paths and stale assumptions; keep them as evidence pointers only.
- External dependencies (`SSL_Anti-spoofing`, `MiMo-Audio-Tokenizer`) are local/ignored and not yet turnkey public setup, although expected revisions/hashes are audited.
- Historical run artifacts are scattered but now indexable as historical provenance records.
- No stored new-layout real smoke/reproduction example has been promoted into the repo yet.

## Assessment conclusion

The assessment phase is concluded in `ASSESSMENT_CONCLUSION.md`.

Carry-forward framing:

- an audited, model-specific case study of wav2vec2/XLS-R and MiMo-Audio-Tokenizer;
- explicit seed/source/protocol caveats;
- MiMo adapter reported only as n=2 exploratory unless future controlled rerun is approved;
- no general contrastive-vs-reconstruction law;
- no feature-distortion proof without representation evidence.

Next phase must start from `ASSESSMENT_CONCLUSION.md`, not the earlier manuscript narrative.

## Codebase/research-system direction

Deep code audit is in `CODEBASE_AUDIT.md`; implementation plan is in `CODE_REWORK_PLAN.md`. The research-framework contract is in `RESEARCH_SYSTEM_SPEC.md`; operating guidance is in `RESEARCH_FRAMEWORK_GUIDELINES.md`.

Decision: rebuild from harvested pieces rather than broad in-place refactor. Keep old `src/` and artifacts during migration. The active direction is contracts first: structured provenance/table/scoring tooling, typed configs, experiment specs, manifests, run layout, indexes, then controlled training/frontends/backend migration.

## Rules for decisions

- Existing artifacts first; evaluation-only reruns second; training reruns last and only with approval.
- Main table gets audited before robustness/figures/prose polish.
- No silent exclusions.
- Claims must be narrower than evidence.
- Prefer deleting/weakening claims over defending messy provenance.
- New parameters/components require stable IDs or schema evolution; never silently reinterpret old runs.
- Keep the framework lean: contracts and generated files before services/databases/platforms.
