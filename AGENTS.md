# AGENTS.md

Instructions for AI agents working in this repository.

Keep public branches clean. Raw logs, reviewer notes, submitted PDFs, generated figures, local paths, and ad-hoc scripts belong in `local_private/` or another ignored directory, not in Git.

## Start every session

1. Run `git status -sb`.
2. Read `README.md` and `docs/current/README.md`.
3. Inspect the relevant code/tests before editing.
4. Work on one task unless the user explicitly changes priority.

## Source of truth

Use this hierarchy:

1. Maintained source code and tests.
2. Machine-readable public summaries in `docs/current/`.
3. Curated current documentation.
4. Ignored local evidence under `local_private/`, `experiments/`, `outputs/`, `logs/`, `features/`, or `models/`.

If local evidence is needed for a public claim, create a sanitized summary rather than committing raw artifacts.

## Hard rules

- No paper number without provenance.
- No silent seed exclusions.
- No long/GPU training without explicit user approval.
- Claims must be narrower than evidence.
- Do not commit raw stdout/stderr logs, PIDs, local paths, submitted PDFs, reviewer notes, generated figures, or checkpoints.
- Every new maintained behavior needs focused tests before commit.
- Failed/interrupted runs are records, but raw records stay local unless sanitized for publication.

## Experiment logging

Before running an experiment command:

1. choose a stable `run_id`;
2. record command, inputs, outputs, environment, git revision, caveats, and expected compute cost in an ignored local ledger;
3. ask before meaningful GPU/long-running work.

After the command finishes:

1. update the local record to `completed`, `failed`, or `interrupted`;
2. write a sanitized public summary only if the result affects shared claims;
3. keep raw logs under ignored local paths.

## Verification

Before claiming work is complete, run relevant checks and report exact output.

Fast checks:

```bash
pytest -q
python -m compileall -q mimodf src train.py
```

MiMo integration tests are opt-in:

```bash
RUN_MIMO_INTEGRATION=1 conda run -n mimo-df python -m pytest tests/test_native_50hz.py -q
```

## When to ask the user

Ask before continuing if:

- a central result is missing evidence;
- seed inclusion/exclusion is ambiguous;
- a rerun needs meaningful GPU time;
- a claim would require stronger language than evidence supports;
- cleanup would remove public files rather than moving them to ignored local storage.
