# Experiment logging protocol

Date: 2026-05-26  
Status: mandatory for all new research execution

Machine-readable ledger:

```text
docs/current/research_execution_log.jsonl
```

Human-readable summary:

```text
docs/current/RESEARCH_EXECUTION_LOG.md
```

Validation/summary helpers:

```bash
python -m mimodf log validate
python -m mimodf log summary
```

## Rule

Every experiment/research command that creates, mutates, scores, indexes, trains, evaluates, downloads, extracts, or summarizes research artifacts must have a tracked execution-log entry.

This includes failed, interrupted, exploratory, diagnostic, dry-run, smoke, and pilot runs. A failed run without a log entry is lost evidence.

## What counts as an experiment command

Log all commands that produce or affect any of these:

- dataset/protocol/index files;
- feature caches and feature manifests;
- training/evaluation runs;
- score files and official scorer outputs;
- probe metrics, reports, plots, or tables;
- run manifests, resolved specs, run indexes, aggregates, comparisons;
- downloaded/extracted datasets or model artifacts used for research;
- any command whose output might later influence a claim or decision.

Do not log ordinary development-only commands unless they create research artifacts:

- `pytest`, `compileall`, `git diff --check`, formatting, and static checks belong in verification reports, not the experiment ledger.

## Required lifecycle

### 1. Before running

Create a `planned` JSONL entry or prepare the exact entry text before execution. The entry must include at minimum:

```json
{
  "schema": "mimodf-research-execution-log/v1",
  "run_id": "short-stable-id",
  "wave": "wave1",
  "kind": "feature_probe",
  "status": "planned",
  "cwd": "<repo>",
  "environment": "conda:mimo-df",
  "git_revision_at_plan": "...",
  "command": "exact shell command to run",
  "inputs": ["..."],
  "planned_outputs": ["..."],
  "notes": ["why this run exists", "known caveats"]
}
```

If the command uses meaningful compute, also record expected duration/size/GPU use before asking for approval.

### 2. During execution

Capture start/finish time when possible. Prefer tools/CLIs that write manifests with:

- `started_unix`;
- `finished_unix`;
- `git_revision`;
- `command_argv`;
- paths to generated artifacts.

Do not overwrite an existing output directory unless the log entry says `overwrite: true` and explains why overwriting is safe.

### 3. After execution

Update or append the final entry immediately. Required fields:

```json
{
  "status": "completed|failed|interrupted",
  "started_at": "ISO-8601 or null",
  "finished_at": "ISO-8601 or null",
  "elapsed_sec_manifest": 12.34,
  "elapsed_sec_wall_observed": 15.67,
  "git_revision_at_run": "...",
  "outputs": ["actual artifact paths"],
  "result_summary": {
    "records": 1797,
    "metrics_or_counts": "only concise scalar facts"
  },
  "failure": null
}
```

For failed/interrupted runs, keep the entry and fill:

```json
{
  "status": "failed",
  "failure": {
    "stage": "feature extraction",
    "exit_code": 1,
    "message": "short exact error or log path"
  }
}
```

## Required fields

| Field | Required | Notes |
|---|---:|---|
| `schema` | yes | Current value: `mimodf-research-execution-log/v1`. |
| `run_id` | yes | Stable, unique, grep-friendly. Never reuse for different semantics. |
| `wave` | yes | Example: `wave0`, `wave1`, `system-audit`, `historical-audit`. |
| `kind` | yes | Example: `feature_extraction`, `feature_probe`, `eval`, `training`, `download`. |
| `status` | yes | `planned`, `completed`, `failed`, or `interrupted`. |
| `cwd` | yes | Absolute working directory used. |
| `environment` | yes | Example: `base python`, `conda:mimo-df`, Docker image, GPU host. |
| `command` | yes | Exact command. Preserve paths and flags. |
| `inputs` | yes for new entries | Key files/datasets/checkpoints/specs. |
| `planned_outputs` | yes before run | Intended output paths. |
| `outputs` | yes after run | Actual output paths. |
| `git_revision_at_plan` | yes when possible | `git rev-parse HEAD` before running. |
| `git_revision_at_run` | yes when possible | Revision recorded by manifest/metrics or current HEAD. |
| `started_at` / `finished_at` | yes when possible | ISO-8601 local time is acceptable. |
| `result_summary` | yes after run | Concise counts/metrics; full output remains in artifacts. |
| `notes` | yes for caveats | Include leakage, batch-size, exploratory/diagnostic caveats. |

## Artifact policy

Heavy generated artifacts stay ignored when appropriate, but the log entry is tracked.

A tracked log entry must make the ignored artifact reproducible or auditable by recording:

- exact command;
- exact input paths;
- exact output paths;
- environment;
- git revision;
- protocol/spec/hash when available;
- caveats and failure state.

If a local artifact is deleted later, do not delete the historical log entry. Add a follow-up entry or note that the artifact was removed/retired.

## Agent workflow

Before running an experiment command, the agent must say or write:

1. the `run_id`;
2. whether the execution-log entry is planned;
3. expected outputs;
4. expected compute cost if nontrivial.

After running, the agent must:

1. update `docs/current/research_execution_log.jsonl`;
2. update `docs/current/RESEARCH_EXECUTION_LOG.md` if the summary changes;
3. update task/result docs when the run affects decisions;
4. commit the log/protocol/doc change with the code or result summary.

If the agent cannot update the log, it must stop and ask. Continuing unlogged research execution is not allowed.

## Current feature/probe CLIs

Current feature and probe CLIs write `command_argv` into generated manifests/metrics going forward:

- `python -m mimodf features mimo-extract ...`
- `python -m mimodf features wav2vec2-extract ...`
- `python -m mimodf features probe ...`

The execution ledger remains mandatory even when artifacts contain `command_argv`, because ignored artifacts can be moved or deleted.

## Review checklist

Before closing a research slice, verify:

- every output directory has a matching log entry;
- every metrics/report artifact has a matching log entry;
- failed/interrupted runs are logged;
- log entries include exact commands, paths, and status;
- summaries do not claim more than logged evidence supports.
