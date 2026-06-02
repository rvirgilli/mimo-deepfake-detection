# Scripts

Active scripts should be portable, bounded, and safe by default.

Historical one-off orchestration scripts are not tracked on the public branch. Keep local/ad-hoc scripts under `local_private/` unless they are reusable enough to maintain.

## Policy for new scripts

New active scripts must:

1. accept paths as CLI arguments or environment variables;
2. avoid hardcoded home-directory paths;
3. avoid launching long training by default;
4. clearly distinguish audit/eval-only commands from GPU-training commands;
5. write raw outputs under ignored evidence dirs (`experiments/`, `outputs/`, `logs/`, `local_private/`) unless intentionally generating curated public summaries;
6. never delete checkpoints or scores unless explicitly requested.
