import json

from mimodf.cli import main
from mimodf.evaluation.plan import (
    build_evaluation_plan,
    render_evaluation_plan_json,
    render_evaluation_plan_markdown,
)


def write_config(path):
    path.write_text(
        """
model: wav2vec2
strategy: adapter
seed: 42
protocol:
  train_set: ASVspoof2019_LA_train
  validation_set: ASVspoof2021_LA_fast_eval_subset
  checkpoint_selection_set: ASVspoof2021_LA_fast_eval_subset
  eval_set: ASVspoof2021_LA_eval_and_DF_eval
optimizer:
  name: adam
  lr: 0.0001
  weight_decay: 0.0
  encoder_lr: null
scorer:
  la: official_asvspoof2021_la
  df: eer_only
""".strip()
    )


def test_evaluation_plan_passes_without_importing_model_code(tmp_path):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.pt"
    eval_root = tmp_path / "eval_root"
    scorer = tmp_path / "evaluate_2021_LA.py"
    score_out = tmp_path / "scores" / "scores_LA_eval.txt"
    write_config(config)
    checkpoint.write_text("checkpoint")
    eval_root.mkdir()
    scorer.write_text("print('score')\n")
    score_out.parent.mkdir()

    plan = build_evaluation_plan(
        config_path=config,
        checkpoint=checkpoint,
        eval_root=eval_root,
        score_out=score_out,
        track="LA",
        scorer=scorer,
    )

    assert plan.passed is True
    assert plan.model == "wav2vec2"
    assert plan.scorer_name == "official_asvspoof2021_la"
    assert all(check.status == "ok" for check in plan.path_checks)
    assert "# Evaluation dry-run plan" in render_evaluation_plan_markdown(plan)
    assert json.loads(render_evaluation_plan_json(plan))["track"] == "LA"


def test_evaluation_plan_reports_missing_and_unsafe_paths(tmp_path):
    config = tmp_path / "config.yaml"
    eval_root = tmp_path / "eval_root"
    score_out = tmp_path / "scores.txt"
    write_config(config)
    eval_root.mkdir()
    score_out.write_text("already exists")

    plan = build_evaluation_plan(
        config_path=config,
        checkpoint=tmp_path / "missing.pt",
        eval_root=eval_root,
        score_out=score_out,
        track="DF",
    )
    statuses = {check.name: check.status for check in plan.path_checks}

    assert plan.passed is False
    assert statuses["checkpoint"] == "missing"
    assert statuses["score_out"] == "exists"
    assert "scorer" not in statuses


def test_evaluation_plan_cli_strict(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.pt"
    eval_root = tmp_path / "eval_root"
    scorer = tmp_path / "evaluate_2021_LA.py"
    write_config(config)
    checkpoint.write_text("checkpoint")
    eval_root.mkdir()
    scorer.write_text("print('score')\n")

    rc = main(
        [
            "eval",
            "plan",
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--eval-root",
            str(eval_root),
            "--score-out",
            str(tmp_path / "scores.txt"),
            "--track",
            "LA",
            "--scorer",
            str(scorer),
            "--format",
            "json",
            "--strict",
        ]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["track"] == "LA"

    strict_fail = main(
        [
            "eval",
            "plan",
            "--config",
            str(config),
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--eval-root",
            str(eval_root),
            "--score-out",
            str(tmp_path / "scores.txt"),
            "--track",
            "LA",
            "--scorer",
            str(scorer),
            "--strict",
        ]
    )
    assert strict_fail == 1
