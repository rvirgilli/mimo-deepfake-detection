import pytest

from mimodf.scoring.evaluate import (
    EvaluationBatch,
    EvaluationItem,
    score_batches,
    write_scores_from_batches,
)
from mimodf.scoring.write_scores import ScoreRecord


def test_score_batches_calls_predictor_in_batch_order():
    batches = [
        EvaluationBatch.from_items([EvaluationItem("utt_b", 2.0), EvaluationItem("utt_a", 1.0)]),
        EvaluationBatch.from_items([EvaluationItem("utt_c", -3.0)]),
    ]
    seen_inputs = []

    def predictor(inputs):
        seen_inputs.append(list(inputs))
        return [value * 10 for value in inputs]

    assert score_batches(batches, predictor) == [
        ScoreRecord("utt_b", 20.0),
        ScoreRecord("utt_a", 10.0),
        ScoreRecord("utt_c", -30.0),
    ]
    assert seen_inputs == [[2.0, 1.0], [-3.0]]


def test_write_scores_from_batches_uses_deterministic_file_order(tmp_path):
    path = tmp_path / "scores_LA_eval.txt"
    batches = [
        EvaluationBatch.from_items([EvaluationItem("utt_b", 2.0), EvaluationItem("utt_a", 1.0)])
    ]

    write_scores_from_batches(batches, lambda inputs: inputs, path)

    assert path.read_text().splitlines() == ["utt_a 1", "utt_b 2"]


def test_predictor_score_count_must_match_inputs():
    batch = EvaluationBatch.from_items([EvaluationItem("utt_a", 1.0), EvaluationItem("utt_b", 2.0)])

    with pytest.raises(ValueError, match="returned 1 scores for 2 inputs"):
        score_batches([batch], lambda inputs: [0.0])


def test_duplicate_ids_across_batches_fail_before_writing(tmp_path):
    batches = [
        EvaluationBatch.from_items([EvaluationItem("utt", 1.0)]),
        EvaluationBatch.from_items([EvaluationItem("utt", 2.0)]),
    ]

    with pytest.raises(ValueError, match="duplicate utterance_id"):
        write_scores_from_batches(batches, lambda inputs: inputs, tmp_path / "scores.txt")


def test_empty_batches_fail():
    with pytest.raises(ValueError, match="no evaluation items"):
        score_batches([], lambda inputs: [])


def test_empty_batch_fails_at_construction():
    with pytest.raises(ValueError, match="must not be empty"):
        EvaluationBatch.from_items([])
