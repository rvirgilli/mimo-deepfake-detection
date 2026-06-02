import pytest

from mimodf.scoring.write_scores import ScoreRecord, read_score_file, write_score_file


def test_write_score_file_is_sorted_and_roundtrips(tmp_path):
    path = tmp_path / "scores_LA_eval.txt"

    write_score_file(
        [ScoreRecord("utt_b", -0.2), ScoreRecord("utt_a", 1.25)],
        path,
    )

    assert path.read_text().splitlines() == ["utt_a 1.25", "utt_b -0.2"]
    assert read_score_file(path) == [ScoreRecord("utt_a", 1.25), ScoreRecord("utt_b", -0.2)]


def test_write_score_file_can_preserve_input_order(tmp_path):
    path = tmp_path / "scores_DF_eval.txt"

    write_score_file(
        [ScoreRecord("utt_b", -0.2), ScoreRecord("utt_a", 1.25)],
        path,
        sort_by_utterance_id=False,
    )

    assert path.read_text().splitlines() == ["utt_b -0.2", "utt_a 1.25"]


def test_write_score_file_rejects_duplicate_utterance_ids(tmp_path):
    with pytest.raises(ValueError, match="duplicate utterance_id"):
        write_score_file(
            [ScoreRecord("utt", 0.1), ScoreRecord("utt", 0.2)],
            tmp_path / "scores.txt",
        )


def test_read_score_file_rejects_malformed_lines(tmp_path):
    path = tmp_path / "bad_scores.txt"
    path.write_text("utt_a 0.1 extra\n")

    with pytest.raises(ValueError, match="expected '<utterance_id> <score>'"):
        read_score_file(path)
