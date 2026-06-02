import json
from pathlib import Path

import pytest

from mimodf.scoring.official import (
    OfficialScoringError,
    parse_official_la_output,
    parse_project_result_file,
)


def test_parse_official_la_output():
    parsed = parse_official_la_output("min_tDCF: 0.2707\neer: 3.32\n")

    assert parsed.min_tdcf == pytest.approx(0.2707)
    assert parsed.eer_percent == pytest.approx(3.32)


def test_project_wrong_scale_tdcf_is_not_official_output():
    text = Path(
        "experiments/paper_final/wav2vec2_adapter_multiseed/seed_42/eval/results_LA_eval.txt"
    ).read_text()

    with pytest.raises(OfficialScoringError, match="project 'min t-DCF:'"):
        parse_official_la_output(text)


def test_project_result_parser_is_diagnostic_only():
    result = parse_project_result_file(
        "experiments/paper_final/wav2vec2_adapter_multiseed/seed_42/eval/results_LA_eval.txt"
    )

    assert result.eer_percent == pytest.approx(3.3217)
    assert result.project_min_tdcf == pytest.approx(0.0073)


def test_official_parse_result_is_json_serializable():
    parsed = parse_official_la_output("min_tDCF: 0.3496\neer: 6.94\n")

    payload = json.dumps(parsed.to_dict())

    assert '"min_tdcf": 0.3496' in payload
    assert '"eer_percent": 6.94' in payload
