import pytest

from mimodf.scoring.tdcf_summary import (
    TDCFValidationError,
    load_tdcf_summary,
    render_tdcf_summary_markdown,
)

TDCF_VALUES = "docs/current/official_tdcf_values.yaml"


def test_load_tdcf_summary_matches_reconciled_means():
    summary = load_tdcf_summary(TDCF_VALUES)
    rows = {row.row_id: row for row in summary.rows}

    assert rows["wav2vec2_adapter"].mean == pytest.approx(0.25528)
    assert rows["mimo_full"].mean == pytest.approx(0.34956)
    assert rows["mimo_adapter"].mean == pytest.approx(0.29725)
    assert rows["mimo_frozen"].n == 4
    assert len(summary.wrong_scale_examples) == 4


def test_render_tdcf_summary_markdown():
    markdown = render_tdcf_summary_markdown(load_tdcf_summary(TDCF_VALUES))

    assert "| wav2vec2_adapter | 5 | 0.2553 | 0.255 | corrected_same_seed_set |" in markdown
    assert "| mimo_frozen | 4 | 0.3605 | 0.361* | partial_tdcf_n4 |" in markdown


def test_tdcf_summary_rejects_inconsistent_reported_mean(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
version: 1
rows:
  - row_id: bad
    reported_mean: 0.5
    display: "0.500"
    status: bad
    values:
      - {seed: 1, tdcf: 0.1}
      - {seed: 2, tdcf: 0.3}
""".strip()
    )

    with pytest.raises(TDCFValidationError, match="reported_mean"):
        load_tdcf_summary(path)
