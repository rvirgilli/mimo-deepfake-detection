from pathlib import Path

import pytest

from mimodf.provenance import ProvenanceError, load_main_table_provenance
from mimodf.tables.main_table import render_main_table, summarize_eer, summarize_tdcf

PROVENANCE = Path("docs/current/main_table_provenance.yaml")


def test_loads_audited_main_table_provenance():
    provenance = load_main_table_provenance(PROVENANCE)

    assert provenance.version == 1
    assert [row.id for row in provenance.rows] == [
        "wav2vec2_frozen",
        "wav2vec2_adapter",
        "wav2vec2_full_local",
        "mimo_frozen",
        "mimo_adapter",
        "mimo_full",
    ]


def test_eer_uses_sample_standard_deviation():
    provenance = load_main_table_provenance(PROVENANCE)

    frozen = provenance.row("wav2vec2_frozen")
    la = summarize_eer(frozen.la_eer_values)
    df = summarize_eer(frozen.df_eer_values)

    assert la.mean == pytest.approx(8.046, abs=1e-6)
    assert la.sample_std == pytest.approx(0.731526, abs=1e-6)
    assert df.mean == pytest.approx(6.758, abs=1e-6)
    assert df.sample_std == pytest.approx(0.611572, abs=1e-6)


def test_tdcf_mean_matches_reconciliation_values():
    provenance = load_main_table_provenance(PROVENANCE)

    assert summarize_tdcf(provenance.row("wav2vec2_adapter").la_tdcf_values).mean == pytest.approx(
        0.25528
    )
    assert summarize_tdcf(provenance.row("mimo_full").la_tdcf_values).mean == pytest.approx(0.34956)
    assert summarize_tdcf(provenance.row("mimo_adapter").la_tdcf_values).mean == pytest.approx(
        0.29725
    )


def test_mimo_frozen_requires_explicit_tdcf_seed_set_note():
    provenance = load_main_table_provenance(PROVENANCE)
    row = provenance.row("mimo_frozen")

    assert len(row.la_eer_values) == 5
    assert len(row.la_tdcf_values) == 4
    assert row.tdcf_note == "n=4 because seed1234 score file is missing"
    assert row.tdcf_marker == "*"


def test_mimo_adapter_is_explicitly_exploratory():
    row = load_main_table_provenance(PROVENANCE).row("mimo_adapter")

    assert row.exploratory is True
    assert row.status == "exploratory"
    assert len(row.la_eer_values) == 2


def test_rendered_table_matches_corrected_assessment_values():
    table = render_main_table(load_main_table_provenance(PROVENANCE))

    assert "| wav2vec2 | Frozen | 5 local | 8.05 ± 0.73 | 0.384 | 6.76 ± 0.61 | partial |" in table
    assert (
        "| wav2vec2 | Adapter | 5 = 4 local + seed1234 external run | 2.77 ± 0.81 | 0.255 | 5.11 ± 0.72 | partial |"
        in table
    )
    assert (
        "| wav2vec2 | Full FT | 3 local reproduced | 1.09 ± 0.06 | 0.215 | 4.41 ± 1.26 | partial |"
        in table
    )
    assert (
        "| MiMo | Frozen | 5 found evals | 7.11 ± 2.35 | 0.361* | 12.86 ± 1.18 | invalid as paper row |"
        in table
    )
    assert (
        "| MiMo | Adapter | 2 found evals | 4.39 ± 0.02 | 0.297 | 9.71 ± 2.84 | exploratory |"
        in table
    )
    assert (
        "| MiMo | Full FT | 5 = 4 local + seed1234 external run | 6.94 ± 2.02 | 0.350 | 12.74 ± 1.18 | partial |"
        in table
    )


def test_seed_set_mismatch_without_note_fails(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
version: 1
rows:
  - id: bad
    model: X
    strategy: Y
    n_source: 2 local
    status: partial
    notes: bad missing footnote
    seeds:
      - id: 1
        source: local
        status: partial
        metrics: {la_eer: 1.0, df_eer: 2.0, la_tdcf: 0.1}
      - id: 2
        source: local
        status: partial
        metrics: {la_eer: 3.0, df_eer: 4.0}
""".strip()
    )

    with pytest.raises(ProvenanceError, match="tDCF seed count differs"):
        load_main_table_provenance(path)
