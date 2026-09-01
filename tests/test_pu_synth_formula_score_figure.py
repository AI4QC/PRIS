from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.image as mpimg
import pytest

from experiments.pu_synthesizability_20260821.plot_formula_score_figure import (
    build_formula_score_figure,
    load_formula_score_figure_data,
)


ROOT = Path(__file__).resolve().parents[1]
BINARY_ANALYSIS = (
    ROOT
    / "outputs"
    / "20260822_pu_formula_scores"
    / "binary_dual_v1"
    / "analysis_sensitivity_v2"
)
FULL_ANALYSIS = (
    ROOT
    / "outputs"
    / "20260822_pu_formula_scores"
    / "full_pool_dual_v2"
    / "analysis_v2"
)


def test_formula_score_figure_data_matches_frozen_aggregates() -> None:
    data = load_formula_score_figure_data(BINARY_ANALYSIS, FULL_ANALYSIS)

    screening = data["binary_raw"].set_index("score_variant")
    assert screening.loc["S_syn", "auroc"] == pytest.approx(0.883513, abs=1e-6)
    assert screening.loc[
        "S_syn", "pu_screened_at_95pct_experimental"
    ] == pytest.approx(0.203356, abs=1e-6)
    assert screening.loc[
        "S_syn_no_D7_D8_madz", "pu_screened_at_95pct_experimental"
    ] == pytest.approx(0.358576, abs=1e-6)
    assert screening.loc["S_stab", "auroc"] == pytest.approx(0.707402, abs=1e-6)
    assert set(screening.experimental_preserved_denominator) == {99_162}
    assert set(screening.pu_screened_denominator_at_95pct_experimental) == {
        364_592
    }

    coverage = data["complete_case_coverage"].set_index(["formula", "cohort"])
    assert coverage.loc[("S_syn", "experimental"), "n_complete"] == 21_477
    assert coverage.loc[("S_syn", "pu_negative"), "n_complete"] == 135
    assert coverage.loc[("S_syn", "full_pool"), "n_complete"] == 430_609
    assert coverage.loc[("S_stab", "experimental"), "fraction"] == pytest.approx(
        28_288 / 99_162
    )
    assert coverage.loc[("S_stab", "pu_negative"), "fraction"] == pytest.approx(
        4_750 / 364_592
    )
    assert coverage.loc[("S_stab", "full_pool"), "fraction"] == pytest.approx(
        978_071 / 8_108_676
    )

    correlations = data["complete_case_correlations"].set_index(
        ["score_variant", "clscore"]
    )
    assert correlations.loc[("S_syn", "CLscore_A"), "spearman_rho"] == pytest.approx(
        0.507561, abs=1e-6
    )
    assert correlations.loc[
        ("S_syn", "CLscore_jang"), "spearman_rho"
    ] == pytest.approx(-0.124899, abs=1e-6)
    assert correlations.loc[("S_stab", "CLscore_B"), "spearman_rho"] == pytest.approx(
        0.349433, abs=1e-6
    )

    pris = data["complete_case_pris"].set_index(
        ["score_variant", "pris_outcome"]
    )
    assert pris.loc[
        ("S_syn", "D7"), "auroc_high_score_predicts_pass"
    ] == pytest.approx(0.597904, abs=1e-6)
    assert pris.loc[
        ("S_syn_no_D7_D8", "D7"), "auroc_high_score_predicts_pass"
    ] == pytest.approx(0.474226, abs=1e-6)
    assert pris.loc[
        ("S_stab", "L4"), "auroc_high_score_predicts_pass"
    ] == pytest.approx(0.767966, abs=1e-6)
    assert pris.loc[("S_syn", "L4"), "n_no_verdict"] == 41_712
    assert data["pris_no_verdict_policy"] == (
        "reported_in_denominator_and_excluded_from_binary_metrics"
    )


def test_formula_score_figure_writes_readable_png_and_pdf(tmp_path: Path) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        paths = build_formula_score_figure(BINARY_ANALYSIS, FULL_ANALYSIS, tmp_path)

    glyph_warnings = [
        str(item.message) for item in caught if "missing from current font" in str(item.message)
    ]
    assert glyph_warnings == []

    assert {path.name for path in paths} == {
        "pris_pu_formula_scores.png",
        "pris_pu_formula_scores.pdf",
    }
    for path in paths:
        assert path.stat().st_size > 25_000

    png = tmp_path / "pris_pu_formula_scores.png"
    image = mpimg.imread(png)
    assert image.shape[0] >= 2_400
    assert image.shape[1] >= 3_400
    assert float(image[..., :3].std()) > 0.08

    pdf = tmp_path / "pris_pu_formula_scores.pdf"
    assert pdf.read_bytes().startswith(b"%PDF")
