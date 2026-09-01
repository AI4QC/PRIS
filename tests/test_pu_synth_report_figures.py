from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import pytest

from experiments.pu_synthesizability_20260821.plot_report_figures import (
    build_report_figures,
    load_binary_figure_data,
    load_full_pool_figure_data,
)


ROOT = Path(__file__).resolve().parents[1]
BINARY_ANALYSIS = (
    ROOT / "outputs" / "20260821_pu_synthesizability" / "analysis_v1"
)
FULL_POOL_ANALYSIS = (
    ROOT
    / "outputs"
    / "20260821_pu_synthesizability"
    / "full_pool_analysis_v1"
)


def test_binary_figure_data_matches_frozen_analysis() -> None:
    data = load_binary_figure_data(BINARY_ANALYSIS)

    assert data["cohort_totals"] == {
        "experimental": 99_162,
        "pu_negative": 364_592,
    }
    assert data["l4_states"]["experimental"] == {
        "pass": 29_952,
        "explicit_violation": 19_153,
        "no_verdict": 50_057,
    }
    assert data["l4_states"]["pu_negative"] == {
        "pass": 279,
        "explicit_violation": 189_159,
        "no_verdict": 175_154,
    }
    assert data["screening_rates"]["L4"] == pytest.approx(
        [0.1931485851, 0.5188237811]
    )
    assert data["pu_mechanism_rates"] == pytest.approx(
        {"L4": 0.5188237811, "D7": 0.5121920390, "0.5 Å": 0.0, "0.7 Å": 0.0}
    )
    assert data["d7_fraction_of_l4_exclusions"] == pytest.approx(0.9872171031)


def test_full_pool_figure_data_preserves_pooled_and_source_specific_trends() -> None:
    data = load_full_pool_figure_data(FULL_POOL_ANALYSIS)

    pooled = data["pooled_deciles"]
    assert len(pooled) == 30
    assert data["common_support_n"] == 8_108_676
    assert data["pool_rows"] == 8_125_976

    endpoints = data["source_endpoints"]
    assert endpoints[("lemat", "CLscore_A")] == pytest.approx(
        (0.708870, 0.263341), abs=5e-7
    )
    assert endpoints[("elementa", "CLscore_A")] == pytest.approx(
        (0.195607, 0.404410), abs=5e-7
    )
    assert endpoints[("lemat", "CLscore_jang")] == pytest.approx(
        (0.648659, 0.298789), abs=5e-7
    )
    assert endpoints[("elementa", "CLscore_jang")] == pytest.approx(
        (0.271493, 0.313330), abs=5e-7
    )

    assert data["stratified_l4_rho"] == pytest.approx(
        {"CLscore_A": 0.033392, "CLscore_B": 0.055623, "CLscore_jang": 0.042850},
        abs=5e-7,
    )
    assert data["stratified_d7_rho"] == pytest.approx(
        {"CLscore_A": -0.021868, "CLscore_B": 0.007573, "CLscore_jang": 0.002308},
        abs=5e-7,
    )


def test_build_report_figures_writes_readable_png_and_pdf(tmp_path: Path) -> None:
    paths = build_report_figures(BINARY_ANALYSIS, FULL_POOL_ANALYSIS, tmp_path)

    assert {path.name for path in paths} == {
        "pris_pu_task1_binary.png",
        "pris_pu_task1_binary.pdf",
        "pris_pu_task2_fullpool.png",
        "pris_pu_task2_fullpool.pdf",
    }
    for path in paths:
        assert path.stat().st_size > 10_000

    for png in tmp_path.glob("*.png"):
        image = mpimg.imread(png)
        assert image.shape[0] >= 1_400
        assert image.shape[1] >= 2_400

    for pdf in tmp_path.glob("*.pdf"):
        assert pdf.read_bytes().startswith(b"%PDF")
