from __future__ import annotations

import json
from pathlib import Path
import warnings

import matplotlib.image as mpimg
import numpy as np
import pytest

from experiments.pu_synthesizability_20260821.clscore_formula_relationship_plots import (
    CLSCORES,
    FORMULAS,
    _equal_count_trend,
    _sha256,
    plot_decile_relationships,
    plot_direct_relationships,
)


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DATA = (
    ROOT
    / "outputs"
    / "20260822_pu_formula_scores"
    / "full_pool_dual_v2"
    / "direct_formula_plots_v3"
)


def test_equal_count_trend_preserves_order_and_quantiles() -> None:
    x = np.linspace(0.0, 1.0, 100)
    y = 2.0 * x - 0.5
    x_median, y_q25, y_median, y_q75, y_mean = _equal_count_trend(
        x, y, bins=10
    )

    assert len(x_median) == 10
    assert np.all(np.diff(x_median) > 0)
    assert np.all(np.diff(y_median) > 0)
    assert np.all(y_q25 <= y_median)
    assert np.all(y_median <= y_q75)
    assert np.all(np.diff(y_mean) > 0)


def test_frozen_direct_plot_data_matches_official_correlations() -> None:
    data_path = PRODUCTION_DATA / "clscore_formula_density_data.npz"
    summary_path = PRODUCTION_DATA / "clscore_formula_density_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["outputs"]["density_data_sha256"] == _sha256(data_path)
    assert summary["formula_input_rows"] == 8_125_976
    assert summary["common_score_rows"] == 8_108_676
    assert summary["trend_definition"] == "10 equal-record CLscore bins"
    assert summary["decile_plot_statistic"] == "arithmetic mean formula score"
    expected = {
        "S_syn": (
            430_609,
            {
                "CLscore_A": 0.507561,
                "CLscore_B": 0.559405,
                "CLscore_jang": -0.124899,
            },
        ),
        "S_stab": (
            978_071,
            {
                "CLscore_A": 0.296452,
                "CLscore_B": 0.349433,
                "CLscore_jang": 0.002480,
            },
        ),
    }
    for formula, (n, correlations) in expected.items():
        assert summary["formulas"][formula]["n"] == n
        for clscore, rho in correlations.items():
            row = summary["formulas"][formula]["clscores"][clscore]
            assert row["n"] == n
            assert row["spearman_rho"] == pytest.approx(rho, abs=5e-7)
            assert row["histogram_count"] / n == pytest.approx(0.98, abs=1e-4)


def test_direct_relationship_plot_writes_two_pngs_and_two_pdfs(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(20260822)
    x_edges = np.linspace(0.0, 1.0, 41)
    arrays: dict[str, np.ndarray] = {"x_edges": x_edges}
    summary: dict[str, object] = {
        "schema_version": 1,
        "trend_definition": "10 equal-record CLscore bins",
        "decile_plot_statistic": "arithmetic mean formula score",
        "formulas": {},
        "outputs": {},
    }
    for formula_index, formula in enumerate(FORMULAS):
        y_edges = np.linspace(-8.0 + formula_index, 3.0 + formula_index, 41)
        arrays[f"{formula}__y_edges"] = y_edges
        formula_summary = {"n": 10_000, "clscores": {}}
        for score_index, clscore in enumerate(CLSCORES):
            histogram = rng.poisson(
                lam=2.0 + formula_index + score_index,
                size=(40, 40),
            ).astype(np.int64)
            prefix = f"{formula}__{clscore}"
            arrays[f"{prefix}__histogram"] = histogram
            arrays[f"{prefix}__x_median"] = np.linspace(0.05, 0.95, 10)
            center = np.linspace(-4.0, 0.5, 10) + 0.2 * score_index
            arrays[f"{prefix}__y_q25"] = center - 0.7
            arrays[f"{prefix}__y_median"] = center
            arrays[f"{prefix}__y_mean"] = center + 0.1
            arrays[f"{prefix}__y_q75"] = center + 0.7
            formula_summary["clscores"][clscore] = {
                "n": 10_000 + 100 * score_index,
                "spearman_rho": 0.50 - 0.15 * score_index,
            }
        summary["formulas"][formula] = formula_summary

    data_path = tmp_path / "data.npz"
    with data_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    summary["outputs"] = {"density_data_sha256": _sha256(data_path)}
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    output_dir = tmp_path / "figures"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        paths = plot_direct_relationships(
            data_path=data_path,
            summary_path=summary_path,
            output_dir=output_dir,
        )

    glyph_warnings = [
        str(item.message)
        for item in caught
        if "missing from current font" in str(item.message)
    ]
    assert glyph_warnings == []
    assert {path.name for path in paths} == {
        "clscore_vs_synthesis_formula.png",
        "clscore_vs_synthesis_formula.pdf",
        "clscore_vs_stability_formula.png",
        "clscore_vs_stability_formula.pdf",
    }
    for path in paths:
        assert path.stat().st_size > 15_000
    for path in output_dir.glob("*.png"):
        image = mpimg.imread(path)
        assert image.shape[0] >= 1_200
        assert image.shape[1] >= 3_000
        assert float(image[..., :3].std()) > 0.10
    for path in output_dir.glob("*.pdf"):
        assert path.read_bytes().startswith(b"%PDF")

    decile_output = tmp_path / "decile_figures"
    decile_paths = plot_decile_relationships(
        data_path=data_path,
        summary_path=summary_path,
        output_dir=decile_output,
    )
    assert {path.name for path in decile_paths} == {
        "clscore_vs_synthesis_formula_deciles.png",
        "clscore_vs_synthesis_formula_deciles.pdf",
        "clscore_vs_stability_formula_deciles.png",
        "clscore_vs_stability_formula_deciles.pdf",
    }
    for path in decile_paths:
        assert path.stat().st_size > 15_000
