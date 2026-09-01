import pandas as pd
import pytest

from src.next6_elementa_diagnostics import (
    paired_cluster_bootstrap,
    score_energy_correlations,
    suffix_diagnostics,
)


pytestmark = pytest.mark.filterwarnings("error")


def _paired_predictions() -> pd.DataFrame:
    common = {
        "sid": ["a", "b", "c", "d"],
        "rk": ["g1", "g1", "g2", "g2"],
        "delta_e": [0.0, 0.3, 0.0, 0.4],
        "exact_min": [True, False, True, False],
        "near_min": [True, False, True, False],
        "valuable": [True, False, True, False],
        "high_energy": [False, True, False, True],
        "decision_support": [True, True, True, True],
        "alpha": [0.03] * 4,
    }
    baseline = pd.DataFrame(
        {
            **common,
            "score": [0.0, 0.1, 0.0, 0.9],
            "decision": ["KEEP", "KEEP", "KEEP", "REJECT"],
            "formula": ["baseline"] * 4,
        }
    )
    candidate = pd.DataFrame(
        {
            **common,
            "score": [0.0, 0.9, 0.8, 1.0],
            "decision": ["KEEP", "REJECT", "REJECT", "REJECT"],
            "formula": ["candidate"] * 4,
        }
    )
    return pd.concat([baseline, candidate], ignore_index=True)


def test_paired_cluster_bootstrap_uses_groups_and_reports_paired_differences():
    predictions = _paired_predictions()
    got = paired_cluster_bootstrap(
        predictions,
        baseline_formula="baseline",
        candidate_formula="candidate",
        alpha=0.03,
        n_resamples=200,
        seed=7,
    )

    assert got["n_groups"] == 2
    assert got["n_rows"] == 4
    assert got["metrics"]["dft_savings"]["baseline"] == pytest.approx(0.25)
    assert got["metrics"]["dft_savings"]["candidate"] == pytest.approx(0.75)
    assert got["metrics"]["dft_savings"]["difference"] == pytest.approx(0.5)
    assert got["metrics"]["exact_min_retention"]["difference"] == pytest.approx(-0.5)
    assert got["metrics"]["near_min_retention"]["difference"] == pytest.approx(-0.5)
    assert got["metrics"]["valuable_item_recall"]["difference"] == pytest.approx(-0.5)
    assert got["metrics"]["high_energy_removal_recall"]["difference"] == pytest.approx(0.5)
    assert len(got["metrics"]["dft_savings"]["difference_ci_95"]) == 2

    repeated = paired_cluster_bootstrap(
        predictions.sample(frac=1, random_state=3),
        baseline_formula="baseline",
        candidate_formula="candidate",
        alpha=0.03,
        n_resamples=200,
        seed=7,
    )
    assert repeated == got


def test_suffix_diagnostics_join_by_sid_and_expose_generator_order_bias():
    predictions = _paired_predictions()
    labels = pd.DataFrame(
        {
            "sid": ["d", "b", "a", "c"],
            "material": ["X_02", "X_02", "X_01", "X_01"],
        }
    )
    got = suffix_diagnostics(predictions, labels, alpha=0.03)
    baseline = got.query("formula == 'baseline'").set_index("suffix")
    candidate = got.query("formula == 'candidate'").set_index("suffix")

    assert baseline.loc["01", "n"] == 2
    assert baseline.loc["01", "reject_rate"] == 0.0
    assert baseline.loc["02", "reject_rate"] == pytest.approx(0.5)
    assert candidate.loc["01", "reject_rate"] == pytest.approx(0.5)
    assert candidate.loc["02", "reject_rate"] == 1.0
    assert baseline.loc["02", "mean_delta_e"] == pytest.approx(0.35)

    correlations = score_energy_correlations(predictions, labels, alpha=0.03)
    assert set(correlations.formula) == {"baseline", "candidate"}
    assert set(correlations.columns) >= {
        "raw_spearman",
        "suffix_residualized_rank_correlation",
        "n_finite",
    }


def test_paired_bootstrap_rejects_unpaired_sid_sets():
    predictions = _paired_predictions()
    predictions = predictions.loc[
        ~((predictions.formula == "candidate") & (predictions.sid == "d"))
    ]
    with pytest.raises(ValueError, match="sid sets"):
        paired_cluster_bootstrap(
            predictions,
            baseline_formula="baseline",
            candidate_formula="candidate",
            alpha=0.03,
            n_resamples=10,
        )
