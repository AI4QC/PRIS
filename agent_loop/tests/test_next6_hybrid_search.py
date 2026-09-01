import numpy as np
import pandas as pd
import pytest

from src.next6_hybrid_search import build_hybrid_features, hybrid_formula


pytestmark = pytest.mark.filterwarnings("error")


def test_hybrid_score_is_keyed_shuffle_invariant_and_positive_physics_correction():
    prepared = pd.DataFrame(
        {
            "sid": ["a", "b", "c"],
            "rk": ["g", "g", "g"],
            "strict_x0_ok": [True, True, True],
            "support_geom": [True, True, True],
            "support_ewald": [True, True, True],
            "support_p2": [True, True, True],
            "geom_packing_fraction": [1.0, 2.0, 3.0],
            "ewald_per_atom": [3.0, 2.0, 1.0],
            "p2_mean_dev": [3.0, 2.0, 1.0],
        }
    )
    mattersim = pd.DataFrame(
        {
            "sid": ["c", "a", "b"],
            "rk": ["g", "g", "g"],
            "mattersim_feature_ok": [True, True, True],
            "mattersim_energy_per_atom": [-1.0, -2.0, -1.8],
        }
    )
    first = build_hybrid_features(prepared, mattersim).set_index("sid")
    shuffled = build_hybrid_features(
        prepared.sample(frac=1, random_state=2),
        mattersim.sample(frac=1, random_state=3),
    ).set_index("sid")

    assert first.hybrid_physics_rank.to_dict() == pytest.approx(
        {"a": 1.0, "b": 0.5, "c": 0.0}
    )
    assert first.mattersim_predicted_gap.to_dict() == pytest.approx(
        {"a": 0.0, "b": 0.2, "c": 1.0}
    )
    assert first.hybrid_gap_p005.to_dict() == pytest.approx(
        {"a": 0.05, "b": 0.225, "c": 1.0}
    )
    pd.testing.assert_series_equal(
        first.sort_index().hybrid_gap_p005,
        shuffled.sort_index().hybrid_gap_p005,
        check_names=False,
    )
    assert first.support_hybrid.tolist() == [True, True, True]
    spec = hybrid_formula()
    assert spec.track == "cohort_margin"
    assert spec.terms[0].weight > 0


def test_hybrid_abstains_if_any_required_physics_or_mlip_term_is_missing():
    prepared = pd.DataFrame(
        {
            "sid": ["a", "b"],
            "rk": ["g", "g"],
            "strict_x0_ok": [True, True],
            "support_geom": [True, True],
            "support_ewald": [True, False],
            "support_p2": [True, True],
            "geom_packing_fraction": [1.0, 2.0],
            "ewald_per_atom": [2.0, 1.0],
            "p2_mean_dev": [2.0, 1.0],
        }
    )
    mattersim = pd.DataFrame(
        {
            "sid": ["a", "b"],
            "rk": ["g", "g"],
            "mattersim_feature_ok": [True, True],
            "mattersim_energy_per_atom": [-2.0, -1.0],
        }
    )
    got = build_hybrid_features(prepared, mattersim).set_index("sid")
    assert not bool(got.loc["a", "support_hybrid"])
    assert not bool(got.loc["b", "support_hybrid"])
    assert np.isnan(got.loc["a", "hybrid_gap_p005"])
