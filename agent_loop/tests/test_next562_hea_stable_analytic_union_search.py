from __future__ import annotations

import numpy as np

from src.next562_hea_stable_analytic_union_search import (
    COMPOSITION_FEATURE_NAMES,
    combine_risks,
    composition_features,
)


def test_composition_features_are_formula_only_and_complete() -> None:
    values = composition_features("Fe2Ni2")

    assert set(values) == set(COMPOSITION_FEATURE_NAMES)
    assert values["composition_ideal_entropy"] == np.log(2.0)
    assert values["composition_element_count"] == 2.0
    assert all(np.isfinite(list(values.values())))


def test_coefficient_free_risk_combinations() -> None:
    a = np.array([0.2, 0.8])
    b = np.array([0.5, 0.4])

    np.testing.assert_allclose(combine_risks([a, b], "mean"), [0.35, 0.6])
    np.testing.assert_allclose(combine_risks([a, b], "maximum"), [0.5, 0.8])
    np.testing.assert_allclose(combine_risks([a, b], "minimum"), [0.2, 0.4])
    np.testing.assert_allclose(combine_risks([a, b], "union"), [0.6, 0.88])
