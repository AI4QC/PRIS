from __future__ import annotations

import numpy as np

from src.next565_hea_mechanism_formula_family import candidate_scores


def test_candidate_family_is_fixed_and_coefficient_free() -> None:
    scores = candidate_scores([0.2, 0.8], [0.5, 0.4], [0.1, 0.9])

    assert set(scores) == {"MEMAX", "MEPU24", "ZEPU24"}
    np.testing.assert_allclose(scores["MEMAX"], [0.5, 0.8])
    np.testing.assert_allclose(
        scores["MEPU24"], 1.0 - (1.0 - np.array([0.2, 0.8]) ** 2)
        * (1.0 - np.array([0.5, 0.4]) ** 4)
    )
