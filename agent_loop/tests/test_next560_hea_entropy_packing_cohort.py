from __future__ import annotations

import numpy as np

from src.next560_hea_entropy_packing_cohort import _midrank


def test_midrank_is_bounded_and_reverses_packing_risk() -> None:
    np.testing.assert_allclose(_midrank([1.0, 2.0, 2.0]), [1 / 6, 2 / 3, 2 / 3])
    np.testing.assert_allclose(_midrank([1.0, 2.0, 2.0], reverse=True), [5 / 6, 1 / 3, 1 / 3])
