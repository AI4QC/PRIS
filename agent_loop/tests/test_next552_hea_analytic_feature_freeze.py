from __future__ import annotations

import numpy as np
import pandas as pd

from src.next552_hea_analytic_feature_freeze import add_frozen_risk_percentiles


def test_risk_percentiles_are_full_cohort_midranks_with_opposed_directions() -> None:
    table = pd.DataFrame(
        {
            "fid": ["a", "b", "c", "d"],
            "partition": ["development", "validation", "development", "validation"],
            "feature": [1.0, 2.0, 2.0, 4.0],
        }
    )

    result = add_frozen_risk_percentiles(table, ("feature",))

    np.testing.assert_allclose(result["feature__risk_high"], [0.125, 0.5, 0.5, 0.875])
    np.testing.assert_allclose(result["feature__risk_low"], [0.875, 0.5, 0.5, 0.125])


def test_risk_percentiles_preserve_missing_support() -> None:
    table = pd.DataFrame(
        {
            "fid": ["a", "b", "c"],
            "partition": ["development", "development", "validation"],
            "feature": [1.0, np.nan, 3.0],
        }
    )

    result = add_frozen_risk_percentiles(table, ("feature",))

    np.testing.assert_allclose(result.loc[[0, 2], "feature__risk_high"], [0.25, 0.75])
    assert np.isnan(result.loc[1, "feature__risk_high"])

