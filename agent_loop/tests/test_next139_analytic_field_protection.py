import math

import numpy as np
import pandas as pd

import src.next139_analytic_field_protection as n139


def test_materialize_analytic_field_protection_is_low_field_and_fail_open() -> None:
    at_center = math.expm1(n139.CENTER)
    table = pd.DataFrame(
        {
            "material_id": ["low", "center", "high", "missing"],
            n139.RAW_FEATURE: [0.0, at_center, at_center * 10.0, np.nan],
            n139.RAW_SUPPORT: [True, True, True, False],
        }
    )

    result = n139.materialize_analytic_field_protection(table)

    assert result[n139.SUPPORT_COLUMN].tolist() == [True, True, True, False]
    assert math.isclose(result[n139.FEATURE_NAME].iloc[0], n139.CLIP_NORMALIZED)
    assert math.isclose(result[n139.FEATURE_NAME].iloc[1], 0.0, abs_tol=1e-12)
    assert math.isclose(result[n139.FEATURE_NAME].iloc[2], 0.0, abs_tol=1e-12)
    assert np.isnan(result[n139.FEATURE_NAME].iloc[3])
