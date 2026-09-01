import numpy as np
import pandas as pd

import src.next143_coulomb_steric_balance_protection as n143


def test_balance_protection_is_one_minus_residual_and_fail_open() -> None:
    table = pd.DataFrame(
        {
            "material_id": ["balanced", "partial", "unbalanced", "missing"],
            n143.RAW_FEATURE: [0.0, 0.25, 1.0, np.nan],
            n143.RAW_SUPPORT: [True, True, True, False],
        }
    )
    result = n143.materialize_balance_protection(table)
    assert np.allclose(result[n143.FEATURE_NAME].iloc[:3], [1.0, 0.75, 0.0])
    assert np.isnan(result[n143.FEATURE_NAME].iloc[3])
    assert result[n143.SUPPORT_COLUMN].tolist() == [True, True, True, False]
