import numpy as np
import pandas as pd

import src.next148_charge_order_spectrum_protection as n148


def test_materialize_charge_order_protection_is_bounded_and_fail_open() -> None:
    frame = pd.DataFrame(
        {
            "material_id": ["a", "b", "c", "d"],
            n148.RAW_FEATURE: [0.0, 0.25, 1.0, np.nan],
            n148.RAW_SUPPORT: [True, True, True, False],
        }
    )
    result = n148.materialize_charge_order_protection(frame)
    assert np.allclose(result[n148.FEATURE_NAME].iloc[:3], [1.0, 0.75, 0.0])
    assert np.isnan(result[n148.FEATURE_NAME].iloc[3])
    assert result[n148.SUPPORT_COLUMN].tolist() == [True, True, True, False]
