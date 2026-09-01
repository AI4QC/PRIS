from __future__ import annotations

import numpy as np
import pandas as pd

from src.next555_hea_extreme_waste_search import apply_extreme_waste_endpoint


def test_extreme_waste_endpoint_is_frozen_at_absolute_thresholds() -> None:
    table = pd.DataFrame(
        {
            "fid": ["stable", "energy", "move", "strain", "volume"],
            "e_above_hull": [0.05, 0.40, 0.05, 0.05, 0.05],
            "disp_p90": [0.05, 0.05, 0.25, 0.05, 0.05],
            "cell_logstrain_max": [0.02, 0.02, 0.02, 0.08, 0.02],
            "volume_logchange": [0.02, 0.02, 0.02, 0.02, 0.10],
        }
    )

    result = apply_extreme_waste_endpoint(table)

    assert result["dft_waste"].tolist() == [False, True, True, True, True]
    assert result["protected"].tolist() == [True, False, False, False, False]
    np.testing.assert_allclose(result["waste_severity"], [0.25, 1, 1, 1, 1])
