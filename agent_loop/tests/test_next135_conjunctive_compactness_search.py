from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.next135_conjunctive_compactness_search import (
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_CONFIGURATION_COUNT,
    PACKING_PRODUCT_FEATURE,
    VOLUME_PRODUCT_FEATURE,
    build_conjunctive_configurations,
    materialize_conjunctive_features,
)


def test_conjunctive_configuration_counts_are_frozen() -> None:
    configurations = build_conjunctive_configurations()
    assert len(configurations) == EXPECTED_CONFIGURATION_COUNT == 49
    assert EXPECTED_CANDIDATE_COUNT == 539


def test_products_require_both_operands_and_are_normalized() -> None:
    table = pd.DataFrame(
        {
            "coordination_protection": [0.9209581129860017, 0.5, np.nan],
            "coordination_protection_supported": [True, True, False],
            "covalent_packing_protection": [1.9773347262377292, np.nan, 1.0],
            "covalent_packing_protection_supported": [True, False, True],
            "low_volume_protection": [1.5310711399624055, 0.5, 1.0],
            "low_volume_protection_supported": [True, True, True],
        }
    )
    result = materialize_conjunctive_features(table)
    assert result[PACKING_PRODUCT_FEATURE].iloc[0] == pytest.approx(1.0)
    assert result[VOLUME_PRODUCT_FEATURE].iloc[0] == pytest.approx(1.0)
    assert result["coordination_covalent_packing_product_supported"].tolist() == [True, False, False]
    assert result["coordination_low_volume_product_supported"].tolist() == [True, True, False]
