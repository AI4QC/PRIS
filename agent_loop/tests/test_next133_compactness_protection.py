from __future__ import annotations

import math

import pandas as pd
import pytest

from src.next133_compactness_protection import (
    PACKING_CENTER,
    VOLUME_CENTER,
    compute_covalent_packing_protection,
    compute_low_volume_protection,
    materialize_compactness_protection,
)


def test_protection_definitions_are_zero_at_frozen_centers_and_directional() -> None:
    volume_center_raw = math.exp(VOLUME_CENTER)
    packing_center_raw = math.expm1(PACKING_CENTER)
    assert compute_low_volume_protection(volume_center_raw) == pytest.approx(0.0)
    assert compute_low_volume_protection(volume_center_raw / 2.0) > 0.0
    assert compute_covalent_packing_protection(packing_center_raw) == pytest.approx(0.0)
    assert compute_covalent_packing_protection(packing_center_raw + 0.5) > 0.0


def test_invalid_raw_inputs_fail_open_independently() -> None:
    assert compute_low_volume_protection(0.0) is None
    assert compute_low_volume_protection(float("nan")) is None
    assert compute_covalent_packing_protection(-1.0) is None
    table = pd.DataFrame(
        {
            "material_id": ["a", "b"],
            "geom_volume_pa": [10.0, float("nan")],
            "geom_covalent_packing": [float("nan"), 1.0],
        }
    )
    result = materialize_compactness_protection(table)
    assert result["low_volume_protection_supported"].tolist() == [True, False]
    assert result["covalent_packing_protection_supported"].tolist() == [False, True]
