"""Contracts for identity-locked NEXT27 endpoint assembly."""

from __future__ import annotations

import pandas as pd
import pytest


def test_select_expected_endpoints_requires_exact_identity_coverage() -> None:
    from src.next27_build_endpoints import select_expected_endpoints

    endpoints = pd.DataFrame(
        {
            "material_id": ["a", "b", "excluded"],
            "force0_max": [0.1, 1.1, 9.0],
        }
    )
    expected = pd.DataFrame({"material_id": ["b", "a"]})
    selected = select_expected_endpoints(endpoints, expected, source_shard="data0001")
    assert selected.material_id.tolist() == ["a", "b"]
    assert selected.source_shard.tolist() == ["data0001", "data0001"]

    with pytest.raises(ValueError, match="missing"):
        select_expected_endpoints(
            endpoints, pd.DataFrame({"material_id": ["a", "missing"]}), source_shard="data0001"
        )
