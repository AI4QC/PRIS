from __future__ import annotations

import inspect
import math

import numpy as np
import pandas as pd
import pytest

from src.next539_odac23_pbaaa_label_free_features import (
    FEATURE_NAME,
    build_odac23_pbaaa_features,
    role_gate_statistics,
)


def _table(unique: int = 20) -> pd.DataFrame:
    rows = []
    for role in ("discovery", "internal_validation", "internal_replication"):
        rows.extend(
            {
                "partition_role": role,
                FEATURE_NAME: index / 100.0,
                "pbaaa_supported": True,
                "pbaaa_failure": None,
            }
            for index in range(unique)
        )
    return pd.DataFrame(rows)


def test_role_gate_accepts_exact_nondegeneracy_boundary() -> None:
    statistics = role_gate_statistics(_table())
    assert statistics["passes"] is True
    for role in ("discovery", "internal_validation", "internal_replication"):
        assert statistics["partitions"][role]["finite_unique_rounded_8"] == 20


def test_role_gate_fails_closed_on_support_semantics_or_degeneracy() -> None:
    semantics = _table()
    semantics.loc[0, FEATURE_NAME] = math.nan
    with pytest.raises(ValueError, match="support semantics differ"):
        role_gate_statistics(semantics)
    degenerate = _table(unique=19)
    assert role_gate_statistics(degenerate)["passes"] is False


def test_formal_builder_interface_has_no_endpoint_or_label_argument() -> None:
    parameters = set(inspect.signature(build_odac23_pbaaa_features).parameters)
    assert parameters == {
        "next54_dir",
        "next60_firewall_path",
        "next538_dir",
        "design_path",
        "output_dir",
        "workers",
        "require_formal_inputs",
    }


def test_role_gate_rejects_nonfinite_unsupported_value_only_when_marked_supported() -> None:
    table = _table()
    table.loc[0, FEATURE_NAME] = np.nan
    table.loc[0, "pbaaa_supported"] = False
    table.loc[0, "pbaaa_failure"] = "unsupported"
    statistics = role_gate_statistics(table)
    assert statistics["partitions"]["discovery"]["unsupported"] == 1
