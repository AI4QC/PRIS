from __future__ import annotations

import math

import pandas as pd
import pytest

import src.next533_odac23_sssp_label_free_features as n533


def test_role_gate_requires_nondegenerate_values_in_every_partition() -> None:
    rows = []
    for role in n533.PARTITIONS:
        for index in range(25):
            rows.append(
                {
                    "partition_role": role,
                    "sssp_same_sign_shell_purity_q10": index / 25,
                    "sssp_supported": True,
                    "sssp_failure": None,
                }
            )
    result = n533.role_gate_statistics(pd.DataFrame(rows))
    assert result["passes"] is True
    assert all(result["partitions"][role]["finite_unique_rounded_10"] == 25 for role in n533.PARTITIONS)


def test_role_gate_reports_unsupported_rows_and_fails_degenerate_role() -> None:
    rows = []
    for role in n533.PARTITIONS:
        for index in range(25):
            supported = role != "internal_replication" or index == 0
            rows.append(
                {
                    "partition_role": role,
                    "sssp_same_sign_shell_purity_q10": index / 25 if supported else math.nan,
                    "sssp_supported": supported,
                    "sssp_failure": None if supported else "unsupported",
                }
            )
    result = n533.role_gate_statistics(pd.DataFrame(rows))
    assert result["passes"] is False
    assert result["partitions"]["internal_replication"]["unsupported"] == 24


def test_role_gate_fails_closed_on_support_semantics_or_role_drift() -> None:
    frame = pd.DataFrame(
        {
            "partition_role": ["discovery"],
            "sssp_same_sign_shell_purity_q10": [math.nan],
            "sssp_supported": [True],
            "sssp_failure": [None],
        }
    )
    with pytest.raises(ValueError, match="support semantics differ"):
        n533.role_gate_statistics(frame)
    frame["sssp_supported"] = False
    with pytest.raises(ValueError, match="partition roles differ"):
        n533.role_gate_statistics(frame)


def test_boundary_excludes_endpoints_and_relaxed_or_learned_inputs() -> None:
    assert n533.EXECUTABLE_INPUT_BOUNDARY == (
        "composition", "one raw initial fully periodic geometry"
    )
    assert n533.BOUNDARY_FLAGS["endpoint_values_opened"] is False
    assert n533.BOUNDARY_FLAGS["relaxed_structures_opened"] is False
    assert n533.BOUNDARY_FLAGS["dft_values_used_by_features"] is False
    assert n533.BOUNDARY_FLAGS["model_or_proxy_potential_used"] is False
