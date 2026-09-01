"""Contracts for nested three-scale ACSC direct confirmation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from ase import Atoms


def _atoms() -> Atoms:
    return Atoms(
        "H2",
        scaled_positions=[[0.15, 0.25, 0.35], [0.62, 0.71, 0.83]],
        cell=[[8.0, 0.0, 0.0], [0.4, 9.0, 0.0], [0.2, 0.3, 10.0]],
        pbc=True,
    )


@pytest.mark.parametrize(
    ("small", "large", "expected"),
    [
        (True, "resolved_negative", True),
        (False, "resolved_negative", False),
        (True, "resolved_nonnegative", False),
        (True, "near_zero_or_inconsistent", False),
    ],
)
def test_nested_confirmation_requires_both_direct_pairs_negative(
    small: bool, large: str, expected: bool
) -> None:
    from src.next13c_acsc_direct_ladder import nested_confirmation

    assert nested_confirmation(small, large) is expected


def test_large_probe_group_is_center_then_h7_and_h8() -> None:
    from src.next13b_acsc_direct import mixed_mode_probe
    from src.next13c_acsc_direct_ladder import LARGE_STEP, ladder_probe_group

    mode = np.zeros(9)
    mode[0] = 3.0 / 5.0
    mode[3] = 4.0 / 5.0
    observed = ladder_probe_group(_atoms(), mode)
    expected = (
        _atoms(),
        mixed_mode_probe(_atoms(), mode, amplitude=LARGE_STEP),
        mixed_mode_probe(_atoms(), mode, amplitude=-LARGE_STEP),
        mixed_mode_probe(_atoms(), mode, amplitude=LARGE_STEP / 2.0),
        mixed_mode_probe(_atoms(), mode, amplitude=-LARGE_STEP / 2.0),
    )

    assert LARGE_STEP == 2**-7
    assert len(observed) == 5
    for first, second in zip(observed, expected, strict=True):
        np.testing.assert_allclose(first.cell.array, second.cell.array, atol=0.0)
        np.testing.assert_allclose(first.positions, second.positions, atol=2e-15)


def test_sealed_modes_require_unit_vectors_and_exact_direct_candidate_rows() -> None:
    from src.next13c_acsc_direct_ladder import sealed_mode_table

    mode = np.zeros(9)
    mode[0] = 1.0
    table = pd.DataFrame(
        {
            "sid": ["b", "a"],
            "rk": ["rk-b", "rk-a"],
            "natoms": [2, 2],
            "mode_json": [json.dumps(mode.tolist())] * 2,
            "recomputed_coupling_only_negative": [True, True],
            "direct_status": ["resolved_negative", "near_zero_or_inconsistent"],
            "direct_confirmed": [True, False],
            "direct_q_h_ev_per_atom": [-1.0, -0.1],
            "direct_q_h2_ev_per_atom": [-1.0, 0.1],
            "direct_q_r_ev_per_atom": [-1.0, 0.2],
            "direct_e_num_ev_per_atom": [0.0, 0.1],
            "direct_u_num_ev_per_atom": [-1.0, 0.3],
        }
    )

    selected = sealed_mode_table(table)

    assert selected["sid"].tolist() == ["a", "b"]
    assert all(np.asarray(value).shape == (9,) for value in selected["mode_vector"])


def test_cli_exposes_no_label_or_endpoint_argument() -> None:
    from src.next13c_acsc_direct_ladder import main

    for forbidden in ("--labels", "--endpoint", "--dft-results"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
