"""Contracts for the sealed ACSC-DIRECT-v0 MatterSim runner."""

from __future__ import annotations

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


def _formal_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sid": ["keep", "candidate"],
            "rk": ["rk-keep", "rk-candidate"],
            "natoms": [2, 2],
            "upstream_phsc_status": ["resolved_nonnegative"] * 2,
            "upstream_chsc_status": ["resolved_nonnegative"] * 2,
            "recomputed_phsc_status": ["resolved_nonnegative"] * 2,
            "recomputed_chsc_status": ["resolved_nonnegative"] * 2,
            "pure_status_drift": [False, False],
            "acsc_status": ["resolved_nonnegative", "resolved_negative"],
            "coupling_only_negative": [False, True],
            "lambda_r_ev_per_atom": [1.0, -1.0],
            "e_num_ev_per_atom": [0.01, 0.02],
            "u_num_ev_per_atom": [1.01, -0.98],
        }
    )


def test_sealed_candidate_selection_uses_only_exact_coupling_negatives() -> None:
    from src.next13b_acsc_direct_mattersim import sealed_candidate_table

    selected = sealed_candidate_table(_formal_table())

    assert selected["sid"].tolist() == ["candidate"]
    assert bool(selected.iloc[0]["coupling_only_negative"]) is True


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("pure_status_drift", True),
        ("recomputed_phsc_status", "near_zero_or_inconsistent"),
        ("recomputed_chsc_status", "resolved_negative"),
        ("acsc_status", "resolved_nonnegative"),
    ],
)
def test_selection_fails_closed_if_true_flag_conflicts_with_sealed_semantics(
    column: str, value: object
) -> None:
    from src.next13b_acsc_direct_mattersim import sealed_candidate_table

    table = _formal_table()
    table.loc[table["sid"] == "candidate", column] = value
    with pytest.raises(ValueError, match="coupling-only"):
        sealed_candidate_table(table)


def test_direct_probe_group_is_center_then_two_frozen_scales() -> None:
    from src.next11_phsc import canonicalize_phsc_geometry
    from src.next13b_acsc_direct import DIRECT_STEP
    from src.next13b_acsc_direct_mattersim import direct_probe_group

    base, _d_star = canonicalize_phsc_geometry(_atoms())
    mode = np.zeros(9)
    mode[0] = 3.0 / 5.0
    mode[3] = 4.0 / 5.0

    probes = direct_probe_group(base, mode)

    assert len(probes) == 5
    np.testing.assert_allclose(probes[0].cell.array, base.cell.array, atol=0.0)
    assert not np.array_equal(probes[1].cell.array, probes[2].cell.array)
    assert not np.array_equal(probes[3].cell.array, probes[4].cell.array)
    # The public builder must use the exact frozen amplitudes. Rebuilding at
    # those amplitudes should be byte-for-byte numerically identical.
    from src.next13b_acsc_direct import mixed_mode_probe

    expected = (
        base,
        mixed_mode_probe(base, mode, amplitude=DIRECT_STEP),
        mixed_mode_probe(base, mode, amplitude=-DIRECT_STEP),
        mixed_mode_probe(base, mode, amplitude=DIRECT_STEP / 2.0),
        mixed_mode_probe(base, mode, amplitude=-DIRECT_STEP / 2.0),
    )
    for observed, reference in zip(probes, expected, strict=True):
        np.testing.assert_allclose(observed.cell.array, reference.cell.array, atol=0.0)
        np.testing.assert_allclose(observed.positions, reference.positions, atol=0.0)


def test_cli_exposes_no_label_or_endpoint_argument() -> None:
    from src.next13b_acsc_direct_mattersim import main

    for forbidden in ("--labels", "--endpoint", "--dft-results"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
