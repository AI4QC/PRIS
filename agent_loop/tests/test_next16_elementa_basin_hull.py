"""Contracts for the frozen NEXT16 ELEMENTA Basin-Hull execution."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
import numpy as np
import pytest


def test_next16_boundary_is_frozen_at_the_wbm_development_candidate() -> None:
    from src.next16_elementa_basin_hull import NEXT16_THRESHOLD_EV_PER_ATOM, next16_decision

    assert NEXT16_THRESHOLD_EV_PER_ATOM == 0.15
    assert next16_decision(0.149999, supported=True) == "KEEP"
    assert next16_decision(0.15, supported=True) == "REJECT"
    assert next16_decision(np.nan, supported=False) == "ABSTAIN"
    with pytest.raises(ValueError, match="finite"):
        next16_decision(np.nan, supported=True)


def test_pauling_control_uses_all_four_frozen_operational_rules() -> None:
    from src.next16_elementa_basin_hull import pauling_control

    atoms = Atoms("LiO", positions=[[0, 0, 0], [1, 1, 1]], cell=[4, 4, 4], pbc=True)

    def values(_atoms):
        return {
            "p2_mean_dev": 0.005,
            "p3_frac_edge_face": 0.20,
            "p4_violate": 0.0,
            "p5_ok": 1.0,
        }, None

    result = pauling_control(atoms, calculator=values)
    assert result["pauling_p2_decision"] == "KEEP"
    assert result["pauling_p3_decision"] == "REJECT"
    assert result["pauling_p4_decision"] == "KEEP"
    assert result["pauling_p5_decision"] == "KEEP"
    assert result["pauling_p2_p5_decision"] == "REJECT"


def test_cli_cannot_change_threshold_relaxation_or_select_rows() -> None:
    from src.next16_elementa_basin_hull import main

    for forbidden in ("--threshold", "--fmax", "--max-steps", "--sids", "--labels"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
