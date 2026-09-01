from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from ase import Atoms
from pymatgen.analysis.phase_diagram import PDEntry
from pymatgen.core import Composition

from src.next15_basin_hull import (
    BASIN_HULL_THRESHOLD_EV_PER_ATOM,
    RELAX_ATOM_BUDGET,
    basin_hull_decision,
    build_reference_entries,
    compute_hull_scores,
    needed_element_subspaces,
    validate_relaxed_snapshot,
)


def test_threshold_is_physical_high_energy_boundary_and_strictly_frozen() -> None:
    assert BASIN_HULL_THRESHOLD_EV_PER_ATOM == 0.20
    assert RELAX_ATOM_BUDGET == 512
    assert basin_hull_decision(0.199999, supported=True) == "KEEP"
    assert basin_hull_decision(0.20, supported=True) == "REJECT"
    assert basin_hull_decision(0.25, supported=True) == "REJECT"
    assert basin_hull_decision(np.nan, supported=False) == "ABSTAIN"


def test_decision_refuses_nonfinite_supported_score() -> None:
    with pytest.raises(ValueError, match="finite"):
        basin_hull_decision(np.nan, supported=True)


def test_needed_element_subspaces_contains_every_nonempty_subset() -> None:
    systems = {frozenset({"Li", "O"}), frozenset({"Na", "Cl", "K"})}
    observed = needed_element_subspaces(systems)
    expected: set[frozenset[str]] = set()
    for system in systems:
        for size in range(1, len(system) + 1):
            expected.update(
                frozenset(values) for values in combinations(sorted(system), size)
            )
    assert observed == expected


def test_reference_filter_uses_raw_mp_energy_and_required_subspaces_only() -> None:
    rows = [
        (
            "mp-li",
            {"composition": {"Li": 1.0}, "energy": -1.0, "correction": -9.0},
        ),
        (
            "mp-o",
            {"composition": {"O": 2.0}, "energy": -2.0, "correction": -8.0},
        ),
        (
            "mp-lio",
            {"composition": {"Li": 1.0, "O": 1.0}, "energy": -4.0},
        ),
        (
            "mp-na",
            {"composition": {"Na": 1.0}, "energy": -3.0},
        ),
    ]
    entries = build_reference_entries(
        rows, needed_subspaces={frozenset({"Li"}), frozenset({"O"}), frozenset({"Li", "O"})}
    )
    assert [entry.name for entry in entries] == ["mp-li", "mp-o", "mp-lio"]
    assert [entry.energy for entry in entries] == [-1.0, -2.0, -4.0]


def test_compute_hull_scores_matches_toy_binary_phase_diagram() -> None:
    entries = [
        PDEntry(Composition("Li"), -1.0, name="li"),
        PDEntry(Composition("O"), -2.0, name="o"),
        PDEntry(Composition("LiO"), -4.0, name="lio"),
    ]
    scores = compute_hull_scores(entries, ["LiO", "LiO"], [-4.0, -3.0])
    assert scores == pytest.approx((0.0, 0.5))


def _snapshot() -> tuple[Atoms, Atoms]:
    base = Atoms(
        "LiO",
        positions=[[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]],
        cell=np.eye(3) * 4.0,
        pbc=True,
    )
    final = base.copy()
    final.set_cell(np.eye(3) * 3.8, scale_atoms=True)
    final.info["total_energy"] = -4.0
    final.info["stress"] = np.eye(3)
    final.arrays["forces"] = np.zeros((2, 3))
    return base, final


def test_validate_relaxed_snapshot_returns_deployable_summary() -> None:
    base, final = _snapshot()
    result = validate_relaxed_snapshot(base, final, prediction_steps=17)
    assert result["supported"] is True
    assert result["prediction_steps"] == 17
    assert result["energy_total_ev"] == -4.0
    assert result["energy_ev_per_atom"] == -2.0
    assert result["fmax_ev_per_a"] == 0.0
    assert 0.25 <= result["volume_ratio"] <= 4.0


@pytest.mark.parametrize("failure", ["composition", "energy", "cell", "volume"])
def test_validate_relaxed_snapshot_fails_closed_to_abstention(failure: str) -> None:
    base, final = _snapshot()
    if failure == "composition":
        final = Atoms("Li2", positions=final.positions, cell=final.cell, pbc=True)
        final.info["total_energy"] = -4.0
        final.info["stress"] = np.eye(3)
        final.arrays["forces"] = np.zeros((2, 3))
    elif failure == "energy":
        final.info["total_energy"] = np.nan
    elif failure == "cell":
        final.cell[0, 0] = np.nan
    elif failure == "volume":
        final.set_cell(np.eye(3) * 0.1, scale_atoms=True)
    result = validate_relaxed_snapshot(base, final, prediction_steps=4)
    assert result["supported"] is False
    assert result["error"]
