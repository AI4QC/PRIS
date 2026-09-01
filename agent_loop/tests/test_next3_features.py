"""Mathematical contract tests for the np-next-20260801c descriptor families."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from next3_features import (  # noqa: E402
    next3_local_features,
    p6_shell_gap_site_stats,
    p7_polyanion_features,
    p8_jaccard_site_stats,
    p9_lewis_features,
    p10_voronoi_volume_site_stats,
)
from next_features import _crystal_nn_info, _voronoi_polyhedra  # noqa: E402


def _nacl():
    from pymatgen.core import Lattice, Structure

    lattice = Lattice.cubic(5.64)
    species = ["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"]
    coords = [
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0],
        [0.5, 0, 0],
        [0, 0.5, 0],
        [0, 0, 0.5],
        [0.5, 0.5, 0.5],
    ]
    return Structure(lattice, species, coords), [1.0] * 4 + [-1.0] * 4


def _azide_like():
    from pymatgen.core import Lattice, Structure

    # Linear N-N-N unit: two nitrogens 1.15 A apart (azide bond).
    lattice = Lattice.tetragonal(6.0, 8.0)
    species = ["Na", "N", "N"]
    coords = [[0.5, 0.5, 0.0], [0.0, 0.0, 0.30], [0.0, 0.0, 0.444]]
    return Structure(lattice, species, coords), [1.0, -1.5, -1.5]


def test_p6_nacl_gap_position_is_six():
    structure, valences = _nacl()
    sphere = structure.get_all_neighbors(8.0)
    stats = p6_shell_gap_site_stats(structure, valences, sphere=sphere)
    assert all(entry is not None for entry in stats)
    assert stats[0]["gap_pos"] == pytest.approx(6.0)
    assert stats[0]["gap_ratio"] > 1.2
    assert stats[0]["shell_width"] == pytest.approx(1.0, abs=1e-9)


def test_p7_azide_detects_short_nn_contact():
    structure, valences = _azide_like()
    sphere = structure.get_all_neighbors(8.0)
    out = p7_polyanion_features(structure, valences, sphere=sphere)
    # N-N 1.15 A vs 2 * 0.71 A covalent reference -> ratio ~0.81 < 1.3.
    assert out["p7poly_an_contact_min"] < 1.0
    assert out["p7poly_an_contact_frac"] == pytest.approx(1.0)


def test_p7_nacl_has_no_short_anion_contact():
    structure, valences = _nacl()
    sphere = structure.get_all_neighbors(8.0)
    out = p7_polyanion_features(structure, valences, sphere=sphere)
    # Cl-Cl nearest in NaCl is 3.99 A vs 1.98 A reference -> ratio ~2.0.
    assert out["p7poly_an_contact_min"] > 1.8
    assert out["p7poly_an_contact_frac"] == pytest.approx(0.0)


def test_p8_nacl_algorithms_mostly_agree():
    structure, valences = _nacl()
    neighbors = _crystal_nn_info(structure)
    polyhedra = _voronoi_polyhedra(structure)
    stats = p8_jaccard_site_stats(
        structure, valences, neighbors=neighbors, polyhedra=polyhedra
    )
    assert all(entry is not None for entry in stats)
    assert max(entry["jaccard"] for entry in stats) < 0.6


def test_p9_nacl_perfect_matching():
    structure, valences = _nacl()
    neighbors = _crystal_nn_info(structure)
    out = p9_lewis_features(structure, valences, neighbors=neighbors)
    # NaCl: acidity 1/6, basicity 1/6 -> mismatch 0 everywhere.
    assert out["p9lew_bond_mismatch_max"] < 1e-9
    assert out["p9lew_cat_site_mismatch_max"] < 1e-9


def test_p10_nacl_centered_and_near_unit_freevol():
    structure, valences = _nacl()
    neighbors = _crystal_nn_info(structure)
    polyhedra = _voronoi_polyhedra(structure)
    stats = p10_voronoi_volume_site_stats(
        structure, valences, neighbors=neighbors, polyhedra=polyhedra
    )
    assert all(entry is not None for entry in stats)
    assert max(entry["offcenter"] for entry in stats) < 1e-6
    # NaCl: cell volume 22.4 A^3; Na sphere (r=1.02) 4.45 -> 5.04,
    # Cl sphere (r=1.81) 24.9 -> 0.90.  Small cations sit above 1.
    freevol = sorted(entry["freevol"] for entry in stats)
    assert freevol[0] == pytest.approx(0.903, rel=0.01)
    assert freevol[-1] == pytest.approx(5.04, rel=0.01)


def test_next3_nacl_all_families_present():
    structure, valences = _nacl()
    out, failures = next3_local_features(structure, valences)
    assert failures == {}
    for prefix in ("p4csm_", "p6gap_", "p7poly_", "p8nnj_", "p9lew_", "p10vor_"):
        assert any(column.startswith(prefix) for column in out), prefix
    # NaCl octahedra are ideal: CSM at the sodium site is ~0.
    assert out["p4csm_cat_max"] < 0.01
    assert out["p6gap_cat_gap_pos_mean"] == pytest.approx(6.0)


def test_next3_family_failure_isolation():
    import next3_features

    structure, valences = _nacl()
    original = next3_features.p4_csm_site_values

    def broken(_structure, _valences):
        raise RuntimeError("synthetic chemenv failure")

    next3_features.p4_csm_site_values = broken
    try:
        out, failures = next3_local_features(structure, valences)
    finally:
        next3_features.p4_csm_site_values = original
    assert failures.get("p4:RuntimeError") == 1
    assert not any(column.startswith("p4csm_cat") for column in out)
    assert any(column.startswith("p6gap_") for column in out)
