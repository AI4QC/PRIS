"""Mathematical contract tests for the np-next-20260801 descriptor families."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from next_features import (  # noqa: E402
    P5HOP_MEFIR_DAMPING,
    _econ_weighted_sum,
    next_local_features,
    p2_voronoi_site_stats,
    p3_hawthorne_features,
    p5_hoppe_features,
)


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
    structure = Structure(lattice, species, coords)
    valences = [1.0] * 4 + [-1.0] * 4
    return structure, valences


def _crystal_nn(structure):
    from pymatgen.analysis.local_env import CrystalNN

    finder = CrystalNN(weighted_cn=False, x_diff_weight=0.0)
    return [finder.get_nn_info(structure, i) for i in range(len(structure))]


# --------------------------------------------------------------------- P2


def test_p2_entropy_cn_and_like_fraction_on_injected_polyhedra():
    from pymatgen.core import Lattice, PeriodicSite, Structure

    lattice = Lattice.cubic(10.0)
    structure = Structure(
        lattice,
        ["Na", "Cl", "Cl"],
        [[0, 0, 0], [0.3, 0.3, 0.3], [0.7, 0.7, 0.7]],
    )
    valences = [1.0, -1.0, -1.0]

    def site(index):
        return PeriodicSite(
            structure[index].specie, structure[index].frac_coords, lattice
        )

    poly_site0 = {
        key: {"site": site(1), "solid_angle": 1.0} for key in range(4)
    }
    poly_site1 = {0: {"site": site(0), "solid_angle": 4.0}}
    poly_site2 = {0: {"site": site(1), "solid_angle": 4.0}}
    stats = p2_voronoi_site_stats(
        structure, valences, polyhedra=[poly_site0, poly_site1, poly_site2]
    )
    # Four equal solid-angle anion facets: entropy CN = 4, like share = 0.
    assert stats[0] is not None
    assert stats[0]["sa_effective_cn"] == pytest.approx(4.0, abs=1e-9)
    assert stats[0]["sa_like_fraction"] == pytest.approx(0.0, abs=1e-12)
    assert stats[0]["sa_max_fraction"] == pytest.approx(0.25, abs=1e-12)
    # Anion site 2 sees one anion facet: like share = 1.
    assert stats[2]["sa_like_fraction"] == pytest.approx(1.0, abs=1e-12)


def test_p2_nacl_tessellation_sums_to_four_pi():
    structure, valences = _nacl()
    from next_features import _voronoi_polyhedra

    poly = _voronoi_polyhedra(structure)
    totals = [sum(p["solid_angle"] for p in site.values()) for site in poly]
    assert totals == pytest.approx([4 * np.pi] * len(structure), abs=1e-8)
    stats = p2_voronoi_site_stats(structure, valences, polyhedra=poly)
    assert all(entry is not None for entry in stats)
    cn = [entry["sa_effective_cn"] for entry in stats]
    assert min(cn) >= 3.0
    assert max(cn) <= 14.0
    like = [entry["sa_like_fraction"] for entry in stats]
    assert max(like) < 0.5  # NaCl is dominated by unlike facets


# --------------------------------------------------------------------- P3


def test_p3_nacl_feasible_and_pauling_gap_zero():
    structure, valences = _nacl()
    neighbors = _crystal_nn(structure)
    out = p3_hawthorne_features(structure, valences, neighbors=neighbors)
    assert out["p3haw_unbonded_charged_fraction"] == pytest.approx(0.0)
    assert out["p3haw_nnls_relres"] < 1e-6
    assert out["p3haw_minnorm_relres"] < 1e-6
    assert out["p3haw_pauling_gap"] < 1e-4
    assert 0.0 <= out["p3haw_rank_deficiency"] < 1.0


def test_p3_inconsistent_valences_have_positive_residual():
    from pymatgen.core import Lattice, Structure

    structure = Structure(
        Lattice.cubic(8.0), ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    injected = [
        [{"site_index": 1}],
        [{"site_index": 0}],
    ]
    out = p3_hawthorne_features(structure, [3.0, -2.0], neighbors=injected)
    # One bond cannot satisfy both |V|=3 and |V|=2: s=2.5 minimises the residual.
    expected = abs(3.0 - 2.5) * np.sqrt(2) / np.linalg.norm([3.0, 2.0])
    assert out["p3haw_nnls_relres"] == pytest.approx(expected, rel=1e-6)
    assert out["p3haw_nnls_relres"] > 0.05
    assert out["p3haw_unbonded_charged_fraction"] == pytest.approx(0.0)


def test_p3_unbonded_charged_sites_counted():
    from pymatgen.core import Lattice, Structure

    structure = Structure(
        Lattice.cubic(8.0), ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    empty = [[], []]
    out = p3_hawthorne_features(structure, [2.0, -2.0], neighbors=empty)
    assert out["p3haw_unbonded_charged_fraction"] == pytest.approx(1.0)
    assert "p3haw_nnls_relres" not in out  # no bonds: solver features abstain


# --------------------------------------------------------------------- P5


def test_p5_econ_weighted_sum_contract():
    assert _econ_weighted_sum(np.asarray([2.0])) == pytest.approx(1.0)
    assert _econ_weighted_sum(np.asarray([2.0, 2.0])) == pytest.approx(2.0)
    near = _econ_weighted_sum(np.asarray([2.0, 2.4]))
    assert 1.05 < near < 1.25  # the 1.2x shell is a small fractional weight
    far = _econ_weighted_sum(np.asarray([2.0, 4.0]))
    assert far == pytest.approx(1.0, abs=1e-12)  # the 2x shell vanishes


def test_p5_nacl_mefir_recovers_shannon_radii():
    structure, valences = _nacl()
    neighbors = _crystal_nn(structure)
    out = p5_hoppe_features(structure, valences, neighbors=neighbors)
    # Na-Cl distance 2.82 = 1.02 (Na, CN6) + 1.81 (Cl): mismatch should vanish.
    assert out["p5hop_mefir_converged_fraction"] == 1.0
    assert abs(out["p5hop_cat_mefir_rel_mean"]) < 0.05
    assert abs(out["p5hop_an_mefir_rel_mean"]) < 0.05
    assert out["p5hop_mefir_site_coverage"] == pytest.approx(1.0)
    # ECoN strict >= CrystalNN-set approximation at every aggregated charge.
    assert out["p5hop_cat_econ_delta_mean"] >= -1e-9
    assert out["p5hop_cat_econ_strict_mean"] >= 5.9


def test_p5_damping_keeps_fixed_point():
    # A damped fixed point equals the undamped one: f(R) = R.
    structure, valences = _nacl()
    neighbors = _crystal_nn(structure)
    out_a = p5_hoppe_features(structure, valences, neighbors=neighbors)
    assert P5HOP_MEFIR_DAMPING == 0.5
    out_b = p5_hoppe_features(structure, valences, neighbors=neighbors)
    assert out_a == out_b  # deterministic


# -------------------------------------------------------------- integration


def test_next_local_features_nacl_all_families_present():
    structure, valences = _nacl()
    out, failures = next_local_features(structure, valences)
    assert failures == {}
    for prefix in ("bvloc_", "p2vor_", "p3haw_", "p5hop_"):
        assert any(column.startswith(prefix) for column in out), prefix
    assert out["bvloc_bond_parameter_coverage"] > 0.99
    assert out["p3haw_nnls_relres"] < 1e-6
    assert out["p2vor_cat_sa_effective_cn_mean"] >= 3.0
    assert np.isfinite(out["p5hop_cat_mefir_delta_mean"])
    # Exact-policy diagnostics are present but never searchable.
    assert "bvlocx_site_coverage" in out
    assert "bvlocx_bond_parameter_coverage" in out


def test_next_local_features_missing_family_marks_failure_not_crash():
    structure, valences = _nacl()

    import next_features

    original = next_features._voronoi_polyhedra

    def broken(_structure):
        raise RuntimeError("synthetic qhull failure")

    next_features._voronoi_polyhedra = broken
    try:
        out, failures = next_local_features(structure, valences)
    finally:
        next_features._voronoi_polyhedra = original
    assert failures.get("p2:RuntimeError") == 1
    assert not any(column.startswith("p2vor_") for column in out)
    assert any(column.startswith("p3haw_") for column in out)  # others survive
