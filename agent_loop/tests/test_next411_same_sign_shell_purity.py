from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next411_same_sign_shell_purity as n


def _distorted_nacl() -> Atoms:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.set_cell(
        np.asarray([[5.64, 0.0, 0.0], [0.27, 5.77, 0.0], [0.18, 0.31, 5.53]]),
        scale_atoms=True,
    )
    atoms.positions[1] += np.asarray([0.08, -0.04, 0.06])
    atoms.wrap()
    return atoms


def _feature(atoms: Atoms) -> float:
    result = n.compute_sssp_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[n.FEATURE_NAMES[0]])


def test_frozen_protocol_schema_direction_and_boundary_are_exact() -> None:
    assert n.PROTOCOL == "2026-08-13-next411-same-sign-shell-purity-v1"
    assert len(n.DESIGN_SHA256) == 64
    assert n.FEATURE_NAMES == ("sssp_same_sign_shell_purity_q10",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_high"}
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())


def test_kernel_uses_the_frozen_lower_tail_order_statistic() -> None:
    result = n.same_sign_shell_purity(
        opposite_shell_radii=(2.0, 2.0, 2.0, 2.0),
        nearest_same_sign_distances=(3.0, 1.0, 1.5, 2.0),
    )
    assert result.supported, result.failure_reason
    assert result.site_purities == pytest.approx((1.0, 0.5, 0.75, 1.0))
    assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(0.5)


def test_intrusion_strictly_reduces_shell_purity() -> None:
    clean = n.same_sign_shell_purity(
        opposite_shell_radii=(2.0,) * 10,
        nearest_same_sign_distances=(2.5,) * 10,
    )
    intruded = n.same_sign_shell_purity(
        opposite_shell_radii=(2.0,) * 10,
        nearest_same_sign_distances=(1.2,) + (2.5,) * 9,
    )
    assert clean.supported and intruded.supported
    assert clean.features[n.FEATURE_NAMES[0]] == 1.0
    assert intruded.features[n.FEATURE_NAMES[0]] == pytest.approx(0.6)
    assert intruded.features[n.FEATURE_NAMES[0]] < clean.features[n.FEATURE_NAMES[0]]


def test_kernel_is_permutation_and_exact_replication_invariant() -> None:
    radii = np.asarray((2.0, 3.0, 4.0, 5.0, 2.5, 3.5, 4.5))
    distances = np.asarray((1.6, 3.3, 2.8, 5.0, 2.0, 2.1, 5.4))
    reference = n.same_sign_shell_purity(
        opposite_shell_radii=radii, nearest_same_sign_distances=distances
    )
    permuted = n.same_sign_shell_purity(
        opposite_shell_radii=radii[[3, 0, 6, 2, 1, 5, 4]],
        nearest_same_sign_distances=distances[[3, 0, 6, 2, 1, 5, 4]],
    )
    replicated = n.same_sign_shell_purity(
        opposite_shell_radii=np.tile(radii, 3),
        nearest_same_sign_distances=np.tile(distances, 3),
    )
    for result in (reference, permuted, replicated):
        assert result.supported, result.failure_reason
        assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(
            reference.features[n.FEATURE_NAMES[0]], abs=1e-14
        )


@pytest.mark.parametrize(
    ("radii", "distances"),
    (
        ((), ()),
        ((2.0,), ()),
        ((0.0,), (1.0,)),
        ((2.0,), (0.0,)),
        ((np.nan,), (1.0,)),
        ((2.0,), (np.inf,)),
        (((2.0, 3.0),), ((1.0, 2.0),)),
    ),
)
def test_malformed_kernel_inputs_fail_closed(radii: object, distances: object) -> None:
    result = n.same_sign_shell_purity(
        opposite_shell_radii=radii, nearest_same_sign_distances=distances
    )
    assert result.supported is False
    assert result.features == {}


def test_distorted_crystal_and_equivalents_are_supported_and_invariant() -> None:
    atoms = _distorted_nacl()
    result = n.compute_sssp_features(atoms)
    assert result.supported, result.failure_reason
    assert result.site_count == len(atoms)
    assert result.edge_count >= len(atoms) // 2
    assert len(result.site_purities) == len(atoms)
    assert 0 < result.features[n.FEATURE_NAMES[0]] <= 1
    reference = _feature(atoms)
    rotated = atoms.copy(); rotated.rotate(31.0, "z", rotate_cell=True)
    translated = atoms.copy(); translated.translate([0.173, 0.291, 0.419]); translated.wrap()
    permuted = atoms[[3, 0, 6, 1, 7, 4, 2, 5]]
    rebased = atoms.copy(); rebased.set_cell(np.asarray([[1,1,0],[0,1,0],[0,0,1]]) @ atoms.cell.array, scale_atoms=False); rebased.wrap()
    for equivalent in (rotated, translated, permuted, rebased, atoms.repeat((2,1,1))):
        assert _feature(equivalent) == pytest.approx(reference, abs=1e-8)


def test_geometry_boundary_and_row_schema_are_exact() -> None:
    atoms = _distorted_nacl()
    changed = []
    item = atoms.copy(); item.calc = Calculator(); changed.append(item)
    item = atoms.copy(); item.info["outcome"] = 1; changed.append(item)
    item = atoms.copy(); item.new_array("energy", np.zeros(len(item))); changed.append(item)
    item = atoms.copy(); item.pbc = False; changed.append(item)
    item = atoms.copy(); item.positions[0, 0] = np.nan; changed.append(item)
    for item in changed:
        result = n.compute_sssp_features(item)
        assert not result.supported
        assert "geometry-only Atoms" in str(result.failure_reason)
    row = n.compute_sssp_row(atoms)
    assert tuple(name for name in row if name.startswith("sssp_")) == (
        "sssp_same_sign_shell_purity_q10", "sssp_supported", "sssp_failure",
        "sssp_site_count", "sssp_edge_count", "sssp_min_site_purity",
        "sssp_valence_policy",
    )
    assert row["sssp_supported"] is True

