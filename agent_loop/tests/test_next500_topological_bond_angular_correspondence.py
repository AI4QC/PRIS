from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next500_topological_bond_angular_correspondence as n


def test_frozen_schema_direction_and_boundary() -> None:
    assert n.PROTOCOL == "2026-08-13-next500-topological-bond-angular-correspondence-v1"
    assert n.DESIGN_SHA256 == "8884d37ebabf6d7653dd83b154274b9b5268256c744f49bf24d495a54077430a"
    assert n.FEATURE_NAMES == ("tbac_topological_bond_angular_correspondence",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_high"}
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())


def test_equal_complete_bipartite_field_matches_equal_angular_territory() -> None:
    result = n.topological_bond_angular_correspondence(
        charges=(1.0, 1.0, -1.0, -1.0),
        endpoints=((0, 2), (0, 3), (1, 2), (1, 3)),
        solid_angles=(2.0, 2.0, 2.0, 2.0),
    )
    assert result.supported, result.failure_reason
    assert result.feasible
    assert result.edge_strengths == pytest.approx((0.5, 0.5, 0.5, 0.5))
    assert result.normalized_mismatch == pytest.approx(0.0)
    assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(1.0)


def test_unequal_angular_territory_creates_exact_bounded_mismatch() -> None:
    result = n.topological_bond_angular_correspondence(
        charges=(1.0, -1.0, -1.0),
        endpoints=((0, 1), (0, 2)),
        solid_angles=(3.0, 1.0),
    )
    # The graph cannot conserve the deliberately non-neutral charges.
    assert not result.supported
    assert result.features == {}

    result = n.topological_bond_angular_correspondence(
        charges=(2.0, -1.0, -1.0),
        endpoints=((0, 1), (0, 2)),
        solid_angles=(3.0, 1.0),
    )
    assert result.supported, result.failure_reason
    assert result.edge_strengths == pytest.approx((1.0, 1.0))
    # Four incidence terms: cation targets 1.5/0.5; anion targets 1/1.
    assert result.normalized_mismatch == pytest.approx(1.0 / 8.0)
    assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(0.875)


def test_negative_path_edge_contributes_maximal_incidence_mismatch() -> None:
    result = n.topological_bond_angular_correspondence(
        charges=(0.25, 0.5, -0.25, -0.5),
        endpoints=((0, 2), (0, 3), (1, 2)),
        solid_angles=(1.0, 1.0, 1.0),
    )
    assert result.supported, result.failure_reason
    assert result.edge_strengths == pytest.approx((-0.25, 0.5, 0.5), abs=1e-12)
    assert result.negative_edge_count == 1
    assert 0.0 < result.normalized_mismatch < 1.0
    assert 0.0 < result.features[n.FEATURE_NAMES[0]] < 1.0


def test_charge_infeasible_disconnected_graph_is_supported_zero() -> None:
    result = n.topological_bond_angular_correspondence(
        charges=(1.0, 0.5, -0.5, -1.0),
        endpoints=((0, 2), (1, 3)),
        solid_angles=(1.0, 1.0),
    )
    assert result.supported, result.failure_reason
    assert result.feasible is False
    assert result.features[n.FEATURE_NAMES[0]] == 0.0


def test_kernel_is_scale_edge_order_and_replication_invariant() -> None:
    charges = (2.0, -1.0, -1.0)
    endpoints = ((0, 1), (0, 2))
    angles = (3.0, 1.0)
    reference = n.topological_bond_angular_correspondence(
        charges=charges, endpoints=endpoints, solid_angles=angles
    )
    scaled = n.topological_bond_angular_correspondence(
        charges=tuple(7.3 * value for value in charges),
        endpoints=endpoints,
        solid_angles=angles,
    )
    reordered = n.topological_bond_angular_correspondence(
        charges=charges,
        endpoints=tuple(reversed(endpoints)),
        solid_angles=tuple(reversed(angles)),
    )
    replicated = n.topological_bond_angular_correspondence(
        charges=charges + charges,
        endpoints=endpoints + tuple((left + 3, right + 3) for left, right in endpoints),
        solid_angles=angles + angles,
    )
    for result in (reference, scaled, reordered, replicated):
        assert result.supported, result.failure_reason
        assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(
            reference.features[n.FEATURE_NAMES[0]], abs=1e-10
        )


@pytest.mark.parametrize(
    ("charges", "endpoints", "angles"),
    (
        ((), (), ()),
        ((1.0, -0.5), ((0, 1),), (1.0,)),
        ((1.0, -1.0), ((1, 0),), (1.0,)),
        ((1.0, -1.0), ((0, 2),), (1.0,)),
        ((1.0, -1.0), ((0, 1),), (0.0,)),
        ((1.0, -1.0), ((0, 1),), (np.nan,)),
        ((1.0, -1.0), ((0, 1),), (1.0, 2.0)),
    ),
)
def test_malformed_inputs_fail_closed(
    charges: object, endpoints: object, angles: object
) -> None:
    result = n.topological_bond_angular_correspondence(
        charges=charges, endpoints=endpoints, solid_angles=angles
    )
    assert not result.supported
    assert result.features == {}


def _distorted_nacl() -> Atoms:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.set_cell(
        [[5.64, 0, 0], [0.27, 5.77, 0], [0.18, 0.31, 5.53]],
        scale_atoms=True,
    )
    atoms.positions[1] += [0.08, -0.04, 0.06]
    atoms.wrap()
    return atoms


def _feature(atoms: Atoms) -> float:
    result = n.compute_tbac_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[n.FEATURE_NAMES[0]])


def test_periodic_equivalents_and_geometry_firewall() -> None:
    atoms = _distorted_nacl()
    reference = _feature(atoms)
    rotated = atoms.copy()
    rotated.rotate(31.0, "z", rotate_cell=True)
    translated = atoms.copy()
    translated.translate([0.17, 0.29, 0.42])
    translated.wrap()
    permuted = atoms[[3, 0, 6, 1, 7, 4, 2, 5]]
    rebased = atoms.copy()
    rebased.set_cell(
        np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]]) @ atoms.cell.array,
        scale_atoms=False,
    )
    rebased.wrap()
    for equivalent in (
        rotated,
        translated,
        permuted,
        rebased,
        atoms.repeat((2, 1, 1)),
    ):
        assert _feature(equivalent) == pytest.approx(reference, abs=1e-8)

    bad = atoms.copy()
    bad.calc = Calculator()
    result = n.compute_tbac_features(bad)
    assert not result.supported and "geometry-only Atoms" in str(result.failure_reason)
    row = n.compute_tbac_row(atoms)
    assert row["tbac_supported"] is True
    assert tuple(key for key in row if key.startswith("tbac_")) == (
        "tbac_topological_bond_angular_correspondence",
        "tbac_supported",
        "tbac_failure",
        "tbac_feasible",
        "tbac_site_count",
        "tbac_edge_count",
        "tbac_negative_edge_count",
        "tbac_normalized_mismatch",
        "tbac_maximum_equality_residual",
        "tbac_maximum_path_residual",
        "tbac_valence_policy",
    )
