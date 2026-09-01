from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next416_opposite_bridge_separation as n


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
    result = n.compute_obs_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[n.FEATURE_NAMES[0]])


def test_frozen_protocol_schema_direction_and_boundary_are_exact() -> None:
    assert n.PROTOCOL == "2026-08-13-next416-opposite-bridge-separation-v1"
    assert len(n.DESIGN_SHA256) == 64
    assert n.FEATURE_NAMES == ("obs_opposite_bridge_separation_q10",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_high"}
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())


def test_kernel_uses_center_minimum_then_frozen_lower_tail() -> None:
    centers = (
        ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((2.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
    )
    result = n.opposite_bridge_separation(center_vectors=centers)
    assert result.supported, result.failure_reason
    assert result.center_separations == pytest.approx((1.0, 2**0.5 / 2, 1.0))
    assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(
        round(2**0.5 / 2, 10)
    )


def test_bending_one_bridge_strictly_reduces_separation() -> None:
    straight = n.opposite_bridge_separation(
        center_vectors=(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),) * 10
    )
    bent = n.opposite_bridge_separation(
        center_vectors=(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            *(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),) * 9,
        )
    )
    assert straight.supported and bent.supported
    assert straight.features[n.FEATURE_NAMES[0]] == 1.0
    assert bent.features[n.FEATURE_NAMES[0]] == pytest.approx(round(2**0.5 / 2, 10))
    assert bent.features[n.FEATURE_NAMES[0]] < straight.features[n.FEATURE_NAMES[0]]


def test_kernel_is_rotation_scale_order_and_replication_invariant() -> None:
    centers = (
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)),
        ((0.2, 0.4, 0.8), (-0.7, 0.1, -0.2)),
    )
    reference = n.opposite_bridge_separation(center_vectors=centers)
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed = tuple(
        tuple((3.7 * (rotation @ np.asarray(vector))).tolist() for vector in reversed(row))
        for row in reversed(centers)
    )
    replicated = centers * 4
    for result in (
        reference,
        n.opposite_bridge_separation(center_vectors=transformed),
        n.opposite_bridge_separation(center_vectors=replicated),
    ):
        assert result.supported, result.failure_reason
        assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(
            reference.features[n.FEATURE_NAMES[0]], abs=1e-14
        )


@pytest.mark.parametrize(
    "vectors",
    (
        (),
        (((1.0, 0.0, 0.0),),),
        (((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),),
        (((np.nan, 0.0, 0.0), (1.0, 0.0, 0.0)),),
        (((1.0, 0.0), (-1.0, 0.0)),),
    ),
)
def test_malformed_kernel_inputs_fail_closed(vectors: object) -> None:
    result = n.opposite_bridge_separation(center_vectors=vectors)
    assert result.supported is False
    assert result.features == {}


def test_distorted_crystal_and_equivalents_are_supported_and_invariant() -> None:
    atoms = _distorted_nacl()
    result = n.compute_obs_features(atoms)
    assert result.supported, result.failure_reason
    assert result.center_count == len(atoms)
    assert result.edge_count >= len(atoms) // 2
    assert len(result.center_separations) == len(atoms)
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
        result = n.compute_obs_features(item)
        assert not result.supported
        assert "geometry-only Atoms" in str(result.failure_reason)
    row = n.compute_obs_row(atoms)
    assert tuple(name for name in row if name.startswith("obs_")) == (
        "obs_opposite_bridge_separation_q10",
        "obs_supported",
        "obs_failure",
        "obs_center_count",
        "obs_edge_count",
        "obs_min_center_separation",
        "obs_valence_policy",
    )
    assert row["obs_supported"] is True
