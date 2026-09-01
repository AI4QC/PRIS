from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import math
import numpy as np
import pytest

import src.next455_path_field_participation_uniformity as n


def test_frozen_schema_direction_and_boundary() -> None:
    assert n.PROTOCOL == "2026-08-13-next455-path-field-participation-uniformity-v1"
    assert n.DESIGN_SHA256 == "4ffde6f73aeb85d004383f0151d764c763fa8ca3dadbc34d9a0ae6b13eb76c7e"
    assert n.FEATURE_NAMES == ("pfpu_path_field_participation_uniformity",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_high"}
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())


def test_uniform_complete_graph_has_unit_participation() -> None:
    result = n.path_field_participation_uniformity(
        charges=(1.0, 1.0, -1.0, -1.0),
        endpoints=((0, 2), (0, 3), (1, 2), (1, 3)),
    )
    assert result.supported, result.failure_reason
    assert result.feasible
    assert result.edge_strengths == pytest.approx((0.5, 0.5, 0.5, 0.5))
    assert result.effective_edge_count == pytest.approx(4.0)
    assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(1.0)


def test_nonuniform_field_matches_shannon_effective_participation() -> None:
    result = n.path_field_participation_uniformity(
        charges=(0.25, 0.5, -0.25, -0.5),
        endpoints=((0, 2), (0, 3), (1, 2)),
    )
    weights = np.asarray((0.25, 0.5, 0.5), dtype=float)
    probability = weights / weights.sum()
    expected = math.exp(-float(np.sum(probability * np.log(probability)))) / 3.0
    assert result.supported, result.failure_reason
    assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(expected, abs=1e-10)
    assert 0.0 < result.features[n.FEATURE_NAMES[0]] < 1.0


def test_obstructed_graph_is_supported_zero() -> None:
    result = n.path_field_participation_uniformity(
        charges=(1.0, 0.5, -0.5, -1.0),
        endpoints=((0, 2), (1, 3)),
    )
    assert result.supported, result.failure_reason
    assert result.feasible is False
    assert result.features[n.FEATURE_NAMES[0]] == 0.0


def test_kernel_is_charge_edge_order_and_exact_replication_invariant() -> None:
    charges = (0.25, 0.5, -0.25, -0.5)
    endpoints = ((0, 2), (0, 3), (1, 2))
    reference = n.path_field_participation_uniformity(
        charges=charges, endpoints=endpoints
    )
    scaled = n.path_field_participation_uniformity(
        charges=tuple(7.3 * value for value in charges), endpoints=endpoints
    )
    reordered = n.path_field_participation_uniformity(
        charges=charges, endpoints=tuple(reversed(endpoints))
    )
    replicated = n.path_field_participation_uniformity(
        charges=charges + charges,
        endpoints=endpoints
        + tuple((left + 4, right + 4) for left, right in endpoints),
    )
    for result in (reference, scaled, reordered, replicated):
        assert result.supported, result.failure_reason
        assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(
            reference.features[n.FEATURE_NAMES[0]], abs=1e-10
        )


@pytest.mark.parametrize(
    ("charges", "endpoints"),
    (
        ((), ()),
        ((1.0, -0.5), ((0, 1),)),
        ((1.0, -1.0), ((1, 0),)),
        ((1.0, -1.0), ((0, 2),)),
        ((1.0, np.nan), ((0, 1),)),
    ),
)
def test_malformed_inputs_fail_closed(charges: object, endpoints: object) -> None:
    result = n.path_field_participation_uniformity(
        charges=charges, endpoints=endpoints
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
    result = n.compute_pfpu_features(atoms)
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
    result = n.compute_pfpu_features(bad)
    assert not result.supported and "geometry-only Atoms" in str(result.failure_reason)
    row = n.compute_pfpu_row(atoms)
    assert row["pfpu_supported"] is True
    assert tuple(key for key in row if key.startswith("pfpu_")) == (
        "pfpu_path_field_participation_uniformity",
        "pfpu_supported",
        "pfpu_failure",
        "pfpu_feasible",
        "pfpu_site_count",
        "pfpu_edge_count",
        "pfpu_effective_edge_count",
        "pfpu_shannon_entropy",
        "pfpu_maximum_equality_residual",
        "pfpu_maximum_path_residual",
        "pfpu_valence_policy",
    )
