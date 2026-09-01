from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next375_periodic_coordination_reciprocity_likelihood as n


def _directed_edges(edges):
    rows = []
    weights = []
    for left, right, weight in edges:
        rows.extend(((left, right, 0, 0, 0), (right, left, 0, 0, 0)))
        weights.extend((weight, weight))
    return np.asarray(rows, dtype=int), np.asarray(weights, dtype=float)


def _distorted_nacl() -> Atoms:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.set_cell(
        np.asarray([[5.64, 0.0, 0.0], [0.21, 5.76, 0.0], [0.13, 0.29, 5.51]]),
        scale_atoms=True,
    )
    atoms.positions[1] += np.asarray([0.07, -0.03, 0.05])
    atoms.wrap()
    return atoms


def _feature(atoms: Atoms) -> float:
    result = n.compute_pcrl_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[n.FEATURE_NAMES[0]])


def test_frozen_protocol_schema_and_direction_are_exact() -> None:
    assert n.PROTOCOL == "2026-08-13-next375-periodic-coordination-reciprocity-likelihood-v1"
    assert n.FEATURE_NAMES == ("pcrl_reciprocity_deficit",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_low"}
    assert len(n.DESIGN_SHA256) == 64


def test_fully_reciprocal_local_prefixes_have_zero_deficit() -> None:
    endpoints, weights = _directed_edges(
        ((0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0))
    )
    result = n.coordination_reciprocity_likelihood(
        n_sites=4, endpoints=endpoints, solid_angles=weights
    )
    assert result.supported, result.failure_reason
    assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(0.0, abs=1.0e-14)


def test_independently_selected_nonreciprocal_prefix_is_penalized() -> None:
    # Site 0 strongly separates 0--1 from 0--2 and omits 0--2.  Site 2 sees
    # 2--0 and 2--3 as nearly equal and selects both, so 2->0 is unreciprocated.
    endpoints, weights = _directed_edges(((0, 1, 1.0), (0, 2, 0.1), (2, 3, 0.09)))
    # Remove 2--3 from sites 0/1 connectivity only by adding a strong 1--3
    # contact; the exact expected condition is checked through selected counts.
    endpoints, weights = _directed_edges(
        ((0, 1, 1.0), (0, 2, 0.1), (2, 3, 0.09), (1, 3, 0.01))
    )
    result = n.coordination_reciprocity_likelihood(
        n_sites=4, endpoints=endpoints, solid_angles=weights
    )
    assert result.supported, result.failure_reason
    assert result.unreciprocated_directed_count > 0
    assert 0.0 < result.features[n.FEATURE_NAMES[0]] <= 1.0


def test_kernel_is_edge_order_weight_scale_and_disjoint_replication_invariant() -> None:
    endpoints, weights = _directed_edges(
        ((0, 1, 1.0), (0, 2, 0.1), (2, 3, 0.09), (1, 3, 0.01))
    )
    reference = n.coordination_reciprocity_likelihood(
        n_sites=4, endpoints=endpoints, solid_angles=weights
    )
    order = np.asarray((4, 1, 7, 0, 3, 6, 2, 5))
    reordered = n.coordination_reciprocity_likelihood(
        n_sites=4, endpoints=endpoints[order], solid_angles=7.0 * weights[order]
    )
    replicated = n.coordination_reciprocity_likelihood(
        n_sites=8,
        endpoints=np.vstack((endpoints, endpoints + np.asarray((4, 4, 0, 0, 0)))),
        solid_angles=np.concatenate((weights, weights)),
    )
    for result in (reference, reordered, replicated):
        assert result.supported, result.failure_reason
        assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(
            reference.features[n.FEATURE_NAMES[0]], abs=1.0e-14
        )


@pytest.mark.parametrize(
    ("n_sites", "endpoints", "weights"),
    (
        (2, [[0, 1, 0, 0, 0]], [1.0]),
        (2, [[0, 2, 0, 0, 0], [2, 0, 0, 0, 0]], [1.0, 1.0]),
        (2, [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], [1.0, 0.9]),
        (2, [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], [0.0, 0.0]),
    ),
)
def test_malformed_or_nonreciprocal_kernel_inputs_fail_closed(
    n_sites: int, endpoints: object, weights: object
) -> None:
    result = n.coordination_reciprocity_likelihood(
        n_sites=n_sites,
        endpoints=np.asarray(endpoints),
        solid_angles=np.asarray(weights),
    )
    assert result.supported is False
    assert result.features == {}


def test_periodic_voronoi_structure_is_supported_and_bounded() -> None:
    result = n.compute_pcrl_features(_distorted_nacl())
    assert result.supported, result.failure_reason
    assert result.site_count == len(_distorted_nacl())
    assert result.directed_face_count > 0
    assert result.selected_directed_count > 0
    assert result.maximum_reverse_angle_error <= n.REVERSE_ANGLE_TOLERANCE
    assert 0.0 <= result.features[n.FEATURE_NAMES[0]] <= 1.0


def test_structure_equivalences_preserve_feature() -> None:
    atoms = _distorted_nacl()
    reference = _feature(atoms)
    rotated = atoms.copy()
    rotated.rotate(31.0, "z", rotate_cell=True)
    translated = atoms.copy()
    translated.translate([0.173, 0.291, 0.419])
    translated.wrap()
    permuted = atoms[[3, 0, 6, 1, 7, 4, 2, 5]]
    rebased = atoms.copy()
    rebased.set_cell(
        np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
        @ atoms.cell.array,
        scale_atoms=False,
    )
    rebased.wrap()
    replicated = atoms.repeat((2, 1, 1))
    for equivalent in (rotated, translated, permuted, rebased, replicated):
        assert _feature(equivalent) == pytest.approx(reference, abs=1.0e-8)


def test_geometry_only_boundary_fails_closed() -> None:
    atoms = _distorted_nacl()
    with_calculator = atoms.copy()
    with_calculator.calc = Calculator()
    with_metadata = atoms.copy()
    with_metadata.info["outcome"] = 1
    with_array = atoms.copy()
    with_array.new_array("energy", np.zeros(len(with_array)))
    nonperiodic = atoms.copy()
    nonperiodic.pbc = False
    nonfinite = atoms.copy()
    nonfinite.positions[0, 0] = np.nan
    for changed in (with_calculator, with_metadata, with_array, nonperiodic, nonfinite):
        result = n.compute_pcrl_features(changed)
        assert result.supported is False
        assert "geometry-only Atoms" in str(result.failure_reason)


def test_row_schema_and_no_dft_boundary_flags_are_exact() -> None:
    row = n.compute_pcrl_row(_distorted_nacl())
    assert tuple(name for name in row if name.startswith("pcrl_")) == (
        "pcrl_reciprocity_deficit",
        "pcrl_supported",
        "pcrl_failure",
        "pcrl_site_count",
        "pcrl_directed_face_count",
        "pcrl_selected_directed_count",
        "pcrl_unreciprocated_directed_count",
        "pcrl_maximum_reverse_angle_error",
    )
    assert row["pcrl_supported"] is True
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())
