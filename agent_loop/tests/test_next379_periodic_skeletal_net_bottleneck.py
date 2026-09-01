from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next379_periodic_skeletal_net_bottleneck as n


def _directed(rows):
    endpoints = []
    weights = []
    for left, right, image, weight in rows:
        reverse = tuple(-int(value) for value in image)
        endpoints.extend(
            (
                (left, right, *image),
                (right, left, *reverse),
            )
        )
        weights.extend((weight, weight))
    return np.asarray(endpoints, dtype=int), np.asarray(weights, dtype=float)


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
    result = n.compute_psnb_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[n.FEATURE_NAMES[0]])


def test_frozen_protocol_schema_and_direction_are_exact() -> None:
    assert n.PROTOCOL == "2026-08-13-next379-periodic-skeletal-net-bottleneck-v1"
    assert n.FEATURE_NAMES == ("psnb_skeletal_3d_bottleneck_q10",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_high"}
    assert len(n.DESIGN_SHA256) == 64


def test_one_site_three_axis_net_has_exact_bottleneck() -> None:
    endpoints, weights = _directed(
        (
            (0, 0, (1, 0, 0), 0.8),
            (0, 0, (0, 1, 0), 0.6),
            (0, 0, (0, 0, 1), 0.4),
        )
    )
    result = n.skeletal_net_bottleneck(
        n_sites=1, endpoints=endpoints, solid_angles=weights
    )
    assert result.supported, result.failure_reason
    assert result.rank3_site_count == 1
    assert result.site_bottlenecks == pytest.approx((0.5,), abs=1.0e-14)
    assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(0.5, abs=1.0e-14)


def test_rank_deficient_net_has_zero_bottleneck() -> None:
    endpoints, weights = _directed(
        (
            (0, 0, (1, 0, 0), 1.0),
            (0, 0, (0, 1, 0), 0.6),
        )
    )
    result = n.skeletal_net_bottleneck(
        n_sites=1, endpoints=endpoints, solid_angles=weights
    )
    assert result.supported, result.failure_reason
    assert result.rank3_site_count == 0
    assert result.site_bottlenecks == (0.0,)
    assert result.features[n.FEATURE_NAMES[0]] == 0.0


def test_simultaneous_ties_are_added_before_rank_assignment() -> None:
    endpoints, weights = _directed(
        (
            (0, 0, (1, 0, 0), 1.0),
            (0, 0, (0, 1, 0), 1.0),
            (0, 0, (0, 0, 1), 1.0),
        )
    )
    result = n.skeletal_net_bottleneck(
        n_sites=1, endpoints=endpoints, solid_angles=weights
    )
    assert result.supported, result.failure_reason
    assert result.site_bottlenecks == (1.0,)


def test_kernel_is_order_scale_and_disjoint_replication_invariant() -> None:
    endpoints, weights = _directed(
        (
            (0, 0, (1, 0, 0), 0.8),
            (0, 0, (0, 1, 0), 0.6),
            (0, 0, (0, 0, 1), 0.4),
        )
    )
    reference = n.skeletal_net_bottleneck(
        n_sites=1, endpoints=endpoints, solid_angles=weights
    )
    order = np.asarray((4, 1, 5, 0, 3, 2))
    reordered = n.skeletal_net_bottleneck(
        n_sites=1,
        endpoints=endpoints[order],
        solid_angles=7.0 * weights[order],
    )
    replicated = n.skeletal_net_bottleneck(
        n_sites=2,
        endpoints=np.vstack(
            (endpoints, endpoints + np.asarray((1, 1, 0, 0, 0)))
        ),
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
        (
            2,
            [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
            [1.0, 1.0, 1.0],
        ),
    ),
)
def test_malformed_kernel_inputs_fail_closed(
    n_sites: int, endpoints: object, weights: object
) -> None:
    result = n.skeletal_net_bottleneck(
        n_sites=n_sites,
        endpoints=np.asarray(endpoints),
        solid_angles=np.asarray(weights),
    )
    assert result.supported is False
    assert result.features == {}


def test_periodic_voronoi_structure_is_supported_and_bounded() -> None:
    result = n.compute_psnb_features(_distorted_nacl())
    assert result.supported, result.failure_reason
    assert result.site_count == len(_distorted_nacl())
    assert result.directed_face_count > 0
    assert result.undirected_edge_count * 2 == result.directed_face_count
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
        result = n.compute_psnb_features(changed)
        assert result.supported is False
        assert "geometry-only Atoms" in str(result.failure_reason)


def test_row_schema_and_no_dft_boundary_flags_are_exact() -> None:
    row = n.compute_psnb_row(_distorted_nacl())
    assert tuple(name for name in row if name.startswith("psnb_")) == (
        "psnb_skeletal_3d_bottleneck_q10",
        "psnb_supported",
        "psnb_failure",
        "psnb_site_count",
        "psnb_directed_face_count",
        "psnb_undirected_edge_count",
        "psnb_rank3_site_count",
        "psnb_maximum_reverse_angle_error",
    )
    assert row["psnb_supported"] is True
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())
