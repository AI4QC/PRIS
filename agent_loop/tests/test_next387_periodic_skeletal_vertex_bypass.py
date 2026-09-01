from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next387_periodic_skeletal_vertex_bypass as n


def _directed(rows):
    endpoints, weights = [], []
    for left, right, image, weight in rows:
        reverse = tuple(-int(x) for x in image)
        endpoints.extend(((left, right, *image), (right, left, *reverse)))
        weights.extend((weight, weight))
    return np.asarray(endpoints, dtype=int), np.asarray(weights, dtype=float)


def _distorted_nacl() -> Atoms:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.set_cell(np.asarray([[5.64, 0, 0], [0.21, 5.76, 0], [0.13, 0.29, 5.51]]), scale_atoms=True)
    atoms.positions[1] += [0.07, -0.03, 0.05]
    atoms.wrap()
    return atoms


def _feature(atoms):
    result = n.compute_psvb_features(atoms)
    assert result.supported, result.failure_reason
    return result.features[n.FEATURE_NAMES[0]]


def test_frozen_schema() -> None:
    assert n.FEATURE_NAMES == ("psvb_skeletal_vertex_bypass4_q10",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_high"}
    assert n.MAXIMUM_BYPASS_LENGTH == 4


def test_periodic_chain_has_zero_bypass() -> None:
    endpoints, weights = _directed(((0, 0, (1, 0, 0), 1.0),))
    result = n.skeletal_vertex_bypass(n_sites=1, endpoints=endpoints, solid_angles=weights)
    assert result.supported, result.failure_reason
    assert result.total_neighbor_pair_count == 1
    assert result.total_bypassed_pair_count == 0
    assert result.site_bypass_fractions == (0.0,)


def test_simple_cubic_has_complete_four_step_bypass() -> None:
    endpoints, weights = _directed(((0, 0, (1, 0, 0), 1.0), (0, 0, (0, 1, 0), 1.0), (0, 0, (0, 0, 1), 1.0)))
    result = n.skeletal_vertex_bypass(n_sites=1, endpoints=endpoints, solid_angles=weights)
    assert result.supported, result.failure_reason
    assert result.total_neighbor_pair_count == 15
    assert result.total_bypassed_pair_count == 15
    assert result.site_bypass_fractions == (1.0,)


def test_ties_order_scale_and_replication_are_invariant() -> None:
    endpoints, weights = _directed(((0, 0, (1, 0, 0), 1), (0, 0, (0, 1, 0), 0.5), (0, 0, (0, 0, 1), 0.5)))
    order = np.asarray((4, 1, 5, 0, 3, 2))
    reference = n.skeletal_vertex_bypass(n_sites=1, endpoints=endpoints, solid_angles=weights)
    changed = n.skeletal_vertex_bypass(n_sites=1, endpoints=endpoints[order], solid_angles=7 * weights[order])
    replicated = n.skeletal_vertex_bypass(n_sites=2, endpoints=np.vstack((endpoints, endpoints + [1, 1, 0, 0, 0])), solid_angles=np.concatenate((weights, weights)))
    assert reference.skeleton_threshold == 0.5 and reference.skeleton_edge_count == 3
    for result in (reference, changed, replicated):
        assert result.supported, result.failure_reason
        assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(reference.features[n.FEATURE_NAMES[0]])


@pytest.mark.parametrize(("n_sites", "endpoints", "weights"), (
    (2, [[0, 1, 0, 0, 0]], [1]),
    (2, [[0, 2, 0, 0, 0], [2, 0, 0, 0, 0]], [1, 1]),
    (2, [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], [1, 0.9]),
    (2, [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], [0, 0]),
))
def test_malformed_fails_closed(n_sites, endpoints, weights) -> None:
    assert not n.skeletal_vertex_bypass(n_sites=n_sites, endpoints=np.asarray(endpoints), solid_angles=np.asarray(weights)).supported


def test_real_structure_and_equivalences() -> None:
    atoms = _distorted_nacl()
    reference = _feature(atoms)
    rotated = atoms.copy(); rotated.rotate(31, "z", rotate_cell=True)
    translated = atoms.copy(); translated.translate([0.173, 0.291, 0.419]); translated.wrap()
    permuted = atoms[[3, 0, 6, 1, 7, 4, 2, 5]]
    rebased = atoms.copy(); rebased.set_cell(np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]]) @ atoms.cell.array, scale_atoms=False); rebased.wrap()
    for equivalent in (rotated, translated, permuted, rebased, atoms.repeat((2, 1, 1))):
        assert _feature(equivalent) == pytest.approx(reference, abs=1e-8)


def test_geometry_only_firewall_and_row_schema() -> None:
    atoms = _distorted_nacl()
    changed = []
    a = atoms.copy(); a.calc = Calculator(); changed.append(a)
    a = atoms.copy(); a.info["outcome"] = 1; changed.append(a)
    a = atoms.copy(); a.new_array("energy", np.zeros(len(a))); changed.append(a)
    a = atoms.copy(); a.pbc = False; changed.append(a)
    a = atoms.copy(); a.positions[0, 0] = np.nan; changed.append(a)
    for a in changed:
        result = n.compute_psvb_features(a)
        assert not result.supported and "geometry-only Atoms" in str(result.failure_reason)
    row = n.compute_psvb_row(atoms)
    assert row["psvb_supported"] is True
    assert set(row) == {
        n.FEATURE_NAMES[0], "psvb_supported", "psvb_failure", "psvb_site_count",
        "psvb_directed_face_count", "psvb_undirected_edge_count", "psvb_rank3_site_count",
        "psvb_skeleton_edge_count", "psvb_skeleton_threshold",
        "psvb_total_neighbor_pair_count", "psvb_total_bypassed_pair_count",
        "psvb_maximum_reverse_angle_error",
    }
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())
