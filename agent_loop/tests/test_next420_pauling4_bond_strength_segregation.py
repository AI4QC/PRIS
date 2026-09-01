from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next420_pauling4_bond_strength_segregation as n


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
    result = n.compute_p4bss_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[n.FEATURE_NAMES[0]])


def test_frozen_protocol_schema_direction_and_boundary_are_exact() -> None:
    assert n.PROTOCOL == "2026-08-13-next420-pauling4-bond-strength-segregation-v1"
    assert len(n.DESIGN_SHA256) == 64
    assert n.FEATURE_NAMES == ("p4bss_bond_strength_pair_avoidance",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_high"}
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())


def test_kernel_distinguishes_avoidance_uniformity_and_clustering_exactly() -> None:
    strengths = (1.0, 1.0, 3.0, 3.0)
    avoided = n.pauling4_bond_strength_pair_avoidance(
        stub_strengths=strengths,
        anion_stub_indices=((0, 2), (1, 3)),
    )
    uniform = n.pauling4_bond_strength_pair_avoidance(
        stub_strengths=(2.0, 2.0, 2.0, 2.0),
        anion_stub_indices=((0, 1), (2, 3)),
    )
    clustered = n.pauling4_bond_strength_pair_avoidance(
        stub_strengths=strengths,
        anion_stub_indices=((0, 1), (2, 3)),
    )
    assert avoided.supported and uniform.supported and clustered.supported
    assert avoided.expected_product == pytest.approx(4.0)
    assert avoided.observed_product == pytest.approx(3.0)
    assert avoided.features[n.FEATURE_NAMES[0]] == pytest.approx(4.0 / 7.0)
    assert uniform.features[n.FEATURE_NAMES[0]] == pytest.approx(0.5)
    assert clustered.observed_product == pytest.approx(5.0)
    assert clustered.features[n.FEATURE_NAMES[0]] == pytest.approx(4.0 / 9.0)
    assert avoided.features[n.FEATURE_NAMES[0]] > 0.5
    assert clustered.features[n.FEATURE_NAMES[0]] < 0.5


def test_kernel_is_strength_scale_order_and_replication_invariant() -> None:
    strengths = (1.0, 2.0, 4.0, 8.0, 3.0, 6.0)
    groups = ((0, 2, 4), (1, 3, 5))
    reference = n.pauling4_bond_strength_pair_avoidance(
        stub_strengths=strengths, anion_stub_indices=groups
    )
    scaled = n.pauling4_bond_strength_pair_avoidance(
        stub_strengths=tuple(7.3 * value for value in strengths),
        anion_stub_indices=groups,
    )
    permutation = (5, 2, 0, 4, 1, 3)
    inverse = {old: new for new, old in enumerate(permutation)}
    reordered = n.pauling4_bond_strength_pair_avoidance(
        stub_strengths=tuple(strengths[index] for index in permutation),
        anion_stub_indices=tuple(
            tuple(inverse[index] for index in reversed(group)) for group in reversed(groups)
        ),
    )
    replicated = n.pauling4_bond_strength_pair_avoidance(
        stub_strengths=strengths * 4,
        anion_stub_indices=tuple(
            tuple(index + offset * len(strengths) for index in group)
            for offset in range(4)
            for group in groups
        ),
    )
    for result in (reference, scaled, reordered, replicated):
        assert result.supported, result.failure_reason
        assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(
            reference.features[n.FEATURE_NAMES[0]], abs=1e-10
        )


@pytest.mark.parametrize(
    ("strengths", "groups"),
    (
        ((), ()),
        ((1.0,), ((0,),)),
        ((1.0, 2.0), ((0, 2),)),
        ((1.0, 0.0), ((0, 1),)),
        ((1.0, np.nan), ((0, 1),)),
        ((1.0, 2.0), ((0, 0),)),
        ((1.0, 2.0), ((0, 1), (1,))),
    ),
)
def test_malformed_kernel_inputs_fail_closed(strengths: object, groups: object) -> None:
    result = n.pauling4_bond_strength_pair_avoidance(
        stub_strengths=strengths, anion_stub_indices=groups
    )
    assert result.supported is False
    assert result.features == {}


def test_distorted_crystal_and_equivalents_are_supported_and_invariant() -> None:
    atoms = _distorted_nacl()
    result = n.compute_p4bss_features(atoms)
    assert result.supported, result.failure_reason
    assert result.site_count == len(atoms)
    assert result.edge_count >= len(atoms) // 2
    assert result.pair_count > 0
    assert 0 < result.features[n.FEATURE_NAMES[0]] < 1
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
        result = n.compute_p4bss_features(item)
        assert not result.supported
        assert "geometry-only Atoms" in str(result.failure_reason)
    row = n.compute_p4bss_row(atoms)
    assert tuple(name for name in row if name.startswith("p4bss_")) == (
        "p4bss_bond_strength_pair_avoidance",
        "p4bss_supported",
        "p4bss_failure",
        "p4bss_site_count",
        "p4bss_edge_count",
        "p4bss_pair_count",
        "p4bss_expected_product",
        "p4bss_observed_product",
        "p4bss_valence_policy",
    )
    assert row["p4bss_supported"] is True
