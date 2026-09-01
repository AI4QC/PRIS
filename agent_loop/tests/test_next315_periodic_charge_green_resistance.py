from __future__ import annotations

import inspect
import math

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

from src.next315_periodic_charge_green_resistance import (
    FEATURE_DIRECTIONS,
    FEATURE_NAMES,
    PROTOCOL,
    build_cross_source_pcgr_features,
    charge_green_resistance_features,
    compute_pcgr_features,
)


def _reciprocal(edges: list[tuple[int, int]]) -> np.ndarray:
    return np.asarray([*edges, *((right, left) for left, right in edges)], dtype=int)


def _distorted_nacl() -> Atoms:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.set_cell(
        np.asarray(
            [[5.64, 0.0, 0.0], [0.27, 5.77, 0.0], [0.18, 0.31, 5.53]]
        ),
        scale_atoms=True,
    )
    atoms.positions[1] += np.asarray([0.08, -0.04, 0.06])
    atoms.wrap()
    return atoms


def _feature_vector(atoms: Atoms) -> np.ndarray:
    result = compute_pcgr_features(atoms)
    assert result.supported, result.failure_reason
    return np.asarray([result.features[name] for name in FEATURE_NAMES], dtype=float)


def test_protocol_feature_and_direction_universes_are_exact() -> None:
    assert PROTOCOL == "2026-08-09-next315-periodic-charge-green-resistance-v1"
    assert FEATURE_NAMES == (
        "pcgr_charge_resistance",
        "pcgr_voltage_drop_q90",
        "pcgr_voltage_drop_max",
    )
    assert FEATURE_DIRECTIONS == {name: "protected_low" for name in FEATURE_NAMES}


def test_two_site_graph_has_exact_green_resistance_and_voltage_drop() -> None:
    result = charge_green_resistance_features(
        charges=np.asarray((1.0, -1.0)),
        directed_contacts=np.asarray(((0, 1), (1, 0))),
    )
    assert result.supported, result.failure_reason
    assert result.site_count == 2
    assert result.directed_contact_count == result.intersite_contact_count == 2
    assert result.positive_mode_count == 1
    assert result.nullity == 1
    assert result.minimum_positive_eigenvalue == result.maximum_eigenvalue == 2.0
    np.testing.assert_allclose(
        [result.features[name] for name in FEATURE_NAMES],
        (0.5, 1.0, 1.0),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_charge_matched_star_has_exact_green_solution() -> None:
    result = charge_green_resistance_features(
        charges=np.asarray((2.0, -1.0, -1.0)),
        directed_contacts=_reciprocal([(0, 1), (0, 2)]),
    )
    assert result.supported, result.failure_reason
    assert result.positive_mode_count == 2
    assert result.nullity == 1
    assert result.minimum_positive_eigenvalue == 1.0
    assert result.maximum_eigenvalue == 3.0
    np.testing.assert_allclose(
        [result.features[name] for name in FEATURE_NAMES],
        (1.0 / 3.0, 1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)),
        rtol=0.0,
        atol=5.1e-11,
    )


def test_self_image_contacts_cancel_from_laplacian_and_drop_population() -> None:
    charges = np.asarray((2.0, -1.0, -1.0))
    contacts = _reciprocal([(0, 1), (0, 2)])
    reference = charge_green_resistance_features(
        charges=charges, directed_contacts=contacts
    )
    with_self_images = charge_green_resistance_features(
        charges=charges,
        directed_contacts=np.vstack((contacts, ((0, 0), (1, 1), (2, 2)))),
    )
    assert reference.supported, reference.failure_reason
    assert with_self_images.supported, with_self_images.failure_reason
    assert with_self_images.directed_contact_count == len(contacts) + 3
    assert with_self_images.intersite_contact_count == len(contacts)
    np.testing.assert_array_equal(
        [with_self_images.features[name] for name in FEATURE_NAMES],
        [reference.features[name] for name in FEATURE_NAMES],
    )


def test_kernel_is_contact_order_and_balanced_replication_invariant() -> None:
    charges = np.asarray((2.0, -1.0, -1.0))
    contacts = _reciprocal([(0, 1), (0, 2), (1, 2)])
    reference = charge_green_resistance_features(
        charges=charges, directed_contacts=contacts
    )
    reordered = charge_green_resistance_features(
        charges=charges,
        directed_contacts=contacts[[3, 0, 5, 1, 4, 2]],
    )
    replicated = charge_green_resistance_features(
        charges=np.concatenate((charges, charges)),
        directed_contacts=np.vstack((contacts, contacts + len(charges))),
    )
    assert reference.supported, reference.failure_reason
    for result in (reordered, replicated):
        assert result.supported, result.failure_reason
        np.testing.assert_array_equal(
            [result.features[name] for name in FEATURE_NAMES],
            [reference.features[name] for name in FEATURE_NAMES],
        )


def test_parallel_contact_multiplicity_is_part_of_frozen_conductance() -> None:
    charges = np.asarray((1.0, -1.0))
    contacts = np.asarray(((0, 1), (1, 0)))
    single = charge_green_resistance_features(
        charges=charges, directed_contacts=contacts
    )
    doubled = charge_green_resistance_features(
        charges=charges, directed_contacts=np.repeat(contacts, 2, axis=0)
    )
    assert single.supported and doubled.supported
    np.testing.assert_allclose(
        [doubled.features[name] for name in FEATURE_NAMES],
        0.5 * np.asarray([single.features[name] for name in FEATURE_NAMES]),
        rtol=0.0,
        atol=5.1e-11,
    )


@pytest.mark.parametrize(
    ("charges", "contacts"),
    (
        (np.asarray((1.0,)), np.asarray(((0, 0),))),
        (np.asarray((1.0, -0.5)), np.asarray(((0, 1), (1, 0)))),
        (np.asarray((1.0, -1.0)), np.asarray(((0, 1),))),
        (np.asarray((1.0, -1.0)), np.asarray(((0, 2), (2, 0)))),
        (np.asarray((1.0, np.nan)), np.asarray(((0, 1), (1, 0)))),
        (np.asarray((1.0, -1.0)), np.asarray(((0, 0), (1, 1)))),
        (np.asarray((1.0, -1.0)), np.empty((0, 2), dtype=int)),
        (
            np.asarray((1.0, -1.0, 1.0, -1.0)),
            _reciprocal([(0, 2), (1, 3)]),
        ),
    ),
)
def test_kernel_fails_open_on_invalid_or_unsolvable_inputs(charges, contacts) -> None:
    result = charge_green_resistance_features(
        charges=charges, directed_contacts=contacts
    )
    assert result.supported is False
    assert result.features == {}


def test_standard_and_distorted_ionic_crystals_are_finite() -> None:
    cases = (
        bulk("NaCl", "rocksalt", a=5.64, cubic=True),
        bulk("CsCl", "cesiumchloride", a=4.12, cubic=True),
        bulk("ZnS", "zincblende", a=5.41, cubic=True),
        _distorted_nacl(),
    )
    for atoms in cases:
        result = compute_pcgr_features(atoms)
        assert result.supported, result.failure_reason
        values = np.asarray([result.features[name] for name in FEATURE_NAMES])
        assert np.isfinite(values).all()
        assert (values >= 0.0).all()
        assert result.intersite_contact_count > 0
        assert result.positive_mode_count > 0
        assert result.nullity > 0
        assert result.maximum_eigenvalue >= result.minimum_positive_eigenvalue > 0.0
        assert result.valence_policy


def test_geometry_equivalences_preserve_features() -> None:
    atoms = _distorted_nacl()
    reference = _feature_vector(atoms)
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
        np.testing.assert_allclose(
            _feature_vector(equivalent), reference, rtol=0.0, atol=1.0e-8
        )


def test_geometry_boundary_fails_closed() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
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
    for changed in (
        with_calculator,
        with_metadata,
        with_array,
        nonperiodic,
        nonfinite,
    ):
        result = compute_pcgr_features(changed)
        assert result.supported is False
        assert "geometry-only Atoms" in str(result.failure_reason)


def test_unsupported_chemistry_fails_open() -> None:
    result = compute_pcgr_features(bulk("Cu", "fcc", a=3.61, cubic=True))
    assert result.supported is False
    assert result.features == {}
    assert result.site_count == result.directed_contact_count == 0


def test_builder_interface_is_discovery_geometry_only() -> None:
    parameters = tuple(inspect.signature(build_cross_source_pcgr_features).parameters)
    assert parameters == (
        "scigen_cohort_dir",
        "wyformer_cohort_dir",
        "design_path",
        "output_dir",
        "workers",
        "require_formal_inputs",
    )
    assert not any(
        token in name
        for name in parameters
        for token in ("endpoint", "label", "validation", "replication", "relax")
    )


def test_builder_fails_closed_on_missing_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="NEXT315 input is missing"):
        build_cross_source_pcgr_features(
            scigen_cohort_dir=tmp_path / "scigen",
            wyformer_cohort_dir=tmp_path / "wyformer",
            design_path=tmp_path / "design",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
