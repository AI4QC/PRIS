from __future__ import annotations

import inspect

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

from src.next319_periodic_contact_shell_neutralization import (
    FEATURE_DIRECTIONS,
    FEATURE_NAMES,
    MAX_COVER_VERTICES_PER_ROOT,
    PROTOCOL,
    build_cross_source_pcsn_features,
    compute_pcsn_features,
    contact_shell_neutralization_features,
)


def _chain_contacts() -> np.ndarray:
    return np.asarray(
        (
            (0, 1, 0, 0, 0),
            (1, 0, 0, 0, 0),
            (0, 1, -1, 0, 0),
            (1, 0, 1, 0, 0),
        ),
        dtype=int,
    )


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
    result = compute_pcsn_features(atoms)
    assert result.supported, result.failure_reason
    return np.asarray([result.features[name] for name in FEATURE_NAMES], dtype=float)


def test_protocol_feature_directions_and_guard_are_exact() -> None:
    assert PROTOCOL == "2026-08-09-next319-periodic-contact-shell-neutralization-v1"
    assert FEATURE_NAMES == (
        "pcsn_shell1_residual_q90",
        "pcsn_shell2_residual_q90",
    )
    assert FEATURE_DIRECTIONS == {name: "protected_low" for name in FEATURE_NAMES}
    assert MAX_COVER_VERTICES_PER_ROOT == 4096


def test_periodic_two_site_chain_has_exact_shell_residuals() -> None:
    result = contact_shell_neutralization_features(
        charges=np.asarray((1.0, -1.0)),
        directed_contacts=_chain_contacts(),
    )
    assert result.supported, result.failure_reason
    assert result.site_count == 2
    assert result.directed_contact_count == result.unique_contact_count == 4
    assert result.minimum_shell1_population == result.maximum_shell1_population == 3
    assert result.minimum_shell2_population == result.maximum_shell2_population == 5
    np.testing.assert_allclose(
        [result.features[name] for name in FEATURE_NAMES],
        (1.0 / 3.0, 1.0 / 5.0),
        rtol=0.0,
        atol=5.1e-11,
    )


def test_contact_order_and_duplicate_incidences_do_not_change_unique_shells() -> None:
    contacts = _chain_contacts()
    reference = contact_shell_neutralization_features(
        charges=np.asarray((1.0, -1.0)), directed_contacts=contacts
    )
    reordered = contact_shell_neutralization_features(
        charges=np.asarray((1.0, -1.0)),
        directed_contacts=contacts[[2, 0, 3, 1]],
    )
    duplicated = contact_shell_neutralization_features(
        charges=np.asarray((1.0, -1.0)),
        directed_contacts=np.repeat(contacts, 2, axis=0),
    )
    assert reference.supported and reordered.supported and duplicated.supported
    assert duplicated.directed_contact_count == 8
    assert duplicated.unique_contact_count == 4
    for result in (reordered, duplicated):
        np.testing.assert_array_equal(
            [result.features[name] for name in FEATURE_NAMES],
            [reference.features[name] for name in FEATURE_NAMES],
        )


@pytest.mark.parametrize(
    ("charges", "contacts"),
    (
        (np.asarray((1.0,)), np.asarray(((0, 0, 1, 0, 0), (0, 0, -1, 0, 0)))),
        (np.asarray((1.0, -0.5)), _chain_contacts()),
        (np.asarray((1.0, np.nan)), _chain_contacts()),
        (np.asarray((1.0, -1.0)), np.asarray(((0, 1, 0, 0, 0),))),
        (
            np.asarray((1.0, -1.0)),
            np.asarray(((0, 1, 0, 0, 0), (1, 0, 1, 0, 0))),
        ),
        (
            np.asarray((1.0, -1.0)),
            np.asarray(((0, 2, 0, 0, 0), (2, 0, 0, 0, 0))),
        ),
        (
            np.asarray((1.0, -1.0)),
            np.asarray(((0.0, 1.0, 0.5, 0.0, 0.0), (1.0, 0.0, -0.5, 0.0, 0.0))),
        ),
        (np.asarray((1.0, -1.0)), np.empty((0, 5), dtype=int)),
    ),
)
def test_kernel_fails_open_on_invalid_charge_or_translated_contacts(
    charges, contacts
) -> None:
    result = contact_shell_neutralization_features(
        charges=charges, directed_contacts=contacts
    )
    assert result.supported is False
    assert result.features == {}


def test_kernel_refuses_cover_population_above_frozen_guard() -> None:
    shifts = np.arange(MAX_COVER_VERTICES_PER_ROOT, dtype=int)
    forward = np.column_stack(
        (
            np.zeros_like(shifts),
            np.ones_like(shifts),
            shifts,
            np.zeros_like(shifts),
            np.zeros_like(shifts),
        )
    )
    reverse = np.column_stack(
        (
            np.ones_like(shifts),
            np.zeros_like(shifts),
            -shifts,
            np.zeros_like(shifts),
            np.zeros_like(shifts),
        )
    )
    result = contact_shell_neutralization_features(
        charges=np.asarray((1.0, -1.0)),
        directed_contacts=np.vstack((forward, reverse)),
    )
    assert result.supported is False
    assert "4,096" in str(result.failure_reason)


def test_standard_and_distorted_ionic_crystals_are_finite() -> None:
    cases = (
        bulk("NaCl", "rocksalt", a=5.64, cubic=True),
        bulk("CsCl", "cesiumchloride", a=4.12, cubic=True),
        bulk("ZnS", "zincblende", a=5.41, cubic=True),
        _distorted_nacl(),
    )
    for atoms in cases:
        result = compute_pcsn_features(atoms)
        assert result.supported, result.failure_reason
        values = np.asarray([result.features[name] for name in FEATURE_NAMES])
        assert np.isfinite(values).all()
        assert ((0.0 <= values) & (values <= 1.0)).all()
        assert result.minimum_shell1_population >= 2
        assert result.maximum_shell2_population <= MAX_COVER_VERTICES_PER_ROOT
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
            _feature_vector(equivalent), reference, rtol=0.0, atol=1.0e-10
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
        result = compute_pcsn_features(changed)
        assert result.supported is False
        assert "geometry-only Atoms" in str(result.failure_reason)


def test_unsupported_chemistry_fails_open() -> None:
    result = compute_pcsn_features(bulk("Cu", "fcc", a=3.61, cubic=True))
    assert result.supported is False
    assert result.features == {}
    assert result.site_count == result.directed_contact_count == 0


def test_builder_interface_is_discovery_geometry_only() -> None:
    parameters = tuple(inspect.signature(build_cross_source_pcsn_features).parameters)
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
    with pytest.raises(FileNotFoundError, match="NEXT319 input is missing"):
        build_cross_source_pcsn_features(
            scigen_cohort_dir=tmp_path / "scigen",
            wyformer_cohort_dir=tmp_path / "wyformer",
            design_path=tmp_path / "design",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
