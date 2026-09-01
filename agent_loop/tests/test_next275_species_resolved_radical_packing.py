from __future__ import annotations

import inspect

from ase import Atoms
import numpy as np
import pytest

import src.next275_species_resolved_radical_packing as n


def test_species_decomposition_obeys_total_variance_identity() -> None:
    result = n.species_variance_features(
        values=[1.0, 3.0, 10.0, 14.0],
        species=["A", "A", "B", "B"],
    )
    # W = 2 + 8 = 10, B = 4*(12-2)^2/4 = 100, T = 110.
    assert result["within_cv"] == pytest.approx(np.sqrt(10.0 / 4.0) / 7.0)
    assert result["between_cv"] == pytest.approx(np.sqrt(100.0 / 4.0) / 7.0)
    assert result["within_variance_fraction"] == pytest.approx(10.0 / 110.0)
    expected_weighted = 0.5 * (1.0 / 2.0) + 0.5 * (2.0 / 12.0)
    assert result["weighted_species_cv"] == pytest.approx(expected_weighted)
    assert result["max_species_cv"] == pytest.approx(0.5)


def test_singleton_species_have_zero_within_dispersion() -> None:
    result = n.species_variance_features(
        values=[1.0, 2.0, 4.0], species=["A", "B", "C"]
    )
    assert result["within_cv"] == 0.0
    assert result["within_variance_fraction"] == 0.0
    assert result["weighted_species_cv"] == 0.0
    assert result["max_species_cv"] == 0.0
    assert result["between_cv"] > 0.0


@pytest.mark.parametrize(
    ("values", "species"),
    [([1.0, np.nan], ["A", "A"]), ([1.0, -1.0], ["A", "A"]), ([1.0], []), ([], [])],
)
def test_species_decomposition_refuses_invalid_population(values, species) -> None:
    with pytest.raises(ValueError, match="population differs"):
        n.species_variance_features(values=values, species=species)


def _nacl() -> Atoms:
    return Atoms(
        symbols=["Na", "Cl"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=np.eye(3) * 5.6,
        pbc=True,
    )


def test_real_structure_features_are_finite_and_replication_invariant() -> None:
    atoms = _nacl()
    first = n.compute_species_resolved_prv_features(atoms)
    repeated = n.compute_species_resolved_prv_features(atoms.repeat((2, 1, 1)))
    assert first.supported and repeated.supported
    assert tuple(first.features) == n.FEATURE_NAMES
    assert np.isfinite(list(first.features.values())).all()
    np.testing.assert_allclose(
        [first.features[name] for name in n.FEATURE_NAMES],
        [repeated.features[name] for name in n.FEATURE_NAMES],
        rtol=0.0,
        atol=2.0e-10,
    )


def test_real_structure_features_are_translation_and_permutation_invariant() -> None:
    atoms = _nacl()
    reference = n.compute_species_resolved_prv_features(atoms)
    shifted = atoms.copy()
    shifted.set_scaled_positions(shifted.get_scaled_positions() + [0.31, -0.17, 0.23])
    permuted = atoms[[1, 0]]
    for candidate in (shifted, permuted):
        got = n.compute_species_resolved_prv_features(candidate)
        assert got.supported
        np.testing.assert_allclose(
            [reference.features[name] for name in n.FEATURE_NAMES],
            [got.features[name] for name in n.FEATURE_NAMES],
            rtol=0.0,
            atol=2.0e-10,
        )


def test_builder_interface_has_no_endpoint_validation_or_replication_input() -> None:
    parameters = tuple(inspect.signature(n.build_species_resolved_prv_features).parameters)
    assert parameters == (
        "scigen_cohort_dir",
        "wyformer_cohort_dir",
        "design_path",
        "output_dir",
        "workers",
        "require_formal_inputs",
    )


def test_builder_fails_closed_on_missing_inputs(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="NEXT275 input is missing"):
        n.build_species_resolved_prv_features(
            scigen_cohort_dir=tmp_path / "scigen",
            wyformer_cohort_dir=tmp_path / "wyformer",
            design_path=tmp_path / "design",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
