from __future__ import annotations

import inspect
import math

from ase import Atoms
import numpy as np
import pytest

import src.next283_power_cell_shape_volume_coupling as n


def _nacl() -> Atoms:
    return Atoms(
        symbols=["Na", "Cl"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=np.eye(3) * 5.6,
        pbc=True,
    )


def _asymmetric() -> Atoms:
    return Atoms(
        symbols=["Si", "O", "Na"],
        scaled_positions=[
            (0.10, 0.20, 0.30),
            (0.52, 0.63, 0.47),
            (0.81, 0.24, 0.72),
        ],
        cell=[(4.1, 0.0, 0.0), (0.7, 4.4, 0.0), (0.4, 0.8, 4.8)],
        pbc=True,
    )


def test_feature_universe_and_directions_are_exactly_frozen() -> None:
    assert n.FEATURE_NAMES == (
        "psvc_sphericity_mean",
        "psvc_sphericity_q10",
        "psvc_log_volume_asphericity_correlation",
        "psvc_inflated_asphericity_mean",
        "psvc_inflated_asphericity_q90",
        "psvc_small_inflated_asphericity_mean",
    )
    assert n.FEATURE_DIRECTIONS == {
        "psvc_sphericity_mean": "protected_high",
        "psvc_sphericity_q10": "protected_high",
        "psvc_log_volume_asphericity_correlation": "protected_low",
        "psvc_inflated_asphericity_mean": "protected_low",
        "psvc_inflated_asphericity_q90": "protected_low",
        "psvc_small_inflated_asphericity_mean": "protected_low",
    }


def test_sphericity_matches_sphere_and_analytic_cube() -> None:
    assert n.cell_sphericity(4.0 * math.pi / 3.0, 4.0 * math.pi) == pytest.approx(1.0)
    expected_cube = math.pi ** (1.0 / 3.0) * 6.0 ** (2.0 / 3.0) / 6.0
    assert n.cell_sphericity(1.0, 6.0) == pytest.approx(expected_cube)
    with pytest.raises(ValueError, match="volume and area differ"):
        n.cell_sphericity(0.0, 1.0)


def test_shape_volume_summaries_match_analytic_population() -> None:
    radii = np.asarray([0.5, 1.0, 2.0])
    log_ratios = np.asarray([2.0, 1.0, 0.0])
    sphericities = np.asarray([0.6, 0.8, 1.0])
    sphere_volumes = (4.0 * math.pi / 3.0) * radii**3
    volumes = sphere_volumes * np.exp(log_ratios)
    areas = math.pi ** (1.0 / 3.0) * (6.0 * volumes) ** (2.0 / 3.0) / sphericities

    result = n.shape_volume_summaries(
        volumes=volumes, surface_areas=areas, radii=radii
    )

    assert result["psvc_sphericity_mean"] == pytest.approx(0.8)
    assert result["psvc_sphericity_q10"] == pytest.approx(0.6)
    assert result["psvc_log_volume_asphericity_correlation"] == pytest.approx(1.0)
    assert result["psvc_inflated_asphericity_mean"] == pytest.approx(0.4 / 3.0)
    assert result["psvc_inflated_asphericity_q90"] == pytest.approx(0.4)
    assert result["psvc_small_inflated_asphericity_mean"] == pytest.approx(0.2)


def test_constant_log_volume_population_has_exact_zero_correlation_and_burden() -> None:
    radii = np.asarray([0.8, 1.0, 1.2])
    volumes = (4.0 * math.pi / 3.0) * radii**3 * math.e
    sphericities = np.asarray([0.7, 0.8, 0.9])
    areas = math.pi ** (1.0 / 3.0) * (6.0 * volumes) ** (2.0 / 3.0) / sphericities
    result = n.shape_volume_summaries(
        volumes=volumes, surface_areas=areas, radii=radii
    )
    assert result["psvc_log_volume_asphericity_correlation"] == 0.0
    assert result["psvc_inflated_asphericity_mean"] == 0.0
    assert result["psvc_inflated_asphericity_q90"] == 0.0
    assert result["psvc_small_inflated_asphericity_mean"] == 0.0


def test_new_cells_reproduce_next267_volumes_and_have_surface_certificates() -> None:
    atoms = _nacl()
    radii = np.asarray([n.n267._tabulated_radius(symbol) for symbol in atoms.symbols])
    cells = n.periodic_shape_volume_cells(atoms, radii=radii)
    legacy = n.n267.periodic_radical_cells(atoms, radii=radii)
    assert len(cells) == len(legacy) == 2
    np.testing.assert_allclose(
        [cell.volume for cell in cells],
        [cell.volume for cell in legacy],
        rtol=0.0,
        atol=1.0e-10,
    )
    assert all(cell.surface_area > 0.0 for cell in cells)
    assert all(0.0 < cell.sphericity <= 1.0 for cell in cells)
    assert all(cell.facet_count >= 4 for cell in cells)


def _feature_vector(atoms: Atoms) -> np.ndarray:
    result = n.compute_power_cell_shape_volume_features(atoms)
    assert result.supported, result.failure_reason
    assert tuple(result.features) == n.FEATURE_NAMES
    assert result.minimum_surface_area > 0.0
    assert 0.0 < result.minimum_sphericity <= result.maximum_sphericity <= 1.0
    return np.asarray([result.features[name] for name in n.FEATURE_NAMES])


def test_real_features_are_representation_invariant() -> None:
    atoms = _asymmetric()
    reference = _feature_vector(atoms)

    angle = 0.731
    axis = np.asarray([1.0, 2.0, -1.0], dtype=float)
    axis /= np.linalg.norm(axis)
    cross = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = (
        math.cos(angle) * np.eye(3)
        + (1.0 - math.cos(angle)) * np.outer(axis, axis)
        + math.sin(angle) * cross
    )
    rotated = atoms.copy()
    rotated.positions = rotated.positions @ rotation.T
    rotated.set_cell(rotated.cell.array @ rotation.T, scale_atoms=False)

    translated = atoms.copy()
    translated.positions += [1.37, -0.62, 0.91]
    translated.wrap()

    permuted = atoms[[2, 0, 1]]

    rebased = atoms.copy()
    operation = np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
    rebased.set_cell(operation @ atoms.cell.array, scale_atoms=False)
    rebased.wrap()

    replicated = atoms.repeat((2, 1, 1))
    for equivalent in (rotated, translated, permuted, rebased, replicated):
        np.testing.assert_allclose(
            _feature_vector(equivalent), reference, rtol=0.0, atol=2.0e-9
        )


def test_one_site_structure_is_supported() -> None:
    atoms = Atoms("Cu", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 3.6, pbc=True)
    result = n.compute_power_cell_shape_volume_features(atoms)
    assert result.supported, result.failure_reason
    assert result.features["psvc_log_volume_asphericity_correlation"] == 0.0
    assert result.features["psvc_inflated_asphericity_mean"] == 0.0


def test_malformed_geometry_fails_closed() -> None:
    atoms = _nacl()
    atoms.pbc = False
    result = n.compute_power_cell_shape_volume_features(atoms)
    assert not result.supported and "periodic" in str(result.failure_reason)


def test_builder_interface_excludes_endpoint_validation_and_replication() -> None:
    assert tuple(inspect.signature(n.build_power_cell_shape_volume_features).parameters) == (
        "scigen_cohort_dir",
        "wyformer_cohort_dir",
        "design_path",
        "amendment_path",
        "output_dir",
        "workers",
        "require_formal_inputs",
    )


def test_builder_fails_closed_on_missing_inputs(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="NEXT283 input is missing"):
        n.build_power_cell_shape_volume_features(
            scigen_cohort_dir=tmp_path / "scigen",
            wyformer_cohort_dir=tmp_path / "wyformer",
            design_path=tmp_path / "design",
            amendment_path=tmp_path / "amendment",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
