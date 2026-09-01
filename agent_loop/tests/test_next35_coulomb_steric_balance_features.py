from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.core.operations import SymmOp
from pymatgen.io.ase import AseAtomsAdaptor

from src.next11_geometry_only_frames import _canonical_frame
from src.next32_omat24_cohort import PROTOCOL as COHORT_PROTOCOL
from src.next34_analytic_field_features import (
    FEATURE_NAME as NEXT34_FEATURE_NAME,
    PROTOCOL as NEXT34_FEATURE_PROTOCOL,
)
from src.next35_coulomb_steric_balance_features import (
    CANDIDATE_FEATURE_NAMES,
    DIAGNOSTIC_FEATURE_NAMES,
    FEATURE_NAME,
    MANIFEST_NAME,
    REUSED_FEATURE_NAMES,
    analytic_vector_balance_features,
    build_coulomb_steric_balance_feature_batch,
    compute_coulomb_steric_balance_features,
)


def _opposed() -> tuple[np.ndarray, np.ndarray]:
    coulomb = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    return coulomb, -coulomb


def test_exact_opposition_has_zero_balance_risk_and_unit_optimal_scale() -> None:
    coulomb, steric = _opposed()

    result = analytic_vector_balance_features(coulomb, steric)

    assert result.supported
    assert tuple(result.features) == CANDIDATE_FEATURE_NAMES + DIAGNOSTIC_FEATURE_NAMES
    for name in CANDIDATE_FEATURE_NAMES:
        assert result.features[name] == pytest.approx(0.0, abs=1e-12)
    assert result.features["acsb_optimal_repulsion_scale"] == pytest.approx(1.0)
    assert result.features["acsb_joint_active_site_fraction"] == pytest.approx(1.0)


def test_aligned_or_single_active_fields_have_maximal_balance_deficit() -> None:
    coulomb, _steric = _opposed()
    aligned = analytic_vector_balance_features(coulomb, coulomb)
    single = analytic_vector_balance_features(coulomb, np.zeros_like(coulomb))
    zero = analytic_vector_balance_features(
        np.zeros_like(coulomb), np.zeros_like(coulomb)
    )

    for result in (aligned, single):
        assert result.supported
        assert result.features["acsb_opposition_deficit"] == pytest.approx(1.0)
        assert result.features["acsb_global_residual"] == pytest.approx(1.0)
        assert result.features["acsb_site_residual_rms"] == pytest.approx(1.0)
        assert result.features["acsb_active_disagreement_fraction"] == pytest.approx(1.0)
    assert zero.supported
    assert all(zero.features[name] == 0.0 for name in CANDIDATE_FEATURE_NAMES)


def test_balance_is_invariant_to_independent_scale_rotation_permutation_and_replication() -> None:
    coulomb = np.array([[1.0, 2.0, 0.0], [-0.5, -1.0, 0.5], [-0.5, -1.0, -0.5]])
    steric = np.array([[-2.0, -3.0, 0.5], [0.8, 1.7, -0.4], [1.2, 1.3, -0.1]])
    angle = np.deg2rad(37.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    order = [2, 0, 1]
    reference = analytic_vector_balance_features(coulomb, steric)
    variants = [
        analytic_vector_balance_features(7.3 * coulomb, 0.21 * steric),
        analytic_vector_balance_features(coulomb @ rotation.T, steric @ rotation.T),
        analytic_vector_balance_features(coulomb[order], steric[order]),
        analytic_vector_balance_features(
            np.concatenate([coulomb, coulomb]), np.concatenate([steric, steric])
        ),
    ]

    assert reference.supported
    for variant in variants:
        assert variant.supported
        for name in CANDIDATE_FEATURE_NAMES:
            assert variant.features[name] == pytest.approx(
                reference.features[name], rel=1e-12, abs=1e-12
            )


def test_invalid_arrays_fail_open_and_schema_has_no_forbidden_tokens() -> None:
    invalid = analytic_vector_balance_features(np.ones((2, 3)), np.ones((3, 3)))

    assert not invalid.supported
    assert invalid.features == {}
    assert not any(
        token in name.lower()
        for name in CANDIDATE_FEATURE_NAMES + DIAGNOSTIC_FEATURE_NAMES
        for token in (
            "energy",
            "force",
            "stress",
            "dft",
            "relax",
            "label",
            "target",
            "mattersim",
            "mlip",
        )
    )


def _compressed_b2(displacement: float = 0.04) -> tuple[Structure, list[float]]:
    return (
        Structure(
            Lattice.cubic(3.0),
            ["Na", "Cl"],
            [[0.0, 0.0, 0.0], [0.5 + displacement, 0.5, 0.5]],
        ),
        [1.0, -1.0],
    )


def _charges(structure: Structure) -> list[float]:
    return [1.0 if site.specie.symbol == "Na" else -1.0 for site in structure]


def test_structure_balance_is_representation_invariant_and_non_neutral_fails_open() -> None:
    structure, charges = _compressed_b2()
    translated = structure.copy()
    translated.translate_sites(range(len(translated)), [0.11, -0.17, 0.09], frac_coords=True)
    rotated = structure.copy()
    rotated.apply_operation(
        SymmOp.from_axis_angle_and_translation([0.0, 1.0, 0.0], 23.0),
        fractional=False,
    )
    permuted = Structure(
        structure.lattice,
        [site.specie for site in reversed(structure)],
        [site.frac_coords for site in reversed(structure)],
    )
    repeated = structure.copy()
    repeated.make_supercell([2, 1, 1])
    reference = compute_coulomb_steric_balance_features(structure, charges)
    variants = [
        compute_coulomb_steric_balance_features(translated, charges),
        compute_coulomb_steric_balance_features(rotated, charges),
        compute_coulomb_steric_balance_features(permuted, list(reversed(charges))),
        compute_coulomb_steric_balance_features(repeated, _charges(repeated)),
    ]

    assert reference.supported, reference.failure_reason
    for variant in variants:
        assert variant.supported, variant.failure_reason
        assert variant.features == pytest.approx(reference.features, rel=2e-5, abs=1e-8)
    bad = compute_coulomb_steric_balance_features(structure, [1.0, 1.0])
    assert not bad.supported
    assert "neutral" in str(bad.failure_reason).lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _batch_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    material_id = "unit::compressed-b2"
    structure, _charges_unused = _compressed_b2()
    atoms = AseAtomsAdaptor.get_atoms(structure)
    archive_path = tmp_path / "geometry_only_frames.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        info = zipfile.ZipInfo(f"{material_id}.extxyz", (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, _canonical_frame(atoms))
    metadata_path = tmp_path / "next32_cohort.parquet"
    pd.DataFrame(
        {
            "material_id": [material_id],
            "source_name": ["unit"],
            "sid": ["must-not-be-read"],
            "parent_id": ["parent-b2"],
            "record_key": [1],
            "natoms": [2],
            "input_role": ["unrelaxed_x0_geometry_only"],
        }
    ).to_parquet(metadata_path, index=False)
    cohort_manifest_path = tmp_path / "cohort-manifest.json"
    cohort_manifest_path.write_text(
        json.dumps(
            {
                "protocol": COHORT_PROTOCOL,
                "output_role": "unrelaxed_x0_geometry_only",
                "endpoint_numeric_fields_parsed": False,
                "label_values_exported": False,
                "labels_opened": False,
                "outputs_sha256": {
                    archive_path.name: _sha256(archive_path),
                    metadata_path.name: _sha256(metadata_path),
                },
            }
        )
        + "\n"
    )
    upstream_path = tmp_path / NEXT34_FEATURE_NAME
    pd.DataFrame(
        {
            "material_id": [material_id],
            "source_name": ["unit"],
            "parent_id": ["parent-b2"],
            "natoms": [2],
            "aefi_residual_max": [0.4],
            "steric_rep12_vector_rms": [0.3],
            "steric_rep12_vector_max": [0.5],
            "sivr_site_imbalance_rms": [0.2],
        }
    ).to_parquet(upstream_path, index=False)
    upstream_manifest_path = tmp_path / "next34-feature-manifest.json"
    upstream_manifest_path.write_text(
        json.dumps(
            {
                "protocol": NEXT34_FEATURE_PROTOCOL,
                "input_role": "unrelaxed_x0_geometry_only",
                "labels_opened": False,
                "endpoint_fields_read": False,
                "dft_values_used": False,
                "classical_analytic_electrostatics_used": True,
                "electronic_structure_calculation_used": False,
                "model_or_proxy_potential_used": False,
                "coordinates_or_cell_modified": False,
                "outputs_sha256": {NEXT34_FEATURE_NAME: _sha256(upstream_path)},
            }
        )
        + "\n"
    )
    return archive_path, metadata_path, cohort_manifest_path, upstream_path, upstream_manifest_path


def test_batch_is_hash_locked_label_free_and_no_replace(tmp_path: Path) -> None:
    archive, metadata, cohort_manifest, upstream, upstream_manifest = _batch_fixture(tmp_path)
    output = tmp_path / "next35-features"

    manifest = build_coulomb_steric_balance_feature_batch(
        archive_path=archive,
        metadata_path=metadata,
        cohort_manifest_path=cohort_manifest,
        next34_feature_path=upstream,
        next34_feature_manifest_path=upstream_manifest,
        output_dir=output,
    )

    frame = pd.read_parquet(output / FEATURE_NAME)
    assert frame.material_id.tolist() == ["unit::compressed-b2"]
    assert frame.acsb_supported.tolist() == [True]
    assert set(CANDIDATE_FEATURE_NAMES + DIAGNOSTIC_FEATURE_NAMES).issubset(frame)
    assert set(REUSED_FEATURE_NAMES).issubset(frame)
    assert "sid" not in frame
    assert manifest["labels_opened"] is False
    assert manifest["dft_values_used"] is False
    assert manifest["coordinates_or_cell_modified"] is False
    assert manifest["outputs_sha256"][FEATURE_NAME] == _sha256(output / FEATURE_NAME)
    assert json.loads((output / MANIFEST_NAME).read_text()) == manifest
    with pytest.raises(FileExistsError):
        build_coulomb_steric_balance_feature_batch(
            archive_path=archive,
            metadata_path=metadata,
            cohort_manifest_path=cohort_manifest,
            next34_feature_path=upstream,
            next34_feature_manifest_path=upstream_manifest,
            output_dir=output,
        )


@pytest.mark.parametrize("corruption", ["cohort_labels", "upstream_labels", "hash"])
def test_batch_rejects_boundary_crossing(tmp_path: Path, corruption: str) -> None:
    archive, metadata, cohort_manifest, upstream, upstream_manifest = _batch_fixture(tmp_path)
    target = cohort_manifest if corruption == "cohort_labels" else upstream_manifest
    value = json.loads(target.read_text())
    if corruption in {"cohort_labels", "upstream_labels"}:
        value["labels_opened"] = True
    else:
        value["outputs_sha256"][upstream.name] = "0" * 64
    target.write_text(json.dumps(value) + "\n")

    with pytest.raises(ValueError, match="label-free|hash"):
        build_coulomb_steric_balance_feature_batch(
            archive_path=archive,
            metadata_path=metadata,
            cohort_manifest_path=cohort_manifest,
            next34_feature_path=upstream,
            next34_feature_manifest_path=upstream_manifest,
            output_dir=tmp_path / "next35-features",
        )
