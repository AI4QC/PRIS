from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from ase import Atoms
import numpy as np
import pandas as pd
import pytest

from src.next11_geometry_only_frames import _canonical_frame
from src.next33_symmetry_steric_features import (
    FEATURE_NAME,
    MANIFEST_NAME,
    REUSED_NEXT32_FEATURE_NAMES,
    STERIC_FEATURE_NAMES,
    SYMMETRY_FEATURE_NAMES,
    build_symmetry_steric_feature_batch,
    compute_directional_steric_features,
    compute_symmetry_recovery_features,
)
from src.next32_inorganic_response_features import (
    FEATURE_NAME as NEXT32_FEATURE_NAME,
    PROTOCOL as NEXT32_FEATURE_PROTOCOL,
)
from src.next32_omat24_cohort import PROTOCOL as COHORT_PROTOCOL


def _near_b2(displacement: float = 0.08) -> Atoms:
    atoms = Atoms(
        numbers=[11, 17],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=np.eye(3) * 4.0,
        pbc=True,
    )
    atoms.positions[1] += np.array([displacement, 0.0, 0.0])
    atoms.wrap()
    return atoms


def test_symmetry_recovery_is_translation_permutation_rotation_and_supercell_invariant() -> None:
    atoms = _near_b2()
    translated = atoms.copy()
    translated.positions += np.array([1.1, -0.7, 0.4])
    translated.wrap()
    permuted = atoms[[1, 0]]
    rotated = atoms.copy()
    rotated.rotate(37.0, "z", rotate_cell=True)
    repeated = atoms.repeat((2, 1, 1))

    reference = compute_symmetry_recovery_features(atoms)
    variants = [
        compute_symmetry_recovery_features(value)
        for value in (translated, permuted, rotated, repeated)
    ]

    assert reference.supported
    assert tuple(reference.features) == SYMMETRY_FEATURE_NAMES
    assert reference.features["sym_recovery_gain_log2"] > 0.0
    for variant in variants:
        assert variant.supported, variant.failure_reason
        assert variant.features == pytest.approx(reference.features, rel=1e-6, abs=1e-8)


def test_larger_near_symmetry_breaking_requires_larger_recovery() -> None:
    small = compute_symmetry_recovery_features(_near_b2(0.06))
    large = compute_symmetry_recovery_features(_near_b2(0.18))

    assert small.supported and large.supported
    assert large.features["sym_recovery_onset_rel"] >= small.features[
        "sym_recovery_onset_rel"
    ]
    assert large.features["sym_recovery_residual_rms_rel"] > small.features[
        "sym_recovery_residual_rms_rel"
    ]
    assert large.features["sym_recovery_residual_max_rel"] > small.features[
        "sym_recovery_residual_max_rel"
    ]


def test_true_p1_without_approximate_operations_is_supported_zero_not_high_risk() -> None:
    atoms = Atoms(
        numbers=[3, 8, 17],
        scaled_positions=[[0.071, 0.193, 0.311], [0.427, 0.529, 0.673], [0.811, 0.277, 0.941]],
        cell=[[3.7, 0.2, 0.1], [0.4, 4.1, 0.3], [0.2, 0.5, 4.6]],
        pbc=True,
    )

    result = compute_symmetry_recovery_features(atoms)

    assert result.supported
    assert result.features == pytest.approx({name: 0.0 for name in SYMMETRY_FEATURE_NAMES})


def test_invalid_symmetry_geometry_fails_open_and_schema_has_no_leakage() -> None:
    invalid = _near_b2()
    invalid.set_cell(np.zeros((3, 3)), scale_atoms=False)

    result = compute_symmetry_recovery_features(invalid)

    assert not result.supported
    assert result.features == {}
    assert not any(name.lower() in {"sid", "space_group_metadata"} for name in SYMMETRY_FEATURE_NAMES)
    assert not any(
        token in name.lower()
        for name in SYMMETRY_FEATURE_NAMES
        for token in (
            "energy",
            "force",
            "stress",
            "dft",
            "relax",
            "label",
            "target",
        )
    )


def test_steric_self_images_are_unique_and_cancel_directionally() -> None:
    atoms = Atoms(
        numbers=[1],
        positions=[[0.0, 0.0, 0.0]],
        cell=np.eye(3) * 0.5,
        pbc=True,
    )
    q = 0.5 / (2.0 * 0.31)
    weight = q**-12 - 1.0

    result = compute_directional_steric_features(atoms)

    assert result.supported
    assert tuple(result.features) == STERIC_FEATURE_NAMES
    assert result.features["steric_rep12_pa"] == pytest.approx(6.0 * weight)
    assert result.features["steric_rep12_site_max"] == pytest.approx(6.0 * weight)
    assert result.features["steric_rep12_vector_max"] == pytest.approx(0.0, abs=1e-12)
    assert result.features["steric_rep12_tensor_deviator"] == pytest.approx(0.0, abs=1e-12)


def test_directional_steric_features_are_representation_invariant() -> None:
    atoms = _near_b2()
    atoms.set_cell(atoms.cell.array * 0.72, scale_atoms=True)
    translated = atoms.copy()
    translated.positions += np.array([0.7, -1.2, 0.3])
    translated.wrap()
    permuted = atoms[[1, 0]]
    rotated = atoms.copy()
    rotated.rotate(23.0, "y", rotate_cell=True)
    repeated = atoms.repeat((2, 1, 1))

    reference = compute_directional_steric_features(atoms)
    variants = [
        compute_directional_steric_features(value)
        for value in (translated, permuted, rotated, repeated)
    ]

    assert reference.supported
    for variant in variants:
        assert variant.supported, variant.failure_reason
        assert variant.features == pytest.approx(reference.features, rel=1e-6, abs=1e-8)


def test_directional_load_detects_broken_cancellation() -> None:
    exact = _near_b2(0.0)
    perturbed = _near_b2(0.12)
    exact.set_cell(exact.cell.array * 0.70, scale_atoms=True)
    perturbed.set_cell(perturbed.cell.array * 0.70, scale_atoms=True)

    balanced = compute_directional_steric_features(exact)
    broken = compute_directional_steric_features(perturbed)

    assert balanced.supported and broken.supported
    assert broken.features["steric_overlap2_vector_rms"] > balanced.features[
        "steric_overlap2_vector_rms"
    ]
    assert broken.features["steric_overlap2_vector_q95"] > balanced.features[
        "steric_overlap2_vector_q95"
    ]


def test_compression_increases_steric_load_and_missing_radius_fails_open() -> None:
    atoms = _near_b2()
    compressed = atoms.copy()
    compressed.set_cell(atoms.cell.array * 0.75, scale_atoms=True)

    original = compute_directional_steric_features(atoms)
    squeezed = compute_directional_steric_features(compressed)
    missing = compute_directional_steric_features(atoms, radii={11: 1.0})

    assert squeezed.features["steric_rep12_pa"] > original.features["steric_rep12_pa"]
    assert squeezed.features["steric_rep12_site_q95"] > original.features[
        "steric_rep12_site_q95"
    ]
    assert not missing.supported
    assert "radius" in str(missing.failure_reason)
    assert missing.features == {}
    assert not any(
        token in name.lower()
        for name in STERIC_FEATURE_NAMES
        for token in ("energy", "force", "stress", "dft", "relax", "label", "target")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _batch_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    material_id = "unit::near-b2"
    atoms = _near_b2()
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
            "sid": ["opaque-id-without-parsing"],
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
    next32_path = tmp_path / NEXT32_FEATURE_NAME
    row = {
        "material_id": material_id,
        "source_name": "unit",
        "parent_id": "parent-b2",
        "natoms": 2,
        "cov_q01": 0.91,
        "cov_q05": 0.93,
        "sivr_edge_mismatch_q95": 0.12,
        "sivr_site_imbalance_rms": 0.08,
    }
    pd.DataFrame([row]).to_parquet(next32_path, index=False)
    next32_manifest_path = tmp_path / "next32-feature-manifest.json"
    next32_manifest_path.write_text(
        json.dumps(
            {
                "protocol": NEXT32_FEATURE_PROTOCOL,
                "labels_opened": False,
                "endpoint_fields_read": False,
                "model_or_proxy_potential_used": False,
                "outputs_sha256": {NEXT32_FEATURE_NAME: _sha256(next32_path)},
            }
        )
        + "\n"
    )
    return (
        archive_path,
        metadata_path,
        cohort_manifest_path,
        next32_path,
        next32_manifest_path,
    )


def test_batch_joins_hash_locked_geometry_and_next32_features_without_labels(
    tmp_path: Path,
) -> None:
    archive, metadata, cohort_manifest, next32, next32_manifest = _batch_fixture(tmp_path)
    output = tmp_path / "next33-features"

    manifest = build_symmetry_steric_feature_batch(
        archive_path=archive,
        metadata_path=metadata,
        cohort_manifest_path=cohort_manifest,
        next32_feature_path=next32,
        next32_feature_manifest_path=next32_manifest,
        output_dir=output,
    )

    frame = pd.read_parquet(output / FEATURE_NAME)
    assert frame.material_id.tolist() == ["unit::near-b2"]
    assert set(SYMMETRY_FEATURE_NAMES + STERIC_FEATURE_NAMES).issubset(frame)
    assert set(REUSED_NEXT32_FEATURE_NAMES).issubset(frame)
    assert manifest["labels_opened"] is False
    assert manifest["endpoint_fields_read"] is False
    assert manifest["sid_metadata_used"] is False
    assert manifest["outputs_sha256"][FEATURE_NAME] == _sha256(output / FEATURE_NAME)
    assert json.loads((output / MANIFEST_NAME).read_text()) == manifest
    with pytest.raises(FileExistsError):
        build_symmetry_steric_feature_batch(
            archive_path=archive,
            metadata_path=metadata,
            cohort_manifest_path=cohort_manifest,
            next32_feature_path=next32,
            next32_feature_manifest_path=next32_manifest,
            output_dir=output,
        )


@pytest.mark.parametrize("corruption", ["cohort_labels", "next32_labels", "geometry_hash"])
def test_batch_rejects_any_upstream_label_or_hash_boundary_crossing(
    tmp_path: Path, corruption: str
) -> None:
    archive, metadata, cohort_manifest, next32, next32_manifest = _batch_fixture(tmp_path)
    target_manifest = cohort_manifest if corruption != "next32_labels" else next32_manifest
    value = json.loads(target_manifest.read_text())
    if corruption in {"cohort_labels", "next32_labels"}:
        value["labels_opened"] = True
    else:
        value["outputs_sha256"][archive.name] = "0" * 64
    target_manifest.write_text(json.dumps(value) + "\n")

    with pytest.raises(ValueError, match="label-free|hash"):
        build_symmetry_steric_feature_batch(
            archive_path=archive,
            metadata_path=metadata,
            cohort_manifest_path=cohort_manifest,
            next32_feature_path=next32,
            next32_feature_manifest_path=next32_manifest,
            output_dir=tmp_path / "next33-features",
        )
