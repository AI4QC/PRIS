from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from ase import Atoms
import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.operations import SymmOp

from src.next11_geometry_only_frames import _canonical_frame
from src.next32_omat24_cohort import PROTOCOL as COHORT_PROTOCOL
from src.next33_symmetry_steric_features import (
    FEATURE_NAME as NEXT33_FEATURE_NAME,
    PROTOCOL as NEXT33_FEATURE_PROTOCOL,
)
from src.next34_analytic_field_features import (
    CANDIDATE_FEATURE_NAMES,
    DIAGNOSTIC_FEATURE_NAMES,
    FEATURE_NAME,
    MANIFEST_NAME,
    REUSED_FEATURE_NAMES,
    build_analytic_field_feature_batch,
    compute_analytic_field_features,
)


def _nacl(displacement: float = 0.0) -> tuple[Structure, list[float]]:
    structure = Structure(
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5 + displacement, 0.5, 0.5]],
    )
    return structure, [1.0, -1.0]


def _charges(structure: Structure) -> list[float]:
    return [1.0 if site.specie.symbol == "Na" else -1.0 for site in structure]


def test_schema_is_dimensionless_and_contains_no_forbidden_model_or_label_tokens() -> None:
    structure, charges = _nacl()

    result = compute_analytic_field_features(structure, charges)

    assert result.supported, result.failure_reason
    assert tuple(result.features) == CANDIDATE_FEATURE_NAMES + DIAGNOSTIC_FEATURE_NAMES
    assert set(result.features) == set(CANDIDATE_FEATURE_NAMES + DIAGNOSTIC_FEATURE_NAMES)
    assert np.isfinite(list(result.features.values())).all()
    assert not any(
        token in name.lower()
        for name in result.features
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


def test_exact_nacl_is_balanced_and_displacement_increases_field_residual() -> None:
    exact, charges = _nacl()
    displaced, _ = _nacl(0.01)

    balanced = compute_analytic_field_features(exact, charges)
    broken = compute_analytic_field_features(displaced, charges)

    assert balanced.supported and broken.supported
    assert balanced.features["aefi_field_max"] == pytest.approx(0.0, abs=1e-12)
    assert broken.features["aefi_field_rms"] > balanced.features["aefi_field_rms"]
    assert broken.features["aefi_residual_max"] > balanced.features["aefi_residual_max"]


def test_field_features_are_representation_scale_and_charge_amplitude_invariant() -> None:
    structure, charges = _nacl(0.01)
    translated = structure.copy()
    translated.translate_sites(range(len(translated)), [0.137, -0.211, 0.083], frac_coords=True)
    permuted = Structure(
        structure.lattice,
        [site.specie for site in reversed(structure)],
        [site.frac_coords for site in reversed(structure)],
    )
    rotated = structure.copy()
    rotated.apply_operation(
        SymmOp.from_axis_angle_and_translation([0.0, 0.0, 1.0], 31.0),
        fractional=False,
    )
    scaled = Structure(
        Lattice(np.asarray(structure.lattice.matrix) * 1.37),
        [site.specie for site in structure],
        [site.frac_coords for site in structure],
    )
    repeated = structure.copy()
    repeated.make_supercell([2, 1, 1])

    reference = compute_analytic_field_features(structure, charges)
    variants = [
        compute_analytic_field_features(translated, charges),
        compute_analytic_field_features(permuted, list(reversed(charges))),
        compute_analytic_field_features(rotated, charges),
        compute_analytic_field_features(scaled, charges),
        compute_analytic_field_features(repeated, _charges(repeated)),
        compute_analytic_field_features(structure, [3.7 * value for value in charges]),
    ]

    assert reference.supported
    for variant in variants:
        assert variant.supported, variant.failure_reason
        assert variant.features == pytest.approx(reference.features, rel=2e-5, abs=1e-9)


@pytest.mark.parametrize(
    "charges, reason",
    [([1.0, 1.0], "neutral"), ([0.0, 0.0], "nonzero"), ([1.0], "match")],
)
def test_invalid_charge_assignments_fail_open(charges: list[float], reason: str) -> None:
    structure, _ = _nacl()

    result = compute_analytic_field_features(structure, charges)

    assert not result.supported
    assert result.features == {}
    assert reason in str(result.failure_reason).lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _batch_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    material_id = "unit::nacl"
    structure, _charges_unused = _nacl(0.01)
    atoms: Atoms = AseAtomsAdaptor.get_atoms(structure)
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
            "parent_id": ["parent-nacl"],
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

    next33_path = tmp_path / NEXT33_FEATURE_NAME
    row: dict[str, object] = {
        "material_id": material_id,
        "source_name": "unit",
        "parent_id": "parent-nacl",
        "natoms": 2,
        "steric_rep12_vector_rms": 0.1,
        "steric_rep12_vector_q95": 0.2,
        "steric_rep12_vector_max": 0.3,
        "steric_overlap2_vector_rms": 0.01,
        "steric_rep12_tensor_deviator": 0.02,
        "sivr_site_imbalance_rms": 0.04,
        "sivr_edge_mismatch_q95": 0.05,
        "cov_q05": 0.92,
    }
    pd.DataFrame([row]).to_parquet(next33_path, index=False)
    next33_manifest_path = tmp_path / "next33-feature-manifest.json"
    next33_manifest_path.write_text(
        json.dumps(
            {
                "protocol": NEXT33_FEATURE_PROTOCOL,
                "input_role": "unrelaxed_x0_geometry_only",
                "labels_opened": False,
                "endpoint_fields_read": False,
                "sid_metadata_used": False,
                "model_or_proxy_potential_used": False,
                "coordinates_or_cell_modified": False,
                "outputs_sha256": {NEXT33_FEATURE_NAME: _sha256(next33_path)},
            }
        )
        + "\n"
    )
    return (
        archive_path,
        metadata_path,
        cohort_manifest_path,
        next33_path,
        next33_manifest_path,
    )


def test_batch_is_hash_locked_label_free_no_replace_and_copies_only_frozen_features(
    tmp_path: Path,
) -> None:
    archive, metadata, cohort_manifest, next33, next33_manifest = _batch_fixture(tmp_path)
    output = tmp_path / "next34-features"

    manifest = build_analytic_field_feature_batch(
        archive_path=archive,
        metadata_path=metadata,
        cohort_manifest_path=cohort_manifest,
        next33_feature_path=next33,
        next33_feature_manifest_path=next33_manifest,
        output_dir=output,
    )

    frame = pd.read_parquet(output / FEATURE_NAME)
    assert frame.material_id.tolist() == ["unit::nacl"]
    assert frame.aefi_supported.tolist() == [True]
    assert set(CANDIDATE_FEATURE_NAMES + DIAGNOSTIC_FEATURE_NAMES).issubset(frame)
    assert set(REUSED_FEATURE_NAMES).issubset(frame)
    assert "sid" not in frame
    assert manifest["labels_opened"] is False
    assert manifest["endpoint_fields_read"] is False
    assert manifest["dft_values_used"] is False
    assert manifest["model_or_proxy_potential_used"] is False
    assert manifest["classical_analytic_electrostatics_used"] is True
    assert manifest["coordinates_or_cell_modified"] is False
    assert manifest["outputs_sha256"][FEATURE_NAME] == _sha256(output / FEATURE_NAME)
    assert json.loads((output / MANIFEST_NAME).read_text()) == manifest
    with pytest.raises(FileExistsError):
        build_analytic_field_feature_batch(
            archive_path=archive,
            metadata_path=metadata,
            cohort_manifest_path=cohort_manifest,
            next33_feature_path=next33,
            next33_feature_manifest_path=next33_manifest,
            output_dir=output,
        )


@pytest.mark.parametrize("corruption", ["cohort_labels", "next33_labels", "feature_hash"])
def test_batch_rejects_label_or_hash_boundary_crossing(
    tmp_path: Path, corruption: str
) -> None:
    archive, metadata, cohort_manifest, next33, next33_manifest = _batch_fixture(tmp_path)
    target = cohort_manifest if corruption == "cohort_labels" else next33_manifest
    value = json.loads(target.read_text())
    if corruption in {"cohort_labels", "next33_labels"}:
        value["labels_opened"] = True
    else:
        value["outputs_sha256"][next33.name] = "0" * 64
    target.write_text(json.dumps(value) + "\n")

    with pytest.raises(ValueError, match="label-free|hash"):
        build_analytic_field_feature_batch(
            archive_path=archive,
            metadata_path=metadata,
            cohort_manifest_path=cohort_manifest,
            next33_feature_path=next33,
            next33_feature_manifest_path=next33_manifest,
            output_dir=tmp_path / "next34-features",
        )
