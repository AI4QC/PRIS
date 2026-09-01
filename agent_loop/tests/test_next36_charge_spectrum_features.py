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
from src.next35_coulomb_steric_balance_features import (
    FEATURE_NAME as NEXT35_FEATURE_NAME,
    PROTOCOL as NEXT35_FEATURE_PROTOCOL,
)
from src.next36_charge_spectrum_features import (
    CANDIDATE_FEATURE_NAMES,
    DIAGNOSTIC_FEATURE_NAMES,
    DIMENSIONLESS_CUTOFF,
    FEATURE_NAME,
    MANIFEST_NAME,
    REUSED_FEATURE_NAMES,
    SMOOTHING_SCALES,
    build_charge_spectrum_feature_batch,
    charge_spectrum_features,
    compute_charge_spectrum_features,
)


def _nacl(displacement: float = 0.03) -> tuple[Structure, list[float]]:
    return (
        Structure(
            Lattice.cubic(3.1),
            ["Na", "Cl"],
            [[0.0, 0.0, 0.0], [0.5 + displacement, 0.5, 0.5]],
        ),
        [1.0, -1.0],
    )


def _charges(structure: Structure) -> list[float]:
    return [1.0 if site.specie.symbol == "Na" else -1.0 for site in structure]


def test_schema_and_frozen_constants_have_no_forbidden_tokens() -> None:
    assert SMOOTHING_SCALES == (0.25, 0.40, 0.60)
    assert DIMENSIONLESS_CUTOFF == 18.0
    assert len(CANDIDATE_FEATURE_NAMES) == 6
    assert len(DIAGNOSTIC_FEATURE_NAMES) == 3
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


def test_invalid_or_non_neutral_inputs_fail_open() -> None:
    result = charge_spectrum_features(
        np.eye(3), np.zeros((2, 3)), np.array([1.0, 1.0])
    )
    mismatch = charge_spectrum_features(
        np.eye(3), np.zeros((2, 3)), np.array([1.0])
    )

    assert not result.supported and "neutral" in str(result.failure_reason).lower()
    assert not mismatch.supported and mismatch.features == {}


def test_long_wavelength_charge_separation_has_larger_long_spectrum() -> None:
    lattice = np.diag([4.0, 8.0, 8.0])
    coords = np.array(
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.50, 0.0, 0.0], [0.75, 0.0, 0.0]]
    )
    separated = charge_spectrum_features(lattice, coords, [1.0, 1.0, -1.0, -1.0])
    alternating = charge_spectrum_features(lattice, coords, [1.0, -1.0, 1.0, -1.0])

    assert separated.supported and alternating.supported
    assert separated.features["csf_gaussian_t060"] > alternating.features[
        "csf_gaussian_t060"
    ]
    assert tuple(separated.features) == CANDIDATE_FEATURE_NAMES + DIAGNOSTIC_FEATURE_NAMES
    assert np.isfinite(list(separated.features.values())).all()


def test_structure_features_are_representation_invariant() -> None:
    structure, charges = _nacl()
    translated = structure.copy()
    translated.translate_sites(range(len(translated)), [0.17, -0.09, 0.12], frac_coords=True)
    wrapped = Structure(
        structure.lattice,
        [site.specie for site in structure],
        structure.frac_coords + np.array([2.0, -1.0, 3.0]),
    )
    rotated = structure.copy()
    rotated.apply_operation(
        SymmOp.from_axis_angle_and_translation([0.2, 0.7, 0.4], 31.0),
        fractional=False,
    )
    permuted = Structure(
        structure.lattice,
        [site.specie for site in reversed(structure)],
        [site.frac_coords for site in reversed(structure)],
    )
    scaled = structure.copy()
    scaled.scale_lattice(structure.volume * 3.7**3)
    repeated = structure.copy()
    repeated.make_supercell([2, 1, 1])
    reference = compute_charge_spectrum_features(structure, charges)
    variants = [
        compute_charge_spectrum_features(translated, charges),
        compute_charge_spectrum_features(wrapped, charges),
        compute_charge_spectrum_features(rotated, charges),
        compute_charge_spectrum_features(permuted, list(reversed(charges))),
        compute_charge_spectrum_features(scaled, charges),
        compute_charge_spectrum_features(structure, np.asarray(charges) * 7.3),
        compute_charge_spectrum_features(repeated, _charges(repeated)),
    ]

    assert reference.supported, reference.failure_reason
    for variant in variants:
        assert variant.supported, variant.failure_reason
        for name in CANDIDATE_FEATURE_NAMES:
            assert variant.features[name] == pytest.approx(
                reference.features[name], rel=2e-8, abs=2e-10
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _batch_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    material_id = "unit::nacl"
    structure, _ = _nacl()
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
    upstream_path = tmp_path / NEXT35_FEATURE_NAME
    pd.DataFrame(
        {
            "material_id": [material_id],
            "source_name": ["unit"],
            "parent_id": ["parent-nacl"],
            "natoms": [2],
            "aefi_residual_max": [0.4],
            "steric_rep12_vector_rms": [0.3],
            "steric_rep12_vector_max": [0.5],
            "sivr_site_imbalance_rms": [0.2],
        }
    ).to_parquet(upstream_path, index=False)
    upstream_manifest_path = tmp_path / "next35-feature-manifest.json"
    upstream_manifest_path.write_text(
        json.dumps(
            {
                "protocol": NEXT35_FEATURE_PROTOCOL,
                "input_role": "unrelaxed_x0_geometry_only",
                "labels_opened": False,
                "endpoint_fields_read": False,
                "dft_values_used": False,
                "classical_analytic_electrostatics_used": True,
                "analytic_steric_field_used": True,
                "electronic_structure_calculation_used": False,
                "model_or_proxy_potential_used": False,
                "coordinates_or_cell_modified": False,
                "outputs_sha256": {NEXT35_FEATURE_NAME: _sha256(upstream_path)},
            }
        )
        + "\n"
    )
    return (
        archive_path,
        metadata_path,
        cohort_manifest_path,
        upstream_path,
        upstream_manifest_path,
    )


def test_batch_is_hash_locked_label_free_and_no_replace(tmp_path: Path) -> None:
    archive, metadata, cohort_manifest, upstream, upstream_manifest = _batch_fixture(tmp_path)
    output = tmp_path / "next36-features"

    manifest = build_charge_spectrum_feature_batch(
        archive_path=archive,
        metadata_path=metadata,
        cohort_manifest_path=cohort_manifest,
        next35_feature_path=upstream,
        next35_feature_manifest_path=upstream_manifest,
        output_dir=output,
    )

    assert manifest["labels_opened"] is False
    assert manifest["dft_values_used"] is False
    assert manifest["weighted_charge_spectrum_used"] is True
    assert manifest["counts"] == {"rows": 1, "atoms": 2, "csf_supported": 1}
    frame = pd.read_parquet(output / FEATURE_NAME)
    assert frame.csf_supported.all()
    assert "sid" not in frame
    assert (output / MANIFEST_NAME).is_file()
    assert _sha256(output / FEATURE_NAME) == manifest["outputs_sha256"][FEATURE_NAME]
    with pytest.raises(FileExistsError):
        build_charge_spectrum_feature_batch(
            archive_path=archive,
            metadata_path=metadata,
            cohort_manifest_path=cohort_manifest,
            next35_feature_path=upstream,
            next35_feature_manifest_path=upstream_manifest,
            output_dir=output,
        )


def test_batch_rejects_opened_or_hash_changed_upstream(tmp_path: Path) -> None:
    archive, metadata, cohort_manifest, upstream, upstream_manifest = _batch_fixture(tmp_path)
    value = json.loads(upstream_manifest.read_text())
    value["labels_opened"] = True
    upstream_manifest.write_text(json.dumps(value) + "\n")

    with pytest.raises(ValueError, match="label-free"):
        build_charge_spectrum_feature_batch(
            archive_path=archive,
            metadata_path=metadata,
            cohort_manifest_path=cohort_manifest,
            next35_feature_path=upstream,
            next35_feature_manifest_path=upstream_manifest,
            output_dir=tmp_path / "output",
        )


def test_batch_failure_is_fail_open(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    archive, metadata, cohort_manifest, upstream, upstream_manifest = _batch_fixture(tmp_path)

    class Unsupported:
        supported = False
        values = None
        policy = None
        failure_reason = "unit unsupported"

    monkeypatch.setattr(
        "src.next36_charge_spectrum_features.infer_valence_assignment",
        lambda _structure: Unsupported(),
    )
    output = tmp_path / "output"
    manifest = build_charge_spectrum_feature_batch(
        archive_path=archive,
        metadata_path=metadata,
        cohort_manifest_path=cohort_manifest,
        next35_feature_path=upstream,
        next35_feature_manifest_path=upstream_manifest,
        output_dir=output,
    )

    frame = pd.read_parquet(output / FEATURE_NAME)
    assert not frame.csf_supported.any()
    assert frame.loc[0, "csf_failure"] == "unit unsupported"
    assert manifest["counts"]["csf_supported"] == 0
