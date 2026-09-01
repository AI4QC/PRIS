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
from src.next32_inorganic_response_features import (
    FEATURE_NAME as NEXT32_FEATURE_NAME,
    PROTOCOL as NEXT32_FEATURE_PROTOCOL,
)
from src.next32_omat24_cohort import PROTOCOL as COHORT_PROTOCOL
from src.next37_self_stress_compatibility_features import (
    FEATURE_NAME as NEXT37_FEATURE_NAME,
    PROTOCOL as NEXT37_FEATURE_PROTOCOL,
)
from src.next38_bond_valence_transport_compatibility_features import (
    CANDIDATE_FEATURE_NAMES,
    DIAGNOSTIC_FEATURE_NAMES,
    FEATURE_NAME,
    MANIFEST_NAME,
    REUSED_FEATURE_NAMES,
    bond_valence_transport_compatibility_features,
    build_bond_valence_transport_compatibility_feature_batch,
    compute_bond_valence_transport_compatibility_features,
    transport_compatibility_from_jacobian,
)


CHARGES = np.asarray([1.0, 1.0, -1.0, -1.0])
ENDPOINTS = np.asarray([[0, 2], [0, 3], [1, 2], [1, 3]])
PRIORS = np.asarray([0.9, 0.1, 0.9, 0.1])
CORRECTION = np.asarray([-0.4, 0.4, -0.4, 0.4])


def test_balanced_prior_is_supported_exact_zero() -> None:
    result = transport_compatibility_from_jacobian(
        charges=CHARGES,
        endpoints=ENDPOINTS,
        priors=[0.5, 0.5, 0.5, 0.5],
        jacobian=np.zeros((4, 1)),
        parameter_sources=("exact",) * 4,
    )

    assert result.supported, result.failure_reason
    assert tuple(result.features) == CANDIDATE_FEATURE_NAMES + DIAGNOSTIC_FEATURE_NAMES
    assert all(result.features[name] == 0.0 for name in CANDIDATE_FEATURE_NAMES)
    assert result.features["bvtc_site_deficit_rms"] == pytest.approx(0.0)


def test_minimum_correction_can_be_fully_compatible() -> None:
    result = transport_compatibility_from_jacobian(
        charges=CHARGES,
        endpoints=ENDPOINTS,
        priors=PRIORS,
        jacobian=CORRECTION[:, None],
        parameter_sources=("exact",) * 4,
    )

    assert result.supported, result.failure_reason
    assert result.features["bvtc_correction_rms"] == pytest.approx(0.4)
    assert result.features["bvtc_compatible_rms"] == pytest.approx(0.4)
    assert result.features["bvtc_compatible_q95"] == pytest.approx(0.4)
    assert result.features["bvtc_incompatible_rms"] == pytest.approx(0.0, abs=1e-12)
    assert result.features["bvtc_incompatible_fraction"] == pytest.approx(0.0)
    assert result.features["bvtc_compatible_localization"] == pytest.approx(1.0)


def test_orthogonal_normalized_jacobian_exposes_incompatibility() -> None:
    orthogonal = np.asarray([1.0, -1.0, -1.0, 1.0])[:, None]
    result = transport_compatibility_from_jacobian(
        charges=CHARGES,
        endpoints=ENDPOINTS,
        priors=PRIORS,
        jacobian=orthogonal,
        parameter_sources=("brown_generic",) * 4,
    )

    assert result.supported
    assert result.features["bvtc_compatible_rms"] == pytest.approx(0.0, abs=1e-12)
    assert result.features["bvtc_incompatible_rms"] == pytest.approx(0.4)
    assert result.features["bvtc_incompatible_fraction"] == pytest.approx(1.0)
    assert result.features["bvtc_compatible_localization"] == pytest.approx(0.0)


def test_pure_bond_valence_kernel_preserves_star_sums_and_replication() -> None:
    vectors = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 1.0, 0.0], [1.1, 0.0, 0.0]]
    )
    reference = bond_valence_transport_compatibility_features(
        charges=CHARGES,
        endpoints=ENDPOINTS,
        vectors=vectors,
        strengths=[9.0, 1.0, 9.0, 1.0],
        decays=[0.37] * 4,
        parameter_sources=("exact",) * 4,
    )
    repeated = bond_valence_transport_compatibility_features(
        charges=np.tile(CHARGES, 2),
        endpoints=np.vstack([ENDPOINTS, ENDPOINTS + 4]),
        vectors=np.vstack([vectors, vectors]),
        strengths=[9.0, 1.0, 9.0, 1.0] * 2,
        decays=[0.37] * 8,
        parameter_sources=("exact",) * 8,
    )
    order = np.asarray([3, 0, 2, 1])
    reordered = bond_valence_transport_compatibility_features(
        charges=CHARGES,
        endpoints=ENDPOINTS[order],
        vectors=vectors[order],
        strengths=np.asarray([9.0, 1.0, 9.0, 1.0])[order],
        decays=np.asarray([0.37] * 4)[order],
        parameter_sources=tuple(np.asarray(["exact"] * 4)[order]),
    )

    assert reference.supported, reference.failure_reason
    assert repeated.supported, repeated.failure_reason
    assert reordered.supported, reordered.failure_reason
    for name in CANDIDATE_FEATURE_NAMES:
        assert repeated.features[name] == pytest.approx(
            reference.features[name], rel=2e-10, abs=2e-12
        )
        assert reordered.features[name] == pytest.approx(
            reference.features[name], rel=2e-10, abs=2e-12
        )


def test_invalid_inputs_fail_open_and_schema_has_no_forbidden_endpoint_tokens() -> None:
    bad = bond_valence_transport_compatibility_features(
        charges=[1.0, -1.0],
        endpoints=[[0, 1]],
        vectors=[[0.0, 0.0, 0.0]],
        strengths=[1.0],
        decays=[0.37],
        parameter_sources=("exact",),
    )
    assert not bad.supported and bad.features == {}
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


def _nacl(displacement: float = 0.04) -> tuple[Structure, list[float]]:
    return (
        Structure(
            Lattice.cubic(3.2),
            ["Na", "Cl"],
            [[0.0, 0.0, 0.0], [0.5 + displacement, 0.5, 0.5]],
        ),
        [1.0, -1.0],
    )


def _charges(structure: Structure) -> list[float]:
    return [1.0 if site.specie.symbol == "Na" else -1.0 for site in structure]


def test_structure_projection_is_representation_invariant() -> None:
    structure, charges = _nacl()
    translated = structure.copy()
    translated.translate_sites(range(len(translated)), [0.11, -0.17, 0.09], frac_coords=True)
    rotated = structure.copy()
    rotated.apply_operation(
        SymmOp.from_axis_angle_and_translation([0.3, 0.6, 0.2], 29.0),
        fractional=False,
    )
    permuted = Structure(
        structure.lattice,
        [site.specie for site in reversed(structure)],
        [site.frac_coords for site in reversed(structure)],
    )
    repeated = structure.copy()
    repeated.make_supercell([2, 1, 1])
    reference = compute_bond_valence_transport_compatibility_features(structure, charges)
    variants = [
        compute_bond_valence_transport_compatibility_features(translated, charges),
        compute_bond_valence_transport_compatibility_features(rotated, charges),
        compute_bond_valence_transport_compatibility_features(
            permuted, list(reversed(charges))
        ),
        compute_bond_valence_transport_compatibility_features(repeated, _charges(repeated)),
    ]

    assert reference.supported, reference.failure_reason
    for variant in variants:
        assert variant.supported, variant.failure_reason
        for name in CANDIDATE_FEATURE_NAMES:
            assert variant.features[name] == pytest.approx(
                reference.features[name], rel=5e-5, abs=5e-8
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _batch_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
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
    next32_path = tmp_path / NEXT32_FEATURE_NAME
    pd.DataFrame(
        {
            "material_id": [material_id],
            "scbv_mismatch_q95": [0.4],
        }
    ).to_parquet(next32_path, index=False)
    next32_manifest = tmp_path / "next32-feature-manifest.json"
    next32_manifest.write_text(
        json.dumps(
            {
                "protocol": NEXT32_FEATURE_PROTOCOL,
                "input_role": "unrelaxed_x0_geometry_only",
                "labels_opened": False,
                "endpoint_fields_read": False,
                "model_or_proxy_potential_used": False,
                "coordinates_or_cell_modified": False,
                "outputs_sha256": {NEXT32_FEATURE_NAME: _sha256(next32_path)},
            }
        )
        + "\n"
    )
    next37_path = tmp_path / NEXT37_FEATURE_NAME
    pd.DataFrame(
        {
            "material_id": [material_id],
            "source_name": ["unit"],
            "parent_id": ["parent-nacl"],
            "natoms": [2],
            "steric_rep12_vector_rms": [0.3],
            "steric_rep12_vector_max": [0.5],
            "sivr_site_imbalance_rms": [0.2],
        }
    ).to_parquet(next37_path, index=False)
    next37_manifest = tmp_path / "next37-feature-manifest.json"
    next37_manifest.write_text(
        json.dumps(
            {
                "protocol": NEXT37_FEATURE_PROTOCOL,
                "input_role": "unrelaxed_x0_geometry_only",
                "labels_opened": False,
                "endpoint_fields_read": False,
                "dft_values_used": False,
                "self_stress_compatibility_projection_used": True,
                "coordinate_displacement_solved_or_applied": False,
                "electronic_structure_calculation_used": False,
                "model_or_proxy_potential_used": False,
                "coordinates_or_cell_modified": False,
                "outputs_sha256": {NEXT37_FEATURE_NAME: _sha256(next37_path)},
            }
        )
        + "\n"
    )
    return (
        archive_path,
        metadata_path,
        cohort_manifest_path,
        next32_path,
        next32_manifest,
        next37_path,
        next37_manifest,
    )


def test_batch_is_hash_locked_label_free_and_no_replace(tmp_path: Path) -> None:
    inputs = _batch_fixture(tmp_path)
    output = tmp_path / "next38-features"
    manifest = build_bond_valence_transport_compatibility_feature_batch(
        archive_path=inputs[0],
        metadata_path=inputs[1],
        cohort_manifest_path=inputs[2],
        next32_feature_path=inputs[3],
        next32_feature_manifest_path=inputs[4],
        next37_feature_path=inputs[5],
        next37_feature_manifest_path=inputs[6],
        output_dir=output,
    )

    assert manifest["labels_opened"] is False
    assert manifest["dft_values_used"] is False
    assert manifest["bond_valence_transport_compatibility_used"] is True
    assert manifest["coordinate_displacement_solved_or_applied"] is False
    assert manifest["counts"] == {"rows": 1, "atoms": 2, "bvtc_supported": 1}
    frame = pd.read_parquet(output / FEATURE_NAME)
    assert frame.bvtc_supported.all() and "sid" not in frame
    assert tuple(frame.loc[:, REUSED_FEATURE_NAMES].iloc[0]) == pytest.approx(
        (0.4, 0.3, 0.5, 0.2)
    )
    assert _sha256(output / FEATURE_NAME) == manifest["outputs_sha256"][FEATURE_NAME]
    assert (output / MANIFEST_NAME).is_file()
    with pytest.raises(FileExistsError):
        build_bond_valence_transport_compatibility_feature_batch(
            archive_path=inputs[0],
            metadata_path=inputs[1],
            cohort_manifest_path=inputs[2],
            next32_feature_path=inputs[3],
            next32_feature_manifest_path=inputs[4],
            next37_feature_path=inputs[5],
            next37_feature_manifest_path=inputs[6],
            output_dir=output,
        )


def test_batch_rejects_opened_upstream_and_failure_is_fail_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _batch_fixture(tmp_path)
    opened = json.loads(inputs[6].read_text())
    opened["labels_opened"] = True
    inputs[6].write_text(json.dumps(opened) + "\n")
    with pytest.raises(ValueError, match="label-free"):
        build_bond_valence_transport_compatibility_feature_batch(
            archive_path=inputs[0],
            metadata_path=inputs[1],
            cohort_manifest_path=inputs[2],
            next32_feature_path=inputs[3],
            next32_feature_manifest_path=inputs[4],
            next37_feature_path=inputs[5],
            next37_feature_manifest_path=inputs[6],
            output_dir=tmp_path / "opened",
        )

    inputs = _batch_fixture(tmp_path / "second")

    class Unsupported:
        supported = False
        values = None
        policy = None
        failure_reason = "unit unsupported"

    monkeypatch.setattr(
        "src.next38_bond_valence_transport_compatibility_features.infer_valence_assignment",
        lambda _structure: Unsupported(),
    )
    output = tmp_path / "fail-open"
    manifest = build_bond_valence_transport_compatibility_feature_batch(
        archive_path=inputs[0],
        metadata_path=inputs[1],
        cohort_manifest_path=inputs[2],
        next32_feature_path=inputs[3],
        next32_feature_manifest_path=inputs[4],
        next37_feature_path=inputs[5],
        next37_feature_manifest_path=inputs[6],
        output_dir=output,
    )
    frame = pd.read_parquet(output / FEATURE_NAME)
    assert not frame.bvtc_supported.any()
    assert frame.loc[0, "bvtc_failure"] == "unit unsupported"
    assert manifest["counts"]["bvtc_supported"] == 0
