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
from src.next36_charge_spectrum_features import (
    FEATURE_NAME as NEXT36_FEATURE_NAME,
    PROTOCOL as NEXT36_FEATURE_PROTOCOL,
)
from src.next37_self_stress_compatibility_features import (
    CANDIDATE_FEATURE_NAMES,
    DIAGNOSTIC_FEATURE_NAMES,
    FEATURE_NAME,
    MANIFEST_NAME,
    REUSED_FEATURE_NAMES,
    build_self_stress_compatibility_feature_batch,
    compute_self_stress_compatibility_features,
    self_stress_compatibility_features,
)


def _two_edge(
    vectors: list[list[float]], residuals: list[float]
):
    return self_stress_compatibility_features(
        n_sites=2,
        endpoints=[[0, 1], [0, 1]],
        vectors=vectors,
        residuals=residuals,
        weights=[1.0, 1.0],
    )


def test_duplicate_rows_with_opposite_residual_are_exact_self_stress() -> None:
    result = _two_edge([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [1.0, -1.0])

    assert result.supported, result.failure_reason
    assert tuple(result.features) == CANDIDATE_FEATURE_NAMES + DIAGNOSTIC_FEATURE_NAMES
    for name in CANDIDATE_FEATURE_NAMES:
        assert result.features[name] == pytest.approx(0.0, abs=1e-12)
    assert result.features["sscp_balanced_fraction"] == pytest.approx(1.0)
    assert result.features["sscp_cokernel_dimension_fraction"] == pytest.approx(0.5)


def test_duplicate_rows_with_equal_residual_are_fully_load_generating() -> None:
    result = _two_edge([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [1.0, 1.0])

    assert result.supported
    assert result.features["sscp_load_fraction"] == pytest.approx(1.0)
    assert result.features["sscp_atomic_load_fraction"] == pytest.approx(1.0)
    assert result.features["sscp_cell_load_fraction"] == pytest.approx(0.0)
    assert result.features["sscp_load_rms"] == pytest.approx(1.0)
    assert result.features["sscp_load_q95"] == pytest.approx(1.0)
    assert result.features["sscp_load_localization"] == pytest.approx(1.0)
    assert result.features["sscp_balanced_fraction"] == pytest.approx(0.0)


def test_opposite_edge_directions_isolate_affine_cell_load() -> None:
    result = _two_edge([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], [1.0, 1.0])

    assert result.supported
    assert result.features["sscp_atomic_load_fraction"] == pytest.approx(0.0, abs=1e-12)
    assert result.features["sscp_load_fraction"] == pytest.approx(1.0)
    assert result.features["sscp_cell_load_fraction"] == pytest.approx(1.0)


def test_zero_residual_is_supported_zero_and_invalid_inputs_fail_open() -> None:
    zero = _two_edge([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], [0.0, 0.0])
    bad = self_stress_compatibility_features(
        n_sites=2,
        endpoints=[[0, 1]],
        vectors=[[0.0, 0.0, 0.0]],
        residuals=[1.0],
        weights=[1.0],
    )

    assert zero.supported
    assert all(zero.features[name] == 0.0 for name in CANDIDATE_FEATURE_NAMES)
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
    scaled = structure.copy()
    scaled.scale_lattice(structure.volume * 2.7**3)
    repeated = structure.copy()
    repeated.make_supercell([2, 1, 1])
    reference = compute_self_stress_compatibility_features(structure, charges)
    variants = [
        compute_self_stress_compatibility_features(translated, charges),
        compute_self_stress_compatibility_features(rotated, charges),
        compute_self_stress_compatibility_features(permuted, list(reversed(charges))),
        compute_self_stress_compatibility_features(scaled, charges),
        compute_self_stress_compatibility_features(
            structure, np.asarray(charges) * 5.2
        ),
        compute_self_stress_compatibility_features(repeated, _charges(repeated)),
    ]

    assert reference.supported, reference.failure_reason
    for variant in variants:
        assert variant.supported, variant.failure_reason
        for name in CANDIDATE_FEATURE_NAMES:
            assert variant.features[name] == pytest.approx(
                reference.features[name], rel=3e-5, abs=3e-8
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _batch_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
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
    upstream_path = tmp_path / NEXT36_FEATURE_NAME
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
    upstream_manifest_path = tmp_path / "next36-feature-manifest.json"
    upstream_manifest_path.write_text(
        json.dumps(
            {
                "protocol": NEXT36_FEATURE_PROTOCOL,
                "input_role": "unrelaxed_x0_geometry_only",
                "labels_opened": False,
                "endpoint_fields_read": False,
                "dft_values_used": False,
                "weighted_charge_spectrum_used": True,
                "thermodynamic_limit_hyperuniformity_claimed": False,
                "electronic_structure_calculation_used": False,
                "model_or_proxy_potential_used": False,
                "coordinates_or_cell_modified": False,
                "outputs_sha256": {NEXT36_FEATURE_NAME: _sha256(upstream_path)},
            }
        )
        + "\n"
    )
    return archive_path, metadata_path, cohort_manifest_path, upstream_path, upstream_manifest_path


def test_batch_is_hash_locked_label_free_and_no_replace(tmp_path: Path) -> None:
    archive, metadata, cohort_manifest, upstream, upstream_manifest = _batch_fixture(tmp_path)
    output = tmp_path / "next37-features"

    manifest = build_self_stress_compatibility_feature_batch(
        archive_path=archive,
        metadata_path=metadata,
        cohort_manifest_path=cohort_manifest,
        next36_feature_path=upstream,
        next36_feature_manifest_path=upstream_manifest,
        output_dir=output,
    )

    assert manifest["labels_opened"] is False
    assert manifest["dft_values_used"] is False
    assert manifest["self_stress_compatibility_projection_used"] is True
    assert manifest["coordinates_or_cell_modified"] is False
    assert manifest["counts"] == {"rows": 1, "atoms": 2, "sscp_supported": 1}
    frame = pd.read_parquet(output / FEATURE_NAME)
    assert frame.sscp_supported.all() and "sid" not in frame
    assert _sha256(output / FEATURE_NAME) == manifest["outputs_sha256"][FEATURE_NAME]
    assert (output / MANIFEST_NAME).is_file()
    with pytest.raises(FileExistsError):
        build_self_stress_compatibility_feature_batch(
            archive_path=archive,
            metadata_path=metadata,
            cohort_manifest_path=cohort_manifest,
            next36_feature_path=upstream,
            next36_feature_manifest_path=upstream_manifest,
            output_dir=output,
        )


def test_batch_rejects_opened_upstream_and_failure_is_fail_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive, metadata, cohort_manifest, upstream, upstream_manifest = _batch_fixture(tmp_path)
    opened = json.loads(upstream_manifest.read_text())
    opened["labels_opened"] = True
    upstream_manifest.write_text(json.dumps(opened) + "\n")
    with pytest.raises(ValueError, match="label-free"):
        build_self_stress_compatibility_feature_batch(
            archive_path=archive,
            metadata_path=metadata,
            cohort_manifest_path=cohort_manifest,
            next36_feature_path=upstream,
            next36_feature_manifest_path=upstream_manifest,
            output_dir=tmp_path / "opened",
        )

    archive, metadata, cohort_manifest, upstream, upstream_manifest = _batch_fixture(
        tmp_path / "second"
    )

    class Unsupported:
        supported = False
        values = None
        policy = None
        failure_reason = "unit unsupported"

    monkeypatch.setattr(
        "src.next37_self_stress_compatibility_features.infer_valence_assignment",
        lambda _structure: Unsupported(),
    )
    output = tmp_path / "fail-open"
    manifest = build_self_stress_compatibility_feature_batch(
        archive_path=archive,
        metadata_path=metadata,
        cohort_manifest_path=cohort_manifest,
        next36_feature_path=upstream,
        next36_feature_manifest_path=upstream_manifest,
        output_dir=output,
    )
    frame = pd.read_parquet(output / FEATURE_NAME)
    assert not frame.sscp_supported.any()
    assert frame.loc[0, "sscp_failure"] == "unit unsupported"
    assert manifest["counts"]["sscp_supported"] == 0
