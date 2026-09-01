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
from src.next32_inorganic_response_features import (
    CONTACT_FEATURE_NAMES,
    FEATURE_NAME,
    INORGANIC_FEATURE_NAMES,
    MANIFEST_NAME,
    PAULING_NAME,
    build_inorganic_response_feature_batch,
    compute_inorganic_response_features,
    compute_periodic_contact_features,
)
from src.next32_omat24_cohort import PROTOCOL as COHORT_PROTOCOL


def _binary() -> Atoms:
    return Atoms(
        numbers=[11, 17],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=np.eye(3) * 4.0,
        pbc=True,
    )


def test_periodic_contacts_are_translation_permutation_and_supercell_invariant() -> None:
    atoms = _binary()
    translated = atoms.copy()
    translated.positions += np.array([1.3, -0.7, 0.4])
    translated.wrap()
    permuted = atoms[[1, 0]]
    repeated = atoms.repeat((2, 1, 1))

    reference = compute_periodic_contact_features(atoms)
    variants = [
        compute_periodic_contact_features(value)
        for value in (translated, permuted, repeated)
    ]

    assert reference.supported
    assert tuple(reference.features) == CONTACT_FEATURE_NAMES
    for variant in variants:
        assert variant.supported
        assert variant.features == pytest.approx(reference.features)


def test_uniform_compression_monotonically_increases_overlap() -> None:
    atoms = _binary()
    compressed = atoms.copy()
    compressed.set_cell(atoms.cell.array * 0.75, scale_atoms=True)

    original = compute_periodic_contact_features(atoms)
    squeezed = compute_periodic_contact_features(compressed)

    assert squeezed.features["cov_q05"] < original.features["cov_q05"]
    assert squeezed.features["cov_overlap2_pa"] > original.features["cov_overlap2_pa"]
    assert squeezed.features["cov_site_overlap_q95"] > original.features["cov_site_overlap_q95"]


def test_periodic_self_images_are_counted_once_per_undirected_pair() -> None:
    atoms = Atoms(numbers=[1], positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 0.5, pbc=True)

    result = compute_periodic_contact_features(atoms)

    assert result.supported
    assert result.features["cov_contact085_pa"] == pytest.approx(6.0)


def test_missing_or_invalid_radius_fails_open_without_forbidden_features() -> None:
    result = compute_periodic_contact_features(_binary(), radii={11: 1.0})

    assert not result.supported
    assert "radius" in str(result.failure_reason)
    assert result.features == {}
    assert not any(
        token in name.lower()
        for name in CONTACT_FEATURE_NAMES
        for token in ("energy", "force", "stress", "dft", "relax", "label")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _geometry_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    archive_path = tmp_path / "geometry_only_frames.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        info = zipfile.ZipInfo("unit::cscl.extxyz", (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, _canonical_frame(_binary()))
    metadata_path = tmp_path / "next32_cohort.parquet"
    pd.DataFrame(
        {
            "material_id": ["unit::cscl"],
            "source_name": ["unit"],
            "sid": ["cscl"],
            "parent_id": ["parent-cscl"],
            "task_type": ["Structure Optimization"],
            "record_key": [1],
            "natoms": [2],
            "input_role": ["unrelaxed_x0_geometry_only"],
        }
    ).to_parquet(metadata_path, index=False)
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_text(
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
    return archive_path, metadata_path, manifest_path


def test_reused_analytic_families_return_frozen_feature_schema() -> None:
    result = compute_inorganic_response_features(_binary())

    assert tuple(result.features) == INORGANIC_FEATURE_NAMES
    assert set(result.family_supported) == {"contact", "sivr", "madelung", "scbve"}
    assert result.family_supported["contact"]
    assert np.isfinite(result.features["cov_q05"])
    assert not any(
        token in name.lower()
        for name in INORGANIC_FEATURE_NAMES
        for token in ("energy", "force", "stress", "dft", "relax", "label")
    )


def test_batch_requires_label_free_hash_locked_geometry_and_is_no_replace(
    tmp_path: Path,
) -> None:
    archive_path, metadata_path, manifest_path = _geometry_fixture(tmp_path)
    output = tmp_path / "features"

    manifest = build_inorganic_response_feature_batch(
        archive_path=archive_path,
        metadata_path=metadata_path,
        cohort_manifest_path=manifest_path,
        output_dir=output,
    )

    features = pd.read_parquet(output / FEATURE_NAME)
    pauling = pd.read_parquet(output / PAULING_NAME)
    assert features.material_id.tolist() == ["unit::cscl"]
    assert set(INORGANIC_FEATURE_NAMES).issubset(features.columns)
    assert pauling.material_id.tolist() == features.material_id.tolist()
    assert "pauling_p2_p5_decision" in pauling
    assert manifest["labels_opened"] is False
    assert manifest["endpoint_fields_read"] is False
    assert manifest["outputs_sha256"][FEATURE_NAME] == _sha256(output / FEATURE_NAME)
    assert manifest["outputs_sha256"][PAULING_NAME] == _sha256(output / PAULING_NAME)
    assert json.loads((output / MANIFEST_NAME).read_text()) == manifest
    with pytest.raises(FileExistsError):
        build_inorganic_response_feature_batch(
            archive_path=archive_path,
            metadata_path=metadata_path,
            cohort_manifest_path=manifest_path,
            output_dir=output,
        )


@pytest.mark.parametrize("corruption", ["labels", "geometry_hash"])
def test_batch_rejects_opened_labels_or_changed_geometry(
    tmp_path: Path, corruption: str
) -> None:
    archive_path, metadata_path, manifest_path = _geometry_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    if corruption == "labels":
        manifest["labels_opened"] = True
    else:
        manifest["outputs_sha256"][archive_path.name] = "0" * 64
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(ValueError, match="label-free|hash"):
        build_inorganic_response_feature_batch(
            archive_path=archive_path,
            metadata_path=metadata_path,
            cohort_manifest_path=manifest_path,
            output_dir=tmp_path / "features",
        )
