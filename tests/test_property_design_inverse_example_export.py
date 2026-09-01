from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure

from experiments.property_design_20260821.export_inverse_examples import (
    EXPORT_PROTOCOL_VERSION,
    export_inverse_examples,
)


def _structure_hash(structure: Structure) -> str:
    payload = json.dumps(
        structure.as_dict(), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _fixture(
    tmp_path: Path,
    *,
    species: tuple[str, str] = ("Si", "Si"),
) -> tuple[Path, Path, bytes, str]:
    structure = Structure(
        Lattice.cubic(3.5),
        list(species),
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    raw_cif = structure.to(fmt="cif").encode()
    parsed = Structure.from_str(raw_cif.decode(), fmt="cif")
    structure_sha256 = _structure_hash(parsed)
    cif_sha256 = hashlib.sha256(raw_cif).hexdigest()
    formula = str(parsed.composition.reduced_formula)

    archive_path = tmp_path / "generated_crystals_cif.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as handle:
        handle.writestr("crystal_000000.cif", raw_cif)

    source_manifest_sha256 = "a" * 64
    manifest = {
        "protocol_version": "2026-08-21-mattergen-shard-aggregate-v3",
        "source": "mattergen",
        "outputs": {
            "archive_name": archive_path.name,
            "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "structure_count": 1,
            "candidates": [
                {
                    "aggregate_member": "crystal_000000.cif",
                    "source_shard": "/formal/shard_00",
                    "source_member": "generated_crystals/0.cif",
                    "source_manifest_sha256": source_manifest_sha256,
                    "source_seed": 20260830,
                    "cif_sha256": cif_sha256,
                    "structure_sha256": structure_sha256,
                    "formula": formula,
                    "num_sites": 2,
                }
            ],
        },
    }
    manifest_path = tmp_path / "aggregate_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    examples = pd.DataFrame(
        [
            {
                "example_role": "robust_D7_pass",
                "candidate_id": "candidate_0000",
                "archive_member": "crystal_000000.cif",
                "source_shard": "/formal/shard_00",
                "source_member": "generated_crystals/0.cif",
                "source_manifest_sha256": source_manifest_sha256,
                "source_seed": 20260830,
                "cif_sha256": cif_sha256,
                "structure_sha256": structure_sha256,
                "formula": formula,
                "num_sites": 2,
                "clamped_bulk_modulus_proxy_gpa": 250.0,
            }
        ]
    )
    examples_path = tmp_path / "inverse_examples.csv"
    examples.to_csv(examples_path, index=False)
    return examples_path, manifest_path, raw_cif, structure_sha256


def test_export_inverse_examples_preserves_exact_cif_and_provenance(
    tmp_path: Path,
) -> None:
    examples_path, manifest_path, raw_cif, structure_sha256 = _fixture(tmp_path)
    output_dir = tmp_path / "inverse_typical_cifs"

    result = export_inverse_examples(examples_path, manifest_path, output_dir)

    assert result["protocol_version"] == EXPORT_PROTOCOL_VERSION
    assert result["source"] == "mattergen"
    assert result["example_count"] == 1
    record = result["examples"][0]
    exported_path = output_dir / record["exported_cif"]
    assert exported_path.read_bytes() == raw_cif
    assert record["structure_sha256"] == structure_sha256
    assert record["source_member"] == "generated_crystals/0.cif"
    assert hashlib.sha256(exported_path.read_bytes()).hexdigest() == record["cif_sha256"]

    written = json.loads((output_dir / "manifest.json").read_text())
    assert written == result


def test_export_inverse_examples_fails_closed_on_csv_hash_mismatch(
    tmp_path: Path,
) -> None:
    examples_path, manifest_path, _, _ = _fixture(tmp_path)
    examples = pd.read_csv(examples_path)
    examples.loc[0, "cif_sha256"] = "b" * 64
    examples.to_csv(examples_path, index=False)

    with pytest.raises(ValueError, match="provenance mismatch.*cif_sha256"):
        export_inverse_examples(
            examples_path,
            manifest_path,
            tmp_path / "inverse_typical_cifs",
        )


def test_export_inverse_examples_refuses_existing_output_directory(
    tmp_path: Path,
) -> None:
    examples_path, manifest_path, _, _ = _fixture(tmp_path)
    output_dir = tmp_path / "inverse_typical_cifs"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="refusing to reuse"):
        export_inverse_examples(examples_path, manifest_path, output_dir)


def test_export_inverse_examples_preserves_literal_nan_chemical_formula(
    tmp_path: Path,
) -> None:
    examples_path, manifest_path, _, _ = _fixture(
        tmp_path,
        species=("Na", "N"),
    )

    result = export_inverse_examples(
        examples_path,
        manifest_path,
        tmp_path / "inverse_typical_cifs",
    )

    assert result["examples"][0]["formula"] == "NaN"
