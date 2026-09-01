from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pandas as pd
from pymatgen.core import Lattice, Structure

from src.next93b_wyformer_blind_lockbox import (
    GEOMETRY_NAMES,
    INPUT_ROLE,
    MANIFEST_NAME as COHORT_MANIFEST_NAME,
    METADATA_NAME,
    PARTITIONS,
    PROTOCOL as COHORT_PROTOCOL,
)
from src.next94_wyformer_label_free_features import (
    FEATURE_NAMES,
    MANIFEST_NAME,
    build_wyformer_label_free_features,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_cohort(root: Path) -> Path:
    cohort = root / "cohort"
    cohort.mkdir()
    structures = {
        "discovery": Structure(Lattice.cubic(5.5), ["Na", "Cl"], [[0, 0, 0], [0.5] * 3]),
        "internal_validation": Structure(
            Lattice.cubic(5.0), ["Li", "Li", "O"], [[0, 0, 0], [0.5, 0.5, 0], [0.25] * 3]
        ),
        "internal_replication": Structure(
            Lattice.cubic(4.5), ["Mg", "O"], [[0, 0, 0], [0.5] * 3]
        ),
    }
    metadata_rows = []
    hashes: dict[str, str] = {}
    for index, role in enumerate(PARTITIONS):
        material_id = f"test-{index}"
        structure = structures[role]
        geometry = pd.DataFrame(
            {
                "material_id": [material_id],
                "structure_json": [json.dumps(structure.as_dict(), sort_keys=True)],
            }
        )
        path = cohort / GEOMETRY_NAMES[role]
        geometry.to_parquet(path, index=False)
        hashes[GEOMETRY_NAMES[role]] = _sha(path)
        metadata_rows.append(
            {
                "material_id": material_id,
                "raw_material_id": index,
                "full_composition_key": str(index),
                "reduced_formula": structure.composition.reduced_formula,
                "chemical_system": "-".join(sorted(str(x) for x in structure.composition.elements)),
                "natoms": len(structure),
                "generated_space_group": 225,
                "crystal_system": "cubic",
                "partition_role": role,
                "input_role": INPUT_ROLE,
            }
        )
    metadata = pd.DataFrame(metadata_rows)
    metadata.to_parquet(cohort / METADATA_NAME, index=False)
    hashes[METADATA_NAME] = _sha(cohort / METADATA_NAME)
    manifest = {
        "protocol": COHORT_PROTOCOL,
        "labels_opened_by_feature_builder": False,
        "discovery_endpoint_opened": False,
        "validation_endpoint_opened": False,
        "replication_endpoint_opened": False,
        "relaxed_structures_published": False,
        "learned_proxy_execution_input": False,
        "outputs_sha256": hashes,
    }
    (cohort / COHORT_MANIFEST_NAME).write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return cohort


def test_builder_signature_has_no_endpoint_argument() -> None:
    parameters = inspect.signature(build_wyformer_label_free_features).parameters
    assert not any("endpoint" in name for name in parameters)


def test_builder_freezes_all_partitions_without_dft_columns(tmp_path: Path) -> None:
    cohort = _make_cohort(tmp_path)
    design = tmp_path / "design.md"
    design.write_text("frozen", encoding="utf-8")
    output = tmp_path / "features"
    manifest = build_wyformer_label_free_features(
        cohort_dir=cohort,
        design_path=design,
        output_dir=output,
        workers=1,
        require_formal_inputs=False,
    )
    assert manifest["labels_opened"] is False
    assert manifest["endpoint_payloads_opened"] is False
    assert manifest["dft_values_used_by_features"] is False
    assert manifest["learned_energy_force_stress_proxy_used"] is False
    assert (output / MANIFEST_NAME).is_file()
    for role in PARTITIONS:
        frame = pd.read_parquet(output / FEATURE_NAMES[role])
        assert len(frame) == 1
        assert "pauling_p2_p5_decision" in frame
        assert "dft_succeeded" not in frame
        assert "dft_e_above_hull_corrected" not in frame
        assert "endpoint_stratum" not in frame
