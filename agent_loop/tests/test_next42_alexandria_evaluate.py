"""Contracts for opening converged NEXT42 endpoints only after prediction freeze."""

from __future__ import annotations

import bz2
import hashlib
import json
from pathlib import Path
import zipfile

from ase import Atoms
import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure
import pytest

from src.next11_geometry_only_frames import _canonical_frame


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _calculation(
    initial: Structure, *, displacement: float, final_force: float
) -> list[dict]:
    final = initial.copy()
    final.translate_sites([0], [displacement, 0.0, 0.0], frac_coords=False)
    return [
        {
            "steps": [
                {
                    "structure": initial.as_dict(),
                    "energy": 111.0,
                    "forces": [[1.0, 0.0, 0.0]] * len(initial),
                    "stress": [[2.0] * 3] * 3,
                },
                {
                    "structure": final.as_dict(),
                    "energy": -999.0,
                    "forces": [[final_force, 0.0, 0.0]] * len(initial),
                    "stress": [[3.0] * 3] * 3,
                },
            ]
        }
    ]


def _write_shard(path: Path, rows: dict[str, list[dict]]) -> None:
    with bz2.open(path, "wt", encoding="utf-8") as stream:
        json.dump(rows, stream, sort_keys=True)


def _inputs(tmp_path: Path) -> dict[str, Path]:
    from src.next42_alexandria_cohort import (
        COHORT_NAME,
        GEOMETRY_NAME,
        INPUT_ROLE,
        PROTOCOL as COHORT_PROTOCOL,
    )
    from src.next42_next23_predictions import (
        PREDICTIONS_NAME,
        PROTOCOL as PREDICTION_PROTOCOL,
    )

    lattice = Lattice.cubic(5.0)
    structures = {
        "a": Structure(lattice, ["Li", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
        "b": Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
        "c": Structure(lattice, ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
    }
    shard0 = tmp_path / "pbe_0000.json.bz2"
    shard1 = tmp_path / "pbe_0001.json.bz2"
    _write_shard(
        shard0,
        {
            "a": _calculation(structures["a"], displacement=0.05, final_force=0.004),
            "b": _calculation(structures["b"], displacement=0.60, final_force=0.005),
        },
    )
    _write_shard(
        shard1,
        {"c": _calculation(structures["c"], displacement=0.30, final_force=0.006)},
    )

    cohort = tmp_path / "cohort"
    cohort.mkdir()
    metadata = pd.DataFrame(
        [
            {
                "material_id": material_id,
                "source_family": "cgat_comp/ternaries",
                "source_shard": "pbe_0000" if material_id != "c" else "pbe_0001",
                "formula": structure.composition.formula.replace(" ", ""),
                "reduced_formula": structure.composition.reduced_formula,
                "natoms": len(structure),
                "input_role": INPUT_ROLE,
            }
            for material_id, structure in structures.items()
        ]
    )
    metadata_path = cohort / COHORT_NAME
    metadata.to_parquet(metadata_path, index=False)
    geometry_path = cohort / GEOMETRY_NAME
    with zipfile.ZipFile(geometry_path, "w") as archive:
        for material_id in sorted(structures):
            info = zipfile.ZipInfo(
                f"{material_id}.extxyz", date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.extra = b""
            info.comment = b""
            archive.writestr(
                info, _canonical_frame(structures[material_id].to_ase_atoms())
            )
    cohort_manifest = {
        "protocol": COHORT_PROTOCOL,
        "input_role": INPUT_ROLE,
        "later_geometry_accessed": False,
        "dft_values_read": False,
        "mlip_prerelaxation_used": False,
        "physical_relaxation_executed": False,
        "selection": {"endpoint_fields_used": False, "sampled": False},
        "inputs_sha256": {
            "pbe_0000": _sha(shard0),
            "pbe_0001": _sha(shard1),
            "source_table": "test-source-table",
            "source_manifest": "test-source-manifest",
        },
        "outputs_sha256": {
            COHORT_NAME: _sha(metadata_path),
            GEOMETRY_NAME: _sha(geometry_path),
        },
    }
    cohort_manifest_path = cohort / "MANIFEST.json"
    cohort_manifest_path.write_text(json.dumps(cohort_manifest), encoding="utf-8")

    predictions_root = tmp_path / "predictions"
    predictions_root.mkdir()
    predictions = pd.DataFrame(
        [
            {
                "material_id": "a",
                "source_family": "cgat_comp/ternaries",
                "natoms": 2,
                "next23_supported": True,
                "next23_score": -1.0,
                "next23_reject": False,
                "pauling_p2_p5_decision": "KEEP",
            },
            {
                "material_id": "b",
                "source_family": "cgat_comp/ternaries",
                "natoms": 2,
                "next23_supported": True,
                "next23_score": 3.0,
                "next23_reject": True,
                "pauling_p2_p5_decision": "REJECT",
            },
            {
                "material_id": "c",
                "source_family": "cgat_comp/ternaries",
                "natoms": 2,
                "next23_supported": True,
                "next23_score": 2.0,
                "next23_reject": True,
                "pauling_p2_p5_decision": "ABSTAIN",
            },
        ]
    )
    predictions_path = predictions_root / PREDICTIONS_NAME
    predictions.to_parquet(predictions_path, index=False)
    prediction_inputs = {
        "metadata": _sha(metadata_path),
        "geometry": _sha(geometry_path),
        "cohort_manifest": _sha(cohort_manifest_path),
        "frozen_rule": "frozen-rule",
        "frozen_rule_manifest": "frozen-rule-manifest",
    }
    prediction_manifest = {
        "protocol": PREDICTION_PROTOCOL,
        "input_role": "one_raw_pre_dft_pre_mlip_x0_only",
        "frozen_at_utc": "2026-08-03T00:00:00+00:00",
        "later_geometry_opened": False,
        "dft_values_read": False,
        "thresholds_refit": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
        "production_protocol_eligible": False,
        "inputs_sha256": prediction_inputs,
        "outputs_sha256": {PREDICTIONS_NAME: _sha(predictions_path)},
    }
    prediction_manifest_path = predictions_root / "MANIFEST.json"
    prediction_manifest_path.write_text(
        json.dumps(prediction_manifest), encoding="utf-8"
    )
    return {
        "shard0": shard0,
        "shard1": shard1,
        "metadata": metadata_path,
        "geometry": geometry_path,
        "cohort_manifest": cohort_manifest_path,
        "predictions": predictions_path,
        "prediction_manifest": prediction_manifest_path,
    }


def test_final_endpoint_uses_last_nonempty_step_and_force_norm() -> None:
    from src.next42_alexandria_evaluate import final_endpoint

    structure = Structure(
        Lattice.cubic(5.0), ["Li", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    calculations = _calculation(structure, displacement=0.2, final_force=0.003)
    final, force = final_endpoint(calculations)
    assert len(final) == 2
    assert force == pytest.approx(0.003)


def test_evaluation_excludes_nonconverged_rows_and_never_refits(tmp_path: Path) -> None:
    from src import next42_alexandria_evaluate as module

    paths = _inputs(tmp_path)

    def fingerprint(atoms: Atoms) -> np.ndarray:
        return np.asarray([atoms.positions[0, 0]], dtype=float)

    target = tmp_path / "evaluation"
    manifest = module.evaluate_next42(
        shard_0000_path=paths["shard0"],
        shard_0001_path=paths["shard1"],
        metadata_path=paths["metadata"],
        frames_zip_path=paths["geometry"],
        cohort_manifest_path=paths["cohort_manifest"],
        predictions_path=paths["predictions"],
        prediction_manifest_path=paths["prediction_manifest"],
        output_dir=target,
        fingerprint_calculator=fingerprint,
    )
    result = json.loads((target / module.RESULT_NAME).read_text(encoding="utf-8"))
    joined = pd.read_parquet(target / module.JOINED_NAME)
    assert result["rows"] == 3
    assert result["force_converged_rows"] == 2
    assert result["primary_evaluation_rows"] == 2
    assert result["endpoint_class_counts"] == {
        "changed": 1,
        "protected": 1,
        "severe": 1,
        "substantial": 1,
    }
    assert joined.set_index("material_id").loc["c", "force_converged"] == False
    assert manifest["later_geometry_opened_after_prediction_freeze"] is True
    assert manifest["evaluation_only_dft_forces_read"] is True
    assert manifest["law_execution_dft_values_read"] is False
    assert manifest["thresholds_refit"] is False
    assert manifest["model_or_proxy_potential_used"] is False
    with pytest.raises(FileExistsError):
        module.evaluate_next42(
            shard_0000_path=paths["shard0"],
            shard_0001_path=paths["shard1"],
            metadata_path=paths["metadata"],
            frames_zip_path=paths["geometry"],
            cohort_manifest_path=paths["cohort_manifest"],
            predictions_path=paths["predictions"],
            prediction_manifest_path=paths["prediction_manifest"],
            output_dir=target,
            fingerprint_calculator=fingerprint,
        )


def test_evaluation_refuses_predictions_not_bound_to_cohort(tmp_path: Path) -> None:
    from src.next42_alexandria_evaluate import evaluate_next42

    paths = _inputs(tmp_path)
    manifest = json.loads(paths["prediction_manifest"].read_text(encoding="utf-8"))
    manifest["inputs_sha256"]["geometry"] = "changed"
    paths["prediction_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind the frozen cohort"):
        evaluate_next42(
            shard_0000_path=paths["shard0"],
            shard_0001_path=paths["shard1"],
            metadata_path=paths["metadata"],
            frames_zip_path=paths["geometry"],
            cohort_manifest_path=paths["cohort_manifest"],
            predictions_path=paths["predictions"],
            prediction_manifest_path=paths["prediction_manifest"],
            output_dir=tmp_path / "bad",
            fingerprint_calculator=lambda _atoms: np.zeros(1),
        )
