#!/usr/bin/env python3
"""Open converged Alexandria endpoints after NEXT42 predictions are frozen."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

from ase import Atoms
import numpy as np
import pandas as pd
from pymatgen.core import Structure

from src.next11_geometry_only_frames import _load_archive_only
from src.next19_feature_build import _publish_directory_no_replace, _sha256, _strict_json
from src.next23_evaluate import (
    _bootstrap_intervals,
    _continuous_diagnostics,
    _decision_metrics,
)
from src.next23_relaxation_rule import (
    ENDPOINT_COLUMN,
    PRIMARY_GATES,
    PROTECTED_MAX,
    SEVERE_MIN,
    SUBSTANTIAL_MIN,
)
from src.next39_trajectory_evaluate import (
    _fingerprint_environment,
    exact_structure_fingerprint,
    fingerprint_distance,
)
from src.next42_alexandria_cohort import (
    COHORT_NAME,
    GEOMETRY_NAME,
    INPUT_ROLE as COHORT_INPUT_ROLE,
    PROTOCOL as COHORT_PROTOCOL,
)
from src.next42_next23_predictions import (
    PREDICTIONS_NAME,
    PROTOCOL as PREDICTION_PROTOCOL,
)
from src.next18_alexandria_holdout import iter_bz2_object


PROTOCOL = "2026-08-03-next42-alexandria-converged-evaluation-v1"
RESULT_NAME = "NEXT42_ALEXANDRIA_CONVERGED_EVALUATION.json"
JOINED_NAME = "next42_joined_converged_evaluation.parquet"
MANIFEST_NAME = "MANIFEST.json"
FINAL_FORCE_MAX_EV_PER_ANGSTROM = 0.005
FingerprintCalculator = Callable[[Atoms], Sequence[float] | np.ndarray]


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _validate_output_hash(
    manifest: Mapping[str, object], path: Path, *, role: str
) -> None:
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(path.name) != _sha256(path):
        raise ValueError(f"{role} output hash differs")


def final_endpoint(calculations: object) -> tuple[Atoms, float]:
    """Return the last structure and maximum per-atom force norm."""

    if not isinstance(calculations, list):
        raise ValueError("Alexandria trajectory must be a list")
    step: Mapping[str, object] | None = None
    for calculation in reversed(calculations):
        if not isinstance(calculation, Mapping):
            continue
        steps = calculation.get("steps")
        if isinstance(steps, list) and steps:
            candidate = steps[-1]
            if not isinstance(candidate, Mapping):
                raise ValueError("Alexandria final ionic step is invalid")
            step = candidate
            break
    if step is None:
        raise ValueError("Alexandria trajectory lacks a final ionic step")
    raw_structure = step.get("structure")
    if not isinstance(raw_structure, Mapping):
        raise ValueError("Alexandria final ionic step lacks a structure")
    try:
        structure = Structure.from_dict(raw_structure)
    except Exception as exc:
        raise ValueError("invalid Alexandria final structure") from exc
    if len(structure) <= 0 or not all(structure.lattice.pbc):
        raise ValueError("Alexandria final structure must be nonempty and periodic")
    try:
        forces = np.asarray(step.get("forces"), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Alexandria final forces are invalid") from exc
    if forces.shape != (len(structure), 3) or not np.isfinite(forces).all():
        raise ValueError("Alexandria final force shape or values differ")
    maximum = float(np.linalg.norm(forces, axis=1).max())
    return structure.to_ase_atoms(), maximum


def _validate_preopening_inputs(
    *,
    shards: Mapping[str, Path],
    metadata_path: Path,
    frames_zip_path: Path,
    cohort_manifest_path: Path,
    predictions_path: Path,
    prediction_manifest_path: Path,
) -> tuple[
    pd.DataFrame,
    list[Atoms],
    pd.DataFrame,
    dict[str, object],
    dict[str, str],
]:
    paths = {
        **shards,
        "metadata": metadata_path,
        "geometry": frames_zip_path,
        "cohort_manifest": cohort_manifest_path,
        "predictions": predictions_path,
        "prediction_manifest": prediction_manifest_path,
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT42 evaluation input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    cohort = _strict_json(cohort_manifest_path, role="NEXT42 cohort manifest")
    prediction = _strict_json(
        prediction_manifest_path, role="NEXT42 prediction manifest"
    )
    selection = cohort.get("selection")
    if (
        cohort.get("protocol") != COHORT_PROTOCOL
        or cohort.get("input_role") != COHORT_INPUT_ROLE
        or cohort.get("later_geometry_accessed") is not False
        or cohort.get("dft_values_read") is not False
        or cohort.get("mlip_prerelaxation_used") is not False
        or cohort.get("physical_relaxation_executed") is not False
        or not isinstance(selection, Mapping)
        or selection.get("endpoint_fields_used") is not False
        or selection.get("sampled") is not False
    ):
        raise ValueError("NEXT42 cohort crossed the blind-evaluation boundary")
    if (
        prediction.get("protocol") != PREDICTION_PROTOCOL
        or prediction.get("input_role") != "one_raw_pre_dft_pre_mlip_x0_only"
        or prediction.get("later_geometry_opened") is not False
        or prediction.get("dft_values_read") is not False
        or prediction.get("thresholds_refit") is not False
        or prediction.get("model_or_proxy_potential_used") is not False
        or prediction.get("physical_relaxation_executed") is not False
        or not isinstance(prediction.get("frozen_at_utc"), str)
    ):
        raise ValueError("NEXT42 predictions were not frozen before endpoint opening")
    if metadata_path.name != COHORT_NAME or frames_zip_path.name != GEOMETRY_NAME:
        raise ValueError("NEXT42 cohort filenames differ")
    if predictions_path.name != PREDICTIONS_NAME:
        raise ValueError("NEXT42 prediction filename differs")
    _validate_output_hash(cohort, metadata_path, role="cohort metadata")
    _validate_output_hash(cohort, frames_zip_path, role="cohort geometry")
    _validate_output_hash(prediction, predictions_path, role="predictions")
    cohort_inputs = cohort.get("inputs_sha256")
    if not isinstance(cohort_inputs, Mapping) or any(
        cohort_inputs.get(role) != hashes[role] for role in shards
    ):
        raise ValueError("NEXT42 raw trajectory hashes differ from frozen cohort")
    prediction_inputs = prediction.get("inputs_sha256")
    expected_prediction_inputs = {
        "metadata": hashes["metadata"],
        "geometry": hashes["geometry"],
        "cohort_manifest": hashes["cohort_manifest"],
    }
    if not isinstance(prediction_inputs, Mapping) or any(
        prediction_inputs.get(name) != digest
        for name, digest in expected_prediction_inputs.items()
    ):
        raise ValueError("NEXT42 prediction does not bind the frozen cohort")

    metadata = pd.read_parquet(metadata_path).sort_values(
        "material_id", kind="stable", ignore_index=True
    )
    predictions = pd.read_parquet(predictions_path).sort_values(
        "material_id", kind="stable", ignore_index=True
    )
    required_metadata = {
        "material_id",
        "source_family",
        "source_shard",
        "natoms",
        "input_role",
    }
    required_predictions = {
        "material_id",
        "source_family",
        "next23_supported",
        "next23_score",
        "next23_reject",
        "pauling_p2_p5_decision",
    }
    if not required_metadata.issubset(metadata.columns) or not required_predictions.issubset(
        predictions.columns
    ):
        raise ValueError("NEXT42 evaluation table schema differs")
    if (
        metadata.empty
        or metadata.material_id.astype(str).duplicated().any()
        or predictions.material_id.astype(str).duplicated().any()
        or metadata.material_id.astype(str).tolist()
        != predictions.material_id.astype(str).tolist()
        or metadata.source_family.astype(str).tolist()
        != predictions.source_family.astype(str).tolist()
    ):
        raise ValueError("NEXT42 evaluation identities differ")
    ids = tuple(metadata.material_id.astype(str))
    loaded, initial = _load_archive_only(frames_zip_path, ids)
    if loaded != list(ids) or any(
        len(atoms) != int(natoms)
        for atoms, natoms in zip(initial, metadata.natoms, strict=True)
    ):
        raise ValueError("NEXT42 initial geometry identity differs")
    return metadata, initial, predictions, prediction, hashes


def evaluate_next42(
    *,
    shard_0000_path: Path,
    shard_0001_path: Path,
    metadata_path: Path,
    frames_zip_path: Path,
    cohort_manifest_path: Path,
    predictions_path: Path,
    prediction_manifest_path: Path,
    output_dir: Path,
    fingerprint_calculator: FingerprintCalculator | None = None,
) -> dict[str, object]:
    """Evaluate frozen decisions only on force-converged final structures."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    shards = {
        "pbe_0000": Path(shard_0000_path).resolve(),
        "pbe_0001": Path(shard_0001_path).resolve(),
    }
    metadata_path = Path(metadata_path).resolve()
    frames_zip_path = Path(frames_zip_path).resolve()
    cohort_manifest_path = Path(cohort_manifest_path).resolve()
    predictions_path = Path(predictions_path).resolve()
    prediction_manifest_path = Path(prediction_manifest_path).resolve()
    metadata, initial_structures, predictions, prediction_manifest, input_hashes = (
        _validate_preopening_inputs(
            shards=shards,
            metadata_path=metadata_path,
            frames_zip_path=frames_zip_path,
            cohort_manifest_path=cohort_manifest_path,
            predictions_path=predictions_path,
            prediction_manifest_path=prediction_manifest_path,
        )
    )
    calculator = (
        exact_structure_fingerprint
        if fingerprint_calculator is None
        else fingerprint_calculator
    )
    production = bool(
        fingerprint_calculator is None
        and prediction_manifest.get("production_protocol_eligible") is True
    )
    if fingerprint_calculator is None:
        fingerprint_environment: dict[str, object] = _fingerprint_environment()
        if importlib.metadata.version("matminer") != "0.10.1":
            raise RuntimeError("NEXT42 production Matminer version differs")
    else:
        fingerprint_environment = {"calculator": "injected_test_double"}

    initial_by_id = {
        str(material_id): atoms
        for material_id, atoms in zip(
            metadata.material_id.astype(str), initial_structures, strict=True
        )
    }
    expected_ids = set(initial_by_id)
    endpoint_rows: list[dict[str, object]] = []
    failures: Counter[str] = Counter()
    seen: set[str] = set()
    started = time.perf_counter()
    for source_shard, path in shards.items():
        for material_id, calculations in iter_bz2_object(path):
            if material_id not in expected_ids:
                continue
            if material_id in seen:
                raise ValueError(f"duplicate NEXT42 endpoint identity: {material_id}")
            seen.add(material_id)
            failure: str | None = None
            force_max = math.nan
            change = math.nan
            fingerprint_supported = False
            try:
                final, force_max = final_endpoint(calculations)
                change = fingerprint_distance(
                    initial_by_id[material_id], final, calculator=calculator
                )
                fingerprint_supported = True
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
                failures[failure] += 1
            force_converged = bool(
                math.isfinite(force_max)
                and force_max <= FINAL_FORCE_MAX_EV_PER_ANGSTROM
            )
            endpoint_rows.append(
                {
                    "material_id": material_id,
                    "endpoint_source_shard": source_shard,
                    "final_force_max_eV_per_A": force_max,
                    "force_converged": force_converged,
                    ENDPOINT_COLUMN: change,
                    "fingerprint_supported": fingerprint_supported,
                    "primary_evaluation_supported": bool(
                        force_converged and fingerprint_supported
                    ),
                    "evaluation_failure": failure,
                }
            )
    elapsed = time.perf_counter() - started
    missing = expected_ids - seen
    if missing:
        raise ValueError(f"{len(missing)} NEXT42 endpoints are missing")
    endpoints = pd.DataFrame(endpoint_rows).sort_values(
        "material_id", kind="stable", ignore_index=True
    )
    joined = predictions.merge(
        endpoints, on="material_id", how="left", validate="one_to_one"
    ).merge(
        metadata[["material_id", "source_shard"]],
        on="material_id",
        how="left",
        validate="one_to_one",
    )
    primary_mask = joined.primary_evaluation_supported.to_numpy(dtype=bool)
    if not primary_mask.any():
        raise ValueError("no force-converged NEXT42 endpoint could be evaluated")
    evaluated = joined.loc[primary_mask].reset_index(drop=True)
    endpoint = evaluated[ENDPOINT_COLUMN].to_numpy(dtype=float)
    next23_supported = evaluated.next23_supported.to_numpy(dtype=bool)
    next23_reject = evaluated.next23_reject.to_numpy(dtype=bool)
    pauling_decision = evaluated.pauling_p2_p5_decision.astype(str)
    pauling_supported = pauling_decision.ne("ABSTAIN").to_numpy(dtype=bool)
    pauling_reject = pauling_decision.eq("REJECT").to_numpy(dtype=bool)
    next23_metrics = _decision_metrics(
        supported=next23_supported, reject=next23_reject, endpoint=endpoint
    )
    pauling_metrics = _decision_metrics(
        supported=pauling_supported, reject=pauling_reject, endpoint=endpoint
    )
    endpoint_counts = {
        "protected": int((endpoint <= PROTECTED_MAX).sum()),
        "changed": int((endpoint > PROTECTED_MAX).sum()),
        "substantial": int((endpoint >= SUBSTANTIAL_MIN).sum()),
        "severe": int((endpoint >= SEVERE_MIN).sum()),
    }
    finite_forces = endpoints.loc[
        np.isfinite(endpoints.final_force_max_eV_per_A), "final_force_max_eV_per_A"
    ].to_numpy(dtype=float)
    force_quantiles = {
        str(q): float(np.quantile(finite_forces, q))
        for q in (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)
    }
    result: dict[str, object] = {
        "protocol": PROTOCOL,
        "endpoint": {
            "column": ENDPOINT_COLUMN,
            "role": "raw_x0_to_force_converged_DFT_final_structure_change",
            "fingerprint": "CrystalNNFingerprint ops plus SiteStats mean/std_dev/minimum/maximum L2",
            "protected_max": PROTECTED_MAX,
            "substantial_min": SUBSTANTIAL_MIN,
            "severe_min": SEVERE_MIN,
            "final_force_max_eV_per_A": FINAL_FORCE_MAX_EV_PER_ANGSTROM,
            "convergence_source": "https://arxiv.org/html/2512.09169v2#S4.SS1",
        },
        "primary_gates": dict(PRIMARY_GATES),
        "rows": len(joined),
        "force_converged_rows": int(joined.force_converged.sum()),
        "fingerprint_supported_rows": int(joined.fingerprint_supported.sum()),
        "primary_evaluation_rows": int(primary_mask.sum()),
        "endpoint_failure_counts": dict(sorted(failures.items())),
        "final_force_quantiles_eV_per_A": force_quantiles,
        "endpoint_class_counts": endpoint_counts,
        "next23": {
            "decision_metrics": next23_metrics,
            "continuous_diagnostics": _continuous_diagnostics(
                evaluated.next23_score.to_numpy(dtype=float),
                next23_supported,
                endpoint,
            ),
            "bootstrap": _bootstrap_intervals(
                next23_supported, next23_reject, endpoint
            ),
        },
        "pauling_p2_p5": {"decision_metrics": pauling_metrics},
        "interpretation_scope": (
            "source-qualified raw-x0 structural-change validation against force-converged "
            "DFT endpoints; not a thermodynamic hull or dynamical-stability claim"
        ),
    }

    repository = Path(__file__).resolve().parents[1]
    source_names = (
        "src/next18_alexandria_holdout.py",
        "src/next23_evaluate.py",
        "src/next39_trajectory_evaluate.py",
        "src/next42_alexandria_cohort.py",
        "src/next42_next23_predictions.py",
        "src/next42_alexandria_evaluate.py",
    )
    source_hashes = {name: _sha256(repository / name) for name in source_names}
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "later_geometry_opened_after_prediction_freeze": True,
        "evaluation_only_dft_final_geometry_read": True,
        "evaluation_only_dft_forces_read": True,
        "evaluation_only_dft_energy_read": False,
        "evaluation_only_dft_stress_read": False,
        "law_execution_dft_values_read": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
        "thresholds_refit": False,
        "cohort_changed_after_opening": False,
        "counts": {
            "rows": len(joined),
            "force_converged": int(joined.force_converged.sum()),
            "primary_evaluation": int(primary_mask.sum()),
        },
        "execution": {"wall_time_seconds": elapsed},
        "inputs_sha256_before_endpoint_opening": input_hashes,
        "fingerprint_environment": fingerprint_environment,
        "executed_source_sha256": source_hashes,
        "production_protocol_eligible": production,
        "frozen_rule_passes_primary_gates": bool(
            next23_metrics["passes_primary_gates"]
        ),
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        joined_path = staging / JOINED_NAME
        result_path = staging / RESULT_NAME
        joined.to_parquet(joined_path, index=False)
        result_path.write_bytes(_json_bytes(result))
        manifest["outputs_sha256"] = {
            JOINED_NAME: _sha256(joined_path),
            RESULT_NAME: _sha256(result_path),
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        input_paths = {
            **shards,
            "metadata": metadata_path,
            "geometry": frames_zip_path,
            "cohort_manifest": cohort_manifest_path,
            "predictions": predictions_path,
            "prediction_manifest": prediction_manifest_path,
        }
        if any(
            _sha256(path) != input_hashes[name]
            for name, path in input_paths.items()
        ):
            raise RuntimeError("NEXT42 evaluation input changed during publication")
        if any(
            _sha256(repository / name) != digest
            for name, digest in source_hashes.items()
        ):
            raise RuntimeError("NEXT42 evaluation source changed during publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbe-0000", type=Path, required=True)
    parser.add_argument("--pbe-0001", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    evaluate_next42(
        shard_0000_path=args.pbe_0000,
        shard_0001_path=args.pbe_0001,
        metadata_path=args.metadata,
        frames_zip_path=args.geometry,
        cohort_manifest_path=args.cohort_manifest,
        predictions_path=args.predictions,
        prediction_manifest_path=args.prediction_manifest,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
