#!/usr/bin/env python3
"""Open OMat24 later geometries only after NEXT39 predictions are frozen."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

from ase import Atoms
import lmdb
import numpy as np
import pandas as pd

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
from src.next32_omat24_cohort import _record_payload, project_identity_geometry
from src.next39_next23_predictions import (
    PREDICTIONS_NAME,
    PROTOCOL as PREDICTION_PROTOCOL,
)
from src.next39_omat24_trajectory_cohort import (
    COHORT_NAME,
    GEOMETRY_NAME,
    PROTOCOL as COHORT_PROTOCOL,
    TASK_TYPE,
)


PROTOCOL = "2026-08-03-next39-omat24-trajectory-blind-evaluation-v1"
RESULT_NAME = "NEXT39_OMAT24_TRAJECTORY_EVALUATION.json"
JOINED_NAME = "next39_joined_trajectory_evaluation.parquet"
MANIFEST_NAME = "MANIFEST.json"
FingerprintCalculator = Callable[[Atoms], Sequence[float] | np.ndarray]


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _validate_output_hash(
    manifest: Mapping[str, object], path: Path, *, role: str
) -> None:
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(path.name) != _sha256(path):
        raise ValueError(f"{role} output hash differs")


def exact_structure_fingerprint(atoms: Atoms) -> np.ndarray:
    """Return the historical Matbench Discovery CrystalNN site-stat fingerprint."""

    version = importlib.metadata.version("matminer")
    if version != "0.10.1":
        raise RuntimeError(
            f"NEXT39 exact fingerprint requires sealed matminer 0.10.1, found {version}"
        )
    from matminer.featurizers.site import CrystalNNFingerprint
    from matminer.featurizers.structure import SiteStatsFingerprint
    from pymatgen.io.ase import AseAtomsAdaptor

    structure = AseAtomsAdaptor.get_structure(atoms)
    site = CrystalNNFingerprint.from_preset("ops")
    calculator = SiteStatsFingerprint(
        site, stats=("mean", "std_dev", "minimum", "maximum")
    )
    values = np.asarray(calculator.featurize(structure), dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("structure fingerprint is invalid")
    return values


def _fingerprint_environment() -> dict[str, object]:
    """Hash the exact third-party implementations used for production labels."""

    from matminer.featurizers.site import CrystalNNFingerprint
    from matminer.featurizers.structure import SiteStatsFingerprint

    source_paths = {
        "CrystalNNFingerprint": Path(inspect.getsourcefile(CrystalNNFingerprint) or ""),
        "SiteStatsFingerprint": Path(inspect.getsourcefile(SiteStatsFingerprint) or ""),
    }
    if any(not path.is_file() for path in source_paths.values()):
        raise RuntimeError("cannot resolve sealed Matminer fingerprint source")
    return {
        "matminer_version": importlib.metadata.version("matminer"),
        "pymatgen_version": importlib.metadata.version("pymatgen"),
        "numpy_version": importlib.metadata.version("numpy"),
        "source_sha256": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in source_paths.items()
        },
    }


def fingerprint_distance(
    initial: Atoms,
    later: Atoms,
    *,
    calculator: FingerprintCalculator = exact_structure_fingerprint,
) -> float:
    """Compute fingerprint L2 distance after exact atom-order identity checks."""

    if not np.array_equal(initial.numbers, later.numbers):
        raise ValueError("initial and later atomic identities differ")
    first = np.asarray(calculator(initial), dtype=float)
    second = np.asarray(calculator(later), dtype=float)
    if (
        first.ndim != 1
        or second.shape != first.shape
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
    ):
        raise ValueError("structure fingerprints are invalid or misaligned")
    return float(np.linalg.norm(second - first))


def decision_metrics(
    *, supported: np.ndarray, reject: np.ndarray, endpoint: np.ndarray
) -> dict[str, object]:
    """Expose the unchanged NEXT23 operational metrics and Wilson gates."""

    return _decision_metrics(supported=supported, reject=reject, endpoint=endpoint)


def _validate_preopening_inputs(
    *,
    database: Path,
    metadata_path: Path,
    frames_zip_path: Path,
    cohort_manifest_path: Path,
    predictions_path: Path,
    prediction_manifest_path: Path,
) -> tuple[pd.DataFrame, list[Atoms], pd.DataFrame, dict[str, object], dict[str, str]]:
    paths = {
        "aselmdb": database,
        "metadata": metadata_path,
        "geometry": frames_zip_path,
        "cohort_manifest": cohort_manifest_path,
        "predictions": predictions_path,
        "prediction_manifest": prediction_manifest_path,
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT39 evaluation input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    cohort_manifest = _strict_json(cohort_manifest_path, role="NEXT39 cohort manifest")
    prediction_manifest = _strict_json(
        prediction_manifest_path, role="NEXT39 prediction manifest"
    )
    if (
        cohort_manifest.get("protocol") != COHORT_PROTOCOL
        or cohort_manifest.get("later_geometry_opened") is not False
        or cohort_manifest.get("dft_numeric_fields_parsed") is not False
        or cohort_manifest.get("dft_values_read") is not False
    ):
        raise ValueError("NEXT39 cohort crossed the blind-evaluation boundary")
    if (
        prediction_manifest.get("protocol") != PREDICTION_PROTOCOL
        or prediction_manifest.get("later_geometry_opened") is not False
        or prediction_manifest.get("dft_values_read") is not False
        or prediction_manifest.get("thresholds_refit") is not False
        or not isinstance(prediction_manifest.get("frozen_at_utc"), str)
    ):
        raise ValueError("NEXT39 predictions were not frozen before opening")
    if metadata_path.name != COHORT_NAME or frames_zip_path.name != GEOMETRY_NAME:
        raise ValueError("NEXT39 cohort filenames differ")
    if predictions_path.name != PREDICTIONS_NAME:
        raise ValueError("NEXT39 prediction filename differs")
    _validate_output_hash(cohort_manifest, metadata_path, role="cohort metadata")
    _validate_output_hash(cohort_manifest, frames_zip_path, role="cohort geometry")
    _validate_output_hash(prediction_manifest, predictions_path, role="predictions")

    source_db = cohort_manifest.get("inputs_sha256")
    if (
        not isinstance(source_db, Mapping)
        or not isinstance(source_db.get("aselmdb"), Mapping)
        or source_db["aselmdb"].get("sha256") != hashes["aselmdb"]
    ):
        raise ValueError("NEXT39 raw OMat24 source hash differs")
    prediction_inputs = prediction_manifest.get("inputs_sha256")
    expected_prediction_inputs = {
        "metadata": hashes["metadata"],
        "geometry": hashes["geometry"],
        "cohort_manifest": hashes["cohort_manifest"],
    }
    if not isinstance(prediction_inputs, Mapping) or any(
        prediction_inputs.get(name) != value
        for name, value in expected_prediction_inputs.items()
    ):
        raise ValueError("NEXT39 prediction does not bind the frozen cohort")

    metadata = pd.read_parquet(metadata_path).sort_values(
        "material_id", kind="stable", ignore_index=True
    )
    predictions = pd.read_parquet(predictions_path).sort_values(
        "material_id", kind="stable", ignore_index=True
    )
    required_metadata = {
        "material_id",
        "parent_id",
        "trajectory_stem",
        "initial_sid",
        "latest_sid",
        "latest_record_key",
        "latest_step",
        "natoms",
    }
    required_predictions = {
        "material_id",
        "next23_supported",
        "next23_score",
        "next23_reject",
        "pauling_p2_p5_decision",
    }
    if not required_metadata.issubset(metadata.columns) or not required_predictions.issubset(
        predictions.columns
    ):
        raise ValueError("NEXT39 evaluation table schema differs")
    if (
        metadata.material_id.astype(str).duplicated().any()
        or predictions.material_id.astype(str).duplicated().any()
        or metadata.material_id.astype(str).tolist()
        != predictions.material_id.astype(str).tolist()
    ):
        raise ValueError("NEXT39 evaluation identities differ")
    ids = tuple(metadata.material_id.astype(str))
    loaded, initial = _load_archive_only(frames_zip_path, ids)
    if loaded != list(ids) or any(
        len(atoms) != int(natoms)
        for atoms, natoms in zip(initial, metadata.natoms, strict=True)
    ):
        raise ValueError("NEXT39 initial geometry identity differs")
    return metadata, initial, predictions, prediction_manifest, hashes


def evaluate_next39_trajectories(
    *,
    db_path: Path,
    metadata_path: Path,
    frames_zip_path: Path,
    cohort_manifest_path: Path,
    predictions_path: Path,
    prediction_manifest_path: Path,
    output_dir: Path,
    fingerprint_calculator: FingerprintCalculator | None = None,
) -> dict[str, object]:
    """Evaluate sealed predictions against latest-observed structure changes."""

    database = Path(db_path).resolve()
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    metadata_path = Path(metadata_path).resolve()
    frames_zip_path = Path(frames_zip_path).resolve()
    cohort_manifest_path = Path(cohort_manifest_path).resolve()
    predictions_path = Path(predictions_path).resolve()
    prediction_manifest_path = Path(prediction_manifest_path).resolve()
    metadata, initial_structures, predictions, prediction_manifest, input_hashes = (
        _validate_preopening_inputs(
            database=database,
            metadata_path=metadata_path,
            frames_zip_path=frames_zip_path,
            cohort_manifest_path=cohort_manifest_path,
            predictions_path=predictions_path,
            prediction_manifest_path=prediction_manifest_path,
        )
    )
    calculator = exact_structure_fingerprint if fingerprint_calculator is None else fingerprint_calculator
    production = bool(
        fingerprint_calculator is None
        and prediction_manifest.get("production_protocol_eligible") is True
    )
    fingerprint_environment: dict[str, object]
    if fingerprint_calculator is None:
        fingerprint_environment = _fingerprint_environment()
        if fingerprint_environment["matminer_version"] != "0.10.1":
            raise RuntimeError("NEXT39 production Matminer version differs")
    else:
        fingerprint_environment = {"calculator": "injected_test_double"}

    env = lmdb.open(
        str(database), subdir=False, readonly=True, lock=False, readahead=False
    )
    endpoint_rows: list[dict[str, object]] = []
    failures: Counter[str] = Counter()
    started = time.perf_counter()
    try:
        with env.begin() as transaction:
            for upstream, initial in zip(
                metadata.to_dict("records"), initial_structures, strict=True
            ):
                failure: str | None = None
                change = math.nan
                try:
                    payload = _record_payload(
                        transaction, int(upstream["latest_record_key"])
                    )
                    identity, later = project_identity_geometry(payload)
                    if (
                        identity["sid"] != str(upstream["latest_sid"])
                        or identity["parent_id"] != str(upstream["parent_id"])
                        or identity["task_type"] != TASK_TYPE
                    ):
                        raise ValueError("latest-observed identity differs")
                    change = fingerprint_distance(
                        initial, later, calculator=calculator
                    )
                except Exception as exc:
                    failure = f"{type(exc).__name__}: {exc}"
                    failures[failure] += 1
                endpoint_rows.append(
                    {
                        "material_id": str(upstream["material_id"]),
                        "latest_step": int(upstream["latest_step"]),
                        ENDPOINT_COLUMN: change,
                        "evaluation_supported": failure is None,
                        "evaluation_failure": failure,
                    }
                )
    finally:
        env.close()
    elapsed = time.perf_counter() - started
    endpoints = pd.DataFrame(endpoint_rows).sort_values(
        "material_id", kind="stable", ignore_index=True
    )
    joined = predictions.merge(
        endpoints, on="material_id", how="left", validate="one_to_one"
    ).merge(
        metadata[["material_id", "latest_step", "observed_step_count"]].rename(
            columns={"latest_step": "cohort_latest_step"}
        ),
        on="material_id",
        how="left",
        validate="one_to_one",
    )
    evaluation_mask = joined.evaluation_supported.to_numpy(dtype=bool)
    if not evaluation_mask.any():
        raise ValueError("no NEXT39 trajectory endpoint could be evaluated")
    evaluated = joined.loc[evaluation_mask].reset_index(drop=True)
    endpoint = evaluated[ENDPOINT_COLUMN].to_numpy(dtype=float)
    next23_supported = evaluated.next23_supported.to_numpy(dtype=bool)
    next23_reject = evaluated.next23_reject.to_numpy(dtype=bool)
    pauling_decision = evaluated.pauling_p2_p5_decision.astype(str)
    pauling_supported = pauling_decision.ne("ABSTAIN").to_numpy(dtype=bool)
    pauling_reject = pauling_decision.eq("REJECT").to_numpy(dtype=bool)

    next23_metrics = decision_metrics(
        supported=next23_supported, reject=next23_reject, endpoint=endpoint
    )
    pauling_metrics = decision_metrics(
        supported=pauling_supported, reject=pauling_reject, endpoint=endpoint
    )
    result: dict[str, object] = {
        "protocol": PROTOCOL,
        "endpoint": {
            "column": ENDPOINT_COLUMN,
            "role": "step0_to_latest_observed_sampled_structure_change",
            "latest_frame_is_proven_converged": False,
            "fingerprint": "CrystalNNFingerprint ops plus SiteStats mean/std_dev/minimum/maximum L2",
            "protected_max": PROTECTED_MAX,
            "substantial_min": SUBSTANTIAL_MIN,
            "severe_min": SEVERE_MIN,
        },
        "primary_gates": dict(PRIMARY_GATES),
        "rows": len(joined),
        "evaluation_supported_rows": int(evaluation_mask.sum()),
        "evaluation_failed_rows": int((~evaluation_mask).sum()),
        "evaluation_failure_counts": dict(sorted(failures.items())),
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
            "independent cross-source structural-change confirmation only; "
            "not a DFT energy, hull stability, or convergence claim"
        ),
    }

    repository = Path(__file__).resolve().parents[1]
    source_names = (
        "src/next23_evaluate.py",
        "src/next32_omat24_cohort.py",
        "src/next39_omat24_trajectory_cohort.py",
        "src/next39_next23_predictions.py",
        "src/next39_trajectory_evaluate.py",
    )
    source_hashes = {name: _sha256(repository / name) for name in source_names}
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "later_geometry_opened": True,
        "latest_observed_sampled_geometry_only": True,
        "dft_values_read": False,
        "dft_energy_force_stress_read": False,
        "model_or_proxy_potential_used": False,
        "thresholds_refit": False,
        "cohort_changed_after_opening": False,
        "execution": {"wall_time_seconds": elapsed},
        "counts": {
            "rows": len(joined),
            "evaluation_supported": int(evaluation_mask.sum()),
            "evaluation_failed": int((~evaluation_mask).sum()),
        },
        "inputs_sha256_before_later_geometry_opening": input_hashes,
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
            "aselmdb": database,
            "metadata": metadata_path,
            "geometry": frames_zip_path,
            "cohort_manifest": cohort_manifest_path,
            "predictions": predictions_path,
            "prediction_manifest": prediction_manifest_path,
        }
        if any(
            _sha256(path) != input_hashes[name] for name, path in input_paths.items()
        ):
            raise RuntimeError("NEXT39 evaluation input changed during publication")
        if any(_sha256(repository / name) != digest for name, digest in source_hashes.items()):
            raise RuntimeError("NEXT39 evaluation source changed during publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--frames-zip", required=True, type=Path)
    parser.add_argument("--cohort-manifest", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--prediction-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    evaluate_next39_trajectories(
        db_path=args.db,
        metadata_path=args.metadata,
        frames_zip_path=args.frames_zip,
        cohort_manifest_path=args.cohort_manifest,
        predictions_path=args.predictions,
        prediction_manifest_path=args.prediction_manifest,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "JOINED_NAME",
    "MANIFEST_NAME",
    "PROTOCOL",
    "RESULT_NAME",
    "decision_metrics",
    "evaluate_next39_trajectories",
    "exact_structure_fingerprint",
    "fingerprint_distance",
]
