"""NEXT92 one-shot validation of the frozen SCIGEN RLI candidate."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from src.next87_scigen_sparse_law_search import (
    DEFAULT_GATES,
    _operating_pass,
    _pauling_baseline,
    _publish_directory_no_replace,
    _read_json,
    _sha256_file,
    auc_diagnostics,
    decision_metrics,
)
from src.next86_scigen_endpoint_router import (
    ENDPOINT_NAME,
    MANIFEST_NAME as ENDPOINT_MANIFEST_NAME,
    PROTOCOL as ENDPOINT_PROTOCOL,
)
from src.next91_scigen_fixed_rli_candidate import (
    FORMULA_NAME,
    MANIFEST_NAME as FROZEN_MANIFEST_NAME,
    PREDICTION_NAMES,
    PROTOCOL as FROZEN_PROTOCOL,
    RLI_THRESHOLD,
)


PROTOCOL = "2026-08-03-next92-scigen-rli-one-shot-validation-v1"
MANIFEST_NAME = "MANIFEST.json"
EVALUATION_NAME = "NEXT92_RLI_INTERNAL_VALIDATION.json"
EXPECTED_INPUT_SHA256 = {
    "frozen_manifest": "abcbacc6e1668b732bda8f6a8cbe796f47e33c4807d398e4941b6d022be16838",
    "frozen_formula": "71db9be3bd2cc77b4ba4d227f2a6ac7e11f1a933a907fb3ddb9ede9ad2edaede",
    "validation_predictions": "4edad7a016354506822fc27205edc176426c2289c0281dc355885e2b33b165d0",
    "validation_endpoint_manifest": "38b491ca3f1cc1143f2188c77de3124746ac557e7d38aac849a24dc47c2b399d",
    "validation_endpoint": "22dd427bd63f0769c0bdb6ee786acfcaeb1b384e30a17fc0882bf0db40477807",
    "validation_features": "f266e6143bc23d9e131b5ec788676b520db928aa46a57a1fcba6fd8530a80c8a",
}


def evaluate_rli_partition(
    *,
    predictions: pd.DataFrame,
    endpoints: pd.DataFrame,
    pauling: pd.DataFrame,
    expected_role: str,
    gates: Mapping[str, float] = DEFAULT_GATES,
) -> dict[str, object]:
    """Evaluate one frozen RLI partition without fitting or calibration."""

    prediction_required = {
        "material_id",
        "partition_role",
        "rli_score",
        "rli_supported",
        "rli_reject",
        "rli_decision",
        "formula_sha256",
    }
    endpoint_required = {
        "material_id",
        "lattice_class",
        "partition_role",
        "distortion_ratio",
    }
    if (
        prediction_required - set(predictions.columns)
        or endpoint_required - set(endpoints.columns)
        or {"material_id", "pauling_p2_p5_decision"} - set(pauling.columns)
        or predictions["material_id"].astype(str).duplicated().any()
        or endpoints["material_id"].astype(str).duplicated().any()
        or pauling["material_id"].astype(str).duplicated().any()
        or set(predictions["partition_role"].astype(str)) != {expected_role}
        or set(endpoints["partition_role"].astype(str)) != {expected_role}
        or predictions["formula_sha256"].astype(str).nunique() != 1
    ):
        raise ValueError("NEXT92 partition identities differ")
    joined = predictions.merge(
        endpoints.loc[:, ["material_id", "lattice_class", "distortion_ratio"]],
        on="material_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        pauling.loc[:, ["material_id", "pauling_p2_p5_decision"]],
        on="material_id",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(predictions) or len(joined) != len(endpoints) or len(joined) != len(pauling):
        raise ValueError("NEXT92 partition identity join differs")
    score = pd.to_numeric(joined["rli_score"], errors="coerce").to_numpy(float)
    supported = joined["rli_supported"].to_numpy(bool)
    reject = joined["rli_reject"].to_numpy(bool)
    expected_reject = supported & np.isfinite(score) & (score >= RLI_THRESHOLD)
    expected_decision = np.where(expected_reject, "REJECT", "KEEP")
    if (
        not np.array_equal(reject, expected_reject)
        or not np.array_equal(joined["rli_decision"].astype(str).to_numpy(), expected_decision)
        or np.isfinite(score[~supported]).any()
    ):
        raise ValueError("NEXT92 frozen RLI decisions differ")
    endpoint = pd.to_numeric(joined["distortion_ratio"], errors="coerce").to_numpy(float)
    metrics = decision_metrics(
        supported=supported, reject=reject, distortion_ratio=endpoint
    )
    diagnostics = auc_diagnostics(
        score=score,
        supported=supported,
        distortion_ratio=endpoint,
        lattice_class=joined["lattice_class"].astype(str).to_numpy(),
    )
    pauling_metrics = _pauling_baseline(joined, endpoint)
    beats_pauling = (
        int(metrics["severe_rejected"]) > int(pauling_metrics["severe_rejected"])
        and float(metrics["severe_rejection_precision_lower"])
        > float(pauling_metrics["severe_rejection_precision_lower"])
    )
    auc_pass = (
        diagnostics["pooled_extreme_auc"] is not None
        and float(diagnostics["pooled_extreme_auc"]) >= float(gates["pooled_extreme_auc"])
        and diagnostics["macro_lattice_auc"] is not None
        and float(diagnostics["macro_lattice_auc"]) >= float(gates["macro_lattice_auc"])
        and diagnostics["worst_lattice_auc"] is not None
        and float(diagnostics["worst_lattice_auc"]) >= float(gates["worst_lattice_auc"])
        and int(diagnostics["evaluable_lattices"]) >= int(gates["evaluable_lattices"])
    )
    merged_metrics = {
        **metrics,
        **{
            key: diagnostics[key]
            for key in (
                "pooled_extreme_auc",
                "macro_lattice_auc",
                "worst_lattice_auc",
                "evaluable_lattices",
            )
        },
    }
    return {
        "partition_role": expected_role,
        "metrics": merged_metrics,
        "pauling_baseline": pauling_metrics,
        "beats_pauling": beats_pauling,
        "lattice_diagnostics": diagnostics["lattices"],
        "passes_all_gates": bool(
            _operating_pass(metrics, gates) and auc_pass and beats_pauling
        ),
    }


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def run_one_shot_rli_validation(
    *,
    frozen_dir: Path,
    validation_endpoint_dir: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Open only internal validation and evaluate the immutable RLI predictions."""

    frozen_root = Path(frozen_dir).resolve()
    endpoint_root = Path(validation_endpoint_dir).resolve()
    target = Path(output_dir).resolve()
    initial_paths = {
        "frozen_manifest": frozen_root / FROZEN_MANIFEST_NAME,
        "frozen_formula": frozen_root / FORMULA_NAME,
        "validation_predictions": frozen_root / PREDICTION_NAMES["internal_validation"],
        "validation_endpoint_manifest": endpoint_root / ENDPOINT_MANIFEST_NAME,
        "validation_endpoint": endpoint_root / ENDPOINT_NAME,
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in initial_paths.values()):
        raise FileNotFoundError("NEXT92 input is missing")
    frozen_manifest = _read_json(initial_paths["frozen_manifest"], role="NEXT91 manifest")
    frozen_inputs = frozen_manifest.get("inputs_sha256")
    feature_entry = (
        frozen_inputs.get("features_internal_validation")
        if isinstance(frozen_inputs, Mapping)
        else None
    )
    if (
        not isinstance(feature_entry, Mapping)
        or not isinstance(feature_entry.get("path"), str)
        or not isinstance(feature_entry.get("sha256"), str)
    ):
        raise ValueError("NEXT91 validation feature receipt differs")
    paths = {
        **initial_paths,
        "validation_features": Path(str(feature_entry["path"])).resolve(),
    }
    if not paths["validation_features"].is_file():
        raise FileNotFoundError("NEXT92 validation feature receipt path is missing")
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("NEXT92 formal input identity differs")

    frozen_outputs = frozen_manifest.get("outputs_sha256")
    formula = _read_json(paths["frozen_formula"], role="NEXT91 formula")
    if (
        frozen_manifest.get("protocol") != FROZEN_PROTOCOL
        or frozen_manifest.get("endpoint_payloads_opened") is not False
        or frozen_manifest.get("validation_endpoint_opened") is not False
        or frozen_manifest.get("replication_endpoint_opened") is not False
        or not isinstance(frozen_outputs, Mapping)
        or frozen_outputs.get(FORMULA_NAME) != hashes["frozen_formula"]
        or frozen_outputs.get(PREDICTION_NAMES["internal_validation"])
        != hashes["validation_predictions"]
        or feature_entry.get("sha256") != hashes["validation_features"]
        or formula.get("protocol") != FROZEN_PROTOCOL
        or formula.get("validation_endpoint_opened") is not False
        or formula.get("replication_endpoint_opened") is not False
        or not np.isclose(float(formula.get("threshold", float("nan"))), RLI_THRESHOLD)
    ):
        raise ValueError("NEXT91 frozen RLI provenance differs")
    endpoint_manifest = _read_json(
        paths["validation_endpoint_manifest"], role="NEXT86 validation endpoint manifest"
    )
    endpoint_outputs = endpoint_manifest.get("outputs_sha256")
    if (
        endpoint_manifest.get("protocol") != ENDPOINT_PROTOCOL
        or endpoint_manifest.get("partition_role") != "internal_validation"
        or endpoint_manifest.get("lockbox_state") != "unopened_for_model_development"
        or not isinstance(endpoint_outputs, Mapping)
        or endpoint_outputs.get(ENDPOINT_NAME) != hashes["validation_endpoint"]
    ):
        raise ValueError("NEXT86 validation endpoint provenance differs")

    predictions = pd.read_parquet(paths["validation_predictions"])
    endpoints = pd.read_parquet(paths["validation_endpoint"])
    features = pd.read_parquet(
        paths["validation_features"], columns=["material_id", "pauling_p2_p5_decision"]
    )
    result = evaluate_rli_partition(
        predictions=predictions,
        endpoints=endpoints,
        pauling=features,
        expected_role="internal_validation",
    )

    source_path = Path(__file__).resolve()
    source_hash = _sha256_file(source_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        evaluation = {
            "protocol": PROTOCOL,
            "candidate_name": "Rigidity-Load Incompatibility (RLI)",
            "evaluation_mode": "one_shot_internal_validation_no_refit",
            **result,
            "formula_sha256": hashes["frozen_formula"],
            "replication_authorized": bool(result["passes_all_gates"]),
        }
        evaluation_path = staging / EVALUATION_NAME
        evaluation_path.write_bytes(_json_bytes(evaluation))
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "mode": "one_shot_internal_validation_of_frozen_rli",
            "passes_validation_gates": result["passes_all_gates"],
            "validation_endpoint_opened": True,
            "replication_endpoint_opened": False,
            "formula_or_threshold_changed": False,
            "relaxed_structures_opened": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_executable_formula": False,
            "inputs_sha256": {
                name: {"path": str(path), "sha256": hashes[name]}
                for name, path in paths.items()
            },
            "executed_source_sha256": {
                "src/next92_scigen_one_shot_validation.py": source_hash
            },
            "outputs_sha256": {EVALUATION_NAME: _sha256_file(evaluation_path)},
            "replication_authorized": bool(result["passes_all_gates"]),
            "scientific_improvement_claim": bool(result["passes_all_gates"]),
            "universal_or_dft_equivalence_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if any(_sha256_file(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT92 input changed before publication")
        if _sha256_file(source_path) != source_hash:
            raise RuntimeError("NEXT92 source changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


__all__ = [
    "EVALUATION_NAME",
    "MANIFEST_NAME",
    "PROTOCOL",
    "evaluate_rli_partition",
    "run_one_shot_rli_validation",
]
