"""One-shot validation of the frozen WyFormer dual-operating analytic law."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src.next87_scigen_sparse_law_search import (
    _pauling_baseline,
    assign_group_folds,
    auc_diagnostics,
    decision_metrics,
)
from src.next93_wyformer_source_lockbox import _sha256_file, _write_json
from src.next93b_wyformer_blind_lockbox import (
    ENDPOINT_NAME,
    MANIFEST_NAME as ENDPOINT_MANIFEST_NAME,
    PROTOCOL as ENDPOINT_PROTOCOL,
)
from src.next95_wyformer_sparse_law_search import DEFAULT_GATES, _endpoint_numeric, _operating_pass
from src.next96_wyformer_dual_operating_candidate import (
    BROAD_MIN_PRECISION_LOWER,
    BROAD_THRESHOLD,
    FORMULA_NAME,
    MANIFEST_NAME as FROZEN_MANIFEST_NAME,
    PREDICTION_NAMES,
    PROTOCOL as FROZEN_PROTOCOL,
    SAFE_THRESHOLD,
    _safe_auc_pass,
    pauling_dominance,
)


PROTOCOL = "2026-08-04-next97-wyformer-dual-operating-one-shot-validation-v1"
MANIFEST_NAME = "MANIFEST.json"
EVALUATION_NAME = "NEXT97_WYFORMER_DUAL_OPERATING_VALIDATION.json"
GROUP_FOLDS = 5
EXPECTED_INPUT_SHA256 = {
    "frozen_manifest": "fbf5d0a4cc0980fb33edf9ea951e9169b8717eab7c99dcdb6bcd9679af582625",
    "frozen_formula": "047b27e3207e6eb3bf577cb25bf99354e565028824d568b0c89257534041479c",
    "validation_predictions": "15337bc321e719461eb2a60142bf1520a2dab82683d3eeb7f85d896b39744560",
    "validation_endpoint_manifest": "d5b87e8e2902fb14112e20cf60fb0c59593c7586655f19e1b248131f8df2cd7f",
    "validation_endpoint": "514d8ba4ac9e335f9ffced15f021b76e4573ad263fe26a438aaeaece6ad128f5",
}


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _expected_decision(
    supported: np.ndarray, score: np.ndarray, threshold: float
) -> np.ndarray:
    rejected = supported & (score >= float(threshold))
    return np.where(~supported, "ABSTAIN", np.where(rejected, "REJECT", "KEEP"))


def evaluate_dual_partition(
    *,
    predictions: pd.DataFrame,
    endpoints: pd.DataFrame,
    gates: Mapping[str, float] = DEFAULT_GATES,
    broad_min_precision_lower: float = BROAD_MIN_PRECISION_LOWER,
) -> dict[str, object]:
    """Evaluate immutable SAFE/BROAD decisions without refitting or calibration."""

    prediction_required = {
        "material_id",
        "reduced_formula",
        "crystal_system",
        "pauling_p2_p5_decision",
        "score",
        "supported",
        "safe_decision",
        "broad_decision",
    }
    endpoint_required = {"material_id", "endpoint_stratum"}
    if (
        prediction_required - set(predictions.columns)
        or endpoint_required - set(endpoints.columns)
        or predictions["material_id"].astype(str).duplicated().any()
        or endpoints["material_id"].astype(str).duplicated().any()
    ):
        raise ValueError("NEXT97 partition identities differ")
    joined = predictions.merge(
        endpoints.loc[:, ["material_id", "endpoint_stratum"]],
        on="material_id",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(predictions) or len(joined) != len(endpoints):
        raise ValueError("NEXT97 partition identity join differs")

    score = pd.to_numeric(joined["score"], errors="coerce").to_numpy(float)
    supported = joined["supported"].to_numpy(bool)
    if not np.array_equal(supported, np.isfinite(score)):
        raise ValueError("NEXT97 frozen support mask differs")
    expected_safe = _expected_decision(supported, score, SAFE_THRESHOLD)
    expected_broad = _expected_decision(supported, score, BROAD_THRESHOLD)
    if (
        not np.array_equal(
            joined["safe_decision"].astype(str).to_numpy(), expected_safe
        )
        or not np.array_equal(
            joined["broad_decision"].astype(str).to_numpy(), expected_broad
        )
    ):
        raise ValueError("NEXT97 frozen dual decisions differ")

    endpoint = _endpoint_numeric(joined["endpoint_stratum"])
    safe_reject = expected_safe == "REJECT"
    broad_reject = expected_broad == "REJECT"
    safe_metrics = decision_metrics(
        supported=supported, reject=safe_reject, distortion_ratio=endpoint
    )
    broad_metrics = decision_metrics(
        supported=supported, reject=broad_reject, distortion_ratio=endpoint
    )
    pauling_metrics = _pauling_baseline(joined, endpoint)
    diagnostics = auc_diagnostics(
        score=score,
        supported=supported,
        distortion_ratio=endpoint,
        lattice_class=joined["crystal_system"].astype(str).to_numpy(),
    )
    folds = assign_group_folds(joined["reduced_formula"].astype(str).to_numpy())
    fold_records: list[dict[str, object]] = []
    safe_all_folds = True
    broad_all_folds = True
    for held_out in range(GROUP_FOLDS):
        mask = folds == held_out
        if not mask.any():
            raise ValueError("NEXT97 reduced-formula fold is empty")
        safe_fold = decision_metrics(
            supported=supported[mask],
            reject=safe_reject[mask],
            distortion_ratio=endpoint[mask],
        )
        broad_fold = decision_metrics(
            supported=supported[mask],
            reject=broad_reject[mask],
            distortion_ratio=endpoint[mask],
        )
        pauling_fold = _pauling_baseline(joined.loc[mask], endpoint[mask])
        safe_pass = _operating_pass(safe_fold, gates)
        dominance = pauling_dominance(broad_fold, pauling_fold)
        safe_all_folds &= bool(safe_pass)
        broad_all_folds &= bool(dominance["passes_all"])
        fold_records.append(
            {
                "held_out_fold": held_out,
                "rows": int(mask.sum()),
                "safe_metrics": safe_fold,
                "safe_passes_operating_gates": bool(safe_pass),
                "broad_metrics": broad_fold,
                "pauling_metrics": pauling_fold,
                "broad_pauling_dominance": dominance,
            }
        )

    safe_operating_pass = bool(_operating_pass(safe_metrics, gates))
    safe_auc_pass = bool(_safe_auc_pass_with_gates(diagnostics, gates))
    broad_precision_pass = bool(
        float(broad_metrics["severe_rejection_precision_lower"])
        >= float(broad_min_precision_lower)
    )
    pooled_dominance = pauling_dominance(broad_metrics, pauling_metrics)
    passes = bool(
        safe_operating_pass
        and safe_auc_pass
        and safe_all_folds
        and broad_precision_pass
        and pooled_dominance["passes_all"]
        and broad_all_folds
    )
    return {
        "partition_role": "internal_validation",
        "safe_threshold": SAFE_THRESHOLD,
        "broad_threshold": BROAD_THRESHOLD,
        "safe_metrics": safe_metrics,
        "safe_auc_diagnostics": diagnostics,
        "safe_passes_operating_gates": safe_operating_pass,
        "safe_passes_auc_gates": safe_auc_pass,
        "safe_all_folds_pass": bool(safe_all_folds),
        "broad_metrics": broad_metrics,
        "broad_min_severe_precision_lower": float(broad_min_precision_lower),
        "broad_passes_precision_gate": broad_precision_pass,
        "pauling_metrics": pauling_metrics,
        "broad_pooled_pauling_dominance": pooled_dominance,
        "broad_all_folds_dominate_pauling": bool(broad_all_folds),
        "folds": fold_records,
        "passes_all_validation_gates": passes,
    }


def _safe_auc_pass_with_gates(
    diagnostics: Mapping[str, object], gates: Mapping[str, float]
) -> bool:
    if gates is DEFAULT_GATES:
        return _safe_auc_pass(diagnostics)
    return bool(
        diagnostics["pooled_extreme_auc"] is not None
        and float(diagnostics["pooled_extreme_auc"])
        >= float(gates["pooled_extreme_auc"])
        and diagnostics["macro_lattice_auc"] is not None
        and float(diagnostics["macro_lattice_auc"])
        >= float(gates["macro_lattice_auc"])
        and diagnostics["worst_lattice_auc"] is not None
        and float(diagnostics["worst_lattice_auc"])
        >= float(gates["worst_lattice_auc"])
        and int(diagnostics["evaluable_lattices"])
        >= int(gates["evaluable_lattices"])
    )


def _publish_directory(staging: Path, target: Path) -> None:
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    os.replace(staging, target)


def run_wyformer_one_shot_validation(
    *,
    frozen_dir: Path,
    validation_endpoint_dir: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Open the validation endpoint once and evaluate frozen predictions only."""

    frozen_root = Path(frozen_dir).resolve()
    endpoint_root = Path(validation_endpoint_dir).resolve()
    paths = {
        "frozen_manifest": frozen_root / FROZEN_MANIFEST_NAME,
        "frozen_formula": frozen_root / FORMULA_NAME,
        "validation_predictions": frozen_root
        / PREDICTION_NAMES["internal_validation"],
        "validation_endpoint_manifest": endpoint_root / ENDPOINT_MANIFEST_NAME,
        "validation_endpoint": endpoint_root / ENDPOINT_NAME,
    }
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT97 input is missing")
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("NEXT97 formal input identity differs")

    frozen_manifest = _read_json(paths["frozen_manifest"])
    formula = _read_json(paths["frozen_formula"])
    endpoint_manifest = _read_json(paths["validation_endpoint_manifest"])
    frozen_outputs = frozen_manifest.get("outputs_sha256")
    if (
        frozen_manifest.get("protocol") != FROZEN_PROTOCOL
        or frozen_manifest.get("passes_discovery_dual_operating_protocol") is not True
        or frozen_manifest.get("formula_or_threshold_changed_after_freeze") is not False
        or frozen_manifest.get("validation_endpoint_opened") is not False
        or frozen_manifest.get("replication_endpoint_opened") is not False
        or frozen_manifest.get("validation_authorized") is not True
        or not isinstance(frozen_outputs, Mapping)
        or frozen_outputs.get(FORMULA_NAME) != hashes["frozen_formula"]
        or frozen_outputs.get(PREDICTION_NAMES["internal_validation"])
        != hashes["validation_predictions"]
        or formula.get("kind")
        != "nonnegative_sum_of_at_most_three_one_sided_robust_hinges"
        or formula.get("missing_policy") != "ABSTAIN"
        or float(formula.get("safe_threshold", math.nan)) != SAFE_THRESHOLD
        or float(formula.get("broad_threshold", math.nan)) != BROAD_THRESHOLD
        or float(formula.get("broad_min_severe_precision_lower", math.nan))
        != BROAD_MIN_PRECISION_LOWER
    ):
        raise ValueError("NEXT97 frozen candidate provenance differs")
    if (
        endpoint_manifest.get("protocol") != ENDPOINT_PROTOCOL
        or endpoint_manifest.get("partition_role") != "internal_validation"
        or endpoint_manifest.get("endpoint_payload_opened") is not False
        or endpoint_manifest.get("formula_or_threshold_fitted") is not False
        or endpoint_manifest.get("endpoint_sha256") != hashes["validation_endpoint"]
    ):
        raise ValueError("NEXT97 validation endpoint provenance differs")

    predictions = pd.read_parquet(paths["validation_predictions"])
    endpoints = pd.read_parquet(paths["validation_endpoint"])
    result = evaluate_dual_partition(predictions=predictions, endpoints=endpoints)

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    source_hash = _sha256_file(source_path)
    try:
        evaluation = {
            "protocol": PROTOCOL,
            "candidate_name": "Contact-Valence-Rigidity Risk (CVR-Risk)",
            "evaluation_mode": "one_shot_internal_validation_no_refit",
            **result,
            "formula_sha256": hashes["frozen_formula"],
            "prediction_sha256": hashes["validation_predictions"],
            "replication_authorized": bool(result["passes_all_validation_gates"]),
        }
        evaluation_path = staging / EVALUATION_NAME
        _write_json(evaluation_path, evaluation)
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "passes_validation_gates": bool(result["passes_all_validation_gates"]),
            "validation_endpoint_opened": True,
            "replication_endpoint_opened": False,
            "formula_or_threshold_changed": False,
            "prediction_changed_after_endpoint_open": False,
            "replication_authorized": bool(result["passes_all_validation_gates"]),
            "dft_calculation_executed": False,
            "dft_values_used_by_executable_formula": False,
            "learned_energy_force_stress_proxy_used": False,
            "physical_relaxation_executed": False,
            "internal_validation_improvement_claim": bool(
                result["passes_all_validation_gates"]
            ),
            "scientific_improvement_claim": False,
            "universal_or_dft_equivalence_claim": False,
            "inputs_sha256": hashes,
            "executed_source_sha256": {
                "src/next97_wyformer_one_shot_validation.py": source_hash
            },
            "outputs_sha256": {EVALUATION_NAME: _sha256_file(evaluation_path)},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT97 input changed before publication")
        if _sha256_file(source_path) != source_hash:
            raise RuntimeError("NEXT97 source changed before publication")
        _publish_directory(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "BROAD_THRESHOLD",
    "EVALUATION_NAME",
    "MANIFEST_NAME",
    "PROTOCOL",
    "SAFE_THRESHOLD",
    "evaluate_dual_partition",
    "run_wyformer_one_shot_validation",
]
