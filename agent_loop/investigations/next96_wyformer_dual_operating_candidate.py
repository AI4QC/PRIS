"""Freeze one WyFormer analytic score with SAFE and BROAD operating points."""

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
    _term_risk,
    assign_group_folds,
    auc_diagnostics,
    decision_metrics,
)
from src.next93_wyformer_source_lockbox import _sha256_file, _write_json
from src.next93b_wyformer_blind_lockbox import (
    ENDPOINT_NAME,
    MANIFEST_NAME as ENDPOINT_MANIFEST_NAME,
    PARTITIONS,
    PROTOCOL as ENDPOINT_PROTOCOL,
)
from src.next94_wyformer_label_free_features import (
    CATALOGUE_NAME as FEATURE_CATALOGUE_NAME,
    FEATURE_NAMES,
    MANIFEST_NAME as FEATURE_MANIFEST_NAME,
    PROTOCOL as FEATURE_PROTOCOL,
)
from src.next95_wyformer_sparse_law_search import (
    DEFAULT_GATES,
    EVALUATION_NAME as NEXT95_EVALUATION_NAME,
    MANIFEST_NAME as NEXT95_MANIFEST_NAME,
    PROTOCOL as NEXT95_PROTOCOL,
    _endpoint_numeric,
    _operating_pass,
)


PROTOCOL = "2026-08-04-next96-wyformer-dual-operating-candidate-freeze-v1"
MANIFEST_NAME = "MANIFEST.json"
FORMULA_NAME = "NEXT96_FROZEN_DUAL_OPERATING_FORMULA.json"
EVIDENCE_NAME = "NEXT96_DISCOVERY_DUAL_OPERATING_EVIDENCE.json"
PREDICTION_NAMES = {
    role: f"next96_frozen_dual_predictions_{role}.parquet" for role in PARTITIONS
}
SAFE_THRESHOLD = 3.356904710858153
BROAD_THRESHOLD = 0.5035394897502813
BROAD_MIN_PRECISION_LOWER = 0.45
GROUP_FOLDS = 5
EXPECTED_INPUT_SHA256 = {
    "feature_manifest": "fb66f7c5caade419a46b9a3fa6fef1bc5b3afa3eebeb95a4bc53baddabc0f659",
    "feature_catalogue": "2fcec0f8564294ec1267546532c974a6e059e9f48b3b30bf95dc3dd58ca80991",
    "feature_discovery": "c515baec0fccef5bc03c7672f1d4e1aca278f5ed4d7b6f1bf7f66c734e2b87f7",
    "feature_internal_validation": "26d95746e8aa56087150737a62035f5d4c5ce51b1d2e10424ed6cb267ea1983c",
    "feature_internal_replication": "7e52e8ab32b380882082ee9a9315c3d18b4d22fe100a83766060b86e50ff19d9",
    "next95_manifest": "e9bbf1bdf27e905b9bd527cdba68dd4e97b9b3b79cd56fe791659b30ee693532",
    "next95_evaluation": "a2436a5bea9c65dedb55c6fe43bb48b850f219b8a0cadbab5743457a74a5b850",
    "discovery_endpoint_manifest": "3cf3a196ab497851131d5d1604f272d15121c19a943eeb3103a268e7e8b332f5",
    "discovery_endpoint": "f39836e62a1da03ed823479e87d6f75fc0d01da60a8c0a2faa696638cc2fb9d7",
    "design": "4d5c9f4e36d19c6f0a5ede033622640ddfc183f611d8e69c1cde30cf93c2f439",
}


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def apply_dual_formula(
    features: pd.DataFrame,
    formula: Mapping[str, object],
    *,
    safe_threshold: float = SAFE_THRESHOLD,
    broad_threshold: float = BROAD_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply one unchanged score and two inclusive operating thresholds."""

    terms = formula.get("terms")
    if (
        formula.get("kind")
        != "nonnegative_sum_of_at_most_three_one_sided_robust_hinges"
        or formula.get("missing_policy") != "ABSTAIN"
        or not isinstance(terms, list)
        or not 1 <= len(terms) <= 3
        or not math.isfinite(float(safe_threshold))
        or not math.isfinite(float(broad_threshold))
        or float(broad_threshold) >= float(safe_threshold)
    ):
        raise ValueError("NEXT96 frozen formula schema differs")
    score = np.zeros(len(features), dtype=float)
    supported = np.ones(len(features), dtype=bool)
    for term in terms:
        if not isinstance(term, Mapping):
            raise ValueError("NEXT96 term schema differs")
        weight = term.get("weight")
        if not isinstance(weight, (int, float)) or not math.isfinite(float(weight)) or float(weight) <= 0:
            raise ValueError("NEXT96 term weight differs")
        risk, term_supported = _term_risk(features, term)
        score += float(weight) * risk
        supported &= term_supported
    score[~supported] = np.nan
    safe_reject = supported & (score >= float(safe_threshold))
    broad_reject = supported & (score >= float(broad_threshold))
    return score, supported, safe_reject, broad_reject


def pauling_dominance(
    law_metrics: Mapping[str, object], pauling_metrics: Mapping[str, object]
) -> dict[str, object]:
    comparisons = {
        "coverage_lower_strictly_higher": float(law_metrics["coverage_lower"])
        > float(pauling_metrics["coverage_lower"]),
        "protected_kept_not_lower": int(law_metrics["protected_kept"])
        >= int(pauling_metrics["protected_kept"]),
        "severe_rejected_strictly_higher": int(law_metrics["severe_rejected"])
        > int(pauling_metrics["severe_rejected"]),
        "severe_precision_lower_strictly_higher": float(
            law_metrics["severe_rejection_precision_lower"]
        )
        > float(pauling_metrics["severe_rejection_precision_lower"]),
        "savings_lower_strictly_higher": float(law_metrics["savings_lower"])
        > float(pauling_metrics["savings_lower"]),
    }
    return {"comparisons": comparisons, "passes_all": all(comparisons.values())}


def _decision_text(supported: np.ndarray, rejected: np.ndarray) -> np.ndarray:
    return np.where(~supported, "ABSTAIN", np.where(rejected, "REJECT", "KEEP"))


def _prediction_frame(
    *,
    frame: pd.DataFrame,
    score: np.ndarray,
    supported: np.ndarray,
    safe_reject: np.ndarray,
    broad_reject: np.ndarray,
) -> pd.DataFrame:
    """Freeze identities, grouping fields, baseline, score, and decisions together."""

    required = [
        "material_id",
        "reduced_formula",
        "crystal_system",
        "pauling_p2_p5_decision",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"NEXT96 prediction identity columns missing: {missing}")
    return pd.DataFrame(
        {
            "material_id": frame["material_id"].astype(str),
            "reduced_formula": frame["reduced_formula"].astype(str),
            "crystal_system": frame["crystal_system"].astype(str),
            "pauling_p2_p5_decision": frame["pauling_p2_p5_decision"].astype(str),
            "score": score,
            "supported": supported,
            "safe_decision": _decision_text(supported, safe_reject),
            "broad_decision": _decision_text(supported, broad_reject),
        }
    )


def _safe_auc_pass(diagnostics: Mapping[str, object]) -> bool:
    return bool(
        diagnostics["pooled_extreme_auc"] is not None
        and float(diagnostics["pooled_extreme_auc"])
        >= float(DEFAULT_GATES["pooled_extreme_auc"])
        and diagnostics["macro_lattice_auc"] is not None
        and float(diagnostics["macro_lattice_auc"])
        >= float(DEFAULT_GATES["macro_lattice_auc"])
        and diagnostics["worst_lattice_auc"] is not None
        and float(diagnostics["worst_lattice_auc"])
        >= float(DEFAULT_GATES["worst_lattice_auc"])
        and int(diagnostics["evaluable_lattices"])
        >= int(DEFAULT_GATES["evaluable_lattices"])
    )


def _publish_directory(staging: Path, target: Path) -> None:
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    os.replace(staging, target)


def freeze_wyformer_dual_operating_candidate(
    *,
    feature_dir: Path,
    next95_dir: Path,
    discovery_endpoint_dir: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Reproduce discovery evidence, then freeze all-partition predictions."""

    feature_root = Path(feature_dir).resolve()
    next95_root = Path(next95_dir).resolve()
    endpoint_root = Path(discovery_endpoint_dir).resolve()
    paths = {
        "feature_manifest": feature_root / FEATURE_MANIFEST_NAME,
        "feature_catalogue": feature_root / FEATURE_CATALOGUE_NAME,
        **{f"feature_{role}": feature_root / FEATURE_NAMES[role] for role in PARTITIONS},
        "next95_manifest": next95_root / NEXT95_MANIFEST_NAME,
        "next95_evaluation": next95_root / NEXT95_EVALUATION_NAME,
        "discovery_endpoint_manifest": endpoint_root / ENDPOINT_MANIFEST_NAME,
        "discovery_endpoint": endpoint_root / ENDPOINT_NAME,
        "design": Path(design_path).resolve(),
    }
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT96 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("NEXT96 formal input identity differs")

    feature_manifest = _read_json(paths["feature_manifest"])
    next95_manifest = _read_json(paths["next95_manifest"])
    next95_evaluation = _read_json(paths["next95_evaluation"])
    endpoint_manifest = _read_json(paths["discovery_endpoint_manifest"])
    if (
        feature_manifest.get("protocol") != FEATURE_PROTOCOL
        or feature_manifest.get("labels_opened") is not False
        or feature_manifest.get("endpoint_payloads_opened") is not False
        or feature_manifest.get("dft_values_used_by_features") is not False
    ):
        raise ValueError("NEXT94 feature provenance differs")
    if (
        next95_manifest.get("protocol") != NEXT95_PROTOCOL
        or next95_manifest.get("discovery_endpoint_opened") is not True
        or next95_manifest.get("validation_endpoint_opened") is not False
        or next95_manifest.get("replication_endpoint_opened") is not False
        or next95_manifest.get("passes_discovery_gates") is not False
    ):
        raise ValueError("NEXT95 negative single-threshold provenance differs")
    if (
        endpoint_manifest.get("protocol") != ENDPOINT_PROTOCOL
        or endpoint_manifest.get("partition_role") != "discovery"
        or endpoint_manifest.get("endpoint_sha256") != input_hashes["discovery_endpoint"]
    ):
        raise ValueError("NEXT93b discovery endpoint provenance differs")

    formula = next95_evaluation.get("selected_formula")
    if not isinstance(formula, dict) or float(formula.get("threshold", math.nan)) != SAFE_THRESHOLD:
        raise ValueError("NEXT95 selected formula or SAFE threshold differs")
    frozen_score_formula = {
        "kind": formula["kind"],
        "missing_policy": formula["missing_policy"],
        "terms": formula["terms"],
    }
    discovery_features = pd.read_parquet(paths["feature_discovery"])
    discovery_endpoint = pd.read_parquet(paths["discovery_endpoint"])
    discovery = discovery_features.merge(
        discovery_endpoint[["material_id", "endpoint_stratum"]],
        on="material_id",
        how="inner",
        validate="one_to_one",
    )
    if len(discovery) != len(discovery_features) or len(discovery) != len(discovery_endpoint):
        raise ValueError("NEXT96 discovery row accounting differs")
    endpoint = _endpoint_numeric(discovery["endpoint_stratum"])
    score, supported, safe_reject, broad_reject = apply_dual_formula(
        discovery, frozen_score_formula
    )
    safe_metrics = decision_metrics(
        supported=supported, reject=safe_reject, distortion_ratio=endpoint
    )
    broad_metrics = decision_metrics(
        supported=supported, reject=broad_reject, distortion_ratio=endpoint
    )
    pauling = _pauling_baseline(discovery, endpoint)
    diagnostics = auc_diagnostics(
        score=score,
        supported=supported,
        distortion_ratio=endpoint,
        lattice_class=discovery["crystal_system"].astype(str).to_numpy(),
    )
    folds = assign_group_folds(discovery["reduced_formula"].astype(str).to_numpy())
    fold_records: list[dict[str, object]] = []
    safe_all_folds = True
    broad_all_folds = True
    for held_out in range(GROUP_FOLDS):
        mask = folds == held_out
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
        pauling_fold = _pauling_baseline(discovery.loc[mask], endpoint[mask])
        safe_pass = _operating_pass(safe_fold, DEFAULT_GATES)
        dominance = pauling_dominance(broad_fold, pauling_fold)
        safe_all_folds &= safe_pass
        broad_all_folds &= bool(dominance["passes_all"])
        fold_records.append(
            {
                "held_out_fold": held_out,
                "safe_metrics": safe_fold,
                "safe_passes_operating_gates": safe_pass,
                "broad_metrics": broad_fold,
                "pauling_metrics": pauling_fold,
                "broad_pauling_dominance": dominance,
            }
        )
    pooled_dominance = pauling_dominance(broad_metrics, pauling)
    passes = bool(
        _operating_pass(safe_metrics, DEFAULT_GATES)
        and _safe_auc_pass(diagnostics)
        and safe_all_folds
        and float(broad_metrics["severe_rejection_precision_lower"])
        >= BROAD_MIN_PRECISION_LOWER
        and pooled_dominance["passes_all"]
        and broad_all_folds
    )
    if not passes:
        raise RuntimeError("NEXT96 discovery dual-operating protocol did not reproduce")

    evidence = {
        "protocol": PROTOCOL,
        "safe_threshold": SAFE_THRESHOLD,
        "broad_threshold": BROAD_THRESHOLD,
        "safe_metrics": safe_metrics,
        "safe_auc_diagnostics": diagnostics,
        "broad_metrics": broad_metrics,
        "pauling_metrics": pauling,
        "broad_pooled_pauling_dominance": pooled_dominance,
        "folds": fold_records,
        "safe_all_folds_pass": safe_all_folds,
        "broad_all_folds_dominate_pauling": broad_all_folds,
        "passes_discovery_dual_operating_protocol": passes,
    }
    frozen_formula = {
        **frozen_score_formula,
        "safe_threshold": SAFE_THRESHOLD,
        "broad_threshold": BROAD_THRESHOLD,
        "broad_min_severe_precision_lower": BROAD_MIN_PRECISION_LOWER,
        "execution_input": "one raw generated x0 structure",
        "forbidden_execution_inputs": [
            "DFT calculation or value",
            "relaxed structure or trajectory",
            "learned energy force stress proxy or MLIP",
            "same-composition alternative",
        ],
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    source_hash = _sha256_file(source_path)
    output_paths: list[Path] = []
    prediction_counts: dict[str, object] = {}
    try:
        _write_json(staging / FORMULA_NAME, frozen_formula)
        _write_json(staging / EVIDENCE_NAME, evidence)
        output_paths.extend([staging / FORMULA_NAME, staging / EVIDENCE_NAME])
        for role in PARTITIONS:
            frame = pd.read_parquet(paths[f"feature_{role}"])
            local_score, local_supported, local_safe, local_broad = apply_dual_formula(
                frame, frozen_score_formula
            )
            predictions = _prediction_frame(
                frame=frame,
                score=local_score,
                supported=local_supported,
                safe_reject=local_safe,
                broad_reject=local_broad,
            )
            prediction_path = staging / PREDICTION_NAMES[role]
            predictions.to_parquet(prediction_path, index=False)
            output_paths.append(prediction_path)
            prediction_counts[role] = {
                "rows": int(len(predictions)),
                "supported": int(local_supported.sum()),
                "safe_rejected": int(local_safe.sum()),
                "broad_rejected": int(local_broad.sum()),
            }
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "passes_discovery_dual_operating_protocol": True,
            "formula_or_threshold_changed_after_freeze": False,
            "prediction_partitions_frozen": list(PARTITIONS),
            "prediction_counts": prediction_counts,
            "discovery_endpoint_opened": True,
            "validation_endpoint_opened": False,
            "replication_endpoint_opened": False,
            "validation_authorized": True,
            "replication_authorized": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_formula_at_execution": False,
            "learned_energy_force_stress_proxy_used": False,
            "physical_relaxation_executed": False,
            "scientific_improvement_claim": False,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": {
                "src/next96_wyformer_dual_operating_candidate.py": source_hash
            },
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT96 input changed before publication")
        if _sha256_file(source_path) != source_hash:
            raise RuntimeError("NEXT96 source changed before publication")
        _publish_directory(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "BROAD_THRESHOLD",
    "EVIDENCE_NAME",
    "FORMULA_NAME",
    "MANIFEST_NAME",
    "PREDICTION_NAMES",
    "PROTOCOL",
    "SAFE_THRESHOLD",
    "apply_dual_formula",
    "freeze_wyformer_dual_operating_candidate",
    "pauling_dominance",
]
