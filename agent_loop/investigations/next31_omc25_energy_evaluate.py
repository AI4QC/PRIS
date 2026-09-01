"""Frozen prospective evaluation for the NEXT31 OMC25 energy-response rule."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


PROTOCOL = "2026-08-03-next31-omc25-energy-prospective-evaluation-v1"
ENERGY_POSITIVE_MIN = 0.04
PROTECTED_MAX = 0.01
ONE_SIDED_95_Z = 1.6448536269514722
GATES = {
    "coverage_lower_at_least": 0.95,
    "protected_recall_lower_at_least": 0.95,
    "reject_precision_lower_at_least": 0.70,
    "dft_savings_lower_at_least": 0.02,
    "auc_energy_positive_at_least": 0.85,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    def safe(item: object) -> object:
        if isinstance(item, dict):
            return {str(key): safe(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(value) for value in item]
        if isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
            return None
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.bool_):
            return bool(item)
        return item

    return (
        json.dumps(safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def freeze_evaluation_protocol(
    *, frozen_rule_path: Path, output_dir: Path
) -> dict[str, object]:
    rule_path = Path(frozen_rule_path).resolve()
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if not rule_path.is_file():
        raise FileNotFoundError(str(rule_path))
    protocol = {
        "protocol": PROTOCOL,
        "labels_opened": False,
        "endpoint_artifacts_opened": False,
        "energy_positive_min_ev_per_atom": ENERGY_POSITIVE_MIN,
        "protected_max_ev_per_atom": PROTECTED_MAX,
        "gates": GATES,
        "frozen_rule": {"path": str(rule_path), "sha256": _sha256(rule_path)},
        "physical_never_read_lockbox": False,
        "claim_if_passed": (
            "DFT-free-at-execution ranking and conservative pre-screening of "
            "large OMC25 DFT relaxation-energy response on the prospective cohort"
        ),
        "claim_excluded_even_if_passed": (
            "thermodynamic stability, convex-hull stability, or replacement of DFT"
        ),
    }
    manifest = {
        "protocol": PROTOCOL,
        "mode": "pre_label_opening_evaluation_protocol_freeze",
        "labels_opened": False,
        "outputs_sha256": {},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        protocol_path = staging / "NEXT31_EVALUATION_PROTOCOL.json"
        protocol_path.write_bytes(_json_bytes(protocol))
        manifest["outputs_sha256"] = {
            protocol_path.name: _sha256(protocol_path)
        }
        (staging / "MANIFEST.json").write_bytes(_json_bytes(manifest))
        if _sha256(rule_path) != protocol["frozen_rule"]["sha256"]:  # type: ignore[index]
            raise RuntimeError("NEXT31 rule changed before protocol publication")
        os.rename(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _wilson_lower(successes: int, total: int) -> float:
    if total <= 0:
        return math.nan
    p = successes / total
    z = ONE_SIDED_95_Z
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - radius)


def _metric(successes: int, total: int) -> dict[str, float | int]:
    return {
        "numerator": int(successes),
        "denominator": int(total),
        "estimate": float(successes / total) if total else math.nan,
        "wilson_lower_onesided_95": _wilson_lower(successes, total),
    }


def energy_metrics(
    *,
    energy: np.ndarray,
    supported: np.ndarray,
    reject: np.ndarray,
    score: np.ndarray,
) -> dict[str, object]:
    energy = np.asarray(energy, dtype=float)
    supported = np.asarray(supported, dtype=bool)
    reject = np.asarray(reject, dtype=bool)
    score = np.asarray(score, dtype=float)
    if not (len(energy) == len(supported) == len(reject) == len(score)):
        raise ValueError("NEXT31 evaluation arrays have different lengths")
    if not np.isfinite(energy).all() or np.any(reject & ~supported):
        raise ValueError("NEXT31 endpoint or fail-open contract differs")
    protected = energy <= PROTECTED_MAX
    positive = energy >= ENERGY_POSITIVE_MIN
    diagnostic = supported & np.isfinite(score)
    auc = math.nan
    rho = math.nan
    if diagnostic.any() and len(np.unique(positive[diagnostic])) == 2:
        auc = float(roc_auc_score(positive[diagnostic], score[diagnostic]))
    if diagnostic.sum() >= 2:
        rho = float(spearmanr(score[diagnostic], energy[diagnostic]).statistic)
    metrics: dict[str, object] = {
        "counts": {
            "rows": len(energy),
            "supported": int(supported.sum()),
            "rejected": int(reject.sum()),
            "protected": int(protected.sum()),
            "energy_positive": int(positive.sum()),
            "rejected_positive": int((reject & positive).sum()),
            "rejected_protected": int((reject & protected).sum()),
        },
        "coverage": _metric(int(supported.sum()), len(energy)),
        "protected_recall": _metric(
            int((protected & ~reject).sum()), int(protected.sum())
        ),
        "reject_precision": _metric(
            int((reject & positive).sum()), int(reject.sum())
        ),
        "energy_positive_recall": _metric(
            int((reject & positive).sum()), int(positive.sum())
        ),
        "dft_savings": _metric(int(reject.sum()), len(energy)),
        "auc_energy_positive": auc,
        "spearman_energy_drop": rho,
    }
    clauses = {
        "coverage_lower_at_least": float(
            metrics["coverage"]["wilson_lower_onesided_95"]  # type: ignore[index]
        )
        >= GATES["coverage_lower_at_least"],
        "protected_recall_lower_at_least": float(
            metrics["protected_recall"]["wilson_lower_onesided_95"]  # type: ignore[index]
        )
        >= GATES["protected_recall_lower_at_least"],
        "reject_precision_lower_at_least": float(
            metrics["reject_precision"]["wilson_lower_onesided_95"]  # type: ignore[index]
        )
        >= GATES["reject_precision_lower_at_least"],
        "dft_savings_lower_at_least": float(
            metrics["dft_savings"]["wilson_lower_onesided_95"]  # type: ignore[index]
        )
        >= GATES["dft_savings_lower_at_least"],
        "auc_energy_positive_at_least": math.isfinite(auc)
        and auc >= GATES["auc_energy_positive_at_least"],
    }
    metrics["clauses"] = clauses
    metrics["prospective_gate_pass"] = bool(all(clauses.values()))
    return metrics


def evaluate_frozen_predictions(
    *,
    predictions_path: Path,
    prediction_manifest_path: Path,
    endpoints_path: Path,
    evaluation_protocol_path: Path,
    output_dir: Path,
    require_prediction_output_hash: bool = True,
) -> dict[str, object]:
    """Open identity-locked endpoints only after validating frozen predictions."""

    paths = {
        "predictions": Path(predictions_path).resolve(),
        "prediction_manifest": Path(prediction_manifest_path).resolve(),
        "endpoints": Path(endpoints_path).resolve(),
        "evaluation_protocol": Path(evaluation_protocol_path).resolve(),
    }
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT31 evaluation input is missing")
    prediction_manifest = json.loads(paths["prediction_manifest"].read_text("utf-8"))
    if (
        prediction_manifest.get("labels_opened") is not False
        or prediction_manifest.get("endpoint_artifacts_opened") is not False
    ):
        raise ValueError("NEXT31 predictions were not sealed before endpoint opening")
    if require_prediction_output_hash:
        outputs = prediction_manifest.get("outputs_sha256")
        if not isinstance(outputs, dict) or outputs.get(paths["predictions"].name) != _sha256(
            paths["predictions"]
        ):
            raise ValueError("NEXT31 prediction output hash differs")
    protocol = json.loads(paths["evaluation_protocol"].read_text("utf-8"))
    if (
        protocol.get("protocol") != PROTOCOL
        or protocol.get("labels_opened") is not False
        or protocol.get("gates") != GATES
        or protocol.get("energy_positive_min_ev_per_atom") != ENERGY_POSITIVE_MIN
        or protocol.get("protected_max_ev_per_atom") != PROTECTED_MAX
    ):
        raise ValueError("NEXT31 frozen evaluation protocol differs")
    protocol_rule = protocol.get("frozen_rule")
    prediction_inputs = prediction_manifest.get("inputs_sha256")
    prediction_rule = (
        prediction_inputs.get("frozen_rule")
        if isinstance(prediction_inputs, dict)
        else None
    )
    if (
        not isinstance(protocol_rule, dict)
        or not isinstance(prediction_rule, dict)
        or prediction_rule.get("sha256") != protocol_rule.get("sha256")
    ):
        raise ValueError("NEXT31 prediction/protocol rule binding differs")
    preopening_hashes = {role: _sha256(path) for role, path in paths.items()}
    predictions = pd.read_parquet(paths["predictions"])
    required_predictions = {
        "material_id",
        "source_shard",
        "analytic_supported",
        "next31_risk_score",
        "reject",
    }
    if not required_predictions.issubset(predictions.columns):
        raise ValueError("NEXT31 predictions lack required fields")
    predictions["material_id"] = predictions["material_id"].astype(str)
    if predictions["material_id"].duplicated().any():
        raise ValueError("NEXT31 prediction IDs are not unique")
    # Endpoint values are first parsed only after all checks above and hashes are sealed.
    endpoints = pd.read_parquet(
        paths["endpoints"],
        columns=["material_id", "source_shard", "energy_drop_pa"],
    )
    endpoints["material_id"] = endpoints["material_id"].astype(str)
    if (
        endpoints["material_id"].duplicated().any()
        or set(endpoints["material_id"]) != set(predictions["material_id"])
    ):
        raise ValueError("NEXT31 endpoint identities differ from frozen predictions")
    joined = predictions.merge(
        endpoints,
        on=["material_id", "source_shard"],
        validate="one_to_one",
    ).sort_values("material_id", kind="stable", ignore_index=True)
    if len(joined) != len(predictions):
        raise ValueError("NEXT31 endpoint shard identity differs")
    metrics = energy_metrics(
        energy=joined["energy_drop_pa"].to_numpy(float),
        supported=joined["analytic_supported"].to_numpy(bool),
        reject=joined["reject"].to_numpy(bool),
        score=joined["next31_risk_score"].to_numpy(float),
    )
    per_shard = {
        str(shard): energy_metrics(
            energy=part["energy_drop_pa"].to_numpy(float),
            supported=part["analytic_supported"].to_numpy(bool),
            reject=part["reject"].to_numpy(bool),
            score=part["next31_risk_score"].to_numpy(float),
        )
        for shard, part in joined.groupby("source_shard", sort=True)
    }
    result: dict[str, object] = {
        "protocol": PROTOCOL,
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "labels_opened_after_predictions_frozen": True,
        "thresholds_refit": False,
        "physical_never_read_lockbox": False,
        "metrics": metrics,
        "per_shard": per_shard,
        "inputs_sha256_before_opening": preopening_hashes,
        "claim_boundary": (
            "DFT relaxation-energy response only; not convex-hull, formation-energy, "
            "or thermodynamic stability"
        ),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        result_path = staging / "NEXT31_ENERGY_EVALUATION.json"
        joined_path = staging / "next31_joined.parquet"
        result_path.write_bytes(_json_bytes(result))
        joined.to_parquet(joined_path, index=False)
        manifest = {
            "protocol": PROTOCOL,
            "labels_opened": True,
            "outputs_sha256": {
                result_path.name: _sha256(result_path),
                joined_path.name: _sha256(joined_path),
            },
        }
        (staging / "MANIFEST.json").write_bytes(_json_bytes(manifest))
        if any(_sha256(path) != preopening_hashes[role] for role, path in paths.items()):
            raise RuntimeError("NEXT31 evaluation input changed during publication")
        os.rename(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return result
