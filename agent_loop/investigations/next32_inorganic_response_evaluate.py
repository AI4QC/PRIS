"""Frozen confirmation evaluation for the NEXT32 inorganic response law."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from src.next19_feature_build import (
    _publish_directory_no_replace,
    _sha256,
    _strict_json,
)
from src.next32_inorganic_response_features import (
    PAULING_NAME,
    PROTOCOL as FEATURE_PROTOCOL,
)
from src.next32_inorganic_response_rule import (
    FROZEN_RULE_NAME,
    GATE_THRESHOLDS,
    PREDICTIONS_NAME,
    PROMOTION_GATE_NAMES,
    PROTOCOL as RULE_PROTOCOL,
    _candidate_metrics,
    classify_dft_response,
    promotion_gates,
)


PROTOCOL = "2026-08-03-next32-omat24-inorganic-response-evaluation-v1"
CONFIRMATION_PROTOCOL_NAME = "NEXT32_CONFIRMATION_PROTOCOL.json"
MANIFEST_NAME = "MANIFEST.json"
AGGREGATE_GATE_NAMES = PROMOTION_GATE_NAMES
SOURCE_GATE_NAMES = (
    "coverage",
    "protected_recall",
    "severe_precision",
    "savings",
    "auc",
)
PAULING_CONTROL_COLUMNS: Mapping[str, str] = {
    "pauling_p2": "pauling_p2_decision",
    "pauling_p3": "pauling_p3_decision",
    "pauling_p4": "pauling_p4_decision",
    "pauling_p5": "pauling_p5_decision",
    "pauling_p2_p5": "pauling_p2_p5_decision",
}


def source_confirmation_gates(metrics: Mapping[str, float]) -> dict[str, bool]:
    """Evaluate the five preregistered per-source confirmation gates."""

    required = {
        "coverage_lb": 0.90,
        "protected_recall_lb": 0.95,
        "severe_precision_lb": 0.75,
        "savings_lb": 0.02,
        "auc": 0.75,
    }
    try:
        values = {name: float(metrics[name]) for name in required}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("per-source confirmation metrics are incomplete") from exc
    return {
        "coverage": bool(np.isfinite(values["coverage_lb"]) and values["coverage_lb"] >= 0.90),
        "protected_recall": bool(
            np.isfinite(values["protected_recall_lb"])
            and values["protected_recall_lb"] >= 0.95
        ),
        "severe_precision": bool(
            np.isfinite(values["severe_precision_lb"])
            and values["severe_precision_lb"] >= 0.75
        ),
        "savings": bool(np.isfinite(values["savings_lb"]) and values["savings_lb"] >= 0.02),
        "auc": bool(np.isfinite(values["auc"]) and values["auc"] >= 0.75),
    }


def _validated_identity_frame(
    frame: pd.DataFrame, *, role: str, required: set[str]
) -> pd.DataFrame:
    missing = required - set(frame)
    if missing:
        raise ValueError(f"confirmation {role} lacks columns: {sorted(missing)}")
    result = frame.copy()
    result["material_id"] = result["material_id"].astype(str)
    result["source_name"] = result["source_name"].astype(str)
    if (
        result.material_id.isna().any()
        or result.material_id.duplicated().any()
        or result.source_name.eq("").any()
    ):
        raise ValueError(f"confirmation {role} identity is invalid")
    return result


def _json_metrics(metrics: Mapping[str, float | int]) -> dict[str, float | int]:
    return {
        key: int(value) if isinstance(value, (int, np.integer)) else float(value)
        for key, value in metrics.items()
    }


def _response_metrics(
    *,
    risk: np.ndarray,
    support: np.ndarray,
    reject: np.ndarray,
    severe: np.ndarray,
    protected: np.ndarray,
) -> dict[str, float | int]:
    if not (
        risk.shape == support.shape == reject.shape == severe.shape == protected.shape
    ):
        raise ValueError("confirmation response arrays differ")
    if np.any(reject & ~support):
        raise ValueError("confirmation reject decision violates fail-open support")
    return _candidate_metrics(
        score=risk,
        support=support,
        reject=reject,
        severe=severe,
        protected=protected,
    )


def evaluate_confirmation_tables(
    *,
    predictions: pd.DataFrame,
    endpoints: pd.DataFrame,
    pauling_controls: pd.DataFrame,
    expected_sources: Sequence[str],
) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate an exact identity-locked confirmation without fitting anything."""

    sources = tuple(str(source) for source in expected_sources)
    if not sources or len(set(sources)) != len(sources) or any(not source for source in sources):
        raise ValueError("expected confirmation sources are invalid")
    prediction = _validated_identity_frame(
        predictions,
        role="predictions",
        required={
            "material_id",
            "source_name",
            "analytic_supported",
            "next32_risk_score",
            "reject",
        },
    )
    endpoint = _validated_identity_frame(
        endpoints,
        role="endpoints",
        required={
            "material_id",
            "source_name",
            "force_max",
            "force_rms",
            "stress_norm",
        },
    )
    pauling = _validated_identity_frame(
        pauling_controls,
        role="Pauling controls",
        required={"material_id", "source_name", *PAULING_CONTROL_COLUMNS.values()},
    )
    expected_source_set = set(sources)
    for frame, role in (
        (prediction, "predictions"),
        (endpoint, "endpoints"),
        (pauling, "Pauling controls"),
    ):
        if set(frame.source_name) != expected_source_set:
            raise ValueError(f"confirmation {role} source lock differs")
    identity_sets = [
        set(zip(frame.material_id, frame.source_name, strict=True))
        for frame in (prediction, endpoint, pauling)
    ]
    if identity_sets[0] != identity_sets[1] or identity_sets[0] != identity_sets[2]:
        raise ValueError("confirmation identity/source lock differs")

    endpoint_columns = [
        "material_id",
        "source_name",
        "force_max",
        "force_rms",
        "stress_norm",
    ]
    joined = prediction.merge(
        endpoint.loc[:, endpoint_columns],
        on=["material_id", "source_name"],
        validate="one_to_one",
    ).merge(
        pauling.loc[:, ["material_id", "source_name", *PAULING_CONTROL_COLUMNS.values()]],
        on=["material_id", "source_name"],
        validate="one_to_one",
    )
    joined = joined.sort_values(
        ["source_name", "material_id"], kind="stable", ignore_index=True
    )
    if len(joined) != len(prediction):
        raise ValueError("confirmation join lost identities")
    severe, protected = classify_dft_response(joined)
    joined["severe_response"] = severe
    joined["protected_response"] = protected
    risk = pd.to_numeric(joined.next32_risk_score, errors="coerce").to_numpy(float)
    support = joined.analytic_supported.fillna(False).to_numpy(bool)
    reject = joined.reject.fillna(False).to_numpy(bool)
    if np.any(support & ~np.isfinite(risk)) or np.any(~support & np.isfinite(risk)):
        raise ValueError("confirmation risk/support accounting differs")
    aggregate_metrics = _response_metrics(
        risk=risk,
        support=support,
        reject=reject,
        severe=severe,
        protected=protected,
    )
    aggregate_gates = promotion_gates(
        {key: float(aggregate_metrics[key]) for key in (
            "coverage_lb",
            "protected_recall_lb",
            "severe_precision_lb",
            "savings_lb",
            "auc",
            "precision_lift_over_prevalence",
        )}
    )

    by_source: dict[str, object] = {}
    for source in sources:
        selected = joined.source_name.eq(source).to_numpy()
        metrics = _response_metrics(
            risk=risk[selected],
            support=support[selected],
            reject=reject[selected],
            severe=severe[selected],
            protected=protected[selected],
        )
        gates = source_confirmation_gates(
            {key: float(metrics[key]) for key in (
                "coverage_lb",
                "protected_recall_lb",
                "severe_precision_lb",
                "savings_lb",
                "auc",
            )}
        )
        by_source[source] = {
            "metrics": _json_metrics(metrics),
            "gates": gates,
            "passed": all(gates.values()),
        }

    pauling_results: dict[str, object] = {}
    for control, column in PAULING_CONTROL_COLUMNS.items():
        decisions = joined[column].astype(str).to_numpy()
        if not set(decisions).issubset({"KEEP", "REJECT", "ABSTAIN"}):
            raise ValueError(f"confirmation {control} decisions are invalid")
        control_support = decisions != "ABSTAIN"
        control_reject = decisions == "REJECT"
        control_risk = np.full(len(joined), np.nan, dtype=float)
        control_risk[control_support] = control_reject[control_support].astype(float)
        metrics = _response_metrics(
            risk=control_risk,
            support=control_support,
            reject=control_reject,
            severe=severe,
            protected=protected,
        )
        gates = promotion_gates(
            {key: float(metrics[key]) for key in (
                "coverage_lb",
                "protected_recall_lb",
                "severe_precision_lb",
                "savings_lb",
                "auc",
                "precision_lift_over_prevalence",
            )}
        )
        pauling_results[control] = {
            "metrics": _json_metrics(metrics),
            "aggregate_gates": gates,
            "aggregate_gates_passed": all(gates.values()),
        }

    next32_passed = all(aggregate_gates.values()) and all(
        bool(item["passed"]) for item in by_source.values()  # type: ignore[union-attr]
    )
    all_pauling_failed = all(
        not bool(item["aggregate_gates_passed"])
        for item in pauling_results.values()  # type: ignore[union-attr]
    )
    evaluation: dict[str, object] = {
        "protocol": PROTOCOL,
        "expected_sources": list(sources),
        "next32_aggregate_metrics": _json_metrics(aggregate_metrics),
        "next32_aggregate_gates": aggregate_gates,
        "next32_aggregate_passed": all(aggregate_gates.values()),
        "next32_by_source": by_source,
        "next32_confirmation_passed": next32_passed,
        "pauling_controls": pauling_results,
        "all_pauling_controls_failed_aggregate_gates": all_pauling_failed,
        "beyond_pauling_on_this_endpoint": next32_passed and all_pauling_failed,
        "thresholds_or_formula_refit": False,
        "claim_boundary": "severe OMat24 initial DFT force/stress response only",
    }
    return evaluation, joined


def freeze_confirmation_protocol(
    *,
    predictions_path: Path,
    predictions_manifest_path: Path,
    frozen_rule_path: Path,
    frozen_rule_manifest_path: Path,
    pauling_paths: Sequence[Path],
    pauling_manifest_paths: Sequence[Path],
    expected_sources: Sequence[str],
    expected_rows_per_source: int,
    output_dir: Path,
) -> dict[str, object]:
    """Freeze confirmation identities, inputs, endpoints and gates before labels."""

    paths = {
        "predictions": Path(predictions_path).resolve(),
        "predictions_manifest": Path(predictions_manifest_path).resolve(),
        "frozen_rule": Path(frozen_rule_path).resolve(),
        "frozen_rule_manifest": Path(frozen_rule_manifest_path).resolve(),
    }
    pauling_candidates = [Path(path).resolve() for path in pauling_paths]
    pauling_manifest_candidates = [
        Path(path).resolve() for path in pauling_manifest_paths
    ]
    sources = tuple(str(source) for source in expected_sources)
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if (
        not sources
        or len(set(sources)) != len(sources)
        or any(not source for source in sources)
        or type(expected_rows_per_source) is not int
        or expected_rows_per_source <= 0
        or not pauling_candidates
        or len(pauling_candidates) != len(pauling_manifest_candidates)
        or any(not path.is_file() for path in (*paths.values(), *pauling_candidates, *pauling_manifest_candidates))
    ):
        raise ValueError("NEXT32 confirmation protocol inputs are invalid")
    if paths["predictions"].name != PREDICTIONS_NAME or paths["frozen_rule"].name != FROZEN_RULE_NAME:
        raise ValueError("NEXT32 confirmation prediction or rule path/name is invalid")

    prediction_manifest = _strict_json(
        paths["predictions_manifest"], role="NEXT32 prediction manifest"
    )
    prediction_outputs = prediction_manifest.get("outputs_sha256")
    if (
        prediction_manifest.get("protocol") != RULE_PROTOCOL
        or prediction_manifest.get("labels_opened") is not False
        or prediction_manifest.get("endpoint_fields_read") is not False
        or not isinstance(prediction_outputs, Mapping)
        or prediction_outputs.get(PREDICTIONS_NAME) != _sha256(paths["predictions"])
    ):
        raise ValueError("NEXT32 predictions crossed the pre-label boundary")
    rule_manifest = _strict_json(
        paths["frozen_rule_manifest"], role="NEXT32 frozen-rule manifest"
    )
    rule_outputs = rule_manifest.get("outputs_sha256")
    if (
        rule_manifest.get("protocol") != RULE_PROTOCOL
        or rule_manifest.get("promoted") is not True
        or not isinstance(rule_outputs, Mapping)
        or rule_outputs.get(FROZEN_RULE_NAME) != _sha256(paths["frozen_rule"])
    ):
        raise ValueError("NEXT32 frozen rule is not promoted and hash-locked")

    prediction = _validated_identity_frame(
        pd.read_parquet(paths["predictions"]),
        role="predictions",
        required={
            "material_id",
            "source_name",
            "analytic_supported",
            "next32_risk_score",
            "reject",
        },
    )
    if set(prediction.source_name) != set(sources):
        raise ValueError("NEXT32 confirmation prediction sources differ")
    counts = prediction.groupby("source_name").size().to_dict()
    if any(counts.get(source) != expected_rows_per_source for source in sources):
        raise ValueError("NEXT32 confirmation source row counts differ")

    pauling_parts: list[pd.DataFrame] = []
    pauling_input_hashes: list[dict[str, str]] = []
    for pauling_path, manifest_path in zip(
        pauling_candidates, pauling_manifest_candidates, strict=True
    ):
        if pauling_path.name != PAULING_NAME:
            raise ValueError("NEXT32 Pauling artifact path/name is invalid")
        manifest = _strict_json(manifest_path, role="NEXT32 Pauling feature manifest")
        outputs = manifest.get("outputs_sha256")
        if (
            manifest.get("protocol") != FEATURE_PROTOCOL
            or manifest.get("labels_opened") is not False
            or manifest.get("endpoint_fields_read") is not False
            or not isinstance(outputs, Mapping)
            or outputs.get(PAULING_NAME) != _sha256(pauling_path)
        ):
            raise ValueError("NEXT32 Pauling artifact crossed the pre-label boundary")
        pauling_parts.append(
            _validated_identity_frame(
                pd.read_parquet(pauling_path),
                role="Pauling controls",
                required={"material_id", "source_name", *PAULING_CONTROL_COLUMNS.values()},
            )
        )
        pauling_input_hashes.append(
            {
                "pauling_path": str(pauling_path),
                "pauling_sha256": _sha256(pauling_path),
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
            }
        )
    pauling = pd.concat(pauling_parts, ignore_index=True)
    if pauling.material_id.duplicated().any():
        raise ValueError("NEXT32 combined Pauling identities are duplicated")
    prediction_identities = set(zip(prediction.material_id, prediction.source_name, strict=True))
    pauling_identities = set(zip(pauling.material_id, pauling.source_name, strict=True))
    if prediction_identities != pauling_identities:
        raise ValueError("NEXT32 prediction and Pauling identity locks differ")

    ordered = prediction.sort_values(
        ["source_name", "material_id"], kind="stable", ignore_index=True
    )
    identity_payload = "\n".join(
        f"{source}\t{material_id}"
        for material_id, source in zip(
            ordered.material_id, ordered.source_name, strict=True
        )
    ) + "\n"
    fixed_inputs: dict[str, object] = {
        role: {"path": str(path), "sha256": _sha256(path)}
        for role, path in paths.items()
    }
    fixed_inputs["pauling_artifacts"] = pauling_input_hashes
    protocol_document: dict[str, object] = {
        "protocol": PROTOCOL,
        "status": "frozen_before_confirmation_endpoint_opening",
        "confirmation_labels_opened": False,
        "expected_sources": list(sources),
        "expected_rows_per_source": expected_rows_per_source,
        "rows": len(ordered),
        "identity_order_sha256": hashlib.sha256(identity_payload.encode("utf-8")).hexdigest(),
        "severe_response_endpoint": {
            "force_max_ge": 1.0,
            "force_rms_ge": 0.40,
            "stress_norm_ge": 0.030,
            "logic": "OR",
        },
        "protected_response_endpoint": {
            "force_max_le": 0.50,
            "force_rms_le": 0.20,
            "stress_norm_le": 0.015,
            "logic": "AND",
        },
        "aggregate_gate_thresholds": dict(GATE_THRESHOLDS),
        "source_gate_thresholds": {
            "coverage_lb": 0.90,
            "protected_recall_lb": 0.95,
            "severe_precision_lb": 0.75,
            "savings_lb": 0.02,
            "auc": 0.75,
        },
        "pauling_controls": list(PAULING_CONTROL_COLUMNS),
        "inputs_sha256": fixed_inputs,
        "thresholds_or_formula_refit_after_opening": False,
    }
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    manifest_document: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "confirmation_protocol_freeze_before_endpoints",
        "labels_opened": False,
        "endpoint_fields_read": False,
        "inputs_sha256": fixed_inputs,
        "executed_source_sha256": {
            "src/next32_inorganic_response_evaluate.py": source_hash
        },
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        protocol_path = staging / CONFIRMATION_PROTOCOL_NAME
        protocol_path.write_text(
            json.dumps(protocol_document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        manifest_document["outputs_sha256"] = {
            CONFIRMATION_PROTOCOL_NAME: _sha256(protocol_path)
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest_document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for role, path in paths.items():
            expected = fixed_inputs[role]
            assert isinstance(expected, Mapping)
            if _sha256(path) != expected["sha256"]:
                raise RuntimeError("NEXT32 confirmation protocol input changed")
        for item in pauling_input_hashes:
            if _sha256(Path(item["pauling_path"])) != item["pauling_sha256"] or _sha256(
                Path(item["manifest_path"])
            ) != item["manifest_sha256"]:
                raise RuntimeError("NEXT32 Pauling input changed during protocol freeze")
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT32 confirmation protocol source changed")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest_document


__all__ = [
    "AGGREGATE_GATE_NAMES",
    "CONFIRMATION_PROTOCOL_NAME",
    "MANIFEST_NAME",
    "PAULING_CONTROL_COLUMNS",
    "PROTOCOL",
    "SOURCE_GATE_NAMES",
    "evaluate_confirmation_tables",
    "freeze_confirmation_protocol",
    "source_confirmation_gates",
]
