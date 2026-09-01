from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.next32_inorganic_response_evaluate import (
    AGGREGATE_GATE_NAMES,
    CONFIRMATION_PROTOCOL_NAME,
    MANIFEST_NAME,
    SOURCE_GATE_NAMES,
    evaluate_confirmation_tables,
    freeze_confirmation_protocol,
    source_confirmation_gates,
)
from src.next32_inorganic_response_features import (
    PAULING_NAME,
    PROTOCOL as FEATURE_PROTOCOL,
)
from src.next32_inorganic_response_rule import (
    FROZEN_RULE_NAME,
    PREDICTIONS_NAME,
    PROTOCOL as RULE_PROTOCOL,
)


def test_source_confirmation_gates_are_exact_and_all_required() -> None:
    metrics = {
        "coverage_lb": 0.90,
        "protected_recall_lb": 0.95,
        "severe_precision_lb": 0.75,
        "savings_lb": 0.02,
        "auc": 0.75,
    }

    gates = source_confirmation_gates(metrics)

    assert tuple(gates) == SOURCE_GATE_NAMES
    assert all(gates.values())
    for name in metrics:
        failed = dict(metrics)
        failed[name] = np.nextafter(metrics[name], -np.inf)
        assert not all(source_confirmation_gates(failed).values())


def _confirmation(source_names: tuple[str, ...] = ("rattled-300", "rattled-500", "rattled-1000")):
    prediction_parts = []
    endpoint_parts = []
    pauling_parts = []
    for source in source_names:
        n = 1000
        identities = [f"{source}::{index:04d}" for index in range(n)]
        severe = np.arange(n) < 100
        prediction_parts.append(
            pd.DataFrame(
                {
                    "material_id": identities,
                    "source_name": source,
                    "analytic_supported": True,
                    "next32_risk_score": np.linspace(2.0, 0.0, n),
                    "reject": severe,
                }
            )
        )
        endpoint_parts.append(
            pd.DataFrame(
                {
                    "material_id": identities,
                    "source_name": source,
                    "force_max": np.where(severe, 1.5, 0.1),
                    "force_rms": np.where(severe, 0.5, 0.1),
                    "stress_norm": np.where(severe, 0.04, 0.005),
                }
            )
        )
        pauling = {"material_id": identities, "source_name": source}
        for name in ("pauling_p2", "pauling_p3", "pauling_p4", "pauling_p5"):
            pauling[f"{name}_decision"] = "KEEP"
        pauling["pauling_p2_p5_decision"] = "KEEP"
        pauling_parts.append(pd.DataFrame(pauling))
    return tuple(
        pd.concat(parts, ignore_index=True)
        for parts in (prediction_parts, endpoint_parts, pauling_parts)
    )


def test_confirmation_requires_aggregate_every_source_and_all_pauling_fail() -> None:
    predictions, endpoints, pauling = _confirmation()

    evaluation, joined = evaluate_confirmation_tables(
        predictions=predictions,
        endpoints=endpoints,
        pauling_controls=pauling,
        expected_sources=("rattled-300", "rattled-500", "rattled-1000"),
    )

    assert tuple(evaluation["next32_aggregate_gates"]) == AGGREGATE_GATE_NAMES
    assert evaluation["next32_aggregate_passed"] is True
    assert all(item["passed"] for item in evaluation["next32_by_source"].values())
    assert not any(item["aggregate_gates_passed"] for item in evaluation["pauling_controls"].values())
    assert evaluation["beyond_pauling_on_this_endpoint"] is True
    assert len(joined) == 3000


def test_one_failed_source_prevents_beyond_pauling_claim() -> None:
    predictions, endpoints, pauling = _confirmation()
    mask = predictions.source_name.eq("rattled-500")
    predictions.loc[mask, "next32_risk_score"] = np.linspace(0.0, 2.0, mask.sum())
    predictions.loc[mask, "reject"] = False

    evaluation, _joined = evaluate_confirmation_tables(
        predictions=predictions,
        endpoints=endpoints,
        pauling_controls=pauling,
        expected_sources=("rattled-300", "rattled-500", "rattled-1000"),
    )

    assert evaluation["next32_by_source"]["rattled-500"]["passed"] is False
    assert evaluation["beyond_pauling_on_this_endpoint"] is False


def test_confirmation_join_requires_exact_identity_and_source_lock() -> None:
    predictions, endpoints, pauling = _confirmation(("rattled-300",))
    endpoints.loc[0, "source_name"] = "wrong-source"

    try:
        evaluate_confirmation_tables(
            predictions=predictions,
            endpoints=endpoints,
            pauling_controls=pauling,
            expected_sources=("rattled-300",),
        )
    except ValueError as exc:
        assert "identity" in str(exc) or "source" in str(exc)
    else:
        raise AssertionError("source mismatch was accepted")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def test_confirmation_protocol_freezes_ids_and_hashes_before_endpoints(
    tmp_path: Path,
) -> None:
    predictions, _endpoints, pauling = _confirmation()
    prediction_path = tmp_path / PREDICTIONS_NAME
    prediction_path.write_bytes(b"")
    predictions.to_parquet(prediction_path, index=False)
    prediction_manifest_path = tmp_path / "prediction-manifest.json"
    prediction_manifest_path.write_text(
        json.dumps(
            {
                "protocol": RULE_PROTOCOL,
                "labels_opened": False,
                "endpoint_fields_read": False,
                "outputs_sha256": {PREDICTIONS_NAME: _sha256(prediction_path)},
            }
        )
        + "\n"
    )
    rule_path = tmp_path / FROZEN_RULE_NAME
    rule_path.write_text(json.dumps({"protocol": RULE_PROTOCOL, "eligible": True}) + "\n")
    rule_manifest_path = tmp_path / "rule-manifest.json"
    rule_manifest_path.write_text(
        json.dumps(
            {
                "protocol": RULE_PROTOCOL,
                "promoted": True,
                "outputs_sha256": {FROZEN_RULE_NAME: _sha256(rule_path)},
            }
        )
        + "\n"
    )
    pauling_path = tmp_path / PAULING_NAME
    pauling.to_parquet(pauling_path, index=False)
    pauling_manifest_path = tmp_path / "pauling-manifest.json"
    pauling_manifest_path.write_text(
        json.dumps(
            {
                "protocol": FEATURE_PROTOCOL,
                "labels_opened": False,
                "endpoint_fields_read": False,
                "outputs_sha256": {PAULING_NAME: _sha256(pauling_path)},
            }
        )
        + "\n"
    )

    output = tmp_path / "protocol"
    manifest = freeze_confirmation_protocol(
        predictions_path=prediction_path,
        predictions_manifest_path=prediction_manifest_path,
        frozen_rule_path=rule_path,
        frozen_rule_manifest_path=rule_manifest_path,
        pauling_paths=[pauling_path],
        pauling_manifest_paths=[pauling_manifest_path],
        expected_sources=("rattled-300", "rattled-500", "rattled-1000"),
        expected_rows_per_source=1000,
        output_dir=output,
    )

    protocol = json.loads((output / CONFIRMATION_PROTOCOL_NAME).read_text())
    assert protocol["confirmation_labels_opened"] is False
    assert protocol["rows"] == 3000
    assert protocol["identity_order_sha256"]
    assert manifest["outputs_sha256"][CONFIRMATION_PROTOCOL_NAME] == _sha256(
        output / CONFIRMATION_PROTOCOL_NAME
    )
    with pytest.raises(FileExistsError):
        freeze_confirmation_protocol(
            predictions_path=prediction_path,
            predictions_manifest_path=prediction_manifest_path,
            frozen_rule_path=rule_path,
            frozen_rule_manifest_path=rule_manifest_path,
            pauling_paths=[pauling_path],
            pauling_manifest_paths=[pauling_manifest_path],
            expected_sources=("rattled-300", "rattled-500", "rattled-1000"),
            expected_rows_per_source=1000,
            output_dir=output,
        )
