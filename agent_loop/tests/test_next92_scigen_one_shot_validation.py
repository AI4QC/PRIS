import hashlib
import inspect
import json

import numpy as np
import pandas as pd


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def test_partition_evaluation_uses_frozen_scores_and_beats_pauling() -> None:
    from src.next92_scigen_one_shot_validation import evaluate_rli_partition

    rows = []
    endpoints = []
    decisions = []
    for index in range(120):
        lattice = f"L{index % 8}"
        for state, score, distortion in (
            ("protected", 0.0, 0.5),
            ("severe", 5.0, 2.5),
        ):
            material_id = f"m{index}-{state}"
            rows.append(
                {
                    "material_id": material_id,
                    "partition_role": "internal_validation",
                    "rli_score": score,
                    "rli_supported": True,
                    "rli_reject": state == "severe",
                    "rli_decision": "REJECT" if state == "severe" else "KEEP",
                    "formula_sha256": "a" * 64,
                }
            )
            endpoints.append(
                {
                    "material_id": material_id,
                    "lattice_class": lattice,
                    "partition_role": "internal_validation",
                    "distortion_ratio": distortion,
                }
            )
            decisions.append(
                {
                    "material_id": material_id,
                    "pauling_p2_p5_decision": (
                        "REJECT" if index == 0 and state == "severe" else "ABSTAIN"
                    ),
                }
            )

    result = evaluate_rli_partition(
        predictions=pd.DataFrame(rows),
        endpoints=pd.DataFrame(endpoints),
        pauling=pd.DataFrame(decisions),
        expected_role="internal_validation",
    )

    assert result["passes_all_gates"] is True
    assert result["metrics"]["pooled_extreme_auc"] == 1.0
    assert result["metrics"]["evaluable_lattices"] == 8
    assert result["metrics"]["severe_rejected"] == 120
    assert result["pauling_baseline"]["severe_rejected"] == 1
    assert result["beats_pauling"] is True


def test_one_shot_runner_has_no_replication_input_and_publishes_validation(
    tmp_path,
) -> None:
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
    )
    from src.next92_scigen_one_shot_validation import (
        EVALUATION_NAME,
        MANIFEST_NAME,
        run_one_shot_rli_validation,
    )

    assert "replication_endpoint" not in inspect.signature(
        run_one_shot_rli_validation
    ).parameters
    frozen_dir = tmp_path / "frozen"
    endpoint_dir = tmp_path / "validation_endpoint"
    frozen_dir.mkdir()
    endpoint_dir.mkdir()
    formula_path = frozen_dir / FORMULA_NAME
    formula_path.write_bytes(
        _json_bytes(
            {
                "protocol": FROZEN_PROTOCOL,
                "candidate_name": "Rigidity-Load Incompatibility (RLI)",
                "threshold": 3.915855102781074,
                "validation_endpoint_opened": False,
                "replication_endpoint_opened": False,
            }
        )
    )
    formula_hash = _sha256(formula_path)
    predictions = []
    endpoints = []
    pauling_rows = []
    for index in range(120):
        lattice = f"L{index % 8}"
        for state, score, distortion in (
            ("protected", 0.0, 0.5),
            ("severe", 5.0, 2.5),
        ):
            material_id = f"v{index}-{state}"
            reject = state == "severe"
            predictions.append(
                {
                    "material_id": material_id,
                    "partition_role": "internal_validation",
                    "rli_score": score,
                    "rli_supported": True,
                    "rli_reject": reject,
                    "rli_decision": "REJECT" if reject else "KEEP",
                    "formula_sha256": formula_hash,
                }
            )
            endpoints.append(
                {
                    "material_id": material_id,
                    "lattice_class": lattice,
                    "partition_role": "internal_validation",
                    "distortion_ratio": distortion,
                }
            )
            pauling_rows.append(
                {
                    "material_id": material_id,
                    "partition_role": "internal_validation",
                    "pauling_p2_p5_decision": (
                        "REJECT" if index == 0 and state == "severe" else "ABSTAIN"
                    ),
                }
            )
    prediction_path = frozen_dir / PREDICTION_NAMES["internal_validation"]
    pd.DataFrame(predictions).to_parquet(prediction_path, index=False)
    feature_path = tmp_path / "validation_features.parquet"
    pd.DataFrame(pauling_rows).to_parquet(feature_path, index=False)
    (frozen_dir / FROZEN_MANIFEST_NAME).write_bytes(
        _json_bytes(
            {
                "protocol": FROZEN_PROTOCOL,
                "endpoint_payloads_opened": False,
                "validation_endpoint_opened": False,
                "replication_endpoint_opened": False,
                "inputs_sha256": {
                    "features_internal_validation": {
                        "path": str(feature_path),
                        "sha256": _sha256(feature_path),
                    }
                },
                "outputs_sha256": {
                    FORMULA_NAME: formula_hash,
                    PREDICTION_NAMES["internal_validation"]: _sha256(prediction_path),
                },
            }
        )
    )
    endpoint_path = endpoint_dir / ENDPOINT_NAME
    pd.DataFrame(endpoints).to_parquet(endpoint_path, index=False)
    (endpoint_dir / ENDPOINT_MANIFEST_NAME).write_bytes(
        _json_bytes(
            {
                "protocol": ENDPOINT_PROTOCOL,
                "partition_role": "internal_validation",
                "lockbox_state": "unopened_for_model_development",
                "outputs_sha256": {ENDPOINT_NAME: _sha256(endpoint_path)},
            }
        )
    )
    output_dir = tmp_path / "next92"

    manifest = run_one_shot_rli_validation(
        frozen_dir=frozen_dir,
        validation_endpoint_dir=endpoint_dir,
        output_dir=output_dir,
        require_formal_inputs=False,
    )

    assert manifest["passes_validation_gates"] is True
    assert manifest["validation_endpoint_opened"] is True
    assert manifest["replication_endpoint_opened"] is False
    assert (output_dir / EVALUATION_NAME).is_file()
    assert (output_dir / MANIFEST_NAME).is_file()
