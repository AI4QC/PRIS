import hashlib
import inspect
import json

import pandas as pd
import pytest


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _eligible_terms() -> list[dict[str, object]]:
    return [
        {
            "term_id": "sivr_edge_mismatch_max__high",
            "feature": "sivr_edge_mismatch_max",
            "group": "valence_rigidity",
            "direction": 1,
            "transform": "log1p_nonnegative",
            "center": 0.34809689849136527,
            "scale": 0.2268496027212349,
        },
        {
            "term_id": "sscp_load_rms__high",
            "feature": "sscp_load_rms",
            "group": "self_stress_compatibility",
            "direction": 1,
            "transform": "log1p_nonnegative",
            "center": 0.09650974330938514,
            "scale": 0.07475030243877033,
        },
    ]


def test_build_formula_is_the_exact_two_term_rli_candidate() -> None:
    from src.next91_scigen_fixed_rli_candidate import (
        RLI_THRESHOLD,
        build_frozen_rli_formula,
    )

    formula = build_frozen_rli_formula(_eligible_terms())

    assert formula["kind"] == "nonnegative_hinge_sum"
    assert formula["missing_policy"] == "KEEP"
    assert formula["threshold"] == RLI_THRESHOLD == 3.915855102781074
    assert [term["term_id"] for term in formula["terms"]] == [
        "sivr_edge_mismatch_max__high",
        "sscp_load_rms__high",
    ]
    assert [term["weight"] for term in formula["terms"]] == [1.0, 4.0]

    with pytest.raises(ValueError, match="fixed RLI term"):
        build_frozen_rli_formula(_eligible_terms()[:-1])


def test_freezer_has_no_endpoint_input_and_publishes_three_clean_predictions(
    tmp_path,
) -> None:
    from src.next85_scigen_label_free_features import (
        CATALOGUE_NAME as FEATURE_CATALOGUE_NAME,
        FEATURE_NAMES,
        MANIFEST_NAME as FEATURE_MANIFEST_NAME,
        PROTOCOL as FEATURE_PROTOCOL,
    )
    from src.next86_scigen_term_catalogue import (
        CATALOGUE_NAME as TERM_CATALOGUE_NAME,
        MANIFEST_NAME as TERM_MANIFEST_NAME,
        PROTOCOL as TERM_PROTOCOL,
    )
    from src.next87_scigen_sparse_law_search import (
        EVALUATION_NAME as NEXT87_EVALUATION_NAME,
        MANIFEST_NAME as NEXT87_MANIFEST_NAME,
        PROTOCOL as NEXT87_PROTOCOL,
        SEARCH_RECORD_NAME as NEXT87_SEARCH_RECORD_NAME,
    )
    from src.next91_scigen_fixed_rli_candidate import (
        FORMULA_NAME,
        MANIFEST_NAME,
        PREDICTION_NAMES,
        freeze_scigen_rli_candidate,
    )

    assert not {
        "discovery_endpoint",
        "validation_endpoint",
        "replication_endpoint",
    } & set(inspect.signature(freeze_scigen_rli_candidate).parameters)
    feature_dir = tmp_path / "features"
    term_dir = tmp_path / "terms"
    next87_dir = tmp_path / "next87"
    feature_dir.mkdir()
    term_dir.mkdir()
    next87_dir.mkdir()
    feature_paths = {}
    for role in FEATURE_NAMES:
        table = pd.DataFrame(
            {
                "material_id": [f"{role}-keep", f"{role}-reject"],
                "partition_role": role,
                "sivr_edge_mismatch_max": [0.0, 3.0],
                "sscp_load_rms": [0.0, 3.0],
            }
        )
        path = feature_dir / FEATURE_NAMES[role]
        table.to_parquet(path, index=False)
        feature_paths[role] = path
    feature_catalogue_path = feature_dir / FEATURE_CATALOGUE_NAME
    feature_catalogue_path.write_bytes(
        _json_bytes({"protocol": FEATURE_PROTOCOL, "labels_opened": False})
    )
    (feature_dir / FEATURE_MANIFEST_NAME).write_bytes(
        _json_bytes(
            {
                "protocol": FEATURE_PROTOCOL,
                "labels_opened": False,
                "endpoint_payloads_opened": False,
                "relaxed_structures_opened": False,
                "outputs_sha256": {
                    FEATURE_CATALOGUE_NAME: _sha256(feature_catalogue_path),
                    **{
                        FEATURE_NAMES[role]: _sha256(path)
                        for role, path in feature_paths.items()
                    },
                },
            }
        )
    )
    term_catalogue_path = term_dir / TERM_CATALOGUE_NAME
    term_catalogue_path.write_bytes(
        _json_bytes(
            {
                "protocol": TERM_PROTOCOL,
                "labels_opened": False,
                "eligible_terms": _eligible_terms(),
            }
        )
    )
    (term_dir / TERM_MANIFEST_NAME).write_bytes(
        _json_bytes(
            {
                "protocol": TERM_PROTOCOL,
                "labels_opened": False,
                "endpoint_payloads_opened": False,
                "outputs_sha256": {TERM_CATALOGUE_NAME: _sha256(term_catalogue_path)},
            }
        )
    )
    search_path = next87_dir / NEXT87_SEARCH_RECORD_NAME
    pd.DataFrame(
        [
            {
                "term_ids_json": json.dumps(
                    [
                        "sivr_edge_mismatch_max__high",
                        "sscp_load_rms__high",
                    ],
                    separators=(",", ":"),
                ),
                "weights_json": "[1.0,4.0]",
                "threshold": 3.915855102781074,
                "metric_severe_rejected": 1359,
                "metric_severe_rejection_precision_lower": 0.9429092910402352,
                "metric_protected_recall_lower": 0.9746034559783944,
                "metric_coverage_lower": 0.9794285414401448,
                "metric_savings_lower": 0.11953000620096647,
                "pooled_extreme_auc": 0.7788719322036274,
            }
        ]
    ).to_parquet(search_path, index=False)
    evaluation_path = next87_dir / NEXT87_EVALUATION_NAME
    evaluation_path.write_bytes(
        _json_bytes({"protocol": NEXT87_PROTOCOL, "passes_discovery_gates": False})
    )
    (next87_dir / NEXT87_MANIFEST_NAME).write_bytes(
        _json_bytes(
            {
                "protocol": NEXT87_PROTOCOL,
                "validation_endpoint_opened": False,
                "replication_endpoint_opened": False,
                "outputs_sha256": {
                    NEXT87_SEARCH_RECORD_NAME: _sha256(search_path),
                    NEXT87_EVALUATION_NAME: _sha256(evaluation_path),
                },
            }
        )
    )
    design = tmp_path / "next91-design.md"
    design.write_text("fixed RLI candidate\n")
    output_dir = tmp_path / "next91"

    manifest = freeze_scigen_rli_candidate(
        feature_dir=feature_dir,
        term_catalogue_dir=term_dir,
        next87_dir=next87_dir,
        design_path=design,
        output_dir=output_dir,
        require_formal_inputs=False,
    )

    assert manifest["endpoint_payloads_opened"] is False
    assert manifest["validation_endpoint_opened"] is False
    assert manifest["replication_endpoint_opened"] is False
    assert (output_dir / FORMULA_NAME).is_file()
    for role in FEATURE_NAMES:
        prediction = pd.read_parquet(output_dir / PREDICTION_NAMES[role])
        assert set(prediction["partition_role"]) == {role}
        assert not {"distortion_ratio", "protected", "severe"} & set(prediction.columns)
    assert (output_dir / MANIFEST_NAME).is_file()
