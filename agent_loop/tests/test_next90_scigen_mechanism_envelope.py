import hashlib
import inspect
import json

import numpy as np
import pandas as pd
import pytest


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _term(term_id: str, feature: str) -> dict[str, object]:
    return {
        "term_id": term_id,
        "feature": feature,
        "direction": 1,
        "transform": "asinh",
        "center": 0.0,
        "scale": 1.0,
    }


def _synthetic_mechanism_case(prefix: str = "M"):
    from src.next87_scigen_sparse_law_search import assign_group_folds
    from src.next90_scigen_mechanism_envelope import FIXED_ENVELOPE_TERMS

    groups_by_fold: dict[int, list[str]] = {fold: [] for fold in range(5)}
    cursor = 0
    while any(len(groups) < 25 for groups in groups_by_fold.values()):
        group = f"{prefix}{cursor}N"
        fold = int(assign_group_folds([group])[0])
        if len(groups_by_fold[fold]) < 25:
            groups_by_fold[fold].append(group)
        cursor += 1

    eligible_terms = []
    for term_id in (term for terms in FIXED_ENVELOPE_TERMS.values() for term in terms):
        feature = term_id.rsplit("__", 1)[0]
        low = term_id.endswith("__low")
        eligible_terms.append(
            {
                "term_id": term_id,
                "feature": feature,
                "group": "synthetic",
                "direction": -1 if low else 1,
                "transform": "log1p_nonnegative",
                "center": float(np.log1p(3.0)) if low else 0.0,
                "scale": 1.0,
            }
        )

    rows = []
    endpoints = []
    for group_index, group in enumerate(
        group for fold in range(5) for group in groups_by_fold[fold]
    ):
        lattice = f"L{group_index % 8}"
        for state, distortion in (("protected", 0.5), ("severe", 2.5)):
            row = {
                "material_id": f"{group}-{state}",
                "reduced_formula": group,
                "lattice_class": lattice,
                "partition_role": "discovery",
                "pauling_p2_p5_decision": (
                    "REJECT" if group_index == 0 and state == "severe" else "ABSTAIN"
                ),
            }
            for term in eligible_terms:
                low = int(term["direction"]) == -1
                row[str(term["feature"])] = (
                    3.0 if (state == "protected") == low else 0.0
                )
            rows.append(row)
            endpoints.append(distortion)
    return pd.DataFrame(rows), np.asarray(endpoints), eligible_terms


def test_apply_mechanism_formula_uses_max_envelopes_and_any_missing_keeps() -> None:
    from src.next90_scigen_mechanism_envelope import apply_mechanism_formula

    features = pd.DataFrame(
        {
            "b1": [2.0, np.nan],
            "b2": [3.0, 4.0],
            "v": [0.0, 0.0],
            "e": [0.0, 0.0],
            "l": [0.0, 0.0],
        }
    )
    formula = {
        "kind": "coupled_mechanism_envelope",
        "missing_policy": "KEEP",
        "envelopes": [
            {
                "envelope_id": "B",
                "aggregation": "max",
                "weight": 1.0,
                "terms": [_term("b1__high", "b1"), _term("b2__high", "b2")],
            },
            {
                "envelope_id": "V",
                "aggregation": "max",
                "weight": 0.25,
                "terms": [_term("v__high", "v")],
            },
            {
                "envelope_id": "E",
                "aggregation": "max",
                "weight": 0.5,
                "terms": [_term("e__high", "e")],
            },
            {
                "envelope_id": "L",
                "aggregation": "max",
                "weight": 2.0,
                "terms": [_term("l__high", "l")],
            },
        ],
        "threshold": 1.5,
    }

    score, supported, reject = apply_mechanism_formula(features, formula)

    assert score[0] == pytest.approx(np.arcsinh(3.0))
    assert np.isnan(score[1])
    assert supported.tolist() == [True, False]
    assert reject.tolist() == [True, False]

    invalid = {**formula, "envelopes": formula["envelopes"][:-1]}
    with pytest.raises(ValueError, match="B, V, E, L"):
        apply_mechanism_formula(features, invalid)


def test_fixed_envelope_catalogue_matches_the_frozen_design() -> None:
    from src.next90_scigen_mechanism_envelope import FIXED_ENVELOPE_TERMS

    assert FIXED_ENVELOPE_TERMS == {
        "B": (
            "scbv_anion_mismatch_rms__high",
            "scbv_mismatch_q95__high",
            "scbv_mismatch_max__high",
        ),
        "V": (
            "sivr_edge_mismatch_rms__high",
            "sivr_edge_mismatch_max__high",
            "sivr_stiffness_min__low",
        ),
        "E": (
            "aefi_residual_rms__high",
            "aefi_residual_q95__high",
            "aefi_residual_max__high",
        ),
        "L": (
            "sscp_load_rms__high",
            "sscp_load_q95__high",
            "sscp_load_fraction__low",
            "prlr_residual_fraction__high",
            "prlr_cell_residual_fraction__high",
            "prlr_risk__high",
        ),
    }


def test_mechanism_search_requires_exact_weight_stability_and_beats_pauling() -> None:
    from src.next90_scigen_mechanism_envelope import search_mechanism_envelope_law

    features, endpoints, eligible_terms = _synthetic_mechanism_case()

    result = search_mechanism_envelope_law(
        features=features,
        distortion_ratio=endpoints,
        eligible_terms=eligible_terms,
    )

    assert result["candidate_count"] == 125
    assert result["passes_discovery_gates"] is True
    assert result["weight_stability"]["selected_weight_win_count"] == 5
    assert result["discovery_metrics"]["severe_rejected"] == 125
    assert result["pauling_baseline"]["severe_rejected"] == 1
    assert all(fold["passes_raw_fold_gates"] for fold in result["fold_diagnostics"])


def test_runner_cannot_receive_locked_endpoints_and_freezes_clean_predictions(
    tmp_path,
) -> None:
    from src.next85_scigen_label_free_features import (
        CATALOGUE_NAME as FEATURE_CATALOGUE_NAME,
        FEATURE_NAMES,
        MANIFEST_NAME as FEATURE_MANIFEST_NAME,
        PROTOCOL as FEATURE_PROTOCOL,
    )
    from src.next86_scigen_endpoint_router import (
        ENDPOINT_NAME,
        MANIFEST_NAME as ENDPOINT_MANIFEST_NAME,
        PROTOCOL as ENDPOINT_PROTOCOL,
    )
    from src.next86_scigen_term_catalogue import (
        CATALOGUE_NAME as TERM_CATALOGUE_NAME,
        MANIFEST_NAME as TERM_MANIFEST_NAME,
        PROTOCOL as TERM_PROTOCOL,
    )
    from src.next90_scigen_mechanism_envelope import (
        MANIFEST_NAME,
        PREDICTION_NAMES,
        run_scigen_mechanism_search,
    )

    signature = inspect.signature(run_scigen_mechanism_search)
    assert "validation_endpoint" not in signature.parameters
    assert "replication_endpoint" not in signature.parameters
    discovery, distortion, eligible_terms = _synthetic_mechanism_case("Q")
    feature_dir = tmp_path / "features"
    term_dir = tmp_path / "terms"
    endpoint_dir = tmp_path / "endpoint"
    feature_dir.mkdir()
    term_dir.mkdir()
    endpoint_dir.mkdir()

    feature_paths = {}
    for role in FEATURE_NAMES:
        table = discovery.copy()
        table["partition_role"] = role
        if role != "discovery":
            table["material_id"] = role + "-" + table["material_id"].astype(str)
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
                "eligible_terms": eligible_terms,
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
    endpoint_path = endpoint_dir / ENDPOINT_NAME
    pd.DataFrame(
        {
            "material_id": discovery["material_id"].astype(str),
            "partition_role": "discovery",
            "distortion_ratio": distortion,
        }
    ).to_parquet(endpoint_path, index=False)
    (endpoint_dir / ENDPOINT_MANIFEST_NAME).write_bytes(
        _json_bytes(
            {
                "protocol": ENDPOINT_PROTOCOL,
                "partition_role": "discovery",
                "lockbox_state": "discovery_available_after_routing",
                "outputs_sha256": {ENDPOINT_NAME: _sha256(endpoint_path)},
            }
        )
    )
    design = tmp_path / "next90-design.md"
    design.write_text("frozen NEXT90 design\n")
    output_dir = tmp_path / "next90"

    manifest = run_scigen_mechanism_search(
        feature_dir=feature_dir,
        term_catalogue_dir=term_dir,
        discovery_endpoint_dir=endpoint_dir,
        design_path=design,
        output_dir=output_dir,
        require_formal_inputs=False,
    )

    assert manifest["passes_discovery_gates"] is True
    assert manifest["validation_endpoint_opened"] is False
    assert manifest["replication_endpoint_opened"] is False
    for role in ("internal_validation", "internal_replication"):
        prediction = pd.read_parquet(output_dir / PREDICTION_NAMES[role])
        assert set(prediction["partition_role"]) == {role}
        assert not {"distortion_ratio", "protected", "severe"} & set(prediction.columns)
    assert (output_dir / MANIFEST_NAME).is_file()
