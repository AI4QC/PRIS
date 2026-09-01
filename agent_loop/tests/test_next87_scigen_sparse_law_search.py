import hashlib
import inspect
import json
import math

import numpy as np
import pandas as pd
import pytest


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def test_apply_formula_uses_frozen_one_sided_hinges_and_fails_open() -> None:
    from src.next87_scigen_sparse_law_search import apply_scigen_formula

    features = pd.DataFrame(
        {
            "x": [0.0, 3.0, np.nan],
            "y": [-math.sinh(2.0), 0.0, 1.0],
        }
    )
    formula = {
        "kind": "nonnegative_hinge_sum",
        "missing_policy": "KEEP",
        "terms": [
            {
                "term_id": "x__high",
                "feature": "x",
                "direction": 1,
                "transform": "log1p_nonnegative",
                "center": 0.0,
                "scale": math.log(4.0),
                "weight": 1.0,
            },
            {
                "term_id": "y__low",
                "feature": "y",
                "direction": -1,
                "transform": "asinh",
                "center": 0.0,
                "scale": 2.0,
                "weight": 0.5,
            },
        ],
        "threshold": 0.75,
    }

    score, supported, reject = apply_scigen_formula(features, formula)

    np.testing.assert_allclose(score[:2], [0.5, 1.0])
    assert np.isnan(score[2])
    assert supported.tolist() == [True, True, False]
    assert reject.tolist() == [False, True, False]

    invalid = {**formula, "terms": formula["terms"] * 2}
    with pytest.raises(ValueError, match="one to three"):
        apply_scigen_formula(features, invalid)


def test_group_folds_are_deterministic_order_invariant_and_group_isolated() -> None:
    from src.next87_scigen_sparse_law_search import assign_group_folds

    groups = np.asarray(["NaCl", "SiO2", "NaCl", "Al2O3", "SiO2", "MgO"])
    folds = assign_group_folds(groups)
    reordered = np.asarray(["MgO", "NaCl", "Al2O3", "SiO2", "NaCl", "SiO2"])
    reordered_folds = assign_group_folds(reordered)

    assert folds.shape == groups.shape
    assert set(folds).issubset(set(range(5)))
    mapping = {group: int(fold) for group, fold in zip(groups, folds, strict=True)}
    reordered_mapping = {
        group: int(fold)
        for group, fold in zip(reordered, reordered_folds, strict=True)
    }
    assert mapping == reordered_mapping


def test_decision_metrics_exclude_middle_rows_from_precision() -> None:
    from src.next87_scigen_sparse_law_search import decision_metrics

    metrics = decision_metrics(
        supported=np.ones(5, dtype=bool),
        reject=np.asarray([False, True, True, True, False]),
        distortion_ratio=np.asarray([0.5, 0.8, 1.5, 2.0, 3.0]),
    )

    assert metrics["rows"] == 5
    assert metrics["protected"] == 2
    assert metrics["protected_kept"] == 1
    assert metrics["protected_recall"] == pytest.approx(0.5)
    assert metrics["rejected"] == 3
    assert metrics["rejected_extremes"] == 2
    assert metrics["severe_rejected"] == 1
    assert metrics["severe_rejection_precision"] == pytest.approx(0.5)
    assert metrics["savings"] == pytest.approx(0.6)
    assert 0.0 < metrics["coverage_lower"] < 1.0


def test_threshold_selection_maximizes_extreme_precision_not_middle_rejection() -> None:
    from src.next87_scigen_sparse_law_search import select_threshold

    score = np.asarray([0.0, 0.1, 0.2, 0.3, 10.0, 1.0, 2.0, 3.0, 4.0])
    endpoint = np.asarray([0.5, 0.5, 0.5, 0.5, 1.5, 2.0, 2.0, 2.0, 2.0])
    selected = select_threshold(
        score=score,
        supported=np.ones(len(score), dtype=bool),
        distortion_ratio=endpoint,
        gates={
            "coverage_lower": 0.0,
            "protected_recall_lower": 0.0,
            "severe_rejection_precision_lower": 0.0,
            "savings_lower": 0.0,
        },
    )

    assert selected is not None
    assert selected["threshold"] == pytest.approx(1.0)
    assert selected["metrics"]["severe_rejected"] == 4
    assert selected["metrics"]["rejected"] == 5
    assert selected["metrics"]["rejected_extremes"] == 4
    assert selected["metrics"]["severe_rejection_precision"] == 1.0


def test_auc_diagnostics_report_pooled_macro_worst_and_evaluable_lattices() -> None:
    from src.next87_scigen_sparse_law_search import auc_diagnostics

    lattice = np.repeat([f"L{index}" for index in range(8)], 2)
    endpoint = np.tile([0.5, 2.0], 8)
    score = np.tile([0.0, 1.0], 8)

    diagnostics = auc_diagnostics(
        score=score,
        supported=np.ones(len(score), dtype=bool),
        distortion_ratio=endpoint,
        lattice_class=lattice,
    )

    assert diagnostics["pooled_extreme_auc"] == 1.0
    assert diagnostics["macro_lattice_auc"] == 1.0
    assert diagnostics["worst_lattice_auc"] == 1.0
    assert diagnostics["evaluable_lattices"] == 8
    assert len(diagnostics["lattices"]) == 8


def test_sparse_search_requires_stable_group_folds_and_beats_pauling() -> None:
    from src.next87_scigen_sparse_law_search import (
        assign_group_folds,
        search_scigen_sparse_law,
    )

    groups_by_fold: dict[int, list[str]] = {fold: [] for fold in range(5)}
    cursor = 0
    while any(len(groups) < 25 for groups in groups_by_fold.values()):
        group = f"X{cursor}Y"
        fold = int(assign_group_folds([group])[0])
        if len(groups_by_fold[fold]) < 25:
            groups_by_fold[fold].append(group)
        cursor += 1

    rows = []
    endpoints = []
    for group_index, group in enumerate(
        group for fold in range(5) for group in groups_by_fold[fold]
    ):
        lattice = f"L{group_index % 8}"
        rows.extend(
            [
                {
                    "material_id": f"{group}-protected",
                    "reduced_formula": group,
                    "lattice_class": lattice,
                    "signal": 0.0,
                    "aaa_noise": 0.0,
                    "pauling_p2_p5_decision": "ABSTAIN",
                },
                {
                    "material_id": f"{group}-severe",
                    "reduced_formula": group,
                    "lattice_class": lattice,
                    "signal": 3.0,
                    "aaa_noise": 0.0,
                    "pauling_p2_p5_decision": (
                        "REJECT" if group_index == 0 else "ABSTAIN"
                    ),
                },
            ]
        )
        endpoints.extend([0.5, 2.5])
    features = pd.DataFrame(rows)
    eligible_terms = [
        {
            "term_id": "signal__high",
            "feature": "signal",
            "group": "synthetic_mechanism",
            "direction": 1,
            "transform": "log1p_nonnegative",
            "center": 0.0,
            "scale": 1.0,
        },
        {
            "term_id": "aaa_noise__high",
            "feature": "aaa_noise",
            "group": "synthetic_noise",
            "direction": 1,
            "transform": "log1p_nonnegative",
            "center": 0.0,
            "scale": 1.0,
        },
    ]

    result = search_scigen_sparse_law(
        features=features,
        distortion_ratio=np.asarray(endpoints),
        eligible_terms=eligible_terms,
    )

    assert result["candidate_count"] == 7
    assert result["passes_discovery_gates"] is True
    assert [term["term_id"] for term in result["selected_formula"]["terms"]] == [
        "signal__high"
    ]
    assert result["fold_stability"]["selected_term_list_win_count"] == 5
    assert all(fold["passes_raw_fold_gates"] for fold in result["fold_diagnostics"])
    assert result["discovery_metrics"]["pooled_extreme_auc"] == 1.0
    assert result["discovery_metrics"]["evaluable_lattices"] == 8
    assert result["discovery_metrics"]["severe_rejected"] == 125
    assert result["pauling_baseline"]["severe_rejected"] == 1
    assert result["single_shortlist_term_ids"][0] == "signal__high"
    assert result["triple_shortlist_term_ids"][0] == "signal__high"


def test_runner_has_no_locked_endpoint_argument_and_freezes_label_free_predictions(
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
    from src.next87_scigen_sparse_law_search import (
        MANIFEST_NAME,
        PREDICTION_NAMES,
        run_scigen_sparse_search,
    )

    assert "validation_endpoint" not in inspect.signature(run_scigen_sparse_search).parameters
    assert "replication_endpoint" not in inspect.signature(run_scigen_sparse_search).parameters

    groups_by_fold: dict[int, list[str]] = {fold: [] for fold in range(5)}
    from src.next87_scigen_sparse_law_search import assign_group_folds

    cursor = 0
    while any(len(groups) < 25 for groups in groups_by_fold.values()):
        group = f"R{cursor}S"
        fold = int(assign_group_folds([group])[0])
        if len(groups_by_fold[fold]) < 25:
            groups_by_fold[fold].append(group)
        cursor += 1
    rows = []
    endpoint_rows = []
    for group_index, group in enumerate(
        group for fold in range(5) for group in groups_by_fold[fold]
    ):
        lattice = f"L{group_index % 8}"
        for state, signal, distortion in (("protected", 0.0, 0.5), ("severe", 3.0, 2.5)):
            material_id = f"{group}-{state}"
            rows.append(
                {
                    "material_id": material_id,
                    "reduced_formula": group,
                    "lattice_class": lattice,
                    "partition_role": "discovery",
                    "signal": signal,
                    "pauling_p2_p5_decision": (
                        "REJECT" if group_index == 0 and state == "severe" else "ABSTAIN"
                    ),
                }
            )
            endpoint_rows.append(
                {
                    "material_id": material_id,
                    "lattice_class": lattice,
                    "partition_role": "discovery",
                    "distortion_ratio": distortion,
                }
            )
    discovery = pd.DataFrame(rows)
    feature_dir = tmp_path / "features"
    term_dir = tmp_path / "terms"
    endpoint_dir = tmp_path / "discovery_endpoint"
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
    feature_catalogue = feature_dir / FEATURE_CATALOGUE_NAME
    feature_catalogue.write_bytes(_json_bytes({"protocol": FEATURE_PROTOCOL, "labels_opened": False}))
    feature_manifest = {
        "protocol": FEATURE_PROTOCOL,
        "labels_opened": False,
        "endpoint_payloads_opened": False,
        "relaxed_structures_opened": False,
        "outputs_sha256": {
            FEATURE_CATALOGUE_NAME: _sha256(feature_catalogue),
            **{FEATURE_NAMES[role]: _sha256(path) for role, path in feature_paths.items()},
        },
    }
    (feature_dir / FEATURE_MANIFEST_NAME).write_bytes(_json_bytes(feature_manifest))

    term_catalogue = {
        "protocol": TERM_PROTOCOL,
        "labels_opened": False,
        "eligible_terms": [
            {
                "term_id": "signal__high",
                "feature": "signal",
                "group": "synthetic_mechanism",
                "direction": 1,
                "transform": "log1p_nonnegative",
                "center": 0.0,
                "scale": 1.0,
            }
        ],
    }
    term_catalogue_path = term_dir / TERM_CATALOGUE_NAME
    term_catalogue_path.write_bytes(_json_bytes(term_catalogue))
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
    pd.DataFrame(endpoint_rows).to_parquet(endpoint_path, index=False)
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
    design = tmp_path / "design.md"
    implementation = tmp_path / "implementation.md"
    design.write_text("frozen design\n")
    implementation.write_text("post-discovery implementation protocol\n")
    output_dir = tmp_path / "next87"

    manifest = run_scigen_sparse_search(
        feature_dir=feature_dir,
        term_catalogue_dir=term_dir,
        discovery_endpoint_dir=endpoint_dir,
        design_path=design,
        implementation_path=implementation,
        output_dir=output_dir,
        require_formal_inputs=False,
    )

    assert manifest["passes_discovery_gates"] is True
    assert manifest["validation_endpoint_opened"] is False
    assert manifest["replication_endpoint_opened"] is False
    for role in ("internal_validation", "internal_replication"):
        prediction = pd.read_parquet(output_dir / PREDICTION_NAMES[role])
        assert set(prediction["partition_role"]) == {role}
        assert "distortion_ratio" not in prediction
        assert "protected" not in prediction
        assert "severe" not in prediction
    assert (output_dir / MANIFEST_NAME).is_file()
