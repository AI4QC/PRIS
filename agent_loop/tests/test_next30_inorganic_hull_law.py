from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.next30_inorganic_hull_law import (
    FORMULAS,
    SALT,
    TERM_SPECS,
    TermSpec,
    _score_formula,
    _validated_feature_frame,
    decision_metrics,
    deterministic_split,
    discover_rule,
    fit_term_parameters,
    freeze_split,
    merge_feature_sources,
    run_development,
    scan_development,
)


def test_deterministic_split_is_disjoint_complete_and_order_invariant() -> None:
    ids = [f"wbm-{index:05d}" for index in range(12)]
    first = deterministic_split(ids, development_size=5, salt=SALT)
    second = deterministic_split(list(reversed(ids)), development_size=5, salt=SALT)

    pd.testing.assert_frame_equal(first, second)
    assert first["material_id"].is_unique
    assert set(first["material_id"]) == set(ids)
    assert first["partition"].value_counts().to_dict() == {
        "confirmation": 7,
        "development": 5,
    }
    development = set(first.loc[first["partition"].eq("development"), "material_id"])
    confirmation = set(first.loc[first["partition"].eq("confirmation"), "material_id"])
    assert development.isdisjoint(confirmation)


def test_feature_validation_rejects_endpoint_like_columns() -> None:
    frame = pd.DataFrame(
        {
            "material_id": ["a", "b"],
            "safe_geometry_term": [0.1, 0.2],
            "e_above_hull": [0.0, 0.4],
        }
    )

    with pytest.raises(ValueError, match="no-DFT contract"):
        _validated_feature_frame(frame, expected_ids=["a", "b"], role="synthetic")


def test_term_parameters_are_fit_only_from_the_passed_development_rows() -> None:
    development = pd.DataFrame({"risk": [0.0, 1.0, 2.0, 3.0]})
    spec = TermSpec(source="synthetic", column="risk", direction=1)

    parameters = fit_term_parameters(development, {"risk": spec})

    assert parameters["risk"]["median"] == pytest.approx(1.5)
    assert parameters["risk"]["scale_iqr"] == pytest.approx(1.5)


def test_zero_iqr_term_disables_only_formulas_that_require_it() -> None:
    frame = pd.DataFrame(
        {
            "flat": [0.0] * 100,
            "risk": np.arange(100, dtype=float),
        }
    )
    specs = {
        "flat": TermSpec(source="synthetic", column="flat", direction=1),
        "risk": TermSpec(source="synthetic", column="risk", direction=1),
    }
    parameters = fit_term_parameters(frame, specs)

    assert parameters["flat"]["available"] is False
    assert parameters["risk"]["available"] is True
    flat_score, flat_supported = _score_formula(
        frame,
        formula=("flat",),
        term_specs=specs,
        parameters=parameters,
    )
    risk_score, risk_supported = _score_formula(
        frame,
        formula=("risk",),
        term_specs=specs,
        parameters=parameters,
    )
    assert not flat_supported.any()
    assert np.isnan(flat_score).all()
    assert risk_supported.all()
    assert np.isfinite(risk_score).all()


def test_formula_score_fails_open_if_any_required_term_is_missing() -> None:
    frame = pd.DataFrame(
        {
            "a": [0.0, 1.0, np.nan],
            "b": [0.0, np.nan, 2.0],
        }
    )
    terms = {
        "a": TermSpec(source="synthetic", column="a", direction=1),
        "b": TermSpec(source="synthetic", column="b", direction=1),
    }
    parameters = {
        "a": {"median": 0.0, "scale_iqr": 1.0},
        "b": {"median": 0.0, "scale_iqr": 1.0},
    }

    score, supported = _score_formula(
        frame,
        formula=("a", "b"),
        term_specs=terms,
        parameters=parameters,
    )

    assert supported.tolist() == [True, False, False]
    assert score[0] == pytest.approx(0.0)
    assert np.isnan(score[1:]).all()


def test_decision_metrics_treat_unsupported_rows_as_fail_open() -> None:
    energy = np.array([0.0, 0.04, 0.3, 0.4, 0.1])
    supported = np.array([True, False, True, True, True])
    reject = np.array([False, False, True, False, False])

    metrics = decision_metrics(supported=supported, reject=reject, energy=energy)

    assert metrics["coverage"]["numerator"] == 4
    assert metrics["valuable_recall"]["numerator"] == 2
    assert metrics["valuable_recall"]["denominator"] == 2
    assert metrics["reject_precision_high_energy"]["numerator"] == 1
    assert metrics["reject_precision_high_energy"]["denominator"] == 1
    assert metrics["dft_savings"]["numerator"] == 1
    assert metrics["dft_savings"]["denominator"] == 5


def test_scan_selects_largest_eligible_savings_from_frozen_grid() -> None:
    frame = pd.DataFrame({"risk": np.arange(400, dtype=float)})
    energy = np.where(frame["risk"].to_numpy() < 100, 0.0, 0.3)
    specs = {"risk": TermSpec(source="synthetic", column="risk", direction=1)}
    parameters = fit_term_parameters(frame, specs)

    result = scan_development(
        frame,
        energy=energy,
        term_specs=specs,
        formulas=(("risk",),),
        rejection_fractions=(0.10, 0.20),
        parameters=parameters,
    )

    assert result["promotion"] is True
    assert result["selected"]["formula"] == ["risk"]
    assert result["selected"]["rejection_fraction"] == pytest.approx(0.20)
    assert result["selected"]["metrics"]["dft_savings"]["estimate"] >= 0.20


def test_discovery_is_invariant_to_confirmation_label_changes() -> None:
    ids = [f"id-{index:03d}" for index in range(400)]
    split = deterministic_split(ids, development_size=200, salt="synthetic")
    features = split[["material_id"]].copy()
    features["risk"] = np.arange(len(features), dtype=float)
    development_ids = set(
        split.loc[split["partition"].eq("development"), "material_id"]
    )
    labels = pd.DataFrame({"material_id": ids})
    labels["hull"] = labels["material_id"].map(
        lambda value: 0.3 if value in development_ids else 0.0
    )
    changed = labels.copy()
    changed.loc[~changed["material_id"].isin(development_ids), "hull"] = 99.0
    specs = {"risk": TermSpec(source="synthetic", column="risk", direction=1)}

    first = discover_rule(
        features=features,
        labels=labels,
        split=split,
        label_column="hull",
        term_specs=specs,
        formulas=(("risk",),),
        rejection_fractions=(0.10,),
    )
    second = discover_rule(
        features=features,
        labels=changed,
        split=split,
        label_column="hull",
        term_specs=specs,
        formulas=(("risk",),),
        rejection_fractions=(0.10,),
    )

    assert first == second


def test_frozen_catalogue_contains_only_sparse_known_terms() -> None:
    assert TERM_SPECS
    assert FORMULAS
    for formula in FORMULAS:
        assert 1 <= len(formula) <= 3
        assert len(set(formula)) == len(formula)
        assert set(formula).issubset(TERM_SPECS)


def test_merge_feature_sources_uses_exact_sources_and_one_to_one_ids() -> None:
    specs = {
        "a": TermSpec(source="left", column="left_value", direction=1),
        "b": TermSpec(source="right", column="right_value", direction=-1),
    }
    metadata = pd.DataFrame({"material_id": ["b", "a"], "natoms": [2, 1]})
    sources = {
        "left": pd.DataFrame(
            {"material_id": ["a", "b"], "left_value": [1.0, 2.0]}
        ),
        "right": pd.DataFrame(
            {"material_id": ["b", "a"], "right_value": [4.0, 3.0]}
        ),
    }

    merged = merge_feature_sources(
        metadata=metadata, source_frames=sources, term_specs=specs
    )

    assert merged["material_id"].tolist() == ["a", "b"]
    assert merged["left_value"].tolist() == [1.0, 2.0]
    assert merged["right_value"].tolist() == [3.0, 4.0]
    with pytest.raises(ValueError, match="exactly"):
        merge_feature_sources(
            metadata=metadata,
            source_frames={"left": sources["left"]},
            term_specs=specs,
        )


def test_freeze_split_publishes_safe_manifest_and_refuses_overwrite(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.parquet"
    pd.DataFrame(
        {
            "material_id": ["d", "c", "b", "a"],
            "natoms": [1, 1, 1, 1],
            "input_role": ["unrelaxed_x0_geometry_only"] * 4,
        }
    ).to_parquet(metadata_path, index=False)
    output_dir = tmp_path / "split"

    manifest = freeze_split(
        metadata_path=metadata_path,
        output_dir=output_dir,
        development_size=2,
        expected_rows=4,
        salt="test-salt",
    )

    assert manifest["labels_opened"] is False
    assert manifest["endpoint_artifacts_opened"] is False
    assert manifest["counts"] == {"confirmation": 2, "development": 2, "rows": 4}
    saved = pd.read_parquet(output_dir / "next30_split.parquet")
    assert saved["material_id"].is_unique
    with pytest.raises(FileExistsError):
        freeze_split(
            metadata_path=metadata_path,
            output_dir=output_dir,
            development_size=2,
            expected_rows=4,
            salt="test-salt",
        )


def test_run_development_no_promotion_does_not_publish_rule(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.parquet"
    metadata = pd.DataFrame(
        {
            "material_id": [f"id-{index}" for index in range(40)],
            "natoms": [2] * 40,
            "input_role": ["unrelaxed_x0_geometry_only"] * 40,
        }
    )
    metadata.to_parquet(metadata_path, index=False)
    split_dir = tmp_path / "split"
    freeze_split(
        metadata_path=metadata_path,
        output_dir=split_dir,
        development_size=20,
        expected_rows=40,
        salt="development-test",
    )
    features_path = tmp_path / "features.parquet"
    pd.DataFrame(
        {
            "material_id": metadata["material_id"],
            "risk": np.arange(40, dtype=float),
        }
    ).to_parquet(features_path, index=False)
    labels_path = tmp_path / "labels.parquet"
    pd.DataFrame(
        {"material_id": metadata["material_id"], "hull_value": [0.0] * 40}
    ).to_parquet(labels_path, index=False)
    specs = {"risk": TermSpec(source="synthetic", column="risk", direction=1)}
    output_dir = tmp_path / "development"

    manifest = run_development(
        metadata_path=metadata_path,
        split_path=split_dir / "next30_split.parquet",
        split_manifest_path=split_dir / "MANIFEST.json",
        feature_paths={"synthetic": features_path},
        labels_path=labels_path,
        output_dir=output_dir,
        label_column="hull_value",
        term_specs=specs,
        formulas=(("risk",),),
        rejection_fractions=(0.10,),
    )

    assert manifest["promotion"] is False
    assert manifest["confirmation_labels_used_for_selection"] is False
    assert (output_dir / "NEXT30_DEVELOPMENT_SCAN.json").is_file()
    assert not (output_dir / "NEXT30_FROZEN_HULL_RULE.json").exists()
