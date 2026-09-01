"""Tests for the finite, label-free MatterSim committee score catalog."""

from __future__ import annotations

import gc
import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path
import weakref

import numpy as np
import pandas as pd
import pytest

from src import next8_mattersim_committee_protocol as protocol
from src import next6_mattersim_baseline as baseline_module
from src import next6_wbm_features as wbm_features_module
from src import next8_mattersim_committee_features as feature_module


def _ready_row(
    sid: str,
    rk: str,
    stage: str,
    m1: float,
    m5: float,
    *,
    m1_fmax: float = 0.0,
    m1_frms: float = 0.0,
    m5_fmax: float = 0.0,
    m5_frms: float = 0.0,
) -> dict[str, object]:
    return {
        "sid": sid,
        "rk": rk,
        "stage": stage,
        "committee_feature_ok": True,
        "m1_prediction_ok": True,
        "m5_prediction_ok": True,
        "m1_energy_ev_per_atom": m1,
        "m5_energy_ev_per_atom": m5,
        "m1_fmax_ev_per_a": m1_fmax,
        "m1_frms_ev_per_a": m1_frms,
        "m5_fmax_ev_per_a": m5_fmax,
        "m5_frms_ev_per_a": m5_frms,
    }


def _search_calibration_features() -> pd.DataFrame:
    # The checkpoints have very different absolute offsets, but q99/q99.5 must
    # use disagreement between same-rk gaps, not raw-energy disagreement.
    features = pd.DataFrame(
        [
            _ready_row("cal-a0", "A", "search_calibration", -100.0, 100.0),
            _ready_row("cal-a1", "A", "search_calibration", -99.0, 101.0),
            _ready_row("cal-b0", "B", "search_calibration", 50.0, -50.0),
            _ready_row("cal-b1", "B", "search_calibration", 54.0, -49.0),
        ]
    )
    features["m5_fmax_ev_per_a"] = [0.0, 0.0, 0.0, 3.0]
    return features


def _formula_selection_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _ready_row("sel-a0", "A", "formula_selection", -10.0, -20.0),
            _ready_row("sel-a1", "A", "formula_selection", -8.0, -19.0),
            _ready_row("sel-b0", "B", "formula_selection", 7.0, 3.0),
            _ready_row("sel-b1", "B", "formula_selection", 7.5, 5.0),
        ]
    )


def _scores_by_formula_and_sid(table: pd.DataFrame) -> pd.DataFrame:
    return table.set_index(["formula", "sid"], drop=False)


def _development_features(*, threshold_groups: int = 6) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stage, prefix, n_groups in (
        ("search_calibration", "search", 2),
        ("formula_selection", "select", 2),
        ("threshold_calibration", "threshold", threshold_groups),
    ):
        for group_index in range(n_groups):
            rk = f"{prefix}-rk-{group_index:03d}"
            rows.extend(
                (
                    _ready_row(
                        f"{prefix}-{group_index:03d}-0",
                        rk,
                        stage,
                        0.0,
                        0.0,
                    ),
                    _ready_row(
                        f"{prefix}-{group_index:03d}-1",
                        rk,
                        stage,
                        0.2 + group_index / 1000.0,
                        0.3 + group_index / 1000.0,
                    ),
                )
            )
    return pd.DataFrame(rows)


def _endpoint_labels(features: pd.DataFrame) -> pd.DataFrame:
    labels = features[["sid", "rk", "stage"]].copy()
    labels["e_per_atom"] = [float(index % 2) * 0.25 for index in range(len(labels))]
    return labels


def _many_group_features(stage: str, n_groups: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_index in range(n_groups):
        rk = f"{stage}-many-{group_index:03d}"
        rows.extend(
            (
                _ready_row(
                    f"{stage}-many-{group_index:03d}-0",
                    rk,
                    stage,
                    0.0,
                    0.0,
                ),
                _ready_row(
                    f"{stage}-many-{group_index:03d}-1",
                    rk,
                    stage,
                    0.2,
                    0.3,
                ),
            )
        )
    return pd.DataFrame(rows)


def _with_m1_group_failures(features: pd.DataFrame, *, n_groups: int) -> pd.DataFrame:
    failed = features.copy()
    groups = failed["rk"].drop_duplicates().iloc[:n_groups]
    failed_indices = (
        failed.loc[failed["rk"].isin(groups)].groupby("rk", sort=False).tail(1).index
    )
    failed.loc[failed_indices, "m1_prediction_ok"] = False
    failed.loc[failed_indices, "committee_feature_ok"] = False
    failed.loc[failed_indices, "m1_energy_ev_per_atom"] = np.nan
    failed.loc[
        failed_indices,
        ["m1_fmax_ev_per_a", "m1_frms_ev_per_a"],
    ] = np.nan
    return failed


def _joint_ecdf_calibration_features(n_rows: int = 200) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _ready_row(
                f"joint-cal-{index:03d}",
                "JOINT-CAL",
                "search_calibration",
                float(index),
                0.0,
                m1_fmax=float(index),
                m1_frms=float(index),
            )
            for index in range(n_rows)
        ]
    )


def _frontier_row(
    formula: str,
    track: str,
    *,
    savings: float,
    safe: bool = True,
    comparator_savings: float | None = None,
) -> dict[str, object]:
    del comparator_savings
    retained = 0.96 if safe else 0.94
    return {
        "formula": formula,
        "track": track,
        "dft_savings": savings,
        "exact_min_retention_lower": retained,
        "near_min_retention_lower": retained,
        "valuable_group_retention_lower": retained,
        "regret_p95": 0.01,
        "all_rejected_groups": 0,
        # Selection must ignore any caller-supplied fake tie-break values.
        "cost": -999,
        "complexity": -999,
    }


def _complete_frontier(
    *,
    primary_savings: dict[str, float] | None = None,
    unsafe: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    primary_savings = primary_savings or {}
    unsafe = unsafe or set()
    rows = []
    for formula in protocol.FORMULA_NAMES:
        for track in ("primary", "comparator"):
            rows.append(
                _frontier_row(
                    formula,
                    track,
                    savings=(
                        primary_savings.get(formula, 0.0)
                        if track == "primary"
                        else 0.99 - protocol.FORMULA_NAMES.index(formula) / 100.0
                    ),
                    safe=(formula, track) not in unsafe,
                )
            )
    return pd.DataFrame(rows)


def _final_rules(
    selected_formula: str,
    *,
    selected_threshold: float,
    baseline_threshold: float,
) -> pd.DataFrame:
    rows = []
    for track in ("primary", "comparator"):
        alpha = 0.01 if track == "primary" else 0.035
        within_group = "max" if track == "primary" else "min"
        for role, formula, threshold in (
            ("selected", selected_formula, selected_threshold),
            ("m5_baseline", "M5", baseline_threshold),
        ):
            rows.append(
                {
                    "role": role,
                    "formula": formula,
                    "track": track,
                    "threshold": threshold,
                    "threshold_state": (
                        "finite" if np.isfinite(threshold) else "keep_all"
                    ),
                    "threshold_source_role": "threshold_fit",
                    "alpha": alpha,
                    "within_group": within_group,
                    "operator": "score > threshold",
                    "unsupported_decision": "ABSTAIN",
                    "calibration_n_groups": 100,
                    "calibration_order_index": 100,
                }
            )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_freeze_inputs(tmp_path: Path, monkeypatch, features=None):
    table = _development_features() if features is None else features.copy()
    features_path = tmp_path / feature_module.OUTPUT_NAME
    labels_path = tmp_path / "development_labels.parquet"
    feature_manifest_path = tmp_path / feature_module.MANIFEST_NAME
    output_dir = tmp_path / "freeze"
    table.to_parquet(features_path, index=False)
    _endpoint_labels(table).to_parquet(labels_path, index=False)

    checkpoints = {
        "m1": tmp_path / "m1.pth",
        "m5": tmp_path / "m5.pth",
    }
    checkpoints["m1"].write_bytes(b"toy-m1-checkpoint")
    checkpoints["m5"].write_bytes(b"toy-m5-checkpoint")
    checkpoint_sha256 = {model: _sha256(path) for model, path in checkpoints.items()}
    monkeypatch.setattr(protocol, "FROZEN_CHECKPOINT_SHA256", checkpoint_sha256)

    repository_root = Path(protocol.__file__).resolve().parents[1]
    feature_sources = (
        Path(feature_module.__file__).resolve(),
        Path(baseline_module.__file__).resolve(),
        Path(wbm_features_module.__file__).resolve(),
    )
    implementation_source = feature_sources[0]
    manifest = {
        "protocol": feature_module.PROTOCOL,
        "mode": "development",
        "stages": list(feature_module.DEVELOPMENT_STAGES),
        "adapter": {
            "mode": "builtin_mattersim",
            "batch_size": 1,
            "device_requested": "cpu",
            "device_resolved": "cpu",
            "implementation": {
                "module": feature_module.MatterSimCommitteePredictor.__module__,
                "qualname": (feature_module.MatterSimCommitteePredictor.__qualname__),
                "source_path": str(implementation_source),
                "source_sha256": _sha256(implementation_source),
                "source_hash_verified": True,
            },
        },
        "production_protocol_eligible": True,
        "evidence_role": "protocol_feature_generation",
        "runtime": {"python_version": "toy"},
        "checkpoints": {
            model: {"path": str(path.resolve()), "sha256": checkpoint_sha256[model]}
            for model, path in checkpoints.items()
        },
        "predictor_loaded_checkpoint_sha256": checkpoint_sha256,
        "inputs_sha256": {
            role: {"path": str(tmp_path / role), "sha256": "1" * 64}
            for role in ("frames", "metadata", "stage_assignments")
        },
        "executed_source_sha256": {
            path.relative_to(repository_root).as_posix(): _sha256(path)
            for path in feature_sources
        },
        "integrity": {"prepublish_rehash": "passed"},
        "outputs_sha256": {features_path.name: _sha256(features_path)},
    }
    feature_manifest_path.write_text(
        json.dumps(manifest, allow_nan=False), encoding="utf-8"
    )
    return {
        "features": features_path,
        "labels": labels_path,
        "feature_manifest": feature_manifest_path,
        "output": output_dir,
        "checkpoints": checkpoints,
    }


def test_cutoff_origin_registry_releases_unreferenced_calibrations():
    initial_size = len(protocol._ORIGIN_REGISTRY)
    cutoffs = protocol.derive_disagreement_cutoffs(_search_calibration_features())
    token_reference = weakref.ref(cutoffs._origin_token)

    assert len(protocol._ORIGIN_REGISTRY) == initial_size + 1

    del cutoffs
    gc.collect()

    assert token_reference() is None
    assert len(protocol._ORIGIN_REGISTRY) == initial_size


def test_threshold_split_is_hash_exact_disjoint_complete_and_group_preserving():
    threshold = _development_features(threshold_groups=5)
    threshold = threshold.loc[
        threshold["stage"].eq("threshold_calibration")
    ].reset_index(drop=True)

    assignment = protocol.split_threshold_groups(threshold)
    reordered = protocol.split_threshold_groups(
        threshold.iloc[::-1].reset_index(drop=True)
    )

    salt = "next8-threshold-fit-gate-v1-20260801"
    groups = sorted(
        threshold["rk"].unique(),
        key=lambda rk: (hashlib.sha256(f"{salt}\0{rk}".encode()).hexdigest(), rk),
    )
    expected_fit = set(groups[: len(groups) // 2])
    expected_gate = set(groups[len(groups) // 2 :])
    observed_fit = set(
        assignment.loc[assignment["threshold_role"].eq("threshold_fit"), "rk"]
    )
    observed_gate = set(
        assignment.loc[assignment["threshold_role"].eq("development_gate"), "rk"]
    )

    assert observed_fit == expected_fit
    assert observed_gate == expected_gate
    assert observed_fit.isdisjoint(observed_gate)
    assert observed_fit | observed_gate == set(groups)
    assert assignment.groupby("rk")["threshold_role"].nunique().eq(1).all()
    assert assignment.groupby("rk").size().eq(2).all()
    assert assignment["split_salt"].eq(salt).all()
    assert assignment["split_rank"].between(0, len(groups) - 1).all()
    assert assignment["split_key_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    pd.testing.assert_frame_equal(
        assignment.sort_values(["rk", "sid"]).reset_index(drop=True),
        reordered.sort_values(["rk", "sid"]).reset_index(drop=True),
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda table: table.assign(stage="formula_selection"),
            "threshold_calibration",
        ),
        (lambda table: table.assign(stage="test"), "threshold_calibration|test"),
        (
            lambda table: table.assign(
                stage=["threshold_calibration", "test"] * (len(table) // 2)
            ),
            "threshold_calibration|stage",
        ),
        (lambda table: table.loc[table["rk"].eq(table.iloc[0]["rk"])], "two"),
    ],
)
def test_threshold_split_rejects_wrong_stage_mixing_or_fewer_than_two_groups(
    mutation, match
):
    threshold = _development_features(threshold_groups=4)
    threshold = threshold.loc[
        threshold["stage"].eq("threshold_calibration")
    ].reset_index(drop=True)

    with pytest.raises(ValueError, match=match):
        protocol.split_threshold_groups(mutation(threshold))


def test_threshold_split_rejects_any_nonfrozen_salt():
    threshold = _development_features(threshold_groups=4)
    threshold = threshold.loc[
        threshold["stage"].eq("threshold_calibration")
    ].reset_index(drop=True)

    with pytest.raises(ValueError, match="frozen|salt"):
        protocol.split_threshold_groups(threshold, salt="post-hoc-salt")


def test_existing_output_fails_before_any_label_open(tmp_path, monkeypatch):
    paths = _write_freeze_inputs(tmp_path, monkeypatch)
    paths["output"].mkdir()
    real_read_parquet = pd.read_parquet

    def guarded_read_parquet(path, *args, **kwargs):
        if Path(path) == paths["labels"]:
            raise AssertionError("labels opened before output-exists failure")
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(protocol.pd, "read_parquet", guarded_read_parquet)
    with pytest.raises(FileExistsError, match="overwrite|existing"):
        protocol.run_development_freeze(
            paths["features"],
            paths["labels"],
            paths["feature_manifest"],
            paths["output"],
            checkpoints=paths["checkpoints"],
            n_resamples=10,
        )


@pytest.mark.parametrize(
    "failure",
    (
        "malformed",
        "test_stage",
        "mixed_stage",
        "missing_stage",
        "invalid_cutoff",
        "split",
        "hash",
    ),
)
def test_feature_boundary_failures_happen_before_label_open(
    tmp_path, monkeypatch, failure
):
    paths = _write_freeze_inputs(tmp_path, monkeypatch)
    table = pd.read_parquet(paths["features"])
    manifest = json.loads(paths["feature_manifest"].read_text())
    if failure == "malformed":
        paths["features"].write_bytes(b"not-a-parquet-snapshot")
    elif failure == "test_stage":
        table.loc[0, "stage"] = "test"
    elif failure == "mixed_stage":
        group = table.iloc[0]["rk"]
        group_indices = table.index[table["rk"].eq(group)].tolist()
        table.loc[group_indices[0], "stage"] = "formula_selection"
    elif failure == "missing_stage":
        table = table.loc[~table["stage"].eq("formula_selection")].copy()
    elif failure == "invalid_cutoff":
        search = table["stage"].eq("search_calibration")
        table.loc[
            search,
            [
                "committee_feature_ok",
                "m1_prediction_ok",
                "m5_prediction_ok",
            ],
        ] = False
        table.loc[
            search,
            [
                "m1_energy_ev_per_atom",
                "m5_energy_ev_per_atom",
                "m1_fmax_ev_per_a",
                "m1_frms_ev_per_a",
                "m5_fmax_ev_per_a",
                "m5_frms_ev_per_a",
            ],
        ] = np.nan
    elif failure == "split":
        threshold_groups = table.loc[
            table["stage"].eq("threshold_calibration"), "rk"
        ].drop_duplicates()
        table = table.loc[~table["rk"].isin(threshold_groups.iloc[1:])].copy()
    elif failure == "hash":
        paths["features"].write_bytes(paths["features"].read_bytes() + b"tamper")
    else:  # pragma: no cover - parametrization is frozen above
        raise AssertionError(failure)
    if failure != "hash":
        if failure != "malformed":
            table.to_parquet(paths["features"], index=False)
        manifest["outputs_sha256"][paths["features"].name] = _sha256(paths["features"])
        paths["feature_manifest"].write_text(
            json.dumps(manifest, allow_nan=False), encoding="utf-8"
        )

    label_bytes = paths["labels"].read_bytes()
    real_read_parquet = pd.read_parquet
    real_read_bytes = Path.read_bytes
    real_sha256_file = protocol._sha256_file

    def guarded_read_bytes(path):
        if Path(path) == paths["labels"]:
            raise AssertionError(f"labels snapshotted before {failure} failure")
        return real_read_bytes(path)

    def guarded_sha256_file(path):
        if Path(path) == paths["labels"]:
            raise AssertionError(f"labels hashed before {failure} failure")
        return real_sha256_file(path)

    def guarded_read_parquet(source, *args, **kwargs):
        if isinstance(source, (str, Path)) and Path(source) == paths["labels"]:
            raise AssertionError(f"labels opened before {failure} failure")
        if isinstance(source, io.BytesIO) and source.getvalue() == label_bytes:
            raise AssertionError(f"label snapshot parsed before {failure} failure")
        return real_read_parquet(source, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(protocol, "_sha256_file", guarded_sha256_file)
    monkeypatch.setattr(protocol.pd, "read_parquet", guarded_read_parquet)
    with pytest.raises(Exception) as captured:
        protocol.run_development_freeze(
            paths["features"],
            paths["labels"],
            paths["feature_manifest"],
            paths["output"],
            checkpoints=paths["checkpoints"],
            n_resamples=10,
        )
    assert not isinstance(captured.value, AssertionError)
    assert not paths["output"].exists()


@pytest.mark.parametrize(
    "track,protected_metric",
    (
        ("primary", "valuable_group_retention_lower"),
        ("comparator", "near_min_retention_lower"),
    ),
)
def test_safety_gate_uses_exact_track_specific_retention_regret_and_survivors(
    track, protected_metric
):
    metrics = {
        "exact_min_retention_lower": 0.95,
        "near_min_retention_lower": 0.95,
        "valuable_group_retention_lower": 0.95,
        "regret_p95": 0.05,
        "all_rejected_groups": 0,
    }
    assert protocol.passes_safety_gate(metrics, track)

    for field, unsafe_value in (
        ("exact_min_retention_lower", 0.949999),
        (protected_metric, 0.949999),
        ("regret_p95", 0.050001),
        ("all_rejected_groups", 1),
    ):
        unsafe = dict(metrics)
        unsafe[field] = unsafe_value
        assert not protocol.passes_safety_gate(unsafe, track)


def test_primary_selection_requires_same_formula_to_be_safe_on_both_tracks():
    frontier = _complete_frontier(
        primary_savings={"MAX": 0.90, "MEAN": 0.80},
        unsafe={("MAX", "comparator")},
    )

    selected = protocol.select_primary_formula(frontier)

    assert selected["state"] == "selected"
    assert selected["name"] == "MEAN"
    assert selected["primary_dft_savings"] == pytest.approx(0.80)
    assert selected["cost"] == 6
    assert selected["complexity"] == 2


@pytest.mark.parametrize(
    "savings,expected",
    (
        ({"M1": 0.7, "M5": 0.7}, "M1"),
        ({"MEAN": 0.7, "LCB": 0.7}, "MEAN"),
        ({"MIN": 0.7, "MEAN": 0.7, "MAX": 0.7}, "MIN"),
    ),
)
def test_primary_selection_uses_fixed_cost_complexity_then_catalog_order(
    savings, expected
):
    frontier = _complete_frontier(primary_savings=savings)

    selected = protocol.select_primary_formula(frontier)

    assert selected["name"] == expected


def test_comparator_savings_never_selects_a_separate_winner_and_no_safe_is_null():
    frontier = _complete_frontier(primary_savings={"M5": 0.8, "M1": 0.7})
    frontier.loc[
        (frontier["formula"].eq("M1")) & (frontier["track"].eq("comparator")),
        "dft_savings",
    ] = 1.0
    assert protocol.select_primary_formula(frontier)["name"] == "M5"

    unsafe = _complete_frontier(
        unsafe={(formula, "comparator") for formula in protocol.FORMULA_NAMES}
    )
    assert protocol.select_primary_formula(unsafe) == {
        "state": "null_keep_all",
        "name": "null_keep_all",
    }


def test_provisional_thresholds_are_fit_for_all_eleven_two_tracks_from_search_only():
    search_features = _many_group_features("search_calibration", 100)
    cutoffs = protocol.derive_disagreement_cutoffs(search_features)
    labels = _endpoint_labels(search_features)

    original = protocol.fit_provisional_thresholds(
        search_features, labels, cutoffs=cutoffs
    )
    reversed_labels = labels.copy()
    reversed_labels["e_per_atom"] = np.tile([0.25, 0.0], 100)
    shifted = protocol.fit_provisional_thresholds(
        search_features, reversed_labels, cutoffs=cutoffs
    )

    assert len(original) == 2 * len(protocol.FORMULA_NAMES)
    assert set(original["formula"]) == set(protocol.FORMULA_NAMES)
    assert set(original["track"]) == {"primary", "comparator"}
    assert set(original["threshold_source_stage"]) == {"search_calibration"}
    assert set(original["operator"]) == {"score > threshold"}
    assert original.loc[
        original["formula"].eq("M5") & original["track"].eq("primary"),
        "threshold",
    ].item() == pytest.approx(0.0)
    assert shifted.loc[
        shifted["formula"].eq("M5") & shifted["track"].eq("primary"),
        "threshold",
    ].item() == pytest.approx(0.3)

    wrong_stage_labels = labels.assign(stage="formula_selection")
    with pytest.raises(ValueError, match="search_calibration|stage"):
        protocol.fit_provisional_thresholds(
            search_features, wrong_stage_labels, cutoffs=cutoffs
        )

    different_search = search_features.copy()
    different_search["m1_energy_ev_per_atom"] += 7.0
    wrong_origin = protocol.derive_disagreement_cutoffs(different_search)
    with pytest.raises(ValueError, match="fingerprint|same search_calibration"):
        protocol.fit_provisional_thresholds(
            search_features, labels, cutoffs=wrong_origin
        )


def test_formula_selection_frontier_only_applies_frozen_provisional_rules():
    search_features = _many_group_features("search_calibration", 100)
    selection_features = _many_group_features("formula_selection", 60)
    cutoffs = protocol.derive_disagreement_cutoffs(search_features)
    provisional = protocol.fit_provisional_thresholds(
        search_features,
        _endpoint_labels(search_features),
        cutoffs=cutoffs,
    )

    frontier = protocol.evaluate_formula_selection(
        selection_features,
        _endpoint_labels(selection_features),
        cutoffs=cutoffs,
        provisional_thresholds=provisional,
    )

    assert len(frontier) == 2 * len(protocol.FORMULA_NAMES)
    assert set(frontier["threshold_source_stage"]) == {"search_calibration"}
    assert set(frontier["evaluation_stage"]) == {"formula_selection"}
    assert frontier["search_calibration_n_groups"].eq(100).all()
    assert frontier["search_calibration_order_index"].gt(0).all()
    assert set(frontier["formula"]) == set(protocol.FORMULA_NAMES)
    assert set(frontier["track"]) == {"primary", "comparator"}
    assert (
        frontier.loc[frontier["formula"].eq("M5"), "n"]
        .eq(len(selection_features))
        .all()
    )

    tampered = provisional.copy()
    tampered.loc[0, "threshold_source_stage"] = "formula_selection"
    with pytest.raises(ValueError, match="search_calibration|threshold"):
        protocol.evaluate_formula_selection(
            selection_features,
            _endpoint_labels(selection_features),
            cutoffs=cutoffs,
            provisional_thresholds=tampered,
        )

    negative_infinity = provisional.copy()
    negative_infinity.loc[0, "threshold"] = -np.inf
    with pytest.raises(ValueError, match="positive infinity|finite|threshold"):
        protocol.evaluate_formula_selection(
            selection_features,
            _endpoint_labels(selection_features),
            cutoffs=cutoffs,
            provisional_thresholds=negative_infinity,
        )


def test_final_thresholds_fit_only_selected_and_m5_on_threshold_fit():
    search_features = _many_group_features("search_calibration", 2)
    cutoffs = protocol.derive_disagreement_cutoffs(search_features)
    fit_features = _many_group_features("threshold_calibration", 100)

    rules = protocol.fit_final_thresholds(
        fit_features,
        _endpoint_labels(fit_features),
        cutoffs=cutoffs,
        selection={"state": "selected", "name": "MEAN"},
    )

    assert set(rules["role"]) == {"selected", "m5_baseline"}
    assert set(rules["track"]) == {"primary", "comparator"}
    assert set(rules["formula"]) == {"MEAN", "M5"}
    assert set(rules["threshold_source_role"]) == {"threshold_fit"}
    assert set(rules["operator"]) == {"score > threshold"}
    assert rules["calibration_n_groups"].eq(100).all()
    assert np.isfinite(rules["threshold"]).all()


def test_primary_final_threshold_is_keep_all_when_fit_has_fewer_than_99_groups():
    search_features = _many_group_features("search_calibration", 2)
    cutoffs = protocol.derive_disagreement_cutoffs(search_features)
    fit_features = _many_group_features("threshold_calibration", 98)

    rules = protocol.fit_final_thresholds(
        fit_features,
        _endpoint_labels(fit_features),
        cutoffs=cutoffs,
        selection={"state": "selected", "name": "MAX"},
    )

    primary = rules.loc[rules["track"].eq("primary")]
    comparator = rules.loc[rules["track"].eq("comparator")]
    assert np.isposinf(primary["threshold"]).all()
    assert set(primary["threshold_state"]) == {"keep_all"}
    assert set(primary["operator"]) == {"KEEP_ALL_SUPPORTED"}
    assert set(primary["unsupported_decision"]) == {"ABSTAIN"}
    assert np.isfinite(comparator["threshold"]).all()
    assert set(comparator["threshold_state"]) == {"finite"}


@pytest.mark.parametrize(
    "failed_group_count,expected_valid_groups,expected_state,expected_operator",
    (
        (1, 99, "finite", "score > threshold"),
        (2, 98, "keep_all", "KEEP_ALL_SUPPORTED"),
    ),
)
def test_primary_minimum_uses_formula_specific_valid_groups_at_98_99_boundary(
    failed_group_count,
    expected_valid_groups,
    expected_state,
    expected_operator,
):
    search_features = _many_group_features("search_calibration", 2)
    cutoffs = protocol.derive_disagreement_cutoffs(search_features)
    fit_features = _with_m1_group_failures(
        _many_group_features("threshold_calibration", 100),
        n_groups=failed_group_count,
    )

    rules = protocol.fit_final_thresholds(
        fit_features,
        _endpoint_labels(fit_features),
        cutoffs=cutoffs,
        selection={"state": "selected", "name": "MEAN"},
    )

    selected = rules.loc[rules["role"].eq("selected")]
    selected_primary = selected.loc[selected["track"].eq("primary")].iloc[0]
    assert selected["calibration_n_groups"].eq(expected_valid_groups).all()
    assert selected_primary["threshold_state"] == expected_state
    assert selected_primary["operator"] == expected_operator
    assert selected_primary["unsupported_decision"] == "ABSTAIN"
    assert np.isfinite(selected_primary["threshold"]) == (expected_state == "finite")

    # M5 support is independent: M1 failures cannot reduce its valid-group n.
    baseline = rules.loc[rules["role"].eq("m5_baseline")]
    assert baseline["calibration_n_groups"].eq(100).all()
    assert baseline["threshold_state"].eq("finite").all()


def test_formula_valid_group_requires_supported_finite_protected_row():
    cutoffs = protocol.derive_disagreement_cutoffs(_joint_ecdf_calibration_features())
    fit_features = _many_group_features("threshold_calibration", 100)
    affected_groups = fit_features["rk"].drop_duplicates().iloc[:2]
    protected_rows = (
        fit_features.loc[fit_features["rk"].isin(affected_groups)]
        .groupby("rk", sort=False)
        .head(1)
        .index
    )
    fit_features.loc[protected_rows, "m1_fmax_ev_per_a"] = 199.0

    rules = protocol.fit_final_thresholds(
        fit_features,
        _endpoint_labels(fit_features),
        cutoffs=cutoffs,
        selection={"state": "selected", "name": "CMEAN_JOINT99"},
    )

    selected = rules.loc[rules["role"].eq("selected")]
    primary = selected.loc[selected["track"].eq("primary")].iloc[0]
    assert selected["calibration_n_groups"].eq(98).all()
    assert np.isposinf(primary["threshold"])
    assert primary["threshold_state"] == "keep_all"
    assert primary["operator"] == "KEEP_ALL_SUPPORTED"
    assert primary["unsupported_decision"] == "ABSTAIN"
    assert (
        rules.loc[rules["role"].eq("m5_baseline"), "calibration_n_groups"].eq(100).all()
    )


def test_keep_all_supported_application_keeps_fail_open_abstentions():
    search_features = _many_group_features("search_calibration", 2)
    cutoffs = protocol.derive_disagreement_cutoffs(search_features)
    fit_features = _with_m1_group_failures(
        _many_group_features("threshold_calibration", 100),
        n_groups=2,
    )
    selection = {"state": "selected", "name": "MEAN"}
    rules = protocol.fit_final_thresholds(
        fit_features,
        _endpoint_labels(fit_features),
        cutoffs=cutoffs,
        selection=selection,
    )
    gate_features = _with_m1_group_failures(
        _many_group_features("threshold_calibration", 2),
        n_groups=1,
    )

    result = protocol.evaluate_development_gate(
        gate_features,
        _endpoint_labels(gate_features),
        cutoffs=cutoffs,
        selection=selection,
        final_thresholds=rules,
        n_resamples=10,
        bootstrap_batch_size=5,
    )

    primary_candidate = result["predictions"].loc[
        result["predictions"]["track"].eq("primary")
        & result["predictions"]["formula"].eq("selected_candidate")
    ]
    failed_group = gate_features["rk"].drop_duplicates().iloc[0]
    failed = primary_candidate.loc[primary_candidate["rk"].eq(failed_group)]
    supported = primary_candidate.loc[~primary_candidate["rk"].eq(failed_group)]
    assert set(failed["decision"]) == {"ABSTAIN"}
    assert not failed["supported"].any()
    assert set(supported["decision"]) == {"KEEP"}
    assert supported["supported"].all()


def test_null_selection_final_rule_is_keep_all_but_m5_is_still_fit():
    search_features = _many_group_features("search_calibration", 2)
    cutoffs = protocol.derive_disagreement_cutoffs(search_features)
    fit_features = _many_group_features("threshold_calibration", 100)

    rules = protocol.fit_final_thresholds(
        fit_features,
        _endpoint_labels(fit_features),
        cutoffs=cutoffs,
        selection={"state": "null_keep_all", "name": "null_keep_all"},
    )

    selected = rules.loc[rules["role"].eq("selected")]
    baseline = rules.loc[rules["role"].eq("m5_baseline")]
    assert set(selected["formula"]) == {"null_keep_all"}
    assert np.isposinf(selected["threshold"]).all()
    assert set(selected["operator"]) == {"KEEP_ALL"}
    assert set(selected["unsupported_decision"]) == {"KEEP"}
    assert set(baseline["formula"]) == {"M5"}
    assert np.isfinite(baseline["threshold"]).all()


def test_development_gate_only_applies_final_rules_and_passes_all_frozen_checks(
    monkeypatch,
):
    search_features = _many_group_features("search_calibration", 2)
    cutoffs = protocol.derive_disagreement_cutoffs(search_features)
    gate_features = _many_group_features("threshold_calibration", 60)
    final_rules = _final_rules("M1", selected_threshold=0.1, baseline_threshold=0.4)

    def forbidden_refit(*_args, **_kwargs):
        raise AssertionError("development_gate attempted to refit a threshold")

    monkeypatch.setattr(protocol, "group_conformal_threshold", forbidden_refit)
    result = protocol.evaluate_development_gate(
        gate_features,
        _endpoint_labels(gate_features),
        cutoffs=cutoffs,
        selection={"state": "selected", "name": "M1"},
        final_thresholds=final_rules,
        n_resamples=200,
        bootstrap_seed=20260801,
        bootstrap_batch_size=50,
    )

    predictions = result["predictions"]
    primary_predictions = predictions.loc[predictions["track"].eq("primary")]
    selected = primary_predictions.loc[
        primary_predictions["formula"].eq("selected_candidate")
    ]
    baseline = primary_predictions.loc[primary_predictions["formula"].eq("m5_baseline")]
    assert (
        selected[["sid", "rk"]]
        .reset_index(drop=True)
        .equals(baseline[["sid", "rk"]].reset_index(drop=True))
    )
    assert len(selected) == len(gate_features)
    assert len(baseline) == len(gate_features)
    assert set(result["metrics"]["evaluation_role"]) == {"development_gate"}
    assert set(result["metrics"]["threshold_source_role"]) == {"threshold_fit"}

    bootstrap = result["paired_bootstrap"]
    assert bootstrap["seed"] == 20260801
    assert bootstrap["n_resamples"] == 200
    assert bootstrap["metrics"]["dft_savings"]["difference"] == pytest.approx(0.5)
    assert bootstrap["metrics"]["dft_savings"]["difference_ci_95"][0] > 0
    assert bootstrap["metrics"]["valuable_item_recall"]["difference_ci_95"][0] >= -0.005

    gate = result["improvement_gate"]
    assert gate["dft_savings_delta"] == pytest.approx(0.5)
    assert gate["abstention_rate_delta"] == pytest.approx(0.0)
    assert gate["passes_dft_savings_magnitude"]
    assert gate["passes_dft_savings_paired_ci"]
    assert gate["passes_valuable_recall_noninferiority"]
    assert gate["passes_abstention_delta"]
    assert gate["passes_primary_safety"]
    assert gate["passes_comparator_safety"]
    assert gate["passes_improvement_gate"]


def test_gate_preserves_candidate_abstentions_and_m5_support_is_m5_only():
    search_features = _many_group_features("search_calibration", 2)
    cutoffs = protocol.derive_disagreement_cutoffs(search_features)
    gate_features = _many_group_features("threshold_calibration", 60)
    failed_rk = gate_features.iloc[0]["rk"]
    failed_rows = gate_features["rk"].eq(failed_rk)
    gate_features.loc[failed_rows, "m1_prediction_ok"] = False
    gate_features.loc[failed_rows, "committee_feature_ok"] = False
    gate_features.loc[failed_rows, "m1_energy_ev_per_atom"] = np.nan
    gate_features.loc[failed_rows, "m1_fmax_ev_per_a"] = np.nan
    gate_features.loc[failed_rows, "m1_frms_ev_per_a"] = np.nan

    result = protocol.evaluate_development_gate(
        gate_features,
        _endpoint_labels(gate_features),
        cutoffs=cutoffs,
        selection={"state": "selected", "name": "M1"},
        final_thresholds=_final_rules(
            "M1", selected_threshold=0.1, baseline_threshold=0.4
        ),
        n_resamples=100,
        bootstrap_seed=20260801,
        bootstrap_batch_size=50,
    )
    predictions = result["predictions"]
    candidate = predictions.loc[
        predictions["formula"].eq("selected_candidate")
        & predictions["rk"].eq(failed_rk)
    ]
    baseline = predictions.loc[
        predictions["formula"].eq("m5_baseline") & predictions["rk"].eq(failed_rk)
    ]

    assert set(candidate["decision"]) == {"ABSTAIN"}
    assert set(baseline["decision"]) <= {"KEEP", "REJECT"}
    assert not baseline["decision"].eq("ABSTAIN").any()
    assert set(
        predictions.loc[predictions["formula"].eq("selected_candidate"), "sid"]
    ) == set(predictions.loc[predictions["formula"].eq("m5_baseline"), "sid"])
    assert not result["improvement_gate"]["passes_abstention_delta"]
    assert not result["improvement_gate"]["passes_improvement_gate"]


def test_development_gate_rejects_non_fit_rules_without_refitting():
    search_features = _many_group_features("search_calibration", 2)
    cutoffs = protocol.derive_disagreement_cutoffs(search_features)
    gate_features = _many_group_features("threshold_calibration", 60)
    tampered = _final_rules("M1", selected_threshold=0.1, baseline_threshold=0.4)
    tampered.loc[0, "threshold_source_role"] = "development_gate"

    with pytest.raises(ValueError, match="threshold_fit|source"):
        protocol.evaluate_development_gate(
            gate_features,
            _endpoint_labels(gate_features),
            cutoffs=cutoffs,
            selection={"state": "selected", "name": "M1"},
            final_thresholds=tampered,
            n_resamples=10,
        )


def test_development_gate_rejects_threshold_state_inconsistent_with_value():
    search_features = _many_group_features("search_calibration", 2)
    cutoffs = protocol.derive_disagreement_cutoffs(search_features)
    gate_features = _many_group_features("threshold_calibration", 60)
    inconsistent = _final_rules("M1", selected_threshold=0.1, baseline_threshold=0.4)
    inconsistent.loc[0, "threshold_state"] = "keep_all"

    with pytest.raises(ValueError, match="threshold_state|keep_all|finite"):
        protocol.evaluate_development_gate(
            gate_features,
            _endpoint_labels(gate_features),
            cutoffs=cutoffs,
            selection={"state": "selected", "name": "M1"},
            final_thresholds=inconsistent,
            n_resamples=10,
        )


def test_full_freeze_publishes_exact_strict_hashed_artifacts(tmp_path, monkeypatch):
    full_features = pd.concat(
        [
            _many_group_features("search_calibration", 100),
            _many_group_features("formula_selection", 60),
            _many_group_features("threshold_calibration", 200),
        ],
        ignore_index=True,
    )
    paths = _write_freeze_inputs(tmp_path, monkeypatch, features=full_features)

    manifest = protocol.run_development_freeze(
        paths["features"],
        paths["labels"],
        paths["feature_manifest"],
        paths["output"],
        checkpoints=paths["checkpoints"],
        n_resamples=50,
        bootstrap_seed=20260801,
        bootstrap_batch_size=25,
    )

    expected_names = {
        "threshold_role_assignments.parquet",
        "development_frontier.parquet",
        "threshold_fit_rules.parquet",
        "development_gate_metrics.parquet",
        "PAIRED_BOOTSTRAP.json",
        "IMPROVEMENT_GATE.json",
        "FROZEN_PROTOCOL.json",
        "MANIFEST.json",
    }
    assert {path.name for path in paths["output"].iterdir()} == expected_names
    for name in expected_names & {
        "PAIRED_BOOTSTRAP.json",
        "IMPROVEMENT_GATE.json",
        "FROZEN_PROTOCOL.json",
        "MANIFEST.json",
    }:
        text = (paths["output"] / name).read_text()
        assert "NaN" not in text
        assert "Infinity" not in text
        json.loads(
            text,
            parse_constant=lambda value: (_ for _ in ()).throw(AssertionError(value)),
        )

    loaded_manifest = json.loads((paths["output"] / "MANIFEST.json").read_text())
    assert loaded_manifest == manifest
    assert manifest["protocol"] == (
        "2026-08-01-mattersim-committee-development-freeze-v1"
    )
    assert manifest["integrity"] == {"prepublish_rehash": "passed"}
    assert manifest["split"]["salt"] == ("next8-threshold-fit-gate-v1-20260801")
    assert manifest["bootstrap"] == {
        "method": "paired percentile bootstrap over rk composition clusters",
        "seed": 20260801,
        "n_resamples": 50,
        "batch_size": 25,
    }
    assert manifest["runtime"]["python_version"]
    assert manifest["runtime"]["python_implementation"]
    assert manifest["runtime"]["platform"]
    assert manifest["runtime"]["numpy_version"]
    assert manifest["runtime"]["pandas_version"]
    assert set(manifest["inputs_sha256"]) == {
        "features",
        "labels",
        "feature_manifest",
    }
    assert set(manifest["checkpoints"]) == {"m1", "m5"}
    required_sources = {
        "src/next8_mattersim_committee_protocol.py",
        "src/next6_elementa_protocol.py",
        "src/next6_elementa_diagnostics.py",
        "src/next6_wbm_build.py",
        "src/next8_mattersim_committee_features.py",
        "src/next6_mattersim_baseline.py",
        "src/next6_wbm_features.py",
        "src/next6_wbm_protocol.py",
    }
    assert required_sources <= set(manifest["executed_source_sha256"])
    assert set(manifest["outputs_sha256"]) == expected_names - {"MANIFEST.json"}
    for name, digest in manifest["outputs_sha256"].items():
        assert digest == _sha256(paths["output"] / name)

    roles = pd.read_parquet(paths["output"] / "threshold_role_assignments.parquet")
    assert len(roles) == 400
    assert roles.groupby("rk")["threshold_role"].nunique().eq(1).all()
    assert roles.loc[roles["threshold_role"].eq("threshold_fit"), "rk"].nunique() == 100
    assert (
        roles.loc[roles["threshold_role"].eq("development_gate"), "rk"].nunique() == 100
    )
    frontier = pd.read_parquet(paths["output"] / "development_frontier.parquet")
    rules = pd.read_parquet(paths["output"] / "threshold_fit_rules.parquet")
    metrics = pd.read_parquet(paths["output"] / "development_gate_metrics.parquet")
    assert len(frontier) == 22
    assert len(rules) == 4
    assert len(metrics) == 4
    assert set(rules["threshold_source_role"]) == {"threshold_fit"}
    assert set(metrics["evaluation_role"]) == {"development_gate"}
    assert rules["calibration_n_groups"].eq(100).all()
    assert metrics["n_groups"].eq(100).all()

    frozen = json.loads((paths["output"] / "FROZEN_PROTOCOL.json").read_text())
    catalog_serialized = frozen["catalog"]["serialized"]
    assert (
        frozen["catalog"]["sha256"]
        == hashlib.sha256(catalog_serialized.encode("utf-8")).hexdigest()
    )
    assert frozen["cutoff_provenance"]["feature_sha256"] == _sha256(paths["features"])
    assert frozen["cutoff_provenance"]["feature_manifest_sha256"] == _sha256(
        paths["feature_manifest"]
    )
    assert "token" not in json.dumps(frozen).lower()
    assert frozen["selection"]["name"] == "M1"
    assert not frozen["improvement_gate"]["passes_improvement_gate"]
    assert not any(
        path.name.startswith(f".{paths['output'].name}.staging-")
        for path in tmp_path.iterdir()
    )


@pytest.mark.parametrize("role", ("features", "feature_manifest", "labels"))
def test_prepublish_rehash_detects_changed_input_and_cleans_staging(
    tmp_path, monkeypatch, role
):
    paths = _write_freeze_inputs(tmp_path, monkeypatch)
    original_validate = getattr(
        protocol, "_validate_staged_artifacts", lambda *_args, **_kwargs: None
    )

    def mutate_after_staging(*args, **kwargs):
        original_validate(*args, **kwargs)
        paths[role].write_bytes(paths[role].read_bytes() + b"changed")

    monkeypatch.setattr(
        protocol, "_validate_staged_artifacts", mutate_after_staging, raising=False
    )
    with pytest.raises(RuntimeError, match="changed|hash"):
        protocol.run_development_freeze(
            paths["features"],
            paths["labels"],
            paths["feature_manifest"],
            paths["output"],
            checkpoints=paths["checkpoints"],
            n_resamples=10,
        )

    assert not paths["output"].exists()
    assert not any(
        path.name.startswith(f".{paths['output'].name}.staging-")
        for path in tmp_path.iterdir()
    )


def test_atomic_publish_race_preserves_competing_target_and_cleans_staging(
    tmp_path, monkeypatch
):
    paths = _write_freeze_inputs(tmp_path, monkeypatch)

    def racing_publish(_source, target):
        target.mkdir()
        (target / "racing-writer").write_bytes(b"preserve")
        raise FileExistsError("racing target")

    monkeypatch.setattr(
        protocol, "_atomic_publish_directory_no_replace", racing_publish
    )
    with pytest.raises(FileExistsError, match="racing target"):
        protocol.run_development_freeze(
            paths["features"],
            paths["labels"],
            paths["feature_manifest"],
            paths["output"],
            checkpoints=paths["checkpoints"],
            n_resamples=10,
        )

    assert (paths["output"] / "racing-writer").read_bytes() == b"preserve"
    assert not any(
        path.name.startswith(f".{paths['output'].name}.staging-")
        for path in tmp_path.iterdir()
    )


@pytest.mark.parametrize("failure", ("duplicate", "test_stage", "nonfinite"))
def test_labels_are_exact_stage_matched_and_finite(tmp_path, monkeypatch, failure):
    paths = _write_freeze_inputs(tmp_path, monkeypatch)
    labels = pd.read_parquet(paths["labels"])
    if failure == "duplicate":
        labels = pd.concat([labels, labels.iloc[[0]]], ignore_index=True)
    elif failure == "test_stage":
        labels.loc[0, "stage"] = "test"
    elif failure == "nonfinite":
        labels.loc[0, "e_per_atom"] = np.nan
    labels.to_parquet(paths["labels"], index=False)

    with pytest.raises(ValueError, match="duplicate|stage|finite|keys"):
        protocol.run_development_freeze(
            paths["features"],
            paths["labels"],
            paths["feature_manifest"],
            paths["output"],
            checkpoints=paths["checkpoints"],
            n_resamples=10,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        pytest.param(
            lambda manifest: manifest.update(protocol="wrong"),
            id="protocol",
        ),
        pytest.param(lambda manifest: manifest.update(mode="test"), id="mode"),
        pytest.param(
            lambda manifest: manifest["checkpoints"]["m1"].update(sha256="0" * 64),
            id="checkpoint",
        ),
        pytest.param(
            lambda manifest: manifest["executed_source_sha256"].update(
                {"src/next8_mattersim_committee_features.py": "0" * 64}
            ),
            id="executed-source",
        ),
        pytest.param(
            lambda manifest: manifest["outputs_sha256"].update(
                {"undeclared-extra.parquet": "0" * 64}
            ),
            id="output-closure",
        ),
        pytest.param(
            lambda manifest: manifest.update(production_protocol_eligible=False),
            id="production-false",
        ),
        pytest.param(
            lambda manifest: manifest.update(production_protocol_eligible=1),
            id="production-bool-like-one",
        ),
        pytest.param(
            lambda manifest: manifest.pop("production_protocol_eligible"),
            id="production-missing",
        ),
        pytest.param(
            lambda manifest: manifest.update(evidence_role="testing_only"),
            id="evidence-role-wrong",
        ),
        pytest.param(
            lambda manifest: manifest.pop("evidence_role"),
            id="evidence-role-missing",
        ),
        pytest.param(
            lambda manifest: manifest["adapter"].update(mode="injected_test_double"),
            id="injected-adapter",
        ),
        pytest.param(
            lambda manifest: manifest.pop("adapter"),
            id="adapter-missing",
        ),
        pytest.param(
            lambda manifest: manifest["adapter"].pop("mode"),
            id="adapter-mode-missing",
        ),
        pytest.param(
            lambda manifest: manifest["adapter"].pop("implementation"),
            id="adapter-implementation-missing",
        ),
        pytest.param(
            lambda manifest: manifest["adapter"]["implementation"].update(
                source_hash_verified=False
            ),
            id="source-hash-unverified",
        ),
        pytest.param(
            lambda manifest: manifest["adapter"]["implementation"].update(
                source_hash_verified=1
            ),
            id="source-hash-bool-like-one",
        ),
        pytest.param(
            lambda manifest: manifest["adapter"]["implementation"].pop(
                "source_hash_verified"
            ),
            id="source-hash-verification-missing",
        ),
        pytest.param(
            lambda manifest: manifest["adapter"]["implementation"].update(
                source_path="/wrong/adapter.py"
            ),
            id="implementation-source-path",
        ),
        pytest.param(
            lambda manifest: manifest["adapter"]["implementation"].update(
                source_sha256="0" * 64
            ),
            id="implementation-source-hash",
        ),
    ),
)
def test_feature_manifest_contract_fails_before_label_open(
    tmp_path, monkeypatch, mutation
):
    paths = _write_freeze_inputs(tmp_path, monkeypatch)
    manifest = json.loads(paths["feature_manifest"].read_text())
    mutation(manifest)
    paths["feature_manifest"].write_text(
        json.dumps(manifest, allow_nan=False), encoding="utf-8"
    )
    real_read_parquet = pd.read_parquet
    real_sha256_file = protocol._sha256_file

    def guarded_read_parquet(path, *args, **kwargs):
        if Path(path) == paths["labels"]:
            raise AssertionError("labels opened before manifest failure")
        return real_read_parquet(path, *args, **kwargs)

    def guarded_sha256_file(path):
        if Path(path) == paths["labels"]:
            raise AssertionError("labels hashed before manifest failure")
        return real_sha256_file(path)

    def forbidden_publish(*_args, **_kwargs):
        raise AssertionError("output published before manifest failure")

    monkeypatch.setattr(protocol.pd, "read_parquet", guarded_read_parquet)
    monkeypatch.setattr(protocol, "_sha256_file", guarded_sha256_file)
    monkeypatch.setattr(
        protocol, "_atomic_publish_directory_no_replace", forbidden_publish
    )
    with pytest.raises(ValueError, match="manifest|protocol|mode|checkpoint|source"):
        protocol.run_development_freeze(
            paths["features"],
            paths["labels"],
            paths["feature_manifest"],
            paths["output"],
            checkpoints=paths["checkpoints"],
            n_resamples=10,
        )
    assert not paths["output"].exists()


def test_label_snapshot_hash_and_parse_follow_all_label_free_checks(
    tmp_path, monkeypatch
):
    paths = _write_freeze_inputs(tmp_path, monkeypatch)
    label_bytes = paths["labels"].read_bytes()
    events: list[str] = []
    real_validate_features = protocol._validate_full_development_features
    real_derive_cutoffs = protocol.derive_disagreement_cutoffs
    real_split = protocol.split_threshold_groups
    real_sha256_file = protocol._sha256_file
    real_read_bytes = Path.read_bytes
    real_hashlib_sha256 = protocol.hashlib.sha256
    real_read_parquet = protocol.pd.read_parquet

    def observed_validate_features(*args, **kwargs):
        result = real_validate_features(*args, **kwargs)
        events.append("feature_validation_completed")
        return result

    def observed_derive_cutoffs(*args, **kwargs):
        result = real_derive_cutoffs(*args, **kwargs)
        events.append("cutoff_derivation_completed")
        return result

    def observed_split(*args, **kwargs):
        result = real_split(*args, **kwargs)
        events.append("threshold_split_completed")
        return result

    def observed_read_bytes(path):
        data = real_read_bytes(path)
        if Path(path) == paths["labels"]:
            events.append("label_snapshot")
        return data

    def observed_hashlib_sha256(data=b"", *args, **kwargs):
        if data == label_bytes:
            events.append("label_snapshot_hash")
        return real_hashlib_sha256(data, *args, **kwargs)

    def observed_sha256_file(path):
        if Path(path) == paths["labels"]:
            events.append("label_path_rehash")
        return real_sha256_file(path)

    def observed_read_parquet(source, *args, **kwargs):
        if isinstance(source, (str, Path)) and Path(source) == paths["labels"]:
            events.append("label_path_parquet")
        if isinstance(source, io.BytesIO) and source.getvalue() == label_bytes:
            events.append("label_parquet")
        return real_read_parquet(source, *args, **kwargs)

    monkeypatch.setattr(
        protocol, "_validate_full_development_features", observed_validate_features
    )
    monkeypatch.setattr(
        protocol, "derive_disagreement_cutoffs", observed_derive_cutoffs
    )
    monkeypatch.setattr(protocol, "split_threshold_groups", observed_split)
    monkeypatch.setattr(Path, "read_bytes", observed_read_bytes)
    monkeypatch.setattr(protocol.hashlib, "sha256", observed_hashlib_sha256)
    monkeypatch.setattr(protocol, "_sha256_file", observed_sha256_file)
    monkeypatch.setattr(protocol.pd, "read_parquet", observed_read_parquet)

    protocol.run_development_freeze(
        paths["features"],
        paths["labels"],
        paths["feature_manifest"],
        paths["output"],
        checkpoints=paths["checkpoints"],
        n_resamples=10,
        bootstrap_batch_size=5,
    )

    split_index = events.index("threshold_split_completed")
    assert events.index("feature_validation_completed") < split_index
    assert events.index("cutoff_derivation_completed") < split_index
    assert events.index("label_snapshot") > split_index
    assert events.index("label_snapshot_hash") > events.index("label_snapshot")
    assert events.index("label_parquet") > events.index("label_snapshot_hash")
    assert events.index("label_path_rehash") > events.index("label_parquet")
    assert "label_path_parquet" not in events


@pytest.mark.parametrize("role", ("features", "feature_manifest", "labels"))
def test_computation_parses_initial_snapshot_across_symlink_swap_and_restore(
    tmp_path, monkeypatch, role
):
    paths = _write_freeze_inputs(tmp_path, monkeypatch)
    input_path = paths[role]
    original_table = (
        pd.read_parquet(input_path) if role in {"features", "labels"} else None
    )
    original_manifest = (
        json.loads(input_path.read_text()) if role == "feature_manifest" else None
    )
    snapshot_a = tmp_path / f"{role}-snapshot-a{input_path.suffix}"
    snapshot_b = tmp_path / f"{role}-snapshot-b{input_path.suffix}"
    input_path.rename(snapshot_a)
    if role == "features":
        changed = original_table.copy()
        changed.loc[0, "stage"] = "test"
        changed.to_parquet(snapshot_b, index=False)
    elif role == "labels":
        changed = original_table.copy()
        changed.loc[0, "stage"] = "test"
        changed.to_parquet(snapshot_b, index=False)
    else:
        changed = dict(original_manifest)
        changed["production_protocol_eligible"] = False
        snapshot_b.write_text(json.dumps(changed, allow_nan=False), encoding="utf-8")
    input_path.symlink_to(snapshot_a)
    snapshot_bytes = snapshot_a.read_bytes()

    real_read_parquet = protocol.pd.read_parquet
    real_read_text = Path.read_text
    real_json_loads = protocol.json.loads
    triggered = False

    def switch_to(target):
        input_path.unlink()
        input_path.symlink_to(target)

    def swapped_call(call):
        nonlocal triggered
        triggered = True
        switch_to(snapshot_b)
        try:
            return call()
        finally:
            switch_to(snapshot_a)

    def observed_read_parquet(source, *args, **kwargs):
        source_is_target_path = (
            isinstance(source, (str, Path)) and Path(source) == input_path
        )
        source_is_target_snapshot = (
            isinstance(source, io.BytesIO) and source.getvalue() == snapshot_bytes
        )
        if (
            not triggered
            and role in {"features", "labels"}
            and (source_is_target_path or source_is_target_snapshot)
        ):
            return swapped_call(lambda: real_read_parquet(source, *args, **kwargs))
        return real_read_parquet(source, *args, **kwargs)

    def observed_read_text(path, *args, **kwargs):
        if not triggered and role == "feature_manifest" and Path(path) == input_path:
            return swapped_call(lambda: real_read_text(path, *args, **kwargs))
        return real_read_text(path, *args, **kwargs)

    def observed_json_loads(value, *args, **kwargs):
        value_bytes = value if isinstance(value, bytes) else str(value).encode()
        if (
            not triggered
            and role == "feature_manifest"
            and value_bytes == snapshot_bytes
        ):
            return swapped_call(lambda: real_json_loads(value, *args, **kwargs))
        return real_json_loads(value, *args, **kwargs)

    monkeypatch.setattr(protocol.pd, "read_parquet", observed_read_parquet)
    monkeypatch.setattr(Path, "read_text", observed_read_text)
    monkeypatch.setattr(protocol.json, "loads", observed_json_loads)

    protocol.run_development_freeze(
        paths["features"],
        paths["labels"],
        paths["feature_manifest"],
        paths["output"],
        checkpoints=paths["checkpoints"],
        n_resamples=10,
        bootstrap_batch_size=5,
    )

    assert triggered
    assert input_path.resolve() == snapshot_a.resolve()
    assert paths["output"].is_dir()


def test_cli_routes_both_checkpoints_and_bootstrap_parameters(monkeypatch, tmp_path):
    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"outputs_sha256": {"artifact": "0" * 64}}

    monkeypatch.setattr(protocol, "run_development_freeze", fake_run)
    exit_code = protocol.main(
        [
            "--features",
            str(tmp_path / "features.parquet"),
            "--labels",
            str(tmp_path / "labels.parquet"),
            "--feature-manifest",
            str(tmp_path / "MANIFEST.json"),
            "--output",
            str(tmp_path / "freeze"),
            "--m1-checkpoint",
            str(tmp_path / "m1.pth"),
            "--m5-checkpoint",
            str(tmp_path / "m5.pth"),
            "--n-resamples",
            "37",
            "--bootstrap-seed",
            "20260801",
            "--bootstrap-batch-size",
            "11",
        ]
    )

    assert exit_code == 0
    assert captured["kwargs"]["checkpoints"] == {
        "m1": tmp_path / "m1.pth",
        "m5": tmp_path / "m5.pth",
    }
    assert captured["kwargs"]["n_resamples"] == 37
    assert captured["kwargs"]["bootstrap_seed"] == 20260801
    assert captured["kwargs"]["bootstrap_batch_size"] == 11


def test_catalog_is_exact_frozen_eleven_in_order():
    assert protocol.FORMULA_NAMES == (
        "M5",
        "M1",
        "MIN",
        "MEAN",
        "MAX",
        "LCB",
        "AGREE99",
        "AGREE995",
        "AGREE_EF995",
        "CMEAN",
        "CMEAN_JOINT99",
    )


def test_group_local_gaps_feed_all_eleven_exact_formulas():
    cutoffs = protocol.derive_disagreement_cutoffs(_search_calibration_features())

    scores = protocol.construct_committee_scores(
        _formula_selection_features(),
        cutoffs=cutoffs,
        expected_stage="formula_selection",
    )
    indexed = _scores_by_formula_and_sid(scores)

    assert scores["formula"].drop_duplicates().tolist() == list(protocol.FORMULA_NAMES)
    assert len(scores) == 4 * 11
    row = indexed.loc[(slice(None), "sel-a1"), :].set_index("formula")
    assert row["g1_ev_per_atom"].unique().tolist() == pytest.approx([2.0])
    assert row["g5_ev_per_atom"].unique().tolist() == pytest.approx([1.0])
    assert row["disagreement_ev_per_atom"].unique().tolist() == pytest.approx([1.0])
    assert row.loc[
        [
            "M5",
            "M1",
            "MIN",
            "MEAN",
            "MAX",
            "LCB",
            "AGREE99",
            "AGREE995",
            "AGREE_EF995",
            "CMEAN",
            "CMEAN_JOINT99",
        ],
        "score_ev_per_atom",
    ].tolist() == pytest.approx([1.0, 2.0, 1.0, 1.5, 2.0, 0.5, 1.5, 1.5, 1.5, 1.5, 1.5])
    assert set(row["state"]) == {"KEEP"}

    # A separate composition group must have its own minima.
    b0 = indexed.loc[("M5", "sel-b0")]
    b1 = indexed.loc[("M1", "sel-b1")]
    assert b0["g5_ev_per_atom"] == pytest.approx(0.0)
    assert b1["g1_ev_per_atom"] == pytest.approx(0.5)


def test_cmean_rezeros_mean_after_joint_group_scoring_when_argmins_cross():
    cutoffs = protocol.derive_disagreement_cutoffs(_search_calibration_features())
    crossed = pd.DataFrame(
        [
            _ready_row("cross-a", "CROSS", "formula_selection", 0.0, 10.0),
            _ready_row("cross-b", "CROSS", "formula_selection", 10.0, 0.0),
        ]
    )

    scores = protocol.construct_committee_scores(
        crossed, cutoffs=cutoffs, expected_stage="formula_selection"
    ).set_index(["formula", "sid"])

    assert scores.loc[("MEAN", "cross-a"), "score_ev_per_atom"] == pytest.approx(5.0)
    assert scores.loc[("MEAN", "cross-b"), "score_ev_per_atom"] == pytest.approx(5.0)
    assert scores.loc[("CMEAN", "cross-a"), "score_ev_per_atom"] == pytest.approx(0.0)
    assert scores.loc[("CMEAN", "cross-b"), "score_ev_per_atom"] == pytest.approx(0.0)
    assert scores.loc[
        ("CMEAN_JOINT99", "cross-a"), "score_ev_per_atom"
    ] == pytest.approx(0.0)
    assert scores.loc[
        ("CMEAN_JOINT99", "cross-b"), "score_ev_per_atom"
    ] == pytest.approx(0.0)


def test_joint99_uses_row_weighted_right_ecdfs_and_strict_j_boundary():
    calibration = _joint_ecdf_calibration_features()
    cutoffs = protocol.derive_disagreement_cutoffs(calibration)

    assert cutoffs.joint_reference_n == 200
    assert cutoffs.joint_reference_n_rk == 1
    assert cutoffs.joint_weighting == "row"
    assert cutoffs.joint_ecdf_side == "right"
    assert cutoffs.joint_q99 == pytest.approx(0.995)
    assert cutoffs.joint_reference_dE == tuple(float(i) for i in range(200))
    assert cutoffs.joint_reference_dFmax == tuple(float(i) for i in range(200))
    assert cutoffs.joint_reference_dFrms == tuple(float(i) for i in range(200))
    assert len(cutoffs.joint_reference_sha256) == 64

    later = pd.DataFrame(
        [
            _ready_row("joint-0", "JOINT", "formula_selection", 0.0, 0.0),
            _ready_row(
                "joint-equal-j",
                "JOINT",
                "formula_selection",
                198.0,
                0.0,
                m1_fmax=198.0,
                m1_frms=198.0,
            ),
            _ready_row(
                "joint-above-j",
                "JOINT",
                "formula_selection",
                199.0,
                0.0,
                m1_fmax=199.0,
                m1_frms=199.0,
            ),
        ]
    )
    scores = protocol.construct_committee_scores(
        later, cutoffs=cutoffs, expected_stage="formula_selection"
    ).set_index(["formula", "sid"])

    equality = scores.loc[("CMEAN_JOINT99", "joint-equal-j")]
    assert equality["joint_ecdf_score"] == pytest.approx(0.995)
    assert equality["state"] == "KEEP"
    assert equality["score_ev_per_atom"] == pytest.approx(99.0)
    # Right-continuous H(x)=#(ref<=x)/n means raw equality at the first
    # rank above qJ is already J>qJ and must abstain.
    above = scores.loc[("CMEAN_JOINT99", "joint-above-j")]
    assert above["joint_ecdf_score"] == pytest.approx(1.0)
    assert above["state"] == "ABSTAIN"
    assert above["abstain_reason"] == "joint_ecdf_disagreement_above_threshold"
    assert np.isnan(above["score_ev_per_atom"])
    assert scores.loc[("CMEAN", "joint-above-j"), "state"] == "KEEP"
    assert scores.loc[("CMEAN", "joint-above-j"), "score_ev_per_atom"] == pytest.approx(
        99.5
    )

    calibration_scores = protocol.construct_committee_scores(
        calibration, cutoffs=cutoffs, expected_stage="search_calibration"
    )
    joint_rows = calibration_scores.loc[
        calibration_scores["formula"].eq("CMEAN_JOINT99")
    ]
    assert joint_rows["state"].eq("ABSTAIN").sum() == 1
    assert joint_rows["state"].eq("ABSTAIN").mean() == pytest.approx(0.005)


@pytest.mark.parametrize(
    "force_component,force_kwargs",
    (
        ("fmax", {"m1_fmax": 199.0}),
        ("frms", {"m1_frms": 199.0}),
    ),
)
def test_joint99_each_force_marginal_can_independently_trigger(
    force_component, force_kwargs
):
    cutoffs = protocol.derive_disagreement_cutoffs(_joint_ecdf_calibration_features())
    later = pd.DataFrame(
        [
            _ready_row("force-base", "FORCE-J", "formula_selection", 0.0, 0.0),
            _ready_row(
                f"{force_component}-high",
                "FORCE-J",
                "formula_selection",
                1.0,
                1.0,
                **force_kwargs,
            ),
        ]
    )

    scores = protocol.construct_committee_scores(
        later, cutoffs=cutoffs, expected_stage="formula_selection"
    ).set_index(["formula", "sid"])

    row = scores.loc[("CMEAN_JOINT99", f"{force_component}-high")]
    assert row["disagreement_ev_per_atom"] == pytest.approx(0.0)
    assert row[f"{force_component}_disagreement_ev_per_a"] == pytest.approx(199.0)
    other = "frms" if force_component == "fmax" else "fmax"
    assert row[f"{other}_disagreement_ev_per_a"] == pytest.approx(0.0)
    assert row["joint_ecdf_score"] == pytest.approx(1.0)
    assert row["state"] == "ABSTAIN"
    assert row["abstain_reason"] == ("joint_ecdf_disagreement_above_threshold")


def test_joint_ecdf_reference_hash_is_order_invariant_but_content_sensitive():
    calibration = _joint_ecdf_calibration_features()
    original = protocol.derive_disagreement_cutoffs(calibration)
    reordered = protocol.derive_disagreement_cutoffs(
        calibration.sample(frac=1.0, random_state=20260801).reset_index(drop=True)
    )
    changed_features = calibration.copy()
    changed_features.loc[
        changed_features["sid"].eq("joint-cal-100"),
        "m1_fmax_ev_per_a",
    ] += 0.25
    changed = protocol.derive_disagreement_cutoffs(changed_features)

    assert original.joint_reference_dE == reordered.joint_reference_dE
    assert original.joint_reference_dFmax == reordered.joint_reference_dFmax
    assert original.joint_reference_dFrms == reordered.joint_reference_dFrms
    assert original.joint_q99 == reordered.joint_q99
    assert original.joint_reference_sha256 == (reordered.joint_reference_sha256)
    assert original.joint_reference_sha256 != changed.joint_reference_sha256


@pytest.mark.parametrize("consumer", ("serialize", "construct"))
def test_joint_ecdf_reference_cannot_be_tampered_with_recomputed_hash(consumer):
    cutoffs = protocol.derive_disagreement_cutoffs(_joint_ecdf_calibration_features())
    changed_dE = cutoffs.joint_reference_dE[:-1] + (200.0,)
    changed_hash = protocol._joint_reference_fingerprint(
        dE=changed_dE,
        dFmax=cutoffs.joint_reference_dFmax,
        dFrms=cutoffs.joint_reference_dFrms,
        n=cutoffs.joint_reference_n,
        n_rk=cutoffs.joint_reference_n_rk,
        q99=cutoffs.joint_q99,
    )
    tampered = replace(
        cutoffs,
        joint_reference_dE=changed_dE,
        joint_reference_sha256=changed_hash,
    )

    with pytest.raises(ValueError, match="cutoff fields|registered"):
        _consume_cutoffs(consumer, tampered)


@pytest.mark.parametrize("n_rows", (1, 2, 99, 100, 101, 200, 257))
def test_joint99_search_empirical_added_abstentions_obey_exact_tail_bound(
    n_rows,
):
    calibration = _joint_ecdf_calibration_features(n_rows=n_rows)
    cutoffs = protocol.derive_disagreement_cutoffs(calibration)
    scores = protocol.construct_committee_scores(
        calibration,
        cutoffs=cutoffs,
        expected_stage="search_calibration",
    )
    joint = scores.loc[scores["formula"].eq("CMEAN_JOINT99")]
    n_abstained = int(joint["state"].eq("ABSTAIN").sum())

    assert n_abstained <= int(np.floor(0.01 * (n_rows - 1)))
    assert n_abstained / n_rows < 0.01


def test_new_formula_cost_and_complexity_are_frozen():
    assert protocol.FORMULA_COST["CMEAN"] == 6
    assert protocol.FORMULA_COST["CMEAN_JOINT99"] == 6
    assert protocol.FORMULA_COMPLEXITY["CMEAN"] == 3
    assert protocol.FORMULA_COMPLEXITY["CMEAN_JOINT99"] == 5


def test_finite_inputs_that_overflow_group_gaps_fail_closed():
    cutoffs = protocol.derive_disagreement_cutoffs(_search_calibration_features())
    maximum = np.finfo(float).max
    features = pd.DataFrame(
        [
            _ready_row(
                "overflow-low",
                "OVERFLOW",
                "formula_selection",
                -maximum,
                -maximum,
            ),
            _ready_row(
                "overflow-high",
                "OVERFLOW",
                "formula_selection",
                maximum,
                maximum,
            ),
        ]
    )

    with pytest.raises(ValueError, match="finite|overflow"):
        protocol.construct_committee_scores(
            features,
            cutoffs=cutoffs,
            expected_stage="formula_selection",
        )


def test_stable_mean_keeps_maximum_finite_gaps_finite_and_nonnegative():
    cutoffs = protocol.derive_disagreement_cutoffs(_search_calibration_features())
    maximum = np.finfo(float).max
    features = pd.DataFrame(
        [
            _ready_row("max-0", "MAX", "formula_selection", 0.0, 0.0),
            _ready_row("max-1", "MAX", "formula_selection", maximum, maximum),
        ]
    )

    scores = protocol.construct_committee_scores(
        features,
        cutoffs=cutoffs,
        expected_stage="formula_selection",
    )

    kept = scores.loc[scores["state"].eq("KEEP"), "score_ev_per_atom"]
    assert np.isfinite(kept.to_numpy(dtype=float)).all()
    assert (kept >= 0.0).all()
    indexed = _scores_by_formula_and_sid(scores)
    assert indexed.loc[("MEAN", "max-1"), "score_ev_per_atom"] == maximum
    assert indexed.loc[("LCB", "max-1"), "score_ev_per_atom"] == maximum


def test_disagreement_cutoffs_use_search_calibration_group_gaps_only():
    features = _search_calibration_features()

    cutoffs = protocol.derive_disagreement_cutoffs(features)

    # u values are [0, 0, 0, 3], despite raw m1/m5 offsets near 100-200 eV.
    assert cutoffs.source_stage == "search_calibration"
    assert cutoffs.eligible_row_count == 4
    assert cutoffs.quantile_method == "higher"
    assert cutoffs.q99_ev_per_atom == pytest.approx(3.0)
    assert cutoffs.q995_ev_per_atom == pytest.approx(3.0)
    assert cutoffs.q995_force_ev_per_a == pytest.approx(3.0)

    wrong_stage = features.assign(stage="formula_selection")
    with pytest.raises(ValueError, match="search_calibration"):
        protocol.derive_disagreement_cutoffs(wrong_stage)


def test_formula_catalog_serialization_has_only_math_and_calibration_metadata():
    cutoffs = protocol.derive_disagreement_cutoffs(_search_calibration_features())

    serialized = protocol.serialize_formula_catalog(cutoffs)
    payload = json.loads(serialized)

    assert payload["formulas"] == [
        {"name": "M5", "score": "g5"},
        {"name": "M1", "score": "g1"},
        {"name": "MIN", "score": "min(g1,g5)"},
        {"name": "MEAN", "score": "(g1+g5)/2"},
        {"name": "MAX", "score": "max(g1,g5)"},
        {"name": "LCB", "score": "max(0,(g1+g5)/2-uE)"},
        {
            "name": "AGREE99",
            "score": "(g1+g5)/2",
            "abstain_if": "uE>q99_energy",
        },
        {
            "name": "AGREE995",
            "score": "(g1+g5)/2",
            "abstain_if": "uE>q995_energy",
        },
        {
            "name": "AGREE_EF995",
            "score": "(g1+g5)/2",
            "abstain_if": "uE>q995_energy or uF>q995_force",
        },
        {
            "name": "CMEAN",
            "score": "(g1+g5)/2-min_rk((g1+g5)/2)",
        },
        {
            "name": "CMEAN_JOINT99",
            "score": "(g1+g5)/2-min_rk((g1+g5)/2)",
            "abstain_if": "max(H_E(dE),H_Fmax(dFmax),H_Frms(dFrms))>qJ99",
        },
    ]
    assert payload["calibration"]["source_stage"] == "search_calibration"
    assert payload["calibration"]["n"] == 4
    assert payload["calibration"]["quantile_method"] == "higher"
    assert payload["calibration"]["q99_ev_per_atom"] == pytest.approx(3.0)
    assert payload["calibration"]["q995_ev_per_atom"] == pytest.approx(3.0)
    assert payload["calibration"]["q995_force_ev_per_a"] == pytest.approx(3.0)
    joint = payload["calibration"]["joint_ecdf"]
    assert joint["n"] == 4
    assert joint["n_rk"] == 2
    assert joint["weighting"] == "row"
    assert joint["side"] == "right"
    assert joint["quantile_method"] == "higher"
    assert 0.0 <= joint["qJ99"] <= 1.0
    assert set(joint["sorted_reference"]) == {"dE", "dFmax", "dFrms"}
    assert all(len(values) == 4 for values in joint["sorted_reference"].values())
    assert len(joint["reference_sha256"]) == 64
    fingerprint = payload["calibration"]["calibration_fingerprint_sha256"]
    assert len(fingerprint) == 64
    assert int(fingerprint, 16) >= 0
    forbidden = (
        "label",
        "endpoint",
        "delta_e",
        "exact_min",
        "near_min",
        "valuable",
        "final_ionic_step",
        "suffix",
        "material",
        '"sid":',
        '"rk":',
    )
    lowered = serialized.lower()
    assert not any(token in lowered for token in forbidden)


def test_calibration_fingerprint_is_order_invariant_but_content_sensitive():
    features = _search_calibration_features()

    original = protocol.derive_disagreement_cutoffs(features)
    reordered = protocol.derive_disagreement_cutoffs(
        features.iloc[::-1].reset_index(drop=True)
    )
    shifted_features = features.copy()
    shifted_features["m1_energy_ev_per_atom"] += 7.0
    shifted = protocol.derive_disagreement_cutoffs(shifted_features)
    force_shifted_features = features.copy()
    force_shifted_features["m1_fmax_ev_per_a"] += 11.0
    force_shifted_features["m5_fmax_ev_per_a"] += 11.0
    force_shifted = protocol.derive_disagreement_cutoffs(force_shifted_features)

    assert original.calibration_fingerprint_sha256 == (
        reordered.calibration_fingerprint_sha256
    )
    assert original.q99_ev_per_atom == shifted.q99_ev_per_atom
    assert original.q995_ev_per_atom == shifted.q995_ev_per_atom
    assert original.q995_force_ev_per_a == (force_shifted.q995_force_ev_per_a)
    assert original.calibration_fingerprint_sha256 != (
        shifted.calibration_fingerprint_sha256
    )
    assert original.calibration_fingerprint_sha256 != (
        force_shifted.calibration_fingerprint_sha256
    )


def test_higher_quantile_is_deterministic_for_ties_and_small_samples():
    tied = pd.DataFrame(
        [
            _ready_row("tie-0", "T", "search_calibration", 0.0, 0.0),
            _ready_row("tie-1", "T", "search_calibration", 5.0, 0.0),
        ]
    )
    cutoffs = protocol.derive_disagreement_cutoffs(tied)

    assert cutoffs.eligible_row_count == 2
    assert cutoffs.quantile_method == "higher"
    assert cutoffs.q99_ev_per_atom == pytest.approx(5.0)
    assert cutoffs.q995_ev_per_atom == pytest.approx(5.0)
    assert cutoffs.q995_force_ev_per_a == pytest.approx(0.0)

    singleton = pd.DataFrame(
        [_ready_row("one", "ONLY", "search_calibration", -7.0, 9.0)]
    )
    one = protocol.derive_disagreement_cutoffs(singleton)
    assert one.eligible_row_count == 1
    assert one.q99_ev_per_atom == pytest.approx(0.0)
    assert one.q995_ev_per_atom == pytest.approx(0.0)
    assert one.q995_force_ev_per_a == pytest.approx(0.0)


@pytest.mark.parametrize(
    "failed_model,baseline_formula,failed_formula",
    (("m1", "M5", "M1"), ("m5", "M1", "M5")),
)
def test_model_failure_preserves_other_baseline_but_abstains_joint_formulas(
    failed_model, baseline_formula, failed_formula
):
    cutoffs = protocol.derive_disagreement_cutoffs(_search_calibration_features())
    failed = _formula_selection_features()
    failed.loc[failed["sid"].eq("sel-b1"), f"{failed_model}_prediction_ok"] = False
    failed.loc[failed["sid"].eq("sel-b1"), "committee_feature_ok"] = False
    failed.loc[
        failed["sid"].eq("sel-b1"), f"{failed_model}_energy_ev_per_atom"
    ] = np.nan
    for metric in ("fmax", "frms"):
        failed.loc[
            failed["sid"].eq("sel-b1"),
            f"{failed_model}_{metric}_ev_per_a",
        ] = np.nan

    scores = protocol.construct_committee_scores(
        failed,
        cutoffs=cutoffs,
        expected_stage="formula_selection",
    )

    group_a = scores.loc[scores["rk"].eq("A")]
    group_b = scores.loc[scores["rk"].eq("B")]
    assert set(group_a["state"]) == {"KEEP"}
    assert set(group_a["abstain_reason"]) == {""}
    assert len(group_b) == 2 * len(protocol.FORMULA_NAMES)

    surviving = group_b.loc[group_b["formula"].eq(baseline_formula)]
    assert set(surviving["state"]) == {"KEEP"}
    assert set(surviving["abstain_reason"]) == {""}
    assert np.isfinite(surviving["score_ev_per_atom"]).all()

    failed_baseline = group_b.loc[group_b["formula"].eq(failed_formula)]
    assert set(failed_baseline["state"]) == {"ABSTAIN"}
    assert set(failed_baseline["abstain_reason"]) == {
        f"incomplete_group_{failed_model}_failure"
    }
    assert failed_baseline["score_ev_per_atom"].isna().all()

    committee_formulas = set(protocol.FORMULA_NAMES) - {"M1", "M5"}
    committee = group_b.loc[group_b["formula"].isin(committee_formulas)]
    assert set(committee["state"]) == {"ABSTAIN"}
    assert set(committee["abstain_reason"]) == {"incomplete_group_committee_failure"}
    assert committee["score_ev_per_atom"].isna().all()


def test_disagreement_abstention_is_row_local_after_complete_group_minima():
    calibration = pd.DataFrame(
        [
            _ready_row("cal-0", "C", "search_calibration", 0.0, 0.0),
            _ready_row("cal-1", "C", "search_calibration", 5.0, 0.0),
        ]
    )
    cutoffs = protocol.derive_disagreement_cutoffs(calibration)
    features = pd.DataFrame(
        [
            _ready_row("edge-high", "EDGE", "formula_selection", 0.0, 10.0),
            _ready_row("edge-equal", "EDGE", "formula_selection", 5.0, 0.0),
        ]
    )

    scores = protocol.construct_committee_scores(
        features,
        cutoffs=cutoffs,
        expected_stage="formula_selection",
    )
    indexed = _scores_by_formula_and_sid(scores)

    # edge-high supplies the m1 group minimum. Removing it before computing
    # gaps would incorrectly change edge-equal g1 from 5 to 0.
    equal = indexed.loc[(slice(None), "edge-equal"), :]
    assert equal["g1_ev_per_atom"].unique().tolist() == pytest.approx([5.0])
    assert equal["g5_ev_per_atom"].unique().tolist() == pytest.approx([0.0])
    assert equal["disagreement_ev_per_atom"].unique().tolist() == pytest.approx([5.0])

    for formula in ("AGREE99", "AGREE995"):
        high = indexed.loc[(formula, "edge-high")]
        at_equality = indexed.loc[(formula, "edge-equal")]
        assert high["state"] == "ABSTAIN"
        assert high["abstain_reason"] == "disagreement_above_threshold"
        assert np.isnan(high["score_ev_per_atom"])
        assert at_equality["state"] == "KEEP"
        assert at_equality["abstain_reason"] == ""
        assert at_equality["score_ev_per_atom"] == pytest.approx(2.5)

    ef_high = indexed.loc[("AGREE_EF995", "edge-high")]
    ef_at_equality = indexed.loc[("AGREE_EF995", "edge-equal")]
    assert ef_high["state"] == "ABSTAIN"
    assert ef_high["abstain_reason"] == ("energy_disagreement_above_threshold")
    assert np.isnan(ef_high["score_ev_per_atom"])
    assert ef_at_equality["state"] == "KEEP"
    assert ef_at_equality["score_ev_per_atom"] == pytest.approx(2.5)

    # Disagreement gating is exclusive to AGREE99/995.
    base_high = indexed.loc[
        (["M5", "M1", "MIN", "MEAN", "MAX", "LCB"], "edge-high"),
        :,
    ]
    assert set(base_high["state"]) == {"KEEP"}


def test_force_disagreement_gate_is_row_local_and_keeps_threshold_equality():
    calibration = pd.DataFrame(
        [
            _ready_row("cal-0", "C", "search_calibration", 0.0, 0.0),
            _ready_row(
                "cal-1",
                "C",
                "search_calibration",
                5.0,
                0.0,
                m5_fmax=5.0,
            ),
        ]
    )
    cutoffs = protocol.derive_disagreement_cutoffs(calibration)
    assert cutoffs.q995_force_ev_per_a == pytest.approx(5.0)
    features = pd.DataFrame(
        [
            _ready_row(
                "force-high",
                "FORCE",
                "formula_selection",
                0.0,
                0.0,
                m5_fmax=10.0,
            ),
            _ready_row(
                "force-equal",
                "FORCE",
                "formula_selection",
                2.0,
                1.0,
                m5_fmax=5.0,
            ),
        ]
    )

    scores = protocol.construct_committee_scores(
        features,
        cutoffs=cutoffs,
        expected_stage="formula_selection",
    )
    indexed = _scores_by_formula_and_sid(scores)

    equal = indexed.loc[("AGREE_EF995", "force-equal")]
    assert equal["g1_ev_per_atom"] == pytest.approx(2.0)
    assert equal["g5_ev_per_atom"] == pytest.approx(1.0)
    assert equal["force_disagreement_ev_per_a"] == pytest.approx(5.0)
    assert equal["state"] == "KEEP"
    assert equal["score_ev_per_atom"] == pytest.approx(1.5)

    high = indexed.loc[("AGREE_EF995", "force-high")]
    assert high["force_disagreement_ev_per_a"] == pytest.approx(10.0)
    assert high["state"] == "ABSTAIN"
    assert high["abstain_reason"] == "force_disagreement_above_threshold"
    assert np.isnan(high["score_ev_per_atom"])
    assert indexed.loc[("AGREE995", "force-high"), "state"] == "KEEP"


@pytest.mark.parametrize(
    "column,value",
    [
        ("m1_energy_ev_per_atom", np.nan),
        ("m1_energy_ev_per_atom", np.inf),
        ("m5_energy_ev_per_atom", -np.inf),
    ],
)
def test_nonfinite_energy_claimed_as_success_is_rejected(column, value):
    features = _search_calibration_features()
    features.loc[0, column] = value

    with pytest.raises(ValueError, match="finite"):
        protocol.derive_disagreement_cutoffs(features)


@pytest.mark.parametrize(
    "values",
    [
        np.asarray([True, False, True, False], dtype=bool),
        np.asarray([0.0 + 1.0j, 1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]),
    ],
    ids=("boolean", "complex"),
)
def test_boolean_or_complex_energy_columns_are_rejected(values):
    features = _search_calibration_features()
    features["m1_energy_ev_per_atom"] = values

    with pytest.raises(ValueError, match="real"):
        protocol.derive_disagreement_cutoffs(features)


@pytest.mark.parametrize(
    "column,value",
    [
        ("m1_fmax_ev_per_a", np.nan),
        ("m1_frms_ev_per_a", np.inf),
        ("m5_fmax_ev_per_a", -0.1),
        ("m5_frms_ev_per_a", -np.inf),
    ],
)
def test_invalid_force_scalar_claimed_as_success_is_rejected(column, value):
    features = _search_calibration_features()
    features.loc[0, column] = value

    with pytest.raises(ValueError, match="finite|nonnegative"):
        protocol.derive_disagreement_cutoffs(features)


@pytest.mark.parametrize(
    "values",
    [
        np.asarray([True, False, True, False], dtype=bool),
        np.asarray([0.0 + 1.0j, 1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]),
    ],
    ids=("boolean", "complex"),
)
def test_boolean_or_complex_force_columns_are_rejected(values):
    features = _search_calibration_features()
    features["m1_fmax_ev_per_a"] = values

    with pytest.raises(ValueError, match="real"):
        protocol.derive_disagreement_cutoffs(features)


@pytest.mark.parametrize(
    "committee_ok,m1_ok,m5_ok",
    ((False, True, True), (True, False, True)),
    ids=("joint_success_but_committee_false", "m1_failure_but_committee_true"),
)
def test_committee_feature_flag_must_equal_model_success_conjunction(
    committee_ok, m1_ok, m5_ok
):
    features = _search_calibration_features()
    features.loc[0, "committee_feature_ok"] = committee_ok
    features.loc[0, "m1_prediction_ok"] = m1_ok
    features.loc[0, "m5_prediction_ok"] = m5_ok

    with pytest.raises(ValueError, match="committee_feature_ok|conjunction"):
        protocol.derive_disagreement_cutoffs(features)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda table: table.drop(columns="m1_prediction_ok"), "missing"),
        (lambda table: table.drop(columns="m1_fmax_ev_per_a"), "missing"),
        (
            lambda table: pd.concat([table, table.iloc[[0]]], ignore_index=True),
            "duplicate",
        ),
        (lambda table: table.assign(stage="test"), "search_calibration"),
        (
            lambda table: table.assign(
                stage=["search_calibration", "search_calibration", "test", "test"]
            ),
            "search_calibration",
        ),
    ],
)
def test_calibration_rejects_missing_duplicate_or_stage_boundary_inputs(
    mutation, match
):
    with pytest.raises(ValueError, match=match):
        protocol.derive_disagreement_cutoffs(mutation(_search_calibration_features()))


@pytest.mark.parametrize(
    "expected_stage,actual_stage",
    [
        ("test", "test"),
        ("formula_selection", "threshold_calibration"),
        ("unknown", "formula_selection"),
    ],
)
def test_score_construction_enforces_single_development_stage_role(
    expected_stage, actual_stage
):
    cutoffs = protocol.derive_disagreement_cutoffs(_search_calibration_features())
    features = _formula_selection_features().assign(stage=actual_stage)

    with pytest.raises(ValueError, match="stage|development"):
        protocol.construct_committee_scores(
            features,
            cutoffs=cutoffs,
            expected_stage=expected_stage,
        )


def test_score_construction_rejects_mixed_stages_and_nonsearch_cutoffs():
    cutoffs = protocol.derive_disagreement_cutoffs(_search_calibration_features())
    mixed = _formula_selection_features()
    mixed.loc[0, "stage"] = "threshold_calibration"
    with pytest.raises(ValueError, match="formula_selection"):
        protocol.construct_committee_scores(
            mixed,
            cutoffs=cutoffs,
            expected_stage="formula_selection",
        )

    forged = replace(cutoffs, source_stage="formula_selection")
    with pytest.raises(ValueError, match="search_calibration"):
        protocol.construct_committee_scores(
            _formula_selection_features(),
            cutoffs=forged,
            expected_stage="formula_selection",
        )


def test_disagreement_cutoffs_cannot_be_publicly_forged():
    with pytest.raises(ValueError, match="derive_disagreement_cutoffs"):
        protocol.DisagreementCutoffs(
            q99_ev_per_atom=1.0,
            q995_ev_per_atom=2.0,
            q995_force_ev_per_a=3.0,
            eligible_row_count=10,
            source_stage="search_calibration",
        )


@pytest.mark.parametrize(
    "consumer",
    ("serialize", "construct"),
)
@pytest.mark.parametrize(
    "changes",
    [
        {"q99_ev_per_atom": 2.5},
        {"q995_ev_per_atom": 3.5},
        {"q995_force_ev_per_a": 3.5},
        {"eligible_row_count": 5},
        {"source_stage": "formula_selection"},
        {"quantile_method": "linear"},
        {"calibration_fingerprint_sha256": "0" * 64},
        {"joint_q99": 0.5},
        {"joint_reference_n": 5},
        {"joint_reference_n_rk": 1},
        {"joint_weighting": "group"},
        {"joint_ecdf_side": "left"},
        {"joint_reference_dE": (0.0, 0.0, 0.0, 4.0)},
        {"joint_reference_dFmax": (0.0, 0.0, 0.0, 4.0)},
        {"joint_reference_dFrms": (0.0, 0.0, 0.0, 1.0)},
        {"joint_reference_sha256": "0" * 64},
    ],
    ids=(
        "q99",
        "q995",
        "q995_force",
        "n",
        "source",
        "method",
        "fingerprint",
        "joint_q99",
        "joint_n",
        "joint_n_rk",
        "joint_weighting",
        "joint_side",
        "joint_dE",
        "joint_dFmax",
        "joint_dFrms",
        "joint_hash",
    ),
)
def test_registered_cutoff_fields_are_bound_at_both_consumers(consumer, changes):
    cutoffs = protocol.derive_disagreement_cutoffs(_search_calibration_features())
    tampered = replace(cutoffs, **changes)

    with pytest.raises(ValueError, match="cutoff|search_calibration|method"):
        _consume_cutoffs(consumer, tampered)


def _consume_cutoffs(consumer, cutoffs):
    if consumer == "serialize":
        return protocol.serialize_formula_catalog(cutoffs)
    if consumer == "construct":
        return protocol.construct_committee_scores(
            _formula_selection_features(),
            cutoffs=cutoffs,
            expected_stage="formula_selection",
        )
    raise AssertionError(f"unsupported test consumer: {consumer}")


def _unit_q_cutoffs():
    features = pd.DataFrame(
        [
            _ready_row("unit-0", "U", "search_calibration", 0.0, 0.0),
            _ready_row("unit-1", "U", "search_calibration", 1.0, 0.0),
        ]
    )
    cutoffs = protocol.derive_disagreement_cutoffs(features)
    assert cutoffs.q99_ev_per_atom == 1.0
    assert cutoffs.q995_ev_per_atom == 1.0
    return cutoffs


def _unit_n_cutoffs():
    features = pd.DataFrame([_ready_row("one", "ONE", "search_calibration", 0.0, 0.0)])
    cutoffs = protocol.derive_disagreement_cutoffs(features)
    assert cutoffs.eligible_row_count == 1
    return cutoffs


def _unit_force_q_cutoffs():
    features = pd.DataFrame(
        [
            _ready_row("force-0", "F", "search_calibration", 0.0, 0.0),
            _ready_row(
                "force-1",
                "F",
                "search_calibration",
                0.0,
                0.0,
                m5_fmax=1.0,
            ),
        ]
    )
    cutoffs = protocol.derive_disagreement_cutoffs(features)
    assert cutoffs.q995_force_ev_per_a == 1.0
    return cutoffs


@pytest.mark.parametrize("consumer", ("serialize", "construct"))
@pytest.mark.parametrize(
    "field",
    (
        "q99_ev_per_atom",
        "q995_ev_per_atom",
        "q995_force_ev_per_a",
        "eligible_row_count",
        "joint_q99",
        "joint_reference_n",
        "joint_reference_n_rk",
    ),
    ids=(
        "q99_true",
        "q995_true",
        "q995_force_true",
        "n_true",
        "joint_q99_true",
        "joint_n_true",
        "joint_n_rk_true",
    ),
)
def test_boolean_cutoff_fields_cannot_collide_with_registered_numeric_fields(
    consumer, field
):
    if field in {
        "eligible_row_count",
        "joint_q99",
        "joint_reference_n",
        "joint_reference_n_rk",
    }:
        cutoffs = _unit_n_cutoffs()
    elif field == "q995_force_ev_per_a":
        cutoffs = _unit_force_q_cutoffs()
    else:
        cutoffs = _unit_q_cutoffs()
    tampered = replace(cutoffs, **{field: True})

    with pytest.raises(ValueError, match="real|int|boolean"):
        _consume_cutoffs(consumer, tampered)


@pytest.mark.parametrize("consumer", ("serialize", "construct"))
def test_numpy_integer_calibration_count_is_rejected_by_strict_schema(consumer):
    cutoffs = _unit_n_cutoffs()
    tampered = replace(cutoffs, eligible_row_count=np.int64(1))

    with pytest.raises(ValueError, match="int"):
        _consume_cutoffs(consumer, tampered)


@pytest.mark.parametrize("consumer", ("serialize", "construct"))
def test_recomputed_legacy_self_hash_cannot_forge_registered_origin(consumer):
    cutoffs = protocol.derive_disagreement_cutoffs(_search_calibration_features())
    legacy_payload = json.dumps(
        [
            float(cutoffs.q99_ev_per_atom),
            float(cutoffs.q995_ev_per_atom),
            int(cutoffs.eligible_row_count),
            cutoffs.source_stage,
            cutoffs.quantile_method,
        ],
        allow_nan=False,
        separators=(",", ":"),
    )
    legacy_hash = hashlib.sha256(legacy_payload.encode("utf-8")).hexdigest()

    forged = protocol.DisagreementCutoffs(
        q99_ev_per_atom=cutoffs.q99_ev_per_atom,
        q995_ev_per_atom=cutoffs.q995_ev_per_atom,
        q995_force_ev_per_a=cutoffs.q995_force_ev_per_a,
        eligible_row_count=cutoffs.eligible_row_count,
        source_stage=cutoffs.source_stage,
        quantile_method=cutoffs.quantile_method,
        calibration_fingerprint_sha256="0" * 64,
        _origin_token=legacy_hash,
    )
    with pytest.raises(ValueError, match="origin|registered"):
        _consume_cutoffs(consumer, forged)

    assert "_cutoff_proof" not in protocol.__dict__


@pytest.mark.parametrize("consumer", ("serialize", "construct"))
def test_unchanged_dataclass_replace_preserves_a_valid_cutoff_record(consumer):
    cutoffs = protocol.derive_disagreement_cutoffs(_search_calibration_features())

    copied = replace(cutoffs)

    expected = _consume_cutoffs(consumer, cutoffs)
    observed = _consume_cutoffs(consumer, copied)
    if consumer == "serialize":
        assert observed == expected
    else:
        pd.testing.assert_frame_equal(observed, expected)
