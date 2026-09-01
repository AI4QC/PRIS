"""Contract tests for the fixed next10 LRRC development-gate diagnostic."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.next6_elementa_protocol import (
    apply_group_threshold,
    attach_energy_labels,
    evaluate_group_triage,
)
from src.next8_mattersim_committee_protocol import (
    DEVELOPMENT_FREEZE_PROTOCOL,
    TRACKS,
    construct_committee_scores,
    derive_disagreement_cutoffs,
    serialize_formula_catalog,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _forbid_private_label_parser(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    message: str,
) -> dict[str, bool]:
    state = {"opened": False}

    def forbidden(data: bytes) -> pd.DataFrame:
        assert type(data) is bytes
        state["opened"] = True
        raise AssertionError(message)

    monkeypatch.setattr(
        module,
        "_parse_sealed_label_bytes",
        forbidden,
        raising=False,
    )
    return state


def _feature_row(
    sid: str,
    rk: str,
    stage: str,
    m1: float,
    m5: float,
) -> dict[str, object]:
    return {
        "sid": sid,
        "rk": rk,
        "material": sid,
        "stage": stage,
        "strict_x0_ok": True,
        "feature_state": "ok",
        "committee_feature_ok": True,
        "committee_feature_error": "",
        "natoms": 2,
        "m1_prediction_ok": True,
        "m1_prediction_error": "",
        "m1_energy_total_ev": 2.0 * m1,
        "m1_energy_ev_per_atom": m1,
        "m1_fmax_ev_per_a": 0.1 + m1,
        "m1_frms_ev_per_a": 0.05 + m1,
        "m5_prediction_ok": True,
        "m5_prediction_error": "",
        "m5_energy_total_ev": 2.0 * m5,
        "m5_energy_ev_per_atom": m5,
        "m5_fmax_ev_per_a": 0.1 + m5,
        "m5_frms_ev_per_a": 0.05 + m5,
    }


def _committee_features() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stage, rk, offset in (
        ("search_calibration", "search", 0.0),
        ("formula_selection", "select", 0.4),
        ("threshold_calibration", "fit", 0.8),
    ):
        for index, gap in enumerate((0.0, 0.04, 0.12, 0.28)):
            rows.append(
                _feature_row(
                    f"{rk}-{index}",
                    rk,
                    stage,
                    offset + gap + (0.01 if index == 2 else 0.0),
                    offset + gap,
                )
            )
    for rk, offset in (("gate-a", 1.2), ("gate-b", 1.8)):
        for index, gap in enumerate((0.0, 0.04, 0.12, 0.28)):
            rows.append(
                _feature_row(
                    f"{rk}-{index}",
                    rk,
                    "threshold_calibration",
                    offset + gap + (0.02 if index == 1 else 0.0),
                    offset + gap,
                )
            )
    return pd.DataFrame(rows)


def _role_assignments(features: pd.DataFrame) -> pd.DataFrame:
    rows = features.loc[features["stage"].eq("threshold_calibration"), ["sid", "rk", "stage"]].copy()
    rows["threshold_role"] = np.where(rows["rk"].eq("fit"), "threshold_fit", "development_gate")
    rows["split_rank"] = np.arange(len(rows), dtype=int)
    rows["split_key_sha256"] = rows["rk"].map(lambda value: _sha256_text(str(value)))
    rows["split_salt"] = "next8-threshold-fit-gate-v1-20260801"
    return rows


def _labels() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    energies = {
        "gate-a": (0.0, 0.02, 0.10, 0.30),
        "gate-b": (0.10, 0.0, 0.06, 0.25),
    }
    for rk, values in energies.items():
        for index, energy in enumerate(values):
            rows.append(
                {
                    "sid": f"{rk}-{index}",
                    "rk": rk,
                    "stage": "threshold_calibration",
                    "e_per_atom": energy,
                }
            )
    return pd.DataFrame(rows)


def _final_rules() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    thresholds = {
        ("primary", "M5"): 0.10,
        ("primary", "AGREE995"): 0.11,
        ("comparator", "M5"): 0.03,
        ("comparator", "AGREE995"): 0.035,
    }
    for track_name in ("primary", "comparator"):
        track = TRACKS[track_name]
        for role, formula in (("selected", "AGREE995"), ("m5_baseline", "M5")):
            records.append(
                {
                    "role": role,
                    "formula": formula,
                    "track": track_name,
                    "threshold": thresholds[(track_name, formula)],
                    "threshold_state": "finite",
                    "threshold_source_role": "threshold_fit",
                    "alpha": track.alpha,
                    "within_group": track.within_group,
                    "protected": track.protected_column,
                    "operator": "score > threshold",
                    "unsupported_decision": "ABSTAIN",
                    "calibration_n_groups": 1,
                    "calibration_order_index": 1,
                }
            )
    return records


def _frozen_protocol(features: pd.DataFrame) -> dict[str, object]:
    search = features.loc[features["stage"].eq("search_calibration")].copy()
    cutoffs = derive_disagreement_cutoffs(search)
    serialized = serialize_formula_catalog(cutoffs)
    return {
        "protocol": DEVELOPMENT_FREEZE_PROTOCOL,
        "state": "frozen",
        "catalog": {
            "serialized": serialized,
            "sha256": _sha256_text(serialized),
        },
        "cutoffs": {
            "q99_ev_per_atom": cutoffs.q99_ev_per_atom,
            "q995_ev_per_atom": cutoffs.q995_ev_per_atom,
            "q995_force_ev_per_a": cutoffs.q995_force_ev_per_a,
            "eligible_row_count": cutoffs.eligible_row_count,
            "source_stage": cutoffs.source_stage,
            "quantile_method": cutoffs.quantile_method,
            "calibration_fingerprint_sha256": cutoffs.calibration_fingerprint_sha256,
        },
        "selection": {
            "state": "selected",
            "name": "AGREE995",
            "catalog_order": 7,
            "complexity": 3,
            "cost": 6,
            "primary_dft_savings": 0.0,
        },
        "tracks": {
            name: {
                "alpha": track.alpha,
                "protected": track.protected_column,
                "protected_ev_per_atom": track.protected_ev_per_atom,
                "within_group": track.within_group,
            }
            for name, track in TRACKS.items()
        },
        "final_rules": _final_rules(),
        "split": {
            "salt": "next8-threshold-fit-gate-v1-20260801",
            "ordering": "sha256(salt+'\\0'+rk),rk",
            "threshold_fit_groups": 1,
            "development_gate_groups": 2,
        },
        "cutoff_provenance": {
            "catalog_serialization_sha256": _sha256_text(serialized),
            "feature_manifest_sha256": "0" * 64,
            "feature_sha256": "0" * 64,
            "protocol_code_sha256": _sha256(Path("src/next8_mattersim_committee_protocol.py")),
        },
        "development_artifacts_sha256": {
            name: "0" * 64
            for name in (
                "threshold_role_assignments.parquet",
                "development_frontier.parquet",
                "threshold_fit_rules.parquet",
                "development_gate_metrics.parquet",
                "PAIRED_BOOTSTRAP.json",
                "IMPROVEMENT_GATE.json",
            )
        },
    }


def _gate_features_and_cutoffs(features: pd.DataFrame, roles: pd.DataFrame):
    cutoffs = derive_disagreement_cutoffs(
        features.loc[features["stage"].eq("search_calibration")].copy()
    )
    gate_sids = set(roles.loc[roles["threshold_role"].eq("development_gate"), "sid"])
    gate = features.loc[features["sid"].isin(gate_sids)].copy()
    return gate, cutoffs


def _baseline_metrics(
    features: pd.DataFrame,
    roles: pd.DataFrame,
    labels: pd.DataFrame,
    frozen: dict[str, object],
) -> pd.DataFrame:
    gate, cutoffs = _gate_features_and_cutoffs(features, roles)
    scores = construct_committee_scores(
        gate,
        cutoffs=cutoffs,
        expected_stage="threshold_calibration",
    )
    labelled = attach_energy_labels(labels)
    joined = scores.merge(
        labelled[["sid", "rk", "delta_e", "exact_min", "near_min", "valuable", "high_energy"]],
        on=["sid", "rk"],
        how="inner",
        validate="many_to_one",
    )
    records = []
    for track_name in ("primary", "comparator"):
        track = TRACKS[track_name]
        for method, formula in (("m5_baseline", "M5"), ("selected_candidate", "AGREE995")):
            rule = next(
                row
                for row in frozen["final_rules"]
                if row["track"] == track_name and row["formula"] == formula
            )
            rows = joined.loc[joined["formula"].eq(formula)].copy()
            supported = rows["state"].eq("KEEP") & np.isfinite(rows["score_ev_per_atom"])
            rows["decision"] = apply_group_threshold(
                rows["score_ev_per_atom"].to_numpy(float),
                supported.to_numpy(bool),
                float(rule["threshold"]),
            )
            metrics = evaluate_group_triage(rows)
            records.append(
                {
                    "track": track_name,
                    "method": method,
                    "source_formula": formula,
                    "evaluation_role": "development_gate",
                    "threshold_source_role": "threshold_fit",
                    "threshold": float(rule["threshold"]),
                    "alpha": track.alpha,
                    **metrics,
                    "passes_safety_gate": False,
                }
            )
    return pd.DataFrame(records)


def _lrrc_features(features: pd.DataFrame, roles: pd.DataFrame) -> pd.DataFrame:
    gate, _ = _gate_features_and_cutoffs(features, roles)
    statuses = (
        ("ok", False),
        ("ok", True),
        ("stationary_fallback", None),
        ("abstain_unsupported_geometry", None),
        ("ok", True),
        ("ok", False),
        ("ok", False),
        ("ok", False),
    )
    rows = []
    for (_, source), (status, negative) in zip(gate.sort_values("sid").iterrows(), statuses, strict=True):
        ok = status == "ok"
        sign = -1.0 if negative is True else 1.0
        rows.append(
            {
                "sid": source["sid"],
                "rk": source["rk"],
                "stage": source["stage"],
                "threshold_role": "development_gate",
                "strict_x0_ok": True,
                "natoms": int(source["natoms"]),
                "lrrc_status": status,
                "lrrc_negative": negative,
                "d_star_angstrom": 2.0 if ok else np.nan,
                "h_angstrom": 2.0 / 256.0 if ok else np.nan,
                "kappa_h_ev_per_a2": sign if ok else np.nan,
                "kappa_h2_ev_per_a2": sign if ok else np.nan,
                "kappa_r_ev_per_a2": sign if ok else np.nan,
                "error_proxy_ev_per_a2": 0.0 if ok else np.nan,
                "u_num_ev_per_a2": sign if ok else np.nan,
                "force_call_count": 5 if ok else 1,
                "error": "" if status in {"ok", "stationary_fallback"} else "synthetic failure",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_inputs():
    features = _committee_features()
    roles = _role_assignments(features)
    labels = _labels()
    frozen = _frozen_protocol(features)
    metrics = _baseline_metrics(features, roles, labels, frozen)
    lrrc = _lrrc_features(features, roles)
    return features, roles, lrrc, labels, frozen, metrics


def test_fixed_catalog_order_and_no_refit_surface(synthetic_inputs) -> None:
    from src.next10_lrrc_gate_diagnostic import (
        FIXED_FORMULA_ORDER,
        evaluate_fixed_catalog,
        validate_catalog_order,
    )

    assert FIXED_FORMULA_ORDER == (
        "M5",
        "AGREE995",
        "M5_LRRC_OR",
        "M5_LRRC_QCRC",
        "AGREE995_LRRC_QCRC",
    )
    validate_catalog_order(FIXED_FORMULA_ORDER)
    with pytest.raises(ValueError, match="catalog"):
        validate_catalog_order(FIXED_FORMULA_ORDER + ("POSTHOC_SCAN",))

    features, roles, lrrc, labels, frozen, metrics = synthetic_inputs
    with pytest.raises(TypeError):
        evaluate_fixed_catalog(
            features,
            roles,
            lrrc,
            labels,
            frozen,
            metrics,
            refit_threshold=0.0,
        )


def test_fixed_catalog_composes_lrrc_and_applies_quota_last(synthetic_inputs) -> None:
    from src.next10_lrrc_gate_diagnostic import (
        FIXED_FORMULA_ORDER,
        evaluate_fixed_catalog,
    )

    result = evaluate_fixed_catalog(*synthetic_inputs, bootstrap_resamples=8)
    predictions = result["predictions"]
    assert list(predictions["track"].drop_duplicates()) == ["primary", "comparator"]
    for _, rows in predictions.groupby("track", sort=False):
        assert list(rows["formula"].drop_duplicates()) == list(FIXED_FORMULA_ORDER)

    primary = predictions.loc[predictions["track"].eq("primary")]
    indexed = primary.set_index(["formula", "sid"])
    assert indexed.loc[("M5_LRRC_OR", "gate-a-1"), "decision"] == "REJECT"
    assert indexed.loc[("M5_LRRC_OR", "gate-a-2"), "decision"] == indexed.loc[
        ("M5", "gate-a-2"), "decision"
    ]
    assert indexed.loc[("M5_LRRC_OR", "gate-a-3"), "decision"] == "ABSTAIN"

    for formula in ("M5_LRRC_QCRC", "AGREE995_LRRC_QCRC"):
        for rk, group in primary.loc[primary["formula"].eq(formula)].groupby("rk"):
            rejected = set(group.loc[group["decision"].eq("REJECT"), "sid"])
            prequota_rejected = set(
                group.loc[group["prequota_decision"].eq("REJECT"), "sid"]
            )
            assert rejected <= prequota_rejected


def test_lrrc_validation_is_independent_of_dataframe_index(synthetic_inputs) -> None:
    from src.next10_lrrc_gate_diagnostic import evaluate_fixed_catalog

    features, roles, lrrc, labels, frozen, metrics = synthetic_inputs
    lrrc = lrrc.copy()
    lrrc.index = np.arange(100, 100 + len(lrrc))

    result = evaluate_fixed_catalog(
        features,
        roles,
        lrrc,
        labels,
        frozen,
        metrics,
        bootstrap_resamples=4,
    )

    assert len(result["predictions"]) == 2 * 5 * len(lrrc)


def test_producer_parquet_contract_accepts_success_empty_error_and_nonstrict_sentinel(
    tmp_path: Path, synthetic_inputs
) -> None:
    from src import next10_lrrc_gate_diagnostic as evaluator
    from src.next10_lrrc_mattersim_features import _strict_output_table

    features, roles, lrrc, _labels_frame, _frozen, _metrics = synthetic_inputs
    features = features.copy()
    sid = str(lrrc.iloc[0]["sid"])
    features.loc[features["sid"].eq(sid), ["strict_x0_ok", "natoms"]] = [False, 0]
    rows = lrrc.to_dict("records")
    for row in rows:
        if row["lrrc_status"] in {"ok", "stationary_fallback"}:
            row["error"] = None
        if row["sid"] == sid:
            row.update(
                {
                    "strict_x0_ok": False,
                    "natoms": 0,
                    "lrrc_status": "abstain_unsupported_geometry",
                    "lrrc_negative": None,
                    "d_star_angstrom": np.nan,
                    "h_angstrom": np.nan,
                    "kappa_h_ev_per_a2": np.nan,
                    "kappa_h2_ev_per_a2": np.nan,
                    "kappa_r_ev_per_a2": np.nan,
                    "error_proxy_ev_per_a2": np.nan,
                    "u_num_ev_per_a2": np.nan,
                    "force_call_count": 0,
                    "error": "nonstrict_x0",
                }
            )
    producer_table = _strict_output_table(rows)
    path = tmp_path / "producer.parquet"
    producer_table.to_parquet(path, index=False)
    reloaded = pd.read_parquet(path)
    _, gate = evaluator._validate_roles(features, roles)

    validated = evaluator._validate_lrrc_features(
        reloaded,
        gate,
        formal_producer=True,
    )

    assert len(validated) == len(gate)
    assert validated.loc[validated["sid"].eq(sid), "natoms"].item() == 0


def test_formal_nonstrict_row_requires_exact_producer_sentinel(
    synthetic_inputs,
) -> None:
    from src import next10_lrrc_gate_diagnostic as evaluator

    features, roles, lrrc, _labels_frame, _frozen, _metrics = synthetic_inputs
    features = features.copy()
    lrrc = lrrc.copy()
    index = lrrc.index[
        lrrc["lrrc_status"].eq("abstain_unsupported_geometry")
    ].item()
    sid = str(lrrc.loc[index, "sid"])
    features.loc[features["sid"].eq(sid), ["strict_x0_ok", "natoms"]] = [False, 0]
    lrrc.loc[
        index,
        ["strict_x0_ok", "natoms", "force_call_count", "error"],
    ] = [False, 0, 0, "handwritten_nonstrict_failure"]
    _, gate = evaluator._validate_roles(features, roles)

    with pytest.raises(ValueError, match="nonstrict_x0"):
        evaluator._validate_lrrc_features(
            lrrc,
            gate,
            formal_producer=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("strict_natoms", 0),
        ("nonstrict_natoms", 1),
        ("ok_calls", 4),
        ("stationary_calls", 0),
        ("abstain_diagnostic", 0.1),
    ],
)
def test_lrrc_status_specific_sentinels_fail_closed(
    synthetic_inputs, field: str, value: object
) -> None:
    from src.next10_lrrc_gate_diagnostic import evaluate_fixed_catalog

    features, roles, lrrc, labels, frozen, metrics = synthetic_inputs
    features = features.copy()
    lrrc = lrrc.copy()
    if field == "strict_natoms":
        lrrc.loc[lrrc["lrrc_status"].eq("ok"), "natoms"] = value
    elif field == "nonstrict_natoms":
        index = lrrc.index[0]
        sid = lrrc.loc[index, "sid"]
        lrrc.loc[index, ["strict_x0_ok", "natoms", "lrrc_status", "lrrc_negative", "force_call_count", "error"]] = [
            False,
            value,
            "abstain_unsupported_geometry",
            None,
            0,
            "nonstrict_x0",
        ]
        lrrc.loc[index, list(lrrc.columns[8:15])] = np.nan
        features.loc[features["sid"].eq(sid), ["strict_x0_ok", "natoms"]] = [False, value]
    elif field == "ok_calls":
        lrrc.loc[lrrc["lrrc_status"].eq("ok"), "force_call_count"] = value
    elif field == "stationary_calls":
        lrrc.loc[lrrc["lrrc_status"].eq("stationary_fallback"), "force_call_count"] = value
    else:
        lrrc.loc[
            lrrc["lrrc_status"].str.startswith("abstain_"), "u_num_ev_per_a2"
        ] = value

    with pytest.raises(ValueError):
        evaluate_fixed_catalog(
            features,
            roles,
            lrrc,
            labels,
            frozen,
            metrics,
            bootstrap_resamples=4,
        )


@pytest.mark.parametrize("mutation", ["threshold", "cutoff", "catalog"])
def test_frozen_protocol_tampering_is_rejected_before_evaluation(
    synthetic_inputs, mutation: str
) -> None:
    from copy import deepcopy

    from src.next10_lrrc_gate_diagnostic import evaluate_fixed_catalog

    features, roles, lrrc, labels, frozen, metrics = synthetic_inputs
    tampered = deepcopy(frozen)
    if mutation == "threshold":
        tampered["final_rules"][0]["operator"] = "score >= threshold"
    elif mutation == "cutoff":
        tampered["cutoffs"]["q995_ev_per_atom"] += 0.01
    else:
        tampered["catalog"]["serialized"] += " "

    with pytest.raises(ValueError):
        evaluate_fixed_catalog(
            features,
            roles,
            lrrc,
            labels,
            tampered,
            metrics,
            bootstrap_resamples=4,
        )


def test_baseline_metric_mismatch_fails_before_lrrc_candidates(synthetic_inputs) -> None:
    from src.next10_lrrc_gate_diagnostic import evaluate_fixed_catalog

    features, roles, lrrc, labels, frozen, metrics = synthetic_inputs
    bad_metrics = metrics.copy()
    bad_metrics.loc[0, "n_reject"] += 1

    with pytest.raises(ValueError, match="baseline metrics"):
        evaluate_fixed_catalog(
            features,
            roles,
            lrrc,
            labels,
            frozen,
            bad_metrics,
            bootstrap_resamples=4,
        )


def test_baseline_reconstruction_has_stable_semantic_sha(synthetic_inputs) -> None:
    from src.next10_lrrc_gate_diagnostic import evaluate_fixed_catalog

    result = evaluate_fixed_catalog(*synthetic_inputs, bootstrap_resamples=4)
    features, roles, lrrc, labels, frozen, metrics = synthetic_inputs
    shuffled = evaluate_fixed_catalog(
        features.sample(frac=1.0, random_state=1).reset_index(drop=True),
        roles.sample(frac=1.0, random_state=2).reset_index(drop=True),
        lrrc.sample(frac=1.0, random_state=3).reset_index(drop=True),
        labels.sample(frac=1.0, random_state=4).reset_index(drop=True),
        frozen,
        metrics,
        bootstrap_resamples=4,
    )

    reproduction = result["baseline_reproduction"]
    assert reproduction["previous_per_row_artifact_available"] is False
    assert reproduction["aggregate_artifact_exact_match"] is True
    assert len(reproduction["per_row_semantic_sha256"]) == 64
    assert reproduction["per_row_semantic_sha256"] == shuffled[
        "baseline_reproduction"
    ]["per_row_semantic_sha256"]


def test_opening_order_hash_mismatch_never_calls_label_reader(
    tmp_path: Path,
    synthetic_inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import next10_lrrc_gate_diagnostic as module

    paths = _write_path_fixture(tmp_path, synthetic_inputs)
    lrrc_manifest = json.loads(paths["lrrc_manifest_path"].read_text())
    lrrc_manifest["outputs_sha256"][paths["lrrc_features_path"].name] = "f" * 64
    paths["lrrc_manifest_path"].write_text(json.dumps(lrrc_manifest), encoding="utf-8")
    state = _forbid_private_label_parser(
        monkeypatch,
        module,
        "labels opened before hash validation",
    )

    with pytest.raises(ValueError, match="hash"):
        module.run_gate_diagnostic(**paths, output_dir=tmp_path / "never")
    assert state["opened"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        "test_double",
        "adapter",
        "missing_frames",
        "loaded_checkpoint",
        "source_closure",
        "cuda_runtime",
        "peak_telemetry",
        "table_counts",
        "mattersim_version",
        "requested_device",
        "model_device",
        "result_device",
        "batch_calls",
        "forward_calls",
        "batch_size",
    ),
)
def test_incomplete_or_test_double_lrrc_manifest_never_opens_labels(
    tmp_path: Path,
    synthetic_inputs,
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import next10_lrrc_gate_diagnostic as module

    paths = _write_path_fixture(tmp_path, synthetic_inputs)
    manifest = json.loads(paths["lrrc_manifest_path"].read_text(encoding="utf-8"))
    if mutation == "test_double":
        manifest["production_protocol_eligible"] = False
        manifest["adapter"]["mode"] = "injected_test_double"
    elif mutation == "adapter":
        manifest["adapter"]["index_alignment_verified"] = False
    elif mutation == "missing_frames":
        manifest["inputs_sha256"].pop("frames")
    elif mutation == "loaded_checkpoint":
        manifest["predictor_loaded_checkpoint_sha256"] = "0" * 64
    elif mutation == "source_closure":
        manifest["executed_source_sha256"].pop("src/next6_mattersim_baseline.py")
    elif mutation == "cuda_runtime":
        manifest["runtime"]["cuda_available"] = False
    elif mutation == "peak_telemetry":
        manifest["execution"].pop("peak_cuda_memory_bytes")
    elif mutation == "table_counts":
        manifest["counts"]["selected_rows"] += 1
    elif mutation == "mattersim_version":
        manifest["runtime"]["mattersim_version"] = "1.2.2"
    elif mutation == "requested_device":
        manifest["adapter"]["device"] = "cuda:1"
        manifest["runtime"]["device"] = "cuda:1"
    elif mutation == "model_device":
        manifest["adapter"]["model_parameter_device"] = "cuda:1"
    elif mutation == "result_device":
        manifest["adapter"]["result_tensor_devices"] = ["cuda:1"]
    elif mutation == "batch_calls":
        manifest["counts"]["batch_predictor_calls"] = 4
        manifest["execution"]["batch_predictor_calls"] = 4
    elif mutation == "forward_calls":
        manifest["execution"]["forward_calls"] = 6
    else:
        manifest["adapter"]["batch_size"] = 1
    paths["lrrc_manifest_path"].write_text(
        json.dumps(manifest, allow_nan=False), encoding="utf-8"
    )
    state = _forbid_private_label_parser(
        monkeypatch,
        module,
        "invalid LRRC manifest opened labels",
    )

    with pytest.raises(ValueError):
        module.run_gate_diagnostic(
            **paths,
            output_dir=tmp_path / "never",
        )
    assert state["opened"] is False


@pytest.mark.parametrize(
    "status",
    ("abstain_force_failure", "abstain_invalid_force"),
)
def test_formal_rejects_impossible_per_row_prediction_failures_before_labels(
    tmp_path: Path,
    synthetic_inputs,
    status: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import next10_lrrc_gate_diagnostic as module

    paths = _write_path_fixture(tmp_path, synthetic_inputs)
    lrrc = pd.read_parquet(paths["lrrc_features_path"])
    index = lrrc.index[
        lrrc["lrrc_status"].eq("abstain_unsupported_geometry")
    ].item()
    lrrc.loc[index, "lrrc_status"] = status
    lrrc.loc[index, "error"] = "synthetic per-row prediction failure"
    lrrc.to_parquet(paths["lrrc_features_path"], index=False)

    manifest = json.loads(paths["lrrc_manifest_path"].read_text(encoding="utf-8"))
    manifest["outputs_sha256"][paths["lrrc_features_path"].name] = _sha256(
        paths["lrrc_features_path"]
    )
    paths["lrrc_manifest_path"].write_text(
        json.dumps(manifest, allow_nan=False), encoding="utf-8"
    )
    state = _forbid_private_label_parser(
        monkeypatch,
        module,
        "impossible formal row opened labels",
    )

    with pytest.raises(ValueError, match="producer-reachable"):
        module.run_gate_diagnostic(
            **paths,
            output_dir=tmp_path / "never",
        )
    assert state["opened"] is False


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("bootstrap_resamples", 8),
        ("bootstrap_seed", 1),
        ("bootstrap_batch_size", 4),
    ],
)
def test_formal_bootstrap_parameters_fail_before_label_read(
    tmp_path: Path,
    synthetic_inputs,
    name: str,
    value: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import next10_lrrc_gate_diagnostic as module

    paths = _write_path_fixture(tmp_path, synthetic_inputs)
    state = _forbid_private_label_parser(
        monkeypatch,
        module,
        "invalid formal parameters opened labels",
    )

    arguments = {name: value}
    with pytest.raises(ValueError, match="formal bootstrap"):
        module.run_gate_diagnostic(
            **paths,
            output_dir=tmp_path / "never",
            **arguments,
        )
    assert state["opened"] is False


def test_duplicate_json_key_is_rejected_before_label_read(
    tmp_path: Path,
    synthetic_inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import next10_lrrc_gate_diagnostic as module

    paths = _write_path_fixture(tmp_path, synthetic_inputs)
    raw = paths["lrrc_manifest_path"].read_text(encoding="utf-8")
    paths["lrrc_manifest_path"].write_text(
        raw.replace("{", '{"protocol":"duplicate",', 1), encoding="utf-8"
    )
    state = _forbid_private_label_parser(
        monkeypatch,
        module,
        "duplicate JSON opened labels",
    )

    with pytest.raises(ValueError, match="duplicate key"):
        module.run_gate_diagnostic(
            **paths,
            output_dir=tmp_path / "never",
        )
    assert state["opened"] is False


def test_formal_api_has_no_label_reader_injection_surface() -> None:
    import inspect

    from src.next10_lrrc_gate_diagnostic import run_gate_diagnostic

    assert "label_reader" not in inspect.signature(run_gate_diagnostic).parameters


def test_label_bytes_are_hashed_then_parsed_from_the_same_buffer_once(
    tmp_path: Path,
    synthetic_inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import next10_lrrc_gate_diagnostic as module

    paths = _write_path_fixture(tmp_path, synthetic_inputs)
    calls = 0
    observed_sha256 = ""

    def byte_reader(data: bytes) -> pd.DataFrame:
        nonlocal calls, observed_sha256
        calls += 1
        assert type(data) is bytes
        observed_sha256 = hashlib.sha256(data).hexdigest()
        return pd.read_parquet(io.BytesIO(data))

    monkeypatch.setattr(
        module,
        "_parse_sealed_label_bytes",
        byte_reader,
        raising=False,
    )

    module.run_gate_diagnostic(
        **paths,
        output_dir=tmp_path / "published",
    )
    assert calls == 1
    next8_manifest = json.loads(paths["next8_manifest_path"].read_text())
    assert observed_sha256 == next8_manifest["inputs_sha256"]["labels"]["sha256"]


def test_baseline_semantic_sha_is_frozen_before_label_parse_and_rebuilt_after(
    tmp_path: Path,
    synthetic_inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import next10_lrrc_gate_diagnostic as module

    paths = _write_path_fixture(tmp_path, synthetic_inputs)
    events: list[str] = []
    reproductions: list[dict[str, object]] = []
    original_reproduction = module._baseline_reproduction_record

    def recording_reproduction(base_rows):
        result = original_reproduction(base_rows)
        events.append("baseline")
        reproductions.append(result)
        return result

    def recording_label_parse(data: bytes) -> pd.DataFrame:
        assert type(data) is bytes
        events.append("labels")
        return pd.read_parquet(io.BytesIO(data))

    monkeypatch.setattr(
        module,
        "_baseline_reproduction_record",
        recording_reproduction,
    )
    monkeypatch.setattr(
        module,
        "_parse_sealed_label_bytes",
        recording_label_parse,
        raising=False,
    )
    output = tmp_path / "published"

    module.run_gate_diagnostic(**paths, output_dir=output)

    assert events == ["baseline", "labels", "baseline"]
    assert reproductions[0] == reproductions[1]
    catalog = json.loads((output / "FROZEN_CATALOG.json").read_text())
    assert catalog["baseline_reproduction"] == reproductions[0]


def test_label_hash_mismatch_fails_before_label_parser(
    tmp_path: Path,
    synthetic_inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import next10_lrrc_gate_diagnostic as module

    paths = _write_path_fixture(tmp_path, synthetic_inputs)
    with paths["labels_path"].open("ab") as stream:
        stream.write(b"drift")
    state = _forbid_private_label_parser(
        monkeypatch,
        module,
        "mismatched label bytes reached parser",
    )

    with pytest.raises(ValueError, match="label.*hash"):
        module.run_gate_diagnostic(
            **paths,
            output_dir=tmp_path / "never",
        )
    assert state["opened"] is False


def test_atomic_publication_is_strict_and_no_replace(
    tmp_path: Path, synthetic_inputs
) -> None:
    from src.next10_lrrc_gate_diagnostic import OUTPUT_NAMES, run_gate_diagnostic

    paths = _write_path_fixture(tmp_path, synthetic_inputs)
    output = tmp_path / "diagnostic"
    published = run_gate_diagnostic(
        **paths,
        output_dir=output,
    )
    assert published == output
    assert {path.name for path in output.iterdir()} == set(OUTPUT_NAMES)
    catalog = json.loads((output / "FROZEN_CATALOG.json").read_text())
    manifest = json.loads((output / "MANIFEST.json").read_text())
    bootstrap = json.loads((output / "PAIRED_BOOTSTRAP.json").read_text())
    assert catalog["formula_order"] == [
        "M5",
        "AGREE995",
        "M5_LRRC_OR",
        "M5_LRRC_QCRC",
        "AGREE995_LRRC_QCRC",
    ]
    assert catalog["accepts_refit_parameters"] is False
    assert manifest["scientific_improvement_claim"] is False
    assert manifest["gate_reused_after_exposure"] is True
    assert bootstrap["n_resamples"] == 20_000
    assert len(bootstrap["comparisons"]) == 6
    for name, expected in manifest["outputs_sha256"].items():
        assert _sha256(output / name) == expected

    original = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError):
        run_gate_diagnostic(
            **paths,
            output_dir=output,
        )
    assert {path.name: path.read_bytes() for path in output.iterdir()} == original


def test_input_drift_during_staging_is_caught_immediately_before_publish(
    tmp_path: Path,
    synthetic_inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import next10_lrrc_gate_diagnostic as module

    paths = _write_path_fixture(tmp_path, synthetic_inputs)
    lrrc_manifest = json.loads(paths["lrrc_manifest_path"].read_text())
    frames = Path(lrrc_manifest["inputs_sha256"]["frames"]["path"])
    original_write = module._write_exclusive
    mutated = False

    def mutating_write(path: Path, data: bytes) -> None:
        nonlocal mutated
        original_write(path, data)
        if path.name == "MANIFEST.json" and not mutated:
            mutated = True
            with frames.open("ab") as stream:
                stream.write(b"drift")

    monkeypatch.setattr(module, "_write_exclusive", mutating_write)
    output = tmp_path / "never-published"
    with pytest.raises(RuntimeError, match="changed"):
        module.run_gate_diagnostic(**paths, output_dir=output)
    assert not output.exists()


def _write_path_fixture(tmp_path: Path, synthetic_inputs) -> dict[str, Path]:
    features, roles, lrrc, labels, frozen, metrics = synthetic_inputs
    feature_dir = tmp_path / "next8-features"
    freeze_dir = tmp_path / "next8-freeze"
    lrrc_dir = tmp_path / "next10-features"
    feature_dir.mkdir()
    freeze_dir.mkdir()
    lrrc_dir.mkdir()

    feature_path = feature_dir / "mattersim_committee_features.parquet"
    roles_path = freeze_dir / "threshold_role_assignments.parquet"
    metrics_path = freeze_dir / "development_gate_metrics.parquet"
    labels_path = tmp_path / "development_labels.parquet"
    lrrc_path = lrrc_dir / "lrrc_features.parquet"
    features.to_parquet(feature_path, index=False)
    roles.to_parquet(roles_path, index=False)
    metrics.to_parquet(metrics_path, index=False)
    labels.to_parquet(labels_path, index=False)
    lrrc.to_parquet(lrrc_path, index=False)

    support_files = {}
    for name in ("frames", "metadata", "stage_assignments", "checkpoint"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("ascii"))
        support_files[name] = path
    root = Path(__file__).resolve().parents[1]
    next8_source = root / "src/next8_mattersim_committee_protocol.py"

    feature_manifest_path = feature_dir / "MANIFEST.json"
    feature_manifest = {
        "protocol": "2026-08-01-mattersim-dual-checkpoint-x0-v1",
        "mode": "development",
        "production_protocol_eligible": True,
        "evidence_role": "protocol_feature_generation",
        "outputs_sha256": {feature_path.name: _sha256(feature_path)},
        "inputs_sha256": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in support_files.items()
            if name != "checkpoint"
        },
        "executed_source_sha256": {
            "src/next8_mattersim_committee_protocol.py": _sha256(next8_source)
        },
        "checkpoints": {
            "m1": {"path": str(support_files["checkpoint"].resolve()), "sha256": _sha256(support_files["checkpoint"])},
            "m5": {"path": str(support_files["checkpoint"].resolve()), "sha256": _sha256(support_files["checkpoint"])},
        },
        "predictor_loaded_checkpoint_sha256": {
            "m1": _sha256(support_files["checkpoint"]),
            "m5": _sha256(support_files["checkpoint"]),
        },
        "integrity": {"prepublish_rehash": "passed"},
    }
    feature_manifest_path.write_text(json.dumps(feature_manifest), encoding="utf-8")

    lrrc_manifest_path = lrrc_dir / "MANIFEST.json"
    lrrc_sources = (
        "src/next10_lrrc_mattersim_features.py",
        "src/next9_lrrc.py",
        "src/next8_mattersim_committee_features.py",
        "src/next6_mattersim_baseline.py",
    )
    status = lrrc["lrrc_status"].astype(str)
    force_evaluations = int(lrrc["force_call_count"].sum())
    lrrc_manifest = {
        "protocol": "2026-08-01-next10-lrrc-mattersim-features-v1",
        "mode": "development_gate",
        "evidence_role": "label_free_lrrc_feature_generation",
        "production_protocol_eligible": True,
        "labels_opened": False,
        "scientific_improvement_claim": False,
        "selection": {
            "stage": "threshold_calibration",
            "threshold_role": "development_gate",
        },
        "adapter": {
            "mode": "builtin_indexed_mattersim",
            "device": "cuda:0",
            "batch_size": 32,
            "index_alignment": "sid_indexed_exact_one_to_one",
            "index_alignment_verified": True,
            "model_parameter_device": "cuda:0",
            "result_tensor_devices": ["cuda:0"],
            "evaluations": force_evaluations,
        },
        "predictor_loaded_checkpoint_sha256": _sha256(support_files["checkpoint"]),
        "feature_columns": list(lrrc.columns),
        "outputs_sha256": {lrrc_path.name: _sha256(lrrc_path)},
        "inputs_sha256": {
            "committee_features": {"path": str(feature_path.resolve()), "sha256": _sha256(feature_path)},
            "threshold_roles": {"path": str(roles_path.resolve()), "sha256": _sha256(roles_path)},
            "frames": {"path": str(support_files["frames"].resolve()), "sha256": _sha256(support_files["frames"])},
            "feature_manifest": {"path": str(feature_manifest_path.resolve()), "sha256": _sha256(feature_manifest_path)},
            "checkpoint": {"path": str(support_files["checkpoint"].resolve()), "sha256": _sha256(support_files["checkpoint"])},
        },
        "executed_source_sha256": {
            relative: _sha256(root / relative) for relative in lrrc_sources
        },
        "runtime": {
            "python_version": "test",
            "python_implementation": "CPython",
            "platform": "test-linux",
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "ase_version": "test",
            "mattersim_version": "1.2.3",
            "device": "cuda:0",
            "torch_version": "test",
            "cuda_available": True,
            "cuda_version": "test",
            "gpu_name": "test-gpu",
        },
        "counts": {
            "feature_rows": len(features),
            "role_assignment_rows": len(roles),
            "selected_rows": len(lrrc),
            "strict_rows": int(lrrc["strict_x0_ok"].sum()),
            "nonstrict_rows": int((~lrrc["strict_x0_ok"]).sum()),
            "ok_rows": int(status.eq("ok").sum()),
            "stationary_rows": int(status.eq("stationary_fallback").sum()),
            "abstained_rows": int(status.str.startswith("abstain_").sum()),
            "batch_predictor_calls": 5,
            "force_evaluations": force_evaluations,
        },
        "execution": {
            "batch_predictor_calls": 5,
            "forward_calls": 5,
            "peak_cuda_memory_bytes": 1024,
            "wall_time_seconds": 1.0,
        },
        "integrity": {"prepublish_rehash": "passed"},
    }
    lrrc_manifest_path.write_text(json.dumps(lrrc_manifest), encoding="utf-8")

    frozen = json.loads(json.dumps(frozen))
    frozen["cutoff_provenance"]["feature_sha256"] = _sha256(feature_path)
    frozen["cutoff_provenance"]["feature_manifest_sha256"] = _sha256(feature_manifest_path)
    for name in (
        "development_frontier.parquet",
        "threshold_fit_rules.parquet",
        "PAIRED_BOOTSTRAP.json",
        "IMPROVEMENT_GATE.json",
    ):
        path = freeze_dir / name
        path.write_bytes(name.encode("ascii"))
    artifact_paths = {
        "threshold_role_assignments.parquet": roles_path,
        "development_gate_metrics.parquet": metrics_path,
        **{
            name: freeze_dir / name
            for name in (
                "development_frontier.parquet",
                "threshold_fit_rules.parquet",
                "PAIRED_BOOTSTRAP.json",
                "IMPROVEMENT_GATE.json",
            )
        },
    }
    frozen["development_artifacts_sha256"] = {
        name: _sha256(path) for name, path in artifact_paths.items()
    }
    frozen_path = freeze_dir / "FROZEN_PROTOCOL.json"
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")

    freeze_manifest_path = freeze_dir / "MANIFEST.json"
    freeze_manifest = {
        "protocol": DEVELOPMENT_FREEZE_PROTOCOL,
        "state": "frozen",
        "inputs_sha256": {
            "features": {"path": str(feature_path.resolve()), "sha256": _sha256(feature_path)},
            "feature_manifest": {"path": str(feature_manifest_path.resolve()), "sha256": _sha256(feature_manifest_path)},
            "labels": {"path": str(labels_path.resolve()), "sha256": _sha256(labels_path)},
        },
        "outputs_sha256": {
            "FROZEN_PROTOCOL.json": _sha256(frozen_path),
            **frozen["development_artifacts_sha256"],
        },
        "executed_source_sha256": {
            "src/next8_mattersim_committee_protocol.py": _sha256(next8_source)
        },
        "integrity": {"prepublish_rehash": "passed"},
    }
    freeze_manifest_path.write_text(json.dumps(freeze_manifest), encoding="utf-8")

    return {
        "committee_features_path": feature_path,
        "committee_manifest_path": feature_manifest_path,
        "threshold_roles_path": roles_path,
        "lrrc_features_path": lrrc_path,
        "lrrc_manifest_path": lrrc_manifest_path,
        "frozen_protocol_path": frozen_path,
        "next8_manifest_path": freeze_manifest_path,
        "baseline_metrics_path": metrics_path,
        "labels_path": labels_path,
    }
