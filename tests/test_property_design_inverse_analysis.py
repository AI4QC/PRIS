from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.property_design_20260821.inverse_analysis import (
    EVALUATION_PROTOCOL_VERSION,
    run_analysis,
    select_typical_examples,
    validate_evaluation_inputs,
)


N_FORMAL = 1024

ALL_VERDICT_COLUMNS = (
    "rung_L1_verdict",
    "rung_L1_prime_verdict",
    "rung_L2_verdict",
    "rung_L3_verdict",
    "rung_L4_verdict",
    "pred_D1_735_verdict",
    "pred_D1_804_verdict",
    "pred_D2_verdict",
    "pred_D3_verdict",
    "pred_D4_verdict",
    "pred_D5_verdict",
    "pred_D6_verdict",
    "pred_D7_verdict",
    "pred_D8_verdict",
    "pred_D7_symprec_0p1_verdict",
    "distance_0.5_verdict",
    "distance_0.7_verdict",
)


def _features(**updates: float) -> str:
    values = {
        "bl_min": 0.90,
        "bl_mean": 1.00,
        "cn_an_mean": 4.00,
        "madz_range": 10.00,
        "mad_max": 5.00,
        "frac_like_bonds": 0.00,
        "fi": 0.60,
        "wyckoff_econ": 0.50,
        "bv_rel_mean": 0.10,
    }
    values.update(updates)
    return json.dumps(values, sort_keys=True)


def _formal_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(N_FORMAL):
        row: dict[str, object] = {
            "candidate_id": f"candidate_{index:04d}",
            "archive_member": f"candidate_{index:04d}.cif",
            "source_shard": f"/formal/shard_{index // 128:02d}",
            "source_member": f"gen_{index % 128}.cif",
            "source_manifest_sha256": f"{index // 128 + 100:064x}",
            "source_seed": 20260830 + index // 128,
            "cif_sha256": f"{index + N_FORMAL:064x}",
            "structure_sha256": f"{index:064x}",
            "formula": "Si",
            "num_sites": 2,
            "volume_a3": 40.0,
            "charge_assignment_route": "failed",
            "charge_assignment_values_json": "null",
            "feature_error": None,
            "minimum_pair_distance_a": 0.80,
            "features_json": _features(),
            "predicates_json": json.dumps({}, sort_keys=True),
            "wyckoff_econ_symprec_0p1": 0.50,
            "fit_valid": True,
            "fit_reason": "ok",
            "clamped_bulk_modulus_proxy_gpa": 100.0,
            "equilibrium_volume_a3": 40.0,
            "fit_r2": 0.999,
            "volume_factors_json": "[0.96,0.98,1.0,1.02,1.04]",
            "volumes_a3_json": "[38.4,39.2,40.0,40.8,41.6]",
            "energies_ev_json": "[0.2,0.05,0.0,0.05,0.2]",
            "geometry_state": "raw_unrelaxed",
            "symmetry_relaxation_applied": False,
        }
        row.update({column: "pass" for column in ALL_VERDICT_COLUMNS})
        rows.append(row)

    frame = pd.DataFrame(rows)

    # Robust D7 violation examples.  Both match the Ir-Os-Ru preference, so
    # the frozen proxy-descending ordering must select candidate_0001.
    frame.loc[0, ["formula", "clamped_bulk_modulus_proxy_gpa"]] = [
        "Ir5Os6Ru",
        500.0,
    ]
    frame.loc[1, ["formula", "clamped_bulk_modulus_proxy_gpa"]] = [
        "Os2Ir2Ru",
        510.0,
    ]
    for index in (0, 1):
        frame.loc[
            index,
            ["pred_D7_verdict", "pred_D7_symprec_0p1_verdict", "rung_L4_verdict"],
        ] = ["reject", "reject", "reject"]
        frame.loc[index, "features_json"] = _features(wyckoff_econ=0.90)
        frame.loc[index, "wyckoff_econ_symprec_0p1"] = 0.88

    # L1 demonstrates that no-verdict remains in the queue rather than being
    # collapsed into either a pass or an explicit violation.
    frame.loc[0, "rung_L1_verdict"] = "reject"
    frame.loc[1, "rung_L1_verdict"] = "no verdict"

    # Robust D7 pass example.  A non-preferred candidate has a higher proxy,
    # the Re-Ir preference must still win before frozen numeric sorting.
    frame.loc[2, ["formula", "clamped_bulk_modulus_proxy_gpa"]] = ["Re2Ir", 480.0]
    frame.loc[3, ["formula", "clamped_bulk_modulus_proxy_gpa"]] = ["C", 550.0]

    # Charge-resolved L2 violation with the numerical D4 evidence carried into
    # the examples table.
    frame.loc[4, ["formula", "clamped_bulk_modulus_proxy_gpa"]] = ["B2CN2", 470.0]
    frame.loc[4, "num_sites"] = 5
    frame.loc[4, "charge_assignment_route"] = "integer"
    frame.loc[4, "charge_assignment_values_json"] = "[3.0,3.0,0.0,-3.0,-3.0]"
    frame.loc[4, ["rung_L2_verdict", "rung_L4_verdict", "pred_D4_verdict"]] = [
        "reject",
        "reject",
        "reject",
    ]
    frame.loc[4, "features_json"] = _features(madz_range=45.0)

    # Invalid fits never enter a high-proxy denominator, even if a stale
    # numeric proxy happens to be present.
    frame.loc[5, ["fit_valid", "clamped_bulk_modulus_proxy_gpa"]] = [False, 999.0]
    return frame


def _summary(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "source": "mattergen",
        "cohort_status": "formal",
        "n_generated": int(len(frame)),
        "n_unique_structure_hashes": int(frame.structure_sha256.nunique()),
        "fit": {"valid_count": int(frame.fit_valid.sum())},
        "input_archive": "/formal/generated_crystals_cif.zip",
        "input_archive_sha256": "c" * 64,
        "input_manifest": "/formal/aggregate_manifest.json",
        "input_manifest_sha256": "d" * 64,
        "aggregate_protocol_version": "2026-08-21-mattergen-shard-aggregate-v3",
        "aggregate_protocol_sha256": "a" * 64,
        "source_protocol": {
            "condition": {"ml_bulk_modulus": 400},
            "sampling": {"guidance_factor": 2.0},
        },
        "geometry_state": "raw_unrelaxed",
        "symmetry_relaxation_applied": False,
        "pris_apply_rules_sha256": "e" * 64,
        "outputs": {"predictions": "predictions.parquet"},
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, pd.DataFrame]:
    frame = _formal_frame()
    predictions_path = tmp_path / "predictions.parquet"
    summary_path = tmp_path / "summary.json"
    frame.to_parquet(predictions_path, index=False)
    summary = _summary(frame)
    input_manifest_path = tmp_path / "aggregate_manifest.json"
    input_archive_path = tmp_path / "generated_crystals_cif.zip"
    input_archive_path.write_bytes(b"formal aggregate archive fixture\n")
    protocol_payload = json.dumps(
        summary["source_protocol"], sort_keys=True, separators=(",", ":")
    ).encode()
    summary["aggregate_protocol_sha256"] = hashlib.sha256(
        protocol_payload
    ).hexdigest()
    input_manifest = {
        "source": summary["source"],
        "protocol_version": summary["aggregate_protocol_version"],
        "protocol_sha256": summary["aggregate_protocol_sha256"],
        "source_protocol": summary["source_protocol"],
        "outputs": {
            "archive_name": input_archive_path.name,
            "archive_sha256": hashlib.sha256(
                input_archive_path.read_bytes()
            ).hexdigest(),
        },
    }
    input_manifest_path.write_text(
        json.dumps(input_manifest, indent=2, sort_keys=True) + "\n"
    )
    summary["input_manifest"] = str(input_manifest_path.resolve())
    summary["input_manifest_sha256"] = hashlib.sha256(
        input_manifest_path.read_bytes()
    ).hexdigest()
    summary["input_archive"] = str(input_archive_path.resolve())
    summary["input_archive_sha256"] = hashlib.sha256(
        input_archive_path.read_bytes()
    ).hexdigest()
    summary["outputs"]["predictions_sha256"] = hashlib.sha256(
        predictions_path.read_bytes()
    ).hexdigest()
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n")
    return predictions_path, summary_path, frame


def _args(predictions: Path, summary: Path, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        predictions=str(predictions),
        summary=str(summary),
        output_dir=str(output_dir),
    )


def test_run_analysis_writes_neutral_queue_metrics_and_auxiliary_retention(tmp_path: Path):
    predictions_path, summary_path, _ = _write_inputs(tmp_path)
    output_dir = tmp_path / "inverse_analysis"

    payload = run_analysis(_args(predictions_path, summary_path, output_dir))

    assert payload["n_generated"] == N_FORMAL
    l1 = payload["main_queue_metrics"]["L1"]
    assert l1 == {
        "verdict_column": "rung_L1_verdict",
        "generated_count": N_FORMAL,
        "pass_count": N_FORMAL - 2,
        "explicit_violation_count": 1,
        "no_verdict_count": 1,
        "retained_queue_count": N_FORMAL - 1,
        "queue_reduction_fraction": pytest.approx(1 / N_FORMAL),
        "queue_reduction_wilson_95_low": pytest.approx(
            0.0001724080656205127
        ),
        "queue_reduction_wilson_95_high": pytest.approx(
            0.005510821357131506
        ),
    }
    assert "reject_count" not in l1
    assert payload["main_queue_metrics"]["D7_symprec_0p01"][
        "explicit_violation_count"
    ] == 2
    assert payload["main_queue_metrics"]["D7_symprec_0p1"][
        "explicit_violation_count"
    ] == 2

    high200_l1 = payload["auxiliary_high_proxy_retention"]["200"]["methods"]["L1"]
    assert high200_l1["high_proxy_candidate_count"] == 5
    assert high200_l1["high_proxy_pass_count"] == 3
    assert high200_l1["high_proxy_explicit_violation_count"] == 1
    assert high200_l1["high_proxy_no_verdict_count"] == 1
    assert high200_l1["high_proxy_retained_queue_count"] == 4
    assert high200_l1["high_proxy_retention_fraction"] == pytest.approx(4 / 5)
    assert high200_l1["high_proxy_retention_wilson_95_low"] == pytest.approx(
        0.37553462976252533
    )
    assert high200_l1["high_proxy_retention_wilson_95_high"] == pytest.approx(
        0.9637758913675698
    )
    assert payload["proportion_interval_method"] == {
        "name": "Wilson score interval",
        "confidence_level": 0.95,
        "z": pytest.approx(1.959963984540054),
    }
    assert set(payload["auxiliary_high_proxy_retention"]) == {"200", "300", "400"}
    evaluation_summary = json.loads(summary_path.read_text())
    assert payload["evaluation_provenance"] == {
        key: evaluation_summary[key]
        for key in (
            "input_archive",
            "input_archive_sha256",
            "input_manifest",
            "input_manifest_sha256",
            "aggregate_protocol_version",
            "aggregate_protocol_sha256",
            "source_protocol",
            "geometry_state",
            "symmetry_relaxation_applied",
            "pris_apply_rules_sha256",
        )
    }

    assert (output_dir / "inverse_analysis.json").is_file()
    assert (output_dir / "inverse_examples.csv").is_file()
    assert json.loads((output_dir / "inverse_analysis.json").read_text()) == payload


def test_examples_preserve_mechanism_numbers_and_use_frozen_preferences(tmp_path: Path):
    predictions_path, summary_path, _ = _write_inputs(tmp_path)
    output_dir = tmp_path / "inverse_analysis"

    run_analysis(_args(predictions_path, summary_path, output_dir))
    examples = pd.read_csv(output_dir / "inverse_examples.csv")

    assert examples.example_role.tolist() == [
        "robust_D7_violation",
        "robust_D7_pass",
        "charge_resolved_L2_violation",
    ]
    assert examples.candidate_id.tolist() == [
        "candidate_0001",
        "candidate_0002",
        "candidate_0004",
    ]
    assert examples.candidate_id.is_unique
    assert examples.minimum_pair_distance_a.gt(0.7).all()
    assert examples.fit_valid.all()
    assert examples.clamped_bulk_modulus_proxy_gpa.ge(200.0).all()
    assert examples.loc[0, "feature_wyckoff_econ_symprec_0p01"] == pytest.approx(0.90)
    assert examples.loc[0, "wyckoff_econ_symprec_0p1"] == pytest.approx(0.88)
    assert examples.loc[0, "pred_D7_verdict"] == "reject"
    assert examples.loc[0, "pred_D7_symprec_0p1_verdict"] == "reject"
    assert examples.loc[2, "feature_madz_range"] == pytest.approx(45.0)
    assert examples.loc[2, "pred_D4_verdict"] == "reject"
    assert examples.loc[2, "rung_L2_verdict"] == "reject"
    assert examples.loc[2, "charge_assignment_route"] == "integer"
    assert json.loads(examples.loc[2, "charge_assignment_values_json"]) == [
        3.0,
        3.0,
        0.0,
        -3.0,
        -3.0,
    ]
    assert json.loads(examples.loc[2, "l2_violated_predicates_json"]) == ["D4"]
    trigger = json.loads(examples.loc[2, "l2_trigger_details_json"])["D4"]
    assert trigger == {
        "feature": "madz_range",
        "satisfied_if": "<=",
        "threshold": 31.45,
        "value": 45.0,
    }
    assert examples.source_seed.tolist() == [20260830, 20260830, 20260830]
    assert examples.source_member.tolist() == ["gen_1.cif", "gen_2.cif", "gen_4.cif"]
    assert examples.cohort_status.eq("formal").all()
    assert examples.selection_status.tolist() == [
        "formal_preferred",
        "formal_preferred",
        "formal_selected",
    ]
    assert examples.geometry_state.eq("raw_unrelaxed").all()
    assert (~examples.symmetry_relaxation_applied).all()
    assert json.loads(examples.loc[0, "volume_factors_json"]) == [
        0.96,
        0.98,
        1.0,
        1.02,
        1.04,
    ]


def test_analysis_is_byte_deterministic_across_new_output_directories(tmp_path: Path):
    predictions_path, summary_path, _ = _write_inputs(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    run_analysis(_args(predictions_path, summary_path, first))
    run_analysis(_args(predictions_path, summary_path, second))

    assert (first / "inverse_analysis.json").read_bytes() == (
        second / "inverse_analysis.json"
    ).read_bytes()
    assert (first / "inverse_examples.csv").read_bytes() == (
        second / "inverse_examples.csv"
    ).read_bytes()


def test_charge_resolved_example_is_kept_when_it_also_fills_a_d7_role():
    frame = _formal_frame()
    frame.loc[4, ["rung_L2_verdict", "rung_L4_verdict", "pred_D4_verdict"]] = [
        "pass",
        "pass",
        "pass",
    ]
    frame.loc[4, "charge_assignment_route"] = "failed"
    frame.loc[1, ["charge_assignment_route", "rung_L2_verdict", "pred_D4_verdict"]] = [
        "integer",
        "reject",
        "reject",
    ]
    frame.loc[1, "features_json"] = _features(
        wyckoff_econ=0.90,
        madz_range=45.0,
    )

    examples, selection = select_typical_examples(frame)

    selected = dict(zip(examples.example_role, examples.candidate_id, strict=True))
    assert selected["robust_D7_violation"] == "candidate_0001"
    assert selected["charge_resolved_L2_violation"] == "candidate_0001"
    charge_role = selection["roles"]["charge_resolved_L2_violation"]
    assert charge_role["selected"] is True
    assert charge_role["selected_candidate_id"] == "candidate_0001"
    assert charge_role["reused_candidate"] is True
    charge_example = examples.loc[
        examples.example_role.eq("charge_resolved_L2_violation")
    ].iloc[0]
    assert bool(charge_example.reused_candidate) is True


def test_example_selection_prefers_a_distinct_candidate_before_reuse():
    frame = _formal_frame()
    frame.loc[4, ["rung_L2_verdict", "rung_L4_verdict", "pred_D4_verdict"]] = [
        "pass",
        "pass",
        "pass",
    ]
    frame.loc[4, "charge_assignment_route"] = "failed"
    for index in (0, 1):
        frame.loc[
            index,
            ["charge_assignment_route", "rung_L2_verdict", "pred_D4_verdict"],
        ] = ["integer", "reject", "reject"]
        frame.loc[index, "features_json"] = _features(
            wyckoff_econ=0.90,
            madz_range=45.0,
        )

    examples, selection = select_typical_examples(frame)

    selected = dict(zip(examples.example_role, examples.candidate_id, strict=True))
    assert selected["robust_D7_violation"] == "candidate_0001"
    assert selected["charge_resolved_L2_violation"] == "candidate_0000"
    charge_role = selection["roles"]["charge_resolved_L2_violation"]
    assert charge_role["selected_candidate_id"] == "candidate_0000"
    assert charge_role["reused_candidate"] is False
    assert charge_role["composition_preference_matched"] is False
    assert selection["candidate_reuse_policy"] == (
        "after composition preference and frozen sorting, select the "
        "highest-ranked as-yet-unselected candidate. Reuse the highest-ranked "
        "candidate only when every candidate remaining after composition "
        "preference was already selected"
    )
    charge_example = examples.loc[
        examples.example_role.eq("charge_resolved_L2_violation")
    ].iloc[0]
    assert bool(charge_example.reused_candidate) is False


def test_validation_rejects_non_v2_protocol():
    frame = _formal_frame()
    summary = _summary(frame)
    summary["protocol_version"] = "2026-08-21-property-design-v1"

    with pytest.raises(ValueError, match="protocol_version"):
        validate_evaluation_inputs(frame, summary)


@pytest.mark.parametrize(
    "missing_key",
    [
        "cohort_status",
        "input_archive",
        "input_archive_sha256",
        "input_manifest",
        "input_manifest_sha256",
        "aggregate_protocol_version",
        "aggregate_protocol_sha256",
        "source_protocol",
        "geometry_state",
        "symmetry_relaxation_applied",
        "pris_apply_rules_sha256",
    ],
)
def test_validation_requires_complete_formal_provenance(missing_key: str):
    frame = _formal_frame()
    summary = _summary(frame)
    summary.pop(missing_key)

    with pytest.raises(ValueError, match=missing_key):
        validate_evaluation_inputs(frame, summary)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([3.0, 3.0, 4.0, -3.0, -3.0], "charge neutral"),
        ([1.0, 1.0, 1.0, 1.0, 1.0], "positive and negative"),
    ],
)
def test_validation_enforces_charge_assignment_invariants(values, message):
    frame = _formal_frame()
    summary = _summary(frame)
    frame.loc[4, "charge_assignment_values_json"] = json.dumps(values)

    with pytest.raises(ValueError, match=message):
        validate_evaluation_inputs(frame, summary)


def test_validation_accepts_fractional_mean_from_primary_composition_route():
    frame = _formal_frame()
    summary = _summary(frame)
    frame.loc[4, "charge_assignment_values_json"] = json.dumps(
        [-3.0, -3.0, -3.0, 4.5, 4.5]
    )

    validate_evaluation_inputs(frame, summary)


def test_analysis_documents_historical_charge_route_semantics(tmp_path: Path):
    predictions_path, summary_path, _ = _write_inputs(tmp_path)

    result = run_analysis(
        _args(predictions_path, summary_path, tmp_path / "analysis")
    )

    assert "fractional mean oxidation state" in result[
        "charge_assignment_route_semantics"
    ]["integer"]


def test_run_analysis_binds_summary_to_predictions_sha256(tmp_path: Path):
    predictions_path, summary_path, _ = _write_inputs(tmp_path)
    summary = json.loads(summary_path.read_text())
    summary["outputs"]["predictions_sha256"] = "0" * 64
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="predictions SHA256 mismatch"):
        run_analysis(_args(predictions_path, summary_path, tmp_path / "out"))


def test_run_analysis_binds_summary_to_input_manifest(tmp_path: Path):
    predictions_path, summary_path, _ = _write_inputs(tmp_path)
    summary = json.loads(summary_path.read_text())
    summary["source_protocol"]["condition"]["ml_bulk_modulus"] = 300
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="aggregate provenance mismatch"):
        run_analysis(_args(predictions_path, summary_path, tmp_path / "out"))


def test_run_analysis_binds_summary_source_to_input_manifest(tmp_path: Path):
    predictions_path, summary_path, _ = _write_inputs(tmp_path)
    summary = json.loads(summary_path.read_text())
    summary["source"] = "llm"
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="aggregate provenance mismatch"):
        run_analysis(_args(predictions_path, summary_path, tmp_path / "out"))


def test_run_analysis_binds_summary_to_aggregate_archive(tmp_path: Path):
    predictions_path, summary_path, _ = _write_inputs(tmp_path)
    summary = json.loads(summary_path.read_text())
    Path(summary["input_archive"]).write_bytes(b"tampered archive\n")

    with pytest.raises(ValueError, match="input archive SHA256 mismatch"):
        run_analysis(_args(predictions_path, summary_path, tmp_path / "out"))


def test_validation_requires_at_least_1024_exact_unique_rows():
    frame = _formal_frame().iloc[:-1].copy()
    summary = _summary(frame)

    with pytest.raises(ValueError, match="at least 1024"):
        validate_evaluation_inputs(frame, summary)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("row_count", "n_generated"),
        ("duplicate_candidate_id", "candidate_id"),
        ("duplicate_structure_hash", "structure_sha256"),
        ("reported_unique_hashes", "n_unique_structure_hashes"),
        ("fit_count", "valid_count"),
    ],
)
def test_validation_fails_closed_on_count_and_identity_mismatches(
    mutation: str,
    message: str,
):
    frame = _formal_frame()
    summary = _summary(frame)
    if mutation == "row_count":
        summary["n_generated"] = N_FORMAL + 1
    elif mutation == "duplicate_candidate_id":
        frame.loc[1, "candidate_id"] = frame.loc[0, "candidate_id"]
    elif mutation == "duplicate_structure_hash":
        frame.loc[1, "structure_sha256"] = frame.loc[0, "structure_sha256"]
        summary["n_unique_structure_hashes"] = N_FORMAL - 1
    elif mutation == "reported_unique_hashes":
        summary["n_unique_structure_hashes"] = N_FORMAL - 1
    elif mutation == "fit_count":
        summary["fit"]["valid_count"] = N_FORMAL

    with pytest.raises(ValueError, match=message):
        validate_evaluation_inputs(frame, summary)


@pytest.mark.parametrize("bad_verdict", ["violated", "unknown", "", None])
def test_validation_accepts_only_the_evaluator_verdict_vocabulary(bad_verdict: object):
    frame = _formal_frame()
    summary = _summary(frame)
    frame.loc[10, "rung_L2_verdict"] = bad_verdict

    with pytest.raises(ValueError, match="verdict vocabulary"):
        validate_evaluation_inputs(frame, summary)


def test_existing_output_directory_is_never_reused(tmp_path: Path):
    predictions_path, summary_path, _ = _write_inputs(tmp_path)
    output_dir = tmp_path / "already_exists"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="refusing to reuse"):
        run_analysis(_args(predictions_path, summary_path, output_dir))
