from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.next19_feature_build import _sha256
from src.next39_next23_predictions import (
    PREDICTIONS_NAME,
    apply_frozen_next23_rule,
    run_next39_predictions,
)
from src.next39_omat24_trajectory_cohort import freeze_trajectory_cohort
from tests.test_next39_omat24_trajectory_cohort import _payload, _write_db


def _write_rule(directory: Path) -> tuple[Path, Path]:
    directory.mkdir()
    rule = {
        "protocol": "2026-08-02-next23-relaxation-change-rule-freeze-v1",
        "selected_candidate": "B+E",
        "selected_terms": ["B", "E"],
        "threshold": 1.0,
        "reject_when": "supported and score >= threshold",
        "missing_policy": "fail_open_do_not_reject",
        "executable_inputs": "one_unrelaxed_structure_plus_element_tables",
        "dft_or_relaxed_input_used": False,
        "model_or_proxy_potential_used": False,
        "same_composition_candidates_used": False,
        "base_parameters": {
            "B": {
                "column": "voronoi_q0__sivr_cell_anisotropy",
                "direction": 1,
                "median": 0.0,
                "scale_iqr": 1.0,
                "source": "sivr",
            },
            "E": {
                "column": "scbv_vector_asymmetry_rms",
                "direction": 1,
                "median": 0.0,
                "scale_iqr": 1.0,
                "source": "scbve",
            },
        },
    }
    rule_path = directory / "NEXT23_FROZEN_RELAXATION_RULE.json"
    rule_path.write_text(json.dumps(rule, sort_keys=True), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    sources = {
        name: _sha256(root / name)
        for name in (
            "src/next20_valence_rigidity.py",
            "src/next21_normalized_madelung.py",
            "src/next22_bond_valence_equilibrium.py",
            "src/next23_relaxation_rule.py",
        )
    }
    manifest = {
        "protocol": rule["protocol"],
        "blind_labels_opened": False,
        "outputs_sha256": {
            rule_path.name: hashlib.sha256(rule_path.read_bytes()).hexdigest()
        },
        "executed_source_sha256": sources,
    }
    manifest_path = directory / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return rule_path, manifest_path


def test_frozen_rule_score_and_fail_open() -> None:
    rule = {
        "selected_terms": ["B", "E"],
        "threshold": 1.0,
        "base_parameters": {
            "B": {"column": "b", "direction": 1, "median": 1.0, "scale_iqr": 2.0},
            "E": {"column": "e", "direction": 1, "median": 0.0, "scale_iqr": 1.0},
        },
    }

    supported = apply_frozen_next23_rule({"b": 3.0, "e": 0.5}, rule)
    missing = apply_frozen_next23_rule({"b": 3.0}, rule)

    assert supported == {"supported": True, "score": 1.5, "reject": True}
    assert missing["supported"] is False
    assert missing["reject"] is False


def test_prediction_is_label_free_hash_locked_and_no_overwrite(tmp_path: Path) -> None:
    database = tmp_path / "tiny.aselmdb"
    _write_db(
        database,
        [
            _payload(sid="a_traj_0", parent_id="a"),
            _payload(sid="a_traj_20", parent_id="a", position=1.7),
        ],
    )
    cohort = tmp_path / "cohort"
    freeze_trajectory_cohort(
        db_path=database,
        source_name="rattled-relax",
        salt="fixed",
        minimum_latest_step=20,
        output_dir=cohort,
    )
    rule_path, rule_manifest = _write_rule(tmp_path / "rule")
    output = tmp_path / "predictions"

    manifest = run_next39_predictions(
        metadata_path=cohort / "next39_trajectory_cohort.parquet",
        frames_zip_path=cohort / "geometry_only_frames.zip",
        cohort_manifest_path=cohort / "MANIFEST.json",
        frozen_rule_path=rule_path,
        frozen_rule_manifest_path=rule_manifest,
        output_dir=output,
        term_calculator=lambda _atoms: (
            {
                "voronoi_q0__sivr_cell_anisotropy": 0.75,
                "scbv_vector_asymmetry_rms": 0.75,
            },
            None,
        ),
        pauling_feature_calculator=lambda _atoms: (
            {"p2_mean_dev": 0.0, "p3_frac_edge_face": 0.0, "p4_violate": 0.0, "p5_ok": 1.0},
            None,
        ),
    )

    table = pd.read_parquet(output / PREDICTIONS_NAME)
    assert table.next23_score.tolist() == [pytest.approx(1.5)]
    assert table.next23_reject.tolist() == [True]
    assert table.pauling_p2_p5_decision.tolist() == ["KEEP"]
    assert manifest["later_geometry_opened"] is False
    assert manifest["dft_values_read"] is False
    assert manifest["thresholds_refit"] is False
    assert manifest["production_protocol_eligible"] is False
    with pytest.raises(FileExistsError):
        run_next39_predictions(
            metadata_path=cohort / "next39_trajectory_cohort.parquet",
            frames_zip_path=cohort / "geometry_only_frames.zip",
            cohort_manifest_path=cohort / "MANIFEST.json",
            frozen_rule_path=rule_path,
            frozen_rule_manifest_path=rule_manifest,
            output_dir=output,
        )
