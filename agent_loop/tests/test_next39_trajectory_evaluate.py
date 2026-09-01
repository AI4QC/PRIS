from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.next39_omat24_trajectory_cohort import freeze_trajectory_cohort
from src.next39_next23_predictions import run_next39_predictions
from src.next39_trajectory_evaluate import (
    RESULT_NAME,
    decision_metrics,
    evaluate_next39_trajectories,
    fingerprint_distance,
)
from src.next23_relaxation_rule import ENDPOINT_COLUMN
from tests.test_next39_next23_predictions import _write_rule
from tests.test_next39_omat24_trajectory_cohort import _payload, _write_db


def test_fingerprint_distance_checks_atomic_identity() -> None:
    from ase import Atoms

    first = Atoms("NaCl", positions=[[0, 0, 0], [1, 1, 1]], cell=[3, 3, 3], pbc=True)
    later = first.copy()
    later.positions[1, 0] += 0.25
    value = fingerprint_distance(first, later, calculator=lambda atoms: atoms.positions.ravel())
    assert value == pytest.approx(0.25)
    reordered = Atoms("ClNa", positions=later.positions, cell=later.cell, pbc=True)
    with pytest.raises(ValueError, match="atomic identities"):
        fingerprint_distance(first, reordered, calculator=lambda atoms: atoms.positions.ravel())


def test_decision_metrics_reuse_frozen_next23_gates() -> None:
    endpoint = np.array([0.0] * 100 + [0.3] * 100)
    reject = np.array([False] * 100 + [True] * 100)
    metrics = decision_metrics(
        supported=np.ones(200, dtype=bool), reject=reject, endpoint=endpoint
    )
    assert metrics["passes_primary_gates"] is True
    assert metrics["rejection_precision"] == 1.0
    assert metrics["protected_recall"] == 1.0


def test_evaluation_opens_later_geometry_only_after_frozen_prediction(tmp_path: Path) -> None:
    database = tmp_path / "tiny.aselmdb"
    _write_db(
        database,
        [
            _payload(sid="a_traj_0", parent_id="a", position=1.5),
            _payload(sid="a_traj_20", parent_id="a", position=1.8),
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
    predictions = tmp_path / "predictions"
    run_next39_predictions(
        metadata_path=cohort / "next39_trajectory_cohort.parquet",
        frames_zip_path=cohort / "geometry_only_frames.zip",
        cohort_manifest_path=cohort / "MANIFEST.json",
        frozen_rule_path=rule_path,
        frozen_rule_manifest_path=rule_manifest,
        output_dir=predictions,
        term_calculator=lambda _atoms: (
            {"voronoi_q0__sivr_cell_anisotropy": 1.0, "scbv_vector_asymmetry_rms": 1.0},
            None,
        ),
        pauling_feature_calculator=lambda _atoms: (
            {"p2_mean_dev": 0.0, "p3_frac_edge_face": 0.0, "p4_violate": 0.0, "p5_ok": 1.0},
            None,
        ),
    )
    output = tmp_path / "evaluation"

    manifest = evaluate_next39_trajectories(
        db_path=database,
        metadata_path=cohort / "next39_trajectory_cohort.parquet",
        frames_zip_path=cohort / "geometry_only_frames.zip",
        cohort_manifest_path=cohort / "MANIFEST.json",
        predictions_path=predictions / "next39_next23_predictions.parquet",
        prediction_manifest_path=predictions / "MANIFEST.json",
        output_dir=output,
        fingerprint_calculator=lambda atoms: atoms.positions.ravel(),
    )

    result = json.loads((output / RESULT_NAME).read_text())
    joined = pd.read_parquet(output / "next39_joined_trajectory_evaluation.parquet")
    assert joined[ENDPOINT_COLUMN].tolist() == [pytest.approx(np.sqrt(3.0) * 0.3)]
    assert result["endpoint"]["role"] == "step0_to_latest_observed_sampled_structure_change"
    assert manifest["later_geometry_opened"] is True
    assert manifest["dft_values_read"] is False
    assert manifest["production_protocol_eligible"] is False
