"""Contracts for one-shot NEXT25 OMatG DFT-reference CSP evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle

from ase import Atoms
import lmdb
import numpy as np
import pandas as pd
import pytest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _ReferenceOnly(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        values = {
            "atomic_numbers": np.asarray([3, 8], dtype=np.int32),
            "pos": np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            "cell": np.eye(3) * 5.0,
        }
        if key not in values:
            raise AssertionError(f"forbidden reference field: {key}")
        return values[key]

    def __iter__(self):
        raise AssertionError("reference record iteration is forbidden")

    def __len__(self) -> int:
        raise AssertionError("reference record length is not required")


def test_reference_extractor_reads_geometry_triplet_only() -> None:
    from src.next25_omatg_evaluate import extract_reference_geometry

    atoms = extract_reference_geometry(_ReferenceOnly())
    assert atoms.get_atomic_numbers().tolist() == [3, 8]
    np.testing.assert_array_equal(atoms.positions, [[0, 0, 0], [1, 1, 1]])
    np.testing.assert_array_equal(atoms.cell.array, np.eye(3) * 5)


def _fixture(tmp_path: Path) -> dict[str, Path]:
    from src.next11_geometry_only_frames import _ParsedFrame, _write_deterministic_archive
    from src.next25_apply_rule import PROTOCOL as PREDICTION_PROTOCOL
    from src.next25_omatg_compositions import PROTOCOL as COMPOSITION_PROTOCOL
    from src.next25_omatg_holdout import PROTOCOL as HOLDOUT_PROTOCOL
    from src.next25_pauling_controls import PROTOCOL as PAULING_PROTOCOL

    tmp_path.mkdir(parents=True, exist_ok=True)
    ids = [f"next25-test-{index:04d}" for index in range(6)]
    numbers = [[3, 8], [11, 17], [12, 8], [20, 9, 9], [13, 8], [14, 8, 8]]
    formula = ["LiO", "NaCl", "MgO", "CaF2", "AlO", "SiO2"]
    source_indices = [0, 1, 2, 3, 4, 5]
    cohort = tmp_path / "composition_cohort.parquet"
    pd.DataFrame(
        {
            "material_id": ids,
            "source_split": "test",
            "source_index": source_indices,
            "formula": formula,
            "reduced_formula": formula,
            "atomic_numbers_json": [json.dumps(row) for row in numbers],
            "natoms": [len(row) for row in numbers],
            "selection_key": [str(index) * 64 for index in range(6)],
            "selection_rank": range(6),
            "input_role": "composition_only",
        }
    ).to_parquet(cohort, index=False)

    test_lmdb = tmp_path / "test.lmdb"
    generated_frames = {}
    with (
        lmdb.Environment(str(test_lmdb), subdir=False, map_size=1 << 24, lock=False) as env,
        env.begin(write=True) as txn,
    ):
        for index, (material_id, zs) in enumerate(zip(ids, numbers, strict=True)):
            positions = np.arange(len(zs) * 3, dtype=float).reshape(len(zs), 3) * 0.2
            cell = np.eye(3) * (5.0 + index * 0.1)
            record = {
                "atomic_numbers": np.asarray(zs, dtype=np.int32),
                "pos": np.asarray(positions, dtype=np.float64),
                "cell": np.asarray(cell, dtype=np.float64),
                "band_gap": 99.0,
                "ids": f"SECRET-{index}",
            }
            txn.put(str(index).encode(), pickle.dumps(record))
            generated_frames[material_id] = _ParsedFrame(
                atoms=Atoms(zs, positions=positions + 0.01, cell=cell, pbc=True),
                dropped_comment_fields=(),
                dropped_atom_properties=(),
            )
    geometry = tmp_path / "geometry_only_frames.zip"
    _write_deterministic_archive(geometry, generated_frames)
    metadata = tmp_path / "holdout_metadata.parquet"
    pd.DataFrame(
        {
            "material_id": ids,
            "rk": "omatg_mp20_csp_linear_ode",
            "formula": formula,
            "natoms": [len(row) for row in numbers],
            "input_role": "unrelaxed_x0_geometry_only",
        }
    ).to_parquet(metadata, index=False)
    holdout_manifest = tmp_path / "holdout-MANIFEST.json"
    holdout_manifest.write_text(
        json.dumps(
            {
                "protocol": HOLDOUT_PROTOCOL,
                "input_role": "unrelaxed_x0_geometry_only",
                "all_generator_outputs_retained": True,
                "labels_opened": False,
                "endpoint_artifacts_opened": False,
                "relaxed_structures_opened": False,
                "outputs_sha256": {
                    metadata.name: _sha(metadata),
                    geometry.name: _sha(geometry),
                },
            }
        ),
        encoding="utf-8",
    )
    composition_manifest = tmp_path / "composition-MANIFEST.json"
    composition_manifest.write_text(
        json.dumps(
            {
                "protocol": COMPOSITION_PROTOCOL,
                "input_role": "composition_only",
                "reference_geometry_fields_accessed": False,
                "labels_opened": False,
                "inputs_sha256": {
                    "test_lmdb": {"path": str(test_lmdb), "sha256": _sha(test_lmdb)}
                },
                "outputs_sha256": {cohort.name: _sha(cohort)},
            }
        ),
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.parquet"
    pd.DataFrame(
        {
            "material_id": ids,
            "analytic_supported": [True, True, True, True, False, True],
            "next23_risk_score": [-1.0, 3.0, 2.5, 0.0, np.nan, 4.0],
            "reject": [False, True, True, False, False, True],
            "input_role": "unrelaxed_x0_geometry_only",
        }
    ).to_parquet(predictions, index=False)
    prediction_manifest = tmp_path / "prediction-MANIFEST.json"
    prediction_manifest.write_text(
        json.dumps(
            {
                "protocol": PREDICTION_PROTOCOL,
                "frozen_at_utc": "2026-01-01T00:00:00+00:00",
                "blind_labels_opened": False,
                "endpoint_fields_read": False,
                "thresholds_refit": False,
                "formula_or_parameters_changed": False,
                "outputs_sha256": {predictions.name: _sha(predictions)},
            }
        ),
        encoding="utf-8",
    )
    pauling = tmp_path / "pauling.parquet"
    decisions = ["KEEP", "REJECT", "ABSTAIN", "KEEP", "KEEP", "REJECT"]
    pd.DataFrame(
        {
            "material_id": ids,
            "rk": "omatg_mp20_csp_linear_ode",
            "formula": formula,
            "natoms": [len(row) for row in numbers],
            "pauling_feature_error": None,
            "pauling_p2_value": np.nan,
            "pauling_p2_decision": decisions,
            "pauling_p3_value": np.nan,
            "pauling_p3_decision": decisions,
            "pauling_p4_value": np.nan,
            "pauling_p4_decision": decisions,
            "pauling_p5_value": np.nan,
            "pauling_p5_decision": decisions,
            "pauling_p2_p5_decision": decisions,
        }
    ).to_parquet(pauling, index=False)
    pauling_manifest = tmp_path / "pauling-MANIFEST.json"
    pauling_manifest.write_text(
        json.dumps(
            {
                "protocol": PAULING_PROTOCOL,
                "labels_opened": False,
                "endpoint_artifacts_opened": False,
                "thresholds_refit": False,
                "rules_changed": False,
                "outputs_sha256": {pauling.name: _sha(pauling)},
            }
        ),
        encoding="utf-8",
    )
    return {
        "composition_cohort": cohort,
        "composition_manifest": composition_manifest,
        "test_lmdb": test_lmdb,
        "metadata": metadata,
        "geometry": geometry,
        "holdout_manifest": holdout_manifest,
        "predictions": predictions,
        "prediction_manifest": prediction_manifest,
        "pauling": pauling,
        "pauling_manifest": pauling_manifest,
    }


def _evaluate(paths: dict[str, Path], output: Path, match_function):
    from src.next25_omatg_evaluate import evaluate_omatg_csp

    return evaluate_omatg_csp(
        composition_cohort_path=paths["composition_cohort"],
        composition_manifest_path=paths["composition_manifest"],
        test_lmdb_path=paths["test_lmdb"],
        cohort_metadata_path=paths["metadata"],
        geometry_zip_path=paths["geometry"],
        holdout_manifest_path=paths["holdout_manifest"],
        predictions_path=paths["predictions"],
        prediction_manifest_path=paths["prediction_manifest"],
        pauling_controls_path=paths["pauling"],
        pauling_manifest_path=paths["pauling_manifest"],
        output_dir=output,
        require_formal_inputs=False,
        match_function=match_function,
    )


def test_evaluator_opens_references_after_freeze_and_computes_exact_metrics(
    tmp_path: Path,
) -> None:
    from src import next25_omatg_evaluate as module

    paths = _fixture(tmp_path)
    outcomes = iter([0.1, None, None, 0.2, 0.15, None])
    result = _evaluate(paths, tmp_path / "evaluation", lambda _gen, _ref: next(outcomes))
    analytic = result["analytic_rule"]
    assert analytic["rows"] == 6
    assert analytic["supported"] == 5
    assert analytic["rejected"] == 3
    assert analytic["reference_matches"] == 3
    assert analytic["matched_kept"] == 3
    assert analytic["nonmatches_rejected"] == 3
    assert analytic["coverage"] == pytest.approx(5 / 6)
    assert analytic["match_protection_recall"] == 1.0
    assert analytic["nonmatch_rejection_precision"] == 1.0
    assert analytic["savings"] == 0.5
    assert result["endpoint"]["matching_tolerances"] == {
        "ltol": 0.3,
        "stol": 0.5,
        "angle_tol_degrees": 10.0,
    }
    assert result["endpoint"]["reference_matches"] == 3
    assert result["claim_boundary"]["nonmatch_is_thermodynamic_instability"] is False
    assert result["thresholds_refit_after_reference_opening"] is False
    diagnostics = result["secondary_diagnostics"]
    assert diagnostics["atom_count_subgroups"]["2-5"]["rows"] == 6
    assert diagnostics["element_count_subgroups"]["binary"]["rows"] == 6
    assert sum(row["rows"] for row in diagnostics["decision_disagreement"]) == 6
    assert sum(row["rows"] for row in diagnostics["risk_score_calibration"]) == 5
    assert datetime.fromisoformat(result["opened_at_utc"]) > datetime.fromisoformat(
        "2026-01-01T00:00:00+00:00"
    )
    joined = pd.read_parquet(tmp_path / "evaluation" / module.JOINED_NAME)
    assert joined.reference_match.tolist() == [True, False, False, True, True, False]
    assert joined.corrected_rmsd.tolist() == pytest.approx([0.1, 0.5, 0.5, 0.2, 0.15, 0.5])
    manifest = json.loads(
        (tmp_path / "evaluation" / module.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert (
        "docs/plans/2026-08-03-next25-omatg-blind-csp-design.md"
        in manifest["executed_source_sha256"]
    )


def test_evaluator_refuses_unfrozen_prediction_overwrite_and_refit_cli(tmp_path: Path) -> None:
    from src.next25_omatg_evaluate import main

    paths = _fixture(tmp_path)
    manifest = json.loads(paths["prediction_manifest"].read_text())
    manifest["blind_labels_opened"] = True
    paths["prediction_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    called = False

    def matcher(_gen, _ref):
        nonlocal called
        called = True
        return None

    with pytest.raises(ValueError, match="frozen"):
        _evaluate(paths, tmp_path / "bad", matcher)
    assert called is False

    paths = _fixture(tmp_path / "again")
    output = tmp_path / "evaluation"
    _evaluate(paths, output, lambda _gen, _ref: None)
    with pytest.raises(FileExistsError):
        _evaluate(paths, output, lambda _gen, _ref: None)
    for forbidden in ("--refit", "--threshold", "--formula", "--search"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2


def test_formal_evaluator_requires_the_frozen_composition_and_eligible_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import next25_omatg_evaluate as module

    paths = _fixture(tmp_path)
    monkeypatch.setattr(module, "FORMAL_TEST_LMDB_SHA256", _sha(paths["test_lmdb"]))
    monkeypatch.setattr(
        module,
        "FORMAL_COMPOSITION_COHORT_SHA256",
        _sha(paths["composition_cohort"]),
    )
    monkeypatch.setattr(
        module,
        "FORMAL_COMPOSITION_MANIFEST_SHA256",
        _sha(paths["composition_manifest"]),
    )
    with pytest.raises(ValueError, match="production-eligible"):
        module.evaluate_omatg_csp(
            composition_cohort_path=paths["composition_cohort"],
            composition_manifest_path=paths["composition_manifest"],
            test_lmdb_path=paths["test_lmdb"],
            cohort_metadata_path=paths["metadata"],
            geometry_zip_path=paths["geometry"],
            holdout_manifest_path=paths["holdout_manifest"],
            predictions_path=paths["predictions"],
            prediction_manifest_path=paths["prediction_manifest"],
            pauling_controls_path=paths["pauling"],
            pauling_manifest_path=paths["pauling_manifest"],
            output_dir=tmp_path / "formal-evaluation",
            require_formal_inputs=True,
            match_function=lambda _generated, _reference: None,
        )


def test_default_structure_matcher_matches_identical_structure() -> None:
    from src.next25_omatg_evaluate import default_match_pair

    atoms = Atoms("LiO", positions=[[0, 0, 0], [1, 1, 1]], cell=[5, 5, 5], pbc=True)
    assert default_match_pair(atoms, atoms.copy()) == pytest.approx(0.0, abs=1e-12)
