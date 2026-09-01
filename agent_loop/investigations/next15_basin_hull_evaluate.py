"""Evaluate the sealed NEXT15 Basin-Hull rule on previously opened WBM labels."""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.next13d_acsc_dft_pairs import _json_bytes, _sha256_file, _strict_json
from src.next14_wbm_evaluate import (
    DECISIONS,
    PRIVATE_JOINED_NAME as NEXT14_JOINED_NAME,
    PROTOCOL as NEXT14_PROTOCOL,
    _bootstrap_difference,
    _complete_superiority,
    _method_metrics,
)
from src.next14_wbm_holdout import _publish_directory_no_replace
from src.next15_basin_hull import (
    BASIN_HULL_THRESHOLD_EV_PER_ATOM,
    PROTOCOL as FEATURE_PROTOCOL,
    basin_hull_decision,
)


PROTOCOL = "2026-08-02-next15-wbm-basin-hull-retrospective-v1"
RESULT_NAME = "NEXT15_WBM_BASIN_HULL_RETROSPECTIVE.json"
PRIVATE_JOINED_NAME = "joined_basin_hull_predictions_labels.parquet"
MANIFEST_NAME = "MANIFEST.json"
FROZEN_BOOTSTRAP_REPS = 10_000
FROZEN_BOOTSTRAP_SEED = 20260815
FROZEN_FORMAL_SHA256: Mapping[str, str] = {
    "features": "57e2684836c14273f431fab29a47ab45d7a87c3714774f755a9fbefcf13d895b",
    "features_manifest": "1c928de3f938787ce012e18eef15c23571cbb2f7f0f7008c62622f87890d7915",
    "next14_joined": "e3c4c5ef2761f675308d407731699f183ca736dce31df072abb8f776f3c4d604",
    "next14_private_manifest": "0a84250b0f0c1e28cf401b7d05a9a27c3b93b4d346ec63f4d39a9622324f9456",
}


def _validate_feature_artifact(
    features_path: Path, manifest_path: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    data = features_path.read_bytes()
    manifest = _strict_json(manifest_path.read_bytes(), role="NEXT15 feature manifest")
    if manifest.get("protocol") != FEATURE_PROTOCOL:
        raise ValueError("NEXT15 feature protocol differs")
    if manifest.get("thresholds_refit") is not False:
        raise ValueError("NEXT15 threshold was refit")
    if manifest.get("wbm_endpoint_bytes_read_by_execution") is not False:
        raise ValueError("NEXT15 feature execution read WBM endpoints")
    expected_rule = {
        "formula": "B64 = E_MatterSim_relaxed/N - E_raw_MP_hull(composition)",
        "comparison": ">=",
        "threshold_ev_per_atom": BASIN_HULL_THRESHOLD_EV_PER_ATOM,
        "failure_policy": "ABSTAIN",
    }
    rule = manifest.get("rule")
    if not isinstance(rule, Mapping) or any(rule.get(key) != value for key, value in expected_rule.items()):
        raise ValueError("NEXT15 sealed rule differs")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(features_path.name) != hashlib.sha256(data).hexdigest():
        raise ValueError("NEXT15 feature hash differs from manifest")
    table = pd.read_parquet(io.BytesIO(data))
    required = {
        "material_id", "rk", "supported", "error", "capped_at_max_steps",
        "basin_hull_score_ev_per_atom", "basin_hull_decision",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"NEXT15 feature table lacks columns: {sorted(missing)}")
    table = table.loc[:, sorted(required)].copy()
    table["material_id"] = table["material_id"].astype(str)
    if table["material_id"].isna().any() or table["material_id"].duplicated().any():
        raise ValueError("NEXT15 feature material IDs must be unique")
    if not set(table["basin_hull_decision"].astype(str)).issubset(DECISIONS):
        raise ValueError("NEXT15 decision is invalid")
    expected = [
        basin_hull_decision(score, supported=bool(supported))
        for score, supported in zip(
            table["basin_hull_score_ev_per_atom"], table["supported"], strict=True
        )
    ]
    if expected != table["basin_hull_decision"].astype(str).tolist():
        raise ValueError("NEXT15 decision differs from sealed score and threshold")
    return table.sort_values("material_id", kind="stable", ignore_index=True), manifest


def _validate_next14_private_join(
    joined_path: Path, manifest_path: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    data = joined_path.read_bytes()
    manifest = _strict_json(manifest_path.read_bytes(), role="NEXT14 private manifest")
    if (
        manifest.get("protocol") != NEXT14_PROTOCOL
        or manifest.get("identifier_bearing") is not True
        or manifest.get("storage_role") != "external private prediction/label join"
    ):
        raise ValueError("NEXT14 private join contract differs")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(joined_path.name) != hashlib.sha256(data).hexdigest():
        raise ValueError("NEXT14 private join hash differs from manifest")
    table = pd.read_parquet(io.BytesIO(data))
    required = {
        "material_id", "formula_key", "pauling_p2_p5_decision",
        "e_above_hull_mp2020_corrected_ppd_mp",
        "site_stats_fingerprint_init_final_norm_diff", "stable",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"NEXT14 private join lacks columns: {sorted(missing)}")
    table = table.loc[:, sorted(required)].copy()
    table["material_id"] = table["material_id"].astype(str)
    if table["material_id"].isna().any() or table["material_id"].duplicated().any():
        raise ValueError("NEXT14 private join material IDs must be unique")
    if not set(table["pauling_p2_p5_decision"].astype(str)).issubset(DECISIONS):
        raise ValueError("Pauling decision is invalid")
    energy = pd.to_numeric(
        table["e_above_hull_mp2020_corrected_ppd_mp"], errors="coerce"
    ).to_numpy(float)
    stable_expected = np.isfinite(energy) & (energy <= 0.0)
    if not np.array_equal(table["stable"].astype(bool).to_numpy(), stable_expected):
        raise ValueError("WBM stable label differs from frozen hull definition")
    return table.sort_values("material_id", kind="stable", ignore_index=True), manifest


def _binary_auc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    finite = np.isfinite(scores)
    scores = scores[finite]
    positive = positive[finite]
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(scores).rank(method="average").to_numpy(float)
    value = (ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(value)


def _score_diagnostics(joined: pd.DataFrame) -> dict[str, object]:
    score = joined["basin_hull_score_ev_per_atom"].to_numpy(float)
    energy = joined["e_above_hull_mp2020_corrected_ppd_mp"].to_numpy(float)
    finite = np.isfinite(score) & np.isfinite(energy)
    correlation = None
    if int(finite.sum()) >= 2 and np.std(score[finite]) > 0.0 and np.std(energy[finite]) > 0.0:
        correlation = float(np.corrcoef(score[finite], energy[finite])[0, 1])
    return {
        "finite_pairs": int(finite.sum()),
        "pearson_score_vs_corrected_dft_hull": correlation,
        "roc_auc_unstable_gt_0": _binary_auc(score, np.isfinite(energy) & (energy > 0.0)),
        "roc_auc_high_energy_ge_0_20": _binary_auc(
            score, np.isfinite(energy) & (energy >= 0.20)
        ),
    }


def evaluate_basin_hull_retrospective(
    *,
    features_path: Path,
    features_manifest_path: Path,
    next14_joined_path: Path,
    next14_private_manifest_path: Path,
    private_output_dir: Path,
    aggregate_output_dir: Path,
    bootstrap_reps: int = FROZEN_BOOTSTRAP_REPS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Join a sealed, endpoint-free execution to already-opened WBM outcomes."""

    private_target = Path(private_output_dir).resolve()
    aggregate_target = Path(aggregate_output_dir).resolve()
    if private_target == aggregate_target:
        raise ValueError("private and aggregate outputs must be separated")
    for target in (private_target, aggregate_target):
        if os.path.lexists(target):
            raise FileExistsError(f"refusing existing output: {target}")
    if type(bootstrap_reps) is not int or bootstrap_reps <= 0:
        raise ValueError("bootstrap_reps must be a positive exact integer")
    paths = {
        "features": Path(features_path).resolve(),
        "features_manifest": Path(features_manifest_path).resolve(),
        "next14_joined": Path(next14_joined_path).resolve(),
        "next14_private_manifest": Path(next14_private_manifest_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    input_hashes = {role: _sha256_file(path) for role, path in paths.items()}
    if require_formal_inputs:
        if input_hashes != dict(FROZEN_FORMAL_SHA256):
            raise ValueError("formal NEXT15 retrospective inputs differ")
        if bootstrap_reps != FROZEN_BOOTSTRAP_REPS:
            raise ValueError("formal NEXT15 bootstrap repetitions differ")

    features, _ = _validate_feature_artifact(paths["features"], paths["features_manifest"])
    labels, _ = _validate_next14_private_join(
        paths["next14_joined"], paths["next14_private_manifest"]
    )
    if set(features["material_id"]) != set(labels["material_id"]):
        raise ValueError("NEXT15 features and NEXT14 labels cover different material IDs")
    joined = labels.merge(features, on="material_id", validate="one_to_one")
    if not (joined["rk"].astype(str) == joined["formula_key"].astype(str)).all():
        raise ValueError("NEXT15 reduced compositions differ from WBM labels")

    method_columns = {
        "pauling_p2_p5": "pauling_p2_p5_decision",
        "next15_basin_hull": "basin_hull_decision",
    }
    methods = {
        name: _method_metrics(joined, column) for name, column in method_columns.items()
    }
    comparison = _bootstrap_difference(
        joined,
        method_columns["next15_basin_hull"],
        method_columns["pauling_p2_p5"],
        reps=bootstrap_reps,
        seed=FROZEN_BOOTSTRAP_SEED,
    )
    complete, clauses = _complete_superiority(
        methods["next15_basin_hull"], methods["pauling_p2_p5"], comparison
    )
    comparison["superiority_clauses"] = clauses
    comparison["complete_superiority_over_pauling"] = complete
    energy = joined["e_above_hull_mp2020_corrected_ppd_mp"].to_numpy(float)
    capped = joined["capped_at_max_steps"].astype(bool).to_numpy()
    result: dict[str, object] = {
        "protocol": PROTOCOL,
        "evidence_role": "WBM external-source retrospective after labels were opened by NEXT14",
        "labels_previously_opened": True,
        "fresh_lockbox": False,
        "feature_execution_read_wbm_endpoints": False,
        "thresholds_refit": False,
        "fixed_rule": {
            "formula": "B64 = E_MatterSim_relaxed/N - E_raw_MP_hull(composition)",
            "comparison": ">=",
            "threshold_ev_per_atom": BASIN_HULL_THRESHOLD_EV_PER_ATOM,
        },
        "counts": {
            "rows": len(joined),
            "composition_groups": int(joined["formula_key"].nunique()),
            "stable_rows": int(joined["stable"].sum()),
            "high_energy_rows": int((np.isfinite(energy) & (energy >= 0.20)).sum()),
            "capped_at_max_steps": int(capped.sum()),
            "capped_rejected": int(
                (capped & joined["basin_hull_decision"].astype(str).eq("REJECT").to_numpy()).sum()
            ),
        },
        "methods": methods,
        "comparisons_to_pauling": {"next15_basin_hull": comparison},
        "score_diagnostics": _score_diagnostics(joined),
        "external_source_retrospective_support": complete,
        "scientific_improvement_claim": False,
        "interpretation_guard": (
            "Passing is retrospective support only: labels were already visible, the MP reference contains prior DFT, "
            "and this is not proof of universal DFT equivalence."
        ),
    }

    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next15_basin_hull_evaluate.py": Path(__file__).resolve(),
        "src/next15_basin_hull.py": repository_root / "src/next15_basin_hull.py",
        "src/next14_wbm_evaluate.py": repository_root / "src/next14_wbm_evaluate.py",
    }
    source_hashes = {relative: _sha256_file(path) for relative, path in source_paths.items()}
    private_manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "identifier_bearing": True,
        "storage_role": "external private NEXT15 prediction/label join",
        "inputs_sha256": input_hashes,
        "executed_source_sha256": source_hashes,
    }
    aggregate_manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "identifier_bearing": False,
        "labels_previously_opened": True,
        "fresh_lockbox": False,
        "thresholds_refit": False,
        "inputs_sha256": input_hashes,
        "executed_source_sha256": source_hashes,
        "private_output_path": str(private_target),
        "scientific_improvement_claim": False,
    }

    private_target.parent.mkdir(parents=True, exist_ok=True)
    private_staging = Path(
        tempfile.mkdtemp(prefix=f".{private_target.name}.staging-", dir=private_target.parent)
    )
    try:
        joined_path = private_staging / PRIVATE_JOINED_NAME
        joined.to_parquet(joined_path, index=False)
        private_manifest["outputs_sha256"] = {PRIVATE_JOINED_NAME: _sha256_file(joined_path)}
        (private_staging / MANIFEST_NAME).write_bytes(_json_bytes(private_manifest))
        _publish_directory_no_replace(private_staging, private_target)
    except Exception:
        shutil.rmtree(private_staging, ignore_errors=True)
        raise

    aggregate_target.parent.mkdir(parents=True, exist_ok=True)
    aggregate_staging = Path(
        tempfile.mkdtemp(prefix=f".{aggregate_target.name}.staging-", dir=aggregate_target.parent)
    )
    try:
        result_path = aggregate_staging / RESULT_NAME
        result_path.write_bytes(_json_bytes(result))
        aggregate_manifest["outputs_sha256"] = {RESULT_NAME: _sha256_file(result_path)}
        aggregate_manifest["private_outputs_sha256"] = private_manifest["outputs_sha256"]
        (aggregate_staging / MANIFEST_NAME).write_bytes(_json_bytes(aggregate_manifest))
        _publish_directory_no_replace(aggregate_staging, aggregate_target)
    except Exception:
        shutil.rmtree(aggregate_staging, ignore_errors=True)
        raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--features-manifest", required=True, type=Path)
    parser.add_argument("--next14-joined", required=True, type=Path)
    parser.add_argument("--next14-private-manifest", required=True, type=Path)
    parser.add_argument("--private-output-dir", required=True, type=Path)
    parser.add_argument("--aggregate-output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    evaluate_basin_hull_retrospective(
        features_path=arguments.features,
        features_manifest_path=arguments.features_manifest,
        next14_joined_path=arguments.next14_joined,
        next14_private_manifest_path=arguments.next14_private_manifest,
        private_output_dir=arguments.private_output_dir,
        aggregate_output_dir=arguments.aggregate_output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
