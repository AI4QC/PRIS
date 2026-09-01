#!/usr/bin/env python3
"""Run a frozen short forward-stagewise path of bounded raw-x0 risk lifts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import src.next164_interior_attenuation_broad_residual as n164
import src.next213_two_signal_broad_diagnostic as n213
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


n212 = n213.n212
PROTOCOL = "2026-08-08-next214-forward-stagewise-risk-lift-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT214_FORWARD_STAGEWISE_CATALOGUE.json"
EVALUATION_NAME = "NEXT214_FORWARD_STAGEWISE_EVALUATION.json"
FORMULA_NAME = "NEXT214_FINAL_PATH_FORMULA.json"
SEARCH_NAME = "next214_forward_stagewise_candidates.parquet"
EXPECTED_DESIGN_SHA256 = (
    "2f8eee54c6af050d00ec09f76ca39d2b5751390393e23c5d347a52fe129ad630"
)
EXPECTED_START_KEY_SHA256 = (
    "927b4473b0720df41473a9def0bacaf75a6d69ae8be26ae14e700ad12834e895"
)
EXPECTED_START_SHORTFALL = 0.2725415699844472
MAX_TERMS = 8
IMPROVEMENT_TOLERANCE = 1.0e-12
SEARCH_WORKERS = 4
EXPECTED_NEXT213_SOURCE_SHA256 = (
    "4ed1d506456eee6619fd7d85974ae37c5229051f372dcfd6c5071bb80d4a9846"
)
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n213.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next213_design": n213.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next213_manifest": (
        "9dec88549c84fdf8b8d64042fa50b1c0a2646e29db7e3784468a53c7d329c2df"
    ),
    "next213_diagnostic": (
        "e4cf2851a54bde52ce43f3444bccc0775efb9bad8cccef1ce61cc6ce2925d05d"
    ),
    "next213_table": (
        "6157821669f00fa5da5cbaa9a21c790e5afd131014cbee2dace98dc7d9e009f2"
    ),
}


def build_stage_specs(
    *,
    current_path_key: str,
    current_terms: Sequence[Mapping[str, object]],
    normalizations: Mapping[str, Mapping[str, object]],
    amplitude_fractions: Sequence[float] = n212.n210.AMPLITUDE_FRACTIONS,
    risk_scale: float,
    residual_threshold: float,
) -> list[dict[str, object]]:
    """Build unchanged path plus every unused hypothesis/amplitude proposal."""

    if not isinstance(current_path_key, str) or not current_path_key:
        raise ValueError("NEXT214 current path key differs")
    terms = [dict(term) for term in current_terms]
    used = [str(term.get("hypothesis", "")) for term in terms]
    if not terms or any(not value for value in used) or len(used) != len(set(used)):
        raise ValueError("NEXT214 current path terms differ")
    if not isinstance(normalizations, Mapping) or not normalizations:
        raise ValueError("NEXT214 normalization universe differs")
    amplitudes = tuple(float(value) for value in amplitude_fractions)
    numerators = tuple(n212.n210._amplitude_numerator(value) for value in amplitudes)
    if not amplitudes or len(numerators) != len(set(numerators)):
        raise ValueError("NEXT214 amplitude grid differs")
    scale = float(risk_scale)
    threshold = float(residual_threshold)
    if not math.isfinite(scale) or scale <= 0.0 or not math.isfinite(threshold):
        raise ValueError("NEXT214 fixed scale or threshold differs")
    common = {
        "parent_path_key_sha256": hashlib.sha256(current_path_key.encode()).hexdigest(),
        "current_terms": terms,
        "risk_scale": scale,
        "residual_threshold": threshold,
        "missing_policy": "PROPOSED_TERM_OFF_KEEP_CURRENT_PATH",
        "score_composition": "current_path_plus_one_frozen_bounded_directional_risk_lift",
    }
    payloads: list[dict[str, object]] = [
        {
            **common,
            "proposed_hypothesis": None,
            "proposed_feature": None,
            "proposed_direction": None,
            "proposed_q_lo": None,
            "proposed_q_hi": None,
            "proposed_amplitude_fraction": 0.0,
            "proposed_amplitude_numerator": 0,
        }
    ]
    for hypothesis in sorted(set(str(value) for value in normalizations) - set(used)):
        norm = dict(normalizations[hypothesis])
        if (
            str(norm.get("hypothesis", hypothesis)) != hypothesis
            or norm.get("direction") not in n212.n210.n208.n207.PROTECTION_DIRECTIONS
            or norm.get("feature") is None
        ):
            raise ValueError("NEXT214 normalization record differs")
        for amplitude, numerator in zip(amplitudes, numerators, strict=True):
            payloads.append(
                {
                    **common,
                    "proposed_hypothesis": hypothesis,
                    "proposed_feature": str(norm["feature"]),
                    "proposed_direction": str(norm["direction"]),
                    "proposed_q_lo": float(norm["q_lo"]),
                    "proposed_q_hi": float(norm["q_hi"]),
                    "proposed_amplitude_fraction": amplitude,
                    "proposed_amplitude_numerator": numerator,
                }
            )
    specs = [
        {
            **payload,
            "candidate_key": json.dumps(payload, sort_keys=True, separators=(",", ":")),
        }
        for payload in payloads
    ]
    if len({str(spec["candidate_key"]) for spec in specs}) != len(specs):
        raise RuntimeError("NEXT214 stage candidate keys are not unique")
    return specs


def strictly_improves(
    candidate: Mapping[str, object],
    current: Mapping[str, object],
    tolerance: float = IMPROVEMENT_TOLERANCE,
) -> bool:
    """Apply the frozen lexicographic strict-improvement rule."""

    candidate_failures = int(candidate["failed_constraint_count"])
    current_failures = int(current["failed_constraint_count"])
    candidate_shortfall = float(candidate["normalized_shortfall_sum"])
    current_shortfall = float(current["normalized_shortfall_sum"])
    tol = float(tolerance)
    if any(not math.isfinite(value) for value in (candidate_shortfall, current_shortfall, tol)):
        raise ValueError("NEXT214 improvement records differ")
    return bool(
        candidate_failures < current_failures
        or (
            candidate_failures == current_failures
            and candidate_shortfall < current_shortfall - tol
        )
    )


def _materialize_stage(
    *, features: pd.DataFrame, current_score: np.ndarray,
    activation_score: np.ndarray, support: np.ndarray,
    specs: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    columns: dict[str, np.ndarray] = {}
    terms: list[dict[str, object]] = []
    runtime: list[dict[str, object]] = []
    for raw in specs:
        spec = dict(raw)
        feature = spec["proposed_feature"]
        values = (
            np.full(len(features), np.nan)
            if feature is None
            else pd.to_numeric(features[str(feature)], errors="coerce").to_numpy(float)
        )
        score, got_support, _ = n212.anchored_two_signal_score(
            anchor_score=current_score, activation_score=activation_score,
            base_support=support, feature_values=values,
            direction=spec["proposed_direction"], q_lo=spec["proposed_q_lo"],
            q_hi=spec["proposed_q_hi"],
            residual_threshold=float(spec["residual_threshold"]),
            amplitude_fraction=float(spec["proposed_amplitude_fraction"]),
            risk_scale=float(spec["risk_scale"]),
        )
        maximum = float(np.max(score[got_support])) if got_support.any() else 0.0
        divisor = max(1.0, maximum / 50.0)
        encoded = np.full(len(features), np.nan)
        encoded[got_support] = np.sinh(score[got_support] / divisor)
        key = str(spec["candidate_key"])
        term_id = f"next214_virtual_candidate__{hashlib.sha256(key.encode()).hexdigest()[:24]}"
        column = f"_{term_id}_value"
        columns[column] = encoded
        terms.append(
            {
                "term_id": term_id, "feature": column, "direction": 1,
                "transform": "asinh", "center": 0.0, "scale": 1.0 / divisor,
                "group": "next214_forward_stagewise_risk_lift",
                "encoding": "asinh_sinh_exact_stagewise_risk_score",
                "physical_candidate_key": key,
            }
        )
        runtime.append(
            {
                "candidate_key": key, "base_term_ids": [term_id],
                "base_weights": [1.0], "optional_term_id": None,
                "optional_weight": 0.0,
            }
        )
    return (
        pd.concat([features.reset_index(drop=True), pd.DataFrame(columns)], axis=1),
        terms,
        runtime,
    )


def _paths(
    roots: Mapping[str, Path], freeze_path: Path, next202_design_path: Path,
    next205_design_path: Path, next207_design_path: Path,
    next208_design_path: Path, next209_design_path: Path,
    next210_design_path: Path, next211_design_path: Path,
    next212_design_path: Path, next213_design_path: Path, design_path: Path,
) -> dict[str, Path]:
    paths = n213._paths(
        roots, freeze_path, next202_design_path, next205_design_path,
        next207_design_path, next208_design_path, next209_design_path,
        next210_design_path, next211_design_path, next212_design_path,
        next213_design_path,
    )
    paths["next213_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next213_manifest": roots["next213"] / n213.MANIFEST_NAME,
            "next213_diagnostic": roots["next213"] / n213.DIAGNOSTIC_NAME,
            "next213_table": roots["next213"] / n213.TABLE_NAME,
        }
    )
    return paths


def _verify_next213(
    paths: Mapping[str, Path], input_hashes: Mapping[str, str]
) -> tuple[tuple[str, ...], str, str, dict[str, object]]:
    prior_paths = dict(paths)
    prior_paths["design"] = paths["next213_design"]
    prior_hashes = dict(input_hashes)
    prior_hashes["design"] = input_hashes["next213_design"]
    eligible, primary_key, _, _ = n213._verify_next212(prior_paths, prior_hashes)
    manifest = json.loads(paths["next213_manifest"].read_text())
    diagnostic = json.loads(paths["next213_diagnostic"].read_text())
    table = pd.read_parquet(paths["next213_table"])
    closest = diagnostic.get("global_closest", {})
    start_key = str(closest.get("candidate_key", ""))
    expected_outputs = {
        n213.DIAGNOSTIC_NAME: input_hashes["next213_diagnostic"],
        n213.TABLE_NAME: input_hashes["next213_table"],
    }
    if (
        manifest.get("protocol") != n213.PROTOCOL
        or manifest.get("candidate_count") != n213.EXPECTED_CANDIDATE_COUNT
        or manifest.get("candidate_key_sha256") != n213.EXPECTED_CANDIDATE_KEY_SHA256
        or manifest.get("next212_record_population_reproduced") is not True
        or manifest.get("next212_candidate_universe_reproduced") is not True
        or manifest.get("next212_all_gate_candidate_count") != 0
        or manifest.get("two_signal_risk_lift_branch_closed") is not True
        or manifest.get("new_formula_searched") is not False
        or manifest.get("new_formula_selected") is not False
        or manifest.get("outputs_sha256") != expected_outputs
        or any(
            manifest.get(key) is not value
            for key, value in n212.n210.n208.BOUNDARY_FLAGS.items()
        )
        or manifest.get("executed_source_sha256", {}).get(
            "src/next213_two_signal_broad_diagnostic.py"
        )
        != EXPECTED_NEXT213_SOURCE_SHA256
        or _sha256_file(Path(n213.__file__).resolve()) != EXPECTED_NEXT213_SOURCE_SHA256
        or diagnostic.get("protocol") != n213.PROTOCOL
        or diagnostic.get("candidate_count") != n213.EXPECTED_CANDIDATE_COUNT
        or diagnostic.get("candidate_key_sha256") != n213.EXPECTED_CANDIDATE_KEY_SHA256
        or diagnostic.get("new_formula_searched") is not False
        or diagnostic.get("validation_outputs_opened") is not False
        or hashlib.sha256(start_key.encode()).hexdigest() != EXPECTED_START_KEY_SHA256
        or not math.isclose(
            float(closest.get("normalized_shortfall_sum")),
            EXPECTED_START_SHORTFALL, rel_tol=0.0, abs_tol=1.0e-15,
        )
        or len(table) != n213.EXPECTED_CANDIDATE_COUNT
    ):
        raise ValueError("NEXT214 NEXT213 provenance differs")
    return eligible, primary_key, start_key, closest


def run_forward_stagewise_risk_lift(
    *,
    scigen_feature_dir: Path,
    scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path,
    wyformer_discovery_endpoint_dir: Path,
    next98_dir: Path, next110_dir: Path, next111_dir: Path,
    next113_dir: Path, next114_dir: Path, next116_dir: Path,
    next117_dir: Path, next120_dir: Path, next121_dir: Path,
    next122_dir: Path, next124_dir: Path, next125_dir: Path,
    next129_dir: Path, next130_dir: Path, next133_dir: Path,
    next134_dir: Path, next163_dir: Path, next164_dir: Path,
    next168_dir: Path, next173_dir: Path, next179_dir: Path,
    next180_dir: Path, next181_dir: Path, next182_dir: Path,
    next183_dir: Path, next184_dir: Path, next185_dir: Path,
    next186_dir: Path, next188_dir: Path, next190_dir: Path,
    next192_dir: Path, next194_dir: Path, next199_dir: Path,
    next200_dir: Path, next201_dir: Path, next202_dir: Path,
    next203_dir: Path, next204_dir: Path, next205_dir: Path,
    next206_dir: Path, next207_dir: Path, next208_dir: Path,
    next209_dir: Path, next210_dir: Path, next211_dir: Path,
    next212_dir: Path, next213_dir: Path,
    next135_freeze_path: Path, next202_design_path: Path,
    next205_design_path: Path, next207_design_path: Path,
    next208_design_path: Path, next209_design_path: Path,
    next210_design_path: Path, next211_design_path: Path,
    next212_design_path: Path, next213_design_path: Path,
    design_path: Path, output_dir: Path,
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only forward-stagewise search."""

    stage_values = (
        (98, next98_dir), (110, next110_dir), (111, next111_dir),
        (113, next113_dir), (114, next114_dir), (116, next116_dir),
        (117, next117_dir), (120, next120_dir), (121, next121_dir),
        (122, next122_dir), (124, next124_dir), (125, next125_dir),
        (129, next129_dir), (130, next130_dir), (133, next133_dir),
        (134, next134_dir), (163, next163_dir), (164, next164_dir),
        (168, next168_dir), (173, next173_dir), (179, next179_dir),
        (180, next180_dir), (181, next181_dir), (182, next182_dir),
        (183, next183_dir), (184, next184_dir), (185, next185_dir),
        (186, next186_dir), (188, next188_dir), (190, next190_dir),
        (192, next192_dir),
    )
    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{f"next{stage}": Path(value).resolve() for stage, value in stage_values},
        **{
            f"next{stage}": Path(value).resolve()
            for stage, value in (
                (194, next194_dir), (199, next199_dir), (200, next200_dir),
                (201, next201_dir), (202, next202_dir), (203, next203_dir),
                (204, next204_dir), (205, next205_dir), (206, next206_dir),
                (207, next207_dir), (208, next208_dir), (209, next209_dir),
                (210, next210_dir), (211, next211_dir), (212, next212_dir),
                (213, next213_dir),
            )
        },
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots, Path(next135_freeze_path).resolve(),
        Path(next202_design_path).resolve(), Path(next205_design_path).resolve(),
        Path(next207_design_path).resolve(), Path(next208_design_path).resolve(),
        Path(next209_design_path).resolve(), Path(next210_design_path).resolve(),
        Path(next211_design_path).resolve(), Path(next212_design_path).resolve(),
        Path(next213_design_path).resolve(), Path(design_path).resolve(),
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(search_workers) is not int or search_workers <= 0:
        raise ValueError("NEXT214 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT214 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT214 formal input identity differs: {differing}")
    eligible, primary_key, start_key, start_residual = _verify_next213(
        paths, input_hashes
    )
    combined, feature_tables, base_key, base_score, base_support, endpoint = (
        n212.n210.n208._reconstruct_next206(paths=paths)
    )
    next210_specs = n212.n210.build_candidate_specs(
        base_candidate_key=base_key, eligible_hypotheses=eligible,
        features=combined, base_score=base_score, base_support=base_support,
        residual_threshold=n212.n210.n208.n207.EXPECTED_RESIDUAL_THRESHOLD,
    )
    normalizations = n212._unique_hypothesis_specs(next210_specs)
    primary_matches = [
        spec for spec in next210_specs if str(spec["candidate_key"]) == primary_key
    ]
    if len(primary_matches) != 1:
        raise ValueError("NEXT214 primary anchor differs")
    primary = primary_matches[0]
    primary_values = pd.to_numeric(
        combined[str(primary["feature"])], errors="coerce"
    ).to_numpy(float)
    primary_score, support, _ = n212.n210.residual_risk_lift_score(
        base_score=base_score, base_support=base_support,
        feature_values=primary_values, direction=str(primary["direction"]),
        q_lo=float(primary["q_lo"]), q_hi=float(primary["q_hi"]),
        residual_threshold=float(primary["residual_threshold"]),
        amplitude_fraction=float(primary["amplitude_fraction"]),
        risk_scale=float(primary["risk_scale"]),
    )
    two_specs = n212.build_candidate_specs(
        anchor_spec=primary, next210_specs=next210_specs
    )
    start_matches = [spec for spec in two_specs if str(spec["candidate_key"]) == start_key]
    if len(start_matches) != 1:
        raise ValueError("NEXT214 two-signal starting path differs")
    second = start_matches[0]
    second_values = pd.to_numeric(
        combined[str(second["secondary_feature"])], errors="coerce"
    ).to_numpy(float)
    current_score, support, _ = n212.anchored_two_signal_score(
        anchor_score=primary_score, activation_score=base_score,
        base_support=support, feature_values=second_values,
        direction=str(second["secondary_direction"]),
        q_lo=float(second["secondary_q_lo"]), q_hi=float(second["secondary_q_hi"]),
        residual_threshold=float(second["residual_threshold"]),
        amplitude_fraction=float(second["secondary_amplitude_fraction"]),
        risk_scale=float(second["risk_scale"]),
    )
    current_terms: list[dict[str, object]] = [
        {
            "hypothesis": str(primary["hypothesis"]),
            "feature": str(primary["feature"]),
            "direction": str(primary["direction"]),
            "q_lo": float(primary["q_lo"]), "q_hi": float(primary["q_hi"]),
            "amplitude_fraction": float(primary["amplitude_fraction"]),
        },
        {
            "hypothesis": str(second["secondary_hypothesis"]),
            "feature": str(second["secondary_feature"]),
            "direction": str(second["secondary_direction"]),
            "q_lo": float(second["secondary_q_lo"]),
            "q_hi": float(second["secondary_q_hi"]),
            "amplitude_fraction": float(second["secondary_amplitude_fraction"]),
        },
    ]
    current_key = start_key
    current_residual = {
        "failed_constraint_count": int(start_residual["failed_constraint_count"]),
        "normalized_shortfall_sum": float(start_residual["normalized_shortfall_sum"]),
    }
    folds = n164.assign_group_folds(combined["reduced_formula"].astype(str).to_numpy())
    sources = combined["source_dataset"].astype(str).to_numpy()
    cells = n164.build_source_fold_cells(source=sources, folds=folds)
    pauling_by_cell = {
        str(cell["cell_id"]): n164._pauling_baseline(
            combined.loc[np.asarray(cell["mask"], dtype=bool)],
            endpoint[np.asarray(cell["mask"], dtype=bool)],
        )
        for cell in cells
    }
    stage_summaries: list[dict[str, object]] = []
    all_records: list[pd.DataFrame] = []
    all_gate_found = False
    stop_reason = "maximum_depth_reached"
    total_elapsed = 0.0
    for depth in range(len(current_terms) + 1, MAX_TERMS + 1):
        specs = build_stage_specs(
            current_path_key=current_key, current_terms=current_terms,
            normalizations=normalizations, risk_scale=float(primary["risk_scale"]),
            residual_threshold=float(primary["residual_threshold"]),
        )
        virtual, terms, runtime = _materialize_stage(
            features=combined, current_score=current_score,
            activation_score=base_score, support=support, specs=specs,
        )
        started = time.perf_counter()
        result = n212.n210.n208.n205.n203.n202.n200.n194.n130.n125.search_optional_guard_laws_parallel(
            features=virtual, endpoint=endpoint, old_terms=terms,
            optional_terms=[], candidate_specs=runtime, workers=search_workers,
        )
        elapsed = time.perf_counter() - started
        total_elapsed += elapsed
        records = pd.DataFrame(result["candidate_records"])
        spec_by_key = {str(spec["candidate_key"]): spec for spec in specs}
        records["depth"] = depth
        records["proposed_hypothesis"] = records["candidate_key"].map(
            lambda key: spec_by_key[str(key)]["proposed_hypothesis"]
        )
        records["proposed_amplitude_fraction"] = records["candidate_key"].map(
            lambda key: spec_by_key[str(key)]["proposed_amplitude_fraction"]
        )
        all_records.append(records)
        all_gate = records["passes_all_discovery_gates"].fillna(False).astype(bool)
        if all_gate.any():
            selected_record = result["selected"]["record"]
            if not bool(selected_record["passes_all_discovery_gates"]):
                selected_record = records.loc[all_gate].sort_values(
                    "candidate_key", kind="mergesort"
                ).iloc[0].to_dict()
            selected_spec = spec_by_key[str(selected_record["candidate_key"])]
            proposal = selected_spec["proposed_hypothesis"]
            if proposal is not None:
                current_terms.append(
                    {
                        "hypothesis": str(proposal),
                        "feature": str(selected_spec["proposed_feature"]),
                        "direction": str(selected_spec["proposed_direction"]),
                        "q_lo": float(selected_spec["proposed_q_lo"]),
                        "q_hi": float(selected_spec["proposed_q_hi"]),
                        "amplitude_fraction": float(
                            selected_spec["proposed_amplitude_fraction"]
                        ),
                    }
                )
            current_key = str(selected_record["candidate_key"])
            all_gate_found = True
            stop_reason = "all_discovery_gates_passed"
            stage_summaries.append(
                {
                    "depth": depth, "candidate_count": len(records),
                    "auc_safe_non_broad_count": 0,
                    "all_gate_count": int(all_gate.sum()),
                    "accepted": True, "accepted_hypothesis": proposal,
                    "elapsed_seconds": elapsed,
                }
            )
            break
        diagnostic_mask = (
            records["passes_source_auc_gates"].fillna(False).astype(bool)
            & records["passes_safe_all_cells"].fillna(False).astype(bool)
            & ~records["passes_broad_all_cells"].fillna(False).astype(bool)
        )
        diagnostic = records.loc[diagnostic_mask]
        if diagnostic.empty:
            stop_reason = "no_auc_safe_non_broad_candidate"
            stage_summaries.append(
                {
                    "depth": depth, "candidate_count": len(records),
                    "auc_safe_non_broad_count": 0, "all_gate_count": 0,
                    "accepted": False, "elapsed_seconds": elapsed,
                }
            )
            break
        residual_rows: list[dict[str, object]] = []
        score_by_key: dict[str, np.ndarray] = {}
        term_by_key = {str(term["physical_candidate_key"]): term for term in terms}
        published_by_key = diagnostic.set_index("candidate_key", drop=False)
        for key in diagnostic["candidate_key"].astype(str):
            score, got_support = n212.n210.n208.n205.n203.n202.n200.n194.n87._term_risk(
                virtual, term_by_key[key]
            )
            tables = n164._threshold_tables(
                score=score, supported=got_support, endpoint=endpoint, cells=cells
            )
            if tables is None:
                raise RuntimeError("NEXT214 candidate has no threshold table")
            residual = n164.diagnose_broad_threshold_tables(
                tables=tables, cells=cells, pauling_by_cell=pauling_by_cell,
                safe_threshold=float(published_by_key.loc[key, "safe_threshold"]),
            )
            residual_rows.append(
                {
                    "candidate_key": key,
                    "failed_constraint_count": int(residual["failed_constraint_count"]),
                    "normalized_shortfall_sum": float(
                        residual["normalized_shortfall_sum"]
                    ),
                    "best_threshold": float(residual["best_threshold"]),
                    "failures_json": json.dumps(
                        residual["failures"], sort_keys=True, separators=(",", ":")
                    ),
                }
            )
            score_by_key[key] = score
        residual_frame = pd.DataFrame(residual_rows).sort_values(
            ["failed_constraint_count", "normalized_shortfall_sum", "candidate_key"],
            kind="mergesort",
        )
        best = residual_frame.iloc[0].to_dict()
        best_spec = spec_by_key[str(best["candidate_key"])]
        proposal = best_spec["proposed_hypothesis"]
        accepted = bool(
            proposal is not None and strictly_improves(best, current_residual)
        )
        stage_summaries.append(
            {
                "depth": depth, "candidate_count": len(records),
                "auc_safe_non_broad_count": len(diagnostic),
                "all_gate_count": 0, "accepted": accepted,
                "accepted_hypothesis": proposal if accepted else None,
                "accepted_amplitude_fraction": (
                    float(best_spec["proposed_amplitude_fraction"])
                    if accepted else None
                ),
                "best_failed_constraint_count": int(best["failed_constraint_count"]),
                "best_normalized_shortfall_sum": float(
                    best["normalized_shortfall_sum"]
                ),
                "best_threshold": float(best["best_threshold"]),
                "failures": json.loads(str(best["failures_json"])),
                "elapsed_seconds": elapsed,
            }
        )
        if not accepted:
            stop_reason = "no_strict_residual_improvement"
            break
        current_score = score_by_key[str(best["candidate_key"])]
        current_key = str(best["candidate_key"])
        current_residual = {
            "failed_constraint_count": int(best["failed_constraint_count"]),
            "normalized_shortfall_sum": float(best["normalized_shortfall_sum"]),
        }
        current_terms.append(
            {
                "hypothesis": str(proposal),
                "feature": str(best_spec["proposed_feature"]),
                "direction": str(best_spec["proposed_direction"]),
                "q_lo": float(best_spec["proposed_q_lo"]),
                "q_hi": float(best_spec["proposed_q_hi"]),
                "amplitude_fraction": float(best_spec["proposed_amplitude_fraction"]),
            }
        )

    candidates = pd.concat(all_records, ignore_index=True)
    final_shortfall = (
        None if all_gate_found else float(current_residual["normalized_shortfall_sum"])
    )
    formula = {
        "protocol": PROTOCOL,
        "kind": "forward_stagewise_bounded_x0_risk_sum_no_dft_score",
        "terms": current_terms,
        "term_count": len(current_terms),
        "risk_scale": float(primary["risk_scale"]),
        "residual_threshold": float(primary["residual_threshold"]),
        "support_policy": "UNCHANGED_FROM_NEXT206",
        "missing_policy": "TERM_OFF_KEEP_CURRENT_PATH",
        "dft_values_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
    }
    evaluation = {
        "protocol": PROTOCOL,
        "evaluation_mode": "frozen_forward_stagewise_discovery_search",
        "start_candidate_key_sha256": EXPECTED_START_KEY_SHA256,
        "start_normalized_shortfall_sum": EXPECTED_START_SHORTFALL,
        "rows": {
            "scigen": int(len(feature_tables["scigen"])),
            "wyformer": int(len(feature_tables["wyformer"])),
            "total": int(len(combined)),
        },
        "maximum_terms": MAX_TERMS,
        "final_term_count": len(current_terms),
        "stage_summaries": stage_summaries,
        "total_candidate_evaluations": len(candidates),
        "elapsed_seconds": total_elapsed,
        "search_workers": search_workers,
        "all_discovery_gates_passed": all_gate_found,
        "freeze_authorized": all_gate_found,
        "stop_reason": stop_reason,
        "final_failed_constraint_count": (
            0 if all_gate_found else int(current_residual["failed_constraint_count"])
        ),
        "final_normalized_shortfall_sum": final_shortfall,
        "normalized_shortfall_reduction_from_start": (
            None if final_shortfall is None else EXPECTED_START_SHORTFALL - final_shortfall
        ),
        "final_formula": formula,
        "requires_unopened_internal_validation_before_claim": True,
    }
    catalogue = {
        "protocol": PROTOCOL,
        "design_sha256": input_hashes["design"],
        "eligible_hypothesis_count": len(eligible),
        "amplitude_fractions": list(n212.n210.AMPLITUDE_FRACTIONS),
        "maximum_terms": MAX_TERMS,
        "improvement_tolerance": IMPROVEMENT_TOLERANCE,
        "normalization_refit": False,
        "base_support_unchanged": True,
        "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {
        "src/next164_interior_attenuation_broad_residual.py": Path(n164.__file__).resolve(),
        "src/next213_two_signal_broad_diagnostic.py": Path(n213.__file__).resolve(),
        "src/next214_forward_stagewise_risk_lift.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    try:
        catalogue_path = staging / CATALOGUE_NAME
        evaluation_path = staging / EVALUATION_NAME
        formula_path = staging / FORMULA_NAME
        search_path = staging / SEARCH_NAME
        _write_json(catalogue_path, catalogue)
        _write_json(evaluation_path, evaluation)
        _write_json(formula_path, formula)
        candidates.to_parquet(search_path, index=False)
        outputs = [catalogue_path, evaluation_path, formula_path, search_path]
        manifest = {
            "protocol": PROTOCOL,
            "start_candidate_key_sha256": EXPECTED_START_KEY_SHA256,
            "maximum_terms": MAX_TERMS,
            "final_term_count": len(current_terms),
            "stage_count": len(stage_summaries),
            "total_candidate_evaluations": len(candidates),
            "passes_all_cross_source_discovery_gates": all_gate_found,
            "freeze_authorized": all_gate_found,
            "stop_reason": stop_reason,
            "requires_unopened_internal_validation_before_claim": True,
            "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
            "discovery_outcomes_used_as_offline_labels": True,
            **n212.n210.n208.BOUNDARY_FLAGS,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {path.name: _sha256_file(path) for path in outputs},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT214 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT214 source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scigen-feature-dir", type=Path, required=True)
    parser.add_argument("--scigen-discovery-endpoint-dir", type=Path, required=True)
    parser.add_argument("--wyformer-feature-dir", type=Path, required=True)
    parser.add_argument("--wyformer-discovery-endpoint-dir", type=Path, required=True)
    stages = (
        98, 110, 111, 113, 114, 116, 117, 120, 121, 122, 124, 125,
        129, 130, 133, 134, 163, 164, 168, 173, 179, 180, 181, 182,
        183, 184, 185, 186, 188, 190, 192,
    )
    later_stages = (
        194, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
        210, 211, 212, 213,
    )
    for stage in stages + later_stages:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    for stage in (202, 205, 207, 208, 209, 210, 211, 212, 213):
        parser.add_argument(f"--next{stage}-design-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_forward_stagewise_risk_lift(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{f"next{stage}_dir": getattr(args, f"next{stage}_dir") for stage in stages + later_stages},
        next135_freeze_path=args.next135_freeze_path,
        **{
            f"next{stage}_design_path": getattr(args, f"next{stage}_design_path")
            for stage in (202, 205, 207, 208, 209, 210, 211, 212, 213)
        },
        design_path=args.design_path,
        output_dir=args.output_dir,
        search_workers=args.search_workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "MAX_TERMS", "build_stage_specs", "run_forward_stagewise_risk_lift",
    "strictly_improves",
]


if __name__ == "__main__":
    raise SystemExit(main())
