#!/usr/bin/env python3
"""Run frozen forward-stagewise signed repair-band redistribution."""

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

import src.next221_signed_redistribution_broad_diagnostic as n221
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


n220 = n221.n220
n215 = n220.n215
PROTOCOL = "2026-08-08-next222-forward-stagewise-signed-redistribution-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT222_FORWARD_SIGNED_CATALOGUE.json"
EVALUATION_NAME = "NEXT222_FORWARD_SIGNED_EVALUATION.json"
FORMULA_NAME = "NEXT222_FINAL_PATH_FORMULA.json"
SEARCH_NAME = "next222_forward_signed_candidates.parquet"
EXPECTED_DESIGN_SHA256 = (
    "a5d931c3735866c51d5527ed671a9540912d2a23af9254c5fa14d2ca058c7628"
)
EXPECTED_NEXT221_SOURCE_SHA256 = (
    "19dd5681869b102379f549fa990465dacdb9351b21995d48616570991d10b972"
)
EXPECTED_START_KEY_SHA256 = (
    "bff96d48e29e07155f759ea37c3d7cb2d4b76a5fdd2f1246ba2cc97e8b48b688"
)
EXPECTED_START_SHORTFALL = 0.2572028547126239
EXPECTED_START_FAILED_COUNT = 6
MAX_SIGNED_TERMS = 6
IMPROVEMENT_TOLERANCE = 1.0e-12
SEARCH_WORKERS = n220.SEARCH_WORKERS
BOUNDARY_FLAGS = n220.BOUNDARY_FLAGS
REQUIRED_STAGES = n221.REQUIRED_STAGES + (221,)
REQUIRED_DESIGN_STAGES = n221.REQUIRED_DESIGN_STAGES + (221,)
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n221.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next221_design": n221.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next221_manifest": (
        "356c85620a86474f80d3f0da79d0a67e48116896e235b8d931430cf07b77c822"
    ),
    "next221_diagnostic": (
        "b4169ac7a5f4178d49fd918a4a20d88c22c2dcd9d2853397f4a4783fe542c0e6"
    ),
    "next221_table": (
        "6368431c5b0ea2e66856491ddd63a9c5901c1f32079db4a394da0952b4af3ccf"
    ),
}


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
    tolerance = float(tolerance)
    if any(
        not math.isfinite(value)
        for value in (candidate_shortfall, current_shortfall, tolerance)
    ):
        raise ValueError("NEXT222 improvement records differ")
    return bool(
        candidate_failures < current_failures
        or (
            candidate_failures == current_failures
            and candidate_shortfall < current_shortfall - tolerance
        )
    )


def cumulative_signed_score(
    *,
    base_score: object,
    current_delta: object,
    base_support: object,
    feature_values: object,
    direction: str | None,
    q_lo: float | None,
    q_hi: float | None,
    lower: float,
    upper: float,
    beta_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Add one signed term on the original band and apply the frozen zero floor."""

    base = np.asarray(base_score, dtype=float)
    delta = np.asarray(current_delta, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    values = np.asarray(feature_values, dtype=float)
    low = float(lower)
    high = float(upper)
    beta = float(beta_fraction)
    if (
        base.ndim != 1
        or delta.shape != base.shape
        or support.shape != base.shape
        or values.shape != base.shape
        or not math.isfinite(low)
        or not math.isfinite(high)
        or not high > low
        or not math.isfinite(beta)
        or not 0.0 <= beta <= 0.25
        or np.any(~np.isfinite(base[support]))
        or np.any(~np.isfinite(delta[support]))
    ):
        raise ValueError("NEXT222 cumulative score inputs differ")
    proposed = delta.copy()
    if direction is None and q_lo is None and q_hi is None and beta == 0.0:
        score = np.maximum(0.0, base + proposed)
        return score, support.copy(), np.zeros(base.shape, dtype=bool), proposed
    if (
        direction not in n215.PROTECTION_DIRECTIONS
        or q_lo is None
        or q_hi is None
        or beta <= 0.0
    ):
        raise ValueError("NEXT222 signed term specification differs")
    protection = n220.n216.bounded_directional_protection(
        values, direction, float(q_lo), float(q_hi)
    )
    active = (
        support
        & (base >= low)
        & (base < high)
        & np.isfinite(protection)
    )
    proposed[active] += beta * (high - low) * (
        1.0 - 2.0 * protection[active]
    )
    score = np.maximum(0.0, base + proposed)
    return score, support.copy(), active, proposed


def build_stage_specs(
    *,
    current_path_key: str,
    current_terms: Sequence[Mapping[str, object]],
    normalizations: Mapping[str, Mapping[str, object]],
    beta_fractions: Sequence[float] = n220.BETA_FRACTIONS,
) -> list[dict[str, object]]:
    """Build unchanged path plus every unused signed hypothesis/beta proposal."""

    terms = [dict(term) for term in current_terms]
    used = [str(term.get("hypothesis", "")) for term in terms]
    if (
        not isinstance(current_path_key, str)
        or not current_path_key
        or not terms
        or any(not value for value in used)
        or len(used) != len(set(used))
        or not isinstance(normalizations, Mapping)
        or not normalizations
    ):
        raise ValueError("NEXT222 current path or normalization universe differs")
    betas = tuple(float(value) for value in beta_fractions)
    numerators = tuple(n220._beta_numerator(value) for value in betas)
    common = {
        "parent_path_key_sha256": hashlib.sha256(current_path_key.encode()).hexdigest(),
        "current_terms": terms,
        "beta_denominator": n220.BETA_DENOMINATOR,
        "missing_policy": "PROPOSED_TERM_OFF_KEEP_CURRENT_PATH",
        "score_composition": "nonnegative_next214_plus_cumulative_signed_band_terms",
    }
    payloads: list[dict[str, object]] = [
        {
            **common,
            "proposed_hypothesis": None,
            "proposed_feature": None,
            "proposed_direction": None,
            "proposed_q_lo": None,
            "proposed_q_hi": None,
            "proposed_beta_fraction": 0.0,
            "proposed_beta_numerator": 0,
        }
    ]
    for hypothesis in sorted(set(normalizations) - set(used)):
        norm = dict(normalizations[hypothesis])
        if (
            norm.get("direction") not in n215.PROTECTION_DIRECTIONS
            or norm.get("feature") is None
        ):
            raise ValueError("NEXT222 normalization record differs")
        for beta, numerator in zip(betas, numerators, strict=True):
            payloads.append(
                {
                    **common,
                    "proposed_hypothesis": hypothesis,
                    "proposed_feature": str(norm["feature"]),
                    "proposed_direction": str(norm["direction"]),
                    "proposed_q_lo": float(norm["q_lo"]),
                    "proposed_q_hi": float(norm["q_hi"]),
                    "proposed_beta_fraction": beta,
                    "proposed_beta_numerator": numerator,
                }
            )
    specs = [
        {
            **payload,
            "candidate_key": json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ),
        }
        for payload in payloads
    ]
    if len({str(spec["candidate_key"]) for spec in specs}) != len(specs):
        raise RuntimeError("NEXT222 stage candidate keys are not unique")
    return specs


def _materialize_stage(
    *,
    features: pd.DataFrame,
    base_score: np.ndarray,
    current_delta: np.ndarray,
    support: np.ndarray,
    specs: Sequence[Mapping[str, object]],
) -> tuple[
    pd.DataFrame,
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, np.ndarray],
]:
    columns: dict[str, np.ndarray] = {}
    terms: list[dict[str, object]] = []
    runtime: list[dict[str, object]] = []
    delta_by_key: dict[str, np.ndarray] = {}
    for raw in specs:
        spec = dict(raw)
        feature = spec["proposed_feature"]
        values = (
            np.full(len(features), np.nan)
            if feature is None
            else pd.to_numeric(features[str(feature)], errors="coerce").to_numpy(float)
        )
        score, got_support, _, proposed_delta = cumulative_signed_score(
            base_score=base_score,
            current_delta=current_delta,
            base_support=support,
            feature_values=values,
            direction=spec["proposed_direction"],
            q_lo=spec["proposed_q_lo"],
            q_hi=spec["proposed_q_hi"],
            lower=n215.REPAIR_LOWER_THRESHOLD,
            upper=n215.REPAIR_UPPER_THRESHOLD,
            beta_fraction=float(spec["proposed_beta_fraction"]),
        )
        maximum = float(np.max(score[got_support])) if got_support.any() else 0.0
        divisor = max(1.0, maximum / 50.0)
        encoded = np.full(len(features), np.nan)
        encoded[got_support] = np.sinh(score[got_support] / divisor)
        key = str(spec["candidate_key"])
        term_id = (
            "next222_virtual_candidate__"
            f"{hashlib.sha256(key.encode()).hexdigest()[:24]}"
        )
        column = f"_{term_id}_value"
        columns[column] = encoded
        terms.append(
            {
                "term_id": term_id,
                "feature": column,
                "direction": 1,
                "transform": "asinh",
                "center": 0.0,
                "scale": 1.0 / divisor,
                "group": "next222_forward_signed",
                "encoding": "asinh_sinh_exact_cumulative_signed_score",
                "physical_candidate_key": key,
            }
        )
        runtime.append(
            {
                "candidate_key": key,
                "base_term_ids": [term_id],
                "base_weights": [1.0],
                "optional_term_id": None,
                "optional_weight": 0.0,
            }
        )
        delta_by_key[key] = proposed_delta
    return (
        pd.concat([features.reset_index(drop=True), pd.DataFrame(columns)], axis=1),
        terms,
        runtime,
        delta_by_key,
    )


def _paths(
    *, roots: Mapping[str, Path], next135_freeze_path: Path,
    design_paths: Mapping[int, Path], design_path: Path,
) -> dict[str, Path]:
    paths = n221._paths(
        roots=roots,
        next135_freeze_path=next135_freeze_path,
        design_paths={stage: design_paths[stage] for stage in n221.REQUIRED_DESIGN_STAGES},
        design_path=design_paths[221],
    )
    paths["next221_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next221_manifest": roots["next221"] / n221.MANIFEST_NAME,
            "next221_diagnostic": roots["next221"] / n221.DIAGNOSTIC_NAME,
            "next221_table": roots["next221"] / n221.TABLE_NAME,
        }
    )
    return paths


def _verify_next221(
    paths: Mapping[str, Path], input_hashes: Mapping[str, str]
) -> tuple[tuple[str, ...], tuple[str, ...], str, str, dict[str, object], str]:
    prior_paths = dict(paths)
    prior_paths["design"] = paths["next221_design"]
    prior_hashes = dict(input_hashes)
    prior_hashes["design"] = input_hashes["next221_design"]
    eligible, eligible214, primary_key, base_start_key, formula214, _, _ = (
        n221._verify_next220(prior_paths, prior_hashes)
    )
    manifest = json.loads(paths["next221_manifest"].read_text())
    diagnostic = json.loads(paths["next221_diagnostic"].read_text())
    table = pd.read_parquet(paths["next221_table"])
    closest = diagnostic.get("global_closest", {})
    start_key = str(closest.get("candidate_key", ""))
    if (
        manifest.get("protocol") != n221.PROTOCOL
        or manifest.get("candidate_count") != n221.EXPECTED_CANDIDATE_COUNT
        or manifest.get("candidate_key_sha256") != n221.EXPECTED_CANDIDATE_KEY_SHA256
        or manifest.get("strict_residual_improvement_observed") is not True
        or manifest.get("signed_redistribution_branch_closed") is not False
        or manifest.get("outputs_sha256") != {
            n221.DIAGNOSTIC_NAME: input_hashes["next221_diagnostic"],
            n221.TABLE_NAME: input_hashes["next221_table"],
        }
        or any(manifest.get(key) is not value for key, value in BOUNDARY_FLAGS.items())
        or manifest.get("executed_source_sha256", {}).get(
            "src/next221_signed_redistribution_broad_diagnostic.py"
        ) != EXPECTED_NEXT221_SOURCE_SHA256
        or _sha256_file(Path(n221.__file__).resolve()) != EXPECTED_NEXT221_SOURCE_SHA256
        or diagnostic.get("improves_over_next214_global_residual") is not True
        or closest.get("hypothesis") != "sivr_cell_hydro_abs__protected_low"
        or not math.isclose(float(closest.get("beta_fraction")), 0.25)
        or int(closest.get("failed_constraint_count")) != EXPECTED_START_FAILED_COUNT
        or not math.isclose(
            float(closest.get("normalized_shortfall_sum")),
            EXPECTED_START_SHORTFALL, rel_tol=0.0, abs_tol=1.0e-15,
        )
        or hashlib.sha256(start_key.encode()).hexdigest() != EXPECTED_START_KEY_SHA256
        or len(table) != n221.EXPECTED_CANDIDATE_COUNT
    ):
        raise ValueError("NEXT222 NEXT221 provenance differs")
    return eligible, eligible214, primary_key, base_start_key, formula214, start_key


def run_forward_stagewise_signed_redistribution(
    *,
    scigen_feature_dir: Path,
    scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path,
    wyformer_discovery_endpoint_dir: Path,
    stage_dirs: Mapping[int, Path],
    next135_freeze_path: Path,
    design_paths: Mapping[int, Path],
    design_path: Path,
    output_dir: Path,
    search_workers: int = SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only signed forward loop."""

    if not set(REQUIRED_STAGES).issubset(stage_dirs):
        raise ValueError("NEXT222 stage directory universe differs")
    if not set(REQUIRED_DESIGN_STAGES).issubset(design_paths):
        raise ValueError("NEXT222 design path universe differs")
    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{f"next{s}": Path(stage_dirs[s]).resolve() for s in REQUIRED_STAGES},
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots=roots,
        next135_freeze_path=Path(next135_freeze_path).resolve(),
        design_paths={s: Path(design_paths[s]).resolve() for s in REQUIRED_DESIGN_STAGES},
        design_path=Path(design_path).resolve(),
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(search_workers) is not int or search_workers <= 0:
        raise ValueError("NEXT222 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT222 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT222 formal input identity differs: {differing}")
    (
        eligible,
        eligible214,
        primary_key,
        base_start_key,
        formula214,
        start_key,
    ) = _verify_next221(paths, input_hashes)
    combined, feature_tables, base_score, support, endpoint = (
        n215._reconstruct_next214_final(
            paths=paths, eligible=eligible214,
            primary_key=primary_key,
            start_key=base_start_key,
            formula=formula214,
        )
    )
    next214_table = pd.read_parquet(paths["next214_search"])
    accepted = next214_table.loc[
        next214_table["depth"].eq(3)
        & next214_table["proposed_hypothesis"].eq(
            "steric_overlap2_vector_q95__protected_low"
        )
        & next214_table["proposed_amplitude_fraction"].eq(0.0625)
    ]
    if len(accepted) != 1:
        raise ValueError("NEXT222 NEXT214 base identity differs")
    base_key = str(accepted.iloc[0]["candidate_key"])
    initial_specs = n220.build_signed_candidate_specs(
        base_candidate_key=base_key,
        eligible_hypotheses=eligible,
        features=combined,
        base_score=base_score,
        base_support=support,
    )
    normalizations = {
        str(spec["hypothesis"]): spec
        for spec in initial_specs if spec["hypothesis"] is not None
    }
    primary_matches = [
        spec for spec in initial_specs if str(spec["candidate_key"]) == start_key
    ]
    if len(primary_matches) != 1:
        raise ValueError("NEXT222 signed starting path differs")
    primary = primary_matches[0]
    primary_values = pd.to_numeric(
        combined[str(primary["feature"])], errors="coerce"
    ).to_numpy(float)
    _, support, _, current_delta = cumulative_signed_score(
        base_score=base_score, current_delta=np.zeros(len(combined)),
        base_support=support, feature_values=primary_values,
        direction=str(primary["direction"]), q_lo=float(primary["q_lo"]),
        q_hi=float(primary["q_hi"]), lower=n215.REPAIR_LOWER_THRESHOLD,
        upper=n215.REPAIR_UPPER_THRESHOLD,
        beta_fraction=float(primary["beta_fraction"]),
    )
    current_terms: list[dict[str, object]] = [
        {
            "hypothesis": str(primary["hypothesis"]),
            "feature": str(primary["feature"]),
            "direction": str(primary["direction"]),
            "q_lo": float(primary["q_lo"]),
            "q_hi": float(primary["q_hi"]),
            "beta_fraction": float(primary["beta_fraction"]),
        }
    ]
    current_key = start_key
    current_residual = {
        "failed_constraint_count": EXPECTED_START_FAILED_COUNT,
        "normalized_shortfall_sum": EXPECTED_START_SHORTFALL,
    }
    n164 = n215.n214.n164
    folds = n164.assign_group_folds(combined["reduced_formula"].astype(str).to_numpy())
    sources = combined["source_dataset"].astype(str).to_numpy()
    cells = n164.build_source_fold_cells(source=sources, folds=folds)
    pauling_by_cell = {
        str(cell["cell_id"]): n164._pauling_baseline(
            combined.loc[np.asarray(cell["mask"], dtype=bool)],
            endpoint[np.asarray(cell["mask"], dtype=bool)],
        ) for cell in cells
    }
    stage_summaries: list[dict[str, object]] = []
    all_records: list[pd.DataFrame] = []
    all_gate_found = False
    stop_reason = "maximum_depth_reached"
    total_elapsed = 0.0
    for depth in range(2, MAX_SIGNED_TERMS + 1):
        specs = build_stage_specs(
            current_path_key=current_key,
            current_terms=current_terms,
            normalizations=normalizations,
        )
        virtual, terms, runtime, delta_by_key = _materialize_stage(
            features=combined, base_score=base_score,
            current_delta=current_delta, support=support, specs=specs,
        )
        started = time.perf_counter()
        result = (
            n215.n214.n212.n210.n208.n205.n203.n202.n200.n194.n130.n125
            .search_optional_guard_laws_parallel(
                features=virtual, endpoint=endpoint, old_terms=terms,
                optional_terms=[], candidate_specs=runtime, workers=search_workers,
            )
        )
        elapsed = time.perf_counter() - started
        total_elapsed += elapsed
        records = pd.DataFrame(result["candidate_records"])
        spec_by_key = {str(spec["candidate_key"]): spec for spec in specs}
        records["depth"] = depth
        records["proposed_hypothesis"] = records["candidate_key"].map(
            lambda key: spec_by_key[str(key)]["proposed_hypothesis"]
        )
        records["proposed_beta_fraction"] = records["candidate_key"].map(
            lambda key: spec_by_key[str(key)]["proposed_beta_fraction"]
        )
        all_records.append(records)
        all_gate = records["passes_all_discovery_gates"].fillna(False).astype(bool)
        if all_gate.any():
            winner = records.loc[all_gate].sort_values(
                "candidate_key", kind="mergesort"
            ).iloc[0]
            spec = spec_by_key[str(winner["candidate_key"])]
            proposal = spec["proposed_hypothesis"]
            if proposal is not None:
                current_terms.append(
                    {
                        "hypothesis": str(proposal),
                        "feature": str(spec["proposed_feature"]),
                        "direction": str(spec["proposed_direction"]),
                        "q_lo": float(spec["proposed_q_lo"]),
                        "q_hi": float(spec["proposed_q_hi"]),
                        "beta_fraction": float(spec["proposed_beta_fraction"]),
                    }
                )
            current_key = str(winner["candidate_key"])
            all_gate_found = True
            stop_reason = "all_discovery_gates_passed"
            stage_summaries.append(
                {"depth": depth, "candidate_count": len(records),
                 "all_gate_count": int(all_gate.sum()), "accepted": True,
                 "accepted_hypothesis": proposal, "elapsed_seconds": elapsed}
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
                {"depth": depth, "candidate_count": len(records),
                 "auc_safe_non_broad_count": 0, "accepted": False,
                 "elapsed_seconds": elapsed}
            )
            break
        term_by_key = {str(term["physical_candidate_key"]): term for term in terms}
        published = diagnostic.set_index("candidate_key", drop=False)
        residual_rows: list[dict[str, object]] = []
        for key in diagnostic["candidate_key"].astype(str):
            score, got_support = n164._term_risk(virtual, term_by_key[key])
            tables = n164._threshold_tables(
                score=score, supported=got_support, endpoint=endpoint, cells=cells
            )
            if tables is None:
                raise RuntimeError("NEXT222 candidate has no threshold table")
            residual = n164.diagnose_broad_threshold_tables(
                tables=tables, cells=cells, pauling_by_cell=pauling_by_cell,
                safe_threshold=float(published.loc[key, "safe_threshold"]),
            )
            residual_rows.append(
                {"candidate_key": key,
                 "failed_constraint_count": int(residual["failed_constraint_count"]),
                 "normalized_shortfall_sum": float(residual["normalized_shortfall_sum"]),
                 "best_threshold": float(residual["best_threshold"]),
                 "failures_json": json.dumps(
                     residual["failures"], sort_keys=True, separators=(",", ":"))}
            )
        residual_frame = pd.DataFrame(residual_rows).sort_values(
            ["failed_constraint_count", "normalized_shortfall_sum", "candidate_key"],
            kind="mergesort",
        )
        best = residual_frame.iloc[0].to_dict()
        spec = spec_by_key[str(best["candidate_key"])]
        proposal = spec["proposed_hypothesis"]
        accepted_step = bool(
            proposal is not None and strictly_improves(best, current_residual)
        )
        stage_summaries.append(
            {"depth": depth, "candidate_count": len(records),
             "auc_safe_non_broad_count": len(diagnostic), "all_gate_count": 0,
             "accepted": accepted_step,
             "accepted_hypothesis": proposal if accepted_step else None,
             "accepted_beta_fraction": float(spec["proposed_beta_fraction"])
             if accepted_step else None,
             "best_failed_constraint_count": int(best["failed_constraint_count"]),
             "best_normalized_shortfall_sum": float(best["normalized_shortfall_sum"]),
             "best_threshold": float(best["best_threshold"]),
             "failures": json.loads(str(best["failures_json"])),
             "elapsed_seconds": elapsed}
        )
        if not accepted_step:
            stop_reason = "no_strict_residual_improvement"
            break
        current_delta = delta_by_key[str(best["candidate_key"])]
        current_key = str(best["candidate_key"])
        current_residual = {
            "failed_constraint_count": int(best["failed_constraint_count"]),
            "normalized_shortfall_sum": float(best["normalized_shortfall_sum"]),
        }
        current_terms.append(
            {"hypothesis": str(proposal), "feature": str(spec["proposed_feature"]),
             "direction": str(spec["proposed_direction"]),
             "q_lo": float(spec["proposed_q_lo"]),
             "q_hi": float(spec["proposed_q_hi"]),
             "beta_fraction": float(spec["proposed_beta_fraction"])}
        )

    candidates = pd.concat(all_records, ignore_index=True)
    final_shortfall = None if all_gate_found else float(
        current_residual["normalized_shortfall_sum"]
    )
    formula = {
        "protocol": PROTOCOL,
        "kind": "forward_stagewise_cumulative_signed_x0_no_dft_score",
        "terms": current_terms,
        "signed_term_count": len(current_terms),
        "repair_lower_threshold": n215.REPAIR_LOWER_THRESHOLD,
        "repair_upper_threshold": n215.REPAIR_UPPER_THRESHOLD,
        "nonnegative_floor": 0.0,
        "support_policy": "UNCHANGED_FROM_NEXT214",
        "missing_policy": "TERM_OFF_KEEP_CURRENT_PATH",
        "dft_values_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
    }
    evaluation = {
        "protocol": PROTOCOL,
        "evaluation_mode": "frozen_forward_stagewise_signed_discovery_search",
        "start_candidate_key_sha256": EXPECTED_START_KEY_SHA256,
        "start_normalized_shortfall_sum": EXPECTED_START_SHORTFALL,
        "rows": {"scigen": int(len(feature_tables["scigen"])),
                 "wyformer": int(len(feature_tables["wyformer"])),
                 "total": int(len(combined))},
        "maximum_signed_terms": MAX_SIGNED_TERMS,
        "final_signed_term_count": len(current_terms),
        "stage_summaries": stage_summaries,
        "total_candidate_evaluations": len(candidates),
        "elapsed_seconds": total_elapsed,
        "all_discovery_gates_passed": all_gate_found,
        "freeze_authorized": all_gate_found,
        "stop_reason": stop_reason,
        "final_failed_constraint_count": 0 if all_gate_found else int(
            current_residual["failed_constraint_count"]
        ),
        "final_normalized_shortfall_sum": final_shortfall,
        "normalized_shortfall_reduction_from_start": None if final_shortfall is None
        else EXPECTED_START_SHORTFALL - final_shortfall,
        "final_formula": formula,
        "requires_unopened_internal_validation_before_claim": True,
    }
    catalogue = {
        "protocol": PROTOCOL, "design_sha256": input_hashes["design"],
        "eligible_hypothesis_count": len(eligible),
        "eligible_hypothesis_sha256": n220.EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256,
        "beta_fractions": list(n220.BETA_FRACTIONS),
        "maximum_signed_terms": MAX_SIGNED_TERMS,
        "improvement_tolerance": IMPROVEMENT_TOLERANCE,
        "normalization_refit": False, "base_support_unchanged": True,
        "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {
        "src/next221_signed_redistribution_broad_diagnostic.py": Path(n221.__file__).resolve(),
        "src/next222_forward_stagewise_signed_redistribution.py": Path(__file__).resolve(),
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
            "maximum_signed_terms": MAX_SIGNED_TERMS,
            "final_signed_term_count": len(current_terms),
            "stage_count": len(stage_summaries),
            "total_candidate_evaluations": len(candidates),
            "passes_all_cross_source_discovery_gates": all_gate_found,
            "freeze_authorized": all_gate_found,
            "stop_reason": stop_reason,
            "requires_unopened_internal_validation_before_claim": True,
            "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
            "discovery_outcomes_used_as_offline_labels": True,
            **BOUNDARY_FLAGS,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {path.name: _sha256_file(path) for path in outputs},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT222 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT222 source changed before publication")
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
    for stage in REQUIRED_STAGES:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    for stage in REQUIRED_DESIGN_STAGES:
        parser.add_argument(f"--next{stage}-design-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_forward_stagewise_signed_redistribution(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        stage_dirs={s: getattr(args, f"next{s}_dir") for s in REQUIRED_STAGES},
        next135_freeze_path=args.next135_freeze_path,
        design_paths={s: getattr(args, f"next{s}_design_path") for s in REQUIRED_DESIGN_STAGES},
        design_path=args.design_path, output_dir=args.output_dir,
        search_workers=args.search_workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "MAX_SIGNED_TERMS", "build_stage_specs", "cumulative_signed_score",
    "run_forward_stagewise_signed_redistribution", "strictly_improves",
]


if __name__ == "__main__":
    raise SystemExit(main())
