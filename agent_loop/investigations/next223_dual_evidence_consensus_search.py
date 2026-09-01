#!/usr/bin/env python3
"""Frozen dual-evidence consensus search on the exact NEXT222 path."""

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

import src.next222_forward_stagewise_signed_redistribution as n222
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-09-next223-dual-evidence-consensus-search-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT223_DUAL_EVIDENCE_CATALOGUE.json"
EVALUATION_NAME = "NEXT223_DISCOVERY_EVALUATION.json"
FORMULA_NAME = "NEXT223_FROZEN_CANDIDATE.json"
SEARCH_NAME = "next223_dual_evidence_consensus_search.parquet"
SCORE_COMPOSITION = "nonnegative_next222_plus_allocated_dual_evidence_band_term"
BETA_FRACTIONS = (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4)
PROTECTION_BUDGET_FRACTIONS = (0.25, 0.5, 0.75)
EXPECTED_HYPOTHESIS_COUNT = 22
EXPECTED_TOTAL_CANDIDATE_COUNT = 1 + (
    EXPECTED_HYPOTHESIS_COUNT**2
    * len(BETA_FRACTIONS)
    * len(PROTECTION_BUDGET_FRACTIONS)
)
EXPECTED_CONTROL_COUNT = EXPECTED_HYPOTHESIS_COUNT * len(BETA_FRACTIONS)
EXPECTED_ELIGIBLE_COUNT = (
    EXPECTED_TOTAL_CANDIDATE_COUNT - EXPECTED_CONTROL_COUNT - 1
)
REQUIRED_STAGES = (*n222.REQUIRED_STAGES, 222)
REQUIRED_DESIGN_STAGES = (*n222.REQUIRED_DESIGN_STAGES, 222)
SEARCH_WORKERS = n222.SEARCH_WORKERS
BOUNDARY_FLAGS = n222.BOUNDARY_FLAGS
EXPECTED_DESIGN_SHA256 = (
    "cf04c3b005b80968782934f708ba7fce7bcdd2d08c4bbbfceb1a46f8325be92d"
)
EXPECTED_NEXT222_SOURCE_SHA256 = (
    "8eece3a3e746253ec7bd51910f84d41161bdc8c216183dbec0b5910d3d73844b"
)
EXPECTED_NEXT222_FINAL_KEY_SHA256 = (
    "df456754fe58852f65fd2e2fd2977e47d4d80f4ba01cc90018e7bcdd394335ea"
)
EXPECTED_NEXT222_FAILED_COUNT = 6
EXPECTED_NEXT222_SHORTFALL = 0.1564570050830728
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n222.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next222_design": n222.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next222_manifest": (
        "2626ca3660dd7c218916c30ca35d4b8679fb1a390b64a078a9dfd7d0db8d0232"
    ),
    "next222_catalogue": (
        "e4c2a6bc0388294fd280be43e272eb8cf4be4748317332c26c89b5ff3c024fd7"
    ),
    "next222_evaluation": (
        "86fd288ef0034c7d89d00527b65ca0aaee83e14f3fe04e157e0afe8157118a26"
    ),
    "next222_formula": (
        "42a20e31643af65d73abdda494d873dfc3b7fe94e6221cb8b45117d081d8c9f4"
    ),
    "next222_search": (
        "31fc875bfc70bf3c72ef088ae90b3616dc03fdcc2ee25542816d481830aa8156"
    ),
}


def dual_evidence_consensus_score(
    *,
    base_score: object,
    current_delta: object,
    base_support: object,
    protection_values: object,
    protection_direction: str,
    protection_q_lo: float,
    protection_q_hi: float,
    risk_values: object,
    risk_direction: str,
    risk_q_lo: float,
    risk_q_hi: float,
    lower: float,
    upper: float,
    beta_fraction: float,
    protection_budget_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Add one ordered dual-evidence term on the original repair band."""

    base = np.asarray(base_score, dtype=float)
    current = np.asarray(current_delta, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    protect_raw = np.asarray(protection_values, dtype=float)
    risk_raw = np.asarray(risk_values, dtype=float)
    low = float(lower)
    high = float(upper)
    beta = float(beta_fraction)
    budget = float(protection_budget_fraction)
    if (
        base.ndim != 1
        or current.shape != base.shape
        or support.shape != base.shape
        or protect_raw.shape != base.shape
        or risk_raw.shape != base.shape
        or not math.isfinite(low)
        or not math.isfinite(high)
        or not high > low
        or not math.isfinite(beta)
        or not 0.0 < beta <= 0.25
        or budget not in PROTECTION_BUDGET_FRACTIONS
        or protection_direction not in n222.n215.PROTECTION_DIRECTIONS
        or risk_direction not in n222.n215.PROTECTION_DIRECTIONS
        or np.any(~np.isfinite(base[support]))
        or np.any(~np.isfinite(current[support]))
    ):
        raise ValueError("NEXT223 dual-evidence score inputs differ")
    protect = n222.n220.n216.bounded_directional_protection(
        protect_raw,
        protection_direction,
        float(protection_q_lo),
        float(protection_q_hi),
    )
    risk_protection = n222.n220.n216.bounded_directional_protection(
        risk_raw,
        risk_direction,
        float(risk_q_lo),
        float(risk_q_hi),
    )
    active = (
        support
        & (base >= low)
        & (base < high)
        & np.isfinite(protect)
        & np.isfinite(risk_protection)
    )
    proposed = current.copy()
    width = high - low
    proposed[active] += 2.0 * beta * width * (
        (1.0 - budget) * (1.0 - risk_protection[active])
        - budget * protect[active]
    )
    score = np.maximum(0.0, base + proposed)
    return score, support.copy(), active, proposed


def build_dual_evidence_candidate_specs(
    *,
    current_path_key: str,
    current_terms: Sequence[Mapping[str, object]],
    normalizations: Mapping[str, Mapping[str, object]],
    beta_fractions: Sequence[float] = BETA_FRACTIONS,
    protection_budget_fractions: Sequence[float] = PROTECTION_BUDGET_FRACTIONS,
) -> list[dict[str, object]]:
    """Build the frozen no-op plus complete ordered allocated-pair grammar."""

    if not isinstance(current_path_key, str) or not current_path_key:
        raise ValueError("NEXT223 current path key differs")
    terms = [dict(term) for term in current_terms]
    if not terms or not isinstance(normalizations, Mapping) or not normalizations:
        raise ValueError("NEXT223 current terms or normalizations differ")
    hypotheses = sorted(str(value) for value in normalizations)
    if len(hypotheses) != len(set(hypotheses)):
        raise ValueError("NEXT223 hypothesis identities differ")
    for hypothesis in hypotheses:
        norm = dict(normalizations[hypothesis])
        if (
            str(norm.get("hypothesis", hypothesis)) != hypothesis
            or norm.get("direction") not in n222.n215.PROTECTION_DIRECTIONS
            or norm.get("feature") is None
            or not math.isfinite(float(norm.get("q_lo", math.nan)))
            or not math.isfinite(float(norm.get("q_hi", math.nan)))
        ):
            raise ValueError("NEXT223 normalization record differs")
    betas = tuple(float(value) for value in beta_fractions)
    budgets = tuple(float(value) for value in protection_budget_fractions)
    if (
        not betas
        or any(not 0.0 < value <= 0.25 for value in betas)
        or len(betas) != len(set(betas))
        or budgets != tuple(value for value in PROTECTION_BUDGET_FRACTIONS if value in budgets)
        or len(budgets) != len(set(budgets))
    ):
        raise ValueError("NEXT223 amplitude or budget grid differs")
    common = {
        "parent_path_key_sha256": hashlib.sha256(
            current_path_key.encode()
        ).hexdigest(),
        "current_terms": terms,
        "missing_policy": "PAIR_TERM_OFF_KEEP_CURRENT_PATH",
        "score_composition": SCORE_COMPOSITION,
    }
    payloads: list[dict[str, object]] = [
        {
            **common,
            "protection_hypothesis": None,
            "protection_feature": None,
            "protection_direction": None,
            "protection_q_lo": None,
            "protection_q_hi": None,
            "risk_hypothesis": None,
            "risk_feature": None,
            "risk_direction": None,
            "risk_q_lo": None,
            "risk_q_hi": None,
            "beta_fraction": 0.0,
            "protection_budget_fraction": None,
            "is_reproduction_control": False,
            "eligible_new_candidate": False,
        }
    ]
    for protection_hypothesis in hypotheses:
        protection = dict(normalizations[protection_hypothesis])
        for risk_hypothesis in hypotheses:
            risk = dict(normalizations[risk_hypothesis])
            for beta in betas:
                for budget in budgets:
                    control = bool(
                        protection_hypothesis == risk_hypothesis
                        and budget == 0.5
                    )
                    payloads.append(
                        {
                            **common,
                            "protection_hypothesis": protection_hypothesis,
                            "protection_feature": str(protection["feature"]),
                            "protection_direction": str(protection["direction"]),
                            "protection_q_lo": float(protection["q_lo"]),
                            "protection_q_hi": float(protection["q_hi"]),
                            "risk_hypothesis": risk_hypothesis,
                            "risk_feature": str(risk["feature"]),
                            "risk_direction": str(risk["direction"]),
                            "risk_q_lo": float(risk["q_lo"]),
                            "risk_q_hi": float(risk["q_hi"]),
                            "beta_fraction": beta,
                            "protection_budget_fraction": budget,
                            "is_reproduction_control": control,
                            "eligible_new_candidate": not control,
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
        raise RuntimeError("NEXT223 candidate keys are not unique")
    return specs


def _paths(
    *,
    roots: Mapping[str, Path],
    next135_freeze_path: Path,
    design_paths: Mapping[int, Path],
    design_path: Path,
) -> dict[str, Path]:
    paths = n222._paths(
        roots=roots,
        next135_freeze_path=next135_freeze_path,
        design_paths={
            stage: design_paths[stage] for stage in n222.REQUIRED_DESIGN_STAGES
        },
        design_path=design_paths[222],
    )
    paths["next222_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next222_manifest": roots["next222"] / n222.MANIFEST_NAME,
            "next222_catalogue": roots["next222"] / n222.CATALOGUE_NAME,
            "next222_evaluation": roots["next222"] / n222.EVALUATION_NAME,
            "next222_formula": roots["next222"] / n222.FORMULA_NAME,
            "next222_search": roots["next222"] / n222.SEARCH_NAME,
        }
    )
    return paths


def _verify_next222(
    paths: Mapping[str, Path], input_hashes: Mapping[str, str]
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    str,
    str,
    dict[str, object],
    str,
    dict[str, object],
    pd.DataFrame,
]:
    prior_paths = dict(paths)
    prior_paths["design"] = paths["next222_design"]
    prior_hashes = dict(input_hashes)
    prior_hashes["design"] = input_hashes["next222_design"]
    (
        eligible,
        eligible214,
        primary_key,
        base_start_key,
        formula214,
        _,
    ) = n222._verify_next221(prior_paths, prior_hashes)
    manifest = json.loads(paths["next222_manifest"].read_text())
    catalogue = json.loads(paths["next222_catalogue"].read_text())
    evaluation = json.loads(paths["next222_evaluation"].read_text())
    formula = json.loads(paths["next222_formula"].read_text())
    table = pd.read_parquet(paths["next222_search"])
    accepted = table.loc[
        table["depth"].eq(2)
        & table["proposed_hypothesis"].eq(
            "prlr_contact_weight_rms__protected_low"
        )
        & table["proposed_beta_fraction"].eq(1 / 64)
    ]
    final_key = "" if len(accepted) != 1 else str(accepted.iloc[0]["candidate_key"])
    expected_outputs = {
        n222.CATALOGUE_NAME: input_hashes["next222_catalogue"],
        n222.EVALUATION_NAME: input_hashes["next222_evaluation"],
        n222.FORMULA_NAME: input_hashes["next222_formula"],
        n222.SEARCH_NAME: input_hashes["next222_search"],
    }
    if (
        manifest.get("protocol") != n222.PROTOCOL
        or manifest.get("final_signed_term_count") != 2
        or manifest.get("stop_reason") != "no_strict_residual_improvement"
        or manifest.get("passes_all_cross_source_discovery_gates") is not False
        or manifest.get("outputs_sha256") != expected_outputs
        or any(manifest.get(key) is not value for key, value in BOUNDARY_FLAGS.items())
        or manifest.get("executed_source_sha256", {}).get(
            "src/next222_forward_stagewise_signed_redistribution.py"
        )
        != EXPECTED_NEXT222_SOURCE_SHA256
        or _sha256_file(Path(n222.__file__).resolve())
        != EXPECTED_NEXT222_SOURCE_SHA256
        or catalogue.get("eligible_hypothesis_count") != EXPECTED_HYPOTHESIS_COUNT
        or evaluation.get("all_discovery_gates_passed") is not False
        or evaluation.get("final_failed_constraint_count")
        != EXPECTED_NEXT222_FAILED_COUNT
        or not math.isclose(
            float(evaluation.get("final_normalized_shortfall_sum", math.nan)),
            EXPECTED_NEXT222_SHORTFALL,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or formula.get("signed_term_count") != 2
        or any(
            formula.get(key) is not False
            for key in (
                "dft_values_used_by_executable_formula",
                "learned_energy_force_stress_proxy_used",
                "model_or_proxy_potential_used",
                "physical_relaxation_executed",
            )
        )
        or len(table) != int(evaluation.get("total_candidate_evaluations", -1))
        or hashlib.sha256(final_key.encode()).hexdigest()
        != EXPECTED_NEXT222_FINAL_KEY_SHA256
    ):
        raise ValueError("NEXT223 NEXT222 provenance differs")
    return (
        eligible,
        eligible214,
        primary_key,
        base_start_key,
        formula214,
        final_key,
        formula,
        table,
    )


def _reconstruct_next222_delta(
    *,
    features: pd.DataFrame,
    base_score: np.ndarray,
    support: np.ndarray,
    formula: Mapping[str, object],
) -> np.ndarray:
    current = np.zeros(len(features), dtype=float)
    terms = formula.get("terms")
    if not isinstance(terms, list) or len(terms) != 2:
        raise ValueError("NEXT223 NEXT222 formula terms differ")
    for raw in terms:
        term = dict(raw)
        feature = str(term.get("feature", ""))
        if feature not in features.columns:
            raise ValueError("NEXT223 NEXT222 term feature differs")
        values = pd.to_numeric(features[feature], errors="coerce").to_numpy(float)
        _, got_support, _, current = n222.cumulative_signed_score(
            base_score=base_score,
            current_delta=current,
            base_support=support,
            feature_values=values,
            direction=str(term["direction"]),
            q_lo=float(term["q_lo"]),
            q_hi=float(term["q_hi"]),
            lower=n222.n215.REPAIR_LOWER_THRESHOLD,
            upper=n222.n215.REPAIR_UPPER_THRESHOLD,
            beta_fraction=float(term["beta_fraction"]),
        )
        if not np.array_equal(got_support, support):
            raise RuntimeError("NEXT223 NEXT222 support differs")
    return current


def materialize_dual_evidence_candidates(
    *,
    features: pd.DataFrame,
    base_score: object,
    current_delta: object,
    base_support: object,
    specs: Sequence[Mapping[str, object]],
) -> tuple[
    pd.DataFrame,
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, dict[str, int]],
]:
    """Encode every exact NEXT223 score as one evaluator term."""

    base = np.asarray(base_score, dtype=float)
    current = np.asarray(current_delta, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    raw_specs = [dict(value) for value in specs]
    if (
        not isinstance(features, pd.DataFrame)
        or base.shape != (len(features),)
        or current.shape != base.shape
        or support.shape != base.shape
        or not raw_specs
        or len({str(spec.get("candidate_key", "")) for spec in raw_specs})
        != len(raw_specs)
    ):
        raise ValueError("NEXT223 materializer inputs differ")
    source = features["source_dataset"].astype(str).to_numpy()
    raw_cache: dict[str, np.ndarray] = {}
    columns: dict[str, np.ndarray] = {}
    terms: list[dict[str, object]] = []
    runtime: list[dict[str, object]] = []
    activity: dict[str, dict[str, int]] = {}
    for spec in raw_specs:
        protection_feature = spec.get("protection_feature")
        risk_feature = spec.get("risk_feature")
        if protection_feature is None and risk_feature is None:
            score = np.maximum(0.0, base + current)
            got_support = support.copy()
            active = np.zeros(len(features), dtype=bool)
        else:
            if (
                protection_feature is None
                or risk_feature is None
                or str(protection_feature) not in features.columns
                or str(risk_feature) not in features.columns
            ):
                raise ValueError("NEXT223 materializer feature differs")
            for feature in (str(protection_feature), str(risk_feature)):
                if feature not in raw_cache:
                    raw_cache[feature] = pd.to_numeric(
                        features[feature], errors="coerce"
                    ).to_numpy(float)
            score, got_support, active, _ = dual_evidence_consensus_score(
                base_score=base,
                current_delta=current,
                base_support=support,
                protection_values=raw_cache[str(protection_feature)],
                protection_direction=str(spec["protection_direction"]),
                protection_q_lo=float(spec["protection_q_lo"]),
                protection_q_hi=float(spec["protection_q_hi"]),
                risk_values=raw_cache[str(risk_feature)],
                risk_direction=str(spec["risk_direction"]),
                risk_q_lo=float(spec["risk_q_lo"]),
                risk_q_hi=float(spec["risk_q_hi"]),
                lower=n222.n215.REPAIR_LOWER_THRESHOLD,
                upper=n222.n215.REPAIR_UPPER_THRESHOLD,
                beta_fraction=float(spec["beta_fraction"]),
                protection_budget_fraction=float(
                    spec["protection_budget_fraction"]
                ),
            )
        maximum = float(np.max(score[got_support])) if got_support.any() else 0.0
        divisor = max(1.0, maximum / 50.0)
        encoded = np.full(len(features), np.nan)
        encoded[got_support] = np.sinh(score[got_support] / divisor)
        key = str(spec["candidate_key"])
        term_id = (
            "next223_virtual_candidate__"
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
                "group": "next223_dual_evidence",
                "encoding": "asinh_sinh_exact_allocated_dual_evidence_score",
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
        activity[key] = {
            "rows": int(active.sum()),
            "scigen": int((active & (source == "scigen")).sum()),
            "wyformer": int((active & (source == "wyformer")).sum()),
        }
    virtual = pd.concat(
        [features.reset_index(drop=True), pd.DataFrame(columns)], axis=1
    )
    return virtual, terms, runtime, activity


def _record_rank(record: Mapping[str, object]) -> tuple[object, ...]:
    return (
        int(bool(record["passes_all_discovery_gates"])),
        int(bool(record["passes_broad_all_cells"])),
        int(record["safe_passing_cells"]),
        float(record["safe_worst_cell_severe_recall"]),
        float(record["safe_worst_cell_precision_lower"]),
        min(float(record["scigen_pooled_auc"]), float(record["wyformer_pooled_auc"])),
        -int(record["term_count"]),
    )


def select_best_eligible_record(records: pd.DataFrame) -> pd.Series:
    """Apply the unchanged evaluator rank within eligible AUC+SAFE records."""

    eligible = records.loc[
        records["eligible_new_candidate"].fillna(False).astype(bool)
        & records["passes_source_auc_gates"].fillna(False).astype(bool)
        & records["passes_safe_all_cells"].fillna(False).astype(bool)
    ]
    if eligible.empty:
        raise RuntimeError("NEXT223 produced no eligible AUC+SAFE candidate")
    ranked = sorted(
        (row for _, row in eligible.iterrows()),
        key=lambda row: (
            tuple(-float(value) for value in _record_rank(row)),
            str(row["candidate_key"]),
        ),
    )
    return ranked[0]


def _gate_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = {
        name: int(frame[name].fillna(False).astype(bool).sum())
        for name in (
            "passes_source_auc_gates",
            "passes_safe_all_cells",
            "passes_broad_all_cells",
            "passes_all_discovery_gates",
        )
    }
    counts["passes_auc_and_safe_but_not_broad"] = int(
        (
            frame["passes_source_auc_gates"].fillna(False).astype(bool)
            & frame["passes_safe_all_cells"].fillna(False).astype(bool)
            & ~frame["passes_broad_all_cells"].fillna(False).astype(bool)
        ).sum()
    )
    return counts


def _assert_record_reproduction(
    observed: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    metrics = (
        "safe_threshold",
        "scigen_pooled_auc",
        "scigen_macro_auc",
        "scigen_worst_auc",
        "wyformer_pooled_auc",
        "wyformer_macro_auc",
        "wyformer_worst_auc",
    )
    for name in metrics:
        left = observed[name]
        right = expected[name]
        left_missing = left is None or bool(pd.isna(left))
        right_missing = right is None or bool(pd.isna(right))
        if left_missing or right_missing:
            if left_missing != right_missing:
                raise RuntimeError("NEXT223 control metric differs")
            continue
        elif not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("NEXT223 control metric differs")
    for name in (
        "passes_source_auc_gates",
        "passes_safe_all_cells",
        "passes_broad_all_cells",
        "passes_all_discovery_gates",
    ):
        if bool(observed[name]) != bool(expected[name]):
            raise RuntimeError("NEXT223 control gate differs")


def run_dual_evidence_consensus_search(
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
    search_workers: int = n222.SEARCH_WORKERS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only NEXT223 search."""

    if not set(REQUIRED_STAGES).issubset(stage_dirs):
        raise ValueError("NEXT223 stage directory universe differs")
    if not set(REQUIRED_DESIGN_STAGES).issubset(design_paths):
        raise ValueError("NEXT223 design path universe differs")
    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{
            f"next{stage}": Path(stage_dirs[stage]).resolve()
            for stage in REQUIRED_STAGES
        },
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots=roots,
        next135_freeze_path=Path(next135_freeze_path).resolve(),
        design_paths={
            stage: Path(design_paths[stage]).resolve()
            for stage in REQUIRED_DESIGN_STAGES
        },
        design_path=Path(design_path).resolve(),
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(search_workers) is not int or search_workers <= 0:
        raise ValueError("NEXT223 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT223 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT223 formal input identity differs: {differing}")
    (
        eligible,
        eligible214,
        primary_key,
        base_start_key,
        formula214,
        current_key,
        formula222,
        table222,
    ) = _verify_next222(paths, input_hashes)
    combined, feature_tables, base_score, support, endpoint = (
        n222.n215._reconstruct_next214_final(
            paths=paths,
            eligible=eligible214,
            primary_key=primary_key,
            start_key=base_start_key,
            formula=formula214,
        )
    )
    current_delta = _reconstruct_next222_delta(
        features=combined,
        base_score=base_score,
        support=support,
        formula=formula222,
    )
    next214_table = pd.read_parquet(paths["next214_search"])
    accepted214 = next214_table.loc[
        next214_table["depth"].eq(3)
        & next214_table["proposed_hypothesis"].eq(
            "steric_overlap2_vector_q95__protected_low"
        )
        & next214_table["proposed_amplitude_fraction"].eq(0.0625)
    ]
    if len(accepted214) != 1:
        raise ValueError("NEXT223 NEXT214 base identity differs")
    initial_specs = n222.n220.build_signed_candidate_specs(
        base_candidate_key=str(accepted214.iloc[0]["candidate_key"]),
        eligible_hypotheses=eligible,
        features=combined,
        base_score=base_score,
        base_support=support,
    )
    normalizations = {
        str(spec["hypothesis"]): dict(spec)
        for spec in initial_specs
        if spec["hypothesis"] is not None
    }
    if len(normalizations) != EXPECTED_HYPOTHESIS_COUNT:
        raise ValueError("NEXT223 eligible normalization universe differs")
    specs = build_dual_evidence_candidate_specs(
        current_path_key=current_key,
        current_terms=[dict(value) for value in formula222["terms"]],
        normalizations=normalizations,
    )
    control_count = sum(bool(spec["is_reproduction_control"]) for spec in specs)
    eligible_count = sum(bool(spec["eligible_new_candidate"]) for spec in specs)
    if (
        len(specs) != EXPECTED_TOTAL_CANDIDATE_COUNT
        or control_count != EXPECTED_CONTROL_COUNT
        or eligible_count != EXPECTED_ELIGIBLE_COUNT
    ):
        raise RuntimeError("NEXT223 frozen candidate universe differs")
    virtual, terms, runtime, activity = materialize_dual_evidence_candidates(
        features=combined,
        base_score=base_score,
        current_delta=current_delta,
        base_support=support,
        specs=specs,
    )
    runtime_by_key = {str(value["candidate_key"]): value for value in runtime}
    eligible_runtime = [
        runtime_by_key[str(spec["candidate_key"])]
        for spec in specs
        if spec["eligible_new_candidate"]
    ]
    fixed_runtime = [
        runtime_by_key[str(spec["candidate_key"])]
        for spec in specs
        if not spec["eligible_new_candidate"]
    ]
    evaluator = (
        n222.n215.n214.n212.n210.n208.n205.n203.n202.n200.n194.n130.n125
        .search_optional_guard_laws_parallel
    )
    started = time.perf_counter()
    eligible_result = evaluator(
        features=virtual,
        endpoint=endpoint,
        old_terms=terms,
        optional_terms=[],
        candidate_specs=eligible_runtime,
        workers=search_workers,
    )
    fixed_result = evaluator(
        features=virtual,
        endpoint=endpoint,
        old_terms=terms,
        optional_terms=[],
        candidate_specs=fixed_runtime,
        workers=search_workers,
    )
    elapsed = time.perf_counter() - started
    if (
        int(eligible_result["candidate_count"]) != EXPECTED_ELIGIBLE_COUNT
        or int(fixed_result["candidate_count"])
        != EXPECTED_TOTAL_CANDIDATE_COUNT - EXPECTED_ELIGIBLE_COUNT
        or eligible_result["cells"] != fixed_result["cells"]
        or eligible_result["pauling_by_cell"] != fixed_result["pauling_by_cell"]
    ):
        raise RuntimeError("NEXT223 evaluator accounting differs")
    spec_by_key = {str(spec["candidate_key"]): spec for spec in specs}
    raw_records = [
        *eligible_result["candidate_records"],
        *fixed_result["candidate_records"],
    ]
    for record in raw_records:
        spec = spec_by_key[str(record["candidate_key"])]
        record.update(
            {
                "protection_hypothesis": spec["protection_hypothesis"],
                "protection_feature": spec["protection_feature"],
                "protection_direction": spec["protection_direction"],
                "protection_q_lo": spec["protection_q_lo"],
                "protection_q_hi": spec["protection_q_hi"],
                "risk_hypothesis": spec["risk_hypothesis"],
                "risk_feature": spec["risk_feature"],
                "risk_direction": spec["risk_direction"],
                "risk_q_lo": spec["risk_q_lo"],
                "risk_q_hi": spec["risk_q_hi"],
                "beta_fraction": spec["beta_fraction"],
                "protection_budget_fraction": spec[
                    "protection_budget_fraction"
                ],
                "is_reproduction_control": spec["is_reproduction_control"],
                "eligible_new_candidate": spec["eligible_new_candidate"],
                "pair_active_rows": activity[str(record["candidate_key"])]["rows"],
                "pair_active_scigen": activity[str(record["candidate_key"])][
                    "scigen"
                ],
                "pair_active_wyformer": activity[str(record["candidate_key"])][
                    "wyformer"
                ],
                "missing_policy": spec["missing_policy"],
                "score_composition": SCORE_COMPOSITION,
            }
        )
    records = pd.DataFrame(raw_records).sort_values(
        "candidate_key", kind="mergesort"
    ).reset_index(drop=True)

    no_op_key = str(specs[0]["candidate_key"])
    no_op = records.loc[records["candidate_key"].eq(no_op_key)]
    reference_no_op = table222.loc[
        table222["depth"].eq(3) & table222["proposed_hypothesis"].isna()
    ]
    if len(no_op) != 1 or len(reference_no_op) != 1:
        raise RuntimeError("NEXT223 no-op reproduction identity differs")
    _assert_record_reproduction(no_op.iloc[0], reference_no_op.iloc[0])
    used = {str(term["hypothesis"]) for term in formula222["terms"]}
    reproduced_controls = 0
    for spec in specs:
        hypothesis = spec["protection_hypothesis"]
        if (
            not spec["is_reproduction_control"]
            or hypothesis in used
        ):
            continue
        expected = table222.loc[
            table222["depth"].eq(3)
            & table222["proposed_hypothesis"].eq(hypothesis)
            & table222["proposed_beta_fraction"].eq(spec["beta_fraction"])
        ]
        observed = records.loc[records["candidate_key"].eq(spec["candidate_key"])]
        if len(expected) != 1 or len(observed) != 1:
            raise RuntimeError("NEXT223 diagonal control identity differs")
        _assert_record_reproduction(observed.iloc[0], expected.iloc[0])
        reproduced_controls += 1
    if reproduced_controls != 100:
        raise RuntimeError("NEXT223 diagonal reproduction count differs")

    selected_row = select_best_eligible_record(records)
    selected_key = str(selected_row["candidate_key"])
    selected_result = evaluator(
        features=virtual,
        endpoint=endpoint,
        old_terms=terms,
        optional_terms=[],
        candidate_specs=[runtime_by_key[selected_key]],
        workers=1,
    )
    if str(selected_result["selected"]["record"]["candidate_key"]) != selected_key:
        raise RuntimeError("NEXT223 selected candidate reproduction differs")
    selected = selected_result["selected"]
    selected_spec = spec_by_key[selected_key]
    for name, value in selected_row.items():
        if name in selected["record"]:
            selected["record"][name] = value
    eligible_frame = records.loc[records["eligible_new_candidate"].astype(bool)]
    controls_frame = records.loc[records["is_reproduction_control"].astype(bool)]
    passes = bool(
        eligible_frame["passes_all_discovery_gates"].fillna(False).astype(bool).any()
    )
    if passes != bool(selected["record"]["passes_all_discovery_gates"]):
        raise RuntimeError("NEXT223 all-gate selection differs")
    diagnostic_mask = (
        eligible_frame["passes_source_auc_gates"].fillna(False).astype(bool)
        & eligible_frame["passes_safe_all_cells"].fillna(False).astype(bool)
        & ~eligible_frame["passes_broad_all_cells"].fillna(False).astype(bool)
    )
    diagnostic_keys = sorted(
        eligible_frame.loc[diagnostic_mask, "candidate_key"].astype(str)
    )
    diagnostic_sha = hashlib.sha256(
        "\n".join(diagnostic_keys).encode()
    ).hexdigest()
    formula = {
        "protocol": PROTOCOL,
        "kind": "allocated_dual_evidence_consensus_x0_no_dft_score",
        "base_protocol": n222.PROTOCOL,
        "base_candidate_key_sha256": EXPECTED_NEXT222_FINAL_KEY_SHA256,
        "base_terms": formula222["terms"],
        "protection_hypothesis": selected_spec["protection_hypothesis"],
        "protection_feature": selected_spec["protection_feature"],
        "protection_direction": selected_spec["protection_direction"],
        "protection_q_lo": selected_spec["protection_q_lo"],
        "protection_q_hi": selected_spec["protection_q_hi"],
        "risk_hypothesis": selected_spec["risk_hypothesis"],
        "risk_feature": selected_spec["risk_feature"],
        "risk_direction": selected_spec["risk_direction"],
        "risk_q_lo": selected_spec["risk_q_lo"],
        "risk_q_hi": selected_spec["risk_q_hi"],
        "beta_fraction": selected_spec["beta_fraction"],
        "protection_budget_fraction": selected_spec[
            "protection_budget_fraction"
        ],
        "repair_lower_threshold": n222.n215.REPAIR_LOWER_THRESHOLD,
        "repair_upper_threshold": n222.n215.REPAIR_UPPER_THRESHOLD,
        "nonnegative_floor": 0.0,
        "support_policy": "UNCHANGED_FROM_NEXT214",
        "missing_policy": "PAIR_TERM_OFF_KEEP_NEXT222_PATH",
        "score_composition": SCORE_COMPOSITION,
        "dft_values_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
    }
    catalogue = {
        "protocol": PROTOCOL,
        "design_sha256": input_hashes["design"],
        "base_candidate_key_sha256": EXPECTED_NEXT222_FINAL_KEY_SHA256,
        "eligible_hypotheses": list(eligible),
        "eligible_hypothesis_count": len(eligible),
        "eligible_hypothesis_sha256": n222.n220.EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256,
        "beta_fractions": list(BETA_FRACTIONS),
        "protection_budget_fractions": list(PROTECTION_BUDGET_FRACTIONS),
        "candidate_count": len(records),
        "eligible_new_candidate_count": len(eligible_frame),
        "diagonal_control_count": len(controls_frame),
        "exact_next222_depth3_reproduction_control_count": reproduced_controls,
        "normalization_refit": False,
        "normalization_fit_uses_endpoint": False,
        "base_support_unchanged": True,
        "ordered_role_allocation": True,
        "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }
    evaluation = {
        "protocol": PROTOCOL,
        "evaluation_mode": "frozen_allocated_dual_evidence_discovery_search",
        "next222_final_path_reproduced": True,
        "next222_depth3_control_count_reproduced": reproduced_controls,
        "rows": {
            "scigen": int(len(feature_tables["scigen"])),
            "wyformer": int(len(feature_tables["wyformer"])),
            "total": int(len(combined)),
        },
        "candidate_count": len(records),
        "eligible_new_candidate_count": len(eligible_frame),
        "diagonal_control_count": len(controls_frame),
        "elapsed_seconds": elapsed,
        "search_workers": search_workers,
        "counts_all": _gate_counts(records),
        "counts_eligible_new": _gate_counts(eligible_frame),
        "counts_diagonal_controls": _gate_counts(controls_frame),
        "selected_record": selected["record"],
        "selected_formula": formula,
        "selected_safe": selected["safe"],
        "selected_safe_diagnostic": selected["safe_diagnostic"],
        "selected_broad": selected["broad"],
        "selected_source_diagnostics": selected["source_diagnostics"],
        "pauling_by_cell": eligible_result["pauling_by_cell"],
        "cells": eligible_result["cells"],
        "passes_all_cross_source_discovery_gates": passes,
        "freeze_authorized": passes,
        "next224_diagnostic_authorized": bool(not passes and diagnostic_keys),
        "next224_candidate_count": len(diagnostic_keys),
        "next224_candidate_key_sha256": diagnostic_sha,
        "requires_unopened_internal_validation_before_claim": True,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    source_paths = {
        "src/next222_forward_stagewise_signed_redistribution.py": Path(
            n222.__file__
        ).resolve(),
        "src/next223_dual_evidence_consensus_search.py": Path(__file__).resolve(),
    }
    source_hashes = {
        name: _sha256_file(path) for name, path in source_paths.items()
    }
    try:
        catalogue_path = staging / CATALOGUE_NAME
        evaluation_path = staging / EVALUATION_NAME
        formula_path = staging / FORMULA_NAME
        search_path = staging / SEARCH_NAME
        _write_json(catalogue_path, catalogue)
        _write_json(evaluation_path, evaluation)
        _write_json(formula_path, formula)
        records.to_parquet(search_path, index=False)
        outputs = [catalogue_path, evaluation_path, formula_path, search_path]
        manifest = {
            "protocol": PROTOCOL,
            "candidate_count": len(records),
            "eligible_new_candidate_count": len(eligible_frame),
            "diagonal_control_count": len(controls_frame),
            "exact_next222_depth3_reproduction_control_count": reproduced_controls,
            "eligible_hypothesis_count": len(eligible),
            "eligible_hypothesis_sha256": (
                n222.n220.EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256
            ),
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
            "next224_diagnostic_authorized": bool(not passes and diagnostic_keys),
            "next224_candidate_count": len(diagnostic_keys),
            "next224_candidate_key_sha256": diagnostic_sha,
            "requires_unopened_internal_validation_before_claim": True,
            "scigen_discovery_endpoint_opened": True,
            "wyformer_discovery_endpoint_opened": True,
            "discovery_outcomes_used_as_offline_labels": True,
            **BOUNDARY_FLAGS,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {
                path.name: _sha256_file(path) for path in outputs
            },
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(
            _sha256_file(path) != input_hashes[name]
            for name, path in paths.items()
        ):
            raise RuntimeError("NEXT223 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT223 source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scigen-feature-dir", type=Path, required=True)
    parser.add_argument(
        "--scigen-discovery-endpoint-dir", type=Path, required=True
    )
    parser.add_argument("--wyformer-feature-dir", type=Path, required=True)
    parser.add_argument(
        "--wyformer-discovery-endpoint-dir", type=Path, required=True
    )
    for stage in REQUIRED_STAGES:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    for stage in REQUIRED_DESIGN_STAGES:
        parser.add_argument(
            f"--next{stage}-design-path", type=Path, required=True
        )
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=SEARCH_WORKERS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_dual_evidence_consensus_search(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        stage_dirs={
            stage: getattr(args, f"next{stage}_dir") for stage in REQUIRED_STAGES
        },
        next135_freeze_path=args.next135_freeze_path,
        design_paths={
            stage: getattr(args, f"next{stage}_design_path")
            for stage in REQUIRED_DESIGN_STAGES
        },
        design_path=args.design_path,
        output_dir=args.output_dir,
        search_workers=args.search_workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "BETA_FRACTIONS",
    "EXPECTED_CONTROL_COUNT",
    "EXPECTED_ELIGIBLE_COUNT",
    "EXPECTED_TOTAL_CANDIDATE_COUNT",
    "PROTECTION_BUDGET_FRACTIONS",
    "build_dual_evidence_candidate_specs",
    "dual_evidence_consensus_score",
    "materialize_dual_evidence_candidates",
    "run_dual_evidence_consensus_search",
    "select_best_eligible_record",
]


if __name__ == "__main__":
    raise SystemExit(main())
