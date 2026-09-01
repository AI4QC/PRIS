#!/usr/bin/env python3
"""Frozen agreement-gated consensus search on the exact NEXT222 path."""

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

import src.next224_dual_evidence_broad_diagnostic as n224
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


n223 = n224.n223
n222 = n223.n222
PROTOCOL = "2026-08-09-next225-agreement-gated-consensus-search-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT225_AGREEMENT_GATED_CATALOGUE.json"
EVALUATION_NAME = "NEXT225_DISCOVERY_EVALUATION.json"
FORMULA_NAME = "NEXT225_FROZEN_CANDIDATE.json"
SEARCH_NAME = "next225_agreement_gated_consensus_search.parquet"
SCORE_COMPOSITION = "nonnegative_next222_plus_agreement_gated_band_term"
BETA_FRACTIONS = n223.BETA_FRACTIONS
PROTECTION_BUDGET_FRACTIONS = n223.PROTECTION_BUDGET_FRACTIONS
EXPECTED_HYPOTHESIS_COUNT = n223.EXPECTED_HYPOTHESIS_COUNT
EXPECTED_PAIR_COUNT = EXPECTED_HYPOTHESIS_COUNT * (
    EXPECTED_HYPOTHESIS_COUNT + 1
) // 2
EXPECTED_TOTAL_CANDIDATE_COUNT = 1 + (
    EXPECTED_PAIR_COUNT
    * len(BETA_FRACTIONS)
    * len(PROTECTION_BUDGET_FRACTIONS)
)
EXPECTED_CONTROL_COUNT = EXPECTED_PAIR_COUNT * len(BETA_FRACTIONS)
EXPECTED_ELIGIBLE_COUNT = (
    EXPECTED_TOTAL_CANDIDATE_COUNT - EXPECTED_CONTROL_COUNT - 1
)
EXPECTED_NEXT223_OFF_DIAGONAL_CONTROL_COUNT = (
    EXPECTED_HYPOTHESIS_COUNT
    * (EXPECTED_HYPOTHESIS_COUNT - 1)
    // 2
    * len(BETA_FRACTIONS)
)
EXPECTED_NEXT222_DIAGONAL_CONTROL_COUNT = (
    (EXPECTED_HYPOTHESIS_COUNT - 2) * len(BETA_FRACTIONS)
)
EXPECTED_CLOSED_FORM_CONTROL_COUNT = 2 * len(BETA_FRACTIONS)
REQUIRED_STAGES = (*n224.REQUIRED_STAGES, 224)
REQUIRED_DESIGN_STAGES = (*n224.REQUIRED_DESIGN_STAGES, 224)
SEARCH_WORKERS = n224.SEARCH_WORKERS
BOUNDARY_FLAGS = n224.BOUNDARY_FLAGS
EXPECTED_DESIGN_SHA256 = (
    "6a30100cd6523ccc23481cf154885fc7ba13ecc27efe22da659814afc4c9a37a"
)
EXPECTED_NEXT224_SOURCE_SHA256 = (
    "9a336815d6814b60b099026eec8cba4f27d7f5dc7bdaaa408cd38e53df7cb0a8"
)
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n224.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "next224_design": n224.EXPECTED_INPUT_SHA256["design"],
    "design": EXPECTED_DESIGN_SHA256,
    "next224_manifest": (
        "343677cd335ce327b052835bd34c1272185d1a08a32b976001d144731d83efa7"
    ),
    "next224_diagnostic": (
        "70881f1d428e3d078976357476a8ec75e2bbbf1929262a7b358af7380a724587"
    ),
    "next224_table": (
        "4d51a4525bba7bc1aa35f0f4994a8b4059ffe3302208168cf45653567bf783fe"
    ),
}


def agreement_gated_consensus_score(
    *,
    base_score: object,
    current_delta: object,
    base_support: object,
    first_values: object,
    first_direction: str,
    first_q_lo: float,
    first_q_hi: float,
    second_values: object,
    second_direction: str,
    second_q_lo: float,
    second_q_hi: float,
    lower: float,
    upper: float,
    beta_fraction: float,
    protection_budget_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Add one symmetric agreement-gated term on the original repair band."""

    base = np.asarray(base_score, dtype=float)
    current = np.asarray(current_delta, dtype=float)
    support = np.asarray(base_support, dtype=bool)
    first_raw = np.asarray(first_values, dtype=float)
    second_raw = np.asarray(second_values, dtype=float)
    low = float(lower)
    high = float(upper)
    beta = float(beta_fraction)
    budget = float(protection_budget_fraction)
    if (
        base.ndim != 1
        or current.shape != base.shape
        or support.shape != base.shape
        or first_raw.shape != base.shape
        or second_raw.shape != base.shape
        or not math.isfinite(low)
        or not math.isfinite(high)
        or not high > low
        or not math.isfinite(beta)
        or not 0.0 < beta <= 0.25
        or budget not in PROTECTION_BUDGET_FRACTIONS
        or first_direction not in n222.n215.PROTECTION_DIRECTIONS
        or second_direction not in n222.n215.PROTECTION_DIRECTIONS
        or np.any(~np.isfinite(base[support]))
        or np.any(~np.isfinite(current[support]))
    ):
        raise ValueError("NEXT225 agreement-gated score inputs differ")
    first = n222.n220.n216.bounded_directional_protection(
        first_raw,
        first_direction,
        float(first_q_lo),
        float(first_q_hi),
    )
    second = n222.n220.n216.bounded_directional_protection(
        second_raw,
        second_direction,
        float(second_q_lo),
        float(second_q_hi),
    )
    active = (
        support
        & (base >= low)
        & (base < high)
        & np.isfinite(first)
        & np.isfinite(second)
    )
    proposed = current.copy()
    width = high - low
    first_active = first[active]
    second_active = second[active]
    risk_consensus = (1.0 - first_active) * (1.0 - second_active)
    protection_consensus = first_active * second_active
    proposed[active] += 2.0 * beta * width * (
        (1.0 - budget) * risk_consensus
        - budget * protection_consensus
    )
    score = np.maximum(0.0, base + proposed)
    return score, support.copy(), active, proposed


def build_agreement_candidate_specs(
    *,
    current_path_key: str,
    current_terms: Sequence[Mapping[str, object]],
    normalizations: Mapping[str, Mapping[str, object]],
    beta_fractions: Sequence[float] = BETA_FRACTIONS,
    protection_budget_fractions: Sequence[float] = PROTECTION_BUDGET_FRACTIONS,
) -> list[dict[str, object]]:
    """Build the no-op plus complete unordered agreement-gated grammar."""

    if not isinstance(current_path_key, str) or not current_path_key:
        raise ValueError("NEXT225 current path key differs")
    terms = [dict(term) for term in current_terms]
    if not terms or not isinstance(normalizations, Mapping) or not normalizations:
        raise ValueError("NEXT225 current terms or normalizations differ")
    hypotheses = sorted(str(value) for value in normalizations)
    if len(hypotheses) != len(set(hypotheses)):
        raise ValueError("NEXT225 hypothesis identities differ")
    for hypothesis in hypotheses:
        norm = dict(normalizations[hypothesis])
        if (
            str(norm.get("hypothesis", hypothesis)) != hypothesis
            or norm.get("direction") not in n222.n215.PROTECTION_DIRECTIONS
            or norm.get("feature") is None
            or not math.isfinite(float(norm.get("q_lo", math.nan)))
            or not math.isfinite(float(norm.get("q_hi", math.nan)))
        ):
            raise ValueError("NEXT225 normalization record differs")
    betas = tuple(float(value) for value in beta_fractions)
    budgets = tuple(float(value) for value in protection_budget_fractions)
    if (
        not betas
        or any(not 0.0 < value <= 0.25 for value in betas)
        or len(betas) != len(set(betas))
        or budgets
        != tuple(
            value for value in PROTECTION_BUDGET_FRACTIONS if value in budgets
        )
        or len(budgets) != len(set(budgets))
    ):
        raise ValueError("NEXT225 amplitude or budget grid differs")
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
            "first_hypothesis": None,
            "first_feature": None,
            "first_direction": None,
            "first_q_lo": None,
            "first_q_hi": None,
            "second_hypothesis": None,
            "second_feature": None,
            "second_direction": None,
            "second_q_lo": None,
            "second_q_hi": None,
            "beta_fraction": 0.0,
            "protection_budget_fraction": None,
            "is_reproduction_control": False,
            "eligible_new_candidate": False,
        }
    ]
    for first_index, first_hypothesis in enumerate(hypotheses):
        first = dict(normalizations[first_hypothesis])
        for second_hypothesis in hypotheses[first_index:]:
            second = dict(normalizations[second_hypothesis])
            for beta in betas:
                for budget in budgets:
                    control = bool(budget == 0.5)
                    payloads.append(
                        {
                            **common,
                            "first_hypothesis": first_hypothesis,
                            "first_feature": str(first["feature"]),
                            "first_direction": str(first["direction"]),
                            "first_q_lo": float(first["q_lo"]),
                            "first_q_hi": float(first["q_hi"]),
                            "second_hypothesis": second_hypothesis,
                            "second_feature": str(second["feature"]),
                            "second_direction": str(second["direction"]),
                            "second_q_lo": float(second["q_lo"]),
                            "second_q_hi": float(second["q_hi"]),
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
        raise RuntimeError("NEXT225 candidate keys are not unique")
    return specs


def _paths(
    *,
    roots: Mapping[str, Path],
    next135_freeze_path: Path,
    design_paths: Mapping[int, Path],
    design_path: Path,
) -> dict[str, Path]:
    paths = n224._paths(
        roots=roots,
        next135_freeze_path=next135_freeze_path,
        design_paths={
            stage: design_paths[stage]
            for stage in n224.REQUIRED_DESIGN_STAGES
        },
        design_path=design_paths[224],
    )
    paths["next224_design"] = paths.pop("design")
    paths.update(
        {
            "design": design_path,
            "next224_manifest": roots["next224"] / n224.MANIFEST_NAME,
            "next224_diagnostic": roots["next224"] / n224.DIAGNOSTIC_NAME,
            "next224_table": roots["next224"] / n224.TABLE_NAME,
        }
    )
    return paths


def _verify_priors(
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
    pd.DataFrame,
]:
    prior_paths = dict(paths)
    prior_paths["design"] = paths["next224_design"]
    prior_hashes = dict(input_hashes)
    prior_hashes["design"] = input_hashes["next224_design"]
    (
        eligible,
        eligible214,
        primary_key,
        base_start_key,
        formula214,
        current_key,
        formula222,
        table222,
        selected223,
    ) = n224._verify_next223(prior_paths, prior_hashes)
    manifest = json.loads(paths["next224_manifest"].read_text())
    diagnostic = json.loads(paths["next224_diagnostic"].read_text())
    table = pd.read_parquet(paths["next224_table"])
    expected_outputs = {
        n224.DIAGNOSTIC_NAME: input_hashes["next224_diagnostic"],
        n224.TABLE_NAME: input_hashes["next224_table"],
    }
    if (
        manifest.get("protocol") != n224.PROTOCOL
        or manifest.get("candidate_count") != n224.EXPECTED_CANDIDATE_COUNT
        or manifest.get("candidate_key_sha256")
        != n224.EXPECTED_CANDIDATE_KEY_SHA256
        or manifest.get("strict_residual_improvement_observed") is not True
        or manifest.get("dual_evidence_branch_closed") is not False
        or manifest.get("new_formula_searched") is not False
        or manifest.get("new_formula_selected") is not False
        or manifest.get("outputs_sha256") != expected_outputs
        or any(
            manifest.get(key) is not value
            for key, value in BOUNDARY_FLAGS.items()
        )
        or manifest.get("executed_source_sha256", {}).get(
            "src/next224_dual_evidence_broad_diagnostic.py"
        )
        != EXPECTED_NEXT224_SOURCE_SHA256
        or _sha256_file(Path(n224.__file__).resolve())
        != EXPECTED_NEXT224_SOURCE_SHA256
        or diagnostic.get("protocol") != n224.PROTOCOL
        or diagnostic.get("candidate_count") != n224.EXPECTED_CANDIDATE_COUNT
        or diagnostic.get("candidate_key_sha256")
        != n224.EXPECTED_CANDIDATE_KEY_SHA256
        or diagnostic.get("improves_over_next222_global_residual") is not True
        or diagnostic.get("dual_evidence_branch_closed") is not False
        or diagnostic.get("new_formula_searched") is not False
        or len(table) != n224.EXPECTED_CANDIDATE_COUNT
        or n224.candidate_key_sha256(table)
        != n224.EXPECTED_CANDIDATE_KEY_SHA256
    ):
        raise ValueError("NEXT225 NEXT224 provenance differs")
    return (
        eligible,
        eligible214,
        primary_key,
        base_start_key,
        formula214,
        current_key,
        formula222,
        table222,
        selected223,
    )


def materialize_agreement_candidates(
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
    """Encode every exact NEXT225 score as one evaluator term."""

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
        raise ValueError("NEXT225 materializer inputs differ")
    source = features["source_dataset"].astype(str).to_numpy()
    raw_cache: dict[str, np.ndarray] = {}
    columns: dict[str, np.ndarray] = {}
    terms: list[dict[str, object]] = []
    runtime: list[dict[str, object]] = []
    activity: dict[str, dict[str, int]] = {}
    for spec in raw_specs:
        first_feature = spec.get("first_feature")
        second_feature = spec.get("second_feature")
        if first_feature is None and second_feature is None:
            score = np.maximum(0.0, base + current)
            got_support = support.copy()
            active = np.zeros(len(features), dtype=bool)
        else:
            if (
                first_feature is None
                or second_feature is None
                or str(first_feature) not in features.columns
                or str(second_feature) not in features.columns
            ):
                raise ValueError("NEXT225 materializer feature differs")
            for feature in (str(first_feature), str(second_feature)):
                if feature not in raw_cache:
                    raw_cache[feature] = pd.to_numeric(
                        features[feature], errors="coerce"
                    ).to_numpy(float)
            score, got_support, active, _ = agreement_gated_consensus_score(
                base_score=base,
                current_delta=current,
                base_support=support,
                first_values=raw_cache[str(first_feature)],
                first_direction=str(spec["first_direction"]),
                first_q_lo=float(spec["first_q_lo"]),
                first_q_hi=float(spec["first_q_hi"]),
                second_values=raw_cache[str(second_feature)],
                second_direction=str(spec["second_direction"]),
                second_q_lo=float(spec["second_q_lo"]),
                second_q_hi=float(spec["second_q_hi"]),
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
            "next225_virtual_candidate__"
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
                "group": "next225_agreement_gated_consensus",
                "encoding": "asinh_sinh_exact_agreement_gated_score",
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


def _assert_record_reproduction(
    observed: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    n223._assert_record_reproduction(observed, expected)


def run_agreement_gated_consensus_search(
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
    """Run the frozen discovery-only NEXT225 search."""

    if not set(REQUIRED_STAGES).issubset(stage_dirs):
        raise ValueError("NEXT225 stage directory universe differs")
    if not set(REQUIRED_DESIGN_STAGES).issubset(design_paths):
        raise ValueError("NEXT225 design path universe differs")
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
        raise ValueError("NEXT225 search_workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT225 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT225 formal input identity differs: {differing}")
    (
        eligible,
        eligible214,
        primary_key,
        base_start_key,
        formula214,
        current_key,
        formula222,
        table222,
        _,
    ) = _verify_priors(paths, input_hashes)
    combined, feature_tables, base_score, support, endpoint = (
        n222.n215._reconstruct_next214_final(
            paths=paths,
            eligible=eligible214,
            primary_key=primary_key,
            start_key=base_start_key,
            formula=formula214,
        )
    )
    current_delta = n223._reconstruct_next222_delta(
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
        raise ValueError("NEXT225 NEXT214 base identity differs")
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
        raise ValueError("NEXT225 eligible normalization universe differs")
    specs = build_agreement_candidate_specs(
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
        raise RuntimeError("NEXT225 frozen candidate universe differs")
    virtual, terms, runtime, activity = materialize_agreement_candidates(
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
        raise RuntimeError("NEXT225 evaluator accounting differs")
    spec_by_key = {str(spec["candidate_key"]): spec for spec in specs}
    raw_records = [
        *eligible_result["candidate_records"],
        *fixed_result["candidate_records"],
    ]
    for record in raw_records:
        spec = spec_by_key[str(record["candidate_key"])]
        record.update(
            {
                "first_hypothesis": spec["first_hypothesis"],
                "first_feature": spec["first_feature"],
                "first_direction": spec["first_direction"],
                "first_q_lo": spec["first_q_lo"],
                "first_q_hi": spec["first_q_hi"],
                "second_hypothesis": spec["second_hypothesis"],
                "second_feature": spec["second_feature"],
                "second_direction": spec["second_direction"],
                "second_q_lo": spec["second_q_lo"],
                "second_q_hi": spec["second_q_hi"],
                "beta_fraction": spec["beta_fraction"],
                "protection_budget_fraction": spec[
                    "protection_budget_fraction"
                ],
                "is_reproduction_control": spec["is_reproduction_control"],
                "eligible_new_candidate": spec["eligible_new_candidate"],
                "pair_active_rows": activity[str(record["candidate_key"])][
                    "rows"
                ],
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
        raise RuntimeError("NEXT225 no-op reproduction identity differs")
    _assert_record_reproduction(no_op.iloc[0], reference_no_op.iloc[0])

    used = {str(term["hypothesis"]) for term in formula222["terms"]}
    table223 = pd.read_parquet(paths["next223_search"])
    reproduced_next222 = 0
    reproduced_next223 = 0
    closed_form_controls = 0
    for spec in specs:
        if not spec["is_reproduction_control"]:
            continue
        first_hypothesis = str(spec["first_hypothesis"])
        second_hypothesis = str(spec["second_hypothesis"])
        observed = records.loc[
            records["candidate_key"].eq(spec["candidate_key"])
        ]
        if len(observed) != 1:
            raise RuntimeError("NEXT225 control identity differs")
        if first_hypothesis != second_hypothesis:
            expected = table223.loc[
                table223["protection_hypothesis"].eq(first_hypothesis)
                & table223["risk_hypothesis"].eq(second_hypothesis)
                & table223["beta_fraction"].eq(spec["beta_fraction"])
                & table223["protection_budget_fraction"].eq(0.5)
            ]
            if len(expected) != 1:
                raise RuntimeError("NEXT225 NEXT223 control identity differs")
            _assert_record_reproduction(observed.iloc[0], expected.iloc[0])
            reproduced_next223 += 1
        elif first_hypothesis not in used:
            expected = table222.loc[
                table222["depth"].eq(3)
                & table222["proposed_hypothesis"].eq(first_hypothesis)
                & table222["proposed_beta_fraction"].eq(spec["beta_fraction"])
            ]
            if len(expected) != 1:
                raise RuntimeError("NEXT225 NEXT222 control identity differs")
            _assert_record_reproduction(observed.iloc[0], expected.iloc[0])
            reproduced_next222 += 1
        else:
            closed_form_controls += 1
    if (
        reproduced_next223 != EXPECTED_NEXT223_OFF_DIAGONAL_CONTROL_COUNT
        or reproduced_next222 != EXPECTED_NEXT222_DIAGONAL_CONTROL_COUNT
        or closed_form_controls != EXPECTED_CLOSED_FORM_CONTROL_COUNT
    ):
        raise RuntimeError("NEXT225 reproduction control count differs")

    selected_row = n223.select_best_eligible_record(records)
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
        raise RuntimeError("NEXT225 selected candidate reproduction differs")
    selected = selected_result["selected"]
    selected_spec = spec_by_key[selected_key]
    for name, value in selected_row.items():
        if name in selected["record"]:
            selected["record"][name] = value
    eligible_frame = records.loc[records["eligible_new_candidate"].astype(bool)]
    controls_frame = records.loc[records["is_reproduction_control"].astype(bool)]
    passes = bool(
        eligible_frame["passes_all_discovery_gates"]
        .fillna(False)
        .astype(bool)
        .any()
    )
    if passes != bool(selected["record"]["passes_all_discovery_gates"]):
        raise RuntimeError("NEXT225 all-gate selection differs")
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
        "kind": "agreement_gated_consensus_x0_no_dft_score",
        "base_protocol": n222.PROTOCOL,
        "base_candidate_key_sha256": n223.EXPECTED_NEXT222_FINAL_KEY_SHA256,
        "base_terms": formula222["terms"],
        "first_hypothesis": selected_spec["first_hypothesis"],
        "first_feature": selected_spec["first_feature"],
        "first_direction": selected_spec["first_direction"],
        "first_q_lo": selected_spec["first_q_lo"],
        "first_q_hi": selected_spec["first_q_hi"],
        "second_hypothesis": selected_spec["second_hypothesis"],
        "second_feature": selected_spec["second_feature"],
        "second_direction": selected_spec["second_direction"],
        "second_q_lo": selected_spec["second_q_lo"],
        "second_q_hi": selected_spec["second_q_hi"],
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
        "base_candidate_key_sha256": n223.EXPECTED_NEXT222_FINAL_KEY_SHA256,
        "eligible_hypotheses": list(eligible),
        "eligible_hypothesis_count": len(eligible),
        "eligible_hypothesis_sha256": (
            n222.n220.EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256
        ),
        "unordered_pair_count": EXPECTED_PAIR_COUNT,
        "beta_fractions": list(BETA_FRACTIONS),
        "protection_budget_fractions": list(PROTECTION_BUDGET_FRACTIONS),
        "candidate_count": len(records),
        "eligible_new_candidate_count": len(eligible_frame),
        "equal_budget_control_count": len(controls_frame),
        "exact_next223_off_diagonal_control_count": reproduced_next223,
        "exact_next222_depth3_reproduction_control_count": reproduced_next222,
        "closed_form_control_count": closed_form_controls,
        "normalization_refit": False,
        "normalization_fit_uses_endpoint": False,
        "base_support_unchanged": True,
        "unordered_symmetric_pairs": True,
        "next224_winner_promoted": False,
        "validation_outputs_opened": False,
        "dft_values_used_by_executable_formula": False,
    }
    evaluation = {
        "protocol": PROTOCOL,
        "evaluation_mode": "frozen_agreement_gated_consensus_discovery_search",
        "next222_final_path_reproduced": True,
        "next224_winner_promoted": False,
        "next223_off_diagonal_control_count_reproduced": reproduced_next223,
        "next222_depth3_control_count_reproduced": reproduced_next222,
        "closed_form_control_count": closed_form_controls,
        "rows": {
            "scigen": int(len(feature_tables["scigen"])),
            "wyformer": int(len(feature_tables["wyformer"])),
            "total": int(len(combined)),
        },
        "candidate_count": len(records),
        "eligible_new_candidate_count": len(eligible_frame),
        "equal_budget_control_count": len(controls_frame),
        "elapsed_seconds": elapsed,
        "search_workers": search_workers,
        "counts_all": n223._gate_counts(records),
        "counts_eligible_new": n223._gate_counts(eligible_frame),
        "counts_equal_budget_controls": n223._gate_counts(controls_frame),
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
        "next226_diagnostic_authorized": bool(not passes and diagnostic_keys),
        "next226_candidate_count": len(diagnostic_keys),
        "next226_candidate_key_sha256": diagnostic_sha,
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
        "src/next223_dual_evidence_consensus_search.py": Path(
            n223.__file__
        ).resolve(),
        "src/next224_dual_evidence_broad_diagnostic.py": Path(
            n224.__file__
        ).resolve(),
        "src/next225_agreement_gated_consensus_search.py": Path(
            __file__
        ).resolve(),
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
            "equal_budget_control_count": len(controls_frame),
            "exact_next223_off_diagonal_control_count": reproduced_next223,
            "exact_next222_depth3_reproduction_control_count": reproduced_next222,
            "closed_form_control_count": closed_form_controls,
            "eligible_hypothesis_count": len(eligible),
            "eligible_hypothesis_sha256": (
                n222.n220.EXPECTED_ELIGIBLE_HYPOTHESIS_SHA256
            ),
            "passes_all_cross_source_discovery_gates": passes,
            "freeze_authorized": passes,
            "next226_diagnostic_authorized": bool(
                not passes and diagnostic_keys
            ),
            "next226_candidate_count": len(diagnostic_keys),
            "next226_candidate_key_sha256": diagnostic_sha,
            "requires_unopened_internal_validation_before_claim": True,
            "next224_winner_promoted": False,
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
            raise RuntimeError("NEXT225 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT225 source changed before publication")
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
    manifest = run_agreement_gated_consensus_search(
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
    "agreement_gated_consensus_score",
    "build_agreement_candidate_specs",
    "materialize_agreement_candidates",
    "run_agreement_gated_consensus_search",
]


if __name__ == "__main__":
    raise SystemExit(main())
