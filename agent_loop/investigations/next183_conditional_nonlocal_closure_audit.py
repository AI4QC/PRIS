#!/usr/bin/env python3
"""Frozen audit of nonlocal-cleanliness conditioned directional closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import src.next130_coordination_protection_search as n130
import src.next135_conjunctive_compactness_search as n135
import src.next151_violation_multiplicity_audit as n151
import src.next163_interior_family_attenuation_search as n163
import src.next164_interior_attenuation_broad_residual as n164
import src.next169_periodic_local_directional_rigidity_audit as n169
import src.next179_strong_neighborhood_directional_closure as n179
import src.next180_strong_neighborhood_directional_closure_audit as n180
import src.next182_local_family_closure_attenuation_search as n182
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json
from src.next87_scigen_sparse_law_search import _term_risk


PROTOCOL = "2026-08-08-next183-conditional-nonlocal-closure-audit-v1"
MANIFEST_NAME = "MANIFEST.json"
AUDIT_NAME = "NEXT183_CONDITIONAL_NONLOCAL_CLOSURE_AUDIT.json"
TABLE_NAME = "next183_conditional_nonlocal_closure_audit.parquet"
EXPECTED_DESIGN_SHA256 = (
    "989cbf9d9f6f10d1337d3c424808e2112eb89510847241462720482e69141334"
)
EXPECTED_BASE_CANDIDATE_KEY_SHA256 = n182.EXPECTED_BASE_CANDIDATE_KEY_SHA256
SAFE_THRESHOLD = n182.SAFE_THRESHOLD
BROAD_THRESHOLD = n182.BROAD_THRESHOLD
MIN_FULL_SUPPORT = n180.MIN_FULL_SUPPORT
NONLOCAL_FAMILIES = (
    "charge_flow_feasibility",
    "contact_robustness",
    "valence_transport",
)
CLEANLINESS_NAMES = ("clean_max", "clean_mean", "clean_product")
CONJUNCTIONS = ("product", "minimum")
CLOSURE_FEATURES = (
    "psndc_crystalnn_closure_mean",
    "psndc_crystalnn_closure_min",
    "psndc_crystalnn_closure_q10",
    "psndc_crystalnn_volume_mean",
    "psndc_crystalnn_volume_q10",
    "psndc_voronoi_closure_min",
)
if set(CLOSURE_FEATURES) != set(n182.ELIGIBLE_FEATURES):
    raise RuntimeError("NEXT183 frozen closure universe differs from NEXT180 audit")
HYPOTHESES = {
    f"{feature}__{cleanliness}__{conjunction}__high": (
        feature,
        cleanliness,
        conjunction,
        1,
    )
    for feature in CLOSURE_FEATURES
    for cleanliness in CLEANLINESS_NAMES
    for conjunction in CONJUNCTIONS
}
EXPECTED_INPUT_SHA256 = {
    **n182.EXPECTED_INPUT_SHA256,
    "design": EXPECTED_DESIGN_SHA256,
    "next182_manifest": "31e5db748cc56fa17f9c1ef2aaf38fb78d0653fd35e3ce158f87e7f9a0a0c2e1",
    "next182_catalogue": "aa8ecde73728cf2feb8e188eaf13b2d753b49bb196acf538b217a5e109a17121",
    "next182_evaluation": "2ce081e372afece305994d4da065b7bf3d9fa7891f4db4026d3ff7b6a5c41029",
    "next182_formula": "883e4054693f92d010682b412a07854d2377500c08d6f3e7687c1f062ec26f95",
    "next182_search": "b473677fa302d1e0c4657dd2c1b908c3c76a969084267577ddae6d99c25fc724",
}


def reconstruct_family_means(
    *,
    features: pd.DataFrame,
    physical_terms: Sequence[Mapping[str, object]],
    base_spec: Mapping[str, object],
    base_support: object,
) -> dict[str, np.ndarray]:
    """Reconstruct the exact capped NEXT163 mean for every mechanism family."""

    support = np.asarray(base_support, dtype=bool)
    if support.shape != (len(features),):
        raise ValueError("NEXT183 base support shape differs")
    physical_by_id = {str(term.get("term_id")): term for term in physical_terms}
    if len(physical_by_id) != len(physical_terms):
        raise ValueError("NEXT183 physical term IDs differ")
    term_ids = [str(value) for value in base_spec.get("base_term_ids", ())]
    try:
        weights = [float(value) for value in base_spec.get("base_weights", ())]
    except (TypeError, ValueError) as exc:
        raise ValueError("NEXT183 base weights differ") from exc
    if (
        not term_ids
        or len(term_ids) != len(weights)
        or len(set(term_ids)) != len(term_ids)
        or any(term_id not in physical_by_id for term_id in term_ids)
        or any(not np.isfinite(weight) or weight < 0.0 for weight in weights)
    ):
        raise ValueError("NEXT183 base term schema differs")

    members: dict[str, list[int]] = {family: [] for family in n163.FAMILY_PREFIXES}
    contributions: list[np.ndarray] = []
    term_supports: list[np.ndarray] = []
    for index, (term_id, weight) in enumerate(zip(term_ids, weights, strict=True)):
        matches = [
            family
            for family, prefixes in n163.FAMILY_PREFIXES.items()
            if term_id.startswith(prefixes)
        ]
        if len(matches) != 1:
            raise ValueError("NEXT183 term-to-family assignment differs")
        members[matches[0]].append(index)
        risk, supported = _term_risk(features, physical_by_id[term_id])
        contributions.append(weight * risk)
        term_supports.append(supported)
    if any(not indices for indices in members.values()):
        raise ValueError("NEXT183 family coverage differs")
    support_matrix = np.column_stack(term_supports)
    if not np.array_equal(support_matrix.all(axis=1), support):
        raise RuntimeError("NEXT183 physical/base support differs")
    values = np.column_stack(contributions)
    if np.any(~np.isfinite(values[support])) or np.any(values[support] < -1.0e-12):
        raise ValueError("NEXT183 physical contribution differs")
    capped = np.minimum(np.maximum(values, 0.0), n163.CONTRIBUTION_CAP)
    result: dict[str, np.ndarray] = {}
    for family, indices in members.items():
        family_mean = np.full(len(features), np.nan)
        family_mean[support] = capped[support][:, indices].mean(axis=1)
        result[family] = family_mean
    return result


def compute_nonlocal_cleanliness(
    *, family_means: Mapping[str, object], base_support: object
) -> dict[str, np.ndarray]:
    """Calculate the three frozen parameter-free nonlocal cleanliness scores."""

    if set(family_means) != set(n163.FAMILY_PREFIXES):
        raise ValueError("NEXT183 family mean schema differs")
    support = np.asarray(base_support, dtype=bool)
    if support.ndim != 1:
        raise ValueError("NEXT183 base support differs")
    arrays = {name: np.asarray(value, dtype=float) for name, value in family_means.items()}
    if any(value.shape != support.shape for value in arrays.values()):
        raise ValueError("NEXT183 family/support shape differs")
    nonlocal_values = np.column_stack([arrays[name] for name in NONLOCAL_FAMILIES])
    selected = nonlocal_values[support]
    if (
        np.any(~np.isfinite(selected))
        or np.any(selected < -1.0e-12)
        or np.any(selected > n163.CONTRIBUTION_CAP + 1.0e-12)
    ):
        raise ValueError("NEXT183 family mean is outside the frozen cap")
    normalized = np.clip(
        nonlocal_values / n163.CONTRIBUTION_CAP,
        0.0,
        1.0,
    )
    outputs = {
        "clean_max": 1.0 - np.max(normalized, axis=1),
        "clean_mean": 1.0 - np.mean(normalized, axis=1),
        "clean_product": np.prod(1.0 - normalized, axis=1),
    }
    for values in outputs.values():
        values[~support] = np.nan
    return outputs


def conditional_certificate(
    *, closure: object, cleanliness: object, conjunction: str
) -> np.ndarray:
    """Combine closure and cleanliness with one frozen monotone conjunction."""

    closure_values = np.asarray(closure, dtype=float)
    clean_values = np.asarray(cleanliness, dtype=float)
    if (
        closure_values.ndim != 1
        or clean_values.shape != closure_values.shape
        or conjunction not in CONJUNCTIONS
    ):
        raise ValueError("NEXT183 conditional certificate schema differs")
    for values in (closure_values, clean_values):
        finite = np.isfinite(values)
        if np.any((values[finite] < -1.0e-12) | (values[finite] > 1.0 + 1.0e-12)):
            raise ValueError("NEXT183 conditional certificate input is outside [0,1]")
    valid = np.isfinite(closure_values) & np.isfinite(clean_values)
    result = np.full(len(closure_values), np.nan)
    bounded_closure = np.clip(closure_values[valid], 0.0, 1.0)
    bounded_clean = np.clip(clean_values[valid], 0.0, 1.0)
    if conjunction == "product":
        result[valid] = bounded_closure * bounded_clean
    else:
        result[valid] = np.minimum(bounded_closure, bounded_clean)
    return result


def eligibility_from_metrics(
    *,
    scigen_full_support: float,
    wyformer_full_support: float,
    scigen_shell_worst_auc: float,
    scigen_shell_evaluable_folds: int,
    wyformer_shell_pooled_auc: float,
    scigen_full_pooled_auc: float,
    wyformer_full_pooled_auc: float,
) -> bool:
    """Apply the unchanged frozen NEXT169/NEXT180 audit gates."""

    return n169.eligibility_from_metrics(
        scigen_full_support=scigen_full_support,
        wyformer_full_support=wyformer_full_support,
        scigen_shell_worst_auc=scigen_shell_worst_auc,
        scigen_shell_evaluable_folds=scigen_shell_evaluable_folds,
        wyformer_shell_pooled_auc=wyformer_shell_pooled_auc,
        scigen_full_pooled_auc=scigen_full_pooled_auc,
        wyformer_full_pooled_auc=wyformer_full_pooled_auc,
    )


def select_conditional_closure_hypothesis(
    records: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    """Rank eligible certificates with deterministic tie breaking."""

    required = {
        "hypothesis",
        "eligible_for_search",
        "ranking_min_auc",
        "ranking_mean_auc",
    }
    if required - set(records.columns) or records["hypothesis"].astype(str).duplicated().any():
        raise ValueError("NEXT183 audit record schema differs")
    table = records.sort_values(
        ["eligible_for_search", "ranking_min_auc", "ranking_mean_auc", "hypothesis"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    eligible = table.loc[table["eligible_for_search"].fillna(False).astype(bool)]
    selected = None if eligible.empty else eligible.iloc[0].to_dict()
    return table, selected


def _paths(
    roots: Mapping[str, Path], freeze_path: Path, design_path: Path
) -> dict[str, Path]:
    paths = n182._paths(roots, freeze_path, design_path)
    paths.update(
        {
            "next182_manifest": roots["next182"] / n182.MANIFEST_NAME,
            "next182_catalogue": roots["next182"] / n182.CATALOGUE_NAME,
            "next182_evaluation": roots["next182"] / n182.EVALUATION_NAME,
            "next182_formula": roots["next182"] / n182.FORMULA_NAME,
            "next182_search": roots["next182"] / n182.SEARCH_NAME,
        }
    )
    return paths


def run_conditional_nonlocal_closure_audit(
    *,
    scigen_feature_dir: Path,
    scigen_discovery_endpoint_dir: Path,
    wyformer_feature_dir: Path,
    wyformer_discovery_endpoint_dir: Path,
    next98_dir: Path,
    next110_dir: Path,
    next111_dir: Path,
    next113_dir: Path,
    next114_dir: Path,
    next116_dir: Path,
    next117_dir: Path,
    next120_dir: Path,
    next121_dir: Path,
    next122_dir: Path,
    next124_dir: Path,
    next125_dir: Path,
    next129_dir: Path,
    next130_dir: Path,
    next133_dir: Path,
    next134_dir: Path,
    next163_dir: Path,
    next164_dir: Path,
    next168_dir: Path,
    next173_dir: Path,
    next179_dir: Path,
    next180_dir: Path,
    next181_dir: Path,
    next182_dir: Path,
    next135_freeze_path: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only NEXT183 audit and publish atomically."""

    roots = {
        "scigen_features": Path(scigen_feature_dir).resolve(),
        "scigen_endpoint": Path(scigen_discovery_endpoint_dir).resolve(),
        "wyformer_features": Path(wyformer_feature_dir).resolve(),
        "wyformer_endpoint": Path(wyformer_discovery_endpoint_dir).resolve(),
        **{
            f"next{stage}": Path(value).resolve()
            for stage, value in (
                (98, next98_dir),
                (110, next110_dir),
                (111, next111_dir),
                (113, next113_dir),
                (114, next114_dir),
                (116, next116_dir),
                (117, next117_dir),
                (120, next120_dir),
                (121, next121_dir),
                (122, next122_dir),
                (124, next124_dir),
                (125, next125_dir),
                (129, next129_dir),
                (130, next130_dir),
                (133, next133_dir),
                (134, next134_dir),
                (163, next163_dir),
                (164, next164_dir),
                (168, next168_dir),
                (173, next173_dir),
                (179, next179_dir),
                (180, next180_dir),
                (181, next181_dir),
                (182, next182_dir),
            )
        },
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots, Path(next135_freeze_path).resolve(), Path(design_path).resolve()
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT183 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT183 formal input identity differs: {differing}")

    manifest182 = json.loads(paths["next182_manifest"].read_text())
    if (
        manifest182.get("protocol") != n182.PROTOCOL
        or manifest182.get("candidate_count") != n182.EXPECTED_CANDIDATE_COUNT
        or manifest182.get("base_endpoint_reproduced") is not True
        or manifest182.get("passes_all_cross_source_discovery_gates") is not False
        or manifest182.get("freeze_authorized") is not False
        or manifest182.get("local_family_closure_attenuation_branch_terminated") is not True
        or manifest182.get("opened_validation_outputs_used") is not False
        or manifest182.get("scigen_replication_endpoint_opened") is not False
        or manifest182.get("wyformer_replication_endpoint_opened") is not False
        or manifest182.get("dft_calculation_executed") is not False
        or manifest182.get("dft_values_used_by_executable_formula") is not False
        or manifest182.get("learned_energy_force_stress_proxy_used") is not False
        or manifest182.get("physical_relaxation_executed") is not False
        or manifest182.get("outputs_sha256", {}).get(n182.CATALOGUE_NAME)
        != input_hashes["next182_catalogue"]
        or manifest182.get("outputs_sha256", {}).get(n182.EVALUATION_NAME)
        != input_hashes["next182_evaluation"]
        or manifest182.get("outputs_sha256", {}).get(n182.FORMULA_NAME)
        != input_hashes["next182_formula"]
        or manifest182.get("outputs_sha256", {}).get(n182.SEARCH_NAME)
        != input_hashes["next182_search"]
        or manifest182.get("executed_source_sha256", {}).get(
            "src/next182_local_family_closure_attenuation_search.py"
        )
        != _sha256_file(Path(n182.__file__).resolve())
    ):
        raise ValueError("NEXT183 NEXT182 provenance differs")

    diagnostic164 = json.loads(paths["next164_diagnostic"].read_text())
    base_key = str(diagnostic164.get("global_closest", {}).get("candidate_key", ""))
    if hashlib.sha256(base_key.encode()).hexdigest() != EXPECTED_BASE_CANDIDATE_KEY_SHA256:
        raise ValueError("NEXT183 base candidate identity differs")

    extended, _, old_terms, mhcr_terms = n130._join_label_free_features(paths)
    compact_frames = []
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next133_{source}_features"]).copy()
        table["material_id"] = source + ":" + table["material_id"].astype(str)
        compact_frames.append(table)
    extended = extended.merge(
        pd.concat(compact_frames, ignore_index=True),
        on="material_id",
        how="inner",
        validate="one_to_one",
    )
    extended = pd.concat(
        [
            extended.reset_index(drop=True),
            n135.materialize_conjunctive_features(extended).reset_index(drop=True),
        ],
        axis=1,
    )
    closure_frames = []
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next179_{source}_features"]).copy()
        table["material_id"] = source + ":" + table["material_id"].astype(str)
        closure_frames.append(table)
    combined = extended.merge(
        pd.concat(closure_frames, ignore_index=True),
        on="material_id",
        how="inner",
        validate="one_to_one",
    )
    if set(CLOSURE_FEATURES) - set(combined.columns):
        raise ValueError("NEXT183 frozen closure feature schema differs")

    physical_terms = [*old_terms, *mhcr_terms]
    physical_ids = {str(term["term_id"]) for term in physical_terms}
    all_bases = n130.n127.select_next125_bases(
        pd.read_parquet(paths["next125_search_records"])
    )
    bases = n135.n132.select_extended_bases(
        pd.read_parquet(paths["next130_search_records"]), all_bases
    )
    base_specs = n163.build_candidate_specs(
        bases=bases, physical_term_ids=physical_ids
    )
    selected_specs = [
        spec for spec in base_specs if str(spec["candidate_key"]) == base_key
    ]
    if len(selected_specs) != 1:
        raise ValueError("NEXT183 base reconstruction differs")
    combined, base_terms, runtime = n163.materialize_candidates(
        features=combined,
        physical_terms=physical_terms,
        specs=selected_specs,
    )
    if len(base_terms) != 1 or len(runtime) != 1:
        raise RuntimeError("NEXT183 base score materialization differs")
    base_score, base_support = _term_risk(combined, base_terms[0])
    family_means = reconstruct_family_means(
        features=combined,
        physical_terms=physical_terms,
        base_spec=selected_specs[0],
        base_support=base_support,
    )
    cleanliness = compute_nonlocal_cleanliness(
        family_means=family_means, base_support=base_support
    )

    scigen_endpoint = pd.read_parquet(paths["scigen_endpoint"])
    wyformer_endpoint = pd.read_parquet(paths["wyformer_endpoint"])
    endpoint_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "material_id": "scigen:"
                    + scigen_endpoint["material_id"].astype(str),
                    "_endpoint": pd.to_numeric(
                        scigen_endpoint["distortion_ratio"], errors="coerce"
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "material_id": "wyformer:"
                    + wyformer_endpoint["material_id"].astype(str),
                    "_endpoint": n130.n125.n121.prior._endpoint_numeric(
                        wyformer_endpoint["endpoint_stratum"]
                    ),
                }
            ),
        ],
        ignore_index=True,
    )
    combined = combined.merge(
        endpoint_frame, on="material_id", how="inner", validate="one_to_one"
    )
    endpoint = pd.to_numeric(combined.pop("_endpoint"), errors="coerce").to_numpy(
        float
    )
    if len(combined) != len(base_score) or not np.isfinite(endpoint).all():
        raise ValueError("NEXT183 endpoint row accounting differs")

    sources = combined["source_dataset"].astype(str).to_numpy()
    folds = n164.assign_group_folds(combined["reduced_formula"].astype(str).to_numpy())
    extremes = base_support & ((endpoint <= 1.0) | (endpoint >= 2.0))
    shell = (
        extremes
        & (base_score >= BROAD_THRESHOLD)
        & (base_score < SAFE_THRESHOLD)
    )
    populations = {
        "scigen_shell": shell & (sources == "scigen"),
        "wyformer_shell": shell & (sources == "wyformer"),
        "scigen_full": extremes & (sources == "scigen"),
        "wyformer_full": extremes & (sources == "wyformer"),
    }
    population_counts = {
        name: {
            "rows": int(mask.sum()),
            "protected": int((mask & (endpoint <= 1.0)).sum()),
            "severe": int((mask & (endpoint >= 2.0)).sum()),
        }
        for name, mask in populations.items()
    }

    records: list[dict[str, object]] = []
    for hypothesis in sorted(HYPOTHESES):
        feature, clean_name, conjunction, direction = HYPOTHESES[hypothesis]
        values = conditional_certificate(
            closure=pd.to_numeric(combined[feature], errors="coerce").to_numpy(float),
            cleanliness=cleanliness[clean_name],
            conjunction=conjunction,
        )
        evaluations = {}
        for population, mask in populations.items():
            evaluation = n151._evaluate_auc(
                values=values[mask],
                protected=endpoint[mask] <= 1.0,
                folds=folds[mask],
                direction=direction,
            )
            if evaluation is None:
                raise ValueError("NEXT183 AUC population differs")
            evaluations[population] = evaluation
        support = {
            source: float(np.isfinite(values[sources == source]).mean())
            for source in ("scigen", "wyformer")
        }
        scigen_worst = evaluations["scigen_shell"]["worst_auc"]
        key_aucs = [
            0.0 if scigen_worst is None else float(scigen_worst),
            float(evaluations["wyformer_shell"]["pooled_auc"]),
            float(evaluations["scigen_full"]["pooled_auc"]),
            float(evaluations["wyformer_full"]["pooled_auc"]),
        ]
        eligible = eligibility_from_metrics(
            scigen_full_support=support["scigen"],
            wyformer_full_support=support["wyformer"],
            scigen_shell_worst_auc=key_aucs[0],
            scigen_shell_evaluable_folds=int(
                evaluations["scigen_shell"]["evaluable_folds"]
            ),
            wyformer_shell_pooled_auc=key_aucs[1],
            scigen_full_pooled_auc=key_aucs[2],
            wyformer_full_pooled_auc=key_aucs[3],
        )
        record: dict[str, object] = {
            "hypothesis": hypothesis,
            "closure_feature": feature,
            "cleanliness": clean_name,
            "conjunction": conjunction,
            "direction": direction,
            "eligible_for_search": eligible,
            "scigen_full_support": support["scigen"],
            "wyformer_full_support": support["wyformer"],
            "ranking_min_auc": float(min(key_aucs)),
            "ranking_mean_auc": float(np.mean(key_aucs)),
        }
        for population, evaluation in evaluations.items():
            for metric in (
                "pooled_auc",
                "macro_auc",
                "worst_auc",
                "evaluable_folds",
                "protected",
                "severe",
            ):
                record[f"{population}_{metric}"] = evaluation[metric]
            record[f"{population}_fold_aucs_json"] = json.dumps(
                evaluation["fold_aucs"], separators=(",", ":")
            )
        records.append(record)

    audit_table, selected = select_conditional_closure_hypothesis(
        pd.DataFrame(records)
    )
    eligible_names = audit_table.loc[
        audit_table["eligible_for_search"].fillna(False).astype(bool), "hypothesis"
    ].astype(str).tolist()
    audit = {
        "protocol": PROTOCOL,
        "design_sha256": input_hashes["design"],
        "candidate_key_sha256": EXPECTED_BASE_CANDIDATE_KEY_SHA256,
        "safe_threshold": SAFE_THRESHOLD,
        "broad_threshold": BROAD_THRESHOLD,
        "contribution_cap": n163.CONTRIBUTION_CAP,
        "nonlocal_families": list(NONLOCAL_FAMILIES),
        "cleanliness_definitions": {
            "clean_max": "1-max(nonlocal_family_means)/0.5",
            "clean_mean": "1-mean(nonlocal_family_means)/0.5",
            "clean_product": "product(1-nonlocal_family_mean/0.5)",
        },
        "conjunction_definitions": {
            "product": "closure*cleanliness",
            "minimum": "min(closure,cleanliness)",
        },
        "population_counts": population_counts,
        "hypotheses": {
            name: {
                "closure_feature": value[0],
                "cleanliness": value[1],
                "conjunction": value[2],
                "direction": value[3],
            }
            for name, value in HYPOTHESES.items()
        },
        "eligibility_gates": {
            "minimum_full_support": MIN_FULL_SUPPORT,
            "scigen_shell_worst_auc": 0.55,
            "wyformer_shell_pooled_auc": 0.55,
            "full_source_pooled_auc": 0.50,
            "required_scigen_shell_folds": 5,
        },
        "eligible_hypotheses": eligible_names,
        "selected_hypothesis": selected,
        "new_formula_searched": False,
        "validation_or_replication_opened": False,
        "dft_calculation_executed": False,
        "dft_values_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False,
        "physical_relaxation_executed": False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    source_paths = {
        "src/next163_interior_family_attenuation_search.py": Path(n163.__file__).resolve(),
        "src/next179_strong_neighborhood_directional_closure.py": Path(n179.__file__).resolve(),
        "src/next180_strong_neighborhood_directional_closure_audit.py": Path(n180.__file__).resolve(),
        "src/next182_local_family_closure_attenuation_search.py": Path(n182.__file__).resolve(),
        "src/next183_conditional_nonlocal_closure_audit.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    try:
        audit_path = staging / AUDIT_NAME
        table_path = staging / TABLE_NAME
        _write_json(audit_path, audit)
        audit_table.to_parquet(table_path, index=False)
        manifest = {
            "protocol": PROTOCOL,
            "hypothesis_count": len(HYPOTHESES),
            "eligible_hypothesis_count": len(eligible_names),
            "conditional_nonlocal_closure_branch_terminated": selected is None,
            "new_formula_searched": False,
            "discovery_outcomes_used_as_offline_labels": True,
            "opened_validation_outputs_used": False,
            "scigen_replication_endpoint_opened": False,
            "wyformer_replication_endpoint_opened": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_executable_formula": False,
            "learned_energy_force_stress_proxy_used": False,
            "physical_relaxation_executed": False,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {
                AUDIT_NAME: _sha256_file(audit_path),
                TABLE_NAME: _sha256_file(table_path),
            },
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(
            _sha256_file(path) != input_hashes[name]
            for name, path in paths.items()
        ):
            raise RuntimeError("NEXT183 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT183 source changed before publication")
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
        98,
        110,
        111,
        113,
        114,
        116,
        117,
        120,
        121,
        122,
        124,
        125,
        129,
        130,
        133,
        134,
        163,
        164,
        168,
        173,
        179,
        180,
        181,
        182,
    )
    for stage in stages:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_conditional_nonlocal_closure_audit(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{f"next{stage}_dir": getattr(args, f"next{stage}_dir") for stage in stages},
        next135_freeze_path=args.next135_freeze_path,
        design_path=args.design_path,
        output_dir=args.output_dir,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "CLEANLINESS_NAMES",
    "CLOSURE_FEATURES",
    "CONJUNCTIONS",
    "HYPOTHESES",
    "compute_nonlocal_cleanliness",
    "conditional_certificate",
    "eligibility_from_metrics",
    "reconstruct_family_means",
    "run_conditional_nonlocal_closure_audit",
    "select_conditional_closure_hypothesis",
]


if __name__ == "__main__":
    raise SystemExit(main())
