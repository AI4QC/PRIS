#!/usr/bin/env python3
"""Frozen audit of local/nonlocal contradiction-relief hypotheses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

import numpy as np
import pandas as pd

import src.next130_coordination_protection_search as n130
import src.next135_conjunctive_compactness_search as n135
import src.next151_violation_multiplicity_audit as n151
import src.next163_interior_family_attenuation_search as n163
import src.next164_interior_attenuation_broad_residual as n164
import src.next169_periodic_local_directional_rigidity_audit as n169
import src.next179_strong_neighborhood_directional_closure as n179
import src.next183_conditional_nonlocal_closure_audit as n183
import src.next185_conditional_closure_broad_residual_diagnostic as n185
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json
from src.next87_scigen_sparse_law_search import _term_risk


PROTOCOL = "2026-08-08-next186-local-nonlocal-contradiction-relief-audit-v1"
MANIFEST_NAME = "MANIFEST.json"
AUDIT_NAME = "NEXT186_LOCAL_NONLOCAL_CONTRADICTION_RELIEF_AUDIT.json"
TABLE_NAME = "next186_local_nonlocal_contradiction_relief_audit.parquet"
EXPECTED_DESIGN_SHA256 = (
    "5049f3167cd8fcff5971e90109a0f8e3768c34ab1aa0fa9812e80ae18ba7c9f6"
)
EXPECTED_BASE_CANDIDATE_KEY_SHA256 = n183.EXPECTED_BASE_CANDIDATE_KEY_SHA256
SAFE_THRESHOLD = n183.SAFE_THRESHOLD
BROAD_THRESHOLD = n183.BROAD_THRESHOLD
MIN_FULL_SUPPORT = n183.MIN_FULL_SUPPORT
CONTRIBUTION_CAP = n163.CONTRIBUTION_CAP
NONLOCAL_FAMILIES = n183.NONLOCAL_FAMILIES
SURPLUS_NAMES = ("surplus_max", "surplus_mean")
CONJUNCTIONS = ("product", "minimum")
CLOSURE_FEATURES = n183.CLOSURE_FEATURES
HYPOTHESES = {
    f"{feature}__{surplus_name}__{conjunction}__high": (
        feature,
        surplus_name,
        conjunction,
        1,
    )
    for feature in CLOSURE_FEATURES
    for surplus_name in SURPLUS_NAMES
    for conjunction in CONJUNCTIONS
}
EXPECTED_INPUT_SHA256 = {
    **n185.EXPECTED_INPUT_SHA256,
    "design": EXPECTED_DESIGN_SHA256,
    "next185_manifest": "e869b99a027c1f0788370b87d6511b95ce9ba9ef62333f0f2d421ac8b689849e",
    "next185_diagnostic": "e3168040ffbb1acfaec1765f1591922f31e58e6685cbb0b8d00a9e5321a0e886",
    "next185_table": "823cb892ce398f7ca7e4d328052b602a865c900480235ee906e1078ab0ff285a",
}


def compute_local_nonlocal_surplus(
    *, family_means: Mapping[str, object], base_support: object
) -> dict[str, np.ndarray]:
    """Compute strict and mean-reference local-risk surplus in risk units."""

    if set(family_means) != set(n163.FAMILY_PREFIXES):
        raise ValueError("NEXT186 family mean schema differs")
    support = np.asarray(base_support, dtype=bool)
    if support.ndim != 1:
        raise ValueError("NEXT186 base support differs")
    arrays = {
        name: np.asarray(values, dtype=float)
        for name, values in family_means.items()
    }
    if any(values.shape != support.shape for values in arrays.values()):
        raise ValueError("NEXT186 family/support shape differs")
    matrix = np.column_stack([arrays[name] for name in n163.FAMILY_PREFIXES])
    selected = matrix[support]
    if (
        np.any(~np.isfinite(selected))
        or np.any(selected < -1.0e-12)
        or np.any(selected > CONTRIBUTION_CAP + 1.0e-12)
    ):
        raise ValueError("NEXT186 family mean is outside the frozen cap")
    local = arrays["local_geometry"]
    nonlocal_values = np.column_stack([arrays[name] for name in NONLOCAL_FAMILIES])
    outputs = {
        "surplus_max": np.maximum(0.0, local - np.max(nonlocal_values, axis=1)),
        "surplus_mean": np.maximum(
            0.0, local - np.mean(nonlocal_values, axis=1)
        ),
    }
    for values in outputs.values():
        values[~support] = np.nan
    return outputs


def contradiction_relief(
    *, closure: object, surplus: object, conjunction: str
) -> np.ndarray:
    """Combine closure and local-risk surplus in the original risk units."""

    closure_values = np.asarray(closure, dtype=float)
    surplus_values = np.asarray(surplus, dtype=float)
    if (
        closure_values.ndim != 1
        or surplus_values.shape != closure_values.shape
        or conjunction not in CONJUNCTIONS
    ):
        raise ValueError("NEXT186 contradiction relief schema differs")
    finite_closure = np.isfinite(closure_values)
    finite_surplus = np.isfinite(surplus_values)
    if np.any(
        (closure_values[finite_closure] < -1.0e-12)
        | (closure_values[finite_closure] > 1.0 + 1.0e-12)
    ) or np.any(
        (surplus_values[finite_surplus] < -1.0e-12)
        | (surplus_values[finite_surplus] > CONTRIBUTION_CAP + 1.0e-12)
    ):
        raise ValueError("NEXT186 contradiction relief input is outside bounds")
    valid = finite_closure & finite_surplus
    result = np.full(len(closure_values), np.nan)
    bounded_closure = np.clip(closure_values[valid], 0.0, 1.0)
    bounded_surplus = np.clip(surplus_values[valid], 0.0, CONTRIBUTION_CAP)
    if conjunction == "product":
        result[valid] = bounded_closure * bounded_surplus
    else:
        result[valid] = np.minimum(
            CONTRIBUTION_CAP * bounded_closure, bounded_surplus
        )
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
    """Apply the unchanged frozen NEXT169/NEXT183 audit gates."""

    return n169.eligibility_from_metrics(
        scigen_full_support=scigen_full_support,
        wyformer_full_support=wyformer_full_support,
        scigen_shell_worst_auc=scigen_shell_worst_auc,
        scigen_shell_evaluable_folds=scigen_shell_evaluable_folds,
        wyformer_shell_pooled_auc=wyformer_shell_pooled_auc,
        scigen_full_pooled_auc=scigen_full_pooled_auc,
        wyformer_full_pooled_auc=wyformer_full_pooled_auc,
    )


def select_contradiction_relief_hypothesis(
    records: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    """Rank eligible relief hypotheses with deterministic tie breaking."""

    required = {
        "hypothesis",
        "eligible_for_search",
        "ranking_min_auc",
        "ranking_mean_auc",
    }
    if required - set(records.columns) or records["hypothesis"].astype(str).duplicated().any():
        raise ValueError("NEXT186 audit record schema differs")
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
    paths = n185._paths(roots, freeze_path, design_path)
    paths.update(
        {
            "next185_manifest": roots["next185"] / n185.MANIFEST_NAME,
            "next185_diagnostic": roots["next185"] / n185.DIAGNOSTIC_NAME,
            "next185_table": roots["next185"] / n185.PER_CANDIDATE_NAME,
        }
    )
    return paths


def run_local_nonlocal_contradiction_relief_audit(
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
    next183_dir: Path,
    next184_dir: Path,
    next185_dir: Path,
    next135_freeze_path: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only NEXT186 audit and publish atomically."""

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
                (183, next183_dir),
                (184, next184_dir),
                (185, next185_dir),
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
        raise FileNotFoundError("NEXT186 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT186 formal input identity differs: {differing}")

    manifest185 = json.loads(paths["next185_manifest"].read_text())
    expected_outputs185 = {
        n185.DIAGNOSTIC_NAME: input_hashes["next185_diagnostic"],
        n185.PER_CANDIDATE_NAME: input_hashes["next185_table"],
    }
    if (
        manifest185.get("protocol") != n185.PROTOCOL
        or manifest185.get("candidate_count") != n185.EXPECTED_CANDIDATE_COUNT
        or manifest185.get("next184_records_reproduced") is not True
        or manifest185.get("new_formula_searched") is not False
        or manifest185.get("new_formula_selected") is not False
        or manifest185.get("conditional_closure_broad_residual_diagnosed") is not True
        or manifest185.get("opened_validation_outputs_used") is not False
        or manifest185.get("scigen_replication_endpoint_opened") is not False
        or manifest185.get("wyformer_replication_endpoint_opened") is not False
        or manifest185.get("dft_calculation_executed") is not False
        or manifest185.get("dft_values_used_by_executable_formula") is not False
        or manifest185.get("learned_energy_force_stress_proxy_used") is not False
        or manifest185.get("physical_relaxation_executed") is not False
        or manifest185.get("outputs_sha256") != expected_outputs185
        or manifest185.get("executed_source_sha256", {}).get(
            "src/next185_conditional_closure_broad_residual_diagnostic.py"
        )
        != _sha256_file(Path(n185.__file__).resolve())
    ):
        raise ValueError("NEXT186 NEXT185 provenance differs")

    diagnostic164 = json.loads(paths["next164_diagnostic"].read_text())
    base_key = str(diagnostic164.get("global_closest", {}).get("candidate_key", ""))
    if hashlib.sha256(base_key.encode()).hexdigest() != EXPECTED_BASE_CANDIDATE_KEY_SHA256:
        raise ValueError("NEXT186 base candidate identity differs")

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
        raise ValueError("NEXT186 frozen closure feature schema differs")

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
        raise ValueError("NEXT186 base reconstruction differs")
    combined, base_terms, runtime = n163.materialize_candidates(
        features=combined,
        physical_terms=physical_terms,
        specs=selected_specs,
    )
    if len(base_terms) != 1 or len(runtime) != 1:
        raise RuntimeError("NEXT186 base materialization differs")
    base_score, base_support = _term_risk(combined, base_terms[0])
    family_means = n183.reconstruct_family_means(
        features=combined,
        physical_terms=physical_terms,
        base_spec=selected_specs[0],
        base_support=base_support,
    )
    surplus = compute_local_nonlocal_surplus(
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
        raise ValueError("NEXT186 endpoint row accounting differs")

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
        feature, surplus_name, conjunction, direction = HYPOTHESES[hypothesis]
        values = contradiction_relief(
            closure=pd.to_numeric(combined[feature], errors="coerce").to_numpy(float),
            surplus=surplus[surplus_name],
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
                raise ValueError("NEXT186 AUC population differs")
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
            "surplus_name": surplus_name,
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

    audit_table, selected = select_contradiction_relief_hypothesis(
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
        "contribution_cap": CONTRIBUTION_CAP,
        "nonlocal_families": list(NONLOCAL_FAMILIES),
        "surplus_definitions": {
            "surplus_max": "max(0,local_geometry-max(nonlocal_family_means))",
            "surplus_mean": "max(0,local_geometry-mean(nonlocal_family_means))",
        },
        "conjunction_definitions": {
            "product": "closure*surplus",
            "minimum": "min(0.5*closure,surplus)",
        },
        "population_counts": population_counts,
        "hypotheses": {
            name: {
                "closure_feature": value[0],
                "surplus_name": value[1],
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
        "src/next183_conditional_nonlocal_closure_audit.py": Path(n183.__file__).resolve(),
        "src/next185_conditional_closure_broad_residual_diagnostic.py": Path(n185.__file__).resolve(),
        "src/next186_local_nonlocal_contradiction_relief_audit.py": Path(__file__).resolve(),
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
            "local_nonlocal_contradiction_relief_branch_terminated": selected is None,
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
            raise RuntimeError("NEXT186 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT186 source changed before publication")
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
        183,
        184,
        185,
    )
    for stage in stages:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_local_nonlocal_contradiction_relief_audit(
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
    "CLOSURE_FEATURES",
    "CONJUNCTIONS",
    "HYPOTHESES",
    "SURPLUS_NAMES",
    "compute_local_nonlocal_surplus",
    "contradiction_relief",
    "eligibility_from_metrics",
    "run_local_nonlocal_contradiction_relief_audit",
    "select_contradiction_relief_hypothesis",
]


if __name__ == "__main__":
    raise SystemExit(main())
