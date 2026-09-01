#!/usr/bin/env python3
"""Audit weakest-site confidence confirmed by independent motif cleanliness."""

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

import src.next200_cross_source_motif_audit as n200
import src.next201_motif_weight_floor_repair as n201
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-08-next202-motif-conjunction-audit-v1"
MANIFEST_NAME = "MANIFEST.json"
AUDIT_NAME = "NEXT202_MOTIF_CONJUNCTION_AUDIT.json"
TABLE_NAME = "next202_motif_conjunction_audit.parquet"
EXPECTED_DESIGN_SHA256 = (
    "505c14d1f3505c67c65031460791f0d317b943a4a3792dc7c368f03cbe306f8a"
)
EXPECTED_INPUT_SHA256 = {
    **n201.EXPECTED_INPUT_SHA256,
    "design": EXPECTED_DESIGN_SHA256,
    "next201_manifest": "1c90b7c276f7ed1423eea8c71e5fbc813a714898ceaf782821e4a32c4c620fa6",
    "next201_catalogue": "d4812c7bd94ca8ec8ae31742b0fc4ed0482c548dd77b84b3603aa596cefa0298",
    "next201_evaluation": "6ed773fdc6172ab2dfad19608783f1d5f97476ae43b08cd8b7cbf5d712268aa5",
    "next201_formula": "8b4f24064c43f819c336c1ae1ac845d7f0c0f6c927782eed5be29cee3fffba10",
    "next201_search": "09d13de34d879a490988780b53c3e5fdae27031678bbd257a7f5c53ce6af6a84",
}
FLOOR_LEVELS = (
    ("tau0", 0.0),
    ("tau3_4", 3.0 / 4.0),
    ("tau15_16", 15.0 / 16.0),
    ("tau63_64", 63.0 / 64.0),
    ("tau255_256", 255.0 / 256.0),
    ("tau1023_1024", 1023.0 / 1024.0),
)
SECONDARY_FEATURES = (
    "motif_global_dispersion_rms",
    "motif_weight_sum_std",
)
CONJUNCTIONS = ("product", "minimum")
HYPOTHESES = {
    f"motif_weight_sum_min__{label}__{secondary}__{conjunction}__protected_high": (
        secondary,
        floor,
        conjunction,
        1,
    )
    for secondary in SECONDARY_FEATURES
    for label, floor in FLOOR_LEVELS
    for conjunction in CONJUNCTIONS
}
SAFE_THRESHOLD = n200.SAFE_THRESHOLD
BROAD_THRESHOLD = n200.BROAD_THRESHOLD


def weakest_site_confidence(
    values: object, *, floor_threshold: float
) -> np.ndarray:
    """Map the weakest-site CrystalNN weight through one frozen ramp."""

    tau = float(floor_threshold)
    if not np.isfinite(tau) or tau < 0.0 or tau >= 1.0:
        raise ValueError("NEXT202 floor threshold differs")
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("NEXT202 weakest-site schema differs")
    result = np.full(array.shape, np.nan, dtype=float)
    finite = np.isfinite(array)
    bounded = np.clip(array[finite], 0.0, 1.0)
    result[finite] = np.clip((bounded - tau) / (1.0 - tau), 0.0, 1.0)
    return result


def secondary_cleanliness(values: object, *, feature: str) -> np.ndarray:
    """Return one of the two fixed, bounded motif-cleanliness maps."""

    if feature not in SECONDARY_FEATURES:
        raise ValueError("NEXT202 secondary feature differs")
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("NEXT202 secondary schema differs")
    result = np.full(array.shape, np.nan, dtype=float)
    finite = np.isfinite(array)
    nonnegative = np.maximum(array[finite], 0.0)
    if feature == "motif_global_dispersion_rms":
        result[finite] = 1.0 / (1.0 + nonnegative)
    else:
        result[finite] = np.clip(1.0 - 2.0 * nonnegative, 0.0, 1.0)
    return result


def motif_conjunction_certificate(
    *, weakest_site: object, secondary: object, conjunction: str
) -> np.ndarray:
    """Combine two bounded protection certificates on their common support."""

    if conjunction not in CONJUNCTIONS:
        raise ValueError("NEXT202 conjunction differs")
    weakest = np.asarray(weakest_site, dtype=float)
    clean = np.asarray(secondary, dtype=float)
    if weakest.ndim != 1 or clean.shape != weakest.shape:
        raise ValueError("NEXT202 certificate schema differs")
    valid = np.isfinite(weakest) & np.isfinite(clean)
    for values in (weakest, clean):
        finite = np.isfinite(values)
        if np.any(
            (values[finite] < -1.0e-12) | (values[finite] > 1.0 + 1.0e-12)
        ):
            raise ValueError("NEXT202 certificate input is outside bounds")
    result = np.full(weakest.shape, np.nan, dtype=float)
    left = np.clip(weakest[valid], 0.0, 1.0)
    right = np.clip(clean[valid], 0.0, 1.0)
    if conjunction == "product":
        result[valid] = left * right
    else:
        result[valid] = np.minimum(left, right)
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
    """Apply the unchanged frozen NEXT200 cross-source gates."""

    return n200.eligibility_from_metrics(
        scigen_full_support=scigen_full_support,
        wyformer_full_support=wyformer_full_support,
        scigen_shell_worst_auc=scigen_shell_worst_auc,
        scigen_shell_evaluable_folds=scigen_shell_evaluable_folds,
        wyformer_shell_pooled_auc=wyformer_shell_pooled_auc,
        scigen_full_pooled_auc=scigen_full_pooled_auc,
        wyformer_full_pooled_auc=wyformer_full_pooled_auc,
    )


def select_motif_conjunction_hypothesis(
    records: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    """Rank eligible hypotheses with deterministic tie breaking."""

    required = {
        "hypothesis",
        "eligible_for_search",
        "ranking_min_auc",
        "ranking_mean_auc",
    }
    if required - set(records.columns) or records["hypothesis"].astype(str).duplicated().any():
        raise ValueError("NEXT202 audit record schema differs")
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
    paths = n201._paths(roots, freeze_path, design_path)
    paths.update(
        {
            "next201_manifest": roots["next201"] / n201.MANIFEST_NAME,
            "next201_catalogue": roots["next201"] / n201.CATALOGUE_NAME,
            "next201_evaluation": roots["next201"] / n201.EVALUATION_NAME,
            "next201_formula": roots["next201"] / n201.FORMULA_NAME,
            "next201_search": roots["next201"] / n201.SEARCH_NAME,
        }
    )
    return paths


def _verify_prior_boundaries(
    paths: Mapping[str, Path], input_hashes: Mapping[str, str]
) -> dict[str, object]:
    n200._verify_provenance(paths, input_hashes)
    n201._verify_next200(paths, input_hashes)
    manifest = json.loads(paths["next201_manifest"].read_text())
    evaluation = json.loads(paths["next201_evaluation"].read_text())
    if (
        manifest.get("protocol") != n201.PROTOCOL
        or manifest.get("candidate_count") != n201.EXPECTED_CANDIDATE_COUNT
        or manifest.get("base_endpoint_reproduced") is not True
        or manifest.get("passes_all_cross_source_discovery_gates") is not False
        or manifest.get("freeze_authorized") is not False
        or manifest.get("motif_weight_floor_search_branch_terminated") is not True
        or manifest.get("opened_validation_outputs_used") is not False
        or manifest.get("scigen_replication_endpoint_opened") is not False
        or manifest.get("wyformer_replication_endpoint_opened") is not False
        or manifest.get("dft_calculation_executed") is not False
        or manifest.get("dft_values_used_by_executable_formula") is not False
        or manifest.get("learned_energy_force_stress_proxy_used") is not False
        or manifest.get("model_or_proxy_potential_used") is not False
        or manifest.get("physical_relaxation_executed") is not False
        or manifest.get("outputs_sha256")
        != {
            n201.CATALOGUE_NAME: input_hashes["next201_catalogue"],
            n201.EVALUATION_NAME: input_hashes["next201_evaluation"],
            n201.FORMULA_NAME: input_hashes["next201_formula"],
            n201.SEARCH_NAME: input_hashes["next201_search"],
        }
        or manifest.get("executed_source_sha256", {}).get(
            "src/next201_motif_weight_floor_repair.py"
        )
        != _sha256_file(Path(n201.__file__).resolve())
    ):
        raise ValueError("NEXT202 NEXT201 provenance differs")
    if (
        evaluation.get("protocol") != n201.PROTOCOL
        or evaluation.get("candidate_count") != n201.EXPECTED_CANDIDATE_COUNT
        or evaluation.get("base_endpoint_reproduced") is not True
        or evaluation.get("passes_all_cross_source_discovery_gates") is not False
        or evaluation.get("freeze_authorized") is not False
        or evaluation.get("requires_unopened_internal_validation_before_claim") is not True
    ):
        raise ValueError("NEXT202 NEXT201 evaluation boundary differs")
    return json.loads(paths["next200_audit"].read_text())


def run_motif_conjunction_audit(
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
    next186_dir: Path,
    next188_dir: Path,
    next190_dir: Path,
    next192_dir: Path,
    next194_dir: Path,
    next199_dir: Path,
    next200_dir: Path,
    next201_dir: Path,
    next135_freeze_path: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only NEXT202 audit and publish atomically."""

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
        "next194": Path(next194_dir).resolve(),
        "next199": Path(next199_dir).resolve(),
        "next200": Path(next200_dir).resolve(),
        "next201": Path(next201_dir).resolve(),
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots, Path(next135_freeze_path).resolve(), Path(design_path).resolve()
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT202 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT202 formal input identity differs: {differing}")
    audit200 = _verify_prior_boundaries(paths, input_hashes)

    diagnostic164 = json.loads(paths["next164_diagnostic"].read_text())
    base_key = str(diagnostic164.get("global_closest", {}).get("candidate_key", ""))
    if hashlib.sha256(base_key.encode()).hexdigest() != n201.EXPECTED_BASE_CANDIDATE_KEY_SHA256:
        raise ValueError("NEXT202 base candidate identity differs")

    extended, _, old_terms, mhcr_terms = n200.n194.n130._join_label_free_features(paths)
    compact_frames = []
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next133_{source}_features"]).copy()
        table["material_id"] = source + ":" + table["material_id"].astype(str)
        compact_frames.append(table)
    extended = extended.merge(
        pd.concat(compact_frames, ignore_index=True),
        on="material_id", how="inner", validate="one_to_one",
    )
    extended = pd.concat(
        [
            extended.reset_index(drop=True),
            n200.n194.n135.materialize_conjunctive_features(extended).reset_index(drop=True),
        ],
        axis=1,
    )
    closure_frames = []
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next179_{source}_features"]).copy()
        table["material_id"] = source + ":" + table["material_id"].astype(str)
        closure_frames.append(table)
    extended = extended.merge(
        pd.concat(closure_frames, ignore_index=True),
        on="material_id", how="inner", validate="one_to_one",
    )
    motif_frames = []
    motif_columns = [
        "material_id", "motif_weight_sum_min", *SECONDARY_FEATURES
    ]
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next199_{source}_features"])[motif_columns].copy()
        if table["material_id"].astype(str).duplicated().any():
            raise ValueError(f"NEXT202 {source} motif identity differs")
        table["material_id"] = source + ":" + table["material_id"].astype(str)
        motif_frames.append(table)
    motif_table = pd.concat(motif_frames, ignore_index=True)
    extended = extended.merge(
        motif_table, on="material_id", how="inner", validate="one_to_one"
    )
    if len(extended) != len(motif_table):
        raise ValueError("NEXT202 motif row accounting differs")

    physical_terms = [*old_terms, *mhcr_terms]
    physical_ids = {str(term["term_id"]) for term in physical_terms}
    all_bases = n200.n194.n130.n127.select_next125_bases(
        pd.read_parquet(paths["next125_search_records"])
    )
    bases = n200.n194.n135.n132.select_extended_bases(
        pd.read_parquet(paths["next130_search_records"]), all_bases
    )
    base_specs = n200.n194.n163.build_candidate_specs(
        bases=bases, physical_term_ids=physical_ids
    )
    selected_specs = [
        spec for spec in base_specs if str(spec["candidate_key"]) == base_key
    ]
    if len(selected_specs) != 1:
        raise ValueError("NEXT202 base reconstruction differs")
    extended, base_terms, runtime = n200.n194.n163.materialize_candidates(
        features=extended, physical_terms=physical_terms, specs=selected_specs
    )
    if len(base_terms) != 1 or len(runtime) != 1:
        raise RuntimeError("NEXT202 base materialization differs")
    base_score, base_support = n200.n194.n87._term_risk(extended, base_terms[0])

    scigen_endpoint = pd.read_parquet(paths["scigen_endpoint"])
    wyformer_endpoint = pd.read_parquet(paths["wyformer_endpoint"])
    endpoint_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "material_id": "scigen:" + scigen_endpoint["material_id"].astype(str),
                    "_endpoint": pd.to_numeric(
                        scigen_endpoint["distortion_ratio"], errors="coerce"
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "material_id": "wyformer:" + wyformer_endpoint["material_id"].astype(str),
                    "_endpoint": n200.n194.n130.n125.n121.prior._endpoint_numeric(
                        wyformer_endpoint["endpoint_stratum"]
                    ),
                }
            ),
        ],
        ignore_index=True,
    )
    extended = extended.merge(
        endpoint_frame, on="material_id", how="inner", validate="one_to_one"
    )
    endpoint = pd.to_numeric(extended.pop("_endpoint"), errors="coerce").to_numpy(float)
    if len(extended) != len(base_score) or not np.isfinite(endpoint).all():
        raise ValueError("NEXT202 endpoint row accounting differs")
    sources = extended["source_dataset"].astype(str).to_numpy()
    folds = n200.n194.n164.assign_group_folds(
        extended["reduced_formula"].astype(str).to_numpy()
    )
    extremes = base_support & ((endpoint <= 1.0) | (endpoint >= 2.0))
    shell = extremes & (base_score >= BROAD_THRESHOLD) & (base_score < SAFE_THRESHOLD)
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

    raw_weakest = pd.to_numeric(
        extended["motif_weight_sum_min"], errors="coerce"
    ).to_numpy(float)
    weakest_by_floor = {
        floor: weakest_site_confidence(raw_weakest, floor_threshold=floor)
        for _, floor in FLOOR_LEVELS
    }
    clean_by_feature = {
        feature: secondary_cleanliness(
            pd.to_numeric(extended[feature], errors="coerce").to_numpy(float),
            feature=feature,
        )
        for feature in SECONDARY_FEATURES
    }
    records: list[dict[str, object]] = []
    for hypothesis in sorted(HYPOTHESES):
        secondary_feature, floor, conjunction, direction = HYPOTHESES[hypothesis]
        values = motif_conjunction_certificate(
            weakest_site=weakest_by_floor[floor],
            secondary=clean_by_feature[secondary_feature],
            conjunction=conjunction,
        )
        evaluations = {}
        for population, mask in populations.items():
            evaluation = n200.n194.n151._evaluate_auc(
                values=values[mask],
                protected=endpoint[mask] <= 1.0,
                folds=folds[mask],
                direction=direction,
            )
            if evaluation is None:
                raise ValueError("NEXT202 AUC population differs")
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
            "secondary_feature": secondary_feature,
            "floor_threshold": floor,
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
                "pooled_auc", "macro_auc", "worst_auc", "evaluable_folds",
                "protected", "severe",
            ):
                record[f"{population}_{metric}"] = evaluation[metric]
            record[f"{population}_fold_aucs_json"] = json.dumps(
                evaluation["fold_aucs"], separators=(",", ":")
            )
        records.append(record)

    audit_table, selected = select_motif_conjunction_hypothesis(pd.DataFrame(records))
    eligible_names = audit_table.loc[
        audit_table["eligible_for_search"].fillna(False).astype(bool), "hypothesis"
    ].astype(str).tolist()
    audit = {
        "protocol": PROTOCOL,
        "design_sha256": input_hashes["design"],
        "candidate_key_sha256": n201.EXPECTED_BASE_CANDIDATE_KEY_SHA256,
        "safe_threshold": SAFE_THRESHOLD,
        "broad_threshold": BROAD_THRESHOLD,
        "population_counts": population_counts,
        "floor_levels": [{"label": label, "value": floor} for label, floor in FLOOR_LEVELS],
        "secondary_cleanliness": {
            "motif_global_dispersion_rms": "1/(1+max(0,x))",
            "motif_weight_sum_std": "clip(1-2*max(0,x),0,1)",
        },
        "conjunctions": list(CONJUNCTIONS),
        "hypotheses": {
            name: {
                "secondary_feature": value[0],
                "floor_threshold": value[1],
                "conjunction": value[2],
                "direction": value[3],
            }
            for name, value in HYPOTHESES.items()
        },
        "eligibility_gates": audit200.get("eligibility_gates"),
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
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {
        "src/next199_cross_source_motif_features.py": Path(n200.n199.__file__).resolve(),
        "src/next200_cross_source_motif_audit.py": Path(n200.__file__).resolve(),
        "src/next201_motif_weight_floor_repair.py": Path(n201.__file__).resolve(),
        "src/next202_motif_conjunction_audit.py": Path(__file__).resolve(),
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
            "motif_conjunction_branch_terminated": selected is None,
            "next203_search_authorized": selected is not None,
            "new_formula_searched": False,
            "discovery_outcomes_used_as_offline_labels": True,
            "discovery_endpoints_opened": True,
            "opened_validation_outputs_used": False,
            "scigen_replication_endpoint_opened": False,
            "wyformer_replication_endpoint_opened": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_executable_formula": False,
            "learned_energy_force_stress_proxy_used": False,
            "model_or_proxy_potential_used": False,
            "physical_relaxation_executed": False,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {
                AUDIT_NAME: _sha256_file(audit_path),
                TABLE_NAME: _sha256_file(table_path),
            },
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT202 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT202 source changed before publication")
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
    for stage in stages:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next194-dir", type=Path, required=True)
    parser.add_argument("--next199-dir", type=Path, required=True)
    parser.add_argument("--next200-dir", type=Path, required=True)
    parser.add_argument("--next201-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_motif_conjunction_audit(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{f"next{stage}_dir": getattr(args, f"next{stage}_dir") for stage in stages},
        next194_dir=args.next194_dir,
        next199_dir=args.next199_dir,
        next200_dir=args.next200_dir,
        next201_dir=args.next201_dir,
        next135_freeze_path=args.next135_freeze_path,
        design_path=args.design_path,
        output_dir=args.output_dir,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "CONJUNCTIONS",
    "FLOOR_LEVELS",
    "HYPOTHESES",
    "SECONDARY_FEATURES",
    "eligibility_from_metrics",
    "motif_conjunction_certificate",
    "run_motif_conjunction_audit",
    "secondary_cleanliness",
    "select_motif_conjunction_hypothesis",
    "weakest_site_confidence",
]


if __name__ == "__main__":
    raise SystemExit(main())
