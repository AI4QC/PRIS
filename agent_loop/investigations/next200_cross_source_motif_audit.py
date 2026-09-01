#!/usr/bin/env python3
"""Audit frozen x0 motif-coherence directions on both discovery sources."""

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

import src.next194_signed_local_closure_audit as n194
import src.next199_cross_source_motif_features as n199
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-08-next200-cross-source-motif-audit-v1"
MANIFEST_NAME = "MANIFEST.json"
AUDIT_NAME = "NEXT200_CROSS_SOURCE_MOTIF_AUDIT.json"
TABLE_NAME = "next200_cross_source_motif_audit.parquet"
EXPECTED_DESIGN_SHA256 = (
    "6d7d8a3cacc609c089ea60b8863683f852c72a9f843d3de6ca90d4dd0a7e4703"
)
EXPECTED_INPUT_SHA256 = {
    **n194.EXPECTED_INPUT_SHA256,
    "design": EXPECTED_DESIGN_SHA256,
    "next194_manifest": "d865d436012f3740933dc843dc45f498a71942955af02d650fb4535ebfffb27a",
    "next194_audit": "07717d4c441dec74553015b0b8f3cc76507ef6f19913011aa944dfbe3f0de783",
    "next194_table": "a878b8f9e43845d99e404a4ec341b53482c20b1803660aeea254b4d2bc4b2315",
    "next199_manifest": "e465bdd35d1e66fd11ed04f414dfc5a0c77d1aa03a75376056c044783c012837",
    "next199_catalogue": "8e0987447b7694e754ef3b94aedcbd77bbe5fbf5c1d6c9ed1e928c158b9d7927",
    "next199_scigen_features": "0f51fea8b967df20caa60d160e8cb66ffbacebe9e4c75ebf2d4987768fefef49",
    "next199_wyformer_features": "cfa342b5611e782579827a0348ec0a647a5f3a34a68de2f89615319b0157ffa4",
}

HIGH_PROTECTION_FEATURES = (
    "motif_weight_sum_mean",
    "motif_weight_sum_min",
    "motif_cn_dominance_mean",
    "motif_cn_dominance_min",
    "motif_effective_cn_mean",
    "motif_order_strength_mean",
    "motif_order_strength_min",
    "motif_fingerprint_norm_mean",
    "motif_species_centroid_separation_mean",
)
LOW_PROTECTION_FEATURES = (
    "motif_weight_sum_std",
    "motif_cn_dominance_std",
    "motif_cn_entropy_mean",
    "motif_cn_entropy_q95",
    "motif_effective_cn_std",
    "motif_effective_cn_range",
    "motif_order_strength_std",
    "motif_fingerprint_norm_std",
    "motif_same_element_dispersion_rms",
    "motif_same_element_dispersion_q95",
    "motif_same_element_dispersion_max",
    "motif_global_dispersion_rms",
)
HYPOTHESES = {
    **{
        f"{feature}__protected_high": (feature, 1)
        for feature in HIGH_PROTECTION_FEATURES
    },
    **{
        f"{feature}__protected_low": (feature, -1)
        for feature in LOW_PROTECTION_FEATURES
    },
}
SAFE_THRESHOLD = n194.n192.n190.n188.n186.SAFE_THRESHOLD
BROAD_THRESHOLD = n194.n192.n190.n188.n186.BROAD_THRESHOLD


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
    """Apply the unchanged frozen NEXT194 cross-source gates."""

    return n194.eligibility_from_metrics(
        scigen_full_support=scigen_full_support,
        wyformer_full_support=wyformer_full_support,
        scigen_shell_worst_auc=scigen_shell_worst_auc,
        scigen_shell_evaluable_folds=scigen_shell_evaluable_folds,
        wyformer_shell_pooled_auc=wyformer_shell_pooled_auc,
        scigen_full_pooled_auc=scigen_full_pooled_auc,
        wyformer_full_pooled_auc=wyformer_full_pooled_auc,
    )


def select_motif_hypothesis(
    records: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    """Rank eligible frozen directions with deterministic tie breaking."""

    required = {
        "hypothesis",
        "eligible_for_search",
        "ranking_min_auc",
        "ranking_mean_auc",
    }
    if required - set(records.columns) or records["hypothesis"].astype(str).duplicated().any():
        raise ValueError("NEXT200 audit record schema differs")
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
    paths = n194._paths(roots, freeze_path, design_path)
    paths.update(
        {
            "next194_manifest": roots["next194"] / n194.MANIFEST_NAME,
            "next194_audit": roots["next194"] / n194.AUDIT_NAME,
            "next194_table": roots["next194"] / n194.TABLE_NAME,
            "next199_manifest": roots["next199"] / n199.MANIFEST_NAME,
            "next199_catalogue": roots["next199"] / n199.CATALOGUE_NAME,
            "next199_scigen_features": roots["next199"] / n199.FEATURE_FILES["scigen"],
            "next199_wyformer_features": roots["next199"] / n199.FEATURE_FILES["wyformer"],
        }
    )
    return paths


def _verify_provenance(paths: Mapping[str, Path], input_hashes: Mapping[str, str]) -> dict[str, object]:
    manifest194 = json.loads(paths["next194_manifest"].read_text())
    audit194 = json.loads(paths["next194_audit"].read_text())
    if (
        manifest194.get("protocol") != n194.PROTOCOL
        or manifest194.get("hypothesis_count") != len(n194.HYPOTHESES)
        or int(manifest194.get("eligible_hypothesis_count", -1)) <= 0
        or manifest194.get("next195_search_authorized") is not True
        or manifest194.get("new_formula_searched") is not False
        or manifest194.get("opened_validation_outputs_used") is not False
        or manifest194.get("scigen_replication_endpoint_opened") is not False
        or manifest194.get("wyformer_replication_endpoint_opened") is not False
        or manifest194.get("dft_calculation_executed") is not False
        or manifest194.get("dft_values_used_by_executable_formula") is not False
        or manifest194.get("learned_energy_force_stress_proxy_used") is not False
        or manifest194.get("physical_relaxation_executed") is not False
        or manifest194.get("outputs_sha256")
        != {
            n194.AUDIT_NAME: input_hashes["next194_audit"],
            n194.TABLE_NAME: input_hashes["next194_table"],
        }
        or manifest194.get("executed_source_sha256", {}).get(
            "src/next194_signed_local_closure_audit.py"
        )
        != _sha256_file(Path(n194.__file__).resolve())
    ):
        raise ValueError("NEXT200 NEXT194 provenance differs")
    if (
        audit194.get("protocol") != n194.PROTOCOL
        or not audit194.get("eligible_hypotheses")
        or audit194.get("selected_hypothesis") is None
        or audit194.get("new_formula_searched") is not False
        or audit194.get("validation_or_replication_opened") is not False
    ):
        raise ValueError("NEXT200 NEXT194 audit boundary differs")

    manifest199 = json.loads(paths["next199_manifest"].read_text())
    catalogue199 = json.loads(paths["next199_catalogue"].read_text())
    if (
        manifest199.get("protocol") != n199.PROTOCOL
        or manifest199.get("labels_opened") is not False
        or manifest199.get("discovery_endpoints_opened") is not False
        or manifest199.get("internal_validation_geometry_opened") is not False
        or manifest199.get("internal_replication_geometry_opened") is not False
        or manifest199.get("validation_endpoints_opened") is not False
        or manifest199.get("replication_endpoints_opened") is not False
        or manifest199.get("dft_calculation_executed") is not False
        or manifest199.get("dft_values_used_by_features") is not False
        or manifest199.get("learned_energy_force_stress_proxy_used") is not False
        or manifest199.get("model_or_proxy_potential_used") is not False
        or manifest199.get("physical_relaxation_executed") is not False
        or manifest199.get("outputs_sha256")
        != {
            n199.CATALOGUE_NAME: input_hashes["next199_catalogue"],
            n199.FEATURE_FILES["scigen"]: input_hashes["next199_scigen_features"],
            n199.FEATURE_FILES["wyformer"]: input_hashes["next199_wyformer_features"],
        }
        or manifest199.get("executed_source_sha256", {}).get(
            "src/next199_cross_source_motif_features.py"
        )
        != _sha256_file(Path(n199.__file__).resolve())
    ):
        raise ValueError("NEXT200 NEXT199 provenance differs")
    if (
        catalogue199.get("protocol") != n199.PROTOCOL
        or catalogue199.get("feature_count") != len(n199.FEATURE_NAMES)
        or catalogue199.get("feature_names") != list(n199.FEATURE_NAMES)
        or catalogue199.get("endpoint_columns_present") is not False
        or catalogue199.get("labels_opened") is not False
        or catalogue199.get("source_partitions_read")
        != {"scigen": ["discovery"], "wyformer": ["discovery"]}
    ):
        raise ValueError("NEXT200 NEXT199 catalogue differs")
    return audit194


def run_cross_source_motif_audit(
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
    next135_freeze_path: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only NEXT200 audit and publish atomically."""

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
    }
    target = Path(output_dir).resolve()
    paths = _paths(
        roots, Path(next135_freeze_path).resolve(), Path(design_path).resolve()
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT200 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT200 formal input identity differs: {differing}")
    audit194 = _verify_provenance(paths, input_hashes)

    diagnostic164 = json.loads(paths["next164_diagnostic"].read_text())
    base_key = str(diagnostic164.get("global_closest", {}).get("candidate_key", ""))
    if hashlib.sha256(base_key.encode()).hexdigest() != n194.n192.n190.n186_candidate_key_sha256():
        raise ValueError("NEXT200 base candidate identity differs")

    extended, _, old_terms, mhcr_terms = n194.n130._join_label_free_features(paths)
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
            n194.n135.materialize_conjunctive_features(extended).reset_index(drop=True),
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
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next199_{source}_features"])[
            ["material_id", *n199.FEATURE_NAMES, "motif_supported"]
        ].copy()
        if table["material_id"].astype(str).duplicated().any():
            raise ValueError(f"NEXT200 {source} motif identity differs")
        table["material_id"] = source + ":" + table["material_id"].astype(str)
        motif_frames.append(table)
    motif_table = pd.concat(motif_frames, ignore_index=True)
    extended = extended.merge(
        motif_table, on="material_id", how="inner", validate="one_to_one"
    )
    if len(extended) != len(motif_table) or set(n199.FEATURE_NAMES) - set(extended.columns):
        raise ValueError("NEXT200 motif row accounting differs")

    physical_terms = [*old_terms, *mhcr_terms]
    physical_ids = {str(term["term_id"]) for term in physical_terms}
    all_bases = n194.n130.n127.select_next125_bases(
        pd.read_parquet(paths["next125_search_records"])
    )
    bases = n194.n135.n132.select_extended_bases(
        pd.read_parquet(paths["next130_search_records"]), all_bases
    )
    base_specs = n194.n163.build_candidate_specs(
        bases=bases, physical_term_ids=physical_ids
    )
    selected_specs = [
        spec for spec in base_specs if str(spec["candidate_key"]) == base_key
    ]
    if len(selected_specs) != 1:
        raise ValueError("NEXT200 base reconstruction differs")
    extended, base_terms, runtime = n194.n163.materialize_candidates(
        features=extended, physical_terms=physical_terms, specs=selected_specs
    )
    if len(base_terms) != 1 or len(runtime) != 1:
        raise RuntimeError("NEXT200 base materialization differs")
    base_score, base_support = n194.n87._term_risk(extended, base_terms[0])

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
                    "_endpoint": n194.n130.n125.n121.prior._endpoint_numeric(
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
        raise ValueError("NEXT200 endpoint row accounting differs")
    sources = extended["source_dataset"].astype(str).to_numpy()
    folds = n194.n164.assign_group_folds(
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

    records: list[dict[str, object]] = []
    for hypothesis in sorted(HYPOTHESES):
        feature, direction = HYPOTHESES[hypothesis]
        values = pd.to_numeric(extended[feature], errors="coerce").to_numpy(float)
        evaluations = {}
        for population, mask in populations.items():
            evaluation = n194.n151._evaluate_auc(
                values=values[mask],
                protected=endpoint[mask] <= 1.0,
                folds=folds[mask],
                direction=direction,
            )
            if evaluation is None:
                raise ValueError("NEXT200 AUC population differs")
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
            "feature": feature,
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

    audit_table, selected = select_motif_hypothesis(pd.DataFrame(records))
    eligible_names = audit_table.loc[
        audit_table["eligible_for_search"].fillna(False).astype(bool), "hypothesis"
    ].astype(str).tolist()
    audit = {
        "protocol": PROTOCOL,
        "design_sha256": input_hashes["design"],
        "candidate_key_sha256": n194.n192.n190.n186_candidate_key_sha256(),
        "safe_threshold": SAFE_THRESHOLD,
        "broad_threshold": BROAD_THRESHOLD,
        "population_counts": population_counts,
        "hypotheses": {
            name: {"feature": value[0], "direction": value[1]}
            for name, value in HYPOTHESES.items()
        },
        "eligibility_gates": audit194.get("eligibility_gates"),
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
        "src/next194_signed_local_closure_audit.py": Path(n194.__file__).resolve(),
        "src/next199_cross_source_motif_features.py": Path(n199.__file__).resolve(),
        "src/next200_cross_source_motif_audit.py": Path(__file__).resolve(),
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
            "motif_coherence_branch_terminated": selected is None,
            "next201_search_authorized": selected is not None,
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
            raise RuntimeError("NEXT200 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT200 source changed before publication")
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
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_cross_source_motif_audit(
        scigen_feature_dir=args.scigen_feature_dir,
        scigen_discovery_endpoint_dir=args.scigen_discovery_endpoint_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        wyformer_discovery_endpoint_dir=args.wyformer_discovery_endpoint_dir,
        **{f"next{stage}_dir": getattr(args, f"next{stage}_dir") for stage in stages},
        next194_dir=args.next194_dir,
        next199_dir=args.next199_dir,
        next135_freeze_path=args.next135_freeze_path,
        design_path=args.design_path,
        output_dir=args.output_dir,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "HIGH_PROTECTION_FEATURES",
    "LOW_PROTECTION_FEATURES",
    "HYPOTHESES",
    "eligibility_from_metrics",
    "run_cross_source_motif_audit",
    "select_motif_hypothesis",
]


if __name__ == "__main__":
    raise SystemExit(main())
