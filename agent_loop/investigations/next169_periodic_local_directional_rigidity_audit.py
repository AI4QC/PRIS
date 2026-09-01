#!/usr/bin/env python3
"""Audit frozen local directional rigidity in the exact NEXT164 repair shell."""

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
import src.next168_periodic_local_directional_rigidity as n168
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json
from src.next87_scigen_sparse_law_search import _term_risk, assign_group_folds


PROTOCOL = "2026-08-08-next169-periodic-local-directional-rigidity-audit-v1"
MANIFEST_NAME = "MANIFEST.json"
AUDIT_NAME = "NEXT169_PERIODIC_LOCAL_DIRECTIONAL_RIGIDITY_AUDIT.json"
TABLE_NAME = "next169_periodic_local_directional_rigidity_audit.parquet"
MIN_FULL_SUPPORT = 0.90
EXPECTED_DESIGN_SHA256 = "65f6e73b2de5ed8dd7ae7be22e582e69282fa141d76d31157f8f2bd318b79b5f"
EXPECTED_CANDIDATE_KEY_SHA256 = "1d0ea8331f38aa69cfdedbe664d5ceb46c14e166e121bae92d9e14dd4fc6109e"
SAFE_THRESHOLD = 0.5415470292150686
BROAD_THRESHOLD = 0.21976295573076796
EXPECTED_INPUT_SHA256 = {
    **n164.EXPECTED_INPUT_SHA256,
    "design": EXPECTED_DESIGN_SHA256,
    "next164_manifest": "21489c49397e98fc3b5c1f7e4b45d4ee8b4bd2108e2fc4fbeffa17b4b9b5eb58",
    "next164_diagnostic": "12176c0ab6069ee48b1eeaf0dcc04ba7a3b24ed85f785a49221ad7bc703ad20d",
    "next164_per_candidate": "8487d76fbf345aab2778428be383c3006907e4e5c74f92837f3041c0dd146278",
    "next168_manifest": "e83eeae0ba6074996aa2d29042d8fe79e724f092112842b6b93fb131efd9942c",
    "next168_catalogue": "61d1fc2dbb098df1f508d87093e59e1e1f7e4317acd95c32ad24df0c2340d899",
    "next168_scigen_features": "01e718d3085746cd51b46d0fa927a5d83d489a127b33051dd45629793032432a",
    "next168_wyformer_features": "826b66bed9e205de6218ad54e771480f2959170d46e8d517a8c36dcaf51b5153",
}

HYPOTHESES = {
    f"{mode}_{suffix}__high": (f"pldr_{mode}_{suffix}", 1)
    for mode in ("voronoi", "crystalnn")
    for suffix in (
        "tightness_min",
        "tightness_q10",
        "tightness_mean",
        "volume_q10",
        "volume_mean",
    )
}


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
    """Apply every frozen NEXT169 gate without rounding."""

    return bool(
        scigen_full_support >= MIN_FULL_SUPPORT
        and wyformer_full_support >= MIN_FULL_SUPPORT
        and scigen_shell_worst_auc >= 0.55
        and scigen_shell_evaluable_folds == 5
        and wyformer_shell_pooled_auc >= 0.55
        and scigen_full_pooled_auc >= 0.50
        and wyformer_full_pooled_auc >= 0.50
    )


def select_directional_rigidity_hypothesis(
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
        raise ValueError("NEXT169 audit record schema differs")
    table = records.sort_values(
        [
            "eligible_for_search",
            "ranking_min_auc",
            "ranking_mean_auc",
            "hypothesis",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    eligible = table.loc[table["eligible_for_search"].fillna(False).astype(bool)]
    selected = None if eligible.empty else eligible.iloc[0].to_dict()
    return table, selected


def _paths(
    roots: Mapping[str, Path], freeze_path: Path, design_path: Path
) -> dict[str, Path]:
    paths = n164._paths(roots, freeze_path, design_path)
    paths.update(
        {
            "next164_manifest": roots["next164"] / n164.MANIFEST_NAME,
            "next164_diagnostic": roots["next164"] / n164.DIAGNOSTIC_NAME,
            "next164_per_candidate": roots["next164"] / n164.PER_CANDIDATE_NAME,
            "next168_manifest": roots["next168"] / n168.MANIFEST_NAME,
            "next168_catalogue": roots["next168"] / n168.CATALOGUE_NAME,
            "next168_scigen_features": roots["next168"] / n168.SCIGEN_NAME,
            "next168_wyformer_features": roots["next168"] / n168.WYFORMER_NAME,
        }
    )
    return paths


def run_periodic_local_directional_rigidity_audit(
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
    next135_freeze_path: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only NEXT169 audit and publish atomically."""

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
        raise FileNotFoundError("NEXT169 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            key
            for key in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(key) != EXPECTED_INPUT_SHA256.get(key)
        )
        raise ValueError(f"NEXT169 formal input identity differs: {differing}")

    manifest164 = json.loads(paths["next164_manifest"].read_text())
    diagnostic164 = json.loads(paths["next164_diagnostic"].read_text())
    closest = diagnostic164.get("global_closest", {})
    candidate_key = str(closest.get("candidate_key", ""))
    if (
        manifest164.get("protocol") != n164.PROTOCOL
        or manifest164.get("opened_validation_outputs_used") is not False
        or manifest164.get("dft_values_used_by_executable_formula") is not False
        or manifest164.get("outputs_sha256", {}).get(n164.DIAGNOSTIC_NAME)
        != input_hashes["next164_diagnostic"]
        or hashlib.sha256(candidate_key.encode()).hexdigest()
        != EXPECTED_CANDIDATE_KEY_SHA256
        or not np.isclose(
            float(closest.get("safe_threshold", np.nan)),
            SAFE_THRESHOLD,
            rtol=0.0,
            atol=1.0e-15,
        )
        or not np.isclose(
            float(closest.get("best_threshold", np.nan)),
            BROAD_THRESHOLD,
            rtol=0.0,
            atol=1.0e-15,
        )
    ):
        raise ValueError("NEXT169 closest-candidate provenance differs")
    manifest168 = json.loads(paths["next168_manifest"].read_text())
    if (
        manifest168.get("protocol") != n168.PROTOCOL
        or manifest168.get("labels_or_endpoints_opened") is not False
        or manifest168.get("validation_geometry_opened") is not False
        or manifest168.get("replication_geometry_opened") is not False
        or manifest168.get("dft_calculation_executed") is not False
        or manifest168.get("dft_values_used_by_features") is not False
        or manifest168.get("learned_energy_force_stress_proxy_used") is not False
        or manifest168.get("physical_relaxation_executed") is not False
        or manifest168.get("outputs_sha256", {}).get(n168.CATALOGUE_NAME)
        != input_hashes["next168_catalogue"]
        or manifest168.get("outputs_sha256", {}).get(n168.SCIGEN_NAME)
        != input_hashes["next168_scigen_features"]
        or manifest168.get("outputs_sha256", {}).get(n168.WYFORMER_NAME)
        != input_hashes["next168_wyformer_features"]
        or manifest168.get("executed_source_sha256", {}).get(
            "src/next168_periodic_local_directional_rigidity.py"
        )
        != _sha256_file(Path(n168.__file__).resolve())
    ):
        raise ValueError("NEXT169 directional-rigidity provenance differs")

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
    conjunctive = n135.materialize_conjunctive_features(extended)
    extended = pd.concat(
        [extended.reset_index(drop=True), conjunctive.reset_index(drop=True)], axis=1
    )
    rigidity_frames = []
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(paths[f"next168_{source}_features"]).copy()
        table["material_id"] = source + ":" + table["material_id"].astype(str)
        rigidity_frames.append(table)
    extended = extended.merge(
        pd.concat(rigidity_frames, ignore_index=True),
        on="material_id",
        how="inner",
        validate="one_to_one",
    )

    physical_terms = [*old_terms, *mhcr_terms]
    physical_ids = {str(term["term_id"]) for term in physical_terms}
    all_bases = n130.n127.select_next125_bases(
        pd.read_parquet(paths["next125_search_records"])
    )
    bases = n135.n132.select_extended_bases(
        pd.read_parquet(paths["next130_search_records"]), all_bases
    )
    specs = n163.build_candidate_specs(bases=bases, physical_term_ids=physical_ids)
    selected_specs = [
        spec for spec in specs if str(spec["candidate_key"]) == candidate_key
    ]
    if len(selected_specs) != 1:
        raise ValueError("NEXT169 candidate reconstruction differs")

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
                    "_endpoint": n130.n125.n121.prior._endpoint_numeric(
                        wyformer_endpoint["endpoint_stratum"]
                    ),
                }
            ),
        ],
        ignore_index=True,
    )
    combined = extended.merge(
        endpoint_frame, on="material_id", how="inner", validate="one_to_one"
    )
    endpoint = pd.to_numeric(combined.pop("_endpoint"), errors="coerce").to_numpy(float)
    if len(combined) != len(extended) or not np.isfinite(endpoint).all():
        raise ValueError("NEXT169 endpoint row accounting differs")
    combined, virtual_terms, runtime = n163.materialize_candidates(
        features=combined,
        physical_terms=physical_terms,
        specs=selected_specs,
    )
    if len(virtual_terms) != 1 or len(runtime) != 1:
        raise RuntimeError("NEXT169 score materialization differs")
    published_score, published_support = _term_risk(combined, virtual_terms[0])

    sources = combined["source_dataset"].astype(str).to_numpy()
    folds = assign_group_folds(combined["reduced_formula"].astype(str).to_numpy())
    extremes = published_support & ((endpoint <= 1.0) | (endpoint >= 2.0))
    shell = extremes & (published_score >= BROAD_THRESHOLD) & (published_score < SAFE_THRESHOLD)
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

    records = []
    for hypothesis in sorted(HYPOTHESES):
        column, direction = HYPOTHESES[hypothesis]
        values = pd.to_numeric(combined[column], errors="coerce").to_numpy(float)
        evaluations = {}
        for population, mask in populations.items():
            result = n151._evaluate_auc(
                values=values[mask],
                protected=endpoint[mask] <= 1.0,
                folds=folds[mask],
                direction=direction,
            )
            if result is None:
                raise ValueError("NEXT169 AUC population differs")
            evaluations[population] = result
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
            "feature": column,
            "direction": direction,
            "eligible_for_search": eligible,
            "scigen_full_support": support["scigen"],
            "wyformer_full_support": support["wyformer"],
            "ranking_min_auc": float(min(key_aucs)),
            "ranking_mean_auc": float(np.mean(key_aucs)),
        }
        for population, result in evaluations.items():
            for metric in (
                "pooled_auc",
                "macro_auc",
                "worst_auc",
                "evaluable_folds",
                "protected",
                "severe",
            ):
                record[f"{population}_{metric}"] = result[metric]
            record[f"{population}_fold_aucs_json"] = json.dumps(
                result["fold_aucs"], separators=(",", ":")
            )
        records.append(record)

    audit_table, selected = select_directional_rigidity_hypothesis(
        pd.DataFrame(records)
    )
    eligible_names = audit_table.loc[
        audit_table["eligible_for_search"].fillna(False).astype(bool), "hypothesis"
    ].astype(str).tolist()
    audit = {
        "protocol": PROTOCOL,
        "candidate_key_sha256": EXPECTED_CANDIDATE_KEY_SHA256,
        "safe_threshold": SAFE_THRESHOLD,
        "broad_threshold": BROAD_THRESHOLD,
        "population_counts": population_counts,
        "hypotheses": {
            name: {"feature": value[0], "direction": value[1]}
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
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_paths = {
        "src/next168_periodic_local_directional_rigidity.py": Path(n168.__file__).resolve(),
        "src/next169_periodic_local_directional_rigidity_audit.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    try:
        audit_path = staging / AUDIT_NAME
        table_path = staging / TABLE_NAME
        _write_json(audit_path, audit)
        audit_table.to_parquet(table_path, index=False)
        terminated = selected is None
        manifest = {
            "protocol": PROTOCOL,
            "eligible_hypothesis_count": len(eligible_names),
            "periodic_local_directional_rigidity_branch_terminated": terminated,
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
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT169 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name] for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT169 source changed before publication")
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
    )
    for stage in stages:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_periodic_local_directional_rigidity_audit(
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
    "HYPOTHESES",
    "eligibility_from_metrics",
    "run_periodic_local_directional_rigidity_audit",
    "select_directional_rigidity_hypothesis",
]


if __name__ == "__main__":
    raise SystemExit(main())
