#!/usr/bin/env python3
"""Audit latent-symmetry recoverability in the frozen repair shell."""

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
import src.next188_contradiction_severity_audit as n188
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json
from src.next87_scigen_sparse_law_search import _term_risk


PROTOCOL = "2026-08-08-next190-latent-symmetry-recoverability-audit-v1"
MANIFEST_NAME = "MANIFEST.json"
AUDIT_NAME = "NEXT190_LATENT_SYMMETRY_RECOVERABILITY_AUDIT.json"
TABLE_NAME = "next190_latent_symmetry_recoverability_audit.parquet"
EXPECTED_DESIGN_SHA256 = (
    "7b73091ddd559d7d2fe5d33861ae81aeb7e8cd1f4ad20d277bfeef3cd37792fa"
)
EXPECTED_INPUT_SHA256 = {
    **n188.EXPECTED_INPUT_SHA256,
    "design": EXPECTED_DESIGN_SHA256,
    "next188_manifest": "0e715eeac538046e714e0fcd76b398c650ac4910a7de6e6f771fd02a403a1ff3",
    "next188_audit": "e505b41027a7ef0dfb3a93974282e10087408d38e6fe7322e6c315e10edd4557",
    "next188_table": "76f9fcab3114706c102ff9100b740406932fe5526bbb2a2be6e7b7894e0162c6",
}
FEATURES = (
    "sym_recovery_onset_rel",
    "sym_recovery_gain_log2",
    "sym_orbit_collapse",
    "sym_recovery_residual_rms_rel",
    "sym_recovery_residual_q95_rel",
    "sym_recovery_residual_max_rel",
)
HYPOTHESES = {
    f"{feature}__recoverable_high": (feature, 1) for feature in FEATURES
}
NORMALIZATION_SCALES = {
    "sym_recovery_onset_rel": 0.12,
    "sym_recovery_gain_log2": float(np.log2(48.0)),
    "sym_orbit_collapse": 1.0,
    "sym_recovery_residual_rms_rel": 0.24,
    "sym_recovery_residual_q95_rel": 0.24,
    "sym_recovery_residual_max_rel": 0.24,
}
POPULATIONS = ("scigen_shell", "wyformer_shell", "scigen_full", "wyformer_full")


def normalize_recoverability(*, feature: str, values: object) -> np.ndarray:
    """Apply the frozen physical normalization while preserving missingness."""

    if feature not in NORMALIZATION_SCALES:
        raise ValueError("NEXT190 recoverability feature differs")
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("NEXT190 recoverability value schema differs")
    result = np.full(array.shape, np.nan, dtype=float)
    finite = np.isfinite(array)
    result[finite] = np.clip(
        array[finite] / NORMALIZATION_SCALES[feature], 0.0, 1.0
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
    """Apply the unchanged frozen cross-source audit gates."""

    return n188.eligibility_from_metrics(
        scigen_full_support=scigen_full_support,
        wyformer_full_support=wyformer_full_support,
        scigen_shell_worst_auc=scigen_shell_worst_auc,
        scigen_shell_evaluable_folds=scigen_shell_evaluable_folds,
        wyformer_shell_pooled_auc=wyformer_shell_pooled_auc,
        scigen_full_pooled_auc=scigen_full_pooled_auc,
        wyformer_full_pooled_auc=wyformer_full_pooled_auc,
    )


def select_recoverability_hypothesis(
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
        raise ValueError("NEXT190 audit record schema differs")
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
    paths = n188._paths(roots, freeze_path, design_path)
    paths.update(
        {
            "next188_manifest": roots["next188"] / n188.MANIFEST_NAME,
            "next188_audit": roots["next188"] / n188.AUDIT_NAME,
            "next188_table": roots["next188"] / n188.TABLE_NAME,
        }
    )
    return paths


def run_latent_symmetry_recoverability_audit(
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
    next135_freeze_path: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only NEXT190 audit and publish atomically."""

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
                (186, next186_dir),
                (188, next188_dir),
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
        raise FileNotFoundError("NEXT190 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT190 formal input identity differs: {differing}")

    manifest188 = json.loads(paths["next188_manifest"].read_text())
    audit188 = json.loads(paths["next188_audit"].read_text())
    expected_outputs188 = {
        n188.AUDIT_NAME: input_hashes["next188_audit"],
        n188.TABLE_NAME: input_hashes["next188_table"],
    }
    if (
        manifest188.get("protocol") != n188.PROTOCOL
        or manifest188.get("hypothesis_count") != len(n188.HYPOTHESES)
        or manifest188.get("eligible_hypothesis_count") != 0
        or manifest188.get("contradiction_severity_branch_terminated") is not True
        or manifest188.get("next189_search_authorized") is not False
        or manifest188.get("new_formula_searched") is not False
        or manifest188.get("discovery_endpoints_reopened") is not False
        or manifest188.get("opened_validation_outputs_used") is not False
        or manifest188.get("scigen_replication_endpoint_opened") is not False
        or manifest188.get("wyformer_replication_endpoint_opened") is not False
        or manifest188.get("dft_calculation_executed") is not False
        or manifest188.get("dft_values_used_by_executable_formula") is not False
        or manifest188.get("learned_energy_force_stress_proxy_used") is not False
        or manifest188.get("physical_relaxation_executed") is not False
        or manifest188.get("outputs_sha256") != expected_outputs188
        or manifest188.get("executed_source_sha256", {}).get(
            "src/next188_contradiction_severity_audit.py"
        )
        != _sha256_file(Path(n188.__file__).resolve())
    ):
        raise ValueError("NEXT190 NEXT188 provenance differs")
    if (
        audit188.get("protocol") != n188.PROTOCOL
        or audit188.get("eligible_hypotheses") != []
        or audit188.get("selected_hypothesis") is not None
        or audit188.get("new_formula_searched") is not False
        or audit188.get("validation_or_replication_opened") is not False
    ):
        raise ValueError("NEXT190 NEXT188 audit boundary differs")

    diagnostic164 = json.loads(paths["next164_diagnostic"].read_text())
    base_key = str(diagnostic164.get("global_closest", {}).get("candidate_key", ""))
    if hashlib.sha256(base_key.encode()).hexdigest() != n186_candidate_key_sha256():
        raise ValueError("NEXT190 base candidate identity differs")

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
    if set(FEATURES) - set(extended.columns):
        raise ValueError("NEXT190 frozen symmetry feature schema differs")

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
        raise ValueError("NEXT190 base reconstruction differs")
    extended, base_terms, runtime = n163.materialize_candidates(
        features=extended,
        physical_terms=physical_terms,
        specs=selected_specs,
    )
    if len(base_terms) != 1 or len(runtime) != 1:
        raise RuntimeError("NEXT190 base materialization differs")
    base_score, base_support = _term_risk(extended, base_terms[0])

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
    extended = extended.merge(
        endpoint_frame, on="material_id", how="inner", validate="one_to_one"
    )
    endpoint = pd.to_numeric(extended.pop("_endpoint"), errors="coerce").to_numpy(
        float
    )
    if len(extended) != len(base_score) or not np.isfinite(endpoint).all():
        raise ValueError("NEXT190 endpoint row accounting differs")

    sources = extended["source_dataset"].astype(str).to_numpy()
    folds = n164.assign_group_folds(extended["reduced_formula"].astype(str).to_numpy())
    extremes = base_support & ((endpoint <= 1.0) | (endpoint >= 2.0))
    shell = (
        extremes
        & (base_score >= n188.n186.BROAD_THRESHOLD)
        & (base_score < n188.n186.SAFE_THRESHOLD)
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
        feature, direction = HYPOTHESES[hypothesis]
        values = pd.to_numeric(extended[feature], errors="coerce").to_numpy(float)
        normalized = normalize_recoverability(feature=feature, values=values)
        if not np.array_equal(np.isfinite(values), np.isfinite(normalized)):
            raise ValueError("NEXT190 normalized support differs")
        evaluations = {}
        for population, mask in populations.items():
            evaluation = n151._evaluate_auc(
                values=values[mask],
                protected=endpoint[mask] <= 1.0,
                folds=folds[mask],
                direction=direction,
            )
            if evaluation is None:
                raise ValueError("NEXT190 AUC population differs")
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
            "normalization_scale": NORMALIZATION_SCALES[feature],
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

    audit_table, selected = select_recoverability_hypothesis(pd.DataFrame(records))
    eligible_names = audit_table.loc[
        audit_table["eligible_for_search"].fillna(False).astype(bool), "hypothesis"
    ].astype(str).tolist()
    audit = {
        "protocol": PROTOCOL,
        "design_sha256": input_hashes["design"],
        "candidate_key_sha256": n186_candidate_key_sha256(),
        "safe_threshold": n188.n186.SAFE_THRESHOLD,
        "broad_threshold": n188.n186.BROAD_THRESHOLD,
        "hypothesis_direction_fixed_from_independent_omat_result": True,
        "symmetry_features_frozen_before_current_endpoints": True,
        "normalization_scales": NORMALIZATION_SCALES,
        "population_counts": population_counts,
        "hypotheses": {
            name: {"feature": value[0], "direction": value[1]}
            for name, value in HYPOTHESES.items()
        },
        "eligibility_gates": audit188.get("eligibility_gates"),
        "eligible_hypotheses": eligible_names,
        "selected_hypothesis": selected,
        "new_formula_searched": False,
        "validation_or_replication_opened": False,
        "dft_calculation_executed": False,
        "dft_values_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False,
        "physical_relaxation_executed": False,
        "generator_protocol_signal_risk": True,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    source_paths = {
        "src/next188_contradiction_severity_audit.py": Path(n188.__file__).resolve(),
        "src/next190_latent_symmetry_recoverability_audit.py": Path(__file__).resolve(),
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
            "latent_symmetry_recoverability_branch_terminated": selected is None,
            "next191_search_authorized": selected is not None,
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
            raise RuntimeError("NEXT190 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT190 source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def n186_candidate_key_sha256() -> str:
    """Return the frozen base identity through the audited provenance chain."""

    return n188.n186.EXPECTED_BASE_CANDIDATE_KEY_SHA256


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
        186,
        188,
    )
    for stage in stages:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_latent_symmetry_recoverability_audit(
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
    "FEATURES",
    "HYPOTHESES",
    "NORMALIZATION_SCALES",
    "eligibility_from_metrics",
    "normalize_recoverability",
    "run_latent_symmetry_recoverability_audit",
    "select_recoverability_hypothesis",
]


if __name__ == "__main__":
    raise SystemExit(main())
