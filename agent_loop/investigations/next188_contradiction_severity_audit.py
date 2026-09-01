#!/usr/bin/env python3
"""Frozen audit of local/nonlocal contradiction as a severity signal."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

import numpy as np
import pandas as pd

import src.next186_local_nonlocal_contradiction_relief_audit as n186
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-08-next188-contradiction-severity-audit-v1"
MANIFEST_NAME = "MANIFEST.json"
AUDIT_NAME = "NEXT188_CONTRADICTION_SEVERITY_AUDIT.json"
TABLE_NAME = "next188_contradiction_severity_audit.parquet"
EXPECTED_DESIGN_SHA256 = (
    "ff64051bafef71ce4b90220d0e1be462ff6d52424d9ecffb4b113ed1de8805a9"
)
EXPECTED_INPUT_SHA256 = {
    **n186.EXPECTED_INPUT_SHA256,
    "design": EXPECTED_DESIGN_SHA256,
    "next186_manifest": "ae3ce43ec76ae7763efce3bd632dca3c60e50fac5279586313c89badeebac99e",
    "next186_audit": "3a083455d128535f8c779c0720ff5144379de4cc0640d3497157ec10a6534344",
    "next186_table": "3999d9dc8674fe1e0be0efa05f0fb3b75b3b5bd17a334f0dad7a73d37524c94a",
}
POPULATIONS = ("scigen_shell", "wyformer_shell", "scigen_full", "wyformer_full")
HYPOTHESES = {
    f"{feature}__{surplus_name}__{conjunction}__severe_high": (
        feature,
        surplus_name,
        conjunction,
        -1,
    )
    for feature in n186.CLOSURE_FEATURES
    for surplus_name in n186.SURPLUS_NAMES
    for conjunction in n186.CONJUNCTIONS
}


def reverse_auc_evaluation(raw: Mapping[str, object]) -> dict[str, object]:
    """Reverse a fixed-sample AUC evaluation without reopening endpoints."""

    required = {
        "pooled_auc",
        "macro_auc",
        "worst_auc",
        "evaluable_folds",
        "fold_aucs_json",
        "protected",
        "severe",
    }
    if set(raw) != required:
        raise ValueError("NEXT188 AUC evaluation schema differs")
    pooled = float(raw["pooled_auc"])
    parsed = json.loads(str(raw["fold_aucs_json"]))
    if not isinstance(parsed, list):
        raise ValueError("NEXT188 fold AUC schema differs")
    folds = [
        1.0 - float(value)
        for value in parsed
        if value is not None and np.isfinite(float(value))
    ]
    if (
        not np.isfinite(pooled)
        or not folds
        or len(folds) != int(raw["evaluable_folds"])
        or any(value < -1.0e-12 or value > 1.0 + 1.0e-12 for value in folds)
    ):
        raise ValueError("NEXT188 AUC evaluation is not reversible")
    return {
        "pooled_auc": 1.0 - pooled,
        "macro_auc": float(np.mean(folds)),
        "worst_auc": float(np.min(folds)),
        "evaluable_folds": len(folds),
        "fold_aucs": folds,
        "protected": int(raw["protected"]),
        "severe": int(raw["severe"]),
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
    """Apply the unchanged frozen cross-source audit gates."""

    return n186.eligibility_from_metrics(
        scigen_full_support=scigen_full_support,
        wyformer_full_support=wyformer_full_support,
        scigen_shell_worst_auc=scigen_shell_worst_auc,
        scigen_shell_evaluable_folds=scigen_shell_evaluable_folds,
        wyformer_shell_pooled_auc=wyformer_shell_pooled_auc,
        scigen_full_pooled_auc=scigen_full_pooled_auc,
        wyformer_full_pooled_auc=wyformer_full_pooled_auc,
    )


def select_contradiction_severity_hypothesis(
    records: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    """Rank eligible severity hypotheses with deterministic tie breaking."""

    required = {
        "hypothesis",
        "eligible_for_search",
        "ranking_min_auc",
        "ranking_mean_auc",
    }
    if required - set(records.columns) or records["hypothesis"].astype(str).duplicated().any():
        raise ValueError("NEXT188 audit record schema differs")
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
    paths = n186._paths(roots, freeze_path, design_path)
    paths.update(
        {
            "next186_manifest": roots["next186"] / n186.MANIFEST_NAME,
            "next186_audit": roots["next186"] / n186.AUDIT_NAME,
            "next186_table": roots["next186"] / n186.TABLE_NAME,
        }
    )
    return paths


def run_contradiction_severity_audit(
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
    next135_freeze_path: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run the frozen discovery-only NEXT188 audit and publish atomically."""

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
        raise FileNotFoundError("NEXT188 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT188 formal input identity differs: {differing}")

    manifest186 = json.loads(paths["next186_manifest"].read_text())
    audit186 = json.loads(paths["next186_audit"].read_text())
    expected_outputs186 = {
        n186.AUDIT_NAME: input_hashes["next186_audit"],
        n186.TABLE_NAME: input_hashes["next186_table"],
    }
    if (
        manifest186.get("protocol") != n186.PROTOCOL
        or manifest186.get("hypothesis_count") != len(n186.HYPOTHESES)
        or manifest186.get("eligible_hypothesis_count") != 0
        or manifest186.get("local_nonlocal_contradiction_relief_branch_terminated") is not True
        or manifest186.get("new_formula_searched") is not False
        or manifest186.get("opened_validation_outputs_used") is not False
        or manifest186.get("scigen_replication_endpoint_opened") is not False
        or manifest186.get("wyformer_replication_endpoint_opened") is not False
        or manifest186.get("dft_calculation_executed") is not False
        or manifest186.get("dft_values_used_by_executable_formula") is not False
        or manifest186.get("learned_energy_force_stress_proxy_used") is not False
        or manifest186.get("physical_relaxation_executed") is not False
        or manifest186.get("outputs_sha256") != expected_outputs186
        or manifest186.get("executed_source_sha256", {}).get(
            "src/next186_local_nonlocal_contradiction_relief_audit.py"
        )
        != _sha256_file(Path(n186.__file__).resolve())
    ):
        raise ValueError("NEXT188 NEXT186 provenance differs")
    if (
        audit186.get("protocol") != n186.PROTOCOL
        or audit186.get("eligible_hypotheses") != []
        or audit186.get("selected_hypothesis") is not None
        or audit186.get("new_formula_searched") is not False
        or audit186.get("validation_or_replication_opened") is not False
    ):
        raise ValueError("NEXT188 NEXT186 audit boundary differs")

    relief = pd.read_parquet(paths["next186_table"])
    required_columns = {
        "hypothesis",
        "closure_feature",
        "surplus_name",
        "conjunction",
        "direction",
        "scigen_full_support",
        "wyformer_full_support",
        *{
            f"{population}_{suffix}"
            for population in POPULATIONS
            for suffix in (
                "pooled_auc",
                "macro_auc",
                "worst_auc",
                "evaluable_folds",
                "protected",
                "severe",
                "fold_aucs_json",
            )
        },
    }
    if (
        required_columns - set(relief.columns)
        or len(relief) != len(n186.HYPOTHESES)
        or set(relief["hypothesis"].astype(str)) != set(n186.HYPOTHESES)
        or set(pd.to_numeric(relief["direction"], errors="coerce")) != {1}
    ):
        raise ValueError("NEXT188 NEXT186 audit table differs")

    records: list[dict[str, object]] = []
    for severe_name in sorted(HYPOTHESES):
        feature, surplus_name, conjunction, direction = HYPOTHESES[severe_name]
        relief_name = f"{feature}__{surplus_name}__{conjunction}__high"
        row = relief.loc[relief["hypothesis"].astype(str) == relief_name]
        if len(row) != 1:
            raise ValueError("NEXT188 hypothesis provenance differs")
        raw_row = row.iloc[0]
        if (
            str(raw_row["closure_feature"]) != feature
            or str(raw_row["surplus_name"]) != surplus_name
            or str(raw_row["conjunction"]) != conjunction
        ):
            raise ValueError("NEXT188 hypothesis metadata differs")
        evaluations: dict[str, dict[str, object]] = {}
        for population in POPULATIONS:
            evaluations[population] = reverse_auc_evaluation(
                {
                    suffix: raw_row[f"{population}_{suffix}"]
                    for suffix in (
                        "pooled_auc",
                        "macro_auc",
                        "worst_auc",
                        "evaluable_folds",
                        "fold_aucs_json",
                        "protected",
                        "severe",
                    )
                }
            )
        key_aucs = [
            float(evaluations["scigen_shell"]["worst_auc"]),
            float(evaluations["wyformer_shell"]["pooled_auc"]),
            float(evaluations["scigen_full"]["pooled_auc"]),
            float(evaluations["wyformer_full"]["pooled_auc"]),
        ]
        eligible = eligibility_from_metrics(
            scigen_full_support=float(raw_row["scigen_full_support"]),
            wyformer_full_support=float(raw_row["wyformer_full_support"]),
            scigen_shell_worst_auc=key_aucs[0],
            scigen_shell_evaluable_folds=int(
                evaluations["scigen_shell"]["evaluable_folds"]
            ),
            wyformer_shell_pooled_auc=key_aucs[1],
            scigen_full_pooled_auc=key_aucs[2],
            wyformer_full_pooled_auc=key_aucs[3],
        )
        record: dict[str, object] = {
            "hypothesis": severe_name,
            "source_relief_hypothesis": relief_name,
            "closure_feature": feature,
            "surplus_name": surplus_name,
            "conjunction": conjunction,
            "direction": direction,
            "eligible_for_search": eligible,
            "scigen_full_support": float(raw_row["scigen_full_support"]),
            "wyformer_full_support": float(raw_row["wyformer_full_support"]),
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

    audit_table, selected = select_contradiction_severity_hypothesis(
        pd.DataFrame(records)
    )
    eligible_names = audit_table.loc[
        audit_table["eligible_for_search"].fillna(False).astype(bool), "hypothesis"
    ].astype(str).tolist()
    audit = {
        "protocol": PROTOCOL,
        "design_sha256": input_hashes["design"],
        "candidate_key_sha256": n186.EXPECTED_BASE_CANDIDATE_KEY_SHA256,
        "safe_threshold": n186.SAFE_THRESHOLD,
        "broad_threshold": n186.BROAD_THRESHOLD,
        "population_counts": audit186.get("population_counts"),
        "hypothesis_values_reused_from_next186": True,
        "direction_reversal_identity": "AUC(-x)=1-AUC(x) on every fixed evaluable population/fold",
        "hypotheses": {
            name: {
                "closure_feature": value[0],
                "surplus_name": value[1],
                "conjunction": value[2],
                "direction": value[3],
            }
            for name, value in HYPOTHESES.items()
        },
        "eligibility_gates": audit186.get("eligibility_gates"),
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
        "src/next186_local_nonlocal_contradiction_relief_audit.py": Path(
            n186.__file__
        ).resolve(),
        "src/next188_contradiction_severity_audit.py": Path(__file__).resolve(),
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
            "contradiction_severity_branch_terminated": selected is None,
            "next189_search_authorized": selected is not None,
            "new_formula_searched": False,
            "discovery_outcomes_used_as_offline_labels": True,
            "discovery_endpoints_reopened": False,
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
            raise RuntimeError("NEXT188 input changed before publication")
        if any(
            _sha256_file(path) != source_hashes[name]
            for name, path in source_paths.items()
        ):
            raise RuntimeError("NEXT188 source changed before publication")
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
        186,
    )
    for stage in stages:
        parser.add_argument(f"--next{stage}-dir", type=Path, required=True)
    parser.add_argument("--next135-freeze-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = run_contradiction_severity_audit(
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
    "reverse_auc_evaluation",
    "run_contradiction_severity_audit",
    "select_contradiction_severity_hypothesis",
]


if __name__ == "__main__":
    raise SystemExit(main())
