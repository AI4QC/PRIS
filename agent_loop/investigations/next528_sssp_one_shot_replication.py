#!/usr/bin/env python3
"""Sequential dual-source replication of the frozen standalone SSSP law."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

import src.next411_same_sign_shell_purity as n411
import src.next525_sssp_standalone_freeze as n525
import src.next526_sssp_holdout_feature_freeze as n526
import src.next527_sssp_one_shot_validation as n527
from src.next95_wyformer_sparse_law_search import _endpoint_numeric
from src.next347_periodic_allocation_redistribution_capacity import _sha256_file


PROTOCOL = "2026-08-13-next528-sssp-one-shot-dual-source-replication-v1"
VALIDATION_PROTOCOL = n527.PROTOCOL
ROLE = "internal_replication"
GATES = n527.GATES
BOOTSTRAP_DRAWS = n527.BOOTSTRAP_DRAWS
BOOTSTRAP_SEED = n527.BOOTSTRAP_SEED
MANIFEST_NAME = "MANIFEST.json"
EVALUATION_NAME = "NEXT528_SSSP_INTERNAL_REPLICATION.json"
PREDICTION_NAMES = {
    source: f"next528_{source}_sssp_internal_replication_predictions.parquet"
    for source in ("scigen", "wyformer")
}
EXPECTED_INPUT_SHA256 = {
    "design": n525.DESIGN_SHA256,
    "next525_manifest": "e15217dafaa1d86dc5a70640dd0ab96a99a9cc0bb04eff44ca850c88e4ff3140",
    "next525_formula": "e98f7cf1bf6d0947b653c133100495650a57265dddde46ce8e2c4dd9521e09cf",
    "next526_manifest": "2f9c196a04312679223bf1658670ba999fa640a1e8b84c841745a22261f9662b",
    "next526_catalogue": "7f63cd81f839c76b60e339ecc94a12004c7c0785fd02a5744bba4c838ee4e502",
    "next526_scigen": "cc2d10407f38e2f14f2ff62eadc9ef13248d009c16b1098f803c23d00c3c096b",
    "next526_wyformer": "85c7b94a653ab9fa243151118eb49a35d7f923bd1ca95ffb19b4abca40b48af4",
    "next527_source": "659602b1db36fcda4095a5a2f27d6a2d3f236f00423773ebfc73ed3147fe1185",
    "next527_manifest": "f5c48c14b55713ad385d393523319b83ca29ef6079a689553b6d1fb4dd67fe32",
    "next527_evaluation": "33fd648cca4bb1e63c030d7c3ad5d7aae49e5c406b82e9cd7da51884875ae971",
    "scigen_feature_manifest": "8dcb8118f85ee4a3acbf0905f01c2b173d58742a1e16dcd6004adbbbedcf63cc",
    "scigen_features": "2d420ac76f8b9e1ea6a7908df92a4db1198bc0ef0b2d410875225d51536214b2",
    "wyformer_feature_manifest": "fb66f7c5caade419a46b9a3fa6fef1bc5b3afa3eebeb95a4bc53baddabc0f659",
    "wyformer_features": "7e52e8ab32b380882082ee9a9315c3d18b4d22fe100a83766060b86e50ff19d9",
    "scigen_endpoint_manifest": "ce97c4594efc0951f2c4da713dc3c6e9f2d9f5c25eab5974d3284f4c2f61e2df",
    "scigen_endpoint": "f2dd83b9bc9f65c95d188ee679ed65a62614c0ef5875e0b9e5a10d03933e699c",
    "wyformer_endpoint_manifest": "91bd25c04a6e06ee35f8c76d081330cef22ad60c2cc8d91cb9afd8143e5444ba",
    "wyformer_endpoint": "e6ec7632d971646ba307d7ac5893be8d4af4e3b792d446e7fa20af9ab8cc0aae",
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def validate_replication_authorization(
    validation_manifest: dict[str, object], validation_evaluation: dict[str, object]
) -> None:
    if (
        validation_manifest.get("protocol") != VALIDATION_PROTOCOL
        or validation_manifest.get("passes_all_validation_gates") is not True
        or validation_manifest.get("next528_internal_replication_authorized") is not True
        or validation_manifest.get("internal_validation_endpoint_values_opened") is not True
        or validation_manifest.get("internal_replication_endpoint_values_opened") is not False
        or validation_manifest.get("formula_or_threshold_modified") is not False
        or validation_evaluation.get("protocol") != VALIDATION_PROTOCOL
        or validation_evaluation.get("partition_role") != "internal_validation"
        or validation_evaluation.get("passes_all_validation_gates") is not True
        or validation_evaluation.get("scientific_status") != "advance_to_internal_replication"
        or validation_evaluation.get("formula_or_threshold_modified") is not False
    ):
        raise PermissionError("NEXT528 replication is not authorized")


def _join_source(
    *, source: str, sssp: pd.DataFrame, base: pd.DataFrame, endpoint: pd.DataFrame
) -> pd.DataFrame:
    endpoint_column = "distortion_ratio" if source == "scigen" else "endpoint_stratum"
    required = {
        "material_id", "reduced_formula", "partition_role",
        n411.FEATURE_NAMES[0], "sssp_supported",
    }
    if (
        source not in {"scigen", "wyformer"}
        or required - set(sssp)
        or {"material_id", "pauling_p2_p5_decision"} - set(base)
        or {"material_id", endpoint_column} - set(endpoint)
        or set(sssp["partition_role"].astype(str)) != {ROLE}
        or any(frame["material_id"].astype(str).duplicated().any() for frame in (sssp, base, endpoint))
    ):
        raise ValueError(f"NEXT528 {source} input schema differs")
    joined = (
        sssp.merge(
            base[["material_id", "pauling_p2_p5_decision"]],
            on="material_id", validate="one_to_one",
        ).merge(
            endpoint[["material_id", endpoint_column]],
            on="material_id", validate="one_to_one",
        )
    )
    if len(joined) != len(sssp) or len(joined) != len(base) or len(joined) != len(endpoint):
        raise ValueError(f"NEXT528 {source} material identity differs")
    joined["endpoint"] = (
        pd.to_numeric(joined[endpoint_column], errors="coerce").to_numpy(float)
        if source == "scigen" else _endpoint_numeric(joined[endpoint_column])
    )
    return joined


def run_one_shot_replication(
    *,
    next525_dir: Path,
    next526_dir: Path,
    next527_dir: Path,
    scigen_feature_dir: Path,
    wyformer_feature_dir: Path,
    scigen_endpoint_dir: Path,
    wyformer_endpoint_dir: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Open replication only after validating the complete NEXT527 pass."""

    freeze, holdout, validation = (
        Path(next525_dir).resolve(), Path(next526_dir).resolve(), Path(next527_dir).resolve()
    )
    sf, wf = Path(scigen_feature_dir).resolve(), Path(wyformer_feature_dir).resolve()
    se, we = Path(scigen_endpoint_dir).resolve(), Path(wyformer_endpoint_dir).resolve()
    paths = {
        "design": Path(design_path).resolve(),
        "next525_manifest": freeze / n525.MANIFEST_NAME,
        "next525_formula": freeze / n525.FORMULA_NAME,
        "next526_manifest": holdout / n526.MANIFEST_NAME,
        "next526_catalogue": holdout / n526.CATALOGUE_NAME,
        "next526_scigen": holdout / n526.FEATURE_FILES["scigen"][ROLE],
        "next526_wyformer": holdout / n526.FEATURE_FILES["wyformer"][ROLE],
        "next527_source": Path(n527.__file__).resolve(),
        "next527_manifest": validation / n527.MANIFEST_NAME,
        "next527_evaluation": validation / n527.EVALUATION_NAME,
        "scigen_feature_manifest": sf / "MANIFEST.json",
        "scigen_features": sf / "features_internal_replication.parquet",
        "wyformer_feature_manifest": wf / "MANIFEST.json",
        "wyformer_features": wf / "wyformer_x0_features_internal_replication.parquet",
        "scigen_endpoint_manifest": se / "MANIFEST.json",
        "scigen_endpoint": se / "scigen_dft_distortion_endpoints.parquet",
        "wyformer_endpoint_manifest": we / "MANIFEST.json",
        "wyformer_endpoint": we / "wyformer_dft_screening_endpoints.parquet",
    }
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT528 input is missing")
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name for name in set(hashes) | set(EXPECTED_INPUT_SHA256)
            if hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT528 formal input identity differs: {differing}")
    validation_manifest = _read_json(paths["next527_manifest"])
    validation_evaluation = _read_json(paths["next527_evaluation"])
    validate_replication_authorization(validation_manifest, validation_evaluation)
    next526_manifest = _read_json(paths["next526_manifest"])
    formula = _read_json(paths["next525_formula"])
    endpoint_manifests = {
        "scigen": _read_json(paths["scigen_endpoint_manifest"]),
        "wyformer": _read_json(paths["wyformer_endpoint_manifest"]),
    }
    if (
        next526_manifest.get("protocol") != n526.PROTOCOL
        or next526_manifest.get("replication_endpoint_opened") is not False
        or formula.get("protocol") != n525.PROTOCOL
        or float(formula.get("threshold", math.nan)) != n525.EXPECTED_THRESHOLD
        or formula.get("dft_inputs") != []
        or formula.get("learned_model_inputs") != []
        or formula.get("relaxation_inputs") != []
        or endpoint_manifests["scigen"].get("partition_role") != ROLE
        or endpoint_manifests["scigen"].get("endpoint_values_summarized_or_inspected") is not False
        or endpoint_manifests["wyformer"].get("partition_role") != ROLE
        or endpoint_manifests["wyformer"].get("endpoint_payload_opened") is not False
    ):
        raise ValueError("NEXT528 frozen provenance differs")
    # Replication endpoint deserialization begins only after every authorization check.
    joined = {
        "scigen": _join_source(
            source="scigen",
            sssp=pd.read_parquet(paths["next526_scigen"]),
            base=pd.read_parquet(paths["scigen_features"]),
            endpoint=pd.read_parquet(paths["scigen_endpoint"]),
        ),
        "wyformer": _join_source(
            source="wyformer",
            sssp=pd.read_parquet(paths["next526_wyformer"]),
            base=pd.read_parquet(paths["wyformer_features"]),
            endpoint=pd.read_parquet(paths["wyformer_endpoint"]),
        ),
    }
    results = {
        source: n527.evaluate_sssp_source(
            frame=frame, threshold=float(formula["threshold"]),
            bootstrap_draws=BOOTSTRAP_DRAWS,
        )
        for source, frame in joined.items()
    }
    passes = all(bool(result["passes_source_gates"]) for result in results.values())
    evaluation = {
        "protocol": PROTOCOL,
        "partition_role": ROLE,
        "gates": GATES,
        "minimum_fold_class_count": n527.MINIMUM_FOLD_CLASS_COUNT,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "formula_or_threshold_modified": False,
        "validation_pass_sha256": hashes["next527_evaluation"],
        "sources": {
            source: {
                key: value for key, value in result.items()
                if key not in {"score", "supported", "reject"}
            }
            for source, result in results.items()
        },
        "passes_all_replication_gates": passes,
        "scientific_status": (
            "replicated_standalone_zero_dft_prescreen_candidate" if passes
            else "internal_replication_failure"
        ),
    }
    predictions = {}
    for source, frame in joined.items():
        result = results[source]
        predictions[source] = pd.DataFrame(
            {
                "material_id": frame["material_id"].astype(str),
                "reduced_formula": frame["reduced_formula"].astype(str),
                "partition_role": ROLE,
                "endpoint": frame["endpoint"].to_numpy(float),
                n411.FEATURE_NAMES[0]: pd.to_numeric(
                    frame[n411.FEATURE_NAMES[0]], errors="coerce"
                ),
                "supported": result["supported"],
                "risk_score": result["score"],
                "reject": result["reject"],
                "pauling_p2_p5_decision": frame["pauling_p2_p5_decision"].astype(str),
            }
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_hash = _sha256_file(Path(__file__).resolve())
    try:
        evaluation_path = staging / EVALUATION_NAME
        evaluation_path.write_bytes(_json_bytes(evaluation))
        output_paths = [evaluation_path]
        for source, frame in predictions.items():
            path = staging / PREDICTION_NAMES[source]
            frame.to_parquet(path, index=False)
            output_paths.append(path)
        manifest = {
            "protocol": PROTOCOL,
            "mode": "sequential_frozen_sssp_dual_source_internal_replication",
            "internal_validation_endpoint_values_opened": True,
            "internal_replication_endpoint_values_opened": True,
            "formula_or_threshold_modified": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_executable_formula": False,
            "learned_energy_force_stress_proxy_used": False,
            "model_or_proxy_potential_used": False,
            "physical_relaxation_executed": False,
            "passes_all_replication_gates": passes,
            "inputs_sha256": {
                name: {"path": str(path), "sha256": hashes[name]}
                for name, path in paths.items()
            },
            "executed_source_sha256": {
                "src/next528_sssp_one_shot_replication.py": source_hash,
                "src/next527_sssp_one_shot_validation.py": hashes["next527_source"],
            },
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
            "independent_report_authorized": passes,
            "canonical_integration_authorized": False,
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256_file(Path(__file__).resolve()) != source_hash:
            raise RuntimeError("NEXT528 source changed before publication")
        if any(_sha256_file(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT528 input changed before publication")
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next525-dir", type=Path, required=True)
    parser.add_argument("--next526-dir", type=Path, required=True)
    parser.add_argument("--next527-dir", type=Path, required=True)
    parser.add_argument("--scigen-feature-dir", type=Path, required=True)
    parser.add_argument("--wyformer-feature-dir", type=Path, required=True)
    parser.add_argument("--scigen-endpoint-dir", type=Path, required=True)
    parser.add_argument("--wyformer-endpoint-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = run_one_shot_replication(
        next525_dir=args.next525_dir,
        next526_dir=args.next526_dir,
        next527_dir=args.next527_dir,
        scigen_feature_dir=args.scigen_feature_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        scigen_endpoint_dir=args.scigen_endpoint_dir,
        wyformer_endpoint_dir=args.wyformer_endpoint_dir,
        design_path=args.design,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "passes": manifest["passes_all_replication_gates"],
        "report_authorized": manifest["independent_report_authorized"],
    }, indent=2))


__all__ = [
    "BOOTSTRAP_DRAWS", "BOOTSTRAP_SEED", "EVALUATION_NAME", "GATES",
    "MANIFEST_NAME", "PROTOCOL", "ROLE", "VALIDATION_PROTOCOL",
    "run_one_shot_replication", "validate_replication_authorization",
]


if __name__ == "__main__":
    main()
