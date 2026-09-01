#!/usr/bin/env python3
"""Compute frozen NEXT411 SSSP on validation and replication raw geometries."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import pandas as pd

import src.next267_periodic_radical_voronoi_packing as n267
import src.next411_same_sign_shell_purity as n411
import src.next412_sssp_formal_build as n412
import src.next525_sssp_standalone_freeze as n525
from src.next347_periodic_allocation_redistribution_capacity import _sha256_file


PROTOCOL = "2026-08-13-next526-sssp-all-holdout-label-free-feature-freeze-v1"
DESIGN_SHA256 = n525.DESIGN_SHA256
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT526_SSSP_HOLDOUT_FEATURE_CATALOGUE.json"
ROLES = ("internal_validation", "internal_replication")
EXPECTED_ROWS = {
    "scigen": {"internal_validation": 5_174, "internal_replication": 6_098},
    "wyformer": {"internal_validation": 2_135, "internal_replication": 2_213},
}
FEATURE_FILES = {
    source: {
        role: f"next526_{source}_sssp_{role}.parquet" for role in ROLES
    }
    for source in ("scigen", "wyformer")
}
MINIMUM_FORMAL_COVERAGE = 0.95
BOUNDARY_FLAGS = {
    "validation_endpoint_opened": False,
    "replication_endpoint_opened": False,
    "endpoint_fields_read": False,
    "dft_calculation_executed": False,
    "dft_values_used_by_features": False,
    "learned_energy_force_stress_proxy_used": False,
    "model_or_proxy_potential_used": False,
    "physical_relaxation_executed": False,
    "relaxed_coordinates_used_by_features": False,
}
BLOCKED_METADATA_SUBSTRINGS = (
    "endpoint", "distortion", "force", "e_above_hull", "energy", "relax"
)
EXPECTED_INPUT_SHA256 = {
    "design": DESIGN_SHA256,
    "next411_source": "172543534328a387b7d2b12ffd6cad919793ace56ec1124dd6e228f96d8cc9a4",
    "next412_source": "b4ae4016a92217237d8eaccd2449fc7bfcee2193d31f615f0a79fd49f9fedaca",
    "next525_manifest": "e15217dafaa1d86dc5a70640dd0ab96a99a9cc0bb04eff44ca850c88e4ff3140",
    "next525_formula": "e98f7cf1bf6d0947b653c133100495650a57265dddde46ce8e2c4dd9521e09cf",
    "scigen_manifest": "dc5bf33c6ce6dc2c10bcd3704688055058145fbe7269ada23ffbe4b141d75fe7",
    "scigen_metadata": "f91455f23b0a96f60fd1c779249e2be46a7ecf94fcdde2b146426a95aac05bde",
    "scigen_validation_geometry": "d79d6c5466a4dcb06fb22df3c2aa118687fd5f1434efc7aede25b4a6444ea278",
    "scigen_replication_geometry": "7c335c6a49fa9a1674c893ceae70654e7365a51fadf6124d60a10d2f86f5c087",
    "wyformer_manifest": "e0539d556538cb4c052431bc6a1e5c1663bc3de427677dbc8a446dcc3b4fbc54",
    "wyformer_metadata": "3b152b4b84c8d3f7ff5e85611dc1fd2728296f150e907ac4578ce55d2b27dd2b",
    "wyformer_validation_geometry": "fa2c017b8ece8600d0810f9851013a7015688e6b8c87545d7accc07901682fb8",
    "wyformer_replication_geometry": "485f88ce5798acf37b27688b04109bc1c47da7637091196636cccca65983455d",
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def select_holdout_metadata(
    metadata: pd.DataFrame, *, expected_rows: dict[str, int]
) -> dict[str, pd.DataFrame]:
    required = {
        "material_id", "reduced_formula", "chemical_system", "natoms",
        "partition_role", "input_role",
    }
    if (
        required - set(metadata)
        or metadata["material_id"].astype(str).duplicated().any()
        or set(expected_rows) != set(ROLES)
    ):
        raise ValueError("NEXT526 metadata differs")
    blocked = sorted(
        column for column in metadata
        if any(token in str(column).lower() for token in BLOCKED_METADATA_SUBSTRINGS)
    )
    if blocked:
        raise ValueError(f"NEXT526 endpoint field present in metadata: {blocked}")
    selected = {}
    for role in ROLES:
        frame = metadata.loc[metadata["partition_role"].eq(role)].copy()
        frame = frame.sort_values("material_id", kind="mergesort").reset_index(drop=True)
        if len(frame) != int(expected_rows[role]):
            raise ValueError(f"NEXT526 {role} metadata differs")
        selected[role] = frame
    return selected


def _paths(
    *, scigen_cohort_dir: Path, wyformer_cohort_dir: Path,
    next525_dir: Path, design_path: Path,
) -> dict[str, Path]:
    scigen = Path(scigen_cohort_dir).resolve()
    wyformer = Path(wyformer_cohort_dir).resolve()
    freeze = Path(next525_dir).resolve()
    return {
        "design": Path(design_path).resolve(),
        "next411_source": Path(n411.__file__).resolve(),
        "next412_source": Path(n412.__file__).resolve(),
        "next525_manifest": freeze / n525.MANIFEST_NAME,
        "next525_formula": freeze / n525.FORMULA_NAME,
        "scigen_manifest": scigen / n267.n85.COHORT_MANIFEST_NAME,
        "scigen_metadata": scigen / n267.n85.COHORT_METADATA_NAME,
        "scigen_validation_geometry": scigen / n267.n85.GEOMETRY_NAMES["internal_validation"],
        "scigen_replication_geometry": scigen / n267.n85.GEOMETRY_NAMES["internal_replication"],
        "wyformer_manifest": wyformer / n267.n94.COHORT_MANIFEST_NAME,
        "wyformer_metadata": wyformer / n267.n94.COHORT_METADATA_NAME,
        "wyformer_validation_geometry": wyformer / n267.n94.GEOMETRY_NAMES["internal_validation"],
        "wyformer_replication_geometry": wyformer / n267.n94.GEOMETRY_NAMES["internal_replication"],
    }


def build_sssp_holdout_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    next525_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Read only raw holdout geometries and freeze SSSP for both roles."""

    paths = _paths(
        scigen_cohort_dir=scigen_cohort_dir,
        wyformer_cohort_dir=wyformer_cohort_dir,
        next525_dir=next525_dir,
        design_path=design_path,
    )
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(workers) is not int or workers <= 0:
        raise ValueError("NEXT526 workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT526 input is missing")
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name for name in set(hashes) | set(EXPECTED_INPUT_SHA256)
            if hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT526 formal input identity differs: {differing}")
    next525_manifest = _read_json(paths["next525_manifest"])
    formula = _read_json(paths["next525_formula"])
    scigen_manifest = _read_json(paths["scigen_manifest"])
    wyformer_manifest = _read_json(paths["wyformer_manifest"])
    if (
        next525_manifest.get("protocol") != n525.PROTOCOL
        or next525_manifest.get("next526_holdout_feature_freeze_authorized") is not True
        or next525_manifest.get("validation_endpoint_opened") is not False
        or next525_manifest.get("replication_endpoint_opened") is not False
        or formula.get("feature") != n411.FEATURE_NAMES[0]
        or float(formula.get("threshold", math.nan)) != n525.EXPECTED_THRESHOLD
        or formula.get("dft_inputs") != []
        or formula.get("learned_model_inputs") != []
        or formula.get("relaxation_inputs") != []
        or scigen_manifest.get("labels_opened") is not False
        or scigen_manifest.get("endpoint_payloads_opened") is not False
        or wyformer_manifest.get("discovery_endpoint_opened") is not False
        or wyformer_manifest.get("validation_endpoint_opened") is not False
        or wyformer_manifest.get("replication_endpoint_opened") is not False
    ):
        raise ValueError("NEXT526 label-free provenance differs")
    metadata = {
        "scigen": select_holdout_metadata(
            pd.read_parquet(paths["scigen_metadata"]), expected_rows=EXPECTED_ROWS["scigen"]
        ),
        "wyformer": select_holdout_metadata(
            pd.read_parquet(paths["wyformer_metadata"]), expected_rows=EXPECTED_ROWS["wyformer"]
        ),
    }
    payloads = {
        "scigen": {
            "internal_validation": n267.n85._archive_payloads(
                paths["scigen_validation_geometry"],
                metadata["scigen"]["internal_validation"]["material_id"].astype(str).tolist(),
            ),
            "internal_replication": n267.n85._archive_payloads(
                paths["scigen_replication_geometry"],
                metadata["scigen"]["internal_replication"]["material_id"].astype(str).tolist(),
            ),
        },
        "wyformer": {
            "internal_validation": n267.n94._payloads(
                paths["wyformer_validation_geometry"],
                metadata["wyformer"]["internal_validation"]["material_id"].astype(str).tolist(),
            ),
            "internal_replication": n267.n94._payloads(
                paths["wyformer_replication_geometry"],
                metadata["wyformer"]["internal_replication"]["material_id"].astype(str).tolist(),
            ),
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    started = time.perf_counter()
    source_hash = _sha256_file(Path(__file__).resolve())
    try:
        counts: dict[str, dict[str, object]] = {source: {} for source in metadata}
        statistics: dict[str, dict[str, object]] = {source: {} for source in metadata}
        output_paths = []
        for source in ("scigen", "wyformer"):
            for role in ROLES:
                computed = n412._compute_many(payloads[source][role], source=source, workers=workers)
                computed_frame = pd.DataFrame(
                    [{"material_id": material_id, **row} for material_id, row in computed]
                )
                base = metadata[source][role]
                if (
                    computed_frame["material_id"].astype(str).duplicated().any()
                    or set(computed_frame["material_id"].astype(str))
                    != set(base["material_id"].astype(str))
                ):
                    raise RuntimeError(f"NEXT526 {source} {role} material identity differs")
                table = base.merge(computed_frame, on="material_id", validate="one_to_one")
                supported = table["sssp_supported"].fillna(False).to_numpy(bool)
                values = pd.to_numeric(table[n411.FEATURE_NAMES[0]], errors="coerce").to_numpy(float)
                finite = np.isfinite(values)
                if not np.array_equal(supported, finite):
                    raise RuntimeError(f"NEXT526 {source} {role} support semantics differ")
                coverage = float(supported.mean())
                unique = int(np.unique(np.round(values[finite], 10)).size)
                if coverage < MINIMUM_FORMAL_COVERAGE or unique < 20:
                    raise RuntimeError(f"NEXT526 {source} {role} feature gate failed")
                failure_counts = Counter(table.loc[~supported, "sssp_failure"].fillna("unknown"))
                counts[source][role] = {
                    "rows": int(len(table)),
                    "supported": int(supported.sum()),
                    "failures": int((~supported).sum()),
                    "coverage": coverage,
                    "unique_rounded_10": unique,
                    "failure_counts": {
                        str(key): int(value) for key, value in failure_counts.items()
                    },
                }
                statistics[source][role] = n412.label_free_statistics(table)
                path = staging / FEATURE_FILES[source][role]
                table.to_parquet(path, index=False)
                output_paths.append(path)
        catalogue = {
            "protocol": PROTOCOL,
            "feature_names": list(n411.FEATURE_NAMES),
            "feature_directions": n411.FEATURE_DIRECTIONS,
            "formula_sha256": hashes["next525_formula"],
            "roles": list(ROLES),
            "minimum_formal_coverage": MINIMUM_FORMAL_COVERAGE,
            "endpoint_fields_present": False,
            "statistics": statistics,
        }
        catalogue_path = staging / CATALOGUE_NAME
        catalogue_path.write_bytes(_json_bytes(catalogue))
        output_paths.append(catalogue_path)
        manifest = {
            "protocol": PROTOCOL,
            "mode": "all_holdout_raw_x0_label_free_sssp_freeze",
            "counts": counts,
            "elapsed_seconds": time.perf_counter() - started,
            "internal_validation_geometry_opened": True,
            "internal_replication_geometry_opened": True,
            **BOUNDARY_FLAGS,
            "inputs_sha256": {
                name: {"path": str(path), "sha256": hashes[name]}
                for name, path in paths.items()
            },
            "executed_source_sha256": {
                "src/next526_sssp_holdout_feature_freeze.py": source_hash,
                "src/next411_same_sign_shell_purity.py": hashes["next411_source"],
            },
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
            "next527_internal_validation_authorized": True,
            "next528_internal_replication_authorized": False,
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256_file(Path(__file__).resolve()) != source_hash:
            raise RuntimeError("NEXT526 source changed before publication")
        if any(_sha256_file(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT526 input changed before publication")
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scigen-cohort-dir", type=Path, required=True)
    parser.add_argument("--wyformer-cohort-dir", type=Path, required=True)
    parser.add_argument("--next525-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_sssp_holdout_features(
        scigen_cohort_dir=args.scigen_cohort_dir,
        wyformer_cohort_dir=args.wyformer_cohort_dir,
        next525_dir=args.next525_dir,
        design_path=args.design,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


__all__ = [
    "BOUNDARY_FLAGS", "CATALOGUE_NAME", "FEATURE_FILES", "MANIFEST_NAME",
    "PROTOCOL", "build_sssp_holdout_features", "select_holdout_metadata",
]


if __name__ == "__main__":
    main()
