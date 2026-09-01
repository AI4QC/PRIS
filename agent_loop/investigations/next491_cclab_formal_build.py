#!/usr/bin/env python3
"""Physically isolated full discovery build for frozen NEXT490 CCLAB."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
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
import src.next347_periodic_allocation_redistribution_capacity as n347
import src.next490_coordination_conditioned_lewis_acidity_balance as n490


PROTOCOL = "2026-08-13-next491-cclab-formal-build-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT491_CCLAB_FEATURE_CATALOGUE.json"
FEATURE_FILES = {
    "scigen": "next491_scigen_coordination_conditioned_lewis_acidity_balance.parquet",
    "wyformer": "next491_wyformer_coordination_conditioned_lewis_acidity_balance.parquet",
}
EXPECTED_ROWS = {"scigen": 13_470, "wyformer": 5_232}
MINIMUM_FORMAL_COVERAGE = 0.95
EXPECTED_PROBE_SHA256 = "947269ac0c96ea267843edd0b95cd932b4450ed928276425e530d184916cec2c"
EXPECTED_INPUT_SHA256 = {
    "design": "1237ec348ea7b830935f831b7e5c0877497b3afa5e3c52815989b16e1b462ae6",
    "probe_result": EXPECTED_PROBE_SHA256,
    "characteristic_acidity_asset": "1f8cceb8eaade9368f96aefbf8da5e5665627c02271641cf079e199da70e4c9c",
    "next267_source": "8f1e7ed9eb73a81a5755d455ffc05aab6f539cbd66afbbbfc384ca88391adca1",
    "next490_source": "c61ec17eb57af9b45fce4d1ed75ed551ca4a711c94fc643b21163aca5ff9f853",
    "scigen_manifest": "dc5bf33c6ce6dc2c10bcd3704688055058145fbe7269ada23ffbe4b141d75fe7",
    "scigen_metadata": "f91455f23b0a96f60fd1c779249e2be46a7ecf94fcdde2b146426a95aac05bde",
    "scigen_discovery_geometry": "e561ef12343c66dcc72bcabf6b8719ad727e01c9582a094e281da73b862ab575",
    "wyformer_manifest": "e0539d556538cb4c052431bc6a1e5c1663bc3de427677dbc8a446dcc3b4fbc54",
    "wyformer_metadata": "3b152b4b84c8d3f7ff5e85611dc1fd2728296f150e907ac4578ce55d2b27dd2b",
    "wyformer_discovery_geometry": "f1ce5ae4fba8c13fcbf3e25de4f596b919d9b41da5b072d9a28eefeaffc69784",
}
PROBE_BOUNDARY_NAMES = (
    "labels_opened",
    "endpoint_fields_read",
    "validation_geometry_opened",
    "replication_geometry_opened",
    "dft_calculation_executed",
    "dft_values_used",
    "learned_energy_force_stress_proxy_used",
    "model_or_proxy_potential_used",
    "physical_relaxation_executed",
)


def validate_probe_authorization(probe: dict[str, object]) -> None:
    gates = probe.get("gates")
    if (
        probe.get("protocol")
        != "2026-08-13-next490-cclab-label-blind-novelty-probe-v1"
        or probe.get("design_sha256") != n490.DESIGN_SHA256
        or probe.get("asset_sha256") != n490.ASSET_SHA256
        or probe.get("minimum_novelty_joint_finite") != 40
        or probe.get("probe_passed") is not True
        or probe.get("next491_formal_build_authorized") is not True
        or not isinstance(gates, dict)
        or set(gates.values()) != {True}
        or any(probe.get(name) is not False for name in PROBE_BOUNDARY_NAMES)
    ):
        raise ValueError("NEXT491 label-blind probe authorization differs")


def _error_row(exc: Exception) -> dict[str, object]:
    return {
        n490.FEATURE_NAMES[0]: math.nan,
        "cclab_supported": False,
        "cclab_failure": f"{type(exc).__name__}: {exc}",
        "cclab_feasible": False,
        "cclab_site_count": 0,
        "cclab_edge_count": 0,
        "cclab_cation_count": 0,
        "cclab_anion_count": 0,
        "cclab_normalized_mismatch": math.nan,
        "cclab_valence_policy": None,
        "cclab_asset_sha256": n490.ASSET_SHA256,
    }


def _compute_scigen_payload(item):
    material_id, payload = item
    try:
        atoms = n267.n85._parse_frame(payload, strict_output=True).atoms
        return material_id, n490.compute_cclab_row(atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_wyformer_payload(item):
    material_id, payload = item
    try:
        atoms = n267.AseAtomsAdaptor.get_atoms(
            n267.Structure.from_dict(json.loads(payload))
        )
        return material_id, n490.compute_cclab_row(atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_many(payloads, *, source: str, workers: int):
    worker = _compute_scigen_payload if source == "scigen" else _compute_wyformer_payload
    if workers == 1:
        return [worker(item) for item in payloads]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(worker, payloads, chunksize=8))


def label_free_statistics(table: pd.DataFrame) -> dict[str, object]:
    values = pd.to_numeric(
        table[n490.FEATURE_NAMES[0]], errors="coerce"
    ).dropna().to_numpy(float)
    if len(values) < 1:
        raise RuntimeError("NEXT491 label-free feature population is empty")
    return {
        n490.FEATURE_NAMES[0]: {
            "minimum": float(np.min(values)),
            "q10": float(np.quantile(values, 0.10, method="inverted_cdf")),
            "median": float(np.median(values)),
            "q90": float(np.quantile(values, 0.90, method="inverted_cdf")),
            "maximum": float(np.max(values)),
            "unique_rounded_10": int(np.unique(np.round(values, 10)).size),
        }
    }


def _formal_paths(
    *,
    scigen: Path,
    wyformer: Path,
    design_path: Path,
    probe_result_path: Path,
    asset_path: Path,
) -> dict[str, Path]:
    return {
        "scigen_manifest": scigen / n267.n85.COHORT_MANIFEST_NAME,
        "scigen_metadata": scigen / n267.n85.COHORT_METADATA_NAME,
        "scigen_discovery_geometry": scigen / n267.n85.GEOMETRY_NAMES["discovery"],
        "wyformer_manifest": wyformer / n267.n94.COHORT_MANIFEST_NAME,
        "wyformer_metadata": wyformer / n267.n94.COHORT_METADATA_NAME,
        "wyformer_discovery_geometry": wyformer / n267.n94.GEOMETRY_NAMES["discovery"],
        "design": Path(design_path).resolve(),
        "probe_result": Path(probe_result_path).resolve(),
        "characteristic_acidity_asset": Path(asset_path).resolve(),
        "next267_source": Path(n267.__file__).resolve(),
        "next490_source": Path(n490.__file__).resolve(),
    }


def _validate_geometry_provenance(paths: dict[str, Path]) -> None:
    scigen_manifest = n267._read_manifest(paths["scigen_manifest"])
    wyformer_manifest = n267._read_manifest(paths["wyformer_manifest"])
    if (
        scigen_manifest.get("protocol") != n267.n85.COHORT_PROTOCOL
        or scigen_manifest.get("labels_opened") is not False
        or scigen_manifest.get("endpoint_payloads_opened") is not False
        or scigen_manifest.get("relaxed_structures_opened") is not False
        or wyformer_manifest.get("protocol") != n267.n94.COHORT_PROTOCOL
        or wyformer_manifest.get("discovery_endpoint_opened") is not False
        or wyformer_manifest.get("validation_endpoint_opened") is not False
        or wyformer_manifest.get("replication_endpoint_opened") is not False
        or wyformer_manifest.get("relaxed_structures_published") is not False
    ):
        raise ValueError("NEXT491 discovery geometry provenance differs")


def build_cross_source_cclab_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    probe_result_path: Path,
    asset_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    scigen = Path(scigen_cohort_dir).resolve()
    wyformer = Path(wyformer_cohort_dir).resolve()
    target = Path(output_dir).resolve()
    paths = _formal_paths(
        scigen=scigen,
        wyformer=wyformer,
        design_path=design_path,
        probe_result_path=probe_result_path,
        asset_path=asset_path,
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(workers) is not int or workers <= 0:
        raise ValueError("NEXT491 workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT491 input is missing")
    input_hashes = {
        name: n347._sha256_file(path) for name, path in paths.items()
    }
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT491 formal input identity differs: {differing}")
    validate_probe_authorization(json.loads(paths["probe_result"].read_text()))
    _validate_geometry_provenance(paths)
    metadata = {
        "scigen": pd.read_parquet(paths["scigen_metadata"]),
        "wyformer": pd.read_parquet(paths["wyformer_metadata"]),
    }
    required = {
        "material_id",
        "reduced_formula",
        "chemical_system",
        "natoms",
        "partition_role",
        "input_role",
    }
    discovery: dict[str, pd.DataFrame] = {}
    for source, frame in metadata.items():
        if required - set(frame) or frame["material_id"].astype(str).duplicated().any():
            raise ValueError(f"NEXT491 {source} metadata differs")
        selected = (
            frame.loc[frame["partition_role"].eq("discovery")]
            .copy()
            .sort_values("material_id", kind="mergesort")
            .reset_index(drop=True)
        )
        if len(selected) != EXPECTED_ROWS[source]:
            raise ValueError(f"NEXT491 {source} discovery identity differs")
        discovery[source] = selected
    payloads = {
        "scigen": n267.n85._archive_payloads(
            paths["scigen_discovery_geometry"],
            discovery["scigen"]["material_id"].astype(str).tolist(),
        ),
        "wyformer": n267.n94._payloads(
            paths["wyformer_discovery_geometry"],
            discovery["wyformer"]["material_id"].astype(str).tolist(),
        ),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    root = Path(__file__).resolve().parent.parent
    executed_paths = {
        "src/next491_cclab_formal_build.py": Path(__file__).resolve(),
        "src/next490_coordination_conditioned_lewis_acidity_balance.py": Path(n490.__file__).resolve(),
        "experiments/next490_cclab_label_blind_engineering_probe.py": root / "experiments/next490_cclab_label_blind_engineering_probe.py",
        "experiments/next490_cclab_label_blind_novelty_probe.py": root / "experiments/next490_cclab_label_blind_novelty_probe.py",
        "tests/test_next490_coordination_conditioned_lewis_acidity_balance.py": root / "tests/test_next490_coordination_conditioned_lewis_acidity_balance.py",
        "tests/test_next490_cclab_label_blind_engineering_probe.py": root / "tests/test_next490_cclab_label_blind_engineering_probe.py",
        "tests/test_next490_cclab_label_blind_novelty_probe.py": root / "tests/test_next490_cclab_label_blind_novelty_probe.py",
        "tests/test_next491_cclab_formal_build.py": root / "tests/test_next491_cclab_formal_build.py",
    }
    executed_hashes = {
        name: n347._sha256_file(path) for name, path in executed_paths.items()
    }
    started = time.perf_counter()
    try:
        computed = {
            source: _compute_many(payloads[source], source=source, workers=workers)
            for source in ("scigen", "wyformer")
        }
        counts: dict[str, dict[str, object]] = {}
        statistics: dict[str, dict[str, object]] = {}
        outputs: list[Path] = []
        for source in ("scigen", "wyformer"):
            computed_frame = pd.DataFrame(
                [
                    {"material_id": material_id, **row}
                    for material_id, row in computed[source]
                ]
            )
            if (
                computed_frame["material_id"].astype(str).duplicated().any()
                or set(computed_frame["material_id"].astype(str))
                != set(discovery[source]["material_id"].astype(str))
            ):
                raise RuntimeError(f"NEXT491 {source} material identity differs")
            table = discovery[source].merge(
                computed_frame, on="material_id", how="left", validate="one_to_one"
            )
            supported = table["cclab_supported"].fillna(False).astype(bool)
            finite = np.isfinite(
                pd.to_numeric(table[n490.FEATURE_NAMES[0]], errors="coerce").to_numpy(float)
            )
            if not finite[supported].all() or finite[~supported].any():
                raise RuntimeError(f"NEXT491 {source} finite support semantics differ")
            coverage = float(supported.mean())
            statistics[source] = label_free_statistics(table)
            failures = Counter(table.loc[~supported, "cclab_failure"].fillna("unknown"))
            counts[source] = {
                "rows": int(len(table)),
                "supported": int(supported.sum()),
                "failures": int((~supported).sum()),
                "coverage": coverage,
                "coverage_gate_passed": bool(coverage >= MINIMUM_FORMAL_COVERAGE),
                "site_count": int(pd.to_numeric(table.loc[supported, "cclab_site_count"]).sum()),
                "edge_count": int(pd.to_numeric(table.loc[supported, "cclab_edge_count"]).sum()),
                "cation_count": int(pd.to_numeric(table.loc[supported, "cclab_cation_count"]).sum()),
                "anion_count": int(pd.to_numeric(table.loc[supported, "cclab_anion_count"]).sum()),
                "finite_feature_count": int(finite.sum()),
                "failure_counts": {
                    str(key): int(value) for key, value in failures.items()
                },
            }
            output_path = staging / FEATURE_FILES[source]
            table.to_parquet(output_path, index=False)
            outputs.append(output_path)
        coverage_gate_passed = bool(
            all(counts[source]["coverage_gate_passed"] for source in counts)
        )
        catalogue = {
            "protocol": PROTOCOL,
            "feature_count": 1,
            "feature_names": list(n490.FEATURE_NAMES),
            "feature_directions": n490.FEATURE_DIRECTIONS,
            "directions_frozen_before_outcome": True,
            "graph": "opposite-sign periodic Voronoi",
            "formula": "1-sum_j distance(|q_j|,[sum_i a_i_low,sum_i a_i_high])/sum_j(|q_j|+projection)",
            "characteristic_acidity_asset_sha256": n490.ASSET_SHA256,
            "output_grid": 1.0 / n490.OUTPUT_GRID,
            "minimum_formal_coverage": MINIMUM_FORMAL_COVERAGE,
            "label_free_statistics": statistics,
            "coverage_gate_passed": coverage_gate_passed,
            "next492_audit_authorized": coverage_gate_passed,
            **n490.BOUNDARY_FLAGS,
        }
        catalogue_path = staging / CATALOGUE_NAME
        n347._write_json(catalogue_path, catalogue)
        outputs.append(catalogue_path)
        manifest = {
            "protocol": PROTOCOL,
            "mode": "physically_isolated_discovery_x0_coordination_conditioned_lewis_acidity_freeze",
            "workers": workers,
            "elapsed_seconds": float(time.perf_counter() - started),
            "counts": counts,
            "coverage_gate_passed": coverage_gate_passed,
            "source_partitions_read": {"scigen": ["discovery"], "wyformer": ["discovery"]},
            "labels_opened": False,
            "endpoint_fields_read": False,
            "internal_validation_geometry_opened": False,
            "internal_replication_geometry_opened": False,
            "scientific_improvement_claim": False,
            "next492_audit_authorized": coverage_gate_passed,
            **n490.BOUNDARY_FLAGS,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": executed_hashes,
            "outputs_sha256": {
                path.name: n347._sha256_file(path) for path in outputs
            },
        }
        n347._write_json(staging / MANIFEST_NAME, manifest)
        if any(
            n347._sha256_file(path) != executed_hashes[name]
            for name, path in executed_paths.items()
        ):
            raise RuntimeError("NEXT491 executed artifact changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scigen-cohort-dir", required=True, type=Path)
    parser.add_argument("--wyformer-cohort-dir", required=True, type=Path)
    parser.add_argument("--design-path", required=True, type=Path)
    parser.add_argument("--probe-result-path", required=True, type=Path)
    parser.add_argument("--asset-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    manifest = build_cross_source_cclab_features(
        scigen_cohort_dir=args.scigen_cohort_dir,
        wyformer_cohort_dir=args.wyformer_cohort_dir,
        design_path=args.design_path,
        probe_result_path=args.probe_result_path,
        asset_path=args.asset_path,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CATALOGUE_NAME",
    "EXPECTED_PROBE_SHA256",
    "EXPECTED_ROWS",
    "FEATURE_FILES",
    "MANIFEST_NAME",
    "MINIMUM_FORMAL_COVERAGE",
    "PROBE_BOUNDARY_NAMES",
    "PROTOCOL",
    "build_cross_source_cclab_features",
    "label_free_statistics",
    "n490",
    "validate_probe_authorization",
]
