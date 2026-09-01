#!/usr/bin/env python3
"""Full discovery coverage build for conservative NEXT495 CCLAB-CDE."""

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
import src.next491_cclab_formal_build as n491
import src.next495_cclab_conservative_domain_extension as n495


PROTOCOL = "2026-08-13-next496-cclab-cde-formal-build-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT496_CCLAB_CDE_FEATURE_CATALOGUE.json"
FEATURE_FILES = {
    "scigen": "next496_scigen_cclab_conservative_domain_extension.parquet",
    "wyformer": "next496_wyformer_cclab_conservative_domain_extension.parquet",
}
EXPECTED_ROWS = dict(n491.EXPECTED_ROWS)
MINIMUM_FORMAL_COVERAGE = 0.95
EXPECTED_PROBE_SHA256 = "7c8e608adda62ea79a4dd5b418c269edbf9c0ae094a97a9eeb00525ea674411d"
EXPECTED_INPUT_SHA256 = {
    "design": "26cefbfee766bd408064738a076e6b29730b44cd2a4b3a035a1426ae761ead04",
    "probe_result": EXPECTED_PROBE_SHA256,
    "characteristic_acidity_asset": "1f8cceb8eaade9368f96aefbf8da5e5665627c02271641cf079e199da70e4c9c",
    "base_next491_manifest": "13abffe1951ce104f6302ba1f757af8feb6e9d37a237ac3e18aa68be425dfdb4",
    "next267_source": "8f1e7ed9eb73a81a5755d455ffc05aab6f539cbd66afbbbfc384ca88391adca1",
    "next495_source": "7b044889f291ea64af7ff6ef226b0322505f3d2fade0117f6879315c58e9f3ef",
    "scigen_manifest": "dc5bf33c6ce6dc2c10bcd3704688055058145fbe7269ada23ffbe4b141d75fe7",
    "scigen_metadata": "f91455f23b0a96f60fd1c779249e2be46a7ecf94fcdde2b146426a95aac05bde",
    "scigen_discovery_geometry": "e561ef12343c66dcc72bcabf6b8719ad727e01c9582a094e281da73b862ab575",
    "wyformer_manifest": "e0539d556538cb4c052431bc6a1e5c1663bc3de427677dbc8a446dcc3b4fbc54",
    "wyformer_metadata": "3b152b4b84c8d3f7ff5e85611dc1fd2728296f150e907ac4578ce55d2b27dd2b",
    "wyformer_discovery_geometry": "f1ce5ae4fba8c13fcbf3e25de4f596b919d9b41da5b072d9a28eefeaffc69784",
}
PROBE_BOUNDARY_NAMES = n491.PROBE_BOUNDARY_NAMES


def validate_probe_authorization(probe: dict[str, object]) -> None:
    gates = probe.get("gates")
    if (
        probe.get("protocol") != "2026-08-13-next495-cclab-cde-label-blind-probe-v1"
        or probe.get("design_sha256") != n495.DESIGN_SHA256
        or probe.get("asset_sha256") != n495.ASSET_SHA256
        or probe.get("probe_passed") is not True
        or probe.get("next496_formal_build_authorized") is not True
        or not isinstance(gates, dict)
        or set(gates.values()) != {True}
        or any(probe.get(name) is not False for name in PROBE_BOUNDARY_NAMES)
    ):
        raise ValueError("NEXT496 label-blind probe authorization differs")


def _error_row(exc: Exception) -> dict[str, object]:
    return {
        n495.FEATURE_NAMES[0]: math.nan,
        "cclab_cde_supported": False,
        "cclab_cde_failure": f"{type(exc).__name__}: {exc}",
        "cclab_cde_feasible": False,
        "cclab_cde_site_count": 0,
        "cclab_cde_edge_count": 0,
        "cclab_cde_cation_count": 0,
        "cclab_cde_anion_count": 0,
        "cclab_cde_unknown_cation_count": 0,
        "cclab_cde_unknown_anion_neighborhood_count": 0,
        "cclab_cde_normalized_mismatch": math.nan,
        "cclab_cde_valence_policy": None,
        "cclab_cde_asset_sha256": n495.ASSET_SHA256,
    }


def _compute_scigen_payload(item):
    material_id, payload = item
    try:
        atoms = n267.n85._parse_frame(payload, strict_output=True).atoms
        return material_id, n495.compute_cclab_cde_row(atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_wyformer_payload(item):
    material_id, payload = item
    try:
        atoms = n267.AseAtomsAdaptor.get_atoms(
            n267.Structure.from_dict(json.loads(payload))
        )
        return material_id, n495.compute_cclab_cde_row(atoms)
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
        table[n495.FEATURE_NAMES[0]], errors="coerce"
    ).dropna().to_numpy(float)
    if len(values) < 1:
        raise RuntimeError("NEXT496 label-free feature population is empty")
    return {
        n495.FEATURE_NAMES[0]: {
            "minimum": float(np.min(values)),
            "q10": float(np.quantile(values, 0.10, method="inverted_cdf")),
            "median": float(np.median(values)),
            "q90": float(np.quantile(values, 0.90, method="inverted_cdf")),
            "maximum": float(np.max(values)),
            "unique_rounded_10": int(np.unique(np.round(values, 10)).size),
        }
    }


def _formal_paths(
    *, scigen: Path, wyformer: Path, design_path: Path, probe_result_path: Path,
    asset_path: Path, base_manifest_path: Path,
) -> dict[str, Path]:
    paths = n491._formal_paths(
        scigen=scigen, wyformer=wyformer, design_path=design_path,
        probe_result_path=probe_result_path, asset_path=asset_path,
    )
    paths["base_next491_manifest"] = Path(base_manifest_path).resolve()
    paths["next495_source"] = Path(n495.__file__).resolve()
    paths.pop("next490_source")
    paths["characteristic_acidity_asset"] = paths.pop("characteristic_acidity_asset")
    return paths


def build_cross_source_cclab_cde_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    probe_result_path: Path,
    asset_path: Path,
    base_manifest_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    scigen, wyformer = Path(scigen_cohort_dir).resolve(), Path(wyformer_cohort_dir).resolve()
    target = Path(output_dir).resolve()
    paths = _formal_paths(
        scigen=scigen, wyformer=wyformer, design_path=design_path,
        probe_result_path=probe_result_path, asset_path=asset_path,
        base_manifest_path=base_manifest_path,
    )
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(workers) is not int or workers <= 0:
        raise ValueError("NEXT496 workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT496 input is missing")
    input_hashes = {name: n347._sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT496 formal input identity differs: {differing}")
    validate_probe_authorization(json.loads(paths["probe_result"].read_text()))
    base = json.loads(paths["base_next491_manifest"].read_text())
    if base.get("coverage_gate_passed") is not False or base.get("labels_opened") is not False:
        raise ValueError("NEXT496 base coverage certificate differs")
    n491._validate_geometry_provenance(paths)
    metadata = {
        "scigen": pd.read_parquet(paths["scigen_metadata"]),
        "wyformer": pd.read_parquet(paths["wyformer_metadata"]),
    }
    required = {"material_id", "reduced_formula", "chemical_system", "natoms", "partition_role", "input_role"}
    discovery = {}
    for source, frame in metadata.items():
        if required - set(frame) or frame["material_id"].astype(str).duplicated().any():
            raise ValueError(f"NEXT496 {source} metadata differs")
        selected = frame.loc[frame["partition_role"].eq("discovery")].copy().sort_values("material_id", kind="mergesort").reset_index(drop=True)
        if len(selected) != EXPECTED_ROWS[source]:
            raise ValueError(f"NEXT496 {source} discovery identity differs")
        discovery[source] = selected
    payloads = {
        "scigen": n267.n85._archive_payloads(paths["scigen_discovery_geometry"], discovery["scigen"]["material_id"].astype(str).tolist()),
        "wyformer": n267.n94._payloads(paths["wyformer_discovery_geometry"], discovery["wyformer"]["material_id"].astype(str).tolist()),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    root = Path(__file__).resolve().parent.parent
    executed_paths = {
        "src/next496_cclab_cde_formal_build.py": Path(__file__).resolve(),
        "src/next495_cclab_conservative_domain_extension.py": Path(n495.__file__).resolve(),
        "experiments/next495_cclab_cde_label_blind_probe.py": root / "experiments/next495_cclab_cde_label_blind_probe.py",
        "tests/test_next495_cclab_conservative_domain_extension.py": root / "tests/test_next495_cclab_conservative_domain_extension.py",
        "tests/test_next495_cclab_cde_label_blind_probe.py": root / "tests/test_next495_cclab_cde_label_blind_probe.py",
        "tests/test_next496_cclab_cde_formal_build.py": root / "tests/test_next496_cclab_cde_formal_build.py",
    }
    executed_hashes = {name: n347._sha256_file(path) for name, path in executed_paths.items()}
    started = time.perf_counter()
    try:
        computed = {source: _compute_many(payloads[source], source=source, workers=workers) for source in ("scigen", "wyformer")}
        counts, statistics, outputs = {}, {}, []
        for source in ("scigen", "wyformer"):
            computed_frame = pd.DataFrame([{"material_id": material_id, **row} for material_id, row in computed[source]])
            if computed_frame["material_id"].astype(str).duplicated().any() or set(computed_frame["material_id"].astype(str)) != set(discovery[source]["material_id"].astype(str)):
                raise RuntimeError(f"NEXT496 {source} material identity differs")
            table = discovery[source].merge(computed_frame, on="material_id", how="left", validate="one_to_one")
            supported = table["cclab_cde_supported"].fillna(False).astype(bool)
            finite = np.isfinite(pd.to_numeric(table[n495.FEATURE_NAMES[0]], errors="coerce").to_numpy(float))
            if not finite[supported].all() or finite[~supported].any():
                raise RuntimeError(f"NEXT496 {source} finite support semantics differ")
            coverage = float(supported.mean())
            statistics[source] = label_free_statistics(table)
            failures = Counter(table.loc[~supported, "cclab_cde_failure"].fillna("unknown"))
            counts[source] = {
                "rows": int(len(table)), "supported": int(supported.sum()), "failures": int((~supported).sum()),
                "coverage": coverage, "coverage_gate_passed": bool(coverage >= MINIMUM_FORMAL_COVERAGE),
                "site_count": int(pd.to_numeric(table.loc[supported, "cclab_cde_site_count"]).sum()),
                "edge_count": int(pd.to_numeric(table.loc[supported, "cclab_cde_edge_count"]).sum()),
                "unknown_cation_count": int(pd.to_numeric(table.loc[supported, "cclab_cde_unknown_cation_count"]).sum()),
                "unknown_anion_neighborhood_count": int(pd.to_numeric(table.loc[supported, "cclab_cde_unknown_anion_neighborhood_count"]).sum()),
                "finite_feature_count": int(finite.sum()),
                "failure_counts": {str(key): int(value) for key, value in failures.items()},
            }
            output_path = staging / FEATURE_FILES[source]
            table.to_parquet(output_path, index=False); outputs.append(output_path)
        coverage_gate_passed = bool(all(counts[source]["coverage_gate_passed"] for source in counts))
        catalogue = {
            "protocol": PROTOCOL, "feature_count": 1, "feature_names": list(n495.FEATURE_NAMES),
            "feature_directions": n495.FEATURE_DIRECTIONS, "directions_frozen_before_outcome": True,
            "post_coverage_extension": True, "new_novelty_claim": False,
            "formula": "NEXT490 CCLAB with projected received acidity fixed to zero for any anion neighborhood containing an unknown donor element",
            "minimum_formal_coverage": MINIMUM_FORMAL_COVERAGE, "label_free_statistics": statistics,
            "coverage_gate_passed": coverage_gate_passed, "next497_audit_authorized": coverage_gate_passed,
            **n495.BOUNDARY_FLAGS,
        }
        catalogue_path = staging / CATALOGUE_NAME; n347._write_json(catalogue_path, catalogue); outputs.append(catalogue_path)
        manifest = {
            "protocol": PROTOCOL, "mode": "post_coverage_label_blind_conservative_domain_extension_discovery_x0",
            "workers": workers, "elapsed_seconds": float(time.perf_counter() - started), "counts": counts,
            "coverage_gate_passed": coverage_gate_passed, "next497_audit_authorized": coverage_gate_passed,
            "post_coverage_extension": True, "new_novelty_claim": False,
            "source_partitions_read": {"scigen": ["discovery"], "wyformer": ["discovery"]},
            "labels_opened": False, "endpoint_fields_read": False,
            "internal_validation_geometry_opened": False, "internal_replication_geometry_opened": False,
            "scientific_improvement_claim": False, **n495.BOUNDARY_FLAGS,
            "inputs_sha256": input_hashes, "executed_source_sha256": executed_hashes,
            "outputs_sha256": {path.name: n347._sha256_file(path) for path in outputs},
        }
        n347._write_json(staging / MANIFEST_NAME, manifest)
        if any(n347._sha256_file(path) != executed_hashes[name] for name, path in executed_paths.items()):
            raise RuntimeError("NEXT496 executed artifact changed before publication")
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
    parser.add_argument("--base-manifest-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    result = build_cross_source_cclab_cde_features(
        scigen_cohort_dir=args.scigen_cohort_dir, wyformer_cohort_dir=args.wyformer_cohort_dir,
        design_path=args.design_path, probe_result_path=args.probe_result_path,
        asset_path=args.asset_path, base_manifest_path=args.base_manifest_path,
        output_dir=args.output_dir, workers=args.workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
