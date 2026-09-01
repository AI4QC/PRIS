#!/usr/bin/env python3
"""Qualify raw-x0 Alexandria path identities before endpoint evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from src.next13d_acsc_dft_pairs import _json_bytes, _sha256_file
from src.next14_wbm_holdout import _publish_directory_no_replace
from src.next16_alexandria_source_map import _scan_all, classify_location
from src.next18_alexandria_holdout import iter_bz2_object


PROTOCOL = "2026-08-03-next42-alexandria-source-qualification-v1"
OUTPUT_NAME = "alexandria_source_qualification.parquet"
MANIFEST_NAME = "MANIFEST.json"
FROZEN_FORMAL_SHA256: Mapping[str, str] = {
    "pbe_0000": "9f83c116839d528a6c625ad158b060298a969f76ce69dd1e29a74806376e389d",
    "pbe_0001": "dff2091cc3a8eaf38472ef0487d1bd678bda3d62ad6c91e226a5a187058387dd",
    "benchmarks_pbe": "f72bec462833e1ce8ef7540cffe335b694afc0a67d640a4dc72aa684a8a07966",
}
RAW_X0_SOURCE_FAMILIES = frozenset(
    {"cgat_comp/binaries", "cgat_comp/ternaries"}
)
DOCUMENTED_MLIP_SOURCE_PREFIXES = ("m3gnet/", "orbital/")
DOCUMENTED_MLIP_SOURCE_FAMILIES = frozenset(
    {"cgat_comp/quaternaries", "cgat_comp_2/binaries", "cgat_comp_2/ternaries"}
)
EVIDENCE_URLS: Mapping[str, str] = {
    "round_mapping": "https://zenodo.org/records/12582650",
    "round1_workflow": "https://doi.org/10.1002/adma.202210788",
    "round2_round3_prerelaxation": "https://doi.org/10.1016/j.mtphys.2024.101560",
    "alexandria_2025_paths": "https://alexandria.icams.ruhr-uni-bochum.de/datasets.html",
}


def _load_id_file(path: Path) -> tuple[set[str], int]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "# mat_id":
        raise ValueError("Alexandria benchmark header differs")
    values = [line.strip() for line in lines[1:] if line.strip()]
    return set(values), len(values)


def source_qualification(
    source_family: str, *, official_benchmark: bool
) -> tuple[bool, str]:
    """Return a fail-closed raw-x0 provenance decision."""

    if type(official_benchmark) is not bool:
        raise TypeError("official_benchmark must be an exact boolean")
    family = str(source_family)
    if official_benchmark:
        return False, "official_benchmark_identity"
    if family in RAW_X0_SOURCE_FAMILIES:
        return True, "eligible_round1_raw_x0"
    if family in DOCUMENTED_MLIP_SOURCE_FAMILIES or family.startswith(
        DOCUMENTED_MLIP_SOURCE_PREFIXES
    ):
        return False, "documented_mlip_prerelaxation"
    return False, "unverified_raw_x0_provenance"


def _trajectory_ids(paths: Mapping[str, Path]) -> tuple[set[str], dict[str, int]]:
    seen: set[str] = set()
    counts: dict[str, int] = {}
    for role, path in paths.items():
        count = 0
        for material_id, _calculations in iter_bz2_object(path):
            if material_id in seen:
                raise ValueError(f"duplicate Alexandria path identity: {material_id}")
            seen.add(material_id)
            count += 1
        counts[role] = count
    return seen, counts


def build_source_audit(
    *,
    shard_0000_path: Path,
    shard_0001_path: Path,
    benchmark_ids_path: Path,
    database_dir: Path,
    output_dir: Path,
    workers: int = 2,
    require_formal_inputs: bool = True,
    expected_path_rows: int = 20_000,
) -> dict[str, object]:
    """Map and freeze source eligibility without emitting scientific labels."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(f"refusing existing output: {target}")
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive exact integer")
    if type(expected_path_rows) is not int or expected_path_rows <= 0:
        raise ValueError("expected_path_rows must be a positive exact integer")
    path_shards = {
        "pbe_0000": Path(shard_0000_path).resolve(),
        "pbe_0001": Path(shard_0001_path).resolve(),
    }
    benchmark = Path(benchmark_ids_path).resolve()
    database = Path(database_dir).resolve()
    if any(not path.is_file() for path in path_shards.values()) or not benchmark.is_file():
        raise FileNotFoundError("NEXT42 path shard or benchmark list is missing")
    if not database.is_dir():
        raise FileNotFoundError("NEXT42 Alexandria final database is missing")

    fixed_hashes = {
        **{role: _sha256_file(path) for role, path in path_shards.items()},
        "benchmarks_pbe": _sha256_file(benchmark),
    }
    if require_formal_inputs and fixed_hashes != dict(FROZEN_FORMAL_SHA256):
        raise ValueError("formal NEXT42 source identities differ")

    path_ids, shard_counts = _trajectory_ids(path_shards)
    if len(path_ids) != expected_path_rows:
        raise ValueError("NEXT42 path row count differs")
    benchmark_ids, benchmark_rows = _load_id_file(benchmark)
    database_shards = sorted(database.glob("alexandria_*.json.bz2"))
    if not database_shards:
        raise FileNotFoundError("no Alexandria final database shards found")
    database_identities, raw_locations = _scan_all(
        database_shards, path_ids, workers=workers
    )

    locations: dict[str, str] = {}
    for material_id, location in raw_locations:
        previous = locations.setdefault(material_id, location)
        if previous != location:
            raise ValueError(
                f"Alexandria path identity has conflicting locations: {material_id}"
            )
    missing = path_ids - set(locations)
    if missing:
        raise ValueError(f"missing {len(missing)} path identities from Alexandria database")
    unexpected = set(locations) - path_ids
    if unexpected:
        raise ValueError("source scanner returned non-path identities")

    rows: list[dict[str, object]] = []
    for material_id in sorted(path_ids):
        location = locations[material_id]
        source_family = classify_location(location)
        is_benchmark = material_id in benchmark_ids
        eligible, reason = source_qualification(
            source_family, official_benchmark=is_benchmark
        )
        rows.append(
            {
                "material_id": material_id,
                "source_family": source_family,
                "location": location,
                "official_benchmark": is_benchmark,
                "raw_x0_eligible": eligible,
                "qualification_reason": reason,
            }
        )
    table = pd.DataFrame(rows)
    if table.material_id.duplicated().any() or len(table) != expected_path_rows:
        raise ValueError("NEXT42 qualification table identity differs")
    reasons = Counter(table.qualification_reason.astype(str))
    families = Counter(table.source_family.astype(str))
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "evidence_role": "source qualification before converged endpoint evaluation",
        "scientific_labels_emitted": False,
        "selection_fields_emitted": list(table.columns),
        "source_qualification_frozen_before_endpoint_evaluation": True,
        "raw_trajectory_containers_include_endpoint_fields": True,
        "trajectory_endpoint_values_accessed_for_qualification": False,
        "final_database_containers_include_endpoint_fields": True,
        "final_database_fields_emitted": ["material_id", "location"],
        "raw_x0_source_families": sorted(RAW_X0_SOURCE_FAMILIES),
        "documented_mlip_source_families": sorted(DOCUMENTED_MLIP_SOURCE_FAMILIES),
        "documented_mlip_source_prefixes": list(DOCUMENTED_MLIP_SOURCE_PREFIXES),
        "provenance_evidence": dict(EVIDENCE_URLS),
        "counts": {
            "path_rows": len(table),
            "path_shards": shard_counts,
            "benchmark_file_rows": benchmark_rows,
            "benchmark_file_unique_ids": len(benchmark_ids),
            "official_benchmark_overlap": int(table.official_benchmark.sum()),
            "raw_x0_eligible": int(table.raw_x0_eligible.sum()),
            "qualification_reasons": dict(sorted(reasons.items())),
            "source_families": dict(sorted(families.items())),
        },
        "inputs": {
            "fixed_sha256": fixed_hashes,
            "paths": {
                role: {"path": str(path), "bytes": path.stat().st_size}
                for role, path in path_shards.items()
            },
            "benchmarks_pbe": {
                "path": str(benchmark),
                "bytes": benchmark.stat().st_size,
            },
            "database_dir": str(database),
            "database_shards": sorted(
                database_identities, key=lambda value: str(value["name"])
            ),
        },
        "executed_source_sha256": {
            "src/next16_alexandria_source_map.py": _sha256_file(
                Path(__file__).resolve().with_name("next16_alexandria_source_map.py")
            ),
            "src/next18_alexandria_holdout.py": _sha256_file(
                Path(__file__).resolve().with_name("next18_alexandria_holdout.py")
            ),
            "src/next42_alexandria_source_audit.py": _sha256_file(
                Path(__file__).resolve()
            ),
        },
        "scientific_improvement_claim": False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        output_path = staging / OUTPUT_NAME
        table.to_parquet(output_path, index=False)
        manifest["outputs_sha256"] = {OUTPUT_NAME: _sha256_file(output_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if {
            **{role: _sha256_file(path) for role, path in path_shards.items()},
            "benchmarks_pbe": _sha256_file(benchmark),
        } != fixed_hashes:
            raise RuntimeError("NEXT42 fixed source changed during publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbe-0000", type=Path, required=True)
    parser.add_argument("--pbe-0001", type=Path, required=True)
    parser.add_argument("--benchmarks-pbe", type=Path, required=True)
    parser.add_argument("--database-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    build_source_audit(
        shard_0000_path=args.pbe_0000,
        shard_0001_path=args.pbe_0001,
        benchmark_ids_path=args.benchmarks_pbe,
        database_dir=args.database_dir,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
