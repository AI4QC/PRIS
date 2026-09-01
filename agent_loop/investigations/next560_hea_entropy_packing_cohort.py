#!/usr/bin/env python3
"""Freeze EPCU predictions on 2,000 new HEA identities before endpoint opening."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.next11_geometry_only_frames import _write_deterministic_archive
from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next546_lisi_analytic_mechanism_search import primitive_geometry_features
import src.next551_hea_initial_cohort as n551
import src.next559_hea_entropy_packing_discovery as n559


PROTOCOL = "2026-08-14-next560-hea-entropy-packing-cohort-v1"
EXPECTED_UNSEEN_SYSTEM_ROWS = 425
KNOWN_ORDERED_ROWS = 800
KNOWN_SQS_ROWS = 775
EXPECTED_ROWS = 2_000
METADATA_NAME = "next560_hea_x0_metadata.parquet"
GEOMETRY_NAME = "geometry_only_frames.zip"
PREDICTIONS_NAME = "next560_hea_epcu_predictions.parquet"
MANIFEST_NAME = "MANIFEST.json"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _known_hash(fid: str) -> str:
    return __import__("hashlib").sha256(f"NEXT560-known-v1|{fid}".encode()).hexdigest()


def _midrank(values: object, reverse: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (rankdata(-values if reverse else values, method="average") - 0.5) / len(values)


def build_cohort(
    *, source_csv: Path, next551_dir: Path, next559_dir: Path,
    design_path: Path, output_dir: Path,
) -> dict[str, object]:
    source_csv = Path(source_csv).resolve()
    root551, root559 = Path(next551_dir).resolve(), Path(next559_dir).resolve()
    design_path, target = Path(design_path).resolve(), Path(output_dir).resolve()
    paths = {
        "source_csv": source_csv,
        "design": design_path,
        "next551_metadata": root551 / n551.METADATA_NAME,
        "next551_manifest": root551 / n551.MANIFEST_NAME,
        "next559_manifest": root559 / n559.MANIFEST_NAME,
        "next559_formula": root559 / n559.FORMULA_NAME,
        "next559_source": Path(n559.__file__).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT560 input is missing")
    if _sha256(source_csv) != n551.SOURCE_SHA256 or _sha256(design_path) != n559.DESIGN_SHA256:
        raise ValueError("NEXT560 formal source or design differs")
    manifest559 = json.loads(paths["next559_manifest"].read_text())
    outputs559 = manifest559.get("outputs_sha256")
    if (
        manifest559.get("protocol") != n559.PROTOCOL
        or manifest559.get("next560_cohort_authorized") is not True
        or manifest559.get("next560_endpoint_values_opened") is not False
        or not isinstance(outputs559, dict)
        or outputs559.get(n559.FORMULA_NAME) != _sha256(paths["next559_formula"])
    ):
        raise ValueError("NEXT560 frozen discovery identity differs")
    prior = pd.read_parquet(paths["next551_metadata"])
    used_fids = set(prior["fid"].astype(str))
    used_systems = set(prior["chemical_system"].astype(str))
    eligible, source_stats = n551._scan_label_free_metadata(source_csv)
    remaining = [row for row in eligible if str(row["fid"]) not in used_fids]
    unseen = [row for row in remaining if str(row["chemical_system"]) not in used_systems]
    known = [row for row in remaining if str(row["chemical_system"]) in used_systems]
    if len(unseen) != EXPECTED_UNSEEN_SYSTEM_ROWS or len({str(r['chemical_system']) for r in unseen}) != 40:
        raise ValueError("NEXT560 unseen-system source population differs")
    selected: list[dict[str, object]] = []
    for row in unseen:
        value = dict(row)
        value["replication_stratum"] = "unseen_chemical_system"
        selected.append(value)
    for family, count in (("ordered", KNOWN_ORDERED_ROWS), ("sqs", KNOWN_SQS_ROWS)):
        members = [row for row in known if row["size_family"] == family]
        members.sort(key=lambda row: (_known_hash(str(row["fid"])), str(row["fid"])))
        if len(members) < count:
            raise ValueError(f"NEXT560 known-system {family} population differs")
        for row in members[:count]:
            value = dict(row)
            value["replication_stratum"] = "new_identity_known_system"
            selected.append(value)
    selected.sort(key=lambda row: str(row["fid"]))
    if len(selected) != EXPECTED_ROWS or len({str(row["fid"]) for row in selected}) != EXPECTED_ROWS:
        raise ValueError("NEXT560 selected cohort identity differs")
    frames, geometry_hashes = n551._extract_selected_initial_structures(source_csv, selected)
    rows: list[dict[str, object]] = []
    for row in selected:
        fid = str(row["fid"])
        atoms = frames[fid].atoms
        primitive = primitive_geometry_features(atoms)
        rows.append(
            {
                **row,
                "x0_geometry_sha256": geometry_hashes[fid],
                "natoms_decoded": len(atoms),
                n559.ENTROPY_RAW: n559.composition_entropy(atoms),
                n559.PACKING_RAW: primitive["primitive_covalent_packing_fraction"],
                "input_role": "unrelaxed_x0_geometry_only",
            }
        )
    table = pd.DataFrame(rows)
    table[n559.ENTROPY_RISK] = _midrank(table[n559.ENTROPY_RAW])
    table[n559.PACKING_RISK] = _midrank(table[n559.PACKING_RAW], reverse=True)
    table[n559.SCORE] = n559.entropy_packing_union(
        table[n559.ENTROPY_RISK], table[n559.PACKING_RISK]
    )
    stratum_counts = Counter(table["replication_stratum"].astype(str))
    family_counts = Counter(table["size_family"].astype(str))
    score = table[n559.SCORE].to_numpy(float)
    gates = {
        "rows": len(table),
        "strata": dict(sorted(stratum_counts.items())),
        "families": dict(sorted(family_counts.items())),
        "chemical_systems": int(table["chemical_system"].nunique()),
        "unique_geometry_fraction": float(table["x0_geometry_sha256"].nunique() / len(table)),
        "score_coverage": float(np.isfinite(score).mean()),
        "score_unique_rounded_12": int(np.unique(np.round(score, 12)).size),
        "score_bounded": bool(np.all((score >= 0) & (score <= 1))),
    }
    gates["passes"] = bool(
        len(table) == EXPECTED_ROWS
        and stratum_counts["unseen_chemical_system"] == EXPECTED_UNSEEN_SYSTEM_ROWS
        and stratum_counts["new_identity_known_system"] == KNOWN_ORDERED_ROWS + KNOWN_SQS_ROWS
        and family_counts["ordered"] == 1_019
        and family_counts["sqs"] == 981
        and gates["unique_geometry_fraction"] >= 0.99
        and gates["score_coverage"] == 1.0
        and gates["score_unique_rounded_12"] >= 500
        and gates["score_bounded"]
    )
    if gates["passes"] is not True:
        raise RuntimeError(f"NEXT560 label-blind gates failed: {gates}")
    table = table.sort_values(["replication_stratum", "size_family", "fid"], kind="mergesort")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    try:
        metadata_path, predictions_path, geometry_path = (
            staging / METADATA_NAME, staging / PREDICTIONS_NAME, staging / GEOMETRY_NAME
        )
        metadata_columns = [
            "fid", "reduced_formula", "chemical_system", "nelements", "nions",
            "size_family", "replication_stratum", "source_row_index", "x0_geometry_sha256",
            "natoms_decoded", "input_role",
        ]
        table[metadata_columns].to_parquet(metadata_path, index=False)
        table.to_parquet(predictions_path, index=False)
        _write_deterministic_archive(geometry_path, frames)
        outputs = {
            METADATA_NAME: _sha256(metadata_path), PREDICTIONS_NAME: _sha256(predictions_path),
            GEOMETRY_NAME: _sha256(geometry_path),
        }
        manifest = {
            "protocol": PROTOCOL,
            "source_counts": source_stats,
            "gates": gates,
            "inputs_sha256": {
                name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()
            },
            "outputs_sha256": outputs,
            "executed_source_sha256": {
                "src/next560_hea_entropy_packing_cohort.py": source_hash
            },
            "endpoint_or_final_structure_columns_copied_or_decoded": False,
            "dft_values_opened": False,
            "dft_values_used_by_executable_formula": False,
            "model_or_proxy_potential_used": False,
            "coordinates_or_cell_modified": False,
            "selection_uses_endpoint_values": False,
            "next561_endpoint_opening_authorized": True,
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", required=True, type=Path)
    parser.add_argument("--next551-dir", required=True, type=Path)
    parser.add_argument("--next559-dir", required=True, type=Path)
    parser.add_argument("--design-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = build_cohort(
        source_csv=args.source_csv, next551_dir=args.next551_dir,
        next559_dir=args.next559_dir, design_path=args.design_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
