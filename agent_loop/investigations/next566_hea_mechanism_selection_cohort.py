#!/usr/bin/env python3
"""Freeze three mechanism-law predictions on 4,000 new HEA identities."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from src.next11_geometry_only_frames import _write_deterministic_archive
from src.next19_feature_build import _publish_directory_no_replace, _sha256
import src.next551_hea_initial_cohort as n551
import src.next552_hea_analytic_feature_freeze as n552
import src.next560_hea_entropy_packing_cohort as n560
import src.next562_hea_stable_analytic_union_search as n562
import src.next565_hea_mechanism_formula_family as n565


PROTOCOL = "2026-08-14-next566-hea-mechanism-selection-cohort-v1"
EXPECTED_PER_FAMILY = 2_000
EXPECTED_ROWS = 4_000
METADATA_NAME = "next566_hea_x0_metadata.parquet"
GEOMETRY_NAME = "geometry_only_frames.zip"
PREDICTIONS_NAME = "next566_hea_mechanism_predictions.parquet"
MANIFEST_NAME = "MANIFEST.json"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def selection_hash(fid: str) -> str:
    return hashlib.sha256(f"NEXT566-v1|{fid}".encode()).hexdigest()


def build_cohort(
    *, source_csv: Path, next551_dir: Path, next560_dir: Path,
    next565_dir: Path, design_path: Path, output_dir: Path,
) -> dict[str, object]:
    source_csv = Path(source_csv).resolve()
    root551, root560, root565 = map(
        lambda value: Path(value).resolve(), (next551_dir, next560_dir, next565_dir)
    )
    design_path, target = Path(design_path).resolve(), Path(output_dir).resolve()
    paths = {
        "source_csv": source_csv,
        "design": design_path,
        "next551_manifest": root551 / n551.MANIFEST_NAME,
        "next551_metadata": root551 / n551.METADATA_NAME,
        "next560_manifest": root560 / n560.MANIFEST_NAME,
        "next560_metadata": root560 / n560.METADATA_NAME,
        "next565_manifest": root565 / n565.MANIFEST_NAME,
        "next565_formula": root565 / n565.FORMULA_NAME,
        "next565_source": Path(n565.__file__).resolve(),
        "composition_source": Path(n562.__file__).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT566 input is missing")
    if _sha256(source_csv) != n551.SOURCE_SHA256 or _sha256(design_path) != n565.DESIGN_SHA256:
        raise ValueError("NEXT566 formal source or design differs")
    manifest565 = json.loads(paths["next565_manifest"].read_text())
    if (
        manifest565.get("protocol") != n565.PROTOCOL
        or manifest565.get("next566_cohort_authorized") is not True
        or manifest565.get("next566_endpoint_values_opened") is not False
        or manifest565.get("outputs_sha256", {}).get(n565.FORMULA_NAME)
        != _sha256(paths["next565_formula"])
    ):
        raise ValueError("NEXT566 frozen family identity differs")
    formula = json.loads(paths["next565_formula"].read_text())
    if tuple(formula.get("candidate_order", [])) != n565.CANDIDATE_NAMES:
        raise ValueError("NEXT566 candidate family differs")
    used = set(pd.read_parquet(paths["next551_metadata"])["fid"].astype(str))
    used.update(pd.read_parquet(paths["next560_metadata"])["fid"].astype(str))
    if len(used) != n551.EXPECTED_SELECTED_ROWS + n560.EXPECTED_ROWS:
        raise ValueError("NEXT566 exclusion identity differs")
    eligible, source_stats = n551._scan_label_free_metadata(source_csv)
    remaining = [row for row in eligible if str(row["fid"]) not in used]
    selected: list[dict[str, object]] = []
    for family in ("ordered", "sqs"):
        members = [row for row in remaining if str(row["size_family"]) == family]
        members.sort(key=lambda row: (selection_hash(str(row["fid"])), str(row["fid"])))
        if len(members) < EXPECTED_PER_FAMILY:
            raise ValueError(f"NEXT566 {family} source population differs")
        selected.extend(dict(row) for row in members[:EXPECTED_PER_FAMILY])
    selected.sort(key=lambda row: str(row["fid"]))
    if len(selected) != EXPECTED_ROWS or len({str(row["fid"]) for row in selected}) != EXPECTED_ROWS:
        raise ValueError("NEXT566 selected identity differs")
    frames, geometry_hashes = n551._extract_selected_initial_structures(source_csv, selected)
    rows: list[dict[str, object]] = []
    for row in selected:
        fid = str(row["fid"])
        comp = n562.composition_features(str(row["reduced_formula"]))
        rows.append({
            **row,
            "x0_geometry_sha256": geometry_hashes[fid],
            "natoms_decoded": len(frames[fid].atoms),
            **comp,
            "input_role": "composition_plus_unrelaxed_x0_geometry",
        })
    table = pd.DataFrame(rows)
    table["u_H"] = n552._midrank(table["composition_ideal_entropy"])
    table["u_M"] = n552._midrank(table["composition_atomic_mass_cv"])
    table["u_Z"] = n552._midrank(table["composition_atomic_number_cv"])
    for name, score in n565.candidate_scores(table["u_H"], table["u_M"], table["u_Z"]).items():
        table[name] = score
    family_counts = Counter(table["size_family"].astype(str))
    score_gates = {
        name: {
            "coverage": float(np.isfinite(table[name]).mean()),
            "unique_rounded_12": int(np.unique(np.round(table[name], 12)).size),
            "bounded": bool(np.all((table[name] >= 0) & (table[name] <= 1))),
        }
        for name in n565.CANDIDATE_NAMES
    }
    gates = {
        "rows": len(table),
        "families": dict(sorted(family_counts.items())),
        "chemical_systems": int(table["chemical_system"].nunique()),
        "unique_geometry_fraction": float(table["x0_geometry_sha256"].nunique() / len(table)),
        "candidate_scores": score_gates,
    }
    gates["passes"] = bool(
        len(table) == EXPECTED_ROWS
        and family_counts["ordered"] == EXPECTED_PER_FAMILY
        and family_counts["sqs"] == EXPECTED_PER_FAMILY
        and gates["unique_geometry_fraction"] >= 0.99
        and all(
            row["coverage"] == 1.0 and row["unique_rounded_12"] >= 500 and row["bounded"]
            for row in score_gates.values()
        )
    )
    if gates["passes"] is not True:
        raise RuntimeError(f"NEXT566 label-blind gates failed: {gates}")
    table = table.sort_values(["size_family", "fid"], kind="mergesort").reset_index(drop=True)

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
            "size_family", "source_row_index", "x0_geometry_sha256", "natoms_decoded",
            "input_role",
        ]
        prediction_columns = metadata_columns + [
            "composition_ideal_entropy", "composition_atomic_mass_cv",
            "composition_atomic_number_cv", "u_H", "u_M", "u_Z", *n565.CANDIDATE_NAMES,
        ]
        table[metadata_columns].to_parquet(metadata_path, index=False)
        table[prediction_columns].to_parquet(predictions_path, index=False)
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
                "src/next566_hea_mechanism_selection_cohort.py": source_hash
            },
            "endpoint_or_final_structure_columns_copied_or_decoded": False,
            "dft_values_opened": False,
            "dft_values_used_by_executable_formula": False,
            "model_or_proxy_potential_used": False,
            "coordinates_or_cell_modified": False,
            "selection_uses_endpoint_values": False,
            "next566b_endpoint_opening_authorized": True,
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
    parser.add_argument("--next560-dir", required=True, type=Path)
    parser.add_argument("--next565-dir", required=True, type=Path)
    parser.add_argument("--design-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = build_cohort(
        source_csv=args.source_csv, next551_dir=args.next551_dir,
        next560_dir=args.next560_dir, next565_dir=args.next565_dir,
        design_path=args.design_path, output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_cohort", "selection_hash"]
