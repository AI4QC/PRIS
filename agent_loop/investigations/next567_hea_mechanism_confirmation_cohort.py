#!/usr/bin/env python3
"""Freeze the selected mechanism law on 4,000 final HEA identities."""

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
import src.next566_hea_mechanism_selection_cohort as n566
import src.next566b_hea_mechanism_selection as n566b


PROTOCOL = "2026-08-14-next567-hea-mechanism-confirmation-cohort-v1"
EXPECTED_PER_FAMILY = 2_000
EXPECTED_ROWS = 4_000
METADATA_NAME = "next567_hea_x0_metadata.parquet"
GEOMETRY_NAME = "geometry_only_frames.zip"
PREDICTIONS_NAME = "next567_hea_mechanism_prediction.parquet"
MANIFEST_NAME = "MANIFEST.json"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def selection_hash(fid: str) -> str:
    return hashlib.sha256(f"NEXT567-v1|{fid}".encode()).hexdigest()


def build_cohort(
    *, source_csv: Path, next551_dir: Path, next560_dir: Path,
    next566_dir: Path, next566b_dir: Path, design_path: Path, output_dir: Path,
) -> dict[str, object]:
    source_csv = Path(source_csv).resolve()
    root551, root560, root566, root566b = map(
        lambda value: Path(value).resolve(),
        (next551_dir, next560_dir, next566_dir, next566b_dir),
    )
    design_path, target = Path(design_path).resolve(), Path(output_dir).resolve()
    paths = {
        "source_csv": source_csv,
        "design": design_path,
        "next551_metadata": root551 / n551.METADATA_NAME,
        "next560_metadata": root560 / n560.METADATA_NAME,
        "next566_metadata": root566 / n566.METADATA_NAME,
        "next566b_manifest": root566b / n566b.MANIFEST_NAME,
        "selected_formula": root566b / n566b.SELECTED_FORMULA_NAME,
        "selection_source": Path(n566b.__file__).resolve(),
        "composition_source": Path(n562.__file__).resolve(),
        "formula_source": Path(n565.__file__).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT567 input is missing")
    if _sha256(source_csv) != n551.SOURCE_SHA256 or _sha256(design_path) != n565.DESIGN_SHA256:
        raise ValueError("NEXT567 formal source or design differs")
    manifest566b = json.loads(paths["next566b_manifest"].read_text())
    selected_formula = json.loads(paths["selected_formula"].read_text())
    selected = manifest566b.get("selected_candidate")
    if (
        manifest566b.get("protocol") != n566b.PROTOCOL
        or manifest566b.get("selection_pass") is not True
        or manifest566b.get("next567_confirmation_cohort_authorized") is not True
        or manifest566b.get("outputs_sha256", {}).get(n566b.SELECTED_FORMULA_NAME)
        != _sha256(paths["selected_formula"])
        or selected not in n565.CANDIDATE_NAMES
        or selected_formula.get("selected_candidate") != selected
        or selected_formula.get("next567_endpoints_opened") is not False
    ):
        raise ValueError("NEXT567 selected formula identity differs")
    used: set[str] = set()
    for path in (paths["next551_metadata"], paths["next560_metadata"], paths["next566_metadata"]):
        used.update(pd.read_parquet(path)["fid"].astype(str))
    if len(used) != n551.EXPECTED_SELECTED_ROWS + n560.EXPECTED_ROWS + n566.EXPECTED_ROWS:
        raise ValueError("NEXT567 exclusion identity differs")
    eligible, source_stats = n551._scan_label_free_metadata(source_csv)
    remaining = [row for row in eligible if str(row["fid"]) not in used]
    selected_rows: list[dict[str, object]] = []
    for family in ("ordered", "sqs"):
        members = [row for row in remaining if str(row["size_family"]) == family]
        members.sort(key=lambda row: (selection_hash(str(row["fid"])), str(row["fid"])))
        if len(members) < EXPECTED_PER_FAMILY:
            raise ValueError(f"NEXT567 {family} source population differs")
        selected_rows.extend(dict(row) for row in members[:EXPECTED_PER_FAMILY])
    selected_rows.sort(key=lambda row: str(row["fid"]))
    if len(selected_rows) != EXPECTED_ROWS:
        raise ValueError("NEXT567 selected cohort identity differs")
    frames, geometry_hashes = n551._extract_selected_initial_structures(source_csv, selected_rows)
    rows: list[dict[str, object]] = []
    for row in selected_rows:
        fid = str(row["fid"])
        comp = n562.composition_features(str(row["reduced_formula"]))
        rows.append({
            **row, "x0_geometry_sha256": geometry_hashes[fid],
            "natoms_decoded": len(frames[fid].atoms), **comp,
            "input_role": "composition_plus_unrelaxed_x0_geometry",
        })
    table = pd.DataFrame(rows)
    table["u_H"] = n552._midrank(table["composition_ideal_entropy"])
    table["u_M"] = n552._midrank(table["composition_atomic_mass_cv"])
    table["u_Z"] = n552._midrank(table["composition_atomic_number_cv"])
    all_scores = n565.candidate_scores(table["u_H"], table["u_M"], table["u_Z"])
    table[str(selected)] = all_scores[str(selected)]
    family_counts = Counter(table["size_family"].astype(str))
    score = table[str(selected)].to_numpy(float)
    gates = {
        "rows": len(table), "families": dict(sorted(family_counts.items())),
        "chemical_systems": int(table["chemical_system"].nunique()),
        "unique_geometry_fraction": float(table["x0_geometry_sha256"].nunique() / len(table)),
        "selected_candidate": selected,
        "score_coverage": float(np.isfinite(score).mean()),
        "score_unique_rounded_12": int(np.unique(np.round(score, 12)).size),
        "score_bounded": bool(np.all((score >= 0) & (score <= 1))),
    }
    gates["passes"] = bool(
        len(table) == EXPECTED_ROWS
        and family_counts["ordered"] == EXPECTED_PER_FAMILY
        and family_counts["sqs"] == EXPECTED_PER_FAMILY
        and gates["unique_geometry_fraction"] >= 0.99
        and gates["score_coverage"] == 1.0
        and gates["score_unique_rounded_12"] >= 500
        and gates["score_bounded"]
    )
    if gates["passes"] is not True:
        raise RuntimeError(f"NEXT567 label-blind gates failed: {gates}")
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
            "composition_atomic_number_cv", "u_H", "u_M", "u_Z", str(selected),
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
            "selected_candidate": selected,
            "source_counts": source_stats,
            "gates": gates,
            "inputs_sha256": {
                name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()
            },
            "outputs_sha256": outputs,
            "executed_source_sha256": {
                "src/next567_hea_mechanism_confirmation_cohort.py": source_hash
            },
            "endpoint_or_final_structure_columns_copied_or_decoded": False,
            "dft_values_opened": False,
            "dft_values_used_by_executable_formula": False,
            "model_or_proxy_potential_used": False,
            "coordinates_or_cell_modified": False,
            "selection_uses_endpoint_values": False,
            "next568_endpoint_opening_authorized": True,
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
    parser.add_argument("--next566-dir", required=True, type=Path)
    parser.add_argument("--next566b-dir", required=True, type=Path)
    parser.add_argument("--design-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = build_cohort(
        source_csv=args.source_csv, next551_dir=args.next551_dir,
        next560_dir=args.next560_dir, next566_dir=args.next566_dir,
        next566b_dir=args.next566b_dir, design_path=args.design_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_cohort", "selection_hash"]
