#!/usr/bin/env python3
"""Freeze NEXT23 and Pauling predictions on NEXT42 raw x0 structures."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import tempfile
import time

from ase import Atoms
import numpy as np
import pandas as pd

from src.next11_geometry_only_frames import _load_archive_only
from src.next12_pauling_controls import DECISIONS, RULES, _classical_features
from src.next19_feature_build import _publish_directory_no_replace, _sha256, _strict_json
from src.next39_next23_predictions import (
    FROZEN_RULE_SHA256,
    TERM_COLUMNS,
    _json_bytes,
    _load_frozen_rule,
    _pauling_values,
    apply_frozen_next23_rule,
    compute_next23_terms,
)
from src.next42_alexandria_cohort import (
    COHORT_NAME,
    GEOMETRY_NAME,
    INPUT_ROLE as COHORT_INPUT_ROLE,
    PROTOCOL as COHORT_PROTOCOL,
)


PROTOCOL = "2026-08-03-next42-next23-frozen-predictions-v1"
PREDICTIONS_NAME = "next42_next23_predictions.parquet"
MANIFEST_NAME = "MANIFEST.json"
FeatureCalculator = Callable[
    [Atoms], tuple[Mapping[str, object] | None, str | None]
]


def _validate_output_hash(
    manifest: Mapping[str, object], path: Path, *, role: str
) -> None:
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(path.name) != _sha256(path):
        raise ValueError(f"{role} output hash differs")


def _load_cohort(
    *, metadata_path: Path, frames_zip_path: Path, manifest_path: Path
) -> tuple[pd.DataFrame, list[Atoms], dict[str, object]]:
    manifest = _strict_json(manifest_path, role="NEXT42 cohort manifest")
    selection = manifest.get("selection")
    if (
        manifest.get("protocol") != COHORT_PROTOCOL
        or manifest.get("input_role") != COHORT_INPUT_ROLE
        or manifest.get("later_geometry_accessed") is not False
        or manifest.get("dft_values_read") is not False
        or manifest.get("mlip_prerelaxation_used") is not False
        or manifest.get("physical_relaxation_executed") is not False
        or not isinstance(selection, Mapping)
        or selection.get("endpoint_fields_used") is not False
        or selection.get("sampled") is not False
    ):
        raise ValueError("NEXT42 cohort crossed the prediction boundary")
    if metadata_path.name != COHORT_NAME or frames_zip_path.name != GEOMETRY_NAME:
        raise ValueError("NEXT42 cohort filenames differ")
    _validate_output_hash(manifest, metadata_path, role="cohort metadata")
    _validate_output_hash(manifest, frames_zip_path, role="cohort geometry")
    table = pd.read_parquet(metadata_path).sort_values(
        "material_id", kind="stable", ignore_index=True
    )
    required = {
        "material_id",
        "source_family",
        "source_shard",
        "formula",
        "reduced_formula",
        "natoms",
        "input_role",
    }
    if not required.issubset(table.columns):
        raise ValueError(f"NEXT42 cohort lacks columns: {sorted(required - set(table))}")
    if (
        table.empty
        or table.material_id.astype(str).duplicated().any()
        or not table.input_role.eq(COHORT_INPUT_ROLE).all()
    ):
        raise ValueError("NEXT42 cohort identity or role differs")
    ids = tuple(table.material_id.astype(str))
    loaded, atoms = _load_archive_only(frames_zip_path, ids)
    if loaded != list(ids) or any(
        len(structure) != int(natoms)
        for structure, natoms in zip(atoms, table.natoms, strict=True)
    ):
        raise ValueError("NEXT42 raw-x0 geometry identity differs")
    return table, atoms, manifest


def run_next42_predictions(
    *,
    metadata_path: Path,
    frames_zip_path: Path,
    cohort_manifest_path: Path,
    frozen_rule_path: Path,
    frozen_rule_manifest_path: Path,
    output_dir: Path,
    term_calculator: FeatureCalculator | None = None,
    pauling_feature_calculator: FeatureCalculator | None = None,
) -> dict[str, object]:
    """Seal all predictions before an Alexandria final geometry is opened."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = {
        "metadata": Path(metadata_path).resolve(),
        "geometry": Path(frames_zip_path).resolve(),
        "cohort_manifest": Path(cohort_manifest_path).resolve(),
        "frozen_rule": Path(frozen_rule_path).resolve(),
        "frozen_rule_manifest": Path(frozen_rule_manifest_path).resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT42 prediction input is missing")
    input_hashes = {name: _sha256(path) for name, path in paths.items()}
    metadata, structures, _cohort_manifest = _load_cohort(
        metadata_path=paths["metadata"],
        frames_zip_path=paths["geometry"],
        manifest_path=paths["cohort_manifest"],
    )
    rule, _rule_manifest = _load_frozen_rule(
        rule_path=paths["frozen_rule"], manifest_path=paths["frozen_rule_manifest"]
    )
    term_fn = compute_next23_terms if term_calculator is None else term_calculator
    pauling_fn = (
        _classical_features
        if pauling_feature_calculator is None
        else pauling_feature_calculator
    )
    production = bool(
        term_calculator is None
        and pauling_feature_calculator is None
        and input_hashes["frozen_rule"] == FROZEN_RULE_SHA256
    )

    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    for upstream, atoms in zip(metadata.to_dict("records"), structures, strict=True):
        try:
            raw_terms, error = term_fn(atoms)
        except Exception as exc:
            raw_terms, error = None, f"calculator failed: {type(exc).__name__}: {exc}"
        terms = dict(raw_terms) if isinstance(raw_terms, Mapping) else {}
        decision = apply_frozen_next23_rule(terms, rule)
        row: dict[str, object] = {
            "material_id": str(upstream["material_id"]),
            "source_family": str(upstream["source_family"]),
            "natoms": int(upstream["natoms"]),
            TERM_COLUMNS[0]: terms.get(TERM_COLUMNS[0], np.nan),
            TERM_COLUMNS[1]: terms.get(TERM_COLUMNS[1], np.nan),
            "next23_feature_error": error,
            "next23_supported": bool(decision["supported"]),
            "next23_score": decision["score"],
            "next23_reject": bool(decision["reject"]),
        }
        row.update(_pauling_values(atoms, pauling_fn))
        rows.append(row)
    elapsed = time.perf_counter() - started
    table = pd.DataFrame(rows).sort_values(
        "material_id", kind="stable", ignore_index=True
    )
    if len(table) != len(metadata) or table.material_id.duplicated().any():
        raise RuntimeError("NEXT42 prediction identity accounting differs")
    if bool((table.next23_reject & ~table.next23_supported).any()):
        raise RuntimeError("NEXT42 unsupported rows did not fail open")

    repository = Path(__file__).resolve().parents[1]
    source_names = (
        "src/next12_pauling_controls.py",
        "src/next19_valence_transport.py",
        "src/next20_valence_rigidity.py",
        "src/next22_bond_valence_equilibrium.py",
        "src/next23_relaxation_rule.py",
        "src/next32_inorganic_response_features.py",
        "src/next39_next23_predictions.py",
        "src/next42_alexandria_cohort.py",
        "src/next42_next23_predictions.py",
    )
    source_hashes = {name: _sha256(repository / name) for name in source_names}
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "evidence_role": "predictions frozen before converged Alexandria endpoints open",
        "input_role": "one_raw_pre_dft_pre_mlip_x0_only",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "later_geometry_opened": False,
        "dft_values_read": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
        "coordinates_or_cell_modified": False,
        "same_composition_candidates_used": False,
        "thresholds_refit": False,
        "frozen_rule_sha256": input_hashes["frozen_rule"],
        "frozen_candidate": rule["selected_candidate"],
        "frozen_terms": rule["selected_terms"],
        "frozen_threshold": rule["threshold"],
        "pauling_rules": {name: dict(value) for name, value in RULES.items()},
        "counts": {
            "rows": len(table),
            "next23_supported": int(table.next23_supported.sum()),
            "next23_rejected": int(table.next23_reject.sum()),
            "pauling": {
                decision: int(table.pauling_p2_p5_decision.eq(decision).sum())
                for decision in DECISIONS
            },
        },
        "execution": {"wall_time_seconds": elapsed},
        "inputs_sha256": input_hashes,
        "executed_source_sha256": source_hashes,
        "production_protocol_eligible": production,
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        output_path = staging / PREDICTIONS_NAME
        table.to_parquet(output_path, index=False)
        manifest["outputs_sha256"] = {PREDICTIONS_NAME: _sha256(output_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if any(_sha256(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT42 prediction input changed during publication")
        if any(
            _sha256(repository / name) != digest
            for name, digest in source_hashes.items()
        ):
            raise RuntimeError("NEXT42 prediction source changed during publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--frozen-rule", type=Path, required=True)
    parser.add_argument("--frozen-rule-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run_next42_predictions(
        metadata_path=args.metadata,
        frames_zip_path=args.geometry,
        cohort_manifest_path=args.cohort_manifest,
        frozen_rule_path=args.frozen_rule,
        frozen_rule_manifest_path=args.frozen_rule_manifest,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
