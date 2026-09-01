"""Run the frozen 0.15 eV/atom Basin-Hull candidate on geometry-only ELEMENTA x0."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import importlib
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Callable, Mapping, Sequence

from ase import Atoms
import numpy as np
import pandas as pd

from src.next11_geometry_only_frames import _load_archive_only
from src.next12_pauling_controls import (
    RULES as PAULING_RULES,
    _classical_features,
    _combined_decision,
    _rule_decision,
)
from src.next13d_acsc_dft_pairs import _json_bytes, _sha256_file, _strict_json
from src.next14_wbm_holdout import _publish_directory_no_replace
from src.next15_basin_hull import (
    DECISIONS,
    RELAX_ATOM_BUDGET,
    RELAX_FMAX_EV_PER_A,
    RELAX_MAX_PREDICTION_STEPS,
    _runtime_identity,
    _unsupported_snapshot,
    build_reference_entries,
    compute_hull_scores,
    needed_element_subspaces,
    validate_relaxed_snapshot,
)
from src.next16_elementa_holdout import PROTOCOL as HOLDOUT_PROTOCOL


PROTOCOL = "2026-08-02-next16-elementa-basin-hull-v1"
OUTPUT_NAME = "elementa_basin_hull_features.parquet"
MANIFEST_NAME = "MANIFEST.json"
NEXT16_THRESHOLD_EV_PER_ATOM = 0.15
PAULING_WORKERS = 6
FROZEN_FORMAL_SHA256: Mapping[str, str] = {
    "metadata": "e985f1540059c0a22e7daa6c19174f15a40363d06873d67a6ddb4d450d6426b2",
    "geometry_only_frames": "96b29264053eb14dba55b77dd34e07cf69428fd2917c0e82a20bc59c257886d3",
    "holdout_manifest": "d19ae89088534a891b8950b985a27ed520b6b4ab2535b3cafed6b52fcac9319a",
    "checkpoint": "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5",
    "mp_reference": "553d6272f049a8f4ec26e503b89751e2616dd3af53d086545f6ea00f317a361f",
}
FeatureCalculator = Callable[[Atoms], tuple[Mapping[str, object] | None, str | None]]


def next16_decision(score: object, *, supported: bool) -> str:
    if type(supported) is not bool:
        raise ValueError("supported must be an exact boolean")
    if not supported:
        return "ABSTAIN"
    try:
        value = float(score)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("supported NEXT16 score must be finite") from exc
    if not math.isfinite(value):
        raise ValueError("supported NEXT16 score must be finite")
    return "REJECT" if value >= NEXT16_THRESHOLD_EV_PER_ATOM else "KEEP"


def pauling_control(
    atoms: Atoms, *, calculator: FeatureCalculator = _classical_features
) -> dict[str, object]:
    try:
        features, error = calculator(atoms)
    except Exception as exc:
        features, error = None, f"calculator failed: {type(exc).__name__}: {exc}"
    values = dict(features) if isinstance(features, Mapping) else {}
    result: dict[str, object] = {"pauling_feature_error": error or ""}
    decisions: list[str] = []
    for name, rule in PAULING_RULES.items():
        value = values.get(str(rule["feature"]), np.nan)
        decision = _rule_decision(
            value,
            operator=str(rule["operator"]),
            threshold=float(rule["threshold"]),
        )
        result[f"{name}_value"] = value
        result[f"{name}_decision"] = decision
        decisions.append(decision)
    result["pauling_p2_p5_decision"] = _combined_decision(decisions)
    return result


def _load_holdout(
    metadata_path: Path, frames_path: Path, manifest_path: Path
) -> tuple[pd.DataFrame, list[Atoms], dict[str, object]]:
    metadata_data = metadata_path.read_bytes()
    manifest = _strict_json(manifest_path.read_bytes(), role="NEXT16 ELEMENTA holdout manifest")
    if (
        manifest.get("protocol") != HOLDOUT_PROTOCOL
        or manifest.get("input_role") != "unrelaxed_x0_geometry_only"
        or manifest.get("endpoint_bytes_read_by_execution") is not False
    ):
        raise ValueError("NEXT16 ELEMENTA holdout contract differs")
    outputs = manifest.get("outputs_sha256")
    expected = {
        metadata_path.name: _sha256_file(metadata_path),
        frames_path.name: _sha256_file(frames_path),
    }
    if not isinstance(outputs, Mapping) or any(outputs.get(name) != digest for name, digest in expected.items()):
        raise ValueError("NEXT16 ELEMENTA holdout output hash differs")
    metadata = pd.read_parquet(metadata_path)
    required = {"material_id", "rk", "formula", "natoms", "input_role"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"NEXT16 ELEMENTA metadata lacks columns: {sorted(missing)}")
    metadata = metadata.loc[:, ["material_id", "rk", "formula", "natoms", "input_role"]].copy()
    metadata["material_id"] = metadata["material_id"].astype(str)
    if metadata["material_id"].duplicated().any() or not metadata["input_role"].eq("unrelaxed_x0_geometry_only").all():
        raise ValueError("NEXT16 ELEMENTA metadata identity or input role differs")
    sids, structures = _load_archive_only(frames_path, tuple(metadata["material_id"]))
    if sids != metadata["material_id"].tolist():
        raise ValueError("NEXT16 ELEMENTA geometry order differs")
    if any(len(atoms) != int(n) for atoms, n in zip(structures, metadata["natoms"], strict=True)):
        raise ValueError("NEXT16 ELEMENTA atom count differs")
    return metadata, structures, manifest


def _production_pauling(structures: Sequence[Atoms]) -> list[dict[str, object]]:
    with ProcessPoolExecutor(max_workers=PAULING_WORKERS) as executor:
        raw = list(executor.map(_classical_features, structures, chunksize=8))
    return [
        pauling_control(atoms, calculator=lambda _atoms, value=value: value)
        for atoms, value in zip(structures, raw, strict=True)
    ]


def run_elementa_basin_hull_features(
    *,
    metadata_path: Path,
    frames_zip_path: Path,
    holdout_manifest_path: Path,
    checkpoint_path: Path,
    mp_reference_path: Path,
    output_dir: Path,
    device: str = "cuda:0",
) -> dict[str, object]:
    """Execute the frozen candidate without reading an ELEMENTA endpoint artifact."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(f"refusing existing output: {target}")
    paths = {
        "metadata": Path(metadata_path).resolve(),
        "geometry_only_frames": Path(frames_zip_path).resolve(),
        "holdout_manifest": Path(holdout_manifest_path).resolve(),
        "checkpoint": Path(checkpoint_path).resolve(),
        "mp_reference": Path(mp_reference_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    input_hashes = {role: _sha256_file(path) for role, path in paths.items()}
    if input_hashes != dict(FROZEN_FORMAL_SHA256):
        raise ValueError("formal NEXT16 ELEMENTA inputs differ")
    metadata, structures, _ = _load_holdout(
        paths["metadata"], paths["geometry_only_frames"], paths["holdout_manifest"]
    )

    started = time.perf_counter()
    pauling_rows = _production_pauling(structures)
    runtime = _runtime_identity(device)
    if runtime["mattersim_version"] != "1.2.3" or runtime["cuda_available"] is not True:
        raise RuntimeError("NEXT16 requires MatterSim 1.2.3 with CUDA")
    from mattersim.applications.batch_relax import BatchRelaxer
    from mattersim.forcefield import Potential

    potential = Potential.from_checkpoint(
        str(paths["checkpoint"]), device=device, load_training_state=False
    )
    relaxer = BatchRelaxer(
        potential,
        optimizer="FIRE",
        filter="FRECHETCELLFILTER",
        fmax=RELAX_FMAX_EV_PER_A,
        max_natoms_per_batch=RELAX_ATOM_BUDGET,
        max_n_steps=RELAX_MAX_PREDICTION_STEPS,
    )
    trajectories = relaxer.relax(structures)
    summaries: list[dict[str, object]] = []
    for index, base in enumerate(structures):
        trajectory = trajectories.get(index, [])
        if not trajectory:
            summaries.append(_unsupported_snapshot("missing relaxation trajectory", prediction_steps=1))
        else:
            summaries.append(
                validate_relaxed_snapshot(base, trajectory[-1], prediction_steps=len(trajectory))
            )
    del trajectories, relaxer, potential
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    systems = {frozenset(atoms.get_chemical_symbols()) for atoms in structures}
    required_spaces = needed_element_subspaces(systems)
    raw_reference = pd.read_json(paths["mp_reference"])
    if list(raw_reference.columns) != ["material_id", "entry"]:
        raise ValueError("MP reference JSON schema differs from the frozen source")
    source_reference_rows = len(raw_reference)
    reference_entries = build_reference_entries(
        zip(raw_reference["material_id"], raw_reference["entry"], strict=True),
        needed_subspaces=required_spaces,
    )
    del raw_reference
    gc.collect()
    supported_indices = [i for i, summary in enumerate(summaries) if summary["supported"] is True]
    scores = compute_hull_scores(
        reference_entries,
        [str(metadata.iloc[i]["formula"]) for i in supported_indices],
        [float(summaries[i]["energy_total_ev"]) for i in supported_indices],
    )
    score_by_index = dict(zip(supported_indices, scores, strict=True))

    rows: list[dict[str, object]] = []
    for index, upstream in enumerate(metadata.to_dict("records")):
        summary = summaries[index]
        score = score_by_index.get(index, np.nan)
        rows.append(
            {
                "material_id": str(upstream["material_id"]),
                "rk": str(upstream["rk"]),
                "formula": str(upstream["formula"]),
                "natoms": int(upstream["natoms"]),
                **pauling_rows[index],
                **summary,
                "basin_hull_score_ev_per_atom": score,
                "basin_hull_decision": next16_decision(score, supported=bool(summary["supported"])),
            }
        )
    table = pd.DataFrame(rows)
    if len(table) != len(metadata) or table["material_id"].nunique() != len(table):
        raise ValueError("NEXT16 output lost or duplicated rows")
    for column in ("pauling_p2_p5_decision", "basin_hull_decision"):
        if not set(table[column]).issubset(DECISIONS):
            raise ValueError(f"NEXT16 output contains an unknown decision in {column}")

    repo_root = Path(__file__).resolve().parents[1]
    source_paths = [
        Path(__file__).resolve(),
        repo_root / "src/next15_basin_hull.py",
        repo_root / "src/next12_pauling_controls.py",
        repo_root / "src/next11_geometry_only_frames.py",
    ]
    installed_batch_relax = Path(
        importlib.import_module("mattersim.applications.batch_relax").__file__
    ).resolve()
    source_hashes = {
        str(path.relative_to(repo_root)) if path.is_relative_to(repo_root) else str(path): _sha256_file(path)
        for path in [*source_paths, installed_batch_relax]
    }
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "evidence_role": "historical ELEMENTA external-source retrospective; candidate threshold selected on WBM",
        "elementa_endpoint_bytes_read_by_execution": False,
        "labels_previously_opened_elsewhere": True,
        "fresh_lockbox": False,
        "thresholds_refit": False,
        "scientific_improvement_claim": False,
        "rule": {
            "formula": "B64 = E_MatterSim_relaxed/N - E_raw_MP_hull(composition)",
            "threshold_ev_per_atom": NEXT16_THRESHOLD_EV_PER_ATOM,
            "comparison": ">=",
            "failure_policy": "ABSTAIN",
            "selection_origin": "post hoc WBM development sweep; frozen before this ELEMENTA execution",
        },
        "relaxation": {
            "optimizer": "FIRE",
            "filter": "FRECHETCELLFILTER",
            "fmax_ev_per_a": RELAX_FMAX_EV_PER_A,
            "max_prediction_steps": RELAX_MAX_PREDICTION_STEPS,
            "atom_budget": RELAX_ATOM_BUDGET,
        },
        "pauling_rules": {name: dict(rule) for name, rule in PAULING_RULES.items()},
        "counts": {
            "rows": len(table),
            "groups": int(table["rk"].nunique()),
            "supported": int(table["supported"].sum()),
            "abstained": int(table["basin_hull_decision"].eq("ABSTAIN").sum()),
            "kept": int(table["basin_hull_decision"].eq("KEEP").sum()),
            "rejected": int(table["basin_hull_decision"].eq("REJECT").sum()),
            "capped_at_max_steps": int(table["capped_at_max_steps"].sum()),
            "pauling_decisions": {
                value: int(table["pauling_p2_p5_decision"].eq(value).sum()) for value in DECISIONS
            },
            "mp_source_entries": source_reference_rows,
            "mp_retained_reference_entries": len(reference_entries),
            "target_element_systems": len(systems),
            "required_element_subspaces": len(required_spaces),
        },
        "execution": {"wall_time_seconds": time.perf_counter() - started},
        "runtime": runtime,
        "inputs_sha256": {
            role: {"path": str(paths[role]), "sha256": digest}
            for role, digest in input_hashes.items()
        },
        "executed_source_sha256": source_hashes,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        output_path = staging / OUTPUT_NAME
        table.to_parquet(output_path, index=False)
        manifest["outputs_sha256"] = {OUTPUT_NAME: _sha256_file(output_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--frames-zip", required=True, type=Path)
    parser.add_argument("--holdout-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--mp-reference", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args(argv)
    run_elementa_basin_hull_features(
        metadata_path=arguments.metadata,
        frames_zip_path=arguments.frames_zip,
        holdout_manifest_path=arguments.holdout_manifest,
        checkpoint_path=arguments.checkpoint,
        mp_reference_path=arguments.mp_reference,
        output_dir=arguments.output_dir,
        device=arguments.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
