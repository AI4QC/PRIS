"""Compute label-free strict full-cell MatterSim gaps for complete candidate groups."""

from __future__ import annotations

import argparse
import gc
import importlib
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.next13d_acsc_dft_pairs import _json_bytes, _sha256_file
from src.next14_wbm_holdout import _publish_directory_no_replace
from src.next15_basin_hull import (
    RELAX_ATOM_BUDGET,
    _runtime_identity,
    _unsupported_snapshot,
    validate_relaxed_snapshot,
)
from src.next16_elementa_basin_hull import _load_holdout


PROTOCOL = "2026-08-02-next17-elementa-strict-relax-gap-v1"
OUTPUT_NAME = "elementa_strict_relax_gap_features.parquet"
MANIFEST_NAME = "MANIFEST.json"
STRICT_RELAX_FMAX_EV_PER_A = 0.005
STRICT_RELAX_MAX_PREDICTION_STEPS = 64
FROZEN_FORMAL_SHA256: Mapping[str, str] = {
    "metadata": "e985f1540059c0a22e7daa6c19174f15a40363d06873d67a6ddb4d450d6426b2",
    "geometry_only_frames": "96b29264053eb14dba55b77dd34e07cf69428fd2917c0e82a20bc59c257886d3",
    "holdout_manifest": "d19ae89088534a891b8950b985a27ed520b6b4ab2535b3cafed6b52fcac9319a",
    "checkpoint": "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5",
}


def add_group_relative_gaps(table: pd.DataFrame) -> pd.DataFrame:
    """Add fail-open within-composition energy gaps without any endpoint label."""

    required = {"material_id", "rk", "supported", "energy_ev_per_atom"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"strict relaxation table lacks columns: {sorted(missing)}")
    result = table.copy()
    result["material_id"] = result["material_id"].astype(str)
    result["rk"] = result["rk"].astype(str)
    if result["material_id"].isna().any() or result["material_id"].duplicated().any():
        raise ValueError("strict relaxation material identities must be unique")
    supported = result["supported"]
    if not supported.map(lambda value: type(value) in (bool, np.bool_)).all():
        raise ValueError("supported values must be exact booleans")
    supported = supported.astype(bool)
    energy = pd.to_numeric(result["energy_ev_per_atom"], errors="coerce")
    if not np.isfinite(energy[supported].to_numpy(float)).all():
        raise ValueError("a supported energy is not finite")
    group_supported = supported.groupby(result["rk"], sort=False).transform("all")
    finite = np.isfinite(energy.to_numpy(float))
    group_finite = pd.Series(finite, index=result.index).groupby(
        result["rk"], sort=False
    ).transform("all")
    group_supported = group_supported & group_finite
    minimum = energy.where(group_supported).groupby(result["rk"], sort=False).transform("min")
    gap = (energy - minimum).where(group_supported)
    if (gap.dropna() < -1.0e-10).any():
        raise ValueError("strict relative gap is negative")
    gap = gap.mask(gap.abs() <= 1.0e-10, 0.0)
    result["strict_group_supported"] = group_supported.astype(bool)
    result["strict_relative_gap_ev_per_atom"] = gap.astype(float)
    return result


def run_strict_relax_gap_features(
    *,
    metadata_path: Path,
    frames_zip_path: Path,
    holdout_manifest_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    device: str = "cuda:0",
) -> dict[str, object]:
    """Run the frozen strict relaxation without reading endpoint or MP-hull data."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(f"refusing existing output: {target}")
    paths = {
        "metadata": Path(metadata_path).resolve(),
        "geometry_only_frames": Path(frames_zip_path).resolve(),
        "holdout_manifest": Path(holdout_manifest_path).resolve(),
        "checkpoint": Path(checkpoint_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    input_hashes = {role: _sha256_file(path) for role, path in paths.items()}
    if input_hashes != dict(FROZEN_FORMAL_SHA256):
        raise ValueError("formal NEXT17 inputs differ")
    metadata, structures, _ = _load_holdout(
        paths["metadata"], paths["geometry_only_frames"], paths["holdout_manifest"]
    )

    started = time.perf_counter()
    runtime = _runtime_identity(device)
    if runtime["mattersim_version"] != "1.2.3" or runtime["cuda_available"] is not True:
        raise RuntimeError("NEXT17 requires MatterSim 1.2.3 with CUDA")
    from mattersim.applications.batch_relax import BatchRelaxer
    from mattersim.forcefield import Potential

    potential = Potential.from_checkpoint(
        str(paths["checkpoint"]), device=device, load_training_state=False
    )
    relaxer = BatchRelaxer(
        potential,
        optimizer="FIRE",
        filter="FRECHETCELLFILTER",
        fmax=STRICT_RELAX_FMAX_EV_PER_A,
        max_natoms_per_batch=RELAX_ATOM_BUDGET,
        max_n_steps=STRICT_RELAX_MAX_PREDICTION_STEPS,
    )
    trajectories = relaxer.relax(structures)
    summaries: list[dict[str, object]] = []
    for index, base in enumerate(structures):
        trajectory = trajectories.get(index, [])
        if not trajectory:
            summaries.append(
                _unsupported_snapshot("missing relaxation trajectory", prediction_steps=1)
            )
        else:
            summaries.append(
                validate_relaxed_snapshot(
                    base, trajectory[-1], prediction_steps=len(trajectory)
                )
            )
    del trajectories, relaxer, potential
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    rows = [
        {
            "material_id": str(upstream["material_id"]),
            "rk": str(upstream["rk"]),
            "formula": str(upstream["formula"]),
            "natoms": int(upstream["natoms"]),
            **summary,
        }
        for upstream, summary in zip(
            metadata.to_dict("records"), summaries, strict=True
        )
    ]
    table = add_group_relative_gaps(pd.DataFrame(rows))
    if len(table) != len(metadata) or table["material_id"].nunique() != len(table):
        raise ValueError("NEXT17 output lost or duplicated rows")

    repo_root = Path(__file__).resolve().parents[1]
    installed_batch_relax = Path(
        importlib.import_module("mattersim.applications.batch_relax").__file__
    ).resolve()
    source_paths = [
        Path(__file__).resolve(),
        repo_root / "src/next16_elementa_basin_hull.py",
        repo_root / "src/next15_basin_hull.py",
        repo_root / "src/next11_geometry_only_frames.py",
        installed_batch_relax,
    ]
    source_hashes = {
        str(path.relative_to(repo_root)) if path.is_relative_to(repo_root) else str(path): _sha256_file(path)
        for path in source_paths
    }
    group_supported = table["strict_group_supported"].astype(bool)
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "evidence_role": "historically exposed ELEMENTA development feature execution",
        "input_role": "unrelaxed_x0_geometry_only",
        "elementa_endpoint_bytes_read_by_execution": False,
        "mp_hull_bytes_read_by_execution": False,
        "labels_previously_opened_elsewhere": True,
        "fresh_lockbox": False,
        "threshold_selected": False,
        "thresholds_refit": False,
        "scientific_improvement_claim": False,
        "formula": (
            "R64s(i) = E_MatterSim_strict_relaxed(i)/N_i "
            "- min_j_in_same_composition E_MatterSim_strict_relaxed(j)/N_j"
        ),
        "relaxation": {
            "optimizer": "FIRE",
            "filter": "FRECHETCELLFILTER",
            "fmax_ev_per_a": STRICT_RELAX_FMAX_EV_PER_A,
            "max_prediction_steps": STRICT_RELAX_MAX_PREDICTION_STEPS,
            "atom_budget": RELAX_ATOM_BUDGET,
        },
        "support_policy": "any unsupported row makes its complete composition group unsupported",
        "counts": {
            "rows": len(table),
            "groups": int(table["rk"].nunique()),
            "supported_rows": int(table["supported"].astype(bool).sum()),
            "supported_groups": int(
                table.loc[group_supported, "rk"].nunique()
            ),
            "group_supported_rows": int(group_supported.sum()),
            "capped_at_max_steps": int(table["capped_at_max_steps"].astype(bool).sum()),
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
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args(argv)
    run_strict_relax_gap_features(
        metadata_path=arguments.metadata,
        frames_zip_path=arguments.frames_zip,
        holdout_manifest_path=arguments.holdout_manifest,
        checkpoint_path=arguments.checkpoint,
        output_dir=arguments.output_dir,
        device=arguments.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
