#!/usr/bin/env python3
"""Run a label-free MatterSim single-point baseline on strict ELEMENTA x0 frames."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Callable, Sequence
import zipfile

from ase import Atoms
import numpy as np
import pandas as pd

from src.next6_wbm_build import sha256_file
from src.next6_wbm_features import parse_extxyz


EnergyPredictor = Callable[[list[Atoms]], Sequence[float]]


def frame_to_atoms(text: str) -> Atoms:
    """Sanitize an extxyz frame to species, coordinates, cell, and PBC only."""

    frame = parse_extxyz(text)
    return Atoms(
        symbols=frame.species,
        positions=frame.cart_coords,
        cell=frame.lattice,
        pbc=True,
    )


def _mattersim_predictor(
    checkpoint: Path,
    *,
    device: str,
    batch_size: int,
) -> EnergyPredictor:
    from mattersim.datasets.utils.build import build_dataloader
    from mattersim.forcefield import Potential

    potential = Potential.from_checkpoint(
        str(checkpoint), device=device, load_training_state=False
    )
    cutoff = float(potential.model.model_args.get("cutoff", 5.0))
    threebody_cutoff = float(
        potential.model.model_args.get("threebody_cutoff", 4.0)
    )

    def predict(atoms: list[Atoms]) -> Sequence[float]:
        loader = build_dataloader(
            atoms,
            cutoff=cutoff,
            threebody_cutoff=threebody_cutoff,
            batch_size=batch_size,
            only_inference=True,
        )
        energies, _, _ = potential.predict_properties(
            loader, include_forces=False, include_stresses=False
        )
        return energies

    return predict


def _predict_with_fallback(
    atoms: list[Atoms],
    predictor: EnergyPredictor,
) -> tuple[list[float], list[str]]:
    try:
        energy = [float(value) for value in predictor(atoms)]
        if len(energy) != len(atoms):
            raise ValueError("predictor output length mismatch")
        return energy, [""] * len(atoms)
    except Exception as batch_error:
        energies: list[float] = []
        errors: list[str] = []
        for structure in atoms:
            try:
                values = [float(value) for value in predictor([structure])]
                if len(values) != 1:
                    raise ValueError("single prediction output length mismatch")
                energies.append(values[0])
                errors.append("")
            except Exception as exc:
                energies.append(np.nan)
                errors.append(f"{type(exc).__name__}:{exc}")
        if all(errors):
            prefix = f"batch={type(batch_error).__name__}:{batch_error};"
            errors = [prefix + error for error in errors]
        return energies, errors


def run_mattersim_baseline(
    elementa_dir: Path,
    p9_dir: Path,
    output_dir: Path,
    *,
    checkpoint: Path,
    device: str,
    batch_size: int,
    chunk_size: int,
    predictor: EnergyPredictor | None = None,
) -> dict[str, object]:
    """Predict total/x0 per-atom energy without reading any endpoint labels."""

    elementa_dir = Path(elementa_dir)
    p9_dir = Path(p9_dir)
    output_dir = Path(output_dir)
    checkpoint = Path(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_path = elementa_dir / "elementa_initial_frames.zip"
    metadata_path = p9_dir / "elementa_x0_p9_features.parquet"
    metadata = pd.read_parquet(
        metadata_path,
        columns=[
            "sid",
            "rk",
            "material",
            "strict_x0_ok",
            "initial_ionic_step",
        ],
    )
    if metadata["sid"].duplicated().any():
        raise ValueError("MatterSim metadata sid must be unique")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if batch_size <= 0 or chunk_size <= 0:
        raise ValueError("batch_size and chunk_size must be positive")
    energy_predictor = predictor or _mattersim_predictor(
        checkpoint, device=device, batch_size=batch_size
    )

    output_rows: list[dict[str, object]] = []
    with zipfile.ZipFile(frames_path) as archive:
        member_by_sid = {Path(name).stem: name for name in archive.namelist()}
        if set(member_by_sid) != set(metadata["sid"].astype(str)):
            raise ValueError("MatterSim frame and metadata sid sets differ")
        records = metadata.to_dict("records")
        for start in range(0, len(records), chunk_size):
            chunk = records[start : start + chunk_size]
            strict_records: list[dict[str, object]] = []
            atoms: list[Atoms] = []
            for record in chunk:
                if not bool(record["strict_x0_ok"]):
                    output_rows.append(
                        {
                            **record,
                            "input_role": "trajectory_earliest_available",
                            "mattersim_feature_ok": False,
                            "mattersim_feature_error": "nonzero_initial_ionic_step",
                            "mattersim_energy_total": np.nan,
                            "mattersim_energy_per_atom": np.nan,
                        }
                    )
                    continue
                sid = str(record["sid"])
                try:
                    structure = frame_to_atoms(
                        archive.read(member_by_sid[sid]).decode("utf-8")
                    )
                    strict_records.append({**record, "n_atoms": len(structure)})
                    atoms.append(structure)
                except Exception as exc:
                    output_rows.append(
                        {
                            **record,
                            "input_role": "unrelaxed_x0_only",
                            "mattersim_feature_ok": False,
                            "mattersim_feature_error": f"{type(exc).__name__}:{exc}",
                            "mattersim_energy_total": np.nan,
                            "mattersim_energy_per_atom": np.nan,
                        }
                    )
            if atoms:
                energies, errors = _predict_with_fallback(atoms, energy_predictor)
                for record, energy, error in zip(strict_records, energies, errors):
                    ok = not error and np.isfinite(energy)
                    output_rows.append(
                        {
                            "sid": record["sid"],
                            "rk": record["rk"],
                            "material": record["material"],
                            "strict_x0_ok": record["strict_x0_ok"],
                            "initial_ionic_step": record["initial_ionic_step"],
                            "input_role": "unrelaxed_x0_only",
                            "mattersim_feature_ok": bool(ok),
                            "mattersim_feature_error": error
                            if error
                            else ("" if ok else "nonfinite_energy"),
                            "mattersim_energy_total": energy if ok else np.nan,
                            "mattersim_energy_per_atom": (
                                energy / int(record["n_atoms"]) if ok else np.nan
                            ),
                        }
                    )
            print(
                json.dumps(
                    {"processed": min(start + chunk_size, len(records)), "total": len(records)}
                ),
                flush=True,
            )

    table = pd.DataFrame(output_rows)
    if table["sid"].duplicated().any() or set(table["sid"]) != set(metadata["sid"]):
        raise ValueError("MatterSim output sid mismatch")
    table = table.sort_values("sid", kind="stable").reset_index(drop=True)
    output_path = output_dir / "mattersim_x0_predictions.parquet"
    table.to_parquet(output_path, index=False)
    try:
        version = importlib.metadata.version("mattersim")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    manifest: dict[str, object] = {
        "protocol": "2026-08-01-mattersim-x0-baseline-v1",
        "evidence_role": "strong MLIP baseline; not an interpretable law",
        "model": {
            "package": "mattersim",
            "version": version,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "device": device,
            "batch_size": int(batch_size),
            "chunk_size": int(chunk_size),
        },
        "counts": {
            "input_rows": len(metadata),
            "strict_x0_rows": int(metadata["strict_x0_ok"].sum()),
            "predicted_rows": int(table["mattersim_feature_ok"].sum()),
            "abstained_rows": int((~table["mattersim_feature_ok"]).sum()),
        },
        "inputs_sha256": {
            frames_path.name: sha256_file(frames_path),
            metadata_path.name: sha256_file(metadata_path),
        },
        "outputs_sha256": {output_path.name: sha256_file(output_path)},
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elementa", type=Path, required=True)
    parser.add_argument("--p9", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args(argv)
    manifest = run_mattersim_baseline(
        args.elementa,
        args.p9,
        args.output,
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )
    print(json.dumps(manifest["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["frame_to_atoms", "run_mattersim_baseline"]
