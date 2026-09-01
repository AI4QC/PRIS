"""Additive MatterSim basin-relaxation plus MP phase-hull screening rule."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
from itertools import combinations
import json
import math
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence
import warnings

from ase import Atoms
import numpy as np
import pandas as pd
from pymatgen.analysis.phase_diagram import PDEntry, PatchedPhaseDiagram
from pymatgen.core import Composition

from src.next13d_acsc_dft_pairs import _json_bytes, _sha256_file
from src.next14_wbm_holdout import _publish_directory_no_replace
from src.next14_wbm_pauling import _load_holdout


PROTOCOL = "2026-08-02-next15-basin-hull-v1"
OUTPUT_NAME = "basin_hull_features.parquet"
MANIFEST_NAME = "MANIFEST.json"
BASIN_HULL_THRESHOLD_EV_PER_ATOM = 0.20
RELAX_FMAX_EV_PER_A = 0.05
RELAX_MAX_PREDICTION_STEPS = 64
# MatterSim BatchRelaxer's upstream default.  A 4096-atom inference batch
# exceeds the 12 GB deployment GPU even for these small cells; this changes
# packing only, not any per-structure optimizer or decision parameter.
RELAX_ATOM_BUDGET = 512
MIN_SUPPORTED_VOLUME_RATIO = 0.25
MAX_SUPPORTED_VOLUME_RATIO = 4.0
DECISIONS = ("KEEP", "REJECT", "ABSTAIN")


def basin_hull_decision(score: float, *, supported: bool) -> str:
    """Apply the frozen physical 0.20 eV/atom Basin-Hull boundary."""

    if type(supported) is not bool:
        raise ValueError("supported must be an exact boolean")
    if not supported:
        return "ABSTAIN"
    try:
        value = float(score)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("supported Basin-Hull score must be finite") from exc
    if not math.isfinite(value):
        raise ValueError("supported Basin-Hull score must be finite")
    return "REJECT" if value >= BASIN_HULL_THRESHOLD_EV_PER_ATOM else "KEEP"


def needed_element_subspaces(
    systems: Iterable[frozenset[str]],
) -> set[frozenset[str]]:
    """Return every nonempty elemental subspace needed by target systems."""

    result: set[frozenset[str]] = set()
    observed = list(systems)
    if not observed:
        raise ValueError("at least one target element system is required")
    for system in observed:
        if not isinstance(system, frozenset) or not system:
            raise ValueError("target systems must be nonempty frozensets")
        if any(type(symbol) is not str or not symbol for symbol in system):
            raise ValueError("element symbols must be nonempty exact strings")
        ordered = sorted(system)
        for size in range(1, len(ordered) + 1):
            result.update(
                frozenset(values) for values in combinations(ordered, size)
            )
    return result


def build_reference_entries(
    rows: Iterable[tuple[object, Mapping[str, object]]],
    *,
    needed_subspaces: set[frozenset[str]],
) -> list[PDEntry]:
    """Strip MP structures/corrections and retain raw-energy reference entries."""

    if not needed_subspaces or any(
        not isinstance(space, frozenset) or not space for space in needed_subspaces
    ):
        raise ValueError("needed_subspaces must contain nonempty frozensets")
    entries: list[PDEntry] = []
    names: set[str] = set()
    for raw_name, payload in rows:
        if not isinstance(payload, Mapping):
            raise ValueError("MP entry payload must be a mapping")
        composition = payload.get("composition")
        if not isinstance(composition, Mapping) or not composition:
            raise ValueError("MP entry composition must be a nonempty mapping")
        space = frozenset(str(symbol) for symbol in composition)
        if space not in needed_subspaces:
            continue
        name = str(raw_name)
        if not name or name in names:
            raise ValueError("MP reference entry names must be unique and nonempty")
        try:
            energy = float(payload["energy"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("MP raw energy must be a finite scalar") from exc
        if not math.isfinite(energy):
            raise ValueError("MP raw energy must be a finite scalar")
        entries.append(PDEntry(Composition(dict(composition)), energy, name=name))
        names.add(name)
    if not entries:
        raise ValueError("no MP entries matched the required chemical subspaces")
    return entries


def compute_hull_scores(
    reference_entries: Sequence[PDEntry],
    formulas: Sequence[str],
    total_energies_ev: Sequence[float],
) -> tuple[float, ...]:
    """Compute raw MatterSim energy above the raw MP patched phase hull."""

    if not reference_entries:
        raise ValueError("reference_entries must not be empty")
    if len(formulas) != len(total_energies_ev):
        raise ValueError("formulas and total energies must align")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"No suitable PhaseDiagrams found for .*"
        )
        phase_diagram = PatchedPhaseDiagram(
            list(reference_entries), keep_all_spaces=False, verbose=False
        )
        scores: list[float] = []
        for formula, energy in zip(formulas, total_energies_ev, strict=True):
            try:
                finite_energy = float(energy)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("candidate total energy must be finite") from exc
            if type(formula) is not str or not formula or not math.isfinite(finite_energy):
                raise ValueError("candidate formula and total energy must be valid")
            score = phase_diagram.get_e_above_hull(
                PDEntry(Composition(formula), finite_energy), allow_negative=True
            )
            if score is None or not math.isfinite(float(score)):
                raise ValueError("phase diagram did not return a finite hull score")
            scores.append(float(score))
    return tuple(scores)


def _unsupported_snapshot(error: str, *, prediction_steps: int) -> dict[str, object]:
    return {
        "supported": False,
        "error": str(error),
        "prediction_steps": int(prediction_steps),
        "capped_at_max_steps": prediction_steps >= RELAX_MAX_PREDICTION_STEPS,
        "energy_total_ev": np.nan,
        "energy_ev_per_atom": np.nan,
        "fmax_ev_per_a": np.nan,
        "stress_frobenius": np.nan,
        "volume_ratio": np.nan,
    }


def validate_relaxed_snapshot(
    base: Atoms,
    relaxed: Atoms,
    *,
    prediction_steps: int,
) -> dict[str, object]:
    """Validate one deployable relaxed snapshot; failures become abstentions."""

    if type(prediction_steps) is not int or prediction_steps <= 0:
        raise ValueError("prediction_steps must be a positive exact integer")
    try:
        if not isinstance(base, Atoms) or not isinstance(relaxed, Atoms):
            raise ValueError("base and relaxed structures must be ASE Atoms")
        if len(base) == 0 or len(base) != len(relaxed):
            raise ValueError("relaxation changed the atom count")
        if not np.array_equal(base.numbers, relaxed.numbers):
            raise ValueError("relaxation changed the ordered composition")
        positions = np.asarray(relaxed.positions, dtype=float)
        cell = np.asarray(relaxed.cell.array, dtype=float)
        if positions.shape != (len(base), 3) or not np.isfinite(positions).all():
            raise ValueError("relaxed positions are invalid")
        if cell.shape != (3, 3) or not np.isfinite(cell).all():
            raise ValueError("relaxed cell is invalid")
        if not np.asarray(relaxed.pbc, dtype=bool).all():
            raise ValueError("relaxed structure is not fully periodic")
        base_volume = float(base.get_volume())
        relaxed_volume = float(relaxed.get_volume())
        if not all(math.isfinite(value) and value > 0.0 for value in (base_volume, relaxed_volume)):
            raise ValueError("cell volume is invalid")
        volume_ratio = relaxed_volume / base_volume
        if not MIN_SUPPORTED_VOLUME_RATIO <= volume_ratio <= MAX_SUPPORTED_VOLUME_RATIO:
            raise ValueError("relaxed cell volume ratio is outside the support domain")
        energy = float(relaxed.info["total_energy"])
        if not math.isfinite(energy):
            raise ValueError("relaxed energy is invalid")
        forces = np.asarray(relaxed.arrays["forces"], dtype=float)
        if forces.shape != (len(base), 3) or not np.isfinite(forces).all():
            raise ValueError("relaxed forces are invalid")
        stress = np.asarray(relaxed.info["stress"], dtype=float)
        if stress.size not in {6, 9} or not np.isfinite(stress).all():
            raise ValueError("relaxed stress is invalid")
        fmax = float(np.linalg.norm(forces, axis=1).max())
        stress_frobenius = float(np.linalg.norm(stress))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return _unsupported_snapshot(str(exc), prediction_steps=prediction_steps)
    return {
        "supported": True,
        "error": "",
        "prediction_steps": prediction_steps,
        "capped_at_max_steps": prediction_steps >= RELAX_MAX_PREDICTION_STEPS,
        "energy_total_ev": energy,
        "energy_ev_per_atom": energy / len(base),
        "fmax_ev_per_a": fmax,
        "stress_frobenius": stress_frobenius,
        "volume_ratio": volume_ratio,
    }


def _runtime_identity(device: str) -> dict[str, object]:
    import ase
    import pymatgen
    import torch

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "ase_version": ase.__version__,
        "pymatgen_version": getattr(pymatgen, "__version__", importlib.metadata.version("pymatgen")),
        "mattersim_version": importlib.metadata.version("mattersim"),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else None,
    }


def run_basin_hull_features(
    *,
    metadata_path: Path,
    frames_zip_path: Path,
    holdout_manifest_path: Path,
    checkpoint_path: Path,
    mp_reference_path: Path,
    output_dir: Path,
    device: str = "cuda:0",
) -> dict[str, object]:
    """Run label-free NEXT15 features after the frozen threshold is recorded."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
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
    metadata, structures = _load_holdout(
        metadata_path=paths["metadata"],
        frames_zip_path=paths["geometry_only_frames"],
        manifest_path=paths["holdout_manifest"],
    )
    if len(metadata) != len(structures):
        raise ValueError("holdout metadata and structures are not aligned")

    runtime = _runtime_identity(device)
    if runtime["mattersim_version"] != "1.2.3" or runtime["cuda_available"] is not True:
        raise RuntimeError("NEXT15 requires MatterSim 1.2.3 with CUDA")
    from mattersim.applications.batch_relax import BatchRelaxer
    from mattersim.forcefield import Potential

    started = time.perf_counter()
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
            continue
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

    systems = {
        frozenset(structure.get_chemical_symbols()) for structure in structures
    }
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

    supported_indices = [
        index for index, summary in enumerate(summaries) if summary["supported"] is True
    ]
    supported_formulas = [str(metadata.iloc[index]["formula"]) for index in supported_indices]
    supported_energies = [
        float(summaries[index]["energy_total_ev"]) for index in supported_indices
    ]
    scores = compute_hull_scores(
        reference_entries, supported_formulas, supported_energies
    )
    score_by_index = dict(zip(supported_indices, scores, strict=True))

    rows: list[dict[str, object]] = []
    for index, upstream in enumerate(metadata.to_dict("records")):
        summary = summaries[index]
        supported = bool(summary["supported"])
        score = score_by_index.get(index, np.nan)
        decision = basin_hull_decision(score, supported=supported)
        rows.append(
            {
                "material_id": str(upstream["material_id"]),
                "rk": str(upstream["rk"]),
                "formula": str(upstream["formula"]),
                "natoms": int(upstream["natoms"]),
                **summary,
                "basin_hull_score_ev_per_atom": score,
                "basin_hull_decision": decision,
            }
        )
    table = pd.DataFrame(rows)
    if len(table) != len(metadata) or table["material_id"].nunique() != len(table):
        raise ValueError("NEXT15 output lost or duplicated holdout rows")
    if not set(table["basin_hull_decision"]).issubset(DECISIONS):
        raise ValueError("NEXT15 output contains an unknown decision")

    repo_root = Path(__file__).resolve().parents[1]
    source_paths = [
        Path(__file__).resolve(),
        repo_root / "src/next7_mattersim_prerelax.py",
        repo_root / "src/next14_wbm_pauling.py",
    ]
    installed_batch_relax = Path(importlib.import_module("mattersim.applications.batch_relax").__file__).resolve()
    source_hashes = {
        str(path.relative_to(repo_root)) if path.is_relative_to(repo_root) else str(path): _sha256_file(path)
        for path in [*source_paths, installed_batch_relax]
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        output_path = temporary / OUTPUT_NAME
        table.to_parquet(output_path, index=False)
        manifest = {
            "protocol": PROTOCOL,
            "evidence_role": "retrospective WBM method execution after NEXT14 label opening; this execution did not read WBM endpoints",
            "wbm_endpoint_bytes_read_by_execution": False,
            "thresholds_refit": False,
            "scientific_improvement_claim": False,
            "rule": {
                "formula": "B64 = E_MatterSim_relaxed/N - E_raw_MP_hull(composition)",
                "threshold_ev_per_atom": BASIN_HULL_THRESHOLD_EV_PER_ATOM,
                "comparison": ">=",
                "failure_policy": "ABSTAIN",
            },
            "relaxation": {
                "optimizer": "FIRE",
                "filter": "FRECHETCELLFILTER",
                "fmax_ev_per_a": RELAX_FMAX_EV_PER_A,
                "max_prediction_steps": RELAX_MAX_PREDICTION_STEPS,
                "atom_budget": RELAX_ATOM_BUDGET,
                "volume_ratio_support": [
                    MIN_SUPPORTED_VOLUME_RATIO,
                    MAX_SUPPORTED_VOLUME_RATIO,
                ],
            },
            "counts": {
                "rows": len(table),
                "supported": int(table["supported"].sum()),
                "abstained": int((table["basin_hull_decision"] == "ABSTAIN").sum()),
                "kept": int((table["basin_hull_decision"] == "KEEP").sum()),
                "rejected": int((table["basin_hull_decision"] == "REJECT").sum()),
                "capped_at_max_steps": int(table["capped_at_max_steps"].sum()),
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
            "outputs_sha256": {OUTPUT_NAME: _sha256_file(output_path)},
            "known_limitations": [
                "The MP reference contains prior DFT calculations, although no new WBM endpoint is read.",
                "WBM test labels were already opened by NEXT14 before this rule was proposed.",
                "The raw-energy phase diagram omits MP2020 compatibility corrections for deployability without endpoint metadata.",
                "MatterSim is a strong MLIP baseline rather than a purely geometric analytic law.",
            ],
        }
        (temporary / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        _publish_directory_no_replace(temporary, target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--frames-zip", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mp-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    run_basin_hull_features(
        metadata_path=args.metadata,
        frames_zip_path=args.frames_zip,
        holdout_manifest_path=args.holdout_manifest,
        checkpoint_path=args.checkpoint,
        mp_reference_path=args.mp_reference,
        output_dir=args.output_dir,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
