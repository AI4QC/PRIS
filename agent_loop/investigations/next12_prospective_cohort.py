"""Freeze a prospective, geometry-only SSAGEN cohort before any energy access."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
from typing import Protocol, Sequence
import zipfile

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import write

from src.next8_mattersim_committee_features import (
    _atomic_publish_directory_no_replace,
)


PROTOCOL = "2026-08-02-next12-ssagen-prospective-x0-v1"
GENERATOR_NAME = "SSAGEN-CIVAE-Transformer-500"
FROZEN_SEED = 20260802
FROZEN_ATTEMPT_COUNT = 256
FROZEN_DEVICE = "cuda:0"
FROZEN_MODEL_CLASS = "sasgen.civae.pl_modules.symgen_new.CIVAE"
FROZEN_MODEL_PARAMETER_COUNT = 23_989_052
FROZEN_INPUT_SHA256 = {
    "checkpoint": "b300df1faec44031505dd9556c357403fe711d020297077aec4a990eb89749ee",
    "lattice_scaler": "51222a43208c0d8b7d3ac66784325c759e8d76127eff5c6c71260175e7e0a75e",
    "prop_scaler": "b833dd6ea7c7ea1bf08bbc963f1e35b72b715ef45fbb90377bf4a44fe626e4cc",
}
COHORT_NAME = "cohort.parquet"
ARCHIVE_NAME = "geometry_only_frames.zip"
MANIFEST_NAME = "MANIFEST.json"
COHORT_COLUMNS = (
    "attempt_index",
    "sid",
    "generator",
    "seed",
    "latent_sha256",
    "generation_status",
    "error",
    "natoms",
    "formula",
    "volume_angstrom3",
    "volume_per_atom_angstrom3",
    "geometry_sha256",
    "archive_member",
)
EXTERNAL_SOURCE_RELATIVE = (
    "sasgen/generators.py",
    "sasgen/civae/pl_modules/symgen_new.py",
    "sasgen/civae/pl_modules/gnn.py",
    "sasgen/civae/common/data_utils.py",
)


class GeneratorAdapter(Protocol):
    model_class: str
    model_parameter_count: int
    latent_dim: int
    device: str

    def generate(self, latent: np.ndarray, attempt_index: int) -> Atoms:
        """Return one raw x0 structure or raise an attempt-local exception."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_record(path: Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"input is not a file: {resolved}")
    return {"path": str(resolved), "sha256": _sha256_file(resolved)}


def _exception_text(exc: Exception) -> str:
    detail = str(exc)
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def _canonical_geometry(atoms: Atoms) -> Atoms:
    if not isinstance(atoms, Atoms):
        raise TypeError("generator output must be an ase.Atoms instance")
    count = len(atoms)
    if count < 1:
        raise ValueError("generated geometry must contain at least one atom")
    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    cell = np.asarray(atoms.cell.array, dtype=np.float64)
    pbc = np.asarray(atoms.pbc)
    if positions.shape != (count, 3) or not np.all(np.isfinite(positions)):
        raise ValueError("generated positions must be finite with shape (N,3)")
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        raise ValueError("generated cell must be a finite 3 by 3 matrix")
    if not np.array_equal(pbc, np.ones(3, dtype=bool)):
        raise ValueError("generated geometry must be periodic in three dimensions")
    determinant = float(np.linalg.det(cell))
    if not np.isfinite(determinant) or determinant == 0.0:
        raise ValueError("generated cell must be nonsingular")
    scaled = np.linalg.solve(cell.T, positions.T).T
    if not np.all(np.isfinite(scaled)):
        raise ValueError("generated fractional coordinates must be finite")
    wrapped = scaled - np.floor(scaled)
    clean = Atoms(
        numbers=np.asarray(atoms.numbers, dtype=int).copy(),
        cell=cell.copy(),
        pbc=(True, True, True),
    )
    clean.set_scaled_positions(wrapped)
    return clean


def _geometry_payload(atoms: Atoms) -> bytes:
    clean = Atoms(
        numbers=np.asarray(atoms.numbers, dtype=int).copy(),
        positions=np.asarray(atoms.positions, dtype=np.float64).copy(),
        cell=np.asarray(atoms.cell.array, dtype=np.float64).copy(),
        pbc=(True, True, True),
    )
    stream = io.StringIO()
    write(stream, clean, format="extxyz", write_results=False)
    payload = stream.getvalue().encode("utf-8")
    lowered = payload.lower()
    for forbidden in (b"energy=", b"forces", b"stress", b"endpoint"):
        if forbidden in lowered:
            raise RuntimeError(f"geometry-only payload contains forbidden field: {forbidden!r}")
    return payload


def _zip_info(member: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


class _SSAGENAdapter:
    """Fail-closed adapter for the explicitly recovered real CIVAE checkpoint."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        lattice_scaler_path: Path,
        prop_scaler_path: Path,
        ssagen_root: Path,
        device: str,
    ) -> None:
        root = Path(ssagen_root).resolve()
        os.environ.setdefault("PROJECT_ROOT", str(root))
        os.environ.setdefault("TORCH_DYNAMO_DISABLE", "1")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import math

        if not hasattr(np, "math"):
            np.math = math  # type: ignore[attr-defined]
        import torch
        from pymatgen.core.periodic_table import Element
        from pymatgen.io.ase import AseAtomsAdaptor
        from sasgen.civae.pl_modules.symgen_new import CIVAE
        from sasgen.generators import CIVAEGenerator

        if device != FROZEN_DEVICE or not torch.cuda.is_available():
            raise RuntimeError("formal SSAGEN cohort requires available cuda:0")
        model = CIVAE.load_from_checkpoint(
            str(Path(checkpoint_path).resolve()),
            map_location=device,
            weights_only=False,
        )
        model.lattice_scaler = torch.load(
            Path(lattice_scaler_path).resolve(), map_location=device, weights_only=False
        )
        model.scaler = torch.load(
            Path(prop_scaler_path).resolve(), map_location=device, weights_only=False
        )
        model.eval().to(device)
        helper = CIVAEGenerator.__new__(CIVAEGenerator)
        helper.model = model
        helper.device = torch.device(device)
        helper.elements_dict = {
            number: Element.from_Z(number).symbol for number in range(1, 119)
        }
        helper.elements_dict_inv = {
            symbol: number for number, symbol in helper.elements_dict.items()
        }
        self._torch = torch
        self._model = model
        self._helper = helper
        self._adaptor = AseAtomsAdaptor()
        self.model_class = f"{type(model).__module__}.{type(model).__name__}"
        self.model_parameter_count = int(sum(value.numel() for value in model.parameters()))
        self.latent_dim = int(model.hparams.latent_dim)
        self.device = device

    def generate(self, latent: np.ndarray, attempt_index: int) -> Atoms:
        del attempt_index
        vector = self._torch.as_tensor(
            np.asarray(latent, dtype=np.float32),
            dtype=self._torch.float32,
            device=self.device,
        ).reshape(1, self.latent_dim)
        structures = self._helper.get_struct_from_z(vector)
        if len(structures) != 1:
            raise RuntimeError(f"real CIVAE returned {len(structures)} structures")
        return self._adaptor.get_atoms(structures[0])


def _adapter_identity(adapter: GeneratorAdapter) -> tuple[str, int, int, str]:
    try:
        model_class = str(adapter.model_class)
        parameter_count = int(adapter.model_parameter_count)
        latent_dim = int(adapter.latent_dim)
        device = str(adapter.device)
    except Exception as exc:
        raise RuntimeError(f"generator adapter identity is incomplete: {_exception_text(exc)}") from exc
    if "Mock" in model_class or "mock" in model_class:
        raise RuntimeError(f"Mock generator models are forbidden: {model_class}")
    if not model_class or parameter_count <= 0 or latent_dim <= 0 or not device:
        raise RuntimeError("generator adapter identity values are invalid")
    return model_class, parameter_count, latent_dim, device


def freeze_prospective_cohort(
    *,
    checkpoint_path: Path,
    lattice_scaler_path: Path,
    prop_scaler_path: Path,
    output_dir: Path,
    ssagen_root: Path = Path("<other-repo>/"),
    adapter: GeneratorAdapter | None = None,
    attempt_count: int = FROZEN_ATTEMPT_COUNT,
    seed: int = FROZEN_SEED,
    device: str = FROZEN_DEVICE,
) -> dict[str, object]:
    """Generate and atomically freeze every attempt before any energy call."""

    # Bind every filesystem argument before importing or calling the generator.
    # SSAGEN imports may change the process working directory via PROJECT_ROOT.
    target = Path(output_dir).resolve()
    resolved_ssagen_root = Path(ssagen_root).resolve()
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
    if type(attempt_count) is not int or attempt_count <= 0:
        raise ValueError("attempt_count must be a positive exact integer")
    if type(seed) is not int or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative exact integer")
    input_paths = {
        "checkpoint": Path(checkpoint_path).resolve(),
        "lattice_scaler": Path(lattice_scaler_path).resolve(),
        "prop_scaler": Path(prop_scaler_path).resolve(),
    }
    input_records = {role: _hash_record(path) for role, path in input_paths.items()}
    builtin = adapter is None
    if builtin:
        observed_hashes = {role: record["sha256"] for role, record in input_records.items()}
        if observed_hashes != FROZEN_INPUT_SHA256:
            raise ValueError("formal generator inputs differ from frozen identities")
        if attempt_count != FROZEN_ATTEMPT_COUNT or seed != FROZEN_SEED or device != FROZEN_DEVICE:
            raise ValueError("formal generator count, seed, and device are frozen")
        active_adapter: GeneratorAdapter = _SSAGENAdapter(
            checkpoint_path=input_paths["checkpoint"],
            lattice_scaler_path=input_paths["lattice_scaler"],
            prop_scaler_path=input_paths["prop_scaler"],
            ssagen_root=resolved_ssagen_root,
            device=device,
        )
        adapter_mode = "builtin_real_ssagen"
    else:
        active_adapter = adapter
        adapter_mode = "injected_test_double"
    model_class, parameter_count, latent_dim, adapter_device = _adapter_identity(active_adapter)
    if builtin and (
        model_class != FROZEN_MODEL_CLASS
        or parameter_count != FROZEN_MODEL_PARAMETER_COUNT
        or adapter_device != FROZEN_DEVICE
    ):
        raise RuntimeError("loaded SSAGEN model identity differs from the frozen real model")

    random.seed(seed)
    np.random.seed(seed % (2**32))
    rng = np.random.default_rng(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
    except ImportError:
        if builtin:
            raise

    rows: list[dict[str, object]] = []
    frames: list[tuple[str, bytes]] = []
    for attempt_index in range(attempt_count):
        sid = f"ssagen-t500-s{seed}-a{attempt_index:04d}"
        latent = rng.standard_normal(latent_dim).astype("<f4", copy=False)
        latent_sha256 = hashlib.sha256(latent.tobytes(order="C")).hexdigest()
        try:
            generated = active_adapter.generate(latent.copy(), attempt_index)
            atoms = _canonical_geometry(generated)
            payload = _geometry_payload(atoms)
            geometry_sha256 = hashlib.sha256(payload).hexdigest()
            member = f"frames/{sid}.extxyz"
            frames.append((member, payload))
            volume = float(abs(np.linalg.det(atoms.cell.array)))
            row = {
                "attempt_index": attempt_index,
                "sid": sid,
                "generator": GENERATOR_NAME,
                "seed": seed,
                "latent_sha256": latent_sha256,
                "generation_status": "generated",
                "error": "",
                "natoms": len(atoms),
                "formula": atoms.get_chemical_formula(mode="hill"),
                "volume_angstrom3": volume,
                "volume_per_atom_angstrom3": volume / len(atoms),
                "geometry_sha256": geometry_sha256,
                "archive_member": member,
            }
        except Exception as exc:
            row = {
                "attempt_index": attempt_index,
                "sid": sid,
                "generator": GENERATOR_NAME,
                "seed": seed,
                "latent_sha256": latent_sha256,
                "generation_status": "failed",
                "error": _exception_text(exc),
                "natoms": 0,
                "formula": "",
                "volume_angstrom3": np.nan,
                "volume_per_atom_angstrom3": np.nan,
                "geometry_sha256": None,
                "archive_member": None,
            }
        rows.append(row)
    table = pd.DataFrame(rows, columns=COHORT_COLUMNS)
    if len(table) != attempt_count or table["attempt_index"].tolist() != list(range(attempt_count)):
        raise RuntimeError("prospective cohort did not retain every ordered attempt")

    repository_source = Path(__file__).resolve()
    executed_sources = {"src/next12_prospective_cohort.py": _sha256_file(repository_source)}
    external_sources: dict[str, str] = {}
    if builtin:
        root = resolved_ssagen_root
        external_sources = {
            relative: _sha256_file(root / relative) for relative in EXTERNAL_SOURCE_RELATIVE
        }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        cohort_path = staging / COHORT_NAME
        archive_path = staging / ARCHIVE_NAME
        table.to_parquet(cohort_path, index=False)
        with zipfile.ZipFile(archive_path, "x") as archive:
            for member, payload in frames:
                archive.writestr(_zip_info(member), payload)
        generated = int(table["generation_status"].eq("generated").sum())
        failed = int(table["generation_status"].eq("failed").sum())
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "mode": "prospective_x0_geometry_freeze",
            "labels_opened": False,
            "energy_or_force_models_called": False,
            "all_attempts_retained": True,
            "generation": {
                "adapter_mode": adapter_mode,
                "model_class": model_class,
                "model_parameter_count": parameter_count,
                "latent_dim": latent_dim,
                "device": adapter_device,
                "seed": seed,
                "attempt_count": attempt_count,
            },
            "geometry_schema": {
                "allowed": ["species", "pos", "Lattice", "pbc"],
                "forbidden": ["energy", "forces", "stress", "endpoint labels"],
            },
            "counts": {
                "attempts": attempt_count,
                "generated": generated,
                "failed": failed,
                "archive_frames": len(frames),
                "total_atoms": int(table["natoms"].sum()),
            },
            "inputs_sha256": input_records,
            "executed_source_sha256": executed_sources,
            "external_source_sha256": external_sources,
            "outputs_sha256": {
                COHORT_NAME: _sha256_file(cohort_path),
                ARCHIVE_NAME: _sha256_file(archive_path),
            },
            "production_protocol_eligible": bool(builtin),
            "scientific_improvement_claim": False,
            "known_limitations": [
                "The recovered checkpoint was trained on only 500 structures and is a weak generator.",
                "This artifact freezes x0 geometries only and contains no stability result.",
                "Generation failures are retained but have no geometry frame.",
            ],
        }
        manifest_path = staging / MANIFEST_NAME
        payload = json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
        with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        for role, path in input_paths.items():
            if _sha256_file(path.resolve()) != input_records[role]["sha256"]:
                raise RuntimeError(f"input {role} changed before publication")
        if _sha256_file(repository_source) != executed_sources["src/next12_prospective_cohort.py"]:
            raise RuntimeError("executed cohort source changed before publication")
        if builtin:
            root = resolved_ssagen_root
            for relative, digest in external_sources.items():
                if _sha256_file(root / relative) != digest:
                    raise RuntimeError(f"SSAGEN source {relative} changed before publication")
        _atomic_publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--lattice-scaler", required=True, type=Path)
    parser.add_argument("--prop-scaler", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ssagen-root", type=Path, default=Path("<other-repo>/"))
    parser.add_argument("--attempt-count", type=int, default=FROZEN_ATTEMPT_COUNT)
    parser.add_argument("--seed", type=int, default=FROZEN_SEED)
    parser.add_argument("--device", default=FROZEN_DEVICE)
    arguments = parser.parse_args(argv)
    freeze_prospective_cohort(
        checkpoint_path=arguments.checkpoint,
        lattice_scaler_path=arguments.lattice_scaler,
        prop_scaler_path=arguments.prop_scaler,
        output_dir=arguments.output_dir,
        ssagen_root=arguments.ssagen_root,
        attempt_count=arguments.attempt_count,
        seed=arguments.seed,
        device=arguments.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
