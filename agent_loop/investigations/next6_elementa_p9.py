#!/usr/bin/env python3
"""Compute corrected and charge-robust P9 features on strict ELEMENTA x0 frames.

This additive migration reads only the saved initial-frame archive and the
label-free x0 metadata table.  Frames whose earliest available ionic step is
not zero remain in the output as unsupported rows so deployment accounting can
ABSTAIN instead of silently treating a partially relaxed geometry as x0.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
import shlex
import sys
from typing import Mapping, Sequence
import zipfile

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure

from src.next6_wbm_build import sha256_file
from src.next6_wbm_features import parse_extxyz

# next4 is intentionally retained unchanged and supports its original direct
# script imports.  Make that existing source root importable here rather than
# editing the historical module's import style.
_SRC_ROOT = Path(__file__).resolve().parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from next4_features import (  # noqa: E402
    _crystal_nn_info,
    infer_formal_valences,
    p9c_lewis_features,
)
from polymorph_rank2 import AN_Z, CAT_Z  # noqa: E402


P9_COLUMNS = (
    "p9c_bond_mismatch_mean",
    "p9c_bond_mismatch_q95",
    "p9c_bond_mismatch_max",
    "p9c_cat_site_mismatch_max",
)
P9_ROBUST_COLUMNS = tuple(
    f"{column.replace('p9c_', 'p9r_')}_{bound}"
    for column in P9_COLUMNS
    for bound in ("min", "max")
)


def _comment_fields(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if len(lines) < 2:
        raise ValueError("extxyz frame is missing its comment")
    fields: dict[str, str] = {}
    for token in shlex.split(lines[1]):
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


def enumerate_uniform_charge_assignments(structure: Structure) -> list[np.ndarray]:
    """Enumerate every legacy-table, element-uniform charge-balanced assignment.

    This exposes rather than hides the first-solution behavior of the historical
    ``balance`` helper.  It deliberately makes no mixed-valence claim.
    """

    amounts = structure.composition.element_composition.get_el_amt_dict()
    anions = [symbol for symbol in amounts if symbol in AN_Z]
    if len(anions) != 1:
        return []
    anion = anions[0]
    cations = sorted(symbol for symbol in amounts if symbol != anion)
    if not cations or any(symbol not in CAT_Z for symbol in cations):
        return []
    target = -float(AN_Z[anion]) * float(amounts[anion])
    assignments: list[np.ndarray] = []
    for states in itertools.product(*(CAT_Z[symbol] for symbol in cations)):
        total = sum(float(state) * float(amounts[symbol]) for symbol, state in zip(cations, states))
        if not np.isclose(total, target, atol=1e-8, rtol=0.0):
            continue
        mapping = {symbol: float(state) for symbol, state in zip(cations, states)}
        mapping[anion] = float(AN_Z[anion])
        assignments.append(
            np.asarray([mapping[site.specie.symbol] for site in structure], dtype=float)
        )
    return assignments


def robust_p9_charge_envelope(structure: Structure) -> dict[str, object]:
    """Return P9 minima/maxima across every supported balanced charge assignment."""

    assignments = enumerate_uniform_charge_assignments(structure)
    out: dict[str, object] = {
        "p9r_assignment_count": len(assignments),
        "p9r_feature_ok": False,
        "p9r_feature_error": "",
        **{column: np.nan for column in P9_ROBUST_COLUMNS},
    }
    if not assignments:
        out["p9r_feature_error"] = "no_uniform_charge_assignment"
        return out
    try:
        neighbors = _crystal_nn_info(structure)
        blocks = [
            p9c_lewis_features(structure, values, neighbors=neighbors)
            for values in assignments
        ]
        if any(not all(column in block for column in P9_COLUMNS) for block in blocks):
            raise ValueError("incomplete P9 block")
        for column in P9_COLUMNS:
            values = np.asarray([block[column] for block in blocks], dtype=float)
            stem = column.replace("p9c_", "p9r_")
            out[f"{stem}_min"] = float(np.min(values))
            out[f"{stem}_max"] = float(np.max(values))
        out["p9r_feature_ok"] = True
    except Exception as exc:  # feature failure must become ABSTAIN downstream
        out["p9r_feature_error"] = f"{type(exc).__name__}:{exc}"
    return out


def corrected_p9_initial_features(row: Mapping[str, object]) -> dict[str, object]:
    """Compute the next5 P9 signal and its charge envelope on one saved x0 frame."""

    sid = str(row["sid"])
    text = str(row["text"])
    fields = _comment_fields(text)
    try:
        ionic_step = int(fields["ionic_step"])
    except KeyError:
        ionic_step = -1
    strict_x0 = ionic_step == 0
    out: dict[str, object] = {
        "sid": sid,
        "rk": str(row["rk"]),
        "material": str(row["material"]),
        "input_role": "unrelaxed_x0_only" if strict_x0 else "trajectory_earliest_available",
        "initial_ionic_step": ionic_step,
        "strict_x0_ok": strict_x0,
        "p9c_feature_ok": False,
        "p9c_feature_error": "",
        "p9c_valence_method": "",
        **{column: np.nan for column in P9_COLUMNS},
        "p9r_assignment_count": 0,
        "p9r_feature_ok": False,
        "p9r_feature_error": "",
        **{column: np.nan for column in P9_ROBUST_COLUMNS},
    }
    if not strict_x0:
        out["p9c_feature_error"] = "nonzero_initial_ionic_step"
        out["p9r_feature_error"] = "nonzero_initial_ionic_step"
        return out

    frame = parse_extxyz(text)
    structure = Structure(
        Lattice(frame.lattice),
        frame.species,
        frame.cart_coords,
        coords_are_cartesian=True,
    )
    try:
        valences, method = infer_formal_valences(structure)
        neighbors = _crystal_nn_info(structure)
        block = p9c_lewis_features(structure, valences, neighbors=neighbors)
        if not all(column in block for column in P9_COLUMNS):
            raise ValueError("incomplete P9 block")
        out.update({column: float(block[column]) for column in P9_COLUMNS})
        out["p9c_valence_method"] = str(method)
        out["p9c_feature_ok"] = True
    except Exception as exc:
        out["p9c_feature_error"] = f"{type(exc).__name__}:{exc}"
    out.update(robust_p9_charge_envelope(structure))
    return out


def _worker(row: Mapping[str, object]) -> dict[str, object]:
    return corrected_p9_initial_features(row)


def run_p9_extraction(input_dir: Path, output_dir: Path, *, workers: int) -> dict[str, object]:
    """Extract P9 features from a label-free x0 artifact directory."""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = input_dir / "elementa_x0_features.parquet"
    frames_path = input_dir / "elementa_initial_frames.zip"
    metadata = pd.read_parquet(metadata_path, columns=["sid", "rk", "material", "input_role"])
    if metadata["sid"].duplicated().any():
        raise ValueError("x0 metadata sid must be unique")
    if not metadata["input_role"].eq("unrelaxed_x0_only").all():
        raise ValueError("source metadata contains a non-x0 role")

    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(frames_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("initial-frame archive contains duplicate members")
        member_by_sid = {Path(name).stem: name for name in names}
        if set(member_by_sid) != set(metadata["sid"].astype(str)):
            raise ValueError("initial-frame and metadata sid sets differ")
        for record in metadata.to_dict("records"):
            sid = str(record["sid"])
            rows.append(
                {
                    "sid": sid,
                    "rk": str(record["rk"]),
                    "material": str(record["material"]),
                    "text": archive.read(member_by_sid[sid]).decode("utf-8"),
                }
            )

    if workers <= 1:
        features = [corrected_p9_initial_features(row) for row in rows]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            features = list(executor.map(_worker, rows, chunksize=16))
    table = pd.DataFrame(features)
    if table["sid"].duplicated().any() or set(table["sid"]) != set(metadata["sid"]):
        raise ValueError("P9 output sid mismatch")
    output_path = output_dir / "elementa_x0_p9_features.parquet"
    table.to_parquet(output_path, index=False)
    manifest: dict[str, object] = {
        "protocol": "2026-08-01-elementa-x0-p9-migration-v1",
        "input_policy": "strict ionic_step=0; later earliest frames retained as unsupported",
        "counts": {"input_frames": len(rows), "output_rows": len(table)},
        "inputs_sha256": {
            metadata_path.name: sha256_file(metadata_path),
            frames_path.name: sha256_file(frames_path),
        },
        "outputs_sha256": {output_path.name: sha256_file(output_path)},
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    manifest = run_p9_extraction(args.input, args.output, workers=args.workers)
    print(json.dumps(manifest["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "P9_COLUMNS",
    "P9_ROBUST_COLUMNS",
    "corrected_p9_initial_features",
    "enumerate_uniform_charge_assignments",
    "robust_p9_charge_envelope",
    "run_p9_extraction",
]
