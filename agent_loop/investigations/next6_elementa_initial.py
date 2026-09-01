#!/usr/bin/env python3
"""Extract ELEMENTA ionic polymorph x0 frames and compute pre-DFT features."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Iterator, Mapping, TextIO

import numpy as np
import pandas as pd

from src.next6_wbm_build import sha256_file
from src.next6_wbm_features import geometry_features, parse_extxyz


DEFAULT_RANK = Path(
    "$PRIS_FEATURES/polymorph_rank2.parquet"
)
DEFAULT_ENDPOINTS = Path(
    "<other-repo>/data/raw/acquired/elementa/work/endpoints_open.tsv"
)
DEFAULT_ARCHIVE = Path(
    "<other-repo>/data/raw/acquired/elementa/ELEMENTA_open.extxyz.tar.zst"
)
_SID = re.compile(r"elem-(\d+)")


def select_ranked_endpoints(rank: pd.DataFrame, endpoints: pd.DataFrame) -> pd.DataFrame:
    """Resolve one-indexed ``elem-N`` IDs and validate the final-energy join."""

    required_rank = {"sid", "rk", "e_per_atom", "nat"}
    required_endpoints = {
        "material",
        "formula",
        "structure",
        "spin",
        "ionic_step",
        "n_sites",
        "energy",
        "max_force",
    }
    if missing := sorted(required_rank - set(rank.columns)):
        raise ValueError(f"rank columns missing: {missing}")
    if missing := sorted(required_endpoints - set(endpoints.columns)):
        raise ValueError(f"endpoint columns missing: {missing}")

    indices: list[int] = []
    for sid in rank["sid"].astype(str):
        match = _SID.fullmatch(sid)
        if match is None:
            raise ValueError(f"invalid ELEMENTA sid: {sid}")
        index = int(match.group(1)) - 1
        if index < 0 or index >= len(endpoints):
            raise ValueError(f"ELEMENTA sid outside endpoint table: {sid}")
        indices.append(index)
    selected = endpoints.iloc[indices].reset_index(drop=True).copy()
    selected.insert(0, "sid", rank["sid"].astype(str).to_numpy())
    selected.insert(1, "rk", rank["rk"].astype(str).to_numpy())
    selected.insert(2, "e_per_atom", rank["e_per_atom"].to_numpy(dtype=float))
    selected.insert(3, "nat", rank["nat"].to_numpy(dtype=int))
    if not np.array_equal(
        selected["nat"].to_numpy(dtype=int), selected["n_sites"].to_numpy(dtype=int)
    ):
        raise ValueError("site-count mismatch between rank and endpoint tables")
    endpoint_epa = selected["energy"].to_numpy(dtype=float) / selected["n_sites"].to_numpy(
        dtype=float
    )
    if not np.allclose(
        endpoint_epa,
        selected["e_per_atom"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError("energy mismatch between rank and endpoint tables")
    if selected["material"].duplicated().any():
        raise ValueError("selected core ELEMENTA material IDs must be unique")
    return selected.rename(
        columns={"ionic_step": "final_ionic_step", "max_force": "final_max_force"}
    )


def _comment_scalar(comment: str, key: str) -> str:
    """Read one unquoted extxyz scalar without tokenizing the full comment."""

    marker = " " + key + "="
    start = comment.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    end = comment.find(" ", start)
    return comment[start:] if end < 0 else comment[start:end]


def iter_selected_initial_frames(
    stream: TextIO,
    targets: Mapping[str, Mapping[str, object]],
) -> Iterator[dict[str, object]]:
    """Yield exactly the first frame of every selected contiguous trajectory."""

    seen: set[str] = set()
    while True:
        count_line = stream.readline()
        if not count_line:
            break
        if not count_line.strip():
            continue
        n_atoms = int(count_line.strip())
        comment = stream.readline()
        material = _comment_scalar(comment, "material")
        if material not in targets or material in seen:
            for _ in range(n_atoms):
                if stream.readline() == "":
                    raise ValueError("truncated ELEMENTA extxyz frame")
            continue
        atom_lines = [stream.readline() for _ in range(n_atoms)]
        if any(line == "" for line in atom_lines):
            raise ValueError("truncated ELEMENTA extxyz frame")
        seen.add(material)
        target = targets[material]
        yield {
            **target,
            "material": material,
            "initial_ionic_step": int(_comment_scalar(comment, "ionic_step") or "-1"),
            "text": count_line + comment + "".join(atom_lines),
        }
        if len(seen) == len(targets):
            return


def ionic_initial_features(row: Mapping[str, object]) -> dict[str, object]:
    """Compute geometry, Pauling, Shannon/Born, Ewald, and BVS on x0 only."""

    from pymatgen.core import Lattice, Structure

    from src.bv_judge import bvs_features
    from src.phys_feat import one as ewald_features
    from src.phys_law import phys_feats
    from src.polymorph_rank2 import balance, one as pauling_features

    frame = parse_extxyz(str(row["text"]))
    out: dict[str, object] = {
        "sid": str(row["sid"]),
        "rk": str(row["rk"]),
        "material": str(row.get("material", frame.material_id)),
        "input_role": "unrelaxed_x0_only",
    }
    out.update({f"geom_{key}": value for key, value in geometry_features(frame).items()})
    valence_map = balance(str(row["formula"]))
    if valence_map is None:
        out.update(
            ionic_feature_ok=False,
            ionic_feature_error="no_unique_charge_balance",
            pauling_feature_ok=False,
            shannon_feature_ok=False,
            ewald_feature_ok=False,
            bvs_feature_ok=False,
        )
        return out

    record = {
        "sid": str(row["sid"]),
        "rk": str(row["rk"]),
        "nat": len(frame.species),
        "e_per_atom": np.nan,
        "lattice": frame.lattice,
        "species": list(frame.species),
        "coords": frame.cart_coords,
        "vmap": valence_map,
    }
    structure = Structure(
        Lattice(frame.lattice),
        frame.species,
        frame.cart_coords,
        coords_are_cartesian=True,
    )
    valences = [float(valence_map[site.specie.symbol]) for site in structure]

    blocks = (
        ("pauling_feature_ok", pauling_features(record)),
        ("shannon_feature_ok", phys_feats(structure, valences)),
        ("ewald_feature_ok", ewald_features(record)),
        ("bvs_feature_ok", bvs_features(record)),
    )
    successes = []
    for status_key, block in blocks:
        ok = block is not None
        out[status_key] = ok
        successes.append(ok)
        if block:
            for key, value in block.items():
                if key not in {"sid", "rk", "e_per_atom", "nat", "energy", "forces"}:
                    out[key] = value
    out["ionic_feature_ok"] = bool(any(successes))
    out["ionic_feature_error"] = "" if any(successes) else "all_ionic_blocks_failed"
    return out


def build_elementa_initial_artifacts(
    *,
    rank_path: Path,
    endpoints_path: Path,
    archive_path: Path,
    output_dir: Path,
    workers: int,
) -> dict[str, object]:
    """Stream the archive once, persist selected x0 frames, and featurize them."""

    rank_path = Path(rank_path)
    endpoints_path = Path(endpoints_path)
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rank = pd.read_parquet(rank_path, columns=["sid", "rk", "e_per_atom", "nat"])
    endpoints = pd.read_csv(endpoints_path, sep="\t")
    inventory = select_ranked_endpoints(rank, endpoints)
    labels = inventory[
        [
            "sid",
            "rk",
            "material",
            "formula",
            "e_per_atom",
            "nat",
            "final_ionic_step",
            "final_max_force",
        ]
    ].copy()
    labels_path = output_dir / "elementa_labels.parquet"
    labels.to_parquet(labels_path, index=False)
    targets = {
        str(row.material): {
            "sid": str(row.sid),
            "rk": str(row.rk),
            "formula": str(row.formula),
        }
        for row in inventory.itertuples(index=False)
    }

    process = subprocess.Popen(
        ["tar", "--use-compress-program=unzstd", "-xOf", str(archive_path)],
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1 << 20,
    )
    assert process.stdout is not None
    selected_rows: list[dict[str, object]] = []
    frames_zip_path = output_dir / "elementa_initial_frames.zip"
    try:
        with zipfile.ZipFile(
            frames_zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for row in iter_selected_initial_frames(process.stdout, targets):
                archive.writestr(f"{row['sid']}.extxyz", str(row["text"]))
                selected_rows.append(row)
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait()
    if len(selected_rows) != len(targets):
        found = {str(row["material"]) for row in selected_rows}
        missing = sorted(set(targets) - found)
        raise ValueError(f"missing {len(missing)} selected initial frames; first={missing[:3]}")

    if workers <= 0:
        raise ValueError("workers must be positive")
    if workers == 1:
        feature_rows = list(map(ionic_initial_features, selected_rows))
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            feature_rows = list(pool.imap(ionic_initial_features, selected_rows, chunksize=16))
    features = pd.DataFrame(feature_rows).sort_values("sid", kind="stable")
    features_path = output_dir / "elementa_x0_features.parquet"
    features.to_parquet(features_path, index=False)
    manifest: dict[str, object] = {
        "protocol": "2026-08-01-dft-pre-screening-design-v1",
        "input_role": "unrelaxed_x0_only",
        "counts": {
            "targets": len(targets),
            "initial_frames": len(selected_rows),
            "feature_rows": len(features),
        },
        "inputs": {
            "rank": str(rank_path),
            "rank_sha256": sha256_file(rank_path),
            "endpoints": str(endpoints_path),
            "endpoints_sha256": sha256_file(endpoints_path),
            "archive": str(archive_path),
            "archive_bytes": archive_path.stat().st_size,
        },
        "outputs_sha256": {
            labels_path.name: sha256_file(labels_path),
            features_path.name: sha256_file(features_path),
            frames_zip_path.name: sha256_file(frames_zip_path),
        },
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=Path, default=DEFAULT_RANK)
    parser.add_argument("--endpoints", type=Path, default=DEFAULT_ENDPOINTS)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) // 2))
    args = parser.parse_args()
    manifest = build_elementa_initial_artifacts(
        rank_path=args.rank,
        endpoints_path=args.endpoints,
        archive_path=args.archive,
        output_dir=args.output,
        workers=args.workers,
    )
    print(json.dumps(manifest["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
