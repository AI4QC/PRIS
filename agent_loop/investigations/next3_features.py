#!/usr/bin/env python3
"""Six new descriptor families for np-next-20260801c (frozen vocabulary).

Families (docs/plans/2026-08-01-six-families-design.md):

- P4  ChemEnv CSM at cation sites (repository recipe from build_bonds.py).
- P6  coordination shell-gap (cutoff-free bondedness) on the 8 A sphere.
- P7  polyanion / homonuclear anion-contact detector (fixed covalent radii).
- P8  CrystalNN-vs-Voronoi neighbour-set Jaccard disagreement.
- P9  Hawthorne R3 Lewis acid-base bond-strength matching (CrystalNN graph).
- P10 Voronoi free volume and site off-centering.

Formal valences from ``discriminate.guess_oxi``; ``BVAnalyzer`` never called.
Reads only the physically isolated records; writes outside the repository.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from next_features import (  # noqa: E402
    _crystal_nn_info,
    _hetero_bond_edges,
    _load_records,
    _voronoi_polyhedra,
)

P6_MAX_SHELL = 12
P7_SHORT_CONTACT_RATIO = 1.3


# --------------------------------------------------------------------- P4


def p4_csm_site_values(structure, formal_valences: Sequence[float]):
    """Per-cation-site ChemEnv CSM with the repository's exact recipe."""

    from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
        LocalGeometryFinder,
    )
    from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import (
        MultiWeightsChemenvStrategy,
    )
    from pymatgen.analysis.chemenv.coordination_environments.structure_environments import (
        LightStructureEnvironments,
    )

    charges = np.asarray(formal_valences, dtype=float)
    finder = LocalGeometryFinder()
    finder.setup_parameters(
        centering_type="centroid",
        include_central_site_in_centroid=True,
        structure_refinement=LocalGeometryFinder.STRUCTURE_REFINEMENT_NONE,
    )
    finder.setup_structure(structure=structure)
    envs = finder.compute_structure_environments(
        only_cations=True,
        valences=[int(round(v)) for v in charges],
        maximum_distance_factor=1.41,
    )
    light = LightStructureEnvironments.from_structure_environments(
        strategy=MultiWeightsChemenvStrategy.stats_article_weights_parameters(),
        structure_environments=envs,
    )
    values: list[float | None] = []
    for index in range(len(structure)):
        entry = (
            light.coordination_environments[index]
            if index < len(light.coordination_environments)
            else None
        )
        if not entry:
            values.append(None)
            continue
        csm = float(entry[0].get("csm", float("nan")))
        values.append(csm if np.isfinite(csm) else None)
    return values


# --------------------------------------------------------------------- P6


def p6_shell_gap_site_stats(structure, formal_valences, *, sphere) -> list[dict | None]:
    charges = np.asarray(formal_valences, dtype=float)
    stats: list[dict | None] = []
    for center in range(len(structure)):
        distances = sorted(
            float(nb.nn_distance)
            for nb in sphere[center]
            if int(nb.index) != center and float(nb.nn_distance) > 0
        )
        if len(distances) < 2:
            stats.append(None)
            continue
        ratios = [
            distances[k + 1] / distances[k]
            for k in range(min(len(distances) - 1, P6_MAX_SHELL))
        ]
        gap_index = int(np.argmax(ratios))  # 0-based: gap after neighbour k+1
        stats.append(
            {
                "gap_ratio": float(ratios[gap_index]),
                "gap_pos": float(gap_index + 1),
                "shell_width": float(distances[gap_index] / distances[0]),
            }
        )
    return stats


# --------------------------------------------------------------------- P7


def p7_polyanion_features(structure, formal_valences, *, sphere) -> dict[str, float]:
    charges = np.asarray(formal_valences, dtype=float)
    from pymatgen.core import Element

    radii: dict[str, float] = {}
    min_ratio = np.full(len(structure), np.nan)
    for center in range(len(structure)):
        if charges[center] >= 0:
            continue
        symbol = structure[center].specie.symbol
        if symbol not in radii:
            # pymatgen 2026 removed Element.covalent_radius; the atomic radius
            # is the remaining fixed, unfitted tabulated length scale.
            radii[symbol] = float(Element(symbol).atomic_radius)
        reference = 2.0 * radii[symbol]
        if reference <= 0:
            continue
        for nb in sphere[center]:
            other = int(nb.index)
            if other == center or charges[other] >= 0:
                continue
            if structure[other].specie.symbol != symbol:
                continue
            distance = float(nb.nn_distance)
            if distance <= 0:
                continue
            ratio = distance / reference
            if not np.isfinite(min_ratio[center]) or ratio < min_ratio[center]:
                min_ratio[center] = ratio
    anion = charges < 0
    covered = anion & np.isfinite(min_ratio)
    out: dict[str, float] = {}
    if covered.any():
        out["p7poly_an_contact_min"] = float(np.min(min_ratio[covered]))
        out["p7poly_an_contact_frac"] = float(
            np.mean(min_ratio[covered] < P7_SHORT_CONTACT_RATIO)
        )
    out["p7poly_an_coverage"] = float(covered.sum() / max(int(anion.sum()), 1))
    return out


# --------------------------------------------------------------------- P8


def _voronoi_index_map(structure) -> dict[tuple[float, float, float], int]:
    index_of: dict[tuple[float, float, float], int] = {}
    for idx, site in enumerate(structure):
        key = tuple(np.round(site.frac_coords % 1.0, 5).tolist())
        index_of.setdefault(key, idx)
    return index_of


def p8_jaccard_site_stats(
    structure,
    formal_valences,
    *,
    neighbors,
    polyhedra,
) -> list[dict | None]:
    charges = np.asarray(formal_valences, dtype=float)
    index_of = _voronoi_index_map(structure)
    stats: list[dict | None] = []
    for center in range(len(structure)):
        cnn_set = {
            int(nb["site_index"])
            for nb in neighbors[center]
            if int(nb["site_index"]) != center
        }
        vor_set = set()
        for entry in polyhedra[center].values():
            key = tuple(np.round(entry["site"].frac_coords % 1.0, 5).tolist())
            found = index_of.get(key)
            if found is not None and found != center:
                vor_set.add(found)
        if not cnn_set and not vor_set:
            stats.append(None)
            continue
        union = cnn_set | vor_set
        if not union:
            stats.append(None)
            continue
        jaccard = len(cnn_set & vor_set) / len(union)
        stats.append({"jaccard": float(1.0 - jaccard)})
    return stats


# --------------------------------------------------------------------- P9


def p9_lewis_features(structure, formal_valences, *, neighbors) -> dict[str, float]:
    charges = np.asarray(formal_valences, dtype=float)
    edges = _hetero_bond_edges(structure, charges, neighbors)
    out: dict[str, float] = {}
    if not edges:
        return out
    cn = np.asarray([len(neighbors[i]) for i in range(len(structure))], dtype=float)
    acidity = np.where(
        (charges > 0) & (cn > 0), charges / np.maximum(cn, 1), np.nan
    )
    basicity = np.where(
        (charges < 0) & (cn > 0), np.abs(charges) / np.maximum(cn, 1), np.nan
    )
    mismatches = []
    per_cation: list[list[float]] = [[] for _ in range(len(structure))]
    for left, right in edges:
        cation = left if charges[left] > 0 else right
        anion = right if charges[left] > 0 else left
        if not (np.isfinite(acidity[cation]) and np.isfinite(basicity[anion])):
            continue
        value = abs(float(acidity[cation]) - float(basicity[anion]))
        mismatches.append(value)
        per_cation[cation].append(value)
    if mismatches:
        array = np.asarray(mismatches)
        out["p9lew_bond_mismatch_mean"] = float(np.mean(array))
        out["p9lew_bond_mismatch_q95"] = float(np.quantile(array, 0.95))
        out["p9lew_bond_mismatch_max"] = float(np.max(array))
    site_means = [np.mean(values) for values in per_cation if values]
    if site_means:
        out["p9lew_cat_site_mismatch_max"] = float(np.max(site_means))
    return out


# -------------------------------------------------------------------- P10


def p10_voronoi_volume_site_stats(
    structure,
    formal_valences,
    *,
    neighbors,
    polyhedra,
    shannon_radius=None,
) -> list[dict | None]:
    charges = np.asarray(formal_valences, dtype=float)
    if shannon_radius is None:
        from phys_law import shannon

        shannon_radius = shannon
    stats: list[dict | None] = []
    for center in range(len(structure)):
        entries = polyhedra[center]
        if not entries:
            stats.append(None)
            continue
        volume = float(sum(float(entry.get("volume", 0.0)) for entry in entries.values()))
        # Off-centering as the solid-angle-weighted normal asymmetry:
        # ||sum_j Omega_j n_j|| / sum_j Omega_j.  (In pymatgen 2026 the face
        # 'verts' are vertex indices without coordinates, so a cell-centroid
        # definition is not available from this API.)
        weighted = np.zeros(3)
        total = 0.0
        for entry in entries.values():
            omega = float(entry.get("solid_angle", 0.0))
            normal = np.asarray(entry.get("normal", []), dtype=float)
            if omega <= 0 or normal.shape != (3,):
                continue
            weighted += omega * normal
            total += omega
        if volume <= 0 or total <= 0:
            stats.append(None)
            continue
        offcenter = float(np.linalg.norm(weighted) / total)
        try:
            reference = float(
                shannon_radius(
                    structure[center].specie.symbol,
                    charges[center],
                    max(len(neighbors[center]), 1),
                )
            )
        except Exception:
            reference = float("nan")
        if not np.isfinite(reference) or reference <= 0:
            stats.append(None)
            continue
        freevol = volume / (4.0 / 3.0 * np.pi * reference**3)
        stats.append(
            {
                "freevol": float(freevol),
                "offcenter": offcenter,
            }
        )
    return stats


# ------------------------------------------------------------ aggregation


def _aggregate(out: dict, prefix: str, stats, charges, metrics, aggregates):
    charges = np.asarray(charges, dtype=float)
    charged = charges != 0
    covered = np.asarray([entry is not None for entry in stats], dtype=bool)
    denominator = int(charged.sum())
    out[f"{prefix}_site_coverage"] = (
        float((covered & charged).sum() / denominator) if denominator else 0.0
    )
    for short, mask in (("cat", charges > 0), ("an", charges < 0)):
        indices = [
            int(index) for index in np.flatnonzero(mask) if stats[int(index)] is not None
        ]
        if not indices:
            continue
        for metric in metrics:
            values = np.asarray(
                [float(stats[index][metric]) for index in indices], dtype=float
            )
            for aggregate in aggregates:
                if aggregate == "mean":
                    value = float(np.mean(values))
                elif aggregate == "q95":
                    value = float(np.quantile(values, 0.95))
                elif aggregate == "max":
                    value = float(np.max(values))
                else:
                    raise ValueError(f"unsupported aggregate: {aggregate}")
                out[f"{prefix}_{short}_{metric}_{aggregate}"] = value


# ------------------------------------------------------------- one struct


def next3_local_features(
    structure,
    formal_valences: Sequence[float],
) -> tuple[dict[str, float], dict[str, int]]:
    """Compute all six frozen families for one structure."""

    charges = np.asarray(formal_valences, dtype=float)
    if len(structure) != len(charges):
        raise ValueError("structure and formal_valences must have equal length")
    out: dict[str, float] = {}
    failures: Counter[str] = Counter()

    neighbors = None
    try:
        neighbors = _crystal_nn_info(structure)
    except Exception as exc:
        failures[f"crystalnn:{type(exc).__name__}"] += 1
    polyhedra = None
    try:
        polyhedra = _voronoi_polyhedra(structure)
    except Exception as exc:
        failures[f"voronoi:{type(exc).__name__}"] += 1
    sphere = None
    try:
        sphere = structure.get_all_neighbors(8.0)
    except Exception as exc:
        failures[f"sphere:{type(exc).__name__}"] += 1

    # P4 ChemEnv CSM (cations only, per the repository recipe).
    try:
        csm_values = p4_csm_site_values(structure, formal_valences)
        finite = [
            value
            for index, value in enumerate(csm_values)
            if value is not None and charges[index] > 0
        ]
        out["p4csm_site_coverage"] = float(
            len(finite) / max(int((charges > 0).sum()), 1)
        )
        if finite:
            array = np.asarray(finite, dtype=float)
            out["p4csm_cat_mean"] = float(np.mean(array))
            out["p4csm_cat_q95"] = float(np.quantile(array, 0.95))
            out["p4csm_cat_max"] = float(np.max(array))
    except Exception as exc:
        failures[f"p4:{type(exc).__name__}"] += 1

    # P6 shell gap.
    if sphere is not None:
        try:
            stats = p6_shell_gap_site_stats(structure, formal_valences, sphere=sphere)
            _aggregate(
                out,
                "p6gap",
                stats,
                charges,
                ("gap_ratio", "shell_width", "gap_pos"),
                ("mean", "max"),
            )
        except Exception as exc:
            failures[f"p6:{type(exc).__name__}"] += 1

    # P7 polyanion contact.
    if sphere is not None:
        try:
            out.update(p7_polyanion_features(structure, formal_valences, sphere=sphere))
        except Exception as exc:
            failures[f"p7:{type(exc).__name__}"] += 1

    # P8 Jaccard disagreement.
    if neighbors is not None and polyhedra is not None:
        try:
            stats = p8_jaccard_site_stats(
                structure,
                formal_valences,
                neighbors=neighbors,
                polyhedra=polyhedra,
            )
            _aggregate(out, "p8nnj", stats, charges, ("jaccard",), ("mean", "max"))
        except Exception as exc:
            failures[f"p8:{type(exc).__name__}"] += 1

    # P9 Lewis matching.
    if neighbors is not None:
        try:
            out.update(p9_lewis_features(structure, formal_valences, neighbors=neighbors))
        except Exception as exc:
            failures[f"p9:{type(exc).__name__}"] += 1

    # P10 Voronoi volume / off-centering.
    if neighbors is not None and polyhedra is not None:
        try:
            stats = p10_voronoi_volume_site_stats(
                structure,
                formal_valences,
                neighbors=neighbors,
                polyhedra=polyhedra,
            )
            _aggregate(
                out,
                "p10vor",
                stats,
                charges,
                ("freevol", "offcenter"),
                ("mean", "max"),
            )
        except Exception as exc:
            failures[f"p10:{type(exc).__name__}"] += 1

    return out, dict(failures)


# ---------------------------------------------------------------- workers


def _real_worker(record: Mapping[str, object]):
    from pymatgen.core import Structure

    from discriminate import guess_oxi, read_blob_cif

    failures: Counter[str] = Counter()
    try:
        structure = Structure.from_str(
            read_blob_cif(int(record["off"]), int(record["ln"])), fmt="cif"
        )
        valences, ok = guess_oxi(structure)
        if not ok:
            failures["valence:guess_oxi"] += 1
            return None, failures
        out, family_failures = next3_local_features(structure, valences)
        failures.update(family_failures)
        out.update(source_id=record["sid"], split=record["split"])
        return out, failures
    except Exception as exc:
        failures[f"structure:{type(exc).__name__}"] += 1
        return None, failures


def _bad_worker(record: Mapping[str, object]):
    from pymatgen.core import Structure

    from discriminate import guess_oxi, read_blob_cif
    from make_negatives import perturb, swapped_val
    from phys_law import seed_of

    failures: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    try:
        structure = Structure.from_str(
            read_blob_cif(int(record["off"]), int(record["ln"])), fmt="cif"
        )
        valences, ok = guess_oxi(structure)
        if not ok:
            failures["valence:guess_oxi"] += 1
            return rows, failures
        rng = np.random.default_rng(seed_of(str(record["sid"])))
        wanted = set(str(record["kinds"]).split(","))
        for kind in ("S1", "S2", "S3", "S4", "S5"):
            changed = perturb(structure, kind, rng, valences)
            if changed is None or kind not in wanted:
                continue
            try:
                out, family_failures = next3_local_features(
                    changed, swapped_val(changed, valences)
                )
                failures.update(family_failures)
                out.update(
                    sid=f"{record['sid']}_{kind}",
                    kind=kind,
                    parent=record["sid"],
                    split=record["split"],
                )
                rows.append(out)
            except Exception as exc:
                failures[f"{kind}:{type(exc).__name__}"] += 1
    except Exception as exc:
        failures[f"structure:{type(exc).__name__}"] += 1
    return rows, failures


def _false_positive_worker(record: Mapping[str, object]):
    from pymatgen.core import Structure

    from discriminate import read_blob_cif
    from polymorph_rank2 import balance

    failures: Counter[str] = Counter()
    try:
        structure = Structure.from_str(
            read_blob_cif(int(record["off"]), int(record["ln"])), fmt="cif"
        )
        if len(structure) > 80:
            failures["structure:too_many_sites"] += 1
            return None, failures
        valence_map = balance(structure.composition.reduced_formula.replace(" ", ""))
        if valence_map is None:
            failures["valence:balance"] += 1
            return None, failures
        valences = [float(valence_map[site.specie.symbol]) for site in structure]
        out, family_failures = next3_local_features(structure, valences)
        failures.update(family_failures)
        out.update(sid=record["sid"], split=record["split"])
        return out, failures
    except Exception as exc:
        failures[f"structure:{type(exc).__name__}"] += 1
        return None, failures


# ------------------------------------------------------------------ driver


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("real", "bad", "false-positive"))
    parser.add_argument("--isolated-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunksize", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--materials-database", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    records = _load_records(
        args.isolated_dir,
        args.mode,
        limit=args.limit,
        materials_database=args.materials_database,
        features_dir=args.features_dir,
    )
    print(f"{args.mode}: resolved {len(records):,} records", flush=True)
    worker = {
        "real": _real_worker,
        "bad": _bad_worker,
        "false-positive": _false_positive_worker,
    }[args.mode]
    single = args.mode in {"real", "false-positive"}
    rows: list[dict[str, object]] = []
    failures: Counter[str] = Counter()

    def consume(result) -> None:
        payload, counter = result
        failures.update(counter)
        if single:
            if payload is not None:
                rows.append(payload)
        else:
            rows.extend(payload)

    if args.workers == 1:
        for index, result in enumerate(map(worker, records), start=1):
            consume(result)
            if index % 500 == 0:
                print(f"  {index:,}/{len(records):,} -> {len(rows):,}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for index, result in enumerate(
                executor.map(worker, records, chunksize=args.chunksize), start=1
            ):
                consume(result)
                if index % 500 == 0:
                    print(f"  {index:,}/{len(records):,} -> {len(rows):,}", flush=True)
    if not rows:
        raise SystemExit("no descriptor rows were produced")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(args.out, index=False)
    feature_columns = sorted(
        column
        for column in frame
        if column.startswith(("p4csm_", "p6gap_", "p7poly_", "p8nnj_", "p9lew_", "p10vor_"))
    )
    coverage = {
        column: float(np.isfinite(frame[column].to_numpy(dtype=float)).mean())
        for column in feature_columns
    }
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "np-next-20260801c",
        "command": [str(value) for value in (argv if argv is not None else sys.argv[1:])],
        "mode": args.mode,
        "n_input_records": len(records),
        "n_output_rows": len(frame),
        "split_counts": {
            str(key): int(value)
            for key, value in frame["split"].value_counts().sort_index().items()
        },
        "feature_columns": feature_columns,
        "feature_finite_coverage": coverage,
        "failure_counts": dict(sorted(failures.items())),
        "valence_source": (
            "discriminate.guess_oxi (composition only)"
            if args.mode != "false-positive"
            else "polymorph_rank2.balance (composition only)"
        ),
        "lockbox_access": False,
        "lockbox_rows_in_output": False,
        "source_access_note": (
            "descriptor records come from the physically isolated tables; "
            "no monolithic split-bearing table is read in this step"
            if args.mode != "false-positive"
            else "false-positive audit inputs contain no experimental lockbox rows"
        ),
        "implementation_sha256": _sha256(Path(__file__)),
    }
    metadata_path = args.out.with_suffix(args.out.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {args.out} ({len(frame):,} rows)", flush=True)
    print(f"wrote {metadata_path}", flush=True)
    if failures:
        print(f"failure counts: {dict(sorted(failures.items()))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
