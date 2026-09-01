#!/usr/bin/env python3
"""Corrected P2/P6/P7/P9 descriptors for experiment np-next-20260801d.

This additive module leaves every np-next-20260801c implementation and cache
unchanged.  It fixes periodic-neighbour handling in P6/P7, uses all anion sites
in the P7 short-contact denominator, defines P9 coordination on the same
opposite-sign periodic graph as its mismatch edges, and applies one formal-
valence cascade to every dataset role.
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

from apply_rules import frac_oxi
from discriminate import guess_oxi
from polymorph_rank2 import balance
from next_features import _crystal_nn_info, _load_records, _voronoi_polyhedra

NEIGHBOR_EPSILON = 1e-8
P6_MAX_SHELL = 12
P7_SPHERE_CUTOFF = 8.0
P7_SHORT_CONTACT_RATIO = 1.3

# Frozen from pymatgen's Element.atomic_radius table on 2026-08-01.  Keeping
# the values here makes the descriptor independent of later API/table drift.
P7_ATOMIC_RADII = {
    "H": 0.25,
    "B": 0.85,
    "C": 0.70,
    "N": 0.65,
    "O": 0.60,
    "F": 0.50,
    "Si": 1.10,
    "P": 1.00,
    "S": 1.00,
    "Cl": 1.00,
    "As": 1.15,
    "Se": 1.15,
    "Br": 1.15,
    "Sb": 1.45,
    "Te": 1.40,
    "I": 1.40,
}

if min(P7_SPHERE_CUTOFF / (2.0 * radius) for radius in P7_ATOMIC_RADII.values()) <= P7_SHORT_CONTACT_RATIO:
    raise RuntimeError("the frozen P7 cutoff cannot safely right-censor all radii")


def _validated_valences(structure, values) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.shape != (len(structure),) or not np.isfinite(array).all():
        return None
    return array


def infer_formal_valences(structure) -> tuple[np.ndarray, str]:
    """Apply one composition-only valence cascade, independent of dataset role."""

    guessed, ok = guess_oxi(structure)
    values = _validated_valences(structure, guessed if ok else None)
    if values is not None:
        return values, "guess_oxi"

    values = _validated_valences(structure, frac_oxi(structure))
    if values is not None:
        return values, "frac_oxi"

    mapping = balance(structure.composition.reduced_formula.replace(" ", ""))
    if mapping is not None:
        try:
            values = _validated_valences(
                structure,
                [float(mapping[site.specie.symbol]) for site in structure],
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            values = None
        if values is not None:
            return values, "balance"
    raise ValueError("formal-valence cascade failed")


def _image_key(neighbor) -> tuple[int, int, int]:
    image = getattr(neighbor, "image", None)
    if image is None and isinstance(neighbor, Mapping):
        image = neighbor.get("image")
    if image is None:
        return (0, 0, 0)
    return tuple(int(round(float(value))) for value in image)


def _neighbor_index(neighbor) -> int:
    if isinstance(neighbor, Mapping):
        return int(neighbor["site_index"])
    return int(neighbor.index)


def _periodic_distances(neighbors) -> list[tuple[float, int, tuple[int, int, int]]]:
    unique: dict[tuple[int, tuple[int, int, int]], float] = {}
    for neighbor in neighbors:
        distance = float(neighbor.nn_distance)
        if not np.isfinite(distance) or distance <= NEIGHBOR_EPSILON:
            continue
        index = _neighbor_index(neighbor)
        image = _image_key(neighbor)
        key = (index, image)
        unique[key] = min(distance, unique.get(key, distance))
    return sorted(
        (distance, index, image)
        for (index, image), distance in unique.items()
    )


def p2c_voronoi_site_stats(
    structure,
    formal_valences: Sequence[float],
    *,
    polyhedra=None,
) -> list[dict[str, float] | None]:
    """P2 recomputed under the shared valence policy, retaining periodic faces."""

    charges = np.asarray(formal_valences, dtype=float)
    if charges.shape != (len(structure),):
        raise ValueError("structure and formal_valences must have equal length")
    if polyhedra is None:
        polyhedra = _voronoi_polyhedra(structure)
    if len(polyhedra) != len(structure):
        raise ValueError("polyhedra must provide one entry per site")

    def original_index(neighbor_site) -> int | None:
        for index, site in enumerate(structure):
            if neighbor_site.is_periodic_image(site):
                return index
        return None

    stats: list[dict[str, float] | None] = []
    for center, entries in enumerate(polyhedra):
        omegas: list[float] = []
        like: list[bool] = []
        for entry in entries.values():
            other = original_index(entry["site"])
            if other is None:
                continue
            omega = float(entry.get("solid_angle", float("nan")))
            if not np.isfinite(omega) or omega <= 0:
                continue
            omegas.append(omega)
            like.append(
                charges[other] != 0
                and charges[center] != 0
                and np.sign(charges[other]) == np.sign(charges[center])
            )
        if not omegas:
            stats.append(None)
            continue
        weights = np.asarray(omegas, dtype=float)
        probabilities = weights / weights.sum()
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        stats.append(
            {
                "sa_effective_cn": float(np.exp(entropy)),
                "sa_like_fraction": float(
                    np.sum(probabilities[np.asarray(like, dtype=bool)])
                ),
                "sa_max_fraction": float(np.max(probabilities)),
            }
        )
    return stats


def p6c_shell_gap_site_stats(
    structure,
    formal_valences: Sequence[float],
    *,
    sphere,
) -> list[dict[str, float] | None]:
    """Corrected P6: keep nonzero periodic images, including self-images."""

    if len(structure) != len(formal_valences) or len(sphere) != len(structure):
        raise ValueError("structure, valences, and sphere must have matching lengths")
    stats: list[dict[str, float] | None] = []
    for neighbors in sphere:
        distances = [entry[0] for entry in _periodic_distances(neighbors)]
        if len(distances) < 2:
            stats.append(None)
            continue
        ratios = np.asarray(
            [
                distances[index + 1] / distances[index]
                for index in range(min(len(distances) - 1, P6_MAX_SHELL))
            ],
            dtype=float,
        )
        gap_index = int(np.argmax(ratios))
        stats.append(
            {
                "gap_ratio": float(ratios[gap_index]),
                "gap_pos": float(gap_index + 1),
                "shell_width": float(distances[gap_index] / distances[0]),
            }
        )
    return stats


def p7c_polyanion_features(
    structure,
    formal_valences: Sequence[float],
    *,
    sphere,
) -> dict[str, float]:
    """Corrected P7 with periodic self-images and right-censored no-contact sites."""

    charges = np.asarray(formal_valences, dtype=float)
    if charges.shape != (len(structure),) or len(sphere) != len(structure):
        raise ValueError("structure, valences, and sphere must have matching lengths")
    anion_indices = np.flatnonzero(charges < 0)
    if not len(anion_indices):
        return {}

    ratios: list[float] = []
    short: list[bool] = []
    observed = 0
    for center in anion_indices:
        symbol = structure[int(center)].specie.symbol
        radius = P7_ATOMIC_RADII.get(symbol)
        if radius is None or radius <= 0:
            raise ValueError(f"no frozen P7 radius for anion {symbol}")
        same_species = []
        for distance, other, _image in _periodic_distances(sphere[int(center)]):
            if charges[other] >= 0:
                continue
            if structure[other].specie.symbol != symbol:
                continue
            same_species.append(distance / (2.0 * radius))
        if same_species:
            ratio = float(min(same_species))
            observed += 1
            ratios.append(ratio)
            short.append(ratio < P7_SHORT_CONTACT_RATIO)
        else:
            ratios.append(P7_SPHERE_CUTOFF / (2.0 * radius))
            short.append(False)

    total = len(anion_indices)
    return {
        "p7c_an_contact_min": float(min(ratios)),
        "p7c_an_short_contact_frac": float(np.mean(short)),
        "p7c_an_within8_fraction": float(observed / total),
        "p7c_an_censored_fraction": float(1.0 - observed / total),
        "p7c_radius_available": 1.0,
    }


def _periodic_opposite_edges(
    structure,
    charges: np.ndarray,
    neighbors: Sequence[Sequence[Mapping[str, object]]],
) -> list[tuple[int, int, tuple[int, int, int]]]:
    edges: set[tuple[int, int, tuple[int, int, int]]] = set()
    for center, entries in enumerate(neighbors):
        for entry in entries:
            other = _neighbor_index(entry)
            if other < 0 or other >= len(structure):
                continue
            if charges[center] == 0 or charges[other] == 0:
                continue
            if np.sign(charges[center]) == np.sign(charges[other]):
                continue
            image = _image_key(entry)
            distance = float(structure.get_distance(center, other, jimage=image))
            if not np.isfinite(distance) or distance <= NEIGHBOR_EPSILON:
                continue
            forward = (center, other, image)
            reverse = (other, center, tuple(-value for value in image))
            edges.add(min(forward, reverse))
    return sorted(edges)


def p9c_lewis_features(
    structure,
    formal_valences: Sequence[float],
    *,
    neighbors: Sequence[Sequence[Mapping[str, object]]],
) -> dict[str, float]:
    """Corrected P9 using degree in the opposite-sign periodic graph."""

    charges = np.asarray(formal_valences, dtype=float)
    if charges.shape != (len(structure),) or len(neighbors) != len(structure):
        raise ValueError("structure, valences, and neighbors must have matching lengths")
    edges = _periodic_opposite_edges(structure, charges, neighbors)
    if not edges:
        return {}
    degree = np.zeros(len(structure), dtype=float)
    for left, right, _image in edges:
        degree[left] += 1.0
        degree[right] += 1.0

    mismatches: list[float] = []
    per_cation: list[list[float]] = [[] for _ in range(len(structure))]
    for left, right, _image in edges:
        cation = left if charges[left] > 0 else right
        anion = right if charges[left] > 0 else left
        if degree[cation] <= 0 or degree[anion] <= 0:
            continue
        acidity = float(charges[cation] / degree[cation])
        basicity = float(abs(charges[anion]) / degree[anion])
        mismatch = abs(acidity - basicity)
        mismatches.append(mismatch)
        per_cation[cation].append(mismatch)
    if not mismatches:
        return {}
    values = np.asarray(mismatches, dtype=float)
    site_means = [float(np.mean(items)) for items in per_cation if items]
    return {
        "p9c_bond_mismatch_mean": float(np.mean(values)),
        "p9c_bond_mismatch_q95": float(np.quantile(values, 0.95)),
        "p9c_bond_mismatch_max": float(np.max(values)),
        "p9c_cat_site_mismatch_max": float(max(site_means)),
    }


def _aggregate_sites(
    out: dict[str, float],
    prefix: str,
    site_stats: Sequence[Mapping[str, float] | None],
    charges: np.ndarray,
    metrics: Sequence[str],
    aggregates: Sequence[str],
) -> None:
    charged = charges != 0
    covered = np.asarray([entry is not None for entry in site_stats], dtype=bool)
    out[f"{prefix}_site_coverage"] = float(
        (covered & charged).sum() / max(int(charged.sum()), 1)
    )
    for short, mask in (("cat", charges > 0), ("an", charges < 0)):
        indices = [
            int(index)
            for index in np.flatnonzero(mask)
            if site_stats[int(index)] is not None
        ]
        for metric in metrics:
            values = np.asarray(
                [float(site_stats[index][metric]) for index in indices], dtype=float
            )
            if not len(values):
                continue
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


def next4_local_features(
    structure,
    formal_valences: Sequence[float],
) -> tuple[dict[str, float], dict[str, int]]:
    """Compute corrected P2/P6/P7/P9 while isolating family failures."""

    charges = np.asarray(formal_valences, dtype=float)
    if charges.shape != (len(structure),):
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
        sphere = structure.get_all_neighbors(P7_SPHERE_CUTOFF)
    except Exception as exc:
        failures[f"sphere:{type(exc).__name__}"] += 1

    if polyhedra is not None:
        try:
            stats = p2c_voronoi_site_stats(
                structure, formal_valences, polyhedra=polyhedra
            )
            _aggregate_sites(
                out,
                "p2c",
                stats,
                charges,
                ("sa_effective_cn", "sa_like_fraction", "sa_max_fraction"),
                ("mean", "q95", "max"),
            )
        except Exception as exc:
            failures[f"p2:{type(exc).__name__}"] += 1

    if sphere is not None:
        try:
            stats = p6c_shell_gap_site_stats(
                structure, formal_valences, sphere=sphere
            )
            _aggregate_sites(
                out,
                "p6c",
                stats,
                charges,
                ("gap_ratio", "shell_width", "gap_pos"),
                ("mean", "max"),
            )
        except Exception as exc:
            failures[f"p6:{type(exc).__name__}"] += 1
        try:
            out.update(
                p7c_polyanion_features(
                    structure, formal_valences, sphere=sphere
                )
            )
        except Exception as exc:
            failures[f"p7:{type(exc).__name__}"] += 1

    if neighbors is not None:
        try:
            out.update(
                p9c_lewis_features(
                    structure, formal_valences, neighbors=neighbors
                )
            )
        except Exception as exc:
            failures[f"p9:{type(exc).__name__}"] += 1
    return out, dict(failures)


def _read_structure(record: Mapping[str, object]):
    from pymatgen.core import Structure

    from discriminate import read_blob_cif

    return Structure.from_str(
        read_blob_cif(int(record["off"]), int(record["ln"])), fmt="cif"
    )


def _real_worker(record: Mapping[str, object]):
    failures: Counter[str] = Counter()
    try:
        structure = _read_structure(record)
        if len(structure) > 80:
            failures["structure:too_many_sites"] += 1
            return None, failures
        valences, source = infer_formal_valences(structure)
        out, family_failures = next4_local_features(structure, valences)
        failures.update(family_failures)
        out.update(
            source_id=record["sid"],
            split=record["split"],
            next4_valence_source=source,
        )
        return out, failures
    except ValueError as exc:
        failures[f"valence:{type(exc).__name__}"] += 1
        return None, failures
    except Exception as exc:
        failures[f"structure:{type(exc).__name__}"] += 1
        return None, failures


def _bad_worker(record: Mapping[str, object]):
    from make_negatives import perturb, swapped_val
    from phys_law import seed_of

    failures: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    try:
        structure = _read_structure(record)
        if len(structure) > 80:
            failures["structure:too_many_sites"] += 1
            return rows, failures
        valences, source = infer_formal_valences(structure)
        rng = np.random.default_rng(seed_of(str(record["sid"])))
        wanted = set(str(record["kinds"]).split(","))
        for kind in ("S1", "S2", "S3", "S4", "S5"):
            changed = perturb(structure, kind, rng, valences)
            if changed is None or kind not in wanted:
                continue
            try:
                out, family_failures = next4_local_features(
                    changed, swapped_val(changed, valences)
                )
                failures.update(family_failures)
                out.update(
                    sid=f"{record['sid']}_{kind}",
                    kind=kind,
                    parent=record["sid"],
                    split=record["split"],
                    next4_valence_source=source,
                )
                rows.append(out)
            except Exception as exc:
                failures[f"{kind}:{type(exc).__name__}"] += 1
    except ValueError as exc:
        failures[f"valence:{type(exc).__name__}"] += 1
    except Exception as exc:
        failures[f"structure:{type(exc).__name__}"] += 1
    return rows, failures


def _false_positive_worker(record: Mapping[str, object]):
    failures: Counter[str] = Counter()
    try:
        structure = _read_structure(record)
        if len(structure) > 80:
            failures["structure:too_many_sites"] += 1
            return None, failures
        valences, source = infer_formal_valences(structure)
        out, family_failures = next4_local_features(structure, valences)
        failures.update(family_failures)
        out.update(
            sid=record["sid"],
            split=record["split"],
            next4_valence_source=source,
        )
        return out, failures
    except ValueError as exc:
        failures[f"valence:{type(exc).__name__}"] += 1
        return None, failures
    except Exception as exc:
        failures[f"structure:{type(exc).__name__}"] += 1
        return None, failures


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
                    print(
                        f"  {index:,}/{len(records):,} -> {len(rows):,}",
                        flush=True,
                    )
    if not rows:
        raise SystemExit("no descriptor rows were produced")

    frame = pd.DataFrame(rows)
    if frame["split"].astype(str).str.lower().eq("lockbox").any():
        raise SystemExit("refusing to write a descriptor cache containing lockbox rows")
    feature_columns = sorted(
        column
        for column in frame
        if column.startswith(("p2c_", "p6c_", "p7c_", "p9c_"))
    )
    coverage = {
        column: float(np.isfinite(frame[column].to_numpy(dtype=float)).mean())
        for column in feature_columns
    }
    valence_counts = {
        str(key): int(value)
        for key, value in frame["next4_valence_source"]
        .value_counts()
        .sort_index()
        .items()
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)

    from discriminate import BLOB

    input_paths: dict[str, Path] = {
        "structures_blob": Path(BLOB),
        "isolated_manifest": args.isolated_dir / "isolated_manifest.json",
    }
    if args.mode == "real":
        input_paths["records"] = args.isolated_dir / "records_real.parquet"
    elif args.mode == "bad":
        input_paths["records"] = args.isolated_dir / "records_bad.parquet"
    else:
        if args.features_dir is not None:
            input_paths["false_positive_ids"] = (
                args.features_dir / "false_positive.parquet"
            )
        if args.materials_database is not None:
            input_paths["materials_database"] = args.materials_database
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "np-next-20260801d",
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
        "valence_policy": ["guess_oxi", "frac_oxi", "balance"],
        "valence_source_counts": valence_counts,
        "lockbox_access": False,
        "lockbox_rows_in_output": False,
        "input_sha256": {
            name: _sha256(path) for name, path in input_paths.items() if path.exists()
        },
        "implementation_sha256": _sha256(Path(__file__)),
    }
    metadata_path = args.out.with_suffix(args.out.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(f"wrote {args.out} ({len(frame):,} rows)", flush=True)
    print(f"wrote {metadata_path}", flush=True)
    if failures:
        print(f"failure counts: {dict(sorted(failures.items()))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
