#!/usr/bin/env python3
"""P2/P3/P5 local descriptors (plus a recomputed P1) for np-next-20260801.

Families (frozen in docs/plans/2026-08-01-p235-isolated-design.md):

- P1  bond-valence local triplet, recomputed through
      ``advanced_local_features.bond_valence_local_features`` so the
      definitions stay byte-identical to the previous round.  Searchable
      ``bvloc_*`` columns use the frozen-fallback parameter policy; the
      exact policy contributes only ``bvlocx_*`` coverage diagnostics.
- P2  Voronoi solid-angle entropy CN, like-charge solid-angle share, and
      dominant-neighbour share from one tessellation per structure.
- P3  Hawthorne prior-bond-strength network diagnostics on the CrystalNN
      bond graph: NNLS/min-norm residuals, Pauling gap, rank deficiency.
      Topology and formal valences only; no bond lengths.
- P5  strict iterative Hoppe ECoN/MEFIR on the full 8 Angstrom neighbour
      sphere, plus per-site deltas against the existing CrystalNN-set
      approximations recomputed here.

Formal valences always come from ``discriminate.guess_oxi`` (composition
only); ``BVAnalyzer`` is never called.  Descriptor rows are written to an
external cache directory, never into the repository.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

P2_METRICS = ("sa_effective_cn", "sa_like_fraction", "sa_max_fraction")
P2_AGGREGATES = ("mean", "q95", "max")

P5HOP_SPHERE_CUTOFF = 8.0
P5HOP_MEFIR_MAX_ITER = 400
P5HOP_MEFIR_TOL = 1e-4
P5HOP_MEFIR_DAMPING = 0.5
P5HOP_MEFIR_CLIP = (0.2, 3.0)


# --------------------------------------------------------------------- P2


def _voronoi_polyhedra(structure):
    from pymatgen.analysis.local_env import VoronoiNN

    return VoronoiNN().get_all_voronoi_polyhedra(structure)


def p2_voronoi_site_stats(
    structure,
    formal_valences: Sequence[float],
    *,
    polyhedra=None,
) -> list[dict[str, float] | None]:
    """Per-site solid-angle statistics; None where the tessellation failed."""

    charges = np.asarray(formal_valences, dtype=float)
    if len(structure) != len(charges):
        raise ValueError("structure and formal_valences must have equal length")
    if polyhedra is None:
        polyhedra = _voronoi_polyhedra(structure)
    if len(polyhedra) != len(structure):
        raise ValueError("polyhedra must provide one entry per site")

    index_of: dict[tuple[float, float, float], int] = {}
    for idx, site in enumerate(structure):
        key = tuple(np.round(site.frac_coords % 1.0, 5).tolist())
        index_of.setdefault(key, idx)

    def original_index(neighbor_site) -> int | None:
        key = tuple(np.round(neighbor_site.frac_coords % 1.0, 5).tolist())
        found = index_of.get(key)
        if found is not None:
            return found
        for idx, site in enumerate(structure):
            if neighbor_site.is_periodic_image(site):
                return idx
        return None

    stats: list[dict[str, float] | None] = []
    for center in range(len(structure)):
        entries = polyhedra[center]
        omegas: list[float] = []
        like: list[bool] = []
        for entry in entries.values():
            other = original_index(entry["site"])
            if other is None or other == center:
                continue
            omega = float(entry["solid_angle"])
            if not np.isfinite(omega) or omega <= 0:
                continue
            omegas.append(omega)
            like.append(
                charges[other] != 0
                and np.sign(charges[other]) == np.sign(charges[center])
            )
        if not omegas:
            stats.append(None)
            continue
        weights = np.asarray(omegas, dtype=float)
        total = float(weights.sum())
        if total <= 0:
            stats.append(None)
            continue
        probabilities = weights / total
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        like_fraction = (
            float(np.sum(probabilities[np.asarray(like, dtype=bool)]))
            if like
            else 0.0
        )
        stats.append(
            {
                "sa_effective_cn": float(np.exp(entropy)),
                "sa_like_fraction": like_fraction,
                "sa_max_fraction": float(np.max(probabilities)),
            }
        )
    return stats


# --------------------------------------------------------------------- P3


def _hetero_bond_edges(
    structure,
    charges: np.ndarray,
    neighbors: Sequence[Sequence[Mapping[str, object]]],
) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for center in range(len(structure)):
        if charges[center] == 0:
            continue
        for neighbor in neighbors[center]:
            other = int(neighbor["site_index"])
            if other == center or other < 0 or other >= len(structure):
                continue
            if charges[other] == 0 or np.sign(charges[other]) == np.sign(
                charges[center]
            ):
                continue
            edges.add((min(center, other), max(center, other)))
    return sorted(edges)


def p3_hawthorne_features(
    structure,
    formal_valences: Sequence[float],
    *,
    neighbors: Sequence[Sequence[Mapping[str, object]]],
) -> dict[str, float]:
    """Topology+valence bond-strength network diagnostics (no bond lengths)."""

    charges = np.asarray(formal_valences, dtype=float)
    if len(structure) != len(charges):
        raise ValueError("structure and formal_valences must have equal length")
    if len(neighbors) != len(structure):
        raise ValueError("neighbors must provide one list per structure site")

    edges = _hetero_bond_edges(structure, charges, neighbors)
    charged = np.flatnonzero(charges != 0)
    out: dict[str, float] = {}
    if not len(charged):
        return out
    degree = np.zeros(len(structure), dtype=int)
    for left, right in edges:
        degree[left] += 1
        degree[right] += 1
    unbonded = int(sum(degree[index] == 0 for index in charged))
    out["p3haw_unbonded_charged_fraction"] = float(unbonded / len(charged))
    out["p3haw_n_bonds"] = float(len(edges))
    if not edges:
        return out

    row_of = {int(site): row for row, site in enumerate(charged)}
    incidence = np.zeros((len(charged), len(edges)), dtype=float)
    for column, (left, right) in enumerate(edges):
        incidence[row_of[left], column] = 1.0
        incidence[row_of[right], column] = 1.0
    target = np.abs(charges[charged])
    target_norm = float(np.linalg.norm(target))

    from scipy.sparse.linalg import lsqr
    from scipy.optimize import nnls
    from scipy import sparse

    solution = lsqr(
        sparse.csr_matrix(incidence),
        target,
        atol=1e-12,
        btol=1e-12,
        iter_lim=max(1000, 20 * len(edges)),
    )
    s_star = np.asarray(solution[0], dtype=float)
    minnorm_relres = (
        float(np.linalg.norm(incidence @ s_star - target) / target_norm)
        if target_norm > 0
        else float("nan")
    )
    s_plus, rnorm = nnls(incidence, target)
    nnls_relres = float(rnorm / target_norm) if target_norm > 0 else float("nan")
    site_relres = np.abs(incidence @ s_plus - target) / np.maximum(target, 1e-12)

    s_pauling = np.zeros(len(edges), dtype=float)
    for column, (left, right) in enumerate(edges):
        cation = left if charges[left] > 0 else right
        s_pauling[column] = abs(charges[cation]) / max(degree[cation], 1)
    pauling_norm = float(np.linalg.norm(s_pauling))
    pauling_gap = (
        float(np.linalg.norm(s_star - s_pauling) / pauling_norm)
        if pauling_norm > 0
        else float("nan")
    )
    rank = int(np.linalg.matrix_rank(incidence))
    out.update(
        {
            "p3haw_nnls_relres": nnls_relres,
            "p3haw_minnorm_relres": minnorm_relres,
            "p3haw_pauling_gap": pauling_gap,
            "p3haw_rank_deficiency": float((len(edges) - rank) / len(edges)),
            "p3haw_site_relres_q95": float(np.quantile(site_relres, 0.95)),
            "p3haw_site_relres_max": float(np.max(site_relres)),
        }
    )
    return out


# --------------------------------------------------------------------- P5


def _neighbor_distances(
    structure,
    neighbors: Sequence[Sequence[Mapping[str, object]]],
) -> list[list[tuple[int, float]]]:
    """(site_index, distance) pairs honouring periodic images."""
    out: list[list[tuple[int, float]]] = []
    for center in range(len(structure)):
        pairs: list[tuple[int, float]] = []
        for neighbor in neighbors[center]:
            other = int(neighbor["site_index"])
            if other == center or other < 0 or other >= len(structure):
                continue
            image_value = neighbor.get("image")
            image = (
                np.zeros(3, dtype=float)
                if image_value is None
                else np.asarray(image_value, dtype=float)
            )
            displacement = structure.lattice.get_cartesian_coords(
                structure[other].frac_coords + image - structure[center].frac_coords
            )
            distance = float(np.linalg.norm(displacement))
            if np.isfinite(distance) and distance > 0:
                pairs.append((other, distance))
        out.append(pairs)
    return out


def _econ_weighted_sum(distances: np.ndarray) -> float:
    minimum = float(distances.min())
    weights = np.exp(1.0 - (distances / minimum) ** 6)
    return float(weights.sum())


def p5_hoppe_features(
    structure,
    formal_valences: Sequence[float],
    *,
    neighbors: Sequence[Sequence[Mapping[str, object]]],
    sphere=None,
    shannon_radius=None,
) -> dict[str, float]:
    """Strict-sphere ECoN and iterative MEFIR with approximation deltas."""

    charges = np.asarray(formal_valences, dtype=float)
    if len(structure) != len(charges):
        raise ValueError("structure and formal_valences must have equal length")
    if sphere is None:
        sphere = structure.get_all_neighbors(P5HOP_SPHERE_CUTOFF)
    if shannon_radius is None:
        from phys_law import shannon

        shannon_radius = shannon

    n = len(structure)
    econ_strict = np.full(n, np.nan)
    sphere_pairs: list[list[tuple[int, float]]] = []
    for center in range(n):
        pairs = [
            (int(nb.index), float(nb.nn_distance))
            for nb in sphere[center]
            if int(nb.index) != center and float(nb.nn_distance) > 0
        ]
        sphere_pairs.append(pairs)
        if pairs:
            econ_strict[center] = _econ_weighted_sum(
                np.asarray([distance for _, distance in pairs], dtype=float)
            )

    crystal_pairs = _neighbor_distances(structure, neighbors)
    econ_approx = np.full(n, np.nan)
    for center in range(n):
        if crystal_pairs[center]:
            econ_approx[center] = _econ_weighted_sum(
                np.asarray(
                    [distance for _, distance in crystal_pairs[center]],
                    dtype=float,
                )
            )
    econ_delta = econ_strict - econ_approx

    # Iterative MEFIR on opposite-sign sphere neighbours.
    initial = np.full(n, np.nan)
    for center in range(n):
        try:
            initial[center] = float(
                shannon_radius(
                    structure[center].specie.symbol,
                    charges[center],
                    max(len(neighbors[center]), 1),
                )
            )
        except Exception:
            initial[center] = np.nan
    mefir_pairs: list[list[tuple[int, float]]] = []
    for center in range(n):
        if charges[center] == 0:
            mefir_pairs.append([])
            continue
        mefir_pairs.append(
            [
                (other, distance)
                for other, distance in sphere_pairs[center]
                if charges[other] != 0
                and np.sign(charges[other]) != np.sign(charges[center])
            ]
        )
    active = np.asarray(
        [
            bool(mefir_pairs[center])
            and np.isfinite(initial[center])
            and initial[center] > 0
            and charges[center] != 0
            for center in range(n)
        ]
    )
    # Row-normalised weight matrix over opposite-sign partners with finite
    # initial radii, plus the constant term c_i = sum_j wbar_ij d_ij.  The
    # fixed-point equations R_i = c_i - sum_j wbar_ij R_j are then iterated
    # with sparse matvecs instead of per-site Python loops.
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    constant = np.zeros(n)
    for center in range(n):
        partners = [
            (other, distance)
            for other, distance in mefir_pairs[center]
            if np.isfinite(initial[other]) and initial[other] > 0
        ]
        if not partners:
            continue
        distances = np.asarray([distance for _, distance in partners], dtype=float)
        minimum = float(distances.min())
        weights = np.exp(1.0 - (distances / minimum) ** 6)
        weights /= float(weights.sum())
        for (other, distance), weight in zip(partners, weights):
            rows.append(center)
            cols.append(other)
            data.append(float(weight))
        constant[center] = float(np.sum(weights * distances))

    from scipy import sparse as _sparse

    weight_matrix = _sparse.csr_matrix(
        (np.asarray(data), (np.asarray(rows), np.asarray(cols))), shape=(n, n)
    )
    active = (
        active
        & np.isfinite(initial)
        & (np.asarray(weight_matrix.sum(axis=1)).ravel() > 0)
    )
    radii = initial.copy()
    converged = False
    iterations = 0
    if active.any():
        # Hoppe's alternating scheme: relax one charge partition from the
        # other, in place, then swap.  On the (near-)bipartite bond graph
        # this is the block Gauss-Seidel form of the Jacobi iteration; the
        # fixed point is unchanged.
        partitions = (
            np.flatnonzero(active & (charges > 0)),
            np.flatnonzero(active & (charges < 0)),
        )
        for step in range(P5HOP_MEFIR_MAX_ITER):
            previous = radii.copy()
            for group in partitions:
                estimate = constant[group] - weight_matrix[group] @ radii
                radii[group] = np.clip(
                    radii[group] + P5HOP_MEFIR_DAMPING * (estimate - radii[group]),
                    *P5HOP_MEFIR_CLIP,
                )
            delta = float(np.max(np.abs(radii - previous)[active]))
            iterations = step + 1
            if delta < P5HOP_MEFIR_TOL:
                converged = True
                break
    mefir_rel = (radii - initial) / initial
    mefir_rel[~active] = np.nan

    # Existing one-sided approximation (geom_feat formula, per site, both
    # charge signs) on the CrystalNN neighbour set.
    mefir_rel_approx = np.full(n, np.nan)
    for center in range(n):
        if charges[center] == 0 or not np.isfinite(initial[center]):
            continue
        partners = [
            (other, distance)
            for other, distance in crystal_pairs[center]
            if charges[other] != 0
            and np.sign(charges[other]) != np.sign(charges[center])
        ]
        if not partners:
            continue
        distances = np.asarray([distance for _, distance in partners], dtype=float)
        minimum = float(distances.min())
        weights = np.exp(1.0 - (distances / minimum) ** 6)
        reference = []
        for other, _ in partners:
            try:
                reference.append(
                    float(
                        shannon_radius(
                            structure[other].specie.symbol,
                            charges[other],
                            max(len(neighbors[other]), 1),
                        )
                    )
                )
            except Exception:
                reference.append(np.nan)
        reference = np.asarray(reference, dtype=float)
        usable = np.isfinite(reference) & (reference > 0)
        if not usable.any():
            continue
        estimate = float(
            np.sum(weights[usable] * (distances[usable] - reference[usable]))
            / np.sum(weights[usable])
        )
        mefir_rel_approx[center] = (estimate - initial[center]) / initial[center]
    mefir_delta = mefir_rel - mefir_rel_approx

    out: dict[str, float] = {}
    charged = charges != 0
    for short, mask in (("cat", charges > 0), ("an", charges < 0)):
        base = mask & charged
        for name, values in (
            ("econ_strict", econ_strict),
            ("econ_delta", econ_delta),
            ("mefir_rel", mefir_rel),
            ("mefir_delta", mefir_delta),
        ):
            finite = base & np.isfinite(values)
            if not finite.any():
                continue
            selected = values[finite]
            out[f"p5hop_{short}_{name}_mean"] = float(np.mean(selected))
            out[f"p5hop_{short}_{name}_max"] = float(np.max(selected))
            if name == "mefir_rel":
                out[f"p5hop_{short}_{name}_min"] = float(np.min(selected))
    out["p5hop_econ_site_coverage"] = float(
        (charged & np.isfinite(econ_strict)).sum() / max(int(charged.sum()), 1)
    )
    out["p5hop_mefir_site_coverage"] = float(
        int((active & charged).sum()) / max(int(charged.sum()), 1)
    )
    out["p5hop_mefir_converged_fraction"] = (
        1.0 if converged or not active.any() else 0.0
    )
    out["p5hop_mefir_iterations"] = float(iterations)
    return out


# ------------------------------------------------------------ aggregation


def _aggregate_sites(
    out: dict[str, float],
    prefix: str,
    site_stats: Sequence[Mapping[str, float] | None],
    formal_valences: Sequence[float],
    metrics: Sequence[str],
    aggregates: Sequence[str],
) -> None:
    charges = np.asarray(formal_valences, dtype=float)
    charged = charges != 0
    covered = np.asarray([stats is not None for stats in site_stats], dtype=bool)
    denominator = int(charged.sum())
    out[f"{prefix}_site_coverage"] = (
        float((covered & charged).sum() / denominator) if denominator else 0.0
    )
    for short, mask in (("cat", charges > 0), ("an", charges < 0)):
        indices = [
            index
            for index in np.flatnonzero(mask)
            if site_stats[int(index)] is not None
        ]
        if not indices:
            continue
        for metric in metrics:
            values = [float(site_stats[index][metric]) for index in indices]
            array = np.asarray(values, dtype=float)
            for aggregate in aggregates:
                if aggregate == "mean":
                    value = float(np.mean(array))
                elif aggregate == "q95":
                    value = float(np.quantile(array, 0.95))
                elif aggregate == "max":
                    value = float(np.max(array))
                else:
                    raise ValueError(f"unsupported aggregate: {aggregate}")
                out[f"{prefix}_{short}_{metric}_{aggregate}"] = value


# ------------------------------------------------------------- one struct


def _crystal_nn_info(structure):
    from pymatgen.analysis.local_env import CrystalNN

    finder = CrystalNN(weighted_cn=False, x_diff_weight=0.0)
    built = []
    for index in range(len(structure)):
        try:
            built.append(finder.get_nn_info(structure, index))
        except Exception:
            built.append([])
    return built


def next_local_features(
    structure,
    formal_valences: Sequence[float],
    *,
    parameter_policy: str = "frozen-fallback",
) -> tuple[dict[str, float], dict[str, int]]:
    """Compute all frozen families for one structure.

    Returns the feature dictionary and a per-family failure counter.  A
    failed family leaves its columns absent (offline semantics treat them as
    unknown/abstain) and increments exactly one counter key.
    """

    from advanced_local_features import bond_valence_local_features

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

    # P1 (searchable columns under the frozen-fallback policy).
    if neighbors is not None:
        try:
            p1 = bond_valence_local_features(
                structure,
                formal_valences,
                neighbors=neighbors,
                parameter_policy=parameter_policy,
            )
            out.update(p1)
        except Exception as exc:
            failures[f"p1:{type(exc).__name__}"] += 1
        try:
            p1_exact = bond_valence_local_features(
                structure,
                formal_valences,
                neighbors=neighbors,
                parameter_policy="exact",
            )
            out["bvlocx_site_coverage"] = p1_exact.get("bvloc_site_coverage")
            out["bvlocx_bond_parameter_coverage"] = p1_exact.get(
                "bvloc_bond_parameter_coverage"
            )
        except Exception as exc:
            failures[f"p1_exact:{type(exc).__name__}"] += 1

    # P2 Voronoi solid angle.
    try:
        polyhedra = _voronoi_polyhedra(structure)
        stats = p2_voronoi_site_stats(
            structure,
            formal_valences,
            polyhedra=polyhedra,
        )
        _aggregate_sites(
            out,
            "p2vor",
            stats,
            formal_valences,
            P2_METRICS,
            P2_AGGREGATES,
        )
    except Exception as exc:
        failures[f"p2:{type(exc).__name__}"] += 1

    # P3 Hawthorne network.
    if neighbors is not None:
        try:
            out.update(
                p3_hawthorne_features(
                    structure,
                    formal_valences,
                    neighbors=neighbors,
                )
            )
        except Exception as exc:
            failures[f"p3:{type(exc).__name__}"] += 1

    # P5 strict Hoppe.
    if neighbors is not None:
        try:
            out.update(
                p5_hoppe_features(
                    structure,
                    formal_valences,
                    neighbors=neighbors,
                )
            )
        except Exception as exc:
            failures[f"p5:{type(exc).__name__}"] += 1

    return out, dict(failures)


# ---------------------------------------------------------------- workers


def _real_worker(record: Mapping[str, object]):
    from pymatgen.core import Structure

    from discriminate import guess_oxi, read_blob_cif

    failures: Counter[str] = Counter()
    try:
        structure = Structure.from_str(
            read_blob_cif(int(record["off"]), int(record["ln"])),
            fmt="cif",
        )
        valences, ok = guess_oxi(structure)
        if not ok:
            failures["valence:guess_oxi"] += 1
            return None, failures
        out, family_failures = next_local_features(structure, valences)
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
            read_blob_cif(int(record["off"]), int(record["ln"])),
            fmt="cif",
        )
        valences, ok = guess_oxi(structure)
        if not ok:
            failures["valence:guess_oxi"] += 1
            return rows, failures
        rng = np.random.default_rng(seed_of(str(record["sid"])))
        wanted = set(str(record["kinds"]).split(","))
        # Advance the generator through the canonical sequence even when a
        # baseline row is absent for one kind (same lineage contract as
        # phys_bad/elec_bad/geom_bad and the previous round).
        for kind in ("S1", "S2", "S3", "S4", "S5"):
            changed = perturb(structure, kind, rng, valences)
            if changed is None or kind not in wanted:
                continue
            try:
                out, family_failures = next_local_features(
                    changed,
                    swapped_val(changed, valences),
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

    from advanced_local_features import composition_guard_features
    from discriminate import read_blob_cif
    from polymorph_rank2 import balance

    failures: Counter[str] = Counter()
    try:
        structure = Structure.from_str(
            read_blob_cif(int(record["off"]), int(record["ln"])),
            fmt="cif",
        )
        if len(structure) > 80:
            failures["structure:too_many_sites"] += 1
            return None, failures
        valence_map = balance(structure.composition.reduced_formula.replace(" ", ""))
        if valence_map is None:
            failures["valence:balance"] += 1
            return None, failures
        valences = [float(valence_map[site.specie.symbol]) for site in structure]
        out, family_failures = next_local_features(structure, valences)
        failures.update(family_failures)
        out.update(composition_guard_features(structure, valences))
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


def _load_records(
    isolated_dir: Path,
    mode: str,
    *,
    limit: int,
    materials_database: Path | None,
    features_dir: Path | None,
) -> list[dict[str, object]]:
    if mode == "real":
        frame = pd.read_parquet(isolated_dir / "records_real.parquet")
        records = [
            {
                "sid": row.source_id,
                "off": int(row.blob_offset),
                "ln": int(row.blob_length),
                "split": str(row.split),
            }
            for row in frame.itertuples(index=False)
        ]
    elif mode == "bad":
        frame = pd.read_parquet(isolated_dir / "records_bad.parquet")
        records = [
            {
                "sid": row.parent,
                "off": int(row.blob_offset),
                "ln": int(row.blob_length),
                "split": str(row.split),
                "kinds": str(row.kinds),
            }
            for row in frame.itertuples(index=False)
        ]
    elif mode == "false-positive":
        if materials_database is None or features_dir is None:
            raise SystemExit(
                "false-positive mode requires --materials-database and --features-dir"
            )
        audit_ids = (
            pd.read_parquet(features_dir / "false_positive.parquet", columns=["sid"])[
                "sid"
            ]
            .dropna()
            .astype(str)
            .tolist()
        )
        resolved: dict[str, tuple[int, int]] = {}
        connection = sqlite3.connect(f"file:{materials_database}?mode=ro", uri=True)
        try:
            for start in range(0, len(audit_ids), 500):
                batch = audit_ids[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                query = (
                    "SELECT material_id, blob_offset, blob_length FROM materials "
                    f"WHERE material_id IN ({placeholders})"
                )
                for material_id, offset, length in connection.execute(query, batch):
                    resolved[str(material_id)] = (int(offset), int(length))
        finally:
            connection.close()
        records = [
            {
                "sid": sid,
                "off": resolved[sid][0],
                "ln": resolved[sid][1],
                "split": "false_positive_audit",
            }
            for sid in audit_ids
            if sid in resolved
        ]
    else:
        raise SystemExit(f"unknown mode: {mode}")
    if limit:
        records = records[:limit]
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute frozen P1/P2/P3/P5 descriptors on isolated splits."
    )
    parser.add_argument("mode", choices=("real", "bad", "false-positive"))
    parser.add_argument("--isolated-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunksize", type=int, default=8)
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
            if index % 1000 == 0:
                print(f"  {index:,}/{len(records):,} -> {len(rows):,}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for index, result in enumerate(
                executor.map(worker, records, chunksize=args.chunksize), start=1
            ):
                consume(result)
                if index % 1000 == 0:
                    print(f"  {index:,}/{len(records):,} -> {len(rows):,}", flush=True)
    if not rows:
        raise SystemExit("no descriptor rows were produced")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(args.out, index=False)

    id_column = {"real": "source_id", "bad": "sid", "false-positive": "sid"}[
        args.mode
    ]
    feature_columns = sorted(
        column
        for column in frame
        if column.startswith(("bvloc", "p2vor_", "p3haw_", "p5hop_"))
    )
    coverage = {
        column: float(np.isfinite(frame[column].to_numpy(dtype=float)).mean())
        for column in feature_columns
    }
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "np-next-20260801",
        "command": [str(value) for value in (argv if argv is not None else sys.argv[1:])],
        "mode": args.mode,
        "n_input_records": len(records),
        "n_output_rows": len(frame),
        "split_counts": {
            str(key): int(value)
            for key, value in frame["split"].value_counts().sort_index().items()
        },
        "id_column": id_column,
        "feature_columns": feature_columns,
        "feature_finite_coverage": coverage,
        "failure_counts": dict(sorted(failures.items())),
        "families": {
            "p1": "bond-valence triplet; searchable columns use frozen-fallback",
            "p2": "Voronoi solid-angle triplet (one tessellation per structure)",
            "p3": "Hawthorne network on CrystalNN topology; no bond lengths",
            "p5": (
                f"strict iterative Hoppe on {P5HOP_SPHERE_CUTOFF} A sphere; "
                f"damping {P5HOP_MEFIR_DAMPING}, tol {P5HOP_MEFIR_TOL}, "
                f"max iter {P5HOP_MEFIR_MAX_ITER}"
            ),
        },
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
        "input_sha256": (
            {
                name: _sha256(args.isolated_dir / name)
                for name in (("records_real.parquet",) if args.mode == "real" else ("records_bad.parquet",))
            }
            if args.mode != "false-positive"
            else {}
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
