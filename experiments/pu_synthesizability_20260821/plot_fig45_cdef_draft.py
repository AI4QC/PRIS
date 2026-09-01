#!/usr/bin/env python3
"""Additive c--f draft for the merged PRIS Fig. 4--5 narrative.

This script intentionally does not touch the canonical manuscript or existing
figures.  It answers the pre-DFT question with *independent* choices:

* panel c compares L4, S_syn and geometric distance cutoffs on one
  retention--screening plane; D7 is deliberately absent;
* panel d has two nested rows with a shared percentile x-axis.  The upper row
  is L4 violation versus a normalized A/B rank consensus and the lower row is
  S_syn on the same x-axis;
* panel e uses the frozen held-out curves for the synthesis and stability
  scores; panel f uses the frozen two-way ladder to connect the binary rules
  to hull, phonon and experimental-record axes.

The frozen full-pool checkout contains decile aggregates but not the row-level
 A/B common-ID table.  Therefore the default panel-d view is explicitly marked
``provisional_decile_aligned``.  Passing ``--consensus-deciles`` replaces it
with an exact, precomputed common-ID table without changing the plotting code.
The expected exact schema is documented in ``write_status`` below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = ROOT / "outputs/20260821_pu_synthesizability/analysis_v1"
DEFAULT_INDEPENDENT = ROOT / "outputs/20260822_pu_formula_scores/independent_choices_v1"
DEFAULT_FULL_POOL = ROOT / "outputs/20260821_pu_synthesizability/full_pool_analysis_v1"
DEFAULT_FORMULA_NPZ = ROOT / (
    "outputs/20260822_pu_formula_scores/full_pool_dual_v2/"
    "direct_formula_plots_v3/clscore_formula_density_data.npz"
)
DEFAULT_F3_JSON = ROOT / "outputs/20260814_f3_synth/resolve_f3.json"
DEFAULT_F2R_JSON = ROOT / "outputs/20260814_f2r_stability/resolve_f2r.json"
DEFAULT_TWOWAY = ROOT / "paper/data/fig5_twoway_ladder.csv"
DEFAULT_THREEAXIS = ROOT / "paper/data/fig6_threeaxis.csv"
DEFAULT_OUTPUT = ROOT / "outputs/20260822_pu_formula_scores/fig45_cdef_draft_v1"


COLORS = {
    "L4": "#007A5E",
    "S_syn": "#D88900",
    "distance": "#687889",
    "eHull": "#4C6A9A",
    "consensus": "#173F5F",
    "neutral": "#A5ACB7",
    "pending": "#7D8490",
    "ink": "#202630",
    "F3": "#0072B2",
    "F2R": "#D55E00",
    "hull": "#009E73",
    "phonon": "#E69F00",
    "made": "#0072B2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def style() -> None:
    # Nature-figure contract: editable SVG/PDF text and a compact journal-size
    # type scale.  Do not use CJK-only fonts so the generated vector remains
    # portable when parent figures are assembled elsewhere.
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8.6,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "axes.unicode_minus": False,
            "legend.frameon": False,
        }
    )


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        -0.11,
        1.04,
        text,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=COLORS["ink"],
    )


def require(frame: pd.DataFrame, cols: set[str], path: Path) -> None:
    missing = sorted(cols.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")


def load_c_data(binary_dir: Path, independent_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load independent operating choices for panel c.

    The returned frames use fractions, not percentages.  A distance cutoff is
    represented by its measured experimental-retention point; there is no
    interpolation between the two geometric cutoffs.
    """

    rung_path = binary_dir / "rung_summary.csv"
    distance_path = binary_dir / "distance_cutoff_summary.csv"
    independent_path = independent_dir / "independent_frontier.csv"
    rung = pd.read_csv(rung_path)
    distance = pd.read_csv(distance_path)
    independent = pd.read_csv(independent_path)
    require(
        rung,
        {"cohort", "unit", "rule_set", "total_weight", "explicit_violation_rate", "queue_retained"},
        rung_path,
    )
    require(
        distance,
        {"cohort", "cutoff_a", "total_weight", "explicit_violation_rate", "queue_retained"},
        distance_path,
    )
    require(
        independent,
        {"method", "experimental_retention", "pu_screened_rate", "pu_screened_n", "pu_total_n"},
        independent_path,
    )

    l4 = independent.loc[independent["method"].eq("L4")].copy()
    syn = independent.loc[independent["method"].eq("S_syn")].copy()
    if len(l4) != 1 or syn.empty:
        raise ValueError("independent frontier must contain one L4 row and S_syn rows")

    # Full-pool discrete PRIS ladder for the same retention/screening plane
    # used by the independent L4 and S_syn operating-point table.  L1--L4 are
    # separate gates, not a cascade, so retain them as independent points.
    rung_u = rung.loc[
        rung["unit"].eq("unique_cifs")
        & rung["cohort"].isin(["experimental", "pu_negative"])
        & rung["rule_set"].isin(["L1", "L2", "L3", "L4"])
    ].copy()
    ladder_rows = []
    for rule in ["L1", "L2", "L3", "L4"]:
        erow = rung_u.loc[(rung_u["cohort"] == "experimental") & (rung_u["rule_set"] == rule)]
        prow = rung_u.loc[(rung_u["cohort"] == "pu_negative") & (rung_u["rule_set"] == rule)]
        if len(erow) != 1 or len(prow) != 1:
            raise ValueError(f"unique-CIF rung summary is incomplete for {rule}")
        ladder_rows.append(
            {
                "rule": rule,
                "experimental_retention": float(erow.iloc[0]["queue_retained"]),
                "pu_screened_rate": float(prow.iloc[0]["explicit_violation_rate"]),
                "experimental_n": int(round(erow.iloc[0]["total_weight"])),
                "pu_n": int(round(prow.iloc[0]["total_weight"])),
            }
        )
    rule_ladder = pd.DataFrame(ladder_rows)
    # Use the independently frozen L4 point as the shared endpoint; this also
    # makes any harmless source-rounding difference explicit and reproducible.
    l4_idx = rule_ladder.index[rule_ladder["rule"].eq("L4")][0]
    rule_ladder.loc[l4_idx, "experimental_retention"] = float(l4.iloc[0]["experimental_retention"])
    rule_ladder.loc[l4_idx, "pu_screened_rate"] = float(l4.iloc[0]["pu_screened_rate"])

    d = distance.loc[
        distance["cohort"].eq("experimental")
        | distance["cohort"].eq("pu_negative")
    ].copy()
    exp_d = d.loc[d["cohort"].eq("experimental")].set_index("cutoff_a")
    pu_d = d.loc[d["cohort"].eq("pu_negative")].set_index("cutoff_a")
    rows = []
    for cutoff in sorted(set(exp_d.index).intersection(pu_d.index)):
        erow = exp_d.loc[cutoff]
        prow = pu_d.loc[cutoff]
        rows.append(
            {
                "method": f"distance {float(cutoff):.1f} Å",
                "retention": float(erow["queue_retained"]),
                "screening": float(prow["explicit_violation_rate"]),
                "n_experimental": int(round(erow["total_weight"])),
                "n_pu": int(round(prow["total_weight"])),
                "cutoff_a": float(cutoff),
            }
        )
    distance_points = pd.DataFrame(rows)
    metadata = {
        "l4_source": str(independent_path),
        "s_syn_source": str(independent_path),
        "distance_source": str(distance_path),
        "distance_points_are_discrete": True,
        "d7_included": False,
        "l4_experimental_retention": float(l4.iloc[0]["experimental_retention"]),
        "l4_pu_screening": float(l4.iloc[0]["pu_screened_rate"]),
        "s_syn_rows": int(len(syn)),
        "distance_rows": int(len(distance_points)),
        "rule_ladder_rows": int(len(rule_ladder)),
    }
    return {
        "l4": l4,
        "s_syn": syn.sort_values("experimental_retention"),
        "distance": distance_points,
        "rule_ladder": rule_ladder,
    }, metadata


def load_ehull(path: Path | None) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Load a corrected eHull threshold curve if explicitly supplied.

    Expected columns are ``threshold_ev_per_atom``, ``cohort``,
    ``keep_rate_of_total`` (or ``keep_rate_of_supported``), and
    ``reject_rate_of_total`` (or ``reject_rate_of_supported``).  The function
    never auto-selects a pilot: the caller must pass the path deliberately.
    """

    if path is None:
        return None, {"status": "pending", "reason": "no corrected eHull file supplied"}
    path = Path(path)
    if not path.exists():
        return None, {"status": "pending", "reason": f"missing file: {path}"}
    frame = pd.read_csv(path)
    require(frame, {"cohort", "threshold_ev_per_atom"}, path)
    # Treat an unparsable or non-converged row as an unflagged candidate, in
    # the same way as the PRIS three-state audit.  Therefore total-denominator
    # rates are preferred whenever the corrected runner provides them.  The
    # supported-denominator columns remain available in the raw CSV for the
    # convergence sensitivity analysis in SI.
    reject_col = "reject_rate_of_total" if "reject_rate_of_total" in frame else "reject_rate_of_supported"
    require(frame, {reject_col}, path)
    # The corrected runner records total-denominator rejection but not always
    # an explicit total-denominator keep column.  Derive it so ABSTAIN rows
    # remain in the queue rather than being silently dropped from x.
    if "keep_rate_of_total" in frame:
        keep_col = "keep_rate_of_total"
    elif "reject_rate_of_total" in frame:
        frame["_keep_rate_of_total_derived"] = 1.0 - pd.to_numeric(frame["reject_rate_of_total"], errors="coerce")
        keep_col = "_keep_rate_of_total_derived"
    else:
        keep_col = "keep_rate_of_supported"
    require(frame, {keep_col}, path)
    selected = frame.loc[frame["cohort"].eq("pu_negative")].copy()
    if selected.empty:
        raise ValueError(f"eHull curve has no pu_negative rows: {path}")
    # The x-axis requires experimental retention.  Merge the experimental rows
    # by threshold; an absent experimental row is a deliberate abstention.
    experimental = frame.loc[frame["cohort"].eq("experimental"), ["threshold_ev_per_atom", keep_col]].rename(
        columns={keep_col: "experimental_retention"}
    )
    selected = selected.merge(experimental, on="threshold_ev_per_atom", how="left")
    selected["retention"] = pd.to_numeric(selected["experimental_retention"], errors="coerce")
    selected["screening"] = pd.to_numeric(selected[reject_col], errors="coerce")
    selected = selected.dropna(subset=["retention", "screening"]).sort_values("retention")
    if selected.empty:
        return None, {"status": "pending", "reason": "no matched experimental retention rows"}
    threshold_numeric = pd.to_numeric(selected["threshold_ev_per_atom"], errors="coerce")
    operating_idx = int(np.nanargmin(np.abs(threshold_numeric.to_numpy(float) - 0.20)))
    operating_row = selected.iloc[operating_idx]
    return selected, {
        "status": "loaded",
        "source": str(path),
        "rows": int(len(selected)),
        "label": "MatterSim basin–hull proxy (corrected full-cell run)",
        "operating_threshold_ev_per_atom": float(operating_row["threshold_ev_per_atom"]),
        "operating_experimental_retention_total_denominator": float(operating_row["retention"]),
        "operating_pu_screening_total_denominator": float(operating_row["screening"]),
        "warning": "retain proxy wording until the energy reference is confirmed self-consistent; total-denominator rates keep ABSTAIN rows in the queue",
    }


def _provisional_consensus(
    full_pool_dir: Path, formula_npz_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a transparent aggregate fallback for panel d.

    A/B are normalized to their within-model decile ranks before averaging the
    two curves.  Since no row-level A/B pairing is available in the checkout,
    this is *not* advertised as an exact per-structure consensus; the status is
    persisted and the plot carries the same qualifier.
    """

    decile_path = full_pool_dir / "score_deciles.csv"
    deciles = pd.read_csv(decile_path)
    require(deciles, {"score_model", "decile", "n", "L4_explicit_violation_rate"}, decile_path)
    ab = deciles.loc[deciles["score_model"].isin(["CLscore_A", "CLscore_B"])].copy()
    piv = ab.pivot(index="decile", columns="score_model", values="L4_explicit_violation_rate")
    n_piv = ab.pivot(index="decile", columns="score_model", values="n")
    if not {"CLscore_A", "CLscore_B"}.issubset(piv.columns):
        raise ValueError("score_deciles lacks both CLscore_A and CLscore_B")
    data = np.load(formula_npz_path, allow_pickle=False)
    a = np.asarray(data["S_syn__CLscore_A__y_mean"], dtype=float)
    b = np.asarray(data["S_syn__CLscore_B__y_mean"], dtype=float)
    if len(a) != 10 or len(b) != 10:
        raise ValueError("formula NPZ does not contain ten S_syn deciles")
    q25a = np.asarray(data["S_syn__CLscore_A__y_q25"], dtype=float)
    q25b = np.asarray(data["S_syn__CLscore_B__y_q25"], dtype=float)
    q75a = np.asarray(data["S_syn__CLscore_A__y_q75"], dtype=float)
    q75b = np.asarray(data["S_syn__CLscore_B__y_q75"], dtype=float)
    k = np.arange(1, 11, dtype=int)
    l4 = piv.reindex(k).mean(axis=1).to_numpy(dtype=float)
    n = n_piv.reindex(k).min(axis=1).to_numpy(dtype=int)
    result = pd.DataFrame(
        {
            "decile": k,
            "consensus_percentile": (k - 0.5) / 10.0,
            "n": n,
            "L4_explicit_violation_rate": l4,
            "S_syn_mean": (a + b) / 2.0,
            "S_syn_q25": np.minimum(q25a, q25b),
            "S_syn_q75": np.maximum(q75a, q75b),
        }
    )
    return result, {
        "status": "provisional_decile_aligned",
        "source_score_deciles": str(decile_path),
        "source_formula_npz": str(formula_npz_path),
        "normalization": "within-model percentile rank, represented by aligned ten-bin centers",
        "warning": "row-level A/B common-ID pairing is unavailable; replace with exact consensus_deciles.csv before publication",
    }


def load_consensus(
    consensus_path: Path | None,
    full_pool_dir: Path,
    formula_npz_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if consensus_path is None:
        return _provisional_consensus(full_pool_dir, formula_npz_path)
    path = Path(consensus_path)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    require(
        frame,
        {"decile", "consensus_percentile", "n", "L4_explicit_violation_rate", "S_syn_mean"},
        path,
    )
    frame = frame.sort_values("consensus_percentile").reset_index(drop=True)
    x = pd.to_numeric(frame["consensus_percentile"], errors="coerce").to_numpy(float)
    if not np.isfinite(x).all() or np.any(np.diff(x) <= 0) or (x <= 0).any() or (x >= 1).any():
        raise ValueError("consensus_percentile must be finite and strictly increasing in (0,1)")
    for column in ["L4_explicit_violation_rate", "S_syn_mean"]:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"consensus table contains missing {column}")
    return frame, {
        "status": "exact_common_id",
        "source": str(path),
        "normalization": "R_A=rank(A)/(n+1), R_B=rank(B)/(n+1), consensus=(R_A+R_B)/2",
    }


def plot_panel_c(ax: plt.Axes, data: dict[str, pd.DataFrame], ehull: pd.DataFrame | None) -> None:
    syn = data["s_syn"]
    l4 = data["l4"].iloc[0]
    ax.plot(
        syn["experimental_retention"] * 100,
        syn["pu_screened_rate"] * 100,
        color=COLORS["S_syn"],
        marker="o",
        markersize=3.2,
        linewidth=1.7,
        label=r"$S_{\mathrm{syn}}$ (independent threshold)",
    )
    ax.scatter(
        [float(l4["experimental_retention"]) * 100],
        [float(l4["pu_screened_rate"]) * 100],
        color=COLORS["L4"],
        marker="*",
        s=90,
        zorder=5,
        label="L4 (independent rule gate)",
    )
    # Put the two user-facing operating choices directly on the frontier.
    # These are independent calibrations, not a cascade or a combined gate.
    ax.annotate(
        f"L4\n{float(l4['experimental_retention']) * 100:.1f}% / {float(l4['pu_screened_rate']) * 100:.1f}%",
        (float(l4["experimental_retention"]) * 100, float(l4["pu_screened_rate"]) * 100),
        xytext=(9, -5),
        textcoords="offset points",
        fontsize=6.2,
        color=COLORS["L4"],
        ha="left",
        va="top",
    )
    # The S_syn curve contains many calibrated thresholds.  Label only the
    # matched-retention point and the conservative 95% point to keep the
    # panel legible while retaining the concrete values.
    syn_ret = pd.to_numeric(syn["experimental_retention"], errors="coerce").to_numpy(float)
    syn_screen = pd.to_numeric(syn["pu_screened_rate"], errors="coerce").to_numpy(float)
    if len(syn_ret):
        targets = [(float(l4["experimental_retention"]), "matched", (8, 8)), (0.95, "95%", (-38, 8))]
        used: set[int] = set()
        for target, tag, offset in targets:
            idx = int(np.nanargmin(np.abs(syn_ret - target)))
            if idx in used:
                continue
            used.add(idx)
            ax.annotate(
                f"$S_{{\\rm syn}}$ {tag}\n{syn_ret[idx] * 100:.1f}% / {syn_screen[idx] * 100:.1f}%",
                (syn_ret[idx] * 100, syn_screen[idx] * 100),
                xytext=offset,
                textcoords="offset points",
                fontsize=6.0,
                color=COLORS["S_syn"],
                ha="left",
                va="bottom" if offset[1] >= 0 else "top",
            )
    points = data["distance"]
    if not points.empty:
        ax.plot(
            points["retention"] * 100,
            points["screening"] * 100,
            color=COLORS["distance"],
            marker="s",
            linestyle="none",
            markersize=4.2,
            label="Distance cutoffs (discrete points)",
        )
        for row in points.itertuples(index=False):
            # The two geometric points are nearly coincident at zero screened
            # PU rows; separate their labels so neither hides the other.
            offset = (-25, 9) if abs(float(row.cutoff_a) - 0.5) < 1e-6 else (9, 9)
            ax.annotate(
                f"{row.cutoff_a:g} Å",
                (100 * row.retention, 100 * row.screening),
                xytext=offset,
                textcoords="offset points",
                fontsize=6.5,
                color=COLORS["distance"],
            )
    if ehull is not None and not ehull.empty:
        ax.plot(
            ehull["retention"] * 100,
            ehull["screening"] * 100,
            color=COLORS["eHull"],
            linestyle="--",
            linewidth=1.35,
            marker="^",
            markersize=3.1,
            label="MatterSim basin–hull proxy",
        )
        # Highlight the preregistered B64 threshold used in the audit.  The
        # line remains a threshold response curve; this marker is not a new
        # fitted operating point.
        threshold = pd.to_numeric(ehull["threshold_ev_per_atom"], errors="coerce")
        if threshold.notna().any():
            idx = int(np.nanargmin(np.abs(threshold.to_numpy(float) - 0.20)))
            row = ehull.iloc[idx]
            ax.scatter([float(row["retention"]) * 100], [float(row["screening"]) * 100],
                       color=COLORS["eHull"], marker="^", s=36, edgecolor="white", linewidth=0.45,
                       zorder=6)
            ax.annotate(
                f"basin–hull proxy\n{float(row['retention']) * 100:.1f}% / {float(row['screening']) * 100:.1f}%",
                (float(row["retention"]) * 100, float(row["screening"]) * 100),
                xytext=(7, -27),
                textcoords="offset points",
                fontsize=6.0,
                color=COLORS["eHull"],
                ha="left",
                va="top",
            )
    ax.set_xlim(55, 100.5)
    ax.set_ylim(-2, 101)
    ax.xaxis.set_major_formatter(PercentFormatter(100))
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_xlabel("experimental structures retained (%)")
    ax.set_ylabel("PU hard-negative structures screened (%)")
    ax.grid(axis="both", alpha=0.18, linewidth=0.6)
    # Keep the formula legend parent glyph >=7.5 pt so MathText subscripts
    # remain above the 5 pt rendered-glyph floor.
    ax.legend(loc="lower left", ncol=1, fontsize=7.5)
    ax.text(
        0.02,
        0.98,
        "independent routes; no cascade",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        color="#596273",
    )
    panel_label(ax, "c")


def plot_panel_d(outer: plt.Axes, consensus: pd.DataFrame, status: dict[str, Any]) -> None:
    """Draw two rows with exactly one shared x-axis."""

    outer.set_axis_off()
    grid = outer.get_subplotspec().subgridspec(2, 1, hspace=0.05, height_ratios=(1, 1))
    upper = outer.figure.add_subplot(grid[0])
    lower = outer.figure.add_subplot(grid[1], sharex=upper)
    x = consensus["consensus_percentile"].to_numpy(float) * 100
    l4 = consensus["L4_explicit_violation_rate"].to_numpy(float) * 100
    syn = consensus["S_syn_mean"].to_numpy(float)
    upper.plot(x, l4, color=COLORS["consensus"], marker="o", markersize=3.0, linewidth=1.5)
    upper.set_ylabel("L4 violation (%)", labelpad=2)
    upper.set_ylim(20, 65)
    upper.grid(axis="y", alpha=0.18, linewidth=0.6)
    upper.tick_params(labelbottom=False)
    lower.plot(x, syn, color=COLORS["S_syn"], marker="o", markersize=3.0, linewidth=1.5)
    if {"S_syn_q25", "S_syn_q75"}.issubset(consensus.columns):
        q25 = consensus["S_syn_q25"].to_numpy(float)
        q75 = consensus["S_syn_q75"].to_numpy(float)
        lower.fill_between(x, q25, q75, color=COLORS["S_syn"], alpha=0.14, linewidth=0)
    lower.axhline(0, color="#8A9099", linestyle=":", linewidth=0.7)
    lower.set_ylabel(r"$S_{\mathrm{syn}}$ mean", labelpad=2)
    lower.set_xlabel("normalized consensus CLscore percentile")
    lower.set_xlim(0, 100)
    lower.set_xticks(np.arange(0, 101, 20))
    lower.grid(axis="y", alpha=0.18, linewidth=0.6)
    upper.text(
        0.02,
        0.92,
        "consensus = mean of within-model percentiles\nCGCNN-PU OOB + MatterSim-embedding PU OOB",
        transform=upper.transAxes,
        fontsize=6.2,
        color="#596273",
        va="top",
    )
    qualifier = "exact common-ID ranks" if status.get("status") == "exact_common_id" else "provisional aligned deciles; row-level common IDs unavailable"
    lower.text(
        0.02,
        0.04,
        qualifier,
        transform=lower.transAxes,
        fontsize=6.2,
        color="#596273",
        va="bottom",
    )
    panel_label(upper, "d")


def _curve_bounds(
    record: dict[str, Any],
    estimate_key: str = "acc",
    ci_key: str = "ci",
) -> tuple[float, float, float]:
    """Return (estimate, lower, upper) from a frozen curve record."""
    estimate = float(record[estimate_key])
    ci = record.get(ci_key)
    if ci is None or len(ci) != 2:
        return estimate, estimate, estimate
    return estimate, float(ci[0]), float(ci[1])


def load_heldout_curves(f3_path: Path, f2r_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the frozen synthesis (F3/S_syn) and stability (F2R/S_stab) curves."""
    f3_path = Path(f3_path)
    f2r_path = Path(f2r_path)
    if not f3_path.exists():
        raise FileNotFoundError(f3_path)
    if not f2r_path.exists():
        raise FileNotFoundError(f2r_path)
    f3 = json.loads(f3_path.read_text(encoding="utf-8"))
    f2r = json.loads(f2r_path.read_text(encoding="utf-8"))
    if not isinstance(f3.get("F3_curve"), dict) or not isinstance(f2r.get("F2R_curve"), dict):
        raise ValueError("held-out curve files do not contain F3_curve/F2R_curve")
    return f3, f2r


def _ladder_index(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the four indexes used by the audited Fig. 5f ladder."""
    require(table, {"rung", "axis", "direction", "class", "p", "lo", "hi", "n"}, DEFAULT_TWOWAY)
    fw = table.loc[table.direction.eq("forward")].set_index(["axis", "rung", "class"])
    rv = table.loc[table.direction.eq("reverse")].set_index(["axis", "rung", "class"])
    marginal = table.loc[table.direction.eq("marginal")].copy()
    acc = marginal.loc[
        (marginal.axis.eq("-")) & marginal["class"].eq("rung accepts | determinate")
    ]
    base = marginal.loc[marginal["class"].eq("class true | determinate")].set_index(["axis", "rung"])
    ndet = marginal.loc[
        (marginal.axis.eq("-")) & marginal["class"].eq("determinate | all")
    ].set_index("rung")
    return fw, rv, acc, base, ndet


LADDER_RUNGS = ["L1", "L1'", "L2", "L3", "L4"]
LADDER_AXES = [
    ("on_hull", "on hull", COLORS["hull"]),
    ("dyn_stable", "no imaginary modes", COLORS["phonon"]),
    ("made", "experimentally recorded", COLORS["made"]),
]


def draw_panel_e(
    ax: plt.Axes,
    f3_path: Path = DEFAULT_F3_JSON,
    f2r_path: Path = DEFAULT_F2R_JSON,
    callback: Callable[[plt.Axes], None] | None = None,
) -> dict[str, Any]:
    """Draw the two frozen held-out score curves in a compact nested panel.

    The upper row is the synthesis score (F3, shown as ``S_syn``) and its
    DFT-hull comparator.  The lower row is the stability score (F2R, shown as
    ``S_stab``) as a function of the minimum energy-gap floor.  The two rows
    answer different questions and are not combined into a gate.
    """
    if callback is not None:
        callback(ax)
        panel_label(ax, "e")
        return {"status": "callback"}
    f3, f2r = load_heldout_curves(f3_path, f2r_path)
    ax.set_axis_off()
    grid = ax.get_subplotspec().subgridspec(2, 1, hspace=0.43, height_ratios=(1, 1))
    top = ax.figure.add_subplot(grid[0])
    bottom = ax.figure.add_subplot(grid[1])

    # F3/S_syn and the hull reference use the same confidence ordering.
    f3_keys = sorted((float(k) for k in f3["F3_curve"]), reverse=False)
    x3 = np.asarray(f3_keys, dtype=float)
    f3_records = [f3["F3_curve"][f"{x:.2f}"] for x in f3_keys]
    y3, lo3, hi3 = zip(*(_curve_bounds(r) for r in f3_records))
    top.plot(x3 * 100, np.asarray(y3) * 100, "-o", color=COLORS["F3"], lw=1.35, ms=3.2,
             label=r"$S_{\mathrm{syn}}$ (held-out)")
    top.fill_between(x3 * 100, np.asarray(lo3) * 100, np.asarray(hi3) * 100,
                     color=COLORS["F3"], alpha=0.14, lw=0)
    if isinstance(f3.get("e_hull_curve"), dict):
        hull_records = [f3["e_hull_curve"].get(f"{x:.2f}") for x in f3_keys]
        if all(r is not None for r in hull_records):
            yh, loh, hih = zip(*(_curve_bounds(r) for r in hull_records))
            top.plot(x3 * 100, np.asarray(yh) * 100, "--^", color=COLORS["hull"], lw=1.05, ms=2.8,
                     label="DFT hull energy")
            top.fill_between(x3 * 100, np.asarray(loh) * 100, np.asarray(hih) * 100,
                             color=COLORS["hull"], alpha=0.08, lw=0)
    top.set_ylabel("pair accuracy (%)", labelpad=1)
    top.set_xlabel("fraction of held-out pairs retained", labelpad=1)
    top.set_ylim(68, 101)
    top.set_xlim(0, 103)
    top.xaxis.set_major_formatter(PercentFormatter(100))
    top.grid(axis="y", alpha=0.18, lw=0.6)
    top.set_xticks([5, 10, 20, 30, 50, 100])
    top.set_xticklabels(["5", "10", "20", "30", "50", "100"], fontsize=6.2)
    top.legend(loc="lower right", fontsize=7.5, handlelength=1.5, labelspacing=0.2)
    top.text(0.02, 0.96, "synthesis ranking", transform=top.transAxes, va="top",
             fontsize=6.5, color=COLORS["F3"])
    top.text(0.98, 0.96, f"n={int(f3.get('n_pairs', 0)):,} pairs", transform=top.transAxes,
             va="top", ha="right", fontsize=6.0, color="#596273")

    # F2R/S_stab curve: x is an interpretable energy-gap floor, not a second
    # score threshold.  This is the same frozen grouping used in the original
    # ranking panel and preserves its Wilson intervals.
    gap = f2r.get("gap_bins", {})
    if not isinstance(gap, dict) or not gap:
        raise ValueError("F2R file has no gap_bins")
    x2 = np.asarray(sorted(float(k) for k in gap), dtype=float)
    records2 = [gap[f"{x:.3f}"] for x in x2]
    y2, lo2, hi2 = zip(*(_curve_bounds(r, "group_equal_acc", "ci_group") for r in records2))
    bottom.plot(x2, np.asarray(y2) * 100, "-o", color=COLORS["F2R"], lw=1.35, ms=3.2,
                label=r"$S_{\mathrm{stab}}$ (held-out)")
    bottom.fill_between(x2, np.asarray(lo2) * 100, np.asarray(hi2) * 100,
                        color=COLORS["F2R"], alpha=0.14, lw=0)
    bottom.axhline(50, color="#8A9099", ls=":", lw=0.7)
    bottom.text(0.02, 0.06, "chance", transform=bottom.transAxes, fontsize=6.1, color="#596273")
    bottom.set_xlabel("minimum same-composition energy gap (eV atom$^{-1}$)", labelpad=1)
    bottom.set_ylabel("pair accuracy (%)", labelpad=1)
    bottom.set_ylim(45, 82)
    bottom.set_xlim(-0.004, max(x2) + 0.006)
    bottom.set_xticks(x2)
    bottom.set_xticklabels([f"{x:.3f}" for x in x2], fontsize=6.2)
    bottom.grid(axis="y", alpha=0.18, lw=0.6)
    bottom.legend(loc="lower right", fontsize=7.5, handlelength=1.5, labelspacing=0.2)
    bottom.text(0.02, 0.95, "stability ranking", transform=bottom.transAxes, va="top",
                fontsize=6.5, color=COLORS["F2R"])
    bottom.text(0.98, 0.95, f"n={int(f2r.get('n_pairs', 0)):,} pairs", transform=bottom.transAxes,
                va="top", ha="right", fontsize=6.0, color="#596273")
    panel_label(top, "e")
    return {
        "status": "loaded",
        "f3_source": str(f3_path),
        "f2r_source": str(f2r_path),
        "f3_pairs": int(f3.get("n_pairs", 0)),
        "f2r_pairs": int(f2r.get("n_pairs", 0)),
    }


def draw_panel_f(
    ax: plt.Axes,
    ladder_path: Path = DEFAULT_TWOWAY,
    callback: Callable[[plt.Axes], None] | None = None,
) -> dict[str, Any]:
    """Port the audited two-way ladder (old Fig. 5f) into the additive draft."""
    if callback is not None:
        callback(ax)
        panel_label(ax, "f")
        return {"status": "callback"}
    table = pd.read_csv(ladder_path)
    fw, rv, acc, base, ndet = _ladder_index(table)
    ax.set_axis_off()
    grid = ax.get_subplotspec().subgridspec(2, 1, hspace=0.46, height_ratios=(1, 1))
    upper = ax.figure.add_subplot(grid[0])
    lower = ax.figure.add_subplot(grid[1], sharex=upper)
    x = np.arange(len(LADDER_RUNGS))

    def curve(target: plt.Axes, index: pd.DataFrame, axis_name: str, cls: str,
              color: str, linestyle: str) -> None:
        rows = [index.loc[(axis_name, rung, cls)] for rung in LADDER_RUNGS]
        p = np.asarray([r.p for r in rows], dtype=float)
        lo = np.asarray([r.lo for r in rows], dtype=float)
        hi = np.asarray([r.hi for r in rows], dtype=float)
        target.fill_between(x, lo * 100, hi * 100, color=color, alpha=0.10, lw=0, zorder=1)
        target.plot(x, p * 100, linestyle=linestyle, color=color, lw=1.0, marker="o", ms=2.5,
                    mfc=color if linestyle == "-" else "white", mec=color, mew=0.7, zorder=2)

    for axis_name, _label, color in LADDER_AXES:
        curve(upper, fw, axis_name, "class true", color, "-")
        curve(upper, fw, axis_name, "class false", color, "--")
        curve(lower, rv, axis_name, "rung accepts", color, "-")
        curve(lower, rv, axis_name, "rung rejects", color, "--")
        # The reverse base rate is useful context, but is deliberately faint.
        lower.plot(x, [float(base.loc[(axis_name, rung), "p"]) * 100 for rung in LADDER_RUNGS],
                   color=color, lw=0.65, alpha=0.35, zorder=0)
    marginal = acc.set_index("rung").reindex(LADDER_RUNGS)
    upper.plot(x, marginal["p"].to_numpy(float) * 100, color="#777777", lw=0.85,
               ls=":", zorder=0)

    upper.set_ylabel("satisfaction (%)", labelpad=1)
    upper.set_ylim(60, 102)
    upper.set_xlim(-0.3, 4.3)
    upper.set_xticks(x)
    upper.tick_params(labelbottom=False)
    upper.grid(axis="y", alpha=0.18, lw=0.55)
    lower.set_ylabel("class enrichment (%)", labelpad=1)
    lower.set_xlabel("PRIS rule set", labelpad=1)
    lower.set_ylim(25, 72)
    lower.set_xticks(x)
    lower.set_xticklabels(["L1", "L1′", "L2", "L3", "L4"], fontsize=6.6)
    lower.grid(axis="y", alpha=0.18, lw=0.55)
    lower.text(0.98, 0.96, "n=26,600 computed structures", transform=lower.transAxes,
               ha="right", va="top", fontsize=6.0, color="#596273")
    handles = [Line2D([], [], color=color, lw=1.1) for _, label, color in LADDER_AXES]
    handles.append(Line2D([], [], color="#777777", lw=0.9, ls=":"))
    upper.legend(handles, [label for _, label, _ in LADDER_AXES] + ["marginal"],
                 loc="lower left", fontsize=7.0, ncol=2, handlelength=1.35,
                 columnspacing=0.7, labelspacing=0.15)
    style_handles = [
        Line2D([], [], color="#555555", lw=1.0, ls="-", marker="o", ms=2.5),
        Line2D([], [], color="#555555", lw=1.0, ls="--", marker="o", ms=2.5, mfc="white"),
        Line2D([], [], color="#888888", lw=0.7, ls="-", alpha=0.5),
    ]
    lower.legend(style_handles, ["class true / accepts", "class false / rejects", "base rate"],
                 loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=7.0,
                 ncol=2, handlelength=1.5, columnspacing=0.8, labelspacing=0.15)
    panel_label(upper, "f")
    return {"status": "loaded", "source": str(ladder_path), "n_rows": int(len(table)), "n_structures": 26600}


def draw_panel_f_fourcell(
    ax: plt.Axes,
    threeaxis_path: Path = DEFAULT_THREEAXIS,
    callback: Callable[[plt.Axes], None] | None = None,
) -> dict[str, Any]:
    """Draw the four-way stability/record explanation panel.

    Each cell fixes a thermodynamic/dynamical state (on hull versus
    metastable, no imaginary modes versus imaginary modes) and compares
    experimentally recorded with computed-only structures.  D1 and D7 are
    shown as separate bars; neither is used as a combined gate.
    """
    if callback is not None:
        callback(ax)
        panel_label(ax, "f")
        return {"status": "callback"}
    threeaxis_path = Path(threeaxis_path)
    table = pd.read_csv(threeaxis_path)
    require(table, {"dyn_stable", "on_hull", "experimental", "n", "pass_d1_735", "pass_d7"}, threeaxis_path)
    if len(table) != 8:
        raise ValueError(f"four-cell table should have 8 state rows, found {len(table)}")
    table = table.copy()
    for col in ["dyn_stable", "on_hull", "experimental"]:
        if table[col].dtype != bool:
            table[col] = table[col].astype(str).str.lower().map({"true": True, "false": False})
    if table[["dyn_stable", "on_hull", "experimental"]].isna().any().any():
        raise ValueError(f"invalid boolean state in {threeaxis_path}")
    ax.set_axis_off()
    grid = ax.get_subplotspec().subgridspec(2, 2, wspace=0.38, hspace=0.58)
    states = [
        (True, True, "no imaginary modes\n/on hull"),
        (True, False, "no imaginary modes\n/metastable"),
        (False, True, "imaginary modes\n/on hull"),
        (False, False, "imaginary modes\n/metastable"),
    ]
    bar_colors = [COLORS["L4"], COLORS["F2R"]]
    handles = None
    for i, (dyn, hull, title) in enumerate(states):
        sub = ax.figure.add_subplot(grid[i // 2, i % 2])
        subset = table.loc[(table.dyn_stable.eq(dyn)) & (table.on_hull.eq(hull))]
        subset = subset.set_index("experimental").reindex([True, False])
        if subset["n"].isna().any():
            raise ValueError(f"missing experimental/computed-only cell for state {dyn},{hull}")
        x = np.arange(2)
        width = 0.32
        d1 = subset["pass_d1_735"].to_numpy(float) * 100
        d7 = subset["pass_d7"].to_numpy(float) * 100
        bars1 = sub.bar(x - width / 2, d1, width, color=bar_colors[0], label="D1")
        bars2 = sub.bar(x + width / 2, d7, width, color=bar_colors[1], label="D7")
        for bars, vals in [(bars1, d1), (bars2, d7)]:
            for bar, value in zip(bars, vals):
                sub.text(bar.get_x() + bar.get_width() / 2, min(value + 2.0, 103),
                         f"{value:.0f}", ha="center", va="bottom", fontsize=6.8,
                         color=bar.get_facecolor())
        n_exp = int(subset.loc[True, "n"])
        n_comp = int(subset.loc[False, "n"])
        sub.set_title(title, fontsize=7.5, loc="left", pad=2.0)
        sub.set_ylim(0, 108)
        sub.set_xlim(-0.62, 1.62)
        sub.set_xticks(x)
        sub.set_xticklabels(
            [f"recorded\nn={n_exp:,}", f"computed\nn={n_comp:,}"],
            fontsize=6.8,
            rotation=0,
        )
        sub.set_yticks([0, 50, 100])
        sub.tick_params(axis="y", labelsize=6.8, pad=1)
        sub.grid(axis="y", alpha=0.18, lw=0.5)
        if i % 2 == 1:
            sub.tick_params(labelleft=False)
        if i == 0:
            handles = [bars1, bars2]
            panel_label(sub, "f")
    # A shared key is placed below the cells and above the figure footer;
    # keeping it outside the axes leaves the n labels and bars unobstructed.
    if handles is not None:
        ax.figure.legend(handles, ["D1 (rule 1 threshold)", "D7 (site complexity)"],
                         loc="lower right", bbox_to_anchor=(0.965, 0.022), fontsize=6.8,
                         frameon=False, ncol=2, handlelength=1.0, columnspacing=0.8)
    return {
        "status": "loaded_four_cell",
        "source": str(threeaxis_path),
        "n_rows": int(len(table)),
        "n_structures": int(table["n"].sum()),
        "metrics": ["pass_d1_735", "pass_d7"],
    }


def write_status(
    output_dir: Path,
    *,
    sources: list[Path],
    c_metadata: dict[str, Any],
    ehull_metadata: dict[str, Any],
    consensus_metadata: dict[str, Any],
    panel_e_metadata: dict[str, Any],
    panel_f_metadata: dict[str, Any],
) -> Path:
    status = {
        "schema_version": 1,
        "figure_scope": "additive Fig.4-5 c-f draft; canonical paper untouched",
        "panel_c": {
            "methods": ["L4", "S_syn", "distance cutoffs"],
            "D7": "omitted by design",
            "Jang": "absent by design; this panel compares independent pre-DFT choices",
            "independent_choices": True,
            "data": c_metadata,
        },
        "panel_d": {
            "x_definition_exact": "R_CGCNN=rank(CGCNN-PU OOB)/(n+1), R_MatterSim=rank(MatterSim-embedding PU OOB)/(n+1), consensus=(R_CGCNN+R_MatterSim)/2",
            "data": consensus_metadata,
        },
        "panel_e": panel_e_metadata,
        "panel_f": panel_f_metadata,
        "sources": {str(path): sha256(path) for path in sources if path.exists()},
        "expected_exact_consensus_csv": {
            "columns": ["decile", "consensus_percentile", "n", "L4_explicit_violation_rate", "S_syn_mean", "S_syn_q25", "S_syn_q75"],
            "note": "all rates are fractions; consensus_percentile is in (0,1)",
        },
        "ehull": ehull_metadata,
        "eHull_labeling": "proxy only unless the energy reference is confirmed self-consistent",
        "CLscore_normalization": "panel d uses row-level common-ID rank consensus when supplied; otherwise provisional aligned deciles",
    }
    path = output_dir / "STATUS.json"
    path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def render(
    *,
    binary_dir: Path,
    independent_dir: Path,
    full_pool_dir: Path,
    formula_npz: Path,
    output_dir: Path,
    ehull_path: Path | None = None,
    consensus_path: Path | None = None,
    f3_path: Path = DEFAULT_F3_JSON,
    f2r_path: Path = DEFAULT_F2R_JSON,
    ladder_path: Path = DEFAULT_TWOWAY,
    threeaxis_path: Path = DEFAULT_THREEAXIS,
    f_layout: str = "ladder",
    stem_name: str = "pris_fig45_cdef_draft",
) -> dict[str, Any]:
    style()
    output_dir.mkdir(parents=True, exist_ok=True)
    c_data, c_metadata = load_c_data(binary_dir, independent_dir)
    ehull, ehull_metadata = load_ehull(ehull_path)
    consensus, consensus_metadata = load_consensus(consensus_path, full_pool_dir, formula_npz)

    # Manual margins are intentional here.  The nested two-row panels d/e/f
    # make constrained_layout collapse the left
    # column in some Matplotlib versions; fixed margins keep all four slots
    # equal and reproducible across renderers.
    fig = plt.figure(figsize=(7.25, 7.8), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.28, 1.0), height_ratios=(1.0, 1.0), wspace=0.34, hspace=0.34,
                            left=0.10, right=0.97, bottom=0.09, top=0.90)
    ax_c = fig.add_subplot(grid[0, 0])
    ax_d = fig.add_subplot(grid[0, 1])
    ax_e = fig.add_subplot(grid[1, 0])
    ax_f = fig.add_subplot(grid[1, 1])
    plot_panel_c(ax_c, c_data, ehull)
    plot_panel_d(ax_d, consensus, consensus_metadata)
    panel_e_metadata = draw_panel_e(ax_e, f3_path=f3_path, f2r_path=f2r_path)
    if f_layout == "fourcell":
        panel_f_metadata = draw_panel_f_fourcell(ax_f, threeaxis_path=threeaxis_path)
    elif f_layout == "ladder":
        panel_f_metadata = draw_panel_f(ax_f, ladder_path=ladder_path)
    else:
        raise ValueError(f"unknown f_layout: {f_layout}")
    fig.suptitle("PRIS pre-DFT choices and the next validation links", fontsize=11, fontweight="bold", color=COLORS["ink"], y=0.965)
    fig.text(0.5, 0.018, "Draft only. L4 and S_syn are independent options; panel d is provisional until row-level common-ID ranks are supplied.", ha="center", va="bottom", fontsize=6.4, color="#596273")

    stem = output_dir / stem_name
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", metadata={"Title": "PRIS Fig.4-5 c-f additive draft"})
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    consensus_out = output_dir / "consensus_deciles_used.csv"
    consensus.to_csv(consensus_out, index=False)
    status_path = write_status(
        output_dir,
        sources=[
            binary_dir / "rung_summary.csv",
            binary_dir / "distance_cutoff_summary.csv",
            independent_dir / "independent_frontier.csv",
            full_pool_dir / "score_deciles.csv",
            formula_npz,
        ] + ([ehull_path] if ehull_path is not None else []) + ([consensus_path] if consensus_path is not None else []) + [f3_path, f2r_path, ladder_path] + ([threeaxis_path] if f_layout == "fourcell" else []),
        c_metadata=c_metadata,
        ehull_metadata=ehull_metadata,
        consensus_metadata=consensus_metadata,
        panel_e_metadata=panel_e_metadata,
        panel_f_metadata=panel_f_metadata,
    )
    report = output_dir / "README.md"
    report.write_text(
        """# Fig. 4–5 c–f additive draft

Panel c compares L4, S_syn and the two distance cutoffs as separate
retention–screening choices. D7 is intentionally absent. Panel d uses the
same normalized-consensus percentile x-axis for L4 violation and S_syn. The
consensus is the mean of within-model percentile ranks from CGCNN-PU OOB and
MatterSim-embedding PU OOB. The current checkout provides aligned decile
aggregates, so the plot is marked provisional until a row-level common-ID table is passed with
`--consensus-deciles`. Panel e shows the frozen held-out synthesis and
stability ranking curves. Panel f uses the frozen two-way ladder or its
four-cell thermodynamic/dynamical explanation view, selected by `--f-layout`.

No canonical TeX, paper figure, or frozen result was modified.
""",
        encoding="utf-8",
    )
    outputs = [stem.with_suffix(ext) for ext in (".svg", ".pdf", ".png", ".tiff")] + [consensus_out, status_path, report]
    sums = output_dir / "SHA256SUMS"
    sums.write_text("\n".join(f"{sha256(path)}  {path.name}" for path in sorted(outputs)) + "\n", encoding="utf-8")
    return {
        "figure": str(stem.with_suffix(".svg")),
        "status": str(status_path),
        "consensus_status": consensus_metadata["status"],
        "ehull_status": ehull_metadata["status"],
        "outputs": [str(path) for path in outputs],
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary-dir", type=Path, default=DEFAULT_BINARY)
    p.add_argument("--independent-dir", type=Path, default=DEFAULT_INDEPENDENT)
    p.add_argument("--full-pool-dir", type=Path, default=DEFAULT_FULL_POOL)
    p.add_argument("--formula-npz", type=Path, default=DEFAULT_FORMULA_NPZ)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--ehull", type=Path, default=None, help="explicit corrected eHull threshold CSV")
    p.add_argument("--consensus-deciles", type=Path, default=None, help="exact common-ID consensus table")
    p.add_argument("--f3-json", type=Path, default=DEFAULT_F3_JSON, help="frozen synthesis held-out curve")
    p.add_argument("--f2r-json", type=Path, default=DEFAULT_F2R_JSON, help="frozen stability held-out curve")
    p.add_argument("--ladder-csv", type=Path, default=DEFAULT_TWOWAY, help="frozen two-way ladder table")
    p.add_argument("--threeaxis-csv", type=Path, default=DEFAULT_THREEAXIS, help="four-cell stability/record table")
    p.add_argument("--f-layout", choices=("ladder", "fourcell"), default="ladder", help="panel-f layout")
    p.add_argument("--stem", dest="stem_name", default="pris_fig45_cdef_draft", help="output file stem")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = render(
        binary_dir=args.binary_dir,
        independent_dir=args.independent_dir,
        full_pool_dir=args.full_pool_dir,
        formula_npz=args.formula_npz,
        output_dir=args.output_dir,
        ehull_path=args.ehull,
        consensus_path=args.consensus_deciles,
        f3_path=args.f3_json,
        f2r_path=args.f2r_json,
        ladder_path=args.ladder_csv,
        threeaxis_path=args.threeaxis_csv,
        f_layout=args.f_layout,
        stem_name=args.stem_name,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
