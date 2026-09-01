#!/usr/bin/env python3
"""Attribute the L4 gate to its seven member laws for the Supplementary Information.

L4 is the fail-on-evidence union of D1 and D3--D8; D2 is not an L4 member.
The first panel uses the Shapley allocation of this union, which divides a
structure that violates k laws equally among those k laws and therefore sums
exactly to the observed L4 violation rate.  The second panel removes one law
at a time and reports the corresponding experimental-satisfaction gain and
hard-negative-screening loss.  Unknown inputs remain retained throughout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CM = 1 / 2.54
L4_LAWS = ("D1", "D3", "D4", "D5", "D6", "D7", "D8")
COHORT_ORDER = ("experimental", "pu_negative")
COHORT_LABELS = {
    "experimental": "Experimental structures",
    "pu_negative": "hard-to-synthesize set",
}
LAW_COLORS = {
    "D1": "#005B93",
    "D3": "#54ACD6",
    "D4": "#35A7D8",
    "D5": "#9E67B6",
    "D6": "#E46F64",
    "D7": "#0A5A3C",
    "D8": "#D6564C",
}
# The release tables keep the archived ``D<k>_verdict`` column names; the figure
# prints the manuscript names (Law 1--Law 8, Set 1--Set 4).
LAW_LABELS = {law: "Law " + law[1:] for law in L4_LAWS}
EXPECTED_ROWS = {"experimental": 99_162, "pu_negative": 364_771}
EXPECTED_UNIQUE = {"experimental": 99_162, "pu_negative": 364_592}
ALLOWED = {"pass", "explicit_violation", "no_verdict"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.2,
            "axes.labelsize": 8.6,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def _darken(color: str, factor: float = 0.70) -> tuple[float, float, float, float]:
    rgb = np.asarray(to_rgb(color), dtype=float) * factor
    return float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0


def load_cohort(directories: list[Path], *, cohort: str) -> tuple[pd.DataFrame, dict]:
    if cohort not in EXPECTED_ROWS:
        raise ValueError(f"unexpected cohort: {cohort}")
    columns = [
        "record_index",
        "cif_sha256",
        "L4_verdict",
        *(f"{law}_verdict" for law in L4_LAWS),
    ]
    files = [file for directory in directories for file in sorted(directory.glob("part-*.parquet"))]
    if not files:
        raise FileNotFoundError(f"no Parquet shards under {directories}")
    parts = []
    for file in files:
        part = pd.read_parquet(file, columns=columns)
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    if len(frame) != EXPECTED_ROWS[cohort]:
        raise ValueError(
            f"{cohort} row count {len(frame):,} != {EXPECTED_ROWS[cohort]:,}"
        )
    indices = pd.to_numeric(frame["record_index"], errors="raise").astype(int)
    if indices.duplicated().any() or set(indices) != set(range(len(frame))):
        raise ValueError(f"{cohort} record_index accounting failed")
    return frame, {
        "directories": [str(path.resolve()) for path in directories],
        "files": [str(path.resolve()) for path in files],
        "rows": int(len(frame)),
    }


def summarize_cohort(
    frame: pd.DataFrame,
    *,
    cohort: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return overlap-aware and leave-one-out L4 attribution by unique CIF."""

    verdict_columns = ["L4_verdict", *(f"{law}_verdict" for law in L4_LAWS)]
    required = {"cif_sha256", *verdict_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if frame["cif_sha256"].isna().any():
        raise ValueError("cif_sha256 contains null values")
    for column in verdict_columns:
        values = set(frame[column].dropna().astype(str))
        if frame[column].isna().any() or not values.issubset(ALLOWED):
            raise ValueError(f"invalid verdict values in {column}: {sorted(values)}")

    consistency = frame.groupby("cif_sha256", sort=False)[verdict_columns].nunique()
    if int(consistency.to_numpy().max(initial=0)) > 1:
        raise ValueError("duplicate release records disagree on a PRIS verdict")
    unique = frame.drop_duplicates("cif_sha256").reset_index(drop=True)
    matrix = np.column_stack(
        [
            unique[f"{law}_verdict"].astype(str).eq("explicit_violation").to_numpy()
            for law in L4_LAWS
        ]
    )
    n_active = matrix.sum(axis=1)
    union = matrix.any(axis=1)
    stored = unique["L4_verdict"].astype(str).eq("explicit_violation").to_numpy()
    mismatches = int(np.count_nonzero(union != stored))
    if mismatches:
        raise ValueError(f"member-law union differs from L4 for {mismatches} unique CIFs")

    n = len(unique)
    full_rate_pp = 100.0 * float(union.mean())
    rows = []
    for index, law in enumerate(L4_LAWS):
        active = matrix[:, index]
        allocated = np.divide(
            active.astype(float),
            n_active,
            out=np.zeros(n, dtype=float),
            where=n_active > 0,
        )
        without = np.delete(matrix, index, axis=1).any(axis=1)
        direct_rate_pp = 100.0 * float(active.mean())
        unique_rate_pp = 100.0 * float((active & (n_active == 1)).mean())
        loo_rate_pp = 100.0 * float(without.mean())
        rows.append(
            {
                "cohort": cohort,
                "law": law,
                "unique_cifs": n,
                "l4_violation_rate_pp": full_rate_pp,
                "direct_violation_rate_pp": direct_rate_pp,
                "gross_coverage_of_l4_pct": (
                    100.0 * float(active[union].mean()) if union.any() else 0.0
                ),
                "allocated_contribution_pp": 100.0 * float(allocated.mean()),
                "unique_violation_n": int(np.count_nonzero(active & (n_active == 1))),
                "leave_one_out_loss_pp": full_rate_pp - loo_rate_pp,
                "leave_one_out_l4_violation_rate_pp": loo_rate_pp,
                "unique_contribution_pp": unique_rate_pp,
            }
        )
    result = pd.DataFrame(rows)
    if not np.isclose(
        result["allocated_contribution_pp"].sum(), full_rate_pp, atol=1e-10, rtol=0
    ):
        raise AssertionError("allocated contributions do not sum to the L4 rate")
    metadata = {
        "cohort": cohort,
        "rows": int(len(frame)),
        "unique_cifs": int(n),
        "duplicate_release_rows": int(len(frame) - n),
        "l4_violation_n": int(union.sum()),
        "l4_violation_rate_pp": full_rate_pp,
        "l4_satisfaction_rate_pct": 100.0 - full_rate_pp,
        "l4_union_mismatches": mismatches,
        "unknown_policy": "retained unless another L4 member is an explicit violation",
        "members": list(L4_LAWS),
        "excluded_pris_law": "D2 is not a member of L4",
    }
    return result, metadata


def draw_figure(
    summary: pd.DataFrame,
    metadata: dict[str, dict[str, object]],
) -> plt.Figure:
    _style()
    required = {
        "cohort",
        "law",
        "allocated_contribution_pp",
        "leave_one_out_loss_pp",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"summary lacks columns: {missing}")
    fig = plt.figure(figsize=(18.3 * CM, 7.8 * CM), facecolor="white")
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.18, 0.82),
        left=0.095,
        right=0.985,
        bottom=0.20,
        top=0.82,
        wspace=0.36,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])

    y = np.arange(len(COHORT_ORDER))
    left = np.zeros(len(COHORT_ORDER), dtype=float)
    for law in L4_LAWS:
        values = np.asarray(
            [
                float(
                    summary.loc[
                        summary["cohort"].eq(cohort) & summary["law"].eq(law),
                        "allocated_contribution_pp",
                    ].iloc[0]
                )
                for cohort in COHORT_ORDER
            ]
        )
        color = LAW_COLORS[law]
        ax_a.barh(
            y,
            values,
            left=left,
            height=0.52,
            color=to_rgba(color, 0.80),
            edgecolor=_darken(color),
            linewidth=0.45,
        )
        for yi, start, width in zip(y, left, values, strict=True):
            if width >= 2.4:
                ax_a.text(
                    start + width / 2,
                    yi,
                    LAW_LABELS[law],
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color="white",
                    fontweight="bold",
                )
        left += values
    for yi, cohort in zip(y, COHORT_ORDER, strict=True):
        total = float(metadata[cohort]["l4_violation_rate_pp"])
        ax_a.text(total + 0.8, yi, f"{total:.1f}%", ha="left", va="center", fontsize=7.6)
    ax_a.set_yticks(y, [COHORT_LABELS[name] for name in COHORT_ORDER])
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, max(56.0, float(left.max()) + 4.0))
    ax_a.set_xlabel("Contribution to Set 4 violation rate (percentage points)")
    ax_a.tick_params(axis="y", length=0)
    ax_a.grid(False)
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markersize=6.0,
            markerfacecolor=to_rgba(LAW_COLORS[law], 0.80),
            markeredgecolor=_darken(LAW_COLORS[law]),
            label=LAW_LABELS[law],
        )
        for law in L4_LAWS
    ]
    ax_a.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.03),
        ncol=7,
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.55,
        borderaxespad=0,
    )

    exp = summary.loc[summary["cohort"].eq("experimental")].set_index("law")
    pu = summary.loc[summary["cohort"].eq("pu_negative")].set_index("law")
    x = exp.loc[list(L4_LAWS), "leave_one_out_loss_pp"].to_numpy(float)
    y_loss = pu.loc[list(L4_LAWS), "leave_one_out_loss_pp"].to_numpy(float)
    plot_y = np.maximum(y_loss, 5e-4)
    for xi, yi, law in zip(x, plot_y, L4_LAWS, strict=True):
        color = LAW_COLORS[law]
        ax_b.scatter(
            [xi],
            [yi],
            s=54,
            facecolor=to_rgba(color, 0.68),
            edgecolor=_darken(color),
            linewidth=0.7,
            zorder=3,
        )
        offset = (-5, 4) if law in {"D1", "D7"} else (5, 2)
        align = "right" if law in {"D1", "D7"} else "left"
        ax_b.annotate(
            LAW_LABELS[law],
            (xi, yi),
            xytext=offset,
            textcoords="offset points",
            ha=align,
            va="bottom",
            fontsize=7.4,
            color=_darken(color),
            fontweight="bold",
        )
    ax_b.set_yscale("log")
    ticks = [0.001, 0.01, 0.1, 1, 10, 50]
    ax_b.yaxis.set_major_locator(FixedLocator(ticks))
    ax_b.yaxis.set_major_formatter(FixedFormatter(["0.001", "0.01", "0.1", "1", "10", "50"]))
    ax_b.set_xlim(-0.15, max(5.6, float(x.max()) + 0.6))
    ax_b.set_ylim(5e-4, 80)
    ax_b.set_xlabel("Experimental satisfaction regained\n(percentage points)")
    ax_b.set_ylabel("Hard-negative screening lost\n(percentage points; log scale)")
    ax_b.grid(False)

    for ax, label in ((ax_a, "a"), (ax_b, "b")):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            -0.12,
            1.10,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    return fig


def render(
    output_dir: Path,
    *,
    experimental_dirs: list[Path],
    pu_dirs: list[Path],
    stem_name: str = "2026-08-23-SI-L4-per-law-contributions",
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {}
    sources = {}
    frames["experimental"], sources["experimental"] = load_cohort(
        experimental_dirs, cohort="experimental"
    )
    frames["pu_negative"], sources["pu_negative"] = load_cohort(
        pu_dirs, cohort="pu_negative"
    )
    tables = []
    cohort_metadata = {}
    for cohort in COHORT_ORDER:
        table, cohort_metadata[cohort] = summarize_cohort(frames[cohort], cohort=cohort)
        if int(cohort_metadata[cohort]["unique_cifs"]) != EXPECTED_UNIQUE[cohort]:
            raise ValueError(
                f"{cohort} unique-CIF count differs from the frozen denominator"
            )
        tables.append(table)
    summary = pd.concat(tables, ignore_index=True)
    fig = draw_figure(summary, cohort_metadata)
    stem = output_dir / stem_name
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white", metadata={"Title": ""})
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)
    table_path = output_dir / f"{stem_name}-data.csv"
    summary.to_csv(table_path, index=False, float_format="%.10g")
    status = {
        "figure": stem_name,
        "members": list(L4_LAWS),
        "Law 2": "not a member of Set 4",
        "attribution": "coverage-game Shapley allocation; each k-fold violation contributes 1/k to each active law",
        "leave_one_out": "loss in the Set 4 explicit-violation rate after removing one law",
        "cohorts": cohort_metadata,
        "sources": sources,
        "outputs": {
            suffix: str(stem.with_suffix(suffix))
            for suffix in (".pdf", ".svg", ".png", ".tiff")
        },
        "data": str(table_path),
    }
    status_path = output_dir / f"{stem_name}-STATUS.json"
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    caption_path = output_dir / f"{stem_name}-CAPTION.md"
    caption_path.write_text(
        "**Supplementary Figure | Law-wise attribution of the Set 4 gate.** "
        "**a**, Additive contribution of Law 1 and Law 3--Law 8 to the Set 4 explicit-violation rate "
        "for 99,162 unique experimental structures and 364,592 unique PU hard negatives. "
        "For structures violating multiple laws, one count is divided equally among the "
        "active laws, so the segments sum exactly to the Set 4 rate. **b**, Leave-one-law-out "
        "trade-off. The horizontal axis is the experimental-structure satisfaction regained "
        "when a law is removed; the vertical axis is the hard-negative screening lost. "
        "Only explicit violations remove structures and no-verdict cases remain in the queue. "
        "Law 2 is not shown because it is not a member of Set 4.\n",
        encoding="utf-8",
    )
    manifest = [
        stem.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png", ".tiff")
    ] + [table_path, status_path, caption_path]
    (output_dir / f"{stem_name}-SHA256SUMS").write_text(
        "\n".join(f"{_sha256(path)}  {path.name}" for path in manifest) + "\n",
        encoding="utf-8",
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/20260823_l4_contribution_si",
    )
    parser.add_argument(
        "--stem",
        default="2026-08-23-SI-L4-per-law-contributions",
    )
    args = parser.parse_args()
    status = render(
        args.output_dir,
        experimental_dirs=[
            ROOT / "outputs/20260821_pu_synthesizability/binary_v1/experimental",
            ROOT / "outputs/20260821_pu_synthesizability/binary_v1_tail/experimental",
        ],
        pu_dirs=[ROOT / "outputs/20260821_pu_synthesizability/binary_v1/pu_negative"],
        stem_name=args.stem,
    )
    print(json.dumps({"outputs": status["outputs"], "cohorts": status["cohorts"]}, indent=2))


if __name__ == "__main__":
    main()
