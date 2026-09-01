#!/usr/bin/env python3
"""Render the former main-text Fig. 4c,d and Fig. 5a-c as SI assets.

The source figures remain untouched.  The first asset captures the actual
``paper_figs.fig4_validation`` artists before its normal save step and then
reflows only panels c and d.  The second calls the original
``fig5_ranking.panel_a``--``panel_c`` functions on their original source
tables.  Thus the scientific transforms, labels and numerical annotations are
shared with the canonical plotting code rather than reconstructed from a PDF.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import paper_figs as pf  # noqa: E402
import fig5_ranking as ranking  # noqa: E402
from experiments.pu_synthesizability_20260821 import (  # noqa: E402
    plot_merged_fig45_nature as merged,
)


# paper_figs' custom mathtext setup leaves the generic ``cursive`` fallback
# unresolved on the render host.  Point that fallback at the same installed
# sans family so vector export stays warning-free and typographically unified.
mpl.rcParams.update({
    "font.cursive": ["Arial", "Liberation Sans"],
    "mathtext.cal": "Arial",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


CM = 1.0 / 2.54
DEFAULT_OUTPUT_DIR = ROOT / "tex" / "key-file"
VALIDATION_STEM = "2026-08-23-SI-validation-boundary-and-omission-robustness"
RANKING_STEM = "2026-08-23-SI-polymorph-ranking-boundaries"
PHYSICAL_STEM = "2026-08-23-SI-energy-phonon-record"


def _capture_fig4_validation() -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Run the canonical Fig. 4 builder without writing its canonical files."""
    captured: dict[str, Any] = {}
    original_save = pf.save
    original_stamp = pf.stamp

    def capture_save(fig: plt.Figure, name: str) -> None:
        captured["figure"] = fig
        captured["name"] = name

    def capture_stamp(fig: plt.Figure, items, *args, **kwargs) -> None:
        del fig, args, kwargs
        captured["panels"] = {item[1]: item[0] for item in items}

    pf.save = capture_save
    pf.stamp = capture_stamp
    try:
        pf.fig4_validation()
    finally:
        pf.save = original_save
        pf.stamp = original_stamp

    if captured.get("name") != "fig4_validation":
        raise RuntimeError("Canonical Fig. 4 builder did not reach its save step")
    panels = captured.get("panels", {})
    if not {"c", "d"}.issubset(panels):
        raise RuntimeError("Canonical Fig. 4 builder did not expose panels c and d")
    return captured["figure"], panels


def _stamp_si_panels(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    labels = [chr(ord("a") + i) for i in range(len(axes))]
    pf.stamp(fig, list(zip(axes, labels)), pad_x=0.009, pad_y=0.008, size=8)


def build_validation_boundary_figure() -> tuple[plt.Figure, dict[str, Any]]:
    """Reflow old Fig. 4c,d into a two-panel SI figure."""
    fig, original_panels = _capture_fig4_validation()
    ax_boundary = original_panels["c"]
    ax_omission = original_panels["d"]

    for ax in list(fig.axes):
        if ax not in (ax_boundary, ax_omission):
            ax.remove()

    width_cm, height_cm = 18.3, 7.6
    fig.set_size_inches(width_cm * CM, height_cm * CM, forward=True)
    ax_boundary.set_position([1.55 / width_cm, 1.28 / height_cm,
                              7.25 / width_cm, 4.72 / height_cm])
    ax_omission.set_position([10.18 / width_cm, 1.28 / height_cm,
                              7.42 / width_cm, 4.72 / height_cm])

    # The original legends already encode the necessary series.  Standardise
    # their type size after the geometry change; no prose or title is added.
    for ax in (ax_boundary, ax_omission):
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(7.2)

    ax_boundary.set_ylabel("Fraction of structures")
    ax_omission.set_ylabel("Damage detection")

    _stamp_si_panels(fig, [ax_boundary, ax_omission])

    rho_hist = pd.read_csv(pf.DATA / "fig4_rho_hist.csv")
    omission = pd.read_csv(pf.DATA / "fig3_loko.csv")
    meta = {
        "panel_map": {"a": "old Fig. 4c", "b": "old Fig. 4d"},
        "source_logic": "src/paper_figs.py::fig4_validation",
        "rho_hist_rows": int(len(rho_hist)),
        "omitted_classes": omission["held"].astype(str).tolist(),
        "width_cm": width_cm,
        "height_cm": height_cm,
    }
    return fig, meta


def _load_ranking_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ca = pd.read_csv(ranking.DATA / "fig3_coverage_accuracy.csv")
    ca["s"] = ca.rule.map(ranking.SHORT)
    ca["c"] = ca.s.map(ranking.CMAP)

    top1 = pd.read_csv(ranking.DATA / "fig7_top1.csv")
    top1["s"] = top1.rule.map(ranking.SHORT)
    top1["c"] = top1.s.map(ranking.CMAP)

    rulesets = pd.read_csv(ranking.DATA / "rank_rulesets.csv").set_index("rule")
    return ca, top1, rulesets


def build_polymorph_ranking_figure() -> tuple[plt.Figure, dict[str, Any]]:
    """Reflow old Fig. 5a-c into a three-panel SI figure."""
    ca, top1, rulesets = _load_ranking_inputs()
    width_cm, height_cm = 18.3, 13.4
    fig = plt.figure(figsize=(width_cm * CM, height_cm * CM), facecolor="white")

    # The commitment plane is the primary evidence and receives the full top
    # row.  The two boundary views share the lower row at the same visual rank.
    ax_commit = fig.add_axes([1.75 / width_cm, 7.75 / height_cm,
                              15.85 / width_cm, 4.78 / height_cm])
    ax_ties = fig.add_axes([3.15 / width_cm, 1.24 / height_cm,
                            5.30 / width_cm, 5.18 / height_cm])
    ax_energy_wrong = fig.add_axes([12.15 / width_cm, 1.24 / height_cm,
                                    5.45 / width_cm, 5.18 / height_cm])

    ranking.panel_a(ax_commit, ca, rulesets)
    ranking.panel_b(ax_ties, top1, rulesets)
    ranking.panel_c(ax_energy_wrong, ca)
    ax_commit.set_xlabel("Fraction of pairs distinguished")
    ax_commit.set_ylabel("Accuracy on distinguished pairs")
    ax_ties.set_xlabel("Groups where no single structure\ncan be picked (%)")
    ax_energy_wrong.set_xlabel("Accuracy on pairs\nDFT energy ranks wrongly")
    _stamp_si_panels(fig, [ax_commit, ax_ties, ax_energy_wrong])

    meta = {
        "panel_map": {
            "a": "old Fig. 5a",
            "b": "old Fig. 5b",
            "c": "old Fig. 5c",
        },
        "source_logic": "src/fig5_ranking.py::panel_a--panel_c",
        "ranking_criteria": int(len(ca) + len(rulesets)),
        "rule_sets": list(ranking.RSET),
        "width_cm": width_cm,
        "height_cm": height_cm,
    }
    return fig, meta


def build_energy_phonon_record_figure() -> tuple[plt.Figure, dict[str, Any]]:
    """Render the former energy/phonon/record panel as a standalone SI asset."""
    merged.style()
    width_cm, height_cm = 10.0, 7.0
    fig = plt.figure(figsize=(width_cm * CM, height_cm * CM), facecolor="white")
    grid = fig.add_gridspec(1, 1, left=0.16, right=0.97, bottom=0.16, top=0.97)
    outer = fig.add_subplot(grid[0])
    record = merged.panel_physical_states_si(
        outer,
        merged.DATA / "fig6_threeaxis.csv",
        panel_letter=None,
    )
    return fig, {
        "panel_map": {"single": "old Fig. 5f"},
        "source_logic": "plot_merged_fig45_nature.py::panel_physical_states_si",
        "width_cm": width_cm,
        "height_cm": height_cm,
        **record,
    }


def _save_bundle(fig: plt.Figure, stem: Path) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [stem.with_suffix(ext) for ext in (".pdf", ".svg", ".png")]
    # paper_figs uses tight bounding boxes for its canonical page.  These SI
    # assets instead preserve the explicit 18.3-cm production width.
    with warnings.catch_warnings(), mpl.rc_context(
        {"savefig.bbox": None, "savefig.pad_inches": 0.0}
    ):
        warnings.filterwarnings(
            "ignore",
            message="The py23 module has been deprecated.*",
            category=DeprecationWarning,
        )
        fig.savefig(outputs[0], facecolor="white")
        fig.savefig(outputs[1], facecolor="white")
        fig.savefig(outputs[2], dpi=600, facecolor="white")
    return [str(path) for path in outputs]


def render_all(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir = Path(output_dir)
    result: dict[str, Any] = {}
    builders = (
        (VALIDATION_STEM, build_validation_boundary_figure),
        (RANKING_STEM, build_polymorph_ranking_figure),
        (PHYSICAL_STEM, build_energy_phonon_record_figure),
    )
    for stem_name, builder in builders:
        fig, meta = builder()
        try:
            meta["files"] = _save_bundle(fig, output_dir / stem_name)
            result[stem_name] = meta
        finally:
            plt.close(fig)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    result = render_all(args.output_dir)
    if args.metadata is not None:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
