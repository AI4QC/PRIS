#!/usr/bin/env python3
"""Synthesis-score operating curve for inverse-design queue reduction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _finite_scores(frame: pd.DataFrame, score_column: str) -> np.ndarray:
    if score_column not in frame:
        raise ValueError(f"missing score column: {score_column}")
    scores = pd.to_numeric(frame[score_column], errors="raise").to_numpy(float)
    if not np.isfinite(scores).all():
        raise ValueError(f"{score_column} must be finite for every row")
    return scores


def support_matched_calibration_mask(
    forward: pd.DataFrame,
    inverse: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, object]]:
    """Match the PSS descriptor support used to calibrate an inverse queue.

    PSS values with different numbers of observed descriptors are not directly
    exchangeable because missing terms use frozen training medians.  When both
    queues record descriptor coverage and the inverse queue has one common
    support level, calibrate only on experimental forward structures at that
    same level.  Small synthetic test frames that predate these audit columns
    retain the legacy all-experimental behaviour explicitly.
    """

    if "made" not in forward:
        raise ValueError("forward frame lacks made")
    experimental = forward["made"].astype(bool).to_numpy()
    forward_column = next(
        (name for name in ("synthesis_n_observed", "n_observed") if name in forward),
        None,
    )
    inverse_column = next(
        (name for name in ("synthesis_n_observed", "n_observed") if name in inverse),
        None,
    )
    if forward_column is None or inverse_column is None:
        return experimental, {
            "mode": "all experimental rows; descriptor-support columns unavailable",
            "n_observed": None,
            "experimental_n": int(experimental.sum()),
        }

    forward_support = pd.to_numeric(forward[forward_column], errors="raise")
    inverse_support = pd.to_numeric(inverse[inverse_column], errors="raise")
    if forward_support.isna().any() or inverse_support.isna().any():
        raise ValueError("PSS descriptor-support counts must be complete")
    inverse_levels = sorted({int(value) for value in inverse_support.unique()})
    if len(inverse_levels) != 1:
        raise ValueError(
            "inverse queue must have one descriptor-support level for matched calibration"
        )
    support = inverse_levels[0]
    matched = experimental & forward_support.eq(support).to_numpy()
    if not matched.any():
        raise ValueError(
            f"no experimental forward structures match n_observed={support}"
        )
    return matched, {
        "mode": "descriptor-support matched",
        "n_observed": support,
        "experimental_n": int(matched.sum()),
        "forward_experimental_support_counts": {
            str(int(key)): int(value)
            for key, value in forward_support[experimental]
            .value_counts()
            .sort_index()
            .items()
        },
        "inverse_support_counts": {
            str(int(key)): int(value)
            for key, value in inverse_support.value_counts().sort_index().items()
        },
    }


def build_design_frontier(
    forward: pd.DataFrame,
    inverse: pd.DataFrame,
    *,
    score_column: str,
    target_retentions: Iterable[float],
    inverse_priority_mask: Iterable[bool] | None = None,
    experimental_calibration_mask: Iterable[bool] | None = None,
) -> pd.DataFrame:
    """Transfer thresholds calibrated on forward experimental structures.

    Lower synthesis scores are screened.  A strict inequality retains every
    score tied at the cutoff, matching the established synthesis-score policy.
    The same cutoff is then applied to the forward theoretical queue and the
    inverse-design queue.
    """

    if "made" not in forward:
        raise ValueError("forward frame lacks made")
    experimental_mask = forward["made"].astype(bool).to_numpy()
    if not experimental_mask.any() or experimental_mask.all():
        raise ValueError("forward frame must contain experimental and theoretical rows")
    forward_scores = _finite_scores(forward, score_column)
    inverse_scores = _finite_scores(inverse, score_column)
    if inverse_priority_mask is None:
        priority = np.ones(len(inverse), dtype=bool)
    else:
        priority = np.asarray(tuple(inverse_priority_mask), dtype=bool)
        if len(priority) != len(inverse) or not priority.any():
            raise ValueError("inverse priority mask must match the frame and be non-empty")
    if experimental_calibration_mask is None:
        calibration_mask = experimental_mask
    else:
        calibration_mask = np.asarray(
            tuple(experimental_calibration_mask), dtype=bool
        )
        if len(calibration_mask) != len(forward):
            raise ValueError(
                "experimental calibration mask must match the forward frame"
            )
        if not calibration_mask.any() or np.any(
            calibration_mask & ~experimental_mask
        ):
            raise ValueError(
                "experimental calibration mask must be a non-empty subset "
                "of experimental rows"
            )
    experimental_scores = forward_scores[calibration_mask]
    theoretical_scores = forward_scores[~experimental_mask]

    rows: list[dict[str, float | int]] = []
    for target in target_retentions:
        target = float(target)
        if not 0.0 < target <= 1.0:
            raise ValueError("target retentions must lie in (0, 1]")
        screened_budget = int(
            np.floor((1.0 - target) * len(experimental_scores) + 1e-12)
        )
        ordered = np.sort(experimental_scores)
        cutoff = (
            float("-inf")
            if screened_budget == 0
            else float(ordered[screened_budget])
        )
        experimental_screened = experimental_scores < cutoff
        theory_screened = theoretical_scores < cutoff
        inverse_screened = inverse_scores < cutoff
        rows.append(
            {
                "target_experimental_retention": target,
                "cutoff_score": cutoff,
                "experimental_retention": float((~experimental_screened).mean()),
                "forward_theory_reduction": float(theory_screened.mean()),
                "inverse_queue_reduction": float(inverse_screened.mean()),
                "inverse_priority_retention": float(
                    (~inverse_screened[priority]).mean()
                ),
                "experimental_count": int(len(experimental_scores)),
                "forward_theory_count": int(len(theoretical_scores)),
                "inverse_count": int(len(inverse_scores)),
            }
        )
    return pd.DataFrame(rows)


def build_transferred_frontier(
    forward: pd.DataFrame,
    inverse: pd.DataFrame,
    *,
    cutoffs: pd.DataFrame,
    score_column: str,
) -> pd.DataFrame:
    """Apply externally calibrated synthesis-score cutoffs without refitting."""

    required = {"reference_experimental_retention", "cutoff_score"}
    missing = sorted(required.difference(cutoffs.columns))
    if missing:
        raise ValueError(f"cutoff table lacks columns: {missing}")
    if "made" not in forward:
        raise ValueError("forward frame lacks made")
    experimental = forward["made"].astype(bool).to_numpy()
    if not experimental.any() or experimental.all():
        raise ValueError("forward frame must contain experimental and theoretical rows")
    forward_scores = _finite_scores(forward, score_column)
    inverse_scores = _finite_scores(inverse, score_column)
    rows: list[dict[str, float | int]] = []
    for row in cutoffs.itertuples(index=False):
        reference_retention = float(row.reference_experimental_retention)
        cutoff = float(row.cutoff_score)
        if not 0.0 < reference_retention <= 1.0 or not np.isfinite(cutoff):
            raise ValueError("reference retentions and cutoffs must be finite and valid")
        forward_screened = forward_scores < cutoff
        inverse_screened = inverse_scores < cutoff
        rows.append(
            {
                "reference_experimental_retention": reference_retention,
                "cutoff_score": cutoff,
                "forward_experimental_retention": float(
                    (~forward_screened[experimental]).mean()
                ),
                "forward_theory_reduction": float(
                    forward_screened[~experimental].mean()
                ),
                "inverse_queue_reduction": float(inverse_screened.mean()),
                "forward_experimental_count": int(experimental.sum()),
                "forward_theory_count": int((~experimental).sum()),
                "inverse_count": int(len(inverse)),
            }
        )
    return pd.DataFrame(rows)


def rule_operating_point(
    forward: pd.DataFrame,
    inverse: pd.DataFrame,
    *,
    verdict_column: str,
    method: str,
    inverse_priority_mask: Iterable[bool] | None = None,
) -> dict[str, float | str]:
    """Return one discrete-rule point, removing explicit violations only."""

    if "made" not in forward:
        raise ValueError("forward frame lacks made")
    if verdict_column not in forward or verdict_column not in inverse:
        raise ValueError(f"missing verdict column: {verdict_column}")
    experimental = forward["made"].astype(bool).to_numpy()
    if not experimental.any() or experimental.all():
        raise ValueError("forward frame must contain experimental and theoretical rows")
    forward_violation = forward[verdict_column].astype(str).eq("reject").to_numpy()
    inverse_violation = inverse[verdict_column].astype(str).eq("reject").to_numpy()
    if inverse_priority_mask is None:
        priority = np.ones(len(inverse), dtype=bool)
    else:
        priority = np.asarray(tuple(inverse_priority_mask), dtype=bool)
        if len(priority) != len(inverse) or not priority.any():
            raise ValueError("inverse priority mask must match the frame and be non-empty")
    return {
        "method": str(method),
        "experimental_retention": float((~forward_violation[experimental]).mean()),
        "forward_theory_reduction": float(forward_violation[~experimental].mean()),
        "inverse_queue_reduction": float(inverse_violation.mean()),
        "inverse_priority_retention": float((~inverse_violation[priority]).mean()),
    }


def _figure_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 6.9,
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
            "savefig.pad_inches": 0.02,
        }
    )


def _rule_marker_sizes(rule_points: pd.DataFrame) -> dict[str, float]:
    """Nest coincident rule markers so that every method remains visible."""

    groups: dict[tuple[float, float], list[str]] = {}
    for point in rule_points.itertuples(index=False):
        key = (
            round(float(point.inverse_priority_retention), 2),
            round(float(point.inverse_queue_reduction), 2),
        )
        groups.setdefault(key, []).append(str(point.method))

    sizes: dict[str, float] = {}
    for methods in groups.values():
        if len(methods) == 1:
            sizes[methods[0]] = 52.0
            continue
        if len(methods) == 2:
            sizes.update({methods[0]: 90.0, methods[1]: 34.0})
            continue
        for method, size in zip(
            methods,
            np.geomspace(210.0, 32.0, len(methods), dtype=float),
            strict=True,
        ):
            sizes[method] = float(size)
    return sizes


def draw_design_panel(
    ax,
    frontier: pd.DataFrame,
    rule_points: pd.DataFrame,
    *,
    inverse_priority_label: str,
    panel_letter: str | None = "f",
    visual_style: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Draw the inverse-design queue trade-off on an existing axes.

    This is the single visual implementation used by both the standalone
    review panel and the merged manuscript candidate.  The panel contains
    only measured data, axes, a compact legend and an optional panel letter;
    protocol qualifications belong in the caption.
    """

    from matplotlib.colors import to_rgba, to_rgb
    from matplotlib.lines import Line2D

    required_frontier = {
        "inverse_priority_retention",
        "inverse_queue_reduction",
    }
    missing_frontier = sorted(required_frontier.difference(frontier.columns))
    if missing_frontier:
        raise ValueError(f"frontier lacks columns: {missing_frontier}")
    required_rules = required_frontier | {"method"}
    missing_rules = sorted(required_rules.difference(rule_points.columns))
    if missing_rules:
        raise ValueError(f"rule points lack columns: {missing_rules}")

    orange = "#E88A8E"
    ink = "#1F2022"
    rule_colors = {
        "L1": "#005B93",
        "L2": "#9861B0",
        "L3": "#D6564C",
        "L4": "#0A5A3C",
    }
    marker_shapes = {"L1": "o", "L2": "s", "L3": "D", "L4": "^"}
    unknown = sorted(set(rule_points["method"].astype(str)) - set(rule_colors))
    if unknown:
        raise ValueError(f"unsupported rule methods: {unknown}")

    def edge(color: str) -> tuple[float, float, float, float]:
        rgb = np.asarray(to_rgb(color), dtype=float) * 0.70
        return float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0

    resolved_style: dict[str, object] = {
        "line_width": 1.7,
        "line_alpha": 1.0,
        "marker_area": 22.0,
        "marker_size": 4.6,
        "marker_face_alpha": 0.56,
        "rule_face_alpha": 0.70,
        "marker_edge_width": 0.55,
        "uniform_rule_markers": False,
        "rule_marker_shapes": None,
        "rule_marker_area_overrides": None,
        "gid_prefix": None,
    }
    if visual_style is not None:
        resolved_style.update(dict(visual_style))
    custom_shapes = resolved_style.get("rule_marker_shapes")
    if custom_shapes is not None:
        if not isinstance(custom_shapes, Mapping):
            raise TypeError("rule_marker_shapes must be a mapping")
        marker_shapes.update({str(key): str(value) for key, value in custom_shapes.items()})
    line_width = float(resolved_style["line_width"])
    line_alpha = float(resolved_style["line_alpha"])
    marker_area = float(resolved_style["marker_area"])
    marker_size = float(resolved_style["marker_size"])
    marker_face_alpha = float(resolved_style["marker_face_alpha"])
    rule_face_alpha = float(resolved_style["rule_face_alpha"])
    marker_edge_width = float(resolved_style["marker_edge_width"])
    gid_prefix = resolved_style.get("gid_prefix")
    series_gid = f"{gid_prefix}-primary-series" if gid_prefix else None
    marker_gid = f"{gid_prefix}-primary-marker" if gid_prefix else None

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)

    ix = 100 * frontier["inverse_priority_retention"].to_numpy(float)
    iy = 100 * frontier["inverse_queue_reduction"].to_numpy(float)
    (curve,) = ax.plot(
        ix,
        iy,
        color=to_rgba(orange, line_alpha),
        lw=line_width,
        zorder=2,
    )
    curve.set_gid(series_gid)
    stride = max(1, len(frontier) // 18)
    threshold_markers = ax.scatter(
        ix[::stride],
        iy[::stride],
        s=marker_area,
        facecolor=to_rgba(orange, marker_face_alpha),
        edgecolor=edge(orange),
        linewidth=marker_edge_width,
        zorder=3,
    )
    threshold_markers.set_gid(marker_gid)

    if bool(resolved_style["uniform_rule_markers"]):
        marker_sizes = {str(method): marker_area for method in rule_points["method"]}
    else:
        marker_sizes = _rule_marker_sizes(rule_points)
    # One marker area does not buy one visual weight: a five-pointed star leaves
    # most of its bounding box empty, so a caller that draws a star beside disks
    # can raise that method's area on its own.  Everything else keeps the shared
    # area, and the key entry follows the marker it stands for.
    area_overrides = resolved_style.get("rule_marker_area_overrides")
    if area_overrides is not None:
        if not isinstance(area_overrides, Mapping):
            raise TypeError("rule_marker_area_overrides must be a mapping")
        overridden = {str(key) for key in area_overrides}
        marker_sizes.update(
            {
                str(key): float(value)
                for key, value in area_overrides.items()
                if str(key) in marker_sizes
            }
        )
    else:
        overridden = set()

    def legend_marker_size(name: str) -> float:
        if name in overridden:
            return float(np.sqrt(marker_sizes[name]))
        return marker_size

    for point in rule_points.itertuples(index=False):
        method = str(point.method)
        color = rule_colors[method]
        is_outer_ring = marker_sizes[method] > 100
        rule_marker = ax.scatter(
            [100 * point.inverse_priority_retention],
            [100 * point.inverse_queue_reduction],
            s=marker_sizes[method],
            marker=marker_shapes[method],
            facecolor="none" if is_outer_ring else to_rgba(color, rule_face_alpha),
            edgecolor=edge(color),
            linewidth=1.2 if is_outer_ring else marker_edge_width,
            clip_on=False,
            zorder=7 if is_outer_ring else 5,
        )
        rule_marker.set_gid(marker_gid)

    ax.set_xlabel(str(inverse_priority_label), labelpad=2)
    ax.set_ylabel("Generated queue\nreduction (%)", labelpad=3)
    rule_x = 100 * rule_points["inverse_priority_retention"].to_numpy(float)
    rule_y = 100 * rule_points["inverse_queue_reduction"].to_numpy(float)
    xmin = max(0.0, min(float(np.min(ix)), float(np.min(rule_x))) - 3.0)
    plot_xlim = (xmin, 102.5)
    ax.set_xlim(*plot_xlim)
    ymax = max(float(np.max(iy)), float(np.max(rule_y)))
    plot_ylim = (-2.5, min(100.0, ymax * 1.18 + 3.0))
    ax.set_ylim(*plot_ylim)

    handles = [
        Line2D(
            [0],
            [0],
            color=to_rgba(orange, line_alpha),
            lw=line_width,
            marker="o",
            markersize=marker_size,
            markerfacecolor=to_rgba(orange, marker_face_alpha),
            markeredgecolor=edge(orange),
            markeredgewidth=marker_edge_width,
            label="PSS",
        )
    ] + [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=marker_shapes[name],
            markersize=legend_marker_size(name),
            markerfacecolor=to_rgba(rule_colors[name], rule_face_alpha),
            markeredgecolor=edge(rule_colors[name]),
            markeredgewidth=marker_edge_width,
            # 归档的 verdict 列名仍是 L1--L4;图例用正文的 Set 1--Set 4
            label="Set " + name[1:],
        )
        # only the operating points actually plotted: a key entry for a marker that is not
        # on the panel invites the reader to look for it
        for name in [str(value) for value in rule_points["method"]]
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
        ncol=1,
        columnspacing=0.8,
        handlelength=1.5,
        handletextpad=0.45,
        borderaxespad=0.2,
    )
    if panel_letter:
        ax.text(
            -0.14,
            1.08,
            str(panel_letter),
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            color=ink,
            ha="left",
            va="bottom",
        )
    return {
        "plot_limits": {
            "x": [float(value) for value in plot_xlim],
            "y": [float(value) for value in plot_ylim],
        },
        "curve_points": int(len(frontier)),
        "rule_point_count": int(len(rule_points)),
    }


def render_design_figure(
    forward: pd.DataFrame,
    inverse: pd.DataFrame,
    *,
    output_dir: str | Path,
    highlight_retention: float = 0.975,
    target_retentions: Iterable[float] | None = None,
    inverse_priority_mask: Iterable[bool] | None = None,
    inverse_priority_label: str = "highest-property generated candidates retained (%)",
    score_column: str = "synthesis_score",
) -> dict[str, object]:
    """Render an inverse-design trade-off panel from forward-calibrated cutoffs."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if target_retentions is None:
        target_retentions = np.linspace(0.84, 1.0, 161)
    target_retentions = tuple(float(value) for value in target_retentions)
    calibration_mask, calibration_support = support_matched_calibration_mask(
        forward,
        inverse,
    )
    frontier = build_design_frontier(
        forward,
        inverse,
        score_column=score_column,
        target_retentions=target_retentions,
        inverse_priority_mask=inverse_priority_mask,
        experimental_calibration_mask=calibration_mask,
    )
    highlight_index = int(
        np.argmin(
            np.abs(
                frontier["target_experimental_retention"].to_numpy(float)
                - float(highlight_retention)
            )
        )
    )
    highlight = frontier.iloc[highlight_index]
    if inverse_priority_mask is None:
        priority = np.ones(len(inverse), dtype=bool)
    else:
        priority = np.asarray(tuple(inverse_priority_mask), dtype=bool)

    rule_columns = {
        "L1": "rung_L1_verdict",
        "L2": "rung_L2_verdict",
        "L3": "rung_L3_verdict",
        "L4": "rung_L4_verdict",
    }
    rule_points = pd.DataFrame(
        [
            rule_operating_point(
                forward,
                inverse,
                verdict_column=column,
                method=method,
                inverse_priority_mask=priority,
            )
            for method, column in rule_columns.items()
        ]
    )

    _figure_style()
    cm = 1 / 2.54
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(8.65 * cm, 4.65 * cm),
        facecolor="white",
    )
    panel_metadata = draw_design_panel(
        ax,
        frontier,
        rule_points,
        inverse_priority_label=inverse_priority_label,
        panel_letter="f",
    )

    experimental = forward["made"].astype(bool).to_numpy()
    cutoff = float(highlight.cutoff_score)
    forward_screened = _finite_scores(forward, score_column) < cutoff
    inverse_screened = _finite_scores(inverse, score_column) < cutoff
    experimental_retained_n = int((~forward_screened[calibration_mask]).sum())
    all_experimental_retained_n = int((~forward_screened[experimental]).sum())
    forward_theory_screened_n = int(forward_screened[~experimental].sum())
    inverse_screened_n = int(inverse_screened.sum())
    inverse_priority_retained_n = int((~inverse_screened[priority]).sum())

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "property_design_synthesis_score_candidate"
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)

    plot_data = frontier.copy()
    plot_data.to_csv(output / "plot_data.csv", index=False)
    rule_points.to_csv(output / "rule_points.csv", index=False)
    status: dict[str, object] = {
        "scope": "additive candidate figure; manuscript untouched",
        "score_label": "PRIS-derived synthesis score (PSS)",
        "calibration_support": calibration_support,
        "threshold_rule": "screen score < cutoff; retain equality ties",
        "plot_limits": panel_metadata["plot_limits"],
        "highlight": {
            "drawn_in_figure": False,
            "target_experimental_retention": float(
                highlight.target_experimental_retention
            ),
            "cutoff_score": cutoff,
            "experimental_retained_n": experimental_retained_n,
            "experimental_total_n": int(calibration_mask.sum()),
            "all_forward_experimental_retained_n": all_experimental_retained_n,
            "all_forward_experimental_total_n": int(experimental.sum()),
            "forward_theory_screened_n": forward_theory_screened_n,
            "forward_theory_total_n": int((~experimental).sum()),
            "inverse_screened_n": inverse_screened_n,
            "inverse_total_n": int(len(inverse)),
            "inverse_priority_retained_n": inverse_priority_retained_n,
            "inverse_priority_total_n": int(priority.sum()),
        },
        "outputs": {
            suffix: str(stem.with_suffix(suffix))
            for suffix in (".pdf", ".svg", ".png")
        },
    }
    (output / "STATUS.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return status
