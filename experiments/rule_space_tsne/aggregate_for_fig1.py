#!/usr/bin/env python3
"""Aggregate the 2,047,123-candidate t-SNE map into committable figure data.

The repository convention is that plotting code reads paper/data/*.csv and does no
computation. The raw projection (out/tsne.csv, 2.05M rows) is too large to commit, so
this script reduces it to a density grid plus the eight surviving rules' positions.

Writes:
  paper/data/fig1_rulespace_grid.csv   x_bin, y_bin, n   (non-empty bins only)
  paper/data/fig1_rulespace_stars.csv  rule, x, y, feature, threshold
"""
import json
import pathlib
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
DATA = HERE.parent.parent / "paper" / "data"
NBIN = 220


def main():
    d = pd.read_csv(OUT / "tsne.csv")
    stars = json.load(open(OUT / "stars.json"))
    print(f"read {len(d):,} candidates")

    lo = min(d.x.min(), d.y.min())
    hi = max(d.x.max(), d.y.max())
    edges = np.linspace(lo, hi, NBIN + 1)
    h, _, _ = np.histogram2d(d.x.values, d.y.values, bins=[edges, edges])
    centres = 0.5 * (edges[:-1] + edges[1:])
    ix, iy = np.nonzero(h)
    grid = pd.DataFrame({"x": centres[ix].round(4),
                         "y": centres[iy].round(4),
                         "n": h[ix, iy].astype(int)})
    grid.to_csv(DATA / "fig1_rulespace_grid.csv", index=False)
    print(f"grid: {len(grid):,} non-empty bins of {NBIN}x{NBIN}, "
          f"max {grid.n.max():,}, total {grid.n.sum():,}")

    ids = {v["rule_id"]: k for k, v in stars.items()}
    s = d[d.rule_id.isin(ids)].copy()
    s["rule"] = s.rule_id.map(ids)
    s["feature"] = s.rule.map(lambda r: stars[r]["feature"])
    s["threshold"] = s.rule.map(lambda r: stars[r]["target_threshold"])
    s = s.sort_values("rule")[["rule", "x", "y", "feature", "threshold"]]
    s.to_csv(DATA / "fig1_rulespace_stars.csv", index=False)
    print(f"stars: {len(s)} rows\n{s.to_string(index=False)}")

    # bin-level extent, so the figure can set limits without the raw file
    meta = dict(n_candidates=int(len(d)), nbin=NBIN,
                lo=float(lo), hi=float(hi),
                x_range=[float(d.x.min()), float(d.x.max())],
                y_range=[float(d.y.min()), float(d.y.max())])
    json.dump(meta, open(DATA / "fig1_rulespace_meta.json", "w"), indent=1)
    print("wrote meta")


if __name__ == "__main__":
    main()
