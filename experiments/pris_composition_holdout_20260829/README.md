# PRIS unseen-chemistry sensitivity analysis

This independent experiment tests the already frozen PRIS law sets after
removing held-out structures whose exact reduced composition appeared in the
discovery partition. It also reports a stricter exact-element-set sensitivity
analysis. No law was selected or refitted, no threshold was changed, and no
manuscript file is an input or output of the workflow.

## Reproduce

From the repository root:

```bash
PYTHONPATH=. pytest -q experiments/pris_composition_holdout_20260829
PYTHONPATH=. python experiments/pris_composition_holdout_20260829/run_analysis.py
PYTHONPATH=. python experiments/pris_composition_holdout_20260829/plot_figures.py
```

The external data locations can be replaced with `--feature-root` and
`--law-root`. The analysis script fails closed if it encounters a split other
than `discovery` or `calibration`, verifies that every damaged structure
inherits its parent's split and chemistry, and refuses to emit subgroup results
unless the original held-out PRIS metrics are reproduced.

## Outputs

- `RESULTS.md`: concise interpretation and key Set 4 estimates.
- `results/metrics.csv`: all five frozen law sets, row-micro and group-equal
  estimates, and cluster-bootstrap intervals.
- `results/set4_per_damage_class.csv`: S1--S5 sensitivity results.
- `results/cohort_counts.csv`: overlap and sample-size audit.
- `results/feature_coverage.csv`: explicit coverage under the frozen
  missing-as-satisfied convention.
- `results/manifest.json`: input hashes, seed, replicate count and run metadata.
- `figures/`: two SI-ready candidates in editable PDF/SVG and 400-dpi PNG.

`Chemical system` is the exact sorted set of elements. It is deliberately not
called a chemical family: oxide/halide/prototype families would require a
separate classification fixed before evaluation.
