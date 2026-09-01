# PGHT Pre-Probe Analytic Amendment

This amendment was frozen after the first RED test was written but before the
NEXT339 module existed, before any real-structure feature evaluation, and
before any label-blind or outcome-bearing probe.

The implementation plan's Task 1 prose incorrectly states that a two-site
periodic loop with two opposite x-directed edges of conductance `g1=1` and
`g2=4` has affine retention `0.8`. Direct minimization gives

```text
min_delta g1 (1 + delta)^2 + g2 (-1 + delta)^2
    = 4 g1 g2 / (g1 + g2),

retention = [4 g1 g2 / (g1 + g2)] / (g1 + g2)
          = 4 g1 g2 / (g1 + g2)^2
          = 16/25
          = 0.64.
```

Only this analytic expected value changes. The graph, conductance `A/d`,
cell problem, affine normalization, sole feature, protected direction,
engineering/novelty gates, information boundary, and all contingent stop
rules remain exactly as in the original plan. No empirical value motivated
this correction.
