# Second pre-outcome amendment to NEXT279--NEXT282

**Status:** frozen during analytic test design, before any formal NEXT279
feature or discovery outcome was computed or opened.

The parent plan and first amendment remain unchanged. The classical finite-
sample Geary factor `(N - 1) / (2 W)` is not invariant when an identical
periodic structure is represented by an integral supercell: all site, edge,
and squared-difference sums replicate, but `N - 1` does not. NEXT279 therefore
uses the periodic-population normalization

```text
PeriodicGeary(y) = (N / (2 W))
                   * sum_(i,j in E) (x_i - x_j)^2 / sum_i x_i^2.
```

This is an engineering invariance correction, not an outcome-driven direction
change. The feature names, eight frozen directions, active-facet contacts,
Moran and absolute-Moran formulas, extreme-edge statistic, quantiles, gates,
search grid, diagnostic rank, and stopping rules are unchanged.

No discovery label, endpoint, validation output, replication output, DFT
quantity, proxy-potential result, or relaxed structure informed this amendment.
