# NEXT175 Weighted-Rigidity Repair-Width Search

## Authorization

NEXT174 found five graph-weighted CrystalNN rigidity hypotheses eligible under
all frozen cross-source repair-shell gates, with every key AUC higher than the
corresponding unweighted NEXT169 feature. This authorizes one separately
frozen formula family. It does not authorize validation or a scientific claim.

## Frozen formula

Let

```text
B = 0.21976295573076796
S = 0.5415470292150686
W = S - B
active = supported_feature and B <= base_score < S
score' = max(0, base_score - alpha * W * weighted_local_rigidity)  if active
score' = base_score                                                   otherwise
```

The interval is evaluated once on the original NEXT163 base score. Rows below
`B`, rows at or above `S`, and rows with a missing feature are exactly
unchanged. Support is exactly the base support. `W` is the already-frozen
repair-shell width, not an outcome-fitted scale.

Eligible features, in frozen order:

1. `pwldr_crystalnn_tightness_min`;
2. `pwldr_crystalnn_tightness_q10`;
3. `pwldr_crystalnn_tightness_mean`;
4. `pwldr_crystalnn_volume_q10`;
5. `pwldr_crystalnn_volume_mean`.

Frozen attenuation grid:

```text
alpha in {0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00}
```

The exact universe is one unchanged base plus five features by eight alpha
values: 41 candidates. No feature cutoff, second feature, source-specific
parameter, alternative interval, threshold change, or graph-mode choice is
searched.

## Gates and termination

- Reproduce the published NEXT163 base record to absolute tolerance `1e-12`.
- Evaluate the same source-AUC, SAFE12, and BROAD gates as prior searches.
- A candidate is successful only if all gates pass without rounding.
- If none passes, terminate this operator family. Do not extend alpha, change
  the interval, add a cutoff, or append a second feature on these outcomes.
- Even if discovery succeeds, keep validation and replication sealed and write
  only the standalone report before requesting user confirmation.

## Boundaries and outputs

- The executable law uses only initial structure, fixed analytic graph weights,
  and existing pre-DFT analytic terms.
- Discovery outcomes are offline labels only.
- No DFT calculation/value, learned energy/force/stress proxy, or relaxation.
- The runner has no validation or replication path.
- Preserve all prior scripts and artifacts.

Publish atomically under
`$PRIS_ARCHIVE/next175_weighted_rigidity_repair_width_search_v1`.
