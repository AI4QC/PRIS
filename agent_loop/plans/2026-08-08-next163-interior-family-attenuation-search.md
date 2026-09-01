# NEXT163 interior family-attenuation search freeze

## Boundary and motivation

The frozen `gamma=0` family-capmean score in NEXT158 produced three candidates
that pass source AUC and SAFE12. The independently frozen `gamma=0.1` endpoint
in NEXT162 retained 88 source-AUC passes but its best candidate passed only
11/12 SAFE cells, failing solely WyFormer fold1 severe-precision lower bound.
NEXT163 searches four predeclared interior points between these two completed
endpoints. No point outside the interval is admitted.

This is an additive pre-DFT discovery search. Formula execution uses only
frozen analytic structure/composition features and assigned formal oxidation
states. It performs no DFT calculation, reads no DFT value, uses no learned
energy/force/stress proxy, and performs no relaxation. Discovery outcomes are
offline labels only. Validation and replication endpoints remain sealed; no
prior or canonical artifact is replaced.

## Frozen score and candidate universe

For weighted nonnegative contributions `c_i = w_i R_i` and the four frozen
mechanism families, define

```text
M_f(x)       = mean_{i in I_f} min(c_i(x), 0.5)
A_gamma(x)   = sum_f M_f(x) - gamma max_f M_f(x)
S_gamma(x)   = max(0, A_gamma(x)
                       - alpha P_coord(x)
                       - beta P_coord-pack(x))
```

The interior grid is fixed as:

- `gamma in {0.01, 0.025, 0.05, 0.075}`;
- 11 exact NEXT132 bases;
- `alpha in {0, 0.5, 1, 2}`;
- `beta in {0, 0.1, 0.25, 0.5}`;
- 704 candidates total;
- base-formula SHA-256:
  `d1f8763331cbe36f54e898e4efc88d0f88d2ae5d6284883acc4850e58d9678b5`;
- candidate-key SHA-256 over newline-joined sorted canonical JSON keys:
  `d34508ca8fb03cfc4bd1d59c946fca2bdb3b72ef14e3653029993e0c29f349d5`.

The contribution cap, family mapping, reductions, protections, and all grid
points are frozen. No additional gamma, base, term weight, protection, threshold
rule, gate, or ordering may be added after outcomes open.

## Endpoint and evaluation checks

Before publishing, reconstruct the completed NEXT158 `gamma=0` and NEXT162
`gamma=0.1` endpoint candidate universes and verify their candidate records are
reproduced to the same absolute `1e-12` metric tolerance when evaluated by the
unchanged grouped-fold cross-source evaluator. Then evaluate all 704 interior
candidates using the same source-AUC, SAFE12, BROAD, and deterministic selection
rules.

If no candidate passes every discovery gate, terminate the interior
attenuation branch and keep validation/replication sealed. If a candidate
passes, freeze its full formula and threshold before any separate one-shot
validation. NEXT163 itself opens no validation/replication output and makes no
scientific improvement claim.

## Outputs

Atomically publish a manifest, catalogue, discovery evaluation, and complete
candidate table under
`$PRIS_ARCHIVE/next163_interior_family_attenuation_search_v1`.
Only the standalone research report may be updated after the run.
