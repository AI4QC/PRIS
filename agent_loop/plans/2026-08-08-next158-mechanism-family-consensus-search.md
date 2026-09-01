# NEXT158 mechanism-family consensus search freeze

## Boundary

This is an additive pre-DFT discovery search. Its executable score uses only
the frozen analytic structure/composition features and assigned formal
oxidation states. It performs no DFT calculation, reads no DFT value at formula
execution time, uses no learned energy/force/stress proxy, and performs no
relaxation. Discovery outcomes are offline labels only. Internal-validation
and replication endpoints remain sealed, and no prior or canonical artifact is
replaced.

NEXT157 admitted three consensus statistics and selected exactly the top-ranked
one, `family_capped_mean_sum`, for this search. The lower-ranked eligible
statistics `family_max_sum_minus_largest` and `family_max_second`, plus every
ineligible statistic, are excluded.

## Frozen physical families

Every weighted nonnegative base contribution `c_i = w_i R_i` is assigned by
its frozen term-ID prefix to exactly one family:

- local geometry: `cov_`, `scbv_`, `sivr_`;
- charge-flow feasibility: `cmvo_`, `hcid_`;
- valence transport: `bvtbd_`, `bvtc_`;
- contact robustness: `mhcr_`.

Each of the 11 exact NEXT132-selected bases contains all four families. Family
membership is fixed before the search and is not learned from endpoints.

## Frozen score

For family `f`, let `I_f` be its contribution indices and define

```text
M_f(x) = mean_{i in I_f} min(c_i(x), 0.5)
F(x)   = sum_f M_f(x)
S(x)   = max(0, F(x) - alpha P_coord(x) - beta P_coord-pack(x))
```

`P_coord` is the bounded NEXT129 coordination protection and `P_coord-pack` is
the bounded NEXT135 coordination-by-covalent-packing product. Unsupported
protections are inactive and cannot enlarge the intersection support of the
physical base terms.

The per-contribution cap `0.5`, the four families, and the within-family mean
are fixed and are not search parameters. The candidate grid is:

- 11 exact NEXT132 bases;
- `alpha in {0, 0.5, 1, 2}`;
- `beta in {0, 0.1, 0.25, 0.5}`;
- 176 candidates total;
- base-formula SHA-256:
  `d1f8763331cbe36f54e898e4efc88d0f88d2ae5d6284883acc4850e58d9678b5`;
- candidate-key SHA-256 over newline-joined sorted canonical JSON keys:
  `5f7697c38cd895267717c8d4779ed9c395de4c9cecdb814d70eb7a31ccd94483`.

No cap, family, prefix, averaging rule, base term, base weight, protection,
grid point, threshold rule, gate, or candidate-ordering rule may be added or
changed after results are opened.

## Evaluation

Use the unchanged grouped-fold cross-source discovery evaluator, source-AUC
gates, SAFE12 gates, BROAD gates, and deterministic selection order. Prove
before publication that zero protection exactly reproduces the frozen family
statistic, protections only subtract on active rows, base support is unchanged,
all terms map to exactly one represented family, and all 176 identities occur
exactly once.

If no candidate passes every discovery gate, terminate this mechanism-family
consensus branch and keep validation/replication sealed. If a candidate passes,
freeze its complete formula and threshold before any separate one-shot
validation. NEXT158 itself never opens validation or replication data and
makes no scientific improvement claim.

## Outputs

Atomically publish a manifest, catalogue, discovery evaluation, and complete
candidate table under
`$PRIS_ARCHIVE/next158_mechanism_family_consensus_search_v1`.
Only the standalone research report may be updated after the run.
