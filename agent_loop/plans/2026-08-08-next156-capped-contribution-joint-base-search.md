# NEXT156 capped-contribution joint-base search freeze

## Boundary

This is an additive pre-DFT discovery search.  Its executable score uses only
the frozen analytic structure/composition features and assigned formal
oxidation states.  It performs no DFT calculation, reads no DFT value at formula
execution time, uses no learned energy/force/stress proxy, and performs no
relaxation.  Discovery outcomes are offline labels only.  Internal-validation
and replication endpoints remain sealed, and no prior or canonical artifact is
replaced.

NEXT155 admitted exactly one top-ranked robust aggregation to this search:
`sum_clip_0p5`.  The eligible but lower-ranked `sum_clip_0p25` and every
ineligible statistic are excluded.

## Frozen score

For each exact NEXT132-selected base with weighted nonnegative physical
contributions `c_i = w_i R_i`, define

```text
C(x) = sum_i min(c_i(x), 0.5)
S(x) = max(0, C(x) - alpha P_coord(x) - beta P_coord-pack(x))
```

`P_coord` is the bounded NEXT129 coordination protection and `P_coord-pack` is
the bounded NEXT135 coordination-by-covalent-packing product.  Unsupported
protections are inactive and cannot enlarge the intersection support of the
physical base terms.

The cap `0.5` is fixed and is not a search parameter.  The candidate grid is:

- 11 exact NEXT132 bases;
- `alpha in {0, 0.5, 1, 2}`;
- `beta in {0, 0.1, 0.25, 0.5}`;
- 176 candidates total;
- base-formula SHA-256:
  `d1f8763331cbe36f54e898e4efc88d0f88d2ae5d6284883acc4850e58d9678b5`;
- candidate-key SHA-256 over newline-joined sorted canonical JSON keys:
  `8b9f7fc423970dd5c6713cca4d79b11b1bb0d223d1b90976da40b0ccfe48c235`.

No cap, base term, base weight, protection, grid point, threshold rule, gate,
or candidate-ordering rule may be added or changed after results are opened.

## Evaluation

Use the unchanged grouped-fold cross-source discovery evaluator, source-AUC
gates, SAFE12 gates, BROAD gates, and deterministic selection order.  Prove
before publication that zero protection is exactly the capped sum, protections
only subtract on active rows, base support is unchanged, and all 176 identities
occur exactly once.

If no candidate passes every discovery gate, terminate this capped-contribution
branch and keep validation/replication sealed.  If a candidate passes, freeze
its complete formula and threshold before any separate one-shot validation.
NEXT156 itself never opens validation or replication data and makes no
scientific improvement claim.

## Outputs

Atomically publish a manifest, catalogue, discovery evaluation, and complete
candidate table under
`$PRIS_ARCHIVE/next156_capped_contribution_joint_base_search_v1`.
Only the standalone research report may be updated after the run.
