# NEXT162 family-dominance attenuation search freeze

## Boundary

NEXT161 admitted four nonzero family-dominance attenuations and selected exactly
the top-ranked `gamma=0.1`. NEXT162 searches that value only. The eligible but
lower-ranked `gamma in {0.25, 0.5, 0.75}`, the ineligible `gamma=1`, and the
known-failed `gamma=0` endpoint are excluded.

This is an additive pre-DFT discovery search. Formula execution uses only
frozen analytic structure/composition features and assigned formal oxidation
states. It performs no DFT calculation, reads no DFT value, uses no learned
energy/force/stress proxy, and performs no relaxation. Discovery outcomes are
offline labels only. Validation and replication endpoints remain sealed, and
no prior or canonical artifact is replaced.

## Frozen score

Assign each weighted nonnegative base contribution `c_i = w_i R_i` to the four
frozen NEXT157 mechanism families. Define

```text
M_f(x) = mean_{i in I_f} min(c_i(x), 0.5)
A(x)   = sum_f M_f(x) - 0.1 max_f M_f(x)
S(x)   = max(0, A(x) - alpha P_coord(x) - beta P_coord-pack(x))
```

`P_coord` is the bounded NEXT129 coordination protection and `P_coord-pack` is
the bounded NEXT135 coordination-by-covalent-packing product. Unsupported
protections are inactive and cannot enlarge physical-base support.

The cap `0.5`, attenuation `0.1`, family mapping, within-family mean, and
dominant-family maximum are fixed rather than searched.

## Frozen candidate universe

- 11 exact NEXT132-selected bases;
- `alpha in {0, 0.5, 1, 2}`;
- `beta in {0, 0.1, 0.25, 0.5}`;
- 176 candidates total;
- base-formula SHA-256:
  `d1f8763331cbe36f54e898e4efc88d0f88d2ae5d6284883acc4850e58d9678b5`;
- candidate-key SHA-256 over newline-joined sorted canonical JSON keys:
  `f5ad03d87fa11a06aee0c0aec07eb8a70848353497f8305f9003e5056d7823aa`.

No cap, attenuation, family, prefix, base, term weight, protection, grid point,
threshold rule, gate, or candidate ordering may change after outcomes open.

## Evaluation and stopping rule

Use the unchanged grouped-fold cross-source discovery evaluator, source-AUC
gates, SAFE12 gates, BROAD gates, and deterministic selection. Verify exact
zero-protection aggregation, unchanged support, subtractive-only active
protections, unique family assignment, and all 176 identities before
publication.

If no candidate passes every discovery gate, terminate this attenuation branch
and keep validation/replication sealed. If a candidate passes, freeze its full
formula and threshold before any separate one-shot validation. NEXT162 itself
never opens validation/replication outputs and makes no improvement claim.

## Outputs

Atomically publish a manifest, catalogue, discovery evaluation, and complete
candidate table under
`$PRIS_ARCHIVE/next162_family_dominance_attenuation_search_v1`.
Only the standalone research report may be updated after the run.
