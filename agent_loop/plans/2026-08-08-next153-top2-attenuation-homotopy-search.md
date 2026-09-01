# NEXT153 top-two attenuation homotopy search freeze

## Boundary

This is an additive, discovery-only, pre-DFT experiment.  The executable score
uses only the already frozen analytic structure/composition features and formal
oxidation states.  It uses no DFT calculation or value, no learned
energy/force/stress proxy, and no relaxation.  Internal-validation and
replication endpoints remain sealed.  No prior script, result, report, paper,
README, preregistration, note, or TeX source is replaced.

NEXT152 showed that complete removal of the two largest contributions can pass
SAFE12 but misses the WyFormer pooled-AUC gate.  NEXT153 therefore tests only a
pre-frozen one-dimensional interpolation between the existing summed law and
the fully trimmed law.  No new feature or residual-derived exception is added.

## Frozen score family

For weighted nonnegative physical contributions `c_i = w_i R_i`, let `c_(1)`
and `c_(2)` be the two largest values.  For attenuation `gamma`, define

```text
T_gamma = sum_i c_i - gamma * (c_(1) + c_(2))
alpha_gamma = 2 * (1 - gamma)
S_gamma = max(0,
              T_gamma
              - alpha_gamma * P_coord
              - 0.5 * P_coord-pack)
```

`P_coord` is the bounded NEXT129 coordination protection and `P_coord-pack` is
the bounded NEXT135 coordination-by-covalent-packing product.  Unsupported
protection is inactive and cannot enlarge base support.

The only free parameter is
`gamma in {0, 0.1, 0.25, 0.5, 0.75, 1}`.  `gamma=0` is required to reproduce
the corresponding NEXT135 `coordination=2, packing-product=0.5` candidate.
`gamma=1` is required to reproduce the corresponding NEXT152
`coordination=0, packing-product=0.5` candidate.  These endpoint reproductions
are hard execution checks, not post-result comparisons.

## Frozen universe and gates

- Exact NEXT132-selected bases: 11.
- Attenuation values per base: 6.
- Candidate count: 66.
- Base-formula SHA-256:
  `d1f8763331cbe36f54e898e4efc88d0f88d2ae5d6284883acc4850e58d9678b5`.
- Candidate-key SHA-256 over newline-joined sorted canonical JSON keys:
  `6a7047514b4926a514b9fdb39c55ab594009444c03f3cc331201aeaec4dd8c31`.

Use the unchanged grouped-fold cross-source discovery evaluator, threshold
search, source-AUC gates, SAFE12 gates, BROAD gates, and deterministic candidate
ordering.  No gamma value, weight, threshold, formula, or selection rule may be
changed after NEXT153 discovery results are opened.

If no candidate passes every discovery gate, terminate this homotopy branch and
keep validation/replication sealed.  If a candidate passes, freeze its formula
and threshold before any separate one-shot validation.  NEXT153 itself never
opens validation or replication data and makes no scientific improvement claim.

## Outputs

Atomically publish a manifest, frozen catalogue, discovery evaluation, and full
candidate table under
`$PRIS_ARCHIVE/next153_top2_attenuation_homotopy_search_v1`.
Only the standalone research report may be updated after the run.
