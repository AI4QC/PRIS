# NEXT151 Violation-Multiplicity Audit Freeze

Protocol: `2026-08-08-next151-violation-multiplicity-audit-v1`

Reconstruct the formally published NEXT136 global-closest candidate identified
by candidate-key SHA-256
`44b9eabae5e1ff3014ef4746758bbc3a79a4f193bad94507dd17c7db0edd1919`,
with SAFE threshold `3.4014264642057306` and closest BROAD threshold
`0.8669460357541353`.

The primary audit population is the fixed discovery score shell

```text
0.8669460357541353 <= published_score < 3.4014264642057306
```

restricted to protected (`endpoint <= 1`) and severe (`endpoint >= 2`)
structures. The secondary population is every supported protected/severe
discovery extreme.

For each structure, reconstruct the 11 nonnegative weighted physical-risk
contributions of the candidate's base terms before coordination and packing
protections. Audit these fixed aggregations:

- `sum_all`, `max_one`, `second_largest`, `third_largest`;
- `top2_sum`, `top3_sum`, `sum_minus_max`, `sum_minus_top2`;
- `effective_violation_count = sum_all^2 / sum(contribution^2)`;
- `max_fraction = max_one / sum_all`;
- counts of contributions strictly above `0.1`, `0.25`, `0.5`, and `1.0`.

The hypothesis fixes the protected-class AUC direction before outcomes:
`max_fraction` is `+1`; every other aggregation is `-1`. Evaluate pooled,
macro-fold, and worst-fold AUC in the SCIGEN shell, then apply the same fixed
direction to the WyFormer shell and both full-extreme populations.

An aggregation is eligible for a later frozen law search only if all of the
following hold:

1. SCIGEN shell worst-fold AUC is at least `0.55`;
2. WyFormer shell pooled AUC is at least `0.55`;
3. SCIGEN and WyFormer full-extreme pooled AUC are each at least `0.50`;
4. all five SCIGEN shell folds are evaluable;
5. the aggregation uses no endpoint, source, lattice-class, or dataset field
   at execution time.

Rank eligible aggregations by the minimum of those four AUC quantities, then
their mean, then lexical name. This audit searches no formula and changes no
threshold or gate.

Discovery endpoints are offline labels only. Validation and replication stay
unopened. No DFT calculation or DFT value, learned energy/force/stress proxy,
or physical relaxation is used.
