# NEXT155 robust contribution-transform audit freeze

## Boundary and hypothesis

This additive audit searches no formula and opens no internal-validation or
replication endpoint.  Discovery outcomes are offline labels only.  Every
audited quantity is computed from the 11 frozen nonnegative analytic physical
contributions of the exact NEXT136 closest candidate.  No DFT calculation or
value, learned energy/force/stress proxy, or relaxation is permitted.

NEXT154 terminated top-two attenuation because it did not reduce the BROAD
residual.  The distinct hypothesis here is that benign structures contain
large but saturable violations across one or more mechanisms, while severe
structures accumulate moderate violations.  Smooth or fixed-scale saturation
may preserve distributed evidence without allowing any term to dominate.

## Frozen statistics

For weighted contributions `c_i >= 0`, audit exactly these structure-level
statistics, all with the predeclared direction “lower means protected”:

```text
sum_all                 = sum c_i                 (control)
sum_sqrt                = sum sqrt(c_i)
sum_log1p               = sum log(1 + c_i)
sum_tanh                = sum tanh(c_i)
sum_rational            = sum c_i / (1 + c_i)
sum_clip_0p25           = sum min(c_i, 0.25)
sum_clip_0p5            = sum min(c_i, 0.5)
sum_clip_1              = sum min(c_i, 1)
sum_clip_2              = sum min(c_i, 2)
```

No scale, cap, statistic, direction, population, gate, or ranking rule may be
added or changed after the audit labels are opened.

## Frozen populations and eligibility

Reconstruct the candidate and its published score exactly.  Use the same
extremes (`protected endpoint <= 1`, `severe endpoint >= 2`) and the same score
shell bounded by the frozen NEXT136 closest BROAD threshold and SAFE threshold:

```text
0.8669460357541353 <= published score < 3.4014264642057306
```

Evaluate each statistic on SCIGEN shell, WyFormer shell, SCIGEN full extremes,
and WyFormer full extremes with the existing reduced-formula five-fold split.
Eligibility requires all of:

- SCIGEN shell worst-fold AUC >= 0.55 with all five folds evaluable;
- WyFormer shell pooled AUC >= 0.55;
- SCIGEN full pooled AUC >= 0.50; and
- WyFormer full pooled AUC >= 0.50.

Rank eligible statistics by minimum of those four AUCs, then their mean, then
name.  If none is eligible, terminate the saturation branch.  If one or more is
eligible, only the top-ranked statistic may enter a separately frozen formula
search.  NEXT155 itself never searches a formula, authorizes validation, or
makes a scientific improvement claim.

## Outputs

Atomically publish a manifest, JSON audit, and full ranking parquet under
`$PRIS_ARCHIVE/next155_robust_contribution_transform_audit_v1`.
Only the standalone report may be updated afterward.
