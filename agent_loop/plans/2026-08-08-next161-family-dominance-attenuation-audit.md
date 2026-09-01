# NEXT161 family-dominance attenuation audit freeze

## Boundary and motivation

NEXT158 showed that the capped within-family mean summed across four physical
families can pass both source-AUC gates and SAFE12, but NEXT159 localized its
BROAD residual to SCIGEN protected structures rejected at an aggressive
threshold. NEXT160 then showed that hard concurrence statistics (the second
largest family or deleting the largest raw-family maximum) lose too much
WyFormer discrimination. NEXT161 therefore audits a smooth, family-level
homotopy between these regimes.

This additive audit searches no operational threshold or deployable formula.
It uses discovery outcomes only as offline labels, opens no validation or
replication endpoint, performs no DFT calculation, reads no DFT value at
formula execution time, uses no learned energy/force/stress proxy, and performs
no relaxation. No prior or canonical artifact is replaced.

## Frozen base and statistics

Use the deterministic global-closest NEXT159 candidate, whose canonical key has
SHA-256
`8bde10516eaf06a8a933b1595ef0e6256f8405d3caecd4de670620b0da90cfe4`.
Only its weighted nonnegative physical contributions are used; its operational
packing protection and thresholds are not used to define the audit statistics.

Assign contributions to the same four frozen families as NEXT157. For family
`f`, define

```text
M_f(x) = mean_{i in I_f} min(c_i(x), 0.5)
A_gamma(x) = sum_f M_f(x) - gamma max_f M_f(x)
```

Audit exactly five nonzero attenuation values:

```text
gamma in {0.1, 0.25, 0.5, 0.75, 1.0}
```

The known failed endpoint `gamma=0` is excluded from selection and may not be
reintroduced. Every statistic has frozen direction `-1` for protected-vs-severe
AUC (lower is more protected). No additional gamma, cap, family transform, or
direction may be added after outcomes open.

## Frozen populations and gates

Reconstruct the exact NEXT159 base support and the published NEXT135 score used
to define the difficult shell. Evaluate each statistic on:

- SCIGEN shell: supported extremes with published score in
  `[BROAD_THRESHOLD, SAFE_THRESHOLD)`;
- WyFormer shell: the same definition;
- SCIGEN full supported extremes;
- WyFormer full supported extremes.

Use formula-group five-fold AUCs. A statistic is eligible only if:

- SCIGEN shell worst-fold AUC >= 0.55 with all five folds evaluable;
- WyFormer shell pooled AUC >= 0.55;
- SCIGEN full pooled AUC >= 0.50;
- WyFormer full pooled AUC >= 0.50.

Rank eligible statistics by descending minimum of those four AUCs, then
descending mean, then ascending statistic name. At most the single top-ranked
eligible gamma may enter a separately frozen search. If none is eligible, the
family-dominance attenuation branch terminates.

## Outputs

Atomically publish a manifest, JSON audit, and complete parquet table under
`$PRIS_ARCHIVE/next161_family_dominance_attenuation_audit_v1`.
Only the standalone research report may be updated after the run.
