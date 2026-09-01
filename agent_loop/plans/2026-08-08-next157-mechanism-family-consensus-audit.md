# NEXT157 mechanism-family consensus audit freeze

## Boundary and motivation

This additive audit searches no formula and opens no internal-validation or
replication endpoint.  Discovery outcomes are offline labels only.  Every
quantity is computed from the frozen nonnegative analytic contributions of the
exact NEXT136 closest candidate.  No DFT calculation or value, learned
energy/force/stress proxy, or relaxation is permitted.

NEXT151--156 show that operating calibration is not fixed by deleting the two
largest terms or by capping every term equally.  The new hypothesis is that
several correlated terms from one physical mechanism should not count like
independent evidence.  Severe structures should instead be identified by
agreement across distinct mechanism families.

## Frozen families

Assign the frozen physical terms by prefix to exactly four families:

1. `local_geometry`: `cov_*`, `scbv_*`, `sivr_*`;
2. `charge_flow_feasibility`: `cmvo_*`, `hcid_*`;
3. `valence_transport`: `bvtbd_*`, `bvtc_*`;
4. `contact_robustness`: `mhcr_*`.

Every contribution must belong to exactly one family and all four families
must be represented.  Family definitions cannot be changed after labels are
opened.

For weighted contributions `c_i >= 0`, define each family's row-wise mean and
maximum.  Audit exactly these statistics, all with the predeclared direction
“lower means protected”:

```text
family_mean_sum                 = sum_f mean_f(c)
family_max_sum                  = sum_f max_f(c)
family_max_second               = second-largest_f max_f(c)
family_max_third                = third-largest_f max_f(c)
family_max_sum_minus_largest    = sum_f max_f(c) - largest_f max_f(c)
family_max_geomean1p            = exp(mean_f log(1 + max_f(c))) - 1
family_active_count_0p25        = count_f[max_f(c) > 0.25]
family_active_count_0p5         = count_f[max_f(c) > 0.5]
family_capped_mean_sum          = sum_f mean_f(min(c, 0.5))
family_rational_mean_sum        = sum_f mean_f(c / (1 + c))
```

No additional family, statistic, scale, threshold, direction, or ranking rule
may be introduced after the audit starts.

## Frozen populations and eligibility

Reconstruct the exact NEXT136 candidate and published score.  Use protected
endpoint `<= 1`, severe endpoint `>= 2`, the reduced-formula five-fold split,
and the fixed shell

```text
0.8669460357541353 <= published score < 3.4014264642057306
```

Evaluate SCIGEN shell, WyFormer shell, SCIGEN full extremes, and WyFormer full
extremes.  Eligibility requires SCIGEN shell worst-fold AUC `>= 0.55` with all
five folds evaluable, WyFormer shell pooled AUC `>= 0.55`, and both full-source
pooled AUCs `>= 0.50`.  Rank eligible statistics by the minimum of those four
AUCs, then their mean, then statistic name.

If none is eligible, terminate the mechanism-family branch.  If one or more is
eligible, only the top-ranked statistic may enter a separately frozen law
search.  NEXT157 itself never searches a formula, authorizes validation, or
makes a scientific improvement claim.

## Outputs

Atomically publish a manifest, JSON audit, and complete ranking parquet under
`$PRIS_ARCHIVE/next157_mechanism_family_consensus_audit_v1`.
Only the standalone research report may be updated afterward.
