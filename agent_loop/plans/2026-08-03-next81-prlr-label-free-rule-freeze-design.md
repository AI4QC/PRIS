# NEXT81 label-free PRLR rule freeze

## Frozen operation

Read only the immutable NEXT80 raw-x0 feature artifact.  Select rows whose
`partition_role` is `discovery`, require finite `prlr_risk` and
`repulsive_load_supported=true`, and set the one permitted threshold to the
inverted-CDF 95th percentile of that unlabeled score distribution.  No endpoint
or relaxed-coordinate file is an input.

Publish one direct rule:

```text
PRLR_RISK(x0) = prlr_residual_fraction * log1p(prlr_contact_weight_rms)
REJECT iff supported and PRLR_RISK >= frozen discovery-x0 q95
missing or unsupported -> KEEP
```

Apply the frozen threshold to all 7,815 x0 feature rows and hash the rule and
predictions before any endpoint labels are read.  Ties at the threshold are all
rejected.  There is no feature, sign, coefficient, rejection fraction, or
threshold search.

## Prospective gates

NEXT82 may read robust discovery labels only after NEXT81 publication.  It must
use the unchanged seven ODAC23 gates and additionally require a reject-precision
Wilson lower bound of at least 0.80 before internal replication can be opened.
Failure stops this independent hypothesis.

## Boundary

The rule depends only on one raw unrelaxed x0, frozen radii, deterministic
periodic graphs, analytic repulsive weights, and sparse linear algebra.  It uses
no DFT calculation or value, relaxed geometry, learned energy/force/stress
proxy, physical relaxation, or same-composition alternative.
