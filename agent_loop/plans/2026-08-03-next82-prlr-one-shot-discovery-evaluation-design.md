# NEXT82 one-shot PRLR discovery evaluation

## Frozen input

Evaluate exactly the NEXT81 v2 formula and predictions:

```text
prlr_risk = prlr_residual_fraction * log1p(prlr_contact_weight_rms)
REJECT iff supported and prlr_risk >= 3.4573392313463684
missing or unsupported -> KEEP
```

The threshold is the unlabeled discovery-x0 inverted-CDF 95th percentile.  The
formula SHA-256 is
`bc9f070e6cec67973fc95cf8e105d0e988119762c768d4e59102c1c05ade791c`;
the all-partition prediction SHA-256 is
`11b1851612f02fa21c53f4ecc21b255a45e967411b68c13eece574eba96622f8`.
No formula term, score, threshold, support rule, or prediction may change.

## Evaluation

Read only the physically isolated robust discovery endpoint labels and merge by
`material_id`.  Reuse the exact protected/severe definitions, Wilson intervals,
three frozen strata, seven gates, and AUC implementation from NEXT57--NEXT79.
Record every row; do not silently drop unsupported predictions.

The original seven gates are necessary but not sufficient.  Because the source
cohort has supported prior discovery iterations, advancement additionally
requires a discovery reject-precision Wilson lower bound of at least 0.80.
Only if all conditions pass may a separate protocol open internal replication.
Otherwise this independent hypothesis stops and replication remains unopened.

## Boundary

The executable rule is raw-x0-only and contains no DFT calculation/value,
relaxed geometry, learned energy/force/stress proxy, physical relaxation, or
same-composition alternative.  Existing ODAC23 relaxed-coordinate information
appears only in the offline discovery endpoint used after predictions were
frozen; it is not an executable input.
