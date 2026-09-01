# NEXT559--NEXT561 HEA entropy-packing law

## Discovery provenance

All 2,400 NEXT551 endpoints are now open after NEXT553/NEXT558.  Exploratory
composition descriptors showed that the coefficient-free probabilistic union
of x0 packing deficit and high ideal composition entropy is robust across the
two formerly separated halves: AUC 0.756 overall, 0.756/0.754 by old partition,
and 0.803/0.721 for ordered/SQS structures.  This is opened-data discovery,
not confirmation.

## Frozen formula

From composition fractions `x_i`, compute ideal entropy
`H=-sum_i x_i ln(x_i)`.  From the raw x0 cell compute covalent packing fraction
`phi=sum_i 4*pi*r_cov_i^3/(3 V)`.  Within a frozen candidate batch, compute
midrank percentiles `u_H=percentile(H)` and `u_phi=percentile(-phi)`, then

`EPCU = 1 - (1-u_H)(1-u_phi)`.

There are no weights, fitted coefficients, DFT inputs, model potentials, or
coordinate changes.  Reject the highest-risk 15%, breaking ties by FID.

## NEXT560 new endpoint-sealed cohort

Exclude all 2,400 NEXT551 FIDs.  Every eligible remaining row from a chemical
system absent from NEXT551 is included (frozen source count: 425 rows from 40
systems).  From chemical systems present in NEXT551, add the lowest
`SHA256("NEXT560-known-v1|"+fid)` 800 ordered and 775 SQS rows.  The result is
2,000 new identities: 425 unseen-system and 1,575 known-system replication
rows.  Decode only initial structures, publish sanitized geometry, and freeze
EPCU before any new endpoint is opened.

## NEXT561 one-time confirmation

Open endpoints only for NEXT560 FIDs and apply the absolute NEXT555 extreme
waste definition without recalibration.  Require at least 100 rows per class,
at least 30 rows per class in the unseen-system subset, and at least 50
protected rows.  Confirmation gates are: score coverage at least 0.99; overall
AUC at least 0.72 with chemical-system cluster lower bound at least 0.68;
unseen-system AUC at least 0.68; known-system replication AUC at least 0.72;
ordered and SQS AUC at least 0.65; Spearman severity at least 0.35 with cluster
lower bound at least 0.25; top-15% lift at least 1.50 with protected fraction at
most 0.02; and overall AUC at least 0.03 above each frozen component.

Pauling controls are computed independently.  If at least 100 rows are
supported, EPCU must beat Pauling AUC by 0.05 on common support; otherwise EPCU
must cover at least 99% while Pauling covers at most 25%.

Passing authorizes an independent HEA-domain report only.  It does not
authorize canonical report/paper edits or a universal DFT-replacement claim.

