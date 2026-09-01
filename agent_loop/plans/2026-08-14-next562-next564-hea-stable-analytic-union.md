# NEXT562--NEXT564 stable HEA analytic-union search

## Status boundary

NEXT561 opened the endpoints of its 2,000 authorized identities and rejected
EPCU: overall AUC was 0.7137 and the chemical-system cluster lower bound was
0.6446.  All NEXT551 and NEXT560 rows are therefore development data.  No
claim is carried forward from EPCU.

## NEXT562 bounded opened-data search

Recompute the fixed 82-feature NEXT552 x0 analytic bank on NEXT560, then join
the 2,400 NEXT551 and 2,000 NEXT560 opened rows.  Add only predetermined
composition descriptors derived from atomic fractions and tabulated atomic
numbers, masses, and covalent radii: ideal entropy, element count, weighted
mean/standard deviation/range/coefficient of variation for each tabulation.
No DFT value, final structure, trajectory, model potential, relaxation, or
coordinate modification is an executable input.

Convert every raw descriptor into full-development midrank risks in both
directions.  A direction may enter only when overall AUC is at least 0.64 and
every one of old-development, old-validation, NEXT560-known, NEXT560-unseen,
ordered, and SQS has AUC at least 0.55.  Rank directions by their worst stratum
then overall AUC, remove absolute Spearman redundancy above 0.95, and retain at
most 16.

Enumerate coefficient-free pairs using mean, maximum, minimum, and
probabilistic union.  Also enumerate probabilistic unions of three among the
retained directions.  A candidate must have overall AUC at least 0.75; every
provenance stratum at least 0.68; ordered and SQS at least 0.70; severity
Spearman at least 0.38; top-15% lift at least 1.55 with protected fraction at
most 0.02; overall AUC at least 0.02 above every component; and a
chemical-system bootstrap AUC lower bound at least 0.68.  Select one candidate
by worst provenance AUC, then overall AUC, then fewer terms, then lexical
formula identity.  Bootstrap candidates in that fixed order and inspect at
most the first 20; the first one passing the bootstrap gate is frozen.  This
is discovery, not confirmation.

## NEXT563 endpoint-sealed identities

Exclude every NEXT551/NEXT560 FID.  Select 3,000 remaining identities by the
lowest SHA256 of `NEXT563-v1|fid`, stratified to 1,500 ordered and 1,500 SQS.
Decode only initial structures, compute the frozen NEXT562 formula and publish
predictions before opening any endpoint.

## NEXT564 one-time confirmation

Open endpoints only for NEXT563 FIDs and apply the unchanged NEXT555 extreme
waste definition.  Require at least 200 rows per class and 75 protected rows.
Confirmation requires coverage at least 0.99; overall AUC at least 0.72 with
chemical-system cluster lower bound at least 0.68; ordered and SQS AUC at least
0.66; severity Spearman at least 0.35 with cluster lower bound at least 0.25;
top-15% lift at least 1.50 with protected fraction at most 0.02; overall AUC at
least 0.02 above every frozen component; and the same Pauling comparison rule
as NEXT561.  Passing authorizes only an independent, same-source HEA-domain
report.  Canonical reports and papers remain untouched pending user approval.
