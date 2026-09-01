# NEXT27 periodic contact-pressure design

## Why NEXT26 did not transport

NEXT26's frozen conjunction of global covalent packing and mass-density
deviation passed development but rejected only 7/290 structures in the next
checksum-locked shard.  It failed savings and precision, despite retaining
continuous DFT-response AUC 0.809 and Spearman correlation 0.826 with initial
DFT stress.  The absolute distribution of density was confounded by molecular
composition and shard membership.

Both evaluated shards are now development-only.  NEXT27 must retain their
published NEXT26 artifacts unchanged and use a new, composition-normalized
periodic contact law.  A third unopened ASE-LMDB shard is required for any new
prospective claim.

## DFT-free contact construction

From one raw periodic x0 structure:

1. enumerate atom pairs in neighbouring periodic images with exact lattice
   shift vectors;
2. infer covalent edges when `d/(r_cov,i+r_cov,j) <= 1.25`;
3. enumerate exact periodic graph paths of one, two, or three covalent edges;
4. exclude only pair/image tuples reachable by those paths (1-2, 1-3, and 1-4
   intramolecular relations), while retaining intermolecular periodic copies;
5. define `q=d/(r_vdw,i+r_vdw,j)` for the remaining pairs and retain the
   dimensionless contact shell `q <= 1.6`.

The primary analytic pressure term is

`Pi2 = (1/N) sum[max(0, 1-q)^2]`.

Related fixed diagnostics include the cubic overlap, capped inverse-power
repulsion, contact coordination below q=1.00/1.05/1.10, low q quantiles, and
per-atom contact count.  All inputs are geometry and tabulated radii; no DFT,
MLIP, energy proxy, force proxy, relaxation, or same-composition alternative is
used at execution.

## Development evidence and freeze

- `data0031` and `data0037` are labelled development shards.
- Candidate formulas contain at most two fixed-sign robust-standardized terms
  combined by a single term, equal-weight sum, or conjunction/minimum.
- A threshold is eligible only if the four original Wilson gates pass on the
  pooled development set and independently on both shards.  This shard-stable
  requirement is added to prevent the NEXT26 rejection-rate collapse.
- Formula, constants, threshold, endpoint definition, source hashes, and gates
  are frozen before any scientific field of the third shard is decoded.

The primary DFT-response endpoint and four gate cutoffs remain exactly those
in NEXT26.  No endpoint redefinition is allowed in response to NEXT26 failure.

## Third-shard protocol

Extract the next ASE-LMDB member as opaque bytes; scan only record metadata and
project frame-zero geometry; exclude all CSD refcodes seen in either
development shard; publish DFT-free features and checksum-locked predictions;
then open first/last DFT fields once.  The shared LMDB remains a procedural,
not physically isolated, lockbox and must be reported as such.

