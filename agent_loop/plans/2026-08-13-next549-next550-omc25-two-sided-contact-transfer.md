# NEXT549--NEXT550 OMC25 two-sided contact transfer audit

## Status and claim boundary

This is an additive, retrospective transfer audit.  The OMC25 validation
shards were opened by the earlier NEXT26--NEXT31 and NEXT47 work, so no result
from this branch may be described as prospective, unseen-source confirmation,
or a replacement for DFT.  Existing scripts, reports, and paper files remain
unchanged.

The executable rule may use only composition and one raw, unrelaxed, fully
periodic frame-zero structure.  It may not use DFT values, later geometries,
trajectories, relaxed structures, energies, forces, stresses, ML/MLIP outputs,
or any physical or virtual coordinate relaxation.

## Frozen hypothesis

NEXT546 found one physically interpretable near-miss on Li--Si: large response
can arise from either global under-packing or a locally over-crowded closest
contact.  Test exactly that hypothesis without fitting weights or thresholds:

1. Compute all canonical periodic pair ratios `d_ij/(r_cov,i+r_cov,j)` from
   the unmodified frame-zero structure.
2. Let `q10` and `q50` be the 0.10 and 0.50 quantiles of those ratios.
3. Across the fixed 3,099-row OMC25 cohort, compute midrank percentiles
   `u_low_q10 = percentile(-q10)` and `u_high_q50 = percentile(q50)`.
4. Freeze `TCSE = max(u_low_q10, u_high_q50)`.

There is one formula, no coefficient search, no grid search, and no endpoint
dependent normalization.  The earlier frozen NEXT31 score and each individual
component are comparators, not inputs to TCSE.

## NEXT549 label-free freeze

The cohort is the exact union of the 1,308 NEXT31 prediction rows and 1,791
NEXT47 prediction rows.  Those prediction tables contain identifiers and
label-free scores only.  Read the matching sanitized
`geometry_only_frames.zip` artifacts, require exactly one frame per identifier,
and reject any frame carrying a calculator, non-geometric arrays, or modified
periodicity.

Label-blind publication gates are:

- exactly 3,099 unique rows;
- TCSE and both components have at least 99% coverage;
- TCSE has at least 100 distinct values after rounding to 12 decimals;
- no rounded TCSE point mass exceeds 5%;
- every score lies in `[0,1]`;
- endpoint files are not accepted by the NEXT549 interface.

NEXT549 publishes a Parquet prediction table, a machine-readable formula, and
a checksum manifest before NEXT550 is run.

## NEXT550 retrospective endpoint audit

After verifying the NEXT549 manifest and checksums, join the two preserved
OMC25 endpoint tables.  Define large DFT relaxation response exactly as in
NEXT31: `energy_drop_pa >= 0.04 eV/atom`; define the protected set as
`energy_drop_pa <= 0.01 eV/atom`.

Report for TCSE, its two components, and NEXT31:

- ROC AUC for large response;
- Spearman correlation with continuous energy drop;
- precision and recall in the highest-risk 15%;
- protected structures admitted to the highest-risk 15%;
- source-shard metrics;
- a deterministic 2,000-draw CSD-refcode cluster bootstrap for TCSE AUC and
  Spearman correlation.

The predeclared diagnostic-support clauses are TCSE coverage at least 0.99,
AUC at least 0.70 with cluster-bootstrap lower bound at least 0.65, Spearman at
least 0.25 with lower bound at least 0.20, top-15% precision at least twice the
cohort prevalence, zero protected structures in the top 15%, and AUC at least
0.01 above both single components.  They cannot establish scientific success
because the source is historically opened.  A new independent report remains
forbidden until a genuinely unseen-source confirmation passes its own frozen
protocol.

