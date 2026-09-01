# NEXT58 ODAC23 shared local-motif feature design

## Motivation and firewall state

NEXT57 showed that global analytic mechanics alone does not meet the discovery
gates.  Internal-validation, internal-replication, official validation, test,
and OOD labels remain unopened.  NEXT58 follows the independently motivated
QMOF result that metal/donor-resolved local coordination is more informative
than a single global motif average.

## Executable boundary

For each selected raw x0, run the deterministic matminer
`CrystalNNFingerprint.from_preset("ops")` once per site.  It uses only periodic
geometry, atom identities, Voronoi neighbours, and analytic order parameters.
No DFT value, relaxed coordinate, energy/force/stress proxy, learned potential,
physical relaxation, or alternative structure is available to the builder.
Unsupported motif rows remain `NaN` and force `KEEP` for formulas using them.

## Frozen feature catalogue

Retain all 56 sealed NEXT55 features and append:

- all 21 NEXT46 global motif-coherence descriptors;
- all 30 NEXT52 metal/donor site-resolved descriptors;
- fourteen predefined tail/outlier descriptors: metal and donor CN-dominance
  q10; order-strength q10; entropy maxima; effective-CN minima and maxima;
  donor fingerprint dispersion RMS and q95; and fractions of metal/donor sites
  with CN dominance below 0.5.

The shared fingerprint matrix must reproduce NEXT46 and NEXT52 values exactly
on the same x0.  Environment versions and the matminer implementation hash are
sealed.  The feature process accepts no label path.
