# NEXT70 ODAC23 metal-donor bond-valence x0 features

## Motivation

NEXT69's general opposite-sign charge graph supported only 21/64 CrystalNN and
46/64 Voronoi examples in a label-free real-x0 audit, below the unchanged 0.95
coverage gate.  NEXT70 therefore targets the chemically relevant metal-donor
subgraph already supported for 7,814/7,815 NEXT63 frameworks.  NEXT69 is kept as
an auditable failed route and is not overwritten.

## Frozen algorithm

Use the existing periodic covalent-radius graph with ratio cutoff 1.25.  For
each recognized metal site, retain adjacent N, O, F, P, S, Cl, Br, or I donors.
Assign fixed donor valences N/P=-3, O/S=-2, and halogens=-1.  Enumerate the
metal's positive common oxidation states (falling back to its positive listed
states, restricted to 1--8).  For every candidate state, resolve the frozen
bond-valence parameter for every observed metal-donor distance with the
existing exact/nearest/fallback policy and compute the bond-valence sum.

Select the oxidation state minimizing absolute relative mismatch
`abs(BVS-oxidation)/oxidation`, with lower oxidation state as the exact tie
break.  From the selected state compute signed mismatch, deficit, excess,
vector asymmetry, effective coordination, oxidation-state ambiguity gap,
parameter provenance, and donor count.  Aggregate intensive distribution
statistics over evaluable metal sites and record the evaluable-metal fraction.
Sites without a donor are omitted from site aggregates; a structure fails open
only when no metal site is evaluable.

## Boundary and publication

Build all three sealed train partitions together from one raw framework x0 and
the frozen elemental/bond-valence tables.  Do not open labels, relaxed
coordinates, DFT values, energy/force/stress models, proxy potentials, or
same-composition alternatives.  Merge by exact material ID onto NEXT65 and
publish a new no-replace artifact.  Preserve NEXT65 support independently of
this optional feature family.
