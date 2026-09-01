# NEXT375--NEXT378 Periodic Coordination Reciprocity Likelihood Plan

**Status:** frozen before computing any NEXT375 feature value or opening any
outcome.  Date: 2026-08-13 (America/Chicago).

## Goal and information boundary

Test a classical, non-DFT coordination principle that is not represented by the
existing feature catalogue: coordination environments inferred independently
at the two ends of a contact should be reciprocal.  The executable object may
receive only element identities and one initial raw, unrelaxed, fully periodic
geometry.  It must not read or calculate a DFT value, use an energy/force/stress
model or proxy potential, relax a structure, inspect a later geometry or
trajectory, or open validation/replication data.

The mechanism is motivated by the coordination-reciprocity construction of
O'Keeffe and Hyde as formalized for ordered solid-angle coordination strings by
Wagner *et al.*, Acta Cryst. A81 (2025), DOI
`10.1107/S2053273325001945`.  That paper obtains contact surfaces from electron
density, which is outside this project's boundary.  NEXT375 uses only the
paper's explicitly discussed geometric Voronoi--Dirichlet analogue: periodic
Voronoi facets and their solid angles.  No QTAIM/electron density, DFT, or
paper-derived calculated value enters the executable formula.

## Repository novelty audit made before the freeze

- P6/Brunner already measures a local distance-shell gap.
- P4 and NEXT46/NEXT52 already measure continuous polyhedral symmetry and
  local-environment clarity/coherence.
- NEXT239/NEXT251 already measure Voronoi facet evenness, bond order, face
  topology, and same-element consistency.
- NEXT303 measures reciprocity of a fixed fourth-nearest opposite-sign cage,
  not reciprocity of independently selected solid-angle coordination strings.
- NEXT307 and later graph features measure circulation, rigidity, closure, or
  spectral properties, not the double-ended consistency of locally preferred
  coordination prefixes.

Thus the frozen object below is mechanistically distinct enough for a
label-blind novelty probe.  Measured correlation with all prior label-free
controls remains a mandatory stop gate.

## Frozen NEXT375 construction

1. Strictly validate an ASE `Atoms` object: at least one site, full three-axis
   periodicity, finite nonsingular cell and positions, no calculator, empty
   `info`, and exactly the `numbers` and `positions` arrays.  Minkowski-reduce
   and wrap the cell using immutable NEXT267 geometry code.
2. Construct the ordinary periodic Voronoi--Dirichlet tessellation with
   `VoronoiNN(weight="solid_angle", tol=0, cutoff=13)`.  Retain every finite,
   positive facet.  Represent a directed periodic contact as `(i,j,T)` and its
   reverse as `(j,i,-T)`.  Require complete reverse incidence.  The two reported
   facet solid angles must agree to absolute `1e-8`; replace them by their
   arithmetic mean so both directions receive exactly one shared weight.
3. At every site sort directed contacts by decreasing shared solid angle, then
   by the exact periodic contact key.  Normalize by that site's largest solid
   angle to obtain `r_1 >= ... >= r_k > 0`, set `r_(k+1)=0`, quantize the ratios
   to the `1e-10` grid, and compute the linear coordination likelihood gaps
   `g_j = r_j-r_(j+1)`.
4. The locally preferred coordination number is the `j` with largest `g_j`;
   an exact tie selects the larger `j` (the more inclusive prefix).  The first
   `j` contacts form that site's independently selected directed coordination
   prefix.  This is fixed before outcome access; no fitted contact threshold is
   permitted.
5. A selected directed contact is reciprocal exactly when its reverse is also
   selected.  Publish one dimensionless formula

```text
pcrl_reciprocity_deficit
  = sum_selected omega(i,j,T) * 1[(j,i,-T) not selected]
    / sum_selected omega(i,j,T).
```

   Quantize only the final aggregate to `1e-10`.  Its frozen direction is
   `protected_low`.  Zero is allowed; every supported value must be finite in
   `[0,1]`.

## Label-blind gate before any formal build

Use the unchanged deterministic 80-record discovery-only probe from each of
SCIGEN and WyFormer.  The probe may open raw discovery geometries and prior
label-free feature tables through NEXT367 only.  It may not open outcome,
endpoint, validation, replication, or relaxed-geometry artifacts.

NEXT376 formal construction is authorized only if all are true on both
sources:

- support at least 72/80;
- finite domain `[0,1]`;
- at least 20 distinct values after rounding to ten decimals;
- maximum error across rigid rotation, translation, atom permutation,
  unimodular rebase, and exact `2x1x1` replication at most `1e-8`;
- maximum absolute Spearman correlation with any prior label-free control is
  strictly below `0.90`, using only controls with at least 40 joint finite rows.

Failure of any gate terminates this branch without opening a label.  Passing
the probe authorizes a full discovery-only NEXT376 build, after which NEXT377
may audit the single frozen `protected_low` hypothesis with the unchanged
cross-source discovery gates.  NEXT378 may search only if NEXT377 explicitly
authorizes it.  Validation and replication remain sealed in every case unless
separately authorized later.

## Additive reporting and verification

All new scripts, tests, probe output, and conclusions are additive.  Preserve
every previous script and result.  Append findings only to the independent
no-DFT search report; do not edit `paper/`, `tex/`, `notes/`, `README.md`, or
`PREREG.md`.  Verify focused tests, boundary flags, artifact hashes, the full
test suite, CodeGraph health, and absence of unauthorized NEXT376--NEXT378
artifacts before reporting a stopped branch.
