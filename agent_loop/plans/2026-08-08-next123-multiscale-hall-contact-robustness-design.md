# NEXT123 multiscale Hall-contact robustness design

Date: 2026-08-08

Status: feature definition frozen before production implementation or feature
materialization

## Boundary

NEXT123 is additive.  It preserves every existing script, result, report, and
canonical document.  Its executable input is one raw unrelaxed structure plus
the already-frozen analytic element catalogue and Voronoi contact construction.
It must not read or execute DFT quantities, relaxed structures or trajectories,
same-composition alternatives, MLIP outputs, learned energies, forces, or
stresses.  Later discovery outcomes may be used only as offline labels after
the feature catalogue and finite candidate universe are frozen.

## Motivation from the frozen residual

NEXT122 leaves two adjacent discovery frontiers:

- the closest SAFE12 law misses only SCIGEN worst-lattice AUC by
  `0.002164028435214904`;
- `250` laws pass all six source-AUC gates and `11/12` SAFE cells, and every one
  fails only `wyformer:fold2` severe-rejection precision.

Further fitting BVTC/PRLR weights on the same outcomes is not justified.  The
new hypothesis instead asks whether charge feasibility depends on weak
geometric contacts that the unweighted HCID graph treats identically to robust
Voronoi faces.

## Frozen weighted graph

Use exactly NEXT109's element-state catalogue, electronegativity-oriented sign
patterns, legacy obstruction rank, and lexicographic tie-break.  The selected
sign pattern must therefore be identical to NEXT109/NEXT115.

For each positive site, call the same frozen Voronoi neighbor finder used by
NEXT109.  Retain the same valid opposite-sign endpoint identities.  Let the
reported positive finite Voronoi neighbor weight be `q`.  Normalize it by the
largest valid neighbor weight reported for that same positive origin, including
same-sign neighbors:

```text
w_e = q_e / max(q_valid_from_same_origin).
```

Thus `0 < w_e <= 1`.  Multiple periodic images of the same endpoint are reduced
by their maximum normalized strength.  The unweighted endpoint set must match
NEXT109 exactly or the structure abstains.

## Multiscale generalized-Hall deficit

Freeze the four dimensionless contact-strength thresholds

```text
T = (0.05, 0.10, 0.25, 0.50).
```

For `tau` in `T`, retain `E_tau = {e: w_e >= tau}`.  For an origin sign side
`S` and its opposite-side neighborhood in `E_tau`, compute the exact maximum
generalized-Hall deficit

```text
Delta_side(tau) = max_A [sum(l_i for i in A)
                         - sum(u_j for j in N_tau(A))].
```

The maximum-closure LP is integral, exactly as in NEXT115.  Let
`Delta_side(0)` use every valid opposite-sign edge and let
`L_side = sum(l_i)` over all sites on that origin side.  Publish only the
increment caused by deleting weak contacts:

```text
G_side(tau) = max(Delta_side(tau) - Delta_side(0), 0) / L_side.
```

The eight frozen features are

```text
mhcr_positive_deficit_gain_tau05
mhcr_positive_deficit_gain_tau10
mhcr_positive_deficit_gain_tau25
mhcr_positive_deficit_gain_tau50
mhcr_negative_deficit_gain_tau05
mhcr_negative_deficit_gain_tau10
mhcr_negative_deficit_gain_tau25
mhcr_negative_deficit_gain_tau50
```

No aggregate, learned combination, label-calibrated threshold, polarity flip,
or alternative edge score is allowed in NEXT123.

## Required mathematical and engineering invariants

- Every feature is finite and in `[0, 1]` up to numerical tolerance.
- For either sign, gains are nondecreasing with `tau`.
- Site permutation, edge ordering, periodic-image ordering, and duplicate edges
  do not change the result.
- Common positive charge scaling does not change the normalized result.
- Exact integer replication of the signed weighted graph does not change the
  normalized result.
- If every edge survives `tau=0.50`, all eight gains are zero.
- The full endpoint set exactly reproduces NEXT109's graph.
- Invalid intervals, orientations, strengths, or solver outcomes fail closed.
- Unsupported raw structures retain the existing NEXT109 technical abstention.

## Frozen downstream routing

1. NEXT123 implements and tests only the graph certificate and raw-structure
   wrapper.
2. NEXT124 materializes all eight label-free features for the already-frozen
   SCIGEN and WyFormer discovery structure catalogues outside the repository.
3. Before any new feature is joined to outcomes, a label-free audit may retain
   at most four nondegenerate, nonredundant high-risk terms.  Selection may use
   support, exact-zero rate, IQR, cap, and pairwise correlation, but no outcome.
4. The frozen base frontier is the union of all `250` NEXT122 AUC+SAFE11
   formulas and the first `256` unique NEXT122 SAFE12 formulas ranked by minimum
   six-AUC margin, with deterministic identity tie-breaks.
5. For at most four retained terms, NEXT125 permits the base, every single, and
   every pair at weights `(0.1, 0.25, 0.5, 1.0)`.  No triple is allowed.
6. Candidate identities and all input hashes must be frozen before the two
   already-open discovery outcome tables are reread.  Every base must reproduce
   NEXT122.
7. Validation and replication remain unopened unless one fixed candidate passes
   all six source-AUC, all SAFE, BROAD, and every other inherited discovery gate.

## Claim boundary

Passing graph invariants establishes an analytic robustness certificate, not a
stability predictor.  A discovery-gate pass would authorize the already-defined
next validation step; it would not by itself establish replacement of Pauling
rules or DFT-quality screening.
