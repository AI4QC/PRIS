# NEXT33 design: approximate symmetry recovery and directional crowding laws

Date: 2026-08-03  
Status: frozen before the features are computed and before any association with labels; new files
only, with no modification to NEXT32, the old reports or the paper

## Objective and boundaries

NEXT32's absolute contact quantiles, SIVR, normalized Madelung and SCBVE reach a maximum AUC of
only 0.6751 over the 4,096 OMat24 `rattled-relax` development structures. NEXT33 tests two
physical hypotheses that are not yet adequately expressed and that still rest entirely on a
single unrelaxed `x0`:

1. a high initial DFT response may come from directional disruption of an originally coordinated
   environment, not merely from scalar bond-length mismatch;
2. a structure that has been randomly displaced may retain approximate symmetry operations
   identifiable at a loose tolerance, and the displacement needed to recover them can serve as a
   scale of geometric breaking.

Execution may use elements, the cell, fractional coordinates, periodic boundaries, frozen
covalent radii, spglib symmetry operations, deterministic periodic distances and linear algebra.
Execution may not use DFT numbers, relaxed structures or trajectories, MatterSim or any MLIP,
learned energy/force/stress surrogates, same-composition candidates, or any structural
modification. DFT forces and stresses serve only as confirmation labels, during development and
after the predictions are frozen.

OMat24's `sid` carries generation metadata including the original space group; NEXT33 **may not
parse or use those strings**. Symmetry must be recomputed from `x0`'s cell, elements and
coordinates alone.

## Comparison of the options

### A. Multi-tolerance symmetry recovery + directional crowding -- adopted

The OMat24 paper states that its structure generation includes random Gaussian displacements of
Alexandria equilibrium structures, rattled relaxation and Boltzmann rattling. A multi-tolerance
recovery quantity measures approximate structural relations directly within `x0`, while
directional crowding quantities fill the gap left by NEXT32, which counts only scalar overlap.
Neither needs a reference structure or DFT.

The risk is that the symmetry terms may recognise nothing more than "this has been artificially
perturbed". The repository's existing `sym_feat.py` records that failure mode explicitly. Symmetry
terms alone are therefore diagnostic only and may not be promoted on their own; a promotable
formula must contain at least one independent directional-crowding or existing analytic physical
term, and must still clear the low-response protection gate.

### B. Local electron counting and bond-order dissatisfaction -- later

This is closer to general chemical stability, but it overlaps the existing bond-valence/SCBVE
terms and needs a more elaborate frozen element and valence strategy, which would lower coverage.
Two mechanism families are not enlarged in the same round.

### C. Returning to the hull or formation energy as the primary endpoint -- deferred

The thermodynamic endpoint is closer to the ultimate goal, but NEXT30 already showed the current
analytic quantities to be insufficient; this round first fills the missing structural-level
mechanism, so that changing features and endpoint at once does not make attribution impossible.

## The symmetry-recovery features

Define the representation-independent characteristic length

\[
\ell=(V/N)^{1/3}.
\]

The spglib tolerances use the relative grid

```text
tau = {0.003, 0.01, 0.02, 0.04, 0.08, 0.12} * ell
```

Starting from the strictest tolerance, record the number of point-group operations after
deducting supercell translation multiples, and the orbit fraction. If some tolerance is the first
to raise the point-operation count above the strict value, define
`sym_recovery_onset_rel` as that relative tolerance; if there is no recovery it is 0. At the
loosest tolerance, emit:

- `sym_recovery_gain_log2`: log2 of the normalised gain in point operations;
- `sym_orbit_collapse`: the strict orbit fraction minus the loose-tolerance orbit fraction;
- `sym_recovery_residual_rms_rel`;
- `sym_recovery_residual_q95_rel`;
- `sym_recovery_residual_max_rel`.

Residuals are computed only for operations newly identified at the loose tolerance whose RMS
displacement exceeds the strict relative tolerance. For each operation, match the transformed
fractional coordinates one to one against the originals by element with a Hungarian assignment,
using the shortest periodic Cartesian distance divided by `ell`. The algorithm does not emit,
store or evaluate any symmetrized or refined structure.

The features must be numerically invariant under translation, atom ordering, rigid rotation and
integer supercell representation. A genuine P1 structure with no approximate recovery takes all
zeros rather than high risk, so that low symmetry is not itself judged implausible.

## The directional-crowding features

Reusing the frozen covalent radii, enumerate the unique periodic atom pairs with
`q=d/(r_i+r_j)<=1.6`. Define two fixed, dimensionless geometric kernels:

\[
w_{12}(q)=\max(0,\max(q,0.45)^{-12}-1),\qquad
w_2(q)=\max(0,1-q)^2.
\]

They fit no exponent and carry no units of energy or force. For each edge's unit direction
`u_ij`, accumulate `w*u` with opposite signs at the two endpoints to obtain a site directional
load; also accumulate the scalar load and `w u\otimes u`. Emit:

- `steric_rep12_pa`, `steric_rep12_site_q95`, `steric_rep12_site_max`;
- `steric_rep12_vector_rms`, `steric_rep12_vector_q95`, `steric_rep12_vector_max`;
- `steric_rep12_tensor_deviator`;
- `steric_overlap2_vector_rms`, `steric_overlap2_vector_q95`;
- `steric_overlap2_tensor_deviator`.

The per-atom quantities and quantiles must be supercell invariant, and uniform compression must
increase the repulsive load monotonically. The directional quantities are residuals of geometric
vector cancellation; they neither calibrate nor predict any DFT force.

## Development candidates and the promotion gate

Continue with the already-exposed 4,096-structure `rattled-relax` parent-unique cohort and its
sealed DFT endpoints; the whole source is still described as exposed development only. The new
features are sealed first and only then associated with the endpoints.

The candidates are the 16 new quantities above, plus the strongest-signal frozen terms from
NEXT32: `cov_q01_low`, `cov_q05_low`, `sivr_edge_mismatch_high`, `sivr_site_imbalance_high`. Each
term is robust-z scored using the development median and IQR, with its risk direction fixed in
advance. The candidates comprise:

- every non-symmetry single term;
- equally weighted two-term sums with a clear mechanism;
- symmetry terms are promotable only when paired with at least one non-symmetry physical term;
- symmetry single terms still emit a diagnostic row, but with `promotion_eligible=false`.

The rejection fractions remain fixed at `{0.025, 0.05, 0.075, 0.10, 0.15}`. A candidate must meet
all six NEXT32 development gates simultaneously: coverage lower bound 0.95, protective recall
lower bound 0.98, severe-response precision lower bound 0.90, savings lower bound 0.05, AUC 0.85,
and the precision lower bound minus the severe base-rate upper bound at least 0.20. The gates may
not be lowered because NEXT32 failed.

If several candidates pass, the unique choice is still made by precision lower bound, then savings
lower bound, then AUC, then term count, then rejection fraction, then lexicographic order. If no
candidate passes, stop and write an independent negative result, and continue not downloading the
three confirmation archives.

## Confirmation and the limits of any claim

Only if development passes are `rattled-300/500/1000` downloaded and processed together under the
protocol already frozen in NEXT32, at 2,048 parent-disjoint structures per source; the DFT labels
are opened only after every prediction and the Pauling comparators are sealed. The overall and
per-source gates are unchanged and no refit is permitted.

Even on success, the only permissible claim is that the fixed Pauling 2--5 comparators of this
project are surpassed on the severe-initial-DFT-response endpoint across OMat24's three
independent rattled sources; no claim may be made about hull stability, kinetic stability,
synthesizability, or replacing DFT in general. The paper, README, PREREG and old reports are not
modified before the user confirms.
