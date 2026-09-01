# NEXT80 periodic repulsive-load resolvability certificate

## Independent hypothesis

A raw periodic crystal geometry is less plausible when its covalent/coordination
bar framework cannot statically resolve the repulsive generalized load generated
by nonbonded van der Waals overlaps.  This is a mechanics necessary-condition
hypothesis, fixed without inspecting ODAC23 endpoint labels and not an additive
extension of the NEXT79 discovery formula.

Periodic framework theory represents infinitesimal motions with a rigidity
matrix and periodic equilibrium stresses with the kernel of its transpose,
including columns for affine cell motion.  Tensegrity theory assigns sign
constraints to cables and struts while bars remain sign-free.  NEXT80 therefore
treats covalent-radius graph edges as bars and noncovalent pairs with
`d/(r_vdw_i+r_vdw_j) < 1` as compression-only struts.  The descriptor is a
heuristic load-resolvability certificate, not a theorem that equilibrium alone
proves thermodynamic stability.

Primary theory references:

- S. C. Power, *Crystal frameworks, symmetry and affinely periodic flexes*,
  arXiv:1103.1914.
- S. C. Power, *Equilibrium stresses and rigidity for infinite tensegrities and
  frameworks*, Journal of Mathematical Analysis and Applications 2023,
  arXiv:2207.14369.
- A. Nixon, B. Schulze et al., *Minimally Rigid Tensegrity Frameworks*,
  Discrete & Computational Geometry 2025, for equilibrium and strut/cable sign
  conventions.

## Frozen geometry and load construction

1. Use the unchanged raw x0 and the existing frozen covalent and van der Waals
   radius tables.
2. Build canonical periodic covalent edges with the existing NEXT49 cutoff
   `d/(r_cov_i+r_cov_j) <= 1.25`; these are sign-free bars.
3. Build canonical periodic noncovalent contacts with
   `d/(r_vdw_i+r_vdw_j) < 1`.  Exclude every exact periodic covalent edge.
4. Give each contact a fixed compression magnitude
   `w=max(ratio,0.45)^(-12)-1`.  Normalize contact magnitudes to sum to one for
   the equilibrium residual while retaining their unnormalized RMS and maximum
   as intensity diagnostics.
5. Each edge contributes equal-and-opposite unit atomic forces plus six affine
   cell components.  The cell components use the length-normalized symmetric
   dyad `(d/L) u tensor u`, where `L=(V/N)^(1/3)`; off-diagonal entries carry
   `sqrt(2)` so Euclidean norm equals Frobenius norm.
6. Solve the sign-free bar stresses which minimize the combined atomic-plus-cell
   residual of the fixed compressive contact load using deterministic sparse
   LSQR.  No coordinates or cell parameters move.

## Frozen features and risk score

Record total, atomic, and cell residual fractions; bar-stress RMS,
amplification, and localization; raw contact RMS/max; contact and covalent edge
densities; LSQR convergence diagnostics; and

```text
prlr_risk = prlr_residual_fraction * log1p(prlr_contact_weight_rms)
```

All quantities must be finite and invariant under an exact integer supercell.
No-label cases with no noncovalent overlap have zero load, zero residual, and
zero risk rather than being discarded.  A structure without a covalent graph is
unsupported and remains fail-open.

## Prospective evaluation protocol

NEXT80 only builds features for all frozen x0 partitions.  A separate NEXT81
stage will read no endpoint labels and freeze `REJECT` at the discovery-x0 95th
percentile of `prlr_risk`, with missing=KEEP.  Only after the formula and all
predictions are hashed may NEXT82 read robust discovery labels once.  There is
one score and one threshold, with no direction, weight, feature, or threshold
search.  Advancement requires all seven original gates and a discovery reject
precision Wilson lower bound of at least 0.80.  Failure stops this hypothesis;
internal replication stays unopened.

## Boundary

Inputs are one raw unrelaxed x0, frozen elemental radii, deterministic periodic
graphs, sparse linear algebra, and analytic pair repulsion.  No DFT calculation
or value, relaxed geometry, opened validation result, learned energy/force/
stress proxy, physical relaxation, or same-composition alternative is used.
