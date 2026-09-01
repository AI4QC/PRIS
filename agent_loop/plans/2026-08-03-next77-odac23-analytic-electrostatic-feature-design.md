# NEXT77 ODAC23 analytic electrostatic x0 features

## Physical hypothesis

Framework relaxation can be driven by uncompensated classical electrostatic
fields or by disagreement between electrostatic and short-range steric vector
fields even when local bond lengths look plausible.  NEXT77 ports the existing
label-free normalized Madelung, analytic Ewald-field, Coulomb--steric balance,
and finite-cell charge-spectrum kernels to ODAC23 frameworks.

## Frozen calculation

Infer a neutral site charge assignment with the existing deterministic
integer/fractional/electronegativity fallback.  On the unchanged raw x0,
evaluate NEXT21's charge-sign and concentration diagnostics, NEXT34's dimensionless
field and residual summaries, NEXT35's closed-form Coulomb--steric vector
balance, and NEXT36's scale-free Gaussian reciprocal charge spectrum.  Retain
only feature columns proven finite and supercell invariant by the pre-label
NaCl test; exclude all NEXT21 Ewald magnitude terms, reciprocal vector counts,
and other representation-size diagnostics if they fail that test.

## Firewall and publication

Build all three selected train partitions together and merge onto the
label-free NEXT75 table by exact material ID.  The opened internal-validation
endpoint/result and unopened internal-replication endpoint are forbidden
inputs.  Classical analytic electrostatics is allowed, but no DFT
calculation/value, relaxed geometry, energy/force/stress model, proxy
potential, physical relaxation, or same-composition alternative is allowed.
Publish to a new no-replace artifact and preserve all old content.
