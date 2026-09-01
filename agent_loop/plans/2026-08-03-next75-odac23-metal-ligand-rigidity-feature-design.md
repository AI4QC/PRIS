# NEXT75 ODAC23 metal-ligand rigidity x0 features

## Physical hypothesis

Local metal-donor distances alone do not measure whether their simultaneous
geometric mismatch can be supported by the coordination network.  NEXT75 adds
central-force stiffness and self-stress compatibility descriptors on the
metal-donor subgraph.  This hypothesis is developed on discovery only; the
already opened internal-validation endpoint is forbidden input, and internal
replication remains unopened.

## Frozen graph and kernels

From the existing periodic covalent-radius graph (ratio cutoff 1.25), retain
only edges joining a recognized metal to N, O, F, P, S, Cl, Br, or I.  Compact
the incident sites without changing coordinates.  Use unit edge weights,
frozen covalent-radius sums, periodic edge vectors, and the existing NEXT20
scale-invariant rigidity kernel.  Exclude its non-intensive edge/site counts
and the smallest stiffness eigenvalue, which a pre-label NaCl audit showed is
not invariant to an otherwise identical supercell representation.

Using the identical edges, define the dimensionless edge residual as the
log-distance/radius ratio minus the NEXT20 weighted median.  Project this
residual through the existing NEXT37 atomic+affine rigidity columns and retain
load fraction/RMS/q95, atomic and cell load fractions, localization, balanced
fraction, and cokernel-dimension fraction.  Append intensive active-site and
edge-density fractions.  Fail open only when the metal-donor subgraph or either
analytic kernel is unsupported.

## Boundary and publication

Build discovery, internal validation, and internal replication together from
the sealed raw framework x0 archive, merging onto the label-free NEXT70 table
by exact material ID.  No endpoint label, opened validation result, relaxed
coordinate, DFT value/calculation, energy/force/stress model, proxy potential,
physical relaxation, or same-composition alternative is allowed.  Publish a
new no-replace artifact and preserve all old scripts/artifacts.
