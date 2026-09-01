# NEXT351--NEXT354 periodic deviatoric strain rigidity

Date: 2026-08-13

Status: frozen before any NEXT351 feature value or endpoint outcome is opened.

## Question and executable boundary

Can a raw periodic contact framework resist every homogeneous volume-preserving
strain after all periodic internal site displacements have been allowed to
cancel contact extension?  A viable crystal should not possess a cheap global
hinge or Guest mode that removes an imposed shear from every retained
opposite-sign contact.

The executable descriptor may use only composition, deterministic tabulated
element data, and one initial raw unrelaxed periodic geometry.  It must not use
or execute DFT, an energy/force/stress label or predictor, a learned potential,
relaxation, a trajectory, or any later geometry.  The least-squares internal
displacement below is a closed-form kinematic projection.  It does not move
the input atoms and is not a relaxation or proxy energy calculation.

All work is additive.  Existing scripts and content are retained.  Canonical
`paper/`, `tex/`, `notes/`, `README.md`, and `PREREG.md` remain unchanged until
the independent report is reviewed.

## Alternatives considered

1. Full six-dimensional strain retention was rejected because its hydrostatic
   direction is strongly entangled with already tested packing and radius
   mismatch descriptors.
2. A local site tensor was rejected because NEXT168/NEXT173 already test local
   directional rigidity.  Local enclosure does not certify a globally rigid
   periodic framework.
3. A residual-specific compatibility projection was rejected because NEXT37
   already projects a frozen bond-length mismatch vector.  The selected PDSR
   construction instead interrogates the complete five-dimensional deviatoric
   strain space without supplying a mismatch or outcome vector.

## Frozen graph

Use the existing NEXT19 analytic valence assignment and its `voronoi`
opposite-sign periodic graph.  This graph uses no endpoint and showed formal
label-free support above 0.98 in both sources in NEXT168.  Each retained edge
has endpoints `(i,j)`, periodic Cartesian displacement `d_e`, length `l_e`,
unit direction `n_e=d_e/l_e`, and positive Voronoi solid-angle weight `w_e`.
No graph mode, cutoff, exponent, or chemistry subgroup is searched.

## Frozen kinematic law

Let `U` be the weighted relative-extension matrix for periodic internal
Cartesian displacements:

```text
U[e, 3*i:3*i+3] = -sqrt(w_e) n_e / l_e
U[e, 3*j:3*j+3] = +sqrt(w_e) n_e / l_e.
```

For a self-image edge the two endpoint blocks cancel, as required for a
primitive-periodic internal displacement.  Let `E_1,...,E_5` be a fixed
Frobenius-orthonormal basis of symmetric trace-free 3 x 3 tensors, and define

```text
D[e,a] = sqrt(w_e) n_e.T E_a n_e.
```

The affine Gram matrix is `H0=D.T D`.  If `H0` is not positive definite, the
raw contact directions do not span all five deviatoric strains and the row
abstains.  Otherwise project the affine extensions away from all periodic
internal displacements:

```text
Z  = (I - U U^+) D
H  = Z.T Z
M  = H0^(-1/2) H H0^(-1/2)
lambda = eigvalsh(M).
```

The sole public feature is

```text
pdsr_deviatoric_retention_floor = min(lambda),
```

quantized to `1e-10` and directed `protected_high`.  Exact numerical roundoff
outside `[0,1]` may be clipped only within a frozen tolerance; other violations
abstain.  No threshold is selected in NEXT351.

`lambda=0` means at least one volume-preserving homogeneous strain is exactly
cancelled by a periodic internal hinge motion.  `lambda=1` means internal site
motions cannot cancel any part of the weakest deviatoric strain.  By the
fundamental theorem of linear algebra the residual is the component coupled to
the framework's self-stress space, but no force law or stress value is used.

## Exact invariances and certificates

The kernel must be invariant to global translation, rigid rotation, site and
edge permutation, common rescaling of all edge weights, uniform coordinate
scaling, and unimodular lattice rebasing.  Exact supercell replication must
also preserve the feature.  For a repeated primitive graph, the uniform
affine forcing lies entirely in the zero-wavevector block of the block-cyclic
compatibility matrix.  Nonzero-wavevector internal modes are orthogonal to
that forcing, so enlarging the represented cell cannot lower its normalized
projection residual.  Tests must compare a primitive quotient to an explicit
cover, in addition to structure-level transforms.

## Sequential gates

### NEXT351 label-blind probe

Select 80 discovery structures per source by the already used deterministic
ordering `sort(natoms, chemical_system, material_id)` followed by evenly
spaced indices.  Open raw discovery geometries only.  Compare the sole PDSR
feature against the complete frozen label-free feature population through
NEXT347, including PARC values but not its audit outcomes.

All gates are mandatory in both SCIGEN and WyFormer:

- support at least 72/80;
- finite strict domain `0 <= value <= 1`;
- at least 20 unique values after rounding to 10 decimals;
- maximum transform error at most `1e-8`;
- maximum absolute Spearman correlation with every prior label-free feature
  strictly below `0.90`.

Failure terminates the branch without a formal build or endpoint audit.

### NEXT351 formal label-free build

If the probe passes, build all 13,470 SCIGEN and 5,232 WyFormer discovery rows.
Each source must have at least 0.90 supported coverage.  Labels, discovery
endpoints, validation, and replication remain unopened.  Failure terminates
the branch without NEXT352.

### NEXT352 fixed discovery audit

If formal coverage passes, reuse the unchanged NEXT224/NEXT268/NEXT324 audit:
the frozen rejected-extreme cohort, reduced-formula five folds, inverted-CDF
quantiles `1/16` and `15/16`, minimum cell coverage 0.90, minimum class count
20, pooled AUC 0.55, macro-fold AUC 0.53, and worst-fold AUC 0.50.  Direction
is fixed as `protected_high`.  Discovery outcomes are offline labels only;
validation and replication stay sealed.

Every source and every gate must pass.  Otherwise the branch terminates with
zero eligible hypotheses and NEXT353/NEXT354 are not created.  Gates will not
be weakened and the graph, formula, direction, chemistry, unsupported-row
policy, or thresholds will not be changed after outcomes are opened.

### NEXT353--NEXT354

NEXT353 is authorized only by a passing NEXT352 manifest and may run only the
already bounded law-search protocol.  NEXT354 is authorized only by a frozen
NEXT353 winner and must preserve the existing sealed validation/replication
policy.  Neither stage is created prospectively.

## TDD and reporting

Tests precede implementation.  Unit tests cover analytic zero and unit
certificates, generalized-spectrum bounds, input refusal, rotation and gauge
invariance, explicit-cover replication, and geometry-only boundaries.  Probe
tests cover deterministic IDs, prior-feature completeness, novelty, gates,
and absence of any outcome interface.  Results are appended only to the
independent no-DFT report with exact hashes and stop decisions; canonical
documents remain untouched.
