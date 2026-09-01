# NEXT267--NEXT270 Periodic Radical-Voronoi Packing Plan

## Objective and hard boundary

Test whether a radius-weighted periodic power diagram exposes raw-geometry
packing defects that the existing ordinary Voronoi, pair-overlap, bond-order,
rigidity, void-cage, void-network, and persistent-homology branches miss.

Every executable feature and candidate law may use only elemental identities,
tabulated elemental radii, and the initial unrelaxed three-dimensional periodic
geometry. Discovery outcomes may be used only as offline labels after the
feature code, feature catalogue, hypotheses, directions, normalizations, and
decision gates are frozen. No DFT calculation or per-structure DFT value,
learned energy/force/stress proxy, model or proxy potential, relaxed structure,
trajectory, physical relaxation, validation output, or replication output may
enter the executable path.

All work is additive. Existing scripts and report content remain unchanged.
Only the independent report may be appended after the branch is complete;
`paper/`, `tex/`, `notes/`, `README.md`, and `PREREG.md` remain untouched.

## Scientific mechanism

For sites at Cartesian positions `p_i` with frozen tabulated radii `r_i`, the
periodic radical/power cell of site `i` is

```text
C_i = {x : |x-p_i|^2-r_i^2 <= |x-p_j-T|^2-r_j^2 for every j,T}.
```

In coordinates `y=x-p_i`, each competitor contributes the half-space

```text
u_ijT dot y <= c_ijT,
u_ijT = d_ijT / |d_ijT|,
c_ijT = (|d_ijT|^2 + r_i^2 - r_j^2) / (2 |d_ijT|).
```

Unlike an ordinary Voronoi diagram, this partition treats chemically unequal
atomic sizes explicitly. It can expose an empty power cell, exclusion of the
generating atom from its own cell, failure of its radius sphere to fit inside
the cell, abnormal allocation of periodic volume among species, or a strongly
off-centred/anisotropic many-body cage.

## Frozen elemental-radius policy

Use `pymatgen.core.periodic_table.Element.atomic_radius_calculated`; if absent,
fall back to `Element.atomic_radius`. A structure abstains if a finite positive
radius cannot be obtained for every site. No oxidation-state inference or
structure-dependent radius selection is allowed.

## Frozen periodic construction

1. Require a finite, positive-volume, fully periodic 3D ASE `Atoms` object.
2. Minkowski-reduce the lattice using ASE, preserve Cartesian coordinates, and
   wrap sites into the reduced cell.
3. Construct the lattice Wigner--Seitz cell from every nonzero translation in
   `[-2,2]^3`; its maximum vertex distance is `R_WS`.
4. Enumerate periodic site images with ASE's neighbor list to the conservative
   cutoff `R_WS + sqrt(R_WS^2 + r_max^2) + 1e-8 Angstrom`.
   Abstain if this produces more than 2,000,000 directed neighbor images for
   one structure; this is a fixed resource guard, not a label-dependent filter.
5. Keep every plane with `c_ijT <= R_WS + 1e-8 Angstrom`. Reject zero-distance,
   nonfinite, unbounded, or numerically inconsistent systems.
6. Find the maximum-inscribed-ball centre and radius by a fixed HiGHS linear
   program. A nonpositive interior radius denotes an empty or lower-dimensional
   power cell and contributes zero cell volume.
7. For nonempty cells, use SciPy `HalfspaceIntersection` and `ConvexHull` with
   `Qx`; verify all vertices satisfy all planes. Compute cell volume, a
   tetrahedral volume centroid, unique active facet count, and vertex-covariance
   anisotropy `1-lambda_min/lambda_max`.
8. Require the sum of all nonempty labelled power-cell volumes to reproduce the
   periodic cell volume within relative tolerance `1e-6`; otherwise the whole
   structure abstains.
9. Quantize published scalar values on a fixed `1e10` grid before structural
   aggregation. Quantiles use NumPy `inverted_cdf`; standard deviations are
   population values.

## NEXT267 frozen feature catalogue

Publish exactly these sixteen finite structure features on supported rows:

```text
prv_empty_cell_fraction
prv_generator_excluded_fraction
prv_sphere_crossing_fraction
prv_allocation_total_variation
prv_volume_ratio_q10
prv_volume_ratio_q90
prv_volume_ratio_cv
prv_chebyshev_ratio_q10
prv_chebyshev_ratio_q90
prv_chebyshev_ratio_cv
prv_centroid_offset_mean
prv_centroid_offset_q90
prv_vertex_anisotropy_mean
prv_vertex_anisotropy_q90
prv_facet_count_mean
prv_facet_count_cv
```

Definitions:

- `empty_cell_fraction`: fraction of labelled cells with no positive-volume
  interior.
- `generator_excluded_fraction`: fraction whose minimum `c_ijT/r_i <= 0`.
- `sphere_crossing_fraction`: fraction whose minimum `c_ijT/r_i < 1`.
- `allocation_total_variation`: one half the L1 distance between power-cell
  volume fractions and normalized `r_i^3` fractions.
- `volume_ratio`: cell volume divided by `4*pi*r_i^3/3`, over nonempty cells.
- `chebyshev_ratio`: maximum-inscribed-ball radius divided by `r_i`, over
  nonempty cells.
- `centroid_offset`: generator-to-cell-centroid distance divided by the cell's
  equal-volume sphere radius, over nonempty cells.
- `vertex_anisotropy`: `1-lambda_min/lambda_max` of unique cell vertices about
  the volume centroid, over nonempty cells.
- `facet_count`: number of active power planes with a 2D facet, over nonempty
  cells.

NEXT267 reads only physically isolated discovery geometry and matching
identifier metadata. Unsupported structures abstain. It publishes separate
SCIGEN and WyFormer tables, a frozen catalogue, counts, source/input/output
hashes, and explicit false boundary flags.

## Label-free engineering gates

Before any discovery endpoint is opened:

- unit tests cover analytic one-site cubic cells, FCC, BCC, NaCl, diamond, an
  asymmetric triclinic structure, a radius-dominated empty-cell example, and
  malformed inputs;
- rotation, periodic translation, atom permutation, equivalent lattice-basis
  change, and exact `2 x 1 x 1` replication preserve features within `1e-8`;
- supported outputs are finite and satisfy the volume-tiling certificate;
- a read-only discovery-geometry sample demonstrates acceptable support and
  runtime without reading labels or endpoint fields.

## NEXT268 frozen offline feature audit

After NEXT267 is published and hashed, audit exactly the Cartesian product of
the sixteen feature names and directions `protected_low`, `protected_high`: 32
hypotheses. For each feature, freeze the combined-discovery inverse-CDF
quantiles `(1/16,15/16)` and map the interval linearly to protection, with
conservative abstention outside finite support.

Audit inside the exact reproduced NEXT224 supported, rejected extreme cohort:
protected endpoint `<=1.0`, severe endpoint `>=2.0`, score at least
`0.1520033762332462`. Reproduce all frozen source/fold cohort counts. A
hypothesis is eligible for search only if it passes the unchanged NEXT227 raw
coverage, class-count, aggregate-AUC, macro-fold-AUC, and worst-fold-AUC gates
in both discovery sources. Ranking is reporting-only and cannot authorize an
otherwise failing hypothesis.

If zero hypotheses are eligible, terminate this branch without formula search.

## Conditional NEXT269 search

Only if NEXT268 authorizes at least one exact hypothesis, evaluate one exact
NEXT224 reproduction control plus, for every eligible hypothesis, the frozen
triangular margin-local grid of seven width fractions and three amplitude
fractions used by NEXT261. Preserve the NEXT214 support mask and use a
nonnegative correction only; no label-dependent feature transformation is
allowed.

Require the unchanged dual-source AUC, twelve-cell SAFE, and BROAD gates. A
reporting leader is not a frozen law. Freeze authorization is true only for a
candidate passing every discovery gate.

## Conditional NEXT270 diagnostic

Only if NEXT269 produces exact candidates that pass both-source AUC and SAFE
but fail BROAD, reproduce only those identities and compute unchanged BROAD
residual certificates. Compare lexicographically with the frozen NEXT235
reference `(failed_constraint_count, normalized_shortfall_sum) =
(5,0.12339543654931197)`. A tie in failure count requires strictly smaller
shortfall to count as progress. This diagnostic cannot authorize validation,
change a threshold, or select a law.

## Verification and reporting

Run focused tests, the complete repository suite, manifest/input/source/output
hash verification, boundary scans, and CodeGraph status. Append an independent
report section with coverage, failures, all audited/search counts, exact gate
metrics, hashes, and a conservative conclusion. Do not claim replacement of or
superiority to Pauling without a prospectively frozen candidate that later
passes separately authorized sealed validation and replication.
