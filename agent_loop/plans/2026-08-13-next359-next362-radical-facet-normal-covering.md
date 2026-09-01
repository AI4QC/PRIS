# NEXT359--NEXT362 radical-facet normal covering

Date: 2026-08-13

Status: frozen after closing NEXT355/NEXT356 and before computing any NEXT359
feature value or opening any endpoint outcome.

## Understanding lock

- The target is an interpretable pre-DFT screen for raw generated or theoretical
  periodic crystals, not a surrogate energy model.
- The executable may use composition, deterministic tabulated radii, and one
  initial raw unrelaxed periodic geometry only.
- DFT values or calculations, energy/force/stress predictors, learned or proxy
  potentials, relaxation, trajectories, and later geometries are prohibited.
- Discovery endpoints may be opened only after a frozen label-blind engineering
  probe and a complete formal feature build both pass.
- Validation and replication remain physically sealed.
- All work is additive. Existing scripts and content remain unchanged, and only
  the independent no-DFT report may be extended before user review.
- Success requires the same direction to clear fixed cross-source gates; a good
  result in only one source is a branch failure, not a law.

The user previously authorized autonomous continuation under these boundaries,
so there are no unresolved scope questions.

## Alternatives reviewed before selection

1. **Generator-to-power-cell centroid displacement.** Rejected before
   implementation because NEXT267 already contains `prv_centroid_offset_mean`
   and `prv_centroid_offset_q90`.
2. **Polar/Mahler volume of each power cell.** Rejected for this branch because
   it adds a translation-center convention and is likely to reproduce existing
   sphericity, anisotropy, and shape--volume features.
3. **Radical-facet normal covering (selected).** It measures the worst uncovered
   direction of the actual local cage and is distinct from positive force
   balance, individual facet area participation, the second-moment Minkowski
   tensor, and global deviatoric rigidity.

## Frozen graph and formula

Use the exact reciprocal radius-weighted radical power-facet graph certified by
NEXT339. Radii use the frozen calculated-atomic-radius then atomic-radius
fallback. Each reciprocal graph edge supplies the outward unit normal at each
endpoint. A periodic self-image edge supplies both opposite normals. Coincident
normals are deterministically deduplicated exactly as in NEXT327.

For site `i`, let `u_ij` be its distinct outward unit facet normals and define

```text
K_i = conv{u_ij}
c_i = min_F distance(0, aff(F)),
```

where `F` runs over the triangular facets of the three-dimensional convex hull
`K_i`. Equivalently,

```text
c_i = min_{|x|=1} max_j x dot u_ij.
```

Thus `acos(c_i)` is the largest angular hole in the local facet-normal cover.
For a bounded three-dimensional power cell, the origin lies strictly inside
`K_i`, and `0 < c_i <= 1`. A rank-deficient hull or failed interior certificate
abstains rather than being repaired.

The sole structure feature is

```text
rfnc_directional_covering_floor_q10
    = inverted_cdf_quantile_0.10({c_i}),
```

quantized to `1e-10`. Its sole frozen direction is `protected_high`: a larger
value means every local cage has a smaller worst directional escape hole. No
area exponent, radius variant, quantile, subgroup, graph mode, failure repair,
or direction is searched.

## Certificates and sequential gates

Unit tests must establish the analytic regular-tetrahedron value `1/3`, the
octahedral value `1/sqrt(3)`, monotonicity when a direction is added, closed
domain, the geometry-only firewall, rigid-transform/site-order/unimodular-basis
invariance, and exact two-copy supercell invariance. Reciprocal facet-area and
volume-tiling certificates remain mandatory.

The NEXT359 label-blind probe uses the deterministic 80 discovery structures
per source. It may open raw discovery geometry and prior label-free features
only. Required gates in each source are:

- at least `72/80` supported structures;
- values in `[0,1]` and at least 20 distinct values at 10 decimals;
- maximum equivalent-representation error at most `1e-8`;
- maximum absolute Spearman correlation strictly below `0.90` against all
  formal prior label-free features through NEXT355.

No endpoint field is read during this probe. Failure terminates the branch and
NEXT360--NEXT362 are not created.

Only a passing probe authorizes the all-row NEXT359 label-free build. Formal
coverage must be at least `0.90` in each of 13,470 SCIGEN and 5,232 WyFormer
discovery rows. Only a passing manifest authorizes NEXT360, which must reuse the
unchanged NEXT224/NEXT268/NEXT324 audit: frozen rejected-extreme cohort and
reduced-formula five-folds, inverted-CDF `1/16` and `15/16`, coverage `0.90`,
class count `20`, pooled AUC `0.55`, macro AUC `0.53`, worst-fold AUC `0.50`,
and the frozen `protected_high` direction.

If either source or any gate fails, the branch has zero eligible hypotheses and
NEXT361/NEXT362 are not created. Nothing may be tuned after outcomes are
opened. Results, exact hashes, and the stop decision go only into the
independent report.

## Non-functional assumptions

- Determinism and fail-closed provenance take priority over runtime; the formal
  build may use the existing bounded process-level parallelism.
- Geometry and label archives stay local and are never transmitted.
- A numerical or provenance failure is an abstention, never an imputed score.
- The additive modules remain owned by this research workflow until a branch
  passes independent validation and the user authorizes integration.

## Decision log

- Rejected the centroidal candidate after CodeGraph exposed the exact NEXT267
  implementation.
- Selected a direction-only spherical covering certificate to isolate a new
  local geometric mechanism from prior area and moment summaries.
- Fixed one feature, one direction, one aggregation, and unchanged sequential
  gates before observing any new feature value.
- Kept validation, replication, canonical reports, and paper files out of scope.
