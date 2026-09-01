# NEXT283--NEXT286 Power-Cell Shape--Volume Coupling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Status:** frozen before opening any NEXT283--NEXT286 feature or discovery outcome.

**Goal:** Test whether surface-area-resolved radical-cell shape--volume mismatch improves the current no-DFT discovery frontier while preserving every prior script and keeping validation and replication sealed.

**Architecture:** NEXT283 independently recomputes the exact NEXT267 periodic radical power cells from initial geometry, adds convex-cell surface area and sphericity, and publishes six preregistered structure summaries for the two physically isolated discovery sources. NEXT284 audits exactly those six fixed directions in the unchanged NEXT224 rejected-extreme cohort. NEXT285 and NEXT286 are conditionally authorized only by the frozen gates below.

**Tech Stack:** Python 3.11, ASE, pymatgen, NumPy, pandas, SciPy `HalfspaceIntersection`/`ConvexHull`/`linprog`, pytest, Parquet, SHA-256 manifests.

## Scientific motivation and alternatives

Ericson, Wolpert, and Poon report that in polydisperse ensembles small generators preferentially receive power cells that are larger and less spherical than their additively weighted counterparts, and identify this coupling as a primary mechanism behind power/additive divergence ([PCCP 27, 16204--16218](https://doi.org/10.1039/D5CP00763A)). NEXT267 already measures power-cell volume allocation and a vertex-covariance anisotropy, but it does not measure true cell surface area, sphericity, or the coupling of cell asphericity to radius-normalized volume inflation.

Three approaches were considered before outcomes:

1. Directly compute both three-dimensional additively weighted and power cells. This is closest to the paper, but curved-face construction has a much larger numerical failure surface and would confound a small prospective test.
2. Measure exact power-cell surface area and test only the preregistered shape--volume mechanism. This is selected because convex power-cell facets are already constructed by the frozen NEXT267 half-space method and `ConvexHull.area` adds no learned or energetic input.
3. Add more contact-graph shape correlations. This is rejected because NEXT279--NEXT280 already found zero transferable preregistered autocorrelation direction.

## Immutable boundary

- Executable inputs are only element identities, neutral tabulated radii, the raw unrelaxed periodic lattice, and raw fractional coordinates.
- No DFT calculation is executed. No per-structure DFT energy, force, stress, band, charge, convergence value, relaxation, trajectory, or relaxed geometry enters a feature or formula.
- No learned energy/force/stress predictor, interatomic potential, proxy potential, or physical relaxation is permitted.
- SCIGEN and WyFormer discovery outcomes are opened only as offline labels after NEXT283 outputs and their identities are frozen.
- Validation and replication geometry, endpoints, and outputs remain sealed.
- Existing scripts, results, report text, and canonical documents are not modified or replaced. Only new plan/source/test/formal-output files and an appended section in the independent report are allowed.
- No canonical `paper/`, `tex/`, `notes/`, `README.md`, or `PREREG.md` file may be changed.

## Frozen cell construction

For each raw site, NEXT283 must reuse the NEXT267 neutral-radius policy, Minkowski-reduced-cell handling, periodic-image bound, lattice Wigner--Seitz guard, radical half-spaces, Chebyshev-interior certificate, Qhull options, feasibility tolerance, volume-tiling tolerance, and output quantization grid. It must not modify or replace NEXT267.

For every nonempty convex power cell with volume `V_i`, hull surface area `A_i`, and tabulated radius `r_i`, define

```text
Psi_i = pi^(1/3) (6 V_i)^(2/3) / A_i
a_i   = 1 - Psi_i
u_i   = log(V_i / ((4 pi / 3) r_i^3))
m_u   = median_i(u_i)
p_i   = max(u_i - m_u, 0)
```

`Psi_i` must be finite and in `[0, 1]` up to a frozen numerical tolerance; values within tolerance are clipped. The median uses NumPy's deterministic default linear interpolation. All structure summaries use the same `1e-10` output grid as NEXT267. A constant population has correlation exactly zero. A one-site periodic structure is supported because these are single-cell, not graph, summaries.

The small-generator subset is `S = {i : r_i < mean_j(r_j)}`. If `S` is empty, its burden is exactly zero. This definition is invariant to site permutation, cell rotation/translation, uniform spatial scale, and integral periodic replication.

## Frozen feature family and directions

Exactly six features are authorized:

```text
psvc_sphericity_mean                         protected_high
psvc_sphericity_q10                          protected_high
psvc_log_volume_asphericity_correlation      protected_low
psvc_inflated_asphericity_mean               protected_low
psvc_inflated_asphericity_q90                protected_low
psvc_small_inflated_asphericity_mean         protected_low
```

The correlation is the population Pearson correlation between `u_i` and `a_i`; it is zero when either centered sum of squares is at most one output-grid unit. The two unrestricted coupling burdens summarize `a_i p_i` by mean and inverted-CDF 0.90 quantile. The small-generator burden is `mean_{i in S}(a_i p_i)`, or zero for empty `S`.

The directions encode the sole preregistered hypothesis: protected structures have more spherical cells and weaker coupling between radius-normalized cell inflation and asphericity. Opposite directions, alternative quantiles, and post-outcome transformations are prohibited in this branch.

## Task 1: NEXT283 tests first

**Files:**
- Create: `tests/test_next283_power_cell_shape_volume_coupling.py`
- Create later: `src/next283_power_cell_shape_volume_coupling.py`

1. Write failing tests for the exact six-name/direction schema and analytic sphere/box sphericity helper behavior.
2. Run `python -m pytest tests/test_next283_power_cell_shape_volume_coupling.py -q` and confirm failure because NEXT283 does not exist.
3. Add failing tests for constant-correlation zero, analytic coupling summaries, malformed geometry, and exact builder boundary parameters.
4. Add failing representation tests for rigid rotation, uniform scale, periodic translation, site permutation, unimodular lattice rebasing, and integral periodic replication.
5. Implement only enough NEXT283 code to pass, without changing any prior file.
6. Re-run the focused test until green.

## Task 2: NEXT283 formal discovery materialization

**Files:**
- Create: `src/next283_power_cell_shape_volume_coupling.py`
- Create externally: `$PRIS_ARCHIVE/next283_power_cell_shape_volume_coupling_v1/`

1. Freeze the plan SHA-256 in the source before running either discovery source.
2. Verify the exact formal SCIGEN/WyFormer geometry-cohort hashes and upstream source hashes.
3. Run NEXT283 on discovery geometry only with the `newpauling` environment.
4. Verify row identity, finite/support equivalence, positive surface area, sphericity bounds, NEXT267 volume-tiling tolerance, boundary flags, output hashes, and executed-source hashes.
5. Do not open any endpoint while materializing features.

## Task 3: NEXT284 prospective raw-feature audit

**Files:**
- Create: `tests/test_next284_power_cell_shape_volume_feature_audit.py`
- Create: `src/next284_power_cell_shape_volume_feature_audit.py`
- Create externally: `$PRIS_ARCHIVE/next284_power_cell_shape_volume_feature_audit_v1/`

1. Write and run failing tests for exactly the six frozen directions, no validation/replication interface, identity alignment, bounded protection, deterministic selection, and fail-closed missing input.
2. Implement the audit by reusing the unchanged NEXT268/NEXT227 reconstruction and gates.
3. Use combined finite discovery `1/16` and `15/16` inverted-CDF normalization, the same five reduced-formula folds, source aggregate/macro/worst-fold AUC gates, class counts, and cell coverage.
4. Freeze sorted eligible identities and their SHA-256.
5. If no direction passes all fixed gates in both sources, terminate the branch and do not create NEXT285/NEXT286.

## Task 4: Conditional NEXT285 local search

**Files, only if NEXT284 authorizes:**
- Create: `tests/test_next285_power_cell_shape_volume_margin_local_search.py`
- Create: `src/next285_power_cell_shape_volume_margin_local_search.py`
- Create externally: `$PRIS_ARCHIVE/next285_power_cell_shape_volume_margin_local_search_v1/`

1. Copy no adaptive choices: reuse exactly the NEXT269/NEXT273/NEXT277 seven width fractions, three nonnegative amplitude fractions, one NEXT224 reproduction control, normalization, triangular term, folds, source AUC gates, twelve SAFE cells, and BROAD gates.
2. Search only the eligible NEXT284 identities and frozen directions.
3. A candidate passes only if every existing cross-source discovery gate passes.
4. Authorize NEXT286 only for exact new candidate identities that pass both source-AUC gates and all SAFE cells but fail BROAD.

## Task 5: Conditional NEXT286 BROAD diagnostic

**Files, only if NEXT285 authorizes:**
- Create: `tests/test_next286_power_cell_shape_volume_broad_diagnostic.py`
- Create: `src/next286_power_cell_shape_volume_broad_diagnostic.py`
- Create externally: `$PRIS_ARCHIVE/next286_power_cell_shape_volume_broad_diagnostic_v1/`

1. Reproduce only authorized NEXT285 records under unchanged BROAD thresholds.
2. Rank by `(failed_constraint_count, normalized_shortfall_sum, candidate_key)`.
3. Compare with frozen NEXT270 reference `(5, 0.0955435292756307)`.
4. Perform no new feature, direction, threshold, coefficient, or formula search.
5. Do not open validation or replication output.

## Task 6: Verification and independent report

**File:**
- Append only: `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`

1. Run all newly added focused tests.
2. Run the complete repository suite.
3. Verify plan/source/test/formal-output hashes and every `MANIFEST.json` output hash.
4. Verify the no-DFT/no-proxy/no-relaxation flags and sealed validation/replication flags.
5. Check CodeGraph status after all edits and read only any explicitly pending files.
6. Append a conservative NEXT283--NEXT286 section to the independent report; do not modify canonical research documents.
7. Report negative or partial results as such. Do not claim a new law or scientific improvement unless the frozen gates actually establish it.

## Execution note

The active checkout is intentionally used because the user's long-running research state and external formal artifacts are already tied to it. The explicit scope forbids commits, branch cleanup, canonical edits, or replacement of prior files, so the worktree/commit steps suggested by generic development workflows are not applicable here.
