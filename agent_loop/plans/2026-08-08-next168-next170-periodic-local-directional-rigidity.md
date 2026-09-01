# NEXT168--NEXT170 Periodic Local Directional Rigidity Implementation Plan

> **For Codex:** REQUIRED SUB-SKILLS: Use `superpowers:test-driven-development` for every new implementation and `superpowers:verification-before-completion` before reporting a result.

**Goal:** Test whether a parameter-free, local directional-rigidity certificate supplies the missing no-DFT information needed to repair the frozen NEXT164 shell on both SCIGEN and WyFormer discovery sources.

**Architecture:** NEXT168 reuses only the frozen discovery geometries, formal-valence assignment, and the two NEXT19 periodic opposite-sign contact graphs. For every site, it forms the unweighted second moment of incident unit contact directions and summarizes the weakest local 3D support. NEXT169 audits ten direction-fixed single-feature hypotheses against the already frozen discovery outcomes. NEXT170 is authorized only if at least one NEXT169 hypothesis passes every prespecified cross-source gate.

**Tech Stack:** Python 3.11 from `<env>`, NumPy, pandas, pymatgen, pytest, Parquet, SHA-256 manifests.

## Scientific boundary and alternatives

- The executable features use initial geometry, composition-derived formal valence, and analytic neighbor graphs only. They do not use DFT values, calculators, learned energy/force/stress proxies, relaxation, endpoint payloads, or validation/replication geometry.
- Discovery outcomes may be opened only by NEXT169, after the NEXT168 feature schema and the NEXT169 hypothesis directions and gates are frozen.
- A global Maxwell rank/nullity branch is rejected because NEXT37 already reports the rank and cokernel of an atomic-plus-affine compatibility matrix.
- A global singular-spectrum branch is deferred because its spectrum changes when the same crystal is represented by a supercell.
- Local directional rigidity is selected because it is an intensive, supercell-invariant property and distinguishes locally linear or planar coordination even when NEXT167 says the periodic graph is globally three-dimensional.

## Frozen NEXT168 definition

For site \(i\), let \(u_{ie}\) be the unit Cartesian direction of every incident edge in one frozen NEXT19 graph. Define

\[
G_i = \frac{1}{d_i}\sum_{e\ni i}u_{ie}u_{ie}^{\mathsf T},
\]

with \(G_i=0\) for an isolated site. If \(\lambda_{i,1}\leq\lambda_{i,2}\leq\lambda_{i,3}\) are its eigenvalues, define bounded per-site certificates

\[
T_i=3\lambda_{i,1},\qquad V_i=27\det(G_i).
\]

Both lie in \([0,1]\): zero means a missing local direction (isolated, linear, or planar), and one is an isotropic local frame. Edge weights, radii, fitted coefficients, outcome-derived cutoffs, and SVD tolerances are not used. Eigenvalue clipping is allowed only inside a fixed `1e-12` roundoff guard; values outside that guard fail open.

For each graph mode `voronoi` and `crystalnn`, freeze exactly five features:

1. `pldr_<mode>_tightness_min`
2. `pldr_<mode>_tightness_q10`
3. `pldr_<mode>_tightness_mean`
4. `pldr_<mode>_volume_q10`
5. `pldr_<mode>_volume_mean`

The 0.10 quantile uses NumPy's deterministic `inverted_cdf` rule and aggregates all crystallographic sites, including isolated sites. Each graph mode fails open independently.

### Task 1: Write NEXT168 kernel tests first

**Files:**

- Create: `tests/test_next168_periodic_local_directional_rigidity.py`
- Create after the red run: `src/next168_periodic_local_directional_rigidity.py`

**Steps:**

1. Test that three orthogonal periodic directions between two sites give all five certificates equal to one.
2. Test that a planar frame gives zero weakest-direction and volume certificates.
3. Test invariance to rigid rotation, edge reversal, edge ordering, and uniform duplication of the complete edge set.
4. Test exact frozen schema, finite `[0,1]` bounds, and independent fail-open rows.
5. Test a real rocksalt conventional cell against a `2x1x1` supercell in both graph modes.
6. Run `python -m pytest tests/test_next168_periodic_local_directional_rigidity.py -q`; expect failure because the module does not yet exist.

### Task 2: Implement and build NEXT168 discovery-only features

**Files:**

- Create: `src/next168_periodic_local_directional_rigidity.py`
- Test: `tests/test_next168_periodic_local_directional_rigidity.py`

**Steps:**

1. Implement a pure `local_directional_rigidity_features` kernel with strict shape, finiteness, endpoint, distance, schema, and numerical-bound checks.
2. Implement `compute_periodic_local_directional_rigidity` by reusing `infer_valence_assignment` and `build_periodic_edge_geometry` for both frozen graph modes.
3. Reuse the NEXT166 discovery-only loaders and provenance checks without accepting any endpoint or validation/replication path.
4. Publish atomically to `$PRIS_ARCHIVE/next168_periodic_local_directional_rigidity_v1` with catalogue, separate SCIGEN/WyFormer Parquet files, and `MANIFEST.json` containing input, source, and output hashes plus explicit false boundary flags.
5. Record support, unique counts, minima, maxima, and quantiles without opening labels.
6. Run the test file and the full directed NEXT142--NEXT168 suite.

## Frozen NEXT169 audit

Freeze exactly ten hypotheses before opening discovery outcomes: each of the five NEXT168 features in each graph mode has direction `high`, meaning more complete local 3D support predicts a more reasonable structure. No `low` direction, transform, combination, threshold, family split, or post-hoc exception is permitted.

A hypothesis is eligible only if all conditions hold:

1. full support is at least 90% in both discovery sources;
2. SCIGEN repair-shell AUC is at least 0.55 in every one of the five already frozen folds;
3. WyFormer repair-shell pooled AUC is at least 0.55;
4. full-discovery pooled AUC is at least 0.50 in both sources.

The repair shell and closest frozen score must be read exactly from NEXT164. Validation and replication endpoints remain unopened.

### Task 3: TDD and run NEXT169

**Files:**

- Create: `tests/test_next169_periodic_local_directional_rigidity_audit.py`
- Create after the red run: `src/next169_periodic_local_directional_rigidity_audit.py`

**Steps:**

1. Test that the hypothesis schema contains exactly ten high-direction entries.
2. Test deterministic gate evaluation and ranking, including a synthetic all-pass and one-gate-fail case.
3. Run the tests red, implement the smallest audit, and run green.
4. Publish atomically to `$PRIS_ARCHIVE/next169_periodic_local_directional_rigidity_audit_v1` with all identities and boundary flags.
5. If zero hypotheses are eligible, mark this branch terminated and do not run NEXT170.

## Conditional NEXT170 search

NEXT170 may be designed and executed only if NEXT169 has at least one eligible hypothesis. Its finite grammar may use only the eligible, direction-fixed NEXT168 terms as a bounded guard or monotone correction to the frozen NEXT163/NEXT164 base. It must preserve the existing source-AUC and SAFE12 gates, pass the BROAD repair gates on both discovery sources, and remain no-DFT at execution. If NEXT169 has no eligible hypothesis, write the negative result to the standalone report and move to a different physical mechanism rather than tuning this branch.

### Task 4: Report and verification

**Files:**

- Modify only: `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`

**Steps:**

1. Append the NEXT168 feature definition, provenance, support, nondegeneracy, and artifact hashes.
2. Append the NEXT169 gates, complete audit result, and explicit NEXT170 authorization or termination decision.
3. Do not edit `paper/`, `tex/`, `notes/`, `README.md`, or `PREREG.md`.
4. Recompute all output hashes, verify manifest identities and false boundary flags, run the full directed tests, confirm balanced Markdown fences, check CodeGraph pending sync, and inspect scoped Git status.

