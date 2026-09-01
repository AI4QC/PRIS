# NEXT101 Discrete Oxidation-State Bond-Valence Realizability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an additive, deterministic, no-DFT descriptor that asks whether any chemically admissible charge-neutral integer oxidation-state assignment is geometrically realizable by the raw structure's periodic bond-valence network.

**Architecture:** NEXT101 enumerates the complete bounded set of element-uniform, charge-neutral integer oxidation-state assignments from frozen pymatgen common and ICSD tables. Each assignment is evaluated independently on the unchanged raw structure with the existing analytic periodic-neighbor and scale-calibrated bond-valence machinery; the new descriptor summarizes the best attainable mismatch, the runner-up gap, ensemble robustness, assignment ambiguity, and parameter provenance. Unsupported structures abstain. Mixed-valence site allocation is a separately frozen second-stage extension and is not introduced after viewing discovery labels.

**Tech Stack:** Python 3.11 from `<env>`, NumPy, pymatgen 2026.5.4 oxidation-state tables, existing `src.next19_valence_transport` periodic geometry, existing `src.next22_bond_valence_equilibrium`, pytest.

## Scientific boundary and alternatives

The execution input is exactly one raw, unrelaxed structure. Element identities, lattice, periodic coordinates, frozen tabulated oxidation states, deterministic CrystalNN/Voronoi geometry, and analytic bond-valence parameters are allowed. DFT values or calculations, relaxed structures or trajectories, learned energies/forces/stresses, MLIP/MatterSim calls, and same-composition alternatives are forbidden. Offline DFT outcomes remain labels used only after feature materialization on physically isolated discovery files.

Three mechanisms were considered:

1. **Formula-only charge neutrality.** Fast and invariant, but has little structural discrimination and is especially weak for composition-conditioned generators.
2. **Complete site-level mixed-valence enumeration.** Most expressive, but combinatorial, hard to make supercell invariant, and likely to create coverage-dependent selection bias.
3. **Complete element-uniform neutral ensemble plus structural bond-valence competition.** Recommended for NEXT101. It is finite, auditable, supercell invariant, and differs from the current single first-guess oxidation-state policy by measuring whether *any* admissible discrete chemical explanation fits the raw geometry and whether that explanation is unique.

NEXT101 therefore freezes option 3 before discovery evaluation. A later NEXT101b may add at most one mixed-valence element using a separately preregistered, permutation-invariant allocation rule, but only after NEXT101 results are archived and without replacing them.

## Frozen v1 contract

- State catalogue per element: sorted union of nonzero finite integer `Element.common_oxidation_states` and `Element.icsd_oxidation_states`. The wider `oxidation_states` table is excluded from v1 to avoid exotic post-hoc explanations.
- Enumeration: exact depth-first enumeration of all element-uniform assignments whose stoichiometric charge is exactly zero and that contain both signs. Element order and state order are canonical. The raw Cartesian product must not exceed 65,536 combinations and the neutral result set must not exceed 512; exceeding either bound causes an explicit abstention, never silent truncation.
- Geometry: `build_periodic_edge_geometry` on the unchanged structure for each distinct charge-sign pattern, cached within one call. No structure relaxation or mutation.
- Per-assignment features: existing `bond_valence_features_from_periodic_geometry` with frozen-fallback parameter provenance.
- Objective ordering: ascending `scbv_mismatch_rms`, then `scbv_mismatch_q95`, then `scbv_mismatch_max`, then lexicographic oxidation-state tuple. No label enters the ordering.
- Fail-open behavior: no neutral assignment, bound overflow, graph failure for every assignment, or missing parameters for every assignment yields `supported=False` and an explicit reason. Partially supported ensembles remain supported while exposing the supported fraction.
- Version provenance: record pymatgen version and a SHA-256 digest of the actual per-element state catalogue used for each structure build.
- v1 feature schema:
  - `dobvr_neutral_assignment_count`
  - `dobvr_supported_assignment_fraction`
  - `dobvr_best_mismatch_rms`
  - `dobvr_best_mismatch_q95`
  - `dobvr_best_mismatch_max`
  - `dobvr_median_mismatch_rms`
  - `dobvr_runner_up_gap_rms`
  - `dobvr_best_parameter_exact_fraction`
  - `dobvr_best_parameter_generic_fraction`
  - `dobvr_best_mean_abs_oxidation`
  - `dobvr_best_max_abs_oxidation`
  - `dobvr_assignment_log_count`

## Task 1: Core contract tests

**Files:**

- Create: `tests/test_next101_discrete_oxidation_bv_realizability.py`
- Create later: `src/next101_discrete_oxidation_bv_realizability.py`

**Step 1: Write the failing import/schema test**

Assert that the new module exports the exact frozen schema, its names contain no forbidden endpoint/model terms, and the protocol identifies uniform discrete oxidation-state ensemble v1.

**Step 2: Run the test and verify RED**

Run:

```bash
python -m pytest tests/test_next101_discrete_oxidation_bv_realizability.py -q
```

Expected: collection fails only because `src.next101_discrete_oxidation_bv_realizability` does not exist.

**Step 3: Add enumeration behavior tests**

Test NaCl for a deterministic `Na(+1), Cl(-1)` member; test Fe2O3 for `Fe(+3), O(-2)`; test an impossible single-element structure for explicit no-neutral-assignment abstention; test exact charge neutrality and deterministic ordering for every returned assignment; test that a 2x supercell yields the same element-uniform assignment catalogue.

**Step 4: Add structure-evaluation behavior tests**

Test raw CsCl without mutation, deterministic repeated output, exact feature schema, finite values, supported fraction in `(0, 1]`, and intensive feature invariance under a 2x supercell except documented count fields. Test a deliberately unsupported oxidation-state catalogue through dependency injection and verify fail-open behavior.

## Task 2: Minimal NEXT101 core implementation

**Files:**

- Create: `src/next101_discrete_oxidation_bv_realizability.py`
- Test: `tests/test_next101_discrete_oxidation_bv_realizability.py`

**Step 1: Implement immutable result objects**

Add `OxidationAssignment`, `OxidationEnumerationResult`, and `DOBVRFeatureResult`. Results carry `supported`, `failure_reason`, assignments/features, catalogue digest, and pymatgen version.

**Step 2: Implement the frozen state catalogue**

Normalize table entries to nonzero Python integers, canonicalize them, and permit a test-only explicit catalogue mapping at the public enumeration/evaluation boundary. Do not add a mutable global override.

**Step 3: Implement bounded exact neutral enumeration**

Use integer site counts and divide them by their greatest common divisor before the neutrality search so primitive and supercell representations produce identical candidates. Precompute product size before search, fail on the frozen bound, and never truncate neutral assignments.

**Step 4: Run focused tests until GREEN**

Run the exact command from Task 1. Fix production code, not assertions, unless the frozen written contract is internally inconsistent.

**Step 5: Implement analytic structure evaluation**

Convert each element assignment to one charge per site, cache periodic geometry by the per-element sign pattern, evaluate with existing NEXT22 bond-valence code, rank by the frozen objective, and compute only the twelve registered aggregate features.

**Step 6: Re-run NEXT101 and dependency regressions**

Run:

```bash
python -m pytest \
  tests/test_next101_discrete_oxidation_bv_realizability.py \
  tests/test_next19_valence_transport.py \
  tests/test_next22_bond_valence_equilibrium.py -q
```

Expected: all pass with no new warnings attributable to NEXT101.

## Task 3: Discovery-only feature materialization

**Files:**

- Create: `src/next102_cross_source_dobvr_features.py`
- Create: `tests/test_next102_cross_source_dobvr_features.py`
- Create external output directory: `$PRIS_ARCHIVE/next102_cross_source_dobvr_features_v1`

**Step 1: Write failing firewall and routing tests**

Require physically isolated SCIGEN discovery and WyFormer discovery inputs. Reject paths or manifests carrying validation/replication roles. Assert the builder imports neither calculators nor endpoint labels while computing features.

**Step 2: Implement an append-only builder**

Reuse source IDs and raw structures only for row routing. Persist DOBVR features, support/failure fields, protocol, table digest, environment versions, source manifest hashes, and `MANIFEST.sha256`. Do not merge into NEXT85 or NEXT94 feature tables.

**Step 3: Smoke-test 10 rows per source**

Measure support and runtime without reading labels. Abort full materialization if the schema/protocol/digest varies unexpectedly or if raw structures mutate.

**Step 4: Materialize only both discovery partitions**

Replication feature materialization is forbidden at this stage, even though features are label-free, to preserve the stronger never-read boundary.

## Task 4: Preregistered cross-source discovery test

**Files:**

- Create: `src/next103_cross_source_dobvr_search.py`
- Create: `tests/test_next103_cross_source_dobvr_search.py`
- Create: `docs/plans/2026-08-04-next103-dobvr-cross-source-search-amendment.md`

**Step 1: Freeze the candidate grammar before labels**

Permit each individual risk-oriented DOBVR term and sparse nonnegative sums of at most two DOBVR terms, optionally added to the already frozen CVR-Risk terms. Thresholds are fitted only within each discovery training fold. Missing DOBVR values abstain and may not be imputed from source identity.

**Step 2: Freeze gates**

Require dual-source AUC improvement or noninferiority, all preregistered SAFE cells, coverage, and source-wise robustness. Do not relax Wilson lower-bound gates after viewing results.

**Step 3: Run discovery and archive all candidates**

Report support coverage separately from conditional ranking quality. A candidate cannot pass by excluding hard structures.

**Step 4: Decide the branch without reopening the contract**

- If a frozen candidate passes every discovery gate, freeze it and authorize feature-only construction on both still-unopened replication partitions, followed by one-shot endpoint evaluation.
- If none passes, do not open replication. Archive NEXT101 as a negative/diagnostic result and preregister NEXT101b mixed-valence allocation before any new label-bearing search.

## Task 5: Verification and independent report

**Files:**

- Create: `reports/2026-08-04-next101-next103-dobvr-no-dft-search.md`
- Do not modify: `paper/`, `notes/`, `tex/`, `README.md`, `PREREG.md`, or prior reports.

**Step 1: Run targeted and full tests**

Run the focused tests first, then:

```bash
python -m pytest -q
```

**Step 2: Verify provenance and boundaries**

Validate all manifests and hashes, inspect CodeGraph pending sync, confirm no forbidden imports or fields in new execution modules, and confirm protected paths are unchanged.

**Step 3: Write the standalone report**

Separate implementation validity, support coverage, discovery evidence, strict failures, and unopened replication status. Do not claim replacement of Pauling rules unless every frozen cross-source and replication gate passes.

No git commit is part of this plan; the current checkout is an additive untracked research workspace and the user requested preservation rather than repository history changes.
