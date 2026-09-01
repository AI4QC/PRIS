# NEXT19 Valence Transport Law Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build, search, freeze, and externally falsify a per-structure crystal plausibility law that uses only raw x0 geometry and analytic crystal-chemistry quantities.

**Architecture:** A pure feature module converts an oxidation-state-labelled periodic cation-anion graph into a sparse linear transport problem and emits dimensionless feasibility scores. A geometry-only batch builder seals identifier-bearing features outside the repository. Separate evaluators join historically exposed WBM/ELEMENTA labels, freeze one qualifying candidate, then evaluate sealed Alexandria predictions without allowing endpoint data into the law path.

**Tech Stack:** Python 3.11, NumPy, pandas, SciPy `linprog`, pymatgen, ASE, pytest, deterministic ZIP/Parquet/JSON manifests.

**Workspace note:** A Git worktree is intentionally not used because 226 user-owned/untracked research artifacts contain required loop code and data contracts while only 151 files are tracked. Every implementation file is additive and every output directory is no-replace. Git commits are not made unless the user explicitly requests them; SHA-256 manifests provide checkpoints.

### Task 1: Lock the no-DFT execution contract

**Files:**
- Create: `tests/test_next19_valence_transport.py`
- Create: `src/next19_valence_transport.py`

**Steps:**
1. Write tests asserting that the public feature API accepts only structures and formal valences, returns no field containing `energy`, `force`, `stress`, `relax`, `mattersim`, or `dft`, and fail-opens unsupported inputs.
2. Run `python -m pytest -q tests/test_next19_valence_transport.py`; expect failure because the module does not exist.
3. Add immutable protocol constants, result dataclasses, forbidden-field validation, and a placeholder public API that validates inputs.
4. Run the focused tests; expect all Task 1 tests to pass.
5. Record `sha256sum src/next19_valence_transport.py tests/test_next19_valence_transport.py` in the working log.

### Task 2: Implement the sparse valence-transport solver with TDD

**Files:**
- Modify: `tests/test_next19_valence_transport.py`
- Modify: `src/next19_valence_transport.py`

**Steps:**
1. Add synthetic graph tests: one perfectly balanced Na-Cl graph gives zero overload/reallocation; an asymmetric bottleneck requires `kappa > 1`; a disconnected supply-demand graph returns unsupported; reordering edges leaves results unchanged.
2. Run the new tests and verify they fail for missing solver behavior.
3. Implement canonical edge sorting, equality constraints, the first `scipy.optimize.linprog` overload solve, and the second L1 reallocation solve.
4. Add numerical checks for charge neutrality, nonnegative priors, equality residuals, solver status, and deterministic tolerances.
5. Run the focused tests and verify exact pass/fail counts.

### Task 3: Build the periodic geometry graph and analytic descriptors

**Files:**
- Modify: `tests/test_next19_valence_transport.py`
- Modify: `src/next19_valence_transport.py`

**Steps:**
1. Add NaCl, CsCl, distorted-NaCl, same-sign-only, and oxidation-state-failure tests using pymatgen structures.
2. Verify the tests fail before implementation.
3. Reuse the repository's unified `guess_oxi` then fractional fallback. If both fail, apply the frozen scale-free Pauling electronegativity partition and record the selected policy. Construct periodic opposite-sign edges using the fixed `CrystalNN` and Voronoi modes, retaining periodic image multiplicity.
4. Compute the fixed `alpha={0,2,4,6}` prior catalogue and all transport descriptors. Never attach a calculator or change coordinates/cell.
5. Add classical baseline descriptors needed for ablation: Pauling-style anion mismatch, minimum contact ratio, and optional existing Madelung sign diagnostics. Keep failures isolated by family.
6. Run focused tests and verify all pass.

### Task 4: Seal geometry-only feature batches

**Files:**
- Create: `tests/test_next19_feature_build.py`
- Create: `src/next19_feature_build.py`

**Steps:**
1. Write tests for exact SID coverage, shuffled metadata rejection, forbidden metadata columns, archive hash closure, deterministic row order, fail-open rows, and atomic no-replace publication.
2. Run the tests and verify failure because the builder is absent.
3. Implement a CLI using `src.next11_geometry_only_frames.load_geometry_only_archive`; inputs are archive, manifest, metadata, source role, and output directory only.
4. Write identifier-bearing Parquet plus a JSON manifest outside the repository. The manifest records source hashes, code hashes, feature-family coverage/failure counts, and explicitly states that endpoint bytes were not read.
5. Run tests, then build WBM and ELEMENTA batches under `$PRIS_ARCHIVE/next19_*`.
6. Validate output hashes and confirm no output column violates the no-DFT contract.

### Task 5: Search the frozen candidate catalogue on development sources

**Files:**
- Create: `tests/test_next19_evaluate.py`
- Create: `src/next19_evaluate.py`

**Steps:**
1. Add tests for Wilson intervals, fixed threshold catalogue scanning, fail-open decisions, complete-group metrics, no-all-rejected gate, source-wise non-refit, Pauling comparison, deterministic cluster bootstrap, and no identifiers in aggregate output.
2. Run tests and verify they fail before implementation.
3. Implement WBM selection using the fixed single-score, two-score, and consensus catalogues. Do not generate new features or thresholds from evaluation residuals.
4. Apply each WBM-eligible candidate unchanged to ELEMENTA and enforce the design gates. Rank only candidates that pass every absolute safety gate; break ties by larger ELEMENTA savings, then simpler formula, then larger safety margin.
5. Produce private joined tables outside the repository and aggregate JSON/manifests in `outputs/20260802_next19_valence_transport_development`.
6. If no candidate passes, stop scientific promotion and prepare a negative report. Do not open Alexandria endpoints.

### Task 6: Freeze one qualifying law

**Files:**
- Create: `tests/test_next19_freeze.py`
- Create: `src/next19_freeze.py`

**Steps:**
1. Write tests that reject a freeze if the development result failed, source hashes differ, a threshold is refit, forbidden model/energy fields appear, or an output already exists.
2. Run tests and verify they fail before implementation.
3. Implement a no-label freeze command producing `FROZEN_PROTOCOL.json` and `MANIFEST.json` in a new output directory.
4. Bind the exact graph mode, alpha, descriptors, formula, threshold, valence inference policy, fail-open behavior, source/code hashes, and development result hash.
5. Run tests and validate freeze hashes.

### Task 7: Produce sealed Alexandria predictions before endpoints

**Files:**
- Modify: `tests/test_next19_feature_build.py`
- Modify: `src/next19_feature_build.py`
- Create: `tests/test_next19_external_predict.py`
- Create: `src/next19_external_predict.py`

**Steps:**
1. Test that prediction accepts only a frozen protocol plus geometry-only features and refuses any endpoint-like column or changed threshold.
2. Build Alexandria analytic features from the 379-row geometry-only archive without accessing raw shard endpoint fields.
3. Apply the frozen law, write sealed predictions outside the repository, and record exact feature/prediction/freeze hashes.
4. Verify decision counts, coverage, no-replace behavior, and absence of forbidden columns.

### Task 8: Open Alexandria endpoints once and evaluate externally

**Files:**
- Create: `tests/test_next19_alexandria_evaluate.py`
- Create: `src/next19_alexandria_evaluate.py`

**Steps:**
1. Write fixture tests for streaming extraction of only the final DFT energy and necessary convergence fields after sealed prediction validation.
2. Require the exact two source shard hashes, 379 selected IDs, frozen protocol hash, and sealed prediction hash before any endpoint extraction.
3. Extract endpoint labels to a private external directory, derive same-composition DFT regret, and evaluate NEXT19 and Pauling with the frozen gates and group bootstrap.
4. Write only aggregate results/manifests inside the repository. Record that the raw containers contain endpoint bytes and were read at this stage.
5. Never modify the law after observing this result. A failure is final for NEXT19.

### Task 9: Verification and standalone report

**Files:**
- Create after results: `reports/2026-08-02-next19-valence-transport-law.md`

**Steps:**
1. Run all focused NEXT19 tests, then `python -m pytest -q` for the full repository.
2. Run `git diff --check` and validate every new JSON, Parquet schema, ZIP, SHA-256 manifest, and no-replace contract.
3. Confirm existing reports and canonical paper paths were not modified by NEXT19.
4. Write an independent report covering formula, literature boundary, data access, all fixed gates, WBM/ELEMENTA development results, Alexandria external result if legitimately opened, negative results, and limitations.
5. Stop before modifying any canonical report or paper; wait for explicit user approval for integration.
