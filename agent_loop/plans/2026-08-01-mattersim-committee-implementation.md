# MatterSim Committee Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and evaluate an additive, frozen 1M/5M MatterSim committee for DFT pre-screening without modifying any existing script or canonical research artifact.

**Architecture:** A label-free feature stage predicts x0 energies and forces with both fixed checkpoints in one pinned runtime. A separate development protocol constructs a finite catalog of same-composition consensus scores, performs three-stage selection/calibration, and atomically freezes one candidate before any historical test application.

**Tech Stack:** Python 3.12, PyTorch 2.13.0 CUDA 13.0, MatterSim 1.2.3, ASE, pandas/pyarrow, NumPy/SciPy, pytest.

### Task 1: Dual-checkpoint label-free feature contract

**Files:**
- Create: `tests/test_next8_mattersim_committee_features.py`
- Create: `src/next8_mattersim_committee_features.py`

1. Write failing tests for strict stage allowlists, exact `sid/rk` preservation, duplicate-key rejection, endpoint-label stripping, unsupported-row ABSTAIN, and no-overwrite atomic publication.
2. Run `pytest tests/test_next8_mattersim_committee_features.py -v` and confirm failure because the module is absent.
3. Implement the smallest model-agnostic predictor interface returning per-model energy, Fmax and Frms while never importing endpoint labels.
4. Run the focused tests and confirm they pass.
5. Add failing tests for runtime/checkpoint/source/input hashes, prediction/evaluation counts and manifest output hashes.
6. Implement those manifest fields and re-run focused tests.

### Task 2: Real MatterSim 1M/5M adapters

**Files:**
- Modify: `tests/test_next8_mattersim_committee_features.py`
- Modify: `src/next8_mattersim_committee_features.py`

1. Write failing tests around injected fake calculators for two-model ordering, batch fallback, non-finite outputs and per-model errors.
2. Verify the tests fail for the intended missing behavior.
3. Implement lazy MatterSim adapters using the frozen checkpoint paths; keep package imports out of the no-MatterSim test environment.
4. Verify focused tests and `py_compile` pass.
5. Run a 32-row GPU smoke test under the pinned `uv run` environment and verify exact counts and hashes.

### Task 3: Finite committee score catalog

**Files:**
- Create: `tests/test_next8_mattersim_committee_protocol.py`
- Create: `src/next8_mattersim_committee_protocol.py`

1. Write failing tests for the exact eleven formulas `M5/M1/MIN/MEAN/MAX/LCB/AGREE99/AGREE995/AGREE_EF995/CMEAN/CMEAN_JOINT99` in that order, group-local gaps, energy q99/q99.5 and force-disagreement q99.5 derivation from search-calibration only, and formula serialization without labels or forbidden identifiers. `CMEAN` must first form `0.5*g1+0.5*g5` and then re-zero it over the full joint-complete `rk`; it must not alias `MEAN` when model argmins differ.
2. Run the focused test and observe the expected missing-module failure.
3. Implement only the frozen catalog and label-free score construction.
4. Re-run focused tests.
5. Write failing tests for formula-specific incomplete groups (M5 must not depend on M1 and vice versa), model failures, finite derived arithmetic, row-local energy/force disagreement ABSTAIN without changing a complete group's gap minimum, and equality-at-threshold KEEP semantics. Add RED tests for `CMEAN_JOINT99`: row-weighted/right-continuous three-marginal ECDFs from search-calibration joint-complete rows only; `J=max(H_E,H_Fmax,H_Frms)`; q99 `method=higher`; exact empirical tail bound; serialization of `n`, `n_rk`, ECDF semantics, all sorted references and hashes; and application to later stages without refitting.
6. Implement fail-open semantics and verify tests pass.

### Task 4: Leakage-controlled development freeze

**Files:**
- Modify: `tests/test_next8_mattersim_committee_protocol.py`
- Modify: `src/next8_mattersim_committee_protocol.py`

1. Write failing tests proving search-calibration, formula-selection and threshold-calibration roles cannot be interchanged or include test.
2. Before reading labels, split complete `threshold_calibration` rk groups deterministically with frozen salt into disjoint `threshold_fit` and `development_gate`; test exact disjointness, completeness and minimum calibration size.
3. Add tests proving provisional thresholds come only from search-calibration, the primary winner only from formula-selection, final selected/M5 thresholds only from threshold-fit, and all +3 pp/paired-CI/recall/abstention gates only from development-gate. Count the 99-group minimum independently for each formula/track using only groups with at least one `protected & supported & finite(score)` row; an insufficient primary rule must keep all supported rows while unsupported/nonfinite rows still ABSTAIN.
4. Add tests for the primary safety gates, same-formula comparator audit, deterministic cost/complexity/catalog tie-break order and formula-specific M5 baseline support.
5. Add tests for atomic `renameat2(RENAME_NOREPLACE)` publication, prepublish rehash and complete input/model/code hash closure. Before any label hash or parquet access, require a production-eligible feature manifest with the exact formal evidence role, builtin adapter mode, verified implementation source and matching executed-source path/SHA; injected test-double artifacts must fail closed. Hash and parse feature parquet, feature manifest and labels from one immutable snapshot per input, create the label snapshot only after feature validation/cutoff/split, and test path/symlink swap-and-restore so computed bytes cannot differ from manifest-bound bytes.
6. Implement the minimum selection/calibration/freeze path.
7. Run focused tests, then all next8 tests.

### Task 5: Development feature generation and selection

**Files:**
- Create: `outputs/20260801_mattersim_committee/smoke32/*`
- Create: `outputs/20260801_mattersim_committee/development_features/*`
- Create: `outputs/20260801_mattersim_committee/development_freeze/*`

1. Record hashes of x0 frames, stage assignments, metadata, source files and both checkpoints before inference.
2. Run the 32-row smoke test in the pinned environment; verify outputs before full inference.
3. Run both checkpoints on development stages only.
4. Verify row/key/stage counts, strict JSON, all manifest hashes and source hashes.
5. Run the frozen leakage-controlled protocol once and publish the freeze directory without overwrite.
6. Evaluate the development improvement gate only on the frozen `development_gate` subset. If it fails, do not open OMat24 payload or add models.

### Task 6: Conditional historical falsification

**Files:**
- Create: `tests/test_next8_mattersim_committee_evaluate.py`
- Create: `src/next8_mattersim_committee_evaluate.py`
- Optionally create: `outputs/20260801_mattersim_committee/historical_test/*`

1. Only if Task 5 passes the development magnitude gate, write failing tests for strict frozen-hash validation, stage=`test`, one-shot application and composition-paired bootstrap.
2. Implement the evaluator after observing the tests fail.
3. Generate label-free test predictions before reading labels; record a provenance opening artifact.
4. Apply the frozen candidate once and label the evidence `historically seen discovery; not confirmatory`.
5. Never use the result to change formulas, thresholds, checkpoints or success gates.

### Task 7: External evidence routing

**Files:**
- Create: `outputs/20260801_mattersim_committee/external_source_audit/*`

1. Hash and describe Bartel `matgen_baselines` commit and files, explicitly recording that CSV labels were exposed and CIFs appear DFT-relaxed.
2. Keep OMat24 validation payload unopened unless Task 5 passes; if opened later, freeze the exact record sample and error metrics first.
3. Record Alexandria 2025/new-generator x0→DFT as the required confirmatory route, not as a completed result.

### Task 8: Independent report and verification

**Files:**
- Create: `reports/2026-08-01-mattersim-committee-followup.md`

1. Write an evidence-first report separating engineering validation, development selection, historical falsification and missing confirmatory evidence.
2. Run focused tests, full `pytest -q`, `py_compile`, strict JSON checks and all manifest hash verifiers.
3. Confirm `git diff --check` and no tracked/canonical protected paths changed.
4. Obtain an independent read-only scientific/protocol review and correct all P1/P2 issues.
5. Stop before modifying `paper/`, `notes/`, `tex/`, `README.md`, `PREREG.md` or existing reports; wait for user confirmation.
