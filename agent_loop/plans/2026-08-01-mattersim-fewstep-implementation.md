# MatterSim Few-Step Pre-Relaxation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an additive, leakage-auditable fixed-cell MatterSim FIRE trajectory experiment and determine whether it improves the frozen x0 baseline before any DFT relaxation.

**Architecture:** A label-free generator sanitizes x0 frames, advances independent ASE FIRE states through one batched MatterSim force call per snapshot, and writes only allowlisted trajectory summaries. A separate development command selects one of six frozen gap formulas and seals its threshold before a retrospective evaluator is allowed to open the historically seen test split.

**Tech Stack:** Python 3.11, pandas/pyarrow, NumPy, ASE, MatterSim 1.2.3, PyTorch/CUDA, pytest, SHA-256 manifests.

**Shared-checkout note:** The prior research chain is untracked in the current checkout and the new modules import it. Creating a clean worktree or committing next7 alone would produce a broken partial history, so this plan uses additive filenames and hash manifests instead of Git commits.

### Task 1: Frozen trajectory statistics

**Files:**
- Create: `tests/test_next7_mattersim_prerelax.py`
- Create: `src/next7_mattersim_prerelax.py`

1. Write failing tests for snapshot validation, force/stress summaries, minimum-image displacement, atom-budget packing, and dangerous trajectory abstention.
2. Run `pytest tests/test_next7_mattersim_prerelax.py -v` and confirm failure because the module/API is absent.
3. Implement the minimal pure functions and frozen constants.
4. Re-run the focused test and confirm pass.

### Task 2: Label-free batched FIRE engine

**Files:**
- Modify: `tests/test_next7_mattersim_prerelax.py`
- Modify: `src/next7_mattersim_prerelax.py`

1. Add a failing harmonic-potential test proving snapshots are x0/x2/x4/x8, exactly eight updates occur, each move is capped at 0.05 Angstrom, and endpoint labels never enter `Atoms`.
2. Verify RED with the focused pytest command.
3. Implement a predictor-injected fixed-step batch engine using separate ASE FIRE state per structure and a lazy MatterSim `Potential.predict_properties` adapter.
4. Verify GREEN, then refactor without changing behavior.

### Task 3: Additive trajectory artifact generator

**Files:**
- Modify: `tests/test_next7_mattersim_prerelax.py`
- Modify: `src/next7_mattersim_prerelax.py`

1. Add a failing integration test with a tiny extxyz zip and stage assignment table. Require key-based joins, non-x0 abstention, stage allowlisting, unique sid, runtime/call accounting, parquet output, and hashes.
2. Verify RED.
3. Implement `run_fewstep_features(...)` and CLI arguments for explicit stages, checkpoint, device, atom budget, and structure chunk size.
4. Verify GREEN and ensure no labels path exists in this module's API.

### Task 4: Development-only finite candidate selection

**Files:**
- Create: `tests/test_next7_fewstep_protocol.py`
- Create: `src/next7_fewstep_protocol.py`

1. Write failing tests for the six and only six formulas, same-composition gaps, fail-open support, development-stage-only selection, deterministic tie-breaking toward lower cost, and frozen threshold serialization.
2. Verify RED.
3. Implement development selection for the strict primary track and historical comparison track. Reject any input containing test-stage rows.
4. Verify GREEN and hash `FROZEN_PROTOCOL.json` plus development frontier artifacts.

### Task 5: Frozen retrospective evaluator and paired comparison

**Files:**
- Create: `tests/test_next7_fewstep_evaluate.py`
- Create: `src/next7_fewstep_evaluate.py`

1. Write failing tests requiring a matching frozen protocol/checkpoint/input hash, a new opening log explicitly marked `historically seen discovery`, fixed threshold application, and composition-paired bootstrap versus S0.
2. Verify RED.
3. Implement the minimal evaluator and cost accounting. It must never select step, formula, alpha, or threshold from test labels.
4. Verify GREEN.

### Task 6: Development inference and freeze

**Files:**
- Create: `outputs/20260801_mattersim_fewstep/development_features/*`
- Create: `outputs/20260801_mattersim_fewstep/development_freeze/*`

1. Run a 32-structure label-free CUDA smoke test and verify x0 energy agreement, finite forces/stresses, exact call counts, and memory/runtime.
2. Generate only the three development stages.
3. Run candidate selection once and write the frozen protocol.
4. Check all declared hashes before allowing test inference.

### Task 7: One retrospective test opening

**Files:**
- Create: `outputs/20260801_mattersim_fewstep/test_features/*`
- Create: `outputs/20260801_mattersim_fewstep/retrospective_evaluation/*`

1. Generate test trajectories only after the freeze artifact exists.
2. Apply frozen decisions once; do not rerun candidate selection.
3. Compute strict primary metrics, the historical comparator, paired bootstrap intervals, suffix/chemistry diagnostics, and MLIP cost.
4. Verify every manifest hash and preserve raw outputs.

### Task 8: Verification and standalone report

**Files:**
- Create: `reports/2026-08-01-mattersim-fewstep-followup.md`

1. Run focused tests, then the full suite.
2. Check CodeGraph has no pending sync and inspect the new entry points.
3. Programmatically verify output SHA-256 declarations.
4. Write a standalone evidence-first report. State explicitly whether the improvement gate passed; do not modify the canonical paper/report/README.
