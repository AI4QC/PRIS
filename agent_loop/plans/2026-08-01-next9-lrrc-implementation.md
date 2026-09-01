# Next9 LRRC-v0 Synthetic Protocol Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a label-free, synthetic-only LRRC-v0 negative-curvature prototype and a separate Quota-CRC safety layer without changing any existing script, report, paper, README, or preregistration.

**Architecture:** `src/next9_lrrc.py` contains small pure functions for geometry scaling, translation-projected force directions, two-scale directional curvature, decision composition, and group quota enforcement. `src/next9_lrrc_synthetic.py` contains deterministic analytic force oracles and an atomic JSON-manifest CLI; it never imports MatterSim or reads scientific labels. Tests establish the numerical and failure contracts before implementation.

**Tech Stack:** Python 3.11, NumPy, ASE, pytest, JSON, SHA-256.

**Isolation note:** The current research checkout contains user-owned untracked next1-next8 work that is not present in Git, so a fresh worktree would silently omit required context. Use additive `next9_*` filenames in the current checkout, never stage or commit unrelated files, and do not commit unless the user explicitly asks.

### Task 1: Freeze public data and method contracts

**Files:**
- Create: `docs/plans/2026-08-01-next9-lrrc-design.md`
- Create: `docs/plans/2026-08-01-next9-lrrc-implementation.md`

**Step 1: Record the three compared routes**

Document Quota-CRC as policy-only, LRRC-v0 as the recommended orthogonal signal, and heterogeneous conformal lower bounds as future work.

**Step 2: Record the no-label boundary**

State that old development labels, historical test, OMat24 payload, and existing canonical documents remain unopened/unchanged.

**Step 3: Verify paths are additive**

Run: `git status --short -- docs/plans src/next9_lrrc.py src/next9_lrrc_synthetic.py tests/test_next9_lrrc.py tests/test_next9_lrrc_synthetic.py`

Expected: only the two new plan files before code starts.

### Task 2: Specify direction and geometry scaling with RED tests

**Files:**
- Create: `tests/test_next9_lrrc.py`
- Create: `src/next9_lrrc.py`

**Step 1: Write failing direction tests**

Add tests requiring `translation_projected_direction(forces)` to return a zero-mean direction with unit per-atom RMS, to return `None` below the frozen `1e-12` force floor, and to reject non-finite or wrong-shaped forces.

**Step 2: Write failing length-scale tests**

Add tests requiring `median_nearest_neighbor_distance(atoms)` to use minimum-image distances, remain invariant under periodic wrapping and atom permutation, and fail for fewer than two atoms or no positive finite distance.

**Step 3: Run the focused tests and verify RED**

Run: `pytest -q tests/test_next9_lrrc.py -k 'direction or nearest_neighbor'`

Expected: collection/import failure because `src.next9_lrrc` does not exist.

**Step 4: Implement the minimal pure functions**

Create frozen constants:

```python
LRRC_VERSION = "LRRC-v0"
STEP_FRACTION = 2.0 ** -8
FORCE_RMS_FLOOR = 1.0e-12
```

Implement validated NumPy arrays and ASE minimum-image distances. Do not add data-derived tolerances.

**Step 5: Run the focused tests and verify GREEN**

Run: `pytest -q tests/test_next9_lrrc.py -k 'direction or nearest_neighbor'`

Expected: all selected tests pass.

### Task 3: Specify the two-scale LRRC evaluator with RED tests

**Files:**
- Modify: `tests/test_next9_lrrc.py`
- Modify: `src/next9_lrrc.py`

**Step 1: Write analytic positive-curvature tests**

Use a deterministic isotropic quadratic force oracle `F=-k(x-x0)` and require `kappa_h`, `kappa_h2`, and `u_num` to equal positive `k` within numerical tolerance.

**Step 2: Write analytic negative-curvature tests**

Use `F=+k(x-x0)` at a nonzero displacement and require both finite-difference curvatures and `u_num` to be negative, with `negative_curvature=True`.

**Step 3: Write invariance tests**

Require the scalar results to be invariant under global translation, an orthogonal rigid rotation applied consistently to structure/oracle, atom permutation, and periodic wrapping.

**Step 4: Write failure-state tests**

Require stable status codes for oracle exceptions, non-finite forces, wrong force shape, unsupported geometry, and stationary projected force. The exact stationary saddle must produce `STATIONARY_FALLBACK`, not a false certificate.

**Step 5: Run the evaluator tests and verify RED**

Run: `pytest -q tests/test_next9_lrrc.py -k 'curvature or invariant or failure or stationary'`

Expected: failures because the evaluator/result type is not implemented.

**Step 6: Implement the minimal evaluator**

Add frozen dataclasses/enums and exactly four perturbed oracle calls (`+h`, `-h`, `+h/2`, `-h/2`). Compute:

```python
kappa_r = (4.0 * kappa_h2 - kappa_h) / 3.0
error_proxy = abs(kappa_h2 - kappa_h) / 3.0
u_num = kappa_r + error_proxy
negative = kappa_h < 0.0 and kappa_h2 < 0.0 and u_num < 0.0
```

Do not call the unperturbed oracle more than once and do not add a real checkpoint adapter.

**Step 7: Run the evaluator tests and verify GREEN**

Run: `pytest -q tests/test_next9_lrrc.py -k 'curvature or invariant or failure or stationary'`

Expected: all selected tests pass.

### Task 4: Specify decision composition and Quota-CRC with RED tests

**Files:**
- Modify: `tests/test_next9_lrrc.py`
- Modify: `src/next9_lrrc.py`

**Step 1: Write OR-composition tests**

Cover baseline KEEP + negative LRRC -> REJECT, baseline REJECT + nonnegative LRRC -> REJECT, successful nonnegative LRRC -> baseline decision, evaluator failure -> ABSTAIN, and stationary fallback -> baseline decision.

**Step 2: Write quota tests**

For each supported finite group require `k=ceil(sqrt(n))`, force the lowest `k` scores to KEEP, retain every tie at the kth boundary, leave ABSTAIN unchanged, and prove the final rejection set is a subset of the input rejection set.

**Step 3: Run focused tests and verify RED**

Run: `pytest -q tests/test_next9_lrrc.py -k 'decision or quota'`

Expected: failures because decision/quota functions do not exist.

**Step 4: Implement minimal deterministic functions**

Use explicit enum values `KEEP`, `REJECT`, and `ABSTAIN`. Reject duplicate row identifiers and ambiguous/non-finite scores rather than silently ordering them.

**Step 5: Run focused tests and verify GREEN**

Run: `pytest -q tests/test_next9_lrrc.py -k 'decision or quota'`

Expected: all selected tests pass.

### Task 5: Add a deterministic synthetic manifest runner

**Files:**
- Create: `tests/test_next9_lrrc_synthetic.py`
- Create: `src/next9_lrrc_synthetic.py`
- Create at runtime: `outputs/20260801_lrrc_synthetic/MANIFEST.json`

**Step 1: Write failing runner tests**

Require a fixed case order, strict JSON without NaN/Infinity, source SHA-256, frozen constants/formulas, per-case expected/observed status and sign, overall `engineering_pass`, and the explicit string `scientific_improvement_claim: false`.

**Step 2: Write atomic-publication tests**

Require creation in a temporary sibling directory, prepublication rehash, and no overwrite of an existing output directory.

**Step 3: Run runner tests and verify RED**

Run: `pytest -q tests/test_next9_lrrc_synthetic.py`

Expected: import failure because the runner does not exist.

**Step 4: Implement the runner**

Include positive quadratic, negative quadratic, stationary saddle, invariance, oracle failure, OR decision, quota/tie cases, plus `known_limitations`. The CLI accepts only `--output-dir`; it reads no external dataset.

**Step 5: Run runner tests and verify GREEN**

Run: `pytest -q tests/test_next9_lrrc_synthetic.py`

Expected: all tests pass.

**Step 6: Produce the formal synthetic artifact once**

Run: `python -m src.next9_lrrc_synthetic --output-dir outputs/20260801_lrrc_synthetic`

Expected: one immutable directory with `MANIFEST.json`, `engineering_pass=true`, and `scientific_improvement_claim=false`.

### Task 6: Verify scope, package health, and index freshness

**Files:**
- Verify: `src/next9_lrrc.py`
- Verify: `src/next9_lrrc_synthetic.py`
- Verify: `tests/test_next9_lrrc.py`
- Verify: `tests/test_next9_lrrc_synthetic.py`
- Verify: `outputs/20260801_lrrc_synthetic/MANIFEST.json`

**Step 1: Run focused tests**

Run: `pytest -q tests/test_next9_lrrc.py tests/test_next9_lrrc_synthetic.py`

Expected: all next9 tests pass.

**Step 2: Compile new modules**

Run: `python -m py_compile src/next9_lrrc.py src/next9_lrrc_synthetic.py`

Expected: exit code 0.

**Step 3: Run the full regression suite**

Run: `pytest -q`

Expected: no regression relative to the current `569 passed` next8 baseline.

**Step 4: Verify the artifact independently**

Recompute source and manifest hashes, parse JSON with rejection of non-finite constants, and verify every expected synthetic case.

**Step 5: Verify protected paths**

Run: `git status --short -- paper notes tex README.md PREREG.md reports`

Expected: no next9 modification to protected/canonical paths; the existing standalone reports remain byte-for-byte untouched.

**Step 6: Check CodeGraph**

Run `codegraph_status`; if a staleness banner lists either next9 source, wait for sync and query status again.

Expected: both new source files indexed with no pending sync.

### Task 7: Scientific handoff without overclaiming

**Files:**
- Do not create a success report unless a new physical cohort passes the preregistered scientific gate.

**Step 1: Report the exact outcome**

State whether synthetic engineering passed and list known limitations. Explicitly state that no scientific improvement has yet been measured.

**Step 2: Preserve the external-data gate**

Before downloading Alexandria relaxation paths, first verify that `m3gnet/rng` source tags and complete groups can be reconstructed from a bounded set of shards. If not, stop at a data-acquisition report rather than silently sampling survivors.

**Step 3: Await user confirmation for canonical edits**

Only after a future standalone scientific report is reviewed may any existing report or paper be modified.
