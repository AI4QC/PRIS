# NEXT12 CHSC and Independent DFT Cohort Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a label-free homogeneous-cell negative-curvature criterion (CHSC-v0), combine it additively with the frozen M5+PHSC gate, and prepare a prospective generated-structure cohort for physically isolated DFT validation.

**Architecture:** CHSC-v0 reconstructs the complete six-dimensional fixed-fractional-coordinate strain Hessian from 21 deterministic directional energy curvatures at two predeclared step sizes. It reuses the PHSC Richardson interval logic but has its own protocol, outputs, hashes, and three-state result. A separate cohort freezer records raw generator outcomes before any MatterSim or DFT access; a separate endpoint opener is the only component allowed to read completed DFT results.

**Tech Stack:** Python 3.10+, NumPy, SciPy, ASE, MatterSim 1.2.3, pandas/Parquet, pytest, VASP input files when an executable or external scheduler becomes available.

## Frozen scientific definition

Let the six Frobenius-orthonormal symmetric strain generators be

\[
B_1=e_xe_x^T,\ B_2=e_ye_y^T,\ B_3=e_ze_z^T,\
B_4=(e_ye_z^T+e_ze_y^T)/\sqrt{2},\
B_5=(e_xe_z^T+e_ze_x^T)/\sqrt{2},\
B_6=(e_xe_y^T+e_ye_x^T)/\sqrt{2}.
\]

For a unit direction \(v\in\mathbb R^6\), deform a row-vector lattice as
\(A(t)=A_0\exp(t\sum_i v_i B_i)\), preserving fractional coordinates. Define

\[
q_h(v)=\frac{E(+hv)-2E(0)+E(-hv)}{N h^2}.
\]

Use the fixed 21 directions \(e_i\) and \((e_i+e_j)/\sqrt 2\). Reconstruct
\(H_{ii}=q(e_i)\) and
\(H_{ij}=q((e_i+e_j)/\sqrt2)-(H_{ii}+H_{jj})/2\).
The two frozen steps are \(h=2^{-7}\) and \(h/2=2^{-8}\). For the two matrices
\(H_h,H_{h/2}\), define

\[
H_R=(4H_{h/2}-H_h)/3,\qquad
e_{num}=\lVert(H_{h/2}-H_h)/3\rVert_2.
\]

CHSC is `RESOLVED_NEGATIVE` only when the minimum eigenvalues at both scales and
the Richardson upper endpoint \(\lambda_{min}(H_R)+e_{num}\) are below
\(-\tau_{alg}\). It is `RESOLVED_NONNEGATIVE` only under the corresponding
positive lower-endpoint conditions; all other cases abstain. The criterion is a
candidate necessary rejection gate, not a sufficient stability certificate.

## Protocol boundaries

- Never change M5, PHSC-v0, their outputs, or the existing standalone report.
- Do not inspect prospective DFT energies, forces, convergence, trajectories, or
  relaxed geometries until cohort membership and all gate decisions are frozen.
- Store raw x0 geometry separately from endpoint results. The gate process must not
  receive an endpoint path or endpoint-bearing archive.
- Retain generator parse failures, unsupported elements, over-limit cells, model
  failures, CHSC failures, and DFT failures as explicit rows.
- No threshold tuning on the prospective cohort. Any future CHSC-v1 is evaluated on
  a later cohort.

### Task 1: Core CHSC mathematics

**Files:**
- Create: `tests/test_next12_chsc.py`
- Create: `src/next12_chsc.py`

**Step 1: Write failing tests**

Test basis orthonormality, 21-direction order, exact reconstruction of a supplied
symmetric 6x6 quadratic form, negative/positive/near-zero Richardson states, input
shape and non-finite rejection, and affine lattice deformation with fixed fractional
coordinates.

**Step 2: Verify RED**

Run: `pytest -q tests/test_next12_chsc.py`

Expected: collection fails because `src.next12_chsc` does not exist.

**Step 3: Implement the minimum API**

Implement `strain_basis()`, `direction_set()`, `deform_cell()`,
`directional_curvatures_to_hessian()`, `analyze_strain_hessian_pair()`, and typed
result/status/error objects. Reuse only the frozen PHSC scalar state classifier;
do not import any PHSC artifact or data.

**Step 4: Verify GREEN**

Run: `pytest -q tests/test_next12_chsc.py`

Expected: all tests pass.

### Task 2: Synthetic falsification runner

**Files:**
- Create: `tests/test_next12_chsc_synthetic.py`
- Create: `src/next12_chsc_synthetic.py`

**Step 1: Write failing tests**

Cover stable diagonal, one-mode saddle, rotated saddle, quartic scale-inconsistent,
near-zero, non-finite, immutable publication, source rehash, and no-overwrite cases.

**Step 2: Verify RED**

Run: `pytest -q tests/test_next12_chsc_synthetic.py`

Expected: collection fails because the runner is absent.

**Step 3: Implement and publish**

Use analytic energy functions only. Atomically publish `MANIFEST.json` in a new
target directory and record all executed source SHA-256 values.

**Step 4: Verify GREEN**

Run: `pytest -q tests/test_next12_chsc_synthetic.py`

Expected: all tests pass.

### Task 3: Batched MatterSim feature runner

**Files:**
- Create: `tests/test_next12_chsc_mattersim_features.py`
- Create: `src/next12_chsc_mattersim_features.py`

**Step 1: Write failing tests**

Test 85 energy evaluations per supported structure (one center plus 21 directions,
two signs, two scales), deterministic row order, exact geometry-only input allowlist,
batch chunking, failure retention, checkpoint/source rehash before no-replace publish,
and manifest assertions that labels and endpoint paths were not opened.

**Step 2: Verify RED**

Run: `pytest -q tests/test_next12_chsc_mattersim_features.py`

Expected: collection fails because the runner is absent.

**Step 3: Implement the runner**

Batch deformed ASE structures through the same frozen MatterSim 5M checkpoint used by
PHSC. Store the two 6x6 matrices, spectral diagnostics, status, timing, and explicit
errors in a new Parquet artifact. Never mutate the source frames.

**Step 4: Verify GREEN**

Run: `pytest -q tests/test_next12_chsc_mattersim_features.py`

Expected: all tests pass.

### Task 4: Additive stop composition

**Files:**
- Create: `tests/test_next12_chsc_label_free_stop.py`
- Create: `src/next12_chsc_label_free_stop.py`

**Step 1: Write failing tests**

Test the frozen rule `REJECT = M5_REJECT or PHSC_RESOLVED_NEGATIVE or
CHSC_RESOLVED_NEGATIVE`, abstention preservation, row-set/hash equality, and no-label
manifest checks.

**Step 2: Verify RED**

Run: `pytest -q tests/test_next12_chsc_label_free_stop.py`

Expected: collection fails because the composer is absent.

**Step 3: Implement and verify GREEN**

Run: `pytest -q tests/test_next12_chsc_label_free_stop.py`

Expected: all tests pass.

### Task 5: Prospective generator cohort freezer

**Files:**
- Create: `tests/test_next12_prospective_cohort.py`
- Create: `src/next12_prospective_cohort.py`

**Step 1: Write failing tests**

Require explicit real-model class, checkpoint SHA-256, scaler SHA-256, seed, attempt
index, generator status, geometry hash, and immutable raw output. Reject mock models
and silent energy-based filtering. Preserve every failed attempt in the manifest.

**Step 2: Verify RED, implement, and verify GREEN**

Run: `pytest -q tests/test_next12_prospective_cohort.py`

Expected before/after: missing-module failure, then all tests pass.

### Task 6: DFT queue and isolated endpoint evaluator

**Files:**
- Create: `tests/test_next12_dft_queue.py`
- Create: `src/next12_dft_queue.py`
- Create: `tests/test_next12_dft_endpoint_evaluate.py`
- Create: `src/next12_dft_endpoint_evaluate.py`

**Step 1: TDD the queue builder**

Freeze VASP PBE PAW choices, ENCUT rule, k-point density, electronic/ionic convergence,
spin policy, relaxation stages, timeout/retry accounting, POTCAR identity hashes, and
one row per attempted x0. Do not copy or expose licensed POTCAR contents in reports.

**Step 2: TDD the isolated evaluator**

Require the frozen cohort and decision manifests, then open endpoint results in a
separate process. Report Wilson intervals for stable-structure false rejection,
unstable-structure rejection recall, coverage, abstention, compute saved, and paired
comparisons against each Pauling rule and M5/PHSC baselines.

**Step 3: Verify**

Run: `pytest -q tests/test_next12_dft_queue.py tests/test_next12_dft_endpoint_evaluate.py`

Expected: all tests pass without a VASP executable.

### Task 7: Evidence runs and reporting

**Files:**
- Create only new directories under `outputs/`
- Create a new standalone report under `reports/` only after a successful frozen run

Run focused tests, the full test suite, synthetic cases, an engineering smoke, then
the frozen no-label cohort. If no DFT executable or external scheduler is available,
stop at a hash-verified runnable queue and state that scientific superiority remains
unproved. Do not edit `paper/`, `notes/`, `tex/`, `README.md`, `PREREG.md`, or the
existing PHSC report before explicit user confirmation.

## Execution note

The primary checkout contains the uncommitted PHSC dependency chain and user-owned
research outputs. A clean worktree would omit those dependencies, so this plan is
executed additively in the current checkout with unique NEXT12 paths and no edits to
existing artifacts. No automatic commits are made unless the user asks for them.
