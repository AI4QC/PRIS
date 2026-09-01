# Next10 LRRC Fixed-Gate Diagnostic Implementation Plan

> **For Codex:** Use strict test-driven development and keep every change additive.

**Goal:** Run one fixed, posthoc LRRC falsification on the already-exposed next8 development gate without refitting thresholds or touching historical test/OMat24.

**Architecture:** `src/next10_lrrc_mattersim_features.py` performs label-free, five-pass batched 5M force inference and seals one feature artifact. `src/next10_lrrc_gate_diagnostic.py` separately verifies the sealed feature artifact, reproduces next8 baselines, then evaluates the predeclared LRRC/Quota-CRC catalog. Existing next8/next9 modules are imported but never modified.

**Outputs:** `outputs/20260801_lrrc_gate_features/` and, only after the feature directory is sealed, `outputs/20260801_lrrc_gate_diagnostic/`.

### Task 1: Test and implement batch LRRC feature construction

**Files:**
- Create: `tests/test_next10_lrrc_mattersim_features.py`
- Create: `src/next10_lrrc_mattersim_features.py`

Write RED tests for deterministic sid ordering, exact base/`+h`/`-h`/`+h/2`/`-h/2` batch order, stationary handling, invalid force failure, replay equivalence to scalar next9, and no label-path/API surface. Implement the smallest injectable batch-predictor core and make every scientific input explicit.

### Task 2: Test sealed input and archive validation

**Files:**
- Modify: `tests/test_next10_lrrc_mattersim_features.py`
- Modify: `src/next10_lrrc_mattersim_features.py`

Require exact feature/role/frame/checkpoint hashes, unique archive member stems, one-to-one selected sid coverage, strict x0 validation, before/after rehash, source hashes, runtime identity, and no-replace atomic publication. A per-row production prediction failure is fatal rather than silently reducing the cohort.

### Task 3: Produce the label-free development-gate feature artifact

Run a small injected-predictor smoke first. Then run the fixed MatterSim 5M checkpoint on exactly the next8 `development_gate` rows. Independently verify parquet schema, row order/count, finite successful diagnostics, status counts, five-call telemetry, manifest closure and checkpoint/source hashes. Do not inspect endpoint labels during this task.

### Task 4: Test and implement fixed decision catalog

**Files:**
- Create: `tests/test_next10_lrrc_gate_diagnostic.py`
- Create: `src/next10_lrrc_gate_diagnostic.py`

Write RED tests for exact next8 M5/AGREE995 reproduction, frozen primary/comparator thresholds, OR composition, stationary fallback, LRRC failure abstention, Quota-CRC subset/tie behavior, formula order, and a hard rejection of refit parameters or unknown candidates.

### Task 5: Test opening order and immutable evaluation publication

Require the evaluator to validate every feature/frozen-protocol/baseline-metric hash before its first label read. Test mismatches with a label-reader sentinel proving labels remain unopened. Seal long-form predictions, metrics, paired bootstrap JSON, frozen catalog, manifest, and `scientific_improvement_claim=false` with no-replace atomic publication.

### Task 6: Run the one-shot posthoc diagnostic

After the catalog and feature artifact are sealed, run exactly once with 20,000 `rk`-paired bootstrap resamples and seed `20260801`. Recompute M5/AGREE995 baseline metrics before candidate evaluation; abort if they differ from next8.

### Task 7: Verify and hand off

Run focused tests, `py_compile`, then the full pytest suite. Independently parse strict JSON, rehash all outputs and sources, check CodeGraph freshness, and verify no next10 changes under `paper/`, `notes/`, `tex/`, `README.md`, `PREREG.md`, or existing report files.

If no candidate passes the exploratory direction screen, report the negative result and stop LRRC. If a candidate passes, write a new standalone exploratory report that clearly labels the reused gate and awaits user confirmation before any canonical document edit; external confirmation remains mandatory for a scientific claim.
