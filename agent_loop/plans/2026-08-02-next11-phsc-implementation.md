# next11 PHSC-v0 implementation plan

**Goal:** Add a fixed, label-free PHSC-v0 experiment without modifying any
existing script, result, report, paper source, README, or preregistration.

**Architecture:** A pure numerical module owns probe ordering, Hessian
construction, Helmert projection, spectral states, and synthetic oracles.  A
separate MatterSim runner reuses the already verified indexed 5M adapter,
batches complete four-probe coordinate groups, and seals a sid-aligned feature
artifact.  A third label-free command reconstructs frozen baseline decisions
and applies only the predeclared 66-new-reject stop.

## Task 1: pure PHSC-v0 contract

Files:

- create `src/next11_phsc.py`
- create `tests/test_next11_phsc.py`

Write failing tests first for validation, deterministic Helmert projection,
probe order, exact quadratic spectra, algorithmic tolerance, three successful
states, abstentions, and exact `12N` calls.  Implement only the frozen design.

## Task 2: synthetic falsification package

Files:

- create `src/next11_phsc_synthetic.py`
- create `tests/test_next11_phsc_synthetic.py`

Predeclare positive, saddle, stationary-saddle, force-orthogonal-saddle,
translation, weak-near-zero, and two-scale-inconsistent cases.  Publish a new
no-overwrite synthetic directory with strict JSON, source hashes, and
`scientific_improvement_claim=false`.

## Task 3: batched MatterSim feature runner

Files:

- create `src/next11_phsc_mattersim_features.py`
- create `tests/test_next11_phsc_mattersim_features.py`

Reuse the next10 sealed-checkpoint indexed adapter.  Stream complete
`(+h,-h,+h/2,-h/2)` coordinate groups in fixed-size chunks of 256 groups
(1,024 probe structures), immediately reduce each group to two Hessian columns,
and never materialize all probe structures at once.  Test doubles may use a
smaller chunk, but production eligibility requires the frozen value 256 and a
separate MatterSim model batch size of 32.  Seal exactly one feature table plus
manifest.

## Task 4: label-free necessary-condition gate

Files:

- create `src/next11_phsc_label_free_stop.py`
- create `tests/test_next11_phsc_label_free_stop.py`

Accept no label path or label table.  The only data inputs are the committee
feature table/manifest, threshold-role assignments, PHSC feature table/manifest,
and the frozen protocol.  Do not read the broader next8 manifest because it
contains unnecessary label-location metadata.  Pin the frozen protocol SHA-256
to `b8049ad2f627ad91973ae86178c704871086097462f287b21c5330e3d4916fd4`
and require the following byte identities before reconstruction:

```text
committee features  65f0234010f17f43a96789bde7858bae038ffaa4aaa2130eaee163fd3245bc8c
committee manifest  e59848270c0fd1693d6f7d579ee327aebf4f34399ee73d27eb2c97f947cab9dd
threshold roles     e6de5f5b5fc9545944043bda46e313fa2060833f1baa31dd93dcca12e4769602
```

Verify that the frozen protocol's own provenance fields close these same
identities.  Reconstruct
primary/comparator M5 and AGREE995 decisions, and publish overlap counts plus
the frozen `>=66` net pass/fail result.  Never calculate recall, endpoint
energy, or a scientific-improvement metric.

## Task 5: verification and execution

1. Run focused tests and independent code/science review.
2. Run a small injected-oracle/MatterSim CUDA smoke test in a new directory.
3. Run the full 2,171-row label-free feature job only after the smoke artifact
   and source closure pass.
4. Run the label-free stop command.
5. Do not open labels regardless of the result until a new physically isolated
   DFT cohort exists.

## Task 6: standalone report

Create a new file under `reports/` that reports engineering evidence, exact
runtime/provenance, label-free counts, limitations, and the next decision.  Do
not edit any previous report or canonical manuscript content.
