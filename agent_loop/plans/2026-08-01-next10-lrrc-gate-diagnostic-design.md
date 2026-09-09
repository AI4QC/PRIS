# Next10 LRRC fixed-gate exploratory diagnostic design

## 1. Objective and evidence level

next8's `AGREE995` carries a `+1.26325` percentage-point signal at formula selection, but only
`+0.18425` percentage points on the already-opened development gate, where the paired CI crosses
zero and comparator safety fails. next9 therefore froze `LRRC-v0`, a local second-order response
orthogonal to the disagreement of the same checkpoint.

This round answers one narrow question: **without refitting the next8 thresholds and without
sweeping the LRRC parameters**, can LRRC negative curvature add worthwhile rejection signal to a
strong M5 baseline on the already-exposed development gate? That gate was opened by earlier work,
so the result can only be called a `posthoc exploratory diagnostic` -- not an independent
validation, not a scientific success, and not a new law.

The historical test set, OMat24, the paper, the old reports, the README and PREREG all stay
closed or unchanged.

## 2. The two-stage opening order

### 2.1 Label-free feature stage

Read and hash only:

1. the next8 development feature manifest and feature parquet;
2. the next8 threshold-role assignment;
3. the raw x0 frame zip;
4. the fixed MatterSim 5M checkpoint;
5. the next9 LRRC and next10 runner source.

Select only the sids with `threshold_role == development_gate`, and verify that this selection
corresponds one to one with the next8 feature rows.
This stage may not accept a label path and may not import the protocol evaluation code. Its
output contains only the LRRC numerical diagnostics, statuses, input hashes, checkpoint hash and
run telemetry. Only once it has been published and rehashed may the evaluation stage read the old
development labels.

### 2.2 Post-hoc evaluation stage

The evaluator first verifies the sealed LRRC feature manifest, the next8 frozen protocol, the
next8 development-gate metric artefact and every input hash. It must then reproduce the old
M5/AGREE995 development-gate decision and core metrics item by item; if the reproduction fails it
fails closed before any candidate is evaluated.

## 3. The fixed LRRC computation

For each supported structure, use only MatterSim 5M:

- 1 unperturbed force batch prediction;
- a fixed direction and step size constructed from next9's `translation_projected_direction` and
  the MIC `d_star`;
- 4 perturbed force batch predictions: `+h`, `-h`, `+h/2`, `-h/2`;
- a call to next9 `evaluate_lrrc` through a fixed-order replay oracle, so that the scalar
  implementation is the only implementation of the formula;
- exactly 5 force sets per successful non-stationary structure; stationary points use 1 force set
  only.

Any checkpoint change, sid/frame mismatch, misaligned batch output, non-finite force, or change
of an input before publication fails closed as a whole.
Failed rows are not quietly discarded; the attributable geometric or numerical state is written
explicitly into the feature parquet.

## 4. The frozen candidate catalogue

Both next8 tracks keep the original M5/AGREE995 final thresholds and the strict
`score > threshold` rule:

| formula | fixed decision |
|---|---|
| `M5` | the next8 M5 baseline, used only for exact reproduction |
| `AGREE995` | the formula next8 selected, used only as a weak-signal reference |
| `M5_LRRC_OR` | `M5 REJECT or LRRC_negative` |
| `M5_LRRC_QCRC` | `M5_LRRC_OR` first, then next9's Quota-CRC applied by M5 score |
| `AGREE995_LRRC_QCRC` | `AGREE995 REJECT or LRRC_negative` first, then Quota-CRC applied by AGREE995 score |

`LRRC OK/nonnegative` and `STATIONARY_FALLBACK` keep the base decision; a geometric, model or
numerical failure of LRRC is ABSTAIN. Quota-CRC is the last layer and can only turn a REJECT back
into a KEEP, leaving ABSTAIN unchanged. The quota is fixed at `ceil(sqrt(n))` and all boundary
ties are KEEP.

The catalogue is frozen before the labels are opened; no formula is added afterwards on the basis
of the negative-curvature fraction, of individual labels, or of the results.

## 5. Evaluation and stopping rules

Each candidate reports both primary and comparator quantities:

- DFT savings and macro savings;
- exact/near/valuable retention;
- high-energy removal recall and reject precision;
- abstention, all-reject groups and regret;
- a 20,000-draw `rk` paired bootstrap against the corresponding base formula.

Results on a reused gate serve only to screen for direction:

- if no LRRC candidate increases savings, or if any point-estimate gain comes with valuable
  recall clearly out of bounds, stop this line;
- only if at least one Quota-CRC version simultaneously shows a positive gain in savings, a
  non-inferior point estimate of valuable recall, and no all-reject group, does it go on to the
  WBM retrospective audit or the Alexandria 2025 out-of-time cohort;
- however good the point estimates, this round records
  `scientific_improvement_claim=false` and may not open the historical test set or OMat24.

Only if a subsequent new physical cohort passes a pre-frozen scientific gate is a separate success
report written, and only then, after the user confirms, may the paper or the old reports be
modified.
