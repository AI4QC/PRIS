# NEXT529–NEXT532: SSSP with bond-valence coherence safeguard

Date frozen: 2026-08-13 (America/Chicago)

## Motivation and scientific status

NEXT527 validated the frozen standalone SSSP threshold on SCIGEN and
WyFormer.  NEXT528 replicated every SCIGEN gate and every WyFormer ranking,
fold, bootstrap, coverage, and Pauling-dominance gate, but failed the frozen
WyFormer operating gates: protected-recall lower bound `0.948709 < 0.95` and
severe-precision lower bound `0.559544 < 0.60`.  NEXT525–NEXT528 are closed
unchanged and do not authorize a success report.

After that endpoint opening, a six-cell development audit (SCIGEN and
WyFormer discovery/validation/replication) tested already-existing zero-DFT
analytic features only inside the low-SSSP rejected extreme cohort.  The most
consistent independent discriminator was scale-calibrated bond-valence (SCBV)
mismatch: protected low-SSSP structures had lower mismatch than severe
low-SSSP structures in all six cells.  Oriented protected-positive AUCs for
`scbv_mismatch_rms` were `0.729, 0.734, 0.820, 0.614, 0.717, 0.714` in the
fixed cell order.  This makes the safeguard an outcome-informed next
generation and means none of the six cells is independent evidence for the
new formula.

## Immutable executable boundary

The executable formula receives composition and one raw, initial, fully
periodic geometry only.  It may use frozen elemental tables, formal-valence
inference, periodic Voronoi topology, and frozen bond-valence parameters.  It
may not use DFT values, a relaxed/later geometry, a trajectory, learned
energy/force/stress, an MLIP or proxy potential, or any relaxation.

DFT-derived structural-change outcomes are used only as offline labels after
the external x0 cohort, feature tables, formula, predictions, gates, and
hashes are frozen.

## Frozen formula: SSSP-BVC

Let `S` be unchanged NEXT411 SSSP and let `M` be unchanged NEXT22
`scbv_mismatch_rms`, both computed on the same x0 geometry.

The constants are frozen from the six-cell development audit:

- `t_S = 0.5231805323` (the unchanged NEXT525 SSSP cutoff);
- `t_M = 0.33695346214642063` (the largest observed development threshold
  attaining the frozen lexicographic optimum).

Define

```
d_S = max(0, t_S - S) / t_S
d_M = max(0, M - t_M) / t_M
R_BVC = d_S * d_M
REJECT iff S is supported, M is supported, S <= t_S, and M >= t_M
```

If SSSP is unsupported, abstain.  If SSSP is supported but SCBV is
unsupported, keep: an unavailable safeguard may not create a false rejection.
No coefficient, transform, threshold, direction, or missing policy may change
after this document is frozen.

The physical interpretation is conjunctive.  Same-sign crowding is only a
screening violation when the opposite-sign bond network also cannot satisfy
site valence magnitudes after one closed-form global amplitude calibration.
Low SSSP with coherent bond valence is treated as an allowed dense motif.

## Development accounting

At the frozen constants, all six exposed source-role cells satisfy Wilson
lower protected recall `>= 0.95` and Wilson lower savings `>= 0.02`.
The minimum Wilson lower severe-rejection precision is `0.712092`.  These are
development results only.  The WyFormer severe recall is deliberately small
(`0.040–0.065`) because this is a safe high-precision prescreen.

## NEXT530: new disjoint WBM x0 cohort

Create a new 8,192-row WBM cohort from the label-free test-ID universe and the
official initial-geometry archive.  Before ranking, exclude the union of:

- all 2,048 NEXT14 exposed-development IDs; and
- all 8,192 NEXT23 prior relaxation-holdout IDs.

Retain 2–12 atom cells.  Rank remaining size-eligible IDs by ascending
`SHA256("next530-sssp-bvc-wbm-relaxation-v1|" + material_id)` and select the
first 8,192.  Publish only sanitized x0 geometries, formula/reduced-formula
metadata, exact exclusions, and hashes.  The WBM summary, relaxed archive,
energies, and endpoint values must not be opened by NEXT530.

## NEXT531: label-free feature and prediction freeze

For every NEXT530 x0 structure compute unchanged NEXT411 SSSP and unchanged
NEXT22 SCBV with the periodic Voronoi graph.  Apply the exact SSSP-BVC formula
and freeze predictions before opening WBM endpoints.  Minimum SSSP coverage is
`0.90`; SCBV coverage is reported but does not reduce formula coverage because
unsupported SCBV keeps rather than rejects.  At least 20 rounded unique values
are required for each analytic feature.

Freeze a Pauling P2–P5 decision for the identical geometry cohort using the
existing analytic control implementation.  If a compatible Pauling control
cannot be computed without broadening the input boundary, report that and do
not manufacture a substitute.

## NEXT532: one-shot external relaxation-change evaluation

Only after NEXT531 publishes may NEXT532 read exactly two columns from the
official WBM summary: `material_id` and
`site_stats_fingerprint_init_final_norm_diff`.  The latter is a DFT-relaxed
offline endpoint and never enters the executable formula.

Use the existing NEXT23 operational strata:

- protected: endpoint `<= 0.10`;
- severe: endpoint `>= 0.50`;
- middle values affect savings but not rejection precision.

The external role passes only if all applicable frozen gates pass:

- Wilson lower formula coverage `>= 0.90`;
- Wilson lower protected recall `>= 0.95`;
- Wilson lower severe-rejection precision `>= 0.60`;
- Wilson lower savings `>= 0.02`;
- at least 25 rejected extreme rows and at least 10 rejected severe rows;
- five reduced-formula folds are populated and each retains at least 20
  protected and 20 severe formula-supported rows;
- the binary rejection AUC on all protected/severe rows is `> 0.50`;
- if the Pauling control is available, SSSP-BVC must exceed it in binary AUC,
  Wilson lower coverage, protected recall, and rejection precision.

Report SSSP and `R_BVC` continuous diagnostics, but they are not acceptance
gates because the new formula is explicitly a conservative conjunctive
operating rule rather than a global ranker.

## Reporting boundary

If NEXT532 passes, write a new independent report that distinguishes:

1. replicated continuous SSSP ranking evidence;
2. failed raw SSSP operating replication;
3. outcome-informed SSSP-BVC development; and
4. the new disjoint WBM external result.

Do not edit any canonical report, README, preregistration, notes, paper, or
manuscript until the user reviews that independent report.  If NEXT532 fails,
preserve all artifacts, do not write a success report, and continue with a
non-duplicate mechanism or new evidence source.
