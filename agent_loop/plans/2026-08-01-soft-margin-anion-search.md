# Sequential soft-margin anion search (`np-next-20260801e`)

## Status and scope

This design was frozen after inspecting `np-next-20260801d`.  It is therefore
an adaptive, development-only sensitivity experiment, not a preregistered or
confirmatory test.  It must not overwrite earlier scripts, outputs, reports,
the manuscript, or any canonical project document.  It must not read the
lockbox.

The strict experiment enforced zero empirical satisfaction loss in every
eligible discovery anion and anion-by-fold cell.  That restored anion stability
on the fitted split but reduced the rejection gain to noise and failed true
LOKO.  This experiment tests one pre-specified relaxation of those constraints.

## Frozen inputs and candidate vocabulary

- Reuse the isolated downstream discovery/calibration tables and their manifest
  verification from `np-next-20260801d`.
- Reuse the corrected `next4` feature caches, including the unified valence
  cascade and corrected P2/P6/P7/P9 definitions.
- Reuse the `next4` candidate vocabulary.  Tainted historical P2/P6/P7/P9
  columns remain excluded.  Corrected P7 remains guard-only.
- Fit only on discovery rows.  Calibration has already been adaptively reused
  and is only a historical diagnostic.

## Frozen search

- Existing-loop baseline: the same discovery-only beam used by `next4`, with
  overall satisfaction floor 0.98, width 24, and at most 12 rules.
- Additive search: worst-perturbation-kind-first robust beam, width 96, at most
  12 rules, and minimum incremental bad-row rejection 0.0015.
- Stable four-fold assignment: CRC32 of `source_id`, exactly as in `next4`.
- Eligible full-anion strata: at least 200 discovery rows.
- Eligible anion-by-fold cells: at least 50 discovery rows.
- Anions with fewer than 200 rows are pooled into one `other-anions` full
  stratum when that pool has at least 50 rows; it uses the full-anion margin.
- Full-anion floor: paired existing-loop satisfaction minus 0.0025, clipped at
  zero.
- Anion-by-fold floor: paired existing-loop satisfaction minus 0.01, clipped at
  zero.
- Overall discovery satisfaction must still be at least 0.98.
- No alternative margins, widths, rankings, or candidate vocabularies will be
  tried within this experiment identity.

## Diagnostics and interpretation

Report discovery and reused-calibration values for satisfaction, pooled bad-row
rejection, minimum perturbation-kind rejection, worst shared-anion satisfaction,
four deterministic folds, and every robust stratum.

The historical metric gate remains unchanged:

- at least one corrected/additive descriptor is selected;
- calibration satisfaction is no worse than the baseline by more than 0.005;
- calibration pooled rejection improves by at least 0.02, or minimum-kind
  rejection improves by at least 0.03;
- calibration worst shared-anion satisfaction delta is at least -0.01;
- selected-feature coverage is at least 0.90 on both real and perturbed rows;
- downstream materialized source tables contain zero lockbox rows.

The following pre-run audit additions are also required before calling a
candidate successful:

- selected target-and-guard joint finite coverage is at least 0.90 within every
  eligible full-anion stratum and within `other-anions` when present;
- pooled rejection delta is non-negative in at least three of four deterministic
  discovery folds and no fold is below -0.02;
- true LOKO refit has non-negative signed macro rejection delta relative to the
  existing loop and no held perturbation kind is below -0.02;
- the all-295 unknown-fails-closed pass-rate delta is at least -0.03, after its
  joint required-feature coverage gate of at least 0.90.

Because calibration is adaptively reused, passing this gate would identify a
candidate for future physically isolated validation, not a confirmed new law.
Failure of the gate is a negative result.  If the metric gate passes, run the
already frozen all-295 unknown-fails-closed coverage/false-positive diagnostic
and a true LOKO refit without opening the lockbox.  If it fails, those checks may
still be run for characterization but cannot rescue the result.

## Provenance

Write only to `outputs/20260801_soft_margin_anion_search/` and a new standalone
report.  Every JSON must store input hashes, implementation hash, design hash,
the exact margins, and `lockbox_access: false`.  Refuse to overwrite an existing
output.
