# NEXT122 SAFE12 BVTC/PRLR rescue design

Date: 2026-08-08

## Scope

NEXT122 is an additive, minimal follow-up to the failed-but-near NEXT121
discovery search. It does not replace any prior script, result, or canonical
document. The executable remains one raw unrelaxed `x0` plus frozen analytic
geometry, element, bond-valence, electrostatic, and symmetry data. It executes
no DFT, consumes no DFT value, performs no relaxation or coordinate/cell
update, and calls no learned energy/force/stress proxy.

Discovery outcomes already opened in prior stages may be reused only as offline
labels after the complete NEXT122 candidate universe is hashed. Internal
validation and replication remain unopened unless all discovery gates pass.

## Frozen hypothesis

NEXT121 reduced the closest SAFE12 formula's only source-AUC deficit to SCIGEN
worst-lattice AUC `0.5449107866991079`, `0.005089213300892181` below the frozen
`0.55` gate. Earlier label-free analytic terms already include:

- `bvtc_correction_rms__high`, testing how much edge-valence correction the
  analytic geometry Jacobian must carry;
- `prlr_bar_stress_amplification__high`, testing how strongly the analytic
  repulsive-load solution concentrates bar stress.

An earlier disposable diagnosis found the pair at weight `0.1` each to be the
best positive analytic rescue on an AUC+SAFE11 representative. NEXT122 tests
that single fixed increment without introducing a new feature, calibration,
threshold, or weight grid.

## Frozen bases and candidates

All `3,573` NEXT121 candidates passing all `12` SAFE cells are retained as
bases. Each complete flattened physical formula receives exactly four variants:

1. base reproduction;
2. append `bvtc_correction_rms__high` at weight `0.1`;
3. append `prlr_bar_stress_amplification__high` at weight `0.1`;
4. append both terms at weight `0.1` each.

Neither rescue term occurs in any selected base. All `3,573` base formulas and
all `14,292` candidate formulas are unique. No other term or weight is allowed.

All source-AUC, SAFE, BROAD, cell, severe-recall, precision-lower-bound, and
coverage gates remain byte-for-byte inherited from NEXT121. The base variants
must reproduce NEXT121 booleans and SAFE counts exactly and all six AUC values
within `2e-5`.

## Execution order

1. Verify frozen inputs and prior provenance.
2. Reconstruct only the label-free analytic feature/term table.
3. Select and flatten the published NEXT121 SAFE12 bases.
4. enumerate the fixed four variants and hash the full candidate universe.
5. Only then re-read the two already-open discovery outcome tables.
6. Run the unchanged evaluator and publish atomically.
7. Keep validation and replication closed unless every discovery gate passes.
