# NEXT108: Frozen near-miss base rescue with expanded CMVF flow pair

Date: 2026-08-04

## Scope

NEXT108 is the final adaptive discovery search in the CMVF branch.  It tests
whether the only NEXT107 pair that reached 10/12 safe cells can rescue base
laws that were excluded solely because their original source AUC metrics were
just below the frozen gates.

The executable boundary is unchanged: one raw unrelaxed x0 plus frozen
analytic element/geometry/Voronoi/bond-valence/electrostatic/symmetry data.
No executable DFT value or calculation, relaxed structure or trajectory,
learned energy/force/stress, MLIP, physical relaxation, or same-composition
alternative is allowed.  DFT-derived quantities remain isolated discovery
labels used only by the frozen evaluator.

This is additive and does not replace any prior script, result, report, or
manuscript content.

## Frozen adaptive evidence

NEXT107 formal outputs are pinned as:

- manifest: `388099bf9513e252488a301be1814ade74ad227376dcf740ab833b2666eca9ed`;
- catalogue: `731e2021b716c084ea106ae83dcbf9eee49f52d74d927e6672cbf114ee990777`;
- evaluation: `65fe4767d52c1807d288e2d6b3e2a1a9d86c93f8c791f57cd2067e5e068d6f82`;
- search records: `bc06ee907f223ab800a623f44ef2b924260b51fcdb080da2161fbced0a684a53`.

NEXT107 evaluated 12,127 candidates and did not pass all gates.  Twelve
candidates reached 10/12 safe cells; all twelve used exactly expanded CMVF
overload plus expanded CMVF reallocation and none passed every source AUC
gate.  The closest 10-cell candidate missed the frozen worst AUC margin by
0.007137.  Validation and replication remained unopened.

NEXT98b contains 12,111 base formulas.  Exactly 67 pass all source AUC gates;
exactly 353 have all six stored source AUC metrics no more than 0.01 below the
corresponding frozen gate.  NEXT106/107 searched only the 67 already-passing
bases.  NEXT108 closes this finite near-miss blind spot.

This search is explicitly adaptive to discovery results.  Even a discovery
pass is not evidence for a scientific improvement until a separately frozen
internal validation succeeds.  The current task stops for a standalone report
and user confirmation before any such validation is opened.

## Frozen input identities

Reuse and pin every NEXT107 formal input identity:

- SCIGEN features `7031d86e4fb6e469c674d208f680ace1dbe5e11e45f3d4b2befefd747efdde16`;
- SCIGEN discovery endpoint `f86cff6f5e9124ee82aae13911ffe55a125c6fe111fc1f64122a610febf67958`;
- WyFormer features `c515baec0fccef5bc03c7672f1d4e1aca278f5ed4d7b6f1bf7f66c734e2b87f7`;
- WyFormer discovery endpoint `f39836e62a1da03ed823479e87d6f75fc0d01da60a8c0a2faa696638cc2fb9d7`;
- NEXT98 manifest/catalogue `5fcd924b125767e52ac1826203595692af868ab35366899e12b82aea2726e32c`,
  `f2165f548a56cda04559a11a0d575f0654d3e8a17cf3b85b76e7974ea65dee41`;
- NEXT98b manifest/search `b20d2f500ce74a6fd8b1a8a992bca3fff3ee5952fc38c09d3ad34ca317c3084d`,
  `748a4623ecfc725636837f3944b70482a97b2df39a495a81e3f8e09f5d09a4e4`;
- NEXT105 manifest/features `a2340605d9e8f97165ed8fad10c33f401dc17cdade6c5552e0867923fe5002e3`,
  `d4d7974439ea9a39cf9db0bf458c13253f80e1baf5d9faf31594182473e2a90a`,
  `299f5ab2060aebaa4c5915aac7543fadc16728ffc055a3bd341373d820aeba99`;
- the four NEXT107 outputs above;
- this design file, whose SHA-256 is frozen into the runner after publication.

No validation or replication path may be a runner argument.

## Frozen near-miss base pool

For each NEXT98b record require all six stored AUC values to be finite and:

- pooled AUC >= `0.75 - 0.01` for both sources;
- macro lattice AUC >= `0.60 - 0.01` for both sources;
- worst lattice AUC >= `0.55 - 0.01` for both sources.

This must select exactly 353 base records on the pinned input.  No safe-cell,
threshold, or CMVF result is used to choose bases.

## Frozen CMVF pair and candidate grammar

Recompute the six NEXT106 label-free term calibrations before endpoints are
opened and require exact equality to the pinned NEXT107 catalogue.  Use only:

- `cmvf_expanded_overload__high`;
- `cmvf_expanded_reallocation__high`.

For each of the 353 base records enumerate:

1. the base with no CMVF guard;
2. the expanded pair with independent weights from
   `(0.25, 0.5, 1.0, 2.0, 4.0)`.

The candidate count is exactly `353 * (1 + 5*5) = 9,178`.  The risk formula is
the same nonnegative hinge sum used by NEXT107.  Outside expanded CMVF support,
both optional guards are off and the base is retained.  The base missing policy
is abstention.  Low score is never interpreted as proof of stability.

## Frozen evaluation, decision, and stopping rule

Reuse the exact NEXT103/NEXT106/NEXT107 source AUC, 12-cell SAFE, broad,
confidence-bound, threshold, Pauling comparison, and deterministic selection
logic.  Publish every candidate record.

If no candidate passes all gates, freeze nothing, open neither validation nor
replication, and terminate the CMVF combination branch with a standalone
report.  If a candidate passes, freeze it only as a discovery candidate, write
the standalone report, and stop for user confirmation before any validation,
replication, report, or manuscript modification.

