# NEXT107: Frozen same-catalogue two-axis CMVF guard search

Date: 2026-08-04

## Purpose and scientific boundary

NEXT107 asks one narrow question: can two complementary outputs of the
label-free Convex Mixed-Valence Flow Certificate (CMVF) repair the remaining
cross-source discovery cells that no zero- or one-CMVF-guard candidate repaired
in NEXT106?

The eventual executable remains a pre-DFT screen.  Its only per-structure input
is one raw, unrelaxed x0 plus frozen analytic element, geometry, Voronoi,
bond-valence, electrostatic, and symmetry data.  DFT values, DFT execution,
relaxed structures or trajectories, learned energies/forces/stresses, MLIPs,
physical relaxation, and same-composition alternatives are forbidden from the
executable.  Previously isolated DFT-derived outcomes are opened only as
discovery labels for finite candidate evaluation.

This is an additive experiment.  It does not replace NEXT98, NEXT98b,
NEXT104, NEXT105, NEXT106, any earlier script, or any canonical report or
manuscript content.

## Frozen evidence before this design

- NEXT105 manifest SHA-256:
  `a2340605d9e8f97165ed8fad10c33f401dc17cdade6c5552e0867923fe5002e3`
- NEXT105 SCIGEN feature SHA-256:
  `d4d7974439ea9a39cf9db0bf458c13253f80e1baf5d9faf31594182473e2a90a`
- NEXT105 WyFormer feature SHA-256:
  `299f5ab2060aebaa4c5915aac7543fadc16728ffc055a3bd341373d820aeba99`
- NEXT106 manifest SHA-256:
  `352fd653e9de5425894971a344116ef9ad2e50b71af823a1d855f2b1b8638534`
- NEXT106 optional-term catalogue SHA-256:
  `d54cb249f56921b56176dc0268a3f1c825f9588653e31b9b76b992fccad19150`
- NEXT106 discovery evaluation SHA-256:
  `c9bc8611f730d43883b3f3c900e4385042a520f8899ceb8a027fe6e5d91fa5ce`
- NEXT106 search-record SHA-256:
  `6c14c99c1db78bfa912e63cd364805d837fd924e1a3226954afad8a57cd57d07`
- NEXT106 result: 2,077 candidates, six label-free eligible CMVF terms,
  maximum 9/12 safe cells, zero candidates passing all discovery gates,
  validation and replication outputs unopened.

The label-free NEXT105 feature audit showed that reallocation and overload are
strongly correlated within a catalogue, while either flow quantity and the
log-scale mismatch are weakly correlated in both sources.  NEXT107 nevertheless
enumerates all three within-catalogue metric pairs, rather than choosing a pair
from discovery endpoints.  Cross-catalogue pairs are forbidden because core and
expanded catalogues are nested alternative views of the same certificate.

## Frozen inputs

The runner must pin and validate the exact inputs already used by NEXT106:

- SCIGEN label-free features:
  `7031d86e4fb6e469c674d208f680ace1dbe5e11e45f3d4b2befefd747efdde16`
- SCIGEN discovery endpoint:
  `f86cff6f5e9124ee82aae13911ffe55a125c6fe111fc1f64122a610febf67958`
- WyFormer label-free features:
  `c515baec0fccef5bc03c7672f1d4e1aca278f5ed4d7b6f1bf7f66c734e2b87f7`
- WyFormer discovery endpoint:
  `f39836e62a1da03ed823479e87d6f75fc0d01da60a8c0a2faa696638cc2fb9d7`
- NEXT98 manifest and term catalogue:
  `5fcd924b125767e52ac1826203595692af868ab35366899e12b82aea2726e32c`,
  `f2165f548a56cda04559a11a0d575f0654d3e8a17cf3b85b76e7974ea65dee41`
- NEXT98b manifest and search records:
  `b20d2f500ce74a6fd8b1a8a992bca3fff3ee5952fc38c09d3ad34ca317c3084d`,
  `748a4623ecfc725636837f3944b70482a97b2df39a495a81e3f8e09f5d09a4e4`
- the NEXT105/NEXT106 outputs listed above;
- this design file, whose hash is frozen into the runner after publication.

No validation or replication endpoint or geometry path is a runner argument.

## Frozen label-free calibration

Reuse the six NEXT106 term templates and the NEXT106 calibration protocol.  A
term is eligible only when, before endpoint tables are opened, each source has
at least 0.15 finite active coverage and the pooled active values have at least
eight unique transformed values.  The transform is `log1p_nonnegative`; center
is the pooled median and scale is the pooled 10th-to-90th percentile span.

The optional terms are exactly:

- core: reallocation, overload, log-scale mismatch;
- expanded: reallocation, overload, log-scale mismatch.

Missing optional terms are guards-off, base-kept.  Missing base terms remain an
abstention.

## Frozen finite candidate grammar

Start from the same 67 NEXT98b bases that pass both sources' discovery AUC
gates.  For each base enumerate exactly one of:

1. no CMVF guard;
2. one eligible CMVF guard;
3. two distinct eligible CMVF guards from the same catalogue mode.

The optional weight grid is exactly `(0.25, 0.5, 1.0, 2.0, 4.0)` for each
guard.  Pair order is canonical and cannot create duplicate candidates.  When
all six terms are eligible, each base has
`1 + 6*5 + 2*C(3,2)*5*5 = 181` candidates and the full search has exactly
`67*181 = 12,127` candidates.

For a supported row the risk is

`R = R_base + sum_j w_j max(0, (log1p(x_j) - center_j) / scale_j)`.

This is a nonnegative necessary-condition guard.  Low risk is not a claim of
stability.  A structure unsupported by CMVF is not rejected by CMVF.

## Frozen evaluation and gates

Reuse the exact NEXT103/NEXT106 evaluator, cells, thresholds, confidence
bounds, safe gates, broad gates, source AUC gates, and deterministic selection
order.  The 12 safe cells are both source aggregates and five frozen folds per
source.  A candidate is eligible for freezing only if it passes every frozen
cross-source discovery gate.  Candidate count and every record are published,
including failures.

If no candidate passes all discovery gates:

- `freeze_authorized=false`;
- no formula is frozen;
- no validation or replication output is opened;
- no improvement claim is made.

If at least one candidate passes, publish a new standalone report and stop for
user confirmation before opening validation/replication or editing canonical
reports/manuscript files.
