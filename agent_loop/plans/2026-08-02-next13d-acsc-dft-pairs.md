# NEXT13d: blinded paired DFT falsification plan

## Scope and frozen boundary

- Add new scripts, tests, and outputs only.
- Do not edit any existing NEXT13 report, old script, paper, note, README, or preregistration artifact.
- Do not inspect DFT endpoints while selecting pairs or freezing thresholds.
- Treat this as an independent falsification of the conservative three-scale ACSC subset, not as evidence that ACSC is already better than DFT.

## Label-free treatment and control definitions

- Treatment candidates are the 58 rows that are both `nested_three_scale_confirmed == True` and old `M5+PHSC+CHSC == KEEP`.
- Control candidates are old `M5+PHSC+CHSC == KEEP` rows whose formal ACSC status is `resolved_nonnegative`.
- Controls are used without replacement and no DFT-derived field is available to the matcher.

## Deterministic matching hierarchy

1. Same reduced composition (`rk`) and same atom count; minimize absolute M5-gap difference.
2. Same `rk` with different atom count; minimize atom-count difference, then M5-gap difference.
3. For unmatched treatments, same atom count with chemistry-distance matching based only on element sets, reduced stoichiometry, M5 gap, and stable SID ordering.
4. A different-atom-count fallback is implemented and frozen but is not expected for the formal cohort.

The expected formal allocation, measured before any DFT execution, is 37 + 9 + 12 = 58 pairs. The same-`rk` subset is the primary composition-controlled relaxed-energy comparison; all 58 pairs contribute to structural-relaxation and failure endpoints.

## Blinding and licensed inputs

- Give the executor only opaque task IDs, a blinded execution table, the VASP task archive, and the run protocol.
- Keep SID, ACSC role, and pair mapping in a separate private mapping table for post-run evaluation.
- Never include POTCAR contents. Record only the selected potential name, source/content SHA-256 identities, TITEL, and ENMAX.

## Frozen VASP protocol

- PBE PAW, `ENCUT = ceil(1.3 * max(ENMAX))`, `KSPACING = 0.22`, gamma centered, spin polarized.
- One static x0 stage followed by one full-cell `ISIF=3` relaxation.
- `EDIFF=1e-6`, `EDIFFG=-0.03 eV/A`, at most 200 ionic steps, one attempt per stage, 24 h timeout per stage.
- Failures and timeouts remain outcomes and are never silently dropped.

## Endpoints frozen before DFT

For every pair, compare treatment minus control for:

- failure/timeout;
- relaxation energy drop per atom;
- initial maximum force;
- maximum atomic displacement;
- absolute logarithmic volume change;
- a severe-relaxation composite: energy drop >= 0.10 eV/atom, initial maximum force >= 1.0 eV/A, displacement >= 0.5 A, absolute log volume change >= 0.10, or non-convergence.

For same-`rk` converged pairs, additionally compare relaxed energy per atom. Positive treatment-minus-control energy supports ACSC's rejection direction. Report exact paired discordance/sign tests and confidence intervals; do not refit thresholds.

## Acceptance checks before handoff

- Exactly 58 treatments, 58 distinct controls, and 116 runnable opaque tasks.
- Formal tier counts are exactly 37 same-rk/same-size, 9 same-rk/different-size, and 12 same-size chemistry fallback.
- No SID or ACSC role occurs in the blinded queue or task archive.
- Every input, source, and output is hash-bound; publication is atomic and refuses overwrite.
- Endpoint evaluator refuses incomplete accounting, wrong queue binding, duplicate task IDs, or endpoints stored beside the queue.
- Focused tests and the full repository suite pass.

