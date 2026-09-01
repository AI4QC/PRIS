# NEXT237 Selection-Domain Implementation Correction

Date: 2026-08-09

Status: frozen after publication and audit of the original NEXT237 v1 output,
before modifying the implementation or computing the corrected representative
record.

## Observed implementation defect

The original NEXT237 v1 formal run evaluated the frozen catalogue correctly:
one exact NEXT224 reproduction control plus 2,625 eligible new candidates. It
also correctly reported zero all-gate candidates and the frozen NEXT238
population of 2,140 eligible AUC+SAFE/non-BROAD candidates with sorted-key
digest
`005ebc0a0c56cab758e75903430cde2811e196d5105064b1f9ded68156d887a9`.

However, the reporting-only call to `select_best_eligible_record` received the
complete table, including the reproduction control. The control won that
ranking, so the published representative record and formula contain null
conditioned-certificate fields. This is contrary to the frozen requirement to
report the best *eligible new* AUC+SAFE record.

## Frozen correction

- Preserve the original v1 directory and every artifact in it unchanged.
- Change only the reporting selection domain from the full records table to
  rows with `eligible_new_candidate == true`.
- Add a fail-closed assertion that any selected specification is an eligible
  new candidate, never the reproduction control.
- Change the implementation protocol identifier and external result directory
  suffix from v1 to v2 so the correction is additive and auditable.
- Do not change the catalogue, scores, gates, candidate keys, feature universe,
  widths, amplitudes, evaluator, or NEXT238 population.
- Rerun all 2,626 records from the same 265 immutable inputs. The v2 parquet
  search table must have SHA-256
  `d1658f35132bfb778ac78631f6fc2b39e74a1fc8b0ea30378d52e112cc329423`,
  exactly matching v1. The all-gate count must remain zero and the NEXT238
  population count and digest must remain 2,140 and the digest above.
- If any of those invariants differs, reject v2 and do not run NEXT238.

This correction does not inspect or open validation or replication data and
does not alter the no-DFT executable boundary.
