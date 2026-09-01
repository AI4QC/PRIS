# next10 LRRC pre-opening stop rule

Status: **frozen before any next10 label-byte opening**.

This note is additive. It does not alter the frozen next10 candidate catalog,
the next8 artifacts, any prior report, or any paper source.

## Terminology boundary

- LRRC-v0 is a per-atom directional curvature along the translation-projected
  M5 force direction. It is not a minimum Hessian eigenvalue.
- `U_num` is a deterministic two-scale numerical-consistency proxy. It is not
  a confidence bound or a rigorous truncation-error upper bound.
- The existing code/protocol identifier `QCRC` is retained for provenance, but
  scientific prose must call the operation the **sqrt-quota policy**. No
  conformal-risk-control guarantee is claimed.

## Label-free feasibility stop

The sealed development cohort contains 2,171 rows. A three-percentage-point
absolute increase in formula-selection savings therefore requires at least

```text
ceil(0.03 * 2171) = 66
```

additional deterministic rejects relative to the corresponding frozen
baseline.

After the formal label-free LRRC feature table has passed provenance and
runtime verification, compute the reject counts for all five already-frozen
next10 policies without reading label bytes. If the most aggressive policy,
`M5_LRRC_OR`, adds fewer than 66 rejects over `M5`, stop next10 immediately and
do not open the development labels. No less aggressive subset policy can meet
the +3 percentage-point savings gate in that case.

## One-opening and stopping rule

If the label-free feasibility stop passes, the development labels may be
opened once for the already-frozen five-policy catalog. Do not add or tune a
candidate after seeing those results.

Stop this gate if every candidate has non-positive savings improvement, or if
any candidate under consideration has one of the following:

- valuable-recall point difference below -0.5 percentage points;
- any fully rejected group;
- abstention-rate increase above 1 percentage point.

Even a directional signal remains exploratory on this reused development
gate. At most one candidate may be selected for a genuinely new external
cohort. No step-size, stationarity-threshold, curvature-threshold, or quota
exponent scan is permitted after label opening.
