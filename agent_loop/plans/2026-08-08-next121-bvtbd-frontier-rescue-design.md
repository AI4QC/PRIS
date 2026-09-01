# NEXT121 BVTBD frontier-rescue design

Date: 2026-08-08

## Objective and hard boundary

NEXT121 is an additive discovery experiment. It asks whether the closed-form
NEXT119/120 bond-valence transport budget descriptors can resolve the remaining
NEXT117 AUC--SAFE conflict. It does not replace any earlier script, result, or
canonical document.

The executable law consumes one raw, unrelaxed `x0`, frozen analytic
element/bond-valence/geometry data, and no same-composition alternative. It
does not consume a DFT value, execute DFT, update coordinates or a cell, run a
relaxation, or call a learned energy/force/stress proxy. Previously opened
discovery outcomes are used only as offline labels after the complete NEXT121
feature grammar and candidate universe have been hashed. Internal validation
and replication partitions remain unopened unless every frozen discovery gate
passes.

## Label-free BVTBD term audit

Only NEXT120 discovery feature tables are used for this audit. Physical centers
come from the frozen 10% dimensionless deformation budget. Numerical caps are
the SCIGEN discovery 99.5th percentiles among supported rows; no endpoint
column is present in those tables.

The four retained nonnegative risk operands are:

1. Overall path-budget excess in decades:
   `max(0, log10(required_linf_budget / 0.10))`.
2. Residual path debt beyond a 10% budget:
   `max(0, (deformation_debt_tau10 - 0.50) / 0.50)`.
3. Coordinate localization beyond one-half of the minimum-norm solution:
   `max(0, (required_linf_budget / ||z_*||_2 - 0.50) / 0.50)`, with
   `||z_*||_2 = minimum_motion_rms * sqrt(3N + 6)`.
4. Cell-strain budget excess in decades:
   `max(0, log10(cell_strain_frobenius / 0.10))`.

The localization term is the distinct shape descriptor. The other three are
retained despite correlation because they express different falsifiable
questions: total required motion, unresolved response, and specifically cell
strain. Unsupported rows deactivate the optional guard and keep the base law;
they are never assigned risk by missingness.

## Frozen adaptive frontier and grammar

The base set is selected only from already-published NEXT117 discovery records:

- all 44 source-AUC-passing candidates at the maximum observed SAFE count, 11;
- all 463 candidates that pass all 12 SAFE cells.

The 507 flattened physical formulas are unique. Each base receives either no
new term, one of the four BVTBD terms, or every pair of BVTBD terms. Single
weights are `{0.10, 0.25, 0.50, 1.00, 2.00}` and pair weights are independently
chosen from `{0.10, 0.25, 0.50, 1.00}`. This yields 116 nonempty configurations
and exactly `507 * 117 = 59,319` candidates.

All AUC, SAFE, broad, severe-recall, precision-lower-bound, cell, and source
gates are unchanged from NEXT117. Base-only candidates must reproduce NEXT117
booleans and SAFE counts exactly, and all six AUC values within `2e-5`.

## Execution order

1. Test the pure BVTBD derivations, missing policy, frontier flattening, frozen
   configuration count, and endpoint-read ordering.
2. Materialize the label-free BVTBD catalogue and candidate identities.
3. Hash that catalogue.
4. Only then re-read the two already-open discovery endpoint tables and run the
   fixed search.
5. Publish results atomically with input, source, and output hashes.
6. Do not open internal validation or replication unless all discovery gates
   pass.
