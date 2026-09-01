# NEXT91--NEXT93: fixed RLI candidate freeze and one-shot validation

Date: 2026-08-03

Status: candidate freeze written after all SCIGEN discovery analyses and before
opening the physically isolated internal-validation endpoint. Internal
validation and internal replication endpoints are still unopened.

## Evidentiary status

This is deliberately not described as a pre-source preregistration. The exact
candidate was selected after inspecting SCIGEN discovery labels and the
published NEXT87 6,186-candidate record. It is therefore a development
candidate. Its first confirmatory evidence can only come from the already
isolated internal-validation partition, whose formulas do not cross discovery
and whose endpoint parquet has not been read.

NEXT87 and NEXT90 remain failed protocols. This protocol does not retroactively
change their term-list/weight-stability gates. It instead freezes one explicit
candidate whose fixed predictions showed direct stability across all five
discovery composition folds.

## Fixed law: Rigidity--Load Incompatibility (RLI)

Use the exact NEXT86 frozen one-sided hinges:

`h_g(x) = max(0, direction * (transform(g(x)) - center) / scale)`.

The candidate is:

```text
R_RLI(x) = h_sivr_edge_mismatch_max(x)
         + 4 h_sscp_load_rms(x)

REJECT if R_RLI >= 3.915855102781074
missing/non-finite constituent -> KEEP
```

Exact term parameters:

| term | transform | direction | center | scale | weight |
|---|---|---:|---:|---:|---:|
| `sivr_edge_mismatch_max` | `log1p_nonnegative` | +1 | 0.34809689849136527 | 0.2268496027212349 | 1 |
| `sscp_load_rms` | `log1p_nonnegative` | +1 | 0.09650974330938514 | 0.07475030243877033 | 4 |

The executable inputs remain one raw unrelaxed generated `x0` structure plus
frozen analytic element/geometry/graph/bond-valence/rigidity calculations. No
DFT value, relaxed structure, trajectory, learned energy/force/stress proxy,
ML interatomic potential, physical relaxation, same-composition alternative,
identity shortcut, or lattice-class ID is permitted.

## Why this candidate was frozen

Among 24 NEXT87 formulas that passed every full-discovery statistical,
lattice, and Pauling-comparison gate except the original fold-winner selection
rule, RLI is the simplest two-term candidate with substantial margins:

| discovery metric | RLI | required |
|---|---:|---:|
| severe rejected | 1,359 | > Pauling 882 |
| severe precision Wilson lower | 0.942909 | > 0.80 and > Pauling 0.818051 |
| protected recall Wilson lower | 0.974603 | >= 0.95 |
| support coverage Wilson lower | 0.979429 | >= 0.90 |
| savings Wilson lower | 0.119530 | >= 0.02 |
| pooled extreme AUC | 0.778872 | >= 0.75 |
| macro lattice AUC | 0.660207 | >= 0.65 |
| worst lattice AUC | 0.575061 | >= 0.55 |

Applying the same formula and same threshold directly to each of five held-out
discovery composition folds gave:

- support-coverage lower: 0.954729--0.995029;
- protected-recall lower: 0.966029--0.970283;
- severe-precision lower: 0.913656--0.940161;
- savings lower: 0.100030--0.123554.

All five fixed-prediction folds therefore pass the four operating gates
without fold-specific thresholds. This diagnostic is recorded as discovery
robustness, not as independent validation.

## NEXT91: endpoint-free freeze

The freeze interface accepts:

- NEXT85 label-free feature partitions;
- the NEXT86 term catalogue;
- the immutable NEXT87 manifest/evaluation/search record;
- this design.

It accepts no endpoint path. It verifies the candidate row and its discovery
metrics in the immutable NEXT87 record, reconstructs the exact term parameters
from NEXT86, publishes one formula JSON, and applies it to all three feature
partitions. Prediction parquet files contain only material ID, partition role,
score, support, decision, and formula hash. They contain no DFT endpoint or
endpoint-derived field.

Both internal-validation and internal-replication predictions must be frozen
before either endpoint is opened.

## NEXT92: one-shot internal validation

NEXT92 accepts only the NEXT91 frozen artifact and the physically isolated
internal-validation endpoint directory. It has no discovery or replication
endpoint argument. It joins predictions to endpoints by exact material ID and
evaluates without changing any term, transform, direction, center, scale,
weight, threshold, missing policy, or subgroup behavior.

Validation gates are exactly:

- support coverage Wilson lower >= 0.90;
- protected recall Wilson lower >= 0.95;
- severe-rejection precision Wilson lower >= 0.80;
- total savings Wilson lower >= 0.02;
- pooled protected-versus-severe AUC >= 0.75;
- macro lattice AUC >= 0.65;
- worst eligible lattice AUC >= 0.55;
- at least eight lattice classes contain both protected and severe examples;
- RLI rejects more severe rows than validation Pauling P2--P5 and has a higher
  severe-rejection precision Wilson lower bound.

Exact per-lattice metrics are always reported. Failure stops the protocol and
leaves replication unopened. No rescue or threshold calibration is permitted.

## NEXT93: one-shot internal replication

Only an exact NEXT92 pass authorizes NEXT93 to open the physically isolated
internal-replication endpoint. NEXT93 uses the same frozen predictions and the
same gates. A pass supports an RLI candidate-law claim only for the SCIGEN
post-prescreen domain; it does not justify a universal-crystal or
DFT-equivalence claim.

