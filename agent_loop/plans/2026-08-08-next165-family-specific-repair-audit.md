# NEXT165 Family-Specific Repair Audit Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: use `superpowers:test-driven-development`
> and `superpowers:verification-before-completion` while executing this plan.

**Goal:** Determine whether one frozen physical mechanism family identifies a
cross-source protected subpopulation inside the exact NEXT164 repair shell,
without searching a law or using DFT.

**Architecture:** Reconstruct the exact NEXT164 closest candidate and its
published score, decompose its 11 analytic contributions into the four frozen
NEXT157 families, and audit a finite family-concentration catalogue on the
fixed discovery shell. Publish only a ranking and one eligible statistic for a
separately frozen later law search.

**Tech stack:** Python, NumPy, pandas, scikit-learn ROC AUC, parquet, pytest.

## Boundary and alternatives

This additive audit opens no internal-validation or replication endpoint and
searches no formula, cutoff, weight, or threshold. Discovery outcomes are
offline labels only. No DFT calculation or value, learned
energy/force/stress proxy, or relaxation may be used.

Three approaches were considered:

1. **Family-specific concentration audit (selected):** tests whether a false
   positive is dominated by one coherent analytic mechanism family. It is the
   smallest new decision-topology hypothesis consistent with NEXT164.
2. A new global audit over all unused numeric descriptors: rejected because
   NEXT139--150 already showed that marginal AUC often fails at the common
   threshold and invites repeated feature mining.
3. Direct search over family-conditional exemptions: rejected at this stage
   because choosing cutoffs and weights before establishing a cross-source
   family signal would add avoidable degrees of freedom.

## Frozen candidate and population

Use exactly the NEXT164 global-closest candidate:

```text
dominant-family attenuation gamma = 0.075
coordination protection alpha     = 0
packing protection beta           = 0.5
SAFE threshold                    = 0.5415470292150686
closest BROAD threshold           = 0.21976295573076796
```

Use protected discovery endpoint `<= 1`, severe endpoint `>= 2`, the frozen
reduced-formula five-fold split, and the exact repair shell

```text
0.21976295573076796 <= published score < 0.5415470292150686
```

Evaluate SCIGEN shell, WyFormer shell, SCIGEN full extremes, and WyFormer full
extremes. The score and both thresholds are inputs from the completed NEXT164
diagnostic and cannot change after this plan is hashed.

## Frozen family statistics

For weighted analytic contributions `c_i >= 0`, use the unchanged NEXT157
families and define

```text
M_f = mean_{i in f} min(c_i, 0.5)
T   = sum_f M_f
```

Audit exactly 15 statistics. For each of the four families
`local_geometry`, `charge_flow_feasibility`, `valence_transport`, and
`contact_robustness`, include:

```text
share_f       = M_f / T
margin_f      = max(0, M_f - max_{g != f} M_g)
is_dominant_f = I[M_f > 0 and M_f = max_g M_g]
```

All 12 family-specific statistics have the frozen direction “larger means
protected.” Ties may activate more than one `is_dominant_f`; this is
deterministic and avoids a family-order tie break. Also include:

```text
largest_family_share    = max_f M_f / T                 (larger protected)
effective_family_count  = T^2 / sum_f M_f^2             (smaller protected)
normalized_family_entropy = -sum_f p_f log(p_f)/log(4)  (smaller protected)
```

When `T=0`, shares, margins, indicators, effective count, and entropy are
exactly zero. No additional statistic, direction, family, transform, or
ranking rule may be introduced after execution begins.

## Frozen eligibility and termination

For each statistic, compute protected-versus-severe AUC in the four
populations. Eligibility requires:

- SCIGEN repair-shell worst-fold AUC `>= 0.55`, with all five folds evaluable;
- WyFormer repair-shell pooled AUC `>= 0.55`;
- SCIGEN and WyFormer full-extreme pooled AUC each `>= 0.50`.

Rank eligible statistics by the minimum of those four AUCs, then their mean,
then statistic name. If none is eligible, terminate this branch. If one or
more are eligible, only the top-ranked statistic may enter a separately frozen
NEXT166 conditional-law search. NEXT165 itself cannot claim a replacement or
authorize validation.

## Implementation tasks

### Task 1: Unit contract

**Files:**

- Create: `tests/test_next165_family_specific_repair_audit.py`
- Create later: `src/next165_family_specific_repair_audit.py`

1. Write a test for the exact 15-statistic schema and hand-calculated shares,
   margins, dominance indicators, effective count, and entropy.
2. Write a test for the all-zero row and invalid family coverage.
3. Run with `python -m pytest` and
   confirm failure because the module does not exist.
4. Implement only the tested statistic function and rerun to green.

### Task 2: Frozen audit runner

**Files:**

- Modify: `src/next165_family_specific_repair_audit.py`

1. Reuse the frozen NEXT163 score reconstruction and NEXT151 AUC evaluator.
2. Verify all NEXT164 input hashes, protocol flags, candidate identity,
   thresholds, and no-DFT/no-validation provenance before reading endpoints.
3. Build the four populations, rank the fixed catalogue, and atomically write
   the manifest, JSON audit, and complete parquet table.
4. Refuse to overwrite an existing formal output directory.

### Task 3: Formal execution and verification

1. Run the unit test and the full NEXT142--NEXT165 directed suite.
2. Execute once into
   `$PRIS_ARCHIVE/next165_family_specific_repair_audit_v1`.
3. Independently verify every output hash and all no-DFT/no-validation flags.
4. Update only the standalone research report after the formal result is
   known; do not edit canonical manuscript/report artifacts.
