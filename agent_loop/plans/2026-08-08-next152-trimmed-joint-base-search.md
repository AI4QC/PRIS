# NEXT152 trimmed joint-base search freeze

## Scope and hard boundary

This is an additive discovery-only experiment.  It does not replace any prior
script, result, report, or formula.  It must not read an internal-validation or
replication endpoint.  Discovery outcomes may be used only as offline labels;
the executable score must use crystal structure, composition, assigned formal
oxidation states, and the already frozen analytic features.  It must not use a
DFT calculation, a DFT value, a learned energy/force/stress proxy, or a physical
relaxation.

The only hypothesis admitted from NEXT151 is its uniquely eligible aggregation,
`sum_minus_top2`.  No other NEXT151 statistic may enter this search.

## Frozen physical score

For one of the 11 bases already frozen by the NEXT132 selection, let

```text
c_i(x) = w_i R_i(x) >= 0
```

be its 11 weighted, nonnegative physical-risk contributions.  Sort the
contributions in descending order as `c_(1) >= c_(2) >= ... >= c_(11)` and set

```text
T(x) = sum_i c_i(x) - c_(1)(x) - c_(2)(x)
S(x) = max(0, T(x) - alpha P_coord(x) - beta P_coord-pack(x))
```

`P_coord` is the bounded NEXT129 coordination protection.
`P_coord-pack` is the bounded NEXT135 coordination-by-covalent-packing product.
If a protection is unsupported it is inactive and the supported trimmed base is
kept.  Base support is the intersection of the 11 physical term supports and is
not enlarged by either protection.

The interpretation is deliberately narrow: a generated structure is high risk
only when several independent physical checks remain violated after its two
largest isolated discrepancies are treated as possible benign special cases.

## Candidate universe frozen before search

- Bases: the 11 exact NEXT132-selected base formulas.
- Aggregation: `sum_minus_top2` only.
- Coordination protection weights `alpha`: `0, 0.5, 1, 2`.
- Coordination-covalent-packing weights `beta`: `0, 0.1, 0.25, 0.5`.
- No volume protection and no ACSB, analytic-field, conditional-exemption, or
  charge-order term.
- Candidate count: `11 * 4 * 4 = 176`.
- Frozen base-formula SHA-256 over sorted formula identities:
  `d1f8763331cbe36f54e898e4efc88d0f88d2ae5d6284883acc4850e58d9678b5`.
- Frozen candidate-key SHA-256 over newline-joined sorted canonical JSON keys:
  `6433e8f2b1fab1235a37ecbe53011a466eb26d6d8bd458578fa36729b98fc058`.

The two grids reuse previously searched, interpretable protection scales while
allowing for the smaller trimmed-base scale.  They are fixed here and may not be
changed after any NEXT152 endpoint result is viewed.

## Evaluation and selection

Use the unchanged grouped-fold, cross-source discovery evaluator and its frozen
source-AUC, SAFE12, and BROAD gates.  Every candidate receives the same threshold
search and cell diagnostics.  Selection is exactly the existing evaluator's
deterministic ordering; no manual exception or endpoint-specific choice is
allowed.

Before accepting the search, prove numerically that:

1. the contribution implementation equals row-wise `sum - largest - second`;
2. zero protections reproduce the trimmed aggregation exactly;
3. protection terms only subtract on active rows and never change base support;
4. all 176 frozen candidate identities are present exactly once; and
5. the code and all input artifacts are unchanged between hashing and atomic
   publication.

If no candidate passes all discovery gates, terminate the trimmed-joint-base
branch and keep validation/replication sealed.  If at least one candidate passes
all gates, freeze the selected formula and threshold first; only then may a
separate, explicitly recorded one-shot validation step be considered.  NEXT152
itself never opens validation or replication data and makes no scientific
improvement claim.

## Additive outputs

The formal output directory is
`$PRIS_ARCHIVE/next152_trimmed_joint_base_search_v1`.
It contains a manifest, frozen catalogue, discovery evaluation, and complete
candidate table.  Results are summarized only in the standalone report; no
canonical paper, README, preregistration, note, or TeX file is modified.
