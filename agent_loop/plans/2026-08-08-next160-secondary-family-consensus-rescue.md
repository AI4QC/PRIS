# NEXT160 secondary mechanism-family consensus rescue freeze

## Boundary and motivation

NEXT158's top-ranked capped-family-mean branch yielded three AUC+SAFE12
candidates but no BROAD pass. NEXT159 showed that all three residual profiles
are identical and fail only SCIGEN `protected_kept` constraints at the closest
BROAD threshold. This motivates a stricter notion of cross-mechanism
concurrence: a structure should not be rejected merely because one physical
family has a large violation.

NEXT157 had already declared exactly two additional eligible family statistics:
`family_max_sum_minus_largest` and `family_max_second`. NEXT160 admits both and
only those two previously eligible secondary statistics. It excludes the
failed top-ranked `family_capped_mean_sum` and every statistic that failed the
NEXT157 eligibility gates.

This is an additive pre-DFT discovery search. Formula execution uses only
frozen analytic structure/composition features and assigned formal oxidation
states. It performs no DFT calculation, reads no DFT value, uses no learned
energy/force/stress proxy, and performs no relaxation. Discovery outcomes are
offline labels only. Validation and replication endpoints remain sealed; no
prior or canonical artifact is replaced.

## Frozen physical families and scores

Each weighted nonnegative contribution `c_i = w_i R_i` is assigned by term-ID
prefix to exactly one frozen family:

- local geometry: `cov_`, `scbv_`, `sivr_`;
- charge-flow feasibility: `cmvo_`, `hcid_`;
- valence transport: `bvtbd_`, `bvtc_`;
- contact robustness: `mhcr_`.

For each family `f`, define `X_f(x) = max_{i in I_f} c_i(x)`. The two admitted
base aggregations are:

```text
A_excl1(x) = sum_f X_f(x) - max_f X_f(x)
A_second(x) = second_largest_f X_f(x)
```

For either frozen aggregation `A`, the searchable law is

```text
S(x) = max(0, A(x) - alpha P_coord(x) - beta P_coord-pack(x))
```

`P_coord` is the bounded NEXT129 coordination protection and `P_coord-pack` is
the bounded NEXT135 coordination-by-covalent-packing product. Unsupported
protections are inactive and cannot enlarge physical-base support.

## Frozen candidate universe

- 2 aggregations listed above, with no tunable aggregation parameter;
- 11 exact NEXT132-selected bases;
- `alpha in {0, 0.5, 1, 2}`;
- `beta in {0, 0.1, 0.25, 0.5}`;
- 352 candidates total;
- base-formula SHA-256:
  `d1f8763331cbe36f54e898e4efc88d0f88d2ae5d6284883acc4850e58d9678b5`;
- candidate-key SHA-256 over newline-joined sorted canonical JSON keys:
  `d61b06bc2a117c208e2fdbd68dece32dd933b4a1f5201b93f03b20fc01d2e235`.

No aggregation, family, prefix, base, term weight, protection, grid point,
threshold rule, gate, or candidate ordering may change after outcomes open.

## Evaluation and stopping rule

Use the unchanged grouped-fold cross-source discovery evaluator, source-AUC
gates, SAFE12 gates, BROAD gates, and deterministic selection. Verify exact
score construction, unchanged support, subtractive-only active protections,
one-family-per-term coverage, and all 352 identities before publication.

If no candidate passes every discovery gate, terminate this secondary-family
branch and keep validation/replication sealed. If a candidate passes, freeze
its complete formula and threshold before any separate one-shot validation.
NEXT160 itself never opens validation/replication outputs and makes no
scientific improvement claim.

## Outputs

Atomically publish a manifest, catalogue, discovery evaluation, and complete
candidate table under
`$PRIS_ARCHIVE/next160_secondary_family_consensus_rescue_v1`.
Only the standalone research report may be updated after the run.
