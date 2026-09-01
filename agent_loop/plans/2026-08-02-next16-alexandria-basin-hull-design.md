# NEXT16 Alexandria Basin-Hull external validation design

## Status and evidence boundary

This is an additive experiment. Existing scripts, outputs, reports, and paper
sources remain unchanged.

NEXT15 fixed the Basin-Hull boundary at 0.20 eV/atom before its WBM execution.
That rule strongly improved high-energy rejection and stable retention but did
not exceed Pauling's absolute rejection count. After the WBM labels were open,
a threshold sweep selected 0.15 eV/atom as the least aggressive round-numbered
candidate whose five retrospective WBM gates all passed. Therefore 0.15 is a
development-selected NEXT16 candidate, not an independent WBM confirmation.

## Frozen candidate

For an initial structure `x0` with composition `c`, perform a full-cell
MatterSim 5M FIRE relaxation with the same NEXT15 settings:

- `FrechetCellFilter`
- `fmax = 0.05 eV/angstrom`
- at most 64 prediction steps
- volume-ratio support interval `[0.25, 4.0]`
- failures and unsupported geometries become `ABSTAIN`

Define

`B64(x0) = E_MatterSim(relax64(x0))/N - E_raw_MP_hull(c)`.

The frozen NEXT16 decision is `REJECT` exactly when
`B64 >= 0.15 eV/atom`; otherwise it is `KEEP`.

No Alexandria endpoint may be used to change the formula, threshold,
relaxation settings, cohort, or failure policy.

## External source qualification gate

Use the official Alexandria PBE geometry-optimization paths dated 2025-07-02.
Before acquiring more than a bounded pilot shard, verify all of the following:

1. records contain a stable material identifier and an ordered relaxation path;
2. the official `benchmarks_pbe.csv` identifiers can be matched without reading
   an energy or stability field to select survivors;
3. `m3gnet/rng` provenance or an equivalent official benchmark-cohort identity
   can be reconstructed;
4. the initial structure is available independently of the final DFT endpoint;
5. a complete, deterministic cohort can be selected by identifier hash before
   labels are evaluated.

If these conditions fail, stop at a data-qualification result. Do not silently
substitute final relaxed structures or label-selected survivors.

## Validation contract

The geometry stage publishes only identifiers, compositions, atom counts, and
initial structures. The label-free feature stage consumes only that artifact
and records that no endpoint bytes were read. Endpoint labels are joined only
after the candidate feature artifact is sealed.

Primary metrics are coverage, DFT savings, stable/valuable recall, high-energy
rejection recall, and reject precision, with Wilson intervals and paired
identifier-group bootstrap comparisons to the classical Pauling 2--5 control.
The same five superiority clauses used in NEXT14/NEXT15 apply. Alexandria is an
independent source but not claimed as a fresh blind lockbox because its 2025
metadata and some final structures existed in earlier local exploratory work.

No standalone success report is written unless the source gate is satisfied and
the frozen 0.15 candidate passes the external metrics. Even then, canonical
reports and paper files wait for user confirmation.
