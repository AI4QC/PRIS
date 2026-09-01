# NEXT54 ODAC23 train representative and partition design

## Purpose

Create a computationally tractable, leakage-resistant development cohort from
the sealed NEXT53 ODAC23 train-only artifact.  This design is frozen before any
row-level endpoint is joined to a selected x0.

## Why exact x0 groups are not merged by framework name

NEXT53 contains 118,091 exact framework geometries under 7,815 official
framework names.  A geometry-only audit found both harmless serialization
jitter and genuine differences in atom identity, supercell, cell, or
coordinates among records sharing a name.  Therefore same-name geometries must
not be averaged, aligned, or tolerance-clustered into a synthetic x0.

## Label-independent representative selection

For every exact NEXT53 row, calculate

`SHA256("NEXT54-REP-v1\0" + geometry_sha256)`.

Within each `framework_name`, retain the row with the lexicographically smallest
representative hash.  Ties are broken by `material_id`.  This chooses one real,
unchanged raw x0 per official framework name without consulting a DFT endpoint.
All non-selected NEXT53 records remain in the prior immutable artifact.

## Framework-isolated train partitions

Calculate

`SHA256("NEXT54-SPLIT-v1\0" + framework_name)`

and interpret the first eight digest bytes as an unsigned big-endian integer
divided by `2**64`.

- `[0.00, 0.60)`: discovery;
- `[0.60, 0.80)`: internal validation;
- `[0.80, 1.00)`: internal replication.

Because selection and partitioning use only frozen identity and raw-geometry
hashes, row-level DFT displacement labels cannot influence membership.  No
framework name may occur in more than one role.

## Opening order

1. Publish selected metadata, partitions, and selected x0 archive from NEXT53
   metadata/geometries only.
2. Build and seal all executable x0 features without opening the selected label
   table.
3. Join discovery labels for finite formula search.
4. Freeze one candidate before opening internal-validation labels.
5. If it passes, freeze it again before opening internal-replication labels.
6. Only a candidate passing all train roles may be frozen for one-shot official
   `val` confirmation.

## Endpoint and gates

Each selected row keeps its NEXT53 exact-x0 endpoint: the median over exact
duplicates of per-relaxation framework p95 displacement.  Protected and severe
thresholds and all advancement gates remain exactly those frozen in the
NEXT53 split protocol.  Missing executable features force `KEEP`.
