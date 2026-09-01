# NEXT53 ODAC23 extraction deviation and recovery record

## Event

After the split protocol was frozen, archive headers showed that the official
archive contains `train` and `val` targets, while the packaged test and OOD
directories are named `*_no_targets`.  The first extraction command used a
non-anchored member-name option whose match was broader than intended.  It
copied every `.lmdb` member, including non-training members, into the temporary
development extraction root.

No LMDB from `val`, `test_no_targets`, or any `ood_*_no_targets` directory was
opened, deserialized, queried, or used to inspect structure, labels, or record
counts.  Only archive member names had been inventoried at that point.

## Immediate recovery

Before any LMDB payload was deserialized, all non-training directories were
moved byte-for-byte to the independent quarantine root:

`$PRIS_ARCHIVE/next53_odac23_accidental_nontrain_quarantine_v1/is2r/`

The quarantined directories are:

- `val`;
- `test_no_targets`;
- `ood_big_no_targets`;
- `ood_linker_no_targets`;
- `ood_topology_no_targets`;
- `ood_linker_topology_no_targets`.

The development raw root now contains only the 200 official training LMDB
shards under:

`$PRIS_ARCHIVE/next53_odac23_development_raw_v1/is2r/train/`

## Consequence

The intended payload firewall was restored before deserialization, so no
confirmation payload informed feature or formula development.  The accidental
copy is nevertheless a protocol deviation and must remain disclosed in every
NEXT53/NEXT54 standalone report.

The available official package also narrows the eventual one-shot confirmation
claim: `val` has targets, whereas the packaged test and OOD splits do not.  A
successful validation result cannot be described as public ODAC23 OOD endpoint
confirmation unless a separately authenticated target source is obtained.
