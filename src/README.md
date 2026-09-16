# PRIS/src — Week 1 data scripts

Single environment: `python` (no new conda environment).
Before running anything in parallel: `export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.
Output directory: `$PRIS_FEATURES/`.

---

## build_energetics.py

The main path for filling in the energy fields of `materials.sqlite`, which are all NULL.

```bash
python src/build_energetics.py --limit 2   # smoke test
python src/build_energetics.py             # full run, measured at 41 s
python src/build_energetics.py --force     # ignore the idempotence skip
```

Outputs:
| file | rows | size | granularity |
|---|---|---|---|
| `energetics_mp.parquet` | 154,377 | 14.0 MB | one row per MP material (25 columns) |
| `icsd_mp_link.parquet` | 73,823 | 6.6 MB | one row per ICSD experimental structure, with E_hull backfilled |

### Measured schema notes (differences from the plan's §14 skeleton)

- Each page of the MP summary has 69 top-level fields, `{"data": [...1000 entries...], "meta": {...}}`.
- `material_id` is an **alphabetic hash** (`mp-aaahikie`), no longer the old `mp-1234`. Any
  downstream code that parses mp-ids as numbers will break.
- `database_IDs` is **`null`** for the great majority of entries (not an empty dict). The plan's
  `d.get("database_IDs").get("icsd")` raises `AttributeError`; it has to be written
  `(d.get("database_IDs") or {}).get("icsd") or []`.
- Elements of `database_IDs["icsd"]` measured **100% with an `icsd-` prefix** (`exp004`), the same
  shape as `source_id` in `synth_meta.tsv`. The join key is still normalised to an integer
  collection code, which is more robust to `ICSD-xxx` or bare-number forms appearing later.
- There is also `database_IDs["pauling"]` (15,765 entries, the Pauling File) — same name as the
  project, but nothing to do with Pauling's five rules. Not currently used.
- `symmetry` is a dict; the space-group number is at `symmetry.number`, not at top-level
  `spacegroup_number`.
- `deprecated == True` measures 0 entries; the snapshot is already cleaned.

### Known data pitfalls

- **4 entries have NULL `energy_above_hull` / `formation_energy_per_atom` / `band_gap`**, all
  elemental `Yb` (mp-aaacdikq / mp-aaacppfn / mp-aaaaaact / mp-aaaaaagg). MP's Yb pseudopotential
  has a known problem. Two of them carry ICSD IDs (exp005, exp003), which is why "58,246 joined by
  ID" but "only 58,244 got an E_hull".
- **2,333 entries are `theoretical=True` yet carry an ICSD ID** (4.52% of entries with an ICSD ID).
  ICSD is an experimental database, so this is semantically contradictory. The median e_hull of
  these entries is 0.110 eV/atom (far above the 0.0034 of the full experimental set); they are
  probably conservative labels MP applied to high-energy polymorphs or ambiguous matches, or
  hypothetical structures deposited in ICSD.
  **Downstream, do not trust the `theoretical` field alone when selecting "experimentally known";
  use `theoretical==False OR n_icsd>0` and record the flag.**
- One ICSD code may be referenced by several MP entries (481 measured). `build_icsd_link` takes the
  one with the **smallest e_hull** as representative and keeps the match count in the
  `n_mp_matches` column so downstream code can judge the ambiguity.

### Idempotence

If the outputs exist and their mtime is later than every input (155 gz files + synth_meta.tsv), the
computation is skipped, but the parquet files are still re-read and the full report printed.

---

## build_provenance.py

The provenance table for the experimental set. Input `materials.sqlite` (read-only) +
`synth_meta.tsv`, output `features/provenance.parquet` (99,162 rows × 46 columns, 11.1 MB zstd,
about 3 s for a full run).

```bash
python src/build_provenance.py --limit 200 --dry-run   # smoke test
python src/build_provenance.py                         # full run
python src/build_provenance.py --force
```

### Measured vs expected

| item | measured | expected | |
|---|---|---|---|
| rows | 99,162 | — | all of sqlite `dataset='experimental'` |
| join hit rate | **100.0000%** | never verified by the main agent | see below |
| `n_sites == n_atoms` | 100% | — | independent corroboration that the join is right |
| `orig_spg` missing | 394 (0.40%) | — | the middle segment of bawl_hash is empty |
| `orig_spg == spacegroup_number` | **98.9446%** (98,727 comparable, 1,042 disagree) | — | see below |
| `in_analysis_set` | **38,307** (icsd 27,408 + cod 10,899) | 38,307 | reproduced exactly |
| `oxide_strict` | **23,728** (icsd 16,414 + cod 7,314) | 23,728 | reproduced exactly |

### The join key (verified)

`materials.source_index` is an integer row number stored as TEXT; within the experimental set it
runs 0..99161, contiguous and unique. `synth_meta.index` is likewise 0..99161 and unique. The two
are a strict bijection: 100% hit, nothing missing.
`source_split` (train 94,204 / val 4,958) does **not** participate in the numbering, so no per-split
offset is needed.
The SYN path written in plan §14 is an old copy under CSAgent; the live copy is at
`matdata/data/sources/experimental/synth_meta.tsv`, and the script uses that one.

### orig_spg vs spacegroup_number

`bawl_hash` = `<32-char md5>_<original space-group number>_<reduced formula>`, e.g.
`f5e701d27fce4666407370fabff735f6_14_Cr4Te8O22`.

Of the 1,042 disagreements, **100% have sqlite's number > bawl's original**, none the other way.
That is, the sqlite side used a looser symprec and assigned the structure to a higher-symmetry
group. Frequent transitions: 14→62, 123→221, 139→225, 11→63, 2→12, 47→123, 69→139, 160→215 — all
standard "subgroup rises to supergroup once the tolerance is loosened" paths, not corrupted data.
By source: exp008 / cod 256.

Nulls: sqlite `spacegroup_number` is NULL for 435 entries and `orig_spg` is NA for 394; the latter
is a **strict subset** of the former (394 entries where neither can decide, plus 41 where bawl can
and sqlite cannot).

**Use**: the `spg_agree` column (nullable boolean) is a ready-made proxy for symprec sensitivity.
Those 1,042 entries sit on the boundary of the symmetry determination and should be looked at
separately when testing laws, so that a symprec artefact is not read as a physical regularity.

### The two analysis-set definitions — note that neither contains the other

- `in_analysis_set`: **exactly one** of {O,S,Se,Te,N,P,F,Cl,Br,I}, and no H and no C.
  Anion distribution: O 19,833 / S 4,783 / F 2,839 / Se 2,745 / N 1,673 / P 1,663 / Te 1,558 /
  Cl 1,434 / I 909 / Br 870.
- `oxide_strict`: contains O and none of H/C/N/F/Cl/Br/I/S/Se/Te. **P is not on the exclusion list.**

Intersections: `ox ∩ set` = 19,833 (single anion, and that anion is O); `ox \ set` = 3,895, and
**all 3,895 contain P** (phosphates are rejected by `in_analysis_set` as two-anion systems but kept
by `oxide_strict`); `set \ ox` = 18,474 (single-anion non-oxide systems). Downstream sampling must
say explicitly which one it uses.

### Output columns

`pk, material_id, source, source_id, bawl_hash, bawl_md5, bawl_formula, orig_spg,
spacegroup_number, spacegroup_symbol, crystal_system, spg_agree, formula, chemical_system,
elements, n_elements, n_atoms, n_sites, anion, n_anion_kinds, in_analysis_set, oxide_strict,
blob_offset, blob_length, dataset, source_split, source_index, is_experimental, lattice_*`,
plus 12 boolean `has_<element>` columns (O/S/Se/Te/N/P/F/Cl/Br/I/H/C).
`blob_offset`/`blob_length` point straight into `structures.blob`, so downstream code can fetch a
CIF without going back to sqlite.

### Built-in assertions (they fire when a definition changes)

Join hit rate < 99.9%, `n_sites`/`n_atoms` agreement < 99%, non-numeric or duplicated
`source_index`, or a `bawl_hash` that is not three segments → raises. On a full run,
`in_analysis_set != 38307` or `oxide_strict != 23728` → AssertionError. Both of the latter are
skipped under `--limit`.

---

## build_icsd_meta.py

Extracts metadata by pure regex from 203,830 raw ICSD CIFs
(`<other-repo>/data/icsd_extracted/cif/*.cif`, 1.1 GB, CRLF line endings) into
`features/icsd_meta.parquet` (203,830 rows × 15 columns, 3.7 MB). This is the input to the §10.1
self-reference-over-time analysis.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=python
$PY src/build_icsd_meta.py --limit 4000        # smoke test (evenly spaced stride sample, ~1 s)
$PY src/build_icsd_meta.py --workers 16        # full run, measured at 27 s
$PY src/build_icsd_meta.py --force             # force a recompute even when the output is current
```

Idempotence criterion: skip if the output's mtime > max(CIF directory mtime, this script's mtime)
(rather than stat-ing 200,000 files individually).

### Column notes and downstream traps

| column | note |
|---|---|
| `source_id` | `icsd-<N>`, same convention as `source_id` in `synth_meta.tsv`, so it joins directly |
| `pub_year` | **year of publication**, from `_citation_year` in the citation loop. Only this column may be used for a time axis |
| `audit_year` | **year of FIZ entry**, from `_audit_creation_date`. **Its only legitimate use is as an entry-batch fixed effect** |
| `lag_years` | `audit_year - pub_year`, only for checking the §10.1.2 quantiles; **do not** use it to fill gaps in `pub_year` |
| `temperature` | in **K** |
| `pressure_kpa` | in **kPa**. For GPa you must **divide by 1e6** |
| `R_factor` | `_refine_ls_R_factor_all`, normally 0–1; 50 entries measure >1 (entered as a percentage), so clip before modelling |
| `n_reflns` / `n_params` | **measured entirely empty**, see below |
| `n_year_cand` | diagnostic column, the number of distinct years matched in that file; measured as 0 or 1 throughout the corpus |

### Measured vs the plan's expectations (full run of 203,830, 2026-07-28)

All six hit rates agree digit for digit with plan §14 (pub_year 96.73% / audit_year 100% /
R_factor 64.73% / temperature 35.02% / pressure_kpa 3.64% / structure_type 78.13%); the lag
quantiles p5/p25/p50/p75/p95/p99/max = 0/1/2/17/44/57/95 match §10.1.2 exactly; and the
decade-by-decade histograms of pub_year and audit_year agree bucket for bucket.

### Three places where reality differs from the plan

1. **`n_reflns` / `n_params` are entirely empty.** The PAT dictionary in plan §14 lists
   `_refine_ls_number_reflns` / `_refine_ls_number_parameters`, but the local ICSD corpus **does not
   contain these fields at all** (0 hits when grepping a random sample of 5,000 files; across the
   whole corpus the only `^_refine*` that exists is `_refine_ls_R_factor_all`). Both columns are
   kept only for schema compatibility, the script emits a warning, and **downstream code must not
   use them**. The §10.1.3 quality covariates have to rest on `R_factor` + `has_aniso_adp` +
   `is_powder`.
2. **`--limit` defaults to an evenly spaced stride sample, not the first N.** ICSD numbering
   correlates strongly with date (the first 500 are all 1970s, all audited in 1980), so
   `--limit 500 --limit-mode head` gives thoroughly misleading hit rates such as 5.8% for
   temperature. Smoke tests must use the stride (the default).
3. **The rule "take the earliest when the citation loop has several" never fires in this corpus**
   (`n_year_cand > 1` is 0 throughout). The `min(yrs)` in the code stays, but it does nothing.

### Known dirty data (the script reports and does not clean, so rows stay one-to-one with the raw CIFs)

- `lag_years < 0` in 114 entries (0.058%), of which 104 are −1 (most likely in-press entries, not a
  regex error); the handful at −14/−19 are the regex catching a volume number. Negligible; filter
  with `lag_years >= -2` when modelling.
- `temperature` up to 7,493 K and `pressure_kpa` up to 1.7e9 kPa (= 1,700 GPa) are both raw ICSD
  entry errors.
- `pub_year` is missing for 6,672 entries (3.27%); the citation loop in those entries is laid out in
  neither of the two known ways.

---

## build_cod_delta.py

The one lifeline for the §8.3 L3 temporal-extrapolation hold-out: pick out of the full COD the clean
carbon-free post-2019 entries that matdata does not yet hold.

```bash
python src/build_cod_delta.py --limit 3000 --min-year 2004   # smoke test (really does fetch cifs)
python src/build_cod_delta.py                                # full run, measured 151 s / peak RSS 6.0 GB
python src/build_cod_delta.py --no-cif                       # counts and histograms only, seconds
```

Outputs:

| file | rows | size | contents |
|---|---|---|---|
| `cod_delta.parquet` | 2,439 | 386 MB | the post-2019 delta, 26 columns + `cif_text` |
| `cod_delta_meta_all.parquet` | 50,122 | 3.2 MB | everything "clean, carbon-free and not in matdata", **without cif_text**, for the §8.3 data-source hold-out |
| `cod_year_hist.csv` | 39 | 1.2 KB | 5 filter levels × year (1990–2026, plus rows `-1` = <1990 and `-2` = year missing) |
| `cod_delta_stats.json` | — | — | per-level counts / runtime / peak RSS / the plan's expected values, for reconciling round by round |

### Memory strategy for a 30 GB file (this script's chief risk)

Measured: of the 30.4 GB compressed, the `cif_text` column alone is **30.3 GB (99.7%)** and the
other 74 columns come to 0.1 GB; decompressed the whole table is **117 GB**, and the 54 row groups
are wildly uneven (rg0 = 0.08 GB, rg51 = 9.4 GB).

- **Stage 1 (filtering)**: duckdb `SELECT * EXCLUDE (cif_text)` pulls the whole table into pandas
  for 0.19 GB in **1.0 s**.
- **Stage 2 (fetching CIFs)**: pyarrow `iter_batches(batch_size=256, use_threads=False)` with
  `ParquetFile(..., pre_buffer=False)`. **144 s, peak RSS 6.0 GB.**
- **Why stage 2 does not use duckdb**: page-level indexing would take this step down to about 12 s,
  but duckdb makes non-spillable raw allocations for large string pages, and a `memory_limit` of
  both 8 GB and 12 GB measured `failed to allocate data of size 512.0 MiB`; at the 10 GB that does
  work, peak RSS is already 8.2 GB. Not worth it on a shared 23 GB machine.
- **pyarrow also needs `pre_buffer` off**: streaming the full table with it on peaks at 8.3 GB RSS;
  off, the largest row group only reaches 4.2 GB.
- The `file` column is **not ordered** across row groups (rg0 min/max = 1000000/7206032,
  rg53 = 1576311/7720906), so statistics-based pruning does nothing. Instead, pre-scan by reading
  only the `file` column (<10 MB) to locate the row groups needed. A full run hits **17/54**
  (post-2019 entries cluster in the later groups); a smoke test often needs just one.

### The carbon-free criterion: how it differs from the plan's regex

The measured `formula` format is `- C5 H17 Al N2 O8 P2 -` (a `-` at each end, elements in Hill
order, whitespace separated); **532,993** of 533,486 rows match it, and the remaining **493 rows are
the literal `?`**.

The plan's `~formula.str.contains(" C[0-9 ]")` does **not** actually fail on this format — the
tokenising criterion this script uses instead (trim the `- ` → split on whitespace → take
`^[A-Z][a-z]?` from each token as the element symbol → test `== "C"`) agrees with it **row for row**
(74,029 == 74,029). The only difference is the **493 rows with `formula = "?"`**: the regex admits
them as "carbon-free", while the token criterion excludes them with an extra `formula_ok` flag
(unknown composition ≠ carbon-free).
So `clean_noC` = 73,536 (token) vs 74,029 (the plan's regex), and the difference of 493 comes
entirely from this one point, and is **entirely pre-2019**, so it does not affect the post-2019
denominator.

Where the token criterion is more robust: it does not depend on the positional assumption that "C
must be followed by a digit or a space". If `formula` ever loses its leading and trailing `-`, or
gains a space between element and count, the regex fails silently and the token criterion does not.

### Measured vs the plan's expectations (full run, 2026-07-28)

| filter level | plan's [measured] expectation | measured here | Δ |
|---|---|---|---|
| total rows | 533,486 | 533,486 | 0 |
| clean (`status` empty) | 526,854 | **533,284** | +6,430 |
| clean and carbon-free | 72,197 | **73,536** (74,029 on the plan's regex) | +1,339 / +1,832 |
| and not in matdata | 48,782 | **50,122** | +1,340 |
| **and post-2019** | **2,454** | **2,439** | **−15** |

**Attribution of the difference (not fully reproduced; recorded as it is)**: `status` takes only
four values, `''` 533,284 / `retracted` 151 / `warnings` 41 / `errors` 10, and is **never NULL**, so
executing the plan's `status.isna() | (status=="")` literally can only give 533,284. All 64
combinations of `duplicateof IS NULL` / `optimal IS NULL` / `onhold` / `year IS NOT NULL` /
`formula<>'?'` / `flags LIKE '%has coordinates%'` were enumerated and **none gives (526,854, 72,197)
simultaneously**; the closest is `nodup+noopt+hasyear` → (527,213, 72,065).
Judgement: those two numbers in the plan most likely came from an earlier snapshot or a one-off
count with an extra filter, and are not reproducible against this snapshot.
**The number that matters, post-2019 = 2,439, is only 15 (0.6%) off the expected 2,454 and is
extremely insensitive to variations of the criterion** (dropping the status filter still gives
2,439; dropping `duplicateof` duplicates as well gives 2,438), so the conclusion is unaffected.

### Notes on the year histogram

Subtotals for 2019+ across the 5 filter levels: all 98,448 → clean 98,442 → clean carbon-free
**3,864** → not in matdata **2,439** (1,703 of the 2019+ entries are already in matdata). Year by
year: 2019 411 / 2020 466 / 2021 507 / 2022 316 / 2023 224 / 2024 186 / 2025 277 / 2026 52.
The inorganic (carbon-free) share of COD falls from about 50% in the 1990s to about 3% in the 2020s
— **COD is increasingly an organic/MOF database**, and that, not a mistake in the filter, is the
fundamental reason the 2,439 denominator cannot be raised.

The composition-level analysable subset (no H/D and exactly one anion, **not yet through the
disorder filter**): **1,072 entries**, at the top of the 700–1,100 range given in plan §14
[extrapolation]. Anion composition O 750 / S 112 / Se 84 / P 31 / F 27 / I 21 / Te 20 / Cl 19 /
N 6 / Br 2. The script writes the `has_H` and `n_anion_kinds` columns straight into both parquets.

### Other measured details

- `year` spans 1915–2026, empty for 619 entries (0.12%); 2026 already has 3,059 entries (the
  snapshot goes to 2025-08-21 and backfilling continues after that).
- **All** 25,339 COD entries in matdata match into `cod_full.parquet` by `file`: a 100% match rate.
- The `cif_text` of all 2,439 delta entries **contains `_atom_site_fract_x` 100% of the time**, so
  structures can be built directly; median length 33.8 KB, mean 356 KB, max 47 MB (entries with an
  Fobs table).
- `--limit N` takes the **first N rows** (it is not a sample), and COD's `file` numbering correlates
  with date, so counts from a smoke test are meaningless and only exercise the code path. To reach
  stage 2 in a smoke test, lower `--min-year` as well.

---

## build_features.py — the MPU-1 feature store (site / pair / struct)

**Environment: `python`** (not the csagent environment named at the top of this file; only the
newpauling environment has pymatgen + spglib 2.7.0 + pyfixest).

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python src/build_features.py --extract-icsd-ox --force   # 0. one-off, 11 s
python src/build_features.py --limit 400 --workers 20 --chunk 10 --force   # 1. smoke test, 1.4 min
nohup python src/build_features.py --workers 20 --chunk 20 --force \
      > $PRIS_FEATURES/build_features.log 2>&1 &   # 2. full run
tail -f $PRIS_FEATURES/build_features.log          # watch progress
```

Input: the 38,307 rows of `provenance.parquet` with `in_analysis_set==True` (ICSD 27,408 /
COD 10,899); CIFs are seek+read out of `structures.blob` by `blob_offset`/`blob_length` and then
decompressed with **zlib** (not zstd).

### The oxidation-state prerequisite: measured conclusions (the most important output of this script)

PREREG §5 freezes "only two levels are allowed, `cif` (native to ICSD) and `guess` (pure
compositional inference); BVAnalyzer-derived states are excluded wholesale".
`icsd_meta.parquet` carries no oxidation states, so the raw CIFs were checked first. Four
measurements:

| question | measured | scope |
|---|---|---|
| do the raw ICSD CIFs have `_atom_type_oxidation_number` | **27,408 / 27,408 = 100.00%** | all ICSD entries in the analysis set; 0 missing CIFs, 0 missing oxidation loops |
| do the CIFs in the blob keep the oxidation-state decoration (ICSD) | **27,408 / 27,408 = 100.00%** | as above |
| does the blob decoration agree with the raw CIF | **847,608 / 847,608 = 100.000%** | site-level comparison aligned by `_atom_site_label`, tolerance 0.005 |
| do the CIFs in the blob keep the decoration (**COD**) | **192 / 10,899 = 1.76%** | — |

Conclusions:

1. **The `cif` level exists, but it only covers ICSD.** PREREG §5 does not need overturning, but it
   does need a **clarification**: `cif` actually reaches only **27,408/38,307 = 71.5%** of the
   analysis set, and the 28.5% from COD has no native oxidation states.
   This should go into the PREREG revision log (append only, no rewriting).
2. **Use the blob decoration directly as `ox_source='cif'` rather than going back to the raw CIF.**
   Site-level agreement is 100.000%, and site-level mixed valence (Fe2+/Fe3+, Bi3+/Bi5+, Se⁻¹/Se⁻²,
   Au+/Au3+ and so on) is **preserved case by case rather than averaged**, with fractional valences
   (e.g. `Ru4.33+ 4.333`) kept at full precision. Going back to the raw CIF only adds a layer of
   label-alignment risk for no gain.
   `icsd_ox.parquet` (239,756 rows) is kept as the evidence for and audit entry point to this
   conclusion, not as an input to the main path.
3. **COD's 1.76% decoration cannot be trusted as `cif`.** There are no raw COD CIFs locally (no cod
   directory under `matdata/data/raw/acquired/`, only tcod), so the provenance cannot be verified —
   it may have come with the COD deposition, or it may have been filled in upstream by the matdata
   pipeline using BVAnalyzer. PREREG §5 excludes BVAnalyzer-derived states outright, and **what
   cannot be falsified may not be used**.
   → All of COD goes through `guess`, with `blob_ox_present` kept as a first-class column for later
   audit.
4. **BVAnalyzer is never called** (a PREREG §5 frozen item). Structures whose valences cannot be
   determined get `ox_source='none'`; their sites still enter the store with `ox_state` left NaN and
   are **not dropped** (the §6.2 tier-4 convention).

`guess` uses `Composition.oxi_state_guesses(max_sites=-1)` (reduce to the simplest formula first,
then enumerate, which is faster). When several solutions come back, the one scoring highest on ICSD
frequency is taken, and `n_guess_sol` / `guess_unique` are recorded — the "unique solution"
convention that §6.2 asks for can be recovered downstream with `guess_unique==True` without
rerunning.

### The three main tables

| file | granularity | note |
|---|---|---|
| `site.parquet` | one row per **cation** site | three algorithms' CN side by side (wide table) |
| `pair.parquet` | one row per connected polyhedron pair | ChemEnv route only |
| `struct.parquet` | one row per structure | aggregates + `status` + `wall_ms` |
| `failure.parquet` | one row per failed structure | `err_type` / `err` / truncated traceback |

**About the `nn_algo` column**: in `pair.parquet` it takes real values (currently always `chemenv`;
§9.1 specifies `len(d['ligands'])` from `sc.environment_subgraph()`, and a CrystalNN route would be
appended as extra rows later). `site.parquet` is a **wide** table where `nn_algo` is always
`'multi'`; to stratify, use the three pairs of columns `cn_chemenv` / `cn_crystalnn` / `cn_brunner`
and `ok_chemenv` / `ok_crystalnn` / `ok_brunner`.
Why wide and not long: G6 (robustness of the bond definition) needs **agreement between three
algorithms at the same site**, and a wide table lets you write `df.cn_chemenv == df.cn_crystalnn`
directly, where a long table would have to be pivoted back.

`ox_source` is a first-class column in all three tables.

### Algorithm parameters (each with its source)

- ChemEnv: `MultiWeightsChemenvStrategy.stats_article_weights_parameters()`,
  `maximum_distance_factor=1.41`, `only_cations=True` and **`valences=` passed explicitly**
  (§6.3 **pitfall A**: without valences, `only_cations=True` returns garbage). Structures with
  `ox_source='none'` have no valences to pass and fall back to `only_cations=False` (running all
  sites, which is slower), after which cation sites are selected by "not the anion element of that
  structure".
- CrystalNN: `weighted_cn=False`, `x_diff_weight=3.0` (§6.3 **pitfall B**: the pymatgen default is
  3.0, not 1.5).
- BrunnerNN_relative: default parameters.
- BVS: Brown–Altermatt `Σ exp((R0−R)/b)`, IUCr `bvparm2020.cif` (vendored into `<repo>/data/`).
  **§6.3 pitfall C: the loop header line has leading whitespace and must be `strip()`ed before
  parsing**, otherwise the whole table parses as empty and it silently falls back to Brown's
  element-level generic formula, inflating GII from 0.168 to 0.479.
  A sentinel was added to the script: parsing fewer than 1000 entries raises `RuntimeError` rather
  than degrading silently.
  The cutoff is fixed at **3.5 Å**, decoupled from CN (§6.3: using CrystalNN's discrete neighbour
  set systematically truncates long weak bonds).
  Only opposite-sign (cation–anion) pairs are summed. GII is summed over cation sites only (Brown's
  original definition).
- `bvs_dev` uses `n_bvs_bonds > 0` to decide validity, **not** `if bvs` (the §6.5-1 falsy bug: a site
  whose BVS is exactly 0 would be judged nan, which happens to mask the most serious violations).
- Symmetry: `SpacegroupAnalyzer` at two levels, `symprec=0.1` and `0.01`, each giving `orbit_id` /
  `wyckoff` / `mult`.
- `I_G = −Σ pᵢ log₂ pᵢ` with `pᵢ = mᵢ/N` over Wyckoff orbits (Krivovichev's structural-complexity
  information entropy, used in §8.1 as the control for the confounder "complex structures have more
  sites, so they violate mechanically more easily"), one copy per symprec level.

### Engineering

- `ProcessPoolExecutor` with 20 processes (§6.5-4: **not 28** — each process takes 300–500 MB, and
  ChemEnv on high-coordination structures spikes to 1.5 GB).
- Each worker opens its own blob handle to read CIFs and writes its own parquet shard; the main
  process only collects counts (§6.5-2: the pilot's `p.map` returned every dict to the main process
  and then called `pd.DataFrame`, which is orders of magnitude worse).
- Shards are merged by streaming them one at a time through `pq.ParquetWriter`, never reading all
  rows into memory.
- **Idempotent**: shards for which `struct_<k>.parquet` already exists are skipped, so an
  interrupted run resumes on restart; `--force` empties the shard directory and starts over.
- **300 s timeout per structure** (`signal.setitimer`). §6.5-3 originally said 60 s, but ChemEnv
  routinely exceeds 60 s on cells above 200 atoms, and tightening to 60 s would systematically drop
  large cells — the kind of selection bias §6.4 is about. The analysis set's maximum `n_atoms` is
  300 and only 290 entries exceed 200 atoms, so 300 s is enough. Timeouts and exceptions are written
  to `failure.parquet` and the run continues; the batch does not die.
- `--limit N` smoke tests use a **stratified sample** (in the ICSD/COD proportion) and write outputs
  with a `_smoke` suffix, so they do not contaminate the full tables.

### Full run measured (2026-07-28, 20 processes, shared machine)

**98.5 min wall clock / 32.4 core-hours** for all 38,307 entries. Outputs:

| table | rows | size |
|---|---|---|
| `site.parquet` | **458,940** | 20.0 MB |
| `pair.parquet` | **338,135** | 7.8 MB |
| `struct.parquet` | **38,307** | 16.7 MB |
| `failure.parquet` | **62** | 0.2 MB |
| `icsd_ox.parquet` (for audit) | 239,756 | 0.7 MB |

The `_shards/` directory (110 MB) is kept for resuming; it can be deleted once the results check out.

**Against the §6.6 expectations**

| item | §6.6 plan | measured | note |
|---|---|---|---|
| time per structure | 2.0 s, conservatively | **mean 3.05 s / median 1.19 s** | the mean is pulled up by the tail: P95 11.8 s, P99 27.8 s, max 320 s (one entry hit the 300 s cap) |
| wall clock on 20 processes | 2.3 h | **1.64 h** | **29% faster** than planned |
| core-hours | 45 | **32.4** | 28% below plan |
| output size | 1.0 GB | **45 MB** | the plan counted a bonds table; these three tables have nothing at bond level |

Throughput decays from 9.5 struct/s at the start to 6.5 struct/s at the end — shards are cut in
`source_id` order and large cells are denser in the later part of the ICSD numbering. It is not
memory or contention (available memory stayed ≥ 18 GB throughout).

**Failure rates**

- Structure-level failures overall: **62 / 38,307 = 0.162%** (ICSD 34 / COD 28), **all** from the
  same cause: `ValueError: Invalid CIF file with no structures!` — those 62 CIFs in the blob are
  themselves corrupt, and this is not an algorithm failure. All are recorded in `failure.parquet`
  and the batch was not interrupted.
- Algorithm-level (over the 38,245 successful structures): **ChemEnv 0.060% / CrystalNN 0.000% /
  BrunnerNN 0.000%**. §6.3 pitfall A says "about 17% hard IndexError failures when valences are not
  passed"; passing them explicitly measures **0.06%**, confirming the fix works.
- Site-level missing CN: `cn_chemenv` **7.45%** / `cn_crystalnn` 0.02% / `cn_brunner` 0.00%.
  ChemEnv's 7.45% is it failing to identify an environment within `max_dist_factor=1.41` (**not an
  exception**); across the three `ox_source` levels it is 7.61% / 7.75% / 4.11%, so it is **largely
  independent of oxidation-state provenance** and can be treated as missing at random.
- CE symbol assignment rate **92.56%** (424,772 / 458,940 cation sites).

**Oxidation-state coverage (the denominator of the main statistics)**

| | cif | guess | none | total |
|---|---|---|---|---|
| ICSD | 27,374 | 0 | 0 | 27,374 |
| COD | 0 | 9,382 | 1,489 | 10,871 |
| total | **27,374 (71.6%)** | **9,382 (24.5%)** | 1,489 (3.9%) | 38,245 |

`cif+guess` covers **96.11%**, above the §6.2 expectation of "tier0+tier1 around 85% on experimental
oxides".
Unique solutions make up **63.8%** of `guess` (5,985 entries), which lands exactly on the "coverage
about 63%" quoted in §6.2 — but that 63% in §6.2 refers to the **total** coverage of `guess`,
whereas what is measured here is the **unique-solution** share within `guess`. The numbers coincide
but the definitions differ; **do not read this as mutual confirmation.**
There are 1,087 mixed-valence structures (2.84%) and 3,153 with fractional valences — the §6.2
concern that "FIZ integerises mixed valence and causes under-reporting" can be stratified directly
in this store with the `mixed_valence` and `frac_ox` columns.

**Three cross-validations (the §6.6 gate before going to a full run)**

| metric | measured here | reference | verdict |
|---|---|---|---|
| corner/edge/face sharing (**oxides, CN≤8**, n=154,435 pairs) | **73.7 / 25.2 / 1.1 %** | George 2020 (ChemEnv) 73.3 / 25.0 / 1.6 | **almost exact agreement** |
| same, restricted to `ox_source='cif'` (n=112,378) | 73.7 / 25.1 / 1.2 % | as above | insensitive to oxidation-state provenance |
| full analysis set, CN≤8 (n=316,698) | 72.7 / 25.8 / 1.6 % | — | first figures for the all-anion set |
| full analysis set, all CN (n=338,135) | 70.6 / 25.5 / 3.9 % | — | high coordination pushes face sharing up |
| median GII (oxides + `ox_source='cif'`) | **0.229** (RMS) / 0.194 (mean\|dev\|) | the §6.3 pilot reports 0.168 | **does not agree, see below** |

**GII does not match 0.168; half the reason is understood, recorded as it is, and the expectation is not adjusted:**

1. The anion is the main factor. Median by anion: O 0.223 / F 0.191 / Cl 0.230 / I 0.262 / S 0.268 /
   Se 0.315 / Br 0.333 / N 0.384 / Te 0.475 / P 0.481.
   **IUCr bond-valence parameters are markedly worse outside the oxides**, and the 0.246 of the full
   analysis set is pulled up by Te/P/N/Se.
   The 0.168 of §6.3 is an **oxide-scope** number; using it as the expectation for the all-anion set
   is itself a mismatch.
2. Oxidation-state provenance comes second: median GII is 0.203 at the `cif` level and 0.363 at the
   `guess` level (from a 400-entry smoke sample).
3. The remaining gap of 0.229 vs 0.168 is **not explained**. Two hypotheses are ruled out:
   (a) it is not the §6.5-1 falsy bug — `exp((R0−d)/b) > 0` always holds and **not a single site
       measures a BVS of exactly 0** (the 35,702 sites with `n_bvs_bonds==0`, 7.78%, take the NaN
       branch), so that bug cannot fire under this definition;
   (b) it is not the RMS/MAD definition — switching to `mean|dev|` only brings it to 0.194, still
       above 0.168.
   The most likely remaining cause is a **different sample population**: the pilot's `exp_oxide` is
   "all experimental structures containing O" (via sqlite `dataset='experimental'`, including
   multi-anion systems), whereas this round uses the analysis set (single anion, no H, no C); and
   the pilot's oxidation states cascade `cif → bva → guess`, while bva is disabled here.
   **Settle it once P3 has run over the full experimental store.**

**Three-algorithm agreement (the input to G6)**: at site level, over the 424,681 sites where all
three have a value — ChemEnv==CrystalNN **82.1%**, ChemEnv==Brunner 72.9%, CrystalNN==Brunner 79.5%,
**all three equal 70.5%**. §6.3 says "the pilot already showed connectivity statistics differ from
ChemEnv by <1.5 pt at CN≤8", but that is a difference in **aggregate statistics**; the **site-level**
agreement is only 82%. The two are not contradictory but they mean entirely different things, G6
needs the latter, and **the <1.5 pt figure must not be treated as evidence that G6 has passed**.

**Other**: median `I_G` 2.503 (symprec 0.01) and median Wyckoff orbit count 6; **1.73%** of
structures get a different space group at the two symprec levels; and rule 5's `max_distinct_ce`
distributes as 1→23,371 / 2→7,815 / 0→4,264 (no CE) / 3→1,720 / 4→658 / 5→224.

### Known gaps (to be filled next round)

1. **There is no bond-level (cation–anion) table.** Pauling's second rule needs Σs = Σ(z/CN) at
   **anion sites**, and these three tables only reach cation sites and polyhedron connection pairs,
   so **the 18.3% cross-validation item of §6.6 cannot be computed**. The `gii` in `struct.parquet`
   is computed on the fly from 3.5 Å neighbours internally, with no bond detail written out.
2. `pair.parquet` has only the ChemEnv route. The CrystalNN version §9.1 asks for (intersection of
   labelled quotient-graph ligand sets) is not implemented.
3. `splits.parquet` does not exist yet, the three columns `proto_id` / `w_debias` / `split_id` are
   not in the tables, and the `ρ` / `deff` that PREREG §3.1 requires cannot yet be measured.
4. The 62 corrupt CIFs have not been rescued from the raw ICSD CIFs (34 of them, the ICSD ones,
   should in principle be recoverable).

---

## `measure_deff.py` — measuring PREREG §3.1's ρ / deff / N_eff

Outputs: `features/deff.json` (all the numbers) and `features/deff_prereg_block.md` (ready to append
to the PREREG revision log).
Usage: `python src/measure_deff.py --boot 500 --force`; a full run is 71 s single-core, `--limit N`
for a smoke test.
sha256[:16] = `fca0821bf32e0413` (**this repository currently has no .git, so the git tag PREREG and
plan §12 require cannot be applied — a `git init` has to come first; until then the sha256 is the
only version anchor**).

Three choices that have to be argued for, detailed in §A/§B/§C at the head of the script:

- **§A which variable ρ is computed on**: on residual indicator variables, not on continuous
  features. deff enters `L_total` only through `N_eff·H(residual|S)`. One literature-prior residual
  per compression target is used (modal-look-up residual / Pauling 3 violation / BVS out of window),
  none of which involves any product of the search, satisfying "measured before any candidate law is
  seen". Empirically, the ICC of the continuous `bvs_dev` ranges over 0.001–0.717 across the four
  prototype schemes and is unusable.
- **§B missing prototypes**: `structure_type` hits only **54.77%** on the analysis set (not 78.13%,
  which is the whole-icsd_meta figure), and is entirely absent for COD. The main convention, hybrid,
  uses `structure_type` where it hits and the proxy `spg_s01|anonymised formula` where it does not;
  strict / proxy_all / typed_only are reported alongside. **This choice moves N\* more than
  bootstrap sampling error does** (3 → 13).
- **§C how m is taken**: the Kish weighted mean `Σm²/Σm` is the headline (the correct form for
  unequal clusters), with the arithmetic mean reported beside it. The two differ by a factor of 4–6
  (15.7 vs 73.3); the m=25 in the plan's §4.5.1 table is the arithmetic one.

Main results (hybrid, orbit-weighted, N_data = 138,668):
ρ = 0.204 / 0.608 / 0.204 (T_CE / T_CONN / T_BV), deff = 15.7 / 44.3 / 16.7,
N_eff = 7,658 / 2,897 / 7,617.
Negative control ρ = −0.0005, deff = 1.0 (the estimator is unbiased).

**Determination**: the §3.1 fallback clause is triggered, so N is reported as an interval rather
than an argmin (three grounds, in `deff_prereg_block.md`).
Registered intervals under the λ=1 main convention: T_CE / T_BV **N ∈ [5, 11]**,
T_CONN **N ∈ [3, 5]**.

Known limitations:
1. The Zipf mapping is inverse-calibrated — plan §4.5.3 gives only the argmin result, not the
   accuracy decay law, and the five anchors cannot be fitted simultaneously within a single
   (theta, p0) family (log-RMSE 0.337). A second independent mapping (a power law) gives 15 rather
   than 10 at the same N_eff.
   **The interval on N is more sensitive to model specification than to sampling error.**
2. `splits.parquet` still does not exist, so ρ is measured over the whole analysis set rather than
   separately by discovery/calibration/lockbox. The two-way clustering of §8.1 must reuse the
   `proto_hybrid` definition in this file; the two places may not each define their own.
3. T_CONN has only the ChemEnv route (a limitation of the pair table); anion-site Σs is still
   missing, so the anion-side residual of Pauling 2 cannot be measured.

---

## `reproduce_george.py` + `george_table.py` + `pauling_radii.py` — reproducing George 2020 (PREREG gate G-A)

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python src/reproduce_george.py --stage compute --limit 200 --workers 20 --chunk 5 --force   # smoke test, 0.7 min
nohup python src/reproduce_george.py --stage compute --workers 20 --chunk 20 --force \
      > $PRIS_FEATURES/reproduce_george.log 2>&1 &    # full run
python src/reproduce_george.py --stage table            # emit Table S1 (reads the full outputs)
python src/reproduce_george.py --stage table --smoke    # read the _smoke outputs
```

Outputs: `george_site / george_anion / george_pair / george_struct / george_fail .parquet` and
`george_tableS1.csv`; shards live in `_gshards/` (that is what resuming relies on — rerun without
`--force` to continue).

### Why `site/pair/struct.parquet` cannot be reused (three reasons it must be recomputed)

1. **Wrong domain.** George's domain is oxides, and our comparable subset is
   `provenance.oxide_strict` = 23,728, whereas those three tables are built on `in_analysis_set`.
   **Neither contains the other**: `oxide_strict \ in_analysis_set` = 3,895 entries that **all
   contain P** — `in_analysis_set` counts P as an anion candidate, so phosphates are excluded with
   `n_anion_kinds==2`. George uses InPO4 as his main example for the second rule, so **phosphates
   must be inside the domain**.
2. **There is no cation–anion bond-strength table**, so Σs at anion sites cannot be computed and
   Pauling's second rule cannot be done at all.
3. **`pair.parquet` has only the ChemEnv route**, so the three-algorithm connectivity G6 needs is
   unavailable.

### An upstream bug found by measurement: structures whose `cif` oxidation states are all zero are silently emptied

`build_features.assign_oxi` decides the `cif` level with `blob_ox_present = all(x is not None)`, but
a batch of ICSD entries have `_atom_type_oxidation_number` **all zero** (e.g. `exp007`'s ZnO
decorated as `Zn0+ O0-`). Those structures get `ox_source` recorded as `cif` while `is_cat = [v>0]`
is all False, so **`n_cation_sites == 0` and not one site row is emitted**. Worse, ChemEnv given
all-zero valences degrades under `only_cations=True` and **O–O polyhedron pairs** appear in
`pair.parquet` (checked case by case on `exp006` `Mn0+ Au0+ O0+`: 14 of the old table's 23 pairs
were O–O).

- Extent: `n_cation_sites==0` in `struct.parquet` covers **3,658 / 38,307 = 9.55%**, of which 3,637
  have `ox_source='cif'`.
- Fix (`reproduce_george.assign_oxi_fixed`): trust `cif` only when the anion has ox<0 **and** at
  least one site has ox>0; otherwise fall back to `guess` → `none`, leaving a `cif_all_zero` flag.
- **Downstream impact**: the connectivity statistics `build_features` reports (corner/edge/face)
  include these O–O pairs and must be recomputed accordingly; this script's `george_pair.parquet` is
  already the clean version.

### Connectivity enumeration: one enumerator shared by all three algorithms (a necessary condition for G6)

`enumerate_connections()` works directly on ligand sets: the number of shared ligands between
polyhedra (i@0) and (j@T) is `|L_i ∩ (L_j + T)|`, with the candidate translations T given by the
image differences of identically numbered ligands. The de-duplication convention is: for `i<j` take
all T; for `i==j` require T≠0 and keep only the lexicographically larger of T and −T — exactly
matching ChemEnv `environment_subgraph()`'s "one edge per primitive cell".
**Cross-check**: compared structure by structure against `pair.parquet`'s ChemEnv results on the
smoke set, **24 of the 26 comparable structures give identical counts in all three modes**, and the
2 that differ are precisely victims of the all-zero oxidation-state bug above.

Why not use ChemEnv's own `ConnectivityFinder`: that would make G6 compare more than the single
degree of freedom "neighbour definition" — it would swap the connectivity algorithm at the same
time, and the differences between the three algorithms could not be attributed.

### Granularity (PREREG §4.3 requires it stated explicitly; George mixes granularities)

| rule | granularity | does symprec matter |
|---|---|---|
| 1 radius ratio | `orbit` (cation sites de-duplicated by crystallographic orbit), with `site` as an alternative | yes |
| 2 electrostatic valence | `orbit` (**anion** sites), with `site` as an alternative | yes |
| 3 connection type | `pair` (George's original convention); the 60-cell table de-duplicates to `orbit-pair` | only under the orbit-pair convention |
| 4 adjacent polyhedra | `structure` | **no**, the two columns are identical by construction |
| 5 parsimony | `structure` | **no**, as above |
| 2–5 conjunction | `structure` | no |

So **24 of the 60 cells are identical by construction**. That is not copy-paste, it is a genuine
consequence of the definitions.

### The two definitions of rule 1 (8 pt apart; they must be reported as a pair)

`pauling_radii.py` admits only the **univalent radii Pauling himself published** (closed-shell ions,
`tier='published'`); the level extrapolated to open-shell d-block ions is marked `tier='extended'`
and is **our extrapolation, not Pauling's table**, used only as a sensitivity layer.

The criterion also has two definitions. George's text: "A coordination environment is stable only if
the radius ratio falls within the geometrically derived stability window **of this environment**".
The hard-sphere stability window is defined only for CN ∈ {2,3,4,6,8,12}, so sites observed at
CN = 5/7/9/10/11 have **no testable window**; on the wording "tested local environments" they should
be excluded. Both definitions are reported in the comparison table; the 60-cell table uses the
former (`R1_radius_ratio`), and the strict all-CN version is kept separately as
`R1_radius_ratio_allCN`.

### The domain-alignment layer

A non-null `mp_id` in `icsd_mp_link.parquet` means "this ICSD entry is in Materials Project", which
is exactly George's domain filter. This layer is what decides whether a discrepancy is a bug or a
domain difference, and it is the key discriminator for go/no-go.

### Full run measured (2026-07-28, 20 processes, shared machine)

**58.8 min wall clock**, 23,673 of 23,728 entries completed (55 failures = 0.23%, all corrupt CIFs
in the blob).
`site` 376,247 / `anion` 637,679 / `pair` **5,478,931** (three algorithms combined) / `struct` 23,728
rows.
`ox_source`: cif 15,497 / guess 6,685 / none 1,491; 887 entries (3.75%) downgraded by `cif_all_zero`.

**Rule by rule against George 2020** (ChemEnv, symprec 0.01, ox ∈ {cif, guess}):

| rule | George | this reproduction | Δ pt |
|---|---|---|---|
| 1 radius ratio (strict all CN, published radii) | 66.0 | 52.5 | −13.5 |
| 1 radius ratio (**CN with a defined hard-sphere window**) | 66.0 | 61.7 | −4.3 |
| 1 radius ratio (hard-sphere window + extrapolated radii) | 66.0 | **64.8** | −1.2 |
| 2 \|Σs−2\|≤0.01 (orbit / site) | 20.0 | 17.9 / **20.2** | −2.1 / +0.2 |
| 3 corner/edge/face (all CN) | 62.5/27.2/10.3 | 66.5/25.8/7.7 | +4.0/−1.4/−2.6 |
| 3 corner/edge/face (**CN≤8**) | 73.3/25.0/1.6 | **73.4/24.9/1.8** | +0.1/−0.1/+0.2 |
| 4 violation rate (ChemEnv / CrystalNN / Brunner) | 40.0 | 34.7 / **40.3** / 38.4 | −5.3/+0.3/−1.6 |
| 5 parsimony (CN criterion) | 70.3 | 67.8 | −2.5 |
| 2–5 jointly | 13.0 | 11.6 | −1.4 |
| 2–5 jointly (CN≤8) | 20.0 | 15.7 | −4.3 |

The domain-alignment layer (ICSD ∩ MP, 13,149 entries) barely moves these: rule 2 = 17.8,
rule 4 = 35.1, rule 5 = **70.9**, 2–5 = 11.9, rule 3 at CN≤8 = 72.8/25.3/2.0. **So the gap is not
caused by the domain.**

**G6 (three-algorithm spread, max over 12 cells)**: R2 **0.83 pt** ✓ / R5 **1.81 pt** ✓ /
R1_allCN 2.89 ✓; R1 4.05 ✗ / R3 3.93 ✗ / **R4 5.74 ✗**. All three algorithms give the same CN at
only **82.0%** of sites (n=357,082).

**Two decompositions of rule 4** (ChemEnv): the oxidation-state version violates at 56.3%, the CN
version at 52.6%.
The more direct test (the core assertion of George's Fig. 4b) is the connection propensity:
`P(connected | min CN)` = 7.9% (CN1) → 30.4% (6) → **62.3% (12)**, monotonically rising;
`P(connected | min ox)` = 28.5% (+1) → 33.4% (+2) → 10.8% (+5) → **2.8% (+7)**.
**Both show a marginal effect, but the ox branch is heavily confounded by CN** (high-charge cations
are nearly all low-coordinate), and until they can be separated, George's conclusion that
"oxidation state does not affect connectivity" may not simply be copied over. The two-dimensional
conditional version is left to MPU-2.

**The distortion dependence of rule 2** (right panel of George's Fig. 2b): max CSM ≤ 5/1/0.1/0.01 →
22.0/32.7/47.6/**57.8**%.
The direction agrees with Baur's hypothesis, but it is **nowhere near the "nearly perfect" George
reports**; our CSM threshold is our own and George's "221 materials" scope was never published, so
this one is not comparable and only the trend is reported.

**Rule 1 element by element** (n≥1500): P 98.2 / Si 95.6 / Ti 84.0 / B 64.2 / Al 58.8 / Li 52.0 /
Mo 32.8 / Na 30.9 / Sr 23.7 / Ca 21.8 / Ge 16.0 / V 13.5 / K 10.9 / Ba 10.3 / Cs 7.3.
Qualitatively the same pattern as George's Fig. 1b (tetrahedron formers high, alkalis and alkaline
earths low).
Pauling's published radii cover 65.7% of cation sites, 87.0% with the extrapolation; the top
uncovered are W/Fe/Mn/U/Cu/Bi/Mo/Pb/V/Co/Nd.

---

## `build_bonds.py` — closing the MPU-2 gaps (bond / anion_sum / three-algorithm pair / proto_id / split)

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=python
$PY src/build_bonds.py --stage compute --limit 60 --workers 12 --chunk 5 --force  # smoke test, 0.7 min
nohup $PY src/build_bonds.py --stage compute --workers 20 --chunk 20 &            # full run, 121.6 min
$PY src/build_bonds.py --stage assemble   # 2 min
$PY src/build_bonds.py --stage report     # ε curves + rule 3 three-algorithm table (discovery only)
```

Domain = `in_analysis_set ∪ oxide_strict` = **42,202** (38,307 + 3,895 phosphates).
The latter are not in `splits.parquet` and get `split = "unsplit"`; **downstream they may not be
used for anything beyond discovery**.

### Outputs

| file | rows | size | granularity |
|---|---|---|---|
| `bond.parquet` | **8,948,970** | 52.5 MB | one cation–anion bond per row (chemenv 2,785,858 / crystalnn 3,028,229 / brunner 3,134,883) |
| `anion_sum.parquet` | **2,690,532** | 29.7 MB | 896,844 anion sites × 3 algorithms |
| `pair.parquet` | **12,012,499** | 43.7 MB | polyhedron connection pairs × 4 routes (the old version is backed up as `pair_v1_chemenv_only.parquet`) |
| `bond_struct.parquet` | 42,202 | 0.8 MB | structure-level audit (ok 42,134 / fail 68) |

`bond` columns: `source_id / nn_algo / cation_site / anion_site / el_* / ox_* / img_a,b,c / d /
cn_cation (after anion filtering) / cnall_cation (unfiltered) / s_pauling / s_bv / ce_cation /
csm_cation / proto_id / split`. `s_pauling = z/CN` is a **T1 quantity**; `s_bv` is a **T2 quantity
and by iron rule 2 may not enter a T4 Guard or Body**.

`anion_sum` columns: `sigma_pauling / sigma_bv / z_abs / dev_* / n_cat / d_min / d_max /
ok_{pauling,bv}_e{01,05,1,2,3,5}` (nullable boolean).
**Anions with zero coordination (1.99%) record Σs = 0 rather than NaN** — `george_anion` judged them
NaN and dropped them, and those are exactly the most serious violations, which must not be dropped.

### Pitfall D: `ConnectivityFinder` silently swallows 78.6% of structures (overturning the old `pair.parquet`)

`pymatgen.analysis.chemenv.connectivity.ConnectivityFinder.get_structure_connectivity` has two
branches that **raise on the whole structure**:

1. With `multiple_environments_choice=None` (the default), any site MultiWeights judges a "mix"
   (`len(neighbors_sets[i]) > 1`) triggers
   `raise ValueError("... is a mix and nothing is asked about it")`.
   Measured: **29,164 / 42,202 = 69.1%** of structures contain at least one mix site.
2. The loop only guards against `neighbors_sets is None`, not against an **empty list** →
   `IndexError: list index out of range`. Measured: **8,423** structures hit this.

`build_features.py` catches both exceptions into the `ce_err` column with no row-count alarm, so the
old `pair.parquet`'s 338,135 rows **come from only 8,212 / 38,307 = 21.4% of structures**.
The ones that get through are the high-symmetry structures with clean environment assignments, so
the corner-sharing share is systematically inflated — a textbook silent selection bias.
**No rule 3 or rule 4 number from the old `pair.parquet` is usable.**
(The 73.4/24.9/1.8 reported for MPU-1 comes from `george_pair`'s unified enumerator and is not
affected by this pitfall; it is usable.)

`robust_env_subgraph()` is the patched rewrite: for a mix, take the largest `ce_fraction`
(pymatgen's own `TAKE_HIGHEST_FRACTION`, which is off by default); for an empty list, skip that site
rather than abandoning the structure. After the fix it reaches **40,625 / 42,202**.
The three columns `n_mix_env / n_empty_env / n_ceconn_noncat` record this per structure, so it can be
rechecked if the convention changes.

### The four pair routes, and separating "enumerator" from "neighbour definition"

`chemenv` uses ChemEnv's own `environment_subgraph()` (`len(d["ligands"])`);
`crystalnn` / `brunner` use the intersection of shared ligand sets; `chemenv_uni` is the **audit
line** — the same ChemEnv neighbours, but through the unified enumerator. On structures where
ChemEnv's ligand sets are purely anionic, the edges of `chemenv` and `chemenv_uni` are **identical
one by one** (51/56 structures in the smoke set, with 1922/644/233 identical across the three
categories).
So the unified enumerator is an exact rewrite of `environment_subgraph()`, and **G6's 3.93 / 5.74 pt
come entirely from the neighbour definition, with zero contribution from the enumerator** — a clean
attribution. The row-count difference between `chemenv` and `chemenv_uni` comes only from the anion
filter (ChemEnv natively counts cation–cation shared ligands too).

### proto_id (the hybrid convention)

Where the ICSD `structure_type` hits, record `ST:<type>` (covering **53.60%**, and 0% for COD);
where it is missing, record `PX:spg<spg_s01>|<anonymized_formula>`. Over the whole domain there are
**9,571 unique proto_ids**.
It has been merged into all six tables (`site / pair / struct / bond / anion_sum / bond_struct`),
along with `split`.

### Consistency check against `george_anion`

On the same subset (discovery+unsplit, oxide_strict∩O, chemenv) the Σs of the two tables are
identical at **99.81%** of sites; the median |Δ| at the differing sites is exactly **1.000** (one
whole bond), which is tie-break jitter at ChemEnv's neighbour boundary, not a systematic bias.
Satisfaction at ε=0.01: `anion_sum` 18.60% vs `george_anion` 18.57%.

---

# MPU-3 rule search (`search_rules.py` / `report_rules.py`)

**Reads discovery only.** Every retrieval filters on `split == "discovery"` and asserts it;
`calibration` and `lockbox` were never opened this round. Outputs, on the `t4_instances`
convention: `_t4_orbits_raw.parquet`, `_tierA1/_tierA2.parquet`, `rule_candidates.parquet`,
`rules_surviving.parquet`, `rules_top.csv`, `search_log.json` (containing `negctl_i` and
`negctl_iii_rule3`).

## Domain and labels

The basic unit of the primary target T4 is the **crystallographic orbit** (`orbit_id_s01`), and the
predicted quantity is constant within a `(source_id, element, ox)` **species** — that is the finest
granularity T0 inputs can express (getting down to the site would require knowing how many orbits
that species occupies, which is T1).
Discovery cation sites 259,026 → those with all three algorithms' CN present
(ChemEnv / CrystalNN / BrunnerNN) 239,548 (92.48%, as G6's common-base requirement demands) →
**72,087 orbits / 19,430 structures / 40,097 species instances**.
All three labels `cn_chemenv / cn_crystalnn / cn_brunner` are retained, so G6 can assess the
accuracy spread across the three directly.

## The modal look-up baseline: the 80% figure is an artefact of averaging over species unweighted

| convention | CN (CrystalNN) | ce_symbol |
|---|---|---|
| instance-weighted (all anions) | **0.564** | 0.519 |
| instance-weighted (oxides) | 0.603 | 0.553 |
| species-unweighted, n≥30 (147 species) | 0.648 | 0.603 |
| **species-unweighted, all (954 species)** | **0.765** | 0.746 |

The "≈80%" cited in PREREG §2 holds only on the **last row**, and that row is inflated by hundreds of
rare species that occur once or twice and have a trivially perfect purity of 1.0.
**The hard lower bound usable for prediction is 0.563 (ChemEnv) / 0.580 (CrystalNN)**, and every
matched-coverage comparison this round is made against it. Pauling's first rule (Pauling's univalent
radii plus five frozen break points) scores **0.324** on the same domain (89.6% coverage), far below
the 66% George reports — his is oxides at site granularity.

## Two layers of search

* **Tier A1** `IF f relop θ THEN cn = c`: for each (f, relop, θ), take the exhaustively optimal c.
  Prefix sums plus `searchsorted`, 61 features × 2 relops × 20 quantile thresholds, **0.8 s**,
  1,713 rules.
* **Tier A2** (template family 2 of plan §7.4)
  `cn == ARGMIN n IN CN_SET : |φ(n) − (f−a)/b|`.
  φ ∈ {n, 1/n, √n, log n, Pauling's ideal radius ratio, ox/n}, CN_SET ∈ {FULL(2..12), PAULING,
  COMMON}.
  **Key implementation**: φ is monotone, so ARGMIN is equivalent to a monotone staircase in f whose
  cut points are set by the two free parameters (a,b); one sort plus a prefix count per class
  therefore enumerates the whole 24×40 quantised (a,b) grid exactly, without evaluating 11 `|·|`
  per instance. The `s_pauling` branch (where φ depends on ox) is handled the same way after
  stratifying by ox. **19.9 s**, 1,037 rules.
* **Tier B** `pysubgroup.Apriori` (exhaustive with anti-monotone pruning), a selector pool of **135**
  (hard cap 150, including 20 random features as negative control ii; degenerate selectors with
  coverage >97% or <0.5% have been removed), depth=3, `StandardQF(a)` swept over
  a ∈ {0.25, 0.5, 0.75}. **Measured 13.1 s per depth-3 pass / 1.3 s at depth 2**
  (72,087 rows × 135 selectors), and 51 Bodies × 3 sweeps comes to **1,194 s**.
  An order of magnitude faster than the 240 s cited in plan §7.4, because there are 3× fewer rows
  and the numba fast path engages.

## G3 bit accounting (the vocabulary convention of plan §4.3)

`L = 9.3 (header fields) + log₂6 (Body production)
+ [log₂13 | log₂6 + log₂3 + log₂47 + log₂24 + log₂40] + |guard|·log₂250`.
A constant Body plus a 3-literal guard is **39.5 bit**; an ARGMIN Body plus a 3-literal guard is
**55.4 bit**, and 4 literals reaches 63.4 bit and busts the budget — so on the ARGMIN family the
60-bit budget pins the guard at **exactly** 3 literals, independently arriving at the same bound as
the `<guard> ≤ 3 literals` in the §7.1 grammar.

## Gates and the funnel (2,044 → 164)

See `report_rules.py`. **G4 (leave-one-anion-out worst-fold τ ≥ 0.90) is the only gate actually
killing candidates** (15.9% pass it alone); after G4, G6 is not binding at all (relaxing its
threshold from 3 pt to 100 pt takes survivors from 164 to 169) — the opposite of the MPU-1
intuition, and the reason is that rules able to pass G4 already sit in the "clean" region where the
three algorithms agree.
`uses_rnd`: 697 of the 2,044 contain a random feature; **36 of the 200 that clear G1–G5 (18%) still
contain a random literal**, so the explicit filter of negative control ii does real work and G1–G5
alone do not stop it.

## Negative controls

* **(i) Block permutation of labels** (permuted within `proto_id` blocks, preserving confounding
  structure): Tier A2's best precision falls 0.4475 → **0.3481**, against 0.5628 for the modal
  look-up — after permutation the best candidate cannot even come close to the look-up table, so the
  null distribution discriminates.
  The per-rule `perm_z` is computed in one shot from a B=200 block-permuted label matrix (2.4 s to
  generate, O(200·n_trig) per rule).
* **(ii) 20 random features**: see above; the final surviving set contains no random literal.
* **(iii) Rediscovering Pauling's third rule (gate G-D, PREREG §6)**: **passed.**
  Over 16 combinations (4 pair routes × 4 stratifications, D≤2, drawn from the Guard vocabulary),
  the search puts `corner > edge > face` first among the 6 possible frequency orderings **every
  single time**.
  Stratified by D=1 (cation element): ρ = 0.926 / 0.924 / 0.921 / 0.886
  (chemenv / chemenv_uni / crystalnn / brunner), with Wilson 95% lower bounds 0.848 / 0.844 /
  0.838 / 0.797, far above the 0.5 §4.4 requires.
  D=1 (oxidation state): ρ = 0.655–0.694, lower bounds 0.596–0.636, still passing.
  **D=2 (oxidation state × CN band) fails throughout** (ρ = 0.461–0.493, lower bounds 0.417–0.450) —
  which is exactly the over-stratification failure mode §4.4 predicts (487–645 strata, with the
  median stratum's sample size collapsing), not chemistry.
  The pipeline's ability to discover is confirmed by this, so it may continue.

## The substantive conclusion of this round (a negative result that must go into the paper as it is)

**Not one T0 candidate beats the modal look-up table on MDL.** For all 164 surviving rules,
`ΔL = N_eff·[H(look-up) − H(rule)] − L(R)` (`N_eff = n_trig / deff`, with `deff = 15.7` taken from
the T_CE compression target of PREREG revision R1) is **negative**, the best at −25 bit. The gain in
matched coverage over the look-up table is only **+0.17 to +0.53 pt**.
The reason is diagnosable: the high-coverage guards Apriori finds under `StandardQF` happen to
enclose **species-homogeneous** regions (high ionic potential → Si/P/S/B → already 95% CN=4), where
the look-up table is already saturated; whereas the regions where the look-up table is weakest
(CN=2 for Cu⁺/Ag⁺, CN=8/12 for large cations, where the table gets only 17–19%) can indeed be lifted
**+15 to +19 pt** by composition-level rules, but **not one such candidate clears all the gates**
(of the 80 with a gain >2 pt, none passes all eight — only 10 pass G4 and only 21 pass G6; of the 18
with a gain >10 pt, 1 passes G4 and 0 pass G6).
G4's failure mode is very specific: they are oxide-only and do not hold for sulfides or halides
(leave-one-anion-out worst-fold τ < 0.90).
Together these two things make one publishable judgement: **T0 composition-level information adds
almost nothing on top of "the species identity is known"**, and what it does add concentrates in a
few (species × anion family) cells, which is precisely what G8 and G4 exist to block.

---

# MPU-3 / Tier D: set assembly and the `L_total(N)` curve

Script `src/assemble_set.py` (`python assemble_set.py` for the main path, 68 s;
`python assemble_set.py ungated` for the gate-free diagnostic). Outputs: `assemble_result.json`
(every curve and baseline), `Ltotal_curve.csv` (flat table, 4 targets × 4 λ × N=0..12),
`rule_decomposition.csv` (the ΔL decomposition of the 19 de-duplicated rules), and
`assemble_ungated.json`.
**Discovery-only throughout**: retrieval goes through `_t4_orbits_raw.parquet` (generated by
`search_rules.prep()`, with asserts), and `split == "discovery"` is asserted once more at each of
`pair.parquet` and `site.parquet`.

## The candidate pool and its reconstruction

The 164 surviving rules are reconstructed **literally** into per-instance predictions from `body_id`
plus the guard string and compared rule by rule against the archived `(cov, acc)`: **164/164 agree
exactly** (asserted in the script).
De-duplicating by "effective prediction vector" (a prediction where triggered, −1 where not) takes
**164 → 19** — the other 145 are the same guard with an equivalent Body. The selection cost
`log2 C(M,N)` is still computed with **M = 164** (the search-space convention).

## The objective and the four curves

    L_total(S) = λ·[log2 C(M,N) + Σ L(l_i)] + PAR(S) + N_eff·H(residual | S)

All three `PAR` conventions are reported: `reg` = the global `(2^N−1)/2·log2 N_eff` written literally
in PREREG §4.5.1; `full` = the per-pattern `(2^|G|−1)/2·log2 n_eff_G` conditioned on the guard;
`obs` = charging only for realised cells (the most permissive, and the default). `N*` differs by at
most 1 across the three.

| target | deff | what is compressed | residual / cell |
|---|---|---|---|
| `T_CE` | 15.7 | the sequence of orbit CN labels | binary error indicator, cell = guard pattern; uncovered instances **fall back to the modal look-up** (the table cost of 3,382 bit is a constant paid at every point, so the N=0 point is `SET-MODE`) |
| `T_CE_MC` | 15.7 | as above | **13-class conditional entropy** (the corrected convention, see below); uncovered instances fall back to a single global modal cell |
| `T_CONN` | 44.3 | whether a site participates in edge or face sharing | cell = (trigger pattern, each rule's predicted CN), both T0-decodable; the residual may not be used as a condition |
| `T_BV` | 16.7 | \|bvs_dev\| > 0.2 vu | as above |

The domain is 72,087 orbits / 19,430 structures / 914 (element, oxidation state) species; look-up
top-1 = 0.5628.
`L(SET-MODE) = 914·log2 13 = 3,382 bit` (the estimate in PREREG §4.5.5 is 240 species × log2 68 =
1,461; both the species count and the alphabet have to be recomputed for the actual domain).

## The registered residual code is unusable (a methodological finding requiring an added PREREG revision)

The residual code registered in PREREG is `N_eff·H_b(error rate)`. **`H_b` is non-monotone as the
accuracy crosses 50%** (`H_b(0.60) = 0.971 < H_b(0.44) = 0.988`), which produces two absurdities on
`T_CE` that have to be acknowledged:

* Pauling's first rule (top-1 only 32.4%) has `L_total` = **7,613 bit**, *below* the modal look-up
  table's **7,928** at 56.3%;
* the question "can a law set replace the look-up table" cannot be assessed under it at all —
  swapping in a worse fallback makes things cheaper.

So `T_CE_MC` recomputes with the 13-class conditional entropy `Σ_cell n_eff_c·Ĥ(y|cell)` (monotone
in information content), reported as a co-equal main convention. **Both curves are reported; neither
is cherry-picked.**

## Curve values (λ = 1, in bits)

| N | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|
| `T_CE` | **7928** | **7760** | 7796 | 7834 | 7879 | 7931 | 7984 | 8093 | 8198 | 8309 |
| `T_CE_MC` | 12831 | 12298 | **12271** | 12303 | 12348 | 12397 | 12446 | 12549 | 12652 | 12761 |
| `T_CONN` | **1276** | 1279 | 1324 | 1369 | 1423 | 1475 | 1528 | 1632 | 1736 | 1859 |
| `T_BV` | **4198** | 4226 | 4270 | 4314 | 4359 | 4411 | 4463 | 4568 | 4672 | 4783 |

Where the four baselines sit on the curves:

| | `T_CE` | `T_CE_MC` | `T_CONN` | `T_BV` |
|---|---|---|---|---|
| `SET-MODE` | 7928 (= N=0) | 13457 | 4979 | 7711 |
| null model (global mode / single cell) | — | 12831 | 1276 | 4198 |
| `SET-P5` | 7696 | **11108** | 1356 (including the P4 partition) | 4303 |
| `SET-P25` | 8011 | 12914 | 1335 (including the P4 partition) | 4282 |
| `SET-H3` | 7994 | 12897 | — | 4264 |
| Pauling's first rule alone | 7613 | **11024** | — | — |
| `SET-OURS(N*)` | 7760 | 12271 | 1276 | 4198 |

On `T_CE`, `SET-P25` and `SET-H3` have **no independent T0 member**: they pay the model cost and
compress nothing — the quantitative version of the plan's §1 remark that "the T0 curve currently has
only two known points".
On `T_CONN`, Pauling's third rule is an **unguarded universal assertion that partitions no domain,
so its MDL compression is identically 0**; only Pauling's fourth rule's high-charge guard genuinely
partitions, and the data bits it saves (about 25) do not repay its own 83 bit.

## Determination

* **Knee: none.** All four curves take their minimum at N=0 or 1 and rise monotonically thereafter;
  there is no "flat then rising" knee.
  `argmin N*` across the obs / full / reg conventions: `T_CE` 1/1/1, `T_CE_MC` 2/1/2,
  `T_CONN` 0/0/1, `T_BV` 0/0/0.
  **G-C does not pass**, so per §12.0.5 we pivot to "the regularity of crystal chemistry is not
  low-rank".
* **Fixed criterion (λ=1, cumulative compression computed on the data term)**: `T_CE` N80=1 / N90=2;
  `T_CE_MC` 1 / 2; `T_CONN` 1 / 2; `T_BV` 2 / 4. The two levels do not differ by 2×, and the N
  interval lands at **[1, 2]** (T_BV at [2,4]).
* **Against the intervals registered in PREREG revision R1**: registered `T_CE [5,11]` /
  `T_BV [5,11]` / `T_CONN [3,5]`, and the measurements **all fall outside, below the lower bound**.
  Reported as they are; the registered values are not adjusted.
* **λ sensitivity**: over λ ∈ {1,3,10,30}, `N*` goes `T_CE` 1→1→0→0, `T_CE_MC` 2→1→1→0, and
  `T_CONN`/`T_BV` stay at 0. **A movement of ≤ 2, which satisfies the L0 criterion of §7 (≤ ±3)** —
  but it is a degenerate satisfaction: `N*` is pinned at 0–2 and λ has no leverage to exert. The
  paper has to say it this way and cannot simply write "passed".

## Matched coverage vs the modal look-up table (the second criterion of PREREG §2)

| N | coverage | set acc | look-up acc (same coverage) | Δ | McNemar p | clustered bootstrap 95% CI (structure / prototype) |
|---|---|---|---|---|---|---|
| 1 | 7.32% | 0.9568 | 0.9522 | +0.46 pt | 0.0022 | [+0.04,+0.86] / **[−0.05,+0.95]** |
| 2 | 8.14% | 0.9598 | 0.9525 | +0.73 pt | 1.3e−6 | [+0.34,+1.13] / [+0.23,+1.22] |
| 3 | 11.3% | 0.8825 | 0.8772 | +0.53 pt | 1.8e−6 | [+0.25,+0.81] / [+0.17,+0.88] |

The contradiction rate (instances where ≥2 rules fire and their top-1 disagrees) stays ≤ 0.4%
throughout.
**At N=1 the CI clustered by structure prototype contains 0; N=2 and N=3 are significant, but the
magnitude is only +0.5 to +0.7 pt, and the rules were selected on the same discovery partition, so
these are in-sample numbers.**

## The ΔL decomposition: 97% of the gain comes from stratification, not from prediction

`rule_decomposition.csv` splits each rule's data-term gain into
`ΔL_predict = n_eff·[H_b(look-up error rate) − H_b(rule error rate)]` (the registered single-rule
convention) and the remainder, `ΔL_stratify`.
For the best rule: `ΔL_data = 210.0 bit`, of which **`ΔL_predict` is only 6.7 bit and `ΔL_stratify`
is 203.3 bit**. All 19 have this shape (predict 1.2–7.9, stratify 171–211).
In other words, the bits these guards earn come almost entirely from "enclosing a region where the
look-up table happens to be especially reliable, so the table's error sequence can be coded in two
segments", not from "the rule predicts better than the table".
**A 31.5-bit rule cannot earn back its own 31.5 bit (its prediction term is 6.7 bit).**

## The pipeline's gates would kill Pauling's first rule

Every Pauling-form radius-ratio rule in `rule_candidates.parquet` (coverage 89.6–99.7%) **fails the
gates**: `G4` leave-one-anion-out worst-fold τ = 0.40–0.62 (threshold 0.90), `G5` gain over the
look-up table is negative (−13.5 to −18.3 pt), and half fail `G6` with a three-algorithm spread of
2.7–4.4 pt (threshold 3.0).
Yet under the corrected `T_CE_MC` convention, **Pauling's first rule alone (11,024 bit) compresses
better than any set we searched out (best 12,271) and better than the modal look-up table
(13,457)** — because multi-class conditional entropy rewards information content rather than top-1
hits, and Pauling 1 cuts 89.6% of the domain into 6 CN prediction cells, lowering the conditional
entropy by 2.0 bit per effective instance even at a top-1 of only 32.4%.
**Top-1 gates and MDL are systematically inconsistent here, and that is an independent
methodological conclusion.**

## Gate-free diagnostic: are the gates too strict, or is the information not there? (`assemble_set.py ungated`, 770 s)

Turning G1–G8 all off and reassembling from **all 2,044 candidates** (859 distinct prediction
vectors after de-duplication), on `T_CE_MC` at λ=1:

| N | 0 | 1 | 2 | 3 | 4 | 5 | 6 | **7** | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `L_total` | 12831 | 10384 | 10076 | 9969 | 9930 | 9924 | 9923 | **9861** | 9871 | 9887 | 9925 |
| data term | 12754 | 10080 | 9353 | 9055 | 8923 | 8662 | 8545 | 8074 | 8048 | 7934 | 7862 |

**Without gates the curve does have an interior minimum at `N* = 7`, inside the `T_CE [5,11]`
registered in PREREG revision R1**; the fixed criterion lands inside as well (N80 = 5, N90 = 7). The
compression is 2,970 bit, 5.3× the 560 bit obtained with gates.

**But this road is closed, for three hard reasons**:

1. Of the 12 members of that `N* = 7` set, **9 fail G4** (leave-one-anion-out worst-fold τ < 0.90),
   **6 fail G6** (three-algorithm spread ≥ 3 pt), and **1 contains the random feature `rnd_Zmod7`**
   (which trips negative control ii outright).
2. Its top-1 accuracy is **0.389, 17.4 pt below the modal look-up table's 0.563**, directly violating
   the matched-coverage criterion of PREREG §2 — and §2 together with R16 require MDL and matched
   coverage to hold **simultaneously**.
3. Its minimum is a **shallow basin**: between N=4 and N=12 `L_total` moves only within 9,861–9,930
   (0.7%). The knee is not sharp, and under any reasonable tolerance only an interval can be
   reported.

The conclusion, stated firmly: **"there is indeed a low-rank structure of N ≈ 5–7 in T0
composition-level information, but it is about the shape of the CN distribution (low conditional
entropy), not about the CN mode (poor top-1), and it does not hold across anion families."**
That single sentence explains why G-C and G-E fail together, and it is the evidence the §12.0
contingency for "no T0 set is obtainable" needs to cite.

---

# Phase two: plausibility laws and the score formula (2026-07-29)

Once the target changed from "predict the coordination environment" to **"judge whether a structure
is plausible"**, the following scripts were added.

## The feature layer

| script | outputs | contents |
|---|---|---|
| `phys_law.py` | `phys_real` / `phys_bad` | Shannon coordination-dependent radius quantities: `bl_min` (shortest cation–anion bond / radius sum), `bl_mean`, `bl_cat_max`, `bl_rsd_max`, `sh_pack`, plus the charge-topology quantities `frac_like_bonds` / `min_opp_frac` |
| `elec_feat.py` | `elec_real` / `elec_bad` | Ewald decomposition, site Madelung energies (including the scale-free `madz_*`), Hoppe's effective coordination number ECoN, bond-valence mismatch (**using formal charges; BVAnalyzer is disabled**) |
| `geom_feat.py` | `geom_real` / `geom_bad` | three geometric coordination quantities selected by multi-agent survey: `aa_min` (ligand–ligand contact ratio, **the geometric kernel of Pauling's first rule**), `phi` (convex-hull packing fraction of the polyhedron, which specifically catches cation transposition), `mef` (Hoppe MEFIR effective-radius mismatch, signed) |
| `sym_feat.py` | `sym_real` / `sym_bad` / `sym_lemat` | symmetry: space-group number, number of symmetry-inequivalent sites, Wyckoff entropy `I_G`. **The conclusion is not to adopt it**, see below |
| `t0_guard.py` | `t0_guard` | purely compositional measures of ionicity (electronegativity difference, Pauling ionic-character fraction `fi`). **Used only as a premise**; as a law its exclusion power is identically 0 |
| `robust_blmin.py` | `robust_blmin` | robustness self-check for the core law: spread across three neighbour algorithms, Shannon table hit rate, threshold sensitivity |
| `false_positive.py` | `false_positive` | **the false-positive test**: apply the laws to LeMat's DFT-relaxed candidates. Those structures are geometrically reasonable and simply have not been synthesised, so the laws should largely pass them; if a large batch is rejected, what is being measured is a database fingerprint rather than plausibility |

**Two scripts seed their perturbations from the same `seed_of()` (crc32)**, so the same `sid` in
`phys_bad` and `elec_bad` points at **the same** perturbed structure and they can safely be merged
on sid.
This previously used `abs(hash(sid))`, but Python's string hash is randomised by PYTHONHASHSEED and
**differs across processes** — the perturbation could not be rebuilt, and features computed in two
passes would merge onto the wrong rows. Replay with the same seed has been measured as 28/28
identical.

## The application layer

| script | purpose |
|---|---|
| `apply_rules.py` | **the delivery entry point**: judge the plausibility of any structure file. `--set single/core4/five`, and `--verbose` for the verdict and measured value law by law |

```
python src/apply_rules.py foo.cif bar.cif           # the recommended five
python src/apply_rules.py --set core4 *.cif         # the trusted core of four only
python src/apply_rules.py --verbose foo.cif         # see each law's measured value
```

Measured on CsTaO₃ and its cation–anion swapped version, this confirms the S5 blind spot exactly:

| law | real structure | after the swap |
|---|---|---|
| `bl_min` | 0.8819 ✓ | 0.9647 ✓ **the value actually improves** |
| `bl_mean` | 0.9400 ✓ | 1.0290 ✓ |
| `madz_range` | 19.62 ✓ | 22.13 ✓ |
| `mad_max` | −5.89 ✓ | 0.41 ✓ |
| **fraction of like-charge bonds** | **0.0000 ✓** | **0.2500 ✗** |

**All four bond-length and electrostatic laws pass it; only the fifth catches it.** Drop the fifth
and this structure is judged plausible.

## The search layer

| script | purpose |
|---|---|
| `rules_final.py` | beam search over law sets. `assert_clean()` hard-asserts that no lockbox data is present; `--guards` enables "if G then T"; `--strat-floor` imposes a satisfaction floor per chemical stratum; `--min-cov` controls the threshold on the computable fraction of a feature; `--ban` disables named features |
| `formula2.py` | the score formula. Within-composition pairing, antisymmetric double-writing to pin the intercept, GroupKFold by group, clustered bootstrap CIs, abstention curves. `--protocol holdout\|cv` |
| `verify_negatives.py` | **the negative control for the negatives** — use Madelung energies to confirm that each perturbation class really ought to be excluded |

## Three checks that must be kept

1. **Negative samples need a negative control** (`verify_negatives.py`). S2 once had ΔE identically 0
   because the charges did not follow the swapped elements, and four rounds of feature engineering
   were chasing a signal that did not exist; S6, shear strain, was dropped because neither probe
   could tell it was implausible.
2. **The split assertions have to be in the code** (`assert_clean`). "I know there is a lockbox" is
   not the same as the code knowing.
3. **The negative-sample lineage has to cover different axes of failure.** The leave-one-perturbation-
   class-out test shows the laws **only constrain the damage directions they have seen**: never
   having seen uniform expansion, they will not select an upper bound on bond length. Hence the
   addition of S5, the cation–anion swap — geometrically all but indistinguishable from the real
   structure (`bl_min` 0.930 vs 0.937) but electrostatically +5.8 eV/atom away.
   Measured, every bond-length and coordination law has an exclusion power of only 0.03–0.21 on S5,
   while a single `frac_like_bonds ≤ 0` (no like-charge bonds exist) catches 96%.
4. **The decision criterion must be fixed in writing before the numbers are seen.**
   `sym_feat.py` is a case in point: the rule "a structure must have symmetry" has a satisfaction
   rate of **99.17%** and an exclusion power of **39.5%**, both better-looking than the core law
   `bl_min`. But its exclusion power against uniform expansion is **0.00** and against random
   displacement is **0.98** — exactly matching "this class of perturbation moves the fractional
   coordinates", which is a property of the negative-sample generator; the false-positive drop is
   +8.3 pt; and P1 accounts for 0.83% of real structures against 9.17% of LeMat, an elevenfold gap
   that is the database fingerprint itself.
   The decision criterion is written at the head of `sym_feat.py`, before anything was computed.
   **Explained after the fact, those two numbers would persuade anyone to put it in the main text.**

5. **`--ban` has to filter on `col2`.** For a rule with a premise, `col` is
   `"G:premise column:target column"`, so matching `col` alone makes the ban a no-op — several
   "identical before and after ablation" conclusions were invalidated by this.

6. **Ceiling measurements must come with an out-of-distribution test.** An unconstrained GBDT
   excludes 87% at a satisfaction rate of 99%, but the leave-one-perturbation-class-out test shows
   it collapsing to 0.45–0.87 — it is recognising the perturbation signature, not implausibility.
   Under the same protocol the law set collapses only to 0.243.
