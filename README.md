<p align="center">
  <img src="assets/pris-logo.png" alt="PRIS" width="760">
</p>

# PRIS — Plausibility Rules for Inorganic Structures

**Autonomous discovery of new structure plausibility laws for explainable and rapid
crystal diagnosis and screening**

An autonomous agent ran 572 numbered investigations over 99,162 experimental crystal
structures under a pre-registered protocol, evaluated 2,037,606 candidate laws, and kept
eight one-line laws. Those eight are **PRIS**, and a synthesis score derived from them is
**PSS**. This repository holds the manuscript, the analysis code, the aggregate data every
figure is drawn from, and the pre-registrations that fixed the rules of each exercise
before it was run.

The eight laws encode five complementary physicochemical mechanisms: short-range
repulsion, ionic contact and packing, electrostatic balance, bond-valence conservation,
and crystallographic site complexity. A structure that fails is told which mechanism it
violated, which is what a bare distance cutoff cannot say.

Every rule is judged on two numbers at once: its **satisfaction rate**, the fraction of
real crystals that satisfy it, and its **detection rate**, the fraction of deliberately
corrupted structures it identifies as implausible. Pauling's rules of 1929 had never
been scored on the second axis. Measured on both, the comparison reverses: Pauling's
rules 2–5 are jointly satisfied by 6.5% of real crystals and fail to distinguish most
pairs of competing structures, while PRIS is satisfied by 82–99% of held-out real
crystals and detects up to 91% of damaged ones, where the distance cutoffs deployed in
generative pipelines detect 1.6–3.2%.

## The laws

Eight one-line laws over quantities computable from a structure file and a radius table.
They are applied as nested sets. `Set 1` to `Set 4` are the chain the paper reports;
`Set 1'` is the guarded band that stands beside the chain rather than inside it, which is
why the catalogue holds eight laws while `Set 4` applies seven:

| | law | Set 1 | Set 1′ | Set 2 | Set 3 | Set 4 |
|---|---|---|---|---|---|---|
| **Law 1** | ρ ≥ τ | τ = 0.735 | τ = 0.735 | τ = 0.804 | τ = 0.804 | τ = 0.804 |
| **Law 2** | f<sub>i</sub> > 0.50 ⇒ ρ ≤ 1.05 | – | ✓ | – | – | – |
| **Law 3** | mean anion CN ≤ 3.333 ⇒ mean d/(r₊+r₋) ≤ 1.081 | – | – | ✓ | ✓ | ✓ |
| **Law 4** | range<sub>i</sub> of V<sub>M</sub>(i)/v<sub>i</sub> ≤ 31.45 eV | – | – | ✓ | ✓ | ✓ |
| **Law 5** | max<sub>i</sub> V<sub>M</sub>(i) ≤ 15.17 eV | – | – | ✓ | ✓ | ✓ |
| **Law 6** | f<sub>i</sub> > 0.55 ⇒ no like-charge bonds | – | – | – | ✓ | ✓ |
| **Law 7** | inequivalent sites / sites ≤ 2/3 | – | – | – | – | ✓ |
| **Law 8** | mean of \|BV sum − v<sub>i</sub>\| / v<sub>i</sub> ≤ 0.7143 | – | – | – | – | ✓ |

where **ρ** is the reduced contact ratio, the shortest cation–anion distance divided by
the sum of the two Shannon radii; **f<sub>i</sub>** is Pauling's composition-based
estimate of ionic character; **V<sub>M</sub>(i)** is the site Madelung energy from an
Ewald sum over formal charges, with **v<sub>i</sub>** the magnitude of the formal charge
at site *i*; and site complexity is inequivalent sites over sites at spglib symprec 0.01.
A conditional law is satisfied by any structure whose trigger is not met, so the clauses
of Law 2, Law 3 and Law 6 confine each law to its domain.

On the held-out benchmark (5,297 real, 3,612 damaged structures, never seen by any
fitting step):

| set | laws | satisfaction rate (real) | detection rate (damaged) |
|---|---|---|---|
| Set 1 | Law 1 (permissive τ) | 0.9919 | 0.2890 |
| Set 1′ | Law 1 (permissive) + Law 2 | 0.9894 | 0.3837 |
| Set 2 | Law 1, Law 3–Law 5 | 0.9579 | 0.6121 |
| Set 3 | Law 1, Law 3–Law 6 | 0.9171 | 0.7004 |
| Set 4 | Law 1, Law 3–Law 8 | 0.8180 | 0.9111 |

Set 4's worst damage class is still detected at 0.7338. Pauling's rules 2–5 applied
jointly are satisfied by 0.0651 of experimental structures. On the re-damaged
parent benchmark the same sets read 0.991 / 0.268 for Set 1 and 0.830 / 0.879 for
Set 4; satisfaction and detection are always quoted from one population, and the
population is always named.

## The synthesis score

`PSS` is a PRIS-derived synthesis score. Every term is a quantity one of the laws already
measures, refitted to the experimental record of what has been made, with volume per atom
dominating (standardised weight −4.90):

- At matched satisfaction on experimental structures, PSS screens **83.7%** of
  hard-to-synthesize structures, **31.8 percentage points** more than a computed
  hull-energy threshold, which reaches 72.0% and additionally requires a relaxation and a
  phase-hull reference.
- Retained on 80.7% of experimental structures, it screens 51.9% of the candidate set.
- On the most confident fifth of same-composition pairs it reaches **0.944** accuracy
  against **0.844** for DFT energy above the hull.
- No synthesis label entered the discovery of the laws, so the correlation between PRIS
  plausibility and synthesizability is a finding rather than a fitting target.

Set 1 to Set 4 are conservative discrete screens that name a violated mechanism; PSS
gives continuously tunable control over how strongly the combined evidence shortens a
queue.

## Judging a structure

Give it a structure file. It reports every law, the measured quantity, the threshold,
the verdict, the mechanism the law tests, all five nested sets, and PSS.

```bash
python src/pris_analyze.py mystructure.cif
python src/pris_analyze.py --quiet *.cif        # one verdict line per file
python src/pris_analyze.py --json POSCAR        # machine-readable
```

```
  MgAl2O4, 14 sites, charges from integer charge balancing, ionic character f_i = 0.759

    law    quantity                                measured  threshold  verdict  applied in       mechanism
    Law 1  reduced contact rho                       0.9865     0.8040     ok    1, 1', 2, 3, 4   short-range repulsion
    Law 2  reduced contact rho                       0.9865     1.0500     ok    1'               ionic contact
    Law 3  mean reduced cation-anion contact         4.0000     1.0810     --    2, 3, 4          packing
    Law 4  range of site Madelung energy / valence    3.6122    31.4500     ok    2, 3, 4          electrostatic balance
    Law 5  largest site Madelung energy            -20.2144    15.1700     ok    2, 3, 4          electrostatic balance
    Law 6  fraction of like-charge bonds             0.0000     0.0001     ok    3, 4             electrostatic balance
    Law 7  inequivalent sites / sites                0.2143     0.6667     ok    4                crystallographic site complexity
    Law 8  mean |BV sum - v_i| / v_i                 0.0384     0.7143     ok    4                bond-valence conservation

    Set 4  crystal chemistry           plausible
    PSS  +3.915        VERDICT  PLAUSIBLE
```

When a structure fails, the point is not the verdict but the last column:

```
$ python src/pris_analyze.py --quiet damaged/*.cif
IMPLAUSIBLE  compressed.cif   bond-valence conservation, short-range repulsion
IMPLAUSIBLE  expanded.cif     bond-valence conservation
```

Thresholds and PSS weights are read from the frozen artefacts in
`agent_loop/frozen/`, so a verdict produced today is the verdict the manuscript
reports. `src/apply_rules.py` remains as a lighter alternative that needs neither
spglib nor bond-valence parameters and therefore covers Set 1 to Set 3 only.

Charges are formal oxidation states inferred from composition. `BVAnalyzer` is never
called: it infers valence from bond lengths, so testing a bond-length law on its output
would use the conclusion as the premise. Roughly 19% of structures cannot be judged
(multiple anions, complex molecular groups, no integer or fractional charge assignment).
**"Skipped" does not mean "passed."**

## Layout

```
PREREG.md            pre-registration: criteria, split and vocabulary, frozen before any evaluation
src/                 the laws, the score, and the figure code
figures/             every figure of the manuscript -> the script that draws it
agent_loop/          the 572 numbered investigations, their plans, tests and frozen chains
paper/data/          aggregate statistics each main figure is drawn from
paper/si_data/       aggregate statistics for the supplementary figures
data/                small reference tables (bond-valence parameters, Lewis acidity)
dft/                 the first-principles campaign: protocol, inputs and collected results
experiments/         the analyses behind Fig. 4 and five supplementary figures
pipeline/            data acquisition
tests/               checks on the laws, the score and the figures
manuscript/          submission sources and the two compiled PDFs
```

`manuscript/main.pdf` and `manuscript/si.pdf` are the compiled manuscript and
supplementary information. `agent_loop/README.md` explains which investigation
lines are the main line and which were later corrected.

## Reproducing

**Figures from committed data** — no external data needed:

```bash
pip install -r requirements.txt
python figures/make.py --list      # which script draws which figure
python figures/make.py "Fig. 1"    # one figure
python figures/make.py --all       # every generator once
cd manuscript && ./build.sh        # needs tectonic; writes main.pdf and si.pdf
```

Thirty figures, thirteen generators. Three script names predate the final
numbering and disagree with the figure they draw — `figures/manifest.json` is the
authority and `figures/README.md` spells the mismatches out. Four figures
(Fig. 4, S17, S19, S22) additionally need the positive–unlabelled score shards,
which are not redistributed; `figures/README.md` says so and gives the aggregate
numbers instead.

**Analyses from structures** — these need the derived feature store, which is gigabytes
of parquet and is not in this repository. Point the scripts at it:

```bash
export PRIS_FEATURES=/path/to/features/
python src/validity_rulesets.py   # rule sets vs deployed generative-model validity filters
python src/rank_rulesets.py       # rule sets on the same-composition ranking task
```

Both scripts reproduce previously published rows to four decimal places before emitting
any new number, and `rank_rulesets.py` refuses to write output if that check fails.

The scripts that fitted the law sets and the score (`src/l4_*.py`, `src/f2r*.py`,
`src/f3_*.py`) refuse to rerun their sealed evaluations; the frozen results are the ones
quoted above. `f2r` and `f3` are the original chain identifiers for what the manuscript
now calls PSS and its predecessors. Most other scripts still
contain machine-specific absolute paths from the original run. They are kept as the
audit trail of what was actually executed, not as a portable library.

## Data licensing — read before redistributing anything

- **ICSD** is **not redistributable**: FIZ Karlsruhe copyright plus the European Union
  *sui generis* database right.
- **ELEMENTA** is CC-BY-NC-4.0.
- **COD** is CC0.
- `data/bvparm2020.cif` is I. D. Brown's bond-valence parameter table (McMaster
  University); it may be redistributed free of charge for non-commercial use with
  its copyright notice intact, which the file header carries.

Only aggregate statistics are committed here. Per-structure intermediates keyed to ICSD
identifiers are excluded by `.gitignore` (`paper/data/*_raw.csv`) and must stay excluded.
Any public benchmark released from this work is built on COD alone.

## Two things worth knowing before trusting a number

**The split discipline.** Thresholds were fitted only on the `discovery` split (12,632
real / 8,590 corrupted). `calibration` (5,297 / 3,612) was never seen by a fitting step.
Satisfaction and detection rates are always quoted from the same split and the split is
always named. Assignment is by a seeded hash of each structure identifier, not grouped
by composition, so structures sharing a composition may cross partitions. A lockbox of
5,748 structures was sealed with a quota of three permitted openings. One opening has
been used (2026-08-01, a fitted-score validation that failed its pre-registered gate;
the full record is SI Note S8); two remain. Separately, when the analysis set was
widened after the freeze, 3,215 lockbox rows carried no split label and entered one
full-sample quantile fit. The thresholds have therefore seen part of the sealed set, and
its value as a final test should be discounted accordingly.

**The refutation ledger.** Eleven conclusions were written down as results and then
overturned by the agent that produced them. They are published with the surviving
results (`agent_loop/refutation_ledger.csv`, and Supplementary Figs. S1 and S2), each
paired with the cheap diagnostic that exposed it, because those diagnostics outlive the
claims that motivated them.

The manuscript and its Supplementary Information are the record for every number.
Where a figure and the text disagree, the text wins; `manuscript/main.pdf` and
`manuscript/si.pdf` are the compiled versions this repository was released with.

## Citation

If you use PRIS, PSS, or the code in this repository, please cite the manuscript:

```bibtex
@unpublished{song2026pris,
  title   = {Autonomous discovery of new structure plausibility laws for explainable
             and rapid crystal diagnosis and screening},
  author  = {Song, Zhilong and Cheng, Lixue},
  year    = {2026},
  note    = {Manuscript under review},
}
```

To cite the software and the frozen law definitions specifically:

```bibtex
@software{pris2026software,
  title   = {{PRIS}: Plausibility Rules for Inorganic Structures},
  author  = {Song, Zhilong and Cheng, Lixue},
  year    = {2026},
  url     = {https://github.com/AI4QC/PRIS},
  license = {MIT},
}
```

`CITATION.cff` carries the same metadata in machine-readable form, so GitHub's
"Cite this repository" button stays in step with this section.

## Licence

Code and manuscript text: MIT (see `LICENSE`). This does **not** extend to the third-party
structural databases described above, nor to the Springer Nature LaTeX class files
(`tex/sn-jnl.cls`, `tex/sn-nature.bst`), which carry their own terms.
