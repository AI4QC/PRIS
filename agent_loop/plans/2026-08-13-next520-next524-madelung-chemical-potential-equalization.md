# NEXT520--NEXT524 Madelung chemical-potential equalization

> Additive frozen design. Preserve every prior script, result, report, and
> canonical document. Validation and replication remain sealed.

## 1. Scientific question and repository boundary

Rappe and Goddard's charge-equilibration construction predicts atomic charge
distributions by equalizing site chemical potentials built from experimental
atomic ionization potentials, electron affinities, atomic hardnesses, and
electrostatic interactions (https://doi.org/10.1021/j100161a070). Mulliken's
absolute electronegativity is the half-sum of ionization potential and electron
affinity (https://doi.org/10.1063/1.1749394). This branch asks a deliberately
narrower inverse question: do the composition-derived formal charges already
placed on one raw periodic structure make those analytic site chemical
potentials approximately equal?

The repository already contains total/component/site Ewald summaries
(NEXT21), Ewald coordinate derivatives (NEXT34/NEXT77), charge spectra
(NEXT36/NEXT148/NEXT363), graph Poisson potentials (NEXT315), site-energy
class homogeneity (NEXT505), and charge--Madelung-potential ordering (NEXT510).
None combines atomic Mulliken electronegativity and hardness with the periodic
Madelung site potential and tests charge-space stationarity. This is therefore
a new operational hypothesis here, not a new QEq method and not a theorem of
crystal stability.

The frozen elemental inputs are pymatgen 2026.5.4 `periodic_table.json.gz`,
SHA-256 `b11669f8ccb0a9fe7647d9026ecbd30ee15ded7c464df828820a15768556d0aa`.
Only its tabulated first atomic ionization energy and electron affinity fields
are read. They are fixed elemental constants, not structure-level DFT labels.
All 87 elements occurring in the two discovery sources have finite values.

## 2. Hard executable boundary

The executable may read only element identities, deterministic NEXT19 formal
valences, cell, and positions from one initial raw unrelaxed periodic
structure; the frozen atomic table above; and a classical analytic
point-charge Ewald sum with `compute_forces=False`. It must not run or read
DFT; read an electronic density, DFT energy/force/stress, relaxed structure,
trajectory, later geometry, or same-composition alternative; invoke an MLIP,
learned energy/force/stress proxy, model/proxy potential, or empirical
short-range potential; optimize charges; or move coordinates/cell.

Discovery DFT outcomes may be used only as offline labels after this design,
the code, ordered blind probes, and full discovery feature tables are frozen.
Validation and replication remain sealed through NEXT520--NEXT524.

## 3. Frozen MCPE formula

For site element `Z_i`, formal charge `q_i`, first ionization energy `I_i`,
electron affinity `A_i`, and analytic Ewald site potential `phi_i`, define

```text
chi_i = (I_i + A_i) / 2
eta_i = I_i - A_i
mu_i  = chi_i + eta_i q_i + phi_i
d_i   = chi_i + eta_i |q_i| + |phi_i|.
```

The sign convention is positive `q_i` for electron deficiency. The QEq
stationarity condition is equality of all `mu_i` under total-charge
conservation. Freeze the bounded all-site-pair discrepancy

```text
delta_ij = |mu_i-mu_j| / (d_i+d_j),       i,j in {1,...,N}
MCPE(x0) = round_1e-10(1 - mean_ij delta_ij).
```

Because `|mu_i| <= d_i`, every `delta_ij` lies in `[0,1]`. Diagonal pairs are
included with zero discrepancy. Using all `N^2` ordered pairs makes exact
integer supercell replication duplicate numerator and denominator populations
by the same square multiplicity. The score is also invariant to site order,
rigid rotation/translation, and unimodular rebasing. Uniform cell scaling is
not an equivalence: it physically changes the Coulomb/atomic competition.

Freeze exactly one feature,
`mcpe_madelung_chemical_potential_equalization`, in direction
`protected_high`. No charge optimization, screened interaction, fitted
parameter, element subset, pair subset, quantile, alternate sign, alternate
normalization, companion feature, or direction is eligible.

## 4. Analytic and firewall tests

Tests must establish exact equal/unequal chemical-potential cases; boundedness;
the algebraic `chi`, `eta`, `mu`, and pair population; deterministic rounding;
Ewald energy--potential identity; atomic-table identity; rigid, translation,
site-permutation, unimodular-rebase and exact-supercell invariance. Malformed,
nonneutral, zero-charge, missing atomic-data, nonperiodic, nonfinite,
calculator-bearing, metadata-bearing, or extra-array inputs fail closed.

## 5. Frozen sequential gates

1. **NEXT520 engineering:** unchanged 80 discovery geometries/source, no prior
   feature or label access. Require `>=72/80` support, range `[0,1]`, at least
   20 distinct values at `1e-10`, and representation error `<=1e-8` per source.
2. **NEXT520 novelty:** only after engineering passes, compare the same rows
   with the complete frozen numeric label-free universe and direct NEXT21,
   NEXT505, NEXT510, atomic-property-dispersion, Madelung-potential, and raw
   chemical-potential-spread controls. Require at least 40 joint finite rows
   and maximum absolute Spearman `<0.90` in each source. No label is opened.
3. **NEXT521 build:** only after both blind gates pass, build all discovery
   rows and require coverage `>=0.95` in both sources.
4. **NEXT522 audit:** use unchanged NEXT224/NEXT413 rejected-extreme cohort,
   inverse-CDF mapping, folds, and two-source gates in frozen high direction.
   Discovery outcomes are offline labels only. Any failure closes the branch.
5. **NEXT523/NEXT524:** a finite frozen formula search and BROAD diagnostic are
   permitted only if NEXT522 explicitly authorizes them. Validation and
   replication remain sealed.

After any failed gate, no direction, atomic definition, sign, denominator,
pair population, transform, threshold or gate may be repaired using the
observed result. Continuation must use a different physical mechanism.
