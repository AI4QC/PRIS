# NEXT510--NEXT514 Madelung charge--potential antitonicity

> Additive frozen design. Preserve every prior script, result, report, and
> canonical document. Validation and replication remain sealed.

## 1. Scientific question and prior-work boundary

For a fixed periodic geometry and neutral formal-charge assignment, the
point-charge Madelung energy is a quadratic form in the charges and its
derivative with respect to a site charge is that site's Madelung potential.
Moving a small amount of charge from one site to another therefore gives a
parameter-free first-order test of whether the more positive ion occupies the
more negative electrostatic site. This branch asks whether the pair population
of those exchange margins is a transferable raw-structure plausibility law.

The repository already contains normalized total and component Ewald energies
plus unconditioned site-energy extrema/spread (NEXT21), coordinate derivatives
of the analytic Coulomb sum (NEXT34), reciprocal charge spectra (NEXT36),
charge-order spectra (NEXT148), contact-graph charge modes (NEXT311/NEXT315),
and element/formal-valence homogeneity of site energies (NEXT505). None tests
the antitone ordering of formal charge and Ewald site potential over every pair
of unequal-valence sites. In particular, NEXT505 discards all between-class
ordering after its ANOVA partition, whereas this candidate uses only that
ordering.

Site-resolved Madelung potentials are standard electrostatic quantities, and
published structure studies use their ordering to select favourable ionic
occupancies. Chen's Madelung-matrix method also identifies high-throughput
evaluation of cation distributions and antisite configurations as an
application of electrostatic matrix analysis. These results motivate the
mechanism but do not publish the bounded pair score below; MCPA is a new
operational hypothesis of this project, not a literature theorem of crystal
stability.

References:

- https://pubmed.ncbi.nlm.nih.gov/22242970/
- https://doi.org/10.1002/jcc.24360
- https://journals.iucr.org/m/issues/2018/06/00/lt5011/index.html
- https://jp-minerals.org/vesta/en/doc/VESTAch14.html

Kapustinskii and Born--Lande/Born--Mayer residuals were rejected before this
freeze. They require empirical ionic radii, a repulsive constant or exponent,
or a nearest-neighbour convention, and would substantially duplicate the
failed radius/contact and total-Madelung branches rather than isolate a new
site-ordering mechanism.

## 2. Hard executable boundary

The executable may read only element identities, deterministic NEXT19 formal
valences, the periodic cell, and positions from one initial raw unrelaxed
structure. It may evaluate a classical analytic point-charge Ewald sum with
`compute_forces=False`. It must not run or read DFT; use an electronic density,
learned energy/force/stress proxy, MLIP, model/proxy potential, empirical
short-range pair potential, or relaxation; read a trajectory, later geometry,
same-composition alternative, or discovery outcome before the blind gates.

The Ewald sum is part of the proposed analytic law itself. It has no
exchange-correlation functional, wavefunction, fitted energy term, learned
parameter, force evaluation, or geometry update. Discovery outcomes may be
opened only as offline labels after the design, implementation, blind
engineering, blind novelty, and complete label-free build are frozen.
Validation and replication remain sealed throughout NEXT510--NEXT514.

## 3. Frozen MCPA formula

Let `q_i` be nonzero neutral formal charges and `E_i` the site contributions
returned by the analytic Ewald decomposition, so that

```text
E_M = sum_i E_i = (1/2) sum_i q_i phi_i,
phi_i = 2 E_i / q_i.
```

For every unordered pair `i<j` with `q_i != q_j`, define the normalized
first-order exchange margin

```text
a_ij = -sign(q_i-q_j) (phi_i-phi_j) / (|phi_i|+|phi_j|),
```

using `a_ij=0` only when both potentials are exactly zero. Since
`|phi_i-phi_j| <= |phi_i|+|phi_j|`, every margin lies in `[-1,1]`. The
unnormalized numerator is the first-order change per absolute transferred
charge when the two current formal charges are exchanged: positive values mean
the more positive charge already occupies the lower potential.

Freeze the sole score

```text
MCPA(x0) = round_1e-10((1 + mean_{i<j,q_i!=q_j} a_ij) / 2).
```

The sole feature is
`mcpa_madelung_charge_potential_antitonicity`, with the sole prospective
direction `protected_high`. No minimum, quantile, charge-magnitude weighting,
element subset, sign-pair subset, class restriction, radius, threshold,
alternate Ewald component, direction, companion feature, or formula search is
available. Neutral nonzero charges of both signs guarantee at least one
eligible pair.

The common charge magnitude and inverse-length scale cancel pairwise. Exact
supercell replication repeats every unequal-charge primitive pair by the
square of the replication multiplicity and therefore leaves the mean
unchanged. The score is also invariant to site order, rigid motion,
translation, unimodular rebasing, uniform geometry scale, and a common nonzero
charge scale (including global sign reversal).

## 4. Required analytic and firewall properties

Tests must establish exact stable/reversed/zero-potential limits, bounds,
deterministic rounding, the Ewald energy--potential identity, and all
invariances listed above. Malformed, non-neutral, zero-charge, nonperiodic,
nonfinite, calculator-bearing, metadata-bearing, or extra-array inputs fail
closed. Formal-valence inference failure is explicit; no charge or direction
is learned from outcomes.

## 5. Frozen sequential gates

1. **NEXT510 engineering probe:** deterministically select the unchanged 80
   discovery records per source without opening prior features or outcomes.
   Require at least `72/80` finite support, range `[0,1]`, at least 20 distinct
   values at `1e-10`, and maximum equivalent-representation error `<=1e-8` in
   each source.
2. **NEXT510 novelty probe:** only after engineering passes, compare the same
   records with every frozen numeric label-free control and direct
   recomputations of NEXT21 total/component/site features, NEXT505 MVCH, and
   the immediate charge/potential diagnostics. Require at least 40 joint finite
   rows and maximum absolute Spearman `<0.90` in each source. No endpoint is
   opened.
3. **NEXT511 formal build:** only after both blind probes pass, build all
   discovery rows with coverage `>=0.95` in each source and immutable hashes.
4. **NEXT512 discovery audit:** use the unchanged NEXT224/NEXT413
   rejected-extreme population, inverse-CDF aggregation, folds, source gates,
   and frozen direction. Open discovery outcomes only as offline labels. If no
   frozen hypothesis is eligible, terminate immediately.
5. **NEXT513/NEXT514:** run a bounded formula search and BROAD diagnostic only
   if NEXT512 explicitly authorizes them. Validation and replication remain
   sealed.

Any failed gate closes the MCPA branch. Its direction, pair population,
normalization, aggregation, quantiles, or gates may not be repaired after the
failure. The overall zero-DFT law search then continues with a physically
different mechanism.
