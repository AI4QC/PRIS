# NEXT363--NEXT366 first non-extinct charge Bragg wavenumber

Date: 2026-08-13

Status: frozen after closing NEXT359/NEXT360 and before computing any NEXT363
feature value or opening any NEXT363 endpoint outcome.

## Scope and information boundary

The target remains an interpretable pre-DFT screen for raw generated or
theoretical periodic crystals. The executable may use element identities,
deterministic formal-charge inference, and one initial raw unrelaxed periodic
geometry only. It must not execute or consume DFT, energy/force/stress labels
or predictors, learned or proxy potentials, relaxation, trajectories, or
later geometries. Discovery endpoints may be opened only after a frozen
label-blind probe and complete formal label-free build pass. Validation and
replication stay physically sealed.

All files are additive. Existing scripts and content remain unchanged, and
only the independent no-DFT report may be extended before user review.

## Alternatives reviewed before selection

1. A general low-wavenumber Gaussian charge spectrum is rejected as a
   duplicate: NEXT36 already implements it and NEXT85/NEXT94 formally contain
   its six candidate features.
2. A normalized Ewald/Madelung statistic is rejected as a duplicate of NEXT21
   and the corresponding base feature population.
3. Unit-cell dipole or quadrupole cancellation is rejected because its value
   has an origin/polarization-branch ambiguity under periodic boundary
   conditions.
4. The first non-extinct charge Bragg wavenumber is retained. Unlike a
   Gaussian spectral sum, it is an exact extinction/onset statistic and has a
   representation-invariant integer-supercell interpretation.

## Frozen charge assignment and reciprocal enumeration

Use NEXT19 `infer_valence_assignment` unchanged, including its deterministic
integer oxidation-state, fractional oxidation-state, then electronegativity
partition policies. The returned site charges must be finite, neutral, and
contain both signs. No charge policy or subgroup is searched.

Let the row-vector lattice be `A`, wrapped fractional site coordinates be
`f_i`, cell volume be `V`, site count be `N`, and

```text
ell = (V/N)^(1/3),
G_h = 2 pi A^(-T) h,
rho_h = sum_i q_i exp(-2 pi i h dot f_i),
I_h = |rho_h|^2 / (N sum_i q_i^2).
```

Enumerate every nonzero integer vector `h` with
`0 < |G_h| ell <= 18` using the exact coefficient bounds already used by
NEXT36. The sole extinction tolerance is frozen at

```text
I_h >= 1e-12.
```

If no non-extinct vector exists inside the complete cutoff, abstain. The sole
public feature is

```text
fcbe_first_charge_bragg_wavenumber
    = min_{h != 0, I_h >= 1e-12} |G_h| ell,
```

quantized to `1e-10`. The sole frozen direction is `protected_high`: pushing
the first charge-density Bragg mode to a shorter dimensionless wavelength is
hypothesized to indicate stronger short-range charge compensation. No second
peak, intensity, smoothing width, cutoff, extinction tolerance, aggregation,
charge policy, or direction is searched.

## Representation theorem and certificates

For an integer supercell with translation quotient `T`, a new reciprocal mode
has amplitude equal to the primitive amplitude times the character sum

```text
sum_{t in T} exp(-i G dot t).
```

That sum is the supercell multiplicity for primitive reciprocal vectors and
zero for every newly introduced fractional reciprocal vector. At retained
vectors, numerator and `N sum(q_i^2)` both scale by the square of the
multiplicity, while `(V/N)^(1/3)` is unchanged. Consequently the normalized
intensity and first non-extinct dimensionless wavenumber are invariant under
any exact integer supercell.

Unit tests must cover analytic one-dimensional alternation embedded in three
dimensions, global scale and charge-amplitude invariance, rigid rotation,
periodic translation, site permutation, unimodular lattice rebasing, explicit
`2x1x1` and non-diagonal integer supercells, numerical extinction, complete
enumeration, and the geometry-only firewall.

## Sequential gates

The label-blind probe selects the same deterministic 80 discovery structures
per source and reads raw discovery geometry plus label-free controls only. In
each source it requires:

- support at least `72/80`;
- finite values in `(0,18]` and at least 20 distinct values at 10 decimals;
- maximum equivalent-representation error at most `1e-8`;
- maximum absolute Spearman correlation strictly below `0.90` against all
  numeric NEXT85/NEXT94 discovery base features, including charge-spectrum
  and Ewald/Madelung columns, plus all formal later label-free features through
  NEXT359.

No endpoint is read during the probe. Any failure terminates the branch and
NEXT364--NEXT366 are not created.

Only a passing probe authorizes the all-row NEXT363 label-free build. Formal
coverage must be at least `0.90` independently in 13,470 SCIGEN and 5,232
WyFormer discovery rows. Only a passing immutable manifest authorizes
NEXT364, which reuses the unchanged NEXT224/NEXT268/NEXT324 audit: the frozen
rejected-extreme cohort, reduced-formula five-folds, inverted-CDF `1/16` and
`15/16`, coverage `0.90`, class count `20`, pooled AUC `0.55`, macro AUC
`0.53`, worst-fold AUC `0.50`, and the frozen `protected_high` direction.

Failure in either source gives zero eligible hypotheses and terminates the
branch. NEXT365/NEXT366 may exist only if NEXT364 explicitly authorizes the
predeclared formula-search stage. Nothing may be reversed, repaired, imputed,
or tuned after outcomes are opened.

## Non-functional assumptions and decision log

- Complete reciprocal enumeration and deterministic fail-closed behavior take
  priority over runtime; the existing two-million-vector guard remains fixed.
- Geometry and label archives stay local and are never transmitted.
- Numerical/provenance failures are abstentions, never imputed values.
- Rejected duplicate mechanisms remain documented rather than silently
  retried.
- The first-Bragg onset was chosen only after proving integer-supercell
  invariance and before observing any NEXT363 value or outcome.
- The full base feature table is included in novelty control because this
  mechanism is intentionally adjacent to NEXT36, and a later-stage-only
  comparison would be insufficient.
