# NEXT19 design: a bond-valence transport feasibility law

**Status:** an autonomous-execution design, written after the boundaries were confirmed. This
design creates only new files and new artefacts; it overwrites no existing script, report or the
paper.

## 1. Objective and hard boundaries

NEXT19 aims to find a structural plausibility law for crystals that executes independently on a
single structure, is interpretable, and needs no DFT. It must return `KEEP`, `REJECT` or
`ABSTAIN` before a candidate structure enters DFT, and while executing it may read only:

- the raw, unrelaxed lattice and atomic coordinates;
- the stoichiometry and element-table properties;
- the periodic adjacency, coordination, Voronoi, bond-valence, formal-valence and Ewald/Madelung
  quantities derived from the raw structure;
- fixed empirical constants, formulas and thresholds.

The law's execution path may not read or compute DFT energies, forces or stresses, DFT-relaxed
structures, MatterSim/MLIP energies, machine-learning potentials, surrogate energies, or the
relative energies of same-composition candidates. Nor may it depend on the other structures in a
group, so an isolated candidate must be decidable on its own. DFT labels may appear in exactly two
places: threshold selection on the historical development set, and the external blind evaluation
after the law and thresholds are frozen.

Every computational failure fails open: return `ABSTAIN`, and never reject a structure because
valences, neighbours or a numerical solve were unavailable.

## 2. The boundary against the literature, and the new hypothesis

Bond-valence sums and the Global Instability Index already measure structural strain through the
deviation of per-site bond-valence sums from formal valences; Charge Distribution/CHARDI already
distributes formal charge over bonds by effective coordination number. NEXT19 therefore does not
claim "bond-valence sums" or "distributing charge by coordination" as new laws in themselves.

The new hypothesis to be tested is: **the geometric connection graph of a real or low-DFT-energy
structure should be able to carry a whole-cell bond-valence flow satisfying every cation supply
and every anion demand simultaneously, without seriously violating the geometric prior; an
implausible structure requires either a few abnormally overloaded edges or large-scale
redistribution.**

That extends Pauling's second rule from a local averaged equality into a global conservation
problem on a periodic multigraph, and makes explicit the topological bottlenecks a local average
can conceal.

## 3. The core formulas

First infer each site's formal valence `z_i` through the repository's single entry point. If
neither integer nor fractional formal valences have a solution, fall back to a pre-frozen Pauling
electronegativity partition: each site takes `q_i = mean(chi)-chi_i`, and the positive and
negative sides are separately normalised to `+1` and `-1`, so that neutrality holds exactly. The
main transport quantities are insensitive to the overall charge scale; this fallback fixes only
the signs and the relative supply and demand, and introduces no energy. A single-element
structure, or one whose electronegativities are all identical so that no two-sided partition
exists, still returns `ABSTAIN`. Keep only opposite-sign periodic neighbour edges
`e=(c,a,image)`, where `c` is a cation and `a` an anion. For each cation edge, compute the fixed
geometric weight

\[
w_{ca}=\omega_{ca}\exp\{-\alpha[(d_{ca}/d_{c,\min})-1]\},
\]

where `omega` is the non-negative adjacency weight from periodic Voronoi/CrystalNN, `d_c,min` is
that cation's nearest opposite-sign distance, and `alpha` comes from a small pre-declared
catalogue. The Pauling-type bond valence the cation emits under the geometric prior is

\[
p_{ca}=z_c\frac{w_{ca}}{\sum_b w_{cb}}.
\]

Let `s_ca` be the final bond-valence flow. The first linear programme minimises the overload
`kappa`:

\[
\begin{aligned}
\min_{s,\kappa}\quad & \kappa \\
\text{s.t.}\quad
& \sum_a s_{ca}=z_c, \\
& \sum_c s_{ca}=|z_a|, \\
& 0\le s_{ca}\le \kappa p_{ca},\quad \kappa\ge1.
\end{aligned}
\]

The second linear programme minimises `sum |s_ca-p_ca|` at the fixed `kappa*`. Three main
quantities are emitted:

- `vt_overload = kappa* - 1`: the largest relative edge overload needed to satisfy global valence
  conservation;
- `vt_reallocation = sum|s-p|/(2 sum z_c)`: the fraction of formal charge that has to be
  rerouted;
- `vt_anion_mismatch_max`: the relative mismatch of the unoptimised Pauling/CHARDI-style prior at
  the worst anion.

Coverage, the periodic edge count, the graph's connected components, the solver status and
diagnostics are also emitted, but no energy.

## 4. The candidate catalogue and the comparators

Fix the following small catalogue before reading any development label, so that the symbolic
search cannot grow without limit:

- valence strategies: integer oxidation states, restricted fractional oxidation states, and the
  Pauling electronegativity-partition fallback, in that fixed order;
- adjacency: `CrystalNN` weights and periodic Voronoi solid-angle weights;
- distance decay: `alpha in {0, 2, 4, 6}`;
- single-quantity laws: each of the three main quantities exceeding a fixed threshold;
- two-quantity laws: a monotone linear combination of `overload` and `reallocation`, at most two
  terms;
- consensus laws: a bond-valence transport violation together with a violation of an existing,
  independent short-contact or Madelung-sign guard;
- traditional comparators: Pauling P2-P5, the old P9 Lewis mismatch, and BVS/GII or CHARDI-style
  mismatch where computable.

Thresholds may come only from a pre-listed numeric grid or from development-set quantiles, and
must be validated source-wise. Any candidate added after the external Alexandria results become
visible belongs to the next round and may not be written back into NEXT19.

## 5. Data flow and isolation

1. **Development source 1: WBM.** Extract features from the 2,048 x0 geometry-only structures of
   `next14_wbm_acsc_holdout`; the historical private table supplies the stable, high-energy and
   hull labels only inside the standalone evaluator.
2. **Development source 2: ELEMENTA.** Use the 1,988 x0 geometry-only structures of
   `next16_elementa_holdout_v2` for cross-source validation; labels are joined only inside the
   standalone evaluator.
3. **External source: Alexandria.** Use the 379 geometry-only structures already isolated. The
   endpoint fields may not be extracted until the NEXT19 law and thresholds are frozen and the
   predictions and hashes are sealed.

The feature builder's CLI accepts only a geometry archive, a geometry manifest, label-free
metadata and an output directory. It rejects any table containing energy, force, stress, relaxed,
MatterSim or endpoint fields. The evaluator is the only module permitted to read the historical
development labels. Alexandria endpoint extraction and evaluation use a separate module and
require the frozen protocol hash and the external prediction hash to match exactly.

## 6. Selection gates and failure conditions

WBM is used only to select thresholds from the fixed candidate catalogue; ELEMENTA must then pass
without any refitting:

- coverage Wilson lower bound at least 0.90;
- group-minimum recall Wilson lower bound at least 0.95;
- valuable recall Wilson lower bound at least 0.95;
- reject precision Wilson lower bound at least 0.90, targeting 0.95;
- DFT savings Wilson lower bound at least 0.10;
- no same-composition group may be rejected in its entirety;
- the key safety metrics relative to Pauling must improve, and high-energy rejection must not
  degrade appreciably.

Only a candidate clearing those gates is frozen. The external Alexandria evaluation uses exactly
the same rules, thresholds, valence strategy and neighbour parameters. If no candidate passes,
NEXT19 is a strict negative result: keep the code, the aggregate results and the mechanistic
diagnostics, write a standalone negative-result report, and do not open or repeatedly reuse the
external labels to tune.

## 7. Outputs and the boundary against the paper

Code goes in `src/next19_*`, tests in `tests/test_next19_*`, aggregate artefacts in a new
`outputs/20260802_next19_*`, and the identifier-bearing feature and join tables in
`$PRIS_ARCHIVE/next19_*`. On success or failure, the only addition is
`reports/2026-08-02-next19-valence-transport-law.md`. Without the user's confirmation, nothing in
`paper/`, `notes/`, `tex/`, `README.md`, `PREREG.md` or any existing report is modified.
