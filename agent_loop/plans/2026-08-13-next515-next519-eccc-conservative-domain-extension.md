# NEXT515--NEXT519 ECCC conservative domain extension

> Additive post-coverage design. Preserve every prior script/result and all
> canonical documents. Validation and replication remain sealed.

## 1. Why this continuation is admissible

NEXT470 froze element characteristic-coordination compatibility (ECCC) from
Hawthorne 2026 Appendix 3 and passed its complete 80+80 label-blind support,
nondegeneracy, invariance, and `<0.90` novelty gates. The full NEXT471 build
then covered `0.961915` of SCIGEN but only `0.932339` of WyFormer, below the
prospective `0.95` floor. NEXT472 was not authorized and no ECCC discovery
outcome was opened. Thus the ECCC direction and values remain outcome-blind.

The exact branch failed because the empirical table omits a small set of
elements and because some formal-charge assignments have an isolated site.
Inventing a characteristic CN, interpolating between elements, dropping those
sites, or lowering the coverage gate would be unjustified. This branch instead
freezes a conservative applicability-domain extension: any record outside the
exact ECCC domain receives compatibility zero. It cannot make an unknown-domain
record look plausible and is exactly ECCC on every shared supported record.

This follows the already explicit conservative-domain principle used by
NEXT495 CCLAB-CDE, but applies it to the still-unopened ECCC candidate. It is a
post-coverage exploratory extension, not a new preregistered novelty claim.

## 2. Hard boundary

Executable inputs are limited to element identities, deterministic NEXT19
formal valences, the Hawthorne Appendix 3 characteristic-CN asset, and one
raw initial unrelaxed periodic geometry used to build the unchanged
opposite-sign periodic Voronoi contact multigraph. It may not run/read DFT,
energy, force, stress, an MLIP or learned/model/proxy potential, relaxation,
trajectory, later geometry, same-composition alternative, validation, or
replication. No outcome is opened before the blind extension and complete
label-free coverage gates pass.

## 3. Frozen ECCC-CDE formula

For records in the exact NEXT470 applicability domain, reuse its formula
unchanged. For every cation site `c`, observed translated degree `CN_c` and
the fixed set `T_E` of printed characteristic CNs for element `E`, let

```text
t_c = argmin_(t in T_E) (|CN_c-t|, t),
D = sum_c |CN_c-t_c| / sum_c (CN_c+t_c),
ECCC = 1-D.
```

Freeze

```text
ECCC-CDE(x0) = ECCC(x0), if exact NEXT470 is supported;
               0,        if a cation element lacks T_E or any charged site is
                         isolated/no opposite-sign contact graph exists;
               fail,     for all other numerical/input failures.
```

The sole feature is `eccc_cde_conservative_domain_extension`, with sole
direction `protected_high`. No unknown-element interpolation, nearest-element
substitution, site omission, partial averaging, threshold, alternate graph,
direction, or companion feature is available. A label-blind optimistic
diagnostic may ignore unknown cations while retaining known ones; the
candidate must never exceed it. The diagnostic is never eligible as a law.

## 4. Sequential gates

1. **NEXT515 blind extension probe:** unchanged 80+80 geometry sample; require
   support `>=72/80`, `[0,1]`, at least 20 distinct values, equivalent-
   representation error `<=1e-8`, exact equality to NEXT470 on every shared
   supported row, and zero violations of candidate `<=` optimistic diagnostic.
   No prior feature or endpoint is opened.
2. **NEXT516 full build:** all 13,470/5,232 discovery rows, source coverage
   `>=0.95`, immutable hashes, and no imputation beyond the frozen supported
   zero cases.
3. **NEXT517 discovery audit:** only after NEXT516 passes, open the unchanged
   discovery outcomes as offline labels and apply the exact NEXT224/NEXT413
   rejected-extreme cohort, combined `1/16,15/16` inverse-CDF mapping, source
   and five-fold gates, with the frozen high direction.
4. **NEXT518/NEXT519:** run the inherited finite margin-local search and BROAD
   diagnostic only if NEXT517 explicitly authorizes them. Validation and
   replication remain sealed.

Any failed gate closes this exact branch. Once outcomes are visible, unknown
handling, formula, direction, quantiles, thresholds, widths, and amplitudes may
not be changed.
