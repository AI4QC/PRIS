# NEXT22 Scale-Calibrated Bond-Valence Equilibrium Design

NEXT22 is an additive pure-analytic branch.  It uses one raw structure,
composition-derived charges, a periodic opposite-sign neighbor graph, and the
repository's frozen IUCr/Brown bond-valence parameter table with the existing
non-fitted nearest-valence/Brown fallback.  It does not use DFT at execution,
learned potentials, relaxation, or same-composition candidates.

For bond strengths s_ij=exp((R0_ij-r_ij)/B_ij), site sums b_i, and charge
magnitudes a_i=|q_i|, the unknown global charge amplitude is removed by the
closed-form scalar

    lambda = argmin_l sum_i (l b_i-a_i)^2
           = sum_i b_i a_i / sum_i b_i^2.

This is descriptor normalization on the supplied structure, not coordinate or
cell optimization.  Frozen outputs are site-mismatch RMS/q95/max and separate
cation/anion RMS values, bond-valence-vector asymmetry RMS/max, effective
coordination mean/min, isolated-site fraction, parameter-source fractions,
and the disclosed global scale.  WBM alone selects any threshold; ELEMENTA is
unchanged validation; Alexandria remains unopened until a full freeze.
