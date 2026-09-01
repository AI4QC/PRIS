# Independent L4 versus formula screening

This additive analysis compares two independent choices for a pre-DFT queue. L4
removes only explicit PRIS violations. $S_{\rm syn}$ removes structures below a
threshold calibrated on the experimental cohort; it is not combined with L4.
The PU-low-score cohort is a model-selected proxy, not confirmed synthesis
failures.

At the L4 operating point, L4 retains 80.69% of
experimental structures and screens 51.88% of the PU
cohort (189,159/364,592). The formula, calibrated to
the same experimental retention, screens 83.68% of the PU cohort and
leaves 59,517 candidates instead of L4's 175,433.
It uses a low-score cutoff of -0.729385. These are alternative
operating modes: L4 supplies a discrete physical reason for every removal,
whereas $S_{\rm syn}$ supplies a tunable ranking threshold.
 D7 covers 98.72% of L4's PU exclusions in the separate mechanism audit.

No row in this output represents an L4-plus-formula cascade. Combined gates are
intentionally excluded from the main comparison because they spend the same
experimental-error budget twice and answer a different operational question.
