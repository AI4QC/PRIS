# MatterSim basin–hull pilot (diagnostic on the Fig. 4c candidates)

This is an exploratory GPU experiment that does not change the main text. It draws 1,024
structures each from the experimental structures and from the low-CLscore PU structures,
restricted to 2–80 atoms and at least two elements, and requires the raw MP snapshot to contain
every elemental subspace the chemical system needs. Sampling is stratified by whether L4 retains
or screens the structure, by $S_{\rm syn}$ quartile and by data source; at most two structures
per chemical system.

MatterSim 1.2.3 (5M checkpoint) completed the NEXT15 relaxation of all 2,048 structures on an
A40. 2,045 obtained a valid basin–hull score; 3 experimental CIFs could not be parsed by pymatgen
on the HPC system and were recorded as ABSTAIN. At the fixed threshold
$B_{64}\ge0.20$ eV atom$^{-1}$, 141/1021 (13.8%) of the experimental group are screened out and
737/1024 (72.0%) of the PU group. The ROC AUC over the supported rows (PU=1) is 0.878
(bootstrap 95% CI 0.861–0.894).

The result supports treating eHull as a third screening signal alongside L4 and $S_{\rm syn}$: at
the same threshold, the PU queue contains more high-$B_{64}$ structures. But the raw-MP/MatterSim
energy reference for PU is not internally consistent, which produces extreme negative scores (the
lowest is -0.16 eV atom$^{-1}$). What is reported here is therefore the discriminating power and
threshold response of the basin–hull proxy, not a validated prediction of synthesizability.
Source heterogeneity is also pronounced; see `e_hull_by_source.csv`.

Figure: `e_hull_pilot_fig4c.png` (a, threshold curves; b, score ECDF; c, stratified by L4 and by
source).
