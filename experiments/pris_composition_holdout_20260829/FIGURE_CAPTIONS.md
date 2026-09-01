# Candidate SI figure captions

## `pris_composition_holdout_overview`

**Frozen PRIS law sets retain their performance on compositions unseen during
law discovery.** **a**, Chemistry overlap created by the original
structure-identifier split. Held-out experimental structures and their
chemically damaged counterparts are partitioned into compositions present in
the discovery population (grey), unseen compositions whose element set was
seen (light green), and unseen exact chemical systems (dark green). Damaged
structures inherit the composition of their experimental parent. **b**,
Experimental-structure satisfaction for the five frozen PRIS law sets in the
full held-out population (grey diamonds), the composition-shared subset (open
circles), and the composition-unseen subset (filled circles). **c**, Damage
detection for the same populations. Points are structure-weighted estimates;
bars are 95% percentile intervals from 10,000 bootstrap resamples of whole
reduced-composition clusters (seed 20260829). The rules, thresholds and
missing-feature convention were not changed.

## `pris_set4_unseen_chemistry_sensitivity`

**Set 4 remains effective under increasingly strict unseen-chemistry
definitions.** **a**, Experimental-structure satisfaction and pooled damage
detection in all held-out data (grey diamonds), compositions absent from the
discovery population (light-green circles), and exact chemical systems absent
from discovery (dark-green squares). **b**, Set 4 detection of the five
composition-preserving damage classes in the same populations. Points are
structure-weighted estimates. Bars are 95% percentile cluster-bootstrap
intervals from 10,000 resamples (seed 20260829), using reduced composition as
the cluster for composition-unseen data and exact element set for
chemical-system-unseen data. Missing features count as satisfied under the
frozen publication convention; their coverage is reported in the accompanying
source data.
