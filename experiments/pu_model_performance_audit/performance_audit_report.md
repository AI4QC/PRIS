# Performance and provenance audit of the PU synthesizability scorers

This report is an independent audit artefact; it does not modify the paper's main text or
Supplementary Information. What is audited is the two PU-learning pipelines behind
`clscore_all.csv` (scorer A) and `clscore_b_all.csv` (scorer B) on the server. Every AUC in this
report comes from the saved per-bag logs; they are validation metrics, not a real detection rate
for failed synthesis.

## Conclusion first

Scorers A and B use the same 50 PU bag splits and the same out-of-bag (OOB) masks, but different
encoders and training heads, so they can serve as complementary ranking opinions under two
structural representations:

| output label | model | best validation ROC-AUC over 50 bags (mean +/- sample SD) | range |
|---|---|---:|---:|
| CGCNN-PU OOB | crystal graph convolutional network (CGCNN) trained from scratch | 0.976538 +/- 0.000763 | 0.9749-0.9781 |
| MatterSim-1M-MLP-PU OOB | frozen MatterSim-v1.0.0-1M representation + an MLP PU head | 0.946561 +/- 0.000941 | 0.9443-0.9486 |

The validation labels here are "experimental positive structures" against "pseudo-negative
structures drawn for that bag from the unlabelled pool". The source data contains no observed
hard-to-synthesise or failed-synthesis negative labels, and no untouched independent test file.
The table above therefore may not be called test accuracy, real synthesizability accuracy, or
recall on failed syntheses.

The optimised MatterSim recomputation (Slurm 209663) additionally gives per-bag PR-AUC
`0.938953 +/- 0.001369`, accuracy at threshold 0.5 `0.877183 +/- 0.001771`, balanced accuracy
`0.877033 +/- 0.001875` and F1 `0.871581 +/- 0.002296`. These still describe only the validation
task of "experimental positives versus sampled pseudo-negatives".

## What the two scores actually are

### A: CGCNN-PU OOB

The model description in `03_train_bags.py` gives CGCNN's 3 convolution layers, 64-dimensional
atom features, 128-dimensional crystal features and a binary classification head (lines 2-13);
each bag computes ROC-AUC on its validation set and saves the current best checkpoint (lines
50-62 and 109-118). The model is constructed at `pu_common.py` lines 244-258, calling
`CrystalGraphConvNet` with `atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1`.

Lines 2-6 and 51-69 of `04_predict_clscore.py` define the whole-pool score: the mean model
probability over the bags a structure was not drawn into, that is

\[
  \mathrm{CLscore}_A(x)=
  \operatorname{mean}_{b:x\notin\mathrm{bag}_b}P_b(y=1\mid x).
\]

That is the OOB whole-pool ranking score; OOB is not a synonym for "independent test set".

### B: MatterSim-1M-MLP-PU OOB

Lines 2-23 of `07_embed_pu_head.py` state that B uses a frozen MatterSim-v1.0.0-1M encoder and a
small MLP head, without fine-tuning the potential; lines 282-287 give the head structure
`128 -> 256 -> 256 -> 2` (SiLU, Dropout, LogSoftmax). Lines 301-325 show that the training and
validation rows follow the same bag splits and that the best head is selected by validation
ROC-AUC; lines 329-357 write `clscore_b_all.csv` using the same bag-membership masks.

Lines 4-13 of `11_compare_scorers.py` describe A and B as two scorers with different structural
inductive biases and note that they share the OOB masks. A shared mask supports paired
per-structure comparison and consensus screening, but it does not mean the two scores have been
probability-calibrated.

B's MatterSim preprocessing skips a small number of disordered structures, very large cells and
near-coincident atoms (the implementation is at `07_embed_pu_head.py` lines 211-230). The
coverage of the current whole-pool files is therefore not identical: `clscore_all.csv` has
8,109,897 unlabelled structures and `clscore_b_all.csv` has 8,108,676, with B short by 1,221.
Any per-structure comparison of A and B should first take the common IDs; a missing row must not
be treated as a low score or as a negative.

## The split protocol and checkable counts

Lines 2-13 and 38-56 of `02_split_bagging.py` specify: 20% of the experimental positives are
fixed as the validation positives shared by every bag; each bag draws as many pseudo-negatives
from the unlabelled pool as there are positives, 20% of which become that bag's validation
pseudo-negatives and the rest training data; only the unlabelled structures a bag did not draw
enter that bag's OOB scoring.

`remote/protocol_audit.json`, produced by the server recomputation, gives the exact counts for
this snapshot:

- the pool has 8,208,887 rows, of which 98,990 are experimental positives and 8,109,897 are
  unlabelled;
- 50 bags, each with 39,596 validation rows (19,798 positives + 19,798 pseudo-negatives) and
  158,384 training rows;
- the fixed positive validation set is identical across all 50 bags, and the training/validation
  intersection is 0;
- the audit records `independent_test_available: false` explicitly, because there are no real
  negatives and no untouched test file.

The raw per-bag log text and its line-number index are kept in
`remote/log_line_provenance.tsv`. The A logs are the `DONE best vAUC` lines in
`/data1/home/hzxy10/csllm/logs/train_200535_1.out`, `train_200536_*.out` and
`train_200537_*.out`; the B log is lines 1-50 of
`/data1/home/hzxy10/csllm/logs/headB_200736.out`.

## What the figures and tables mean

`figure_cgcnn_pu_validation_performance.*` and
`figure_mattersim_1m_mlp_pu_validation_performance.*` show the distribution of the best
validation ROC-AUC over the 50 bags for each model (in PNG, PDF and SVG). The figures state
explicitly that:

1. the labels are experimental positives and sampled pseudo-negatives;
2. the values are validation ROC-AUC, not independent-test ROC-AUC;
3. OOB means only the out-of-bag average used when scoring the whole pool.

The raw per-bag values are in `per_bag_validation_auc.csv`, the machine-readable summary in
`model_summary.json`, and file hashes and source scripts in `manifest.json`. The optimised B
output is kept in `raw_B_opt/`; the standard classification figure (one row per model, ROC/AUC
on the left and the threshold-0.5 confusion matrix on the right) is
`standard_roc_confusion/pu_models_roc_auc_confusion.pdf`. The confusion-matrix numbers are
per-bag mean counts, and the percentages in each cell are normalised along the validation-label
rows.

The original pipeline README (`script/pu_screening_v2/README.md` lines 42-48) writes A as
`0.9766 +/- 0.0016` and B as `0.9466 +/- 0.0022`. Those two +/- values are each half the per-bag
min-max range (0.0016 for A and 0.00215 for B, rounded to 0.0022), not the per-bag sample
standard deviation. This package also gives the sample standard deviation, labelled explicitly
as `SD`, so that the two kinds of summary are not conflated.

## Should A and B be averaged

The raw, uncalibrated `CLscore_A` and `CLscore_B` should not be averaged arithmetically and the
result called a new "synthesis probability" or "combined accuracy score". Although both lie in
0-1 and share the OOB masks, A is the output of a CGCNN classification head and B that of a
different MLP classification head over a MatterSim representation; the PU pseudo-negative
labels, the positive fraction, the scale of the representation and the calibration error may all
differ. The present evidence supports only:

- reporting the ranking and validation performance of A and B separately;
- using the intersection of the two low-score tails as a conservative consensus
  (`08_consensus_negatives.py` lines 105-115); or
- calibrating or quantile-transforming both on the same genuinely independent, reliably labelled
  data first, and only then defining a combined score with explicit weights.

The existing script ranks candidates by `CLscore_A + CLscore_B` in rank mode (lines 9 and
133-138). That should be described as an engineering rank aggregator, not as an average derived
from physical chemistry or from a probabilistic model.

## Recomputation status

The earlier job 209577 was stopped because of an implementation that decompressed the MatterSim
embeddings repeatedly. After fixing it to materialise `emb.npy` once per shard, the optimised job
209663 completed the validation recomputation of 97 shards and 50 bags in 1 min 26 s;
`raw_B_opt/` keeps the five compact output files and their hashes. The fix changes only the
computation path, not the splits, the model head or the labelling protocol. There is still no
independent test split, so this report continues not to call these results test-set performance.
