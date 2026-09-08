# PU learning performance audit

An independent performance package; it does not modify the paper.

- `CGCNN-PU OOB`: CGCNN trained from scratch, 50 PU bags.
- `MatterSim-1M-MLP-PU OOB`: frozen MatterSim-v1.0.0-1M embeddings and 50 MLP heads.
- `per_bag_validation_auc.csv`: the best validation ROC-AUC saved per bag in the raw logs.
- `figure_*_validation_performance.{png,pdf,svg}`: per-model validation distributions, labelled explicitly as having no independent test split.
- `standard_roc_confusion/pu_models_roc_auc_confusion.pdf`: ROC/AUC and the threshold=0.5 confusion matrix.
- `raw_B_opt/`: the MatterSim 50-bag output after fixing the repeated decompression of the embeddings.
- `performance_audit_report.md/.pdf`: provenance, split protocol, metrics and the recommendation on aggregating A and B.
- `provenance_audit.json`: model provenance at file and line level, with checkable evidence.
- `remote/`: the protocol and log provenance audit; the state of the earlier 209577 is kept here.

The validation labels are experimental positives and per-bag sampled unlabelled pseudo-negatives, and may not be read as a real detection rate for failed synthesis;
OOB means only the out-of-bag average used when scoring the whole pool, not an independent test set.
