# PU model performance audit

Status: **COMPLETE_WITH_VALIDATION_ONLY**

This package reports per-bag validation performance from the frozen CSAgent-v2 artifacts. No independent test split or observed negative synthesis-failure labels exist in the source checkout. The word OOB refers to the full-pool CLscore construction, not to an independent test accuracy estimate.

The SI-ready standard diagnostic is `standard_roc_confusion/pu_models_roc_auc_confusion.pdf`: each model has a ROC/AUC panel and a threshold=0.5 confusion matrix. Slurm job 209577 was stopped after the original embedding path stalled. The corrected job 209663 completed all 97 embedding shards and 50 bags in 1 min 26 s; its compact outputs are in `raw_B_opt/`. The validation protocol still has no independent test split.
