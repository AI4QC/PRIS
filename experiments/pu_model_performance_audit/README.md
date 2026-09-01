# PU learning performance audit

独立性能包，不修改论文。

- `CGCNN-PU OOB`: 从头训练 CGCNN，50 个 PU bags。
- `MatterSim-1M-MLP-PU OOB`: MatterSim-v1.0.0-1M 冻结嵌入和 50 个 MLP heads。
- `per_bag_validation_auc.csv`: 原始 logs 中每袋保存的 best validation ROC-AUC。
- `figure_*_validation_performance.{png,pdf,svg}`: 单模型验证分布图；明确标注无 independent test split。
- `standard_roc_confusion/pu_models_roc_auc_confusion.{png,pdf,svg}`: 两个模型的标准 ROC/AUC + threshold=0.5 混淆矩阵图。
- `raw_B_opt/`: 修复 embedding 重复解压后的 50-bag MatterSim 验证输出（Slurm 209663）。
- `性能审计报告.md/.pdf`: 中文来源、划分协议、指标和 A/B 聚合建议。
- `provenance_audit.json`: 文件/行号级模型来源与可复核证据。
- `remote/`: 协议与日志 provenance 审计；早期 209577 已停止，优化作业 209663 的结果另存于 `raw_B_opt/`。

验证集标签是实验正例与按袋抽取的未标注伪负例，不能解释为真实合成失败识别率；
OOB 只表示全池评分时的 bag 外平均，不是独立测试集。
