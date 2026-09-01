# MatterSim basin–hull pilot（Fig. 4c 候选诊断）

这是一个不改正文的探索性 GPU 试验。它从实验结构和 PU 低 CLscore 结构中各抽取 1,024 条，限制为 2–80 个原子、至少两种元素，并要求 raw MP 快照包含该化学体系所需的全部元素子空间。采样按 L4 保留/筛除、$S_{\rm syn}$ 四分位和数据来源分层；同一化学体系最多两条。

MatterSim 1.2.3（5M checkpoint）在 A40 上完成 2,048 条结构的 NEXT15 relaxation。2,045 条获得有效 basin–hull 分数，3 条实验 CIF 在 HPC pymatgen 下无法解析并记为 ABSTAIN。固定阈值 $B_{64}\ge0.20$ eV atom$^{-1}$ 时，实验组筛除 141/1021（13.8%），PU 组筛除 737/1024（72.0%）。支持行的 ROC AUC（PU=1）为 0.878（bootstrap 95% CI 0.861–0.894）。

结果支持把 eHull 作为与 L4、$S_{\rm syn}$ 并列的第三种筛选信号：在相同阈值下，PU 队列出现更多高 B64 结构。但 PU 的 raw-MP/MatterSim 能量参考并不自洽，导致极端负分（最低值 -0.16 eV atom$^{-1}$）；因此这里报告的是 basin–hull proxy 的区分能力和阈值响应，不把它表述为已验证的可合成性预测。来源异质性也很明显，详见 `e_hull_by_source.csv`。

图：`e_hull_pilot_fig4c.png`（a 阈值曲线，b 分数 ECDF，c L4/来源分层）。
