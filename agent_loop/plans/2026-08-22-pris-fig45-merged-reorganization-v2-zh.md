# PRIS Fig.4–5 合并重排与 PU 诊断补充 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在保留 PRIS 物理化学意义的前提下，把现有 Fig.4/5 和两个 PU 实验重排成一条清晰的证据链，并把逐法则贡献及两个 PU 模型的 held-out 性能完整放入 SI。

**Architecture:** 先冻结数据口径和 panel contract，再生成一张合并主图及配套 SI 图。正文和 LaTeX 只在图、事实和交叉引用审计通过后修改。二元 PRIS、连续任务公式和 PU 模型分数各自承担不同任务，不构造新的联合筛选门。

**Tech Stack:** Python 3.11、pandas、NumPy、PyArrow、SciPy、Matplotlib、pymatgen、MatterSim、LaTeX、pytest、SHA-256 manifest。

---

## 0. 本次必须遵守的边界

本文件是实施计划；计划写完后才开始新增实验和草图，本轮不直接改正文。

1. 合并主图 c 不画 D7，不用 D7 代表整个 PRIS，也不在 c 中展示单条法则贡献。
2. 每条法则具体贡献放 SI 的逐法则诊断图。
3. 合并主图 d 采用上下两行并共享同一个 x 轴：上面是 PRIS 法则与 CLscore 的关系，下面是合成性公式与同一 CLscore 分箱的关系。
4. 主图删除 Jang score。图例写出两个模型全名：CGCNN-PU OOB 和 MatterSim-embedding PU OOB。若要给平均 CLscore，先在各模型内转为 percentile/rank，再求算术平均，不能直接平均原始量纲。
5. L4 与 S_syn 是两条独立的用户选择路线，不画 L4 OR S_syn 或 L4 AND S_syn 的联合方案。
6. PU 低分结构只能称为 PU 模型选出的难合成代理队列，不能称为逐条实验证实的合成失败。
7. MatterSim basin–hull 只作为下游参照。旧 pilot 因约化化学式与整胞能量计量不一致而作废；修正后的 GPU 结果通过 QA 前不得引用，也不得标成 DFT E_hull。
8. 正文图片只使用英文。新图和 caption 使用 satisfaction、screening、damage detection、explicit violation 等统一术语，避免 reject、retention 这类内部词突然出现。

## 1. 主线和图的顺序

主图按以下问题链排列：

1. a–b 证明二元机制筛选能在昂贵计算前发现化学损伤，且各机制互补。
2. c 说明二元门为何需要连续公式，并给出两种实际前置筛选选择：L4 保守，S_syn 更激进但会牺牲实验结构保留。
3. d 检查这些机制/公式是否与独立 PU 模型分数相关。上、下两行使用完全相同的 CLscore percentile x 轴。
4. e 展示 S_stab 和 S_syn 分别解决低能量排序与实验记录排序，不能合并成一个无目标总分。
5. f 回到能量、声子和实验记录的矛盾，展示位点复杂度在相同能量/声子状态内仍有独立信号。

旧 Fig.5a 的 commitment plane/tie-rate 是 c 的第一层理由，旧 Fig.5e 的标准化系数是连续公式物理来源的补充理由；两者移到 SI，在正文用一句具体的平局统计和任务定义承接，不在主图重复。

## 2. 合并主图 panel contract

暂称为 Main Fig. M。先生成完整大图，最终编号优先沿用 Fig.4，以保留当前 Fig.4a/b 的重要正文证据；旧 Fig.5 面板转入 SI。

| panel | 内容 | 读者应得到的结论 | 明确移除 |
|---|---|---|---|
| a | 当前 Fig.4a 总体 benchmark。保留 440 个 parent、2,024 个 damaged structures、L4 的 83.0% experimental satisfaction 和 87.9% damage detection，以及距离 cutoff 的对照。 | PRIS 在进入昂贵计算前能发现距离检查遗漏的化学损伤。 | 不把 controlled damage 百分比和 PU 队列百分比混在一起。 |
| b | 当前 Fig.4b 按损伤类型的热图，保持 a→b 顺序。 | 五类机制互补，单一距离阈值有明确盲点。 | 不放逐法则贡献；不改成 D7 专题图。 |
| c | 重画 docs/figures/pris_pu_task1_binary.png 的 c 为 retention–screening frontier。x 为实验结构保留率，y 为 PU hard-negative 队列筛除率；比较 L4、S_syn 的两个独立工作点、0.5/0.7 Å distance baseline，以及修正并通过 QA 的 MatterSim basin–hull proxy 灰色参照。 | L4 是保守且能保留大部分实验结构的机制筛选；S_syn 可更激进地压缩难合成候选，但两项目标不可同时最优，用户按任务选择。 | D7 单独柱、D7 覆盖率、D1–D8 贡献、任何联合门。c 中不出现 D7 作为方法标签或主结论。 |
| d | 上行画 PRIS/L4 explicit-violation fraction 对 normalized CLscore；下行画 S_syn mean 对同一 CLscore percentile bins。两个模型的独立曲线和 rank-normalized consensus 都要能追溯到模型全名。 | PRIS 与独立 PU 排序有关系但不是同一黑箱分数；连续公式能显示二元门没有的梯度。 | Jang score、原始 A/B 直接平均、不同 x 分箱、合成成功概率表述。 |
| e | 两个小轴：S_stab 的 group-equal accuracy 随能量差阈值提高的曲线；S_syn 的 held-out confidence–accuracy 曲线，可加 DFT hull 灰色参照。 | 两个公式是同一 PRIS 描述符空间对两个不同科学目标的任务专用投影。 | 只列系数而没有性能；把 S_syn 当成直接合成概率。 |
| f | 恢复历史四格 D7/D1 面板：dynamically stable/on-hull、dynamically stable/metastable、imaginary modes/on-hull、imaginary modes/metastable；每格分 experimentally recorded 与 computed-only。 | 固定能量和声子状态后，实验记录仍偏向较简单的 site description，补足当前晶体争议的解释。 | 当前抽象 full ladder、旧的“91–99%”概括和过多 Wilson 区间。 |

### c 的冻结工作点

先读取现有冻结结果，不重新调阈：

- L4：实验结构保留约 80.69%，PU 队列筛除 51.88%。
- S_syn：约 80.69% 实验结构保留点的 PU 筛除约 83.68%；另显示 95% 实验结构保留点的约 20.34% 筛除，作为保守工作点。
- distance cutoff：PU 队列筛除 0%。
- MatterSim 参照：修正计量后重跑；若支持覆盖不足或 QA 不通过，只保留独立报告，不进入主图。

各点用不同 marker 表示独立用户选择，不用箭头暗示串联执行。点旁直接写结构数和百分比，避免只写抽象的内部术语。

## 3. SI 的内容安排

### SI-A/B：连续公式的两个补充理由

- SI-A 保留旧 Fig.5a commitment plane，并把 tie-rate 作为 inset；正文只引用“二元 PRIS 在同组成对中经常平局”，详细 choice fraction、accuracy 和 tie 数在 SI。
- SI-B 保留旧 Fig.5e standardized coefficients；分别标明 S_stab 和 S_syn 的监督目标、符号方向、样本数和标准化方式，说明两式不是一个总分的拆分。
- 旧 Fig.5b、Fig.5c wrong-hull-pair、full ladder，以及若未进入主图的 Fig.4c/d，全部迁入 SI 并更新内部引用。

### SI-d：逐法则贡献

新建独立逐法则图，编号暂定 SI-d（最终字母由 SI 排版统一）：

- x 轴按物理机制顺序列 D1–D8；需要展示组合时另列 L1/L2/L4，不能把组合写成单条法则。
- 上行显示实验结构和 PU hard-negative 结构的 per-law satisfaction/explicit-violation fraction，并给出可判断 n。
- 下行显示每条法则对 L4 队列缩短的 incremental coverage 或 leave-one-law-out 增量。由于一条结构可违反多条法则，使用覆盖率而不是互斥归因。
- missing/no-verdict 始终保留在队列，不计算为违反。

### SI-e/f：两个 PU 模型的 held-out 性能

新增两张同模板图，分别对应：

1. CGCNN-PU OOB scorer，来源为 CSAgent 的 03_train_bags.py 和 04_predict_clscore.py。
2. MatterSim-embedding PU OOB scorer，来源为 CSAgent 的 07_embed_pu_head.py。

每张图至少包含 ROC-AUC、PR-AUC（只有在标签协议允许时）、bag/seed 的均值和不确定度，以及 held-out positive 与未标注/PU 分数分布。先审计原始 split、OOB mask、checkpoint hash 和去重清单；只有真正独立 test 才标 test，否则明确标 validation。性能图不得从 full-pool 分数反推。

## 4. d 的统计口径

1. 使用全池共同评分的 8,108,676 条结构，主图不读入 Jang 文件。
2. 对两个模型分别转 percentile，构造 consensus percentile = mean(percentile_CGCNN, percentile_MatterSim)。原始分数范围和模型相关系数放 SI。
3. 上、下两行使用相同的十等分或二十等分边界、相同 n 和相同 bootstrap 分组；不能看似共用 x 轴而实际使用不同分母。
4. 上行 y 轴统一为 PRIS L4 explicit-violation fraction；下行统一为 mean standardized S_syn，并写清分数方向和 formula support。
5. 区间按来源/化学式或结构组 bootstrap，避免 811 万条结构造成伪精确。关系图只表达关联，不表达因果、概率校准或真实失败标签。

## 5. 得到确认后执行的任务

### Task 1：冻结数据契约

**Files:** Create experiments/pu_synthesizability_20260821/fig45_reorganization_contract.py; create outputs/20260822_fig45_reorganization_v1/FIGURE_CONTRACT.json; test tests/test_fig45_reorganization_contract.py。

先写测试，验证唯一 CIF 去重、三态分母、L4/S_syn 独立路线、主图 c 无 D7、主图 d 无 Jang。然后固定输入路径、版本、模型全名、公式和 checkpoint hash、随机种子，生成 manifest 和 panel labels。

### Task 2：修正并重跑 MatterSim

**Files:** additive runner under experiments/pu_synthesizability_20260821/; output outputs/20260822_pu_mattersim_basin_hull_gpu_corrected/。

修正完整 chemical formula 与整胞能量计量；在 GPU 队列重新运行实验/PU 分层 pilot；保留 Slurm、节点、checkpoint、输入输出 hash。完成 energy-zero、composition-scaling、duplicate-row、ABSTAIN 和 extreme-tail QA。通过后才允许把点放入 c。

### Task 3：绘制主图 a–c

**Files:** create or refactor experiments/pu_synthesizability_20260821/plot_merged_fig45.py; output outputs/20260822_fig45_reorganization_v1/main_fig_m.{png,pdf,svg}; test tests/test_merged_fig45_panels.py。

复用当前 Fig.4a/b 样式；把 task1 c 改成 frontier，删除所有 D7 文本/柱/归因，加入 L4、S_syn、distance 和合格 eHull proxy。运行图形冒烟、pdfinfo 和 rasterized visual audit。

### Task 4：绘制主图 d–f

**Files:** reuse/port src/fig5_ranking.py; modify the additive plot module; output main_fig_m_diagnostics.{png,pdf}; test tests/test_merged_fig45_diagnostics.py。

生成 d 的共享 x 双行，删除 Jang；生成 e 的两条任务公式曲线；恢复 f 的四格 D7/D1，full ladder 另存 SI。核对两行 bin edges、n 和 CI 完全一致。

### Task 5：绘制 SI 和 PU 模型性能

**Files:** create experiments/pu_synthesizability_20260821/plot_si_fig45_diagnostics.py and audit_pu_model_test_performance.py; output outputs/20260822_fig45_reorganization_v1/si/; tests test_si_fig45_diagnostics.py and test_pu_model_test_performance.py。

生成 SI-d 逐法则贡献图；审计两个模型的 held-out artifacts；为每个模型生成一张同模板性能图；迁移旧 Fig.5a/e、tie-rate、wrong-hull、full ladder 和未进主图的 validation panels。

### Task 6：图确认后才集成正文

**Deferred files:** tex/body.tex、tex/si_body.tex、src/fig5_ranking.py、paper/FACTS.md 及 figure cross-reference audit。

Results 固定为四个结论性小标题：PRIS detects chemically damaged structures before expensive calculations；Independent pre-DFT choices expose hard-to-synthesize candidates；Continuous PRIS formulas connect mechanisms to independent PU scores；PRIS explains residual disagreement among energy, phonons and experiments。先用 SI-A 的平局证据解释 c 中引入公式的第一理由，再用 c 的 PU trade-off 解释第二理由，d/e/f 顺次完成关联、性能和物理解释。

## 6. 验收标准

- 读者按 a→b→c→d→e→f 能读出“二元发现 → 连续公式动机 → PU 关系 → 任务性能 → 能量/声子/实验矛盾”。
- c 不出现 D7；逐法则贡献只在 SI-d。
- c 明确给出 L4 与 S_syn 两条独立路线及具体队列数，不能误读为联合筛选。
- d 上下两行共享同一 x 分箱，模型来源全名可读，Jang 完全消失。
- e 同时保留 S_stab 与 S_syn 的目标差异和 held-out 性能。
- 两个 PU 模型各有独立性能图，标签、split、n、指标和 hash 可追溯。
- MatterSim 参照通过计量和能量零点 QA；不通过则不进主图。
- missing/no-verdict 不隐式当作违反；主图无中文、无 reject、无未定义 retention。
- 正文、caption、SI 的子图引用顺序一致；PDF、事实清单、渲染、SHA-256 和 git diff --check 全部通过。

## 7. 本轮停止点

计划文件写入后立即进入 Task 1–5 的实验和草图阶段，但在所有数据审计完成前不改正文、caption 或 canonical PDF。正文集成作为最后一个独立阶段。
