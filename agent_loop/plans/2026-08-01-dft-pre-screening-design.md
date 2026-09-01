# DFT 前晶体结构筛选：未弛豫输入确认方案（冻结稿）

日期：2026-08-01  
状态：在任何 WBM 特征—标签联合搜索之前冻结  
范围：只新增脚本、测试、输出和独立报告；不修改已有脚本、已有结果、论文、README 或正式报告

## 1. 目标与不可变边界

目标不是继续提高“DFT 已弛豫结构上的配对胜率”，而是寻找只读取原始候选结构
`x0` 就能安全减少完整 DFT 弛豫次数的可解释准则。

准则只允许读取：

- 元素和化学计量；
- `x0` 的晶胞和坐标；
- 在看测试结果前冻结的元素表、键价参数和公式常数。

禁止读取：DFT 终态坐标、终态体积、终态邻居图、DFT 能量、力、应力、离子步数，
以及由这些量派生的任何特征。氧化态无唯一可靠解、参数缺失或邻居计算失败时必须
`ABSTAIN`，并送入 DFT；不得把 unknown 当成正确拒绝。

输出为三态：`KEEP / REJECT / ABSTAIN`。部署统计中 `ABSTAIN` 与 `KEEP` 一样消耗
DFT 预算。

## 2. 数据角色

### 2.1 Discovery（只能提出候选，不能确认部署能力）

- 现有 `polymorph_rank2.parquet`、`elem_rank.parquet`、`alex_rank.parquet`、
  `lemat_rank.parquet`：特征主要来自 DFT 弛豫终态。
- `real_rank.parquet`：实验结构安全审计。
- `synth_rank.parquet`：正—未标注的合成富集诊断，不能把无 ICSD 当真阴性。

这些数据已经反复查看，全部视作 discovery。

### 2.2 Initial-to-final 迁移诊断

ELEMENTA core 原始包含 38,808,603 帧、2,028,008 条连续 DFT 弛豫轨迹。
只用每条轨迹首帧计算规则，用末帧能量、收敛状态和首末结构漂移作标签。该数据用于：

- 检查终态发现的候选在 `x0` 上是否仍成立；
- 同组成能量 regret；
- 规则从首帧到末帧的分数/排序翻转率。

ELEMENTA 与本库既有搜索同源，因此不能单独承担最终外部确认。

### 2.3 未弛豫外部基准

本地 Matbench Discovery/WBM 包含 256,963 对初始和 DFT 弛豫结构，官方
`unique_prototype` 子集 215,488 个。输入固定为 initial extxyz；标签固定为：

```text
stable := e_above_hull_mp2020_corrected_ppd_mp <= 0
```

同时报告 `E_hull <= 0.05` 和 `<= 0.10 eV/atom` 的灵敏度分析，但不能用它们替换
主标签。`site_stats_fingerprint_init_final_norm_diff` 只作为结构存活结果，不能进入特征。

WBM 已被读取过总体行数和总体阳性数，因此本轮不能声称“从未读取的锁箱”；但在本稿
冻结后，不允许依据 WBM 特征—标签联合结果改变候选族、方向、主标签或主门槛。

### 2.4 最终确认层（本轮之外）

若 WBM 和 ELEMENTA 都通过，再冻结公式，对至少两个未参与搜索的生成器产生的新候选
做统一工作流的前瞻 DFT。只有这一层可以支持强部署结论。

## 3. 冻结切分和独立单位

- 所有切分以 reduced composition 为最小单位；相同组成不得跨 calibration/test。
- 同一 prototype、同一生成谱系和同一 DFT 轨迹的不同帧必须留在同一侧。
- WBM 用 `sha256(reduced_formula)` 固定映射：前 20% composition 为外层
  calibration，后 80% 为 test。外层 calibration 再用独立的
  `sha256("stage:" + reduced_formula)` 奇偶位分成 `formula_selection` 和
  `threshold_calibration`；前者只选公式，后者只定拒绝阈值和风险上界。
- 三部分先物理写出独立标签、`x0` 特征文件和 SHA-256 manifest；test 只执行一次并写
  opening log。
- 置信区间按 composition 整簇 bootstrap；不得把结构对当独立样本。

## 4. 冻结候选法则族

搜索只允许下列具有固定物理方向的非负组合，不允许逐元素自由拟合：

\[
\Phi(x;q,\lambda)=
a G_q(\lambda x)^2+b R_{\rm rep}(\lambda x;q)
+c \widehat E_{\rm Ewald}(\lambda x;q)+d P_{\rm pack}(\lambda x)
+\eta |\log\lambda|.
\]

- `G_q`：按原子平均的 bond-valence-sum mismatch；越小越合理。
- `R_rep`：短程重叠的 softplus/Born 型排斥；越小越合理。
- `Ewald`：电荷平衡氧化态下的每原子静电代理；越低越合理。
- `P_pack`：过密和过疏的双侧 packing 惩罚；越小越合理。
- `q`：有限的电荷平衡氧化态组合；无可靠解则 abstain。
- `lambda`：仅在 `[0.8, 1.2]` 的预注册网格上取尺度包络，并惩罚偏离 1。

复杂度按顺序增加：

1. static 单项规则；
2. scale-envelope 单项规则；
3. 最多三个物理项的非负稀疏组合；
4. 若前三层有稳定增益，再评估解析伪力和固定少步代理势预弛豫。

系数、排斥指数、尺度网格和拒绝阈值只能从预注册有限网格选择；报告完整 Pareto 前沿，
不得只保存最佳行。

## 5. 基线

在完全相同 coverage/abstention 记账下比较：

1. no-filter 和随机拒绝；
2. 原始 Pauling 规则及其最佳单项；
3. `bl_min`/短键、packing、每原子体积；
4. 传统 BVS/GII；
5. Ewald 单项；
6. 当前冻结的稀疏新公式。

官方 Matbench Discovery 的 MP-only 与宽数据 MLIP 结果只作性能参照，除非本地实际运行，
不得写成同环境复现实验。

## 6. 主指标

WBM 主任务报告：

- stable recall 与 false-negative rate；
- precision、F1、DAF（precision / prevalence）；
- 实际拒绝比例，即理论 DFT 数量节省；
- coverage、abstention、特征失败数量；
- top-k precision（k = 10,000 和固定预算分数）；
- 按主要化学族、元素 OOD、prototype batch 的最差分层值；
- composition-cluster bootstrap 95% CI。

ELEMENTA 同组成任务报告：

- group-min retention；
- 近最优候选（25/50 meV/atom）false rejection；
- 最低保留能量的 median/p90/p95 regret；
- 高能候选（>= 0.20 eV/atom）拒绝率；
- 首末帧规则翻转率和排序翻转率；
- 候选数量与离子步近似加权的 DFT 节省。

配对胜率只保留为次要诊断。

## 7. 预注册验收门

### 有效预筛（必须全部满足）

- stable recall 的单侧 95% 下界 >= 0.99；
- 拒绝比例的单侧 95% 下界 >= 0.30；
- ELEMENTA group-min retention 的单侧 95% 下界 >= 0.95；
- ELEMENTA p95 energy regret <= 0.05 eV/atom；
- 在相同漏筛风险下，DFT 节省严格超过 Pauling、短键、packing、BVS/GII、Ewald；
- 每个预注册独立来源都通过，不能用 pooled 平均掩盖来源失败；
- unknown 全部计入 abstention。

### 接近 DFT 的高门（必须全部满足）

- 节省至少 50% 完整 DFT 时，stable recall 下界 >= 0.99；
- group-min retention 下界 >= 0.99；
- p95 regret <= 0.025 eV/atom；
- 每个主要化学族/来源的 retention 点估计 >= 0.95。

若未过门，结论必须是“探索性物理代理/负结果”，不能缩小成功定义。

## 8. 证据边界

以完整 DFT 弛豫作为标签时，不能声称在同一标签上“超过 DFT”。可检验的合法表述是：

- 在极低漏筛风险下接近 DFT 的筛选决策并减少完整弛豫次数；
- 超过未弛豫结构的粗 DFT single-point 或现有非 DFT 基线；
- 未来以 r2SCAN、声子、实验存活/合成为外层标签时，检验是否超过 PBE 决策。

## 9. 产物隔离

新增代码使用 `src/next6_*`，测试使用 `tests/test_next6_*`，输出使用独立的
`outputs/20260801_dft_prescreen/`，最终先写独立报告。旧脚本、旧输出和论文均保持原样。
