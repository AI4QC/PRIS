# NEXT19 价键流可行性法则设计

**状态：** 已确认边界后的自主执行设计。该设计只创建新文件和新产物，不覆盖旧脚本、旧报告或论文。

## 1. 目标与硬边界

NEXT19 的目标是寻找一个可逐结构独立执行、可解释、无需 DFT 的晶体结构合理性法则。它必须在任何候选结构进入 DFT 之前给出 `KEEP`、`REJECT` 或 `ABSTAIN`，且执行时只允许读取：

- 原始、未松弛的晶格和原子坐标；
- 化学计量与元素表属性；
- 由原始结构解析得到的周期邻接、配位、Voronoi、键价、形式价态和 Ewald/Madelung 量；
- 固定的经验常数、公式和阈值。

法则执行路径禁止读取或计算 DFT 能量、力、应力、DFT 松弛结构、MatterSim/MLIP 能量、机器学习势、代理能量或同成分候选的相对能量。它也不能依赖同组其他结构，因此一个孤立候选必须能独立判定。DFT 标签只可在两个位置出现：历史开发集上的阈值选择，以及法则与阈值冻结后的外部盲评。

所有计算失败都采用 fail-open：返回 `ABSTAIN`，绝不因缺失价态、近邻或数值求解失败而拒绝结构。

## 2. 文献边界与新假设

Bond-valence sum 和 Global Instability Index 已经用逐位点键价和与形式价态的偏差衡量结构应变；Charge Distribution/CHARDI 已经按有效配位数在键上分配形式电荷。因此 NEXT19 不把“键价和”或“按配位分配电荷”本身声称为新法则。

待检验的新假设是：**真实或低 DFT 能量结构的几何连接图，应该能在不严重违背几何先验的前提下，承载一个同时满足所有阳离子供给与阴离子需求的全晶胞键价流；不合理结构则需要少数边异常过载或大规模重分配。**

这把 Pauling 第二法则从局部平均等式扩展为带周期多重边的全局守恒问题，并显式暴露局部平均可能掩盖的拓扑瓶颈。

## 3. 核心公式

先用仓库统一入口推断每个位点的形式价态 `z_i`。若整数与分数形式价态均无解，则使用预先冻结的 Pauling 电负性分区回退：每个位点取 `q_i = mean(chi)-chi_i`，把正、负两侧分别归一到 `+1` 与 `-1`，从而严格中和。运输主量对整体电荷尺度不敏感；该回退只确定符号和相对供需，不引入能量。单元素或电负性完全相同、无法形成两侧分区的结构仍然 `ABSTAIN`。只保留异号周期近邻边 `e=(c,a,image)`，其中 `c` 是阳离子、`a` 是阴离子。对每个阳离子的边计算固定几何权重

\[
w_{ca}=\omega_{ca}\exp\{-\alpha[(d_{ca}/d_{c,\min})-1]\},
\]

其中 `omega` 是周期 Voronoi/CrystalNN 的非负邻接权，`d_c,min` 是该阳离子的最近异号距离，`alpha` 来自预先声明的小目录。阳离子按几何先验发出的 Pauling 型键价为

\[
p_{ca}=z_c\frac{w_{ca}}{\sum_b w_{cb}}.
\]

令 `s_ca` 为最终键价流。第一个线性规划求最小过载 `kappa`：

\[
\begin{aligned}
\min_{s,\kappa}\quad & \kappa \\
\text{s.t.}\quad
& \sum_a s_{ca}=z_c, \\
& \sum_c s_{ca}=|z_a|, \\
& 0\le s_{ca}\le \kappa p_{ca},\quad \kappa\ge1.
\end{aligned}
\]

第二个线性规划在固定 `kappa*` 下最小化 `sum |s_ca-p_ca|`。输出三个主量：

- `vt_overload = kappa* - 1`：满足全局价态守恒需要的最大相对边过载；
- `vt_reallocation = sum|s-p|/(2 sum z_c)`：必须重路由的形式电荷比例；
- `vt_anion_mismatch_max`：未经优化的 Pauling/CHARDI 风格先验在最差阴离子上的相对失配。

同时输出覆盖、周期边数、图连通分量、求解状态和诊断量，但不输出能量。

## 4. 候选目录与对照

在读取开发标签前固定以下小目录，避免无限符号搜索：

- 价态策略：整数氧化态、受限分数氧化态、Pauling 电负性分区回退，按该固定顺序；
- 邻接：`CrystalNN` 权重与周期 Voronoi solid-angle 权重；
- 距离衰减：`alpha in {0, 2, 4, 6}`；
- 单量法则：三个主量分别超过固定阈值；
- 双量法则：`overload` 与 `reallocation` 的单调线性组合，最多两项；
- 共识法则：价键流违反且已有的独立短接触或 Madelung 符号守卫违反；
- 传统对照：Pauling P2-P5、旧 P9 Lewis mismatch、可计算时的 BVS/GII 或 CHARDI 风格失配。

阈值只能来自预先列出的数值网格或开发集分位点，且必须经 source-wise 验证。任何在外部 Alexandria 结果可见后添加的候选都属于下一轮，不能回写 NEXT19。

## 5. 数据流与隔离

1. **开发源 1：WBM。** 使用 `next14_wbm_acsc_holdout` 的 2,048 个 x0 geometry-only 结构提取特征；历史私有表只在独立 evaluator 中提供 stable、high-energy 和 hull 标签。
2. **开发源 2：ELEMENTA。** 使用 `next16_elementa_holdout_v2` 的 1,988 个 x0 geometry-only 结构作跨来源验证；标签只在独立 evaluator 中连接。
3. **外部源：Alexandria。** 使用已经隔离的 379 个 geometry-only 结构。NEXT19 法则和阈值冻结、预测和哈希封存之前，不允许提取 endpoint 字段。

特征构建器的 CLI 只接受 geometry archive、geometry manifest、无标签 metadata 和输出目录。它拒绝包含 energy、force、stress、relaxed、MatterSim 或 endpoint 字段的表。评价器是唯一允许读取历史开发标签的模块。Alexandria endpoint 提取与评价使用单独模块，并要求冻结协议哈希和外部预测哈希精确匹配。

## 6. 选择门与失败条件

WBM 只用于从固定候选目录选择阈值；ELEMENTA 必须在不重拟合下通过：

- coverage Wilson 下界至少 0.90；
- group-minimum recall Wilson 下界至少 0.95；
- valuable recall Wilson 下界至少 0.95；
- reject precision Wilson 下界至少 0.90，目标为 0.95；
- DFT savings Wilson 下界至少 0.10；
- 不得完整拒绝任何同成分组；
- 相对 Pauling 的关键安全指标必须提高，高能拒绝不能明显退化。

只有通过上述门的候选才冻结。Alexandria 外部评价使用完全相同的规则、阈值、价态策略和近邻参数。若没有候选通过，NEXT19 是严格负结果：保留代码、聚合结果和机制诊断，写独立负结果报告，不打开或反复使用外部标签调参。

## 7. 输出与论文边界

代码使用 `src/next19_*`，测试使用 `tests/test_next19_*`，聚合产物写入新的 `outputs/20260802_next19_*`，带标识符的特征与连接表写到 `$PRIS_ARCHIVE/next19_*`。成功或失败后只新增 `reports/2026-08-02-next19-valence-transport-law.md`。未经用户确认，不修改 `paper/`、`notes/`、`tex/`、`README.md`、`PREREG.md` 或任何已有报告。
