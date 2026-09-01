# MatterSim 少步预弛豫筛选设计（冻结稿）

日期：2026-08-01  
状态：在生成任何 next7 测试分区轨迹前冻结  
范围：只新增 `next7` 代码、测试、输出和独立报告；保留全部既有脚本、结果、报告、论文与 README

## 1. 问题与证据边界

现有 MatterSim x0 同组成能量差是本库最强的 DFT 前筛基线，但在历史 ELEMENTA
测试集上尚未同时达到 30% DFT 数量节省和 95% exact/near-min retention 下界。
本阶段检验一个有限、可证伪的新假设：x0 的局部几何误差使单点能量排序失真，固定少量
力驱动坐标更新可在很小成本下改善排序。

当前 ELEMENTA 四个分区都已属于 discovery。新结果即使改善，也只能写成
retrospective mechanism evidence；不能重新切分后称 blind/confirmatory。真正确认必须在
公式、阈值、代码、checkpoint 和决策全部封存后，用物理隔离的新生成器批次或从未进入
既有队列的新 composition holdout 完成，并为所有候选计算标签。

## 2. 三种路线与选择

1. `x0 energy + force diagnostics`：不移动原子，成本最低，是必报控制组。
2. `fixed-cell few-step FIRE`：固定晶胞，只更新原子位置，推荐为本阶段主实验。
3. `cell + position relaxation`：更接近完整弛豫，但引入 stress、体积塌缩和模型外推；
   本阶段不进入候选池，只有路线 2 出现清晰增益后才能另行预注册。

采用路线 2。使用 MatterSim 1.2.3 的同一 5M checkpoint，仅用 species、cell、PBC 和
x0 coordinates。批量推理与每个结构独立的 ASE FIRE 状态结合，避免逐结构 calculator
的吞吐瓶颈；不修改安装包中的 `BatchRelaxer`。

## 3. 冻结优化轨迹

```text
cell=fixed
snapshots={0,2,4,8}
optimizer=ASE FIRE
dt=0.05
dtmax=0.20
maxstep=0.05 Angstrom
Nmin=5
finc=1.1
fdec=0.5
astart=0.1
fa=0.99
early_stop=false
```

“step”严格定义为一次 FIRE 坐标更新。得到 x8 需要 x0 到 x8 共 9 次能量/力评估和
8 次坐标更新。每个结构从同一个 x0 出发并沿一条连续轨迹取快照，不把独立的 2、4、8
步任务串成 2、6、14 步。晶胞不变，坐标可按周期边界回卷但不改变 minimum-image 位移。

## 4. 标签自由特征与有限公式

每个快照只保存：总能量、每原子能量、`Fmax`、`Frms`、stress Frobenius norm、最大
主应力、相对 x0 的 minimum-image RMS/max displacement、最短成对距离、单步能量变化、
运行错误、force-evaluation 数、GPU 时间和峰值显存。raw extxyz 中的 DFT energy、forces、
stress 和离子终态字段必须在构造 ASE `Atoms` 时清除。

只允许以下六个无拟合权重的同组成坏度分数：

```text
S0     = gap(E0/N)
S2     = gap(E2/N)
S4     = gap(E4/N)
S8     = gap(E8/N)
Sbest4 = gap(min(E0,E2,E4)/N)
Sbest8 = gap(min(E0,E2,E4,E8)/N)
```

`gap(x_i)=x_i-min_j(x_j)`，`j` 只遍历同组成且受支持的候选。不得扫描连续 step、连续
线性权重或把 candidate suffix、sid、rk 顺序作为分数。`rk` 只用于 x0 化学计量相同的
候选成组，不作为模型输入。

## 5. Fail-open 支持域

以下任一情况必须 `ABSTAIN` 并送入 DFT，不能自动 REJECT：

- 非真正 `ionic_step=0`，解析/模型/优化失败，或任一保存量非有限；
- 同组成受支持候选少于 2；
- 任一步最短成对距离相对 x0 降到危险短接触区；
- 单次坐标更新超过强制 `0.05 Angstrom`，或 x8 累计最大位移超过 `0.40 Angstrom`；
- 相邻保存快照能量上升超过 `0.02 eV/atom`；
- 任一保存快照 `Fmax > 20 eV/Angstrom`。

短接触的工程检查固定为：若 x0 的现有 `geom_min_pair_ratio < 0.45`，或少步轨迹的最短
绝对距离非有限/不为正，则弃权。力、位移和短接触只控制支持域，不进入加权评分。

## 6. 分区与选择

先只生成 `search_calibration`、`formula_selection` 和 `threshold_calibration` 的轨迹：

1. `search_calibration` 只为六个公式建立 conformal 阈值；
2. `formula_selection` 在固定安全门后，以 DFT 数量节省、成本和较小步数选一个公式；
3. `threshold_calibration` 为选中公式冻结最终阈值；
4. 写出包含代码/输入/checkpoint SHA-256 的 `FROZEN_PROTOCOL.json`；
5. 只有冻结文件存在后，才允许生成历史 `test` 轨迹和读取其标签。

主轨固定为 `protected=valuable (delta_E <= 0.05 eV/atom)`、`within_group=max`、
`alpha=0.01`。副轨固定为历史可比的 `protected=near_min (1 meV/atom)`、
`within_group=min`、`alpha=0.035`，只能解释为机制比较，不能作为部署结论。

## 7. 改善门槛

在当前 ELEMENTA 上，只在 paired composition bootstrap 同时满足下列条件时称“可信的
回顾性改善”：相对 step0 savings 增加至少 3 个百分点且 95% CI 下界大于 0；valuable
recall 差的 95% CI 下界不低于 -0.005；abstention 增量不超过 1 个百分点。否则只报告
方向性或负结果。

真正成功仍要求物理隔离的新确认批次同时达到：valuable/stable recall 单侧 95% 下界
至少 0.99、exact/near-min retention 下界至少 0.99、valuable-all group retention 下界
至少 0.95、DFT savings 下界至少 0.30、p95 regret 不超过 0.025 eV/atom，并在每个生成器
单独通过。MLIP GPU 成本还必须低于节省 DFT 成本的 10%。

## 8. 产物隔离

新增代码使用 `src/next7_*`，测试使用 `tests/test_next7_*`，输出使用
`outputs/20260801_mattersim_fewstep/`，报告使用新的 `reports/2026-08-01-*.md`。不修改
任何 `next6` 文件、既有报告、论文或规范性文档。
