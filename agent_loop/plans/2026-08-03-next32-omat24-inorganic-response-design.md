# NEXT32 OMat24 无机 DFT 响应预筛法则设计

日期：2026-08-03  
状态：实施前冻结；新增文件，不修改已有脚本、报告或论文

## 目标

NEXT32 直接补 NEXT31 尚未覆盖的无机晶体域：只给一个生成或理论预测的
未弛豫周期结构 `x0`，能否在任何 DFT 计算之前，高精度筛掉会产生严重 DFT
初始力或应力的结构。

法则执行时只允许元素、晶胞、坐标、周期边界、冻结元素表以及确定性的几何、
Voronoi、键价、静电和线性代数。禁止读取或调用 DFT 数值、弛豫结构、轨迹、
同组成候选、MatterSim/MLIP，以及任何学习得到的能量、力或应力代理。DFT
力与应力只在法则开发或预测冻结后的评估阶段作为标签。

NEXT32 即使通过，也只证明 DFT 单点严重响应预筛，不等同于形成能、凸包、
动力学稳定性、可合成性或替代 DFT。

## 数据选择

官方 [OMat24 数据卡](https://huggingface.co/datasets/facebook/OMAT24/blob/main/README.md)
说明，该数据包含无机非平衡结构的 DFT 总能、力和应力，采用 ASE-compatible
LMDB；论文为 [Barroso-Luque 等，OMat24](https://arxiv.org/abs/2410.12771)。
数据许可为 CC BY 4.0。

三条路线比较如下。

1. **OMat24 独立扰动来源，采用。** `rattled-relax` 只作暴露开发，
   `rattled-300/500/1000` 作三个确认来源。它们端点一致、材料域为无机体相，
   且归档可独立封存。
2. **Alexandria/MP 完整弛豫对，后续。** 能量下降更接近目标，但下载、轨迹
   身份和初终态完整性成本更高，不应与当前单点任务混在同一轮。
3. **NEXT23+NEXT31 路由级联，不采用为科学确认。** 它可形成工程预筛器，
   但没有增加无机 DFT 响应证据。

`rattled-relax` 验证源有 95,206 条记录，已为模式审计打开，因此整个来源只
能称作暴露开发源。它是抽样的非平衡帧，不保留每条完整弛豫轨迹；NEXT32
不得从不完整帧首尾构造能量下降主端点。

开发 cohort 从 `rattled-relax` 中按

```text
sha256("NEXT32-DEV-v1|" + parent_id + "|" + sid)
```

排序，每个 `parent_id` 最多保留一条，取前 4,096 条。选择只读身份与几何，
不使用 DFT 标签。

确认归档为尚未打开的 `rattled-300`、`rattled-500` 和 `rattled-1000`。每个
来源先用只投影几何的解析器读取 `sid`、`parent_id`、原子序数、坐标、晶胞和
PBC，跳过顶层 `energy/forces/stress` 数值。排除全部开发 `parent_id`，并在
三个确认源间累计排除先前已选 parent；每个来源按固定盐取 2,048 个唯一 parent，
合计 6,144 条。三个来源必须在标签开启前一起完成特征、Pauling 对照和预测
封存。

原始 LMDB 同时包含几何与标签，所以这仍是程序性隔离，清单必须记录
`physical_never_read_lockbox=false`；不得声称物理 never-read lockbox。

## 离线端点

对一个 DFT 单点记录定义

\[
F_{\max}=\max_i\|\mathbf F_i\|,\qquad
F_{\rm rms}=\sqrt{N^{-1}\sum_i\|\mathbf F_i\|^2},\qquad
S=\|\boldsymbol\sigma\|_2.
\]

严重响应标签固定为

\[
y_+=1\quad\Longleftrightarrow\quad
F_{\max}\ge1.0\ {\rm eV/\AA}
\;\lor\;F_{\rm rms}\ge0.40\ {\rm eV/\AA}
\;\lor\;S\ge0.030\ {\rm eV/\AA^3}.
\]

需要保护的低响应结构固定为同时满足

\[
F_{\max}\le0.50,qquad F_{\rm rms}\le0.20,qquad S\le0.015.
\]

这些阈值沿用 NEXT26–NEXT28 的严重响应量级，并在确认标签开启前冻结。能量
只作开封后的诊断，不进入公式、选择或主门槛。

## 解析候选

### 绝对周期接触项

为每个原子取冻结表列共价半径 (r_i)。枚举满足
(d_{ij\mathbf n}/(r_i+r_j)\le1.60) 的唯一周期原子对，不做分子 1–4 路径
排除。定义

\[
q_{ij\mathbf n}=\frac{d_{ij\mathbf n}}{r_i+r_j},\qquad
\delta_{ij\mathbf n}=\max(0,1-q_{ij\mathbf n}).
\]

只发布无量纲几何量：`cov_q01`、`cov_q05`、每原子 `q<0.85` 接触数、每原子
平方重叠量，以及位点重叠负荷的 95 分位和最大值。它们是几何拥挤度，不计算
势能、解析力或虚拟弛豫。

### 已验证解析项

复用但不修改 NEXT20–NEXT22：

- SIVR：`sivr_edge_mismatch_q95`、`sivr_site_imbalance_rms`、
  `sivr_cell_anisotropy`；
- normalized Madelung：`nm_total_reduced` 的弱结合方向与 `nm_site_spread`；
- SCBVE：`scbv_mismatch_q95`、`scbv_vector_asymmetry_rms`；
- `abs(log(scbv_global_scale / median_dev))` 作为闭式双侧尺度失配。

任一项只用开发中位数和 IQR 做 robust-z。风险方向在候选表中预先写死；
不拟合任意连续权重，不使用树、神经网络或核模型。

候选只包括单项以及机制明确的二项等权和，最多两项。拒绝比例只允许
`{0.025, 0.05, 0.075, 0.10, 0.15}`；缺失所需项时 fail-open，不拒绝。

## 开发晋级门槛

开发集上唯一公式必须同时满足单侧 95% Wilson 下界：

- 解析覆盖率 `>=0.95`；
- 低响应保护召回 `>=0.98`；
- 严重响应拒绝精度 `>=0.90`；
- DFT 节省率 `>=0.05`；
- 连续风险对严重响应的 ROC AUC `>=0.85`；
- 拒绝精度下界减严重响应总体基率的单侧 95% 上界 `>=0.20`。

若多个候选通过，依次按精度下界、节省下界、AUC、项数少、拒绝比例小和
字典序确定唯一公式。若无候选通过，停止；不下载或打开确认标签来寻找第二次
机会。

## 确认与 Pauling 比较

唯一公式晋级后，冻结公式、归一化常数、阈值、缺失策略、确认 ID 和评估协议。
三个确认来源先同时生成不可覆盖的预测和 Pauling 2–5 固定控制，再打开 DFT
力/应力。

总体确认沿用开发六门槛；另外每个来源必须满足：覆盖下界 `>=0.90`、低响应
保护召回下界 `>=0.95`、拒绝精度下界 `>=0.75`、节省下界 `>=0.02`、AUC
`>=0.75`。逐来源门防止某一个高阳性率来源掩盖迁移失败。

只有 NEXT32 通过总体及全部来源门槛，且 Pauling 2、3、4、5 与联合控制在同一
cohort、同一 fail-open 语义下都未通过总体主门槛，才允许写：

> 在 OMat24 三种无机扰动来源的严重 DFT 初始响应端点上，NEXT32 超越本项目
> 的固定 Pauling 2–5 操作性对照。

即使满足，也不得写成全面超越 Pauling、预测凸包稳定性或达到/超过 DFT 能量。

## 失败处理与产物

- 所有目录只发布一次，不覆盖；输入、代码、规则、预测和端点均保存 SHA-256。
- 标签开启前失败可修工程缺陷并重新生成新版本目录；标签开启后不得改公式、
  阈值、确认 cohort 或门槛。
- 开发失败保留扫描结果；确认失败保留冻结公式和失败证据，不重拟合。
- 最终只新增 NEXT32 代码、测试、外部数据产物和独立报告；用户确认前不修改
  论文、README、PREREG 或旧报告。

