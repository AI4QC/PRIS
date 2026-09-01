# Next9 LRRC-v0 与组级配额设计

## 1. 目标与证据边界

本轮从 next8 的后验结果出发，但不回到已经看过的 development labels 调参。next8 的
`AGREE995` 在 formula-selection 上相对 M5 有信号，在独立 development gate 上却只剩约
`+0.18` 个百分点，置信区间跨零，并且 valuable-recall 安全门失败。其主要问题不是阈值写法，
而是 M1/5M 同属 MatterSim 家族，分歧没有提供足够正交信息。

next9 采用两个严格分开的组件：

1. `Quota-CRC` 是组级风险策略，只改变拒绝预算的分配，不称为新物理法则；
2. `LRRC-v0`（local restoring-response criterion）加入局部势能面二阶响应，是唯一可能新增
   物理信息的候选。

本轮只允许 synthetic 数学、数值和接口验证。没有新的完整 `x0 -> DFT endpoint` cohort 前，
不得把任何结果写成科学性能提升、超过 Pauling rules 或接近/超过 DFT。

## 2. 为什么不直接继续扩大同源委员会

next8 已证明 M1/5M gap 在开发各 stage 高度相关；极端 disagreement gate 的主要效果是增加
ABSTAIN，而不是稳定增加高能候选识别。继续添加同家族 checkpoint、扫描 disagreement
分位数或回调阈值，都会在同一开发证据上扩大研究者自由度，且不解决共同训练偏差。

真正的异质委员会至少需要三个训练数据或架构显著不同的模型，并在全新 DFT calibration 组上
构造 conformal lower bound。本轮只保留该方向的未来接口，不把 M1/5M 冒充异质模型。

## 3. Quota-CRC：安全策略，不是新法则

对支持且分数有限的 composition group `G`，令：

\[
k_G=\lceil\sqrt{|G|}\rceil,\qquad q_G=s_{(k_G)}.
\]

其中 `s_i` 是预先冻结的 M5 或委员会风险分数，`s_(k_G)` 为第 `k_G` 小的分数。候选只有在：

\[
s_i>\tau \quad\text{且}\quad s_i>q_G
\]

时才可 REJECT。配额边界并列全部 KEEP；不支持或分数非有限的行保持 ABSTAIN。

同一阈值下有：

\[
R_{quota}(\tau)\subseteq R_{base}(\tau),
\]

所以它不可能创造额外 savings。它的唯一合理用途，是在未来全新 calibration 上允许更激进的
阈值，同时防止小组被过度拒绝。未来实证必须同时报告 fixed-threshold 与 refit-threshold
ablation；若增益只来自后者，只能称为 policy gain。

## 4. LRRC-v0：局部恢复响应

### 4.1 方向

对固定胞结构的 M5 原子力去除整体平移：

\[
f'_i=f_i-\frac{1}{N}\sum_j f_j,
\qquad
u_i=\frac{f'_i}{\sqrt{N^{-1}\sum_j\|f'_j\|^2}}.
\]

这样 `mean(u)=0` 且 `mean(||u_i||^2)=1`。若投影后的 RMS 小于冻结的纯数值下限
`1e-12 eV/angstrom`，LRRC 不构造方向，标记 `STATIONARY_FALLBACK` 并回退基础规则；这也明确
暴露了精确驻点鞍的已知盲区。

### 4.2 无标签步长

令 `d_star` 为使用最小镜像距离计算的逐原子最近邻距离中位数，固定：

\[
h=2^{-8}d_\star.
\]

`2^-8` 是冻结的数值离散定义，不从任何旧标签或真实 checkpoint 扫描。`N<2`、无有限正最近邻
距离或非法周期胞均不支持。

### 4.3 两尺度方向曲率

在 `h` 和 `h/2` 上各做中心差分：

\[
\kappa_h=-\frac1N\sum_i u_i\cdot
\frac{F_i(x+h u)-F_i(x-h u)}{2h}.
\]

再定义：

\[
\kappa_R=\frac{4\kappa_{h/2}-\kappa_h}{3},
\qquad
e_{num}=\frac{|\kappa_{h/2}-\kappa_h|}{3},
\qquad
U_{num}=\kappa_R+e_{num}.
\]

`U_num` 只是确定性的两尺度保守代理，不宣称统计置信上界或严格余项上界。LRRC 信号要求
`kappa_h < 0`、`kappa_h2 < 0` 且 `U_num < 0`，即两个尺度符号一致并且 Richardson 代理仍为负。

### 4.4 决策组合

未来候选决策为：

\[
REJECT \iff g_{5M}>\tau_E \quad\lor\quad LRRC\_negative.
\]

LRRC 成功且非负时沿用基础决策；LRRC force oracle 失败、产生非有限值或几何不支持时 ABSTAIN。
`STATIONARY_FALLBACK` 是已知、可诊断的无新增信号状态，沿用基础决策。Quota-CRC 必须作为
最后一层应用，并可把配额内的 REJECT 改回 KEEP；ABSTAIN 永不改写。

## 5. Synthetic 验证矩阵

必须验证：

- 正定二次势给出正曲率且不新增拒绝；
- 倒置二次势在非零位置给出两个尺度一致的负曲率；
- 平移、刚体旋转、原子置换与周期回卷不改变结果；
- 二次势上 `kappa_h` 与 `kappa_h2` 收敛到解析值；
- force oracle 抛错、形状错误或非有限时 fail open 为 ABSTAIN；
- LRRC 的 OR 组合能新增 REJECT，而不是通过增加 ABSTAIN 制造名义 savings；
- 精确 `F=0` 鞍点明确回退基础规则；
- Quota-CRC 固定 `ceil(sqrt(n))`、边界并列 KEEP、ABSTAIN 不变；
- manifest 保存公式、常数、执行源哈希和 synthetic case 结果，不包含真实标签或受保护标识符。

## 6. 数据路线

本地审计没有发现同时满足“标签未打开、物理隔离、完整候选组、x0/DFT endpoint 配对”的
cohort。ELEMENTA Spin 的旧提取还把 `(material, structure, spin)` 错按 material 合并，只留下
78,027/207,185 条轨迹，不能使用。

外部首选是 2025 Alexandria 扩展中的 `m3gnet/rng` 控制 cohort。官方论文报告 29,671 个成功
DFT 终点，来源时间晚于 MatterSim 2024，且是随机抽取而非经稳定性模型筛选。它仍有两个边界：
输入是 M3GNet 预弛豫结构而非最原始生成态；MatterSim 训练明细不公开，训练重叠只能记为未知。
在确认 source tag、候选组完整性和分片映射前，不下载或解析全量约百 GB relaxation paths。

参考：

- [Alexandria 2025 expansion](https://arxiv.org/html/2512.09169v2)
- [Alexandria geometry-optimization paths](https://alexandria.icams.rub.de/data/geo_opt_paths/2025.07.02/pbe/)
- [Conformal Risk Control](https://research.google/pubs/conformal-risk-control/)

## 7. 成功门

Synthetic 通过只表示工程和代数契约成立。真正科学成功必须在预先物理拆分的新 cohort 上同时满足：

1. 相对冻结基础规则的 DFT savings 增量达到预注册门，并且 paired 95% CI 下界为正；
2. valuable recall 非劣、exact/protected group retention 单侧 95% 下界不低于 0.95；
3. 无全拒组，abstention 增量在预注册上限内；
4. fixed-threshold、refit-threshold、LRRC-only 与 quota-only ablation 全部报告；
5. 外层 gate 只开一次，失败后不追加调参。

在上述门通过前，不修改现有报告、论文、README 或预注册文件。
