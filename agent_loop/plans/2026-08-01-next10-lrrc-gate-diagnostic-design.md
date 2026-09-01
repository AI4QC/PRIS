# Next10 LRRC 固定门探索诊断设计

## 1. 目标与证据等级

next8 的 `AGREE995` 在 formula-selection 有 `+1.26325` 个百分点信号，但在已打开的
development gate 只剩 `+0.18425` 个百分点，paired CI 跨零且 comparator safety 失败。
next9 因此冻结了一个与同源 checkpoint 分歧正交的局部二阶响应 `LRRC-v0`。

本轮只回答一个窄问题：在**不重拟合 next8 阈值、不扫描 LRRC 参数**的前提下，LRRC 负曲率
能否在已暴露的 development gate 上给强 M5 基线增加有价值的拒绝信号。这个 gate 已被旧工作
打开，因此结果只能称为 `posthoc exploratory diagnostic`，不能称为独立验证、科学成功或新法则。

历史 test、OMat24、论文、旧报告、README 与 PREREG 全部保持关闭或不变。

## 2. 两阶段 opening 顺序

### 2.1 标签自由特征阶段

只读取并哈希：

1. next8 development feature manifest 与 feature parquet；
2. next8 threshold-role assignment；
3. 原始 x0 frame zip；
4. 固定 MatterSim 5M checkpoint；
5. next9 LRRC 与 next10 runner 源码。

只选 `threshold_role == development_gate` 的 sid，并验证该选择与 next8 feature 行一一对应。
这一阶段不得接受 label path，也不得导入协议评价代码。输出只含 LRRC 数值诊断、状态、输入哈希、
checkpoint 哈希和运行遥测。发布完成并重新哈希后，才允许评价阶段读取旧 development labels。

### 2.2 后验评价阶段

评价器先验证 sealed LRRC feature manifest、next8 frozen protocol、next8 development-gate metric
artifact 与所有输入哈希。然后必须逐项重现旧 M5/AGREE995 的 development-gate 决策和核心指标；
重现失败则在候选评价前 fail closed。

## 3. 固定 LRRC 计算

对每个支持结构只使用 MatterSim 5M：

- 1 次未扰动力批预测；
- 由 next9 `translation_projected_direction` 与 MIC `d_star` 构造固定方向和步长；
- 4 次扰动力批预测：`+h`、`-h`、`+h/2`、`-h/2`；
- 用一个固定顺序的 replay oracle 调用 next9 `evaluate_lrrc`，使标量实现成为唯一公式实现；
- 每个非驻点成功结构必须恰有 5 份力，驻点只使用 1 份力。

任何 checkpoint 变化、sid/frame 不一致、批输出错位、非有限力或发布前输入变化均整体 fail closed。
不把失败行悄悄丢弃；可归因的几何/数值状态显式写入 feature parquet。

## 4. 冻结候选目录

两条 next8 track 均沿用原 M5/AGREE995 最终阈值和严格 `score > threshold` 规则：

| formula | 固定决策 |
|---|---|
| `M5` | next8 M5 基线，仅用于精确重现 |
| `AGREE995` | next8 已选公式，仅作为弱信号参考 |
| `M5_LRRC_OR` | `M5 REJECT or LRRC_negative` |
| `M5_LRRC_QCRC` | 先 `M5_LRRC_OR`，再按 M5 score 应用 next9 Quota-CRC |
| `AGREE995_LRRC_QCRC` | 先 `AGREE995 REJECT or LRRC_negative`，再按 AGREE995 score 应用 Quota-CRC |

`LRRC OK/nonnegative` 与 `STATIONARY_FALLBACK` 沿用基础决策；LRRC 几何、模型或数值失败为
ABSTAIN。Quota-CRC 是最后一层，只能把 REJECT 改回 KEEP，ABSTAIN 不变。配额固定为
`ceil(sqrt(n))` 且边界并列全部 KEEP。

目录在打开 labels 前冻结，不根据负曲率比例、逐行标签或结果再增加公式。

## 5. 评价与停止规则

每个候选同时报告 primary/comparator：

- DFT savings 与 macro savings；
- exact/near/valuable retention；
- high-energy removal recall 与 reject precision；
- abstention、全拒组与 regret；
- 相对相应基础公式的 20,000 次 `rk` paired bootstrap。

复用门上的结果只作方向筛选：

- 若所有 LRRC 候选 savings 不增，或任何点估计增益都伴随 valuable recall 明显越界，停止该路线；
- 若至少一个 Quota-CRC 版本同时出现 savings 正增、valuable recall 点估计非劣、无全拒组，才把它
  带到 WBM 回顾性审计或 Alexandria 2025 时间外 cohort；
- 无论点估计多好，本轮都写 `scientific_improvement_claim=false`，不得打开历史 test 或 OMat24。

只有后续新物理 cohort 通过预先冻结的科学门，才另写独立成功报告并等待用户确认后修改论文或旧报告。

