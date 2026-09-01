# PREREG-L4: 合理性两轴同时 ≥0.80 的可解释法则集(新物理谓词扩展 L3)

日期冻结:2026-08-14(在矩阵增强特征计算完成前、任何谓词搜索前写下)。
状态:**冻结**;仅可追加"修订记录"。

## 0. 动机

L3 在 calibration 为 0.9171 / 0.7004:排除力低于 0.80。certified depth-3 树证明
(cost ratio 1.0)satisfaction 0.8895 / exclusion 0.8553 的点位在本特征空间内存在,
但树是损坏算子的指纹(LOPO 崩塌)且不可作一行式法则。本链条检验:把 PREREG-F3 §4
的经典 Born / 对称性特征作为**一行式物理谓词**加入 L3,能否在可解释形式下同时达到
两轴 ≥0.80。

## 1. 数据(先于本预注册存在)

`next20260801/law_{real,bad}.parquet`(audited isolated,无 lockbox:17,929 real /
12,202 perturbed,discovery 12,632+8,590,calibration 5,297+3,612)。增强特征由
`records_{real,bad}.parquet` 的 blob 记录 + `make_negatives.perturb` 确定性再生
(与 `phys_law._bad` 完全同一配方:`seed_of(sid)`、S1–S5 固定顺序、共享 rng、
`swapped_val`)计算,定义与 PREREG-F3 §4 逐字相同,写入新文件
`law_real_aug.parquet` / `law_bad_aug.parquet`。缺失值按论文约定视为满足。

## 2. 程序(冻结)

- **L4 = L3(五谓词原样,阈值不动)+ 至多 4 条新谓词。**
- 候选谓词 = (特征, 方向, 阈值, guard):特征 ∈ 增强特征 ∪ 既有 79 特征;
  阈值 ∈ 该特征在 discovery **real** 行上的分位网格
  {0.5,1,2,3,4,5,10,20,...,90,95,96,97,98,99,99.5}%;guard ∈ {无, fi>0.50, fi>0.55}。
- 贪心:每步在"discovery satisfaction ≥ 0.81"约束内选 pooled exclusion 增益最大的
  候选;增益 < +0.005 或已加 4 条即停。选定即冻结,记录 SHA-256。
- 同时报告(不入选择):从零开始的同规则数 beam 对照、certified 树 frontier 点。

## 3. 门(calibration,冻结的一次求值;calibration 复用按论文口径披露)

- **C1**:satisfaction ≥ 0.80
- **C2**:pooled exclusion ≥ 0.80
- **C3**:五类各自 exclusion ≥ 0.55(保持"无盲区"性质)
- **C4(披露)**:新谓词逐条 LOPO:删除其目标类重选后在该类上的排除力;
  连同各类分母一并报告
- 失败则如实报告达到的点位;禁止事后微调阈值再评。

## 4. 边界声明(冻结)

- 本链条针对合成扰动的排除力;LOPO 披露透明化"算子指纹"风险,不宣称跨算子普适。
- 新谓词必须可写成一行物理陈述(Born 同号排斥和、接触壁、弹性应变、Wyckoff 简约度);
  不可解释特征即使增益更高也不入选(此为模型类定义,非事后选择)。
- lockbox 不接触。
