# PREREG-F3: 无 DFT 结构分数在可合成性排序任务上正面挑战 DFT E_hull

日期冻结:2026-08-14(在任何 dev 拟合、任何 holdout 接触之前写下)
状态:**冻结**。本文档写下后,除追加"修订记录"小节外不得修改。

## 0. 动机与先验证据(全部来自已发表/已归档数字,非新计算)

论文的同组成可合成性排序任务(S13 协议)上,已发表基线为:

| 判据 | commit | group-equal acc | acc \| e_hull 错误对 |
|---|---:|---:|---:|
| DFT E_hull | 0.9930 | **0.7501** | 0.0(定义) |
| vol_per_atom | 0.9863 | 0.6435 | 0.5463 |
| Shannon packing | 0.9879 | 0.6281 | 0.5630 |
| bl_min (ρc) | 1.0000 | 0.5688 | 0.5959 |
| Pauling 5 | 0.2230 | 0.6553 | 0.5675 |

关键事实:(a) F2 是按 e_hull 标签拟合后迁移到本任务的,**从未有分数直接按
synth 标签拟合**;(b) 结构判据在 e_hull 排错的对上准确率 0.55–0.60,存在
真实互补信号;(c) R10 发现已合成条目更对称(中位空间群 87 vs 62),而对称性
特征不在现有特征表中。

## 1. 数据(已冻结,先于本预注册存在)

`features/synth_rank.parquet`(6,878 行,85 特征)。任务群体:synth 两类
都有的组成,复现论文口径:**1,508 组 / 6,758 结构 / 18,920 对**(已核对)。
任何新行发表前必须先通过 `rank_rulesets.py` 的四行 4 位小数复现检查。

## 2. 分割(冻结)

```python
dev = zlib.crc32(f"{rk}|synthsplit20260814".encode()) % 10 < 6
```

- dev:919 组 / 4,187 结构 / 14,563 对(SiO2 落在 dev)
- holdout:589 组 / 2,571 结构 / 4,357 对

holdout 组成在拟合、选特征、调参、看单特征准确率的任何一步都不得读取。
holdout 允许接触次数:**1**(第 6 节的一次性求值)。

## 3. 特征许可(冻结)

可用:synth_rank.parquet 的浮点特征 + 新增对称性/经典能特征(第 4 节),
排除以下各项:

- 标签与身份:`synth`, `e_hull`, `mp_id`, `rk`
- 尺寸混杂(广延量):`nsites`, `n_sites`, `p2_n_bad_020`, `p2_sum_dev`,
  `p3_n_pairs`, `p3_n_face`, `p3_n_edge`, `p4_n_viol`
- 总量形式的 Ewald 分解:`ewald_real`, `ewald_recip`, `ewald_point`
  (保留 `ewald_per_atom`)
- 表覆盖伪量:`bv_param_cov`

任何 DFT 量、机器学习势、松弛轨迹一律禁止(执行边界与论文一致)。
组成层特征在同组成任务上恒为平局,无害但不入选。

## 4. 新增特征(冻结定义;计算不看标签)

对 6,878 个结构从 MP 快照重建,写入**新文件** `synth_rank_aug.parquet`:

对称性(SpacegroupAnalyzer/spglib,symprec ∈ {0.01, 0.1}):
- `sg_num_001`, `sg_num_01`:国际空间群号
- `csys_rank_001`:晶系序(三斜1…立方7)
- `wyckoff_econ_001`, `wyckoff_econ_01`:不等价位点数 / 位点数(Pauling 第五规则的结构化连续量)

经典 Born 项(电荷=composition-only 整数平衡价态,半径=`phys_law.shannon`
按 (元素,氧化态,CrystalNN CN);对所有 d < 1.25×r_sum 的近邻对):
- `rep9_ca_pa`, `rep9_aa_pa`, `rep9_cc_pa`:Σ(r_sum/d)^9 / N,按电荷符号对分类
- `repexp_ca_pa`, `repexp_aa_pa`, `repexp_cc_pa`:Σ exp((r_sum−d)/0.345 Å) / N
- `strain2_ca_pa`:Σ((d−r_sum)/r_sum)² / N(异号接触弹性应变)
- `density`:质量密度 g/cm³

## 5. 模型类与选择程序(冻结)

- 缺失值:dev 中位数插补;标准化:dev 均值/方差。两者冻结后用于 holdout。
- 配对样本:组内 (synth=1, synth=0) 全配对,X = f₁ − f₀;反对称 logistic
  (无截距),每对权重 1/组内对数(组等权,与评估度量一致)。
- **F3(交付物)**:贪心前向选择,评分 = dev 上 5 折 GroupKFold(按组成)
  验证折的 group-equal accuracy(用 `rank_rulesets.evaluate` 同一实现);
  停止条件:最优增益 < 0.002 或达 8 项。选定后在全 dev 重拟合,系数冻结,
  记录 SHA-256。
- **F3-full(仅作上限参考)**:全部许可特征,L2 强度由同一 CV 选取。
  主张只挂在稀疏 F3 上。
- 允许在 dev 内做任意诊断;dev 上的一切数字均标 development。

## 6. 一次性 holdout 求值与成功门(冻结)

冻结的 F3 对 holdout 各组求值一次。配对比较均在两判据皆 commit 的组上。
聚类自助:重采样 holdout 组成组,B = 2000,seed 20260728,对**差值**取
95% 百分位区间。

- **G1(主门)**:Δ = acc(F3) − acc(E_hull),95% 下界 > 0
- **G2**:F3 的 commit ≥ 0.99
- **G3**:top-1 lift(F3)≥ top-1 lift(E_hull)(点估计)
- **G4**:acc(F3) 同时高于 vol_per_atom、sh_pack、bl_min 三个单量在
  holdout 的准确率(点估计;否则组合无意义)
- **G5**:汇报最大组占比与组数;不以任何单组主导的口径另立主张

结果分级(冻结措辞):
- G1–G4 全过:"frozen DFT-free score surpasses DFT E_hull on held-out
  compositions of this task"(仍须写明标签混杂:合成史偏好)
- G1 未过但 Δ 点估计 ≥ 0 且 G2–G4 过:"parity with DFT within CI"——
  以负/中性结果如实并入论文,不得再调参重试
- 其余:负结果,写入反驳账本

holdout 求值后,任何进一步修改都只能作为**新的预注册链条**在新的数据上确认。

## 7. 边界声明(冻结)

- 标签是"MP 条目带 ICSD 编号",受合成史选择偏好污染;正类污染压低上限。
- 结构为 MP 的 DFT 松弛几何;判据执行本身不调用 DFT,与论文 T3 框架一致。
- 本任务是同组成相对排序,不是绝对合理性判断;与 D1–D6 语义不同,不得混排。

(SHA-256 of this file at freeze time is recorded in
`outputs/20260814_f3_synth/PREREG_SHA256`.)

## 修订记录

**修订1(2026-08-14,holdout 接触前;dev 侧发现)**:F2R 链条的贪心在
committed-only accuracy 下选出低覆盖高准确率的退化解(cn_cat_max,cov 0.458)
——正是论文 §sec:pauling 诊断的"弃权换准确率"模式。修订:前向选择的 CV
评分对 fold 平均 coverage < 0.99 的候选集判 −1(硬拒绝)。G2 门本已要求
最终 coverage ≥ 0.99;此修订把同一约束前移到选择程序,防止把唯一一次
holdout 接触浪费在必然失败的退化解上。同一修订同时应用于 PREREG-F2R。
holdout 与 calibration 在本修订时刻均未被接触(HOLDOUT_CONTACT.log 与
CALIB_CONTACT.log 均不存在)。

**修订2(2026-08-14,holdout 接触前)**:增加次级冻结模型 **F3H(混合)**:
在 dev 上以反对称 logistic 拟合两特征 [s_F3, e_hull](s_F3 为冻结 F3 分数;
标准化统计取自 dev),系数冻结后与 F3 在**同一次** holdout 接触中一并求值。
次级门 **G6**:acc(F3H) − acc(E_hull) 的聚类自助 95% 下界 > 0。G6 检验的
主张与 G1 不同且独立:"结构化学携带 DFT 稳定性之外的可合成性信息"。F3H
含 DFT 量,不得表述为无 DFT 判据;G1 的表述不因 G6 改变。holdout 在本修订
时刻未被接触(HOLDOUT_CONTACT.log 不存在)。
