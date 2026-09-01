# PREREG-F2R: 实验域 e_hull 排序公式重建(修复 F2 不可再生 + 冲击更高准确率)

日期冻结:2026-08-14(在任何 discovery 拟合之前写下)。
状态:**冻结**;仅可追加"修订记录"。

## 1. 动机

论文承认 F2 的 0.5725/0.7313 是"当前唯一无法从归档特征表再生的数字"
(fold 内标准化统计未保存)。本链条:(a) 用可再生协议重建稳定性公式;
(b) 用增强特征池(对称性 + 经典 Born 项 + 密度)冲击更高的组等权准确率。

## 2. 数据与分割(先于本预注册存在,不变)

- `real_rank.parquet` + phys/elec/geom 特征合并(`formula2.load()` 同口径,
  剔除 lockbox 与 split 未知行)+ 新增 `real_rank_aug.parquet`(定义与
  PREREG-F3 §4 相同的 13 个新特征,由 `f6_features.py` 生成)。
- 拟合仅用 `split == discovery`;插补中位数与标准化统计仅来自 discovery,
  冻结后应用于 calibration。
- 评估在 `split == calibration` 的组成组;标签 = 组内 e_hull 排序;度量 =
  S13 组等权 accuracy(`rank_rulesets.evaluate` 同一实现)。
- calibration 在本仓库历史上被适应性重用过;本链条数字标注同一限定语,
  与论文对 calibration 的表述一致。lockbox 不接触。

## 3. 程序(冻结,与 PREREG-F3 §5 相同的模型类)

- 组内全配对(e_hull 严格不相等的对),X = z(低hull) − z(高hull),y = 1,
  对权重 1/组内对数;反对称 logistic 无截距。
- 贪心前向选择:discovery 内 5 折 GroupKFold(按组成),评分 = 验证折
  组等权 accuracy;停止:增益 < 0.002 或 8 项。全 discovery 重拟合后冻结。
- 同时输出 F2R-full(全特征 L2)作上限参考。
- **F2 重述基线**:论文 7 项系数原样,z 统计取自 discovery(冻结),在
  calibration 求值 —— 这是"可再生版 F2",与新公式做配对比较。

## 4. 成功门(冻结)

- H1:acc(F2R) − acc(F2 重述) 在 calibration 组聚类自助(B=2000, seed
  20260728)95% 下界 > 0
- H2:acc(F2R) > bl_min 单量与 vol_per_atom 单量(点估计)
- H3:无论 H1 成败,归档全部 fold 统计与冻结 z 统计,使每个数字可再生;
  论文中 F2 段落改写为可再生表述

措辞分级:H1 过 →"a regenerable seven-or-fewer-term score improves on F2";
H1 不过 →"F2 restated regenerably; no material gain from the enlarged pool",
一样并入论文(负结果照报)。

## 修订记录

**修订1(2026-08-14,calibration 接触前)**:见 PREREG-F3 修订1,同一内容。
首次 fit 的退化冻结文件(F2R_frozen.json,单项 cn_cat_max)作废并被覆盖;
其 cv 数字(0.6660@cov0.458)保留在本句中作记录。

## 追加链条 F2R-G(2026-08-14,新链;calibration 复用披露)

目标:能隙可分辨区(|ΔE_hull| ≥ 25 meV/atom)的组等权准确率冲 0.80。
程序:与 F2R 完全同一模型类/特征池/贪心程序,仅有两处不同——(a) 拟合与 CV 评分
的配对均限制在 |ΔE| ≥ 25 meV;(b) calibration 求值也限制在该配对集(组等权 +
聚类自助 CI)。冻结后一次求值,输出 outputs/20260814_f2rg_gap25/。
门:G-gap = 组等权 acc ≥ 0.80(点估计)。未达即如实报告达到的点位与 CI,
并与 F2R 在同一配对集上的 0.7438 [0.675, 0.808] 作配对比较。
本链条与 F2R 的 CALIB_CONTACT 相互独立;calibration 的累计复用次数在 SI 披露。
