# PRIS Fig.4–5 正刊风格重绘与应用主线修订计划

## 目标

把当前 additive 的 c–f 草图重做为包含原 Fig.4 a、b 的完整 a–f 主图，并统一到正文已有的 Arial、白底、无网格和旧版 PRIS 配色。主线从“L4 是保守应用门”改为：二元 PRIS 负责给出逐条物理化学解释和受控损伤证据，描述符推导的连续合成分数 (S_{\rm syn}) 在相同实验结构保留率下提供更强的 pre-DFT 队列压缩。

## 主图 a–f

1. **a：受控损伤基准的总体比较。** 保留原 Fig.4a 的距离阈值、SMACT、L1–L4 结果，突出 L4 在约 83% experimental-structure satisfaction 下的 87.9% damage detection；恢复每根柱顶的 detection 数字和顶部 satisfaction 数字，删去不必要的解释性长句。
2. **b：五类损伤的机制互补性。** 保留原 Fig.4b 热图、S1–S5 顺序、每个格子的数值、组间分隔线和距离筛选盲区黑框。这里的小字号只用于承载完整测量值，不用大段说明文字。
3. **c：应用层面的公平队列比较。** 横轴为 experimental-structure satisfaction，纵轴为 PU hard-negative screening。完整呈现 L1、L2、L3、L4 四个离散法则门，再绘制 (S_{\rm syn}) frontier、距离基线和 MatterSim basin–hull proxy。四个法则点是独立 operating choices，不表示级联。重点标出同一 80.7% satisfaction 下 (S_{\rm syn}) 的 83.7% 对 L4 的 51.9%，即 +31.8 percentage points，队列由 175,433 缩短到 59,517。95% satisfaction 点作为保守 operating point，不能把“全面超越”误写成所有 retention 区间都占优。
4. **d：独立 PU 分数关联。** 上下两行共享 within-model CLscore percentile；每一行都画三条线：CGCNN-PU、MatterSim-1M-MLP-PU，以及两者的逐十分位均值。模型名完整写出，不出现未定义的 A/B 或 Jang；去掉网格和大段说明文字，点与区间使用透明度区分层次。
5. **e：合成公式的独立验证。** 只保留 (S_{\rm syn}) 的 held-out synthesis ranking，并与 hull-energy 基线同图比较。稳定性分数不再放入主图：它在完整应用口径下的区分能力有限，且容易让读者误以为它是合成性或声子稳定性分数。
6. **f：物理化学解释。** 保留 on-hull/metastable × no-imaginary/imaginary 四格，比较 recorded 与 computed-only 的 D1/D7；去掉每个柱下的大样本数，样本数放 caption/SI。

### 关于稳定性分数的处理

冻结的稳定性分数只在明确能量间隔的同组成对中改善能量排序（全对准确率约 52.6%，能隙至少 0.10 eV atom$^{-1}$ 时约 75.9%），而在全池、中位数填补的应用口径下 AUC 约 0.707，95% 实验结构保留率下仅筛掉约 3.4% 的 PU 代理。它既不是声子稳定性、实验出现率，也不是合成性分数；把它放入本文会把两个不同问题硬串在一起。因此本轮从正文、主图和 SI 叙事中全部移除，只保留原始结果文件作为研究档案。

## SI 模型性能图

两个 PU 模型合并为一张标准分类模型图（2×2 子图），每个模型占一行：

- 左侧：50 个 bag 的 ROC 曲线、均值 ROC 曲线和 95% bag 区间，标注 mean ROC-AUC；
- 右侧：validation 标签在决策阈值 0.5 下的混淆矩阵（TP、FP、TN、FN 及百分比）；
- 图注明确标签是实验正例与抽样伪负例，OOB 是全池评分协议，不是 independent test；
- 若服务器重算没有输出原始预测，则不从 AUC 反推混淆矩阵，保留 validation-only 说明。

## 视觉门禁

- 统一正文已有 Arial、旧版蓝/橙/红/绿/灰配色、0.6–1.0 pt 线宽。
- 白底、无网格、只保留必要的左/下坐标轴线。
- 主图不放大段说明、模型路径、n 值或内部状态词；精确计数和协议写在 caption/SI。
- 主图 a–f 都必须存在，子图按 a→b→c→d→e→f 阅读。

## 数据口径

Fig.4 a、b 是 440-parent controlled-damage benchmark；Fig.4 c 是全池 PU hard-negative proxy cohort，二者不混写。(S_{\rm syn}) 的 matched-point 结果来自当前冻结公式流程，完整观测/中位数填补比例放 SI。

本轮只生成 additive 图、SI 图和计划/审计文件，不修改 `tex/`、`paper/` 或 canonical PDF；图确认后再进入正文集成阶段。
