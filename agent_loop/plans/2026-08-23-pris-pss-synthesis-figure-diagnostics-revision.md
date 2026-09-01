# PRIS–PSS 合成性叙事与图 3/4 诊断修订计划

## 修订目标

本轮不把六个问题拆成零散补丁，而是重建一条连续的学术论证链：PRIS 首先判断结构是否符合可检验的物理化学边界，并指出违反的机制；从理论结构走向实验实现还必须面对合成性，因此智能体再从 PRIS 相关描述符推导连续的 PRIS-derived synthesis score（PSS）；独立的 PU-learning 模型随后检验 PRIS 与合成性的关系；最后用真实运行的 MatterGen–UMA 逆向设计任务检验这些判断能否保护后续验证队列并解释哪些结构被筛去。

## 证据与术语账本

- `PRIS`：Plausibility Rules for Inorganic Structures，离散、机制可归因的结构合理性法则。
- `PSS`：PRIS-derived synthesis score，正文符号固定为 `S_syn`，是由六个 PRIS 相关描述符构成的连续分数。
- `PU learning`：方法基础引用 Jang 等 2020 年 JACS 论文（DOI `10.1021/jacs.0c07384`），该工作以 CGCNN 输出 crystal-likeness score（CLscore）。
- `CGCNN-PU`：沿用上述结构图编码思路的任务专用 PU 模型。
- `MatterSim-1M-MLP-PU`：以冻结的 MatterSim 表征接 MLP 头的第二种 PU 模型，用于检验关联是否依赖旧式 CGCNN 表征。
- `hard negatives`：两个 PU 模型判为难合成的代理负例，不能写成实验已证明不可合成。
- `MatterGen–UMA test`：本文实际运行的性质条件逆向设计测试。MatterGen 以 400 GPa 为目标生成结构，UMA 独立提供体模量代理值。

## 正文论证顺序

1. Fig. 4a–b 先证明 PRIS 在受控损伤中兼具筛选效率与机制诊断。
2. 用“理论候选能否走向实验不仅取决于结构合理性，还取决于合成性”自然引出 PSS，删除以候选池规模或计算成本为中心的旧句。
3. 在正文直接展示 PSS：先给紧凑的标准化线性式，再在公式下定义六个缩写和 `z(x)`；强调六项分别承载静电稳定、位点简约、键价守恒、致密堆积和配位网络拓扑信息。
4. 先交代 2020 JACS 的 PU-learning/CLscore 基础，再说明本文扩充实验与未标注结构库，并增加 MatterSim 表征模型，最后才给出 99,162 与 364,592 的规模和 Fig. 4c 结果。
5. Fig. 4c 将 L1–L4 的保守机制门控与 PSS 的连续可调筛选写成两种互补使用方式。PSS 在相同实验结构满足率下提高难合成代理负例的筛选率。
6. Fig. 4d 用两种 PU 表征中一致的单调关系完成关键推论：PRIS 的发现阶段没有使用合成标签，但其违反率仍随 CLscore 升高而下降；PSS 同时升高，因此这种关系不是某个特定模型的产物。
7. Fig. 4e 说明连续 PSS 为什么还能处理离散法则经常打平的同成分结构排序。
8. Fig. 4f 明确写成“为直接测试这一用途，我们真实运行了 MatterGen”。先说明实验设计，再给 1,081 个唯一结构和 UMA 预测结果，最后给队列缩减与高性质候选保留。
9. Fig. 4f 增加被 PSS 筛去结构的机制诊断：当前两特征支持域中，61 个被筛结构均具有 0.01 对称容差下全部位点不等价的特征，并处于更大的每原子体积区间；它们全部通过 0.7 Å 距离阈值，却全部触发 D7。用一个被筛结构和一个保留的高性质结构展示这种差异。

## 图形改动

- Fig. 3a：先以投影边界回归测试复现 S4 裁切，再给 3D 视图留足原子半径和晶胞线的投影余量，不改变其余四种损伤的构图。
- Fig. 4f：保留现有 PSS 曲线和 L1–L4 运行点，在左下空白区嵌入两幅小型晶体示意图及极短数据标签；完整统计进入正文和状态文件，不用大段说明文字占据图面。
- 所有新图仍保持正文统一的 Arial、无网格、半透明填充和同色深边框风格，并复制到 `tex/key-file/`。

## 测试先行

1. 在 LaTeX 集成测试中先断言正文含 PSS 显式公式、六项缩写定义、2020 JACS 引文、扩充数据集的首次引入，以及 “we ran MatterGen” 的直接实验表述。
2. 增加 Fig. 3 S4 投影留白测试，使当前裁切实现先失败。
3. 增加 Fig. 4f 诊断数据测试，固定 61 个筛去结构、140/140 高性质候选保留、61/61 D7 违反、61/61 通过 0.7 Å 距离阈值及典型结构的可追溯来源。
4. 增加图 4 状态文件和输出资产测试，确保诊断统计与两幅结构缩略图真实进入图，而不是只出现在文字里。

## 具体文件

- 修改：`tex/body.tex`、`tex/methods.tex`、`tex/si_body.tex`、`tex/refs.bib`、必要时 `tex/front_body.tex` 与 `tex/front_meta.tex`。
- 修改：`src/fig3_anatomy.py`。
- 修改：`experiments/pu_synthesizability_20260821/plot_merged_fig45_nature.py`，必要时抽取一个可复用的逆向设计诊断函数。
- 新增或扩展：`tests/test_tex_pris_section24_integration.py`、Fig. 3 与 Fig. 4 相关测试文件。
- 重建：`paper/figs/fig3_anatomy.*`、`tex/figs/fig4_validation_synthesis.pdf`、正文与 SI PDF，并同步 `tex/key-file/` 中的关键图和编号 PDF。

## 验收

- 定向测试与完整相关测试全部通过。
- 正文和 SI 均可无错误编译。
- 逐页渲染检查 Fig. 3a S4 无裁切，Fig. 4f 缩略图与曲线不重叠，PSS 公式不溢出栏宽。
- 首次出现审计确认每个模型、数据集和数字都有前因，正文子图引用仍按 a→b→c→d→e→f 顺序。
- `git diff --check` 通过，且不覆盖工作区内与本轮无关的用户修改。
