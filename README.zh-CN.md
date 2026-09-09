<p align="center">
  <img src="assets/pris-logo.png" alt="PRIS" width="760">
</p>

# PRIS —— 无机结构合理性法则（Plausibility Rules for Inorganic Structures）

**自主发现新的结构合理性法则，用于可解释、快速的晶体诊断与筛选**

[![arXiv](https://img.shields.io/badge/arXiv-2609.01209-b31b1b.svg)](https://arxiv.org/abs/2609.01209)
[![项目主页](https://img.shields.io/badge/project%20page-ai4qc.github.io%2FPRIS-4CC9F0.svg)](https://ai4qc.github.io/PRIS/)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

[English](README.md) · **简体中文**

> 本文是 [`README.md`](README.md) 的中文翻译，仅为方便中文读者阅读。**英文版为准**：
> 若两者出现分歧，以英文版和论文为准。

**项目主页：** <https://ai4qc.github.io/PRIS/> —— 八条法则、主要结果、五张主图、分析器的
完整示例，以及论文的单页海报。

一个自主智能体在预注册协议下，对 99,162 个实验晶体结构完成了 572 项编号调查，评估了
2,037,606 条候选法则，最终保留八条一行式法则。这八条即 **PRIS**，由它们导出的合成分数为
**PSS**。本仓库收录论文、分析代码、每张图所依据的汇总数据，以及在每项工作开始之前就已
固定其规则的预注册文件。

这八条法则编码了五种互补的物理化学机制：短程排斥、离子接触与堆积、静电平衡、键价守恒，
以及晶体学位点复杂度。一个结构若未通过，会被告知它违反了哪一种机制——这正是单纯的距离
截断说不出来的。

每条规则都同时用两个数字来衡量：**满足率**，即真实晶体中满足该规则的比例；以及**检出率**，
即在被刻意破坏的结构中被判定为不合理的比例。泡林 1929 年提出的规则从未在第二个坐标轴上
被打过分。两轴同时衡量时，结论发生反转：泡林第 2–5 条同时被 6.5% 的真实晶体满足，且无法
区分大多数相互竞争的结构对；而 PRIS 在留出的真实晶体上满足率为 82–99%，对受损结构的检出
率最高达 91%，相比之下，生成式管线中实际部署的距离截断只能检出 1.6–3.2%。

## 八条法则

八条一行式法则，只用结构文件和一张半径表就能算出的量。它们以嵌套集合的形式应用。
`Set 1` 到 `Set 4` 是论文报告的主链；`Set 1'` 是与主链并列而非嵌在其中的带守卫区间，
这也是为什么目录里有八条法则而 `Set 4` 只用到七条：

| | 法则 | Set 1 | Set 1′ | Set 2 | Set 3 | Set 4 |
|---|---|---|---|---|---|---|
| **Law 1** | ρ ≥ τ | τ = 0.735 | τ = 0.735 | τ = 0.804 | τ = 0.804 | τ = 0.804 |
| **Law 2** | f<sub>i</sub> > 0.50 ⇒ ρ ≤ 1.05 | – | ✓ | – | – | – |
| **Law 3** | 平均阴离子 CN ≤ 3.333 ⇒ 平均 d/(r₊+r₋) ≤ 1.081 | – | – | ✓ | ✓ | ✓ |
| **Law 4** | V<sub>M</sub>(i)/v<sub>i</sub> 的极差 ≤ 31.45 eV | – | – | ✓ | ✓ | ✓ |
| **Law 5** | max<sub>i</sub> V<sub>M</sub>(i) ≤ 15.17 eV | – | – | ✓ | ✓ | ✓ |
| **Law 6** | f<sub>i</sub> > 0.55 ⇒ 不存在同号离子成键 | – | – | – | ✓ | ✓ |
| **Law 7** | 不等价位点数 / 位点数 ≤ 2/3 | – | – | – | – | ✓ |
| **Law 8** | \|键价和 − v<sub>i</sub>\| / v<sub>i</sub> 的均值 ≤ 0.7143 | – | – | – | – | ✓ |

其中 **ρ** 是约化接触比，即最短的阳离子–阴离子距离除以两者 Shannon 半径之和；
**f<sub>i</sub>** 是泡林基于组成的离子性估计；**V<sub>M</sub>(i)** 是对形式电荷做 Ewald 求和
得到的位点马德隆能，**v<sub>i</sub>** 为位点 *i* 上形式电荷的绝对值；位点复杂度是在 spglib
symprec 0.01 下不等价位点数与位点数之比。条件型法则对任何未触发前提的结构都算作满足，
因此 Law 2、Law 3 和 Law 6 的条件子句把各自的法则限制在其适用域内。

在留出基准上（5,297 个真实结构、3,612 个受损结构，任何拟合步骤都未曾见过）：

| 集合 | 法则 | 满足率（真实） | 检出率（受损） |
|---|---|---|---|
| Set 1 | Law 1（宽松 τ） | 0.9919 | 0.2890 |
| Set 1′ | Law 1（宽松）+ Law 2 | 0.9894 | 0.3837 |
| Set 2 | Law 1、Law 3–Law 5 | 0.9579 | 0.6121 |
| Set 3 | Law 1、Law 3–Law 6 | 0.9171 | 0.7004 |
| Set 4 | Law 1、Law 3–Law 8 | 0.8180 | 0.9111 |

Set 4 表现最差的那一类损伤仍有 0.7338 的检出率。泡林第 2–5 条同时满足的实验结构比例为
0.0651。在重新施加损伤的母体基准上，同样这些集合的读数为 Set 1 的 0.991 / 0.268 与 Set 4
的 0.830 / 0.879；满足率与检出率始终取自同一个总体，且总体始终被明确写出。

## 合成分数

`PSS` 是由 PRIS 导出的合成分数。它的每一项都是某条法则已经在测量的量，只是重新拟合到
"什么已经被做出来过"的实验记录上，其中每原子体积占主导（标准化权重 −4.90）：

- 在实验结构上满足率匹配的条件下，PSS 筛除了 **83.7%** 的难合成结构，比计算得到的凸包能
  阈值高出 **31.8 个百分点**——后者达到 72.0%，且额外需要一次弛豫和一个相图凸包参照。
- 保留 80.7% 的实验结构时，它筛除了候选集的 51.9%。
- 在同组成配对中最有把握的那五分之一上，它达到 **0.944** 的准确率，而 DFT 凸包上能量为
  **0.844**。
- 法则的发现过程没有用到任何合成标签，因此 PRIS 合理性与可合成性之间的相关是一项发现，
  而不是一个拟合目标。

Set 1 到 Set 4 是保守的离散筛选，能指出被违反的机制；PSS 则提供连续可调的控制，决定综合
证据在多大程度上缩短验证队列。

## 判定一个结构

给它一个结构文件。它会报告每条法则、实测量、阈值、判定结果、该法则检验的机制、全部五个
嵌套集合，以及 PSS。

```bash
python src/pris_analyze.py mystructure.cif
python src/pris_analyze.py --quiet *.cif        # 每个文件一行判定
python src/pris_analyze.py --json POSCAR        # 机器可读
```

```
  MgAl2O4, 14 sites, charges from integer charge balancing, ionic character f_i = 0.759

    law    quantity                                measured  threshold  verdict  applied in       mechanism
    Law 1  reduced contact rho                       0.9865     0.8040     ok    1, 1', 2, 3, 4   short-range repulsion
    Law 2  reduced contact rho                       0.9865     1.0500     ok    1'               ionic contact
    Law 3  mean reduced cation-anion contact         4.0000     1.0810     --    2, 3, 4          packing
    Law 4  range of site Madelung energy / valence    3.6122    31.4500     ok    2, 3, 4          electrostatic balance
    Law 5  largest site Madelung energy            -20.2144    15.1700     ok    2, 3, 4          electrostatic balance
    Law 6  fraction of like-charge bonds             0.0000     0.0001     ok    3, 4             electrostatic balance
    Law 7  inequivalent sites / sites                0.2143     0.6667     ok    4                crystallographic site complexity
    Law 8  mean |BV sum - v_i| / v_i                 0.0384     0.7143     ok    4                bond-valence conservation

    Set 4  crystal chemistry           plausible
    PSS  +3.915        VERDICT  PLAUSIBLE
```

当一个结构未通过时，重点不在于判定本身，而在于最后一列：

```
$ python src/pris_analyze.py --quiet damaged/*.cif
IMPLAUSIBLE  compressed.cif   bond-valence conservation, short-range repulsion
IMPLAUSIBLE  expanded.cif     bond-valence conservation
```

阈值和 PSS 权重从 `agent_loop/frozen/` 中的冻结产物读取，因此今天得到的判定就是论文报告
的判定。`src/apply_rules.py` 作为更轻量的替代保留下来，它既不需要 spglib 也不需要键价
参数，因此只覆盖 Set 1 到 Set 3。

电荷是由组成推断出的形式氧化态。`BVAnalyzer` 从不被调用：它从键长反推价态，因此拿它的
输出去检验一条关于键长的法则，等于把结论当作前提。大约 19% 的结构无法判定（多种阴离子、
含复杂分子基团、整数与分数电荷分配均无解）。**"跳过"不等于"通过"。**

## 目录结构

```
PREREG.md            预注册：判据、数据划分与词汇表，在任何评估之前冻结
src/                 法则、分数与作图代码
figures/             论文的每一张图 -> 绘制它的脚本
agent_loop/          572 项编号调查、它们的计划、测试与冻结链条
paper/data/          每张主图所依据的汇总统计
paper/si_data/       补充材料各图的汇总统计
data/                小型参考表（键价参数、Lewis 酸度）
dft/                 第一性原理计算：协议、输入与收集到的结果
experiments/         图 4 与五张补充图背后的分析
pipeline/            数据获取
tests/               对法则、分数与图的检查
manuscript/          投稿源文件与两份编译好的 PDF
docs/                部署在 https://ai4qc.github.io/PRIS/ 的项目主页
```

`manuscript/main.pdf` 与 `manuscript/si.pdf` 是编译好的正文与补充材料。
`agent_loop/README.md` 说明了哪些调查线属于主线，以及哪些在后来被修正。

## 复现

**从已提交的数据出发作图** —— 不需要任何外部数据：

```bash
pip install -r requirements.txt
python figures/make.py --list      # 哪个脚本画哪张图
python figures/make.py "Fig. 1"    # 单张图
python figures/make.py --all       # 每个生成器各跑一次
cd manuscript && ./build.sh        # 需要 tectonic；写出 main.pdf 与 si.pdf
```

三十张图，十三个生成器。有三个脚本名早于最终编号方案，与它们实际绘制的图不一致——
以 `figures/manifest.json` 为准，`figures/README.md` 逐条说明了这些不一致之处。另有四张图
（Fig. 4、S17、S19、S22）还需要正例–未标注（PU）分数分片，这部分不再分发；
`figures/README.md` 说明了这一点，并给出相应的汇总数字。

**从结构出发的分析** —— 这些需要派生的特征库，它有数 GB 的 parquet，不在本仓库内。
请把脚本指向它：

```bash
export PRIS_FEATURES=/path/to/features/
python src/validity_rulesets.py   # 法则集 vs 生成模型中已部署的 validity 过滤器
python src/rank_rulesets.py       # 法则集在同组成排序任务上的表现
```

两个脚本都会先把已发表的行复现到小数点后四位，然后才输出任何新数字；若该检查失败，
`rank_rulesets.py` 拒绝写出结果。

拟合法则集与分数的那些脚本（`src/l4_*.py`、`src/f2r*.py`、`src/f3_*.py`）拒绝重跑它们
已封存的评估；上文引用的就是那些冻结结果。`f2r` 与 `f3` 是论文现在称为 PSS 及其前身的
链条的原始标识符。其余多数脚本仍保留着初次运行时与具体机器绑定的绝对路径。保留它们是
作为"当时究竟执行了什么"的审计轨迹，而不是作为一个可移植的库。

## 数据许可 —— 再分发任何内容之前请先阅读

- **ICSD** **不可再分发**：FIZ Karlsruhe 版权，外加欧盟数据库*特别权*。
- **ELEMENTA** 为 CC-BY-NC-4.0。
- **COD** 为 CC0。
- `data/bvparm2020.cif` 是 I. D. Brown 的键价参数表（麦克马斯特大学）；在保留其版权声明
  的前提下，可免费用于非商业用途的再分发，该声明写在文件头部。

本仓库只提交汇总统计。以 ICSD 标识符为键的逐结构中间产物已由 `.gitignore` 排除
（`paper/data/*_raw.csv`），并且必须保持排除。本工作发布的任何公开基准都只建立在 COD 之上。

## 相信任何一个数字之前，值得先知道的两件事

**划分纪律。** 阈值只在 `discovery` 划分上拟合（12,632 个真实 / 8,590 个受损）。
`calibration`（5,297 / 3,612）从未被任何拟合步骤看到。满足率与检出率始终取自同一个划分，
且划分始终被明确写出。划分由每个结构标识符的带种子哈希决定，不按组成分组，因此共享同一
组成的结构可能落到不同分区。另有 5,748 个结构的封箱被封存，开封配额为三次。目前已使用
一次（2026-08-01，一次未通过其预注册门的拟合分数验证；完整记录见 SI Note S8），还剩两次。
另外，当分析集在冻结之后被扩大时，有 3,215 行封箱数据没有划分标签，进入了一次全样本分位
拟合。因此阈值已经见过封箱集的一部分，它作为最终测试的价值应相应打折。

**反驳账本。** 有十一条结论曾被写成结果，随后又被得出它们的那个智能体推翻。这些结论与
存活下来的结果一并公开（`agent_loop/refutation_ledger.csv`，以及补充图 S1 和 S2），每一条
都与揭穿它的那个廉价诊断配对，因为那些诊断比催生它们的主张活得更久。

论文及其补充材料是每一个数字的最终依据。图与正文若有出入，以正文为准；
`manuscript/main.pdf` 与 `manuscript/si.pdf` 是本仓库发布时所用的编译版本。

## 引用

若你使用了 PRIS、PSS 或本仓库中的代码，请引用预印本
[arXiv:2609.01209](https://arxiv.org/abs/2609.01209)：

```bibtex
@article{song2026pris,
  title         = {Autonomous discovery of new structure-plausibility laws for explainable
                   and rapid crystal diagnosis and screening},
  author        = {Song, Zhilong and Cheng, Lixue},
  year          = {2026},
  eprint        = {2609.01209},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.mtrl-sci},
  url           = {https://arxiv.org/abs/2609.01209},
}
```

若要专门引用软件与冻结的法则定义：

```bibtex
@software{pris2026software,
  title   = {{PRIS}: Plausibility Rules for Inorganic Structures},
  author  = {Song, Zhilong and Cheng, Lixue},
  year    = {2026},
  url     = {https://github.com/AI4QC/PRIS},
  license = {MIT},
}
```

`CITATION.cff` 以机器可读的形式收录了同样的元数据，因此 GitHub 的 "Cite this repository"
按钮与本节保持一致。

## 许可

代码与论文正文：MIT（见 `LICENSE`）。这**不**延伸到上文所述的第三方结构数据库，也不延伸到
Springer Nature 的 LaTeX 类文件（`tex/sn-jnl.cls`、`tex/sn-nature.bst`），后者有各自的条款。
