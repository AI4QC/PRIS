# `tex-submission` 逐段逐句修改计划

**日期：** 2026-08-28  
**状态：** 仅审阅与计划，尚未修改任何论文源文件  
**目标：** 旗舰刊 `Nature` 的简洁性、跨学科清晰度和逻辑连贯性

## 0. 本计划的严格范围

只审阅当前投稿包实际引用的论文文本：

- `tex-submission/main.tex`
- `tex-submission/front_meta.tex`
- `tex-submission/front_body.tex`
- `tex-submission/body.tex`
- `tex-submission/methods.tex`
- `tex-submission/si.tex`
- `tex-submission/si_body.tex`

不审阅 cover letter、旧版 `front.tex`、参考文献核验记录、其他目录、事实账本、代码、数据或 DFT 记录。

本计划只处理三类问题：

1. 表达啰嗦或重复；
2. 非本领域读者不容易理解；
3. 句间、段间或章节间逻辑不连贯。

作者要求保留所有强结论和 headline results。本计划不包含降调、收窄结论或 overclaim 修改意见。

## 1. 执行符号

- `拆`：把一个多命题长句拆成两个或三个句子。
- `并`：合并相邻重复句，只保留一次信息。
- `删`：删除没有新增功能的重复句、元话语或空泛连接词。
- `移`：将句子移到更合适的段落或紧邻其证据的位置。
- `改`：保留事实和结论，仅重写主语、谓语、指代或平行结构。
- `分段`：保持句子顺序，但建立新的段落控制思想。
- `保留`：不需要修改。

---

# A. 主文逐段逐句计划

## A1. `main.tex`

### 作者信息、关键词与声明

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| M-01 | `main.tex:50-51`，`crystal chemistry ... autonomous scientific discovery` | 六个关键词跨度很大，顺序没有反映论文主线 | `改`：按“核心对象 PRIS/晶体化学 → 方法 autonomous discovery → 应用 generative materials”排序；不增删科学概念 | 让检索词顺序与标题和摘要一致 |
| M-02 | `main.tex:71-75`，`Entries from ... are not redistributable ... and ELEMENTA ... The public benchmark ... therefore...` | 一个句子同时承担两种许可状态和公开基准来源 | `拆`：第一句分别说明 ICSD 与 ELEMENTA 的许可；第二句单独说明公开 benchmark 仅由 COD 构成 | 清楚区分“不能再分发什么”和“实际公开什么” |
| M-03 | `main.tex:75-76`，`Derived scalar features, split assignments and all numerical results...` | 无明显问题 | `保留` | 说明 Source Data 的范围 |
| M-04 | `main.tex:76-80`，`The 260 design candidates ... are provided ... one CIF per candidate, with an index giving...` | 句尾连续列出五类元数据，主干过长 | `拆`：第一句只说明提供 260 个 relaxed CIF；第二句列出 index 中的 composition、space group、site fraction、PSS 与 bulk modulus | 分开数据可用性与索引字段 |
| M-05 | `main.tex:63-68` Acknowledgements 与 Competing interests | 表达简洁明确 | `保留` | 不改 |
| M-06 | `main.tex:82-84` Code availability | 表达简洁明确 | `保留` | 不改 |

## A2. `front_meta.tex`：标题与摘要

### 标题，7-8 行

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| FM-01 | `front_meta.tex:7-8`，`Autonomous discovery of crystal-structure plausibility laws for explainable and rapid crystal screening and diagnosis` | `crystal` 重复，连续名词和双重修饰语过多 | `改`：保留 autonomous discovery、plausibility laws、screening/diagnosis 三层信息，压缩重复的 `crystal`，使 `rapid` 与 `explainable` 修饰对象平行；最终再按 Nature 标题长度检查 | 一遍读懂“发现了什么”和“用来做什么” |

### 摘要，11-28 行

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| FM-02 | `11-12`，`faster than density functional theory (DFT) energy and phonon calculations, or experiments, can assess them` | 三种验证方式句法不平行，插入逗号妨碍阅读 | `改`：使 DFT energy calculations、phonon calculations 和 experiments 成为严格平行的三项 | 建立生成速度与验证速度的矛盾 |
| FM-03 | `12-14`，`Deciding which merit expensive computation is therefore the bottleneck, yet most screens...` | 一句同时给瓶颈和现有方法缺口 | `拆`：第一句明确候选优先级排序是瓶颈；第二句以 current rapid screens 为主语说明只排除原子重叠 | 从问题自然转入方法缺口 |
| FM-04 | `14-16`，`Herein, agents propose, implement, test and actively refute...` | `Herein` 生硬，四个动作动词拥挤 | `改`：用 `Here` 或明确主语开头；把动作压为“generate and actively test/refute”，保留两百万与八条法则 | 简洁交代方法规模和核心发现 |
| FM-05 | `16-18`，`These laws encode five mechanisms: ...` | 列表清楚，但与前句连接可更紧 | `并`：保留五机制完整列表，将 `These laws` 明确回指 eight PRIS laws；不拆散列表 | 解释八条法则的物理化学内容 |
| FM-06 | `18-19`，`Our laws keep 82--99% ... against 6.5%...` | `against` 省略了第二个比较谓语 | `改`：写成两个句法对称的分句，双方都明确使用 keep/satisfy 指标 | 无需回读即可理解对比 |
| FM-07 | `19-20`，`They also detect 87.9% of damage...` | `They` 可能指八条单法则或所有 law sets；`damage` 首次出现不具体 | `改`：直接点名 strictest PRIS set，并明确 controlled structural damage | 准确标识 87.9% 对应对象 |
| FM-08 | `20-21`，`Moreover, the plausibility they measure is proven to be linearly correlated...` | `Moreover` 空泛，被动结构拖慢主结论 | `改`：删除空泛连接词和被动式，以 PRIS plausibility 为主语直接陈述线性相关强结论 | 从 damage detection 转入 synthesizability |
| FM-09 | `21-23`，`therefore explainably screens...` | `explainably` 搭配不自然，读者不知道“可解释”具体指什么 | `改`：用 `mechanism-resolved screening` 或等义表达，保留 83.7% 与 80.7% | 说明解释性来自命名失败机制 |
| FM-10 | `23-25`，`cut the DFT validation queue by up to 67.3% while keeping 99.2%...` | 两个 headline numbers 挤在一个长句，第二个容易被吞没 | `拆`：第一句突出 queue reduction；第二句突出 target-reaching candidates 的 99.2% retention；参数调节关系在 Results/Methods 说明 | 分别突出节省和保留性能 |
| FM-11 | `25-26`，`explains why GNoME ... and why falsified crystals...` | 两个性质不同的外部案例共用一个 `why` 结构 | `拆`：GNoME chemical ordering 和 fixed-coordinate wrong-element diagnosis 各用一个平行分句或短句 | 两个案例各承担一个诊断功能 |
| FM-12 | `26-28`，`moves screening from pass-or-fail to why ... showing autonomous agents can discover...` | `from ... to why` 不平行；一句同时收束 PRIS 与 agentic science | `拆`：第一句用平行结构表达从 binary verdict 到 chemical diagnosis；第二句以 `More broadly` 引出 autonomous active-refutation 的强结论 | 先收束工具价值，再给广义意义 |

摘要段落执行后仍保持原始顺序：`瓶颈 → 自主发现 → 五机制 → 定量性能 → PSS/逆向设计 → 外部案例 → 广义意义`。

## A3. `front_body.tex`：Introduction

### 第 1 段，4-21 行：筛选瓶颈与精确缺口

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| I-01 | `5-11`，`High-throughput databases ... now supply candidates faster than they can be assessed, whether by...` | 主语列表、文献、三种评估方式嵌套在一个长句 | `拆`：第一句报告候选生成速度；第二句平行列出 DFT energies、phonons 与 experiments | 解释为何验证成为瓶颈 |
| I-02 | `12-13`，`An initial screen must therefore decide which predictions consume...` | 与 4-5 行 `deciding which predictions warrant...` 重复 | `并`：把 gatekeeper 含义并入上一组句子的结尾，删除重复动词结构 | 用一次表述完成“速度失配 → 筛选需求” |
| I-03 | `13-17`，`Yet many generative pipelines test little more than... and avoiding gross overlap establishes neither...` | 现状与局限塞在同一句 | `拆`：第一句说明 fixed minimum distance；第二句以 `Passing this test` 为主语列出 coordination、electrostatics、bond valence、ordering | 先说做了什么，再说遗漏什么 |
| I-04 | `17-18`，`Commentary now argues that data alone...` | 文献评论式起手打断从筛选局限到本文缺口的直线逻辑 | `并`：删除 `Commentary now argues` 的元话语；将“AI must learn chemical rules”与下一句缺口合并，引用保留 | 直接导向需要可解释化学规则 |
| I-05 | `19-20`，`rules ... that name the assumption a structure violates` | `name the assumption` 抽象 | `改`：明确为 identify the physical or chemical constraint violated | 定义本文“可解释”的具体含义 |
| I-06 | `20-21`，`Such rules would sit between...` | 清楚、承担关键层级定位 | `保留` | 收束第一段 |

### 第 2 段，23-37 行：Pauling 与距离阈值的相反失效

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| I-07 | `23-26`，`they connect ionic size ... and need only a radius table...` | 化学内容和低计算成本是两个命题 | `拆`：第一句列四类化学思想；第二句单独说明只需 radius table 与 formal charges | 分别呈现丰富性和经济性 |
| I-08 | `26-28`，`They belong to a tradition of simple laws...` | 泛泛评价独立成句，信息增量小 | `并`：并入 I-07 第二句，改成“chemical content + low input cost”的总结；引用保留 | 紧凑说明 Pauling 先例的价值 |
| I-09 | `29-32`，literature `13%`、current `6.5%` 与 distance `1.6/3.2%` | 文献结果和本文两种结果连续堆叠 | `拆`：先给文献 audit；再给本文 6.5%；用 `At the opposite extreme` 单独引出距离阈值 | 形成“过严—过松”的对称结构 |
| I-10 | `32-34`，`The classical rules are ... and the distance cutoff is...` | 对称清楚 | `保留`，仅检查两分句长度和平行语法 | 一句总结相反失效模式 |
| I-11 | `34-36`，`How many experimental structures ... since a bound loose enough...` | 原理和例子嵌套在一个因果句 | `拆`：第一句说明 retention 不能单独衡量筛查；第二句用极松阈值解释原因 | 为双指标标准建立直观逻辑 |
| I-12 | `36-37`，`A good law must therefore...` | 三项要求平行、功能明确 | `保留` | 得出评价标准 |

### 第 3 段，39-46 行：两个实际问题

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| I-13 | `39`，`Two further questions decide what such rules would be worth` | 路标过于抽象 | `改`：直接点名 synthesizability 与 structural correctness 两项测试 | 预告本段的两问 |
| I-14 | `41-45`，`Databases record ... so synthesizability was estimated ... none of which ...; whether...` | 数据困难、三类方法和未知问题挤在一长句 | `拆`：依次写数据库困难、三类替代方法、这些方法不读 chemical bound、本文未知问题 | 按“困难 → 现状 → gap”推进 |
| I-15 | `45-46`，`whether the structure handed downstream is correct at all, because every assessment...` | 问题和重要性原因在同一句，`it` 指代远 | `拆`：第一句提出 correctness；第二句用 `the supplied structure` 明确说明后续评估的假设 | 清楚提出第二问 |

### 第 4 段，48-66 行：GNoME/A-Lab 与错误元素案例

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| I-16 | `48`，`Prominent cases show that it need not be` | `it` 的先行词模糊 | `改`：直接写 structures handed downstream can be chemically misassigned | 承接 structural correctness |
| I-17 | `48-51`，GNoME `381,000...421,000... enriched...` | 数据库规模与低对称现象挤在一句 | `拆`：第一句给规模；第二句专门陈述 rare low-symmetry enrichment 和 distance-filter pass | 分开背景规模与待解释现象 |
| I-18 | `51-53`，`similar elements, such as ... ordered artificially...` | 元素例子插在主干中间 | `改`：先说相似元素被有序化到本应等价位点，再把 rare-earth/Zr-Hf 作为句末例子 | 先理解机制，再看化学实例 |
| I-19 | `53-55`，`The same variable recurred at A-Lab...` | `same variable` 含糊 | `改`：直接用 chemical ordering；与后续复核信息组成同一个 A-Lab 案例组 | 明确两个案例的共同变量 |
| I-20 | `55-57`，`A reanalysis disputed ... and an author correction...` | 两次后续事件塞在一句 | `拆`：分别写 reanalysis 与 author correction；与 I-19 保持连续 | 清楚交代争议演进 |
| I-21 | `58-60`，`documented at least 70 falsified structures that reused...` | 绕行表达 | `改`：直接写 genuine diffraction data were paired with altered element identities | 快速交代第二类失败机制 |
| I-22 | `60-61`，`An archive ... repeats this...` | `this` 模糊，独立句信息不足 | `并`：并入 I-21，作为该模式跨 Cu/Ni/Mn/Fe 标签重复的补充 | 形成完整历史案例句 |
| I-23 | `61-62`，`These cases share one missing check ... examined by neither...` | 插入同位语导致主谓不清 | `改`：以 distance filters and energy calculations 为主语，直接说它们不检查 chemical ordering or elemental identity | 抽取两个案例的共同缺口 |
| I-24 | `62-65`，`two open cases of different kinds: ...` | `different kinds` 空泛，冒号后两项不平行 | `拆`：两句分别写 artificial ordering 与 wrong occupant at plausible coordinates | 为两个 Results 应用建立平行问题 |
| I-25 | `65-66`，`Closing either requires laws...` | 清楚 | `保留` | 收束案例到本文需求 |

### 第 5 段，68-92 行：研究路线与结果预告

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| I-26 | `68-71`，`Given experimental structures ... they proposed, implemented, tested and refuted...` | 输入与四个动作挤在一句 | `拆`：第一句列输入；第二句压缩为 generated/implemented/actively tested candidates | 先交代系统获得什么，再交代做什么 |
| I-27 | `71-72`，`Failed claims stayed on record...` | 与上一句因果关系可更直接 | `并`：用 `Because failed claims remained...` 接在方法句之后 | 解释何谓 refutation-driven progress |
| I-28 | `72-74`，`Two million candidate laws were reduced ... to eight...` | 强结论本身清楚 | `保留`；确保本段只完整报告一次 | 交代搜索漏斗 |
| I-29 | `74-76`，`Each law encodes ... five mechanisms ... so every violation...` | 机制列表与诊断结果是两个命题 | `拆`：列表一句；每次 violation 指向机制一句 | 分开“是什么”和“有什么用” |
| I-30 | `76-78`，`kept 82--99% ... and the strictest detected 87.9%...` | 两项性能同句 | `拆`：retention 与 damage detection 各一句 | 并列突出两项性能 |
| I-31 | `78-79`，`No synthesis label ... yet the plausibility they measure...` | `their/they` 指代松散 | `改`：直接点名 PRIS laws 与 PRIS plausibility，保留强线性相关结论 | 从结构性能转入 synthesizability |
| I-32 | `79-81`，`PSS screened 83.7% ... keeping 80.7% ... and every removal...` | 两个比例和解释性结果拥挤 | `拆`：性能一句；mechanism traceability 一句 | 分别突出性能与解释性 |
| I-33 | `82-83`，`A hull-energy threshold reached 72.0%...` | 在 Introduction 结尾插入次级基线，打断主线 | `删`：从 Introduction 删除，完整比较保留在 Results | 避免路线段变成结果清单 |
| I-34 | `83-87`，inverse-design task、`67.3%`、`260`、`99.2%`、relaxed cells | 一句承担任务、节省、验证和数据可用性 | `拆/压`：Introduction 只保留任务与两项 headline result；260 个计算和 data availability 留 Results/声明 | 紧凑预告应用，不复述方法 |
| I-35 | `87-89`，`The same mechanisms speak to both open cases...` | 比喻式 `speak to`，两个案例同句 | `改/拆`：用 applies mechanism-resolved diagnosis；GNoME 与 wrong element 各一句或平行分句 | 明确回扣前文两问 |
| I-36 | `89-90`，`turns ... from a pass-or-fail check into a chemical diagnosis` | 核心句清楚 | `保留` | 收束 PRIS 价值 |
| I-37 | `90-92`，`It also shows...` | `It` 指代不明确 | `改`：用 `This active-refutation workflow` 作主语 | 准确落到 autonomous science 强结论 |

**段落动作：** 在 78 行前 `分段`。68-78 行只讲发现过程、八条定律、机制和结构性能；78-92 行讲 synthesizability、inverse design、两个外部案例与总体意义。

## A4. `body.tex`：Results 与 Discussion

### Results 小标题

| 编号 | 行号与原标题 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-H1 | `8`，`Autonomous agents discover eight laws through proposal, test and refutation` | 过长，三个过程名词占据标题主体 | `改`：保留 autonomous discovery、eight laws、active refutation，压缩为一个结论型短标题 | 一眼看到本节新发现 |
| R-H2 | `66`，`PRIS balances experimental-structure satisfaction with damage detection` | 表达准确但过长 | `改`：压缩复合名词，保留 satisfaction–detection balance | 突出性能权衡 |
| R-H3 | `165`，`Five complementary mechanisms turn screening into diagnosis` | 核心信息清楚，仅长度略高 | `改`：删除可由正文表达的修饰词，保留 five mechanisms 与 diagnosis | 强调物理化学意义 |
| R-H4 | `235`，`PRIS screens candidates before expensive calculations` | 清楚 | `保留`，仅在最终统一小标题长度时检查 | 引出应用 |
| R-H5 | `381`，`Plausibility precedes stability checks and addresses controversy and identity failure in crystallography` | 同时塞入三个结论，长度远高于其他标题 | `改`：标题只保留 ordering/identity diagnosis 或 plausibility-before-stability 的一个控制思想；另一层作为 386 行段首主题句 | 让本节只回答一个总问题 |

### Results 2.1，第 11-21 行：提出、测试和反驳

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-01 | `11-12`，`their value depends on eliminating their own false explanations` | 两个 `their` 回指不同对象，拟人化较重 | `改`：以 candidate statements 为明确主语，用 testing and eliminating false explanations 表达 | 建立“提出容易，排错决定价值” |
| R-02 | `13-15`，`experimental structures satisfy and damaged versions fail` | `damaged versions` 未说明来自相同 parent | `改`：首次明确为 controlled perturbations of experimental parents | 定义正负结构关系 |
| R-03 | `15-17`，`proposed and implemented hypotheses, selected thresholds..., designed counterexamples...` | 构建和反驳阶段挤在一句 | `拆`：第一句写 proposal/implementation/threshold；第二句写 counterexamples/held-out/physical controls | 按发现流程推进 |
| R-04 | `18-21`，`Refuted claims stayed ... making that search a sequence...` | 失败记录用途与方法学结论同句 | `拆`：先说记录成为下一轮 diagnostics；再用 `Thus` 总结 falsifiable experiments | 清楚解释 refutation loop |

### Figure 1 图注，第 27-41 行

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-05 | `31-33`，t-SNE `projection of ... statements; blue intensity ... red circles...` | 数据对象、颜色和红点三类编码同句 | `拆`：第一句说明二维投影对象；第二句说明 blue density；第三句说明 red circles 对应 Law 1–8 | 按“对象 → 颜色 → 标记”读图 |
| R-06 | `36-37`，`The eight laws Law 1--Law 8... Set 1, Set 1'...` | `eight laws Law 1--Law 8` 重复，集合列表拖长 | `删/改`：删除重复命名；保留 predicates、set membership 和两项 performance | 简洁说明 panel d |
| R-07 | `38-41`，running best、两条曲线、inset、strip | 四种视觉信息在一个长句 | `拆`：主曲线一句；inset 与 outcome strip 各一短句 | 让 panel e 可顺序解码 |

### Results 2.1，第 45-56 行：搜索规模和结果

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-08 | `45-48`，`fixed splits of ionic structures from 99,162 experimental ICSD and COD structures` | `structures` 重复，数据源插入主干 | `改`：把总数放主句，ICSD/COD 与 Methods/SI 指针后置 | 快速交代分析总体 |
| R-09 | `50-51`，`Scale mattered because it multiplied the chances...` | `Scale mattered` 空泛 | `改`：直接说明 repeated datasets and controls created more opportunities to refute passing claims | 解释规模的科学作用 |
| R-10 | `51-53`，572 investigations 与 2,037,606 evaluations | 清楚但与 56 行再次出现 two million | `保留`规模句；56 行只承担“八条存活”的收束，避免两次完整解释同一数字 | 一次报告规模、一次报告结果 |
| R-11 | `53-55`，`failed controls through misleading metrics, class imbalance, recognition...` | 四类失败连续枚举，读者难分层 | `拆`：先说 11 个结论失败；再把原因组织为 metrics/data imbalance、perturbation recognition、implementation 三组 | 展示反驳结果 |
| R-12 | `55-56`，`Most investigations changed nothing: ... improved only four times...` | 冒号后包含两层时间信息 | `拆`：第一句说多数调查不改变 best set；第二句报告 only four improvements 与 last after 500 | 突出发现并非线性累积 |
| R-13 | `56`，`Of two million evaluations, eight one-line laws survived...` | 段落结论有力 | `保留`，仅与 R-10 去重 | 完成搜索漏斗 |

### Results 2.1，第 58-64 行：law-set sequence

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-14 | `58-61`，Set 1–4、Set 1'、图和 SI 指针同句 | 主线集合与辅助集合拥挤 | `拆`：第一句只定义 Set 1–4 的 strictness sequence；第二句说明 Set 1' 与完整定义位置 | 先建立主序列，再补辅助选择 |
| R-15 | `62-63`，`early screening may favour..., database auditing fuller...` | 后半省略谓语，平行结构不完整 | `改`：补 `whereas database auditing may favour fuller damage detection` | 解释任务为何选择不同严格度 |
| R-16 | `63-64`，`Choosing ... therefore requires two measurements` | 清楚 | `保留` | 引出下一小节 |

### Results 2.2，第 69-74 行：两项指标和基线

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-17 | `69-70` 两个定义 | 清楚但行内过密 | `拆`：Satisfaction 与 damage detection 各自独立成句，保持相同句式 | 平行定义核心指标 |
| R-18 | `70-72`，`bound that every structure satisfies ... converse bound scores the reverse` | `the reverse` 需要读者自行推导 | `改/拆`：分别明确 loose bound 与 strict bound 的两项数值方向，再总结 both needed | 直观解释 trade-off |
| R-19 | `72-74`，charge neutrality 与 fixed-distance 两个基线 | 两种失败原因同段但没有显式对照 | `拆`：composition-only baseline 一句；element-blind distance baseline 一句 | 形成两类 baseline map |

### Results 2.2，第 76-87 行：Law 1 与 DFT 能量尺度

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-20 | `76-84`，Law 1 引入、公式和变量定义 | 结构清楚 | `保留`；仅把 equation 后定义控制在一句 | 建立 reduced contact |
| R-21 | `84-85`，absolute cutoff 对 small ions 更严格 | 清楚 | `保留` | 解释尺度归一化 |
| R-22 | `85-87`，threshold definition 与 performance 同句 | 规则定义和结果混合 | `拆`：第一句定义 0.735；第二句报告 99.2%/28.9% | 分开 law 与 evidence |
| R-23 | `87`，`To put ... we rigidly scaled twenty...` | 目的、样本和 DFT 动作可分 | `拆`：方法一句；结果另起句 | 先说明验证设计 |
| R-24 | `87`，`median 0.1 eV ... so both floors ... and ... 1.80 times... which is what makes...` | 能量 crossing、阈值解释、跨化学优势四层嵌套 | `拆`成三句：crossing；两个 floors 的能量区域；1.80 倍 localization 与 shared coordinate 的意义 | 清楚完成物理标定 |

### Results 2.2，第 89-112 行：八条公式和定义

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-25 | `89`，`Seven further laws take the same form` 后公式又列 Law 1 | 引导语与显示内容不完全对应 | `改`：写成 `Together, the eight laws are`，或正文只列 Law 2–8 | 消除“七条/八条”的瞬时冲突 |
| R-26 | `90-105` 完整八条公式 | 主文信息密度高，但公式本身无语言问题 | `保留公式内容`；实施时评估只保留代表式和 Fig. 1d 指针，完整式仍在同一投稿包 SI | 降低主文认知负荷，不丢信息 |
| R-27 | `106-108`，Madelung、formal charge、BV 连在一句，`and BV bond valence` 语法残缺 | 三个量定义不完整 | `拆/改`：每句定义一个量；明确 BV 的全称和所指 quantity | 逐项建立符号 |
| R-28 | `109-111`，trigger condition 与 no verdict | 两种条件规则在同一句 | `拆`：第一句说明 unmet trigger satisfies conditional law；第二句说明 missing required input returns no verdict | 区分条件不触发与输入缺失 |
| R-29 | `111-112`，`Each law answers one question...` | 清楚 | `保留` | 引出组合必要性 |

### Results 2.2，第 114-125 行：组合规律的性能递进

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-30 | `114-116`，expansion、exchange、same closest distance 三例同句 | 三种 blind spots 列表稍重 | `拆`：总判断一句；三例作为平行短句或紧凑列举 | 建立为何需要互补机制 |
| R-31 | `117-118`，Set 3 performance | 清楚 | `保留` | 报告中间 law set |
| R-32 | `119-120`，`two and a half times ... one in twelve rather than...` | 同一句同时给 detection gain 与 rejection cost | `拆`：gain 一句、cost 一句；保留所有数字 | 让 trade-off 易扫读 |
| R-33 | `120-124`，Set 4 composition 与 performance | 规则构成和结果连续较密 | `拆`：添加 Law 7/8 一句；overall/per-class performance 一句 | 先说为何更严，再报效果 |
| R-34 | `124-125`，`gain came from combining laws...` | 清楚 | `保留` | 局部推论 |

### Results 2.2，第 127-133 行：与既有准则比较

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-35 | `127-130`，distance filter 与 Pauling 对比、样本条件 | 主比较和样本限定跨两句分布 | `改`：先用对称句说明两种相反失败；再把 5,297/charge-balanced 条件后置于 6.5% 结果 | 突出核心对照 |
| R-36 | `131-132`，`not greater strictness but readable chemical discrimination...` | 冒号后是名词片段 | `改`成完整因果句：PRIS occupies the useful region by combining readable chemical discrimination rather than uniform strictness | 解释优势来源 |
| R-37 | `132-133`，cutoffs fixed → test verdict dependence | 清楚 | `保留` | 引出 split check |

### Figure 2 图注，第 138-149 行

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-38 | `147-149`，re-derived constants、relative displacement、right-hand flips 同句 | 结果点、归一化和右列三种读图信息拥挤 | `拆`：先说明 points at unity；再说明 right-hand verdict changes | 让 panel d 顺序可读 |
| R-39 | 其他 panel 说明 | 清楚 | `保留` | 不改 |

### Results 2.2，第 153-163 行：split consistency

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-40 | `153-157`，三类 cutoffs、electrostatic cutoffs、verdict changes、Law 7 | 多个量连续堆叠 | `拆`：小幅 threshold shifts 一句；electrostatic shifts 一句；verdict stability 一句 | 区分参数变化与输出变化 |
| R-41 | `157-161`，连续列出五幅 Supplementary Figures 及其主题 | 图号清单打断主论证 | `压/移`：将五图归为 threshold、band、chemistry 三类，并把引用集中放段末 | 保留导航但不让正文变目录 |
| R-42 | `161-163`，`Verdicts transfer because ... while ...` | 一个句子包含 mechanism、direction、domain、population threshold、interpretability | `拆`：第一句解释 mechanism fixes what/direction/domain；第二句说明 population only sets cutoff；第三句给 testability 推论 | 清楚完成逻辑解释 |

### Results 2.3，第 168-177 行：五机制总览与 MgAl2O4

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-43 | `168-172`，screen useful、law form、eight laws、five mechanisms | 两句重复 `mechanism`，列表过长 | `并/拆`：主题句定义 mechanism-resolved diagnosis；随后将五机制列表分成 geometry/packing 与 electrostatics/bonding/symmetry 两组 | 给读者一张机制地图 |
| R-44 | `173-174`，MgAl2O4 five damages 与 inverse-spinel 插入语 | 案例动作和背景注释同句 | `拆`：主句只说明五种 damages；inverse-spinel 作为独立短注 | 先建立案例，再补自然对应 |
| R-45 | `174-176`，compression、expansion、exchange 三种响应连续 | 信息密度高 | `拆`：contact change 一句；unchanged distance but like-charge bonds 一句 | 展示为什么单一距离不够 |
| R-46 | `176-177`，`so no single ... Each unsatisfied...` | 推论清楚 | `保留`，可把第二句作为段末 | 收束为 diagnosis |

### Figure 3 图注，第 182-196 行

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-47 | `190-196` panel c | 曲线、双轴、阈值、domains、0.1 eV line、SI 指针集中 | `拆`为三句：曲线与分布；law boundaries 与 0.1 eV；per-compound/control 的 SI 位置 | 按数据、阈值、扩展证据读图 |
| R-48 | panels a/b | 清楚 | `保留` | 不改 |

### Results 2.3，第 200-223 行：机制解释

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-49 | `200-211` Law 1 → Law 2 → Law 3 | 递进清楚，句子功能单一 | `保留` | 作为最强机制段之一 |
| R-50 | `213-215`，`Packing bounds still cannot ... so Law 4–6 move... and distinguish...` | blind spot、层级迁移和三种 failures 同句 | `拆`：先说明 fixed-coordinate exchange blind spot；再引出 electrostatic laws | 从 geometry 转入 electrostatics |
| R-51 | `215-216`，`Law 4 limits ... and Law 5 the largest...` | Law 5 省略谓语 | `改/拆`：两条 law 各自完整句，分别解释 range 与 maximum | 区分两条静电诊断 |
| R-52 | `216-217`，Law 6 domain 与 fixed-distance exchange | 清楚但可与前句平行 | `改`：以 Law 6 为明确主语，完整表达 like-charge condition | 完成三条 electrostatic laws |
| R-53 | `217-223`，Ewald limitation、Law 8、distance response、median values | 四层信息连续 | `分段`：Law 8 motivation 与 physical response 一段；experimental/damaged medians 一句收尾 | 从机制解释进入量化证据 |

### Results 2.3，第 225-233 行：Law 7 与应用问题

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-54 | `225-229`，定义 Law 7、damage action、Pauling relation | 三个命题在两句中密集 | `拆`：定义 threshold；说明 perturbation/element reassignment effect；最后说明 empirical chemical-order interpretation | 建立 site-complexity mechanism |
| R-55 | `229-230`，`Because ... detection gaps, damage detection accumulates...` | 名词化、主干靠后 | `改`：以 adding complementary mechanisms 为主语，说明逐级补上 detection gaps | 解释 Fig. 3b 的累积性能 |
| R-56 | `230-233`，diagnosis list 与 rhetorical question | 逻辑清楚 | `保留`；必要时在 rhetorical question 前断句 | 引向应用小节 |

### Results 2.4，第 238-251 行：进入 validation queue

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-57 | `238-239`，`The two criteria in use there ... same benchmark shows both` | `there/two criteria/both` 都需等到句尾才能解码 | `改`：句首直接点名 Pauling rules 与 distance floor，删除含糊回指 | 立即建立两个 baseline |
| R-58 | `240-243`，`We benchmarked ... cutoffs ..., which detected...` | `which` 可能修饰 pipelines 或 cutoffs | `拆`：比较设置一句；以 two distance cutoffs 为主语报告 1.6–3.2% | 分开方法和结果 |
| R-59 | `243-247`，Set 4、matched cutoff、chemical selectivity、positioning | 四层结论连续 | `拆`：Set 4 performance；matched comparison；chemical-selectivity inference 各一句 | 建立完整但易读的比较链 |
| R-60 | `247-250`，aggregate concern 与 class-resolved/withheld classes | 冒号后放入两类验证 | `拆`：先提出 single-class dominance；再分别写 selected classes 与 withheld classes | 排除单一扰动主导 |
| R-61 | `250-251`，`The same evidence can also be asked to screen...` | 英语搭配生硬 | `改`：直接提出 `We next asked whether...` | 过渡到 synthesizability |

### Results 2.4，第 255-276 行：PSS 定义

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-62 | `255-258`，plausibility barrier、continuous companion、fit pairs | 两句逻辑清楚但术语集中 | `改`：第一句明确 discrete PRIS → continuous PSS 的需求；第二句把 recorded/computed-only pairs 简化为同组成 pair | 建立 PSS 动机和训练对象 |
| R-63 | `269-271`，文字只列五类但公式有六项 | 读者可能误以为漏掉一项 | `改`：明确 six terms；说明 connectivity 提供 `kmax` 与 `fiso` 两项 | 文字与公式逐项一致 |
| R-64 | `271-274`，dominant coefficient、random halves、ranking/sign、correlation | 性质和稳健性检查挤在一起 | `拆`：dominant volume effect 一句；coefficient resampling 一句；term correlation 一句 | 分开 score meaning 与 robustness |
| R-65 | `275-276`，`stronger expression ... and testing it requires...` | `it` 指 score 还是解释不清 | `拆/改`：先定义 higher PSS；再明确 `Testing this interpretation requires...` | 平滑转入 PU proxy |

### Results 2.4，第 278-308 行：PU proxy 与线性关系

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-66 | `278-282`，数据库困难、CLscore、训练 pool | 文献背景与当前训练设置连接过快 | `拆`：scarce labels 一句；PU/CLscore 原理一句；当前 pool 一句 | 从问题进入解决方案 |
| R-67 | `282-286`，两个 scorer 的完整架构同句 | 非 ML 读者认知负荷高 | `拆`：CGCNN-PU 与 frozen-MatterSim PU 各一句，保持 learned representation / inherited representation 平行 | 解释两条不同路线 |
| R-68 | `286-289`，`Two such different routes ... make ... unlikely... The 364,592...` | 抽象推论和 proxy-set 定义衔接绕 | `改/拆`：先说 agreement reduces model-specific artefact；再定义 consensus-low set | 得到 hard-negative proxy |
| R-69 | `291-294`，Set 4 与 PSS matched-satisfaction comparison | 三个百分比同句 | `拆`：Set 4 一句；`At the same satisfaction` 起句给 PSS | 直接可比 |
| R-70 | `294-296`，hull threshold 的 performance 与额外计算要求 | 结果和代价同句 | `拆`：先报 72.0%；再说明 relaxation/phase-hull 与 direct screen 的差异 | 比较效果和路径 |
| R-71 | `296-298`，`agreement can manufacture a trend ... each trend below...` | 比喻式表达，`each trend below` 模糊 | `改`：直接点名 shared-selection risk；明确后文分别检查 CLscore–Set 4 与 CLscore–PSS | 解释逐模型验证目的 |
| R-72 | `300-303`，再次完整描述两个模型 | 与 282-286 重复 | `删/并`：改为 `Both PU models`，方法与图引用统一放句末 | 聚焦共同方向结果 |
| R-73 | `303-305`，两个趋势、端点和两个 R2 | 数值对应关系需回读 | `拆`：Set 4 trend 与 PSS trend 各一句；R2 紧跟对应分析 | 让数值映射清楚 |
| R-74 | `306-308`，population relation → individual polymorph test | 推理清楚 | `保留`，可把第二句作为新段首 | 引出 ranking |

### Figure 4 图注，第 313-337 行

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-75 | `313-314`，图题同时含 damage、synthesizability、inverse design | 三个独立任务并列 | `改`：以 physicochemical screening 为统领，删去可由 panels 表达的细项 | 给 Figure 4 单一问题 |
| R-76 | `319-322` panel c | axes、四组 laws、三类 curves、connectors 同句 | `拆`：坐标/曲线一句；matched connectors 一句 | 先识别内容，再识别关键比较 |
| R-77 | `323-326` panel d | series、shading、straight-line R2 同句 | `拆`为曲线、范围、fit 三句 | 顺序读图 |
| R-78 | `330-337` panel f | 主图、Set 4 point、inset、DFT dashed curve、scale conversion | `拆`为三句：threshold sweep；inset descriptors；DFT verification | 分清主分析、解释和验证 |
| R-79 | panels a/b/e | 清楚 | `保留` | 不改 |

### Results 2.4，第 341-355 行：同组成排序

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-80 | `341-345`，task difference、Set 4 ties、active refutation | 前两句清楚，第三句成孤立元话语 | `并`：将 `Active refutation was needed...` 与具体 tie-counting error 合并 | 直接进入错误和修正 |
| R-81 | `345-350`，tie error、symmetry explanation、audit、space-group medians、tie correction | 一个案例占六句，打断排名主线 | `压`成三步：错误计分；composition-controlled refutation；ties treated as no choice。详细中位数留 SI | 保留发现过程但缩短插曲 |
| R-82 | `347`，`A ... audit refuted it` | `it` 不清 | `改`：明确 refuted the symmetry-based explanation | 避免误解为推翻数据 |
| R-83 | `350-355`，overall accuracy、top fifth、PSS/DFT division、inverse-design transition | 结果和下一实验入口同句 | `拆`：overall；high-confidence；division of labour；下一实验各一句 | 从排序结果自然到 inverse design |

### Results 2.4，第 357-364 行：inverse-design 主结果

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-84 | `357-360`，MatterGen runs、1,081 candidates、UMA definition、140 target candidates | 任务设置、模型身份和结果连续 | `拆`：generation setup；UMA role；140 result 各一句 | 按任务 → proxy → subset 推进 |
| R-85 | `360-362`，`These structures supplied only...` | `These structures` 可指 1,081 或 140 | `改`：明确 `the 1,081 generated candidates`，再说明 two PSS terms 与 threshold calibration | 清楚限定输入集合 |
| R-86 | `362-364`，61 removed、all 140 retained、5.6% reduction | 三个等价/相关数字同句 | `拆`：screening count/reduction 一句；retention 一句 | 突出效率和保留 |
| R-87 | `364`，从 first-principles moduli 到 proxy ratio、one candidate、ranking、99.2% 全在同一源行 | 一个物理验证包含四个命题 | `分段/拆`：DFT campaign；proxy–DFT scale；pair ranking；99.2% retention 各句 | 建立独立的 DFT verification 段 |
| R-88 | `364`，`The proxy ran high by a median factor of 0.940` | `ran high` 与小于 1 的因子表面矛盾 | `改`：明确 ratio 的分子/分母，再说明 proxy 偏高 | 消除方向歧义 |
| R-89 | `364`，`Relaxation also revised what the screen had been reading` | 拟人化且突然切到 Law 7 | `分段/改`：新段直接写 DFT relaxation changed Law 7 classifications | 从 property verification 转入 structural change |
| R-90 | `364`，61→113、`not one moved the other way` | 两时点和方向同句 | `拆`：before、after、direction 三个短句 | 清楚展示变化 |
| R-91 | `364`，`The gain sat in...` | 搭配不自然，`gain` 模糊 | `改`：明确 all additional Law 7 passes occurred among PSS-retained candidates | 指明变化来源 |

### Results 2.4，第 366-379 行：被筛结构及参数可调性

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-92 | `366`，`what the 61 share` | 仅用数字回指，与前文多个 61 冲突 | `改`：写 `the 61 PSS-screened candidates` | 锁定集合 |
| R-93 | `367-371`，Law 7/open packing 与 Ir2Os7 两结构的四个指标 | 示例数值多但逻辑清楚 | `改`为严格平行句：screened candidate 的 symmetry/volume/PSS；retained candidate 同序列 | 让 pair comparison 易扫读 |
| R-94 | `373-375`，`they span two extremes` | `they/extremes` 抽象 | `改`：明确 one operating point barely shortens queue；the other shortens it strongly | 为 Set 1–3/Set 4 数字搭框架 |
| R-95 | `375-377`，PSS subset definition 与 140/140 retention | 条件和结果嵌套 | `拆`：先定义 PSS 在 Set 4 violations 中筛哪类；再报告保留全部 high-property candidates | 解释连续 score 的作用 |
| R-96 | `377-378`，`reduced ... by up to 67.3%` | 与前文 5.6% 相距较远，读者可能不清楚来自可调 operating points | `改`：保留 67.3% 强结论，同时明确这是 across tunable PRIS/PSS operating points 的最大 queue reduction | 把参数调节与 headline result 连起来 |
| R-97 | `378-379`，`PRIS states why ... PSS sets how strongly...` | 简洁、平行 | `保留` | 收束 PRIS/PSS 分工 |

### Results 2.5，第 386-398 行：从 queue 到数据库问题

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-98 | `386-391`，energy–phonon classes、recorded/unrecorded percentages、distance/relaxation/site splitting | 约 60 词且四层信息嵌套 | `拆`：百分比观察一句；distance/relaxation 仍会遗漏 site splitting 的解释一句 | 建立 Law 7 增量价值 |
| R-99 | `391-395`，GNoME 与 A-Lab controversy | Introduction 已完整讲过历史 | `压`为一句背景，只保留为何 site ordering 是关键问题及引用 | 避免 Results 重复 Introduction |
| R-100 | `395-398`，`remain enriched ... controversy turns on why ... diagnosis...` | `controversy` 重复，`turns on why` 含糊 | `删/改`：删除抽象尾语；直接以 Law 7 为主语说明它测量 geometric screens 遗漏的 site splitting | 快速进入可检验机制 |

### Figure 5 图注，第 403-429 行

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-101 | `404-408` panel a | axes、control、markers、offset 四类说明 | `拆`：数据对象/坐标一句；marker encoding 一句；offset 一句 | 顺序读图 |
| R-102 | `409-413` panel b | Upper/Lower 子图与三类标记同句 | `拆`：Upper 一句；Lower 一句；median/300 K 标记随 Lower 说明 | 两子图独立解码 |
| R-103 | `414-419` panel c | MLIP distributions、DFT curve、为什么不画 damaged DFT 三层 | `拆`：main distributions；DFT parent curve；population note | 分开图形内容与样本说明 |
| R-104 | `420-423` panel d | curves、medians、damage references、axis range 同句 | `拆`为 data、reference、axis 三句 | 降低图注负荷 |
| R-105 | `427-429` panel f | per-structure CPU 与 queue total 同句 | `拆`：下轴单结构；上轴整队列 | 区分两种尺度 |
| R-106 | panel e | 清楚 | `保留` | 不改 |

### Results 2.5，第 433-451 行：七个生成器与 GNoME

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-107 | `433-435`，seven generators sample | 清楚 | `保留` | 交代样本 |
| R-108 | `436-440`，distance result、Set 4 symmetry dependence、三组 satisfaction | 总结与数值同句过密 | `拆`：distance baseline 一句；symmetry dependence 一句；三组数值一句 | 先基线，再主现象 |
| R-109 | `440-445`，Law 7、relaxation、overlap、Wyckoff construction、design choice | 一个段落内完成证据与设计推论，句子过长 | `拆`：Law 7 evidence；overlap vs symmetry；shared generator construction；design inference | 从观察推进到设计原则 |
| R-110 | `445-451`，`The same failure ... GNoME ... sample ... rates ... strain alternative` | 从 generators 到 GNoME 没有段落分界 | `分段`于 446；第一句明确 `We next tested this pattern in GNoME`；随后 sample/result/alternative 各句 | 形成独立 GNoME evidence unit |

### Results 2.5，第 453-461 行：relaxation-energy 排除 gross strain

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-111 | `453-455`，MatterSim 150 parents 与三类 medians | 清楚但结果句稍长 | `拆`：parent release；compression/displacement medians | 建立 severity scale |
| R-112 | `455`，`To confirm ... 200 identical cells: ... 0.953, and twenty ... 0.0001` | 目的、样本、相关性、绝对量四层 | `拆`为验证目的/样本、rank correlation、parent median 三句 | 说明 why、agreement、scale |
| R-113 | `455-458`，MatterGen question 与 `Yet` 结果 | 答案延迟 | `改`：直接写 `The answer was no`，随后给 0.007/0.006 | 立即回答 small-energy 是否充分 |
| R-114 | `459-461`，gross strain inference → merge hypothesis | 逻辑正确但换问题较快 | `拆`：第一句收束 gross strain；第二句明确 site splitting remains；第三句提出 merge test | 自然转入标签干预 |

### Results 2.5，第 463-472 行：merge intervention 与 DFT ordering

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-115 | `463-466`，113/150、78%、79%、27 controls | 两项结果和 control 用 `latter` 回指 | `拆`：mergeable prevalence；bound restoration；space-group increase；experimental control 各句 | 逐层展示干预结果 |
| R-116 | `467-469`，sample conclusion、controversy、label-only change、`thermodynamically empty` | 结论与下一 DFT 试验挤在一句组 | `分段`：先用一句强结论收束 merge；再用新段引出 DFT ordering | 区分结构干预与热力学验证 |
| R-117 | `469`，23 entries/10 controls、0.0001/0.036、18/23/0 control | 方法、能量、温度结论同句 | `拆`：enumeration method；energy comparison；300 K counts 三句 | 建立清晰热力学证据链 |
| R-118 | `470-472`，critique → measured mechanism → `converse error` | 结论和下一主题过渡同句 | `拆`：第一句收束 ordering；将 complementary identity error 移为下一段主题句 | 从 ordering 进入 wrong element |

### Results 2.5，第 474-492 行：固定坐标元素身份

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-119 | `474-478`，Harrison history 与 Spek 发现过程 | 第二句历史细节较长 | `压`：保留 genuine diffraction/changed identity 与 coordinate checks 的必要背景；将发现过程细节缩为一从句 | 快速建立实验先例 |
| R-120 | `479-482`，reports may alter cells → fixed-coordinate exchange → three quantities → conclusion | 实验设计和机制结果连续 | `拆`：why fixed coordinates；which physical quantities change；why coordinate screen misses | 建立 controlled identity test |
| R-121 | `484-488`，six PRIS vs four check families 与两类 numerators | 比较设置清楚，数字句较密 | `拆`：comparison design；cation–cation；cation–anion 各句 | 平行报告两类交换 |
| R-122 | `489`，`A recovered archive ... agrees` | 从 benchmark 突然跳历史 archive | `分段`：该句作为新段首；改为明确 `The same mechanisms recur...` | 区分 controlled test 与 archive case |
| R-123 | `489-492`，four framework entries、two metrics、fifth entry | 样本与两项 law values 过密 | `拆`：four related entries；Law 1 result；Law 8 result；fifth entry 各句 | 逐项映射 archive 到 laws |

### Results 2.5，第 494-509 行：速度和层级

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-124 | `494-497`，Law 1、all eight laws、DFT queue time | 三个时间尺度连续 | `拆`：Law 1；eight-law implementation；DFT estimate 各句 | 形成成本梯度 |
| R-125 | `497-500`，millionth cost、three application settings、mechanism review | 一个结论句拖长 | `拆`：relative cost；where used；diagnostic value 各句 | 从速度到应用意义 |
| R-126 | `502-506`，thermodynamic/dynamic/synthesis distinction 与三组统计 | 三种集合和统计同句 | `拆`：on-hull imaginary modes；recorded metastability；4,271 unrecorded stable entries 各句 | 分开三类证据 |
| R-127 | `507-509`，plausibility precedes them | 核心层级句清楚 | `保留` | 收束 Results |

### Discussion，第 514-526 行：三个决策尺度

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-128 | `514-515`，`three levels ... entry to queue down to element on site` | 总领句漏掉 database level，空间方向绕 | `改`：直接并列 validation queue、database、individual site | 给后三层清晰地图 |
| R-129 | `515-522`，三次 `At the ... level` | 平行结构好，但每句略长 | `保留三句结构`；各句只保留一项机制和一项意义，删去 Results 已完整报告的解释 | 完成跨结果综合 |
| R-130 | `522-526`，independent layer、what it asks、precedes three checks、cannot collapse | 四层总结在两句中密集 | `拆`：establish layer；what it asks；why it precedes；why scores differ | 逐步给出概念结论 |

### Discussion，第 528-544 行：PRIS/PSS 分工

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-131 | `528`，`The first of these checks` | `first` 可能回指 queue 或 plausibility | `改`：直接写 structural-plausibility check | 明确主语 |
| R-132 | `528-531`，diagnosis、strictness、five mechanisms | 两句清楚 | `保留`，只消除重复 `mechanism` | 解释 PRIS 输出 |
| R-133 | `533-536`，PU link、PSS continuous score、five-term list | Results 中已定义 score，这里再次完整列举 | `压`：保留 unexpected link 和 continuous tunable score；删除完整 descriptor 列表 | Discussion 负责综合而非再定义 |
| R-134 | `536-537`，inverse-design dense packing/Law 7 detail | 重复 Results 局部机制 | `压/并`到 PRIS/PSS division of labour，一句说明 PSS weighs competing mechanisms | 形成跨结果综合 |
| R-135 | `537-542` 与 `539-542` 两次说明 PSS sets threshold、PRIS names mechanism | 相邻重复 | `并`：只保留一次最完整的 `PRIS bounds/names; PSS orders`；加入 measurable vocabulary 与 lower-cost outcome | 一次说清职责和价值 |
| R-136 | `542-544`，database/training labels 与 no-verdict | 从 inverse design 突然转应用 | `分段`并加 `Beyond queue screening`；数据库/训练用途一句，missing-input handling 一句 | 增补 deployment 用途 |

### Discussion，第 546-556 行：active refutation 收束

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| R-137 | `546`，`Confidence ... rests ... because active refutation...` | 两层因果嵌套 | `拆`：先说 confidence rests on retention process；再以 active refutation 为主语 | 进入方法学意义 |
| R-138 | `546-551`，572、2,037,606、11、8 再次完整出现 | 与 Results 搜索规模段重复 | `压`为一句规模锚点，保留所有强数字但不再次解释全过程 | Discussion 不重演 Results |
| R-139 | `551-553`，failure record、文献倡议、`This record supplies it` | `it` 指代抽象 | `改/并`：明确 preserved record provides systematic reporting of failed attempts；引用紧跟倡议 | 连接记录与领域需求 |
| R-140 | `553-554`，`no longer asks whether ... but explains why` | `asks whether/explains why` 不平行 | `改`：`does not stop at a pass/fail verdict; it explains why...` | 强而清楚的诊断结论 |
| R-141 | `554-556`，最终 autonomous-agents 句 | 强结论清楚 | `保留`；只检查插入语位置和句长 | 作为全文最后一句 |

## A5. `methods.tex`：Methods

### 开场段，第 3-7 行

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| ME-01 | `3-7`，`The analysis coupled ... The subsections below...` | 两句各自功能明确 | `保留` | 交代总体设计与 Methods/SI 分工 |

### Data sets and study design，第 11 行

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| ME-02 | `11`，`Threshold fitting and held-out assessment used splits assigned before any evaluation.` | 这是关键设计原则，但随后才出现更具体的 written protocols，信息顺序倒置 | `移/并`：以 written protocols 固定 split、criteria、allowed quantities 为段首；随后说明 split was assigned before evaluation | 先给预先固定原则，再给执行结果 |
| ME-03 | `11`，`We analysed 99,162 experimental structures...` | 清楚 | `保留` | 建立数据总体 |
| ME-04 | `11`，`For law discovery, a seeded hash ... discovery, held-out or reserve split.` | 清楚但需要紧邻 split 原则 | `移`到 ME-02 后；保留 seeded hash 和三种 split | 说明如何实现预先划分 |
| ME-05 | `11`，`Eligibility ... ionic structures, and splitting divided structures rather than compositions.` | eligibility 与 split unit 是两个独立设计决定 | `拆`：一句说明 ionic eligibility；一句说明 structure-level split | 避免读者误把二者当成同一限制 |
| ME-06 | `11`，`Every damaged structure ... inherited its parent's assignment.` | 关键防泄漏规则清楚 | `保留`并紧跟 split-unit 句 | 完成 parent–child split 逻辑 |
| ME-07 | `11`，`Written protocols fixed ... before evaluation.` | 位置过晚，重复段首 | 按 ME-02 `移`到段首并与首句合并 | 先声明 protocol lock |
| ME-08 | `11`，`Thresholds were fitted on 12,632 ... then assessed on 5,297...` | 四个样本数在一个句子中仍可读，但 discovery/held-out 应视觉分层 | `拆`：discovery sample 一句；held-out sample 一句 | 让拟合与评估一一对应 |
| ME-09 | `11`，`No threshold was fitted ... although those results informed...` | “未拟合”与“影响推进”是重要区别，`although` 容易被略读 | `拆`：第一句保留 no threshold fitted；第二句明确 held-out results were used only for advancement decisions | 清楚区分 parameter fitting 与 model-set selection |
| ME-10 | `11`，`After law-set selection, a split-labelling error ... so the reserve...` | 错误、操作、后果三层压在一句 | `拆`：先说明 reserve rows entered one full-sample fit；再说明 reserve 因此不再是 independent final test | 直接说明 incident 和 consequence |
| ME-11 | `11`，`Supplementary Note ... reports the error ... Database inventories...` | 两个 SI 导航用途不同 | `分段`：incident citation 留在 split 段末；inventory/licence citation 移到下一短句或段末 | 避免导航句干扰主设计链 |

### Structural descriptors and law-set evaluation，第 15-19 行

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| ME-12 | `15`，`On these data, each Law 1--Law 8 quantity ...` | 清楚 | `保留` | 声明逐律可评估原则 |
| ME-13 | `15`，formal oxidation states 与 external fallback 两句 | 主分析与外部分析规则清楚，但可形成显式对照 | `并/改`为 `In the primary analysis...; in the external analyses...` 两个平行短句 | 一眼看出两种 charge convention |
| ME-14 | `15`，`Native CIF oxidation annotations ... were excluded ... so that...` | 排除项与防 circularity 原因同句偏长 | `拆`：第一句列 exclusion；第二句说明避免 charge assignment 与 bond-valence evaluation 共用 bond length | 把规则与理由分开 |
| ME-15 | `15`，`Primary neighbours came from CrystalNN ..., and Law 7 used spglib...` | 两个互不相同的算法约定以 `and` 相连 | `拆`：neighbour assignment 一句；symmetry convention 一句 | 分开 coordination 与 symmetry |
| ME-16 | `15`，`On a separate 3,000-structure sample...` | 这是稳健性结果，插在定义段中打断方法链 | `移`到本小节末或对应 SI 导航前，作为 robustness check 独立句 | 先定义，再报告方法敏感性 |
| ME-17 | `15`，`Discovery used deposited cells, whereas ... primitive cells, so...` | 两种 cell convention 与比较限制同句 | `拆`：第一句说明 discovery/external conventions；第二句说明 rates containing Law 7 只在共同 convention 下比较 | 规则和其统计后果分开 |
| ME-18 | `17`，`Each law combines ...` | 清楚 | `保留` | 进入八条法则定义 |
| ME-19 | `17`，Law 1 的 shared coordinate、radius sum、repulsion、cutoffs、perturbation response 五句 | 信息顺序基本正确，但单段过密 | `分段`：先写 normalized contact 与 radius-sum meaning；再写 physical direction/cutoffs；最后写 perturbation response | 从定义到物理依据再到操作响应 |
| ME-20 | `17`，`The ionic-character conditions exclude ... Law 2 and Law 6 ... so Law 2 ... and Law 6...` | 两条法则的 domain restriction 和两个阈值/条件塞在一句 | `拆`：共用 ionic-domain 原则一句；Law 2 condition 一句；Law 6 domain 一句，并把 Law 6 细节移近其 exchange 机制 | 避免 Law 2/Law 6 交叉回读 |
| ME-21 | `17`，`The coordination condition of Law 3 ... because...` | 清楚 | `保留` | 定义 Law 3 的适用域和原因 |
| ME-22 | `17`，Law 4 与 Law 5 两句 | 两个概念可读，但应严格平行 | `改`：两句都采用“quantity/contrast → detects what”的同一语法；各保留一条机制 | 便于比较 global spread 与 worst-site instability |
| ME-23 | `17`，`Both respond ... A cation--anion exchange...` | 第一处说 wrong-site exchange，下一句才具体说明 cation–anion mechanism | `并/改`：先明确 coordinate-preserving exchanges；再用 cation–anion case 解释为何 distance matrix 不变而 electrostatics 改变 | 把现象和机制紧邻 |
| ME-24 | `17`，`Law 8 places ... ceiling at 0.7143 and thereby...` | 清楚但与前面 Law 4/5 的交换机制挤在同段 | `分段`：Law 8 单独一句成段；coverage citation 随后 | 结束 law-definition sequence |
| ME-25 | `19`，`The main benchmark rates ... and applications expose ... no-verdict outcome.` | benchmark denominator 与 deployment outcome 是两件事 | `拆`：一句说明 benchmark evaluates only available inputs；一句说明 applications report unavailable inputs as no verdict | 清楚区分 denominator convention 与 operational label |
| ME-26 | `19`，`When required charges or radii are absent ... and a structure that violates...` | missing-input 与 violation aggregation 以 `and` 相连，逻辑不同 | `拆`：missing inputs → no verdict for affected ionic laws；any evaluable violation → implausible | 定义两条独立 decision rules |
| ME-27 | `19`，SI 导航句 | 清楚 | `保留` | 指向完整定义与 lookup rules |

### Controlled damage and law selection，第 23-25 行

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| ME-28 | `23`，`The damaged structures came from five perturbations...` | 五项列表清楚 | `保留` | 定义 damage families |
| ME-29 | `23`，`Independent contact, electrostatic and MatterSim-relaxation controls...` | 清楚 | `保留` | 说明 severity controls |
| ME-30 | `23`，`For each mechanism ... Law selection maximised...` | 从机制语义直接跳到 optimization objective | `分段`：mechanism interpretability 句收束 damage design；selection objective 作为新段首 | 分开 damage construction 与 law selection |
| ME-31 | `23`，`Fixed Set 4 was then evaluated once ... and compared with...` | 一句包含 one-shot evaluation 与 optimal-tree comparator | `拆`：held-out Set 4 evaluation 一句；depth-three tree comparison 一句 | 分开 evaluation target 与 comparator |
| ME-32 | `23`，`Re-derived at the held-out percentiles, Law 4 moved ... and Law 5 ... and these shifts altered...` | 两个阈值变化和两个 verdict impacts 过密 | `拆`：Law 4 change；Law 5 change；two verdict impacts 三句 | 读者能对应每条 law 和影响 |
| ME-33 | `23`，`For leave-one-damage-class-out ...` | 新验证协议藏在长段中 | `分段`于此；先说明 tree/single-threshold fully repeated | 明确进入 transfer test |
| ME-34 | `23`，`For Set 4, only its additions ... exposed to all five classes.` | 关键限制清楚但句长 | `拆`：fixed Set 3 base 一句；only additions reselected 一句 | 说明 LOFO 的冻结/重选边界 |
| ME-35 | `23`，`That variant therefore tests ... rather than...` | 清楚 | `保留` | 准确定义测试含义 |
| ME-36 | `23`，LOFO 的 51% 与 68.6–100% 两句 | 两种结果对象不同 | `保留两句`，在第二句显式点名 Set 4 additions | 报告单 law 与 additions 的 transfer |
| ME-37 | `25`，distance-cutoff benchmark 的 sample、结果和 SI citations | 基本清楚；三项数值应保持平行 | `保留段落结构`；将 99.1%/26.8% 与 33.9% 的主语统一为 method/cutoff | 便于直接比较 benchmark outcomes |

### External evaluation, PSS and statistics，第 29-37 行

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| ME-38 | `29`，`With PRIS thresholds fixed, external evaluation covered...` | 六类外部分析连续罗列 | `拆`成“generated/computed structures”与“coordinate/history/relaxation diagnostics”两句；保留 fixed thresholds 在首句 | 给外部验证建立分类 |
| ME-39 | `29`，generator cutoff 与 Law 7 rates 两句 | 方法基线和主要结果清楚 | `保留`，但在第二句先明确 same relaxed benchmark/control convention | 防止不同 denominator 被误读 |
| ME-40 | `29`，GNoME sample protocol 与 (Pm/Cm) rates 两句 | 从 seven generators 直接切 GNoME | `分段`，以 `We next sampled GNoME...` 为新段首；protocol 与 result 各一句 | 建立独立 GNoME evidence block |
| ME-41 | `29`，MatterGen relaxation-energy comparison 一句 | sample、three verdict classes 同句 | `拆`：comparison purpose/sample 一句；134/11/355 outcomes 一句 | 区分 analysis 和 classification result |
| ME-42 | `29`，26,600 Materials Project structures 句 | 与前一句没有显式关系 | `分段/改`：明确该 cohort 用于 thermodynamic–dynamical–record comparison | 给数字一个任务主语 |
| ME-43 | `31`，same-composition ranking 两句 | 表达清楚 | `保留`；只把 “selected no unique structure” 和 pair-level distinction 保持为两个指标 | 避免 composition-level/pair-level 混读 |
| ME-44 | `33`，`PSS was fitted ... as an antisymmetric, zero-intercept logistic score...` | 两个统计术语对跨学科读者不透明 | `拆`：先给模型形式；再用一句白话说明 swapping pair order reverses the score and equal structures have zero offset | 解释为什么用 antisymmetric/zero-intercept |
| ME-45 | `33`，`Each development pair compared...` | 清楚 | `保留` | 定义 training pair |
| ME-46 | `33`，六个 descriptor 的五句 | 定义完整，但一段内名词密度过高 | `分段`为 packing/electrostatics/bond valence 与 site/connectivity 两组；每句先给通俗功能，再给符号 | 让非本领域读者先理解意义再读公式 |
| ME-47 | `33`，`Descriptors are standardised ... and unavailable descriptors ... medians.` | standardization 与 imputation 是两项处理 | `拆`为两句 | 明确 preprocessing steps |
| ME-48 | `33`，`The score was then frozen ... before transfer...` | clear | `保留`并作为段末 | 完成 fit → freeze → evaluate → transfer 顺序 |
| ME-49 | `35`，`Building on ... expanded ...` 与 `In previous work...` | 现工作先讲、前期工作后讲，时间线倒置 | `移` prior-work 句到段首；下一句写 current expansion | 先 lineage，再 current study |
| ME-50 | `35`，`Consensus between a retrained 50-bag ... and 50 MLP ... defined...` | 两个模型架构、bag 数和 consensus outcome 同句 | `拆`：CGCNN-PU 一句；MatterSim-MLP-PU 一句；consensus tail 一句 | 分清 models 与 selection rule |
| ME-51 | `35`，deduplication 与 AUC 两句 | selection result 与 validation performance 混在同段 | `分段`于 validation；先 364,592 unique structures，再分别报告两模型 AUC | 从 proxy construction 进入 model validation |
| ME-52 | `35`，`At matched ... PSS screened 31.8 ... Trends were aligned ... and a balanced comparison...` | PSS result、percentile alignment 和 hull comparator 三层连续 | `拆`：matched-satisfaction result；alignment method；balanced comparator 各句 | 先结论，再说明公平比较方式 |
| ME-53 | `35`，`In that comparison, the hull threshold retained 86.2%...` | 清楚 | `保留` | 完成 comparator result |
| ME-54 | `37`，MatterGen generation/UMA 句 | 两个阶段清楚但同句偏长 | `拆`：13-run generation/deduplication；UMA property proxy | 分开生成与独立属性评估 |
| ME-55 | `37`，`A set of 541 ... fixed threshold, retaining 528 ... before application to 1,081...` | calibration、retention、application 三层 | `拆`：calibration cohort；528 retained；threshold applied to generated set | 清楚区分 threshold setting 与 deployment |
| ME-56 | `37`，frozen medians 句 | 清楚 | `保留` | 说明 missing-term handling |
| ME-57 | `37`，61 removed vs retained volume 句 | 对比清楚 | `保留` | 报告 screening direction |
| ME-58 | `37`，Set 4 45/140 与 PSS 61-of-728/95 retention 两句 | strict-set result 与 tunable-screen result 挤在一起 | `分段`：先 Set 4 outcome；再 PSS-screened subset；最后单句突出 all 95 retention | 展示 hard rules 与 PSS 的不同角色 |
| ME-59 | `37`，长 SI 导航句 | 五个主题混列 | `改`：按 data/model validation 与 inverse-design/DFT/statistics 两组引用；不重复主文结果 | 清楚指向复现信息 |

### First-principles verification，第 41 行

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| ME-60 | `41`，`Four quantities ...: contact bounds ..., ordering ..., damage-severity ..., property target...` | 开场句同时含方法原则和四项清单 | `拆`：第一句说明 four learned quantities were re-derived under a pre-fixed DFT protocol；第二句并列四项，必要时用分号 | 先目的，再列对象 |
| ME-61 | `41`，VASP/PBE/cutoff/k-mesh/EDIFF/smearing/relaxation/five volumes | 八项设置在一个超长句 | `拆`：code/pseudopotential/cutoff 一句；k-point/convergence/smearing 一句；relaxation/volume sampling 一句 | 降低技术参数负荷 |
| ME-62 | `41`，`Each first-principles quantity ... follows from those runs.` | 元话语，无新增操作信息 | `删/并`入开场句 | 减少自我指涉 |
| ME-63 | `41`，reduced-contact landscape 句 | 定义清楚 | `保留` | 对应 contact bounds |
| ME-64 | `41`，relaxation-energy release 句 | 定义和 DFT/ML potential comparability 同句 | `拆`：quantity definition 一句；identical cell-by-cell computation 一句 | 分开“是什么”和“为何可比较” |
| ME-65 | `41`，order–disorder temperature 句 | numerator、denominator、enumeration procedure 三层嵌套 | `拆`：energy numerator；configurational-entropy denominator；symmetry-distinct ordering relaxation procedure | 让 estimator 可顺序复现 |
| ME-66 | `41`，bulk modulus 句 | 清楚 | `保留` | 对应 design property |
| ME-67 | `41`，1,917 tasks、SI、260 CIF 三句 | 清楚但属于 campaign/archive 信息 | `分段`：独立结尾段，先 campaign size，再 SI protocol，再 data availability | 与 quantity definitions 分离 |

### Autonomous agents, human oversight and reproducibility，第 45 行

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| ME-68 | `45` 全段 | 七句依次说明 boundaries、agent actions、human actions、confirmation、responsibility、records、reproduction；逻辑完整 | `保留`；仅在实际改写时检查 `agents/agent instances` 的用词一致性，不删任何 oversight 句 | 作为透明度声明 |

## A6. `si.tex`：SI wrapper

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| SW-01 | `64-66`，SI title | 需要与最终主文标题完全同步 | 主文标题确认后逐字同步；不单独改写第二个版本 | 避免主文/SI 标题漂移 |
| SW-02 | `84-86`，`This document provides the data sources ... exact procedures ... full record...` | 五类材料连续堆叠 | `拆`：第一句只列 sources/licences、definitions、damage protocol；第二句列 search/testing 与 eleven refuted claims | 建立“定义与协议 → 审计记录”结构 |
| SW-03 | `86-89`，`It also documents the data-split incident ... additional tests ... PSS ... applications...` | incident、reproduction、validation、PSS、applications 五层同句 | `拆`：incident/reproduction 一句；additional physical tests 一句；PSS/applications 一句 | 用三句给 SI 内容地图 |
| SW-04 | `68-82, 91-104`，作者、单位、目录和 bibliography wrapper | 无语言问题 | `保留` | 不改排版控制代码 |

# B. Supplementary Information 逐段逐句计划

以下只审阅 `si_body.tex` 的论文文字。表格中的纯数值单元、LaTeX 控制命令和引用键不作为语言修改对象；表题、图注和解释句仍逐项列出。

## B1. Supplementary Note S1：agent record、研究边界与审计

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| S1-01 | `42-44`，`This is the sense in which ...` | 元话语先解释“本文所说的意义”，主结论出现太晚 | `改`为直接句：autonomy is established by preserved proposals/tests/refutations rather than post-hoc narration | 直接说明可核验依据 |
| S1-02 | `48-51`，investigation dates 与 `All subsequent investigations...` | dates/period 在相邻句重复 | `并`：只保留一次 time window；随后直接引 Table S1 | 压缩时间说明 |
| S1-03 | `72`，Table S1 中 structure counts、PU subsets 同一单元 | 两类数据用途混在一格 | `拆`为结构-law study 与 PU/PSS study 两项 | 区分两条分析线 |
| S1-04 | `75`，Set 4 与 PSS success criteria 同一表格句 | fixed-law 与 continuous-score criteria 不平行 | `拆`成两行，并明确各自 outcome/constraint | 便于核对预设标准 |
| S1-05 | `84-86`，`An attempt strip ... The audit separated...` | attempt strip 定义与 lane 分类连在一起 | `拆`：先定义 one attempt；再列 discovery/confirmation/exploration lanes | 先单位，再分类 |
| S1-06 | `86-89`，175 attempts、131/37/7 和 no-mark 说明 | 总量、标记数、未标记原因挤在一句组 | `拆`：总数一句；三类数量一句；无 mark 原因一句 | 清楚交代审计 denominator |
| S1-07 | `89-92`，adjacent-screening exclusion 与 optimal-tree marking | 两条 audit rule 不相干 | `拆`为两句，并分别说明影响的 attempt 类型 | 避免规则串联 |
| S1-08 | `94-98`，`What is counted ... scope ... traceability` | definition、scope、traceability 三层连续 | `拆`：count unit；not counted；how linked to files/logs | 可核验地定义 attempt count |
| S1-09 | `98-100`，2,037,606 与 8,466 adjacent screens | 总搜索数与相邻 screening 次数容易被当成同一统计 | `改`为显式对照：total hypotheses vs local follow-up screens | 区分 global 与 local search |
| S1-10 | `100-104`，pre-confirmation 与 post-confirmation failures | 时间阶段与失败性质嵌套 | `拆`成 pre-confirmation failures 和 post-confirmation checks 两句 | 建立时间线 |
| S1-11 | `108-112`，多次 `caught`/`later caught` | 同一审计作用反复表达 | 保留第一次；将 later 11 failures 压为一条结果句 | 减少修辞重复 |
| S1-12 | `112-114`，missing diagnostic → checklist | 先说 checklist 再说缺失诊断，因果反序 | `移` missing diagnostic 到前；用 `This omission motivated...` 引 checklist | 缺口自然导向改进 |

## B2. Supplementary Note S2：data inventory 与 split flow

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| S2-01 | `175-178`，arrow chain 描述 data flow | 箭头式文本把 processing flow 与 sample coverage 混在一起 | `改`为两句：第一句按顺序写 flow；第二句给各阶段 coverage | 可读地表达数据管线 |
| S2-02 | `185-188`，trajectory frames、sample counts、endpoints | trajectory sampling 与 endpoint definition 同句 | `拆`：frame sampling；number selected；which endpoint retained | 交代时间序列取样 |
| S2-03 | `192-193`，`Composition of failures:` | 句子碎片 | `改`为完整句，并明确下列比例以什么为 denominator | 修复语法与统计指代 |
| S2-04 | `214-217`，coverage definition 与 numbers | 定义和结果挤在一句 | `拆`：先定义 evaluable coverage；再报告各 law/set 数值 | 先指标后结果 |
| S2-05 | `217-220`，fixed laws applied without refit 与 comparative outcomes | protocol 与 result 同句 | `拆`：第一句强调 no refitting；后续句报告 source-specific rates | 分开迁移规则和表现 |

## B3. Supplementary Note S3：descriptor conventions 与八条法则

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| S3-01 | `224-226`，开头 fragment、CrystalNN 与 valence source | 定义以句子碎片开头，两个来源混列 | `改`为完整引导句；neighbour source 与 valence source 各一句 | 建立 conventions 地图 |
| S3-02 | `228-231`，四条 Pauling criteria | 四项句法不平行 | `改`为严格平行编号列表，每项均采用“input/condition → test” | 便于读者扫描 |
| S3-03 | `231-234`，missing convention 在 satisfaction rule 后 | 读者先看到 verdict 才知道缺失值处理 | `移` missing-input convention 到 satisfaction definition 前 | 先定义可评估性再定义通过 |
| S3-04 | `235-237` 附近，T0/T1/T3 表格直接出现 | 标签首次出现缺乏一句话解释 | 表前增加一句分别说明 T0、T1、T3 的角色；不改数值 | 让表格可独立阅读 |
| S3-05 | `271-272`，`existing structures` 重复 | 同一来源/集合相邻复述 | `并`为一次精确命名 | 去除词语重复 |
| S3-06 | `279-282`，missing-value retention reason | 规则、保留原因和 denominator 后果在一句 | `拆`为 missing policy、why retained、how counted | 防止误解为 imputation |
| S3-07 | `287-289`，`The implementation follows ...` | 元话语先行，具体 lookup sequence 后置 | `删`空泛开头，直接列 fixed lookup order | 提高操作清晰度 |
| S3-08 | `292-296`，CN fallback、element-radius fallback、fixed radius | 三层 fallback 嵌套 | `拆`为按优先级排序的三句或编号列表 | 可复现地表达 hierarchy |
| S3-09 | `299-303`，bond-valence source/key 与 parameter selection | 数据来源和选择算法同句 | `拆`：source/key 一句；lookup/selection rule 一句 | 分开数据与算法 |
| S3-10 | `305-307`，emitted condition 与 average calculation | output condition 与 statistic 计算同句 | `拆` | 明确何时有值、值如何计算 |
| S3-11 | `308`，`implemented verbatim` 类元话语 | 无新增定义 | `删`或并入具体公式句 | 减少自我说明 |
| S3-12 | `312-315`，primary 与 robustness neighbour algorithms | 两套算法连续罗列 | `改`为 primary convention 一句、robustness alternatives 一句 | 清楚区分主分析与敏感性分析 |
| S3-13 | `315-317`，`32x` 与三个 exact values | 倍数和数值挤在一起，`percentage points` 省略 | `拆`：先报告 exact rates；再用完整 `percentage points`/fold change 解释 | 避免单位含混 |
| S3-14 | `317-320`，`does not license...` | 以否定许可方式表达结论，绕 | `改`为直接陈述该 robustness check 支持的具体解释 | 正面说明证据含义 |
| S3-15 | `324-328`，完整定义和 SI navigation | 清楚 | `保留` | 收束 conventions |

## B4. Supplementary Note S4：search defect 与修复

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| S4-01 | `338-341`，model form、objective、constraints | 三项搜索定义在一句 | `拆`：candidate model；objective；constraints | 可顺序理解 search specification |
| S4-02 | `345-348`，search defect 与 consequence | bug mechanism 和统计后果混在一起 | `拆`：what failed；which candidates remained/left | 先缺陷再影响 |
| S4-03 | `348-349` 与 `369-371`，complete beam allocation 重复 | 同一资源分配说明出现两次 | `移`完整说明到 defect 段；后一处删重复只保留结果 | 去除重复 |
| S4-04 | `353`，小标题 `--ban did not ban` | 口语化且需读正文才知对象 | `改`为直接技术标题：ban-filter failure and correction | 一眼知道问题 |
| S4-05 | `353-355`，composite key 与 filtering failure | implementation detail 和 outcome 同句 | `拆` | 说明 bug 如何发生 |
| S4-06 | `361-364`，candidate triple、per-column handling、conditional logic、T0/T1 | 四层逻辑过密 | `拆`成 ordered procedure，并显式标出 T0/T1 | 可复现修复步骤 |
| S4-07 | `366-369`，three operating points | 三项定义句法不平行 | `改`为同一模板：threshold source + intended use | 便于比较 operating points |
| S4-08 | `391-395`，Fig. S3b axes、labels、references | 图注同时解释坐标、颜色和参照线 | `拆`为 panel content、encoding、reference 三句 | 图注可快速解析 |

## B5. Supplementary Note S5：damage controls

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| S5-01 | `438-441`，missing repulsion 与 `bl_min` supplement | missing descriptor 和补充 contact measure 混在一句 | `拆`：为什么 primary metric unavailable；如何用 `bl_min` 补充 | 交代 fallback 依据 |
| S5-02 | `443-446`，S6 decision、two probes、design meaning | 决定、验证和含义三层 | `拆`为 decision → probes → implication | 展示 control 如何支持设计 |
| S5-03 | `446-447`，R1/S6 distinction | 与前句重复两者差异 | `并`入 S5-02 末句 | 一次说明两种 control |
| S5-04 | `449-451`，topic、geometric、electrostatic labels | 三类解释词没有并列语法 | `改`为平行三项 | 明确每项 probe 的物理对象 |
| S5-05 | `451-454`，D5 gain 与 other-class maximum | 主结果和比较基线同句 | `拆`：D5 improvement；largest non-D5 change | 突出 specificity |

## B6. Supplementary Note S6：refuted claims

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| S6-01 | `485-492`，complete ledger、Fig. S1、Fig. S2、machine ledger | “完整记录”连续重复但三种载体功能未区分 | `并/改`：Fig. S1=overview，Fig. S2=timeline，machine ledger=full trace | 用一次表述说明三层证据 |
| S6-02 | `496-498`，equality observation 与 past interpretation | observation 和曾经错误解释同句 | `拆` | 分开事实与被否定推论 |
| S6-03 | `509-513`，general principle、Pauling example、decision/tie rates | 原则、例子、数值三层 | `拆`为三句 | 从 refutation rule 到例证 |
| S6-04 | `513-517`，scoring error、corrected result、symmetry refutation | 两个不同 refutation 串在一起 | `拆/分段`：先 correction and result；再 symmetry claim refutation | 避免案例混杂 |

## B7. Supplementary Note S7：distance-cutoff benchmark

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| S7-01 | `524-525`，benchmark setup fragment | 句子碎片 | `改`为完整主谓句，并明确 sample 与 methods compared | 可独立理解 benchmark |
| S7-02 | S7 其余段落 | sample、cutoffs、Set 1/Set 4 matched comparison 结构清楚 | `保留` | 不作无必要改写 |

## B8. Supplementary Note S8：protocol、split incident 与 reserve

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| S8-01 | `560-563`，preregistration/tag/commit/seed fragments | 名词短语不是完整句 | `改`为两到三句完整陈述：protocol location/version；code commit；seed/split assignment | 给 reproducibility items 明确谓语 |
| S8-02 | `568-569`，R1/R2 与 change timing | 两个 revision 的作用同句 | `拆`：R1 content；R2 content and timing | 建立 protocol timeline |
| S8-03 | `579-581`，authorization date 与 V4 definition | 授权事件和 model version 定义混在一起 | `拆` | 区分 governance 与 analysis object |
| S8-04 | `581-583`，two success criteria | 两项标准语法不平行 | `改`为相同模板并显式写 numerator/threshold | 便于审计 |
| S8-05 | `583-587`，three effects、direction、decision | incident 的三个后果塞在一句组 | `拆`：effect 1/2/3；direction；decision | 明确错误影响链 |
| S8-06 | `588-591`，control values 与 mechanism | 数值和为何构成 control 同句 | `拆`：先数值；再机制解释 | 读者先看到证据再理解含义 |
| S8-07 | `592-596`，three datasets、method meaning、remaining reserve | 数据集合与独立性结论过密 | `拆`：分别命名 discovery/held-out/reserve status；最后一句说明 what remains valid | 清楚标出各 split 当前地位 |

## B9. Supplementary Note S9：reproduction

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| S9-01 | `611-638`，commands 与 public benchmark | 命令注释简洁，benchmark licence 句清楚 | `保留` | 不改可执行命令或许可含义 |

## B10. Supplementary Note S10：exact shallow-tree search

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| S10-01 | `642-643`，Purpose | 清楚 | `保留` | 定义 exact-search question |
| S10-02 | `645-648`，discovery matrix、STreeD、quantile binarization | 数据规模、算法全称、参数同句过密 | `拆`：matrix 一句；algorithm/class 一句；binarization settings 一句 | 先数据，再算法，再离散化 |
| S10-03 | `648-651`，candidate splits、search class、runtime | 原理、model-class relation 和 runtime 依次挤在同段 | `分段`：quantile meaning；class containment；runtime 各句 | 说明 exactness 范围与计算代价 |
| S10-04 | `653-656`，objectives、identical tree、cost ratio、sample replication | objective invariance 与 cost-ratio control 同句组 | `拆`：F1/accuracy result；what moves curve；how implemented | 清楚解释 trade-off control |
| S10-05 | `677-678`，omitted-class test setup | 清楚 | `保留` | 引出表格 |
| S10-06 | `697-702`，single threshold、mean absolute change、signed-mean cancellation、interpretation | 四层比较连续 | `拆`：threshold rates；define mean absolute change；tree/threshold values；why signed mean misleads | 逐步解释稳定性指标 |
| S10-07 | `704-708`，`Two corroborating details` 后的 tree features、loss specificity、interpretation | 两个 details 未明确编号，第一句列表很长 | `改`为 `First`/`Second` 两段；feature list 保持一处；第二段说明 class-specific loss | 让标题与正文对应 |
| S10-08 | `710-717`，replication 与 supported conclusion | 两段清楚 | `保留` | 报告独立复现和模型类内强结论 |

## B11. Supplementary Note S11：one-sided contact criterion

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| S11-01 | `721-725`，what lower bound misses | 清楚 | `保留` | 定义 expansion blind spot |
| S11-02 | `727-729`，grid scan 与 matched-satisfaction fairness | 扫描范围和公平比较原则同句 | `拆`：grid definition；comparison at same satisfaction | 先说明搜索，再说明比较 |
| S11-03 | `750-752`，upper bound snaps、0.99-row error/R11 | threshold behaviour 与历史错误同句组 | `拆`：第一句只报告 abrupt change；第二句解释只报 0.99 row 导致 R11 | 从结果到 refutation |
| S11-04 | `754-755`，标题 `What the upper tail actually is` 与句子 | 标题口语化，句子用 “not exotic ... but ...” 绕 | `改`标题为 `Chemical character of the upper tail`；正文直接说 upper-tail structures lie outside ionic domain | 直接命名发现 |
| S11-05 | `774-776`，radius-model applicability、charge filter、anion figure | 三句关系略跳 | `并/改`：说明 charge balance alone does not establish ionic-domain applicability；再引 anion analysis | 连接 domain mismatch 与后续分析 |
| S11-06 | `778-783`，`Hence` 后直接断到 displayed law | `Hence` 构成句子碎片 | `改`为完整引导句：these distributions motivate the conditional law below | 语法完整地引公式 |
| S11-07 | `823-824`，full law-set sequence | 清楚 | `保留` | 收束 expansion result |
| S11-08 | `826-827`，`Selecting ... only:` | 冒号前是句子碎片 | `改`为完整句，明确 condition/threshold were selected on four retained classes | 独立说明 LOFO setup |
| S11-09 | `860-862`，same threshold/stability/omitted result | threshold stability 与 performance 含义同句 | `拆`：same threshold in five repeats；omitted-type result and what it isolates | 区分 selection stability 与 transfer |
| S11-10 | `864-868`，stated limitation | 四句逻辑清楚 | `保留`，只把 `It` 明确回指 leave-one-type test | 避免代词歧义 |
| S11-11 | Fig. S7–S10，`873-916` | 图注内容清楚但部分单句承载 axes、encodings、reference lines | 每个 panel 按“what is plotted → encoding → reference”三句模板轻拆；不改任何 figure meaning | 统一 SI 图注节奏 |
| S11-12 | Fig. S11 panel a，`923-931` | compounds、axis、colour、three reference bounds、结论同段 | `拆`为 data/axes；colour；horizontal/vertical references；observed rise 四句 | 读者先读图再读结论 |
| S11-13 | Fig. S11 panel b，`931-935` | potentials、两轴、diagonal、annotation 同句 | `拆`为 comparison；axes/diagonal；annotation | 解释 hard-potential control |
| S11-14 | Fig. S11 panel c，`935-940` | crossing definition、normalization、boxes、1.80× interpretation 一句到底 | `拆`为 coordinates；normalization；box encoding；1.80× result/meaning | 让图注中的核心结果突出 |
| S11-15 | Fig. S12–S13，`947-969` | 基本清楚 | `保留`；只统一 panel sentence 起始结构 | 不作实质改写 |

## B12. Supplementary Note S12：damage operators

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| S12-01 | `974-995`，D1–D5 procedure table | 表格结构清楚 | `保留` | 保持 exact operators |
| S12-02 | `998-1001`，D2 wrong swap、charges unchanged、zero energy、relabeling | implementation error、zero consequence 与 interpretation 同句过长 | `拆`：original error；zero-energy result；why it was only relabelling | 清楚重现 R1 因果链 |

## B13. Supplementary Note S13：composition-grouped ranking

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| S13-01 | `1005-1007`，pairing rule、group/structure/pair counts、eligibility | 规则、样本量和 restriction 同句 | `拆`：pairing rule；ionic restriction；counts | 定义 ranking cohort |
| S13-02 | `1009-1013`，usable comparison、tie、pair decision rate、group tie rate | pair-level 与 group-level definitions 连续 | `拆/分段`：pair verdict definitions；pair decision rate；group tie rate | 防止层级混淆 |
| S13-03 | `1012-1017`，accuracy、ties、Top-1 hit、baseline、lift | 四个指标在一段内密集 | `拆`：conditional accuracy；tie handling；Top-1 definition/baseline；lift | 每个 metric 一句 |
| S13-04 | `1019-1022`，within-group statistics、equal weighting、different contributing groups | aggregation 和 noncommon cohort 同句 | `拆`：group-wise calculation；equal weighting；criterion-specific contributing groups | 交代 denominator difference |
| S13-05 | `1022`，`Both conventions are essential...` | `Both` 指代不清，前面超过两项 convention | `改`为直接点名 tie handling 与 group-equal weighting | 消除代词歧义 |
| S13-06 | `1022-1026`，77.7% ties error 与 SiO2 pair weighting reversal | 两个不同错误案例挤在一个段落 | `分段`：R10 tie-scoring 一段；R9 SiO2 weighting 一段；每段先 error 后 corrected comparison | 每个审计案例独立 |
| S13-07 | `1028-1031`，cluster bootstrap、reason、pair resampling consequence | 方法与选择理由同句过长 | `拆`：resampling unit/procedure；within-group correlation reason；pair-resampling consequence | 明确为何 cluster bootstrap |
| S13-08 | `1031-1032`，`Folds are split... B=500, seed...` | 最后一句是参数碎片 | `改`为完整句，并把 fold split 与 bootstrap settings 分开 | 复现信息语法完整 |

## B14. Supplementary Note S14：figure conventions

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| S14-01 | `1034-1037` 全段 | 三句简洁 | `保留` | 说明 figure style 与 source data |

## B15. Supplementary Note S15：PSS fitting、evaluation 与 stability

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| S15-01 | `1041-1047`，PSS definition 与 why new analysis | 清楚 | `保留` | 从 binary law 转 continuous ranking |
| S15-02 | `1049-1052`，programme relation、date、reserve event、reuse、statistics | lineage 和 reuse 信息过密 | `拆`：same programme/date 一句；relation to reserve event 一句；reused data/code 一句；statistics inclusion 一句 | 清楚交代 provenance |
| S15-03 | `1054-1058`，initial plan、revision 1、revision 2 | 两次 revision 挤在一个长句 | `拆`：initial preregistration；R1 trigger/change；R2 change/timing 各句 | 建立版本时间线 |
| S15-04 | `1058-1060`，four baseline reproduction | 清楚 | `保留` | 声明 new row 前的 reproduction |
| S15-05 | `1062-1064`，development/held-out groups、structures、pairs | 两个 split 的三种计数用斜杠串接 | `拆`为 development 与 held-out 两句或小表 | 一眼对应 split counts |
| S15-06 | `1064-1068`，SiO2 placement、dominance implication、single evaluation | 三层清楚但应分段 | `分段`：composition-dominance control；one-shot held-out evaluation 独立句 | 区分 cohort property 与 evaluation discipline |
| S15-07 | `1070-1077`，85-feature exclusions 与 nine additions | exclusion、addition 和九项 feature 大列表同句 | `拆`：excluded families 一句；added symmetry features；repulsion/strain/density features | 按物理功能组织 feature pool |
| S15-08 | `1077-1079`，no DFT、charge/radius conventions | 清楚 | `保留` | 定义 low-cost inputs |
| S15-09 | `1081-1086`，model class、standardization、weighting、forward selection、refit/archive | 流程正确但单段五步密集 | `拆`为 model form；preprocessing/weighting；selection/stopping；final refit/hash | 顺序展示 training pipeline |
| S15-10 | `1088-1092`，forward-selection sequence 与 stopping | 六步数值使用一条冒号长串 | `改`为编号/分号列表；最后单句说明 seventh candidate below 0.002 | 可扫描地展示增量 |
| S15-11 | `1092-1097`，all-feature model、hull、archive、leading singles | benchmark results 与 archive inventory 混在一起 | `拆/分段`：models/hull scores；archive statement；leading single-feature audit | 分开性能比较与可复现材料 |
| S15-12 | `1099-1100`，experimental records | 清楚 | `保留` | 回指 S13 label convention |
| S15-13 | `1102-1108`，PSS/full/hybrid/reference scores/bootstrap differences | 六组结果在一段中堆叠 | `拆`：three model scores；four baselines；paired differences | 先模型，再基线，再 uncertainty |
| S15-14 | `1108-1112`，decision rate、best structural single、Top-1、not overtaking hull、wording | 五个结论同句组 | `拆`：decision/structural comparison；Top-1 comparison；overall hull comparison；main-text wording | 保留全部强结果且避免吞数值 |
| S15-15 | `1112-1114`，development-to-held-out drop 与 evaluation log | 解释与审计记录清楚 | `保留` | 说明 repeated-development optimism |
| S15-16 | `1116-1119`，confidence heading 与 parentheses 中的 metric/CI/archive | 关键统计定义塞在括号 | `改`：标题缩短；正文分别说明 ordering metric、pair accuracy、cluster CI、archive | 让 analysis definition 进入主句 |
| S15-17 | `1137-1140`，post-hoc status、CeSe2、all-pair/group-equal difference | 时间状态、composition share 与 aggregation explanation 连续 | `拆`：descriptive status；largest composition；why all-pair differs | 清楚标出 exploratory breakdown |
| S15-18 | `1141-1145`，one interval、Bonferroni、remove composition | 两个 robustness objections 在一个长句 | `拆`：primary fifth result；multiplicity check；composition-removal check | 每个 robustness check 独立 |
| S15-19 | `1145-1147`，five largest-composition shares | 五个裸比例未映射到 pair fractions | `改`为显式 `100%, 50%, 30%, 20%, 10% → shares` 映射 | 数值不再需要猜顺序 |
| S15-20 | `1147`，exploratory status | 清楚 | `保留` | 收束这一 breakdown |
| S15-21 | `1149-1153`，fixed terms、coefficient-only refit、development counts、held-out not reused | 两种 scope boundary 同段但可更明晰 | `拆`：第一句说明 feature search not reopened；第二句说明 only development groups used；第三句说明 held-out not re-evaluated | 清楚锁定 analysis scope |
| S15-22 | `1155-1159`，coefficient movement、rank stability、cosine stability | 总结与两种 stability statistics 同句 | `拆`：coefficient movement；score-rank correlation；vector-direction similarity | 区分 prediction 与 parameter stability |
| S15-23 | `1159-1163`，magnitude looseness、two examples、sign retention | 三层数值同句 | `拆`：magnitude variability；eta/Mz contrast；sign frequency | 突出哪些量稳定 |
| S15-24 | `1163-1168`，cluster bootstrap、term ordering、ratio range、sign | 方法和三项结果过密 | `拆`：bootstrap method；signal-to-error ordering；sign consistency | 清楚报告 second stability analysis |
| S15-25 | `1168-1170`，what data fix、decimal warning | 强总结清楚 | `保留`；将 `parts` 改为具体 `signs and relative weights` 作主语 | 直接给 interpretation |
| S15-26 | Fig. S14 panel a，`1175-1180` | fixed-term setup、box/median/whisker/dashed line 连续 | `拆`：setup；distribution encoding；reference line | 降低图注密度 |
| S15-27 | Fig. S14 panel b，`1181-1183` | bootstrap interval、published values、right labels 同句 | `拆`为 interval/procedure；markers；labels | 三类视觉编码分开 |
| S15-28 | Fig. S14 panel c，`1184-1187` | 两种 agreement metric 与 lines/shading 同句 | `拆`：rank/vector metrics；mean line；interval shading | 区分 metric 和 encoding |
| S15-29 | `1191-1196`，structure-level correlations 与 within-composition diagnostics | 两个分析尺度挤在一个段落 | `拆/分段`：across-structure correlations；within-composition correlations/VIF/condition number | 先总体再实际 fit space |
| S15-30 | `1197-1199`，spread inference | 清楚但需明确 `spread` 所指 | `改`为 `The coefficient spread...` | 消除图号回指负担 |
| S15-31 | Fig. S15，`1204-1211` | panel a/b 清楚 | `保留`；只统一 `Entries above/below` 的平行语法 | 不作实质改写 |

## B16. PU transfer、property screen 与 inverse design，第 1215-1393 行

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| PU-01 | `1217-1229`，Jang lineage、later application、expanded positives/unlabelled、two models、consensus、deduplication | 一段依次跨过文献历史、现工作数据、模型和筛选结果，主线过密 | `分段`为 prior lineage；current pool expansion；two-model consensus/deduplication 三段 | 先背景，再本研究，再 hard-negative cohort |
| PU-02 | `1217-1222`，Jang et al. 与 later application 两句 | 清楚 | `保留`，只在第二句明确 `the pretrained CLscore model` | 维持时间线 |
| PU-03 | `1222-1225`，positive/unlabelled expansion | 两种 pool 与四个来源连续 | `拆`：positive set 及 ICSD/COD counts；unlabelled set 及 LeMat/ELEMENTA count | 区分 labelled/unlabelled |
| PU-04 | `1225-1229`，50-bag CGCNN、50 MLP heads、consensus 364,771、dedup 364,592 | model construction 与 cohort construction 同段 | `拆`：两种 models；consensus rule/result；deduplication/result | 建立 model → selection → final cohort |
| PU-05 | `1231-1234`，four Set rates 以八个数字连续呈现 | 读者难把 satisfaction/screening 与 Set 1–4 对齐 | `改`为小表或四个严格平行分句，每组固定 `satisfaction; screening` 顺序 | 清楚展示 law-set progression |
| PU-06 | `1234-1235`，PSS curve 与 matched point | 清楚 | `保留` | 给 PSS matched-satisfaction 结果 |
| PU-07 | `1235-1238`，descriptor completeness 与 median imputation | coverage 数和 missing rule 混在结果段 | `分段`：observed coverage 一句；frozen-median rule 一句 | 区分 result 与 evaluation convention |
| PU-08 | `1240-1244`，fractional attribution 与 leave-one-law-out | 两种 attribution 方法连续但未先标明互补关系 | `拆`：fractional allocation 定义/normalization；LOO definition | 分别解释 contribution 与 necessity |
| PU-09 | `1244-1248`，Law 7 share、removal impact、Law 1 task contrast | 数值和跨任务解释连续 | `拆`：fractional share；LOO loss/recovery；controlled-damage contrast | 从 result 到 task dependence |
| PU-10 | Fig. S16，`1253-1260` | panel a/b definitions 清楚 | `保留`；将 `Records without a verdict...` 移至 caption 开头的 common-cohort 说明 | 共用 denominator 先说明 |
| PU-11 | `1264-1267`，two PU representations | 两种 model 定义清楚 | `保留两句` | 解释 complementary encodings |
| PU-12 | `1267-1274`，validation bags、AUC、bag sizes、figure contents | evaluation protocol、performance 和 sample size 同段过密 | `拆`：bag protocol；two AUCs；CGCNN bag size；MatterSim availability；figure navigation | 按 protocol → result → cohort 顺序 |
| PU-13 | Fig. S17，`1279-1287` | panels a/c 和 b/d 各自清楚 | `保留`；把 bag composition 共用句移到 panel a/c 前 | common setup 在 panel details 之前 |
| PU-14 | Fig. S18，`1294-1299`，axes、260/261、non-convergence | 图形定义和 missing candidate 同段 | `拆`：axes/fits；shown cohort；one non-converged cell | 建立 denominator |
| PU-15 | Fig. S18，`1299-1301`，below diagonal、0.940 factor、400→376 thresholds | `because` 句把 direction、scale factor 与两阈值串在一起 | `拆`：systematic scale relation；median factor；calibration mapping | 清楚解释两个尺度 |
| PU-16 | Fig. S18，`1301-1304`，0.966 ranking、3/140 removed、Re2IrOs6 exception | overall result 与 exception 同段 | `拆`：ranking retention；top-140 count；named exception | 既突出整体也突出反例 |
| PU-17 | Fig. S18，`1305-1306`，transfer note | 清楚 | `保留` | 限定 cohort sampling，不弱化主结果 |
| PU-18 | Fig. S19，`1313-1321` | 三个 panel 定义清楚 | `保留`；将 tie convention 放到 panel c 句内 | 让每个 panel 可独立理解 |
| PU-19 | Fig. S20，`1328-1330`，caption title 后 `All 260 candidates...` | 第二句是无谓语片段 | `并`入 caption opening：the panels show all 260 candidates | 修复 fragment |
| PU-20 | Fig. S20 panel a，`1330-1332` | plot 与 composition conclusion 清楚 | `保留` | 展示 property conditioning |
| PU-21 | Fig. S20 panel b/c，`1333-1339` | crystal system、Law 7 quantity、colour、diagonal、bound、data availability 连续 | `拆`：panel b definition/convention；panel c axes/colour；diagonal meaning；bounds；archive | 分开 before–after symmetry 与 visual encoding |
| PU-22 | `1344-1349`，CLscore relation | procedure、two endpoint trends、cross-model inference 四句清楚 | `保留`；首句补全为什么不平均 raw scales | 说明 percentile alignment rationale |
| PU-23 | `1351-1354`，balanced pilot、finite energies、hull reference、sweep | 四步处理连续 | `拆`为 sample；relaxation success；hull construction；threshold sweep | 可复现 energy route |
| PU-24 | `1355-1357`，0.20-eV result 与 cost inference | 清楚 | `保留`，使 retained/screened 主语平行 | 对比 energy screen 与 PRIS |
| PU-25 | `1359-1364`，MatterGen version/commit、13 shards、parsed count、dedup parameters/result | generation 与 deduplication 信息过密 | `分段`：generation settings/count；StructureMatcher settings；dedup outcome | 建立 candidate cohort |
| PU-26 | `1365-1369`，UMA proxy、five volumes、quadratic fit、all passed、140 high-property | property calculation 五步连续 | `拆`：checkpoint/task；volume sampling；fit criterion；success/high-property counts | 清楚说明 proxy derivation |
| PU-27 | `1369-1374`，frozen PSS、2/6 descriptors、541 calibration cohort、97.5% target、cutoff | score state、support matching 与 cutoff calibration 混在一起 | `拆/分段`：frozen score/imputation；calibration cohort；target satisfaction；selected cutoff | 明确 cutoff 未使用 generated labels |
| PU-28 | `1374-1378`，screen/tie rule、97.6%/61/140、no generated data、support stratum | decision rule、result 与 leakage control 连续 | `拆`：screening rule；calibration outcome；61/140 application result；no generated data in fitting；support-only role | 展示 calibration → application → control |
| PU-29 | `1378-1382`，two-descriptor equation | 清楚 | `保留` | 给 matched-stratum exact form |
| PU-30 | `1383-1386`，61 candidates 的 site fraction、distance pass、Law 7、volume range/means、subset relation | 五个特征堆叠 | `拆`：shared verdict profile；screened volume distribution；retained mean comparison；subset relation | 描绘 PSS removed cohort |
| PU-31 | `1387-1388`，Set 4 removed 95/140 与 denser packing retained by PSS | 强对比清楚 | `保留` | 说明 fixed-law/PSS division of labour |
| PU-32 | `1388-1392`，matched Ir2Os7 examples 与两结构 descriptor/property | 两个例子应用同一模板但挤在段末 | `分段`：screened structure 一句；retained structure 一句，字段顺序一致 | 形成直接 matched example |
| PU-33 | `1392-1393`，archive inventory | 五类 archive item 连续 | `拆`为 generation/model artifacts 与 evaluation artifacts 两组 | 清楚说明复现材料 |

## B17. Supplementary Note S17：Law 7/8、外部结构与机制诊断

### S17 opening、selection 与 held-out evaluation，第 1395-1450 行

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| S17-01 | `1395`，section title | 标题把两个发现和 sequence 挤在一起，过长 | `改`为更短的 `Extending the plausibility sequence with structural simplicity and bond valence`，不改变结论 | 快速识别本节主题 |
| S17-02 | `1397-1402`，preregistration、hash、data reuse、recreation、Set 3 reproduction | protocol lock、data provenance 与 gate 五步同段 | `拆`：plan/hash；reserve-free tables/recreation；Set 3 reproduction gate | 建立 search 前置条件 |
| S17-03 | `1403`，`Model class: Set 3 unchanged...` | 句子碎片 | `改`为完整句，以 `The model class kept...` 开头 | 修复语法 |
| S17-04 | `1403-1406`，model class 与 8,466 candidates/94 quantities | rule form、threshold grid 和 candidate count 过密 | `拆`：allowed additions；threshold source；candidate count/eligible quantities | 定义 fixed search space |
| S17-05 | `1406-1409`，objective、one-shot held-out、previous reuse | selection objective 与 evaluation governance 同段 | `分段`：objective/constraint；held-out evaluation；S8 disclosure | 分开 fitting 与 evaluation history |
| S17-06 | `1411-1416`，Law 7 bullet | definition、Pauling meaning、grid percentile、population percentile 一个 bullet 长句 | `拆`成 threshold law；physical interpretation；threshold provenance | 一项 law 三层信息 |
| S17-07 | `1417-1419`，Law 8 bullet | definition 与 principle 同句 | `拆`为 hard bound 与 physical meaning | 与 Law 7 结构平行 |
| S17-08 | `1421`，`Discovery: satisfaction..., damage detection...` | 句子碎片 | `改`为完整 result sentence | 报告 discovery performance |
| S17-09 | `1421-1422`，third candidate gain/stopping | 清楚但句长 | `拆`：gain value；below pre-fixed threshold, therefore stopped | 明确 stopping rule |
| S17-10 | `1424-1430`，overall rates、five type rates、experimental three outcomes、damaged three outcomes、criteria | 十二个数值在一段 | `改`为小表或五句：overall；by damage type；experimental outcomes；damaged outcomes；criteria/met margin | 让 held-out evaluation 可核验 |
| S17-11 | `1432-1438`，LOFO five values、same laws in three folds、alternate laws in two folds | result list与 selection pattern 连续 | `拆`：five unseen-type results；three same-law folds；two alternate-law folds | 分开 performance 与 model identity |
| S17-12 | `1438-1441`，physicochemical interpretation 与 tree comparison caveat | inference 和 non-direct comparator 同句组 | `分段`：先保留 physicochemical conclusion；再明确 Set 3-fixed additions vs full tree refit 的协议差异 | 防止读者把 0.2318 直接同口径比较 |
| S17-13 | `1443-1447`，further population、440/2024、parent provenance、partition counts/non-disjoint | benchmark design 与 provenance 同段 | `拆`：application without refit/sample；parent origin；partition overlap | 先分析对象，再数据关系 |
| S17-14 | `1447-1450`，overall/by-type results、held-out consistency、27×、source data | 七个数值与两个结论同句 | `拆`：overall rates；by-type rates；held-out consistency；27× comparison；data availability | 突出外部 benchmark 强结果 |

### Law 7/8 分工、matched-satisfaction 与 amplitude，第 1452-1533 行

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| S17-15 | `1454-1458`，Set 3+Law 7 与 Set 3+Law 8 的 overall/by-type rates | 一句承载两组各七个数值 | `改`为两行小表或两个完全平行句；固定 satisfaction → overall detection → D1–D5 顺序 | 直接比较两条 additions |
| S17-16 | `1459-1460`，Law 7/8 mechanism contributions | 两句平行、清楚 | `保留` | 总结互补分工 |
| S17-17 | `1460-1461`，`Two facts ... must be stated plainly` | 元话语占位 | `删`并直接用 `First` 引第一项结果 | 更快进入实质 |
| S17-18 | `1461-1466`，Law 7 class sensitivity、R5、combination、experimental merge control | detection pattern、past test、law-set role 与 control 混在一起 | `拆/分段`：class-specific response；R5 relation；why combined set works；merge-control pointer | 完整说明 Law 7 的作用而不重复 |
| S17-19 | `1467-1471`，symmetry tolerance、flip rates、symmetrized input、noisy structures | convention、robustness result 和 practice advice 一句过长 | `拆`：primary convention；0.1-tolerance flip rates；input assumption；refinement-noise practice | 从 definition 到 deployment |
| S17-20 | `1472-1475`，D5-omitted refit selects space-group law 与 family-level inference | 结果和解释同句 | `拆`：which law selected；what this shows about family vs exact threshold | 强化 search recurrence |
| S17-21 | `1479-1484`，distribution summary、grid percentile、GII/practice convention | 三种尺度连续 | `拆`：distribution quantiles；threshold location；GII comparison | 交代阈值处于何处 |
| S17-22 | `1484-1488`，permissive threshold、distribution separation、two medians、extreme exclusion | 总结、证据、结论重复推进 | `并/拆`：先定性说明 permissive upper-tail bound；再给 medians/separation；最后说明 excluded extreme tail | 一次完成“宽松但有区分度” |
| S17-23 | `1489-1490`，missing BVS parameters | 清楚 | `保留` | 报告 coverage rule |
| S17-24 | `1494-1497`，five required satisfactions、damage-detection range、0.81 interpretation | grid、result 和 inference 同句组 | `拆`：tested satisfaction levels；damage range；what 0.81 primarily sets | 展示 threshold sensitivity |
| S17-25 | `1497-1500`，exact pair only at 0.81、Law 8 variants、space-group replacements | 三种 selection outcome 一句 | `拆`：exact pair；Law 8 recurrence；symmetry-family recurrence | 表明 family stability |
| S17-26 | `1500-1505`，exclude all Pauling families vs symmetry-only exclusion | 两项 ablation 的 excluded features、scores 和 selected mechanisms 过密 | `拆`为 all-family exclusion 与 symmetry-only exclusion 两段，字段保持平行 | 清楚比较 ablations |
| S17-27 | `1505-1507`，final interpretation | 结论清楚但 `and` 连两个主张 | `拆`：Pauling families not necessary for high detection；free search selects them first/final gain | 保留强结论并增强节奏 |
| S17-28 | `1511-1515`，27× different-satisfaction、matched cutoff/0.339 vs 0.879/2.6×、Set 1 2.1× | caveat、setup、结果三层过密 | `拆`：why matching needed；tuned cutoff/result；Set 4 margin；Set 1 margin | 先公平性再强对比 |
| S17-29 | `1516-1517`，two bootstrap intervals | 两项 interval 同句 | `拆`：satisfaction CI；damage-detection CI | 便于对应 endpoints |
| S17-30 | `1518-1521`，reserve-parent result 与 why not untouched | 数值和 split status 同段 | `拆`：reserve-parent rates；S8 incident implication | 说明数值而不混淆独立性 |
| S17-31 | `1525-1529`，one amplitude、graded repeats、onset、distance-filter result | setup 与两项 response 同句过长 | `拆`：why graded test；which perturbations/population；PRIS onset；distance cutoff response | 建立 amplitude-response comparison |
| S17-32 | `1530-1533`，Gaussian displacement saturation、Wyckoff cause、law-set progression | response/cause 与 broader sequence 同段 | `拆`：Law 7 saturation/cause；Law 1–3 progressive response | 区分 symmetry sensitivity 与 sequence behaviour |

### Generator outputs 与 GNoME，第 1535-1646 行

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| S17-33 | `1537-1540`，why external test、MatterGen sample provenance、distance filters | motivation、sample 和 baseline 同段 | `拆`：new external question；sample/source；distance-filter result | 引出 generator test |
| S17-34 | `1541-1543`，Set 1–4 full-sample fractions 与 denominator convention | 四个 rates 加 no-verdict rule | `改`为一行 progression table；下一句单独定义 denominator/no-verdict | 防止把 no verdict 当 fail |
| S17-35 | `1544-1546`，338 unassigned 与 assignable 4/155/3 | charge coverage 与 conditional outcomes 连续 | `拆`：unassignable count；assignable denominator；three Set 4 outcomes | 明确两级 denominator |
| S17-36 | `1547-1549`，Law 1 none fail → later laws source | 观察与解释清楚 | `保留` | 指向 chemical/electrostatic/site-order bounds |
| S17-37 | `1549-1553`，no DFT、what satisfaction means、potential check、charge limit | 四句都短但尾部重复限制 | `并`：no DFT + later energy check；保留 charge assignment as coverage limit | 收束 external screen |
| S17-38 | `1558-1562`，release/download、554,054、sample/seed/before parse | acquisition 与 sampling protocol 同段 | `拆`：release/source/size；uniform sample/seed/timing | 先 population 再 sample |
| S17-39 | `1562-1567`，primitive cell/spglib/deposited label、difference from discovery、comparability、parse success | symmetry convention 与 consequence 四层 | `拆`：orbit calculation；label not used；cell-convention difference；rate-comparison consequence；parse success | 清楚标出 external convention |
| S17-40 | `1569-1572`，overall Law 7 failure 与 per-space-group list | 总体数值和六组分类堆叠 | `拆`：overall rate；P1/Pm/Cm；remaining groups | 展示 anomaly concentration |
| S17-41 | `1572-1577`，tolerance sensitivity concern/test/result | robustness result 先出现，concern 后出现 | `重排`：concern；strict-to-coarse test；failure-rate change；label-agreement result | 按 question → test → result |
| S17-42 | `1577-1580`，concentration conclusion、median/bound、archive | 结论、distribution summary 与 availability 连续 | `拆`：space-group anomaly conclusion；median vs bound；archive | 收束 GNoME prevalence |
| S17-43 | `1582-1589`，merge hypothesis、expected positive/negative outcome | hypothesis 分三句且重复 “similar/distinct” | `压/并`：一句定义 over-ordering hypothesis；一句给 mergeable prediction；一句给 geometrically distinct control prediction | 清晰建立干预逻辑 |
| S17-44 | `1589-1593`，300 stratified sample、four space groups、merge class、recompute | sample design 与 operator 同句 | `拆`：sample strata/count；merge rule；recomputed quantities/convention | 可复现 merge test |
| S17-45 | `1594-1596`，mergeable prevalence 113/150 vs 78/150 | 清楚 | `保留` | 说明 merge opportunity |
| S17-46 | `1596-1603`，strict tolerance little change、loose tolerance 78/79%、space group median、nonmergeable control | 两个 tolerance 与四项 result 一个长句 | `拆`：strict result/reason；why 0.1 used；bound restoration；space-group increase；nonmergeable control | 展示 intervention outcome |
| S17-47 | `1603-1606`，generative inference、satisfying entries | primary inference 和 internal control 清楚 | `保留两句` | 收束 generative sample |
| S17-48 | `1607-1615`，experimental control sample、27 cases、0/27、examples、why sites differ、final inference | control design、result、chemical examples、interpretation一段过长 | `拆/分段`：control cohort；27 eligible/0 restored；examples of genuine ordering；comparison inference | 强化 generative vs experimental contrast |
| S17-49 | `1616-1620`，database lookup、2/300、novelty criterion、why merge is direct | complementary test 和 interpretation同句 | `拆`：lookup method/result；relation to novelty criterion；why intervention is direct check | 完成 alternative explanation check |
| S17-50 | `1624-1628`，second protocol、Law 8 audit mismatch、recomputation、preservation | protocol、error、correction和 archive 同段 | `拆`：pre-fixed extension protocol；audit finding；recomputed definition；preserved original artifacts | 清楚记录 implementation correction |
| S17-51 | `1628-1631`，integer/fallback assignment 与 no-verdict | 两步 assignment 和 outcome 同段 | `拆`：integer attempt；mean-valence fallback；neither → no verdict | 定义 charge pipeline |
| S17-52 | `1632-1635`，7.5% subset、intermetallic remainder、per-law satisfaction/expected reason | coverage 与 conditional performance 同段 | `拆`：assignable coverage/why low；per-law rates；within-bound interpretation | 区分 applicability 与 performance |
| S17-53 | `1636-1638`，Set 3 vs seven-law rates、335/314 denominators、159/62 outcomes | 两个 set 与多种 outcomes 一句 | `改`为小表：set、evaluable n、satisfied/failed/no-verdict | 防止 denominator 混读 |
| S17-54 | `1638-1641`，155 failures 分成 four Law 7/8 classes | 四类数量长串 | `改`为平行列表或小表，确保四类总数可见 | 展示新增 laws 的 attribution |
| S17-55 | `1641-1643`，Law 8 coverage 与 36/254 | coverage limit 与 failure rate同句组 | `拆`：coverage/limiting factor；conditional result | 先可评估性再 verdict |
| S17-56 | `1643-1646`，Law 7 26.1% vs 40.6%、subset symmetry、different denominators | comparison、reason和 denominator note 过密 | `拆`：two rates；subset chemistry/symmetry reason；denominator warning | 解释看似不一致的数值 |

### Seven-generator benchmark 与 coordinate checks，第 1648-1757 行

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| S17-57 | `1650-1655`，seven models、deposit/licence、500 per model、MP-20 control definition | benchmark provenance、sample size和 control parenthetical 同句过长 | `拆`：dataset/models/licence；per-model sample；experimental control definition | 建立外部 benchmark |
| S17-58 | `1655-1656`，distance filters do not separate、law sequence does | 对比短而有力 | `保留` | 进入 PRIS result |
| S17-59 | `1657-1663`，charge no-verdict rates 与 three generator groups/Set 4 rates/control | applicability、three model classes 和 satisfaction rates 过密 | `拆`：charge coverage；coordinate-only models；symmetry-constrained models；experimental control | 按生成方式呈现 separation |
| S17-60 | `1663-1664`，other unavailable measurements/no verdict | 两句可合并 | `并`为一条 denominator rule | 避免重复 no-verdict 定义 |
| S17-61 | `1665-1667`，strict Law 7 rates 与 archive | 结果和 availability 混在一起 | `拆`：Law 7 outcome；archive statement | 收束 raw-output analysis |
| S17-62 | `1669-1672`，`Two objections... The first...` | 元话语略长 | `改`为直接编号结构：`First, ...`；首句只说明 two measured checks | 快速进入 robustness |
| S17-63 | `1672-1677`，full-denominator Law 7 rates、same separation、charge explanation | test、three rates和 inference 一句组 | `拆`：why Law 7 uses full cohort；no-symmetry/symmetry/control rates；what this rules out | 完成 first objection |
| S17-64 | `1677-1679`，second objection、numerical triclinic、unrelaxed noise | concern 清楚但句长 | `拆`：state objection；why unrelaxed orbit detection may reflect noise | 建立 relaxation test |
| S17-65 | `1679-1685`，2,500 relaxation protocol、0.1 tolerance、five model/control rates | setup 和全部结果同句 | `拆`：five cohorts/protocol；tolerance rationale；MatterGen/DiffCSP；control/SymmCD；MiAD | 清楚报告 relaxation robustness |
| S17-66 | `1686-1690`，reduces-not-erases inference 与 scope sentence | 两句清楚 | `保留` | 给 tested-sets strong conclusion |
| S17-67 | `1694-1697`，checkCIF required inputs and not run | required inputs 与 decision同句 | `拆`：which inputs unavailable；therefore established software not run | 直接说明选择原因 |
| S17-68 | `1697-1704`，four reimplemented test families | 四项含各自 thresholds 的超长列表 | `改`为编号列表：missed symmetry、short contacts、voids、floating atoms；每项一行给 threshold/sensitivity | 让 coordinate checks 可核验 |
| S17-69 | `1704-1708`，no alert-code claim、Law 8 audit、recomputation/preservation | software naming boundary 和 unrelated audit 混在一起 | `分段`：coordinate-check naming 一段；Law 8 correction 一段 | 避免读者误解为官方 checkCIF output |
| S17-70 | `1709-1712`，primitive-cell convention、paired comparability、discovery incomparability | method 和两种 comparison consequence 一段 | `拆`：conversion rule；paired comparison validity；absolute-rate noncomparability | 清楚标出 cell convention |
| S17-71 | `1714-1718`，seeded 8,000 scan/first 300、enrichment、uniform sample 222/5000 | sample selection 和 prevalence correction 连续 | `拆`：enriched sample method；what it cannot estimate；uniform-sample reference rate | 防止把 300 当代表性样本 |
| S17-72 | `1718-1721`，perturbations、computed parents as controls、why not experimental | operator 和 control choice同段 | `拆`：damage generation；control population；licence reason | 定义 paired benchmark |
| S17-73 | `1721-1724`，parent failure rates 与 added paired analysis | initial result 和 post-result design change 连续 | `拆`：unpaired parent rates；when paired analysis was added；paired detection definition | 时间透明且定义明确 |
| S17-74 | `1725-1727`，six-law set definition、Law 7/8 consistency counts | method label 与 consistency result同段 | `拆`：law-set composition；sample rates；reference-subset rates | 核对 cohort consistency |
| S17-75 | `1747-1751`，pre-specified comparison 0.365/0.021 与 added paired 0.430/0.042 | 两种 analysis 条件与数值挤在一起 | `拆`：primary restriction/results；post-result paired restriction/results | 严格区分两次 analysis |
| S17-76 | `1751-1755`，two symmetry tests、coordinate-check definition/limitation、Law 7 question | contrast 清楚但四句可更平行 | `改`为 `coordinate check asks...; Law 7 asks...`；将 wrong-site result 置于前者之后 | 一眼看到两种 test 的不同 |
| S17-77 | `1756-1757`，archive inventory | 清楚 | `保留` | 指向 per-structure evidence |

### Historical falsified depositions、relaxation 和 composition screen，第 1759-1839 行

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| S17-78 | `1762-1765`，why historical cohort、no refit、source | 三句清楚 | `保留` | 从 controlled damage 转 real cases |
| S17-79 | `1767-1776`，184 reports/seven notices、general fraud pattern、missing parent provenance、no inference | corpus size、mechanism历史和 provenance restriction 同段 | `分段`：archive scale；reported pattern；missing lineage；what is not inferred | 建立 historical evidence boundary |
| S17-80 | `1776-1783`，retrieval source、177 parsed、molecular domain、172 no verdict、one control too large、five evaluable all fail | retrieval、parse、domain和 verdict pipeline过密 | `拆`：where CIFs retrieved；parse count；domain composition；172 no verdict；control status；five evaluable result | 从 archive 到 evaluable cohort |
| S17-81 | `1783-1786`，four M(C2H2N3Cl) entries、four metals、no common-parent inference | sample grouping 与 provenance boundary同句组 | `拆`：four related-framework records；metal labels；no common-parent claim | 精确说明 observed grouping |
| S17-82 | `1786-1791`，two common violations、rho value/reason、BVS range、mechanism match | 共同结论、两个量和机制解释连续 | `拆`：common failures；contact result/physical meaning；bond-valence result；connection to cation-swap mechanism | 展示 fixed-coordinate diagnosis |
| S17-83 | `1791-1793`，fifth entry 与 archive | result 和 availability 同句 | `拆`：KNOF2 result；archive statement | 完成 fifth case |
| S17-84 | `1797-1799`，MatterSim settings 与 convergence/cap | 五项设置在括号中 | `拆`：model/cell/optimizer；force/cap；convergence status | 定义 common relaxation protocol |
| S17-85 | `1799-1806`，150 GNoME parents、damage counterparts、parent median、three damage medians、swap n limitation | sample、baseline、damage results和 missing-charge limitation 一段 | `拆`：sample/seed；parent response；compression/expansion/displacement；swap-class sample limitation | 清楚给 severity scale |
| S17-86 | `1807-1811`，MatterGen addendum、162 assignable result、338 no-charge result | timing、two cohorts和多个 statistics 同段 | `拆`：pre-fixed addendum；assignable cohort; no-charge cohort | 平行比较 raw outputs |
| S17-87 | `1812-1815`，what sample cannot measure、what it shows、separate experiment、protocol repeated | `protocol fixed` 在开头和末尾重复；两次解释 `only` | 删除末句重复；`拆`为 limitation、supported result、relation to damage experiment | 精炼 interpretation |
| S17-88 | `1820-1826`，why compare composition check、validity lineage、two component tests | motivation 与 metric provenance 同段 | `拆`：why included；metric/source；composition and distance components | 定义 incumbent baseline |
| S17-89 | `1826-1829`，smact settings、76.8%、stricter chemistry comparison | protocol、result、interpretation 同句 | `拆`：implementation；satisfaction；what it tests | 说明 composition-screen selectivity |
| S17-90 | `1829-1833`，structurally zero、composition preserved、original verdict、conditional 0.000、conclusion repeated | 零结果以四种方式重复 | `压`：保留 measured 0.000 by all types 和 composition-preserving mechanism；删除重复 “nothing to measure” 表述 | 简洁给出 exact zero 与原因 |
| S17-91 | `1833-1834`，archive | 清楚 | `保留` | 指向 source data |
| S17-92 | `1836-1839`，trade-off paragraph | 三句逻辑清楚、结论有力 | `保留`；只把最后一句的 line break/并列词顺平 | 保持明确的 use-case decision |

## B18. Supplementary Note S18：robustness、equations 与 implementation

### Reduced contact 与 exact trees，第 1841-1879 行

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| S18-01 | `1843-1844`，Note S18 导航句 | 清楚 | `保留` | 说明本节收纳量化细节 |
| S18-02 | `1848-1851`，tail shape、quadratic/log density、RSS/curvature、repulsive-wall interpretation | fit setup、numbers 和 physical meaning 同句 | `拆`：distribution/fitted range；linear vs quadratic residuals/curvature；repulsive-tail meaning | 从统计形状到物理解释 |
| S18-03 | `1851-1854`，0.735 percentile、not constant、four-point sweep | 定义与 threshold list 同句 | `拆`：first-percentile provenance；no sharp transition；four sweep values用平行列表 | 清楚说明参数可调且非物理常数 |
| S18-04 | `1854-1857`，anion-stratified satisfaction 与 phosphide damage result | overall family stability 和 exception 同句 | `拆`：ten-family satisfaction；phosphide damage detection/control count | 突出 chemistry-specific response |
| S18-05 | `1857-1859`，`criterion is not wrong ... almost no detection ... satisfaction alone` | 以“not wrong”绕述，三次解释同一差异 | `改`为直接句：for phosphides the law preserves satisfaction but contributes little damage detection | 一句说清两种指标 |
| S18-06 | `1859-1860`，DFT-relaxed satisfaction | 清楚 | `保留` | 排除 relaxation inflation |
| S18-07 | `1864-1867`，tree vs core、cost-ratio plateau、F1 invariance、only cost moves curve | performance comparison 和 objective robustness 同段 | `拆`：tree/core point；ratio plateau；objective invariance；trade-off controller | 清楚界定 operating point |
| S18-08 | `1868-1869`，depth-4 runtime/no proof 与 exactness scope | 两句清楚 | `保留` | 限定 optimality 到 depth≤3 |
| S18-09 | `1869-1871`，omitted-type mean drop 与 single-threshold comparison | 两组 seen/omitted numbers同句 | `拆`：tree seen→omitted；single threshold omitted；comparison | 突出 omitted-class sensitivity |
| S18-10 | `1871-1874`，five signed changes、cancellation、mean absolute change | 五项列表和 summary 同句 | `拆`：signed changes；why mean cancels；absolute-change comparison | 解释 metric choice |
| S18-11 | `1874-1876`，tree wins 3/5 与 aggregate/per-type disagreement | 清楚 | `保留` | 同时报告 aggregate 和 per-type |
| S18-12 | `1876-1879`，selected features 与 damage-type pattern | feature list 和 inference同句 | `拆`：which features selected；their per-type pattern | 说明 tree mechanism differs |

### Cross-database ranking、ties 与 equations，第 1881-1958 行

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| S18-13 | `1883-1888`，four energy spans、order-of-magnitude meaning、matched-window four accuracies、ranking reversal | 两组四项数字和解释同段 | `拆`：energy spans；task-difficulty interpretation；matched window；four accuracies/ranking reversal | 清楚区分 distribution shift 与 matched result |
| S18-14 | `1889-1895`，four largest-group shares、pair/group weighting reversal、reporting principle | composition dominance 和 weighting result同段 | `拆`：largest shares；pair-weighted result；group-equal result；reporting implication | 解释 R9 的来源 |
| S18-15 | `1896-1900`，descriptive full-set PSS、three exclusion conditions、unchanged conclusion | exploratory status 与 three equal scores 同段 | `拆`：analysis status；full/exclude-SiO2/exclude-framework values；unchanged inference | 清楚报告 composition robustness |
| S18-16 | `1900-1903`，volume score、denser proportion、non-oxide persistence | 三个 supporting results 连续 | `拆`：volume baseline；non-silica density direction；non-oxide result | 支撑 packing interpretation |
| S18-17 | `1905-1909`，high-pressure exclusion check | 表达清楚 | `保留` | 排除 density term 由 high-pressure phases 驱动 |
| S18-18 | `1913-1917`，four law sets 的 decision rates、group tie rates、conditional accuracies、39–225 groups | 三种 metrics × 四 sets 堆成两句 | `改`为小表：set、pair decision、no-unique-group、conditional accuracy、contributing groups | 同时看coverage与accuracy |
| S18-19 | `1917-1920`，ties appropriate、thresholding discards ordering、rho continuous | 三句逻辑清楚 | `保留` | 解释 filter/ranking distinction |
| S18-20 | `1924-1928`，`The eight PRIS laws, written as applied...` | 开头是句子碎片；rho/f_i/EM 三个定义挤在后句 | `改`为完整引导句；每个 symbol 单独短句；把 Ewald/model-bound 说明分开 | 清楚引出 equations |
| S18-21 | `1929-1941`，Law 1–8 equations | 数学表达是最清楚形式 | `保留` | 不改 thresholds 或 definitions |
| S18-22 | `1943-1945`，standardization 与 missing imputation | 两项 preprocessing 在一个从句 | `拆`：define standardized value；define frozen-median (x^\dagger) | 分清缩放与缺失处理 |
| S18-23 | `1945-1947`，positive difference 与 zero-intercept score intro | 清楚 | `保留` | 解释 score direction |
| S18-24 | `1954-1958`，three descriptor definitions 与 two topology definitions | 公式句清楚 | `保留`；可把 topology 两项写为平行结构 | 完成 PSS glossary |

### Phonons 与 energy/order distinction，第 1960-2107 行

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| S18-25 | `1962-1965`，26,600 cohort、two phonon sources/counts/framework、pre-fixed protocol | cohort definition 和 provenance 过密 | `拆`：total cohort；DFPT source/count；finite-displacement source/count；protocol | 建立 phonon population |
| S18-26 | `1965-1969`，three properties、two imaginary-mode definitions/threshold sensitivity | 三项 analysis variables 和 source-specific cutoff一长句 | `改`为编号三项；imaginary-mode item 下再分 DFPT/DOS definitions | 让 categories 可复现 |
| S18-27 | `1969-1971`，cross-method 与 band-vs-DOS agreement | 两个 agreement denominator 不同 | `拆`为两句并明确 1,175 common cohort | 防止混读 88.8%/99.1% |
| S18-28 | `1971-1974`，three main findings | 三个 headline findings 一句 | `拆`为 on-hull imaginary modes；recorded metastability；unrecorded stable structures 三句 | 每类概念独立 |
| S18-29 | `1975`，`Satisfaction of each law is the source...` | 句法生硬、回指不明 | `改`为直接导航：the per-law satisfaction analysis underlying the distinct-site comparison appears in S18.7 | 明确指向后文 |
| S18-30 | `1975-1982`，two limitations 与 lenient-tolerance result | sampling bias、tolerance sensitivity、two population shifts同段 | `拆`：sampling bias；finite-displacement sensitivity；experimental shift；theoretical shift；comparative conclusion | 分开 limitation 与 robust contrast |
| S18-31 | `1986-1987`，`two conditions on this population` | `two conditions` 未具体命名 | `改`为直接写 experimental record、hull energy、imaginary modes 中实际比较的两类条件/维度 | 消除模糊回指 |
| S18-32 | `1987-1989`，all five evaluable because stored relaxed structures not downloaded again | “not downloaded again” 是实现过程细节，打断科学定义 | `压`为 all law quantities were computed on the same stored relaxed structures | 保留可比性而去除操作琐碎 |
| S18-33 | `1989-1994`，charge convention、requirements、no-verdict denominator、five subset sizes | charge algorithm 和五个 denominators 一句组 | `拆`：charge assignment；ionic requirement；per-set denominator rule；five subset counts用表或并列 | 定义 evaluable cohorts |
| S18-34 | `1995-1996`，14,986 common cohort、four-set sequence、Set1' separate | 清楚 | `保留` | 说明 table comparison population |
| S18-35 | `2018-2021`，3,929 failures 与 per-law counts | 六条 law counts 长串 | `改`为小表或分号列表，注明 multi-law failures | 可读地给 mechanism frequencies |
| S18-36 | `2021-2029`，dominant mechanisms、Law 7 all vs BVS subset rates、subset bias、denominator principle | result、population effect和 reporting rule一段过长 | `拆`：dominant contributors；Law 7 full rate；joint-coverage rate；subset composition；same-population rule | 解释 denominator dependence |
| S18-37 | `2029-2032`，two robustness checks intro 与 neighbour alternative result | `Two` 但第二项直到后文，结构不清 | `改`为 `First` 段：primary Law 6 convention、alternative shell、Set 3/4 changes | 明确 robustness check 1 |
| S18-38 | `2033-2037`，molecular-anion chemistry 与 treatment | 例子列表和 conclusion 清楚 | `保留`，以 `Second` 开头 | 明确 robustness check 2 |
| S18-39 | `2038-2041`，implementation audit、old radial definitions、recompute、old columns | error/correction/status 同段 | `拆`：audit finding；production recomputation；role of earlier columns | 清楚记录修复 |
| S18-40 | `2043-2049`，sampling limitation、benchmark vs application missing convention、comparison advice、source tables | 两种限制和导航连续 | `拆`：sampling limitation；missing-input convention difference；comparison implication；data availability | 收束 table interpretation |
| S18-41 | Fig. S21，`2054-2060` | axes/class crossing、two laws、two record statuses、lines 同段 | `拆`：cohort/axis；upper/lower profiles；markers/lines | 图注按图形元素分层 |
| S18-42 | Fig. S22，`2067-2072`，merge-group ordering/relaxation、points、bar、sorting、300K line | setup 和四个 visual encodings 连续 | `拆`：enumeration/DFT setup；points；horizontal bar/sorting；temperature line | 读者先理解 estimator |
| S18-43 | Fig. S22，`2072-2077`，two medians/358×、0 vs 18/23、solid-solution inference、note navigation | 两组结果、热力学解释和导航同段 | `拆`：GNoME/control energies；factor；300-K counts；physical interpretation；Note pointer | 突出 ordering result |
| S18-44 | Fig. S23 panel a，`2084-2091`，200-cell composition、five operators、50 parents/20 GNoME、same protocol、correlation、floor | cohort、methods和 result过密 | `拆`：total cohort；damaged component；undamaged component；paired calculation；correlation；floor points | 清楚定义 paired DFT validation |
| S18-45 | Fig. S23 panel b，`2091-2093` | purpose 与 bar encoding同句 | `拆`：class-resolved purpose；filled/hatched mapping | 图注更易扫读 |
| S18-46 | Fig. S24，`2099-2105` | flow 与 no-verdict convention 清楚 | `保留` | 展示 charge coverage |

### Threshold stability 与 timing，第 2109-2159 行

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| S18-47 | `2111-2115`，bound definition、same-percentile rederivation、value movement、verdict flips | definition 与 three-step stability procedure 同段 | `拆`：percentile provenance；rederive on held-out；measure threshold shift；measure verdict flips | 明确 stability test |
| S18-48 | `2136-2137`，contact/packing/valence shifts 与 Law 8 same grid | 清楚 | `保留` | 总结 stable bounds |
| S18-49 | `2137-2141`，electrostatic tail、15→4 eV、<0.4% flips、interpretation | threshold shift 与 verdict stability同句组 | `拆`：why tails sparse；threshold change；experimental flip rate；strong verdict-stability conclusion | 区分数值与 classifications |
| S18-50 | `2141-2142`，Law 7 discrete grid | 清楚 | `保留` | 解释 2/3→3/4 |
| S18-51 | `2146-2149`，benchmark setup 与 Law 1 three cell sizes | setup 和 timings同段 | `拆`：hardware/process protocol；Law 1 operation；three timings | 建立 microbenchmark |
| S18-52 | `2150-2154`，full implementation operations/timings、charge cost、cached estimate | current cost 与 optimization potential同段 | `拆`：eight-law implementation；three timings；charge overhead；cached-neighbour estimate | 区分 measured 与 projected optimization |
| S18-53 | `2154-2158`，10k queue law timings 与 DFT projection | two queue estimates和 literature projection同段 | `拆`：contact queue；eight-law queue；DFT per-structure assumption；10k projection | 构建成本层级 |
| S18-54 | `2158-2159`，order-of-magnitude disclaimer | 清楚 | `保留` | 明确 DFT number is projection |

## B19. Supplementary Note S19：shared evaluation procedure

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| S19-01 | `2163-2166`，three raw split counts 与 analysis-subset definition | 总数和可分析子集同段 | `拆`：raw partition counts；charge/features eligibility | 区分 assigned 与 analysed populations |
| S19-02 | `2166-2170`，structure-ID split、same composition across partitions、what consistency measures、omitted-type evidence | split unit 和两种 validation meaning 连续 | `拆`：structure-level hashing；composition overlap；split-consistency interpretation；omitted-class evidence | 清楚限定 random-split 含义 |
| S19-03 | `2172-2174`，PSS preregistration/hash 与 model form | governance 和 statistical definition 同段 | `拆`：protocol lock/archive；antisymmetric zero-intercept model | 先计划，再模型 |
| S19-04 | `2174-2178`，imputation、standardization、pair weighting、forward selection、CV、two stopping rules、eight-term cap | 七项 training rule 挤在一起 | `改`为 ordered procedure：preprocess → weight → select/CV → stopping/cap | 可复现 PSS fitting |
| S19-05 | `2179-2181`，1,508 split counts、seeded hash、script refuses repeat | cohort split 与 evaluation gate 同句 | `拆`：919/589 assignment；one-shot held-out gate | 区分 data split 和 enforcement |
| S19-06 | `2181-2184`，full-pool transfer 与 archive inventory | application 和 reproducibility 同段 | `拆`：no-refit transfer；archived coefficients/statistics/verdicts/logs | 收束 PSS protocol |
| S19-07 | `2186-2188`，Set 4 prereg、feature timing、Set 3 reproduction、damage recreation | 四项 gate 一句 | `拆`：plan timing；Set 3 reproduction；damage recreation | 建立 selection prerequisites |
| S19-08 | `2188-2191`，greedy objective、satisfaction floor、percentile grid、one-shot held-out | fitting rules 与 evaluation 同段 | `拆`：objective/constraint；grid；held-out evaluation | 分开 selection 和 test |
| S19-09 | `2191-2194`，omitted variants 与 PSS confidence breakdown | Set 4 transfer rule 和 PSS exploratory analysis 不属同一主题 | `分段`：Set 4 LOFO rule；PSS post-evaluation confidence analysis | 避免跨分析跳转 |

## B20. DFT Supplement：four learned quantities，第 2196-2258 行

| 编号 | 行号与原句定位 | 问题 | 具体修改动作 | 修改后功能 |
|---|---|---|---|---|
| DFT-01 | `2198-2201`，four learned quantities | 四项中有 bounds、claim、severity scale、property target，语法不平行 | `改`为编号或分号列表，统一成四个 noun phrases | 给 DFT campaign 清晰地图 |
| DFT-02 | `2201-2204`，each rederived/pre-fixed protocol/repository artifacts | 方法原则与 archive 路径清楚 | `拆`为 pre-fixed DFT rule；artifact location | 分开 scientific protocol 与 availability |
| DFT-03 | `2206-2210`，`VASP 6.3.0 with...` | 段首是无谓语句子碎片；code、potentials、cutoff、mesh、EDIFF、smearing 全列一串 | `改`为完整句；拆成 code/potentials/cutoff 与 k-mesh/convergence/smearing 两句 | 修复语法并降低参数密度 |
| DFT-04 | `2210-2212`，cell relaxation、constant-volume ISIF、1,917 tasks/344 node-hours | 两种 task settings 和 campaign scale 连续 | `拆`：relaxation settings；energy-volume settings；campaign size | 区分计算类型与规模 |
| DFT-05 | `2214-2218`，`Five experiments answer four questions` 后的四项列举 | contact landscape/control 被算成两 experiment，读者需自行对应 5→4 | `改`为编号映射：Q1 includes two experiments；Q2/Q3/Q4 each one | 让 experiment–question 对应明确 |
| DFT-06 | `2218-2219`，`Two of them need their estimator stated... first one that comes to hand` | 口语化元话语，具体两项未点名 | `改`为直接写 ordering and property tests require distinct evaluation quantities | 进入两个 estimator |
| DFT-07 | `2221-2224`，released ordering above best、13/23 at minimum、what it does/does not measure | metric distinction 与 positive result 同句组 | `拆`：ground-state metric definition；13/23 result；why this metric does not decide disorder | 清楚区分 ground state 与 thermal order |
| DFT-08 | `2224-2227`，random-ordering cost、0 vs 18/23 at 300 K、figure | estimator、result、navigation 同句 | `拆`：order-disorder estimator；control/GNoME counts；figure pointer | 突出 thermodynamic test |
| DFT-09 | `2229-2232`，400-GPa proxy target、0.940 factor、absolute threshold consequence、0.7% | calibration 与 absolute-scale failure 一段 | `拆`：target scale；median calibration factor；why raw threshold is invalid；0.7% illustration | 说明为何需 rescale |
| DFT-10 | `2233-2235`，r=0.769、ranking transfer、rescaled selection | correlation、transfer object 与 result同句 | `拆`：correlation；ranking-not-absolute transfer；rescaled selection outcome | 明确验证的对象 |
| DFT-11 | `2237-2241`，260 CIF、one per candidate、index five fields | data availability 和长字段列表同句 | `拆`：CIF availability；index before/after fields；PSS/modulus fields | 清楚说明 Supplementary Data |
| DFT-12 | `2241-2242`，`Two things ... worth stating...` | 元话语 | `删`，直接进入 composition result | 减少过渡冗余 |
| DFT-13 | `2242-2244`，11 elements、Os/Ir/Re dominance | 清楚 | `保留` | 报告 periodic-table concentration |
| DFT-14 | `2244-2246`，句首 `And` 与 Law 7 symmetry contrast | 衔接口语化、主语绕 | `改`为直接句：relaxation changes the symmetry used for Law 7 assessment | 转入 before/after result |
| DFT-15 | `2246-2248`，61→113、none reverse、merging explanation | 数值、direction 与 mechanism 同句 | `拆`：before/after counts；no reverse moves；site-merging mechanism | 清楚说明 relaxation effect |
| DFT-16 | `2248-2252`，57.9% vs 11.7%、unrelaxed score anticipates、construction could not arrange | subgroup result 和 independent-check conclusion同段 | `拆`：retained/removed comparison；predictive relation；independence of check | 强力收束 property-screen validation |
| DFT-17 | `2254-2258`，four-parameter/five-point fit、leave-one-out shift、three-orders comparison | fit degrees、sensitivity test 和 magnitude 同段 | `拆`：one degree of freedom；drop-one procedure；median shift；relative magnitude | 清楚报告 modulus-fit stability |

# C. 全文段落覆盖表

此表用于保证实施时不是“只改几个典型例子”。`详列`表示该段中每个需要改的句子已在上表定位；`保留`表示逐句读过但未发现属于“啰嗦、难懂、逻辑不连贯”的问题。

| 文件 | 覆盖范围 | 状态 | 对应编号 |
|---|---|---|---|
| `main.tex` | active author/front/body/methods includes，以及 acknowledgements、data/code availability | 详列或明确保留 | M-01–M-06 |
| `front_meta.tex` | title 与 abstract 全部句子 | 详列 | FM-01–FM-12 |
| `front_body.tex` | Introduction 全部段落 | 详列或逐句保留 | I-01–I-37 |
| `body.tex` | Results 2.1–2.5 与 Discussion 全部段落、全部 figure captions | 详列或逐句保留 | R-H1–R-H5；R-01–R-141 |
| `methods.tex` | opening、五个 subsections 的全部段落 | 详列或明确保留 | ME-01–ME-68 |
| `si.tex` | SI title、abstract 与 wrapper text | 详列或明确保留 | SW-01–SW-04 |
| `si_body.tex` | S1–S19、DFT note 的全部 prose paragraphs、captions 和 table-introduction sentences | 详列或明确保留 | S1-01–S19-09；PU-01–PU-33；DFT-01–DFT-17 |
| `si_body.tex` | 纯公式、纯数值表格单元、LaTeX layout commands | 保留 | 不作语言修改 |

# D. 用户确认后的实施顺序

1. 先修改 `front_meta.tex` 与 `front_body.tex`，把标题、摘要和 Introduction 的 narrative spine 定住。
2. 再按 R-01–R-141 修改 `body.tex`；所有 headline results、数值和强结论保持不变，只做拆句、去重、指代修复和段落重排。
3. 按 ME-01–ME-68 修改 `methods.tex`，优先澄清 split、denominator、no-verdict、PSS fitting 与 DFT protocol。
4. 同步 `si.tex` 标题，并按 B1–B20 修改 `si_body.tex`；每处主文压缩后需要保留的技术细节仍留在 SI。
5. 完成一次术语一致性检查：PRIS/PSS、satisfaction/damage detection、fail/not satisfy/no verdict、discovery/held-out/reserve、deposited/as-given/primitive/relaxed cell。
6. 编译主文和 SI，检查 cross-reference、表格/图注溢出、分页和标题同步；只修本次语言改动导致的问题。
7. 交付逐文件 diff，并附一份“原句 → 修改句 → 修改理由”核对表，供作者逐项接受或回退。

# E. 实施时的不可变约束

- 不降低结论强度，不把 `proves`、`can`、`shows` 等强结论自动改成弱化措辞。
- 不删除 headline numbers，不更改任何数值、公式、样本量、阈值、引用或图表结论。
- 不改变论文现有发现顺序：autonomous discovery → laws/mechanisms → benchmark performance → PSS/synthesizability → inverse design/external diagnosis → broader significance。
- 不审阅或修改本计划范围之外的文件。
- 在用户明确确认前，不修改上述任何论文源文件。
