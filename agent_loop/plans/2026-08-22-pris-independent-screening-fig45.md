# PRIS 独立筛选与 Fig. 4–5 集成实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标：** 将 L4 法则和冻结的合成公式 (S_{\mathrm{syn}}) 呈现为两条分别应用的预计算队列筛选路线，并重排 Fig. 5，使物理含义和任务专用排序角色自然承接 Fig. 4 的筛选结果。

**总体方案：** L4 只移除明确违反 PRIS 的结构；(S_{\mathrm{syn}}) 在实验结构上校准连续低分阈值。两条路线分别作出决定，主图不构造 L4 与公式的串联、并联或联合门。PU 低分结构是模型挑出的代理队列，不是已经证实合成失败的负例。现有损伤验证图保留在 SI，待新增图和正文叙事确认后再做正文集成。

**技术栈：** Python、pandas、NumPy、PyArrow、Matplotlib、LaTeX、pytest、SHA-256 清单。

## 已冻结的证据和解释口径

- 实验队列：99,162 个唯一 CIF。
- PU 低分队列：发布的 364,771 条记录中去掉 179 条重复 CIF 后，得到 364,592 个唯一 CIF；它是模型选择的代理队列，不是失败合成实验真值。
- L4 只移除 `explicit_violation`；`no_verdict` 始终留在后续队列中。
- L4 固定工作点：实验结构保留 80.6851%，PU 队列筛除 189,159/364,592 = 51.8824%。
- 在相同实验结构保留率下，独立校准的 (S_{\mathrm{syn}}) 筛除 305,075/364,592 = 83.6757% 的 PU 队列；在更保守的 95% 实验结构保留点，筛除 74,142/364,592 = 20.3356%。
- “相同保留率”按实际严格阈值理解：L4 保留 80,009/99,162，公式保留 80,010/99,162，仅相差 1 个实验结构，图注写作 approximately matched。
- (S_{\mathrm{syn}}) 使用冻结系数、冻结标准化参数和训练集冻结中位数填补缺失项。六项全部观测的结构只有实验组 21,477/99,162（21.66%）和 PU 组 135/364,592（0.037%）；这部分支持度放在 SI/方法中，不能把全队列结果写成普适精度。
- 在独立的机制审计中，D7 覆盖 L4 所筛除 PU 结构的 186,741/189,159 = 98.72%。这是解释性覆盖率，不是第二条筛选路线。

## 当前已经生成的新增产物

- `experiments/pu_synthesizability_20260821/independent_screening.py`：两条独立决策路线的计算和工作点函数；不返回任何联合决定。
- `experiments/pu_synthesizability_20260821/plot_independent_screening.py`：生成英文 Fig. 4 草图。
- `experiments/pu_synthesizability_20260821/plot_draft_fig5.py`：复用当前冻结面板实现，按建议顺序生成英文 Fig. 5 草图。
- `outputs/20260822_pu_formula_scores/independent_choices_v1/`：CSV、JSON、英文 PNG/PDF、中文结果说明和 SHA-256 清单。
- `tests/test_pu_synth_independent_screening.py`、`tests/test_pu_synth_draft_figures.py`：决策逻辑和图形渲染冒烟测试。

## 任务 1：保持两条独立决策路线的接口

**文件：**

- 修改：`experiments/pu_synthesizability_20260821/independent_screening.py`
- 测试：`tests/test_pu_synth_independent_screening.py`

**步骤：**

1. 保持 `build_independent_frontier()` 的行为：每个给定保留率只校准公式阈值，并额外加入一次自然 L4 工作点；不得增加 `L4 OR S_syn` 或 `L4 AND S_syn` 行。
2. 保持 `build_operating_point_summary()` 的行为：在 L4 实验保留率处单独校准公式，并断言所有输出行的 `combined == False`。
3. 每一行 CSV 都保留实验队列和 PU 队列的总数，避免把结构数误读成百分比。
4. 运行 `python -m pytest -q tests/test_pu_synth_independent_screening.py`，预期 2 个测试通过。

## 任务 2：冻结并核验 Fig. 4 草图

**文件：**

- 修改：`experiments/pu_synthesizability_20260821/plot_independent_screening.py`
- 生成：`outputs/20260822_pu_formula_scores/independent_choices_v1/`
- 测试：`tests/test_pu_synth_draft_figures.py`

**步骤：**

1. 所有读者可见的图中文字使用英文；中文只放在结果报告和计划中。
2. 保持 a→b→c→d 的叙事链：
   - **a, Independent pre-DFT operating choices：** (S_{\mathrm{syn}}) 的连续工作曲线；L4 是一个固定星标。
   - **b, Matched experimental retention：** L4 和 (S_{\mathrm{syn}}) 在约 80.69% 实验结构保留率下分别计算。
   - **c, Queue length after pre-screening：** L4、同保留率的 (S_{\mathrm{syn}})、以及 95% 保留率的 (S_{\mathrm{syn}})；三个柱子是三个独立选择，不能解释成串联结果。
   - **d, Continuous score and mechanism：** (S_{\mathrm{syn}}) 分布、L4 对应公式阈值、冻结中位数填补说明，以及独立的 D7 解释性覆盖率。
3. 运行：

   ```bash
   python -m experiments.pu_synthesizability_20260821.plot_independent_screening \
     --output-dir outputs/20260822_pu_formula_scores/independent_choices_v1
   ```

4. 运行 `python -m pytest -q tests/test_pu_synth_independent_screening.py tests/test_pu_synth_draft_figures.py`。
5. 逐图检查 PNG：不得出现中文、字面量 `\\n`、队列柱顶文字重叠，也不得出现任何标记为 combined 的曲线或柱子。

## 任务 3：冻结建议的 Fig. 5 顺序

**文件：**

- 修改：`experiments/pu_synthesizability_20260821/plot_draft_fig5.py`
- 生成：`outputs/20260822_pu_formula_scores/independent_choices_v1/pris_draft_fig5_physical_meaning.{png,pdf}`

**步骤：**

1. 保留现有标题中的关键词，并使用以下 a→b→c→d 顺序：
   - **a, Two task-specific projections of PRIS mechanisms：** (S_{\mathrm{stab}})（凸包能量排序）与 (S_{\mathrm{syn}})（实验记录排序）的标准化系数。正文要明确它们是同一连续机制描述符空间面向不同监督标签的投影，不是二元 L4 状态加权得到的总分。
   - **b, Strong damage detection does not imply polymorph ranking：** commitment plane；“平局”表示没有唯一选择，不是错误。
   - **c, Confidence-dependent formula accuracy：** 合成公式和凸包能量的置信度曲线，保留现有 exploratory 限定。
   - **d, Energy, phonons and experimental records select different structures：** 现有三轴 ladder，作为矛盾收束。
2. 如果版面拥挤，旧 Fig. 5b 的 tie-rate 柱可以作为 b 的 inset 或移 SI；旧 Fig. 5c 的 wrong-hull-pair 面板在 Fig. 4 使用下游能量参照后优先移 SI。
3. 运行 `python -m experiments.pu_synthesizability_20260821.plot_draft_fig5`，并逐图检查所有文字为英文、Fig. 5d 下半部分没有裁切。

## 任务 4：评估下游能量参照，不把它误写成第三条预筛路线

**文件：**

- 检查：`src/next15_basin_hull.py`、`outputs/20260802_next15_wbm_basin_hull_retrospective/`、二元队列清单和本地 CIF 数据源。
- 如果确认需要新增试验，只能新增独立 pilot 脚本和输出目录，不修改正文。

**已知审计结论和执行规则：**

1. PU CIF 本地完整可得：`CSAgent/data/from_hpc/release/negatives/train.csv` 和 `val.csv` 共 364,771 条，`material_id` 与二元队列一一对应；实验 CIF 可由 `structures.blob` 和 `experiments/pu_synthesizability_20260821/data.py::decode_blob_cif` 解码。
2. 现有 MatterSim 协议计算的是 MatterSim 松弛能量减去 MP 参考凸包的 **MLIP basin–hull proxy**。若没有同一批结构的 DFT 能量，绝不能把它标成 DFT (E_{\mathrm{hull}})。
3. 当前 MP 参考只覆盖 2,722/91,561 个 PU 化学体系（约 2.97%），对应 12,239/364,771 条记录（约 3.35%）。因此不要把现有 proxy 曲线放进 Fig. 4 主比较；最多在 SI 或 Fig. 4c 以灰色 downstream reference 展示，并同时报告 supported-subset coverage。
4. 如运行 bounded pilot，阈值必须只由实验结构校准，再应用到 PU；它是下游参照，不是第三个 pre-DFT 用户选项，也不能与 L4 或 (S_{\mathrm{syn}}) 合并。
5. 如果数据支持或 GPU 运行不稳定，直接记录阻塞原因，不用不完整结果填补主图。

## 任务 5：得到确认后再集成正文

**未来才允许修改的文件：**

- 新建或修改：`src/fig4_pre_dft.py`（建议与旧 damage 图绘图函数分离）。
- 修改：`src/fig5_ranking.py` 的面板调用顺序和 caption 数据。
- 修改：`tex/body.tex`、`tex/front_body.tex`、`tex/si_body.tex`、`paper/FACTS.md`。

**集成步骤：**

1. 先把当前 Fig. 4 的 damage-validation 证据复制为 SI 图或紧凑 inset，不得静默删除。
2. Fig. 4 caption 标题保留现有完整短语并追加：`PRIS improves screening before expensive calculations: selected bounds detect damage types omitted during selection; the queue comparison adds two separately applied choices, an interpretable L4 gate and a continuous synthesis score.` 如果旧 damage 面板完全移到 SI，正文标题仍保留前半句，后半句作为新增分句。
3. Fig. 5 caption 标题保留现有完整短语并追加：`What structural plausibility can and cannot decide. Task-specific formulas extend this diagnosis.`
4. Fig. 4 结果后加入英文桥接句：队列实验说明两种用户选择的效用，但不是合成成功率；L4 命名违反机制，(S_{\mathrm{syn}}) 提供面向实验记录标签的可调排序。
5. Fig. 5a 紧接解释两条公式的监督任务和连续描述符来源，再按 b/c/d 讨论多晶型平局、置信度排序和能量–声子–实验记录矛盾。
6. 删除或改写正文中空泛的“四种性质各不相同”句子，替换成带有当前队列数、保留率和任务标签的具体桥接。
7. 更新所有正文与 caption 的子图引用，逐一核对 a→b→c→d；旧 Fig. 4 的 omission/damage 面板引用全部指向 SI（如决定移出主文）。
8. 最后重建正文和 SI PDF，运行事实、渲染、`pdftotext`、PDF 页数、SHA-256 和 `git diff --check` 审计。

## 必须执行的核验命令

```bash
python -m pytest -q \
  tests/test_pu_synth_independent_screening.py \
  tests/test_pu_synth_draft_figures.py \
  tests/test_pu_synth_formula_score_results.py \
  tests/test_pu_synth_formula_score_figure.py
python -m experiments.pu_synthesizability_20260821.plot_independent_screening \
  --output-dir outputs/20260822_pu_formula_scores/independent_choices_v1
python -m experiments.pu_synthesizability_20260821.plot_draft_fig5
cd outputs/20260822_pu_formula_scores/independent_choices_v1
sha256sum -c SHA256SUMS
cd - >/dev/null
python - <<'PY'
from pathlib import Path
for p in Path('outputs/20260822_pu_formula_scores/independent_choices_v1').glob('*.pdf'):
    assert p.stat().st_size > 1000, p
print('draft PDFs have non-trivial size')
PY
```

本阶段在生成新增数据、报告和英文草图后停止，不修改 canonical `tex/`、正文图源或现有 PDF。正文集成前必须保留当前 dirty worktree，并把旧 PDF 复制到 SI/backup 后再替换主图。
