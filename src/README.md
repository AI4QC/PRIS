# PRIS/src —— Week 1 数据脚本

统一环境:`python`(不新建 conda 环境)。
并行前必须 `export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`。
产物目录:`$PRIS_FEATURES/`。

---

## build_energetics.py

补 `materials.sqlite` 里全为 NULL 的能量字段的主路径。

```bash
python src/build_energetics.py --limit 2   # 冒烟
python src/build_energetics.py             # 全量,实测 41 s
python src/build_energetics.py --force     # 忽略幂等跳过
```

产物:
| 文件 | 行数 | 大小 | 粒度 |
|---|---|---|---|
| `energetics_mp.parquet` | 154,377 | 14.0 MB | 一个 MP 材料一行(25 列) |
| `icsd_mp_link.parquet` | 73,823 | 6.6 MB | 一个 ICSD 实验结构一行,带回填的 E_hull |

### 实测 schema 要点(与计划 §14 骨架的差异)

- MP summary 每页 69 个顶层字段,`{"data": [...1000 条...], "meta": {...}}`。
- `material_id` 是 **字母 hash**(`mp-aaahikie`),不再是老版的 `mp-1234`。任何按数字解析 mp-id
  的下游代码都会挂。
- `database_IDs` 对绝大多数条目是 **`null`**(不是空 dict)。计划里写的
  `d.get("database_IDs").get("icsd")` 会 `AttributeError`,必须写
  `(d.get("database_IDs") or {}).get("icsd") or []`。
- `database_IDs["icsd"]` 的元素实测 **100% 带 `icsd-` 前缀**(`exp004`),与
  `synth_meta.tsv` 的 `source_id` 同形。但连接键仍统一转成整数 collection code,
  对将来混入 `ICSD-xxx` / 纯数字的写法更鲁棒。
- 另有 `database_IDs["pauling"]`(15,765 条,Pauling File)——项目同名,但不是 Pauling 五定律,
  别混淆。当前未使用。
- `symmetry` 是 dict,空间群号在 `symmetry.number`,不是顶层 `spacegroup_number`。
- `deprecated == True` 的条目实测为 0 条,快照已经是清洗过的。

### 已知数据坑

- **4 条 `energy_above_hull` / `formation_energy_per_atom` / `band_gap` 为 NULL**,全是单质 `Yb`
  (mp-aaacdikq / mp-aaacppfn / mp-aaaaaact / mp-aaaaaagg)。MP 的 Yb 赝势有已知问题。
  其中两条带 ICSD ID(exp005、exp003),导致「按 ID 连上 58,246 条」但
  「拿到 E_hull 只有 58,244 条」。
- **`theoretical=True` 却带 ICSD ID:2,333 条**(占带 ICSD ID 条目的 4.52%)。ICSD 是实验库,
  这在语义上是矛盾的。这些条目 e_hull 中位 0.110 eV/atom(远高于全实验集的 0.0034),
  推测是 MP 对高能多形体/歧义匹配打的保守标签,或 ICSD 收录的假想结构。
  **下游筛"实验已知"时不要只信 `theoretical` 字段,要 `theoretical==False OR n_icsd>0` 并记录旗标。**
- 一个 ICSD code 可能被多个 MP 条目引用(实测 481 个)。`build_icsd_link` 取 **e_hull 最小**
  的那条作代表,并在 `n_mp_matches` 列保留匹配数供下游判断歧义。

### 幂等

产物存在且 mtime 晚于所有输入(155 个 gz + synth_meta.tsv)时直接跳过重算,
但仍会重新读 parquet 并打印完整报告。

---

## build_provenance.py

实验集溯源表。输入 `materials.sqlite`(只读)+ `synth_meta.tsv`,输出
`features/provenance.parquet`(99,162 行 × 46 列,11.1 MB zstd,全量 ~3 s)。

```bash
python src/build_provenance.py --limit 200 --dry-run   # 冒烟
python src/build_provenance.py                         # 全量
python src/build_provenance.py --force
```

### 实测 vs 预期

| 项 | 实测 | 预期 | |
|---|---|---|---|
| 行数 | 99,162 | — | sqlite `dataset='experimental'` 全量 |
| join 命中率 | **100.0000%** | 主 agent 未验证过 | 见下 |
| `n_sites == n_atoms` | 100% | — | join 正确性的独立佐证 |
| `orig_spg` 缺失 | 394(0.40%) | — | bawl_hash 中段为空 |
| `orig_spg == spacegroup_number` | **98.9446%**(98,727 可比,1,042 不一致) | — | 见下 |
| `in_analysis_set` | **38,307**(icsd 27,408 + cod 10,899) | 38,307 | 完全复现 |
| `oxide_strict` | **23,728**(icsd 16,414 + cod 7,314) | 23,728 | 完全复现 |

### 连接键(已验证)

`materials.source_index` 是 TEXT 存的整数行号,实验集内 0..99161 连续且唯一;
`synth_meta.index` 同样 0..99161 唯一。二者是严格双射,100% 命中,零缺失。
`source_split`(train 94,204 / val 4,958)**不**参与编号,不需要按 split 分段偏移。
计划 §14 里写的 SYN 路径是 CSAgent 下的旧副本,现役副本在
`matdata/data/sources/experimental/synth_meta.tsv`,脚本以后者为准。

### orig_spg vs spacegroup_number

`bawl_hash` = `<md5 32位>_<原始空间群号>_<还原式>`,例如
`f5e701d27fce4666407370fabff735f6_14_Cr4Te8O22`。

1,042 条不一致中 **100% 是 sqlite 的号 > bawl 的原始号**,零反向。即 sqlite 侧用了更宽松的
symprec,把结构判进了更高对称的群。高频迁移:14→62、123→221、139→225、11→63、2→12、
47→123、69→139、160→215——都是标准的「容差放宽后子群升母群」路径,不是数据错乱。
不一致按来源:exp008 / cod 256。

空值:sqlite `spacegroup_number` NULL 435 条,`orig_spg` NA 394 条,后者是前者的**真子集**
(394 条两边都判不出,另有 41 条 bawl 判得出而 sqlite 判不出)。

**用途**:`spg_agree` 列(nullable boolean)是 symprec 敏感性的现成代理变量。
那 1,042 条处在对称性判定边界上,做定律检验时应单独看,免得把 symprec 伪影读成物理规律。

### 两个分析集口径 —— 注意它们不是包含关系

- `in_analysis_set`:{O,S,Se,Te,N,P,F,Cl,Br,I} **恰好含一种**,且不含 H、不含 C。
  阴离子分布:O 19,833 / S 4,783 / F 2,839 / Se 2,745 / N 1,673 / P 1,663 / Te 1,558 /
  Cl 1,434 / I 909 / Br 870。
- `oxide_strict`:含 O 且不含 H/C/N/F/Cl/Br/I/S/Se/Te。**P 不在排除列表里。**

交叉:`ox ∩ set` = 19,833(= 单一阴离子且该阴离子是 O);`ox \ set` = 3,895,
**这 3,895 条全部含 P**(磷酸盐被 `in_analysis_set` 当成双阴离子剔除,却被 `oxide_strict` 保留);
`set \ ox` = 18,474(非氧的单一阴离子体系)。下游选样本必须明确挑哪一个。

### 输出列

`pk, material_id, source, source_id, bawl_hash, bawl_md5, bawl_formula, orig_spg,
spacegroup_number, spacegroup_symbol, crystal_system, spg_agree, formula, chemical_system,
elements, n_elements, n_atoms, n_sites, anion, n_anion_kinds, in_analysis_set, oxide_strict,
blob_offset, blob_length, dataset, source_split, source_index, is_experimental, lattice_*`,
外加 12 个 `has_<元素>` 布尔列(O/S/Se/Te/N/P/F/Cl/Br/I/H/C)。
`blob_offset`/`blob_length` 直接指向 `structures.blob`,下游取 CIF 不必再回 sqlite。

### 内置断言(改口径时会响)

join 命中率 < 99.9%、`n_sites`/`n_atoms` 一致率 < 99%、`source_index` 非数字或重复、
`bawl_hash` 不是 3 段 → 抛异常;全量跑时 `in_analysis_set != 38307` 或
`oxide_strict != 23728` → 抛 AssertionError。`--limit` 时跳过后两条。

---

## build_icsd_meta.py

从 203,830 个 ICSD 原始 CIF(`<other-repo>/data/icsd_extracted/cif/*.cif`,1.1 GB,CRLF 行尾)
纯正则抽元数据,产出 `features/icsd_meta.parquet`(203,830 行 × 15 列,3.7 MB)。§10.1 年代自指分析的输入。

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=python
$PY src/build_icsd_meta.py --limit 4000        # 冒烟(等距 stride 抽样,~1 s)
$PY src/build_icsd_meta.py --workers 16        # 全量,实测 27 s
$PY src/build_icsd_meta.py --force             # 产物已最新时强制重算
```

幂等判据:产物 mtime > max(CIF 目录 mtime, 本脚本 mtime) 就跳过(不逐个 stat 20 万个文件)。

### 列说明与下游陷阱

| 列 | 说明 |
|---|---|
| `source_id` | `icsd-<N>`,与 `synth_meta.tsv` 的 `source_id` 同口径,可直接 join |
| `pub_year` | **文献发表年**,来自 citation loop 的 `_citation_year`。时间轴只能用这一列 |
| `audit_year` | **FIZ 录入年**,来自 `_audit_creation_date`。**唯一合法用途是录入批次固定效应** |
| `lag_years` | `audit_year - pub_year`,只为复核 §10.1.2 的分位;**不要**用它补 `pub_year` 空缺 |
| `temperature` | 单位 **K** |
| `pressure_kpa` | 单位 **kPa**。要 GPa 必须 **除 1e6** |
| `R_factor` | `_refine_ls_R_factor_all`,正常 0–1;实测有 50 条 >1(按百分数录入),建模前自己截断 |
| `n_reflns` / `n_params` | **实测全空**,见下 |
| `n_year_cand` | 诊断列,该文件命中的不同年份数;实测全语料恒为 0 或 1 |

### 实测 vs 计划预期(全量 203,830,2026-07-28)

命中率 6 项全部与计划 §14 逐位吻合(pub_year 96.73% / audit_year 100% / R_factor 64.73% /
temperature 35.02% / pressure_kpa 3.64% / structure_type 78.13%);lag 分位 p5/p25/p50/p75/p95/p99/max
= 0/1/2/17/44/57/95 与 §10.1.2 完全一致;pub_year 与 audit_year 的十年分桶亦逐桶一致。

### 三处与计划不符的地方

1. **`n_reflns` / `n_params` 全空。** 计划 §14 的 PAT 字典里写了
   `_refine_ls_number_reflns` / `_refine_ls_number_parameters`,但本机 ICSD 语料**根本没有这两个字段**
   (5,000 文件随机抽样 grep 0 命中;整个语料里 `^_refine*` 只存在 `_refine_ls_R_factor_all` 一种)。
   两列保留只为 schema 兼容,脚本会打 warning,**下游不要用**。§10.1.3 的 quality 协变量只能靠
   `R_factor` + `has_aniso_adp` + `is_powder`。
2. **`--limit` 默认是等距 stride 抽样,不是取前 N 个。** ICSD 编号与年代强相关(前 500 号全是
   1970 年代、audit 全是 1980),`--limit 500 --limit-mode head` 会给出 temperature 5.8% 这种
   完全误导的命中率。冒烟必须用 stride(默认)。
3. **"citation loop 多条时取最早"这条规则在本语料里从不触发**(`n_year_cand > 1` 恒为 0 条)。
   代码里的 `min(yrs)` 保留,但它没有实际作用。

### 已知脏数据(脚本只报告不清洗,保持与原始 CIF 一一对应)

- `lag_years < 0` 114 条(0.058%),其中 104 条是 −1(多半是 in-press 录入,非正则错误),
  少数 −14/−19 是正则抓错卷号。占比可忽略,建模时按 `lag_years >= -2` 过滤即可。
- `temperature` 上限 7,493 K、`pressure_kpa` 上限 1.7e9 kPa(=1,700 GPa)均为 ICSD 原始录入错误。
- `pub_year` 空缺 6,672 条(3.27%),这些条目的 citation loop 排版不属于已知两种。

---

## build_cod_delta.py

§8.3 L3 时间外推留出的唯一救命路径:从 COD 全库里筛出 matdata 尚未收录的干净无碳 post-2019 条目。

```bash
python src/build_cod_delta.py --limit 3000 --min-year 2004   # 冒烟(会真的走一遍 cif 抓取)
python src/build_cod_delta.py                                # 全量,实测 151 s / 峰值 RSS 6.0 GB
python src/build_cod_delta.py --no-cif                       # 只出计数和直方图,秒级
```

产物:

| 文件 | 行数 | 大小 | 内容 |
|---|---|---|---|
| `cod_delta.parquet` | 2,439 | 386 MB | post-2019 增量,26 列 + `cif_text` |
| `cod_delta_meta_all.parquet` | 50,122 | 3.2 MB | 全部"clean 无 C 且不在 matdata",**不含 cif_text**,给 §8.3 数据源留出用 |
| `cod_year_hist.csv` | 39 | 1.2 KB | 5 个筛选级 × year(1990–2026,外加 `-1`=<1990、`-2`=year 缺失两行) |
| `cod_delta_stats.json` | — | — | 各级计数 / 耗时 / 峰值 RSS / 计划预期值,便于逐轮对账 |

### 30 GB 文件的内存策略(本脚本的头号风险)

实测:压缩 30.4 GB 里 `cif_text` 一列独占 **30.3 GB(99.7%)**,其余 74 列合计仅 0.1 GB;
解压后全表 **117 GB**,54 个 row group 大小极不均匀(rg0 = 0.08 GB,rg51 = 9.4 GB)。

- **Stage 1(筛选)**:duckdb `SELECT * EXCLUDE (cif_text)`,整表拉进 pandas 只占 0.19 GB,**1.0 s**。
- **Stage 2(取 CIF)**:pyarrow `iter_batches(batch_size=256, use_threads=False)`,
  `ParquetFile(..., pre_buffer=False)`。**144 s,峰值 RSS 6.0 GB。**
- **为什么 Stage 2 不用 duckdb**:duckdb 靠页级索引能把这步压到 ~12 s,但它对大字符串页做的是
  不可 spill 的裸分配,`memory_limit` 给 8 GB 和 12 GB 都实测抛
  `failed to allocate data of size 512.0 MiB`;给到能跑通的 10 GB 时峰值 RSS 已到 8.2 GB。
  在一台非独占的 23 GB 机器上不值得。
- **pyarrow 也要关 `pre_buffer`**:开着全量流式读峰值 RSS 8.3 GB,关掉后最大 row group 只到 4.2 GB。
- `file` 列在 row group 间**不有序**(rg0 min/max = 1000000/7206032,rg53 = 1576311/7720906),
  统计量剪枝无效;改为先只读 `file` 列(<10 MB)预扫一遍定位需要的 row group。
  全量时命中 **17/54**(post-2019 条目集中在后段),冒烟时常常只需 1 个。

### 无碳判据:与计划正则的差异

`formula` 实测格式是 `- C5 H17 Al N2 O8 P2 -`(前后各一个 `-`,元素按 Hill 序、空白分隔),
533,486 行里 **532,993 行**符合,余下 **493 行是字面量 `?`**。

计划写的 `~formula.str.contains(" C[0-9 ]")` 在这个格式上其实**没有失效**——本脚本改用的
token 化判据(trim 掉 `- ` → 按空白切 → 每个 token 取 `^[A-Z][a-z]?` 作元素符号 → 判是否 `== "C"`)
与它**逐行一致**(74,029 == 74,029)。两者唯一的差异是 **`formula = "?"` 的 493 行**:
正则把它们当"无碳"收进来,token 判据用额外的 `formula_ok` 标记把它们剔掉(成分未知 ≠ 无碳)。
所以 `clean_noC` = 73,536(token)vs 74,029(计划正则),差 493 全部来自这一条,且**全在 2019 年以前**,
对 post-2019 分母没有影响。

token 判据更稳的地方在于:它不依赖 "C 后面必须紧跟数字或空格" 这个位置假设,
formula 若哪天不带首尾 `-`、或元素与计数之间加了空格,正则会静默漏判而 token 判据不会。

### 实测 vs 计划预期(全量,2026-07-28)

| 筛选级 | 计划【实测】预期 | 本次实测 | 差 |
|---|---|---|---|
| 总行数 | 533,486 | 533,486 | 0 |
| clean(`status` 空) | 526,854 | **533,284** | +6,430 |
| clean 且无 C | 72,197 | **73,536**(计划正则口径 74,029) | +1,339 / +1,832 |
| 且不在 matdata | 48,782 | **50,122** | +1,340 |
| **且 post-2019** | **2,454** | **2,439** | **−15** |

**差异归因(未能完全复现,如实记录)**:`status` 只有四种取值,`''` 533,284 / `retracted` 151 /
`warnings` 41 / `errors` 10,且**从不为 NULL**,所以按计划代码 `status.isna() | (status=="")`
字面执行只能得到 533,284。穷举了 `duplicateof IS NULL` / `optimal IS NULL` / `onhold` /
`year IS NOT NULL` / `formula<>'?'` / `flags LIKE '%has coordinates%'` 的全部 64 种组合,
**没有任何一种同时给出 (526,854, 72,197)**,最接近的是 `nodup+noopt+hasyear` → (527,213, 72,065)。
判断:计划里那两个数多半来自更早的快照或一次带额外过滤的临时统计,不是本快照可复现的口径。
**关键数字 post-2019 = 2,439 与预期 2,454 只差 15(0.6%),且对判据变体极不敏感**
(去掉 status 过滤仍是 2,439,再去掉 `duplicateof` 重复条目是 2,438),结论不受影响。

### year 直方图要点

2019+ 的 5 个筛选级小计:all 98,448 → clean 98,442 → clean 无 C **3,864** → 不在 matdata **2,439**
(其中 1,703 条 2019+ 已经在 matdata 里了)。逐年:
2019 411 / 2020 466 / 2021 507 / 2022 316 / 2023 224 / 2024 186 / 2025 277 / 2026 52。
COD 的无机(无 C)条目占比从 1990 年代的 ~50% 一路跌到 2020 年代的 ~3%,**COD 越来越是有机/MOF 库**,
这是 2,439 这个分母上不去的根本原因,不是筛选写错了。

组成级的可分析子集(无 H/D 且恰好一种阴离子,**尚未过无序过滤**):**1,072 条**,
落在计划 §14【外推】的 700–1,100 区间上沿。阴离子构成 O 750 / S 112 / Se 84 / P 31 / F 27 /
I 21 / Te 20 / Cl 19 / N 6 / Br 2。脚本把 `has_H` / `n_anion_kinds` 两列直接写进两个 parquet。

### 其他实测细节

- `year` 范围 1915–2026,空 619 条(0.12%);2026 年已有 3,059 条(快照收录至 2025-08-21 之后仍有回填)。
- matdata 的 25,339 条 COD **全部**能在 `cod_full.parquet` 里按 `file` 对上,匹配率 100%。
- 2,439 条 delta 的 `cif_text` **100% 含 `_atom_site_fract_x`**,可直接建结构;
  中位长 33.8 KB、均值 356 KB、最大 47 MB(含 Fobs 表的条目)。
- `--limit N` 是取**前 N 行**(不是抽样),而 COD 的 `file` 编号与年代相关,所以冒烟时计数无意义,
  只用来验证代码路径;冒烟要触发 Stage 2 需要配合调低 `--min-year`。

---

## build_features.py —— MPU-1 特征库(site / pair / struct)

**环境:`python`**(不是本文件开头那个 csagent 环境;
newpauling 环境才有 pymatgen + spglib 2.7.0 + pyfixest)。

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python src/build_features.py --extract-icsd-ox --force   # 0. 一次性,11 s
python src/build_features.py --limit 400 --workers 20 --chunk 10 --force   # 1. 冒烟,1.4 min
nohup python src/build_features.py --workers 20 --chunk 20 --force \
      > $PRIS_FEATURES/build_features.log 2>&1 &   # 2. 全量
tail -f $PRIS_FEATURES/build_features.log          # 查进度
```

输入:`provenance.parquet` 的 `in_analysis_set==True` 共 38,307 条(ICSD 27,408 / COD 10,899);
CIF 按 `blob_offset`/`blob_length` 从 `structures.blob` seek+read 再 **zlib** 解压(不是 zstd)。

### 氧化态前置问题:实测结论(本节是本脚本最重要的产出)

PREREG §5 冻结"只允许 `cif`(ICSD 原生)与 `guess`(纯组成推断)两级,BVAnalyzer 派生整批排除"。
`icsd_meta.parquet` 里没有氧化态,所以先查了原始 CIF。四条实测:

| 问题 | 实测 | 口径 |
|---|---|---|
| ICSD 原始 CIF 有 `_atom_type_oxidation_number` 吗 | **27,408 / 27,408 = 100.00%** | 分析集全部 ICSD 条目,CIF 缺失 0、无氧化态 loop 0 |
| blob 里的 CIF 保留氧化态装饰吗(ICSD) | **27,408 / 27,408 = 100.00%** | 同上 |
| blob 装饰 vs 原始 CIF 一致吗 | **847,608 / 847,608 = 100.000%** | 按 `_atom_site_label` 对齐的位点级比对,容差 0.005 |
| blob 里的 CIF 保留氧化态装饰吗(**COD**) | **192 / 10,899 = 1.76%** | — |

结论:

1. **`cif` 这一级存在,但只覆盖 ICSD。** PREREG §5 不需要推翻,但需要一处**澄清**:
   `cif` 实际只到分析集的 **27,408/38,307 = 71.5%**,COD 的 28.5% 拿不到原生氧化态。
   建议写进 PREREG 修订记录(只追加,不改写)。
2. **直接用 blob 的装饰当 `ox_source='cif'`,不回原始 CIF。** 位点级 100.000% 一致,
   且位点级混合价(Fe2+/Fe3+、Bi3+/Bi5+、Se⁻¹/Se⁻²、Au+/Au3+ 等)**逐例保留、未被平均**,
   分数价(如 `Ru4.33+ 4.333`)也带全精度。回原始 CIF 只多一层 label 对齐风险,零收益。
   `icsd_ox.parquet`(239,756 行)保留为这条结论的证据与审计入口,不是主链路输入。
3. **COD 的 1.76% 装饰不可采信为 `cif`。** 本地没有 COD 原始 CIF
   (`matdata/data/raw/acquired/` 下无 cod 目录,只有 tcod),来源无法核验——
   可能是 COD 沉积自带的,也可能是上游 matdata 管线用 BVAnalyzer 补的。
   PREREG §5 明令 BVAnalyzer 派生整批排除,**无法证伪即不得使用**。
   → COD 全部走 `guess`,同时留 `blob_ox_present` 一等列供事后审计。
4. **全程不调用 BVAnalyzer**(PREREG §5 冻结项)。定不出价的结构 `ox_source='none'`,
   位点仍入库、`ox_state` 留 NaN,**不删**(§6.2 tier 4 口径)。

`guess` 用 `Composition.oxi_state_guesses(max_sites=-1)`(先约化到最简式再枚举,提速)。
返回多解时取 ICSD 频率打分最高的一解,并记 `n_guess_sol` / `guess_unique`
——§6.2 要求的"唯一解"口径可由下游用 `guess_unique==True` 还原,不必重跑。

### 三张主表

| 文件 | 粒度 | 说明 |
|---|---|---|
| `site.parquet` | 每个**阳离子**位点一行 | 三算法 CN 并列(宽表) |
| `pair.parquet` | 每对相连多面体一行 | 仅 ChemEnv 路线 |
| `struct.parquet` | 每个结构一行 | 汇总量 + `status` + `wall_ms` |
| `failure.parquet` | 每个失败结构一行 | `err_type` / `err` / 截断 traceback |

**关于 `nn_algo` 这一列**:`pair.parquet` 里它真实取值(目前恒为 `chemenv`,§9.1 指定用
`sc.environment_subgraph()` 的 `len(d['ligands'])`,CrystalNN 路线以后要加就再追加一批行)。
`site.parquet` 是**宽表**,`nn_algo` 恒为 `'multi'`,分层请用 `cn_chemenv` / `cn_crystalnn` /
`cn_brunner` 与 `ok_chemenv` / `ok_crystalnn` / `ok_brunner` 三对列。
选宽表不选长表的理由:G6(键定义鲁棒性)要的是**同一位点上三个算法的一致性**,
宽表可以直接 `df.cn_chemenv == df.cn_crystalnn`,长表还得自己 pivot 回去。

`ox_source` 在三张表里都是一等列。

### 算法参数(每个都注明来源)

- ChemEnv:`MultiWeightsChemenvStrategy.stats_article_weights_parameters()`,
  `maximum_distance_factor=1.41`,`only_cations=True` 且**显式传 `valences=`**(§6.3 **坑 A**:
  不传 valences 时 `only_cations=True` 返回垃圾)。`ox_source='none'` 的结构没有 valences 可传,
  退化为 `only_cations=False`(全位点跑,更慢),事后按"非该结构阴离子元素"筛阳离子位点。
- CrystalNN:`weighted_cn=False`,`x_diff_weight=3.0`(§6.3 **坑 B**:pymatgen 默认是 3.0 不是 1.5)。
- BrunnerNN_relative:默认参数。
- BVS:Brown-Altermatt `Σ exp((R0−R)/b)`,IUCr `bvparm2020.cif`(已固化到
  `<repo>/data/`)。**§6.3 坑 C:loop 头行有前导空格,解析前必须
  `strip()`**,否则整表解析为空、静默回退到 Brown 元素级通用式,GII 从 0.168 虚高到 0.479。
  脚本里加了哨兵:解析出 <1000 条直接 `RuntimeError`,不静默降级。
  截断固定 **3.5 Å**,与 CN 解耦(§6.3:用 CrystalNN 的离散邻居集会系统性截断长弱键)。
  只累加异号(阳-阴)对。GII 只对阳离子位点求和(Brown 原始定义)。
- `bvs_dev` 用 `n_bvs_bonds > 0` 判定有效,**不是** `if bvs`(§6.5-1 falsy bug:
  BVS 恰为 0 的位点会被判 nan,恰好掩盖最严重的违例)。
- 对称性:`SpacegroupAnalyzer` 两档 `symprec=0.1` / `0.01`,各给 `orbit_id` / `wyckoff` / `mult`。
- `I_G = −Σ pᵢ log₂ pᵢ`,`pᵢ = mᵢ/N` 按 Wyckoff 轨道(Krivovichev 结构复杂度信息熵,
  §8.1 用作"复杂结构位点多 → 机械性更易违规"这个混杂的控制变量),两档 symprec 各一份。

### 工程

- `ProcessPoolExecutor`,20 进程(§6.5-4:**不是 28**,每进程 300–500 MB,ChemEnv 高配位结构飙到 1.5 GB)。
- 每个 worker 自己开 blob 句柄读 CIF、自己写 parquet 分片,主进程只收计数
  (§6.5-2:pilot 的 `p.map` 把全部 dict 收回主进程再 `pd.DataFrame`,差几个数量级)。
- 分片合并用 `pq.ParquetWriter` 逐片流式写,不把全部行读进内存。
- **幂等**:已存在 `struct_<k>.parquet` 的分片直接跳过,断了重跑即续;`--force` 清空分片目录重来。
- **单结构超时 300 s**(`signal.setitimer`)。§6.5-3 原定 60 s,实测 ChemEnv 在 >200 原子胞上
  常态超 60 s,收紧到 60 s 会把大胞系统性剔除,那是 §6.4 式的选择偏倚。分析集 `n_atoms` 最大 300、
  >200 原子的只有 290 条,300 s 足够。超时/异常写 `failure.parquet` 并继续,整批不崩。
- `--limit N` 冒烟走**分层抽样**(按 ICSD/COD 比例),产物带 `_smoke` 后缀,不污染全量表。

### 全量实测(2026-07-28,20 进程,机器非独占)

**98.5 min 墙钟 / 32.4 核-时**,38,307 条全跑完。产物:

| 表 | 行数 | 大小 |
|---|---|---|
| `site.parquet` | **458,940** | 20.0 MB |
| `pair.parquet` | **338,135** | 7.8 MB |
| `struct.parquet` | **38,307** | 16.7 MB |
| `failure.parquet` | **62** | 0.2 MB |
| `icsd_ox.parquet`(审计用) | 239,756 | 0.7 MB |

分片 `_shards/`(110 MB)保留,断点续跑靠它;确认无误后可删。

**与 §6.6 预期的对比**

| 项 | §6.6 规划 | 实测 | 说明 |
|---|---|---|---|
| 单结构耗时 | 保守 2.0 s | **均值 3.05 s / 中位 1.19 s** | 均值被长尾拉高:P95 11.8 s、P99 27.8 s、max 320 s(1 条撞 300 s 超时上限) |
| 20 进程墙钟 | 2.3 h | **1.64 h** | 比规划**快 29%** |
| 核-时 | 45 | **32.4** | 比规划省 28% |
| 产物体积 | 1.0 GB | **45 MB** | 规划把 bonds 表算进去了,本轮三张表不含 bond 级 |

吞吐从起步的 9.5 struct/s 衰减到收尾的 6.5 struct/s——分片按 source_id 顺序切,
大胞结构在 ICSD 编号后段更密集,不是内存或竞争问题(全程 available ≥ 18 GB)。

**失败率**

- 结构级整体失败 **62 / 38,307 = 0.162%**(ICSD 34 / COD 28),**全部**是同一个原因:
  `ValueError: Invalid CIF file with no structures!` —— blob 里那 62 条 CIF 本身坏了,
  不是算法失败。已全部记入 `failure.parquet`,批次未中断。
- 算法级(在 38,245 条成功结构上):**ChemEnv 0.060% / CrystalNN 0.000% / BrunnerNN 0.000%**。
  §6.3 坑 A 说"不传 valences 时 ~17% IndexError 硬失败",显式传入后实测 **0.06%**,坑 A 的修复确认有效。
- 位点级 CN 缺失:`cn_chemenv` **7.45%** / `cn_crystalnn` 0.02% / `cn_brunner` 0.00%。
  ChemEnv 那 7.45% 是它在 `max_dist_factor=1.41` 内判不出环境(**不是异常**),
  三个 `ox_source` 层上分别是 7.61% / 7.75% / 4.11%,**与氧化态来源基本无关**,可视为随机缺失。
- CE 符号分配率 **92.56%**(424,772 / 458,940 阳离子位点)。

**氧化态覆盖(主统计的分母)**

| | cif | guess | none | 合计 |
|---|---|---|---|---|
| ICSD | 27,374 | 0 | 0 | 27,374 |
| COD | 0 | 9,382 | 1,489 | 10,871 |
| 合计 | **27,374 (71.6%)** | **9,382 (24.5%)** | 1,489 (3.9%) | 38,245 |

`cif+guess` 覆盖 **96.11%**,高于 §6.2 预期的"tier0+tier1 在实验氧化物上约 85%"。
`guess` 里唯一解占 **63.8%**(5,985 条),恰好落在 §6.2 说的"覆盖率约 63%"上——
但那 63% 在 §6.2 是指 `guess` 的**总**覆盖率,实测是 `guess` 内部的**唯一解**比例,
两者数值巧合、口径不同,**不要当成互相验证**。
混合价结构 1,087 条(2.84%),分数价 3,153 条——§6.2 担心的"FIZ 把混合价整数化导致漏报"
在本库里可用 `mixed_valence` / `frac_ox` 两列直接分层。

**三项交叉验证(§6.6 推全量前的闸门)**

| 指标 | 本轮实测 | 对照 | 判定 |
|---|---|---|---|
| 共角/共边/共面(**氧化物 CN≤8**,n=154,435 对) | **73.7 / 25.2 / 1.1 %** | George 2020 (ChemEnv) 73.3 / 25.0 / 1.6 | **几乎完全吻合** |
| 同上,限 `ox_source='cif'`(n=112,378) | 73.7 / 25.1 / 1.2 % | 同上 | 对氧化态来源不敏感 |
| 全分析集 CN≤8(n=316,698) | 72.7 / 25.8 / 1.6 % | — | 全阴离子集首次给出 |
| 全分析集 全 CN(n=338,135) | 70.6 / 25.5 / 3.9 % | — | 高配位把共面推高 |
| GII 中位(氧化物 + `ox_source='cif'`) | **0.229**(RMS)/ 0.194(mean\|dev\|) | §6.3 pilot 报 0.168 | **不吻合,见下** |

**GII 对不上 0.168,原因已查清一半,如实记录,不改预期:**

1. 阴离子是主因。逐阴离子中位:O 0.223 / F 0.191 / Cl 0.230 / I 0.262 / S 0.268 /
   Se 0.315 / Br 0.333 / N 0.384 / Te 0.475 / P 0.481。
   **IUCr 键价参数对氧化物之外的体系明显更差**,全分析集 0.246 就是被 Te/P/N/Se 拉上去的。
   §6.3 的 0.168 是**氧化物口径**的数,拿它当全阴离子集的预期本身就是错配。
2. 氧化态来源次之:`cif` 层 GII 中位 0.203,`guess` 层 0.363(400 条冒烟样本上的数)。
3. 剩下的 0.229 vs 0.168 差距**没有查清**。已排除两个假设:
   (a) 不是 §6.5-1 的 falsy bug —— `exp((R0−d)/b) > 0` 恒成立,BVS 实测**没有一个位点等于 0**
       (`n_bvs_bonds==0` 的 35,702 个位点走 NaN 分支,占 7.78%),那个 bug 在本口径下无从触发;
   (b) 不是 RMS/MAD 定义之差 —— 换成 `mean|dev|` 只降到 0.194,仍高于 0.168。
   最可能的残余原因是**样本population 不同**:pilot 的 `exp_oxide` 是"含 O 的全部实验结构"
   (走 sqlite `dataset='experimental'`,含多阴离子体系),本轮是分析集(单一阴离子、无 H 无 C);
   且 pilot 的氧化态走 `cif → bva → guess` 级联,本轮禁用了 bva。**待 P3 全实验库跑完再定论。**

**三算法一致性(G6 的输入)**:位点级,三者都有值的 424,681 个位点上——
ChemEnv==CrystalNN **82.1%**、ChemEnv==Brunner 72.9%、CrystalNN==Brunner 79.5%、
**三者全同 70.5%**。§6.3 说"pilot 已证在 CN≤8 下连接性统计与 ChemEnv 差 <1.5 pt",
那是**聚合统计**的差;**位点级**一致率只有 82%,两者不矛盾但含义完全不同,
G6 要用的是后者,**不要拿 <1.5 pt 当 G6 已经过关的依据**。

**其他**:`I_G` 中位 2.503(symprec 0.01),Wyckoff 轨道数中位 6;
两档 symprec 给出不同空间群的结构占 **1.73%**;
规则 5 的 `max_distinct_ce` 分布 1→23,371 / 2→7,815 / 0→4,264(无 CE) / 3→1,720 / 4→658 / 5→224。

### 已知缺口(下一轮补)

1. **没有 bond 级表(cation–anion)。** 泡林第二定律要的是**阴离子位点**上的 Σs = Σ(z/CN),
   本轮三张表只到阳离子位点与多面体连接对,**算不了 §6.6 那个 18.3% 的交叉验证项**。
   `struct.parquet` 里的 `gii` 是内部用 3.5 Å 邻居即时算的,没有落盘 bond 明细。
2. `pair.parquet` 只有 ChemEnv 路线。§9.1 要的 CrystalNN 版(标记商图配体集合交集)未实现。
3. `splits.parquet` 尚不存在,`proto_id` / `w_debias` / `split_id` 三列未进表,
   PREREG §3.1 要的 `ρ` / `deff` 还没法测。
4. 62 条坏 CIF 未回原始 ICSD CIF 重救(ICSD 那 34 条原则上可救)。

---

## `measure_deff.py` —— PREREG §3.1 的 ρ / deff / N_eff 实测

产物:`features/deff.json`(全部数字)、`features/deff_prereg_block.md`(可直接追加进 PREREG 修订记录)。
跑法:`python src/measure_deff.py --boot 500 --force`;全量 71 s 单核,`--limit N` 冒烟。
sha256[:16] = `fca0821bf32e0413`(**本仓库目前没有 .git,PREREG 与计划 §12 要求的 git tag 无法执行 —— 需要
先 `git init` 再打 tag,在那之前 sha256 是唯一的版本锚点**)。

三个必须论证的选择,详见脚本头部 §A/§B/§C:

- **§A 对哪个变量算 ρ**:对残差指示变量,不对连续特征。deff 只通过 `N_eff·H(残差|S)` 进入 `L_total`。
  用三个压缩靶各一个文献先验残差(众数查表残差 / 泡林 3 违例 / BVS 超窗),不含任何搜索产物,
  满足"看到候选规则之前测完"。实证:连续 `bvs_dev` 的 ICC 在四套原型方案下取 0.001–0.717,不可用。
- **§B 原型缺失**:`structure_type` 在分析集上只命中 **54.77%**(不是 78.13%,那是全 icsd_meta 口径),
  COD 全缺。主口径 hybrid = 命中者用 `structure_type`,缺失者用代理 `spg_s01|匿名约化式`;
  另报 strict / proxy_all / typed_only 三套。**这个选择比 bootstrap 抽样误差更能挪动 N\***(3 → 13)。
- **§C m 的取法**:主报 Kish 加权均值 `Σm²/Σm`(不等簇的正确取法),算术均值并列。
  两者差 4–6 倍(15.7 vs 73.3),计划 §4.5.1 表里的 m=25 是算术口径。

主结果(hybrid,轨道加权,N_data = 138,668):
ρ = 0.204 / 0.608 / 0.204(T_CE / T_CONN / T_BV),deff = 15.7 / 44.3 / 16.7,N_eff = 7,658 / 2,897 / 7,617。
阴性对照 ρ = −0.0005、deff = 1.0(估计量无偏)。

**判定**:触发 §3.1 备用条款,N 报区间不报 argmin(理由三条见 `deff_prereg_block.md`)。
λ=1 主口径登记区间:T_CE / T_BV **N ∈ [5, 11]**、T_CONN **N ∈ [3, 5]**。

已知局限:
1. Zipf 映射是反标定的 —— 计划 §4.5.3 只给了 argmin 结果、没给准确率衰减律,五个锚点在单一
   (theta, p0) 族里无法同时拟合(log-RMSE 0.337)。第二套独立映射(幂律)在同一 N_eff 上给 15 而非 10。
   **N 的区间对模型设定的敏感度大于对抽样误差的敏感度。**
2. `splits.parquet` 仍不存在,ρ 在全分析集上测,未按 discovery/calibration/lockbox 分开。
   §8.1 的双向聚类必须复用本文件的 `proto_hybrid` 定义,两处不许各定各的。
3. T_CONN 只有 ChemEnv 路线(pair 表限制);阴离子位点 Σs 仍缺,泡林 2 的阴离子端残差测不了。

---

## `reproduce_george.py` + `george_table.py` + `pauling_radii.py` —— 复现 George 2020(PREREG 门 G-A)

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python src/reproduce_george.py --stage compute --limit 200 --workers 20 --chunk 5 --force   # 冒烟 0.7 min
nohup python src/reproduce_george.py --stage compute --workers 20 --chunk 20 --force \
      > $PRIS_FEATURES/reproduce_george.log 2>&1 &    # 全量
python src/reproduce_george.py --stage table            # 出 Table S1(读全量产物)
python src/reproduce_george.py --stage table --smoke    # 读 _smoke 产物
```

产物:`george_site / george_anion / george_pair / george_struct / george_fail .parquet`
与 `george_tableS1.csv`;分片在 `_gshards/`(断点续跑靠它,重跑不加 `--force` 即续)。

### 为什么不复用 `site/pair/struct.parquet`(必须重算的三条理由)

1. **论域不对**。George 的论域是氧化物,我们的可比子集是 `provenance.oxide_strict` = 23,728;
   而那三张表建在 `in_analysis_set` 上。**两者不是包含关系**:`oxide_strict \ in_analysis_set`
   = 3,895 条**全部含 P** —— `in_analysis_set` 把 P 记作阴离子候选,磷酸盐于是
   `n_anion_kinds==2` 被排除。George 恰恰用 InPO4 当第二定律的主例,**磷酸盐必须在论域内**。
2. **没有阳离子–阴离子键级表**,阴离子位点的 Σs 算不出来 → 泡林第二定律做不了。
3. **`pair.parquet` 只有 ChemEnv 一条路线**,G6 要的三算法连接性拿不到。

### 实测发现的一个上游 bug:`cif` 氧化态全 0 的结构被静默掏空

`build_features.assign_oxi` 用 `blob_ox_present = all(x is not None)` 判 `cif` 级,
但 ICSD 有一批条目的 `_atom_type_oxidation_number` **全是 0**(例:`exp007` 的 ZnO
装饰成 `Zn0+ O0-`)。这些结构 `ox_source` 被记为 `cif`,而 `is_cat = [v>0]` 全 False,
于是 **`n_cation_sites == 0`,site 行一条不出**;更糟的是 ChemEnv 拿到全 0 的 valences 后
`only_cations=True` 退化,`pair.parquet` 里出现 **O–O 多面体对**(逐例核对 `exp006`
`Mn0+ Au0+ O0+`:旧表 23 对里有 14 对是 O–O)。

- 影响面:`struct.parquet` 里 `n_cation_sites==0` 共 **3,658 / 38,307 = 9.55%**,其中 3,637 条 `ox_source='cif'`。
- 修法(`reproduce_george.assign_oxi_fixed`):`cif` 只在"阴离子 ox<0 **且** 至少一个位点 ox>0"
  时采信,否则降级到 `guess` → `none`,并留 `cif_all_zero` 旗标。
- **下游影响**:`build_features` 报的连接性统计(共角/共边/共面)含这批 O–O 对,需按此重算;
  本脚本的 `george_pair.parquet` 已是干净版。

### 连接性枚举:三算法共用一个枚举器(G6 的必要条件)

`enumerate_connections()` 直接在配体集合上做:两个多面体 (i@0) 与 (j@T) 的共享配体数
= `|L_i ∩ (L_j + T)|`,候选平移 T 由同编号配体的像差给出。去重约定 `i<j` 取全部 T、
`i==j` 时 T≠0 且 T 与 −T 只取字典序较大者 —— 与 ChemEnv `environment_subgraph()` 的
"每原胞一条边"完全一致。**核验**:在冒烟集上与 `pair.parquet` 的 ChemEnv 结果逐结构比,
26 个可比结构里 **24 个三模式计数完全相同**,2 个不同的正是上面那个全 0 氧化态 bug 的受害者。

不用 ChemEnv 自带的 `ConnectivityFinder` 的理由:那样 G6 比较的就不只是"近邻定义"这一个
自由度,而是把连接性算法也一起换了,三算法的差异无法归因。

### 粒度(PREREG §4.3 要求显式声明,George 混了粒度)

| 规则 | granularity | symprec 是否起作用 |
|---|---|---|
| 1 半径比 | `orbit`(阳离子位点按晶体学轨道去重),备 `site` | 是 |
| 2 静电价 | `orbit`(**阴离子**位点),备 `site` | 是 |
| 3 连接类型 | `pair`(George 原口径);60 格表里改用 `orbit-pair` 去重 | 仅 orbit-pair 口径下 |
| 4 相邻多面体 | `structure` | **否**,两列按构造相同 |
| 5 简约 | `structure` | **否**,同上 |
| 2–5 合取 | `structure` | 否 |

所以 60 格里有 **24 格是按构造相同的**,这不是复制粘贴,是口径的真实结论。

### 规则 1 的两个口径(差 8 pt,必须成对报)

`pauling_radii.py` 只收**泡林本人发表过的单价半径**(闭壳层离子,`tier='published'`),
外推到开壳层 d 区离子的那一档标 `tier='extended'`,**是我们的外推不是泡林的表**,只作敏感性层。

判据也有两个口径。George 原文:"A coordination environment is stable only if the radius ratio
falls within the geometrically derived stability window **of this environment**"。硬球稳定窗
只对 CN ∈ {2,3,4,6,8,12} 有定义,观测 CN = 5/7/9/10/11 的位点**没有可检验的窗**;
按原文 "tested local environments" 的措辞应当排除。两个口径都在对比表里报,60 格表用前者
(`R1_radius_ratio`),严格全 CN 版另存 `R1_radius_ratio_allCN`。

### 论域对齐层

`icsd_mp_link.parquet` 的 `mp_id` 非空 = "该 ICSD 条目在 Materials Project 里",
正是 George 的论域筛选条件。这一层用来判断差距是 bug 还是论域,是 go/no-go 的关键判别器。

### 全量实测(2026-07-28,20 进程,机器非独占)

**58.8 min 墙钟**,23,728 条跑完 23,673(失败 55 = 0.23%,全是 blob 里 CIF 本身坏了)。
`site` 376,247 / `anion` 637,679 / `pair` **5,478,931**(三算法合计)/ `struct` 23,728 行。
`ox_source`:cif 15,497 / guess 6,685 / none 1,491;`cif_all_zero` 降级 887 条(3.75%)。

**与 George 2020 逐条**(ChemEnv,symprec 0.01,ox ∈ {cif,guess}):

| 规则 | George | 本复现 | Δ pt |
|---|---|---|---|
| 1 半径比(严格全 CN,published 半径) | 66.0 | 52.5 | −13.5 |
| 1 半径比(**硬球窗有定义的 CN**) | 66.0 | 61.7 | −4.3 |
| 1 半径比(硬球窗 + 外推半径) | 66.0 | **64.8** | −1.2 |
| 2 \|Σs−2\|≤0.01(orbit / site) | 20.0 | 17.9 / **20.2** | −2.1 / +0.2 |
| 3 corner/edge/face(全 CN) | 62.5/27.2/10.3 | 66.5/25.8/7.7 | +4.0/−1.4/−2.6 |
| 3 corner/edge/face(**CN≤8**) | 73.3/25.0/1.6 | **73.4/24.9/1.8** | +0.1/−0.1/+0.2 |
| 4 违例率(ChemEnv / CrystalNN / Brunner) | 40.0 | 34.7 / **40.3** / 38.4 | −5.3/+0.3/−1.6 |
| 5 简约(CN 判据) | 70.3 | 67.8 | −2.5 |
| 2–5 同时 | 13.0 | 11.6 | −1.4 |
| 2–5 同时(CN≤8) | 20.0 | 15.7 | −4.3 |

论域对齐层(ICSD ∩ MP,13,149 条)几乎不动这些数:规则 2 = 17.8、规则 4 = 35.1、
规则 5 = **70.9**、2–5 = 11.9,规则 3 CN≤8 = 72.8/25.3/2.0。**说明差距不是论域造成的。**

**G6(三算法波动,max over 12 格)**:R2 **0.83 pt** ✓ / R5 **1.81 pt** ✓ / R1_allCN 2.89 ✓;
R1 4.05 ✗ / R3 3.93 ✗ / **R4 5.74 ✗**。位点级三算法 CN 全同只有 **82.0%**(n=357,082)。

**规则 4 的两个分解**(ChemEnv):氧化态版违例 56.3%、CN 版违例 52.6%。
更直接的检验(George Fig.4b 的核心断言)是连接倾向:
`P(相连 | min CN)` = 7.9%(CN1)→ 30.4%(6)→ **62.3%(12)** 单调上升;
`P(相连 | min ox)` = 28.5%(+1)→ 33.4%(+2)→ 10.8%(+5)→ **2.8%(+7)**。
**两者都有边际效应,但 ox 那一支被 CN 强烈混杂**(高价阳离子基本都是低配位),
分不开之前不能照抄 George "氧化态不影响连接性"的结论。二维条件版留给 MPU-2。

**规则 2 的畸变依赖**(George Fig.2b 右):max CSM ≤ 5/1/0.1/0.01 → 22.0/32.7/47.6/**57.8**%。
方向与 Baur 假说一致,但**远达不到 George 说的 "nearly perfect"**;
我们的 CSM 阈值是自定的,George 的"221 个材料"口径未公开,这一条不可比,只报趋势。

**规则 1 逐元素**(n≥1500):P 98.2 / Si 95.6 / Ti 84.0 / B 64.2 / Al 58.8 / Li 52.0 /
Mo 32.8 / Na 30.9 / Sr 23.7 / Ca 21.8 / Ge 16.0 / V 13.5 / K 10.9 / Ba 10.3 / Cs 7.3。
与 George Fig.1b 的定性图案一致(四面体形成体高、碱/碱土低)。
泡林已发表半径覆盖 65.7% 的阳离子位点,加外推 87.0%;未覆盖 Top:W/Fe/Mn/U/Cu/Bi/Mo/Pb/V/Co/Nd。

---

## `build_bonds.py` —— MPU-2 缺口补齐(bond / anion_sum / pair 三算法 / proto_id / split)

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=python
$PY src/build_bonds.py --stage compute --limit 60 --workers 12 --chunk 5 --force  # 冒烟 0.7 min
nohup $PY src/build_bonds.py --stage compute --workers 20 --chunk 20 &            # 全量 121.6 min
$PY src/build_bonds.py --stage assemble   # 2 min
$PY src/build_bonds.py --stage report     # ε 曲线 + 规则 3 三算法表(只看 discovery)
```

论域 = `in_analysis_set ∪ oxide_strict` = **42,202**(38,307 + 磷酸盐 3,895)。
后者不在 `splits.parquet` 里,`split = "unsplit"`,**下游禁止混进 discovery 之外的用途**。

### 产物

| 文件 | 行数 | 大小 | 粒度 |
|---|---|---|---|
| `bond.parquet` | **8,948,970** | 52.5 MB | 一条阳-阴键一行(chemenv 2,785,858 / crystalnn 3,028,229 / brunner 3,134,883) |
| `anion_sum.parquet` | **2,690,532** | 29.7 MB | 896,844 个阴离子位点 × 3 算法 |
| `pair.parquet` | **12,012,499** | 43.7 MB | 多面体连接对 × 4 条路线(旧版备份为 `pair_v1_chemenv_only.parquet`) |
| `bond_struct.parquet` | 42,202 | 0.8 MB | 结构级审计(ok 42,134 / fail 68) |

`bond` 列:`source_id / nn_algo / cation_site / anion_site / el_* / ox_* / img_a,b,c / d /
cn_cation(阴离子过滤后) / cnall_cation(未过滤) / s_pauling / s_bv / ce_cation / csm_cation /
proto_id / split`。`s_pauling = z/CN` 是 **T1 量**;`s_bv` 是 **T2 量,按铁律 2 不得进 T4 的 Guard/Body**。

`anion_sum` 列:`sigma_pauling / sigma_bv / z_abs / dev_* / n_cat / d_min / d_max /
ok_{pauling,bv}_e{01,05,1,2,3,5}`(nullable boolean)。
**零配位阴离子(1.99%)记 Σs = 0 而不是 NaN** —— `george_anion` 把它们判 NaN 丢掉了,
那恰好是最严重的违例,不能丢。

### 坑 D:`ConnectivityFinder` 静默吃掉 78.6% 的结构(推翻旧 `pair.parquet`)

`pymatgen.analysis.chemenv.connectivity.ConnectivityFinder.get_structure_connectivity` 有两个
**整结构抛异常**的分支:

1. `multiple_environments_choice=None`(默认)时,只要任一位点被 MultiWeights 判成 "mix"
   (`len(neighbors_sets[i]) > 1`)就 `raise ValueError("... is a mix and nothing is asked about it")`。
   实测 **29,164 / 42,202 = 69.1%** 的结构含至少一个 mix 位点。
2. 循环只挡了 `neighbors_sets is None`,没挡**空 list** → `IndexError: list index out of range`。
   实测 **8,423** 个结构中招。

`build_features.py` 把这两个异常 `except` 进 `ce_err` 列,没有行数报警,于是旧
`pair.parquet` 的 338,135 行**只来自 8,212 / 38,307 = 21.4% 的结构**。
能跑通的都是"环境判定干净"的高对称结构,共角占比系统性偏高 —— 典型的静默选择偏倚。
**旧 `pair.parquet` 的任何规则 3/4 数字都不可用。**
(MPU-1 报的 73.4/24.9/1.8 来自 `george_pair` 的统一枚举器,不受此坑影响,可用。)

`robust_env_subgraph()` 是打过补丁的重写版:mix → 取 `ce_fraction` 最大者(pymatgen 自带但默认
关闭的 `TAKE_HIGHEST_FRACTION`);空 list → 跳过该位点而非整结构放弃。修完命中 **40,625 / 42,202**。
`n_mix_env / n_empty_env / n_ceconn_noncat` 三列逐结构记录,改口径时可复查。

### 四条 pair 路线与"枚举器 vs 近邻定义"的归因

`chemenv` 用 ChemEnv 自带的 `environment_subgraph()`(`len(d["ligands"])`);
`crystalnn` / `brunner` 用共享配体集合的交集;`chemenv_uni` 是**审计线**——同 ChemEnv 近邻,
但走统一枚举器。在 ChemEnv 配体集合纯阴离子的结构上,`chemenv` 与 `chemenv_uni` 的边**逐条完全相同**
(冒烟 51/56 结构,1922/644/233 三档完全一致)。
=> 统一枚举器是 `environment_subgraph()` 的精确重写,**G6 的 3.93 / 5.74 pt 全部来自近邻定义,
枚举器贡献为零**,归因干净。`chemenv` 与 `chemenv_uni` 的行数差只来自阴离子过滤
(ChemEnv 原生把阳-阳共享配体也算进来)。

### proto_id(hybrid 口径)

ICSD `structure_type` 命中者记 `ST:<type>`(覆盖 **53.60%**,COD 为 0%);
缺失者记 `PX:spg<spg_s01>|<anonymized_formula>`。全论域 **9,571 个唯一 proto_id**。
已并进 `site / pair / struct / bond / anion_sum / bond_struct` 六张表,同时并了 `split`。

### 与 `george_anion` 的一致性核验

同一子集(discovery+unsplit, oxide_strict∩O, chemenv)上,两表 Σs 在 **99.81%** 的位点逐条相同;
差异位点的 |Δ| 中位恰为 **1.000**(整整一条键),是 ChemEnv 边界近邻的 tie-break 抖动,非系统偏差。
ε=0.01 满足率:`anion_sum` 18.60% vs `george_anion` 18.57%。

---

# MPU-3 规则搜索(`search_rules.py` / `report_rules.py`)

**只读 discovery。** 全部取数在 `split == "discovery"` 上过滤并 assert;`calibration` /
`lockbox` 在本轮从未被打开。产物:`t4_instances` 口径的 `_t4_orbits_raw.parquet`、
`_tierA1/_tierA2.parquet`、`rule_candidates.parquet`、`rules_surviving.parquet`、
`rules_top.csv`、`search_log.json`(含 `negctl_i` 与 `negctl_iii_rule3`)。

## 论域与标签

主目标 T4 的基本单位是**晶体学轨道**(`orbit_id_s01`),预测量在
`(source_id, element, ox)` **物种**内恒定 —— 这是 T0 输入能表达的最细粒度
(要落到位点就得知道该物种占几个轨道,那是 T1)。
discovery 阳离子位点 259,026 → 三算法(ChemEnv / CrystalNN / BrunnerNN)CN 齐全 239,548
(92.48%,G6 同底要求)→ **72,087 轨道 / 19,430 结构 / 40,097 物种实例**。
三份标签 `cn_chemenv / cn_crystalnn / cn_brunner` 同时保留,G6 直接考核三者的准确率极差。

## 众数查表基线:80% 这个数是"物种未加权平均"的产物

| 口径 | CN(CrystalNN) | ce_symbol |
|---|---|---|
| 实例加权(全阴离子) | **0.564** | 0.519 |
| 实例加权(氧化物) | 0.603 | 0.553 |
| 物种未加权,n≥30(147 个) | 0.648 | 0.603 |
| **物种未加权,全部(954 个)** | **0.765** | 0.746 |

PREREG §2 引的 "≈80%" 只在**最后一行**成立,而那一行被几百个只出现一两次、纯度平凡为 1.0 的
稀有物种拉高。**可用于预测的硬下界是 0.563(ChemEnv)/ 0.580(CrystalNN)**,
本轮全部匹配覆盖率比较都以它为准。泡林第一定律(泡林单价半径 + 五个冻结断点)
在同一论域上 **0.324**(覆盖 89.6%),远低于 George 报的 66% —— 后者是氧化物 + 位点粒度。

## 两层搜索

* **Tier A1** `IF f relop θ THEN cn = c`:每个 (f, relop, θ) 上对 c 取穷举最优。
  前缀和 + `searchsorted`,61 特征 × 2 relop × 20 分位阈值,**0.8 s**,1,713 条。
* **Tier A2**(计划 7.4 节模板族 2)`cn == ARGMIN n IN CN_SET : |φ(n) − (f−a)/b|`。
  φ ∈ {n, 1/n, √n, log n, 泡林理想半径比, ox/n},CN_SET ∈ {FULL(2..12), PAULING, COMMON}。
  **关键实现**:φ 单调 ⇒ ARGMIN 等价于 f 的单调阶梯,切点由 (a,b) 两个自由参数决定,
  于是用"排一次序 + 每类前缀计数"就能对整个 24×40 的 (a,b) 量化网格精确穷举,
  而不必逐实例算 11 个 |·|。`s_pauling` 支(φ 依赖 ox)按 ox 分层后同法处理。**19.9 s**,1,037 条。
* **Tier B** `pysubgroup.Apriori`(穷举 + 反单调剪枝),selector 池 **135 个**(硬上限 150,
  含 20 个随机特征做负对照 ii;覆盖率 >97% 或 <0.5% 的退化 selector 已剔除),
  depth=3,`StandardQF(a)` 扫 a ∈ {0.25, 0.5, 0.75}。**实测 depth3 单次 13.1 s / depth2 1.3 s**
  (72,087 行 × 135 selector),51 个 Body × 3 次扫描共 **1,194 s**。
  与计划 7.4 节引用的 240 s 相比快一个量级,原因是行数少 3 倍且 numba 快路径生效。

## G3 比特记账(计划 4.3 节词汇表口径)

`L = 9.3(头字段) + log₂6(Body 产生式) + [log₂13 | log₂6 + log₂3 + log₂47 + log₂24 + log₂40]
+ |guard|·log₂250`。常数 Body + 3 文字 guard = **39.5 bit**;ARGMIN Body + 3 文字 guard =
**55.4 bit**,4 文字就 63.4 bit 越预算 —— 即 60 bit 在 ARGMIN 族上**恰好**把 guard 卡在 3 文字,
与 §7.1 文法的 `<guard> ≤ 3 文字` 独立地给出同一上限。

## 门与漏斗(2,044 → 164)

见 `report_rules.py`。**G4(阴离子留一 worst-fold τ ≥ 0.90)是唯一真正在杀候选的门**
(单门通过率 15.9%);G6 在 G4 之后完全不 binding(阈值从 3 pt 放宽到 100 pt,存活数 164 → 169)
—— 这与 MPU-1 的直觉相反,原因是能过 G4 的规则本来就落在三算法一致的"干净"区域。
`uses_rnd`:2,044 条里 697 条含随机特征;**过完 G1–G5 的 200 条里仍有 36 条(18%)含随机文字**,
负对照 ii 的显式过滤是有实际作用的,G1–G5 单独拦不住它。

## 负对照

* **(i) 标签块置换**(在 `proto_id` 块内置换,保留混杂结构):Tier A2 最优精确率
  0.4475 → **0.3481**,而众数查表是 0.5628 —— 置换后最优候选连查表都远打不过,零分布有区分力。
  逐规则的 `perm_z` 用 B=200 份块置换标签矩阵一次性算出(2.4 s 生成,每条规则 O(200·n_trig))。
* **(ii) 20 个随机特征**:见上,最终存活集合 0 条含随机文字。
* **(iii) 重新发现泡林第三定律(G-D 门,PREREG §6)**:**通过**。
  在 4 条 pair 路线 × 4 种分层(D≤2,取自 Guard 词汇)共 16 个组合上,
  搜索从 6 个可能的频率序里**每一次都把 `corner > edge > face` 排到第一**。
  D=1(阳离子元素)分层:ρ = 0.926 / 0.924 / 0.921 / 0.886(chemenv / chemenv_uni /
  crystalnn / brunner),Wilson 95% 下界 0.848 / 0.844 / 0.838 / 0.797,远超 §4.4 要求的 0.5。
  D=1(氧化态):ρ = 0.655–0.694,下界 0.596–0.636,仍过。
  **D=2(氧化态 × CN 档)全部不过**(ρ = 0.461–0.493,下界 0.417–0.450)—— 这正是 §4.4
  预言的过度分层失效模式(层数 487–645,中位层样本量掉下去),而不是化学。
  管线的发现能力经此确认,可以继续。

## 本轮的实质结论(负结果,必须如实写进论文)

**没有一条 T0 候选在 MDL 上打得过众数查表。** 164 条存活规则的 `ΔL = N_eff·[H(查表) − H(规则)] − L(R)`
(`N_eff = n_trig / deff`,`deff = 15.7` 取自 PREREG 修订 R1 的 T_CE 压缩靶)**全部为负**,
最好的一条 −25 bit。对查表的匹配覆盖率增益只有 **+0.17 ~ +0.53 pt**。
原因是可诊断的:Apriori 在 `StandardQF` 下找到的高覆盖 guard 恰好圈出**物种同质**的区域
(高离子势 → Si/P/S/B → 本来就 95% 是 CN=4),那里查表已经饱和;
而查表最弱的区域(Cu⁺/Ag⁺ 的 CN=2、大阳离子的 CN=8/12,查表只有 17–19%)确实能被组成级
规则拉高 **+15 ~ +19 pt**,但**这类候选一条都没过完门**:增益 >2 pt 的 80 条里 0 条通过全部八门
(过 G4 的仅 10 条、过 G6 的仅 21 条);增益 >10 pt 的 18 条里过 G4 的 1 条、过 G6 的 0 条。
G4 的失败模式很具体:它们是氧化物专属的,换到硫化物/卤化物就不成立(阴离子留一 worst-fold τ < 0.90)。
这两件事合起来是一条可发表的判断:**T0 组成级信息在"物种身份已知"之上几乎没有增量**,
增量集中在少数几个 (物种 × 阴离子族) 单元里,而那正是 G8 与 G4 要挡的东西。

---

# MPU-3 / Tier D:集合组装与 `L_total(N)` 曲线

脚本 `src/assemble_set.py`(`python assemble_set.py` 主流程 68 s;`python assemble_set.py ungated`
是无门诊断)。产物:`assemble_result.json`(全部曲线与基线)、`Ltotal_curve.csv`(扁平表,
4 靶 × 4 λ × N=0..12)、`rule_decomposition.csv`(19 条去重规则的 ΔL 分解)、`assemble_ungated.json`。
**全程只读 discovery**:取数走 `_t4_orbits_raw.parquet`(由 `search_rules.prep()` 生成,带 assert),
`pair.parquet` / `site.parquet` 两处各再 assert 一次 `split == "discovery"`。

## 候选池与还原

164 条存活规则用 `body_id` + guard 字符串**逐字还原**成逐实例预测,与存档的
`(cov, acc)` 逐条比对,**164/164 完全一致**(脚本内 assert)。
按"有效预测向量"(触发处给预测、未触发给 −1)去重后 **164 → 19** 条 —— 其余 145 条是
同一 guard 配等价 Body 的重复。选择成本 `log2 C(M,N)` 仍按 **M = 164** 计(搜索空间口径)。

## 目标函数与四条曲线

    L_total(S) = λ·[log2 C(M,N) + Σ L(l_i)] + PAR(S) + N_eff·H(残差 | S)

`PAR` 三口径全报:`reg` = PREREG §4.5.1 字面的全局 `(2^N−1)/2·log2 N_eff`;
`full` = 条件于 guard 的逐模式 `(2^|G|−1)/2·log2 n_eff_G`;`obs` = 只对已实现的格计价(最宽松,
作默认)。三档下 `N*` 至多差 1 条。

| 靶 | deff | 压缩对象 | 残差 / cell |
|---|---|---|---|
| `T_CE` | 15.7 | 轨道 CN 标签序列 | 二值误差指示,cell = guard 模式;未覆盖处由**众数查表兜底**(表代价 3,382 bit 是每点都付的常数,故 N=0 点即 `SET-MODE`) |
| `T_CE_MC` | 15.7 | 同上 | **13 类条件熵**(修正口径,见下);未覆盖处退回单一全局众数格 |
| `T_CONN` | 44.3 | 位点是否参与共边/共面 | cell = (触发模式, 各规则预测 CN),两者 T0 可解码;残差不可用作条件 |
| `T_BV` | 16.7 | \|bvs_dev\| > 0.2 vu | 同上 |

论域 72,087 轨道 / 19,430 结构 / 914 个 (元素, 氧化态) 物种;查表 top-1 = 0.5628。
`L(SET-MODE) = 914·log2 13 = 3,382 bit`(PREREG §4.5.5 的估计值是 240 物种 × log2 68 = 1,461,
物种数与字母表都要按实际论域重算)。

## 登记口径的残差码不可用(方法学发现,需要 PREREG 追加修订)

PREREG 登记的残差码是 `N_eff·H_b(错误率)`。**`H_b` 在准确率跨过 50% 时非单调**
(`H_b(0.60) = 0.971 < H_b(0.44) = 0.988`),于是在 `T_CE` 上出现两个必须承认的荒谬:

* 泡林第一定律(top-1 只有 32.4%)的 `L_total` = **7,613 bit**,比众数查表(56.3%)的 **7,928** 还低;
* "定律集能不能取代查表"这个问法在它下面无法评估 —— 换个更差的兜底反而更便宜。

因此 `T_CE_MC` 用 13 类条件熵 `Σ_cell n_eff_c·Ĥ(y|cell)` 重算(对信息量单调),作并列主口径。
**两条曲线都报,不挑。**

## 曲线数值(λ = 1,单位 bit)

| N | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|
| `T_CE` | **7928** | **7760** | 7796 | 7834 | 7879 | 7931 | 7984 | 8093 | 8198 | 8309 |
| `T_CE_MC` | 12831 | 12298 | **12271** | 12303 | 12348 | 12397 | 12446 | 12549 | 12652 | 12761 |
| `T_CONN` | **1276** | 1279 | 1324 | 1369 | 1423 | 1475 | 1528 | 1632 | 1736 | 1859 |
| `T_BV` | **4198** | 4226 | 4270 | 4314 | 4359 | 4411 | 4463 | 4568 | 4672 | 4783 |

四个基线在曲线上的位置:

| | `T_CE` | `T_CE_MC` | `T_CONN` | `T_BV` |
|---|---|---|---|---|
| `SET-MODE` | 7928(= N=0) | 13457 | 4979 | 7711 |
| 零模型(全局众数 / 单格) | — | 12831 | 1276 | 4198 |
| `SET-P5` | 7696 | **11108** | 1356(含 P4 划分) | 4303 |
| `SET-P25` | 8011 | 12914 | 1335(含 P4 划分) | 4282 |
| `SET-H3` | 7994 | 12897 | — | 4264 |
| 泡林第一定律单条 | 7613 | **11024** | — | — |
| `SET-OURS(N*)` | 7760 | 12271 | 1276 | 4198 |

`SET-P25` / `SET-H3` 在 `T_CE` 上**没有独立的 T0 成员**,只付模型代价、零压缩 —— 这正是
计划 §1 那句"T0 曲线上目前只有两个已知点"的定量版本。
`T_CONN` 上泡林第三定律是**无 guard 的全称断言,不划分论域,MDL 压缩恒为 0**;
只有泡林第四定律的高价 guard 才真正划分,而它省下的数据比特(约 25)也补不回自己的 83 bit。

## 判定

* **拐点:没有。** 四条曲线都是"N=0 或 1 处取极小,此后单调上升",没有"先平后升"的膝。
  `argmin N*`(obs / full / reg 三档):`T_CE` 1/1/1,`T_CE_MC` 2/1/2,`T_CONN` 0/0/1,`T_BV` 0/0/0。
  **G-C 不通过**,按 §12.0.5 转向"晶体化学的规律性不是低秩的"。
* **固定判据(λ=1,在数据项上算累计压缩)**:`T_CE` N80=1 / N90=2;`T_CE_MC` 1 / 2;
  `T_CONN` 1 / 2;`T_BV` 2 / 4。两档判据不差 2 倍,但 N 区间落在 **[1, 2]**(T_BV 是 [2,4])。
* **与 PREREG 修订 R1 登记区间的对照**:登记 `T_CE [5,11]` / `T_BV [5,11]` / `T_CONN [3,5]`,
  实测 **全部落在区间外(低于下界)**。如实报告,不调整登记值。
* **λ 敏感性**:`N*` 在 λ ∈ {1,3,10,30} 下 `T_CE` 1→1→0→0、`T_CE_MC` 2→1→1→0、
  `T_CONN`/`T_BV` 恒 0。**移动 ≤ 2 条,满足 §7 的 L0 判据(≤ ±3)** —— 但这是退化性的满足:
  `N*` 被钉在 0–2,λ 无处发挥杠杆。论文里必须这么写,不能只写"通过"。

## 匹配覆盖率下 vs 众数查表(PREREG §2 的第二条判据)

| N | 覆盖 | 集合 acc | 查表 acc(同覆盖) | Δ | McNemar p | 簇 bootstrap 95% CI(结构 / 原型) |
|---|---|---|---|---|---|---|
| 1 | 7.32% | 0.9568 | 0.9522 | +0.46 pt | 0.0022 | [+0.04,+0.86] / **[−0.05,+0.95]** |
| 2 | 8.14% | 0.9598 | 0.9525 | +0.73 pt | 1.3e−6 | [+0.34,+1.13] / [+0.23,+1.22] |
| 3 | 11.3% | 0.8825 | 0.8772 | +0.53 pt | 1.8e−6 | [+0.25,+0.81] / [+0.17,+0.88] |

矛盾率(≥2 条同时触发的实例中 top-1 不一致的比例)全程 ≤ 0.4%。
**N=1 时按结构原型聚类的 CI 含 0;N=2 / N=3 显著,但幅度只有 +0.5 ~ +0.7 pt,
且规则是在同一 discovery 分区上选出来的,是样本内数字。**

## ΔL 分解:增益 97% 来自分层,不是来自预测

`rule_decomposition.csv` 把每条规则的数据项收益拆成
`ΔL_predict = n_eff·[H_b(查表错误率) − H_b(规则错误率)]`(登记的单规则口径)与其余的 `ΔL_stratify`。
最好的一条:`ΔL_data = 210.0 bit`,其中 **`ΔL_predict` 只有 6.7 bit,`ΔL_stratify` 203.3 bit**。
19 条全部是这个形状(predict 1.2–7.9,stratify 171–211)。
换句话说:这些 guard 挣到的比特几乎全部来自"圈出一块查表本来就特别可靠的区域,
于是查表的错误率序列可以分两段编码",而不是来自"规则比查表预测得准"。
**一条 31.5 bit 的规则挣不回自己的 31.5 bit(预测项 6.7 bit)。**

## 管线的门会毙掉泡林第一定律

`rule_candidates.parquet` 里所有泡林形式的半径比规则(覆盖 89.6–99.7%)**全部不过门**:
`G4` 阴离子留一 worst-fold τ = 0.40–0.62(阈值 0.90),`G5` 对查表的增益为负(−13.5 ~ −18.3 pt),
`G6` 三算法极差 2.7–4.4 pt(阈值 3.0)有半数不过。
而在修正后的 `T_CE_MC` 口径下,**泡林第一定律单条(11,024 bit)压缩得比我们搜出来的任何集合
(最好 12,271)和众数查表(13,457)都好** —— 因为多类条件熵奖励的是"信息量"而不是 top-1 命中,
泡林 1 把 89.6% 的论域切成 6 个 CN 预测格,即使 top-1 只有 32.4%,条件熵也降了 2.0 bit/有效实例。
**top-1 门与 MDL 在此处系统性不一致,这是一条独立的方法学结论。**

## 无门诊断:是门太严,还是信息不够?(`assemble_set.py ungated`,770 s)

把 G1–G8 全部关掉,从**全部 2,044 条候选**(去重后 859 个不同预测向量)重跑组装,
`T_CE_MC`、λ=1:

| N | 0 | 1 | 2 | 3 | 4 | 5 | 6 | **7** | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `L_total` | 12831 | 10384 | 10076 | 9969 | 9930 | 9924 | 9923 | **9861** | 9871 | 9887 | 9925 |
| 数据项 | 12754 | 10080 | 9353 | 9055 | 8923 | 8662 | 8545 | 8074 | 8048 | 7934 | 7862 |

**无门时曲线确实有内部极小,`N* = 7`,落在 PREREG 修订 R1 登记的 `T_CE [5,11]` 区间内**;
固定判据同样落在里面(N80 = 5,N90 = 7)。压缩量 2,970 bit,是有门时(560 bit)的 5.3 倍。

**但这条路走不通,三个理由都是硬的**:

1. `N* = 7` 那个集合的 12 条成员里 **9 条 G4 不过**(阴离子留一 worst-fold τ < 0.90)、
   **6 条 G6 不过**(三算法极差 ≥ 3 pt)、**1 条含随机特征 `rnd_Zmod7`**(负对照 ii 直接触发)。
2. 它的 top-1 准确率是 **0.389,比众数查表的 0.563 低 17.4 pt**,直接违反 PREREG §2 的
   匹配覆盖率判据 —— 而 §2 与 R16 要求 MDL 与匹配覆盖率两条**同时**成立。
3. 它的极小是**浅盆**:N=4..12 之间 `L_total` 只在 9,861–9,930 之间波动(0.7%),
   膝不锐利,按任何合理容差都只能报区间。

结论写死:**"T0 组成级信息里确实存在一个 N ≈ 5–7 的低秩结构,但它是关于 CN 分布形状的
(条件熵低),不是关于 CN 众数的(top-1 差),并且它跨阴离子族不成立。"**
这句话同时解释了 G-C 与 G-E 为什么一起失败,也是 §12.0"T0 集合拿不到"预案要引用的证据。

---

# 第二阶段:合理性法则与判据公式(2026-07-29)

靶从"预测配位环境"换成**"判断结构合不合理"**后,新增以下脚本。

## 特征层

| 脚本 | 产物 | 内容 |
|---|---|---|
| `phys_law.py` | `phys_real` / `phys_bad` | Shannon 配位相关半径类:`bl_min`(最短阳-阴键 / 半径和)、`bl_mean`、`bl_cat_max`、`bl_rsd_max`、`sh_pack`,以及电荷拓扑量 `frac_like_bonds` / `min_opp_frac` |
| `elec_feat.py` | `elec_real` / `elec_bad` | Ewald 分解、位点马德隆能(含尺度无关的 `madz_*`)、Hoppe 有效配位数 ECoN、键价失配(**用形式电荷,BVAnalyzer 禁用**) |
| `geom_feat.py` | `geom_real` / `geom_bad` | 多智能体调研筛出的三个几何配位量:`aa_min`(配体-配体接触比,**泡林第一定律的几何内核**)、`phi`(多面体凸包填充率,专抓阳离子换位)、`mef`(Hoppe MEFIR 有效半径失配,有符号) |
| `sym_feat.py` | `sym_real` / `sym_bad` / `sym_lemat` | 对称性:空间群号、对称不等价位点数、Wyckoff 熵 `I_G`。**结论是不予采纳**,见下 |
| `t0_guard.py` | `t0_guard` | 纯组分的离子性度量(电负性差、泡林离子性分数 `fi`)。**只当前提用**,当法则用排除力恒为 0 |
| `robust_blmin.py` | `robust_blmin` | 核心法则的稳健性自查:三套近邻算法极差、Shannon 表命中率、阈值敏感性 |
| `false_positive.py` | `false_positive` | **假阳性检验**:把法则套到 LeMat 的 DFT 弛豫候选上。那些结构几何合理、只是没被合成,法则应当基本放行;若大批被毙,说明测的是数据库指纹不是合理性 |

**两个脚本用同一个 `seed_of()`(crc32)播种扰动**,所以 `phys_bad` 与 `elec_bad`
的同一 `sid` 指向**同一个**扰动结构,可按 sid 安全合并。
此前用 `abs(hash(sid))`,而 Python 字符串 hash 受 PYTHONHASHSEED 随机化、
**跨进程不同** —— 扰动无法重建,分两次算的特征合并会配错。已实测同种子重放 28/28 一致。

## 使用层

| 脚本 | 作用 |
|---|---|
| `apply_rules.py` | **交付入口**:对任意结构文件判定合理性。`--set single/core4/five`,`--verbose` 逐条给出判定与实测值 |

```
python src/apply_rules.py foo.cif bar.cif           # 推荐五条
python src/apply_rules.py --set core4 *.cif         # 只用可信核心四条
python src/apply_rules.py --verbose foo.cif         # 看每条法则的实测值
```

实测(CsTaO₃ 与它的阴阳离子互换版)完整印证了 S5 那条盲区:

| 法则 | 真实结构 | 互换后 |
|---|---|---|
| `bl_min` | 0.8819 ✓ | 0.9647 ✓ **数值反而更好** |
| `bl_mean` | 0.9400 ✓ | 1.0290 ✓ |
| `madz_range` | 19.62 ✓ | 22.13 ✓ |
| `mad_max` | −5.89 ✓ | 0.41 ✓ |
| **同号成键占比** | **0.0000 ✓** | **0.2500 ✗** |

**四条键长/静电法则全部放行,只有第五条抓住它。** 去掉第五条,这个结构会被判为合理。

## 搜索层

| 脚本 | 作用 |
|---|---|
| `rules_final.py` | 法则集束搜索。`assert_clean()` 硬断言无 lockbox;`--guards` 启用「若 G 则 T」;`--strat-floor` 要求每个化学层的满足率下限;`--min-cov` 控制特征可算比例门槛;`--ban` 禁用指定特征 |
| `formula2.py` | 判据公式。同组成组内配对、反对称双写钉死截距、按组 GroupKFold、聚类自助 CI、弃权曲线。`--protocol holdout|cv` |
| `verify_negatives.py` | **负样本的负对照** —— 用马德隆能确认每一类扰动真的该被排除 |

## 三个必须保留的检查

1. **负样本要有负对照**(`verify_negatives.py`)。S2 曾因电荷未跟随元素交换而
   ΔE 恒为 0,四轮特征工程都在追一个不存在的信号;S6 剪切应变因两个探针都判不出
   不合理而被废弃。
2. **切分断言要写进代码**(`assert_clean`)。"我知道有 lockbox"不等于代码知道。
3. **负样本谱系要覆盖不同的失效轴**。留一类扰动检验显示法则**只能约束它见过的破坏方向**:
   没见过整体膨胀,就不会去选键长上界。为此新增 S5 阴阳离子互换 ——
   它几何上与真实结构几乎无法区分(`bl_min` 0.930 vs 0.937),静电上差 +5.8 eV/atom。
   实测所有键长/配位类法则对 S5 排除力仅 0.03–0.21,而一条
   `frac_like_bonds ≤ 0`(不存在同号离子成键)抓住 96%。
4. **判定标准必须在看到数字之前写死。**
   `sym_feat.py` 是一个实例:"结构必须有对称性"这条规则满足率 **99.17%**、
   排除力 **39.5%**,两个数都比核心法则 `bl_min` 好看。但它对均匀膨胀的排除力是
   **0.00**、对随机位移是 **0.98** —— 完全对应"这类扰动动不动分数坐标",
   是负样本生成器的性质;假阳性落差 +8.3 pt;真实结构 P1 占 0.83% 而
   LeMat 占 9.17%,十一倍差距就是数据库指纹本身。
   判定标准写在 `sym_feat.py` 的文件头,在算之前。**若事后再解释,
   那两个数字足以说服任何人把它写进正文。**

5. **`--ban` 要按 `col2` 过滤。** 带前提规则的 `col` 是 `"G:前提列:目标列"`,
   只匹配 `col` 会让禁用形同虚设 —— 几组"消融前后完全相同"的结论曾因此失效。

6. **天花板测量必须配分布外检验**。无约束 GBDT 在 99% 满足率下排除 87%,
   但留一类扰动检验显示塌陷 0.45–0.87 —— 那是在识别扰动签名,不是识别不合理。
   同口径下法则集塌陷仅 0.243。
