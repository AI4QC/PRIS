# MatterSim 双容量共识与恢复门控设计

日期：2026-08-01

## 1. 目标与非目标

本轮只检验一个有限问题：在不读取 DFT endpoint labels 的 x0 推理阶段，MatterSim
1M/5M 两个 checkpoint 的同组成相对能量共识与分歧，能否比当前 5M x0 基线更安全地
筛掉高能候选；如果质量确有提升，再把 next7 已发现的组级自适应少步策略作为降成本层。

本轮不是把两个高度相关的 checkpoint 称为独立 ensemble，也不声称 MLIP 共识是新的
普适物理定律。成功只意味着找到一个值得拿到新来源验证的 DFT 前筛候选。旧脚本、旧输出、
既有报告、论文、README 和 PREREG 全部保留；只新增 `next8` 文件和独立报告。

## 2. 已知证据与设计理由

next7 表明固定 8 步 FIRE 相对 5M x0 只增加 `0.0685` 个百分点历史筛除率，远低于
预设 `+3` 个百分点门；但组级自适应回放可 100% 复现 S8 决策，并把平均力评估从 9
降至 1.815。这说明少步轨迹适合做条件计算层，不足以单独成为更好的质量准则。

文献也不支持“高 Fmax 等于坏结构”。高力可能代表仍有弛豫下降空间；单一 universal
MLIP 在高能或 OOD 构型上还可能系统性软化。因此本轮只在两个模型都给出高相对能量且
分歧受控时拒绝；高分歧一律 ABSTAIN，送入 DFT。

## 3. 数据与隔离

### 3.1 开发数据

沿用 ELEMENTA 固定三段 development，但在读取任何 endpoint label 前，把
`threshold_calibration` 的 composition groups 按冻结 salt 的 SHA-256 排序并物理拆成两个
互斥子集：

- `search_calibration`：只确定无标签分歧分位点及各候选安全阈值；
- `formula_selection`：在冻结候选目录中选一个公式；
- `threshold_calibration/threshold_fit`：只为已选公式和 M5 重新校准最终阈值；
- `threshold_calibration/development_gate`：只应用最终阈值并计算改善门，不再拟合或选择。

冻结 salt 为 `next8-threshold-fit-gate-v1-20260801`。按 `sha256(salt + "\0" + rk)`
稳定排序后，前一半 composition groups 分到 `threshold_fit`，后一半分到
`development_gate`；两边保留完整 `rk` 组。主轨的有效组必须按当前公式自身计数：该组至少
有一条 `protected & supported & finite(score)` 的行。若 `threshold_fit` 对主轨不足 99 个
这种有效组，主轨阈值只能 keep-all-supported，不能用不受支持组的 `-inf` 凑数，也不能合并
gate 数据补样本；模型失败或非有限的行仍 ABSTAIN，不能因 keep-all 变成 KEEP。

特征进程只可读 x0 species、cell、PBC、coordinates、`sid/rk/material/stage` 和严格输入标志。
它不得读取 endpoint energy、最终 ionic step、suffix、DFT forces/stress 或稳定性标签。
正式 freeze 必须在读取或哈希 endpoint label 之前验证特征 manifest：
`production_protocol_eligible` 必须是精确布尔 `true`，`evidence_role` 必须为
`protocol_feature_generation`，adapter 必须是源码已复核的 `builtin_mattersim`，且其实际实现
source path/SHA 必须与 executed-source hash 闭包一致；测试注入 predictor 的产物不得进入正式
freeze。feature parquet、feature manifest 和 label parquet 的 hash 与解析必须各自来自同一次
不可变 bytes snapshot（或同一固定文件描述符），不能先按路径 hash、随后重新按可变路径解析；
label snapshot 只能在完整 feature 验证、search cutoff 推导和 threshold-role 拆分后创建。发布前
仍须重哈希原路径并与 snapshot hash 相等。

### 3.2 历史 test

ELEMENTA test 已多次暴露，只能在代码、公式、阈值和 checkpoint 全部冻结后做一次历史
反证。即使数值很好，也只能写成 retrospective discovery，不能写成 confirmatory。

### 3.3 外部数据

- Bartel `matgen_baselines` commit
  `770129797a9919955d84f3c3e59cc389e3b04315` 含六类生成/模板方法各 500 个 DFT
  decomposition-energy 标签，但本轮审计已打开 CSV，且 CIF 很可能是 DFT 松弛终态；只可
  作外部 ceiling/falsification。
- OMat24 validation 的 8 个 ASE-LMDB、955,135 个配置尚未打开 DFT payload；它只适合
  能量、力、恢复和不确定性机制审计，不是 hull 稳定性确认。
- 真正外层确认应使用物理隔离的 Alexandria 2025 时间切片，或新生成器随机化 ID 后的
  x0→统一 DFT endpoint 批次。所有候选都必须有标签，不能只计算模型选中的结构。

## 4. 运行时和模型身份

开发特征必须在同一进程、同一 Torch/CUDA 环境中重算 1M 与 5M，不能把新 1M 输出与
旧 runtime 的 5M 输出直接拼接。固定模型为：

```text
MatterSim package = 1.2.3
1M checkpoint SHA-256 = 28b0b0b0f13efefee06b47ea4c9105a26bd3e2c8396da193430da96b3b49a8be
5M checkpoint SHA-256 = e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5
```

manifest 必须记录 Python、Torch、CUDA、GPU、包版本、两个 checkpoint 路径/哈希、输入哈希、
推理计数、wall time、峰值 CUDA 显存和源代码哈希。输出采用 atomic no-replace 发布；目标
目录已存在时 fail closed。

## 5. 冻结候选目录

对每个受支持结构和每个 checkpoint，先在同一 `rk` composition group 内计算：

\[
g_m(i)=e_m(i)-\min_{j\in rk(i)}e_m(j),\qquad m\in\{1M,5M\}.
\]

定义能量分歧 `u_E=|g_1-g_5|`，并只把模型间力分歧而非绝对高力作为 OOD 代理：

\[
u_F=\max\left(|F_{\max,1}-F_{\max,5}|,
               |F_{{\rm rms},1}-F_{{\rm rms},5}|\right).
\]

只允许以下十一个无拟合权重候选。前九个保持原顺序不变；在任何 endpoint 标签打开前，
基于无标签数学审计追加两个先集成后归零的候选：

1. `M5 = g5`；
2. `M1 = g1`；
3. `MIN = min(g1,g5)`；
4. `MEAN = (g1+g5)/2`；
5. `MAX = max(g1,g5)`；
6. `LCB = max(0, (g1+g5)/2-u_E)`；
7. `AGREE99 = MEAN`，但 `u_E` 高于 search-calibration 的无标签 q99 时 ABSTAIN；
8. `AGREE995 = MEAN`，但 `u_E` 高于无标签 q99.5 时 ABSTAIN；
9. `AGREE_EF995 = MEAN`，但 `u_E` 高于无标签 q99.5 或 `u_F` 高于无标签
   q99.5 时 ABSTAIN；
10. `CMEAN = c`，其中
    \(h_i=0.5g_1(i)+0.5g_5(i)\)，
    \(c_i=h_i-\min_{j\in rk(i)}h_j\)。这等价于先平均两模型绝对每原子能量、再在
    同一 `rk` 内归零，但不等价于保留两模型 argmin 分歧所产生组公共偏移的 `MEAN`；
11. `CMEAN_JOINT99 = CMEAN`，但下述无标签联合分歧分数 `J` 高于 search-calibration
    的 q99 时 ABSTAIN。

`CMEAN_JOINT99` 只用 search-calibration 中 joint-complete 的行，分别对
`d_E=|g1-g5|`、`d_Fmax=|Fmax1-Fmax5|`、`d_Frms=|Frms1-Frms5|` 建立 row-weighted
右连续经验 CDF：

\[
H_k(x)=n^{-1}\sum_{\ell=1}^{n}\mathbf 1[d_k(\ell)\le x],\qquad
J=\max(H_E,H_{F\max},H_{F\mathrm{rms}}).
\]

冻结 `q_J=quantile(J,0.99,method=higher)`，只在 `J>q_J` 时 ABSTAIN，等号 KEEP。
必须序列化 `n`、joint-complete `n_rk`、`weighting=row`、`side=right`、quantile
方法、`q_J`、三份有序参考分布及其哈希；不能只保存 `q_J` 后猜测原量阈值。若 search
中有 `n` 个 eligible 行，该 gate 的额外经验弃权数至多
`floor(0.01*(n-1))`，比例严格小于 1%；这不含不完整组弃权，也不对后续 stage 作同样保证。

候选目录冻结后不得增加连续权重、按元素阈值、suffix、material identity 或事后模型。两模型
共享训练体系，`u_E/u_F` 只是容量分歧代理，不是校准过的 epistemic uncertainty。
`AGREE_EF995` 的两个 cutoff 都使用 `method=higher`；两个 0.5% 尾部的并集在
search-calibration 经验样本上的新增弃权上限约为 1%。

## 6. 选择、阈值与成功门

主轨保持：`valuable <= 50 meV/atom`、`within_group=max`、`alpha=0.01`。历史比较轨保持：
`near_min <= 1 meV/atom`、`within_group=min`、`alpha=0.035`。未知、任一必要模型失败或
不完整 composition group 都 fail-open 为 ABSTAIN：`M5` 只依赖完整的 5M 组，`M1` 只
依赖完整的 1M 组，其余公式才要求两模型都完整，不能让 M1 失败拖累 M5 基线。
`AGREE99/995/EF995/CMEAN_JOINT99` 只让分歧越界的目标行 ABSTAIN；完整组的 gap 与
`CMEAN` 重归零仍由全部有限模型输出计算，不能因先删除高分歧行而改变组内最小值。

每个候选在 search-calibration 校准阈值，在 formula-selection 必须先满足：exact-min
retention 95% 下界 `>=0.95`、主轨 valuable-group（比较轨 near-min）retention 95% 下界
`>=0.95`、`regret_p95<=0.05 eV/atom` 且 `all_rejected_groups=0`，再按：

```text
dft_savings descending
measured/evaluation cost ascending
formula complexity ascending
catalog order ascending
```

确定性 cost units 固定为 `M1=1`、`M5=5`、双模型 `=6`；complexity units 固定为单模型
`=1`、`MIN/MEAN/MAX=2`、`LCB/单分歧门/CMEAN=3`、energy+force 双 cutoff 门 `=4`、
三 ECDF 联合门 `CMEAN_JOINT99=5`，不以本次有噪声 wall time 做 tie-break。主轨选择唯一
生产公式；comparator 只对同一公式复核，不再独立选择第二个赢家。

选择后，`threshold_fit` 只重新校准已选公式和 M5 基线；`development_gate` 只应用这两个
最终规则。相对 M5 的改善门只在 development_gate 上计算，并必须同时满足：

- DFT savings 绝对增加至少 `0.03`，且 composition-paired 95% CI 下界大于 0；
- valuable-item recall 差的 95% CI 下界不低于 `-0.005`；
- exact/near-min retention 继续通过各自下界；
- abstention-rate 增量不超过 `0.01`；
- 不允许全拒绝 composition group。

若 development 未过 `+3` 个百分点门，停止质量路线：不打开 OMat24 payload，不增加第三模型，
只报告负结果。若 development 通过，才冻结 evaluator 并运行一次历史 test；只有新来源外层测试
同时达到 savings 下界 `>=30%`、stable recall 下界 `>=99%`、group-min retention 下界
`>=95%` 且超过 Pauling/BVS/Ewald/几何/单模型 MLIP 基线，才可请求修改正式论文。

## 7. 自适应恢复层

自适应少步不参加本轮质量公式选择。只有 committee 质量门通过后才启用：整个 `rk` 组同步
推进；在 k=0/2/4，若任一结构距任一冻结阈值处于 `1.0/0.75/0.5 meV/atom` 边界带，或
当前 `Fmax>0.15 eV/Angstrom`，才推进到下一 snapshot，否则停止。该阈值来自已暴露历史回放，
必须在新批次前预注册。成本结论以实际 eval count 和 wall time分别报告，不能把计数节省当作
GPU 时间节省。

## 8. 产物与停止条件

新增根目录为 `outputs/20260801_mattersim_committee/`，至少包含 development features、
development freeze、可选 historical test、诊断与全哈希 manifest。独立报告写到
`reports/2026-08-01-mattersim-committee-followup.md`。

任何以下情况立即 fail closed：stage 越界、输入/模型/代码哈希不一致、重复 `sid`、不完整组、
非有限能量、目标输出存在、标签先于冻结读取、test 结果触发新增候选或阈值扫描。
