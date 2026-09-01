import os
#!/usr/bin/env python3
"""造真正的负类:对真实结构做破坏性扰动。

# 为什么要重造负类

前两版用 LeMat/ELEMENTA 候选当负类,排除率只到 28%。原因:那些是 **DFT 局域极小**,
几何上完全合理,只是能量稍高。用它们当"不合理"的样本,法则只能学到"体积大"这类
表面差异,学不到"什么叫结构上讲不通"。

真正该被排除的是几何上就荒谬的结构。所以对真实结构做四类破坏,每类都对应
泡林规则关心的一种失效模式:

  S1 晶格各向异性压缩 —— 破坏堆积,制造过短/过长的键
  S2 阳离子换位       —— 破坏配位环境与电荷分布(对应泡林第五条)
  S3 随机位移         —— 破坏局部对称与键长均一性(对应泡林第二条)
  S4 晶格整体膨胀     —— 制造配位不足的疏松结构

这些结构**在化学上不存在**,但组成、化学计量、原子数与母体完全相同 ——
所以法则若能把它们排除,排除的就是几何合理性本身,不是组成或尺寸的表面差异。
"""
import sys, argparse, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discriminate import criteria, guess_oxi, read_blob_cif
F=os.environ.get("PRIS_FEATURES", "features/")

def swapped_val(s, val):
    """取出扰动后应当使用的价态数组。S2 换位时电荷要跟着元素走。"""
    sw=getattr(s,'_swapped_val',None)
    if sw is None: return val
    v=list(val); i,j=sw; v[i],v[j]=v[j],v[i]; return v


def perturb(st, kind, rng, val=None):
    from pymatgen.core import Structure
    s=st.copy()
    if kind=='S1':                      # 各向异性压缩 15-30%
        f=np.eye(3); ax=rng.integers(0,3); f[ax,ax]=1-rng.uniform(0.15,0.30)
        s.lattice=type(s.lattice)(np.dot(f,s.lattice.matrix))
    elif kind=='S2':                    # 阳离子换位
        # 关键修正:必须交换**电荷不同**的两个阳离子。
        # 原实现只要求元素不同,但若两者电荷相同(如都是 +2),
        # 马德隆能严格不变 —— 那不是另一个结构,是同一结构的重新标号。
        # 实测原实现的 126 个 S2 样本静电能惩罚**全部精确为 0**,
        # 所以四轮特征工程都排不掉它:它本来就是合理的。
        if val is None: return None
        cats=[i for i in range(len(s)) if val[i]>0]
        pairs=[(i,j) for i in cats for j in cats
               if i<j and abs(val[i]-val[j])>=1.0 and s[i].specie.symbol!=s[j].specie.symbol]
        if not pairs: return None
        i,j=pairs[rng.integers(0,len(pairs))]
        si,sj=s[i].specie,s[j].specie
        s.replace(int(i),sj); s.replace(int(j),si)
        # 电荷必须跟着元素一起换。此前的实现只换元素,而下游用
        # add_oxidation_state_by_site(val) / criteria(p,val) 按**位点索引**
        # 把原电荷贴回去 —— 于是换位对一切依赖电荷的量都不可见,
        # 马德隆能 ΔE 严格为 0。这是 S2 四轮都排不掉的真正原因。
        s._swapped_val=(int(i),int(j))
    elif kind=='S3':                    # 随机位移 0.3-0.8 A
        amp=rng.uniform(0.3,0.8)
        for i in range(len(s)):
            s.translate_sites(i, rng.normal(0,amp,3), frac_coords=False)
    elif kind=='S4':                    # 整体膨胀 20-40%
        f=1+rng.uniform(0.20,0.40)
        s.lattice=type(s.lattice)(s.lattice.matrix*f)
    elif kind=='S5':                    # 阴阳离子互换
        # 留一类扰动检验发现:法则**只能约束它见过的破坏方向**(见结果总结 5.1c)。
        # S1-S4 全是几何破坏,没有一类直接考"电荷放错位点"。
        # 互换一个阳离子与一个阴离子:组成、化学计量完全不变,
        # 但正负电荷的空间排布被彻底打乱,静电上应当极不利。
        if val is None: return None
        cats=[i for i in range(len(s)) if val[i]>0]
        ans =[i for i in range(len(s)) if val[i]<0]
        if not cats or not ans: return None
        i=cats[rng.integers(0,len(cats))]; j=ans[rng.integers(0,len(ans))]
        si,sj=s[i].specie,s[j].specie
        s.replace(int(i),sj); s.replace(int(j),si)
        s._swapped_val=(int(i),int(j))   # 电荷跟着元素走,同 S2
    # 曾试过 S6 剪切应变(10-25%),**已废弃**:两个独立探针都判不出它明显不合理 ——
    # 马德隆能中位 ΔE = -0.004 eV/atom(剪切保体积,静电对它基本是盲的),
    # bl_min 中位 0.881 对真实结构 0.937,只有 10% 跌破 0.804。
    # 硬塞进负样本就是重蹈 S2 的覆辙:花力气去排除一批本来就未必该排除的样本。
    return s

def one(r):
    from pymatgen.core import Structure
    out=[]
    try:
        st=Structure.from_str(read_blob_cif(r['off'],r['ln']),fmt='cif')
        if len(st)>80: return out
        val,ok=guess_oxi(st)
        if not ok: return out
        rng=np.random.default_rng(abs(hash(r['sid']))%(2**31))
        for kind in ['S1','S2','S3','S4']:
            p=perturb(st,kind,rng,val)
            if p is None: continue
            try:
                c=criteria(p,swapped_val(p,val))
            except Exception:
                continue
            if c is None: continue
            c.update(sid=r['sid']+'_'+kind,kind=kind,parent=r['sid'])
            out.append(c)
    except Exception: pass
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=6000)
    ap.add_argument('--workers',type=int,default=18); a=ap.parse_args()
    prov=pd.read_parquet(F+'provenance.parquet',
        columns=['source_id','in_analysis_set','blob_offset','blob_length','n_elements'])
    d=prov[prov.in_analysis_set & (prov.n_elements>=2)].sample(n=min(a.n,len(prov)),random_state=0)
    recs=[{'sid':t.source_id,'off':int(t.blob_offset),'ln':int(t.blob_length)} for t in d.itertuples()]
    print(f'母体 {len(recs):,} 条,每条造 4 类扰动',flush=True)
    from concurrent.futures import ProcessPoolExecutor
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i,rr in enumerate(ex.map(one,recs,chunksize=8)):
            rows.extend(rr)
            if (i+1)%1500==0: print(f'  {i+1:,}/{len(recs):,} -> {len(rows):,}',flush=True)
    o=pd.DataFrame(rows); o.to_parquet(F+'negatives.parquet',index=False)
    print(f'写出 {len(o):,} 条负样本')
    if len(o): print(o.kind.value_counts().to_dict())
    return 0
if __name__=='__main__': raise SystemExit(main())
