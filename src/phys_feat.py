import os
#!/usr/bin/env python3
"""物理描述符:马德隆能 + 有效配位数 + 堆积分数 + 次近邻。

为什么加这些:现有 32 个描述符全是拓扑/计数量,GBDT 天花板 0.7143、线性 0.6673。
差距不在模型形式(PySR 非线性也只有 0.6619),在**信息量本身**。

马德隆能(Ewald 求和)是关键:泡林五条本质是静电论证的近似 ——
第二条(键强和=电荷)是局部电中性,第三、四条(共边共面不稳)是阳离子间库仑排斥。
Ewald 是这些论证的**精确版**,只用结构+形式电荷,不碰 DFT,完全在经验判据范畴内,
而且正是泡林自己用的那套物理。
"""
import sys, argparse, warnings, collections
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from polymorph_rank2 import scan, balance
F=os.environ.get("PRIS_FEATURES", "features/")
# Shannon 半径粗表(配位无关的代表值),用于堆积分数
R={'Li':0.76,'Na':1.02,'K':1.38,'Rb':1.52,'Cs':1.67,'Be':0.45,'Mg':0.72,'Ca':1.00,'Sr':1.18,
 'Ba':1.35,'Al':0.535,'Ga':0.62,'In':0.80,'Sc':0.745,'Y':0.90,'La':1.032,'Ti':0.605,'Zr':0.72,
 'Hf':0.71,'V':0.54,'Nb':0.64,'Ta':0.64,'Cr':0.615,'Mo':0.59,'W':0.60,'Mn':0.83,'Fe':0.645,
 'Co':0.745,'Ni':0.69,'Cu':0.73,'Zn':0.74,'Cd':0.95,'Ag':1.15,'Si':0.40,'Ge':0.53,'Sn':0.69,
 'Pb':1.19,'B':0.27,'P':0.38,'As':0.46,'Sb':0.60,'Bi':1.03,'O':1.40,'S':1.84,'Se':1.98,'Te':2.21,
 'N':1.46,'F':1.33,'Cl':1.81,'Br':1.96,'I':2.20,'Th':0.94,'U':0.73,'Ce':1.01,'Eu':1.17,'Yb':0.868}

def one(rec):
    from pymatgen.core import Structure, Lattice
    from pymatgen.analysis.ewald import EwaldSummation
    from pymatgen.analysis.local_env import CrystalNN
    try:
        st=Structure(Lattice(rec['lattice']),rec['species'],rec['coords'],coords_are_cartesian=True)
        val=[float(rec['vmap'][s.specie.symbol]) for s in st]
        sd=st.copy(); sd.add_oxidation_state_by_site(val)
        ew=EwaldSummation(sd,compute_forces=False)
        n=len(st)
        out={'sid':rec['sid'],'rk':rec['rk'],'e_per_atom':rec['e_per_atom'],
             'ewald_per_atom':float(ew.total_energy)/n,
             'ewald_real':float(ew.real_space_energy)/n,
             'ewald_recip':float(ew.reciprocal_space_energy)/n,
             'ewald_point':float(ew.point_energy)/n}
        # 位点级马德隆势的分布。这个版本的 EwaldSummation 没有 site_energies 属性,
        # 用 get_site_energy(i) 逐位点取;取不到就跳过这三列。
        try:
            sm=np.array([ew.get_site_energy(i) for i in range(n)])
            out['mad_std']=float(np.std(sm)); out['mad_max']=float(np.max(sm)); out['mad_min']=float(np.min(sm))
        except Exception:
            pass
        # 堆积分数(离子球体积/晶胞体积)
        vol=sum(4/3*np.pi*R.get(s.specie.symbol,1.0)**3 for s in st)
        out['pack_frac']=float(vol/st.volume)
        # 有效配位数(连续版):ECoN,基于键长的加权
        cnn=CrystalNN(weighted_cn=False,x_diff_weight=0.0)
        econ=[]; d2=[]
        for i in range(n):
            try: nbs=cnn.get_nn_info(st,i)
            except Exception: continue
            if not nbs: continue
            ds=np.array([st[i].distance(st[nb['site_index']],jimage=nb.get('image')) for nb in nbs])
            dmin=ds.min(); w=np.exp(1-(ds/dmin)**6)
            econ.append(float(w.sum())); d2.append(float(np.std(ds)/np.mean(ds)))
        if econ:
            out['econ_mean']=float(np.mean(econ)); out['econ_std']=float(np.std(econ))
            out['econ_max']=float(np.max(econ)); out['dist_rsd']=float(np.mean(d2))
        return out
    except Exception:
        return None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--groups',type=int,default=12000)
    ap.add_argument('--workers',type=int,default=18); a=ap.parse_args()
    g=scan(a.groups); recs=[r for v in g.values() for r in v]
    print(f'{len(g):,} 组 / {len(recs):,} 端点',flush=True)
    from concurrent.futures import ProcessPoolExecutor
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i,r in enumerate(ex.map(one,recs,chunksize=8)):
            if r: rows.append(r)
            if (i+1)%4000==0: print(f'  {i+1:,}/{len(recs):,} -> {len(rows):,}',flush=True)
    d=pd.DataFrame(rows); d.to_parquet(F+'phys_feat.parquet',index=False)
    print(f'写出 {len(d):,} 行 / {d.rk.nunique():,} 组')
    return 0
if __name__=='__main__': raise SystemExit(main())
