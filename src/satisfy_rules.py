import os
#!/usr/bin/env python3
"""满足性法则搜索:N 条真实离子晶体几乎全部满足、同时对未实现候选有排除力的法则。

# 与前面所有工作的区别

前面做的是**排序**(同组成两个结构哪个更稳),这里做的是**满足性**——
George 2020 的口径:泡林 2-5 条只有 13% 的结构同时满足(CN<=8 时 21%)。
目标是把这个数推到接近 100%,同时保住排除力。

# 判据必须是两个数,缺一不可

  满足率 = 真实离子晶体中满足该法则的比例        -> 要 >=99%
  排除率 = 未实现候选中**不**满足的比例          -> 越高越好

只报满足率会让恒真规则得满分(满足率 100%、排除率 0),那毫无内容。
泡林五条的毛病不是太松是太紧;走到另一个极端同样没用。

# 法则形态

单侧区间约束:feature <= hi 或 feature >= lo,阈值取真实结构的 (1-alpha) 分位。
双侧:lo <= feature <= hi。全部只用结构量,阈值有明确物理含义。
"""
import sys, json, itertools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
F=os.environ.get("PRIS_FEATURES", "features/")

def load():
    real=pd.read_parquet(F+'real_all.parquet')
    negs=[]
    for f,tag in [('lemat_rank.parquet','lemat'),('polymorph_rank2.parquet','elementa')]:
        try:
            n=pd.read_parquet(F+f); n['_src']=tag; negs.append(n)
        except Exception: pass
    neg=pd.concat(negs,ignore_index=True) if negs else None
    return real,neg

def main():
    real,neg=load()
    cols=[c for c in real.columns if real[c].dtype.kind=='f']
    cols=[c for c in cols if c in neg.columns and real[c].notna().mean()>0.9 and neg[c].notna().mean()>0.9]
    print(f'真实 {len(real):,} | 候选 {len(neg):,} | 共同特征 {len(cols)}')
    print(f'真实侧阴离子: {dict(real.anion.value_counts().head(8))}' if 'anion' in real else '')
    rows=[]
    for c in cols:
        rv=real[c].dropna().values; nv=neg[c].dropna().values
        if len(rv)<1000 or len(nv)<1000: continue
        for alpha,side in itertools.product([0.005,0.01,0.02],['hi','lo','both']):
            if side=='hi':
                th=np.quantile(rv,1-alpha); sat=(rv<=th).mean(); rej=(nv>th).mean()
                desc=f'{c} <= {th:.4g}'
            elif side=='lo':
                th=np.quantile(rv,alpha); sat=(rv>=th).mean(); rej=(nv<th).mean()
                desc=f'{c} >= {th:.4g}'
            else:
                l,h=np.quantile(rv,[alpha/2,1-alpha/2]); sat=((rv>=l)&(rv<=h)).mean()
                rej=((nv<l)|(nv>h)).mean(); desc=f'{l:.4g} <= {c} <= {h:.4g}'
            if sat<0.98: continue
            rows.append(dict(col=c,side=side,alpha=alpha,desc=desc,sat=sat,rej=rej))
    r=pd.DataFrame(rows).sort_values('rej',ascending=False)
    print(f'\n=== 满足率>=98% 的候选法则:{len(r)} 条,按排除率排序 Top 15 ===')
    for t in r.head(15).itertuples():
        print(f'  满足={t.sat:.4f} 排除={t.rej:.4f}  {t.desc}')
    # 贪心组集合:最大化联合排除率,同时保持联合满足率
    print(f'\n=== 组装法则集(每步选联合排除率增量最大的)===')
    R=real[cols].values; N=neg[cols].values; ci={c:i for i,c in enumerate(cols)}
    def mask(t,M):
        v=M[:,ci[t.col]]
        if t.side=='hi': return v<=float(t.desc.split('<=')[-1])
        if t.side=='lo': return v>=float(t.desc.split('>=')[-1])
        p=t.desc.split('<=');l=float(p[0]);h=float(p[2]); return (v>=l)&(v<=h)
    satR=np.ones(len(R),bool); satN=np.ones(len(N),bool); chosen=[]
    pool=[t for t in r.itertuples() if t.rej>0.02]
    for step in range(10):
        best=None
        for t in pool:
            if any(t.Index==c.Index for c in chosen): continue
            mR=mask(t,R); mN=mask(t,N)
            nsR=satR&np.nan_to_num(mR,nan=True); nsN=satN&np.nan_to_num(mN,nan=True)
            if nsR.mean()<0.95: continue
            gain=satN.mean()-nsN.mean()
            if best is None or gain>best[0]: best=(gain,t,nsR,nsN)
        if best is None or best[0]<0.005: break
        _,t,nsR,nsN=best; chosen.append(t); satR,satN=nsR,nsN
        print(f'  {len(chosen)}. {t.desc:44s} 联合满足={satR.mean():.4f} 联合排除={1-satN.mean():.4f}')
    print(f'\n=== 法则集 N={len(chosen)} ===')
    print(f'  真实离子晶体满足率 = {satR.mean():.4f}   (泡林 2-5 条: 0.13,CN<=8 时 0.21)')
    print(f'  未实现候选排除率   = {1-satN.mean():.4f}')
    json.dump({'N':len(chosen),'sat':float(satR.mean()),'rej':float(1-satN.mean()),
      'rules':[{'desc':t.desc,'sat':float(t.sat),'rej':float(t.rej)} for t in chosen]},
      open(F+'satisfy_rules.json','w'),ensure_ascii=False,indent=2)
    print(f'\n写出 {F}satisfy_rules.json')
    return 0
if __name__=='__main__': raise SystemExit(main())
