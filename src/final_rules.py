import os
#!/usr/bin/env python3
"""合理性法则集:三分对照下的最终搜索。

三个数一起报,缺一不可:
  真实离子晶体满足率  -> 越高越好(对照泡林 2-5 条的 13%,CN<=8 时 21%)
  破坏结构排除率      -> 越高越好(法则有没有牙齿)
  DFT 候选排除率      -> **应该低**(那些结构其实合理,排掉就是误杀)

第三个数是防自欺的关键:一条"体积<=36"的规则能排掉大量东西,
但它把合理的疏松结构也一起排了 —— 只看前两个数发现不了。
"""
import sys, json, itertools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
F=os.environ.get("PRIS_FEATURES", "features/")

def main():
    real=pd.read_parquet(F+'real_all.parquet')
    bad=pd.read_parquet(F+'negatives.parquet')
    cand=[]
    for f in ['lemat_rank.parquet','polymorph_rank2.parquet']:
        try: cand.append(pd.read_parquet(F+f))
        except Exception: pass
    cand=pd.concat(cand,ignore_index=True)
    cols=[c for c in real.columns if real[c].dtype.kind=='f'
          and c in bad.columns and c in cand.columns
          and real[c].notna().mean()>0.9 and bad[c].notna().mean()>0.85]
    print(f'真实 {len(real):,} | 破坏 {len(bad):,} | DFT候选 {len(cand):,} | 特征 {len(cols)}')
    if 'kind' in bad: print(f'破坏类型: {bad.kind.value_counts().to_dict()}')
    def G(df):
        z=df.z_cat_max.values if 'z_cat_max' in df else np.full(len(df),np.nan)
        c=df.cn_cat_max.values if 'cn_cat_max' in df else np.full(len(df),np.nan)
        return {'全域':np.ones(len(df),bool),'低价':z<=2.5,'中价':(z>2.5)&(z<=4.5),'高价':z>4.5,
                '低配位':c<=4.5,'中配位':(c>4.5)&(c<=6.5),'高配位':c>6.5}
    GR,GB,GC=G(real),G(bad),G(cand)
    rows=[]
    for g in GR:
        if GR[g].sum()<1500 or GB[g].sum()<500: continue
        for c in cols:
            rv=real[c].values[GR[g]]; rv=rv[np.isfinite(rv)]
            if len(rv)<800: continue
            for a in [0.005,0.01,0.02]:
                for side in ['hi','lo']:
                    th=np.quantile(rv,1-a if side=='hi' else a)
                    sat=(rv<=th).mean() if side=='hi' else (rv>=th).mean()
                    def rej(df,GG):
                        v=df[c].values; m=GG[g]&np.isfinite(v)
                        viol=(v>th) if side=='hi' else (v<th)
                        return (viol&m).sum()/len(df)
                    rb,rc=rej(bad,GB),rej(cand,GC)
                    if sat<0.98 or rb<0.01: continue
                    rows.append(dict(g=g,col=c,side=side,th=float(th),sat=sat,rej_bad=rb,rej_cand=rc,
                        desc=f'若[{g}] 则 {c} {"<=" if side=="hi" else ">="} {th:.4g}',
                        score=rb-rc))          # 牙齿减误杀
    r=pd.DataFrame(rows).sort_values('score',ascending=False)
    print(f'\n候选法则 {len(r)} 条。Top 12(按 破坏排除率 − 候选误杀率):')
    for t in r.head(12).itertuples():
        print(f'  满足={t.sat:.4f} 破坏排除={t.rej_bad:.4f} 误杀={t.rej_cand:.4f}  {t.desc}')
    def M(t,df,GG):
        v=df[t.col].values; ok=(v<=t.th) if t.side=='hi' else (v>=t.th)
        return (~GG[t.g])|(~np.isfinite(v))|ok
    sR=np.ones(len(real),bool); sB=np.ones(len(bad),bool); sC=np.ones(len(cand),bool); ch=[]
    print('\n=== 组装(约束:真实满足率 >= 0.97)===')
    for _ in range(15):
        best=None
        for t in r.itertuples():
            if any(t.Index==x.Index for x in ch): continue
            nR=sR&M(t,real,GR)
            if nR.mean()<0.97: continue
            nB=sB&M(t,bad,GB); nC=sC&M(t,cand,GC)
            gain=(sB.mean()-nB.mean())-(sC.mean()-nC.mean())
            if best is None or gain>best[0]: best=(gain,t,nR,nB,nC)
        if best is None or best[0]<0.003: break
        _,t,nR,nB,nC=best; ch.append(t); sR,sB,sC=nR,nB,nC
        print(f'  {len(ch)}. {t.desc[:50]:50s} 满足={sR.mean():.4f} 排除破坏={1-sB.mean():.4f} 误杀={1-sC.mean():.4f}')
    print(f'\n=== 合理性法则集 N={len(ch)} ===')
    print(f'  真实离子晶体满足率 = {sR.mean():.4f}   ← 泡林2-5条: 0.13(CN<=8: 0.21)')
    print(f'  破坏结构排除率     = {1-sB.mean():.4f}')
    print(f'  DFT候选误杀率      = {1-sC.mean():.4f}')
    if 'kind' in bad:
        print('\n  分破坏类型的排除率:')
        for k,gg in bad.groupby('kind'):
            print(f'    {k}: {1-sB[bad.kind.values==k].mean():.4f}')
    json.dump({'N':len(ch),'sat_real':float(sR.mean()),'rej_bad':float(1-sB.mean()),
      'rej_cand':float(1-sC.mean()),
      'rules':[{'desc':t.desc,'sat':float(t.sat),'rej_bad':float(t.rej_bad),
                'rej_cand':float(t.rej_cand)} for t in ch]},
      open(F+'final_rules.json','w'),ensure_ascii=False,indent=2)
    print(f'\n写出 {F}final_rules.json')
    return 0
if __name__=='__main__': raise SystemExit(main())
