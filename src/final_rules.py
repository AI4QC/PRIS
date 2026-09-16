import os
#!/usr/bin/env python3
"""Plausibility law sets: the final search under a three-way comparison.

All three numbers are reported together; none may be omitted:
  satisfaction on real ionic crystals -> the higher the better (against 13% for Pauling 2-5,
                                         21% at CN<=8)
  exclusion on damaged structures     -> the higher the better (does the law have teeth)
  exclusion on DFT candidates         -> **should be low** (those structures are in fact
                                         plausible, so excluding one is a false kill)

The third number is what guards against self-deception: a rule "volume<=36" can exclude a
great deal, but it excludes plausible loose structures along with the rest -- which the
first two numbers alone cannot reveal.
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
    print(f'real {len(real):,} | damaged {len(bad):,} | DFT candidates {len(cand):,} | features {len(cols)}')
    if 'kind' in bad: print(f'damage classes: {bad.kind.value_counts().to_dict()}')
    def G(df):
        z=df.z_cat_max.values if 'z_cat_max' in df else np.full(len(df),np.nan)
        c=df.cn_cat_max.values if 'cn_cat_max' in df else np.full(len(df),np.nan)
        return {'all':np.ones(len(df),bool),'low charge':z<=2.5,
                'mid charge':(z>2.5)&(z<=4.5),'high charge':z>4.5,
                'low CN':c<=4.5,'mid CN':(c>4.5)&(c<=6.5),'high CN':c>6.5}
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
                        desc=f'if [{g}] then {c} {"<=" if side=="hi" else ">="} {th:.4g}',
                        score=rb-rc))          # teeth minus false kills
    r=pd.DataFrame(rows).sort_values('score',ascending=False)
    print(f'\ncandidate laws: {len(r)}. Top 12 (by damage exclusion - candidate false-kill rate):')
    for t in r.head(12).itertuples():
        print(f'  sat={t.sat:.4f} damage excl={t.rej_bad:.4f} false kill={t.rej_cand:.4f}  {t.desc}')
    def M(t,df,GG):
        v=df[t.col].values; ok=(v<=t.th) if t.side=='hi' else (v>=t.th)
        return (~GG[t.g])|(~np.isfinite(v))|ok
    sR=np.ones(len(real),bool); sB=np.ones(len(bad),bool); sC=np.ones(len(cand),bool); ch=[]
    print('\n=== assembling (constraint: satisfaction on real >= 0.97) ===')
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
        print(f'  {len(ch)}. {t.desc[:50]:50s} sat={sR.mean():.4f} damage excl={1-sB.mean():.4f} false kill={1-sC.mean():.4f}')
    print(f'\n=== plausibility law set N={len(ch)} ===')
    print(f'  satisfaction on real ionic crystals = {sR.mean():.4f}   <- Pauling 2-5: 0.13 (CN<=8: 0.21)')
    print(f'  exclusion on damaged structures     = {1-sB.mean():.4f}')
    print(f'  false-kill rate on DFT candidates   = {1-sC.mean():.4f}')
    if 'kind' in bad:
        print('\n  exclusion by damage class:')
        for k,gg in bad.groupby('kind'):
            print(f'    {k}: {1-sB[bad.kind.values==k].mean():.4f}')
    json.dump({'N':len(ch),'sat_real':float(sR.mean()),'rej_bad':float(1-sB.mean()),
      'rej_cand':float(1-sC.mean()),
      'rules':[{'desc':t.desc,'sat':float(t.sat),'rej_bad':float(t.rej_bad),
                'rej_cand':float(t.rej_cand)} for t in ch]},
      open(F+'final_rules.json','w'),ensure_ascii=False,indent=2)
    print(f'\nwrote {F}final_rules.json')
    return 0
if __name__=='__main__': raise SystemExit(main())
