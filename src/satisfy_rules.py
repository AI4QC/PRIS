import os
#!/usr/bin/env python3
"""Satisfiability law search: N laws that nearly every real ionic crystal satisfies while
still excluding unrealised candidates.

# How this differs from everything before it

Everything before this did **ranking** (of two same-composition structures, which is more
stable); this does **satisfiability** -- George 2020's convention, under which only 13% of
structures satisfy Pauling 2-5 jointly (21% at CN<=8). The goal is to push that number
towards 100% while keeping the exclusion power.

# The criterion must be two numbers; neither alone will do

  satisfaction = fraction of real ionic crystals that satisfy the law   -> want >=99%
  exclusion    = fraction of unrealised candidates that do **not**      -> the higher the better

Reporting satisfaction alone gives a tautology full marks (satisfaction 100%, exclusion 0),
which says nothing at all. What is wrong with Pauling's five rules is not that they are too
loose but that they are too tight; going to the opposite extreme is no more useful.

# The form of a law

One-sided interval constraints: feature <= hi or feature >= lo, with the threshold at the
(1-alpha) quantile of the real structures. Two-sided: lo <= feature <= hi. Structural
quantities only, and every threshold has a definite physical meaning.
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
    print(f'real {len(real):,} | candidates {len(neg):,} | shared features {len(cols)}')
    print(f'anions on the real side: {dict(real.anion.value_counts().head(8))}' if 'anion' in real else '')
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
    print(f'\n=== candidate laws with satisfaction >=98%: {len(r)}; top 15 by exclusion ===')
    for t in r.head(15).itertuples():
        print(f'  sat={t.sat:.4f} excl={t.rej:.4f}  {t.desc}')
    # greedy assembly: maximise joint exclusion while holding joint satisfaction up
    print(f'\n=== assembling the law set (each step takes the largest gain in joint exclusion) ===')
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
        print(f'  {len(chosen)}. {t.desc:44s} joint sat={satR.mean():.4f} joint excl={1-satN.mean():.4f}')
    print(f'\n=== law set N={len(chosen)} ===')
    print(f'  satisfaction on real ionic crystals = {satR.mean():.4f}   (Pauling 2-5: 0.13, 0.21 at CN<=8)')
    print(f'  exclusion on unrealised candidates  = {1-satN.mean():.4f}')
    json.dump({'N':len(chosen),'sat':float(satR.mean()),'rej':float(1-satN.mean()),
      'rules':[{'desc':t.desc,'sat':float(t.sat),'rej':float(t.rej)} for t in chosen]},
      open(F+'satisfy_rules.json','w'),ensure_ascii=False,indent=2)
    print(f'\nwrote {F}satisfy_rules.json')
    return 0
if __name__=='__main__': raise SystemExit(main())
