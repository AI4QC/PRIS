import os
#!/usr/bin/env python3
"""Build genuine negatives: damaging perturbations of real structures.

# Why the negatives had to be rebuilt

The first two versions used LeMat/ELEMENTA candidates as negatives and reached only 28%
exclusion. The reason: those are **DFT local minima**, geometrically entirely reasonable and
merely slightly higher in energy. With them as the "implausible" samples, the laws can only
learn surface differences such as "large volume", never "what makes a structure not make
sense".

What ought to be excluded is structures that are geometrically absurd. So four classes of
damage are applied to real structures, each corresponding to a failure mode one of Pauling's
rules cares about:

  S1 anisotropic lattice compression -- destroys the packing, creating over-short and
                                        over-long bonds
  S2 cation transposition            -- destroys the coordination environment and the charge
                                        distribution (Pauling's fifth rule)
  S3 random displacement             -- destroys local symmetry and bond-length uniformity
                                        (Pauling's second rule)
  S4 uniform lattice expansion       -- creates loose, under-coordinated structures

These structures **do not exist chemically**, yet their composition, stoichiometry and atom
count are identical to the parent's -- so if the laws can exclude them, what they exclude is
geometric plausibility itself, not a surface difference in composition or size.
"""
import sys, argparse, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discriminate import criteria, guess_oxi, read_blob_cif
F=os.environ.get("PRIS_FEATURES", "features/")

def swapped_val(s, val):
    """Return the valence array to use after perturbation. Under an S2 transposition the
    charge has to travel with the element."""
    sw=getattr(s,'_swapped_val',None)
    if sw is None: return val
    v=list(val); i,j=sw; v[i],v[j]=v[j],v[i]; return v


def perturb(st, kind, rng, val=None):
    from pymatgen.core import Structure
    s=st.copy()
    if kind=='S1':                      # anisotropic compression, 15-30%
        f=np.eye(3); ax=rng.integers(0,3); f[ax,ax]=1-rng.uniform(0.15,0.30)
        s.lattice=type(s.lattice)(np.dot(f,s.lattice.matrix))
    elif kind=='S2':                    # cation transposition
        # Crucial correction: the two cations swapped must have **different charges**.
        # The original implementation only required different elements, but if their charges
        # match (both +2, say) the Madelung energy is strictly unchanged -- that is not
        # another structure, it is the same structure relabelled.
        # Measured: all 126 S2 samples from the original implementation had an electrostatic
        # penalty of **exactly 0**, which is why four rounds of feature engineering could not
        # exclude them: they were plausible all along.
        if val is None: return None
        cats=[i for i in range(len(s)) if val[i]>0]
        pairs=[(i,j) for i in cats for j in cats
               if i<j and abs(val[i]-val[j])>=1.0 and s[i].specie.symbol!=s[j].specie.symbol]
        if not pairs: return None
        i,j=pairs[rng.integers(0,len(pairs))]
        si,sj=s[i].specie,s[j].specie
        s.replace(int(i),sj); s.replace(int(j),si)
        # The charge must be swapped along with the element. The earlier implementation
        # swapped only the element, while downstream code reattached the original charges
        # **by site index** through add_oxidation_state_by_site(val) / criteria(p,val) -- so
        # the transposition was invisible to every charge-dependent quantity and the Madelung
        # dE was strictly 0. That is the real reason S2 survived four rounds.
        s._swapped_val=(int(i),int(j))
    elif kind=='S3':                    # random displacement, 0.3-0.8 A
        amp=rng.uniform(0.3,0.8)
        for i in range(len(s)):
            s.translate_sites(i, rng.normal(0,amp,3), frac_coords=False)
    elif kind=='S4':                    # uniform expansion, 20-40%
        f=1+rng.uniform(0.20,0.40)
        s.lattice=type(s.lattice)(s.lattice.matrix*f)
    elif kind=='S5':                    # cation-anion swap
        # The leave-one-perturbation-class-out test showed the laws **only constrain the
        # damage directions they have seen** (see the results summary, 5.1c).
        # S1-S4 are all geometric damage; none of them directly tests "charge placed on the
        # wrong site".
        # Swapping one cation with one anion leaves composition and stoichiometry entirely
        # unchanged, but scrambles the spatial arrangement of positive and negative charge,
        # which should be strongly unfavourable electrostatically.
        if val is None: return None
        cats=[i for i in range(len(s)) if val[i]>0]
        ans =[i for i in range(len(s)) if val[i]<0]
        if not cats or not ans: return None
        i=cats[rng.integers(0,len(cats))]; j=ans[rng.integers(0,len(ans))]
        si,sj=s[i].specie,s[j].specie
        s.replace(int(i),sj); s.replace(int(j),si)
        s._swapped_val=(int(i),int(j))   # the charge travels with the element, as in S2
    # S6, shear strain (10-25%), was tried and **dropped**: neither independent probe could
    # tell it was clearly implausible -- the median Madelung dE is -0.004 eV/atom (shear
    # preserves volume, so electrostatics is all but blind to it), and the median bl_min is
    # 0.881 against 0.937 for real structures, with only 10% falling below 0.804.
    # Forcing it into the negatives would repeat the S2 mistake: spending effort excluding a
    # batch of samples that may not deserve excluding.
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
    print(f'{len(recs):,} parents, 4 perturbation classes each',flush=True)
    from concurrent.futures import ProcessPoolExecutor
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i,rr in enumerate(ex.map(one,recs,chunksize=8)):
            rows.extend(rr)
            if (i+1)%1500==0: print(f'  {i+1:,}/{len(recs):,} -> {len(rows):,}',flush=True)
    o=pd.DataFrame(rows); o.to_parquet(F+'negatives.parquet',index=False)
    print(f'wrote {len(o):,} negative samples')
    if len(o): print(o.kind.value_counts().to_dict())
    return 0
if __name__=='__main__': raise SystemExit(main())
