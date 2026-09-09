import os
#!/usr/bin/env python3
"""Physical descriptors: Madelung energy + effective coordination number + packing
fraction + second-nearest neighbours.

Why add these: the existing 32 descriptors are all topological or counting quantities, and
they cap out at 0.7143 for GBDT and 0.6673 for a linear model. The gap is not in the model
form (nonlinear PySR reaches only 0.6619) but in the **information content itself**.

The Madelung energy (Ewald summation) is the key one: Pauling's five rules are essentially
approximations to electrostatic arguments -- the second (bond strengths sum to the charge)
is local electroneutrality, and the third and fourth (edge and face sharing are unstable)
are Coulomb repulsion between cations.
Ewald is the **exact version** of those arguments, using only structure and formal charges,
never touching DFT, entirely within the scope of empirical criteria -- and it is exactly
the physics Pauling himself used.
"""
import sys, argparse, warnings, collections
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from polymorph_rank2 import scan, balance
F=os.environ.get("PRIS_FEATURES", "features/")
# coarse Shannon radius table (coordination-independent representative values), for the packing fraction
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
        # distribution of the site-level Madelung potential. This version of EwaldSummation
        # has no site_energies attribute, so take them one site at a time with
        # get_site_energy(i); if that fails, skip these three columns.
        try:
            sm=np.array([ew.get_site_energy(i) for i in range(n)])
            out['mad_std']=float(np.std(sm)); out['mad_max']=float(np.max(sm)); out['mad_min']=float(np.min(sm))
        except Exception:
            pass
        # packing fraction (volume of the ionic spheres / cell volume)
        vol=sum(4/3*np.pi*R.get(s.specie.symbol,1.0)**3 for s in st)
        out['pack_frac']=float(vol/st.volume)
        # effective coordination number (continuous form): ECoN, weighted by bond length
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
    print(f'{len(g):,} groups / {len(recs):,} endpoints',flush=True)
    from concurrent.futures import ProcessPoolExecutor
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i,r in enumerate(ex.map(one,recs,chunksize=8)):
            if r: rows.append(r)
            if (i+1)%4000==0: print(f'  {i+1:,}/{len(recs):,} -> {len(rows):,}',flush=True)
    d=pd.DataFrame(rows); d.to_parquet(F+'phys_feat.parquet',index=False)
    print(f'wrote {len(d):,} rows / {d.rk.nunique():,} groups')
    return 0
if __name__=='__main__': raise SystemExit(main())
