"""DM tests vs H-DCC, MCS (stationary bootstrap), LR path on walk-forward forecasts."""
import numpy as np, pandas as pd
from backtest_core import load, load_daily
from bench_core import ewma_lams_ll, pooled_lambda
from bench_wf import jan_refits, MODELS

def nw_se(d, L):
    T=len(d); g0=np.var(d,ddof=0); s=g0
    for l in range(1,L+1):
        w=1-l/(L+1)
        s+=2*w*np.cov(d[l:],d[:-l],ddof=0)[0,1]
    return np.sqrt(max(s,1e-12)/T)

def dm_table(QL, base='HDCC', freq='m'):
    L = 12 if freq=='m' else 22
    rows=[]
    for m in QL.columns:
        if m==base: continue
        d = (QL[base]-QL[m]).dropna().values  # negative => HDCC better (lower loss)
        t = d.mean()/nw_se(d,L)
        rows.append({'모형':m,'평균손실차(HDCC−모형)':round(d.mean(),3),
                     'DM t-stat':round(t,2),
                     '해석':'H-DCC 우위***' if t<-2.58 else ('H-DCC 우위' if t<-1.64 else ('열위***' if t>2.58 else ('열위' if t>1.64 else '무차별')))})
    return pd.DataFrame(rows)

def mcs(QL, B=500, l_block=None, alpha=0.10, seed=7):
    """Hansen MCS, range statistic, stationary bootstrap."""
    rng=np.random.default_rng(seed)
    L = QL.dropna().values  # (T,M)
    T,M = L.shape
    if l_block is None: l_block = max(2,int(T**(1/3)))
    models = list(QL.columns)
    incl = list(range(M))
    pvals = {}
    def boot_idx():
        idx=np.empty(T,dtype=int); i=rng.integers(T)
        for t in range(T):
            if rng.random()<1/l_block: i=rng.integers(T)
            idx[t]=i; i=(i+1)%T
        return idx
    bidx = [boot_idx() for _ in range(B)]
    while len(incl)>1:
        Ls = L[:,incl]
        dbar = Ls.mean(0)
        # pairwise t stats
        tmax=-np.inf; elim=None
        # compute d_ij and bootstrap var
        Tstats=np.zeros((len(incl),len(incl)))
        boot_means = np.stack([Ls[bi].mean(0) for bi in bidx])  # (B,m)
        for a in range(len(incl)):
            for b in range(a+1,len(incl)):
                d = dbar[a]-dbar[b]
                bd = boot_means[:,a]-boot_means[:,b]
                se = bd.std(ddof=1)
                tt = abs(d)/max(se,1e-12)
                Tstats[a,b]=Tstats[b,a]=tt
        TR = Tstats.max()
        # bootstrap distribution of range stat under null (center)
        TR_b=np.zeros(B)
        for bi in range(B):
            bm = boot_means[bi]-dbar
            M2=np.zeros((len(incl),len(incl)))
            for a in range(len(incl)):
                for b in range(a+1,len(incl)):
                    bd = boot_means[:,a]-boot_means[:,b]
                    se = bd.std(ddof=1)
                    M2[a,b]=M2[b,a]=abs(bm[a]-bm[b])/max(se,1e-12)
            TR_b[bi]=M2.max()
        p = (TR_b>=TR).mean()
        worst = int(np.argmax(dbar))
        pvals[models[incl[worst]]] = round(p,3)
        if p>=alpha:
            for k in incl: pvals.setdefault(models[k], round(p,3))
            break
        incl.pop(worst)
    if len(incl)==1: pvals.setdefault(models[incl[0]],1.0)
    return pvals

for freq in ['m','d']:
    QL = pd.read_csv(f'bench_ql_{freq}.csv',index_col=0,parse_dates=True)
    print(f"\n════ {'월간' if freq=='m' else '일간'} (워크포워드 1999~2025) ════")
    print("QLIKE 평균(낮을수록 좋음):")
    print(QL.mean().round(3).sort_values().to_string())
    print("\nDM vs H-DCC:")
    print(dm_table(QL, freq=freq).to_string(index=False))
    pv = mcs(QL, B=300)
    print("\nMCS p-values (p≥0.10 = 신뢰집합 포함):")
    print(pd.Series(pv).sort_values(ascending=False).to_string())
    # OOS-only subset
    QLo = QL[QL.index>='2010-01-01']
    print("\n[OOS 2010~25만] QLIKE 평균:")
    print(QLo.mean().round(3).sort_values().to_string())
    pd.Series(pv).to_csv(f'bench_mcs_{freq}.csv')
    dm_table(QL,freq=freq).to_csv(f'bench_dm_{freq}.csv',index=False)

# ═══ LR path (heterogeneous vs pooled lambda) at each refit ═══
print("\n════ LR 경로 (자산별 λ vs 단일 λ, df=11) ════")
for freq in ['m','d']:
    if freq=='m':
        Y,_,_=load(); X=Y.values; dates=Y.index; lo,hi,G=0.79,0.986,50; min_t=48
    else:
        D=load_daily(); X=D.values; dates=D.index; lo,hi,G=0.90,0.999,40; min_t=1008
    rows=[]
    for t0 in jan_refits(dates,min_t)[::3]:  # every 3rd year for brevity
        Xf=X[:t0]
        _,llm,_=ewma_lams_ll(Xf,lo,hi,G)
        _,llp=pooled_lambda(Xf,lo,hi,G)
        LR = llp - llm.sum()
        rows.append({'refit연도':dates[t0].year,'LR':round(LR,1),'p<0.001(χ²₁₁>31.3)':'✓' if LR>31.3 else '✗'})
    print(('월간' if freq=='m' else '일간'), pd.DataFrame(rows).to_string(index=False))
    pd.DataFrame(rows).to_csv(f'bench_lr_{freq}.csv',index=False)
