"""Run all benchmark models walk-forward (expanding): monthly + daily.
Saves rp + qlike series per model, prints era tables."""
import numpy as np, pandas as pd, time, sys
from backtest_core import load, load_daily
from bench_wf import build_paths, portfolio_from_H, qlike_from_H, MODELS
from daily_engine import sharpe_d, ERAS

freq = sys.argv[1] if len(sys.argv)>1 else 'm'
if freq=='m':
    Y,_,_ = load(); X=Y.values; dates=Y.index
else:
    D = load_daily(); X=D.values; dates=D.index

rps={}; qls={}
for m in MODELS:
    t0=time.time()
    Hp = build_paths(m, X, dates, freq, 'exp')
    rp = portfolio_from_H(Hp, X, dates, freq, '1999-01-01')
    ql = qlike_from_H(Hp, X, dates, '1999-01-01')
    rps[m]=rp; qls[m]=ql
    if freq=='m':
        sh = rp.mean()*12/(rp.std(ddof=1)*np.sqrt(12))
    else:
        sh = sharpe_d(rp)
    print(f"{m:5s}: SR(full99-25)={sh:.3f}  QLIKE(mean)={ql.mean():.3f}  [{time.time()-t0:.0f}s]", flush=True)

RP = pd.concat(rps,axis=1); RP.columns=MODELS
QL = pd.concat(qls,axis=1); QL.columns=MODELS
RP.to_csv(f'bench_rp_{freq}.csv'); QL.to_csv(f'bench_ql_{freq}.csv')

# era table
rows=[]
for name,a,b in ERAS:
    r={'구간':name}
    for m in MODELS:
        seg = RP[m][(RP.index>=a)&(RP.index<=b)].dropna()
        if freq=='m':
            r[m]=round(seg.mean()*12/(seg.std(ddof=1)*np.sqrt(12)),3) if len(seg)>=12 else np.nan
        else:
            r[m]=round(sharpe_d(seg),3)
    rows.append(r)
era = pd.DataFrame(rows)
print(era.to_string(index=False))
era.to_csv(f'bench_era_{freq}.csv',index=False)
