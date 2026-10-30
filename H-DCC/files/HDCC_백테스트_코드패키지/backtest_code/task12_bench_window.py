"""Window sensitivity for all benchmark models (OOS 2010-2025 fixed), monthly + daily."""
import numpy as np, pandas as pd, time, sys
from backtest_core import load, load_daily
from bench_wf import build_paths, portfolio_from_H, MODELS
from daily_engine import sharpe_d

freq = sys.argv[1]
if freq=='m':
    Y,_,_=load(); X=Y.values; dates=Y.index
    wins=[('3년',36),('5년',60),('7년',84),('10년',120),('확장',\
'exp'),('IS고정','isfix')]
else:
    D=load_daily(); X=D.values; dates=D.index
    wins=[('3년',756),('5년',1260),('7년',1764),('10년',2520),('확장','exp'),('IS고정','isfix')]

rows=[]
for wname,w in wins:
    r={'윈도우':wname}
    for m in MODELS:
        t0=time.time()
        Hp = build_paths(m, X, dates, freq, w, refit_from=2009)
        rp = portfolio_from_H(Hp, X, dates, freq, '2010-01-01')
        if freq=='m':
            sh = rp.mean()*12/(rp.std(ddof=1)*np.sqrt(12))
        else:
            sh = sharpe_d(rp)
        r[m]=round(sh,3)
    rows.append(r); print(r, flush=True)
df=pd.DataFrame(rows)
df.to_csv(f'bench_window_{freq}.csv',index=False)
