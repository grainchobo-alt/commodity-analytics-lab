import numpy as np, pandas as pd, time, sys
from backtest_core import load_daily
from bench_wf import build_paths, portfolio_from_H
from daily_engine import sharpe_d

models = sys.argv[1].split(',')
D=load_daily(); X=D.values; dates=D.index
wins=[('3년',756),('5년',1260),('7년',1764),('10년',2520),('확장','exp'),('IS고정','isfix')]
rows=[]
for wname,w in wins:
    r={'윈도우':wname}
    for m in models:
        Hp = build_paths(m, X, dates, 'd', w, refit_from=2009)
        rp = portfolio_from_H(Hp, X, dates, 'd', '2010-01-01')
        r[m]=round(sharpe_d(rp),3)
    rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv(f'bench_window_d_{"_".join(models)}.csv',index=False)
