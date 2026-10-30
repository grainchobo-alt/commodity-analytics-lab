import pandas as pd, numpy as np
from daily_engine import build_paths, daily_backtest, sharpe_d

rows=[]
for wname, w in [('3년(756d)',756),('5년(1260d)',1260),('7년(1764d)',1764),
                 ('10년(2520d)',2520),('확장(expanding)','exp'),('IS고정(94-09)','isfix')]:
    r={'λ 추정 윈도우':wname}
    for model in ['hdcc','tvp']:
        dates,Yv,SIG,Z = build_paths(model, lam_mode=w)
        rp = daily_backtest(dates,Yv,SIG,Z,'2010-01-01')
        r[model.upper()+' SR']=round(sharpe_d(rp),3)
    r['Δ(TVP−H)']=round(r['TVP SR']-r['HDCC SR'],3)
    rows.append(r); print(r)
pd.DataFrame(rows).to_csv('/home/claude/hdcc/task5_window_daily.csv',index=False)
