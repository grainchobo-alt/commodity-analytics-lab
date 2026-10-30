import pandas as pd, numpy as np
from daily_engine import build_paths, daily_backtest, sharpe_d, ERAS

res = {}
for model in ['hdcc','tvp']:
    dates, Yv, SIG, Z = build_paths(model, lam_mode='exp')
    rp = daily_backtest(dates, Yv, SIG, Z, '1999-01-01')
    res[model] = rp
    print(f"{model}: n_days={len(rp)}, full SR={sharpe_d(rp):.3f}")

rows=[]
for name,a,b in ERAS:
    r={'구간':name,'기간':f"{a[:7]}~{b[:7]}"}
    for m in ['hdcc','tvp']:
        seg = res[m][(res[m].index>=a)&(res[m].index<=b)]
        r[m.upper()+' SR'] = round(sharpe_d(seg),3)
    r['Δ(TVP−H)'] = round(r['TVP SR']-r['HDCC SR'],3)
    rows.append(r)
df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv('/home/claude/hdcc/task3_era_daily.csv', index=False)
res['hdcc'].to_csv('/home/claude/hdcc/rp_d_hdcc_wf.csv'); res['tvp'].to_csv('/home/claude/hdcc/rp_d_tvp_wf.csv')
