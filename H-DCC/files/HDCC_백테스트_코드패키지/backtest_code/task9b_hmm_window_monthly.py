import numpy as np, pandas as pd, time
from backtest_core import load
from hmm_wf import hmm_lambda_path, sigma_z
from task8_hmm_wf import monthly_bt, jan_refits

Y,_,_ = load(); dates=Y.index; Yv=Y.values
ref_all = jan_refits(dates, 48)
ref = [t for t in ref_all if dates[t].year >= 2009]
isM = int(np.where(dates<='2009-12-31')[0][-1])+1
rows=[]
for wname, wm in [('직전 3년',36),('직전 5년',60),('직전 7년',84),('직전 10년',120),
                  ('확장(expanding)','exp'),('IS 고정','isfix')]:
    t0=time.time()
    rf = [isM] if wm=='isfix' else ref
    LAM = hmm_lambda_path(Yv, dates, rf, wm, 0.79, 0.986, 50, n_iter=80)
    SIG,Z = sigma_z(Yv, LAM, 13)
    rp = monthly_bt(Yv, SIG, Z, dates, '2010-01-01')
    sh = rp.mean()*12/(rp.std(ddof=1)*np.sqrt(12))
    rows.append({'윈도우':wname,'HMM월':round(sh,3)})
    print(rows[-1], f"({time.time()-t0:.0f}s)")
pd.DataFrame(rows).to_csv('task9b_hmm_window_m.csv',index=False)
