"""HMM window-length OOS analysis (2010-2025 fixed): monthly + daily, 6 window configs."""
import numpy as np, pandas as pd, time
from backtest_core import load, load_daily
from hmm_wf import hmm_lambda_path, sigma_z
from task8_hmm_wf import monthly_bt, daily_bt, jan_refits
from daily_engine import sharpe_d

Y,_,_ = load(); datesM=Y.index; YvM=Y.values
D = load_daily(); datesD=D.index; YvD=D.values

# refits only needed from 2009 onward (first OOS rebalance uses params as of Jan 2010; include 2009 for coverage)
refM_all = jan_refits(datesM, 48)
refD_all = jan_refits(datesD, 1008)
refM = [t for t in refM_all if datesM[t].year >= 2009]
refD = [t for t in refD_all if datesD[t].year >= 2009]
# isfix uses IS end:
isM = int(np.where(datesM<='2009-12-31')[0][-1])+1
isD = int(np.searchsorted(datesD, pd.Timestamp('2009-12-31'),'right'))

rows=[]
for wname, wm, wd in [('직전 3년',36,756),('직전 5년',60,1260),('직전 7년',84,1764),
                      ('직전 10년',120,2520),('확장(expanding)','exp','exp'),('IS 고정','isfix','isfix')]:
    t0=time.time()
    # monthly
    rfM = [isM] if wm=='isfix' else refM
    LAM = hmm_lambda_path(YvM, datesM, rfM, wm, 0.79, 0.986, 50, n_iter=40)
    SIG,Z = sigma_z(YvM, LAM, 13)
    rpM = monthly_bt(YvM, SIG, Z, datesM, '2010-01-01')
    shM = rpM.mean()*12/(rpM.std(ddof=1)*np.sqrt(12))
    # daily
    rfD = [isD] if wd=='isfix' else refD
    LAMd = hmm_lambda_path(YvD, datesD, rfD, wd, 0.90, 0.999, 30, n_iter=15)
    SIGd,Zd = sigma_z(YvD, LAMd, 253)
    rpD = daily_bt(YvD, SIGd, Zd, datesD, '2010-01-01')
    shD = sharpe_d(rpD)
    rows.append({'윈도우':wname,'HMM월':round(shM,3),'HMM일':round(shD,3)})
    print(rows[-1], f"({time.time()-t0:.0f}s)")
pd.DataFrame(rows).to_csv('task9_hmm_window.csv',index=False)
