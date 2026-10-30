"""HMM walk-forward era (monthly+daily) and window-length OOS (monthly+daily)."""
import numpy as np, pandas as pd
from backtest_core import load, load_daily, minvar_lo, TC
from hmm_wf import hmm_lambda_path, sigma_z
from daily_engine import month_end_idx, sharpe_d, ERAS

def monthly_bt(Yv, SIG, Z, dates, start):
    K=Yv.shape[1]; rets=[]; idxs=[]; w_prev=np.ones(K)/K
    for i in np.where(dates>=start)[0]:
        if i<60: continue
        s=SIG[i-1]
        R=np.corrcoef(Z[i-60:i],rowvar=False); R=(R+R.T)/2
        H=np.outer(s,s)*R+np.eye(K)*1e-10
        w=minvar_lo(H); to=np.abs(w-w_prev).sum()
        rets.append(w@Yv[i]-TC*to); idxs.append(dates[i]); w_prev=w
    return pd.Series(rets, index=pd.DatetimeIndex(idxs))

def daily_bt(Yv, SIG, Z, dates, start, corr_win=252):
    K=Yv.shape[1]
    mes=month_end_idx(dates)
    mes=[m for m in mes if dates[m]>=pd.Timestamp(start)-pd.offsets.MonthEnd(1)]
    w_prev=np.ones(K)/K; rr=[]; dd=[]
    for j in range(len(mes)-1):
        t0,t1=mes[j],mes[j+1]
        if t0<corr_win+5: continue
        s=SIG[t0]
        R=np.corrcoef(Z[t0-corr_win:t0],rowvar=False); R=(R+R.T)/2
        H=np.outer(s,s)*R+np.eye(K)*1e-12
        w=minvar_lo(H); to=np.abs(w-w_prev).sum()
        seg=Yv[t0+1:t1+1]@w
        if len(seg)==0: continue
        seg=seg.copy(); seg[0]-=TC*to
        rr.extend(seg.tolist()); dd.extend(dates[t0+1:t1+1].tolist()); w_prev=w
    rp=pd.Series(rr,index=pd.DatetimeIndex(dd))
    return rp[rp.index>=start]

def jan_refits(dates, min_t):
    idx=[]; seen=set()
    for t in range(len(dates)):
        y=dates[t].year
        if y not in seen:
            seen.add(y)
            if t>=min_t: idx.append(t)
    return idx

import time
# ═══ A. ERA walk-forward: HMM monthly ═══
t0=time.time()
Y,_,_ = load(); datesM=Y.index; YvM=Y.values
refM = jan_refits(datesM, 48)
LAM = hmm_lambda_path(YvM, datesM, refM, 'exp', 0.79, 0.986, 50, n_iter=40)
SIG,Z = sigma_z(YvM, LAM, 13)
rpM = monthly_bt(YvM, SIG, Z, datesM, '1999-01-01')
shM = rpM.mean()*12/(rpM.std(ddof=1)*np.sqrt(12))
print(f"[A1] HMM monthly era WF: full SR={shM:.3f} ({time.time()-t0:.0f}s)")
rpM.to_csv('rp_hmm_m_wf.csv')

# ═══ B. ERA walk-forward: HMM daily ═══
t0=time.time()
D = load_daily(); datesD=D.index; YvD=D.values
refD = jan_refits(datesD, 1008)
LAMd = hmm_lambda_path(YvD, datesD, refD, 'exp', 0.90, 0.999, 30, n_iter=15)
SIGd,Zd = sigma_z(YvD, LAMd, 253)
rpD = daily_bt(YvD, SIGd, Zd, datesD, '1999-01-01')
print(f"[B1] HMM daily era WF: full SR={sharpe_d(rpD):.3f} ({time.time()-t0:.0f}s)")
rpD.to_csv('rp_hmm_d_wf.csv')

rows=[]
for name,a,b in ERAS:
    sm = rpM[(rpM.index>=a)&(rpM.index<=b)]
    sd = rpD[(rpD.index>=a)&(rpD.index<=b)]
    rows.append({'구간':name,
        'HMM월':round(sm.mean()*12/(sm.std(ddof=1)*np.sqrt(12)),3) if len(sm)>=12 else np.nan,
        'HMM일':round(sharpe_d(sd),3)})
print(pd.DataFrame(rows).to_string(index=False))
pd.DataFrame(rows).to_csv('task8_hmm_era.csv',index=False)
