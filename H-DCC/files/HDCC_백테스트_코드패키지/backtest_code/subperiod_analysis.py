"""Analysis 1: Walk-forward sub-period comparison (1999-2025, causal).
At each Jan, lambda_k re-estimated by QMLE on ALL data up to t-1 (expanding).
TVP Kalman runs causally from 1994. Rolling-60 corr. MinVar long-only, TC 7.5bp.
Slice portfolio returns by economic eras + compute per-era Sharpe.
"""
import pandas as pd, numpy as np
from backtest_core import load, minvar_lo, TC
from tvp_kalman import kalman_tvp_ar1, ewma_qmle_lambda

def causal_walkforward(model='hdcc', Q=1e-5, start_eval='1999-01-01'):
    Y, _, _ = load()
    dates = Y.index; K = Y.shape[1]; T = len(dates); Yv = Y.values

    # innovations
    if model == 'tvp':
        EPS = np.column_stack([kalman_tvp_ar1(Yv[:,k], Q) for k in range(K)])
        EPS[0] = Yv[0]  # seed
    else:
        EPS = Yv.copy()

    # expanding lambda: re-estimate each January on data up to prior Dec
    LAM = np.zeros((T,K))
    cur = np.full(K, 0.94)
    for t in range(T):
        if dates[t].month == 1 and t >= 48:  # need >=4y history
            for k in range(K):
                e = EPS[1:t,k]; e = e[~np.isnan(e)]
                cur[k] = ewma_qmle_lambda(e, grid=50)
        LAM[t] = cur

    # sigma path with (causal) time-varying lambda
    SIG = np.zeros((T,K))
    for k in range(K):
        s2 = np.nanvar(EPS[1:13,k])
        for t in range(1,T):
            x = EPS[t-1,k]
            if not np.isnan(x): s2 = LAM[t,k]*s2 + (1-LAM[t,k])*x*x
            SIG[t,k] = np.sqrt(max(s2,1e-12))
    Z = np.where(SIG>0, EPS/SIG, 0.0)

    rets, idxs = [], []
    w_prev = np.ones(K)/K
    for i in np.where(dates >= start_eval)[0]:
        s = SIG[i-1]
        R = np.corrcoef(Z[i-60:i], rowvar=False); R=(R+R.T)/2
        H = np.outer(s,s)*R + np.eye(K)*1e-10
        w = minvar_lo(H)
        to = np.abs(w-w_prev).sum()
        rets.append(w @ Yv[i] - TC*to); idxs.append(dates[i]); w_prev = w
    return pd.Series(rets, index=pd.DatetimeIndex(idxs))

def era_table(rp_dict):
    eras = [
        ("Dot-com/금융화 이전", '1999-01-01','2003-12-31'),
        ("원자재 붐(금융화)",   '2004-01-01','2008-06-30'),
        ("글로벌 금융위기(GFC)",'2008-07-01','2009-12-31'),
        ("포스트-GFC/QE",      '2010-01-01','2014-12-31'),
        ("유가폭락/중간사이클", '2015-01-01','2019-12-31'),
        ("COVID-19",           '2020-01-01','2021-12-31'),
        ("우크라이나전쟁/인플레",'2022-01-01','2023-12-31'),
        ("최근(2024~)",        '2024-01-01','2025-12-31'),
        ("전체(1999~2025)",    '1999-01-01','2025-12-31'),
    ]
    rows=[]
    for name,a,b in eras:
        row={'구간':name,'기간':f"{a[:7]}~{b[:7]}"}
        for lbl,rp in rp_dict.items():
            seg = rp[(rp.index>=a)&(rp.index<=b)]
            n=len(seg)
            sh = seg.mean()*12/(seg.std(ddof=1)*np.sqrt(12)) if n>=12 else np.nan
            row[f'{lbl} SR']=round(sh,3); row['n(월)']=n
        rows.append(row)
    return pd.DataFrame(rows)

if __name__=='__main__':
    rp_h = causal_walkforward('hdcc')
    rp_t = causal_walkforward('tvp')
    tab = era_table({'H-DCC':rp_h,'TVP':rp_t})
    tab['Δ(TVP−H)'] = (tab['TVP SR']-tab['H-DCC SR']).round(3)
    print(tab.to_string(index=False))
    tab.to_csv('/home/claude/hdcc/subperiod_results.csv', index=False)
    rp_h.to_csv('/home/claude/hdcc/rp_hdcc_wf.csv'); rp_t.to_csv('/home/claude/hdcc/rp_tvp_wf.csv')
