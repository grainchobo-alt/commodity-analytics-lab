"""Daily engine: walk-forward era analysis + window-length analysis, 252d rolling corr."""
import numpy as np, pandas as pd
from backtest_core import load_daily, minvar_lo, TC, ewma_qmle_vec
from tvp_kalman import kalman_tvp_ar1

DLO, DHI = 0.90, 0.999  # daily lambda bounds

def month_end_idx(dates):
    s = pd.Series(np.arange(len(dates)), index=dates)
    return s.groupby([dates.year, dates.month]).last().values

def build_paths(model='hdcc', lam_mode='exp', Q=1e-5):
    """Returns dates, Yv, SIG, Z with causal lambda per lam_mode:
    lam_mode: 'exp' (annual expanding re-est), int W (trailing days), 'isfix'."""
    D = load_daily(); dates = D.index; K = D.shape[1]; Yv = D.values; T = len(dates)
    if model == 'tvp':
        EPS = np.column_stack([kalman_tvp_ar1(Yv[:,k], Q) for k in range(K)])
        EPS[0] = Yv[0]
    else:
        EPS = Yv.copy()
    is_end = np.searchsorted(dates, pd.Timestamp('2009-12-31'), 'right') - 1

    LAM = np.zeros((T,K))
    if lam_mode == 'isfix':
        lam0 = np.array([ewma_qmle_vec(EPS[1:is_end+1,k], DLO, DHI, 40) for k in range(K)])
        LAM[:] = lam0
    else:
        cur = np.full(K, 0.97)
        years = dates.year
        jan_first = np.zeros(T, bool)
        seen = set()
        for t in range(T):
            y = years[t]
            if y not in seen:
                seen.add(y); jan_first[t] = True
        for t in range(T):
            if jan_first[t] and t >= 756:  # >=3y history
                for k in range(K):
                    lo = 1 if lam_mode == 'exp' else max(1, t - lam_mode)
                    e = EPS[lo:t,k]; e = e[~np.isnan(e)]
                    if len(e) >= 500:
                        cur[k] = ewma_qmle_vec(e, DLO, DHI, 40)
            LAM[t] = cur

    SIG = np.zeros((T,K))
    for k in range(K):
        s2 = np.nanvar(EPS[1:253,k])
        for t in range(1,T):
            x = EPS[t-1,k]
            if not np.isnan(x): s2 = LAM[t,k]*s2 + (1-LAM[t,k])*x*x
            SIG[t,k] = np.sqrt(max(s2,1e-14))
    Z = EPS / np.where(SIG>0, SIG, 1.0)
    return dates, Yv, SIG, Z

def daily_backtest(dates, Yv, SIG, Z, start, corr_win=252):
    """Monthly rebalance at month-ends >= start; returns daily portfolio return series."""
    K = Yv.shape[1]
    mes = month_end_idx(dates)
    mes = [m for m in mes if dates[m] >= pd.Timestamp(start) - pd.offsets.MonthEnd(1)]
    w_prev = np.ones(K)/K
    out_r, out_d = [], []
    for j in range(len(mes)-1):
        t0, t1 = mes[j], mes[j+1]
        if t0 < corr_win + 5: continue
        s = SIG[t0]
        R = np.corrcoef(Z[t0-corr_win:t0], rowvar=False); R = (R+R.T)/2
        H = np.outer(s,s)*R + np.eye(K)*1e-12
        w = minvar_lo(H)
        to = np.abs(w-w_prev).sum()
        seg = Yv[t0+1:t1+1] @ w
        if len(seg)==0: continue
        seg = seg.copy(); seg[0] -= TC*to
        out_r.extend(seg.tolist()); out_d.extend(dates[t0+1:t1+1].tolist())
        w_prev = w
    rp = pd.Series(out_r, index=pd.DatetimeIndex(out_d))
    return rp[rp.index >= start]

def sharpe_d(rp):
    return rp.mean()*252/(rp.std(ddof=1)*np.sqrt(252)) if len(rp)>40 else np.nan

ERAS = [
    ("닷컴버블/금융화 이전", '1999-01-01','2003-12-31'),
    ("원자재 붐(금융화)",   '2004-01-01','2008-06-30'),
    ("글로벌 금융위기(GFC)",'2008-07-01','2009-12-31'),
    ("포스트-GFC/QE",      '2010-01-01','2014-12-31'),
    ("유가폭락/중간사이클", '2015-01-01','2019-12-31'),
    ("COVID-19",           '2020-01-01','2021-12-31'),
    ("우크라이나/인플레",   '2022-01-01','2023-12-31'),
    ("최근(2024~)",        '2024-01-01','2025-12-31'),
    ("전체(1999~2025)",    '1999-01-01','2025-12-31'),
]
