"""Daily TVP-AR(1), full sample 1994-2025, monthly rebalanced.
Compare vs daily H-DCC-style baseline (fixed lambda) under identical conventions.
Sharpe computed on monthly portfolio returns (x12/sqrt12) for comparability with 0.479 line,
AND on daily returns (x252/sqrt252) for comparability with the 0.624/0.890 technote line.
"""
import pandas as pd, numpy as np
from backtest_core import minvar_lo, TC
from tvp_kalman import kalman_tvp_ar1, ewma_qmle_lambda

M = '/home/claude/hdcc/HDCC_Master_Data.xlsx'

def load_daily():
    d = pd.read_excel(M, sheet_name='02_Daily_Returns_12', index_col=0)
    d.index = pd.to_datetime(d.index)
    return d.dropna(how='all').ffill(limit=2).dropna()

def month_ends(idx):
    s = pd.Series(1, index=idx)
    return s.groupby([idx.year, idx.month]).apply(lambda x: x.index[-1]).values

def run(daily_mode='tvp', Q=1e-5, corr_win=252):
    D = load_daily(); dates = D.index; K = D.shape[1]; Dv = D.values
    is_end = np.searchsorted(dates, pd.Timestamp('2009-12-31'), 'right')-1

    # innovations
    if daily_mode=='tvp':
        EPS = np.column_stack([kalman_tvp_ar1(Dv[:,k], Q) for k in range(K)])
    else:
        EPS = Dv.copy()  # raw (H-DCC daily uses raw Y for variance per paper note)

    # daily lambda via QMLE on IS (coarse grid for speed)
    lams = np.array([ewma_qmle_lambda(EPS[1:is_end+1,k][~np.isnan(EPS[1:is_end+1,k])][:3500], grid=40, lo=0.90, hi=0.999) for k in range(K)])

    SIG = np.zeros_like(Dv)
    for k in range(K):
        e = EPS[:,k]; lam=lams[k]; s2=np.nanvar(e[1:60])
        for t in range(1,len(e)):
            x = e[t-1]
            if not np.isnan(x): s2 = lam*s2+(1-lam)*x*x
            SIG[t,k]=np.sqrt(max(s2,1e-14))
    Z = np.where(SIG>0, EPS/SIG, 0.0)

    mes = month_ends(dates)
    mes = [m for m in mes if m >= pd.Timestamp('2009-12-31')]
    w_prev = np.ones(K)/K
    daily_rets = []
    monthly_rets = []
    for j in range(len(mes)-1):
        t0 = np.searchsorted(dates, mes[j])
        t1 = np.searchsorted(dates, mes[j+1])
        if dates[t0] < pd.Timestamp('2009-12-31'): continue
        s = SIG[t0]
        zw = Z[max(0,t0-corr_win):t0]
        R = np.corrcoef(zw, rowvar=False); R=(R+R.T)/2
        H = np.outer(s,s)*R + np.eye(K)*1e-12
        w = minvar_lo(H)
        to = np.abs(w-w_prev).sum()
        seg = Dv[t0+1:t1+1] @ w
        seg[0] -= TC*to
        daily_rets.extend(seg.tolist())
        monthly_rets.append(np.exp(np.log1p(pd.Series(seg)).sum())-1 if False else seg.sum())
        w_prev = w
    dr = np.array(daily_rets); mr = np.array(monthly_rets)
    sh_d = dr.mean()*252/(dr.std(ddof=1)*np.sqrt(252))
    sh_m = mr.mean()*12/(mr.std(ddof=1)*np.sqrt(12))
    return dict(mode=daily_mode, lam_med=np.median(lams), sh_daily=sh_d, sh_monthly=sh_m, n_m=len(mr))

if __name__=='__main__':
    import time
    for mode in ['baseline','tvp']:
        t0=time.time()
        r = run(mode)
        print(f"{mode:9s}: Sharpe(daily-conv)={r['sh_daily']:.3f}  Sharpe(monthly-conv)={r['sh_monthly']:.3f}  "
              f"median λ={r['lam_med']:.4f}  [{time.time()-t0:.0f}s, {r['n_m']}m]")
