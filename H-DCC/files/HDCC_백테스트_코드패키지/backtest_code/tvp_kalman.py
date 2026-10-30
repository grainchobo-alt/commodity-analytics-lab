"""TVP-AR(1) Kalman: time-varying (alpha_t, phi_t) per asset.
IS 1994-2009 calibration; OOS expanding (filtered estimates only, no look-ahead).
Innovation extraction replaces Ridge-VAR; rest of H-DCC pipeline identical:
  sigma_k,t = EWMA(lambda_k) on TVP residuals; R_t = rolling-60 corr of standardized resid.
"""
import pandas as pd, numpy as np
from backtest_core import load, minvar_lo, TC

def kalman_tvp_ar1(r, Q_scale=1e-5, R_var=None):
    """Random-walk TVP-AR(1). Returns one-step-ahead residuals (filtered, causal).
    State x=[alpha,phi], obs r_t = [1, r_{t-1}] x_t + eps."""
    r = np.asarray(r, float)
    T = len(r)
    if R_var is None:
        R_var = np.var(r[1:61]) if T > 61 else np.var(r)
    x = np.array([np.mean(r[:60]), 0.0])
    P = np.eye(2) * 0.1
    Q = np.eye(2) * Q_scale
    eps = np.full(T, np.nan)
    for t in range(1, T):
        # predict
        P = P + Q
        Hm = np.array([1.0, r[t-1]])
        # one-step-ahead prediction residual (this is the causal innovation)
        pred = Hm @ x
        eps[t] = r[t] - pred
        # update
        S = Hm @ P @ Hm + R_var
        Kg = P @ Hm / S
        x = x + Kg * eps[t]
        P = (np.eye(2) - np.outer(Kg, Hm)) @ P
    return eps

def ewma_qmle_lambda(e, lo=0.79, hi=0.986, grid=200):
    """QMLE for EWMA decay on residual series (IS only)."""
    e = e[~np.isnan(e)]
    lams = np.linspace(lo, hi, grid)
    best, bl = np.inf, lams[0]
    for lam in lams:
        s2 = np.var(e[:12]); ll = 0.0
        for t in range(1, len(e)):
            s2 = lam*s2 + (1-lam)*e[t-1]**2
            s2 = max(s2, 1e-12)
            ll += np.log(s2) + e[t]**2/s2
        if ll < best: best, bl = ll, lam
    return bl

def run_tvp(Q_scale=1e-5):
    Y, sig_paper, z_paper = load()
    dates = Y.index; K = Y.shape[1]
    is_mask = dates <= '2009-12-31'
    T_is = is_mask.sum()

    # 1) Kalman residuals over FULL sample (causal: filtered only, params init from first 60m of IS)
    EPS = np.column_stack([kalman_tvp_ar1(Y.iloc[:,k].values, Q_scale) for k in range(K)])

    # 2) lambda_k: QMLE on IS residuals only
    lams = np.array([ewma_qmle_lambda(EPS[1:T_is,k]) for k in range(K)])

    # 3) EWMA sigma path on residuals (full sample, seeded on first 12m)
    SIG = np.full_like(EPS, np.nan)
    for k in range(K):
        e = EPS[:,k]; lam = lams[k]
        s2 = np.nanvar(e[1:13])
        for t in range(1, len(e)):
            if not np.isnan(e[t-1]):
                s2 = lam*s2 + (1-lam)*e[t-1]**2
            SIG[t,k] = np.sqrt(max(s2,1e-12))

    # 4) standardized residuals
    Z = EPS / SIG

    # 5) backtest OOS
    win = 60
    rets, tos = [], []
    w_prev = np.ones(K)/K
    oos_idx = np.where(dates >= '2010-01-01')[0]
    for i in oos_idx:
        s = SIG[i-1]  # info to t-1 (monthly decimal already)
        zwin = Z[i-win:i]
        R = np.corrcoef(zwin, rowvar=False); R=(R+R.T)/2
        H = np.outer(s,s)*R + np.eye(K)*1e-10
        w = minvar_lo(H)
        to = np.abs(w-w_prev).sum()
        rets.append(w @ Y.iloc[i].values - TC*to)
        tos.append(to); w_prev = w
    rp = np.array(rets)
    sh = rp.mean()*12/(rp.std(ddof=1)*np.sqrt(12))
    cum=np.cumprod(1+rp); mdd=((cum-np.maximum.accumulate(cum))/np.maximum.accumulate(cum)).min()
    return dict(sharpe=sh, maxdd=mdd, ann_ret=rp.mean()*12, ann_vol=rp.std(ddof=1)*np.sqrt(12),
                avg_to=np.mean(tos), lams=lams), rp

if __name__=='__main__':
    for Q in [1e-6, 1e-5, 1e-4]:
        res,_ = run_tvp(Q)
        print(f"Q={Q:.0e}: Sharpe={res['sharpe']:.3f} MaxDD={res['maxdd']*100:.1f}% "
              f"Vol={res['ann_vol']*100:.1f}% TO={res['avg_to']*100:.1f}%")
