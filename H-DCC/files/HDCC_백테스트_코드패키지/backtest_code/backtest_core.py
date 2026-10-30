"""H-DCC backtest core — loads from consolidated master file."""
import pandas as pd
import numpy as np

M = '/home/claude/hdcc/HDCC_Master_Data.xlsx'
TC = 0.00075  # 7.5bp one-way

def load():
    Y = pd.read_excel(M, sheet_name='01_Monthly_Returns_12', index_col=0)
    Y.index = pd.to_datetime(Y.index)
    sig = pd.read_excel(M, sheet_name='08_EWMA_Vol_Path')
    sig['Date'] = pd.to_datetime(sig['Date']); sig = sig.set_index('Date')
    sig.columns = [c.split(' (')[0] for c in sig.columns]
    z = pd.read_excel(M, sheet_name='09_Std_Residuals_z')
    z['Date'] = pd.to_datetime(z['Date']); z = z.set_index('Date')
    return Y, sig, z

def load_daily():
    D = pd.read_excel(M, sheet_name='02_Daily_Returns_12', index_col=0)
    D.index = pd.to_datetime(D.index)
    return D.ffill(limit=5).dropna()

def minvar_lo(H):
    K = H.shape[0]
    free = np.ones(K, bool)
    wf = np.ones(K)/K
    for _ in range(50):
        Hf = H[np.ix_(free, free)]
        try:
            inv1 = np.linalg.solve(Hf, np.ones(free.sum()))
        except np.linalg.LinAlgError:
            inv1 = np.linalg.pinv(Hf) @ np.ones(free.sum())
        wf = inv1 / inv1.sum()
        if (wf >= -1e-12).all():
            w = np.zeros(K); w[free] = np.clip(wf, 0, None); w /= w.sum()
            return w
        idx = np.where(free)[0]
        free[idx[np.argmin(wf)]] = False
        if free.sum() == 1:
            w = np.zeros(K); w[free] = 1.0
            return w
    w = np.zeros(K); w[free] = np.clip(wf, 0, None); w /= w.sum()
    return w

def backtest(Y, sig, z, oos_start='2010-01-01', win=60, label=''):
    dates = Y.index; K = Y.shape[1]
    rets, tos, w_prev = [], [], np.ones(K)/K
    for t in dates[dates >= oos_start]:
        i = dates.get_loc(t)
        s = sig.iloc[i-1].values / 100.0 / np.sqrt(12.0)
        R = np.corrcoef(z.iloc[i-win:i].values, rowvar=False); R = (R+R.T)/2
        H = np.outer(s, s) * R + np.eye(K)*1e-10
        w = minvar_lo(H)
        to = np.abs(w - w_prev).sum()
        rets.append(w @ Y.loc[t].values - TC*to); tos.append(to); w_prev = w
    rp = np.array(rets)
    sharpe = rp.mean()*12/(rp.std(ddof=1)*np.sqrt(12))
    cum = np.cumprod(1+rp); mdd = ((cum-np.maximum.accumulate(cum))/np.maximum.accumulate(cum)).min()
    return dict(label=label, sharpe=sharpe, ann_ret=rp.mean()*12, ann_vol=rp.std(ddof=1)*np.sqrt(12),
                maxdd=mdd, avg_to=np.mean(tos), n=len(rp)), rp

def ewma_qmle_vec(e, lo=0.79, hi=0.986, grid=50):
    """Vectorized-over-grid EWMA QMLE (fast for daily)."""
    e = np.asarray(e, float); e = e[~np.isnan(e)]
    lams = np.linspace(lo, hi, grid)
    s2 = np.full(grid, max(np.var(e[:min(60,len(e))]), 1e-12))
    ll = np.zeros(grid)
    e2 = e**2
    for t in range(1, len(e)):
        s2 = lams*s2 + (1-lams)*e2[t-1]
        np.maximum(s2, 1e-14, out=s2)
        ll += np.log(s2) + e2[t]/s2
    return lams[np.argmin(ll)]
