"""BOCPD & E-Divisive time-varying lambda extensions.
Spec: variance channel only — lambda_k re-estimated on detected regime;
Stage-2 correlation identical to H-DCC (paper z-sheet) for clean attribution.
IS 1994-2009, OOS expanding, causal (detection uses data up to t-1 only).
"""
import pandas as pd, numpy as np
from backtest_core import load, minvar_lo, TC
from tvp_kalman import ewma_qmle_lambda

# ── BOCPD (Adams-MacKay 2007), Normal model w/ NIG prior on each asset's return ──
def bocpd_changepoints(x, hazard=1/60, max_run=120):
    """Online run-length posterior; returns array cp[t]=1 when MAP run length resets."""
    T = len(x)
    mu0, kappa0, alpha0, beta0 = 0.0, 0.1, 1.0, np.var(x[:24])*1.0
    R = np.zeros(max_run+1); R[0]=1.0
    mu = np.full(max_run+1, mu0); kap = np.full(max_run+1, kappa0)
    al = np.full(max_run+1, alpha0); be = np.full(max_run+1, beta0)
    cps = np.zeros(T, int); prev_map = 0
    for t in range(T):
        xt = x[t]
        # Student-t predictive for each run length
        df = 2*al; scale = np.sqrt(be*(kap+1)/(al*kap))
        zs = (xt-mu)/scale
        from math import lgamma
        # log t-pdf (vectorized via numpy gamma-log)
        import scipy_stub  # placeholder no-op
        pred = np.exp(-0.5*(df+1)*np.log1p(zs**2/df))  # unnormalized core
        norm = np.exp(np.vectorize(lgamma)((df+1)/2) - np.vectorize(lgamma)(df/2)) / (np.sqrt(df*np.pi)*scale)
        pred = pred*norm
        growth = R*pred*(1-hazard)
        cp_mass = (R*pred*hazard).sum()
        Rn = np.zeros(max_run+1)
        Rn[1:] = growth[:-1]; Rn[0] = cp_mass
        s = Rn.sum(); Rn = Rn/s if s>0 else Rn
        # update sufficient stats (shift)
        mu_n = np.empty_like(mu); kap_n=np.empty_like(kap); al_n=np.empty_like(al); be_n=np.empty_like(be)
        mu_n[1:] = (kap*mu+xt)[:-1]/(kap[:-1]+1); kap_n[1:]=kap[:-1]+1
        al_n[1:] = al[:-1]+0.5
        be_n[1:] = be[:-1] + 0.5*kap[:-1]*(xt-mu[:-1])**2/(kap[:-1]+1)
        mu_n[0], kap_n[0], al_n[0], be_n[0] = mu0, kappa0, alpha0, beta0
        mu,kap,al,be,R = mu_n,kap_n,al_n,be_n,Rn
        mp = int(R.argmax())
        if mp < prev_map - 3 and mp <= 2:  # MAP collapse → changepoint
            cps[t] = 1
        prev_map = mp
    return cps

# stub module to avoid scipy import error
import sys, types
sys.modules['scipy_stub'] = types.ModuleType('scipy_stub')

# ── E-Divisive (energy-based binary segmentation, causal expanding) ──
def energy_stat(a, b):
    a=a.reshape(-1,1); b=b.reshape(-1,1)
    dab = np.abs(a-b.T).mean(); daa=np.abs(a-a.T).mean(); dbb=np.abs(b-b.T).mean()
    n,m = len(a), len(b)
    return (n*m/(n+m))*(2*dab-daa-dbb)

def ediv_last_cp(x, min_seg=24):
    """Return index of most recent significant changepoint via greedy max energy split."""
    T=len(x)
    if T < 2*min_seg: return 0
    best_q, best_i = -np.inf, 0
    for i in range(min_seg, T-min_seg):
        q = energy_stat(x[:i], x[i:])
        if q > best_q: best_q, best_i = q, i
    # significance via simple threshold on scaled stat (permutation-free approx)
    return best_i if best_q > 2.0 else 0

def run_cpd(method='bocpd'):
    Y, sig_paper, z_paper = load()
    dates = Y.index; K = Y.shape[1]; T = len(dates)
    is_end = np.where(dates<='2009-12-31')[0][-1]
    Yv = Y.values

    # base lambdas (IS, raw) — same as H-DCC
    base_lam = np.array([ewma_qmle_lambda(Yv[:is_end+1,k]) for k in range(K)])

    # detect changepoints causally; re-estimate lambda on post-CP segment yearly
    LAM = np.tile(base_lam,(T,1))
    if method=='bocpd':
        for k in range(K):
            cps = bocpd_changepoints(Yv[:,k])
            last_cp = 0
            for t in range(is_end+1, T):
                if cps[t-1]==1: last_cp = t-1
                seg = Yv[max(0,last_cp):t, k]
                if len(seg)>=36 and last_cp>0:
                    LAM[t,k] = ewma_qmle_lambda(seg, grid=60)
    else:  # e-divisive, re-run every 12 months
        for k in range(K):
            cp_idx = 0
            for t in range(is_end+1, T):
                if (t-is_end) % 12 == 1:
                    cp_idx = ediv_last_cp(Yv[:t,k])
                seg = Yv[cp_idx:t,k]
                if len(seg)>=36 and cp_idx>0:
                    LAM[t,k] = ewma_qmle_lambda(seg, grid=60)

    # sigma path with time-varying lambda
    SIG = np.zeros((T,K))
    for k in range(K):
        s2 = np.var(Yv[:12,k])
        for t in range(1,T):
            lam = LAM[t,k]
            s2 = lam*s2 + (1-lam)*Yv[t-1,k]**2
            SIG[t,k]=np.sqrt(max(s2,1e-12))

    # backtest (Stage 2 = paper z)
    Z = z_paper.values
    rets,tos=[],[]; w_prev=np.ones(K)/K
    for i in np.where(dates>='2010-01-01')[0]:
        s = SIG[i-1]
        R = np.corrcoef(Z[i-60:i],rowvar=False); R=(R+R.T)/2
        H = np.outer(s,s)*R + np.eye(K)*1e-10
        w = minvar_lo(H); to=np.abs(w-w_prev).sum()
        rets.append(w@Yv[i]-TC*to); tos.append(to); w_prev=w
    rp=np.array(rets)
    sh=rp.mean()*12/(rp.std(ddof=1)*np.sqrt(12))
    cum=np.cumprod(1+rp); mdd=((cum-np.maximum.accumulate(cum))/np.maximum.accumulate(cum)).min()
    return dict(method=method, sharpe=sh, maxdd=mdd, avg_to=np.mean(tos))

if __name__=='__main__':
    for m in ['bocpd','ediv']:
        r = run_cpd(m)
        print(f"{m:8s}: Sharpe={r['sharpe']:.3f} MaxDD={r['maxdd']*100:.1f}% TO={r['avg_to']*100:.1f}%")
