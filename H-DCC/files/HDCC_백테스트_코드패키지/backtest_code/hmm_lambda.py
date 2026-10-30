"""HMM regime-switching lambda — unified full-sample protocol.
2-state Gaussian HMM per asset on monthly returns (variance regimes).
EM fitted on IS (1994-2009) only; OOS regime probs via causal forward filtering.
lambda_s estimated by QMLE on IS months classified to each regime (gamma>0.5).
sigma update uses regime-prob-weighted lambda_t = P_low*lam_low + P_high*lam_high.
Stage 2: rolling-60 corr of standardized residuals (same as H-DCC protocol).
"""
import pandas as pd, numpy as np
from backtest_core import load, minvar_lo, TC
from tvp_kalman import ewma_qmle_lambda

def fit_hmm_2state(x, n_iter=60):
    """EM for 2-state Gaussian HMM (means free, variances free). Returns params."""
    x = np.asarray(x, float)
    T = len(x)
    # init: split by abs deviation
    med = np.median(np.abs(x - x.mean()))
    lab = (np.abs(x - x.mean()) > 1.5*med).astype(int)
    mu = np.array([x[lab==0].mean(), x[lab==1].mean() if (lab==1).any() else x.mean()])
    var = np.array([max(x[lab==0].var(),1e-8), max(x[lab==1].var() if (lab==1).any() else x.var()*4,1e-8)])
    A = np.array([[0.95,0.05],[0.10,0.90]])
    pi = np.array([0.8,0.2])
    for _ in range(n_iter):
        # E-step (forward-backward, scaled)
        B = np.column_stack([np.exp(-0.5*(x-mu[s])**2/var[s])/np.sqrt(2*np.pi*var[s]) for s in range(2)])
        B = np.clip(B, 1e-300, None)
        alpha = np.zeros((T,2)); c = np.zeros(T)
        alpha[0] = pi*B[0]; c[0]=alpha[0].sum(); alpha[0]/=c[0]
        for t in range(1,T):
            alpha[t] = (alpha[t-1]@A)*B[t]; c[t]=alpha[t].sum(); alpha[t]/=c[t]
        beta = np.zeros((T,2)); beta[-1]=1
        for t in range(T-2,-1,-1):
            beta[t] = (A@(B[t+1]*beta[t+1]))/c[t+1]
        gam = alpha*beta; gam/= gam.sum(1,keepdims=True)
        xi_num = np.zeros((2,2))
        for t in range(T-1):
            m = np.outer(alpha[t], B[t+1]*beta[t+1])*A/c[t+1]
            xi_num += m
        # M-step
        A = xi_num/np.clip(gam[:-1].sum(0)[:,None],1e-12,None)
        A = A/np.clip(A.sum(1,keepdims=True),1e-12,None)
        pi = gam[0]
        for s in range(2):
            w = gam[:,s]; W=w.sum()
            mu[s] = (w@x)/W
            var[s] = max((w@(x-mu[s])**2)/W, 1e-10)
        # keep state 1 = high variance
        if var[0] > var[1]:
            mu=mu[::-1]; var=var[::-1]; A=A[::-1,::-1]; pi=pi[::-1]; gam=gam[:,::-1]
    return dict(mu=mu, var=var, A=A, pi=pi, gamma=gam)

def causal_filter(x, prm):
    """Forward-only filtered P(state) — no look-ahead."""
    mu,var,A,pi = prm['mu'],prm['var'],prm['A'],prm['pi']
    T=len(x); P=np.zeros((T,2))
    B = np.column_stack([np.exp(-0.5*(x-mu[s])**2/var[s])/np.sqrt(2*np.pi*var[s]) for s in range(2)])
    B = np.clip(B,1e-300,None)
    a = pi*B[0]; a/=a.sum(); P[0]=a
    for t in range(1,T):
        a = (a@A)*B[t]; a/=a.sum(); P[t]=a
    return P

def run_hmm():
    Y,_,zp = load()
    dates=Y.index; K=Y.shape[1]; T=len(dates); Yv=Y.values
    is_end = np.where(dates<='2009-12-31')[0][-1]

    LAM = np.zeros((T,K))
    for k in range(K):
        xis = Yv[:is_end+1,k]
        prm = fit_hmm_2state(xis)
        # regime-specific lambda on IS
        g = prm['gamma']
        lo_idx = g[:,0]>0.5; hi_idx=~lo_idx
        lam_lo = ewma_qmle_lambda(xis[lo_idx], grid=50) if lo_idx.sum()>=36 else 0.95
        lam_hi = ewma_qmle_lambda(xis[hi_idx], grid=50) if hi_idx.sum()>=36 else 0.85
        # causal filtering over full sample with IS-fitted params
        P = causal_filter(Yv[:,k], prm)
        LAM[:,k] = P[:,0]*lam_lo + P[:,1]*lam_hi

    SIG=np.zeros((T,K))
    for k in range(K):
        s2=np.var(Yv[:12,k])
        for t in range(1,T):
            s2 = LAM[t,k]*s2 + (1-LAM[t,k])*Yv[t-1,k]**2
            SIG[t,k]=np.sqrt(max(s2,1e-12))
    Z = Yv/np.where(SIG>0,SIG,1)

    rets=[]; w_prev=np.ones(K)/K
    for i in np.where(dates>='2010-01-01')[0]:
        s=SIG[i-1]
        R=np.corrcoef(Z[i-60:i],rowvar=False); R=(R+R.T)/2
        H=np.outer(s,s)*R+np.eye(K)*1e-10
        w=minvar_lo(H); to=np.abs(w-w_prev).sum()
        rets.append(w@Yv[i]-TC*to); w_prev=w
    rp=np.array(rets)
    sh=rp.mean()*12/(rp.std(ddof=1)*np.sqrt(12))
    cum=np.cumprod(1+rp); mdd=((cum-np.maximum.accumulate(cum))/np.maximum.accumulate(cum)).min()
    return sh, mdd

if __name__=='__main__':
    sh,mdd = run_hmm()
    print(f"HMM(2-state, unified): Sharpe={sh:.3f}  MaxDD={mdd*100:.1f}%")
