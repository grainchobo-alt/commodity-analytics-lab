"""Benchmark models with grid-vectorized fitting + causal H-path builders.
Models: HDCC(asset-lambda EWMA + rolling corr), POOL(single lambda + rolling corr),
RM(RiskMetrics EWMA cov), CCC, DCC, ADCC, BEKK(scalar, cov targeting).
All walk-forward: annual Jan refits on expanding (or rolling) training windows.
H_path[t] = one-step-ahead covariance forecast E[r_t r_t' | info_{t-1}].
"""
import numpy as np, pandas as pd
from backtest_core import load, load_daily, minvar_lo, TC

# ───────── vectorized univariate fits ─────────
def ewma_lams_ll(E, lo, hi, G=50):
    """Per-asset EWMA QMLE. E:(T,K). Returns lams(K,), ll_min(K,) on -2LL scale (sans const)."""
    T,K = E.shape
    lams = np.linspace(lo,hi,G)
    s2 = np.tile(np.var(E[:min(60,T)],axis=0),(G,1)); s2=np.clip(s2,1e-14,None)
    ll = np.zeros((G,K)); E2=E**2
    for t in range(1,T):
        s2 = lams[:,None]*s2 + (1-lams[:,None])*E2[t-1]
        np.clip(s2,1e-14,None,out=s2)
        ll += np.log(s2) + E2[t]/s2
    gi = ll.argmin(0)
    return lams[gi], ll[gi,np.arange(K)], ll  # also full grid ll for pooled

def pooled_lambda(E, lo, hi, G=50):
    _,_,llg = ewma_lams_ll(E, lo, hi, G)
    tot = llg.sum(1)
    gi = tot.argmin()
    return np.linspace(lo,hi,G)[gi], tot[gi]

def garch_fit_vec(E, n_a=10, n_b=12):
    """GARCH(1,1) grid QMLE with variance targeting. Returns alpha(K,), beta(K,)."""
    T,K = E.shape
    al = np.linspace(0.02,0.30,n_a); be = np.linspace(0.50,0.97,n_b)
    A,B = np.meshgrid(al,be,indexing='ij'); A=A.ravel(); B=B.ravel()
    ok = (A+B)<0.999; A,B = A[ok],B[ok]; G=len(A)
    vbar = np.var(E,axis=0)
    om = (1-A[:,None]-B[:,None])*vbar[None,:]   # (G,K)
    h = np.tile(vbar,(G,1)); ll = np.zeros((G,K)); E2=E**2
    for t in range(1,T):
        h = om + A[:,None]*E2[t-1] + B[:,None]*h
        np.clip(h,1e-14,None,out=h)
        ll += np.log(h) + E2[t]/h
    gi = ll.argmin(0)
    return A[gi], B[gi]

def garch_h_path(E, alpha, beta, vbar):
    """one-step-ahead h_{t|t-1} path. h[0]=vbar."""
    T,K = E.shape
    h = np.zeros((T,K)); h[0]=vbar
    om = (1-alpha-beta)*vbar
    for t in range(1,T):
        h[t] = om + alpha*E[t-1]**2 + beta*h[t-1]
        np.clip(h[t],1e-14,None,out=h[t])
    return h

# ───────── correlation models (grid over (a,b) [,g]) ─────────
def dcc_fit(Z, n_a=8, n_b=10, asym=False, n_g=4):
    """Z:(T,K) standardized residuals. Grid MLE. Returns (a,b[,g])."""
    T,K = Z.shape
    Qb = np.corrcoef(Z,rowvar=False); Qb=(Qb+Qb.T)/2
    a_ = np.linspace(0.0,0.12,n_a); b_=np.linspace(0.50,0.99,n_b)
    if asym:
        g_ = np.linspace(0.0,0.06,n_g)
        AA,BB,GG = np.meshgrid(a_,b_,g_,indexing='ij'); AA=AA.ravel();BB=BB.ravel();GG=GG.ravel()
        N = np.minimum(Z,0); Nb = (N.T@N)/T
        ok = (AA+BB+0.5*GG)<0.999
    else:
        AA,BB = np.meshgrid(a_,b_,indexing='ij'); AA=AA.ravel();BB=BB.ravel(); GG=np.zeros_like(AA)
        Nb = np.zeros_like(Qb); ok = (AA+BB)<0.999
    AA,BB,GG = AA[ok],BB[ok],GG[ok]; G=len(AA)
    Q = np.tile(Qb,(G,1,1))
    ll = np.zeros(G)
    const = Qb - AA[:,None,None]*Qb - BB[:,None,None]*Qb - GG[:,None,None]*Nb
    for t in range(1,T):
        z = Z[t-1]
        zz = np.outer(z,z)
        nn = np.outer(np.minimum(z,0),np.minimum(z,0))
        Q = const + AA[:,None,None]*zz + BB[:,None,None]*Q + GG[:,None,None]*nn
        d = np.sqrt(np.clip(np.einsum('gii->gi',Q),1e-12,None))
        R = Q/(d[:,:,None]*d[:,None,:])
        zt = Z[t]
        sign,logdet = np.linalg.slogdet(R)
        sol = np.linalg.solve(R, np.tile(zt,(G,1))[:,:,None])[:,:,0]
        ll += logdet + (sol*zt).sum(1)
    gi = ll.argmin()
    return (AA[gi],BB[gi],GG[gi]), Qb, Nb

def dcc_R_path(Z, params, Qb, Nb):
    a,b,g = params
    T,K = Z.shape
    Rp = np.zeros((T,K,K)); Q = Qb.copy()
    Rp[0]=Qb
    const = Qb - a*Qb - b*Qb - g*Nb
    for t in range(1,T):
        z=Z[t-1]; n=np.minimum(z,0)
        Q = const + a*np.outer(z,z) + b*Q + g*np.outer(n,n)
        d = np.sqrt(np.clip(np.diag(Q),1e-12,None))
        Rp[t] = Q/np.outer(d,d)
    return Rp

def bekk_fit(E, n_a=8, n_b=10):
    """Scalar BEKK w/ cov targeting on RETURNS: H=(1-a-b)S + a rr' + b H."""
    T,K = E.shape
    S = np.cov(E,rowvar=False)
    a_ = np.linspace(0.005,0.15,n_a); b_=np.linspace(0.50,0.99,n_b)
    AA,BB = np.meshgrid(a_,b_,indexing='ij'); AA=AA.ravel();BB=BB.ravel()
    ok=(AA+BB)<0.999; AA,BB=AA[ok],BB[ok]; G=len(AA)
    H = np.tile(S,(G,1,1)); ll=np.zeros(G)
    const = (1-AA-BB)[:,None,None]*S
    for t in range(1,T):
        rr = np.outer(E[t-1],E[t-1])
        H = const + AA[:,None,None]*rr + BB[:,None,None]*H
        sign,logdet = np.linalg.slogdet(H)
        sol = np.linalg.solve(H, np.tile(E[t],(G,1))[:,:,None])[:,:,0]
        ll += logdet + (sol*E[t]).sum(1)
    gi=ll.argmin()
    return AA[gi],BB[gi],S

def bekk_H_path(E, a, b, S):
    T,K=E.shape
    Hp=np.zeros((T,K,K)); H=S.copy(); Hp[0]=S
    const=(1-a-b)*S
    for t in range(1,T):
        H = const + a*np.outer(E[t-1],E[t-1]) + b*H
        Hp[t]=H
    return Hp

def rm_H_path(E, lam):
    T,K=E.shape
    Hp=np.zeros((T,K,K)); H=np.cov(E[:min(60,T)],rowvar=False); Hp[0]=H
    for t in range(1,T):
        H = lam*H + (1-lam)*np.outer(E[t-1],E[t-1])
        Hp[t]=H
    return Hp

def ewma_sig_path(E, lams, seed_n):
    T,K=E.shape
    S=np.zeros((T,K))
    s2=np.var(E[1:seed_n],axis=0)
    S[0]=np.sqrt(s2)
    for t in range(1,T):
        s2 = lams*s2 + (1-lams)*E[t-1]**2
        S[t]=np.sqrt(np.clip(s2,1e-14,None))
    return S
