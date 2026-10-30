"""Asset-vectorized 3-state Gaussian HMM: walk-forward era analysis + window-length analysis,
monthly & daily. Causal: annual refits use only data up to refit date."""
import numpy as np, pandas as pd
from backtest_core import load, load_daily, minvar_lo, TC, ewma_qmle_vec

S = 3

def em_fit_vec(X, n_iter=40):
    """X: (T,K). Returns mu,var,A,pi each per-asset. Vectorized across assets."""
    T, K = X.shape
    dev = np.abs(X - X.mean(0))
    q1, q2 = np.quantile(dev, [1/3, 2/3], axis=0)
    lab = (dev > q1).astype(int) + (dev > q2).astype(int)  # (T,K) 0/1/2
    mu = np.zeros((K,S)); var = np.zeros((K,S))
    for s in range(S):
        m = (lab == s)
        cnt = np.clip(m.sum(0), 1, None)
        mu[:,s] = (X*m).sum(0)/cnt
        var[:,s] = np.clip(((X-mu[:,s])**2*m).sum(0)/cnt, 1e-10, None)
    A = np.tile(np.full((S,S), 0.05/(S-1)) + np.eye(S)*(0.95-0.05/(S-1)), (K,1,1))
    pi = np.full((K,S), 1/S)
    for _ in range(n_iter):
        B = np.exp(-0.5*(X[:,:,None]-mu[None])**2/var[None]) / np.sqrt(2*np.pi*var[None])
        B = np.clip(B, 1e-300, None)  # (T,K,S)
        alpha = np.zeros((T,K,S)); c = np.zeros((T,K))
        a = pi*B[0]; c[0] = a.sum(1); alpha[0] = a/c[0][:,None]
        for t in range(1,T):
            a = np.einsum('ks,kst->kt', alpha[t-1], A) * B[t]
            c[t] = a.sum(1); alpha[t] = a/np.clip(c[t][:,None],1e-300,None)
        beta = np.zeros((T,K,S)); beta[-1] = 1
        xi = np.zeros((K,S,S))
        for t in range(T-2,-1,-1):
            bb = B[t+1]*beta[t+1]
            beta[t] = np.einsum('kst,kt->ks', A, bb)/np.clip(c[t+1][:,None],1e-300,None)
            xi += alpha[t][:,:,None]*A*bb[:,None,:]/np.clip(c[t+1][:,None,None],1e-300,None)
        gam = alpha*beta
        gam /= np.clip(gam.sum(2,keepdims=True),1e-300,None)
        A = xi/np.clip(gam[:-1].sum(0)[:,:,None],1e-12,None)
        A /= A.sum(2,keepdims=True)
        pi = gam[0]
        gs = gam.sum(0)  # (K,S)
        mu = np.einsum('tks,tk->ks', gam, X)/np.clip(gs,1e-12,None)
        var = np.clip(np.einsum('tks,tks->ks', gam, (X[:,:,None]-mu[None])**2)/np.clip(gs,1e-12,None), 1e-10, None)
        order = np.argsort(var, axis=1)
        for k in range(K):
            o = order[k]
            mu[k]=mu[k,o]; var[k]=var[k,o]; A[k]=A[k][np.ix_(o,o)]; pi[k]=pi[k,o]
        # reorder gam too for next pass consistency (only needed at end)
    # final gamma with ordered params
    return mu, var, A, pi, gam, order

def causal_filter_vec(X, mu, var, A, pi):
    T, K = X.shape
    P = np.zeros((T,K,S))
    B0 = np.exp(-0.5*(X[0][:,None]-mu)**2/var)/np.sqrt(2*np.pi*var)
    a = pi*np.clip(B0,1e-300,None); a/=a.sum(1,keepdims=True); P[0]=a
    for t in range(1,T):
        B = np.exp(-0.5*(X[t][:,None]-mu)**2/var)/np.sqrt(2*np.pi*var)
        a = np.einsum('ks,kst->kt', a, A)*np.clip(B,1e-300,None)
        a /= np.clip(a.sum(1,keepdims=True),1e-300,None)
        P[t]=a
    return P

def regime_lambdas(Xfit, gam, lo, hi, grid):
    """lambda_s per asset from fit-window classification."""
    T, K = Xfit.shape
    lab = gam.argmax(2)  # (T,K)
    lams = np.zeros((K,S))
    for k in range(K):
        for s in range(S):
            seg = Xfit[lab[:,k]==s, k]
            lams[k,s] = ewma_qmle_vec(seg, lo, hi, grid) if len(seg) >= (30 if T<500 else 200) else (lo+hi)/2
    return lams

def hmm_lambda_path(X, dates, refit_idx, win_mode, lo, hi, grid, n_iter):
    """LAM (T,K): at each refit point, fit on window, λ_s, causal filter forward."""
    T, K = X.shape
    LAM = np.full((T,K), (lo+hi)/2)
    for j, t0 in enumerate(refit_idx):
        if win_mode == 'isfix':
            w0, w1 = 0, t0
        elif win_mode == 'exp':
            w0, w1 = 0, t0
        else:
            w0, w1 = max(0, t0-win_mode), t0
        Xf = X[w0:w1]
        min_len = 36 if T < 1000 else 600
        if len(Xf) < min_len:
            continue
        mu,var,A,pi,gam,_ = em_fit_vec(Xf, n_iter)
        lams = regime_lambdas(Xf, gam, lo, hi, grid)
        P = causal_filter_vec(X[:t0+ (refit_idx[j+1]-t0 if j+1<len(refit_idx) else T-t0)], mu,var,A,pi) if False else causal_filter_vec(X, mu,var,A,pi)
        seg_end = refit_idx[j+1] if j+1 < len(refit_idx) else T
        LAM[t0:seg_end] = np.einsum('tks,ks->tk', P[t0:seg_end], lams)
        if win_mode == 'isfix':
            LAM[:] = np.einsum('tks,ks->tk', P, lams)
            break
    return LAM

def sigma_z(X, LAM, seed_n):
    T,K = X.shape
    SIG = np.zeros((T,K))
    for k in range(K):
        s2 = np.var(X[1:seed_n,k])
        for t in range(1,T):
            s2 = LAM[t,k]*s2 + (1-LAM[t,k])*X[t-1,k]**2
            SIG[t,k] = np.sqrt(max(s2,1e-14))
    return SIG, X/np.where(SIG>0,SIG,1)
