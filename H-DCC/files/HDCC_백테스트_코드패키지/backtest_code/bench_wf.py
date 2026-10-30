"""Walk-forward H-path orchestrator for 7 benchmark models, monthly & daily."""
import numpy as np, pandas as pd, time
from backtest_core import load, load_daily, minvar_lo, TC
from bench_core import (ewma_lams_ll, pooled_lambda, garch_fit_vec, garch_h_path,
                        dcc_fit, dcc_R_path, bekk_fit, bekk_H_path, rm_H_path, ewma_sig_path)
from daily_engine import month_end_idx, sharpe_d, ERAS

def jan_refits(dates, min_t):
    idx=[]; seen=set()
    for t in range(len(dates)):
        y=dates[t].year
        if y not in seen:
            seen.add(y)
            if t>=min_t: idx.append(t)
    return idx

def build_paths(model, X, dates, freq, win_mode='exp', refit_from=None):
    """Returns Hp (T,K,K) one-step-ahead causal covariance forecasts.
    freq: 'm'|'d'. win_mode: 'exp' | int(trailing obs) | 'isfix'."""
    T,K = X.shape
    if freq=='m':
        lo,hi,G = 0.79,0.986,50; corr_win=60; min_t=48; seed=13; rm_lam=0.97
    else:
        lo,hi,G = 0.90,0.999,40; corr_win=252; min_t=1008; seed=253; rm_lam=0.94
    Hp = np.zeros((T,K,K))
    if model=='RM':
        return rm_H_path(X, rm_lam)
    if win_mode=='isfix':
        is_end = np.searchsorted(dates, pd.Timestamp('2009-12-31'),'right')
        refits=[is_end]
    else:
        refits = jan_refits(dates, min_t)
        if refit_from is not None:
            refits = [t for t in refits if dates[t].year >= refit_from]
    for j,t0 in enumerate(refits):
        w0 = 0 if win_mode in ('exp','isfix') else max(0, t0-win_mode)
        Xf = X[w0:t0]
        if len(Xf) < (36 if freq=='m' else 600): continue
        seg_end = refits[j+1] if (j+1<len(refits) and win_mode!='isfix') else T
        if model in ('HDCC','POOL'):
            if model=='HDCC':
                lams,_,_ = ewma_lams_ll(Xf, lo, hi, G)
            else:
                lp,_ = pooled_lambda(Xf, lo, hi, G); lams = np.full(K, lp)
            SIG = ewma_sig_path(X, lams, seed)
            Z = X/np.clip(SIG,1e-12,None)
            for t in range(max(t0,corr_win+1), seg_end):
                R = np.corrcoef(Z[t-corr_win:t],rowvar=False); R=(R+R.T)/2
                s = SIG[t]
                Hp[t] = np.outer(s,s)*R
        elif model in ('CCC','DCC','ADCC'):
            a,b = garch_fit_vec(Xf)
            vbar = np.var(Xf,axis=0)
            h_full = garch_h_path(X, a, b, vbar)
            Zf = Xf/np.sqrt(np.clip(garch_h_path(Xf,a,b,vbar),1e-14,None))
            Z_full = X/np.sqrt(np.clip(h_full,1e-14,None))
            D = np.sqrt(h_full)
            if model=='CCC':
                Rb = np.corrcoef(Zf,rowvar=False); Rb=(Rb+Rb.T)/2
                for t in range(t0, seg_end):
                    Hp[t] = np.outer(D[t],D[t])*Rb
            else:
                prm,Qb,Nb = dcc_fit(Zf, n_a=6, n_b=8, asym=(model=='ADCC'), n_g=3)
                Rp = dcc_R_path(Z_full, prm, Qb, Nb)
                for t in range(t0, seg_end):
                    Hp[t] = np.outer(D[t],D[t])*Rp[t]
        elif model=='BEKK':
            pa,pb,S = bekk_fit(Xf)
            Hf = bekk_H_path(X, pa, pb, S)
            Hp[t0:seg_end] = Hf[t0:seg_end]
        if win_mode=='isfix': break
    return Hp

def portfolio_from_H(Hp, X, dates, freq, start):
    K = X.shape[1]
    if freq=='m':
        rets,idxs=[],[]; w_prev=np.ones(K)/K
        for i in np.where(dates>=start)[0]:
            H = Hp[i]
            if not np.any(H): continue
            H = H + np.eye(K)*1e-10
            w = minvar_lo(H)
            to = np.abs(w-w_prev).sum()
            rets.append(w@X[i]-TC*to); idxs.append(dates[i]); w_prev=w
        return pd.Series(rets,index=pd.DatetimeIndex(idxs))
    else:
        mes = month_end_idx(dates)
        mes=[m for m in mes if dates[m]>=pd.Timestamp(start)-pd.offsets.MonthEnd(1)]
        w_prev=np.ones(K)/K; rr,dd=[],[]
        for j in range(len(mes)-1):
            t0,t1=mes[j],mes[j+1]
            if t0+1>=len(X): break
            H=Hp[t0+1]
            if not np.any(H): continue
            H=H+np.eye(K)*1e-12
            w=minvar_lo(H); to=np.abs(w-w_prev).sum()
            seg=X[t0+1:t1+1]@w
            if len(seg)==0: continue
            seg=seg.copy(); seg[0]-=TC*to
            rr.extend(seg.tolist()); dd.extend(dates[t0+1:t1+1].tolist()); w_prev=w
        rp=pd.Series(rr,index=pd.DatetimeIndex(dd))
        return rp[rp.index>=start]

def qlike_from_H(Hp, X, dates, start):
    K=X.shape[1]; out=[];idxs=[]
    for i in np.where(dates>=start)[0]:
        H=Hp[i]
        if not np.any(H): continue
        H=H+np.eye(K)*1e-10
        sign,ld = np.linalg.slogdet(H)
        try: q = X[i]@np.linalg.solve(H,X[i])
        except: q = X[i]@np.linalg.pinv(H)@X[i]
        out.append(ld+q); idxs.append(dates[i])
    return pd.Series(out,index=pd.DatetimeIndex(idxs))

MODELS = ['HDCC','POOL','RM','CCC','DCC','ADCC','BEKK']
