"""Mahalanobis diagnostics d_t = r_t' H_t^{-1} r_t for H-DCC / TVP-AR(1) / HMM(3-state).
Target E[d]=K=12; chi2(12) 95% bound = 21.03. Monthly (rolling-60) and Daily (rolling-252).
All H_t causal (info to t-1). r_t = raw returns (common across models)."""
import numpy as np, pandas as pd
from backtest_core import load, load_daily, ewma_qmle_vec
from tvp_kalman import kalman_tvp_ar1, ewma_qmle_lambda
from hmm3_lambda import fit_hmm, causal_filter

CHI95 = 21.026

def mahal_series(Yv, SIG, Z, dates, win, start):
    K = Yv.shape[1]
    out, idx = [], []
    start_i = max(win+1, np.searchsorted(dates, pd.Timestamp(start)))
    for i in range(start_i, len(dates)):
        s = SIG[i-1]
        R = np.corrcoef(Z[i-win:i], rowvar=False); R = (R+R.T)/2 + np.eye(K)*1e-8
        H = np.outer(s,s)*R
        try:
            d = Yv[i] @ np.linalg.solve(H, Yv[i])
        except np.linalg.LinAlgError:
            d = Yv[i] @ np.linalg.pinv(H) @ Yv[i]
        out.append(d); idx.append(dates[i])
    return pd.Series(out, index=pd.DatetimeIndex(idx))

def stats(d, a=None, b=None, label=''):
    if a: d = d[(d.index>=a)&(d.index<=b)]
    return dict(구간=label, 평균=round(d.mean(),2), 중앙값=round(d.median(),2),
                초과율_95пр=f"{(d>CHI95).mean()*100:.1f}%", n=len(d))

# ═══ MONTHLY ═══
def monthly_paths():
    Y, sig_p, z_p = load()
    dates = Y.index; K = Y.shape[1]; T = len(dates); Yv = Y.values
    is_end = np.where(dates<='2009-12-31')[0][-1]
    paths = {}
    # H-DCC: paper sigma (annualized %) and paper z (Ridge-based)
    SIG_h = sig_p.values/100/np.sqrt(12)
    paths['H-DCC'] = (SIG_h, z_p.values)
    # TVP
    EPS = np.column_stack([kalman_tvp_ar1(Yv[:,k],1e-5) for k in range(K)]); EPS[0]=Yv[0]
    lam_t = np.array([ewma_qmle_lambda(EPS[1:is_end+1,k][~np.isnan(EPS[1:is_end+1,k])],grid=50) for k in range(K)])
    SIG_t = np.zeros((T,K))
    for k in range(K):
        s2=np.nanvar(EPS[1:13,k])
        for t in range(1,T):
            x=EPS[t-1,k]
            if not np.isnan(x): s2=lam_t[k]*s2+(1-lam_t[k])*x*x
            SIG_t[t,k]=np.sqrt(max(s2,1e-12))
    paths['TVP-AR(1)'] = (SIG_t, EPS/np.where(SIG_t>0,SIG_t,1))
    # HMM 3-state
    LAM=np.zeros((T,K))
    for k in range(K):
        xis=Yv[:is_end+1,k]; prm=fit_hmm(xis,S=3)
        lab=prm['gamma'].argmax(1)
        lams=np.array([ewma_qmle_lambda(xis[lab==s],grid=50) if (lab==s).sum()>=30 else (0.95 if s==0 else 0.85) for s in range(3)])
        LAM[:,k]=causal_filter(Yv[:,k],prm)@lams
    SIG_m=np.zeros((T,K))
    for k in range(K):
        s2=np.var(Yv[:12,k])
        for t in range(1,T):
            s2=LAM[t,k]*s2+(1-LAM[t,k])*Yv[t-1,k]**2
            SIG_m[t,k]=np.sqrt(max(s2,1e-12))
    paths['HMM(3)'] = (SIG_m, Yv/np.where(SIG_m>0,SIG_m,1))
    return dates, Yv, paths

# ═══ DAILY ═══
def daily_paths():
    D = load_daily(); dates=D.index; K=D.shape[1]; T=len(dates); Yv=D.values
    is_end = np.searchsorted(dates, pd.Timestamp('2009-12-31'),'right')-1
    paths={}
    # H-DCC daily: raw, IS-fixed lambda
    lam_h = np.array([ewma_qmle_vec(Yv[1:is_end+1,k],0.90,0.999,40) for k in range(K)])
    def sig_path(EPS, lams):
        S=np.zeros((T,K))
        for k in range(K):
            s2=np.var(EPS[1:253,k])
            for t in range(1,T):
                s2=lams[t,k]*s2+(1-lams[t,k])*EPS[t-1,k]**2
                S[t,k]=np.sqrt(max(s2,1e-14))
        return S
    L = np.tile(lam_h,(T,1))
    S_h = sig_path(Yv, L)
    paths['H-DCC'] = (S_h, Yv/np.where(S_h>0,S_h,1))
    # TVP daily
    EPS = np.column_stack([kalman_tvp_ar1(Yv[:,k],1e-5) for k in range(K)]); EPS[0]=Yv[0]
    lam_t = np.array([ewma_qmle_vec(EPS[1:is_end+1,k],0.90,0.999,40) for k in range(K)])
    S_t = sig_path(EPS, np.tile(lam_t,(T,1)))
    paths['TVP-AR(1)'] = (S_t, EPS/np.where(S_t>0,S_t,1))
    # HMM(3) daily: EM on IS (subsampled iters), causal filter full
    LAM=np.zeros((T,K))
    for k in range(K):
        xis=Yv[:is_end+1,k]
        prm=fit_hmm(xis,S=3,n_iter=25)
        lab=prm['gamma'].argmax(1)
        lams=np.array([ewma_qmle_vec(xis[lab==s],0.90,0.999,30) if (lab==s).sum()>=200 else 0.97 for s in range(3)])
        LAM[:,k]=causal_filter(Yv[:,k],prm)@lams
    S_m=sig_path(Yv,LAM)
    paths['HMM(3)']=(S_m, Yv/np.where(S_m>0,S_m,1))
    return dates, Yv, paths

if __name__=='__main__':
    print("═══ 월간 (rolling-60, 1999~) ═══")
    dates,Yv,paths = monthly_paths()
    rowsM=[]
    for name,(S,Z) in paths.items():
        d = mahal_series(Yv,S,Z,dates,60,'1999-01-01')
        for lbl,a,b in [('전체 1999~2025',None,None),('IS측 1999~2009','1999-01-01','2009-12-31'),('OOS 2010~2025','2010-01-01','2025-12-31')]:
            st = stats(d,a,b,lbl); st['모형']=name; rowsM.append(st)
        print(f"{name:10s} 전체: mean={d.mean():.2f} med={d.median():.2f} exceed95={(d>21.03).mean()*100:.1f}% (target: 12 / 5%)")
    pd.DataFrame(rowsM).to_csv('/home/claude/hdcc/task4_mahal_monthly.csv',index=False)

    print("\n═══ 일간 (rolling-252, 1999~) ═══")
    dates,Yv,paths = daily_paths()
    rowsD=[]
    for name,(S,Z) in paths.items():
        d = mahal_series(Yv,S,Z,dates,252,'1999-01-01')
        for lbl,a,b in [('전체 1999~2025',None,None),('OOS 2010~2025','2010-01-01','2025-12-31')]:
            st = stats(d,a,b,lbl); st['모형']=name; rowsD.append(st)
        print(f"{name:10s} 전체: mean={d.mean():.2f} med={d.median():.2f} exceed95={(d>21.03).mean()*100:.1f}%")
    pd.DataFrame(rowsD).to_csv('/home/claude/hdcc/task4_mahal_daily.csv',index=False)
