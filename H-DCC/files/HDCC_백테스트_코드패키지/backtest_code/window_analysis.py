"""Analysis 2: lambda calibration window length. OOS fixed 2010-2025.
Windows: trailing 3y/5y/7y/10y, expanding(annual re-est), IS-fixed(94-09).
Models: H-DCC (raw innovations) and TVP-AR(1) (Kalman innovations, causal).
"""
import pandas as pd, numpy as np
from backtest_core import load, minvar_lo, TC
from tvp_kalman import kalman_tvp_ar1, ewma_qmle_lambda

def run(model, window):  # window: 36/60/84/120 months, 'exp', 'isfix'
    Y,_,_ = load()
    dates=Y.index; K=Y.shape[1]; T=len(dates); Yv=Y.values
    if model=='tvp':
        EPS = np.column_stack([kalman_tvp_ar1(Yv[:,k],1e-5) for k in range(K)])
        EPS[0]=Yv[0]
    else:
        EPS = Yv.copy()
    is_end = np.where(dates<='2009-12-31')[0][-1]

    LAM = np.zeros((T,K))
    if window=='isfix':
        lam0 = np.array([ewma_qmle_lambda(EPS[1:is_end+1,k][~np.isnan(EPS[1:is_end+1,k])],grid=50) for k in range(K)])
        LAM[:] = lam0
    else:
        cur = np.full(K,0.94)
        for t in range(T):
            if dates[t].month==1 and t>=48:
                for k in range(K):
                    lo = 1 if window=='exp' else max(1, t-window)
                    e = EPS[lo:t,k]; e=e[~np.isnan(e)]
                    if len(e)>=30: cur[k]=ewma_qmle_lambda(e,grid=50)
            LAM[t]=cur

    SIG=np.zeros((T,K))
    for k in range(K):
        s2=np.nanvar(EPS[1:13,k])
        for t in range(1,T):
            x=EPS[t-1,k]
            if not np.isnan(x): s2=LAM[t,k]*s2+(1-LAM[t,k])*x*x
            SIG[t,k]=np.sqrt(max(s2,1e-12))
    Z=np.where(SIG>0,EPS/np.where(SIG>0,SIG,1),0.0)

    rets=[]; w_prev=np.ones(K)/K
    for i in np.where(dates>='2010-01-01')[0]:
        s=SIG[i-1]
        R=np.corrcoef(Z[i-60:i],rowvar=False); R=(R+R.T)/2
        H=np.outer(s,s)*R+np.eye(K)*1e-10
        w=minvar_lo(H); to=np.abs(w-w_prev).sum()
        rets.append(w@Yv[i]-TC*to); w_prev=w
    rp=np.array(rets)
    return rp.mean()*12/(rp.std(ddof=1)*np.sqrt(12))

if __name__=='__main__':
    rows=[]
    for wname,w in [('3년(36m)',36),('5년(60m)',60),('7년(84m)',84),('10년(120m)',120),
                    ('확장(expanding)','exp'),('IS고정(94-09)','isfix')]:
        sh_h = run('hdcc',w); sh_t = run('tvp',w)
        rows.append({'λ 추정 윈도우':wname,'H-DCC SR':round(sh_h,3),'TVP SR':round(sh_t,3),
                     'Δ(TVP−H)':round(sh_t-sh_h,3)})
        print(rows[-1])
    pd.DataFrame(rows).to_csv('/home/claude/hdcc/window_results.csv',index=False)
