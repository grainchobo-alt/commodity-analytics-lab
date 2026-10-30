"""3-state Gaussian HMM regime lambda — unified protocol (monthly).
States sorted by variance: low / mid / high. lambda_s per regime (QMLE on IS months
classified by smoothed gamma argmax). OOS: causal forward filtering, prob-weighted lambda."""
import numpy as np, pandas as pd
from backtest_core import load, minvar_lo, TC
from tvp_kalman import ewma_qmle_lambda

def fit_hmm(x, S=3, n_iter=80, seed=0):
    x = np.asarray(x, float); T = len(x)
    rng = np.random.default_rng(seed)
    # init by quantiles of |x - mean|
    dev = np.abs(x - x.mean())
    qs = np.quantile(dev, np.linspace(0, 1, S+1))
    lab = np.clip(np.searchsorted(qs, dev, 'right')-1, 0, S-1)
    mu = np.array([x[lab==s].mean() if (lab==s).any() else x.mean() for s in range(S)])
    var = np.array([max(x[lab==s].var(), 1e-8) if (lab==s).sum()>3 else x.var() for s in range(S)])
    A = np.full((S,S), 0.05/(S-1)) + np.eye(S)*(0.95-0.05/(S-1))
    A = A/A.sum(1, keepdims=True)
    pi = np.full(S, 1/S)
    for _ in range(n_iter):
        B = np.column_stack([np.exp(-0.5*(x-mu[s])**2/var[s])/np.sqrt(2*np.pi*var[s]) for s in range(S)])
        B = np.clip(B, 1e-300, None)
        alpha = np.zeros((T,S)); c = np.zeros(T)
        alpha[0] = pi*B[0]; c[0]=alpha[0].sum(); alpha[0]/=c[0]
        for t in range(1,T):
            alpha[t] = (alpha[t-1]@A)*B[t]; c[t]=alpha[t].sum(); alpha[t]/=c[t]
        beta = np.zeros((T,S)); beta[-1]=1
        for t in range(T-2,-1,-1):
            beta[t] = (A@(B[t+1]*beta[t+1]))/c[t+1]
        gam = alpha*beta; gam /= gam.sum(1,keepdims=True)
        xi = np.zeros((S,S))
        for t in range(T-1):
            xi += np.outer(alpha[t], B[t+1]*beta[t+1])*A/c[t+1]
        A = xi/np.clip(gam[:-1].sum(0)[:,None],1e-12,None); A=A/A.sum(1,keepdims=True)
        pi = gam[0]
        for s in range(S):
            w=gam[:,s]; W=max(w.sum(),1e-12)
            mu[s]=(w@x)/W; var[s]=max((w@(x-mu[s])**2)/W,1e-10)
        order = np.argsort(var)
        mu,var = mu[order],var[order]; A=A[np.ix_(order,order)]; pi=pi[order]; gam=gam[:,order]
    ll = np.log(c).sum()
    return dict(mu=mu,var=var,A=A,pi=pi,gamma=gam,ll=ll,S=S)

def causal_filter(x, prm):
    mu,var,A,pi,S = prm['mu'],prm['var'],prm['A'],prm['pi'],prm['S']
    T=len(x); P=np.zeros((T,S))
    B = np.column_stack([np.exp(-0.5*(x-mu[s])**2/var[s])/np.sqrt(2*np.pi*var[s]) for s in range(S)])
    B=np.clip(B,1e-300,None)
    a=pi*B[0]; a/=a.sum(); P[0]=a
    for t in range(1,T):
        a=(a@A)*B[t]; a/=a.sum(); P[t]=a
    return P

def run_hmm_S(S=3):
    Y,_,_ = load()
    dates=Y.index; K=Y.shape[1]; T=len(dates); Yv=Y.values
    is_end = np.where(dates<='2009-12-31')[0][-1]
    LAM=np.zeros((T,K)); regime_info=[]
    for k in range(K):
        xis = Yv[:is_end+1,k]
        prm = fit_hmm(xis, S=S)
        g = prm['gamma']; lab = g.argmax(1)
        lams=[]
        for s in range(S):
            seg = xis[lab==s]
            lams.append(ewma_qmle_lambda(seg, grid=50) if len(seg)>=30 else (0.95 if s==0 else 0.85))
        lams=np.array(lams)
        P = causal_filter(Yv[:,k], prm)
        LAM[:,k] = P@lams
        regime_info.append((lams, (lab[:,None]==np.arange(S)).mean(0)))
    SIG=np.zeros((T,K))
    for k in range(K):
        s2=np.var(Yv[:12,k])
        for t in range(1,T):
            s2=LAM[t,k]*s2+(1-LAM[t,k])*Yv[t-1,k]**2
            SIG[t,k]=np.sqrt(max(s2,1e-12))
    Z=Yv/np.where(SIG>0,SIG,1)
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
    return sh, mdd, regime_info, SIG, Z, LAM

if __name__=='__main__':
    Y,_,_ = load(); assets=list(Y.columns)
    for S in [2,3]:
        sh,mdd,ri,_,_,_ = run_hmm_S(S)
        print(f"HMM {S}-state: Sharpe={sh:.3f}  MaxDD={mdd*100:.1f}%")
        if S==3:
            print("  자산별 레짐 λ (저/중/고변동) 및 IS 점유율:")
            for a,(lams,occ) in zip(assets,ri):
                print(f"    {a:9s}: λ={np.round(lams,3)}  점유율={np.round(occ*100,0).astype(int)}%")
