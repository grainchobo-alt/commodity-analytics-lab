"""H-DCC-X: multiplicative macro long-run component.
sigma2_k,t = tau_k,t * g_k,t ; g = EWMA(lambda_k) on raw returns (paper Stage 1)
tau_k,t = exp(gamma_k' F_{t-lag}) / IS-mean-normalized ; gamma_k by OLS of log(r2/g) on lagged factors.
Variants: X2 (USD,VIX; full sample 1994~), X5 (+dFF,dCPI,dCH-leading; 2004~).
Backtest identical to paper: z=r/sigma, rolling-60 corr, MinVar OOS 2010-25, TC 7.5bp.
"""
import numpy as np, pandas as pd
from backtest_core import load, minvar_lo, TC
from bench_core import ewma_lams_ll, pooled_lambda, ewma_sig_path

M='/home/claude/hdcc/HDCC_Master_Data.xlsx'

def build_factors():
    aux = pd.read_excel(M, sheet_name='03_Daily_Aux_3', index_col=0)
    aux.index = pd.to_datetime(aux.index)
    me = aux.resample('ME').last()
    usd = -np.log(me['EuroUS Dollar FX Spot'].replace(0,np.nan)).diff()*100   # +: dollar strength
    vix = np.log(me['CBOE Market Volatility Index'].replace(0,np.nan)).diff()*100
    m4 = pd.read_excel(M, sheet_name='04_Macro_146_Stationary', index_col=0)
    m4.index = pd.to_datetime(m4.index)
    dff  = m4['US|US FEDERAL FUNDS TARGET RATE (EP) NADJ']
    dcpi = m4['US|US CPI - ALL URBAN: ALL ITEMS SADJ']
    dch  = m4['CH|CH MACROECONOMIC CLIMATE INDEX - LEADING INDEX SADJ']
    F = pd.DataFrame({'USD':usd,'VIX':vix,'dFF':dff,'dCPI':dcpi,'dCH':dch})
    F.index = F.index.to_period('M').to_timestamp()  # month-START align (returns convention)
    return F

def lagged_factors(F, dates):
    """Market vars lag-1, announcement vars lag-2; aligned to return dates (month-end)."""
    L = pd.DataFrame(index=dates)
    for c, lag in [('USD',1),('VIX',1),('dFF',1),('dCPI',2),('dCH',2)]:
        L[c] = F[c].shift(lag).reindex(dates)
    return L

def fit_gamma(logratio, Fl, cols, fit_idx):
    """OLS per asset: logratio_t ~ const + F. Returns gamma (K, len(cols)), tstats."""
    X = Fl.loc[fit_idx, cols].values
    ok = ~np.isnan(X).any(axis=1)
    X = X[ok]
    Xc = np.column_stack([np.ones(len(X)), X])
    G=[]; Tst=[]
    for k in range(logratio.shape[1]):
        y = logratio.loc[fit_idx].values[ok,k]
        m = ~np.isnan(y)
        b, res, *_ = np.linalg.lstsq(Xc[m], y[m], rcond=None)
        e = y[m]-Xc[m]@b
        s2 = (e@e)/max(len(e)-len(b),1)
        se = np.sqrt(np.diag(s2*np.linalg.pinv(Xc[m].T@Xc[m])))
        G.append(b); Tst.append(b/np.clip(se,1e-12,None))
    return np.array(G), np.array(Tst)   # (K, 1+F)

def run_variant(cols, label, win_mode='isfix'):
    Y,_,_ = load(); dates=Y.index; Yv=Y.values; T,K=Yv.shape
    F = build_factors(); Fl = lagged_factors(F, dates)
    is_mask = dates<='2009-12-31'
    fit_dates = dates[is_mask & (~Fl[cols].isna().any(axis=1)).values]

    # Stage 1 baseline lambda + g path (paper protocol: raw, IS)
    lams,_,_ = ewma_lams_ll(Yv[is_mask], 0.79, 0.986, 50)
    SIGb = ewma_sig_path(Yv, lams, 13)
    g = SIGb**2
    # gamma fit on IS (factor-available part)
    lr = pd.DataFrame(np.log(np.clip(Yv**2,1e-10,None)) - np.log(np.clip(g,1e-12,None)), index=dates)
    gamma, tst = fit_gamma(lr, Fl, cols, fit_dates)
    # tau path (causal: uses lagged F); normalize to IS mean 1
    Xall = np.column_stack([np.ones(T), Fl[cols].values])
    raw_tau = np.exp(np.clip(Xall @ gamma.T, -3, 3))   # (T,K), clip for stability
    raw_tau[np.isnan(Xall).any(axis=1)] = np.nan
    norm = np.nanmean(raw_tau[is_mask.values if hasattr(is_mask,'values') else is_mask], axis=0)
    tau = raw_tau/norm
    tau = pd.DataFrame(tau).ffill().fillna(1.0).values  # pre-availability: 1
    sig = np.sqrt(np.clip(tau*g,1e-14,None))
    return dates, Yv, sig, SIGb, gamma, tst, lams

def backtest(dates, Yv, SIG, start='2010-01-01'):
    K=Yv.shape[1]
    Z = Yv/np.clip(SIG,1e-12,None)
    rets=[]; idxs=[]; w_prev=np.ones(K)/K
    ql=[]
    for i in np.where(dates>=start)[0]:
        s=SIG[i]   # one-step-ahead by construction (uses r_{i-1})
        R=np.corrcoef(Z[i-60:i],rowvar=False); R=(R+R.T)/2
        H=np.outer(s,s)*R+np.eye(K)*1e-10
        w=minvar_lo(H); to=np.abs(w-w_prev).sum()
        rets.append(w@Yv[i]-TC*to); idxs.append(dates[i]); w_prev=w
        sign,ld=np.linalg.slogdet(H)
        ql.append(ld + Yv[i]@np.linalg.solve(H,Yv[i]))
    rp=pd.Series(rets,index=pd.DatetimeIndex(idxs))
    sh=rp.mean()*12/(rp.std(ddof=1)*np.sqrt(12))
    cum=(1+rp).cumprod(); mdd=((cum-cum.cummax())/cum.cummax()).min()
    return sh, mdd, np.mean(ql), rp, np.array(ql)

def nw_t(d, L=12):
    T=len(d); g0=np.var(d,ddof=0); s=g0
    for l in range(1,L+1):
        s += 2*(1-l/(L+1))*np.cov(d[l:],d[:-l],ddof=0)[0,1]
    return d.mean()/np.sqrt(max(s,1e-12)/T)

if __name__=='__main__':
    Y,_,_ = load(); assets=list(Y.columns)
    # Baseline (tau=1)
    dates,Yv,_,SIGb,_,_,lams = run_variant(['USD','VIX'],'base')
    shB,mddB,qlB,rpB,qlsB = backtest(dates,Yv,SIGb)
    print(f"H-DCC baseline (raw, IS고정): Sharpe={shB:.3f} MaxDD={mddB*100:.1f}% QLIKE={qlB:.3f}")
    results={}
    for cols,label in [(['USD','VIX'],'X2 (USD+VIX, 1994~)'),
                       (['USD','VIX','dFF','dCPI','dCH'],'X5 (5팩터, γ 2004~09 적합)')]:
        dates,Yv,sigX,_,gamma,tst,_ = run_variant(cols,label)
        sh,mdd,ql,rp,qls = backtest(dates,Yv,sigX)
        t_sh = nw_t((rp-rpB).values)       # return diff
        t_ql = nw_t(qlsB-qls)              # loss diff: + => X better
        print(f"\nH-DCC-{label}: Sharpe={sh:.3f} (Δ{sh-shB:+.3f}, ret-diff t={t_sh:.2f})  "
              f"MaxDD={mdd*100:.1f}%  QLIKE={ql:.3f} (Δ{ql-qlB:+.3f}, DM t={t_ql:.2f})")
        results[label]=(gamma,tst,cols)
    # gamma cross-section for X5
    gamma,tst,cols = results['X5 (5팩터, γ 2004~09 적합)']
    print("\nγₖ 횡단면 (X5, |t|>2 강조):")
    hdr = "자산      " + "".join(f"{c:>8s}" for c in cols)
    print(hdr)
    for k,a in enumerate(assets):
        row=f"{a:9s}"
        for j,c in enumerate(cols):
            v=gamma[k,j+1]; t=tst[k,j+1]
            mark='*' if abs(t)>2 else ' '
            row+=f"{v:7.3f}{mark}"
        print(row)
