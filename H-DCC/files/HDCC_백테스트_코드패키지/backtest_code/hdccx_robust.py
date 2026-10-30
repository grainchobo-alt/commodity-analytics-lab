"""Robustness: single-factor VIX, gamma shrinkage, tau smoothing."""
import numpy as np, pandas as pd
from backtest_core import load
from bench_core import ewma_lams_ll, ewma_sig_path
from hdccx_macro import build_factors, lagged_factors, fit_gamma, backtest

Y,_,_ = load(); dates=Y.index; Yv=Y.values; T,K=Yv.shape
F = build_factors(); Fl = lagged_factors(F, dates)
is_mask = (dates<='2009-12-31')
lams,_,_ = ewma_lams_ll(Yv[is_mask],0.79,0.986,50)
SIGb = ewma_sig_path(Yv, lams, 13); g = SIGb**2
lr = pd.DataFrame(np.log(np.clip(Yv**2,1e-10,None))-np.log(np.clip(g,1e-12,None)), index=dates)

shB,mddB,qlB,rpB,qlsB = backtest(dates,Yv,SIGb)
print(f"baseline: SR={shB:.3f} QLIKE={qlB:.3f}")

def variant(cols, shrink=1.0, smooth=0, clip=1.0, label=''):
    fit_idx = dates[is_mask & (~Fl[cols].isna().any(axis=1)).values]
    gamma,tst = fit_gamma(lr, Fl, cols, fit_idx)
    gamma[:,1:] *= shrink
    Xall = np.column_stack([np.ones(T), Fl[cols].values])
    tau = np.exp(np.clip(Xall@gamma.T, -clip, clip))
    tau[np.isnan(Xall).any(axis=1)] = np.nan
    tau = tau/np.nanmean(tau[is_mask],axis=0)
    tau = pd.DataFrame(tau).ffill().fillna(1.0)
    if smooth>0: tau = tau.rolling(smooth,min_periods=1).mean()
    sig = np.sqrt(np.clip(tau.values*g,1e-14,None))
    sh,mdd,ql,rp,qls = backtest(dates,Yv,sig)
    from hdccx_macro import nw_t
    print(f"{label:38s}: SR={sh:.3f} (Δ{sh-shB:+.3f})  QLIKE={ql:.3f} (DM t={nw_t(qlsB-qls):+.2f})")

variant(['VIX'], 1.0, 0, 1.0, 'VIX 단일 (1994~)')
variant(['USD'], 1.0, 0, 1.0, 'USD 단일 (1994~)')
variant(['USD','VIX'], 0.5, 0, 0.5, 'X2 + γ×0.5 축소 + clip 0.5')
variant(['USD','VIX'], 1.0, 3, 0.5, 'X2 + 3개월 평활')
variant(['USD','VIX','dFF','dCPI','dCH'], 0.3, 3, 0.5, 'X5 + γ×0.3 + 평활 (과적합 보정)')
