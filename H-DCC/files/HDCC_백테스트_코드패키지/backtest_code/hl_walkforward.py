"""Half-life (HL) stability under walk-forward vs IS-fixed, monthly & daily.
HL_k = ln(0.5)/ln(lambda_k). Compares:
  (1) IS-fixed (1994-2009) lambda  — paper baseline
  (2) WF-expanding (annual refit, all data to date)
  (3) WF-rolling (trailing window)
Each under BOUNDED grid (paper bounds) vs UNBOUNDED grid (wide), per asset.
"""
import numpy as np, pandas as pd
from backtest_core import load, load_daily

def qmle_lambda_scalar(e, lo, hi, G=200):
    e = np.asarray(e,float); e=e[~np.isnan(e)]
    if len(e)<12: return np.nan
    lams=np.linspace(lo,hi,G)
    s2=np.full(G, np.var(e[:min(60,len(e))])); s2=np.clip(s2,1e-14,None)
    ll=np.zeros(G); e2=e**2
    for t in range(1,len(e)):
        s2=lams[:,None].ravel()*s2+(1-lams)*e2[t-1]
        np.clip(s2,1e-14,None,out=s2)
        ll+=np.log(s2)+e2[t]/s2
    return lams[np.argmin(ll)]

def HL(lam):
    return np.log(0.5)/np.log(lam) if (lam and 0<lam<1) else np.nan

# Paper bounds: monthly lambda in [0.79,0.986] (HL 2.9~49.5mo)
# Daily: map -> [0.90,0.999]; unbounded wide: monthly [0.50,0.9995], daily [0.70,0.99995]
CONF = {
  'm': dict(loader=load, bound=(0.79,0.986), wide=(0.50,0.9995), refit_min=48, roll=60, freq='월'),
  'd': dict(loader=load_daily, bound=(0.90,0.999), wide=(0.70,0.99995), refit_min=1008, roll=756, freq='일'),
}

def jan_idx(dates, min_t):
    seen=set(); out=[]
    for t in range(len(dates)):
        y=dates[t].year
        if y not in seen:
            seen.add(y)
            if t>=min_t: out.append(t)
    return out

def analyze(freq):
    c=CONF[freq]
    Yo=c['loader']()
    Y = Yo[0] if isinstance(Yo,tuple) else Yo
    dates=Y.index; X=Y.values; K=X.shape[1]; assets=list(Y.columns)
    is_end=np.searchsorted(dates,pd.Timestamp('2009-12-31'),'right')
    rows=[]
    for bound_name,(lo,hi) in [('경계O',c['bound']),('경계X',c['wide'])]:
        # (1) IS-fixed
        lam_is=np.array([qmle_lambda_scalar(X[:is_end,k],lo,hi) for k in range(K)])
        # (2) WF-expanding: lambda at each Jan refit, using all data to date; take terminal + path stats
        refits=jan_idx(dates,c['refit_min'])
        lam_exp_path=np.array([[qmle_lambda_scalar(X[:t,k],lo,hi) for k in range(K)] for t in refits])
        # (3) WF-rolling
        lam_roll_path=np.array([[qmle_lambda_scalar(X[max(0,t-c['roll']):t,k],lo,hi) for k in range(K)] for t in refits])
        for k in range(K):
            hl_is=HL(lam_is[k])
            hl_exp=HL(lam_exp_path[-1,k])             # terminal expanding
            hl_exp_mean=np.nanmean([HL(x) for x in lam_exp_path[:,k]])
            hl_exp_std=np.nanstd([HL(x) for x in lam_exp_path[:,k]])
            hl_roll_mean=np.nanmean([HL(x) for x in lam_roll_path[:,k]])
            hl_roll_std=np.nanstd([HL(x) for x in lam_roll_path[:,k]])
            rows.append(dict(주기=c['freq'],경계=bound_name,자산=assets[k],
                IS_HL=round(hl_is,1), WFexp_종단_HL=round(hl_exp,1),
                WFexp_평균_HL=round(hl_exp_mean,1), WFexp_표준편차=round(hl_exp_std,1),
                WFroll_평균_HL=round(hl_roll_mean,1), WFroll_표준편차=round(hl_roll_std,1),
                IS대비_종단편차=round(hl_exp-hl_is,1)))
    return pd.DataFrame(rows)

if __name__=='__main__':
    dm=analyze('m'); dd=analyze('d')
    for name,df in [('월간',dm),('일간',dd)]:
        print(f"\n══════ {name} ══════")
        for bn in ['경계O','경계X']:
            sub=df[df.경계==bn]
            print(f"\n[{bn}] IS고정 vs WF확장(종단) 반감기:")
            print(sub[['자산','IS_HL','WFexp_종단_HL','WFexp_평균_HL','WFexp_표준편차','IS대비_종단편차']].to_string(index=False))
    dm.to_csv('hl_wf_monthly.csv',index=False); dd.to_csv('hl_wf_daily.csv',index=False)
