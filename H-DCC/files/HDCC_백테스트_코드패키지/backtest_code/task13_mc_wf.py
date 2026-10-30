"""Monte Carlo: does walk-forward lambda estimation track true heterogeneous lambdas?
DGP: EWMA variances with paper's lambda_k (truth), Gaussian shocks with IS corr.
Monthly: S=400, T=384 — IS-fixed vs WF-expanding vs WF-rolling36.
Daily:   S=100, T=7911 — checkpoints (biennial) expanding vs rolling-756.
"""
import numpy as np, pandas as pd, time

M = '/home/claude/hdcc/HDCC_Master_Data.xlsx'
lamtab = pd.read_excel(M, sheet_name='07_EWMA_Lambda_HL')
lam_col = [c for c in lamtab.columns if 'λ' in str(c) or 'lambda' in str(c).lower()]
true_lam_m = lamtab[lam_col[0]].values[:12].astype(float)
corr = pd.read_excel(M, sheet_name='10_Corr_IS_Uncond', index_col=0).values[:12,:12].astype(float)
K = 12
Lc = np.linalg.cholesky(corr + np.eye(K)*1e-8)
print("true λ (monthly):", np.round(true_lam_m,3))

def simulate(T, lams, S, seed):
    rng = np.random.default_rng(seed)
    out = np.zeros((S,T,K))
    for s in range(S):
        z = rng.standard_normal((T,K)) @ Lc.T
        sig2 = np.full(K, 1.0)
        r = np.zeros((T,K))
        for t in range(T):
            r[t] = np.sqrt(sig2)*z[t]
            sig2 = lams*sig2 + (1-lams)*r[t]**2
        out[s]=r
    return out

def qmle_panel(E, lo, hi, G):
    """E: (T,M) panel. Returns lam_hat (M,)."""
    T,Mn = E.shape
    lams = np.linspace(lo,hi,G)
    s2 = np.tile(np.var(E[:min(60,T)],axis=0),(G,1)); np.clip(s2,1e-14,None,out=s2)
    ll = np.zeros((G,Mn)); E2=E**2
    for t in range(1,T):
        s2 = lams[:,None]*s2 + (1-lams[:,None])*E2[t-1]
        np.clip(s2,1e-14,None,out=s2)
        ll += np.log(s2)+E2[t]/s2
    return lams[ll.argmin(0)]

def rankcorr(a,b):
    ra = np.argsort(np.argsort(a)); rb=np.argsort(np.argsort(b))
    return np.corrcoef(ra,rb)[0,1]

# ═══ MONTHLY ═══
S=400; T=384
t0=time.time()
R = simulate(T, true_lam_m, S, 11)
panel = R.transpose(1,0,2).reshape(T, S*K)  # (T, S*K)

res_rows=[]
def eval_lams(lh, label):
    lh = lh.reshape(S,K)
    bias = (lh-true_lam_m).mean(0)
    rmse = np.sqrt(((lh-true_lam_m)**2).mean(0))
    rc = np.mean([rankcorr(lh[s],true_lam_m) for s in range(S)])
    res_rows.append({'설정':label,'평균 bias':round(bias.mean(),4),
        '평균 RMSE':round(rmse.mean(),4),'순위상관(λ̂,λ)':round(rc,3)})

# (a) IS-fixed: first 192
eval_lams(qmle_panel(panel[:192],0.79,0.986,50), '월간 IS고정(T=192)')
# (b) full sample
eval_lams(qmle_panel(panel,0.79,0.986,50), '월간 전체표본(T=384)')
# (c) WF expanding: refits every 12m from t=48; track path stability + final
refits = list(range(48,T,12))
path = np.zeros((len(refits), S*K))
for j,t in enumerate(refits):
    path[j] = qmle_panel(panel[:t],0.79,0.986,50)
eval_lams(path[-1], '월간 WF확장 최종(T≈372)')
pp = path.reshape(len(refits),S,K)
stab = pp.std(axis=0).mean()
rc_path = np.mean([[rankcorr(pp[j,s],true_lam_m) for s in range(S)] for j in range(len(refits))])
res_rows.append({'설정':'월간 WF확장 경로','평균 bias':'—','평균 RMSE':f'경로 std={stab:.4f}','순위상관(λ̂,λ)':round(rc_path,3)})
# (d) WF rolling-36
path_r = np.zeros((len(refits), S*K))
for j,t in enumerate(refits):
    path_r[j] = qmle_panel(panel[max(0,t-36):t],0.79,0.986,50)
ppr = path_r.reshape(len(refits),S,K)
stab_r = ppr.std(axis=0).mean()
rc_r = np.mean([[rankcorr(ppr[j,s],true_lam_m) for s in range(S)] for j in range(len(refits))])
res_rows.append({'설정':'월간 WF롤링36 경로','평균 bias':round((ppr.mean(0)-true_lam_m).mean(),4),
    '평균 RMSE':f'경로 std={stab_r:.4f}','순위상관(λ̂,λ)':round(rc_r,3)})
print(f"monthly MC done {time.time()-t0:.0f}s")

# ═══ DAILY ═══
# daily true lambdas: map monthly HL to daily lambda: HL_d = HL_m * 21 → λ_d = 0.5^(1/HL_d)
HL_m = np.log(0.5)/np.log(true_lam_m)
lam_d = 0.5**(1/(HL_m*21))
S2=100; T2=7911
t0=time.time()
Rd = simulate(T2, lam_d, S2, 22)
panel_d = Rd.transpose(1,0,2).reshape(T2, S2*K)
eval_rows_d=[]
def eval_d(lh,label):
    lh=lh.reshape(S2,K)
    rmse=np.sqrt(((lh-lam_d)**2).mean(0))
    rc=np.mean([rankcorr(lh[s],lam_d) for s in range(S2)])
    eval_rows_d.append({'설정':label,'평균 RMSE':round(rmse.mean(),5),'순위상관':round(rc,3)})
eval_d(qmle_panel(panel_d[:4000],0.90,0.999,40), '일간 IS고정(T≈4000)')
cps = list(range(1260,T2,504))  # biennial from year ~5
path_d=np.zeros((len(cps),S2*K)); path_dr=np.zeros((len(cps),S2*K))
for j,t in enumerate(cps):
    path_d[j]=qmle_panel(panel_d[:t],0.90,0.999,40)
    path_dr[j]=qmle_panel(panel_d[t-756:t],0.90,0.999,40)
eval_d(path_d[-1],'일간 WF확장 최종')
pd_=path_d.reshape(len(cps),S2,K); pdr=path_dr.reshape(len(cps),S2,K)
rc_pe=np.mean([[rankcorr(pd_[j,s],lam_d) for s in range(S2)] for j in range(len(cps))])
rc_pr=np.mean([[rankcorr(pdr[j,s],lam_d) for s in range(S2)] for j in range(len(cps))])
eval_rows_d.append({'설정':'일간 WF확장 경로','평균 RMSE':f'경로 std={pd_.std(0).mean():.5f}','순위상관':round(rc_pe,3)})
eval_rows_d.append({'설정':'일간 WF롤링756 경로','평균 RMSE':f'경로 std={pdr.std(0).mean():.5f}','순위상관':round(rc_pr,3)})
print(f"daily MC done {time.time()-t0:.0f}s")

dfm=pd.DataFrame(res_rows); dfd=pd.DataFrame(eval_rows_d)
print("\n═══ 월간 MC (S=400, K=12) ═══"); print(dfm.to_string(index=False))
print("\n═══ 일간 MC (S=100, K=12) ═══"); print(dfd.to_string(index=False))
dfm.to_csv('mc_wf_monthly.csv',index=False); dfd.to_csv('mc_wf_daily.csv',index=False)
