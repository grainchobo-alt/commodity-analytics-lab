"""DCC / ADCC / BEKK / HMM 정합성 테스트 — 시뮬레이션으로 모수 복원 확인 후 실데이터 적용."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import mvgarch as M
import analytics as A

t0 = time.time()
def ok(m): print(f"[{time.time()-t0:6.1f}s] OK  {m}")
rng = np.random.default_rng(1)
idx = pd.bdate_range("2000-01-03", periods=3000)

# ---- DCC 시뮬레이션: a=0.05, b=0.90, 3자산, 단변량 GARCH(0.05,0.90) --------------------
N, T, a, b = 3, 3000, 0.05, 0.90
Qbar = np.array([[1, .5, .3], [.5, 1, .4], [.3, .4, 1.]])
Q = Qbar.copy(); h = np.ones(N); eps = np.zeros((T, N)); z_prev = np.zeros(N)
for t in range(T):
    if t:
        Q = (1 - a - b) * Qbar + a * np.outer(z_prev, z_prev) + b * Q
    d = np.sqrt(np.diag(Q)); Rt = Q / np.outer(d, d)
    z = np.linalg.cholesky(Rt) @ rng.standard_normal(N)
    h = 0.05 + 0.05 * (eps[t - 1] ** 2 if t else 1) + 0.90 * h
    eps[t] = np.sqrt(h) * z; z_prev = z
R = pd.DataFrame(eps, index=idx, columns=list("XYZ"))
d = M.dcc_fit(R, asymmetric=False, kind="GARCH", dist="normal", ann=252)
pa, pb = d["params"]["계수"].values
assert abs(pa - a) < 0.025 and abs(pb - b) < 0.05, (pa, pb)
ok(f"DCC 모수 복원 a={pa:.3f} (참 0.05), b={pb:.3f} (참 0.90), t값 {d['params']['t값'].round(1).tolist()}")
da = M.dcc_fit(R, asymmetric=True, kind="GARCH", dist="normal", ann=252)
assert da["params"].loc["g (비대칭)", "계수"] < 0.03
ok(f"ADCC (비대칭 없는 데이터) g={da['params'].loc['g (비대칭)', '계수']:.4f} ≈ 0")

# ---- BEKK 시뮬레이션: 대각 A=0.3, B=0.92, X→Y 충격 전이 a12=0.15 --------------------
N, T = 2, 3000
Ct = np.array([[0.3, 0], [0.1, 0.25]]); At = np.array([[0.30, 0.15], [0.0, 0.30]]); Bt = np.diag([0.92, 0.92])
H = np.eye(2); e = np.zeros((T, 2))
for t in range(T):
    if t:
        H = Ct @ Ct.T + At.T @ np.outer(e[t - 1], e[t - 1]) @ At + Bt.T @ H @ Bt
    e[t] = np.linalg.cholesky(H) @ rng.standard_normal(2)
Rb = pd.DataFrame(e, index=idx, columns=["X", "Y"])
bk = M.bekk_fit(Rb, "Full", ann=252)
Ae, Ate = bk["A"].values, bk["A_t"].values
E_ = np.ascontiguousarray((Rb - Rb.mean()).values)
ll_true = -M._bekk_core(Ct, At, Bt, E_, np.cov(E_, rowvar=False), False)[0]
assert bk["loglik"] >= ll_true - 1e-6, "최적화가 참값보다 낮은 우도에 멈춤"
assert abs(Ae[0, 1] - 0.15) < 0.1 and abs(Ate[0, 1]) > 2, (Ae, Ate)     # 표본 변동 허용 (seed별 0.12~0.24)
assert abs(Ate[1, 0]) < 2
ok(f"BEKK Full: A=\n{np.round(Ae, 3)}\n   t(A12: X→Y 전이)={Ate[0,1]:.1f}, stationarity={bk['stationarity']:.3f}, converged={bk['converged']}")
sp = M.spillover_table(bk)
assert sp[(sp["유형"].str.startswith("충격")) & (sp["from → to"] == "X → Y")]["유의(5%)"].iloc[0] == "✔"
ok("BEKK 전이표: X → Y 충격전이 유의, Y → X 비유의" if sp[(sp["유형"].str.startswith("충격")) & (sp["from → to"] == "Y → X")]["유의(5%)"].iloc[0] == "" else "BEKK 전이표")
for kind in ["Diagonal", "Scalar"]:
    b2 = M.bekk_fit(Rb, kind, ann=252); ok(f"BEKK {kind}: loglik={b2['loglik']:.1f}, AIC={b2['aic']:.1f} (Full AIC={bk['aic']:.1f})")

# ---- HMM: 2레짐 (저변동 평균 +, 고변동 평균 -) 복원 -----------------------------------
T = 2000; s = np.zeros(T, int); P = np.array([[0.98, 0.02], [0.05, 0.95]])
for t in range(1, T): s[t] = rng.choice(2, p=P[s[t - 1]])
x = rng.standard_normal((T, 2))
y = np.where(s == 0, 0.2 + 0.8 * x[:, 0], -0.5 - 0.8 * x[:, 0]) + np.where(s == 0, 0.5, 2.0) * rng.standard_normal(T)
Rh = pd.DataFrame({"y": y, "x1": x[:, 0], "x2": x[:, 1]}, index=pd.bdate_range("2000-01-03", periods=T))
D = A.build_design(Rh, "y", ["x1", "x2"], "explain", train_frac=0.7)
out = A.run_model("Hidden Markov (HMM)", D, 252, mode="explain", hmm_states=2, hmm_cov="full")
sb = out["extra"]["state_betas"]
assert sb.loc["레짐1", "x1"] > 0.5 and sb.loc["레짐2", "x1"] < -0.4, sb
ols = A.run_model("OLS", D, 252)
ok(f"HMM 레짐별 베타 복원 x1: 레짐1 {sb.loc['레짐1','x1']:.2f} (참 0.8), 레짐2 {sb.loc['레짐2','x1']:.2f} (참 -0.8) · "
   f"테스트 R² HMM {out['metrics']['R² (테스트)']:.3f} vs OLS {ols['metrics']['R² (테스트)']:.3f}")

# ---- 실데이터 ----
cat = A.load_catalog(); U = dict(zip(cat.id, cat.unit)); L = A.load_levels()
ids = ["yf:ZC=F", "yf:ZS=F", "yf:ZW=F", "yf:CL=F", "yf:HG=F"]
Rw = A.make_panel(L, ids, U, "W-FRI", "logret", "2005-01-01")
gt, gv = M.garch_table(Rw, 52)
ok("GARCH 요약표\n" + gt[["현재 GARCH 변동성", "장기 변동성", "22기 예측", "현재 백분위(%)", "지속성"]].round(2).to_string())
dr = M.dcc_fit(Rw, asymmetric=True, ann=52)
ok("ADCC 실데이터\n" + dr["params"].round(4).to_string() + f"\n   평균 DCC 상관 최근 {dr['avg_corr'].iloc[-1]:.3f}, 최대 {dr['avg_corr'].max():.3f} ({dr['avg_corr'].idxmax():%Y-%m-%d})")
b3 = M.bekk_fit(Rw[["yf:ZC=F", "yf:CL=F", "yf:HG=F"]], "Full", ann=52)
ok(f"BEKK Full 실데이터 (옥수수·WTI·구리) converged={b3['converged']} stationarity={b3['stationarity']:.4f}")
print(M.spillover_table(b3, {"yf:ZC=F": "Corn", "yf:CL=F": "WTI", "yf:HG=F": "Copper"}).round(3).to_string(index=False))
D2 = A.build_design(Rw, "yf:ZC=F", ids[1:], "predict", lags=2)
o2 = A.run_model("Hidden Markov (HMM)", D2, 52, mode="predict", hmm_states=2, hmm_cov="full")
ok("HMM 예측모드 실데이터\n" + o2["extra"]["regime"].round(3).to_string() + f"\n   OOS R²={o2['metrics']['OOS R² (vs 학습평균)']:.4f}")
print("MVGARCH/HMM TEST PASSED")
