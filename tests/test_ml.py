"""D그룹 검증: 비선형 구조 복원, SHAP 가법성, GP 관련도, conformal 커버리지·적응성·미래정보 차단."""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import analytics as A, models_ml as M

t0 = time.time()
def ok(m): print(f"[{time.time()-t0:6.1f}s] OK  {m}")
rng = np.random.default_rng(7)
T = 2000
idx = pd.bdate_range("2000-01-07", periods=T, freq="W-FRI")
X = rng.standard_normal((T, 8))
cols = [f"x{i}" for i in range(8)]

# ---- 1) LightGBM: 비선형 + 상호작용, SHAP 가법성 -------------------------------------------
y = np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2] + 0.3 * rng.standard_normal(T)
D = A.build_design(pd.DataFrame(np.column_stack([y, X]), index=idx, columns=["y"] + cols), "y", cols, "explain", train_frac=0.7)
lg = A.run_model("LightGBM", D, 52); ols = A.run_model("OLS", D, 52)
r2 = lambda o: o["metrics"]["R² (테스트)"]
assert r2(lg) > r2(ols) + 0.4, (r2(lg), r2(ols))
ex = lg["extra"]
recon = ex["shap_base"] + ex["shap"].sum(1).values
assert np.allclose(recon, lg["pred_te"], atol=1e-6)
top3 = set(ex["importance"].index[-3:])
assert top3 == {"x0", "x1", "x2"}, ex["importance"]
ok(f"LightGBM: 테스트 R² {r2(lg):.3f} vs OLS {r2(ols):.3f}, SHAP 가법성 (기준값+Σ기여 = 예측), 상위 3 = {sorted(top3)}, 나무 {ex['n_trees']}개")

# ---- 2) Gaussian Process: 평활 비선형 + ARD 관련도 + 자체 구간 ----------------------------
Xg = X[:, :4]; yg = np.sin(2 * Xg[:, 0]) + 0.3 * rng.standard_normal(T)
Dg = A.build_design(pd.DataFrame(np.column_stack([yg, Xg]), index=idx, columns=["y"] + cols[:4]), "y", cols[:4], "explain", train_frac=0.7)
gp = A.run_model("Gaussian Process", Dg, 52, gp_nmax=800)
assert r2(gp) > 0.75 and gp["extra"]["importance"].idxmax() == "x0", (r2(gp), gp["extra"]["importance"])
sd = gp["extra"]["pred_std"].values; e = Dg.y_te - gp["pred_te"]
cov = np.mean(np.abs(e) <= 1.645 * sd)
assert 0.85 < cov < 0.95, cov
ok(f"Gaussian Process: R² {r2(gp):.3f}, 관련도 1위 x0 (ARD), 자체 90% 구간 커버리지 {cov:.1%}, 분산 구성 {gp['extra']['var_share'].round(2).to_dict()}")

kr = A.run_model("Kernel Ridge", Dg, 52); olg = A.run_model("OLS", Dg, 52)
assert r2(kr) > r2(olg) + 0.3, (r2(kr), r2(olg))
ok(f"Kernel Ridge: R² {r2(kr):.3f} vs OLS {r2(olg):.3f} (α={kr['extra']['alpha']:g}, γ·p={kr['extra']['gamma']:g})")

# ---- 3) Conformal: i.i.d. 에서 명목 커버리지 ------------------------------------------------
yl = X[:, :3] @ np.array([0.5, -0.3, 0.2]) + rng.standard_normal(T)
Dl = A.build_design(pd.DataFrame(np.column_stack([yl, X]), index=idx, columns=["y"] + cols), "y", cols, "explain", train_frac=0.6)
c = M.conformal(Dl, "Ridge", 52, alpha=0.1, normalize=False)
cs, ca = c["summary"].loc["커버리지"]
assert 0.86 < cs < 0.94 and 0.86 < ca < 0.94, c["summary"]
ok(f"Conformal (i.i.d.): 분할 {cs:.1%}, ACI {ca:.1%} (목표 90%)")

# ---- 4) 변동성 국면 변화: 테스트 구간 후반 잡음 3배 → 분할은 미달, ACI·정규화는 회복 ------------
noise = np.where(np.arange(T) > 1500, 3.0, 1.0) * rng.standard_normal(T)
ys = X[:, :3] @ np.array([0.5, -0.3, 0.2]) + noise
Ds = A.build_design(pd.DataFrame(np.column_stack([ys, X]), index=idx, columns=["y"] + cols), "y", cols, "explain", train_frac=0.6)
c0 = M.conformal(Ds, "Ridge", 52, alpha=0.1, normalize=False, gamma=0.02)
c1 = M.conformal(Ds, "Ridge", 52, alpha=0.1, normalize=True)
late = Ds.idx_te > idx[1600]
cov_late = lambda c, k: float(((c["y"] >= c[k].lo) & (c["y"] <= c[k].hi))[late].mean())
s_late, a_late, n_late = cov_late(c0, "split"), cov_late(c0, "aci"), cov_late(c1, "split")
assert s_late < 0.75 and a_late > 0.83 and n_late > 0.83, (s_late, a_late, n_late)
ok(f"국면 변화(잡음 3배) 이후 커버리지: 분할 {s_late:.1%} → ACI {a_late:.1%}, 변동성 정규화 분할 {n_late:.1%}")

# ---- 5) 미래정보: 테스트 k번째 이후 y를 바꿔도 t < k+h 구간은 불변 --------------------------------
k = 150
Dp = A.Design(Ds.X_tr, Ds.y_tr, Ds.X_te, Ds.y_te.copy(), Ds.idx_tr, Ds.idx_te, Ds.names, Ds.mu, Ds.sd, Ds.y_mu, None, 1, "explain", 0.0, "y")
Dp.y_te[k:] = Dp.y_te[k:] * 5 + 10
a, b = M.conformal(Ds, "Ridge", 52), M.conformal(Dp, "Ridge", 52)
for key in ["split", "aci"]:
    assert np.allclose(a[key].iloc[:k + 1].values, b[key].iloc[:k + 1].values), key
    assert not np.allclose(a["aci"].iloc[k + 5:].values, b["aci"].iloc[k + 5:].values)
ok("미래정보: 테스트 이후 값을 바꿔도 그 이전 구간(분할·ACI) 불변 — 실현된 오차만 사용")
print("ML TEST PASSED")
