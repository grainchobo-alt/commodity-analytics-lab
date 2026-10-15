"""A그룹(요인 압축·모형 평균·베이지안 축소) 검증 — 시뮬레이션으로 알려진 구조를 복원하는지 확인."""
import sys, time
from math import comb
from itertools import combinations
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import analytics as A, models_dense as MD

t0 = time.time()
def ok(m): print(f"[{time.time()-t0:6.1f}s] OK  {m}")
rng = np.random.default_rng(11)
idx = pd.bdate_range("2000-01-03", periods=1200)

def design(X, y, names=None, mode="explain", target=""):
    df = pd.DataFrame(X, index=idx[:len(y)], columns=names or [f"x{i}" for i in range(X.shape[1])]); df["y"] = y
    D = A.build_design(df, "y", list(df.columns[:-1]), mode, train_frac=0.7)
    D.target = target
    return D

# ---- 1) 요인 구조: X = FΛ + e (30개 변수), y = F·β + ε → 요인 모델이 희소 모델보다 나아야 함 ----
T, N, K = 1200, 30, 2
F = rng.standard_normal((T, K)); Lam = rng.standard_normal((K, N))
X = F @ Lam + 1.0 * rng.standard_normal((T, N)); y = F @ np.array([1.0, -0.6]) + 1.0 * rng.standard_normal(T)
D = design(X, y)
res = {m: A.run_model(m, D, 252) for m in ["OLS", "Lasso", "PCR (주성분회귀)", "PLS (부분최소제곱)", "3PRF (3단계 회귀필터)", "Complete Subset", "Minnesota BVAR"]}
r2 = {m: r["metrics"]["R² (테스트)"] for m, r in res.items()}
oracle = 1 - 1.0 / np.var(F @ np.array([1.0, -0.6]) + rng.standard_normal(T))
assert res["PCR (주성분회귀)"]["extra"]["k"] in (2, 3), res["PCR (주성분회귀)"]["extra"]["k"]
for m in ["PCR (주성분회귀)", "PLS (부분최소제곱)", "3PRF (3단계 회귀필터)"]:
    assert r2[m] >= r2["Lasso"] - 0.01, (m, r2)
ok("요인 구조 복원: PCR k=%d, 테스트 R² %s (이론 상한 ≈ %.2f)" % (res["PCR (주성분회귀)"]["extra"]["k"],
   {m.split(" ")[0]: round(v, 3) for m, v in r2.items()}, oracle))

# ---- 2) Complete Subset: 조합 수 ≤ M이면 전수 평균과 정확히 일치 ------------------------
Xs = rng.standard_normal((300, 6)); ys = Xs @ np.array([1, .5, 0, 0, -.5, 0]) + rng.standard_normal(300)
Xs = (Xs - Xs.mean(0)) / Xs.std(0); ys = ys - ys.mean()
b, _ = MD._csr(Xs, ys, 3, M=10_000)
ref = np.zeros(6)
for S in combinations(range(6), 3):
    S = list(S); ref[S] += np.linalg.lstsq(Xs[:, S], ys, rcond=None)[0]
assert np.allclose(b, ref / comb(6, 3), atol=1e-8)
ok("Complete Subset: 20개 조합 OLS 평균과 정확히 일치")

# ---- 3) BMA: 희소 구조에서 진짜 변수 PIP ↑, 무관 변수 PIP ↓ -------------------------------
Xb = rng.standard_normal((800, 15)); beta = np.zeros(15); beta[[0, 3, 7]] = [0.6, -0.4, 0.3]
yb = Xb @ beta + rng.standard_normal(800)
Db = design(Xb, yb)
ob = A.run_model("BMA (베이즈 모형평균)", Db, 252)
pip = ob["extra"]["pip"].values
assert (pip[[0, 3, 7]] > 0.9).all() and (np.delete(pip, [0, 3, 7]) < 0.3).all(), np.round(pip, 2)
ok(f"BMA PIP: 진짜 변수 {np.round(pip[[0,3,7]], 2)}, 무관 변수 최대 {np.delete(pip, [0,3,7]).max():.2f} (방문 모형 {ob['extra']['visited']}개)")

# ---- 4) Horseshoe: 큰 계수는 보존, 잡음 계수는 0 근처로 강하게 축소 ------------------------
oh = A.run_model("Horseshoe (베이즈 축소)", Db, 252)
ci = oh["extra"]["ci"]
assert np.allclose(ci["사후평균"].values[[0, 3, 7]], [0.6, -0.4, 0.3], atol=0.12), ci.iloc[[0, 3, 7]]
assert np.abs(np.delete(ci["사후평균"].values, [0, 3, 7])).max() < 0.08
assert (ci["5%"].values[[0, 7]] > 0).all() and ci["95%"].values[3] < 0
ols = A.run_model("OLS", Db, 252)["coef"].values
assert np.abs(np.delete(ci["사후평균"].values, [0, 3, 7])).mean() < np.abs(np.delete(ols, [0, 3, 7])).mean()
ok(f"Horseshoe: 진짜 계수 {np.round(ci['사후평균'].values[[0,3,7]], 2)}, 잡음 계수 평균|β| {np.abs(np.delete(ci['사후평균'].values,[0,3,7])).mean():.3f} < OLS {np.abs(np.delete(ols,[0,3,7])).mean():.3f}")

# ---- 5) Minnesota: λ→∞ = OLS, λ→0 = 0, 먼 시차일수록 더 축소 ------------------------------
names = ["y (t)", "y (t-1)", "y (t-2)", "z (t)", "z (t-1)", "z (t-2)"]
Xm = rng.standard_normal((500, 6)); ym = Xm @ np.array([.3, .3, .3, .3, .3, .3]) + rng.standard_normal(500)
Xm = (Xm - Xm.mean(0)) / Xm.std(0); ym = ym - ym.mean()
b_big, _ = MD._minnesota(Xm, ym, 1e6, names, "y"); b_ols = np.linalg.lstsq(Xm, ym, rcond=None)[0]
b_small, _ = MD._minnesota(Xm, ym, 1e-6, names, "y")
assert np.allclose(b_big, b_ols, atol=1e-6) and np.abs(b_small).max() < 1e-4
b_mid, _ = MD._minnesota(Xm, ym, 0.05, names, "y", theta=0.5)
assert abs(b_mid[0]) > abs(b_mid[2]) and abs(b_mid[0]) > abs(b_mid[3]), np.round(b_mid, 3)
ok(f"Minnesota: λ→∞=OLS, λ→0=0, 축소 순서 자기 t > 자기 t-2, 자기 > 타변수: {np.round(b_mid, 3)}")

# ---- 6) Stacking: 가중치 비음·합 1 ---------------------------------------------------------
os_ = A.run_model("Stacking (가중 앙상블)", D, 252)
w = os_["extra"]["stack_weights"]
assert (w >= -1e-12).all() and abs(w.sum() - 1) < 1e-9
ok(f"Stacking 가중치 {w.round(3).to_dict()}, 테스트 R² {os_['metrics']['R² (테스트)']:.3f}")

# ---- 7) 실데이터 예측모드 (주간 옥수수, 특성 포함) -------------------------------------------
cat = A.load_catalog(); U = dict(zip(cat.id, cat.unit)); L = A.load_levels()
ids = ["yf:ZC=F", "yf:ZS=F", "yf:ZW=F", "yf:CL=F", "yf:HG=F", "yf:GC=F", "fred:VIXCLS", "yf:DX-Y.NYB"]
R = A.make_panel(L, ids, U, "W-FRI", "logret", "2005-01-01")
Fz = A.make_features(L, "yf:ZC=F", U, "W-FRI", R.index, A.FEATURE_KINDS)
Dr = A.build_design(R, "yf:ZC=F", ids[1:], "predict", lags=2, extra=Fz)
for m in MD.DENSE_MODELS:
    o = A.run_model(m, Dr, 52)
    print(f"   {m:24s} OOS R² {o['metrics']['OOS R² (vs 학습평균)']:+.4f}  Sharpe {o['metrics']['부호전략 Sharpe (비용 후)']:+.2f}")
ok("실데이터 8개 모델 실행")
print("DENSE TEST PASSED")
