"""B-2 MIDAS 검증 — 시뮬레이션으로 가중치·모수 복원, 실데이터 GARCH-MIDAS."""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import analytics as A, midas as M

t0 = time.time()
def ok(m): print(f"[{time.time()-t0:6.1f}s] OK  {m}")
rng = np.random.default_rng(3)

# ---- 1) MIDAS 회귀: 진짜 Beta(1,8) 가중치 복원 ----------------------------------------
n, K = 400, 60
Lx = rng.standard_normal((n, K))
w_true = M.beta_weights(K, 1.0, 8.0)
Y = 0.5 + 2.0 * (Lx @ w_true) + 0.3 * rng.standard_normal(n)
fit = M.midas_fit(Y, [Lx])
assert abs(fit["theta"][0] - 8.0) < 2.0 and abs(fit["beta"][1] - 2.0) < 0.25, (fit["theta"], fit["beta"])
assert np.abs(fit["weights"][0] - w_true).max() < 0.02
ok(f"MIDAS 가중치 복원: θ2 = {fit['theta'][0]:.2f} (참 8), β = {fit['beta'][1]:.2f} (참 2)")

# ---- 2) hf_lags: 결정일 이하 최근 K개만 (미래 미포함) ------------------------------------
s = pd.Series(np.arange(100.0), index=pd.bdate_range("2020-01-01", periods=100))
d = pd.DatetimeIndex([s.index[50]])
assert np.array_equal(M.hf_lags(s, d, 5)[0], [50, 49, 48, 47, 46])
ok("고빈도 시차행렬: 결정일 당일 포함 과거 K개, 미래 미포함")

# ---- 3) GARCH-MIDAS: 장기 성분을 월간 요인이 결정하는 시뮬레이션 → θ>0 복원 -------------------
months = pd.period_range("2000-01", periods=240, freq="M")
Xm = pd.Series(np.cumsum(rng.standard_normal(240)) * 0.15, index=months.to_timestamp())
days = pd.bdate_range("2000-01-03", "2019-12-31")
mid = np.searchsorted(months.asi8, days.to_period("M").asi8)
K = 12; w = M.beta_weights(K, 1.0, 4.0)
xs = (Xm.values - Xm.values.mean()) / Xm.values.std()
lw = np.array([xs[m - K:m][::-1] @ w if m >= K else 0 for m in range(240)])
tau = np.exp(0.0 + 0.6 * lw)[mid]
g = np.ones(len(days)); r = np.zeros(len(days)); a, b = 0.06, 0.90
for t in range(1, len(days)):
    g[t] = (1 - a - b) + a * r[t - 1] ** 2 / tau[t - 1] + b * g[t - 1]
    r[t] = np.sqrt(tau[t] * g[t]) * rng.standard_normal()
gm = M.garch_midas(pd.Series(r, index=days), Xm, K=K, log_x=False)
th = gm["params"].loc["θ (장기 민감도)", "계수"]
assert 0.35 < th < 0.9 and gm["params"].loc["α", "계수"] < 0.12, gm["params"]
ok(f"GARCH-MIDAS 복원: θ = {th:.2f} (참 0.6), α = {gm['params'].loc['α','계수']:.3f} (참 0.06), β = {gm['params'].loc['β','계수']:.3f} (참 0.90), VR = {gm['VR']:.2f}")

# ---- 4) 실데이터: GARCH-MIDAS (변동성 예측 경주는 ◆ 리스크 risk.vol_models로 일원화 → tests/test_risk.py) ----
L = A.load_levels(); cat = A.load_catalog(); U = dict(zip(cat.id, cat.unit))
gm = M.garch_midas(A.make_panel(L, ["yf:CL=F"], U, "B", "logret", "2001-01-01")["yf:CL=F"], None, K=24)
print(gm["params"].round(4).to_string(), f"\n   VR(장기 성분 비중) = {gm['VR']:.2f}")
ok("실데이터 GARCH-MIDAS (WTI, 월 실현분산 24개월)")
print("MIDAS TEST PASSED")
