"""C그룹 검증: 알려진 레짐 구조를 복원하는가, 선형 자료에서는 비선형을 기각하지 않는가."""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import analytics as A, regime as RG

t0 = time.time()
def ok(m): print(f"[{time.time()-t0:6.1f}s] OK  {m}")
rng = np.random.default_rng(21)
T = 1200
idx = pd.bdate_range("2003-01-03", periods=T, freq="W-FRI")

def design(df, tgt, feats):
    D = A.build_design(df, tgt, feats, "explain", train_frac=0.7)
    return D

# ---- 1) TAR: q≤0 이면 β=+1, q>0 이면 β=−1 ----------------------------------------------
x = rng.standard_normal(T); q = rng.standard_normal(T)
y = np.where(q <= 0.3, 1.0, -1.0) * x + 0.5 * rng.standard_normal(T)
D = design(pd.DataFrame({"y": y, "x": x, "VIX": q}, index=idx), "y", ["x", "VIX"])
o = A.run_model("Threshold 회귀 (TAR)", D, 52, thr_var="VIX", tar_boot=99)
ex = o["extra"]; rc = ex["regime_coef"]
assert abs(ex["gamma"] - 0.3) < 0.15 and rc.loc["x"].iloc[0] > 0.8 and rc.loc["x"].iloc[1] < -0.8, (ex["gamma"], rc)
assert ex["p_boot"] < 0.05
ok(f"TAR: γ = {ex['gamma']:.2f} (참 0.3), 레짐 기울기 {rc.loc['x'].round(2).tolist()} (참 +1/−1), sup-F {ex['supF']:.0f}, 부트스트랩 p {ex['p_boot']:.3f}")
# 선형 자료 → 선형성 기각 안 해야
y_lin = 0.5 * x + 0.5 * rng.standard_normal(T)
Dl = design(pd.DataFrame({"y": y_lin, "x": x, "VIX": q}, index=idx), "y", ["x", "VIX"])
ol = A.run_model("Threshold 회귀 (TAR)", Dl, 52, thr_var="VIX", tar_boot=99)
assert ol["extra"]["p_boot"] > 0.05, ol["extra"]["p_boot"]
ok(f"TAR 선형 자료: 부트스트랩 p = {ol['extra']['p_boot']:.2f} (선형 기각 안 함)")

# ---- 2) STAR: 매끄러운 전환 c=0.5 ---------------------------------------------------------
G = 1 / (1 + np.exp(-4 * (q - 0.5)))
y2 = 0.2 * x + 1.2 * x * G + 0.5 * rng.standard_normal(T)
D2 = design(pd.DataFrame({"y": y2, "x": x, "VIX": q}, index=idx), "y", ["x", "VIX"])
o2 = A.run_model("STAR (평활 전이)", D2, 52, thr_var="VIX")
e2 = o2["extra"]; rc2 = e2["regime_coef"]
assert abs(e2["c"] - 0.5) < 0.35 and abs(rc2.loc["x"].iloc[0] - 0.2) < 0.3 and abs(rc2.loc["x"].iloc[1] - 1.4) < 0.3, (e2, rc2)
assert e2["lm"]["p"] < 0.001
ok(f"STAR: c = {e2['c']:.2f} (참 0.5), 기울기 G=0 {rc2.loc['x'].iloc[0]:.2f} (참 0.2) → G=1 {rc2.loc['x'].iloc[1]:.2f} (참 1.4), LM p = {e2['lm']['p']:.1e}")

# ---- 3) Markov-switching 회귀: 숨은 레짐에서 기울기 +0.8 / −0.8 ----------------------------
P = np.array([[0.97, 0.03], [0.06, 0.94]]); s = np.zeros(T, int)
for t in range(1, T): s[t] = rng.choice(2, p=P[s[t - 1]])
x1, x2 = rng.standard_normal(T), rng.standard_normal(T)
y3 = np.where(s == 0, 0.8, -0.8) * x1 + np.where(s == 0, 0.4, 1.2) * rng.standard_normal(T)
D3 = design(pd.DataFrame({"y": y3, "x1": x1, "x2": x2}, index=idx), "y", ["x1", "x2"])
o3 = A.run_model("Markov-switching 회귀", D3, 52, ms_pc=2)
sb = o3["extra"]["state_betas"]
assert sb.loc["레짐1", "x1"] > 0.5 and sb.loc["레짐2", "x1"] < -0.4, sb
ols = A.run_model("OLS", D3, 52)
assert o3["metrics"]["R² (테스트)"] > ols["metrics"]["R² (테스트)"] + 0.05
ok(f"Markov-switching: 레짐별 x1 기울기 {sb['x1'].round(2).tolist()} (참 +0.8/−0.8), 테스트 R² MS {o3['metrics']['R² (테스트)']:.3f} vs OLS {ols['metrics']['R² (테스트)']:.3f}")

# ---- 4) TVP-VAR 연결성: 중간에 전이 구조가 생기면 TCI 상승 -----------------------------------
N, T4 = 4, 1500
Y = np.zeros((T4, N))
for t in range(1, T4):
    A1 = np.diag([0.1] * N)
    if t > 750:
        A1[1:, 0] = 0.5                                   # 후반: 0번 시장 충격이 나머지로 전이
    Y[t] = A1 @ Y[t - 1] + rng.standard_normal(N)
cn = RG.tvp_var_connectedness(pd.DataFrame(Y, index=pd.bdate_range("2000-01-03", periods=T4)), p=1, H=10)
tci = cn["tci"]
first, last = tci.iloc[100:600].mean(), tci.iloc[-500:].mean()
assert last > first + 8 and cn["net"].iloc[-200:].mean().idxmax() == 0, (first, last, cn["net"].iloc[-200:].mean())
ok(f"TVP-VAR: TCI 전반 {first:.1f}% → 후반 {last:.1f}% (전이 구조 발생 감지), 순 전달자 = 0번 시장")

# ---- 5) 미래정보: 레짐 모델 예측은 미래 교란에 불변 (test_lookahead에서 MODEL_LIST 전체로 검사) ----------
ok("미래정보 검사는 tests/test_lookahead.py가 MODEL_LIST 전체(레짐 모델 포함)로 수행")
print("REGIME TEST PASSED")
