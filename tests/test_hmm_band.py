"""무거래 구간·HMM 레짐별 회귀 검증: 포지션 규칙의 인과성, 회전율 감소, BIC 상태 수 선택, 예측 부호 다양성, 테스트 구간 미래정보 차단."""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import analytics as A

t0 = time.time()
def ok(m): print(f"[{time.time()-t0:6.1f}s] OK  {m}")
rng = np.random.default_rng(0)

# 1) 무거래 구간: c=0이면 sign, c>0이면 회전율 감소, t 시점 포지션은 p_1..p_t만 사용
p = rng.normal(0, 1, 2000)
assert np.array_equal(A.band_positions(p, 0.0), np.sign(p))
pos = A.band_positions(p, 0.5)
turn = lambda x: np.abs(np.diff(x)).sum()
assert turn(pos) < 0.7 * turn(np.sign(p))
p2 = p.copy(); p2[1500:] = rng.normal(0, 5, 500)
assert np.array_equal(A.band_positions(p2, 0.5)[:1500], pos[:1500])
small = np.abs(p) < 0.5 * pd.Series(p).expanding(20).std().values
assert np.all(pos[1:][small[1:]] == pos[:-1][small[1:]])           # 작은 예측에선 직전 포지션 유지
y = rng.normal(0, 1, 2000)
s0, s1 = A.strategy_stats(p, y, 52, 1, 0.1, 0.0), A.strategy_stats(p, y, 52, 1, 0.1, 0.5)
assert s1["연간 회전율"] < s0["연간 회전율"]
ok(f"무거래 구간 c=0.5: 회전율 {s0['연간 회전율']:.0f} → {s1['연간 회전율']:.0f}, 과거 예측만 사용, 작은 예측에서 포지션 유지")

# 2) HMM: 레짐마다 X→y 베타가 반대인 자료 → 레짐별 회귀가 부호를 맞추고, BIC가 상태 수를 고름, 예측 부호가 바뀜
n = 1500
reg = np.repeat(rng.integers(0, 2, n // 50), 50)[:n]
x = rng.normal(0, np.where(reg == 0, 1.0, 3.0), n)
yv = np.where(reg == 0, 0.5, -0.5) * x + rng.normal(0, 1, n)
idx = pd.bdate_range("2005-01-03", periods=n)
R = pd.DataFrame({"y": yv, "x": x}, index=idx)
R["y"] = R["y"].shift(1)          # y_{t+1} = β_t·x_t + e (build_design 이 y를 shift(-1) 해 x_t 로 y_{t+1} 예측)
R = R.dropna()
D = A.build_design(R, "y", ["x"], "predict", lags=1, horizon=1, own_lags=False)
o = A.run_model("Hidden Markov (HMM)", D, 252, mode="predict", hmm_states=0, hmm_cov="full")
K = o["extra"]["n_states"]
assert 2 <= K <= 5 and o["extra"]["bic"] is not None
pr = o["pred_te"] + D.y_mu
assert (np.sign(pr) > 0).mean() not in (0.0, 1.0)
hit = np.mean(np.sign(pr) == np.sign(D.y_te + D.y_mu))
assert hit > 0.55, hit
ok(f"HMM 레짐별 회귀: BIC 선택 {K}상태, 예측 부호 다양(롱 {np.mean(pr > 0):.0%}), 레짐 반대 베타 자료 적중률 {hit:.0%}")

# 3) 미래정보: 테스트 구간 뒷부분 X·y를 바꿔도 앞부분 예측 불변 (전방 필터 + 학습구간 회귀)
D2 = A.build_design(R, "y", ["x"], "predict", lags=1, horizon=1, own_lags=False)
cut = len(D2.y_te) // 2
D2.X_te[cut:] = rng.normal(0, 10, D2.X_te[cut:].shape); D2.y_te[cut:] = rng.normal(0, 10, len(D2.y_te) - cut)
o2 = A.run_model("Hidden Markov (HMM)", D2, 252, mode="predict", hmm_states=K, hmm_cov="full")
o1 = A.run_model("Hidden Markov (HMM)", D, 252, mode="predict", hmm_states=K, hmm_cov="full")
assert np.allclose(o1["pred_te"][:cut], o2["pred_te"][:cut])
ok("HMM 예측: 테스트 뒷부분을 바꿔도 앞부분 예측 불변")
print("HMM BAND TEST PASSED")
