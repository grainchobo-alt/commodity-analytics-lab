"""다변량 레짐: 패널 변환·전방 필터·walk-forward 미래정보 차단 검사."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import logging; logging.getLogger("hmmlearn").setLevel(logging.ERROR)
import numpy as np, pandas as pd
import regime_mv as RM

rng = np.random.default_rng(0)
n = 520
idx = pd.date_range("2010-01-01", periods=n, freq="W-FRI")
s = np.r_[np.zeros(200), np.ones(150), np.zeros(170)].astype(int)            # 두 국면: 평시 / 위기
X = pd.DataFrame({"y": rng.normal(np.where(s, -0.5, 0.2), np.where(s, 3, 1)),
                  "vix": rng.normal(np.where(s, 1.5, 0), 0.5)}, index=idx)
# 1) 국면 복원: 전 기간 추정에서 위기 구간(레짐1 = Y 약세)이 대부분 레짐1로 분류
d = RM.describe(X, "y", 2)
acc = (d["smooth"].values.argmax(1) == (1 - s)).mean()
assert max(acc, 1 - acc) > 0.9, acc
# 2) 전방 필터는 미래 관측과 무관: 뒤쪽 값을 바꿔도 앞쪽 필터 확률 불변
m = RM.fit_hmm(((X - X.mean()) / X.std()).values, 2)
Z = ((X - X.mean()) / X.std()).values
F1 = RM.filter_probs(m, Z); Z2 = Z.copy(); Z2[400:] += 5
F2 = RM.filter_probs(m, Z2)
assert np.allclose(F1[:400], F2[:400])
# 3) walk-forward: 마지막 해의 자료를 바꿔도 그 이전 해의 예측 E는 그대로
W1 = RM.walk_forward(X, "y", 2, "full", "W", 1)
X2 = X.copy(); X2.loc[X2.index.year == X2.index.year.max(), "y"] += 10
W2 = RM.walk_forward(X2, "y", 2, "full", "W", 1)
prev = W1.index.year < X.index.year.max()
assert W1.loc[prev, "E"].notna().sum() > 50
assert np.allclose(W1.loc[prev, "E"].astype(float).fillna(0), W2.loc[prev, "E"].astype(float).fillna(0))
S = RM.evaluate(W1, X["y"], 52, n_placebo=50)
assert {"롱숏 Sharpe", "위약 95%", "OOS R²"} <= set(S)
print("REGIME_MV TEST PASSED")
