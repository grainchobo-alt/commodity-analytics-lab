"""롱숏 확장: 칼만 추세·칼만 페어·횡단면 비중이 미래 가격을 쓰지 않는지, 롱숏 비중이 달러 중립인지 검사."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import ls_extra as LX

rng = np.random.default_rng(3)
idx = pd.date_range("2006-01-06", periods=500, freq="W-FRI")
ids = ["yf:ZC=F", "yf:ZW=F", "yf:CL=F", "yf:HO=F", "yf:GC=F", "yf:SI=F", "yf:KC=F", "yf:LE=F"]
P = pd.DataFrame(np.exp(4 + rng.normal(0, 0.03, (500, len(ids))).cumsum(0)), index=idx, columns=ids)
C = idx[350]; P2 = P.copy(); P2.loc[P2.index > C] *= 1.7
# 1) 칼만 추세: 미래 가격 교란 → 과거 기울기 불변
k1, k2 = LX.kalman_trend(np.log(P["yf:ZC=F"])), LX.kalman_trend(np.log(P2["yf:ZC=F"]))
assert np.allclose(k1.loc[:C].values, k2.loc[:C].values, equal_nan=True)
# 2) 칼만 페어: 과거 포지션 불변
w1, w2 = LX.kalman_pair(P, "yf:CL=F", "yf:HO=F"), LX.kalman_pair(P2, "yf:CL=F", "yf:HO=F")
assert np.allclose(w1.loc[:C].values, w2.loc[:C].values)
# 3) 횡단면 비중: 롱 합 0.5 · 숏 합 −0.5 (달러 중립), 과거 비중 불변
S1, S2 = np.log(P).diff(26), np.log(P2).diff(26)
W1, W2 = LX.xs_weights(S1), LX.xs_weights(S2)
act = W1.abs().sum(1) > 0
assert np.allclose(W1[act].clip(lower=0).sum(1), 0.5) and np.allclose(W1[act].clip(upper=0).sum(1), -0.5)
assert np.allclose(W1.loc[:C].values, W2.loc[:C].values)
Ws = LX.xs_weights(S1, sector_neutral=True)
assert np.allclose(Ws[Ws.abs().sum(1) > 0].sum(1), 0.0)
print("LS EXTRA TEST PASSED")
