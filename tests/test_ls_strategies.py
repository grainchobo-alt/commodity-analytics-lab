"""롱숏 전략: 미래 가격을 바꿔도 과거 비중은 그대로인지(미래정보 차단)와 손익 시점 정렬 검사."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import ls_strategies as LS

rng = np.random.default_rng(1)
idx = pd.date_range("2005-01-07", periods=600, freq="W-FRI")
f = rng.normal(0, 2, 600).cumsum()
P = pd.DataFrame({"yf:CL=F": np.exp(4 + 0.01 * f + rng.normal(0, 0.03, 600).cumsum() * 0.3),
                  "yf:HO=F": np.exp(1 + 0.01 * f + rng.normal(0, 0.03, 600)),
                  "yf:GC=F": np.exp(7 + rng.normal(0, 0.02, 600).cumsum())}, index=idx)
C = idx[450]
P2 = P.copy(); P2.loc[P2.index > C] *= 1.5
for fn in [lambda X: LS.pair_weights(X, "yf:CL=F", "yf:HO=F", rho_min=0.3)[0], lambda X: LS.tsmom_weights(X, 52), lambda X: LS.xsmom_weights(X, 26)]:
    W1, W2 = fn(P), fn(P2)
    assert np.allclose(W1.loc[:C].values, W2.loc[:C].values), "미래 가격이 과거 비중에 영향"
R = LS._ret(P)
W = pd.DataFrame(0.0, index=idx, columns=P.columns); W.iloc[100, 0] = 1.0
r = LS._book(W, R, 0.0)
assert abs(r.loc[idx[101]] - R.iloc[101, 0]) < 1e-12 and r.drop(idx[101]).abs().sum() == 0   # t 결정 → t+1 손익
ar = LS.absorption(R, 52)
assert ar.dropna().between(0, 1).all()
# 캐리: 월물 교체 주의 수익률은 '교체 전 차월물 → 교체 후 근월물'(같은 계약) 기준
d = pd.date_range("2020-01-03", periods=4, freq="W-FRI")
Cx = pd.DataFrame({"date": d, "id": "ak:M0", "f1": [100, 101, 110, 111], "f2": [105, 106, 112, 113], "months": 2.0,
                   "c1": [2001, 2001, 2003, 2003], "c2": [2003, 2003, 2005, 2005]})
K, Rr = LS.carry_panel(Cx, pd.DataFrame())
assert abs(Rr["ak:M0"].iloc[2] - 100 * np.log(110 / 106)) < 1e-9 and abs(Rr["ak:M0"].iloc[1] - 100 * np.log(101 / 100)) < 1e-9
assert K["ak:M0"].iloc[0] < 0                                            # 콘탱고(f2 > f1) → 캐리 음수
print("LS STRATEGIES TEST PASSED")
