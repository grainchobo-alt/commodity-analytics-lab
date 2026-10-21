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
print("LS STRATEGIES TEST PASSED")
