"""대안 시계열 모델: Prophet 방식·소형 신경망 예측이 미래 가격을 쓰지 않는지 (미래 교란 → 과거 예측 불변) 검사."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import alt_ts as AT

rng = np.random.default_rng(5)
idx = pd.date_range("2006-01-06", periods=500, freq="W-FRI")
ids = ["yf:ZC=F", "yf:ZW=F", "yf:CL=F", "yf:HO=F", "yf:GC=F", "yf:SI=F", "yf:KC=F", "yf:LE=F"]
P = pd.DataFrame(np.exp(4 + rng.normal(0, 0.03, (500, len(ids))).cumsum(0)), index=idx, columns=ids)
C = idx[450]; P2 = P.copy(); P2.loc[P2.index > C] *= 1.7
P2.loc[P2.index > C] *= np.exp(rng.normal(0, 0.05, (int((P2.index > C).sum()), len(ids))))
# 1) Prophet 방식
a, b = AT.prophet_like(P, start=str(idx[440].date())), AT.prophet_like(P2, start=str(idx[440].date()))
assert a.loc[:C].notna().values.any() and np.allclose(a.loc[:C].values, b.loc[:C].values, equal_nan=True)
# 2) 소형 신경망 (학습 = 그해 1월 이전 자료, 입력 = t까지)
for k in ("itrans", "gnn", "patch", "gnn_x"):
    a, b = AT.nn_pred(P, k, epochs=1, first_year=C.year), AT.nn_pred(P2, k, epochs=1, first_year=C.year)
    assert a.loc[:C].notna().values.any() and np.allclose(a.loc[:C].values, b.loc[:C].values, equal_nan=True, atol=1e-5), k
# 2b) neuralforecast 정식판: 그해 첫 결정일까지 학습 · 결정일까지 입력
for k in ("PatchTST", "TFT"):
    a, b = AT.nf_pred(P, k, first_year=C.year, max_steps=5), AT.nf_pred(P2, k, first_year=C.year, max_steps=5)
    assert a.loc[:C].notna().values.any() and np.allclose(a.loc[:C].values, b.loc[:C].values, equal_nan=True, atol=1e-5), k
# 2c) 매크로 입력판: 미래 가격 + 미래 매크로를 함께 교란해도 과거 예측 불변
Mc = pd.DataFrame(rng.normal(0, 1, (500, 3)), index=idx, columns=["vix", "credit", "dxy4"])
Mc2 = Mc.copy(); Mc2.loc[Mc2.index > C] += 3.0
for k in ("tft", "itrans", "gnn"):
    a, b = AT.nn_pred(P, k, epochs=1, first_year=C.year, macro=Mc), AT.nn_pred(P2, k, epochs=1, first_year=C.year, macro=Mc2)
    assert a.loc[:C].notna().values.any() and np.allclose(a.loc[:C].values, b.loc[:C].values, equal_nan=True, atol=1e-5), "macro " + k
for k in ("TFT", "iTransformer"):
    a, b = AT.nf_pred(P, k, first_year=C.year, max_steps=5, macro=Mc), AT.nf_pred(P2, k, first_year=C.year, max_steps=5, macro=Mc2)
    assert a.loc[:C].notna().values.any() and np.allclose(a.loc[:C].values, b.loc[:C].values, equal_nan=True, atol=1e-5), "macro nf " + k
# 3) 매매 손익: 미래 교란 → C까지 손익 불변
pr = np.log(P) - np.log(P).shift(52)
for md in ("dir", "xs"):
    assert np.allclose(AT.signal_book(pr, P, md).loc[:C].values, AT.signal_book(np.log(P2) - np.log(P2).shift(52), P2, md).loc[:C].values)
print("ALT TS TEST PASSED")
