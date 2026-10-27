"""④ 예측 확장: 멀티모달 설명변수 변환 · 백테스트 그리드 · Prophet 방식 + 회귀변수가 미래 정보를 쓰지 않는지 (미래 교란 → 과거 불변) 검사."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import predict_lab as PL

rng = np.random.default_rng(3)
idx = pd.date_range("2001-01-05", periods=420, freq="W-FRI")
ids = ["yf:ZC=F", "yf:ZW=F", "yf:CL=F", "yf:GC=F"]
C = idx[380]
# 1) 멀티모달 특성 → 분석 주기 (과거 표준화 + 마지막 값 유지)
S = pd.DataFrame(rng.normal(0, 1, (420, 4)), index=idx, columns=ids)
S2 = S.copy(); S2.loc[S2.index > C] += 5.0
F, F2 = {"갈래": {"특성": S}}, {"갈래": {"특성": S2}}
day = pd.bdate_range(idx[0], idx[-1])
a, b = PL.mm_item_frame(F, "yf:ZC=F", ["갈래 · 특성"], day), PL.mm_item_frame(F2, "yf:ZC=F", ["갈래 · 특성"], day)
assert np.allclose(a.loc[:C].values, b.loc[:C].values)
# 2) 그리드 한 품목: 미래 목표·특성 교란 → C가 속한 해 이전 예측 불변
P = pd.DataFrame(np.exp(4 + rng.normal(0, 0.03, (420, 4)).cumsum(0)), index=idx, columns=ids)
lp = np.log(P); y = 100 * (lp.shift(-4) - lp)["yf:ZC=F"]
own = pd.DataFrame({f"m{k}": (lp - lp.shift(k))["yf:ZC=F"] for k in (1, 4, 13)})
ext = pd.DataFrame({"x": rng.normal(0, 1, 420)}, index=idx)
cb = {PL.OWN: own, "ext": ext}
cut = pd.Timestamp(f"{C.year}-01-01")
y2 = y.copy(); y2.loc[y2.index >= cut] *= -3
ext2 = ext.copy(); ext2.loc[ext2.index >= cut] += 4
old = PL.GRID_START
PL.GRID_START = "2005-01-01"
r1 = PL._grid_item("yf:ZC=F", cb, {"s": [PL.OWN, "ext"]}, y, idx, ["Ridge", "LightGBM"], 0)
r2 = PL._grid_item("yf:ZC=F", {PL.OWN: own, "ext": ext2}, {"s": [PL.OWN, "ext"]}, y2, idx, ["Ridge", "LightGBM"], 0)
PL.GRID_START = old
for k in r1:
    p1, p2 = r1[k].loc[:cut - pd.Timedelta(days=1)], r2[k].loc[:cut - pd.Timedelta(days=1)]
    assert p1.notna().any() and np.allclose(p1.values, p2.values, equal_nan=True), k
# 3) Prophet 방식 + 회귀변수: 미래 가격·회귀변수 교란 → C까지 예측 불변
P2 = P.copy(); P2.loc[P2.index > C] *= 1.5
reg, reg2 = {"g": S}, {"g": S2}
a = PL.prophet_like_x(P, reg, start=str(idx[370].date())); b = PL.prophet_like_x(P2, reg2, start=str(idx[370].date()))
assert a.loc[:C].notna().values.any() and np.allclose(a.loc[:C].values, b.loc[:C].values, equal_nan=True)
print("PREDICT LAB TEST PASSED")
