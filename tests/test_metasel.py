"""모델 선택 메타 전략(metasel) 검사.
1) 미래정보 교란: 시점 C 이후 가격을 교란해도 C 이전의 후보 OOS 예측·레짐 라벨·메타 예측·선택 모델이 변하지 않아야 한다.
2) 손실 교란: 후보 예측·y를 C 이후에만 바꿔도 C 이전 메타 선택이 그대로여야 한다 (shift(h) 확인, 합성 자료).
3) SPA: 귀무(평균 0)에서는 기각하지 않고, 뚜렷한 우위 전략은 찾아내야 한다."""
import sys, time, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import analytics as A
import metasel as MS

t0 = time.time()
def ok(m): print(f"[{time.time()-t0:6.1f}s] OK  {m}")

# ---- 2) 합성 자료: 손실은 shift(h) 뒤에만 사용 ------------------------------------------------
rng = np.random.default_rng(0)
idx = pd.bdate_range("2004-01-01", periods=1500)
y = pd.Series(rng.normal(0, 1, len(idx)), idx)
P = pd.DataFrame(rng.normal(0, 0.3, (len(idx), 4)), idx, columns=list("abcd"))
G = pd.DataFrame({"vol": rng.integers(0, 3, len(idx)).astype(float)}, idx)
C = idx[1000]
for h in (1, 5):
    M1, K1 = MS.meta_predictions(P, y, h, 252, 3, 3, G)
    y2, P2, G2 = y.copy(), P.copy(), G.copy()
    y2[y2.index >= C] = rng.normal(5, 3, (y2.index >= C).sum())          # y_s 는 s+h 에 실현 → s ≥ C 바꾸면 t < C+h 영향 없어야
    P2[P2.index > C] = rng.normal(0, 5, P2[P2.index > C].shape)
    G2[G2.index > C] = 2.0
    M2, K2 = MS.meta_predictions(P2, y2, h, 252, 3, 3, G2)
    pre = idx[idx < C]
    assert np.allclose(M1.loc[pre], M2.loc[pre], equal_nan=True), f"h={h}: 메타 예측이 미래 손실에 반응"
    assert (K1.loc[pre] == K2.loc[pre]).all().all(), f"h={h}: 선택 모델이 미래 손실에 반응"
    # 경계: C 시점 예측은 y_{C-h}까지만 → y_C 변경이 C..C+h-1 의 선택에 영향 없어야
    edge = idx[(idx >= C) & (idx < idx[1000 + h])]
    P3 = P.copy(); y3 = y.copy(); y3[y3.index >= C] += 100
    M3, _ = MS.meta_predictions(P3, y3, h, 252, 3, 3, G)
    sel = [c for c in M1.columns if not c.startswith("(e)")]
    assert np.allclose(M1.loc[edge, sel], M3.loc[edge, sel]), f"h={h}: 실현 전 손실(y_C)을 사용"
ok("합성: 메타 예측·선택은 실현이 끝난 손실(s ≤ t−h)만 사용 (h=1, 5)")

# ---- 3) SPA 크기·검정력 ----------------------------------------------------------------------
pn = [MS.spa_test(rng.normal(0, 1, (2000, 8)), B=300, seed=s)["SPA_p"] for s in range(10)]
assert np.mean(np.array(pn) < 0.05) <= 0.2, pn
d = rng.normal(0, 1, (2000, 8)); d[:, 2] += 0.12
r = MS.spa_test(d, B=300)
assert r["SPA_p"] < 0.05 and r["RC_p"] < 0.05 and r["최고"] == 2, r
ok(f"SPA: 귀무 기각률 {np.mean(np.array(pn) < 0.05):.0%}, 우위 전략 p={r['SPA_p']:.3f}")

# 위약: 항상 롱인 전략의 위약 = Buy&Hold 그 자체
F = pd.DataFrame({"long": np.ones(len(idx))}, idx)
pl = MS.placebo(F, y, 1, 252, 0.0, B=50)
assert abs(pl["table"].loc["long", "위약 중앙값"] - pl["table"].loc["long", "Sharpe (비용 후)"]) < 1e-9
ok("위약: 롱 비중 100% 전략의 위약 Sharpe = Buy&Hold")

# ---- 1) 실제 자료: 가격 교란 ------------------------------------------------------------------
cat = A.load_catalog(); U = dict(zip(cat.id, cat.unit)); L = A.load_levels(roll_adjusted=True)
tgt = "yf:ZC=F"; feats = ["yf:ZS=F", "yf:ZW=F", "yf:CL=F", "fred:VIXCLS"]; corr_ids = ["yf:ZC=F", "yf:ZS=F", "yf:ZW=F", "yf:CL=F"]
CUT = pd.Timestamp("2016-06-30")
L2 = L.copy()
fut = L2.index > CUT
ids = list(dict.fromkeys([tgt] + feats + corr_ids))
L2.loc[fut, ids] = L2.loc[fut, ids] * np.exp(rng.normal(0, 0.3, (fut.sum(), len(ids))).cumsum(axis=0) * 0.1)
res = []
for LL in (L, L2):
    R = A.make_panel(LL, [tgt] + feats, U, "W-FRI", "logret", "2000-01-01")
    Rd = A.make_panel(LL, corr_ids, U, "B", "logret", "2000-01-01")
    D = A.build_design(R, tgt, feats, "predict", 2, 1, True, 0.7)
    Pm, yy = MS.candidate_paths(D, ["OLS", "Ridge", "Kalman TVP", "Gradient Boosting"], 52, 26, oos_start="2006-01-01")
    Gm = MS.regimes(LL[tgt], Rd, Pm.index)
    Mm, Km = MS.meta_predictions(Pm, yy, 1, 52, 6, 6, Gm, min_obs=30)
    res.append((Pm, Gm, Mm, Km))
pre = res[0][0].index[res[0][0].index < CUT - pd.Timedelta(days=7)]
assert len(pre) > 300 and res[0][0].index.max() > CUT
for nm, a, b in [("후보 OOS 예측", res[0][0], res[1][0]), ("레짐 라벨", res[0][1], res[1][1]), ("메타 예측", res[0][2], res[1][2])]:
    assert np.allclose(a.loc[pre].values.astype(float), b.reindex(a.index).loc[pre].values.astype(float), equal_nan=True, atol=1e-9), f"{nm}: C 이전 값이 미래 교란에 반응 → 누수"
assert (res[0][3].loc[pre] == res[1][3].loc[pre]).all().all(), "선택 모델이 미래 교란에 반응"
assert res[0][1].loc[pre].notna().mean().min() > 0.5, "레짐 라벨이 대부분 NaN"
ok(f"가격 교란(C={CUT.date()}): 후보 4모델 OOS·레짐(HMM·변동성·상관)·메타 예측·선택 모두 C 이전 불변 ({len(pre)}주)")
print("METASEL TEST PASSED")
