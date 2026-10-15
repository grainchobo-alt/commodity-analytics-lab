"""미래정보(look-ahead) 누수 검사: 시점 C 이후의 가격을 무작위로 교란해도, C 이전의 모든 예측·특성·지표가 변하지 않아야 한다."""
import sys, time, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import analytics as A

t0 = time.time()
def ok(m): print(f"[{time.time()-t0:6.1f}s] OK  {m}")
cat = A.load_catalog(); U = dict(zip(cat.id, cat.unit)); L = A.load_levels()
ids = ["yf:ZC=F", "yf:ZS=F", "yf:ZW=F", "yf:CL=F", "yf:HG=F", "yf:GC=F", "fred:DCOILWTICO", "fred:VIXCLS"]
CUT = pd.Timestamp("2021-06-30")
L2 = L.copy()
rng = np.random.default_rng(7)
fut = L2.index > CUT
L2.loc[fut, ids] = L2.loc[fut, ids] * np.exp(rng.normal(0, 0.3, (fut.sum(), len(ids))).cumsum(axis=0) * 0.1)

for fq in ["W-FRI", "B"]:
    R1 = A.make_panel(L, ids, U, fq, "logret", "2005-01-01")
    R2 = A.make_panel(L2, ids, U, fq, "logret", "2005-01-01")
    pre = R1.index[R1.index <= CUT]
    assert np.allclose(R1.loc[pre], R2.loc[pre], equal_nan=True), "패널(클리핑 포함)이 미래 데이터에 의존"
    F1 = A.make_features(L, "yf:CL=F", U, fq, R1.index, A.FEATURE_KINDS)
    F2 = A.make_features(L2, "yf:CL=F", U, fq, R2.index, A.FEATURE_KINDS)
    assert np.allclose(F1.loc[pre], F2.loc[pre], equal_nan=True), "특성이 미래 데이터에 의존"
    ok(f"[{fq}] 패널·특성(모멘텀·변동성·계절성·COT·베이시스): C 이전 값 불변")
    feats = [i for i in ids if i != "yf:ZC=F"]
    D1 = A.build_design(R1, "yf:ZC=F", feats, "predict", lags=2, horizon=1, train_frac=0.6, xlag=1 if fq == "B" else 0)
    D2 = A.build_design(R2, "yf:ZC=F", feats, "predict", lags=2, horizon=1, train_frac=0.6, xlag=1 if fq == "B" else 0)
    assert D1.idx_te.min() < CUT, "교란 시점이 테스트 구간 안에 있어야 의미 있는 검사"
    models = [m for m in A.MODEL_LIST if m != "LSTM"] if fq == "W-FRI" else ["OLS", "Lasso", "MCP", "Kalman TVP", "Hidden Markov (HMM)"]
    for m in models:
        p1 = A.run_model(m, D1, A.PPY[fq], mode="predict")["pred_te"]
        p2 = A.run_model(m, D2, A.PPY[fq], mode="predict")["pred_te"]
        before = D1.idx_te < CUT - pd.Timedelta(days=7)
        assert np.allclose(np.asarray(p1)[before], np.asarray(p2)[before], atol=1e-9), f"{m}: C 이전 예측이 미래 교란에 반응 → 누수"
    ok(f"[{fq}] {len(models)}개 모델: 테스트 구간에서 C 이전 예측 불변 ({', '.join(models)})")
    # walk-forward
    p1, _, _ = A.walk_forward(D1, "Ridge", A.PPY[fq], 26 if fq == "W-FRI" else 126, 0.5)
    p2, _, _ = A.walk_forward(D2, "Ridge", A.PPY[fq], 26 if fq == "W-FRI" else 126, 0.5)
    b = p1.index < CUT - pd.Timedelta(days=7)
    assert np.allclose(p1[b], p2.reindex(p1.index)[b]), "walk-forward 누수"
    ok(f"[{fq}] walk-forward Ridge: C 이전 예측 불변")
# 공표 지연: EIA 현물은 5영업일 뒤에야 설계에 들어감
R = A.make_panel(L, ["yf:ZC=F", "fred:DCOILWTICO"], U, "B", "logret", "2015-01-01")
D = A.build_design(R, "yf:ZC=F", ["fred:DCOILWTICO"], "predict", lags=1)
assert any("(t-5)" in n for n in D.names), D.names
ok(f"공표 지연 반영: EIA WTI 현물 → {[n for n in D.names if 'DCOIL' in n]}")
print("LOOKAHEAD TEST PASSED")
