"""analytics.py 스모크/정합성 테스트: python -I tests/test_analytics.py"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import analytics as A

t0 = time.time()
def ok(msg): print(f"[{time.time()-t0:6.1f}s] OK  {msg}")

# ---- 합성 데이터로 정합성 확인 ----
rng = np.random.default_rng(0)
n, p = 1500, 8
X = rng.standard_normal((n, p)); beta = np.array([1.5, -1.0, 0.8, 0, 0, 0, 0, 0])
y = X @ beta + rng.standard_normal(n)
X = (X - X.mean(0)) / X.std(0); y = y - y.mean()
for pen, g in [("MCP", 3.0), ("SCAD", 3.7)]:
    b, lam, *_ = A.nonconvex_cv(X, y, pen, g)
    assert np.allclose(b[:3], beta[:3], atol=0.12), (pen, b)
    assert (np.abs(b[3:]) < 0.05).all(), (pen, b)
    ok(f"{pen} 계수 복원 {np.round(b, 2)}")
# Kalman TVP: β가 중간에 바뀌는 경우 추적
x = rng.standard_normal((800, 1)); bt = np.r_[np.ones(400), -np.ones(400)]
yk = x[:, 0] * bt + 0.3 * rng.standard_normal(800)
_, bs, _ = A.kalman_tvp(x, yk, q=1e-2)
assert bs[:350, 0].mean() > 0.8 and bs[450:, 0].mean() < -0.8
ok("Kalman TVP 구조변화 추적")

# ---- 부호전략 지표: 손으로 검산 가능한 예 ----
p = np.array([1, 1, -1, -1, 1.]); yv = np.array([2, -1, -3, 1, 2.])
st = A.strategy_stats(p, yv, ann=52, h=1, cost=0.1)
g = np.array([2, -1, 3, -1, 2.]); turn = np.array([1, 0, 2, 0, 2.]); net = g - turn * 0.1
assert np.isclose(st["부호전략 Sharpe"], g.mean() / g.std() * np.sqrt(52))
assert np.isclose(st["부호전략 Sharpe (비용 후)"], net.mean() / net.std() * np.sqrt(52))
assert np.isclose(st["방향 적중률"], 3 / 5) and np.isclose(st["롱 비중"], 3 / 5)
assert np.isclose(st["연간 회전율"], turn.mean() * 52)
st2 = A.strategy_stats(p, yv, ann=52, h=2)          # h=2 → [::2] 표본, 연율화 √(52/2)
g2 = (np.sign(p) * yv)[::2]
assert np.isclose(st2["부호전략 Sharpe"], g2.mean() / g2.std() * np.sqrt(26))
ok("부호전략 지표 검산 (비용·회전율·h>1 비겹침)")

# ---- 실데이터 ----
cat = A.load_catalog(); units = dict(zip(cat.id, cat.unit))
L = A.load_levels()
ids = ["yf:ZC=F", "yf:ZS=F", "yf:ZM=F", "yf:ZW=F", "yf:CL=F", "yf:HG=F", "yf:GC=F", "ak:macro_shipping_bdi", "fred:VIXCLS", "yf:DX-Y.NYB"]
R = A.make_panel(L, ids, units, "B", "logret", "2001-01-01")
ok(f"패널 {R.shape} {R.index.min().date()}~{R.index.max().date()}")
Rw = A.make_panel(L, ids, units, "W-FRI", "logret", "2001-01-01"); ok(f"주간 패널 {Rw.shape}")

ev, load, sc = A.pca(R); ok(f"PCA PC1 설명력 {ev[0]:.2%}")
ar = A.rolling_absorption(R, 120, 1, 10); ok(f"흡수비율 {len(ar)}점, 최근 {ar.iloc[-1]:.2f}")
apc = A.avg_pairwise_corr(Rw, 26); ok(f"평균쌍별상관 최근 {apc.dropna().iloc[-1]:.2f}")
tbl, tot = A.dy_spillover(Rw, 2, 10); ok(f"DY 전체 스필오버 {tot:.1f}%")
rdy = A.rolling_dy(Rw, 150, 2, 10, 13); ok(f"롤링 DY {len(rdy)}점")

r = R["yf:ZC=F"]
for k in ["GARCH", "GJR", "EGARCH"]:
    pt, vol, info, res = A.garch_fit(r, k)
    e, h = A.news_impact(res, k)
    ok(f"{k} gamma={info['gamma']:.4f} p={info['gamma_p']:.3f} persist={info['persistence']:.3f}")
lc = A.leverage_corr(R["yf:CL=F"]); ok(f"레버리지 상관 WTI {lc.corr().iloc[0,1]:.3f}")
prob, summ = A.markov_switching(Rw["yf:CL=F"], 2); ok("Markov\n" + summ.round(3).to_string())

feats = [i for i in ids if i != "yf:ZC=F"]
for mode in ["explain", "predict"]:
    D = A.build_design(R, "yf:ZC=F", feats, mode, lags=2, horizon=1)
    for m in A.MODEL_LIST:
        out = A.run_model(m, D, 252, epochs=5)
        mt = out["metrics"]
        if mode == "explain":   # 동시점 모드는 거래 불가 → 전략 지표 없음
            assert np.isnan(mt["부호전략 Sharpe"]), (m, mt["부호전략 Sharpe"])
        print(f"   {mode:8s} {m:18s} R2tr={mt['R² (학습)']:.3f} R2te={mt['R² (테스트)']:.3f} OOS={mt['OOS R² (vs 학습평균)']:.3f} hit={mt['방향 적중률']:.3f}")
    ok(f"모델 {mode}")

# ---- 특성: 미래정보 누수 검사 ----
Fw = A.make_features(L, "yf:ZC=F", units, "W-FRI", Rw.index, A.FEATURE_KINDS)
assert {"MOM 1M", "RVOL 1M", "SEAS sin1", "COT 상업HP z"} <= set(Fw.columns)
# 모멘텀 1M은 t까지의 최근 4주 수익률 합이어야 함 (t+1 이후 정보 미포함)
px = A._to_freq(L["yf:ZC=F"], "W-FRI"); rr = A.log_return(px)
d0 = Fw.index[500]
assert np.isclose(Fw.loc[d0, "MOM 1M"], rr.loc[:d0].iloc[-4:].sum())
# COT: 화요일 보고치는 다음 주 월요일 이후 날짜에만 반영
cot = pd.read_parquet(A.DATA / "cot.parquet"); c = cot[cot.name == "Corn"].sort_values("date")
rep = pd.Timestamp(c.date.iloc[-30]); fri = rep + pd.Timedelta(days=3)
v_fri, v_next = Fw["COT 상업HP z"].asof(fri), Fw["COT 상업HP z"].asof(fri + pd.Timedelta(days=7))
assert v_fri != v_next, "같은 주 금요일에는 직전 보고치, 다음 주 금요일에 새 보고치가 반영돼야 함"
ok("특성 생성: 모멘텀 시점 정합, COT 발표 지연 반영")
Dwf = A.build_design(Rw, "yf:ZC=F", [], "predict", 1, 1, True, 0.5, extra=Fw)
p_, y_, m_ = A.walk_forward(Dwf, "Ridge", 52, 52, 0.5)
assert len(p_) > 300 and np.isfinite(m_["OOS R² (vs 학습평균)"])
ok(f"walk-forward (옥수수 주간, 특성만): OOS R² {m_['OOS R² (vs 학습평균)']:.4f}, Sharpe {m_['부호전략 Sharpe']:.2f}, 재추정 {m_['재추정 횟수']}회")
st_ = A.stationarity_table(L, ["yf:ZC=F", "yf:CL=F"], units, "W-FRI", "2001-01-01")
assert (st_["수익률 ADF p"] < 0.05).all()
ok("정상성 표: 수익률 ADF 정상 판정")

fv = A.favar(Rw, Rw[["yf:ZC=F"]], k=3, H=12)
assert fv["oirf"].shape[0] == 13
ok(f"FAVAR p={fv['p']} nobs={fv['nobs']} FEVD(Y,h=12) from F: {fv['fevd'][-1, -1, :3].round(3)}")
print("ALL PASSED")
