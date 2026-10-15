"""포트폴리오 엔진 검증: Newton 해 = 일반 최적화기(SLSQP) 해, 리스크 패리티 위험기여 균등, 비용 계산, 미래정보 차단."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from scipy.optimize import minimize
import portfolio as P
import analytics as A

t0 = time.time()
def ok(m): print(f"[{time.time()-t0:6.1f}s] OK  {m}")
rng = np.random.default_rng(3)

# ---- 1) Newton vs SLSQP (같은 목적함수) -------------------------------------------
eps = 1e-4
for trial in range(20):
    n = rng.integers(3, 12)
    G = rng.standard_normal((n, n)); S = G @ G.T / n * 1e-3 + np.eye(n) * 1e-4
    mu = rng.normal(0, 2e-3, n); w0 = rng.dirichlet(np.ones(n)); c = rng.uniform(2e-4, 1e-3, n); lam = rng.uniform(2, 20)
    for mode in ["long_only", "long_short"]:
        L = 1.0
        w, info = P.newton_mv(mu, S, w0, c, lam, mode, L, eps=eps)
        obj = lambda x: -(mu @ x - 0.5 * lam * x @ S @ x - c @ (np.sqrt((x - w0) ** 2 + eps * eps) - eps))
        if mode == "long_only":
            cons = [{"type": "eq", "fun": lambda x: x.sum() - 1}]; bnds = [(0, 1)] * n; x0 = np.full(n, 1 / n)
        else:
            cons = [{"type": "ineq", "fun": lambda x: L - (np.sqrt(x * x + eps * eps) - eps).sum()}]; bnds = None; x0 = np.zeros(n)
        ref = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"ftol": 1e-14, "maxiter": 1000})
        assert -obj(w) >= -ref.fun - 1e-10, (trial, mode, -obj(w), -ref.fun)
        if mode == "long_only":
            assert abs(w.sum() - 1) < 1e-9 and w.min() > -1e-12
        else:
            assert (np.sqrt(w * w + eps * eps) - eps).sum() <= L + 1e-9
ok("Newton 해의 목적함수값 ≥ SLSQP 해 (20개 무작위 문제 × 롱온리·롱숏), 제약 충족")

# 종목별 비중 상한 (|w_i| ≤ w_max) 포함 비교
for trial in range(15):
    n = rng.integers(4, 12)
    G = rng.standard_normal((n, n)); S = G @ G.T / n * 1e-3 + np.eye(n) * 1e-4
    mu = rng.normal(0, 3e-3, n); w0 = rng.dirichlet(np.ones(n)); c = rng.uniform(2e-4, 1e-3, n); lam = rng.uniform(1, 5)
    for mode, wm in [("long_only", 0.3), ("long_short", 0.2)]:
        w, info = P.newton_mv(mu, S, w0, c, lam, mode, 1.0, eps=eps, w_max=wm)
        obj = lambda x: -(mu @ x - 0.5 * lam * x @ S @ x - c @ (np.sqrt((x - w0) ** 2 + eps * eps) - eps))
        capc = {"type": "ineq", "fun": lambda x: np.sqrt(wm ** 2 + eps * eps) - np.sqrt(x * x + eps * eps)}
        if mode == "long_only":
            cons = [{"type": "eq", "fun": lambda x: x.sum() - 1}, capc]; bnds = [(0, 1)] * n; x0 = np.full(n, 1 / n)
        else:
            cons = [{"type": "ineq", "fun": lambda x: 1.0 - (np.sqrt(x * x + eps * eps) - eps).sum()}, capc]; bnds = None; x0 = np.zeros(n)
        ref = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"ftol": 1e-14, "maxiter": 2000})
        assert -obj(w) >= -ref.fun - 1e-10, (trial, mode, -obj(w), -ref.fun)
        assert (np.abs(w) <= wm + 1e-9).all(), (mode, w.max())
ok("비중 상한 포함: Newton ≥ SLSQP (15개 문제 × 롱온리 30%·롱숏 20% 상한), 상한 준수")

# 비용이 클수록 거래가 줄어드는지 (w가 w0에 가까워짐)
n = 6; G = rng.standard_normal((n, n)); S = G @ G.T / n * 1e-3 + np.eye(n) * 1e-4
mu = rng.normal(0, 3e-3, n); w0 = np.full(n, 1 / n)
d = [np.abs(P.newton_mv(mu, S, w0, np.full(n, cc), 5)[0] - w0).sum() for cc in [0, 5e-4, 2e-3, 1e-2]]
assert all(d[i] >= d[i + 1] - 1e-9 for i in range(3)), d
ok(f"거래비용 ↑ → 회전율 ↓: {np.round(d, 3)}")

# ---- 2) 리스크 패리티: 위험기여 균등 ------------------------------------------------
w, info = P.newton_risk_parity(S)
rcb = w * (S @ w); rcb = rcb / rcb.sum()
assert np.allclose(rcb, 1 / n, atol=1e-8), rcb
ok(f"리스크 패리티 위험기여 = 1/N (Newton {info['iter']}회)")

# ---- 3) 거래비용: 옥수수 450¢ → 1틱 12.5$ + 수수료 3$ / 22,500$ ----------------------
tc, rc = P.cost_frames(pd.DataFrame({"yf:ZC=F": [450.0]}))
assert np.isclose(tc.iloc[0, 0], (12.5 + 3) / 22500)
assert np.isclose(rc.iloc[0, 0], 5 * (12.5 + 6) / 22500)
ok(f"옥수수 편도 매매비용 {tc.iloc[0,0]*1e4:.1f}bp, 연간 롤비용 {rc.iloc[0,0]*1e4:.1f}bp")

# ---- 4) 미래정보 차단: 미래 가격을 바꿔도 과거 비중이 변하지 않아야 함 ---------------------
cat = A.load_catalog(); U = dict(zip(cat.id, cat.unit))
La, Lc = A.load_levels(roll_adjusted=True), A.load_levels(roll_adjusted=False)
ids = ["yf:ZC=F", "yf:ZS=F", "yf:CL=F", "yf:HG=F", "yf:GC=F"]
for basis, cols in [("futures", ids), ("etf", [P.ETF_MAP[i][0] for i in ids])]:
    b1 = P.backtest(La, Lc, ids, U, "W-FRI", "2015-01-01", "2020-12-31", signal="momentum", basis=basis)
    La2 = La.copy(); La2.loc["2019-01-01":, cols] *= np.exp(np.random.default_rng(0).normal(0, 0.2, La2.loc["2019-01-01":, cols].shape)).cumprod(axis=0)
    b2 = P.backtest(La2, Lc, ids, U, "W-FRI", "2015-01-01", "2020-12-31", signal="momentum", basis=basis)
    W1, W2 = b1["results"]["최적화(Newton)"]["W"], b2["results"]["최적화(Newton)"]["W"]
    cut = W1.index[W1.index < "2018-12-28"]
    assert np.allclose(W1.loc[cut], W2.loc[cut]), f"[{basis}] 미래 가격 변경이 과거 비중에 영향 → 누수"
    assert not np.allclose(W1.loc["2019-03-01":], W2.loc["2019-03-01":])
    ok(f"[{basis}] 미래 가격(2019~)을 교란해도 2018년까지 비중 불변 → 미래정보 누수 없음")
# ETF 기반 초과수익: GLD 초과수익 ≈ 금 선물 − 금융비용 (연 1~3%p 차이)
er = P.etf_excess_levels(La, ["yf:GC=F"])["yf:GC=F"]
j = pd.concat([er, La["yf:GC=F"]], axis=1).dropna(); yrs = (j.index[-1] - j.index[0]).days / 365.25
g_er, g_fut = (j.iloc[-1, 0] / j.iloc[0, 0]) ** (1 / yrs) - 1, (j.iloc[-1, 1] / j.iloc[0, 1]) ** (1 / yrs) - 1
assert 0 < g_fut - g_er < 0.04, (g_fut, g_er)
ok(f"ETF 기반 금 초과수익 {g_er:.2%} vs 연속월물 {g_fut:.2%} (차이 = 금융비용, 이론과 일치)")
# 실행일 = 결정일 다음 거래일
dd = P.decision_dates(La.index[La.index.dayofweek < 5], "W-FRI")
assert (dd.dayofweek == 4).mean() > 0.95
ok("주간 결정일 = 금요일(휴장 시 직전 거래일), 실행 = 다음 거래일 종가")
st = b1["results"]["최적화(Newton)"]["stats"]
ok(f"백테스트 실행: Sharpe(비용후) {st['Sharpe (비용 후)']:.2f}, 연간 비용 {st['연간 비용 부담']*100:.2f}%, Newton 평균 반복 {b1['diag']['iter'].mean():.0f}, 수렴률 {b1['diag']['converged'].mean():.0%}")
# 리밸런싱 주기 분리: 일간 손익 경로의 복리 = 보유기간 수익 (비용 0)
bw = P.backtest(La, Lc, ids, U, "B", "2016-01-01", "2016-12-31", signal="momentum", basis="etf", rebalance="ME", liq_mult=0, comm=0)
r = bw["results"]["동일가중 1/N"]
W = r["W"]; e0, e1 = W.index[2], W.index[3]
held = r["gross"][(r["gross"].index > e0) & (r["gross"].index <= e1)]
Pa = P.etf_excess_levels(La, ids)[[c for c in W.columns]]
rel = Pa.loc[e1] / Pa.loc[e0]
assert np.isclose((1 + held).prod() - 1, float(W.loc[e0] @ (rel - 1))), "일간 경로 복리 ≠ 보유기간 수익"
assert np.isclose(r["stats"]["연간 리밸런싱 횟수"], 12, atol=1.5)
ok(f"일간 데이터 + 월간 리밸런싱: 일간 손익 복리 = 보유기간 수익, 연 {r['stats']['연간 리밸런싱 횟수']:.0f}회 리밸런싱")
try:
    P.backtest(La, Lc, ids, U, "ME", "2016-01-01", "2016-12-31", rebalance="B"); raise AssertionError("검증 실패")
except ValueError:
    ok("월간 데이터 + 일간 리밸런싱 요청 → 오류 (데이터보다 잦은 리밸런싱 불가)")

# 모멘텀 기간: 12-1 표준 = 이전 정의(12개월 창에서 최근 1개월 제외)와 동일, 모든 기간이 계산됨
rh = np.log(P.etf_excess_levels(La, ids)).diff().resample("W-FRI").sum().loc["2010":"2016"].dropna(how="all")
m_std = P.momentum_alpha(rh, 52, 0.05, "12-1개월 (표준)")
k12, k1 = 52, 4; h = rh.iloc[-k12:]; sig = rh.iloc[-52:].std()
m_old = (0.05 * sig * (h.iloc[:-k1].sum() / (sig * np.sqrt(k12 - k1))).clip(-2, 2)).fillna(0)
assert np.allclose(m_std.values, m_old.values)
for sp in P.MOM_SPECS:
    assert np.isfinite(P.momentum_alpha(rh, 52, 0.05, sp).values).all(), sp
ok(f"모멘텀 기간 {len(P.MOM_SPECS)}종 계산, 12-1개월 = 기존 정의와 동일")

# 무거래 구간: 같은 신호에서 회전율·거래 횟수 감소 / 변동성 타기팅: 실현 변동성이 목표 근처
kw = dict(signal="momentum", basis="etf", mode="long_short", w_max=0.25)
bf = P.backtest(La, Lc, ids, U, "W-FRI", "2014-01-01", "2019-12-31", **kw)
bb = P.backtest(La, Lc, ids, U, "W-FRI", "2014-01-01", "2019-12-31", trade_rule="band", band=0.2, max_hold=63, **kw)
sf, sb = bf["results"]["최적화(Newton)"]["stats"], bb["results"]["최적화(Newton)"]["stats"]
assert sb["연간 회전율(편도 합)"] < sf["연간 회전율(편도 합)"] and sb["연간 리밸런싱 횟수"] < sf["연간 리밸런싱 횟수"]
ok(f"무거래 구간 20%: 회전율 {sf['연간 회전율(편도 합)']:.1f} → {sb['연간 회전율(편도 합)']:.1f}, 거래 {sf['연간 리밸런싱 횟수']:.0f} → {sb['연간 리밸런싱 횟수']:.0f}회/년")
bv = P.backtest(La, Lc, ids, U, "W-FRI", "2014-01-01", "2019-12-31", vol_model="MIDAS-RV", vol_target=0.10, max_lev=3.0, **kw)
rv_ = bv["results"]["최적화(Newton)"]["stats"]["연환산 변동성"]
assert 0.07 < rv_ < 0.13, rv_
ok(f"변동성 타기팅 10% (MIDAS-RV 예측): 실현 변동성 {rv_:.1%}")

# 미래정보: 일간 변동성 예측은 t 이후 데이터를 바꿔도 t까지 불변
import midas as MD
rr = (np.log(La["yf:CL=F"].dropna()).diff() * 100).dropna()["2008":"2016"]
rr = rr[rr.index.dayofweek < 5]
f1 = MD.daily_vol_forecasts(rr.rename("a"), h=5)
rr2 = rr.copy(); rr2.loc["2014-01-01":] *= 3
f2 = MD.daily_vol_forecasts(rr2.rename("b"), h=5)
cut = rr.index[rr.index < "2014-01-01"][-1]
assert np.allclose(f1.loc[:cut].fillna(-1).values, f2.loc[:cut].fillna(-1).values)
ok("일간 변동성 예측(EWMA·HAR·MIDAS): 이후 데이터 변경에 과거 예측 불변 (walk-forward)")

# 리스크 버짓(신호 기울기): τ=0 → 리스크 패리티, 위험 기여 비율 = 예산
rng_ = np.random.default_rng(0)
A_ = rng_.standard_normal((5, 5)); S_ = A_ @ A_.T / 5 + np.diag([0.5, 0.2, 0.1, 0.3, 0.05])
mu_ = np.array([0.02, -0.01, 0.005, 0.03, -0.02])
assert np.allclose(P.risk_budget_tilt(mu_, S_, 0.0, "long_only"), P.newton_risk_parity(S_)[0], atol=1e-8)
for tilt in (0.3, 1.0):
    w_ = P.risk_budget_tilt(mu_, S_, tilt, "long_short", 1.0)
    sg = np.sign(mu_); St = S_ * np.outer(sg, sg); y_ = np.abs(w_)
    rc = y_ * (St @ y_) / (y_ @ St @ y_)
    z_ = np.abs(mu_ / np.sqrt(np.diag(S_))); b_ = (1 - tilt) / 5 + tilt * z_ / z_.sum()
    assert np.allclose(rc, b_ / b_.sum(), atol=1e-6) and np.allclose(np.sign(w_), sg) and np.isclose(np.abs(w_).sum(), 1.0)
bl = P.backtest(La, Lc, ids, U, "W-FRI", "2014-01-01", "2016-12-31", signal="momentum", basis="etf", mode="long_only", rb_tilt=0.0)
assert np.allclose(bl["results"]["리스크 버짓(신호 기울기)"]["ret"].values, bl["results"]["리스크 패리티(Newton)"]["ret"].values)
ok("리스크 버짓: τ=0 = 리스크 패리티(백테스트 수익률 동일), 위험 기여 비율 = 예산(신호 강도 기울기), 롱숏 부호 = 신호 부호")
print("PORTFOLIO TEST PASSED")
