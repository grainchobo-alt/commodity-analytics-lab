"""B-1 패널 모델 검증."""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import analytics as A, portfolio as P, panel as PN

t0 = time.time()
def ok(m): print(f"[{time.time()-t0:6.1f}s] OK  {m}")

# ---- 1) 합성 패널: 특성 c0가 다음기 수익을 예측 (IC 약 0.1), 나머지는 잡음 ------------------
rng = np.random.default_rng(5)
T, N, K = 500, 15, 6
dates = pd.bdate_range("2005-01-07", periods=T, freq="W-FRI")
idx = pd.MultiIndex.from_product([dates, [f"a{i}" for i in range(N)]], names=["date", "id"])
Xr = rng.uniform(-0.5, 0.5, (T * N, K))
y = 0.3 * Xr[:, 0] + rng.standard_normal(T * N)
pan = dict(X=pd.DataFrame(Xr, index=idx, columns=[f"c{j}" for j in range(K)]), y=pd.Series(y, index=idx))
for m in ["Pooled OLS", "Ridge", "Lasso", "PCR", "Gradient Boosting"]:
    wf = PN.walk_forward_panel(pan, m, list(pan["X"].columns), refit=52, min_train=150, n_jobs=1)
    p = wf["pred"]; yy = pan["y"].reindex(p.index)
    ic = pd.DataFrame({"p": p, "y": yy}).groupby(level="date").apply(lambda g: g["p"].corr(g["y"], method="spearman")).mean()
    assert ic > 0.04, (m, ic)
    if wf["importance"] is not None and m != "PCR":
        assert np.argmax(np.abs(wf["importance"].values)) == 0, (m, wf["importance"])
    print(f"   {m:18s} 평균 IC {ic:.3f}  (재학습 {wf['n_refits']}회)")
ok("합성 패널: 모든 모델이 진짜 특성(c0)을 포착 (IC>0), 선형 모델 계수 최대 = c0")

# ---- 2) 미래정보 차단: 학습에 쓰인 행은 목표가 결정일 이전에 실현된 것만 -----------------------
wf = PN.walk_forward_panel(pan, "Pooled OLS", ["c0"], refit=52, min_train=150, n_jobs=1)
pan2 = dict(X=pan["X"].copy(), y=pan["y"].copy())
cut = dates[300]
pan2["y"][pan2["y"].index.get_level_values("date") >= cut] += 50.0       # 미래 목표를 크게 교란
wf2 = PN.walk_forward_panel(pan2, "Pooled OLS", ["c0"], refit=52, min_train=150, n_jobs=1)
before = wf["pred"].index.get_level_values("date") <= cut
assert np.allclose(wf["pred"][before], wf2["pred"].reindex(wf["pred"].index)[before])
ok("미래 목표 교란 → 해당 시점 이전 예측 불변 (학습창 = 날짜 ≤ d_{k-1})")

# ---- 3) run_weights: 단일 종목 100% 매수, 비용 0 → 실행일 이후 가격비와 일치 -----------------
cat = A.load_catalog(); U = dict(zip(cat.id, cat.unit))
La, Lr = A.load_levels(roll_adjusted=True), A.load_levels(roll_adjusted=False)
dd = P.decision_dates(La.index[(La.index >= "2016-01-01") & (La.index <= "2016-12-31") & (La.index.dayofweek < 5)], "ME")
W = pd.DataFrame(0.0, index=dd, columns=["yf:GC=F", "yf:CL=F"]); W["yf:GC=F"] = 1.0
bt = P.run_weights(W, La, Lr, "etf", liq_mult=0, comm=0)
er = P.etf_excess_levels(La, ["yf:GC=F", "yf:CL=F"])
er = er.ffill(limit=5); er = er[er.index.dayofweek < 5]["yf:GC=F"]            # 엔진과 같은 영업일 달력
d1 = bt["ret"].index[-1]
e0 = er.index[er.index.get_loc(dd[0]) + 1]
assert np.isclose((1 + bt["ret"]).prod(), er.loc[d1] / er.loc[e0], rtol=1e-9), ((1 + bt["ret"]).prod(), er.loc[d1] / er.loc[e0])
ok("run_weights: 금 100% 보유 손익 = ETF 초과수익 지수 변화 (다음 거래일 실행)")

# ---- 3b) 계량 패널: FM·FE·섹터 기울기 (시뮬레이션) ---------------------------------------
ents = ["yf:ZC=F", "yf:ZS=F", "yf:ZW=F", "yf:CL=F", "yf:NG=F", "yf:HO=F", "yf:GC=F", "yf:SI=F", "yf:HG=F", "yf:KC=F"]
T2 = 400; d2 = pd.bdate_range("2005-01-07", periods=T2, freq="W-FRI")
ix = pd.MultiIndex.from_product([d2, ents], names=["date", "id"])
alpha = np.repeat(np.linspace(-1, 1, len(ents))[None, :], T2, 0).ravel()          # 품목 고정효과
x1 = rng.standard_normal(T2 * len(ents)) + 0.8 * alpha                              # x1이 α와 상관 → Pooled 편의
x2 = rng.standard_normal(T2 * len(ents))
lam = np.repeat(rng.standard_normal(T2), len(ents))                                 # 시간 효과
yv = 0.3 * x1 + alpha + lam + rng.standard_normal(T2 * len(ents))
pan_s = dict(X=pd.DataFrame({"x1": x1, "x2": x2}, index=ix), y=pd.Series(yv, index=ix), r_next=pd.Series(yv, index=ix))
po = PN.panel_ols(pan_s, ["x1", "x2"], fe="none", cluster="time")["table"]
fe2 = PN.panel_ols(pan_s, ["x1", "x2"], fe="twoway", cluster="twoway")["table"]
assert po.loc["x1", "계수"] > 0.5 and abs(fe2.loc["x1", "계수"] - 0.3) < 0.05 and abs(fe2.loc["x2", "t값"]) < 2.5, (po, fe2)
ok(f"패널 OLS: Pooled β1={po.loc['x1','계수']:.2f} (누락 α로 편의) → 양방향 FE β1={fe2.loc['x1','계수']:.3f} (참 0.3)")
fm = PN.fama_macbeth(pan_s, ["x1", "x2"], nw_lags=4)["table"]
assert fm.loc["x1", "NW t값"] > 3 and abs(fm.loc["x2", "NW t값"]) < 2.5
ok(f"Fama-MacBeth: x1 평균기울기 {fm.loc['x1','평균 기울기']:.2f} (NW t={fm.loc['x1','NW t값']:.1f}), x2 t={fm.loc['x2','NW t값']:.1f}")
sec = np.array([PN.SECTOR[e] for e in ix.get_level_values("id")])
y3 = np.where(sec == "곡물", 0.5, 0.0) * x2 + lam + rng.standard_normal(len(x2))
pan3 = dict(X=pd.DataFrame({"x2": x2, "x1": x1}, index=ix), y=pd.Series(y3, index=ix), r_next=pd.Series(y3, index=ix))
ss = PN.sector_slopes(pan3, ["x2", "x1"], fe="time", cluster="time")
assert abs(ss["coef"].loc["x2", "곡물"] - 0.5) < 0.07 and abs(ss["coef"].loc["x2", "에너지"]) < 0.08
assert ss["wald"].loc["x2", "p값"] < 0.001 and ss["wald"].loc["x1", "p값"] > 0.01
ok(f"섹터 기울기: x2 곡물 {ss['coef'].loc['x2','곡물']:.2f}/에너지 {ss['coef'].loc['x2','에너지']:.2f}, Wald p (x2 차이 있음) {ss['wald'].loc['x2','p값']:.1e}, (x1 동일) {ss['wald'].loc['x1','p값']:.2f}")
# walk-forward 품목 FE: α_i가 지속되면 FE 예측이 Pooled보다 나아야 하고, 미래 교란에 과거 예측 불변
wf0 = PN.walk_forward_panel(pan_s, "Pooled OLS", ["x1", "x2"], 52, 150, n_jobs=1, fe="none")
wf1 = PN.walk_forward_panel(pan_s, "Pooled OLS", ["x1", "x2"], 52, 150, n_jobs=1, fe="entity")
mse = lambda w: float(((pan_s["y"].reindex(w["pred"].index) - pan_s["y"].reindex(w["pred"].index).groupby(level="date").transform("mean")
                        - (w["pred"] - w["pred"].groupby(level="date").transform("mean"))) ** 2).mean())
assert mse(wf1) < mse(wf0)
pan_s2 = dict(X=pan_s["X"], y=pan_s["y"].copy()); pan_s2["y"][pan_s2["y"].index.get_level_values("date") >= d2[300]] += 99
wf1b = PN.walk_forward_panel(pan_s2, "Pooled OLS", ["x1", "x2"], 52, 150, n_jobs=1, fe="entity")
bb = wf1["pred"].index.get_level_values("date") <= d2[300]
assert np.allclose(wf1["pred"][bb], wf1b["pred"].reindex(wf1["pred"].index)[bb])
wfs = PN.walk_forward_panel(pan3, "Ridge", ["x2", "x1"], 52, 150, n_jobs=1, sector=True, fe="time")
assert len(wfs["pred"]) > 0
ok(f"walk-forward FE: 횡단면 MSE Pooled {mse(wf0):.3f} → 품목 FE {mse(wf1):.3f}, 미래 교란 시 과거 예측 불변, 섹터 기울기 옵션 실행")

# ---- 4) 실데이터: ETF 기준 13개 품목, 주간 ------------------------------------------------
pan = PN.build_panel(La, Lr, list(P.ETF_MAP), U, "W-FRI", "etf", "vol_cs")
X = pan["X"]
print(f"   패널: {X.index.get_level_values('date').nunique()}주 × 최대 {X.index.get_level_values('id').nunique()}종목 = {len(X):,}행, 특성 {X.shape[1]}개")
assert (X.abs() <= 0.5 + 1e-9).all().all()
for m in ["Ridge", "Gradient Boosting"]:
    wf = PN.walk_forward_panel(pan, m, PN.CHARS, refit=26, min_train=104)
    ev = PN.evaluate(pan, wf["pred"], La, Lr)
    print(f"   {m:18s} OOS R² {ev['oos_r2']:+.4f}  평균 IC {ev['ic_mean']:+.3f} (t={ev['ic_t']:+.2f})  롱숏 Sharpe {ev['ls_stats'].get('Sharpe (비용 후)', np.nan):+.2f}")
ok("실데이터 패널 walk-forward 실행")
# 미래 가격 교란 → 과거 특성 불변
La2 = La.copy(); fut = La2.index > "2020-06-30"
cols = [P.ETF_MAP[i][0] for i in P.ETF_MAP]
La2.loc[fut, cols] *= np.exp(rng.normal(0, .05, (fut.sum(), len(cols))).cumsum(0))
pan2 = PN.build_panel(La2, Lr, list(P.ETF_MAP), U, "W-FRI", "etf", "vol_cs")
pre = pan["raw"].index.get_level_values("date") < pd.Timestamp("2020-06-26")
a, b = pan["raw"][pre], pan2["raw"].reindex(pan["raw"].index)[pre]
assert np.allclose(a.values, b.values, equal_nan=True), "특성이 미래 가격에 의존"
ok("실데이터: 미래 가격 교란 → 과거 특성(원값) 불변")
print("PANEL TEST PASSED")
