"""리스크 엔진 검증: 미래정보 차단(시변 λ·HMM·Kalman·GARCH·VaR·범위), 검정의 크기·검정력(시뮬레이션), 기여도 합."""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import analytics as A, risk as RK

t0 = time.time()
def ok(m): print(f"[{time.time()-t0:6.1f}s] OK  {m}")
La = A.load_levels(roll_adjusted=True)
r = RK.returns(La["yf:ZC=F"]).loc["2003":]

# 1) 미래정보: 기준일 이후 수익률을 바꿔도 기준일까지의 분산 예측(8개 모형)·λ 경로 불변
cut = pd.Timestamp("2016-06-30")
r2 = r.copy(); m_ = r2.index > cut
r2[m_] = r2[m_] * np.exp(np.random.default_rng(0).normal(0, 0.5, m_.sum()))
o1, o2 = RK.vol_models(r), RK.vol_models(r2)
for m in RK.VOL_MODELS:
    a, b = o1["S"][m].loc[:cut], o2["S"][m].loc[:cut]
    assert np.allclose(a.fillna(-1), b.fillna(-1), rtol=1e-8, atol=1e-10), m
    assert a.notna().sum() > 1000, m
for m in o1["lam"]:
    assert np.allclose(o1["lam"][m].loc[:cut].fillna(-1), o2["lam"][m].loc[:cut].fillna(-1)), m
ok("8개 변동성 모형 (롤링 λ·HMM 3상태·Kalman TVP·GARCH·HAR·MIDAS 포함): 기준일 이후 교란에도 기준일까지 예측·λ 불변")

lam = o1["lam"]["EWMA 롤링 λ (직전 3년)"].dropna()
assert lam.between(0.90, 0.999).all() and lam.nunique() > 3
hm = o1["hmm"]["lam"]; hm = hm[np.isfinite(hm)]
assert (hm >= 0.90).all() and (hm <= 0.999).all() and np.nanstd(hm) > 0
ok(f"시변 λ: 롤링 {lam.nunique()}개 값 (연도별), HMM λ는 범위 안에서 날마다 변함 (표준편차 {np.std(hm):.4f})")

# 2) VaR·ES 미래정보 + 4주 범위
for side in ["long", "short"]:
    for h in [1, 20]:
        V1 = RK.var_forecasts(r, o1["S"]["EWMA 품목별 λ (확장)"], 0.99, side, h)["table"]
        V2 = RK.var_forecasts(r2, o2["S"]["EWMA 품목별 λ (확장)"], 0.99, side, h)["table"]
        cols = [c for c in V1 if c.startswith(("VaR", "ES"))]
        a, b = V1.loc[:cut - pd.Timedelta(days=40), cols], V2.loc[:cut - pd.Timedelta(days=40), cols]
        assert np.allclose(a.fillna(-1).values, b.reindex(a.index).fillna(-1).values), (side, h)
sd = np.sqrt(o1["S"]["EWMA 품목별 λ (확장)"]).shift(-1)
sd2 = np.sqrt(o2["S"]["EWMA 품목별 λ (확장)"]).shift(-1)
assert np.allclose(RK.hday_band(r, sd, 20, asof=cut), RK.hday_band(r2, sd2, 20, asof=cut))
ok("VaR·ES (5방법 × 롱/숏 × 1·20일)와 4주 범위 분위: 기준일 이후 교란에도 그 이전 값 불변")

# 3) 크기·검정력 (시뮬레이션)
rng = np.random.default_rng(1)
idx = pd.bdate_range("2000-01-03", periods=6000)
zn = pd.Series(rng.normal(size=6000), idx); one = pd.Series(1.0, idx)
Vn = RK.var_forecasts(zn, one, 0.99, "long", 1, methods=["정규", "FHS (필터링 역사적)"])
Bn = RK.backtest_table(Vn)
assert (Bn["Kupiec p (적중률)"] > 0.01).all(), Bn
zt = pd.Series(rng.standard_t(3, 6000) / np.sqrt(3), idx)
Bt = RK.backtest_table(RK.var_forecasts(zt, one, 0.99, "long", 1, methods=["정규", "FHS (필터링 역사적)", "EVT-GPD (조건부 POT)"]))
assert Bt.loc["정규", "초과율"] > 0.013 and abs(Bt.loc["FHS (필터링 역사적)", "초과율"] - 0.01) < 0.004
assert abs(Bt.loc["EVT-GPD (조건부 POT)", "초과율"] - 0.01) < 0.004
ok(f"VaR: 정규 자료에서 Kupiec 통과, t(3) 꼬리에서 정규 초과율 {Bt.loc['정규', '초과율']:.2%} vs FHS {Bt.loc['FHS (필터링 역사적)', '초과율']:.2%} · EVT {Bt.loc['EVT-GPD (조건부 POT)', '초과율']:.2%}")

L = pd.DataFrame({"good": rng.normal(1.0, 1, 3000), "same": rng.normal(1.0, 1, 3000), "bad": rng.normal(1.3, 1, 3000)})
pv = RK.mcs(L, B=300)
assert pv["bad"] < 0.05 and pv[["good", "same"]].min() >= 0.10, pv
dm = RK.dm_table(L, "good")
assert dm.loc["bad", "DM t"] > 3 and abs(dm.loc["same", "DM t"]) < 3
ok(f"MCS: 동등한 두 모형 포함 (p={pv['good']:.2f}, {pv['same']:.2f}), 열등 모형 제외 (p={pv['bad']:.3f}); DM 방향 정상")

e = np.zeros(5000); s2 = 1.0
for t in range(1, 5000):                                            # 음(−) 충격 후 변동성 ↑
    s2 = 0.05 + 0.85 * s2 + (0.2 if e[t - 1] < 0 else 0.0) * e[t - 1] ** 2
    e[t] = np.sqrt(s2) * rng.normal()
zs = e[1:] / np.sqrt(RK.ewma_path(e, 0.94)[1:])
assert RK.sign_bias(zs)["p"] < 0.01 and RK.sign_bias(rng.normal(size=5000))["p"] > 0.001
ok("Engle-Ng 부호 편향: 비대칭 GARCH 자료에서 기각, 대칭 백색잡음에서 미기각")

def sim(lam_, n=4000):
    x = np.zeros(n); v = 1.0
    for t in range(1, n):
        v = lam_ * v + (1 - lam_) * x[t - 1] ** 2; x[t] = np.sqrt(v) * rng.normal()
    return x
LLa, LLb, LLc = RK.ewma_ll_grid(sim(0.92)), RK.ewma_ll_grid(sim(0.99)), RK.ewma_ll_grid(sim(0.92))
assert RK.lr_common_lambda({"a": LLa, "b": LLb})["p"] < 0.01
assert RK.lr_common_lambda({"a": LLa, "c": LLc})["p"] > 0.001
ok("우도비 공통 λ: λ=0.92 vs 0.99 기각, 같은 λ 미기각")

# 4) 물량 위험: 오일러 기여도 합 = 정규 VaR, 숏 노출 손실 = 가격 상승
ids = ["yf:ZC=F", "yf:ZS=F"]
OO = {i: RK.vol_models(RK.returns(La[i]), ["EWMA 품목별 λ (확장)"]) for i in ids}
Rp = pd.DataFrame({i: OO[i]["r"] for i in ids}).dropna()
Sp = pd.DataFrame({i: OO[i]["S"]["EWMA 품목별 λ (확장)"] for i in ids}).reindex(Rp.index).dropna()
P = RK.position_risk(Rp.loc[Sp.index], Sp, pd.Series({"yf:ZC=F": -5e6, "yf:ZS=F": -3e6}), 0.99, 20)
assert abs(P["table"]["VaR 기여 ($, 정규)"].sum() - P["var_normal"]) < 1e-6 * P["var_normal"]
assert 0 < P["var_normal"] <= P["undiversified"] + 1e-6 and P["es_fhs"] >= P["var_fhs"]
ok(f"물량 위험: 기여도 합 = 정규 VaR ${P['var_normal']:,.0f} ≤ 단순 합 ${P['undiversified']:,.0f}, FHS ES ≥ VaR")
print("RISK TEST PASSED")
