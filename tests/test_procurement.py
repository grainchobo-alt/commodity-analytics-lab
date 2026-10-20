"""구매 타이밍 검증: 미래정보 차단, 물량 정확히 완료, k=0이면 균등 매수와 동일, 결정→체결 시차."""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import analytics as A, procurement as PR

t0 = time.time()
def ok(m): print(f"[{time.time()-t0:6.1f}s] OK  {m}")
La = A.load_levels(roll_adjusted=True); Lr = A.load_levels(roll_adjusted=False)
i = "yf:ZC=F"

# 1) 미래정보: 기준일 이후 가격을 바꿔도 기준일까지의 신호·점수 불변
p = La[i].dropna()
cut = pd.Timestamp("2018-06-29")
p2 = p.copy(); p2[p2.index > cut] *= np.linspace(0.5, 2.0, (p2.index > cut).sum())
s1, s2 = PR.signals(p), PR.signals(p2)
a, b = s1.loc[:cut], s2.loc[:cut]
assert np.allclose(a.fillna(-9).values, b.fillna(-9).values), "미래 가격이 과거 신호에 영향"
assert np.allclose(PR.score(s1).loc[:cut].fillna(-9), PR.score(s2).loc[:cut].fillna(-9))
ok("신호·점수: 기준일 이후 가격을 바꿔도 기준일까지 불변 (계절성은 올해 이전 연도만)")

# 2) 속도 규칙: 어떤 점수 경로든 마감 주에 남은 물량 전량 → 총 매수량 = 1
rng = np.random.default_rng(0)
for _ in range(200):
    rem, W = 1.0, int(rng.integers(2, 30))
    for t in range(W):
        q = PR.pace(rng.uniform(), rng.uniform(), rem, W - t, k=rng.uniform(0, 3), m_max=3.0)
        assert -1e-12 <= q <= rem + 1e-12
        rem -= q
    assert abs(rem) < 1e-12
ok("속도 규칙: 무작위 점수 200회 — 매주 0 ≤ 매수량 ≤ 잔량, 기간 말 잔량 0 (물량 정확히 완료)")

# 3) k=0 → 매주 균등 → 평균 매입가 = 균등 매수
B0 = PR.backtest(Lr[i], La[i], 13, 4, k=0.0)
assert np.allclose(B0["규칙"], B0["균등"]) and np.allclose(B0["절감_규칙"], 0)
B = PR.backtest(Lr[i], La[i], 13, 4)
assert (B["사후최저"] <= B[["규칙", "균등", "일괄"]].min(axis=1) + 1e-9).all()
assert (B["end"] > B.index).all()
Sm = PR.summarize(B, 13, 4, halflife_years=5)["규칙"]
assert np.isfinite(Sm["최근 가중 평균 (bp)"]) and np.isfinite(Sm["t값 (비중첩)"])
ok(f"k=0 = 균등 매수 (절감 0), 사후 최저 ≤ 모든 규칙, 기간 {len(B)}개 · 규칙 평균 절감 {B['절감_규칙'].mean():+.1f}bp")

# 4) 체결 시차: 매입가는 결정일(금) '다음' 거래일 실제 종가
pa = Lr[i].dropna()
first = B.index[0]
assert first.dayofweek != 4 or pa.index[pa.index.get_loc(first) - 1].dayofweek == 4
ok("체결 = 결정일 다음 거래일 (결정 시점 가격으로 체결하지 않음)")

# 5) 고도화 요소의 미래정보 차단: HMM 위기 확률(연 단위 재추정 + 필터), 상대가치(실제 시세 관계), MIDAS
cut2 = pd.Timestamp("2016-06-30")
pz = La[i].copy(); pz[pz.index > cut2] *= np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.03, (pz.index > cut2).sum())))
h1, h2 = PR.hmm_crisis_prob(La[i]), PR.hmm_crisis_prob(pz)
assert np.allclose(h1.loc[:cut2].fillna(-1), h2.loc[:cut2].fillna(-1))
Lr2 = Lr.copy(); Lr2.loc[Lr2.index > cut2, "yf:ZW=F"] *= 1.7
r1, r2 = PR.rel_series(Lr, i), PR.rel_series(Lr2, i)
g1 = PR.signals(La[i], rel=r1, rel_sign=-1)["relval"]; g2 = PR.signals(La[i], rel=r2, rel_sign=-1)["relval"]
assert np.allclose(g1.loc[:cut2].fillna(-1), g2.loc[:cut2].fillna(-1))
m1, m2 = PR.midas_var(La[i]), PR.midas_var(pz)
assert np.allclose(m1.loc[:cut2].fillna(-1), m2.loc[:cut2].fillna(-1))
ok("HMM 위기 확률·상대가치·MIDAS 위험: 기준일 이후 데이터를 바꿔도 기준일까지 불변")

# 6) 보유비용: 균등 매수 실효가 = 가격 × (1 + c·남은 주 평균), 일괄(첫 주) 매수에 가장 크게 부과
Bc = PR.backtest(Lr[i], La[i], 13, 4, carry_bp_week=10.0, k=0.0)
assert (Bc["일괄"] / B0["일괄"] - 1).round(6).eq(0.012).all()                 # 12주 × 10bp
assert np.allclose(Bc["규칙"], Bc["균등"])
ok("보유비용 10bp/주: 기간 초 일괄 매수 실효가 +1.2% (12주 보유), k=0이면 여전히 균등과 동일")

# 7) 철강 원료·구리 상대가치: 기준일 이후 DCE·철근·환율·금 가격을 바꿔도 기준일까지의 관계·신호 불변, 위안 품목 달러 환산
cut3 = pd.Timestamp("2020-06-30")
Lr3 = Lr.copy()
for c_ in ["ak:I0", "ak:RB0", "ak:J0", PR.FX_CNY, "yf:GC=F"]:
    Lr3.loc[Lr3.index > cut3, c_] *= 1.5
for j_ in PR.METAL_IDS:
    q1, q2 = PR.rel_series(Lr, j_), PR.rel_series(Lr3, j_)
    assert q1 is not None and len(q1.loc[:cut3]) > 500, j_
    assert np.allclose(q1.loc[:cut3].values, q2.loc[:cut3].values), j_
    sg_ = PR.REL_DEF[j_][1]
    v1 = PR.signals(La[j_], rel=q1, rel_sign=sg_)["relval"]; v2 = PR.signals(La[j_], rel=q2, rel_sign=sg_)["relval"]
    assert np.allclose(v1.loc[:cut3].fillna(-1), v2.loc[:cut3].fillna(-1)), j_
u1, u2 = PR.usd_t(Lr, "ak:JM0"), PR.usd_t(Lr3, "ak:JM0")
assert np.allclose(u1.loc[:cut3].dropna(), u2.loc[:cut3].dropna()) and 50 < u1.dropna().iloc[-1] < 1000
ok("철광석(SGX/DCE 차익)·점결탄(고로 마진)·구리(구리/금): 기준일 이후 데이터 교란에도 기준일까지 불변, 점결탄 달러 환산 정상")

# 8) 확장 품목 · 다변량 상대가치: 기준일 이후 상대 품목 가격을 바꿔도 기준일까지의 단일·다변량(분위 평균·공적분 잔차) 관계 불변
cut4 = pd.Timestamp("2018-06-29")
Lr4 = Lr.copy()
for c_ in ["yf:ZW=F", "yf:ZS=F", "yf:CL=F", "ak:C0", "yf:HO=F"]:
    Lr4.loc[Lr4.index > cut4, c_] *= np.exp(np.cumsum(np.random.default_rng(2).normal(0, 0.03, (Lr4.index > cut4).sum())))
for j_, opt in [("yf:ZC=F", PR.MULTI_AVG), ("yf:ZC=F", PR.MULTI_COINT), ("yf:CL=F", PR.MULTI_COINT), ("yf:HE=F", "돈육/옥수수 (hog-corn ratio)"),
                ("ak:Y0", PR.MULTI_COINT)]:
    (q1, s1), (q2, s2) = PR.rel_option(Lr, j_, opt), PR.rel_option(Lr4, j_, opt)
    assert q1 is not None and len(q1.loc[:cut4]) > 300 and s1 == s2, (j_, opt)
    assert np.allclose(q1.loc[:cut4].values, q2.reindex(q1.index).loc[:cut4].values), (j_, opt)
assert PR.rel_option(Lr, "yf:ZC=F", PR.MULTI_AVG)[0].between(0, 1).all()
ok("확장 품목·다변량 상대가치: 기준일 이후 교란에도 기준일까지 불변 (분위 평균 0~1)")

# 9) 벌크 일괄 구매: τ > 1 이면 항상 마감 주 매수 (= 마감 가격), τ < 0 이면 항상 첫 주(= 즉시), 미래 가격 교란에도 과거 창 결과 불변
Bm = PR.backtest_lump(Lr[i], La[i], 8, 4, tau=1.5)
assert (Bm["매수주"] == 8).all() and np.allclose(Bm["규칙"], Bm["마감"]) and not Bm["신호매수"].any()
Bi = PR.backtest_lump(Lr[i], La[i], 8, 4, tau=-1.0)
assert (Bi["매수주"] == 1).all() and np.allclose(Bi["규칙"], Bi["즉시"]) and np.allclose(Bi["규칙_vs_즉시"], 0)
B1 = PR.backtest_lump(Lr[i], La[i], 8, 4, tau=0.6)
cut5 = pd.Timestamp("2018-06-29"); La5, Lr5 = La.copy(), Lr.copy()
f5 = np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.03, (La5.index > cut5).sum())))
La5.loc[La5.index > cut5, i] *= f5; Lr5.loc[Lr5.index > cut5, i] *= f5
B2 = PR.backtest_lump(Lr5[i], La5[i], 8, 4, tau=0.6)
done = B1.index[B1.index < cut5 - pd.DateOffset(weeks=10)]
assert len(done) > 50 and np.allclose(B1.loc[done, "규칙"], B2.loc[done, "규칙"]) and (B1.loc[done, "매수주"] == B2.loc[done, "매수주"]).all()
WS = PR.wait_stats(Lr[i], PR.score(PR.signals(La[i])), 4)
assert WS["표본 수"].sum() > 500 and WS["더 싼 날 올 확률 (−1% 이상)"].between(0, 1).all()
assert PR.item_lookback("yf:ZM=F") == 252 and PR.item_lookback("yf:ZC=F") == 126
ok("벌크 일괄 구매: τ 극단값 = 마감·즉시와 일치, 기준일 이후 교란에도 끝난 창의 매수 시점·가격 불변, 점수대별 대기 통계")

# 10) 재고 기반 반복 구매: τ>1 이면 강제 매수만 (재고 바닥 직전) → 카고 수 ≈ 정기, 미래 교란에도 과거 매수 불변, 예측은 그 해 이전 자료로만
sg_ = PR.signals(La[i]); W1 = PR.weekly_frame(Lr[i], sg_, PR.score(sg_))
o_hi = PR.backtest_inventory(W1, 1.5, 4, 2.0, "2008-01-01")
assert (o_hi["buys"]["사유"] == "강제").all() and abs(o_hi["카고"] - o_hi["정기카고"]) <= 2
o_lo = PR.backtest_inventory(W1, -1.0, 4, 2.0, "2008-01-01", carry_bp_week=5.0)
assert o_lo["평균재고"] > o_hi["평균재고"] and o_lo["보유비용bp"] > 0 and abs(o_lo["카고"] - o_lo["정기카고"]) <= 3
sg5 = PR.signals(La5[i]); W5 = PR.weekly_frame(Lr5[i], sg5, PR.score(sg5))
b1_, b5_ = PR.backtest_inventory(W1, 0.5, 4, 2.0, "2008-01-01")["buys"], PR.backtest_inventory(W5, 0.5, 4, 2.0, "2008-01-01")["buys"]
pre_ = b1_.index[b1_.index < cut5 - pd.DateOffset(weeks=2)]
assert len(pre_) > 30 and (b1_.loc[pre_].index == b5_.loc[pre_].index).all() and np.allclose(b1_.loc[pre_, "px"], b5_.loc[pre_, "px"])
F1_, F5_ = PR.fc_walkforward(W1, 1), PR.fc_walkforward(W5, 1)
yr_ = F1_.index.year < cut5.year
assert F1_.p_down[yr_].notna().sum() > 100 and np.allclose(F1_.p_down[yr_].fillna(-1), F5_.p_down.reindex(F1_.index)[yr_].fillna(-1))
ok("재고 기반 반복 구매: 강제 매수만이면 카고 수 ≈ 정기, 일찍 사면 평균 재고·보유비용 ↑, 미래 교란에도 과거 매수·예측 불변")

T = PR.today_table(La, Lr, list(PR.BUY_IDS))
assert len(T) == len(PR.BUY_IDS) and T["매수 점수"].between(0, 100).all() and (T["4주 하단 90%"] < T["가격"]).all()
ok(f"오늘의 신호판: {len(T)}품목, 점수 0~100, 4주 범위 하단 < 현재가")
print("PROCUREMENT TEST PASSED")
