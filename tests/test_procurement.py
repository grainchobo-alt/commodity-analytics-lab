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

T = PR.today_table(La, Lr, list(PR.BUY_IDS))
assert len(T) == 6 and T["매수 점수"].between(0, 100).all() and (T["4주 하단 90% ($/t)"] < T["가격 ($/t)"]).all()
ok("오늘의 신호판: 6품목, 점수 0~100, 4주 범위 하단 < 현재가")
print("PROCUREMENT TEST PASSED")
