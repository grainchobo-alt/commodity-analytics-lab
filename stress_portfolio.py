"""포트폴리오 스트레스 테스트 — 지금의 50/25/25 혼합(품목별 순비중)에 VaR · 과거 위기 재현 · 가상 충격 · 상관 스트레스 · 역스트레스를 적용.

VaR와 스트레스 테스트의 차이
  VaR/ES  : '보통의 나쁜 날' — 최근 변동성 · 상관으로 본 손실 분포의 99% 분위(VaR)와 그 너머 평균(ES). 포트폴리오 전체 = 품목별 노출 × 공분산(대시보드 ◆ '구매·보유 물량 위험'과 같은 risk.position_risk, FHS · 오일러 기여도).
  스트레스 : '분포 밖의 날' — 확률을 묻지 않고 "그 일이 다시 일어나면 지금 장부는 얼마를 잃나"를 직접 계산.
구성
  ① 지금 품목별 순비중(자본 대비 %): MinVar · 52주 모멘텀 · 가치 각각 목표 10%로 맞춘 비중을 50/25/25로 섞고, 혼합 배율(직전 26주 실현 변동성으로 목표 10%, 상한 4배)까지 반영
  ② 포트폴리오 VaR · ES (1 · 5 · 20 영업일, 99%) + 품목별 기여도(합 = 정규 VaR)
  ③ 과거 위기 재현: 그 구간의 실제 품목 가격 변화를 '지금 비중'에 그대로 적용 (대시보드 위기 구간 + 짧고 급한 구간)
     + 같은 구간에 이 전략이 실제로 낸 손익(그때 비중) — 둘이 다른 이유: 전략은 그 사이 비중을 바꿈
  ④ 가상 충격: 곡물 · 에너지 · 귀금속 · 축산 급락/급등, 전 품목 동반 ±15%
  ⑤ 상관 스트레스: 품목 간 상관을 위기 수준(0.6 · 0.9)으로 올렸을 때 VaR
  ⑥ 역스트레스: 자본 10%를 잃으려면 전 품목이 같은 방향으로 몇 % 움직여야 하나 · 한 품목만 움직이면 몇 %
실행: .venv\\Scripts\\python.exe stress_portfolio.py [자본 $, 기본 1천만] → data/stress_portfolio.csv
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
import analytics as A  # noqa: E402
import ls_extra as LX  # noqa: E402
import ls_strategies as LS  # noqa: E402
import risk as RK  # noqa: E402

ASOF = "2026-10-26"
SHARP = {"2008 하반기 원자재 폭락": ("2008-07-01", "2008-12-31"), "2011-09 위험자산 투매": ("2011-08-31", "2011-10-04"),
         "2012 미국 가뭄(곡물 급등)": ("2012-06-01", "2012-08-31"), "2014–15 유가 폭락": ("2014-07-01", "2015-01-31"),
         "2020 코로나 급락": ("2020-02-20", "2020-04-30"), "2022-03 전쟁 급등": ("2022-02-24", "2022-03-08"),
         "2022 하반기 급반락": ("2022-06-08", "2022-07-15")}
GROUPS = {"곡물": ["yf:ZC=F", "yf:ZW=F", "yf:ZS=F", "yf:ZM=F", "yf:ZL=F"], "에너지": ["yf:CL=F", "yf:HO=F", "yf:RB=F", "yf:NG=F"],
          "귀금속": ["yf:GC=F", "yf:SI=F", "yf:PL=F"], "축산": ["yf:LE=F", "yf:HE=F"]}
SHOCKS = {"곡물 −20%": {"곡물": -20}, "곡물 +20% (가뭄)": {"곡물": 20}, "에너지 −30%": {"에너지": -30}, "에너지 +30%": {"에너지": 30},
          "귀금속 −15%": {"귀금속": -15}, "축산 −20%": {"축산": -20}, "전 품목 동반 −15%": {"*": -15}, "전 품목 동반 +15%": {"*": 15}}


def name(i: str) -> str:
    from llm import tools as T
    return T.name_of(i)


def current_weights(L: pd.DataFrame, tgt: float = 10.0) -> tuple[pd.Series, pd.Series, float]:
    """품목별 순비중(자본 대비 비율), 혼합 배율, 마지막 주. ls_extra.item_contrib와 같은 규칙."""
    P = LS.weekly(L, LS.BASKET, "2001-01-01"); P = P[[c for c in LS.BASKET if c in P]]; R = LS._ret(P)
    lp = np.log(P)
    Ws = {"MinVar": LS._vol_target(LS.alloc_weights_cap(P, 0.25), R, tgt),
          "모멘텀": LS._vol_target(LS.tsmom_weights(P, 52, port_tgt=None), R, tgt),
          "가치": LS._vol_target(LX.xs_weights(-(lp - lp.rolling(260, min_periods=156).mean())), R, tgt)}
    mix = {"MinVar": 0.5, "모멘텀": 0.25, "가치": 0.25}
    raw = sum(w * (Ws[k].shift(1) * R).fillna(0).sum(1) for k, w in mix.items())
    k = float((tgt / (raw.rolling(26, min_periods=13).std().shift(1) * np.sqrt(52))).clip(upper=4.0).fillna(1.0).iloc[-1])
    net = sum(w * Ws[k_].iloc[-1] for k_, w in mix.items()) * k
    return net[net.abs() > 1e-4], pd.Series({k_: (Ws[k_].iloc[-1] * k * w).abs().sum() for k_, w in mix.items()}), k


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cap = float(sys.argv[1]) if len(sys.argv) > 1 else 1e7
    L = A.load_levels(roll_adjusted=True).loc[:ASOF]
    w, gross_by, k = current_weights(L)
    expo = w * cap
    rows = []
    print(f"기준일 {ASOF} · 자본 ${cap:,.0f} · 혼합 배율 {k:.2f} · 총 노출 {w.abs().sum():.2f}배 (롱 {w[w>0].sum():.2f} · 숏 {w[w<0].sum():.2f})")
    print("품목별 순비중(%):", ", ".join(f"{name(i)} {v*100:+.1f}" for i, v in w.sort_values(key=abs, ascending=False).items()))
    # ② VaR · ES
    ids = list(w.index)
    Rd = pd.DataFrame({i: RK.returns(L[i]) for i in ids}).dropna(how="all")
    Sd = pd.DataFrame({i: pd.Series(RK.ewma_path(Rd[i].dropna().values, RK.RM_LAMBDA), index=Rd[i].dropna().index) for i in ids})
    Rd, Sd = Rd.loc[Sd.dropna().index], Sd.dropna()
    for h in (1, 5, 20):
        o = RK.position_risk(Rd, Sd, expo, 0.99, h)
        rows.append({"구분": "② VaR · ES (99%)", "항목": f"{h}영업일", "손실 ($)": o["var_fhs"], "손실 (% 자본)": o["var_fhs"] / cap * 100,
                     "비고": f"ES ${o['es_fhs']:,.0f} · 정규 VaR ${o['var_normal']:,.0f} · 분산 효과 전 합계 ${o['undiversified']:,.0f}"})
        if h == 20:
            top = o["table"].sort_values("VaR 기여 ($, 정규)", ascending=False).head(5)
            for i, r_ in top.iterrows():
                rows.append({"구분": "② 20일 VaR 기여 상위", "항목": name(i), "손실 ($)": r_["VaR 기여 ($, 정규)"], "손실 (% 자본)": r_["VaR 기여 ($, 정규)"] / cap * 100,
                             "비고": f"노출 ${r_['노출 ($)']:,.0f} · 기여 비중 {r_['기여 비중']*100:.0f}%"})
            C = o["corr"]; s_now = np.sqrt(Sd.iloc[-1].values) / 100; q = stats.norm.ppf(0.99)
            for rho in (0.6, 0.9):                                   # ⑤ 상관 스트레스: 위기 때처럼 모든 품목이 같은 방향으로 (품목 간 상관 = ρ)
                Cs = np.full_like(C.values, rho); np.fill_diagonal(Cs, 1)
                Hs = np.outer(s_now, s_now) * Cs * 20
                v = float(np.sqrt(expo.values @ Hs @ expo.values) * q)
                rows.append({"구분": "⑤ 상관 스트레스 (20일 99% 정규 VaR)", "항목": f"품목 간 상관 모두 {rho} (평소 약 0.1)", "손실 ($)": v, "손실 (% 자본)": v / cap * 100,
                             "비고": f"평소 정규 VaR ${o['var_normal']:,.0f}"})
    # ③ 과거 위기 재현
    Lr = A.load_levels(roll_adjusted=True)
    W = pd.read_parquet(A.DATA / "voltgt_weekly.parquet")
    mixr = W["혼합 · MinVar 50 + 모멘텀 25 + 가치 25 (사후 선택 주의)|10|4"]
    for src, cr in [("대시보드 위기 구간", A.CRISES), ("짧고 급한 구간", SHARP)]:
        T = RK.crisis_scenarios(Lr, expo, cr)
        for nm, r_ in T.iterrows():
            a, b = cr[nm]
            act = mixr.loc[a:b]
            real = float(((1 + act / 100).prod() - 1) * 100) if len(act) else np.nan
            rows.append({"구분": f"③ 위기 재현 ({src})", "항목": nm, "손실 ($)": -r_["손익 ($)"], "손실 (% 자본)": -r_["손익 ($)"] / cap * 100,
                         "비고": f"{a}~{b} · 그때 전략 실제 {real:+.1f}%" if np.isfinite(real) else f"{a}~{b} · 전략 자료 없음(2007 이전)"})
    # ④ 가상 충격
    for nm, sh in SHOCKS.items():
        mv = pd.Series(0.0, index=w.index)
        for g, x in sh.items():
            for i in (w.index if g == "*" else [i for i in GROUPS[g] if i in w.index]):
                mv[i] = x
        pnl = float((w * mv / 100).sum() * cap)
        rows.append({"구분": "④ 가상 충격", "항목": nm, "손실 ($)": -pnl, "손실 (% 자본)": -pnl / cap * 100, "비고": ""})
    # ⑥ 역스트레스
    net = float(w.sum())
    x_all = -10 / net if abs(net) > 1e-6 else np.inf
    rows.append({"구분": "⑥ 역스트레스", "항목": "자본 10% 손실 — 전 품목 같은 방향 이동", "손실 ($)": 0.1 * cap, "손실 (% 자본)": 10.0,
                 "비고": f"순노출 {net:+.2f}배 → 전 품목 {x_all:+.0f}% 동시 이동 필요"})
    for i, v in w.abs().sort_values(ascending=False).head(3).items():
        rows.append({"구분": "⑥ 역스트레스", "항목": f"자본 5% 손실 — {name(i)} 한 품목만", "손실 ($)": 0.05 * cap, "손실 (% 자본)": 5.0,
                     "비고": f"비중 {w[i]*100:+.1f}% → {name(i)} {(-5 / (w[i]*100)) * 100:+.0f}% 이동 필요"})
    T = pd.DataFrame(rows)
    T.to_csv(A.DATA / "stress_portfolio.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
    print(T.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
