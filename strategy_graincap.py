"""곡물 합계 비중 상한 백테스트 — 스트레스 테스트에서 곡물(대두 · 밀 · 대두박 등)이 VaR의 약 절반 → 곡물 몫을 제한하면?

방법(품목 단위): 3축(MinVar 상한 25% · 52주 모멘텀 · 가치)의 목표 10% 품목 비중을 50/25/25로 합친 '품목별 순비중'을 매주 만들고,
  곡물 5개(옥수수 · 밀 · 대두 · 대두박 · 대두유) 비중 절댓값 합이 전체 절댓값 합의 상한(25 · 35 · 50%)을 넘으면 곡물 비중만 비율대로 줄임
  (줄인 만큼은 다른 품목에 재배분하지 않음 → 섞은 뒤 목표 10%로 다시 맞추면서 자연히 다른 품목 비중이 커짐).
  손익 = ls_strategies._book(품목 비중, 품목 수익, 편도 5bp) → 직전 26주 실현 변동성으로 목표 10%(상한 4배).
  비교 기준 = 같은 방식으로 계산한 상한 없음(품목 단위 합산이라 축 단위 합산 결과와 약간 다름).
지표: 학습 2008–16 · 시험 2017– Sharpe, 최대 낙폭, 곡물 몫 평균, 2012 가뭄 · 2022 전쟁 구간, 지금 비중 기준 20일 99% VaR와 곡물 기여 비중.
실행: .venv\\Scripts\\python.exe strategy_graincap.py → data/strategy_graincap_compare.csv
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import analytics as A  # noqa: E402
import ls_extra as LX  # noqa: E402
import ls_strategies as LS  # noqa: E402
import risk as RK  # noqa: E402
import strategy_meta as SM  # noqa: E402

ANN, TGT, COST = 52, 10.0, 5.0
FROZEN_END = "2026-10-26"
GRAINS = ["yf:ZC=F", "yf:ZW=F", "yf:ZS=F", "yf:ZM=F", "yf:ZL=F"]


def item_mix(L: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    P = LS.weekly(L, LS.BASKET, "2001-01-01"); P = P[[c for c in LS.BASKET if c in P]]; R = LS._ret(P)
    lp = np.log(P)
    Ws = [(0.5, LS._vol_target(LS.alloc_weights_cap(P, 0.25), R, TGT)), (0.25, LS._vol_target(LS.tsmom_weights(P, 52, port_tgt=None), R, TGT)),
          (0.25, LS._vol_target(LX.xs_weights(-(lp - lp.rolling(260, min_periods=156).mean())), R, TGT))]
    return sum(w * W.fillna(0.0) for w, W in Ws), R


def cap_grains(W: pd.DataFrame, cap: float | None) -> pd.DataFrame:
    if cap is None:
        return W
    g = [c for c in GRAINS if c in W]
    ga, tot = W[g].abs().sum(1), W.abs().sum(1)
    other = tot - ga
    f = (cap * other / ((1 - cap) * ga)).where(ga > cap * tot, 1.0).clip(upper=1.0).fillna(1.0)
    W = W.copy(); W[g] = W[g].mul(f, axis=0)
    return W


def run(W: pd.DataFrame, R: pd.DataFrame) -> pd.Series:
    raw = LS._book(W, R, COST)
    pv = raw.rolling(26, min_periods=13).std().shift(1) * np.sqrt(ANN)
    return (raw * (TGT / pv).clip(upper=4.0).fillna(1.0)).loc["2005":]


def var20(W_last: pd.Series, L: pd.DataFrame, k: float, cap: float = 1e7) -> tuple[float, float]:
    w = W_last[W_last.abs() > 1e-4] * k; ids = list(w.index)
    Rd = pd.DataFrame({i: RK.returns(L[i]) for i in ids}).dropna(how="all")
    Sd = pd.DataFrame({i: pd.Series(RK.ewma_path(Rd[i].dropna().values, RK.RM_LAMBDA), index=Rd[i].dropna().index) for i in ids})
    Rd, Sd = Rd.loc[Sd.dropna().index], Sd.dropna()
    o = RK.position_risk(Rd, Sd, w * cap, 0.99, 20)
    t = o["table"]; gshare = float(t.loc[[i for i in ids if i in GRAINS], "VaR 기여 ($, 정규)"].sum() / t["VaR 기여 ($, 정규)"].sum())
    return o["var_fhs"] / cap * 100, gshare * 100


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    L = A.load_levels(roll_adjusted=True).loc[:FROZEN_END]
    W0, R = item_mix(L)
    rows, ref = [], None
    for cap in (None, 0.5, 0.35, 0.25):
        W = cap_grains(W0, cap); r = run(W, R)
        if ref is None:
            ref = r
        g = [c for c in GRAINS if c in W]
        share = (W[g].abs().sum(1) / W.abs().sum(1)).loc["2008":]
        raw = LS._book(W, R, COST); k = float((TGT / (raw.rolling(26, min_periods=13).std().shift(1) * np.sqrt(ANN))).clip(upper=4.0).fillna(1.0).iloc[-1])
        v, gs = var20(W.iloc[-1], L, k)
        o = SM.score(r, ref)
        rows.append({"곡물 상한 (총노출 대비)": "없음 (기준)" if cap is None else f"{int(cap*100)}%", **o,
                     "2012 가뭄 (6–8월) 수익 (%)": float(r.loc["2012-06":"2012-08"].sum()), "2022 수익 (%)": float(r.loc["2022"].sum()),
                     "곡물 몫 평균 (%)": float(share.mean() * 100), "상한 걸린 주 (%)": float((share >= (cap or 1) - 1e-6).mean() * 100) if cap else 0.0,
                     "지금 20일 99% VaR (%)": v, "지금 VaR 곡물 기여 (%)": gs})
    T = pd.DataFrame(rows)
    T.to_csv(A.DATA / "strategy_graincap_compare.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 260)
    print(T.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
