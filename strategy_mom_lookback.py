"""3축 혼합 안의 모멘텀 기간 백테스트 — 52주 대신 4 · 8 · 13 · 26 · 39주(짧게)로 보면? 여러 기간 평균은?

재료: strategy_meta.sleeves의 MinVar · 가치 그대로 + 모멘텀 축만 기간을 바꿔 다시 만듦(ls_strategies.tsmom_weights(look), 품목 역변동성 · 목표 10% · 상한 4배 · 편도 5bp).
  기간: 4 · 8 · 13 · 26 · 39 · 52주(기준), 여러 기간 평균(4 · 13 · 52주 신호 비중을 평균 = 빠른 반응 + 긴 추세)
평가: 모멘텀 단독 · 혼합 50/25/25(모멘텀 자리만 교체) — 학습 2008–16 · 시험 2017–, 최대 낙폭, 2022 1 · 2분기, 2025–26, 연 회전율, 기준 대비 시험 차이 t.
이전 실험(experiments.momentum_lookback, ETF 바스켓 · 2010~): 3개월 −0.1 안팎 · 12-1개월이 최선 — 이번은 3축 혼합 안에서 학습/시험으로 재확인.
실행: .venv\\Scripts\\python.exe strategy_mom_lookback.py → data/strategy_mom_lookback_compare.csv
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import analytics as A  # noqa: E402
import ls_strategies as LS  # noqa: E402
import strategy_meta as SM  # noqa: E402
import strategy_switch as SS  # noqa: E402

ANN, TGT, COST = 52, 10.0, 5.0
FROZEN_END = "2026-10-26"


def sh(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(ANN)) if len(x) > 10 and x.std() > 0 else np.nan


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    L = A.load_levels(roll_adjusted=True).loc[:FROZEN_END]
    S, R, P = SM.sleeves(L)
    P = P[[c for c in LS.BASKET if c in P]]
    base = SS.run_weights(S, SS.const(S, dict(zip(SM.AXES, SM.BASE["고정 50/25/25"]))))
    Wl = {n: LS.tsmom_weights(P, n, port_tgt=None) for n in (4, 8, 13, 26, 39, 52)}
    Wl["여러 기간 평균 (4 · 13 · 52주)"] = (Wl[4].fillna(0) + Wl[13].fillna(0) + Wl[52].fillna(0)) / 3
    rows = []
    for n, W in Wl.items():
        Wv = LS._vol_target(W, R, TGT); m = LS._book(Wv, R, COST).reindex(S.index).fillna(0.0)
        X = S.assign(모멘텀=m)
        mix = SS.run_weights(X, SS.const(X, dict(zip(SM.AXES, SM.BASE["고정 50/25/25"]))))
        sc = SM.score(mix, base)
        lab = f"{n}주" if isinstance(n, int) else n
        rows.append({"모멘텀 기간": lab + (" (기준)" if n == 52 else ""), "단독 학습 2008–16": sh(m.loc["2008":"2016"]), "단독 시험 2017–": sh(m.loc["2017":]),
                     "단독 2008–": sh(m.loc["2008":]), "연 회전율 (배)": float(Wv.diff().abs().sum(1).loc["2008":].sum() / (len(Wv.loc["2008":]) / ANN)),
                     "혼합 학습 2008–16": sc["Sharpe 학습 2008–16"], "혼합 시험 2017–": sc["Sharpe 시험 2017–"], "혼합 2008–": sc["Sharpe 2008–"],
                     "혼합 최대 낙폭 (%)": sc["최대 낙폭 (%)"], "혼합 시험 차이 t": sc["시험 차이 t"],
                     "모멘텀 2022 1분기 (%)": float(m.loc["2022-01":"2022-03"].sum()), "모멘텀 2022 2분기 (%)": float(m.loc["2022-04":"2022-06"].sum()),
                     "모멘텀 2025–26 (%)": float(m.loc["2025":].sum())})
        print(lab, "완료", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(A.DATA / "strategy_mom_lookback_compare.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 260)
    print(T.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
