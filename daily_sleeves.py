"""Portfolio Customization '일간' 모드 재료 — 일간 자료로 새로 계산한 전략별 일간 수익률(%, 각 목표 변동성 10%, 편도 5bp).

3축(MinVar · 52주 모멘텀 · 가치)은 strategy_daily_freq(D1: 결정 주기는 주간과 같고 손익 · 변동성은 일간), 나머지는 일간 실험에서 쓴 같은 함수.
창은 달력 기간이 같게 ×5(52주 = 260일, 26주 = 130일, 104주 = 520일, 5년 = 1,300일). 주간 모드의 롱숏 · 대안 시계열 · 멀티모달 재료는 주간 모델이라 일간에 없음.
저장: data/frozen/sleeves_daily_{FROZEN_END}.pkl (app.py `_frozen`이 없으면 build()로 만듦, 약 1–2분)
실행: .venv\\Scripts\\python.exe daily_sleeves.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

FROZEN_END = "2026-10-26"
ANN, VW = 252, 130
NAMES = {"MinVar": "기준 · MinVar 롱 (⑥)", "모멘텀": "기준 · 52주 모멘텀 (④)", "가치": "V1 가치 (5년 평균 대비 싼 품목 롱)"}


def build() -> tuple[pd.DataFrame, dict]:
    import analytics as A
    import ls_strategies as LS
    import strategy_daily_freq as DF
    import strategy_daily_more as DM
    L = A.load_levels(roll_adjusted=True).loc[:FROZEN_END]
    ids = [c for c in LS.BASKET if c in L]
    P = L[ids].loc["2001-01-01":]; P = P[P.index.dayofweek < 5].ffill(limit=5); R = LS._ret(P)
    S = DF.axes(P, R, ANN, 5, 5, 20, 20, 5)
    X, G = {}, {}

    def add(nm, r, g="수치 전략"):
        X[nm] = r.reindex(S.index).fillna(0.0); G[nm] = g
    for k, nm in NAMES.items():
        add(nm, S[k])
    add("혼합 · MinVar 70 + 모멘텀 30 (기존)", DF.mix(S, [0.7, 0.3, 0.0], VW, ANN))
    add("일간 · 모멘텀 130일 (26주)", DM.mom_axis(P, R, [130]))
    add("일간 · 모멘텀 20일 (4주)", DM.mom_axis(P, R, [20]))
    add("일간 · 모멘텀 여러 기간 평균 (20·65·260일)", DM.mom_axis(P, R, [20, 65, 260]))
    add("일간 · 상승 추세 집중 3개 (130·260일)", DM.up_trend(P, R, 130, 260, 3, 20))
    add("일간 · 상승 추세 집중 2개 (130·260일)", DM.up_trend(P, R, 130, 260, 2, 20))
    add("일간 · 짧은 추세 65일 · 상위 3개", DM.up_trend(P, R, 65, None, 3, 5))
    add("일간 · MinVar 품목 상한 없음", DM.book(DF.vt(DF.minvar_w(P, R, 520, 20, cap=1.0), R, VW, ANN, 5), R))
    return pd.DataFrame(X).sort_index(), G


if __name__ == "__main__":
    import pickle
    from pathlib import Path
    X, G = build()
    p = Path(__file__).parent / "data" / "frozen" / f"sleeves_daily_{FROZEN_END}.pkl"
    p.parent.mkdir(exist_ok=True); pickle.dump((X, G), open(p, "wb"))
    print(X.shape, X.index[0], X.index[-1]); print((X.mean() / X.std() * np.sqrt(ANN)).round(2).to_string())
