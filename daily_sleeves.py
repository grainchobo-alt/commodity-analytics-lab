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


# ------------------------------------------------------------------ 일간 신호 오버레이 (대시보드 '일간 신호로 노출 조절')
# 신호 정의는 strategy_daily_signals.py와 같음 — 값이 문턱보다 크면 '위험'으로 보고 노출을 low(0 또는 0.5)로 줄이고 다음 날부터 적용.
SIGNALS = {  # 표시 이름: (신호 열, 기본 문턱, 최소, 최대, 간격, 설명)
    "DCC 동조성 (원자재 8개 평균 상관의 분위)": ("동조성", 0.90, 0.60, 0.99, 0.01, "일간 DCC를 20일마다 최근 3,900일로 재추정한 평균 상관의 그때까지 분위"),
    "VIX (분위)": ("VIX", 0.90, 0.60, 0.99, 0.01, "FRED 일간 VIX 수준의 그때까지 분위"),
    "HMM 위기 확률 (주간 확률을 이어 붙임)": ("HMM", 0.90, 0.30, 0.99, 0.01, "원자재 6 + 거시 8 다변량 HMM의 위기 국면 확률 — 주간 값을 금요일부터 다음 금요일까지 유지(일간 HMM은 없음)"),
    "원자재 변동성 국면 (65일 변동성 ÷ 중앙값)": ("변동성", 1.50, 1.00, 3.00, 0.1, "원자재 19개 평균 변동성의 그때까지 중앙값 대비 배수"),
    "흡수비율 z (20일 변화)": ("흡수비율", 1.50, 0.50, 3.00, 0.1, "19개 252일 1주성분 몫의 20일 변화 z"),
    "뉴스 감성 악화 (−감성 z)": ("뉴스", 1.00, 0.00, 2.50, 0.1, "베이지북 · WASDE · STEO 문서 감성 — 문서 공표일에만 갱신"),
    "칼만 평균 < 0 (내 포트폴리오 수익)": ("칼만", 0.0, 0.0, 0.0, 0.0, "내 포트폴리오 일간 수익의 지역 수준 칼만 평균이 0보다 작으면"),
    "낙폭 손절 (내 포트폴리오 누적 낙폭 −x%)": ("낙폭", 5.0, 3.0, 20.0, 1.0, "누적 수익이 고점 대비 −x% 아래로 내려가면 현금, −x/2% 위로 회복하면 복귀"),
}


def load_signals() -> pd.DataFrame:
    from pathlib import Path
    p = Path(__file__).parent / "data" / "frozen" / f"signals_daily_{FROZEN_END}.pkl"
    return pd.read_pickle(p) if p.exists() else pd.DataFrame()


def kalman_mean(r: pd.Series, q: float = 2e-4) -> pd.Series:
    """지역 수준 칼만 필터 (strategy_daily_signals.kalman_mean_d와 같음): 처음 780일 분산을 관측 분산으로, 그 시점까지 자료만 사용."""
    x = r.fillna(0.0).values; v = float(np.var(x[:780])) or 1e-6; Q = q * v; m, Pv = 0.0, v; out = np.empty(len(x))
    for t, y in enumerate(x):
        Pv += Q; K = Pv / (Pv + v); m += K * (y - m); Pv *= (1 - K); out[t] = m * ANN
    return pd.Series(out, r.index)


def dd_exposure(r: pd.Series, x: float, low: float) -> pd.Series:
    cum, peak, on, e = 0.0, 0.0, False, []
    for v in r.fillna(0.0).values:
        cum += v; peak = max(peak, cum); dd = cum - peak
        if not on and dd < -x:
            on = True
        elif on and dd > -x / 2:
            on = False
        e.append(low if on else 1.0)
    return pd.Series(e, r.index)


def exposure(signal: str, thr: float, low: float, r: pd.Series, sigs: pd.DataFrame) -> pd.Series:
    col = SIGNALS[signal][0]
    if col == "칼만":
        return pd.Series(np.where(kalman_mean(r) < 0, low, 1.0), r.index)
    if col == "낙폭":
        return dd_exposure(r, thr, low)
    s = sigs[col].reindex(r.index).ffill()
    return pd.Series(np.where(s > thr, low, 1.0), r.index)


def overlay(r: pd.Series, e: pd.Series, gross: float = 1.15, cost_bp: float = 5.0) -> pd.Series:
    """노출 e는 오늘 신호로 정하고 다음 날부터 적용, 노출을 바꾸면 |Δe| × 1.15 × 5bp 비용 (현금 몫 수익 0)."""
    held = e.reindex(r.index).ffill().fillna(1.0).shift(1).fillna(1.0)
    return held * r - held.diff().abs().fillna(0) * gross * cost_bp / 100


if __name__ == "__main__":
    import pickle
    from pathlib import Path
    X, G = build()
    p = Path(__file__).parent / "data" / "frozen" / f"sleeves_daily_{FROZEN_END}.pkl"
    p.parent.mkdir(exist_ok=True); pickle.dump((X, G), open(p, "wb"))
    print(X.shape, X.index[0], X.index[-1]); print((X.mean() / X.std() * np.sqrt(ANN)).round(2).to_string())
