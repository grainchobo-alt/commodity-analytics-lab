"""분수 차분(López de Prado) 진단 (Z): 우리 19개 품목 주간 로그 가격에서 d(0~1)별 ADF 정상성 · 원 로그 가격과의 상관(기억 보존) · 품목별 최소 d.
고정폭 창 분수 차분(가중치 |w| < 1e-4에서 절단), ADF = 상수항 · AIC. 예측 성능 시험은 아님(진단).
실행: .venv/Scripts/python.exe strategy_ffd_check.py → data/strategy_ffd_check.csv (d별 요약) · strategy_ffd_check_items.csv (품목별)
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import alt_ts as AT  # noqa: E402
import analytics as A  # noqa: E402
import predict_lab as PL  # noqa: E402
from statsmodels.tsa.stattools import adfuller  # noqa: E402

DS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]


def ffd_weights(d: float, thr: float = 1e-4, maxk: int = 600) -> np.ndarray:
    w = [1.0]
    for k in range(1, maxk):
        w_ = -w[-1] * (d - k + 1) / k
        if abs(w_) < thr:
            break
        w.append(w_)
    return np.array(w)


def ffd(x: pd.Series, d: float) -> pd.Series:
    w = ffd_weights(d); n = len(w); v = x.values; out = np.full(len(v), np.nan)
    for t in range(n - 1, len(v)):
        out[t] = w @ v[t - n + 1:t + 1][::-1]
    return pd.Series(out, index=x.index)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    P = AT.weekly_px(PL.frozen_levels(True), "2001-01-01"); lp = np.log(P); rows = []
    for c in lp.columns:
        x = lp[c].dropna()
        if len(x) < 800:
            continue
        for d in DS:
            z = ffd(x, d).dropna()
            rows.append({"품목": c, "d": d, "ADF p": adfuller(z.values, regression="c", autolag="AIC")[1], "원 로그가격과 상관": float(np.corrcoef(z.values, x.reindex(z.index).values)[0, 1]), "가중치 길이(주)": len(ffd_weights(d)), "관측 주": len(z)})
    I = pd.DataFrame(rows); I.to_csv(A.DATA / "strategy_ffd_check_items.csv", index=False, encoding="utf-8-sig")
    S = I.groupby("d").agg(**{"ADF p 평균": ("ADF p", "mean"), "정상 품목 수 (p<0.05)": ("ADF p", lambda s: int((s < 0.05).sum())), "원 로그가격과 상관 평균": ("원 로그가격과 상관", "mean"), "품목 수": ("품목", "nunique")}).reset_index()
    S.to_csv(A.DATA / "strategy_ffd_check.csv", index=False, encoding="utf-8-sig")
    mins = I[I["ADF p"] < 0.05].groupby("품목")["d"].min()
    print(S.round(3).to_string(index=False)); print("품목별 최소 d(격자 0.1 단위): 평균", round(float(mins.mean()), 2), "중앙값", float(mins.median()), "범위", float(mins.min()), "~", float(mins.max()), "· 분포", mins.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
