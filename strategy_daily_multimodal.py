"""멀티모달 갈래(텍스트 · 리포트 수치 · COT 포지션 · 기상 · 매크로 · 금융 · 월간 거시)를 일간 실행으로 — 주간 결과(multimodal_compare.csv)와 같은 신호, 일간 손익.

신호 자체는 주간 파이프라인(`multimodal.build`)이 만든 갈래별 순위 신호를 쓴다. 이유: 이 신호들의 정보는 주간 이하로만 갱신됨
  COT 주 1회(화요일 기준 · 금요일 발표, 기준일 +10일부터 사용), EIA 주 1회, NASS 주 1회, WASDE · STEO · 베이지북 월 1회, ENSO · SOI · FRED 월간, Open-Meteo 기상만 일간.
  → 일간 인덱스로 다시 만들어도 값이 새로 쓰이는 날짜는 같고(발표 지연 규칙이 인덱스와 무관), 달라지는 것은 일간 실행 · 일간 변동성 목표뿐.
방법: 금요일 신호를 다음 금요일까지 일간으로 이어 붙임 → 20일마다 상위 1/3 롱 · 하위 1/3 숏(역변동성, 130일) → 목표 변동성 10%(일간 130일) → 편도 5bp.
평가: 갈래 단독 + MinVar 50 · 모멘텀 25 · 갈래 25 혼합(일간 3축 기준 대비), 주간 결과와 나란히. 설정은 주간 때와 같고 이번에 새로 고른 것 없음(부호 롤링 · 고정 두 가지 모두 표시).
한계: 주중 발표(수요일 EIA 등)를 당일 반영하는 '더 빠른 정보'는 넣지 못함 — 주간 신호 이어 붙임이라 금요일 값. 기상만 일간 원천인데 신호는 주간으로 요약됨.
실행: .venv\\Scripts\\python.exe strategy_daily_multimodal.py (신호 만들기 포함 약 10~20분) → data/strategy_daily_multimodal_compare.csv
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import analytics as A  # noqa: E402
import ls_extra as LX  # noqa: E402
import ls_strategies as LS  # noqa: E402
import multimodal as MM  # noqa: E402
import strategy_daily_freq as DF  # noqa: E402
import strategy_daily_more as DM  # noqa: E402
import strategy_daily_signals as SDS  # noqa: E402
import strategy_switch as SS  # noqa: E402

ANN, VW = 252, 130


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t00 = time.time()
    L = A.load_levels(roll_adjusted=True).loc[:SDS.FROZEN_END]
    ids = [c for c in LS.BASKET if c in L]
    Pw = LS.weekly(L, LS.BASKET, "2001-01-01"); Pw = Pw[[c for c in LS.BASKET if c in Pw]]
    P = L[ids].loc["2001-01-01":]; P = P[P.index.dayofweek < 5].ffill(limit=5); R = LS._ret(P)
    S = DF.axes(P, R, ANN, 5, 5, 20, 20, 5)
    print(f"일간 3축 준비 {time.time() - t00:.0f}s · 주간 신호 만드는 중(multimodal.build)…", flush=True)
    B = MM.build(L, Pw)
    print(f"신호 준비 {time.time() - t00:.0f}s", flush=True)
    base = SDS.run_weights_d(S, SS.const(S, dict(zip(SDS.AX, SDS.BASE))))
    vol = R.rolling(VW, min_periods=VW // 2).std()
    sig = {}
    for mode, tag in (("rolling", "부호 롤링"), ("fixed", "부호 고정")):
        for fam, s in B[f"family_{mode}"].items():
            sig[f"갈래 {fam} ({tag})"] = s
        sig[f"멀티모달 전체 평균 ({tag})"] = B[f"multimodal_{mode}"]
    wk = pd.read_csv(A.DATA / "multimodal_compare.csv", encoding="utf-8-sig").set_index("전략")
    rows, store = [], {}
    for nm, Sg in sig.items():
        Sd = Sg.reindex(P.index.union(Sg.index)).ffill().reindex(P.index)
        W = LX.xs_weights(Sd, hold=20, inv_vol=1 / vol)
        x = DM.book(DF.vt(W, R, VW, ANN, 5), R).reindex(S.index).fillna(0.0)
        X = S.assign(신호=x)
        mix = SDS.run_weights_d(X, pd.DataFrame([[0.5, 0.25, 0.25, 0.25 * 0] if False else [0.5, 0.25, 0.0, 0.25]] * len(X), index=X.index, columns=["MinVar", "모멘텀", "가치", "신호"]))
        sx, sm = SDS.score(x, base), SDS.score(mix, base)
        key = nm.replace("갈래 ", "갈래 단독 · ").replace(" (부호 롤링)", "").replace(" (부호 고정)", "")
        key = (f"갈래 단독 (고정 부호) · {nm.split(' ', 1)[1].rsplit(' (', 1)[0]}" if "고정" in nm and nm.startswith("갈래") else key)
        wkx = wk["Sharpe"].get(key, np.nan) if nm.startswith("갈래") else np.nan
        rows.append({"신호": nm, "단독 학습 2008–16": sx["Sharpe 학습 2008–16"], "단독 시험 2017–": sx["Sharpe 시험 2017–"], "단독 2008–": sx["Sharpe 2008–"],
                     "단독 주간 합산 2008–": sx["주간 합산 2008–"], "주간 실행 단독 Sharpe (참고, 2007–)": wkx,
                     "혼합 학습": sm["Sharpe 학습 2008–16"], "혼합 시험": sm["Sharpe 시험 2017–"], "혼합 주간 합산 시험": sm["주간 합산 시험"], "혼합 시험 기준 대비 (%p)": sm["시험 기준 대비 (%p)"],
                     "혼합 시험 차이 t": sm["시험 차이 t"]})
        store[nm] = x
        print(f"  {nm}  단독 {sx['Sharpe 학습 2008–16']:+.2f}/{sx['Sharpe 시험 2017–']:+.2f} · 혼합 시험 {sm['Sharpe 시험 2017–']:.2f} (t {sm['시험 차이 t']:+.2f})", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(A.DATA / "strategy_daily_multimodal_compare.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({k: v.resample("W-FRI").sum() for k, v in store.items()}).to_parquet(A.DATA / "strategy_daily_multimodal_weekly.parquet")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 40)
    print(T.round(2).to_string(index=False))
    print(f"기준(일간 3축 50/25/25): 학습 {SDS.sh(base.loc['2008':'2016']):.2f} 시험 {SDS.sh(base.loc['2017':]):.2f} · 총 {time.time() - t00:.0f}s")


if __name__ == "__main__":
    main()
