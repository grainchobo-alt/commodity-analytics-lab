"""짧은 추세(4 · 8 · 13주) 상승 집중 백테스트 — 2022년 같은 급등은 1분기 안에 끝나 26 · 52주 추세는 늦게 타고 늦게 내림. 짧게 보면 나아지나?

재료: strategy_crisis_mom.up_conc와 같은 '상승 추세 집중'(롱만, 같은 위험)이되 판단 기간을 짧게 · 점검을 매주로.
  상승 추세 집중 (n주, k개): 직전 n주 수익 > 0 인 품목 중 n주 수익 ÷ 26주 변동성 상위 k개 롱, 매주 다시 고름(편도 5bp, 목표 10% · 상한 4배)
  n = 4 · 8 · 13 · 26(비교), k = 2 · 3 · 5
평가: 단독 Sharpe(학습 2008–16 · 시험 2017–), 2022 1분기 · 2분기, 2025–26 수익, 연 회전율(거래 비용 부담),
      그리고 혼합 25/50/25(모멘텀 자리에 이 판) — 고정 50/25/25 대비. 무엇을 들었나(2022-03, 2025).
실행: .venv\\Scripts\\python.exe strategy_short_trend.py → data/strategy_short_trend_compare.csv
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


def up_short(P: pd.DataFrame, R: pd.DataFrame, n: int, k: int) -> tuple[pd.DataFrame, dict]:
    lp = np.log(P); s = lp.diff(n); v = R.rolling(26, min_periods=13).std()
    W = pd.DataFrame(0.0, index=P.index, columns=P.columns); held = {}
    for t in range(len(P)):
        sc = (s.iloc[t] / v.iloc[t])[s.iloc[t] > 0].dropna()
        if len(sc):
            top = sc.nlargest(k).index; w = 1 / v.iloc[t][top]
            W.loc[P.index[t], top] = (w / w.sum()).values; held[P.index[t]] = list(top)
    return W, held


def sh(x: pd.Series) -> float:
    return float(x.mean() / x.std() * np.sqrt(ANN)) if len(x) > 10 and x.std() > 0 else np.nan


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from llm import tools as T_
    L = A.load_levels(roll_adjusted=True).loc[:FROZEN_END]
    S, R, P = SM.sleeves(L)
    base = SS.run_weights(S, SS.const(S, dict(zip(SM.AXES, SM.BASE["고정 50/25/25"]))))
    rows = [{"판": "고정 50/25/25 (기준 혼합)", **SM.score(base, base), "2022 1분기 (%)": float(base.loc["2022-01":"2022-03"].sum()),
             "2022 2분기 (%)": float(base.loc["2022-04":"2022-06"].sum()), "2025–26 (%)": float(base.loc["2025":].sum())}]
    rows.append({"판": "52주 모멘텀 단독 (기존)", "Sharpe 학습 2008–16": sh(S["모멘텀"].loc["2008":"2016"]), "Sharpe 시험 2017–": sh(S["모멘텀"].loc["2017":]),
                 "Sharpe 2008–": sh(S["모멘텀"].loc["2008":]), "2022 1분기 (%)": float(S["모멘텀"].loc["2022-01":"2022-03"].sum()),
                 "2022 2분기 (%)": float(S["모멘텀"].loc["2022-04":"2022-06"].sum()), "2025–26 (%)": float(S["모멘텀"].loc["2025":].sum())})
    for n in (4, 8, 13, 26):
        for k in (2, 3, 5):
            W, held = up_short(P, R, n, k)
            Wv = LS._vol_target(W, R, TGT); x = LS._book(Wv, R, COST).reindex(S.index).fillna(0.0)
            turn = float(Wv.diff().abs().sum(1).loc["2008":].sum() / (len(Wv.loc["2008":]) / ANN))
            X = S.assign(단기=x)
            mix = SS.run_weights(X, pd.DataFrame([[0.25, 0.0, 0.25, 0.5]] * len(X), index=X.index, columns=["MinVar", "모멘텀", "가치", "단기"]))
            ms = SM.score(mix, base)
            h22 = pd.Series([i for d, h in held.items() if "2022-02-15" <= f"{d:%Y-%m-%d}" <= "2022-04-15" for i in h]).value_counts().head(4)
            h25 = pd.Series([i for d, h in held.items() if f"{d:%Y}" == "2025" for i in h]).value_counts().head(4)
            rows.append({"판": f"상승 추세 집중 {n}주 · 상위 {k}개 (단독)", "Sharpe 학습 2008–16": sh(x.loc["2008":"2016"]), "Sharpe 시험 2017–": sh(x.loc["2017":]),
                         "Sharpe 2008–": sh(x.loc["2008":]), "2022 1분기 (%)": float(x.loc["2022-01":"2022-03"].sum()),
                         "2022 2분기 (%)": float(x.loc["2022-04":"2022-06"].sum()), "2025–26 (%)": float(x.loc["2025":].sum()), "연 회전율 (배)": turn,
                         "혼합 25/50/25 학습": ms["Sharpe 학습 2008–16"], "혼합 25/50/25 시험": ms["Sharpe 시험 2017–"], "혼합 시험 차이 t": ms["시험 차이 t"],
                         "2022-03 전후 보유": ", ".join(f"{T_.name_of(i)} {c}" for i, c in h22.items()),
                         "2025 보유": ", ".join(f"{T_.name_of(i)} {c}" for i, c in h25.items())})
            print(f"{n}주 · {k}개 완료", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(A.DATA / "strategy_short_trend_compare.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 60)
    print(T.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
