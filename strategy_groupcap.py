"""품목군별 위험 한도 백테스트 — 곡물 · 에너지 · 축산 합계 비중 상한을 따로 · 함께 걸면? (strategy_graincap.py 확장)

방법은 strategy_graincap과 같음: 3축 품목 비중을 50/25/25로 합친 품목별 순비중에서, 품목군 절댓값 합이 전체 절댓값 합의 상한을 넘으면
그 품목군 비중만 비율대로 줄이고 → 품목 손익(편도 5bp) → 목표 10%로 다시 맞춤. 여러 품목군을 함께 걸 때는 곡물 → 에너지 → 축산 순서로 적용.
품목군: 곡물(옥수수 · 밀 · 대두 · 대두박 · 대두유) · 에너지(WTI · 난방유 · 휘발유 · 천연가스) · 축산(생우 · 돈육) · 금속(구리 · 금 · 은 · 백금, 참고).
지표: 학습 2008–16 · 시험 2017– Sharpe, 최대 낙폭, 품목군 몫 평균 · 상한 걸린 주, 지금 20일 99% VaR와 품목군별 VaR 기여.
실행: .venv\\Scripts\\python.exe strategy_groupcap.py → data/strategy_groupcap_compare.csv
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import analytics as A  # noqa: E402
import ls_strategies as LS  # noqa: E402
import risk as RK  # noqa: E402
import strategy_graincap as GC  # noqa: E402
import strategy_meta as SM  # noqa: E402

FROZEN_END = "2026-10-26"
GROUPS = {"곡물": GC.GRAINS, "에너지": ["yf:CL=F", "yf:HO=F", "yf:RB=F", "yf:NG=F"], "축산": ["yf:LE=F", "yf:HE=F"],
          "금속": ["yf:HG=F", "yf:GC=F", "yf:SI=F", "yf:PL=F"]}


def cap_group(W: pd.DataFrame, cols: list[str], cap: float) -> pd.DataFrame:
    g = [c for c in cols if c in W]
    ga, tot = W[g].abs().sum(1), W.abs().sum(1)
    f = (cap * (tot - ga) / ((1 - cap) * ga)).where(ga > cap * tot, 1.0).clip(upper=1.0).fillna(1.0)
    W = W.copy(); W[g] = W[g].mul(f, axis=0)
    return W


def var_by_group(W_last: pd.Series, L: pd.DataFrame, k: float, cap: float = 1e7) -> dict:
    w = W_last[W_last.abs() > 1e-4] * k; ids = list(w.index)
    Rd = pd.DataFrame({i: RK.returns(L[i]) for i in ids}).dropna(how="all")
    Sd = pd.DataFrame({i: pd.Series(RK.ewma_path(Rd[i].dropna().values, RK.RM_LAMBDA), index=Rd[i].dropna().index) for i in ids})
    Rd, Sd = Rd.loc[Sd.dropna().index], Sd.dropna()
    o = RK.position_risk(Rd, Sd, w * cap, 0.99, 20); t = o["table"]; tot = t["VaR 기여 ($, 정규)"].sum()
    out = {"지금 20일 99% VaR (%)": o["var_fhs"] / cap * 100}
    for g, cols in GROUPS.items():
        out[f"VaR 기여 {g} (%)"] = float(t.loc[[i for i in ids if i in cols], "VaR 기여 ($, 정규)"].sum() / tot * 100)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    L = A.load_levels(roll_adjusted=True).loc[:FROZEN_END]
    W0, R = GC.item_mix(L)
    rules = {"없음 (기준)": {}, "곡물 25%": {"곡물": 0.25}, "에너지 25%": {"에너지": 0.25}, "에너지 15%": {"에너지": 0.15},
             "축산 25%": {"축산": 0.25}, "축산 15%": {"축산": 0.15}, "곡물 25% + 에너지 25% + 축산 25%": {"곡물": 0.25, "에너지": 0.25, "축산": 0.25},
             "곡물 25% + 에너지 15% + 축산 15%": {"곡물": 0.25, "에너지": 0.15, "축산": 0.15}, "모든 품목군 25% (금속 포함)": {g: 0.25 for g in GROUPS}}
    rows, ref, weekly = [], None, {}
    for nm, caps in rules.items():
        W = W0
        for g, c in caps.items():
            W = cap_group(W, GROUPS[g], c)
        r = GC.run(W, R)
        ref = r if ref is None else ref
        weekly[nm] = r
        raw = LS._book(W, R, GC.COST)
        k = float((GC.TGT / (raw.rolling(26, min_periods=13).std().shift(1) * np.sqrt(GC.ANN))).clip(upper=4.0).fillna(1.0).iloc[-1])
        sh = {f"{g} 몫 평균 (%)": float((W[[c for c in cols if c in W]].abs().sum(1) / W.abs().sum(1)).loc["2008":].mean() * 100) for g, cols in GROUPS.items()}
        rows.append({"상한": nm, **SM.score(r, ref), **sh, **var_by_group(W.iloc[-1], L, k)})
        print(nm, "완료", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(A.DATA / "strategy_groupcap_compare.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(weekly).to_parquet(A.DATA / "strategy_groupcap_weekly.parquet")   # 대시보드 ⑤ Portfolio Customization '품목군 위험 한도'
    pd.set_option("display.width", 280)
    cols = ["상한", "Sharpe 학습 2008–16", "Sharpe 시험 2017–", "Sharpe 2008–", "최대 낙폭 (%)", "시험 차이 t", "곡물 몫 평균 (%)", "에너지 몫 평균 (%)", "축산 몫 평균 (%)",
            "지금 20일 99% VaR (%)", "VaR 기여 곡물 (%)", "VaR 기여 에너지 (%)", "VaR 기여 축산 (%)", "VaR 기여 금속 (%)"]
    print(T[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
