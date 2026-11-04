"""위험 신호로 목표 변동성을 동적으로 조절 + 단기 금리 이자 반영 백테스트.

기본: 고정 50/25/25 혼합(목표 변동성 10%로 맞춘 주간 순수익, 비용 포함). 노출 e_t = 그 주 목표 변동성 ÷ 10%.
  e < 1 = 일부 현금(목표 변동성을 낮춤), e > 1 = 차입·추가 레버리지(잔잔할 때 목표를 높임). t주 말에 정해 t+1주 적용, |Δe| × 1.15 × 5bp 비용.
위험 점수: strategy_cash.risk_signals 6종(HMM · 변동성 · 동조성 · 흡수비율 · VIX · 뉴스)
  계단형  : 켜진 신호 개수 n(각 신호의 가운데 문턱) → n=0 이면 hi%, 1 → 10%, 2 → 7%, 3개 이상 → 5%
  연속형  : 6개 신호를 그때까지 자료로 표준화한 평균 z → e = clip(1 − β·z, 0.5, 1.3)
단기 금리(미 3개월 국채, fred:DTB3, 주간 평균 → 그 주 이자 = 연율 ÷ 52) 두 가지 관점
  A 현금 몫만 이자: 수익 = e·전략 + (1 − e)·금리   (e > 1이면 차입 이자를 냄 — '현금을 따로 들고 있는 계좌' 가정)
  B 선물 구조    : 수익 = e·전략 + 금리             (선물은 증거금만 쓰고 자본 전체가 국채 이자를 받음 — 현금 피신해도 추가 이자 없음)
지표: 초과수익 Sharpe(수익 − 금리) — A · B에서 같은 값이 나오는 '위험 대비 성과', 총수익 연수익 · 최대 낙폭 · 최악 1년.
대조군: 같은 평균 노출의 고정 노출(늘 e̅). 설정은 학습 2008–2016 초과 Sharpe로만 고르고 시험 2017–에서 확인. 위약: 노출 경로를 13주 블록으로 섞기 200회.
실행: .venv\\Scripts\\python.exe strategy_voltgt_dyn.py → data/strategy_voltgt_dyn_compare.csv
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import analytics as A  # noqa: E402
import strategy_cash as SC  # noqa: E402
import strategy_meta as SM  # noqa: E402
import strategy_switch as SS  # noqa: E402

ANN, GROSS, COST = 52, 1.15, 0.05
FROZEN_END, SPLIT = "2026-10-26", "2017-01-01"
MID = {"HMM": 0.7, "변동성": 1.5, "동조성": 0.9, "흡수비율": 1.5, "VIX": 0.9, "뉴스": 1.0}


def run(r: pd.Series, rf: pd.Series, e: pd.Series, view: str) -> pd.Series:
    e = e.reindex(r.index).ffill().fillna(1.0)
    held = e.shift(1).fillna(1.0)
    cost = held.diff().abs().fillna(0) * GROSS * COST
    return held * r + ((1 - held) if view == "A" else 1.0) * rf - cost


def stats(tot: pd.Series, rf: pd.Series, e: pd.Series | None = None) -> dict:
    tot = tot.loc["2008":]; ex = (tot - rf.reindex(tot.index)).dropna()
    sh = lambda x: float(x.mean() / x.std() * np.sqrt(ANN))       # noqa: E731
    cum = (1 + tot / 100).cumprod()
    o = {"초과 Sharpe 2008–": sh(ex), "초과 Sharpe 학습 2008–16": sh(ex.loc[:"2016"]), "초과 Sharpe 시험 2017–": sh(ex.loc[SPLIT:]),
         "총 연수익 (%)": float(tot.mean() * ANN), "최대 낙폭 (%)": float((cum / cum.cummax() - 1).min() * 100),
         "최악 1년 (%)": float(tot.groupby(tot.index.year).sum().min())}
    if e is not None:
        ee = e.reindex(tot.index).ffill().fillna(1.0); o["평균 노출"] = float(ee.mean()); o["노출 < 1 비율 (%)"] = float((ee < 1).mean() * 100)
    return o


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    L = A.load_levels(roll_adjusted=True).loc[:FROZEN_END]
    S, R, P = SM.sleeves(L); idx = S.index
    base = SS.run_weights(S, SS.const(S, dict(zip(SM.AXES, SM.BASE["고정 50/25/25"]))))
    rf = A.load_levels(roll_adjusted=False)["fred:DTB3"].resample("W-FRI").mean().reindex(idx).ffill() / ANN
    print(f"단기 금리 평균 (연) 2008–16 {rf.loc['2008':'2016'].mean()*52:.2f}% · 2017– {rf.loc['2017':].mean()*52:.2f}%", flush=True)
    sig = SC.risk_signals(L, R, idx)
    n_on = sum((sig[k] > MID[k]).astype(int) for k in MID)
    Z = pd.DataFrame({k: (v - v.expanding(104).mean()) / v.expanding(104).std() for k, v in sig.items()}).clip(-3, 3)
    zbar = Z.mean(1)
    cands = {}
    for hi in (10, 12):
        tg = np.select([n_on == 0, n_on == 1, n_on == 2], [hi, 10, 7], 5)
        cands[("계단형 (켜진 신호 수 → 목표)", f"0개 {hi}% · 1개 10% · 2개 7% · 3개+ 5%")] = pd.Series(tg / 10, idx)
    for b in (0.1, 0.2, 0.3):
        cands[("연속형 (위험 z 평균)", f"β={b}")] = (1 - b * zbar).clip(0.5, 1.3).fillna(1.0)
    rows = []
    for view in ("A", "B"):
        rows.append({"이자 관점": view, "규칙": "고정 목표 10% (기준)", "설정": "", **stats(run(base, rf, pd.Series(1.0, idx), view), rf, pd.Series(1.0, idx))})
        for c in (0.7, 0.5):
            e = pd.Series(c, idx)
            rows.append({"이자 관점": view, "규칙": f"고정 목표 {int(c*10)}% (대조군)", "설정": "", **stats(run(base, rf, e, view), rf, e)})
        sc = {k: stats(run(base, rf, e, view), rf, e) for k, e in cands.items()}
        best = {}
        for k, s in sc.items():
            if k[0] not in best or s["초과 Sharpe 학습 2008–16"] > sc[best[k[0]]]["초과 Sharpe 학습 2008–16"]:
                best[k[0]] = k
        for k, s in sc.items():
            e = cands[k]
            ctl = stats(run(base, rf, pd.Series(float(e.loc["2008":].mean()), idx), view), rf)
            row = {"이자 관점": view, "규칙": k[0], "설정": k[1], **s, "학습 최고": "←" if best[k[0]] == k else "",
                   "같은 평균 노출 고정 시 최대 낙폭": ctl["최대 낙폭 (%)"], "같은 평균 노출 고정 시 최악 1년": ctl["최악 1년 (%)"]}
            if best[k[0]] == k:
                rng = np.random.default_rng(0); blocks = [idx[i:i + 13] for i in range(0, len(idx), 13)]; pl = []
                for _ in range(200):
                    o = rng.permutation(len(blocks)); ep = pd.concat([e.reindex(blocks[j]) for j in o]).iloc[:len(idx)]; ep.index = idx
                    pl.append(stats(run(base, rf, ep, view), rf)["초과 Sharpe 시험 2017–"])
                row["위약 p (시험)"] = float((np.array(pl) >= s["초과 Sharpe 시험 2017–"]).mean())
            rows.append(row)
        # 현금 피신(strategy_cash 학습 최고 규칙) + 이자: 낙폭 5% 손절 · 뉴스 · 칼만
        for nm, e in [("현금 피신 · 낙폭 5% 손절 (50% 현금)", SC.dd_rule(base, 5.0, 0.5)),
                      ("현금 피신 · 뉴스 감성 z>0.5 (전액 현금)", pd.Series(np.where(sig["뉴스"] > 0.5, 0.0, 1.0), idx)),
                      ("현금 피신 · 칼만 평균<0 (전액 현금)", pd.Series(np.where(SC.SA.kalman_mean(base.to_frame("m"), 1e-3)["m"] < 0, 0.0, 1.0), idx))]:
            rows.append({"이자 관점": view, "규칙": nm, "설정": "strategy_cash 학습 최고", **stats(run(base, rf, e, view), rf, e)})
    T = pd.DataFrame(rows)
    T.to_csv(A.DATA / "strategy_voltgt_dyn_compare.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 260)
    print(T.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
