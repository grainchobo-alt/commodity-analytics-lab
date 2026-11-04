"""현금 파킹(위험 회피) 백테스트 — 위험 신호가 켜지면 3축 혼합의 일부 · 전부를 현금(이자 0 가정)으로 옮기면 Sharpe · 낙폭이 나아지나?

기본 포트폴리오: ① 고정 50/25/25 ② DCC 위험 균등(strategy_adaptive) — 둘 다 목표 변동성 10%로 이미 맞춘 주간 순수익(비용 포함).
노출 e_t ∈ {0, 0.5, 1}: t주 말 신호로 정해 t+1주에 적용. 현금 몫 수익 0. 노출을 바꾸면 |Δe| × 평균 명목 노출 1.15 × 5bp 비용.
  ※ 현금으로 줄인 뒤 다시 목표 변동성으로 키우지 않음(그러면 현금 효과가 사라짐).
위험 신호(모두 그 주까지 알 수 있는 값)
  HMM   위기 국면 확률(원자재 6 + 거시 8, 매년 재추정 · 전방 필터) > 문턱
  변동성 원자재 바스켓 13주 실현 변동성 ÷ 그때까지 중앙값 > 문턱
  동조성 원자재 8개 DCC 평균 상관(4주마다 재추정)의 그때까지 분위 > 문턱
  흡수비율 원자재 19개 1주성분 몫 4주 변화 z > 문턱
  VIX   VIX 수준의 그때까지 분위 > 문턱
  뉴스   문서 감성 지수(베이지북 · WASDE · STEO) z < −문턱
  칼만   혼합 수익의 칼만 평균(지금의 평균 수익) < 0
  낙폭   혼합 누적 수익이 고점 대비 −x% 아래로 떨어지면 현금, 고점 대비 −x/2% 위로 회복하면 복귀
  결합   HMM · 변동성 · 동조성 · VIX 중 k개 이상 켜짐
문턱 · 노출 수준은 학습 2008–2016 Sharpe로만 고르고 시험 2017–2026-10에서 확인. 위약: 현금 구간을 13주 블록으로 섞어(같은 현금 비율) 200회.
실행: .venv\\Scripts\\python.exe strategy_cash.py → data/strategy_cash_compare.csv · strategy_cash_weekly.parquet
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import analytics as A  # noqa: E402
import ls_strategies as LS  # noqa: E402
import mvgarch as MG  # noqa: E402
import strategy_adaptive as SA  # noqa: E402
import strategy_meta as SM  # noqa: E402
import strategy_switch as SS  # noqa: E402

ANN, GROSS, COST = 52, 1.15, 0.05
FROZEN_END, DEC0, SPLIT = "2026-10-26", "2008-01-01", "2017-01-01"


def expanding_pct(x: pd.Series, min_n: int = 104) -> pd.Series:
    """그때까지 값들 중 지금 값의 분위 (0~1)."""
    v = x.values; out = np.full(len(v), np.nan)
    for t in range(len(v)):
        h = v[:t + 1]; h = h[np.isfinite(h)]
        if len(h) >= min_n and np.isfinite(v[t]):
            out[t] = (h <= v[t]).mean()
    return pd.Series(out, x.index)


def comove_dcc(R: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.Series:
    Ri = R[SA.DCC_IDS].dropna(); s = pd.Series(np.nan, index=idx)
    for t in range(0, len(idx), 4):
        d = idx[t]
        if d < pd.Timestamp(DEC0) - pd.Timedelta(weeks=8):
            continue
        try:
            s.loc[d] = float(MG.dcc_fit(Ri.loc[:d].iloc[-780:], False, "GJR", "t", ANN)["avg_corr"].iloc[-1])
        except Exception:
            pass
    return s.ffill()


def apply(r: pd.Series, e: pd.Series) -> pd.Series:
    e = e.reindex(r.index).ffill().fillna(1.0).clip(0, 1)
    held = e.shift(1).fillna(1.0)
    return held * r - held.diff().abs().fillna(0) * GROSS * COST


def dd_rule(r: pd.Series, x: float, low: float) -> pd.Series:
    cum, peak, on, e = 0.0, 0.0, False, []
    for v in r.values:
        cum += v; peak = max(peak, cum); dd = cum - peak            # 누적 % (혼합 자체 기준 — 현금 여부와 무관)
        if not on and dd < -x:
            on = True
        elif on and dd > -x / 2:
            on = False
        e.append(low if on else 1.0)
    return pd.Series(e, r.index)


def score(r: pd.Series, bench: pd.Series, e: pd.Series | None = None) -> dict:
    o = SM.score(r, bench)
    ev = r.loc["2008":]
    o["최악 1년 (%)"] = float(ev.groupby(ev.index.year).sum().min())
    o["연변동성 (%)"] = float(ev.std() * np.sqrt(ANN))
    if e is not None:
        ee = e.reindex(ev.index).ffill().fillna(1.0)
        o["평균 노출"] = float(ee.mean()); o["현금 구간 (%)"] = float((ee < 1).mean() * 100)
        o["연 전환 횟수"] = float((ee.diff().abs() > 0).sum() / (len(ee) / ANN))
    return o


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    L = A.load_levels(roll_adjusted=True).loc[:FROZEN_END]
    S, R, P = SM.sleeves(L); idx = S.index
    bases = {"고정 50/25/25": SS.run_weights(S, SS.const(S, dict(zip(SM.AXES, SM.BASE["고정 50/25/25"]))))}
    Wad = pd.read_parquet(A.DATA / "strategy_adaptive_weights.parquet")
    bases["DCC 위험 균등"] = SS.run_weights(S, Wad["D"].reindex(idx).ffill())
    bench = bases["고정 50/25/25"]
    print("신호 계산 중 (HMM · DCC)…", flush=True)
    bv = R.std(1).rolling(13, min_periods=8).mean().reindex(idx)
    ar = LS.absorption(R, 52); dz = (ar - ar.shift(4)) / (ar - ar.shift(4)).rolling(156, min_periods=52).std()
    Wk = L.resample("W-FRI").last()
    sig = {"HMM": LS.stress_prob(L).reindex(idx).ffill(),
           "변동성": (bv / bv.expanding(104).median()),
           "동조성": expanding_pct(comove_dcc(R, idx)),
           "흡수비율": dz.reindex(idx).ffill(),
           "VIX": expanding_pct(Wk["fred:VIXCLS"].reindex(idx).ffill()),
           "뉴스": -SA.news_index(idx)}
    grids = {"HMM": [0.5, 0.7, 0.9], "변동성": [1.2, 1.5, 2.0], "동조성": [0.8, 0.9, 0.95], "흡수비율": [1.0, 1.5, 2.0],
             "VIX": [0.8, 0.9, 0.95], "뉴스": [0.5, 1.0, 1.5]}
    rows, weekly = [], {}
    for bn, base in bases.items():
        rows.append({"기본": bn, "신호": "현금 없음 (기준)", "문턱": "", "현금 시 노출": 1.0, **score(base, bench)}); weekly[f"{bn} | 기준"] = base
        for frac in (0.7, 0.5):                                       # 신호 없이 늘 일부 현금 — 타이밍 효과와 구분하는 대조군
            e = pd.Series(frac, idx)
            rows.append({"기본": bn, "신호": f"늘 {int(frac*100)}% 투자 (대조군)", "문턱": "", "현금 시 노출": frac, **score(apply(base, e), bench, e)})
        cands = {}
        for nm, s in sig.items():
            for thr in grids[nm]:
                for low in (0.5, 0.0):
                    cands[(nm, thr, low)] = pd.Series(np.where(s > thr, low, 1.0), idx)
        km = SA.kalman_mean(base.to_frame("m"), 1e-3)["m"]
        for low in (0.5, 0.0):
            cands[("칼만 평균 < 0", 0, low)] = pd.Series(np.where(km < 0, low, 1.0), idx)
            for x in (5.0, 10.0, 15.0):
                cands[("낙폭", x, low)] = dd_rule(base, x, low)
        flags = {nm: sig[nm] > grids[nm][1] for nm in ("HMM", "변동성", "동조성", "VIX")}   # 결합은 각 신호의 가운데 문턱
        cnt = sum(f.astype(int) for f in flags.values())
        for k in (1, 2, 3):
            for low in (0.5, 0.0):
                cands[("결합 (4개 중 k개)", k, low)] = pd.Series(np.where(cnt >= k, low, 1.0), idx)
        res = {key: (apply(base, e), e) for key, e in cands.items()}
        sc = {key: score(r, bench, e) for key, (r, e) in res.items()}
        best_by = {}
        for key, s_ in sc.items():
            nm = key[0]
            if nm not in best_by or s_["Sharpe 학습 2008–16"] > sc[best_by[nm]]["Sharpe 학습 2008–16"]:
                best_by[nm] = key
        for key, s_ in sc.items():
            rows.append({"기본": bn, "신호": key[0], "문턱": key[1], "현금 시 노출": key[2], **s_, "학습 최고": "←" if best_by[key[0]] == key else ""})
        for nm, key in best_by.items():
            weekly[f"{bn} | {nm} {key[1]} 노출 {key[2]}"] = res[key][0]
        if bn == "고정 50/25/25":                                     # 위약: 학습 최고 신호들의 현금 구간을 13주 블록으로 섞기
            rng = np.random.default_rng(0)
            for nm, key in best_by.items():
                r0, e0 = res[key]; real = sc[key]["Sharpe 시험 2017–"]
                blocks = [idx[i:i + 13] for i in range(0, len(idx), 13)]
                pl = []
                for _ in range(200):
                    o = rng.permutation(len(blocks)); ep = pd.concat([e0.reindex(blocks[j]) for j in o]).iloc[:len(idx)]; ep.index = idx
                    pl.append(score(apply(base, ep), bench, ep)["Sharpe 시험 2017–"])
                for row in rows:
                    if row["기본"] == bn and row["신호"] == key[0] and row["문턱"] == key[1] and row["현금 시 노출"] == key[2]:
                        row["위약 p (시험)"] = float((np.array(pl) >= real).mean())
    T = pd.DataFrame(rows)
    T.to_csv(A.DATA / "strategy_cash_compare.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(weekly).to_parquet(A.DATA / "strategy_cash_weekly.parquet")
    pd.set_option("display.width", 260)
    cols = ["기본", "신호", "문턱", "현금 시 노출", "Sharpe 학습 2008–16", "Sharpe 시험 2017–", "Sharpe 2008–", "연수익 (%)", "최대 낙폭 (%)", "최악 1년 (%)",
            "평균 노출", "연 전환 횟수", "위약 p (시험)"]
    show = T[(T["학습 최고"] == "←") | T["신호"].str.contains("기준|대조군")]
    print(show[[c for c in cols if c in show]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
