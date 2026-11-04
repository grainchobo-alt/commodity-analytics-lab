"""사후 최적 전략 미리 읽기(메타 레이어) 백테스트 — 기본 3축(MinVar · 52주 모멘텀 · 가치=5년 평균회귀) 비중을 '다음 분기에 어느 축이 이길지' 예측으로 기울이면 Sharpe가 오르나?

재료: ls_extra.run_extra와 같은 규칙으로 2001년부터 다시 만든 3축 주간 수익률(품목 비중 → 목표 10% · 상한 4배 · 편도 5bp 차감).
     첫 신호가 다 차는 2005년부터 사용. 학습 2005–2016 · 시험 2017–2026-10.
1) 연도별 사후 최적: 해마다 그해 자료로 최대 Sharpe 비중을 구해 표로 — '사후 최적'이 얼마나 자주 · 크게 바뀌는지.
2) 메타 레이어(모두 그 주까지 알 수 있는 값):
   L1 위기 국면 — 원자재 흡수비율(1주성분 몫) 수준 · 4주 변화 z
   L2 변동성 국면 — 바스켓 26주 실현 변동성 ÷ 직전 3년 중앙값
   L3 추세 환경 — 품목 52주 수익률 절댓값 평균(추세가 뚜렷할수록 모멘텀 유리 가설)
   L4 가치 스프레드 — 5년 평균 대비 괴리의 품목 간 폭(벌어질수록 가치 유리 가설)
   L5 축 자체 모멘텀 — 각 축의 직전 13 · 52주 수익
   L6 거시 — VIX 수준 · 달러 26주 변화 · 미 10년 금리 26주 변화
   → 축마다 '다음 13주 상대 수익(축 − 3축 평균)'을 릿지 회귀로 예측. 매년 말 그때까지 자료로만 다시 학습(목표가 겹치는 13주는 학습에서 뺌).
3) 비중: 기본 50/25/25(또는 동일 비중)에서 예측 1등 축 +κ, 꼴찌 축 −κ (4주마다, 0 아래는 0, 합 100%). κ는 학습 기간에서만 고름.
   갈아타기 비용 = |Δ비중| 합 × 명목 노출 1.15배 × 5bp, 섞은 뒤 목표 10%로 다시 맞춤(strategy_switch.run_weights).
4) 검증: 시험 Sharpe · 고정 대비 차이 t · 분기 1등 축 적중률(무작위 33%) · 위약(레이어 값을 13주 블록으로 섞어 200회 → 시험 Sharpe 분포).
실행: .venv\\Scripts\\python.exe strategy_meta.py → data/strategy_meta_compare.csv · strategy_meta_yearly.csv · strategy_meta_weekly.parquet
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
import analytics as A  # noqa: E402
import ls_extra as LX  # noqa: E402
import ls_strategies as LS  # noqa: E402
import strategy_switch as SS  # noqa: E402

ANN, TGT, COST = 52, 10.0, 5.0
FROZEN_END, START, SPLIT, H = "2026-10-26", "2005-01-01", "2017-01-01", 13
AXES = ["MinVar", "모멘텀", "가치"]
BASE = {"고정 50/25/25": np.array([0.5, 0.25, 0.25]), "동일 비중": np.full(3, 1 / 3)}


def sleeves(L: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """run_extra와 같은 3축(MinVar 상한 25% · 52주 TSMOM · V1 가치) 주간 순수익 %, 2001~. 품목 수익 R · 가격 P도 반환."""
    P = LS.weekly(L, LS.BASKET, "2001-01-01"); P = P[[c for c in LS.BASKET if c in P]]; R = LS._ret(P)
    lp = np.log(P)
    book = lambda W: LS._book(LS._vol_target(W, R, TGT), R, COST)   # noqa: E731
    S = pd.DataFrame({"MinVar": book(LS.alloc_weights_cap(P, 0.25)), "모멘텀": book(LS.tsmom_weights(P, 52, port_tgt=None)),
                      "가치": book(LX.xs_weights(-(lp - lp.rolling(260, min_periods=156).mean())))})
    return S.loc[START:].fillna(0.0), R, P


def layers(L: pd.DataFrame, R: pd.DataFrame, P: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    idx = S.index
    ar = LS.absorption(R, 52); dz = (ar - ar.shift(4)) / (ar - ar.shift(4)).rolling(156, min_periods=52).std()
    bv = R.std(1).rolling(26, min_periods=13).mean(); vreg = bv / bv.rolling(156, min_periods=52).median()
    lp = np.log(P)
    trend = lp.diff(52).abs().mean(1)
    dev = lp - lp.rolling(260, min_periods=156).mean(); vspread = dev.quantile(0.8, axis=1) - dev.quantile(0.2, axis=1)
    W = L.resample("W-FRI").last()
    mac = pd.DataFrame({"VIX": W.get("fred:VIXCLS"), "달러 26주": np.log(W.get("yf:DX-Y.NYB")).diff(26) * 100,
                        "10년 금리 26주": W.get("fred:DGS10").diff(26)})
    F = pd.DataFrame({"L1 흡수비율": ar, "L1 흡수비율 4주 z": dz, "L2 변동성 국면": vreg, "L3 추세 강도": trend, "L4 가치 스프레드": vspread}).join(mac)
    for a in AXES:
        F[f"L5 {a} 13주"] = S[a].rolling(13).sum(); F[f"L5 {a} 52주"] = S[a].rolling(52).sum()
    return F.reindex(idx).ffill()


def targets(S: pd.DataFrame) -> pd.DataFrame:
    fwd = S[::-1].rolling(H).sum()[::-1].shift(-1)                  # t+1 ~ t+13 누적
    return fwd.sub(fwd.mean(1), axis=0)


def walk_predict(F: pd.DataFrame, Y: pd.DataFrame, alpha: float = 10.0, first: int = 2008) -> pd.DataFrame:
    """매년 말 그때까지(목표가 끝난 표본만) 표준화 → 축별 릿지 → 다음 해 매주 예측."""
    from sklearn.linear_model import Ridge
    pred = pd.DataFrame(np.nan, index=F.index, columns=Y.columns)
    for yr in range(first, F.index.max().year + 1):
        cut = pd.Timestamp(f"{yr - 1}-12-31")
        tr = F.index[(F.index <= cut - pd.Timedelta(weeks=H))]
        Xtr = F.loc[tr].dropna(); ok = Xtr.index.intersection(Y.dropna().index)
        if len(ok) < 104:
            continue
        mu, sd = Xtr.loc[ok].mean(), Xtr.loc[ok].std().replace(0, 1)
        te = F.index[(F.index > cut) & (F.index <= pd.Timestamp(f"{yr}-12-31"))]
        Xte = ((F.loc[te] - mu) / sd).fillna(0)
        for a in Y.columns:
            m = Ridge(alpha=alpha).fit(((Xtr.loc[ok] - mu) / sd).values, Y.loc[ok, a].values)
            pred.loc[te, a] = m.predict(Xte.values)
    return pred


def tilt(pred: pd.DataFrame, base: np.ndarray, kappa: float, reb: int = 4) -> pd.DataFrame:
    W = pd.DataFrame(np.nan, index=pred.index, columns=pred.columns)
    for t in range(0, len(pred), reb):
        p = pred.iloc[t]
        w = base.copy()
        if p.notna().all() and kappa > 0:
            o = np.argsort(p.values); w[o[-1]] += kappa; w[o[0]] -= kappa
            w = np.clip(w, 0, None); w = w / w.sum()
        W.iloc[t] = w
    return W.ffill().fillna(pd.Series(base, pred.columns))


def ex_post_yearly(S: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for yr, X in S.groupby(S.index.year):
        if len(X) < 40:
            continue
        mu, C = X.mean().values * ANN, X.cov().values * ANN
        r = minimize(lambda w: -(w @ mu) / np.sqrt(w @ C @ w), np.full(3, 1 / 3), bounds=[(0, 1)] * 3,
                     constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], method="SLSQP")
        sh = {a: X[a].mean() / X[a].std() * np.sqrt(ANN) for a in AXES}
        rows.append({"연도": yr, **{f"사후 최적 {a} (%)": 100 * w for a, w in zip(AXES, r.x)}, "1등 축": max(sh, key=sh.get),
                     **{f"{a} Sharpe": v for a, v in sh.items()}, "50/25/25 Sharpe": float((X @ BASE["고정 50/25/25"]).mean() / (X @ BASE["고정 50/25/25"]).std() * np.sqrt(ANN))})
    T = pd.DataFrame(rows)
    T["전년 1등 축과 같음"] = T["1등 축"].eq(T["1등 축"].shift())
    return T


def score(r: pd.Series, bench: pd.Series) -> dict:
    ev = r.loc["2008":]; b = bench.loc["2008":]
    cum = (1 + ev / 100).cumprod()
    sh = lambda x: float(x.mean() / x.std() * np.sqrt(ANN))       # noqa: E731
    return {"Sharpe 2008–": sh(ev), "Sharpe 학습 2008–16": sh(ev.loc[:"2016"]), "Sharpe 시험 2017–": sh(ev.loc[SPLIT:]),
            "연수익 (%)": float(ev.mean() * ANN), "최대 낙폭 (%)": float((cum / cum.cummax() - 1).min() * 100),
            "시험 고정 대비 (%p)": float((ev - b).loc[SPLIT:].mean() * ANN), "시험 차이 t": SS.nw_t((ev - b).loc[SPLIT:])}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    L = A.load_levels(roll_adjusted=True).loc[:FROZEN_END]
    S, R, P = sleeves(L)
    print(f"3축 {S.index[0]:%Y-%m} ~ {S.index[-1]:%Y-%m}, 학습 ~2016 · 시험 2017~", flush=True)
    YT = ex_post_yearly(S); YT.to_csv(A.DATA / "strategy_meta_yearly.csv", index=False, encoding="utf-8-sig")
    print(YT[["연도", *[f"사후 최적 {a} (%)" for a in AXES], "1등 축", "전년 1등 축과 같음"]].round(0).to_string(index=False))
    print(f"전년 1등 축이 올해도 1등: {YT['전년 1등 축과 같음'].iloc[1:].mean():.0%} (무작위 33%)", flush=True)
    F = layers(L, R, P, S); Y = targets(S)
    pred = walk_predict(F, Y)
    ok_ = pred.dropna().index.intersection(Y.dropna().index)
    best_true, best_pred = Y.loc[ok_].idxmax(1), pred.loc[ok_].idxmax(1)
    q = ok_[::H]
    hit = lambda sl: float((best_true.loc[q][sl] == best_pred.loc[q][sl]).mean())   # noqa: E731
    print(f"분기 1등 축 적중률: 학습 {hit(slice(None, '2016')):.0%} · 시험 {hit(slice(SPLIT, None)):.0%} (무작위 33%)", flush=True)
    rows, weekly = [], {}
    bench = SS.run_weights(S, SS.const(S, dict(zip(AXES, BASE["고정 50/25/25"]))))
    for bn, base in BASE.items():
        rb = SS.run_weights(S, SS.const(S, dict(zip(AXES, base)))); weekly[bn] = rb
        rows.append({"규칙": bn, "κ": 0.0, **score(rb, bench)})
        res = {}
        for k in (0.1, 0.2, 0.3):
            W = tilt(pred, base, k); r = SS.run_weights(S, W); res[k] = (r, W)
        kb = max(res, key=lambda k: score(res[k][0], bench)["Sharpe 학습 2008–16"])
        for k, (r, W) in res.items():
            weekly[f"{bn} + 메타 κ={k}"] = r
            rows.append({"규칙": f"{bn} + 메타 레이어 기울이기 κ={k}" + (" ← 학습 최고" if k == kb else ""), "κ": k, **score(r, bench),
                         "연간 비중 변경": float(W.loc["2008":].diff().abs().sum(1).sum() / (len(W.loc['2008':]) / ANN))})
        if bn == "고정 50/25/25":                                    # 위약: 레이어 값을 13주 블록으로 섞어도 같은 성과가 나오나
            rng, pl = np.random.default_rng(0), []
            blocks = [F.index[i:i + H] for i in range(0, len(F), H)]
            for _ in range(200):
                order = rng.permutation(len(blocks)); Fi = pd.concat([F.loc[blocks[j]] for j in order]).iloc[:len(F)]; Fi.index = F.index
                rp = SS.run_weights(S, tilt(walk_predict(Fi, Y), base, kb))
                pl.append(score(rp, bench)["Sharpe 시험 2017–"])
            real = score(res[kb][0], bench)["Sharpe 시험 2017–"]
            rows[-3 + [0.1, 0.2, 0.3].index(kb)]["위약 p (시험 Sharpe)"] = float((np.array(pl) >= real).mean())
            print(f"위약 200회: 시험 Sharpe 실제 {real:.2f} vs 위약 중앙 {np.median(pl):.2f} · 95% {np.quantile(pl, 0.95):.2f}", flush=True)
    # 레이어별 기여: 레이어 하나씩만 써서 같은 κ로
    groups = {g: [c for c in F.columns if c.startswith(g) or (g == "L6" and c in ("VIX", "달러 26주", "10년 금리 26주"))] for g in ("L1", "L2", "L3", "L4", "L5", "L6")}
    for g, cols in groups.items():
        r = SS.run_weights(S, tilt(walk_predict(F[cols], Y), BASE["고정 50/25/25"], 0.2))
        rows.append({"규칙": f"고정 50/25/25 + {g}만 κ=0.2", "κ": 0.2, **score(r, bench)})
    T = pd.DataFrame(rows)
    T.to_csv(A.DATA / "strategy_meta_compare.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(weekly).to_parquet(A.DATA / "strategy_meta_weekly.parquet")
    pd.set_option("display.width", 250)
    print(T.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
