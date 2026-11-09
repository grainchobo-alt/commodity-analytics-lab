"""3축 혼합(MinVar · 52주 모멘텀 · 가치)을 일간 데이터로 다시 돌리면 Sharpe가 어떻게 나오나 — 주간 결과(시험 1.07)와 같은 조건 비교.

방법: 같은 코드로 주간 자료 · 일간 자료를 모두 돌린다. 창 길이는 달력 기간이 같도록 ×5 (52주 = 252일, 26주 = 130일, 104주 = 520일, 5년 = 1,300일).
  W  주간 자료 + 주간 창           → 기존 sleeves 결과를 그대로 재현해야 함 (검증)
  D1 일간 자료 · 결정 주기는 주간과 같음(모멘텀 5일 · 가치 20일 · MinVar 20일), 손익 · 변동성 계산은 일간
  D2 일간 자료 · 매일 결정(모멘텀 1일 · 가치 5일 · MinVar 20일)
공통: 결정일 종가까지의 정보, 다음 날(주) 수익으로 손익, 편도 5bp(비중 변화량 기준), 축마다 목표 변동성 10%(직전 26주 실현, 상한 4배),
      혼합(50/25/25)은 수익률 가중합 후 다시 10%. 학습 2008–16 · 시험 2017–. 일간 Sharpe는 √252, 주간 환산 Sharpe(일간 수익을 주 단위로 합산, √52)도 함께 기록.
주의: 일간 자료는 휴장일 가격을 직전 값으로 채움(수익 0) — 품목별 거래소 휴일이 달라 비동기 마감 · 정체 가격 효과가 섞일 수 있음.
실행: .venv\\Scripts\\python.exe strategy_daily_freq.py → data/strategy_daily_freq_compare.csv, strategy_daily_freq_weekly.parquet
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
import strategy_meta as SM  # noqa: E402

COST, TGT, CAP = 5.0, 10.0, 0.25
FROZEN_END, START, SPLIT = "2026-10-26", "2005-01-01", "2017-01-01"
BASE = np.array([0.5, 0.25, 0.25])


def minvar_w(P, R, win, reb):
    from scipy.optimize import minimize
    from sklearn.covariance import LedoitWolf
    W = pd.DataFrame(np.nan, index=P.index, columns=P.columns)
    for t in range(win, len(P), reb):
        wd = R.iloc[t - win + 1:t + 1]
        cols = [c for c in P.columns if wd[c].notna().sum() >= win * 0.8]
        if len(cols) < 4:
            continue
        S = LedoitWolf().fit(wd[cols].fillna(0.0).values / 100).covariance_; n = len(cols); capn = max(CAP, 1.0 / n + 1e-6)
        res = minimize(lambda w: 1e4 * (w @ S @ w), np.ones(n) / n, jac=lambda w: 1e4 * 2 * S @ w, bounds=[(0, capn)] * n,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
        x = np.clip(res.x, 0, capn)
        w = x / x.sum() if (abs(res.x.sum() - 1) < 1e-3 and np.isfinite(res.x).all()) else np.ones(n) / n
        W.loc[P.index[t], cols] = w; W.loc[P.index[t], [c for c in P.columns if c not in cols]] = 0.0
    return W.ffill().fillna(0.0)


def sample(W, k):
    """k 기간마다만 비중 갱신 (사이 기간은 보유)."""
    if k <= 1:
        return W
    m = pd.Series(np.arange(len(W)) % k == 0, index=W.index)
    return W.where(m, np.nan).ffill().fillna(0.0)


def vt(W, R, vwin, ann, k):
    """목표 변동성 배율 (직전 vwin 기간 실현, 상한 4배). 배율도 k 기간마다만 갱신."""
    rb = (W.shift(1) * R).sum(1)
    pv = rb.rolling(vwin, min_periods=vwin // 2).std() * np.sqrt(ann)
    sc = (TGT / pv).clip(upper=4.0).fillna(0.0)
    if k > 1:
        m = pd.Series(np.arange(len(W)) % k == 0, index=W.index)
        sc = sc.where(m, np.nan).ffill().fillna(0.0)
    return W.mul(sc, axis=0)


def axes(P, R, ann, f, k_mom, k_val, k_mv, ks):
    """f = 주→기간 환산 배수 (주간 1, 일간 5). ks = 목표 변동성 배율 갱신 주기 (기존 주간 = 매주)."""
    lp = np.log(P)
    book = lambda W: LS._book(W, R, COST).loc[START:].fillna(0.0)          # noqa: E731
    vwin = 26 * f
    Wm = minvar_w(P, R, 104 * f, 4 * f if k_mv is None else k_mv)
    sig = np.sign(lp.diff(52 * f)); vol = R.rolling(vwin, min_periods=vwin // 2).std() * np.sqrt(ann)
    Wt = (sig * (TGT / vol)); Wt = Wt.div(Wt.notna().sum(1).replace(0, np.nan), axis=0).clip(-2, 2).fillna(0.0)
    Wv = LX.xs_weights(-(lp - lp.rolling(260 * f, min_periods=156 * f).mean()), hold=k_val)
    out = {}
    for nm, W in [("MinVar", Wm), ("모멘텀", sample(Wt, k_mom)), ("가치", Wv)]:
        out[nm] = book(vt(W, R, vwin, ann, ks))
    return pd.DataFrame(out).fillna(0.0)


def mix(S, w, vwin, ann):
    m = pd.Series(S.values @ np.asarray(w), S.index)
    pv = m.rolling(vwin, min_periods=vwin // 2).std().shift(1) * np.sqrt(ann)
    return m * (TGT / pv).clip(upper=4.0).fillna(1.0)


def stats(r, ann):
    r = r.dropna(); c = r.cumsum()
    sh = lambda x: float(x.mean() / x.std() * np.sqrt(ann)) if len(x) > 30 and x.std() > 0 else np.nan   # noqa: E731
    wk = r.resample("W-FRI").sum()
    return {"Sharpe 2008–": sh(r.loc["2008":]), "Sharpe 학습 2008–16": sh(r.loc["2008":"2016"]), "Sharpe 시험 2017–": sh(r.loc[SPLIT:]),
            "주간 환산 Sharpe 2008–": float(wk.loc["2008":].mean() / wk.loc["2008":].std() * np.sqrt(52)),
            "주간 환산 학습": float(wk.loc["2008":"2016"].mean() / wk.loc["2008":"2016"].std() * np.sqrt(52)),
            "주간 환산 시험": float(wk.loc[SPLIT:].mean() / wk.loc[SPLIT:].std() * np.sqrt(52)),
            "연수익 (%)": float(r.loc["2008":].mean() * ann), "연변동성 (%)": float(r.loc["2008":].std() * np.sqrt(ann)),
            "최대 낙폭 (%)": float((c.loc["2008":] - c.loc["2008":].cummax()).min())}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t00 = time.time()
    L = A.load_levels(roll_adjusted=True).loc[:FROZEN_END]
    S3, Rw, Pw = SM.sleeves(L)
    ids = [c for c in LS.BASKET if c in L]
    Pwk = LS.weekly(L, ids, "2001-01-01"); Pwk = Pwk[[c for c in ids if c in Pwk]]; Rwk = LS._ret(Pwk)
    Pd_ = L[ids].loc["2001-01-01":]
    Pd_ = Pd_[Pd_.index.dayofweek < 5].ffill(limit=5); Rd = LS._ret(Pd_)
    print(f"일간 표본 {len(Pd_):,}일, 품목 {Pd_.shape[1]}개 · 일간 수익률 0 비율 {(Rd.fillna(0).abs() < 1e-12).mean().mean():.1%} (주간 {(Rwk.fillna(0).abs() < 1e-12).mean().mean():.1%})", flush=True)
    cfg = {"W 주간 자료 (기존 재현)": (Pwk, Rwk, 52, 1, 1, 4, 4, 1),
           "D1 일간 자료 · 주간과 같은 결정 주기": (Pd_, Rd, 252, 5, 5, 20, 20, 5),
           "D2 일간 자료 · 모멘텀 매일 결정": (Pd_, Rd, 252, 5, 1, 5, 20, 1)}
    rows, store = [], {}
    for nm, (P, R, ann, f, km, kv, kmv, ks) in cfg.items():
        t0 = time.time()
        S = axes(P, R, ann, f, km, kv, kmv, ks); vwin = 26 * f
        if nm.startswith("W"):
            print("  [검증] 주간 재현 vs 기존 sleeves 최대 차이:", {c: round(float((S[c] - S3[c]).abs().max()), 6) for c in S.columns}, flush=True)
        books = {"MinVar 축": S["MinVar"], "모멘텀 축": S["모멘텀"], "가치 축": S["가치"], "혼합 50/25/25": mix(S, BASE, vwin, ann),
                 "혼합 MinVar 70 + 모멘텀 30 (기존)": mix(S[["MinVar", "모멘텀", "가치"]], [0.7, 0.3, 0.0], vwin, ann), "혼합 동일 비중": mix(S, np.full(3, 1 / 3), vwin, ann)}
        for bn, r in books.items():
            rows.append({"자료": nm, "전략": bn, **stats(r, ann)}); store[f"{nm}|{bn}"] = r
        print(f"  {nm} {time.time() - t0:.0f}s", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(A.DATA / "strategy_daily_freq_compare.csv", index=False, encoding="utf-8-sig")
    # 일간 → 주간 합산 시계열 (같은 주의 일간 결과와 주간 결과 상관)
    wk = pd.DataFrame({k.split("|")[0][:2] + "|" + k.split("|")[1]: v.resample("W-FRI").sum() for k, v in store.items()})
    wk.to_parquet(A.DATA / "strategy_daily_freq_weekly.parquet")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 40)
    print(T.round(2).to_string(index=False))
    for a in ["MinVar 축", "모멘텀 축", "가치 축", "혼합 50/25/25"]:
        c1 = wk[f"W |{a}"].corr(wk[f"D1|{a}"]); c2 = wk[f"W |{a}"].corr(wk[f"D2|{a}"])
        print(f"  주간 합산 상관 · {a}: W–D1 {c1:.2f} · W–D2 {c2:.2f}")
    print(f"총 {time.time() - t00:.0f}s")


if __name__ == "__main__":
    main()
