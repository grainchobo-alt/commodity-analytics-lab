"""전략 갈아타기(트리거) 백테스트 — '해마다 투자선 위의 최적 조합으로 바꾸면 고정 혼합(50/25/25)보다 나은가?'

재료 = 롱숏 탭 전략의 주간 수익률(data/voltgt_weekly.parquet, 각각 목표 변동성 10% · 상한 4배 · 편도 5bp 차감, 2007–2026-10).
  기본 3종: MinVar 롱 · 52주 모멘텀 · 가치 / 확장 4종: + 섹터 내 따라잡기(평균회귀)
규칙 (모든 판단은 그 시점까지의 자료만 — 미래정보 없음)
  고정      : 50/25/25(사후 선택 — 2007–2026 결과를 보고 고른 비중이라 이 비교에서 유리) · 동일 비중(사전 정보 없음)
  연 1회    : 매년 말 직전 N년(3 · 5)으로 최대 Sharpe(투자선의 접점) · 최소분산 · 역변동성 · 축소(50% 고정 + 50% 최대 Sharpe)
  트리거    : T1 성과 — 지금 혼합의 직전 26주 Sharpe < 문턱이면 재최적화(최소 13주 간격)
              T2 위기 — 원자재 19개 흡수비율 4주 변화 z > 문턱이면 MinVar 100%, 풀리면 원래 비중
              T3 전략 모멘텀 — 4주마다 직전 52주 수익 상위 k개 동일 비중
              T4 투자선 이탈 — 지금 비중이 직전 5년 투자선보다 같은 위험에서 수익이 x%p 이상 낮으면 최대 Sharpe로 재최적화(최소 26주 간격)
              결합 — 고정 50/25/25 + T2 위기 전환 + T4 이탈 재최적화(축소)
  · 비중을 바꾸면 갈아타기 비용 = |Δ비중| 합 × 평균 명목 노출 1.15배 × 5bp
  · 섞은 뒤 직전 26주 실현 변동성으로 목표 10%에 다시 맞춤(배율 상한 4배) — 대시보드 Portfolio Customization과 같은 방식
  · 트리거 문턱은 학습 기간(2012–2016)에서만 고르고 2017– 시험 기간에서 확인. 고정 50/25/25 대비 차이의 Newey-West t
실행: .venv\\Scripts\\python.exe strategy_switch.py  → data/strategy_switch_compare.csv · data/strategy_switch_weekly.parquet
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
import analytics as A  # noqa: E402
import ls_strategies as LS  # noqa: E402

ANN, TGT, COST, GROSS = 52, 10.0, 0.05, 1.15                  # 수익률 % 단위 → 5bp = 0.05
FROZEN_END = "2026-10-26"
BASE3 = {"MinVar": "기준 · MinVar 롱 (⑥)", "모멘텀": "기준 · 52주 모멘텀 (④)", "가치": "V1 가치 (5년 평균 대비 싼 품목 롱)"}
EXT = {**BASE3, "따라잡기": "S2 섹터 내 따라잡기 (과열 숏·낙오 롱)"}
FIX = {"MinVar": 0.5, "모멘텀": 0.25, "가치": 0.25}
EVAL0, SPLIT = "2012-01-01", "2017-01-01"                     # 5년 창이 처음 차는 2012부터 평가, 2017–는 시험
OUT = A.DATA / "strategy_switch_compare.csv"


def load(uni: dict) -> pd.DataFrame:
    W = pd.read_parquet(A.DATA / "voltgt_weekly.parquet")
    return pd.DataFrame({k: W[f"{v}|10|4"] for k, v in uni.items()}).loc["2007":].dropna()


# ---------------------------------------------------------------------------- 비중 계산 (과거 창 X만 사용)
def _opt(X: pd.DataFrame, kind: str) -> np.ndarray:
    mu, S, k = X.mean().values * ANN, X.cov().values * ANN, X.shape[1]
    w0, b, c = np.full(k, 1 / k), [(0, 1)] * k, [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if kind == "maxsharpe":
        f = lambda w: -(w @ mu) / np.sqrt(w @ S @ w)            # noqa: E731
    elif kind == "minvar":
        f = lambda w: w @ S @ w                                  # noqa: E731
    elif kind == "invvol":
        w = 1 / np.sqrt(np.diag(S)); return w / w.sum()
    else:
        raise ValueError(kind)
    r = minimize(f, w0, bounds=b, constraints=c, method="SLSQP")
    return r.x if r.success else w0


def frontier_gap(X: pd.DataFrame, w: np.ndarray) -> float:
    """지금 비중 w가 X로 그린 투자선보다 같은 위험에서 수익이 몇 %p 낮은가 (≥ 0)."""
    mu, S, k = X.mean().values * ANN, X.cov().values * ANN, X.shape[1]
    v = float(np.sqrt(w @ S @ w))
    c = [{"type": "eq", "fun": lambda x: x.sum() - 1}, {"type": "ineq", "fun": lambda x: v ** 2 - x @ S @ x}]
    r = minimize(lambda x: -(x @ mu), w, bounds=[(0, 1)] * k, constraints=c, method="SLSQP")
    return max(0.0, float(-r.fun - w @ mu)) if r.success else 0.0


# ---------------------------------------------------------------------------- 비중 경로 → 순수익
def run_weights(R: pd.DataFrame, Wt: pd.DataFrame) -> pd.Series:
    """Wt[t] = t주 말에 정한 비중 → t+1주 수익에 적용. 갈아타기 비용 차감 후 목표 10%로 다시 맞춤."""
    Wt = Wt.reindex(R.index).ffill().fillna(0.0)
    held = Wt.shift(1).fillna(0.0)
    raw = (held * R).sum(1) - held.diff().abs().sum(1).fillna(0) * GROSS * COST
    pv = raw.rolling(26, min_periods=13).std().shift(1) * np.sqrt(ANN)
    return raw * (TGT / pv).clip(upper=4.0 * TGT / 10.0).fillna(1.0)


def const(R, w: dict) -> pd.DataFrame:
    return pd.DataFrame([[w.get(c, 0.0) for c in R.columns]] * len(R), index=R.index, columns=R.columns)


def yearly(R, kind: str, years: int = 5, shrink: float = 0.0) -> pd.DataFrame:
    Wt = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
    fix = np.array([FIX.get(c, 0.0) for c in R.columns])
    for d in R.index[R.index.to_series().dt.year.diff().fillna(0) != 0] - pd.Timedelta(days=7):   # 매년 마지막 주
        d = R.index[R.index <= d][-1]
        X = R.loc[d - pd.DateOffset(years=years):d]
        if len(X) < 52 * years * 0.9:
            continue
        w = _opt(X, kind)
        Wt.loc[d] = (1 - shrink) * w + shrink * fix
    first = Wt.dropna().index.min()
    Wt.loc[:first] = Wt.loc[:first].fillna(pd.Series(fix if fix.sum() else np.full(len(fix), 1 / len(fix)), R.columns))
    return Wt


def t1_perf(R, thr: float, years: int = 5) -> pd.DataFrame:
    Wt = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
    w = np.array([FIX.get(c, 1 / R.shape[1] if not FIX else 0.0) for c in R.columns]); last = -99
    for t in range(len(R)):
        if t >= 26 and t - last >= 13 and R.index[t] >= pd.Timestamp(R.index[0] + pd.DateOffset(years=years)):
            m = R.iloc[t - 25:t + 1] @ w
            if m.std() > 0 and m.mean() / m.std() * np.sqrt(ANN) < thr:
                w = _opt(R.loc[R.index[t] - pd.DateOffset(years=years):R.index[t]], "maxsharpe"); last = t
        Wt.iloc[t] = w
    return Wt


def crisis_signal(R_items: pd.DataFrame) -> pd.Series:
    ar = LS.absorption(R_items, 52)
    d = ar - ar.shift(4)
    return d / d.rolling(156, min_periods=52).std()


def t2_crisis(R, base: pd.DataFrame, z: pd.Series, thr: float) -> pd.DataFrame:
    on = (z.reindex(R.index).ffill() > thr)
    W = base.reindex(R.index).ffill().copy()
    safe = pd.Series([1.0 if c == "MinVar" else 0.0 for c in R.columns], R.columns)
    W.loc[on] = safe.values
    return W


def t3_mom(R, k: int = 2, look: int = 52, reb: int = 4) -> pd.DataFrame:
    Wt = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
    for t in range(look, len(R), reb):
        top = R.iloc[t - look + 1:t + 1].sum().nlargest(k).index
        Wt.iloc[t] = [1 / k if c in top else 0.0 for c in R.columns]
    return Wt.fillna(Wt.bfill().iloc[0]) if Wt.notna().any().any() else Wt


def t4_gap(R, gap: float, years: int = 5, shrink: float = 0.0, check: int = 4, min_gap: int = 26) -> pd.DataFrame:
    Wt = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
    fix = np.array([FIX.get(c, 0.0) for c in R.columns]); w = fix.copy(); last = -99
    for t in range(0, len(R)):
        d = R.index[t]
        if t % check == 0 and t - last >= min_gap and d >= R.index[0] + pd.DateOffset(years=years):
            X = R.loc[d - pd.DateOffset(years=years):d]
            if frontier_gap(X, w) > gap:
                w = (1 - shrink) * _opt(X, "maxsharpe") + shrink * fix; last = t
        Wt.iloc[t] = w
    return Wt


# ---------------------------------------------------------------------------- 평가
def nw_t(x: pd.Series, lags: int = 8) -> float:
    x = x.dropna(); n = len(x); e = x - x.mean()
    v = (e @ e) / n + 2 * sum((1 - l / (lags + 1)) * (e[l:].values @ e[:-l].values) / n for l in range(1, lags + 1))
    return float(x.mean() / np.sqrt(v / n)) if v > 0 else np.nan


def evaluate(r: pd.Series, bench: pd.Series, Wt: pd.DataFrame | None) -> dict:
    r = r.loc[EVAL0:]; b = bench.loc[EVAL0:]
    m = LS.metrics(r); cum = (1 + r / 100).cumprod()
    out = {"Sharpe": m["Sharpe"], "연수익 (%)": m["연수익 (%)"], "연변동성 (%)": m["연변동성 (%)"],
           "최대 낙폭 (%)": float((cum / cum.cummax() - 1).min() * 100),
           "Sharpe 학습 2012–16": LS.metrics(r.loc[:"2016"])["Sharpe"], "Sharpe 시험 2017–": LS.metrics(r.loc[SPLIT:])["Sharpe"],
           "고정 대비 연수익 차 (%p)": float((r - b).mean() * ANN), "차이 t (NW)": nw_t(r - b),
           "시험 고정 대비 (%p)": float((r - b).loc[SPLIT:].mean() * ANN), "시험 차이 t": nw_t((r - b).loc[SPLIT:])}
    if Wt is not None:
        W = Wt.reindex(r.index).ffill()
        out["연간 비중 변경 (|Δw| 합)"] = float(W.diff().abs().sum(1).sum() / (len(W) / ANN))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    L = A.load_levels(roll_adjusted=True).loc[:FROZEN_END]
    P = LS.weekly(L, LS.BASKET, "2001-01-01"); z = crisis_signal(LS._ret(P[[c for c in LS.BASKET if c in P]]))
    rows, weekly = [], {}
    for uni_nm, uni in [("3종", BASE3), ("4종 (+따라잡기)", EXT)]:
        R = load(uni)
        fixW = const(R, FIX); bench = run_weights(R, fixW)
        eqW = const(R, {c: 1 / R.shape[1] for c in R.columns})

        def add(name, Wt, note=""):
            r = run_weights(R, Wt); weekly[f"{uni_nm} | {name}"] = r
            rows.append({"재료": uni_nm, "규칙": name, "메모": note, **evaluate(r, bench, Wt)})
        add("고정 50/25/25 (사후 선택)", fixW, "비교 기준 — 2007–2026 결과를 보고 고른 비중")
        add("고정 동일 비중", eqW, "사전 정보 없는 기준")
        for yrs in (3, 5):
            for kind, nm in [("maxsharpe", "최대 Sharpe(접점)"), ("minvar", "최소분산"), ("invvol", "역변동성")]:
                add(f"연 1회 · 직전 {yrs}년 {nm}", yearly(R, kind, yrs))
            add(f"연 1회 · 직전 {yrs}년 축소(고정 50% + 접점 50%)", yearly(R, "maxsharpe", yrs, 0.5))
        # 트리거: 문턱은 학습 기간 Sharpe로 고름
        def pick(make, grid, label):
            best = max(grid, key=lambda g: evaluate(run_weights(R, make(g)), bench, None)["Sharpe 학습 2012–16"])
            for g in grid:
                add(f"{label} {g}" + (" ← 학습 최고" if g == best else ""), make(g), "문턱 후보")
            return best
        b1 = pick(lambda g: t1_perf(R, g), [-0.5, 0.0, 0.5], "T1 성과 트리거 · 26주 Sharpe <")
        b2 = pick(lambda g: t2_crisis(R, fixW, z, g), [1.0, 1.5, 2.0], "T2 위기 트리거 · 흡수비율 z >")
        b3 = pick(lambda g: t3_mom(R, g), [1, 2], "T3 전략 모멘텀 · 상위 k =")
        b4 = pick(lambda g: t4_gap(R, g), [1.0, 2.0, 4.0], "T4 투자선 이탈 · 갭 %p >")
        comb = t2_crisis(R, t4_gap(R, b4, shrink=0.5), z, b2)
        add(f"결합 · 고정 + T2(z>{b2}) + T4(갭>{b4}%p, 축소 50%)", comb, "학습에서 고른 문턱으로 결합")
        print(f"[{uni_nm}] 학습 최고 문턱: T1 {b1} · T2 {b2} · T3 {b3} · T4 {b4}", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(OUT, index=False, encoding="utf-8-sig")
    pd.DataFrame(weekly).to_parquet(A.DATA / "strategy_switch_weekly.parquet")
    pd.set_option("display.width", 250)
    print(T.drop(columns="메모").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
