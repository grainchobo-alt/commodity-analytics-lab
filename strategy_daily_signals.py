"""신호 전략(DCC · 칼만 · HMM · VIX 변동성 국면 · 뉴스 감성)을 일간 자료로 다시 돌린다 — 현금 피신(D) · 환경 적응형 비중(C), 주간 결과와 같은 규칙.

재료: strategy_daily_freq의 일간 3축(MinVar · 52주 모멘텀 · 가치; 일간 자료 · 주간과 같은 결정 주기, 학습 2008–16 0.61 / 시험 2017– 주간 합산 1.24).
일간 신호 (창은 달력 기간 같게 ×5: 13주 = 65일, 52주 = 252일, 156주 = 780일, 4주 = 20일)
  변동성  원자재 바스켓 65일 실현 변동성 ÷ 그때까지 중앙값            — 일간에서 새로 계산
  흡수비율 19개 252일 1주성분 몫의 20일 변화 z                         — 일간에서 새로 계산
  VIX    VIX(FRED 일간) 수준의 그때까지 분위                             — 일간 원천
  동조성  원자재 8개 DCC 평균 상관 (20일마다 최근 3,900일로 재추정)     — 일간 DCC 새로 추정
  칼만   혼합 수익의 지역 수준 칼만 평균 < 0 (q는 주간과 같은 달력 속도 ÷ 5)  — 일간에서 새로 계산
  HMM    주간 HMM 위기 확률을 일간으로 이어 붙임 (regime_mv가 주간 · 월간만 지원 → 일간 HMM은 없음, 금요일에만 갱신)
  뉴스   베이지북 · WASDE · STEO 문서 감성 (문서 공표일에만 바뀜 → 일간으로 늘려도 새 정보 없음)
D 현금 피신: 노출 e ∈ {0, 0.5, 1}을 신호로 정해 다음 날 적용, 노출 변경 |Δe| × 1.15 × 5bp. 문턱 · 노출은 학습 2008–16 Sharpe로만 고름, 위약 = 현금 구간을 65일 블록으로 섞기 200회.
C 환경 적응형: 블랙-리터만 방식(strategy_adaptive.bl_weights) — 20일마다 3축 DCC 공분산, 신호별 20일 선행 수익 회귀 관점, 기본 50/25/25 ±25%p. λ · q는 학습에서 고름.
공통: 결정일 종가까지의 정보, 다음 날 수익으로 손익, 합친 뒤 직전 130일 실현 변동성으로 10%(상한 4배). 일간 Sharpe는 √252, 비교용 주간 합산 Sharpe(√52)도 기록.
실행: .venv\\Scripts\\python.exe strategy_daily_signals.py [D C]  (약 30~40분, 일간 DCC 추정이 대부분) → data/strategy_daily_signals_compare.csv
"""
from __future__ import annotations

import pickle
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import analytics as A  # noqa: E402
import ls_strategies as LS  # noqa: E402
import mvgarch as MG  # noqa: E402
import strategy_adaptive as SA  # noqa: E402
import strategy_cash as SC  # noqa: E402
import strategy_daily_freq as DF  # noqa: E402
import strategy_meta as SM  # noqa: E402
import strategy_switch as SS  # noqa: E402

ANN, GROSS, COST, TGT = 252, 1.15, 0.05, 10.0
H, REB, VWIN = 20, 20, 130
FROZEN_END, DEC0, SPLIT = "2026-10-26", "2008-01-01", "2017-01-01"
AX, BASE = SM.AXES, np.array([0.5, 0.25, 0.25])
CACHE = Path(tempfile.gettempdir()) / "cf_daily_signals_cache.pkl"


# ------------------------------------------------------------------ 공통
def run_weights_d(S: pd.DataFrame, Wt: pd.DataFrame) -> pd.Series:
    Wt = Wt.reindex(S.index).ffill().fillna(0.0)
    held = Wt.shift(1).fillna(0.0)
    raw = (held * S).sum(1) - held.diff().abs().sum(1).fillna(0) * GROSS * COST
    pv = raw.rolling(VWIN, min_periods=VWIN // 2).std().shift(1) * np.sqrt(ANN)
    return raw * (TGT / pv).clip(upper=4.0).fillna(1.0)


def nw_t(x: pd.Series, lags: int = 40) -> float:
    return SS.nw_t(x, lags)


def sh(x: pd.Series, ann=ANN) -> float:
    return float(x.mean() / x.std() * np.sqrt(ann)) if len(x) > 30 and x.std() > 0 else np.nan


def score(r: pd.Series, bench: pd.Series, e: pd.Series | None = None, lo: str = "2008") -> dict:
    ev, b = r.loc[lo:], bench.loc[lo:]
    wk = ev.resample("W-FRI").sum(); cum = ev.cumsum()
    o = {"Sharpe 2008–": sh(ev), "Sharpe 학습 2008–16": sh(ev.loc[:"2016"]), "Sharpe 시험 2017–": sh(ev.loc[SPLIT:]),
         "주간 합산 학습": sh(wk.loc[:"2016"], 52), "주간 합산 시험": sh(wk.loc[SPLIT:], 52), "주간 합산 2008–": sh(wk, 52),
         "연수익 (%)": float(ev.mean() * ANN), "연변동성 (%)": float(ev.std() * np.sqrt(ANN)), "최대 낙폭 (%)": float((cum - cum.cummax()).min()),
         "최악 1년 (%)": float(ev.groupby(ev.index.year).sum().min()),
         "시험 기준 대비 (%p)": float((ev - b).loc[SPLIT:].mean() * ANN), "시험 차이 t": nw_t((ev - b).loc[SPLIT:])}
    if e is not None:
        ee = e.reindex(ev.index).ffill().fillna(1.0)
        o["평균 노출"] = float(ee.mean()); o["현금 구간 (%)"] = float((ee < 1).mean() * 100); o["연 전환 횟수"] = float((ee.diff().abs() > 0).sum() / (len(ee) / ANN))
    return o


def load():
    L = A.load_levels(roll_adjusted=True).loc[:FROZEN_END]
    ids = [c for c in LS.BASKET if c in L]
    Pd_ = L[ids].loc["2001-01-01":]; Pd_ = Pd_[Pd_.index.dayofweek < 5].ffill(limit=5); Rd = LS._ret(Pd_)
    S = DF.axes(Pd_, Rd, ANN, 5, 5, 20, 20, 5)
    return L, S, Rd


def kalman_mean_d(S: pd.DataFrame, q: float) -> pd.DataFrame:
    out = pd.DataFrame(index=S.index, columns=S.columns, dtype=float)
    for c in S:
        x = S[c].values; r = float(np.var(x[:780])); Q = q * r; m, Pv = 0.0, r; vals = np.empty(len(x))
        for t, v in enumerate(x):
            Pv += Q; K = Pv / (Pv + r); m += K * (v - m); Pv *= (1 - K); vals[t] = m * ANN
        out[c] = vals
    return out


def absorption_d(R: pd.DataFrame, win=252) -> pd.Series:
    return LS.absorption(R, win)


# ------------------------------------------------------------------ 일간 신호
def daily_signals(L, S, Rd, cache: dict) -> dict:
    idx = S.index
    bv = Rd.std(1).rolling(65, min_periods=40).mean().reindex(idx)
    ar = absorption_d(Rd, 252); d20 = ar - ar.shift(20); dz = (d20 / d20.rolling(780, min_periods=260).std()).reindex(idx)
    vix = L["fred:VIXCLS"].reindex(Rd.index).ffill().reindex(idx)
    t0 = time.time()
    if "comove" not in cache:
        Ri = Rd[SA.DCC_IDS].dropna(); com = pd.Series(np.nan, index=idx)
        for t in range(0, len(idx), REB):
            d = idx[t]
            if d < pd.Timestamp(DEC0) - pd.Timedelta(weeks=8):
                continue
            try:
                com.loc[d] = float(MG.dcc_fit(Ri.loc[:d].iloc[-3900:], False, "GJR", "t", ANN)["avg_corr"].iloc[-1])
            except Exception:
                pass
            if t % 500 == 0:
                print(f"  동조성 DCC {d:%Y-%m}  {time.time() - t0:.0f}s", flush=True)
        cache["comove"] = com.ffill()
    stress = LS.stress_prob(L).reindex(idx.union(LS.stress_prob(L).index)).ffill().reindex(idx) if "stress" not in cache else cache["stress"]
    cache["stress"] = stress
    return {"HMM": stress, "변동성": bv / bv.expanding(520).median(), "동조성": SC.expanding_pct(cache["comove"], 520),
            "흡수비율": dz, "VIX": SC.expanding_pct(vix, 520), "뉴스": -SA.news_index(idx)}


# ------------------------------------------------------------------ D 현금 피신
def run_D(S, sig, bench_w):
    base = run_weights_d(S, bench_w); idx = S.index
    grids = {"HMM": [0.5, 0.7, 0.9], "변동성": [1.2, 1.5, 2.0], "동조성": [0.8, 0.9, 0.95], "흡수비율": [1.0, 1.5, 2.0], "VIX": [0.8, 0.9, 0.95], "뉴스": [0.5, 1.0, 1.5]}
    rows, store = [], {"고정 50/25/25": base}
    rows.append({"신호": "현금 없음 (기준)", "문턱": "", "현금 시 노출": 1.0, **score(base, base)})
    for frac in (0.7, 0.5):
        e = pd.Series(frac, idx); rows.append({"신호": f"늘 {int(frac * 100)}% 투자 (대조군)", "문턱": "", "현금 시 노출": frac, **score(SC.apply(base, e), base, e)})
    cands = {}
    for nm, s in sig.items():
        for thr in grids[nm]:
            for low in (0.5, 0.0):
                cands[(nm, thr, low)] = pd.Series(np.where(s > thr, low, 1.0), idx)
    km = kalman_mean_d(base.to_frame("m"), 2e-4)["m"]
    for low in (0.5, 0.0):
        cands[("칼만 평균 < 0", 0, low)] = pd.Series(np.where(km < 0, low, 1.0), idx)
        for x in (5.0, 10.0, 15.0):
            cands[("낙폭", x, low)] = SC.dd_rule(base, x, low)
    cnt = sum((sig[nm] > grids[nm][1]).astype(int) for nm in ("HMM", "변동성", "동조성", "VIX"))
    for k in (1, 2, 3):
        for low in (0.5, 0.0):
            cands[("결합 (4개 중 k개)", k, low)] = pd.Series(np.where(cnt >= k, low, 1.0), idx)
    res = {key: (SC.apply(base, e), e) for key, e in cands.items()}
    sc = {key: score(r, base, e) for key, (r, e) in res.items()}
    best = {}
    for key, s_ in sc.items():
        if key[0] not in best or s_["Sharpe 학습 2008–16"] > sc[best[key[0]]]["Sharpe 학습 2008–16"]:
            best[key[0]] = key
    for key, s_ in sc.items():
        rows.append({"신호": key[0], "문턱": key[1], "현금 시 노출": key[2], **s_, "학습 최고": "←" if best[key[0]] == key else ""})
    rng = np.random.default_rng(0); blocks = [idx[i:i + 65] for i in range(0, len(idx), 65)]
    for nm, key in best.items():
        r0, e0 = res[key]; real = sc[key]["Sharpe 시험 2017–"]; pl = []
        for _ in range(200):
            o = rng.permutation(len(blocks)); ep = pd.concat([e0.reindex(blocks[j]) for j in o]).iloc[:len(idx)]; ep.index = idx
            pl.append(score(SC.apply(base, ep), base, ep)["Sharpe 시험 2017–"])
        for row in rows:
            if row["신호"] == key[0] and row["문턱"] == key[1] and row["현금 시 노출"] == key[2]:
                row["위약 p (시험)"] = float((np.array(pl) >= real).mean())
        store[f"{nm} {key[1]} 노출 {key[2]}"] = res[key][0]
    return rows, store


# ------------------------------------------------------------------ C 환경 적응형 비중
def cond_view_d(S, sig, d):
    y = S[::-1].rolling(H).sum()[::-1].shift(-1)
    ok = S.index[(S.index <= d - pd.Timedelta(days=28))]
    x = sig.reindex(ok); m = x.notna() & y.loc[ok].notna().all(1)
    if m.sum() < 520 or not np.isfinite(sig.get(d, np.nan)):
        return np.full(3, np.nan)
    X = np.c_[np.ones(m.sum()), x[m].values]
    return np.array([(np.linalg.lstsq(X, y.loc[ok][c][m].values, rcond=None)[0] @ np.array([1.0, sig[d]])) * ANN / H for c in AX])


def build_C(S, cov, views, lam, kind):
    W = pd.DataFrame(np.nan, index=S.index, columns=AX)
    for d, Sig in cov.items():
        if kind == "D":
            W.loc[d] = SA.risk_parity(Sig); continue
        vs = [views[k].loc[d].values for k in (["K", "H", "N", "C"] if kind == "ALL" else [kind])]
        vs = [v for v in vs if np.isfinite(v).all()]
        W.loc[d] = SA.bl_weights(Sig, np.mean(vs, 0) if vs else None, lam)
    W.loc[:DEC0] = W.loc[:DEC0].fillna(pd.Series(BASE, AX))
    return W.ffill()


def run_C(S, sig, cache):
    idx = S.index; t0 = time.time()
    if "cov" not in cache:
        cov = {}
        for t in range(0, len(idx), REB):
            d = idx[t]
            if d < pd.Timestamp(DEC0) - pd.Timedelta(weeks=8):
                continue
            try:
                cov[d] = MG.dcc_fit(S.loc[:d], False, "GJR", "t", ANN)["H"][-1] * ANN
            except Exception:
                cov[d] = S.loc[:d].iloc[-780:].cov().values * ANN
            if t % 500 == 0:
                print(f"  3축 DCC {d:%Y-%m}  {time.time() - t0:.0f}s", flush=True)
        cache["cov"] = cov
    cov = cache["cov"]; dates = list(cov)
    sigs = {"H": sig["HMM"], "N": -sig["뉴스"], "C": sig["동조성"] if False else cache["comove"]}
    bench = run_weights_d(S, SS.const(S, dict(zip(AX, BASE))))
    rows, store = [{"규칙": "고정 50/25/25", "λ·q": "", **score(bench, bench)}], {"고정 50/25/25": bench}
    eq = run_weights_d(S, SS.const(S, {a: 1 / 3 for a in AX})); rows.append({"규칙": "동일 비중", "λ·q": "", **score(eq, bench)}); store["동일 비중"] = eq
    rd = run_weights_d(S, build_C(S, cov, {}, 0, "D")); rows.append({"규칙": "D 위험 균등 (3축 DCC 공분산)", "λ·q": "", **score(rd, bench)}); store["D 위험 균등"] = rd

    def views_from(sigs_, kal):
        V = {"K": kal.reindex(S.index)}
        for k, sg in sigs_.items():
            V[k] = pd.DataFrame({d: cond_view_d(S, sg, d) for d in dates}, index=AX).T.reindex(S.index)
        return V
    best = {}
    for q in (2e-5, 2e-4, 2e-3):                                    # 주간 q 1e-4 · 1e-3 · 1e-2와 같은 달력 속도
        V = views_from(sigs, kalman_mean_d(S, q))
        for kind in ("K", "H", "N", "C", "ALL"):
            if kind in ("H", "N", "C") and q != 2e-4:
                continue
            for lam in (0.25, 0.5, 1.0):
                r = run_weights_d(S, build_C(S, cov, V, lam, kind)); sc = score(r, bench)
                tag = f"q={q:g} λ={lam}" if kind in ("K", "ALL") else f"λ={lam}"
                rows.append({"규칙": {"K": "K 칼만 평균 추적", "H": "H HMM 위기 국면", "N": "N 뉴스 감성", "C": "C DCC 동조성", "ALL": "ALL 관점 4개 평균"}[kind], "λ·q": tag, **sc})
                if kind not in best or sc["Sharpe 학습 2008–16"] > best[kind][0]["Sharpe 학습 2008–16"]:
                    best[kind] = (sc, r, tag, lam, q)
        print(f"  C q={q:g} 완료 {time.time() - t0:.0f}s", flush=True)
    for kind, (sc, r, tag, lam, q) in best.items():
        store[f"{kind} {tag}"] = r
        for row in rows:
            if row["규칙"].startswith(kind + " ") and row["λ·q"] == tag:
                row["학습 최고"] = "←"
    sc, r, tag, lam, q = best["ALL"]; rng, pl = np.random.default_rng(0), []
    blocks = [idx[i:i + 65] for i in range(0, len(idx), 65)]
    for _ in range(60):
        o = rng.permutation(len(blocks)); sh_ = {}
        for k, sg in sigs.items():
            x = pd.concat([sg.reindex(blocks[j]) for j in o]).iloc[:len(idx)]; x.index = idx; sh_[k] = x
        Vp = views_from(sh_, kalman_mean_d(S, q)[list(rng.permutation(AX))].set_axis(AX, axis=1))
        pl.append(score(run_weights_d(S, build_C(S, cov, Vp, lam, "ALL")), bench)["Sharpe 시험 2017–"])
    real = sc["Sharpe 시험 2017–"]
    for row in rows:
        if row["규칙"].startswith("ALL") and row["λ·q"] == tag:
            row["위약 p (시험)"] = float((np.array(pl) >= real).mean())
    print(f"위약 60회(ALL {tag}): 시험 Sharpe 실제 {real:.2f} vs 위약 중앙 {np.median(pl):.2f} · 95% {np.quantile(pl, 0.95):.2f}", flush=True)
    return rows, store


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    which = set(sys.argv[1:]) or {"D", "C"}
    t00 = time.time()
    L, S, Rd = load()
    cache = pickle.load(open(CACHE, "rb")) if CACHE.exists() else {}
    bw = SS.const(S, dict(zip(AX, BASE)))
    base = run_weights_d(S, bw)
    print(f"일간 3축 기준 혼합: 학습 {sh(base.loc['2008':'2016']):.2f} · 시험 {sh(base.loc[SPLIT:]):.2f} (일간 √252) · 주간 합산 시험 {sh(base.loc[SPLIT:].resample('W-FRI').sum(), 52):.2f}", flush=True)
    sig = daily_signals(L, S, Rd, cache); pickle.dump(cache, open(CACHE, "wb"))
    print(f"신호 준비 {time.time() - t00:.0f}s", flush=True)
    allrows, store = [], {}
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    if "D" in which:
        rows, st = run_D(S, sig, bw)
        for r in rows:
            allrows.append({"실험": "D 현금 피신 (일간)", **r})
        store.update({f"D|{k}": v for k, v in st.items()})
        T = pd.DataFrame(rows); show = T[(T.get("학습 최고") == "←") | T["신호"].str.contains("기준|대조군")]
        print(show[["신호", "문턱", "현금 시 노출", "Sharpe 학습 2008–16", "Sharpe 시험 2017–", "주간 합산 시험", "최대 낙폭 (%)", "평균 노출", "위약 p (시험)"]].round(2).to_string(index=False), flush=True)
    if "C" in which:
        rows, st = run_C(S, sig, cache); pickle.dump(cache, open(CACHE, "wb"))
        for r in rows:
            allrows.append({"실험": "C 환경 적응형 (일간)", "신호": r.pop("규칙"), "문턱": r.pop("λ·q"), **r})
        store.update({f"C|{k}": v for k, v in st.items()})
        T = pd.DataFrame([r for r in allrows if r["실험"].startswith("C")])
        show = T[(T.get("학습 최고") == "←") | T["신호"].str.contains("고정|동일|D ")]
        print(show[["신호", "문턱", "Sharpe 학습 2008–16", "Sharpe 시험 2017–", "주간 합산 시험", "시험 차이 t", "위약 p (시험)"]].round(2).to_string(index=False), flush=True)
    suffix = "" if which == {"D", "C"} else "_" + "".join(sorted(which))
    pd.DataFrame(allrows).to_csv(A.DATA / f"strategy_daily_signals_compare{suffix}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({k: v.resample("W-FRI").sum() for k, v in store.items()}).to_parquet(A.DATA / f"strategy_daily_signals_weekly{suffix}.parquet")
    print(f"저장 완료 · 총 {time.time() - t00:.0f}s")


if __name__ == "__main__":
    main()
