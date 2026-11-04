"""환경 적응형 3축 비중 백테스트 — 칼만 · HMM · 뉴스 감성 · DCC 동조성으로 MinVar · 52주 모멘텀 · 가치 비중을 유연하게 조정하면 고정 50/25/25보다 나은가?

재료: strategy_meta.sleeves — 3축 주간 순수익(2001년 자료로 재구성, 2005~, 목표 10% · 상한 4배 · 편도 5bp). 판단 2008~, 학습 2008–2016 · 시험 2017–2026-10.
모든 신호는 그 주까지 알 수 있는 값만 쓰고, 신호 → 축별 기대수익 모형은 4주마다 그때까지 자료로 다시 추정(목표 4주가 겹치는 표본 제외).
신호(관점)
  K 칼만     : 축마다 지역 수준 모형(수익 = 천천히 변하는 평균 + 잡음)으로 '지금의 평균 수익'을 추적 → 안 먹히는 축은 평균이 내려가 비중 축소
  H HMM      : 원자재 6 + 거시 8 다변량 HMM의 '위기 국면 확률'(매년 재추정, 전방 필터) → 국면별로 어느 축이 잘 됐는지 회귀
  N 뉴스     : 문서 감성 지수 = 연준 베이지북(농업 · 에너지 · 금속) · USDA WASDE · EIA STEO 문장 감성(FinBERT) 평균(공표 후 날짜에만) → 감성별 축 성과 회귀
  C 동조성   : 원자재 8개 DCC 평균 상관(4주마다 그때까지 자료로 재추정) → 동조성 높을 때 · 낮을 때 축 성과 회귀
비중: 블랙-리터만 방식 — 기본 50/25/25가 최적이 되는 '기본 기대수익'(축 공분산은 3축 DCC로 4주마다 갱신)을
     관점 쪽으로 λ만큼 옮긴 뒤 최대 Sharpe 비중. 기본 대비 ±25%p 안 · 롱온리 · 합 100%. 4주마다.
     D = 관점 없이 DCC 공분산으로 위험 균등(축마다 같은 위험) — 순수 위험 기반 적응.
     ALL = K · H · N · C 관점 평균.
비용: 축 안 거래 5bp(재료에 포함) + 비중 변경 |Δw| × 1.15 × 5bp. 섞은 뒤 목표 10%로 다시 맞춤(strategy_switch.run_weights).
λ는 학습 기간 Sharpe로만 고름. 검증: 시험 Sharpe · 고정 대비 차이 t · 위약(신호를 13주 블록으로 섞기 100회).
실행: .venv\\Scripts\\python.exe strategy_adaptive.py → data/strategy_adaptive_compare.csv · strategy_adaptive_weekly.parquet · strategy_adaptive_weights.parquet
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
import mvgarch as MG  # noqa: E402
import strategy_meta as SM  # noqa: E402
import strategy_switch as SS  # noqa: E402

ANN, H, REB, BAND = 52, 4, 4, 0.25
FROZEN_END, DEC0, SPLIT = "2026-10-26", "2008-01-01", "2017-01-01"
AX = SM.AXES
BASE = np.array([0.5, 0.25, 0.25])
DCC_IDS = ["yf:ZC=F", "yf:ZS=F", "yf:ZM=F", "yf:ZW=F", "yf:CL=F", "yf:HG=F", "yf:GC=F", "yf:LE=F"]
MM = A.DATA / "mm"


# ---------------------------------------------------------------------------- 신호
def kalman_mean(S: pd.DataFrame, q: float = 1e-3) -> pd.DataFrame:
    """지역 수준 칼만 필터: μ_t = μ_{t-1} + η (분산 q·r), r_t = μ_t + ε (분산 r = 처음 3년 분산). 그 주까지 자료만(필터). 연율 %."""
    out = pd.DataFrame(index=S.index, columns=S.columns, dtype=float)
    for c in S:
        x = S[c].values; r = float(np.var(x[:156])); Q = q * r
        m, Pv = 0.0, r
        for t, v in enumerate(x):
            Pv += Q; K = Pv / (Pv + r); m += K * (v - m); Pv *= (1 - K)
            out.iloc[t, out.columns.get_loc(c)] = m * ANN
    return out


def news_index(idx: pd.DatetimeIndex) -> pd.Series:
    """문서 감성 평균 — 출처별로 그때까지 자료로 표준화(확장창) 뒤 평균. 공표 지연: 베이지북 월 +1개월, WASDE 발표 다음 주, STEO 빈티지 +1주."""
    parts = []
    b = pd.read_parquet(MM / "beige.parquet"); b = b.groupby("month").tone.mean(); b.index = pd.to_datetime(b.index) + pd.DateOffset(months=1)
    parts.append(b)
    w = pd.read_parquet(MM / "wasde_text.parquet"); w = w.groupby("release").tone.mean()
    w.index = pd.to_datetime(w.index).tz_localize(None).normalize() + pd.Timedelta(days=7); parts.append(w)
    s = pd.read_parquet(MM / "steo_text.parquet"); s = s.groupby("vintage").tone.mean(); s.index = pd.to_datetime(s.index) + pd.Timedelta(days=7)
    parts.append(s)
    z = []
    for p in parts:
        p = p.sort_index(); p = p[~p.index.duplicated()]
        zz = (p - p.expanding(12).mean()) / p.expanding(12).std()
        z.append(zz.reindex(idx.union(zz.index)).ffill().reindex(idx))
    return pd.concat(z, axis=1).mean(1)


def dcc_signals(S: pd.DataFrame, R: pd.DataFrame) -> tuple[dict, pd.Series]:
    """4주마다 그때까지 자료로 3축 DCC(공분산) · 원자재 8개 DCC(평균 상관) 재추정 — 마지막 시점 값만 사용."""
    cov, comove = {}, pd.Series(np.nan, index=S.index)
    Ri = R[DCC_IDS].dropna()
    for t in range(0, len(S), REB):
        d = S.index[t]
        if d < pd.Timestamp(DEC0) - pd.Timedelta(weeks=8):
            continue
        try:
            ds = MG.dcc_fit(S.loc[:d], False, "GJR", "t", ANN); cov[d] = ds["H"][-1] * ANN
        except Exception:
            cov[d] = S.loc[:d].iloc[-156:].cov().values * ANN
        try:
            dc = MG.dcc_fit(Ri.loc[:d].iloc[-780:], False, "GJR", "t", ANN); comove.loc[d] = float(dc["avg_corr"].iloc[-1])
        except Exception:
            pass
        if t % 52 == 0:
            print(f"  DCC {d:%Y-%m}", flush=True)
    return cov, comove.ffill()


def cond_view(S: pd.DataFrame, sig: pd.Series, d: pd.Timestamp) -> np.ndarray:
    """축별: 다음 4주 수익 ~ a + b·신호(그때까지 자료, 목표가 끝난 표본만) → 지금 신호로 예측, 연율 %."""
    y = S[::-1].rolling(H).sum()[::-1].shift(-1)
    ok = S.index[(S.index <= d - pd.Timedelta(weeks=H))]
    x = sig.reindex(ok); m = x.notna() & y.loc[ok].notna().all(1)
    if m.sum() < 104 or not np.isfinite(sig.get(d, np.nan)):
        return np.full(3, np.nan)
    X = np.c_[np.ones(m.sum()), x[m].values]
    out = []
    for c in AX:
        b = np.linalg.lstsq(X, y.loc[ok][c][m].values, rcond=None)[0]
        out.append((b[0] + b[1] * sig[d]) * ANN / H)
    return np.array(out)


# ---------------------------------------------------------------------------- 비중
def bl_weights(Sig: np.ndarray, view: np.ndarray | None, lam: float) -> np.ndarray:
    vol = np.sqrt(BASE @ Sig @ BASE)
    prior = (0.5 / vol) * Sig @ BASE                               # 기본 비중이 최적이 되는 기대수익 (기본 Sharpe 0.5 가정)
    mu = prior if view is None or not np.isfinite(view).all() else (1 - lam) * prior + lam * view
    bnd = [(max(0.0, b - BAND), min(1.0, b + BAND)) for b in BASE]
    r = minimize(lambda w: -(w @ mu) / np.sqrt(w @ Sig @ w), BASE, bounds=bnd,
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], method="SLSQP")
    return r.x if r.success else BASE


def risk_parity(Sig: np.ndarray) -> np.ndarray:
    f = lambda w: ((w * (Sig @ w)) / (w @ Sig @ w) - 1 / 3) @ ((w * (Sig @ w)) / (w @ Sig @ w) - 1 / 3)   # noqa: E731
    r = minimize(f, np.full(3, 1 / 3), bounds=[(0.01, 1)] * 3, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], method="SLSQP")
    return r.x


def build(S, cov, views: dict, lam: float, kind: str) -> pd.DataFrame:
    W = pd.DataFrame(np.nan, index=S.index, columns=AX)
    for d, Sig in cov.items():
        if kind == "D":
            W.loc[d] = risk_parity(Sig); continue
        vs = [views[k].loc[d].values for k in (["K", "H", "N", "C"] if kind == "ALL" else [kind])]
        vs = [v for v in vs if np.isfinite(v).all()]
        W.loc[d] = bl_weights(Sig, np.mean(vs, 0) if vs else None, lam)
    W.loc[:DEC0] = W.loc[:DEC0].fillna(pd.Series(BASE, AX))
    return W.ffill()


def views_from(S, sigs: dict, kal: pd.DataFrame, dates) -> dict:
    V = {"K": kal.reindex(S.index)}
    for k, sg in sigs.items():
        V[k] = pd.DataFrame({d: cond_view(S, sg, d) for d in dates}, index=AX).T.reindex(S.index)
    return V


def score(r, bench):
    return SM.score(r, bench)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    L = A.load_levels(roll_adjusted=True).loc[:FROZEN_END]
    S, R, P = SM.sleeves(L)
    print(f"3축 {S.index[0]:%Y-%m}~{S.index[-1]:%Y-%m} · 판단 {DEC0[:4]}~ · 학습 ~2016 · 시험 2017~", flush=True)
    stress = LS.stress_prob(L).reindex(S.index).ffill()
    news = news_index(S.index)
    cov, comove = dcc_signals(S, R)
    dates = list(cov)
    sigs = {"H": stress, "N": news, "C": comove}
    bench = SS.run_weights(S, SS.const(S, dict(zip(AX, BASE))))
    rows, weekly, wts = [], {"고정 50/25/25": bench}, {}
    rows.append({"규칙": "고정 50/25/25", "λ·q": "", **score(bench, bench)})
    eq = SS.run_weights(S, SS.const(S, {a: 1 / 3 for a in AX})); weekly["동일 비중"] = eq
    rows.append({"규칙": "동일 비중", "λ·q": "", **score(eq, bench)})
    Wd = build(S, cov, {}, 0, "D"); rd = SS.run_weights(S, Wd); weekly["D 위험 균등(DCC)"] = rd; wts["D"] = Wd
    rows.append({"규칙": "D 위험 균등 (3축 DCC 공분산)", "λ·q": "", **score(rd, bench)})
    best = {}
    for q in (1e-4, 1e-3, 1e-2):                                     # 칼만 속도 q · 관점 반영 λ 모두 학습에서 고름
        V = views_from(S, sigs, kalman_mean(S, q), dates)
        for kind in ("K", "H", "N", "C", "ALL"):
            if kind in ("H", "N", "C") and q != 1e-3:
                continue                                              # q는 칼만 관점에만 영향
            for lam in (0.25, 0.5, 1.0):
                W = build(S, cov, V, lam, kind); r = SS.run_weights(S, W)
                sc = score(r, bench); tag = f"q={q:g} λ={lam}" if kind in ("K", "ALL") else f"λ={lam}"
                rows.append({"규칙": {"K": "K 칼만 평균 추적", "H": "H HMM 위기 국면", "N": "N 뉴스 감성", "C": "C DCC 동조성", "ALL": "ALL 관점 4개 평균"}[kind], "λ·q": tag, **sc})
                if kind not in best or sc["Sharpe 학습 2008–16"] > best[kind][0]["Sharpe 학습 2008–16"]:
                    best[kind] = (sc, r, W, tag, V, lam, q)
    for kind, (sc, r, W, tag, V, lam, q) in best.items():
        weekly[f"{kind} {tag}"] = r; wts[kind] = W
        for row in rows:
            if row["규칙"].startswith(kind + " ") and row["λ·q"] == tag:
                row["학습 최고"] = "←"
    # 위약: 학습 최고 ALL의 신호를 13주 블록으로 섞어 100회 (칼만은 축 순서를 섞어 '어느 축이 좋다'는 정보를 지움)
    sc, r, W, tag, V, lam, q = best["ALL"]
    rng, pl = np.random.default_rng(0), []
    blocks = [S.index[i:i + 13] for i in range(0, len(S), 13)]
    for _ in range(100):
        o = rng.permutation(len(blocks)); sh_ = {}
        for k, sg in sigs.items():
            x = pd.concat([sg.reindex(blocks[j]) for j in o]).iloc[:len(S)]; x.index = S.index; sh_[k] = x
        Vp = views_from(S, sh_, kalman_mean(S, q)[list(rng.permutation(AX))].set_axis(AX, axis=1), dates)
        pl.append(score(SS.run_weights(S, build(S, cov, Vp, lam, "ALL")), bench)["Sharpe 시험 2017–"])
    real = sc["Sharpe 시험 2017–"]
    for row in rows:
        if row["규칙"].startswith("ALL") and row["λ·q"] == tag:
            row["위약 p (시험)"] = float((np.array(pl) >= real).mean())
    print(f"위약 100회(ALL {tag}): 시험 Sharpe 실제 {real:.2f} vs 위약 중앙 {np.median(pl):.2f} · 95% {np.quantile(pl, 0.95):.2f}", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(A.DATA / "strategy_adaptive_compare.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(weekly).to_parquet(A.DATA / "strategy_adaptive_weekly.parquet")
    pd.concat(wts, axis=1).to_parquet(A.DATA / "strategy_adaptive_weights.parquet")
    pd.set_option("display.width", 250)
    print(T.round(2).to_string(index=False))
    for k in ("K", "ALL"):                                            # 가치가 안 먹힌 2008–2015에 가치 비중을 줄였나
        Wk = wts[k]
        print(f"{k} 평균 비중 2008–15:", (Wk.loc["2008":"2015"].mean() * 100).round(0).to_dict(), " 2016–:", (Wk.loc["2016":].mean() * 100).round(0).to_dict())


if __name__ == "__main__":
    main()
