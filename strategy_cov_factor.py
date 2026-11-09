"""공분산 추정 · 3축 리스크 패리티 · 다신호 μ Newton · 요인 모형 백테스트 (M · N · O · P).

M. 공분산 추정 비교 — MinVar 축(품목 상한 25%)의 Σ만 바꿈: 표본 · Ledoit-Wolf(기준) · OAS · 대각 · 상수상관 · PCA 요인(1·2·3·5) · FF식 요인모형(시장+모멘텀+가치+저변동)
   × 창 52 · 104 · 156주. MinVar 단독 + 3축 혼합(50/25/25) 안에서 평가. 예측 변동성 편향(실현/예측) · 회전율 · 유효 품목 수도 기록.
N. 3축 단위 리스크 패리티 — 3축(MinVar · 모멘텀 · 가치) 수익률의 비중을 고정 50/25/25 · 동일 · 역변동성 · ERC(Newton, 표본/LW 공분산) · 최소분산으로 정함.
O. 여러 신호를 합친 μ의 품목 Newton — μ = IC(0.05) × σ × 합성 z (모멘텀 · 가치 · 칼만의 횡단면 z 평균), portfolio.newton_mv(롱온리 · 롱숏) × λ 5개. 위약 = 품목 순서 섞기.
P. FF식 요인 모형 × 포트폴리오 — 요인(시장 · 모멘텀 · 가치 · 저변동) 수익률로 (a) 요인 공분산(M에 포함) (b) 요인 프리미엄 × 적재량으로 μ를 만들어 Newton.

공통: 결정일 금요일 종가까지의 정보만 사용(창은 모두 과거), 다음 주 수익률로 손익, 편도 5bp, 목표 변동성 10%(직전 26주 실현, 상한 4배),
      학습 2008–2016 · 시험 2017–, 설정 선택은 학습에서만(학습 최고 표시), 평가는 SM.score(2008~) 재사용.
실행: .venv\\Scripts\\python.exe strategy_cov_factor.py [M N O P]  → data/strategy_cov_factor_compare.csv (+ 실험별 weekly parquet)
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
import portfolio as PF  # noqa: E402
import strategy_meta as SM  # noqa: E402

ANN, TGT, COST = 52, 10.0, 5.0
FROZEN_END, START, SPLIT = "2026-10-26", "2005-01-01", "2017-01-01"
CAP = 0.25
BASE_W = np.array([0.5, 0.25, 0.25])


# ------------------------------------------------------------------ 공통
def load():
    L = A.load_levels(roll_adjusted=True).loc[:FROZEN_END]
    S3, R, P = SM.sleeves(L)                                       # 3축 주간 순수익 %, 품목 수익 %, 가격
    return L, S3, R, P


def vt_book(W: pd.DataFrame, R: pd.DataFrame) -> pd.Series:
    return LS._book(LS._vol_target(W, R, TGT), R, COST).loc[START:].fillna(0.0)


def mix3(mv: pd.Series, S3: pd.DataFrame, w=BASE_W) -> pd.Series:
    X = pd.DataFrame({"MinVar": mv.reindex(S3.index).fillna(0.0), "모멘텀": S3["모멘텀"], "가치": S3["가치"]})
    m = X.values @ w
    m = pd.Series(m, X.index)
    pv = m.rolling(26, min_periods=13).std().shift(1) * np.sqrt(ANN)
    return m * (TGT / pv).clip(upper=4.0).fillna(1.0)


def sh(x: pd.Series) -> float:
    return float(x.mean() / x.std() * np.sqrt(ANN)) if x.std() > 0 else np.nan


def summarize(r: pd.Series, ref: pd.Series) -> dict:
    o = SM.score(r, ref)
    return {"Sharpe 2008–": o["Sharpe 2008–"], "Sharpe 학습 2008–16": o["Sharpe 학습 2008–16"], "Sharpe 시험 2017–": o["Sharpe 시험 2017–"],
            "연수익 (%)": o["연수익 (%)"], "최대 낙폭 (%)": o["최대 낙폭 (%)"], "시험 기준 대비 (%p)": o["시험 고정 대비 (%p)"], "시험 차이 t": o["시험 차이 t"]}


def turnover(W: pd.DataFrame) -> float:
    return float(W.diff().abs().sum(1).loc["2008":].sum() / (len(W.loc["2008":]) / ANN))


# ------------------------------------------------------------------ 공분산 추정기 (X: T×n 주간 소수 수익률, F: T×m 요인 수익률 소수)
def est_sample(X, F=None):
    return np.cov(X, rowvar=False) + 1e-10 * np.eye(X.shape[1])


def est_lw(X, F=None):
    from sklearn.covariance import LedoitWolf
    return LedoitWolf().fit(X).covariance_


def est_oas(X, F=None):
    from sklearn.covariance import OAS
    return OAS().fit(X).covariance_


def est_diag(X, F=None):
    return np.diag(X.var(0, ddof=1) + 1e-12)


def est_ccor(X, F=None):
    S = np.cov(X, rowvar=False); sd = np.sqrt(np.diag(S)); C = S / np.outer(sd, sd)
    rho = (C.sum() - len(C)) / (len(C) * (len(C) - 1))
    return np.outer(sd, sd) * (rho * np.ones_like(C) + (1 - rho) * np.eye(len(C)))


def est_pca(k):
    def f(X, F=None):
        S0 = np.cov(X, rowvar=False); w, V = np.linalg.eigh(S0); ix = np.argsort(w)[::-1][:k]
        Sf = (V[:, ix] * w[ix]) @ V[:, ix].T
        return Sf + np.diag(np.clip(np.diag(S0) - np.diag(Sf), 1e-8, None))
    f.__name__ = f"pca{k}"
    return f


def est_ff(X, F):
    """FF식 요인 모형: X = a + B F + e → Σ = B Cov(F) B′ + diag(Var(e))."""
    Z = np.column_stack([np.ones(len(F)), F])
    coef, *_ = np.linalg.lstsq(Z, X, rcond=None)
    B = coef[1:].T; E = X - Z @ coef
    return B @ np.cov(F, rowvar=False) @ B.T + np.diag(np.clip(E.var(0, ddof=Z.shape[1]), 1e-8, None))


EST = {"표본 공분산": est_sample, "Ledoit-Wolf (기준)": est_lw, "OAS": est_oas, "대각 (상관 0)": est_diag, "상수 상관": est_ccor,
       "PCA 1요인": est_pca(1), "PCA 2요인": est_pca(2), "PCA 3요인": est_pca(3), "PCA 5요인": est_pca(5), "FF식 4요인 (시장·모멘텀·가치·저변동)": est_ff}


def make_factors(P: pd.DataFrame, R: pd.DataFrame) -> pd.DataFrame:
    """품목 횡단면 롱숏 요인(%, 주간, 비용 0): 모멘텀(52주) · 가치(5년 평균 대비) · 저변동(52주 변동성 역) + 시장(동일가중). 비중은 t까지 정보, 수익은 다음 주."""
    lp = np.log(P)
    mom = lp.diff(52); val = -(lp - lp.rolling(260, min_periods=156).mean()); lvol = -R.rolling(52, min_periods=26).std()
    f = {nm: LS._book(LX.xs_weights(s), R, 0.0) for nm, s in [("MOM", mom), ("VAL", val), ("LVOL", lvol)]}
    f["MKT"] = R.mean(1).iloc[1:]
    return pd.DataFrame(f)[["MKT", "MOM", "VAL", "LVOL"]].fillna(0.0)


def minvar_w(P, R, est, win=104, reb=4, cap=CAP, Fac=None):
    """품목 상한 cap 최소분산 롱온리. 창 = 결정일 t까지 win주. 반환: 비중, 예측 변동성(주간, 단위비중)."""
    from scipy.optimize import minimize
    W = pd.DataFrame(np.nan, index=P.index, columns=P.columns); pv = pd.Series(np.nan, index=P.index)
    for t in range(win, len(P), reb):
        wd = R.iloc[t - win + 1:t + 1]
        cols = [c for c in P.columns if wd[c].notna().sum() >= win * 0.8]
        if len(cols) < 4:
            continue
        X = wd[cols].fillna(0.0).values / 100
        Fw = None if Fac is None else Fac.reindex(wd.index).fillna(0.0).values / 100
        S = est(X, Fw); n = len(cols); capn = max(cap, 1.0 / n + 1e-6)
        res = minimize(lambda w: 1e4 * (w @ S @ w), np.ones(n) / n, jac=lambda w: 1e4 * 2 * S @ w, bounds=[(0, capn)] * n,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
        x = np.clip(res.x, 0, capn)
        w = x / x.sum() if (abs(res.x.sum() - 1) < 1e-3 and np.isfinite(res.x).all()) else np.ones(n) / n
        W.loc[P.index[t], cols] = w; W.loc[P.index[t], [c for c in P.columns if c not in cols]] = 0.0
        pv.iloc[t] = float(np.sqrt(w @ S[np.ix_(range(n), range(n))] @ w))
    return W.ffill().fillna(0.0), pv.ffill()


def bias_stat(W, pv, R, lo="2017-01-01"):
    """예측 변동성 편향: 단위 비중 포트폴리오의 다음 주 수익 / 예측 σ 의 표준편차 (1 = 정확, >1 = 위험 과소 예측)."""
    ru = (W.shift(1) * R / 100).sum(1)
    z = (ru / pv.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    return float(z.loc[lo:].std()), float(z.loc["2008":].std())


# ------------------------------------------------------------------ M. 공분산 추정 비교
def run_M(L, S3, R, P, Fac):
    rows, rets = [], {}
    ref_mv = ref_mx = None
    plan = [(nm, 104) for nm in EST] + [(nm, w) for w in (52, 156) for nm in ("표본 공분산", "Ledoit-Wolf (기준)", "OAS", "PCA 3요인", "FF식 4요인 (시장·모멘텀·가치·저변동)")]
    for nm, win in plan:
        t0 = time.time()
        W, pv = minvar_w(P, R, EST[nm], win, Fac=Fac)
        mv = vt_book(W, R); mx = mix3(mv, S3)
        if nm.startswith("Ledoit") and win == 104:
            ref_mv, ref_mx = mv, mx
            print("  [검증] 기준 MinVar vs 기존 sleeves MinVar 최대 차이:", float((mv - S3["MinVar"]).abs().max()))
        rets[(nm, win)] = (mv, mx, W, pv)
        print(f"  M {nm} {win}주  {time.time() - t0:.0f}s", flush=True)
    for (nm, win), (mv, mx, W, pv) in rets.items():
        ref_mv_w = rets[("Ledoit-Wolf (기준)", win)][0]; ref_mx_w = rets[("Ledoit-Wolf (기준)", win)][1]
        s1 = summarize(mv, ref_mv_w); s2 = summarize(mx, ref_mx_w)
        b_te, b_all = bias_stat(W, pv, R)
        rows.append({"실험": "M 공분산 추정", "구성": nm, "창(주)": win, "설정": "MinVar 단독",
                     **s1, "연 회전율 (배)": turnover(W), "유효 품목 수": float((1 / (W[W.sum(1) > 0] ** 2).sum(1)).mean()),
                     "위험 예측 편향 시험": b_te, "위험 예측 편향 전체": b_all,
                     "혼합 학습": s2["Sharpe 학습 2008–16"], "혼합 시험": s2["Sharpe 시험 2017–"], "혼합 2008–": s2["Sharpe 2008–"], "혼합 시험 차이 t": s2["시험 차이 t"]})
    return rows, {f"M|{k[0]}|{k[1]}": v[0] for k, v in rets.items()}


# ------------------------------------------------------------------ N. 3축 단위 리스크 패리티
def axis_weights(S3, method, win=104, reb=4):
    from scipy.optimize import minimize
    from sklearn.covariance import LedoitWolf
    n = S3.shape[1]; W = pd.DataFrame(np.nan, index=S3.index, columns=S3.columns)
    for i in range(0, len(S3), reb):
        if method == "고정 50/25/25":
            w = BASE_W
        elif method == "동일 비중":
            w = np.full(n, 1 / n)
        else:
            if i < win:
                continue
            X = S3.iloc[i - win:i].values / 100                     # 결정 시점 전 주까지 (i번째 주 수익은 미사용)
            if method == "역변동성 (52주)":
                sd = S3.iloc[i - 52:i].std().values; w = (1 / sd) / (1 / sd).sum()
            elif method.startswith("ERC 표본"):
                w, _ = PF.newton_risk_parity(np.cov(X, rowvar=False))
            elif method.startswith("ERC LW"):
                w, _ = PF.newton_risk_parity(LedoitWolf().fit(X).covariance_)
            elif method.startswith("최소분산 LW"):
                C = LedoitWolf().fit(X).covariance_
                r_ = minimize(lambda w: 1e4 * w @ C @ w, np.full(n, 1 / n), bounds=[(0, 1)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], method="SLSQP")
                w = r_.x / r_.x.sum()
            else:
                raise ValueError(method)
        W.iloc[i] = w
    return W.ffill()


def run_N(S3):
    rows, ref = [], None
    plan = [("고정 50/25/25", 104), ("동일 비중", 104), ("역변동성 (52주)", 104), ("ERC 표본 104주", 104), ("ERC 표본 156주", 156), ("ERC LW 104주", 104), ("ERC LW 156주", 156),
            ("최소분산 LW 104주", 104)]
    out = {}
    for nm, win in plan:
        W = axis_weights(S3, nm, win)
        Wf = W.fillna(pd.Series(BASE_W, S3.columns))                  # 초기 창 동안은 고정 비중
        raw = (Wf * S3).sum(1) - Wf.diff().abs().sum(1).fillna(0) * 1.15 * COST / 100
        pv = raw.rolling(26, min_periods=13).std().shift(1) * np.sqrt(ANN)
        r = raw * (TGT / pv).clip(upper=4.0).fillna(1.0)
        out[nm] = (r, Wf)
        if ref is None:
            ref = r
    for nm, (r, Wf) in out.items():
        rows.append({"실험": "N 3축 리스크 패리티", "구성": nm, "설정": "3축 비중 결정", **summarize(r, ref),
                     "MinVar 평균 비중 (%)": float(Wf["MinVar"].loc["2008":].mean() * 100), "모멘텀 평균 비중 (%)": float(Wf["모멘텀"].loc["2008":].mean() * 100),
                     "가치 평균 비중 (%)": float(Wf["가치"].loc["2008":].mean() * 100), "연 비중 변경 (|Δw| 합)": float(Wf.diff().abs().sum(1).loc["2008":].sum() / (len(Wf.loc["2008":]) / ANN))})
    return rows, {f"N|{k}": v[0] for k, v in out.items()}


# ------------------------------------------------------------------ O. 다신호 μ 품목 Newton
def xz(S: pd.DataFrame) -> pd.DataFrame:
    z = S.sub(S.mean(1), axis=0).div(S.std(1).replace(0, np.nan), axis=0)
    return z.clip(-2, 2)


def signals(P, R):
    lp = np.log(P)
    z = {"모멘텀": xz(lp.diff(52)), "가치": xz(-(lp - lp.rolling(260, min_periods=156).mean()))}
    z["칼만"] = xz(pd.DataFrame({c: LX.kalman_trend(lp[c].ffill(), q_slope=0.001)["t"] for c in lp}).where(P.notna()))
    return z


SIGSETS = {"모멘텀": ["모멘텀"], "가치": ["가치"], "칼만": ["칼만"], "모멘텀+가치": ["모멘텀", "가치"], "모멘텀+가치+칼만": ["모멘텀", "가치", "칼만"]}
LAMS = [2, 5, 10, 20, 50]


def newton_weights(Z, P, R, lam, mode, win=104, reb=4, ic=0.05, wmax=CAP, mu_fn=None, Fac=None):
    W = pd.DataFrame(np.nan, index=P.index, columns=P.columns); w_prev = pd.Series(0.0, index=P.columns)
    for t in range(win, len(P), reb):
        wd = R.iloc[t - win + 1:t + 1]
        cols = [c for c in P.columns if wd[c].notna().sum() >= win * 0.8]
        if len(cols) < 4:
            continue
        X = wd[cols].fillna(0.0).values / 100
        S = est_lw(X); sd = np.sqrt(np.diag(S)); n = len(cols)
        if mu_fn is None:
            zt = Z.iloc[t][cols]
            if zt.isna().all():
                continue
            mu = ic * sd * zt.fillna(0.0).values
        else:
            mu = mu_fn(t, cols, wd, S)
        w0 = w_prev.reindex(cols).fillna(0.0).values
        if mode == "long_only":
            w0 = w0 / w0.sum() if w0.sum() > 0 else np.full(n, 1 / n)
        w, _ = PF.newton_mv(mu, S, w0, np.full(n, COST / 1e4), lam, mode, 1.0, w_max=wmax)
        wf = pd.Series(0.0, index=P.columns); wf[cols] = w
        W.loc[P.index[t]] = wf; w_prev = wf
    return W.ffill().fillna(0.0)


def run_O(S3, R, P, base_mix, base_mv):
    sg = signals(P, R); rows, store = [], {}
    for mode in ("long_only", "long_short"):
        for sn, parts in SIGSETS.items():
            Z = sum(sg[p] for p in parts) / len(parts)
            for lam in LAMS:
                t0 = time.time()
                W = newton_weights(Z, P, R, lam, mode)
                r = vt_book(W, R); mx = mix3(r, S3) if mode == "long_only" else None
                s1 = summarize(r, base_mix)
                row = {"실험": "O 다신호 μ Newton", "구성": f"{sn} μ", "설정": f"{'롱온리 25% 상한' if mode == 'long_only' else '롱숏 총노출 1'} · λ={lam}", "모드": mode, "λ": lam, "신호": sn,
                       **s1, "연 회전율 (배)": turnover(W), "MinVar 대비 시험 차이 (%p)": float((r - base_mv).loc[SPLIT:].mean() * ANN)}
                if mx is not None:
                    s2 = summarize(mx, base_mix); row.update({"혼합 학습": s2["Sharpe 학습 2008–16"], "혼합 시험": s2["Sharpe 시험 2017–"], "혼합 시험 차이 t": s2["시험 차이 t"]})
                rows.append(row); store[f"O|{mode}|{sn}|{lam}"] = r
                print(f"  O {mode} {sn} λ={lam}  {time.time() - t0:.0f}s  학습 {s1['Sharpe 학습 2008–16']:+.2f} 시험 {s1['Sharpe 시험 2017–']:+.2f}", flush=True)
    D = pd.DataFrame(rows)
    D["학습 최고"] = ""
    for mode in ("long_only", "long_short"):
        sub = D[(D["모드"] == mode)]
        D.loc[sub["Sharpe 학습 2008–16"].idxmax(), "학습 최고"] = "◆"
    return D, store, sg


def placebo_O(best_row, sg, S3, R, P, base_mix, n=40, seed=0):
    rng = np.random.default_rng(seed)
    Z0 = sum(sg[p] for p in SIGSETS[best_row["신호"]]) / len(SIGSETS[best_row["신호"]])
    te, tr = [], []
    for k in range(n):
        perm = rng.permutation(Z0.shape[1])
        Zp = pd.DataFrame(Z0.values[:, perm], index=Z0.index, columns=Z0.columns)     # 품목-신호 짝을 섞음 (시간 구조 유지)
        W = newton_weights(Zp, P, R, int(best_row["λ"]), best_row["모드"])
        r = vt_book(W, R)
        te.append(sh(r.loc[SPLIT:])); tr.append(sh(r.loc["2008":"2016"]))
    te, tr = np.array(te), np.array(tr)
    return {"위약 시험 Sharpe 평균": float(np.nanmean(te)), "위약 시험 95%": float(np.nanpercentile(te, 95)),
            "위약 p (시험)": float((te >= best_row["Sharpe 시험 2017–"]).mean()), "위약 p (학습)": float((tr >= best_row["Sharpe 학습 2008–16"]).mean())}


# ------------------------------------------------------------------ P. FF식 요인 모형 × 포트폴리오 (요인 프리미엄 μ)
def run_P(S3, R, P, Fac, base_mix, base_mv):
    rows, store = [], {}
    Fd = Fac / 100

    def mu_factory(shrink, ffcov):
        def f(t, cols, wd, S):
            idx = wd.index
            Fw = Fd.reindex(idx).fillna(0.0).values; X = wd[cols].fillna(0.0).values / 100
            Z = np.column_stack([np.ones(len(Fw)), Fw]); coef, *_ = np.linalg.lstsq(Z, X, rcond=None); B = coef[1:].T      # n×m 적재량
            lam_f = Fd.loc[:P.index[t]].iloc[-260:].mean().values * shrink                                                    # 최근 260주 요인 평균 × 축소
            return B @ lam_f
        return f

    for shrink in (0.5, 1.0):
        for lam in (5, 20, 50):
            t0 = time.time()
            W = newton_weights(None, P, R, lam, "long_only", mu_fn=mu_factory(shrink, False))
            r = vt_book(W, R); mx = mix3(r, S3); s1 = summarize(r, base_mix); s2 = summarize(mx, base_mix)
            rows.append({"실험": "P FF식 요인 프리미엄 μ", "구성": f"요인 프리미엄 × 적재량 (축소 {shrink})", "설정": f"롱온리 25% 상한 · λ={lam}", "모드": "long_only", "λ": lam, "축소": shrink,
                         **s1, "연 회전율 (배)": turnover(W), "MinVar 대비 시험 차이 (%p)": float((r - base_mv).loc[SPLIT:].mean() * ANN),
                         "혼합 학습": s2["Sharpe 학습 2008–16"], "혼합 시험": s2["Sharpe 시험 2017–"], "혼합 시험 차이 t": s2["시험 차이 t"]})
            store[f"P|{shrink}|{lam}"] = r
            print(f"  P 축소{shrink} λ={lam} {time.time() - t0:.0f}s  학습 {s1['Sharpe 학습 2008–16']:+.2f} 시험 {s1['Sharpe 시험 2017–']:+.2f}", flush=True)
    D = pd.DataFrame(rows); D["학습 최고"] = ""
    D.loc[D["Sharpe 학습 2008–16"].idxmax(), "학습 최고"] = "◆"
    return D, store


def placebo_P(best_row, S3, R, P, Fac, n=30, seed=1):
    rng = np.random.default_rng(seed); Fd = Fac / 100; te, tr = [], []
    for k in range(n):
        partner = dict(zip(P.columns, P.columns[rng.permutation(P.shape[1])]))      # 품목 ↔ 다른 품목 짝 (시간에 걸쳐 고정)

        def f(t, cols, wd, S, partner=partner):
            Fw = Fd.reindex(wd.index).fillna(0.0).values; X = wd[cols].fillna(0.0).values / 100
            Z = np.column_stack([np.ones(len(Fw)), Fw]); coef, *_ = np.linalg.lstsq(Z, X, rcond=None); B = coef[1:].T
            pos = {c: j for j, c in enumerate(cols)}
            B = B[[pos.get(partner[c], j) for j, c in enumerate(cols)]]                # 품목 적재량을 짝 품목 것으로 바꿔치기
            return B @ (Fd.loc[:P.index[t]].iloc[-260:].mean().values * best_row["축소"])
        W = newton_weights(None, P, R, int(best_row["λ"]), "long_only", mu_fn=f)
        r = vt_book(W, R); te.append(sh(r.loc[SPLIT:])); tr.append(sh(r.loc["2008":"2016"]))
    te, tr = np.array(te), np.array(tr)
    return {"위약 시험 Sharpe 평균": float(np.nanmean(te)), "위약 시험 95%": float(np.nanpercentile(te, 95)),
            "위약 p (시험)": float((te >= best_row["Sharpe 시험 2017–"]).mean()), "위약 p (학습)": float((tr >= best_row["Sharpe 학습 2008–16"]).mean())}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    which = set(sys.argv[1:]) or {"M", "N", "O", "P"}
    t00 = time.time()
    L, S3, R, P = load()
    Fac = make_factors(P, R).reindex(P.index).fillna(0.0)
    base_mix = mix3(S3["MinVar"], S3); base_mv = S3["MinVar"]
    sc = SM.score(base_mix, base_mix)
    print(f"[기준 3축 혼합 50/25/25] 학습 {sc['Sharpe 학습 2008–16']:.2f} · 시험 {sc['Sharpe 시험 2017–']:.2f} · 2008– {sc['Sharpe 2008–']:.2f}  (로그 기준: 0.61 / 1.07 / 0.86)")
    allrows, store = [], {}
    pd.set_option("display.width", 280); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
    if "M" in which:
        print("M. 공분산 추정 비교", flush=True)
        rows, st = run_M(L, S3, R, P, Fac); allrows += rows; store.update(st)
        print(pd.DataFrame(rows)[["구성", "창(주)", "Sharpe 학습 2008–16", "Sharpe 시험 2017–", "시험 차이 t", "연 회전율 (배)", "위험 예측 편향 시험", "혼합 학습", "혼합 시험", "혼합 시험 차이 t"]].round(2).to_string(index=False))
    if "N" in which:
        print("N. 3축 리스크 패리티", flush=True)
        rows, st = run_N(S3); allrows += rows; store.update(st)
        print(pd.DataFrame(rows).round(2).to_string(index=False))
    if "O" in which:
        print("O. 다신호 μ Newton", flush=True)
        D, st, sg = run_O(S3, R, P, base_mix, base_mv); store.update(st)
        for mode in ("long_only", "long_short"):
            b = D[(D["모드"] == mode) & (D["학습 최고"] == "◆")].iloc[0]
            pl = placebo_O(b, sg, S3, R, P, base_mix)
            for k, v in pl.items():
                D.loc[b.name, k] = v
            print(f"  위약({mode}, {b['신호']}, λ={b['λ']}):", {k: round(v, 3) for k, v in pl.items()}, flush=True)
        allrows += D.to_dict("records")
        print(D[["구성", "설정", "Sharpe 학습 2008–16", "Sharpe 시험 2017–", "시험 차이 t", "연 회전율 (배)", "학습 최고", "위약 p (시험)"]].round(2).to_string(index=False))
    if "P" in which:
        print("P. FF식 요인 프리미엄 μ", flush=True)
        D, st = run_P(S3, R, P, Fac, base_mix, base_mv); store.update(st)
        b = D[D["학습 최고"] == "◆"].iloc[0]
        pl = placebo_P(b, S3, R, P, Fac)
        for k, v in pl.items():
            D.loc[b.name, k] = v
        print("  위약:", {k: round(v, 3) for k, v in pl.items()}, flush=True)
        allrows += D.to_dict("records")
        print(D[["구성", "설정", "Sharpe 학습 2008–16", "Sharpe 시험 2017–", "시험 차이 t", "혼합 시험", "학습 최고", "위약 p (시험)"]].round(2).to_string(index=False))
    T = pd.DataFrame(allrows)
    suffix = "" if which == {"M", "N", "O", "P"} else "_" + "".join(sorted(which))
    T.to_csv(A.DATA / f"strategy_cov_factor_compare{suffix}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(store).to_parquet(A.DATA / f"strategy_cov_factor_weekly{suffix}.parquet")
    print(f"저장 완료 · 총 {time.time() - t00:.0f}s")


if __name__ == "__main__":
    main()
