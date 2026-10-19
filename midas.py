"""B-2. MIDAS (혼합 주기) — 일간 데이터로 월간 목표를 예측 · 일간 변동성 예측 · GARCH-MIDAS

  Beta 시차 가중치 (Ghysels, Santa-Clara & Valkanov 2004/2006):
      w_k ∝ x_k^(θ1−1) (1−x_k)^(θ2−1),  x_k = k/(K+1),  k = 1(가장 최근) … K,  Σw = 1
      θ1=1, θ2>1 이면 최근일수록 큰 단조 감소 — 몇 개의 모수로 수백 개 일간 시차를 요약
  MIDAS 회귀  : y_{m+1} = β0 + Σ_j β_j Σ_k w_k(θ_j) x_{j, t−k+1}    (θ는 비선형최소제곱, β는 θ가 주어지면 OLS)
  일간 변동성 : 매 거래일 다음 h영업일 일평균 분산을 EWMA(RiskMetrics) · HAR-RV(Corsi 2009) · MIDAS-RV로 예측
                (daily_vol_forecasts, 월말마다 walk-forward 재추정) → 구매 타이밍 위험·포트폴리오 변동성 모형·◆ 리스크 경주에 사용.
                모형 간 예측 경주(QLIKE·DM·MCS)는 risk.vol_models / ◆ 리스크 탭으로 일원화.
  GARCH-MIDAS : r_t = μ + √(τ_m g_t) ε_t,   g_t = (1−α−β) + α (r_{t−1}−μ)²/τ_m + β g_{t−1}
                log τ_m = m + θ Σ_{k=1}^{K} φ_k(w) X_{m−k}   (X = 월 실현분산 또는 거시 변수, φ = Beta(1,w))
                → 장기(τ) vs 단기(g) 분해, 분산비 VR = Var(log τ)/Var(log τg)
모든 예측은 결정 시점까지의 데이터만 사용 (walk-forward), 결정 시점 = 월말 종가.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize

import analytics as A
import risk as RK


def beta_weights(K: int, t1: float, t2: float) -> np.ndarray:
    x = np.arange(1, K + 1) / (K + 1)
    lw = (t1 - 1) * np.log(x) + (t2 - 1) * np.log1p(-x)
    w = np.exp(lw - lw.max())
    return w / w.sum()


def hf_lags(x: pd.Series, dates: pd.DatetimeIndex, K: int) -> np.ndarray:
    """각 결정일 d에 대해 d 이하의 최근 K개 고빈도 관측치 [x_d, x_{d-1}, ...] (결측이면 NaN 행)."""
    v = x.dropna()
    pos = v.index.searchsorted(dates, side="right") - 1
    arr = v.values
    M = np.full((len(dates), K), np.nan)
    for i, p in enumerate(pos):
        if p >= K - 1:
            M[i] = arr[p - K + 1:p + 1][::-1]
    return M


# ============================================================================
# MIDAS 회귀 (θ 비선형 + β OLS 프로파일링)
# ============================================================================
def midas_fit(Y: np.ndarray, Ls: list[np.ndarray], restrict_t1: bool = True, starts=None):
    """Y: (n,), Ls: 고빈도 시차행렬 목록 [(n,K_j)]. return dict(beta, thetas, weights, ssr)."""
    J = len(Ls)

    def design(th):
        cols = [np.ones(len(Y))]
        for j, L in enumerate(Ls):
            t1, t2 = (1.0, th[j]) if restrict_t1 else (th[2 * j], th[2 * j + 1])
            cols.append(L @ beta_weights(L.shape[1], t1, t2))
        return np.column_stack(cols)

    def ssr(th):
        X = design(th)
        b = np.linalg.lstsq(X, Y, rcond=None)[0]
        return float(np.sum((Y - X @ b) ** 2))

    dim = J if restrict_t1 else 2 * J
    bounds = [(1.0001, 300.0)] * dim if restrict_t1 else [(0.5, 50.0), (1.0001, 300.0)] * J
    starts = starts or ([[s] * dim for s in (1.5, 5.0, 20.0, 80.0)] if restrict_t1 else [[1.0, s] * J for s in (1.5, 5.0, 20.0)])
    best = None
    for s0 in starts:
        r = minimize(ssr, np.array(s0, float), method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    X = design(best.x)
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    ws = [beta_weights(L.shape[1], *((1.0, best.x[j]) if restrict_t1 else (best.x[2 * j], best.x[2 * j + 1]))) for j, L in enumerate(Ls)]
    return dict(beta=b, theta=best.x, weights=ws, ssr=float(best.fun), predict=lambda Lnew: _midas_pred(b, ws, Lnew))


def _midas_pred(b, ws, Lnew):
    out = np.full(len(Lnew[0]), b[0])
    for j, L in enumerate(Lnew):
        out = out + b[j + 1] * (L @ ws[j])
    return out


# ============================================================================
# 일간 변동성 예측 (구매 타이밍 위험·포트폴리오 변동성 모형·◆ 리스크 경주의 HAR·MIDAS)
# ============================================================================
_DVF_CACHE: dict = {}


def daily_vol_forecasts(r: pd.Series, h: int = 5, K: int = 252, min_train: int = 36) -> pd.DataFrame:
    """일간 로그수익률(%) → 매 거래일 t에 '다음 h영업일의 일평균 분산'(%²/일) 예측. 열: EWMA, HAR-RV, MIDAS-RV.
    walk-forward: 월말마다 그 시점까지 '실현이 끝난' 목표(월말 표본)로 HAR·MIDAS 재추정, 다음 월말 전까지는
    고정 모수 + 매일의 최신 시차로 예측 → 시점 t의 예측은 t까지의 정보만 사용."""
    r = r.dropna()
    key = (r.name, h, K, min_train, len(r), r.index[-1], float(r.iloc[-1]))
    if key in _DVF_CACHE:
        return _DVF_CACHE[key]
    r2 = r ** 2; n = len(r); days = r.index
    tgt = (r2.rolling(h).sum().shift(-h) / h).values                   # t+1..t+h 일평균 실현분산
    rv_d, rv_w, rv_m = r2.values, r2.rolling(5).mean().values, r2.rolling(21).mean().values
    ew = RK.ewma_var(r).values                                         # RiskMetrics λ=0.94 (공용)
    L = hf_lags(r2, days, K)
    me = pd.Series(np.arange(n), index=days).groupby(days.to_period("M")).max().values
    ok = ~np.isnan(L).any(1) & ~np.isnan(rv_m)
    out = np.full((n, 3), np.nan); out[:, 0] = ew
    th_prev = None
    for k, p in enumerate(me):
        nxt = me[k + 1] if k + 1 < len(me) else n
        tr = me[(me + h <= p) & ok[me] & ~np.isnan(tgt[me])]
        if len(tr) < min_train:
            continue
        y = tgt[tr]; floor = 0.5 * float(np.min(y[y > 0])) if (y > 0).any() else 1e-8
        Xh = np.column_stack([np.ones(len(tr)), rv_d[tr], rv_w[tr], rv_m[tr]])
        bh = np.linalg.lstsq(Xh, y, rcond=None)[0]
        mid = midas_fit(y, [L[tr]], starts=[[th_prev]] + [[5.0]] if th_prev is not None else None)
        th_prev = float(mid["theta"][0])
        seg = np.arange(p, nxt)
        seg = seg[ok[seg]]
        if len(seg):
            fl = np.maximum(floor, 0.25 * ew[seg])                     # 선형 예측의 0 근처 과소예측 방지 (당일 수준 연동)
            out[seg, 1] = np.maximum(np.column_stack([np.ones(len(seg)), rv_d[seg], rv_w[seg], rv_m[seg]]) @ bh, fl)
            out[seg, 2] = np.maximum(mid["predict"]([L[seg]]), fl)
    res = pd.DataFrame(out, index=days, columns=["EWMA", "HAR-RV", "MIDAS-RV"])
    _DVF_CACHE[key] = res
    return res


# ============================================================================
# GARCH-MIDAS
# ============================================================================
@njit(cache=True)
def _gm_nll(r, mid, Xlag_w, mu, alpha, beta, m0, theta, first):
    n = len(r)
    g = 1.0
    nll = 0.0
    for t in range(first, n):
        tau = np.exp(m0 + theta * Xlag_w[mid[t]])
        if t > first:
            e = r[t - 1] - mu
            tau_prev = np.exp(m0 + theta * Xlag_w[mid[t - 1]])
            g = (1 - alpha - beta) + alpha * e * e / tau_prev + beta * g
        v = tau * g
        e2 = (r[t] - mu) ** 2
        nll += 0.5 * (np.log(v) + e2 / v)
    return nll


def garch_midas(r: pd.Series, X: pd.Series | None = None, K: int = 24, log_x: bool = True):
    """r: 일간 수익률(%), X: 월간 장기변동 요인 (None = 해당 품목의 월 실현분산). return dict(params, se, tau, g, VR, ...)."""
    r = r.dropna()
    months = r.index.to_period("M")
    um = months.unique()
    if X is None:
        Xm = (r ** 2).groupby(months).sum()
        Xm.index = um
        src = "월 실현분산 (자기)"
    else:
        Xm = X.groupby(X.index.to_period("M")).mean().reindex(um).ffill()
        src = "거시 변수"
    xv = np.log(Xm.values + 1e-8) if (log_x and (Xm.values > 0).all()) else Xm.values
    xv = (xv - np.nanmean(xv)) / np.nanstd(xv)
    mid = np.searchsorted(um.asi8, months.asi8)
    first_month = K
    first = int(np.argmax(mid >= first_month))
    rv = r.values.astype(np.float64)

    def lagw(w):
        wts = beta_weights(K, 1.0, w)
        out = np.zeros(len(um))
        for m in range(K, len(um)):
            out[m] = xv[m - K:m][::-1] @ wts            # 직전 K개월 (m-1, m-2, …) — 당월 정보 미포함
        return out

    def nll(p):
        mu, a, b, m0, th, w = p
        if a <= 0 or b <= 0 or a + b >= 0.999 or w <= 1.0001:
            return 1e12
        return _gm_nll(rv, mid, lagw(w), mu, a, b, m0, th, first)

    v0 = float(np.var(rv))
    best = None
    for th0 in (0.1, 0.3, -0.1):
        for w0 in (2.0, 6.0):
            x0 = np.array([rv.mean(), 0.05, 0.90, np.log(v0), th0, w0])
            res = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 6000, "xatol": 1e-7, "fatol": 1e-7})
            if best is None or res.fun < best.fun:
                best = res
    p = best.x
    from mvgarch import _num_hessian
    try:
        H = _num_hessian(nll, p, 1e-4); se = np.sqrt(np.abs(np.diag(np.linalg.pinv(H))))
    except Exception:
        se = np.full(len(p), np.nan)
    names = ["μ", "α", "β", "m (장기 절편)", "θ (장기 민감도)", "w (Beta 감쇠)"]
    lw = lagw(p[5])
    tau_m = np.exp(p[3] + p[4] * lw)
    tau = pd.Series(tau_m[mid], index=r.index)
    g = np.ones(len(rv)); mu, a, b = p[0], p[1], p[2]
    for t in range(first + 1, len(rv)):
        g[t] = (1 - a - b) + a * (rv[t - 1] - mu) ** 2 / tau.iloc[t - 1] + b * g[t - 1]
    g = pd.Series(g, index=r.index)
    sl = slice(first, None)
    vr = float(np.var(np.log(tau.iloc[sl])) / np.var(np.log(tau.iloc[sl] * g.iloc[sl])))
    return dict(params=pd.DataFrame({"계수": p, "표준오차": se, "t값": p / se}, index=names),
                tau_vol=np.sqrt(tau.iloc[sl] * 252), total_vol=np.sqrt(tau.iloc[sl] * g.iloc[sl] * 252),
                VR=vr, weights=beta_weights(K, 1.0, p[5]), loglik=-best.fun, source=src, K=K)


# ============================================================================
# MIDAS 수익 예측 (월간 목표, 일간 설명변수)
# ============================================================================
def midas_return_wf(target: pd.Series, regs: dict[str, pd.Series], K: int = 63, min_train: int = 60):
    """target: 일간 로그수익(%) 품목, regs: {이름: 일간 시계열(수익률 등)} → 다음 달 수익 예측 walk-forward.
    비교: 같은 변수를 '월 평균(동일가중)'으로 집계한 회귀 (MIDAS 가중이 주는 가치 확인)."""
    t = target.dropna()
    dec = pd.DatetimeIndex(pd.Series(t.index, index=t.index).groupby(t.index.to_period("M")).max().values)
    mret = t.groupby(t.index.to_period("M")).sum().values
    y_next = np.r_[mret[1:], np.nan]
    Ls = [hf_lags(s, dec, K) for s in regs.values()]
    ok = ~np.isnan(y_next) & np.all([~np.isnan(L).any(1) for L in Ls], axis=0)
    idx = np.flatnonzero(ok)
    rows = []
    for n_i, i in enumerate(idx):
        if n_i < min_train:
            continue
        tr = idx[:n_i]
        mid = midas_fit(y_next[tr], [L[tr] for L in Ls])
        pm = float(mid["predict"]([L[[i]] for L in Ls])[0])
        Xf = np.column_stack([np.ones(len(tr))] + [L[tr].mean(1) for L in Ls])
        bf = np.linalg.lstsq(Xf, y_next[tr], rcond=None)[0]
        pf = float(np.r_[1, [L[i].mean() for L in Ls]] @ bf)
        rows.append(dict(date=dec[i], y=y_next[i], MIDAS=pm, 동일가중=pf, 과거평균=float(y_next[tr].mean())))
    T = pd.DataFrame(rows).set_index("date")
    oos = {m: 1 - ((T.y - T[m]) ** 2).sum() / ((T.y - T["과거평균"]) ** 2).sum() for m in ["MIDAS", "동일가중"]}
    full = midas_fit(y_next[idx], [L[idx] for L in Ls])
    return dict(table=T, oos=oos, weights={k: w for k, w in zip(regs, full["weights"])}, beta=full["beta"])
