"""A. 변수를 버리지 않는 예측 기법 — 요인 압축 · 모형 평균 · 베이지안 연속 축소

요인 압축
  PCR    : X의 주성분 k개로 회귀 (Stock & Watson 2002 '확산지수'). k는 시계열 CV
  PLS    : y와의 공분산이 큰 방향으로 요인 추출 (지도 요인). k는 시계열 CV
  3PRF   : Kelly & Pruitt (2015) 3단계 회귀 필터 — ① 변수별로 프록시(y)에 시계열 회귀 → 민감도 φ
           ② 시점별로 변수들을 φ에 횡단면 회귀 → 요인 F_t  ③ y를 F에 회귀. 자동 프록시로 L개 요인. L은 CV
모형 평균
  Complete Subset : 크기 k인 모든(또는 무작위 M개) 변수 조합 OLS 예측의 평균 (Elliott, Gargano & Timmermann 2013)
  BMA    : Zellner g-prior(g=max(n,p²), Fernández-Ley-Steel 2001) + 베타-이항 모형 사전분포, MC³ 표본추출.
           변수별 사후 포함확률(PIP) = 포인트-매스 spike-and-slab
  Stacking : 기본 모델(Ridge·Lasso·PCR·PLS·Gradient Boosting)의 시계열 CV 외표본(OOF) 예측에 비음 가중치(NNLS, 합=1) 학습
             (Wolpert 1992, Breiman 1996). 비선형 기본모델은 RF 대신 20배 빠른 GBM
베이지안 연속 축소
  Horseshoe : Carvalho-Polson-Scott (2010), Makalic-Schmidt (2016) 보조변수 Gibbs. 계수를 0으로 자르지 않고
              작은 계수만 강하게 축소 → 사후평균·90% 신용구간·P(β>0)
  Minnesota BVAR : 목표식에 대한 Litterman/Minnesota 사전분포 — 시차 L의 사전분산 (λ/(L+1))², 타 변수는 θ² 배
              (먼 시차·타 변수일수록 강하게 0으로 축소). λ는 CV
모든 추정은 학습구간 내부에서만 수행 (CV도 학습구간 내부, h>1이면 gap) → 미래정보 없음.
"""
from __future__ import annotations

import re
from math import comb

import numpy as np
import pandas as pd

DENSE_MODELS = ["PCR (주성분회귀)", "PLS (부분최소제곱)", "3PRF (3단계 회귀필터)", "Complete Subset",
                "BMA (베이즈 모형평균)", "Horseshoe (베이즈 축소)", "Minnesota BVAR", "Stacking (가중 앙상블)"]


def _splits(n, gap=0, n_splits=5):
    from sklearn.model_selection import TimeSeriesSplit
    return list(TimeSeriesSplit(n_splits=n_splits, gap=gap).split(np.arange(n)))


def _cv_pick(fit, X, y, grid, gap):
    """fit(Xtr, ytr, param) -> (beta, intercept). 시계열 CV 평균제곱오차 최소 파라미터."""
    errs = []
    for g in grid:
        e = 0.0
        for tr, va in _splits(len(y), gap):
            ym = y[tr].mean()
            b, c = fit(X[tr], y[tr] - ym, g)
            e += np.mean((y[va] - (X[va] @ b + c + ym)) ** 2)
        errs.append(e)
    return grid[int(np.argmin(errs))], pd.Series(errs, index=grid)


# ---- 요인 압축 ------------------------------------------------------------------
def _pcr(X, y, k):
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt[:k].T
    g = np.linalg.lstsq(X @ V, y, rcond=None)[0]
    return V @ g, 0.0


def _pls(X, y, k):
    from sklearn.cross_decomposition import PLSRegression
    m = PLSRegression(n_components=k, scale=False).fit(X, y)
    b = np.asarray(m.coef_).reshape(-1)
    c = float(np.mean(m.predict(X).ravel() - X @ b))
    return b, c


def _prf3(X, y, L):
    T, N = X.shape
    proxies = [y]
    for l in range(L):
        Zc = np.column_stack([np.ones(T)] + proxies)
        Phi = np.linalg.lstsq(Zc, X, rcond=None)[0][1:].T          # ① N × l 민감도
        Pc = np.column_stack([np.ones(N), Phi])
        Pinv = np.linalg.pinv(Pc)                                   # ② [c_t; F_t] = Pinv x_t
        F = (Pinv[1:] @ X.T).T
        Fc = np.column_stack([np.ones(T), F])
        b = np.linalg.lstsq(Fc, y, rcond=None)[0]                   # ③ y ~ F
        if l < L - 1:
            proxies.append(y - Fc @ b)                              # 자동 프록시: 잔차로 다음 요인
    beta = Pinv[1:].T @ b[1:]
    return beta, float(b[0])


# ---- 모형 평균 -------------------------------------------------------------------
def _subsets(p, k, M, rng):
    if comb(p, k) <= M:
        from itertools import combinations
        return [np.array(s) for s in combinations(range(p), k)]
    return [np.sort(rng.choice(p, k, replace=False)) for _ in range(M)]


def _csr(X, y, k, M=2000, seed=0):
    rng = np.random.default_rng(seed)
    p = X.shape[1]
    XtX, Xty = X.T @ X, X.T @ y
    B = np.zeros(p); subs = _subsets(p, k, M, rng)
    for S in subs:
        B[S] += np.linalg.solve(XtX[np.ix_(S, S)] + 1e-9 * np.eye(len(S)), Xty[S])
    return B / len(subs), 0.0


def bma_mc3(X, y, iters=20000, burn=2000, seed=0):
    """Zellner g-prior BMA, MC³. return (사후평균 β, PIP, 상위 모형표, 방문 모형 수)."""
    from scipy.special import gammaln
    rng = np.random.default_rng(seed)
    n, p = X.shape
    g = max(n, p * p)
    XtX, Xty, yty = X.T @ X, X.T @ y, float(y @ y)
    cache = {}

    def evaluate(mask):
        key = mask.tobytes()
        if key in cache:
            return cache[key]
        S = np.flatnonzero(mask); k = len(S)
        if k:
            b = np.linalg.solve(XtX[np.ix_(S, S)] + 1e-10 * np.eye(k), Xty[S])
            R2 = min(float(Xty[S] @ b) / yty, 1 - 1e-12)
        else:
            b, R2 = np.zeros(0), 0.0
        lml = 0.5 * (n - 1 - k) * np.log1p(g) - 0.5 * (n - 1) * np.log1p(g * (1 - R2))
        lpr = gammaln(k + 1) + gammaln(p - k + 1) - gammaln(p + 2)          # 베타-이항(모형 크기 균등)
        cache[key] = (lml + lpr, S, b * g / (1 + g))
        return cache[key]

    cur = np.zeros(p, bool); lp_cur, S_cur, b_cur = evaluate(cur)
    pip = np.zeros(p); beta = np.zeros(p); visits = {}
    for it in range(iters):
        prop = cur.copy(); j = rng.integers(p); prop[j] = ~prop[j]
        lp_new, S_new, b_new = evaluate(prop)
        if np.log(rng.random()) < lp_new - lp_cur:
            cur, lp_cur, S_cur, b_cur = prop, lp_new, S_new, b_new
        if it >= burn:
            pip += cur; beta[S_cur] += b_cur
            visits[cur.tobytes()] = visits.get(cur.tobytes(), 0) + 1
    m = iters - burn
    top = sorted(visits.items(), key=lambda kv: -kv[1])[:10]
    top_tbl = pd.DataFrame([{"변수 수": int(np.frombuffer(k_, dtype=bool).sum()), "사후확률(방문비율)": v / m,
                             "변수": list(np.flatnonzero(np.frombuffer(k_, dtype=bool)))} for k_, v in top])
    return beta / m, pip / m, top_tbl, len(visits)


# ---- 베이지안 연속 축소 ---------------------------------------------------------
def horseshoe_gibbs(X, y, iters=2000, burn=500, seed=0):
    """Makalic & Schmidt (2016) 보조변수 Gibbs. return (사후평균, 5%, 95%, P(β>0), 보존비율 1−κ)."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    XtX, Xty = X.T @ X, X.T @ y
    ig = lambda a, b: b / rng.gamma(a, 1.0, size=np.shape(b)) if np.ndim(b) else b / rng.gamma(a)
    sigma2 = float(np.var(y)); lam2 = np.ones(p); tau2 = 1.0; nu = np.ones(p); xi = 1.0
    draws, kappa = [], []
    for it in range(iters):
        A = XtX + np.diag(1.0 / (lam2 * tau2))
        Lc = np.linalg.cholesky(A)
        mean = np.linalg.solve(Lc.T, np.linalg.solve(Lc, Xty))
        beta = mean + np.sqrt(sigma2) * np.linalg.solve(Lc.T, rng.standard_normal(p))
        res = y - X @ beta
        sigma2 = ig((n + p) / 2, (res @ res + np.sum(beta ** 2 / (lam2 * tau2))) / 2)
        lam2 = ig(1.0, 1.0 / nu + beta ** 2 / (2 * tau2 * sigma2))
        tau2 = ig((p + 1) / 2, 1.0 / xi + np.sum(beta ** 2 / lam2) / (2 * sigma2))
        nu = ig(1.0, 1.0 + 1.0 / lam2)
        xi = ig(1.0, 1.0 + 1.0 / tau2)
        if it >= burn:
            draws.append(beta)
            kappa.append(1.0 / (1.0 + n * tau2 * lam2))            # 축소계수 κ (표준화 X: X'X ≈ nI)
    Bd = np.array(draws)
    return Bd.mean(0), np.percentile(Bd, 5, 0), np.percentile(Bd, 95, 0), (Bd > 0).mean(0), 1 - np.mean(kappa, 0)


def _minnesota(X, y, lam, names, target, theta=0.5, sigma2=None):
    """목표식 Minnesota 사전분포 사후평균: β = (X'X + σ²V⁻¹)⁻¹ X'y,  V_j = (λ·s_y/(L_j+1))² × (θ² if 타변수)."""
    sy2 = float(np.var(y)) if sigma2 is None else sigma2
    v = []
    for nm in names:
        mlag = re.search(r"\(t-(\d+)\)$", nm)
        L = int(mlag.group(1)) if mlag else 0
        own = bool(target) and nm.startswith(target + " (")
        v.append((lam / (L + 1)) ** 2 * sy2 * (1.0 if own else theta ** 2))
    V = np.array(v)
    res2 = float(np.var(y - X @ np.linalg.lstsq(X, y, rcond=None)[0]))
    return np.linalg.solve(X.T @ X + np.diag(res2 / V), X.T @ y), 0.0


# ---- 실행기 ---------------------------------------------------------------------
def run_dense(name: str, D, ann: int, **hp):
    import analytics as A
    Xtr, ytr, Xte, yte = D.X_tr, D.y_tr, D.X_te, D.y_te
    gap = max(0, D.horizon - 1)
    p = Xtr.shape[1]
    extra = {}
    if name.startswith("PCR"):
        k, cv = _cv_pick(_pcr, Xtr, ytr, list(range(1, min(hp.get("k_max", 10), p) + 1)), gap)
        b, c = _pcr(Xtr, ytr, k)
        _, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
        extra.update(k=k, cv=cv, explained=pd.Series((s ** 2 / (s ** 2).sum())[:10], index=[f"PC{i+1}" for i in range(min(10, len(s)))]),
                     loadings=pd.DataFrame(Vt[:k].T, index=D.names, columns=[f"PC{i+1}" for i in range(k)]))
    elif name.startswith("PLS"):
        k, cv = _cv_pick(_pls, Xtr, ytr, list(range(1, min(hp.get("k_max", 10), p) + 1)), gap)
        b, c = _pls(Xtr, ytr, k); extra.update(k=k, cv=cv)
    elif name.startswith("3PRF"):
        L, cv = _cv_pick(_prf3, Xtr, ytr, [1, 2, 3], gap)
        b, c = _prf3(Xtr, ytr, L); extra.update(L=L, cv=cv)
    elif name.startswith("Complete Subset"):
        M = hp.get("csr_M", 2000)
        k, cv = _cv_pick(lambda X, y, k: _csr(X, y, k, min(300, M)), Xtr, ytr,
                         list(range(1, min(hp.get("csr_kmax", 8), p) + 1)), gap)
        b, c = _csr(Xtr, ytr, k, M); extra.update(k=k, cv=cv, n_subsets=min(M, comb(p, k)))
    elif name.startswith("BMA"):
        b, pip, top, nvis = bma_mc3(Xtr, ytr, hp.get("bma_iters", 20000))
        c = 0.0
        top["변수"] = top["변수"].apply(lambda S: ", ".join(D.names[j] for j in S) if len(S) else "(상수만)")
        extra.update(pip=pd.Series(pip, index=D.names), top_models=top, visited=nvis)
    elif name.startswith("Horseshoe"):
        b, lo, hi, ppos, keep = horseshoe_gibbs(Xtr, ytr, hp.get("hs_iters", 2000))
        c = 0.0
        extra.update(ci=pd.DataFrame({"사후평균": b, "5%": lo, "95%": hi, "P(β>0)": ppos, "보존비율 1−κ": keep}, index=D.names))
    elif name.startswith("Minnesota"):
        theta = hp.get("mn_theta", 0.5); tgt = getattr(D, "target", "")
        lam, cv = _cv_pick(lambda X, y, l: _minnesota(X, y, l, D.names, tgt, theta), Xtr, ytr,
                           list(np.logspace(-3, 1, 17)), gap)
        b, c = _minnesota(Xtr, ytr, lam, D.names, tgt, theta)
        extra.update(lam=float(lam), theta=theta, cv=cv)
    elif name.startswith("Stacking"):
        return _stacking(D, ann, hp.get("stack_base", ["Ridge", "Lasso", "PCR (주성분회귀)", "PLS (부분최소제곱)", "Gradient Boosting"]), **hp)
    else:
        raise ValueError(name)
    b = np.asarray(b, float)
    p_tr, p_te = Xtr @ b + c, Xte @ b + c
    met = A._metrics(ytr, p_tr, yte, p_te, ann, D.horizon, D)
    return dict(coef=pd.Series(b, index=D.names), pred_tr=p_tr, pred_te=p_te, metrics=met, extra=extra)


def _stacking(D, ann, base, **hp):
    import analytics as A
    from scipy.optimize import nnls
    Xtr, ytr = D.X_tr, D.y_tr
    gap = max(0, D.horizon - 1)
    oof = np.full((len(ytr), len(base)), np.nan)
    hp2 = {k: v for k, v in hp.items() if k not in ("stack_base", "n_jobs")}
    hp2.update(_fast=True)
    tasks = []
    for tr, va in _splits(len(ytr), gap):
        for j, m in enumerate(base):
            tasks.append((tr, va, j, m))

    def one(tr, va, j, m):
        ym = ytr[tr].mean()
        Dk = A.Design(Xtr[tr], ytr[tr] - ym, Xtr[va], ytr[va] - ym, D.idx_tr[tr], D.idx_tr[va], D.names,
                      D.mu, D.sd, ym, None, D.horizon, D.mode, D.cost, getattr(D, "target", ""))
        return A.run_model(m, Dk, ann, **dict(hp2, _inner_jobs=1))["pred_te"] + ym
    if hp.get("_inner_jobs", -1) == 1:      # 이미 병렬 작업자 안 (walk-forward) → 순차
        outs = [one(*tk) for tk in tasks]
    else:
        from joblib import Parallel, delayed
        outs = Parallel(n_jobs=-1, prefer="threads")(delayed(one)(*tk) for tk in tasks)
    for (tr, va, j, m), o in zip(tasks, outs):
        oof[va, j] = o
    ok = ~np.isnan(oof).any(axis=1)
    w, _ = nnls(oof[ok], ytr[ok])
    w = w / w.sum() if w.sum() > 0 else np.full(len(base), 1 / len(base))
    fits = {m: A.run_model(m, D, ann, **{k: v for k, v in hp2.items() if k != "_fast"}) for m in base}
    p_tr = sum(w[j] * np.asarray(fits[m]["pred_tr"]) for j, m in enumerate(base))
    p_te = sum(w[j] * np.asarray(fits[m]["pred_te"]) for j, m in enumerate(base))
    lin = [m for m in base if fits[m]["coef"] is not None]
    coef = sum(w[base.index(m)] * fits[m]["coef"] for m in lin) if len(lin) == len(base) else None
    base_met = pd.DataFrame({m: fits[m]["metrics"] for m in base}).T[["R² (테스트)", "OOS R² (vs 학습평균)"]]
    return dict(coef=coef, pred_tr=p_tr, pred_te=p_te, metrics=A._metrics(ytr, p_tr, D.y_te, p_te, ann, D.horizon, D),
                extra=dict(stack_weights=pd.Series(w, index=base), base_metrics=base_met))
