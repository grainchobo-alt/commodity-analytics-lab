"""다변량 GARCH (직접 구현, numba 가속)

  · 단변량 GARCH 요약표     : GARCH / GJR-GARCH (t 분포) — 현재·장기 변동성, 지속성, 반감기, 변동성 기간구조 예측
  · DCC  (Engle 2002)        : 2단계 추정 — ① 단변량 GARCH 표준화잔차 z ② 상관 동학
                                Q_t = (1-a-b)Q̄ + a z_{t-1}z_{t-1}' + b Q_{t-1},  R_t = diag(Q_t)^-½ Q_t diag(Q_t)^-½
  · ADCC (Cappiello-Engle-Sheppard 2006) : + g (n_{t-1}n_{t-1}' - N̄),  n = min(z, 0)  → 동반 하락 시 상관 상승(상관의 레버리지)
  · BEKK(1,1) (Engle-Kroner 1995) : H_t = CC' + A' e_{t-1}e_{t-1}' A + B' H_{t-1} B
                                Full / Diagonal / Scalar. A[i,j]·B[i,j] (i≠j) = i 시장 → j 시장 충격/변동성 전이(spillover)

표준오차: 수치 헤시안 역행렬 (DCC는 2단계 추정이므로 1단계 불확실성 미반영 — Engle-Sheppard 보정 생략).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize


# ============================================================================
# 단변량 GARCH
# ============================================================================
def univariate(R: pd.DataFrame, kind: str = "GJR", dist: str = "t"):
    """각 열에 GARCH 적합. return (Z 표준화잔차, S 조건부 표준편차(기간단위), results dict)."""
    from arch import arch_model
    Z, S, res = {}, {}, {}
    for c in R.columns:
        am = arch_model(R[c], mean="Constant", vol="GARCH", p=1, o=1 if kind == "GJR" else 0, q=1, dist=dist)
        r = am.fit(disp="off", options={"maxiter": 500})
        res[c] = r
        S[c] = r.conditional_volatility
        Z[c] = r.std_resid
    return pd.DataFrame(Z), pd.DataFrame(S), res


def garch_table(R: pd.DataFrame, ann: int, kind: str = "GJR", dist: str = "t", names: dict | None = None) -> pd.DataFrame:
    """품목별 변동성 요약: 현재 조건부 변동성, 장기(무조건부) 변동성, h기 예측, 백분위."""
    _, S, res = univariate(R, kind, dist)
    rows = []
    for c, r in res.items():
        p = r.params
        persist = p.get("alpha[1]", 0) + p.get("beta[1]", 0) + 0.5 * p.get("gamma[1]", 0)
        # 지속성 ≥ 0.99 (IGARCH에 근접) 이면 ω/(1-지속성)이 폭발 → 장기 변동성 미표시
        lr = np.sqrt(p["omega"] / (1 - persist)) * np.sqrt(ann) if persist < 0.99 else np.nan
        cur = S[c].iloc[-1] * np.sqrt(ann)
        f = r.forecast(horizon=22, reindex=False).variance.values[-1]
        rv20 = R[c].iloc[-20:].std() * np.sqrt(ann)
        rows.append({"품목": (names or {}).get(c, c), "현재 GARCH 변동성": cur, "장기 변동성": lr,
                     "1기 예측": np.sqrt(f[0] * ann), "5기 예측": np.sqrt(f[:5].mean() * ann),
                     "22기 예측": np.sqrt(f.mean() * ann), "최근 20기 실현": rv20,
                     "현재 백분위(%)": float((S[c] <= S[c].iloc[-1]).mean() * 100),
                     "α": p.get("alpha[1]", np.nan), "β": p.get("beta[1]", np.nan), "γ(비대칭)": p.get("gamma[1]", np.nan),
                     "지속성": persist, "반감기": np.log(0.5) / np.log(persist) if 0 < persist < 1 else np.nan,
                     "id": c})
    return pd.DataFrame(rows).set_index("품목"), S * np.sqrt(ann)


# ============================================================================
# DCC / ADCC
# ============================================================================
@njit(cache=True)
def _dcc_core(a, b, g, Z, Qbar, Nbar, want_path):
    T, N = Z.shape
    Q = Qbar.copy()
    const = (1.0 - a - b) * Qbar - g * Nbar
    nll = 0.0
    Rpath = np.zeros((T, N, N)) if want_path else np.zeros((1, N, N))
    for t in range(T):
        if t > 0:
            z = Z[t - 1]
            n = np.minimum(z, 0.0)
            Q = const + a * np.outer(z, z) + g * np.outer(n, n) + b * Q
        d = np.sqrt(np.diag(Q))
        Rt = Q / np.outer(d, d)
        L = np.linalg.cholesky(Rt)
        y = np.linalg.solve(L, Z[t])
        nll += 0.5 * (2.0 * np.sum(np.log(np.diag(L))) + y @ y - Z[t] @ Z[t])
        if want_path:
            Rpath[t] = Rt
    return nll, Rpath


def _num_hessian(f, x, eps=1e-4):
    k = len(x); H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ej = np.zeros(k); ei[i] = eps; ej[j] = eps
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * eps * eps)
    return H


def dcc_fit(R: pd.DataFrame, asymmetric: bool = False, kind: str = "GJR", dist: str = "t", ann: int = 52):
    """return dict: params(DataFrame), R_t (T,N,N), Z, S(연율), loglik, pairs(DataFrame 시변상관), H(조건부 공분산)."""
    Z, S, ures = univariate(R, kind, dist)
    Zv = np.ascontiguousarray(Z.values)
    T, N = Zv.shape
    Qbar = np.cov(Zv, rowvar=False, bias=True)
    Nn = np.minimum(Zv, 0.0)
    Nbar = Nn.T @ Nn / T
    # ADCC 정상성: a + b + δ g < 1,  δ = Q̄^-½ N̄ Q̄^-½ 의 최대 고유값
    w, V = np.linalg.eigh(Qbar)
    Qm = V @ np.diag(w ** -0.5) @ V.T
    delta = float(np.max(np.linalg.eigvalsh(Qm @ Nbar @ Qm)))

    def nll(x):
        a, b = x[0], x[1]
        g = x[2] if asymmetric else 0.0
        if a < 0 or b < 0 or g < 0 or a + b + delta * g >= 0.999:
            return 1e10
        try:
            return _dcc_core(a, b, g, Zv, Qbar, Nbar, False)[0]
        except Exception:
            return 1e10

    best = None
    starts = [(0.02, 0.96, 0.01), (0.05, 0.90, 0.02), (0.01, 0.98, 0.005)]
    for s in starts:
        x0 = np.array(s[: 3 if asymmetric else 2])
        r = minimize(nll, x0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
        if best is None or r.fun < best.fun:
            best = r
    x = best.x
    names = ["a (충격)", "b (지속)"] + (["g (비대칭)"] if asymmetric else [])
    try:
        H = _num_hessian(nll, x)
        se = np.sqrt(np.diag(np.linalg.inv(H)))
    except Exception:
        se = np.full(len(x), np.nan)
    params = pd.DataFrame({"계수": x, "표준오차": se, "t값": x / se}, index=names)
    g = x[2] if asymmetric else 0.0
    _, Rp = _dcc_core(x[0], x[1], g, Zv, Qbar, Nbar, True)
    cols = list(R.columns)
    pairs = {}
    for i in range(N):
        for j in range(i + 1, N):
            pairs[(cols[i], cols[j])] = Rp[:, i, j]
    pairs = pd.DataFrame(pairs, index=R.index)
    Sv = S.values
    Hc = Rp * Sv[:, :, None] * Sv[:, None, :]          # 기간 단위 조건부 공분산
    # 단변량 우도 + 상관 우도 = 전체 로그우도
    ll_uni = sum(r.loglikelihood for r in ures.values())
    off = ~np.eye(N, dtype=bool)
    avg = pd.Series(Rp[:, off].mean(axis=1), index=R.index)
    return dict(params=params, R=Rp, Z=Z, S=S * np.sqrt(ann), pairs=pairs, H=Hc, avg_corr=avg,
                loglik=ll_uni - best.fun, corr_loglik=-best.fun, persistence=x[0] + x[1] + delta * g, delta=delta,
                cols=cols, uni=ures)


# ============================================================================
# BEKK(1,1)
# ============================================================================
@njit(cache=True)
def _bekk_core(C, A, B, E, H0, want_path):
    T, N = E.shape
    CC = C @ C.T
    H = H0.copy()
    nll = 0.0
    Hpath = np.zeros((T, N, N)) if want_path else np.zeros((1, N, N))
    for t in range(T):
        if t > 0:
            e = E[t - 1]
            H = CC + A.T @ np.outer(e, e) @ A + B.T @ H @ B
        L = np.linalg.cholesky(H)
        y = np.linalg.solve(L, E[t])
        nll += 0.5 * (2.0 * np.sum(np.log(np.diag(L))) + y @ y + N * np.log(2 * np.pi))
        if want_path:
            Hpath[t] = H
    return nll, Hpath


def _bekk_unpack(x, N, kind):
    k = N * (N + 1) // 2
    C = np.zeros((N, N)); C[np.tril_indices(N)] = x[:k]
    if kind == "Full":
        A = x[k:k + N * N].reshape(N, N); B = x[k + N * N:k + 2 * N * N].reshape(N, N)
    elif kind == "Diagonal":
        A = np.diag(x[k:k + N]); B = np.diag(x[k + N:k + 2 * N])
    else:  # Scalar
        A = np.eye(N) * x[k]; B = np.eye(N) * x[k + 1]
    return C, A, B


def bekk_fit(R: pd.DataFrame, kind: str = "Full", ann: int = 52):
    """return dict: A, B (DataFrame), A_t, B_t (t값), C, H 경로, 조건부 상관·변동성, 정상성, 로그우도, AIC/BIC."""
    E = np.ascontiguousarray((R - R.mean()).values)
    T, N = E.shape
    S0 = np.cov(E, rowvar=False)
    k = N * (N + 1) // 2
    C0 = np.linalg.cholesky(S0 * 0.05)[np.tril_indices(N)]
    if kind == "Full":
        x0 = np.r_[C0, (np.eye(N) * 0.25).ravel(), (np.eye(N) * 0.95).ravel()]
    elif kind == "Diagonal":
        x0 = np.r_[C0, np.full(N, 0.25), np.full(N, 0.95)]
    else:
        x0 = np.r_[C0, 0.25, 0.95]

    def stat_root(A, B):
        return float(np.max(np.abs(np.linalg.eigvals(np.kron(A, A) + np.kron(B, B)))))

    def nll(x):
        C, A, B = _bekk_unpack(x, N, kind)
        if stat_root(A, B) >= 0.9995:
            return 1e10
        try:
            v = _bekk_core(C, A, B, E, S0, False)[0]
            return v if np.isfinite(v) else 1e10
        except Exception:
            return 1e10

    r = minimize(nll, x0, method="L-BFGS-B", options={"maxiter": 3000, "maxfun": 200000})
    if not r.success or r.fun >= 1e9:   # 수렴 실패 시 Nelder-Mead로 재시도
        r2 = minimize(nll, r.x if r.fun < 1e9 else x0, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-6})
        r = r2 if r2.fun < r.fun else r
    x = r.x.copy()
    C, A, B = _bekk_unpack(x, N, kind)
    # 식별: A, B의 (1,1) 원소를 양수로 (A→-A 동일 우도)
    if A[0, 0] < 0: A = -A; x = _bekk_pack(C, A, B, kind)
    if B[0, 0] < 0: B = -B; x = _bekk_pack(C, A, B, kind)
    try:
        Hs = _num_hessian(nll, x, 1e-4)
        se = np.sqrt(np.abs(np.diag(np.linalg.pinv(Hs))))
    except Exception:
        se = np.full(len(x), np.nan)
    _, Ase, Bse = _bekk_unpack(se, N, kind)
    _, Hp = _bekk_core(C, A, B, E, S0, True)
    cols = list(R.columns)
    vol = pd.DataFrame(np.sqrt(np.einsum("tii->ti", Hp)) * np.sqrt(ann), index=R.index, columns=cols)
    d = np.sqrt(np.einsum("tii->ti", Hp))
    Rp = Hp / (d[:, :, None] * d[:, None, :])
    pairs = pd.DataFrame({(cols[i], cols[j]): Rp[:, i, j] for i in range(N) for j in range(i + 1, N)}, index=R.index)
    ll = -r.fun
    npar = len(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        At = np.where(Ase > 0, A / Ase, np.nan); Bt = np.where(Bse > 0, B / Bse, np.nan)
    df = lambda M: pd.DataFrame(M, index=cols, columns=cols)
    return dict(A=df(A), B=df(B), A_t=df(At), B_t=df(Bt), C=df(C), H=Hp, R=Rp, vol=vol, pairs=pairs,
                stationarity=stat_root(A, B), loglik=ll, aic=2 * npar - 2 * ll, bic=npar * np.log(T) - 2 * ll,
                npar=npar, converged=bool(r.success), cols=cols)


def _bekk_pack(C, A, B, kind):
    N = C.shape[0]
    c = C[np.tril_indices(N)]
    if kind == "Full":
        return np.r_[c, A.ravel(), B.ravel()]
    if kind == "Diagonal":
        return np.r_[c, np.diag(A), np.diag(B)]
    return np.r_[c, A[0, 0], B[0, 0]]


def spillover_table(bk: dict, names: dict | None = None, level: float = 1.96) -> pd.DataFrame:
    """BEKK 비대각 원소 중 유의한 전이 경로 목록. A[i,j]: i의 충격(ε²) → j의 분산, B[i,j]: i의 변동성 → j의 변동성."""
    nm = names or {}
    rows = []
    for M, Mt, lbl in [(bk["A"], bk["A_t"], "충격 전이 (ARCH, A)"), (bk["B"], bk["B_t"], "변동성 전이 (GARCH, B)")]:
        for i in M.index:
            for j in M.columns:
                if i == j:
                    continue
                t = Mt.loc[i, j]
                rows.append({"유형": lbl, "from → to": f"{nm.get(i, i)} → {nm.get(j, j)}", "계수": M.loc[i, j],
                             "t값": t, "유의(5%)": "✔" if np.isfinite(t) and abs(t) > level else ""})
    return pd.DataFrame(rows)
