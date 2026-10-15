"""C. 레짐·비선형 계량 모델

  Markov-switching 회귀 (Hamilton 1989, Kim-Nelson 1999)
      y_t = c_{s_t} + z_t'β_{s_t} + σ_{s_t} ε_t,  s_t ∈ {0,1,…} 은닉 마르코프 레짐
      z = 설명변수의 주성분(학습구간 PCA, 기본 3개) — 레짐마다 기울기가 달라 모수가 2배가 되므로 차원 축소
      예측 = Σ_s P(s_t | t−1까지) · (c_s + z_t'β_s)   ← 학습구간 모수 고정, 필터(과거만)로 레짐 확률 갱신
  Threshold 회귀 TAR (Hansen 2000) — 관측 가능한 변수 q가 임계값 γ를 넘는지에 따라 기울기가 달라짐
      y = x'β₁·1(q ≤ γ) + x'β₂·1(q > γ) + e,  γ는 격자 탐색(분위 15~85%),  선형성 검정 sup-F + 고정설명변수 부트스트랩 p값
  STAR (Teräsvirta 1994) — 레짐 전환이 매끄러운 로지스틱 함수
      y = x'β₁ + x'β₂·G(q; γ, c),  G = 1/(1+exp(−γ(q−c)/σ_q)),  선형성 LM 검정 (Luukkonen-Saikkonen-Teräsvirta 1988)
  TVP-VAR 연결성 (Antonakakis, Chatziantoniou & Gabauer 2020; Koop-Korobilis 2014 망각계수 칼만필터)
      VAR 계수가 매 시점 변함 → 롤링창 없이 시점별 일반화 FEVD → 총 연결성 지수(TCI)·순 전이
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================================
# Markov-switching 회귀
# ============================================================================
def ms_regression(D, k: int = 2, n_pc: int = 3, seed: int = 0):
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    Xtr, Xte, ytr = D.X_tr, D.X_te, D.y_tr
    U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    m = max(1, min(n_pc, Xtr.shape[1]))
    V = Vt[:m].T
    Ztr, Zte = Xtr @ V, Xte @ V
    ysd = ytr.std() or 1.0
    mod = MarkovRegression(ytr / ysd, k_regimes=k, exog=Ztr, switching_variance=True)
    best = None
    # 초기값: 기본값(탐색 없음) + 고정 시드 무작위 탐색 6회 → 성공한 것 중 최대우도 (결정적·견고)
    tries = [dict(search_reps=0)] + [dict(search_reps=20, search_iter=10, rng=seed + r) for r in range(6)]
    for kw in tries:
        try:
            r = mod.fit(disp=False, maxiter=300, em_iter=20, **kw)
            if np.isfinite(r.llf) and (best is None or r.llf > best.llf + 1e-9):
                best = r
        except Exception:
            continue
    if best is None:
        raise RuntimeError("Markov-switching 추정 실패")
    params = pd.Series(best.params, index=best.model.param_names)
    # 전체 표본에 '고정 모수'로 필터만 돌림 → P(s_t | t−1까지): 미래정보 없음
    Zall = np.vstack([Ztr, Zte]); yall = np.r_[ytr, D.y_te] / ysd
    full = MarkovRegression(yall, k_regimes=k, exog=Zall, switching_variance=True).filter(best.params)
    pred_prob = np.asarray(full.predicted_marginal_probabilities)               # (T, k)
    c = np.array([params[f"const[{i}]"] for i in range(k)])
    B = np.array([[params[f"x{j+1}[{i}]"] for j in range(m)] for i in range(k)])   # (k, m)
    sig = np.sqrt(np.array([params[f"sigma2[{i}]"] for i in range(k)]))
    fit_s = c[None, :] + Zall @ B.T                                             # (T, k)
    pred = (pred_prob * fit_s).sum(1) * ysd
    order = np.argsort(sig)                                                     # 레짐 = 변동성 오름차순
    P = np.asarray(best.regime_transition)[:, :, 0] if np.ndim(best.regime_transition) == 3 else np.asarray(best.regime_transition)
    stay = np.array([P[i, i] for i in range(k)])
    labels = [f"레짐{i+1}" for i in range(k)]
    idx = D.idx_tr.append(D.idx_te)
    probs = pd.DataFrame(np.asarray(full.filtered_marginal_probabilities)[:, order], index=idx, columns=labels)
    betas_x = pd.DataFrame((B[order] @ V.T) * ysd, index=labels, columns=D.names)   # 원 설명변수 공간의 레짐별 기울기
    regime = pd.DataFrame({"절편": c[order] * ysd, "변동성 σ": sig[order] * ysd, "유지확률": stay[order],
                           "기대지속기간": 1 / (1 - stay[order]), "평균 비중": probs.mean().values}, index=labels)
    n = len(ytr)
    return pred[:n], pred[n:], dict(probs=probs, regime=regime, state_betas=betas_x, n_pc=m, loglik=float(best.llf))


# ============================================================================
# Threshold (TAR) · STAR
# ============================================================================
def _ols(X, y):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    return b, float(e @ e)


def ridge_on(Z, y):
    """레짐 확장 설계 Z에 대한 Ridge (λ는 시계열 CV) — 상수열 포함 Z 그대로 사용."""
    import analytics as A
    b, _ = A.ridge_cv(Z, y)
    return b


def tar_fit(X, y, q, trim: float = 0.15, n_grid: int = 50):
    """γ 격자 탐색. X에 상수 포함 안 됨 → 레짐별 상수 추가."""
    n = len(y)
    Xc = np.column_stack([np.ones(n), X])
    grid = np.unique(np.quantile(q, np.linspace(trim, 1 - trim, n_grid)))
    best = (np.inf, None, None)
    for g in grid:
        lo = (q <= g)[:, None]
        Z = np.column_stack([Xc * lo, Xc * ~lo])
        b, ssr = _ols(Z, y)
        if ssr < best[0]:
            best = (ssr, g, b)
    ssr1, gamma, b = best
    lo = (q <= gamma)[:, None]
    b = ridge_on(np.column_stack([Xc * lo, Xc * ~lo]), y)          # 예측용 계수는 Ridge
    _, ssr0 = _ols(Xc, y)
    k = Xc.shape[1]
    supF = (ssr0 - ssr1) / (ssr1 / (n - 2 * k))
    return dict(gamma=float(gamma), b_low=b[:k], b_high=b[k:], ssr=ssr1, ssr_lin=ssr0, supF=float(supF),
                share_low=float((q <= gamma).mean()), grid=grid)


def tar_bootstrap_p(X, y, q, supF, B: int = 199, trim=0.15, n_grid=30, seed=0):
    """Hansen(1996) 고정설명변수 부트스트랩: 귀무(선형) 잔차를 무작위 부호/재표본 → sup-F 분포."""
    rng = np.random.default_rng(seed)
    n = len(y)
    Xc = np.column_stack([np.ones(n), X])
    b0, _ = _ols(Xc, y); e0 = y - Xc @ b0
    grid = np.unique(np.quantile(q, np.linspace(trim, 1 - trim, n_grid)))
    k = Xc.shape[1]
    # 각 γ의 투영을 미리 계산해 반복 속도 향상
    Zs = [np.column_stack([Xc * (q <= g)[:, None], Xc * (q > g)[:, None]]) for g in grid]
    pinvs = [np.linalg.pinv(Z) for Z in Zs]
    P0 = np.linalg.pinv(Xc)
    cnt = 0
    for _ in range(B):
        yb = e0 * rng.standard_normal(n)                       # 귀무 하 y* (선형 부분은 F 통계량에 무관)
        e_lin = yb - Xc @ (P0 @ yb); ssr0 = e_lin @ e_lin
        ssr1 = min(float(np.sum((yb - Z @ (Pi @ yb)) ** 2)) for Z, Pi in zip(Zs, pinvs))
        Fb = (ssr0 - ssr1) / (ssr1 / (n - 2 * k))
        cnt += Fb >= supF
    return (cnt + 1) / (B + 1)


def _G(q, gamma, c, sq):
    return 1.0 / (1.0 + np.exp(-np.clip(gamma * (q - c) / sq, -50, 50)))


def star_fit(X, y, q):
    from scipy.optimize import minimize
    n = len(y); Xc = np.column_stack([np.ones(n), X]); sq = float(np.std(q)) or 1.0

    def ssr_of(lg, c):
        G = _G(q, np.exp(lg), c, sq)[:, None]
        return _ols(np.column_stack([Xc, Xc * G]), y)[1]
    best = (np.inf, 0.0, 0.0)
    for c in np.quantile(q, np.linspace(0.15, 0.85, 15)):
        for lg in np.log([0.5, 1, 2, 5, 10, 25, 50]):
            v = ssr_of(lg, c)
            if v < best[0]:
                best = (v, lg, c)
    lo_c, hi_c = np.quantile(q, [0.10, 0.90])
    r = minimize(lambda p: ssr_of(p[0], p[1]), np.array([best[1], best[2]]), method="L-BFGS-B",
                 bounds=[(np.log(0.5), np.log(50.0)), (lo_c, hi_c)])   # 전환점이 표본 밖으로 나가 G가 상수가 되는 것 방지
    lg, c = (r.x if r.fun < best[0] else (best[1], best[2]))
    gamma = float(np.exp(lg))
    G = _G(q, gamma, c, sq)[:, None]
    k = Xc.shape[1]
    Z = np.column_stack([Xc, Xc * G])
    _, ssr = _ols(Z, y)
    b = ridge_on(Z, y)                                             # 예측용 계수는 Ridge
    return dict(gamma=gamma, c=float(c), sq=sq, b1=b[:k], b2=b[k:], ssr=ssr)


def star_lm_test(X, y, q):
    """선형성 LM 검정 (Luukkonen-Saikkonen-Teräsvirta 1988, 3차 테일러) — 이분산 강건판 (Wooldridge 1990):
    보조항 W=[X·q, X·q², X·q³]를 X에 회귀한 잔차 R, 선형 잔차 e → 1을 e·R에 회귀, LM = n − SSR ~ χ²(dim W).
    원자재 수익률은 변동성 군집이 강해 일반 LM은 선형성을 과도하게 기각함."""
    from scipy import stats
    n = len(y); Xc = np.column_stack([np.ones(n), X])
    b, _ = _ols(Xc, y); e = y - Xc @ b
    qs = (q - q.mean()) / (q.std() or 1)
    W = np.column_stack([X * qs[:, None], X * (qs ** 2)[:, None], X * (qs ** 3)[:, None]])
    Rw = W - Xc @ np.linalg.lstsq(Xc, W, rcond=None)[0]
    M = e[:, None] * Rw
    _, ssr = _ols(M, np.ones(n))
    lm = n - ssr
    df = W.shape[1]
    return dict(LM=float(lm), df=df, p=float(stats.chi2.sf(lm, df)))


def threshold_column(D, thr_var: str | None):
    """임계 변수: 설계행렬의 열 이름(부분 일치). 기본 = VIX가 있으면 VIX, 아니면 첫 열."""
    names = D.names
    if thr_var:
        for j, nm in enumerate(names):
            if nm == thr_var or nm.startswith(thr_var + " (") or nm.startswith(thr_var):
                return j
    for j, nm in enumerate(names):
        if "VIX" in nm:
            return j
    return 0


# ============================================================================
# TVP-VAR 연결성 (망각계수 칼만필터)
# ============================================================================
def tvp_var_connectedness(R: pd.DataFrame, p: int = 1, H: int = 10, kappa1: float = 0.99, kappa2: float = 0.96,
                          burn: int = 60) -> dict:
    """Koop-Korobilis(2014) 망각계수 TVP-VAR → 시점별 일반화 FEVD (Pesaran-Shin) → 연결성.
    β_t | t−1 ~ N(β_{t−1|t−1}, P/κ1),  Σ_t = κ2 Σ_{t−1} + (1−κ2) ε_t ε_t'.  초기값 = 첫 burn 기간 OLS.
    필터(과거만)를 쓰므로 각 시점의 연결성은 그 시점까지의 정보로 계산됨."""
    Y = R.values.astype(float); T, N = Y.shape
    Zrows = [np.r_[1.0, np.concatenate([Y[t - l] for l in range(1, p + 1)])] for t in range(p, T)]
    Z = np.array(Zrows); Yt = Y[p:]
    m = Z.shape[1]
    B0 = np.linalg.lstsq(Z[:burn], Yt[:burn], rcond=None)[0]                 # (m, N)
    E0 = Yt[:burn] - Z[:burn] @ B0
    Sig = np.cov(E0, rowvar=False)
    beta = B0.T.reshape(-1)                                                  # N*m (식별 순)
    Pm = np.eye(N * m) * 0.1
    tci, net, to_, from_, dates, Acoef = [], [], [], [], [], []
    for t in range(len(Yt)):
        Xk = np.kron(np.eye(N), Z[t][None, :])                               # (N, N*m)
        Pp = Pm / kappa1
        yhat = Xk @ beta
        e = Yt[t] - yhat
        S = Xk @ Pp @ Xk.T + Sig
        K = Pp @ Xk.T @ np.linalg.inv(S)
        beta = beta + K @ e
        Pm = Pp - K @ Xk @ Pp
        e_post = Yt[t] - Xk @ beta
        Sig = kappa2 * Sig + (1 - kappa2) * np.outer(e_post, e_post)
        if t < burn:
            continue
        Bt = beta.reshape(N, m)                                              # 행 = 식
        A = [Bt[:, 1 + (l * N):1 + (l + 1) * N] for l in range(p)]
        Phi = [np.eye(N)]
        for h in range(1, H):
            Phi.append(sum(A[l] @ Phi[h - 1 - l] for l in range(min(p, h))))
        num = np.zeros((N, N)); den = np.zeros(N)
        for h in range(H):
            M_ = Phi[h] @ Sig
            num += M_ ** 2
            den += np.diag(Phi[h] @ Sig @ Phi[h].T)
        th = num / np.diag(Sig)[None, :] / den[:, None]
        th = th / th.sum(1, keepdims=True) * 100
        off = th.sum() - np.trace(th)
        tci.append(off / N * N / (N - 1))                                    # ACG(2020) 조정: N/(N−1)
        t_ = th.sum(0) - np.diag(th); f_ = th.sum(1) - np.diag(th)
        to_.append(t_); from_.append(f_); net.append(t_ - f_)
        dates.append(R.index[t + p]); Acoef.append(Bt[:, 1:1 + N].copy())
    idx = pd.DatetimeIndex(dates)
    cols = list(R.columns)
    return dict(tci=pd.Series(tci, index=idx), net=pd.DataFrame(net, index=idx, columns=cols),
                to=pd.DataFrame(to_, index=idx, columns=cols), frm=pd.DataFrame(from_, index=idx, columns=cols),
                A=np.array(Acoef), cols=cols)
