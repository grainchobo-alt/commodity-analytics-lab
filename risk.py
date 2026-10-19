"""리스크 엔진 — 시변 감쇠 변동성 · 예측 검증 · VaR/ES · 꼬리 위험.

변동성 모형 (모두 walk-forward: 매년 1월 그 이전 데이터로만 재추정, s2[t] = r[t]의 분산 예측 (t−1 종가까지 정보)):
  RiskMetrics λ=0.94       : 모든 품목 공통 감쇠 (J.P. Morgan 1996)
  EWMA 품목별 λ (확장)      : σ²ₜ = λσ²ₜ₋₁ + (1−λ)r²ₜ₋₁, λ = QMLE (과거 전체)
  EWMA 롤링 λ (직전 3년)    : 같은 식, λ를 직전 3년만으로 재추정 → 같은 품목 안에서도 λ가 해마다 바뀜
  HMM 3상태 λ               : 저·중·고변동 레짐별 λₛ (레짐 확률 가중 QMLE) → λₜ = Σₛ P(sₜ₋₁=s | r₁..ₜ₋₁)·λₛ (필터 확률만)
  Kalman TVP-AR(1) + EWMA   : rₜ = αₜ + φₜrₜ₋₁ + εₜ (계수 랜덤워크, 칼만 필터 1기 선행 혁신) → 혁신에 품목별 EWMA
  GARCH(1,1) · HAR-RV · MIDAS-RV : 학계 표준 비교 기준
검증: QLIKE 손실, Diebold-Mariano(Newey-West), Model Confidence Set(Hansen·Lunde·Nason 2011, 블록 부트스트랩),
      우도비(품목 간 공통 λ / 기간 간 공통 λ), Engle-Ng 부호 편향, Ljung-Box(잔여 ARCH), 다변량 마할라노비스 적합 진단.
VaR/ES: 정규 · Student-t · FHS(필터링 역사적 시뮬레이션) · EVT-GPD(조건부 POT, McNeil-Frey) · 역사적 시뮬레이션(500일)
        백테스트: Kupiec POF, Christoffersen 독립성·조건부 적중, McNeil-Frey ES 검정, 바젤 신호등.
수익률 단위: 일간 로그수익률 % (100×Δlog). 포지션 방향: long = 가격 하락이 손실, short = 가격 상승이 손실 (매입 예정 물량).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

LAM_GRID = np.linspace(0.90, 0.999, 100)        # 일간 λ 범위 (반감기 6.6일 ~ 1.9년)
VOL_MODELS = ["RiskMetrics λ=0.94", "EWMA 품목별 λ (확장)", "EWMA 롤링 λ (직전 3년)", "HMM 3상태 λ",
              "Kalman TVP-AR(1) + EWMA", "GARCH(1,1)", "HAR-RV", "MIDAS-RV"]
TV_MODELS = ["EWMA 롤링 λ (직전 3년)", "HMM 3상태 λ", "Kalman TVP-AR(1) + EWMA"]   # 같은 품목 안에서 λ(또는 잔차 구조)가 시변
VAR_METHODS = ["정규", "Student-t", "FHS (필터링 역사적)", "EVT-GPD (조건부 POT)", "역사적 시뮬레이션 (500일)"]
hl_days = lambda lam: np.log(0.5) / np.log(np.asarray(lam, float))


def returns(p: pd.Series) -> pd.Series:
    p = p.dropna(); p = p[(p > 0) & (p.index.dayofweek < 5)]
    return (np.log(p).diff() * 100).dropna().rename(p.name)


# ============================================================================
# EWMA · QMLE
# ============================================================================
RM_LAMBDA = 0.94                                # RiskMetrics 일간 감쇠 (J.P. Morgan 1996)


def ewma_var(r: pd.Series, lam: float = RM_LAMBDA) -> pd.Series:
    """RiskMetrics 지수가중 분산 σ²ₜ = λσ²ₜ₋₁ + (1−λ)r²ₜ (평균 0, 초기값 r₀²). 값 t = t 종가까지의 정보로 본 '다음 날' 분산.
    midas.daily_vol_forecasts(EWMA 열 → 포트폴리오 변동성 모형)가 사용. walk-forward λ·시변 λ는 ewma_path."""
    return (r ** 2).ewm(alpha=round(1 - lam, 10), adjust=False).mean()


def ewma_std(r: pd.Series, lam: float = RM_LAMBDA) -> pd.Series:
    """지수가중 표준편차 (평균 차감, pandas ewm.std) — 구매 타이밍의 변동성 분위·연 변동성 표시용."""
    return r.ewm(alpha=round(1 - lam, 10), adjust=False).std()


def _seed(e: np.ndarray) -> float:
    return float(np.var(e[:20])) if len(e) >= 20 else float(np.var(e) or 1.0)


def ewma_ll_grid(e: np.ndarray, lams: np.ndarray = LAM_GRID) -> np.ndarray:
    """λ 그리드별 −2·로그우도 기여 (G, T). 열 t는 r_t의 예측(≤ t−1 정보)으로 평가 → 어느 창의 합을 취해도 인과적."""
    G, T = len(lams), len(e)
    s2 = np.full(G, max(_seed(e), 1e-8)); e2 = e ** 2
    LL = np.zeros((G, T))
    for t in range(1, T):
        s2 = lams * s2 + (1 - lams) * e2[t - 1]
        np.maximum(s2, 1e-10, out=s2)
        LL[:, t] = np.log(s2) + e2[t] / s2
    return LL


def ewma_path(e: np.ndarray, lam) -> np.ndarray:
    """시변 가능 λ (스칼라 또는 길이 T 배열, λ[t]는 t−1까지의 정보로 정한 값) → s2[t]."""
    lam = np.broadcast_to(np.asarray(lam, float), (len(e),))
    s2 = np.empty(len(e)); s2[0] = max(_seed(e), 1e-8)
    for t in range(1, len(e)):
        s2[t] = max(lam[t] * s2[t - 1] + (1 - lam[t]) * e[t - 1] ** 2, 1e-10)
    return s2


def _years(idx: pd.DatetimeIndex):
    y = idx.year.values
    return y, np.unique(y)


def wf_lambda(e: np.ndarray, idx: pd.DatetimeIndex, window_years: int | None = None, min_obs: int = 500,
              LL: np.ndarray | None = None, lams: np.ndarray = LAM_GRID) -> np.ndarray:
    """매년 1월 λ를 그 이전 데이터(확장 또는 직전 window_years년)로 QMLE 재추정 → 날짜별 λ (그해 동안 고정)."""
    LL = ewma_ll_grid(e, lams) if LL is None else LL
    y, uy = _years(idx)
    out = np.full(len(e), np.nan)
    for yr in uy:
        tr = (y < yr) & ((y >= yr - window_years) if window_years else True)
        tr[0] = False
        if tr.sum() >= min_obs:
            out[y == yr] = lams[LL[:, tr].sum(1).argmin()]
    return out


def lambda_ci(LLsum: np.ndarray, lams: np.ndarray = LAM_GRID, level: float = 0.95) -> tuple[float, float, float]:
    """프로파일 우도 신뢰구간: 2·Δℓ ≤ χ²₁ 인 λ 범위."""
    c = stats.chi2.ppf(level, 1)
    i = LLsum.argmin(); ok = np.flatnonzero(LLsum - LLsum[i] <= c)
    return lams[i], lams[ok.min()], lams[ok.max()]


# ============================================================================
# HMM 3상태 λ
# ============================================================================
def hmm_filter(x: np.ndarray, m) -> np.ndarray:
    """가우시안 HMM(hmmlearn, 1차원) 전방 필터 P(s_t | x_1..t) (평활 확률은 미래를 쓰므로 사용 안 함).
    상태는 분산 오름차순으로 정렬해 반환 → 마지막 열 = 고변동(위기) 레짐. 구매 타이밍 HMM 위기 확률도 이 함수를 씀."""
    mu = m.means_.ravel(); sd = np.sqrt(np.array([np.ravel(c)[0] for c in m.covars_]))
    o = np.argsort(sd)
    lb = stats.norm.logpdf(x[:, None], mu[None, :], sd[None, :])
    A = m.transmat_; a = m.startprob_ * np.exp(lb[0] - lb[0].max()); a /= a.sum()
    F = np.empty((len(x), len(mu))); F[0] = a
    for t in range(1, len(x)):
        a = (a @ A) * np.exp(lb[t] - lb[t].max()); s = a.sum()
        a = a / s if s > 0 else np.full(len(mu), 1 / len(mu)); F[t] = a
    return F[:, o]


def hmm_lambda(e: np.ndarray, idx: pd.DatetimeIndex, n_states: int = 3, min_years: int = 5, LL: np.ndarray | None = None,
               lams: np.ndarray = LAM_GRID, seed: int = 0) -> dict:
    """매년 그 이전 데이터로 HMM 재적합 + 레짐별 λₛ = argmin Σₜ γₜₛ·ℓₜ(λ) (학습 구간 안의 평활 확률 = 과거 정보만).
    그해의 λₜ = Σₛ P(sₜ₋₁ | r₁..ₜ₋₁)·λₛ."""
    from hmmlearn.hmm import GaussianHMM
    LL = ewma_ll_grid(e, lams) if LL is None else LL
    y, uy = _years(idx)
    lam_t = np.full(len(e), np.nan); prob = np.full((len(e), n_states), np.nan)
    table = []
    for yr in uy:
        tr = y < yr
        if tr.sum() < 252 * min_years:
            continue
        x = e[tr]
        try:
            m = GaussianHMM(n_states, "diag", n_iter=100, tol=1e-3, random_state=seed).fit(x[:, None])
        except Exception:
            continue
        sd = np.sqrt(np.array([np.ravel(c)[0] for c in m.covars_])); o = np.argsort(sd)
        gam = m.predict_proba(x[:, None])[:, o]                             # 학습 구간(과거)만의 평활 확률
        ls = np.array([lams[(LL[:, :tr.sum()] * gam[:, s][None, :]).sum(1).argmin()] for s in range(n_states)])
        seg = np.flatnonzero(y <= yr)
        F = hmm_filter(e[seg], m)
        cur = seg[y[seg] == yr]
        prev = np.clip(cur - 1, 0, None)                                    # t−1 시점의 필터 확률
        lam_t[cur] = F[prev] @ ls
        prob[cur] = F[cur]
        table.append(dict(연도=int(yr), **{f"λ_{nm}": float(v) for nm, v in zip(["저변동", "중변동", "고변동"][:n_states], ls)},
                          **{f"점유율_{nm}": float(g) for nm, g in zip(["저변동", "중변동", "고변동"][:n_states], gam.mean(0))},
                          **{f"σ_{nm}(%)": float(v) for nm, v in zip(["저변동", "중변동", "고변동"][:n_states], sd[o])}))
    return dict(lam=lam_t, prob=prob, table=pd.DataFrame(table))


# ============================================================================
# Kalman TVP-AR(1)
# ============================================================================
def kalman_tvp_ar1(r: np.ndarray, q: float = 1e-5, r_var: float | None = None) -> dict:
    """rₜ = αₜ + φₜrₜ₋₁ + εₜ, [α,φ]ₜ = [α,φ]ₜ₋₁ + η (η~N(0,qI)). 반환: 1기 선행 혁신 eₜ = rₜ − Hₜx̂ₜ|ₜ₋₁ (t−1 정보로 예측), 계수 경로.
    관측 잡음 R은 초기 1년 분산으로 고정 (이후 데이터를 쓰지 않음)."""
    T = len(r)
    R = float(np.var(r[:252])) if r_var is None else r_var
    x = np.zeros(2); P = np.eye(2) * 1.0; Q = np.eye(2) * q * R
    innov = np.full(T, np.nan); mu = np.full(T, np.nan); coef = np.full((T, 2), np.nan)
    for t in range(1, T):
        P = P + Q
        H = np.array([1.0, r[t - 1]])
        mu[t] = H @ x; innov[t] = r[t] - mu[t]
        S = H @ P @ H + R; K = P @ H / S
        x = x + K * innov[t]; P = P - np.outer(K, H) @ P
        coef[t] = x
    innov[0] = r[0]
    return dict(innov=innov, mu=np.nan_to_num(mu), coef=coef)


# ============================================================================
# 변동성 모형 경주 (walk-forward)
# ============================================================================
def garch_path(e: np.ndarray, idx: pd.DatetimeIndex, min_obs: int = 750) -> np.ndarray:
    from arch import arch_model
    y, uy = _years(idx)
    out = np.full(len(e), np.nan)
    for yr in uy:
        tr = y < yr
        if tr.sum() < min_obs:
            continue
        try:
            res = arch_model(e[tr], mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(disp="off", show_warning=False)
            seg = y <= yr
            fx = arch_model(e[seg], mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fix(res.params)
            out[y == yr] = np.asarray(fx.conditional_volatility)[y[seg] == yr] ** 2
        except Exception:
            continue
    return out


def vol_models(r: pd.Series, models: list[str] | None = None, q_kalman: float = 1e-5) -> dict:
    """품목 1개의 일간 수익률(%) → 모형별 분산 예측 S (DataFrame, s2[t] = r[t] 분산 예측), λ 경로, HMM 확률, Kalman 계수."""
    models = models or VOL_MODELS
    r = r.dropna(); e = r.values; idx = r.index
    LL = ewma_ll_grid(e)
    S, lam = {}, {}
    if "RiskMetrics λ=0.94" in models:
        S["RiskMetrics λ=0.94"] = ewma_path(e, 0.94); lam["RiskMetrics λ=0.94"] = np.full(len(e), 0.94)
    if "EWMA 품목별 λ (확장)" in models:
        l_ = wf_lambda(e, idx, None, LL=LL); S["EWMA 품목별 λ (확장)"] = np.where(np.isnan(l_), np.nan, ewma_path(e, np.nan_to_num(l_, nan=0.94)))
        lam["EWMA 품목별 λ (확장)"] = l_
    if "EWMA 롤링 λ (직전 3년)" in models:
        l_ = wf_lambda(e, idx, 3, LL=LL); S["EWMA 롤링 λ (직전 3년)"] = np.where(np.isnan(l_), np.nan, ewma_path(e, np.nan_to_num(l_, nan=0.94)))
        lam["EWMA 롤링 λ (직전 3년)"] = l_
    hm = None
    if "HMM 3상태 λ" in models:
        hm = hmm_lambda(e, idx, 3, LL=LL)
        S["HMM 3상태 λ"] = np.where(np.isnan(hm["lam"]), np.nan, ewma_path(e, np.nan_to_num(hm["lam"], nan=0.94)))
        lam["HMM 3상태 λ"] = hm["lam"]
    kf = None
    if "Kalman TVP-AR(1) + EWMA" in models:
        kf = kalman_tvp_ar1(e, q_kalman)
        ek = np.nan_to_num(kf["innov"]); l_ = wf_lambda(ek, idx, None)
        S["Kalman TVP-AR(1) + EWMA"] = np.where(np.isnan(l_), np.nan, ewma_path(ek, np.nan_to_num(l_, nan=0.94)) + kf["mu"] ** 2)
        lam["Kalman TVP-AR(1) + EWMA"] = l_
    if "GARCH(1,1)" in models:
        S["GARCH(1,1)"] = garch_path(e, idx)
    if "HAR-RV" in models or "MIDAS-RV" in models:
        import midas as MD
        f = MD.daily_vol_forecasts(r, h=1).shift(1)                    # t−1에 만든 t의 예측
        for c in ["HAR-RV", "MIDAS-RV"]:
            if c in models:
                S[c] = f[c].reindex(idx).values
    S = pd.DataFrame(S, index=idx)[[m for m in models if m in S]]
    return dict(r=r, S=S, lam=pd.DataFrame(lam, index=idx), hmm=hm, kalman=kf, LL=LL)


# ============================================================================
# 예측 검증
# ============================================================================
def qlike(r: pd.Series, S: pd.DataFrame) -> pd.DataFrame:
    """QLIKE 손실 = log σ̂² + r²/σ̂² (Patton 2011: 잡음 섞인 실현 대리변수에도 순위가 일관된 손실)."""
    s = S.clip(lower=1e-8)
    return np.log(s).add((r ** 2).values[:, None] / s.values, axis=0) if len(S) else S


def nw_se(d: np.ndarray, L: int) -> float:
    d = d - d.mean(); T = len(d); s = d @ d / T
    for l in range(1, L + 1):
        s += 2 * (1 - l / (L + 1)) * (d[l:] @ d[:-l]) / T
    return float(np.sqrt(max(s, 1e-14) / T))


def dm_table(Lo: pd.DataFrame, base: str, L: int = 5) -> pd.DataFrame:
    """Diebold-Mariano: dₜ = 손실(모형) − 손실(기준). t < −1.96 → 모형이 기준보다 유의하게 우수."""
    rows = []
    for m in Lo.columns:
        if m == base:
            continue
        d = (Lo[m] - Lo[base]).dropna().values
        t = d.mean() / nw_se(d, L)
        rows.append({"모형": m, "평균 손실차 (모형−기준)": d.mean(), "DM t": t, "p값 (양측)": 2 * stats.norm.sf(abs(t)),
                     "모형이 나은 날 비율": float((d < 0).mean())})
    return pd.DataFrame(rows).set_index("모형")


def mcs(Lo: pd.DataFrame, B: int = 500, block: int = 10, seed: int = 0) -> pd.Series:
    """Model Confidence Set (Hansen·Lunde·Nason 2011, T_max 통계량, 이동 블록 부트스트랩). 반환: 모형별 MCS p값
    (p ≥ 0.10 → 90% 신뢰집합에 포함 = 최선과 통계적으로 구별되지 않음)."""
    X = Lo.dropna().values; T, M = X.shape
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    idx = (rng.integers(0, T - block + 1, size=(B, nb))[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :T]
    Xb_means = np.stack([X[ii].mean(0) for ii in idx])               # (B, M) 부트스트랩 평균 손실
    xbar = X.mean(0)
    alive = list(range(M)); pv = {}; pmax = 0.0
    while len(alive) > 1:
        a = np.array(alive)
        dbar = xbar[a] - xbar[a].mean()                                # 모형 i의 평균 대비 초과 손실
        db = Xb_means[:, a] - Xb_means[:, a].mean(1, keepdims=True)
        se = np.sqrt(((db - dbar) ** 2).mean(0)) + 1e-12
        t = dbar / se
        tb = ((db - dbar) / se).max(1)
        p = float((tb >= t.max()).mean())
        pmax = max(pmax, p)
        worst = a[int(t.argmax())]
        pv[Lo.columns[worst]] = pmax
        alive.remove(worst)
    pv[Lo.columns[alive[0]]] = 1.0
    return pd.Series(pv).reindex(Lo.columns)


def lr_common_lambda(LLs: dict[str, np.ndarray], mask: dict[str, np.ndarray] | None = None) -> dict:
    """우도비 검정 H₀: 모든 품목이 같은 λ. LLs: 품목별 λ그리드 −2ℓ 기여 (G,T). LR = Σ min_k − min Σ_k ~ χ²(K−1).
    (각 품목 표본이 달라도 됨: 품목별 로그우도가 독립 가법 → 공통 λ는 합계 최소)"""
    tot = {k: (v[:, mask[k]] if mask else v[:, 1:]).sum(1) for k, v in LLs.items()}
    own = sum(t.min() for t in tot.values()); pooled = sum(tot.values())
    LR = float(pooled.min() - own); df = len(tot) - 1
    return dict(LR=LR, df=df, p=float(stats.chi2.sf(LR, df)), common_lambda=float(LAM_GRID[pooled.argmin()]),
                own={k: float(LAM_GRID[t.argmin()]) for k, t in tot.items()})


def lr_time_lambda(LL: np.ndarray, idx: pd.DatetimeIndex, block_years: int = 3) -> dict:
    """우도비 검정 H₀: 같은 품목의 λ가 기간(block_years년 블록) 간 동일. 블록별 λ 자유 vs 공통 λ, df = 블록 수 − 1.
    (재귀 상태는 전체 경로에서 이어 받는 근사 — 블록 경계의 영향은 작음). 블록별 λ와 95% 프로파일 신뢰구간도 반환."""
    y = idx.year.values
    b = (y - y.min()) // block_years
    rows, own = [], 0.0
    full = LL[:, 1:].sum(1)
    for k in np.unique(b):
        m = (b == k); m[0] = False
        if m.sum() < 250:
            continue
        s = LL[:, m].sum(1); own += s.min()
        lam, lo, hi = lambda_ci(s)
        rows.append({"기간": f"{y[m].min()}–{y[m].max()}", "λ": lam, "95% 하한": lo, "95% 상한": hi,
                     "반감기 (거래일)": float(hl_days(lam)), "관측": int(m.sum())})
    sel = np.isin(b, [k for k in np.unique(b) if ((b == k).sum() >= 250)]); sel[0] = False
    pooled = LL[:, sel].sum(1).min()
    LR = float(pooled - own); df = max(len(rows) - 1, 1)
    return dict(LR=LR, df=df, p=float(stats.chi2.sf(LR, df)), table=pd.DataFrame(rows))


def sign_bias(z: np.ndarray) -> dict:
    """Engle-Ng (1993) 부호 편향 검정: z²ₜ = c + b₁S⁻ₜ₋₁ + b₂S⁻ₜ₋₁zₜ₋₁ + b₃S⁺ₜ₋₁zₜ₋₁. 유의하면 대칭 모형(EWMA 등)이 놓친 비대칭."""
    z = z[np.isfinite(z)]
    y = z[1:] ** 2; zl = z[:-1]; Sm = (zl < 0).astype(float)
    X = np.column_stack([np.ones(len(y)), Sm, Sm * zl, (1 - Sm) * zl])
    b, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ b
    s2 = e @ e / (len(y) - 4); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    R2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    LM = len(y) * R2
    nm = ["부호 편향 (S⁻)", "음(−) 크기 편향", "양(+) 크기 편향"]
    return dict(table=pd.DataFrame({"계수": b[1:], "t값": b[1:] / se[1:]}, index=nm), LM=float(LM), p=float(stats.chi2.sf(LM, 3)))


def ljung_box_sq(z: np.ndarray, lags: int = 10) -> tuple[float, float]:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    z = z[np.isfinite(z)]
    o = acorr_ljungbox(z ** 2, lags=[lags], return_df=True)
    return float(o["lb_stat"].iloc[0]), float(o["lb_pvalue"].iloc[0])


def mahalanobis(R: pd.DataFrame, S: pd.DataFrame, win: int = 252) -> pd.Series:
    """다변량 적합 진단 d²ₜ = rₜ′Hₜ⁻¹rₜ, Hₜ = DₜCₜDₜ (D = 각 품목 모형 변동성, C = 직전 win일 표준화 잔차 상관).
    모형이 맞으면 E[d²] = K, χ²(K) 95% 경계 초과 비율 ≈ 5%."""
    Z = (R / np.sqrt(S)).values; Rv = R.values; K = R.shape[1]
    out = np.full(len(R), np.nan)
    for t in range(win, len(R)):
        Zw = Z[t - win:t]; Zw = Zw[np.isfinite(Zw).all(1)]
        if len(Zw) < win // 2 or not np.isfinite(Z[t]).all():
            continue
        C = np.corrcoef(Zw, rowvar=False)
        out[t] = Z[t] @ np.linalg.solve(C + 1e-8 * np.eye(K), Z[t])
    return pd.Series(out, index=R.index)


# ============================================================================
# VaR · ES
# ============================================================================
def _t_unit(nu):
    return np.sqrt((nu - 2) / nu)


def _quant_es(u_past: np.ndarray, alpha: float, method: str) -> tuple[float, float]:
    """표준화 손실 u의 과거 표본 → α 분위수(VaR 배수)와 ES 배수."""
    u = u_past[np.isfinite(u_past)]
    if method == "정규":
        q = stats.norm.ppf(alpha); return q, stats.norm.pdf(q) / (1 - alpha)
    if method == "Student-t":
        nu = float(np.clip(stats.t.fit(u, floc=0)[0], 2.2, 50)) if len(u) > 200 else 8.0
        c = _t_unit(nu); q = stats.t.ppf(alpha, nu)
        es = stats.t.pdf(q, nu) / (1 - alpha) * (nu + q ** 2) / (nu - 1)
        return q * c, es * c
    if method == "FHS (필터링 역사적)":
        q = float(np.quantile(u, alpha)); return q, float(u[u >= q].mean())
    if method == "EVT-GPD (조건부 POT)":
        u0 = float(np.quantile(u, 0.90)); exc = u[u > u0] - u0
        if len(exc) < 30:
            q = float(np.quantile(u, alpha)); return q, float(u[u >= q].mean())
        xi, _, beta = stats.genpareto.fit(exc, floc=0)
        xi = float(np.clip(xi, -0.4, 0.8)); pu = len(exc) / len(u)
        q = u0 + beta / xi * (((1 - alpha) / pu) ** (-xi) - 1) if abs(xi) > 1e-6 else u0 - beta * np.log((1 - alpha) / pu)
        return float(q), float((q + beta - xi * u0) / (1 - xi))
    raise ValueError(method)


def var_forecasts(r: pd.Series, s2: pd.Series, alpha: float = 0.99, side: str = "long", h: int = 1,
                  methods: list[str] | None = None, start=None, hs_window: int = 500) -> dict:
    """h일 VaR·ES (손실 %, 양수). 결정일 t의 정보(종가까지)로 t+1..t+h 손실을 예측.
    σ: s2[t+1] (t 종가까지 반영된 다음 날 분산 예측)·√h. 표준화 손실 분포의 모수는 매년 1월 그 이전에 '실현이 끝난' 표본으로 재추정.
    반환: table(날짜 × [실현손실, 방법별 VaR, ES]), hits."""
    methods = methods or VAR_METHODS
    sgn = 1.0 if side == "long" else -1.0
    r = r.dropna(); s2 = s2.reindex(r.index)
    loss1 = -sgn * r
    lossh = loss1[::-1].rolling(h).sum()[::-1]                          # t..t+h−1 누적 손실
    sig_next = np.sqrt(s2.shift(-1))                                    # t 종가 이후 첫날의 σ 예측 (t까지 정보)
    dec = r.index[:-h] if h > 0 else r.index                            # 결정일: t, 실현: t+1..t+h
    L_real = lossh.shift(-1).reindex(dec)                               # t+1..t+h 손실
    sig_h = (sig_next * np.sqrt(h)).reindex(dec)
    u_all = (L_real / sig_h)                                            # 표준화 h일 손실 (실현 시점 = t+h)
    real_end = pd.Series(r.index[np.arange(len(dec)) + h], index=dec)   # 그 표준화 손실이 확정되는 날
    y = dec.year.values
    out = pd.DataFrame(index=dec); out["실현손실"] = L_real.values; out["σ_h"] = sig_h.values
    for m in methods:
        out[f"VaR {m}"] = np.nan; out[f"ES {m}"] = np.nan
    for yr in np.unique(y):
        cut = pd.Timestamp(f"{yr}-01-01")
        past = u_all[(real_end < cut).values].values
        past = past[np.isfinite(past)]
        if len(past) < 500:
            continue
        sel = y == yr
        for m in methods:
            if m.startswith("역사적"):
                continue
            q, es = _quant_es(past, alpha, m)
            out.loc[sel, f"VaR {m}"] = q * out.loc[sel, "σ_h"]; out.loc[sel, f"ES {m}"] = es * out.loc[sel, "σ_h"]
    if any(m.startswith("역사적") for m in methods):
        m = [m for m in methods if m.startswith("역사적")][0]
        Lh_past = lossh.reindex(r.index)                                 # 날짜 t의 값 = t..t+h−1 손실 → t+h−1에 확정
        hist = Lh_past.shift(h - 1)                                     # t에 확정된 h일 손실 (t−h+1..t)
        out[f"VaR {m}"] = hist.rolling(hs_window, min_periods=hs_window // 2).quantile(alpha).reindex(dec).values
        tail = lambda a: a[a >= np.quantile(a, alpha)].mean()
        out[f"ES {m}"] = hist.rolling(hs_window, min_periods=hs_window // 2).apply(tail, raw=True).reindex(dec).values
    if start is not None:
        out = out[out.index >= pd.Timestamp(start)]
    out = out.dropna(subset=["실현손실"])
    if h > 1:
        out = out.iloc[::h]                                              # 비중첩 평가
    return dict(table=out, alpha=alpha, h=h, side=side, methods=methods)


def kupiec(hits: np.ndarray, alpha: float) -> tuple[float, float]:
    n = len(hits); x = int(hits.sum()); p = 1 - alpha
    if n == 0:
        return np.nan, np.nan
    ph = x / n
    l0 = (n - x) * np.log(1 - p) + x * np.log(p)
    l1 = ((n - x) * np.log(1 - ph) if ph < 1 else 0) + (x * np.log(ph) if ph > 0 else 0)
    LR = -2 * (l0 - l1); return float(LR), float(stats.chi2.sf(LR, 1))


def christoffersen(hits: np.ndarray) -> tuple[float, float]:
    h = hits.astype(int); a, b = h[:-1], h[1:]
    n00, n01 = ((a == 0) & (b == 0)).sum(), ((a == 0) & (b == 1)).sum()
    n10, n11 = ((a == 1) & (b == 0)).sum(), ((a == 1) & (b == 1)).sum()
    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1); p = (n01 + n11) / max(len(a), 1)
    ll = lambda k, q: k * np.log(q) if q > 0 else 0.0
    l0 = ll(n00 + n10, 1 - p) + ll(n01 + n11, p)
    l1 = ll(n00, 1 - p01) + ll(n01, p01) + ll(n10, 1 - p11) + ll(n11, p11)
    LR = -2 * (l0 - l1); return float(LR), float(stats.chi2.sf(LR, 1))


def es_test(L: np.ndarray, es: np.ndarray, var: np.ndarray, sig: np.ndarray) -> tuple[float, float]:
    """McNeil-Frey (2000): 초과일의 (손실 − ES)/σ 평균 = 0 (단측: 양수면 ES 과소추정). 반환 (t, p)."""
    m = L > var
    if m.sum() < 5:
        return np.nan, np.nan
    x = (L[m] - es[m]) / sig[m]
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)) + 1e-12)
    return float(t), float(stats.t.sf(t, len(x) - 1))


def backtest_table(V: dict) -> pd.DataFrame:
    T, a = V["table"], V["alpha"]
    rows = []
    for m in V["methods"]:
        d = T.dropna(subset=[f"VaR {m}"])
        if not len(d):
            continue
        hits = (d["실현손실"] > d[f"VaR {m}"]).values
        LRuc, puc = kupiec(hits, a); LRind, pind = christoffersen(hits)
        LRcc = LRuc + LRind; pcc = float(stats.chi2.sf(LRcc, 2))
        sig = d["σ_h"].values if not m.startswith("역사적") else np.full(len(d), d["실현손실"].std())
        tES, pES = es_test(d["실현손실"].values, d[f"ES {m}"].values, d[f"VaR {m}"].values, sig)
        last = hits[-250:]
        n250 = int(last.sum()); zone = ("🟢 녹색" if n250 <= 4 else "🟡 황색" if n250 <= 9 else "🔴 적색") if abs(a - 0.99) < 1e-9 else "–"
        rows.append({"방법": m, "평가일수": len(d), "초과 횟수": int(hits.sum()), "기대 초과": len(d) * (1 - a),
                     "초과율": hits.mean(), "Kupiec p (적중률)": puc, "Christoffersen p (독립성)": pind, "조건부 적중 p": pcc,
                     "ES 검정 p (과소추정)": pES, "평균 VaR (%)": d[f"VaR {m}"].mean(), "평균 ES (%)": d[f"ES {m}"].mean(),
                     "최근 250일 초과": n250, "바젤 신호등 (99%)": zone})
    return pd.DataFrame(rows).set_index("방법")


# ============================================================================
# 꼬리 위험
# ============================================================================
def hill(x: np.ndarray, frac: float = 0.05) -> float:
    """Hill 꼬리 지수 α (작을수록 두꺼운 꼬리; α<4면 첨도 무한, 정규분포는 ∞)."""
    x = np.sort(x[x > 0])[::-1]; k = max(int(len(x) * frac), 10)
    if len(x) <= k:
        return np.nan
    return float(1 / np.mean(np.log(x[:k] / x[k])))


def tail_stats(r: pd.Series, z: np.ndarray | None = None) -> dict:
    x = r.dropna().values
    out = {"왜도": float(stats.skew(x)), "초과첨도": float(stats.kurtosis(x)), "Jarque-Bera p": float(stats.jarque_bera(x).pvalue),
           "하락 꼬리 Hill α": hill(-x), "상승 꼬리 Hill α": hill(x),
           "1일 최악 (%)": float(x.min()), "1일 최고 (%)": float(x.max()),
           "1% 하위 분위 (%)": float(np.quantile(x, 0.01)), "99% 상위 분위 (%)": float(np.quantile(x, 0.99))}
    if z is not None:
        zz = z[np.isfinite(z)]
        out.update({"표준화 잔차 초과첨도": float(stats.kurtosis(zz)), "표준화 잔차 |z|>3 비율": float((np.abs(zz) > 3).mean()),
                    "정규 기대 |z|>3 비율": 0.0027})
    return out


def tail_dependence(R: pd.DataFrame, q: float = 0.05, upper: bool = False) -> pd.DataFrame:
    """경험적 꼬리 의존 λ(q) = P(Uⱼ ≤ q | Uᵢ ≤ q) (순위 기반). 독립이면 q, 완전 의존이면 1."""
    U = R.rank(pct=True)
    if upper:
        U = 1 - U + 1 / len(U)
    E = (U <= q).astype(float)
    n = E.sum()
    M = (E.T @ E) / n.values[:, None]
    return pd.DataFrame(M.values, index=R.columns, columns=R.columns)


def stress_table(p: pd.Series, horizons=(1, 5, 20), n: int = 5) -> pd.DataFrame:
    """가장 나빴던 h일 구간 (겹치지 않게 n개). 가격 수준 기준 (롤 보정 가격 권장)."""
    lp = np.log(p.dropna()); rows = []
    for h in horizons:
        ch = (lp.shift(-h) - lp).dropna() * 100
        for side, srt in [("하락", ch.sort_values()), ("상승", ch.sort_values(ascending=False))]:
            taken = []
            for d, v in srt.items():
                if all(abs((d - t).days) > h * 1.6 for t in taken):
                    taken.append(d); rows.append({"기간 (영업일)": h, "방향": side, "시작": d.date(), "변화 (%)": float(np.expm1(v / 100) * 100)})
                if len(taken) >= n:
                    break
    return pd.DataFrame(rows)


def drawdown(p: pd.Series) -> dict:
    p = p.dropna(); dd = p / p.cummax() - 1
    t = dd.idxmin(); peak = p.loc[:t].idxmax()
    rec = p.loc[t:][p.loc[t:] >= p.loc[peak]]
    du = p / p.cummin() - 1; tu = du.idxmax(); trough = p.loc[:tu].idxmin()
    return {"최대 낙폭 (%)": float(dd.min() * 100), "고점": peak.date(), "저점": t.date(),
            "회복": rec.index[0].date() if len(rec) else "미회복", "현재 낙폭 (%)": float(dd.iloc[-1] * 100),
            "최대 상승폭 (%) (구매자 위험)": float(du.max() * 100), "상승 저점": trough.date(), "상승 고점": tu.date(),
            "현재 저점 대비 (%)": float(du.iloc[-1] * 100), "series": dd * 100, "up_series": du * 100}


# ============================================================================
# 구매·보유 물량 포트폴리오 위험
# ============================================================================
def position_risk(R: pd.DataFrame, S: pd.DataFrame, expo: pd.Series, alpha: float = 0.99, h: int = 20, win: int = 500) -> dict:
    """expo: 품목별 달러 노출 (+ = 보유 재고(가격 하락 손실), − = 매입 예정 물량(가격 상승 손실)).
    정규 VaR(현재 σ × 직전 win일 표준화 잔차 상관) + 다변량 FHS(과거 표준화 잔차 벡터 × 현재 σ, h일 = 연속 h일 블록 합)
    + 오일러 기여도 + 과거 위기 구간 시나리오."""
    ids = list(expo.index)
    Z = (R[ids] / np.sqrt(S[ids])).dropna()
    s_now = np.sqrt(S[ids].iloc[-1].values) / 100                       # 다음 날 σ (비율)
    C = np.corrcoef(Z.values[-win:], rowvar=False)
    w = expo.values.astype(float)
    Hd = np.outer(s_now, s_now) * C * h
    sig_p = float(np.sqrt(w @ Hd @ w))
    q = stats.norm.ppf(alpha)
    contrib = w * (Hd @ w) / sig_p * q
    Zv = Z.values[-max(win, 2 * h):]
    sims = np.array([Zv[k:k + h].sum(0) for k in range(len(Zv) - h + 1)]) * s_now      # h일 수익률(비율) 시나리오
    pnl = sims @ w
    loss = -pnl
    vf = float(np.quantile(loss, alpha)); ef = float(loss[loss >= vf].mean())
    tbl = pd.DataFrame({"노출 ($)": w, "일 σ (%)": s_now * 100, f"{h}일 σ ($)": np.abs(w) * s_now * np.sqrt(h),
                        "VaR 기여 ($, 정규)": contrib, "기여 비중": contrib / contrib.sum()}, index=ids)
    return dict(table=tbl, var_normal=sig_p * q, es_normal=sig_p * stats.norm.pdf(q) / (1 - alpha), var_fhs=vf, es_fhs=ef,
                sigma=sig_p, corr=pd.DataFrame(C, index=ids, columns=ids), sims=loss,
                undiversified=float((np.abs(w) * s_now * np.sqrt(h)).sum() * q))


def crisis_scenarios(L: pd.DataFrame, expo: pd.Series, crises: dict) -> pd.DataFrame:
    """과거 위기 구간의 실제 가격 변화를 현재 노출에 적용한 손익 ($). L: 가격 수준(롤 보정)."""
    rows = []
    for nm, (a, b) in crises.items():
        seg = L.loc[a:b, expo.index].dropna(how="all")
        if len(seg) < 5:
            continue
        ch = seg.ffill().iloc[-1] / seg.bfill().iloc[0] - 1
        pnl = float((ch.fillna(0) * expo).sum())
        rows.append({"시나리오": nm, "기간": f"{a} ~ {b}", "손익 ($)": pnl, **{f"{c} 변화": float(ch.get(c, np.nan)) for c in expo.index}})
    return pd.DataFrame(rows).set_index("시나리오") if rows else pd.DataFrame()


# ============================================================================
# h일 가격 범위 보정 (구매 타이밍 '4주 90% 범위')
# ============================================================================
def hday_band(r: pd.Series, sig_d: pd.Series, h: int = 20, lo: float = 0.05, hi: float = 0.95, asof=None,
              min_n: int = 500) -> tuple[float, float]:
    """표준화 h일 수익률 uₜ = Σ rₜ₊₁..ₜ₊ₕ / (σₜ√h)의 경험적 분위수 (asof 시점까지 실현이 끝난 표본만).
    σₜ = t 종가까지의 정보로 본 다음 날 일간 σ (%). 표본 부족 시 정규 (±1.645)."""
    r = r.dropna(); s = sig_d.reindex(r.index)
    fwd = r[::-1].rolling(h).sum()[::-1].shift(-1)                     # t+1..t+h
    u = fwd / (s * np.sqrt(h))
    end = pd.Series(r.index, index=r.index).shift(-h)                  # 실현 확정일
    asof = r.index[-1] if asof is None else pd.Timestamp(asof)
    x = u[(end <= asof).values].dropna().values
    if len(x) < min_n:
        return stats.norm.ppf(lo), stats.norm.ppf(hi)
    return float(np.quantile(x, lo)), float(np.quantile(x, hi))


def band_coverage(r: pd.Series, sig_d: pd.Series, h: int = 20, start="2008-01-01", step: int = 5) -> dict:
    """주 1회 결정일마다 (a) 정규 ±1.645σ√h, (b) 매년 1월 재추정한 경험적 분위 범위의 실제 적중률."""
    r = r.dropna(); s = sig_d.reindex(r.index)
    fwd = r[::-1].rolling(h).sum()[::-1].shift(-1)
    u = (fwd / (s * np.sqrt(h)))
    d = u[(u.index >= pd.Timestamp(start))].dropna().iloc[::step]
    q = {}
    for yr in np.unique(d.index.year):
        q[yr] = hday_band(r, sig_d, h, asof=pd.Timestamp(f"{yr}-01-01") - pd.Timedelta(days=1))
    lo = np.array([q[y][0] for y in d.index.year]); hi = np.array([q[y][1] for y in d.index.year])
    return {"정규 적중률": float((d.abs() <= 1.645).mean()), "경험적 적중률": float(((d >= lo) & (d <= hi)).mean()),
            "하단 이탈률 (경험적)": float((d < lo).mean()), "상단 이탈률 (경험적)": float((d > hi).mean()),
            "평균 하단 배수": float(lo.mean()), "평균 상단 배수": float(hi.mean()), "n": len(d)}


def next_var(o: dict, model: str) -> float:
    """마지막 거래일 종가까지 반영한 '다음 날' 분산 예측 (EWMA 계열은 한 단계 갱신, 그 외는 마지막 예측)."""
    r = o["r"]; S = o["S"][model]
    lam = o["lam"][model].dropna() if model in o["lam"] else None
    s2 = float(S.dropna().iloc[-1])
    if lam is not None and len(lam):
        e_last = float(o["kalman"]["innov"][-1]) if model.startswith("Kalman") and o.get("kalman") else float(r.iloc[-1])
        return float(lam.iloc[-1]) * s2 + (1 - float(lam.iloc[-1])) * e_last ** 2
    return s2


def var_now(V: dict, r: pd.Series, s2_next: float, hs_window: int = 500) -> pd.DataFrame:
    """오늘 기준 h일 VaR·ES (손실 %, 양수): 과거 실현 표준화 손실 전체(실현 완료분)로 분위 추정 × 현재 σ√h."""
    T, a, h = V["table"], V["alpha"], V["h"]
    u = (T["실현손실"] / T["σ_h"]).values
    sig_h = np.sqrt(s2_next * h)
    rows = []
    for m in V["methods"]:
        if m.startswith("역사적"):
            sgn = 1.0 if V["side"] == "long" else -1.0
            Lh = (-sgn * r).rolling(h).sum().dropna().values[-hs_window:]
            q = float(np.quantile(Lh, a)); es = float(Lh[Lh >= q].mean())
        else:
            qq, ee = _quant_es(u, a, m); q, es = qq * sig_h, ee * sig_h
        rows.append({"방법": m, "VaR (%)": q, "ES (%)": es})
    return pd.DataFrame(rows).set_index("방법")
