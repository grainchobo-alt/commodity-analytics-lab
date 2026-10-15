"""D. 머신러닝 확장 — 부스팅 + 설명가능성 · 커널/가우시안 과정 · 분포가정 없는 예측구간

LightGBM          : 히스토그램 그래디언트 부스팅 (Ke et al. 2017). 얕은 나무(잎 7개)·강한 L2·행/열 표본추출로 과적합 억제,
                    반복 횟수는 학습구간 끝 20%(시간순, h>1이면 gap)로 조기종료 후 전체 학습구간 재적합.
                    SHAP = LightGBM 내장 TreeSHAP(pred_contrib, Lundberg et al. 2020) — 예측 = 기준값 + Σ 변수별 기여 (정확한 가법 분해)
Gaussian Process  : 커널 = 신호분산·RBF(비선형) + 선형(DotProduct) + 백색잡음, 초모수는 주변우도 최대화 (Rasmussen-Williams 2006).
                    예측평균과 함께 예측표준편차를 냄. O(n³)이라 학습구간 최근 gp_nmax개(기본 1500)만 사용.
                    변수 ≤ 15개면 ARD(변수별 길이척도 → 1/ℓ = 관련도)
Kernel Ridge      : RBF 커널 리지 — 비선형 평활. α·γ는 시계열 CV 격자 탐색
Conformal 예측구간 : 어떤 모델이든 분포가정 없이 (1−α) 구간.
    분할(split) : 학습구간 끝 25%를 보정용으로 떼어 |잔차| 분위수 (Vovk 2005, Lei et al. 2018)
    적응형(ACI) : 테스트 구간에서 실현된 미포함 여부로 α_t를 온라인 갱신 (Gibbs & Candès 2021) — 변동성 국면 변화에 대응.
                  h기 목표는 h기 뒤에야 실현 → 시점 t의 구간에는 t−h까지 실현된 오차만 사용 (미래정보 없음)
    변동성 정규화 : 점수 = |잔차| / σ̂_t (σ̂ = 실현된 과거 목표의 EWMA 표준편차) → 고변동기엔 넓게, 저변동기엔 좁게
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ML_MODELS = ["LightGBM", "Gaussian Process", "Kernel Ridge"]


def _lgb_params(n: int, hp: dict) -> dict:
    return dict(objective="regression", learning_rate=hp.get("lgb_lr", 0.03), num_leaves=hp.get("lgb_leaves", 7),
                max_depth=3, min_child_samples=max(20, n // 50), feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                lambda_l2=hp.get("lgb_l2", 5.0), verbose=-1, seed=0, deterministic=True, force_row_wise=True,
                num_threads=1 if hp.get("_inner_jobs") == 1 else 4)


def lgb_fit(Xtr, ytr, gap: int, hp: dict):
    """조기종료로 반복 수 선택 → 전체 학습구간 재적합. return (booster, best_iter)."""
    import lightgbm as lgb
    n = len(ytr); params = _lgb_params(n, hp)
    n_max = hp.get("lgb_rounds", 600)
    nv = max(30, int(n * 0.2)); end = n - nv - gap
    best = min(100, n_max)
    if end >= 100:
        dtr = lgb.Dataset(Xtr[:end], ytr[:end]); dva = lgb.Dataset(Xtr[end + gap:], ytr[end + gap:], reference=dtr)
        b0 = lgb.train(params, dtr, n_max, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        best = max(20, b0.best_iteration or n_max)
    booster = lgb.train(params, lgb.Dataset(Xtr, ytr), best)
    return booster, best


def _gp_kernel(p: int, ard: bool, var: float):
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C, DotProduct
    ls = np.full(p, np.sqrt(p)) if ard else np.sqrt(p)
    return (C(0.1, (1e-4, 1e1)) * RBF(ls, (1e-1, 1e3)) + C(0.01, (1e-5, 1e1)) * DotProduct(sigma_0=0.0, sigma_0_bounds="fixed")
            + WhiteKernel(0.9, (1e-3, 1e1)))


def run_ml(name: str, D, ann: int, **hp):
    import analytics as A
    Xtr, ytr, Xte, yte = D.X_tr, D.y_tr, D.X_te, D.y_te
    gap = max(0, D.horizon - 1)
    extra = {}
    if name == "LightGBM":
        m, best = lgb_fit(Xtr, ytr, gap, hp)
        p_tr, p_te = m.predict(Xtr), m.predict(Xte)
        extra["n_trees"] = best
        if not hp.get("_fast"):
            sh = m.predict(Xte, pred_contrib=True)                       # (n_te, p+1): 마지막 열 = 기준값(학습 평균 예측)
            S = pd.DataFrame(sh[:, :-1], index=D.idx_te, columns=D.names)
            extra["shap"] = S
            extra["shap_base"] = float(sh[0, -1])
            extra["importance"] = S.abs().mean().sort_values()            # 평균 |SHAP| (예측 단위)
            sh_tr = m.predict(Xtr, pred_contrib=True)[:, :-1]
            extra["shap_dir"] = pd.Series([np.corrcoef(Xtr[:, j], sh_tr[:, j])[0, 1] if Xtr[:, j].std() > 0 and sh_tr[:, j].std() > 0 else 0.0
                                           for j in range(Xtr.shape[1])], index=D.names)   # 변수↑ → 기여↑ 이면 +
            extra["gain"] = pd.Series(m.feature_importance("gain"), index=D.names).sort_values()
    elif name == "Gaussian Process":
        from sklearn.gaussian_process import GaussianProcessRegressor
        nmax = hp.get("gp_nmax", 1500)
        Xg, yg = Xtr[-nmax:], ytr[-nmax:]
        p = Xtr.shape[1]
        ard = p <= 15 and not hp.get("_fast")
        sy = yg.std() or 1.0
        gp = GaussianProcessRegressor(_gp_kernel(p, ard, 1.0), normalize_y=False,
                                      n_restarts_optimizer=0 if hp.get("_fast") else 2, random_state=0)
        gp.fit(Xg, (yg - yg.mean()) / sy)
        mu_tr = gp.predict(Xtr)
        mu_te, sd_te = gp.predict(Xte, return_std=True)
        p_tr, p_te = mu_tr * sy + yg.mean(), mu_te * sy + yg.mean()
        k = gp.kernel_
        extra["kernel"] = str(k)
        extra["n_used"] = len(yg)
        extra["pred_std"] = pd.Series(sd_te * sy, index=D.idx_te)
        noise = k.k2.noise_level; sig = k.k1.k1.k1.constant_value; lin = k.k1.k2.k1.constant_value
        extra["var_share"] = pd.Series({"비선형(RBF)": sig, "선형": lin * p, "잡음": noise}) / (sig + lin * p + noise)
        if ard:
            ls = np.atleast_1d(k.k1.k1.k2.length_scale)
            extra["importance"] = pd.Series(1 / ls, index=D.names).sort_values()   # 관련도 = 1/길이척도
    elif name == "Kernel Ridge":
        from sklearn.kernel_ridge import KernelRidge
        p = Xtr.shape[1]
        alphas = np.logspace(-1, 3, 9); gammas = np.array([0.03, 0.1, 0.3, 1.0]) / p
        err = np.zeros((len(alphas), len(gammas)))
        for tr, va in A._tscv().split(Xtr):
            for j, g in enumerate(gammas):
                for i, a in enumerate(alphas):
                    m = KernelRidge(alpha=a, kernel="rbf", gamma=g).fit(Xtr[tr], ytr[tr])
                    err[i, j] += np.mean((ytr[va] - m.predict(Xtr[va])) ** 2)
        i, j = np.unravel_index(err.argmin(), err.shape)
        m = KernelRidge(alpha=alphas[i], kernel="rbf", gamma=gammas[j]).fit(Xtr, ytr)
        p_tr, p_te = m.predict(Xtr), m.predict(Xte)
        extra.update(alpha=alphas[i], gamma=gammas[j] * p,
                     cv=pd.DataFrame(err / 5, index=[f"α={a:g}" for a in alphas], columns=[f"γ·p={g * p:g}" for g in gammas]))
    else:
        raise ValueError(name)
    return dict(coef=None, pred_tr=p_tr, pred_te=p_te, metrics=A._metrics(ytr, p_tr, yte, p_te, ann, D.horizon, D), extra=extra)


# ============================================================================
# Conformal 예측구간
# ============================================================================
def _ewm_scale(y_known: np.ndarray, halflife: int = 26) -> float:
    s = pd.Series(y_known).ewm(halflife=halflife, min_periods=10).std().iloc[-1]
    return float(s) if np.isfinite(s) and s > 0 else float(np.std(y_known) or 1.0)


def conformal(D, model: str, ann: int = 52, alpha: float = 0.1, cal_frac: float = 0.25, gamma: float = 0.01,
              normalize: bool = True, window: int = 500, **hp) -> dict:
    """D의 학습구간을 [적합 | (gap) | 보정]으로 나눠 보정 점수를 얻고, 테스트 구간에 분할·적응형 conformal 구간.
    점 예측은 전체 학습구간으로 적합한 모델 (run_model과 동일)."""
    import analytics as A
    h = D.horizon; gap = h - 1
    n = len(D.y_tr); n_cal = max(30, int(n * cal_frac)); n_fit = n - n_cal - gap
    if n_fit < 60:
        raise ValueError("학습 표본이 너무 짧음 (보정용 25%를 떼면 60개 미만)")
    sl_c = slice(n_fit + gap, n)
    Dc = A.Design(D.X_tr[:n_fit], D.y_tr[:n_fit], D.X_tr[sl_c], D.y_tr[sl_c], D.idx_tr[:n_fit], D.idx_tr[sl_c],
                  D.names, D.mu, D.sd, D.y_mu, None, h, D.mode, D.cost, D.target)
    p_cal = A.run_model(model, Dc, ann, **dict(hp, _fast=True))["pred_te"]
    o = A.run_model(model, D, ann, **dict(hp, _fast=True))
    p_te = o["pred_te"]
    y_all = np.concatenate([D.y_tr, D.y_te])                             # 시간순 (학습 → 테스트)
    # 정규화 척도: 시점 t에서 이미 실현된 목표(인덱스 ≤ t−h)만으로 EWMA 표준편차
    def scale_at(pos):
        known = y_all[:max(10, pos - h + 1)]
        return _ewm_scale(known) if normalize else 1.0
    cal_pos = np.arange(n_fit + gap, n)
    s_cal = np.array([scale_at(t) for t in cal_pos])
    scores = list(np.abs(D.y_tr[sl_c] - p_cal) / s_cal)
    qlev = min(1.0, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
    q_split = float(np.quantile(scores, qlev))
    T = len(p_te)
    s_te = np.array([scale_at(n + t) for t in range(T)])
    lo_s, hi_s = p_te - q_split * s_te, p_te + q_split * s_te
    # 적응형 conformal (ACI): α_t 온라인 갱신, 점수 창은 최근 window개
    a_t = alpha; lo_a = np.empty(T); hi_a = np.empty(T); alphas = np.empty(T)
    for t in range(T):
        j = t - h                                                        # 시점 t에 실현이 확인된 가장 최근 테스트 관측
        if j >= 0:
            err = float((D.y_te[j] < lo_a[j]) or (D.y_te[j] > hi_a[j]))
            a_t = a_t + gamma * (alpha - err)
            scores.append(abs(D.y_te[j] - p_te[j]) / s_te[j])
        sc = np.asarray(scores[-window:])
        a_eff = min(max(a_t, 1e-3), 0.999)
        q = float(np.quantile(sc, min(1.0, np.ceil((len(sc) + 1) * (1 - a_eff)) / len(sc))))
        lo_a[t], hi_a[t] = p_te[t] - q * s_te[t], p_te[t] + q * s_te[t]
        alphas[t] = a_t
    y = D.y_te; idx = D.idx_te
    cov = lambda lo, hi: float(np.mean((y >= lo) & (y <= hi)))
    res = dict(pred=pd.Series(p_te, index=idx), y=pd.Series(y, index=idx),
               split=pd.DataFrame({"lo": lo_s, "hi": hi_s}, index=idx), aci=pd.DataFrame({"lo": lo_a, "hi": hi_a}, index=idx),
               alpha_path=pd.Series(alphas, index=idx), q_split=q_split, n_cal=len(cal_pos), target=1 - alpha,
               summary=pd.DataFrame({"분할 conformal": [cov(lo_s, hi_s), float(np.mean(hi_s - lo_s))],
                                     "적응형 ACI": [cov(lo_a, hi_a), float(np.mean(hi_a - lo_a))]},
                                    index=["커버리지", "평균 폭"]))
    inside = pd.DataFrame({"분할": (y >= lo_s) & (y <= hi_s), "ACI": (y >= lo_a) & (y <= hi_a)}, index=idx).astype(float)
    res["rolling_cov"] = inside.rolling(max(20, T // 10), min_periods=10).mean()
    if model == "Gaussian Process" and "pred_std" in o["extra"]:          # GP 자체 정규분포 구간과 비교
        from scipy.stats import norm
        z = norm.ppf(1 - alpha / 2); sd = o["extra"]["pred_std"].values
        res["summary"]["GP 자체 구간"] = [cov(p_te - z * sd, p_te + z * sd), float(np.mean(2 * z * sd))]
    return res
