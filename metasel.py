"""모델 선택 메타 전략 — '언제 어떤 예측 모델을 쓸지'를 그 시점까지의 OOS 성과만으로 고르는 중첩 walk-forward.

1단계 (후보 경로) : analytics.walk_forward로 후보 모델마다 확장창 OOS 예측 경로 p_m,t 를 만든다.
2단계 (메타 선택) : 시점 t의 선택·가중은 손실 ℓ_m,s = (y_s − p_m,s)² 중 s ≤ t−h (이미 실현된 것)만 사용.
   (a) 직전 N개월 최소손실 모델         argmin_m Σ_{s∈(t−h−W, t−h]} ℓ_m,s
   (b) 지수가중 역MSE 결합             w_m ∝ 1/EWMSE_m (반감기 H, Stock-Watson 2004의 할인 MSE 가중)
   (c) 레짐 조건부 선택                 현재 레짐 k에서 과거(s ≤ t−h, 레짐(s)=k) 평균 손실이 가장 작은 모델.
       레짐 = HMM 3상태 필터 확률의 최대 상태(risk.hmm_lambda, 매년 과거로 재적합) · 21일 실현변동성의 확장창 3분위 ·
              63일 평균 쌍별 상관(analytics.avg_pairwise_corr)의 확장창 3분위. 해당 레짐 표본이 min_obs 미만이면 전체 과거 최선.
   (d) 동일가중 앙상블                 후보 예측의 단순평균
   (e) 과거 평균 (무정보 기준)          y의 확장창 평균 (t−h까지)
   (f) Hedge 가중                      전문가(모델)별 '실현된 부호전략 성과'(σ로 표준화)의 지수가중 t값 → 소프트맥스 가중
                                       w_m ∝ exp(η·t_m) (Freund-Schapire 1997 Hedge의 할인판: 최근 잘 맞힌 모델에 비중, 망각으로 국면 전환 추종)
   (g) 레짐 게이트 전문가 혼합          현재 레짐 칸(실현변동성 3분위 × 평균 상관 3분위 = 9칸)에서의 과거 성과 t값으로 (f)와 같은 소프트맥스 가중.
                                       칸 표본 n_k가 적으면 전체 기간 가중으로 수축: λ = n_k/(n_k+κ) (Jacobs et al. 1991 mixture of experts의 단순판)
   (h) 전문가 혼합 + 기권               (g)에서 최고 전문가의 수축 t값 < 임계(기본 Φ⁻¹(1−0.2/M), 모델 6개면 1.73)면 포지션 0 (지금 국면에선 어떤 모델도 믿을 근거가 없음)
3단계 (평가) : OOS R²(기준 = 과거 평균), 부호전략 Sharpe(비용 전후), Buy&Hold, 롱 비중, 하위기간(2008~16 / 2017~).
   데이터 스누핑 보정 — (i) 부호 위약: 각 전략의 롱 비중·전환 빈도를 그대로 맞춘 2상태 마르코프 무작위 포지션의 Sharpe 분포(95% 분위),
   '후보 중 최댓값' 위약: 매 위약 시행에서 전 전략 위약 Sharpe의 최댓값 분포, (ii) White(2000) Reality Check · Hansen(2005) SPA
   (정상 부트스트랩, Politis-Romano 1994)."""
from __future__ import annotations

import numpy as np
import pandas as pd

CANDIDATES = ["OLS", "Ridge", "Lasso", "Elastic Net", "Gradient Boosting", "LightGBM"]   # v1.12.0: Kalman TVP → ② 레짐 탭
REGIME_NAMES = {"hmm": "HMM 3상태 (필터)", "vol": "실현변동성 3분위", "corr": "평균 쌍별 상관 3분위"}
META_NAMES = {"recent": "(a) 직전 {n}개월 최소손실", "ewc": "(b) 지수가중 역MSE 결합", "hmm": "(c) 레짐: HMM 3상태",
              "vol": "(c) 레짐: 실현변동성", "corr": "(c) 레짐: 평균 상관", "eq": "(d) 동일가중 앙상블", "mean": "(e) 과거 평균 (기준)",
              "hedge": "(f) Hedge 가중 (최근 성과)", "moe": "(g) 레짐 게이트 전문가 혼합", "moe_abs": "(h) 전문가 혼합 + 기권"}
SUBPERIODS = [("2008~16", "2008-01-01", "2016-12-31"), ("2017~", "2017-01-01", "2100-01-01")]


# ============================================================================
# 1단계: 후보 모델 OOS 경로
# ============================================================================
def candidate_paths(D, models: list[str], ann: int, step: int, oos_start=None, start_frac: float = 0.3,
                    progress=None, **hp) -> tuple[pd.DataFrame, pd.Series]:
    """후보 모델마다 walk_forward OOS 예측 (원 수준, 학습평균 포함). oos_start(날짜)가 있으면 그 날짜부터 OOS."""
    import analytics as A
    if oos_start is not None:
        start_frac = float(np.searchsorted(D.raw.index, pd.Timestamp(oos_start)) / len(D.raw))
    P, y = {}, None
    for j, m in enumerate(models):
        p, y_, _ = A.walk_forward(D, m, ann, step, start_frac, **hp)
        P[m] = p
        y = y_ if y is None else y
        if progress: progress((j + 1) / len(models))
    return pd.DataFrame(P).reindex(y.index), y


# ============================================================================
# 레짐 (모두 t 시점까지의 정보만)
# ============================================================================
def _tercile(x: pd.Series, min_periods: int = 252) -> pd.Series:
    """확장창 백분위 → 0(하)/1(중)/2(상). t의 분위는 x_1..t 만으로 계산."""
    pr = x.expanding(min_periods=min_periods).rank(pct=True)
    return pd.Series(np.where(pr.isna(), np.nan, np.minimum((pr.values * 3).astype(float) // 1, 2)), index=x.index)


def regimes(target_px: pd.Series, R: pd.DataFrame | None, index: pd.DatetimeIndex, corr_win: int = 63,
            kinds=("hmm", "vol", "corr")) -> pd.DataFrame:
    """target_px: 대상 품목 가격(롤 보정), R: 동조성 측정용 수익률 패널(일간), index: 예측 시점 날짜.
    반환: 레짐 라벨 (0·1·2, NaN=판정 불가) — 날짜 t의 라벨은 t 종가까지의 정보."""
    import risk as RK
    import analytics as A
    r = RK.returns(target_px)
    out = {}
    if "hmm" in kinds:
        h = RK.hmm_lambda(r.values, r.index, n_states=3)
        pr = pd.DataFrame(h["prob"], index=r.index)
        lab = pd.Series(np.where(pr.notna().all(1), pr.fillna(0).values.argmax(1), np.nan), index=r.index)
        out["hmm"] = lab
    if "vol" in kinds:
        out["vol"] = _tercile(r.rolling(21).std())
    if "corr" in kinds and R is not None and R.shape[1] >= 2:
        out["corr"] = _tercile(A.avg_pairwise_corr(R.dropna(), corr_win).dropna())
    G = pd.DataFrame(out)
    G = G[~G.index.duplicated()].sort_index()
    return G.reindex(G.index.union(index)).ffill(limit=5).reindex(index)


# ============================================================================
# 2단계: 메타 선택 (모두 shift(h) 된 손실만 사용)
# ============================================================================
def _softmax(T: np.ndarray, eta: float) -> np.ndarray:
    T = np.nan_to_num(T, nan=0.0) * eta
    T = T - T.max(-1, keepdims=True)
    e = np.exp(T)
    return e / e.sum(-1, keepdims=True)


def mixture_predictions(P: pd.DataFrame, y: pd.Series, h: int = 1, ann: int = 252, halflife_months: int = 6, G: pd.DataFrame | None = None,
                        min_obs: int = 60, eta: float = 1.0, kappa: float = 60.0, abstain_t: float | None = None) -> tuple[dict, dict]:
    """(f) Hedge · (g) 레짐 게이트 전문가 혼합 · (h) 기권. 성과 g_m,s = sign(p_m,s)·y_s / σ̂_y (s ≤ t−h 만 사용).
    결합 예측 = Σ w_m · z_m,t (z = 각 모델 예측을 자기 확장창 표준편차로 나눈 값) × 후보 예측 평균 규모 → 부호만 전략에 쓰임."""
    ppm = ann / 12
    from scipy.stats import norm
    if abstain_t is None:                       # 모델 M개 중 최댓값이므로 다중 비교 보정: 1 − 0.2/M 분위 (M=6이면 ≈1.73)
        abstain_t = float(norm.ppf(1 - 0.2 / max(P.shape[1], 1)))
    sd_y = y.expanding(min_periods=min_obs).std().shift(h)
    Gn = (np.sign(P).mul(y, axis=0)).div(sd_y, axis=0).shift(h)              # t에 알려진 실현 성과
    Z = P / P.expanding(min_periods=min_obs).std()
    scale = P.abs().expanding(min_periods=min_obs).mean().mean(1)
    names = np.array(P.columns); eq = P.mean(1)
    out, pick = {}, {}
    hl = max(1.0, halflife_months * ppm)
    mu = Gn.ewm(halflife=hl, min_periods=min_obs).mean(); sdg = Gn.ewm(halflife=hl, min_periods=min_obs).std()
    neff = np.minimum(Gn.notna().cumsum().values, hl / np.log(2) * 2)          # 지수가중 유효 표본 (≈ 2·반감기/ln2)
    T_h = (mu / sdg.clip(lower=1e-9)).values * np.sqrt(np.maximum(neff, 1))
    ok = np.isfinite(T_h).all(1) & Z.notna().all(1).values
    W = _softmax(T_h, eta)
    comb = (np.nan_to_num(Z.values) * W).sum(1) * scale.values
    out["hedge"] = pd.Series(np.where(ok, comb, eq.values), index=P.index)
    pick["hedge"] = pd.Series(np.where(ok, names[W.argmax(1)], "앙상블(초기)"), index=P.index)
    if G is not None and {"vol", "corr"} <= set(G.columns):
        lab = (G["vol"] * 3 + G["corr"]).reindex(P.index)
        lab_l = lab.shift(h).values; labv = lab.values
        Gv = Gn.values; Zv = Z.values; M = P.shape[1]
        s1 = np.zeros((9, M)); s2 = np.zeros((9, M)); nk = np.zeros(9)
        a1 = np.zeros(M); a2 = np.zeros(M); na = 0
        pm = eq.values.copy(); pa = eq.values.copy()
        pkm = np.array(["앙상블(초기)"] * len(P), dtype=object); pka = pkm.copy()
        for t in range(len(P)):
            g = Gv[t]
            if np.isfinite(g).all():
                a1 += g; a2 += g * g; na += 1
                if np.isfinite(lab_l[t]):
                    k = int(lab_l[t]); s1[k] += g; s2[k] += g * g; nk[k] += 1
            if na < min_obs or not np.isfinite(Zv[t]).all():
                continue
            ma = a1 / na; sa = np.sqrt(np.maximum(a2 / na - ma ** 2, 1e-12))
            k = labv[t]
            if np.isfinite(k) and nk[int(k)] >= 5:
                k = int(k); mk = s1[k] / nk[k]; n_ = nk[k]
                lam = n_ / (n_ + kappa)
            else:
                mk, n_, lam = ma, 0.0, 0.0
            m_shr = lam * mk + (1 - lam) * ma                                     # 칸 평균 성과를 전체 평균으로 수축 (표본 적으면 전체)
            tt = m_shr / sa * np.sqrt(max(n_, 1.0))                                # 증거의 양 = 그 칸의 표본 수 → 처음 보는 국면은 기권 쪽
            w = _softmax(tt[None, :], eta)[0]
            v = float(Zv[t] @ w) * scale.values[t]
            pm[t] = v; pkm[t] = names[w.argmax()]
            if tt.max() < abstain_t:
                pa[t] = 0.0; pka[t] = "기권 (포지션 0)"
            else:
                pa[t] = v; pka[t] = names[w.argmax()]
        out["moe"] = pd.Series(pm, index=P.index); pick["moe"] = pd.Series(pkm, index=P.index)
        out["moe_abs"] = pd.Series(pa, index=P.index); pick["moe_abs"] = pd.Series(pka, index=P.index)
    return out, pick


def meta_predictions(P: pd.DataFrame, y: pd.Series, h: int = 1, ann: int = 252, n_months: int = 6, halflife_months: int = 6,
                     G: pd.DataFrame | None = None, min_obs: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    """반환 (메타 예측 DataFrame, 선택된 모델 이름 DataFrame). 선택 정보가 부족한 초기에는 동일가중 앙상블로 대체."""
    ppm = ann / 12
    Lo = (P.sub(y, axis=0) ** 2)
    Lk = Lo.shift(h)                                   # t 시점에 실현이 끝난 손실만 (s ≤ t−h)
    M = P.values; names = np.array(P.columns)
    eq = P.mean(1)
    out, pick = {}, {}
    # (a) 직전 N개월 합계 손실 최소
    W = max(5, int(round(n_months * ppm)))
    S = Lk.rolling(W, min_periods=W).sum()
    ok = S.notna().all(1).values
    j = np.where(ok, np.nan_to_num(S.values, nan=np.inf).argmin(1), -1)
    out["recent"] = pd.Series(np.where(ok, M[np.arange(len(M)), np.clip(j, 0, None)], eq.values), index=P.index)
    pick["recent"] = pd.Series(np.where(ok, names[np.clip(j, 0, None)], "앙상블(초기)"), index=P.index)
    # (b) 지수가중 MSE의 역수로 가중 결합
    E = Lk.ewm(halflife=max(1.0, halflife_months * ppm), min_periods=min_obs).mean()
    w = 1 / E.clip(lower=1e-12)
    w = w.div(w.sum(1), axis=0)
    okb = w.notna().all(1)
    out["ewc"] = (P * w).sum(1).where(okb, eq)
    pick["ewc"] = pd.Series(np.where(okb, names[np.nan_to_num(w.values, nan=-1).argmax(1)], "앙상블(초기)"), index=P.index)
    # (c) 레짐 조건부: 같은 레짐의 과거 손실 평균이 최소인 모델 (레짐 표본 부족 시 전체 과거 평균 최소)
    if G is not None:
        Lv = Lk.values; cum_all = np.nancumsum(np.nan_to_num(Lv), 0); cnt_all = np.cumsum(np.isfinite(Lv).all(1))
        for g in G.columns:
            lab = G[g].reindex(P.index)
            lab_l = lab.shift(h).values                # 손실 ℓ_s 에 붙는 레짐 = s 시점 레짐 (Lk와 같은 이동)
            K = int(np.nanmax(lab.values)) + 1 if lab.notna().any() else 0
            sums = np.zeros((K, M.shape[1])); cnts = np.zeros(K)
            pred = eq.values.copy(); pk = np.array(["앙상블(초기)"] * len(P), dtype=object)
            for t in range(len(P)):
                lt = lab_l[t]
                if np.isfinite(lt) and np.isfinite(Lv[t]).all():
                    sums[int(lt)] += Lv[t]; cnts[int(lt)] += 1
                k = lab.values[t]
                if np.isfinite(k) and cnts[int(k)] >= min_obs:
                    jj = int(np.argmin(sums[int(k)] / cnts[int(k)]))
                elif cnt_all[t] >= min_obs:
                    jj = int(np.argmin(cum_all[t]))
                else:
                    continue
                pred[t] = M[t, jj]; pk[t] = names[jj]
            out[g] = pd.Series(pred, index=P.index); pick[g] = pd.Series(pk, index=P.index)
    out["eq"] = eq
    out["mean"] = y.expanding().mean().shift(h)
    mo, mp = mixture_predictions(P, y, h, ann, halflife_months, G, min_obs)
    out.update(mo); pick.update(mp)
    labels = {k: META_NAMES[k].format(n=n_months) for k in out}
    return pd.DataFrame(out).rename(columns=labels), pd.DataFrame(pick).rename(columns=labels)


# ============================================================================
# 3단계: 평가 · 위약 · SPA
# ============================================================================
def _net(p: np.ndarray, y: np.ndarray, cost: float) -> tuple[np.ndarray, np.ndarray]:
    pos = np.sign(p)
    return pos * y - np.abs(np.diff(np.r_[0.0, pos])) * cost, pos


def _sr(x: np.ndarray, per: float) -> float:
    x = x[np.isfinite(x)]
    return float(x.mean() / x.std() * np.sqrt(per)) if len(x) > 1 and x.std() > 0 else np.nan


def evaluate(F: pd.DataFrame, y: pd.Series, h: int = 1, ann: int = 252, cost: float = 0.0,
             bench: pd.Series | None = None) -> pd.DataFrame:
    """F: 전략별 예측(원 수준). OOS R²의 기준 = 과거 평균(bench, 없으면 y 확장평균). h>1이면 겹치지 않는 [::h] 표본."""
    bench = y.expanding().mean().shift(h) if bench is None else bench
    rows = {}
    for c in F.columns:
        m = F[c].notna() & bench.notna()
        yy, pp, bb = y[m].values, F[c][m].values, bench[m].values
        r2 = 1 - ((yy - pp) ** 2).sum() / ((yy - bb) ** 2).sum()
        ys, ps = yy[::h], pp[::h]
        net, pos = _net(ps, ys, cost)
        row = {"OOS R² (vs 과거평균)": r2, "부호전략 Sharpe": _sr(pos * ys, ann / h), "부호전략 Sharpe (비용 후)": _sr(net, ann / h),
               "Buy&Hold Sharpe": _sr(ys, ann / h), "롱 비중": float(np.mean(pos > 0)),
               "연간 회전율": float(np.abs(np.diff(np.r_[0.0, pos])).mean() * ann / h)}
        idx = y[m].index[::h]
        for nm, a, b in SUBPERIODS:
            s = (idx >= a) & (idx <= b)
            row[f"Sharpe 비용 후 {nm}"] = _sr(net[s], ann / h)
            row[f"B&H {nm}"] = _sr(ys[s], ann / h)
        rows[c] = row
    return pd.DataFrame(rows).T


def placebo(F: pd.DataFrame, y: pd.Series, h: int = 1, ann: int = 252, cost: float = 0.0, B: int = 1000,
            seed: int = 0) -> dict:
    """전략별 '롱 비중·전환 빈도 일치' 무작위 부호 위약 (2상태 마르코프 연쇄).
    반환: table(관측 Sharpe·위약 중앙값·95% 분위·p), max(관측 최댓값, 위약 최댓값 분포 95% 분위, p)."""
    rng = np.random.default_rng(seed)
    m = F.notna().all(1)
    yy = y[m].values[::h]; n = len(yy)
    SR = np.empty((B, F.shape[1])); obs = {}
    for k, c in enumerate(F.columns):
        net, pos = _net(F[c][m].values[::h], yy, cost)
        obs[c] = _sr(net, ann / h)
        q = float(np.clip(np.mean(pos > 0), 1e-6, 1 - 1e-6))
        f = float(np.mean(pos[1:] != pos[:-1])) if n > 1 else 0.0
        pLS, pSL = min(1.0, f / (2 * q)), min(1.0, f / (2 * (1 - q)))
        s = np.empty((B, n)); cur = np.where(rng.random(B) < q, 1.0, -1.0)
        U = rng.random((B, n))
        for t in range(n):
            if t:
                flip = np.where(cur > 0, U[:, t] < pLS, U[:, t] < pSL)
                cur = np.where(flip, -cur, cur)
            s[:, t] = cur
        turn = np.abs(np.diff(np.c_[np.zeros(B), s], axis=1))
        x = s * yy[None, :] - turn * cost
        SR[:, k] = x.mean(1) / x.std(1) * np.sqrt(ann / h)
    o = pd.Series(obs)
    tab = pd.DataFrame({"Sharpe (비용 후)": o, "위약 중앙값": np.median(SR, 0), "위약 95% 분위": np.quantile(SR, 0.95, 0),
                        "위약 p (단일)": [(SR[:, k] >= o.iloc[k]).mean() for k in range(len(o))]}, index=F.columns)
    mx = SR.max(1)
    return dict(table=tab, max=dict(관측최댓값=float(o.max()), 최고전략=str(o.idxmax()), 위약최댓값95=float(np.quantile(mx, 0.95)),
                                    위약최댓값중앙=float(np.median(mx)), p=float((mx >= o.max()).mean())))


def _stationary_idx(n: int, B: int, block: float, rng) -> np.ndarray:
    """Politis-Romano 정상 부트스트랩 인덱스 (B, n), 평균 블록 길이 block."""
    p = 1.0 / block
    I = np.empty((B, n), dtype=np.int64)
    I[:, 0] = rng.integers(0, n, B)
    new = rng.random((B, n)) < p; rnd = rng.integers(0, n, (B, n))
    for t in range(1, n):
        I[:, t] = np.where(new[:, t], rnd[:, t], (I[:, t - 1] + 1) % n)
    return I


def spa_test(d: np.ndarray, B: int = 1000, block: float = 20.0, seed: int = 0) -> dict:
    """d: (n, K) 성과 차이(전략 − 기준, 클수록 좋음). H0: 모든 k에서 E[d_k] ≤ 0.
    White RC: 재중심 = d̄ (최소 유리).  Hansen SPA_c: 재중심 = d̄·1{d̄ ≥ −√(ω²·2 log log n / n)} → 부트스트랩 평균 = d̄* − d̄ + d̄·1{d̄ < −임계} (나쁜 전략의 영향 제거)."""
    d = np.asarray(d, float); d = d[np.isfinite(d).all(1)]
    n, K = d.shape
    rng = np.random.default_rng(seed)
    I = _stationary_idx(n, B, block, rng)
    dbar = d.mean(0)
    Mb = np.stack([d[I[b]].mean(0) for b in range(B)])            # (B, K) 부트스트랩 평균
    om = np.sqrt(n) * Mb.std(0); om[om <= 0] = np.inf
    T_rc = np.sqrt(n) * dbar.max()
    T_spa = max(0.0, float((np.sqrt(n) * dbar / om).max()))
    mu_c = dbar * (dbar < -np.sqrt(om ** 2 / n * 2 * np.log(np.log(n))))   # 확실히 나쁜 전략만 아래로 이동 (나머지는 0에서 재중심)
    Z = Mb - dbar                                                  # 중심화된 부트스트랩 평균
    rc_b = np.sqrt(n) * Z.max(1)
    spa_b = np.maximum(0.0, (np.sqrt(n) * (Z + mu_c) / om).max(1))
    return dict(RC_p=float((rc_b >= T_rc).mean()), SPA_p=float((spa_b >= T_spa).mean()), 최고평균차=float(dbar.max()),
                최고=int(dbar.argmax()), n=n, K=K)


def snooping_tests(F: pd.DataFrame, y: pd.Series, h: int = 1, cost: float = 0.0, B: int = 1000, block: float = 20.0,
                   seed: int = 0, bench: pd.Series | None = None) -> pd.DataFrame:
    """세 가지 기준에 대한 RC·SPA p값: 순수익 vs 0(무포지션), 순수익 vs Buy&Hold, 제곱오차 vs 과거평균."""
    bench = y.expanding().mean().shift(h) if bench is None else bench
    m = F.notna().all(1) & bench.notna()
    yy = y[m].values[::h]; FF = F[m].iloc[::h]
    NET = np.column_stack([_net(FF[c].values, yy, cost)[0] for c in F.columns])
    SE = np.column_stack([(yy - bench[m].values[::h]) ** 2 - (yy - FF[c].values) ** 2 for c in F.columns])
    rows = []
    for nm, d in [("순수익 > 0 (무포지션)", NET), ("순수익 > Buy&Hold", NET - yy[:, None]), ("제곱오차 < 과거평균", SE)]:
        r = spa_test(d, B, block, seed)
        rows.append({"귀무가설 기준": nm, "최고 전략": F.columns[r["최고"]], "최고 평균차": r["최고평균차"],
                     "White RC p": r["RC_p"], "Hansen SPA p": r["SPA_p"], "전략 수": r["K"], "관측치": r["n"]})
    return pd.DataFrame(rows)


def run(D, levels: pd.DataFrame, R: pd.DataFrame | None, target: str, ann: int = 252, step: int = 126,
        models: list[str] | None = None, oos_start="2004-01-01", eval_start="2008-01-01", n_months: int = 6,
        halflife_months: int = 6, B: int = 1000, progress=None, **hp) -> dict:
    """전 과정: 후보 경로 → 레짐 → 메타 예측 → 평가(eval_start~)·위약·SPA.
    levels: 롤 보정 가격, R: 평균 상관 레짐용 일간 수익률 패널(품목들), oos_start: 후보 OOS 시작(메타 선택의 예열 기간 포함)."""
    models = models or CANDIDATES
    P, y = candidate_paths(D, models, ann, step, oos_start=oos_start, progress=progress, **hp)
    G = regimes(levels[target], R, P.index)          # 레짐 판정은 일간 자료 기준 (R = 일간 수익률 패널) → 예측 시점 날짜로 맞춤
    Mp, pick = meta_predictions(P, y, D.horizon, ann, n_months, halflife_months, G)
    F = P.join(Mp)
    ev = F.index >= pd.Timestamp(eval_start)
    F, y_e = F[ev], y[ev]
    bench = Mp["(e) 과거 평균 (기준)"][ev]
    tab = evaluate(F, y_e, D.horizon, ann, D.cost, bench)
    tab.insert(0, "구분", ["후보 모델" if c in models else "메타 전략" for c in tab.index])
    pl = placebo(F, y_e, D.horizon, ann, D.cost, B)
    tab = tab.join(pl["table"].drop(columns="Sharpe (비용 후)"))
    sp = snooping_tests(F, y_e, D.horizon, D.cost, B, bench=bench)
    return dict(table=tab, preds=F, y=y_e, pick=pick[ev], regimes=G, placebo_max=pl["max"], spa=sp, models=models)


def cum_net(F: pd.DataFrame, y: pd.Series, h: int = 1, cost: float = 0.0) -> pd.DataFrame:
    """전략별 누적 순수익(%) + Buy&Hold — 차트용."""
    m = F.notna().all(1)
    yy = y[m].iloc[::h]
    C = {c: pd.Series(_net(F[c][m].values[::h], yy.values, cost)[0], index=yy.index).cumsum() for c in F.columns}
    C["Buy & Hold"] = yy.cumsum()
    return pd.DataFrame(C)
