"""백테스트 과적합 판정 도구 — Purged K-fold(+embargo), Deflated Sharpe Ratio, PBO(CSCV).

purged_kfold : 시계열을 연속 블록 K개로 나누고, 검증 블록과 '목표 기간이 겹치는' 학습 관측치를 제거(purge)한 뒤
               검증 블록 직후 embargo 개 관측치도 학습에서 뺀다 (López de Prado 2018, ch.7). 목표가 h기간 누적이면 horizon=h.
deflated_sharpe : 여러 전략을 시도해 최고를 고를 때의 기대 최대 Sharpe(SR₀)와 표본 길이·왜도·첨도를 반영한 '진짜일 확률'
               (Bailey & López de Prado 2014). DSR ≥ 0.95면 다중 시도·비정규성을 감안해도 Sharpe > SR₀.
pbo_cscv     : 조합 대칭 교차검증(CSCV)으로 'IS 최고 전략이 OOS에서 중앙값 아래일 확률' = 과적합 확률 (Bailey et al. 2017).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import stats

EULER = 0.5772156649


def purged_kfold(n: int, n_splits: int = 5, horizon: int = 1, embargo: int = 0):
    """반환: [(train_idx, test_idx), ...]. 관측치 i의 목표는 [i, i+horizon) 구간의 정보를 사용한다고 가정."""
    bounds = np.linspace(0, n, n_splits + 1).astype(int)
    out = []
    idx = np.arange(n)
    for a, b in zip(bounds[:-1], bounds[1:]):
        test = idx[a:b]
        keep = (idx + horizon <= a) | (idx >= b + embargo)            # 목표가 검증 블록과 겹치지 않고, embargo 이후
        out.append((idx[keep], test))
    return out


def naive_kfold(n: int, n_splits: int = 5, seed: int = 0):
    """비교용: 무작위로 섞은 K-fold (시계열 누수 발생)."""
    rng = np.random.default_rng(seed); perm = rng.permutation(n)
    folds = np.array_split(perm, n_splits)
    return [(np.setdiff1d(perm, f), np.sort(f)) for f in folds]


def sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1)) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan


def expected_max_sr(sr_trials: np.ndarray) -> float:
    """N개 시도의 Sharpe 분산으로 본 기대 최대 Sharpe (기간당 단위)."""
    s = np.asarray(sr_trials, float); s = s[np.isfinite(s)]; N = len(s)
    if N < 2:
        return 0.0
    v = np.var(s, ddof=1)
    return float(np.sqrt(v) * ((1 - EULER) * stats.norm.ppf(1 - 1 / N) + EULER * stats.norm.ppf(1 - 1 / (N * np.e))))


def psr(x: np.ndarray, sr_star: float = 0.0) -> float:
    """확률적 Sharpe: P(진짜 SR > sr_star) (기간당 단위, 왜도·첨도 보정)."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]; T = len(x)
    if T < 10:
        return np.nan
    sr = sharpe(x); g3 = stats.skew(x); g4 = stats.kurtosis(x, fisher=False)
    den = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(stats.norm.cdf((sr - sr_star) * np.sqrt(T - 1) / den))


def deflated_sharpe(best_returns: np.ndarray, sr_trials: np.ndarray) -> dict:
    """best_returns: 선택된(최고) 전략의 기간 수익률, sr_trials: 시도한 모든 전략의 기간당 Sharpe."""
    sr0 = expected_max_sr(sr_trials)
    return {"SR (기간당)": sharpe(best_returns), "기대 최대 SR₀ (기간당)": sr0, "시도 수": int(np.isfinite(sr_trials).sum()),
            "PSR (SR>0)": psr(best_returns, 0.0), "DSR (SR>SR₀)": psr(best_returns, sr0)}


def pbo_cscv(M: np.ndarray, S: int = 16, max_combos: int = 5000, seed: int = 0) -> dict:
    """M: (T, N) 전략별 기간 수익률 (같은 기간). 행을 S개 블록으로 나눠 절반을 IS, 나머지를 OOS로 하는 모든 조합에서
    IS 최고 전략의 OOS 상대 순위 ω → logit λ. PBO = P(λ ≤ 0). 조합이 많으면 max_combos개 무작위 표본."""
    M = np.asarray(M, float)
    M = M[np.isfinite(M).all(1)]
    T, N = M.shape
    if N < 2 or T < S * 4:
        return {"PBO": np.nan, "조합 수": 0}
    blocks = np.array_split(np.arange(T), S)
    combs = list(combinations(range(S), S // 2))
    if len(combs) > max_combos:
        rng = np.random.default_rng(seed); combs = [combs[i] for i in rng.choice(len(combs), max_combos, replace=False)]
    sr = lambda X: X.mean(0) / (X.std(0, ddof=1) + 1e-12)
    lam, deg = [], []
    for c in combs:
        is_ = np.concatenate([blocks[i] for i in c]); oos = np.concatenate([blocks[i] for i in range(S) if i not in c])
        s_is, s_oos = sr(M[is_]), sr(M[oos])
        b = int(np.argmax(s_is))
        w = (stats.rankdata(s_oos)[b]) / (N + 1)
        lam.append(np.log(w / (1 - w))); deg.append((s_is[b], s_oos[b]))
    lam = np.array(lam); deg = np.array(deg)
    return {"PBO": float((lam <= 0).mean()), "조합 수": len(combs), "IS 최고의 OOS SR 평균 (기간당)": float(deg[:, 1].mean()),
            "IS 최고의 IS SR 평균 (기간당)": float(deg[:, 0].mean()), "OOS SR < 0 비율": float((deg[:, 1] < 0).mean())}
