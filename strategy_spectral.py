"""주파수(Spectral) 분석 · Spectral 포트폴리오 백테스트 (W).

질문: 다른 연구의 '장기 주파수일수록 평균 코히어런스(Coh²) · 정보 밀도가 높다'가 우리 원자재 19개 주간 자료에서도 나오나? 나온다면 주파수별로 위험을 다르게 배분하는
Spectral 포트폴리오가 기존 MinVar 축을 개선하나?

1) 코히어런스 · 정보 밀도 (기술 통계, 룩어헤드 무관): 멀티테이퍼(DPSS, K=5) 교차 스펙트럼 → 품목 쌍 평균 Coh² (밴드별), 에너지 몫(표준화 수익의 스펙트럼 비중),
   정보 밀도 = 평균 Coh² ÷ 에너지 몫 (사진 속 정의와 같음: 0.18/72% = 0.25×). 영가설 = 품목별 독립 원형 이동(스펙트럼 유지, 상호 의존 파괴) 평균 Coh²를 빼 초과 Coh²도 계산.
   자료 = 원 수익률 / Ridge-VAR(1) 잔차, 구간 = 전체 · 학습(2005–2016) · 시험(2017–), 밴드 정의 2종.
2) Spectral 포트폴리오 (MinVar 축 대체, 4주마다 재계산, 롱온리 품목 상한 25%, 편도 5bp, 목표 변동성 10% — 3축 혼합 틀):
   S1 주파수 가중 공분산: 창 수익을 푸리에 변환해 밴드별 이득(√λ)을 곱한 뒤 Ledoit-Wolf → MinVar  (λ = 단기 · 중기 · 장기 비중, (1,1,1)=기준)
   S2 주파수별 요인 공분산: 밴드마다 상위 K개 PC로 공통 부분을 만들어 합산 + 품목 고유 대각
   S3 기간별 위험 배분: 밴드별 MinVar(그 밴드 공분산만) 3개를 균등 혼합
실행: .venv\\Scripts\\python.exe strategy_spectral.py → data/strategy_spectral_coherence.csv · strategy_spectral_compare.csv · strategy_spectral_weekly.parquet
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.signal.windows import dpss

warnings.filterwarnings("ignore")
import analytics as A  # noqa: E402
import strategy_cov_factor as CF  # noqa: E402

D = A.DATA
BANDS = {"B1": (8, 26), "B2": (26, 78)}                        # (단기|중기 경계, 중기|장기 경계) 주 — B1: 8주·26주, B2: 6개월·18개월(사진 정의에 가깝게)
BN = ["단기", "중기", "장기"]
SPLIT = pd.Timestamp("2017-01-01")


# ------------------------------------------------------------------ 주파수 도구
def masks(N: int, edges) -> list:
    k = np.arange(N // 2 + 1); per = np.where(k > 0, N / np.maximum(k, 1), np.inf); a, b = edges
    return [(k > 0) & (per < a), (k > 0) & (per >= a) & (per < b), (k > 0) & (per >= b)]


def filt(X: np.ndarray, gains, edges) -> np.ndarray:
    """X(T×n)를 평균 제거 후 푸리에 변환, 밴드별 이득 곱해 되돌림. DC 제거."""
    N = len(X); F = np.fft.rfft(X - X.mean(0), axis=0); g = np.zeros(F.shape[0])
    for m, gn in zip(masks(N, edges), gains):
        g[m] = gn
    return np.fft.irfft(F * g[:, None], n=N, axis=0)


def coherence_table(X: np.ndarray, edges, NW: float = 3.0, K: int = 5, null_draws: int = 30, seed: int = 0) -> list:
    """X(T×n, 표준화) → 밴드별 {에너지 몫, 평균 Coh², 영가설 Coh²}."""
    N, n = X.shape; tap = dpss(N, NW, K); ms = masks(N, edges); rng = np.random.default_rng(seed)

    def coh(Z):
        Y = np.fft.rfft(Z.T[None, :, :] * tap[:, None, :], axis=2)                       # K × n × F
        S = np.einsum("kif,kjf->ijf", Y, Y.conj()); P = np.real(np.einsum("iif->if", S))
        C = np.abs(S) ** 2 / (P[:, None, :] * P[None, :, :] + 1e-30)
        iu = np.triu_indices(n, 1)
        return C[iu[0], iu[1], :].mean(0), P.sum(0)                                      # 주파수별 평균 Coh², 총 파워

    c, p = coh(X); tot = p[1:].sum()
    nulls = []
    for _ in range(null_draws):
        Zs = np.column_stack([np.roll(X[:, j], rng.integers(10, N - 10)) for j in range(n)]); nulls.append(coh(Zs)[0])
    c0 = np.mean(nulls, axis=0)
    out = []
    for nm, m in zip(BN, ms):
        e = p[m].sum() / tot; ch = float(c[m].mean()); c00 = float(c0[m].mean())
        out.append(dict(성분=nm, 에너지몫=float(e), Coh2=ch, 영가설Coh2=c00, 초과Coh2=ch - c00, 정보밀도=ch / e, 초과정보밀도=(ch - c00) / e, 주파수칸=int(m.sum())))
    return out


def ridge_var_resid(X: np.ndarray, alpha: float = 50.0) -> np.ndarray:
    from sklearn.linear_model import Ridge
    A_ = Ridge(alpha=alpha, fit_intercept=True).fit(X[:-1], X[1:]); return X[1:] - A_.predict(X[:-1])


def run_coherence(R: pd.DataFrame) -> pd.DataFrame:
    X0 = R.loc["2005":].dropna(axis=1, thresh=int(len(R.loc["2005":]) * 0.9)).fillna(0.0)
    X0 = (X0 - X0.mean()) / X0.std()
    rows = []
    for dat in ("원 수익률", "Ridge-VAR(1) 잔차"):
        Z = X0 if dat == "원 수익률" else pd.DataFrame(ridge_var_resid(X0.values), index=X0.index[1:], columns=X0.columns)
        for per, sl in (("전체 2005–", Z), ("학습 2005–2016", Z.loc[:"2016"]), ("시험 2017–", Z.loc["2017":])):
            for bd, ed in BANDS.items():
                for r in coherence_table(sl.values, ed):
                    rows.append({"자료": dat, "구간": per, "밴드 정의": f"{bd} (<{ed[0]}주 · {ed[0]}~{ed[1]}주 · ≥{ed[1]}주)", "관측 주": len(sl), "품목 수": sl.shape[1], **r})
        print(f"  코히어런스 {dat} 완료", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 공분산 추정기 (CF.minvar_w용: X = T×n 소수 수익률)
def _rescale(Xf, X):
    s = np.trace(np.cov(X, rowvar=False)) / max(np.trace(np.cov(Xf, rowvar=False)), 1e-30)
    return Xf * np.sqrt(s)


def est_spec(gains, edges):
    def f(X, Fw=None):
        Xf = _rescale(filt(X, gains, edges), X); return CF.est_lw(Xf)
    return f


def est_specfac(K, edges):
    def f(X, Fw=None):
        S0 = np.cov(X, rowvar=False); C = np.zeros_like(S0)
        for b in range(3):
            g = [0.0, 0.0, 0.0]; g[b] = 1.0; Sb = np.cov(filt(X, g, edges), rowvar=False)
            w, V = np.linalg.eigh(Sb); ix = np.argsort(w)[::-1][:K]; C += (V[:, ix] * w[ix]) @ V[:, ix].T
        return C + np.diag(np.clip(np.diag(S0) - np.diag(C), 1e-8, None))
    return f


def est_band(b, edges):
    def f(X, Fw=None):
        g = [0.0, 0.0, 0.0]; g[b] = 1.0; return CF.est_lw(filt(X, g, edges))
    return f


def share_long(r: pd.Series, cut: float = 26.0) -> float:
    """전략 수익의 주기 ≥ cut주 성분 분산 몫 (시험 구간)."""
    x = r.loc[SPLIT:].values; x = x - x.mean(); N = len(x); F = np.abs(np.fft.rfft(x)) ** 2
    per = np.where(np.arange(len(F)) > 0, N / np.maximum(np.arange(len(F)), 1), np.inf); m = (np.arange(len(F)) > 0)
    return float(F[m & (per >= cut)].sum() / F[m].sum())


# ------------------------------------------------------------------ 백테스트
def run_portfolios(L, S3, R, P):
    base_mix = CF.mix3(S3["MinVar"], S3); rows, store = [], {}
    ctrl = {}

    def evaluate(nm, fam, win, edges_name, lam, W, pv, ref_mv, ref_w):
        mv = CF.vt_book(W, R); mx = CF.mix3(mv, S3)
        s1 = CF.summarize(mv, ref_mv); s2 = CF.summarize(mx, base_mix); bt, ba = CF.bias_stat(W, pv, R) if pv is not None else (np.nan, np.nan)
        rows.append({"묶음": fam, "구성": nm, "창(주)": win, "밴드": edges_name, "λ(단·중·장)": lam, **s1, "연 회전율 (배)": CF.turnover(W), "장기(≥26주) 분산 몫 (시험)": share_long(mv),
                     "위험 예측 편향 시험": bt, "혼합 학습": s2["Sharpe 학습 2008–16"], "혼합 시험": s2["Sharpe 시험 2017–"], "혼합 시험 차이 t": s2["시험 차이 t"], "혼합 최대 낙폭 (%)": s2["최대 낙폭 (%)"]})
        store[f"{fam}|{nm}|{win}|{edges_name}|{lam}"] = mv
        print(f"  {fam} {nm} [{win}주 {edges_name} {lam}]: 학습 {s1['Sharpe 학습 2008–16']:+.2f} 시험 {s1['Sharpe 시험 2017–']:+.2f} · 혼합 시험 {s2['Sharpe 시험 2017–']:.2f}", flush=True)

    configs = [("B1", 104), ("B1", 260), ("B2", 260), ("B2", 520)]
    for win in sorted({w for _, w in configs}):
        t0 = time.time(); W, pv = CF.minvar_w(P, R, CF.est_lw, win); mv = CF.vt_book(W, R); ctrl[win] = (mv, W, pv)
        evaluate("기준 Ledoit-Wolf", "기준", win, "–", "–", W, pv, mv, win); print(f"    ({time.time() - t0:.0f}s)", flush=True)
    LAMS = {"단기 강조": (3, 1, 0.5), "중기 강조": (0.5, 3, 1), "장기 강조": (0.5, 1, 3), "장기 집중": (0.1, 0.3, 3)}
    for bn, win in configs:
        ed = BANDS[bn]; rmv = ctrl[win][0]
        for nm, lam in LAMS.items():
            W, pv = CF.minvar_w(P, R, est_spec(lam, ed), win); evaluate(nm, "S1 주파수 가중 공분산", win, bn, str(lam), W, pv, rmv, win)
        for K in (1, 2):
            W, pv = CF.minvar_w(P, R, est_specfac(K, ed), win); evaluate(f"밴드별 PC {K}개", "S2 주파수별 요인 공분산", win, bn, "(1,1,1)", W, pv, rmv, win)
        Ws = []
        for b in range(3):
            Wb, _ = CF.minvar_w(P, R, est_band(b, ed), win); Ws.append(Wb)
        Wm = sum(Ws) / 3.0; evaluate("밴드별 MinVar 균등 혼합", "S3 기간별 위험 배분", win, bn, "(⅓,⅓,⅓)", Wm, None, rmv, win)
        for b, Wb in enumerate(Ws):
            evaluate(f"{BN[b]} 밴드 MinVar만", "S3 기간별 위험 배분", win, bn, f"{BN[b]}만", Wb, None, rmv, win)
    return pd.DataFrame(rows), store


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t00 = time.time(); L, S3, R, P = CF.load()
    print("[1] 코히어런스 · 정보 밀도", flush=True)
    C = run_coherence(R); C.to_csv(D / "strategy_spectral_coherence.csv", index=False, encoding="utf-8-sig")
    print(C[(C["구간"] == "전체 2005–")].round(3).to_string(), flush=True)
    print("[2] Spectral 포트폴리오 백테스트", flush=True)
    T, store = run_portfolios(L, S3, R, P)
    T["학습 최고"] = ""
    for (fam, win, bn), g in T[T["묶음"] != "기준"].groupby(["묶음", "창(주)", "밴드"]):
        T.loc[g["Sharpe 학습 2008–16"].idxmax(), "학습 최고"] = "←"
    T.to_csv(D / "strategy_spectral_compare.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(store).to_parquet(D / "strategy_spectral_weekly.parquet")
    print(f"완료 {time.time() - t00:.0f}s", flush=True)


if __name__ == "__main__" and "--placebo" not in sys.argv:
    main()


# ------------------------------------------------------------------ 위약: 주파수 칸 무작위 배정
def est_spec_rand(gains, edges, seed):
    """밴드 크기는 그대로 두고 어느 주파수 칸이 단 · 중 · 장 이득을 받는지를 무작위로 섞음 (밴드의 '주기 의미'만 제거한 대조)."""
    def f(X, Fw=None):
        N = len(X); ms = masks(N, edges); k = np.arange(N // 2 + 1); idx = np.where(k > 0)[0]
        lab = np.zeros(len(k), dtype=int); sizes = [m.sum() for m in ms]; perm = np.random.default_rng(seed * 1000 + N).permutation(idx)
        g = np.zeros(len(k)); pos = 0
        for gn, sz in zip(gains, sizes):
            g[perm[pos:pos + sz]] = gn; pos += sz
        F = np.fft.rfft(X - X.mean(0), axis=0); Xf = np.fft.irfft(F * g[:, None], n=N, axis=0)
        return CF.est_lw(_rescale(Xf, X))
    return f


def run_placebo(draws: int = 20):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    L, S3, R, P = CF.load(); base_mix = CF.mix3(S3["MinVar"], S3); rows = []
    T = pd.read_csv(D / "strategy_spectral_compare.csv", encoding="utf-8-sig")
    for bn, win in (("B1", 260), ("B2", 260), ("B2", 520)):
        ed = BANDS[bn]; lam = (0.5, 1, 3)
        real = T[(T["구성"] == "장기 강조") & (T["창(주)"] == win) & (T["밴드"] == bn)].iloc[0]
        ctrl = T[(T["묶음"] == "기준") & (T["창(주)"] == win)].iloc[0]
        pl = []
        for s in range(draws):
            W, _ = CF.minvar_w(P, R, est_spec_rand(lam, ed, s), win); mv = CF.vt_book(W, R); mx = CF.mix3(mv, S3)
            a = CF.summarize(mv, mv); b = CF.summarize(mx, base_mix)
            pl.append((a["Sharpe 학습 2008–16"], a["Sharpe 시험 2017–"], b["Sharpe 학습 2008–16"], b["Sharpe 시험 2017–"]))
        pl = np.array(pl); print(f"  위약 {bn} {win}주: 시험 평균 {pl[:, 1].mean():+.2f} (실제 {real['Sharpe 시험 2017–']:+.2f}) · 혼합 시험 평균 {pl[:, 3].mean():.2f} (실제 {real['혼합 시험']:.2f})", flush=True)
        for j, nm in enumerate(["MinVar 축 학습", "MinVar 축 시험", "혼합 학습", "혼합 시험"]):
            rv = [real["Sharpe 학습 2008–16"], real["Sharpe 시험 2017–"], real["혼합 학습"], real["혼합 시험"]][j]
            rows.append({"밴드": bn, "창(주)": win, "지표": nm, "기준(LW)": [ctrl["Sharpe 학습 2008–16"], ctrl["Sharpe 시험 2017–"], ctrl["혼합 학습"], ctrl["혼합 시험"]][j], "장기 강조 실제": rv,
                         "위약 평균": pl[:, j].mean(), "위약 95% 상한": np.quantile(pl[:, j], 0.95), "위약 p": float((pl[:, j] >= rv).mean()), "위약 횟수": draws})
    pd.DataFrame(rows).to_csv(D / "strategy_spectral_placebo.csv", index=False, encoding="utf-8-sig"); print("위약 완료")


if __name__ == "__main__" and "--placebo" in sys.argv:
    run_placebo()
