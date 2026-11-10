"""요인 모형 → 포트폴리오 연결 검토 백테스트 (Q) — PCA · Dynamic/Latent factor · FAVAR 아이디어를 우리 19개 품목에서 같은 틀로 시험.

F1 PC 포트폴리오: 직전 104주 표준화 수익률의 PCA 고유벡터를 그대로 포트폴리오 비중으로(w = v/σ, 총노출 1) — PC1(공통 요인) · PC2 · PC3 단독과 PC1~3 균등.
   PC1 부호 = 적재값 합 양수(고정). PC2 · PC3 부호 = (a) 적재값 합 양수(고정, 임의) (b) 직전 104주 평균 수익이 + 가 되는 쪽(과거 승자 방향).
F2 잠재 요인 평균-분산: 요인 K개(1 · 2 · 3 · 5)의 평균(직전 104주 평균 × 0.5 축소) · 분산으로 요인 공간 평균-분산 w_f = μ_f / σ_f² → 품목 비중 w = V w_f / σ (총노출 1)
   PCA(공분산) vs RP-PCA(Lettau-Pelger 2020: 공분산 + (1+γ)·평균평균′, γ = 0 · 20).
F3 거시 요인 → 기대수익 (FAVAR-lite): 시장 기반 거시 변수 약 20개(VIX · 금리 · 스프레드 · 달러 · 환율 · 금융여건 …)의 주간 변화를 확장창 PCA(요인 3개) → 품목별 '4주 뒤 수익 ~ 요인' 릿지(확장창, 4주 purge) → 예측 μ의 횡단면 순위로 상위 1/3 롱 · 하위 1/3 숏.
   월간 거시 통계(CPI · 산업생산 · 고용 …)는 공표 지연이 커서 제외, 공분산 예측(FAVAR-SV)은 시험하지 않음.
공통: 4주마다 재계산(요인 · 회귀는 그 시점까지 자료만), 다음 주 수익으로 손익, 편도 5bp, 목표 변동성 10%(직전 26주, 상한 4배). 학습 2008–2016 · 시험 2017–. 설정은 학습 Sharpe로만 고름.
평가 기준: 3축 혼합 50/25/25(학습 0.61 · 시험 1.07) 대비, 그리고 'MinVar 50 + 모멘텀 25 + 이 전략 25' 혼합. 위약: 비중(신호)을 13주 블록 순서로 섞어 같은 구조 유지(100회).
실행: .venv\\Scripts\\python.exe strategy_factor_portfolio.py → data/strategy_factor_portfolio_compare.csv · _weekly.parquet
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
import strategy_cov_factor as CF  # noqa: E402
import strategy_meta as SM  # noqa: E402
import strategy_switch as SS  # noqa: E402

WIN, REB = 104, 4
SPLIT = "2017-01-01"


def _eig(C):
    w, V = np.linalg.eigh(C); o = np.argsort(w)[::-1]
    return w[o], V[:, o]


def pca_inputs(X: pd.DataFrame, cols, gamma):
    Xv = X[cols].fillna(0.0).values; sd = Xv.std(0, ddof=1); sd[sd == 0] = 1.0
    Z0 = Xv / sd; m = Z0.mean(0); Zc = Z0 - m
    C = Zc.T @ Zc / (len(Zc) - 1) + (1 + gamma) * np.outer(m, m)              # gamma = -1 → 공분산 PCA, ≥ 0 → RP-PCA
    w, V = _eig(C)
    return sd, Z0, Zc, V, w


def weights_pc(R: pd.DataFrame, k: int, sign: str) -> pd.DataFrame:
    """PC 포트폴리오 비중: w = v_k / σ, 총노출 1. sign = 'fixed'(적재값 합 +) 또는 'trail'(직전 104주 평균 수익 +)."""
    W = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
    for t in range(WIN, len(R), REB):
        X = R.iloc[t - WIN + 1:t + 1]; cols = [c for c in R.columns if X[c].notna().sum() >= WIN * 0.8]
        if len(cols) < 6:
            continue
        sd, Z0, Zc, V, w_ = pca_inputs(X, cols, -1.0)
        v = V[:, k] * (1 if V[:, k].sum() >= 0 else -1)
        if sign == "trail" and k > 0:
            f = Z0 @ v
            v = v if f.mean() >= 0 else -v
        a = v / sd; a = a / np.abs(a).sum()
        W.loc[R.index[t], cols] = a; W.loc[R.index[t], [c for c in R.columns if c not in cols]] = 0.0
    return W.ffill().fillna(0.0)


def weights_lfmv(R: pd.DataFrame, K: int, gamma: float, shrink: float = 0.5) -> pd.DataFrame:
    """잠재 요인 평균-분산: 요인 K개의 (축소 평균, 분산)으로 w_f = μ_f/σ_f², 품목 비중 = V w_f / σ."""
    W = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
    for t in range(WIN, len(R), REB):
        X = R.iloc[t - WIN + 1:t + 1]; cols = [c for c in R.columns if X[c].notna().sum() >= WIN * 0.8]
        if len(cols) < max(6, K + 1):
            continue
        sd, Z0, Zc, V, w_ = pca_inputs(X, cols, gamma); VK = V[:, :K]
        F = Z0 @ VK; mu = shrink * F.mean(0); var = np.clip((Zc @ VK).var(0, ddof=1), 1e-9, None)
        a = (VK @ (mu / var)) / sd
        g = np.abs(a).sum()
        if g <= 0:
            continue
        W.loc[R.index[t], cols] = a / g; W.loc[R.index[t], [c for c in R.columns if c not in cols]] = 0.0
    return W.ffill().fillna(0.0)


# ------------------------------------------------------------------ F3 거시 요인 → 기대수익
MAC_L = ["fred:VIXCLS", "fred:DGS10", "fred:DGS2", "fred:DTB3", "fred:BAA10Y", "fred:T10YIE", "fred:DTWEXBGS", "yf:DX-Y.NYB", "yf:^GSPC"]
MAC_X = ["DFII10", "T5YIFR", "NFCI", "STLFSI4", "OVXCLS", "GVZCLS", "BAMLC0A0CM", "DEXBZUS", "DEXJPUS", "DEXUSAL", "DEXUSEU", "T10Y3M", "DFF"]
LOGCH = {"fred:VIXCLS", "fred:DTWEXBGS", "yf:DX-Y.NYB", "yf:^GSPC", "OVXCLS", "GVZCLS", "DEXBZUS", "DEXJPUS", "DEXUSAL", "DEXUSEU"}


def macro_panel(L: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    fe = pd.read_parquet(A.DATA / "mm" / "fred_extra.parquet"); fe["date"] = pd.to_datetime(fe["date"])
    cols = {c: L[c] for c in MAC_L if c in L}
    for i in MAC_X:
        g = fe[fe["id"] == i].drop_duplicates("date").set_index("date")["value"].sort_index()
        if len(g):
            cols[i] = g
    out = {}
    for c, s in cols.items():
        w = s.dropna().resample("W-FRI").last().reindex(idx).ffill(limit=3).shift(1)            # 1주 지연(공표 · 정렬 안전)
        out[c] = np.log(w).diff() * 100 if c in LOGCH else w.diff()
    M = pd.DataFrame(out)
    return M.loc[:, M.notna().mean() > 0.6]


def macro_mu(P: pd.DataFrame, R: pd.DataFrame, M: pd.DataFrame, K=3, alpha=20.0, first=156):
    """4주마다: 거시 변화 패널 확장창 PCA(요인 K) → 품목별 '4주 뒤 수익 ~ 요인' 릿지(확장창, 4주 purge) → 예측 μ (품목 × 시점)."""
    from sklearn.linear_model import Ridge
    y4 = R.rolling(4).sum().shift(-4)                                                         # t 이후 4주 누적 수익 (t에서는 미지)
    mu = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
    for t in range(WIN, len(R), REB):
        d = R.index[t]
        Xm = M.loc[:d].dropna(thresh=int(M.shape[1] * 0.7)).fillna(0.0)
        if len(Xm) < first + 8:
            continue
        sd = Xm.std().replace(0, 1.0); Zm = (Xm - Xm.mean()) / sd
        w_, V = _eig(np.cov(Zm.values, rowvar=False)); F = pd.DataFrame(Zm.values @ V[:, :K], index=Zm.index)
        F = F / F.std()
        tr = F.index[F.index <= R.index[max(t - 4, 0)]]
        for c in R.columns:
            yy = y4[c].reindex(tr); ok = yy.notna()
            if ok.sum() < first or d not in F.index:
                continue
            mdl = Ridge(alpha=alpha).fit(F.loc[tr][ok.values].values, yy[ok].values)
            mu.loc[d, c] = float(mdl.predict(F.loc[[d]].values)[0])
    return mu.ffill(limit=REB - 1), y4


def rank_ic(mu: pd.DataFrame, y4: pd.DataFrame) -> tuple[float, float, int]:
    ic = []
    for d in mu.index[::REB]:
        a, b = mu.loc[d], y4.loc[d]; ok = a.notna() & b.notna()
        if ok.sum() >= 8:
            ic.append(a[ok].rank().corr(b[ok].rank()))
    s = pd.Series(ic); return float(s.mean()), float(s.mean() / s.std() * np.sqrt(len(s))), len(s)


def placebo(W: pd.DataFrame, R: pd.DataFrame, base_te: float, n=100, seed=0) -> dict:
    rng = np.random.default_rng(seed); blocks = [W.index[i:i + 13] for i in range(0, len(W), 13)]; te, tr = [], []
    for _ in range(n):
        o = rng.permutation(len(blocks)); Wp = pd.concat([W.loc[blocks[j]] for j in o]); Wp.index = W.index
        r = CF.vt_book(Wp, R); te.append(CF.sh(r.loc[SPLIT:])); tr.append(CF.sh(r.loc["2008":"2016"]))
    return {"te": np.array(te), "tr": np.array(tr)}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t00 = time.time(); L, S3, R, P = CF.load()
    base = CF.mix3(S3["MinVar"], S3); sc = SM.score(base, base)
    print(f"[기준 3축 혼합] 학습 {sc['Sharpe 학습 2008–16']:.2f} · 시험 {sc['Sharpe 시험 2017–']:.2f}", flush=True)
    rows, store = [], {}

    def add(fam, nm, W, extra=None, sig=False):
        r = CF.vt_book(W, R); s1 = CF.summarize(r, base)
        X = S3.assign(신호=r.reindex(S3.index).fillna(0.0))
        mx = SS.run_weights(X, pd.DataFrame([[0.5, 0.25, 0.0, 0.25]] * len(X), index=X.index, columns=["MinVar", "모멘텀", "가치", "신호"])) if False else None
        m = (X[["MinVar", "모멘텀", "신호"]].values @ np.array([0.5, 0.25, 0.25])); m = pd.Series(m, X.index)
        pv = m.rolling(26, min_periods=13).std().shift(1) * np.sqrt(52); m = m * (10.0 / pv).clip(upper=4.0).fillna(1.0)
        s2 = CF.summarize(m, base)
        rows.append({"묶음": fam, "구성": nm, **s1, "연 회전율 (배)": CF.turnover(W), "혼합 학습": s2["Sharpe 학습 2008–16"], "혼합 시험": s2["Sharpe 시험 2017–"], "혼합 시험 차이 t": s2["시험 차이 t"], **(extra or {})})
        store[f"{fam}|{nm}"] = r
        print(f"  {fam} {nm}: 학습 {s1['Sharpe 학습 2008–16']:+.2f} 시험 {s1['Sharpe 시험 2017–']:+.2f} · 혼합 시험 {s2['Sharpe 시험 2017–']:.2f}", flush=True)
        return r

    # F1 PC 포트폴리오
    Wpc = {}
    for k, nm in enumerate(["PC1 (공통 요인)", "PC2", "PC3"]):
        for sg_, lab in (("fixed", "부호 고정(적재값 합 +)"), ("trail", "부호 = 직전 104주 승자 방향")):
            if k == 0 and sg_ == "trail":
                continue
            W = weights_pc(R, k, sg_); Wpc[(k, sg_)] = W; add("F1 PC 포트폴리오", f"{nm} · {lab}", W)
    Wc = sum(Wpc[(k, "trail" if k else "fixed")] for k in range(3)) / 3
    add("F1 PC 포트폴리오", "PC1 + PC2 + PC3 균등 (PC2 · PC3은 승자 방향)", Wc)
    # F2 잠재 요인 평균-분산
    Wlf = {}
    for gamma, gl in ((-1.0, "PCA"), (0.0, "RP-PCA γ=0"), (20.0, "RP-PCA γ=20")):
        for K in (1, 2, 3, 5):
            W = weights_lfmv(R, K, gamma); Wlf[(gamma, K)] = W; add("F2 잠재 요인 평균-분산", f"{gl} · 요인 {K}개", W)
    # F3 거시 요인 → 기대수익
    M = macro_panel(L, R.index); print(f"  거시 패널 {M.shape} · 시작 {M.dropna(thresh=int(M.shape[1] * .7)).index[0]:%Y-%m}", flush=True)
    vol = R.rolling(26, min_periods=13).std()
    mu, y4 = macro_mu(P, R, M)
    ic, ict, nic = rank_ic(mu, y4)
    Wm = LX.xs_weights(mu.where(mu.notna()), hold=4, inv_vol=1 / vol)
    add("F3 거시 요인 → 기대수익", "거시 PCA 3요인 → 릿지 μ → 순위 롱숏", Wm, {"순위 IC": ic, "IC t": ict})
    print(f"  F3 순위 IC {ic:+.3f} (t {ict:+.2f}, n={nic})", flush=True)
    T = pd.DataFrame(rows); T["학습 최고"] = ""
    for fam in T["묶음"].unique():
        sub = T[T["묶음"] == fam]; T.loc[sub["Sharpe 학습 2008–16"].idxmax(), "학습 최고"] = "←"
    # 위약: 학습 최고(F1 · F2) + F3
    cand = {"F2 잠재 요인 평균-분산": None, "F1 PC 포트폴리오": None}
    for fam in cand:
        b = T[(T["묶음"] == fam) & (T["학습 최고"] == "←")].iloc[0]
        nm = b["구성"]
        if fam.startswith("F2"):
            g_ = {"PCA": -1.0, "RP-PCA γ=0": 0.0, "RP-PCA γ=20": 20.0}[nm.split(" · ")[0]]; Kk = int(nm.split("요인 ")[1][0]); W = Wlf[(g_, Kk)]
        else:
            W = Wc if nm.startswith("PC1 +") else Wpc[(0, "fixed")] if nm.startswith("PC1 (") else Wpc[(1 if nm.startswith("PC2") else 2, "trail" if "승자" in nm else "fixed")]
        pl = placebo(W, R, b["Sharpe 시험 2017–"])
        i = b.name; T.loc[i, "위약 시험 평균"] = pl["te"].mean(); T.loc[i, "위약 시험 95%"] = np.percentile(pl["te"], 95)
        T.loc[i, "위약 p (시험)"] = float((pl["te"] >= b["Sharpe 시험 2017–"]).mean()); T.loc[i, "위약 p (학습)"] = float((pl["tr"] >= b["Sharpe 학습 2008–16"]).mean())
    b = T[T["묶음"].str.startswith("F3")].iloc[0]; pl = placebo(Wm, R, b["Sharpe 시험 2017–"]); i = b.name
    T.loc[i, "위약 시험 평균"] = pl["te"].mean(); T.loc[i, "위약 시험 95%"] = np.percentile(pl["te"], 95)
    T.loc[i, "위약 p (시험)"] = float((pl["te"] >= b["Sharpe 시험 2017–"]).mean()); T.loc[i, "위약 p (학습)"] = float((pl["tr"] >= b["Sharpe 학습 2008–16"]).mean())
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 46)
    T.to_csv(A.DATA / "strategy_factor_portfolio_compare.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(store).to_parquet(A.DATA / "strategy_factor_portfolio_weekly.parquet")
    print(T[["묶음", "구성", "Sharpe 학습 2008–16", "Sharpe 시험 2017–", "시험 차이 t", "연 회전율 (배)", "혼합 학습", "혼합 시험", "혼합 시험 차이 t", "학습 최고", "위약 p (시험)", "위약 p (학습)"]].round(2).to_string(index=False))
    print(f"총 {time.time() - t00:.0f}s")


if __name__ == "__main__":
    main()
