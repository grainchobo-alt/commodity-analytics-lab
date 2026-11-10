"""요인 모형 → 포트폴리오: 포트폴리오의 요인(PCA) 위험 분해.

내 포트폴리오 수익 r_p = Σ w_j r_j 를 표준화 수익 z_j = (r_j − μ_j)/σ_j 의 PCA 요인으로 풀면
  r_p ≈ Σ_j (w_j σ_j) z_j = Σ_k e_k f_k,   e_k = (w∘σ)·v_k  (포트폴리오의 k번째 요인 노출),  f_k = z·v_k (요인 점수, 서로 무상관, 분산 = λ_k)
  분산 = Σ_k e_k² λ_k  → 요인별 분산 기여 (합 100%), K개를 넘는 요인은 '고유 변동'으로 묶음.
  유효 요인 수(effective number of bets, Meucci 2009) = exp(−Σ p_k ln p_k), p_k = 요인 분산 기여 비중 — 클수록 위험이 여러 요인에 퍼짐.
해석 주의: PC의 경제적 의미(공통 시장 · 에너지 대 농산물 …)는 적재값을 보고 사람이 판단. 이 계산은 '지금 포트폴리오의 위험이 어떤 요인에 몰렸나'를 보는 도구이며 수익 예측이 아님
(요인으로 비중을 짜는 방법은 strategy_factor_portfolio.py에서 시험 — 미채택).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pca_std(R: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """표준화 수익의 공분산(=상관) PCA. 반환: 고유값(내림차순), 고유벡터, 표준편차, 평균. 각 PC 부호 = 적재값 합이 양수."""
    X = R.fillna(0.0); sd = X.std(ddof=1).replace(0, 1.0); mu = X.mean()
    Z = (X - mu) / sd
    w, V = np.linalg.eigh(np.cov(Z.values, rowvar=False)); o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    V = V * np.where(V.sum(0, keepdims=True) >= 0, 1.0, -1.0)
    return w, V, sd.values, mu


def risk_decomposition(R: pd.DataFrame, w: pd.Series, K: int = 3, win: int = 104) -> dict:
    """R = 주간(또는 일간) 수익률 % (열 = 품목), w = 자본 대비 비중(+롱 −숏). 최근 win개 관측으로 PCA → 요인 노출 · 분산 기여."""
    ids = [i for i in w.index if i in R.columns]
    X = R[ids].dropna(how="all").iloc[-win:]
    lam, V, sd, mu = pca_std(X)
    wv = w.reindex(ids).values.astype(float)
    e = (wv * sd) @ V                                              # 요인 노출 (표준화 단위)
    contrib = e ** 2 * lam
    tot = float(contrib.sum())
    share = contrib / tot if tot > 0 else contrib
    K = int(min(K, len(ids)))
    tbl = pd.DataFrame({"시장 설명력": lam / lam.sum(), "포트폴리오 노출 e": e, "분산 기여": share}, index=[f"PC{k + 1}" for k in range(len(ids))])
    top = tbl.iloc[:K].copy()
    rest = pd.DataFrame({"시장 설명력": [tbl["시장 설명력"].iloc[K:].sum()], "포트폴리오 노출 e": [np.nan], "분산 기여": [tbl["분산 기여"].iloc[K:].sum()]}, index=[f"PC{K + 1}~ (고유 변동)"])
    p = share[share > 1e-12]
    enb = float(np.exp(-(p * np.log(p)).sum()))
    L = pd.DataFrame(V[:, :K], index=ids, columns=[f"PC{k + 1}" for k in range(K)])
    return dict(table=pd.concat([top, rest]), full=tbl, loadings=L, enb=enb, n=len(ids), K=K, vol=float(np.sqrt(tot)), nobs=len(X),
                top_share=float(share[:K].sum()), pc1_share=float(share[0]) if len(share) else np.nan, pc1_market=float(lam[0] / lam.sum()))
