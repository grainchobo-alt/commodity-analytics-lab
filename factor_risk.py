"""요인 모형으로 '현재 포트폴리오의 위험'을 진단 — 수익 예측 · 비중 설계가 아니라 '요인이 무엇을 뜻하고 변동성에 얼마나 기여하나'.

모델별 진단 (입력 w = 자본 대비 품목 비중, R = 주간 수익률 %):
  ① dynamic_pca      동적 PCA   — 롤링 창마다 PCA를 다시 추정해 PC1 몫 · 유효 요인 수(ENB) · 요인 구조 안정성의 시간 경로
  ② named_factor_risk 관측 요인  — 시장 · 모멘텀 · 가치 · 저변동 롱숏 요인에 포트폴리오 수익을 회귀, 요인별 분산 기여
  ③ macro_factor_risk 거시 요인  — 거시 변수 PCA 요인(FAVAR-lite 요인)에 회귀, 요인별 분산 기여 + 요인 의미(적재 상위 변수)
  ④ spectral_risk    주파수별   — 단기 · 중기 · 장기 성분으로 나눠 위험이 어느 투자 기간에 몰렸나, 기간별 공통 요인 몫
순수 진단용이라 미래 정보 차단(룩어헤드 방지)이 필요한 예측 백테스트와 다르게, 주어진 구간 전체를 설명하는 데 씀.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import factor_portfolio as FP

MAC_KO = {"fred:VIXCLS": "VIX", "fred:DGS10": "미 10년 금리", "fred:DGS2": "미 2년 금리", "fred:DTB3": "3개월 금리", "fred:BAA10Y": "BAA 신용 스프레드", "fred:T10YIE": "10년 기대 인플레",
          "fred:DTWEXBGS": "달러 지수(광의)", "yf:DX-Y.NYB": "달러 인덱스", "yf:^GSPC": "S&P500", "DFII10": "10년 실질 금리", "T5YIFR": "5년 선행 기대 인플레", "NFCI": "금융 여건(NFCI)",
          "STLFSI4": "금융 스트레스", "OVXCLS": "유가 변동성(OVX)", "GVZCLS": "금 변동성(GVZ)", "BAMLC0A0CM": "투자등급 스프레드", "DEXBZUS": "브라질 헤알", "DEXJPUS": "엔",
          "DEXUSAL": "호주 달러", "DEXUSEU": "유로", "T10Y3M": "장단기 금리차", "DFF": "기준금리"}
NAMED = {"MKT": ("시장", "품목 19개 동일비중 바스켓 — 원자재가 다 같이 오르내리는 방향"), "MOM": ("모멘텀", "지난 1년 오른 품목 롱 · 내린 품목 숏"),
         "VAL": ("가치", "5년 평균보다 싼 품목 롱 · 비싼 품목 숏"), "LVOL": ("저변동", "변동성 낮은 품목 롱 · 높은 품목 숏")}
BANDS = (("단기 (2~8주)", 0, 8), ("중기 (8~26주)", 8, 26), ("장기 (26주 이상)", 26, 1e9))


def port_ret(R: pd.DataFrame, w: pd.Series) -> pd.Series:
    ids = [i for i in w.index if i in R.columns]
    return (R[ids].fillna(0.0) * w.reindex(ids).values).sum(1)


# ---------------------------------------------------------------- ① 동적 PCA
def dynamic_pca(R: pd.DataFrame, w: pd.Series, win: int = 104, step: int = 4) -> pd.DataFrame:
    """창을 굴리며 PCA 재추정(현재 비중 w를 그 구조에 대입). 열: PC1 분산 기여 · PC1 시장 설명력 · ENB · PC1 안정성(직전 창 PC1과 코사인) · 연 변동성(%)."""
    ids = [i for i in w.index if i in R.columns]
    X = R[ids]; out = []; hist = []; lag = max(win // step, 1)                      # 안정성 = 겹치지 않는 직전 창(win주 전)의 PC1과의 코사인
    for t in range(win, len(X) + 1, step):
        Xi = X.iloc[t - win:t]
        if Xi.notna().sum().min() < win * 0.8:
            continue
        o = FP.risk_decomposition(Xi, w.reindex(ids), 3, win)
        v = FP.pca_std(Xi)[1][:, 0]
        cs = float(abs(v @ hist[-lag])) if len(hist) >= lag else np.nan
        hist.append(v)
        out.append(dict(date=X.index[t - 1], pc1_pf=o["pc1_share"], pc1_mkt=o["pc1_market"], enb=o["enb"], stab=cs, vol=o["vol"] * np.sqrt(52)))
    return pd.DataFrame(out).set_index("date")


# ---------------------------------------------------------------- ② 관측 요인
def named_factor_risk(R: pd.DataFrame, w: pd.Series, Fac: pd.DataFrame, n: int = 260) -> dict:
    rp = port_ret(R, w); D = pd.concat([rp.rename("rp"), Fac], axis=1, join="inner").dropna().iloc[-n:]
    y = D["rp"].values; F = D.drop(columns="rp"); Xm = np.column_stack([np.ones(len(F)), F.values])
    b, *_ = np.linalg.lstsq(Xm, y, rcond=None); beta = b[1:]
    var_y = float(np.var(y, ddof=1)); cov = np.array([np.cov(F[c].values, y, ddof=1)[0, 1] for c in F.columns])
    contrib = beta * cov / var_y                                                     # 오일러식: 합 = R²
    r2 = float(contrib.sum()); resid = 1 - r2
    tbl = pd.DataFrame({"요인": [NAMED[c][0] for c in F.columns], "뜻": [NAMED[c][1] for c in F.columns], "베타": beta, "분산 기여": contrib}, index=list(F.columns))
    return dict(table=tbl, r2=r2, resid=resid, n=len(D), vol=float(np.sqrt(var_y * 52)), alpha=float(b[0] * 52))


# ---------------------------------------------------------------- ③ 거시 요인
def macro_factor_risk(R: pd.DataFrame, w: pd.Series, M: pd.DataFrame, K: int = 3, n: int = 260) -> dict:
    rp = port_ret(R, w); D = pd.concat([rp.rename("rp"), M], axis=1, join="inner").iloc[-n:]
    Mm = D.drop(columns="rp"); Mm = Mm.loc[:, Mm.notna().mean() > 0.7].fillna(0.0); y = D["rp"].values
    sd = Mm.std().replace(0, 1.0); Z = (Mm - Mm.mean()) / sd
    lam, V = np.linalg.eigh(np.cov(Z.values, rowvar=False)); o = np.argsort(lam)[::-1]; lam, V = lam[o], V[:, o]
    K = min(K, V.shape[1]); rows = []; load = {}
    tot = 0.0
    for k in range(K):
        v = V[:, k]; j = int(np.argmax(np.abs(v)))
        v = v * (1 if v[j] >= 0 else -1)                                           # 부호: 가장 큰 적재가 +
        f = Z.values @ v; c = float(np.corrcoef(f, y)[0, 1]); beta = c * y.std(ddof=1) / f.std(ddof=1)
        top = pd.Series(v, index=Mm.columns).reindex(pd.Series(v, index=Mm.columns).abs().sort_values(ascending=False).index)[:3]
        lab = " · ".join(f"{MAC_KO.get(i, i)}{'↑' if x > 0 else '↓'}" for i, x in top.items())
        rows.append({"요인": f"거시 요인 {k + 1}", "거시 변수 설명력": lam[k] / lam.sum(), "의미 (적재 상위 변수)": lab, "포트폴리오 상관": c, "분산 기여": c * c}); tot += c * c
        load[f"거시 요인 {k + 1}"] = pd.Series(v, index=[MAC_KO.get(i, i) for i in Mm.columns])
    return dict(table=pd.DataFrame(rows).set_index("요인"), r2=tot, resid=1 - tot, n=len(D), nvar=Mm.shape[1], loadings=pd.DataFrame(load))


# ---------------------------------------------------------------- ④ 주파수별
def spectral_risk(R: pd.DataFrame, w: pd.Series, n: int = 520) -> dict:
    ids = [i for i in w.index if i in R.columns]
    X = R[ids].iloc[-n:].fillna(0.0); X = X - X.mean(); N = len(X)
    F = np.fft.rfft(X.values, axis=0); k = np.arange(F.shape[0]); per = np.where(k > 0, N / np.maximum(k, 1), np.inf)
    rows = []; wv = w.reindex(ids).values.astype(float)
    for nm, lo, hi in BANDS:
        m = (k > 0) & (per >= lo) & (per < hi)                                      # 주기(주) 구간 [lo, hi) — 서로 겹치지 않고 합치면 전체
        G = np.where(m[:, None], F, 0); Xb = pd.DataFrame(np.fft.irfft(G, n=N, axis=0), index=X.index, columns=ids)
        vb = float(np.var(Xb.values @ wv, ddof=1)); o = FP.risk_decomposition(Xb, w.reindex(ids), 3, N)
        rows.append({"성분": nm, "분산": vb, "주파수 몫(무작위 기대)": float(m.sum() / (k > 0).sum()), "PC1 분산 기여": o["pc1_share"], "PC1 시장 설명력": o["pc1_market"], "유효 요인 수": o["enb"]})
    T = pd.DataFrame(rows).set_index("성분"); tot = T["분산"].sum()
    T["분산 몫"] = T["분산"] / tot; T["연 변동성 (%)"] = np.sqrt(T["분산"] * 52)
    return dict(table=T[["분산 몫", "주파수 몫(무작위 기대)", "연 변동성 (%)", "PC1 분산 기여", "PC1 시장 설명력", "유효 요인 수"]], n=N, nitems=len(ids))
