"""횡단면 분석 — 한 시점에 여러 품목을 나란히 놓고 군집(clustering) · 분류(classification)

  특성 (주간, 그 시점까지의 자료만): 수익률 4·13·26·52주, 변동성 13·52주, 왜도·첨도 52주, 52주 고점 대비 낙폭, 52주 범위 내 위치,
        S&P·달러 베타 52주, 다른 품목과의 평균 상관 52주, 추세 강도(26주 수익률 ÷ 26주 변동성)
  군집 (비지도): K-평균 · 계층(Ward) · 가우시안 혼합(GMM) — 특성을 품목 간 z점수로 표준화, K는 실루엣 점수로 추천
         + 상관 기반 계층 군집 (거리 = 1 − 주간 수익률 상관, 평균 연결) → 덴드로그램
  분류 (지도): 목표 = 다음 4주 수익률이 그 시점 품목 중앙값보다 높은가 (상대 강세)
         로지스틱 · 선형판별(LDA, 최소제곱 판별과 동치) · SVM(RBF) · 랜덤포레스트 · 그래디언트 부스팅 · k-최근접
         4주 간격 표본(목표 겹침 방지), 매년 1월 그 이전 자료로 재학습 → 다음 해 예측 (walk-forward), 특성은 시점별 순위 변환
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATS = {"r4": "수익률 4주 (%)", "r13": "수익률 13주 (%)", "r26": "수익률 26주 (%)", "r52": "수익률 52주 (%)",
         "vol13": "변동성 13주 (연 %)", "vol52": "변동성 52주 (연 %)", "skew52": "왜도 52주", "kurt52": "첨도 52주",
         "dd52": "52주 고점 대비 (%)", "pos52": "52주 범위 내 위치 (0 저점–1 고점)", "beta_spx": "S&P 베타 52주",
         "beta_dxy": "달러 베타 52주", "corr_avg": "다른 품목과 평균 상관 52주", "trend": "추세 강도 (26주 수익 ÷ 변동성)"}
CLASSIFIERS = ["로지스틱 회귀", "선형판별 (LDA · 최소제곱)", "SVM (RBF 커널)", "랜덤포레스트", "그래디언트 부스팅", "k-최근접 (k=15)"]


def weekly_prices(L: pd.DataFrame, ids: list[str], start="2001-01-01") -> pd.DataFrame:
    P = L[[i for i in ids if i in L]].resample("W-FRI").last()
    return P[P.index >= pd.Timestamp(start)]


def feature_panel(L: pd.DataFrame, ids: list[str], start="2001-01-01") -> dict:
    """반환 dict(F={특성: date×id}, R=주간 수익률(%), P=가격). 모든 값은 그 주까지의 자료만."""
    P = weekly_prices(L, ids, start)
    R = 100 * np.log(P).diff()
    lp = np.log(P)
    F = {}
    for h in (4, 13, 26, 52):
        F[f"r{h}"] = 100 * (lp - lp.shift(h))
    F["vol13"] = R.rolling(13, min_periods=10).std() * np.sqrt(52)
    F["vol52"] = R.rolling(52, min_periods=40).std() * np.sqrt(52)
    F["skew52"] = R.rolling(52, min_periods=40).skew()
    F["kurt52"] = R.rolling(52, min_periods=40).kurt()
    mx = P.rolling(52, min_periods=40).max(); mn = P.rolling(52, min_periods=40).min()
    F["dd52"] = 100 * (P / mx - 1)
    F["pos52"] = (P - mn) / (mx - mn).replace(0, np.nan)
    for k, mid in [("beta_spx", "yf:^GSPC"), ("beta_dxy", "yf:DX-Y.NYB")]:
        if mid in L:
            m = 100 * np.log(L[mid].resample("W-FRI").last()).diff().reindex(R.index)
            F[k] = R.apply(lambda c: c.rolling(52, min_periods=40).cov(m)).div(m.rolling(52, min_periods=40).var(), axis=0)
    cs = {}
    for c in R:
        others = R.drop(columns=c)
        cs[c] = others.apply(lambda o: R[c].rolling(52, min_periods=40).corr(o)).mean(1)
    F["corr_avg"] = pd.DataFrame(cs)
    F["trend"] = F["r26"] / (R.rolling(26, min_periods=20).std() * np.sqrt(26))
    return dict(F=F, R=R, P=P)


def snapshot(fp: dict, asof=None) -> pd.DataFrame:
    """품목 × 특성 (원값). asof 없으면 가장 최근 주."""
    F = fp["F"]
    dt = F["r4"].dropna(how="all").index.max() if asof is None else F["r4"].index[F["r4"].index <= pd.Timestamp(asof)].max()
    return pd.DataFrame({k: v.loc[dt] for k, v in F.items()}).dropna(how="all"), dt


def zscore(X: pd.DataFrame) -> pd.DataFrame:
    Z = (X - X.mean()) / X.std(ddof=0).replace(0, np.nan)
    return Z.fillna(0.0)


def cluster(X: pd.DataFrame, method: str = "K-평균", k: int | None = None, seed: int = 0) -> dict:
    """품목 군집. k=None이면 2–6 중 실루엣 최대. 반환 labels(Series), k, silhouette 표, PCA 2D 좌표·설명력·적재값."""
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.mixture import GaussianMixture
    Z = zscore(X)

    def fit(kk):
        if method.startswith("K"):
            return KMeans(kk, n_init=20, random_state=seed).fit_predict(Z)
        if method.startswith("계층"):
            return AgglomerativeClustering(kk, linkage="ward").fit_predict(Z)
        return GaussianMixture(kk, covariance_type="diag", n_init=5, random_state=seed).fit(Z).predict(Z)
    sil = {}
    for kk in range(2, min(7, len(Z))):
        lab = fit(kk)
        if len(set(lab)) > 1:
            sil[kk] = float(silhouette_score(Z, lab))
    k = k or (max(sil, key=sil.get) if sil else 2)
    lab = fit(k)
    # 군집 번호를 '52주 수익률 평균' 오름차순으로 다시 매김 (해석 편의)
    order = pd.Series(X["r52"].values if "r52" in X else Z.iloc[:, 0].values).groupby(lab).mean().sort_values().index
    remap = {old: new + 1 for new, old in enumerate(order)}
    labels = pd.Series([remap[l] for l in lab], index=X.index, name="군집")
    pca = PCA(2).fit(Z)
    XY = pd.DataFrame(pca.transform(Z), index=X.index, columns=["PC1", "PC2"])
    load = pd.DataFrame(pca.components_.T, index=X.columns, columns=["PC1", "PC2"])
    return dict(labels=labels, k=k, silhouette=pd.Series(sil, name="실루엣"), xy=XY, evr=pca.explained_variance_ratio_, load=load,
                profile=X.groupby(labels).mean())


def corr_dendrogram(R: pd.DataFrame, weeks: int = 104):
    """상관 기반 계층 군집: 거리 = √(2(1−ρ)), 평균 연결."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    X = R.iloc[-weeks:].dropna(axis=1, thresh=int(weeks * 0.8))
    C = X.corr()
    D = np.sqrt(np.clip(2 * (1 - C.values), 0, None)); np.fill_diagonal(D, 0)
    Lk = linkage(squareform(D, checks=False), "average")
    return C, Lk


def class_panel(fp: dict, h: int = 4, step: int = 4, feats: list[str] | None = None):
    """분류 학습용 (date,id) 패널: 특성은 시점별 순위 변환(−0.5~0.5), 목표 = 다음 h주 수익 > 그 시점 중앙값. step주 간격 표본."""
    F = {k: v for k, v in fp["F"].items() if feats is None or k in feats}
    R = fp["R"]
    fwd = R.rolling(h).sum().shift(-h)
    dates = R.index[::step]
    rows = []
    for d in dates:
        x = pd.DataFrame({k: v.loc[d] for k, v in F.items()})
        f = fwd.loc[d]
        ok = x.notna().all(1)
        if ok.sum() < 6:
            continue
        xr = x[ok].rank(pct=True) - 0.5
        y = (f[ok] > f[ok].median()).astype(float).where(f[ok].notna())
        xr["y"] = y; xr["fwd"] = f[ok]; xr["date"] = d; xr["id"] = xr.index
        rows.append(xr)
    D = pd.concat(rows, ignore_index=True)
    return D, list(F)


def _model(name: str, seed=0):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    return {"로지스틱 회귀": lambda: LogisticRegression(C=0.5, max_iter=500),
            "선형판별 (LDA · 최소제곱)": lambda: LinearDiscriminantAnalysis(),
            "SVM (RBF 커널)": lambda: SVC(C=0.5, gamma="scale"),
            "랜덤포레스트": lambda: RandomForestClassifier(300, min_samples_leaf=20, max_features=0.5, random_state=seed, n_jobs=-1),
            "그래디언트 부스팅": lambda: GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=seed),
            "k-최근접 (k=15)": lambda: KNeighborsClassifier(15)}[name]()


def _score(m, X):
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)[:, 1]
    return m.decision_function(X)


def walk_forward(D: pd.DataFrame, feats: list[str], models: list[str], first_year: int = 2008, split="2017-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """매년 1월 재학습 → 그해 예측. 반환 (성과표, 예측 패널). 성과 = AUC, 상위 1/3 − 하위 1/3 다음 4주 수익 차(%)."""
    from sklearn.metrics import roc_auc_score
    D = D.dropna(subset=["y"]).copy()
    D["year"] = pd.to_datetime(D["date"]).dt.year
    preds = D[["date", "id", "y", "fwd"]].copy()
    for mn in models:
        preds[mn] = np.nan
        for y in sorted(D.year.unique()):
            if y < first_year:
                continue
            tr = D[(D.year < y) & (pd.to_datetime(D["date"]) <= pd.Timestamp(f"{y}-01-01") - pd.Timedelta(weeks=5))]   # 목표 겹침 제거
            te = D.year == y
            if len(tr) < 300 or te.sum() == 0:
                continue
            m = _model(mn).fit(tr[feats].values, tr["y"].values)
            preds.loc[te, mn] = _score(m, D.loc[te, feats].values)
    rows = []
    P_ = preds.dropna(subset=[models[0]]) if models else preds
    for mn in models:
        q = preds.dropna(subset=[mn])
        for per, msk in [("전체", slice(None)), (f"~{split[:4]}", pd.to_datetime(q["date"]) < split), (f"{split[:4]}~", pd.to_datetime(q["date"]) >= split)]:
            qq = q[msk] if not isinstance(msk, slice) else q
            if len(qq) < 50 or qq["y"].nunique() < 2:
                continue
            ls = qq.groupby("date").apply(lambda g: g.loc[g[mn] >= g[mn].quantile(2 / 3), "fwd"].mean() - g.loc[g[mn] <= g[mn].quantile(1 / 3), "fwd"].mean())
            rows.append({"모델": mn, "기간": per, "AUC": float(roc_auc_score(qq["y"], qq[mn])), "판단 시점 수": int(qq["date"].nunique()),
                         "상위 1/3 − 하위 1/3 (4주 %)": float(ls.mean()), "차이 t값": float(ls.mean() / ls.std(ddof=1) * np.sqrt(len(ls))) if ls.std() > 0 else np.nan,
                         "적중 시점 비율": float((ls > 0).mean())})
    return pd.DataFrame(rows), preds


def current_scores(fp: dict, D: pd.DataFrame, feats: list[str], models: list[str]) -> pd.DataFrame:
    """전체 과거(목표가 있는 행)로 학습 → 최신 주 품목별 '다음 4주 상대 강세' 점수."""
    F = fp["F"]
    dt = F["r4"].dropna(how="all").index.max()
    x = pd.DataFrame({k: F[k].loc[dt] for k in feats}).dropna()
    xr = x.rank(pct=True) - 0.5
    tr = D.dropna(subset=["y"])
    out = {}
    for mn in models:
        m = _model(mn).fit(tr[feats].values, tr["y"].values)
        out[mn] = pd.Series(_score(m, xr.values), index=xr.index)
    S = pd.DataFrame(out)
    S["평균 순위"] = S.rank(pct=True).mean(1)
    return S.sort_values("평균 순위", ascending=False), dt


def importance(D: pd.DataFrame, feats: list[str], seed=0) -> pd.Series:
    from sklearn.ensemble import RandomForestClassifier
    tr = D.dropna(subset=["y"])
    m = RandomForestClassifier(300, min_samples_leaf=20, max_features=0.5, random_state=seed, n_jobs=-1).fit(tr[feats].values, tr["y"].values)
    return pd.Series(m.feature_importances_, index=feats).sort_values()
