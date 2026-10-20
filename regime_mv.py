"""다변량 레짐 — 원자재 + 거시(금리·달러·위험·해운)를 함께 넣고 국면을 추정 · walk-forward 백테스트

  패널: 주간(금 종가, 선물 2000~) 또는 월간(세계은행 월평균 가격, 1990~). 변수별 변환
      가격·지수·환율·운임 → 로그수익률(%), 금리 → 차분(%p), 상태 변수(장단기 금리차·신용 스프레드) → 수준, VIX → 로그 수준
  다변량 가우시안 HMM: o_t ~ N(μ_s, Σ_s), s_t 마르코프 — 국면 = 원자재·거시의 '결합 분포'가 다른 구간
  walk-forward: 매년 1월, 그 전 자료만으로 표준화·EM 추정 → 모수 고정 후 전방 필터 P(s_t | o_1..t) (평활 확률 미사용)
      다음 기간 예측 = (P(s_t | ·) · A) · μ_Y,s   (μ_Y,s = 레짐별 대상 품목 평균 수익률, 학습 구간 추정)
  전략: 롱/숏 = 부호, 롱/현금 = 예측 > 0 일 때만 보유. 비교: Y만(단변량) · 원자재만 · Y+거시 · 원자재+거시 (같은 표본)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent

# id: (짧은 이름, 변환)
MACRO = {"fred:DTB3": ("미 3개월 금리 Δ", "diff"), "fred:DGS2": ("미 2년 금리 Δ", "diff"), "fred:DGS10": ("미 10년 금리 Δ", "diff"),
         "spread:10y2y": ("장단기 금리차 (수준)", "level"), "fred:BAA10Y": ("신용 스프레드 (수준)", "level"),
         "fred:T10YIE": ("기대인플레 Δ", "diff"), "fred:VIXCLS": ("VIX (로그 수준)", "loglevel"),
         "yf:DX-Y.NYB": ("달러 인덱스", "ret"), "fred:DTWEXBGS": ("달러 지수 (광의)", "ret"), "yf:KRW=X": ("원/달러", "ret"),
         "yf:CNY=X": ("위안/달러", "ret"), "yf:^GSPC": ("S&P 500", "ret"), "ak:macro_shipping_bdi": ("BDI", "ret"),
         "ak:macro_shipping_bpi": ("BPI (파나막스)", "ret"), "ak:macro_shipping_bci": ("BCI (케이프)", "ret")}
MACRO_DEFAULT = ["fred:DTB3", "fred:DGS10", "spread:10y2y", "fred:BAA10Y", "fred:VIXCLS", "yf:DX-Y.NYB", "yf:^GSPC", "ak:macro_shipping_bdi"]
COMM_W_DEFAULT = ["yf:ZC=F", "yf:ZW=F", "yf:ZS=F", "yf:CL=F", "yf:HG=F", "yf:GC=F"]
COMM_M_DEFAULT = ["wb:Maize", "wb:Wheat, US HRW", "wb:Soybeans", "wb:Crude oil, Brent", "wb:Copper", "wb:Gold"]
SETS = ["Y만 (단변량)", "원자재만", "Y + 거시", "원자재 + 거시"]
ANN = {"W": 52, "M": 12}
MIN_TRAIN = {"W": 260, "M": 120}


def load_wb() -> pd.DataFrame:
    """세계은행 Pink Sheet 월평균 가격 (wide, 월말 인덱스)."""
    p = ROOT / "data" / "monthly.parquet"
    if not p.exists():
        return pd.DataFrame()
    m = pd.read_parquet(p)
    w = m.pivot_table(index="date", columns="id", values="value", aggfunc="last").sort_index()
    w.index = pd.to_datetime(w.index) + pd.offsets.MonthEnd(0)
    return w


def wb_names() -> dict:
    p = ROOT / "data" / "monthly.parquet"
    if not p.exists():
        return {}
    m = pd.read_parquet(p, columns=["id", "name"]).drop_duplicates("id")
    return dict(zip(m.id, m.name))


def build_panel(L: pd.DataFrame, comm: list[str], macro: list[str], freq: str = "W", start="1990-01-01",
                WB: pd.DataFrame | None = None) -> pd.DataFrame:
    """L = 일간 레벨(wide). 반환 = 변환된 공통 표본 패널 (열 순서 = 원자재 → 거시)."""
    rule = "W-FRI" if freq == "W" else "ME"
    cols = {}

    def daily(i):
        if i == "spread:10y2y":
            return (L["fred:DGS10"] - L["fred:DGS2"]).dropna().resample(rule).last()
        return L[i].dropna().resample(rule).last()
    for i in comm:
        if i.startswith("wb:"):
            if freq != "M" or WB is None or i not in WB:
                continue
            s = WB[i].dropna()
        elif i in L:
            s = daily(i)
        else:
            continue
        s = s[s > 0]
        cols[i] = 100 * np.log(s).diff()
    for i in macro:
        if i != "spread:10y2y" and i not in L:
            continue
        if i == "spread:10y2y" and not {"fred:DGS10", "fred:DGS2"} <= set(L.columns):
            continue
        s = daily(i); k = MACRO.get(i, ("", "ret"))[1]
        cols[i] = {"ret": lambda x: 100 * np.log(x[x > 0]).diff(), "diff": lambda x: x.diff(), "level": lambda x: x,
                   "loglevel": lambda x: np.log(x[x > 0])}[k](s)
    X = pd.DataFrame(cols)
    X = X[X.index >= pd.Timestamp(start)].dropna()
    return X


# ============================================================================
# HMM 적합 · 전방 필터
# ============================================================================
def fit_hmm(Z: np.ndarray, K: int = 3, cov: str = "full", n_init: int = 3, seed: int = 0):
    from hmmlearn.hmm import GaussianHMM
    best, sc_b = None, -np.inf
    for r in range(n_init):
        m = GaussianHMM(n_components=K, covariance_type=cov, n_iter=200, tol=1e-3, random_state=seed + r, min_covar=1e-3)
        try:
            m.fit(Z); sc = m.score(Z)
        except Exception:
            continue
        if np.isfinite(sc) and sc > sc_b:
            best, sc_b = m, sc
    if best is None:
        raise RuntimeError("HMM 추정 실패")
    return best


def filter_probs(m, Z: np.ndarray) -> np.ndarray:
    """전방 필터 P(s_t | o_1..t) — 모수 고정, 미래 관측 미사용."""
    from scipy.stats import multivariate_normal
    K = m.n_components; C = m.covars_
    lb = np.column_stack([multivariate_normal.logpdf(Z, m.means_[k], C[k], allow_singular=True) for k in range(K)])
    A = m.transmat_; F = np.empty((len(Z), K))
    a = m.startprob_ * np.exp(lb[0] - lb[0].max()); a /= a.sum(); F[0] = a
    for t in range(1, len(Z)):
        a = (a @ A) * np.exp(lb[t] - lb[t].max()); s = a.sum()
        a = a / s if s > 0 and np.isfinite(s) else np.full(K, 1 / K); F[t] = a
    return F


def _order(m, j_y: int, sd_y: float, mu_y: float):
    """레짐 순서 = 대상 품목 평균 수익률 오름차순 (레짐1 = Y 약세)."""
    my = m.means_[:, j_y] * sd_y + mu_y
    return np.argsort(my), my


def describe(X: pd.DataFrame, target: str, K: int = 3, cov: str = "full", n_init: int = 3, seed: int = 0) -> dict:
    """전 기간 추정 (설명용 — 평활 확률). 레짐 프로파일(변수별 평균 z)·전이행렬·지속기간·현재 필터 확률."""
    mu, sd = X.mean(), X.std().replace(0, 1)
    Z = ((X - mu) / sd).values
    m = fit_hmm(Z, K, cov, n_init, seed)
    j = list(X.columns).index(target)
    o, my = _order(m, j, sd[target], mu[target])
    lab = [f"레짐{k + 1}" for k in range(K)]
    G = pd.DataFrame(m.predict_proba(Z)[:, o], index=X.index, columns=lab)
    F = pd.DataFrame(filter_probs(m, Z)[:, o], index=X.index, columns=lab)
    prof = pd.DataFrame(m.means_[o], index=lab, columns=X.columns)
    T = pd.DataFrame(m.transmat_[np.ix_(o, o)], index=lab, columns=lab)
    vol_y = np.sqrt(np.array([m.covars_[k][j, j] for k in o])) * sd[target]
    summ = pd.DataFrame({"Y 평균 수익률": my[o], "Y 변동성": vol_y, "유지확률": np.diag(T.values),
                         "기대 지속 (기간)": 1 / (1 - np.clip(np.diag(T.values), 0, 0.999)), "비중": G.mean().values}, index=lab)
    return dict(smooth=G, filt=F, profile=prof, trans=T, summary=summ, names=auto_names(prof))


def auto_names(prof: pd.DataFrame, top: int = 2) -> dict:
    """레짐별 가장 두드러진 변수 2개로 이름: 'VIX↑ · 신용 스프레드↑'."""
    out = {}
    for k, r in prof.iterrows():
        c = r.abs().sort_values(ascending=False).index[:top]
        out[k] = " · ".join(f"{label(i)}{'↑' if r[i] > 0 else '↓'}" for i in c)
    return out


def label(i: str) -> str:
    if i in MACRO:
        return MACRO[i][0].split(" (")[0].replace(" Δ", "")
    try:
        import procurement as PR
        if i in PR.BUY_IDS:
            return PR.BUY_IDS[i]
    except Exception:
        pass
    return i.split(":", 1)[-1]


# ============================================================================
# walk-forward
# ============================================================================
def walk_forward(X: pd.DataFrame, target: str, K: int = 3, cov: str = "full", freq: str = "W", n_init: int = 2,
                 seed: int = 0, min_train: int | None = None) -> pd.DataFrame:
    """매년 1월 재추정 → 그해 각 시점 t의 (필터 레짐 확률, 다음 기간 예측 E_t). 반환 index = X.index (학습 전은 NaN)."""
    min_train = min_train or MIN_TRAIN[freq]
    idx = X.index; j = list(X.columns).index(target)
    out = pd.DataFrame(np.nan, index=idx, columns=["E"] + [f"p{k + 1}" for k in range(K)] + ["state"])
    for y in sorted(set(idx.year)):
        tr = idx < pd.Timestamp(f"{y}-01-01")
        if tr.sum() < min_train:
            continue
        te = idx.year == y
        Xa = X[tr | te]
        mu, sd = X[tr].mean(), X[tr].std().replace(0, 1)
        Za = ((Xa - mu) / sd).values; ntr = int(tr.sum())
        try:
            m = fit_hmm(Za[:ntr], K, cov, n_init, seed)
        except RuntimeError:
            continue
        o, my = _order(m, j, sd[target], mu[target])
        F = filter_probs(m, Za)[:, o]
        A = m.transmat_[np.ix_(o, o)]; mys = my[o]
        Ft = F[ntr:]
        out.loc[te, "E"] = (Ft @ A) @ mys
        out.loc[te, [f"p{k + 1}" for k in range(K)]] = Ft
        out.loc[te, "state"] = Ft.argmax(1) + 1
    return out


def evaluate(W: pd.DataFrame, y: pd.Series, ann: int, n_placebo: int = 300, block: int = 13, seed: int = 0) -> dict:
    """W = walk_forward 결과, y = 대상 품목 수익률(%). 예측 E_t → 다음 기간 y_{t+1}."""
    yn = y.shift(-1).reindex(W.index)
    m = W["E"].notna() & yn.notna()
    e, r = W.loc[m, "E"].values.astype(float), yn[m].values
    if len(e) < 30:
        return {}
    sh = lambda x: float(np.mean(x) / np.std(x) * np.sqrt(ann)) if np.std(x) > 0 else np.nan
    pos = np.sign(e); pos_lf = (e > 0).astype(float)
    bench = y.expanding().mean().reindex(W.index)[m].values                        # 그 시점까지 과거 평균
    oos_r2 = 1 - np.sum((r - e) ** 2) / np.sum((r - bench) ** 2)
    st = W.loc[m, "state"].values
    K = int(np.nanmax(st))
    lo, hi = r[st == 1], r[st == K]
    t_ = (hi.mean() - lo.mean()) / np.sqrt(hi.var(ddof=1) / len(hi) + lo.var(ddof=1) / len(lo)) if len(lo) > 5 and len(hi) > 5 else np.nan
    rng = np.random.default_rng(seed); nb = int(np.ceil(len(pos) / block)); pl = []
    for _ in range(n_placebo):                                     # 위약: 포지션 블록을 무작위로 섞음 (롱 비중·지속성 유지)
        b = rng.permutation(nb)
        pp = np.concatenate([pos[k * block:(k + 1) * block] for k in b])[:len(pos)]
        pl.append(sh(pp * r))
    return {"판단 수": len(e), "롱 비중": float((pos > 0).mean()), "롱숏 Sharpe": sh(pos * r), "롱/현금 Sharpe": sh(pos_lf * r),
            "B&H Sharpe": sh(r), "방향 적중률": float(np.mean(np.sign(e) == np.sign(r))), "OOS R²": float(oos_r2),
            "최상−최하 레짐 수익 차": float(hi.mean() - lo.mean()) if len(hi) and len(lo) else np.nan, "레짐 차 t": float(t_),
            "위약 95%": float(np.nanquantile(pl, 0.95)), "위약 p": float(np.mean(np.array(pl) >= sh(pos * r))),
            "전환 횟수/년": float(np.sum(np.diff(st) != 0) / len(st) * ann)}


def set_columns(X: pd.DataFrame, target: str, comm: list[str], macro: list[str]) -> dict:
    c = [i for i in comm if i in X]; mc = [i for i in macro if i in X]
    return {"Y만 (단변량)": [target], "원자재만": list(dict.fromkeys([target] + c)), "Y + 거시": [target] + mc,
            "원자재 + 거시": list(dict.fromkeys([target] + c + mc))}


def compare_sets(X: pd.DataFrame, target: str, comm: list[str], macro: list[str], K: int = 3, cov: str = "full", freq: str = "W",
                 n_init: int = 2, split: str = "2017-01-01") -> tuple[pd.DataFrame, dict]:
    """같은 표본에서 4개 변수 묶음의 walk-forward 성과. 반환 (표, {묶음: walk_forward 결과})."""
    rows, Ws = [], {}
    for nm, cols in set_columns(X, target, comm, macro).items():
        if nm != "Y만 (단변량)" and len(cols) == 1:
            continue
        W = walk_forward(X[cols], target, K, cov, freq, n_init)
        Ws[nm] = W
        for per, msk in [("전체 OOS", slice(None)), (f"~{split[:4]}", W.index < pd.Timestamp(split)), (f"{split[:4]}~", W.index >= pd.Timestamp(split))]:
            S = evaluate(W[msk] if not isinstance(msk, slice) else W, X[target], ANN[freq])
            if S:
                rows.append({"변수 묶음": nm, "기간": per, "변수 수": len(cols), **S})
    return pd.DataFrame(rows), Ws
