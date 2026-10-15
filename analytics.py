"""원자재 시계열 분석 엔진 (UI와 분리된 순수 함수).

동조성   : 상관행렬, 롤링 상관, PCA, 흡수비율(Absorption Ratio), Diebold-Yilmaz 스필오버
변동성   : GARCH / GJR-GARCH / EGARCH (레버리지 효과), 레버리지 상관, Markov-switching
회귀·ML  : OLS, Ridge, Lasso, Elastic Net, Adaptive Lasso, SCAD, MCP, 분위수 회귀,
           Kalman TVP 회귀, Random Forest, Gradient Boosting, LSTM
요인모형 : FAVAR (Bernanke-Boivin-Eliasz 2단계 추정)
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
DATA = Path(__file__).parent / "data"

# 위기 구간 (시작, 끝) — 하이라이트 및 국면별 비교에 사용
CRISES = {
    "닷컴 붕괴": ("2000-03-10", "2002-10-09"),
    "글로벌 금융위기": ("2007-08-01", "2009-06-30"),
    "유럽 재정위기": ("2011-07-01", "2012-06-30"),
    "유가 폭락": ("2014-07-01", "2016-02-29"),
    "코로나": ("2020-02-15", "2020-06-30"),
    "러-우 전쟁·인플레": ("2022-02-15", "2022-12-31"),
}
FREQ = {"일간": "B", "주간": "W-FRI", "월간": "ME"}
ANN = {"B": 252, "W-FRI": 52, "ME": 12}


# ============================================================================
# 데이터
# ============================================================================
def _daily_path(roll_adjusted: bool = False) -> Path:
    """roll_adjusted=True: 롤오버 갭 보정본(roll_adjust.py) → 정제본(clean.py) → 원본 순으로 사용.
    수익률·변동성·상관·예측 분석은 보정본, 가격 '수준' 표시는 정제본을 쓰는 것이 원칙."""
    if roll_adjusted and (DATA / "daily_adj.parquet").exists():
        return DATA / "daily_adj.parquet"
    p = DATA / "daily_clean.parquet"
    return p if p.exists() else DATA / "daily.parquet"


def load_catalog() -> pd.DataFrame:
    d = pd.read_parquet(_daily_path())
    cat = d.groupby("id").agg(name=("name", "first"), group=("group", "first"), unit=("unit", "first"),
                              source=("source", "first"), start=("date", "min"), end=("date", "max"),
                              n=("value", "size")).reset_index()
    return cat


def load_levels(extra: pd.DataFrame | None = None, roll_adjusted: bool = True) -> pd.DataFrame:
    """id × date 레벨 패널 (wide). extra: 사용자 업로드 wide DataFrame(date index).
    roll_adjusted=True(기본): 연속월물 롤오버 갭 보정 가격 사용 (분석용)."""
    d = pd.read_parquet(_daily_path(roll_adjusted))
    w = d.pivot_table(index="date", columns="id", values="value", aggfunc="last").sort_index()
    w.index = pd.to_datetime(w.index)
    if extra is not None and len(extra):
        w = w.join(extra, how="outer")
    return w


def is_rate(unit: str) -> bool:
    return str(unit).strip() == "%"


def log_return(s: pd.Series, unit: str = "") -> pd.Series:
    """금리·스프레드(%) -> 단순차분. 가격 -> 100×로그차분.
    음수·0 가격(WTI 2020-04-20, BCI 2020년 1분기 등)이 일부 있으면 그 시점의 수익률만 결측으로 둔다.
    (이전 버전은 음수가 하나라도 있으면 시리즈 전체를 가격차분으로 계산 -> WTI가 $ 차분으로 분석되던 문제 수정)"""
    if is_rate(unit):
        return s.diff()
    pos = s.where(s > 0)
    if pos.notna().sum() < 0.9 * s.notna().sum():     # 음수가 흔한 시리즈(스프레드 등)는 차분
        return s.diff()
    return 100 * np.log(pos).diff()


def make_panel(levels: pd.DataFrame, ids: list[str], units: dict, freq: str = "B",
               transform: str = "logret", start=None, end=None, ffill_limit: int = 5) -> pd.DataFrame:
    """레벨 -> 분석용 패널.

    transform: 'logret'(로그수익률 %, 금리류는 차분), 'level', 'loglevel'
    - 거래일 불일치(휴장)는 ffill(limit)로 메운 뒤 공통 구간만 남김.
    """
    X = levels[ids].copy()
    X = X.ffill(limit=ffill_limit)
    if freq != "B":
        X = X.resample(freq).last()
    else:
        X = X[X.index.dayofweek < 5]
    if transform == "logret":
        out = {}
        for c in ids:
            out[c] = log_return(X[c], units.get(c, ""))
        X = pd.DataFrame(out)
    elif transform == "loglevel":
        X = X.apply(lambda s: np.log(s) if (s > 0).all() else s)
    if start is not None:
        X = X[X.index >= pd.Timestamp(start)]
    if end is not None:
        X = X[X.index <= pd.Timestamp(end)]
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    # 연속월물 롤오버/데이터 오류에 의한 극단값 완화 (±15σ 클리핑, 로그수익률일 때만)
    if transform == "logret" and len(X) > 50:
        sd = X.expanding(min_periods=50).std().shift(1)      # t-1까지의 정보만 (초기 50개는 클리핑 없음)
        X = X.clip(lower=-15 * sd, upper=15 * sd)
    return X


def crisis_mask(index: pd.DatetimeIndex, names: list[str]) -> pd.Series:
    m = pd.Series(False, index=index)
    for n in names:
        a, b = CRISES[n]
        m |= (index >= pd.Timestamp(a)) & (index <= pd.Timestamp(b))
    return m


# ============================================================================
# 동조성
# ============================================================================
def avg_pairwise_corr(R: pd.DataFrame, window: int) -> pd.Series:
    """롤링 평균 쌍별 상관 (비대각 평균)."""
    n = R.shape[1]
    rc = R.rolling(window).corr()
    vals = rc.groupby(level=0).apply(lambda m: (np.nansum(m.values) - n) / (n * (n - 1)) if m.notna().all().all() else np.nan)
    return vals


def pca(R: pd.DataFrame, k: int | None = None):
    """표준화 수익률의 PCA. return (explained_ratio, loadings DataFrame, scores DataFrame)."""
    Z = (R - R.mean()) / R.std(ddof=0)
    C = np.corrcoef(Z.values, rowvar=False)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    # 부호 정규화: 적재값 합이 양수가 되도록
    V = V * np.sign(V.sum(axis=0, keepdims=True) + 1e-12)
    k = k or len(w)
    cols = [f"PC{i+1}" for i in range(k)]
    load = pd.DataFrame(V[:, :k], index=R.columns, columns=cols)
    scores = pd.DataFrame(Z.values @ V[:, :k], index=R.index, columns=cols)
    return w / w.sum(), load, scores


def rolling_absorption(R: pd.DataFrame, window: int, k: int = 1, step: int = 5) -> pd.Series:
    """Kritzman et al. (2011) Absorption Ratio: 상위 k개 고유값이 설명하는 분산 비중."""
    vals, idx = [], []
    A = R.values
    for t in range(window, len(R) + 1, step):
        C = np.corrcoef(A[t - window:t], rowvar=False)
        if not np.isfinite(C).all():
            continue
        w = np.sort(np.linalg.eigvalsh(C))[::-1]
        vals.append(w[:k].sum() / w.sum())
        idx.append(R.index[t - 1])
    return pd.Series(vals, index=idx)


def _gfevd(res, H: int) -> np.ndarray:
    """Pesaran-Shin 일반화 예측오차 분산분해 (행 정규화). theta[i,j] = j의 충격이 i 분산에 기여하는 비중."""
    Sigma = np.asarray(res.sigma_u)
    Phi = res.ma_rep(H - 1)  # (H, N, N)
    N = Sigma.shape[0]
    num = np.zeros((N, N)); den = np.zeros(N)
    for h in range(H):
        A = Phi[h] @ Sigma
        num += A ** 2
        den += np.diag(Phi[h] @ Sigma @ Phi[h].T)
    theta = num / np.diag(Sigma)[None, :] / den[:, None]
    return theta / theta.sum(axis=1, keepdims=True)


def dy_spillover(R: pd.DataFrame, p: int = 2, H: int = 10):
    """Diebold-Yilmaz (2012) 스필오버 표. return (table %, total index %)."""
    from statsmodels.tsa.api import VAR
    res = VAR(R.values).fit(p)
    th = _gfevd(res, H) * 100
    N = th.shape[0]
    tbl = pd.DataFrame(th, index=R.columns, columns=R.columns)
    off = th.sum() - np.trace(th)
    tbl["FROM (타 품목→)"] = th.sum(axis=1) - np.diag(th)
    to = th.sum(axis=0) - np.diag(th)
    tbl.loc["TO (→타 품목)"] = list(to) + [off / N]
    tbl.loc["NET (TO-FROM)"] = list(to - (th.sum(axis=1) - np.diag(th))) + [np.nan]
    return tbl, off / N


def rolling_dy(R: pd.DataFrame, window: int, p: int = 2, H: int = 10, step: int = 10) -> pd.Series:
    from statsmodels.tsa.api import VAR
    vals, idx = [], []
    A = R.values
    for t in range(window, len(R) + 1, step):
        try:
            res = VAR(A[t - window:t]).fit(p)
            th = _gfevd(res, H) * 100
            vals.append((th.sum() - np.trace(th)) / th.shape[0]); idx.append(R.index[t - 1])
        except Exception:
            pass
    return pd.Series(vals, index=idx)


# ============================================================================
# 변동성 · 레버리지 효과
# ============================================================================
def garch_fit(r: pd.Series, kind: str = "GJR", dist: str = "t", ann: int = 252):
    """kind: GARCH | GJR | EGARCH. r은 % 수익률.

    레버리지 해석
      GJR  : gamma>0  -> 음(-)의 충격이 변동성을 더 키움 (전통적 레버리지 효과)
             gamma<0  -> 양(+)의 충격이 변동성을 더 키움 (원자재의 '역레버리지', 공급충격형)
      EGARCH: gamma<0 -> 레버리지 효과, gamma>0 -> 역레버리지
    """
    from arch import arch_model
    if kind == "GARCH":
        am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist=dist)
    elif kind == "GJR":
        am = arch_model(r, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist=dist)
    else:
        am = arch_model(r, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist=dist)
    res = am.fit(disp="off", options={"maxiter": 500})
    pt = pd.DataFrame({"계수": res.params, "표준오차": res.std_err, "t값": res.tvalues, "p값": res.pvalues})
    p = res.params
    if kind == "EGARCH":
        persist = p.get("beta[1]", np.nan)
    else:
        persist = p.get("alpha[1]", 0) + p.get("beta[1]", 0) + 0.5 * p.get("gamma[1]", 0)
    half = np.log(0.5) / np.log(persist) if 0 < persist < 1 else np.nan
    vol = res.conditional_volatility * np.sqrt(ann)
    info = dict(persistence=persist, half_life=half, loglik=res.loglikelihood, aic=res.aic, bic=res.bic,
                gamma=p.get("gamma[1]", np.nan), gamma_p=res.pvalues.get("gamma[1]", np.nan))
    return pt, vol, info, res


def news_impact(res, kind: str, n: int = 101):
    """뉴스 충격 곡선: 표준화되지 않은 충격 e에 대한 다음기 조건부 분산."""
    p = res.params
    s2 = float(np.mean(res.conditional_volatility ** 2))
    sd = np.sqrt(s2)
    e = np.linspace(-4 * sd, 4 * sd, n)
    if kind == "EGARCH":
        z = e / sd
        lns = p["omega"] + p["alpha[1]"] * (np.abs(z) - np.sqrt(2 / np.pi)) + p.get("gamma[1]", 0) * z + p["beta[1]"] * np.log(s2)
        h = np.exp(lns)
    else:
        h = p["omega"] + (p["alpha[1]"] + p.get("gamma[1]", 0) * (e < 0)) * e ** 2 + p["beta[1]"] * s2
    return e, h


def leverage_corr(r: pd.Series, h: int = 22) -> pd.Series:
    """corr(r_t, RV_{t+1..t+h}): 음수면 하락 후 변동성 확대(레버리지), 양수면 역레버리지.
    반환: 시점별 (r_t, 미래 실현변동성) 쌍을 담은 Series (상관은 호출 측에서 국면별 계산)."""
    rv = np.sqrt((r ** 2).rolling(h).sum().shift(-h))
    return pd.concat([r.rename("r"), rv.rename("rv_fwd")], axis=1).dropna()


def markov_switching(r: pd.Series, k: int = 2):
    """Hamilton 레짐 스위칭: 평균·분산 모두 레짐별로 다름. 레짐은 변동성 오름차순으로 재정렬."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    mod = MarkovRegression(r.values, k_regimes=k, trend="c", switching_variance=True)
    res = mod.fit(disp=False, search_reps=10)
    params = pd.Series(res.params, index=res.model.param_names)
    sig = np.array([params[f"sigma2[{i}]"] for i in range(k)])
    mu = np.array([params[f"const[{i}]"] for i in range(k)])
    order = np.argsort(sig)
    prob = pd.DataFrame(np.asarray(res.smoothed_marginal_probabilities)[:, order], index=r.index,
                        columns=[f"레짐{i+1}" for i in range(k)])
    P = np.asarray(res.regime_transition)[:, :, 0] if np.ndim(res.regime_transition) == 3 else np.asarray(res.regime_transition)
    # statsmodels: P[i,j] = P(s_t=i | s_{t-1}=j)
    stay = np.array([P[i, i] for i in range(k)])[order]
    dur = 1 / (1 - stay)
    summ = pd.DataFrame({"평균수익률": mu[order], "변동성(σ)": np.sqrt(sig[order]), "유지확률": stay,
                         "기대지속기간": dur, "비중(평활확률 평균)": prob.mean().values},
                        index=prob.columns)
    return prob, summ


# ============================================================================
# 특성 생성 (예측 변수) — 모든 값은 t 시점에 실제로 알 수 있었던 정보만 사용
# ============================================================================
PPY = {"B": 252, "W-FRI": 52, "ME": 12}
# 공표 지연(영업일): 해당 날짜 값이 실제로 공개되는 시점. 예측 설계·특성에서 이만큼 늦춰 사용
PUB_LAG_BDAYS = {"fred:DCOILWTICO": 5, "fred:DCOILBRENTEU": 5, "fred:DHHNGSP": 5}


def available(s: pd.Series, sid: str) -> pd.Series:
    """공표 지연을 반영해 '그 날짜에 실제로 알 수 있었던 값'으로 재인덱싱."""
    lag = PUB_LAG_BDAYS.get(sid, 0)
    if not lag:
        return s
    out = s.copy(); out.index = out.index + pd.offsets.BDay(lag)
    return out
COT_MAP = {"yf:ZC=F": "Corn", "yf:ZS=F": "Soybeans", "yf:ZM=F": "Soybean Meal", "yf:ZL=F": "Soybean Oil",
           "yf:ZW=F": "Wheat SRW", "yf:KE=F": "Wheat HRW", "yf:HG=F": "Copper", "yf:CL=F": "WTI Crude", "yf:GC=F": "Gold"}
BASIS_MAP = {"yf:CL=F": "fred:DCOILWTICO", "yf:BZ=F": "fred:DCOILBRENTEU", "yf:NG=F": "fred:DHHNGSP"}
FEATURE_KINDS = ["모멘텀", "실현변동성", "계절성", "COT 헤징압력", "베이시스(캐리)"]


def _to_freq(s: pd.Series, freq: str, ffill_limit: int = 5) -> pd.Series:
    s = s.ffill(limit=ffill_limit)
    return s[s.index.dayofweek < 5] if freq == "B" else s.resample(freq).last()


def make_features(levels: pd.DataFrame, target: str, units: dict, freq: str, index: pd.DatetimeIndex,
                  kinds: list[str]) -> pd.DataFrame:
    """예측 시점 t의 특성 (index = 패널 날짜).

    모멘텀       : 과거 1·3·6·12개월 누적 로그수익률 (시계열 모멘텀, Moskowitz-Ooi-Pedersen 2012) + 12-1개월
    실현변동성   : 과거 약 1개월(일간 21 / 주간 13 / 월간 6) 수익률 표준편차, 연율
    계절성       : 다음 기간의 연중 위치에 대한 Fourier 항 sin/cos (1·2차) — 월 더미보다 모수가 적음
    COT 헤징압력 : 상업(Producer/Merchant) 순포지션/미결제약정, Managed Money 순포지션/미결제약정의 3년 z-score.
                   화요일 기준·금요일 장 마감 후 발표 → 다음 주 월요일부터 사용 (미래정보 차단)
    베이시스     : 100×ln(현물/근월물) — 에너지(EIA 현물)만. 현물 프리미엄(백워데이션)일수록 +, 캐리 대용치
    """
    ppy = PPY[freq]
    px = _to_freq(levels[target], freq)
    r = log_return(px, units.get(target, ""))
    out = {}
    if "모멘텀" in kinds:
        for m in (1, 3, 6, 12):
            out[f"MOM {m}M"] = r.rolling(max(1, round(ppy * m / 12))).sum()
        out["MOM 12-1M"] = out["MOM 12M"] - out["MOM 1M"]
    if "실현변동성" in kinds:
        w = {"B": 21, "W-FRI": 13, "ME": 6}[freq]
        out["RVOL 1M"] = r.rolling(w).std() * np.sqrt(ppy)
    F = pd.DataFrame(out, index=px.index)
    if "계절성" in kinds:
        step = {"B": pd.Timedelta(days=1), "W-FRI": pd.Timedelta(days=7), "ME": pd.Timedelta(days=15)}[freq]
        doy = (index + step).dayofyear.values / 365.25
        S = pd.DataFrame({f"SEAS sin{k}": np.sin(2 * np.pi * k * doy) for k in (1, 2)} |
                         {f"SEAS cos{k}": np.cos(2 * np.pi * k * doy) for k in (1, 2)}, index=index)
        F = F.reindex(index).join(S) if len(F.columns) else S
    if "COT 헤징압력" in kinds and target in COT_MAP and (DATA / "cot.parquet").exists():
        c = pd.read_parquet(DATA / "cot.parquet")
        c = c[c["name"] == COT_MAP[target]].copy()
        c["date"] = pd.to_datetime(c["date"])
        c = c.sort_values("date")
        hp = c["comm_net"] / c["open_interest_all"]; mm = c["mm_net"] / c["open_interest_all"]
        z = lambda s: (s - s.rolling(156, min_periods=52).mean()) / s.rolling(156, min_periods=52).std()
        cot = pd.DataFrame({"COT 상업HP z": z(hp).values, "COT MM z": z(mm).values},
                           index=c["date"] + pd.Timedelta(days=6))           # 화 + 6일 = 다음 주 월요일부터 이용 가능
        cot = cot[~cot.index.duplicated(keep="last")]
        F = F.reindex(index).join(cot.reindex(index, method="ffill")) if len(F.columns) else cot.reindex(index, method="ffill")
    if "베이시스(캐리)" in kinds and target in BASIS_MAP and BASIS_MAP[target] in levels:
        spot = _to_freq(available(levels[BASIS_MAP[target]], BASIS_MAP[target]), freq)
        b = 100 * np.log(spot.where(spot > 0) / px.where(px > 0))
        bF = pd.DataFrame({"BASIS 현물-선물(%)": b, "BASIS 변화": b.diff()})
        F = F.reindex(index).join(bF.reindex(index)) if len(F.columns) else bF.reindex(index)
    return F.reindex(index)


def stationarity_table(levels: pd.DataFrame, ids: list[str], units: dict, freq: str, start=None, end=None,
                       names: dict | None = None) -> pd.DataFrame:
    """로그가격 vs 수익률의 ADF(H0 단위근)·KPSS(H0 정상), 수익률 자기상관(Ljung-Box)·ARCH 효과."""
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
    LV = make_panel(levels, ids, units, freq, "loglevel", start, end)
    RT = make_panel(levels, ids, units, freq, "logret", start, end)
    rows = []
    for i in ids:
        if i not in LV or len(RT) < 40:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows.append({"품목": (names or {}).get(i, i),
                         "레벨 ADF p": adfuller(LV[i].dropna(), autolag="AIC")[1], "레벨 KPSS p": kpss(LV[i].dropna(), nlags="auto")[1],
                         "수익률 ADF p": adfuller(RT[i], autolag="AIC")[1], "수익률 KPSS p": kpss(RT[i], nlags="auto")[1],
                         "자기상관 LB(10) p": acorr_ljungbox(RT[i], [10])["lb_pvalue"].iloc[0],
                         "ARCH 효과 p": het_arch(RT[i], nlags=10)[1]})
    T = pd.DataFrame(rows).set_index("품목") if rows else pd.DataFrame()
    if len(T):
        T["판정"] = np.where((T["레벨 KPSS p"] < 0.05) & (T["수익률 ADF p"] < 0.05) & (T["수익률 KPSS p"] > 0.05),
                           "레벨 비정상 · 수익률 정상 (차분으로 충분)", "추가 확인 필요")
    return T


def resid_diagnostics(resid: np.ndarray, lags: int = 10) -> dict:
    """잔차 진단: 자기상관(Ljung-Box), 조건부 이분산(ARCH-LM), 정규성(Jarque-Bera), 정상성(ADF)."""
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
    from scipy import stats
    e = np.asarray(resid, float); e = e[np.isfinite(e)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {"Ljung-Box(10) p — 자기상관": float(acorr_ljungbox(e, [lags])["lb_pvalue"].iloc[0]),
                "ARCH-LM p — 이분산": float(het_arch(e, nlags=lags)[1]),
                "Jarque-Bera p — 정규성": float(stats.jarque_bera(e).pvalue),
                "ADF p — 단위근": float(adfuller(e, autolag="AIC")[1]),
                "첨도(초과)": float(stats.kurtosis(e)), "왜도": float(stats.skew(e))}


# ============================================================================
# 회귀 · 머신러닝
# ============================================================================
@dataclass
class Design:
    X_tr: np.ndarray; y_tr: np.ndarray; X_te: np.ndarray; y_te: np.ndarray
    idx_tr: pd.DatetimeIndex; idx_te: pd.DatetimeIndex
    names: list; mu: np.ndarray; sd: np.ndarray; y_mu: float
    raw: pd.DataFrame = field(default=None)
    horizon: int = 1
    mode: str = "explain"
    cost: float = 0.0        # 편도 거래비용 (% 단위, 5bp = 0.05)
    target: str = ""         # 종속변수 id (Minnesota 사전분포 등에서 자기시차 식별)


def build_design(R: pd.DataFrame, target: str, features: list[str], mode: str = "explain",
                 lags: int = 1, horizon: int = 1, own_lags: bool = True, train_frac: float = 0.7,
                 extra: pd.DataFrame | None = None, vol_scale: int | None = None, xlag: int = 0) -> Design:
    """mode='explain' : y_t ~ X_t (동시점, 베타/동조성 분석)
       mode='predict' : y_{t+h} ~ X_t, X_{t-1}, ..., X_{t-lags+1} (+ 자기 시차) — 미래정보 누수 없음
    표준화는 학습구간 통계로만 수행."""
    cols = {}
    if mode == "explain":
        for f in features:
            if f != target:
                cols[f] = R[f]
        y = R[target]
    else:
        base = [f for f in features if f != target] + ([target] if own_lags else [])
        step = float(np.median(np.diff(R.index.values).astype("timedelta64[D]").astype(float))) if len(R) > 2 else 1.0
        for f in base:
            extra_lag = xlag if f != target else 0     # 타 시장은 xlag만큼 추가 지연 (마감시각 차이 누수 방지)
            if f in PUB_LAG_BDAYS:                      # 공표 지연 (예: EIA 현물 5영업일 → 일간 5, 주·월간 1기간)
                lagb = PUB_LAG_BDAYS[f]
                extra_lag += lagb if step <= 1.5 else int(np.ceil(lagb * 7 / 5 / step))   # 일간 패널은 영업일 단위
            for L in range(lags):
                cols[f"{f}" + (f" (t-{L + extra_lag})" if L + extra_lag else " (t)")] = R[f].shift(L + extra_lag)
        y = R[target].shift(-horizon)
        if horizon > 1:  # h기간 누적수익률
            y = R[target].rolling(horizon).sum().shift(-horizon)
    D = pd.DataFrame(cols)
    if extra is not None and len(extra.columns):     # 특성(모멘텀·계절성·COT·베이시스): t 시점 값
        D = D.join(extra.reindex(D.index))
    if vol_scale:                                     # 목표를 t 시점까지의 실현변동성으로 나눔 (이분산 제거, 위험 균등화)
        y = y / R[target].rolling(vol_scale).std()
    D["__y"] = y
    D = D.dropna()
    n_tr = int(len(D) * train_frac)
    tr, te = D.iloc[:n_tr], D.iloc[n_tr:]
    Xtr = tr.drop(columns="__y").values; Xte = te.drop(columns="__y").values
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    y_mu = tr["__y"].mean()
    return Design((Xtr - mu) / sd, tr["__y"].values - y_mu, (Xte - mu) / sd, te["__y"].values - y_mu,
                  tr.index, te.index, list(D.columns[:-1]), mu, sd, y_mu, D, horizon if mode == "predict" else 1, mode, 0.0, target)


def strategy_stats(p_raw, y_raw, ann: int, h: int = 1, cost: float = 0.0) -> dict:
    """부호(롱/숏) 타이밍 전략 성과 — 단일 자산, 매 기간 ±1 단위(명목 100%) 포지션.

    포지션_t = sign(예측 수익률_t+h)  (+1 매수 / −1 매도 / 0 무포지션)
    전략수익_t = 포지션_t × 실현 수익률_t+h  (선물 수익률 = 초과수익률이므로 무위험금리 차감 없음)
    리밸런싱 = 데이터 빈도(일·주·월)마다, h>1이면 h기마다(겹치지 않는 표본 [::h])
    비용     = |포지션 변화| × 편도비용 (청산+진입 반전 = 2단위)
    Sharpe   = 평균/표준편차 × √(연간 리밸런싱 횟수 = 연간기간수/h)
    포트폴리오 최적화·자본 배분·레버리지 조정 없음 (자산 간 비중 결정이 아닌 단일 자산 방향 타이밍)."""
    yn = np.asarray(y_raw, float)[::h]; pn = np.asarray(p_raw, float)[::h]
    pos = np.sign(pn)
    gross = pos * yn
    turn = np.abs(np.diff(np.r_[0.0, pos]))
    net = gross - turn * cost
    sr = lambda x: float(x.mean() / x.std() * np.sqrt(ann / h)) if len(x) > 1 and x.std() > 0 else np.nan
    act = pos != 0
    return {"방향 적중률": float(np.mean(np.sign(pn[act]) == np.sign(yn[act]))) if act.any() else np.nan,
            "부호전략 Sharpe": sr(gross), "부호전략 Sharpe (비용 후)": sr(net), "Buy&Hold Sharpe": sr(yn),
            "롱 비중": float(np.mean(pos > 0)), "연간 회전율": float(turn.mean() * ann / h)}


def _metrics(y_tr, p_tr, y_te, p_te, ann, h=1, D=None):
    """y·p는 학습평균으로 중심화된 값. 전략 지표는 원래 수준(+학습평균)으로 되돌려 계산.
    설명(동시점) 모드는 Xₜ와 Yₜ가 같은 기간이라 실제 거래 불가 → 전략 지표를 계산하지 않음."""
    def r2(y, p):
        sst = ((y - y.mean()) ** 2).sum() if len(y) else np.nan
        return 1 - ((y - p) ** 2).sum() / sst if sst else np.nan
    # OOS R² (Campbell-Thompson): 기준예측 = 학습구간 평균(=0, y는 학습평균으로 중심화됨)
    oos = 1 - ((y_te - p_te) ** 2).sum() / (y_te ** 2).sum() if len(y_te) else np.nan
    out = {"R² (학습)": r2(y_tr, p_tr), "R² (테스트)": r2(y_te, p_te), "OOS R² (vs 학습평균)": oos,
           "RMSE (테스트)": float(np.sqrt(np.mean((y_te - p_te) ** 2))) if len(y_te) else np.nan}
    tradable = D is None or getattr(D, "mode", "predict") == "predict"
    ym = D.y_mu if D is not None else 0.0
    cost = getattr(D, "cost", 0.0) if D is not None else 0.0
    st = strategy_stats(np.asarray(p_te) + ym, np.asarray(y_te) + ym, ann, h, cost)
    if not tradable:
        st = {k: np.nan for k in st}
    out.update(st)
    return out


def _wf_one(k, model, Xall, yall, idx, tr_end, step, names, h, mode, cost, target, ann, hp):
    """walk-forward 한 시점: [0, tr_end) 학습 → [k, k+step) 예측 (병렬 작업 단위)."""
    Xtr, ytr = Xall[:tr_end], yall[:tr_end]
    Xte, yte = Xall[k:k + step], yall[k:k + step]
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    ym = ytr.mean()
    Dk = Design((Xtr - mu) / sd, ytr - ym, (Xte - mu) / sd, yte - ym, idx[:tr_end], idx[k:k + step],
                names, mu, sd, ym, None, h, mode, cost, target)
    try:
        return run_model(model, Dk, ann, **dict(hp, _fast=True, _inner_jobs=1))["pred_te"] + ym
    except Exception:
        return np.full(len(yte), ym)


def walk_forward(D: Design, model: str, ann: int, step: int, start_frac: float = 0.5, **hp):
    """확장창(expanding window) 재추정 평가.
    D.raw(표준화 전 설계행렬)를 사용해 시점 k마다 [0, k-h) 로 재학습(표준화·CV도 그 구간 내부) → [k, k+step) 예측.
    재학습 시점들은 서로 독립이므로 CPU 코어 수만큼 병렬 실행 (hp n_jobs=1이면 순차).
    h>1이면 학습 끝의 h-1개 관측치를 제거(purge)해 목표기간 겹침에 의한 누수를 막음."""
    raw = D.raw
    Xall = raw.drop(columns="__y").values; yall = raw["__y"].values; idx = raw.index
    h = D.horizon
    n = len(raw); k0 = int(n * start_frac)
    preds = np.full(n, np.nan)
    ks = [k for k in range(k0, n, step) if k - (h - 1) >= 60]
    jobs = [(k, model, Xall, yall, idx, k - (h - 1), step, D.names, h, D.mode, D.cost, D.target, ann, hp) for k in ks]
    n_jobs = hp.pop("n_jobs", -1)
    outs = []
    if jobs:
        # 첫 재학습 시간을 재서, 남은 작업이 몇 초 이상 걸릴 때만 병렬 (빠른 모델은 프로세스 기동 비용이 더 큼)
        t0 = time.perf_counter(); outs.append(_wf_one(*jobs[0])); t1 = time.perf_counter() - t0
        rest = jobs[1:]
        if n_jobs != 1 and len(rest) >= 3 and t1 * len(rest) > 3.0:
            from joblib import Parallel, delayed
            try:   # 재학습 시점들은 서로 독립 → 프로세스 병렬 (내부 모델은 단일 스레드로 과다구독 방지)
                outs += Parallel(n_jobs=n_jobs, backend="loky")(delayed(_wf_one)(*j) for j in rest)
            except Exception:
                outs += [_wf_one(*j) for j in rest]
        else:
            outs += [_wf_one(*j) for j in rest]
    for k, pk in zip(ks, outs):
        preds[k:k + step] = pk
    m = ~np.isnan(preds)
    y_oos, p_oos = yall[m], preds[m]
    # 기준예측: 각 시점의 과거 평균(확장창) — Campbell-Thompson OOS R²
    bench = pd.Series(yall).expanding().mean().shift(h).values[m]
    met = {"OOS R² (vs 학습평균)": 1 - ((y_oos - p_oos) ** 2).sum() / ((y_oos - bench) ** 2).sum()}
    met.update(strategy_stats(p_oos, y_oos, ann, h, getattr(D, "cost", 0.0)))
    met["재추정 횟수"] = int(np.ceil((n - k0) / step))
    return pd.Series(preds[m], index=idx[m]), pd.Series(y_oos, index=idx[m]), met


_CV_GAP = 0   # run_model이 설계의 예측기간 h에 맞춰 설정 (h-1)


def _tscv(n_splits=5):
    from sklearn.model_selection import TimeSeriesSplit
    return TimeSeriesSplit(n_splits=n_splits, gap=_CV_GAP)


# ---- Ridge 고속 CV: 폴드마다 SVD 한 번 → 40개 λ의 예측을 한꺼번에 -------------------------
RIDGE_ALPHAS = np.logspace(-3, 4, 40)


def ridge_cv(X, y, alphas=RIDGE_ALPHAS):
    """β(α) = V diag(s/(s²+α)) Uᵀy.  sklearn RidgeCV(cv=TimeSeriesSplit)과 같은 선택을 SVD 재사용으로 수십 배 빠르게."""
    err = np.zeros(len(alphas))
    for tr, va in _tscv().split(X):
        ym = y[tr].mean()
        U, sv, Vt = np.linalg.svd(X[tr], full_matrices=False)
        d = sv[None, :] / (sv[None, :] ** 2 + alphas[:, None])               # (α, r)
        P = (X[va] @ Vt.T) @ (d * (U.T @ (y[tr] - ym))[None, :]).T          # (n_va, α)
        err += ((y[va, None] - ym - P) ** 2).sum(0)
    a = float(alphas[int(np.argmin(err))])
    U, sv, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ (sv / (sv ** 2 + a) * (U.T @ y)), a


# ---- SCAD / MCP (좌표하강, 표준화 X 가정, numba 컴파일) --------------------------------
def _soft(z, l):
    return np.sign(z) * np.maximum(np.abs(z) - l, 0)


from numba import njit as _njit  # noqa: E402


@_njit(cache=True)
def _cd_core(X, y, lam, is_mcp, gamma, b, max_iter, tol):
    n, p = X.shape
    r = y - X @ b
    xx = (X ** 2).sum(0) / n
    for _ in range(max_iter):
        bmax = 0.0
        for j in range(p):
            z = X[:, j] @ r / n + xx[j] * b[j]
            v = xx[j]; az = abs(z); sg = 1.0 if z > 0 else (-1.0 if z < 0 else 0.0)
            if is_mcp:
                bj = sg * max(az - lam, 0.0) / (v - 1 / gamma) if az <= gamma * lam * v else z / v
            else:  # SCAD
                if az <= lam * (1 + v):
                    bj = sg * max(az - lam, 0.0) / v
                elif az <= gamma * lam * v:
                    bj = sg * max(az - gamma * lam / (gamma - 1), 0.0) / (v - 1 / (gamma - 1))
                else:
                    bj = z / v
            dlt = bj - b[j]
            if dlt != 0.0:
                r -= X[:, j] * dlt
                b[j] = bj
                if abs(dlt) > bmax:
                    bmax = abs(dlt)
        if bmax < tol:
            break
    return b


def _cd_nonconvex(X, y, lam, penalty, gamma, b0=None, max_iter=500, tol=1e-6):
    X = np.ascontiguousarray(X, dtype=np.float64); y = np.ascontiguousarray(y, dtype=np.float64)
    b = np.zeros(X.shape[1]) if b0 is None else np.array(b0, dtype=np.float64)
    return _cd_core(X, y, float(lam), penalty == "MCP", float(gamma), b, max_iter, tol)


def nonconvex_path(X, y, penalty="MCP", gamma=3.0, n_lam=40, eps=1e-3):
    n = len(y)
    lmax = np.max(np.abs(X.T @ y)) / n
    lams = np.logspace(np.log10(lmax), np.log10(lmax * eps), n_lam)
    B, b = [], None
    for l in lams:
        b = _cd_nonconvex(X, y, l, penalty, gamma, b0=b)
        B.append(b.copy())
    return lams, np.array(B)


def nonconvex_cv(X, y, penalty="MCP", gamma=3.0, n_lam=40):
    lams, _ = nonconvex_path(X, y, penalty, gamma, n_lam)
    err = np.zeros(len(lams))
    for tr, va in _tscv().split(X):
        b = None
        for i, l in enumerate(lams):
            b = _cd_nonconvex(X[tr], y[tr] - y[tr].mean(), l, penalty, gamma, b0=b)
            err[i] += np.mean((y[va] - y[tr].mean() - X[va] @ b) ** 2)
    best = lams[int(np.argmin(err))]
    lams_full, path = nonconvex_path(X, y, penalty, gamma, n_lam)
    b = _cd_nonconvex(X, y, best, penalty, gamma, b0=path[int(np.argmin(err))])
    return b, best, lams_full, path


# ---- Kalman TVP 회귀 ---------------------------------------------------------
def kalman_tvp(X, y, q: float = 1e-4, r: float | None = None):
    """y_t = x_t' β_t + ε_t,  β_t = β_{t-1} + η_t,  η ~ N(0, q·r·I).
    반환: (filtered β_{t|t-1} 기반 1-step 예측, 평활 β (RTS), 필터 β)."""
    n, p = X.shape
    if r is None:
        b_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        r = float(np.var(y - X @ b_ols))
    Q = q * r * np.eye(p)
    b = np.zeros(p); P = np.eye(p) * 1.0
    bp = np.zeros((n, p)); Pp = np.zeros((n, p, p)); bf = np.zeros((n, p)); Pf = np.zeros((n, p, p))
    pred = np.zeros(n)
    for t in range(n):
        P = P + Q                     # 예측
        bp[t], Pp[t] = b, P
        x = X[t]
        pred[t] = x @ b
        S = x @ P @ x + r
        K = P @ x / S
        b = b + K * (y[t] - pred[t])  # 갱신
        P = P - np.outer(K, x) @ P
        bf[t], Pf[t] = b, P
    bs = bf.copy(); Ps = Pf.copy()   # RTS 평활
    for t in range(n - 2, -1, -1):
        C = Pf[t] @ np.linalg.inv(Pp[t + 1])
        bs[t] = bf[t] + C @ (bs[t + 1] - bp[t + 1])
        Ps[t] = Pf[t] + C @ (Ps[t + 1] - Pp[t + 1]) @ C.T
    return pred, bs, bf


# ---- LSTM ------------------------------------------------------------------
def lstm_fit(D: Design, seq: int = 20, hidden: int = 32, epochs: int = 40, lr: float = 1e-3, seed: int = 0):
    import torch
    from torch import nn
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(4)
    Xall = np.vstack([D.X_tr, D.X_te]).astype(np.float32)
    yall = np.concatenate([D.y_tr, D.y_te]).astype(np.float32)
    ysd = D.y_tr.std() or 1.0
    n_tr = len(D.y_tr)
    idx = np.arange(seq - 1, len(yall))
    seqs = np.stack([Xall[i - seq + 1:i + 1] for i in idx])
    tr_mask = idx < n_tr
    Xs_tr, ys_tr = torch.tensor(seqs[tr_mask]), torch.tensor(yall[idx[tr_mask]] / ysd)
    Xs_te = torch.tensor(seqs[~tr_mask])
    nval = max(int(len(Xs_tr) * 0.15), 1)
    Xv, yv = Xs_tr[-nval:], ys_tr[-nval:]; Xt, yt = Xs_tr[:-nval], ys_tr[:-nval]

    class Net(nn.Module):
        def __init__(s, p):
            super().__init__(); s.l = nn.LSTM(p, hidden, batch_first=True); s.d = nn.Dropout(0.2); s.o = nn.Linear(hidden, 1)
        def forward(s, x):
            h, _ = s.l(x); return s.o(s.d(h[:, -1])).squeeze(-1)

    net = Net(Xall.shape[1]); opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.MSELoss(); hist = []; best = (np.inf, None); patience = 0
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 128):
            b = perm[i:i + 128]; opt.zero_grad(); l = lossf(net(Xt[b]), yt[b]); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            tl = lossf(net(Xt), yt).item(); vl = lossf(net(Xv), yv).item()
        hist.append((tl, vl))
        if vl < best[0] - 1e-5:
            best = (vl, {k: v.clone() for k, v in net.state_dict().items()}); patience = 0
        else:
            patience += 1
            if patience >= 8:
                break
    net.load_state_dict(best[1]); net.eval()
    with torch.no_grad():
        p_tr = net(Xs_tr).numpy() * ysd; p_te = net(Xs_te).numpy() * ysd
    # seq-1개 앞부분은 예측 불가 -> 0(학습평균)으로 채움
    p_tr_full = np.zeros(n_tr); p_tr_full[idx[tr_mask]] = p_tr
    return p_tr_full, p_te, pd.DataFrame(hist, columns=["train", "valid"])


# ---- Hidden Markov Model -----------------------------------------------------
def hmm_fit(D: Design, mode: str, n_states: int = 2, cov: str = "full", n_init: int = 5, seed: int = 0):
    """가우시안 HMM 레짐 모델.

    설명(explain): 관측 o_t = [y_t, X_t] 로 HMM 적합 → 레짐별 조건부 회귀
        E[y_t | x_t] = Σ_s w_s,t · (μ_y,s + Σ_yx,s Σ_xx,s⁻¹ (x_t − μ_x,s)),
        w_s,t ∝ P(s_t=s | 과거 관측) · N(x_t; μ_x,s, Σ_xx,s)   (레짐별 베타가 다른 회귀)
    예측(predict): 관측 o_t = 현재시점 변수들 → 필터 확률 α_t → ŷ_{t+h} = (α_t Pʰ) · m,  m_s = 레짐 s의 y 평균(학습구간)
    학습구간에서만 EM 추정, 테스트 구간은 전방 필터(forward filter)만 사용 → 미래정보 누수 없음.
    """
    from hmmlearn.hmm import GaussianHMM
    from scipy.stats import multivariate_normal
    ysd = D.y_tr.std() or 1.0
    names = D.names
    if mode == "explain":
        Otr = np.column_stack([D.y_tr / ysd, D.X_tr]); Ote = np.column_stack([D.y_te / ysd, D.X_te])
    else:
        # 현재 시점 관측치만 사용 (시차 열 "(t-k)" 제외, 특성 열은 포함)
        cur = [i for i, n in enumerate(names) if "(t-" not in n] or list(range(len(names)))
        Otr, Ote = D.X_tr[:, cur], D.X_te[:, cur]
    best, best_sc = None, -np.inf
    for k in range(n_init):
        m = GaussianHMM(n_components=n_states, covariance_type=cov, n_iter=300, tol=1e-4,
                        random_state=seed + k, min_covar=1e-3)
        try:
            m.fit(Otr)
            sc = m.score(Otr)
        except Exception:
            continue
        if sc > best_sc:
            best, best_sc = m, sc
    if best is None:
        raise RuntimeError("HMM 추정 실패")
    m = best
    P, mu, Sg = m.transmat_, m.means_, m.covars_          # covars_: (k, d, d)
    O = np.vstack([Otr, Ote]); T = len(O); K = n_states
    # 학습구간 평활확률로 레짐별 y 평균·변동성 (학습구간 내부 정보만 사용)
    gam = m.predict_proba(Otr)
    m_y = (gam * D.y_tr[:, None]).sum(0) / gam.sum(0)
    s_y = np.sqrt((gam * (D.y_tr[:, None] - m_y) ** 2).sum(0) / gam.sum(0))
    # 전방 필터
    logB = np.column_stack([multivariate_normal(mu[s], Sg[s], allow_singular=True).logpdf(O) for s in range(K)])
    alpha = np.zeros((T, K)); pred = np.zeros(T)
    prior = m.startprob_.copy()
    if mode == "explain":
        mx = [multivariate_normal(mu[s][1:], Sg[s][1:, 1:], allow_singular=True) for s in range(K)]
        betas = np.array([np.linalg.solve(Sg[s][1:, 1:], Sg[s][1:, 0]) for s in range(K)])   # (K, p)
    for t in range(T):
        if mode == "explain":
            x = O[t, 1:]
            lw = np.log(prior + 1e-300) + np.array([mx[s].logpdf(x) for s in range(K)])
            w = np.exp(lw - lw.max()); w /= w.sum()
            pred[t] = sum(w[s] * (mu[s][0] + betas[s] @ (x - mu[s][1:])) for s in range(K)) * ysd
        la = np.log(prior + 1e-300) + logB[t]
        a = np.exp(la - la.max()); a /= a.sum()
        alpha[t] = a
        prior = a @ P
    if mode == "predict":
        h = 1  # 설계행렬의 y가 이미 h기 앞 수익률이므로 한 단계 전이로 근사 (h>1이면 Pʰ)
        Ph = np.linalg.matrix_power(P, max(1, h))
        pred = (alpha @ Ph) @ m_y
    order = np.argsort(s_y)        # 레짐을 y 변동성 오름차순으로 정렬
    idx = D.idx_tr.append(D.idx_te)
    labels = [f"레짐{i+1}" for i in range(K)]
    probs = pd.DataFrame(alpha[:, order], index=idx, columns=labels)
    stay = np.diag(P)[order]
    w_, v_ = np.linalg.eig(P.T); st = np.real(v_[:, np.argmin(np.abs(w_ - 1))]); st = st / st.sum()
    regime = pd.DataFrame({"y 평균(%)": (m_y + D.y_mu)[order], "y 변동성(%)": s_y[order], "유지확률": stay,
                           "기대지속기간": 1 / (1 - stay), "정상분포 비중": st[order]}, index=labels)
    trans = pd.DataFrame(P[np.ix_(order, order)], index=labels, columns=labels)
    extra = dict(probs=probs, regime=regime, transmat=trans, n_states=K, cov=cov, loglik=best_sc * len(Otr))
    if mode == "explain" and cov == "full":
        extra["state_betas"] = pd.DataFrame(betas[order] * ysd, index=labels, columns=names)
    n = len(D.y_tr)
    return pred[:n], pred[n:], extra


# ---- 모델 실행기 ---------------------------------------------------------------
from models_dense import DENSE_MODELS  # noqa: E402  (A. 변수를 버리지 않는 기법)
ML_MODELS = ["LightGBM", "Gaussian Process", "Kernel Ridge"]                          # D. 부스팅·커널 (models_ml)
REGIME_MODELS = ["Markov-switching 회귀", "Threshold 회귀 (TAR)", "STAR (평활 전이)"]   # C. 레짐·비선형
MODEL_LIST = ["OLS", "Ridge", "Lasso", "Elastic Net", "Adaptive Lasso", "SCAD", "MCP"] + DENSE_MODELS + [
              "분위수 회귀", "Kalman TVP", "Hidden Markov (HMM)"] + REGIME_MODELS + ["Random Forest", "Gradient Boosting"] + ML_MODELS + ["LSTM"]


def _run_regime(name: str, D: Design, ann: int, **hp):
    import regime as RG
    Xtr, ytr, Xte, yte = D.X_tr, D.y_tr, D.X_te, D.y_te
    names_c = ["상수"] + list(D.names)
    if name.startswith("Markov"):
        p_tr, p_te, extra = RG.ms_regression(D, hp.get("ms_k", 2), hp.get("ms_pc", 3))
        w = extra["probs"].iloc[-1].values
        coef = pd.Series(w @ extra["state_betas"].values, index=D.names)
        return dict(coef=coef, pred_tr=p_tr, pred_te=p_te, metrics=_metrics(ytr, p_tr, yte, p_te, ann, D.horizon, D), extra=extra)
    j = RG.threshold_column(D, hp.get("thr_var"))
    qtr, qte = Xtr[:, j], Xte[:, j]
    to_orig = lambda v: float(v * D.sd[j] + D.mu[j])                  # 표준화 단위 → 원 단위
    if name.startswith("Threshold"):
        f = RG.tar_fit(Xtr, ytr, qtr)
        Xc_tr, Xc_te = np.column_stack([np.ones(len(ytr)), Xtr]), np.column_stack([np.ones(len(yte)), Xte])
        pred = lambda Xc, q: np.where(q <= f["gamma"], Xc @ f["b_low"], Xc @ f["b_high"])
        p_tr, p_te = pred(Xc_tr, qtr), pred(Xc_te, qte)
        pb = RG.tar_bootstrap_p(Xtr, ytr, qtr, f["supF"], B=hp.get("tar_boot", 199)) if not hp.get("_fast") else np.nan
        extra = dict(thr_name=D.names[j], gamma=to_orig(f["gamma"]), share_low=f["share_low"], supF=f["supF"], p_boot=pb,
                     regime_coef=pd.DataFrame({"q ≤ γ (하위 레짐)": f["b_low"], "q > γ (상위 레짐)": f["b_high"]}, index=names_c),
                     q_series=pd.Series(np.r_[qtr, qte] * D.sd[j] + D.mu[j], index=D.idx_tr.append(D.idx_te)))
        b_now = f["b_high"] if qte[-1] > f["gamma"] else f["b_low"]
    else:
        f = RG.star_fit(Xtr, ytr, qtr)
        G = lambda q: RG._G(q, f["gamma"], f["c"], f["sq"])[:, None]
        Xc_tr, Xc_te = np.column_stack([np.ones(len(ytr)), Xtr]), np.column_stack([np.ones(len(yte)), Xte])
        p_tr = (Xc_tr @ f["b1"]) + (Xc_tr * G(qtr)) @ f["b2"]
        p_te = (Xc_te @ f["b1"]) + (Xc_te * G(qte)) @ f["b2"]
        lm = RG.star_lm_test(Xtr, ytr, qtr) if not hp.get("_fast") else dict(LM=np.nan, df=0, p=np.nan)
        qgrid = np.linspace(np.quantile(qtr, 0.01), np.quantile(qtr, 0.99), 100)
        extra = dict(thr_name=D.names[j], gamma=f["gamma"], c=to_orig(f["c"]), lm=lm,
                     regime_coef=pd.DataFrame({"G=0 (하위 레짐)": f["b1"], "G=1 (상위 레짐)": f["b1"] + f["b2"]}, index=names_c),
                     transition=pd.Series(RG._G(qgrid, f["gamma"], f["c"], f["sq"]), index=qgrid * D.sd[j] + D.mu[j]),
                     G_series=pd.Series(RG._G(np.r_[qtr, qte], f["gamma"], f["c"], f["sq"]), index=D.idx_tr.append(D.idx_te)))
        gl = float(G(qte[-1:])[0, 0]); b_now = f["b1"] + gl * f["b2"]
    coef = pd.Series(b_now[1:], index=D.names)
    return dict(coef=coef, pred_tr=p_tr, pred_te=p_te, metrics=_metrics(ytr, p_tr, yte, p_te, ann, D.horizon, D), extra=extra)


def run_model(name: str, D: Design, ann: int = 252, **hp):
    """반환 dict: coef(Series|None, 표준화 계수), pred_tr, pred_te, metrics, extra(dict)."""
    from sklearn import linear_model as lm
    global _CV_GAP
    _CV_GAP = max(0, D.horizon - 1)   # h기 누적 목표는 인접 관측치가 겹침 → CV 검증구간 앞에 h-1개 간격
    Xtr, ytr, Xte, yte = D.X_tr, D.y_tr, D.X_te, D.y_te
    coef, extra = None, {}
    if name in DENSE_MODELS:
        from models_dense import run_dense
        return run_dense(name, D, ann, **hp)
    if name in REGIME_MODELS:
        return _run_regime(name, D, ann, **hp)
    if name in ML_MODELS:
        from models_ml import run_ml
        return run_ml(name, D, ann, **hp)
    if name == "OLS":
        import statsmodels.api as sm
        res = sm.OLS(ytr, Xtr).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        b = res.params
        extra["table"] = pd.DataFrame({"계수": b, "HAC t값": res.tvalues, "p값": res.pvalues}, index=D.names)
    elif name == "Ridge":
        b, extra["alpha"] = ridge_cv(Xtr, ytr)
    elif name == "Lasso":
        m = lm.LassoCV(cv=_tscv(), fit_intercept=False, alphas=50, max_iter=5000).fit(Xtr, ytr)
        b = m.coef_; extra["alpha"] = m.alpha_
        al, path, _ = lm.lasso_path(Xtr, ytr, alphas=50)
        extra["path"] = (al, path.T)
    elif name == "Elastic Net":
        l1 = hp.get("l1_ratio", [0.1, 0.3, 0.5, 0.7, 0.9])
        m = lm.ElasticNetCV(l1_ratio=l1, cv=_tscv(), fit_intercept=False, alphas=50, max_iter=5000).fit(Xtr, ytr)
        b = m.coef_; extra["alpha"] = m.alpha_; extra["l1_ratio"] = m.l1_ratio_
        al, path, _ = lm.enet_path(Xtr, ytr, l1_ratio=m.l1_ratio_, alphas=50)
        extra["path"] = (al, path.T)
    elif name == "Adaptive Lasso":
        g = hp.get("ada_gamma", 1.0)
        b0, _ = ridge_cv(Xtr, ytr)
        w = 1 / (np.abs(b0) ** g + 1e-8)
        m = lm.LassoCV(cv=_tscv(), fit_intercept=False, alphas=50, max_iter=5000).fit(Xtr / w, ytr)
        b = m.coef_ / w; extra["alpha"] = m.alpha_; extra["weights"] = pd.Series(w, index=D.names)
        al, path, _ = lm.lasso_path(Xtr / w, ytr, alphas=50)
        extra["path"] = (al, (path / w[:, None]).T)
    elif name in ("SCAD", "MCP"):
        g = hp.get("scad_gamma", 3.7) if name == "SCAD" else hp.get("mcp_gamma", 3.0)
        b, lam, lams, path = nonconvex_cv(Xtr, ytr, name, g)
        extra["alpha"] = lam; extra["gamma"] = g; extra["path"] = (lams, path)
    elif name == "분위수 회귀":
        import statsmodels.api as sm
        tau = hp.get("tau", 0.05)
        res = sm.QuantReg(ytr + D.y_mu, sm.add_constant(Xtr)).fit(q=tau, max_iter=5000)
        b = res.params[1:]
        extra["tau"] = tau
        extra["table"] = pd.DataFrame({"계수": b, "t값": res.tvalues[1:], "p값": res.pvalues[1:]}, index=D.names)
        p_tr = res.predict(sm.add_constant(Xtr)) - D.y_mu
        p_te = res.predict(sm.add_constant(Xte, has_constant="add")) - D.y_mu
        coef = pd.Series(b, index=D.names)
        # 분위수 회귀는 평균 예측이 아니므로 R² 대신 pinball loss를 함께 제시
        def pinball(y, p): u = y - p; return float(np.mean(np.maximum(tau * u, (tau - 1) * u)))
        met = _metrics(ytr, p_tr, yte, p_te, ann, D.horizon, D)
        for k_ in ["R² (학습)", "R² (테스트)", "OOS R² (vs 학습평균)", "RMSE (테스트)", "방향 적중률", "부호전략 Sharpe",
                   "부호전략 Sharpe (비용 후)", "롱 비중", "연간 회전율"]:
            met[k_] = np.nan      # 조건부 분위수 예측이므로 평균 기준·방향 전략 지표는 의미 없음
        met["Pinball loss (테스트)"] = pinball(yte + D.y_mu, p_te + D.y_mu)
        met["커버리지 (테스트, y<예측)"] = float(np.mean(yte < p_te))
        return dict(coef=coef, pred_tr=p_tr, pred_te=p_te, metrics=met, extra=extra)
    elif name == "Kalman TVP":
        q = hp.get("kalman_q", 1e-4)
        Xa = np.vstack([Xtr, Xte]); ya = np.concatenate([ytr, yte])
        b_tr = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
        r_tr = float(np.var(ytr - Xtr @ b_tr))            # 관측잡음 분산은 학습구간만으로 추정 (테스트 정보 차단)
        pred, bs, bf = kalman_tvp(Xa, ya, q=q, r=r_tr)
        n = len(ytr)
        extra["beta_smoothed"] = pd.DataFrame(bs, index=D.idx_tr.append(D.idx_te), columns=D.names)
        extra["beta_filtered"] = pd.DataFrame(bf, index=D.idx_tr.append(D.idx_te), columns=D.names)
        extra["q"] = q
        coef = pd.Series(bf[-1], index=D.names)   # 최신 필터 베타
        met = _metrics(ytr, pred[:n], yte, pred[n:], ann, D.horizon, D)
        return dict(coef=coef, pred_tr=pred[:n], pred_te=pred[n:], metrics=met, extra=extra)
    elif name == "Hidden Markov (HMM)":
        mode = hp.get("mode", "explain")
        p_tr, p_te, extra = hmm_fit(D, mode, hp.get("hmm_states", 2), hp.get("hmm_cov", "full"),
                                    n_init=2 if hp.get("_fast") else 5)   # walk-forward: EM 초기값 5→2회
        coef = None
        if "state_betas" in extra:   # 최근 필터확률로 가중한 '현재' 베타
            w = extra["probs"].iloc[-1].values
            coef = pd.Series(w @ extra["state_betas"].values, index=D.names)
        return dict(coef=coef, pred_tr=p_tr, pred_te=p_te, metrics=_metrics(ytr, p_tr, yte, p_te, ann, D.horizon, D), extra=extra)
    elif name in ("Random Forest", "Gradient Boosting"):
        from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
        from sklearn.inspection import permutation_importance
        if name == "Random Forest":
            m = RandomForestRegressor(n_estimators=hp.get("n_trees", 300), max_depth=hp.get("max_depth", 5),
                                      min_samples_leaf=20, n_jobs=hp.get("_inner_jobs", -1), random_state=0).fit(Xtr, ytr)
        else:
            m = HistGradientBoostingRegressor(max_depth=hp.get("max_depth", 3), learning_rate=0.05,
                                              max_iter=hp.get("n_trees", 300), l2_regularization=1.0,
                                              early_stopping=True, random_state=0).fit(Xtr, ytr)
        p_tr, p_te = m.predict(Xtr), m.predict(Xte)
        if not hp.get("_fast"):   # 순열 중요도는 화면 표시용 — walk-forward·stacking 내부에서는 생략 (계산의 대부분)
            pi = permutation_importance(m, Xte, yte, n_repeats=5, random_state=0, n_jobs=hp.get("_inner_jobs", -1))
            extra["importance"] = pd.Series(pi.importances_mean, index=D.names).sort_values()
        return dict(coef=None, pred_tr=p_tr, pred_te=p_te, metrics=_metrics(ytr, p_tr, yte, p_te, ann, D.horizon, D), extra=extra)
    elif name == "LSTM":
        p_tr, p_te_full, hist = lstm_fit(D, seq=hp.get("seq", 20), hidden=hp.get("hidden", 32),
                                         epochs=hp.get("epochs", 40))
        extra["loss"] = hist
        return dict(coef=None, pred_tr=p_tr, pred_te=p_te_full, metrics=_metrics(ytr, p_tr, yte, p_te_full, ann, D.horizon, D), extra=extra)
    else:
        raise ValueError(name)
    b = np.asarray(b)
    p_tr, p_te = Xtr @ b, Xte @ b
    coef = pd.Series(b, index=D.names)
    met = _metrics(ytr, p_tr, yte, p_te, ann, D.horizon, D)
    met["비영(非0) 계수 수"] = int((np.abs(b) > 1e-8).sum())
    return dict(coef=coef, pred_tr=p_tr, pred_te=p_te, metrics=met, extra=extra)


# ============================================================================
# FAVAR (Bernanke, Boivin & Eliasz 2005 — 2단계 PCA + VAR)
# ============================================================================
def favar(panel: pd.DataFrame, Y: pd.DataFrame, k: int = 3, p: int | None = None, H: int = 20, maxlags: int = 8):
    """panel: 정보집합 X (요인 추출용, 표준화), Y: 관측 변수(예: 특정 원자재 수익률).
    BBE 단순화: Y 정보를 제거한 '느린 변수' 회전은 생략하고, PCA 요인 F와 Y로 VAR([F, Y]) 추정.
    Cholesky 순서: 요인 → Y (요인 충격이 Y에 동시점 영향 가능)."""
    from statsmodels.tsa.api import VAR
    idx = panel.index.intersection(Y.index)
    Xp = panel.loc[idx]; Yp = Y.loc[idx]
    ev, load, F = pca(Xp, k)
    Z = pd.concat([F, (Yp - Yp.mean()) / Yp.std()], axis=1)
    model = VAR(Z.values)
    if p is None:
        p = max(1, int(model.select_order(maxlags).aic))
    res = model.fit(p)
    irf = res.irf(H)
    oirf = irf.orth_irfs                      # (H+1, n, n): [h, 반응변수, 충격변수]
    se = irf.stderr(orth=True)
    fe = res.fevd(H).decomp                   # (n, H, n)
    names = list(Z.columns)
    # 패널 변수 반응 = 적재값 × 요인 반응 (누적, 표준화 단위)
    Lam = load.values                         # (N, k)
    cum = oirf[:, :k, :].cumsum(axis=0)       # 요인 반응 누적
    panel_resp = {names[j]: pd.Series(Lam @ cum[-1, :, j], index=Xp.columns) for j in range(len(names))}
    return dict(ev=ev, load=load, F=F, res=res, p=p, oirf=oirf, se=se, fevd=fe, names=names,
                panel_resp=pd.DataFrame(panel_resp), nobs=res.nobs)
