"""B-1. 원자재 패널(횡단면) 예측 모델 — 모든 품목을 함께 학습 (Gu, Kelly & Xiu 2020 방식의 원자재판)

특성 (모두 결정일 t까지의 정보만, 품목별로 계산 후 시점별 순위로 [-0.5, 0.5] 변환 — 결측은 0 = 중립)
  MOM 1/3/6/12M, MOM 12-1M : 시계열 모멘텀 (Moskowitz-Ooi-Pedersen 2012)
  REV 1P                   : 직전 1기간 수익 (단기 반전)
  VOL 1M / 3M              : 일간 수익률 실현변동성 (저변동성 이상현상)
  SKEW 12M                 : 일간 수익률 왜도 (Fernandez-Perez et al. 2018 — 왜도 낮은 품목이 고수익)
  CARRY 12M (실현 롤수익)  : 최근 12개월 [투자가능 수익 − Yahoo 연속월물(≈현물 경로) 수익] 누적 = 실제로 실현된 롤수익
                             (Koijen et al. 2018 carry의 사후 실현치 대용 — 두 값 모두 t에 관측됨)
  COT 상업HP z / MM z      : CFTC 헤징압력·투기 포지션 (발표 다음 주 월요일부터 사용)
  SEASON                   : 같은 달의 과거 연도 평균 수익 (계절 모멘텀, Heston-Sadka 2008) — 올해 이전 연도만
목표   : 다음 기간 수익 (옵션: 변동성 표준화, 시점별 평균 제거)
학습   : walk-forward 확장창 — 결정일 d_k의 예측에는 목표가 d_k 이전에 실현된 행(날짜 ≤ d_{k-1})만 사용
평가   : 패널 OOS R² (기준 0, GKX), 시점별 순위 IC(Spearman), 상·하위 1/3 롱숏(다음 거래일 실행·비용 반영)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import analytics as A
import portfolio as P

CHARS = ["MOM 1M", "MOM 3M", "MOM 6M", "MOM 12M", "MOM 12-1M", "REV 1P", "VOL 1M", "VOL 3M", "SKEW 12M",
         "CARRY 12M (실현 롤수익)", "COT 상업HP z", "COT MM z", "SEASON (동월 과거평균)"]
PANEL_MODELS = ["Pooled OLS", "Ridge", "Lasso", "Elastic Net", "PCR", "Gradient Boosting", "Random Forest", "신경망 (MLP 앙상블)"]


# ============================================================================
# 패널 데이터 구축
# ============================================================================
def build_panel(levels_adj: pd.DataFrame, levels_raw: pd.DataFrame, ids: list[str], units: dict, freq: str = "W-FRI",
                basis: str = "etf", target: str = "vol_cs") -> dict:
    """return dict(X: (date,id)×특성 [순위변환], y: 목표, r_next: 다음기간 원수익, sigma, dates, ids, raw_chars)."""
    if basis == "etf":
        PX = P.etf_excess_levels(levels_adj, ids)
    else:
        PX = levels_adj[[i for i in ids if i in levels_adj]]
    ids = [i for i in ids if i in PX]
    daily = PX[ids].ffill(limit=5); daily = daily[daily.index.dayofweek < 5]
    rd = np.log(daily.where(daily > 0)).diff()
    raw = levels_raw[ids].reindex(daily.index).ffill(limit=5)
    rraw = np.log(raw.where(raw > 0)).diff()
    dec = P.decision_dates(daily.index, freq)
    ppy = A.PPY[freq]
    r = np.log(daily.reindex(dec)).diff()                                   # 기간 수익 r_t = (t-1, t]
    k = lambda m: max(1, round(ppy * m / 12))
    C = {}
    for m in (1, 3, 6, 12):
        C[f"MOM {m}M"] = r.rolling(k(m)).sum()
    C["MOM 12-1M"] = C["MOM 12M"] - C["MOM 1M"]
    C["REV 1P"] = r
    C["VOL 1M"] = (rd.rolling(21).std() * np.sqrt(252)).reindex(dec)
    C["VOL 3M"] = (rd.rolling(63).std() * np.sqrt(252)).reindex(dec)
    C["SKEW 12M"] = rd.rolling(252, min_periods=150).skew().reindex(dec)
    C["CARRY 12M (실현 롤수익)"] = (rd - rraw).rolling(252, min_periods=150).sum().reindex(dec)
    cot_hp, cot_mm = {}, {}
    for i in ids:
        if i in A.COT_MAP:
            F = A.make_features(levels_raw, i, units, freq, dec, ["COT 헤징압력"])
            cot_hp[i], cot_mm[i] = F.get("COT 상업HP z"), F.get("COT MM z")
    C["COT 상업HP z"] = pd.DataFrame(cot_hp, index=dec).reindex(columns=ids)
    C["COT MM z"] = pd.DataFrame(cot_mm, index=dec).reindex(columns=ids)
    # 계절 모멘텀: 다음 기간이 속한 달의, 올해 이전 연도 같은 달 평균 월수익
    mret = np.log(daily.resample("ME").last()).diff()
    seas = {}
    for i in ids:
        s = mret[i].dropna()
        df = pd.DataFrame({"y": s.index.year, "m": s.index.month, "r": s.values})
        tab = df.pivot_table(index="y", columns="m", values="r")
        prior = tab.expanding().mean().shift(1)                           # y년 값 = y년 이전 연도 평균
        nxt = dec + pd.Timedelta(days=3 if freq != "ME" else 20)
        seas[i] = [prior.at[d.year, d.month] if (d.year in prior.index and d.month in prior.columns) else np.nan for d in nxt]
    C["SEASON (동월 과거평균)"] = pd.DataFrame(seas, index=dec)
    # 목표
    r_next = r.shift(-1)
    sig = C["VOL 3M"] / np.sqrt(ppy)                                         # 기간 변동성 (사전)
    if target == "raw":
        y = r_next
    elif target == "vol":
        y = r_next / sig
    else:   # vol_cs: 변동성 표준화 + 시점별 평균 제거 (상대 순위에 집중)
        y = (r_next / sig).sub((r_next / sig).mean(axis=1), axis=0)
    # long 형태 + 시점별 순위 변환
    raw_long = pd.concat({c: v.reindex(index=dec, columns=ids).stack(future_stack=True) for c, v in C.items()}, axis=1)
    raw_long.index.names = ["date", "id"]

    Xr = raw_long.groupby(level="date", group_keys=False).apply(lambda g: g.rank(pct=True).sub(0.5).fillna(0.0))
    Y = y.reindex(index=dec, columns=ids).stack(future_stack=True).reindex(Xr.index)
    Rn = r_next.reindex(index=dec, columns=ids).stack(future_stack=True).reindex(Xr.index)
    Sg = sig.reindex(index=dec, columns=ids).stack(future_stack=True).reindex(Xr.index)
    # 품목이 실제로 존재(가격·변동성 있음)하는 행만
    alive = Sg.notna() & raw_long["MOM 12M"].notna()
    return dict(X=Xr[alive], y=Y[alive], r_next=Rn[alive], sigma=Sg[alive], raw=raw_long[alive], dates=dec, ids=ids,
                freq=freq, basis=basis, target=target, ppy=ppy)


# ============================================================================
# 모델
# ============================================================================
def _date_splits(dates_rows: np.ndarray, n_splits: int = 5):
    """행이 아닌 '날짜 블록' 단위 시계열 CV (같은 날짜의 품목들이 학습·검증에 나뉘지 않게)."""
    ud = np.unique(dates_rows)
    from sklearn.model_selection import TimeSeriesSplit
    for tr, va in TimeSeriesSplit(n_splits=n_splits).split(ud):
        yield np.isin(dates_rows, ud[tr]), np.isin(dates_rows, ud[va])


LINEAR = ("Pooled OLS", "Ridge", "Lasso", "Elastic Net", "PCR")


def _group_mean(v, g):
    """행 그룹(g)별 평균을 각 행에 펼침."""
    _, inv = np.unique(g, return_inverse=True)
    if v.ndim == 1:
        return (np.bincount(inv, v) / np.bincount(inv))[inv]
    cnt = np.bincount(inv)
    return np.column_stack([np.bincount(inv, v[:, j]) / cnt for j in range(v.shape[1])])[inv]


def _fe_apply(Xtr, ytr, dtr, etr, Xte, dte, ete, fe):
    """학습구간 FE 내부변환(within). return (X학습, y학습, X테스트, 테스트 절편).
    시간 FE : 날짜별 횡단면 평균 제거 — 테스트도 그 날짜의 횡단면 평균(t 시점 관측 가능)으로 제거
    품목 FE : 예측 ŷ = ȳ_i + (x − x̄_i)'β  (ȳ_i, x̄_i = 학습구간 품목 평균; 학습에 없던 품목은 전체 평균)
    양방향  : 시간 → 품목 순 내부변환 (불균형 패널은 교대 5회 반복)"""
    Xtr, ytr, Xte = np.array(Xtr, float), np.array(ytr, float), np.array(Xte, float)
    off = np.zeros(len(Xte))
    if fe in ("time", "twoway"):
        Xte = Xte - _group_mean(Xte, dte)
    if fe == "time":
        return Xtr - _group_mean(Xtr, dtr), ytr - _group_mean(ytr, dtr), Xte, off
    if fe in ("entity", "twoway"):
        if fe == "twoway":
            Xtr = Xtr - _group_mean(Xtr, dtr); ytr = ytr - _group_mean(ytr, dtr)
        ue = np.unique(etr)
        xb = {e: Xtr[etr == e].mean(0) for e in ue}; yb = {e: ytr[etr == e].mean() for e in ue}
        Xw, yw = Xtr.copy(), ytr.copy()
        for _ in range(5 if fe == "twoway" else 1):
            Xw = Xw - _group_mean(Xw, etr); yw = yw - _group_mean(yw, etr)
            if fe == "twoway":
                Xw = Xw - _group_mean(Xw, dtr); yw = yw - _group_mean(yw, dtr)
        gx, gy = Xtr.mean(0), ytr.mean()
        Xte = Xte - np.array([xb.get(e, gx) for e in ete])
        off = np.array([yb.get(e, gy) for e in ete])
        return Xw, yw, Xte, off
    return Xtr, ytr, Xte, off


def sector_expand(X, ent, sectors=None):
    """공통 기울기 + 섹터별 추가 기울기 (X ⊗ 섹터 더미). Ridge·Lasso에서는 섹터 차이가 공통으로 축소됨 (계층적).
    sectors: 사용할 섹터 목록 (학습 구간 기준으로 고정해 테스트와 열 수를 맞춤). 학습에 없던 섹터는 공통 기울기만 적용."""
    sec = np.array([SECTOR.get(e, "기타") for e in ent])
    if sectors is None:
        sectors = [s_ for s_ in ["곡물", "에너지", "금속", "소프트", "축산", "기타"] if (sec == s_).any()]
    blocks = [X] + [X * (sec == s_)[:, None] for s_ in sectors]
    return np.column_stack(blocks)


def fit_predict(model: str, Xtr, ytr, dtr, Xte, seed: int = 0, n_jobs: int = 1):
    """패널 모델 학습 → (예측, 계수 또는 중요도 Series 원본 배열)."""
    from sklearn import linear_model as lm
    if model == "Pooled OLS":
        b = np.linalg.lstsq(np.column_stack([np.ones(len(ytr)), Xtr]), ytr, rcond=None)[0]
        return Xte @ b[1:] + b[0], b[1:]
    if model in ("Ridge", "Lasso", "Elastic Net"):
        alphas = np.logspace(-4, 2, 25) if model != "Ridge" else np.logspace(-2, 5, 30)
        l1 = {"Lasso": 1.0, "Elastic Net": 0.5}.get(model)
        err = np.zeros(len(alphas))
        for tr, va in _date_splits(dtr):
            for j, a in enumerate(alphas):
                m = lm.Ridge(alpha=a) if model == "Ridge" else lm.ElasticNet(alpha=a, l1_ratio=l1, max_iter=5000)
                m.fit(Xtr[tr], ytr[tr]); err[j] += np.mean((ytr[va] - m.predict(Xtr[va])) ** 2)
        a = alphas[int(np.argmin(err))]
        m = lm.Ridge(alpha=a) if model == "Ridge" else lm.ElasticNet(alpha=a, l1_ratio=l1, max_iter=5000)
        m.fit(Xtr, ytr)
        return m.predict(Xte), m.coef_
    if model == "PCR":
        mu = Xtr.mean(0); U, s, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
        best, err = 1, []
        for kk in range(1, Xtr.shape[1] + 1):
            e = 0.0
            for tr, va in _date_splits(dtr):
                m0 = Xtr[tr].mean(0); _, _, V2 = np.linalg.svd(Xtr[tr] - m0, full_matrices=False)
                Z = (Xtr[tr] - m0) @ V2[:kk].T
                g = np.linalg.lstsq(np.column_stack([np.ones(len(Z)), Z]), ytr[tr], rcond=None)[0]
                e += np.mean((ytr[va] - (g[0] + (Xtr[va] - m0) @ V2[:kk].T @ g[1:])) ** 2)
            err.append(e)
        kk = int(np.argmin(err)) + 1
        Z = (Xtr - mu) @ Vt[:kk].T
        g = np.linalg.lstsq(np.column_stack([np.ones(len(Z)), Z]), ytr, rcond=None)[0]
        b = Vt[:kk].T @ g[1:]
        return g[0] + (Xte - mu) @ b, b
    if model == "Gradient Boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=200, min_samples_leaf=100,
                                          l2_regularization=1.0, early_stopping=False, random_state=seed).fit(Xtr, ytr)
        return m.predict(Xte), None
    if model == "Random Forest":
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=100, max_features=0.5,
                                  n_jobs=n_jobs, random_state=seed).fit(Xtr, ytr)
        return m.predict(Xte), m.feature_importances_
    if model.startswith("신경망"):
        from sklearn.neural_network import MLPRegressor
        preds = []
        for s_ in range(5):     # GKX처럼 시드 앙상블로 분산 축소
            m = MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-2, learning_rate_init=1e-3, max_iter=200,
                             early_stopping=False, random_state=seed + s_).fit(Xtr, ytr)
            preds.append(m.predict(Xte))
        return np.mean(preds, axis=0), None
    raise ValueError(model)


def _refit_one(model, X, y, d, k_dates, train_end, test_dates, chars_idx, seed, ent=None, fe="none", sector=False):
    tr = d <= train_end
    te = np.isin(d, test_dates)
    if tr.sum() < 200 or te.sum() == 0:
        return None
    Xtr, Xte = X[tr][:, chars_idx], X[te][:, chars_idx]
    ytr = y[tr]; off = np.zeros(te.sum())
    if sector and ent is not None:
        sec_tr = [s_ for s_ in ["곡물", "에너지", "금속", "소프트", "축산", "기타"] if any(SECTOR.get(e, "기타") == s_ for e in set(ent[tr]))]
        Xtr, Xte = sector_expand(Xtr, ent[tr], sec_tr), sector_expand(Xte, ent[te], sec_tr)
    if fe != "none" and model in LINEAR and ent is not None:
        Xtr, ytr, Xte, off = _fe_apply(Xtr, ytr, d[tr], ent[tr], Xte, d[te], ent[te], fe)
    p, imp = fit_predict(model, Xtr, ytr, d[tr], Xte, seed)
    if imp is not None and len(imp) != len(chars_idx):
        imp = np.asarray(imp)[:len(chars_idx)]                # 섹터 확장 시 공통 기울기 부분만 보고
    return te, p + off, imp


def walk_forward_panel(panel: dict, model: str, chars: list[str], refit: int = 26, min_train: int = 156,
                       n_jobs: int = -1, seed: int = 0, fe: str = "none", sector: bool = False) -> dict:
    """확장창 walk-forward. 결정일 dates[k]부터 refit개 결정일을 예측하는 모델은 날짜 ≤ dates[k-1]의 행
    (목표가 dates[k] 이전에 실현)만으로 학습."""
    Xdf, y = panel["X"], panel["y"]
    ok = y.notna()
    Xdf, y = Xdf[ok], y[ok]
    d = Xdf.index.get_level_values("date").values
    X = Xdf.values; yv = y.values
    ci = [list(Xdf.columns).index(c) for c in chars]
    ent = Xdf.index.get_level_values("id").values
    ud = np.unique(d)
    jobs = []
    for k in range(min_train, len(ud), refit):
        jobs.append((model, X, yv, d, k, ud[k - 1], ud[k:k + refit], ci, seed, ent, fe, sector))
    from joblib import Parallel, delayed
    heavy = model in ("Random Forest", "신경망 (MLP 앙상블)", "Gradient Boosting", "PCR", "Lasso", "Elastic Net")
    outs = (Parallel(n_jobs=n_jobs, backend="loky")(delayed(_refit_one)(*j) for j in jobs)
            if heavy and len(jobs) >= 3 and n_jobs != 1 else [_refit_one(*j) for j in jobs])
    pred = pd.Series(np.nan, index=Xdf.index)
    imps = []
    for o in outs:
        if o is None:
            continue
        te, p, imp = o
        pred.iloc[np.flatnonzero(te)] = p
        if imp is not None:
            imps.append(np.abs(imp) if model in ("Random Forest",) else imp)
    importance = pd.Series(np.mean(imps, axis=0), index=chars) if imps else None
    return dict(pred=pred.dropna(), importance=importance, n_refits=len(jobs))


# ============================================================================
# 평가
# ============================================================================
def evaluate(panel: dict, pred: pd.Series, levels_adj, levels_act, cost_liq=1.0, comm=P.COMMISSION, q: float = 1 / 3) -> dict:
    y = panel["y"].reindex(pred.index); rn = panel["r_next"].reindex(pred.index)
    m = y.notna() & pred.notna()
    y, p, rn = y[m], pred[m], rn[m]
    oos = 1 - ((y - p) ** 2).sum() / (y ** 2).sum()
    df = pd.DataFrame({"p": p, "r": rn})
    ic = df.groupby(level="date").apply(lambda g: g["p"].corr(g["r"], method="spearman") if len(g) >= 4 else np.nan).dropna()
    W = long_short_weights(p, q)
    bt = P.run_weights(W, levels_adj, levels_act, panel["basis"], cost_liq, comm)
    st = bt["stats"]
    return dict(oos_r2=float(oos), ic=ic, ic_mean=float(ic.mean()), ic_t=float(ic.mean() / ic.std() * np.sqrt(len(ic))) if ic.std() > 0 else np.nan,
                ic_pos=float((ic > 0).mean()), ls=bt, ls_stats=st, W=W)


def long_short_weights(score: pd.Series, q: float = 1 / 3, gross: float = 1.0) -> pd.DataFrame:
    """시점별 상위 q 매수, 하위 q 매도 (동일비중, 총노출 = gross)."""
    S = score.unstack("id")
    W = pd.DataFrame(0.0, index=S.index, columns=S.columns)
    for dt, row in S.iterrows():
        r = row.dropna()
        n = len(r)
        if n < 4:
            continue
        nq = max(1, int(np.floor(n * q)))
        o = r.sort_values()
        W.loc[dt, o.index[-nq:]] = gross / 2 / nq
        W.loc[dt, o.index[:nq]] = -gross / 2 / nq
    return W


def factor_table(panel: dict, levels_adj, levels_act, chars: list[str], start=None) -> pd.DataFrame:
    """특성 하나씩: 순위 상·하위 1/3 롱숏 성과와 IC (부호는 사전 가설과 무관하게 그대로 보고)."""
    rows = []
    for c in chars:
        sc = panel["X"][c]
        if start is not None:
            sc = sc[sc.index.get_level_values("date") >= pd.Timestamp(start)]
        sc = sc[sc != 0]
        if sc.empty:
            continue
        ev = evaluate(panel, sc, levels_adj, levels_act)
        s = ev["ls_stats"]
        rows.append({"특성": c, "평균 IC": ev["ic_mean"], "IC t값": ev["ic_t"], "IC>0 비율": ev["ic_pos"],
                     "롱숏 Sharpe(비용 후)": s.get("Sharpe (비용 후)", np.nan), "롱숏 연수익": s.get("연환산 수익률(CAGR)", np.nan),
                     "연간 회전율": s.get("연간 회전율(편도 합)", np.nan)})
    return pd.DataFrame(rows).set_index("특성")


def forecasts_for_portfolio(panel: dict, pred: pd.Series) -> pd.DataFrame:
    """패널 예측(표준화 단위) → 기간 기대수익(소수, 로그) 프레임 (결정일 × 종목) — portfolio.backtest(signal='panel')."""
    if panel["target"] in ("vol", "vol_cs"):
        mu = pred * panel["sigma"].reindex(pred.index)
    else:
        mu = pred
    return mu.unstack("id")


# ============================================================================
# 계량경제학 패널 추정 (전체 표본 — 추론·해석용, linearmodels)
# ============================================================================
SECTOR = {"yf:ZC=F": "곡물", "yf:ZS=F": "곡물", "yf:ZM=F": "곡물", "yf:ZL=F": "곡물", "yf:ZW=F": "곡물", "yf:KE=F": "곡물",
          "yf:ZO=F": "곡물", "yf:ZR=F": "곡물", "yf:CL=F": "에너지", "yf:BZ=F": "에너지", "yf:NG=F": "에너지", "yf:HO=F": "에너지",
          "yf:RB=F": "에너지", "yf:HG=F": "금속", "yf:GC=F": "금속", "yf:SI=F": "금속", "yf:PL=F": "금속", "yf:PA=F": "금속",
          "yf:ALI=F": "금속", "yf:TIO=F": "금속", "yf:KC=F": "소프트", "yf:SB=F": "소프트", "yf:CC=F": "소프트", "yf:CT=F": "소프트",
          "yf:OJ=F": "소프트", "yf:LE=F": "축산", "yf:GF=F": "축산", "yf:HE=F": "축산"}
FE_OPTS = {"없음 (Pooled)": "none", "시간 FE": "time", "품목 FE": "entity", "양방향 FE (시간+품목)": "twoway"}
CLUSTER_OPTS = {"시점 군집": "time", "품목 군집": "entity", "양방향 군집": "twoway", "Driscoll-Kraay (횡단면 상관+자기상관)": "dk", "이분산 강건 (White)": "robust"}


def _lm_frame(panel: dict, chars: list[str], target: str = "y"):
    y = panel["y"] if target == "y" else panel["r_next"] * 100
    df = panel["X"][chars].copy()
    df["__y"] = y
    df = df.dropna()
    df = df.swaplevel().sort_index()                 # linearmodels: (entity, time)
    return df


def fama_macbeth(panel: dict, chars: list[str], target: str = "y", nw_lags: int = 4):
    """매 시점 횡단면 OLS → 기울기 시계열 평균, Newey-West(Bartlett) t값.
    직접 구현: 품목 수가 적은 초기 구간이나 특정 시점에 상수인 특성(예: COT 미보유 품목만 남은 주)이 있어도
    해당 시점만 건너뜀 (linearmodels는 모든 시점이 계수 부족이면 실패). 대부분 시점에서 상수인 특성은 제외."""
    from scipy import stats
    df = _lm_frame(panel, chars, target)
    tl = df.index.get_level_values(1)
    var_ok = df[chars].groupby(tl).std().fillna(0).gt(1e-12).mean()
    use = [c for c in chars if var_ok[c] >= 0.5]
    dropped = [c for c in chars if c not in use]
    k = len(use) + 1
    rows, r2s, nobs = {}, [], 0
    for t, g in df.groupby(tl):
        if len(g) < k + 2:
            continue
        X = np.column_stack([np.ones(len(g)), g[use].values]); y = g["__y"].values
        if np.linalg.matrix_rank(X) < k:
            continue
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        e = y - X @ b; sst = ((y - y.mean()) ** 2).sum()
        rows[t] = b; r2s.append(1 - (e @ e) / sst if sst > 0 else np.nan); nobs += len(g)
    if len(rows) < 10:
        raise ValueError(f"Fama-MacBeth: 추정 가능한 횡단면이 {len(rows)}개뿐 (품목 수 < 특성 수 + 3). 특성을 줄이거나 품목을 늘리세요.")
    B = pd.DataFrame(rows, index=["const"] + use).T
    T_ = len(B); m = B.mean(); d = B - m
    lrv = (d ** 2).sum() / T_
    for L in range(1, nw_lags + 1):
        w = 1 - L / (nw_lags + 1)
        lrv += 2 * w * (d.iloc[L:].values * d.iloc[:-L].values).sum(0) / T_
    se = np.sqrt(lrv / T_)
    tv = m / se
    tbl = pd.DataFrame({"평균 기울기": m, "NW t값": tv, "p값": 2 * stats.t.sf(np.abs(tv), T_ - 1)})
    return dict(table=tbl, betas=B, nobs=nobs, periods=T_, r2=float(np.nanmean(r2s)), dropped=dropped)


def _cov_kwargs(cluster: str, nw_lags: int = 4):
    return {"time": dict(cov_type="clustered", cluster_time=True), "entity": dict(cov_type="clustered", cluster_entity=True),
            "twoway": dict(cov_type="clustered", cluster_time=True, cluster_entity=True),
            "dk": dict(cov_type="kernel", kernel="bartlett", bandwidth=nw_lags), "robust": dict(cov_type="robust")}[cluster]


def panel_ols(panel: dict, chars: list[str], fe: str = "time", cluster: str = "time", target: str = "y"):
    """고정효과 패널 OLS + 군집/Driscoll-Kraay 표준오차. return dict(table, r2_within, f_pooled(효과 유의성), nobs)."""
    from linearmodels import PanelOLS
    import statsmodels.api as sm
    df = _lm_frame(panel, chars, target)
    X = sm.add_constant(df[chars])
    mod = PanelOLS(df["__y"], X, entity_effects=fe in ("entity", "twoway"), time_effects=fe in ("time", "twoway"),
                   drop_absorbed=True, check_rank=False)
    res = mod.fit(**_cov_kwargs(cluster))
    tbl = pd.DataFrame({"계수": res.params, "표준오차": res.std_errors, "t값": res.tstats, "p값": res.pvalues})
    fp = getattr(res, "f_pooled", None)
    return dict(table=tbl, r2_within=float(res.rsquared_within), r2=float(res.rsquared), nobs=int(res.nobs),
                f_pooled=(float(fp.stat), float(fp.pval)) if fp is not None and fe != "none" else None)


def sector_slopes(panel: dict, chars: list[str], fe: str = "time", cluster: str = "time", target: str = "y"):
    """섹터별 기울기 (특성 × 섹터 더미) + 특성별 '섹터 간 기울기 동일' Wald 검정."""
    from linearmodels import PanelOLS
    df = _lm_frame(panel, chars, target)
    ent = df.index.get_level_values(0)
    sec = pd.Series([SECTOR.get(e, "기타") for e in ent], index=df.index)
    sectors = [s for s in ["곡물", "에너지", "금속", "소프트", "축산", "기타"] if (sec == s).any()]
    cols = {}
    for c in chars:
        for s in sectors:
            cols[f"{c} × {s}"] = df[c] * (sec == s)
    Xs = pd.DataFrame(cols, index=df.index)
    if fe == "none":
        for s in sectors:
            Xs[f"상수 × {s}"] = (sec == s).astype(float)
    res = PanelOLS(df["__y"], Xs, entity_effects=fe in ("entity", "twoway"), time_effects=fe in ("time", "twoway"),
                   drop_absorbed=True, check_rank=False).fit(**_cov_kwargs(cluster))
    coef = pd.DataFrame({s: [res.params.get(f"{c} × {s}", np.nan) for c in chars] for s in sectors}, index=chars)
    tval = pd.DataFrame({s: [res.tstats.get(f"{c} × {s}", np.nan) for c in chars] for s in sectors}, index=chars)
    names = list(res.params.index)
    wald = {}
    for c in chars:
        terms = [f"{c} × {s}" for s in sectors if f"{c} × {s}" in names]
        if len(terms) < 2:
            continue
        R = np.zeros((len(terms) - 1, len(names)))
        for r_, t_ in enumerate(terms[1:]):
            R[r_, names.index(terms[0])] = 1; R[r_, names.index(t_)] = -1
        w = res.wald_test(R, np.zeros(len(terms) - 1))
        wald[c] = {"Wald 통계량": float(w.stat), "자유도": len(terms) - 1, "p값": float(w.pval)}
    n_by = sec.groupby(sec).size().reindex(sectors)
    return dict(coef=coef, tstat=tval, wald=pd.DataFrame(wald).T, sectors=sectors, n_by_sector=n_by)
