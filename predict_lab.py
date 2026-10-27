"""④ 예측 확장 — 멀티모달 설명변수 · 대안 시계열 모델(단일 품목) · 모델 × 설명변수 백테스트 그리드

  1) 멀티모달 설명변수: multimodal.features()의 품목별 주간 특성(텍스트 FinBERT·리포트 수치·COT·기상·매크로·금융·월간 거시)을
     2026-10-26 기준으로 한 번 계산해 저장(data/mm/features_*.pkl) → ④ 단일·여러 품목·MIDAS 탭에서 설명변수로 선택.
     값은 발표 이후에만 쓰이도록 이미 맞춰져 있음(multimodal.py). 단일 품목 탭에서는 품목별 과거 자료로만 표준화하고 자료가 없는 구간은 0(평균).
  2) 대안 시계열 모델(단일 품목 설계행렬 D 위에서): Prophet(추세·계절 + 설명변수) · neuralforecast PatchTST·iTransformer·TFT ·
     Chronos-Bolt · TimesFM-2.5 (zero-shot) · Chronos-2 (설명변수 공변량). 학습 = 학습 구간, 예측 = 테스트 구간의 각 시점까지 자료.
  3) 백테스트 그리드: 19개 품목 · 매주 결정 · 4주 뒤 수익률 예측 · 매년 1월 재학습(그해 이전 자료, 4주 purge) · 2007-01 ~ 2026-10-26.
     설명변수 묶음 14개 × 표준 모델 6개 + 대안 시계열 모델(설명변수 없음 / 매크로 / 멀티모달) → 방향 매매·순위 롱숏 (편도 5bp, 목표 변동성 10%).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import alt_ts as AT
import ls_strategies as LS

ROOT = Path(__file__).parent
FROZEN_END = "2026-10-26"
FEAT_PKL = ROOT / "data" / "mm" / f"features_{FROZEN_END}.pkl"
GRID_START = "2007-01-01"
TRAIN_START = "2001-01-01"
RECENT = "2021-11-01"
H = 4


def frozen_levels(roll: bool = True) -> pd.DataFrame:
    import analytics as A
    return A.load_levels(None, roll_adjusted=roll).loc[:FROZEN_END]


# ============================================================================
# 1) 멀티모달 설명변수
# ============================================================================
def mm_features(L: pd.DataFrame | None = None) -> dict:
    """{갈래: {특성: 주간 × 19품목}} — 처음 한 번 계산해 저장 (FinBERT 채점 등 원자료는 multimodal.py 캐시)."""
    if FEAT_PKL.exists():
        return pd.read_pickle(FEAT_PKL)
    import multimodal as MMd
    L = frozen_levels() if L is None else L
    F = MMd.features(L, AT.weekly_px(L), use_news=False)                 # GDELT 뉴스 제외 (수집 속도 문제)
    FEAT_PKL.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(F, FEAT_PKL)
    return F


def mm_options(F: dict, item: str | None = None) -> list[str]:
    """선택지 '갈래 · 특성' (item을 주면 그 품목에 값이 있는 특성만)."""
    return [f"{fam} · {nm}" for fam, d in F.items() for nm, S in d.items() if item is None or (item in S and S[item].notna().any())]


def _tz(S, min_obs: int = 26):
    """과거 자료로만 표준화 (확장 창): t 시점 값은 t까지의 평균·표준편차로."""
    return (S - S.expanding(min_obs).mean()) / S.expanding(min_obs).std()


def mm_item_frame(F: dict, item: str, picks: list[str], index: pd.DatetimeIndex) -> pd.DataFrame:
    """품목 하나의 멀티모달 특성 → 분석 주기 index로 (그 시점까지 나온 마지막 주간 값), 과거 표준화 후 결측 = 0 (평균)."""
    cols = {}
    for p in picks:
        fam, nm = p.split(" · ", 1)
        S = F.get(fam, {}).get(nm)
        if S is not None and item in S and S[item].notna().any():
            cols["MM · " + nm] = _tz(S[item].dropna())
    if not cols:
        return pd.DataFrame(index=index)
    W = pd.DataFrame(cols).sort_index()
    return W.reindex(W.index.union(index)).ffill().reindex(index).fillna(0.0)


def mm_panel(F: dict, picks: list[str]) -> dict:
    """여러 품목 탭용 {특성 이름: 주간 × 품목} (원 값 — 패널 탭이 시점별 순위로 바꿈)."""
    out = {}
    for p in picks:
        fam, nm = p.split(" · ", 1)
        if nm in F.get(fam, {}):
            out["MM · " + nm] = F[fam][nm]
    return out


def mm_composite(F: dict) -> dict:
    """갈래별 요약 = 그 갈래 특성들의 품목별 과거 표준화 값 평균 (대안 시계열 모델의 품목별 입력 7개)."""
    out = {}
    for fam, d in F.items():
        Z = [_tz(S, 52) for S in d.values()]
        out[fam] = pd.concat(Z, keys=range(len(Z))).groupby(level=1).mean().clip(-4, 4)
    return out


# ============================================================================
# 2) 대안 시계열 모델 — 단일 품목 설계행렬 D 위에서
# ============================================================================
ALT_SINGLE = {
    "Prophet (추세·계절 + 설명변수)": "prophet",
    "PatchTST (neuralforecast · 자기 시계열만)": "PatchTST",
    "iTransformer (neuralforecast · 설명변수 = 토큰)": "iTransformer",
    "TFT (neuralforecast · 설명변수)": "TFT",
    "Chronos-Bolt-small (zero-shot · 자기 시계열만)": "Chronos-Bolt-small",
    "TimesFM-2.5 (zero-shot · 자기 시계열만)": "TimesFM-2.5",
    "Chronos-2 (zero-shot · 설명변수 공변량)": "Chronos-2",
}
_PIPE = {}


def _input_size(ann: int, n_tr: int) -> int:
    return int(min({252: 120, 52: 52, 12: 24}.get(ann, 52), max(8, n_tr // 4)))


def _top_cols(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """학습 구간 |상관| 상위 k개 설명변수 (Chronos-2·iTransformer 입력 수 제한용, 학습 자료만 사용)."""
    if X.shape[1] <= k:
        return np.arange(X.shape[1])
    c = np.nan_to_num(np.abs([np.corrcoef(X[:, j], y)[0, 1] if X[:, j].std() > 0 else 0 for j in range(X.shape[1])]))
    return np.sort(np.argsort(c)[-k:])


def run_alt_single(label: str, D, ann: int, r: pd.Series, scale: pd.Series | None = None, max_points: int = 400) -> dict:
    """반환 = analytics.run_model과 같은 dict (coef=None). r = 목표 품목의 기간 수익률(%) 전체 시계열.
    D.y는 h기 누적 수익률(학습 평균 제거). 시계열 모델은 '원 수익률'을 예측하므로 변동성 표준화 목표면 scale(σₜ)로 나눔.
    테스트 시점이 max_points보다 많으면 일정 간격으로만 예측하고 그 사이는 직전 예측 유지 (CPU 시간 제한)."""
    import analytics as A
    kind = ALT_SINGLE[label]
    h, n_tr = D.horizon, len(D.idx_tr)
    idx = D.idx_tr.append(D.idx_te)
    X = np.vstack([D.X_tr, D.X_te])
    yraw_tr = D.y_tr + D.y_mu
    te_pos = np.arange(n_tr, len(idx))
    step = max(1, int(np.ceil(len(te_pos) / max_points)))
    calc = te_pos[::step]
    p_tr = np.full(n_tr, np.nan)
    p_at = {}
    sc = scale.reindex(idx).values if scale is not None else np.ones(len(idx))
    extra = {"설명": "", "예측 간격": step}
    if kind == "prophet":
        import logging
        from prophet import Prophet
        for lg in ("cmdstanpy", "prophet"):
            logging.getLogger(lg).setLevel(logging.ERROR)
        cols = [f"x{j}" for j in range(X.shape[1])]
        df = pd.DataFrame(X, columns=cols); df["ds"] = idx
        m = Prophet(yearly_seasonality=3 if ann >= 12 else False, weekly_seasonality=False, daily_seasonality=False,
                    uncertainty_samples=0, changepoint_prior_scale=0.05)
        for c in cols:
            m.add_regressor(c, prior_scale=0.1, standardize=False)
        m.fit(df.iloc[:n_tr].assign(y=yraw_tr))
        yh = m.predict(df)["yhat"].values
        p_tr = yh[:n_tr] - D.y_mu
        p_at = {k: yh[k] for k in te_pos}
        extra["설명"] = f"목표(h기 누적 수익률) = 추세(변곡점) + 연간 계절 + 설명변수 {len(cols)}개 선형 (사전분포 척도 0.1로 축소)"
    elif kind in ("PatchTST", "iTransformer", "TFT"):
        import logging
        from neuralforecast import NeuralForecast
        from neuralforecast.models import PatchTST, iTransformer, TFT
        from neuralforecast.losses.pytorch import MSE
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR); logging.getLogger("lightning").setLevel(logging.ERROR)
        s = r.reindex(idx).fillna(0.0).values
        ins = _input_size(ann, n_tr)
        kw = dict(h=h, input_size=ins, max_steps=300, loss=MSE(), random_seed=1, enable_progress_bar=False,
                  enable_model_summary=False, logger=False, accelerator="cpu")
        if kind == "iTransformer":
            sel = _top_cols(D.X_tr, D.y_tr, 15)
            ser = np.column_stack([s, X[:, sel]])
            long = lambda a, b: pd.concat([pd.DataFrame({"unique_id": f"s{j}", "ds": np.arange(a, b), "y": ser[a:b, j]})   # noqa: E731
                                           for j in range(ser.shape[1])], ignore_index=True)
            nf = NeuralForecast(models=[iTransformer(n_series=ser.shape[1], hidden_size=64, n_heads=4, e_layers=2, d_ff=128, **kw)], freq=1)
            nf.fit(long(0, n_tr))
            for k in calc:
                fc = nf.predict(df=long(max(0, k - ins + 1), k + 1))
                p_at[k] = float(fc[fc.unique_id == "s0"].iloc[:, -1].sum())
            extra["설명"] = f"목표 품목 수익률 + 설명변수 {len(sel)}개(학습 구간 상관 상위)를 각각 토큰으로 → 변수 간 어텐션"
        else:
            ex = [f"x{j}" for j in range(X.shape[1])] if kind == "TFT" else []
            base = pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])]); base["y"] = s; base["ds"] = np.arange(len(idx))
            if kind == "TFT":
                mdl = TFT(hidden_size=32, n_head=4, hist_exog_list=ex or None, scaler_type="robust", **kw)
            else:
                mdl = PatchTST(patch_len=4, stride=4, hidden_size=64, n_heads=4, encoder_layers=2, scaler_type="robust", **kw)
            nf = NeuralForecast(models=[mdl], freq=1)
            nf.fit(base.iloc[:n_tr].assign(unique_id="y")[["unique_id", "ds", "y"] + ex])
            hist = pd.concat([base.iloc[max(0, k - ins + 1):k + 1].assign(unique_id=f"c{k}") for k in calc], ignore_index=True)
            fc = nf.predict(df=hist[["unique_id", "ds", "y"] + ex])
            agg = fc.groupby("unique_id")[fc.columns[-1]].sum()
            p_at = {k: float(agg.get(f"c{k}", np.nan)) for k in calc}
            extra["설명"] = (f"직전 {ins}기 목표 수익률" + (f" + 설명변수 {len(ex)}개(과거 변수)" if ex else " (외부 변수 없음)") +
                            f"로 다음 {h}기 수익률 예측 → 합계. 학습 구간으로 한 번 학습 (최대 300단계)")
    else:
        import torch
        import forecast_lab as FL
        lp = (r.fillna(0.0).cumsum() / 100.0)
        pos = lp.index.get_indexer(idx[calc])
        if kind == "Chronos-2":
            from chronos import BaseChronosPipeline
            if "c2" not in _PIPE:
                _PIPE["c2"] = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu", torch_dtype=torch.float32)
            sel = _top_cols(D.X_tr, D.y_tr, 8)
            Xi = pd.DataFrame(X[:, sel], index=idx).reindex(lp.index).ffill().fillna(0.0).values
            items = [{"target": lp.values[max(0, p - 511):p + 1].astype(np.float32),
                      "past_covariates": {f"x{j}": Xi[max(0, p - 511):p + 1, j].astype(np.float32) for j in range(len(sel))}} for p in pos]
            res = []
            for k in range(0, len(items), 64):
                q, _ = _PIPE["c2"].predict_quantiles(items[k:k + 64], prediction_length=h, quantile_levels=FL.QS)
                res.append(np.stack([np.asarray(x).reshape(-1, h, len(FL.QS))[0] for x in q]))
            Q = np.concatenate(res)
            extra["설명"] = f"사전학습 모델 zero-shot — 직전 512기 로그가격 + 설명변수 {len(sel)}개(학습 구간 상관 상위)를 과거 공변량으로"
        else:
            f = FL._load(kind)
            Q = f([lp.values[max(0, p - 511):p + 1] for p in pos], h)
            extra["설명"] = "사전학습 모델 zero-shot — 직전 512기 로그가격만 보고 예측 (설명변수 사용 안 함)"
        med = Q[:, h - 1, 4]
        p_at = {k: float((med[j] - lp.values[p]) * 100.0) for j, (k, p) in enumerate(zip(calc, pos))}
    pt = pd.Series({k: v for k, v in p_at.items()}).reindex(te_pos).ffill().values      # 계산 안 한 시점은 직전 예측 유지
    p_te = pt / sc[te_pos] - D.y_mu
    p_te = np.where(np.isfinite(p_te), p_te, 0.0)
    if kind not in ("prophet",):
        p_tr = np.full(n_tr, np.nan)
    met = A._metrics(D.y_tr, np.nan_to_num(p_tr) if np.isfinite(p_tr).any() else np.full(n_tr, np.nan), D.y_te, p_te, ann, h, D)
    return dict(coef=None, pred_tr=p_tr, pred_te=p_te, metrics=met, extra=extra)


# ============================================================================
# 3) 백테스트 그리드 — 19개 품목 · 모델 × 설명변수 묶음
# ============================================================================
MACRO_IDS = {"fred:VIXCLS": ("VIX (로그)", "loglvl"), "fred:BAA10Y": ("신용 스프레드", "lvl"), "yf:DX-Y.NYB": ("달러 4주 변화", "logd4"),
             "fred:DGS10": ("10년 금리 4주 변화", "d4"), "fred:DTB3": ("3개월 금리 4주 변화", "d4"), "yf:^GSPC": ("S&P500 4주 수익률", "logd4"),
             "ak:macro_shipping_bdi": ("BDI 4주 변화", "logd4")}
OWN = "자기 정보 (모멘텀 1·4·13·26·52주 · 변동성)"
CROSS = "19개 원자재 (다른 품목 4·52주 수익률)"
MACRO = "거시 (VIX·신용·달러·금리·주가·BDI)"


def _sets(fams: list[str]) -> dict:
    """설명변수 묶음 (이름 → 블록들). 마지막은 위약 = '전부'의 외부 변수를 시간 순서대로 섞음."""
    S = {"① 자기 정보만": [OWN], "② 자기 + 19개 원자재": [OWN, CROSS], "③ 자기 + 거시": [OWN, MACRO],
         "④ 자기 + 19개 원자재 + 거시 (일반적 설정)": [OWN, CROSS, MACRO]}
    for j, f in enumerate(fams):
        S[f"⑤-{j + 1} 자기 + 멀티모달 · {f}"] = [OWN, f]
    S["⑥ 자기 + 멀티모달 전부"] = [OWN] + fams
    S["⑦ 전부 (자기 + 19개 원자재 + 거시 + 멀티모달)"] = [OWN, CROSS, MACRO] + fams
    S["⑧ 위약 · ⑦의 외부 변수를 13주 블록으로 시간 섞기"] = [OWN, CROSS, MACRO] + fams
    return S


GRID_MODELS = ["OLS", "Ridge", "Lasso", "Elastic Net", "PLS", "LightGBM"]


def grid_blocks(L: pd.DataFrame, P: pd.DataFrame, F: dict) -> dict:
    """블록 → {특성 이름: 주간 × 품목} — 모두 그 시점까지 자료로만 만든 값, 품목별 과거 표준화."""
    lp = np.log(P)
    R = LS._ret(P)
    B = {OWN: {}, CROSS: {}, MACRO: {}}
    for k in (1, 4, 13, 26, 52):
        B[OWN][f"모멘텀 {k}주"] = lp - lp.shift(k)
    B[OWN]["변동성 26주"] = R.rolling(26, min_periods=13).std()
    for c in P.columns:
        for k in (4, 52):
            B[CROSS][f"{c} {k}주"] = pd.DataFrame({i: (lp[c] - lp[c].shift(k)) for i in P.columns})
    for mid, (nm, tr) in MACRO_IDS.items():
        if mid not in L:
            continue
        w = L[mid].dropna().resample("W-FRI").last().reindex(P.index).ffill(limit=2)
        v = {"loglvl": np.log(w), "lvl": w, "logd4": 100 * np.log(w).diff(4), "d4": w.diff(4)}[tr].shift(1)     # 1주 늦춤 (발표 지연)
        B[MACRO][nm] = pd.DataFrame({i: v for i in P.columns})
    for fam, d in F.items():
        B[fam] = {nm: S.reindex(index=P.index, columns=P.columns) for nm, S in d.items()}
    return {b: {nm: _tz(S, 52) for nm, S in d.items()} for b, d in B.items()}


def _shuffle_rows(n: int, seed: int, block: int = 13) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks = [np.arange(s, min(s + block, n)) for s in range(0, n, block)]
    return np.concatenate([blocks[j] for j in rng.permutation(len(blocks))])


def _fit(model: str, Xtr, ytr, Xte):
    from sklearn import linear_model as lm
    from sklearn.model_selection import TimeSeriesSplit
    cv = TimeSeriesSplit(3, gap=H)
    if model == "OLS":
        b = np.linalg.lstsq(np.c_[np.ones(len(Xtr)), Xtr], ytr, rcond=None)[0]
        return np.c_[np.ones(len(Xte)), Xte] @ b
    if model == "Ridge":
        return lm.RidgeCV(alphas=np.logspace(-1, 4, 20)).fit(Xtr, ytr).predict(Xte)
    if model == "Lasso":
        return lm.LassoCV(cv=cv, alphas=20, max_iter=3000).fit(Xtr, ytr).predict(Xte)
    if model == "Elastic Net":
        return lm.ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], cv=cv, alphas=15, max_iter=3000).fit(Xtr, ytr).predict(Xte)
    if model == "PLS":
        from sklearn.cross_decomposition import PLSRegression
        best, bk = np.inf, 1
        for k in [c for c in (1, 2, 3, 5) if c <= Xtr.shape[1]]:
            e = 0.0
            for a, b in cv.split(Xtr):
                m = PLSRegression(k, scale=False).fit(Xtr[a], ytr[a]); e += float(((m.predict(Xtr[b]).ravel() - ytr[b]) ** 2).mean())
            if e < best:
                best, bk = e, k
        return PLSRegression(bk, scale=False).fit(Xtr, ytr).predict(Xte).ravel()
    if model == "LightGBM":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=50, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=5.0, n_jobs=1, verbose=-1)
        return m.fit(Xtr, ytr).predict(Xte)
    raise ValueError(model)


def _grid_item(item: str, cols_by_block: dict, sets: dict, y: pd.Series, idx: pd.DatetimeIndex, models: list, seed: int) -> dict:
    """한 품목: 묶음마다 매년 1월 재학습 → 그해 주간 예측. 반환 {(모델, 묶음): Series}."""
    out = {}
    yv = y.values
    years = sorted(set(idx[idx >= pd.Timestamp(GRID_START)].year))
    for sname, blocks in sets.items():
        X = pd.concat([cols_by_block[b] for b in blocks], axis=1).fillna(0.0)
        if sname.startswith("⑧"):                                         # 위약: 자기 정보 외 변수의 행 순서를 블록 단위로 섞음
            ext = [c for c in X.columns if c not in cols_by_block[OWN].columns]
            X.loc[:, ext] = X[ext].values[_shuffle_rows(len(X), seed)]
        Xv = X.values
        valid = (idx >= pd.Timestamp(TRAIN_START) + pd.Timedelta(weeks=52)) & np.isfinite(yv) & cols_by_block[OWN].notna().all(1).values
        preds = {m: pd.Series(np.nan, index=idx) for m in models}
        for yr in years:
            cut = np.flatnonzero(idx < pd.Timestamp(f"{yr}-01-01"))
            if not len(cut):
                continue
            tr = np.flatnonzero(valid[:cut[-1] - H + 1])                 # 목표(4주 뒤)가 학습 끝 전에 실현된 행만
            te = np.flatnonzero((idx.year == yr) & cols_by_block[OWN].notna().all(1).values)
            if len(tr) < 156 or not len(te):
                continue
            mu, sd = Xv[tr].mean(0), Xv[tr].std(0); sd[sd == 0] = 1
            Xtr, Xte = (Xv[tr] - mu) / sd, (Xv[te] - mu) / sd
            ym = yv[tr].mean()
            for m in models:
                try:
                    preds[m].iloc[te] = _fit(m, Xtr, yv[tr] - ym, Xte) + ym
                except Exception:
                    pass
        for m in models:
            out[(m, sname)] = preds[m]
    return out


def grid_tabular(L: pd.DataFrame, models=GRID_MODELS, n_jobs: int = 6, progress=print) -> tuple[pd.DataFrame, dict]:
    from joblib import Parallel, delayed
    P = AT.weekly_px(L, TRAIN_START)
    F = mm_features(L)
    B = grid_blocks(L, P, F)
    sets = _sets(list(F))
    lp = np.log(P)
    fwd = 100 * (lp.shift(-H) - lp)
    jobs = []
    for j, i in enumerate(P.columns):
        cb = {b: pd.DataFrame({nm: S[i] for nm, S in d.items() if not (b == CROSS and nm.startswith(i + " "))}) for b, d in B.items()}
        jobs.append(delayed(_grid_item)(i, cb, sets, fwd[i], P.index, list(models), j))
    progress(f"표준 모델 그리드: 품목 {len(jobs)} × 묶음 {len(sets)} × 모델 {len(models)} (매년 재학습)")
    res = Parallel(n_jobs=n_jobs, verbose=5)(jobs)
    preds = {}
    for i, r in zip(P.columns, res):
        for k, s in r.items():
            preds.setdefault(k, {})[i] = s
    return P, {k: pd.DataFrame(v) for k, v in preds.items()}


def common_exog(L: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    """대안 모델용 공통(모든 품목 같은 값) 외부 변수: 매크로 3종 + 기상 지수 3종 + FRED 금융여건 — 주간, 과거 표준화, 1주 늦춤."""
    import multimodal as MMd
    M = AT.macro_weekly(L).reindex(P.index)
    Wx = MMd.weather_weekly(P.index)
    if "gistemp" not in Wx and "gtemp_noaa" in Wx:
        Wx["gistemp"] = Wx["gtemp_noaa"]
    for v in ("oni", "soi", "gistemp"):
        if v in Wx:
            M[v] = _tz(Wx[v], 104).shift(1)
    fe = pd.read_parquet(MMd.MM / "fred_extra.parquet")
    fe["date"] = pd.to_datetime(fe.date)
    for sid in ("NFCI", "STLFSI4", "DFII10", "T5YIFR", "T10Y3M"):
        s = fe[fe.id == sid].set_index("date").value.sort_index()
        if len(s):
            w = s.resample("W-FRI").last().reindex(P.index).ffill(limit=8)
            M[sid] = _tz(w, 104).shift(1)
    return M.fillna(0.0)


def alt_jobs(L: pd.DataFrame, P: pd.DataFrame) -> list:
    """(이름, 설명변수 묶음, 함수) — 모두 2007년부터 매년 재학습(신경망) 또는 매주 적합(Prophet·zero-shot)."""
    mac = AT.macro_weekly(L)
    cx = common_exog(L, P)
    comp = mm_composite(mm_features(L))
    y0 = int(GRID_START[:4])
    J = [("Prophet 방식 (자체 구현)", "설명변수 없음", lambda: AT.prophet_like(P, start=GRID_START)),
         ("Prophet 방식 (자체 구현)", "품목별 멀티모달 7갈래", lambda: prophet_like_x(P, comp, start=GRID_START))]
    for k, nm in (("patch", "PatchTST 방식 (경량)"), ("itrans", "iTransformer 방식 (경량)"), ("tft", "TFT 방식 (경량)"), ("node", "Neural ODE 방식 (경량)"),
                  ("gnn", "시공간 GNN 방식 (경량)")):
        J.append((nm, "설명변수 없음", lambda k=k: AT.nn_pred(P, k, first_year=y0)))
    for k, nm in (("itrans", "iTransformer 방식 (경량)"), ("tft", "TFT 방식 (경량)"), ("gnn", "시공간 GNN 방식 (경량)")):
        J.append((nm, "매크로 3종", lambda k=k: AT.nn_pred(P, k, first_year=y0, macro=mac)))
        J.append((nm, "매크로 + 공통 멀티모달 (기상·금융 지수)", lambda k=k: AT.nn_pred(P, k, first_year=y0, macro=cx)))
    J.append(("TFT 방식 (경량)", "품목별 멀티모달 7갈래", lambda: AT.nn_pred(P, "tft", first_year=y0, item_exog=comp)))
    for nm in ("Chronos-Bolt-small", "TimesFM-2.5", "Chronos-2"):
        J.append((f"{nm} (zero-shot)", "설명변수 없음", lambda nm=nm: fm_pred(nm, L, P, start=GRID_START)))
    # neuralforecast 정식판은 2년마다 재학습 (매년이면 모델당 수 시간 — 경량판·표준 모델은 매년)
    J.append(("PatchTST (neuralforecast)", "설명변수 없음", lambda: AT.nf_pred(P, "PatchTST", first_year=y0, refit_years=2)))
    for nm in ("iTransformer", "TFT"):
        J.append((f"{nm} (neuralforecast)", "설명변수 없음", lambda nm=nm: AT.nf_pred(P, nm, first_year=y0, refit_years=2)))
        J.append((f"{nm} (neuralforecast)", "매크로 3종", lambda nm=nm: AT.nf_pred(P, nm, first_year=y0, macro=mac, refit_years=2)))
        J.append((f"{nm} (neuralforecast)", "매크로 + 공통 멀티모달 (기상·금융 지수)", lambda nm=nm: AT.nf_pred(P, nm, first_year=y0, macro=cx, refit_years=2)))
    J.append(("TFT (neuralforecast)", "품목별 멀티모달 7갈래", lambda: AT.nf_pred(P, "TFT", first_year=y0, exog_panel=comp, refit_years=2)))
    return J


def prophet_like_x(P: pd.DataFrame, reg: dict, win: int = 156, n_cp: int = 20, alpha: float = 2e-4, start: str = GRID_START) -> pd.DataFrame:
    """Prophet 방식 + 품목별 회귀변수: 로그가격 = 추세 변곡 + 연간 계절 + Σ β·x(t−4). 회귀변수를 4주 늦춰 넣어 '4주 뒤 적합값'에 필요한 x가 지금 아는 값이 되게 함."""
    from sklearn.linear_model import Lasso
    lp = np.log(P)
    out = pd.DataFrame(np.nan, index=P.index, columns=P.columns)
    Rg = {k: S.reindex(index=P.index, columns=P.columns).ffill().fillna(0.0) for k, S in reg.items()}

    def four(dates):
        a = 2 * np.pi * pd.DatetimeIndex(dates).dayofyear.values / 365.25
        return np.column_stack([f(k * a) for k in (1, 2, 3) for f in (np.sin, np.cos)])
    cps = np.linspace(0, 0.8, n_cp + 1)[1:]
    tf = np.arange(win + H) / win
    for e in np.flatnonzero(P.index >= pd.Timestamp(start)):
        if e < win + H:
            continue
        s = slice(e - win + 1, e + 1)
        dts = list(P.index[s]) + [P.index[e] + pd.Timedelta(weeks=k) for k in range(1, H + 1)]
        base = np.column_stack([tf] + [np.maximum(tf - c, 0) for c in cps] + [four(dts)])
        for c in P.columns:
            y = lp[c].values[s]
            if np.isnan(y).sum() > win * 0.2:
                continue
            xr = np.column_stack([Rg[k][c].values[e - win + 1 - H:e + 1] for k in Rg])     # x(t−4): 길이 win + 4 (마지막 4개 = 지금까지 값)
            Xf = np.column_stack([base, xr])
            ok = ~np.isnan(y)
            mdl = Lasso(alpha=alpha, max_iter=5000).fit(Xf[:win][ok], y[ok] - np.nanmean(y))
            yh = mdl.predict(Xf)
            out.iloc[e, out.columns.get_loc(c)] = yh[win - 1 + H] - yh[win - 1]
    return out


def fm_pred(name: str, L: pd.DataFrame, P: pd.DataFrame, start: str = GRID_START) -> pd.DataFrame:
    """파운데이션 모델 zero-shot (alt_ts.fm_pred와 같은 변환, 시작일만 2007로) — 캐시 data/fm/{name}_grid07.npz."""
    import forecast_lab as FL
    series = {i: np.log(L[i].dropna()) for i in P.columns}
    dates = {i: FL.fridays(s.index, start=start, end_pad=0) for i, s in series.items()}
    q = FL.fm_forecast(name, "grid07", series, dates, 20)
    out = pd.DataFrame(np.nan, index=P.index, columns=P.columns)
    for i in P.columns:
        if i not in q or not len(q[i]):
            continue
        lab = dates[i].to_period("W-FRI").to_timestamp(how="end").normalize()
        v = pd.Series(q[i][:, -1, 4] - series[i].reindex(dates[i]).values, index=lab)
        out[i] = v[~v.index.duplicated(keep="last")].reindex(out.index)
    return out


def grid_alt(L: pd.DataFrame, P: pd.DataFrame, cache: dict | None = None, progress=print, on_done=None) -> dict:
    preds = dict(cache or {})
    for nm, xs, fn in alt_jobs(L, P):
        key = (nm, xs)
        if key in preds:
            continue
        progress(f"대안 모델: {nm} · {xs}")
        try:
            preds[key] = fn()
            if on_done:
                on_done(preds)
        except Exception as e:
            print("실패", key, e, flush=True)
    return preds


def grid_table(P: pd.DataFrame, tab: dict, alt: dict, cost_bp: float = 5.0, extra: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """전략마다: 방향 매매·순위 롱숏 Sharpe(전체·하위기간·최근 5년), IC, 적중률, t, 52주 모멘텀 대비 차이. extra = {모델군: {(모델, 설명변수): 예측}}."""
    lp = np.log(P)
    allp = {("표준", m, s): v for (m, s), v in tab.items()}
    allp.update({("대안 시계열", m, s): v for (m, s), v in alt.items()})
    for g, d in (extra or {}).items():
        allp.update({(g, m, s): v for (m, s), v in d.items()})
    allp[("기준", "52주 모멘텀", "자기 정보 (52주 수익률 부호)")] = lp - lp.shift(52)
    base = {md: AT.signal_book(allp[("기준", "52주 모멘텀", "자기 정보 (52주 수익률 부호)")], P, md, cost_bp).loc[GRID_START:FROZEN_END] for md in ("dir", "xs")}
    rows, W = [], {}
    for (grp, m, s), pr in allp.items():
        pr = pr.reindex(index=P.index, columns=P.columns)
        first = pr.dropna(how="all").index.min()
        ev = AT.eval_pred(pr.loc[GRID_START:], P)
        row = {"모델군": grp, "모델": m, "설명변수": s, "예측 시작": f"{first:%Y-%m}" if pd.notna(first) else "–",
               "IC (주별 순위 상관)": ev["IC (주별 순위 상관)"], "방향 적중률": ev["방향 적중률"]}
        for md, lab in (("dir", "방향"), ("xs", "순위 롱숏")):
            r = AT.signal_book(pr, P, md, cost_bp).loc[GRID_START:FROZEN_END]
            r = r.loc[r.index >= (first if pd.notna(first) else r.index[0])]
            W[f"{m} · {s} | {lab}"] = r
            mt = LS.metrics(r)
            d = (r - base[md].reindex(r.index)).dropna()
            row.update({f"{lab} Sharpe": mt["Sharpe"], f"{lab} 2007–2016": LS.metrics(r[:"2016-12-31"])["Sharpe"],
                        f"{lab} 2017–2021-10": LS.metrics(r["2017-01-01":RECENT])["Sharpe"], f"{lab} 최근 5년": LS.metrics(r[RECENT:])["Sharpe"],
                        f"{lab} 공통 2012-07~": LS.metrics(r["2012-07-01":])["Sharpe"], f"{lab} 2017~": LS.metrics(r["2017-01-01":])["Sharpe"],
                        f"{lab} t값": mt["Sharpe"] * np.sqrt(len(r) / 52) if np.isfinite(mt["Sharpe"]) else np.nan,
                        f"{lab} 최대낙폭 (%)": mt["최대낙폭 (%)"],
                        f"{lab} 52주 모멘텀 대비 차이 t": float(d.mean() / d.std() * np.sqrt(len(d))) if len(d) > 52 and d.std() > 0 else np.nan})
        rows.append(row)
    T, Wd = pd.DataFrame(rows), pd.DataFrame(W)
    # 같은 모델의 위약(설명변수 시간 섞기) 대비: 표준 = ⑧, 패널 = P⑤ (같은 모델군·같은 모델). 차이 t = 주간 수익 차이의 t값
    plk = {"표준": "⑧", "패널": "P⑤"}
    for lab in ("방향", "순위 롱숏"):
        dif, dt = [], []
        for _, r_ in T.iterrows():
            pk = plk.get(r_["모델군"])
            pl = T[(T["모델군"] == r_["모델군"]) & (T["모델"] == r_["모델"]) & T["설명변수"].str.startswith(pk)] if pk else T.iloc[:0]
            if not len(pl) or r_["설명변수"].startswith(pk):
                dif.append(np.nan); dt.append(np.nan); continue
            a = Wd[f"{r_['모델']} · {r_['설명변수']} | {lab}"]; b = Wd[f"{r_['모델']} · {pl['설명변수'].iloc[0]} | {lab}"]
            d = (a - b).dropna()
            dif.append(float(r_[f"{lab} Sharpe"] - pl[f"{lab} Sharpe"].iloc[0]))
            dt.append(float(d.mean() / d.std() * np.sqrt(len(d))) if len(d) > 52 and d.std() > 0 else np.nan)
        T[f"{lab} 위약 대비 차이"] = dif
        T[f"{lab} 위약 대비 차이 t"] = dt
    return T, Wd


# ============================================================================
# MIDAS 탭 비교용 — 같은 월말 결정일에 '다음 달(21영업일) 수익률'을 대안 모델로 예측
# ============================================================================
MIDAS_ALT = {"Prophet (직전 3년 주간 가격, 추세·계절)": "prophet", "Chronos-Bolt-small (zero-shot)": "Chronos-Bolt-small",
             "TimesFM-2.5 (zero-shot)": "TimesFM-2.5", "Chronos-2 (zero-shot)": "Chronos-2"}


def midas_alt(label: str, r_daily: pd.Series, dec: pd.DatetimeIndex, h: int = 21) -> pd.Series:
    """r_daily = 일간 로그수익(%) · dec = 월말 결정일 → 결정일까지 자료만으로 다음 h영업일 누적 수익(%) 예측."""
    kind = MIDAS_ALT[label]
    lp = r_daily.fillna(0.0).cumsum() / 100.0
    pos = lp.index.get_indexer(dec)
    out = {}
    if kind == "prophet":
        import logging
        from prophet import Prophet
        for lg in ("cmdstanpy", "prophet"):
            logging.getLogger(lg).setLevel(logging.ERROR)
        wk = lp.resample("W-FRI").last()
        for d in dec:
            y = wk.loc[:d].iloc[-156:]
            if len(y) < 104:
                continue
            m = Prophet(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=3, uncertainty_samples=0)
            m.fit(pd.DataFrame({"ds": y.index, "y": y.values}))
            f = m.predict(pd.DataFrame({"ds": [d, d + pd.Timedelta(days=round(h * 7 / 5))]}))
            out[d] = float(f.yhat.iloc[1] - f.yhat.iloc[0]) * 100.0
    else:
        import forecast_lab as FL
        ok = [(d, p) for d, p in zip(dec, pos) if p >= 128]
        Q = FL._load(kind)([lp.values[max(0, p - 511):p + 1] for _, p in ok], h)
        out = {d: float((Q[j, h - 1, 4] - lp.values[p]) * 100.0) for j, (d, p) in enumerate(ok)}
    return pd.Series(out)


# ============================================================================
# 멀티모달 자료 목록 (⑧ 데이터·방법론) · 엑셀 내보내기
# ============================================================================
MM_DIR = ROOT / "data" / "mm"
_SRC = [  # (갈래, 자료, 출처, 빈도, 파일, 날짜 열, 대상 품목, 발표 시점 처리)
    ("텍스트 (FinBERT)", "WASDE 서술 문단 감성", "USDA WASDE 월간 PDF (원문 data/reports/raw)", "월간", "wasde_text.parquet", "release",
     "밀·옥수수·대두·대두박·대두유·설탕·생우·돈육·면화", "발표 12:00 ET → 그날 금요일 종가부터"),
    ("텍스트 (FinBERT)", "EIA STEO 서술 감성", "EIA 단기에너지전망 월간 보고서 본문", "월간", "steo_text.parquet", "vintage", "WTI·휘발유·난방유·천연가스",
     "발표일 이후"),
    ("텍스트 (FinBERT)", "연준 베이지북 감성 (에너지·금속·농업 문장)", "미니애폴리스 연준 베이지북 아카이브", "연 8회", "beige.parquet", "month",
     "에너지·금속·농산물 품목", "발표 월 다음 주부터"),
    ("텍스트 (FinBERT)", "뉴스 헤드라인 감성", "GDELT 2.0 DOC API — 요청 제한으로 수집 중단, 분석에서 제외", "주간 집계", "gdelt_titles.parquet", "date", "19개 품목", "기사 날짜 이후"),
    ("리포트 수치", "WASDE 기말재고·재고율·생산 수정폭", "USDA WASDE 수치표", "월간", "../reports/wasde_features.parquet", "release", "곡물·유지·설탕·면화",
     "발표 12:00 ET → 그날 금요일 종가부터"),
    ("리포트 수치", "EIA STEO 3개월 가격 전망 수정폭", "EIA STEO 엑셀 (원문 data/reports/raw)", "월간", "../reports/steo_features.parquet", "avail",
     "WTI·휘발유·난방유·천연가스", "발표일 이후"),
    ("리포트 수치", "EIA 주간 재고 (원유·휘발유·증류유·천연가스)", "EIA Open Data API v2", "주간", "eia_weekly.parquet", "date", "WTI·휘발유·난방유·천연가스",
     "기준 주 + 7일 (수요일 발표)"),
    ("리포트 수치", "NASS 작황 우량 비율 (good + excellent)", "USDA NASS Quick Stats API", "주간 (시즌 중)", "nass_condition_raw.parquet", "week_ending",
     "옥수수·대두·대두박·대두유·밀·면화", "주말 기준 + 2일 (월요일 발표), 최대 21일 유지"),
    ("리포트 수치", "LME 구리 재고 · 금/은 ETF 보유량", "akshare (LME·SPDR GLD·iShares SLV)", "일간", "metals.parquet", "date", "구리·금·은", "다음 영업일"),
    ("포지션 (CFTC COT)", "투기·상업 순포지션, 쏠림, 4주 변화", "CFTC Legacy Futures Only (1986–2016 압축 + Socrata API)", "주간", "cot19.parquet", "date",
     "19개 품목", "기준 화요일 + 10일 (금요일 발표 후 다음 주)"),
    ("기상", "엘니뇨 ONI · 남방진동 SOI · 지구 기온 편차", "NOAA CPC · NOAA NCEI (NASA GISTEMP는 접속 실패 시 NOAA로 대체)", "월간", "weather.parquet", "date",
     "19개 품목 (품목별 과거 민감도 β × 값)", "발표 지연 반영 (avail 열)"),
    ("기상", "주산지 22곳 일 기온·강수 → 4주 강수 z·기온 편차·냉난방 수요", "Open-Meteo Historical (ERA5)", "일간 → 주간", "openmeteo.parquet", "date",
     "곡물·소프트·축산(주산지), 천연가스·난방유(수요지)", "결정일 6일 전까지 관측"),
    ("금융 (FRED 추가) · 월간 거시", "실질금리·기대인플레·금리차·환율·금융여건·실업수당·연준 자산·OVX·GVZ / CPI·실업률·산업생산·M2·심리·PPI·고용",
     "FRED (세인트루이스 연준)", "일·주·월간", "fred_extra.parquet", "date", "19개 품목 (품목별 과거 민감도 β × 값)", "주간 1주, 월간 45일 늦춤 (월간은 최신 수정치)"),
]


def mm_catalog() -> pd.DataFrame:
    rows = []
    for fam, nm, src, fq, fn, dc, items, lag in _SRC:
        p = (MM_DIR / fn).resolve()
        st_ = en = "–"; n = 0
        if p.exists():
            d = pd.read_parquet(p)
            t = pd.to_datetime(d[dc], errors="coerce", utc=True).dt.tz_localize(None).dropna()
            if len(t):
                st_, en = f"{t.min():%Y-%m-%d}", f"{t.max():%Y-%m-%d}"
            n = len(d)
        rows.append({"갈래": fam, "자료": nm, "출처": src, "빈도": fq, "시작": st_, "끝": en, "행 수": n, "대상 품목": items, "발표 시점 처리": lag,
                     "저장 파일": "data/" + str(p.relative_to(ROOT / "data")).replace("\\", "/")})
    return pd.DataFrame(rows)


def export_mm_excel(path: Path | None = None) -> Path:
    """멀티모달 원자료를 엑셀 한 파일로 (시트별 원자료 + 목록). 텍스트는 FinBERT 점수와 헤드라인(GDELT)까지 — 보고서 원문(PDF·HTML)은 data/reports/raw."""
    path = path or (ROOT / "data" / "excel" / "멀티모달_원자료.xlsx")
    rd = lambda f: pd.read_parquet(MM_DIR / f) if (MM_DIR / f).exists() else pd.DataFrame()   # noqa: E731
    sheets = {"목록": mm_catalog()}
    sheets["WASDE_서술감성"] = rd("wasde_text.parquet")
    sheets["STEO_서술감성"] = rd("steo_text.parquet")
    sheets["베이지북_감성"] = rd("beige.parquet")
    g = rd("gdelt_titles.parquet")
    if len(g):
        sc = rd("gdelt_scored.parquet")
        sheets["GDELT_헤드라인"] = g.merge(sc[["id", "title", "score"]].drop_duplicates(["id", "title"]), on=["id", "title"], how="left") if len(sc) else g
    sheets["WASDE_수치"] = pd.read_parquet(ROOT / "data" / "reports" / "wasde_features.parquet")
    sheets["STEO_수치"] = pd.read_parquet(ROOT / "data" / "reports" / "steo_features.parquet")
    e = rd("eia_weekly.parquet")
    if len(e):
        sheets["EIA_주간재고"] = e.assign(자료=e.sid.map({k: v[0] for k, v in __import__("multimodal").EIA_SERIES.items()})).pivot_table(
            index="date", columns="자료", values="value").reset_index()
    n = rd("nass_condition_raw.parquet")
    if len(n):
        sheets["NASS_작황"] = n[["week_ending", "commodity_desc", "class_desc", "short_desc", "unit_desc", "Value", "load_time"]]
    sheets["금속_재고"] = rd("metals.parquet")
    sheets["CFTC_COT_19"] = rd("cot19.parquet")
    w = rd("weather.parquet")
    if len(w):
        sheets["기상지수"] = w
    om = rd("openmeteo.parquet")
    if len(om):
        sheets["OpenMeteo_일별"] = om
    f = rd("fred_extra.parquet")
    if len(f):
        sheets["FRED_추가"] = f.pivot_table(index="date", columns="id", values="value").reset_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for nm, d in sheets.items():
            d = d.copy()
            for c in d.columns:
                if d[c].dtype == object or str(d[c].dtype).startswith("str"):
                    d[c] = d[c].astype(str).str.slice(0, 32000)
            for c in d.select_dtypes(include=["datetimetz"]).columns:
                d[c] = d[c].dt.tz_localize(None)
            d.to_excel(xw, sheet_name=nm[:31], index=False)
    return path


# ============================================================================
# 4) 그리드 확장 — 패널 모델 (여러 품목 탭) · MIDAS (혼합 주기 탭)를 같은 19개 품목·같은 매매 규칙으로
# ============================================================================
PANEL_MACRO = ("yf:DX-Y.NYB", "fred:VIXCLS", "fred:DGS10", "fred:BAA10Y")


def _panel_sets(chars0: list, macro_chars: list, mm_chars: list, sum_chars: list) -> dict:
    return {"P① 기본 특성 (모멘텀·반전·변동성·왜도·롤수익·COT·계절)": chars0,
            "P② 기본 + 거시 β×Δ (달러·VIX·10년 금리·신용)": chars0 + macro_chars,
            "P③ 기본 + 멀티모달 전부": chars0 + mm_chars,
            "P③-요약 기본 + 멀티모달 7갈래 요약 (갈래별 과거 표준화 평균)": chars0 + sum_chars,
            "P④ 기본 + 거시 + 멀티모달": chars0 + macro_chars + mm_chars,
            "P⑤ 위약 · P④의 거시·멀티모달 특성을 13주 블록으로 시간 섞기": chars0 + macro_chars + mm_chars}


def _to_weekly(pred: pd.Series, P: pd.DataFrame) -> pd.DataFrame:
    """(결정일, 품목) 예측 → 주간(W-FRI) × 품목 표."""
    W = pred.unstack("id")
    W.index = pd.DatetimeIndex(W.index).to_period("W-FRI").to_timestamp(how="end").normalize()
    W = W[~W.index.duplicated(keep="last")]
    return W.reindex(index=P.index, columns=P.columns)


def grid_panel(L_adj: pd.DataFrame, L_raw: pd.DataFrame, P: pd.DataFrame, units: dict, models=None, progress=print, cache: dict | None = None,
               on_done=None) -> dict:
    """여러 품목 탭의 패널 모델(시점별 순위 특성 → 다음 주 변동성 표준화 상대 수익) · 26주마다 재학습 · 최소 학습 156주."""
    import panel as PN
    models = models or PN.PANEL_MODELS
    F = mm_features(L_adj)
    mm = {"MM · " + nm: S for d in F.values() for nm, S in d.items()}
    mm.update({"MM요약 · " + fam: S for fam, S in mm_composite(F).items()})   # 7갈래 요약 (특성 수를 54 → 7로 줄인 판)
    pan = PN.build_panel(L_adj, L_raw, list(P.columns), units, "W-FRI", "futures", "vol_cs", PANEL_MACRO, None, mm=mm)
    cols = list(pan["X"].columns)
    macro_c = [c for c in cols if c.startswith("거시")]
    mm_c = [c for c in cols if c.startswith("MM · ")]
    sum_c = [c for c in cols if c.startswith("MM요약 · ")]
    chars0 = [c for c in PN.CHARS if c in cols]
    sets = _panel_sets(chars0, macro_c, mm_c, sum_c)
    # 위약판: 거시·멀티모달 특성의 날짜를 13주 블록 단위로 섞음 (품목 간 순위 구조는 유지, 시점 정보만 제거)
    X = pan["X"]
    ud = np.unique(X.index.get_level_values("date"))
    perm = ud[_shuffle_rows(len(ud), 7)]
    mp = dict(zip(ud, perm))
    sh = X[macro_c + mm_c + sum_c].copy()
    src = sh.copy(); src.index = pd.MultiIndex.from_arrays([[mp[d] for d in X.index.get_level_values("date")], X.index.get_level_values("id")])
    sh.loc[:, :] = src.reindex(X.index).fillna(0.0).values
    pan_pl = dict(pan, X=pd.concat([X[chars0], sh], axis=1)[cols])
    out = dict(cache or {})
    for sname, chars in sets.items():
        for m in models:
            key = (m, sname)
            if key in out:
                continue
            progress(f"패널: {m} · {sname[:12]}")
            try:
                wf = PN.walk_forward_panel(pan_pl if sname.startswith("P⑤") else pan, m, chars, refit=26, min_train=156)
                out[key] = _to_weekly(wf["pred"], P)
                if on_done:
                    on_done(out)
            except Exception as e:
                print("패널 실패", key, e, flush=True)
    return out


def _midas_item(i: str, r: pd.Series, regs: dict, K: int) -> dict:
    import midas as M
    try:
        o = M.midas_return_wf(r, regs, K)
    except Exception as e:
        print("MIDAS 실패", i, e, flush=True)
        return {}
    return {"MIDAS (Beta 가중)": o["table"]["MIDAS"], "동일가중 (월 평균 집계)": o["table"]["동일가중"]}


def grid_midas(L: pd.DataFrame, P: pd.DataFrame, units: dict, K: int = 63, n_jobs: int = 6, progress=print, cache: dict | None = None,
               on_done=None) -> dict:
    """MIDAS 탭과 같은 방식: 월말마다 그때까지 자료로 재추정 → 다음 달 수익 예측 → 그 달 동안 매주 같은 예측으로 매매."""
    import analytics as A
    from joblib import Parallel, delayed
    ids = list(P.columns)
    mac = ["yf:DX-Y.NYB", "fred:VIXCLS", "fred:DGS10"]
    R = A.make_panel(L, ids + mac, units, "B", "logret", pd.Timestamp(TRAIN_START), pd.Timestamp(FROZEN_END))
    comp = mm_composite(mm_features(L))
    allc = pd.concat(comp.values(), keys=range(len(comp))).groupby(level=1).mean()       # 7갈래 요약의 평균 (품목별 멀티모달 종합)
    variants = {"M① 자기 일간 수익률": lambda i: {"자기": R[i]},
                "M② 자기 + 거시 일간 (달러·VIX·10년 금리)": lambda i: {"자기": R[i], **{m: R[m] for m in mac}},
                "M③ 자기 + 멀티모달 종합 (7갈래 요약 평균)": lambda i: {"자기": R[i], "멀티모달": allc[i].reindex(allc.index.union(R.index)).ffill().reindex(R.index).fillna(0.0)}}
    out = dict(cache or {})
    for vn, mk in variants.items():
        if all((m, vn) in out for m in ("MIDAS (Beta 가중)", "동일가중 (월 평균 집계)")):
            continue
        progress(f"MIDAS: {vn}")
        res = Parallel(n_jobs=n_jobs, verbose=0)(delayed(_midas_item)(i, R[i], mk(i), K) for i in ids)
        for m in ("MIDAS (Beta 가중)", "동일가중 (월 평균 집계)"):
            W = pd.DataFrame({i: r[m] for i, r in zip(ids, res) if r})
            W.index = pd.DatetimeIndex(W.index)
            Wk = W.reindex(W.index.union(P.index)).ffill(limit=6).reindex(P.index)       # 월말 결정 → 다음 달 매주 같은 예측
            out[(m, vn)] = Wk
        if on_done:
            on_done(out)
    return out
