"""예측 실험실 — 최신 시계열 파운데이션 모델 vs 기준 모델을 '목표별'로 walk-forward 비교 + 과적합 판정 + 메타 라벨링.

파운데이션 모델 (zero-shot, 사전학습만 · 미세조정 없음, 분위 예측 0.1~0.9):
  Chronos-Bolt-small (Amazon, 48M, Apache-2.0) · Chronos-2 (Amazon, 120M, 공변량·다변량, Apache-2.0)
  TimesFM 2.5 (Google Research, 200M, Apache-2.0) · TiRex (NX-AI, 35M xLSTM, 자체 라이선스 — 연구용, CPU 느려 월 1회 결정일만)
기준: 과거 평균 · 모멘텀(12-1개월) · 단기 반전(1개월) · LightGBM 글로벌 모델(9품목 함께 학습, 매년 재학습, Purged K-fold로 하이퍼파라미터 선택)
목표 (매주 금요일 결정, 그 시점까지의 데이터만 컨텍스트):
  방향  : 향후 5영업일 수익률 부호 → 부호전략 Sharpe (주간 비중첩) · 적중률 · 위약 기준선 · DSR · PBO
  변동성: 향후 20영업일 실현분산 → QLIKE (GARCH·품목별 EWMA·HAR 대비)
  스프레드: 상대가치 관계(크러시 마진·가격비·수입 차익·고로 마진)의 20영업일 후 변화 → 랜덤워크 대비 MAE 비율·방향 적중·IC
  순위  : 9품목의 향후 20영업일 수익률 순위 → 횡단면 Spearman IC · 상위3−하위3 스프레드
  구매  : '향후 20영업일 평균가 < 오늘 가격' (기다리면 더 싸게 살 수 있나) → AUC · Brier · 결정 가치(bp, 기다림/즉시 매수 선택 vs 균등 매수)
메타 라벨링: 1차 신호 = ★ 구매 타이밍 '추천' 점수. 2차 LightGBM이 '이번 주 1차 신호의 방향(더 사라/덜 사라)이 맞을 확률' p를 예측 →
  점수' = 0.5 + (점수 − 0.5)·2p (p=0.5면 그대로, p→1이면 확신 2배, p→0이면 중립) → 같은 구매 백테스트로 절감(bp) 비교.
파운데이션 모델 예측은 data/fm/에 캐시 (재실행 시 재사용).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

os.environ.setdefault("TIREX_NO_CUDA", "1")
ROOT = Path(__file__).parent
CACHE = ROOT / "data" / "fm"
QS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FM_NAMES = ["Chronos-Bolt-small", "Chronos-2", "TimesFM-2.5", "TiRex"]
BASE_NAMES = ["과거 평균", "모멘텀 (12-1개월)", "단기 반전 (1개월)", "LightGBM (글로벌, purged CV)"]
H = 20
CTX = 512
EVAL_START = "2012-01-01"


# ============================================================================
# 파운데이션 모델 래퍼: forecast(list[np.ndarray], h) → (n, h, 9) 분위
# ============================================================================
_LOADED: dict = {}


def _load(name: str):
    if name in _LOADED:
        return _LOADED[name]
    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    if name in ("Chronos-Bolt-small", "Chronos-2"):
        from chronos import BaseChronosPipeline
        mid = {"Chronos-Bolt-small": "amazon/chronos-bolt-small", "Chronos-2": "amazon/chronos-2"}[name]
        pipe = BaseChronosPipeline.from_pretrained(mid, device_map="cpu", torch_dtype=torch.float32)

        def f(ctx, h):
            out = []
            for k in range(0, len(ctx), 64):
                part = ctx[k:k + 64]
                if name == "Chronos-2":
                    q, _ = pipe.predict_quantiles([torch.tensor(c, dtype=torch.float32)[None, :] for c in part], prediction_length=h, quantile_levels=QS)
                    out.append(np.stack([np.asarray(x)[0] for x in q]))
                else:
                    q, _ = pipe.predict_quantiles([torch.tensor(c, dtype=torch.float32) for c in part], prediction_length=h, quantile_levels=QS)
                    out.append(np.asarray(q))
            return np.concatenate(out)
    elif name == "TimesFM-2.5":
        import timesfm
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        m.compile(timesfm.ForecastConfig(max_context=CTX, max_horizon=32, per_core_batch_size=64, normalize_inputs=True,
                                         use_continuous_quantile_head=True, fix_quantile_crossing=True))

        def f(ctx, h):
            _, qt = m.forecast(horizon=h, inputs=[np.asarray(c, float) for c in ctx])
            return np.asarray(qt)[:, :h, 1:10]                      # 열 0 = 평균, 1..9 = 0.1..0.9 분위
    elif name == "TiRex":
        import torch
        from tirex import load_model
        tm = load_model("NX-AI/TiRex", device="cpu")

        def f(ctx, h):
            L = max(len(c) for c in ctx)
            X = np.full((len(ctx), L), np.nan)
            for j, c in enumerate(ctx):
                X[j, L - len(c):] = c
            q, _ = tm.forecast(context=torch.tensor(X, dtype=torch.float32), prediction_length=h, batch_size=128)
            return np.asarray(q)
    else:
        raise ValueError(name)
    _LOADED[name] = f
    return f


def fm_forecast(name: str, key: str, series: dict[str, pd.Series], dates: dict[str, pd.DatetimeIndex], h: int, ctx_len: int = CTX,
                progress=None) -> dict[str, np.ndarray]:
    """품목별 결정일마다 '그 날까지'의 컨텍스트로 h기 예측 분위 (n, h, 9). 캐시: data/fm/{name}_{key}.npz"""
    CACHE.mkdir(parents=True, exist_ok=True)
    fn = CACHE / f"{name}_{key}.npz"
    sig = {i: (len(series[i]), str(series[i].index[-1]), str(dates[i][-1]) if len(dates[i]) else "", len(dates[i])) for i in series}
    if fn.exists():
        z = np.load(fn, allow_pickle=True)
        if z["sig"].item() == sig:
            return z["out"].item()
    f = _load(name); out = {}
    for j, (i, s) in enumerate(series.items()):
        s = s.dropna(); pos = s.index.get_indexer(dates[i])
        ctx = [s.values[max(0, p - ctx_len + 1):p + 1] for p in pos]
        out[i] = f(ctx, h) if ctx else np.zeros((0, h, 9))
        if progress:
            progress(f"{name} · {key} · {i} ({j + 1}/{len(series)})")
    np.savez(fn, out=np.array(out, dtype=object), sig=np.array(sig, dtype=object))
    return out


# ============================================================================
# 데이터·결정일·특성
# ============================================================================
def fridays(idx: pd.DatetimeIndex, start=EVAL_START, end_pad: int = H, every: int = 1) -> pd.DatetimeIndex:
    d = pd.Series(idx, index=idx).groupby(idx.to_period("W-FRI")).max()
    d = pd.DatetimeIndex(d.values)
    d = d[(d >= pd.Timestamp(start)) & (d <= idx[-1 - end_pad])]
    return d[::every]


def _features(lp: pd.Series, sig: pd.DataFrame) -> pd.DataFrame:
    r = lp.diff()
    X = pd.DataFrame(index=lp.index)
    for k in [1, 5, 20, 60, 252]:
        X[f"r{k}"] = lp - lp.shift(k)
    X["mom12_1"] = lp.shift(21) - lp.shift(252)
    v20 = r.rolling(20).std(); v120 = r.rolling(120).std()
    X["vol20"] = v20; X["vol_ratio"] = v20 / v120
    for c in ["value", "turn", "relval", "season"]:
        X[c] = sig[c].reindex(lp.index)
    m = lp.index.month
    X["m_sin"], X["m_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    return X


def load_inputs(ids: list[str] | None = None):
    import analytics as A
    import procurement as PR
    La = A.load_levels(roll_adjusted=True); Lr = A.load_levels(roll_adjusted=False)
    ids = ids or [i for i in PR.BUY_IDS if i in La]
    lp, feats, rel = {}, {}, {}
    for i in ids:
        p = La[i].dropna(); p = p[(p > 0) & (p.index.dayofweek < 5)]
        lp[i] = np.log(p)
        inp = PR.inputs(La, Lr, i, "ewma", "price")
        sig = PR.signals(La[i], 126, inp["rel"], inp["rel_sign"], None, None, "ewm")
        feats[i] = _features(lp[i], sig)
        if inp["rel"] is not None:
            rel[i] = inp["rel"]
    return dict(La=La, Lr=Lr, ids=ids, lp=lp, feats=feats, rel=rel)


# ============================================================================
# LightGBM 글로벌 모델 (매년 재학습, Purged K-fold 하이퍼파라미터 선택)
# ============================================================================
LGB_GRID = [dict(n_estimators=150, learning_rate=0.03, num_leaves=7, min_child_samples=200),
            dict(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=100)]


def _lgb(kind: str, **kw):
    import lightgbm as lgb
    base = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=4, random_state=0)
    return (lgb.LGBMClassifier if kind == "clf" else lgb.LGBMRegressor)(**base, **kw)


def lgb_walkforward(X: pd.DataFrame, y: pd.Series, end_of: pd.Series, dec: pd.MultiIndex, kind: str = "reg", horizon_days: int = H,
                    start=EVAL_START, cv: str = "purged") -> pd.Series:
    """X·y: (품목, 날짜) MultiIndex의 결정일 표본. end_of = 그 목표가 확정되는 날. 매년 1월: 확정된 표본만으로 학습
    (purge: 확정일 < 연초), 하이퍼파라미터는 Purged K-fold(embargo = 목표 기간) 평균 손실로 선택. 반환: dec에 대한 예측."""
    import validation as VD
    from sklearn.metrics import log_loss, mean_squared_error
    out = pd.Series(np.nan, index=dec)
    years = sorted(set(dec.get_level_values(1).year))
    for yr in years:
        if pd.Timestamp(f"{yr}-01-01") < pd.Timestamp(start):
            continue
        cut = pd.Timestamp(f"{yr}-01-01")
        tr = (end_of < cut).values & y.notna().values & X.notna().all(1).values
        te = dec[dec.get_level_values(1).year == yr]
        if tr.sum() < 500 or not len(te):
            continue
        Xt, yt = X[tr], y[tr]
        order = np.argsort(Xt.index.get_level_values(1).values, kind="stable")    # 시간 순 (품목 섞임)
        Xt, yt = Xt.iloc[order], yt.iloc[order]
        best, bl = LGB_GRID[0], np.inf
        folds = []
        if cv == "purged":
            dts = Xt.index.get_level_values(1).values; ends = end_of[tr].iloc[order].values
            bounds = np.linspace(0, len(Xt), 5).astype(int); folds = []
            for a, b in zip(bounds[:-1], bounds[1:]):
                t0, t1 = dts[a], dts[b - 1]
                keep = (ends < t0) | (dts > t1 + np.timedelta64(int(horizon_days * 1.5), "D"))   # purge + embargo (목표 기간)
                folds.append((np.flatnonzero(keep), np.arange(a, b)))
        elif cv == "naive":
            folds = VD.naive_kfold(len(Xt), 4)
        for g in LGB_GRID:
            ls = []
            for a_, b_ in folds:
                if len(a_) < 200 or len(b_) < 50:
                    continue
                m = _lgb(kind, **g).fit(Xt.iloc[a_], yt.iloc[a_])
                if kind == "clf":
                    ls.append(log_loss(yt.iloc[b_], m.predict_proba(Xt.iloc[b_])[:, 1], labels=[0, 1]))
                else:
                    ls.append(mean_squared_error(yt.iloc[b_], m.predict(Xt.iloc[b_])))
            if ls and np.mean(ls) < bl:
                bl, best = np.mean(ls), g
        m = _lgb(kind, **best).fit(Xt, yt)
        Xe = X.reindex(te)
        ok = Xe.notna().all(1).values
        if ok.any():
            out.loc[te[ok]] = m.predict_proba(Xe[ok])[:, 1] if kind == "clf" else m.predict(Xe[ok])
    return out


# ============================================================================
# 실험 본체
# ============================================================================
def _sr_ann(x, per_year):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1) * np.sqrt(per_year)) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan


def run(models: list[str] | None = None, log=print) -> dict:
    """전체 실험. 반환 dict of DataFrame: direction, vol, spread, rank, buy, validation, meta."""
    import analytics as A
    import procurement as PR
    import risk as RK
    import validation as VD
    from sklearn.metrics import roc_auc_score, brier_score_loss
    t0 = time.time()
    models = models or FM_NAMES
    D = load_inputs(); ids = D["ids"]; lp = D["lp"]
    dec = {i: fridays(lp[i].index) for i in ids}                             # 평가·파운데이션 모델 결정일 (2012~)
    dec_m = {i: dec[i][::4] for i in ids}                                    # TiRex: 4주에 1번
    dec_all = {i: fridays(lp[i].index, "2001-01-01") for i in ids}           # 표본 프레임 (LightGBM 학습용 과거 포함)
    # ---------- 파운데이션 모델 예측: 로그 가격 20일 경로
    P = {}
    for m in models:
        dd = dec_m if m == "TiRex" else dec
        P[m] = fm_forecast(m, "logprice", {i: lp[i] for i in ids}, dd, H, progress=log)
        log(f"[{time.time() - t0:5.0f}s] {m} 가격 경로 완료")
    # ---------- 표본 프레임 (품목, 결정일)
    rows = []
    for i in ids:
        s = lp[i]; pos = s.index.get_indexer(dec_all[i])
        fut5 = s.values[pos + 5] - s.values[pos]
        fut20 = s.values[pos + H] - s.values[pos]
        twap = np.array([np.exp(s.values[p + 1:p + H + 1]).mean() for p in pos])          # 향후 20일 균등 매수 단가
        imm = np.exp(s.values[pos + 1])                                                     # 다음 날 즉시 매수 단가
        wait = (twap < np.exp(s.values[pos])).astype(float)
        end = s.index[pos + H]
        for k, d in enumerate(dec_all[i]):
            rows.append((i, d, fut5[k], fut20[k], wait[k], end[k], twap[k], imm[k]))
    F = pd.DataFrame(rows, columns=["id", "date", "r5", "r20", "wait", "end", "twap", "imm"]).set_index(["id", "date"])
    # 기준 예측
    for i in ids:
        s = lp[i]; idx = pd.MultiIndex.from_product([[i], dec_all[i]])
        pos = s.index.get_indexer(dec_all[i])
        mu5 = (s.diff(5)).expanding(252).mean().values[pos]
        F.loc[idx, "pred5 과거 평균"] = mu5; F.loc[idx, "pred20 과거 평균"] = mu5 * 4
        mom = s.shift(21).values[pos] - s.shift(252).values[pos]
        F.loc[idx, "pred5 모멘텀 (12-1개월)"] = mom; F.loc[idx, "pred20 모멘텀 (12-1개월)"] = mom
        rev = -(s.values[pos] - s.shift(21).values[pos])
        F.loc[idx, "pred5 단기 반전 (1개월)"] = rev; F.loc[idx, "pred20 단기 반전 (1개월)"] = rev
        base_w = F.loc[idx, "wait"].expanding().mean().shift(4)               # 확정된 과거 비율
        F.loc[idx, "pwait 과거 평균"] = base_w.values
        for m in models:
            q = P[m][i]; dd = dec_m[i] if m == "TiRex" else dec[i]
            if not len(q):
                continue
            ix = pd.MultiIndex.from_product([[i], dd]); cur = s.reindex(dd).values
            F.loc[ix, f"pred5 {m}"] = q[:, 4, 4] - cur
            F.loc[ix, f"pred20 {m}"] = q[:, H - 1, 4] - cur
            F.loc[ix, f"pwait {m}"] = (np.exp(q).mean(1) < np.exp(cur)[:, None]).mean(1)   # 분위 경로 9개 중 '평균가<오늘' 비율
    # LightGBM 글로벌
    X = pd.concat({i: D["feats"][i].reindex(dec_all[i]) for i in ids}, names=["id", "date"])
    X = X.reindex(F.index)
    item_code = pd.Series(pd.Categorical(F.index.get_level_values(0)).codes, index=F.index)
    X = X.assign(item=item_code.values)
    endF = F["end"]
    F["pred5 LightGBM (글로벌, purged CV)"] = lgb_walkforward(X, F["r5"], endF, F.index, "reg", 5)
    F["pred20 LightGBM (글로벌, purged CV)"] = lgb_walkforward(X, F["r20"], endF, F.index, "reg", H)
    F["pwait LightGBM (글로벌, purged CV)"] = lgb_walkforward(X, F["wait"], endF, F.index, "clf", H)
    log(f"[{time.time() - t0:5.0f}s] LightGBM 완료")
    ev = F[F.index.get_level_values(1) >= pd.Timestamp(EVAL_START)]
    names = BASE_NAMES + models
    # ---------- (1) 방향
    drow, strat = [], {}
    for m in names:
        for i in ids:
            g = ev.loc[i].dropna(subset=[f"pred5 {m}"]) if f"pred5 {m}" in ev else pd.DataFrame()
            if len(g) < 30:
                continue
            ret = np.sign(g[f"pred5 {m}"]) * g["r5"] * 100
            per = 52 / (4 if m == "TiRex" else 1)
            pl = A.placebo_sharpe(g["r5"].values * 100, int(per), 1, 0.0, n_models=len(names), long_share=float((g[f"pred5 {m}"] > 0).mean()),
                                  switch=float((np.diff(np.sign(g[f"pred5 {m}"].values)) != 0).mean()), B=300)
            drow.append(dict(모델=m, 품목=PR.BUY_IDS[i], Sharpe=_sr_ann(ret, per), 적중률=float((np.sign(g[f"pred5 {m}"]) == np.sign(g["r5"])).mean()),
                             롱비중=float((g[f"pred5 {m}"] > 0).mean()), BuyHold=_sr_ann(g["r5"], per), 위약95=pl["single95_gross"], 위약최대95=pl["max95_gross"], n=len(g)))
            if m != "TiRex":
                strat[(m, i)] = ret
    DIR = pd.DataFrame(drow)
    # ---------- (2) 변동성: 향후 20일 실현분산
    vrow = []
    W = {}
    for i in ids:
        r = RK.returns(D["La"][i]); r2 = r ** 2
        wk = r2.groupby(r2.index.to_period("W-FRI")).sum(); wk.index = wk.index.to_timestamp(how="end").normalize()
        W[i] = wk
    Vf = {}
    for m in [x for x in models if x != "TiRex"]:
        decw = {i: pd.DatetimeIndex([W[i].index[W[i].index.get_indexer([d], method="ffill")[0]] for d in dec[i]]) for i in ids}
        Vf[m] = fm_forecast(m, "weeklyRV", W, decw, 4, 260, progress=log)
    log(f"[{time.time() - t0:5.0f}s] 변동성 예측 완료")
    for i in ids:
        r = RK.returns(D["La"][i]); o = RK.vol_models(r, ["EWMA 품목별 λ (확장)", "GARCH(1,1)"])
        S = o["S"]; pos = r.index.get_indexer(dec[i]); ok = pos + H < len(r)
        dd = dec[i][ok]; pos = pos[ok]
        rv = np.array([(r.values[p + 1:p + H + 1] ** 2).sum() for p in pos])
        pred = {}
        for m_ in ["EWMA 품목별 λ (확장)", "GARCH(1,1)"]:
            s_next = S[m_].shift(-1).values[pos]                                  # t 종가 이후 첫날 예측 (t까지 정보)
            pred[m_] = H * s_next
        wk = W[i]; Xh = pd.DataFrame({"w": wk, "m4": wk.rolling(4).mean(), "m12": wk.rolling(12).mean(), "y": wk[::-1].rolling(4).sum()[::-1].shift(-1)})
        har = np.full(len(dd), np.nan)
        for k, d in enumerate(dd):
            j = wk.index.get_indexer([d], method="ffill")[0]
            hist = Xh.iloc[:max(j - 4, 0)].dropna()
            if len(hist) > 100:
                b = np.linalg.lstsq(np.column_stack([np.ones(len(hist)), hist[["w", "m4", "m12"]].values]), hist["y"].values, rcond=None)[0]
                har[k] = max(b @ np.r_[1, Xh[["w", "m4", "m12"]].iloc[j].values], 1e-6) * (H / 20)
        pred["HAR (주간)"] = har
        for m in Vf:
            q = Vf[m][i][ok]
            pred[m] = np.clip(q.mean(2), 1e-8, None).sum(1)                       # 분위 평균 ≈ 평균, 4주 합
        evm = dd >= pd.Timestamp(EVAL_START)
        L = {k: np.log(v[evm]) + rv[evm] / v[evm] for k, v in pred.items()}
        Lg = L["GARCH(1,1)"]
        for k, l in L.items():
            okk = np.isfinite(l) & np.isfinite(Lg)
            d_ = l[okk] - Lg[okk]
            vrow.append(dict(모델=k, 품목=PR.BUY_IDS[i], QLIKE=float(np.nanmean(l)), DM_t_vs_GARCH=float(d_.mean() / RK.nw_se(d_, 4)) if k != "GARCH(1,1)" and len(d_) > 20 else 0.0,
                             GARCH보다_나은_비율=float((d_ < 0).mean()) if k != "GARCH(1,1)" else np.nan, n=int(okk.sum())))
    VOL = pd.DataFrame(vrow)
    # ---------- (3) 스프레드
    srow = []
    rel = {i: s.dropna() for i, s in D["rel"].items()}
    rel = {i: s[s.index.dayofweek < 5] for i, s in rel.items()}
    decs = {i: fridays(s.index) for i, s in rel.items()}
    Sf = {m: fm_forecast(m, "spread", rel, decs, H, progress=log) for m in models if m != "TiRex"}
    log(f"[{time.time() - t0:5.0f}s] 스프레드 예측 완료")
    for i, s in rel.items():
        pos = s.index.get_indexer(decs[i]); dd = decs[i]
        dlt = s.values[pos + H] - s.values[pos]
        pred = {"랜덤워크 (변화 0)": np.zeros(len(pos))}
        dev = (s - s.rolling(252).mean()).values[pos]
        ar = np.full(len(pos), np.nan)
        for k in range(len(pos)):
            past = np.flatnonzero(pos + H <= pos[k])                              # 확정된 과거 결정일
            if len(past) > 52 and np.isfinite(dev[past]).sum() > 52:
                okp = np.isfinite(dev[past])
                kappa = np.polyfit(dev[past][okp], dlt[past][okp], 1)
                ar[k] = np.polyval(kappa, dev[k])
        pred["평균회귀 AR (1년 평균 대비)"] = ar
        for m in Sf:
            pred[m] = Sf[m][i][:, H - 1, 4] - s.values[pos]
        evm = dd >= pd.Timestamp(EVAL_START)
        mae_rw = np.mean(np.abs(dlt[evm]))
        for k, p_ in pred.items():
            okk = evm & np.isfinite(p_)
            srow.append(dict(모델=k, 품목=PR.BUY_IDS[i], 관계=PR.REL_DEF.get(i, ("",))[0], MAE_비율_vs_RW=float(np.mean(np.abs(dlt[okk] - p_[okk])) / mae_rw),
                             방향적중=float((np.sign(p_[okk]) == np.sign(dlt[okk])).mean()) if k != "랜덤워크 (변화 0)" else np.nan,
                             IC=float(stats.spearmanr(p_[okk], dlt[okk])[0]) if k != "랜덤워크 (변화 0)" else np.nan, n=int(okk.sum())))
    SPR = pd.DataFrame(srow)
    # ---------- (4) 순위
    rrow = []
    for m in names:
        col = f"pred20 {m}"
        if col not in ev:
            continue
        g = ev[[col, "r20"]].dropna().reset_index()
        g = g[g.groupby("date")["id"].transform("count") >= 6]
        dts = sorted(g["date"].unique())[:: (1 if m == "TiRex" else 4)]          # 4주 비중첩
        ics, ls = [], []
        for d in dts:
            x = g[g.date == d]
            ics.append(stats.spearmanr(x[col], x["r20"])[0])
            o = x.sort_values(col); ls.append((o["r20"].iloc[-3:].mean() - o["r20"].iloc[:3].mean()) * 100)
        ics = np.array(ics)
        rrow.append(dict(모델=m, 평균IC=float(np.nanmean(ics)), IC_t=float(np.nanmean(ics) / (np.nanstd(ics, ddof=1) / np.sqrt(np.isfinite(ics).sum()))),
                         IC양수비율=float((ics > 0).mean()), 상위3하위3_Sharpe=_sr_ann(ls, 12), n=len(dts)))
    RANK = pd.DataFrame(rrow)
    # ---------- (5) 구매: 기다리면 더 싸게 살까
    brow = []
    for m in names:
        col = f"pwait {m}"
        if col not in ev:
            continue
        for i in ids + ["(전체)"]:
            g = (ev if i == "(전체)" else ev.loc[[i]])[[col, "wait", "twap", "imm"]].dropna()
            if len(g) < 30 or g["wait"].nunique() < 2:
                continue
            p_ = g[col].clip(0, 1)
            # 결정 가치: P(기다리면 쌈) > 0.5 → 향후 20일 균등 매수, 아니면 다음 날 즉시 매수. 기준 = 항상 균등 매수 (절감 bp, + = 균등보다 쌈)
            cost = np.where(p_ > 0.5, g["twap"], g["imm"])
            sav = (g["twap"] - cost) / g["twap"] * 1e4
            brow.append(dict(모델=m, 품목=PR.BUY_IDS.get(i, i), AUC=float(roc_auc_score(g["wait"], p_)) if p_.nunique() > 1 else 0.5,
                             Brier=float(brier_score_loss(g["wait"], p_)), 기준비율=float(g["wait"].mean()),
                             기다림선택비율=float((p_ > 0.5).mean()), 결정절감bp=float(sav.mean()),
                             항상즉시매수bp=float(((g["twap"] - g["imm"]) / g["twap"] * 1e4).mean()), n=len(g)))
    BUY = pd.DataFrame(brow)
    # ---------- (6) 과적합 판정: 방향 전략 집합
    M = pd.DataFrame(strat).dropna()
    vrows = []
    if len(M.columns) >= 2:
        srs = M.apply(lambda x: VD.sharpe(x.values)).values
        best = M.columns[int(np.nanargmax(srs))]
        ds = VD.deflated_sharpe(M[best].values, srs)
        pb = VD.pbo_cscv(M.values, 16)
        vrows.append(dict(대상="방향 전략 (모델×품목)", 최고=f"{best[0]} · {PR.BUY_IDS[best[1]]}", 최고_Sharpe_연=ds["SR (기간당)"] * np.sqrt(52),
                          SR0_연=ds["기대 최대 SR₀ (기간당)"] * np.sqrt(52), 시도수=ds["시도 수"], PSR=ds["PSR (SR>0)"], DSR=ds["DSR (SR>SR₀)"],
                          PBO=pb["PBO"], IS최고의_OOS_SR연=pb.get("IS 최고의 OOS SR 평균 (기간당)", np.nan) * np.sqrt(52)))
        Mm = pd.DataFrame({m: M[[c for c in M.columns if c[0] == m]].mean(1) for m in dict.fromkeys(c[0] for c in M.columns)})
        srs_m = Mm.apply(lambda x: VD.sharpe(x.values)).values; bm = Mm.columns[int(np.nanargmax(srs_m))]
        ds = VD.deflated_sharpe(Mm[bm].values, srs_m); pb = VD.pbo_cscv(Mm.values, 16)
        vrows.append(dict(대상="방향 전략 (모델별 9품목 동일가중)", 최고=bm, 최고_Sharpe_연=ds["SR (기간당)"] * np.sqrt(52), SR0_연=ds["기대 최대 SR₀ (기간당)"] * np.sqrt(52),
                          시도수=ds["시도 수"], PSR=ds["PSR (SR>0)"], DSR=ds["DSR (SR>SR₀)"], PBO=pb["PBO"],
                          IS최고의_OOS_SR연=pb.get("IS 최고의 OOS SR 평균 (기간당)", np.nan) * np.sqrt(52)))
    # 누수 시연: 같은 LightGBM 구매 분류기를 '무작위 K-fold'로 평가하면 AUC가 얼마나 부풀려지나 (학습 표본 2001~2011)
    tr = (F["end"] < pd.Timestamp(EVAL_START)).values & X.notna().all(1).values
    Xt, yt = X[tr], F["wait"][tr]
    order = np.argsort(Xt.index.get_level_values(1).values, kind="stable"); Xt, yt = Xt.iloc[order], yt.iloc[order]
    for nm, folds in [("무작위 K-fold (누수)", VD.naive_kfold(len(Xt), 5)), ("Purged K-fold + embargo", None)]:
        if folds is None:
            dts = Xt.index.get_level_values(1).values; ends = F["end"][tr].iloc[order].values
            b_ = np.linspace(0, len(Xt), 6).astype(int); folds = []
            for a, b in zip(b_[:-1], b_[1:]):
                keep = (ends < dts[a]) | (dts > dts[b - 1] + np.timedelta64(30, "D"))
                folds.append((np.flatnonzero(keep), np.arange(a, b)))
        aucs = []
        for a_, b_ in folds:
            mdl = _lgb("clf", **LGB_GRID[1]).fit(Xt.iloc[a_], yt.iloc[a_])
            aucs.append(roc_auc_score(yt.iloc[b_], mdl.predict_proba(Xt.iloc[b_])[:, 1]))
        vrows.append(dict(대상=f"CV 방식 비교 — 구매 분류기 AUC: {nm}", 최고="LightGBM", 최고_Sharpe_연=np.nan, PSR=float(np.mean(aucs))))
    oos_auc = BUY[(BUY.모델 == "LightGBM (글로벌, purged CV)") & (BUY.품목 == "(전체)")]["AUC"]
    vrows.append(dict(대상="CV 방식 비교 — 같은 분류기의 실제 walk-forward AUC (2012~)", 최고="LightGBM", PSR=float(oos_auc.iloc[0]) if len(oos_auc) else np.nan))
    VAL = pd.DataFrame(vrows)
    log(f"[{time.time() - t0:5.0f}s] 판정 완료")
    META = meta_labeling(D, log=log)
    log(f"[{time.time() - t0:5.0f}s] 메타 라벨링 완료")
    return dict(direction=DIR, vol=VOL, spread=SPR, rank=RANK, buy=BUY, validation=VAL, meta=META)


# ============================================================================
# 메타 라벨링 (★ 구매 타이밍 '추천' 점수 위에 2차 분류기)
# ============================================================================
def meta_labeling(D: dict | None = None, weeks: int = 13, start="2008-01-01", log=print) -> pd.DataFrame:
    import procurement as PR
    D = D or load_inputs()
    La, Lr = D["La"], D["Lr"]
    P_ = PR.PRESETS["추천 (고도화: 상대가치+계절+MIDAS)"]
    rows, Xs, ys, ends, keys, scs = [], [], [], [], [], {}
    sigs = {}
    for i in D["ids"]:
        inp = PR.inputs(La, Lr, i, P_["risk"], "price")
        sig = PR.signals(La[i], P_["lookback"], inp["rel"], inp["rel_sign"], inp["vol_fc"], None, P_["season"])
        sc = PR.score(sig, P_["w"]); scs[i] = (sc, inp); sigs[i] = sig
        pa = Lr[i].dropna(); pa = pa[pa > 0]
        days = pa.index.intersection(sig.index)
        fri = pd.DatetimeIndex(pd.Series(days, index=days).groupby(days.to_period("W-FRI")).max().values)
        pos = days.get_indexer(fri)
        ex = np.array([pa[days[p + 1]] if p + 1 < len(days) else np.nan for p in pos])
        for k, d in enumerate(fri):
            if k + weeks >= len(fri):
                break
            fut = np.nanmean(ex[k + 1:k + weeks]); s_ = float(sc.get(d, np.nan))
            if not np.isfinite(s_) or not np.isfinite(ex[k]) or abs(s_ - 0.5) < 1e-9:
                continue
            correct = float((s_ > 0.5 and ex[k] < fut) or (s_ < 0.5 and ex[k] > fut))
            f = sig.loc[d, ["value", "turn", "relval", "season", "vol_pct", "risk"]].to_dict()
            lpi = D["lp"][i]; j = lpi.index.get_indexer([d], method="ffill")[0]
            f.update(score=s_, conv=abs(s_ - 0.5), r20=float(lpi.iloc[j] - lpi.iloc[max(j - 20, 0)]), r60=float(lpi.iloc[j] - lpi.iloc[max(j - 60, 0)]),
                     m_sin=np.sin(2 * np.pi * d.month / 12), m_cos=np.cos(2 * np.pi * d.month / 12), item=D["ids"].index(i))
            Xs.append(f); ys.append(correct); ends.append(fri[k + weeks]); keys.append((i, d))
    X = pd.DataFrame(Xs, index=pd.MultiIndex.from_tuples(keys, names=["id", "date"]))
    y = pd.Series(ys, index=X.index); end = pd.Series(ends, index=X.index)
    p = lgb_walkforward(X, y, end, X.index, "clf", weeks * 5, start="2010-01-01")
    log(f"메타 분류기: 표본 {len(X):,}, 1차 신호 적중률 {y.mean():.3f}")
    from sklearn.metrics import roc_auc_score
    okp = p.notna()
    for i in D["ids"]:
        sc, inp = scs[i]
        pi = p.xs(i, level=0).dropna() if i in p.index.get_level_values(0) else pd.Series(dtype=float)
        sc_meta = sc.copy()
        adj = (0.5 + (sc.reindex(pi.index) - 0.5) * 2 * pi).clip(0, 1)
        sc_meta.loc[adj.index] = adj.values
        sc_meta.loc[sc_meta.index < (pi.index.min() if len(pi) else sc.index.max())] = np.nan   # 메타 예측 전 구간은 비교 제외
        st_i = max(pd.Timestamp("2010-01-01"), pi.index.min()) if len(pi) else pd.Timestamp("2010-01-01")   # 두 방식 같은 기간
        for nm, ov in [("추천 (1차 신호)", None), ("추천 + 메타 라벨링", sc_meta)]:
            B = PR.backtest(Lr[i], La[i], weeks, 4, start=st_i, w=P_["w"], k=P_["k"], lookback=P_["lookback"], inp=inp,
                            season=P_["season"], score_override=ov)
            if ov is None:
                Bb = B
            x = B["절감_규칙"]
            rows.append(dict(품목=PR.BUY_IDS[i], 방식=nm, 평균절감bp=x.mean(), 전반_2010_17=x[x.index < "2018-01-01"].mean(),
                             후반_2018_=x[x.index >= "2018-01-01"].mean(), 균등보다싸게=(x > 0).mean(), 하위5bp=x.quantile(0.05), 기간수=len(x)))
        yi = y.xs(i, level=0).reindex(pi.index)
        rows[-1]["메타_AUC"] = float(roc_auc_score(yi, pi)) if len(pi) > 30 and yi.nunique() > 1 else np.nan
    return pd.DataFrame(rows)
