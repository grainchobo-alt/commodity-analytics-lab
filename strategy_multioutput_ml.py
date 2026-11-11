"""다출력(종속변수 = 설명변수와 같은 19개 품목 수익) vs 단일 출력 ML 비교 (Y).

질문: 수익률 예측에서 '품목 하나(종속변수)를 전체 설명변수로 설명'하는 대신, 종속변수를 설명변수와 같은 19개 품목 전체로 잡고(다출력) 한 모델로 동시에 예측하면
예측 정확도(R² · RMSE)가 좋아지나? 머신러닝 · AI 모델만 대상 (통계 · 계량 모형은 제외, 비교 기준으로 Ridge만 포함).
설계 (기존 ④ 백테스트 그리드와 같은 틀): 19개 품목 주간 · 목표 = 4주 뒤 수익률(%) · 설명변수 = 19개 품목 각각의 직전 1 · 4 · 13 · 52주 수익률 (76개, 시점 t까지 정보만) ·
  매년 1월 재학습(그해 이전 자료, 4주 purge) · 2007–2026-10-26 · 평가 = 학습 평균(역사 평균) 예측 대비 표본 밖 R² · RMSE · 방향 적중률, 학습 2007–2016 / 시험 2017–.
모델:
  단일 출력(품목마다 따로): LightGBM · ExtraTrees · MLP · Ridge
  다출력(19개 동시): ExtraTrees(공통 분할) · MLP(은닉층 공유, 출력 19개) · Ridge(= 단일과 동일해야 함, 검증용)
  통합(품목 공통 함수 + 품목 더미): ExtraTrees pooled — 패널의 '공통 기울기' 가정에 해당
실행: .venv\\Scripts\\python.exe strategy_multioutput_ml.py → data/strategy_multioutput_ml_compare.csv · strategy_multioutput_ml_items.csv · strategy_multioutput_ml_pred.parquet
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import alt_ts as AT  # noqa: E402
import analytics as A  # noqa: E402
import predict_lab as PL  # noqa: E402

D = A.DATA
H = 4
START_FIT = pd.Timestamp("2001-01-01")
GRID_START = pd.Timestamp("2007-01-01")
SPLIT = pd.Timestamp("2017-01-01")
SEEDS = (0, 1, 2)
MODELS = ["Ridge (단일)", "Ridge (다출력)", "LightGBM (단일)", "ExtraTrees (단일)", "ExtraTrees (다출력)", "MLP (단일)", "MLP (다출력)", "MLP 강규제 (단일)", "MLP 강규제 (다출력)", "ExtraTrees (통합 · 품목 더미)"]


def make_data():
    L = PL.frozen_levels(True)
    P = AT.weekly_px(L, "2001-01-01"); lp = np.log(P)
    feats = {f"{c}|{k}w": 100 * (lp[c] - lp[c].shift(k)) for c in P.columns for k in (1, 4, 13, 52)}
    X = pd.DataFrame(feats)
    Y = 100 * (lp.shift(-H) - lp)
    return P, X, Y


def _ridge(Xtr, Ytr, Xte):
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import TimeSeriesSplit
    return RidgeCV(alphas=np.logspace(0, 5, 21), cv=TimeSeriesSplit(5, gap=H)).fit(Xtr, Ytr).predict(Xte)


def _mlp(Xtr, Ytr, Xte, seed, alpha=3.0, hid=32):
    from sklearn.neural_network import MLPRegressor
    m = MLPRegressor(hidden_layer_sizes=(hid,), activation="relu", alpha=alpha, learning_rate_init=0.005, max_iter=150, random_state=seed, early_stopping=False)
    return m.fit(Xtr, Ytr).predict(Xte)


def _year_task(yr, X, Y, idx):
    """한 해: 학습(그 전 자료) → 그해 주간 예측. 반환 {모델: DataFrame(예측)}."""
    from sklearn.ensemble import ExtraTreesRegressor
    import lightgbm as lgb
    cut = np.flatnonzero(idx < pd.Timestamp(f"{yr}-01-01"))
    if not len(cut):
        return yr, {}
    okX = X.notna().all(1).values
    tr = np.flatnonzero(okX[:cut[-1] - H + 1] & (idx[:cut[-1] - H + 1] >= START_FIT + pd.Timedelta(weeks=52)))
    tr = tr[np.isfinite(Y.values[tr]).all(1)]
    te = np.flatnonzero((idx.year == yr) & okX)
    if len(tr) < 156 or not len(te):
        return yr, {}
    Xv = X.values; Yv = Y.values; items = list(Y.columns); n_it = len(items)
    mu, sd = Xv[tr].mean(0), Xv[tr].std(0); sd[sd == 0] = 1
    Xtr, Xte = (Xv[tr] - mu) / sd, (Xv[te] - mu) / sd
    ym, ys = Yv[tr].mean(0), Yv[tr].std(0); ys[ys == 0] = 1
    Ytr = (Yv[tr] - ym) / ys                                                        # 품목별 표준화 목표(출력 간 손실 가중 균등)
    out = {m: np.full((len(te), n_it), np.nan) for m in MODELS}
    # Ridge: 단일(품목마다 CV로 α 선택) vs 다출력(α 하나, 같은 X → 사실상 같은 해)
    out["Ridge (다출력)"] = _ridge(Xtr, Ytr, Xte) * ys + ym
    for j in range(n_it):
        out["Ridge (단일)"][:, j] = _ridge(Xtr, Ytr[:, j], Xte) * ys[j] + ym[j]
    # LightGBM · ExtraTrees 단일 / MLP 단일
    for j in range(n_it):
        g = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, n_jobs=1, verbose=-1)
        out["LightGBM (단일)"][:, j] = g.fit(Xtr, Ytr[:, j]).predict(Xte) * ys[j] + ym[j]
        e = ExtraTreesRegressor(n_estimators=150, min_samples_leaf=40, max_features=0.3, n_jobs=1, random_state=0)
        out["ExtraTrees (단일)"][:, j] = e.fit(Xtr, Ytr[:, j]).predict(Xte) * ys[j] + ym[j]
        out["MLP (단일)"][:, j] = np.mean([_mlp(Xtr, Ytr[:, j], Xte, s) for s in SEEDS], axis=0) * ys[j] + ym[j]
        out["MLP 강규제 (단일)"][:, j] = np.mean([_mlp(Xtr, Ytr[:, j], Xte, s, 100.0, 16) for s in SEEDS], axis=0) * ys[j] + ym[j]
    # 다출력
    e = ExtraTreesRegressor(n_estimators=150, min_samples_leaf=40, max_features=0.3, n_jobs=1, random_state=0)
    out["ExtraTrees (다출력)"] = e.fit(Xtr, Ytr).predict(Xte) * ys + ym
    out["MLP (다출력)"] = np.mean([_mlp(Xtr, Ytr, Xte, s) for s in SEEDS], axis=0) * ys + ym
    out["MLP 강규제 (다출력)"] = np.mean([_mlp(Xtr, Ytr, Xte, s, 100.0, 16) for s in SEEDS], axis=0) * ys + ym
    # 통합: 품목 더미 + 모든 품목의 설명변수 → 한 개의 목표 열로 쌓아서 공통 함수 학습
    dm = np.eye(n_it)
    Xp = np.vstack([np.c_[Xtr, np.tile(dm[j], (len(tr), 1))] for j in range(n_it)]); yp = np.concatenate([Ytr[:, j] for j in range(n_it)])
    e = ExtraTreesRegressor(n_estimators=150, min_samples_leaf=80, max_features=0.3, n_jobs=1, random_state=0).fit(Xp, yp)
    for j in range(n_it):
        out["ExtraTrees (통합 · 품목 더미)"][:, j] = e.predict(np.c_[Xte, np.tile(dm[j], (len(te), 1))]) * ys[j] + ym[j]
    bench = np.tile(Yv[tr].mean(0), (len(te), 1))
    res = {m: pd.DataFrame(v, index=idx[te], columns=items) for m, v in out.items()}
    res["기준 (학습 평균)"] = pd.DataFrame(bench, index=idx[te], columns=items)
    return yr, res


def hac_t(d: np.ndarray, lag: int = H) -> float:
    d = d[np.isfinite(d)]; n = len(d)
    if n < 20:
        return np.nan
    dm = d - d.mean(); g0 = (dm @ dm) / n; s = g0
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (dm[k:] @ dm[:-k]) / n
    return float(d.mean() / np.sqrt(max(s, 1e-30) / n))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from joblib import Parallel, delayed
    t0 = time.time(); P, X, Y = make_data(); idx = P.index
    years = sorted(set(idx[idx >= GRID_START].year))
    print(f"품목 {Y.shape[1]} · 설명변수 {X.shape[1]} · 재학습 {len(years)}회 (매년 1월)", flush=True)
    res = Parallel(n_jobs=6, verbose=5)(delayed(_year_task)(y, X, Y, idx) for y in years)
    preds = {}
    for yr, r in res:
        for m, df in r.items():
            preds.setdefault(m, []).append(df)
    preds = {m: pd.concat(v).sort_index() for m, v in preds.items()}
    y = Y.reindex(preds["기준 (학습 평균)"].index)
    rows, items_rows = [], []
    base = preds["기준 (학습 평균)"]
    for per, mask in (("전체 2007–", y.index >= GRID_START), ("학습 2007–2016", y.index < SPLIT), ("시험 2017–", y.index >= SPLIT)):
        for m in ["기준 (학습 평균)"] + MODELS:
            pr = preds[m]; ok = mask & y.notna().all(1).values
            e = (y[ok] - pr[ok]); e0 = (y[ok] - base[ok])
            sse, sse0 = (e ** 2).sum(), (e0 ** 2).sum()
            r2 = 1 - sse / sse0                                                     # 품목별 표본 밖 R² (역사 평균 대비)
            rmse = np.sqrt((e ** 2).mean())
            hit = ((np.sign(pr[ok] - base[ok]) == np.sign(y[ok] - base[ok]))).mean()
            rows.append({"구간": per, "모델": m, "평균 R² (표본 밖, %)": 100 * float(r2.mean()), "R²>0 품목 수": int((r2 > 0).sum()), "평균 RMSE (%)": float(rmse.mean()),
                         "방향 적중률 (%)": 100 * float(hit.mean()), "RMSE 비 (기준=1)": float((rmse / np.sqrt((e0 ** 2).mean())).mean())})
            if per == "전체 2007–":
                for it in y.columns:
                    items_rows.append({"모델": m, "품목": it, "R² (%)": 100 * float(r2[it]), "RMSE (%)": float(rmse[it])})
    T = pd.DataFrame(rows)
    # 다출력 vs 단일: 손실 차이 DM t (품목별 HAC), 같은 모델 계열
    dm_rows = []
    for a, b in (("ExtraTrees (단일)", "ExtraTrees (다출력)"), ("MLP (단일)", "MLP (다출력)"), ("MLP 강규제 (단일)", "MLP 강규제 (다출력)"), ("Ridge (단일)", "Ridge (다출력)"), ("ExtraTrees (단일)", "ExtraTrees (통합 · 품목 더미)"), ("LightGBM (단일)", "ExtraTrees (다출력)")):
        for per, mask in (("전체 2007–", y.index >= GRID_START), ("시험 2017–", y.index >= SPLIT)):
            ok = mask & y.notna().all(1).values; ts, dmean = [], []
            for it in y.columns:
                d = ((y[ok][it] - preds[a][ok][it]) ** 2 - (y[ok][it] - preds[b][ok][it]) ** 2).values
                ts.append(hac_t(d)); dmean.append(np.nanmean(d))
            ts = np.array(ts)
            dm_rows.append({"비교 (단일 → 대안)": f"{a} → {b}", "구간": per, "대안이 더 좋은 품목 수": int((np.array(dmean) > 0).sum()), "DM t 평균": float(np.nanmean(ts)), "유의(|t|>2) 품목 수": int((np.abs(ts) > 2).sum()), "품목 평균 MSE 변화 (%)": float(-100 * np.mean([np.nanmean(((y[ok][it] - preds[b][ok][it]) ** 2)) / np.nanmean(((y[ok][it] - preds[a][ok][it]) ** 2)) - 1 for it in y.columns]))})
    DM = pd.DataFrame(dm_rows)
    T.to_csv(D / "strategy_multioutput_ml_compare.csv", index=False, encoding="utf-8-sig")
    DM.to_csv(D / "strategy_multioutput_ml_dm.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(items_rows).to_csv(D / "strategy_multioutput_ml_items.csv", index=False, encoding="utf-8-sig")
    pd.concat({m: v for m, v in preds.items()}, axis=1).to_parquet(D / "strategy_multioutput_ml_pred.parquet")
    pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 40)
    print(T.round(2).to_string()); print(DM.round(2).to_string()); print(f"완료 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
