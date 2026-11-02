"""온톨로지 노드로 '방향'이 아니라 '변동성(앞으로 4주 출렁임 크기)'을 예측하는 시험 — 기존 파일은 고치지 않음.

목표: 다음 4주 실현 분산 RV = Σ r²(t+1..t+4) (주간 로그 수익률 %).
모형 (log RV를 릿지 회귀, 13주마다 재학습, 정답 확정 구간만 학습, 설정 사전 고정):
  V0 EWMA   : 4 × EWMA(λ 0.94) 주간 분산 (지금 쓰는 위험 크기 예측)
  V1 HAR    : 지난 1 · 4 · 13주 실현 분산(로그)
  V2 +일정  : V1 + 다음 4주 안 WASDE 수 · 분기 보고서(3 · 6 · 9월 말) 수 · 생육기(6–8월) — 모두 미리 아는 일정
  V3 +온톨로지: V2 + 그 품목 온톨로지 부모(3판: 기상 · 위성 · 작황 · 재고율 · 수출 판매 · 운임 · 환율 · 원유 · 해외 수정 · 연결 품목)의 최근 4주 충격 크기 Σ|x|
  V4 +전부  : V2 + 노드 33개 전부의 최근 4주 충격 크기 (도메인으로 거르기 대조)
채점: QLIKE = RV/F − log(RV/F) − 1 (변동성 예측 표준, 낮을수록 좋음). 시험 2017–, 하위 기간 2017–21 · 2022–, Newey-West t.
채택 기준: 직전 모형 대비 두 하위 기간 모두 개선 · t > 2.
실행: .venv\\Scripts\\python.exe vol_ont.py → data/vol_ont_compare.csv
"""
import sys
import warnings

import numpy as np
import pandas as pd

import causal_grain as CG
import experiments as EX
import foreign_test as FT
import hedge_core as H
import range_bt as RB

warnings.filterwarnings("ignore")
HZ, LAM, RIDGE, REFIT, TEST = 4, 0.94, 10.0, 13, pd.Timestamp("2017-01-01")


def calendar(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """금요일 t 기준 다음 4주(t+1일 ~ t+28일) 안의 보고서 수 — USDA가 미리 공지하는 일정."""
    wd, qd = H.wasde_dates(), H.quarterly_dates(idx[0].year, idx[-1].year + 1)
    cnt = lambda ds, t: int(((ds > t) & (ds <= t + pd.Timedelta(days=28))).sum())   # noqa: E731
    return pd.DataFrame({"WASDE 수": [cnt(wd, t) for t in idx], "분기 보고서 수": [cnt(qd, t) for t in idx],
                         "미국 생육기": idx.month.isin([6, 7, 8]).astype(float)}, index=idx)


def features(X: pd.DataFrame, p: str, cal: pd.DataFrame) -> dict:
    r2 = X[p] ** 2
    har = pd.DataFrame({"rv1": np.log(r2 + 0.01), "rv4": np.log(r2.rolling(4).sum() + 0.01), "rv13": np.log(r2.rolling(13).sum() + 0.01)})
    mag = lambda cols: X[cols].abs().rolling(4).sum().add_prefix("|").add_suffix("|")   # noqa: E731
    par = [q for q in CG.GRAPHS[3][1][p] if q != p]
    allx = [c for c in X.columns if c != p]
    return {"V1 HAR": har, "V2 +일정": har.join(cal), "V3 +온톨로지": har.join(cal).join(mag(par)), "V4 +전부": har.join(cal).join(mag(allx))}


def _ridge(Z, y, a=RIDGE):
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    S = (Z - mu) / sd
    b = np.linalg.solve(S.T @ S + a * np.eye(S.shape[1]), S.T @ (y - y.mean()))
    return lambda z: y.mean() + ((z - mu) / sd) @ b


def run_item(X: pd.DataFrame, p: str, cal: pd.DataFrame) -> pd.DataFrame:
    r = X[p]
    rv = (r ** 2).shift(-1).rolling(HZ).sum().shift(-(HZ - 1))                         # t+1..t+4
    F = {"V0 EWMA": HZ * (r ** 2).ewm(alpha=1 - LAM, adjust=False).mean()}
    feats = features(X, p, cal)
    idx = X.index
    pred = {k: pd.Series(np.nan, index=idx) for k in feats}
    last = None
    for k, t in enumerate(idx):
        if t < TEST or not np.isfinite(rv.iloc[k]):
            continue
        if last is None or k - last >= REFIT:
            tr = np.arange(13, k - HZ + 1)                                                 # t−4까지: 정답 확정
            fits = {}
            for nm, Z in feats.items():
                Zt, y = Z.iloc[tr].values, np.log(rv.iloc[tr].values + 0.01)
                ok = np.isfinite(Zt).all(1) & np.isfinite(y)
                g = _ridge(Zt[ok], y[ok])
                s2 = np.var(y[ok] - g(Zt[ok]))
                fits[nm] = (g, s2)
            last = k
        for nm, Z in feats.items():
            g, s2 = fits[nm]
            z = Z.iloc[k].values
            pred[nm].iloc[k] = np.exp(g(np.nan_to_num(z)) + s2 / 2) - 0.01               # log 정규 보정
    out = pd.DataFrame({"RV": rv, **F, **pred}).loc[lambda d: d.index >= TEST].dropna()
    return out


def qlike(rv, f):
    f = np.maximum(f, 1e-6); x = np.maximum(rv, 1e-6) / f
    return x - np.log(x) - 1


def run():
    sys.stdout.reconfigure(errors="replace")
    pd.set_option("display.width", 220)
    La, _, _, _ = EX._data()
    X2 = CG.nodes(La)
    X = pd.concat([X2, FT.foreign(X2.index)], axis=1)
    cal = calendar(X.index)
    order = ["V0 EWMA", "V1 HAR", "V2 +일정", "V3 +온톨로지", "V4 +전부"]
    rows = []
    for p in CG.PRICES:
        O = run_item(X, p, cal)
        L = pd.DataFrame({k: qlike(O.RV.values, O[k].values) for k in order}, index=O.index)
        half = L.index < pd.Timestamp("2022-01-01")
        for k, base in [("V1 HAR", "V0 EWMA"), ("V2 +일정", "V1 HAR"), ("V3 +온톨로지", "V2 +일정"), ("V4 +전부", "V2 +일정")]:
            d = (L[base] - L[k]).values                                                    # + = 개선
            rows.append(dict(가격=p, 모형=k, 비교=base, 시험주=len(L), QLIKE=round(L[k].mean(), 4), 기준QLIKE=round(L[base].mean(), 4),
                             개선pct=round(100 * d.mean() / L[base].mean(), 1), t_NW=round(RB.nw_t(d, HZ - 1), 2),
                             개선_2017_21=round(100 * d[half].mean() / L[base][half].mean(), 1), 개선_2022_=round(100 * d[~half].mean() / L[base][~half].mean(), 1)))
        print(f"  {p} 완료", flush=True)
    R = pd.DataFrame(rows)
    R["채택"] = (R.개선_2017_21 > 0) & (R.개선_2022_ > 0) & (R.t_NW > 2)
    print(R.to_string(index=False))
    R.to_csv("data/vol_ont_compare.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    run()
