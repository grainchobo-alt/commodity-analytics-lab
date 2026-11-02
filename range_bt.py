"""가격 '구간 확률' 예측 백테스트 (ForecastBench 방식) — 점 예측이 아니라 확률을 채점. 기존 파일은 고치지 않음.

질문: h주 뒤(4 · 13주) 가격 변화가 5개 구간 중 어디에 들까 / 지금보다 높을까 / 80% 범위 안에 들까.
채점: RPS(순위 확률 점수, 구간 확률의 누적 오차 — 낮을수록 좋음) · Brier(오를 확률) · 80% 범위 적중률(목표 80%).
모형 (모두 같은 '표준화 잔차의 과거 분포'를 써서 확률을 만듦, 다른 것은 평균 μ와 크기 σ):
  M0 과거 분포  : μ = 학습 평균, σ = 학습 표준편차 (아무 정보 없음)
  M1 변동성     : μ = 학습 평균, σ = EWMA(λ 0.94) 주간 변동성 × √h  — 대시보드의 위험 크기 예측과 같은 생각
  M2 온톨로지   : M1 + μ = 릿지 회귀(그 품목의 온톨로지 부모 + 자기 자신의 최근 4주 합)
  M3 전부 넣기  : M1 + μ = 릿지 회귀(온톨로지 노드 24개 전부의 최근 4주 합) — '도메인으로 거르기'가 필요한지 대조
학습 = 2008 ~ 시점 t−h (정답이 이미 확정된 구간만), 13주마다 다시 학습. 시험 = 2017–. 릿지 강도·구간·λ는 돌리기 전 고정.
실행: .venv\\Scripts\\python.exe range_bt.py → data/range_bt_compare.csv
"""
import sys
import warnings

import numpy as np
import pandas as pd

import analytics as A
import causal_grain as CG
import ls_strategies as LS

warnings.filterwarnings("ignore")
TEST, LAM, RIDGE, REFIT = pd.Timestamp("2017-01-01"), 0.94, 50.0, 13
EDGES = {4: [-8.0, -3.0, 3.0, 8.0], 13: [-15.0, -5.0, 5.0, 15.0]}                         # 로그 수익률 % 구간 경계 (5구간)
MODELS = ["M0 과거 분포", "M1 변동성", "M2 온톨로지", "M3 전부 넣기"]


def _ridge(X, y, a=RIDGE):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    b = np.linalg.solve(Z.T @ Z + a * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return lambda x: y.mean() + ((x - mu) / sd) @ b


def _probs(z_hist, edges_std):
    """표준화 잔차의 과거 분포로 구간 누적확률."""
    zs = np.sort(z_hist)
    return np.searchsorted(zs, edges_std, side="right") / len(zs)


def run_one(P: pd.Series, X: pd.DataFrame, feats: dict, h: int) -> pd.DataFrame:
    r1 = 100 * np.log(P).diff()
    fwd = 100 * np.log(P.shift(-h) / P)                                                   # 정답: t → t+h
    v = r1.pow(2).ewm(alpha=1 - LAM, adjust=False).mean()
    sig = np.sqrt(h * v)
    F = {k: X[c].rolling(4).sum() for k, c in feats.items()}
    idx = [t for t in X.index if t >= TEST and pd.notna(fwd.get(t))]
    rows, fit, last = [], {}, None
    for t in idx:
        tr_end = P.index[P.index.get_loc(t) - h]                                          # 정답이 확정된 마지막 시점
        if last is None or (t - last).days >= 7 * REFIT:
            tr = [s for s in X.index if s <= tr_end and pd.notna(fwd.get(s)) and pd.notna(sig.get(s))][52:]
            y = fwd.loc[tr].values
            fit = {"m": y.mean(), "s": y.std(), "z0": (y - y.mean()) / y.std(), "z1": (y - y.mean()) / sig.loc[tr].values}
            for k, f in F.items():
                Xf = f.loc[tr].values
                ok = ~np.isnan(Xf).any(1)
                g = _ridge(Xf[ok], (y / sig.loc[tr].values)[ok])                            # σ로 나눈 수익률을 설명 (변동성 큰 시기 과대 반영 방지)
                fit[k] = g
                fit["z" + k] = ((y / sig.loc[tr].values) - g(np.nan_to_num(Xf)))
            last = t
        e = np.array(EDGES[h])
        mu_s = {"M0 과거 분포": (fit["m"], fit["s"], fit["z0"]), "M1 변동성": (fit["m"], sig[t], fit["z1"])}
        for k in feats:
            m_std = float(fit[k](np.nan_to_num(F[k].loc[t].values)))
            mu_s[k] = (m_std * sig[t], sig[t], fit["z" + k])
        y = fwd[t]
        o = np.searchsorted(e, y, side="right")                                          # 실제 구간 0..4
        for nm, (mu, s, z) in mu_s.items():
            cdf = _probs(z, (e - mu) / s)
            obs = (np.arange(4) >= o).astype(float)                                        # 실제 누적 (구간 경계 이하에 들었나)
            p_up = 1 - _probs(z, np.array([(0 - mu) / s]))[0]
            lo, hi = mu + s * np.quantile(z, 0.1), mu + s * np.quantile(z, 0.9)
            rows.append(dict(t=t, 모형=nm, RPS=np.mean((cdf - obs) ** 2), Brier=(p_up - (y > 0)) ** 2, 적중80=float(lo <= y <= hi)))
    return pd.DataFrame(rows)


def nw_t(d: np.ndarray, lag: int) -> float:
    d = d - 0
    n, m = len(d), d.mean()
    u = d - m
    s = u @ u / n + 2 * sum((1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n for k in range(1, lag + 1))
    return m / np.sqrt(s / n)


def run():
    sys.stdout.reconfigure(errors="replace")
    pd.set_option("display.width", 220)
    L = A.load_levels()
    X = CG.nodes(L)
    P = LS.weekly(L, list(CG.PX.values()), CG.START).rename(columns={v: k for k, v in CG.PX.items()}).reindex(X.index)
    allnodes = list(X.columns)
    out = []
    for h in (4, 13):
        for p in CG.PRICES:
            feats = {"M2 온톨로지": [p] + CG.PA2[p], "M3 전부 넣기": allnodes}
            R = run_one(P[p], X, feats, h)
            W = R.pivot(index="t", columns="모형", values="RPS")
            for nm in MODELS:
                g = R[R.모형 == nm]
                row = dict(기간=f"{h}주", 품목=p, 모형=nm, 시험주=len(g), RPS=round(g.RPS.mean(), 4), Brier=round(g.Brier.mean(), 4),
                           적중80=round(g.적중80.mean(), 3))
                if nm != "M0 과거 분포":
                    base = "M0 과거 분포" if nm == "M1 변동성" else "M1 변동성"
                    d = (W[base] - W[nm]).values                                           # + = 개선
                    row["비교"] = base[:2]
                    row["RPS_개선%"] = round(100 * d.mean() / W[base].mean(), 1)
                    row["t_NW"] = round(nw_t(d, h - 1), 2)
                    half = W.index < pd.Timestamp("2022-01-01")
                    row["2017_21"] = round(100 * d[half].mean() / W[base][half].mean(), 1)
                    row["2022_"] = round(100 * d[~half].mean() / W[base][~half].mean(), 1)
                out.append(row)
            print(f"  {h}주 {p} 완료", flush=True)
    O = pd.DataFrame(out)
    print(O.to_string(index=False))
    O.to_csv(A.DATA / "range_bt_compare.csv" if hasattr(A, "DATA") else "data/range_bt_compare.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    run()
