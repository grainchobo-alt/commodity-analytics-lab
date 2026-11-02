"""발표 별첨 7 '시계열 분해: 추세 · 계절성 · 잔차(정상성)' — 곡물 5품목 주간 로그가격을 STL(52주, 계절 평활 51)로 분해.
계절성의 실제 쓸모 = 주차 더미(해마다 같은 모양)로 주간 수익률을 설명한 R² — 같은 기간 안 vs 2008–16로 맞춰 2017–에 적용.

결과: img/c71_stl3d.png(분해 리본) · c72_season3d.png(계절 지형) · c73_resid3d.png(잔차 정상성) — 옥수수, PPT용
      img/ts7.json — 5품목 분해 · 계절 격자 · 잔차 이동 통계(HTML 3D) + 정상성 검정 · 분산 몫(장표 표 · 막대)
사용 (프로젝트 루트): .venv\\Scripts\\python.exe docs\\발표\\scripts\\charts_ts7.py
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import analytics as A  # noqa: E402
import ls_strategies as LS  # noqa: E402

OUT = HERE / "img"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 8})
ITEMS = {"옥수수": "yf:ZC=F", "대두": "yf:ZS=F", "밀": "yf:ZW=F", "대두박": "yf:ZM=F", "대두유": "yf:ZL=F"}
WINS = [13, 26, 52, 104]
LAYERS = ["원가격(평균 뺌)", "추세", "계절성", "잔차"]
COLS = ["#1E3A2F", "#2F6E8F", "#C8A24A", "#D9582B"]


def r(x, n=2):
    return np.round(np.asarray(x, float), n).tolist()


def decompose(p: pd.Series) -> dict:
    y = 100 * np.log(p.dropna())                                                           # 로그가격 × 100 → 차이가 곧 %
    st = STL(y, period=52, seasonal=51, robust=True).fit()                               # 계절 모양이 해마다 천천히만 바뀌게 (기본 7은 잡음까지 계절로 흡수)
    tr, se, re = st.trend, st.seasonal, st.resid
    d = y.diff().dropna()
    var = {k: float(np.var(c.diff().dropna())) for k, c in (("추세", tr), ("계절성", se), ("잔차", re))}
    tot = sum(var.values())
    pv = lambda f, x: float(f(x.dropna(), **({"autolag": "AIC"} if f is adfuller else {"nlags": "auto"}))[1])   # noqa: E731
    tests = {"레벨 ADF": pv(adfuller, y), "레벨 KPSS": pv(kpss, y), "수익률 ADF": pv(adfuller, d), "수익률 KPSS": pv(kpss, d),
             "잔차 ADF": pv(adfuller, re), "잔차 KPSS": pv(kpss, re)}
    wk_d = np.minimum(d.index.isocalendar().week.values.astype(int), 52)                 # 해마다 같은 계절 모양(주차 더미)으로 주간 수익률 설명
    Dm = pd.get_dummies(wk_d).astype(float).values
    r2 = lambda tr_, te_: 1 - np.mean((d.values[te_] - Dm[te_] @ np.linalg.lstsq(Dm[tr_], d.values[tr_], rcond=None)[0]) ** 2) / np.var(d.values[te_])   # noqa: E731
    half = d.index < pd.Timestamp("2017-01-01")
    season_r2 = dict(같은기간=float(r2(np.ones(len(d), bool), np.ones(len(d), bool))), 시험=float(r2(half, ~half)))
    det = (y - tr)                                                                        # 추세 뺀 가격 = 계절 + 잔차
    wk = np.minimum(det.index.isocalendar().week.values.astype(int), 52)
    G = pd.DataFrame({"v": det.values, "w": wk, "yr": det.index.year}).groupby(["yr", "w"]).v.mean().unstack()
    G = G.loc[[yy for yy in G.index if G.loc[yy].notna().sum() >= 50]].interpolate(axis=1, limit_direction="both")
    step = 4
    k0 = max(WINS)                                                                        # 가장 긴 창이 찰 때부터 (빈 구간을 0으로 채우지 않음)
    rm = np.array([re.rolling(w).mean().values[k0::step] for w in WINS])
    rs = np.array([re.rolling(w).std().values[k0::step] for w in WINS])
    return dict(y=y, tr=tr, se=se, re=re, G=G, rm=rm, rs=rs, rd=re.index[k0::step], var={k: v / tot for k, v in var.items()}, tests=tests, sr2=season_r2)


def ribbon(ax, x, k, z, col):
    X = np.vstack([x, x]); Y = np.vstack([np.full_like(x, k - 0.28), np.full_like(x, k + 0.28)]); Z = np.vstack([z, z])
    ax.plot_surface(X, Y, Z, color=col, alpha=0.85, linewidth=0, shade=False)
    ax.plot(x, np.full_like(x, k), z, color=col, lw=0.8)


def pngs(c: dict):
    x = np.arange(len(c["y"])) / 52 + c["y"].index[0].year
    layers = [c["y"] - c["y"].mean(), c["tr"] - c["tr"].mean(), c["se"], c["re"]]
    fig = plt.figure(figsize=(7.2, 4.0), dpi=170); ax = fig.add_subplot(111, projection="3d")
    for k, (z, col) in enumerate(zip(layers, COLS)):
        ribbon(ax, x, k, z.values, col)
    ax.set_yticks(range(4)); ax.set_yticklabels(LAYERS, fontsize=7); ax.set_xlabel("연도"); ax.set_zlabel("로그가격 차이 (%)")
    ax.set_xticks([2010, 2014, 2018, 2022, 2026]); ax.set_xticklabels(["2010", "2014", "2018", "2022", "2026"])
    ax.view_init(24, -64); ax.set_box_aspect((2.4, 1.3, 0.8)); fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(OUT / "c71_stl3d.png", transparent=True); plt.close(fig)
    G = c["G"]
    fig = plt.figure(figsize=(4.2, 3.3), dpi=170); ax = fig.add_subplot(111, projection="3d")
    W, Y = np.meshgrid(G.columns.values, G.index.values)
    ax.plot_surface(W, Y, G.values, cmap="RdYlGn_r", linewidth=0, antialiased=True)
    ax.set_yticks([2010, 2014, 2018, 2022]); ax.set_yticklabels(["2010", "2014", "2018", "2022"]); ax.set_xticks([1, 13, 26, 39, 52])
    ax.set_xlabel("연중 주차"); ax.set_ylabel("연도"); ax.set_zlabel("추세 뺀 가격 (%)"); ax.view_init(30, -55); fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(OUT / "c72_season3d.png", transparent=True); plt.close(fig)
    fig = plt.figure(figsize=(4.2, 3.3), dpi=170); ax = fig.add_subplot(111, projection="3d")
    t = np.array([x.year + x.dayofyear / 365.25 for x in c["rd"]])
    T, K = np.meshgrid(t, np.arange(len(WINS)))
    S = np.nan_to_num(c["rs"]); col = plt.cm.YlOrRd((S - np.nanmin(S)) / (np.nanmax(S) - np.nanmin(S) + 1e-9))
    ax.plot_surface(T, K, np.nan_to_num(c["rm"]), facecolors=col, linewidth=0, shade=False)
    ax.set_yticks(range(len(WINS))); ax.set_yticklabels([f"{w}주" for w in WINS], fontsize=7); ax.set_xlabel("연도"); ax.set_zlabel("잔차 이동 평균 (%)")
    lim = float(np.nanpercentile(np.abs(c["rm"]), 99)) * 1.1
    ax.set_zlim(-lim, lim); ax.set_xticks([2010, 2014, 2018, 2022, 2026]); ax.set_xticklabels(["2010", "2014", "2018", "2022", "2026"]); ax.view_init(28, -60); fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(OUT / "c73_resid3d.png", transparent=True); plt.close(fig)


def main():
    L = A.load_levels(roll_adjusted=True)
    P = LS.weekly(L, list(ITEMS.values()), "2008-01-01").rename(columns={v: k for k, v in ITEMS.items()})
    out = {"items": {}, "tests": [], "share": []}
    for nm in ITEMS:
        c = decompose(P[nm])
        if nm == "옥수수":
            pngs(c)
        out["items"][nm] = dict(d=[f"{x:%Y-%m-%d}" for x in c["y"].index], y=r(c["y"] - c["y"].mean()), tr=r(c["tr"] - c["tr"].mean()), se=r(c["se"]), re=r(c["re"]),
                                gw=[int(w) for w in c["G"].columns], gy=[int(v) for v in c["G"].index], gz=r(c["G"].values, 1),
                                rd=[f"{x:%Y-%m-%d}" for x in c["rd"]], wins=WINS, rm=r(np.nan_to_num(c["rm"], nan=0.0)), rs=r(np.nan_to_num(c["rs"], nan=0.0)))
        t = c["tests"]
        ok = lambda a, k: "정상" if (a < 0.05 and k > 0.05) else ("비정상" if (a > 0.05 and k < 0.05) else "애매")   # noqa: E731
        out["tests"].append(dict(품목=nm, 레벨=ok(t["레벨 ADF"], t["레벨 KPSS"]), 레벨p=f"{t['레벨 ADF']:.2f} / {t['레벨 KPSS']:.2f}",
                                 수익률=ok(t["수익률 ADF"], t["수익률 KPSS"]), 수익률p=f"{t['수익률 ADF']:.2f} / {t['수익률 KPSS']:.2f}",
                                 잔차=ok(t["잔차 ADF"], t["잔차 KPSS"]), 잔차p=f"{t['잔차 ADF']:.2f} / {t['잔차 KPSS']:.2f}"))
        out["share"].append(dict(품목=nm, **{k: round(100 * v, 1) for k, v in c["var"].items()},
                                 계절_같은기간R2=round(100 * c["sr2"]["같은기간"], 1), 계절_시험R2=round(100 * c["sr2"]["시험"], 1)))
        print(nm, out["tests"][-1], out["share"][-1])
    rng = np.random.default_rng(0)                                                        # 가짜 상관: 서로 독립인 랜덤워크 2,000쌍 (주간 자료 길이와 같게)
    n = len(P.dropna()); cl, cr = [], []
    for _ in range(2000):
        e = rng.standard_normal((n, 2)); w = e.cumsum(0)
        cl.append(abs(np.corrcoef(w.T)[0, 1])); cr.append(abs(np.corrcoef(e.T)[0, 1]))
    cl, cr = np.array(cl), np.array(cr)
    out["spur"] = dict(n=n, pairs=2000, level_gt05=round(100 * float((cl > 0.5).mean()), 1), ret_gt01=round(100 * float((cr > 0.1).mean()), 1),
                       level_med=round(float(np.median(cl)), 2), ret_med=round(float(np.median(cr)), 3))
    print("가짜 상관", out["spur"])
    (OUT / "ts7.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("img/c71~c73 PNG · img/ts7.json")


if __name__ == "__main__":
    main()
