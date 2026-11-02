"""발표 5장 '퀀트 전략 16개' 카드용 3D — 그림 하나를 기본 도형(면·선·점·막대)으로 한 번 정의하고
① matplotlib 회전 GIF(img/q16_NN.gif, PPT·HTML 카드) ② Plotly 3D 정의(img/q16_specs.json, HTML 클릭 시 실시간 회전)로 두 번 그림.

사용 (프로젝트 루트): .venv\\Scripts\\python.exe docs\\발표\\scripts\\charts_q16.py
'실제' 표시 그림은 대시보드 자료로 계산, 나머지는 판단 규칙을 보여 주는 개념도.
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
from matplotlib import cm
from PIL import Image

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import analytics as A  # noqa: E402
import ls_strategies as LS  # noqa: E402
import procurement as PR  # noqa: E402

OUT = HERE / "img"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 6})
G, GOLD, RED, BLUE, GREY = "#1E3A2F", "#C8A24A", "#D9582B", "#2F6E8F", "#9A9A94"
FRAMES, SIZE, DPI = 18, (2.4, 1.95), 100
rng = np.random.default_rng(7)


# ── 기본 도형 (dict) → matplotlib · Plotly
def surf(x, y, z, cmap="viridis", alpha=0.92):
    return dict(k="surf", x=np.asarray(x, float), y=np.asarray(y, float), z=np.asarray(z, float), cmap=cmap, alpha=alpha)


def line(x, y, z, color=G, w=2.0):
    return dict(k="line", x=np.asarray(x, float), y=np.asarray(y, float), z=np.asarray(z, float), color=color, w=w)


def pts(x, y, z, color=RED, s=10):
    return dict(k="pts", x=np.asarray(x, float), y=np.asarray(y, float), z=np.asarray(z, float), color=color, s=s)


def bars(x, y, h, colors, dx=0.6, dy=0.6, base=0.0):
    return dict(k="bars", x=np.asarray(x, float), y=np.asarray(y, float), h=np.asarray(h, float), colors=list(colors), dx=dx, dy=dy, base=base)


CMAPS = {"viridis": "Viridis", "RdYlGn": "RdYlGn", "cividis": "Cividis", "YlOrRd": "YlOrRd", "Greys": "Greys", "RdBu_r": "RdBu", "PuBuGn": "PuBuGn", "summer": "Greens"}


def to_mpl(ax, p):
    if p["k"] == "surf":
        X, Y = np.meshgrid(p["x"], p["y"])
        ax.plot_surface(X, Y, p["z"], cmap=p["cmap"], alpha=p["alpha"], linewidth=0, antialiased=True, rstride=1, cstride=1)
    elif p["k"] == "line":
        ax.plot(p["x"], p["y"], p["z"], color=p["color"], lw=p["w"])
    elif p["k"] == "pts":
        ax.scatter(p["x"], p["y"], p["z"], color=p["color"], s=p["s"], depthshade=False)
    elif p["k"] == "bars":
        ax.bar3d(p["x"] - p["dx"] / 2, p["y"] - p["dy"] / 2, np.full(len(p["h"]), p["base"]), p["dx"], p["dy"], p["h"] - p["base"],
                 color=p["colors"], shade=True, alpha=0.95)


def to_plotly(p):
    r = lambda a, d=3: np.round(np.asarray(a, float), d).tolist()
    if p["k"] == "surf":
        return [dict(type="surface", x=r(p["x"]), y=r(p["y"]), z=r(p["z"]), colorscale=CMAPS.get(p["cmap"], "Viridis"), showscale=False,
                     reversescale=p["cmap"] == "RdBu_r", opacity=p["alpha"], hoverinfo="skip")]
    if p["k"] == "line":
        return [dict(type="scatter3d", mode="lines", x=r(p["x"]), y=r(p["y"]), z=r(p["z"]), line=dict(color=p["color"], width=p["w"] * 2.5), hoverinfo="skip")]
    if p["k"] == "pts":
        return [dict(type="scatter3d", mode="markers", x=r(p["x"]), y=r(p["y"]), z=r(p["z"]), marker=dict(color=p["color"], size=max(3, p["s"] / 3)), hoverinfo="skip")]
    out = []                                                                  # 막대 = 상자 메시 (색마다 하나)
    for col in dict.fromkeys(p["colors"]):
        vx, vy, vz, I, J, K = [], [], [], [], [], []
        for x0, y0, h0, c in zip(p["x"], p["y"], p["h"], p["colors"]):
            if c != col:
                continue
            b = len(vx); xs = [x0 - p["dx"] / 2, x0 + p["dx"] / 2]; ys = [y0 - p["dy"] / 2, y0 + p["dy"] / 2]; zs = [p["base"], h0]
            for zz in zs:
                for yy in ys:
                    for xx in xs:
                        vx.append(xx); vy.append(yy); vz.append(zz)
            for f in [(0, 1, 2), (1, 3, 2), (4, 5, 6), (5, 7, 6), (0, 1, 4), (1, 5, 4), (2, 3, 6), (3, 7, 6), (0, 2, 4), (2, 6, 4), (1, 3, 5), (3, 7, 5)]:
                I.append(b + f[0]); J.append(b + f[1]); K.append(b + f[2])
        out.append(dict(type="mesh3d", x=r(vx), y=r(vy), z=r(vz), i=I, j=J, k=K, color=col, flatshading=True, opacity=1, hoverinfo="skip"))
    return out


def render(num, title, prims, axes, real=False, elev=24):
    fig = plt.figure(figsize=SIZE, dpi=DPI); ax = fig.add_subplot(111, projection="3d")
    for p in prims:
        to_mpl(ax, p)
    ax.set_xlabel(axes[0], labelpad=-8); ax.set_ylabel(axes[1], labelpad=-8); ax.set_zlabel(axes[2], labelpad=-9)
    ax.tick_params(pad=-3, labelsize=4.5)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_alpha(0.0)
    fig.subplots_adjust(0, 0, 1, 1)
    frames = []
    for k in range(FRAMES):
        ax.view_init(elev, -60 + 360 * k / FRAMES)
        fig.canvas.draw()
        im = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
        frames.append(im.quantize(colors=96, method=Image.Quantize.MEDIANCUT))
    plt.close(fig)
    frames[0].save(OUT / f"q16_{num:02d}.gif", save_all=True, append_images=frames[1:], duration=140, loop=0, optimize=True, disposal=2)
    traces = sum((to_plotly(p) for p in prims), [])
    return dict(title=title, real=real, axes=list(axes), traces=traces)


def main():
    L = A.load_levels()
    P = LS.weekly(L, LS.BASKET, "2001-01-01"); P = P[[c for c in LS.BASKET if c in P]]; R = LS._ret(P); lp = np.log(P)
    six = ["yf:ZC=F", "yf:ZS=F", "yf:ZW=F", "yf:CL=F", "yf:GC=F", "yf:HG=F"]
    tail = P.index[-156:]; xi = np.arange(len(tail))
    S = {}
    # 1 모멘텀 (실제): 품목 × 최근 3년 × 52주 수익
    m52 = 100 * lp.diff(52).loc[tail, six]
    S[1] = render(1, "모멘텀 — 품목별 지난 52주 수익 (+ 롱 · − 숏)", [surf(xi, range(6), m52.T.values, "RdYlGn")], ("주 (최근 3년)", "품목", "52주 수익 %"), True)
    # 5 칼만 추세 (개념): 잡음 섞인 가격 → 매끄럽게 할수록 추세만 남음
    t = np.arange(120); price = 100 + 0.12 * t + 6 * np.sin(t / 18) + rng.normal(0, 2.2, len(t))
    qs = np.linspace(0.04, 0.6, 12)
    Z = np.array([pd.Series(price).ewm(alpha=q).mean().values for q in qs])
    S[5] = render(5, "추세 추종 (칼만) — 잡음을 걸러 추세 기울기만", [surf(t, qs, Z, "cividis"), line(t, np.full(len(t), 0.62), price, RED, 1.0)], ("시간", "반응 속도 q", "추정 추세"))
    # 14 돌파 (실제 옥수수): 가격 + 55주 채널
    c = P["yf:ZC=F"].loc[P.index[-260:]]; hi = c.rolling(55).max().shift(1); lo = c.rolling(55).min().shift(1); x = np.arange(len(c))
    brk = np.where((c > hi).values)[0]
    S[14] = render(14, "돌파 매매 — 55주 고점·저점 채널을 뚫으면 진입 (옥수수)", [line(x, np.zeros(len(c)), c.values, G, 1.6),
                   surf(x, [0, 1], np.vstack([hi.values, hi.values]), "YlOrRd", 0.5), surf(x, [0, 1], np.vstack([lo.values, lo.values]), "PuBuGn", 0.5),
                   pts(brk, np.zeros(len(brk)), c.values[brk], RED, 6)], ("주 (최근 5년)", "", "가격"), True)
    # 13 단기 반전 (실제): 지난주 수익 분위 × 변동성 분위 → 다음 주 평균 수익
    vol = R.rolling(26, min_periods=13).std(); nxt = R.shift(-1)
    df = pd.DataFrame({"r": R.stack(), "v": vol.stack(), "n": nxt.stack()}).dropna()
    df["rq"] = pd.qcut(df.r, 7, labels=False); df["vq"] = pd.qcut(df.v, 5, labels=False)
    M = df.groupby(["vq", "rq"]).n.mean().unstack().values
    S[13] = render(13, "단기 반전 — 지난주 급등·급락 뒤 다음 주 평균 (거의 평평 = 효과 없음)", [surf(range(7), range(5), M, "RdBu_r"),
                   surf(range(7), range(5), np.zeros((5, 7)), "Greys", 0.25)], ("지난주 수익 분위", "변동성 분위", "다음 주 수익 %"), True)
    # 2 가치 (실제): 5년 평균 대비 편차
    v5 = 100 * (lp - lp.rolling(260, min_periods=156).mean()).loc[tail, six]
    S[2] = render(2, "평균회귀 (가치) — 5년 평균 대비 편차 (아래 = 싸다 → 롱)", [surf(xi, range(6), v5.T.values, "RdYlGn"),
                  surf(xi, range(6), np.zeros((6, len(xi))), "Greys", 0.25)], ("주 (최근 3년)", "품목", "편차 %"), True)
    # 4 페어 (실제): 옥수수·밀 정규화 가격과 간격
    pc = (lp[["yf:ZC=F", "yf:ZW=F"]].loc[tail] - lp[["yf:ZC=F", "yf:ZW=F"]].loc[tail].iloc[0]) * 100
    S[4] = render(4, "페어 트레이딩 — 옥수수·밀 간격이 벌어지면 역으로", [surf(xi, [0, 1], pc.T.values, "cividis", 0.55),
                  line(xi, np.zeros(len(xi)), pc.iloc[:, 0].values, G, 1.8), line(xi, np.ones(len(xi)), pc.iloc[:, 1].values, GOLD, 1.8)], ("주 (최근 3년)", "옥수수 · 밀", "누적 %"), True)
    # 3 PCA 통계적 차익 (실제): 3품목 주간 수익 구름 + 주성분 평면 + 잔차
    X = R[["yf:ZC=F", "yf:ZS=F", "yf:ZW=F"]].dropna().iloc[-150:].values
    mu = X.mean(0); U, s_, Vt = np.linalg.svd(X - mu, full_matrices=False)
    g = np.linspace(-1, 1, 8) * s_[0] / np.sqrt(len(X)) * 2.5
    GA, GB = np.meshgrid(g, g * s_[1] / s_[0])
    plane = mu + GA[..., None] * Vt[0] + GB[..., None] * Vt[1]
    proj = mu + ((X - mu) @ Vt[:2].T) @ Vt[:2]
    Xp = plane[..., 0]; Yp = plane[..., 1]; Zp = plane[..., 2]
    sticks = [line([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], RED, 0.6) for a, b in list(zip(X, proj))[:40]]
    fig_prims = [dict(k="psurf", X=Xp, Y=Yp, Z=Zp)] + [pts(X[:, 0], X[:, 1], X[:, 2], BLUE, 4)] + sticks
    S[3] = render_psurf(3, "통계적 차익 (PCA) — 공통 요인 평면에서 벗어난 거리(빨강)가 신호", fig_prims, ("옥수수 %", "대두 %", "밀 %"))
    # 9 캐리 (개념): 만기별 선물가격 곡선이 백워데이션 → 콘탱고로
    mat = np.arange(1, 13); tt = np.arange(30)
    Zc = np.array([100 + (k - 15) / 15 * 0.8 * (mat - 1) * (-1) for k in tt])
    S[9] = render(9, "캐리 — 근월물이 비싸면(백워데이션) 롱, 싸면(콘탱고) 숏", [surf(mat, tt, Zc, "YlOrRd")], ("만기 (개월)", "시간", "선물가격"))
    # 15 리스크 패리티 (실제 공분산: 옥수수·원유·금): 비중 삼각형 위 위험 기여 불균형
    C = R[["yf:ZC=F", "yf:CL=F", "yf:GC=F"]].dropna().iloc[-260:].cov().values
    gw = np.linspace(0.02, 0.96, 24); Zr = np.full((24, 24), np.nan)
    for a_, w1 in enumerate(gw):
        for b_, w2 in enumerate(gw):
            if w1 + w2 < 0.98:
                w = np.array([w1, w2, 1 - w1 - w2]); rc = w * (C @ w); rc = rc / rc.sum(); Zr[b_, a_] = rc.max() - rc.min()
    S[15] = render(15, "리스크 패리티 — 위험 기여가 같아지는 바닥(0)이 목표 비중", [surf(gw, gw, np.nan_to_num(Zr, nan=np.nanmax(Zr)), "viridis")], ("옥수수 비중", "원유 비중", "위험 기여 차"), True)
    # 16 국면 전환 (개념): 위기 확률 리본
    tt = np.arange(200); pcr = 1 / (1 + np.exp(-(np.sin(tt / 23) * 3 + np.where((tt > 120) & (tt < 140), 5, 0) - 2)))
    S[16] = render(16, "국면 전환 — '지금 위기일 확률'로 위험을 줄이거나 늘림", [surf(tt, [0, 1], np.vstack([1 - pcr, pcr]), "YlOrRd")], ("시간", "평시 · 위기", "확률"))
    # 8 팩터 (개념): 가치 순위 × 모멘텀 순위 → 기대 수익 평면 + 품목 점
    gg = np.linspace(0, 1, 12); Fz = 0.6 * gg[None, :] + 0.4 * gg[:, None] - 0.5
    px_, py_ = rng.uniform(0, 1, 19), rng.uniform(0, 1, 19)
    S[8] = render(8, "팩터 투자 — 가치·모멘텀 순위를 합친 점수로 롱·숏", [surf(gg, gg, Fz, "RdYlGn", 0.7),
                  pts(px_, py_, 0.6 * px_ + 0.4 * py_ - 0.5 + rng.normal(0, 0.08, 19), G, 8)], ("가치 순위", "모멘텀 순위", "점수"))
    # 12 횡단면 (실제): 최근 6개 시점 × 8품목 26주 수익 순위 막대
    items8 = ["yf:ZC=F", "yf:ZS=F", "yf:ZW=F", "yf:CL=F", "yf:NG=F", "yf:GC=F", "yf:HG=F", "yf:KC=F"]
    dts = P.index[-26 * 6::26][:6]; rk = lp.diff(26).loc[dts, items8].rank(axis=1)
    bx, by, bh, bc = [], [], [], []
    for a_, d in enumerate(dts):
        for b_, i in enumerate(items8):
            v = rk.loc[d, i]; bx.append(b_); by.append(a_); bh.append(v); bc.append("#3E8E5E" if v >= 6 else (RED if v <= 3 else "#CFd2CB"))
    S[12] = render(12, "횡단면 알파 — 매 시점 순위 상위(초록) 롱 · 하위(빨강) 숏", [bars(bx, by, bh, bc)], ("품목", "시점 (반년 간격)", "순위"), True)
    # 17 머신러닝 (개념): 곡면 적합 vs 직선 평면
    g2 = np.linspace(-2, 2, 22); GX, GY = np.meshgrid(g2, g2)
    Zm = np.sin(GX * 1.4) * np.cos(GY) + 0.3 * GX
    S[17] = render(17, "머신러닝 알파 — 곡면으로 패턴을 맞추지만 잡음까지 외우기 쉬움", [surf(g2, g2, Zm, "viridis"), surf(g2, g2, 0.3 * GX, "Greys", 0.3)], ("설명변수 1", "설명변수 2", "예측 수익"))
    # 7 이벤트 (실제): 품목 × 보고서 × 변동 배수
    ev = pd.read_csv(ROOT / "data" / "quant16_events.csv", encoding="utf-8-sig", index_col=0)
    ex, ey, eh, ec = [], [], [], []
    for a_, it in enumerate(ev.index):
        for b_, col in enumerate(["WASDE", "분기 보고서"]):
            ex.append(a_); ey.append(b_); eh.append(ev.loc[it, col]); ec.append(GOLD if b_ == 0 else RED)
    S[7] = render(7, "이벤트 드리븐 — USDA 보고서 날 변동 배수 (" + "·".join(ev.index) + ")", [bars(ex, ey, eh, ec, base=0.0),
                  surf([-0.5, len(ev) - 0.5], [-0.5, 1.5], np.ones((2, 2)), "Greys", 0.25)], ("품목", "WASDE · 분기", "평소 대비 배"), True)
    # 18 집행 (실제 규칙): 대시보드 구매 속도 함수 pace(점수, 위험, 남은 물량, 남은 주)
    sc = np.linspace(0, 1, 20); wk = np.arange(1, 14)
    Zp = np.array([[PR.pace(s0, 0.5, 1.0, int(w)) for s0 in sc] for w in wk])
    S[18] = render(18, "집행 (분할 매수) — 점수가 좋을수록·기한이 가까울수록 더 빨리 산다", [surf(sc, wk, Zp, "summer")], ("가격 매력 점수", "남은 주", "이번 주 매수 비율"), True)
    # 6 시장조성 (개념, 회색): 호가창
    pr = np.arange(-5, 6); tt6 = np.arange(6)
    hx, hy, hh, hc = [], [], [], []
    for a_ in tt6:
        for p0 in pr:
            if p0 == 0:
                continue
            hx.append(p0); hy.append(a_); hh.append(np.exp(-abs(p0) / 3) * (1 + rng.uniform(0, 0.5))); hc.append("#C9C9C4" if p0 < 0 else "#A6A6A0")
    S[6] = render(6, "시장조성 — 매수·매도 호가를 양쪽에 (해당 없음)", [bars(hx, hy, hh, hc, 0.7, 0.7)], ("가격 단계", "시간", "호가 수량"))
    (OUT / "q16_specs.json").write_text(json.dumps({str(k): v for k, v in S.items()}, ensure_ascii=False), encoding="utf-8")
    print("saved", len(S), "GIF + specs", sum((OUT / f"q16_{k:02d}.gif").stat().st_size for k in S) // 1024, "KB")


def render_psurf(num, title, prims, axes):
    """임의 방향 평면(X·Y·Z 격자)이 있는 그림용."""
    fig = plt.figure(figsize=SIZE, dpi=DPI); ax = fig.add_subplot(111, projection="3d")
    traces = []
    for p in prims:
        if p["k"] == "psurf":
            ax.plot_surface(p["X"], p["Y"], p["Z"], color=GOLD, alpha=0.45, linewidth=0)
            traces.append(dict(type="surface", x=np.round(p["X"], 3).tolist(), y=np.round(p["Y"], 3).tolist(), z=np.round(p["Z"], 3).tolist(),
                               colorscale=[[0, GOLD], [1, GOLD]], showscale=False, opacity=0.5, hoverinfo="skip"))
        else:
            to_mpl(ax, p); traces += to_plotly(p)
    ax.set_xlabel(axes[0], labelpad=-8); ax.set_ylabel(axes[1], labelpad=-8); ax.set_zlabel(axes[2], labelpad=-9); ax.tick_params(pad=-3, labelsize=4.5)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_alpha(0.0)
    fig.subplots_adjust(0, 0, 1, 1)
    frames = []
    for k in range(FRAMES):
        ax.view_init(22, -60 + 360 * k / FRAMES); fig.canvas.draw()
        frames.append(Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB").quantize(colors=96, method=Image.Quantize.MEDIANCUT))
    plt.close(fig)
    frames[0].save(OUT / f"q16_{num:02d}.gif", save_all=True, append_images=frames[1:], duration=140, loop=0, optimize=True, disposal=2)
    return dict(title=title, real=True, axes=list(axes), traces=traces)


if __name__ == "__main__":
    main()
