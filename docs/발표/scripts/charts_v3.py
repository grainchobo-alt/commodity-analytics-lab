"""v3 추가 3D 차트: 머신러닝 예측 지형(평면 vs 나무 묶음), 위험 크기 지형(품목 × 연도), 연도별 효율적 투자선 3D."""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(r"C:\Users\user\Desktop\Commodity"); sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10, "savefig.dpi": 200})
OUT = Path(__file__).resolve().parent / "img"; OUT.mkdir(exist_ok=True)
G, GOLD, RED, BLUE = "#1E3A2F", "#C8A24A", "#D9582B", "#2F6E8F"


def save(f, name):
    f.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white"); plt.close(f); print("saved", name)


def clean3d(ax):
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((0.97, 0.97, 0.96, 1)); a.pane.set_edgecolor("#dddddd")


# ── 43 머신러닝 예측 지형: 같은 자료를 직선 모델 vs 나무 묶음으로 학습 (합성 예시)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
rng = np.random.default_rng(3)
n = 1500
x1, x2 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)            # x1 = 가격 위치(낮을수록 쌈), x2 = 변동성 분위
y = 2.0 * (x1 < 0.25) * (x2 > 0.5) - 1.2 * (x1 > 0.8) + 0.8 * np.sin(3 * x2) * x1 + rng.normal(0, 0.35, n)
X = np.c_[x1, x2]
g = np.linspace(0, 1, 45); GX, GY = np.meshgrid(g, g); grid = np.c_[GX.ravel(), GY.ravel()]
ZL = LinearRegression().fit(X, y).predict(grid).reshape(GX.shape)
ZB = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0).fit(X, y).predict(grid).reshape(GX.shape)
f = plt.figure(figsize=(11, 4.6))
for k, (Z, ttl, cmap) in enumerate([(ZL, "직선 모델 — 평평한 판 하나", "Greys"), (ZB, "나무 묶음 — 계단 지형으로 복잡한 관계 포착", "viridis")]):
    ax = f.add_subplot(1, 2, k + 1, projection="3d"); clean3d(ax)
    ax.plot_surface(GX, GY, Z, cmap=cmap, linewidth=0, antialiased=True, alpha=0.95, vmin=-1.5, vmax=2.2)
    ax.scatter(x1[::6], x2[::6], y[::6], s=2, color=RED, alpha=0.25)
    ax.set_xlabel("가격 위치 (0 = 1년 최저)", labelpad=2); ax.set_ylabel("변동성 분위", labelpad=2); ax.set_zlabel("4주 뒤 수익률 (예시)", labelpad=0)
    ax.set_zlim(-2, 2.8); ax.view_init(elev=26, azim=-130); ax.set_title(ttl, fontweight="bold", color=G, fontsize=12)
save(f, "c43_ml3d")

# ── 44 위험 크기 지형: 품목 × 연도 × 연 변동성 (실제 자료, 26주 실현 변동성)
import analytics as A, ls_strategies as LS
adj = A.load_levels(roll_adjusted=True).loc[:"2026-10-26"]
KN = {"yf:ZC=F": "옥수수", "yf:ZW=F": "밀", "yf:ZS=F": "대두", "yf:ZM=F": "대두박", "yf:CL=F": "원유", "yf:NG=F": "천연가스", "yf:HG=F": "구리", "yf:GC=F": "금"}
P = LS.weekly(adj, list(KN), "2005-01-01")[list(KN)].rename(columns=KN)
V = (np.log(P).diff().rolling(26).std() * np.sqrt(52) * 100).loc["2007":].resample("ME").last().clip(upper=110)
t = np.array([d.year + (d.month - 0.5) / 12 for d in V.index])
TT, YY = np.meshgrid(t, np.arange(V.shape[1]))
f = plt.figure(figsize=(10, 5.4)); ax = f.add_subplot(projection="3d"); clean3d(ax)
s = ax.plot_surface(TT, YY, V.T.values, cmap="YlOrRd", linewidth=0, antialiased=True, rstride=1, cstride=1, vmin=10, vmax=80)
ax.set_yticks(range(V.shape[1])); ax.set_yticklabels(V.columns, fontsize=9); ax.tick_params(axis="y", pad=-2)
ax.set_zlabel("연 변동성 (%)"); ax.set_box_aspect((2.4, 1.7, 0.7)); ax.view_init(elev=30, azim=-48)
f.colorbar(s, shrink=0.45, pad=0.0, orientation="horizontal", location="bottom", label="연 변동성 % (빨강 = 위험 큼, 천연가스는 110%에서 자름)")
ax.set_title("위험 크기 지형 — 품목 × 연도, 큰 위험은 몰려서 옴", loc="left", fontweight="bold", color=G)
save(f, "c44_vol3d")

# ── 45 연도별 효율적 투자선 3D: 매년 말 직전 5년 자료로 3대 전략 조합의 경계선 → 시간 축으로 쌓기
W = pd.read_parquet(ROOT / "data/voltgt_weekly.parquet")
BASE = {"기준 · MinVar 롱 (⑥)": "최소분산", "기준 · 52주 모멘텀 (④)": "모멘텀", "V1 가치 (5년 평균 대비 싼 품목 롱)": "가치"}
X3 = pd.DataFrame({v_: W[f"{k}|10|4"] for k, v_ in BASE.items()}).loc["2007":].dropna()
Wr = np.random.default_rng(0).dirichlet(np.full(3, 0.5), 6000)
wmix = np.array([0.5, 0.25, 0.25])
f = plt.figure(figsize=(9.6, 6.0)); ax = f.add_subplot(projection="3d"); clean3d(ax)
years = list(range(2011, 2027))
cmap = matplotlib.colormaps["viridis"]
mix_pts, tan_pts = [], []
for j, yr in enumerate(years):
    Xw = X3.loc[f"{yr - 4}":f"{yr}"]
    mu, S = Xw.mean().values * 52, Xw.cov().values * 52
    vol = np.sqrt(np.einsum("ij,jk,ik->i", Wr, S, Wr)); ret = Wr @ mu
    o = np.argsort(vol); best = -np.inf; fv, fr = [], []
    for i in o:                                        # 위험이 커질 때 수익이 더 높아지는 점만 = 경계선(효율적 투자선)
        if ret[i] > best:
            best = ret[i]; fv.append(vol[i]); fr.append(ret[i])
    ax.plot(fv, np.full(len(fv), yr), fr, color=cmap(j / (len(years) - 1)), lw=2.2)
    sh = ret / vol; k = int(np.argmax(sh)); tan_pts.append((vol[k], yr, ret[k]))
    mix_pts.append((np.sqrt(wmix @ S @ wmix), yr, wmix @ mu))
mp, tp = np.array(mix_pts), np.array(tan_pts)
ax.plot(tp[:, 0], tp[:, 1], tp[:, 2], color=RED, lw=1.4, ls="--", marker="o", ms=3, label="그해 최고 Sharpe 조합 (사후)")
ax.plot(mp[:, 0], mp[:, 1], mp[:, 2], color=GOLD, lw=2.4, marker="*", ms=9, mec="#333", label="우리 혼합 (최소분산 50·모멘텀 25·가치 25, 고정)")
ax.set_xlabel("연 변동성 (%)", labelpad=6); ax.set_ylabel("기준 연도 (직전 5년)", labelpad=8); ax.set_zlabel("연 수익 (%)", labelpad=4)
ax.set_box_aspect((1.3, 1.8, 0.9)); ax.view_init(elev=22, azim=-58)
ax.legend(loc="upper left", fontsize=9, frameon=False)
ax.set_title("3D 효율적 투자선 — 연도별 경계선과 우리 혼합의 위치", loc="left", fontweight="bold", color=G)
save(f, "c45_frontier3d")
gap = (tp[:, 2] / tp[:, 0]) - (mp[:, 2] / mp[:, 0])
print("혼합 vs 사후 최고 Sharpe 차이 (연도별 평균):", round(float(gap.mean()), 2), "| 혼합 Sharpe 범위", np.round((mp[:, 2] / mp[:, 0]).min(), 2), np.round((mp[:, 2] / mp[:, 0]).max(), 2))
