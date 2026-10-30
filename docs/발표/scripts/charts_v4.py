"""v4 차트: 퀀트 학문 지도(정밀판), AI 모델 카드별 3D 그림 8개, 겹치지 않는 효율적 투자선."""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(r"C:\Users\user\Desktop\Commodity"); sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 9, "savefig.dpi": 200})
OUT = Path(__file__).resolve().parent / "img"; OUT.mkdir(exist_ok=True)
G, GOLD, RED, BLUE, OCHRE, INK, MUTED = "#1E3A2F", "#C8A24A", "#D9582B", "#2F6E8F", "#B8901C", "#222222", "#6B6B66"


def save(f, name, pad=0.05):
    f.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=pad, facecolor="white"); plt.close(f); print("saved", name)


def ax3d(figsize=(4.2, 3.3)):
    f = plt.figure(figsize=figsize); ax = f.add_subplot(projection="3d")
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((0.97, 0.97, 0.96, 1)); a.pane.set_edgecolor("#dddddd")
    ax.tick_params(labelsize=7, pad=0)
    return f, ax


def lab(ax, x, y, z):
    ax.set_xlabel(x, fontsize=8, labelpad=-4); ax.set_ylabel(y, fontsize=8, labelpad=-4); ax.set_zlabel(z, fontsize=8, labelpad=-6)


# ════════ 46 퀀트 학문 지도: 4개 원 + 원마다 기법·미니 그림 + 겹치는 영역의 결합 분야
f, ax = plt.subplots(figsize=(7.4, 6.3)); ax.set_xlim(-3.15, 3.15); ax.set_ylim(-3.05, 2.85); ax.set_aspect("equal"); ax.axis("off")
R, c = 1.75, 1.0
FIELDS = [("통계학", BLUE, (-c, c), ["확률분포 · 꼬리 위험", "가설 검정 (우연인가?)", "부트스트랩"], (-1, 1)),
          ("계량경제학", G, (c, c), ["GARCH (위험 크기)", "국면 전환 (HMM)", "칼만 필터"], (1, 1)),
          ("머신러닝", OCHRE, (-c, -c), ["Lasso · PCA", "랜덤 포레스트", "LSTM · Chronos · FinBERT"], (-1, -1)),
          ("프로그래밍", RED, (c, -c), ["Python 자동 수집", "백테스트 엔진", "Streamlit 대시보드"], (1, -1))]
for name, col, (cx, cy), items, (sx, sy) in FIELDS:
    ax.add_patch(Circle((cx, cy), R, facecolor=col, alpha=0.16, edgecolor=col, lw=2.2))
for name, col, (cx, cy), items, (sx, sy) in FIELDS:
    tx, ty = cx + sx * 0.72, cy + sy * 0.62                     # 원의 바깥쪽(겹치지 않는) 구역
    ha = "center"
    ax.text(tx, ty + sy * 0.0 + 0.42, name, ha=ha, va="center", fontsize=15, fontweight="bold", color=col)
    for k, it in enumerate(items):
        ax.text(tx, ty + 0.05 - k * 0.27, "· " + it, ha=ha, va="center", fontsize=8.6, color=INK)
# 미니 그림(원 바깥 꼭짓점 쪽)
def inset(x0, y0, w=0.62, h=0.48):
    a = ax.inset_axes([x0, y0, w, h], transform=ax.transData); a.axis("off"); return a
a = inset(-3.1, 2.2); xs = np.linspace(-3.5, 3.5, 200); pdf = np.exp(-xs ** 2 / 2)          # 통계: 종 모양 + 꼬리
a.plot(xs, pdf, color=BLUE, lw=1.6); a.fill_between(xs, 0, pdf, where=xs < -2.33, color=RED, alpha=0.8); a.axvline(-2.33, color=RED, lw=0.8, ls="--")
a = inset(2.48, 2.2); rng = np.random.default_rng(5); v = np.r_[np.full(60, 0.4), np.full(25, 1.6), np.full(60, 0.5)]    # 계량: 변동성 군집
a.plot(rng.normal(0, v), color=G, lw=0.8)
a = inset(-3.1, -2.55)                                                                      # 머신러닝: 판단 나무
nodes = {(0.5, 0.9): None, (0.25, 0.5): (0.5, 0.9), (0.75, 0.5): (0.5, 0.9), (0.12, 0.1): (0.25, 0.5), (0.38, 0.1): (0.25, 0.5), (0.62, 0.1): (0.75, 0.5), (0.88, 0.1): (0.75, 0.5)}
for n, p in nodes.items():
    if p: a.plot([p[0], n[0]], [p[1], n[1]], color=OCHRE, lw=1.2)
for n in nodes: a.scatter(*n, s=40, color=OCHRE, edgecolor="white", zorder=3)
a.set_xlim(0, 1); a.set_ylim(0, 1)
a = inset(2.48, -2.55); a.set_xlim(0, 1); a.set_ylim(0, 1)                                   # 프로그래밍: 코드 창
a.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.9, boxstyle="round,pad=0.02", fc="white", ec=RED, lw=1.3))
a.add_patch(FancyBboxPatch((0.02, 0.78), 0.96, 0.17, boxstyle="round,pad=0.02", fc=RED, ec=RED, lw=0))
a.text(0.5, 0.42, "</>", ha="center", va="center", fontsize=13, fontweight="bold", color=RED, family="Consolas")
# 겹치는 영역: 두 분야가 만나서 생기는 것
LENS = [((0, c + 0.55), "시계열 분석\n(통계 + 계량)"), ((c + 0.55, 0), "백테스트\n(계량 + 코딩)"),
        ((0, -c - 0.55), "AI 모델 구현\n(ML + 코딩)"), ((-c - 0.55, 0), "통계적 학습\n(통계 + ML)")]
for (x, y), t in LENS:
    ax.text(x, y, t, ha="center", va="center", fontsize=8.2, color=INK, fontweight="bold", linespacing=1.25)
ax.add_patch(Circle((0, 0), 0.5, facecolor=G, edgecolor="white", lw=2.5, zorder=5))
ax.text(0, 0.06, "퀀트", ha="center", va="center", fontsize=16, fontweight="bold", color="white", zorder=6)
ax.text(0, -0.24, "Quant", ha="center", va="center", fontsize=8, color="#DCE3DE", zorder=6)
ax.add_patch(FancyBboxPatch((-2.9, -2.98), 5.8, 0.36, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#EDEFEA", ec="#D6D9D2"))
ax.text(0, -2.8, "바탕  ·  수학 (선형대수 · 최적화 · 확률과정)  +  원자재 시장 이해", ha="center", va="center", fontsize=9.5, color=G, fontweight="bold")
save(f, "c46_quant_venn")

# ════════ 47 머신러닝 카드별 3D (예시 자료)
rng = np.random.default_rng(3)
# 47a 변수 고르기: 규제를 세게 할수록 쓸모없는 지표의 계수가 0으로
from sklearn.linear_model import lasso_path
VN = ["달러", "금리", "원유", "재고", "작황", "기상", "포지션", "운임"]
n = 600; Xv = rng.normal(size=(n, len(VN))); beta = np.zeros(len(VN)); beta[[2, 3, 6]] = [0.9, -0.7, 0.5]; beta[[0, 4]] = [0.25, 0.12]
yv = Xv @ beta + rng.normal(0, 1, n)
alphas = np.array([0.005, 0.05, 0.12, 0.22, 0.35])
_, coefs, _ = lasso_path(Xv, yv, alphas=alphas)
f, ax = ax3d()
for j, al in enumerate(alphas):
    cf = np.abs(coefs[:, j]); keep = cf > 1e-6
    ax.bar3d(np.arange(len(VN)) - 0.3, np.full(len(VN), j) - 0.3, 0, 0.6, 0.6, cf, color=[G if k else "#CCCCCC" for k in keep], alpha=0.9, shade=True)
ax.set_xticks(range(len(VN))); ax.set_xticklabels(VN, fontsize=7, rotation=0, ha="right"); ax.set_yticks(range(len(alphas))); ax.set_yticklabels(["약", "", "", "", "강"], fontsize=7)
ax.set_zticks([0, 0.4, 0.8]); lab(ax, "", "걸러내는 강도", "영향력"); ax.view_init(elev=24, azim=-60); ax.set_box_aspect((2.2, 1.2, 0.9))
save(f, "c47a_lasso3d")

# 47b 공통 흐름 요약: 실제 옥수수·대두·밀 주간 수익률 점 구름과 첫 번째 공통 흐름(화살표)
import analytics as A, ls_strategies as LS
adj = A.load_levels(roll_adjusted=True).loc[:"2026-10-26"]
KN3 = {"yf:ZC=F": "옥수수", "yf:ZS=F": "대두", "yf:ZW=F": "밀"}
P3 = LS.weekly(adj, list(KN3), "2007-01-01")[list(KN3)].rename(columns=KN3)
R3 = np.log(P3).diff().dropna(); Z3 = ((R3 - R3.mean()) / R3.std()).clip(-4, 4).values
ev, evec = np.linalg.eigh(np.cov(Z3.T)); pc1 = evec[:, -1] * np.sign(evec[:, -1].sum())
share = ev[-1] / ev.sum()
f, ax = ax3d()
proj = Z3 @ pc1
ax.scatter(Z3[:, 0], Z3[:, 1], Z3[:, 2], c=proj, cmap="viridis", s=3, alpha=0.5)
for sgn in (-1, 1):
    ax.plot(*[[0, sgn * 3.6 * pc1[i]] for i in range(3)], color=RED, lw=2.6)
ax.text(*(3.9 * pc1), f"공통 흐름 1\n({share:.0%} 설명)", color=RED, fontsize=8, fontweight="bold")
lab(ax, "옥수수", "대두", "밀"); ax.set_xticks([-3, 0, 3]); ax.set_yticks([-3, 0, 3]); ax.set_zticks([-3, 0, 3]); ax.view_init(elev=18, azim=-55)
save(f, "c47b_pca3d")
print("PCA 첫 흐름 설명 비율:", round(float(share), 3))

# 47c 판단 나무 숲: 같은 자료를 랜덤 포레스트로 학습한 예측 지형
from sklearn.ensemble import RandomForestRegressor
n = 1500; x1, x2 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
y = 2.0 * (x1 < 0.25) * (x2 > 0.5) - 1.2 * (x1 > 0.8) + 0.8 * np.sin(3 * x2) * x1 + rng.normal(0, 0.35, n)
g = np.linspace(0, 1, 45); GX, GY = np.meshgrid(g, g)
ZR = RandomForestRegressor(n_estimators=300, min_samples_leaf=15, random_state=0, n_jobs=-1).fit(np.c_[x1, x2], y).predict(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)
f, ax = ax3d()
ax.plot_surface(GX, GY, ZR, cmap="viridis", linewidth=0, antialiased=True, vmin=-1.5, vmax=2.2)
lab(ax, "가격 위치", "변동성", "4주 뒤 수익률"); ax.set_zticks([-1, 0, 1, 2]); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1]); ax.view_init(elev=26, azim=-130)
save(f, "c47c_rf3d")

# 47d 순서를 기억하는 신경망: 몇 주 전 정보를 얼마나 쓰는가(기억 가중치 지형, 예시)
lags = np.arange(1, 53); tt = np.arange(0, 60)
LG, TT = np.meshgrid(lags, tt)
calm = 0.5 + 0.5 * np.sin(TT / 60 * 2 * np.pi * 1.3)                       # 출렁임이 큰 시기엔 최근 정보에 더 집중
Wm = np.exp(-(LG - 1) / (3 + 6 * calm)) + 0.45 * np.exp(-((LG - 52) / 2.5) ** 2) + 0.25 * np.exp(-((LG - 26) / 3) ** 2)
f, ax = ax3d()
ax.plot_surface(LG, TT, Wm, cmap="cividis", linewidth=0, antialiased=True)
ax.set_xticks([1, 13, 26, 39, 52]); ax.set_yticks([]); ax.set_zticks([])
lab(ax, "몇 주 전 정보", "예측 시점 →", "기억 비중"); ax.view_init(elev=30, azim=-120)
ax.text(52, 0, 0.62, "1년 전\n(계절)", fontsize=7, color=RED); ax.text(1, 0, 1.12, "최근", fontsize=7, color=RED)
save(f, "c47d_lstm3d")

# ════════ 48 AI·위험·국면 카드별 3D
# 48a 사전학습 AI: 예측은 한 점이 아닌 '분포' — 먼 미래일수록 넓게 퍼짐 (옥수수 주간 출렁임 3.2% 기준 예시)
f, ax = ax3d()
hs = [1, 2, 4, 8, 12]; px = np.linspace(65, 140, 300); cmap = matplotlib.colormaps["Oranges"]
for k, h in enumerate(hs[::-1]):
    sd = 0.032 * np.sqrt(h); mu = np.log(100)
    dens = np.exp(-(np.log(px) - mu) ** 2 / (2 * sd ** 2)) / (px * sd * np.sqrt(2 * np.pi))
    verts = [list(zip(px, np.full_like(px, h), dens)) + [(px[-1], h, 0), (px[0], h, 0)]]
    ax.add_collection3d(Poly3DCollection(verts, facecolor=cmap(0.35 + 0.6 * (1 - k / 4)), edgecolor=RED, lw=0.8, alpha=0.85))
ax.set_xlim(65, 140); ax.set_ylim(0, 13); ax.set_zlim(0, 0.13); ax.set_yticks(hs); ax.set_zticks([])
lab(ax, "가격 (오늘 = 100)", "몇 주 뒤", "가능성"); ax.view_init(elev=22, azim=-70)
save(f, "c48a_chronos3d")

# 48b 문서를 읽는 AI: USDA 수급 보고서(WASDE) 품목 섹션별 FinBERT 어조 점수, 연평균 (실제)
wt = pd.read_parquet(ROOT / "data/mm/wasde_text.parquet")
SEC = {"WHEAT": "밀", "COARSE GRAINS": "옥수수 등", "OILSEEDS": "대두 등"}
wt = wt[wt.section.isin(SEC)]; wt["yr"] = wt.release.dt.year
T = wt[wt.yr >= 2012].groupby(["yr", "section"]).tone.mean().unstack()[list(SEC)]
f, ax = ax3d()
for j, sname in enumerate(SEC):
    v = T[sname].values
    ax.bar3d(T.index.values - 0.35, np.full(len(v), j) - 0.3, np.minimum(v, 0), 0.7, 0.6, np.abs(v), color=[G if t >= 0 else RED for t in v], alpha=0.85, shade=True)
ax.set_yticks(range(len(SEC))); ax.set_yticklabels(list(SEC.values()), fontsize=7); ax.set_xticks([2012, 2016, 2020, 2024])
lab(ax, "", "", "어조 (+ 긍정 / - 부정)"); ax.view_init(elev=22, azim=-62); ax.set_box_aspect((2.0, 1.2, 0.9))
save(f, "c48b_finbert3d")
print("WASDE 어조 범위:", round(float(T.min().min()), 2), round(float(T.max().max()), 2))

# 48c 위험 크기 예측: 품목별 GARCH(1,1) 조건부 변동성 지형 (실제, 주간)
from arch import arch_model
KN = {"yf:ZC=F": "옥수수", "yf:ZW=F": "밀", "yf:ZS=F": "대두", "yf:ZM=F": "대두박", "yf:CL=F": "원유", "yf:HG=F": "구리"}
P = LS.weekly(adj, list(KN), "2005-01-01")[list(KN)].rename(columns=KN)
Rw = np.log(P).diff().dropna() * 100
GV = pd.DataFrame({k: arch_model(Rw[k], vol="GARCH", p=1, q=1, dist="t").fit(disp="off").conditional_volatility * np.sqrt(52) for k in Rw})
GV = GV.loc["2007":].resample("ME").mean().clip(upper=90)
tg = np.array([d.year + (d.month - 0.5) / 12 for d in GV.index]); TTg, YYg = np.meshgrid(tg, np.arange(GV.shape[1]))
f, ax = ax3d()
ax.plot_surface(TTg, YYg, GV.T.values, cmap="YlOrRd", linewidth=0, antialiased=True, rstride=1, cstride=1, vmin=10, vmax=70)
ax.set_yticks(range(GV.shape[1])); ax.set_yticklabels(GV.columns, fontsize=6.5); ax.set_xticks([2010, 2015, 2020, 2025]); ax.set_zticks([20, 50, 80])
lab(ax, "", "", "예측 위험 (연 %)"); ax.set_box_aspect((2.2, 1.3, 0.8)); ax.view_init(elev=30, azim=-55)
save(f, "c48c_garch3d")

# 48d 시장 국면 판별: 원자재 동일 비중 주간 수익률로 2개 국면(평시·위기) 확률 (실제, 숨은 마르코프 모형)
from hmmlearn.hmm import GaussianHMM
ew = Rw.mean(axis=1).loc["2006":]
hm = GaussianHMM(n_components=2, covariance_type="full", n_iter=300, random_state=0).fit(ew.values[:, None])
pr = hm.predict_proba(ew.values[:, None]); crisis = int(np.argmax(hm.covars_.ravel()))
pc = pd.Series(pr[:, crisis], index=ew.index).loc["2007":].resample("ME").mean()
tp = np.array([d.year + (d.month - 0.5) / 12 for d in pc.index])
f, ax = ax3d()
for j, (vals, col, nm) in enumerate([(pc.values, RED, "위기"), (1 - pc.values, BLUE, "평시")]):
    verts = [list(zip(tp, np.full_like(tp, j), vals)) + [(tp[-1], j, 0), (tp[0], j, 0)]]
    ax.add_collection3d(Poly3DCollection(verts, facecolor=col, edgecolor=col, lw=0.6, alpha=0.55))
ax.set_xlim(tp[0], tp[-1]); ax.set_ylim(-0.5, 1.5); ax.set_zlim(0, 1); ax.set_yticks([0, 1]); ax.set_yticklabels(["위기", "평시"], fontsize=7.5)
ax.set_xticks([2008, 2014, 2020, 2026]); ax.set_zticks([0, 0.5, 1]); lab(ax, "", "", "국면 확률"); ax.set_box_aspect((2.3, 1.0, 0.8)); ax.view_init(elev=24, azim=-65)
save(f, "c48d_hmm3d")
print("위기 확률 상위 연도:", pc.groupby(pc.index.year).mean().nlargest(4).round(2).to_dict())

# ════════ 49 효율적 투자선: 연도별 작은 그림(2D)으로 나눠 겹침 제거 + 3D 리본(3년 간격) 보조
W = pd.read_parquet(ROOT / "data/voltgt_weekly.parquet")
BASE = {"기준 · MinVar 롱 (⑥)": "최소분산", "기준 · 52주 모멘텀 (④)": "모멘텀", "V1 가치 (5년 평균 대비 싼 품목 롱)": "가치"}
X3 = pd.DataFrame({v_: W[f"{k}|10|4"] for k, v_ in BASE.items()}).loc["2007":].dropna()
Wr = np.random.default_rng(0).dirichlet(np.full(3, 0.5), 6000); wmix = np.array([0.5, 0.25, 0.25])
years = list(range(2011, 2027)); FR = {}
for yr in years:
    Xw = X3.loc[f"{yr - 4}":f"{yr}"]; mu, S = Xw.mean().values * 52, Xw.cov().values * 52
    vol = np.sqrt(np.einsum("ij,jk,ik->i", Wr, S, Wr)); ret = Wr @ mu
    best = -np.inf; fv, fr = [], []
    for i in np.argsort(vol):
        if ret[i] > best: best = ret[i]; fv.append(vol[i]); fr.append(ret[i])
    k = int(np.argmax(ret / vol))
    FR[yr] = dict(cv=vol[::4], cr=ret[::4], fv=np.array(fv), fr=np.array(fr), mix=(np.sqrt(wmix @ S @ wmix), wmix @ mu), tan=(vol[k], ret[k]))
# 49a 연도별 작은 그림 4×4 — 같은 축 범위
f, axs = plt.subplots(4, 4, figsize=(8.6, 6.4), sharex=True, sharey=True)
for a, yr in zip(axs.ravel(), years):
    d = FR[yr]
    a.scatter(d["cv"], d["cr"], s=1, color="#D8DBD4", rasterized=True)
    a.plot(d["fv"], d["fr"], color=G, lw=1.8)
    a.scatter(*d["tan"], s=26, color=RED, zorder=4, edgecolor="white", lw=0.5)
    a.scatter(*d["mix"], s=90, marker="*", color=GOLD, zorder=5, edgecolor="#333", lw=0.6)
    a.text(0.04, 0.92, str(yr), transform=a.transAxes, fontsize=9, fontweight="bold", color=G, va="top")
    a.grid(color="#EEEEEA", lw=0.5); a.tick_params(labelsize=7)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
f.supxlabel("연 변동성 (%)", fontsize=9, y=0.02); f.supylabel("연 수익 (%)", fontsize=9, x=0.01)
from matplotlib.lines import Line2D
f.legend(handles=[Line2D([], [], color=G, lw=1.8, label="효율적 투자선 (직전 5년)"), Line2D([], [], color="#D8DBD4", marker="o", ls="", ms=3, label="가능한 조합 6,000개"),
                  Line2D([], [], color=RED, marker="o", ls="", ms=5, label="그해 사후 최적 (최고 Sharpe)"), Line2D([], [], color=GOLD, marker="*", ls="", ms=10, mec="#333", label="우리 혼합 (고정)")],
         loc="upper center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 1.02))
f.tight_layout(rect=(0.02, 0.03, 1, 0.97))
save(f, "c49a_frontier_grid")
# 49b 3D 리본: 3년 간격 6개 연도만, 선 아래를 채워 앞뒤가 겹치지 않게
f = plt.figure(figsize=(5.6, 5.0)); ax = f.add_subplot(projection="3d")
for a_ in (ax.xaxis, ax.yaxis, ax.zaxis): a_.pane.set_facecolor((0.97, 0.97, 0.96, 1)); a_.pane.set_edgecolor("#dddddd")
sel = [2011, 2014, 2017, 2020, 2023, 2026]; cm_ = matplotlib.colormaps["viridis"]; zmin = 0
for j, yr in enumerate(sel):
    d = FR[yr]; col = cm_(j / (len(sel) - 1))
    verts = [list(zip(d["fv"], np.full(len(d["fv"]), yr), d["fr"])) + [(d["fv"][-1], yr, zmin), (d["fv"][0], yr, zmin)]]
    ax.add_collection3d(Poly3DCollection(verts, facecolor=col, edgecolor=col, alpha=0.35, lw=1.6))
    ax.plot(d["fv"], np.full(len(d["fv"]), yr), d["fr"], color=col, lw=2.2)
    ax.scatter(d["mix"][0], yr, d["mix"][1], marker="*", s=140, color=GOLD, edgecolor="#333", lw=0.6, depthshade=False, zorder=10)
    ax.scatter(d["tan"][0], yr, d["tan"][1], s=30, color=RED, edgecolor="white", lw=0.5, depthshade=False, zorder=10)
allv = np.concatenate([FR[y]["fv"] for y in sel]); allr = np.concatenate([FR[y]["fr"] for y in sel])
ax.set_xlim(allv.min() - 0.3, allv.max() + 0.3); ax.set_ylim(2009.5, 2027); ax.set_zlim(zmin, allr.max() + 1)
ax.set_yticks(sel); ax.tick_params(labelsize=7.5, pad=0)
ax.set_xlabel("연 변동성 (%)", fontsize=8.5, labelpad=2); ax.set_zlabel("연 수익 (%)", fontsize=8.5, labelpad=0)
ax.set_box_aspect((1.2, 2.0, 0.9)); ax.view_init(elev=24, azim=-68)
save(f, "c49b_frontier_ribbon")

# ════════ 50 시계열 vs 횡단면 3D: 품목 × 월 수익률 막대 + 칸마다 수치 (최근 12개월)
KX = {"yf:ZC=F": "옥수수", "yf:ZW=F": "밀", "yf:ZS=F": "대두", "yf:ZM=F": "대두박", "yf:ZL=F": "대두유", "yf:CL=F": "원유", "yf:NG=F": "천연가스",
      "yf:GC=F": "금", "yf:HG=F": "구리", "yf:KC=F": "커피", "yf:SB=F": "원당", "yf:CT=F": "면화"}
PX = LS.weekly(adj, list(KX), "2020-01-01")[list(KX)].rename(columns=KX)
MX = (np.log(PX).resample("ME").last().diff().loc["2025-11":"2026-10"] * 100)
nm_, ni_ = MX.shape                                                         # 월 수, 품목 수
vmax = float(np.nanpercentile(np.abs(MX.values), 95)); cmr = matplotlib.colormaps["RdYlGn"]
f = plt.figure(figsize=(11, 7.2)); ax = f.add_subplot(projection="3d"); ax.computed_zorder = False   # 그린 순서 = 앞뒤 순서
for a_ in (ax.xaxis, ax.yaxis, ax.zaxis): a_.pane.set_facecolor((0.97, 0.97, 0.96, 1)); a_.pane.set_edgecolor("#dddddd")
HS = 0.4                                                                   # 막대 높이 = 수익률 × HS (±20%에서 자름, 수치는 실제 값)
order = sorted(((j, t) for j in range(ni_) for t in range(nm_)), key=lambda q: (-q[0], q[1]))   # 먼 칸부터 그려 앞 칸이 덮도록
for j, t in order:
    v = float(MX.iat[t, j]); h = np.clip(v, -20, 20) * HS; hl = j == 0 or t == nm_ - 1
    ec = BLUE if j == 0 else (GOLD if t == nm_ - 1 else (0, 0, 0, 0.18))
    ax.bar3d(t - 0.42, j - 0.4, min(h, 0), 0.84, 0.8, max(abs(h), 0.08), color=cmr(0.5 + 0.5 * np.clip(v / vmax, -1, 1)),
             edgecolor=ec, linewidth=1.3 if hl else 0.3, shade=True)
for j, t in order:                                                          # 수치는 막대를 다 그린 뒤 맨 위에
    v = float(MX.iat[t, j]); h = np.clip(v, -20, 20) * HS; hl = j == 0 or t == nm_ - 1
    ax.text(t, j, max(h, 0) + 0.4, "0" if round(v) == 0 else f"{v:+.0f}", fontsize=7.2 if hl else 6.4, ha="center", va="bottom",
            color=(BLUE if j == 0 else "#7a5d10") if hl else "#1a1a1a", fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.55, pad=0.4))
ax.set_xticks(range(nm_)); ax.set_xticklabels([d.strftime("%y.%m") for d in MX.index], fontsize=7.5, rotation=0)
ax.set_yticks(range(ni_)); ax.set_yticklabels(MX.columns, fontsize=8.5)
ax.set_zlabel("월 수익률 (%)", fontsize=9, labelpad=2); ax.tick_params(axis="z", labelsize=7.5)
ax.set_zlim(-20 * HS, 20 * HS); ax.set_zticks([-10 * HS, 0, 10 * HS]); ax.set_zticklabels(["-10", "0", "+10"])
ax.set_box_aspect((2.3, 2.0, 0.5)); ax.view_init(elev=50, azim=-60)
ax.set_title("품목 × 월 수익률 (%, 칸 위 숫자)  ·  파랑 = 시계열(옥수수 12개월)  ·  금색 = 횡단면(2026년 10월)", loc="left", fontsize=11.5, fontweight="bold", color=G)
save(f, "c50_xs_ts3d", pad=0.02)

# ════════ 51 학습 vs 시험 성적 3D 막대 (개념 예시): 선형 · 머신러닝 단독 · 통계 위 머신러닝 접목
OVF = [("통계·계량\n(선형)", 60, 56), ("머신러닝\n단독", 90, 40), ("통계 +\n머신러닝", 72, 63)]
f = plt.figure(figsize=(6.4, 4.4)); ax = f.add_subplot(projection="3d"); ax.computed_zorder = False
for a_ in (ax.xaxis, ax.yaxis, ax.zaxis): a_.pane.set_facecolor((0.97, 0.97, 0.96, 1)); a_.pane.set_edgecolor("#dddddd")
for j in (1, 0):                                                   # 뒤(학습, y=1)부터 그려 앞(시험, y=0)이 덮도록
    for i, (nm, tr, te) in enumerate(OVF):
        v = (te, tr)[j]; col = ((RED if i == 1 else G), "#8DB7CF")[j]
        ax.bar3d(i - 0.28, j - 0.28, 0, 0.56, 0.56, v, color=col, alpha=0.95, shade=True, edgecolor="white", linewidth=0.4)
for j in (1, 0):
    for i, (nm, tr, te) in enumerate(OVF):
        v = (te, tr)[j]; ax.text(i, j, v + 5, f"{v}", ha="center", fontsize=10.5, fontweight="bold", color=(INK, BLUE)[j],
                                 bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))
ax.set_xticks(range(3)); ax.set_xticklabels([o[0] for o in OVF], fontsize=8.5)
ax.set_yticks([0, 1]); ax.set_yticklabels(["시험 성적\n(처음 본 문제)", "학습 성적\n(외운 문제)"], fontsize=8.5)
ax.set_zlim(0, 100); ax.set_zticks([0, 50, 100]); ax.set_zlabel("맞힌 정도 (예시)", fontsize=8.5)
ax.set_box_aspect((1.6, 1.0, 0.9)); ax.view_init(elev=24, azim=-62)
save(f, "c51_ml_overfit3d")

# ════════ 52·53·54 검증 방법 장: 섞기 vs 시간 분리 3D 블록 · 선형 평면 vs 머신러닝 곡면 (html_v4.py와 같은 난수)
SPLIT_MIX = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
f = plt.figure(figsize=(4.4, 4.2)); ax = f.add_subplot(projection="3d"); ax.computed_zorder = False
for a_ in (ax.xaxis, ax.yaxis, ax.zaxis): a_.pane.set_facecolor((0.97, 0.97, 0.96, 1)); a_.pane.set_edgecolor("#dddddd")
TRC, TEC = "#3B8EA5", "#F2A541"
for row, seq in ((1, [0] * 10 + [1] * 4), (0, SPLIT_MIX)):
    for i, m in enumerate(seq):
        ax.bar3d(i, row - 0.32, 0, 0.9, 0.64, 0.5, color=TEC if m else TRC, shade=True, edgecolor="white", linewidth=0.4)
for row, auc, col in ((1, 0.50, "#2E7D52"), (0, 0.69, RED)):
    ax.bar3d(15.2, row - 0.32, 0, 0.9, 0.64, auc * 4, color=col, shade=True, edgecolor="white", linewidth=0.4)
    ax.text(15.6 + (1.6 if row else -0.2), row, auc * 4 + 0.25, f"AUC {auc:.2f}", ha="left" if row else "right", fontsize=9, fontweight="bold", color=col)
ax.set_yticks([0, 1]); ax.set_yticklabels(["무작위로 섞기", "시간 순 분리"], fontsize=8.5); ax.set_xticks([]); ax.set_zticks([])
ax.set_xlabel("시간 →", fontsize=8.5, labelpad=-8); ax.set_box_aspect((2.2, 1.1, 0.9)); ax.view_init(elev=28, azim=-62)
save(f, "c52_split3d")
rs = np.random.default_rng(11)
x1, x2 = rs.uniform(-2, 2, 120), rs.uniform(-2, 2, 120); yl = 0.8 * x1 + 0.5 * x2 + rs.normal(0, 0.35, 120)
bl = np.linalg.lstsq(np.c_[np.ones(120), x1, x2], yl, rcond=None)[0]
gx = np.linspace(-2, 2, 20); GX, GY = np.meshgrid(gx, gx)
f, ax = ax3d((4.4, 3.6)); ax.plot_surface(GX, GY, bl[0] + bl[1] * GX + bl[2] * GY, color=BLUE, alpha=0.55, linewidth=0)
ax.scatter(x1, x2, yl, c=G, s=0.9, depthshade=False); lab(ax, "설명변수 1", "설명변수 2", "수익률"); ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2]); ax.view_init(elev=18, azim=-60)
save(f, "c53_linfit3d")
from sklearn.kernel_ridge import KernelRidge
rs = np.random.default_rng(12)
z1, z2 = rs.uniform(-3, 3, 260), rs.uniform(-3, 3, 260)
yn = 1.5 * np.sin(1.3 * z1) + 0.8 * np.cos(1.1 * z2) + 0.15 * z1 * z2 + rs.normal(0, 0.35, 260)
kr = KernelRidge(kernel="rbf", alpha=0.3, gamma=0.5).fit(np.c_[z1, z2], yn)
gz = np.linspace(-3, 3, 30); ZX, ZY = np.meshgrid(gz, gz); ZK = kr.predict(np.c_[ZX.ravel(), ZY.ravel()]).reshape(ZX.shape)
bn = np.linalg.lstsq(np.c_[np.ones(260), z1, z2], yn, rcond=None)[0]
f, ax = ax3d((4.4, 3.6)); ax.plot_surface(ZX, ZY, ZK, cmap="viridis", alpha=0.75, linewidth=0)
ax.plot_surface(ZX, ZY, bn[0] + bn[1] * ZX + bn[2] * ZY, color="#999999", alpha=0.25, linewidth=0)
ax.scatter(z1, z2, yn, c=RED, s=0.7, depthshade=False); lab(ax, "설명변수 1", "설명변수 2", "수익률"); ax.view_init(elev=22, azim=-55)
save(f, "c54_mlfit3d")

# ════════ 55·56·57 3대 전략 직관: 추세가 이어짐 · 출렁임 그릇의 바닥(달걀) · 평균으로 당겨지는 고무줄 (html_v4_app.js와 같은 식)
def mom_paths():
    t = np.arange(79); S = [1.0, 0.55, -0.5, -0.95]
    return t, [s_ * t * 0.4 + 3 * np.sin(t / 6 + k) for k, s_ in enumerate(S)], S
t, Z, S = mom_paths(); f, ax = ax3d((4.6, 2.2))
for k, (z, s_) in enumerate(zip(Z, S)):
    col = "#2E7D52" if s_ > 0 else RED
    ax.plot(t[:53], [k] * 53, z[:53], color=col, lw=1.8); ax.plot(t[52:], [k] * 27, z[52:], color=col, lw=1.4, ls="--", alpha=0.6)
ax.set_xticks([0, 52]); ax.set_xticklabels(["1년 전", "오늘"], fontsize=7); ax.set_yticks([]); ax.set_zticks([0]); ax.view_init(elev=22, azim=-70); ax.set_box_aspect((2.4, 1, 0.8))
save(f, "c55_mom3d")
def mv_grid(n=41):
    vol = np.array([25, 20, 30.0]); cor = np.array([[1, .2, .1], [.2, 1, .3], [.1, .3, 1]]); cov = np.outer(vol, vol) * cor
    g = np.linspace(0, 1, n); W1, W2 = np.meshgrid(g, g); W3 = 1 - W1 - W2
    V = np.sqrt(np.einsum("...i,ij,...j->...", np.stack([W1, W2, W3], -1), cov, np.stack([W1, W2, W3], -1))); V[W3 < -1e-9] = np.nan
    return g, V
g, V = mv_grid(); f, ax = ax3d((4.6, 2.2)); GA, GB = np.meshgrid(g, g)
ax.plot_surface(GA, GB, np.ma.masked_invalid(V), cmap="YlGnBu_r", alpha=0.85, linewidth=0)
i_ = np.nanargmin(V); ax.scatter([GA.ravel()[i_]], [GB.ravel()[i_]], [V.ravel()[i_] + 0.8], s=60, c="#F3E3C3", edgecolors="#8a6d2b")
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.view_init(elev=28, azim=-130)
save(f, "c56_minvar3d")
def val_path():
    t = np.arange(121); return t, 18 * np.sin(t / 9) + 6 * np.sin(t / 3.7)
t, dv = val_path(); f, ax = ax3d((4.6, 2.2))
ax.plot(t, [0] * len(t), dv, color=INK, lw=1.4); ax.plot(t, [0] * len(t), 0 * t, color=GOLD, lw=1.6)
for q in range(0, 121, 4):
    if abs(dv[q]) > 8: ax.plot([q, q], [0, 0], [0, dv[q]], color=(RED if dv[q] > 0 else "#2E7D52"), lw=1.4)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([0]); ax.view_init(elev=18, azim=-75); ax.set_box_aspect((2.4, 0.6, 0.9))
save(f, "c57_value3d")
