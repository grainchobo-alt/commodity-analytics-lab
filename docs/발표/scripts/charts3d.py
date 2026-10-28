"""3D 그림 2종: 품목 쌍 상관 지형(52주 롤링), 3D 효율적 투자선."""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(r"C:\Users\user\Desktop\Commodity"); sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10, "savefig.dpi": 200})
OUT = Path(__file__).resolve().parent / "img"
import analytics as A, ls_strategies as LS
adj = A.load_levels(roll_adjusted=True).loc[:"2026-10-26"]
KN = {"yf:ZC=F": "옥수수", "yf:ZS=F": "대두", "yf:ZW=F": "밀", "yf:CL=F": "원유", "yf:GC=F": "금", "yf:HG=F": "구리"}
P = LS.weekly(adj, list(KN), "2005-01-01").rename(columns=KN)
R = np.log(P).diff().dropna()
pairs = [("옥수수", "대두"), ("옥수수", "밀"), ("대두", "원유"), ("원유", "구리"), ("구리", "금"), ("옥수수", "원유")]
Z = np.array([R[a].rolling(52).corr(R[b]).values for a, b in pairs])
idx = R.index; ok = ~np.isnan(Z).any(0); Z, idx = Z[:, ok], idx[ok]
t = np.array([d.year + d.dayofyear / 366 for d in idx])
f = plt.figure(figsize=(10, 5.6)); ax = f.add_subplot(projection="3d")
TT, YY = np.meshgrid(t, np.arange(len(pairs)))
s = ax.plot_surface(TT, YY, Z, cmap="RdBu_r", vmin=-0.6, vmax=0.9, rstride=1, cstride=2, linewidth=0, antialiased=True, alpha=0.95)
ax.set_yticks(range(len(pairs))); ax.set_yticklabels([f"{a}–{b}" for a, b in pairs], fontsize=8)
ax.set_zticks([-0.5, 0, 0.5]); ax.set_zlabel(""); ax.set_box_aspect((2.6, 1.0, 0.6)); ax.view_init(elev=30, azim=-58); ax.tick_params(axis="y", pad=-2)
for y0, y1 in [(2008.7, 2009.5), (2020.15, 2020.5), (2022.15, 2022.7)]:
    ax.plot([y0, y0], [0, 0], [-0.6, 0.9], color="#C8A24A", lw=1)
f.colorbar(s, shrink=0.45, pad=0.0, orientation="horizontal", location="bottom", label="52주 상관 (빨강 = 함께 움직임)"); ax.set_title("품목 쌍별 상관 지형 — 위기 때 함께 솟음 (52주 롤링, 주간)", loc="left", fontweight="bold", color="#1E3A2F")
f.savefig(OUT / "c27_corr3d.png", bbox_inches="tight", facecolor="white"); plt.close(f); print("saved c27_corr3d")

W = pd.read_parquet(ROOT / "data/voltgt_weekly.parquet")
BASE = {"기준 · MinVar 롱 (⑥)": "최소분산", "기준 · 52주 모멘텀 (④)": "52주 모멘텀", "V1 가치 (5년 평균 대비 싼 품목 롱)": "가치"}
X = pd.DataFrame({v_: W[f"{k}|10|4"] for k, v_ in BASE.items()}).loc["2007":].dropna()
mu, S = X.mean().values * 52, X.cov().values * 52
Wr = np.random.default_rng(0).dirichlet(np.full(3, 0.6), 4000)
vol, ret = np.sqrt(np.einsum("ij,jk,ik->i", Wr, S, Wr)), Wr @ mu; sh = ret / vol
f = plt.figure(figsize=(7.5, 5.2)); ax = f.add_subplot(projection="3d")
p = ax.scatter(vol, ret, sh, c=sh, cmap="viridis", s=3, alpha=0.7)
wv = np.array([0.5, 0.25, 0.25]); v0, r0 = np.sqrt(wv @ S @ wv), wv @ mu
ax.scatter([v0], [r0], [r0 / v0], marker="*", s=300, color="#C8A24A", edgecolor="#333")
ax.text(v0, r0, r0 / v0 + 0.06, "가치 혼합", fontsize=9)
ax.set_xlabel("연 변동성 (%)"); ax.set_ylabel("연 수익 (%)"); ax.set_zlabel("Sharpe"); ax.view_init(elev=24, azim=-128)
f.colorbar(p, shrink=0.55, pad=0.08, label="Sharpe"); ax.set_title("3D 효율적 투자선 — 높이 = 위험 대비 수익", loc="left", fontweight="bold", color="#1E3A2F")
f.savefig(OUT / "c28_frontier3d.png", bbox_inches="tight", facecolor="white"); plt.close(f); print("saved c28_frontier3d")
