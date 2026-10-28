"""발표 자료용 차트 PNG 생성 (실제 데이터 + 개념 설명용 합성 예시). 출력: deck/img/*.png"""
import sys, warnings, json
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(r"C:\Users\user\Desktop\Commodity")
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

OUT = Path(__file__).resolve().parent / "img"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9a9a94",
                     "axes.labelcolor": "#3a3a36", "xtick.color": "#55554f", "ytick.color": "#55554f",
                     "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlecolor": "#1E3A2F", "savefig.dpi": 200})
G, GOLD, RED, BLUE, GREY, LG = "#1E3A2F", "#C8A24A", "#D9582B", "#2F6E8F", "#8A8A84", "#E9E6DC"
nums = {}


def save(f, name):
    f.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(f)
    print("saved", name)


import analytics as A, ls_strategies as LS
adj = A.load_levels(roll_adjusted=True).loc[:"2026-10-26"]
raw = A.load_levels(roll_adjusted=False).loc[:"2026-10-26"]

# ── 07 시계열: 옥수수 가격 → 주간 수익률
c = raw["yf:ZC=F"].dropna().loc["2006":]
w = np.log(adj["yf:ZC=F"].dropna()).resample("W-FRI").last().diff().dropna().loc["2006":] * 100
f, ax = plt.subplots(2, 1, figsize=(9, 4.8), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1], hspace=0.12))
ax[0].plot(c.index, c.values, color=G, lw=1.2); ax[0].set_ylabel("¢/bu"); ax[0].set_title("옥수수 근월물 가격 (일간)", loc="left")
ax[1].bar(w.index, w.values, width=6, color=np.where(w.values >= 0, G, RED)); ax[1].set_ylabel("%"); ax[1].set_title("주간 수익률 = 이번 주 가격 / 지난주 가격 - 1", loc="left")
save(f, "c07_timeseries")

# ── 09 변동성: 수익률 + 26주 실현 변동성
rv = w.rolling(26).std() * np.sqrt(52)
f, ax = plt.subplots(figsize=(9, 3.6))
ax.bar(w.index, w.values, width=6, color=GREY, alpha=0.55, label="주간 수익률 (%)")
ax2 = ax.twinx(); ax2.plot(rv.index, rv.values, color=RED, lw=2, label="연 변동성 (26주, %)"); ax2.spines["top"].set_visible(False)
ax2.set_ylabel("연 변동성 %", color=RED); ax.set_ylabel("주간 수익률 %")
for a, b, lab in [("2008-06", "2009-03", "금융위기"), ("2012-06", "2012-10", "미국 가뭄"), ("2020-03", "2020-06", "코로나"), ("2022-02", "2022-07", "우크라이나")]:
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=GOLD, alpha=0.18); ax.text(pd.Timestamp(a), ax.get_ylim()[1] * 0.92, lab, fontsize=9, color="#7a5d10")
ax.set_title("옥수수: 큰 움직임은 몰려서 옴 (변동성 군집)", loc="left")
save(f, "c09_volatility")
nums["corn_vol_now"] = float(rv.iloc[-1]); nums["corn_vol_max"] = float(rv.max())

# ── 10 GARCH 뉴스 충격 곡선 (합성)
z = np.linspace(-3, 3, 200)
f, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(z, np.sqrt(0.05 + 0.08 * z ** 2 + 0.9), color=G, lw=2.5, label="GARCH (대칭)")
ax.plot(z, np.sqrt(0.05 + (0.04 + 0.12 * (z < 0)) * z ** 2 + 0.9), color=RED, lw=2.5, label="GJR-GARCH (하락 충격이 더 큼)")
ax.set_xlabel("어제의 뜻밖의 가격 변화 (충격)"); ax.set_ylabel("오늘의 예상 변동성"); ax.legend(frameon=False, fontsize=9)
ax.set_title("뉴스 충격 곡선", loc="left")
save(f, "c10_newsimpact")

# ── 19개 품목 주간 수익률 (상관·HMM·PCA)
P = LS.weekly(adj, LS.BASKET, "2001-01-01"); P = P[[c_ for c_ in LS.BASKET if c_ in P]]
R = np.log(P).diff().dropna(how="all") * 100
KN = {"yf:ZC=F": "옥수수", "yf:ZW=F": "밀", "yf:ZS=F": "대두", "yf:ZM=F": "대두박", "yf:ZL=F": "대두유", "yf:CL=F": "원유", "yf:NG=F": "천연가스",
      "yf:RB=F": "휘발유", "yf:HO=F": "난방유", "yf:GC=F": "금", "yf:SI=F": "은", "yf:HG=F": "구리", "yf:PL=F": "백금", "yf:KC=F": "커피",
      "yf:SB=F": "원당", "yf:CC=F": "코코아", "yf:CT=F": "면화", "yf:LE=F": "생우", "yf:HE=F": "돈육"}
Rn = R.rename(columns=KN)

# ── 26 상관 히트맵
C = Rn.loc["2007":].corr()
order = [k for k in ["옥수수", "밀", "대두", "대두박", "대두유", "원유", "휘발유", "난방유", "천연가스", "금", "은", "백금", "구리", "커피", "원당", "코코아", "면화", "생우", "돈육"] if k in C]
C = C.loc[order, order]
f, ax = plt.subplots(figsize=(7.2, 6.2))
im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(C))); ax.set_xticklabels(C.columns, rotation=90, fontsize=9); ax.set_yticks(range(len(C))); ax.set_yticklabels(C.index, fontsize=9)
for i in range(len(C)):
    for j in range(len(C)):
        if i != j and abs(C.values[i, j]) >= 0.4:
            ax.text(j, i, f"{C.values[i, j]:.1f}", ha="center", va="center", fontsize=7, color="white")
plt.colorbar(im, fraction=0.04); ax.set_title("19개 품목 주간 수익률 상관 (2007–2026)", loc="left")
for s in ax.spines.values(): s.set_visible(False)
save(f, "c26_corr")

# ── 12 HMM: 바스켓 평균 수익률로 2국면 (필터 확률 근사: 사후 확률 사용은 설명용)
from hmmlearn.hmm import GaussianHMM
bk = Rn.loc["2003":].mean(1).dropna()
m = GaussianHMM(2, "full", n_iter=300, random_state=1).fit(bk.values[:, None])
pp = m.predict_proba(bk.values[:, None]); hi = int(np.argmax(m.covars_.ravel()))
pr = pd.Series(pp[:, hi], bk.index).rolling(4).mean()
f, ax = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1], hspace=0.35))
ax[0].plot(bk.index, bk.cumsum(), color=G, lw=1.3); ax[0].set_title("19개 원자재 동일 비중 바스켓 누적 수익 (%)", loc="left")
ax[1].fill_between(pr.index, 0, pr.values, color=RED, alpha=0.75); ax[1].set_ylim(0, 1); ax[1].set_title("HMM이 추정한 '고변동(위기) 국면' 확률", loc="left")
for a_ in ax:
    for a, b in [("2008-09", "2009-06"), ("2020-02", "2020-06"), ("2022-02", "2022-09")]:
        a_.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=GOLD, alpha=0.15)
save(f, "c12_hmm")
nums["hmm_vol_ratio"] = float(np.sqrt(m.covars_.ravel()[hi] / m.covars_.ravel()[1 - hi]))

# ── 13 Kalman (합성, 도감 로직)
rng = np.random.default_rng(4); T = 400
beta = 0.5 + np.cumsum(0.01 * rng.standard_normal(T)); beta[200:] += 1.0
x = rng.standard_normal(T); y = beta * x + 0.5 * rng.standard_normal(T)
b_, P_, q, r_ = 0.0, 1.0, 1e-3, 0.25; est = []
for t in range(T):
    P_ += q; K = P_ * x[t] / (x[t] ** 2 * P_ + r_); b_ += K * (y[t] - x[t] * b_); P_ *= (1 - K * x[t]); est.append(b_)
roll = pd.Series(x * y).rolling(60).sum() / pd.Series(x * x).rolling(60).sum()
f, ax = plt.subplots(figsize=(8, 3.4))
ax.plot(beta, color="#222", lw=2, label="실제 관계 β (위기 때 급변)"); ax.plot(est, color=G, lw=2.5, label="칼만 필터 추정")
ax.plot(roll, color=GOLD, lw=1.8, ls="--", label="60기 롤링 회귀"); ax.axvline(200, color=RED, ls=":", lw=1)
ax.text(203, ax.get_ylim()[0] + 0.1, "급변 시점", color=RED, fontsize=9); ax.legend(frameon=False, fontsize=9, loc="upper left"); ax.set_xlabel("시점")
ax.set_title("관계가 바뀌면 칼만은 빨리, 롤링 회귀는 늦게 따라감", loc="left")
save(f, "c13_kalman")

# ── 14 VaR: 옥수수 일간 손실 분포 + 정규 vs 실제 99%
d = np.log(adj["yf:ZC=F"].dropna()).diff().dropna().loc["2007":] * 100
q_emp, q_norm = np.quantile(d, 0.01), d.mean() - 2.326 * d.std()
f, ax = plt.subplots(figsize=(6.6, 3.6))
ax.hist(d, bins=160, color=GREY, alpha=0.7, density=True)
xx = np.linspace(-8, 8, 400); ax.plot(xx, np.exp(-0.5 * ((xx - d.mean()) / d.std()) ** 2) / (d.std() * np.sqrt(2 * np.pi)), color=BLUE, lw=1.8, label="정규분포 가정")
ax.axvline(q_norm, color=BLUE, ls="--", lw=1.5); ax.axvline(q_emp, color=RED, lw=2)
ax.text(q_norm + 0.15, ax.get_ylim()[1] * 0.7, f"정규 99% VaR\n{q_norm:.1f}%", color=BLUE, fontsize=9)
ax.text(q_emp - 3.2, ax.get_ylim()[1] * 0.45, f"실제 99% 손실\n{q_emp:.1f}%", color=RED, fontsize=9)
ax.set_xlim(-8, 8); ax.set_xlabel("옥수수 일간 수익률 (%)"); ax.legend(frameon=False, fontsize=9); ax.set_title("꼬리는 정규분포보다 두꺼움", loc="left")
save(f, "c14_var")
rm = pd.read_csv(ROOT / "data/risk_models_compare.csv", encoding="utf-8-sig")
v = rm[(rm["구분"] == "VaR") & (rm["기준"] == "1일 99%") & (rm["포지션"].str.startswith("보유"))].groupby("항목")["초과율"].mean() * 100
nums["var_exceed"] = {k: round(float(x_), 2) for k, x_ in v.items()}

# ── 15 효율적 투자선: 기존 3종 무작위 조합
W = pd.read_parquet(ROOT / "data/voltgt_weekly.parquet")
BASE = {"기준 · MinVar 롱 (⑥)": "최소분산", "기준 · 52주 모멘텀 (④)": "52주 모멘텀", "V1 가치 (5년 평균 대비 싼 품목 롱)": "가치"}
X = pd.DataFrame({v_: W[f"{k}|10|4"] for k, v_ in BASE.items()}).loc["2007":].dropna()
mu, S = X.mean().values * 52, X.cov().values * 52
Wr = np.random.default_rng(0).dirichlet(np.full(3, 0.6), 5000)
vol, ret = np.sqrt(np.einsum("ij,jk,ik->i", Wr, S, Wr)), Wr @ mu
f, ax = plt.subplots(figsize=(7, 4.4))
sc = ax.scatter(vol, ret, c=ret / vol, cmap="viridis", s=5, alpha=0.6)
for j, n in enumerate(X.columns):
    ax.scatter(np.sqrt(S[j, j]), mu[j], s=90, color=RED, zorder=3); ax.annotate(n, (np.sqrt(S[j, j]), mu[j]), xytext=(-10, 6), textcoords="offset points", fontsize=10, ha="right")
wv = np.array([0.5, 0.25, 0.25]); ax.scatter(np.sqrt(wv @ S @ wv), wv @ mu, marker="*", s=320, color=GOLD, edgecolor="#333", zorder=4)
ax.annotate("가치 혼합 50/25/25", (np.sqrt(wv @ S @ wv), wv @ mu), xytext=(8, -14), textcoords="offset points", fontsize=10)
plt.colorbar(sc, label="Sharpe"); ax.set_xlabel("연 변동성 (%)"); ax.set_ylabel("연 수익 (%)")
ax.set_title("세 전략을 섞은 5,000가지 조합 — 섞으면 위험이 줄어듦", loc="left")
save(f, "c15_frontier")

# ── 16 전략 누적 수익
f, ax = plt.subplots(figsize=(8.4, 3.8))
cols = [G, BLUE, GOLD]
for j, n in enumerate(X.columns):
    ax.plot(X.index, X[n].cumsum(), color=cols[j], lw=1.6, label=n)
mix = W["혼합 · MinVar 50 + 모멘텀 25 + 가치 25 (사후 선택 주의)|10|4"].loc["2007":]
ax.plot(mix.index, mix.cumsum(), color=RED, lw=2.6, label="가치 혼합 (목표 변동성 10%)")
ax.legend(frameon=False, fontsize=9, ncol=2); ax.set_ylabel("누적 수익 (%, 합산)"); ax.set_title("2007–2026 누적 수익 (각 목표 변동성 10%, 거래비용 차감)", loc="left")
save(f, "c16_strategies")
sh = lambda r: float(r.mean() / r.std() * np.sqrt(52))
nums["sharpe"] = {n: round(sh(X[n]), 2) for n in X.columns}; nums["sharpe"]["가치 혼합"] = round(sh(mix), 2)

# ── 17 PCA 설명 비율 + Lasso 경로 (실제 상관에서)
ev = np.sort(np.linalg.eigvalsh(C.values))[::-1]
f, ax = plt.subplots(figsize=(4.6, 3.2))
ax.bar(range(1, 11), ev[:10] / ev.sum() * 100, color=[G] + [GREY] * 9); ax.set_xlabel("주성분 번호"); ax.set_ylabel("설명 비율 (%)")
ax.set_title("PCA: 19개 품목 움직임의 공통 요인", loc="left")
save(f, "c17_pca")
nums["pc1"] = float(ev[0] / ev.sum() * 100)
from sklearn.linear_model import lasso_path
yy = Rn["옥수수"].loc["2007":]; XX = Rn.drop(columns=["옥수수"]).loc[yy.index].fillna(0)
XX = (XX - XX.mean()) / XX.std(); alphas, coefs, _ = lasso_path(XX.values, (yy - yy.mean()).values, n_alphas=60)
f, ax = plt.subplots(figsize=(4.6, 3.2))
top = np.argsort(-np.abs(coefs[:, -1]))[:5]
for j in range(coefs.shape[0]):
    ax.plot(np.log10(alphas), coefs[j], color=(G if j in top else LG), lw=(2 if j in top else 1))
for j in top[:3]:
    ax.text(np.log10(alphas[-1]) - 0.05, coefs[j, -1], XX.columns[j], fontsize=8, ha="right")
ax.invert_xaxis(); ax.set_xlabel("규제 강도 (오른쪽일수록 약함)"); ax.set_ylabel("계수"); ax.set_title("LASSO: 강할수록 많은 변수가 0", loc="left")
save(f, "c17_lasso")

# ── 18 예측 부채꼴 (합성 예시)
rng = np.random.default_rng(7); hist = 100 * np.exp(np.cumsum(0.012 * rng.standard_normal(120)))
h = np.arange(1, 21); last = hist[-1]; sd = 0.012 * np.sqrt(h)
f, ax = plt.subplots(figsize=(6.4, 3.2))
ax.plot(range(120), hist, color=GREY, lw=1.4, label="과거 (모델 입력)")
for qz, al in [(1.28, 0.18), (0.52, 0.32)]:
    ax.fill_between(119 + h, last * np.exp(-qz * sd), last * np.exp(qz * sd), color=G, alpha=al, lw=0)
ax.plot(119 + h, np.full(20, last), color=G, lw=2, label="중앙값 예측"); ax.fill_between([], [], [], color=G, alpha=0.25, label="10–90% / 30–70% 범위")
ax.legend(frameon=False, fontsize=9, loc="lower left"); ax.set_title("파운데이션 모델 출력 예: 한 점이 아니라 범위(분위)로 예측", loc="left"); ax.set_xticks([])
save(f, "c18_fan")

# ── 24 구매 AUC: 학습 최고 조합을 시험에 그대로 (곡물 4종 + 근거 품목)
D = pd.read_csv(ROOT / "data/procurement_defer_compare.csv", encoding="utf-8-sig")
D = D[D["구분"] == "백테스트"].dropna(subset=["하락AUC"]) if D["하락AUC"].notna().sum() else D
E = pd.read_csv(ROOT / "data/procurement_defer_compare.csv", encoding="utf-8-sig"); E = E[E["하락AUC"].notna()]
rows = []
for it, g in E.groupby("품목"):
    tr = g[g["기간"] == "학습"]; te = g[g["기간"] == "시험"]
    if not len(tr) or not len(te):
        continue
    best = tr.loc[tr["하락AUC"].idxmax()]
    m_ = te[(te["미룸주"] == best["미룸주"]) & (te["계절특성"] == best["계절특성"])]
    rows.append({"품목": it, "학습": float(best["하락AUC"]), "시험": float(m_["하락AUC"].iloc[0]) if len(m_) else np.nan, "미룸주": int(best["미룸주"])})
AU = pd.DataFrame(rows).set_index("품목").sort_values("학습", ascending=False)
nums["auc"] = AU.round(3).to_dict(orient="index")
(OUT.parent / "nums.json").write_text(json.dumps(nums, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(nums, ensure_ascii=False)[:3000])
