"""모델 도감 학습 노트 그림: 모델마다 개념 그림(합성 데이터) + LSTM·GNN 심화 그림 + 대시보드 실제 결과 그림.

사용: .venv\\Scripts\\python.exe docs\\학습\\scripts\\figs.py   → docs\\학습\\scripts\\img\\*.png
"""
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 9, "savefig.dpi": 180,
                     "mathtext.fontset": "cm", "axes.spines.top": False, "axes.spines.right": False})
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "img"; OUT.mkdir(exist_ok=True)
G, GOLD, RED, BLUE, OCHRE, INK, MUTED, GREY, TEAL = "#1E3A2F", "#C8A24A", "#D9582B", "#2F6E8F", "#B8901C", "#222222", "#6B6B66", "#A8A8A2", "#20808D"
PAL = [BLUE, RED, G, GOLD, "#7B5EA7", TEAL, "#8C564B"]
FIGS = {}
FACTS = {}                                             # 그림을 그리며 계산한 숫자 → facts.json (본문에서 사용)


def fig(name):
    def deco(fn):
        FIGS[name] = fn
        return fn
    return deco


def save(f, name):
    f.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.06, facecolor="white"); plt.close(f)


def rng(s=0):
    return np.random.default_rng(s)


def box(ax, x, y, w, h, text, fc, ec=None, fs=9, tc=INK, r=0.02, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.005,rounding_size={r}", fc=fc, ec=ec or fc, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal")


def arrow(ax, a, b, col=MUTED, lw=1.4, ms=10, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms, color=col, lw=lw, linestyle=ls))


# ════════════════════════ 0. 기초
@fig("basic_price_return")
def _():
    r_ = rng(1); t = np.arange(520); ret = 0.03 * r_.standard_t(4, 520) / 1.4; ret[300:340] *= 2.6
    p = 400 * np.exp(np.cumsum(ret))
    f, (a, b) = plt.subplots(2, 1, figsize=(7, 3.6), sharex=True)
    a.plot(t, p, color=BLUE, lw=1.2); a.set_ylabel("가격 (센트/부셸)"); a.set_title("가격: 추세가 있고 수준이 계속 변함 (비정상)", fontsize=9, loc="left")
    b.bar(t, ret * 100, color=np.where(ret > 0, G, RED), width=1); b.set_ylabel("주간 수익률 (%)"); b.set_xlabel("주")
    b.set_title("수익률: 0 근처에서 오르내림 — 대신 큰 움직임이 몰려 나타남 (변동성 군집)", fontsize=9, loc="left")
    b.axvspan(300, 340, color=RED, alpha=0.08); return f


@fig("basic_overfit")
def _():
    k = np.arange(1, 21); tr = 1.0 * np.exp(-k / 5) + 0.15; te = 0.55 * np.exp(-k / 3) + 0.38 + 0.012 * (k - 4).clip(0) ** 1.35
    f, ax = plt.subplots(figsize=(6.2, 2.9)); ax.plot(k, tr, color=BLUE, lw=2, label="학습 오차 (외운 문제)"); ax.plot(k, te, color=RED, lw=2, label="시험 오차 (처음 본 문제)")
    kb = k[np.argmin(te)]; ax.axvline(kb, color=MUTED, ls="--", lw=1); ax.text(kb + 0.3, te.max() * 0.95, "최적 복잡도", color=MUTED)
    ax.annotate("과적합 구간:\n학습은 좋아지는데\n시험은 나빠짐", (16, te[15]), (11.5, 1.0), arrowprops=dict(arrowstyle="->", color=MUTED), fontsize=8.5)
    ax.set_xlabel("모델 복잡도 (변수 수 · 나무 깊이 · 신경망 크기)"); ax.set_ylabel("오차"); ax.legend(frameon=False); return f


@fig("basic_gd")
def _():
    b1, b2 = np.meshgrid(np.linspace(-1, 3, 200), np.linspace(-1, 3, 200)); L = (b1 - 1.6) ** 2 + 2.5 * (b2 - 1.2) ** 2 + 0.8 * (b1 - 1.6) * (b2 - 1.2)
    f, ax = plt.subplots(figsize=(4.8, 3.6)); ax.contour(b1, b2, L, 14, cmap="Greys", linewidths=0.7)
    p = np.array([-0.6, 2.7]); path = [p.copy()]
    for _ in range(14):
        g = np.array([2 * (p[0] - 1.6) + 0.8 * (p[1] - 1.2), 5 * (p[1] - 1.2) + 0.8 * (p[0] - 1.6)]); p = p - 0.17 * g; path.append(p.copy())
    path = np.array(path); ax.plot(path[:, 0], path[:, 1], "o-", color=RED, ms=3.5, lw=1.4); ax.plot(1.6, 1.2, "*", color=GOLD, ms=16, mec=INK)
    ax.text(path[0, 0] + 0.08, path[0, 1] + 0.1, "시작점", color=RED); ax.text(1.7, 1.3, "손실 최소점", fontsize=8.5)
    ax.set_xlabel(r"$\beta_1$"); ax.set_ylabel(r"$\beta_2$"); ax.set_title("경사하강: 등고선의 가장 가파른 내리막으로 한 걸음씩", fontsize=9); return f


# ════════════════════════ 1. 선형 회귀 · 규제
@fig("ols")
def _():
    r_ = rng(2); x = r_.normal(0, 1, 40); y = 0.7 * x + r_.normal(0, 0.6, 40); b = np.polyfit(x, y, 1); yh = np.polyval(b, x)
    f, ax = plt.subplots(figsize=(5.6, 3.3)); ax.vlines(x, y, yh, color=RED, lw=1, alpha=0.7, label="잔차 (실제 − 예측)")
    ax.scatter(x, y, s=16, color=BLUE, zorder=3, label="관측치"); xs = np.linspace(-2.5, 2.5, 10); ax.plot(xs, np.polyval(b, xs), color=G, lw=2.2, label=f"OLS 직선: y = {b[1]:.2f} + {b[0]:.2f}x")
    ax.set_xlabel("x (예: 달러 지수 수익률)"); ax.set_ylabel("y (예: 옥수수 수익률)"); ax.legend(frameon=False, fontsize=8); return f


@fig("ridge_lasso")
def _():
    b1, b2 = np.meshgrid(np.linspace(-0.5, 2.6, 300), np.linspace(-0.5, 2.6, 300)); L = (b1 - 1.9) ** 2 + 1.6 * (b2 - 0.9) ** 2 + 0.9 * (b1 - 1.9) * (b2 - 0.9)
    f, axs = plt.subplots(1, 2, figsize=(7.4, 3.5))
    for ax, kind in zip(axs, ["Lasso (L1): 마름모", "Ridge (L2): 원"]):
        ax.contour(b1, b2, L, [0.05, 0.25, 0.6, 1.1, 1.8, 2.6], colors=GREY, linewidths=0.8); ax.plot(1.9, 0.9, "o", color=INK, ms=4); ax.text(1.95, 0.95, "OLS 해", fontsize=8)
        t = 1.0
        if "L1" in kind:
            ax.fill([t, 0, -t, 0], [0, t, 0, -t], color=BLUE, alpha=0.2, ec=BLUE); ax.plot(1.0, 0.0, "*", color=RED, ms=15, mec=INK); ax.text(1.05, 0.08, r"해: $\beta_2=0$ (정확히 0)", color=RED, fontsize=8.5)
        else:
            th = np.linspace(0, 2 * np.pi, 200); ax.fill(np.cos(th), np.sin(th), color=G, alpha=0.2, ec=G); ax.plot(0.93, 0.37, "*", color=RED, ms=15, mec=INK); ax.text(0.98, 0.45, "해: 둘 다 작아짐 (0 아님)", color=RED, fontsize=8.5)
        ax.axhline(0, color=MUTED, lw=0.6); ax.axvline(0, color=MUTED, lw=0.6); ax.set_aspect("equal"); ax.set_title(kind, fontsize=9.5); ax.set_xlabel(r"$\beta_1$"); ax.set_ylabel(r"$\beta_2$")
    return f


@fig("nonconvex")
def _():
    b = np.linspace(0, 4, 400); lam, a, g = 1.0, 3.7, 3.0
    scad = np.where(b <= lam, lam * b, np.where(b <= a * lam, (2 * a * lam * b - b ** 2 - lam ** 2) / (2 * (a - 1)), lam ** 2 * (a + 1) / 2))
    mcp = np.where(b <= g * lam, lam * b - b ** 2 / (2 * g), g * lam ** 2 / 2)
    z = np.linspace(-4, 4, 400); soft = np.sign(z) * np.maximum(np.abs(z) - lam, 0)
    scad_t = np.where(np.abs(z) <= 2 * lam, soft, np.where(np.abs(z) <= a * lam, ((a - 1) * z - np.sign(z) * a * lam) / (a - 2), z))
    f, (p, q) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    p.plot(b, lam * b, color=BLUE, lw=2, label="Lasso (계속 증가)"); p.plot(b, scad, color=RED, lw=2, label="SCAD"); p.plot(b, mcp, color=G, lw=2, ls="--", label="MCP")
    p.set_title("벌점 크기: 큰 계수에서는 평평해짐", fontsize=9); p.set_xlabel("|β|"); p.legend(frameon=False, fontsize=8)
    q.plot(z, z, color=GREY, lw=1, ls=":", label="벌점 없음 (OLS)"); q.plot(z, soft, color=BLUE, lw=2, label="Lasso: 모두 λ만큼 깎음"); q.plot(z, scad_t, color=RED, lw=2, label="SCAD: 큰 값은 그대로")
    q.set_title("추정치 변환: 작은 값은 0, 큰 값은?", fontsize=9); q.set_xlabel("OLS 추정치"); q.set_ylabel("벌점 추정치"); q.legend(frameon=False, fontsize=7.5); return f


@fig("horseshoe")
def _():
    r_ = rng(3); x = np.linspace(-4, 4, 800)
    hs = []
    lam = np.abs(r_.standard_cauchy(400000)); hs = r_.normal(0, lam)
    den, edges = np.histogram(hs, bins=np.linspace(-4, 4, 161), density=True); mid = (edges[1:] + edges[:-1]) / 2
    f, (p, q) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    p.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), color=G, lw=1.8, label="정규 (Ridge)"); p.plot(x, 0.5 * np.exp(-np.abs(x)), color=BLUE, lw=1.8, label="라플라스 (Lasso)")
    p.plot(mid, den / 2.2, color=RED, lw=2, label="Horseshoe"); p.set_ylim(0, 0.9); p.set_title("사전분포: 0에 뾰족 + 꼬리가 두꺼움", fontsize=9); p.legend(frameon=False, fontsize=8)
    k = np.linspace(0, 1, 200); q.hist(r_.beta(0.5, 0.5, 100000), bins=50, color=RED, alpha=0.6, density=True)
    q.set_title(r"축소 비율 $\kappa = 1/(1+\lambda^2\tau^2)$ 분포", fontsize=9); q.set_xlabel("0 = 전혀 안 줄임   ←→   1 = 0으로 완전히 축소"); q.set_yticks([])
    return f


@fig("quantile")
def _():
    import statsmodels.api as sm
    r_ = rng(4); x = r_.uniform(-2, 2, 500); y = 0.3 * x + (0.5 + 0.35 * (x + 2)) * r_.standard_normal(500)
    X = sm.add_constant(x); xs = np.linspace(-2, 2, 50)
    f, ax = plt.subplots(figsize=(5.8, 3.3)); ax.scatter(x, y, s=6, color=GREY, alpha=0.6)
    for tau, col in ((0.05, RED), (0.5, G), (0.95, BLUE)):
        b = sm.QuantReg(y, X).fit(q=tau).params; ax.plot(xs, b[0] + b[1] * xs, color=col, lw=2.2, label=f"τ = {tau:.2f}: 기울기 {b[1]:+.2f}")
    ax.set_xlabel("x (예: 변동성 지수)"); ax.set_ylabel("y (수익률)"); ax.legend(frameon=False, fontsize=8, title="분위수 회귀선", title_fontsize=8)
    ax.set_title("x가 클수록 아래 꼬리(5%)가 더 빨리 내려감 — 평균선만 보면 놓침", fontsize=9); return f


# ════════════════════════ 2. 차원 축소
@fig("pca")
def _():
    r_ = rng(5); Z = r_.multivariate_normal([0, 0], [[1, 0.85], [0.85, 1]], 300); C = np.cov(Z.T); w, V = np.linalg.eigh(C)
    f, (p, q) = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw=dict(width_ratios=[1.2, 1]))
    p.scatter(Z[:, 0], Z[:, 1], s=7, color=GREY, alpha=0.7)
    for k, col, nm in ((1, RED, "PC1"), (0, BLUE, "PC2")):
        v = V[:, k] * np.sqrt(w[k]) * 2; p.annotate("", v, (0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2)); p.text(v[0] * 1.08, v[1] * 1.08, nm, color=col, fontweight="bold")
    p.set_aspect("equal"); p.set_xlabel("옥수수 수익률 (표준화)"); p.set_ylabel("대두 수익률 (표준화)"); p.set_title("가장 넓게 퍼진 방향 = PC1 ('곡물 공통 흐름')", fontsize=9)
    ev = np.array([0.711, 0.193, 0.096]); q.bar(["PC1", "PC2", "PC3"], ev * 100, color=[RED, BLUE, GREY])
    for i, v in enumerate(ev):
        q.text(i, v * 100 + 1.5, f"{v:.1%}", ha="center")
    q.set_ylabel("설명 비율 (%)"); q.set_title("실제: 옥수수·대두·밀 3품목 → PC1이 71%", fontsize=9); return f


@fig("factor")
def _():
    k = np.arange(1, 13); pcr = 0.12 * (1 - np.exp(-(k - 1) / 4.5)) - 0.004 * k * (k > 8); pls = 0.13 * (1 - np.exp(-k / 1.3)) - 0.006 * (k - 3).clip(0)
    f, ax = plt.subplots(figsize=(5.8, 2.9)); ax.plot(k, pcr * 100, "o-", color=BLUE, label="PCR: X 분산 큰 순서로 요인"); ax.plot(k, pls * 100, "s-", color=RED, label="PLS: y와 공분산 큰 순서로 요인")
    ax.axhline(0, color=MUTED, lw=0.7); ax.set_xlabel("요인 수 k"); ax.set_ylabel("시험 R² (%)"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("y가 'X의 작은 요인'에 달려 있을 때: PLS는 빨리, PCR은 늦게 잡음 (예시)", fontsize=9); return f


# ════════════════════════ 3. 모형 평균
@fig("csr")
def _():
    from itertools import combinations
    r_ = rng(6); n, p = 260, 10; X = r_.normal(0, 1, (n, p)); beta = r_.normal(0, 0.12, p); y = X @ beta + r_.normal(0, 1, n); tr = 160
    out = []
    for kk in range(1, p + 1):
        preds = []
        combs = list(combinations(range(p), kk)); combs = [combs[i] for i in r_.choice(len(combs), min(60, len(combs)), replace=False)]
        for S in combs:
            S = list(S); A = np.c_[np.ones(tr), X[:tr, S]]; b = np.linalg.lstsq(A, y[:tr], rcond=None)[0]; preds.append(np.c_[np.ones(n - tr), X[tr:, S]] @ b)
        ph = np.mean(preds, 0); out.append(1 - np.sum((y[tr:] - ph) ** 2) / np.sum((y[tr:] - y[:tr].mean()) ** 2))
    f, ax = plt.subplots(figsize=(5.8, 2.9)); ax.plot(range(1, p + 1), np.array(out) * 100, "o-", color=G, lw=2); ax.axhline(0, color=MUTED, lw=0.7)
    ax.plot(p, out[-1] * 100, "o", color=RED, ms=9, mfc="none", mew=2); ax.text(p - 0.2, out[-1] * 100 - 0.9, "k = 전체\n= 일반 OLS", color=RED, ha="right", fontsize=8)
    ax.set_xlabel("조합 크기 k (변수 몇 개씩 묶나)"); ax.set_ylabel("시험 R² (%)"); ax.set_title("작은 조합 여러 개의 평균이 '전부 넣은 OLS'보다 안정적", fontsize=9); return f


@fig("bma")
def _():
    from itertools import combinations
    r_ = rng(7); n, p = 200, 8; X = r_.normal(0, 1, (n, p)); beta = np.array([0.4, 0.0, 0.25, 0, 0, 0.15, 0, 0]); y = X @ beta + r_.normal(0, 1, n)
    names = ["달러", "유가", "재고", "금리", "VIX", "운임", "기온", "COT"]; logw, incl = [], []
    for kk in range(0, p + 1):
        for S in combinations(range(p), kk):
            A = np.c_[np.ones(n), X[:, list(S)]]; res = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
            logw.append(-0.5 * (n * np.log(res @ res / n) + (kk + 1) * np.log(n))); incl.append(S)
    w = np.exp(np.array(logw) - max(logw)); w /= w.sum(); pip = [sum(w[i] for i, S in enumerate(incl) if j in S) for j in range(p)]
    f, ax = plt.subplots(figsize=(5.8, 2.9)); ax.bar(names, pip, color=[G if b else GREY for b in beta]); ax.axhline(0.5, color=RED, ls="--", lw=1); ax.text(7.4, 0.52, "0.5", color=RED, fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("사후 포함확률 PIP"); ax.set_title(f"변수 {p}개 → 모형 {2 ** p}개를 BIC 가중으로 평균 (초록 = 실제로 의미 있는 변수)", fontsize=9); return f


@fig("stacking")
def _():
    f, (p, q) = plt.subplots(1, 2, figsize=(7.4, 2.8))
    nm = ["Ridge", "나무 모델", "커널"]; w = [0.55, 0.15, 0.30]; p.bar(nm, w, color=[BLUE, G, GOLD]); p.set_title("학습된 결합 비중 (합 = 1, 음수 없음)", fontsize=9)
    for i, v in enumerate(w):
        p.text(i, v + 0.02, f"{v:.2f}", ha="center")
    mse = [1.000, 1.032, 1.011, 0.991, 0.994]; lab = ["Ridge", "나무", "커널", "Stacking", "단순 평균"]
    q.barh(lab[::-1], mse[::-1], color=[GREY, RED, GREY, GREY, GREY][::-1]); q.set_xlim(0.97, 1.04); q.axvline(1, color=MUTED, lw=0.8)
    q.set_title("시험 MSE (Ridge = 1, 낮을수록 좋음, 예시)", fontsize=9); return f


# ════════════════════════ 4. 레짐 · 상태공간
@fig("kalman")
def _():
    r_ = rng(8); T = 300; beta = np.r_[np.full(120, 0.4), np.linspace(0.4, 1.2, 60), np.full(120, 1.2)] + np.cumsum(r_.normal(0, 0.01, T))
    x = r_.normal(0, 1, T); y = beta * x + r_.normal(0, 0.5, T)
    b, P, q, R = 0.0, 1.0, 0.002, 0.25; kf = []
    for t in range(T):
        P += q; K = P * x[t] / (x[t] ** 2 * P + R); b += K * (y[t] - x[t] * b); P *= (1 - K * x[t]); kf.append(b)
    roll = [np.nan] * 60 + [np.polyfit(x[t - 60:t], y[t - 60:t], 1)[0] for t in range(60, T)]
    f, ax = plt.subplots(figsize=(6.6, 2.9)); ax.plot(beta, color=INK, lw=2.4, label="참 β (보이지 않음)"); ax.plot(kf, color=RED, lw=1.6, label="Kalman 추정")
    ax.plot(roll, color=BLUE, lw=1.4, ls="--", label="60기 롤링 회귀"); ax.axvspan(120, 180, color=GOLD, alpha=0.12); ax.text(122, 0.05, "관계 변화 구간", fontsize=8)
    ax.set_xlabel("시점"); ax.set_ylabel("β (민감도)"); ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left"); return f


@fig("hmm")
def _():
    r_ = rng(9); T = 400; P_ = np.array([[0.97, 0.03], [0.08, 0.92]]); s = [0]
    for _ in range(T - 1):
        s.append(r_.choice(2, p=P_[s[-1]]))
    s = np.array(s); mu, sd = np.array([0.002, -0.004]), np.array([0.012, 0.035]); y = r_.normal(mu[s], sd[s])
    al = np.array([0.5, 0.5]); prob = []
    for t in range(T):
        pr = al @ P_; lik = np.exp(-0.5 * ((y[t] - mu) / sd) ** 2) / sd; al = pr * lik; al /= al.sum(); prob.append(al[1])
    f, (a, b) = plt.subplots(2, 1, figsize=(6.8, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1]))
    a.bar(range(T), y * 100, color=np.where(s == 1, RED, BLUE), width=1); a.set_ylabel("수익률 (%)"); a.set_title("파랑 = 평시, 빨강 = 위기 (실제 상태 — 모델은 모름)", fontsize=9, loc="left")
    b.fill_between(range(T), prob, color=RED, alpha=0.35); b.plot(prob, color=RED, lw=1); b.set_ylim(0, 1); b.set_ylabel("P(위기)"); b.set_xlabel("시점")
    b.set_title("필터 확률: 그날까지의 자료만으로 계산한 '지금 위기일 확률'", fontsize=9, loc="left"); return f


@fig("tar")
def _():
    r_ = rng(10); q = r_.uniform(10, 45, 400); x = r_.normal(0, 1, 400); c = 25
    y = np.where(q > c, 1.1 * x, 0.3 * x) + r_.normal(0, 0.4, 400)
    f, (p, s) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    lo, hi = q <= c, q > c; p.scatter(x[lo], y[lo], s=7, color=BLUE, alpha=0.6, label="VIX ≤ 25: 기울기 0.3"); p.scatter(x[hi], y[hi], s=7, color=RED, alpha=0.6, label="VIX > 25: 기울기 1.1")
    xs = np.linspace(-3, 3, 10); p.plot(xs, 0.3 * xs, color=BLUE, lw=2); p.plot(xs, 1.1 * xs, color=RED, lw=2); p.legend(frameon=False, fontsize=8); p.set_xlabel("x (주가 수익률)"); p.set_ylabel("y (원자재 수익률)")
    p.set_title("임계값을 넘으면 관계가 바뀜", fontsize=9)
    qq = np.linspace(10, 45, 300)
    for gm, col in ((0.3, GOLD), (1.0, G), (20, RED)):
        s.plot(qq, 1 / (1 + np.exp(-gm * (qq - c))), color=col, lw=2, label=f"γ = {gm}" + (" (≈ TAR 계단)" if gm == 20 else ""))
    s.axvline(c, color=MUTED, ls=":"); s.set_xlabel("임계 변수 q (VIX)"); s.set_ylabel("G (위기 쪽 가중)"); s.legend(frameon=False, fontsize=8); s.set_title("STAR: 전환 속도 γ로 부드럽게", fontsize=9)
    return f


# ════════════════════════ 5. 변동성
@fig("garch")
def _():
    r_ = rng(11); T = 600; w, a, g, b = 0.02, 0.05, 0.10, 0.88; h = np.zeros(T); e = np.zeros(T); h[0] = w / (1 - a - g / 2 - b)
    for t in range(1, T):
        h[t] = w + (a + g * (e[t - 1] < 0)) * e[t - 1] ** 2 + b * h[t - 1]; e[t] = np.sqrt(h[t]) * r_.standard_normal()
    f, (p, q) = plt.subplots(1, 2, figsize=(7.6, 3.0), gridspec_kw=dict(width_ratios=[1.6, 1]))
    p.bar(range(T), e, color=GREY, width=1); p.plot(2 * np.sqrt(h), color=RED, lw=1.3, label="±2√h (조건부 변동성)"); p.plot(-2 * np.sqrt(h), color=RED, lw=1.3)
    p.set_title("큰 충격 다음엔 큰 변동이 이어짐 (군집)", fontsize=9); p.legend(frameon=False, fontsize=8); p.set_xlabel("시점")
    z = np.linspace(-3, 3, 200); hb = w / (1 - a - g / 2 - b); q.plot(z, w + a * z ** 2 + b * hb, color=BLUE, lw=2, label="GARCH (대칭)"); q.plot(z, w + (a + g * (z < 0)) * z ** 2 + b * hb, color=RED, lw=2, label="GJR (하락에 더 큼)")
    q.set_xlabel(r"어제 충격 $\varepsilon_{t-1}$"); q.set_ylabel(r"오늘 분산 $h_t$"); q.set_title("뉴스 충격 곡선", fontsize=9); q.legend(frameon=False, fontsize=8); return f


@fig("dcc")
def _():
    r_ = rng(12); T = 500; a, b = 0.05, 0.93; Qb = np.array([[1, 0.3], [0.3, 1]]); Q = Qb.copy(); rho = []
    for t in range(T):
        R = Q / np.sqrt(np.outer(np.diag(Q), np.diag(Q))); rho.append(R[0, 1])
        Rt = R if not (250 < t < 300) else np.array([[1, 0.85], [0.85, 1]]); z = r_.multivariate_normal([0, 0], Rt)
        Q = (1 - a - b) * Qb + a * np.outer(z, z) + b * Q
    f, ax = plt.subplots(figsize=(6.6, 2.7)); ax.plot(rho, color=BLUE, lw=1.4); ax.axhline(0.3, color=MUTED, ls="--", lw=1); ax.text(5, 0.33, "장기 평균 상관 0.3", fontsize=8, color=MUTED)
    ax.axvspan(250, 300, color=RED, alpha=0.1); ax.text(252, 0.92, "위기: 함께 폭락", color=RED, fontsize=8.5); ax.set_ylim(-0.2, 1.05); ax.set_ylabel(r"동적 상관 $\rho_t$"); ax.set_xlabel("시점")
    ax.set_title("DCC: 품목 간 상관이 매 시점 갱신 — 위기 때 분산투자 효과가 사라짐", fontsize=9); return f


@fig("har")
def _():
    f, (p, q) = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw=dict(width_ratios=[1, 1.4]))
    p.bar(["일간\n(어제)", "주간\n(5일 평균)", "월간\n(22일 평균)"], [0.36, 0.28, 0.25], color=[RED, GOLD, BLUE]); p.set_title("계수 예시: 세 시간 척도를 함께", fontsize=9)
    lag = np.arange(1, 23); w = 0.36 * (lag == 1) + 0.28 / 5 * (lag <= 5) + 0.25 / 22
    q.bar(lag, w, color=G); q.set_xlabel("시차 (일)"); q.set_ylabel("시차별 가중"); q.set_title("결국 '계단 모양 가중'으로 과거 22일을 요약", fontsize=9); return f


@fig("midas")
def _():
    k = np.arange(1, 61); f, ax = plt.subplots(figsize=(5.8, 2.8))
    for (t1, t2), col in (((1, 5), RED), ((1, 15), BLUE), ((2, 10), G), ((1, 1), GREY)):
        x = k / 61; w = x ** (t1 - 1) * (1 - x) ** (t2 - 1); w /= w.sum(); ax.plot(k, w, lw=2, color=col, label=f"$\\theta_1$={t1}, $\\theta_2$={t2}")
    ax.set_xlabel("일간 시차 k (1 = 어제)"); ax.set_ylabel("가중치 w(k)"); ax.legend(frameon=False, fontsize=8); ax.set_title("Beta 가중 함수: 모수 2개로 60개 시차의 가중 모양을 결정", fontsize=9); return f


# ════════════════════════ 6. 시스템
@fig("var_dy")
def _():
    nm = ["원유", "옥수수", "대두", "금"]; M = np.array([[78, 9, 7, 6], [21, 55, 18, 6], [15, 22, 57, 6], [8, 3, 3, 86]])
    f, ax = plt.subplots(figsize=(4.6, 3.6)); im = ax.imshow(M, cmap="Greens", vmin=0, vmax=100)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, M[i, j], ha="center", va="center", color="white" if M[i, j] > 50 else INK)
    ax.set_xticks(range(4), nm); ax.set_yticks(range(4), nm); ax.set_xlabel("충격 출처 (주는 쪽)"); ax.set_ylabel("받는 쪽")
    ax.set_title("예측오차 분산분해 (%) — 대각선 밖 합 = 스필오버", fontsize=9); return f


@fig("minnesota")
def _():
    lag = np.arange(1, 7); lam, th = 0.2, 0.5; f, ax = plt.subplots(figsize=(5.6, 2.8))
    ax.bar(lag - 0.18, lam / lag, 0.36, color=BLUE, label="자기 시차 (옥수수 → 옥수수)"); ax.bar(lag + 0.18, lam * th / lag, 0.36, color=GOLD, label="다른 변수 시차 (원유 → 옥수수)")
    ax.set_xlabel(r"시차 $\ell$"); ax.set_ylabel("사전 표준편차"); ax.legend(frameon=False, fontsize=8); ax.set_title("먼 시차·다른 변수일수록 '0 근처'로 강하게 묶음", fontsize=9); return f


@fig("favar")
def _():
    h = np.arange(0, 13); f, ax = plt.subplots(figsize=(5.8, 2.8))
    for nm, (a, d), col in (("위험선호 요인 충격", (0.9, 0.55), RED), ("달러 요인 충격", (-0.6, 0.7), BLUE), ("원유 충격", (0.5, 0.35), GOLD)):
        ax.plot(h, a * (1 - d ** (h + 1)) / (1 - d), "o-", color=col, ms=3.5, lw=1.8, label=nm)
    ax.axhline(0, color=MUTED, lw=0.7); ax.set_xlabel("충격 후 경과 (주)"); ax.set_ylabel("옥수수 누적 반응 (표준화)"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("공통 요인 충격이 옥수수에 퍼지는 경로 (예시)", fontsize=9); return f


# ════════════════════════ 7. 패널
@fig("panel")
def _():
    r_ = rng(13); f, ax = plt.subplots(figsize=(5.8, 3.3)); allx, ally = [], []
    for i, (a, col, nm) in enumerate(((-1.5, BLUE, "옥수수"), (-0.5, G, "밀"), (0.5, GOLD, "대두"), (1.5, RED, "구리"))):
        x = r_.normal(i * 0.9 - 1.3, 0.45, 40); y = a + 0.5 * x + r_.normal(0, 0.3, 40) - 1.6 * (i * 0.9 - 1.3) * 0.9
        ax.scatter(x, y, s=10, color=col, alpha=0.7, label=nm); xs = np.linspace(x.min(), x.max(), 5); ax.plot(xs, y.mean() + 0.5 * (xs - x.mean()), color=col, lw=1.5)
        allx += list(x); ally += list(y)
    b = np.polyfit(allx, ally, 1); xs = np.linspace(min(allx), max(allx), 5); ax.plot(xs, np.polyval(b, xs), color=INK, lw=2.4, ls="--", label=f"Pooled: 기울기 {b[0]:+.2f} (편향)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2); ax.set_xlabel("x (예: 캐리)"); ax.set_ylabel("y (수익률)"); ax.set_title("FE: 품목별 평균을 빼면 공통 기울기 +0.50 회복", fontsize=9); return f


# ════════════════════════ 8. 머신러닝
@fig("trees")
def _():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor
    r_ = rng(14); x = np.sort(r_.uniform(0, 10, 200)); y = np.sin(x) + 0.3 * r_.standard_normal(200); xs = np.linspace(0, 10, 500)[:, None]
    t = DecisionTreeRegressor(max_depth=3).fit(x[:, None], y); gb = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05).fit(x[:, None], y)
    f, ax = plt.subplots(figsize=(6.2, 2.9)); ax.scatter(x, y, s=6, color=GREY); ax.plot(xs, t.predict(xs), color=BLUE, lw=2, label="나무 1개 (깊이 3): 계단 8칸")
    ax.plot(xs, gb.predict(xs), color=RED, lw=2, label="부스팅 200그루: 계단을 쌓아 곡선 근사"); ax.plot(xs, np.sin(xs), color=G, lw=1, ls="--", label="참 관계")
    ax.legend(frameon=False, fontsize=8); ax.set_xlabel("x"); ax.set_ylabel("y"); return f


@fig("gp")
def _():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    r_ = rng(15); x = np.r_[r_.uniform(0, 3, 12), r_.uniform(6, 9, 12)]; y = np.sin(x) + 0.15 * r_.standard_normal(24); xs = np.linspace(-1, 11, 300)[:, None]
    gp = GaussianProcessRegressor(RBF(1.0) + WhiteKernel(0.02)).fit(x[:, None], y); m, s = gp.predict(xs, return_std=True)
    f, ax = plt.subplots(figsize=(6.2, 2.9)); ax.fill_between(xs[:, 0], m - 2 * s, m + 2 * s, color=BLUE, alpha=0.18, label="95% 구간"); ax.plot(xs, m, color=BLUE, lw=2, label="GP 평균 예측")
    ax.scatter(x, y, color=RED, s=14, zorder=3, label="관측치"); ax.text(4.5, -2.2, "자료 없는 구간 → 구간이 넓어짐", ha="center", fontsize=8.5); ax.legend(frameon=False, fontsize=8, loc="upper right"); return f


# ════════════════════════ 9. 대안 시계열
@fig("prophet")
def _():
    t = np.arange(260); r_ = rng(16); trend = np.where(t < 110, 0.02 * t, np.where(t < 190, 2.2 - 0.015 * (t - 110), 1.0 + 0.025 * (t - 190)))
    seas = 0.45 * np.sin(2 * np.pi * t / 52); y = trend + seas + 0.2 * r_.standard_normal(260)
    f, (a, b, c) = plt.subplots(3, 1, figsize=(6.6, 4.2), sharex=True)
    a.plot(t, y, color=GREY, lw=1); a.plot(t, trend + seas, color=BLUE, lw=1.5); a.set_title("관측 = 추세 + 계절 + 잡음", fontsize=9, loc="left")
    b.plot(t, trend, color=RED, lw=2); [b.axvline(v, color=MUTED, ls=":") for v in (110, 190)]; b.set_title("g(t): 구간별 직선 추세 — 기울기가 바뀌는 변곡점", fontsize=9, loc="left")
    c.plot(t, seas, color=G, lw=2); c.set_title("s(t): 1년 주기 계절 (푸리에 합)", fontsize=9, loc="left"); c.set_xlabel("주"); return f


@fig("chronos")
def _():
    r_ = rng(17); x = np.cumsum(r_.normal(0, 1, 40)) + 20; sc = np.abs(x).mean(); xt = x / sc; bins = np.linspace(xt.min() - 0.02, xt.max() + 0.02, 12); tok = np.digitize(xt, bins)
    f, (p, q) = plt.subplots(1, 2, figsize=(7.6, 2.9))
    p.plot(xt, color=BLUE, lw=1.3); [p.axhline(v, color=GREY, lw=0.4) for v in bins]
    for i in range(0, 40, 4):
        p.text(i, xt[i] + 0.006, str(tok[i]), fontsize=7.5, color=RED, ha="center")
    p.set_title("① 평균 크기로 나눈 뒤 '구간 번호(토큰)'로 바꿈", fontsize=9); p.set_xlabel("시점"); p.set_ylabel("스케일된 값")
    h = np.arange(40, 60); base = x[-1]; s = np.sqrt(np.arange(1, 21)) * 1.1; q.plot(range(40), x, color=INK, lw=1.3)
    for lo, hi, al in ((0.1, 0.9, 0.18), (0.25, 0.75, 0.3)):
        from scipy.stats import norm
        q.fill_between(h, base + norm.ppf(lo) * s, base + norm.ppf(hi) * s, color=RED, alpha=al, lw=0)
    q.plot(h, np.full(20, base), color=RED, lw=1.6); q.set_title("② 다음 토큰 확률 → 분위 예측 부채꼴", fontsize=9); q.set_xlabel("시점"); return f


@fig("timesfm")
def _():
    r_ = rng(18); x = np.sin(np.arange(96) / 6) + 0.2 * r_.standard_normal(96); f, ax = plt.subplots(figsize=(6.8, 2.5)); ax.plot(range(64), x[:64], color=INK, lw=1.2)
    for j in range(4):
        ax.add_patch(Rectangle((j * 16, -1.7), 16, 3.4, fc=PAL[j % 4], alpha=0.12, ec=PAL[j % 4])); ax.text(j * 16 + 8, 1.85, f"패치 {j + 1}\n(토큰)", ha="center", fontsize=8)
    ax.add_patch(Rectangle((64, -1.7), 32, 3.4, fc=RED, alpha=0.08, ec=RED, ls="--")); ax.plot(range(64, 96), x[64:], color=RED, lw=1.6); ax.text(80, 1.85, "다음 긴 패치를\n한 번에 예측", ha="center", color=RED, fontsize=8)
    ax.set_ylim(-1.9, 2.5); ax.set_yticks([]); ax.set_xlabel("시점"); return f


@fig("patchtst")
def _():
    r_ = rng(19); f, axs = plt.subplots(3, 1, figsize=(6.8, 3.4), sharex=True)
    for i, (nm, col) in enumerate((("옥수수", BLUE), ("원유", GOLD), ("금", RED))):
        x = np.cumsum(r_.normal(0, 1, 52)); axs[i].plot(x, color=col, lw=1.3); axs[i].set_yticks([]); axs[i].set_ylabel(nm, rotation=0, ha="right", va="center")
        for j in range(0, 52, 4):
            axs[i].axvline(j, color=GREY, lw=0.5)
    axs[0].set_title("4주씩 잘라 패치 13개 = 토큰 13개. 품목마다 따로 같은 트랜스포머 통과 (채널 독립)", fontsize=9); axs[-1].set_xlabel("주"); return f


@fig("itransformer")
def _():
    nm = ["옥수수", "밀", "대두", "원유", "난방유", "금", "은"]; C = np.array([[1, .6, .65, .2, .18, .05, .1], [.6, 1, .5, .15, .12, .05, .08], [.65, .5, 1, .25, .2, .06, .1],
                                                               [.2, .15, .25, 1, .85, .15, .2], [.18, .12, .2, .85, 1, .12, .18], [.05, .05, .06, .15, .12, 1, .8], [.1, .08, .1, .2, .18, .8, 1]])
    A = np.exp(3 * C); A /= A.sum(1, keepdims=True); f, ax = plt.subplots(figsize=(4.8, 3.8)); ax.imshow(A, cmap="Greens")
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if A[i, j] > 0.3 else INK)
    ax.set_xticks(range(7), nm, rotation=45); ax.set_yticks(range(7), nm); ax.set_title("품목 = 토큰 → 품목 간 어텐션 가중치 (행 합 = 1, 예시)", fontsize=9); return f


@fig("tft")
def _():
    f, (p, q) = plt.subplots(1, 2, figsize=(7.6, 2.9), gridspec_kw=dict(width_ratios=[1, 1.4]))
    nm = ["과거 수익률", "재고 지표", "달러", "달력(월)", "운임"]; w = [0.38, 0.24, 0.17, 0.13, 0.08]; p.barh(nm[::-1], w[::-1], color=G); p.set_title("변수 선택 가중 (softmax, 합 = 1)", fontsize=9)
    r_ = rng(20); t = np.arange(60); y = np.cumsum(r_.normal(0, 1, 60)); q.plot(t[:40], y[:40], color=INK, lw=1.2); m = y[39] + 0.05 * np.arange(20); s = 0.9 * np.sqrt(np.arange(1, 21))
    q.fill_between(t[40:], m - 1.28 * s, m + 1.28 * s, color=BLUE, alpha=0.2, label="10–90% 분위"); q.plot(t[40:], m, color=BLUE, lw=1.8, label="50% 분위"); q.plot(t[40:], y[40:], color=RED, lw=1, ls="--", label="실제")
    q.legend(frameon=False, fontsize=8); q.set_title("여러 지평의 분위 예측", fontsize=9); return f


@fig("node")
def _():
    t = np.linspace(0, 12, 400); z = np.exp(-0.12 * t) * np.cos(1.3 * t); r_ = rng(21); to = np.sort(r_.choice(400, 14, replace=False)); f, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.plot(t, z, color=BLUE, lw=2, label="ODE 적분 궤적 z(t) (연속)"); ax.scatter(t[to], z[to] + 0.06 * r_.standard_normal(14), color=RED, zorder=3, s=22, label="불규칙 간격 관측")
    for i in (60, 160, 260):
        d = (-0.12 * z[i] - 1.3 * np.exp(-0.12 * t[i]) * np.sin(1.3 * t[i])); ax.annotate("", (t[i] + 0.7, z[i] + 0.7 * d), (t[i], z[i]), arrowprops=dict(arrowstyle="->", color=G, lw=1.6))
    ax.text(t[60] + 0.75, z[60] + 0.7 * (-0.12 * z[60] - 1.3 * np.exp(-0.12 * t[60]) * np.sin(1.3 * t[60])), "  dz/dt = f(z)", color=G, fontsize=8.5)
    ax.legend(frameon=False, fontsize=8); ax.set_xlabel("시간 t"); return f


# ════════════════════════ 10. 불확실성 · 검증
@fig("conformal")
def _():
    r_ = rng(22); n = 200; e = r_.standard_t(4, n) * 0.8; q = np.quantile(np.abs(r_.standard_t(4, 300) * 0.8), 0.9)
    f, ax = plt.subplots(figsize=(6.4, 2.8)); inside = np.abs(e) <= q; ax.scatter(np.flatnonzero(inside), e[inside], s=8, color=BLUE, label=f"구간 안 ({inside.mean():.0%})")
    ax.scatter(np.flatnonzero(~inside), e[~inside], s=12, color=RED, label="구간 밖"); ax.axhspan(-q, q, color=G, alpha=0.1); ax.axhline(q, color=G, lw=1); ax.axhline(-q, color=G, lw=1)
    ax.text(n + 2, q, f" $+\\hat{{q}}$ = {q:.2f}", va="center", fontsize=8, color=G); ax.set_xlabel("시험 시점"); ax.set_ylabel("오차 (실제 − 예측)"); ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title(r"보정용 |오차|의 90% 분위수 $\hat{q}$로 ±구간 → 시험에서도 약 90% 포함", fontsize=9); return f


@fig("cv")
def _():
    f, ax = plt.subplots(figsize=(6.8, 2.6))
    for k in range(5):
        y = 4 - k; tr = 4 + k * 2; ax.add_patch(Rectangle((0, y), tr, 0.6, fc=BLUE, alpha=0.8)); ax.add_patch(Rectangle((tr, y), 0.5, 0.6, fc=GREY)); ax.add_patch(Rectangle((tr + 0.5, y), 2, 0.6, fc=GOLD))
        ax.text(-0.3, y + 0.3, f"{k + 1}회", ha="right", va="center", fontsize=8)
    ax.text(2, 4.8, "학습 (과거)", color=BLUE, fontsize=8.5); ax.text(6.6, 4.8, "gap", color=MUTED, fontsize=8.5); ax.text(7.4, 4.8, "검증 (미래)", color=OCHRE, fontsize=8.5)
    ax.set_xlim(-1.2, 15); ax.set_ylim(-0.2, 5.3); ax.axis("off"); ax.annotate("", (14.8, -0.1), (0, -0.1), arrowprops=dict(arrowstyle="->", color=INK)); ax.text(14.8, 0.1, "시간", ha="right", fontsize=8)
    return f


@fig("tests")
def _():
    r_ = rng(23); T = 400; d = r_.normal(0.0004, 0.02, T); d[150:200] += 0.006
    f, ax = plt.subplots(figsize=(6.4, 2.7)); ax.plot(np.cumsum(d), color=G, lw=1.8); ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("시험 시점"); ax.set_ylabel("누적 (기준 오차² − 모델 오차²)"); ax.set_title("오르면 모델이 '과거 평균'보다 나음, 내리면 못함 — 일부 구간에서만 오르는지 확인", fontsize=9); return f


# ════════════════════════ 11. 포트폴리오
@fig("newton")
def _():
    f, (p, q) = plt.subplots(1, 2, figsize=(7.6, 3.0))
    r_ = rng(24); mu = np.array([0.06, 0.09, 0.04, 0.07]); S = np.diag([0.18, 0.28, 0.12, 0.22]) ** 2; S += 0.3 * np.sqrt(np.outer(np.diag(S), np.diag(S))) * (1 - np.eye(4))
    W = r_.dirichlet(np.ones(4), 3000); vol = np.sqrt(np.einsum("ij,jk,ik->i", W, S, W)); ret = W @ mu; p.scatter(vol * 100, ret * 100, s=2, color=GREY, alpha=0.5)
    lam = 4; U = ret - lam / 2 * vol ** 2; i = np.argmax(U); p.plot(vol[i] * 100, ret[i] * 100, "*", color=RED, ms=15, mec=INK); p.text(vol[i] * 100 + 0.4, ret[i] * 100, "효용 최대 비중", color=RED, fontsize=8.5)
    p.set_xlabel("변동성 (%)"); p.set_ylabel("기대수익 (%)"); p.set_title("무작위 비중 3,000개와 최적점", fontsize=9)
    w = np.linspace(-0.2, 1.4, 200); F = -(0.08 * w - 2 * 0.2 ** 2 * w ** 2 - 0.01 * np.abs(w - 0.3)); q.plot(w, F, color=BLUE, lw=2)
    xs = [1.25]
    for _ in range(3):
        x = xs[-1]; g = -(0.08 - 4 * 0.04 * x - 0.01 * np.sign(x - 0.3)); h_ = 4 * 0.04; xs.append(x - g / h_)
    q.plot(xs, [-(0.08 * x - 2 * 0.04 * x ** 2 - 0.01 * abs(x - 0.3)) for x in xs], "o--", color=RED, ms=5); q.set_xlabel("비중 w"); q.set_ylabel("−효용 (최소화)"); q.set_title("Newton: 곡률(2차 미분)로 한 번에 크게 점프", fontsize=9)
    return f


@fig("lw")
def _():
    r_ = rng(25); N, T = 19, 104; X = r_.normal(0, 1, (T, N)); ev = np.sort(np.linalg.eigvalsh(np.cov(X.T)))[::-1]; d = 0.35; lw = (1 - d) * ev + d * ev.mean()
    f, ax = plt.subplots(figsize=(6.2, 2.9)); ax.plot(range(1, N + 1), np.ones(N), color=INK, lw=2, label="참 고유값 (모두 1)"); ax.plot(range(1, N + 1), ev, "o-", color=RED, ms=4, label="표본 공분산: 양 끝으로 과장")
    ax.plot(range(1, N + 1), lw, "s-", color=G, ms=4, label=f"Ledoit-Wolf (δ = {d})"); ax.set_yscale("log"); ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xlabel("고유값 순위"); ax.set_ylabel("고유값 (로그)"); ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"품목 {N}개 · 관측 {T}주: 표본 추정은 '퍼짐'을 과장", fontsize=9); return f


@fig("rp")
def _():
    nm = ["금", "옥수수", "원유", "천연가스"]; vol = np.array([0.15, 0.25, 0.35, 0.55]); w_eq = np.ones(4) / 4; w_rp = (1 / vol) / (1 / vol).sum()
    rc = lambda w: w * vol / (w * vol).sum()
    f, (p, q) = plt.subplots(1, 2, figsize=(7.4, 2.8)); x = np.arange(4)
    p.bar(x - 0.2, w_eq * 100, 0.4, color=GREY, label="1/N"); p.bar(x + 0.2, w_rp * 100, 0.4, color=G, label="리스크 패리티"); p.set_xticks(x, nm); p.set_title("자금 비중 (%)", fontsize=9); p.legend(frameon=False, fontsize=8)
    q.bar(x - 0.2, rc(w_eq) * 100, 0.4, color=GREY); q.bar(x + 0.2, rc(w_rp) * 100, 0.4, color=G); q.set_xticks(x, nm); q.axhline(25, color=RED, ls="--", lw=1); q.set_title("위험 기여 (%) — RP는 모두 25%", fontsize=9)
    return f


@fig("band")
def _():
    r_ = rng(26); T = 250; tgt = 0.5 + np.cumsum(r_.normal(0, 0.02, T)); w = [tgt[0]]; b = 0.08; trades = []
    for t in range(1, T):
        if abs(tgt[t] - w[-1]) > b:
            w.append(tgt[t]); trades.append(t)
        else:
            w.append(w[-1])
    f, (p, q) = plt.subplots(2, 1, figsize=(6.6, 3.6), sharex=True)
    p.fill_between(range(T), tgt - b, tgt + b, color=GOLD, alpha=0.2, label="무거래 폭 ±8%p"); p.plot(tgt, color=INK, lw=1, label="목표 비중 (매일 흔들림)"); p.step(range(T), w, where="post", color=RED, lw=1.8, label=f"실제 비중 (거래 {len(trades)}회)")
    p.legend(frameon=False, fontsize=7.5, ncol=3); p.set_ylabel("비중")
    vol = 0.12 + 0.1 * np.exp(-((np.arange(T) - 150) / 20) ** 2) + 0.01 * r_.standard_normal(T); q.plot(np.minimum(4, 0.10 / vol), color=BLUE, lw=1.5); q.set_ylabel("배율 = 10% ÷ 예측 변동성"); q.set_xlabel("거래일")
    q.set_title("변동성이 커지면 노출을 줄임", fontsize=9, loc="left"); return f


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    import figs as B                                   # figs_deep이 'figs' 모듈의 FIGS에 등록 → 그쪽 목록 사용
    import figs_deep  # noqa: F401  LSTM·GNN 심화 그림 등록
    only = set(sys.argv[1:])
    for name, fn in B.FIGS.items():
        if only and name not in only:
            continue
        save(fn(), name); print("saved", name)
    import json
    old = json.loads((HERE / "facts.json").read_text(encoding="utf-8")) if (HERE / "facts.json").exists() else {}
    (HERE / "facts.json").write_text(json.dumps({**old, **B.FACTS}, ensure_ascii=False, indent=1), encoding="utf-8")
