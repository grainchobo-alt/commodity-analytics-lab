"""블랙숄즈 학습 노트 그림 (옵션 랩 ① 탭 해설용). 계산은 프로젝트 루트 options_bs.py 재사용.

사용: .venv\\Scripts\\python.exe docs\\학습\\scripts\\figs_bs.py   → docs\\학습\\scripts\\img\\bs_*.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
import options_bs as O  # noqa: E402

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 9, "savefig.dpi": 180,
                     "mathtext.fontset": "cm", "axes.spines.top": False, "axes.spines.right": False})
OUT = HERE / "img"; OUT.mkdir(exist_ok=True)
G, GOLD, RED, BLUE, INK, MUTED, GREY = "#1E3A2F", "#C8A24A", "#D9582B", "#2F6E8F", "#222222", "#6B6B66", "#A8A8A2"
S0, K0, SIG, R, T0 = 100.0, 100.0, 0.25, 0.04, 0.5          # 노트 전체의 기준 예시 (옵션 랩 기본값과 같음)


def save(f, name):
    f.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.06, facecolor="white"); plt.close(f)


def payoff():
    S = np.linspace(60, 140, 200); prem_c, prem_p = O.bs_price(S0, K0, T0, R, SIG, True), O.bs_price(S0, K0, T0, R, SIG, False)
    f, ax = plt.subplots(1, 4, figsize=(11, 2.6), sharey=True)
    for a, (y, t, c) in zip(ax, [(np.maximum(S - K0, 0) - prem_c, "콜 매수 (살 권리)", G), (-(np.maximum(S - K0, 0) - prem_c), "콜 매도", RED),
                                 (np.maximum(K0 - S, 0) - prem_p, "풋 매수 (팔 권리)", BLUE), (-(np.maximum(K0 - S, 0) - prem_p), "풋 매도", GOLD)]):
        a.plot(S, y, c=c, lw=2); a.axhline(0, c=GREY, lw=.8); a.axvline(K0, c=GREY, ls=":", lw=.8)
        a.set_title(t, fontsize=10); a.set_xlabel("만기 가격 S_T")
    ax[0].set_ylabel("만기 손익 (프리미엄 반영)")
    ax[0].annotate(f"최대 손실 = 프리미엄 {prem_c:.2f}", (70, -prem_c), (62, 15), fontsize=8, arrowprops=dict(arrowstyle="->", color=MUTED))
    f.tight_layout(); save(f, "bs_payoff")


def procure():
    S = np.linspace(70, 140, 200); prem = O.bs_price(S0, 110, T0, R, SIG, True)
    f, a = plt.subplots(figsize=(6.2, 3.0))
    a.plot(S, S, c=GREY, ls="--", label="헤지 없음: 그때 시세로 구매")
    a.plot(S, np.minimum(S, 110) + prem, c=G, lw=2.2, label=f"행사가 110 콜 매수: 최대 110 + 프리미엄 {prem:.2f}")
    a.axhline(110 + prem, c=RED, ls=":", lw=1); a.text(72, 110 + prem + 1.5, f"구매 단가 상한 약 {110 + prem:.1f}", color=RED, fontsize=8)
    a.set_xlabel("만기(구매 시점) 시세"); a.set_ylabel("실제 구매 단가"); a.legend(fontsize=8, frameon=False, loc="upper left")
    f.tight_layout(); save(f, "bs_procure")


def normal():
    x = np.linspace(-4, 4, 400); d = 0.6
    f, ax = plt.subplots(1, 2, figsize=(9, 2.8))
    ax[0].plot(x, norm.pdf(x), c=G, lw=2); ax[0].fill_between(x, norm.pdf(x), where=x <= d, color=GOLD, alpha=.45)
    ax[0].axvline(d, c=RED, lw=1); ax[0].text(d + .1, .3, f"x = {d}", color=RED)
    ax[0].text(-3.8, .32, f"색칠한 넓이 = N({d}) = {norm.cdf(d):.3f}", fontsize=9); ax[0].set_title("표준정규분포의 확률밀도 φ(x)", fontsize=10)
    ax[1].plot(x, norm.cdf(x), c=BLUE, lw=2); ax[1].scatter([d], [norm.cdf(d)], c=RED, zorder=3)
    ax[1].annotate(f"N({d}) = {norm.cdf(d):.3f}", (d, norm.cdf(d)), (1.4, .45), arrowprops=dict(arrowstyle="->", color=MUTED))
    ax[1].set_title("누적분포함수 N(x) — 왼쪽 넓이를 곡선으로", fontsize=10)
    for a in ax: a.set_xlabel("x (표준편차 단위)")
    f.tight_layout(); save(f, "bs_normal")


def sqrt_t():
    rng = np.random.default_rng(3); n, steps = 400, 252
    W = np.cumsum(rng.standard_normal((n, steps)) * np.sqrt(1 / 252), axis=1); t = np.arange(1, steps + 1) / 252
    f, ax = plt.subplots(1, 2, figsize=(9.5, 2.9))
    for w in W[:40]: ax[0].plot(t, w, lw=.6, c=BLUE, alpha=.5)
    ax[0].plot(t, 2 * np.sqrt(t), c=RED, lw=1.6, label="±2√t"); ax[0].plot(t, -2 * np.sqrt(t), c=RED, lw=1.6)
    ax[0].set_title("브라운 운동 W_t 경로 40개 — 퍼지는 폭은 √t에 비례", fontsize=10); ax[0].set_xlabel("시간 t (년)"); ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(t, W.std(0), c=G, lw=2, label="경로 400개의 실제 표준편차"); ax[1].plot(t, np.sqrt(t), c=GOLD, ls="--", lw=2, label="이론 √t")
    ax[1].set_title("시간이 4배면 폭은 2배", fontsize=10); ax[1].set_xlabel("시간 t (년)"); ax[1].legend(frameon=False, fontsize=8)
    f.tight_layout(); save(f, "bs_sqrt_t")


def vol_drag():
    rng = np.random.default_rng(11); f, ax = plt.subplots(1, 2, figsize=(9.5, 2.9))
    path = 100 * np.cumprod([1.1, 0.9] * 10); ax[0].plot(np.r_[100, path], marker="o", ms=3, c=RED)
    ax[0].set_title("+10%, -10%를 반복하면: 평균 수익 0%인데 가격은 줄어듦", fontsize=10); ax[0].set_xlabel("기간"); ax[0].set_ylabel("가격")
    ax[0].text(1, 82, "한 번 왕복마다 ×(1.1×0.9) = ×0.99", fontsize=8.5, color=MUTED)
    sigs = np.linspace(0, .8, 50)
    ax[1].plot(sigs * 100, (-(sigs ** 2) / 2) * 100, c=G, lw=2)
    ax[1].set_title("변동성이 클수록 '전형적' 성장률이 μ보다 σ²/2만큼 낮아짐", fontsize=10)
    ax[1].set_xlabel("변동성 σ (%)"); ax[1].set_ylabel("-σ²/2 (연 %p)")
    f.tight_layout(); save(f, "bs_vol_drag")


def lognormal():
    Sg = np.linspace(20, 260, 500); f, a = plt.subplots(figsize=(6.4, 3.0))
    for t, c in [(0.25, G), (1.0, BLUE), (3.0, GOLD)]:
        mu, sd = np.log(S0) + (R - SIG ** 2 / 2) * t, SIG * np.sqrt(t)
        a.plot(Sg, norm.pdf((np.log(Sg) - mu) / sd) / (Sg * sd), c=c, lw=2, label=f"T = {t}년")
    a.axvline(K0, c=RED, ls=":", lw=1); a.text(K0 + 2, a.get_ylim()[1] * .9, "K", color=RED)
    a.set_xlabel("만기 가격 S_T"); a.set_ylabel("확률밀도"); a.legend(frameon=False, fontsize=8)
    a.set_title("만기 가격의 로그정규분포 — 오른쪽 꼬리가 길고, 시간이 갈수록 퍼짐", fontsize=10)
    f.tight_layout(); save(f, "bs_lognormal")


def binomial():
    f, a = plt.subplots(figsize=(6.6, 3.0)); a.axis("off")
    def node(x, y, t, c=G):
        a.add_patch(FancyBboxPatch((x - .55, y - .28), 1.1, .56, boxstyle="round,pad=0.02", fc="white", ec=c, lw=1.4)); a.text(x, y, t, ha="center", va="center", fontsize=9)
    node(0, 0, "S = 100\nC = ?"); node(3, 1.1, "S = 120\nC = 20", BLUE); node(3, -1.1, "S = 80\nC = 0", RED)
    a.annotate("", (2.4, 1.0), (.6, .15), arrowprops=dict(arrowstyle="->", color=MUTED)); a.annotate("", (2.4, -1.0), (.6, -.15), arrowprops=dict(arrowstyle="->", color=MUTED))
    a.text(1.3, .85, "오름", color=BLUE); a.text(1.3, -.95, "내림", color=RED)
    a.text(4.0, .45, "Δ = (20 - 0)/(120 - 80) = 0.5\n기초자산 0.5개 + 콜 1개 매도\n→ 오르면 60 - 20 = 40, 내리면 40 - 0 = 40\n→ 결과가 같음 = 무위험!", fontsize=8.5,
           bbox=dict(fc="#F3F6F1", ec=G, boxstyle="round,pad=0.4"))
    a.set_xlim(-.8, 8.2); a.set_ylim(-1.6, 1.6); save(f, "bs_binomial")


def call_curve():
    S = np.linspace(60, 150, 300); f, a = plt.subplots(figsize=(6.4, 3.0))
    a.plot(S, np.maximum(S - K0, 0), c=RED, lw=2.2, label="만기 손익 max(S-K, 0) (내재가치)")
    for t, c in [(0.1, GOLD), (0.5, BLUE), (1.5, G)]:
        a.plot(S, O.bs_price(S, K0, t, R, SIG, True), c=c, lw=1.6, label=f"오늘 가치, 만기까지 {t}년")
    a.fill_between(S, np.maximum(S - K0, 0), O.bs_price(S, K0, 1.5, R, SIG, True), color=G, alpha=.08)
    a.text(66, 22, "곡선 - 빨간 선 = 시간가치", color=G, fontsize=8.5)
    a.set_xlabel("현재 가격 S"); a.set_ylabel("콜 가치"); a.legend(frameon=False, fontsize=7.5); f.tight_layout(); save(f, "bs_call_curve")


def params():
    base = dict(S=S0, K=K0, T=T0, r=R, sig=SIG)
    grid = {"S": np.linspace(60, 140, 100), "K": np.linspace(60, 140, 100), "sig": np.linspace(.05, .8, 100), "r": np.linspace(0, .1, 100), "T": np.linspace(.02, 2, 100)}
    lab = {"S": "현재 가격 S", "K": "행사가 K", "sig": "변동성 σ", "r": "무위험 금리 r", "T": "만기까지 T (년)"}
    f, ax = plt.subplots(1, 5, figsize=(13, 2.7))
    for a, k in zip(ax, grid):
        x = grid[k]; kw = {**base, k: x}
        a.plot(x * (100 if k in ("sig", "r") else 1), O.bs_price(kw["S"], kw["K"], kw["T"], kw["r"], kw["sig"], True), c=G, lw=2, label="콜")
        a.plot(x * (100 if k in ("sig", "r") else 1), O.bs_price(kw["S"], kw["K"], kw["T"], kw["r"], kw["sig"], False), c=BLUE, lw=2, label="풋")
        a.axvline(base[k] * (100 if k in ("sig", "r") else 1), c=GREY, ls=":", lw=.8)
        a.set_xlabel(lab[k] + (" (%)" if k in ("sig", "r") else ""))
    ax[0].set_ylabel("옵션 가격"); ax[0].legend(frameon=False, fontsize=8)
    f.suptitle("한 가지만 바꾸고 나머지는 기준값(S=100, K=100, σ=25%, r=4%, T=0.5년)으로 고정", fontsize=9.5, y=1.03)
    f.tight_layout(); save(f, "bs_params")


def greeks():
    S = np.linspace(60, 150, 300); f, ax = plt.subplots(1, 4, figsize=(13, 2.7))
    for t, c in [(0.1, GOLD), (0.5, BLUE), (1.5, G)]:
        g = O.greeks(S, K0, t, R, SIG, True)
        for a, k in zip(ax, ["델타", "감마", "베가", "세타"]):
            a.plot(S, g[k], c=c, lw=1.6, label=f"T = {t}년")
    for a, k, u in zip(ax, ["델타", "감마", "베가", "세타"], ["가격 1당", "가격 1당 델타 변화", "σ 1%p당", "하루당"]):
        a.set_title(f"콜 {k} ({u})", fontsize=10); a.axvline(K0, c=GREY, ls=":", lw=.8); a.set_xlabel("현재 가격 S")
    ax[0].legend(frameon=False, fontsize=7.5); f.tight_layout(); save(f, "bs_greeks")


def iv_invert():
    sg = np.linspace(.02, .8, 200); mkt = 9.5; iv = O.implied_vol(mkt, S0, K0, T0, R, True)
    f, a = plt.subplots(figsize=(6.0, 2.9))
    a.plot(sg * 100, O.bs_price(S0, K0, T0, R, sg, True), c=G, lw=2.2, label="블랙숄즈 콜 가격(σ)")
    a.axhline(mkt, c=RED, ls="--", lw=1.2, label=f"시장 콜 가격 {mkt}"); a.axvline(iv * 100, c=GOLD, ls=":", lw=1.5)
    a.text(iv * 100 + 1, 2, f"내재변동성 = {iv:.1%}", color=GOLD); a.set_xlabel("변동성 σ (%)"); a.set_ylabel("콜 가격")
    a.legend(frameon=False, fontsize=8); f.tight_layout(); save(f, "bs_iv_invert")


def smile():
    m = np.linspace(-.3, .3, 60); f, ax = plt.subplots(1, 3, figsize=(11, 2.8), sharey=True)
    for a, (y, t) in zip(ax, [(.21 + 0 * m, "블랙숄즈 가정: 평평"), (.21 - .02 * m + 1.0 * m ** 2, "스마일 (금·은에서 흔함)"),
                              (.50 + .15 * m + .4 * m ** 2, "스큐 (한쪽이 가파름, 원유 등)")]):
        a.plot(m * 100, y * 100, c=G, lw=2.2); a.set_title(t, fontsize=10); a.set_xlabel("행사가 위치 ln(K/S) (%)")
    ax[0].set_ylabel("내재변동성 (%)"); f.tight_layout(); save(f, "bs_smile")


def surface3d():
    Sg, Tg = np.meshgrid(np.linspace(60, 150, 50), np.linspace(.02, 2, 40)); C = O.bs_price(Sg, K0, Tg, R, SIG, True)
    f = plt.figure(figsize=(6.6, 4.0)); a = f.add_subplot(111, projection="3d")
    a.plot_surface(Sg, Tg, C, cmap="cividis", alpha=.92, linewidth=0); s = np.linspace(60, 150, 100)
    a.plot(s, np.zeros_like(s), np.maximum(s - K0, 0), c=RED, lw=2.5)
    a.set_xlabel("현재 가격 S"); a.set_ylabel("만기까지 T (년)"); a.set_zlabel("콜 가격"); a.view_init(25, -60)
    a.set_title("콜 가격 곡면 — T→0이면 빨간 만기 손익선에 붙음", fontsize=10); save(f, "bs_surface3d")


def hedge():
    rng = np.random.default_rng(5); steps = 60; dt = T0 / steps
    S = S0 * np.exp(np.cumsum(np.r_[0, (R - SIG ** 2 / 2) * dt + SIG * np.sqrt(dt) * rng.standard_normal(steps)]))
    t = np.linspace(0, T0, steps + 1); Tl = np.maximum(T0 - t, 1e-6)
    d = O.greeks(S, K0, Tl, R, SIG, True)["델타"]
    f, ax = plt.subplots(1, 2, figsize=(9.5, 2.8))
    ax[0].plot(t, S, c=G, lw=1.6); ax[0].axhline(K0, c=RED, ls=":", lw=1); ax[0].set_title("기초자산 가격 경로", fontsize=10); ax[0].set_xlabel("시간 (년)")
    ax[1].plot(t, d, c=BLUE, lw=1.6); ax[1].set_ylim(-.05, 1.05); ax[1].set_title("헤지에 필요한 보유량 Δ — 가격 따라 매일 조정", fontsize=10); ax[1].set_xlabel("시간 (년)")
    f.tight_layout(); save(f, "bs_hedge")


if __name__ == "__main__":
    for fn in (payoff, procure, normal, sqrt_t, vol_drag, lognormal, binomial, call_curve, params, greeks, iv_invert, smile, surface3d, hedge):
        fn(); print("saved", fn.__name__)
