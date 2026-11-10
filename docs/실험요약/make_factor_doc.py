"""요인 모형 → 포트폴리오 연결 쉬운 설명 Word 문서 (그림 matplotlib + 본문 python-docx). 숫자는 data/strategy_factor_portfolio_compare.csv와 라이브 PCA에서 직접 읽음.
실행: .venv\\Scripts\\python.exe docs\\실험요약\\make_factor_doc.py → docs/실험요약/요인모형_포트폴리오_연결_쉬운설명_20261110.docx (+ figs_factor/*.png)
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import make_daily_doc as M  # noqa: E402  (Doc 클래스 · 색 · 글꼴 재사용)
import factor_portfolio as FP  # noqa: E402
import strategy_cov_factor as CF  # noqa: E402

FIG = HERE / "figs_factor"; FIG.mkdir(exist_ok=True); M.FIG = FIG
OUT = HERE / "요인모형_포트폴리오_연결_쉬운설명_20261110.docx"
INK, GRN, BLUE, RED, GOLD, GREY = M.INK, M.GRN, M.BLUE, M.RED, M.GOLD, M.GREY
BASE_TR, BASE_TE = 0.61, 1.07                                   # 주간 3축 고정 50/25/25 (학습 · 시험)
NM = {"yf:ZC=F": "옥수수", "yf:ZW=F": "밀", "yf:ZS=F": "대두", "yf:ZM=F": "대두박", "yf:ZL=F": "대두유", "yf:CL=F": "WTI 원유", "yf:HO=F": "난방유", "yf:RB=F": "휘발유", "yf:NG=F": "천연가스",
      "yf:HG=F": "구리", "yf:GC=F": "금", "yf:SI=F": "은", "yf:PL=F": "백금", "yf:KC=F": "커피", "yf:SB=F": "설탕", "yf:CT=F": "면화", "yf:CC=F": "코코아", "yf:LE=F": "생우", "yf:HE=F": "돈육"}
SEC = {"곡물": ["yf:ZC=F", "yf:ZW=F", "yf:ZS=F", "yf:ZM=F", "yf:ZL=F"], "에너지": ["yf:CL=F", "yf:HO=F", "yf:RB=F", "yf:NG=F"], "금속": ["yf:HG=F", "yf:GC=F", "yf:SI=F", "yf:PL=F"],
       "소프트": ["yf:KC=F", "yf:SB=F", "yf:CT=F", "yf:CC=F"], "축산": ["yf:LE=F", "yf:HE=F"]}
SCOL = {"곡물": GOLD, "에너지": RED, "금속": BLUE, "소프트": GRN, "축산": GREY}
FC = pd.read_csv(ROOT / "data" / "strategy_factor_portfolio_compare.csv", encoding="utf-8-sig")
TR, TE = "Sharpe 학습 2008–16", "Sharpe 시험 2017–"


def row(fam, key):
    x = FC[FC["묶음"].str.startswith(fam) & FC["구성"].str.contains(key, regex=False)]
    return x.iloc[0]


# ---------------------------------------------------------------- 그림
def fig_map():
    f, ax = plt.subplots(figsize=(11, 4.4)); ax.set_xlim(0, 11); ax.set_ylim(0, 4.4); ax.axis("off")
    M.box(ax, 0.1, 2.6, 3.0, 1.2, "요인 모형\n(PCA · FAVAR)\n'움직임의 원인'을 뽑는 도구", BLUE)
    M.box(ax, 4.0, 3.2, 3.0, 0.9, "길 1  설명 · 진단\n지금 포트폴리오 위험이\n어느 요인에 몰렸나", GRN)
    M.box(ax, 4.0, 1.6, 3.0, 1.2, "길 2  비중 만들기\nA 공분산 · B PC 포트폴리오\nC 잠재 요인 · D 거시 요인", RED)
    M.arrow(ax, 3.2, 3.3, 3.95, 3.65); M.arrow(ax, 3.2, 3.0, 3.95, 2.3)
    M.box(ax, 7.9, 3.2, 3.0, 0.9, "채택  (대시보드 ①+ 요인 모형\n› C 서브탭 ① 요인 위험 분해)", GRN)
    M.box(ax, 7.9, 1.6, 3.0, 1.2, "미채택\n50/25/25보다 모두 못함", GREY)
    M.arrow(ax, 7.1, 3.65, 7.85, 3.65); M.arrow(ax, 7.1, 2.2, 7.85, 2.2)
    ax.text(0.1, 0.9, "질문: 요인 모형은 설명에만 쓰이고 포트폴리오와 이어지는 곳이 없었다 → 이어 보자", fontsize=10.5, color=INK, fontweight="bold")
    ax.text(0.1, 0.4, "방법: 같은 백테스트 틀(4주마다 재계산 · 비용 5bp · 목표 변동성 10% · 학습 2008–16 / 시험 2017–)에서 기준 50/25/25와 비교", fontsize=9.5, color="#555555")
    f.tight_layout(); f.savefig(FIG / "g01_map.png", dpi=170); plt.close(f)


def fig_loadings(R):
    X = R.iloc[-104:]
    lam, V, sd, mu = FP.pca_std(X)
    ids = [c for s in SEC.values() for c in s]
    f, axs = plt.subplots(1, 3, figsize=(11, 4.6), sharey=True)
    for k, ax in enumerate(axs):
        v = pd.Series(V[:, k], index=X.columns).reindex(ids)
        ax.barh(range(len(ids))[::-1], v.values, color=[SCOL[next(s for s, m in SEC.items() if i in m)] for i in ids], height=0.7)
        ax.axvline(0, color=INK, lw=0.8); ax.set_yticks(range(len(ids))[::-1]); ax.set_yticklabels([NM[i] for i in ids], fontsize=8.5)
        ax.set_title(f"PC{k + 1} · 시장 변동의 {lam[k] / lam.sum():.0%}", loc="left", fontsize=10, fontweight="bold"); ax.set_xlim(-0.65, 0.65)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SCOL.values()]
    f.legend(handles, list(SCOL), ncol=5, loc="lower center", frameon=False, fontsize=9)
    f.suptitle("PCA가 찾은 '움직임의 묶음' — 막대 방향이 같으면 같이 움직이는 품목 (최근 104주)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    f.tight_layout(rect=(0, 0.05, 1, 0.95)); f.savefig(FIG / "g02_loadings.png", dpi=170); plt.close(f)
    return lam, V


def fig_decomp(o):
    f, ax = plt.subplots(figsize=(10, 3.9))
    t = o["full"]; K = 3
    labs = ["PC1 (공통)", "PC2", "PC3", "나머지 16개"]
    mk = [*(t["시장 설명력"].iloc[:K]), t["시장 설명력"].iloc[K:].sum()]; pf = [*(t["분산 기여"].iloc[:K]), t["분산 기여"].iloc[K:].sum()]
    x = np.arange(4); w = 0.38
    ax.bar(x - w / 2, mk, w, color=GREY, label="시장 전체 변동에서 차지하는 몫")
    ax.bar(x + w / 2, pf, w, color=RED, label="내 포트폴리오 분산에서 차지하는 몫")
    for xx, a, b in zip(x, mk, pf):
        ax.text(xx - w / 2, a + 0.012, f"{a:.0%}", ha="center", fontsize=9); ax.text(xx + w / 2, b + 0.012, f"{b:.0%}", ha="center", fontsize=9, color=RED, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylim(0, 0.8); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); ax.legend(frameon=False, loc="upper right")
    ax.set_title("기본 혼합(50/25/25)의 위험은 어디서 오나 — 공통 요인 PC1에 몰림", loc="left", fontsize=11, fontweight="bold")
    f.tight_layout(); f.savefig(FIG / "g03_decomp.png", dpi=170); plt.close(f)


def fig_enb(o):
    f, ax = plt.subplots(figsize=(10, 3.0)); ax.set_xlim(0, 20); ax.set_ylim(0, 3); ax.axis("off")
    ax.text(0, 2.6, "품목 수 19개 (겉보기 분산)", fontsize=10, fontweight="bold", color=INK)
    for i in range(19):
        ax.add_patch(plt.Circle((0.6 + i * 1.0, 2.0), 0.36, color=BLUE))
    ax.text(0, 1.1, f"실제로 서로 다른 위험 베팅 수 (유효 요인 수, ENB) = {o['enb']:.1f}", fontsize=10, fontweight="bold", color=RED)
    for i in range(4):
        ax.add_patch(plt.Circle((0.6 + i * 1.0, 0.4), 0.36, color=RED))
    ax.add_patch(plt.Circle((4.6, 0.4), 0.36 * 0.1 ** 0.5, color=RED))
    ax.text(6.0, 0.35, "← 19개 품목이어도 위험 기준으로는 약 4개 묶음과 같다", fontsize=9.5, color="#555555", va="center")
    f.tight_layout(); f.savefig(FIG / "g04_enb.png", dpi=170); plt.close(f)


def fig_paths():
    items = [("기준 (3축 혼합 50/25/25)", BASE_TE, None),
             ("B PC1 포트폴리오", row("F1 PC 포트폴리오", "PC1")["혼합 시험"], None), ("B PC2 (부호 고정)", row("F1 PC 포트폴리오", "PC2 · 부호 고정")["혼합 시험"], None),
             ("B PC3 (부호 고정)", row("F1 PC 포트폴리오", "PC3 · 부호 고정")["혼합 시험"], None),
             ("C PCA 요인 3개", row("F2 잠재 요인 평균-분산", "PCA · 요인 3개")["혼합 시험"], None), ("C RP-PCA γ=0 · 3개", row("F2 잠재 요인 평균-분산", "γ=0 · 요인 3개")["혼합 시험"], None),
             ("C RP-PCA γ=20 · 3개 (학습 최고)", row("F2 잠재 요인 평균-분산", "γ=20 · 요인 3개")["혼합 시험"], None), ("D 거시 요인 → 기대수익", row("F3 거시 요인", "릿지")["혼합 시험"], None)]
    f, ax = plt.subplots(figsize=(10, 4.2)); y = np.arange(len(items))[::-1]
    cols = [INK] + [RED] * (len(items) - 1)
    ax.barh(y, [v for _, v, _ in items], color=cols, height=0.62)
    for yy, (_, v, _) in zip(y, items):
        ax.text(v - 0.012, yy, f"{v:.2f}", va="center", ha="right", color="white", fontsize=9, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([n for n, _, _ in items], fontsize=9); ax.axvline(BASE_TE, color=INK, ls="--", lw=1.1); ax.set_xlim(0, 1.25)
    ax.set_xlabel("시험 기간(2017–) Sharpe — 가치 자리(25%)에 이 전략을 넣은 혼합 (기준은 원래 3축 혼합)")
    ax.set_title("4가지 길 모두 기준 1.07보다 낮다", loc="left", fontsize=11, fontweight="bold")
    f.tight_layout(); f.savefig(FIG / "g05_paths.png", dpi=170); plt.close(f)


def grouped(ax, labels, tr, te, base=True):
    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w / 2, tr, w, color=BLUE, label="학습 2008–16"); ax.bar(x + w / 2, te, w, color=GRN, label="시험 2017–")
    for xx, a, b in zip(x, tr, te):
        ax.text(xx - w / 2, a + (0.03 if a >= 0 else -0.09), f"{a:+.2f}", ha="center", fontsize=8.5); ax.text(xx + w / 2, b + (0.03 if b >= 0 else -0.09), f"{b:+.2f}", ha="center", fontsize=8.5)
    ax.axhline(0, color=INK, lw=0.8); ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.8)
    if base:
        ax.axhline(BASE_TE, color=INK, ls="--", lw=1); ax.text(len(labels) - 0.5, BASE_TE + 0.03, f"기준 시험 {BASE_TE:.2f}", ha="right", fontsize=8.5, color=INK)
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)


def fig_pc():
    rs = [row("F1 PC 포트폴리오", "PC1"), row("F1 PC 포트폴리오", "PC2 · 부호 고정"), row("F1 PC 포트폴리오", "PC2 · 부호 = 직전"), row("F1 PC 포트폴리오", "PC3 · 부호 고정"), row("F1 PC 포트폴리오", "PC3 · 부호 = 직전")]
    labs = ["PC1", "PC2\n(부호 고정)", "PC2\n(승자 방향)", "PC3\n(부호 고정)", "PC3\n(승자 방향)"]
    f, ax = plt.subplots(figsize=(10, 3.9)); grouped(ax, labs, [r[TR] for r in rs], [r[TE] for r in rs])
    ax.annotate("학습에서 가장 좋았는데\n시험에서 -0.85로 뒤집힘", xy=(3.0, -0.82), xytext=(1.45, -0.8), fontsize=9, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_ylim(-1.15, 1.2); ax.set_title("B. PC 포트폴리오 — 단독 Sharpe (학습 vs 시험)", loc="left", fontsize=11, fontweight="bold")
    f.tight_layout(); f.savefig(FIG / "g06_pc.png", dpi=170); plt.close(f)


def fig_lat():
    f, axs = plt.subplots(1, 1, figsize=(10, 3.9)); labs, tr, te = [], [], []
    for nm, key in [("PCA", "PCA · 요인"), ("RP-PCA\nγ=0", "γ=0 · 요인"), ("RP-PCA\nγ=20", "γ=20 · 요인")]:
        for k in (1, 2, 3, 5):
            r = row("F2 잠재 요인 평균-분산", f"{key} {k}개"); labs.append(f"{nm}\n{k}개"); tr.append(r[TR]); te.append(r[TE])
    grouped(axs, labs, tr, te); axs.set_ylim(-0.1, 1.25)
    axs.set_title("C. 잠재 요인 평균-분산 — 방식 × 요인 개수별 단독 Sharpe", loc="left", fontsize=11, fontweight="bold")
    f.tight_layout(); f.savefig(FIG / "g07_latent.png", dpi=170); plt.close(f)


def fig_macro():
    r = row("F3 거시 요인", "릿지")
    f, axs = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.2, 1]})
    grouped(axs[0], ["거시 요인 → 순위 롱숏"], [r[TR]], [r[TE]], base=False); axs[0].set_ylim(-0.5, 0.6); axs[0].set_title("학습 vs 시험 Sharpe", loc="left", fontsize=10, fontweight="bold")
    ax = axs[1]; ax.barh([0], [r["순위 IC"]], color=GREY, height=0.4); ax.barh([1], [0.05], color=GRN, height=0.4)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["이번 결과\n(IC +0.004)", "쓸 만한 신호\n(IC 약 0.05 이상)"]); ax.set_xlim(0, 0.07)
    ax.set_title("예측과 실제 순위의 상관(IC)", loc="left", fontsize=10, fontweight="bold")
    f.tight_layout(); f.savefig(FIG / "g08_macro.png", dpi=170); plt.close(f)


def fig_pipe():
    f, ax = plt.subplots(figsize=(11, 2.8)); ax.set_xlim(0, 11); ax.set_ylim(0, 2.8); ax.axis("off")
    xs = [0.1, 2.35, 4.6, 6.85, 9.0]; tt = ["거시 변수 21개\n(VIX · 금리 · 달러 …)\n주간 변화, 1주 늦춰 사용", "PCA로 3개 요인\n(확장창, 그 시점까지\n자료만)", "품목별 릿지 회귀\n4주 뒤 수익 ~ 요인", "예측 수익 μ의\n순위", "위쪽 사고\n아래쪽 팔기\n(롱숏)"]
    for x, t, c in zip(xs, tt, [BLUE, BLUE, GRN, GRN, "#6B4F9E"]):
        M.box(ax, x, 0.8, 1.95, 1.4, t, c, fs=8.5)
    for a, b in zip(xs[:-1], xs[1:]):
        M.arrow(ax, a + 1.97, 1.5, b - 0.02, 1.5)
    ax.text(0.1, 0.25, "D 길(FAVAR-lite): 요인은 '예측 재료'일 뿐, 결과가 신호가 되려면 마지막 순위가 실제 수익 순위와 맞아야 한다 → 맞지 않았다 (IC +0.004)", fontsize=9, color=RED)
    f.tight_layout(); f.savefig(FIG / "g09_pipe.png", dpi=170); plt.close(f)


def fig_bc():
    f, ax = plt.subplots(figsize=(11, 5.2)); ax.set_xlim(0, 11); ax.set_ylim(0, 5.2); ax.axis("off")
    M.box(ax, 0.1, 2.0, 2.3, 1.3, "품목 수익률 19개\n표준화 Z = (R - 평균) ÷ σ\n→ PCA", BLUE, fs=8.8)
    M.box(ax, 3.1, 2.1, 2.1, 1.1, "고유벡터 v₁ v₂ v₃ …\n(품목별 적재값)", BLUE, fs=8.8)
    M.arrow(ax, 2.45, 2.65, 3.05, 2.65)
    ax.text(5.5, 4.85, "B. PC 포트폴리오", fontsize=11, fontweight="bold", color=GRN)
    M.box(ax, 5.5, 3.75, 2.4, 0.85, "PC 하나를 고름\n(예: PC1)", GRN, fs=8.8); M.box(ax, 8.3, 3.75, 2.6, 0.85, "비중 = v ÷ σ → 총노출 1\n(= 그 요인의 수익을 그대로 재현)", GRN, fs=8.3)
    M.arrow(ax, 5.25, 2.95, 5.45, 4.0); M.arrow(ax, 7.95, 4.18, 8.25, 4.18)
    ax.text(5.5, 2.55, "C. 잠재 요인 평균-분산", fontsize=11, fontweight="bold", color=RED)
    M.box(ax, 5.5, 1.35, 2.4, 0.95, "K개 요인의 수익률 F_k\n(= B 포트폴리오들)\n을 '자산'으로 봄", RED, fs=8.3); M.box(ax, 8.3, 1.35, 2.6, 0.95, "요인 비중 w_f = μ_f ÷ σ_f²\n(평균 × 0.5, 요인끼리 무상관)", RED, fs=8.3)
    M.arrow(ax, 5.25, 2.35, 5.45, 1.85); M.arrow(ax, 7.95, 1.82, 8.25, 1.82)
    M.box(ax, 8.3, 0.2, 2.6, 0.8, "품목 비중 = V·w_f ÷ σ\n→ 총노출 1", RED, fs=8.5); M.arrow(ax, 9.6, 1.3, 9.6, 1.05)
    ax.text(0.1, 1.2, "RP-PCA는 B · C 와 다른 축 — '요인을 어떻게 뽑나'의 선택:\n  PCA = 공분산만 보고 뽑음 · RP-PCA = 공분산 + 평균(μμ′)도 보고 뽑음\n(B · C 모두 PCA나 RP-PCA 어느 쪽 요인으로도 만들 수 있음)", fontsize=8.8, color="#555555", va="top")
    ax.text(0.1, 0.1, "관계: C에서 요인 1개(K=1) = B의 PC1 포트폴리오(부호만 평균으로 결정). C는 B 포트폴리오들을 평균과 분산으로 섞은 것.", fontsize=9, color=INK, fontweight="bold")
    f.tight_layout(); f.savefig(FIG / "g10_bc.png", dpi=170); plt.close(f)


SC = pd.read_csv(ROOT / "data" / "strategy_spectral_coherence.csv", encoding="utf-8-sig")
ST = pd.read_csv(ROOT / "data" / "strategy_spectral_compare.csv", encoding="utf-8-sig")
SP = pd.read_csv(ROOT / "data" / "strategy_spectral_placebo.csv", encoding="utf-8-sig")


def coh_rows(dat="원 수익률", per="전체 2005–", bd="B1"):
    return SC[(SC["자료"] == dat) & (SC["구간"] == per) & SC["밴드 정의"].str.startswith(bd)].set_index("성분")


def fig_coh():
    f, axs = plt.subplots(1, 3, figsize=(11, 3.8))
    for bd, ax, ttl in (("B1", axs[0], "8주 · 26주 기준"), ("B2", axs[1], "6개월 · 18개월 기준")):
        X = coh_rows(bd=bd); x = np.arange(3); w = 0.38
        ax.bar(x - w / 2, X["Coh2"], w, color=RED, label="실제 평균 Coh²"); ax.bar(x + w / 2, X["영가설Coh2"], w, color=GREY, label="영가설(상호 의존 제거)")
        for xx, a, b in zip(x, X["Coh2"], X["영가설Coh2"]):
            ax.text(xx - w / 2, a + 0.005, f"{a:.2f}", ha="center", fontsize=8.5); ax.text(xx + w / 2, b + 0.005, f"{b:.2f}", ha="center", fontsize=8.5)
        ax.set_xticks(x); ax.set_xticklabels(["단기", "중기", "장기"]); ax.set_ylim(0, 0.38); ax.set_title(f"평균 코히어런스 · {ttl}", loc="left", fontsize=9.5, fontweight="bold")
        if bd == "B1":
            ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axs[2]; x = np.arange(3); w = 0.38
    ax.bar(x - w / 2, coh_rows(bd="B1")["정보밀도"], w, color=BLUE, label="8주 · 26주"); ax.bar(x + w / 2, coh_rows(bd="B2")["정보밀도"], w, color=GOLD, label="6개월 · 18개월")
    for xx, a, b in zip(x, coh_rows(bd="B1")["정보밀도"], coh_rows(bd="B2")["정보밀도"]):
        ax.text(xx - w / 2, a + 0.2, f"{a:.1f}×", ha="center", fontsize=8.5); ax.text(xx + w / 2, b + 0.2, f"{b:.1f}×", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(["단기", "중기", "장기"]); ax.set_title("정보 밀도 (Coh² ÷ 에너지 몫)", loc="left", fontsize=9.5, fontweight="bold"); ax.legend(frameon=False, fontsize=8, loc="upper left")
    f.tight_layout(); f.savefig(FIG / "g11_coh.png", dpi=170); plt.close(f)


def fig_spec():
    f, axs = plt.subplots(1, 2, figsize=(11, 3.9), sharey=False)
    labs = [f"{r['밴드']}\n{r['창(주)']}주" for _, r in SP[SP["지표"] == "MinVar 축 시험"].iterrows()]
    for ax, key, ttl in ((axs[0], "MinVar 축 시험", "MinVar 축 단독 시험 Sharpe"), (axs[1], "혼합 시험", "3축 혼합 시험 Sharpe")):
        S = SP[SP["지표"] == key]; x = np.arange(len(S)); w = 0.26
        ax.bar(x - w, S["기준(LW)"], w, color=GREY, label="기준 (같은 창 Ledoit-Wolf)"); ax.bar(x, S["위약 평균"], w, color=GOLD, label="위약 평균 (칸 무작위)"); ax.bar(x + w, S["장기 강조 실제"], w, color=GRN, label="장기 강조 (0.5, 1, 3)")
        for xx, a, b, c in zip(x, S["기준(LW)"], S["위약 평균"], S["장기 강조 실제"]):
            for dx, v in ((-w, a), (0, b), (w, c)):
                ax.text(xx + dx, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylim(0.7, S[["기준(LW)", "위약 평균", "장기 강조 실제"]].values.max() + 0.12); ax.set_title(ttl, loc="left", fontsize=10, fontweight="bold")
        if key == "MinVar 축 시험":
            ax.legend(frameon=False, fontsize=8, loc="upper right")
    f.tight_layout(); f.savefig(FIG / "g12_spec.png", dpi=170); plt.close(f)


def SPECTRAL_CHAPTER(d):
    d.pb()
    d.h("6. 주파수(Spectral) 분석과 Spectral 포트폴리오 시험", 1)
    d.p("요인 모형의 또 다른 확장은 '시간'이 아니라 '주기'로 나누는 것입니다. 원자재의 움직임을 짧은 주기(단기 수급 충격 · 급등락), 중간 주기(경기 순환 · 수요 변화), 긴 주기(구조적 공급 · 수요 변화)로 나누어 보면, 투자 기간에 따라 위험의 성격이 다를 수 있습니다.")
    d.box("쉬운 비유", ["라디오 소리를 저음 · 중음 · 고음으로 나눠 듣는 것과 같습니다. 가격 움직임에서 '빠르게 흔들리는 부분'과 '천천히 흐르는 부분'을 따로 떼어 서로 얼마나 같이 움직이는지 봅니다."], fill="FBF5E6", bar="#8A6D1E")
    d.h("6-1. 용어", 2)
    d.table(["용어", "쉬운 설명"], [
        ["주파수 · 밴드", "움직임이 한 번 오르내리는 데 걸리는 시간(주기)별 구간. 단기 = 8주 미만, 중기 = 8~26주, 장기 = 26주 이상(밴드 정의 B1). 6개월 · 18개월 기준(B2)도 함께 확인."],
        ["에너지 몫", "그 주기 성분이 전체 변동에서 차지하는 비중. 가격 움직임 대부분은 단기에서 옵니다."],
        ["코히어런스 (Coh²)", "두 품목이 그 주기에서 얼마나 같이 움직이는지(0~1). 0이면 무관, 1이면 완전히 같이 움직임. 모든 품목 쌍의 평균을 씁니다."],
        ["정보 밀도", "평균 Coh² ÷ 에너지 몫. 1보다 크면 '에너지는 작은데 같이 움직임은 큰' 주기입니다."],
        ["영가설 Coh²", "품목마다 시계열을 임의로 밀어 상호 관계만 없앤 값. 추정 편향의 바닥으로 보고 실제에서 뺍니다(초과 Coh²)."]], [3.6, 13.6], size=9.5)
    d.h("6-2. 우리 자료의 코히어런스 · 정보 밀도", 2)
    d.p("다른 연구에서 '장기 주파수일수록 평균 코히어런스와 정보 밀도가 높다'는 결과가 나와서, 우리 원자재 18개 품목 주간 자료(2005–)에서 따로 계산했습니다.")
    d.img("g11_coh.png", 17.0, "그림 10. 밴드별 평균 코히어런스(실제 vs 영가설)와 정보 밀도")
    X1, X2 = coh_rows(bd="B1"), coh_rows(bd="B2")
    d.table(["밴드 정의", "성분", "에너지 몫", "평균 Coh²", "영가설", "초과", "정보 밀도"], [[bd, n, f"{r['에너지몫']:.0%}", f"{r['Coh2']:.3f}", f"{r['영가설Coh2']:.3f}", f"{r['초과Coh2']:+.3f}", f"{r['정보밀도']:.1f}×"]
                                                                  for bd, X in (("8주 · 26주", X1), ("6개월 · 18개월", X2)) for n, r in X.iterrows()], [3.0, 2.0, 2.2, 2.5, 2.2, 2.2, 2.6], size=9, center_from=2)
    d.b([("확인된 것: ", ), "우리 자료에서도 장기일수록 품목이 더 같이 움직입니다(평균 Coh² 0.245 → 0.287, 6개월/18개월 정의에서는 0.244 → 0.322). 원 수익률과 VAR(1) 잔차로 계산해도 거의 같습니다."])
    d.b([("조심할 것: ", ), "장기는 주파수 칸이 적어 영가설 값도 같이 올라갑니다(0.205 → 0.214 ~ 0.223). 초과분만 보면 0.039 → 0.073(또는 0.100)으로 차이가 줄어듭니다. 정보 밀도가 3.8×나 13×로 커 보이는 것은 장기의 에너지 몫이 작다는 분모 효과가 큽니다."])
    d.b([("변동성은 단기에 몰립니다: ", ), "분산의 75%(6개월 기준으로는 92%)가 단기이며, 이는 무작위일 때 기대하는 몫과 같습니다. 다만 장기에는 PC1 쏠림이 가장 커서(66%) 기간이 길수록 분산이 덜 됩니다(대시보드 D › ④)."])
    d.h("6-3. Spectral 포트폴리오 시험", 2)
    d.p("코히어런스가 장기에서 높다면, 장기 성분을 더 중시한 공분산으로 MinVar를 풀면 개선될지 시험했습니다. MinVar 축 자리에 넣고, 같은 창의 Ledoit-Wolf MinVar를 기준으로 비교했습니다(4주마다 재계산, 품목 상한 25%, 비용 5bp, 목표 변동성 10%).")
    d.table(["묶음", "하는 일 (쉬운 말)"], [
        ["S1 주파수 가중 공분산", "창 수익률을 단기 · 중기 · 장기 성분으로 나눠 λ배씩 다시 합친 뒤 Ledoit-Wolf → MinVar. '장기 강조' = (0.5, 1, 3), '장기 집중' = (0.1, 0.3, 3), 단기 · 중기 강조도 시험."],
        ["S2 주파수별 요인 공분산", "밴드마다 상위 K개 PC로 공통 부분을 만들어 합산하고 품목 고유 위험을 더함 (밴드마다 요인 모양이 다름)."],
        ["S3 기간별 위험 배분", "밴드별 공분산만 쓴 MinVar 3개를 균등 혼합(또는 한 밴드만 사용) — '투자 기간별 위험 배분'."]], [4.0, 13.2], size=9.5)
    d.img("g12_spec.png", 17.0, "그림 11. S1 장기 강조(초록) vs 같은 창 Ledoit-Wolf(회색) vs 주파수 칸 무작위 위약(노랑) — 시험 Sharpe")
    d.table(["설정 (창 · 밴드)", "기준 LW 시험", "장기 강조 시험", "혼합 시험 (기준 → 장기 강조)", "위약 p (시험)"],
            [[f"{r['창(주)']}주 · {r['밴드']}", f"{r['기준(LW)']:+.2f}", f"{r['장기 강조 실제']:+.2f}", f"{SP[(SP['밴드'] == r['밴드']) & (SP['창(주)'] == r['창(주)']) & (SP['지표'] == '혼합 시험')].iloc[0]['기준(LW)']:.2f} → {SP[(SP['밴드'] == r['밴드']) & (SP['창(주)'] == r['창(주)']) & (SP['지표'] == '혼합 시험')].iloc[0]['장기 강조 실제']:.2f}", f"{r['위약 p']:.2f}"]
             for _, r in SP[SP["지표"] == "MinVar 축 시험"].iterrows()], [4.0, 3.0, 3.2, 4.4, 2.6], size=9, center_from=1)
    d.b([("방향은 일관됩니다: ", ), "창 260주 이상에서 장기 강조는 학습 · 시험 모두 같은 창 기준보다 소폭 좋았고, 주파수 칸을 무작위로 섞은 위약의 평균보다도 높았습니다(12개 지표 모두, 12개 중 10개는 위약 95% 상한 이상)."])
    d.b([("하지만 크기가 작습니다: ", ), "시험 Sharpe +0.03 ~ +0.10, 차이 t +1.57 · +1.24 · +0.32로 우연과 구별하기 어렵고, 104주 창에서는 오히려 나빠집니다. 위험 예측 편향도 260주 창에서 1.10 → 1.28~1.31로 나빠져(위험을 더 과소 예측) 실제 운용에서는 불리합니다."])
    d.b([("창 길이 효과가 더 큽니다: ", ), "같은 Ledoit-Wolf로도 창을 104주 → 260주로 바꾸면 혼합 시험이 1.07 → 1.14로 오릅니다. 장기 강조의 추가 효과(+0.04 ~ +0.11)는 그 위의 작은 보탬입니다."])
    d.b([("S2 · S3: ", ), "주파수별 요인 공분산과 기간별 MinVar 혼합은 개선이 없거나 불안정했습니다. 장기 밴드 MinVar만 쓰면 일부 설정에서 학습은 좋지만 시험이 약해집니다."])
    d.box("판정", ["미채택(후속 확인 후보). '장기 주파수 강조'는 방향이 일관적이고 위약과도 어느 정도 구분되지만, 크기가 작고 통계적으로 유의하지 않으며 위험 예측 편향이 나빠집니다. 3축 혼합의 MinVar 축을 바꿀 근거로는 부족합니다. 주파수 정보는 지금은 '위험을 읽는 진단'(대시보드 D › ④)으로 쓰는 것이 맞습니다."], fill="F1F1F8", bar="#4A3F8F")
    d.p("이번에 시험하지 않은 것: 주파수별 요인으로 다른 자산군 · 투자 기간별 별도 포트폴리오를 운용하는 구조, 일간 자료의 더 짧은 주기, 거시 변수와의 교차 코히어런스(어느 거시 변수가 어느 주기에 품목과 같이 움직이나).", 9.5, color="5B6575")


def make_figs():
    L, S3, R, P = CF.load()
    w = pd.read_pickle(ROOT / "data" / "frozen" / "pf_current_weights_2026-10-26.pkl")
    o = FP.risk_decomposition(R, w, 3, 104)
    fig_map(); lam, V = fig_loadings(R); fig_decomp(o); fig_enb(o); fig_paths(); fig_pc(); fig_lat(); fig_macro(); fig_pipe(); fig_bc(); fig_coh(); fig_spec()
    # 섹터별 PC1 비중 (본문 표)
    sec_w = {s: float(w.reindex(m).fillna(0).sum()) for s, m in SEC.items()}
    return o, sec_w


# ---------------------------------------------------------------- 본문
def main():
    o, sec_w = make_figs()
    d = M.Doc()
    d.p("요인 모형은 포트폴리오와 어떻게 이어지나", 24, True, INK, 2)
    d.p("PCA · FAVAR로 비중을 짜 봤더니 — 결과와 대시보드에 넣은 것", 13, False, "5B6575", 6)
    d.p("2026-11-10 · 자료 기준일 2026-10-26 · 상세 기록: docs/백테스트_로그.md (U) · 대시보드 ①+ 요인 모형 › C 서브탭", 9.5, color="888888", after=10)
    d.box("이 문서가 하는 일", [
        "대시보드의 '요인 모형' 탭은 원자재들이 왜 같이 움직이는지를 '설명'하는 데서 끝나 있었습니다. 그래서 \"이 요인들을 포트폴리오 비중을 짜는 데 쓸 수 있을까?\"를 검증하고, 결과를 쉬운 말로 정리했습니다.",
        [("결론 먼저: ", ), f"① 요인으로 비중을 짜는 4가지 방법은 모두 기준 전략(MinVar 50 · 모멘텀 25 · 가치 25)을 이기지 못했습니다(미채택). ② 대신 PCA를 '내 포트폴리오의 위험이 어느 요인에 몰렸나'를 보는 진단 도구로 쓰면 유용했습니다. 지금 기본 혼합은 분산의 {o['pc1_share']:.0%}가 공통 요인 하나(PC1)에서 나옵니다."]])
    d.h("차례", 2)
    for t in ["0. 먼저 알아둘 말", "1. 요인 모형이 하는 일 (PCA 쉬운 설명)", "2. 길 1 — 내 포트폴리오의 위험이 어디에 몰렸나 (채택)", "3. 길 2 — 요인으로 비중을 짜 보면 (미채택)", "   A 공분산 · B PC 포트폴리오 · C 잠재 요인 평균-분산 · D 거시 요인 → 기대수익",
              "4. 그래서 50/25/25보다 나아지는가", "5. 대시보드에서 보는 법", "6. 주파수(Spectral) 분석과 Spectral 포트폴리오 시험", "7. 읽을 때 주의할 점과 다음 단계"]:
        d.p(t, 10, after=1)
    d.pb()
    # 0
    d.h("0. 먼저 알아둘 말", 1)
    d.table(["용어", "쉬운 설명"], [
        ["요인 (factor)", "여러 품목을 동시에 움직이게 하는 공통 원인. 예: '경기가 좋아지면 원자재가 다 같이 오른다'의 '경기'."],
        ["PCA (주성분 분석)", "19개 품목의 움직임을 '같이 움직이는 묶음'으로 정리하는 통계 방법. 가장 큰 묶음이 PC1, 그다음이 PC2, PC3 … 입니다. 묶음끼리는 서로 무관하게 만들어집니다."],
        ["적재값", "각 묶음에서 품목이 얼마나 같이 움직이는지. 막대가 같은 방향이면 같이 움직입니다."],
        ["분산 기여", "내 포트폴리오의 출렁임(분산) 중 각 요인이 만들어 내는 몫. 다 합치면 100%."],
        ["유효 요인 수 (ENB)", "위험이 몇 개의 서로 다른 묶음에 나뉘어 있는지의 숫자. 19개 품목이어도 위험이 한 곳에 몰리면 이 숫자가 작습니다."],
        ["RP-PCA", "공분산(출렁임)에 '평균 수익'도 함께 보도록 고친 PCA(Lettau–Pelger, 2020). 출렁임은 작아도 꾸준히 번 묶음을 놓치지 않으려는 방법입니다."],
        ["FAVAR", "요인(PCA로 뽑은 숫자)과 여러 변수를 한 모형에서 함께 움직여 보는 방법. 여기서는 '거시 변수 요인이 품목 수익을 예측하는가'로 단순화했습니다."],
        ["Sharpe · 학습 · 시험 · 위약", "위험 한 단위당 수익 · 설정을 고르는 구간(2008–16) · 확인하는 구간(2017–) · 신호를 무작위로 섞은 가짜 전략(진짜가 가짜보다 확실히 좋아야 함, p가 작을수록 좋음)."],
        ["기준 전략", f"MinVar 50 · 모멘텀 25 · 가치 25, 목표 변동성 10%. 주간 자료 학습 Sharpe {BASE_TR:.2f} · 시험 {BASE_TE:.2f}."]], [3.8, 13.4], size=9.5)
    # 1
    d.h("1. 요인 모형이 하는 일", 1)
    d.p("원자재 19개를 하나씩 보면 19개의 서로 다른 움직임 같지만, 실제로는 상당 부분이 같은 원인으로 같이 움직입니다. 오를 때는 다 같이 오르고 빠질 때는 다 같이 빠집니다. PCA는 이 '같이 움직이는 부분'을 크기 순으로 뽑아 줍니다.")
    d.img("g02_loadings.png", 17.0, "그림 1. 최근 104주 기준 PCA 묶음 — PC1은 막대가 거의 모두 같은 방향(모두 같이 움직임), PC2·PC3은 품목군별로 갈라짐")
    d.b([("PC1 (공통 요인): ", ), "대부분 품목이 같은 방향. 쉽게 말해 '원자재 전체의 방향'입니다. 바스켓을 통째로 사는 것과 거의 같습니다."])
    d.b([("PC2 · PC3: ", ), "품목군끼리 서로 반대로 움직이는 묶음(그림 1 기준 PC2는 에너지 vs 금속, PC3은 곡물 vs 에너지·금속). 다만 부호와 의미가 시기마다 바뀌기 쉽습니다."])
    d.box("한 줄 정리", ["요인 모형은 '지금 시장이 어떤 묶음으로 움직이는가'를 보여 줍니다. 이걸 보고 '그럼 이 묶음에 투자하자'로 가는 것(길 2)과, '내 포트폴리오가 어느 묶음에 쏠렸는지 확인하자'로 가는 것(길 1)은 다른 일입니다."])
    d.img("g01_map.png", 17.0, "그림 2. 요인 모형을 포트폴리오와 잇는 두 길 — 길 1은 채택, 길 2는 미채택")
    d.pb()
    # 2
    d.h("2. 길 1 — 내 포트폴리오의 위험이 어디에 몰렸나", 1)
    d.p("여기서 쓰는 비중은 요인이 만든 것이 아닙니다. 기본 혼합(50/25/25)이 지금 실제로 들고 있는 품목별 비중을 그대로 가져와서, 그 비중이 PCA 묶음들에 얼마나 노출돼 있는지를 계산합니다. 대시보드에서 품목과 비중을 직접 바꿔 다시 돌려 볼 수도 있습니다.")
    d.box("쉬운 비유", ["오케스트라 19명이 연주하는 것처럼 보여도, 실제로는 '큰북 한 대'가 소리의 2/3를 내고 있다면 — 악기를 늘려도 큰 소리가 한 곳에서 나오는 구조는 안 바뀝니다. 요인 위험 분해는 '누가 큰 소리를 내고 있나'를 찾는 일입니다."], fill="FBF5E6", bar="#8A6D1E")
    d.img("g03_decomp.png", 16.0, "그림 3. 기본 혼합의 PC별 분산 기여(빨강)와 시장 전체의 PC별 비중(회색, 최근 104주)")
    d.table(["지표", "값", "읽는 법"], [
        ["PC1이 포트폴리오 분산에서 차지하는 몫", f"{o['pc1_share']:.0%}", "공통 원자재 방향이 위험의 대부분을 만듦"],
        ["PC1이 시장 전체 변동에서 차지하는 몫", f"{o['pc1_market']:.0%}", "시장 평균은 이보다 훨씬 낮음 → 내 포트폴리오가 더 쏠림"],
        ["상위 3개 요인의 기여 합", f"{o['top_share']:.0%}", "PC2는 0%(기본 혼합이 에너지-금속 대립 방향에는 거의 노출되지 않음), 나머지 16개 묶음은 약 1/4"],
        ["유효 요인 수 (ENB)", f"{o['enb']:.1f} / 19", "19개 품목이어도 서로 다른 위험 묶음은 약 4개"],
        ["포트폴리오 변동성 (연율, 위 비중 기준)", f"{o['vol']:.1f}%", "최근 104주 주간 수익률로 계산"]], [6.6, 2.4, 8.2], size=9.5, center_from=1)
    d.img("g04_enb.png", 15.5, "그림 4. 겉보기 분산(품목 수)과 실제 위험 분산(유효 요인 수)의 차이")
    d.h("왜 이게 중요한가", 2)
    d.b("3축(MinVar · 모멘텀 · 가치)으로 전략을 나눠도, 위험의 원천은 같은 '공통 원자재 요인'에 상당 부분 몰려 있습니다. 전략 간 분산이지 위험 요인 간 분산이 아닙니다.")
    d.b("그동안 곡물 비중을 25%로 줄였더니 VaR이 줄었던 것도, 곡물이 공통 요인과 같이 크게 움직이는 품목군이기 때문이라고 해석할 수 있습니다. 같은 현상을 요인 쪽에서 본 것입니다.")
    d.b([("주의: ", ), "'쏠림이 크다'는 사실을 보여 줄 뿐, '줄이면 성과가 좋아진다'는 것은 아직 시험하지 않았습니다. 진단 도구로만 채택한 이유입니다."])
    d.p("참고로 기본 혼합의 현재 섹터별 순비중은 다음과 같습니다(자본 대비, 롱 − 숏).", 10)
    d.table(["섹터", "순비중"], [[s, f"{v:+.1%}"] for s, v in sec_w.items()], [4.0, 3.0], size=9.5, center_from=1)
    d.pb()
    # 3
    d.h("3. 길 2 — 요인으로 비중을 짜 보면", 1)
    d.p("이번에는 반대로, 요인이 직접 비중을 만들게 했습니다. 모두 같은 틀에서 시험했습니다: 4주마다 다시 계산하고, 그 시점까지의 자료(직전 104주 또는 확장창)만 쓰며, 편도 비용 5bp, 목표 변동성 10%. 평가는 '이 전략 단독'과 '혼합'(가치 25% 자리에 이 전략을 넣은 MinVar 50 · 모멘텀 25 · 이 전략 25) 두 가지로 했습니다.")
    d.table(["길", "하는 일 (쉬운 말)", "대응 연구"], [
        ["A 공분산", "품목 사이 출렁임 관계(공분산)를 표본 대신 'PCA 요인 구조'로 대체해 MinVar를 계산", "PCA 기반 공분산 추정"],
        ["B PC 포트폴리오", "PCA 묶음 자체를 포트폴리오로 사용 (비중 = 고유벡터 ÷ 변동성)", "Severini(2022) 등"],
        ["C 잠재 요인 평균-분산", "품목 대신 요인 몇 개 위에서 평균-분산 최적화 (RP-PCA 포함)", "Lettau–Pelger(2020)"],
        ["D 거시 요인 → 기대수익", "거시 변수의 PCA 요인으로 품목 4주 뒤 수익을 예측해 순위 롱숏", "FAVAR · 동적 요인 모형"]], [3.6, 9.6, 4.0], size=9.5)
    d.h("A. PCA 요인 공분산", 2)
    d.p("로직: MinVar는 '가장 안 흔들리게 만드는 비중'이라 품목 사이 관계(공분산)가 핵심 입력입니다. 표본으로 계산하는 대신 Σ = Λ Σ_F Λ′ + Σ_ε(요인 몇 개가 만드는 공통 부분 + 품목 고유 부분)로 바꿔 본 것입니다.")
    d.b([("결과: ", ), "Ledoit-Wolf(현재 사용)와 비교해 차이가 거의 없습니다(차이 t < 1, 이전 실험 M). 품목이 19개로 많지 않아 표본 공분산이 이미 충분히 안정적이어서, 요인으로 줄여도 얻는 게 없습니다."])
    d.b("판정: 미채택 (현행 Ledoit-Wolf 유지).")
    d.h("B. PC 포트폴리오", 2)
    d.p("로직: PCA 묶음(고유벡터)을 그대로 포트폴리오 비중으로 씁니다. 변동성이 큰 품목이 과도하게 담기지 않게 변동성으로 나눠(w = v/σ) 총노출을 1로 맞춥니다. 묶음의 +/− 방향은 임의라서 두 방식으로 시험했습니다: ① '고정 부호'(적재값 합이 +가 되게), ② '승자 방향'(직전 104주에 번 쪽).")
    d.img("g06_pc.png", 16.0, "그림 5. B. PC 포트폴리오 단독 Sharpe — 학습에서 좋은 PC3이 시험에서 뒤집힘")
    r1, r3 = row("F1 PC 포트폴리오", "PC1"), row("F1 PC 포트폴리오", "PC3 · 부호 고정")
    d.b([("PC1: ", ), f"학습 {r1[TR]:+.2f} → 시험 {r1[TE]:+.2f}. 모든 품목이 같은 방향이라 사실상 '바스켓 롱'입니다. 원자재가 오른 시험 기간에 벌었을 뿐, 새로운 정보가 아닙니다."])
    d.b([("PC3: ", ), f"학습 {r3[TR]:+.2f}(위약 대비 p {r3['위약 p (학습)']:.2f}로 유의)였는데 시험 {r3[TE]:+.2f}(위약 p {r3['위약 p (시험)']:.2f}). 학습에서 가장 좋아 보였던 것이 시험에서 가장 나빠졌습니다 — 묶음의 의미와 방향이 안정적이지 않다는 뜻입니다."])
    d.b("판정: 미채택. 혼합에 넣은 시험 Sharpe는 0.52 ~ 0.79로, 기준 1.07보다 낮습니다.")
    d.h("C. 잠재 요인 평균-분산 (RP-PCA)", 2)
    d.p("로직: 요인 몇 개의 수익률을 새로운 '자산'으로 보고 그 위에서 최적화합니다. 요인끼리는 서로 무관하므로 요인별 비중이 간단히 w_f = μ_f ÷ σ_f²(수익 ÷ 분산)로 나옵니다. 평균(μ)은 추정 오차가 커서 절반(×0.5)으로 줄여 씁니다. RP-PCA는 요인을 뽑을 때 출렁임뿐 아니라 평균 수익도 함께 고려해(γ가 클수록 평균 중시) 평균을 잘 설명하는 요인을 우선 고릅니다.")
    d.img("g07_latent.png", 17.0, "그림 6. C. 방식 × 요인 개수별 단독 Sharpe — 학습과 시험 모두 양수지만 기준보다 낮고, 차이가 작음")
    r = row("F2 잠재 요인 평균-분산", "γ=20 · 요인 3개")
    d.b([("가장 좋은 설정: ", ), f"RP-PCA γ=20 · 요인 3개(학습 최고) — 학습 {r[TR]:+.2f} → 시험 {r[TE]:+.2f}. 같은 요인 개수의 일반 PCA보다 시험이 높거나 같습니다(요인 1개 0.64 vs 0.51, 2개 0.60 vs 0.39, 3개 0.44 vs 0.44)."])
    d.b([("하지만: ", ), f"학습 개선이 작고, 위약 p는 시험 {r['위약 p (시험)']:.2f} · 학습 {r['위약 p (학습)']:.2f}로 무작위로 섞은 가짜와 구별되지 않습니다. 혼합 시험은 0.72 ~ 0.81로 기준 1.07보다 낮습니다."])
    d.b("판정: 미채택. RP-PCA가 일반 PCA보다 나은 듯 보이는 것은 흥미롭지만 통계적으로 확신하기 어렵습니다.")
    d.h("D. 거시 요인 → 기대수익 (FAVAR-lite)", 2)
    d.p("로직: 시장에서 매일 관찰되는 거시 변수 21개(VIX · 금리 · 신용 스프레드 · 달러 · 환율 · 금융여건)의 주간 변화를 1주 늦춰서(미래 정보 차단) 사용합니다. 그때까지의 자료로 PCA 3요인을 뽑고, 품목마다 '4주 뒤 수익 ~ 요인'을 릿지 회귀로 예측합니다. 예측 수익이 높은 품목을 사고 낮은 품목을 파는 순위 롱숏입니다.")
    d.img("g09_pipe.png", 17.0, "그림 7. D. 거시 요인 → 기대수익의 처리 순서")
    d.img("g08_macro.png", 16.0, "그림 8. D. 결과 — 학습 +0.35에서 시험 −0.22로 뒤집히고, 예측 순위와 실제 순위의 상관(IC)이 사실상 0")
    d.b([("순위 IC +0.004 (t 0.22): ", ), "예측이 실제 순위를 거의 못 맞춥니다. 위약 p(시험) 0.94. 혼합 시험 0.72."])
    d.b([("제외한 것: ", ), "월간 거시 통계(CPI · 산업생산 · 고용)는 발표 지연과 사후 수정 때문에 시점 정합성을 확보하기 어려워 뺐고, 요인 기반 공분산 예측(FAVAR-SV)은 시험하지 않았습니다."])
    d.b("판정: 미채택.")
    d.h("B와 C는 무엇이 다른가 — 자주 헷갈리는 점", 2)
    d.p("B · C 모두 19개 품목으로 PCA를 돌리고, 요인을 '회귀식의 설명변수'로 쓰지 않습니다. 요인을 포트폴리오(자산)로 취급하고, 마지막에 품목 비중으로 되돌립니다. 차이는 '요인으로 무엇을 결정하느냐'입니다.")
    d.img("g10_bc.png", 17.0, "그림 5-1. B와 C의 구조 — C는 B의 포트폴리오들을 평균과 분산으로 섞는 단계가 더 있음")
    d.table(["", "B. PC 포트폴리오", "C. 잠재 요인 평균-분산"], [
        ["요인으로 하는 일", "요인 하나를 그대로 포트폴리오로 사용", "요인 K개를 '자산'으로 보고 최적화"],
        ["비중 결정", "최적화 없음. 비중 = 고유벡터 ÷ σ, 총노출 1", "요인 비중 w_f = μ_f ÷ σ_f² → 품목 비중 = V·w_f ÷ σ, 총노출 1"],
        ["평균(수익) 사용", "사용하지 않음 (승자 방향 변형만 부호 결정에 사용)", "사용 (오차가 커서 ×0.5로 줄임)"],
        ["분산 사용", "사용하지 않음", "요인별 분산으로 나눔 (요인끼리 무상관이라 간단)"],
        ["요인 개수", "PC 하나씩 따로", "K개를 한꺼번에 (1·2·3·5개)"],
        ["둘의 관계", "C의 K=1 = B의 PC1 포트폴리오 (부호만 평균으로 결정)", "C = B의 포트폴리오들을 μ ÷ σ²로 가중해 섞은 것"]], [3.4, 6.8, 7.0], size=9)
    d.b([("v ÷ σ를 하는 정확한 이유: ", ), "고유벡터 v는 '표준화한 수익률(Z = (R - 평균) ÷ σ)' 기준이라, 품목별 σ로 나눠야 실제 수익률 R로 그 요인 수익을 그대로 재현하는 포지션이 됩니다. 효과로는 변동성이 큰 품목이 과도하게 담기지 않고, 이것이 '품목 비중으로의 역변환'입니다."])
    d.b([("RP-PCA는 B · C와 다른 축입니다. ", ), "'요인을 어떻게 뽑느냐'의 선택입니다. 일반 PCA는 공분산만 보고 뽑고, RP-PCA는 공분산에 평균 항((1+γ)μμ′)을 더해 평균도 함께 보고 뽑습니다. 이번 시험에서 B는 일반 PCA 요인만, C는 PCA(γ=−1)와 RP-PCA(γ=0, 20)를 모두 사용했습니다."])
    d.b([("요인이 '설명변수'로 쓰이는 곳: ", ), "D(거시 요인 → 품목 수익 회귀)뿐입니다. A는 공분산의 구조로, B · C는 자산으로 씁니다."])
    d.pb()
    # 4
    d.h("4. 그래서 50/25/25보다 나아지는가", 1)
    d.img("g05_paths.png", 16.5, "그림 9. 가치 자리(25%)에 4가지 길을 넣은 혼합의 시험 Sharpe — 모두 기준 3축 혼합(1.07)보다 낮음")
    d.table(["길", "단독 학습 → 시험", "혼합 시험 (기준 1.07)", "위약 p (시험)", "판정"], [
        ["A PCA 공분산", "LW와 차이 없음 (t<1)", "–", "–", "미채택"],
        ["B PC 포트폴리오 (PC1)", f"{r1[TR]:+.2f} → {r1[TE]:+.2f}", f"{r1['혼합 시험']:.2f}", "–", "미채택 (사실상 바스켓 롱)"],
        ["B PC3 (부호 고정)", f"{r3[TR]:+.2f} → {r3[TE]:+.2f}", f"{r3['혼합 시험']:.2f}", f"{r3['위약 p (시험)']:.2f}", "미채택 (뒤집힘)"],
        ["C RP-PCA γ=20 · 3개", f"{r[TR]:+.2f} → {r[TE]:+.2f}", f"{r['혼합 시험']:.2f}", f"{r['위약 p (시험)']:.2f}", "미채택 (위약과 구분 안 됨)"],
        ["D 거시 요인 → 기대수익", "+0.35 → −0.22", "0.72", "0.94", "미채택 (IC ≈ 0)"]], [4.2, 3.4, 3.2, 2.4, 4.0], size=9, center_from=1)
    d.h("'설명력이 더해지는가'에 대한 답", 2)
    d.b([("성과 기준 — 아니오. ", ), "4가지 모두 혼합 시험이 기준 1.07보다 낮고, 위약 검정을 통과한 것도 없습니다. 가치 축을 대체해도 못 이깁니다."])
    d.b([("위험 이해 기준 — 예. ", ), "길 1의 요인 위험 분해는 기존 VaR 탭이 보여 주지 못한 '위험의 요인 집중도'를 보여 줍니다."])
    d.b([("왜 비중 만들기는 실패했나: ", ), "① PC1은 새로운 정보가 없는 바스켓 롱이다. ② PC2 이후는 방향·의미가 시기마다 바뀌어 학습 성과가 시험에서 유지되지 않는다. ③ 평균 수익 추정은 어떤 요인 방식으로도 오차가 커서, 요인으로 줄여도 예측력이 생기지 않는다. ④ 거시 변수 요인은 원자재 품목의 순위를 가르는 정보가 아니라 시장 전체의 방향 정보에 가깝다."])
    d.box("이 결론의 성격", ["이것은 '요인 모형이 가치 없다'가 아니라 '기준 혼합이 이미 만든 결과를 요인으로 개선하기 어렵다'입니다. 시험한 설정이 여러 개(4개 길 × 여러 설정)라서 우연히 좋아 보이는 것이 하나쯤 섞일 수 있어 판단은 오히려 보수적이었습니다."], fill="F1F1F8", bar="#4A3F8F")
    # 5
    d.h("5. 대시보드에서 보는 법", 1)
    d.p("위치: ①+ 요인 모형 › 'C. 요인 → 포트폴리오 (위험 분해 · 비중 설계 검증)'")
    d.table(["구역", "무엇을 하나", "조절할 수 있는 것"], [
        ["① 내 포트폴리오의 요인 위험 분해", "품목과 비중을 고르면 PC별 분산 기여 · 유효 요인 수 · 적재값 히트맵을 계산", "시작 비중(기본 혼합 / 직접), 요인 수 K, 추정 창(주), 품목·비중 표"],
        ["② 4가지 길의 백테스트 (고정 결과)", "A~D 결과 표 · 누적 수익 비교 · FAVAR→포트폴리오 구조 설명", "묶음 선택(요약 · F1 · F2 · F3), 누적 수익 비교 대상"],
        ["D 서브탭 · 요인별 위험 진단", "동적 PCA · 관측 요인 · 거시 요인 · 주파수별 · 종합 진단 + 코히어런스 · Spectral 포트폴리오 결과(고정)", "창 · 구간 · 요인 수 · 자료 · 밴드 정의"]], [4.6, 7.2, 5.4], size=9.5)
    d.h("읽는 순서 제안", 2)
    for t in ["① 시작 비중을 '기본 혼합'으로 두고 PC1 분산 기여와 유효 요인 수를 확인합니다.", "② 품목을 빼거나 비중을 바꿔(예: 곡물 25% 캡) PC1 몫이 얼마나 줄어드는지 봅니다.", "③ 추정 창을 104주 → 260주로 늘려 결과가 안정적인지 확인합니다(기본 혼합은 66% → 60%로 큰 차이 없음).", "④ ②의 표에서 요인으로 비중을 짠 결과가 왜 채택되지 않았는지 확인합니다."]:
        d.p(t, 10, after=2)
    # 6
    SPECTRAL_CHAPTER(d)
    d.h("7. 읽을 때 주의할 점과 다음 단계", 1)
    for t in [[("같은 자료 위의 반복 시험: ", ), "모든 실험이 같은 2005–2026 자료입니다. '결과가 같다'가 독립된 증거는 아닙니다."], [("진단 ≠ 처방: ", ), "PC1 쏠림이 크다는 것은 사실이지만 '줄이면 낫다'는 것은 아직 검증되지 않았습니다."], [("PCA는 과거 자료로 만든 묶음: ", ), "위기 때 묶음 구조가 바뀌면(상관 급등) 진단도 바뀝니다."],
              [("시험하지 않은 것: ", ), "월간 거시 통계, FAVAR-SV 공분산 예측, 일간 주기 요인 모형, 곡선·재고·마진 같은 새 정보."], [("검증 범위: ", ), "새 서브탭은 해당 묶음 테스트와 화면 텍스트 확인을 거쳤고, 앱 전체 테스트는 하지 않았습니다."]]:
        d.b(t)
    d.h("다음에 해 볼 만한 것", 2)
    for t in ["PC1 몫이 높은 달과 낮은 달로 나눠, 이후 4주 손실과 VaR 초과가 달랐는지 확인 — 진단이 실제로 쓸모 있는지 검증", "곡물 25% 캡 전후 PC1 몫 비교 — 섹터 캡이 요인 집중을 얼마나 낮추는지", "PC1 몫이 임계치를 넘으면 총노출만 줄이는 규칙 시험 — 비중 설계보다 노출 조절이 맞을 수 있음"]:
        d.b(t)
    d.d.save(str(OUT)); print("saved", OUT)


if __name__ == "__main__":
    main()
