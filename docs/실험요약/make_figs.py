"""실험 요약 문서용 그림 — 결과 CSV에서 직접 그림 (숫자 = docs/백테스트_로그.md).
실행: .venv\\Scripts\\python.exe docs\\실험요약\\make_figs.py → docs/실험요약/figs/*.png
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "figs"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10})
INK, G, BLUE, RED, GOLD, GREY = "#1E3A2F", "#2E7D52", "#2F6E8F", "#C8553D", "#C9A04A", "#8A8F98"
D = ROOT / "data"


def box(ax, x, y, w, h, text, fc, ec=None, fs=9, color="white", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec or fc, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, c=GREY):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4))


def fig_flow():
    """전체 실험 구조: 재료 → 신호 → 쓰는 방식(A–H) → 검증 → 판정."""
    f, ax = plt.subplots(figsize=(11, 6.4)); ax.set_xlim(0, 11); ax.set_ylim(0, 6.4); ax.axis("off")
    ax.text(0.1, 6.15, "① 재료", fontsize=11, fontweight="bold", color=INK)
    for k, (t, c) in enumerate([("MinVar 롱\n(변동성 작은 품목)", BLUE), ("52주 모멘텀\n(추세 추종)", G), ("가치 = 평균회귀\n(5년 평균 대비 싼 품목)", GOLD)]):
        box(ax, 0.1, 4.65 - k * 1.25, 2.0, 1.0, t, c, fs=9)
    ax.text(0.1, 0.95, "고정 50 / 25 / 25\n(학습 1위 · 시험 Sharpe 1.07)", fontsize=9, color=INK, fontweight="bold")
    ax.text(2.6, 6.15, "② 신호", fontsize=11, fontweight="bold", color=INK)
    sig = ["HMM 위기 국면", "칼만 평균 추적", "DCC 동조성 · 공분산", "흡수비율 급등", "VIX · 변동성 국면", "뉴스 감성(FinBERT)", "곡물 상관 + 오름 방향"]
    for k, t in enumerate(sig):
        box(ax, 2.6, 5.45 - k * 0.72, 2.1, 0.58, t, "#EEF2EE", ec=INK, color=INK, fs=8.6)
    ax.text(5.2, 6.15, "③ 신호를 쓰는 방식", fontsize=11, fontweight="bold", color=INK)
    use = [("A 연도별 최적 비중으로 갈아타기", RED), ("B 다음 분기 1등 축 미리 읽기(6레이어)", RED), ("C 비중을 환경에 맞게 기울이기", RED),
           ("D 위험할 때 현금 피신", RED), ("E 목표 변동성 동적 조절 + 금리", RED), ("F 상관↑ + 같이 오를 때 강한 추세 집중", RED),
           ("G 품목군 비중 상한 (곡물 · 에너지 · 축산)", G), ("H 손실 한도 경보 (매매 없음)", BLUE)]
    for k, (t, c) in enumerate(use):
        box(ax, 5.2, 5.5 - k * 0.66, 3.3, 0.52, t, c, fs=8.6)
    ax.text(9.0, 6.15, "④ 검증 → 판정", fontsize=11, fontweight="bold", color=INK)
    for k, t in enumerate(["학습 2008–16에서만 설정 선택", "시험 2017–26에서 확인", "거래 · 갈아타기 비용 포함", "위약(가짜 신호) 비교", "대조군(늘 같은 비중)"]):
        box(ax, 9.0, 5.45 - k * 0.72, 1.9, 0.58, t, "#F6F1E6", ec=GOLD, color=INK, fs=8.4)
    box(ax, 9.0, 0.85, 1.9, 0.9, "미채택 A–F\n채택 후보 G · 운영 H", INK, fs=9, bold=True)
    for k in range(3):
        arrow(ax, 2.1, 5.15 - k * 1.25, 2.55, 3.4)
    arrow(ax, 4.75, 3.4, 5.15, 3.4); arrow(ax, 8.55, 3.4, 8.95, 3.4); arrow(ax, 9.95, 1.95, 9.95, 1.8)
    f.tight_layout(); f.savefig(OUT / "f1_flow.png", dpi=200); plt.close(f)


def fig_results():
    """실험별 시험 Sharpe vs 기준 1.07."""
    rows = [("기준 고정 50/25/25", 1.07, INK), ("동일 비중", 1.11, GREY), ("A 연도별 최적 갈아타기", 0.89, RED), ("B 메타 레이어(1등 축 예측)", 1.05, RED),
            ("C 칼만 적응 비중", 0.79, RED), ("C HMM 적응 비중", 1.08, RED), ("C 4개 신호 결합", 0.85, RED), ("D 현금 피신 · VIX", 0.90, RED),
            ("D 현금 피신 · 낙폭 손절", 1.10, RED), ("E 목표 변동성 동적(계단)", 0.96, RED), ("F 상관↑ → 모멘텀 집중", 0.91, RED),
            ("F 상관↑ + 상승 동조 → 상승 추세 집중", 1.07, RED), ("G 곡물 상한 25%", 1.09, G)]
    f, ax = plt.subplots(figsize=(10, 5.6))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.62)
    for yy, (n, v, _) in zip(y, rows):
        ax.text(v + 0.01, yy, f"{v:.2f}", va="center", fontsize=9)
    ax.axvline(1.07, color=INK, ls="--", lw=1); ax.text(1.075, len(rows) - 0.4, "기준 1.07", fontsize=9, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(0.6, 1.2)
    ax.set_xlabel("시험 기간(2017–2026) Sharpe — 각 실험의 학습 최고 설정"); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("신호로 비중을 바꾼 실험은 모두 기준선 근처나 아래", loc="left", fontweight="bold", color=INK)
    f.tight_layout(); f.savefig(OUT / "f2_results.png", dpi=200); plt.close(f)


def fig_f_logic():
    """F 실험 판단 흐름: 상관 급등? → 방향? → 켜짐(강한 추세 집중) / 꺼짐(50/25/25)."""
    f, ax = plt.subplots(figsize=(11, 4.6)); ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis("off")
    box(ax, 0.1, 1.8, 1.9, 1.0, "매주 금요일\n(그 주까지 자료만)", INK, fs=9)
    box(ax, 2.5, 1.8, 2.3, 1.0, "① 상관 급등?\n곡물 26주 상관 분위 > 0.8–0.9\n또는 DCC 동조성 · 흡수비율", BLUE, fs=8.5)
    box(ax, 5.3, 1.8, 2.3, 1.0, "② 같이 오르는 중?\n직전 13주 곡물(바스켓)\n수익 > 0", GOLD, fs=8.5, color=INK)
    box(ax, 8.1, 2.95, 2.8, 1.3, "③ 켜짐 → 강한 추세에 집중\n26·52주 모두 오른 품목 중\n위험조정 상승 상위 2–3개 롱\n(모멘텀 비중 50–100%)", G, fs=8.5)
    box(ax, 8.1, 0.35, 2.8, 1.2, "꺼짐 → 기본 50/25/25\n(MinVar · 모멘텀 · 가치)", GREY, fs=8.8)
    arrow(ax, 2.0, 2.3, 2.45, 2.3); arrow(ax, 4.8, 2.3, 5.25, 2.3)
    arrow(ax, 7.6, 2.5, 8.05, 3.5, G); ax.text(7.62, 3.1, "예", color=G, fontsize=9)
    arrow(ax, 7.6, 2.1, 8.05, 1.0, GREY); ax.text(5.35, 1.5, "아니오 = 함께 폭락 중 → 제외", color=GREY, fontsize=8.5)
    arrow(ax, 3.65, 1.75, 8.05, 0.75, GREY); ax.text(3.0, 0.75, "상관 정상 → 꺼짐", color=GREY, fontsize=8.5)
    ax.text(0.1, 4.25, "F 실험: '상관이 높으면서 같이 오르는' 구간에만 추세 집중 — 문턱 · 비중은 학습 2008–16에서 선택", fontsize=10.5, fontweight="bold", color=INK)
    f.tight_layout(); f.savefig(OUT / "f3_f_logic.png", dpi=200); plt.close(f)


def fig_cases():
    """2022 곡물 · 원유 / 2025–26 금 · 은: 가격 · 곡물 상관 · 상승 추세 집중 성과."""
    import analytics as A
    import ls_strategies as LS
    L = A.load_levels(roll_adjusted=True).loc[:"2026-10-26"]
    P = LS.weekly(L, LS.BASKET, "2001-01-01"); R = LS._ret(P[[c for c in LS.BASKET if c in P]])
    G5 = ["yf:ZC=F", "yf:ZW=F", "yf:ZS=F", "yf:ZM=F", "yf:ZL=F"]
    gc = pd.Series({R.index[t]: R[G5].iloc[t - 25:t + 1].corr().values[np.triu_indices(5, 1)].mean() for t in range(26, len(R))})
    f, axs = plt.subplots(2, 2, figsize=(11, 6.6), gridspec_kw={"height_ratios": [2, 1]})
    for col, (a, b, ids, nm) in enumerate([("2021-07-01", "2023-06-30", {"yf:ZC=F": "옥수수", "yf:ZW=F": "밀", "yf:CL=F": "WTI 원유"}, "2022 우크라이나 전쟁"),
                                            ("2024-07-01", "2026-10-26", {"yf:GC=F": "금", "yf:SI=F": "은"}, "2025–26 금 · 은 급등")]):
        ax = axs[0, col]
        for (i, n), c in zip(ids.items(), [GOLD, BLUE, RED]):
            p = P[i].loc[a:b]; ax.plot(p.index, p / p.iloc[0] * 100, label=n, color=c, lw=1.8)
        ax.set_title(f"{nm} — 가격(시작 = 100)", loc="left", fontweight="bold", color=INK, fontsize=10); ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        if col == 0:
            ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-03-31"), color=GOLD, alpha=0.15); ax.text(pd.Timestamp("2022-01-05"), ax.get_ylim()[1] * 0.95, "급등 1분기", fontsize=8)
        ax2 = axs[1, col]; s = gc.loc[a:b]
        ax2.plot(s.index, s, color=INK, lw=1.4); ax2.axhline(gc.quantile(0.9), color=RED, ls="--", lw=1)
        ax2.text(s.index[0], gc.quantile(0.9) + 0.02, "문턱(과거 상위 10%)", fontsize=8, color=RED)
        import matplotlib.dates as mdates
        for a_ in (ax, ax2):
            a_.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); a_.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.set_ylim(0, 0.9); ax2.set_title("곡물 5개 26주 평균 상관", loc="left", fontsize=9, color=INK); ax2.spines[["top", "right"]].set_visible(False)
    f.tight_layout(); f.savefig(OUT / "f4_cases.png", dpi=200); plt.close(f)


def fig_cap():
    T = pd.read_csv(D / "strategy_graincap_compare.csv", encoding="utf-8-sig")
    f, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    lab = T["곡물 상한 (총노출 대비)"].tolist()
    axs[0].bar(lab, T["Sharpe 시험 2017–"], color=[GREY, GREY, GREY, G]); axs[0].set_ylim(0.9, 1.15); axs[0].set_title("시험 Sharpe (수익은 그대로)", loc="left", fontsize=10, color=INK)
    axs[1].bar(lab, T["지금 VaR 곡물 기여 (%)"], color=[GREY, GREY, GREY, G]); axs[1].set_title("지금 VaR 중 곡물 기여 (%)", loc="left", fontsize=10, color=INK)
    for ax, c in zip(axs, ["Sharpe 시험 2017–", "지금 VaR 곡물 기여 (%)"]):
        for k, v in enumerate(T[c]):
            ax.text(k, v, f"{v:.2f}" if "Sharpe" in c else f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    f.tight_layout(); f.savefig(OUT / "f5_graincap.png", dpi=200); plt.close(f)


if __name__ == "__main__":
    fig_flow(); fig_results(); fig_f_logic(); fig_cases(); fig_cap()
    print("그림:", sorted(p.name for p in OUT.glob("*.png")))
