"""일간 백테스트 전략 쉬운 설명 Word 문서 — 그림(matplotlib) + 본문(python-docx). 숫자는 data/*_compare.csv · daily parquet에서 직접 읽음.
실행: .venv\\Scripts\\python.exe docs\\실험요약\\make_daily_doc.py → docs/실험요약/일간_백테스트_전략_쉬운설명_20261110.docx (+ figs_daily/*.png)
참고: node · LibreOffice가 없는 PC에서도 만들 수 있게 python-docx 사용(pip install python-docx).
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Cm, Pt, RGBColor  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
FIG = HERE / "figs_daily"; FIG.mkdir(exist_ok=True)
OUT = HERE / "일간_백테스트_전략_쉬운설명_20261110.docx"
D = ROOT / "data"
plt.rcParams.update({"font.family": ["Malgun Gothic", "NanumGothic"], "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
INK, GRN, BLUE, RED, GOLD, GREY = "#1E3A2F", "#2E7D52", "#2F6E8F", "#C8553D", "#C9A04A", "#8A8F98"
FONT = "Malgun Gothic"

rd = lambda n: pd.read_csv(D / n, encoding="utf-8-sig")   # noqa: E731
FQ, SG, MR, MM = rd("strategy_daily_freq_compare.csv"), rd("strategy_daily_signals_compare.csv"), rd("strategy_daily_more_compare.csv"), rd("strategy_daily_multimodal_compare.csv")
CASH_W, ADAPT_W = rd("strategy_cash_compare.csv"), rd("strategy_adaptive_compare.csv")
TR, TE, TW = "Sharpe 학습 2008–16", "Sharpe 시험 2017–", "주간 합산 시험"
BASE_TR, BASE_TE = 0.61, 1.32                                   # 일간 3축 고정 50/25/25 (strategy_daily_freq D1)


def best(df, col="실험", key=None):
    x = df[df["학습 최고"] == "←"]
    return x if key is None else x[x[col].astype(str).str.startswith(key)]


# ---------------------------------------------------------------- 그림
def box(ax, x, y, w, h, t, fc, fs=9, color="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=fc))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=color, fontweight="bold", linespacing=1.6)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.5))


def fig_structure():
    f, ax = plt.subplots(figsize=(11, 4.6)); ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis("off")
    for k, (t, c) in enumerate([("① MinVar 롱\n(덜 출렁이는 품목에 더 담기)", BLUE), ("② 모멘텀\n(1년간 오른 건 사고 내린 건 팔기)", GRN), ("③ 가치\n(5년 평균보다 싼 건 사고 비싼 건 팔기)", GOLD)]):
        box(ax, 0.1, 3.2 - k * 1.5, 3.1, 1.15, t, c)
    arrow(ax, 3.3, 3.75, 4.3, 2.4); arrow(ax, 3.3, 2.25, 4.3, 2.3); arrow(ax, 3.3, 0.75, 4.3, 2.1)
    box(ax, 4.3, 1.7, 2.3, 1.2, "섞기\n50% / 25% / 25%", INK)
    arrow(ax, 6.7, 2.3, 7.5, 2.3)
    box(ax, 7.5, 1.7, 3.3, 1.2, "목표 변동성 10%로 맞추기\n(크루즈컨트롤 · 최대 4배)", "#6B4F9E")
    ax.text(5.45, 0.9, "← 신호 전략(A~F)이 개입하는 곳:\n비중 · 투자 규모(노출)", ha="center", fontsize=9, color=RED, fontweight="bold")
    ax.text(0.1, 4.45, "기준 전략 = 일간 3축 고정 50/25/25 (학습 2008–16 Sharpe 0.61 · 시험 2017– 1.32)", fontsize=10.5, color=INK, fontweight="bold")
    f.tight_layout(); f.savefig(FIG / "f01_structure.png", dpi=170); plt.close(f)


def fig_train_test():
    f, ax = plt.subplots(figsize=(10, 2.2)); ax.set_xlim(2005, 2027); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((2008, 0.35), 9, 0.4, boxstyle="round,pad=0.01", fc=BLUE, ec=BLUE)); ax.text(2012.5, 0.55, "학습 2008–2016 : 규칙 · 문턱 · 설정을 고르는 구간", ha="center", va="center", color="white", fontweight="bold")
    ax.add_patch(FancyBboxPatch((2017, 0.35), 9.6, 0.4, boxstyle="round,pad=0.01", fc=GRN, ec=GRN)); ax.text(2021.8, 0.55, "시험 2017–2026 : 고른 설정을 그대로 적용해 확인", ha="center", va="center", color="white", fontweight="bold")
    ax.text(2008, 0.1, "※ 시험 결과를 보고 설정을 바꾸면 '사후 선택' — 이 문서의 모든 결과는 학습 구간에서만 고른 설정입니다", fontsize=9, color=RED)
    f.tight_layout(); f.savefig(FIG / "f02_traintest.png", dpi=170); plt.close(f)


def daily_series():
    E = pd.read_parquet(D / "strategy_daily_more_daily.parquet")
    S = pd.read_pickle(D / "frozen" / "sleeves_daily_2026-10-26.pkl")[0]
    return E, S


def fig_axes(S, E):
    f, ax = plt.subplots(figsize=(10, 4))
    for c, nm, col in [("기준 · MinVar 롱 (⑥)", "MinVar 롱", BLUE), ("기준 · 52주 모멘텀 (④)", "모멘텀", GRN), ("V1 가치 (5년 평균 대비 싼 품목 롱)", "가치", GOLD)]:
        ax.plot(S.index, S[c].fillna(0).cumsum(), color=col, lw=1.2, label=nm)
    b = E["A|고정 50/25/25"].astype(float); ax.plot(b.index, b.fillna(0).cumsum(), color=INK, lw=2.4, label="혼합 50/25/25 (기준 전략)")
    ax.axvline(pd.Timestamp("2017-01-01"), color=GREY, ls="--", lw=1); ax.text(pd.Timestamp("2017-03-01"), ax.get_ylim()[0] * 0.8, "시험 구간 →", color=GREY, fontsize=9)
    ax.set_ylabel("누적 수익 (%, 일간 로그 합)"); ax.legend(frameon=False, ncol=4, loc="upper left"); ax.set_title("3축과 혼합 전략의 누적 수익 (모두 목표 변동성 10%)", loc="left", fontsize=11, fontweight="bold")
    f.tight_layout(); f.savefig(FIG / "f03_axes.png", dpi=170); plt.close(f)


def hbar(ax, labels, vals, base=None, colors=None, fmt="{:.2f}", xlim=None, base_label=True):
    y = np.arange(len(labels))[::-1]
    tol = 0.05                                       # 기준과 ±0.05 이내는 '같은 수준'(회색) — 모든 차이가 통계적으로 유의하지 않음
    cols = colors or [(GREY if abs(v - base) < tol else GRN if v > base else RED) if base is not None else BLUE for v in vals]
    ax.barh(y, vals, color=cols, height=0.62)
    lo, hi = xlim if xlim else (0, max(vals)); off = 0.012 * (hi - lo)
    for yy, v in zip(y, vals):                       # 값 숫자는 막대 안쪽 끝(흰색)에 — 기준선과 겹치지 않게
        ax.text(v - off, yy, fmt.format(v), va="center", ha="right", fontsize=8.5, color="white", fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5)
    if base is not None:
        ax.axvline(base, color=INK, ls="--", lw=1.2)
        if base_label:
            ax.text(base, len(labels) - 0.2, f" 기준 {base:.2f}", fontsize=8.5, color=INK, va="bottom")
    if xlim:
        ax.set_xlim(*xlim)


def fig_summary():
    rows = []
    for k, nm in [("HMM", "D 현금 피신 · HMM 위기"), ("변동성", "D 현금 피신 · 변동성 국면"), ("동조성", "D 현금 피신 · DCC 동조성"), ("흡수비율", "D 현금 피신 · 흡수비율"), ("VIX", "D 현금 피신 · VIX"),
                  ("뉴스", "D 현금 피신 · 뉴스 감성"), ("칼만 평균 < 0", "D 현금 피신 · 칼만 평균<0"), ("낙폭", "D 현금 피신 · 낙폭 손절")]:
        x = best(SG, "신호", k)
        x = x[x["실험"].str.startswith("D")]
        if len(x):
            rows.append((nm, float(x[TE].iloc[0])))
    for k, nm in [("K 칼만", "C 적응형 · 칼만"), ("H HMM", "C 적응형 · HMM"), ("N 뉴스", "C 적응형 · 뉴스"), ("C DCC", "C 적응형 · DCC"), ("ALL", "C 적응형 · 4개 평균")]:
        x = best(SG, "신호", k)
        x = x[x["실험"].str.startswith("C")]
        if len(x):
            rows.append((nm, float(x[TE].iloc[0])))
    for key, nm in [("A", "A 연도별 갈아타기·트리거 (5년 최소분산)"), ("B", "B 메타 레이어 (κ=0.05)"), ("E", "E 동적 목표 변동성"), ("F", "F 상관 급등기 추세 집중"), ("K", "K 모멘텀 기간 (40일)"), ("J", "J 짧은 추세 (65일·3개)")]:
        x = MR[(MR["학습 최고"] == "←") & MR["실험"].astype(str).str.startswith(key)]
        if len(x):
            rows.append((nm, float(x[TE].iloc[0])))
    g = MR[(MR["실험"].astype(str).str.startswith("G")) & (MR["상한"] == "곡물 25%")]
    if len(g):
        rows.append(("G 곡물 비중 상한 25%", float(g[TE].iloc[0])))
    m = MM.sort_values("혼합 학습", ascending=False).iloc[0]; rows.append((f"멀티모달 갈래 (학습 최고: {m['신호'].replace('갈래 ', '').replace(' (부호 롤링)', '').replace(' (부호 고정)', '')})", float(m["혼합 시험"])))
    f, ax = plt.subplots(figsize=(10, 7)); hbar(ax, [r[0] for r in rows], [r[1] for r in rows], base=BASE_TE, xlim=(0.6, 1.55))
    ax.set_xlabel("시험 기간(2017–2026) Sharpe — 각 실험에서 학습 기간으로 고른 설정"); ax.set_title("신호 전략 대부분이 기준선 근처이거나 아래", loc="left", fontsize=11, fontweight="bold")
    f.tight_layout(); f.savefig(FIG / "f04_summary.png", dpi=170); plt.close(f)
    return rows


def fig_freq():
    X = FQ[FQ["전략"].str.contains("혼합 50/25/25|MinVar 축|모멘텀 축|가치 축|동일 비중")].copy()
    X["자료s"] = X["자료"].str[:2].str.strip()
    labs = ["MinVar 축", "모멘텀 축", "가치 축", "혼합 50/25/25", "혼합 동일 비중"]; keys = ["MinVar 축", "모멘텀 축", "가치 축", "혼합 50/25/25", "혼합 동일 비중"]
    f, ax = plt.subplots(figsize=(10, 3.8)); w = 0.27
    for j, (pre, nm, col) in enumerate([("W", "주간 자료", GREY), ("D1", "일간 · 주간과 같은 결정 주기", GRN), ("D2", "일간 · 모멘텀 매일 결정", GOLD)]):
        v = [float(X[(X["자료s"] == pre) & (X["전략"] == k)]["주간 환산 시험"].iloc[0]) for k in keys]
        ax.bar(np.arange(len(keys)) + (j - 1) * w, v, w, color=col, label=nm)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(labs); ax.set_ylabel("시험 2017– Sharpe (주 단위 환산)"); ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("같은 규칙을 주간 · 일간 자료로 돌린 결과", loc="left", fontsize=11, fontweight="bold")
    f.tight_layout(); f.savefig(FIG / "f05_freq.png", dpi=170); plt.close(f)


def fig_cash(E):
    sg = pd.read_pickle(D / "frozen" / "signals_daily_2026-10-26.pkl"); base = E["A|고정 50/25/25"].astype(float)
    import daily_sleeves as DS
    f, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.1, 1]})
    x = SG[(SG["실험"].str.startswith("D")) & (SG["학습 최고"] == "←")]
    labs = [str(s).replace(" (4개 중 k개)", "") for s in x["신호"]]; vals = x[TE].values
    hbar(a1, labs, vals, base=BASE_TE, xlim=(0.7, 1.5)); a1.set_title("신호별 시험 Sharpe (학습 최고 설정)", loc="left", fontsize=10, fontweight="bold")
    e = DS.exposure(list(DS.SIGNALS)[1], 0.9, 0.0, base, sg); r2 = DS.overlay(base, e)
    a2.plot(base.index, base.cumsum(), color=GREY, lw=1.6, label="신호 없음"); a2.plot(r2.index, r2.cumsum(), color=GRN, lw=1.6, label="VIX 분위 > 0.9 → 현금")
    off = e < 1; idx = e.index
    ymin, ymax = a2.get_ylim(); a2.fill_between(idx, ymin, ymax, where=off.values, color=RED, alpha=0.18, step="mid", label="현금 구간")
    a2.set_ylim(ymin, ymax); a2.legend(frameon=False, fontsize=8, loc="upper left"); a2.set_title("예: VIX가 높으면 다음 날부터 현금", loc="left", fontsize=10, fontweight="bold"); a2.set_ylabel("누적 수익 (%)")
    f.tight_layout(); f.savefig(FIG / "f06_cash.png", dpi=170); plt.close(f)


def fig_adaptive():
    labs = [("K 칼만", "칼만"), ("H HMM", "HMM"), ("N 뉴스", "뉴스"), ("C DCC", "DCC 동조성"), ("ALL", "4개 평균")]
    d, w_ = [], []
    for k, _ in labs:
        x = SG[(SG["실험"].str.startswith("C")) & (SG["신호"].astype(str).str.startswith(k)) & (SG["학습 최고"] == "←")]
        d.append(float(x[TE].iloc[0])); xw = ADAPT_W[(ADAPT_W["규칙"].astype(str).str.startswith(k)) & (ADAPT_W["학습 최고"] == "←")]
        w_.append(float(xw["Sharpe 시험 2017–"].iloc[0]))
    f, ax = plt.subplots(figsize=(9, 3.6)); xx = np.arange(len(labs)); ax.bar(xx - 0.2, w_, 0.4, color=GREY, label="주간 (기준 1.07)"); ax.bar(xx + 0.2, d, 0.4, color=BLUE, label="일간 (기준 1.32)")
    ax.axhline(1.07, color=GREY, ls="--", lw=1); ax.axhline(1.32, color=BLUE, ls="--", lw=1); ax.set_xticks(xx); ax.set_xticklabels([l for _, l in labs]); ax.set_ylabel("시험 Sharpe"); ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="lower right")
    ax.set_title("환경 적응형 비중 — 어떤 관점도 고정 비중을 못 넘음", loc="left", fontsize=11, fontweight="bold"); ax.set_ylim(0.6, 1.5)
    f.tight_layout(); f.savefig(FIG / "f07_adaptive.png", dpi=170); plt.close(f)


def fig_switch():
    A_ = MR[MR["실험"].astype(str).str.startswith("A")]
    pick = ["고정 50/25/25 (기준)", "고정 동일 비중", "연 1회 · 직전 5년 역변동성", "연 1회 · 직전 5년 축소(고정 50% + 접점 50%)", "연 1회 · 직전 5년 최소분산", "연 1회 · 직전 5년 최대 Sharpe(접점)",
            "T1 성과 트리거 · 130일 Sharpe < 0.5", "T2 위기 트리거 · 흡수비율 z > 1.0", "T3 전략 모멘텀 · 상위 k = 2", "T4 투자선 이탈 · 갭 %p > 1.0"]
    x = A_.set_index("규칙").loc[pick]
    f, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    lab = [p.replace("연 1회 · ", "").replace("(고정 50% + 접점 50%)", "").replace(" Sharpe < 0.5", "").replace("투자선 이탈 · ", "이탈 · ") for p in pick]
    hbar(a1, [s[:28] for s in lab], x[TE].values, base=float(x[TE].iloc[0]), xlim=(0.6, 1.5)); a1.set_title("시험 Sharpe (회색 = 기준과 ±0.05 이내)", loc="left", fontsize=10, fontweight="bold")
    dd = -x["최대 낙폭 (%)"].values; dd0 = float(dd[0])
    hbar(a2, [""] * len(pick), dd, colors=[GREY if abs(v - dd0) < 2 else GRN if v < dd0 else RED for v in dd], fmt="{:.0f}%"); a2.set_title("최대 낙폭 (깊을수록 위험)", loc="left", fontsize=10, fontweight="bold"); a2.set_yticks([])
    f.tight_layout(); f.savefig(FIG / "f08_switch.png", dpi=170); plt.close(f)


def fig_meta():
    B_ = MR[MR["실험"].astype(str).str.startswith("B")].copy(); kap = B_[B_["κ"] > 0]
    f, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.plot([0] + list(kap["κ"]), [BASE_TE] + list(kap[TE]), "o-", color=BLUE); a1.axhline(BASE_TE, color=INK, ls="--", lw=1); a1.set_xlabel("기울이기 세기 κ"); a1.set_ylabel("시험 Sharpe"); a1.set_title("시험 Sharpe는 조금 높지만…", loc="left", fontsize=10, fontweight="bold")
    a2.bar(["학습 2008–16", "시험 2017–", "무작위"], [float(kap["1등 축 적중률 학습"].iloc[0]), float(kap["1등 축 적중률 시험"].iloc[0]), 1 / 3], color=[BLUE, GRN, GREY]); a2.set_ylim(0, 0.5); a2.set_ylabel("1등 축 적중률")
    for i, v in enumerate([float(kap["1등 축 적중률 학습"].iloc[0]), float(kap["1등 축 적중률 시험"].iloc[0]), 1 / 3]):
        a2.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=9)
    a2.set_title("…맞히는 확률은 무작위(33%) 이하", loc="left", fontsize=10, fontweight="bold")
    f.tight_layout(); f.savefig(FIG / "f09_meta.png", dpi=170); plt.close(f)


def fig_dyn():
    E_ = MR[MR["실험"].astype(str).str.startswith("E")]
    x = E_[~E_["규칙"].astype(str).str.startswith("고정 목표 10")]
    f, ax = plt.subplots(figsize=(9, 3.8)); hbar(ax, [str(s)[:34] for s in x["규칙"]], x[TE].values, base=BASE_TE, xlim=(1.1, 1.38))
    ax.set_title("동적 목표 변동성 — 속도를 바꿔도 Sharpe는 그대로(낙폭만 줄어드는 건 덜 투자한 몫)", loc="left", fontsize=10, fontweight="bold"); ax.set_xlabel("시험 Sharpe")
    f.tight_layout(); f.savefig(FIG / "f10_dyn.png", dpi=170); plt.close(f)


def fig_corr():
    F_ = MR[MR["실험"].astype(str).str.startswith("F")]
    ctrl = F_[F_["규칙"].astype(str).str.contains("대조군")]; trig = F_[F_["규칙"] == "트리거"]
    f, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    hbar(a1, [str(s).replace(" (대조군)", "") for s in ctrl["규칙"]], ctrl[TE].values, base=BASE_TE, xlim=(0.3, 1.5)); a1.set_title("늘 켜 두면(대조군): 모두 기준 아래", loc="left", fontsize=10, fontweight="bold")
    g = trig.groupby("트리거")["켜진 날 (%)"].first().sort_values(); a2.barh(range(len(g)), g.values, color=GOLD)
    a2.set_yticks(range(len(g))); a2.set_yticklabels([s[:22] for s in g.index], fontsize=8.5)
    for i, v in enumerate(g.values):
        a2.text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=8.5)
    a2.set_title("트리거가 켜진 날(전체 기간 중 %)", loc="left", fontsize=10, fontweight="bold")
    f.tight_layout(); f.savefig(FIG / "f11_corr.png", dpi=170); plt.close(f)


def fig_lookback():
    K = MR[MR["실험"].astype(str).str.startswith("K")]
    f, ax = plt.subplots(figsize=(9, 3.6)); xs = ["여러 기간 평균" if "여러" in str(s) else str(s).replace(" (기준)", "\n(기준)") for s in K["규칙"]]
    xx = np.arange(len(K)); ax.plot(xx, K[TR], "o-", color=BLUE, label="학습 2008–16"); ax.plot(xx, K[TE], "s-", color=GRN, label="시험 2017–")
    ax.set_xticks(xx); ax.set_xticklabels(xs, fontsize=9); ax.set_xlabel("모멘텀을 보는 기간"); ax.set_ylabel("혼합 Sharpe"); ax.legend(frameon=False)
    ax.set_title("추세를 오래 볼수록 좋았다 — 짧게 보면 잦은 방향 전환으로 손해", loc="left", fontsize=11, fontweight="bold")
    f.tight_layout(); f.savefig(FIG / "f12_lookback.png", dpi=170); plt.close(f)


def fig_cap():
    G = MR[MR["실험"].astype(str).str.startswith("G")]
    f, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.3, 1]})
    hbar(a1, [str(s)[:30] for s in G["상한"]], G[TE].values, base=float(G[TE].iloc[0]), xlim=(1.2, 1.4)); a1.set_title("시험 Sharpe", loc="left", fontsize=10, fontweight="bold")
    hbar(a2, [""] * len(G), G[TR].values, base=float(G[TR].iloc[0]), xlim=(0.5, 0.7), base_label=False); a2.set_yticks([]); a2.set_title("학습 Sharpe — 축산 상한은 학습이 깎임", loc="left", fontsize=10, fontweight="bold")
    f.tight_layout(); f.savefig(FIG / "f13_cap.png", dpi=170); plt.close(f)


def fig_mm():
    M = MM[MM["신호"].str.contains("롤링")].copy(); M["n"] = M["신호"].str.replace("갈래 ", "").str.replace(" (부호 롤링)", "")
    f, ax = plt.subplots(figsize=(9, 3.9)); hbar(ax, list(M["n"]), M["혼합 시험"].values, base=BASE_TE, xlim=(0.5, 1.5))
    ax.set_title("멀티모달 갈래를 25% 섞은 혼합의 시험 Sharpe — 모두 기준 아래", loc="left", fontsize=11, fontweight="bold"); ax.set_xlabel("시험 Sharpe")
    f.tight_layout(); f.savefig(FIG / "f14_mm.png", dpi=170); plt.close(f)


def fig_signals(E):
    import daily_sleeves as DS
    sg = pd.read_pickle(D / "frozen" / "signals_daily_2026-10-26.pkl")
    f, axs = plt.subplots(3, 2, figsize=(11, 8.4), sharex=True)
    spec = [("동조성", "DCC 동조성 (분위)", 0.9), ("VIX", "VIX (분위)", 0.9), ("HMM", "HMM 위기 확률", 0.9), ("변동성", "원자재 변동성 국면 (배수)", 1.5), ("흡수비율", "흡수비율 z", 1.5), ("뉴스", "뉴스 감성 악화 (감성 z의 부호를 뒤집은 값)", 1.0)]
    for ax, (c, t, th) in zip(axs.ravel(), spec):
        s = sg[c].dropna(); ax.plot(s.index, s.values, color=BLUE, lw=0.8); ax.axhline(th, color=RED, ls="--", lw=1.1); ax.fill_between(s.index, th, s.values, where=s.values > th, color=RED, alpha=0.3)
        ax.set_title(f"{t} · 기본 문턱 {th}", loc="left", fontsize=9.5, fontweight="bold"); ax.axvline(pd.Timestamp("2017-01-01"), color=GREY, ls=":", lw=1)
    f.suptitle("일간 신호와 문턱 (빨간 영역 = '위험' → 노출 축소) · 점선 세로선 = 시험 구간 시작", x=0.01, ha="left", fontsize=11, fontweight="bold")
    f.tight_layout(rect=(0, 0, 1, 0.96)); f.savefig(FIG / "f15_signals.png", dpi=170); plt.close(f)


def fig_overlay(E):
    import daily_sleeves as DS
    sg = DS.load_signals(); base = E["A|고정 50/25/25"].astype(float)
    f, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (nm, thr, low) in zip(axs, [(list(DS.SIGNALS)[1], 0.9, 0.0), (list(DS.SIGNALS)[-1], 5.0, 0.5)]):
        e = DS.exposure(nm, thr, low, base, sg); r2 = DS.overlay(base, e)
        ax.plot(base.index, base.cumsum(), color=GREY, lw=1.5, label="신호 없음"); ax.plot(r2.index, r2.cumsum(), color=GRN, lw=1.5, label="신호 적용")
        yl = ax.get_ylim(); ax.fill_between(e.index, yl[0], yl[1], where=(e < 1).values, color=RED, alpha=0.15, step="mid"); ax.set_ylim(yl)
        ax.set_title(f"{nm.split(' (')[0]} · 문턱 {thr:g} · 위험 시 노출 {int(low * 100)}%", loc="left", fontsize=9.5, fontweight="bold"); ax.legend(frameon=False, fontsize=8, loc="upper left")
    f.tight_layout(); f.savefig(FIG / "f16_overlay.png", dpi=170); plt.close(f)


def make_figs():
    E, S = daily_series()
    fig_structure(); fig_train_test(); fig_axes(S, E); rows = fig_summary(); fig_freq(); fig_cash(E); fig_adaptive(); fig_switch(); fig_meta(); fig_dyn(); fig_corr(); fig_lookback(); fig_cap(); fig_mm(); fig_signals(E); fig_overlay(E)
    return rows


# ---------------------------------------------------------------- Word 도구
def setfont(run, size=10.5, bold=False, color=None, italic=False):
    run.font.name = FONT; run.font.size = Pt(size); run.bold = bold; run.italic = italic
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    if color:
        run.font.color.rgb = RGBColor.from_string(color.lstrip("#"))


def shade(cell, fill):
    tcPr = cell._element.get_or_add_tcPr(); sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), fill); tcPr.append(sh)


class Doc:
    def __init__(self):
        self.d = Document(); s = self.d.sections[0]
        s.page_width, s.page_height = Cm(21), Cm(29.7); s.left_margin = s.right_margin = Cm(1.9); s.top_margin = s.bottom_margin = Cm(1.8)
        st = self.d.styles["Normal"]; st.font.name = FONT; st.font.size = Pt(10.5); st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        for nm, sz, col in [("Heading 1", 17, INK), ("Heading 2", 13.5, GRN), ("Heading 3", 11.5, INK)]:
            h = self.d.styles[nm]; h.font.name = FONT; h.font.size = Pt(sz); h.font.bold = True; h.font.color.rgb = RGBColor.from_string(col.lstrip("#")); h.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
            h.paragraph_format.space_before = Pt(14 if nm == "Heading 1" else 10); h.paragraph_format.space_after = Pt(5); h.paragraph_format.keep_with_next = True
        fp = s.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = fp.add_run(); setfont(r, 8.5, color="888888")
        for t, txt in (("begin", None), (None, "PAGE"), ("end", None)):
            if t:
                el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), t); r._r.append(el)
            else:
                it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = txt; r._r.append(it)
        z = self.d.settings.element.find(qn("w:zoom"))                      # python-docx 기본 템플릿: zoom에 percent가 없어 XSD 검증 오류
        if z is not None and z.get(qn("w:percent")) is None:
            z.set(qn("w:percent"), "100")
        self.W = 17.2

    def h(self, t, lvl=1):
        return self.d.add_heading(t, lvl)

    def p(self, t, size=10.5, bold=False, color=None, after=4, italic=False, align=None):
        para = self.d.add_paragraph(); para.paragraph_format.space_after = Pt(after)
        if align:
            para.alignment = align
        for k, seg in enumerate(t if isinstance(t, list) else [t]):
            if isinstance(seg, tuple):
                setfont(para.add_run(seg[0]), size, True, seg[1] if len(seg) > 1 else color)
            else:
                setfont(para.add_run(seg), size, bold, color, italic)
        return para

    def b(self, t, size=10.5):
        para = self.d.add_paragraph(style="List Bullet"); para.paragraph_format.space_after = Pt(2)
        for seg in (t if isinstance(t, list) else [t]):
            if isinstance(seg, tuple):
                setfont(para.add_run(seg[0]), size, True, seg[1] if len(seg) > 1 else None)
            else:
                setfont(para.add_run(seg), size)
        return para

    def img(self, f, width_cm, cap):
        self.d.add_picture(str(FIG / f), width=Cm(width_cm)); self.d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c = self.d.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER; c.paragraph_format.space_after = Pt(8); setfont(c.add_run(cap), 8.5, color="666666")

    def box(self, title, lines, fill="EEF4EF", bar=GRN):
        t = self.d.add_table(rows=1, cols=1); t.alignment = WD_TABLE_ALIGNMENT.CENTER; c = t.cell(0, 0); shade(c, fill); c.width = Cm(self.W)
        p0 = c.paragraphs[0]; setfont(p0.add_run(title), 10.5, True, bar.lstrip("#") if bar.startswith("#") else bar)
        for ln in lines:
            pp = c.add_paragraph(); pp.paragraph_format.space_after = Pt(2)
            for seg in (ln if isinstance(ln, list) else [ln]):
                if isinstance(seg, tuple):
                    setfont(pp.add_run(seg[0]), 9.5, True)
                else:
                    setfont(pp.add_run(seg), 9.5)
        self.d.add_paragraph().paragraph_format.space_after = Pt(2)

    def table(self, head, rows, widths_cm, size=9, hl=(), center_from=99):
        t = self.d.add_table(rows=1, cols=len(head)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, h in enumerate(head):
            c = t.rows[0].cells[i]; shade(c, "1E3A2F"); c.width = Cm(widths_cm[i]); c.paragraphs[0].text = ""; setfont(c.paragraphs[0].add_run(h), size, True, "FFFFFF")
        for k, r in enumerate(rows):
            cells = t.add_row().cells
            for i, v in enumerate(r):
                cells[i].width = Cm(widths_cm[i]); cells[i].paragraphs[0].text = ""
                if k in hl:
                    shade(cells[i], "E8F3EC")
                elif k % 2:
                    shade(cells[i], "F7F8F6")
                run = cells[i].paragraphs[0].add_run(str(v)); setfont(run, size, bold=(k in hl))
                if i >= center_from:
                    cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.d.add_paragraph().paragraph_format.space_after = Pt(2)

    def pb(self):
        self.d.add_page_break()


def fm(x, nd=2):
    return "–" if pd.isna(x) else f"{x:+.{nd}f}" if nd else f"{x:.0f}"


def bundle_head(doc, num, title, one, analogy):
    doc.h(f"{num}. {title}", 1)
    doc.box("한 줄 요약", [one]); doc.box("쉬운 비유", [analogy], fill="FBF5E6", bar="#8A6D1E")


def main():
    make_figs()
    doc = Doc(); d = doc
    # ---------- 표지
    d.p("일간 데이터 백테스트 — 전략 쉬운 설명", 24, True, INK, 2)
    d.p("어떤 규칙으로 사고파는지, 설정은 무엇이고, 결과는 어땠나", 13, False, "5B6575", 6)
    d.p("2026-11-10 · 자료 기준일 2026-10-26 · 상세 기록: docs/백테스트_로그.md (R · S · T) · 대시보드 ⑤ 포트폴리오 › F 서브탭", 9.5, color="888888", after=10)
    d.box("이 문서가 하는 일", [
        "대시보드에 들어 있는 일간 백테스트 전략을 '무엇을 언제 얼마나 사고파는가' 중심으로 설명합니다. 모델의 수학 설명은 최소화하고, 각 전략 묶음마다 ① 한 줄 요약 ② 비유 ③ 사고파는 순서 ④ 세부 설정표 ⑤ 결과 그림 ⑥ 판정을 담았습니다.",
        [("결론 먼저: ", ), "기준 전략(MinVar 50 · 모멘텀 25 · 가치 25)에 신호를 얹어 비중이나 투자 규모를 바꾸는 방법은 일간 자료에서도 모두 기준을 의미 있게 넘지 못했습니다(미채택). 주간 자료 결과와 같은 결론입니다."]])
    d.h("차례", 2)
    for t in ["0. 먼저 알아둘 말", "1. 기준 전략: 3축 혼합", "2. 결과 한눈에", "3. 3축 혼합 — 주간 vs 일간", "4. D 현금 피신", "5. C 환경 적응형 비중", "6. A 연도별 갈아타기 · 트리거", "7. B 메타 레이어", "8. E 동적 목표 변동성",
              "9. F 상관 급등기 추세 집중", "10. J · K 짧은 추세 · 모멘텀 기간", "11. G · I 품목군 비중 상한", "12. 멀티모달 갈래 (리포트 · 기상 · COT · 문서)", "13. 일간 신호 오버레이 8종 (대시보드에서 직접 섞어 보기)", "14. 드롭박스의 재료 전략 한 줄 설명", "15. 종합 판정과 읽을 때 주의할 점"]:
        d.p(t, 10, after=1)
    d.pb()
    # ---------- 0
    d.h("0. 먼저 알아둘 말", 1)
    d.table(["용어", "쉬운 설명"], [
        ["일간 / 주간 자료", "가격을 매일 보는지, 매주 금요일 종가만 보는지. 일간 모드는 같은 달력 기간을 쓰도록 창을 5배로 늘렸습니다(52주 ≈ 260영업일, 4주 = 20일)."],
        ["변동성", "가격이 얼마나 출렁이는가. 클수록 위험합니다."],
        ["목표 변동성 10%", "자동차 크루즈컨트롤. 최근 130일의 출렁임을 보고 연 10% 수준이 되도록 투자 규모를 늘리거나 줄입니다(최대 4배)."],
        ["Sharpe", "위험 한 단위당 번 수익. 1이면 꽤 좋은 편이고, 0이면 위험만 졌다는 뜻입니다. ±0.45 안쪽 차이는 우연과 구별하기 어렵습니다."],
        ["학습 / 시험", "2008–2016에서 규칙과 설정을 고르고, 2017–2026은 그대로 적용해 확인합니다. 과거 문제집으로 공부하고 새 문제로 시험 보는 것과 같습니다."],
        ["위약(플라시보)", "신호 시점을 무작위로 섞은 가짜 전략. 진짜 신호가 가짜보다 확실히 좋아야 효과가 있다고 봅니다(위약 p가 작을수록 좋음)."],
        ["기준 전략", "일간 3축 고정 50/25/25. 일간 학습 Sharpe 0.61 · 시험 1.32 (주 단위로 합산하면 시험 1.24, 주간 자료 기준 1.07)."],
        ["차이 t", "기준과의 성과 차이가 우연이 아닌 정도. 절댓값이 2보다 작으면 유의하지 않습니다."],
    ], [3.6, 13.6], size=9.5)
    d.img("f02_traintest.png", 16.5, "그림 0. 설정은 학습 구간에서만 고르고, 시험 구간에서 확인")
    # ---------- 1
    d.h("1. 기준 전략: 3축 혼합", 1)
    d.p("모든 실험은 같은 기준 전략 위에서 시작합니다. 세 가지 전략(축)을 따로 만들고 수익률을 50 : 25 : 25로 섞습니다.")
    d.table(["축", "사고파는 방법 (쉬운 말)", "세부 설정"], [
        ["① MinVar 롱\n(안전형)", "덜 출렁이고 서로 다르게 움직이는 품목에 더 많이 담습니다. 가격이 오를지 예측하지 않고, 포트폴리오가 가장 안 흔들리게만 만듭니다. 롱(매수)만.", "최근 520일(104주) 수익률로 위험을 최소화하는 비중 계산 · 품목당 최대 25% · 20일(4주)마다 다시 계산 · 공분산은 Ledoit-Wolf 방식으로 안정화"],
        ["② 모멘텀\n(추세 추종)", "지난 1년 오른 품목은 사고, 내린 품목은 팝니다(품목마다 따로 판단). 출렁임이 작은 품목에 더 많이 배분합니다.", "관찰 기간 260일(52주) · 비중 ∝ 목표 변동성 ÷ 품목 변동성 · 5일마다 갱신 · 롱·숏 모두"],
        ["③ 가치\n(평균회귀)", "5년 평균보다 많이 싸진 품목 1/3은 사고, 많이 비싸진 품목 1/3은 팝니다. '결국 평균으로 돌아온다'는 가정입니다.", "기준 평균 1,300일(5년) · 상위·하위 1/3 · 20일마다 · 롱·숏 모두"],
        ["섞기 + 속도 조절", "세 축 수익률을 50/25/25로 더한 뒤, 다시 연 10% 출렁임이 되도록 투자 규모를 조절합니다.", "목표 변동성 10% · 최근 130일 실현 변동성 · 레버리지 상한 4배 · 편도 비용 5bp"],
    ], [2.7, 8.2, 6.3], size=9)
    d.img("f01_structure.png", 16.5, "그림 1. 기준 전략의 구조 — 신호 전략은 '섞는 비중' 또는 '투자 규모'를 바꾸려는 시도입니다")
    d.img("f03_axes.png", 16.5, "그림 2. 세 축과 혼합 전략의 누적 수익 — 혼합이 가장 안정적(세 축이 서로 다른 때 번다)")
    d.box("왜 50/25/25인가", ["후보 14개를 학습 기간(2007–2016) 성과로 줄 세운 1위였고, 시험 기간에도 유지돼 채택했습니다. 다만 같은 학습 기간에서 동일 비중(33/33/33)이 더 높게 나온 점은 기록해 두었습니다(차이는 유의하지 않음)."], fill="F1F1F8", bar="#4A3F8F")
    d.pb()
    # ---------- 2
    rows = [("", 0)]
    d.h("2. 결과 한눈에", 1)
    d.p("아래는 각 실험에서 학습 기간(2008–2016)만 보고 고른 설정이 시험 기간(2017–2026)에 얼마나 나왔는지입니다. 점선은 기준 전략의 시험 Sharpe(1.32)입니다. 회색 = 기준과 ±0.05 이내(같은 수준), 초록 = 0.05 이상 높음, 빨강 = 0.05 이상 낮음.")
    d.img("f04_summary.png", 16.5, "그림 3. 시험 Sharpe 비교 — 기준을 눈에 띄게 넘은 것은 B 메타 레이어(+0.05) 하나이고, 그것도 우연 범위입니다")
    d.box("읽는 법", ["회색(같은 수준)이 많다는 것은 신호를 얹어도 성과가 달라지지 않았다는 뜻입니다. 기준보다 높게 나온 것(초록)도 차이가 작고(+0.05 안팎), 통계적으로 유의하지 않거나(차이 t < 2) 학습 기간에는 개선이 없었습니다. 그래서 어느 것도 '채택'하지 않았습니다."])
    # ---------- 3
    bundle_head(d, 3, "3축 혼합 — 주간 vs 일간", "같은 규칙을 주간 자료와 일간 자료에서 돌렸을 때 성과가 유지되는지 확인했습니다.",
                "같은 요리법으로 하루 세 번 간을 보는 것과 일주일에 한 번 보는 것의 차이 — 맛(성과)이 크게 달라지는지 봅니다.")
    d.h("사고파는 순서", 2)
    for t in ["주간 자료(W): 금요일 종가로 결정 → 다음 주 수익으로 평가.", "일간 자료 D1: 결정 주기는 주간과 같게(모멘텀 5일 · 가치 20일 · MinVar 20일) 두고 손익과 변동성만 일간으로 계산.", "일간 자료 D2: 모멘텀을 매일 다시 결정(가치 5일 · MinVar 20일)."]:
        d.b(t)
    d.h("세부 설정", 2)
    d.table(["설정", "주간 (W)", "일간 D1", "일간 D2"], [["MinVar 창 / 재계산", "104주 / 4주", "520일 / 20일", "520일 / 20일"], ["모멘텀 관찰 / 갱신", "52주 / 매주", "260일 / 5일", "260일 / 매일"], ["가치 평균 기간 / 갱신", "260주 / 4주", "1,300일 / 20일", "1,300일 / 5일"], ["변동성 맞추기 창", "26주", "130일", "130일"], ["비용 · 목표", "편도 5bp · 10%", "편도 5bp · 10%", "편도 5bp · 10%"]], [4.6, 4.2, 4.2, 4.2], size=9)
    d.img("f05_freq.png", 16, "그림 4. 주 단위로 환산한 시험 Sharpe — 일간 자료로 바꿔도 떨어지지 않음(혼합 1.07 → 1.24)")
    fq = FQ[FQ["전략"] == "혼합 50/25/25"].set_index(FQ[FQ["전략"] == "혼합 50/25/25"]["자료"].str[:2].str.strip())
    d.p([("결과: ", ), f"혼합 50/25/25의 시험 Sharpe는 주간 {fq.loc['W', '주간 환산 시험']:.2f} → 일간 D1 {fq.loc['D1', '주간 환산 시험']:.2f}(주 단위 환산). 학습 기간은 {fq.loc['W', '주간 환산 학습']:.2f}로 같고 시험 기간만 올랐습니다(연 수익 +2.4%p, 차이 t 1.57 — 유의하지 않음). "
         "일간으로 바꿔도 '나빠지지 않는다'가 정확한 결론입니다. 모멘텀을 매일 다시 결정(D2)해도 이득이 없었습니다."])
    d.pb()
    # ---------- 4 D
    bundle_head(d, 4, "D 현금 피신", "위험 신호가 켜지면 투자 규모를 줄여(현금으로) 손실을 피하려는 전략입니다.",
                "비 오는 날 우산(현금)을 챙기는 것 — 비 예보(신호)가 얼마나 잘 맞는지가 전부입니다. 틀리면 맑은 날에도 우산만 들고 다닙니다.")
    d.h("사고파는 순서", 2)
    for t in ["매일 종가 뒤에 신호 값을 확인합니다(그날까지의 정보만).", "신호가 문턱을 넘으면 → 다음 날부터 투자 규모를 0% 또는 50%로 줄입니다(나머지는 현금, 이자 0).", "신호가 문턱 아래로 돌아오면 → 다시 100%로 복귀합니다.", "투자 규모를 바꿀 때마다 비용 = 바꾼 크기 × 1.15 × 5bp. 현금으로 줄인 뒤 다시 키우지 않습니다."]:
        d.b(t)
    d.h("세부 전략 세팅 (9가지 신호 · 모두 학습에서 가장 좋은 문턱 · 노출을 골랐습니다)", 2)
    d.table(["신호", "무엇을 재나 (쉬운 말)", "시험해 본 문턱", "위험 시 노출"], [
        ["HMM 위기", "시장이 '불안 국면'일 확률(주간 확률을 일간으로 이어 붙임)", "0.5 · 0.7 · 0.9", "50% · 0%"],
        ["변동성 국면", "최근 3개월 출렁임이 평소(중앙값) 대비 몇 배인지", "1.2 · 1.5 · 2.0배", "50% · 0%"],
        ["DCC 동조성", "원자재 8개가 같은 방향으로 움직이는 정도의 역사적 위치", "분위 0.8 · 0.9 · 0.95", "50% · 0%"],
        ["흡수비율", "여러 품목의 움직임을 하나의 공통 요인이 설명하는 정도가 20일 새 얼마나 늘었나", "z 1.0 · 1.5 · 2.0", "50% · 0%"],
        ["VIX", "미국 주식 시장의 공포 지수가 역사적으로 얼마나 높은 위치인가", "분위 0.8 · 0.9 · 0.95", "50% · 0%"],
        ["뉴스 감성", "연준 · USDA · EIA 문서의 분위기가 나빠진 정도(문서 공표일에만 바뀜)", "z 0.5 · 1.0 · 1.5", "50% · 0%"],
        ["칼만 평균 < 0", "내 혼합 전략의 '요즘 평균 수익'(부드럽게 추적)이 0보다 작아졌나", "0 (고정)", "50% · 0%"],
        ["낙폭 손절", "혼합 전략 누적 수익이 고점에서 −x% 아래로 떨어졌나(회복은 −x/2%에서)", "x = 5 · 10 · 15%", "50% · 0%"],
        ["결합", "HMM · 변동성 · 동조성 · VIX 중 k개가 동시에 켜졌나", "k = 1 · 2 · 3", "50% · 0%"],
    ], [2.6, 8.0, 3.9, 2.7], size=8.8)
    d.p("함께 본 대조군: 신호 없이 늘 70% 또는 50%만 투자(타이밍 효과와 '덜 투자한 몫'을 구분). 위약: 현금 구간을 65일 블록으로 섞어 같은 현금 비율로 200번 시험.", 9.5, color="5B6575")
    d.img("f06_cash.png", 17, "그림 5. (왼쪽) 신호별 시험 Sharpe (오른쪽) VIX 신호를 쓴 예: 빨간 구간이 현금으로 피한 기간")
    x = SG[(SG["실험"].str.startswith("D")) & (SG["학습 최고"] == "←")]
    d.p([("결과: ", ), f"어떤 신호도 시험 기준 {BASE_TE:.2f}를 의미 있게 넘지 못했습니다(최대 +0.04). HMM({float(x[x['신호'].str.startswith('HMM')][TE].iloc[0]):.2f})과 변동성({float(x[x['신호'].str.startswith('변동성')][TE].iloc[0]):.2f})은 오히려 크게 나빴습니다 — 위기 때 모멘텀이 버는 구간을 현금으로 놓치기 때문입니다. "
         "낙폭이 줄어든 것(−24.8% → 낙폭 손절 −16.8%)은 평균 투자 비중이 72%로 줄어든 몫이 큽니다: 신호 없이 늘 70%만 투자해도 낙폭 −17.3%에 Sharpe는 같습니다. 흡수비율의 위약 p 0.04는 현금 구간이 거의 없어(평균 노출 99%) 의미가 작습니다."])
    d.box("판정: 미채택", ["주간 자료 결과(HMM 0.92 · 변동성 0.63 …)와 같은 결론입니다. 현금 이자를 넣어도(선물 구조에서는 추가 이자 없음) 달라지지 않았습니다."], fill="FDECEA", bar="#C8553D")
    d.pb()
    # ---------- 5 C
    bundle_head(d, 5, "C 환경 적응형 비중", "시장 환경 신호에 따라 50/25/25 비중을 최대 ±25%p까지 살짝 조정합니다.",
                "기본 레시피는 그대로 두되, 날씨(시장 환경)에 따라 소금을 조금 더하거나 덜어내는 것.")
    d.h("사고파는 순서", 2)
    for t in ["20일마다 신호를 보고 3축의 '앞으로 잘될 정도'(기대수익)를 계산합니다.", "기본 비중(50/25/25)이 가장 좋다는 전제에서 출발해, 신호가 가리키는 쪽으로 기대수익을 λ만큼 기울입니다(블랙-리터만 방식).", "새 기대수익으로 최대 Sharpe 비중을 구하되, 기본 비중에서 ±25%p 안, 롱만, 합 100%.", "합친 뒤 목표 변동성 10%로 맞춥니다. 비중 바꾸는 비용 = |Δ비중| × 1.15 × 5bp."]:
        d.b(t)
    d.h("세부 전략 세팅", 2)
    d.table(["관점(신호)", "의미", "세팅"], [
        ["K 칼만 평균 추적", "축마다 '요즘 평균 수익'을 부드럽게 추적해 잘 안 되는 축을 줄임", "추적 속도 q = 2e-5 · 2e-4 · 2e-3 (주간 1e-4 · 1e-3 · 1e-2와 같은 달력 속도), 반영 강도 λ = 0.25 · 0.5 · 1.0"],
        ["H HMM 위기 국면", "위기 확률에 따라 축 성과가 어떻게 달랐는지 회귀로 학습", "λ = 0.25 · 0.5 · 1.0 (주간 확률 이어 붙임)"],
        ["N 뉴스 감성", "문서 감성에 따른 축 성과를 회귀로 학습", "λ = 0.25 · 0.5 · 1.0"],
        ["C DCC 동조성", "품목 동조성이 높을 때/낮을 때 축 성과를 회귀로 학습", "λ = 0.25 · 0.5 · 1.0"],
        ["ALL 4개 평균", "위 4관점의 평균", "q · λ 조합"],
        ["D 위험 균등(관점 없음)", "3축이 위험을 똑같이 나눠 갖도록(3축 DCC 공분산)", "신호 없음"],
    ], [3.4, 6.8, 7.0], size=8.8)
    d.img("f07_adaptive.png", 14.5, "그림 6. 주간(회색) · 일간(파랑) 시험 Sharpe — 점선은 각 기준")
    d.p([("결과: ", ), "일간에서도 어떤 관점도 고정 50/25/25를 넘지 못했습니다(시험 차이 t −2.0 ~ −0.7). 칼만은 주간과 같이 유의하게 나빴습니다(학습에서는 좋았다가 시험에서 역전). 4개 평균은 위약 중앙값(1.15)보다도 낮았습니다(실제 1.12)."])
    d.box("판정: 미채택", ["신호가 가리키는 쪽으로 비중을 옮기면 한 박자씩 늦어 오히려 손해였습니다. 동일 비중(1.36)과 위험 균등(1.30)은 기준과 같은 수준입니다."], fill="FDECEA", bar="#C8553D")
    d.pb()
    # ---------- 6 A
    bundle_head(d, 6, "A 연도별 갈아타기 · 트리거", "해마다(또는 신호가 켜질 때) 최근 성적표를 보고 3축 비중을 다시 정합니다.",
                "매년 말 성적표를 보고 '올해는 어느 과목에 더 시간을 쓸지' 다시 정하는 공부 계획 — 성적표가 미래를 보장하지는 않습니다.")
    d.h("사고파는 순서", 2)
    for t in ["연 1회: 해마다 마지막 거래일에 직전 3년 또는 5년의 일간 수익률로 비중을 다시 계산합니다(미래 정보 없음).", "트리거: 평소엔 50/25/25를 유지하다가, 신호가 켜지면 갈아탑니다.", "갈아탈 때 비용 = |비중 변화| × 1.15 × 5bp. 평가는 5년 창이 처음 차는 2012년부터(학습 2012–16 · 시험 2017–)."]:
        d.b(t)
    d.h("세부 전략 세팅", 2)
    d.table(["종류", "규칙(쉬운 말)", "세팅"], [
        ["연 1회 · 최대 Sharpe(접점)", "지난 N년에 가장 성적 좋았던 조합으로", "N = 3년 · 5년"],
        ["연 1회 · 최소분산", "지난 N년에 가장 덜 출렁인 조합으로", "N = 3년 · 5년"],
        ["연 1회 · 역변동성", "덜 출렁인 축에 더 많이", "N = 3년 · 5년"],
        ["연 1회 · 축소", "고정 비중 50% + 접점 50%를 섞어 극단 방지", "N = 3년 · 5년"],
        ["T1 성과 트리거", "최근 130일 Sharpe가 문턱보다 낮으면 5년 접점으로 재계산(65일에 한 번만)", "문턱 −0.5 · 0 · 0.5"],
        ["T2 위기 트리거", "흡수비율 z가 문턱을 넘으면 MinVar 100%로 대피", "z 1.0 · 1.5 · 2.0"],
        ["T3 전략 모멘텀", "최근 1년 가장 잘한 축(상위 k개)에만 투자", "k = 1 · 2 (20일마다)"],
        ["T4 투자선 이탈", "내 비중이 지난 5년 효율적 투자선보다 갭(%p) 이상 낮으면 재계산", "갭 1 · 2 · 4%p (20일마다 점검, 130일 간격)"],
    ], [3.8, 8.7, 4.7], size=8.8)
    d.img("f08_switch.png", 17, "그림 7. 시험 Sharpe(왼쪽)와 최대 낙폭(오른쪽) — 갈아타기는 기준보다 낮거나 같고, T2 · T3는 낙폭이 크게 깊어짐")
    d.p([("결과: ", ), "동일 비중 · 역변동성 · 5년 축소는 기준과 같은 수준(1.35~1.39, 차이 t 0.2~0.3)이지만, 최대 Sharpe(접점)를 따라가면 1.14~1.21로 낮았고 T2(위기 대피)는 낙폭이 −44.7%, T3(최근 1등 축 몰기)는 −59.9%로 훨씬 위험했습니다. 최근 성적은 미래를 잘 알려주지 않습니다."])
    d.box("판정: 미채택", ["주간 결과(접점 0.82~0.89 · T3 학습 1.09 → 시험 0.46)와 같은 방향입니다. 동일 비중 · 역변동성은 학습 · 시험 모두 기준 이상이지만 차이가 유의하지 않아 '참고'로만 기록했습니다."], fill="FDECEA", bar="#C8553D")
    d.pb()
    # ---------- 7 B
    bundle_head(d, 7, "B 메타 레이어", "다음 3개월 동안 어느 축이 상대적으로 잘할지 점쳐서 그 축에 조금 더 싣습니다.",
                "다음 분기 우승 후보를 점치는 예상꾼. 맞히면 베팅(비중)을 늘리지만, 맞히는 확률이 동전 던지기 수준이면 소용이 없습니다.")
    d.h("사고파는 순서", 2)
    for t in ["6가지 시장 상태 지표(레이어)를 만듭니다.", "매년 말 그때까지의 자료로 릿지 회귀를 학습해 '다음 13주(65일) 동안 어느 축이 상대적으로 잘할지'를 예측합니다.", "20일마다, 예측 1등 축 비중에 +κ, 꼴등 축에 −κ(음수는 0으로, 합 100%)."]:
        d.b(t)
    d.h("세부 전략 세팅", 2)
    d.table(["레이어", "무엇을 보나"], [["L1 흡수비율", "품목들이 한 공통 요인에 묶이는 정도와 최근 변화"], ["L2 변동성 국면", "시장 출렁임이 평소 대비 큰가"], ["L3 추세 강도", "품목들의 1년 추세가 얼마나 강한가"], ["L4 가치 스프레드", "싼 품목과 비싼 품목의 가격 괴리"], ["L5 축 자체 모멘텀", "세 축 각각의 최근 3개월 · 1년 수익"], ["L6 거시", "VIX · 달러 · 10년 금리 변화"]], [4.2, 13], size=9)
    d.p("기울이기 세기 κ: 0.05 · 0.1 · 0.2 · 0.3 (학습에서 가장 좋은 κ = 0.05).", 9.5, color="5B6575")
    d.img("f09_meta.png", 15.5, "그림 8. (왼쪽) κ별 시험 Sharpe (오른쪽) 1등 축을 맞힌 비율")
    k = MR[(MR["실험"].astype(str).str.startswith("B")) & (MR["κ"] > 0)].iloc[0]
    d.p([("결과: ", ), f"시험 Sharpe는 κ=0.05~0.2에서 1.37~1.41로 기준(1.32)보다 조금 높게 나왔지만 차이 t가 0.6~1.5로 유의하지 않고, 학습 개선은 +0.02뿐이었으며, κ가 클수록 최대 낙폭이 깊어졌습니다(−23.8% → −37.2%). 무엇보다 "
         f"1등 축 적중률이 학습 {k['1등 축 적중률 학습']:.0%} · 시험 {k['1등 축 적중률 시험']:.0%}로 무작위(33%)보다 낮아, 맞혀서 이긴 것이 아니라 기울기가 우연히 시험 기간에 유리했을 가능성이 큽니다."])
    d.box("판정: 미채택", ["기준을 넘은 유일한 묶음처럼 보이지만 근거가 약합니다. 주간 결과(1.05 < 1.07)와 방향이 달라 '불안정'하다는 점도 같이 기록했습니다."], fill="FDECEA", bar="#C8553D")
    # ---------- 8 E
    bundle_head(d, 8, "E 동적 목표 변동성", "시장이 위험해 보이면 목표 속도(변동성)를 낮추고, 잔잔하면 높입니다.",
                "비 오는 날 속도를 줄이고 맑은 날 올리는 운전. 속도를 줄이면 사고(낙폭)가 줄지만 도착 시간(수익)도 같이 늦어집니다.")
    d.h("사고파는 순서", 2)
    for t in ["6가지 위험 신호(HMM · 변동성 · 동조성 · 흡수비율 · VIX · 뉴스) 중 켜진 개수를 셉니다.", "계단형: 켜진 개수에 따라 목표 변동성을 정합니다.", "연속형: 6신호를 표준화해 평균낸 위험 점수 z로 투자 규모를 정합니다(규모 = 1 − β·z, 50%~130%)."]:
        d.b(t)
    d.h("세부 전략 세팅", 2)
    d.table(["방식", "규칙", "세팅"], [["계단형", "켜진 개수 0개 → 10% 또는 12%, 1개 → 10%, 2개 → 7%, 3개 이상 → 5%", "0개일 때 10% · 12%"], ["연속형", "투자 규모 = 1 − β × 위험 점수, 0.5~1.3배", "β = 0.1 · 0.2 · 0.3 · 0.5"], ["대조군", "신호 없이 늘 목표 7% · 5%", "같은 평균 노출 고정"]], [2.8, 10.2, 4.2], size=9)
    d.img("f10_dyn.png", 14.5, "그림 9. 설정별 시험 Sharpe")
    d.p([("결과: ", ), "모든 설정이 기준(1.32)과 같거나 낮았습니다(1.21~1.31). 같은 평균 노출로 고정한 대조군(7%)과 Sharpe가 같고 낙폭만 비례해 줄어듭니다 — '덜 투자한 몫'입니다. 이자를 넣은 선물 구조(관점 B)에서도 결론이 같습니다."])
    d.box("판정: 미채택", ["주간 결과(0.96~1.02 < 1.07)와 같습니다."], fill="FDECEA", bar="#C8553D")
    d.pb()
    # ---------- 9 F
    bundle_head(d, 9, "F 상관 급등기 추세 집중", "품목들이 한꺼번에 같은 방향으로 움직이기 시작하면, 강한 추세 품목에 몰아 담습니다.",
                "모두가 한쪽으로 달려가기 시작할 때(2022 곡물·원유, 2025 금·은) 선두 주자에 올라타는 전략.")
    d.h("사고파는 순서", 2)
    for t in ["트리거(상관·동조성 신호)가 켜지는지 매일 확인합니다.", "켜지면 평소 50/25/25 대신 정해 둔 '켜졌을 때 비중'으로 갈아탑니다. 꺼지면 복귀.", "학습 기간에서 (트리거 × 켜졌을 때 비중) 조합 중 가장 좋은 것을 고릅니다. 위약: 켜진 구간을 65일 블록으로 섞어 200번."]:
        d.b(t)
    d.h("세부 전략 세팅", 2)
    d.table(["구분", "종류", "설명"], [
        ["트리거 5종", "곡물 상관 분위 > 0.8 / > 0.9 (방향 무관)", "곡물 5개 130일 평균 상관이 역사적으로 높은 위치"],
        ["", "곡물 상관 > 0.9 + 상승 동조", "위 조건에 곡물 평균이 최근 65일 올랐을 때만"],
        ["", "DCC 동조성 분위 > 0.9 · 흡수비율 z > 1.5", "원자재 전체 동조성 · 공통 요인 몫 급증"],
        ["켜졌을 때 비중 5종", "모멘텀 확대 25/50/25", "MinVar 25 · 모멘텀 50 · 가치 25"],
        ["", "모멘텀 100% · 모멘텀 집중 100%", "모멘텀만 / 추세 강도 상위 5개 품목 롱·숏"],
        ["", "상승 추세 집중 3개 · 2개 100%", "26주·52주 모두 오른 품목 중 상위 3(2)개만 롱"],
        ["대조군", "위 5가지를 늘 켜 두기", "트리거 효과와 구분"],
    ], [3.0, 6.8, 7.4], size=8.8)
    d.img("f11_corr.png", 17, "그림 10. (왼쪽) 늘 켜 두면 모두 기준 아래 (오른쪽) 트리거가 켜진 날은 전체의 2~7%뿐")
    d.p([("결과: ", ), "학습 기간 최고는 '곡물 상관 > 0.8 → 모멘텀 집중 100%'(학습 0.86, 시험 1.32 = 기준과 같음, 위약 p 0.14)입니다. 트리거가 시험 기간에 거의 켜지지 않아 시험 결과가 기준과 같게 나왔고, 표의 '시험 차이 t'(±3)는 차이 시계열이 거의 0이라 의미가 없습니다. 주간에서도 2022~23년에는 한 번도 켜지지 않았습니다(2022 곡물 상관이 오히려 낮았음)."])
    d.box("판정: 미채택", ["늘 켜 두는 대조군이 모두 악화(모멘텀 100% 0.42, 상승 추세 집중 3개 0.85)였습니다. 사후에 보면 맞는 해가 있지만 미리 정한 규칙으로는 기준을 못 넘습니다."], fill="FDECEA", bar="#C8553D")
    d.pb()
    # ---------- 10 JK
    bundle_head(d, 10, "J · K 짧은 추세 · 모멘텀 기간", "모멘텀을 얼마나 오래 본 추세로 정하는 것이 좋은지(K), 짧은 추세로 상승하는 품목에만 몰아 담으면 어떤지(J) 시험했습니다.",
                "온도계를 하루치로 볼지 1년치로 볼지 정하는 문제 — 짧게 볼수록 민감하지만 잦은 오판(방향 전환)으로 비용이 커집니다.")
    d.h("세부 전략 세팅", 2)
    d.table(["묶음", "규칙", "세팅"], [["K 모멘텀 기간", "혼합의 모멘텀 축을 다른 관찰 기간으로 바꿔 넣음", "20 · 40 · 65 · 130 · 195 · 260일(기준) · 여러 기간 평균(20·65·260일)"], ["J 짧은 추세", "직전 n일 수익이 +인 품목 중 상위 k개만 롱(5일마다, 같은 위험으로), 혼합 MinVar 25 · J 50 · 가치 25", "n = 20 · 40 · 65 · 130일, k = 2 · 3 · 5"]], [3.2, 8.0, 6.0], size=9)
    d.img("f12_lookback.png", 14.5, "그림 11. 모멘텀 관찰 기간별 혼합 Sharpe — 260일(기준)이 시험 1위")
    d.p([("결과: ", ), "K는 시험 기간에서 20~65일로 짧을수록 나빴고(1.10~1.12), 130일 1.23, 195일 1.29, 260일(기준) 1.32로 길수록 좋았습니다. 학습 기간에서는 40일(0.63)이 기준(0.61)보다 조금 높아 '학습 최고'로 뽑혔지만 시험에서 1.12로 기준보다 0.20 낮았습니다 — 학습 성적만 보고 짧은 기간을 고르면 손해를 보는 사례입니다. J는 학습 최고(65일 · 3개)가 학습 0.46 → 시험 1.00이며 최대 낙폭이 −38.7%로 깊었습니다. 짧은 추세로는 2022년 급등을 일부 일찍 잡아도 잦은 반전으로 손해를 봅니다."])
    d.box("판정: 미채택 (52주 모멘텀 유지)", ["주간 결과(4주 0.98 · 13주 0.77 · 39주 1.09 ≈ 52주 1.07)와 같은 결론입니다."], fill="FDECEA", bar="#C8553D")
    # ---------- 11 GI
    bundle_head(d, 11, "G · I 품목군 비중 상한", "곡물 · 에너지 · 축산 한 그룹에 위험이 지나치게 몰리지 않게 그룹별 비중 상한을 둡니다.",
                "계란을 한 바구니에 너무 많이 담지 않기. 수익을 올리려는 전략이 아니라 '한 바구니가 깨질 때' 손실을 줄이려는 안전장치입니다.")
    d.h("사고파는 순서", 2)
    for t in ["3축 품목 비중을 50/25/25로 합친 품목별 순비중을 만듭니다.", "그룹(곡물 5개 · 에너지 4개 · 축산 2개)의 비중 절댓값 합이 전체의 상한을 넘으면 그 그룹만 비율대로 줄입니다.", "그다음 목표 변동성 10%로 다시 맞춥니다."]:
        d.b(t)
    d.h("세부 전략 세팅", 2)
    d.table(["규칙", "상한"], [["곡물 상한", "50% · 35% · 25%"], ["에너지 상한", "25% · 15%"], ["축산 상한", "25% · 15%"], ["복수 그룹", "곡물 25 + 에너지 25 + 축산 25 / 곡물 25 + 에너지 15 + 축산 15"]], [4.2, 13], size=9)
    d.img("f13_cap.png", 17, "그림 12. 상한별 시험(왼쪽) · 학습(오른쪽) Sharpe")
    d.p([("결과: ", ), "일간 Sharpe로 보면 곡물 25% 상한은 시험 1.33 → 1.30으로 이득이 없었고(주간에서는 1.07 → 1.09), 최대 낙폭도 같았습니다. 축산 상한은 학습 Sharpe를 깎았습니다(0.62 → 0.56~0.59) — 생우 · 돈육이 다른 품목과 덜 엮여 분산 역할을 하기 때문입니다. "
         "다만 상한의 목적인 손실 한도(VaR) · 곡물 쏠림 감소는 이번 일간 실험에서 측정하지 않았습니다."])
    d.box("판정: Sharpe 기준으로는 중립", ["주간 실험에서 '수익 손실 없이 VaR 2.87% → 2.25%'로 채택 후보가 된 결과를 일간 Sharpe가 뒷받침하지도 반박하지도 않습니다."], fill="FFF6E0", bar="#8A6D1E")
    d.pb()
    # ---------- 12 MM
    bundle_head(d, 12, "멀티모달 갈래 (리포트 · 기상 · COT · 문서)", "가격이 아닌 정보(재고 · 작황 · 기상 · 투기 포지션 · 문서 분위기)로 품목 순위를 매겨 롱·숏합니다.",
                "신문(문서)·일기예보(기상)·재고 통계·펀드의 베팅(포지션)을 읽고 '어느 품목이 오를까' 순위표를 만드는 전략.")
    d.h("사고파는 순서", 2)
    for t in ["갈래별 특성(품목마다 값)을 만듭니다 — 발표 이후 날짜에만 사용(미래 정보 차단).", "품목 간 순위로 바꾸고, 과거 성과에서 '높을수록 좋은지 나쁜지' 방향(부호)을 정합니다(롤링 = 최근 104주 기준 매번 갱신 / 고정 = 처음 156주로 한 번).", "상위 1/3 롱 · 하위 1/3 숏(20일마다, 역변동성 비중), 목표 변동성 10%.", "혼합 평가: MinVar 50 + 모멘텀 25 + 이 갈래 25."]:
        d.b(t)
    d.h("갈래별 자료", 2)
    d.table(["갈래", "쓰는 자료 (예)"], [["텍스트", "연준 베이지북 · USDA WASDE 서술 · EIA STEO 서술의 긍정/부정 점수(FinBERT)"], ["리포트 수치", "EIA 주간 재고 · USDA WASDE 기말재고/생산 수정 · NASS 작황 우량 비율 · EIA STEO 전망 수정"], ["포지션 (COT)", "CFTC 투기 · 상업 순포지션, 쏠림, 변화"], ["기상", "엘니뇨(ONI) · 남방진동(SOI) · 지구 기온 · 지역 강수/기온 편차 · 난방/냉방 수요"], ["매크로 · 금융 · 월간 거시", "VIX · 신용 스프레드 · 달러 · 실질금리 · 환율 · 금융여건지수 · 물가 · 산업생산 등(품목별 민감도)"]], [4.2, 13], size=9)
    d.img("f14_mm.png", 15, "그림 13. 갈래별 혼합 시험 Sharpe (부호 롤링)")
    d.p([("결과: ", ), "일간 실행 결과는 주간 결과를 거의 그대로 재현했습니다(단독 Sharpe 텍스트 0.10 vs 0.13, 리포트 수치 −0.17 vs −0.17, 기상 0.20 vs 0.20). 이 신호들은 발표가 주간~월간으로만 나와 일간으로 바꿔도 새 정보가 없기 때문입니다. 혼합은 16개 모두 기준(1.32) 미만이었고, 가장 높은 COT(롤링)가 1.17이었습니다."])
    d.box("판정: 미채택", ["단독으로 성과가 좋아 보이는 갈래(COT 시험 +0.56)도 학습 기간 성과가 −0.01이라 미리 고를 수 없었습니다. 주중 발표를 당일 반영하는 '더 빠른 정보'는 넣지 못했습니다."], fill="FDECEA", bar="#C8553D")
    d.pb()
    # ---------- 13 overlay
    d.h("13. 일간 신호 오버레이 8종 — 대시보드에서 직접 섞어 보기", 1)
    d.box("한 줄 요약", ["대시보드 ⑤ A. Customization › 일간 주기 아래쪽 '일간 신호로 노출 조절' 상자에서, 내가 만든 포트폴리오에 8가지 신호 중 하나를 얹어 '위험하면 현금으로' 규칙을 직접 시험할 수 있습니다."])
    d.p("각 신호는 같은 방식으로 작동합니다: 신호 값이 문턱을 넘으면 다음 날부터 투자 규모를 0%(전부 현금) 또는 50%로 줄이고, 아래로 돌아오면 100%로 복귀합니다. 노출을 바꿀 때마다 |Δ노출| × 1.15 × 5bp 비용이 듭니다.")
    d.img("f15_signals.png", 17, "그림 14. 일간 신호의 흐름과 기본 문턱(빨간 점선) — 빨간 영역이 '위험' 신호가 켜진 때")
    import daily_sleeves as DS
    ov = [("DCC 동조성", "원자재 8개가 함께 움직이는 정도(평균 상관)가 역사적으로 얼마나 높은가", "함께 움직이면 분산 효과가 사라져 한꺼번에 손실을 볼 수 있음", "0.90 (0.60~0.99)", "20일마다 최근 3,900일로 DCC 재추정"),
          ("VIX", "미국 주식 시장의 공포 지수가 역사적으로 얼마나 높은가", "시장 전체의 위험 회피 국면", "0.90 (0.60~0.99)", "FRED 일간 VIX의 그때까지 분위"),
          ("HMM 위기 확률", "시장이 '불안 국면'일 확률", "국면이 바뀌면 평소 규칙이 안 통할 수 있음", "0.90 (0.30~0.99)", "주간 확률을 다음 금요일까지 이어 붙임(일간 HMM 없음)"),
          ("원자재 변동성 국면", "최근 65일 출렁임이 평소(중앙값)의 몇 배인가", "출렁임이 커진 때는 손실 폭도 커짐", "1.5배 (1.0~3.0)", "원자재 19개 평균 변동성"),
          ("흡수비율 z", "여러 품목의 움직임을 하나의 공통 요인이 설명하는 정도가 20일 새 얼마나 늘었나", "품목들이 한 요인으로 몰리는 시작 신호", "1.5 (0.5~3.0)", "252일 상관 행렬 1주성분 몫의 20일 변화 표준화"),
          ("뉴스 감성 악화", "연준 · USDA · EIA 문서의 분위기가 평소보다 나빠진 정도", "문서가 공표되는 날에만 값이 바뀜", "1.0 (0.0~2.5)", "베이지북 · WASDE · STEO 감성의 확장창 표준화"),
          ("칼만 평균 < 0", "내 포트폴리오의 '요즘 평균 수익'(부드럽게 추적)이 0보다 작은가", "내 전략이 요즘 안 먹히는지 판단", "문턱 없음", "일간 수익의 지역 수준 칼만 필터"),
          ("낙폭 손절", "내 누적 수익이 고점에서 −x% 아래로 내려갔나", "손실이 커지면 일단 피하는 전통적 손절", "5% (3~20%)", "회복은 −x/2%에서 복귀")]
    d.table(["신호", "무엇을 재나 (쉬운 말)", "왜 쓰나", "기본 문턱(범위)", "계산 방법"], [list(r) for r in ov], [2.5, 5.2, 4.0, 2.6, 2.9], size=8.5)
    d.img("f16_overlay.png", 17, "그림 15. 오버레이 예 — (왼쪽) VIX 분위 0.9 · 전부 현금 (오른쪽) 낙폭 손절 5% · 절반 현금. 빨간 영역 = 노출을 줄인 기간")
    d.box("꼭 알아둘 점", ["문턱을 시험 기간 결과를 보며 고르면 사후 선택입니다 — 이 문서의 결과(D 현금 피신)에서 모든 신호가 기준을 의미 있게 넘지 못했습니다.", "노출을 줄여서 낙폭이 줄어도 '덜 투자한 몫'이 큽니다. 같은 평균 노출의 고정 비중과 Sharpe가 같은지 보세요.", "HMM은 주간 확률을 이어 붙인 값이고, 뉴스는 문서 공표일에만 바뀝니다."], fill="FFF6E0", bar="#8A6D1E")
    d.pb()
    # ---------- 14 sleeve strategies
    d.h("14. 드롭박스의 재료 전략 한 줄 설명", 1)
    d.p("대시보드 Customization(수치 전략)의 드롭박스에 나오는 전략들입니다. 모두 목표 변동성 10% · 편도 5bp로 만든 주간 수익률이며, 아래 Sharpe는 2007–2026 전체와 학습 · 시험입니다.")
    lx = pd.read_pickle(D / "frozen" / "lx_5_True_2026-10-26.pkl")["table"]
    desc = {"기준 · 52주 모멘텀 (④)": "지난 1년 오른 품목 사고 내린 품목 팔기 (추세 추종)", "기준 · MinVar 롱 (⑥)": "가장 덜 출렁이는 조합으로 품목 매수(상한 25%)",
            "K1 칼만 추세 (느림, q=0.001)": "칼만 필터가 추정한 가격 기울기가 +면 매수, −면 매도(느린 반응)", "K1 칼만 추세 (빠름, q=0.01)": "같은 방식, 빠른 반응",
            "K2 칼만 동적 헤지 페어 (12쌍)": "서로 관련 깊은 두 품목의 가격 관계에서 벌어진 것을 되돌림에 베팅(관계 비율이 매주 변함)",
            "S1 섹터 중립 모멘텀 (섹터 내 승자 롱·패자 숏)": "같은 업종 안에서 오른 품목 사고 내린 품목 팔기", "S2 섹터 내 따라잡기 (과열 숏·낙오 롱)": "같은 업종 안에서 많이 오른 품목은 팔고 뒤처진 품목은 사기",
            "V1 가치 (5년 평균 대비 싼 품목 롱)": "5년 평균보다 싼 품목 사고 비싼 품목 팔기", "W1 왜도 (음의 왜도 롱 · 양의 왜도 숏)": "급락 위험이 큰(음의 왜도) 품목을 사고 급등 복권형 품목을 팔기",
            "L1 저변동성 (낮은 변동성 롱 · 높은 변동성 숏)": "덜 출렁이는 품목 사고 많이 출렁이는 품목 팔기", "B1 저베타 BAB (베타 중립)": "시장(바스켓)에 덜 민감한 품목 사고 민감한 품목 팔기(시장 방향은 상쇄)",
            "C1 헤징 압력 (COT, 8품목)": "상업(생산자) 헤지 매도가 많은 품목 사기", "C2 쏠림 역발상 (COT 투기 과열 숏, 8품목)": "투기 펀드가 과열로 쏠린 품목 팔기"}
    rows = []
    for nm, ds in desc.items():
        if nm in lx.index:
            r = lx.loc[nm]; rows.append([nm.split(" (")[0], ds, f"{r['Sharpe']:+.2f}", f"{r['Sharpe ~2017']:+.2f}", f"{r['Sharpe 2017~']:+.2f}"])
    d.table(["전략", "무엇을 사고 무엇을 파나", "전체", "학습", "시험"], rows, [3.6, 9.8, 1.3, 1.3, 1.2], size=8.5, center_from=2)
    d.p("읽는 법: 두 기간 모두 양수이고 시험이 학습보다 낮지 않은 것은 가치(V1)와 섹터 내 따라잡기(S2)뿐이며, 이 둘은 모멘텀과 음의 상관이라 섞으면 도움이 됩니다(그래서 가치가 3축의 한 축입니다). 나머지는 단독 성과가 약하거나 불안정해 채택하지 않았습니다.", 9.5, color="5B6575")
    d.p("드롭박스의 '일간 실험 결과' 항목([A 갈아타기 · 트리거], [B 메타 레이어], [C 환경 적응형 비중], [D 현금 피신], [E 동적 목표 변동성], [F 상관 급등기], [J 짧은 추세], [K 모멘텀 기간], [G·I 품목군 상한], [멀티모달 일간 실행])는 위 4~12장의 설정별 결과이며, 이미 각자 목표 10%로 맞춘 혼합 전략이라 섞으면 '혼합의 혼합'이 됩니다.", 9.5, color="5B6575")
    # ---------- 15
    d.h("15. 종합 판정과 읽을 때 주의할 점", 1)
    d.table(["묶음", "일간 결과 (학습 최고 설정, 시험 기준 1.32)", "판정"], [
        ["3축 혼합 (주간 vs 일간)", "주간 합산 시험 1.07 → 1.24 (차이 t 1.57)", "유지 (유의 개선 아님)"], ["D 현금 피신", "−0.51 ~ +0.04, 최악은 변동성 · HMM", "미채택"], ["C 환경 적응형", "차이 t −2.0 ~ −0.7", "미채택"],
        ["A 갈아타기 · 트리거", "접점 1.14~1.21, 위기 · 모멘텀 트리거는 낙폭 −45~−60%", "미채택"], ["B 메타 레이어", "1.37~1.41이나 적중률 무작위 이하(학습 30% · 시험 21%)", "미채택"], ["E 동적 목표 변동성", "1.21~1.31", "미채택"],
        ["F 상관 급등기", "트리거가 시험 기간에 거의 안 켜짐, 대조군 모두 악화", "미채택"], ["J · K", "짧을수록 나쁨, 260일 유지", "미채택"], ["G · I 품목군 상한", "곡물 25%: 1.33 → 1.30 (VaR는 미측정)", "Sharpe 중립"], ["멀티모달 갈래", "혼합 0.69~1.17, 모두 기준 미만", "미채택"],
    ], [4.2, 9.2, 3.8], size=9)
    d.h("읽을 때 주의할 점", 2)
    for t in [[("같은 자료 위의 반복 시험: ", ), "모든 실험이 같은 2005–2026 자료라 '결과가 같다'가 독립된 증거가 되지는 않습니다."], [("사후 선택: ", ), "여러 설정을 해 보고 가장 좋은 것을 고르면 과대평가됩니다. 이 문서는 학습 기간으로만 고른 설정을 보여 줍니다."], [("일간 Sharpe(√252)는 주간 합산보다 약 4% 높게 나옵니다. ", ), "주간 결과와 비교할 때는 '주간 합산 시험' 값을 보세요."],
              [("HMM · 뉴스의 한계: ", ), "HMM은 일간으로 만들지 않고 주간 확률을 이어 붙였고, 뉴스는 문서 공표일에만 바뀝니다."], [("롤 비용: ", ), "수익은 롤 보정 연속월물 기준이라 ETF 기준보다 연 약 3%p 과대평가되며 주간 · 일간 모두 같습니다."], [("전략이 모두 미채택이라는 뜻: ", ), "'가격 · 보고서 · 기상 신호로는 지금의 기준 혼합을 의미 있게 개선하지 못했다'이며, 곡선 · 재고 · 마진 같은 새 정보는 아직 시험하지 못했습니다(필요 자료 엑셀 참조)."]]:
        d.b(t)
    d.d.save(str(OUT)); print("saved", OUT)


if __name__ == "__main__":
    main()
