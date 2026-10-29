"""v2(10분 임원 보고) 추가 차트: 횡단면 vs 시계열, 랜덤워크·브라운 운동. charts.py 다음에 실행 (같은 img/·nums.json 사용)."""
import sys, warnings, json
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(r"C:\Users\user\Desktop\Commodity")
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
OUT = HERE / "img"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9a9a94",
                     "axes.labelcolor": "#3a3a36", "xtick.color": "#55554f", "ytick.color": "#55554f",
                     "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlecolor": "#1E3A2F", "savefig.dpi": 200})
G, GOLD, RED, BLUE, GREY = "#1E3A2F", "#C8A24A", "#D9582B", "#2F6E8F", "#8A8A84"
nums_p = HERE / "nums.json"
nums = json.loads(nums_p.read_text(encoding="utf-8")) if nums_p.exists() else {}


def save(f, name):
    f.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(f)
    print("saved", name)


import analytics as A, ls_strategies as LS
adj = A.load_levels(roll_adjusted=True).loc[:"2026-10-26"]
KN = {"yf:ZC=F": "옥수수", "yf:ZW=F": "밀", "yf:ZS=F": "대두", "yf:ZM=F": "대두박", "yf:ZL=F": "대두유", "yf:CL=F": "원유", "yf:NG=F": "천연가스",
      "yf:GC=F": "금", "yf:HG=F": "구리", "yf:KC=F": "커피", "yf:SB=F": "원당", "yf:CT=F": "면화"}
P = LS.weekly(adj, list(KN), "2001-01-01")[list(KN)]

# ── 40 횡단면 vs 시계열: 품목 × 월 수익률 표 (행 = 시계열, 열 = 횡단면)
M = np.log(P).resample("ME").last().diff().loc["2025-05":"2026-10"] * 100
M = M.rename(columns=KN)
f, ax = plt.subplots(figsize=(9.6, 4.6))
v = np.nanpercentile(np.abs(M.values), 95)
ax.imshow(M.T.values, cmap="RdYlGn", vmin=-v, vmax=v, aspect="auto")
ax.set_yticks(range(M.shape[1])); ax.set_yticklabels(M.columns, fontsize=10)
ax.set_xticks(range(0, len(M), 2)); ax.set_xticklabels([d.strftime("%y.%m") for d in M.index[::2]], fontsize=9)
for s_ in ax.spines.values():
    s_.set_visible(False)
r0, c0 = 0, len(M) - 1
ax.add_patch(Rectangle((-0.5, r0 - 0.5), len(M), 1, fill=False, ec=BLUE, lw=3))
ax.add_patch(Rectangle((c0 - 0.5, -0.5), 1, M.shape[1], fill=False, ec=GOLD, lw=3))
ax.text(len(M) * 0.5, r0 - 0.85, "시계열 = 한 품목을 시간 순서대로 (옥수수의 18개월)", color=BLUE, fontsize=11, fontweight="bold", ha="center")
ax.text(c0 + 0.7, M.shape[1] / 2, "횡단면 =\n같은 시점에\n품목끼리 비교\n(2026년 10월)", color="#8a6a12", fontsize=11, fontweight="bold", va="center")
ax.set_title("품목 × 월 수익률 (초록 = 상승, 빨강 = 하락)", loc="left", pad=30)
save(f, "c40_xs_ts")

# ── 41 랜덤워크: 실제 옥수수 vs 같은 출렁임의 무작위 경로 (어느 것이 진짜?)
c = np.log(P["yf:ZC=F"].dropna()).loc["2016-10":]
r = c.diff().dropna()
sd = float(r.std())
rng = np.random.default_rng(7)
real = (c - c.iloc[0]).values
paths, ends = [], [real[-1]]
while len(paths) < 5:                      # 화면에 들어오는 경로만(45~210), 끝값이 서로·실제와 겹치지 않게
    p_ = np.r_[0, np.cumsum(rng.normal(0, sd, len(r)))]
    if np.exp(p_).min() > 0.45 and np.exp(p_).max() < 2.1 and min(abs(p_[-1] - e) for e in ends) > 0.12:
        paths.append(p_); ends.append(p_[-1])
real_pos = 3
paths.insert(real_pos, real)
cols6 = [G, BLUE, GOLD, RED, GREY, "#6E5A9B"]
f, ax = plt.subplots(figsize=(6.6, 4.2))
for k, p_ in enumerate(paths):
    ax.plot(c.index, 100 * np.exp(p_), lw=1.5, color=cols6[k])
    ax.text(c.index[-1] + pd.Timedelta(days=40), 100 * np.exp(p_[-1]), "ABCDEF"[k], fontsize=12, fontweight="bold", color=cols6[k], va="center")
ax.axhline(100, color="#bbbbb5", lw=0.8)
ax.set_ylabel("2016년 10월 = 100"); ax.set_title("6개 중 하나만 실제 옥수수 가격 — 나머지는 동전 던지기", loc="left")
save(f, "c41_randomwalk")
nums["rw_answer"] = "ABCDEF"[real_pos]
nums["corn_week_sd"] = round(sd * 100, 2)
ac = {KN[i]: float(np.log(P[i]).diff().autocorr()) for i in KN}
nums["ac1_mean_abs"] = round(float(np.mean(np.abs(list(ac.values())))), 3)

# ── 42 브라운 운동: 오늘에서 출발한 경로 묶음 — 시간이 갈수록 범위가 √시간으로 넓어짐
H = 52
Z = rng.normal(0, sd, (400, H)).cumsum(1)
wk = np.arange(1, H + 1)
f, ax = plt.subplots(figsize=(6.6, 4.2))
for k in range(120):
    ax.plot(wk, 100 * np.exp(Z[k]), color=GREY, lw=0.5, alpha=0.35)
lo, hi = 100 * np.exp(-1.645 * sd * np.sqrt(wk)), 100 * np.exp(1.645 * sd * np.sqrt(wk))
ax.fill_between(wk, lo, hi, color=GOLD, alpha=0.22, label="90% 범위")
ax.plot(wk, lo, color=GOLD, lw=2); ax.plot(wk, hi, color=GOLD, lw=2)
for w_, xt, yt in ((4, 6, 168), (52, 31, 188)):
    ax.annotate(f"{w_}주 뒤 90% 범위\n{lo[w_ - 1] - 100:+.0f}% ~ {hi[w_ - 1] - 100:+.0f}%", (w_, hi[w_ - 1]), xytext=(xt, yt),
                fontsize=10.5, fontweight="bold", color="#7a5d10", bbox=dict(fc="white", ec="none", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="#7a5d10", lw=1))
ax.axhline(100, color=G, lw=1.2)
ax.set_xlabel("오늘부터 몇 주 뒤"); ax.set_ylabel("오늘 가격 = 100"); ax.set_xlim(0, 54); ax.set_ylim(45, 205)
ax.set_title("몬테카를로 시뮬레이션 400회 (옥수수와 같은 출렁임)", loc="left")
save(f, "c42_brownian")
nums["band4"] = [round(lo[3] - 100, 1), round(hi[3] - 100, 1)]; nums["band52"] = [round(lo[51] - 100, 1), round(hi[51] - 100, 1)]

nums_p.write_text(json.dumps(nums, ensure_ascii=False, indent=1), encoding="utf-8")
print("nums:", {k: nums[k] for k in ("rw_answer", "corn_week_sd", "ac1_mean_abs", "band4", "band52")})
