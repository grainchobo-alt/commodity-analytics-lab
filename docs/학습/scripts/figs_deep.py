"""LSTM · GNN 심화 그림 + 손계산 예시(본문 표와 같은 숫자를 쓰도록 함수로 공유)."""
import pickle

import matplotlib.ticker
import numpy as np

import figs as B
from figs import BLUE, G, GOLD, GREY, INK, MUTED, OCHRE, PAL, RED, ROOT, TEAL, arrow, box, fig, plt, rng
from matplotlib.patches import Circle, Rectangle

sig = lambda z: 1 / (1 + np.exp(-z))

# ════════════════════════ LSTM: 설명용으로 가중치를 직접 정한 1차원 LSTM (은닉 1개)
# 게이트 = σ(w_h·h + w_x·x + b). 큰 하락 충격(x ≪ 0) → 망각 게이트↓(옛 기억 지움) · 입력 게이트↑(충격을 새로 기억)
W = dict(f=(0.3, 0.8, 1.5), i=(0.2, -0.9, 0.0), c=(0.4, 0.9, 0.0), o=(0.1, 0.2, 0.5))


def lstm_step(x, h, c):
    g = {k: W[k][0] * h + W[k][1] * x + W[k][2] for k in W}
    f, i, o, ct = sig(g["f"]), sig(g["i"]), sig(g["o"]), np.tanh(g["c"])
    c2 = f * c + i * ct
    return dict(x=x, f=f, i=i, ct=ct, c=c2, o=o, h=o * np.tanh(c2), zf=g["f"], zi=g["i"], zc=g["c"], zo=g["o"])


def lstm_hand(xs=(0.5, -2.0, 0.3), h=0.0, c=0.0):
    """손계산 표: 주간 수익률(표준화) 3개를 차례로 넣음 — 2주차가 큰 하락 충격."""
    rows = []
    for x in xs:
        r = lstm_step(x, h, c); r["h_prev"], r["c_prev"] = h, c; rows.append(r); h, c = r["h"], r["c"]
    return rows


def lstm_run(xs):
    h = c = 0.0; out = []
    for x in xs:
        r = lstm_step(x, h, c); out.append(r); h, c = r["h"], r["c"]
    return out


def demo_series():
    r_ = rng(31); x = 0.6 * r_.standard_normal(90); x[30], x[31], x[62] = -2.6, -1.8, 2.4
    return x


@fig("lstm_gradient")
def _():
    k = np.arange(0, 41); f, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.plot(k, 0.75 ** k, color=RED, lw=2, label="단순 RNN: 매 단계 ×0.75 (곱할수록 사라짐)"); ax.plot(k, 0.97 ** k, color=G, lw=2, label="LSTM 셀 경로: 망각 게이트 ≈ 0.97을 곱함")
    ax.plot(k, np.r_[np.ones(20), np.zeros(21)], color=GREY, lw=1.5, ls="--", label="AR(20): 20기까지만 보고 그 뒤는 0")
    ax.set_xlabel("몇 기 전 정보인가 (k)"); ax.set_ylabel("오늘까지 남는 영향"); ax.legend(frameon=False, fontsize=8); ax.set_title("기울기 소실: 먼 과거의 신호가 학습 중에 사라지는 문제", fontsize=9)
    return f


@fig("lstm_cell")
def _():
    f, ax = plt.subplots(figsize=(7.6, 4.3)); ax.set_xlim(0, 11); ax.set_ylim(0, 6.4); ax.axis("off")
    ax.add_patch(Rectangle((1.2, 0.6), 8.8, 5.2, fc="#F6F6F2", ec=GREY, lw=1))
    ax.annotate("", (10.9, 5.0), (0.1, 5.0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.6)); ax.text(0.1, 5.25, r"$C_{t-1}$", fontsize=12); ax.text(10.3, 5.25, r"$C_t$", fontsize=12)
    ax.text(6.4, 5.45, "셀 스테이트 = 컨베이어 벨트 (덧셈·곱셈만 → 정보가 멀리 감)", fontsize=8.5, color=INK, ha="center")
    ax.plot([0.1, 1.8], [1.2, 1.2], color=INK, lw=1.6); ax.text(0.1, 1.38, r"$h_{t-1}$", fontsize=12); ax.plot([1.8, 1.8], [0.15, 1.2], color=INK, lw=1.6); ax.text(1.9, 0.12, r"$x_t$ (오늘 입력)", fontsize=10)
    ax.plot([1.8, 7.6], [1.2, 1.2], color=INK, lw=1.6)
    gates = [(3.0, "σ", "망각 f", RED), (4.6, "σ", "입력 i", BLUE), (5.9, "tanh", r"후보 $\tilde{C}$", G), (7.6, "σ", "출력 o", GOLD)]
    for x, fn, nm, col in gates:
        ax.plot([x, x], [1.2, 2.1], color=INK, lw=1.3); box(ax, x - 0.45, 2.1, 0.9, 0.6, fn, col, tc="white", fs=11, bold=True); ax.text(x, 1.75, nm, ha="center", fontsize=8.5, color=col, fontweight="bold")
    def op(x, y, s):
        ax.add_patch(Circle((x, y), 0.26, fc="white", ec=INK, lw=1.4, zorder=4)); ax.text(x, y, s, ha="center", va="center", fontsize=12, zorder=5)
    op(3.0, 5.0, "×"); ax.annotate("", (3.0, 4.74), (3.0, 2.7), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.5))
    op(5.25, 3.7, "×"); ax.plot([4.6, 4.6, 5.0], [2.7, 3.7, 3.7], color=BLUE, lw=1.5); ax.plot([5.9, 5.9, 5.5], [2.7, 3.7, 3.7], color=G, lw=1.5)
    op(5.25, 5.0, "+"); ax.annotate("", (5.25, 4.74), (5.25, 3.96), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))
    box(ax, 8.35, 3.95, 0.9, 0.5, "tanh", TEAL, tc="white", fs=10); ax.plot([8.8, 8.8], [5.0, 4.45], color=INK, lw=1.3)
    op(8.8, 3.0, "×"); ax.annotate("", (8.8, 3.26), (8.8, 3.95), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3)); ax.plot([7.6, 7.6, 8.54], [2.7, 3.0, 3.0], color=GOLD, lw=1.5)
    ax.plot([8.8, 8.8, 10.9], [2.74, 1.2, 1.2], color=INK, lw=1.6); ax.annotate("", (10.9, 1.2), (10.0, 1.2), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6)); ax.text(10.25, 1.38, r"$h_t$", fontsize=12)
    ax.annotate("", (9.6, 6.3), (9.6, 1.2), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3)); ax.text(9.7, 6.0, r"$h_t$ (출력·예측에 사용)", fontsize=9)
    ax.text(3.15, 4.1, "① 얼마나\n지울까", fontsize=8, color=RED); ax.text(3.75, 3.2, "② 무엇을\n새로 적을까", fontsize=8, color=BLUE); ax.text(6.95, 3.35, "③ 무엇을\n내보낼까", fontsize=8, color=OCHRE)
    return f


@fig("lstm_act")
def _():
    z = np.linspace(-6, 6, 300); f, (p, q) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    p.plot(z, sig(z), color=RED, lw=2.2); p.axhline(0.5, color=GREY, ls=":"); p.set_title(r"시그모이드 $\sigma(z)=1/(1+e^{-z})$ → 0~1 '밸브 열림 정도'", fontsize=9); p.set_xlabel("z")
    p.text(-5.8, 0.85, "0 = 완전히 잠금\n1 = 완전히 열림", fontsize=8)
    q.plot(z, np.tanh(z), color=G, lw=2.2); q.axhline(0, color=GREY, ls=":"); q.set_title(r"tanh(z) → −1~1 '내용 (방향·크기)'", fontsize=9); q.set_xlabel("z"); return f


@fig("lstm_hand")
def _():
    rows = lstm_hand(); t = np.arange(1, 4); f, (p, q) = plt.subplots(1, 2, figsize=(7.4, 2.8))
    for k, nm, col in (("f", "망각 f", RED), ("i", "입력 i", BLUE), ("o", "출력 o", GOLD)):
        p.plot(t, [r[k] for r in rows], "o-", color=col, lw=2, label=nm)
    p.set_xticks(t, ["1주 (+0.5)", "2주 (−2.0 충격)", "3주 (+0.3)"]); p.set_ylim(0, 1); p.legend(frameon=False, fontsize=8); p.set_title("게이트 값 (0~1)", fontsize=9)
    q.plot(t, [r["c"] for r in rows], "s-", color=INK, lw=2, label="셀 C (기억)"); q.plot(t, [r["h"] for r in rows], "o--", color=TEAL, lw=2, label="출력 h"); q.axhline(0, color=GREY, lw=0.7)
    q.set_xticks(t, ["1주", "2주", "3주"]); q.legend(frameon=False, fontsize=8); q.set_title("기억과 출력", fontsize=9); return f


@fig("lstm_gates_time")
def _():
    x = demo_series(); out = lstm_run(x); t = np.arange(len(x))
    f, axs = plt.subplots(3, 1, figsize=(7.0, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.1, 1]))
    axs[0].bar(t, x, color=np.where(x < 0, RED, G), width=0.8); axs[0].set_ylabel("입력 x\n(주간 수익률)"); axs[0].set_title("설명용 LSTM (가중치를 직접 정함)에 수익률을 흘려보낸 결과", fontsize=9, loc="left")
    axs[1].plot(t, [o["f"] for o in out], color=RED, lw=1.8, label="망각 f"); axs[1].plot(t, [o["i"] for o in out], color=BLUE, lw=1.4, label="입력 i"); axs[1].set_ylim(0, 1); axs[1].legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    axs[1].annotate("30주 하락 충격:\n망각↓ · 입력↑", (30, out[30]["f"]), (38, 0.18), arrowprops=dict(arrowstyle="->", color=MUTED), fontsize=8)
    axs[2].plot(t, [o["c"] for o in out], color=INK, lw=1.8, label="셀 C"); axs[2].plot(t, [o["h"] for o in out], color=TEAL, lw=1.4, ls="--", label="출력 h"); axs[2].axhline(0, color=GREY, lw=0.6)
    axs[2].legend(frameon=False, fontsize=8, ncol=2); axs[2].set_xlabel("주"); return f


@fig("lstm_memory")
def _():
    """∂h_T/∂x_{T−k}: 오늘 출력이 k주 전 입력에 얼마나 민감한가 (수치 미분) — AR의 고정 계수와 비교."""
    x = demo_series()
    def sens(T):
        base = lstm_run(x[:T + 1])[-1]["h"]; out = []
        for k in range(16):
            xx = x[:T + 1].copy(); xx[T - k] += 1e-4; out.append((lstm_run(xx)[-1]["h"] - base) / 1e-4)
        return np.abs(out)
    a, b = sens(28), sens(34)
    f, (p, q) = plt.subplots(1, 2, figsize=(7.6, 2.8), gridspec_kw=dict(width_ratios=[1, 1.5]))
    p.bar(range(1, 6), [0.35, 0.2, 0.12, 0.05, 0.02], color=GREY); p.set_xlim(0.3, 15.7); p.set_title("AR(5): 시차별 계수 고정\n(어느 시점이든 똑같이)", fontsize=9); p.set_xlabel("시차 k")
    k = np.arange(16); q.set_xticks(range(0, 16, 3)); q.plot(k, a / a.max(), "o-", color=BLUE, lw=1.8, ms=4, label="평시 (28주 시점)"); q.plot(k, b / b.max(), "s-", color=RED, lw=1.8, ms=4, label="충격 직후 (34주 시점)")
    q.set_xlabel("몇 주 전 입력인가 k"); q.set_ylabel("민감도 (최대 = 1)"); q.legend(frameon=False, fontsize=8); q.set_title("LSTM: 같은 모델인데 시점마다 '기억 길이'가 달라짐", fontsize=9)
    return f


@fig("lstm_predict")
def _():
    import torch
    torch.manual_seed(0); r_ = rng(0); T = 500; t = np.arange(T); s = np.sin(t / 8) * (1 + 0.5 * (t > 250)) + 0.25 * r_.standard_normal(T); seq = 20
    Xs = np.stack([s[i:i + seq] for i in range(T - seq)]); ys = s[seq:]; Xt = torch.tensor(Xs[:, :, None], dtype=torch.float32); yt = torch.tensor(ys[:, None], dtype=torch.float32); tr = int(0.7 * len(ys))

    class M(torch.nn.Module):
        def __init__(s_):
            super().__init__(); s_.l = torch.nn.LSTM(1, 16, batch_first=True); s_.o = torch.nn.Linear(16, 1)

        def forward(s_, x):
            h, _ = s_.l(x); return s_.o(h[:, -1])
    m = M(); opt = torch.optim.Adam(m.parameters(), 1e-2); hist = []
    for _ in range(150):
        opt.zero_grad(); loss = torch.mean((m(Xt[:tr]) - yt[:tr]) ** 2); loss.backward(); opt.step(); hist.append(loss.item())
    with torch.no_grad():
        pr = m(Xt).numpy().ravel()
    te = np.mean((ys[tr:] - pr[tr:]) ** 2); nv = np.mean((ys[tr:] - s[seq + tr - 1:-1]) ** 2)
    f, (p, q) = plt.subplots(1, 2, figsize=(7.8, 2.8), gridspec_kw=dict(width_ratios=[2.2, 1]))
    p.plot(t, s, color=GREY, lw=0.9, label="시계열"); p.plot(t[seq:], pr, color=BLUE, lw=1.4, label="LSTM 1기 앞 예측"); p.axvspan(seq + tr, T, color=RED, alpha=0.07); p.text(seq + tr + 5, 1.7, "시험 구간", color=RED, fontsize=8)
    p.legend(frameon=False, fontsize=8, loc="lower left"); p.set_title(f"규칙적 패턴: 시험 MSE {te:.3f} vs 직전값 그대로 {nv:.3f}", fontsize=9)
    q.plot(hist, color=INK); q.set_yscale("log"); q.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")); q.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); q.set_xlabel("학습 반복 (epoch)"); q.set_title("학습 손실 (MSE)", fontsize=9)
    B.FACTS["lstm_te"], B.FACTS["lstm_naive"] = te, nv
    return f


@fig("gru_cell")
def _():
    f, ax = plt.subplots(figsize=(7.0, 3.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    ax.annotate("", (9.9, 4.0), (0.1, 4.0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.4)); ax.text(0.1, 4.2, r"$h_{t-1}$", fontsize=12); ax.text(9.3, 4.2, r"$h_t$", fontsize=12)
    ax.plot([0.6, 0.6, 7.2], [4.0, 1.0, 1.0], color=INK, lw=1.3); ax.plot([1.2, 1.2], [0.1, 1.0], color=INK, lw=1.3); ax.text(1.3, 0.12, r"$x_t$", fontsize=11)
    box(ax, 2.0, 1.7, 1.0, 0.6, "σ", BLUE, tc="white", fs=11, bold=True); ax.text(2.5, 1.35, "리셋 r", ha="center", color=BLUE, fontsize=8.5, fontweight="bold")
    box(ax, 4.0, 1.7, 1.0, 0.6, "σ", RED, tc="white", fs=11, bold=True); ax.text(4.5, 1.35, "업데이트 z", ha="center", color=RED, fontsize=8.5, fontweight="bold")
    box(ax, 6.2, 1.7, 1.0, 0.6, "tanh", G, tc="white", fs=10, bold=True); ax.text(6.7, 1.35, r"후보 $\tilde{h}$", ha="center", color=G, fontsize=8.5, fontweight="bold")
    for x in (2.5, 4.5, 6.7):
        ax.plot([x, x], [1.0, 1.7], color=INK, lw=1.1)
    ax.annotate("", (6.2, 2.0), (3.0, 2.0), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.3, ls="--")); ax.text(3.1, 2.62, "r: 옛 기억을 후보 계산에 얼마나 쓸지", fontsize=8, color=BLUE)
    ax.add_patch(Circle((4.5, 4.0), 0.28, fc="white", ec=INK, lw=1.4, zorder=4)); ax.text(4.5, 4.0, "×", ha="center", va="center", fontsize=12, zorder=5)
    ax.add_patch(Circle((6.7, 4.0), 0.28, fc="white", ec=INK, lw=1.4, zorder=4)); ax.text(6.7, 4.0, "+", ha="center", va="center", fontsize=12, zorder=5)
    ax.annotate("", (4.5, 3.72), (4.5, 2.3), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4)); ax.text(4.6, 3.0, "1 - z", color=RED, fontsize=9)
    ax.annotate("", (6.7, 3.72), (6.7, 2.3), arrowprops=dict(arrowstyle="-|>", color=G, lw=1.4)); ax.text(6.8, 3.0, r"z × $\tilde{h}$", color=G, fontsize=9)
    ax.text(5.0, 4.6, "z 하나로 '지우기'와 '적기'를 동시에 (LSTM의 f·i를 하나로)", fontsize=8.5, ha="center")
    return f


# ════════════════════════ GNN
NODES = ["팜유", "대두유", "대두", "옥수수", "밀", "원유", "석탄"]
POS = {"팜유": (-1.6, 0.9), "대두유": (-0.4, 1.5), "대두": (0.6, 0.6), "옥수수": (-0.2, -0.4), "밀": (1.2, -0.9), "원유": (-1.7, -0.6), "석탄": (-1.1, -1.6)}
CORR = np.array([[1, .78, .45, .30, .22, .40, .25], [.78, 1, .62, .35, .25, .35, .20], [.45, .62, 1, .68, .48, .22, .12], [.30, .35, .68, 1, .60, .20, .05],
                 [.22, .25, .48, .60, 1, .15, .08], [.40, .35, .22, .20, .15, 1, .55], [.25, .20, .12, .05, .08, .55, 1]])


def topk_adj(C, k=2):
    C = np.abs(C).copy(); np.fill_diagonal(C, 0); M = np.zeros_like(C)
    for i in range(len(C)):
        nb = np.argsort(C[i])[-k:]; M[i, nb] = 1
    return np.maximum(M, M.T)


def sym_norm(A):
    Ah = A + np.eye(len(A)); d = Ah.sum(1); Dm = np.diag(1 / np.sqrt(d)); return Ah, np.diag(d), Dm, Dm @ Ah @ Dm


def gcn_example():
    """4노드 손계산: 옥수수가 대두·밀·석탄과 연결된 허브(별 모양). 특성 2개 = [최근 4주 수익률(%), 변동성 z]."""
    names = ["옥수수", "대두", "밀", "석탄"]
    A = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], float)
    H = np.array([[2.0, 0.5], [4.0, 0.2], [-1.0, 0.8], [6.0, 2.0]])
    Wm = np.array([[0.5, -0.2], [0.1, 1.0]])
    Ah, D, Dm, An = sym_norm(A)
    AH = An @ H; Z = AH @ Wm; Hn = np.maximum(Z, 0)
    return dict(names=names, A=A, Ah=Ah, D=D, Dm=Dm, An=An, H=H, W=Wm, AH=AH, Z=Z, H1=Hn)


def draw_graph(ax, A, vals=None, cmap="Reds", vmax=1.0, title="", wts=None):
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            if A[i, j]:
                (x1, y1), (x2, y2) = POS[NODES[i]], POS[NODES[j]]; lw = 1.2 if wts is None else 0.5 + 5 * wts[i, j]
                ax.plot([x1, x2], [y1, y2], color=GREY, lw=lw, zorder=1)
                if wts is not None:
                    ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{wts[i, j]:.2f}", fontsize=7, color=MUTED, ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.5))
    cm = plt.get_cmap(cmap)
    for k, n in enumerate(NODES):
        x, y = POS[n]; fc = cm(0.15 + 0.85 * min(1, vals[k] / vmax)) if vals is not None else ([PAL[2]] * 2 + [PAL[0]] * 3 + [PAL[3]] * 2)[k]
        ax.add_patch(Circle((x, y), 0.3, fc=fc, ec=INK, lw=1, zorder=3)); ax.text(x, y - 0.47, n, ha="center", fontsize=8.5, zorder=4)
        if vals is not None:
            ax.text(x, y, f"{vals[k]:.2f}", ha="center", va="center", fontsize=7.5, zorder=4, color="white" if vals[k] / vmax > 0.55 else INK)
    ax.set_xlim(-2.2, 1.8); ax.set_ylim(-2.2, 2.0); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=9)


@fig("gnn_graph")
def _():
    A = topk_adj(CORR, 2); f, (p, q) = plt.subplots(1, 2, figsize=(7.8, 3.5), gridspec_kw=dict(width_ratios=[1, 1.1]))
    draw_graph(p, A, title="노드 = 품목, 엣지 = 상관 상위 2개와 연결", wts=np.where(A > 0, CORR, 0))
    q.imshow(A, cmap="Greens", vmin=0, vmax=1.4)
    for i in range(7):
        for j in range(7):
            q.text(j, i, int(A[i, j]), ha="center", va="center", fontsize=8, color="white" if A[i, j] else MUTED)
    q.set_xticks(range(7), NODES, rotation=45, fontsize=8); q.set_yticks(range(7), NODES, fontsize=8); q.set_title("인접 행렬 A (1 = 연결, 대칭)", fontsize=9)
    q.spines[:].set_visible(False); return f


@fig("gnn_pipeline")
def _():
    f, ax = plt.subplots(figsize=(8.0, 3.8)); ax.set_xlim(0, 16); ax.set_ylim(0, 7.6); ax.axis("off"); r_ = rng(41)
    for k, n in enumerate(NODES):
        y = 6.6 - k * 0.92; s = np.cumsum(r_.normal(0, 1, 30)); s = (s - s.min()) / (np.ptp(s) + 1e-9) * 0.55
        ax.plot(np.linspace(0.9, 2.6, 30), y - 0.27 + s, color=PAL[k % len(PAL)], lw=1.1); ax.text(0.8, y, n, ha="right", va="center", fontsize=8)
    ax.text(1.75, 7.35, r"입력 $x_{n,1:L}$" + "\n(품목별 과거 L주)", ha="center", fontsize=8.5)
    box(ax, 3.3, 2.2, 2.0, 3.4, "GRU\n시간 요약\n\n모든 품목이\n같은 가중치\n공유", BLUE, tc="white", fs=8.5, bold=True); arrow(ax, (2.7, 3.9), (3.3, 3.9))
    for k in range(7):
        for j in range(10):
            ax.add_patch(Rectangle((6.1 + j * 0.16, 6.25 - k * 0.6), 0.16, 0.55, fc=plt.get_cmap("RdBu_r")(r_.uniform(0.2, 0.8)), ec="white", lw=0.3))
    ax.text(6.9, 6.95, r"$H^{(0)}$  ($N \times 64$)", ha="center", fontsize=9); ax.text(6.9, 1.75, "품목마다\n'추세 요약 벡터'", ha="center", fontsize=8); arrow(ax, (5.3, 3.9), (6.0, 3.9))
    box(ax, 8.2, 2.6, 1.9, 2.6, "GCN 1층\n이웃과\n가중 평균\n" + r"+ $W^{(0)}$", G, tc="white", fs=8.5, bold=True); arrow(ax, (7.75, 3.9), (8.2, 3.9))
    box(ax, 10.5, 2.6, 1.9, 2.6, "GCN 2층\n이웃의\n이웃까지\n" + r"+ $W^{(1)}$", G, tc="white", fs=8.5, bold=True); arrow(ax, (10.1, 3.9), (10.5, 3.9))
    box(ax, 12.8, 3.2, 1.4, 1.4, "선형\n출력", GOLD, tc="white", fs=9, bold=True); arrow(ax, (12.4, 3.9), (12.8, 3.9)); arrow(ax, (14.2, 3.9), (14.7, 3.9))
    ax.text(15.3, 3.9, r"$\hat y$" + "\n품목별\n4주 뒤\n수익률", ha="center", va="center", fontsize=8.5)
    ax.text(9.3, 6.1, r"인접 행렬 $A$" + "\n(직전 3년 상관 상위 4개)", ha="center", fontsize=8, color=G); arrow(ax, (9.3, 5.6), (9.3, 5.25), col=G); arrow(ax, (10.3, 5.6), (11.3, 5.25), col=G)
    ax.text(3.3, 0.5, "① 시간 (Time) — 각 품목 혼자", fontsize=9, color=BLUE, fontweight="bold"); ax.text(8.2, 0.5, "② 공간 (Space) — 품목끼리 정보 교환", fontsize=9, color=G, fontweight="bold")
    return f


@fig("gnn_tensor")
def _():
    r_ = rng(42); H = r_.normal(0, 1, (7, 64)); f = plt.figure(figsize=(7.8, 3.6)); ax = f.add_axes([0.12, 0.08, 0.84, 0.62]); a2 = f.add_axes([0.12, 0.8, 0.84, 0.12])
    ax.imshow(H, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto"); ax.set_yticks(range(7), NODES); ax.set_xticks([0, 15, 31, 47, 63], ["1", "16", "32", "48", "64"]); ax.set_xlabel("은닉 차원 (GRU가 만든 64개 숫자)")
    ax.add_patch(Rectangle((-0.5, -0.5), 64, 1, fill=False, ec=GOLD, lw=2.2)); ax.text(64.5, 3, r"$H^{(0)}$" + "\n7 × 64", va="center", fontsize=10)
    a2.imshow(H[:1], cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto"); a2.set_yticks([]); a2.set_xticks([])
    a2.set_title("팜유 벡터 (1 × 64) = [" + ", ".join(f"{v:.2f}" for v in H[0, :4]) + ", …, " + f"{H[0, -1]:.2f}]", fontsize=8.5, loc="left")
    for s in list(ax.spines.values()) + list(a2.spines.values()):
        s.set_visible(False)
    return f


@fig("gnn_norm")
def _():
    E = gcn_example(); nm = E["names"]; f, axs = plt.subplots(1, 4, figsize=(8.4, 2.5))
    for ax, (M, ttl, fmt) in zip(axs, ((E["A"], "① A (연결)", "{:.0f}"), (E["Ah"], r"② $\hat{A}=A+I$ (자기 포함)", "{:.0f}"), (E["D"], r"③ $\hat{D}$ (행 합 = 연결 수)", "{:.0f}"), (E["An"], r"④ $\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$", "{:.2f}"))):
        ax.imshow(M / max(1, M.max()), cmap="Greens", vmin=0, vmax=1.5)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=8)
        ax.set_xticks(range(4), nm, fontsize=7, rotation=45); ax.set_yticks(range(4), nm if ax is axs[0] else [""] * 4, fontsize=7); ax.set_title(ttl, fontsize=8.5)
        ax.spines[:].set_visible(False)
    return f


@fig("gnn_propagate")
def _():
    A = topk_adj(CORR, 2); An = sym_norm(A)[3]; x0 = np.zeros(7); x0[NODES.index("석탄")] = 1.0
    f, axs = plt.subplots(1, 3, figsize=(8.4, 3.0))
    for k, ax in enumerate(axs):
        v = np.linalg.matrix_power(An, k) @ x0; draw_graph(ax, A, vals=v, vmax=1.0 if k == 0 else 0.6, title=["0층: 석탄에만 충격 1", "1층: 바로 연결된 원유·팜유로", "2층: 이웃의 이웃까지 번짐"][k])
    return f


@fig("gnn_oversmooth")
def _():
    A = topk_adj(CORR, 2); An = sym_norm(A)[3]; x = np.array([2.0, 1.2, 0.3, -1.5, -0.6, 2.4, -2.0]); K = 12; tr = np.array([np.linalg.matrix_power(An, k) @ x for k in range(K + 1)])
    f, (p, q) = plt.subplots(1, 2, figsize=(7.6, 2.8), gridspec_kw=dict(width_ratios=[1.6, 1]))
    for i, n in enumerate(NODES):
        p.plot(range(K + 1), tr[:, i], "o-", ms=3, lw=1.4, color=PAL[i % len(PAL)], label=n)
    p.set_xlabel("GCN 층 수 (곱한 횟수)"); p.set_ylabel("품목별 특성 값"); p.legend(frameon=False, fontsize=7, ncol=2); p.set_title("층을 쌓을수록 모든 품목 값이 비슷해짐", fontsize=9)
    q.plot(range(K + 1), tr.std(1), "s-", color=RED, lw=2); q.set_xlabel("층 수"); q.set_title("품목 간 차이 (표준편차)", fontsize=9); return f


@fig("gnn_crisis")
def _():
    """평시엔 옥수수 ↔ 대두·밀(곡물 요인), 300주부터 위기: 옥수수가 에너지(원유·석탄) 요인을 따름.
    정적 그래프 = 처음 156주 상관 상위 2개로 고정(대시보드는 1년마다 갱신), 동적 = 직전 52주 상관으로 매주 갱신.
    지표 = '이웃 가중 평균'이 옥수수 실제 수익률과 얼마나 같이 움직이나(26주 롤링 상관)."""
    r_ = rng(43); T, tc = 520, 300; gr, oil, en = (r_.standard_normal(T) for _ in range(3)); e = lambda: r_.standard_normal(T)
    crisis = np.arange(T) >= tc
    R = np.column_stack([oil + 0.3 * en + 0.5 * e(), oil + 0.4 * gr + 0.5 * e(), 0.8 * gr + 0.3 * oil + 0.5 * e() + crisis * 0.8 * e(),
                         np.where(crisis, 0.9 * en + 0.1 * gr, 0.85 * gr) + 0.5 * e(), 0.75 * gr + 0.5 * e(), en + 0.2 * oil + 0.5 * e(), en + 0.5 * e()])
    ci = NODES.index("옥수수")

    def row(Cw):
        C = np.abs(np.corrcoef(Cw.T)); np.fill_diagonal(C, 0); w = np.zeros(7); nb = np.argsort(C[ci])[-2:]; w[nb] = C[ci, nb]; return w / w.sum()
    w_static = row(R[:156]); agg_s = R @ w_static; agg_d = np.full(T, np.nan); wd = {}
    for t in range(156, T):
        w = row(R[t - 52:t]); agg_d[t] = R[t] @ w; wd[t] = w
    rc = lambda a: np.array([np.nan if t < 182 else np.corrcoef(a[t - 26:t], R[t - 26:t, ci])[0, 1] for t in range(T)])
    cs, cd = rc(agg_s), rc(agg_d)
    f, (p, q) = plt.subplots(1, 2, figsize=(8.2, 3.0), gridspec_kw=dict(width_ratios=[1.7, 1]))
    p.plot(cs, color=RED, lw=1.8, label="정적 그래프 (평시 상관으로 고정)"); p.plot(cd, color=G, lw=1.8, label="동적 그래프 (직전 52주로 매주 갱신)"); p.axvspan(tc, T, color=GOLD, alpha=0.12)
    p.text(tc + 4, 0.93, "위기 구간 (옥수수가 에너지를 따라감)", fontsize=8); p.set_ylim(-0.3, 1.0); p.axhline(0, color=GREY, lw=0.6); p.set_xlim(180, T); p.set_xlabel("주")
    p.set_ylabel("이웃 평균 ↔ 옥수수 상관"); p.legend(frameon=False, fontsize=7.5, loc="lower left"); p.set_title("이웃 정보가 옥수수를 설명하는 정도", fontsize=9)
    x = np.arange(7); q.bar(x - 0.2, w_static, 0.4, color=RED, label="정적"); q.bar(x + 0.2, wd[T - 1], 0.4, color=G, label="동적 (마지막 주)"); q.set_xticks(x, NODES, rotation=45, fontsize=8)
    q.legend(frameon=False, fontsize=8); q.set_title("옥수수가 섞는 이웃 비중", fontsize=9)
    B.FACTS["crisis"] = dict(static_pre=float(np.nanmean(cs[200:tc])), static_post=float(np.nanmean(cs[tc + 30:])), dyn_post=float(np.nanmean(cd[tc + 30:])),
                             nb_static=[NODES[i] for i in np.flatnonzero(w_static)], nb_dyn=[NODES[i] for i in np.flatnonzero(wd[T - 1])])
    return f


@fig("gnn_real")
def _():
    import pandas as pd
    d = pickle.load(open(ROOT / "data" / "alt_ts_preds.pkl", "rb")); lite = next(v for k, v in d.items() if "GNN" in k and "정식" not in k); ex = next(v for k, v in d.items() if "GNN" in k and "정식" in k)
    m = lite.notna() & ex.notna(); x, y = lite.values[m.values], ex.values[m.values]; rho = np.corrcoef(x, y)[0, 1]
    pt = pd.read_csv(ROOT / "data" / "alt_ts_pairs_compare.csv"); row = pt[pt.iloc[:, 0].str.contains("GNN")].iloc[0]
    f, (p, q) = plt.subplots(1, 2, figsize=(7.8, 3.0), gridspec_kw=dict(width_ratios=[1, 1.2]))
    idx = rng(44).choice(len(x), min(4000, len(x)), replace=False); p.scatter(x[idx] * 100, y[idx] * 100, s=2, color=BLUE, alpha=0.3)
    p.set_xlabel("경량(자체 구현) 예측 (%)"); p.set_ylabel("정식(torch_geometric) 예측 (%)"); p.set_title(f"같은 자료·같은 구조인데 예측 상관 {rho:.2f}", fontsize=9)
    cols = [c for c in pt.columns]; get = lambda tag, key: float(row[[c for c in cols if c.startswith(tag) and key in c][0]])
    lab = ["방향 Sharpe", "IC ×10"]; lv = [get("경량", "방향 Sharpe"), 10 * get("경량", "IC")]; ev = [get("정식", "방향 Sharpe"), 10 * get("정식", "IC")]
    xx = np.arange(2); q.bar(xx - 0.2, lv, 0.4, color=GREY, label="경량"); q.bar(xx + 0.2, ev, 0.4, color=G, label="정식"); q.axhline(0, color=INK, lw=0.7)
    q.axhline(0.45, color=RED, ls="--", lw=1); q.text(1.45, 0.47, "기준: 52주 모멘텀 0.45", color=RED, fontsize=7.5, ha="right"); q.set_xticks(xx, lab); q.legend(frameon=False, fontsize=8)
    q.set_title("성과 (2012–2026 주간, 19품목)", fontsize=9)
    B.FACTS["gnn_real"] = dict(rho=float(rho), sign=float(row[[c for c in cols if "부호" in c][0]]), lite_sh=lv[0], ex_sh=ev[0], lite_ic=lv[1] / 10, ex_ic=ev[1] / 10)
    return f
