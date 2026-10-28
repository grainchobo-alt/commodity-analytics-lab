"""⑥ 모델 도감 — 모델을 고르면 '무엇을 하는가 · 어떻게 작동하나(합성 데이터 시각화) · 수식 · 언제 쓰나 · 함정 · 이 앱의 어디에'를 보여줌.
시각화는 원리가 잘 보이도록 만든 합성 데이터에 실제 모델을 돌린 결과 (슬라이더로 잡음·파라미터 조절)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

PAL = ["#20808d", "#a84b2f", "#1f8a5b", "#c98500", "#d55181", "#008300", "#7a6fd0", "#e66767"]
GREY = "#898781"


def _tabs23(c):
    """2D · 3D 보기 탭 — 위의 슬라이더를 함께 써서 매개변수를 바꾸면 두 그림이 같이 바뀜."""
    return c.tabs(["2D 보기", "3D 보기 (마우스로 회전 · 휠로 확대)"])


def _scene(f, h=520, x="", y="", z="", eye=(1.5, -1.5, 0.9), **kw):
    f.update_layout(height=h, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", y=-0.02, font=dict(size=10)),
                    scene=dict(xaxis_title=x, yaxis_title=y, zaxis_title=z, camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2])), **kw))
    return f


def _lay(f, h=380, **kw):
    base = dict(height=h, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.18), hovermode="closest")
    base.update(kw)
    f.update_layout(**base)
    return f


# ============================================================================
# 시각화 함수들 — 각자 위젯(key 접두어 at_)과 그림을 그림
# ============================================================================
def viz_pca(c):
    a, b = c.columns(2)
    rho = a.slider("두 변수 상관", -0.95, 0.95, 0.8, 0.05, key="at_pca_rho")
    n = b.slider("관측치 수", 50, 1000, 300, 50, key="at_pca_n")
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([0, 0], [[1.6, rho * np.sqrt(1.6 * 0.7)], [rho * np.sqrt(1.6 * 0.7), 0.7]], n)
    X = X - X.mean(0)
    ev, V = np.linalg.eigh(np.cov(X.T)); o = ev.argsort()[::-1]; ev, V = ev[o], V[:, o]
    f = go.Figure(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", marker=dict(size=5, color=GREY, opacity=0.5), name="데이터 (예: 옥수수·대두 수익률)"))
    for k in range(2):
        L = 2 * np.sqrt(ev[k])
        f.add_trace(go.Scatter(x=[-V[0, k] * L, V[0, k] * L], y=[-V[1, k] * L, V[1, k] * L], mode="lines",
                               line=dict(color=PAL[k], width=5), name=f"PC{k + 1}: 분산의 {ev[k] / ev.sum():.0%}"))
    _lay(f, 430, title="PCA: 데이터가 가장 넓게 퍼진 방향 = PC1, 그에 직교 = PC2 (화살표 길이 ∝ √분산)",
         xaxis=dict(title="변수 1", scaleanchor="y"), yaxis_title="변수 2")
    c.plotly_chart(f, key="at_pca_fig")
    c.caption(f"상관이 클수록 PC1 하나가 분산 대부분을 설명 → '공통 요인'(예: 원자재 전반의 위험선호). 지금 PC1 = {ev[0] / ev.sum():.0%}. "
              "상관 0이면 두 축이 비슷 → 압축할 공통 요인 없음.")


def viz_ols(c):
    a, b = c.columns(2)
    noise = a.slider("잡음 크기", 0.1, 3.0, 1.0, 0.1, key="at_ols_noise")
    out = b.checkbox("이상치 3개 추가", False, key="at_ols_out")
    rng = np.random.default_rng(1)
    x = rng.uniform(-3, 3, 80); y = 0.8 * x + noise * rng.standard_normal(80)
    if out:
        x = np.r_[x, [2.5, 2.8, 3.0]]; y = np.r_[y, [-6, -7, -6.5]]
    b1, b0 = np.polyfit(x, y, 1)
    xx = np.linspace(-3.2, 3.2, 50)
    f = go.Figure(go.Scatter(x=x, y=y, mode="markers", marker=dict(color=GREY, size=6), name="관측치"))
    f.add_trace(go.Scatter(x=xx, y=b0 + b1 * xx, line=dict(color=PAL[0], width=3), name=f"OLS 추정 β = {b1:.2f}"))
    f.add_trace(go.Scatter(x=xx, y=0.8 * xx, line=dict(color=PAL[2], width=2, dash="dot"), name="참 β = 0.80"))
    for xi, yi in zip(x[:25], y[:25]):
        f.add_trace(go.Scatter(x=[xi, xi], y=[yi, b0 + b1 * xi], mode="lines", line=dict(color=PAL[1], width=1), showlegend=False))
    r2 = 1 - np.sum((y - b0 - b1 * x) ** 2) / np.sum((y - y.mean()) ** 2)
    _lay(f, 420, title=f"OLS: 빨간 세로선(잔차)의 제곱합을 최소로 하는 직선 — R² = {r2:.2f}", xaxis_title="X (예: S&P 수익률)", yaxis_title="Y (예: WTI 수익률)")
    c.plotly_chart(f, key="at_ols_fig")
    c.caption("잡음을 키우면 R²는 떨어지지만 β 추정은 여전히 참값 근처(편향 없음, 분산만 커짐). 이상치를 넣으면 제곱 손실 때문에 직선이 크게 휨 → 분위수 회귀·Huber가 대안.")


def viz_ridge_lasso(c):
    a, b = c.columns(2)
    lam = a.slider("규제 강도 λ (제약 영역 크기의 역수)", 0.0, 3.0, 1.0, 0.05, key="at_rl_lam")
    kind = b.radio("벌점", ["Lasso (L1)", "Ridge (L2)", "Elastic Net"], horizontal=True, key="at_rl_kind")
    # 2계수 문제: 손실 등고선 (OLS 해 = (2, 0.6), 상관된 설계)
    bo = np.array([2.0, 0.6]); H = np.array([[1.0, 0.5], [0.5, 1.0]])
    g = np.linspace(-1, 3, 121); B1, B2 = np.meshgrid(g, g)
    Dd = np.stack([B1 - bo[0], B2 - bo[1]], -1)
    Q = np.einsum("...i,ij,...j->...", Dd, H, Dd)
    t = max(0.05, 2.5 - lam * 0.75)                     # 제약 영역 반경
    th = np.linspace(0, 2 * np.pi, 400)
    if kind.startswith("Lasso"):
        bx, by = t * np.cos(th) / (np.abs(np.cos(th)) + np.abs(np.sin(th))), t * np.sin(th) / (np.abs(np.cos(th)) + np.abs(np.sin(th)))
        pen = lambda b_: np.abs(b_).sum()
    elif kind.startswith("Ridge"):
        bx, by = t * np.cos(th), t * np.sin(th); pen = lambda b_: np.sqrt((b_ ** 2).sum())
    else:
        r = t / (0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th))) + 0.5); bx, by = r * np.cos(th), r * np.sin(th)
        pen = lambda b_: 0.5 * np.abs(b_).sum() + 0.5 * np.sqrt((b_ ** 2).sum())
    # 제약 경계 위에서 손실 최소점
    cand = np.stack([bx, by], 1)
    if pen(bo) <= t + 1e-9:
        sol = bo
    else:
        qq = np.einsum("ni,ij,nj->n", cand - bo, H, cand - bo); sol = cand[qq.argmin()]
    f = go.Figure(go.Contour(x=g, y=g, z=Q, contours=dict(start=0.05, end=6, size=0.4, coloring="lines"), showscale=False,
                             line=dict(width=1), colorscale=[[0, PAL[0]], [1, PAL[0]]], name="손실 등고선"))
    f.add_trace(go.Scatter(x=bx, y=by, fill="toself", fillcolor="rgba(168,75,47,0.15)", line=dict(color=PAL[1], width=2), name="허용 영역 (벌점 ≤ t)"))
    f.add_trace(go.Scatter(x=[bo[0]], y=[bo[1]], mode="markers", marker=dict(size=12, color=GREY, symbol="x"), name="OLS 해 (규제 없음)"))
    f.add_trace(go.Scatter(x=[sol[0]], y=[sol[1]], mode="markers", marker=dict(size=13, color=PAL[2]), name=f"규제 해 β = ({sol[0]:.2f}, {sol[1]:.2f})"))
    _lay(f, 470, title="등고선(손실)이 허용 영역에 처음 닿는 점이 해 — Lasso는 꼭짓점(축 위)에 닿아 계수가 정확히 0",
         xaxis=dict(title="β₁", range=[-1, 3], scaleanchor="y"), yaxis=dict(title="β₂", range=[-1, 3]))
    t2, t3 = _tabs23(c)
    t2.plotly_chart(f, key="at_rl_fig")
    Qc = np.minimum(Q, 6.0)
    qf = lambda b_: float((np.asarray(b_) - bo) @ H @ (np.asarray(b_) - bo))  # noqa: E731
    f3 = go.Figure(go.Surface(x=g, y=g, z=Qc, colorscale="Teal", opacity=0.75, showscale=False, name="손실",
                              contours=dict(z=dict(show=True, usecolormap=True, project_z=True, start=0, end=6, size=0.4))))
    qb = np.einsum("ni,ij,nj->n", cand - bo, H, cand - bo)
    f3.add_trace(go.Scatter3d(x=bx, y=by, z=np.zeros_like(bx), mode="lines", line=dict(color=PAL[1], width=6), name="허용 영역 경계 (바닥)"))
    f3.add_trace(go.Scatter3d(x=bx, y=by, z=np.minimum(qb, 6.0), mode="lines", line=dict(color=PAL[1], width=4, dash="dash"), name="경계를 손실 그릇에 올린 선"))
    f3.add_trace(go.Scatter3d(x=[bo[0]], y=[bo[1]], z=[0], mode="markers", marker=dict(size=6, color=GREY, symbol="x"), name="OLS 해 (그릇 바닥)"))
    f3.add_trace(go.Scatter3d(x=[sol[0]], y=[sol[1]], z=[qf(sol)], mode="markers", marker=dict(size=7, color=PAL[2]), name=f"규제 해 (손실 {qf(sol):.2f})"))
    t3.plotly_chart(_scene(f3, 540, "β₁", "β₂", "손실", eye=(1.4, -1.6, 1.0)), key="at_rl_3d")
    t3.caption("그릇 = 손실(낮을수록 좋음), 바닥 선 = 허용 영역. 해 = 경계를 그릇에 올린 점선에서 가장 낮은 곳. λ를 키우면 경계가 줄어 해가 그릇 벽을 타고 올라감(손실 증가 = 편향).")
    c.caption("λ를 키우면 영역이 작아져 두 계수 모두 0쪽으로. Lasso는 β₂가 먼저 정확히 0 → 변수 선택. Ridge는 둘 다 줄지만 0이 되지 않음. "
              "Elastic Net은 그 중간 (꼭짓점이 둥글어 상관된 변수들을 함께 남김).")


def viz_nonconvex(c):
    gamma = c.slider("SCAD a / MCP γ (클수록 Lasso에 가까움)", 1.5, 6.0, 3.0, 0.1, key="at_nc_g")
    lam = 1.0
    z = np.linspace(-4, 4, 401)
    soft = np.sign(z) * np.maximum(np.abs(z) - lam, 0)
    a_ = max(gamma, 2.01)
    scad = np.where(np.abs(z) <= 2 * lam, soft,
                    np.where(np.abs(z) <= a_ * lam, ((a_ - 1) * z - np.sign(z) * a_ * lam) / (a_ - 2), z))
    g_ = max(gamma, 1.01)
    mcp = np.where(np.abs(z) <= g_ * lam, np.sign(z) * np.maximum(np.abs(z) - lam, 0) / (1 - 1 / g_), z)
    hard = np.where(np.abs(z) > lam, z, 0)
    f = make_subplots(rows=1, cols=2, subplot_titles=("벌점 함수 p(β)", "추정치 = 임계 함수 (OLS 추정 z → 축소된 β̂)"))
    pl = lam * np.abs(z)
    psc = np.where(np.abs(z) <= lam, lam * np.abs(z), np.where(np.abs(z) <= a_ * lam, (2 * a_ * lam * np.abs(z) - z ** 2 - lam ** 2) / (2 * (a_ - 1)),
                                                               (a_ + 1) * lam ** 2 / 2))
    pmc = np.where(np.abs(z) <= g_ * lam, lam * np.abs(z) - z ** 2 / (2 * g_), g_ * lam ** 2 / 2)
    for j, (nm, yv) in enumerate([("Lasso", pl), ("SCAD", psc), ("MCP", pmc)]):
        f.add_trace(go.Scatter(x=z, y=yv, name=nm, line=dict(color=PAL[j], width=2.5)), 1, 1)
    for j, (nm, yv) in enumerate([("Lasso (soft)", soft), ("SCAD", scad), ("MCP", mcp)]):
        f.add_trace(go.Scatter(x=z, y=yv, name=nm, line=dict(color=PAL[j], width=2.5), showlegend=False), 1, 2)
    f.add_trace(go.Scatter(x=z, y=z, name="축소 없음 (OLS)", line=dict(color=GREY, dash="dot")), 1, 2)
    t2, t3 = _tabs23(c)
    t2.plotly_chart(_lay(f, 400), key="at_nc_fig")
    pen3 = t3.radio("곡면", ["MCP", "SCAD"], horizontal=True, key="at_nc_3k")
    zz, gg = np.linspace(-4, 4, 161), np.linspace(1.5, 6.0, 46)
    Zg, Gg = np.meshgrid(zz, gg)
    if pen3 == "MCP":
        Gm = np.maximum(Gg, 1.01)
        P3 = np.where(np.abs(Zg) <= Gm * lam, lam * np.abs(Zg) - Zg ** 2 / (2 * Gm), Gm * lam ** 2 / 2)
        cur = pmc
    else:
        Am = np.maximum(Gg, 2.01)
        P3 = np.where(np.abs(Zg) <= lam, lam * np.abs(Zg), np.where(np.abs(Zg) <= Am * lam, (2 * Am * lam * np.abs(Zg) - Zg ** 2 - lam ** 2) / (2 * (Am - 1)),
                                                                    (Am + 1) * lam ** 2 / 2))
        cur = psc
    f3 = go.Figure(go.Surface(x=zz, y=gg, z=P3, colorscale="Teal", opacity=0.85, showscale=False))
    f3.add_trace(go.Scatter3d(x=z, y=np.full_like(z, gamma), z=cur, mode="lines", line=dict(color=PAL[1], width=7), name=f"지금 γ = {gamma:.1f}"))
    f3.add_trace(go.Scatter3d(x=z, y=np.full_like(z, 6.0), z=lam * np.abs(z), mode="lines", line=dict(color=GREY, width=4, dash="dash"), name="Lasso |β| (비교, 뒷벽)"))
    t3.plotly_chart(_scene(f3, 520, "β", "γ (a)", "벌점 p(β)", eye=(1.6, -1.4, 0.8)), key="at_nc_3d")
    t3.caption("γ가 작을수록 벌점이 일찍 평평해짐(큰 계수는 더 이상 깎지 않음), 클수록 Lasso의 V자에 가까워짐. 주황 선 = 슬라이더의 γ.")
    c.caption("Lasso는 큰 계수도 λ만큼 계속 깎음(편향). SCAD·MCP는 벌점이 일정 크기 이후 평평 → 작은 계수는 0으로, 큰 계수는 깎지 않음 "
              "(오라클 성질). Adaptive Lasso는 1차 추정이 큰 변수의 벌점 가중치를 작게 해 비슷한 효과.")


def viz_factor(c):
    a, b = c.columns(2)
    p = a.slider("설명변수 수", 5, 60, 30, 5, key="at_fac_p")
    k = b.slider("사용 요인 수 k", 1, 10, 2, key="at_fac_k")
    rng = np.random.default_rng(3)
    T = 300
    F = rng.standard_normal((T, 3))
    Lmb = rng.standard_normal((3, p))
    X = F @ Lmb + 0.8 * rng.standard_normal((T, p))
    y = 0.2 * F[:, 0] + 0.9 * F[:, 2] + 0.5 * rng.standard_normal(T)          # y는 3번째(작은) 요인에 주로 의존
    X = (X - X.mean(0)) / X.std(0); y = y - y.mean()
    tr = slice(0, 200); te = slice(200, T)
    U, s, Vt = np.linalg.svd(X[tr], full_matrices=False)
    from sklearn.cross_decomposition import PLSRegression
    out = {}
    for kk in range(1, 11):
        Z = X @ Vt[:kk].T
        bz = np.linalg.lstsq(Z[tr], y[tr], rcond=None)[0]
        pls = PLSRegression(kk, scale=False).fit(X[tr], y[tr])
        e1 = y[te] - Z[te] @ bz; e2 = y[te] - pls.predict(X[te]).ravel()
        out[kk] = (1 - (e1 ** 2).sum() / (y[te] ** 2).sum(), 1 - (e2 ** 2).sum() / (y[te] ** 2).sum())
    R = pd.DataFrame(out, index=["PCR", "PLS"]).T
    f = go.Figure([go.Scatter(x=R.index, y=R["PCR"], name="PCR (X 분산 큰 순)", line=dict(color=PAL[0], width=2.5), mode="lines+markers"),
                   go.Scatter(x=R.index, y=R["PLS"], name="PLS (y와 공분산 큰 순)", line=dict(color=PAL[1], width=2.5), mode="lines+markers")])
    f.add_vline(x=k, line_dash="dot", line_color=GREY)
    _lay(f, 380, title="요인 수별 테스트 R² — y가 'X에서 작은 요인'에 의존할 때 PCR은 늦게, PLS는 빨리 잡음",
         xaxis_title="요인 수 k", yaxis_title="테스트 R²", hovermode="x unified")
    c.plotly_chart(f, key="at_fac_fig")
    c.caption(f"k={k}: PCR R² {R.loc[k, 'PCR']:.2f}, PLS R² {R.loc[k, 'PLS']:.2f}. PCR은 y를 보지 않고 X만 요약(비지도), "
              "PLS·3PRF는 y와 관련된 방향을 우선(지도). 변수가 많을 때 버리지 않고 소수 요인으로 압축하는 '변수 누락 없는' 방법.")


def viz_kalman(c):
    a, b = c.columns(2)
    q = 10 ** a.slider("상태 잡음 log10(q) (β가 얼마나 빨리 변한다고 볼지)", -6.0, -1.0, -3.0, 0.5, key="at_kf_q")
    shock = b.checkbox("중간에 β 급변 (위기)", True, key="at_kf_shock")
    rng = np.random.default_rng(4)
    T = 400
    beta = 0.5 + np.cumsum(0.01 * rng.standard_normal(T))
    if shock:
        beta[200:] += 1.0
    x = rng.standard_normal(T); y = beta * x + 0.5 * rng.standard_normal(T)
    b_, P_, r_ = 0.0, 1.0, 0.25
    est, lo, hi = [], [], []
    for t in range(T):
        P_ = P_ + q
        K = P_ * x[t] / (x[t] ** 2 * P_ + r_)
        b_ = b_ + K * (y[t] - x[t] * b_); P_ = (1 - K * x[t]) * P_
        est.append(b_); lo.append(b_ - 1.96 * np.sqrt(P_)); hi.append(b_ + 1.96 * np.sqrt(P_))
    roll = pd.Series(x * y).rolling(60).sum() / pd.Series(x * x).rolling(60).sum()
    t_ = np.arange(T)
    f = go.Figure([go.Scatter(x=t_, y=hi, line=dict(width=0), showlegend=False),
                   go.Scatter(x=t_, y=lo, fill="tonexty", fillcolor="rgba(32,128,141,0.18)", line=dict(width=0), name="95% 구간"),
                   go.Scatter(x=t_, y=beta, name="참 β(t)", line=dict(color="#222", width=2)),
                   go.Scatter(x=t_, y=est, name="칼만 필터 추정 (그 시점까지 정보만)", line=dict(color=PAL[0], width=2.5)),
                   go.Scatter(x=t_, y=roll, name="60기 롤링 OLS (비교)", line=dict(color=PAL[1], width=1.5, dash="dot"))])
    _lay(f, 420, title="Kalman TVP: 관측될 때마다 '예측 → 오차만큼 보정'으로 시변 계수 추적", xaxis_title="시점", yaxis_title="β", hovermode="x unified")
    c.plotly_chart(f, key="at_kf_fig")
    c.caption("q가 작으면 β를 천천히 바꿔 매끄럽지만 급변에 늦게 반응, 크면 빨리 따라가지만 잡음에 흔들림. 롤링 OLS는 창 길이만큼 지연되고 "
              "창 밖 정보는 갑자기 버림 → 칼만은 그 절충을 확률적으로 최적화. 이 앱: ④ 'Kalman TVP' — 위기 시 헤지 베타 변화 추적.")


def viz_hmm(c):
    a, b = c.columns(2)
    sep = a.slider("두 레짐의 변동성 비 (위기/평시)", 1.0, 5.0, 3.0, 0.25, key="at_hmm_sep")
    stay = b.slider("레짐 유지확률", 0.80, 0.995, 0.97, 0.005, key="at_hmm_p")
    rng = np.random.default_rng(5)
    T = 600; s = np.zeros(T, int)
    for t in range(1, T):
        s[t] = s[t - 1] if rng.random() < stay else 1 - s[t - 1]
    r = np.where(s == 0, 1.0, sep) * rng.standard_normal(T) + np.where(s == 0, 0.05, -0.1)
    from hmmlearn.hmm import GaussianHMM
    m = GaussianHMM(2, "full", n_iter=200, random_state=0).fit(r[:, None])
    post = m.predict_proba(r[:, None]); hi = int(np.argmax(m.covars_.ravel()))
    acc = max(np.mean((post[:, hi] > 0.5) == (s == 1)), np.mean((post[:, hi] > 0.5) == (s == 0)))
    f = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.06,
                      subplot_titles=("수익률 (배경 = 참 위기 레짐)", "HMM이 추정한 위기 레짐 확률"))
    f.add_trace(go.Scatter(x=np.arange(T), y=r, line=dict(color=GREY, width=1), name="수익률"), 1, 1)
    f.add_trace(go.Scatter(x=np.arange(T), y=post[:, hi], fill="tozeroy", line=dict(color=PAL[1], width=1), name="P(위기)"), 2, 1)
    ed = np.flatnonzero(np.diff(np.r_[0, s, 0]))
    for a0, a1 in zip(ed[::2], ed[1::2]):
        f.add_vrect(x0=a0, x1=a1, fillcolor="rgba(168,75,47,0.12)", line_width=0, row=1, col=1)
    c.plotly_chart(_lay(f, 460, hovermode="x unified"), key="at_hmm_fig")
    c.caption(f"레짐 판별 정확도 {acc:.0%}. 관측되지 않는 상태(평시/위기)가 마르코프 연쇄로 전이하고, 각 상태마다 수익률 분포(평균·분산)가 다르다고 가정 → "
              "EM으로 추정, 필터 확률로 '지금 위기일 확률'. 변동성 비가 작거나 레짐이 자주 바뀌면 판별이 어려움. "
              "Markov-switching 회귀는 같은 원리로 '레짐마다 회귀계수(베타)'까지 다르게 둠.")


def viz_tar(c):
    a, b = c.columns(2)
    gam = a.slider("STAR 전환 속도 γ (∞ = TAR)", 0.5, 30.0, 3.0, 0.5, key="at_tar_g")
    th = b.slider("참 임계값 (예: VIX)", -1.0, 1.0, 0.3, 0.1, key="at_tar_th")
    rng = np.random.default_rng(6)
    n = 500; q = rng.standard_normal(n) * 1.2; x = rng.standard_normal(n)
    G = 1 / (1 + np.exp(-gam * (q - th)))
    y = (0.3 + 1.2 * G) * x + 0.4 * rng.standard_normal(n)
    import regime as RG
    Xc = x[:, None]
    tf = RG.tar_fit(Xc, y, q)
    lo = q <= tf["gamma"]
    f = make_subplots(rows=1, cols=2, subplot_titles=("Y vs X (색 = 임계 변수 q 위/아래)", "기울기가 q에 따라 변하는 모양"))
    f.add_trace(go.Scatter(x=x[lo], y=y[lo], mode="markers", marker=dict(color=PAL[0], size=5, opacity=0.6), name="q ≤ γ (평시)"), 1, 1)
    f.add_trace(go.Scatter(x=x[~lo], y=y[~lo], mode="markers", marker=dict(color=PAL[1], size=5, opacity=0.6), name="q > γ (위기)"), 1, 1)
    qq = np.linspace(q.min(), q.max(), 200)
    f.add_trace(go.Scatter(x=qq, y=0.3 + 1.2 / (1 + np.exp(-gam * (qq - th))), name="참 기울기 β(q)", line=dict(color="#222", width=2)), 1, 2)
    b_lo, b_hi = np.polyfit(x[lo], y[lo], 1)[0], np.polyfit(x[~lo], y[~lo], 1)[0]
    f.add_trace(go.Scatter(x=[q.min(), tf["gamma"], tf["gamma"], q.max()], y=[b_lo, b_lo, b_hi, b_hi], name=f"TAR 추정 (γ̂ = {tf['gamma']:.2f})",
                           line=dict(color=PAL[1], width=2.5, shape="hv")), 1, 2)
    t2, t3 = _tabs23(c)
    t2.plotly_chart(_lay(f, 420), key="at_tar_fig")
    qg, xg = np.linspace(np.quantile(q, 0.01), np.quantile(q, 0.99), 60), np.linspace(-2.5, 2.5, 30)
    Q3, X3 = np.meshgrid(qg, xg)
    Ytrue = (0.3 + 1.2 / (1 + np.exp(-gam * (Q3 - th)))) * X3
    Ytar = np.where(Q3 <= tf["gamma"], b_lo, b_hi) * X3
    f3 = go.Figure(go.Surface(x=qg, y=xg, z=Ytrue, colorscale="Teal", opacity=0.8, showscale=False, name="참 STAR 곡면"))
    f3.add_trace(go.Surface(x=qg, y=xg, z=Ytar, colorscale=[[0, PAL[1]], [1, PAL[1]]], opacity=0.35, showscale=False, name="TAR 추정 (γ̂에서 절벽)"))
    f3.add_trace(go.Scatter3d(x=q, y=x, z=y, mode="markers", marker=dict(size=2, color=np.where(lo, PAL[0], PAL[1]), opacity=0.5), name="관측치"))
    t3.plotly_chart(_scene(f3, 560, "임계 변수 q", "X", "Y", eye=(1.6, -1.5, 0.8)), key="at_tar_3d")
    t3.caption(f"청록 곡면 = 참 관계(STAR, γ = {gam:.1f}), 주황 반투명 = TAR 추정(γ̂ = {tf['gamma']:.2f}에서 기울기가 뚝 바뀌는 두 평면). "
               "γ를 키우면 청록 곡면의 휘는 구간이 좁아져 TAR 절벽과 겹침.")
    c.caption("TAR: 관측 가능한 변수 q(예: VIX)가 임계값 γ를 넘으면 기울기가 '뚝' 바뀜 — γ는 SSR 최소 격자탐색, 유의성은 Hansen 부트스트랩. "
              "STAR: 로지스틱 함수로 '서서히' 바뀜. HMM과 달리 레짐이 관측 변수로 정해져 해석이 쉬움 (예: 'VIX 24 이상이면 WTI의 주식 베타 3배').")


def viz_garch(c):
    a, b, d = c.columns(3)
    alpha = a.slider("α (충격 반응)", 0.01, 0.3, 0.08, 0.01, key="at_g_a")
    beta = b.slider("β (지속성)", 0.5, 0.98, 0.9, 0.01, key="at_g_b")
    gam = d.slider("γ (비대칭, GJR)", -0.15, 0.3, 0.0, 0.01, key="at_g_gam")
    if alpha + beta + max(gam, 0) / 2 >= 0.999:
        c.warning("α + β + γ/2 ≥ 1 → 분산이 발산(비정상). β를 낮추세요."); return
    rng = np.random.default_rng(7)
    T = 1000; w = 0.05 * (1 - alpha - beta - max(gam, 0) / 2)
    h = np.empty(T); e = np.empty(T); h[0] = w / (1 - alpha - beta - max(gam, 0) / 2)
    for t in range(T):
        if t:
            h[t] = w + (alpha + gam * (e[t - 1] < 0)) * e[t - 1] ** 2 + beta * h[t - 1]
        e[t] = np.sqrt(h[t]) * rng.standard_normal()
    f = make_subplots(rows=2, cols=2, specs=[[{"colspan": 2}, None], [{}, {}]], vertical_spacing=0.12,
                      subplot_titles=("시뮬레이션 수익률: 변동성 군집 (큰 움직임 뒤에 큰 움직임)", "조건부 변동성 √hₜ", "뉴스 충격 곡선"))
    f.add_trace(go.Scatter(y=e, line=dict(color=GREY, width=1), name="수익률"), 1, 1)
    f.add_trace(go.Scatter(y=np.sqrt(h), line=dict(color=PAL[0], width=1.5), name="√hₜ"), 2, 1)
    z = np.linspace(-3, 3, 100) * np.sqrt(h.mean())
    f.add_trace(go.Scatter(x=z, y=np.sqrt(w + (alpha + gam * (z < 0)) * z ** 2 + beta * h.mean()), line=dict(color=PAL[1], width=2.5),
                           name="다음기 변동성"), 2, 2)
    t2, t3 = _tabs23(c)
    t2.plotly_chart(_lay(f, 520, showlegend=False), key="at_g_fig")
    sd0 = np.sqrt(h.mean())
    eg = np.linspace(-3, 3, 61) * sd0
    hg = np.linspace(np.quantile(h, 0.02), np.quantile(h, 0.98), 40)
    E3, H3 = np.meshgrid(eg, hg)
    S3 = np.sqrt(w + (alpha + gam * (E3 < 0)) * E3 ** 2 + beta * H3)
    f3 = go.Figure(go.Surface(x=eg, y=np.sqrt(hg), z=S3, colorscale="Teal", opacity=0.9, colorbar=dict(title="다음기 √h", thickness=12, len=0.6)))
    f3.add_trace(go.Scatter3d(x=e[:-1], y=np.sqrt(h[:-1]), z=np.sqrt(h[1:]), mode="markers", marker=dict(size=1.6, color=PAL[1], opacity=0.35), name="시뮬레이션 실제 점"))
    t3.plotly_chart(_scene(f3, 540, "직전 충격 εₜ₋₁", "직전 변동성 √hₜ₋₁", "다음기 변동성 √hₜ", eye=(1.5, -1.6, 0.8)), key="at_g_3d")
    t3.caption("곡면 = 뉴스 충격 곡선을 직전 변동성 수준마다 쌓은 것. α↑ → 좌우(충격 방향) 기울기가 가팔라짐, β↑ → 뒤쪽(직전 변동성 큼)이 높아짐(지속성), "
               "γ>0 → 왼쪽(하락 충격) 날개가 더 높아 비대칭.")
    hl = np.log(0.5) / np.log(alpha + beta + max(gam, 0) / 2)
    c.caption(f"hₜ = ω + (α + γ·1[εₜ₋₁<0])·ε²ₜ₋₁ + β·hₜ₋₁. 지속성 α+β = {alpha + beta:.2f} → 충격 반감기 {hl:.0f}기. "
              "γ>0(GJR)이면 하락 충격이 변동성을 더 키움(레버리지) → 뉴스 충격 곡선이 비대칭. 원자재는 상승 충격이 더 큰 '역레버리지'(γ<0)도 흔함. "
              "EGARCH는 로그 분산에 같은 비대칭을 둬 양수 제약이 필요 없음.")


def viz_dcc(c):
    a, b = c.columns(2)
    da = a.slider("DCC a (상관의 충격 반응)", 0.0, 0.2, 0.05, 0.01, key="at_dcc_a")
    db = b.slider("DCC b (상관의 지속성)", 0.5, 0.99, 0.93, 0.01, key="at_dcc_b")
    if da + db >= 0.999:
        c.warning("a + b < 1 이어야 함"); return
    rng = np.random.default_rng(8)
    T = 800
    rho_true = np.where((np.arange(T) > 400) & (np.arange(T) < 520), 0.85, 0.25)
    z = np.empty((T, 2))
    for t in range(T):
        L = np.linalg.cholesky([[1, rho_true[t]], [rho_true[t], 1]]); z[t] = L @ rng.standard_normal(2)
    Qbar = np.corrcoef(z.T); Q = Qbar.copy(); rho = []
    for t in range(T):
        d = np.sqrt(np.diag(Q)); R = Q / np.outer(d, d); rho.append(R[0, 1])
        Q = (1 - da - db) * Qbar + da * np.outer(z[t], z[t]) + db * Q
    roll = pd.Series(z[:, 0]).rolling(60).corr(pd.Series(z[:, 1]))
    f = go.Figure([go.Scatter(y=rho_true, name="참 상관 (중간 = 위기 동조화)", line=dict(color="#222", width=2)),
                   go.Scatter(y=rho, name="DCC 조건부 상관", line=dict(color=PAL[0], width=2.5)),
                   go.Scatter(y=roll, name="60기 롤링 상관", line=dict(color=PAL[1], width=1.5, dash="dot"))])
    _lay(f, 400, title="DCC: 표준화 잔차의 곱으로 상관행렬을 매 시점 갱신 — Qₜ = (1−a−b)Q̄ + a·zz' + b·Qₜ₋₁",
         yaxis=dict(title="상관", range=[-0.2, 1]), xaxis_title="시점", hovermode="x unified")
    t2, t3 = _tabs23(c)
    t2.plotly_chart(f, key="at_dcc_fig")
    zg = np.linspace(-3, 3, 49)
    Z1, Z2 = np.meshgrid(zg, zg)
    Q11 = (1 - da - db) * Qbar[0, 0] + da * Z1 ** 2 + db * Qbar[0, 0]
    Q22 = (1 - da - db) * Qbar[1, 1] + da * Z2 ** 2 + db * Qbar[1, 1]
    Q12 = (1 - da - db) * Qbar[0, 1] + da * Z1 * Z2 + db * Qbar[0, 1]
    R3 = Q12 / np.sqrt(Q11 * Q22)
    f3 = go.Figure(go.Surface(x=zg, y=zg, z=R3, colorscale="RdBu_r", cmid=float(Qbar[0, 1]), colorbar=dict(title="다음기 상관", thickness=12, len=0.6)))
    f3.add_trace(go.Surface(x=zg, y=zg, z=np.full_like(R3, Qbar[0, 1]), colorscale=[[0, GREY], [1, GREY]], opacity=0.25, showscale=False, name="평소 상관 Q̄"))
    t3.plotly_chart(_scene(f3, 540, "품목 1 충격 z₁", "품목 2 충격 z₂", "다음기 상관 ρₜ₊₁", eye=(1.5, -1.5, 0.9)), key="at_dcc_3d")
    t3.caption(f"지금 상관이 평소 수준(Q̄ = {Qbar[0, 1]:.2f})일 때, 이번 기 두 품목의 표준화 충격 조합에 따라 다음 기 상관이 어떻게 바뀌나. "
               "같은 방향 큰 충격(오른쪽 위·왼쪽 아래 모서리) → 상관 상승, 반대 방향 → 하락. a↑ → 곡면이 더 휘고, b↑ → 회색 평면(평소)에 붙어 있음. "
               "ADCC는 왼쪽 아래(동반 하락) 모서리만 더 들어 올림.")
    c.caption("1단계: 품목별 GARCH로 변동성 제거 → 2단계: 남은 표준화 잔차로 상관만 동학 추정. 위기 때 상관 급등(분산투자 효과 소멸)을 롤링보다 빠르게 포착. "
              "ADCC는 동반 하락 충격에 상관이 더 오르는 비대칭 추가. BEKK는 공분산 전체를 한 식으로 — 충격·변동성의 '전이 방향'까지 추정.")


def viz_var_dy(c):
    a, b = c.columns(2)
    k = a.slider("A → B·C 전이 강도", 0.0, 0.8, 0.5, 0.05, key="at_dy_k")
    H = b.slider("예측 기간 H", 1, 20, 10, key="at_dy_h")
    from statsmodels.tsa.api import VAR
    rng = np.random.default_rng(9)
    T = 1500; Y = np.zeros((T, 3))
    A1 = np.diag([0.1, 0.1, 0.1]); A1[1, 0] = k; A1[2, 0] = k * 0.6
    for t in range(1, T):
        Y[t] = A1 @ Y[t - 1] + rng.standard_normal(3)
    res = VAR(Y).fit(1)
    import analytics as Aan
    try:
        tbl = Aan.dy_spillover(pd.DataFrame(Y, columns=["A (원유)", "B (옥수수)", "C (대두)"]), H=H)
        M = tbl if isinstance(tbl, pd.DataFrame) else tbl[0]
    except Exception:
        fe = res.fevd(H).decomp[:, -1, :]
        M = pd.DataFrame(fe / fe.sum(1, keepdims=True) * 100, index=["A (원유)", "B (옥수수)", "C (대두)"], columns=["A (원유)", "B (옥수수)", "C (대두)"])
    M = M.iloc[:3, :3]
    f = go.Figure(go.Heatmap(z=M.values, x=list(M.columns), y=list(M.index), colorscale=[[0, "#f2f1ec"], [1, PAL[0]]],
                             texttemplate="%{z:.1f}", hovertemplate="%{y}의 예측오차 중 %{x} 충격: %{z:.1f}%<extra></extra>"))
    f.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(autorange="reversed"),
                    title="예측오차 분산분해 (행 = 받는 쪽, 열 = 충격 출처, %)")
    c.plotly_chart(f, key="at_dy_fig")
    off = M.values.copy(); np.fill_diagonal(off, 0)
    c.caption(f"총 스필오버(대각선 밖 합 / 전체) ≈ {off.sum() / M.values.sum() * 100:.1f}%. VAR로 각 시장이 서로의 과거에 반응하는 구조를 추정하고, "
              "'B의 H기 앞 예측오차 중 몇 %가 A의 충격 때문인가'로 연결성 측정 (일반화 FEVD → 변수 순서에 무관). "
              "A열 합 − A행 밖 합 = 순 전달자. TVP-VAR은 이 VAR 계수를 칼만 필터로 매 시점 갱신해 롤링창 없이 시변 스필오버.")


def viz_trees(c):
    a, b = c.columns(2)
    m = a.radio("모델", ["의사결정나무 1개", "랜덤 포레스트", "그래디언트 부스팅 (LightGBM 원리)"], key="at_tr_m")
    n = b.slider("나무 수 / 깊이", 1, 300, 50, key="at_tr_n")
    rng = np.random.default_rng(10)
    x = np.sort(rng.uniform(-3, 3, 300)); y = np.sin(1.5 * x) + 0.3 * x + 0.35 * rng.standard_normal(300)
    xx = np.linspace(-3, 3, 400)[:, None]
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    if m.startswith("의사"):
        mdl = DecisionTreeRegressor(max_depth=max(1, min(n, 12)), random_state=0).fit(x[:, None], y); lab = f"깊이 {min(n, 12)}"
    elif m.startswith("랜덤"):
        mdl = RandomForestRegressor(n_estimators=n, max_depth=6, random_state=0).fit(x[:, None], y); lab = f"나무 {n}개 평균"
    else:
        mdl = GradientBoostingRegressor(n_estimators=n, max_depth=2, learning_rate=0.1, random_state=0).fit(x[:, None], y); lab = f"나무 {n}개 순차 합"
    f = go.Figure([go.Scatter(x=x, y=y, mode="markers", marker=dict(color=GREY, size=5, opacity=0.5), name="데이터"),
                   go.Scatter(x=xx.ravel(), y=np.sin(1.5 * xx.ravel()) + 0.3 * xx.ravel(), name="참 함수", line=dict(color="#222", dash="dot")),
                   go.Scatter(x=xx.ravel(), y=mdl.predict(xx), name=f"{m} ({lab})", line=dict(color=PAL[1], width=2.5, shape="hv"))])
    _lay(f, 420, title="나무 모델 = 구간별 상수(계단) 함수", xaxis_title="X", yaxis_title="Y")
    c.plotly_chart(f, key="at_tr_fig")
    c.caption("나무 1개: X를 구간으로 잘라 구간 평균 → 깊으면 잡음까지 외움(과적합). 랜덤 포레스트: 표본·변수를 무작위로 바꾼 깊은 나무 수백 개의 평균(분산↓). "
              "부스팅: 얕은 나무를 앞 모델의 '잔차'에 차례로 맞춰 더함 — 나무가 많을수록 세밀하지만 과적합 → 조기종료. "
              "LightGBM은 이를 히스토그램·잎 단위 성장으로 빠르게 한 구현, SHAP은 각 예측을 변수별 기여로 정확히 분해.")


def viz_gp(c):
    a, b = c.columns(2)
    ls = a.slider("길이척도 ℓ (함수가 얼마나 매끄러운가)", 0.1, 3.0, 0.8, 0.1, key="at_gp_ls")
    nz = b.slider("잡음", 0.05, 1.0, 0.2, 0.05, key="at_gp_nz")
    rng = np.random.default_rng(11)
    x = np.r_[rng.uniform(-3, -0.5, 15), rng.uniform(1.5, 3, 10)]; y = np.sin(2 * x) + nz * rng.standard_normal(len(x))
    xx = np.linspace(-4, 4, 300)
    k = lambda A_, B_: np.exp(-0.5 * (A_[:, None] - B_[None, :]) ** 2 / ls ** 2)
    K = k(x, x) + nz ** 2 * np.eye(len(x)); Ks = k(xx, x)
    al = np.linalg.solve(K, y); mu = Ks @ al
    v = np.linalg.solve(K, Ks.T); sd = np.sqrt(np.maximum(1 - np.sum(Ks * v.T, 1), 0) + nz ** 2)
    f = go.Figure([go.Scatter(x=xx, y=mu + 1.96 * sd, line=dict(width=0), showlegend=False),
                   go.Scatter(x=xx, y=mu - 1.96 * sd, fill="tonexty", fillcolor="rgba(32,128,141,0.18)", line=dict(width=0), name="95% 예측구간"),
                   go.Scatter(x=xx, y=np.sin(2 * xx), name="참 함수", line=dict(color="#222", dash="dot")),
                   go.Scatter(x=xx, y=mu, name="GP 예측평균", line=dict(color=PAL[0], width=2.5)),
                   go.Scatter(x=x, y=y, mode="markers", marker=dict(color=PAL[1], size=8), name="관측치")])
    _lay(f, 420, title="Gaussian Process: 데이터가 없는 곳(가운데·양끝)에서 불확실성이 넓어짐", xaxis_title="X", yaxis_title="Y")
    t2, t3 = _tabs23(c)
    t2.plotly_chart(f, key="at_gp_fig")
    lsg = np.linspace(0.1, 3.0, 30)
    xg = np.linspace(-4, 4, 120)
    M3, SD3 = [], []
    for l_ in lsg:
        kk = lambda A_, B_: np.exp(-0.5 * (A_[:, None] - B_[None, :]) ** 2 / l_ ** 2)  # noqa: E731
        K_ = kk(x, x) + nz ** 2 * np.eye(len(x)); Ks_ = kk(xg, x)
        M3.append(Ks_ @ np.linalg.solve(K_, y))
        SD3.append(np.sqrt(np.maximum(1 - np.sum(Ks_ * np.linalg.solve(K_, Ks_.T).T, 1), 0) + nz ** 2))
    M3, SD3 = np.array(M3), np.array(SD3)
    f3 = go.Figure(go.Surface(x=xg, y=lsg, z=M3, surfacecolor=SD3, colorscale="Teal", opacity=0.9, colorbar=dict(title="예측 표준편차", thickness=12, len=0.6)))
    f3.add_trace(go.Scatter3d(x=xx, y=np.full_like(xx, ls), z=mu, mode="lines", line=dict(color=PAL[1], width=7), name=f"지금 ℓ = {ls:.1f}"))
    f3.add_trace(go.Scatter3d(x=x, y=np.full_like(x, ls), z=y, mode="markers", marker=dict(size=4, color=PAL[1]), name="관측치 (지금 ℓ 위치)"))
    t3.plotly_chart(_scene(f3, 540, "X", "길이척도 ℓ", "GP 예측평균", eye=(1.4, -1.7, 0.9)), key="at_gp_3d")
    t3.caption("앞쪽(ℓ 작음) = 관측치마다 구불구불(과적합), 뒤쪽(ℓ 큼) = 거의 직선. 색 = 불확실성(밝을수록 큼) — 관측치가 없는 가운데·양끝에서 커짐. 주황 선 = 슬라이더의 ℓ.")
    c.caption("'비슷한 X는 비슷한 Y'를 커널(유사도)로 표현하고, 관측치로 조건부 분포를 계산 → 평균과 불확실성을 함께 얻음. "
              "ℓ이 작으면 구불구불(과적합), 크면 직선에 가까움 — 이 앱에서는 주변우도로 자동 선택. Kernel Ridge는 같은 예측평균을 내지만 구간은 없음.")


def viz_conformal(c):
    a, b = c.columns(2)
    cov = a.select_slider("목표 커버리지", [0.5, 0.8, 0.9, 0.95], 0.9, key="at_cf_cov")
    shift = b.slider("테스트 후반 변동성 배수 (국면 변화)", 1.0, 4.0, 2.5, 0.25, key="at_cf_sh")
    rng = np.random.default_rng(12)
    n_cal, T = 300, 400
    cal = rng.standard_normal(n_cal)
    e = rng.standard_normal(T) * np.where(np.arange(T) > 200, shift, 1.0)
    alpha = 1 - cov
    q = np.quantile(np.abs(cal), min(1, np.ceil((n_cal + 1) * cov) / n_cal))
    scores = list(np.abs(cal)); at = alpha; qa = []
    for t in range(T):
        qt = np.quantile(scores[-300:], min(0.999, max(0.001, 1 - at))); qa.append(qt)
        err = float(abs(e[t]) > qt); at = at + 0.02 * (alpha - err); scores.append(abs(e[t]))
    qa = np.array(qa)
    f = go.Figure([go.Scatter(y=qa, line=dict(width=0), showlegend=False),
                   go.Scatter(y=-qa, fill="tonexty", fillcolor="rgba(32,128,141,0.18)", line=dict(width=0), name="적응형 ACI 구간"),
                   go.Scatter(y=[q] * T, line=dict(color=PAL[1], dash="dot"), name="분할 conformal (고정 폭)"),
                   go.Scatter(y=[-q] * T, line=dict(color=PAL[1], dash="dot"), showlegend=False),
                   go.Scatter(y=e, mode="markers", marker=dict(color=GREY, size=4), name="예측 오차")])
    _lay(f, 400, title="Conformal: 보정용 잔차의 분위수로 구간 → 실제 커버리지 보장", xaxis_title="테스트 시점", yaxis_title="오차")
    c.plotly_chart(f, key="at_cf_fig")
    late = np.arange(T) > 200
    c.caption(f"후반(국면 변화) 커버리지 — 고정 폭 {np.mean(np.abs(e[late]) <= q):.0%}, ACI {np.mean(np.abs(e[late]) <= qa[late]):.0%} (목표 {cov:.0%}). "
              "분포 가정 없이 '과거 오차가 이 정도였으니 90%는 이 안'. 적응형은 구간을 벗어나는 빈도를 보고 폭을 온라인 조정 → 위기에도 커버리지 유지.")


def viz_midas(c):
    a, b = c.columns(2)
    t1 = a.slider("θ₁", 1.0, 10.0, 1.0, 0.5, key="at_md_t1")
    t2 = b.slider("θ₂ (클수록 최근에 집중)", 1.0, 30.0, 5.0, 0.5, key="at_md_t2")
    from scipy.stats import beta as B
    K = 60; u = (np.arange(K) + 0.5) / K
    w = B.pdf(u, t1, t2); w /= w.sum()
    f = go.Figure(go.Bar(x=np.arange(1, K + 1), y=w, marker_color=PAL[0]))
    _lay(f, 360, title="MIDAS Beta 가중치: 일간 시차 60개를 모수 2개로 요약", xaxis_title="일간 시차 (1 = 어제)", yaxis_title="가중치")
    c.plotly_chart(f, key="at_md_fig")
    c.caption("월간 목표(예: 다음 달 변동성)를 일간 데이터로 예측할 때, 60개 시차에 계수를 하나씩 주면 과적합 → Beta 함수 모양으로 묶어 θ₁, θ₂만 추정. "
              "월평균으로 뭉개는 것보다 최근 정보에 큰 비중. GARCH-MIDAS는 변동성을 '느린 장기 성분(거시·월간) × 빠른 단기 GARCH'로 분해.")


def viz_newton(c):
    a, b = c.columns(2)
    lam = a.slider("위험회피 λ", 0.5, 20.0, 4.0, 0.5, key="at_nt_lam")
    cost = b.slider("거래비용 (bp)", 0.0, 50.0, 10.0, 1.0, key="at_nt_c")
    mu = np.array([0.06, 0.04, 0.08]); vol = np.array([0.25, 0.15, 0.30]); Rr = np.array([[1, 0.3, 0.5], [0.3, 1, 0.2], [0.5, 0.2, 1]])
    S = np.outer(vol, vol) * Rr
    import portfolio as P
    w0 = np.array([1 / 3] * 3)
    w, info = P.newton_mv(mu, S, w0, np.full(3, cost / 1e4), lam, "long_only")
    rng = np.random.default_rng(13)
    Wr = rng.dirichlet(np.ones(3), 3000)
    er, sr = Wr @ mu, np.sqrt(np.einsum("ni,ij,nj->n", Wr, S, Wr))
    f = go.Figure(go.Scatter(x=sr, y=er, mode="markers", marker=dict(size=3, color=GREY, opacity=0.4), name="무작위 비중 3,000개"))
    f.add_trace(go.Scatter(x=[np.sqrt(w @ S @ w)], y=[w @ mu], mode="markers", marker=dict(size=14, color=PAL[1], symbol="star"),
                           name=f"Newton 최적: 원유 {w[0]:.0%} · 금 {w[1]:.0%} · 구리 {w[2]:.0%}"))
    f.add_trace(go.Scatter(x=[np.sqrt(w0 @ S @ w0)], y=[w0 @ mu], mode="markers", marker=dict(size=11, color=PAL[0]), name="현재(1/N)"))
    _lay(f, 420, title=f"평균-분산-비용 최적화: max μ'w − λ/2·w'Σw − 비용·|w−w₀| (Newton {info['iter']}회 반복)",
         xaxis_title="변동성 (연)", yaxis_title="기대수익 (연)")
    t2, t3 = _tabs23(c)
    t2.plotly_chart(f, key="at_nt_fig")
    wg = np.linspace(0, 1, 61)
    W1, W2 = np.meshgrid(wg, wg)
    W3 = 1 - W1 - W2
    Wt = np.stack([W1, W2, W3], -1)
    U3 = Wt @ mu - lam / 2 * np.einsum("...i,ij,...j->...", Wt, S, Wt) - cost / 1e4 * np.abs(Wt - w0).sum(-1)
    U3 = np.where(W3 >= -1e-9, U3, np.nan) * 100
    uo = lambda w_: float((w_ @ mu - lam / 2 * w_ @ S @ w_ - cost / 1e4 * np.abs(w_ - w0).sum()) * 100)  # noqa: E731
    f3 = go.Figure(go.Surface(x=wg, y=wg, z=U3, colorscale="Teal", opacity=0.9, colorbar=dict(title="목적함수 (%)", thickness=12, len=0.6)))
    f3.add_trace(go.Scatter3d(x=[w[0]], y=[w[1]], z=[uo(w)], mode="markers", marker=dict(size=8, color=PAL[1], symbol="diamond"), name="Newton 최적"))
    f3.add_trace(go.Scatter3d(x=[w0[0]], y=[w0[1]], z=[uo(w0)], mode="markers", marker=dict(size=6, color=PAL[0]), name="현재 (1/N)"))
    t3.plotly_chart(_scene(f3, 540, "원유 비중 w₁", "금 비중 w₂ (구리 = 1 − w₁ − w₂)", "목적함수 μ'w − λ/2·w'Σw − 비용 (%)", eye=(1.5, -1.5, 1.0)), key="at_nt_3d")
    t3.caption("가능한 모든 롱온리 비중(삼각형) 위의 목적함수 언덕 — Newton은 기울기·곡률로 꼭대기(◆)를 몇 번 만에 찾음. "
               "λ를 키우면 언덕이 둥글게 좁아지며 꼭대기가 저위험(금) 쪽으로, 비용을 키우면 1/N 점 주변에 주름(|w−w₀|의 꺾임)이 생겨 꼭대기가 1/N 쪽으로 끌려옴.")
    c.caption("ML이 아니라 수치 최적화: 목적함수의 기울기(1차)와 곡률(2차, 헤시안)로 한 번에 최적점 근처로 점프 → 몇 번 만에 수렴. "
              "λ가 크면 저위험 쪽, 비용이 크면 현재 비중(1/N)에서 덜 움직임(무거래 구간). 입력 μ(예측)·Σ(공분산)의 질이 결과를 좌우.")


def viz_quantile(c):
    tau = c.slider("분위수 τ (0.05 = 하위 5% 꼬리)", 0.05, 0.95, 0.05, 0.05, key="at_qr_tau")
    import statsmodels.api as sm
    rng = np.random.default_rng(21)
    x = rng.uniform(-2, 2, 400); y = 0.3 * x + (0.5 + 0.6 * np.abs(x + 2) / 2) * rng.standard_t(4, 400)
    X = sm.add_constant(x); xx = np.linspace(-2, 2, 50); XX = sm.add_constant(xx)
    f = go.Figure(go.Scatter(x=x, y=y, mode="markers", marker=dict(color=GREY, size=5, opacity=0.5), name="관측치"))
    f.add_trace(go.Scatter(x=xx, y=XX @ sm.OLS(y, X).fit().params, name="OLS (평균)", line=dict(color="#222", dash="dot", width=2)))
    for j, t in enumerate(sorted({0.05, 0.5, 0.95, tau})):
        b = sm.QuantReg(y, X).fit(q=t).params
        f.add_trace(go.Scatter(x=xx, y=XX @ b, name=f"τ = {t:.2f} (기울기 {b[1]:.2f})",
                               line=dict(color=PAL[1] if abs(t - tau) < 1e-9 else PAL[0], width=3 if abs(t - tau) < 1e-9 else 1.5)))
    _lay(f, 430, title="분위수 회귀: 평균이 아니라 분포의 특정 분위(꼬리)가 X에 어떻게 반응하는지", xaxis_title="X (예: 달러 수익률)", yaxis_title="Y")
    c.plotly_chart(f, key="at_qr_fig")
    c.caption("오차가 X에 따라 커지면(이분산) 분위수 선들이 부채꼴로 벌어짐 → '평균 베타'는 같아도 '하락 꼬리(τ=0.05) 베타'는 다를 수 있음. "
              "위기 시 손실 위험(VaR) 민감도 분석에 적합. 손실 = 체크 함수 ρ_τ(u) = u(τ − 1[u<0]) → 이상치에 강건.")


def viz_csr(c):
    a, b = c.columns(2)
    snr = a.slider("신호 강도", 0.05, 1.0, 0.2, 0.05, key="at_csr_s")
    n = b.slider("학습 관측치 수", 60, 600, 150, 30, key="at_csr_n")
    from itertools import combinations
    rng = np.random.default_rng(22)
    p = 10; T = n + 400
    X = rng.standard_normal((T, p)) + 0.5 * rng.standard_normal((T, 1))
    beta = snr * rng.uniform(0.2, 1.0, p) / np.sqrt(p)
    y = X @ beta + rng.standard_normal(T)
    tr, te = slice(0, n), slice(n, T)
    r2 = lambda pr: 1 - np.sum((y[te] - pr) ** 2) / np.sum((y[te] - y[tr].mean()) ** 2)
    out = {}
    for k in range(1, p + 1):
        combs = list(combinations(range(p), k))
        if len(combs) > 120:
            combs = [combs[i] for i in rng.choice(len(combs), 120, replace=False)]
        preds = []
        for cmb in combs:
            Xk = np.column_stack([np.ones(n), X[tr][:, cmb]]); bk = np.linalg.lstsq(Xk, y[tr], rcond=None)[0]
            preds.append(np.column_stack([np.ones(T - n), X[te][:, cmb]]) @ bk)
        out[k] = r2(np.mean(preds, 0))
    s = pd.Series(out)
    f = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines+markers", line=dict(color=PAL[0], width=2.5), name="크기 k 조합들의 예측 평균"))
    f.add_hline(y=s[p], line_dash="dot", line_color=PAL[1], annotation_text="k = 전체 (= OLS 1개)")
    _lay(f, 380, title="Complete Subset: 변수 k개짜리 모든 조합 회귀의 예측을 평균", xaxis_title="조합 크기 k", yaxis_title="테스트 R²", hovermode="x unified")
    c.plotly_chart(f, key="at_csr_fig")
    c.caption(f"최적 k = {int(s.idxmax())} (R² {s.max():.3f}) vs 전체 OLS {s[p]:.3f}. 작은 모형들을 평균하면 각 모형의 추정 잡음이 상쇄되면서 "
              "모든 변수의 정보는 유지 → 변수를 버리지 않는 축소. 신호가 약하고 표본이 짧을수록 OLS 대비 이득이 큼 (Elliott-Gargano-Timmermann 2013).")


def viz_bma(c):
    a, b = c.columns(2)
    n = a.slider("관측치 수", 40, 500, 120, 20, key="at_bma_n")
    snr = b.slider("참 변수의 신호 강도", 0.1, 1.0, 0.35, 0.05, key="at_bma_s")
    from itertools import product
    rng = np.random.default_rng(23)
    p = 8; X = rng.standard_normal((n, p)); true = [0, 3, 5]
    y = X[:, true] @ np.full(3, snr) + rng.standard_normal(n)
    X = (X - X.mean(0)) / X.std(0); y = y - y.mean()
    g = max(n, p ** 2); lml, models = [], []
    tss = y @ y
    for gam in product([0, 1], repeat=p):
        idx = [j for j in range(p) if gam[j]]
        k = len(idx)
        if k:
            Xk = X[:, idx]; bh = np.linalg.lstsq(Xk, y, rcond=None)[0]; R2 = 1 - np.sum((y - Xk @ bh) ** 2) / tss
        else:
            R2 = 0.0
        lml.append(0.5 * (n - 1 - k) * np.log1p(g) - 0.5 * (n - 1) * np.log1p(g * (1 - R2)))   # Zellner g-prior 주변우도
        models.append(gam)
    lml = np.array(lml); w = np.exp(lml - lml.max()); w /= w.sum()
    pip = np.array(models).T @ w
    f = go.Figure(go.Bar(x=[f"x{j}" for j in range(p)], y=pip, marker_color=[PAL[2] if j in true else GREY for j in range(p)]))
    f.add_hline(y=0.5, line_dash="dot", line_color=PAL[1], annotation_text="0.5")
    _lay(f, 380, title="BMA 사후 포함확률 PIP (초록 = 실제로 의미 있는 변수)", yaxis=dict(range=[0, 1], title="P(변수가 모형에 포함 | 데이터)"))
    c.plotly_chart(f, key="at_bma_fig")
    top = np.argsort(w)[::-1][:3]
    c.caption("변수 조합 2⁸ = 256개 모형을 모두 추정하고 데이터 적합도(주변우도)로 가중 평균 — 하나의 '최선 모형'을 고르는 대신 모형 불확실성까지 반영. "
              f"상위 모형 확률: " + ", ".join(f"{{{', '.join(f'x{j}' for j in range(p) if models[i][j]) or '없음'}}} {w[i]:.0%}" for i in top) +
              ". 변수가 많으면 전부 열거가 불가 → 이 앱은 MC³(마르코프 연쇄로 모형 공간 탐색).")


def viz_horseshoe(c):
    tau = c.slider("전역 축소 τ (작을수록 전체적으로 강하게 0쪽으로)", 0.05, 1.0, 0.2, 0.05, key="at_hs_tau")
    z = np.linspace(-6, 6, 241)
    lam = np.exp(np.linspace(np.log(1e-3), np.log(1e3), 600))
    pl = 2 / (np.pi * (1 + lam ** 2))                                   # 반-코시 사전 (λ>0)
    dl = np.gradient(lam)
    v = 1 + (lam * tau) ** 2                                            # y | λ ~ N(0, 1 + λ²τ²)
    post_hs = []
    for zi in z:
        like = np.exp(-zi ** 2 / (2 * v)) / np.sqrt(v)
        wgt = like * pl * dl
        shrink = (lam * tau) ** 2 / v                                   # 1 − κ
        post_hs.append(zi * np.sum(wgt * shrink) / np.sum(wgt))
    lam_l = 1.0 / tau
    lasso = np.sign(z) * np.maximum(np.abs(z) - lam_l * 0.5, 0)
    ridge = z * tau ** 2 / (1 + tau ** 2) * 4 / (1 + 3 * tau ** 2) if False else z * (4 * tau ** 2) / (1 + 4 * tau ** 2)
    f = make_subplots(rows=1, cols=2, subplot_titles=("사전분포 모양 (0 근처 확대)", "관측값 z → 사후평균 β̂ (축소 모양)"))
    bb = np.linspace(-3, 3, 400)
    from scipy.stats import norm, laplace
    hs_den = np.array([np.sum(norm.pdf(x_, 0, lam * tau + 1e-12) * pl * dl) for x_ in bb])
    f.add_trace(go.Scatter(x=bb, y=norm.pdf(bb, 0, 1), name="정규 (Ridge)", line=dict(color=PAL[0])), 1, 1)
    f.add_trace(go.Scatter(x=bb, y=laplace.pdf(bb, 0, 0.7), name="라플라스 (Lasso)", line=dict(color=PAL[2])), 1, 1)
    f.add_trace(go.Scatter(x=bb, y=np.minimum(hs_den, 3), name="Horseshoe", line=dict(color=PAL[1], width=2.5)), 1, 1)
    f.add_trace(go.Scatter(x=z, y=ridge, name="Ridge", line=dict(color=PAL[0]), showlegend=False), 1, 2)
    f.add_trace(go.Scatter(x=z, y=lasso, name="Lasso", line=dict(color=PAL[2]), showlegend=False), 1, 2)
    f.add_trace(go.Scatter(x=z, y=post_hs, name="Horseshoe", line=dict(color=PAL[1], width=2.5), showlegend=False), 1, 2)
    f.add_trace(go.Scatter(x=z, y=z, name="축소 없음", line=dict(color=GREY, dash="dot")), 1, 2)
    c.plotly_chart(_lay(f, 420), key="at_hs_fig")
    c.caption("Horseshoe: 0에 무한히 뾰족한 봉우리 + 두꺼운 꼬리 → 작은 신호는 거의 0으로 강하게 누르고, 큰 신호는 거의 그대로 둠 (Lasso처럼 큰 계수까지 깎지 않음). "
              "계수를 정확히 0으로 자르지 않고 사후분포(신용구간)를 줘서 '이 변수가 유의한가'를 확률로 답함. 추정은 Gibbs 표본추출.")


def viz_minnesota(c):
    a, b = c.columns(2)
    lam = a.slider("전체 강도 λ (작을수록 강한 축소)", 0.05, 1.0, 0.2, 0.05, key="at_mn_lam")
    th = b.slider("타 변수 계수 상대 강도 θ", 0.1, 1.0, 0.5, 0.05, key="at_mn_th")
    lags = np.arange(1, 7)
    f = make_subplots(rows=1, cols=2, subplot_titles=("사전 표준편차: 시차·변수별", "짧은 표본에서 계수 추정 (참 vs OLS vs Minnesota)"))
    f.add_trace(go.Bar(x=[f"t-{l}" for l in lags], y=lam / lags, name="자기 시차", marker_color=PAL[0]), 1, 1)
    f.add_trace(go.Bar(x=[f"t-{l}" for l in lags], y=lam * th / lags, name="타 변수 시차", marker_color=PAL[1]), 1, 1)
    rng = np.random.default_rng(24)
    T, N, P_ = 60, 3, 4
    Y = np.zeros((T + P_, N)); A1 = np.array([[0.4, 0.15, 0], [0, 0.3, 0.1], [0.1, 0, 0.35]])
    for t in range(1, T + P_):
        Y[t] = A1 @ Y[t - 1] + rng.standard_normal(N)
    X = np.column_stack([Y[P_ - l:T + P_ - l] for l in range(1, P_ + 1)]); yy = Y[P_:, 0]
    b_ols = np.linalg.lstsq(X, yy, rcond=None)[0]
    sd = np.array([(lam if j == 0 else lam * th) / l for l in range(1, P_ + 1) for j in range(N)])
    b_mn = np.linalg.solve(X.T @ X + np.diag(1 / sd ** 2), X.T @ yy)
    b_true = np.r_[A1[0], np.zeros(N * (P_ - 1))]
    lab = [f"{'자기' if j == 0 else f'변수{j}'} t-{l}" for l in range(1, P_ + 1) for j in range(N)]
    f.add_trace(go.Scatter(x=lab, y=b_true, mode="markers", marker=dict(color="#222", size=9, symbol="x"), name="참값"), 1, 2)
    f.add_trace(go.Scatter(x=lab, y=b_ols, mode="markers", marker=dict(color=GREY, size=8), name=f"OLS (오차 {np.sqrt(np.mean((b_ols - b_true) ** 2)):.3f})"), 1, 2)
    f.add_trace(go.Scatter(x=lab, y=b_mn, mode="markers", marker=dict(color=PAL[1], size=9), name=f"Minnesota (오차 {np.sqrt(np.mean((b_mn - b_true) ** 2)):.3f})"), 1, 2)
    c.plotly_chart(_lay(f, 440, barmode="group"), key="at_mn_fig")
    c.caption("VAR은 변수·시차가 늘면 모수가 폭증해 짧은 표본에서 추정이 불안정 → '먼 시차일수록, 다른 변수일수록 0에 가깝다'는 상식을 사전분포로 부여 "
              "(Litterman 1986). 데이터가 강하게 반대하지 않는 한 계수를 0 쪽으로 당겨 예측오차를 줄임.")


def viz_stacking(c):
    rng = np.random.default_rng(25)
    n = 600; x = rng.uniform(-3, 3, (n, 2)); y = 0.6 * x[:, 0] + np.sin(2 * x[:, 1]) + 0.5 * rng.standard_normal(n)
    from sklearn.linear_model import Ridge as R_
    from sklearn.ensemble import GradientBoostingRegressor as G_
    from sklearn.neighbors import KNeighborsRegressor as K_
    from scipy.optimize import nnls
    tr, te = slice(0, 400), slice(400, n)
    base = {"Ridge (선형)": lambda: R_(1.0), "GBM (나무)": lambda: G_(n_estimators=150, max_depth=2), "kNN (근접)": lambda: K_(15)}
    oof = np.zeros((400, 3)); folds = np.array_split(np.arange(400), 5)
    for k, (nm, mk) in enumerate(base.items()):
        for i in range(1, 5):                                           # 시계열식: 앞 폴드로 학습 → 다음 폴드 예측
            trn = np.concatenate(folds[:i]); oof[folds[i], k] = mk().fit(x[trn], y[trn]).predict(x[folds[i]])
    ok = np.concatenate(folds[1:])
    w, _ = nnls(oof[ok], y[ok]); w = w / w.sum() if w.sum() > 0 else np.full(3, 1 / 3)
    P_ = np.column_stack([mk().fit(x[tr], y[tr]).predict(x[te]) for mk in base.values()])
    mse = {nm: np.mean((y[te] - P_[:, k]) ** 2) for k, nm in enumerate(base)}
    mse["단순 평균"] = np.mean((y[te] - P_.mean(1)) ** 2); mse["Stacking"] = np.mean((y[te] - P_ @ w) ** 2)
    f = make_subplots(rows=1, cols=2, subplot_titles=("학습된 가중치 (비음·합 1)", "테스트 MSE (낮을수록 좋음)"))
    f.add_trace(go.Bar(x=list(base), y=w, marker_color=PAL[0], showlegend=False), 1, 1)
    f.add_trace(go.Bar(x=list(mse), y=list(mse.values()), marker_color=[GREY] * 3 + [PAL[2], PAL[1]], showlegend=False), 1, 2)
    c.plotly_chart(_lay(f, 400), key="at_st_fig")
    c.caption("데이터에 선형(x₀) + 비선형(sin x₁) 성분이 섞여 있으면 어느 한 모델도 완벽하지 않음 → 각 모델의 '외표본' 예측을 입력으로 최적 결합 비중을 학습. "
              "학습구간 안에서만 시계열 교차검증으로 가중치를 정하므로 미래정보 없음 (Wolpert 1992, Breiman 1996).")


@st.cache_data(show_spinner=False)
def _lstm_demo(seq: int, seed: int = 0):
    import torch
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    T = 500; t = np.arange(T)
    s = np.sin(t / 8) * (1 + 0.5 * (t > 250)) + 0.25 * rng.standard_normal(T)
    Xs = np.stack([s[i:i + seq] for i in range(T - seq)]); ys = s[seq:]
    Xt = torch.tensor(Xs[:, :, None], dtype=torch.float32); yt = torch.tensor(ys[:, None], dtype=torch.float32)
    tr = int(0.7 * len(ys))
    class M(torch.nn.Module):
        def __init__(s_):
            super().__init__(); s_.l = torch.nn.LSTM(1, 16, batch_first=True); s_.o = torch.nn.Linear(16, 1)
        def forward(s_, x):
            h, _ = s_.l(x); return s_.o(h[:, -1])
    m = M(); opt = torch.optim.Adam(m.parameters(), 1e-2)
    for _ in range(150):
        opt.zero_grad(); loss = torch.mean((m(Xt[:tr]) - yt[:tr]) ** 2); loss.backward(); opt.step()
    with torch.no_grad():
        pr = m(Xt).numpy().ravel()
    return s, pr, seq, tr


def viz_lstm(c):
    seq = c.slider("입력 창 길이 (과거 몇 기간을 한꺼번에 보나)", 5, 60, 20, 5, key="at_ls_seq")
    s, pr, seq, tr = _lstm_demo(seq)
    t = np.arange(len(s))
    f = go.Figure([go.Scatter(x=t, y=s, name="시계열", line=dict(color=GREY, width=1)),
                   go.Scatter(x=t[seq:], y=pr, name="LSTM 1기 앞 예측", line=dict(color=PAL[0], width=2))])
    f.add_vrect(x0=seq + tr, x1=len(s), fillcolor="rgba(168,75,47,0.08)", line_width=0, annotation_text="테스트")
    f.add_vrect(x0=seq + tr - seq, x1=seq + tr, fillcolor="rgba(32,128,141,0.15)", line_width=0, annotation_text="입력 창 예시")
    te_mse = np.mean((s[seq + tr:] - pr[tr:]) ** 2); naive = np.mean((s[seq + tr:] - s[seq + tr - 1:-1]) ** 2)
    _lay(f, 420, title=f"LSTM: 창 안의 순서를 기억 셀로 요약해 다음 값을 예측 — 테스트 MSE {te_mse:.3f} vs 직전값 {naive:.3f}",
         xaxis_title="시점", hovermode="x unified")
    c.plotly_chart(f, key="at_ls_fig")
    c.caption("순환 신경망의 한 종류: 입력·망각·출력 게이트가 '무엇을 기억하고 버릴지' 학습 → 긴 패턴(주기, 레짐)을 잡음. "
              "이 예시처럼 규칙적 패턴이 있으면 강력하지만, 원자재 수익률처럼 신호가 약하고 표본이 수천 개뿐이면 과적합이 쉬움 (이 앱 실험에서도 Ridge보다 열위).")


def viz_favar(c):
    a, b = c.columns(2)
    k = a.slider("요인 수 K", 1, 4, 2, key="at_fv_k")
    load = b.slider("요인 → Y 영향", 0.0, 1.0, 0.5, 0.1, key="at_fv_l")
    import analytics as Aan
    rng = np.random.default_rng(26)
    T, N = 400, 30
    F = np.zeros((T, 2))
    for t in range(1, T):
        F[t] = np.array([[0.7, 0], [0.2, 0.5]]) @ F[t - 1] + rng.standard_normal(2)
    Lmb = rng.standard_normal((2, N)); Xp = F @ Lmb + rng.standard_normal((T, N))
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.2 * y[t - 1] + load * F[t - 1, 0] + 0.8 * rng.standard_normal()
    idx = pd.bdate_range("2000-01-03", periods=T)
    res = Aan.favar(pd.DataFrame(Xp, index=idx, columns=[f"시장{i}" for i in range(N)]), pd.DataFrame({"Y (옥수수)": y}, index=idx), k=k, p=1, H=12)
    oirf = res["oirf"]; nm = res["names"]; j_y = nm.index("Y (옥수수)")
    f = go.Figure([go.Scatter(x=np.arange(oirf.shape[0]), y=np.cumsum(oirf[:, j_y, j]), name=f"{nm[j]} 충격 → Y 누적 반응",
                              line=dict(color=PAL[j % 8], width=2.5)) for j in range(len(nm))])
    f.add_hline(y=0, line_color=GREY, line_width=1)
    _lay(f, 400, title="FAVAR 충격반응: 30개 시장에서 뽑은 공통요인의 충격이 Y에 어떻게 퍼지나", xaxis_title="기간", yaxis_title="Y 누적 반응 (표준화)", hovermode="x unified")
    c.plotly_chart(f, key="at_fv_fig")
    c.caption("① 많은 변수(원자재·거시)에서 PCA로 공통요인 F 추출 → ② VAR([F, Y]) → ③ 요인 충격에 대한 Y의 충격반응(IRF)·분산분해. "
              "변수 몇 개만 넣는 VAR의 '변수 누락' 문제를 요인으로 해결 (Bernanke-Boivin-Eliasz 2005). 적재값으로 패널 전체 변수의 반응도 복원.")


def viz_panel(c):
    a_, b_ = c.columns(2)
    rho = a_.slider("품목 고유효과 αᵢ 와 X의 상관 (RE 가정 위반 정도)", 0.0, 0.95, 0.7, 0.05, key="at_pn_rho")
    Tt = b_.slider("품목당 기간 수 T", 3, 60, 5, 1, key="at_pn_T")
    Ni = 30

    def draw(seed):
        rng = np.random.default_rng(seed)
        alpha = rng.standard_normal(Ni) * 0.6
        x = 2 * rho * alpha[:, None] + rng.standard_normal((Ni, Tt)) * 0.8
        y = alpha[:, None] + 0.5 * x + 0.6 * rng.standard_normal((Ni, Tt))
        return x, y

    def estimates(x, y):
        b_pool = np.polyfit(x.ravel(), y.ravel(), 1)[0]
        xd, yd = x - x.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
        b_fe = (xd * yd).sum() / (xd ** 2).sum()
        s_e2 = ((yd - b_fe * xd) ** 2).sum() / (Ni * (Tt - 1) - 1)
        ab = y.mean(1) - b_fe * x.mean(1); s_a2 = max(np.var(ab, ddof=1) - s_e2 / Tt, 1e-6)
        th = 1 - np.sqrt(s_e2 / (s_e2 + Tt * s_a2))
        xq, yq = x - th * x.mean(1, keepdims=True), y - th * y.mean(1, keepdims=True)
        b_re = np.polyfit(xq.ravel(), yq.ravel(), 1)[0]
        b_fm = np.mean([np.polyfit(x[:, t], y[:, t], 1)[0] for t in range(Tt)])
        return np.array([b_pool, b_re, b_fe, b_fm, th])

    E = np.mean([estimates(*draw(s_)) for s_ in range(50)], axis=0)    # 50회 반복 평균 (표본 잡음 제거 → 편향만 보임)
    x, y = draw(27)
    b_pool, _, b_fe, _, _ = estimates(x, y)
    f = go.Figure()
    for i in range(Ni):
        f.add_trace(go.Scatter(x=x[i], y=y[i], mode="markers", marker=dict(size=4, color=PAL[i % 8], opacity=0.5), name=f"품목 {i + 1}", showlegend=False))
        xs = np.array([x[i].min(), x[i].max()])
        f.add_trace(go.Scatter(x=xs, y=y[i].mean() + b_fe * (xs - x[i].mean()), mode="lines", line=dict(color=PAL[i % 8], width=1.5), showlegend=False))
    xs = np.array([x.min(), x.max()])
    f.add_trace(go.Scatter(x=xs, y=y.mean() + b_pool * (xs - x.mean()), line=dict(color="#222", width=3, dash="dash"), name=f"Pooled OLS (이 표본) β = {b_pool:.2f}"))
    _lay(f, 440, title=f"참 β = 0.50 | 50회 반복 평균: Pooled {E[0]:.2f} · RE {E[1]:.2f} (θ={E[4]:.2f}) · FE {E[2]:.2f} · Fama-MacBeth {E[3]:.2f}",
         xaxis_title="X (품목 특성)", yaxis_title="Y (수익률)")
    c.plotly_chart(f, key="at_pn_fig")
    c.caption("색 = 품목. 품목마다 수준(αᵢ)이 다르고 그 수준이 X와 상관되면, 품목 간 차이를 섞어 쓰는 Pooled·Fama-MacBeth(횡단면 회귀 평균)는 기울기가 왜곡됨. "
              "FE는 품목별 평균을 '전부' 빼서(얇은 선 = 품목 안의 기울기) αᵢ를 제거 → 참값 회복. RE는 평균을 θ만큼만 빼므로 T가 짧으면(θ 작음) 편향, "
              "T를 늘리면 θ→1이 되어 FE와 같아짐 (우리 원자재 패널은 T≈1,000주 → RE≈FE). 상관을 0으로 내리면 모든 추정이 일치 → 이 차이를 검정하는 것이 하우스만 검정.")


def viz_lw(c):
    n = c.slider("관측치 수 n (변수 p = 40)", 45, 600, 80, 5, key="at_lw_n")
    from sklearn.covariance import LedoitWolf
    rng = np.random.default_rng(28)
    p = 40; L_ = rng.standard_normal((p, 3)) * 0.5
    Sig = L_ @ L_.T + np.diag(rng.uniform(0.5, 1.5, p))
    X = rng.multivariate_normal(np.zeros(p), Sig, n)
    Ssam = np.cov(X.T); lw = LedoitWolf().fit(X)
    ev = lambda M: np.sort(np.linalg.eigvalsh(M))[::-1]
    f = go.Figure([go.Scatter(y=ev(Sig), name="참 공분산", line=dict(color="#222", width=2)),
                   go.Scatter(y=ev(Ssam), name="표본 공분산", line=dict(color=GREY, width=2, dash="dot")),
                   go.Scatter(y=ev(lw.covariance_), name=f"Ledoit-Wolf (수축 강도 {lw.shrinkage_:.2f})", line=dict(color=PAL[1], width=2.5))])
    _lay(f, 400, title="고유값: 표본 공분산은 큰 것은 더 크게, 작은 것은 더 작게 (과장된 확산)", xaxis_title="고유값 순위", yaxis=dict(type="log", title="고유값 (로그)"),
         hovermode="x unified")
    c.plotly_chart(f, key="at_lw_fig")
    c.caption(f"최소 고유값 — 참 {ev(Sig)[-1]:.2f}, 표본 {ev(Ssam)[-1]:.3f}, LW {ev(lw.covariance_)[-1]:.2f}. 최적화는 '가장 작은 분산 방향'에 몰빵하므로 "
              "표본 공분산의 과소추정된 작은 고유값이 극단적 비중을 만듦 → 표본 공분산을 단순한 목표(대각)와 가중 평균해 확산을 줄임. "
              "가중치는 오차를 최소화하도록 데이터로 결정 (Ledoit-Wolf 2004). n이 p에 가까울수록 효과 큼.")


def viz_rp(c):
    a, b = c.columns(2)
    err = a.slider("기대수익 μ 추정 오차 (연 %p)", 0.0, 10.0, 4.0, 0.5, key="at_rp_err")
    lam = b.slider("위험회피 λ", 1.0, 20.0, 5.0, 1.0, key="at_rp_lam")
    import portfolio as P_
    names = ["원유", "옥수수", "금", "구리"]
    vol = np.array([0.35, 0.25, 0.15, 0.25]); Rr = np.array([[1, .3, .2, .5], [.3, 1, .1, .3], [.2, .1, 1, .3], [.5, .3, .3, 1]])
    S = np.outer(vol, vol) * Rr; mu_true = np.full(4, 0.03)
    rng = np.random.default_rng(29)
    W = {"1/N 동일비중": [], "리스크 패리티": [], "평균-분산 (Newton)": []}
    for _ in range(40):
        mu_hat = mu_true + err / 100 * rng.standard_normal(4)
        W["1/N 동일비중"].append(np.full(4, 0.25))
        W["리스크 패리티"].append(P_.newton_risk_parity(S)[0])
        W["평균-분산 (Newton)"].append(P_.newton_mv(mu_hat, S, np.full(4, 0.25), np.zeros(4), lam, "long_only")[0])
    f = make_subplots(rows=1, cols=2, subplot_titles=("비중: 평균 ± 추정오차에 따른 흔들림 (40회)", "위험 기여도 (평균 비중 기준, 합 = 100%)"))
    for j, (nm, ws) in enumerate(W.items()):
        ws = np.array(ws)
        f.add_trace(go.Bar(x=names, y=ws.mean(0), error_y=dict(type="data", array=ws.std(0)), name=nm, marker_color=PAL[j]), 1, 1)
        w_ = ws.mean(0); rc = w_ * (S @ w_) / (w_ @ S @ w_)
        f.add_trace(go.Bar(x=names, y=rc, name=nm, marker_color=PAL[j], showlegend=False), 1, 2)
    c.plotly_chart(_lay(f, 440, barmode="group"), key="at_rp_fig")
    c.caption("1/N: 금액을 똑같이 → 위험은 변동성 큰 원유가 지배. 리스크 패리티: '위험 기여'를 똑같이 → 변동성 큰 자산을 적게. μ(예측)를 전혀 쓰지 않아 "
              "추정오차에 흔들리지 않음. 평균-분산: μ까지 써서 이론상 최적이지만, μ 오차가 크면 비중이 크게 흔들림(오차 막대). "
              "→ 예측력이 약할수록(원자재 수익률) 리스크 패리티가 실전에서 강한 이유.")


def viz_har(c):
    rng = np.random.default_rng(30)
    T = 1500; h = np.empty(T); e = np.empty(T); h[0] = 1
    for t in range(T):
        if t:
            h[t] = 0.02 + 0.08 * e[t - 1] ** 2 + 0.9 * h[t - 1]
        e[t] = np.sqrt(h[t]) * rng.standard_normal()
    r2 = pd.Series(e ** 2)
    d, w, m = r2, r2.rolling(5).mean(), r2.rolling(22).mean()
    tgt = r2.rolling(5).mean().shift(-5)
    D = pd.DataFrame({"d": d, "w": w, "m": m, "y": tgt}).dropna()
    b = np.linalg.lstsq(np.column_stack([np.ones(len(D)), D[["d", "w", "m"]].values]), D.y.values, rcond=None)[0]
    pred = np.column_stack([np.ones(len(D)), D[["d", "w", "m"]].values]) @ b
    sl = slice(800, 1100)
    f = go.Figure([go.Scatter(x=D.index[sl], y=np.sqrt(D.y.values[sl]), name="실제 다음 주 변동성", line=dict(color=GREY)),
                   go.Scatter(x=D.index[sl], y=np.sqrt(np.maximum(pred[sl], 1e-6)), name="HAR 예측", line=dict(color=PAL[0], width=2.5))])
    _lay(f, 380, title=f"HAR-RV: 다음 주 RV = {b[0]:.2f} + {b[1]:.2f}·일간 + {b[2]:.2f}·주간 + {b[3]:.2f}·월간", xaxis_title="시점", hovermode="x unified")
    c.plotly_chart(f, key="at_har_fig")
    c.caption("일간·주간·월간 평균 실현분산 3개만으로 하는 단순 선형회귀 (Corsi 2009) — 서로 다른 시간 단위로 거래하는 참가자들의 변동성이 겹친다는 직관. "
              "단순하지만 GARCH와 맞먹는 예측력, MIDAS는 이 계단식 가중을 매끄러운 Beta 함수로 일반화.")


def viz_cv(c):
    a, b = c.columns(2)
    k = a.slider("폴드 수", 3, 8, 5, key="at_cv_k")
    gap = b.slider("간격 gap = 예측기간 h − 1", 0, 5, 2, key="at_cv_g")
    n = 100
    f = go.Figure()
    rows = []
    edges = np.linspace(0, n, k + 2).astype(int)
    for i in range(k):
        tr_end = edges[i + 1]; te0 = tr_end + gap; te1 = edges[i + 2]
        rows.append(("시계열 CV", i, [(0, tr_end, "학습", PAL[0]), (tr_end, te0, "gap(버림)", GREY), (te0, te1, "검증", PAL[1])]))
    rng = np.random.default_rng(31); perm = rng.permutation(n)
    for i in range(min(k, 3)):
        te = np.sort(perm[i::k])[:n // k]
        rows.append(("무작위 K-fold (금지)", i, [(int(t), int(t) + 1, "검증", PAL[7]) for t in te]))
    y_lab = []
    for r_i, (kind, i, segs) in enumerate(rows):
        lab = f"{kind} {i + 1}"; y_lab.append(lab)
        if kind.startswith("무작위"):
            f.add_trace(go.Bar(y=[lab], x=[n], base=[0], orientation="h", marker_color=PAL[0], opacity=0.35, showlegend=False))
        for s0, s1, nm, col in segs:
            if s1 > s0:
                f.add_trace(go.Bar(y=[lab], x=[s1 - s0], base=[s0], orientation="h", marker_color=col, name=nm, showlegend=False))
    f.update_layout(barmode="overlay", height=120 + 34 * len(rows), margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(autorange="reversed"),
                    title="파랑 = 학습, 주황 = 검증, 회색 = gap — 시계열은 항상 '과거로 학습 → 미래로 검증'", xaxis_title="시간 →")
    c.plotly_chart(f, key="at_cv_fig")
    c.caption("무작위 K-fold는 미래 데이터로 학습해 과거를 검증하는 꼴(미래정보 누수) → 시계열에선 성과가 과대평가. 목표가 h기 누적이면 학습 끝 h−1개가 "
              "검증 목표와 겹치므로 gap으로 제거. Walk-forward는 이 '확장창 학습 → 다음 구간 예측'을 시간 순으로 반복하며 모델을 재학습 — 실전 운용과 같은 조건.")


def viz_tests(c):
    r2_true = c.slider("참 예측력 (기간당 R², %)", 0.0, 3.0, 0.5, 0.1, key="at_ts_r2") / 100
    rng = np.random.default_rng(32)
    T = 600; x = rng.standard_normal(T); b = np.sqrt(r2_true / (1 - r2_true)) if r2_true > 0 else 0
    y = b * x + rng.standard_normal(T)
    p_m, p_b = [], []
    for t in range(120, T):
        bb = np.polyfit(x[:t - 1], y[1:t] if False else y[:t - 1], 1)
        p_m.append(np.polyval(bb, x[t])); p_b.append(y[:t].mean())
    yt = y[120:]; p_m, p_b = np.array(p_m), np.array(p_b)
    oos = 1 - np.sum((yt - p_m) ** 2) / np.sum((yt - p_b) ** 2)
    fcw = (yt - p_b) ** 2 - ((yt - p_m) ** 2 - (p_b - p_m) ** 2)
    cw = fcw.mean() / (fcw.std(ddof=1) / np.sqrt(len(fcw)))
    cum = np.cumsum((yt - p_b) ** 2 - (yt - p_m) ** 2)
    f = go.Figure(go.Scatter(y=cum, line=dict(color=PAL[0], width=2.5), name="누적 (기준 오차² − 모델 오차²)"))
    f.add_hline(y=0, line_color=GREY)
    _lay(f, 380, title=f"OOS R² = {oos:+.2%} · Clark-West t = {cw:+.2f} ({'유의' if cw > 1.645 else '유의하지 않음'}, 단측 5%)",
         xaxis_title="테스트 시점", yaxis_title="누적 오차 차이", hovermode="x unified")
    c.plotly_chart(f, key="at_ts_fig")
    c.caption("OOS R² = 1 − 모델 오차² / 과거평균 예측 오차² (Campbell-Thompson) — 0 미만이면 '그냥 과거 평균'보다 못함. 선이 우상향하는 구간 = 모델이 이기는 구간 "
              "(Goyal-Welch 그림). 큰 모델이 작은 모델을 포함하면 추정 잡음 때문에 DM 검정이 보수적 → Clark-West 보정. 참 R² 0.5%도 수백 기간이 지나야 겨우 구별됨.")


def viz_band(c):
    a, b = c.columns(2)
    band = a.slider("무거래 폭 (|목표 − 현재|)", 0.0, 0.4, 0.15, 0.01, key="at_bd_b")
    vt = b.checkbox("변동성 타기팅 (목표 10%)", True, key="at_bd_vt")
    rng = np.random.default_rng(33)
    T = 500
    sig = np.cumsum(rng.standard_normal(T)) * 0.05
    target = np.tanh(sig + 0.4 * rng.standard_normal(T))               # 잡음 섞인 일간 목표 비중
    vol = 0.15 * np.exp(0.5 * np.sin(np.arange(T) / 60) + 0.3 * (np.arange(T) > 300))
    if vt:
        target = target * np.minimum(2.0, 0.10 / vol)
    w = np.zeros(T)
    for t in range(1, T):
        w[t] = target[t] if abs(target[t] - w[t - 1]) > band else w[t - 1]
    f = go.Figure([go.Scatter(y=target, name=f"매일 목표로 (회전율 {np.abs(np.diff(target)).sum():.0f})", line=dict(color=GREY, width=1)),
                   go.Scatter(y=w, name=f"무거래 구간 (회전율 {np.abs(np.diff(w)).sum():.0f}, 거래 {int((np.diff(w) != 0).sum())}회)",
                              line=dict(color=PAL[1], width=2.5, shape="hv"))])
    _lay(f, 400, title="필요할 때만 거래: 목표와의 차이가 폭을 넘을 때만 따라감", xaxis_title="거래일", yaxis_title="비중", hovermode="x unified")
    c.plotly_chart(f, key="at_bd_fig")
    c.caption("거래비용이 있으면 최적 정책은 '목표 주변의 무거래 구간 안에서는 가만히' (Leland 2000). 매일 흔들리는 신호의 잡음성 매매를 걸러 회전율을 크게 줄임. "
              "변동성 타기팅은 예측 변동성이 높을수록(그림 후반) 비중을 줄여 위험을 일정하게 — 이 앱 실험: 실현 변동성은 목표에 붙고 낙폭은 줄지만 Sharpe는 거의 그대로.")


# ============================================================================
# 모델 사전
# ============================================================================


# ---------------------------------------------------------------------------
# 대안 시계열 모델 (⑤ 포트폴리오 › 대안 시계열 모델)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _prophet_demo(cps: float, seed: int = 0):
    import logging
    from prophet import Prophet
    for lg in ("cmdstanpy", "prophet"):
        logging.getLogger(lg).setLevel(logging.ERROR)
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2018-01-05", periods=260, freq="W-FRI")
    t = np.arange(260)
    trend = 0.004 * t - 0.012 * np.maximum(t - 110, 0) + 0.015 * np.maximum(t - 190, 0)
    y = trend + 0.08 * np.sin(2 * np.pi * ds.dayofyear / 365.25) + 0.035 * rng.standard_normal(260)
    tr = 208
    m = Prophet(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=3, changepoint_prior_scale=cps, uncertainty_samples=200)
    m.fit(pd.DataFrame({"ds": ds[:tr], "y": y[:tr]}))
    f = m.predict(pd.DataFrame({"ds": ds}))
    deltas = np.abs(m.params["delta"].mean(0))
    return ds, y, trend, f, tr, list(m.changepoints), deltas


def viz_prophet(c):
    cps = c.select_slider("변곡 유연성 (changepoint_prior_scale)", [0.005, 0.02, 0.05, 0.2, 0.8], 0.05, key="at_pr_cps")
    ds, y, trend, f, tr, cpt, dl = _prophet_demo(cps)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=ds, y=y, mode="markers", marker=dict(size=3, color=GREY), name="관측 (주간 로그가격)"), 1, 1)
    fig.add_trace(go.Scatter(x=ds, y=f.yhat_upper, line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=ds, y=f.yhat_lower, fill="tonexty", fillcolor="rgba(32,128,141,0.15)", line=dict(width=0), name="80% 구간"), 1, 1)
    fig.add_trace(go.Scatter(x=ds, y=f.yhat, line=dict(color=PAL[0], width=2), name="적합·예측 (추세 + 계절)"), 1, 1)
    fig.add_trace(go.Scatter(x=ds, y=f.trend, line=dict(color=PAL[1], width=2, dash="dot"), name="추세 성분"), 1, 1)
    big = [d for d, v in zip(cpt, dl) if v > 0.1 * dl.max()]
    for d in big:
        fig.add_vline(x=d, line=dict(color=PAL[3], width=1, dash="dot"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ds, y=f.yearly, line=dict(color=PAL[2], width=2), name="연간 계절 성분"), 2, 1)
    fig.add_vrect(x0=ds[tr], x1=ds[-1], fillcolor="rgba(168,75,47,0.08)", line_width=0, annotation_text="예측 구간", row=1, col=1)
    _lay(fig, 460, title=f"Prophet: 추세(변곡점) + 계절로 분해 — 실제로 꺾인 곳 2곳(110·190주), 강한 변곡 {len(big)}개 감지", hovermode="x unified")
    c.plotly_chart(fig, key="at_pr_fig")
    c.caption("실제 Meta prophet 패키지를 합성 자료에 적합. 유연성을 키우면 추세가 잡음까지 따라가고(과적합), 줄이면 실제 꺾임도 놓침. "
              "예측 구간에서는 마지막 추세 기울기를 그대로 연장 — 원자재처럼 추세가 갑자기 바뀌는 자료에서 4주 수익률 예측력이 약했던 이유.")


@st.cache_data(show_spinner=False)
def _fm_cached(name: str):
    """롱숏 19개 품목 백테스트 때 저장한 파운데이션 모델 예측 (data/fm/{name}_ls19.npz). 없으면 None."""
    from pathlib import Path
    fn = Path(__file__).resolve().parent / "data" / "fm" / f"{name}_ls19.npz"
    if not fn.exists():
        return None
    import analytics as A
    import forecast_lab as FL
    out = np.load(fn, allow_pickle=True)["out"].item()
    L = A.load_levels(roll_adjusted=True)
    res = {}
    for i in ("yf:ZC=F", "yf:CL=F", "yf:GC=F", "yf:HG=F", "yf:KC=F", "yf:LE=F"):
        if i in out and i in L:
            s = np.log(L[i].dropna())
            res[i] = (s, FL.fridays(s.index, start="2012-01-01", end_pad=0), out[i])
    return res


def _fm_fan(c, name: str, key: str):
    data = _fm_cached(name)
    if not data:
        c.info(f"{name} 예측 캐시가 없습니다 — `python experiments.py alt_ts` 실행 후 실제 원자재 예측 부채꼴이 표시됩니다.")
        return
    nm = {"yf:ZC=F": "옥수수", "yf:CL=F": "WTI 원유", "yf:GC=F": "금", "yf:HG=F": "구리", "yf:KC=F": "커피", "yf:LE=F": "생우"}
    k1, k2 = c.columns([1, 2])
    i = k1.selectbox("품목", list(data), format_func=lambda x: nm.get(x, x), key=f"{key}_item")
    s, dates, q = data[i]
    j = k2.slider("결정일 (과거 → 최근)", 0, len(dates) - 1, len(dates) - 30, key=f"{key}_j")
    d0 = dates[j]
    p = s.index.get_loc(d0)
    hist = s.iloc[max(0, p - 120):p + 1]
    fut = s.iloc[p + 1:p + 21]
    fx = pd.bdate_range(d0, periods=21)[1:]
    qq = np.exp(q[j]); px = np.exp
    f = go.Figure([go.Scatter(x=hist.index, y=px(hist), line=dict(color=GREY, width=1.5), name="과거 (모델 입력 일부)")])
    f.add_trace(go.Scatter(x=fx, y=qq[:, 8], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    f.add_trace(go.Scatter(x=fx, y=qq[:, 0], fill="tonexty", fillcolor="rgba(32,128,141,0.15)", line=dict(width=0), name="10–90% 분위"))
    f.add_trace(go.Scatter(x=fx, y=qq[:, 6], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    f.add_trace(go.Scatter(x=fx, y=qq[:, 2], fill="tonexty", fillcolor="rgba(32,128,141,0.3)", line=dict(width=0), name="30–70% 분위"))
    f.add_trace(go.Scatter(x=fx, y=qq[:, 4], line=dict(color=PAL[0], width=2), name="중앙값 예측"))
    if len(fut):
        f.add_trace(go.Scatter(x=fut.index, y=px(fut), line=dict(color=PAL[1], width=2), name="실제 (결과)"))
    _lay(f, 360, title=f"{name} · {nm.get(i, i)} · {d0:%Y-%m-%d} 결정 → 20영업일 분위 예측 (zero-shot, 학습 없음)", hovermode="x unified")
    c.plotly_chart(f, key=f"{key}_fig")


def viz_chronos(c):
    rng = np.random.default_rng(1)
    x = 50 + np.cumsum(rng.normal(0, 1, 60))
    nb = c.select_slider("토큰 구간 수 (예시 · 실제 Chronos는 4,096개)", [8, 16, 32, 64], 16, key="at_ch_nb")
    sc = np.abs(x).mean()
    edges = np.linspace((x / sc).min(), (x / sc).max(), nb + 1)
    tok = np.clip(np.digitize(x / sc, edges) - 1, 0, nb - 1)
    rec = (edges[tok] + edges[tok + 1]) / 2 * sc
    f = go.Figure([go.Scatter(y=x, line=dict(color=GREY, width=1.5), name="원래 시계열"),
                   go.Scatter(y=rec, line=dict(color=PAL[0], width=2, shape="hv"), name=f"토큰으로 복원 ({nb}개 구간)")])
    _lay(f, 260, title="① 평균 절대값으로 나눈 뒤 구간 번호(토큰)로 바꿔 '단어열'처럼 처리", hovermode="x unified")
    c.plotly_chart(f, key="at_ch_tok")
    _fm_fan(c, "Chronos-2", "at_ch")
    c.caption("Chronos는 값을 토큰(단어)으로 바꿔 언어모델(T5) 구조로 '다음 토큰 확률'을 학습 → 여러 번 표본을 뽑아 분위 예측. "
              "Bolt·Chronos-2는 토큰 대신 패치를 직접 넣어 빠르게 분위를 출력. 아래 그림은 이 앱이 실제로 돌린 원자재 예측(캐시).")


def viz_timesfm(c):
    rng = np.random.default_rng(2)
    n = 160
    x = np.sin(np.arange(n) / 9) + 0.004 * np.arange(n) + 0.25 * rng.standard_normal(n)
    pin = c.select_slider("입력 패치 길이", [8, 16, 32], 32, key="at_tf_p")
    f = go.Figure()
    for k in range(n // pin):
        seg = slice(k * pin, (k + 1) * pin)
        f.add_trace(go.Scatter(x=np.arange(n)[seg], y=x[seg], line=dict(color=PAL[k % 2], width=2), name=f"패치 {k + 1}", showlegend=k < 2))
    f.add_vrect(x0=n - 1, x1=n - 1 + 4 * pin, fillcolor="rgba(168,75,47,0.08)", line_width=0, annotation_text="출력 패치 (한 번에 길게)")
    _lay(f, 260, title=f"① 입력을 {pin}개씩 패치로 잘라 토큰화 → 디코더 트랜스포머가 다음 '긴' 패치를 한 번에 예측")
    c.plotly_chart(f, key="at_tf_patch")
    _fm_fan(c, "TimesFM-2.5", "at_tf")
    c.caption("Google TimesFM: 수천억 시점의 실제·합성 시계열로 사전학습한 디코더 전용 모델. 입력 패치보다 긴 출력 패치를 내 긴 예측에서 오차 누적을 줄임. "
              "이 앱 백테스트에서 순위 롱숏 0.45(t 1.7)로 가장 나았지만 우연과 구별되지 않음.")


def viz_patchtst(c):
    rng = np.random.default_rng(3)
    L_ = 52
    r = rng.normal(0, 1, L_)
    p = c.select_slider("패치 길이 (주)", [1, 2, 4, 8, 13], 4, key="at_pt_p")
    k = L_ // p
    f = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35], subplot_titles=("52주 수익률 → 패치", "토큰 수 (어텐션 계산량 ∝ 토큰²)"))
    for j in range(k):
        seg = slice(j * p, (j + 1) * p)
        f.add_trace(go.Bar(x=np.arange(L_)[seg], y=r[seg], marker_color=PAL[j % 2], showlegend=False), 1, 1)
    f.add_trace(go.Bar(x=["점 하나 = 토큰", f"패치 {p}주 = 토큰"], y=[L_, k], marker_color=[GREY, PAL[0]], text=[L_, k], textposition="outside", showlegend=False), 1, 2)
    _lay(f, 340, title=f"PatchTST: {p}주씩 묶어 토큰 {k}개 — 품목마다 따로(채널 독립) 같은 모델 공유")
    c.plotly_chart(f, key="at_pt_fig")
    c.caption("점 하나하나가 아닌 '구간 모양'을 토큰으로 써서 국소 패턴을 보존하고 계산량을 줄임(Nie 외 2023). 채널 독립 = 19개 품목을 각자의 시계열로 보되 가중치는 공유. "
              "정식(neuralforecast)과 자체 경량판의 예측 상관이 0.25에 그쳐 설정에 민감함을 확인.")


def viz_itransformer(c):
    rng = np.random.default_rng(4)
    names = ["옥수수", "밀", "대두", "WTI", "난방유", "가솔린", "금", "은", "구리"]
    F = rng.normal(0, 1, (300, 3))
    X = np.column_stack([F[:, g] * 0.9 + rng.normal(0, 0.6, 300) for g in (0, 0, 0, 1, 1, 1, 2, 2, 0)])
    tau = c.slider("어텐션 온도 (낮을수록 비슷한 변수에 집중)", 0.1, 1.0, 0.3, 0.1, key="at_it_tau")
    Z = (X - X.mean(0)) / X.std(0)
    S = Z.T @ Z / len(Z)
    A_ = np.exp(S / tau); A_ /= A_.sum(1, keepdims=True)
    f = go.Figure(go.Heatmap(z=A_, x=names, y=names, colorscale="Teal", zmin=0, zmax=float(A_.max()), colorbar=dict(title="가중치"),
                             hovertemplate="%{y} → %{x}: %{z:.2f}<extra></extra>"))
    _lay(f, 380, title="iTransformer: 품목 하나의 전체 과거 = 토큰 하나 → 품목 간 어텐션 (예시: 유사도 기반 가중치)", yaxis=dict(autorange="reversed"))
    c.plotly_chart(f, key="at_it_fig")
    c.caption("일반 트랜스포머는 '시점'을 토큰으로 보지만 iTransformer는 축을 뒤집어 '변수'를 토큰으로(Liu 외 2024) → 어텐션이 품목 간 관계(같은 섹터끼리)를 학습. "
              "그림은 상관에 소프트맥스를 씌운 개념 예시이며 실제 가중치는 학습으로 정해짐.")


def viz_tft(c):
    rng = np.random.default_rng(5)
    n = 400
    X = rng.normal(0, 1, (n, 4))
    s = c.slider("잡음 크기", 0.2, 2.0, 0.8, 0.2, key="at_tft_s")
    y = 0.9 * X[:, 0] + 0.4 * X[:, 1] + s * rng.standard_normal(n)
    from sklearn.linear_model import QuantileRegressor
    tr = 300
    imp = np.abs([np.corrcoef(X[:tr, k], y[:tr])[0, 1] for k in range(4)])
    w = np.exp(imp / 0.15); w /= w.sum()
    qs = {}
    for q in (0.1, 0.5, 0.9):
        qs[q] = QuantileRegressor(quantile=q, alpha=0.0, solver="highs").fit(X[:tr], y[:tr]).predict(X[tr:])
    f = make_subplots(rows=1, cols=2, column_widths=[0.35, 0.65], subplot_titles=("변수 선택 가중치 (예시)", "분위 예측 (10·50·90%)"))
    f.add_trace(go.Bar(x=["가격 추세 (강)", "재고 (약)", "달력", "잡음 변수"], y=w, marker_color=[PAL[0], PAL[0], GREY, GREY], showlegend=False), 1, 1)
    t = np.arange(tr, n)
    f.add_trace(go.Scatter(x=t, y=qs[0.9], line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 2)
    f.add_trace(go.Scatter(x=t, y=qs[0.1], fill="tonexty", fillcolor="rgba(32,128,141,0.18)", line=dict(width=0), name="10–90%"), 1, 2)
    f.add_trace(go.Scatter(x=t, y=qs[0.5], line=dict(color=PAL[0], width=1.5), name="중앙값"), 1, 2)
    f.add_trace(go.Scatter(x=t, y=y[tr:], mode="markers", marker=dict(size=3, color=GREY), name="실제"), 1, 2)
    cov = float(np.mean((y[tr:] >= qs[0.1]) & (y[tr:] <= qs[0.9])))
    _lay(f, 360, title=f"TFT: 어떤 변수를 쓸지 고르고(변수 선택) 분위로 예측 — 10–90% 실제 포함률 {cov:.0%}")
    c.plotly_chart(f, key="at_tft_fig")
    c.caption("Temporal Fusion Transformer(Lim 외 2021): 변수 선택 네트워크 → LSTM(국소 순서) → 어텐션(먼 과거) → 분위 손실. 과거 변수·미래에 아는 변수(달력)·정적 변수를 구분해 넣음. "
              "그림은 구조를 보여주는 개념 예시(가중치는 상관 기반, 분위는 선형 분위회귀).")


@st.cache_data(show_spinner=False)
def _node_demo(n_obs: int, seed: int = 0):
    import torch
    from torchdiffeq import odeint
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    A_true = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])
    z0 = torch.tensor([[2.0, 0.0]])
    t_all = torch.linspace(0, 6, 300)
    with torch.no_grad():
        true = odeint(lambda t, z: z @ A_true.T, z0, t_all)[:, 0]
    idx = np.sort(rng.choice(300, n_obs, replace=False)); idx[0] = 0
    t_obs = t_all[idx]; y_obs = true[idx] + 0.08 * torch.randn(n_obs, 2)
    f = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.Tanh(), torch.nn.Linear(32, 2))
    opt = torch.optim.Adam(f.parameters(), 1e-2)
    for _ in range(150):
        pred = odeint(lambda t, z: f(z), y_obs[:1], t_obs, method="rk4", options=dict(step_size=0.1))[:, 0]
        loss = ((pred - y_obs) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        fit = odeint(lambda t, z: f(z), y_obs[:1], t_all, method="rk4", options=dict(step_size=0.05))[:, 0]
    return t_all.numpy(), true.numpy(), t_obs.numpy(), y_obs.numpy(), fit.numpy(), float(loss)


def viz_node(c):
    n = c.slider("관측 개수 (불규칙 간격)", 10, 60, 25, 5, key="at_nd_n")
    t, true, to, yo, fit, loss = _node_demo(n)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], subplot_titles=("시간 축 (첫 번째 상태)", "상태 공간 궤적"))
    fig.add_trace(go.Scatter(x=t, y=true[:, 0], line=dict(color=GREY, width=1.5), name="실제 연속 궤적"), 1, 1)
    fig.add_trace(go.Scatter(x=to, y=yo[:, 0], mode="markers", marker=dict(size=6, color=PAL[1]), name="불규칙 관측"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=fit[:, 0], line=dict(color=PAL[0], width=2), name="Neural ODE 적분 결과"), 1, 1)
    fig.add_trace(go.Scatter(x=true[:, 0], y=true[:, 1], line=dict(color=GREY, width=1.5), showlegend=False), 1, 2)
    fig.add_trace(go.Scatter(x=fit[:, 0], y=fit[:, 1], line=dict(color=PAL[0], width=2), showlegend=False), 1, 2)
    fig.add_trace(go.Scatter(x=yo[:, 0], y=yo[:, 1], mode="markers", marker=dict(size=5, color=PAL[1]), showlegend=False), 1, 2)
    _lay(fig, 380, title=f"Neural ODE: 신경망이 '변화 속도' dz/dt를 배우고 ODE 풀이기로 적분 — 관측 사이도 연속으로 (학습 MSE {loss:.3f})")
    c.plotly_chart(fig, key="at_nd_fig")
    c.caption("실제 torchdiffeq로 학습(감쇠 진동 합성 자료). 관측 간격이 불규칙해도 그대로 처리 가능한 것이 장점(Chen 외 2018). "
              "원자재 주간 자료는 간격이 일정하고 신호가 약해 이 장점이 크지 않음 — 백테스트 순위 롱숏 0.34(t 1.3).")


def viz_gnn(c):
    rng = np.random.default_rng(6)
    names = ["옥수수", "밀", "대두", "대두박", "WTI", "난방유", "가솔린", "천연가스", "금", "은", "백금", "구리"]
    grp = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    F = rng.normal(0, 1, (200, 3))
    X = np.column_stack([F[:, g] * 0.8 + rng.normal(0, 0.7, 200) for g in grp])
    k = c.slider("이웃 수 (상관 상위 k개와 연결)", 1, 5, 3, key="at_gn_k")
    C = np.abs(np.corrcoef(X.T)); np.fill_diagonal(C, 0)
    M = np.zeros_like(C)
    for i in range(len(C)):
        nb = np.argsort(C[i])[-k:]; M[i, nb] = 1
    M = np.maximum(M, M.T)
    ang = 2 * np.pi * np.arange(len(names)) / len(names)
    px_, py_ = np.cos(ang), np.sin(ang)
    sig = np.zeros(len(names)); sig[0] = 1.0                          # 옥수수에 충격
    Ah = M + np.eye(len(M)); Dm = 1 / np.sqrt(Ah.sum(1))
    Ah = Dm[:, None] * Ah * Dm[None, :]
    s1, s2 = Ah @ sig, Ah @ Ah @ sig
    f = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], subplot_titles=("상관 그래프", "옥수수 충격의 전파 (GCN 1·2단계)"))
    for i in range(len(M)):
        for j in range(i + 1, len(M)):
            if M[i, j]:
                f.add_trace(go.Scatter(x=[px_[i], px_[j]], y=[py_[i], py_[j]], mode="lines", line=dict(color=GREY, width=1), showlegend=False, hoverinfo="skip"), 1, 1)
    f.add_trace(go.Scatter(x=px_, y=py_, mode="markers+text", text=names, textposition="top center", marker=dict(size=14, color=[PAL[g] for g in grp]),
                           showlegend=False), 1, 1)
    for nm_, v, col in (("1단계", s1, PAL[0]), ("2단계", s2, PAL[1])):
        f.add_trace(go.Bar(x=names, y=v, name=nm_, marker_color=col), 1, 2)
    f.update_xaxes(visible=False, row=1, col=1); f.update_yaxes(visible=False, row=1, col=1)
    _lay(f, 400, title="시공간 GNN: 품목 = 노드, 상관 = 연결 → 이웃 정보를 섞어(메시지 전달) 각 품목 예측에 반영", barmode="group")
    c.plotly_chart(f, key="at_gn_fig")
    c.caption("GCN 한 단계 = 정규화 인접 행렬 Â로 이웃 값을 가중 평균 → 같은 섹터로 충격이 번짐, 단계가 늘수록 더 멀리. "
              "시간 패턴은 GRU가, 품목 간 관계는 GCN이 담당(T-GCN 계열). 이 앱은 학습 직전 3년 상관 상위 4개로 그래프를 매년 갱신.")


ATLAS = {
    "PCA (주성분 분석)": dict(group="차원 축소 · 동조성", viz=viz_pca,
        what="여러 변수가 함께 움직이는 공통 방향(주성분)을 찾아, 소수의 축으로 데이터를 요약.",
        formula=r"\max_{v} \; v^\top \Sigma v \;\; \text{s.t. } \|v\|=1 \;\Rightarrow\; \Sigma v = \lambda v",
        when="수십 개 품목의 공통 요인(위험선호·달러) 추출, 흡수비율(동조화 지표), 설명변수 압축(PCR·FAVAR).",
        pitfall="방향만 찾고 y는 보지 않음(비지도). 표준화 여부에 결과가 좌우. 주성분의 경제적 해석은 적재값을 보고 사람이 붙여야 함.",
        where="① 시장 구조(PCA·흡수비율), ④ PCR, ① FAVAR 요인"),
    "OLS (최소제곱 회귀)": dict(group="선형 회귀", viz=viz_ols,
        what="잔차 제곱합을 최소화하는 직선(초평면). 모든 회귀 모델의 기준선.",
        formula=r"\hat\beta = \arg\min_\beta \sum_t (y_t - x_t^\top\beta)^2 = (X^\top X)^{-1}X^\top y",
        when="변수가 적고 관측치가 많을 때, 계수 해석(헤지 베타)이 목적일 때. HAC 표준오차로 자기상관·이분산 보정.",
        pitfall="변수가 많으면 과적합(학습 R² ↑, 테스트 R² ↓). 이상치·다중공선성에 민감.",
        where="④ OLS, ② 레짐 회귀의 기본형"),
    "Ridge · Lasso · Elastic Net": dict(group="선형 회귀 (규제)", viz=viz_ridge_lasso,
        what="계수 크기에 벌점을 줘 과적합을 막음. Lasso는 일부 계수를 정확히 0으로 만들어 변수 선택.",
        formula=r"\min_\beta \|y-X\beta\|^2 + \lambda\big[\alpha\|\beta\|_1 + (1-\alpha)\|\beta\|_2^2\big]",
        when="설명변수가 많거나 서로 상관될 때. λ는 시계열 교차검증으로 선택.",
        pitfall="Lasso는 상관된 변수 중 하나만 임의로 남김(→ Elastic Net). 계수가 0쪽으로 편향(→ SCAD·MCP).",
        where="④ Ridge·Lasso·Elastic Net, 포트폴리오 Ridge 신호, 패널 모델"),
    "Adaptive Lasso · SCAD · MCP": dict(group="선형 회귀 (규제)", viz=viz_nonconvex,
        what="Lasso의 '큰 계수도 깎는' 편향을 줄인 변수 선택 — 작은 계수는 0, 큰 계수는 그대로.",
        formula=r"\text{SCAD/MCP: 벌점 } p_\lambda(|\beta|) \text{ 가 } |\beta| \text{ 가 커지면 평평해짐}",
        when="진짜 중요한 변수 몇 개를 정확히 골라내고, 그 계수를 편향 없이 추정하고 싶을 때.",
        pitfall="비볼록 → 해가 여러 개일 수 있음(좌표하강 초기값 의존). γ 선택 필요.",
        where="④ Adaptive Lasso·SCAD·MCP"),
    "PCR · PLS · 3PRF (요인 회귀)": dict(group="차원 축소 회귀", viz=viz_factor,
        what="많은 설명변수를 버리지 않고 소수 요인으로 압축한 뒤 회귀.",
        formula=r"y = \gamma^\top F + \varepsilon,\;\; F = XW \;(\text{PCR: } W=\text{주성분},\; \text{PLS: } \text{cov}(Xw, y)\text{ 최대})",
        when="설명변수가 수십~수백 개(거시·원자재 전체)이고 개별 변수보다 공통 요인이 중요할 때.",
        pitfall="PCR은 y와 무관한 큰 분산 방향을 먼저 잡을 수 있음. 요인 수는 CV로.",
        where="④ PCR·PLS·3PRF"),
    "Kalman 필터 (시변 계수 TVP)": dict(group="레짐 · 상태공간", viz=viz_kalman,
        what="계수(β)가 시간에 따라 조금씩 변한다고 보고, 관측이 올 때마다 '예측 → 오차만큼 보정'으로 추적.",
        formula=r"y_t = x_t^\top\beta_t + \varepsilon_t,\quad \beta_t = \beta_{t-1} + \eta_t,\;\; \eta_t\sim N(0, qI)",
        when="헤지 비율·민감도가 국면마다 서서히 바뀔 때(위기 시 베타 상승). 롤링창의 길이 선택 문제를 피하고 싶을 때.",
        pitfall="q(변화 속도)를 잘못 잡으면 너무 굼뜨거나 잡음을 따라감. 이 앱은 학습구간 우도로 추정.",
        where="② 레짐(Kalman TVP), ④ 모델, TVP-VAR 내부"),
    "Hidden Markov · Markov-switching": dict(group="레짐 · 상태공간", viz=viz_hmm,
        what="보이지 않는 상태(평시/위기)가 확률적으로 바뀌고, 상태마다 수익률 분포나 회귀계수가 다르다고 가정.",
        formula=r"s_t \in \{1..K\},\; P(s_t=j|s_{t-1}=i)=p_{ij},\;\; y_t | s_t \sim N(\mu_{s_t} + x_t^\top\beta_{s_t}, \sigma^2_{s_t})",
        when="지금 위기 레짐일 확률, 레짐별 베타·변동성, 레짐 전환 감지.",
        pitfall="레짐 수 선택, 초기값 의존(EM 국소해), 사후 평활 확률은 미래정보를 씀 → 예측에는 필터 확률만.",
        where="② 레짐 (Markov 2레짐, HMM, MS 회귀), ④ 모델"),
    "TAR · STAR (임계 회귀)": dict(group="레짐 · 비선형", viz=viz_tar,
        what="관측 가능한 변수(VIX 등)가 임계값을 넘으면 회귀계수가 바뀜 — 급격히(TAR) 또는 서서히(STAR).",
        formula=r"y_t = x_t^\top\beta_1 + x_t^\top(\beta_2-\beta_1)\,G(q_t;\gamma,c),\;\; G = \mathbb{1}[q_t>c] \;\text{or}\; \frac{1}{1+e^{-\gamma(q_t-c)}}",
        when="'공포지수가 높을 때 관계가 달라지나?'처럼 레짐 기준이 명확할 때. 해석이 HMM보다 쉬움.",
        pitfall="임계 효과 검정은 표준 분포가 아님 → Hansen 부트스트랩. STAR는 외삽 시 불안정(이 앱은 범위 제한).",
        where="② 레짐, ④ 모델"),
    "GARCH · GJR · EGARCH": dict(group="변동성", viz=viz_garch,
        what="오늘의 변동성이 어제의 충격과 어제의 변동성에 의해 정해짐 → 변동성 군집과 비대칭(레버리지) 포착.",
        formula=r"h_t = \omega + (\alpha + \gamma\,\mathbb{1}[\varepsilon_{t-1}<0])\varepsilon_{t-1}^2 + \beta h_{t-1}",
        when="변동성 예측(위험 관리·포지션 크기), 충격 반감기, 상승/하락 충격의 비대칭 비교.",
        pitfall="α+β가 1에 가까우면 장기 변동성 추정이 불안정. 두꺼운 꼬리는 t분포로.",
        where="③ 변동성 (단변량 GARCH·레버리지), DCC의 1단계"),
    "DCC · ADCC · BEKK (다변량 GARCH)": dict(group="변동성", viz=viz_dcc,
        what="품목 간 상관·공분산이 시간에 따라 변하는 것을 모델링 — 위기 시 동조화 측정.",
        formula=r"Q_t = (1-a-b)\bar Q + a\, z_{t-1}z_{t-1}^\top + b\,Q_{t-1},\;\; R_t = \text{diag}(Q_t)^{-1/2} Q_t\, \text{diag}(Q_t)^{-1/2}",
        when="분산투자 효과가 언제 사라지는지, 동적 헤지 비율, 포트폴리오 위험.",
        pitfall="DCC는 모든 쌍이 같은 동학(a, b)을 공유. BEKK는 모수가 많아 품목 수가 많으면 추정 어려움.",
        where="③ 다변량 GARCH"),
    "VAR · Diebold-Yilmaz · TVP-VAR": dict(group="시스템 · 연결성", viz=viz_var_dy,
        what="여러 시장이 서로의 과거에 반응하는 시스템을 추정하고, 충격이 어디서 어디로 퍼지는지 측정.",
        formula=r"y_t = A_1 y_{t-1} + \dots + A_p y_{t-p} + u_t;\;\; \text{스필오버}_{i\leftarrow j} = \theta^g_{ij}(H)",
        when="'원유 충격이 곡물로 얼마나 전이되나', 위기 시 연결성 급등, 순 전달자/수신자.",
        pitfall="롤링창 길이에 결과가 좌우(→ TVP-VAR). 변수가 많으면 모수 폭증.",
        where="① 시장 구조 (DY 스필오버, TVP-VAR), FAVAR"),
    "나무 모델 (RF · GBM · LightGBM)": dict(group="머신러닝", viz=viz_trees,
        what="X를 구간으로 쪼개 구간 평균으로 예측하는 나무를 여러 개 결합 — 비선형·상호작용 자동 포착.",
        formula=r"\hat f(x) = \sum_{m=1}^{M} \eta\, T_m(x) \;\;(\text{부스팅: } T_m \text{ 은 이전 잔차에 적합})",
        when="비선형·임계·상호작용이 의심될 때, 변수 중요도·SHAP으로 판단 근거를 보고 싶을 때.",
        pitfall="신호가 약한 수익률 예측에서는 잡음을 외우기 쉬움(이 앱 실험: Ridge보다 열위). 외삽 불가(계단 함수).",
        where="④ Random Forest·Gradient Boosting·LightGBM(+SHAP), 패널 모델"),
    "Gaussian Process · Kernel Ridge": dict(group="머신러닝", viz=viz_gp,
        what="'비슷한 입력은 비슷한 출력'을 커널로 정의한 비모수 회귀. GP는 예측 불확실성까지 냄.",
        formula=r"f \sim \mathcal{GP}(0, k),\;\; k(x,x') = \sigma^2 e^{-\|x-x'\|^2/2\ell^2};\;\; \hat f_* = k_*^\top (K+\sigma_n^2 I)^{-1}y",
        when="매끄러운 비선형 관계, 데이터가 적은 영역에서의 불확실성 표시.",
        pitfall="계산량 O(n³) → 관측치 수 제한. 변수가 많으면 거리 개념이 약해짐.",
        where="④ Gaussian Process·Kernel Ridge"),
    "Conformal 예측구간": dict(group="불확실성", viz=viz_conformal,
        what="어떤 모델이든 분포 가정 없이 '90% 확률로 실제값이 들어갈 구간'을 만들고 실제로 보장.",
        formula=r"\hat C(x) = \hat f(x) \pm \hat q_{1-\alpha},\;\; \hat q = \text{보정용 } |y-\hat f(x)| \text{ 의 분위수};\;\; \text{ACI: } \alpha_{t+1} = \alpha_t + \gamma(\alpha - \text{err}_t)",
        when="예측값보다 '얼마나 크게 틀릴 수 있나'가 중요할 때 — 포지션 크기·손절폭·위험 한도.",
        pitfall="교환가능성 가정 → 국면 변화 시 고정 구간은 커버리지 미달(→ 적응형·변동성 정규화).",
        where="④ 단일 품목 → '📏 예측구간'"),
    "MIDAS (혼합 주기)": dict(group="변동성 · 혼합 주기", viz=viz_midas,
        what="저빈도 목표(월간)를 고빈도 데이터(일간)로 예측하며, 수많은 시차를 소수 모수의 가중 함수로 요약.",
        formula=r"y_{m} = \beta_0 + \beta_1 \sum_{k=1}^{K} w(k;\theta_1,\theta_2)\, x_{d-k} + \varepsilon",
        when="월간 변동성·수익을 일간 데이터로, 거시(월간) 정보를 변동성에 반영(GARCH-MIDAS).",
        pitfall="가중 함수 형태 가정. 비선형 최소제곱 → 초기값.",
        where="◆ 리스크 변동성 모형 검증 (MIDAS-RV), ③ GARCH-MIDAS, ④ MIDAS 수익 예측"),
    "Newton-Raphson 포트폴리오 최적화": dict(group="포트폴리오", viz=viz_newton,
        what="예측 수익(μ)·공분산(Σ)·거래비용이 주어졌을 때 효용이 최대인 비중을 수치적으로 찾음. ML이 아님.",
        formula=r"\max_w\; \mu^\top w - \tfrac{\lambda}{2} w^\top\Sigma w - c^\top|w-w_0| \;\; \text{s.t. 제약};\;\; w \leftarrow w - [\nabla^2 F]^{-1}\nabla F",
        when="여러 품목의 비중 결정, 리스크 패리티(같은 Newton으로 위험 기여 균등화).",
        pitfall="μ 추정 오차에 극도로 민감 → 종목 상한·수축 공분산·리스크 패리티로 완화.",
        where="⑤ 포트폴리오"),
    "분위수 회귀 (Quantile)": dict(group="선형 회귀", viz=viz_quantile,
        what="평균 대신 분포의 특정 분위(예: 하위 5% 손실)가 설명변수에 어떻게 반응하는지 추정.",
        formula=r"\hat\beta_\tau = \arg\min_\beta \sum_t \rho_\tau(y_t - x_t^\top\beta),\;\; \rho_\tau(u) = u(\tau - \mathbb{1}[u<0])",
        when="꼬리 위험(VaR) 민감도, 이상치에 강건한 추정, '하락장에서의 베타'.",
        pitfall="극단 분위(τ<0.05)는 표본이 적어 불안정. 분위수 선이 교차할 수 있음.",
        where="④ 분위수 회귀"),
    "Complete Subset 회귀": dict(group="모형 평균 · 앙상블", viz=viz_csr,
        what="변수 k개짜리 모든(또는 무작위) 조합으로 OLS를 돌려 예측을 평균 — 변수를 버리지 않는 축소.",
        formula=r"\hat y = \frac{1}{\binom{p}{k}} \sum_{S:|S|=k} x_S^\top \hat\beta_S",
        when="설명변수가 여럿이고 각각 약한 정보를 가질 때 (거시·원자재 예측의 전형).",
        pitfall="조합 수 폭증 → 무작위 표본. k는 CV로.",
        where="④ Complete Subset"),
    "BMA (베이즈 모형평균)": dict(group="모형 평균 · 앙상블", viz=viz_bma,
        what="가능한 변수 조합(모형)들을 데이터 적합도로 가중 평균 — 모형 선택의 불확실성까지 반영.",
        formula=r"p(y_{new}|D) = \sum_m p(y_{new}|m, D)\, p(m|D),\;\; \text{PIP}_j = \sum_{m \ni j} p(m|D)",
        when="'어떤 변수가 정말 중요한가'를 확률(사후 포함확률)로 답하고 싶을 때.",
        pitfall="사전분포(g-prior, 모형 크기 사전) 선택에 결과가 민감. 변수가 많으면 MC³ 표본추출 필요.",
        where="④ BMA"),
    "Horseshoe (베이즈 축소)": dict(group="선형 회귀 (규제)", viz=viz_horseshoe,
        what="0에 뾰족하고 꼬리가 두꺼운 사전분포로, 작은 계수만 강하게 축소하고 큰 계수는 거의 그대로 둠.",
        formula=r"\beta_j \sim N(0, \lambda_j^2\tau^2),\;\; \lambda_j \sim C^+(0,1),\;\; \tau \sim C^+(0,1)",
        when="변수가 많고 소수만 중요할 때, 계수의 불확실성(신용구간)까지 필요할 때.",
        pitfall="MCMC(Gibbs) 계산 비용. 사후평균은 정확히 0이 되지 않음 → 변수 선택은 구간·P(β>0)로 판단.",
        where="④ Horseshoe"),
    "Minnesota BVAR": dict(group="시스템 · 연결성", viz=viz_minnesota,
        what="VAR 계수에 '먼 시차·타 변수일수록 0에 가깝다'는 사전분포를 줘 짧은 표본에서도 안정 추정.",
        formula=r"\beta_{ij,\ell} \sim N\!\Big(0,\; \big(\tfrac{\lambda\,\theta^{\mathbb{1}[i\neq j]}}{\ell}\big)^2\Big)",
        when="변수·시차가 많은 VAR, 짧은 표본, 거시 예측.",
        pitfall="λ·θ 선택 (이 앱은 λ를 CV). 수익률처럼 자기상관이 약하면 계수 대부분이 0 근처로.",
        where="④ Minnesota BVAR"),
    "Stacking (가중 앙상블)": dict(group="모형 평균 · 앙상블", viz=viz_stacking,
        what="여러 모델의 외표본 예측을 입력으로 '최적 결합 비중'을 학습.",
        formula=r"\hat y = \sum_m w_m \hat y_m,\;\; w = \arg\min_{w\ge 0, \sum w=1} \sum_t \big(y_t - \textstyle\sum_m w_m \hat y_{m,t}^{OOF}\big)^2",
        when="성격이 다른 모델(선형·나무·커널)을 섞어 안정성을 높이고 싶을 때.",
        pitfall="가중치 학습 표본이 짧으면 단순 평균보다 못할 수 있음 (forecast combination puzzle).",
        where="④ Stacking, 앙상블(단순평균)"),
    "LSTM (순환 신경망)": dict(group="머신러닝", viz=viz_lstm,
        what="과거 창의 순서를 게이트가 있는 기억 셀로 요약해 다음 값을 예측하는 딥러닝 시계열 모델.",
        formula=r"c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c[h_{t-1}, x_t]),\;\; h_t = o_t \odot \tanh(c_t)",
        when="규칙적이고 복잡한 시간 패턴, 대량 데이터 (고빈도, 수요 예측 등).",
        pitfall="표본이 적고 신호가 약한 수익률에서는 과적합이 쉬움. 초기값·하이퍼파라미터 민감.",
        where="④ LSTM"),
    "FAVAR (요인 보강 VAR)": dict(group="시스템 · 연결성", viz=viz_favar,
        what="수십 개 변수에서 뽑은 공통요인과 관심 변수를 함께 VAR로 묶어 충격의 전파를 분석.",
        formula=r"\begin{bmatrix}F_t\\Y_t\end{bmatrix} = \Phi(L)\begin{bmatrix}F_{t-1}\\Y_{t-1}\end{bmatrix} + u_t,\;\; X_t = \Lambda F_t + e_t",
        when="'글로벌 공통 충격(위험선호·달러)이 옥수수에 어떻게 퍼지나', 정책·거시 충격 분석.",
        pitfall="요인의 경제적 의미는 적재값으로 해석해야 함. Cholesky 순서 가정.",
        where="① 시장 구조 → FAVAR"),
    "패널: Pooled · RE · FE · Fama-MacBeth": dict(group="패널 · 횡단면", viz=viz_panel,
        what="여러 품목 × 여러 시점 데이터에서 '공통 기울기'를 추정. 품목 고유효과를 어떻게 다루느냐가 핵심.",
        formula=r"y_{it} = \beta x_{it} + \alpha_i + \lambda_t + \varepsilon_{it};\;\; \text{FE: } (y_{it}-\bar y_i) = \beta(x_{it}-\bar x_i) + \dots",
        when="'캐리가 높은 품목이 더 오르나'처럼 품목 공통 법칙, 어느 품목이 상대적으로 오를지(횡단면 예측).",
        pitfall="αᵢ가 X와 상관되면 Pooled·RE는 편향 → FE (하우스만 검정). 시점 간 상관 → 군집·Driscoll-Kraay 표준오차.",
        where="④ 여러 품목(패널) — 계량 추정·walk-forward"),
    "HAR-RV": dict(group="변동성", viz=viz_har,
        what="일간·주간·월간 평균 실현분산 3개로 다음 기간 변동성을 예측하는 단순 선형 모형.",
        formula=r"RV_{t+1}^{(h)} = c + \beta_d RV_t^{(d)} + \beta_w RV_t^{(w)} + \beta_m RV_t^{(m)} + \varepsilon",
        when="빠르고 해석 쉬운 변동성 예측 기준선.",
        pitfall="선형이라 음수·0 근처 예측 가능 → 하한 필요(이 앱 적용).",
        where="◆ 리스크 변동성 모형 검증, ⑤ 변동성 사이징"),
    "Ledoit-Wolf 수축 공분산": dict(group="포트폴리오", viz=viz_lw,
        what="표본 공분산을 단순한 목표 행렬 쪽으로 최적 비율만큼 당겨 추정 오차를 줄임.",
        formula=r"\hat\Sigma_{LW} = (1-\delta)\, S + \delta\, \mu I,\;\; \delta^* = \arg\min E\|\hat\Sigma - \Sigma\|_F^2",
        when="품목 수에 비해 관측치가 적을 때, 최적화 입력 공분산.",
        pitfall="구조적 상관(섹터)을 일부 지움. 시변 공분산은 별도(DCC·EWMA).",
        where="⑤ 포트폴리오 공분산"),
    "리스크 패리티 vs 평균-분산": dict(group="포트폴리오", viz=viz_rp,
        what="리스크 패리티는 각 자산의 '위험 기여'를 같게, 평균-분산은 예측 수익까지 써서 효용 최대. 둘 다 Newton으로 풂.",
        formula=r"\text{RP: } w_i (\Sigma w)_i = \tfrac{1}{N} w^\top\Sigma w \;\; | \;\; \text{MV: } \max_w \mu^\top w - \tfrac{\lambda}{2} w^\top \Sigma w",
        when="RP: 예측력이 약하거나 불확실할 때(원자재). MV: 신뢰할 만한 μ 예측이 있을 때.",
        pitfall="RP는 저변동 자산 비중이 커져 레버리지 없이 수익이 낮을 수 있음. MV는 μ 오차 증폭기.",
        where="⑤ 포트폴리오 (최적화 Newton · 리스크 패리티 Newton · 1/N)"),
    "무거래 구간 · 변동성 타기팅": dict(group="포트폴리오", viz=viz_band,
        what="목표와의 차이가 일정 폭을 넘을 때만 거래하고, 예측 변동성에 맞춰 전체 노출을 조절.",
        formula=r"\text{거래} \iff \textstyle\sum|w^*_t - w_t| > b \;\text{or}\; t - t_{last} \ge M;\;\; \text{배율} = \min(L, \sigma^*/\hat\sigma_t)",
        when="신호가 매일 흔들려 잦은 매매가 생길 때, 위험 한도를 일정하게 운용할 때.",
        pitfall="폭이 너무 크면 신호를 늦게 따라감. 변동성 타기팅은 폭락 후 반등을 놓칠 수 있음(원자재 롱온리에서 확인).",
        where="⑤ 포트폴리오 → 거래 규칙 · 변동성 예측"),
    "Walk-forward · 시계열 교차검증": dict(group="검증 방법론", viz=viz_cv,
        what="항상 과거로 학습하고 미래로 평가 — 실전 운용과 같은 조건의 성과 검증.",
        formula=r"\hat f_k = \text{fit}(\{(x_s, y_s)\}_{s \le t_k - h}),\;\; \text{평가: } y_{t_k+1..t_k+step}",
        when="모든 예측 모델의 하이퍼파라미터 선택(CV)과 최종 성과 평가.",
        pitfall="무작위 K-fold는 미래정보 누수. 겹치는 목표(h>1)는 gap/purge 필요.",
        where="④ walk-forward, 모든 CV (학습구간 내부)"),
    "예측력 검정 (OOS R² · Clark-West)": dict(group="검증 방법론", viz=viz_tests,
        what="모델이 '과거 평균'이라는 무정보 기준보다 실제로 나은지 통계적으로 판단.",
        formula=r"R^2_{OOS} = 1 - \frac{\sum (y_t-\hat y_t)^2}{\sum (y_t-\bar y_{t-1})^2},\;\; f_t^{CW} = e_{b,t}^2 - (e_{m,t}^2 - (\hat y_{b,t}-\hat y_{m,t})^2)",
        when="모델 비교, 새 정보(뉴스·기관 리포트)의 추가 가치 검증.",
        pitfall="표본이 짧으면 검정력이 낮음 — 작은 R²는 수백 기간이 필요. 여러 모델을 비교하면 다중검정 주의.",
        where="④ 성과 지표, 기관 리포트 검증"),
    "Prophet (추세 변곡 + 계절 분해)": dict(group="대안 시계열 · 분해", viz=viz_prophet,
        what="시계열을 '구간별 선형 추세(변곡점) + 푸리에 계절 + 휴일'로 분해해 각 성분을 연장하는 Meta의 예측 도구.",
        formula=r"y(t) = g(t) + s(t) + h(t) + \varepsilon_t,\;\; g(t) = (k + a(t)^\top\delta)\,t + (m + a(t)^\top\gamma),\;\; \delta_j \sim \text{Laplace}(0, \tau)",
        when="뚜렷한 계절·추세가 있는 업무 지표(수요·트래픽), 해석 가능한 분해가 필요할 때.",
        pitfall="마지막 추세를 그대로 연장 → 추세 반전이 잦은 가격 수익률 예측엔 약함. τ(유연성)에 결과가 크게 좌우.",
        where="⑤ 포트폴리오 › 대안 시계열 모델 (정식 prophet · 자체 Lasso 구현)"),
    "Chronos (Amazon 파운데이션)": dict(group="대안 시계열 · 파운데이션", viz=viz_chronos,
        what="값을 토큰으로 바꿔 언어모델 구조로 사전학습한 시계열 파운데이션 모델 — 학습 없이(zero-shot) 분위 예측.",
        formula=r"\tilde x_t = x_t \Big/ \tfrac{1}{C}\textstyle\sum_i |x_i|,\;\; z_t = \text{bin}(\tilde x_t) \in \{1..4096\},\;\; p(z_{t+1} \mid z_{\le t}) = \text{T5}(z_{\le t})",
        when="자료가 짧거나 품목이 많아 개별 모델을 만들기 어려울 때, 빠른 기준선.",
        pitfall="사전학습 자료(수요·기상 등)와 성격이 다른 금융 수익률에선 우위가 불분명. 큰 모델은 CPU에서 느림.",
        where="⑤ 포트폴리오 › 대안 시계열 모델 (Chronos-Bolt-small · Chronos-2), ⑥ 최신 시계열 모델"),
    "TimesFM (Google 파운데이션)": dict(group="대안 시계열 · 파운데이션", viz=viz_timesfm,
        what="입력을 패치로 잘라 디코더 트랜스포머로 다음 긴 패치를 예측하는 Google의 사전학습 시계열 모델.",
        formula=r"\text{token}_j = \text{MLP}(x_{(j-1)p+1:jp}),\;\; \hat x_{\text{다음 } P} = \text{Decoder}(\text{token}_{1..J}),\;\; P > p",
        when="긴 지평 예측, 학습 없이 많은 시계열에 일괄 적용.",
        pitfall="zero-shot이라 원자재 고유 정보(재고·커브)를 못 씀. 이 앱에서 IC 0.024로 가장 높았지만 유의하지 않음.",
        where="⑤ 포트폴리오 › 대안 시계열 모델, ⑥ 최신 시계열 모델"),
    "PatchTST (패치 트랜스포머)": dict(group="대안 시계열 · 딥러닝", viz=viz_patchtst,
        what="시계열을 패치(구간)로 잘라 토큰으로 쓰고, 변수마다 독립적으로 같은 트랜스포머를 적용.",
        formula=r"P_j = x_{(j-1)S+1 : (j-1)S+p},\;\; z = \text{Transformer}(W P_j + e_j),\;\; \hat y = \text{Linear}(\text{flatten}(z))",
        when="긴 입력 창, 변수 많은 장기 예측 벤치마크에서 강함.",
        pitfall="신호가 약하고 표본이 적은 주간 수익률에선 과적합·설정 민감(정식 vs 경량 예측 상관 0.25).",
        where="⑤ 포트폴리오 › 대안 시계열 모델 (neuralforecast · 자체 경량)"),
    "iTransformer (변수 = 토큰)": dict(group="대안 시계열 · 딥러닝", viz=viz_itransformer,
        what="축을 뒤집어 '변수 하나의 전체 과거'를 토큰으로 → 어텐션이 변수 간 관계를 학습.",
        formula=r"h_n = \text{MLP}(x_{n,1:L}),\;\; H' = \text{softmax}\big(QK^\top/\sqrt d\big)V \;\;(\text{변수 간}),\;\; \hat y_n = \text{MLP}(h'_n)",
        when="변수끼리 강하게 얽힌 다변량 예측(교통·전력·여러 품목).",
        pitfall="변수 간 관계가 시기마다 바뀌면 학습한 어텐션이 낡음. 이 앱에서 정식·경량 결과 부호가 엇갈림.",
        where="⑤ 포트폴리오 › 대안 시계열 모델"),
    "TFT (Temporal Fusion Transformer)": dict(group="대안 시계열 · 딥러닝", viz=viz_tft,
        what="변수 선택 → LSTM → 어텐션 → 분위 출력으로, 여러 종류의 입력을 해석 가능하게 섞는 다중 지평 예측 모델.",
        formula=r"\xi_t = \textstyle\sum_j v_{j,t}\,\tilde\xi_{j,t},\;\; v_t = \text{softmax}(\text{GRN}(\Xi_t)),\;\; \mathcal L = \textstyle\sum_q \rho_q(y - \hat y_q)",
        when="과거 변수 · 미래에 아는 변수(달력·정책 일정) · 정적 변수(품목 특성)를 함께 쓸 때, 변수 중요도 해석이 필요할 때.",
        pitfall="구조가 복잡해 표본이 적으면 과적합. 이 앱에선 달력 변수만 넣어 우위 없음(IC ≈ 0).",
        where="⑤ 포트폴리오 › 대안 시계열 모델"),
    "Neural ODE (연속 시간 신경망)": dict(group="대안 시계열 · 딥러닝", viz=viz_node,
        what="신경망으로 상태의 변화 속도 dz/dt를 표현하고 ODE 풀이기로 적분해 연속 시간 궤적을 얻는 모델.",
        formula=r"\frac{dz(t)}{dt} = f_\theta(z(t), t),\;\; z(t_1) = z(t_0) + \int_{t_0}^{t_1} f_\theta(z, t)\,dt",
        when="관측 간격이 불규칙한 자료(의료·센서), 물리적 연속 과정.",
        pitfall="적분 때문에 학습이 느림(이 앱에서 가장 오래 걸림). 간격이 일정한 주간 자료에선 장점이 작음.",
        where="⑤ 포트폴리오 › 대안 시계열 모델 (torchdiffeq dopri5 ODE-RNN · 자체 RK4)"),
    "시공간 GNN (그래프 신경망)": dict(group="대안 시계열 · 딥러닝", viz=viz_gnn,
        what="품목을 노드, 상관을 연결선으로 두고 이웃 정보를 섞는 그래프 합성곱 + 시간 순환망으로 예측.",
        formula=r"H^{(l+1)} = \sigma\big(\hat D^{-1/2}\hat A \hat D^{-1/2} H^{(l)} W^{(l)}\big),\;\; \hat A = A + I,\;\; H^{(0)} = \text{GRU}(x_{n,1:L})",
        when="연결 구조가 뚜렷한 다변량(도로망·공급망·섹터).",
        pitfall="그래프(상관)가 바뀌면 성능 저하, 그래프 구성 방식에 결과가 좌우(정식 vs 경량 예측 상관 0.09).",
        where="⑤ 포트폴리오 › 대안 시계열 모델 (torch_geometric GCN + GRU · 자체 경량)"),
}


def render(*_):
    st.subheader("모델 도감 — 각 모델은 무엇을 하고, 어떻게 작동하나")
    st.caption(f"원리가 잘 보이도록 만든 합성 데이터에 실제 모델을 돌린 그림입니다. 슬라이더로 잡음·파라미터를 바꿔 보세요. (총 {len(ATLAS)}개)")
    groups = {}
    for k, v in ATLAS.items():
        groups.setdefault(v["group"], []).append(k)
    names = [k for g in groups.values() for k in g]
    sel = st.selectbox("모델", names, format_func=lambda k: f"{ATLAS[k]['group']} · {k}", key="at_sel")
    m = ATLAS[sel]
    left, right = st.columns([1, 1.6])
    with left:
        st.markdown(f"#### {sel}")
        st.markdown(f"**한 줄 요약** — {m['what']}")
        st.latex(m["formula"])
        st.markdown(f"**언제 쓰나** — {m['when']}")
        st.markdown(f"**함정** — {m['pitfall']}")
        st.markdown(f"**이 앱의 어디에** — {m['where']}")
    with right:
        m["viz"](st.container())
