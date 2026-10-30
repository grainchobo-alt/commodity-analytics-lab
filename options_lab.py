"""옵션 랩 — 블랙숄즈 원리 · 금·은·WTI 원유 내재변동성 곡선 · 다른 원자재 추가 방법 (기존 대시보드와 별도 실행).

실행:  .venv\\Scripts\\streamlit run options_lab.py   (또는 옵션랩_열기.bat, 포트 8504)
계산은 options_bs.py, 시세 원천 등록은 options_bs.SOURCES.
"""
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import options_bs as O

st.set_page_config(page_title="옵션 랩 — 블랙숄즈", layout="wide", page_icon="📐")
G, GOLD, RED, BLUE = "#1E3A2F", "#C8A24A", "#D9582B", "#2F6E8F"
SCENE = dict(camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)), aspectmode="manual", aspectratio=dict(x=1.3, y=1.1, z=0.75))


def surf_fig(x, y, z, xt, yt, zt, title, extra=(), cs="Viridis", h=520):
    fig = go.Figure([go.Surface(x=x, y=y, z=z, colorscale=cs, opacity=0.93, colorbar=dict(len=0.6, thickness=12)), *extra])
    fig.update_layout(title=title, height=h, margin=dict(l=0, r=0, t=40, b=0),
                      scene=dict(xaxis_title=xt, yaxis_title=yt, zaxis_title=zt, **SCENE))
    return fig


st.title("📐 옵션 랩 — 블랙숄즈와 원자재 내재변동성")
st.caption("기존 대시보드와 따로 실행되는 실험용 화면. 시세 = 공개 ETF 옵션(yfinance, 미국 장 기준 · 지연 시세). 투자 권유 아님.")
t1, t2, t3 = st.tabs(["① 블랙숄즈 원리 (수식 · 3D)", "② 금 · 은 · WTI 원유 내재변동성 곡선", "③ 다른 원자재 추가하는 방법"])

# ════════════════ ① 원리
with t1:
    st.markdown("### 한 줄 요약\n옵션 가격 = **만기에 받을 돈의 평균**을 오늘 가치로 할인한 것. 블랙숄즈는 그 평균을 계산하는 공식이고, "
                "모르는 값은 **변동성 σ 하나뿐**임 → 그래서 시장 옵션 가격을 거꾸로 풀면 '시장이 보는 변동성(내재변동성)'이 나옴.")
    c = st.columns(5)
    S0 = c[0].number_input("현재 가격 S", 10.0, 1000.0, 100.0, 5.0)
    K = c[1].number_input("행사가 K", 10.0, 1000.0, 100.0, 5.0)
    sig = c[2].slider("변동성 σ (연, %)", 5, 100, 25) / 100
    r = c[3].slider("무위험 금리 r (연, %)", 0.0, 10.0, 4.0, 0.25) / 100
    T = c[4].slider("만기까지 T (년)", 0.05, 2.0, 0.5, 0.05)
    call, put = float(O.bs_price(S0, K, T, r, sig, True)), float(O.bs_price(S0, K, T, r, sig, False))
    d1, d2 = map(float, O.d1_d2(S0, K, T, r, sig))
    m = st.columns(5)
    m[0].metric("콜 가격", f"{call:,.2f}"); m[1].metric("풋 가격", f"{put:,.2f}")
    m[2].metric("d1 · d2", f"{float(d1):.2f} · {float(d2):.2f}")
    m[3].metric("N(d2) = 만기에 K 넘을 확률", f"{O.norm.cdf(d2):.0%}"); m[4].metric("N(d1) = 델타(헤지 비율)", f"{O.norm.cdf(d1):.2f}")

    st.markdown("---\n### 1단계 — 가정: 가격은 '퍼센트'로 무작위하게 움직임 (기하 브라운 운동)")
    a, b = st.columns([1, 1.25])
    with a:
        st.latex(r"dS_t = \mu S_t\,dt + \sigma S_t\,dW_t")
        st.markdown("- 매 순간 가격은 **평균 μ만큼 오르고**, 그 위에 **σ 크기의 무작위 충격(dW)** 이 더해짐.\n"
                    "- 충격이 '금액'이 아니라 '비율'이라 가격은 0 아래로 안 내려가고, 만기 가격은 **로그정규분포**(오른쪽으로 꼬리가 긴 종 모양).")
        st.latex(r"\ln S_T \sim \mathcal{N}\!\left(\ln S_0 + (\mu - \tfrac{1}{2}\sigma^2)T,\; \sigma^2 T\right)")
        st.markdown("- 오른쪽 3D: 시간이 갈수록(앞→뒤) 가능한 가격의 분포가 **넓게 퍼짐** — 폭은 σ√T에 비례. 빨간 벽 = 행사가 K.")
    with b:
        Ts = np.linspace(0.05, max(T, 0.1) * 1.5, 30)
        Sg = np.linspace(S0 * 0.3, S0 * 2.2, 90)
        Z = np.array([O.norm.pdf((np.log(Sg / S0) - (r - sig ** 2 / 2) * t) / (sig * np.sqrt(t))) / (Sg * sig * np.sqrt(t)) for t in Ts])
        wall = go.Surface(x=[K, K], y=[Ts[0], Ts[-1]], z=[[0, 0], [Z.max(), Z.max()]], showscale=False, opacity=0.35,
                          colorscale=[[0, RED], [1, RED]], surfacecolor=[[0, 0], [0, 0]], hoverinfo="skip")
        st.plotly_chart(surf_fig(Sg, Ts, Z, "만기 가격 S_T", "만기까지 T (년)", "확률 밀도", "만기 가격의 분포 — 시간이 갈수록 퍼짐", [wall]),
                        width="stretch")

    st.markdown("---\n### 2단계 — 핵심 아이디어: 위험을 지우면 평균 수익률 μ가 사라짐 (위험중립 가격)")
    a, b = st.columns([1, 1])
    with a:
        st.markdown("- 옵션 1개를 팔고 기초자산을 **Δ개** 사 두면, 짧은 순간 가격이 오르든 내리든 손익이 서로 상쇄됨(**델타 헤지**).\n"
                    "- 위험이 없는 묶음은 은행 이자 **r** 만큼만 벌어야 함(더 벌면 공짜 돈 → 누군가 바로 차익거래).\n"
                    "- 이 조건을 식으로 쓰면 아래 **블랙숄즈 편미분방정식** — 식 안에 **μ가 없음**. 사람마다 다른 '기대 수익'은 옵션 값에 영향이 없고, σ만 남음.")
        st.latex(r"\frac{\partial V}{\partial t} + \tfrac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0")
        st.markdown("- 그래서 옵션 가격은 '모두가 위험에 무관심한 세상(μ = r)'에서 계산한 **만기 손익의 평균을 할인**한 값과 같음:")
        st.latex(r"C = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}\!\left[\max(S_T - K,\,0)\right]")
    with b:
        rng = np.random.default_rng(7)
        n, steps = 60, 120
        dt = T / steps
        paths = S0 * np.exp(np.cumsum((r - sig ** 2 / 2) * dt + sig * np.sqrt(dt) * rng.standard_normal((n, steps)), axis=1))
        paths = np.hstack([np.full((n, 1), S0), paths])
        tt = np.linspace(0, T, steps + 1)
        fig = go.Figure()
        for i, p in enumerate(paths):
            fig.add_trace(go.Scatter3d(x=tt, y=np.full_like(tt, i), z=p, mode="lines", showlegend=False, hoverinfo="skip",
                                       line=dict(width=2, color=GOLD if p[-1] > K else "#9AA59E")))
        fig.add_trace(go.Surface(x=[0, T], y=[0, n], z=[[K, K], [K, K]], showscale=False, opacity=0.25, colorscale=[[0, RED], [1, RED]], hoverinfo="skip"))
        hit = (paths[:, -1] > K).mean()
        fig.update_layout(title=f"위험중립 세상의 가격 경로 {n}개 — 금색 = 만기에 K 위({hit:.0%}, 이론 N(d2) {O.norm.cdf(d2):.0%})", height=470,
                          margin=dict(l=0, r=0, t=40, b=0), scene=dict(xaxis_title="시간 (년)", yaxis_title="경로 번호", zaxis_title="가격", **SCENE))
        st.plotly_chart(fig, width="stretch")

    st.markdown("---\n### 3단계 — 평균을 직접 계산하면 블랙숄즈 공식")
    a, b = st.columns([1, 1.25])
    with a:
        st.latex(r"C = S_0\,N(d_1) - K e^{-rT} N(d_2)")
        st.latex(r"P = K e^{-rT} N(-d_2) - S_0\,N(-d_1)")
        st.latex(r"d_1 = \frac{\ln(S_0/K) + (r + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}},\qquad d_2 = d_1 - \sigma\sqrt{T}")
        st.markdown(f"""
- **N(·)** = 표준정규분포 누적확률 (종 모양에서 왼쪽 넓이).
- **N(d2)** = 위험중립 세상에서 만기에 **K보다 비쌀 확률** → K e^(−rT)N(d2) = '행사가를 낼 확률 × 할인된 K'. (지금 {O.norm.cdf(d2):.0%})
- **S0·N(d1)** = K를 넘는 경우에만 받는 **기초자산 가치의 평균**. N(d1)은 곧 **델타**(헤지에 필요한 기초자산 개수). (지금 {O.norm.cdf(d1):.2f})
- **콜 − 풋 = S0 − K e^(−rT)** (풋-콜 패리티): 지금 {call:.2f} − {put:.2f} = {call - put:.2f}, {S0:.0f} − {K * np.exp(-r * T):.2f} = {S0 - K * np.exp(-r * T):.2f}.
- **선물옵션(Black-76)**: S0 자리에 F·e^(−rT)(선물가격의 현재가치). 곡물·원유 선물옵션 자료를 구하면 이 식을 씀.""")
    with b:
        Sg = np.linspace(S0 * 0.5, S0 * 1.6, 60)
        Tg = np.linspace(0.02, 2.0, 40)
        SS, TT = np.meshgrid(Sg, Tg)
        CP = O.bs_price(SS, K, TT, r, sig, True)
        payoff = go.Scatter3d(x=Sg, y=np.zeros_like(Sg), z=np.maximum(Sg - K, 0), mode="lines", line=dict(color=RED, width=6), name="만기 손익 max(S−K, 0)")
        st.plotly_chart(surf_fig(Sg, Tg, CP, "현재 가격 S", "만기까지 T (년)", "콜 가격", "콜 가격 곡면 — 만기가 다가올수록(T→0) 빨간 '하키 스틱'에 붙음",
                                 [payoff], cs="Cividis"), width="stretch")

    st.markdown("---\n### 4단계 — 민감도(그릭스): 무엇이 바뀌면 옵션 값이 얼마나 바뀌나")
    gname = st.radio("민감도", ["델타", "감마", "베가", "세타"], horizontal=True,
                     captions=["가격 1 변화당 옵션 변화", "델타가 얼마나 빨리 바뀌나", "변동성 1%p당 옵션 변화", "하루 지날 때 옵션 변화"])
    GK = O.greeks(SS, K, TT, r, sig, True)[gname]
    st.plotly_chart(surf_fig(Sg, Tg, GK, "현재 가격 S", "만기까지 T (년)", gname, f"콜 {gname} 곡면 — 행사가 근처 · 만기 직전에 가장 민감", cs="RdBu_r", h=480),
                    width="stretch")

    st.markdown("---\n### 5단계 — 거꾸로 풀기: 시장 가격 → 내재변동성")
    a, b = st.columns([1, 1.2])
    with a:
        mkt = st.number_input("시장에서 본 콜 가격", 0.01, float(S0), float(round(call, 2)), 0.5)
        iv = O.implied_vol(mkt, S0, K, T, r, True)
        st.latex(r"\text{찾기: } \sigma_{\text{impl}} \text{ such that } C_{BS}(\sigma_{\text{impl}}) = C_{\text{market}}")
        st.metric("내재변동성", f"{iv:.1%}" if iv == iv else "계산 불가(내재가치 이하)")
        st.markdown("- 콜 가격은 σ가 클수록 단조 증가 → 맞는 σ가 딱 하나. 숫자 탐색(브렌트 방법)으로 찾음.\n"
                    "- 블랙숄즈가 완벽하다면 **모든 행사가에서 같은 σ**가 나와야 함. 실제로는 끝으로 갈수록 높아지는 **스마일**이 나옴 → ② 탭.")
    with b:
        sg = np.linspace(0.02, 1.0, 200)
        fig = go.Figure([go.Scatter(x=sg * 100, y=O.bs_price(S0, K, T, r, sg, True), line=dict(color=G, width=3), name="콜 가격(σ)")])
        fig.add_hline(y=mkt, line=dict(color=RED, dash="dash"), annotation_text="시장 가격")
        if iv == iv:
            fig.add_vline(x=iv * 100, line=dict(color=GOLD, dash="dot"), annotation_text=f"σ = {iv:.1%}")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="변동성 σ (%)", yaxis_title="콜 가격")
        st.plotly_chart(fig, width="stretch")
    with st.expander("블랙숄즈의 한계 — 원자재에서 특히 주의"):
        st.markdown("- **변동성 일정 가정** ↔ 실제는 군집·급변(GARCH) → 시장은 끝 행사가에 웃돈 → 스마일·스큐.\n"
                    "- **가격 점프 없음 가정** ↔ 산지 날씨·OPEC·전쟁 뉴스로 급등락.\n"
                    "- **유럽형(만기에만 행사)** ↔ 미국 ETF·선물옵션 다수는 미국형(언제든 행사) → 외가격 옵션만 쓰면 차이 작음.\n"
                    "- **보유비용·편의수익** ↔ 현물 원자재는 보관비·편의수익이 있어 선물가격 기준 Black-76이 더 자연스러움.")

# ════════════════ ② 시장 곡선
with t2:
    st.markdown("시장 옵션 가격을 1단계~5단계 공식으로 거꾸로 풀어 **행사가 × 만기별 내재변동성**을 구함. "
                "블랙숄즈가 완벽하면 평평한 판이어야 하지만, 실제 모양(스마일·기울기·만기 구조)이 **시장이 걱정하는 위험**을 보여 줌.")
    c = st.columns([1.3, 1, 1, 1])
    name = c[0].selectbox("품목", list(O.SOURCES))
    r2 = c[1].slider("무위험 금리 (%)", 0.0, 8.0, 4.0, 0.25, key="r2") / 100
    max_spread = c[2].slider("호가 차이 상한 (%)", 5, 100, 25, 5) / 100
    min_oi = c[3].number_input("최소 미결제약정", 0, 10000, 10, 10)

    @st.cache_data(ttl=3600, show_spinner="옵션 시세 받는 중…")
    def _chain(nm):
        return O.load_chain(nm)

    try:
        ch = _chain(name)
    except Exception as e:                                                 # 네트워크·원천 오류
        st.error(f"시세를 받지 못함: {e}"); ch = pd.DataFrame(columns=list(O.CSV_COLS))
    model = O.SOURCES[name]["model"]
    pts = O.iv_points(ch, r2, model, max_spread, min_oi)
    fits = O.smile_fit(pts)

    def show_market(ch, pts, fits, label, key):
        if pts.empty or not fits:
            st.warning("곡선을 그릴 만큼 유효한 호가가 없음 — 호가 차이 상한을 넓히거나 미결제약정 기준을 낮춰 보세요."); return
        if pts.attrs.get("note"):
            st.warning(pts.attrs["note"])
        S = float(ch["underlying"].iloc[0]); asof = pd.Timestamp(ch["asof"].iloc[0])
        exps = sorted(fits); near = min(exps, key=lambda e: abs((e - asof).days - 45))
        a0, b0, c0, n0 = fits[near]
        k = st.columns(5)
        k[0].metric("기초자산 가격", f"{S:,.2f}", help=f"시세 날짜 {asof:%Y-%m-%d}")
        k[1].metric(f"등가격 변동성 ({(near - asof).days}일)", f"{a0:.1%}")
        k[2].metric("기울기 (skew)", f"{b0:+.2f}", help="+ = 높은 행사가(급등 대비 콜)가 더 비쌈 · − = 낮은 행사가(급락 대비 풋)가 더 비쌈")
        k[3].metric("휘어짐 (smile)", f"{c0:.2f}", help="클수록 양 끝(큰 변동) 옵션이 블랙숄즈 평평한 판보다 비쌈")
        k[4].metric("사용한 옵션 수", f"{len(pts)} / 만기 {len(exps)}개")
        a, b = st.columns([1, 1.2])
        with a:
            fig = go.Figure()
            pal = ["#1E3A2F", "#2F6E8F", "#C8A24A", "#D9582B", "#6B8E6B", "#8C6BB1", "#9A9A94", "#2E7D52"]
            mm = np.linspace(pts["m"].min(), pts["m"].max(), 60)
            for i, e in enumerate(exps):
                g = pts[pts["expiry"] == e]; col = pal[i % len(pal)]; a_, b_, c_, _ = fits[e]
                lab = f"{e:%Y-%m-%d} ({(e - asof).days}일)"
                fig.add_trace(go.Scatter(x=g["m"] * 100, y=g["iv"] * 100, mode="markers", marker=dict(size=5, color=col), name=lab, legendgroup=lab))
                fig.add_trace(go.Scatter(x=mm * 100, y=(a_ + b_ * mm + c_ * mm ** 2) * 100, mode="lines", line=dict(color=col), showlegend=False, legendgroup=lab))
            fig.add_hline(y=a0 * 100, line=dict(color="#999", dash="dot"), annotation_text="블랙숄즈라면 이 높이로 평평")
            fig.update_layout(title="만기별 변동성 스마일 (점 = 시장, 선 = 2차 곡선)", height=480, margin=dict(l=10, r=10, t=40, b=10),
                              xaxis_title="행사가 위치 ln(K/S) (%)  ← 낮은 행사가 · 높은 행사가 →", yaxis_title="내재변동성 (%)")
            st.plotly_chart(fig, width="stretch", key=f"sm{key}")
        with b:
            Tn = np.array([(e - asof).days / 365 for e in exps])
            mg = np.linspace(max(pts["m"].min(), -0.3), min(pts["m"].max(), 0.3), 40)
            rng_ = {e: (pts.loc[pts["expiry"] == e, "m"].min(), pts.loc[pts["expiry"] == e, "m"].max()) for e in exps}
            Zs = np.array([[fits[e][0] + fits[e][1] * x + fits[e][2] * x * x if rng_[e][0] <= x <= rng_[e][1] else np.nan for x in mg]
                           for e in exps]) * 100                       # 자료가 있는 행사가 범위만(밖으로 늘리면 곡선이 튐)
            sc = go.Scatter3d(x=pts["m"] * 100, y=pts["T"], z=pts["iv"] * 100, mode="markers", marker=dict(size=2.5, color=RED), name="시장 점")
            if len(exps) >= 2:
                st.plotly_chart(surf_fig(mg * 100, Tn, Zs, "ln(K/S) %", "만기 (년)", "내재변동성 %", "내재변동성 곡면 (행사가 × 만기)", [sc], cs="Viridis", h=480),
                                width="stretch", key=f"sf{key}")
            else:
                st.info("만기가 1개뿐이라 곡면 대신 왼쪽 곡선만 표시")
        tab = pd.DataFrame([{"만기": f"{e:%Y-%m-%d}", "남은 일수": (e - asof).days, "등가격 변동성": f"{v[0]:.1%}", "기울기": f"{v[1]:+.3f}",
                             "휘어짐": f"{v[2]:.3f}", "옵션 수": v[3]} for e, v in sorted(fits.items())])
        st.dataframe(tab, hide_index=True, width="stretch")
        st.caption(f"{label} · 외가격 옵션만(콜 K>S, 풋 K<S) · 가격 = 호가 중간값(없으면 최근 체결가) · 무위험 금리 {r2:.2%} 고정 · 배당·보유비용 0 가정. "
                   "ETF 옵션은 미국형이지만 외가격만 써서 유럽형 공식과 차이가 작음.")

    show_market(ch, pts, fits, f"{name} · 모형 {'블랙숄즈' if model == 'BS' else 'Black-76'}", "m")
    with st.expander("읽는 법 · 원자재별로 보이는 특징"):
        st.markdown("- **금·은**: 양 끝이 올라가는 U자(스마일) — 위기 때 급등, 정책·달러 충격 때 급락 둘 다 대비.\n"
                    "- **원유**: 한쪽이 더 가파름(스큐) — 공급 차질로 인한 급등(콜 쪽) 또는 수요 붕괴 급락(풋 쪽) 중 시장이 더 걱정하는 쪽이 비쌈.\n"
                    "- **만기 구조**: 가까운 만기 변동성이 먼 만기보다 높으면 '지금 불안', 낮으면 '앞으로 불안'.\n"
                    "- 대시보드 본판의 OVX(원유)·GVZ(금) 변동성지수 ≈ 약 30일 등가격 변동성 — 이 탭의 등가격 값과 비슷한 수준이어야 함.")
    if not ch.empty and st.button("오늘 시세 저장 (data/options/) — 매일 쌓으면 과거 곡선 자료가 됨"):
        st.success(f"저장: {O.save_snapshot(name, ch)}")

# ════════════════ ③ 다른 원자재 추가
with t3:
    st.markdown("""### 구조 — 시세만 표준 표로 넣으면 계산·그림은 그대로
```
[시세 원천]                      [표준 표: 한 행 = 옵션 1개]          [계산]                         [화면]
yfinance (공개 ETF 옵션)  ─┐
CSV (직접 구한 자료)       ─┼─▶  asof · underlying · expiry ·  ─▶  iv_points: 시장가 → 내재변동성  ─▶  스마일 · 3D 곡면 · 표
(향후) CME·Barchart API   ─┘     strike · type · bid · ask …        smile_fit: 만기별 곡선 맞춤
```
### 단계
1. **자료 구하기** — 곡물·원유 선물옵션 정산가(CME DataMine·Barchart 등 유료) 또는 거래 상대방 견적. 하루치라도 행사가 10개 이상 · 만기 2개 이상이면 곡면이 그려짐.
2. **표준 열로 CSV 저장** — 아래 열 이름 그대로. 선물옵션이면 `underlying`에 **그 옵션 만기에 해당하는 선물가격**을 넣음.
3. **등록** — `options_bs.py`의 `SOURCES`에 한 줄 추가 → 재실행하면 ② 탭 품목 목록에 나타남:
```python
"옥수수 선물옵션 (CBOT ZC)": {"kind": "csv", "path": "data/options/corn_zc.csv", "model": "B76"},
```
4. **(먼저 시험)** 등록 전에 아래에서 CSV를 올려 바로 결과를 확인할 수 있음.
""")
    st.dataframe(pd.DataFrame({"열 이름": list(O.CSV_COLS), "내용": list(O.CSV_COLS.values()),
                               "필수": ["●" if k in ("asof", "underlying", "expiry", "strike", "type") else "선택(가격은 bid·ask 또는 last 중 하나 필요)"
                                        for k in O.CSV_COLS]}), hide_index=True, width="stretch")
    # 양식 = 블랙숄즈로 만든 가상 시세(옥수수 선물 450센트, 스마일 포함) → 내려받아 그대로 올려 보면 흐름 확인 가능
    asof = pd.Timestamp.today().normalize()
    rows = []
    for dte, F in ((30, 450.0), (90, 458.0), (180, 466.0)):
        for Kx in np.arange(360, 561, 10):
            mm = np.log(Kx / F); sv = 0.24 + 0.05 * mm + 0.6 * mm * mm
            for t in "CP":
                px = O.b76_price(F, Kx, dte / 365, 0.04, sv, t == "C")
                rows.append({"asof": asof.date(), "underlying": F, "expiry": (asof + pd.Timedelta(days=dte)).date(), "strike": Kx, "type": t,
                             "bid": round(px * 0.97, 3), "ask": round(px * 1.03, 3), "last": round(px, 3), "open_interest": 500})
    tpl = pd.DataFrame(rows)
    st.download_button("양식 CSV 내려받기 (가상 옥수수 선물옵션 · 만기 3개)", tpl.to_csv(index=False).encode("utf-8-sig"), "options_template.csv", "text/csv")
    st.markdown("---\n### CSV 올려서 바로 시험")
    c = st.columns([2, 1, 1])
    up = c[0].file_uploader("표준 열 CSV", type="csv")
    um = c[1].radio("모형", ["B76", "BS"], captions=["선물옵션", "현물·ETF 옵션"])
    ur = c[2].slider("무위험 금리 (%)", 0.0, 8.0, 4.0, 0.25, key="ru") / 100
    if up is not None:
        try:
            uch = O.chain_csv(io.BytesIO(up.getvalue()))
            upts = O.iv_points(uch, ur, um, max_spread=1.0, min_oi=0)
            show_market(uch, upts, O.smile_fit(upts), f"올린 파일 {up.name} · 모형 {um}", "u")
        except Exception as e:
            st.error(f"읽기 실패: {e}")
    else:
        st.caption("양식 CSV를 내려받아 그대로 올려 보면, 가상 자료에 넣어 둔 스마일(등가격 24%)이 그대로 복원되는 것을 확인할 수 있음.")
