"""원자재 시계열 분석 랩 (Streamlit)

실행:  .venv\\Scripts\\streamlit run app.py
"""
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import analytics as A

st.set_page_config(page_title="Commodity Analytics Lab", layout="wide", page_icon="📈")

try:
    DARK = st.context.theme.type == "dark"
except Exception:
    DARK = False
PAL = (["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"] if DARK else
       ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"])
MID = "#383835" if DARK else "#f0efec"
# 히트맵 색: 단색 2개를 잇는 대신 여러 색상(hue)을 거치게 해 0.1·0.3·0.5 같은 중간값도 구분되게 함
DIV = ([[0, "#c6ecff"], [0.12, "#6cb8f5"], [0.27, "#2f7fd8"], [0.41, "#1d4a8a"], [0.5, "#2b2b2b"],      # 다크: icefire 계열
        [0.59, "#8a2b2b"], [0.73, "#d6452f"], [0.88, "#f59a3c"], [1, "#ffe28a"]] if DARK else
       [[0, "#053061"], [0.1, "#2166ac"], [0.2, "#4393c3"], [0.3, "#92c5de"], [0.4, "#d1e5f0"], [0.5, "#f7f7f7"],  # 라이트: RdBu
        [0.6, "#fddbc7"], [0.7, "#f4a582"], [0.8, "#d6604d"], [0.9, "#b2182b"], [1, "#67001f"]])
SEQ = "Viridis" if DARK else "YlGnBu"
CRISIS_FILL = "rgba(137,135,129,0.18)"


# ============================================================================
# 데이터 로딩 (캐시)
# ============================================================================
@st.cache_data(show_spinner=False)
def _catalog():
    return A.load_catalog()


@st.cache_data(show_spinner=False)
def _levels(extra_bytes: bytes | None, roll: bool = True):
    extra = None
    if extra_bytes:
        extra = pd.read_csv(io.BytesIO(extra_bytes))
        extra[extra.columns[0]] = pd.to_datetime(extra[extra.columns[0]])
        extra = extra.set_index(extra.columns[0]).apply(pd.to_numeric, errors="coerce")
        extra.columns = [f"user:{c}" for c in extra.columns]
    return A.load_levels(extra, roll_adjusted=roll)


@st.cache_data(show_spinner=False)
def _panel(ids: tuple, freq: str, transform: str, start, end, extra_bytes, roll: bool = True):
    L = _levels(extra_bytes, roll)
    return A.make_panel(L, list(ids), UNITS, freq, transform, start, end)


cat = _catalog()
st.sidebar.title("📈 Commodity Analytics Lab")

extra_bytes = None      # (사이드바 CSV 업로드 기능 제거, v1.6.0) — 함수 인자는 호환용으로 유지

NAME = dict(zip(cat.id, cat.name))
UNITS = dict(zip(cat.id, cat.unit))
GROUP = dict(zip(cat.id, cat.group))
order = ["Grains", "Energy", "Metals", "Iron Ore / Coal", "Freight", "China (DCE)", "Softs", "Livestock",
         "FRED daily", "Macro", "FX", "User"]
ALL_IDS = sorted(cat.id, key=lambda i: (order.index(GROUP[i]) if GROUP[i] in order else 99, NAME[i]))
COMMOD_IDS = [i for i in ALL_IDS if GROUP[i] not in ("Macro", "FX", "FRED daily")]
lbl = lambda i: f"{NAME.get(i, i)}  [{GROUP.get(i, '')}]"
short = lambda i: NAME.get(i, i)

freq_lbl = st.sidebar.radio("빈도", list(A.FREQ), index=1, horizontal=True, key="fq_lbl",
                            help="일간은 노이즈가 크고 휴장일 불일치가 있음. 동조성·레짐 분석은 주간 권장.")
FQ = A.FREQ[freq_lbl]; ANN = A.ANN[FQ]
c1, c2 = st.sidebar.columns(2)
start = c1.date_input("시작", pd.Timestamp("2000-01-01"), min_value=pd.Timestamp("1960-01-01"), key="d_start")
end = c2.date_input("끝", pd.Timestamp.today(), key="d_end")
shade_sel = st.sidebar.multiselect("위기 구간 (하이라이트·국면 비교)", list(A.CRISES),
                                   default=["글로벌 금융위기", "코로나", "러-우 전쟁·인플레"])
st.sidebar.caption(" · ".join(f"{k}: {a}~{b}" for k, (a, b) in A.CRISES.items() if k in shade_sel))
st.sidebar.divider()
ROLL = st.sidebar.checkbox("연속월물 롤오버 갭 보정", True, key="roll",
                           help="Yahoo =F 연속월물은 만기 교체일에 두 계약의 가격차만큼 가짜 수익률이 생김 (곡물은 매년 같은 달 → 가짜 계절성). "
                                "거래소 만기 규칙으로 롤일을 찾아 검증된 13개 품목의 롤 당일 수익률을 0으로 보정. 상세: ⑥ 데이터·방법론")
st.sidebar.caption("수익률 = 100×로그차분 (금리·스프레드는 단순차분). 휴장 불일치는 최대 5일 ffill 후 공통 구간만 사용. "
                   "연속월물 롤오버 갭 완화를 위해 ±15σ 클리핑.")


def shade(fig, idx, row=None, col=None):
    if len(idx) == 0:
        return fig
    lo, hi = idx.min(), idx.max()
    for n in shade_sel:
        a, b = map(pd.Timestamp, A.CRISES[n])
        if b < lo or a > hi:
            continue
        kw = dict(row=row, col=col) if row else {}
        fig.add_vrect(x0=max(a, lo), x1=min(b, hi), fillcolor=CRISIS_FILL, line_width=0, layer="below",
                      annotation_text=n, annotation_position="top left", annotation_font_size=10, **kw)
    return fig


def fig_layout(fig, h=360, **kw):
    base = dict(height=h, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                legend=dict(orientation="h", y=-0.15))
    base.update(kw)
    fig.update_layout(**base)
    return fig


def vmax_of(*Ms, nodiag=False, step=0.05):
    """여러 행렬의 |값| 최대 (대각 제외 옵션) → 색 범위를 실제 값 범위에 맞춰 대비를 키움. 같은 값을 넘기면 패널 간 색 비교 가능."""
    v = 0.0
    for M in Ms:
        a = np.abs(np.asarray(M, float))
        if nodiag and a.ndim == 2 and a.shape[0] == a.shape[1]:
            a = a[~np.eye(a.shape[0], dtype=bool)]
        if np.isfinite(a).any():
            v = max(v, float(np.nanmax(a)))
    return (np.ceil(v / step) * step) if v > 0 else 1.0


def heat(M: pd.DataFrame, title, zmin=None, zmax=None, fmt=".2f", h=420, scale=None, text_max=8, nodiag=False):
    """zmin/zmax 생략 → 실제 값 범위로 자동 (발산형은 ±max, 비음수면 0~max). nodiag=True → 대각(자기 자신=1)을 비워 색 범위를 쌍별 값에 씀."""
    Z = M.values.astype(float).copy()
    if nodiag and Z.shape[0] == Z.shape[1]:
        np.fill_diagonal(Z, np.nan)
    if (zmin is None or zmax is None) and scale in (None, DIV, SEQ):     # 그 밖의 색(예: CV 오차)은 Plotly 자동 min~max
        vm = vmax_of(Z)
        zmin = (0.0 if scale is SEQ else -vm) if zmin is None else zmin
        zmax = vm if zmax is None else zmax
    show_text = max(M.shape) <= text_max
    fig = go.Figure(go.Heatmap(z=Z, x=[short(c) for c in M.columns], y=[short(i) for i in M.index],
                               zmin=zmin, zmax=zmax, colorscale=scale or DIV, colorbar=dict(thickness=12),
                               texttemplate=("%{z:" + fmt + "}") if show_text else None,
                               textfont=dict(size=10), hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>"))
    fig.update_layout(title=title, height=h, margin=dict(l=10, r=10, t=40, b=10),
                      yaxis=dict(autorange="reversed", dtick=1), xaxis=dict(dtick=1, tickangle=-45))
    return fig


def heat_row(mats, titles, ranges, scales=None, groups=None, fmt=".2f", h=470, text_max=8, nodiag=True):
    """여러 행렬을 한 그림에 나란히 (y 라벨 1번만, 같은 group은 색 막대 공유) → 좁은 3열 배치보다 칸이 크고 비교 쉬움.
    ranges[i] = (zmin, zmax), groups[i] = 색 축 번호(같으면 같은 색 범위·막대)."""
    n = len(mats); groups = groups or [1] * n; scales = scales or [DIV] * n
    fig = make_subplots(1, n, shared_yaxes=True, subplot_titles=titles, horizontal_spacing=0.03)
    ng = sorted(set(groups))
    for i, M in enumerate(mats):
        Z = M.values.astype(float).copy()
        if nodiag and Z.shape[0] == Z.shape[1]:
            np.fill_diagonal(Z, np.nan)
        fig.add_trace(go.Heatmap(z=Z, x=[short(c) for c in M.columns], y=[short(r) for r in M.index], coloraxis=f"coloraxis{'' if groups[i] == 1 else groups[i]}",
                                 texttemplate=("%{z:" + fmt + "}") if max(M.shape) <= text_max else None, textfont=dict(size=9),
                                 hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>"), 1, i + 1)
    for j, g in enumerate(ng):
        i0 = groups.index(g)
        fig.layout[f"coloraxis{'' if g == 1 else g}"] = dict(colorscale=scales[i0], cmin=ranges[i0][0], cmax=ranges[i0][1],
                                                            colorbar=dict(thickness=12, len=0.9, x=1.0 + 0.09 * j))
    fig.update_yaxes(autorange="reversed", dtick=1)
    fig.update_xaxes(dtick=1, tickangle=-45)
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def panel_or_stop(ids, transform="logret"):
    if len(ids) < 1:
        st.info("품목을 선택하세요."); return None
    R = _panel(tuple(ids), FQ, transform, start, end, extra_bytes, ROLL)
    if len(R) < 60:
        st.warning(f"공통 표본이 {len(R)}개뿐입니다. 시작 시점이 늦은 품목(예: SGX 철광석 2010~, DCE 철광석 2013~)을 빼거나 기간을 넓히세요.")
        return None
    st.caption(f"공통 표본: {R.index.min():%Y-%m-%d} ~ {R.index.max():%Y-%m-%d} · {len(R):,}개 관측 ({freq_lbl}) · {R.shape[1]}개 변수")
    return R


DEF_COMOVE = [i for i in ["yf:ZC=F", "yf:ZS=F", "yf:ZM=F", "yf:ZW=F", "yf:CL=F", "yf:HG=F", "yf:GC=F",
                          "ak:macro_shipping_bdi", "yf:SB=F", "yf:LE=F"] if i in NAME]

tabs = st.tabs(["★ 구매 타이밍", "◆ 리스크 (VaR·꼬리)", "① 시장 구조·동조성", "② 레짐", "③ 변동성·공분산", "④ 수익률 예측", "⑤ 포트폴리오", "⑥ 실험 결과", "⑦ 모델 도감", "⑧ 데이터·방법론"])

# ============================================================================
# ① 동조성
# ============================================================================
def tab_comove():
    st.subheader("원자재 간 동조성 (Co-movement)")
    ids = st.multiselect("분석 품목", ALL_IDS, default=DEF_COMOVE, format_func=lbl, key="cm_ids")
    R = panel_or_stop(ids)
    if R is None:
        return
    m = A.crisis_mask(R.index, shade_sel)

    st.markdown("##### 상관행렬: 평시 vs 위기")
    C_all = R.corr()
    if m.sum() > 20 and (~m).sum() > 20:
        C_n, C_c = R[~m].corr(), R[m].corr()
        vc, vd = vmax_of(C_n, C_c, nodiag=True), vmax_of(C_c - C_n, nodiag=True)
        st.plotly_chart(heat_row([C_n, C_c, C_c - C_n], [f"평시 ({(~m).sum():,}개)", f"위기 구간 ({m.sum():,}개)", "차이 (위기 − 평시)"],
                                 [(-vc, vc), (-vc, vc), (-vd, vd)], groups=[1, 1, 2], h=500), key="h123")
        st.caption(f"색 범위는 실제 값에 맞춤 (평시·위기 공통 ±{vc:.2f}, 차이 ±{vd:.2f}) · 대각(자기 자신 = 1)은 비움.")
        off = ~np.eye(len(C_n), dtype=bool)
        st.caption(f"평균 쌍별 상관: 평시 {C_n.values[off].mean():.3f} → 위기 {C_c.values[off].mean():.3f}. "
                   "위기 시 상관 상승은 분산효과 감소를 의미(단, 변동성 상승에 따른 상관 편향 — Forbes-Rigobon(2002) — 에 유의).")
    else:
        cc = st.columns(2)
        cc[0].plotly_chart(heat(C_all, "전체 기간", nodiag=True), key="h0")
        cc[1].info("선택 기간 안에 위기 구간 표본이 부족합니다.")
    if R.shape[1] >= 2:
        import risk as RK
        q_ = st.select_slider("꼬리 의존: 꼬리 분위 q", [0.01, 0.025, 0.05, 0.10], 0.05, key="cm_tdq")
        TL, TU = RK.tail_dependence(R, q_), RK.tail_dependence(R, q_, upper=True)
        vt = vmax_of(TL, TU, nodiag=True)
        st.plotly_chart(heat_row([TL, TU], [f"동반 폭락 P(j 하위 {q_:.0%} | i 하위 {q_:.0%}) — 보유자 위험", f"동반 급등 (상위 {q_:.0%}) — 구매자 위험"],
                                 [(0, vt), (0, vt)], scales=[SEQ, SEQ], h=500), key="cm_td")
        st.caption(f"상관계수 = 평균적 동조, 꼬리 의존 = '극단적인 날에 같이 움직이는가'. 독립이면 {q_:.0%}, 완전 동조면 100% (순위 기반, 빈도 = 사이드바 설정).")

    st.markdown("##### 시간에 따른 동조성")
    w = st.slider("롤링 창 (관측치 수)", 20, 520, {"B": 120, "W-FRI": 52, "ME": 24}[FQ], key="cm_w")
    col1, col2 = st.columns(2)
    if R.shape[1] >= 2:
        apc = A.avg_pairwise_corr(R, w).dropna()
        f = go.Figure(go.Scatter(x=apc.index, y=apc.values, name="평균 쌍별 상관", line=dict(color=PAL[0], width=2)))
        col1.plotly_chart(shade(fig_layout(f, title=f"평균 쌍별 상관 (롤링 {w})"), apc.index), key="apc")
        ar = A.rolling_absorption(R, w, 1, max(1, w // 20))
        f = go.Figure(go.Scatter(x=ar.index, y=ar.values, name="Absorption Ratio (PC1)", line=dict(color=PAL[1], width=2)))
        col2.plotly_chart(shade(fig_layout(f, title="흡수비율 AR = λ₁/Σλ (Kritzman 2011) — 높을수록 단일요인 지배"), ar.index), key="ar")

        pc = st.columns([1, 1, 3])
        a = pc[0].selectbox("쌍 A", ids, 0, format_func=short, key="pa")
        b = pc[1].selectbox("쌍 B", ids, min(1, len(ids) - 1), format_func=short, key="pb")
        rc = R[a].rolling(w).corr(R[b]).dropna()
        f = go.Figure(go.Scatter(x=rc.index, y=rc.values, line=dict(color=PAL[2], width=2), name="상관"))
        f.add_hline(y=0, line_color="#898781", line_width=1)
        st.plotly_chart(shade(fig_layout(f, 300, title=f"롤링 상관: {short(a)} vs {short(b)}"), rc.index), key="pair")

    st.markdown("##### 주성분 분석 (PCA)")
    ev, load, scores = A.pca(R)
    k = min(4, R.shape[1])
    cp = st.columns([1, 2])
    f = go.Figure(go.Bar(x=[f"PC{i+1}" for i in range(len(ev))], y=ev * 100, marker_color=PAL[0], name="설명력"))
    f.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(ev))], y=np.cumsum(ev) * 100, name="누적",
                           line=dict(color=PAL[1], width=2)))
    cp[0].plotly_chart(fig_layout(f, 380, title="Scree (설명분산 %)", hovermode="x"), key="scree")
    cp[1].plotly_chart(heat(load.iloc[:, :k].T, "적재값 (Loadings) — PC1≈공통 원자재 요인", h=380), key="load")
    pcs = scores["PC1"].cumsum()
    f = go.Figure(go.Scatter(x=pcs.index, y=pcs.values, line=dict(color=PAL[0], width=2), name="PC1 누적 점수"))
    st.plotly_chart(shade(fig_layout(f, 280, title="PC1 누적 점수 (공통요인 '지수')"), pcs.index), key="pc1")

    st.markdown("##### Diebold-Yilmaz 스필오버 (연결성)")
    st.caption("VAR + 일반화 예측오차분산분해(GFEVD). TO=해당 품목이 다른 품목에 주는 충격, FROM=받는 충격, NET>0이면 충격 전달자.")
    d1, d2, d4 = st.columns(3)
    p_dy = d1.number_input("VAR 시차 p", 1, 10, 2, key="dyp")
    H_dy = d2.number_input("예측 기간 H", 2, 30, 10, key="dyh")
    if d4.button("스필오버 계산", type="primary", key="dybtn"):
        with st.spinner("VAR 추정 중..."):
            st.session_state["dy"] = A.dy_spillover(R, int(p_dy), int(H_dy))
    if "dy" in st.session_state:
        tbl, tot = st.session_state["dy"]
        x1, x2 = st.columns([3, 2])
        core = tbl.iloc[:-2, :-1]
        x1.plotly_chart(heat(core, f"스필오버 표 (%) — 총 스필오버 지수 {tot:.1f}% (대각 = 자기 몫, 색에서 제외)", 0, None,
                             ".1f", scale=SEQ, text_max=12, nodiag=True), key="dyh_")
        net = tbl.loc["NET (TO-FROM)"].iloc[:-1].sort_values()
        f = go.Figure(go.Bar(x=net.values, y=[short(i) for i in net.index], orientation="h",
                             marker_color=[PAL[0] if v >= 0 else PAL[1] for v in net.values]))
        x2.plotly_chart(fig_layout(f, 420, title="NET 스필오버 (전달자 > 0 > 수신자)", hovermode="y"), key="dynet")
        st.caption("전체 기간 정적 표. 시간에 따른 연결성은 아래 TVP-VAR (롤링창 선택에 좌우되던 롤링 DY는 v1.4.0에서 TVP-VAR로 대체).")

    st.markdown("##### TVP-VAR 연결성 (롤링창 없음, Antonakakis-Chatziantoniou-Gabauer 2020)")
    st.caption("VAR 계수가 매 시점 칼만필터(망각계수 κ1)로 갱신되고 공분산은 EWMA(κ2) → 시점별 일반화 FEVD. "
               "롤링창 길이 선택에 결과가 좌우되는 DY 롤링의 약점 보완, 첫 60기간은 초기값 추정용.")
    v1_, v2_, v3_, v4_ = st.columns(4)
    tp = v1_.number_input("VAR 시차", 1, 4, 1, key="tvp_p")
    tk1 = v2_.number_input("κ1 (계수 망각)", 0.90, 0.999, 0.99, 0.005, format="%.3f", key="tvp_k1", help="작을수록 계수가 빨리 변함")
    tk2 = v3_.number_input("κ2 (공분산 EWMA)", 0.80, 0.999, 0.96, 0.005, format="%.3f", key="tvp_k2")
    if v4_.button("TVP-VAR 연결성 계산", type="primary", key="tvpbtn"):
        import regime as RG
        with st.spinner("칼만필터 + 시점별 FEVD 계산 중..."):
            st.session_state["tvp"] = RG.tvp_var_connectedness(R, int(tp), 10, float(tk1), float(tk2))
    if "tvp" in st.session_state:
        cn = st.session_state["tvp"]
        f = go.Figure(go.Scatter(x=cn["tci"].index, y=cn["tci"].values, line=dict(color=PAL[6], width=2), name="TVP-VAR TCI"))
        st.plotly_chart(shade(fig_layout(f, 320, title="총 연결성 지수 TCI (%, N/(N−1) 조정)"), cn["tci"].index), key="tvp_tci")
        netm = cn["net"].resample("ME").mean() if len(cn["net"]) > 400 else cn["net"]
        vmx = float(np.abs(netm.values).max()) or 1
        f = go.Figure(go.Heatmap(z=netm.T.values, x=netm.index, y=[short(c) for c in netm.columns], zmin=-vmx, zmax=vmx,
                                 colorscale=DIV, hovertemplate="%{y} %{x|%Y-%m}: %{z:.1f}<extra></extra>"))
        f.update_layout(title="순 전이 (TO − FROM, 파랑 = 충격 전달자, 주황 = 수신자) — 월평균", height=120 + 26 * netm.shape[1],
                        margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(f, key="tvp_net")

# ============================================================================
# ② 변동성 · 레버리지
# ============================================================================
def tab_vol():
    st.subheader("변동성 동학과 레버리지 효과")
    st.caption("레버리지 효과: 가격 하락 후 변동성 확대(주식에서 전형적). 원자재는 공급충격으로 가격 상승 시 변동성이 커지는 **역레버리지**가 "
               "자주 관찰됨(특히 곡물·에너지). GJR γ>0 = 레버리지, γ<0 = 역레버리지 / EGARCH γ<0 = 레버리지.")
    g1, g2, g3 = st.columns(3)
    gid = g1.selectbox("품목", ALL_IDS, ALL_IDS.index("yf:CL=F") if "yf:CL=F" in ALL_IDS else 0, format_func=lbl, key="g_id")
    kind = g2.selectbox("모형", ["GJR", "EGARCH", "GARCH"], key="g_kind")
    dist = g3.selectbox("오차분포", ["t", "skewt", "normal"], key="g_dist")
    R1 = panel_or_stop([gid])
    if R1 is None:
        return
    r = R1[gid]
    try:
        pt, vol, info, res = A.garch_fit(r, kind, dist, ANN)
    except Exception as e:
        st.error(f"추정 실패: {e}"); return
    mc = st.columns(5)
    mc[0].metric("γ (비대칭)", "–" if np.isnan(info["gamma"]) else f"{info['gamma']:.4f}")
    mc[1].metric("γ p값", "–" if np.isnan(info["gamma_p"]) else f"{info['gamma_p']:.3f}")
    mc[2].metric("지속성", f"{info['persistence']:.3f}")
    mc[3].metric("반감기", "–" if np.isnan(info["half_life"]) else f"{info['half_life']:.1f}기")
    mc[4].metric("AIC", f"{info['aic']:,.0f}")
    if not np.isnan(info["gamma"]):
        lev = (info["gamma"] > 0) if kind == "GJR" else (info["gamma"] < 0)
        sig = info["gamma_p"] < 0.05
        st.info(("✅ " if sig else "➖ ") + ("레버리지 효과(하락 충격 → 변동성↑)" if lev else "역레버리지(상승 충격 → 변동성↑)")
                + (" — 5% 유의" if sig else " — 통계적으로 유의하지 않음"))
    v1, v2 = st.columns([2, 1])
    f = go.Figure(go.Scatter(x=vol.index, y=vol.values, line=dict(color=PAL[0], width=1.5), name="조건부 변동성(연율 %)"))
    v1.plotly_chart(shade(fig_layout(f, 340, title=f"{short(gid)} 조건부 변동성 (연율화 %) — {kind}-{dist}"), vol.index), key="gvol")
    e, hv = A.news_impact(res, kind)
    f = go.Figure(go.Scatter(x=e, y=np.sqrt(hv * ANN), line=dict(color=PAL[1], width=2), name="다음기 변동성"))
    f.add_vline(x=0, line_color="#898781", line_width=1)
    v2.plotly_chart(fig_layout(f, 340, title="뉴스 충격 곡선 (비대칭이면 레버리지)", hovermode="x",
                               xaxis_title="충격 ε (%)", yaxis_title="연율 변동성"), key="nic")
    with st.expander("모수 추정치"):
        st.dataframe(pt.style.format("{:.4f}"))

    st.markdown("##### 품목별 레버리지 비교 (전체 / 위기 / 평시)")
    lids = st.multiselect("비교 품목", ALL_IDS, default=[i for i in ["yf:CL=F", "yf:ZC=F", "yf:ZS=F", "yf:ZW=F",
                          "yf:HG=F", "yf:GC=F", "ak:macro_shipping_bdi", "yf:^GSPC"] if i in NAME], format_func=lbl, key="lev_ids")
    hz = st.slider("미래 실현변동성 기간 h (관측치)", 1, 66, {"B": 22, "W-FRI": 4, "ME": 3}[FQ], key="lev_h")
    if st.button("레버리지 비교 실행", type="primary", key="levbtn") and lids:
        rows = []
        prog = st.progress(0.0)
        for j, i in enumerate(lids):
            try:
                rr = _panel((i,), FQ, "logret", start, end, extra_bytes, ROLL)[i]
                _, _, inf, _ = A.garch_fit(rr, "GJR", "t", ANN)
                lc = A.leverage_corr(rr, hz)
                mm = A.crisis_mask(lc.index, shade_sel)
                rows.append({"품목": short(i), "GJR γ": inf["gamma"], "γ p값": inf["gamma_p"],
                             "corr 전체": lc.corr().iloc[0, 1],
                             "corr 위기": lc[mm].corr().iloc[0, 1] if mm.sum() > 20 else np.nan,
                             "corr 평시": lc[~mm].corr().iloc[0, 1] if (~mm).sum() > 20 else np.nan,
                             "시작": rr.index.min().date()})
            except Exception as ex:
                rows.append({"품목": short(i), "GJR γ": np.nan, "오류": str(ex)[:60]})
            prog.progress((j + 1) / len(lids))
        st.session_state["lev"] = pd.DataFrame(rows).set_index("품목")
    if "lev" in st.session_state:
        L_ = st.session_state["lev"]
        f = go.Figure()
        for k_, c_ in zip(["corr 전체", "corr 위기", "corr 평시"], [PAL[0], PAL[7], PAL[2]]):
            if k_ in L_:
                f.add_trace(go.Bar(x=L_.index, y=L_[k_], name=k_, marker_color=c_))
        f.add_hline(y=0, line_color="#898781", line_width=1)
        st.plotly_chart(fig_layout(f, 340, barmode="group", hovermode="x",
                                   title=f"corr(rₜ, 미래 {hz}기 실현변동성) — 음수=레버리지, 양수=역레버리지"), key="levbar")
        st.dataframe(L_.style.format({c: "{:.4f}" for c in L_.columns if L_[c].dtype.kind == "f"}))

# ============================================================================
# ② 레짐 (Markov-switching · HMM · MS 회귀 · Kalman TVP · TAR · STAR)
# ============================================================================
RG_MODELS = ["Hidden Markov (HMM)", "Markov-switching 회귀", "Kalman TVP", "Threshold 회귀 (TAR)", "STAR (평활 전이)"]


def tab_regime():
    st.subheader("레짐 — 지금은 평시인가, 위기인가 · 국면마다 관계가 달라지는가")
    st.caption("숨은 상태를 확률로 추정(Markov-switching·HMM), 관계(베타)가 연속적으로 변한다고 보거나(Kalman TVP), "
               "관측 변수의 임계값으로 국면을 나눔(TAR·STAR). 같은 모델들은 ④ 수익률 예측 탭에서 예측력 비교 후보로도 선택 가능.")
    s1, s2, s3 = st.tabs(["수익률 레짐 (Markov-switching, Hamilton)", "레짐별 관계 (HMM · MS 회귀 · Kalman TVP · TAR · STAR)",
                          "변동성 레짐 (HMM 3상태 · 반감기 안정성)"])
    with s3:
        vol_regime()
    with s1:
        st.caption("단일 품목 수익률의 평균·분산이 숨은 레짐(평시/위기)에 따라 달라진다고 보고, 시점별 레짐 확률을 추정.")
        gid = st.selectbox("품목", ALL_IDS, ALL_IDS.index("yf:CL=F") if "yf:CL=F" in ALL_IDS else 0, format_func=lbl, key="rg_id")
        R1 = panel_or_stop([gid])
        r = R1[gid] if R1 is not None else None
        ms1, ms2 = st.columns([1, 4])
        kreg = ms1.radio("레짐 수", [2, 3], horizontal=True, key="msk")
        if ms1.button("레짐 추정", type="primary", key="msbtn") and r is not None:
            with st.spinner("EM/ML 추정 중... (일간·3레짐은 수십 초)"):
                try:
                    st.session_state["ms"] = (gid, A.markov_switching(r, kreg))
                except Exception as ex:
                    st.error(f"추정 실패: {ex}")
        if "ms" in st.session_state and st.session_state["ms"][0] == gid:
            prob, summ = st.session_state["ms"][1]
            lv = _panel((gid,), FQ, "level", start, end, extra_bytes, False)[gid]   # 가격 표시는 실제 시세
            f = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.05,
                              subplot_titles=(f"{short(gid)} 가격", "평활 레짐 확률 (마지막 = 최고 변동성 레짐)"))
            f.add_trace(go.Scatter(x=lv.index, y=lv.values, line=dict(color=PAL[0], width=1.5), name="가격"), 1, 1)
            for j, c in enumerate(prob.columns):
                f.add_trace(go.Scatter(x=prob.index, y=prob[c], stackgroup="p", name=c, line=dict(width=0.5, color=PAL[[2, 3, 7][j] if kreg == 3 else [2, 7][j]])), 2, 1)
            ms2.plotly_chart(fig_layout(f, 460), key="msfig")
            ms1.dataframe(summ.style.format("{:.3f}"))
    with s2:
        st.caption("Y(품목 수익률)와 X(다른 시장)의 관계가 국면에 따라 달라지는지 추정. 설명 모드 = 동시점 베타(헤지 비율)의 국면 의존성.")
        a1, a2 = st.columns([1, 2])
        tgt = a1.selectbox("종속변수 (Y)", ALL_IDS, ALL_IDS.index("yf:ZC=F") if "yf:ZC=F" in ALL_IDS else 0, format_func=lbl, key="rg_tgt")
        fdef = [i for i in ["yf:CL=F", "yf:DX-Y.NYB", "yf:^GSPC", "fred:VIXCLS"] if i in NAME and i != tgt]
        feats = a2.multiselect("설명변수 (X)", [i for i in ALL_IDS if i != tgt], default=fdef, format_func=lbl, key="rg_feats")
        b1, b2, b3, b4 = st.columns(4)
        rms = b1.multiselect("모델", RG_MODELS, default=["Hidden Markov (HMM)", "Kalman TVP", "Threshold 회귀 (TAR)"], key="rg_models")
        rmode = b2.radio("분석 목적", ["설명 (동시점 베타)", "예측 (다음 기간)"], key="rg_mode")
        thr = b3.selectbox("TAR·STAR 임계 변수", [tgt] + feats, (1 + feats.index("fred:VIXCLS")) if "fred:VIXCLS" in feats else 0,
                           format_func=short, key="rg_thr")
        nst = b4.number_input("레짐 수 (HMM·MS)", 2, 3, 2, key="rg_k")
        if st.button("▶ 레짐 모델 실행", type="primary", key="rg_run") and rms and feats:
            R3 = panel_or_stop([tgt] + feats)
            if R3 is not None:
                md = "explain" if rmode.startswith("설명") else "predict"
                D = A.build_design(R3, tgt, feats, md, 1, 1, True, 0.7)
                hp_ = dict(thr_var=thr, ms_k=int(nst), ms_pc=3, tar_boot=199, hmm_states=int(nst), hmm_cov="full",
                           kalman_q=1e-4, mode=md)
                out_ = {}
                with st.spinner("추정 중..."):
                    for m in rms:
                        try:
                            out_[m] = A.run_model(m, D, ANN, **hp_)
                        except Exception as ex:
                            st.error(f"{m} 실패: {ex}")
                st.session_state["rg_out"] = (D, out_, tgt, md, hp_)
        if "rg_out" in st.session_state and st.session_state["rg_out"][1]:
            D, out_, tgt_, md_, hp_ = st.session_state["rg_out"]
            met = pd.DataFrame({k: v["metrics"] for k, v in out_.items()}).T
            st.dataframe(met[[c for c in met.columns if c in ("R² (학습)", "R² (테스트)", "RMSE (테스트)", "OOS R² (vs 학습평균)")]]
                         .style.format("{:.4f}", na_rep="–"))
            sel = st.selectbox("상세 보기", list(out_), key="rg_sel")
            model_detail(sel, out_[sel], D, tgt_, out_, hp_, P="rg_")


# ============================================================================
# ③ 다변량 GARCH (GARCH · DCC · ADCC · BEKK)
# ============================================================================
def tab_mvgarch():
    import mvgarch as M
    st.subheader("전통 GARCH · DCC · BEKK 변동성 분석")
    st.caption("① 단변량 GARCH로 품목별 변동성 수준·기간구조 → ② DCC/ADCC로 시변 상관 → ③ BEKK로 시장 간 충격·변동성 전이(spillover). "
               "주간 빈도 권장 (일간은 시장별 마감시각 차이로 상관이 과소추정됨).")
    base_ids = [i for i in ["yf:ZC=F", "yf:ZS=F", "yf:ZM=F", "yf:ZW=F", "yf:CL=F", "yf:HG=F", "yf:GC=F", "ak:macro_shipping_bdi"] if i in NAME]
    ids = st.multiselect("분석 품목", ALL_IDS, default=base_ids, format_func=lbl, key="mg_ids")
    R = panel_or_stop(ids)
    if R is None or R.shape[1] < 2:
        if R is not None: st.info("2개 이상 선택하세요.")
        return
    g1, g2 = st.columns(2)
    ukind = g1.selectbox("단변량 모형 (1단계)", ["GJR", "GARCH"], key="mg_kind")
    udist = g2.selectbox("오차분포", ["t", "normal", "skewt"], key="mg_dist")

    # ---- ① 단변량 GARCH 요약 ----
    st.markdown("##### ① 단변량 GARCH 변동성 요약 (연율 %)")
    if st.button("GARCH 요약 계산", type="primary", key="mg_gbtn"):
        with st.spinner("GARCH 추정 중..."):
            st.session_state["mg_g"] = M.garch_table(R, ANN, ukind, udist, {i: short(i) for i in ids})
    if "mg_g" in st.session_state:
        gt, gv = st.session_state["mg_g"]
        c1_, c2_ = st.columns([3, 2])
        c1_.dataframe(gt.drop(columns="id").style.format("{:.2f}", na_rep="–"))
        f = go.Figure()
        f.add_trace(go.Bar(x=gt.index, y=gt["현재 GARCH 변동성"], name="현재", marker_color=PAL[1]))
        f.add_trace(go.Bar(x=gt.index, y=gt["장기 변동성"], name="장기(무조건부)", marker_color=PAL[0]))
        f.add_trace(go.Bar(x=gt.index, y=gt["22기 예측"], name="22기 예측(평균)", marker_color=PAL[2]))
        c2_.plotly_chart(fig_layout(f, 380, barmode="group", hovermode="x", title="현재 vs 장기 vs 예측 변동성"), key="mg_gbar")
        sel_v = st.multiselect("조건부 변동성 차트", list(gv.columns), default=list(gv.columns[:4]), format_func=short, key="mg_vsel")
        f = go.Figure([go.Scatter(x=gv.index, y=gv[c], name=short(c), line=dict(color=PAL[j % 8], width=1.4)) for j, c in enumerate(sel_v)])
        st.plotly_chart(shade(fig_layout(f, 340, title=f"조건부 변동성 (연율 %, {ukind}-{udist})"), gv.index), key="mg_gvol")
        st.caption("현재 백분위 = 현재 조건부 변동성이 표본기간 분포에서 차지하는 위치. 장기 변동성보다 현재가 낮으면 변동성 상승(평균회귀) 여지.")

    # ---- ② DCC / ADCC ----
    st.markdown("##### ② DCC / ADCC — 동적 조건부 상관")
    d1_, d2_ = st.columns([1, 3])
    asym = d1_.radio("모형", ["DCC (Engle 2002)", "ADCC (비대칭)"], key="mg_dcc_t").startswith("ADCC")
    if d1_.button("DCC 추정", type="primary", key="mg_dbtn"):
        with st.spinner("단변량 GARCH + DCC 우도 최적화 중..."):
            try:
                st.session_state["mg_d"] = (M.dcc_fit(R, asym, ukind, udist, ANN), asym)
            except Exception as ex:
                st.error(f"DCC 실패: {ex}")
    if "mg_d" in st.session_state:
        dc, asym_ = st.session_state["mg_d"]
        d1_.dataframe(dc["params"].style.format("{:.4f}"))
        d1_.caption(f"지속성 a+b{'+δg' if asym_ else ''} = {dc['persistence']:.3f} · 상관 로그우도 {dc['corr_loglik']:.1f}")
        cols = dc["cols"]; didx = dc["pairs"].index
        base = d2_.selectbox("기준 품목 (기준 vs 나머지 상관)", cols, format_func=short, key="mg_base")
        f = go.Figure()
        j = 0
        for (a, b), s in dc["pairs"].items():
            if base in (a, b):
                other = b if a == base else a
                f.add_trace(go.Scatter(x=s.index, y=s.values, name=short(other), line=dict(color=PAL[j % 8], width=1.5))); j += 1
        f.add_hline(y=0, line_color="#898781", line_width=1)
        d2_.plotly_chart(shade(fig_layout(f, 380, title=f"DCC 조건부 상관: {short(base)} vs 기타", yaxis_range=[-1, 1]), didx), key="mg_dpair")
        e1_, e2_ = st.columns(2)
        f = go.Figure(go.Scatter(x=dc["avg_corr"].index, y=dc["avg_corr"].values, line=dict(color=PAL[6], width=2), name="평균 DCC 상관"))
        e1_.plotly_chart(shade(fig_layout(f, 300, title="평균 조건부 상관 (전 쌍 평균) — 동조화 지표"), didx), key="mg_davg")
        w = np.full(len(cols), 1 / len(cols))
        pv = np.sqrt(np.einsum("i,tij,j->t", w, dc["H"], w)) * np.sqrt(ANN)
        f = go.Figure(go.Scatter(x=didx, y=pv, line=dict(color=PAL[1], width=1.6), name="동일가중 포트폴리오 변동성"))
        e2_.plotly_chart(shade(fig_layout(f, 300, title="동일가중 포트폴리오 조건부 변동성 (연율 %) — DCC 공분산 기반"), didx), key="mg_dpv")
        dates = didx
        pick = st.select_slider("상관행렬 시점", options=list(dates.strftime("%Y-%m-%d")), value=dates[-1].strftime("%Y-%m-%d"), key="mg_dt")
        ti = dates.get_loc(pd.Timestamp(pick))
        Rm = pd.DataFrame(dc["R"][ti], index=cols, columns=cols)
        st.plotly_chart(heat(Rm, f"DCC 상관행렬 — {pick}  (평균 {dc['avg_corr'].iloc[ti]:.3f})", h=420, nodiag=True), key="mg_dheat")

    # ---- ③ BEKK ----
    st.markdown("##### ③ BEKK(1,1) — 충격·변동성 전이")
    st.caption("Hₜ = CC' + A'εₜ₋₁εₜ₋₁'A + B'Hₜ₋₁B.  A[i,j] (i≠j): i 시장의 충격 → j 시장 분산, B[i,j]: i 시장의 변동성 → j 시장 변동성. "
               "Full BEKK는 모수가 2N²+N(N+1)/2 개로 많아 2~4개 품목 권장.")
    b1_, b2_, b3_ = st.columns([2, 1, 1])
    bids = b1_.multiselect("BEKK 품목 (2~4개 권장)", ids, default=ids[:3], format_func=short, key="mg_bids")
    bkind = b2_.selectbox("유형", ["Full", "Diagonal", "Scalar"], key="mg_bkind")
    if b3_.button("BEKK 추정", type="primary", key="mg_bbtn"):
        if len(bids) < 2:
            st.warning("2개 이상 선택하세요.")
        elif bkind == "Full" and len(bids) > 5:
            st.warning("Full BEKK는 5개 이하로 선택하세요 (모수 폭증).")
        else:
            with st.spinner("BEKK 우도 최적화 중... (Full 3개 기준 주간 ~30초)"):
                try:
                    st.session_state["mg_b"] = M.bekk_fit(R[bids], bkind, ANN)
                except Exception as ex:
                    st.error(f"BEKK 실패: {ex}")
    if "mg_b" in st.session_state:
        bk = st.session_state["mg_b"]
        nm = {c: short(c) for c in bk["cols"]}
        st.caption(f"로그우도 {bk['loglik']:.1f} · 모수 {bk['npar']} · AIC {bk['aic']:.1f} · BIC {bk['bic']:.1f} · "
                   f"공분산 정상성 근 {bk['stationarity']:.4f} (<1) · 수렴 {'✔' if bk['converged'] else '✖ (결과 주의)'}")
        k1_, k2_ = st.columns(2)
        def mat_fig(Mx, Mt, title):
            txt = [[f"{Mx.iloc[i, j]:.3f}<br>t={Mt.iloc[i, j]:.1f}" for j in range(Mx.shape[1])] for i in range(Mx.shape[0])]
            vm = float(np.abs(Mx.values).max()) or 1
            fig = go.Figure(go.Heatmap(z=Mx.values, x=[f"→ {nm[c]}" for c in Mx.columns], y=[f"{nm[c]} →" for c in Mx.index],
                                       zmin=-vm, zmax=vm, colorscale=DIV, text=txt, texttemplate="%{text}", textfont=dict(size=11)))
            fig.update_layout(title=title, height=120 + 70 * Mx.shape[0], margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(autorange="reversed"))
            return fig
        k1_.plotly_chart(mat_fig(bk["A"], bk["A_t"], "A — 충격(ARCH) 전이: 행 → 열"), key="mg_bA")
        k2_.plotly_chart(mat_fig(bk["B"], bk["B_t"], "B — 변동성(GARCH) 전이: 행 → 열"), key="mg_bB")
        if bkind == "Full" or len(bk["cols"]) > 0:
            sp = M.spillover_table(bk, nm)
            if bkind != "Full":
                st.caption("Diagonal/Scalar BEKK는 비대각 원소가 0으로 고정 → 전이 효과를 추정하지 않음 (Full 선택 시 추정).")
            else:
                st.dataframe(sp.style.format({"계수": "{:.4f}", "t값": "{:.2f}"}), hide_index=True)
        l1_, l2_ = st.columns(2)
        f = go.Figure([go.Scatter(x=s.index, y=s.values, name=f"{nm[a]}–{nm[b]}", line=dict(color=PAL[j % 8], width=1.5))
                       for j, ((a, b), s) in enumerate(bk["pairs"].items())])
        l1_.plotly_chart(shade(fig_layout(f, 340, title="BEKK 조건부 상관", yaxis_range=[-1, 1]), R.index), key="mg_bcorr")
        f = go.Figure([go.Scatter(x=bk["vol"].index, y=bk["vol"][c], name=nm[c], line=dict(color=PAL[j % 8], width=1.5))
                       for j, c in enumerate(bk["cols"])])
        l2_.plotly_chart(shade(fig_layout(f, 340, title="BEKK 조건부 변동성 (연율 %)"), R.index), key="mg_bvol")
        if "mg_d" in st.session_state:
            dc = st.session_state["mg_d"][0]
            common = [p for p in bk["pairs"].columns if p in dc["pairs"].columns]
            if common:
                p0 = common[0]
                f = go.Figure([go.Scatter(x=dc["pairs"].index, y=dc["pairs"][p0], name="DCC", line=dict(color=PAL[0], width=1.6)),
                               go.Scatter(x=bk["pairs"].index, y=bk["pairs"][p0], name="BEKK", line=dict(color=PAL[1], width=1.6)),
                               go.Scatter(x=R.index, y=R[p0[0]].rolling({"B": 120, "W-FRI": 52, "ME": 24}[FQ]).corr(R[p0[1]]),
                                          name="롤링 상관", line=dict(color="#898781", width=1.2, dash="dot"))])
                st.plotly_chart(shade(fig_layout(f, 320, title=f"모형 비교: {nm[p0[0]]}–{nm[p0[1]]} 상관 (DCC vs BEKK vs 롤링)"), R.index), key="mg_cmp")


def model_detail(sel, o, D, tgt_, out, hp, P=""):
    """④ 단일 품목·② 레짐 탭 공용: 선택 모델의 계수·중요도·SHAP·레짐·예측구간 등 상세 화면. P = 위젯 key 접두어."""
    d1, d2 = st.columns(2)
    if o["coef"] is not None:
        cs = o["coef"].copy(); cs.index = [c.replace(c.split(" (")[0], short(c.split(" (")[0])) for c in cs.index]
        cs = cs.reindex(cs.abs().sort_values().index)
        f = go.Figure(go.Bar(x=cs.values, y=cs.index, orientation="h",
                             marker_color=[PAL[0] if v >= 0 else PAL[1] for v in cs.values]))
        d1.plotly_chart(fig_layout(f, max(320, 22 * len(cs) + 80), title=f"{sel} 계수" + (" (최신 필터 β)" if sel == "Kalman TVP" else ""), hovermode="y"), key=P + "cbar")
    elif "importance" in o["extra"]:
        imp = o["extra"]["importance"].copy(); imp.index = [c.replace(c.split(" (")[0], short(c.split(" (")[0])) for c in imp.index]
        ttl = {"LightGBM": "평균 |SHAP| (테스트 구간, 예측 % 단위)",
               "Gaussian Process": "ARD 관련도 = 1/길이척도 (클수록 예측에 민감)"}.get(sel, "순열 중요도 (테스트 구간, MSE 증가량)")
        col = None
        if sel == "LightGBM" and "shap_dir" in o["extra"]:
            dv = o["extra"]["shap_dir"].reindex(o["extra"]["importance"].index).values
            col = [PAL[0] if v >= 0 else PAL[1] for v in dv]
            ttl += " — 파랑: 변수↑ → 예측↑, 주황: 변수↑ → 예측↓"
        f = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=col or PAL[0]))
        d1.plotly_chart(fig_layout(f, max(320, 22 * len(imp) + 80), title=ttl, hovermode="y"), key=P + "imp")
    elif "loss" in o["extra"]:
        lo = o["extra"]["loss"]
        f = go.Figure([go.Scatter(y=lo["train"], name="train", line=dict(color=PAL[0])),
                       go.Scatter(y=lo["valid"], name="valid", line=dict(color=PAL[1]))])
        d1.plotly_chart(fig_layout(f, 320, title="LSTM 학습곡선 (표준화 MSE)", xaxis_title="epoch"), key=P + "loss")
    # 실제 vs 예측 (테스트)
    yt = pd.Series(D.y_te + D.y_mu, index=D.idx_te); pt_ = pd.Series(np.asarray(o["pred_te"]) + D.y_mu, index=D.idx_te)
    f = go.Figure([go.Scatter(x=yt.index, y=yt.values, name="실제", line=dict(color="#898781", width=1)),
                   go.Scatter(x=pt_.index, y=pt_.values, name="예측", line=dict(color=PAL[0], width=1.5))])
    d2.plotly_chart(fig_layout(f, 320, title=f"테스트 구간: 실제 vs 예측 ({short(tgt_)})"), key=P + "avp")
    if "table" in o["extra"]:
        with st.expander("추정 상세표 (t값·p값)"):
            tb = o["extra"]["table"]; st.dataframe(tb.style.format("{:.4f}"))
    with st.expander("잔차 진단 (ε의 자기상관·이분산·정규성·정상성)"):
        dg = pd.DataFrame({"학습 잔차": A.resid_diagnostics(D.y_tr - np.asarray(o["pred_tr"])),
                           "테스트 잔차": A.resid_diagnostics(D.y_te - np.asarray(o["pred_te"]))})
        st.dataframe(dg.style.format("{:.4f}"))
        st.caption("p<0.05 해석 — Ljung-Box: 잔차에 남은 자기상관(=아직 설명 안 된 예측가능 패턴), ARCH-LM: 변동성 군집(이분산, "
                   "→ '목표 변동성 표준화' 또는 GARCH로 처리), Jarque-Bera: 두꺼운 꼬리(원자재 수익률은 거의 항상 기각), "
                   "ADF: 잔차 정상 여부(p<0.05면 정상 — 수익률 회귀에서는 거의 항상 정상).")
    hp_info = {k: v for k, v in o["extra"].items() if isinstance(v, (int, float, np.floating))}
    if hp_info:
        st.caption("선택된 하이퍼파라미터: " + ", ".join(f"{k}={v:.4g}" for k, v in hp_info.items()))

    if "path" in o["extra"]:
        al, path = o["extra"]["path"]
        f = go.Figure()
        for j, nm in enumerate(D.names):
            f.add_trace(go.Scatter(x=np.log10(al), y=path[:, j], name=nm.replace(nm.split(" (")[0], short(nm.split(" (")[0])),
                                   line=dict(width=1.5, color=PAL[j % 8]), showlegend=len(D.names) <= 16))
        if "alpha" in o["extra"]:
            f.add_vline(x=np.log10(o["extra"]["alpha"]), line_dash="dot", line_color="#52514e", annotation_text="CV 선택 λ")
        f.update_xaxes(autorange="reversed", title="log10(λ)  ← 규제 약함")
        st.plotly_chart(fig_layout(f, 380, title=f"{sel} 정규화 경로 (λ↓ 일수록 변수 진입)", hovermode="closest"), key=P + "path")

    if sel == "Kalman TVP":
        B = o["extra"]["beta_smoothed"]
        top = B.iloc[-1].abs().sort_values(ascending=False).index[:8]
        show = st.multiselect("표시할 시변 β", list(B.columns), default=list(top), key=P + "tvpsel",
                              format_func=lambda c: c.replace(c.split(" (")[0], short(c.split(" (")[0])))
        f = go.Figure()
        for j, c in enumerate(show):
            f.add_trace(go.Scatter(x=B.index, y=B[c], name=c.replace(c.split(" (")[0], short(c.split(" (")[0])),
                                   line=dict(color=PAL[j % 8], width=1.8)))
        f.add_vline(x=D.idx_te.min(), line_dash="dash", line_color="#52514e", annotation_text="테스트 시작")
        st.plotly_chart(shade(fig_layout(f, 400, title="Kalman 시변 계수 β(t) (RTS 평활) — 위기 시 민감도 변화 확인"), B.index), key=P + "tvp")

    ex_ = o["extra"]
    nm_ = lambda c: c.replace(c.split(" (")[0], short(c.split(" (")[0]))
    if isinstance(ex_.get("cv"), pd.Series):
        cvs = ex_["cv"]
        f = go.Figure(go.Scatter(x=[str(round(v, 4)) if isinstance(v, float) else str(v) for v in cvs.index], y=cvs.values,
                                 mode="lines+markers", line=dict(color=PAL[0], width=2)))
        st.plotly_chart(fig_layout(f, 260, title=f"{sel} — 시계열 CV 오차 (낮을수록 좋음, 최소점이 선택됨)", hovermode="x",
                                   xaxis_title="요인 수 k / 조합 크기 k / λ", yaxis_title="CV MSE 합"), key=P + "dense_cv")
    if "loadings" in ex_:
        st.plotly_chart(heat(ex_["loadings"].rename(index=nm_).T, f"PCR 선택 주성분 적재값 (k={ex_['k']})", None, None,
                             h=140 + 40 * ex_["loadings"].shape[1], text_max=6), key=P + "dense_load")
    if "pip" in ex_:
        pp = ex_["pip"].rename(index=nm_).sort_values()
        f = go.Figure(go.Bar(x=pp.values, y=pp.index, orientation="h",
                             marker_color=[PAL[0] if v >= 0.5 else "#898781" for v in pp.values]))
        f.add_vline(x=0.5, line_dash="dot", line_color="#52514e", annotation_text="중앙확률모형 기준 0.5")
        st.plotly_chart(fig_layout(f, max(300, 22 * len(pp) + 80), title=f"BMA 사후 포함확률 PIP (방문 모형 {ex_['visited']:,}개)",
                                   hovermode="y", xaxis_range=[0, 1]), key=P + "dense_pip")
        with st.expander("상위 모형 (사후확률 순)"):
            st.dataframe(ex_["top_models"], hide_index=True)
    if "ci" in ex_:
        ci = ex_["ci"].rename(index=nm_).sort_values("사후평균")
        f = go.Figure(go.Scatter(x=ci["사후평균"], y=ci.index, mode="markers", marker=dict(color=PAL[0], size=8),
                                 error_x=dict(type="data", symmetric=False, array=ci["95%"] - ci["사후평균"],
                                              arrayminus=ci["사후평균"] - ci["5%"], color=PAL[0])))
        f.add_vline(x=0, line_color="#898781", line_width=1)
        st.plotly_chart(fig_layout(f, max(300, 22 * len(ci) + 80), title="Horseshoe 사후평균과 90% 신용구간 (0을 포함하지 않으면 유의)",
                                   hovermode="y"), key=P + "dense_ci")
        with st.expander("사후 요약표 (P(β>0), 보존비율 1−κ: 1이면 축소 없음, 0이면 0으로 축소)"):
            st.dataframe(ex_["ci"].style.format("{:.4f}"))
    if "stack_weights" in ex_:
        wv = ex_["stack_weights"].sort_values()
        f = go.Figure(go.Bar(x=wv.values, y=wv.index, orientation="h", marker_color=PAL[2]))
        st.plotly_chart(fig_layout(f, 260, title="Stacking 가중치 (학습구간 시계열 CV 외표본 예측으로 학습, 비음·합 1)",
                                   hovermode="y", xaxis_range=[0, 1]), key=P + "dense_w")
        st.dataframe(ex_["base_metrics"].style.format("{:.4f}"))

    if "regime_coef" in o["extra"]:
        ex = o["extra"]
        rc = ex["regime_coef"].copy()
        rc.index = [c.replace(c.split(" (")[0], short(c.split(" (")[0])) if c != "상수" else c for c in rc.index]
        t1_, t2_ = st.columns([2, 1])
        vm = float(np.abs(rc.values).max()) or 1
        t1_.plotly_chart(heat(rc.T, f"{sel} 레짐별 계수 (표준화 X 기준) — 임계 변수: {short(ex['thr_name'].split(' (')[0])}",
                              -vm, vm, ".2f", h=200, text_max=40), key=P + "thr_coef")
        if "supF" in ex:
            t2_.metric("임계값 γ (원 단위)", f"{ex['gamma']:.3f}")
            t2_.metric("하위 레짐 비중", f"{ex['share_low']:.0%}")
            t2_.metric("선형성 sup-F", f"{ex['supF']:.1f}", help="Hansen(1996) — 임계효과가 없다는 귀무가설")
            t2_.metric("부트스트랩 p값", "–" if not np.isfinite(ex["p_boot"]) else f"{ex['p_boot']:.3f}",
                       help="p<0.05면 '기울기가 q의 임계값 위·아래에서 다르다'")
            qs = ex["q_series"]
            f = go.Figure(go.Scatter(x=qs.index, y=qs.values, line=dict(color=PAL[0], width=1.2), name="q"))
            f.add_hline(y=ex["gamma"], line_dash="dash", line_color=PAL[1], annotation_text=f"γ = {ex['gamma']:.3f}")
            st.plotly_chart(shade(fig_layout(f, 260, title="임계 변수 q와 임계값 γ (위 = 상위 레짐)"), qs.index), key=P + "thr_q")
        else:
            t2_.metric("전환 중심 c (원 단위)", f"{ex['c']:.3f}")
            t2_.metric("전환 속도 γ", f"{ex['gamma']:.2f}", help="클수록 급격한 전환 (∞면 TAR)")
            t2_.metric("선형성 LM p값", "–" if not np.isfinite(ex["lm"]["p"]) else f"{ex['lm']['p']:.3g}",
                       help="Luukkonen 등(1988) — p<0.05면 비선형(평활 전환) 존재")
            tr_ = ex["transition"]
            g1_, g2_ = st.columns(2)
            f = go.Figure(go.Scatter(x=tr_.index, y=tr_.values, line=dict(color=PAL[1], width=2)))
            g1_.plotly_chart(fig_layout(f, 260, title="전환 함수 G(q)", xaxis_title="q (원 단위)", yaxis_range=[0, 1], hovermode="x"), key=P + "star_g")
            gs = ex["G_series"]
            f = go.Figure(go.Scatter(x=gs.index, y=gs.values, line=dict(color=PAL[0], width=1.2)))
            g2_.plotly_chart(shade(fig_layout(f, 260, title="시점별 상위 레짐 가중치 G(q_t)", yaxis_range=[0, 1]), gs.index), key=P + "star_gs")

    if "probs" in o["extra"] and "regime" in o["extra"]:
        ex = o["extra"]
        K = ex["probs"].shape[1]
        cols_ = [PAL[2], PAL[3], PAL[1], PAL[7]][:K] if K > 2 else [PAL[2], PAL[7]]
        f = go.Figure()
        for j, c in enumerate(ex["probs"].columns):
            f.add_trace(go.Scatter(x=ex["probs"].index, y=ex["probs"][c], stackgroup="h", name=c,
                                   line=dict(width=0.5, color=cols_[j])))
        f.add_vline(x=D.idx_te.min(), line_dash="dash", line_color="#52514e", annotation_text="테스트 시작 (이후 전방필터)")
        st.plotly_chart(shade(fig_layout(f, 320, title=f"{sel} 필터 레짐 확률 (레짐 번호 ↑ = 변동성 ↑)", yaxis_range=[0, 1]),
                              ex["probs"].index), key=P + "hmmprob")
        h1_, h2_ = st.columns(2)
        h1_.markdown("**레짐 특성**"); h1_.dataframe(ex["regime"].style.format("{:.3f}"))
        if "transmat" in ex:
            h2_.markdown("**전이확률 P(행→열)**"); h2_.dataframe(ex["transmat"].style.format("{:.3f}"))
        elif "n_pc" in ex:
            h2_.caption(f"설명변수 → 주성분 {ex['n_pc']}개로 압축 후 레짐별 기울기 추정, 아래 레짐별 베타는 원 변수 공간으로 환산. 로그우도 {ex['loglik']:.1f}")
        if "state_betas" in ex:
            SB = ex["state_betas"].copy()
            SB.columns = [c.replace(c.split(" (")[0], short(c.split(" (")[0])) for c in SB.columns]
            vm = float(np.abs(SB.values).max()) or 1
            st.plotly_chart(heat(SB, "레짐별 베타 (표준화 X 1σ당 Y 변화, %) — 위기 레짐에서 민감도가 달라지는지 확인",
                                 -vm, vm, ".2f", h=140 + 40 * K, text_max=20), key=P + "hmmbeta")

    if "shap" in ex_:
        S = ex_["shap"]
        top = S.abs().mean().sort_values(ascending=False).index[:10]
        Sm = S[top].resample({"B": "W-FRI", "W-FRI": "ME", "ME": "QE"}[FQ]).mean() if len(S) > 150 else S[top]
        vm = float(np.abs(Sm.values).max()) or 1
        f = go.Figure(go.Heatmap(z=Sm.T.values, x=Sm.index, y=[nm_(c) for c in top], zmin=-vm, zmax=vm, colorscale=DIV,
                                 hovertemplate="%{y} %{x|%Y-%m-%d}: %{z:.3f}<extra></extra>"))
        f.update_layout(title="SHAP 기여의 시간 변화 (상위 10개 변수, 테스트 구간 평균) — 예측 = 기준값 + Σ 기여",
                        height=140 + 28 * len(top), margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(f, key=P + "shap_heat")
        st.caption(f"기준값(학습구간 평균 예측) {ex_['shap_base'] + D.y_mu:+.3f}% · 나무 {ex_['n_trees']}개 (조기종료). "
                   "TreeSHAP은 변수 간 상호작용 기여도 각 변수에 공정하게(섀플리 값) 배분 — 합이 예측과 정확히 일치.")
        jsel = st.selectbox("SHAP 의존도 (변수 값 vs 기여)", list(top), format_func=nm_, key=P + "shap_dep")
        xv = D.X_te[:, D.names.index(jsel)] * D.sd[D.names.index(jsel)] + D.mu[D.names.index(jsel)]
        f = go.Figure(go.Scatter(x=xv, y=S[jsel].values, mode="markers", marker=dict(size=5, color=PAL[0], opacity=0.5)))
        f.add_hline(y=0, line_color="#898781", line_width=1)
        st.plotly_chart(fig_layout(f, 300, title=f"{nm_(jsel)}: 값에 따른 예측 기여 (비선형·임계 효과 확인)",
                                   xaxis_title="변수 값 (원 단위)", yaxis_title="SHAP 기여 (%)", hovermode="closest"), key=P + "shap_dep_fig")
    if "pred_std" in ex_:
        sd_ = ex_["pred_std"]; pm = pd.Series(np.asarray(o["pred_te"]) + D.y_mu, index=D.idx_te)
        f = go.Figure([go.Scatter(x=sd_.index, y=pm + 1.645 * sd_, line=dict(width=0), showlegend=False),
                       go.Scatter(x=sd_.index, y=pm - 1.645 * sd_, fill="tonexty", line=dict(width=0), name="90% 예측구간",
                                  fillcolor="rgba(32,128,141,0.18)"),
                       go.Scatter(x=yt.index, y=yt.values, name="실제", line=dict(color="#898781", width=1)),
                       go.Scatter(x=pm.index, y=pm.values, name="GP 예측평균", line=dict(color=PAL[0], width=1.6))])
        st.plotly_chart(fig_layout(f, 320, title="Gaussian Process 예측평균과 90% 구간 (정규분포 가정)"), key=P + "gp_band")
        st.caption(f"학습에 사용한 최근 관측치 {ex_['n_used']:,}개 · 분산 구성 " +
                   ", ".join(f"{k} {v:.0%}" for k, v in ex_["var_share"].items()) + " — 잡음 비중이 클수록 예측 가능한 부분이 적음")
    if isinstance(ex_.get("cv"), pd.DataFrame) and sel == "Kernel Ridge":
        st.plotly_chart(heat(ex_["cv"], f"Kernel Ridge 시계열 CV 오차 (선택: α={ex_['alpha']:g}, γ·p={ex_['gamma']:g})",
                             None, None, ".3g", h=320, scale="Blues_r", text_max=60), key=P + "krr_cv")

    with st.expander("📏 예측구간 — Conformal (분포 가정 없는 구간 · 실제 커버리지 검증)"):
        st.caption("학습구간 끝 25%를 보정용으로 떼어 잔차 크기의 분위수로 구간을 만듦. 적응형(ACI)은 테스트 중 실제로 구간을 벗어난 비율을 보고 "
                   "구간 폭을 온라인 조정 → 변동성 국면이 바뀌어도 목표 커버리지 유지. h기 목표는 h기 뒤 실현된 오차만 사용 (미래정보 없음).")
        c1_, c2_, c3_, c4_ = st.columns(4)
        cand = [m for m in out if m not in ("분위수 회귀", "앙상블 (단순평균)", "LSTM")]
        cf_m = c1_.selectbox("모델", cand, cand.index(sel) if sel in cand else 0, key=P + "cf_model")
        cf_a = c2_.select_slider("목표 커버리지", [0.5, 0.68, 0.8, 0.9, 0.95], 0.9, key=P + "cf_cov")
        cf_n = c3_.checkbox("변동성 정규화 (|잔차|/σₜ)", True, key=P + "cf_norm", help="고변동기엔 넓게, 저변동기엔 좁게")
        cf_g = c4_.number_input("ACI 적응 속도 γ", 0.001, 0.1, 0.01, 0.001, format="%.3f", key=P + "cf_gamma")
        if st.button("구간 계산", key=P + "cf_run"):
            import models_ml as MM
            with st.spinner("보정 모델 적합 중..."):
                st.session_state[P + "cf"] = (cf_m, MM.conformal(D, cf_m, ANN, alpha=1 - cf_a, normalize=cf_n, gamma=cf_g, **hp))
        if (P + "cf") in st.session_state:
            cm_, C = st.session_state[P + "cf"]
            st.dataframe(C["summary"].style.format("{:.3f}").apply(
                lambda r: ["font-weight:700" if r.name == "커버리지" and abs(v - C["target"]) <= 0.03 else "" for v in r], axis=1))
            ym = D.y_mu
            f = go.Figure([go.Scatter(x=C["aci"].index, y=C["aci"].hi + ym, line=dict(width=0), showlegend=False),
                           go.Scatter(x=C["aci"].index, y=C["aci"].lo + ym, fill="tonexty", line=dict(width=0),
                                      fillcolor="rgba(32,128,141,0.18)", name="ACI 구간"),
                           go.Scatter(x=C["split"].index, y=C["split"].hi + ym, line=dict(color=PAL[1], width=1, dash="dot"), name="분할 구간"),
                           go.Scatter(x=C["split"].index, y=C["split"].lo + ym, line=dict(color=PAL[1], width=1, dash="dot"), showlegend=False),
                           go.Scatter(x=C["y"].index, y=C["y"] + ym, name="실제", line=dict(color="#898781", width=1)),
                           go.Scatter(x=C["pred"].index, y=C["pred"] + ym, name="예측", line=dict(color=PAL[0], width=1.4))])
            st.plotly_chart(shade(fig_layout(f, 340, title=f"{cm_} — {C['target']:.0%} 예측구간 (테스트)"), C["y"].index), key=P + "cf_band")
            rc_ = C["rolling_cov"]
            f = go.Figure([go.Scatter(x=rc_.index, y=rc_[c], name=c, line=dict(color=[PAL[1], PAL[0]][j], width=1.6))
                           for j, c in enumerate(rc_.columns)])
            f.add_hline(y=C["target"], line_dash="dash", line_color="#52514e", annotation_text="목표")
            st.plotly_chart(shade(fig_layout(f, 260, title="이동 커버리지 (구간 안에 들어온 비율) — 목표선 근처가 좋음",
                                             yaxis_range=[0, 1]), rc_.index), key=P + "cf_cov_fig")
            st.caption("활용: 예측 수익률 자체보다 '구간 폭'이 실무에 유용 — 폭이 넓어지는 시점 = 불확실성 확대 → 포지션 축소·손절폭 조정 근거.")



def _wf_table(wfres):
    return pd.DataFrame({k: v[2] for k, v in wfres.items()}).T


def _record_ml(cfg, D, out, wfres, pl, wf_pl):
    """④ 모델 실행 1건을 runs/ 에 기록 (설정 + 성과 표 + walk-forward + 누적 성과) → run_id."""
    import runlog as RL
    sc = "부호전략 Sharpe (비용 후)"
    met = pd.DataFrame({k: v["metrics"] for k, v in out.items()}).T
    summ, hints = {}, []
    real = met.drop(index=[i for i in met.index if i == "분위수 회귀"], errors="ignore")
    if cfg["목적"] == "predict" and sc in real and real[sc].notna().any():
        b = real[sc].idxmax()
        summ.update({"단일 최고 모델": b, "단일 Sharpe(비용 후)": real.loc[b, sc], "단일 OOS R²": real.loc[b, "OOS R² (vs 학습평균)"]})
        if pl:
            summ["단일 위약 최댓값95"] = pl["max95_net"]
            met.loc["위약 · 무작위 부호 95% 분위", ["부호전략 Sharpe", sc]] = [pl["single95_gross"], pl["single95_net"]]
            met.loc[f"위약 · 모델 {pl['n_models']}개 중 최댓값 95% 분위", ["부호전략 Sharpe", sc]] = [pl["max95_gross"], pl["max95_net"]]
            hints.append(("단일 분할: 위약 최댓값 초과 ✔" if real.loc[b, sc] > pl["max95_net"] else "단일 분할: 위약 최댓값 미만 (운과 구별 안 됨)"))
    tables = {"metrics": met}
    if wfres:
        wm = _wf_table(wfres)
        b = wm[sc].idxmax()
        summ.update({"WF 최고 모델": b, "WF Sharpe(비용 후)": wm.loc[b, sc], "WF Sharpe(비용 전)": wm.loc[b, "부호전략 Sharpe"],
                     "WF 적중률": wm.loc[b, "방향 적중률"], "WF OOS R²": wm.loc[b, "OOS R² (vs 학습평균)"], "Buy&Hold Sharpe": wm.loc[b, "Buy&Hold Sharpe"]})
        if wf_pl:
            summ["WF 위약 최댓값95"] = wf_pl["max95_net"]
            summ["WF 초과(운 대비)"] = wm.loc[b, sc] - wf_pl["max95_net"]
            hints.append("WF: 위약 최댓값 초과 ✔ (운 이상)" if wm.loc[b, sc] > wf_pl["max95_net"] else
                         ("WF: 단일 위약은 넘지만 최댓값 위약 미만 (여러 모델 중 고른 효과일 수 있음)" if wm.loc[b, sc] > wf_pl["single95_net"]
                          else "WF: 위약 95% 미만 (운의 범위)"))
            wm.loc["위약 · 무작위 부호 95% 분위", ["부호전략 Sharpe", sc]] = [wf_pl["single95_gross"], wf_pl["single95_net"]]
            wm.loc[f"위약 · 모델 {wf_pl['n_models']}개 중 최댓값 95% 분위", ["부호전략 Sharpe", sc]] = [wf_pl["max95_gross"], wf_pl["max95_net"]]
        if wm.loc[b, "방향 적중률"] < 0.5:
            hints.append("적중률 50% 미만 — 수익은 소수의 큰 움직임에서 (불안정할 수 있음)")
        tables["walk_forward"] = wm
        cum = {k: pd.Series(A.band_positions(p_.iloc[::D.horizon].values, getattr(D, "band", 0.0)) * y_.iloc[::D.horizon].values,
                            index=y_.index[::D.horizon]).cumsum() for k, (p_, y_, m_) in wfres.items()}
        y0 = next(iter(wfres.values()))[1]
        cum["Buy & Hold"] = y0.iloc[::D.horizon].cumsum()
        tables["walk_forward_cum"] = pd.DataFrame(cum)
    title = f"{short(cfg['대상'])} · {freq_lbl} · {'walk-forward' if wfres else '단일 분할'} · 모델 {len(cfg['모델'])}개" + \
            (f" · 특성 {'+'.join(cfg['추가특성'])}" if cfg["추가특성"] else "")
    return RL.record("ml", title, cfg, summ, tables, hints)


def _restore_ml(cfg):
    """기록된 ④ 설정을 위젯에 되돌림 (on_click 콜백 → 다음 실행에서 반영)."""
    ss = st.session_state
    inv = {v: k for k, v in A.FREQ.items()}
    ss["fq_lbl"] = inv.get(cfg.get("빈도"), ss.get("fq_lbl"))
    for k_, c_ in [("d_start", "시작"), ("d_end", "끝")]:
        if cfg.get(c_):
            ss[k_] = pd.Timestamp(cfg[c_]).date()
    ss["tgt"] = cfg["대상"]; ss["feats"] = [f for f in cfg["설명변수"] if f in NAME and f != cfg["대상"]]
    ss["mode"] = "설명 (동시점 베타)" if cfg["목적"] == "explain" else "예측 (미래 수익률)"
    ss["lags"], ss["hor"], ss["own"], ss["trf"] = int(cfg["시차"]), int(cfg["예측기간"]), bool(cfg["자기시차"]), float(cfg["학습비중"])
    ss["models"] = [m for m in cfg["모델"] if m in A.CORE_MODELS]; ss["models_adv"] = [m for m in cfg["모델"] if m in A.ADV_MODELS]
    ss["fk"] = list(cfg["추가특성"]); ss["vs"] = bool(cfg["변동성표준화"]); ss["wf"] = bool(cfg["walk_forward"])
    ss["wf_step"] = int(cfg["재추정주기"]); ss["cost"] = float(cfg["비용bp"]); ss["band"] = float(cfg["무거래c"]); ss["xlag"] = bool(cfg["비동기보정"])
    ss["hmm_k"] = int(cfg.get("hp", {}).get("hmm_states", 0))
    ss["roll"] = bool(cfg.get("롤보정", True))
    ss["restored_msg"] = cfg


# ============================================================================
# ③ 회귀 · ML
# ============================================================================
def tab_ml():
    st.subheader("회귀 · 머신러닝 모델 비교")
    if st.session_state.pop("restored_msg", None):
        st.success("⑥ '내 실행 기록'에서 설정을 불러왔습니다 — ▶ 모델 실행을 누르면 같은 조합으로 다시 계산합니다 (데이터가 갱신됐으면 결과가 조금 다를 수 있음).")
    a1, a2 = st.columns([1, 2])
    tgt = a1.selectbox("종속변수 (Y)", ALL_IDS, ALL_IDS.index("yf:ZC=F") if "yf:ZC=F" in ALL_IDS else 0, format_func=lbl, key="tgt")
    feat_def = [i for i in ["yf:ZS=F", "yf:ZM=F", "yf:ZW=F", "yf:CL=F", "yf:HG=F", "yf:GC=F", "ak:macro_shipping_bdi",
                            "yf:DX-Y.NYB", "fred:VIXCLS", "fred:DGS10", "yf:^GSPC"] if i in NAME and i != tgt]
    feats = a2.multiselect("설명변수 (X)", [i for i in ALL_IDS if i != tgt], default=feat_def, format_func=lbl, key="feats")
    b1, b2, b3, b4, b5 = st.columns(5)
    mode = b1.radio("분석 목적", ["설명 (동시점 베타)", "예측 (미래 수익률)"], key="mode",
                    help="설명: Yₜ ~ Xₜ — 동조성/민감도 측정, R²가 높게 나옴. 예측: Yₜ₊ₕ ~ Xₜ, Xₜ₋₁… — 미래정보 누수 없음, 일간 OOS R²는 보통 0 근처.")
    md = "explain" if mode.startswith("설명") else "predict"
    lags = b2.number_input("시차 수 (예측)", 1, 20, 3, disabled=md == "explain", key="lags",
                           help="입력으로 '얼마나 과거까지' 볼지. 3이면 각 변수의 t, t-1, t-2 시점 수익률을 설명변수로 사용 (뒤를 보는 창).")
    hor = b3.number_input("예측 기간 h", 1, 66, 1, disabled=md == "explain", key="hor",
                          help="'얼마나 앞'을 맞힐지. h=4(주간)이면 다음 4주 누적수익률(t+1~t+4)이 목표 (앞을 보는 창). h>1은 관측치가 겹치므로 지표는 겹침을 제거해 계산.")
    own = b4.checkbox("Y 자기시차 포함", True, disabled=md == "explain", key="own")
    trf = b5.slider("학습 비중", 0.5, 0.9, 0.7, 0.05, key="trf")
    models = st.multiselect(f"모델 선택 (모형군별 대표 {len(A.CORE_MODELS)}개)", A.CORE_MODELS, default=A.CORE_MODELS, key="models",
                            help="기준 OLS · 축소 Ridge · 변수 선택 Lasso · 요인 PLS · 시변 계수 Kalman · 레짐 HMM · 부스팅 LightGBM")
    with st.expander(f"고급: 추가 모델 {len(A.ADV_MODELS)}개 (벌점 회귀 변형 · 밀집 요인 · 베이즈 · 레짐 회귀 · 커널 · LSTM)"):
        models = models + st.multiselect("추가 모델", A.ADV_MODELS, default=[], key="models_adv")
        st.caption("모델을 많이 돌릴수록 '그중 최고'의 성과는 운만으로도 높아집니다 — 성과 표의 '위약 · 모델 N개 중 최댓값' 기준선과 비교하세요.")
    f1, f2, f3, f4 = st.columns([3, 1.3, 1.3, 1])
    avail = [k for k in A.FEATURE_KINDS if not (k == "COT 헤징압력" and tgt not in A.COT_MAP)
             and not (k == "베이시스(캐리)" and tgt not in A.BASIS_MAP)]
    fk = f1.multiselect("추가 특성 (Y 자신의 정보로 만든 예측 변수)", avail, default=[], key="fk",
                        help="모멘텀 1·3·6·12개월 · 실현변동성 · 계절성(Fourier) · COT 헤징압력 z(곡물·구리·WTI·금) · 베이시스(현물-선물, 에너지). "
                             "모두 t 시점에 실제로 알 수 있었던 값만 사용 (COT는 발표 다음 주 월요일부터).")
    vs = f2.checkbox("목표 변동성 표준화 (y/σₜ)", False, key="vs",
                     help="이분산 제거·위험 균등화. 잔차 ARCH 효과는 사라지지만 예측력이 오르는 것은 아님 (⑥ 방법론 참고).")
    wf = f3.checkbox("Walk-forward 검증", False, key="wf", help="확장창으로 주기적으로 재학습하며 평가 — 단일 분할보다 신뢰도 높음.")
    wf_step = f4.number_input("재추정 주기", 5, 500, {"B": 126, "W-FRI": 26, "ME": 6}[FQ], key="wf_step", disabled=not wf)
    g1_, g3_, g2_ = st.columns([1, 1, 2])
    band_c = g3_.number_input("무거래 구간 c", 0.0, 3.0, 0.0, 0.25, key="band",
                              help="|예측| < c × (그 시점까지 예측값 표준편차)이면 거래하지 않고 직전 포지션 유지 — 예측이 0 근처에서 흔들릴 때의 "
                                   "잦은 반전(회전율·비용)을 줄임. 0 = 매 기간 부호대로.")
    cost_bp = g1_.number_input("거래비용 (편도, bp)", 0.0, 100.0, 5.0, 0.5, key="cost",
                               help="포지션 1단위 변경당 비용. CBOT 옥수수 1틱(0.25¢ ≈ 5bp) + 수수료 기준 약 5bp. 롱→숏 반전은 2단위.")
    xlag = 1 if g2_.checkbox("비동기 마감 보정: 타 시장 변수를 1기간 추가 지연 (일간 권장)", FQ == "B", key="xlag",
                             help="일간 데이터는 시장별 마감시각이 달라(CBOT 14:20 ET, S&P 16:00 ET, DCE·Baltic은 아시아·유럽 시간) "
                                  "늦게 마감하는 시장의 t일 수익률에 대상 시장의 t+1일 정보가 일부 섞임 → 1일 더 늦춰 누수 차단.") and md == "predict" else 0
    with st.expander("하이퍼파라미터"):
        h1, h2, h3, h4 = st.columns(4)
        hp = dict(
            l1_ratio=h1.multiselect("Elastic Net l1_ratio 후보", [0.1, 0.3, 0.5, 0.7, 0.9, 0.95], [0.1, 0.5, 0.9]),
            ada_gamma=h1.number_input("Adaptive Lasso γ (가중치 지수)", 0.5, 3.0, 1.0, 0.5),
            scad_gamma=h2.number_input("SCAD a (>2)", 2.1, 10.0, 3.7, 0.1),
            mcp_gamma=h2.number_input("MCP γ (>1)", 1.1, 10.0, 3.0, 0.1),
            tau=h3.slider("분위수 τ", 0.01, 0.99, 0.05, 0.01),
            kalman_q=10 ** h3.slider("Kalman log10(q) (β 변화속도)", -7.0, -1.0, -4.0, 0.5),
            n_trees=h4.number_input("트리 수", 50, 1000, 300, 50),
            max_depth=h4.number_input("트리 깊이", 2, 12, 4),
            seq=h1.number_input("LSTM 시퀀스 길이", 5, 120, 20), hidden=h2.number_input("LSTM 은닉 크기", 8, 128, 32, 8),
            epochs=h3.number_input("LSTM 최대 에폭", 5, 200, 40, 5),
            hmm_states=h4.selectbox("HMM 레짐 수", [0, 2, 3, 4, 5], 0, format_func=lambda k: "자동 (BIC, 2~5)" if k == 0 else str(k), key="hmm_k",
                                    help="예측 모드는 레짐별 회귀(국면마다 X→y 베타가 다름). 자동 = 학습구간 BIC 최소 상태 수"),
            hmm_cov=h4.selectbox("HMM 공분산", ["full", "diag"], help="full: 레짐별 베타(조건부 회귀) 추정 가능 / diag: 모수 적음"),
            k_max=h1.number_input("PCR·PLS 최대 요인 수 (CV 선택)", 1, 30, 10),
            csr_kmax=h2.number_input("Complete Subset 최대 k (CV 선택)", 1, 15, 8),
            csr_M=h2.number_input("Complete Subset 조합 수 M", 100, 20000, 2000, 100),
            bma_iters=h3.number_input("BMA MC³ 반복", 2000, 200000, 20000, 1000),
            hs_iters=h3.number_input("Horseshoe Gibbs 반복", 500, 20000, 2000, 500),
            mn_theta=h4.number_input("Minnesota θ (타변수 축소)", 0.05, 1.0, 0.5, 0.05,
                                     help="1이면 자기·타 변수 동일 취급, 작을수록 타 변수 계수를 더 강하게 0으로"),
            mode=md,
            thr_var=h1.selectbox("TAR·STAR 임계 변수 q", [tgt] + feats, (1 + feats.index("fred:VIXCLS")) if "fred:VIXCLS" in feats else 0,
                                 format_func=short, key="thr_var",
                                 help="레짐을 가르는 관측 변수 (예: VIX 수준, 자기 수익률). 예측 모드에서는 t 시점 값(시차 0)을 사용."),
            ms_k=h2.number_input("Markov-switching 레짐 수", 2, 3, 2),
            ms_pc=h3.number_input("Markov-switching 주성분 수", 1, 8, 3, help="레짐마다 기울기가 달라 모수가 많아지므로 설명변수를 주성분으로 압축"),
            tar_boot=h4.number_input("TAR 부트스트랩 반복 (선형성 검정)", 49, 999, 199, 50),
            lgb_leaves=h1.number_input("LightGBM 잎 수", 3, 63, 7, help="나무 하나의 최대 잎 수 — 작을수록 단순(과적합↓)"),
            lgb_lr=h2.number_input("LightGBM 학습률", 0.005, 0.3, 0.03, 0.005, format="%.3f",
                                   help="반복 횟수는 학습구간 끝 20%로 조기종료 선택"),
            lgb_l2=h3.number_input("LightGBM L2 규제", 0.0, 100.0, 5.0, 1.0),
            gp_nmax=h4.number_input("GP 최대 학습 관측치", 200, 4000, 1500, 100,
                                    help="가우시안 과정은 계산량 O(n³) → 학습구간 최근 n개만 사용"),
        )
        ens = st.checkbox("앙상블(선택 모델 예측의 단순평균) 추가", True,
                          help="예측 결합(forecast combination)은 개별 모델보다 안정적인 경우가 많음 (Timmermann 2006).")
        st.caption("Lasso/EN/Adaptive/SCAD/MCP의 λ는 시계열 교차검증(TimeSeriesSplit 5-fold, 학습구간 내부)으로 선택. "
                   "X는 학습구간 평균·표준편차로 표준화 → 계수는 '1σ 변화당 Y 변화(%)'로 비교 가능.")
    run = st.button("▶ 모델 실행", type="primary", key="runml")
    R3 = panel_or_stop([tgt] + feats)
    if R3 is None:
        return
    with st.expander("정상성 진단 (ADF · KPSS · 자기상관 · ARCH)"):
        st.caption("ADF H0=단위근(비정상) → p<0.05면 정상. KPSS H0=정상 → p<0.05면 비정상. "
                   "가격 레벨은 비정상, 수익률(로그차분)은 정상인 것이 일반적 → 이 앱은 수익률을 쓰므로 추세는 이미 제거된 상태.")
        if st.button("진단 실행", key="statbtn"):
            st.session_state["stat"] = A.stationarity_table(_levels(extra_bytes, ROLL), [tgt] + feats, UNITS, FQ, start, end, NAME)
        if "stat" in st.session_state and len(st.session_state["stat"]):
            st.dataframe(st.session_state["stat"].style.format({c: "{:.4f}" for c in st.session_state["stat"].columns if c != "판정"}))
    if run:
        if not models or (not feats and not fk and not (md == "predict" and own)):
            st.warning("모델과 설명변수(또는 추가 특성)를 하나 이상 선택하세요."); return
        F = A.make_features(_levels(extra_bytes, ROLL), tgt, UNITS, FQ, R3.index, fk) if fk else None
        vsw = {"B": 21, "W-FRI": 13, "ME": 6}[FQ] if vs else None
        D = A.build_design(R3, tgt, feats, md, int(lags), int(hor), own, trf, extra=F, vol_scale=vsw, xlag=xlag)
        D.cost = cost_bp / 100.0     # bp -> % (수익률이 % 단위)
        D.band = float(band_c)
        out, prog = {}, st.progress(0.0, "모델 추정 중")
        for j, mname in enumerate(models):
            prog.progress(j / len(models), f"{mname} 추정 중...")
            try:
                out[mname] = A.run_model(mname, D, ANN, **hp)
            except Exception as ex:
                st.error(f"{mname} 실패: {ex}")
        prog.empty()
        pool = {k: v for k, v in out.items() if k != "분위수 회귀"}
        if ens and len(pool) >= 2:
            ptr = np.mean([np.asarray(v["pred_tr"]) for v in pool.values()], axis=0)
            pte = np.mean([np.asarray(v["pred_te"]) for v in pool.values()], axis=0)
            out["앙상블 (단순평균)"] = dict(coef=None, pred_tr=ptr, pred_te=pte, extra={"구성": list(pool)},
                                       metrics=A._metrics(D.y_tr, ptr, D.y_te, pte, ANN, D.horizon, D))
        wfres = {}
        if wf and md == "predict":
            wm = [m for m in models if m not in ("LSTM", "분위수 회귀")]
            prog = st.progress(0.0, "Walk-forward 검증 중")
            for j, mname in enumerate(wm):
                prog.progress(j / max(1, len(wm)), f"Walk-forward: {mname}")
                try:
                    wfres[mname] = A.walk_forward(D, mname, ANN, int(wf_step), 0.5, **hp)
                except Exception as ex:
                    st.error(f"Walk-forward {mname} 실패: {ex}")
            prog.empty()
            if "LSTM" in models:
                st.info("LSTM은 재학습 비용이 커서 walk-forward에서 제외했습니다.")
        elif wf:
            st.info("Walk-forward는 '예측' 모드에서만 의미가 있습니다.")
        st.session_state["ml"] = (D, out, tgt, md)
        st.session_state["ml_pl"] = None
        if md == "predict":
            mt_ = pd.DataFrame({k: v["metrics"] for k, v in out.items() if k != "분위수 회귀"}).T
            ls_, sw_ = mt_["롱 비중"].mean(), mt_["연간 회전율"].mean() * D.horizon / ANN / 2     # 회전율 2단위 = 전환 1회
            st.session_state["ml_pl"] = A.placebo_sharpe(D.y_te + D.y_mu, ANN, D.horizon, D.cost, n_models=len(mt_),
                                                         long_share=ls_ if np.isfinite(ls_) else 0.5, switch=sw_ if np.isfinite(sw_) else 0.5)
        st.session_state["ml_wf"] = wfres
        st.session_state["ml_vs"] = bool(vs)
        wf_pl = None
        if wfres:
            wm_ = _wf_table(wfres)
            y0_ = next(iter(wfres.values()))[1]
            ls2, sw2 = wm_["롱 비중"].mean(), wm_["연간 회전율"].mean() * D.horizon / ANN / 2
            wf_pl = A.placebo_sharpe(y0_.values, ANN, D.horizon, D.cost, n_models=len(wm_),
                                     long_share=ls2 if np.isfinite(ls2) else 0.5, switch=sw2 if np.isfinite(sw2) else 0.5)
        st.session_state["ml_wf_pl"] = wf_pl
        cfg = dict(대상=tgt, 설명변수=list(feats), 목적=md, 시차=int(lags), 예측기간=int(hor), 자기시차=bool(own), 학습비중=float(trf),
                   모델=list(models), 추가특성=list(fk), 변동성표준화=bool(vs), walk_forward=bool(wf and md == "predict"), 재추정주기=int(wf_step),
                   비용bp=float(cost_bp), 무거래c=float(band_c), 비동기보정=int(xlag), 빈도=FQ, 시작=str(start), 끝=str(end), 롤보정=bool(ROLL),
                   앙상블=bool(ens), hp={k: v for k, v in hp.items() if k != "mode"})
        try:
            st.session_state["ml_run_id"] = _record_ml(cfg, D, out, wfres, st.session_state.get("ml_pl"), wf_pl)
        except Exception as ex:
            st.session_state["ml_run_id"] = None
            st.warning(f"실행 기록 저장 실패: {ex}")

    if "ml" in st.session_state:
        D, out, tgt_, md_ = st.session_state["ml"]
        if not out:
            return
        st.caption(f"학습 {D.idx_tr.min():%Y-%m-%d}~{D.idx_tr.max():%Y-%m-%d} ({len(D.idx_tr):,}) · "
                   f"테스트 {D.idx_te.min():%Y-%m-%d}~{D.idx_te.max():%Y-%m-%d} ({len(D.idx_te):,}) · 변수 {len(D.names)}개")
        met = pd.DataFrame({k: v["metrics"] for k, v in out.items()}).T
        st.markdown("##### 성과 지표")
        best = met["OOS R² (vs 학습평균)"].idxmax() if met["OOS R² (vs 학습평균)"].notna().any() else None
        pl = st.session_state.get("ml_pl") if md_ == "predict" else None
        if pl:
            n_ = pl["n_models"]
            met.loc["위약 · 무작위 부호 95% 분위", ["부호전략 Sharpe", "부호전략 Sharpe (비용 후)"]] = [pl["single95_gross"], pl["single95_net"]]
            met.loc[f"위약 · 모델 {n_}개 중 최댓값 95% 분위", ["부호전략 Sharpe", "부호전략 Sharpe (비용 후)"]] = [pl["max95_gross"], pl["max95_net"]]
        st.dataframe(met.map(lambda v: "–" if pd.isna(v) else f"{v:.4f}").style.apply(
            lambda s: ["font-weight:700" if s.name == best else ("color:#898781;font-style:italic" if s.name.startswith("위약") else "") for _ in s], axis=1))
        if best:
            st.caption(f"굵게 = 테스트 OOS R² 최고 모델: **{best}**")
        if pl:
            beat = met.drop(index=[i for i in met.index if i.startswith("위약")])["부호전략 Sharpe (비용 후)"] > pl["max95_net"]
            st.caption(f"**다중 비교 기준선** — 위약 = 같은 테스트 구간에서 무작위 ±1 포지션 (롱 비중 {pl['long_share']:.0%}·전환 확률 {pl['switch']:.0%}를 모델 평균에 맞춤, {pl['B']:,}회). "
                       f"모델 1개의 Sharpe(비용 후)가 '무작위 부호 95% 분위'({pl['single95_net']:+.2f})를 넘으면 단독으로는 운 이상이지만, "
                       f"{pl['n_models']}개 중 최고를 골랐다면 '최댓값 95% 분위'({pl['max95_net']:+.2f})를 넘어야 운과 구별됩니다 → "
                       + (f"넘은 모델: **{', '.join(beat[beat].index)}**" if beat.any() else "**넘은 모델 없음** (최고 성과도 운의 범위)"))
        if md_ == "predict":
            st.caption(f"**부호전략 정의** — 단일 자산 롱/숏 타이밍: 포지션 = sign(예측 수익률) ∈ {{+1, −1}} (명목 100%, 레버리지·비중최적화 없음), "
                       f"리밸런싱 = {freq_lbl} 매 기간{f'(h={D.horizon}이면 {D.horizon}기마다)' if D.horizon > 1 else ''}, "
                       f"수익 = 포지션 × 다음 기간 실현 로그수익률(선물이므로 무위험금리 차감 없음), "
                       f"Sharpe = 평균/표준편차×√{ANN // D.horizon}. 비용 후 = |포지션 변화|×{D.cost * 100:.1f}bp 차감. "
                       "Buy&Hold Sharpe는 같은 테스트 구간 기준. 기간당 R² → Sharpe ≈ √(R²/(1−R²)×연간기간수).")
        else:
            st.warning("설명(동시점) 모드: Xₜ와 Yₜ가 같은 기간이라 '오늘 다른 시장 움직임을 보고 오늘 수익에 베팅'하는 셈 → 실제 거래 불가. "
                       "전략 지표(적중률·Sharpe)는 계산하지 않습니다. 트레이딩 성과는 '예측' 모드에서 확인하세요.")
        if st.session_state.get("ml_vs"):
            st.caption("목표 = y/σₜ (변동성 표준화). R²·RMSE는 표준화 단위, 부호전략 Sharpe는 '변동성 역가중 포지션' 전략의 Sharpe.")
        wfres = st.session_state.get("ml_wf") or {}
        if wfres:
            st.markdown("##### Walk-forward 검증 (확장창 재추정, 기준예측 = 각 시점의 과거 평균)")
            wm = _wf_table(wfres)
            wpl = st.session_state.get("ml_wf_pl")
            if wpl:
                wm.loc["위약 · 무작위 부호 95% 분위", ["부호전략 Sharpe", "부호전략 Sharpe (비용 후)"]] = [wpl["single95_gross"], wpl["single95_net"]]
                wm.loc[f"위약 · 모델 {wpl['n_models']}개 중 최댓값 95% 분위", ["부호전략 Sharpe", "부호전략 Sharpe (비용 후)"]] = [wpl["max95_gross"], wpl["max95_net"]]
            st.dataframe(wm[["OOS R² (vs 학습평균)", "방향 적중률", "부호전략 Sharpe", "부호전략 Sharpe (비용 후)", "Buy&Hold Sharpe",
                             "롱 비중", "연간 회전율", "재추정 횟수"]].style.format("{:.4f}", na_rep="–").apply(
                lambda r: ["color:#898781;font-style:italic" if r.name.startswith("위약") else "" for _ in r], axis=1))
            if wpl:
                real_ = wm.drop(index=[i for i in wm.index if i.startswith("위약")])["부호전략 Sharpe (비용 후)"]
                bw = real_[real_ > wpl["max95_net"]]
                st.caption(f"**walk-forward 위약 기준선** (같은 walk-forward 평가 구간, {wpl['n_models']}개 모델 중 최고를 고르는 경우): "
                           f"비용 후 Sharpe가 {wpl['max95_net']:+.2f}를 넘어야 운과 구별 → "
                           + (f"넘은 모델: **{', '.join(bw.index)}**" if len(bw) else f"**넘은 모델 없음** (최고 {real_.max():+.2f} = {real_.idxmax()})"))
            f = go.Figure()
            for j, (k, (p_, y_, m_)) in enumerate(wfres.items()):
                s_ = pd.Series(A.band_positions(p_.iloc[::D.horizon].values, getattr(D, "band", 0.0)) * y_.iloc[::D.horizon].values,
                               index=y_.index[::D.horizon]).cumsum()
                f.add_trace(go.Scatter(x=s_.index, y=s_.values, name=k, line=dict(color=PAL[j % 8], width=1.6)))
            y0 = next(iter(wfres.values()))[1]
            bh = y0.iloc[::D.horizon].cumsum()
            f.add_trace(go.Scatter(x=bh.index, y=bh.values, name="Buy & Hold", line=dict(color="#898781", width=1.2, dash="dot")))
            bh_sr = y0.iloc[::D.horizon].mean() / y0.iloc[::D.horizon].std() * np.sqrt(ANN / D.horizon)
            st.plotly_chart(shade(fig_layout(f, 340, title=f"Walk-forward 부호전략 누적성과 (비용 전, Buy&Hold Sharpe {bh_sr:.2f})"), bh.index), key="wfcum")

        rid = st.session_state.get("ml_run_id")
        if rid:
            import runlog as RL
            with st.container(border=True):
                v1, v2, v3, v4 = st.columns([2.4, 1.2, 3, 0.9])
                v1.markdown(f"**📌 이 실행은 자동 기록됨**  \n`{rid}`  \n⑥ 실험 결과 → '📂 내 실행 기록'에서 비교·설정 불러오기")
                vv = v2.selectbox("판정", list(RL.VERDICTS), format_func=RL.VERDICTS.get, key="ml_verdict")
                nt = v3.text_input("메모 (왜 좋은가 / 나쁜가)", key="ml_note")
                if v4.button("판정 저장", key="ml_vsave"):
                    RL.set_verdict(rid, vv, nt); st.toast(f"판정 저장: {RL.VERDICTS[vv]}")

        lin = {k: v["coef"] for k, v in out.items() if v["coef"] is not None}
        if lin:
            CM = pd.DataFrame(lin).T
            CM.columns = [c.replace(c.split(" (")[0], short(c.split(" (")[0])) for c in CM.columns]
            vmax = float(np.nanmax(np.abs(CM.values))) or 1
            st.markdown("##### 계수 비교 (표준화 계수: X 1σ 변화당 Y 변화)")
            fig = go.Figure(go.Heatmap(z=CM.values, x=list(CM.columns), y=list(CM.index), zmin=-vmax, zmax=vmax,
                                       colorscale=DIV, texttemplate="%{z:.2f}" if CM.shape[1] <= 14 else None,
                                       textfont=dict(size=9), hovertemplate="%{y} · %{x}: %{z:.4f}<extra></extra>"))
            fig.update_layout(height=max(260, 40 * len(CM) + 120), margin=dict(l=10, r=10, t=10, b=10),
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, key="coefheat")

        st.markdown("##### 모델 상세")
        sel = st.selectbox("모델", list(out), key="mlsel")
        o = out[sel]
        model_detail(sel, o, D, tgt_, out, hp)

        if md_ != "predict":
            return
        st.markdown(f"##### 테스트 구간 누적 성과 (부호 전략, 비용 {D.cost * 100:.1f}bp 차감)" + (f" — {D.horizon}기마다 리밸런싱(겹침 제거)" if D.horizon > 1 else ""))
        f = go.Figure()
        hz = D.horizon
        yraw = pd.Series(D.y_te + D.y_mu, index=D.idx_te).iloc[::hz]
        for j, (k, v) in enumerate(out.items()):
            if k == "분위수 회귀":
                continue
            pos = np.sign(pd.Series(np.asarray(v["pred_te"]) + D.y_mu, index=D.idx_te).iloc[::hz])   # 원 예측(학습평균 포함)의 부호
            net = pos * yraw - np.abs(pos.diff().fillna(pos)) * D.cost
            s_ = net.cumsum()
            f.add_trace(go.Scatter(x=s_.index, y=s_.values, name=k, line=dict(color=PAL[j % 8], width=2.6 if k.startswith("앙상블") else 1.6)))
        bh = yraw.cumsum()
        f.add_trace(go.Scatter(x=bh.index, y=bh.values, name="Buy & Hold", line=dict(color="#898781", width=1.2, dash="dot")))
        st.plotly_chart(shade(fig_layout(f, 360, yaxis_title="누적 로그수익률 (%)"), bh.index), key="cum")


def ml_meta():
    """④ 단일 품목 — 모델 선택 메타 전략 (시간·레짐별로 어떤 모델을 쓸지를 과거 OOS 성과만으로 선택)."""
    import metasel as MS
    import experiments as EX
    st.divider()
    st.markdown("##### 모델 선택 메타 전략 — 시기·레짐에 따라 어떤 모델을 쓸지 고르기 (중첩 walk-forward)")
    st.caption("후보 모델마다 walk-forward OOS 예측 경로를 만든 뒤, 매 시점 **이미 실현된 OOS 손실만으로** (a) 직전 N개월 손실 최소 모델, "
               "(b) 지수가중 역MSE 가중 결합, (c) 현재 레짐(HMM 3상태 필터·실현변동성 3분위·평균 쌍별 상관 3분위)에서 과거 성과가 가장 좋았던 모델을 "
               "고르고 (d) 동일가중 앙상블·(e) 과거 평균과 비교. 혼합 전략: (f) Hedge = 최근 실현 성과(t값)가 좋은 모델에 소프트맥스 비중 (망각으로 국면 전환 추종), "
               "(g) 레짐 게이트 전문가 혼합 = 지금 레짐 칸(변동성 3분위 × 상관 3분위)에서 잘 맞혔던 모델에 비중 (칸 표본이 적으면 전체 성과로 수축), "
               "(h) + 기권 = 어떤 모델도 그 국면에서 t값 임계(모델 수 보정, 6개면 1.73)를 못 넘으면 포지션 0. (h)는 대부분 기간 무포지션일 수 있어 위약 p·Sharpe 비교가 무의미할 수 있음 — '롱 비중'·'연간 회전율'로 실제 투자 비중을 먼저 확인. 위약 = 롱 비중·전환 빈도를 맞춘 무작위 부호, 스누핑 보정 = White RC · Hansen SPA. "
               "위의 '▶ 모델 실행'(예측 모드)에서 만든 설계(Y·X·시차·비용)를 사용합니다.")
    ml = st.session_state.get("ml")
    if not ml or ml[3] != "predict":
        st.info("먼저 '예측 (미래 수익률)' 모드로 '▶ 모델 실행'을 누르세요."); return
    D, _, tgt_, _ = ml
    m1, m2, m3, m4 = st.columns([3, 1, 1, 1])
    cands = m1.multiselect("후보 모델", [m for m in A.MODEL_LIST if m not in ("LSTM", "분위수 회귀")], MS.CANDIDATES, key="ms_models")
    nmo = m2.number_input("(a) 선택 창 (개월)", 1, 36, 6, key="ms_n")
    hl = m3.number_input("(b) 반감기 (개월)", 1, 36, 6, key="ms_hl")
    stp = m4.number_input("후보 재추정 주기", 5, 500, {"B": 126, "W-FRI": 26, "ME": 6}[FQ], key="ms_step")
    idx = D.raw.index
    oos0 = max(pd.Timestamp("2004-01-01"), idx[int(len(idx) * 0.2)])
    ev0 = oos0 + pd.DateOffset(years=4)
    st.caption(f"후보 OOS 시작 {oos0:%Y-%m} (이전은 첫 학습) → 메타 선택 예열 4년 → 평가 {ev0:%Y-%m}~{idx.max():%Y-%m}. "
               f"하위기간 2008~16 / 2017~. 비용 {D.cost * 100:.1f}bp.")
    if st.button("▶ 메타 전략 백테스트", key="ms_run") and len(cands) >= 2:
        prog = st.progress(0.0, "후보 모델 walk-forward 중")
        Lv = _levels(extra_bytes, ROLL)
        cids = list(dict.fromkeys([tgt_] + [i for i in EX.META_CORR if i in NAME]))
        Rd = _panel(tuple(cids), "B", "logret", start, end, extra_bytes, ROLL)
        st.session_state["ml_meta_res"] = MS.run(D, Lv, Rd, tgt_, ANN, int(stp), cands, oos_start=oos0, eval_start=ev0, n_months=int(nmo),
                                        halflife_months=int(hl), B=500, progress=lambda x: prog.progress(min(1.0, x * 0.95), "후보 모델 walk-forward 중"))
        prog.empty()
    o = st.session_state.get("ml_meta_res")
    if not o:
        return
    T = o["table"]
    st.dataframe(T.style.format({c: "{:.3f}" for c in T.columns if c != "구분"} | {"롱 비중": "{:.0%}", "연간 회전율": "{:.0f}",
                                 "OOS R² (vs 과거평균)": "{:+.4f}", "위약 p (단일)": "{:.2f}"}, na_rep="–"))
    pm = o["placebo_max"]
    st.markdown(f"**'후보 중 최댓값' 위약**: 관측 최고 Sharpe(비용 후) {pm['관측최댓값']:+.2f} ({pm['최고전략']}) vs 위약 최댓값 "
                f"중앙 {pm['위약최댓값중앙']:+.2f} · 95% 분위 {pm['위약최댓값95']:+.2f} → p = {pm['p']:.2f}"
                + (" — **운과 구별되지 않음**" if pm["p"] >= 0.05 else " — 위약 대비 유의"))
    st.dataframe(o["spa"].set_index("귀무가설 기준").style.format({"최고 평균차": "{:+.4f}", "White RC p": "{:.3f}", "Hansen SPA p": "{:.3f}"}))
    C = MS.cum_net(o["preds"], o["y"], D.horizon, D.cost)
    f = go.Figure()
    for j, c in enumerate(C.columns):
        meta = c.startswith("(")
        f.add_trace(go.Scatter(x=C.index, y=C[c], name=c, line=dict(color="#898781" if c == "Buy & Hold" else PAL[j % 8],
                                                                     width=2.4 if meta else 1.0, dash="dot" if c == "Buy & Hold" else None),
                               visible=True if meta or c == "Buy & Hold" else "legendonly"))
    st.plotly_chart(shade(fig_layout(f, 380, title=f"누적 순수익 (부호 전략, 비용 {D.cost * 100:.1f}bp 차감) — 후보 모델은 범례 클릭으로 표시",
                                     yaxis_title="누적 로그수익률 (%)"), C.index), key="ms_cum")
    with st.expander("메타 전략이 고른 모델 비중"):
        sh = o["pick"].apply(lambda s: s.value_counts(normalize=True)).T.fillna(0)
        st.dataframe(sh.style.format("{:.0%}"))

# ============================================================================
# ⑤ 포트폴리오 백테스트 (멀티 원자재 바스켓, Newton-Raphson 최적 비중)
# ============================================================================
def tab_portfolio():
    import portfolio as P
    st.subheader("원자재 바스켓 포트폴리오 백테스트 — Newton-Raphson 최적 비중")
    st.caption(f"신호·공분산 = 사이드바 데이터 주기({freq_lbl}) · 리밸런싱 주기는 아래에서 별도 선택 (데이터 주기와 같거나 느리게). "
               "결정일 종가까지의 정보로 비중 결정 → **다음 거래일 종가 실행** → 다음 실행일까지 보유하며 **손익은 매일 추적**(계약 수 고정, 비중 표류). "
               "Sharpe·변동성·낙폭은 모두 일간 수익률 기준이라 리밸런싱 주기 간 비교 가능.")
    REB_OPTS = {"매일": "B", "매주 (금요일)": "W-FRI", "매월 (월말)": "ME", "분기 (분기말)": "QE"}
    reb_choices = [k for k, v in REB_OPTS.items() if P.REB_DAYS[v] >= P.REB_DAYS[FQ]]
    reb_l = st.selectbox("리밸런싱 주기", reb_choices, 0, key="pf_reb",
                         help="일간 데이터로 신호를 계산해도 거래는 주·월 단위로만 할 수 있음 (거래비용 절감). 최적화의 거래비용 항은 보유기간에 맞춰 조정됨.")
    reb = REB_OPTS[reb_l]
    basis_l = st.radio("손익 기준", ["ETF 기반 선물 초과수익 (실제 롤 비용 반영, 권장)", "Yahoo 연속월물 (롤 보정)"], key="pf_basis", horizontal=True,
                       help="Yahoo 연속월물은 롤 비용(콘탱고)이 대부분 빠져 수익이 연 2~25%p 과대(WTI +8.6%p, 천연가스 +24%p, 금 +1.9%p — ETF 대비 검증). "
                            "ETF 기준 = ETF 총수익 − 3개월 국채금리 + ETF 보수 ≈ 실제 롤을 수행한 선물 초과수익. 거래비용은 두 기준 모두 선물 틱 기준.")
    basis = "etf" if basis_l.startswith("ETF") else "futures"
    if basis == "etf":
        avail = [i for i in P.ETF_MAP if i in NAME and P.ETF_MAP[i][0] in NAME]
        st.caption("ETF 기준 유니버스: WTI(USO)·Brent(BNO)·천연가스(UNG)·휘발유(UGA)·옥수수(CORN)·대두(SOYB)·밀(WEAT)·설탕(CANE)·구리(CPER)·금(GLD)·은(SLV)·백금(PPLT)·팔라듐(PALL). "
                   "ETF 상장 시점부터 순차 편입 (2010~2011년 다수). 대두박·생우·커피 등은 상장 ETF가 없어 제외.")
    else:
        avail = [i for i in P.DEFAULT_UNIVERSE + [k for k in P.SPECS if k not in P.DEFAULT_UNIVERSE] if i in NAME]
        st.warning("⚠ 연속월물 기준은 롤 비용이 과소반영되어 수익률·Sharpe가 과대평가됩니다 (롤 보정 13개 품목도 ETF 대비 연 4~8%p 과대). 비교·참고용으로만 사용하세요.")
    uni = st.multiselect("유니버스 (원자재 선물)", avail, default=avail if basis == "etf" else [i for i in P.DEFAULT_UNIVERSE if i in NAME],
                         format_func=lbl, key=f"pf_uni_{basis}")
    c1_, c2_, c3_ = st.columns(3)
    sig = c1_.radio("기대수익 신호 μ", ["시계열 모멘텀 (기간 선택)", "Ridge 모델 예측 (특성)", "과거 평균 (무정보 기준)",
                                         "패널 모델 예측 (④ 탭 '여러 품목'에서 전송)"], key="pf_sig",
                    help="모멘텀: α = IC × σ × score(선택 기간 수익/변동성, ±2 제한).  Ridge: 종목별 walk-forward 예측(모멘텀·변동성·계절성·COT·베이시스).  과거 평균: 확장창 평균.")
    sigk = {"시계열": "momentum", "Ridge": "ridge", "과거": "hist_mean", "패널": "panel"}[sig.split()[0]]
    pfc = st.session_state.get("panel_fc")
    if sigk == "panel":
        if pfc is None:
            st.warning("⑥ 패널 모델 탭에서 모델을 실행하고 '포트폴리오 탭 신호로 보내기'를 먼저 누르세요.")
        else:
            st.info(f"패널 신호: {pfc['model']} · {pfc['basis']} 기준 · {len(pfc['uni'])}종목 · 데이터 주기 {pfc['freq']}"
                    + ("" if pfc["freq"] == FQ else " ⚠ 사이드바 주기와 다름 — 같은 주기로 맞추세요"))
    mode_l = c2_.radio("제약", ["롱온리 (Σw=1, w≥0)", "롱숏 (총노출 Σ|w| ≤ L)"], key="pf_mode")
    pmode = "long_only" if mode_l.startswith("롱온리") else "long_short"
    capital = c3_.number_input("투자금 (USD)", 100_000, 1_000_000_000, 10_000_000, 1_000_000, key="pf_cap")
    d1_, d2_, d3_, d4_, d5_ = st.columns(5)
    lam = d1_.number_input("위험회피계수 λ", 0.5, 100.0, 5.0, 0.5, key="pf_lam", help="클수록 분산을 강하게 벌점 → 분산투자·저변동")
    gross = d2_.number_input("총노출 한도 L (롱숏)", 0.25, 5.0, 1.0, 0.25, key="pf_L", disabled=pmode == "long_only",
                             help="정해진 투자금 대비 선물 명목금액 합계 한도. 1.0 = 레버리지 없음")
    ic = d3_.number_input("IC (모멘텀)", 0.0, 0.3, 0.05, 0.01, key="pf_ic", disabled=sigk != "momentum",
                          help="정보계수: 신호와 다음 기간 수익률의 예상 상관. 0.02~0.10이 현실적")
    covw = d4_.number_input("공분산 창 (기간)", 20, 1000, {"B": 252, "W-FRI": 104, "ME": 60}[FQ], key="pf_cov",
                            help="Ledoit-Wolf 수축 공분산, 결정일까지의 과거 창")
    liq = d5_.number_input("유동성 비용 배수", 0.0, 5.0, 1.0, 0.25, key="pf_liq", help="1 = 품목별 기본 틱 수(반스프레드+슬리피지). 2 = 비용 2배 스트레스")
    k1_, k2_ = st.columns(2)
    comm = k1_.number_input("계약당 수수료 (USD, 편도)", 0.0, 20.0, P.COMMISSION, 0.5, key="pf_comm")
    w_max = k2_.number_input("종목별 최대 비중 |wᵢ| ≤", 0.05, 1.0, 0.25, 0.05, key="pf_wmax",
                             help="상한이 없으면 평균-분산 최적화가 소수 종목(예: 금 40~70%)에 쏠림 → 결과가 한 종목 베팅이 됨. Newton 배리어로 처리.")
    m1_, m2_, m3_, m4_, m5_ = st.columns(5)
    mom_spec = m1_.selectbox("모멘텀 기간", list(P.MOM_SPECS), 0, key="pf_mom", disabled=sigk != "momentum",
                             help="데이터 주기와 무관하게 '달력 기간'으로 환산 (일간 252일 = 주간 52주 = 12개월). 업계 표준 = 12-1개월 "
                                  "(최근 1개월 제외: 단기 반전 회피). 1주·3주는 원자재에서 반전이 섞이는 구간. CTA 혼합 = EWMA 교차 3쌍 평균.")
    RULES = {"고정 주기 (점검일마다 목표로)": "fixed", "무거래 구간 (필요할 때만 거래)": "band", "부분 조정 (차이의 κ만큼)": "partial"}
    rule_l = m2_.selectbox("거래 규칙", list(RULES), 0, key="pf_rule",
                           help="무거래 구간: 매 점검일 목표 비중을 계산하되 Σ|목표−현재| > 폭 이거나 최대 보유기간이 지나면 거래. "
                                "부분 조정: 매번 차이의 κ만큼만 이동 (Gârleanu-Pedersen 2013). 두 방식 모두 리밸런싱 주기 '매일'과 함께 사용 권장.")
    rule = RULES[rule_l]
    band = m3_.number_input("무거래 폭 Σ|목표−현재|", 0.01, 1.0, 0.10, 0.01, key="pf_band", disabled=rule != "band")
    max_hold = m4_.number_input("최대 보유 (영업일)", 1, 252, 21, key="pf_maxhold", disabled=rule != "band",
                                help="이 기간이 지나면 차이가 작아도 목표로 재조정 (21 = 늦어도 월 1회)")
    kappa = m5_.number_input("부분 조정 κ", 0.05, 1.0, 0.3, 0.05, key="pf_kappa", disabled=rule != "partial")
    v1_, v2_, v3_ = st.columns(3)
    VMS = {"과거 창 공분산 (Ledoit-Wolf)": "lw", "EWMA (λ=0.94)": "EWMA", "HAR-RV": "HAR-RV", "MIDAS-RV (권장)": "MIDAS-RV"}
    vm_l = v1_.selectbox("품목 변동성 예측", list(VMS), 0, key="pf_vm",
                         help="상관구조는 과거 창(Ledoit-Wolf)에서, 품목별 변동성만 일간 walk-forward 예측으로 교체. "
                              "MIDAS-RV는 일간 실현분산 252개 시차를 Beta 가중으로 요약 — ◆ 리스크 '변동성 모형 검증'의 예측 경주 참고.")
    vt = v2_.number_input("목표 변동성 (연, 0 = 사용 안 함)", 0.0, 0.5, 0.0, 0.01, format="%.2f", key="pf_vt",
                          help="예측 포트폴리오 변동성이 목표가 되도록 전체 노출을 배율 조정 (변동성 타기팅). 위기 때 자동 축소, 평온기 확대.")
    mlev = v3_.number_input("최대 배율", 0.5, 5.0, 2.0, 0.25, key="pf_mlev", disabled=vt == 0)
    rb_tilt = st.slider("리스크 버짓 신호 기울기 τ (4번째 전략)", 0.0, 1.0, 0.5, 0.05, key="pf_rbt",
                        help="위험 예산 bᵢ = (1−τ)/N + τ·(신호 강도 비례). τ=0 = 리스크 패리티, τ=1 = 위험을 신호 강도에 비례 배분. "
                             "예측력이 약할수록 작게 — 리스크 패리티의 안정성과 최적화의 신호 활용 사이의 절충.")
    if rule != "fixed" and reb != "B":
        st.caption(f"ℹ️ 현재 점검 주기 = {reb_l}. '필요할 때만 거래'의 이점은 매일 점검할 때 가장 큼 (리밸런싱 주기 '매일' 권장).")
    with st.expander("품목별 거래비용 가정 (현재 가격 기준)"):
        Lc_ = _levels(extra_bytes, False)
        last_px = pd.DataFrame({i: [Lc_[i].dropna().iloc[-1]] for i in uni if i in Lc_})
        if len(last_px.columns):
            tcx, rcx = P.cost_frames(last_px, liq, comm)
            ct = pd.DataFrame({"품목": [short(i) for i in last_px.columns], "승수": [P.SPECS[i]["mult"] for i in last_px.columns],
                               "1틱 (USD)": [P.SPECS[i]["tick"] for i in last_px.columns], "유동성 틱": [P.SPECS[i]["liq"] * liq for i in last_px.columns],
                               "계약 명목 (USD)": [last_px[i].iloc[0] * P.SPECS[i]["scale"] * P.SPECS[i]["mult"] for i in last_px.columns],
                               "편도 매매비용 (bp)": tcx.iloc[0].values * 1e4, "연간 롤 횟수": [P.SPECS[i]["rolls"] for i in last_px.columns],
                               "연간 롤비용 (bp)": rcx.iloc[0].values * 1e4})
            st.dataframe(ct.style.format({"계약 명목 (USD)": "{:,.0f}", "편도 매매비용 (bp)": "{:.1f}", "연간 롤비용 (bp)": "{:.1f}", "1틱 (USD)": "{:.2f}"}), hide_index=True)
            st.caption("편도 매매비용 = (유동성 틱 × 1틱 가치 + 수수료) ÷ 계약 명목금액 — 가격이 오르면 비율이 낮아짐(백테스트는 매 시점 가격으로 재계산). "
                       "롤비용 = 연간 롤 횟수 × (1틱 + 수수료×2) ÷ 명목, 보유 비중에 비례해 매 기간 차감.")
    if st.button("▶ 백테스트 실행", type="primary", key="pf_run"):
        if sigk == "panel" and (pfc is None or pfc["freq"] != FQ):
            st.warning("패널 신호가 없거나 데이터 주기가 다릅니다."); return
        if len(uni) < 2:
            st.warning("2개 이상 선택하세요.")
        else:
            prog = st.progress(0.0, "백테스트 중 (Ridge 신호는 종목별 walk-forward 학습 포함)")
            try:
                st.session_state["pf"] = P.backtest(_levels(extra_bytes, True), _levels(extra_bytes, False), uni, UNITS, FQ, start, end,
                                                    signal=sigk, mode=pmode, lam=lam, gross=gross, ic=ic, cov_window=int(covw),
                                                    capital=float(capital), liq_mult=liq, comm=comm, basis=basis, w_max=float(w_max), rebalance=reb,
                                                    mom_spec=mom_spec, trade_rule=rule, band=float(band), max_hold=int(max_hold), kappa=float(kappa),
                                                    vol_model=VMS[vm_l], vol_target=float(vt) or None, max_lev=float(mlev), rb_tilt=float(rb_tilt), forecasts=pfc['fc'] if (sigk == 'panel' and pfc) else None,
                                                    progress=lambda x: prog.progress(min(1.0, x)))
            except Exception as ex:
                st.error(f"백테스트 실패: {ex}")
            prog.empty()
    if "pf" not in st.session_state:
        return
    bt = st.session_state["pf"]; res = bt["results"]; s_ = bt["settings"]
    if not len(next(iter(res.values()))["ret"]):
        st.warning("유효한 리밸런싱 구간이 없습니다 (이력·공분산 창 부족)."); return
    ppy = bt["ppy"]
    bench_lbl = {"동일가중 1/N": "동일가중 1/N" if s_["mode"] == "long_only" else "신호부호 × 동일비중",
                 "리스크 패리티(Newton)": "리스크 패리티(Newton)" if s_["mode"] == "long_only" else "신호부호 × 리스크패리티",
                 "리스크 버짓(신호 기울기)": f"리스크 버짓 (신호 기울기 τ={s_.get('rb_tilt', 0.5):.2f})"}
    names = {k: bench_lbl.get(k, k) for k in res}
    st.caption("벤치마크 'DBC 보유' = 실제 투자 가능한 원자재 지수 ETF(DBC)를 사서 보유한 초과수익(리밸런싱·추가비용 없음, 보수 반영).")
    stt = pd.DataFrame({names[k]: v["stats"] for k, v in res.items()}).T
    st.markdown("##### 성과 (비용 차감 후, 선물 초과수익 기준 — " + ("ETF 기반, 실제 롤 비용 반영" if s_.get("basis") == "etf" else "⚠ 연속월물, 롤 비용 과소반영") + ")")
    pct_cols = ["연환산 수익률(CAGR)", "연환산 변동성", "최대낙폭(MDD)", "연간 비용 부담", "승률(일간)"]
    fmt = {c: "{:.2%}" for c in pct_cols}
    fmt.update({c: "{:.2f}" for c in ["Sharpe (비용 후)", "Sharpe (비용 전)", "Sortino", "Calmar", "연간 회전율(편도 합)", "연간 리밸런싱 횟수", "평균 총노출", "평균 순노출"]})
    st.dataframe(stt.style.format(fmt, na_rep="–"))
    dg = bt["diag"]
    st.caption(f"리밸런싱 {len(dg):,}회 ({ {'B': '매일', 'W-FRI': '매주', 'ME': '매월', 'QE': '분기'}[bt.get('rebalance', FQ)] }, 데이터 {freq_lbl}) · 평균 편입 {dg['n'].mean():.1f}종목 · Newton 수렴 {dg['converged'].mean():.1%}, "
               f"평균 반복 {dg['iter'].mean():.0f}회(배리어 10단계 합), 최대 KKT 잔차 {dg['kkt'].max():.1e} · "
               f"λ={s_['lam']}, {'L=' + str(s_['gross']) if s_['mode'] != 'long_only' else '완전투자'}, 종목 상한 {s_.get('w_max', 1.0):.0%}, 공분산창 {s_['cov_window']}, "
               f"비용 배수 {s_['liq_mult']}, 수수료 ${s_['comm']}")
    f = go.Figure()
    for j, (k, v) in enumerate(res.items()):
        f.add_trace(go.Scatter(x=v["nav"].index, y=v["nav"].values, name=names[k],
                               line=dict(color=PAL[j] if not k.startswith("DBC") else "#898781", width=2 if j == 0 else 1.4,
                                         dash="dot" if k.startswith("DBC") else None)))
    f.update_yaxes(type="log", title="NAV (USD, 로그)")
    st.plotly_chart(shade(fig_layout(f, 380, title=f"포트폴리오 가치 — 초기 ${bt['capital']:,.0f}"), v["nav"].index), key="pf_nav")
    e1_, e2_ = st.columns(2)
    f = go.Figure()
    for j, (k, v) in enumerate(res.items()):
        dd = v["nav"] / v["nav"].cummax() - 1
        f.add_trace(go.Scatter(x=dd.index, y=dd.values * 100, name=names[k], line=dict(color=PAL[j], width=1.3)))
    e1_.plotly_chart(shade(fig_layout(f, 300, title="낙폭 (%)"), dd.index), key="pf_dd")
    f = go.Figure()
    for j, (k, v) in enumerate(res.items()):
        rs = v["ret"].rolling(ppy).mean() / v["ret"].rolling(ppy).std() * np.sqrt(ppy)
        f.add_trace(go.Scatter(x=rs.index, y=rs.values, name=names[k], line=dict(color=PAL[j], width=1.3)))
    f.add_hline(y=0, line_color="#898781", line_width=1)
    e2_.plotly_chart(shade(fig_layout(f, 300, title="롤링 1년 Sharpe (비용 후)"), rs.index), key="pf_rs")
    W = res["최적화(Newton)"]["W"]
    st.caption(f"최적화 포트폴리오 평균 유효 종목 수(1/Σw²) {1 / (W ** 2).sum(axis=1).replace(0, np.nan).mean():.1f} — 3 미만이면 사실상 소수 종목 베팅")
    Wm = W.resample("ME").last() if ppy > 12 else W
    Wm = Wm.loc[:, (W.abs() > 1e-6).any()]
    vm = float(Wm.abs().values.max()) or 1
    f = go.Figure(go.Heatmap(z=Wm.T.values, x=Wm.index, y=[short(c) for c in Wm.columns], zmin=-vm if s_["mode"] != "long_only" else 0, zmax=vm,
                             colorscale=DIV if s_["mode"] != "long_only" else SEQ, hovertemplate="%{y} %{x|%Y-%m}: %{z:.1%}<extra></extra>"))
    f.update_layout(height=120 + 22 * Wm.shape[1], margin=dict(l=10, r=10, t=40, b=10), title="최적 비중 추이 (월말 스냅샷)")
    st.plotly_chart(f, key="pf_w")
    g1_, g2_ = st.columns(2)
    wl = W.iloc[-1]; wl = wl[wl.abs() > 1e-6].sort_values()
    f = go.Figure(go.Bar(x=wl.values * 100, y=[short(i) for i in wl.index], orientation="h",
                         marker_color=[PAL[0] if v >= 0 else PAL[1] for v in wl.values]))
    g1_.plotly_chart(fig_layout(f, max(300, 22 * len(wl) + 80), title=f"최근 비중 (%) — {W.index[-1]:%Y-%m-%d}", hovermode="y"), key="pf_wl")
    f = go.Figure()
    for j, (k, v) in enumerate(res.items()):
        cc = v["cost"].cumsum() * 100
        f.add_trace(go.Scatter(x=cc.index, y=cc.values, name=names[k], line=dict(color=PAL[j], width=1.4)))
    g2_.plotly_chart(fig_layout(f, 300, title="누적 거래·롤 비용 (NAV 대비 %, 단순합)"), key="pf_cost")
    st.caption("해석 주의: 선물은 증거금 거래이므로 수익률은 '초과수익'(담보 현금의 이자 미포함). 계약 단위 반올림·증거금 제약·세금은 반영하지 않음(분수 계약 가정). "
               "롤 보정 품목 선정(13개)은 전체 표본으로 검증한 규칙을 사용 — 데이터 전처리 단계의 경미한 사후정보.")


# ============================================================================
# ⑥ 패널 모델 (품목 전체를 함께 학습)
# ============================================================================
@st.cache_data(show_spinner=False)
def _panel_data(basis: str, uni: tuple, freq: str, target: str, extra_bytes, roll: bool):
    import panel as PN
    return PN.build_panel(_levels(extra_bytes, True), _levels(extra_bytes, False), list(uni), UNITS, freq, basis, target)


def tab_panel():
    import panel as PN
    import portfolio as P
    st.subheader("원자재 패널 모델 — 모든 품목을 함께 학습해 '어느 품목이 상대적으로 오를까'를 예측")
    st.caption("품목별 특성(모멘텀·변동성·왜도·실현 롤수익·COT·계절성)을 매 시점 품목 간 순위로 바꿔 하나의 모델로 학습 (Gu-Kelly-Xiu 2020 방식). "
               "표본이 품목 수만큼 늘어나고, 평가는 순위 IC와 상·하위 1/3 롱숏 포트폴리오(다음 거래일 실행, 비용 반영)로 함. "
               f"데이터 주기 = 사이드바({freq_lbl}), walk-forward 확장창(재학습 시점 이전에 목표가 실현된 행만 학습).")
    c1_, c2_ = st.columns([1, 2])
    basis_l = c1_.radio("수익 기준", ["ETF 기반 (실제 롤 비용 반영, 13종목·2010~)", "연속월물 롤보정 (19종목·2000~, 수익 과대 가능)"], key="pn_basis")
    basis = "etf" if basis_l.startswith("ETF") else "futures"
    avail = list(P.ETF_MAP) if basis == "etf" else [i for i in P.DEFAULT_UNIVERSE + [k for k in P.SPECS if k not in P.DEFAULT_UNIVERSE]]
    avail = [i for i in avail if i in NAME]
    uni = c2_.multiselect("유니버스", avail, default=avail if basis == "etf" else [i for i in P.DEFAULT_UNIVERSE if i in NAME],
                          format_func=lbl, key=f"pn_uni_{basis}")
    chars = st.multiselect("특성", PN.CHARS, default=PN.CHARS, key="pn_chars")
    d1_, d2_, d3_, d4_ = st.columns(4)
    tgt_l = d1_.selectbox("목표", ["변동성 표준화 + 시점별 평균 제거 (상대 순위)", "변동성 표준화", "원 수익률"], key="pn_tgt")
    target = {"변동성 표준화 + 시점별 평균 제거 (상대 순위)": "vol_cs", "변동성 표준화": "vol", "원 수익률": "raw"}[tgt_l]
    refit = d2_.number_input("재학습 주기 (기간)", 1, 500, {"B": 126, "W-FRI": 26, "ME": 6}[FQ], key="pn_refit")
    min_train = d3_.number_input("최소 학습 기간", 20, 3000, {"B": 504, "W-FRI": 104, "ME": 24}[FQ], key="pn_mt")
    q = d4_.select_slider("롱숏 분위", [0.2, 0.25, 1 / 3, 0.5], 1 / 3, format_func=lambda v: f"상·하위 {v:.0%}", key="pn_q")
    models = st.multiselect("모델", PN.PANEL_MODELS, default=["Pooled OLS", "Ridge", "Elastic Net", "Gradient Boosting"], key="pn_models")
    o1_, o2_ = st.columns(2)
    wf_fe = PN.FE_OPTS[o1_.selectbox("예측 고정효과 (선형 모델)", list(PN.FE_OPTS), 0, key="pn_wffe",
                                     help="품목 FE: 품목별 평균 수익(학습구간)을 예측에 더함 — 품목 간 지속적 차이를 반영. "
                                          "시간 FE: 매 시점 횡단면 평균 제거 (목표가 이미 평균 제거면 효과 작음).")]
    wf_sec = o2_.checkbox("섹터별 기울기 (특성 × 섹터 상호작용)", False, key="pn_wfsec",
                          help="공통 기울기 + 섹터별 추가 기울기. Ridge·Lasso에서는 섹터 차이가 0 쪽으로 축소됨 (계층적 축소).")
    b1_, b2_ = st.columns(2)
    run = b1_.button("▶ 패널 모델 실행", type="primary", key="pn_run")
    run_f = b2_.button("단일 특성 팩터 분석", key="pn_factor")
    if len(uni) < 4:
        st.info("4개 이상 품목을 선택하세요."); return
    with st.spinner("패널 데이터 구성 중..."):
        pan = _panel_data(basis, tuple(uni), FQ, target, extra_bytes, ROLL)
    X = pan["X"]
    st.caption(f"패널: {X.index.get_level_values('date').nunique():,}기간 × {len(pan['ids'])}종목 = {len(X):,}행 "
               f"(품목별 모델이면 품목당 약 {X.index.get_level_values('date').nunique():,}행)")
    La_, Lr_ = _levels(extra_bytes, True), _levels(extra_bytes, False)
    if run and models and chars:
        res = {}
        prog = st.progress(0.0)
        for j, m in enumerate(models):
            prog.progress(j / len(models), f"{m} walk-forward 중...")
            wf = PN.walk_forward_panel(pan, m, chars, int(refit), int(min_train), fe=wf_fe, sector=wf_sec)
            if wf["pred"].empty:
                st.warning(f"{m}: 예측 없음 (최소 학습 기간이 너무 김)"); continue
            res[m] = dict(wf=wf, ev=PN.evaluate(pan, wf["pred"], La_, Lr_, q=q))
        prog.empty()
        st.session_state["pn"] = dict(res=res, basis=basis, uni=uni, freq=FQ, target=target, chars=chars)
    if run_f:
        with st.spinner("특성별 롱숏 계산 중..."):
            st.session_state["pn_ft"] = PN.factor_table(pan, La_, Lr_, chars, start=pan["dates"][min(int(min_train), len(pan["dates"]) - 1)])
    if "pn_ft" in st.session_state:
        st.markdown("##### 특성별 단일 팩터 (순위 상·하위 1/3 롱숏, 비용 후)")
        ft = st.session_state["pn_ft"]
        st.dataframe(ft.style.format({"평균 IC": "{:+.3f}", "IC t값": "{:+.2f}", "IC>0 비율": "{:.1%}", "롱숏 Sharpe(비용 후)": "{:+.2f}",
                                      "롱숏 연수익": "{:+.2%}", "연간 회전율": "{:.1f}"}, na_rep="–"))
        st.caption("IC t값 |t|>2 이면 통계적으로 유의. 부호가 음수인 특성은 반대로 쓰면 수익(예: 왜도·변동성은 낮은 쪽이 고수익이라는 가설과 비교).")
    with st.expander("📐 계량 패널 추정 (전체 표본, 추론용) — Fama-MacBeth · 고정효과 OLS · 섹터별 기울기", expanded=False):
        st.caption("예측(walk-forward)과 달리 전체 표본으로 '특성이 수익을 설명하는가'를 검정. 특성들을 동시에 넣은 다변량 회귀라, "
                   "위 단일 팩터 표(하나씩)와 결과가 다를 수 있음 (특성 간 상관). 목표 = 위에서 선택한 목표.")
        q1_, q2_, q3_ = st.columns(3)
        e_fe = PN.FE_OPTS[q1_.selectbox("고정효과", list(PN.FE_OPTS), 1, key="pn_efe")]
        e_cl = PN.CLUSTER_OPTS[q2_.selectbox("표준오차", list(PN.CLUSTER_OPTS), 2, key="pn_ecl",
                                            help="시점 군집: 같은 주의 공통 충격 / 품목 군집: 품목 내 자기상관 / Driscoll-Kraay: 둘 다 + 시차 상관")]
        e_tg = q3_.radio("목표", ["모델 목표 (변동성 표준화)", "원 수익률 (%)"], key="pn_etg", horizontal=True)
        tg = "y" if e_tg.startswith("모델") else "r"
        n_ent = len(pan["ids"]); k_max = max(1, n_ent - 3)
        pref = [c for c in ["CARRY 12M (실현 롤수익)", "MOM 12M", "VOL 3M", "SKEW 12M", "MOM 1M", "COT 상업HP z"] if c in chars]
        pref += [c for c in chars if c not in pref]
        e_ch = st.multiselect(f"추정에 넣을 특성 (최대 {k_max}개 = 품목 {n_ent} − 3)", chars, pref[:min(len(pref), k_max, 6)], key="pn_ech",
                              max_selections=k_max,
                              help="Fama-MacBeth는 매 시점 품목들로 횡단면 회귀를 하므로 특성 수 + 상수 < 품목 수여야 함. 소수 핵심 특성 권장.")
        if st.button("▶ 계량 추정 실행", key="pn_econ", disabled=not e_ch):
            with st.spinner("추정 중..."):
                fm = PN.fama_macbeth(pan, e_ch, tg)
                cmp_ = {k: PN.panel_ols(pan, e_ch, v, e_cl, tg)["table"]["t값"] for k, v in PN.FE_OPTS.items()}
                po = PN.panel_ols(pan, e_ch, e_fe, e_cl, tg)
                ss = PN.sector_slopes(pan, e_ch, e_fe, e_cl, tg)
            st.session_state["pn_econ_res"] = dict(fm=fm, cmp=pd.DataFrame(cmp_), po=po, ss=ss, fe=e_fe, cl=e_cl)
        if "pn_econ_res" in st.session_state:
            E = st.session_state["pn_econ_res"]
            g1_, g2_ = st.columns(2)
            g1_.markdown(f"**Fama-MacBeth** ({E['fm']['periods']:,}기간 횡단면 회귀 평균, Newey-West t)")
            if E["fm"].get("dropped"):
                g1_.caption("대부분 시점에서 횡단면 변동이 없어 제외: " + ", ".join(E["fm"]["dropped"]))
            g1_.dataframe(E["fm"]["table"].style.format("{:+.4f}").apply(
                lambda r: ["font-weight:700" if abs(r["NW t값"]) > 2 else "" for _ in r], axis=1))
            g2_.markdown(f"**패널 OLS** (고정효과 {E['fe']}, {E['cl']} SE) — 관측 {E['po']['nobs']:,}, within R² {E['po']['r2_within']:.4f}")
            g2_.dataframe(E["po"]["table"].style.format("{:+.4f}", na_rep="–"))
            if E["po"]["f_pooled"]:
                g2_.caption(f"고정효과 유의성 F = {E['po']['f_pooled'][0]:.2f}, p = {E['po']['f_pooled'][1]:.3f} (p<0.05면 효과 존재 → Pooled보다 FE)")
            st.markdown("**고정효과별 t값 비교** — 품목 FE를 넣으면 약해지는 특성 = 효과가 '품목 간 지속적 차이'에서 나옴")
            st.dataframe(E["cmp"].style.format("{:+.2f}", na_rep="–").apply(
                lambda c: ["font-weight:700" if abs(v) > 2 else "" for v in c], axis=0))
            S2 = E["ss"]
            st.plotly_chart(heat(S2["tstat"], "섹터별 기울기 t값 (|t|>2 유의)", -4, 4, ".1f", h=120 + 34 * len(S2["tstat"]), text_max=40),
                            key="pn_sec_heat")
            st.markdown("**섹터 간 기울기 동일 검정 (Wald)** — p<0.05면 '모든 품목에 같은 기울기' 가정이 기각")
            st.dataframe(S2["wald"].style.format({"Wald 통계량": "{:.2f}", "자유도": "{:.0f}", "p값": "{:.3f}"}))
            st.caption("섹터별 표본: " + ", ".join(f"{k} {v:,}" for k, v in S2["n_by_sector"].items()))
    if "pn" not in st.session_state or not st.session_state["pn"]["res"]:
        return
    S = st.session_state["pn"]; res = S["res"]
    st.markdown("##### 모델 성과 (walk-forward 외표본)")
    tbl = pd.DataFrame({m: {"패널 OOS R²": r["ev"]["oos_r2"], "평균 순위 IC": r["ev"]["ic_mean"], "IC t값": r["ev"]["ic_t"],
                            "IC>0 비율": r["ev"]["ic_pos"], "롱숏 Sharpe(비용 후)": r["ev"]["ls_stats"].get("Sharpe (비용 후)", np.nan),
                            "롱숏 연수익": r["ev"]["ls_stats"].get("연환산 수익률(CAGR)", np.nan),
                            "롱숏 MDD": r["ev"]["ls_stats"].get("최대낙폭(MDD)", np.nan),
                            "연간 회전율": r["ev"]["ls_stats"].get("연간 회전율(편도 합)", np.nan),
                            "재학습 횟수": r["wf"]["n_refits"]} for m, r in res.items()}).T
    st.dataframe(tbl.style.format({"패널 OOS R²": "{:+.4f}", "평균 순위 IC": "{:+.3f}", "IC t값": "{:+.2f}", "IC>0 비율": "{:.1%}",
                                   "롱숏 Sharpe(비용 후)": "{:+.2f}", "롱숏 연수익": "{:+.2%}", "롱숏 MDD": "{:.1%}",
                                   "연간 회전율": "{:.1f}", "재학습 횟수": "{:.0f}"}, na_rep="–"))
    st.caption("패널 OOS R²는 기준예측 0 대비 (GKX). 순위 IC = 시점별 예측 순위와 실현 수익 순위의 Spearman 상관 평균 — 횡단면 모델의 핵심 지표 (0.02~0.05면 실무적으로 의미).")
    f = go.Figure()
    for j, (m, r) in enumerate(res.items()):
        nav = (1 + r["ev"]["ls"]["ret"]).cumprod()
        f.add_trace(go.Scatter(x=nav.index, y=nav.values, name=m, line=dict(color=PAL[j % 8], width=1.6)))
    f.add_hline(y=1, line_color="#898781", line_width=1)
    st.plotly_chart(shade(fig_layout(f, 340, title="상·하위 롱숏 누적 (비용 후, 시장 중립)"), nav.index), key="pn_nav")
    sel = st.selectbox("상세 모델", list(res), key="pn_sel")
    r = res[sel]
    e1_, e2_ = st.columns(2)
    ic = r["ev"]["ic"]; w_ = {"B": 252, "W-FRI": 52, "ME": 12}[S["freq"]]
    f = go.Figure([go.Bar(x=ic.index, y=ic.values, name="시점별 IC", marker_color="#c3c2b7"),
                   go.Scatter(x=ic.index, y=ic.rolling(w_, min_periods=w_ // 3).mean(), name="1년 이동평균", line=dict(color=PAL[0], width=2))])
    f.add_hline(y=0, line_color="#898781", line_width=1)
    e1_.plotly_chart(shade(fig_layout(f, 320, title=f"{sel} 순위 IC"), ic.index), key="pn_ic")
    imp = r["wf"]["importance"]
    if imp is not None:
        imp = imp.reindex(imp.abs().sort_values().index)
        f = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=[PAL[0] if v >= 0 else PAL[1] for v in imp.values]))
        e2_.plotly_chart(fig_layout(f, 320, title="특성 계수 (재학습 평균)" if sel != "Random Forest" else "특성 중요도", hovermode="y"), key="pn_imp")
    else:
        e2_.info("이 모델은 계수·중요도를 제공하지 않습니다 (선형 모델 또는 Random Forest 선택).")
    pred = r["wf"]["pred"]; last = pred.index.get_level_values("date").max()
    cur = pred.xs(last, level="date").sort_values()
    f = go.Figure(go.Bar(x=cur.values, y=[short(i) for i in cur.index], orientation="h",
                         marker_color=[PAL[0] if v >= cur.quantile(1 - q) else (PAL[1] if v <= cur.quantile(q) else "#c3c2b7") for v in cur.values]))
    st.plotly_chart(fig_layout(f, max(300, 24 * len(cur) + 80), title=f"최신 신호 — {last:%Y-%m-%d} 결정 기준 예측 순위 (파랑 = 매수 후보, 주황 = 매도 후보)",
                               hovermode="y"), key="pn_cur")
    if st.button("이 모델의 예측을 ⑤ 포트폴리오 탭 신호로 보내기", key="pn_send"):
        st.session_state["panel_fc"] = dict(fc=PN.forecasts_for_portfolio(_panel_data(S["basis"], tuple(S["uni"]), S["freq"], S["target"], extra_bytes, ROLL), pred),
                                            model=sel, basis=S["basis"], uni=S["uni"], freq=S["freq"])
        st.success(f"전송 완료: {sel} ({S['basis']}, {len(S['uni'])}종목). ⑤ 탭에서 신호 '패널 모델 예측'을 선택하세요.")


# ============================================================================
# ⑦ MIDAS (혼합 주기) · 변동성 예측
# ============================================================================
def tab_midas(vol_only: bool = True):
    import midas as M
    st.subheader("MIDAS — 일간 데이터로 월간 목표 예측 · GARCH-MIDAS (장기·단기 변동성 분해)")
    st.caption("MIDAS는 수백 개의 일간 시차를 Beta 가중치(모수 1~2개)로 요약해 월간 목표를 예측 — 월 평균으로 뭉개지 않고 최근 정보에 더 큰 비중. "
               "결정 시점 = 월말 종가, 모든 예측은 확장창 walk-forward (그 시점까지의 데이터로만 재추정).")
    ids_c = [i for i in COMMOD_IDS if GROUP.get(i) not in ("Freight",)]
    if vol_only:
        st.info("변동성 예측 경주(RiskMetrics·품목별 λ·HMM·Kalman·GARCH·HAR-RV·MIDAS-RV, QLIKE·DM·MCS 검증)는 "
                "**◆ 리스크 → '변동성 모형 검증'** 탭으로 일원화했습니다. 여기서는 GARCH-MIDAS 장기·단기 분해만 다룹니다.")
        sec = "② GARCH-MIDAS"
    else:
        sec = "③ MIDAS 수익 예측"
    if sec.startswith("②"):
        c1_, c2_, c3_ = st.columns(3)
        gid = c1_.selectbox("품목", ids_c, ids_c.index("yf:CL=F") if "yf:CL=F" in ids_c else 0, format_func=lbl, key="md_gid")
        mac_opts = ["(자기) 월 실현분산"] + [i for i in ALL_IDS if GROUP.get(i) in ("Macro", "FX")]
        mac = c2_.selectbox("장기 변동성 요인 X", mac_opts, format_func=lambda i: i if i.startswith("(") else lbl(i), key="md_mac",
                            help="거시 변수는 월평균 수준으로 사용 (예: VIX 높으면 장기 변동성↑ → θ>0).")
        Kg = c3_.number_input("장기 성분 시차 K (개월)", 6, 60, 24, key="md_Kg")
        if st.button("▶ GARCH-MIDAS 추정", type="primary", key="md_grun"):
            with st.spinner("최우추정 중 (수십 초)..."):
                rr = _panel((gid,), "B", "logret", start, end, extra_bytes, ROLL)[gid]
                Xs = None if mac.startswith("(") else _levels(extra_bytes, ROLL)[mac].dropna()
                st.session_state["md_gm"] = (gid, mac, M.garch_midas(rr, Xs, int(Kg)))
        if "md_gm" in st.session_state:
            gid_, mac_, gm = st.session_state["md_gm"]
            e1_, e2_ = st.columns([1, 2])
            e1_.dataframe(gm["params"].style.format("{:.4f}"))
            e1_.metric("장기 성분 분산비 VR", f"{gm['VR']:.0%}", help="일간 조건부 분산 변동 중 장기(월간 요인) 성분이 설명하는 비율")
            th_ = gm["params"].loc["θ (장기 민감도)"]
            e1_.caption(f"θ = {th_['계수']:.3f} (t={th_['t값']:.1f}) — X가 1σ 높으면 장기 분산이 약 {np.exp(th_['계수']) - 1:+.0%} 변함")
            f = go.Figure([go.Scatter(x=gm["total_vol"].index, y=gm["total_vol"], name="총 조건부 변동성 √(τg)", line=dict(color="#c3c2b7", width=1)),
                           go.Scatter(x=gm["tau_vol"].index, y=gm["tau_vol"], name="장기 성분 √τ", line=dict(color=PAL[1], width=2.2))])
            e2_.plotly_chart(shade(fig_layout(f, 380, title=f"{short(gid_)} GARCH-MIDAS: 장기 vs 총 변동성 (연율 %) — 요인: {gm['source'] if mac_.startswith('(') else short(mac_)}"),
                                   gm["tau_vol"].index), key="md_gmchart")
    else:
        c1_, c2_, c3_ = st.columns([1, 2, 1])
        tid = c1_.selectbox("목표 품목 (다음 달 수익)", ids_c, ids_c.index("yf:ZC=F") if "yf:ZC=F" in ids_c else 0, format_func=lbl, key="md_tid")
        rids = c2_.multiselect("일간 설명변수 (수익률)", ALL_IDS, default=[i for i in [tid, "yf:DX-Y.NYB", "fred:VIXCLS"] if i in NAME][:3],
                               format_func=lbl, key="md_rids", max_selections=3)
        Kr = c3_.number_input("일간 시차 K", 5, 252, 63, key="md_Kr")
        if st.button("▶ MIDAS 수익 예측 (walk-forward)", type="primary", key="md_rrun") and rids:
            with st.spinner("월마다 재추정 중..."):
                R1 = _panel(tuple(dict.fromkeys([tid] + rids)), "B", "logret", start, end, extra_bytes, ROLL)
                st.session_state["md_r"] = (tid, M.midas_return_wf(R1[tid], {short(i): R1[i] for i in rids}, int(Kr)))
        if "md_r" in st.session_state:
            tid_, o = st.session_state["md_r"]
            st.markdown(f"##### {short(tid_)} 다음 달 수익 — 외표본 R² (기준: 과거 평균)")
            m1_, m2_ = st.columns(2)
            m1_.metric("MIDAS (Beta 가중)", f"{o['oos']['MIDAS']:+.2%}")
            m2_.metric("동일가중 (월 평균 집계)", f"{o['oos']['동일가중']:+.2%}")
            st.caption("두 값의 차이가 'MIDAS 가중'이 주는 가치. 월간 수익 예측의 외표본 R²는 대개 0 근처 — ④ 탭 결과와 같은 맥락.")
            f = go.Figure([go.Bar(x=np.arange(1, len(w) + 1), y=w, name=k, marker_color=PAL[j % 8], opacity=0.75)
                           for j, (k, w) in enumerate(o["weights"].items())])
            st.plotly_chart(fig_layout(f, 260, barmode="group", title="추정된 MIDAS 가중치 (전체 표본)", xaxis_title="시차 (영업일 전)", hovermode="x"),
                            key="md_rw")


# ============================================================================
# ④ FAVAR
# ============================================================================
def tab_favar():
    st.subheader("FAVAR — 요인 보강 VAR (Bernanke, Boivin & Eliasz 2005)")
    st.caption("① 정보집합(다수 원자재·거시변수)에서 PCA로 공통요인 F 추출 → ② VAR([F, Y]) 추정 → "
               "③ 요인 충격에 대한 Y의 충격반응(IRF)·분산분해(FEVD), 그리고 적재값을 통해 패널 전체 변수의 반응 복원. "
               "Cholesky 순서: 요인 → Y.")
    fav_def = [i for i in COMMOD_IDS if i.startswith("yf:") and pd.Timestamp(cat.set_index("id").loc[i, "start"]) < pd.Timestamp("2002-01-01")]
    fav_def += [i for i in ["fred:VIXCLS", "fred:DGS10", "yf:DX-Y.NYB", "yf:^GSPC", "fred:BAA10Y", "ak:macro_shipping_bdi"] if i in NAME]
    pids = st.multiselect("정보집합 (요인 추출 패널)", ALL_IDS, default=fav_def, format_func=lbl, key="fav_ids")
    f1, f2, f3, f4 = st.columns(4)
    yid = f1.selectbox("관측 변수 Y", ALL_IDS, ALL_IDS.index("yf:ZC=F") if "yf:ZC=F" in ALL_IDS else 0, format_func=lbl, key="fav_y")
    kf = f2.slider("요인 수 K", 1, 8, 3, key="fav_k")
    pv = f3.number_input("VAR 시차 (0=AIC 자동)", 0, 12, 0, key="fav_p")
    Hf = f4.slider("IRF 기간 H", 4, 60, {"B": 20, "W-FRI": 12, "ME": 12}[FQ], key="fav_h")
    excl = st.checkbox("Y를 정보집합에서 제외 (권장: 요인이 Y 자신을 담지 않도록)", True, key="fav_ex")
    if st.button("▶ FAVAR 추정", type="primary", key="favbtn"):
        pan_ids = [i for i in pids if not (excl and i == yid)]
        Rf = _panel(tuple(dict.fromkeys(pan_ids + [yid])), FQ, "logret", start, end, extra_bytes, ROLL)
        if len(Rf) < 100 or len(pan_ids) < kf:
            st.warning("표본 또는 변수 수가 부족합니다.")
        else:
            with st.spinner("추정 중..."):
                st.session_state["fav"] = (A.favar(Rf[pan_ids], Rf[[yid]], kf, None if pv == 0 else int(pv), int(Hf)), yid, Rf.index)
    if "fav" in st.session_state:
        fv, yid_, fidx = st.session_state["fav"]
        st.caption(f"표본 {fidx.min():%Y-%m-%d}~{fidx.max():%Y-%m-%d} · 관측 {fv['nobs']:,} · VAR 시차 p={fv['p']} · "
                   f"K개 요인 누적 설명력 {fv['ev'][:kf].sum():.1%}")
        e1, e2 = st.columns([1, 2])
        f = go.Figure(go.Bar(x=[f"PC{i+1}" for i in range(min(10, len(fv['ev'])))], y=fv["ev"][:10] * 100, marker_color=PAL[0]))
        e1.plotly_chart(fig_layout(f, 360, title="요인 설명력 (%)", hovermode="x"), key="fscree")
        e2.plotly_chart(heat(fv["load"].T, "요인 적재값 — 요인 해석용", h=360), key="fload")

        n = len(fv["names"]); ylab = short(yid_)
        st.markdown(f"##### 요인 충격 → {ylab} 충격반응 (직교화, 1σ 충격, ±1.96 s.e.)")
        f = make_subplots(rows=1, cols=n, subplot_titles=[f"{c} 충격" if c != yid_ else f"{ylab} 자체 충격" for c in fv["names"]])
        hgrid = np.arange(fv["oirf"].shape[0])
        for j in range(n):
            resp = fv["oirf"][:, -1, j]; se = fv["se"][:, -1, j]
            f.add_trace(go.Scatter(x=np.r_[hgrid, hgrid[::-1]], y=np.r_[resp + 1.96 * se, (resp - 1.96 * se)[::-1]],
                                   fill="toself", fillcolor="rgba(42,120,214,0.15)", line=dict(width=0), showlegend=False), 1, j + 1)
            f.add_trace(go.Scatter(x=hgrid, y=resp, line=dict(color=PAL[0], width=2), showlegend=False), 1, j + 1)
            f.add_hline(y=0, line_color="#898781", line_width=1, row=1, col=j + 1)
        st.plotly_chart(fig_layout(f, 300, hovermode="x"), key="firf")
        st.caption(f"Y는 표준화되어 있어 반응 단위 = {ylab} 수익률의 표준편차.")

        g1_, g2_ = st.columns(2)
        fe = fv["fevd"][-1]  # Y의 FEVD (H, n)
        f = go.Figure()
        for j in range(n):
            f.add_trace(go.Scatter(x=np.arange(1, fe.shape[0] + 1), y=fe[:, j] * 100, stackgroup="fe", name=fv["names"][j] if fv["names"][j] != yid_ else ylab,
                                   line=dict(width=0.5, color=PAL[j % 8])))
        g1_.plotly_chart(fig_layout(f, 360, title=f"{ylab} 예측오차 분산분해 (%)", xaxis_title="기간", hovermode="x"), key="ffevd")
        PR = fv["panel_resp"].iloc[:, :kf]
        g2_.plotly_chart(heat(PR, f"요인 충격 → 패널 변수 누적반응 (H={fv['oirf'].shape[0]-1}, 표준화 단위)",
                              -float(np.abs(PR.values).max()), float(np.abs(PR.values).max()), h=max(360, 18 * len(PR) + 80)), key="fpan")

# ============================================================================
# ⑤ 데이터 · 방법론
# ============================================================================
def tab_data():
    st.subheader("데이터 카탈로그")
    show = cat.copy(); show["start"] = pd.to_datetime(show["start"]).dt.date; show["end"] = pd.to_datetime(show["end"]).dt.date
    st.dataframe(show.sort_values(["group", "start"]), hide_index=True)
    st.subheader("방법론 가이드 · 추천")
    st.markdown("""
| 목적 | 도구 | 무엇을 보나 | 비고 |
|---|---|---|---|
| **동조성** | 상관행렬(평시/위기) | 위기 시 상관 급등 여부 | 변동성↑ 시 상관 상향편향(Forbes-Rigobon) 주의 |
| | 롤링 평균상관 · **흡수비율(AR)** | 시장이 단일요인에 지배되는 정도 | AR 급등은 시스템 리스크 선행지표로 쓰임 |
| | **PCA** | 공통 원자재 요인, 섹터 요인 | PC1≈'원자재 베타' |
| | **Diebold-Yilmaz 스필오버** ⭐추천 | 누가 충격을 전달/수신하는지, 시기별 연결성 | 위기 시 총 스필오버 급등 |
| **변동성/레버리지** | **GJR-GARCH / EGARCH** ⭐추천 | 비대칭 변동성(γ) | 원자재는 역레버리지 빈번 |
| | 레버리지 상관 corr(rₜ, 미래 RV) | 모형 없는 비대칭 측정 | 국면별(위기/평시) 비교 |
| | **Markov-switching** ⭐추천 | 고/저변동성 레짐 확률, 지속기간 | 칼만필터 레짐 분석과 상호 검증 |
| **회귀/선택** | OLS(HAC) | 기준 모형 | 다중공선성 시 계수 불안정 |
| | Ridge / **Lasso** / **Elastic Net** | 축소·변수선택 | EN은 상관 높은 원자재 묶음에 유리 |
| | **Adaptive Lasso** | 오라클 성질(일관된 변수선택) | 1단계 Ridge 가중치 |
| | **SCAD / MCP** | 비볼록 벌점 → 큰 계수 편향 감소 | 좌표하강 직접 구현, λ는 TS-CV |
| | **분위수 회귀** ⭐추천 | 꼬리(τ=5%) 의존성 — 위기 시 동조성 | 평균회귀가 놓치는 하방 동조 |
| | **Kalman TVP 회귀** | 시변 β(t) — 위기 시 민감도 변화 | q가 클수록 β가 빠르게 변함 |
| **비선형 ML** | Random Forest / Gradient Boosting ⭐추천 | 비선형·상호작용, 순열 중요도 | 과적합 주의, 테스트 지표로 판단 |
| | **LSTM** | 시퀀스 패턴 | 일간 수익률에서는 대개 선형 대비 우위 작음 |
| **요인/구조** | **FAVAR** | 다수 변수 공통요인 충격의 파급 | 요인 회전·식별은 단순화(2단계) |
| **다변량 변동성** | **DCC / ADCC** | 시변 조건부 상관, 동반하락 시 상관 비대칭(ADCC g) | 2단계 추정, SE는 1단계 불확실성 미반영 |
| | **BEKK(1,1)** Full/Diagonal/Scalar | 시장 간 충격(A)·변동성(B) 전이 경로와 유의성 | Full은 품목 2~4개 권장 |
| **레짐 예측** | **Hidden Markov (HMM)** | 레짐 확률, 레짐별 평균·변동성·베타 | 테스트 구간은 전방필터만 사용(누수 없음) |
| **결합** | 앙상블(단순평균) | 모델 불확실성 분산 | 대개 개별 최고 모델과 비슷하거나 더 안정적 |
| **요인 압축** (변수를 버리지 않음) | **PCR** · **PLS** · **3PRF** | 모든 변수를 소수 요인으로 요약해 예측 (Stock-Watson 2002, Kelly-Pruitt 2015) | k·L은 시계열 CV. PLS·3PRF는 y와 관련된 방향만 추출 |
| **모형 평균** | **Complete Subset** · **BMA** · **Stacking** | 작은 모형 수천 개 평균 / 사후확률 가중 + 변수별 포함확률 PIP / OOF 기반 최적 가중 | 변수 선택 불확실성을 평균으로 흡수 |
| **베이지안 연속 축소** | **Horseshoe** · **Minnesota BVAR** | 작은 계수만 강하게 축소 + 신용구간 / 먼 시차·타 변수일수록 강하게 축소 | 계수를 0으로 잘라내지 않음 |
| **패널 (④ 탭 · 여러 품목)** | Pooled OLS·Ridge·Lasso·EN·PCR·GBM·RF·MLP | 모든 품목을 함께 학습, 특성(모멘텀·변동성·왜도·실현롤수익·COT·계절성)의 시점별 순위 → 품목 간 상대 수익 | 순위 IC·상하위 롱숏으로 평가 (Gu-Kelly-Xiu 2020) |
| **혼합 주기 (③ 변동성 · ④ 수익 예측)** | MIDAS 회귀 · MIDAS-RV · GARCH-MIDAS | 일간 시차를 Beta 가중으로 요약 / 변동성 예측 경주는 ◆ 리스크 탭 / 장기·단기 변동성 분해 | 변동성 예측은 QLIKE·DM 검정 |
| **레짐·비선형 (② 레짐 · ④ 탭 모델)** | Markov-switching 회귀 · Threshold(TAR) · STAR | 숨은 레짐 / 관측 변수 임계값 / 매끄러운 전환에 따라 기울기가 달라짐 | 선형성 검정: sup-F 부트스트랩(Hansen), LM(Luukkonen) |
| **부스팅·커널 (④ 탭 모델)** | LightGBM(+TreeSHAP) · Gaussian Process · Kernel Ridge | 비선형·상호작용 자동 포착, SHAP으로 시점별 변수 기여 분해, GP는 예측 불확실성 동시 산출 | 조기종료·CV는 학습구간 내부 |
| **예측구간 (④ 탭)** | 분할 conformal · 적응형 ACI (Gibbs-Candès 2021) | 분포 가정 없이 목표 커버리지 보장, 국면 변화 시 폭 자동 조정 | 실현된 오차만 사용 |
| **TVP-VAR 연결성 (① 탭)** | 망각계수 칼만필터 VAR (Koop-Korobilis 2014) | 롤링창 없는 시변 스필오버 (ACG 2020) | 위기 시 총 연결성 급등 확인 |

**설명·예측에 추가로 고려할 만한 기법 (우선순위순)**

*A. 데이터·평가 측면 — 모델보다 효과가 큰 경우가 많음*
1. **특성(feature) 확장**: 모멘텀(4·12·52주), 실현변동성, COT 순포지션 z-score, 계절성(월 더미), 기간구조(캐리/롤수익), 달러·금리 변화 — 현재는 '타 시장 수익률'뿐
2. **Walk-forward(확장창 재추정) 평가**: 단일 학습/테스트 분할 대신 매 기간 재학습 → 구조변화에 강건, 실전과 동일한 성과 측정
3. **방향 분류 모형**(로짓·RF 분류기·HMM 확률 임계값): 트레이딩은 크기보다 부호가 중요 → 정확도·AUC로 평가

*B. 계량경제 (설명력·구조 해석)*
4. **Johansen 공적분 / VECM**: 대두-대두박-대두유(crush), 옥수수-밀, WTI-Brent 등 장기균형과 오차수정 속도 — 스프레드 트레이딩에 직결
5. **Markov-switching 회귀 (MS-AR, 레짐별 베타)**: HMM과 같은 사상을 회귀식으로 — 칼만필터 레짐과 직접 비교 가능
6. **TVP-VAR 연결성 (Antonakakis 2020)**: 롤링창 없는 시변 스필오버 — DY 롤링의 창 선택 민감도 해소
7. **Local Projection (Jordà 2005)**: VAR 제약 없이 충격반응 추정 (예: 달러·VIX 충격 → 곡물)
8. **Threshold / Smooth Transition (TAR·STAR)**: VIX·재고 수준 등 관측 가능한 변수로 레짐 전환
9. **HAR-RV**: 실현변동성 예측 표준 벤치마크 (GARCH와 비교) / **Copula·CoVaR**: 하방 꼬리 동조·전염

*C. 머신러닝 (예측력)*
10. **XGBoost/LightGBM + SHAP**: GBM 강화판, 비선형 효과 해석
11. **Temporal Fusion Transformer / N-BEATS**: 다중 시계열·다중 기간 예측 (데이터 많을 때)
12. **Conformal prediction**: 모델 무관 예측구간 — 포지션 사이징에 활용
13. **Stacking**: 앙상블 가중치를 검증구간에서 학습 (단순평균 대비)

**포트폴리오 백테스트 (⑤ 탭) 계산 기준**
- 신호·공분산은 데이터 주기(일·주·월)로 계산, 리밸런싱은 별도 주기(매일·매주·매월·분기, 데이터 주기 이상). 결정일 종가까지의 정보만 사용 → **다음 거래일 종가 실행** → 다음 실행일까지 보유하며 손익은 매일 추적(계약 수 고정, 비중 표류). 통계는 모두 일간 수익률 기준.
- 비중: max μ'w − (λ/2)w'Σw − Σcᵢ|wᵢ−wᵢ⁰| 를 **Newton-Raphson**(로그배리어 내부점법, 등식제약은 KKT Newton)으로 풂. wᵢ⁰ = 직전 비중의 가격 표류 후 값 → 비용을 감안해 필요한 만큼만 거래.
  Σ = 결정일까지 과거 창의 Ledoit-Wolf 수축 공분산. 롱온리 Σw=1·w≥0, 롱숏 Σ|w|≤L(정해진 투자금 대비 명목 한도).
- 벤치마크: 동일가중 1/N, 리스크 패리티(Spinu 2013 — 역시 Newton). 같은 리밸런싱·같은 비용 적용.
- 모멘텀 기간: 데이터 주기와 무관한 달력 기간 (12-1개월 = 일간 252일 = 주간 52주). 1주·3주·1개월은 원자재에서 단기 반전이 섞이는 구간.
  CTA 혼합 = 로그가격 EWMA 교차 3쌍(8/24·16/48·32/96일)을 변동성으로 정규화하고 반응함수 φ(y)=y·e^(−y²/4)로 과열 추세 축소 (Baz et al. 2015).
- 거래 규칙: 고정 주기 / **무거래 구간**(Σ|목표−현재| > 폭이거나 최대 보유기간 경과 시에만 거래, Leland 2000) / **부분 조정**(차이의 κ만큼, Gârleanu-Pedersen 2013).
- 변동성 사이징: 상관은 Ledoit-Wolf, 품목 변동성은 EWMA·HAR·MIDAS 일간 walk-forward 예측(월말 재추정, 그 시점까지 실현된 목표만)으로 교체.
  변동성 타기팅 배율 = min(최대 배율, 목표 / 예측 포트폴리오 변동성) — 위기 시 자동 축소.
- 리스크 버짓(신호 기울기): 위험 기여 비율 = 예산 bᵢ = (1−τ)/N + τ·(신호 강도 |μᵢ/σᵢ| 비례), Newton(Spinu)으로 풂.
  롱숏은 부호 반영 공분산 D·Σ·D(D = diag(sign μ))로 위험 기여 계산 → 롱·숏 간 헤지 효과 반영. τ=0 = (부호 반영) 리스크 패리티.
- 비용: 편도 (유동성 틱×1틱 가치 + 수수료) / 계약 명목금액 — 매 시점 실제 가격으로 재계산. + 보유 포지션 롤비용(연간 롤 횟수 × (1틱 + 수수료×2)).
- Sharpe = 비용 후 기간수익률 평균/표준편차 × √(연간 리밸런싱 횟수), 선물 초과수익 기준(담보 이자 미포함).
- 한계: 분수 계약(반올림 없음), 증거금 제약 미반영, 시장충격 비용은 선형 근사.

**정상성과 예측력 (Q&A)**
- *추세·계절성을 제거해 ε을 정상으로 만들어야 하나?* → **추세는 이미 제거됨**: 이 앱은 로그가격이 아닌 수익률(로그차분)을 쓰고, 수익률은 ADF·KPSS 모두 정상으로 판정됨(정상성 진단 참고).
  레벨끼리 회귀하면 가성회귀(spurious regression)가 생기지만, 수익률 회귀에서는 그 문제가 없음.
- **계절성은 제거하지 말고 설명변수로 넣을 것**: 계절성이 유의하면 그 자체가 예측 정보. 제거 후 나머지를 예측하는 것과
  계절 항을 회귀에 포함하는 것은 수학적으로 동일(Frisch-Waugh-Lovell)하나, 제거하고 버리면 예측력을 잃음 → '추가 특성 → 계절성'.
- **⚠ 단, 계절성 분석 전에 연속월물 롤오버 갭을 먼저 보정할 것**: Yahoo '=F'는 만기 교체일에 두 계약의 가격차만큼 가짜 수익률이 생기고,
  곡물은 그 날짜가 매년 같아서 **가짜 계절성**이 됨 (예: 옥수수 2021-07-15 원자료 −19.1% = 7월 올드크롭→9월 뉴크롭 역전 갭).
  롤 보정 전후 계절성 F검정 p: 옥수수 0.008→0.119, 대두 0.003→0.245, 대두박 0.001→0.028. 롤 보정 후 '특성' 모델의 개선도 대부분 사라짐
  (옥수수 월간 OOS R² +2.6%→−0.2%) — 롤 갭을 예측한 것이라 실제 거래로 얻을 수 없는 수익이었음. 사이드바 '연속월물 롤오버 갭 보정'(기본 ON).
- **주말·월말 간격은 문제 아님**: 금요일 종가→월요일 종가 수익률은 주말 동안의 실제 정보(뉴스)를 반영한 진짜 가격 변화. 보간·평활하면 안 됨.
  (월요일 변동성이 약간 큼 — 옥수수 일간 표준편차 월 2.10% vs 화~금 1.6~1.75% — 일간 모델이면 요일 더미로 처리 가능)
- **남은 비정상성은 '분산'**: 모든 품목에서 ARCH 효과(변동성 군집) 유의. '목표 변동성 표준화'로 잔차 이분산은 사라지지만(ARCH p 0.15→0.69),
  OOS R²·Sharpe는 개선되지 않음(walk-forward 실험) → **ε의 정상성은 추정·검정의 타당성을 위한 조건이지, 예측력을 만들어주지는 않음.**
- **예측력의 원천은 정보**: R²를 올리는 것은 변환이 아니라 Y와 관련된 새로운 정보(모멘텀·COT·계절성·캐리). 레벨의 장기균형(공적분)도 정보 → VECM.

**주의사항**
1. Yahoo 연속월물(=F)은 롤오버 시 가격 갭 → ±15σ 클리핑으로 완화했지만, 정밀 분석은 back-adjusted 연속계열 사용 권장 (보유 30년 데이터 업로드).
2. '설명' 모드 R²는 동시점 상관이므로 예측력이 아님. 예측력은 '예측' 모드 OOS R²로만 판단.
3. 모든 하이퍼파라미터 선택·표준화는 학습구간 내부에서만 수행(미래정보 누수 방지).
4. 일간 데이터는 시장별 마감시각 차이(CBOT·LME·DCE·Baltic)로 동시점 상관이 과소추정됨 → 주간 빈도 권장.
""")


# ============================================================================
# ★ 구매 타이밍 (실물 구매 의사결정)
# ============================================================================
@st.cache_data(show_spinner=False)
def _proc_inputs(i: str, risk: str, relm: str, roll: bool, extra_bytes):
    import procurement as PR
    return PR.inputs(_levels(extra_bytes, True), _levels(extra_bytes, False), i, risk, relm)


@st.cache_data(show_spinner=False)
def _proc_bt(i: str, weeks: int, w_items: tuple, k: float, start: str, risk: str, lookback: int, season: str, carry: float,
             roll: bool, extra_bytes):
    import procurement as PR
    La_, Lr_ = _levels(extra_bytes, True), _levels(extra_bytes, False)
    return PR.backtest(Lr_[i], La_[i], weeks, 4, start=start, w=dict(w_items), k=k, lookback=lookback,
                       inp=_proc_inputs(i, risk, "price", roll, extra_bytes), season=season, carry_bp_week=carry)


def tab_procure():
    import procurement as PR
    st.subheader("구매 타이밍 — 필요한 물량을 언제, 얼마나 살까")
    st.caption("평가 기준 = 같은 기간에 매주 똑같이 나눠 사는 '균등 매수' 대비 평균 매입가 절감(bp). 매수 점수는 그 시점까지의 데이터로만 계산 "
               "(롤 보정 가격으로 만기 교체 갭 제거, 대체재·크러시 관계는 실제 근월물 시세), 매입가는 실제 근월물 시세, 체결은 결정일(금) 다음 거래일 종가.")
    ids_av = [i for i in PR.BUY_IDS if i in NAME]
    c1_, c2_, c3_ = st.columns([2, 1.6, 0.8])
    ids = c1_.multiselect(f"구매 품목 ({len(ids_av)}개 중 선택)", ids_av, default=[i for i in ["yf:ZC=F", "yf:ZW=F", "yf:ZS=F", "yf:ZM=F"] if i in ids_av],
                          format_func=lambda i: PR.BUY_IDS[i] + ("" if i in PR.REL_DEF else " · 상대가치 없음"), key="pc_ids",
                          help="곡물·철강 원료·에너지·귀금속·소프트·축산·DCE 전 품목. '상대가치 없음' = 검증을 통과한 대체재·원료 관계가 없어 "
                               "가격 위치·반등·계절성으로만 판단 (⑥ 실험 결과 '전 품목 구매 타이밍')")
    adv_p = c2_.checkbox("고급: 비교용 프리셋 (보수적·공격적) 표시", False, key="pc_adv",
                         help="보수적·공격적은 단계별 검증에서 '추천'보다 절감이 작았던 비교용 설정 (⑥ 실험 결과 → 구매 타이밍)")
    preset = c2_.selectbox("전략 프리셋", list(PR.PRESETS) if adv_p else PR.BASIC_PRESETS, 0, key="pc_preset",
                           help="추천: 단계별 검증에서 전·후반 모두 균등 매수보다 나았고 꼬리 위험도 가장 작은 조합. "
                                "균형: 싸면서 반등을 시작할 때 (상대가치 데이터 없이 가격만으로). "
                                "고급 — 보수적: 계절성 추가, 공격적: 가격 위치만 2배 (꼬리 위험 큼)")
    weeks = c3_.number_input("구매 기간 (주)", 2, 52, 13, key="pc_weeks", help="이 기간 안에 필요한 물량을 모두 산다는 가정")
    P0 = PR.PRESETS[preset]
    with st.expander("신호 · 위험 판단 · 기간 설정 (프리셋 값에서 조정)", expanded=False):
        e = st.columns(5)
        w = {c: e[j].slider(PR.COMP_LBL[c], 0.0, 1.0, float(P0["w"].get(c, 0.0)), 0.05, key=f"pc_w_{c}_{preset}") for j, c in enumerate(PR.COMP_LBL)}
        g = st.columns(5)
        k = g[0].slider("공격성 k", 0.0, 3.0, float(P0["k"]), 0.25, key=f"pc_k_{preset}", help="0 = 균등 매수, 클수록 점수에 따라 크게 당기거나 늦춤 (주당 최대 3배)")
        risk_l = g[1].selectbox("위험 판단 (공격성 자동 조절)", list(PR.RISK_MODES), list(PR.RISK_MODES.values()).index(P0["risk"]), key=f"pc_risk_{preset}",
                                help="위험이 높을수록 균등 매수에 가깝게. MIDAS = 일간 실현분산 시차로 예측한 변동성의 3년 분위, HMM = 고변동 레짐 필터 확률 (모두 walk-forward)")
        lb_l = g[2].selectbox("가격 위치 창", ["6개월", "1년", "2년"], {126: 0, 252: 1, 504: 2}[P0["lookback"]], key=f"pc_lb_{preset}")
        sea_l = g[3].selectbox("계절성 산정", list(PR.SEASON_MODES), list(PR.SEASON_MODES.values()).index(P0["season"]), key=f"pc_sea_{preset}",
                               help="남미 작황 확대 등으로 수확기 패턴이 이동 → 최근 가중이 전체 평균보다 소폭 우수")
        carry = g[4].number_input("보유비용 (bp/주)", 0.0, 20.0, 0.0, 0.5, key="pc_carry",
                                  help="재판매·인도 시점까지 보유하는 금융·보관 비용. 일찍 살수록 커짐 — 백테스트 평가에만 반영")
        st.caption("12개월 추세는 롱숏 전략에선 유효했지만 구매 시점 선택에서는 '오를 때 서둘러 사기'가 평균 −20~−50bp로 손해 → 기본 0. "
                   "상대가치는 경제적 관계(옥수수↔밀 가격비, 대두 크러시 마진, 대두박·대두유 가치/대두, 철광석 SGX↔DCE 수입 차익, "
                   "점결탄↔철근 고로 마진, 구리↔금)의 1년 분위 — 대두유는 바이오연료 정책으로 관계 자체가 이동해 효과가 약하고, "
                   "구리는 대체재(알루미늄) 관계가 최근 기각돼 구리/금만 참고 수준. 점결탄은 DCE 위안 가격을 그날 환율로 달러 환산(증치세 포함), "
                   "철강 원료 3품목은 데이터가 2010~2013년부터라 검증 기간이 짧음. "
                   "확장 품목(에너지·귀금속·소프트·축산·DCE 등)은 후보 관계 중 두 기간 개선 + 무관한 가격비(위약)·자기 가격 대조군을 모두 넘은 것만 채택"
                   "(가스/원유, 생우/송아지, 알루미늄/구리, 브렌트 경유 크랙 등 10품목), "
                   "통과 관계가 없는 품목은 상대가치 없이 가격 위치·반등·계절로만 판단 → ⑥ 실험 결과 '전 품목 구매 타이밍'.")
    risk = PR.RISK_MODES[risk_l]; lookback = {"6개월": 126, "1년": 252, "2년": 504}[lb_l]; season = PR.SEASON_MODES[sea_l]
    if not ids:
        st.info("품목을 선택하세요."); return
    La_, Lr_ = _levels(extra_bytes, True), _levels(extra_bytes, False)

    def sig_of(i):
        inp = _proc_inputs(i, risk, "price", ROLL, extra_bytes)
        return PR.signals(La_[i], lookback, inp["rel"], inp["rel_sign"], inp["vol_fc"], inp["crisis"], season), inp

    rows, sigs = [], {}
    for i in ids:
        sig, inp = sig_of(i); sigs[i] = sig
        sc = PR.score(sig, w)
        last = sig.dropna(subset=["value"]).index.max()
        pa = PR.usd_t(Lr_, i); px = float(pa.loc[:last].iloc[-1]); p1 = pa.loc[last - pd.DateOffset(years=1):last]
        rk = float(sig["risk"].loc[last]); b_lo, b_hi = PR.band_4w(La_[i], sig, last)
        rows.append({"품목": PR.BUY_IDS[i], "id": i, "기준일": last.date(), "단위": PR.unit_of(i), "가격": px, "1년 최저": p1.min(),
                     "1년 최고": p1.max(), "매수 점수": float(sc.loc[last]) * 100,
                     **{PR.COMP_LBL[c]: float(sig[c].loc[last]) * 100 for c in PR.COMP_LBL}, "위험 지표": rk * 100,
                     "4주 하단 90%": px * b_lo, "4주 상단 90%": px * b_hi,
                     "권장 속도 (균등 대비 배)": PR.pace(float(sc.loc[last]), rk, 1.0, int(weeks), k, 3.0) * int(weeks)})
    T = pd.DataFrame(rows)
    st.markdown(f"##### 이번 주 매수 계획 (기준일 {T['기준일'].max()} · {preset})")
    light = lambda s: "🟢 적극 매수" if s >= 65 else ("🟡 평소대로" if s >= 35 else "🔴 매수 늦춤")
    # ---- 이번 주 매수 계획: 남은 물량을 넣으면 이번 주 살 양을 바로 계산 ----
    wl = st.number_input("남은 기간 (주, 이번 주 포함)", 1, 52, int(weeks), key="pc_wl",
                         help="목표 기간 중 남은 주 수. 마지막 주에는 남은 물량을 전부 삼")
    ncol = min(len(T), 4)
    plan = []
    for j, r in T.iterrows():
        if j % ncol == 0:
            qc = st.columns(ncol)
        box = qc[j % ncol].container(border=True)
        u = "온스" if r["단위"] == "$/oz" else "톤"
        rem = box.number_input(f"{r['품목']} — 남은 필요 물량 ({u})", 0.0, 1e8, 30000.0, 1000.0, key=f"pc_q_{r['id']}")
        q = PR.pace(r["매수 점수"] / 100, r["위험 지표"] / 100, rem, int(wl), k, 3.0)
        eq = rem / int(wl)
        box.metric(f"이번 주 매수량 · {light(r['매수 점수'])}", f"{q:,.0f} {u}", f"균등 {eq:,.0f} {u} 대비 {q - eq:+,.0f}" if rem else None, delta_color="off")
        srow = sigs[r["id"]].loc[sigs[r["id"]].dropna(subset=["value"]).index.max()]
        ew = PR.eff_weights(srow, w)
        box.caption(f"점수 **{r['매수 점수']:.0f}** — {PR.reason(srow, w)}  \n"
                    f"현재가 \\${r['가격']:,.0f}{r['단위'][1:]} · 4주 90% 범위 \\${r['4주 하단 90%']:,.0f}~{r['4주 상단 90%']:,.0f}"
                    + ("" if "relval" in ew or w.get("relval", 0) == 0 else
                       "  \n상대가치 관계 없음 → 가중치 재배분: " + " · ".join(f"{PR.COMP_LBL[c].split(' (')[0]} {v:.0%}" for c, v in ew.items())))
        plan.append({"품목": r["품목"], "신호": light(r["매수 점수"]), "매수 점수": r["매수 점수"], f"남은 물량": rem, "이번 주 매수량": q,
                     "균등 매수량": eq, "매수 후 남은 물량": rem - q, "단위": u, "판단 근거": PR.reason(srow, w)})
    st.dataframe(pd.DataFrame(plan).set_index("품목").style.format({"매수 점수": "{:.0f}", "남은 물량": "{:,.0f}", "이번 주 매수량": "{:,.0f}",
                                                                   "균등 매수량": "{:,.0f}", "매수 후 남은 물량": "{:,.0f}"}))
    st.caption(f"이번 주 매수량 = 남은 물량 ÷ 남은 주 × 배율, 배율 = 1 + k·확신·(2·점수 − 1) (0~3배), 확신 = 1 − 0.5·위험 지표({risk_l}) — "
               "점수 50이면 균등, 높을수록 더 사고 낮을수록 덜 삼. 마지막 주는 전량. 판단 근거 = 점수에 가장 크게 작용한 구성요소 (65 이상 유리 · 35 이하 불리).")
    with st.expander("신호 구성 상세표 (구성요소 점수 · 위험 · 4주 범위)"):
        show = T.drop(columns=["id", "권장 속도 (균등 대비 배)"]).set_index("품목")
        st.dataframe(show.style.format({c: "{:,.1f}" for c in show.columns if show[c].dtype.kind == "f"}))
        st.caption(f"점수 = 구성 요소의 가중 평균 (0~100, 가중치 0인 요소는 참고 표시, 값이 없는 요소는 빼고 재배분). 위험 지표 = {risk_l}. "
                   "'4주 90% 범위' = 현재 변동성 × 과거 '표준화 4주 수익률'의 경험적 5%·95% 분위 (정규 ±1.645σ는 꼬리가 두꺼워 실제 적중률 86~87%, "
                   "철광석 75%에 그쳐 보정) → 하단 근처 지정가 매수 주문 참고. 적중률 검증: ◆ 리스크 탭.")

    st.markdown("##### 펀더멘털 맥락 (참고 — 점수에는 미반영)")
    FD = PR.fundamentals(ids)
    st.dataframe(FD.style.format({c: "{:+.2f}" for c in ["전월 대비 (%p)", "USDA 가격 전망 수정 (%)", "펀드 포지션 z (3년)"]}
                                 | {"재고/사용비율 (%)": "{:.1f}", "같은 달 과거 분위": "{:.0%}", "펀드 순포지션 (% OI)": "{:+.1f}"}, na_rep="–"))
    st.caption("WASDE 재고/사용비율이 낮을수록(분위↓) 수급이 빡빡 → 가격 상방 위험. 펀드 포지션 z가 크면 투기적 매수가 몰린 상태 → 되돌림 위험. "
               "검증 결과 이 지표들의 '발표 이후' 가격 예측력은 과거 20년에서 재현되지 않아 점수에 넣지 않고, 위험 맥락으로만 표시 (대두 계열은 미국 수급표 기준).")

    st.markdown("##### 신호 상세")
    di = st.selectbox("품목 선택", list(T["id"]), format_func=lambda i: PR.BUY_IDS[i], key="pc_detail")
    sig, inp = sig_of(di); sc = PR.score(sig, w) * 100
    pa = PR.usd_t(Lr_, di)
    lo = pa.index.max() - pd.DateOffset(years=3)
    pa3, sc3 = pa[pa.index >= lo], sc[sc.index >= lo]
    f = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.45, 0.3, 0.25], vertical_spacing=0.05,
                      subplot_titles=(f"{PR.BUY_IDS[di]} 가격 ($/t, 근월물) — 초록 = 점수 ≥ 65 (적극 매수), 빨강 = 점수 < 35",
                                      "매수 점수와 구성 요소", f"상대가치 관계: {PR.REL_DEF.get(di, ('–',))[0]}"))
    f.add_trace(go.Scatter(x=pa3.index, y=pa3.values, line=dict(color=PAL[0], width=1.6), name="가격"), 1, 1)
    hi_, lo_ = sc3[sc3 >= 65].index.intersection(pa3.index), sc3[sc3 < 35].index.intersection(pa3.index)
    f.add_trace(go.Scatter(x=hi_, y=pa3.reindex(hi_), mode="markers", marker=dict(color=PAL[2], size=5), name="적극 매수 구간"), 1, 1)
    f.add_trace(go.Scatter(x=lo_, y=pa3.reindex(lo_), mode="markers", marker=dict(color=PAL[1], size=4, opacity=0.6), name="매수 늦춤 구간"), 1, 1)
    f.add_trace(go.Scatter(x=sc3.index, y=sc3.values, line=dict(color="#222", width=2), name="매수 점수"), 2, 1)
    for j, c in enumerate([c for c in PR.COMP_LBL if w.get(c, 0) > 0]):
        s_ = sig[c][sig.index >= lo] * 100
        f.add_trace(go.Scatter(x=s_.index, y=s_.values, line=dict(color=PAL[(j + 3) % 8], width=1, dash="dot"), name=PR.COMP_LBL[c]), 2, 1)
    f.add_hline(y=65, line_dash="dash", line_color=PAL[2], row=2, col=1); f.add_hline(y=35, line_dash="dash", line_color=PAL[1], row=2, col=1)
    rel = inp.get("rel")
    if rel is not None:
        r3 = rel[rel.index >= lo]
        f.add_trace(go.Scatter(x=r3.index, y=r3.values, line=dict(color=PAL[6], width=1.4), name="상대가치 관계"), 3, 1)
        q_ = rel.rolling(252, min_periods=126)
        for qq, nm_ in [(0.2, "1년 하위 20%"), (0.8, "1년 상위 20%")]:
            b_ = q_.quantile(qq)[rel.index >= lo]
            f.add_trace(go.Scatter(x=b_.index, y=b_.values, line=dict(color="#898781", width=1, dash="dot"), name=nm_, showlegend=False), 3, 1)
    st.plotly_chart(fig_layout(f, 680), key="pc_detail_fig")
    if risk != "ewma":
        rs_ = sig["risk"][sig.index >= lo] * 100
        f = go.Figure(go.Scatter(x=rs_.index, y=rs_.values, fill="tozeroy", line=dict(color=PAL[1], width=1)))
        st.plotly_chart(fig_layout(f, 200, title=f"위험 지표 ({risk_l}, %) — 높을수록 균등 매수에 가깝게", yaxis_range=[0, 100]), key="pc_risk_fig")
    m = np.log(La_[di].dropna()).resample("ME").mean()
    rel_m = (m - m.groupby(m.index.year).transform("mean")) * 100
    yy = rel_m.index.year.max()
    f = go.Figure()
    for j, (nm_, sub) in enumerate([("전체 연도", rel_m), ("최근 10년", rel_m[rel_m.index.year >= yy - 10])]):
        prof = sub.groupby(sub.index.month).mean()
        f.add_trace(go.Bar(x=[f"{mm}월" for mm in prof.index], y=prof.values, name=nm_, marker_color=[PAL[0], PAL[2]][j]))
    st.plotly_chart(fig_layout(f, 260, barmode="group", title="계절 패턴: 월평균 가격의 연평균 대비 차이 (%, 롤 보정) — 음수 = 평균적으로 싼 달", hovermode="x"),
                    key="pc_season")

    st.markdown("##### 과거 성과 검증 — 이 설정으로 샀다면?")
    v1_, v2_ = st.columns([1, 3])
    y0 = v1_.selectbox("검증 시작 연도", [2008, 2012, 2016, 2020], 0, key="pc_y0", help="긴 표본일수록 통계적으로 신뢰. 최근 효과는 아래 3년 이동 차트·최근 가중 평균으로 확인")
    v2_.caption(f"{int(weeks)}주 구매 기간을 {y0}년부터 4주 간격으로 반복 시작해 균등 매수와 비교 (+ = 균등보다 싸게 삼, 보유비용 {carry:g}bp/주 반영). "
                "'사후 최저' = 그 기간 최저 주에 전량 구매 (불가능한 이론 상한).")
    rows, per = [], {}
    for i in ids:
        B = _proc_bt(i, int(weeks), tuple(sorted(w.items())), float(k), f"{y0}-01-01", risk, lookback, season, float(carry), ROLL, extra_bytes)
        S_ = PR.summarize(B, int(weeks), 4, halflife_years=5)
        hv = B[B["시작위험"] >= 0.7]["절감_규칙"]; lv = B[B["시작위험"] < 0.7]["절감_규칙"]
        rows.append({"품목": PR.BUY_IDS[i], "평균 절감 (bp)": S_["규칙"]["평균 절감 (bp)"], "최근 가중 평균 (bp, 반감기 5년)": S_["규칙"]["최근 가중 평균 (bp)"],
                     "균등보다 싸게 산 비율": S_["규칙"]["균등보다 싸게 산 비율"], "최악 5% (bp)": S_["규칙"]["하위 5% (bp)"],
                     "t값 (비중첩)": S_["규칙"]["t값 (비중첩)"], "고변동 국면 시작 (bp)": hv.mean(), "저변동 국면 시작 (bp)": lv.mean(),
                     "사후 최저 (이론 상한, bp)": S_["사후최저"]["평균 절감 (bp)"], "기간 수": S_["규칙"]["기간 수"]})
        per[PR.BUY_IDS[i]] = B["절감_규칙"]
    R_ = pd.DataFrame(rows).set_index("품목")
    st.dataframe(R_.style.format({"균등보다 싸게 산 비율": "{:.0%}", "기간 수": "{:.0f}"} | {c: "{:+.1f}" for c in R_.columns if c not in ("균등보다 싸게 산 비율", "기간 수")}))
    P_ = pd.DataFrame(per)
    roll3 = P_.mean(axis=1).rolling(max(3, int(156 / 4)), min_periods=10).mean()
    f = go.Figure([go.Scatter(x=roll3.index, y=roll3.values, line=dict(color=PAL[0], width=2.5), name="선택 품목 평균, 3년 이동평균")])
    f.add_hline(y=0, line_color="#222")
    st.plotly_chart(shade(fig_layout(f, 280, title="절감의 시간 안정성: 3년 이동평균 (bp) — 0 아래 구간 = 그 3년은 균등 매수가 나았음"), roll3.index), key="pc_roll")
    f = go.Figure([go.Histogram(x=v.values, name=k_, opacity=0.6, nbinsx=40) for k_, v in per.items()])
    f.add_vline(x=0, line_color="#222")
    st.plotly_chart(fig_layout(f, 280, barmode="overlay", title="구매 기간별 절감 분포 (bp, 균등 매수 대비)", xaxis_title="절감 (bp)", hovermode="closest"), key="pc_hist")
    avg = R_["평균 절감 (bp)"].mean()
    st.info(f"해석: 평균 {avg:+.0f}bp — 예) 연 매입액 1억 달러면 약 ${avg * 1e4:,.0f}. 사후 최저와의 격차(수백 bp)가 보여주듯 최저점 맞히기는 불가능에 가깝고, "
            "규칙의 가치는 '비쌀 때 덜, 쌀 때 더 + 대체재 대비 싼 쪽을 더' 사는 규율과 위험 국면 대응. 단계별 검증·신호 비교는 ⑥ 실험 결과 탭.")


# ============================================================================
# ◆ 리스크 (시변 감쇠 변동성 · 모형 검증 · VaR/ES · 꼬리 위험 · 물량 위험)
# ============================================================================
@st.cache_data(show_spinner=False)
def _risk_vm(i: str, models: tuple | None, roll: bool, extra_bytes):
    import risk as RK
    return RK.vol_models(RK.returns(_levels(extra_bytes, roll)[i]), list(models) if models else None)


def load_exp(name):
    import experiments as EX
    df, _ = EX.load(name)
    return df


RISK_DEF = [i for i in ["yf:ZC=F", "yf:ZS=F", "yf:ZM=F", "yf:ZW=F", "yf:TIO=F", "yf:HG=F", "yf:CL=F"] if i in NAME]


def tab_risk():
    import risk as RK
    st.subheader("리스크 — VaR/ES · 꼬리 위험 · 구매·보유 물량 위험")
    st.caption("모든 예측은 walk-forward (매년 1월 그 이전 데이터로만 재추정, 시점 t의 예측은 t−1 종가까지의 정보). 수익률 = 롤 보정 가격의 일간 로그수익률(%). "
               "구매자에게 위험은 '가격 상승'(매입 예정 물량 = 숏 노출), 재고 보유자에게는 '가격 하락' — 두 방향을 모두 평가.")
    ids_r = [i for i in COMMOD_IDS if GROUP.get(i) not in ("Freight",)]
    c1_, c2_, c3_ = st.columns([2, 1, 1])
    rid = c1_.selectbox("품목", ids_r, ids_r.index("yf:ZC=F") if "yf:ZC=F" in ids_r else 0, format_func=lbl, key="rk_id")
    ev0 = c2_.selectbox("평가 시작", [2005, 2010, 2015, 2020], 1, key="rk_ev0", help="이 시점 이후의 예측으로 손실·검정을 계산 (추정은 그 이전부터 walk-forward)")
    if c3_.button("▶ 모형 추정 (품목당 약 10초)", type="primary", key="rk_run"):
        with st.spinner("변동성 모형 walk-forward 추정 중 (HMM·GARCH 매년 재적합)…"):
            st.session_state["rk_o"] = (rid, _risk_vm(rid, None, ROLL, extra_bytes))
    if "rk_o" not in st.session_state:
        st.info("품목을 고르고 '모형 추정'을 누르세요. σ 모형 비교(QLIKE·DM·MCS)는 ③ 변동성 → '변동성 예측 검증', "
                "변동성 레짐·반감기 안정성은 ② 레짐, 동반 폭락·급등(꼬리 의존)은 ① 시장 구조. 여러 품목 요약: ⑥ 실험 결과 → '리스크 모형'.")
        return
    rid_, o = st.session_state["rk_o"]
    r, S = o["r"], o["S"]
    ev = S.index >= pd.Timestamp(f"{ev0}-01-01")
    c, d, e = st.tabs(["VaR · ES 백테스트", "꼬리 위험 (EVT · 스트레스)", "구매·보유 물량 위험"])
    with c:
        k1_, k2_, k3_, k4_ = st.columns(4)
        side_l = k1_.radio("포지션", ["매입 예정 물량 (가격 상승 위험)", "보유 재고 (가격 하락 위험)"], key="rk_side")
        side = "short" if side_l.startswith("매입") else "long"
        alpha = k2_.selectbox("신뢰수준", [0.95, 0.975, 0.99], 2, format_func=lambda x: f"{x:.1%}", key="rk_alpha")
        h = k3_.selectbox("기간 (영업일)", [1, 5, 10, 20], 0, key="rk_h")
        vm = k4_.selectbox("σ 모형", list(S.columns), list(S.columns).index("GARCH(1,1)") if "GARCH(1,1)" in S else 0, key="rk_vm",
                          help="③ '변동성 예측 검증'에서 12품목 모두 MCS에 든 GARCH(1,1)가 기본")
        V = RK.var_forecasts(r, S[vm], alpha, side, h, start=f"{ev0}-01-01")
        BT = RK.backtest_table(V)
        now = RK.var_now(V, r, RK.next_var(o, vm))
        notional = st.number_input("노출 금액 ($)", 0.0, 1e10, 10_000_000.0, 1_000_000.0, key="rk_notional")
        st.markdown(f"##### 오늘 기준 {h}일 {alpha:.1%} VaR · ES — {short(rid_)} ({side_l})")
        mc = st.columns(len(now))
        for j, (m, row) in enumerate(now.iterrows()):
            mc[j].metric(m, f"{row['VaR (%)']:.2f}%", f"ES {row['ES (%)']:.2f}% · ${notional * row['VaR (%)'] / 100:,.0f}", delta_color="off")
        st.caption("VaR = 이 확률로 넘지 않을 손실 한도, ES(기대 부족분) = VaR를 넘는 날들의 평균 손실 (꼬리의 깊이). "
                   "FHS·EVT는 σ로 표준화한 과거 손실의 실제 분포(두꺼운 꼬리)를 쓰고, 정규는 꼬리를 과소평가하기 쉬움.")
        st.markdown(f"##### 백테스트 ({V['table'].index.min():%Y-%m}~, {'비중첩 ' + str(h) + '일' if h > 1 else '매일'})")
        st.dataframe(BT.style.format({"기대 초과": "{:.1f}", "초과율": "{:.2%}", "Kupiec p (적중률)": "{:.3f}", "Christoffersen p (독립성)": "{:.3f}",
                                      "조건부 적중 p": "{:.3f}", "ES 검정 p (과소추정)": "{:.3f}", "평균 VaR (%)": "{:.2f}", "평균 ES (%)": "{:.2f}"}, na_rep="–"))
        st.caption("Kupiec = 초과 횟수가 1−α와 맞나, Christoffersen = 초과가 몰려 오지 않나(독립성), 조건부 적중 = 둘 다, "
                   "ES 검정(McNeil-Frey) = 초과일의 손실이 ES보다 큰가 — 모두 p ≥ 0.05면 통과. 바젤 신호등 = 최근 250일 99% VaR 초과 0~4 녹색 · 5~9 황색 · 10+ 적색.")
        msel = st.selectbox("차트 방법", V["methods"], 2, key="rk_vchart")
        Tt = V["table"].dropna(subset=[f"VaR {msel}"])
        hit = Tt["실현손실"] > Tt[f"VaR {msel}"]
        f = go.Figure([go.Scatter(x=Tt.index, y=Tt["실현손실"], mode="markers", marker=dict(size=3, color="#898781"), name="실현 손실"),
                       go.Scatter(x=Tt.index, y=Tt[f"VaR {msel}"], line=dict(color=PAL[0], width=1.4), name="VaR"),
                       go.Scatter(x=Tt.index, y=Tt[f"ES {msel}"], line=dict(color=PAL[1], width=1, dash="dot"), name="ES"),
                       go.Scatter(x=Tt.index[hit], y=Tt["실현손실"][hit], mode="markers", marker=dict(size=6, color=PAL[7]), name="초과")])
        st.plotly_chart(shade(fig_layout(f, 360, title=f"{msel}: 실현 손실 vs VaR·ES (%, 양수 = 손실)"), Tt.index), key="rk_vfig")
        ex = load_exp("risk")
        if ex is not None:
            x = ex[ex.구분 == "VaR"].copy(); x["통과"] = (x.Kupiec_p >= 0.05) & (x.Christoffersen_p >= 0.05)
            st.markdown("##### 12품목 요약 (⑥ 실험 결과 '리스크 모형', σ = GARCH(1,1)) — 백테스트 통과 비율 (Kupiec·Christoffersen 모두 p ≥ 0.05)")
            st.dataframe(x.groupby(["기준", "포지션", "항목"])["통과"].mean().unstack(0).style.format("{:.0%}"))
    with d:
        tids = st.multiselect("품목 (꼬리 통계)", ids_r, default=[i for i in RISK_DEF if i in ids_r][:6], format_func=lbl, key="rk_tids")
        if len(tids) >= 1:
            Rt = pd.DataFrame({i: RK.returns(_levels(extra_bytes, ROLL)[i]) for i in tids}).loc[str(start):str(end)]
            z_sel = (r / np.sqrt(S["EWMA 롤링 λ (직전 3년)"])).values if "EWMA 롤링 λ (직전 3년)" in S else None
            TS = pd.DataFrame({short(i): RK.tail_stats(Rt[i].dropna(), z_sel if i == rid_ else None) for i in tids}).T
            st.markdown("##### 꼬리 통계 (일간 수익률)")
            st.dataframe(TS.style.format("{:.3f}", na_rep="–"))
            st.caption("Hill α = 꼬리 지수 (작을수록 두꺼운 꼬리, 2~4가 원자재 전형; 정규분포는 무한대). 표준화 잔차의 초과첨도·|z|>3 비율은 "
                       "변동성 군집을 걸러낸 뒤에도 남는 꼬리 → 정규 VaR가 과소추정하는 이유. (표준화 잔차 항목은 위에서 추정한 품목만)")
        p_ = _levels(extra_bytes, ROLL)[rid_].dropna().loc[str(start):str(end)]
        dd = RK.drawdown(p_)
        st.markdown(f"##### {short(rid_)} 낙폭·급등 · 최악 구간 (롤 보정 가격)")
        g1_, g2_ = st.columns([1, 2])
        g1_.dataframe(pd.Series({k: (f"{v:.1f}" if isinstance(v, float) else str(v)) for k, v in dd.items() if not k.endswith("series")}, name="값"))
        f = go.Figure([go.Scatter(x=dd["series"].index, y=dd["series"], fill="tozeroy", line=dict(color=PAL[0], width=1), name="고점 대비 낙폭 (%)"),
                       go.Scatter(x=dd["up_series"].index, y=dd["up_series"], line=dict(color=PAL[7], width=1), name="저점 대비 상승 (%) — 구매자 위험")])
        g2_.plotly_chart(shade(fig_layout(f, 300, title="보유자 위험(낙폭) vs 구매자 위험(저점 대비 상승)"), p_.index), key="rk_dd")
        st.dataframe(RK.stress_table(p_).set_index(["기간 (영업일)", "방향"]).style.format({"변화 (%)": "{:+.1f}"}))
    with e:
        import procurement as PR
        st.markdown("##### 구매 계획·재고의 가격 위험 — 여러 품목을 함께 볼 때의 VaR · 기여도 · 위기 시나리오")
        pids = st.multiselect("품목", [i for i in PR.BUY_IDS if i in NAME], default=[i for i in ["yf:ZC=F", "yf:ZS=F", "yf:ZM=F"] if i in NAME],
                              format_func=lambda i: PR.BUY_IDS[i], key="rk_pids")
        dirl = st.radio("물량 성격", ["매입 예정 (아직 안 산 물량 → 가격 상승이 손실)", "보유 재고 (가격 하락이 손실)"], horizontal=True, key="rk_dir")
        sg = -1.0 if dirl.startswith("매입") else 1.0
        u1_, u2_ = st.columns(2)
        alpha_p = u1_.selectbox("신뢰수준", [0.95, 0.99], 1, format_func=lambda x: f"{x:.0%}", key="rk_palpha")
        hp = u2_.selectbox("기간 (영업일)", [5, 10, 20, 60], 2, key="rk_ph")
        Lr_ = _levels(extra_bytes, False)
        qc = st.columns(max(len(pids), 1))
        qty = {i: qc[j].number_input(f"{PR.BUY_IDS[i]} ({'온스' if PR.unit_of(i) == '$/oz' else '톤'})", 0.0, 1e8, 30000.0, 1000.0, key=f"rk_qty_{i}") for j, i in enumerate(pids)}
        if len(pids) >= 1 and sum(qty.values()) > 0:
            px = {i: float(PR.usd_t(Lr_, i).dropna().iloc[-1]) for i in pids}
            expo = pd.Series({i: sg * qty[i] * px[i] for i in pids})
            OO = {i: _risk_vm(i, ("EWMA 품목별 λ (확장)",), ROLL, extra_bytes) for i in pids}
            Rp = pd.DataFrame({i: OO[i]["r"] for i in pids}).dropna()
            Sp = pd.DataFrame({i: OO[i]["S"]["EWMA 품목별 λ (확장)"] for i in pids}).reindex(Rp.index).dropna()
            PRk = RK.position_risk(Rp.loc[Sp.index], Sp, expo, alpha_p, int(hp))
            w1_, w2_, w3_, w4_ = st.columns(4)
            w1_.metric("총 노출", f"${expo.abs().sum():,.0f}")
            w2_.metric(f"{hp}일 {alpha_p:.0%} VaR (정규)", f"${PRk['var_normal']:,.0f}", f"ES ${PRk['es_normal']:,.0f}", delta_color="off")
            w3_.metric(f"{hp}일 {alpha_p:.0%} VaR (다변량 FHS)", f"${PRk['var_fhs']:,.0f}", f"ES ${PRk['es_fhs']:,.0f}", delta_color="off")
            w4_.metric("분산 효과", f"${PRk['undiversified'] - PRk['var_normal']:,.0f}", f"단순 합 ${PRk['undiversified']:,.0f}", delta_color="off")
            st.dataframe(PRk["table"].rename(index=lambda i: PR.BUY_IDS[i]).style.format(
                {"노출 ($)": "{:,.0f}", "일 σ (%)": "{:.2f}", f"{hp}일 σ ($)": "{:,.0f}", "VaR 기여 ($, 정규)": "{:,.0f}", "기여 비중": "{:.0%}"}))
            st.caption("σ = 품목별 EWMA λ (walk-forward), 상관 = 직전 500일 표준화 잔차. 다변량 FHS = 과거 표준화 잔차 '벡터'(같은 날 동시 충격)를 현재 σ로 "
                       "다시 키운 연속 h일 시나리오 → 꼬리 동조까지 반영. 기여도 합 = 정규 VaR (오일러 분해) — 위험을 가장 많이 만드는 물량부터 헤지·선매입 검토.")
            f = go.Figure(go.Histogram(x=PRk["sims"], nbinsx=60, marker_color=PAL[0], opacity=0.8))
            f.add_vline(x=PRk["var_fhs"], line_color=PAL[7], annotation_text="FHS VaR")
            st.plotly_chart(fig_layout(f, 260, title=f"{hp}일 손실 시나리오 분포 (FHS, $, 양수 = 손실)", hovermode="closest"), key="rk_phist")
            CS = RK.crisis_scenarios(_levels(extra_bytes, ROLL), expo, A.CRISES)
            if len(CS):
                CS = CS.rename(columns={f"{i} 변화": PR.BUY_IDS[i] for i in pids})
                st.markdown("##### 과거 위기 구간을 지금 물량에 적용하면")
                st.dataframe(CS.style.format({PR.BUY_IDS[i]: "{:+.1%}" for i in pids} | {"손익 ($)": "{:+,.0f}"}, na_rep="–"))
                st.caption("손익 = 구간 시작~끝 가격 변화 × 현재 노출 (구간 중 최악 시점이 아니라 시작·끝 기준). 매입 예정 물량은 가격이 오른 위기(예: 2022년)에 손실.")


def vol_forecast_check():
    """③ 변동성 예측 검증: 8개 σ 모형의 1일 분산 예측 경주 (QLIKE·DM·MCS) + 잔차 진단 + 여러 품목 공통 λ·마할라노비스."""
    import risk as RK
    st.subheader("변동성 예측 검증 — 어떤 σ 모형이 내일의 변동성을 가장 잘 맞히나")
    st.caption("후보: RiskMetrics λ=0.94 · 품목별 EWMA λ(확장) · 롤링 λ(직전 3년) · HMM 3상태 λ · GARCH(1,1) · HAR-RV · MIDAS-RV — "
               "모두 walk-forward (매년 1월 재추정, 시점 t의 예측은 t−1 종가까지). 여기서 고른 모형을 ◆ 리스크 VaR의 σ로 사용.")
    ids_r = [i for i in COMMOD_IDS if GROUP.get(i) not in ("Freight",)]
    c1_, c2_, c3_ = st.columns([2, 1, 1])
    rid = c1_.selectbox("품목", ids_r, ids_r.index("yf:ZC=F") if "yf:ZC=F" in ids_r else 0, format_func=lbl, key="vf_id")
    ev0 = c2_.selectbox("평가 시작", [2005, 2010, 2015, 2020], 1, key="vf_ev0")
    if c3_.button("▶ 변동성 모형 경주 (약 10초)", type="primary", key="vf_run"):
        with st.spinner("변동성 모형 walk-forward 추정 중…"):
            st.session_state["vf_o"] = (rid, _risk_vm(rid, None, ROLL, extra_bytes))
    ex = load_exp("risk")
    if ex is not None:
        v = ex[ex.구분 == "변동성"].copy(); v["포함"] = v.MCS_p >= 0.10; v["순위"] = v.groupby("품목")["QLIKE"].rank()
        with st.expander("12품목 요약 (⑥ 실험 결과 '리스크 모형')", expanded="vf_o" not in st.session_state):
            st.dataframe(v.groupby("항목").agg(**{"MCS 포함 품목 수": ("포함", "sum"), "평균 QLIKE 순위": ("순위", "mean"),
                                                "DM t 중앙값 (vs RiskMetrics)": ("DM_t", "median")}).sort_values("평균 QLIKE 순위")
                         .style.format({"평균 QLIKE 순위": "{:.2f}", "DM t 중앙값 (vs RiskMetrics)": "{:+.2f}"}))
            st.caption("1일 예측은 GARCH(1,1)가 가장 안정적이고, 품목별 고정 λ가 그다음. 매년 λ를 바꾸는 롤링·HMM은 λ가 기간마다 다르다는 "
                       "통계적 근거(② 레짐 → 변동성 레짐)에도 불구하고 추정 잡음 때문에 예측이 나빠짐 → 시변 λ는 진단용, σ에는 GARCH·고정 λ.")
    if "vf_o" not in st.session_state:
        return
    rid_, o = st.session_state["vf_o"]
    r, S = o["r"], o["S"]
    ev = S.index >= pd.Timestamp(f"{ev0}-01-01")
    Lo = RK.qlike(r, S)[ev].dropna()
    base = st.selectbox("DM 기준 모형", list(S.columns), 0, key="vf_base")
    pv = RK.mcs(Lo, B=500); dm = RK.dm_table(Lo, base)
    T_ = pd.DataFrame({"평균 QLIKE": Lo.mean(), "MCS p": pv}).join(dm[["DM t", "p값 (양측)", "모형이 나은 날 비율"]])
    T_["90% MCS 포함"] = np.where(T_["MCS p"] >= 0.10, "✓", "")
    st.markdown(f"##### 1일 분산 예측 정확도 — {short(rid_)}, {Lo.index.min():%Y-%m}~ ({len(Lo):,}일)")
    st.dataframe(T_.sort_values("평균 QLIKE").style.format({"평균 QLIKE": "{:.4f}", "MCS p": "{:.3f}", "DM t": "{:+.2f}", "p값 (양측)": "{:.3f}",
                                                         "모형이 나은 날 비율": "{:.0%}"}, na_rep="–").highlight_min(subset=["평균 QLIKE"], props="font-weight:700"))
    st.caption(f"QLIKE = log σ̂² + r²/σ̂² (낮을수록 좋음). DM t < −1.96 → '{base}'보다 유의하게 우수 (Newey-West). "
               "MCS (Hansen·Lunde·Nason 2011) = 최선과 통계적으로 구별되지 않는 모형 집합 (T_max, 10일 블록 부트스트랩) — p ≥ 0.10이면 포함.")
    zm = st.selectbox("잔차 진단 모형", list(S.columns), list(S.columns).index("EWMA 롤링 λ (직전 3년)") if "EWMA 롤링 λ (직전 3년)" in S else 0, key="vf_zm")
    z = (r / np.sqrt(S[zm]))[ev].values
    sb = RK.sign_bias(z); lb = RK.ljung_box_sq(z, 10)
    q1_, q2_ = st.columns(2)
    q1_.markdown("**Engle-Ng 부호 편향 검정** (표준화 잔차²에 어제의 부호·크기가 남아 있나)")
    q1_.dataframe(sb["table"].style.format("{:+.3f}"))
    q1_.caption(f"결합 LM = {sb['LM']:.1f}, p = {sb['p']:.3f} — p < 0.05면 대칭 모형이 하락/상승 비대칭을 놓침 (원자재는 종종 '역레버리지': 상승 충격이 변동성↑).")
    q2_.markdown("**Ljung-Box (z², 10시차)** — 남은 변동성 군집")
    q2_.metric("Q(10)", f"{lb[0]:.1f}", f"p = {lb[1]:.3f}", delta_color="off")
    q2_.caption("p < 0.05면 모형이 변동성 군집을 다 흡수하지 못함.")
    st.markdown("##### 여러 품목: 공통 λ 우도비 검정 · 다변량 적합 진단 (마할라노비스)")
    mids = st.multiselect("품목", ids_r, default=[i for i in RISK_DEF if i in ids_r][:5], format_func=lbl, key="vf_mids")
    if st.button("▶ 여러 품목 검정", key="vf_mrun") and len(mids) >= 2:
        with st.spinner("품목별 EWMA 추정…"):
            mods = ("RiskMetrics λ=0.94", "EWMA 품목별 λ (확장)", "EWMA 롤링 λ (직전 3년)")
            OO = {i: _risk_vm(i, mods, ROLL, extra_bytes) for i in mids}
            lr = RK.lr_common_lambda({i: OO[i]["LL"] for i in mids})
            Rm = pd.DataFrame({i: OO[i]["r"] for i in mids}).dropna()
            mh = {m: RK.mahalanobis(Rm, pd.DataFrame({i: OO[i]["S"][m] for i in mids}).reindex(Rm.index), 252) for m in mods}
            st.session_state["vf_m"] = (mids, lr, pd.DataFrame(mh))
    if "vf_m" in st.session_state:
        from scipy import stats as _st
        mids_, lr, MH = st.session_state["vf_m"]
        K_ = len(mids_); MH = MH[MH.index >= pd.Timestamp(f"{ev0}-01-01")].dropna()
        c95 = _st.chi2.ppf(0.95, K_)
        l1_, l2_ = st.columns([1, 2])
        l1_.metric("H₀: 모든 품목 같은 λ", f"LR = {lr['LR']:.1f} (df {lr['df']})", f"p = {lr['p']:.2g} · 공통 λ = {lr['common_lambda']:.3f}", delta_color="off")
        l1_.dataframe(pd.DataFrame({"λ (전체 표본)": lr["own"], "반감기 (거래일)": {k: float(RK.hl_days(v)) for k, v in lr["own"].items()}})
                      .rename(index=short).style.format({"λ (전체 표본)": "{:.3f}", "반감기 (거래일)": "{:.1f}"}))
        l1_.caption("전체 표본 우도 (표본 내 검정). p < 0.05 → 품목마다 반감기가 다름 = 공통 λ(RiskMetrics)는 일부 품목을 과대·과소 평활.")
        summ = pd.DataFrame({"평균 d² (기대 = K)": MH.mean(), "χ² 95% 초과율 (기대 5%)": (MH > c95).mean()})
        l2_.dataframe(summ.style.format({"평균 d² (기대 = K)": "{:.2f}", "χ² 95% 초과율 (기대 5%)": "{:.1%}"}))
        mm = MH.rolling(63).mean()
        f = go.Figure([go.Scatter(x=mm.index, y=mm[c_], name=c_, line=dict(color=PAL[j % 8], width=1.4)) for j, c_ in enumerate(mm.columns)])
        f.add_hline(y=K_, line_color=PAL[2], line_dash="dash", annotation_text=f"기대값 K={K_}")
        l2_.plotly_chart(shade(fig_layout(f, 300, title="마할라노비스 d² 63일 평균 — K보다 높으면 공분산 과소추정 (위기에 급등)"), mm.index), key="vf_mh")


def vol_regime():
    """② 변동성 레짐: HMM 3상태(저·중·고변동) 필터 확률과 레짐별 감쇠율, 3년 블록 간 λ 동일성 우도비 검정."""
    import risk as RK
    st.markdown("##### 변동성 레짐 — 저·중·고변동 국면과 변동성 충격의 반감기가 시간에 따라 변하는가")
    st.caption("HMM 3상태 = 일간 수익률을 저·중·고변동 레짐으로 분류 (매년 그 이전 데이터로 재적합, 필터 확률만 사용). "
               "반감기 = ln 0.5 / ln λ (EWMA σ²ₜ = λσ²ₜ₋₁ + (1−λ)r²ₜ₋₁). 3년 블록별 λ와 '블록 간 동일' 우도비 검정으로 같은 품목 안의 시변성을 판정.")
    ids_r = [i for i in COMMOD_IDS if GROUP.get(i) not in ("Freight",)]
    c1_, c2_ = st.columns([3, 1])
    rid = c1_.selectbox("품목", ids_r, ids_r.index("yf:ZC=F") if "yf:ZC=F" in ids_r else 0, format_func=lbl, key="vr_id")
    if c2_.button("▶ 변동성 레짐 추정", type="primary", key="vr_run"):
        with st.spinner("HMM 3상태 walk-forward 재적합 중…"):
            st.session_state["vr_o"] = (rid, _risk_vm(rid, ("EWMA 품목별 λ (확장)", "HMM 3상태 λ"), ROLL, extra_bytes))
    if "vr_o" not in st.session_state:
        return
    rid_, o = st.session_state["vr_o"]
    r = o["r"]; hm = o["hmm"]
    if hm is not None and np.isfinite(hm["prob"]).any():
        P_ = pd.DataFrame(hm["prob"], index=r.index, columns=["저변동", "중변동", "고변동"]).dropna()
        f = go.Figure([go.Scatter(x=P_.index, y=P_[c_], stackgroup="one", name=c_, line=dict(width=0.5, color=PAL[[2, 3, 7][j]]))
                       for j, c_ in enumerate(P_.columns)])
        st.plotly_chart(shade(fig_layout(f, 280, title=f"{short(rid_)} HMM 3상태 필터 확률 (그 시점까지의 정보)", yaxis_range=[0, 1]), P_.index), key="vr_hmmp")
        st.dataframe(hm["table"].set_index("연도").tail(10).style.format("{:.3f}"))
        st.caption("레짐별 λ = 그 레짐에 속한 날들의 가중 QMLE. 옆 '수익률 레짐'(Markov-switching, 수익률 평균·분산 레짐)과 상호 보완 — "
                   "이쪽은 '변동성 수준' 레짐.")
    lt = RK.lr_time_lambda(o["LL"], r.index, 3)
    m1_, m2_ = st.columns([2, 1])
    m1_.dataframe(lt["table"].set_index("기간").style.format({"λ": "{:.3f}", "95% 하한": "{:.3f}", "95% 상한": "{:.3f}", "반감기 (거래일)": "{:.1f}", "관측": "{:,.0f}"}))
    m2_.metric("우도비 검정: 3년 블록 간 λ 동일?", f"LR = {lt['LR']:.1f} (df {lt['df']})", f"p = {lt['p']:.4f}", delta_color="off")
    m2_.caption("p < 0.05 → 같은 품목의 반감기가 기간마다 유의하게 다름. 단, 이를 예측에 쓰면(롤링·HMM λ) 오히려 나빠짐 — ③ 변동성 예측 검증 참고.")


# ============================================================================
# ⑥ 실험 결과 (비교 실험 모음, 재실행 가능)
# ============================================================================
def _exp_header(name):
    import experiments as EX
    E = EX.EXPERIMENTS[name]
    df, meta = EX.load(name)
    h1, h2 = st.columns([4, 1])
    h1.caption(f"마지막 실행: {meta.get('run_at', '–')}" + (f" · 소요 {meta['seconds']}초" if "seconds" in meta else "")
               + (" · 설정: " + ", ".join(f"{k}={v}" for k, v in meta.get("settings", {}).items()) if meta.get("settings") else ""))
    if h2.button(f"재실행 ({E['eta']})", key=f"ex_run_{name}"):
        prog = st.progress(0.0, f"{E['title']} 실행 중…")
        E["fn"](progress=lambda x: prog.progress(min(1.0, x)))
        prog.empty(); st.rerun()
    if df is None:
        st.info("결과 파일이 이전 형식입니다 — 재실행을 누르세요." if meta.get("stale") else "아직 결과가 없습니다 — 재실행을 누르세요.")
    return df


META_READ = ("읽는 법 (2026-10-19 실행): 6품목 × 일간·주간 = 156칸 중 **위약 95% 분위를 넘은 전략은 0개**, '후보 중 최댓값' 위약 p는 12개 설정 모두 ≥ 0.47. "
             "SPA로 'Buy&Hold보다 낫다'·'과거 평균보다 오차가 작다'를 기각한 경우도 없음 → **시기·레짐별 모델 선택으로 얻는 예측력은 운과 구별되지 않음.** "
             "SPA가 '순수익 > 0'을 기각한 것은 대두박뿐인데, 이는 대두박 자체의 상승 추세(B&H Sharpe 0.6)를 롱 비중이 높은 전략이 따라간 것. "
             "품목 평균으로는 (e) 과거 평균(사실상 추세 방향 보유)이 두 하위기간 모두 가장 좋고, (c) 레짐 조건부는 (a)·(b)·(d)보다 낫지만 (e)에 못 미침. "
             "(b) 가중 결합·(d) 동일가중은 예측이 0 근처로 줄어 부호가 자주 바뀌어(회전율↑) 비용 후 최하위. "
             "단일 분할에서 보이던 Sharpe 0.5는 롱 편향 + 여러 모델 중 최댓값 고르기의 결과로 해석됨. 위약은 각 전략의 롱 비중·전환 빈도를 맞춘 무작위 포지션.")
EX_DATA = A.DATA

# ⑥ 개요: 실험마다 '무엇을 물었고 → 결론 → 대시보드 어디에 반영됐나'. (분류, 실험 이름=radio 값, 질문, 결론, 반영, 판정)
EX_OVERVIEW = [
    ("구매", "구매 타이밍 (프리셋 · 단계별)", "기준 프리셋에서 신호를 하나씩 바꾸면 매입가가 얼마나 싸지나? 어떤 프리셋이 가장 좋나?",
     "상대가치(크러시 마진·옥수수/밀 비율)만 전·후반 모두 뚜렷. '추세만'은 손해, 공격적 프리셋은 꼬리 손실 큼",
     "★ 구매 타이밍 '추천' 프리셋의 근거", "✅ 채택"),
    ("구매", "철강 원료·구리 상대가치", "곡물에서 통한 상대가치를 철광석·점결탄·구리에도 쓸 수 있나? 어떤 관계로?",
     "철광석 = SGX/DCE 차익, 점결탄 = 고로 마진이 두 기간 모두 +. 구리/알루미늄은 후반에 무너짐, 구리/금은 작음",
     "★ 금속 3품목의 상대가치 정의", "✅ 채택 (구리는 참고)"),
    ("구매", "전 품목 구매 타이밍 (단일 vs 다변량)", "곡물 외 에너지·귀금속·소프트·축산·DCE 품목에도 상대가치가 통하나? 대체재 하나 대신 여러 관계를 묶으면(다변량) 나아지나?",
     "무관한 가격비(위약)도 31%가 '두 기간 개선'을 통과 → 위약·자기 가격 대조군을 넘은 관계만 인정하면 확장 28품목 중 10개(가스/원유·생우/송아지·"
     "알루미늄/구리·브렌트 경유 크랙 등). 다변량은 단일 관계 평균 수준. 기존 곡물 정의도 엄격 기준으론 대두 크러시만 통과 → 재검증 필요",
     "★ 구매 타이밍 전 품목 선택 + 품목별 채택 관계", "⚠️ 일부만"),
    ("리스크", "리스크 모형", "변동성 예측·VaR·4주 가격 범위는 어떤 모형이 실제로 잘 맞았나?",
     "1일 분산 = GARCH(1,1) 최상, 99% VaR는 정규분포 탈락·FHS/EVT 정확, 4주 범위는 경험적 분위가 적중률 개선",
     "◆ 리스크 기본 σ = GARCH, ★ 4주 범위 = 경험적 분위", "✅ 채택"),
    ("예측", "모델 선택 메타 전략", "시기·레짐마다 ML 모델을 골라 쓰면 수익률 방향 예측이 나아지나?",
     "156칸 중 무작위 위약 95%를 넘은 전략 0개 → 모델 선택으로 얻는 예측력은 운과 구별 안 됨",
     "④ 메타 전략은 '참고용', 기본 기준선 = 과거 평균", "❌ 효과 없음"),
    ("예측", "최신 시계열 모델 · 예측 목표", "Chronos·TimesFM 같은 최신 모델로, 또는 목표를 바꾸면(변동성·스프레드·순위) 예측이 되나?",
     "가격 방향 ✖ · 구매 '기다릴까' ✖(AUC 0.5) · 변동성은 GARCH가 여전히 최상 · 철광석 SGX/DCE 스프레드와 품목 순위(IC 0.10)는 일부 예측 가능",
     "방향 예측 대신 상대가치·변동성에 집중하는 근거", "⚠️ 일부만"),
    ("포트폴리오", "전략 비교 (리스크 버짓)", "신호를 비중에 얼마나 세게 반영할지(τ)와 리스크 패리티·최적화·1/N 중 무엇이 나은가?",
     "약한 신호는 τ 작게, 크기 정보가 있는 신호는 τ 크게 → τ = 0.5가 무난",
     "⑤ 포트폴리오 기본 τ = 0.5", "✅ 채택"),
    ("포트폴리오", "모멘텀 기간", "모멘텀을 몇 개월로 잴 때 가장 좋은가?",
     "업계 표준 12-1개월이 최고, 짧을수록 회전율만 늘고 성과 하락",
     "⑤ 모멘텀 기본값 12-1개월", "✅ 채택"),
    ("포트폴리오", "거래 규칙", "매일 리밸런싱 vs 필요할 때만 거래(무거래 구간) 중 무엇이 나은가?",
     "무거래 구간이 비슷한 Sharpe를 훨씬 적은 거래로 달성",
     "⑤ 무거래 구간 규칙, ④ 무거래 구간 c", "✅ 채택"),
    ("포트폴리오", "변동성 사이징", "변동성 예측으로 포지션 크기를 조절하면 좋아지나?",
     "위험(실현 변동성·낙폭)은 확실히 안정, Sharpe는 거의 그대로 (롱온리는 하락)",
     "⑤ 변동성 타기팅 = 위험 관리용 옵션", "⚠️ 위험 관리용"),
]


def ex_overview():
    import experiments as EX
    st.markdown("##### 실험 한눈에 보기 — 무엇을 물었고, 결론은 무엇이고, 대시보드 어디에 반영됐나")
    st.caption("⑥ 실험 결과 = 다른 탭의 기본값·프리셋을 정한 근거 백테스트 모음 (모두 미래정보 차단·같은 비용 가정). "
               "실험 이름을 위 버튼에서 고르면 상세 표·그림이 나오고, 데이터가 갱신되면 각 실험의 '재실행'으로 최신화. "
               "판정은 '두 하위기간 모두 개선 + 무작위 위약보다 나음' 기준.")
    files = {"구매 타이밍 (프리셋 · 단계별)": "procure_steps", "철강 원료·구리 상대가치": "procure_metals", "전 품목 구매 타이밍 (단일 vs 다변량)": "procure_all", "리스크 모형": "risk", "모델 선택 메타 전략": "metasel",
             "최신 시계열 모델 · 예측 목표": "forecast_lab", "전략 비교 (리스크 버짓)": "rb", "모멘텀 기간": "momentum", "거래 규칙": "rules", "변동성 사이징": "vol"}
    rows = []
    for grp, nm, q, concl, used, verdict in EX_OVERVIEW:
        try:
            run_at = EX.load(files[nm])[1].get("run_at", "–")
        except Exception:
            run_at = "–"
        rows.append({"분류": grp, "실험": nm, "판정": verdict, "질문": q, "결론": concl, "대시보드 반영": used, "마지막 실행": str(run_at)[:16]})
    T = pd.DataFrame(rows)
    cnt = T["판정"].str[0].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("✅ 채택 (기본값에 반영)", int(cnt.get("✅", 0)))
    c2.metric("⚠️ 일부만 / 위험 관리용", int(cnt.get("⚠", 0)))
    c3.metric("❌ 효과 없음 (운과 구별 안 됨)", int(cnt.get("❌", 0)))
    st.table(T.set_index("실험"))   # 긴 문장이 잘리지 않게 줄바꿈 표
    st.caption("큰 그림: 가격 '방향'을 맞히려는 시도(ML·메타 전략·최신 시계열 모델)는 모두 운과 구별되지 않았고, 효과가 확인된 것은 "
               "① 실제 시세 관계(상대가치) ② 변동성·꼬리 위험 추정 ③ 거래를 줄이는 규칙 — 대시보드는 이 세 가지를 중심으로 구성됨.")


def ex_runs():
    """⑥ 내 실행 기록: runs/registry.jsonl 목록 · 운 대비 성과 그림 · 판정 · 상세 · 두 실행 비교 · ④에 설정 불러오기."""
    import runlog as RL
    st.markdown("##### 내 실행 기록 — ④ 수익률 예측에서 ▶ 모델 실행할 때마다 설정·결과가 `runs/` 폴더에 자동 저장")
    st.caption("판정(좋음·보통·나쁨)과 메모를 남겨 두면 나중에 어떤 조합이 왜 좋았는지 비교 가능. '같은 설정 실행 수' = 설정 지문이 같은 실행 "
               "(데이터·코드만 바뀐 재실행). 코드 = git 버전, 데이터 = 정제 데이터 파일 시각.")
    R = RL.load_registry()
    if R.empty:
        st.info("아직 기록이 없습니다 — ④ 수익률 예측에서 ▶ 모델 실행을 누르면 자동으로 기록됩니다.")
        return
    f1, f2, f3 = st.columns([1.2, 1.2, 2])
    vsel = f1.multiselect("판정", list(RL.VERDICTS), list(RL.VERDICTS), format_func=RL.VERDICTS.get, key="rn_v")
    wfo = f2.checkbox("walk-forward 실행만", False, key="rn_wf")
    q = f3.text_input("제목 검색 (품목·모델 등)", "", key="rn_q")
    V = R[R.verdict.isin(vsel)]
    if wfo and "WF Sharpe(비용 후)" in V:
        V = V[V["WF Sharpe(비용 후)"].notna()]
    if q:
        V = V[V.title.str.contains(q, case=False, regex=False)]
    if V.empty:
        st.info("조건에 맞는 기록이 없습니다."); return
    if "WF 초과(운 대비)" in V and V["WF 초과(운 대비)"].notna().any():
        G = V[V["WF 초과(운 대비)"].notna()].iloc[::-1]
        col = {"good": PAL[2], "bad": PAL[7], "neutral": PAL[3], "pending": "#898781"}
        f = go.Figure(go.Bar(x=G["at"].str[5:16] + " · " + G["title"].str.slice(0, 28), y=G["WF 초과(운 대비)"],
                             marker_color=[col[v] for v in G.verdict],
                             customdata=np.c_[G["WF 최고 모델"], G["WF Sharpe(비용 후)"], G["WF 위약 최댓값95"]],
                             hovertemplate="%{x}<br>최고 %{customdata[0]}: Sharpe(비용 후) %{customdata[1]:.2f}<br>위약 최댓값 95% %{customdata[2]:.2f}"
                                           "<br>초과 %{y:+.2f}<extra></extra>"))
        f.add_hline(y=0, line_color="#898781", line_dash="dot", annotation_text="운의 경계 (위약 최댓값 95%)")
        st.plotly_chart(fig_layout(f, 320, title="walk-forward 최고 모델의 '운 대비 초과 Sharpe' (비용 후 − 위약 최댓값 95%) — 0 위여야 운 이상 · 색 = 판정"),
                        key="rn_bar")
    cols = [c for c in ["at", "verdict", "title", "WF 최고 모델", "WF Sharpe(비용 전)", "WF Sharpe(비용 후)", "WF 위약 최댓값95", "WF 적중률",
                        "단일 최고 모델", "단일 Sharpe(비용 후)", "단일 위약 최댓값95", "same_config_runs", "note", "code", "run_id"] if c in V]
    T = V[cols].copy()
    T["verdict"] = T.verdict.map(RL.VERDICTS)
    T["hints"] = V.hints.map(lambda h: " / ".join(h) if isinstance(h, list) else "")
    T = T.rename(columns={"at": "시각", "verdict": "판정", "title": "제목", "same_config_runs": "같은 설정 실행 수", "note": "메모", "code": "코드",
                          "hints": "자동 판정 힌트"})
    st.dataframe(T.set_index("시각").style.format({c: "{:+.2f}" for c in T.columns if "Sharpe" in c or "위약" in c} | {"WF 적중률": "{:.1%}"}, na_rep="–"),
                 height=min(420, 40 + 35 * len(T)))
    st.markdown("**실행 상세 · 판정**")
    rid = st.selectbox("실행", list(V.run_id), format_func=lambda i: f"{R.set_index('run_id').loc[i, 'at']} · {R.set_index('run_id').loc[i, 'title']}",
                       key="rn_sel")
    M = RL.load_run(rid)
    d1, d2, d3 = st.columns([1.2, 3, 1.2])
    vv = d1.selectbox("판정", list(RL.VERDICTS), list(RL.VERDICTS).index(M["verdict"]), format_func=RL.VERDICTS.get, key=f"rn_vv_{rid}")
    nt = d2.text_input("메모", M.get("note") or "", key=f"rn_nt_{rid}")
    if d1.button("판정 저장", key=f"rn_save_{rid}"):
        RL.set_verdict(rid, vv, nt); st.rerun()
    if M.get("kind") == "ml":
        d3.button("④에 이 설정 불러오기", key=f"rn_restore_{rid}", on_click=_restore_ml, args=(M["settings"],), type="primary",
                  help="④ 수익률 예측의 위젯을 이 실행의 설정으로 되돌림 → ④ 탭에서 ▶ 모델 실행")
    if M.get("hints"):
        st.caption("자동 판정 힌트: " + " · ".join(M["hints"]))
    with st.expander("설정 전체 (재현용)"):
        st.json(M["settings"], expanded=False)
        st.caption(f"코드 {M['code']} · 데이터 {M['data']} · 설정 지문 {M['config_hash']} · 폴더 runs/{M['path']}")
    for nm, Tb in M["tables"].items():
        if nm == "walk_forward_cum":
            f = go.Figure([go.Scatter(x=pd.to_datetime(Tb.index), y=Tb[c], name=c,
                                      line=dict(color="#898781" if c == "Buy & Hold" else PAL[j % 8], width=1.2 if c == "Buy & Hold" else 1.6,
                                                dash="dot" if c == "Buy & Hold" else None)) for j, c in enumerate(Tb.columns)])
            st.plotly_chart(fig_layout(f, 300, title="walk-forward 누적 성과 (비용 전)"), key=f"rn_cum_{rid}")
        else:
            st.markdown(f"*{ {'metrics': '성과 지표 (단일 분할)', 'walk_forward': 'walk-forward'}.get(nm, nm) }*")
            st.dataframe(Tb.style.format("{:.4f}", subset=list(Tb.select_dtypes("number").columns), na_rep="–"))
    if len(V) >= 2:
        with st.expander("두 실행 비교 (설정 차이 · 핵심 지표)"):
            c1, c2 = st.columns(2)
            ids = list(V.run_id)
            a_ = c1.selectbox("A", ids, 0, key="rn_a", format_func=lambda i: R.set_index("run_id").loc[i, "title"] + " · " + i[:15])
            b_ = c2.selectbox("B", ids, 1, key="rn_b", format_func=lambda i: R.set_index("run_id").loc[i, "title"] + " · " + i[:15])
            A_, B_ = RL.load_run(a_), RL.load_run(b_)
            st.dataframe(RL.settings_diff(A_["settings"], B_["settings"]))
            key_ = [c for c in ["WF 최고 모델", "WF Sharpe(비용 전)", "WF Sharpe(비용 후)", "WF 위약 최댓값95", "WF 초과(운 대비)", "단일 Sharpe(비용 후)"] if c in R]
            st.dataframe(R.set_index("run_id").loc[[a_, b_], key_].rename(index={a_: "A", b_: "B"}).T.astype(str))


def tab_experiments():
    st.subheader("실험 결과 — 설정을 바꿔가며 같은 백테스트 엔진으로 비교한 결과")
    st.caption("모든 실험은 미래정보 차단(다음 거래일 체결·walk-forward)·같은 비용 가정. 데이터가 갱신되면 '재실행'으로 최신화. "
               "주의: 표에서 가장 좋은 설정을 골라 쓰면 과적합 — 여러 조건에서 일관되게 좋은 설정을 고를 것.")
    sec = st.radio("실험", ["📋 한눈에 보기", "📂 내 실행 기록", "구매 타이밍 (프리셋 · 단계별)", "철강 원료·구리 상대가치", "전 품목 구매 타이밍 (단일 vs 다변량)", "리스크 모형", "모델 선택 메타 전략", "최신 시계열 모델 · 예측 목표", "전략 비교 (리스크 버짓)",
                          "모멘텀 기간", "거래 규칙", "변동성 사이징"],
                   horizontal=True, key="ex_sec")
    pct = lambda cols: {c: "{:.2%}" for c in cols}
    pc_view = None
    if sec == "📋 한눈에 보기":
        ex_overview(); return
    if sec == "📂 내 실행 기록":
        ex_runs(); return
    if sec == "구매 타이밍 (프리셋 · 단계별)":
        pc_view = st.radio("보기", ["단계별 고도화 (기준에서 한 가지씩)", "프리셋 · 신호 비교 (13·26주, 품목별)"], horizontal=True, key="ex_pc_view",
                           help="두 실험 모두 곡물 6품목 · 같은 백테스트 엔진. 단계별 = '추천' 프리셋이 나온 근거, 프리셋 비교 = 최종 프리셋과 단순 신호의 기간별 비교")
    if pc_view and pc_view.startswith("단계별"):
        st.markdown("##### 구매 신호 고도화 — 기준(균형 프리셋)에서 한 가지씩 바꿨을 때의 효과 (6품목 × 13주 구매 기간, 2008~)")
        df = _exp_header("procure_steps")
        if df is None:
            return
        d = df.set_index(["단계", "변형"])
        f = go.Figure()
        for j, c in enumerate(["전반_2008_16", "후반_2017_현재"]):
            f.add_trace(go.Bar(y=[f"{a} · {b}" for a, b in d.index], x=d[c], orientation="h", name=c.replace("_", " "), marker_color=PAL[j]))
        f.add_vline(x=float(d.loc[("0 기준",)].iloc[0]["후반_2017_현재"]), line_dash="dot", line_color=PAL[1], annotation_text="기준(후반)")
        f.update_layout(height=120 + 30 * len(d), margin=dict(l=10, r=10, t=40, b=10), barmode="group", yaxis=dict(autorange="reversed"),
                        title="평균 절감 (bp, 균등 매수 대비) — 두 기간 모두 기준보다 오른쪽이면 개선")
        st.plotly_chart(f, key="ex_steps_bar")
        st.dataframe(d.style.format({"평균절감bp": "{:+.1f}", "균등보다싸게": "{:.0%}", "하위5bp": "{:+.0f}", "t값_비중첩": "{:+.2f}",
                                     "전반_2008_16": "{:+.1f}", "후반_2017_현재": "{:+.1f}", "고변동국면": "{:+.1f}", "저변동국면": "{:+.1f}"}))
        st.caption("읽는 법: 상대가치(실제 시세 관계)만 후반에도 뚜렷 — 같은 아이디어를 롤보정 상대성과로 만들면 효과가 사라짐 → 크러시 마진·대체 가격비 같은 "
                   "경제적 관계가 핵심. MIDAS 위험 판단·6개월 창·최근가중 계절은 소폭 개선과 꼬리 위험 감소. 보유비용 3bp/주는 결과에 거의 영향 없음 "
                   "(규칙이 매수 시점을 크게 앞당기지 않음). '추천' 프리셋 = 상대가치 40% + 위치 27% + 반등 18% + 최근가중 계절 15%, MIDAS 위험, 6개월 창.")
    elif pc_view:
        st.markdown("##### 프리셋 · 신호별 평균 매입가 절감 (균등 매수 대비, bp)")
        df = _exp_header("procure")
        if df is None:
            return
        wk = st.radio("구매 기간", sorted(df["구매기간주"].unique()), horizontal=True, key="ex_pc_wk", format_func=lambda x: f"{x}주")
        d = df[df.구매기간주 == wk]
        P_ = d.pivot_table(index="신호", columns="기간", values=["평균절감bp", "균등보다싸게", "하위5bp"], aggfunc="mean")
        f = go.Figure()
        for j, per in enumerate(["2008-2016", "2017-현재", "전체"]):
            s_ = d[d.기간 == per].groupby("신호")["평균절감bp"].mean()
            f.add_trace(go.Bar(x=s_.index, y=s_.values, name=per, marker_color=PAL[j]))
        f.add_hline(y=0, line_color="#222")
        st.plotly_chart(fig_layout(f, 340, barmode="group", title="품목 평균 절감 (bp) — 두 하위기간 모두 +여야 신뢰", hovermode="x"), key="ex_pc_bar")
        st.dataframe(P_.style.format("{:+.1f}").format({c: "{:.0%}" for c in P_.columns if c[0] == "균등보다싸게"}))
        with st.expander("품목별 상세"):
            st.dataframe(d.set_index(["품목", "신호", "기간"]).drop(columns="구매기간주").style.format(
                {"평균절감bp": "{:+.1f}", "중앙값bp": "{:+.1f}", "균등보다싸게": "{:.0%}", "하위5bp": "{:+.0f}", "사후최저bp": "{:+.0f}", "일괄매수bp": "{:+.0f}"}))
        st.caption("읽는 법: '추세만'이 음수 = 오를 때 서둘러 사면 손해. '가격 위치'·'공격적'은 평균은 크지만 '하위 5%'(꼬리 손실)도 큼. "
                   "보수적·공격적은 '추천'보다 열위라 구매 타이밍 탭에서는 고급 옵션으로만 노출. "
                   "사후 최저(수백 bp)는 이론 상한 — 현실 규칙과의 격차가 '타이밍 예측의 한계'.")
    elif sec == "철강 원료·구리 상대가치":
        st.markdown("##### 철광석·점결탄·구리 — '추천' 프리셋에서 상대가치 정의만 바꿨을 때 (균등 매수 대비, bp)")
        df = _exp_header("procure_metals")
        if df is None:
            return
        wk = st.radio("구매 기간", sorted(df["구매기간주"].unique()), 1, horizontal=True, key="ex_mt_wk", format_func=lambda x: f"{x}주")
        d = df[df.구매기간주 == wk].set_index(["품목", "상대가치"]).drop(columns="구매기간주")
        f = go.Figure()
        for j, c in enumerate(["전반bp", "후반bp"]):
            f.add_trace(go.Bar(y=[f"{a} · {b}" for a, b in d.index], x=d[c], orientation="h", name={"전반bp": "전반 (~2019.6)", "후반bp": "후반 (2019.7~)"}[c],
                               marker_color=PAL[j]))
        f.add_vline(x=0, line_color="#222")
        f.update_layout(height=120 + 30 * len(d), margin=dict(l=10, r=10, t=40, b=10), barmode="group", yaxis=dict(autorange="reversed"),
                        title="평균 절감 (bp) — 두 기간 모두 0 오른쪽 + '상대가치 없음'보다 커야 채택")
        st.plotly_chart(f, key="ex_mt_bar")
        st.dataframe(d.style.format({"평균절감bp": "{:+.1f}", "중앙값bp": "{:+.1f}", "t값_비중첩": "{:+.2f}", "전반bp": "{:+.1f}", "후반bp": "{:+.1f}",
                                     "균등보다싸게": "{:.0%}", "하위5bp": "{:+.0f}", "기간수": "{:.0f}"}))
        st.caption("읽는 법: 철광석은 SGX 달러 가격과 DCE 위안 가격(환율·증치세 환산)의 차익, 점결탄은 철근 고로 마진(철근 − 1.6·철광석 − 0.5·코크스)이 "
                   "8·13·26주 모두에서 전·후반 + → 채택. 구리는 대체재 알루미늄 비율이 후반에 무너져 기각, 구리/금은 일관되지만 작아(t≈1) 참고 수준. "
                   "표본이 2013년부터라 곡물보다 검증 기간이 짧고, DCE 연속월물은 롤 보정이 없어 신호에 만기 교체 갭이 섞일 수 있음.")
    elif sec == "전 품목 구매 타이밍 (단일 vs 다변량)":
        import procurement as PR
        st.markdown("##### 전 품목 — 상대가치를 무엇으로 정할까: 없음 vs 단일 관계 vs 다변량 ('추천' 프리셋, 균등 매수 대비 개선 bp)")
        df = _exp_header("procure_all")
        if df is None:
            return
        import experiments as EX
        dpl, _m = EX.load("procure_placebo")
        G = EX.rel_adoption(df, dpl)
        g = G[G.종류 != "대조군"].copy()
        g["통과"] = g.엄격채택
        g["채택"] = [PR.REL_CHOSEN.get(i) == o or (i in PR.REL_DEF and i not in PR.REL_CHOSEN and o.endswith("(현재)")) for i, o in zip(g.id, g.상대가치)]
        typ = st.radio("보기", ["품목 × 선택지 (개선 bp)", "판정 근거 (위약 · 자기 가격 대조군)", "단일 vs 다변량 요약"], horizontal=True, key="ex_pa_view")
        if typ.startswith("품목"):
            g["선택지"] = np.where(g.종류 == "다변량", g.상대가치.str.replace("다변량: ", "◆ ", regex=False), "단일 " + g.groupby("품목").cumcount().add(1).astype(str))
            M = g.pivot(index="품목", columns="선택지", values="개선")
            L_ = g.pivot(index="품목", columns="선택지", values="상대가치")
            P_ = g.pivot(index="품목", columns="선택지", values="통과")
            C_ = g.pivot(index="품목", columns="선택지", values="채택")
            M = M.loc[M.max(axis=1).sort_values(ascending=False).index]
            L_, P_, C_ = L_.loc[M.index], P_.loc[M.index], C_.loc[M.index]
            txt = [[("" if pd.isna(M.iat[r, c]) else f"{M.iat[r, c]:+.0f}" + (" ★" if C_.iat[r, c] is True else " ✓" if P_.iat[r, c] is True else ""))
                    for c in range(M.shape[1])] for r in range(M.shape[0])]
            vm = vmax_of(M.values, step=5)
            f = go.Figure(go.Heatmap(z=M.values, x=list(M.columns), y=list(M.index), zmin=-vm, zmax=vm, colorscale=DIV, text=txt, texttemplate="%{text}",
                                     customdata=L_.values, hovertemplate="%{y} · %{customdata}<br>개선 %{z:+.1f}bp<extra></extra>", colorbar=dict(thickness=12)))
            f.update_layout(height=140 + 24 * len(M), margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(autorange="reversed", dtick=1),
                            title="'상대가치 없음' 대비 평균 개선 (bp, 13·26주 평균) — ✓ = 엄격 기준 통과, ★ = 대시보드 채택 (마우스 = 관계 이름)")
            st.plotly_chart(f, key="ex_pa_heat")
        elif typ.startswith("판정"):
            B_ = g.sort_values("개선", ascending=False).groupby("품목").head(1).set_index("품목")
            B_ = B_[["상대가치", "개선", "위약90", "대조군개선", "두기간통과", "엄격채택"]].sort_values("개선", ascending=False)
            f = go.Figure()
            f.add_trace(go.Bar(y=B_.index, x=B_.개선, orientation="h", name="최고 후보 개선", marker_color=[PAL[2] if v else "#898781" for v in B_.엄격채택],
                               customdata=B_.상대가치, hovertemplate="%{y} · %{customdata}: %{x:+.1f}bp<extra></extra>"))
            f.add_trace(go.Scatter(y=B_.index, x=B_.위약90, mode="markers", name="위약 90% (무관한 품목 가격비)", marker=dict(symbol="line-ns-open", size=14, color=PAL[7])))
            f.add_trace(go.Scatter(y=B_.index, x=B_.대조군개선, mode="markers", name="대조군 (자기 가격 1년 위치)", marker=dict(symbol="diamond", size=8, color=PAL[3])))
            f.update_layout(height=140 + 22 * len(B_), margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(autorange="reversed", dtick=1),
                            title="품목별 최고 후보 vs 기준선 — 초록 막대 = 위약 90%·대조군을 모두 넘어 채택")
            st.plotly_chart(f, key="ex_pa_judge")
            st.dataframe(B_.style.format({"개선": "{:+.1f}", "위약90": "{:+.1f}", "대조군개선": "{:+.1f}"}, na_rep="–"))
            st.caption("위약 = 그 품목과 경제적 연결이 없는 다른 섹터 품목(10개 무작위)과의 가격비를 상대가치로 넣은 결과 — 무의미한 관계도 36%가 '두 기간 개선'을 통과. "
                       "이유: 내 품목 / 아무 품목 비율의 분위는 내 가격의 1년 위치를 상당 부분 담음 → 대조군(자기 가격 1년 위치)과 위약을 모두 넘어야 '대체재 정보'로 인정.")
        else:
            k = g.groupby(["품목", "종류"]).개선.agg(["mean", "max"]).unstack()
            k.columns = [f"{b} {a}" for a, b in k.columns]
            k = k.dropna(subset=["다변량 mean"]).sort_values("다변량 max", ascending=False)
            f = go.Figure()
            for j, (c, nm) in enumerate([("단일 mean", "단일 관계 평균 (고르지 않을 때)"), ("단일 max", "단일 관계 최고 (사후 선택 · 낙관)"), ("다변량 max", "다변량 2종 중 나은 것")]):
                f.add_trace(go.Bar(x=list(k.index), y=k[c], name=nm, marker_color=[PAL[0], "#898781", PAL[1]][j]))
            f.add_hline(y=0, line_color="#898781")
            st.plotly_chart(fig_layout(f, 360, barmode="group", title="품목별 개선 (bp) — 다변량은 단일 평균과 비슷, 사후 최고 단일보다는 낮음", hovermode="x"), key="ex_pa_bar")
            pr = G[G.종류.isin(["단일", "다변량"])].groupby("종류").두기간통과.mean()
            c1, c2, c3 = st.columns(3)
            c1.metric("두 기간 통과율 · 단일 / 다변량", f"{pr.get('단일', np.nan):.0%} / {pr.get('다변량', np.nan):.0%}")
            if dpl is not None:
                pg = dpl[dpl.종류 == "위약"].groupby(["id", "선택지"]).agg(a=("개선_전반", "min"), b=("개선_후반", "min"))
                c2.metric("두 기간 통과율 · 위약 (무관한 관계)", f"{((pg.a > 0) & (pg.b > 0)).mean():.0%}")
            c3.metric("엄격 기준 통과 품목", f"{g[g.통과].품목.nunique()} / {g.품목.nunique()}")
        with st.expander("전체 표 (구매 기간별)"):
            st.dataframe(df.set_index(["품목", "상대가치", "구매기간주"]).drop(columns=["id"]).style.format(
                {c: "{:+.1f}" for c in df.columns if c.endswith("bp") or c.startswith("개선")} | {"t값_비중첩": "{:+.2f}", "균등보다싸게": "{:.0%}"}, na_rep="–"))
        st.caption("엄격 채택 = ① 13·26주 모두 전·후반 개선 ② +5bp 이상 ③ 위약 90% 초과 ④ 자기 가격 대조군 초과. "
                   "읽는 법: 같은 품목 안의 선택지는 같은 평가 기간(모든 관계 자료가 쌓인 뒤, 2008~)으로 비교, 전·후반 = 그 기간의 중간에서 2등분. "
                   "다변량 = ① 후보 관계들의 1년 분위를 평균 ② 대체재·원료 가격으로 '적정가'를 매월 직전 3년 회귀(공적분)해 잔차가 낮으면 쌈. "
                   "천연가스(가스/원유)·생우(비육 마진)·알루미늄(대 구리)·브렌트(경유 크랙)는 연료·사료·대체재라는 경제적 연결이 뚜렷하고 대조군을 크게 넘음. "
                   "돈육/옥수수(+66bp)·귀리/밀·백금/금은 자기 가격 1년 위치만으로도 거의 같은 개선 → 대체재 정보가 아니라 가격 위치 효과로 판정해 기각. "
                   "WTI·휘발유는 무관한 가격비로도 +22~38bp가 나와 기각. 곡물 6·철광석·점결탄·구리는 기존 검증 정의 유지 — "
                   "엄격 기준으로는 대두 크러시만 통과(옥수수/밀은 13주 후반 −7bp)해 재검토 대상. "
                   "품목마다 2~5개 중 최고를 골랐으므로 채택 관계의 개선 폭은 실제보다 낙관적일 수 있음.")
    elif sec == "최신 시계열 모델 · 예측 목표":
        st.markdown("##### 최신 시계열 파운데이션 모델 vs 기준 모델 — 목표를 바꾸면 예측력이 생기나 (9개 구매 품목, 2012~ walk-forward, 매주 금요일 결정)")
        st.caption("파운데이션 모델 = 사전학습만 된 zero-shot (Chronos-Bolt-small·Chronos-2: Amazon, TimesFM 2.5: Google Research, TiRex: NX-AI xLSTM — "
                   "TiRex는 CPU에서 느려 4주마다 결정). 기준 = 과거 평균·모멘텀(12-1)·단기 반전·LightGBM 글로벌(9품목 함께 학습, 매년 재학습, Purged K-fold로 "
                   "하이퍼파라미터 선택). 주의: 파운데이션 모델의 사전학습 자료에 평가 기간의 원자재 시계열이 일부 포함됐을 수 있음(낙관 편향 가능).")
        df = _exp_header("forecast_lab")
        if df is None:
            return
        g = lambda k: df[df.구분 == k].dropna(axis=1, how="all")
        t1, t2, t3, t4, t5, t6, t7 = st.tabs(["방향 (5일)", "변동성 (20일)", "스프레드 (상대가치)", "품목 순위", "구매: 기다릴까", "과적합 판정", "메타 라벨링"])
        with t1:
            d = g("direction")
            agg = d.groupby("모델").agg(평균Sharpe=("Sharpe", "mean"), 중앙Sharpe=("Sharpe", "median"), 평균적중률=("적중률", "mean"),
                                       위약95_초과품목=("Sharpe", lambda x: int((x > d.loc[x.index, "위약95"]).sum())),
                                       위약최대95_초과품목=("Sharpe", lambda x: int((x > d.loc[x.index, "위약최대95"]).sum())), 품목수=("Sharpe", "size"))
            st.dataframe(agg.sort_values("평균Sharpe", ascending=False).style.format({"평균Sharpe": "{:+.2f}", "중앙Sharpe": "{:+.2f}", "평균적중률": "{:.1%}"}))
            st.caption("부호전략 = 예측 부호대로 다음 5영업일 롱/숏 (주간 비중첩, 비용 전). 위약95 = 같은 롱 비중·전환 빈도의 무작위 부호 1개의 95% 분위, "
                       "위약최대95 = 모델 수만큼 시도했을 때 최댓값의 95% 분위 → 이 기준을 넘어야 '운이 아님'.")
            with st.expander("품목 × 모델"):
                st.dataframe(d.pivot(index="품목", columns="모델", values="Sharpe").style.format("{:+.2f}"))
        with t2:
            d = g("vol")
            d["순위"] = d.groupby("품목")["QLIKE"].rank()
            st.dataframe(d.groupby("모델").agg(평균QLIKE순위=("순위", "mean"), DM_t_중앙값=("DM_t_vs_GARCH", "median"),
                                              GARCH보다_나은_날=("GARCH보다_나은_비율", "mean")).sort_values("평균QLIKE순위")
                         .style.format({"평균QLIKE순위": "{:.2f}", "DM_t_중앙값": "{:+.2f}", "GARCH보다_나은_날": "{:.0%}"}, na_rep="–"))
            st.caption("목표 = 향후 20영업일 실현분산. 파운데이션 모델 입력 = 주간 실현분산 5년(260주). DM t < −1.96이면 GARCH(1,1)보다 유의하게 우수.")
        with t3:
            d = g("spread")
            st.dataframe(d.groupby("모델").agg(MAE비율_중앙=("MAE_비율_vs_RW", "median"), 방향적중_평균=("방향적중", "mean"), IC_평균=("IC", "mean"))
                         .sort_values("MAE비율_중앙").style.format({"MAE비율_중앙": "{:.3f}", "방향적중_평균": "{:.1%}", "IC_평균": "{:+.3f}"}, na_rep="–"))
            st.caption("목표 = 상대가치 관계(크러시 마진·옥수수/밀 가격비·SGX/DCE 차익·고로 마진 등)의 20영업일 후 변화. MAE 비율 < 1이면 랜덤워크(변화 0)보다 정확, "
                       "IC = 예측 변화와 실제 변화의 순위상관.")
            with st.expander("관계별"):
                st.dataframe(d.pivot_table(index=["품목", "관계"], columns="모델", values="MAE_비율_vs_RW").style.format("{:.3f}"))
        with t4:
            d = g("rank").set_index("모델")
            st.dataframe(d.sort_values("평균IC", ascending=False).style.format({"평균IC": "{:+.3f}", "IC_t": "{:+.2f}", "IC양수비율": "{:.0%}",
                                                                                "상위3하위3_Sharpe": "{:+.2f}", "n": "{:.0f}"}))
            st.caption("매 4주(비중첩) 9품목의 향후 20영업일 수익률 순위를 예측 → 횡단면 Spearman IC. 상위3−하위3 = 예측 상위 3품목 롱·하위 3품목 숏의 Sharpe (연율).")
        with t5:
            d = g("buy")
            st.dataframe(d[d.품목 == "(전체)"].set_index("모델").drop(columns="품목").sort_values("결정절감bp", ascending=False)
                         .style.format({"AUC": "{:.3f}", "Brier": "{:.3f}", "기준비율": "{:.0%}", "기다림선택비율": "{:.0%}", "결정절감bp": "{:+.1f}",
                                        "항상즉시매수bp": "{:+.1f}", "n": "{:.0f}"}))
            st.caption("목표 = '향후 20영업일 평균가 < 오늘 가격'(기다리면 균등 매수로 더 싸게 살 수 있었나). 결정 = P(기다리면 쌈) > 0.5면 20일 균등 매수, 아니면 "
                       "다음 날 즉시 매수 — 절감 bp = 항상 균등 매수 대비 (+면 이득). AUC 0.5 = 동전 던지기.")
            with st.expander("품목별"):
                st.dataframe(d[d.품목 != "(전체)"].pivot(index="품목", columns="모델", values="AUC").style.format("{:.3f}"))
        with t6:
            d = g("validation").set_index("대상")
            st.dataframe(d.style.format({c: "{:.3f}" for c in d.columns if d[c].dtype.kind == "f"}, na_rep="–"))
            st.caption("DSR(Deflated Sharpe) = 시도 횟수·표본 길이·왜도·첨도를 감안해 '최고 전략의 Sharpe가 운으로 기대되는 최댓값 SR₀보다 클 확률' — 0.95 이상이어야 신뢰. "
                       "PBO = IS 최고 전략이 OOS에서 중앙값 아래로 떨어질 확률 (CSCV, 16블록) — 0.5 근처면 선택이 무의미. "
                       "CV 비교 행의 PSR 열 = AUC: 무작위 K-fold는 겹치는 목표(20일) 때문에 학습·검증이 섞여 AUC가 부풀려짐 → Purged K-fold + embargo가 실제 walk-forward에 가까움.")
        with t7:
            d = g("meta")
            st.dataframe(d.set_index(["품목", "방식"])[["평균절감bp", "전반_2010_17", "후반_2018_", "균등보다싸게", "하위5bp", "기간수", "메타_AUC"]]
                         .style.format({"평균절감bp": "{:+.1f}", "전반_2010_17": "{:+.1f}", "후반_2018_": "{:+.1f}", "균등보다싸게": "{:.0%}", "하위5bp": "{:+.0f}",
                                        "기간수": "{:.0f}", "메타_AUC": "{:.3f}"}, na_rep="–"))
            st.caption("메타 라벨링 (López de Prado 2018): 1차 신호 = ★ 구매 타이밍 '추천' 점수(상대가치 중심). 2차 LightGBM이 '이번 주 1차 신호 방향(더/덜 사라)이 "
                       "13주 균등 매수보다 나을 확률' p를 예측(매년 재학습, 13주 purge) → 점수' = 0.5 + (점수 − 0.5)·2p. 같은 13주 구매 백테스트(균등 매수 대비 절감 bp).")
    elif sec == "리스크 모형":
        st.markdown("##### 리스크 — 변동성 모형 경주 · VaR 백테스트 · 구매 타이밍 4주 범위 적중률 (12품목, 2010~ 평가)")
        df = _exp_header("risk")
        if df is None:
            return
        v = df[df.구분 == "변동성"].copy(); v["포함"] = v.MCS_p >= 0.10; v["순위"] = v.groupby("품목")["QLIKE"].rank()
        st.dataframe(v.groupby("항목").agg(**{"MCS 포함 품목 수": ("포함", "sum"), "평균 QLIKE 순위": ("순위", "mean"),
                                            "DM t 중앙값 (vs RiskMetrics)": ("DM_t", "median")}).sort_values("평균 QLIKE 순위")
                     .style.format({"평균 QLIKE 순위": "{:.2f}", "DM t 중앙값 (vs RiskMetrics)": "{:+.2f}"}))
        with st.expander("품목 × 모형 MCS p값"):
            st.dataframe(v.pivot(index="품목", columns="항목", values="MCS_p").style.format("{:.2f}").highlight_between(left=0.10, right=1.0, props="font-weight:700"))
        x = df[df.구분 == "VaR"].copy(); x["통과"] = (x.Kupiec_p >= 0.05) & (x.Christoffersen_p >= 0.05)
        c1_, c2_ = st.columns(2)
        c1_.markdown("**VaR 백테스트 통과 비율** (Kupiec·Christoffersen 모두 p ≥ 0.05)")
        c1_.dataframe(x.groupby(["기준", "포지션", "항목"])["통과"].mean().unstack(0).style.format("{:.0%}"))
        c2_.markdown("**평균 초과율** (기대 1% · 5%)")
        c2_.dataframe(x.groupby(["기준", "항목"])["초과율"].mean().unstack(0).style.format("{:.2%}"))
        bnd = df[df.구분 == "4주 범위"][["품목", "정규적중", "경험적적중", "하단이탈", "상단이탈", "n"]].set_index("품목")
        st.markdown("**구매 타이밍 '4주 90% 범위' 실제 적중률** (MIDAS σ, 2008~ 주 1회)")
        st.dataframe(bnd.style.format({"정규적중": "{:.1%}", "경험적적중": "{:.1%}", "하단이탈": "{:.1%}", "상단이탈": "{:.1%}", "n": "{:.0f}"}))
        st.caption("읽는 법: ① 1일 분산 예측은 GARCH(1,1)가 12품목 모두 MCS에 포함, 품목별 고정 λ·Kalman TVP 잔차가 그다음 — 매년 λ를 바꾸는 롤링·HMM은 "
                   "추정 잡음 때문에 예측이 오히려 나빠짐(λ가 기간마다 다르다는 LR 결과와 별개). ② 1일 99% VaR는 정규가 거의 전부 탈락(꼬리 과소), "
                   "FHS·EVT가 가장 정확(평균 초과율 1.0~1.1%). 20일 95%는 대부분 방법이 통과. ③ 4주 범위는 MIDAS σ 기준 정규도 90% 근처이나, "
                   "경험적 분위가 대두박·대두유·철광석의 적중률을 높임 → 구매 타이밍 탭에 경험적 분위 적용.")
    elif sec == "모델 선택 메타 전략":
        st.markdown("##### 모델 선택 메타 전략 — 시기·레짐별로 ML 모델을 고르면 나아지는가 (6품목 × 일간·주간, 평가 2008~, 비용 5bp)")
        df = _exp_header("metasel")
        if df is None:
            return
        a_, b_ = st.columns([1, 2])
        fq = a_.radio("빈도", ["일간", "주간"], horizontal=True, key="ex_ms_fq")
        perf = df[(df.구분 == "성과") & (df.빈도 == fq)]
        sc = "부호전략 Sharpe (비용 후)"
        pv = perf.pivot(index="전략", columns="품목", values=sc).reindex(perf.전략.unique())
        pv.columns = [short(c) for c in pv.columns]
        q95 = perf.pivot(index="전략", columns="품목", values="위약 95% 분위").reindex(perf.전략.unique())
        f = go.Figure(go.Heatmap(z=pv.values, x=list(pv.columns), y=list(pv.index), colorscale=DIV, zmid=0, texttemplate="%{z:+.2f}",
                                 hovertemplate="%{y} · %{x}: %{z:+.3f}<extra></extra>"))
        f.update_layout(height=120 + 28 * len(pv), margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(autorange="reversed"),
                        title=f"{fq} 부호전략 Sharpe (비용 후) — 위약 95% 분위를 넘은 칸 수: {int((pv.values > q95.values).sum())} / {pv.size}")
        st.plotly_chart(f, key="ex_ms_heat")
        sub = perf.groupby("전략", sort=False)[["Sharpe 비용 후 2008~16", "Sharpe 비용 후 2017~"]].mean()
        sub["두 기간 모두 +인 품목 수"] = perf.assign(ok=(perf["Sharpe 비용 후 2008~16"] > 0) & (perf["Sharpe 비용 후 2017~"] > 0)).groupby("전략", sort=False)["ok"].sum()
        sub["OOS R² 중앙값"] = perf.groupby("전략", sort=False)["OOS R² (vs 과거평균)"].median()
        sub["평균 롱 비중"] = perf.groupby("전략", sort=False)["롱 비중"].mean()
        st.markdown("**품목 평균 — 하위기간 일관성**")
        st.dataframe(sub.style.format({"Sharpe 비용 후 2008~16": "{:+.2f}", "Sharpe 비용 후 2017~": "{:+.2f}", "OOS R² 중앙값": "{:+.4f}",
                                       "평균 롱 비중": "{:.0%}", "두 기간 모두 +인 품목 수": "{:.0f}"}))
        st.markdown("**데이터 스누핑 보정** — 13개 전략(후보 6 + 메타 7) 중 최고를 골랐을 때도 유의한가")
        mx = df[(df.구분 == "최댓값 위약") & (df.빈도 == fq)][["품목", "전략", sc, "위약 중앙값", "위약 95% 분위", "위약 p (단일)"]]
        mx = mx.rename(columns={"전략": "최고 전략", sc: "관측 최고 Sharpe", "위약 중앙값": "위약 최댓값 중앙", "위약 95% 분위": "위약 최댓값 95%", "위약 p (단일)": "p"})
        mx["품목"] = mx["품목"].map(short)
        sp = df[(df.구분 == "SPA") & (df.빈도 == fq)].pivot(index="품목", columns="기준", values="Hansen SPA p")
        sp.index = [short(i) for i in sp.index]; sp.columns = [f"SPA p: {c}" for c in sp.columns]
        st.dataframe(mx.set_index("품목").join(sp).style.format({c: "{:+.2f}" for c in ["관측 최고 Sharpe", "위약 최댓값 중앙", "위약 최댓값 95%"]}
                                                                | {"p": "{:.2f}"} | {c: "{:.3f}" for c in sp.columns}))
        i_ = b_.selectbox("누적 성과 품목", list(perf.품목.unique()), format_func=short, key="ex_ms_id")
        cf = EX_DATA / "model_selection_cum.parquet"
        if cf.exists():
            C = pd.read_parquet(cf); C = C[(C.품목 == i_) & (C.빈도 == fq)].pivot(index="date", columns="전략", values="누적")
            f = go.Figure()
            for j, c in enumerate(C.columns):
                meta = c.startswith("(")
                f.add_trace(go.Scatter(x=C.index, y=C[c], name=c, line=dict(color="#898781" if c == "Buy & Hold" else PAL[j % 8], width=2.2 if meta else 1.0,
                                                                             dash="dot" if c == "Buy & Hold" else None),
                                       visible=True if meta or c == "Buy & Hold" else "legendonly"))
            st.plotly_chart(shade(fig_layout(f, 380, title=f"{short(i_)} · {fq} — 누적 순수익 (비용 5bp, 후보 모델은 범례 클릭)", yaxis_title="누적 로그수익률 (%)"),
                                  C.index), key="ex_ms_cum")
        with st.expander("품목별 전체 표"):
            t_ = perf[perf.품목 == i_].set_index("전략").drop(columns=["구분", "품목", "빈도", "기준"], errors="ignore").dropna(axis=1, how="all")
            st.dataframe(t_.style.format("{:.3f}", subset=list(t_.select_dtypes("number").columns)))
        st.caption(META_READ)
    elif sec == "전략 비교 (리스크 버짓)":
        st.markdown("##### 리스크 버짓(신호 기울기 τ) vs 리스크 패리티 vs 최적화 vs 1/N vs DBC")
        df = _exp_header("rb")
        if df is None:
            return
        for nm in df["설정"].unique():
            d = df[df.설정 == nm]
            f = go.Figure()
            rb = d[d.전략 == "리스크 버짓(신호 기울기)"]
            f.add_trace(go.Scatter(x=rb["τ"], y=rb["Sharpe (비용 후)"], mode="lines+markers", line=dict(color=PAL[0], width=3), name="리스크 버짓 (τ)"))
            for j, s_ in enumerate(["리스크 패리티(Newton)", "최적화(Newton)", "동일가중 1/N", "DBC 보유"]):
                v = d[d.전략 == s_]["Sharpe (비용 후)"]
                if len(v):
                    f.add_hline(y=float(v.mean()), line_dash="dot", line_color=[PAL[1], PAL[2], PAL[3], "#898781"][j], annotation_text=s_, annotation_position="right")
            st.plotly_chart(fig_layout(f, 300, title=f"{nm} — Sharpe(비용 후) vs 신호 기울기 τ (τ=0: 부호 반영 리스크 패리티)",
                                       xaxis_title="τ", yaxis_title="Sharpe", hovermode="x unified"), key=f"ex_rb_{nm}")
        with st.expander("전체 표"):
            st.dataframe(df.set_index(["설정", "τ", "전략"]).style.format("{:.3f}"))
        st.caption("위로 갈수록 좋음. 점선 = τ와 무관한 비교 전략. 예측력이 약한 신호(모멘텀 롱숏)는 τ가 작을수록, 크기 정보가 있는 신호(Ridge)는 τ가 클수록 유리 → 기본 τ=0.5 권장.")
    elif sec == "모멘텀 기간":
        st.markdown("##### 모멘텀 기간 비교 (주간 데이터 · 주간 리밸런싱)")
        df = _exp_header("momentum")
        if df is None:
            return
        cc = st.columns(2)
        for j, mode in enumerate(["long_short", "long_only"]):
            d = df[df.모드 == mode]
            f = go.Figure()
            for k_, s_ in enumerate([s for s in d.전략.unique()]):
                v = d[d.전략 == s_]
                f.add_trace(go.Bar(x=v["모멘텀"], y=v["Sharpe (비용 후)"], name=s_, marker_color=PAL[k_]))
            cc[j].plotly_chart(fig_layout(f, 360, barmode="group", title=f"{'롱숏' if mode == 'long_short' else '롱온리'} — Sharpe(비용 후)", hovermode="x"),
                               key=f"ex_mom_{mode}")
        st.dataframe(df.pivot_table(index="모멘텀", columns=["모드", "전략"], values=["Sharpe (비용 후)", "연간 회전율(편도 합)"]).round(3))
        st.caption("데이터 주기와 무관하게 달력 기간으로 계산. 업계 표준 12-1개월이 가장 좋고, 짧을수록 회전율만 늘고 성과는 하락 (1주~1개월은 단기 반전 구간).")
    elif sec == "거래 규칙":
        st.markdown("##### 거래 규칙 비교 (일간 데이터 · 모멘텀 롱숏)")
        df = _exp_header("rules")
        if df is None:
            return
        f = make_subplots(rows=1, cols=2, subplot_titles=("Sharpe (비용 후)", "연간 회전율 (편도 합)"))
        for k_, s_ in enumerate(df.전략.unique()):
            v = df[df.전략 == s_]
            f.add_trace(go.Bar(x=v["규칙"], y=v["Sharpe (비용 후)"], name=s_, marker_color=PAL[k_]), 1, 1)
            f.add_trace(go.Bar(x=v["규칙"], y=v["연간 회전율(편도 합)"], name=s_, marker_color=PAL[k_], showlegend=False), 1, 2)
        st.plotly_chart(fig_layout(f, 400, barmode="group", hovermode="x"), key="ex_rules")
        st.dataframe(df.set_index(["규칙", "전략"]).style.format("{:.3f}").format(pct(["연환산 수익률(CAGR)", "연환산 변동성", "최대낙폭(MDD)", "연간 비용 부담"])))
        st.caption("무거래 구간(필요할 때만 거래 + 최대 보유기간)이 비슷하거나 더 높은 Sharpe를 훨씬 적은 거래로 달성 — 비용이 작은 유동성 선물이라 Sharpe 개선 폭은 작지만 회전율·체결 부담 절감이 핵심.")
    else:
        st.markdown("##### 변동성 예측 → 포지션 크기 (일간 데이터 · 주간 리밸런싱 · 모멘텀)")
        df = _exp_header("vol")
        if df is None:
            return
        a_, b_ = st.columns(2)
        mode = a_.radio("제약", ["long_short", "long_only"], horizontal=True, key="ex_vol_mode", format_func=lambda x: "롱숏" if x == "long_short" else "롱온리")
        strat = b_.selectbox("전략", list(df.전략.unique()), key="ex_vol_strat")
        d = df[(df.모드 == mode) & (df.전략 == strat)]
        f = make_subplots(rows=1, cols=3, subplot_titles=("Sharpe (비용 후)", "실현 변동성", "변동성의 변동성 (낮을수록 안정)"))
        for j, tg in enumerate(["없음", "10%"]):
            v = d[d.목표 == tg]
            for c_i, col in enumerate(["Sharpe (비용 후)", "연환산 변동성", "변동성의변동성"]):
                f.add_trace(go.Bar(x=v["변동성모형"], y=v[col], name=f"타기팅 {tg}", marker_color=PAL[j], showlegend=c_i == 0), 1, c_i + 1)
        st.plotly_chart(fig_layout(f, 360, barmode="group", hovermode="x"), key="ex_vol_fig")
        st.dataframe(d.set_index(["변동성모형", "목표"]).drop(columns=["모드", "전략"]).style.format("{:.3f}").format(
            pct(["연환산 수익률(CAGR)", "연환산 변동성", "최대낙폭(MDD)", "연간 비용 부담"])))
        st.caption("변동성 타기팅 → 실현 변동성이 목표(10%)에 붙고 변동성의 출렁임·낙폭 감소 (위험 관리 효과 확실). Sharpe는 거의 그대로이며 롱온리에선 오히려 하락 "
                   "(폭락 후 반등을 놓침). MIDAS·HAR 예측이 목표를 가장 정확히 맞춤.")


def page_structure():
    a, b = st.tabs(["상관 · PCA · 스필오버 · TVP-VAR", "FAVAR (요인 충격의 전파)"])
    with a:
        tab_comove()
    with b:
        tab_favar()


def page_vol():
    a, b, c, dd = st.tabs(["단변량 GARCH · 레버리지", "다변량 GARCH (DCC · ADCC · BEKK)", "GARCH-MIDAS (장기·단기 변동성 분해)",
                            "변동성 예측 검증 (QLIKE · DM · MCS · 부호 편향)"])
    with a:
        tab_vol()
    with b:
        tab_mvgarch()
    with c:
        tab_midas(vol_only=True)
    with dd:
        vol_forecast_check()


def page_predict():
    a, b, c = st.tabs(["단일 품목 (회귀 · ML · 계량)", "여러 품목 (패널 · 횡단면)", "MIDAS 수익 예측 (혼합 주기)"])
    with a:
        tab_ml()
        ml_meta()
    with b:
        tab_panel()
    with c:
        tab_midas(vol_only=False)


def page_atlas():
    try:
        import model_atlas
        model_atlas.render()
    except ImportError:
        st.info("모델 도감 준비 중")


for _t, _fn in zip(tabs, [tab_procure, tab_risk, page_structure, tab_regime, page_vol, page_predict, tab_portfolio, tab_experiments, page_atlas, tab_data]):
    with _t:
        _fn()
