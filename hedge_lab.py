"""헤지 랩 (초안) — 선물 헤지 A1~A4: 헤지 비율→VaR 곡선 · 최적 헤지 비율·효과 · 헤지 수단 추천 · 크러시 묶음 헤지.
기존 대시보드(app.py)와 별도 실행. 계산은 hedge_core.py, VaR는 기존 risk.py(◆ 리스크 탭과 같은 함수) 재사용.

실행:  .venv\\Scripts\\streamlit run hedge_lab.py   (또는 헤지랩_열기.bat, 포트 8507)
"""
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analytics as A
import hedge_core as H
import procurement as PR
import risk as RK

st.set_page_config(page_title="헤지 랩 (초안)", layout="wide", page_icon="🛡️")
G, GOLD, RED, BLUE, GREY = "#1E3A2F", "#C8A24A", "#D9582B", "#2F6E8F", "#9A9A94"
PAL = [G, BLUE, GOLD, RED, "#6B8E6B", "#8C6BB1", GREY]


@st.cache_data(show_spinner="가격 자료 읽는 중…")
def levels():
    return A.load_levels(), A.load_levels(roll_adjusted=False)


@st.cache_data(show_spinner="변동성 모형(walk-forward) 추정 중…")
def vol(i: str, model: str):
    La, _ = levels()
    o = RK.vol_models(RK.returns(La[i]), [model])
    return o["r"], o["S"][model]


@st.cache_data
def weekly(ids: tuple):
    La, _ = levels()
    return H.weekly_usd(La, list(ids))


def fig(f, h=360, **k):
    f.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2), **k)
    return f


La, Lr = levels()
st.title("🛡️ 헤지 랩 (초안) — 선물로 가격 위험 줄이기")
st.caption("기존 대시보드와 따로 실행되는 초안. '언제 살까'(방향 예측)가 약한 품목에서 '얼마나 헤지할까'로 위험을 관리하는 도구. "
           f"자료 마지막 날짜 {La.index.max():%Y-%m-%d}. 투자·거래 권유 아님 — 실제 헤지는 사내 규정·한도에 따름.")
with st.sidebar:
    st.markdown("### 공통 설정")
    hw = st.radio("헤지 기간 (주)", [1, 4], 1, horizontal=True, help="헤지 효과를 몇 주 단위 손익으로 잴지. 실제 매입까지 남은 기간에 가깝게 — 1주는 시장 간 시차(중국·미국 종가) 잡음이 큼")
    win = st.select_slider("헤지 비율 추정 창 (주)", [52, 104, 156, 260], 104, help="최근 몇 주 자료로 헤지 비율을 정할지. 짧으면 빨리 따라가지만 들쭉날쭉")
    st.caption("모든 헤지 효과 = 표본 외(walk-forward): 그 주의 헤지 비율은 직전 주까지 자료로만 정함")

t1, t2, t3, t4, t5 = st.tabs(["A1 헤지 비율 → VaR", "A2 최적 헤지 비율 · 효과", "A3 헤지 수단 추천", "A4 크러시 묶음 헤지", "읽는 법 · 한계"])

# ════════════════ A1
with t1:
    st.markdown("#### 헤지를 얼마나 하면 위험이 얼마나 줄어드나 — ◆ 리스크 '구매·보유 물량 위험'과 같은 VaR 계산에 헤지 비율만 더함")
    c = st.columns([2.2, 1.3, 0.8, 0.8, 1.1])
    ids = c[0].multiselect("품목", [i for i in PR.BUY_IDS if i in La and i in H.CONTRACT_T], ["yf:ZS=F", "yf:ZM=F", "yf:ZL=F", "yf:ZW=F"],
                           format_func=H.name, key="a1_ids")
    dirl = c[1].radio("물량 성격", ["매입 예정 (가격 상승이 손실)", "보유 재고 (가격 하락이 손실)"], key="a1_dir")
    alpha = c[2].selectbox("신뢰수준", [0.95, 0.99], 1, format_func=lambda x: f"{x:.0%}", key="a1_a")
    hd = c[3].selectbox("기간 (영업일)", [5, 10, 20, 60], 2, key="a1_h")
    vm = c[4].selectbox("σ 모형", ["GARCH(1,1)", "RiskMetrics λ=0.94"], 0, key="a1_vm")
    if ids:
        qc = st.columns(len(ids))
        qty = {i: qc[j].number_input(f"{H.name(i)} (톤)", 0.0, 1e8, 30000.0, 1000.0, key=f"a1_q_{i}") for j, i in enumerate(ids)}
        sg = -1.0 if dirl.startswith("매입") else 1.0
        px = {i: float(PR.usd_t(Lr, i).dropna().iloc[-1]) for i in ids}
        expo = pd.Series({i: sg * qty[i] * px[i] for i in ids})
        VV = {i: vol(i, vm) for i in ids}
        R = pd.DataFrame({i: VV[i][0] for i in ids}).dropna(); S = pd.DataFrame({i: VV[i][1] for i in ids}).reindex(R.index).dropna()
        R = R.loc[S.index]
        # USDA 보고서가 기간 안에 있으면 그날 변동 배수만큼 σ를 키움 (2026-11-02 실험: 분기 보고서 날 변동 1.3~2.9배, 99% VaR 초과율 최대 33%)
        cal = H.report_calendar(La.index.max(), hd)
        with st.expander(f"USDA 보고서 포함 기간 — 앞으로 {hd}영업일 안 보고서 {len(cal)}건", expanded=len(cal) > 0):
            e1, e2 = st.columns([1.1, 1.6])
            with e1:
                use_ev = st.checkbox("이 기간에 USDA 보고서 포함 (보고서 날 변동 배수 반영)", value=len(cal) > 0, key="a1_ev")
                kw = st.number_input("WASDE 일수", 0, hd, int((cal["보고서"] == "WASDE").sum()), key="a1_kw", disabled=not use_ev)
                kq = st.number_input("분기 보고서 일수 (3·6·9월 말: 파종의향·작付면적·곡물재고)", 0, hd, int((cal["보고서"] == "분기 보고서").sum()),
                                     key="a1_kq", disabled=not use_ev)
                if len(cal):
                    st.dataframe(cal.assign(날짜=cal["날짜"].dt.strftime("%Y-%m-%d")), hide_index=True, width="stretch")
                st.caption("미래 WASDE 날짜는 작년 같은 달 발표일로 추정, 분기 보고서는 월말 영업일로 근사 — 실제 일정은 USDA 공지로 확인.")
            mult = {i: H.event_multipliers(R[i], S[i]) for i in ids}
            with e2:
                st.dataframe(pd.DataFrame(mult).T.rename(index=H.name).style.format("{:.2f}배"), width="stretch")
                st.caption(f"보고서 날 변동 배수 = √(보고서 날 z² 평균 / 평소 z² 평균), z = 일간 수익 / {vm} 예측 σ, 2008년 이후. "
                           f"{hd}일 분산 = σ² × [(평소 일수) + Σ 보고서 일수 × 배수²].")
        S0 = S
        if use_ev and kw + kq > 0:
            S = H.event_scale(S, hd, {"WASDE": kw, "분기 보고서": kq}, mult)
            b0, b1 = RK.position_risk(R, S0, expo, alpha, hd), RK.position_risk(R, S, expo, alpha, hd)
            st.info(f"보고서 반영: 헤지 전 {hd}일 {alpha:.0%} VaR (FHS) ${b0['var_fhs']:,.0f} → **${b1['var_fhs']:,.0f}** "
                    f"({b1['var_fhs'] / b0['var_fhs'] - 1:+.0%}). 아래 곡선·표는 모두 보고서 반영 값.")
        if expo.abs().sum() > 0 and len(S) > 600:
            ratios = np.linspace(0, 1, 21)
            CU, CP = H.hedge_curve(R, S, expo, alpha, hd, ratios, "uniform"), H.hedge_curve(R, S, expo, alpha, hd, ratios, "priority")
            a, b = st.columns([1.35, 1])
            with a:
                f = go.Figure()
                f.add_trace(go.Scatter(x=CU["헤지 비율"] * 100, y=CU["VaR (FHS)"], name="같은 비율로 모든 품목 (FHS VaR)", line=dict(color=G, width=3)))
                f.add_trace(go.Scatter(x=CU["헤지 비율"] * 100, y=CU["ES (FHS)"], name="같은 비율 (FHS ES)", line=dict(color=G, dash="dot")))
                f.add_trace(go.Scatter(x=CP["헤지 비율"] * 100, y=CP["VaR (FHS)"], name="위험 기여 큰 품목부터 (FHS VaR)", line=dict(color=GOLD, width=3)))
                f.update_layout(xaxis_title="총 노출 대비 헤지 비율 (%)", yaxis_title=f"{hd}일 {alpha:.0%} 손실 ($)", hovermode="x unified")
                st.plotly_chart(fig(f, 380, title="헤지 비율 → VaR·ES (같은 품목 선물, 베이시스 0 가정)"), width="stretch")
            with b:
                base = RK.position_risk(R, S, expo, alpha, hd)
                k = st.columns(2)
                k[0].metric("총 노출", f"${expo.abs().sum():,.0f}"); k[1].metric(f"헤지 전 {hd}일 {alpha:.0%} VaR (FHS)", f"${base['var_fhs']:,.0f}")
                tb = base["table"].copy()
                tb["1달러당 VaR 기여"] = tb["VaR 기여 ($, 정규)"] / tb["노출 ($)"].abs()
                tb["헤지 우선순위"] = tb["1달러당 VaR 기여"].rank(ascending=False).astype(int)
                st.dataframe(tb.rename(index=H.name)[["노출 ($)", "일 σ (%)", "VaR 기여 ($, 정규)", "기여 비중", "1달러당 VaR 기여", "헤지 우선순위"]].style.format(
                    {"노출 ($)": "{:,.0f}", "일 σ (%)": "{:.2f}", "VaR 기여 ($, 정규)": "{:,.0f}", "기여 비중": "{:.0%}", "1달러당 VaR 기여": "{:.3f}"}), width="stretch")
                st.caption("1달러당 VaR 기여가 클수록 같은 헤지 금액으로 위험을 더 많이 줄임 → 금색 곡선은 이 순서로 헤지 예산을 채움. "
                           "예산이 크면 같은 비율(초록)이 더 나을 수 있음(품목 간 분산 효과 유지).")
            st.markdown("##### 품목별로 직접 정하기")
            hc = st.columns(len(ids))
            hr = pd.Series({i: hc[j].slider(f"{H.name(i)} 헤지 %", 0, 100, 50, 5, key=f"a1_hr_{i}") / 100 for j, i in enumerate(ids)})
            after = RK.position_risk(R, S, expo * (1 - hr), alpha, hd)
            k = st.columns(4)
            k[0].metric("헤지 후 VaR (FHS)", f"${after['var_fhs']:,.0f}", f"{after['var_fhs'] / base['var_fhs'] - 1:+.0%}", delta_color="inverse")
            k[1].metric("헤지 후 ES (FHS)", f"${after['es_fhs']:,.0f}", f"{after['es_fhs'] / base['es_fhs'] - 1:+.0%}", delta_color="inverse")
            k[2].metric("헤지 금액", f"${(expo.abs() * hr).sum():,.0f}")
            k[3].metric("필요 선물 계약 (합)", f"{sum(H.contracts(abs(expo[i]), hr[i], px[i], i) for i in ids):,.0f}계약")
            st.dataframe(pd.DataFrame({"헤지 %": hr * 100, "헤지 물량 (톤)": [qty[i] * hr[i] for i in ids],
                                       "1계약 (톤)": [H.CONTRACT_T[i] for i in ids], "계약 수": [H.contracts(abs(expo[i]), hr[i], px[i], i) for i in ids],
                                       "현재가 ($/t)": [px[i] for i in ids]}, index=[H.name(i) for i in ids]).style.format(
                {"헤지 %": "{:.0f}", "헤지 물량 (톤)": "{:,.0f}", "1계약 (톤)": "{:.1f}", "계약 수": "{:,.0f}", "현재가 ($/t)": "{:,.1f}"}), width="stretch")
            st.caption(f"σ = {vm} walk-forward, 상관·FHS = ◆ 리스크 탭과 같은 risk.position_risk. 매입 예정 물량의 헤지 = 선물 매수(가격이 오르면 선물 이익이 손실 상쇄). "
                       "실제 매입 가격이 선물과 다르게 움직이는 위험(베이시스)은 여기서 0으로 가정 → A2·A3에서 측정.")

# ════════════════ A2
with t2:
    st.markdown("#### 실제로 사는 가격과 선물이 얼마나 같이 움직이나 → 위험이 가장 작아지는 헤지 비율")
    c = st.columns([1.4, 1.4, 1])
    tgt = c[0].selectbox("헤지할 대상 가격", ["(CSV 올리기)"] + [i for i in H.TARGETS if i in La], 1,
                         format_func=lambda i: i if i.startswith("(") else (H.name(i) + (" — 중국 현지 가격, 실물 대용" if i.startswith("ak:") else "")), key="a2_t")
    fut = c[1].selectbox("헤지 수단 (선물)", [i for i in H.HEDGE_CANDS if i in La], 1, format_func=H.name, key="a2_f")
    up_s = None
    if tgt == "(CSV 올리기)":
        up = c[2].file_uploader("date, price ($/t) 두 열", type="csv", key="a2_up")
        st.caption("실제 매입 단가(도착가 등) 기록을 올리면 그 가격 기준으로 헤지 효과를 계산. 파일은 이 PC 안에서만 쓰이고 저장·전송되지 않음.")
        if up is not None:
            d = pd.read_csv(io.BytesIO(up.getvalue())); d.columns = [x.strip().lower() for x in d.columns]
            up_s = d.assign(date=pd.to_datetime(d["date"])).set_index("date")["price"].astype(float).resample("W-FRI").last().dropna()
    P = weekly(tuple(sorted(set(H.TARGETS + H.HEDGE_CANDS))))
    s_lvl = up_s if up_s is not None else (P[tgt] if tgt in P else None)
    if s_lvl is not None and fut in P:
        rs, rf = H.log_ret(s_lvl, hw), H.log_ret(P[fut], hw)
        W = H.wf_minvar(rs, rf.rename(fut), win)
        K = H.kalman_beta(rs, rf, window=win)
        D = pd.concat([rs.rename("s"), rf.rename("f")], axis=1).dropna()
        one = H.he_by_period(D.s, D.s - D.f); mv = H.he_by_period(W["s"], W["e"]); kb = H.he_by_period(K["s"], K["e"])
        full_h = float((D.s * D.f).sum() / (D.f ** 2).sum())
        k = st.columns(5)
        k[0].metric("상관", f"{D.corr().iloc[0, 1]:.2f}")
        k[1].metric("최소분산 비율 (전체 기간)", f"{full_h:.2f}")
        k[2].metric(f"최근 비율 (직전 {win}주)", f"{W[fut].dropna().iloc[-1]:.2f}")
        k[3].metric("칼만 시변 비율 (최근)", f"{K['beta'].dropna().iloc[-1]:.2f}")
        k[4].metric("헤지 효과 (표본 외, 최소분산)", f"{mv['전체']:.0%}", help="1 − 헤지 후 분산/헤지 전 분산. 표본 외 = 그 주 비율은 직전까지 자료로만")
        a, b = st.columns([1.3, 1])
        with a:
            f = go.Figure()
            f.add_trace(go.Scatter(x=W.index, y=W[fut], name=f"최소분산 비율 (직전 {win}주)", line=dict(color=G)))
            f.add_trace(go.Scatter(x=K.index, y=K["beta"], name="칼만 시변 비율", line=dict(color=GOLD)))
            f.add_hline(y=1, line=dict(color=GREY, dash="dot"), annotation_text="가치 1:1")
            st.plotly_chart(fig(f, 340, title="헤지 비율의 시간 변화 — 1보다 작으면 노출 금액보다 적게 헤지하는 게 최적", yaxis_title="헤지 비율"), width="stretch")
        with b:
            T = pd.DataFrame({"가치 1:1": one, "최소분산 (롤링)": mv, "칼만 시변": kb}).T
            st.dataframe(T.style.format("{:.0%}").background_gradient(cmap="RdYlGn", vmin=-0.3, vmax=0.8, axis=None), width="stretch")
            st.caption("헤지 효과 — 기간별. 음수 = 헤지가 오히려 위험을 키움(과대 헤지). 기간마다 크게 다르면 베이시스가 불안정하다는 뜻.")
            f = go.Figure(go.Scatter(x=D.f * 100, y=D.s * 100, mode="markers", marker=dict(size=4, color=BLUE, opacity=0.5), name="주"))
            xx = np.linspace(D.f.min(), D.f.max(), 20) * 100
            f.add_trace(go.Scatter(x=xx, y=xx * full_h, line=dict(color=RED), name=f"기울기 {full_h:.2f}"))
            st.plotly_chart(fig(f, 260, title=f"{hw}주 수익 산점도", xaxis_title=f"{H.name(fut)} (%)", yaxis_title="대상 (%)"), width="stretch")
        st.markdown("##### 헤지 계약 수 계산")
        q = st.columns(3)
        tons = q[0].number_input("헤지할 물량 (톤)", 0.0, 1e8, 30000.0, 1000.0, key="a2_q")
        ratio = q[1].number_input("적용 헤지 비율", 0.0, 2.0, round(float(W[fut].dropna().iloc[-1]), 2), 0.05, key="a2_r")
        tpx = float(s_lvl.dropna().iloc[-1]); fpx = float(PR.usd_t(Lr, fut).dropna().iloc[-1]) if fut in PR.TO_USD_T else float(P[fut].dropna().iloc[-1])
        n = H.contracts(tons * tpx, ratio, fpx, fut)
        q[2].metric("선물 계약 수", f"{n:,.0f}계약" if np.isfinite(n) else "계약 크기 미등록",
                    help=f"= 비율 × 물량 × 대상 가격(${tpx:,.0f}/t) / (선물 ${fpx:,.0f}/t × 1계약 {H.CONTRACT_T.get(fut, np.nan):.1f}톤)")

# ════════════════ A3
with t3:
    st.markdown("#### 이 가격을 헤지하려면 어떤 선물이 가장 좋은가 — 후보마다 표본 외 헤지 효과 비교")
    c = st.columns([1.4, 2.6])
    tgt3 = c[0].selectbox("헤지할 대상 가격", [i for i in H.TARGETS if i in La], 0, format_func=H.name, key="a3_t")
    cands = c[1].multiselect("후보 선물", [i for i in H.HEDGE_CANDS if i in La], [i for i in H.HEDGE_CANDS if i in La and i != tgt3], format_func=H.name, key="a3_c")
    P = weekly(tuple(sorted(set(H.TARGETS + H.HEDGE_CANDS))))
    if cands:
        Rw = H.log_ret(P, hw)
        with st.spinner("후보별 walk-forward 헤지 효과 계산 중…"):
            T = H.compare_hedges(Rw[tgt3], Rw, cands, win)
        if len(T):
            a, b = st.columns([1, 1.4])
            with a:
                f = go.Figure()
                for j, k in enumerate(H.PERIODS):
                    f.add_trace(go.Bar(y=T["수단"], x=T[f"효과 {k}"] * 100, name=k, orientation="h", marker_color=PAL[j]))
                f.update_layout(barmode="group", yaxis=dict(autorange="reversed"), xaxis_title="헤지 효과 (%)")
                st.plotly_chart(fig(f, 120 + 34 * len(T), title="기간별 헤지 효과 (표본 외 최소분산)"), width="stretch")
            with b:
                T2 = T.drop(columns="id").set_index("수단")
                st.dataframe(T2.style.format({"상관": "{:.2f}", "최근 헤지 비율": "{:.2f}", **{c_: "{:.0%}" for c_ in T2.columns if c_.startswith(("효과", "안정성"))}})
                             .background_gradient(cmap="RdYlGn", subset=["효과 (최소분산)", "안정성 (기간 최저)"], vmin=-0.1, vmax=0.6), width="stretch")
                top = T.iloc[0]
                st.success(f"추천: **{top['수단']}** — 표본 외 헤지 효과 {top['효과 (최소분산)']:.0%}, 기간 최저 {top['안정성 (기간 최저)']:.0%}, 최근 비율 {top['최근 헤지 비율']:.2f}")
            st.markdown("##### 두 선물을 함께 쓰면 (상위 4개 중 2개 조합)")
            BP = H.best_pairs(Rw[tgt3], Rw, list(T.id[:4]), win)
            if len(BP):
                st.dataframe(BP.set_index("조합").style.format({c_: ("{:.0%}" if c_.startswith(("효과", "안정성")) else "{:.2f}") for c_ in BP.columns if c_ != "조합"}, na_rep="–"),
                             width="stretch")
                gain = BP["효과 (최소분산)"].iloc[0] - T["효과 (최소분산)"].iloc[0]
                st.caption(f"최고 조합의 추가 효과 {gain:+.1%}p — 1~2%p 수준이면 관리가 단순한 단일 수단이 실무상 나음.")
            st.caption("효과 (가치 1:1) = 노출 금액만큼 그대로 헤지했을 때. 최소분산보다 크게 낮으면 '1:1로 헤지하면 과대 헤지'라는 뜻.")

# ════════════════ A4
with t4:
    st.markdown("#### 대두박·대두유를 함께 산다면 — 각각 헤지할까, 대두 선물 하나로 묶어 헤지할까")
    c = st.columns(3)
    qm = c[0].number_input("대두박 매입 (톤)", 0.0, 1e8, 30000.0, 1000.0, key="a4_m")
    qo = c[1].number_input("대두유 매입 (톤)", 0.0, 1e8, 10000.0, 1000.0, key="a4_o")
    basis = c[2].radio("노출 가격 기준", ["중국 현지 (DCE 대두박·대두유, 실물 대용)", "CBOT (선물과 같은 가격)"], key="a4_b")
    P = weekly(tuple(sorted(set(H.TARGETS + H.HEDGE_CANDS))))
    mi, oi = ("ak:M0", "ak:Y0") if basis.startswith("중국") else ("yf:ZM=F", "yf:ZL=F")
    if qm + qo > 0 and mi in P and oi in P:
        pm, po = P[mi], P[oi]
        V = (qm * pm + qo * po).dropna()                                     # 묶음 가치 ($)
        rv = H.log_ret(V, hw)
        Rw = H.log_ret(P, hw)
        rows = {}
        Wd = H.wf_minvar(rv, Rw[["yf:ZM=F", "yf:ZL=F"]], win); rows["대두박 + 대두유 선물 (각각, 최소분산)"] = H.he_by_period(Wd["s"], Wd["e"])
        wm = (qm * pm / V).reindex(Rw.index).ffill()
        one = (wm * Rw["yf:ZM=F"] + (1 - wm) * Rw["yf:ZL=F"])
        D = pd.concat([rv.rename("s"), one.rename("f")], axis=1).dropna(); rows["대두박 + 대두유 선물 (각각, 가치 1:1)"] = H.he_by_period(D.s, D.s - D.f)
        Ws = H.wf_minvar(rv, Rw[["yf:ZS=F"]], win); rows["대두 선물 하나 (최소분산)"] = H.he_by_period(Ws["s"], Ws["e"])
        D2 = pd.concat([rv.rename("s"), Rw["yf:ZS=F"].rename("f")], axis=1).dropna(); rows["대두 선물 하나 (가치 1:1)"] = H.he_by_period(D2.s, D2.s - D2.f)
        if basis.startswith("중국") and "ak:A0" in Rw:
            Wa = H.wf_minvar(rv, Rw[["ak:A0"]], win); rows["참고: 대련 대두 선물 (최소분산)"] = H.he_by_period(Wa["s"], Wa["e"])
        T = pd.DataFrame(rows).T
        a, b = st.columns([1.1, 1])
        with a:
            st.dataframe(T.style.format("{:.0%}").background_gradient(cmap="RdYlGn", vmin=-0.3, vmax=0.9, axis=None), width="stretch")
            k = st.columns(3)
            k[0].metric("묶음 노출", f"${V.dropna().iloc[-1]:,.0f}")
            k[1].metric("대두 선물 최근 비율", f"{Ws['yf:ZS=F'].dropna().iloc[-1]:.2f}")
            k[2].metric("대두 선물 계약 수", f"{H.contracts(V.dropna().iloc[-1], Ws['yf:ZS=F'].dropna().iloc[-1], float(PR.usd_t(Lr, 'yf:ZS=F').dropna().iloc[-1]), 'yf:ZS=F'):,.0f}계약")
            st.caption("표본 외 헤지 효과(전체·기간별). 대두 하나로 묶으면 계약·관리가 단순하지만, 크러시 마진 변동(아래)만큼 위험이 남음.")
        with b:
            cr = H.board_crush(Lr).resample("W-FRI").last()
            f = go.Figure(go.Scatter(x=cr.index, y=cr, line=dict(color=G), name="보드 크러시 마진"))
            f.add_hline(y=float(cr.median()), line=dict(color=GREY, dash="dot"), annotation_text="중앙값")
            st.plotly_chart(fig(f, 320, title="보드 크러시 마진 ($/부셸) — 대두로 묶어 헤지할 때 남는 위험", yaxis_title="$/부셸"), width="stretch")
            st.caption("= 대두박(44lb)+대두유(11lb) 가치 − 대두 1부셸. 마진이 크게 움직이는 시기(2021~22 등)엔 대두 하나로 묶은 헤지 효과가 떨어짐.")

# ════════════════ 읽는 법
with t5:
    st.markdown("""
### 읽는 법
- **헤지 효과(HE)** = 1 − (헤지 후 손익 분산 / 헤지 전 손익 분산). 60%면 가격 위험의 60%를 없앤 것. 음수면 헤지가 오히려 위험을 키움.
- **표본 외(walk-forward)**: 매주 그 시점까지의 자료로만 헤지 비율을 정해 다음 주 손익에 적용 — 실제로 운영했을 때의 효과.
- **최소분산 헤지 비율** h = Cov(대상, 선물) / Var(선물). 대상이 선물보다 덜 움직이거나 상관이 낮으면 1보다 작음 → 노출 금액보다 적게 헤지하는 게 최적.
- **칼만 시변 비율**: 비율이 매주 조금씩 바뀐다고 보고 새 자료마다 고침 — 관계가 변할 때 롤링보다 빨리 따라감.
- **헤지 기간**: 1주 수익은 중국·미국 시장 마감 시각 차이 때문에 같은 뉴스가 다른 주에 반영돼 상관이 낮게 나옴 → 실제 매입까지 남은 기간(4주 등)으로 보는 게 맞음.

### 한계 (초안)
- A1은 **같은 품목 선물로 헤지(베이시스 0)** 를 가정 — 실제 매입가와 선물의 차이 위험은 A2·A3에서 따로 측정.
- 실물 가격 대용으로 **중국 대련(DCE) 가격**을 씀(위안·증치세 포함 → 달러 환산). 실제 매입 단가 기록이 있으면 A2에서 CSV로 올려 정확히 계산.
- 증거금·롤오버 비용·거래 수수료·환율(원화) 위험 미반영. 헤지는 평균 구매가를 낮추지 않고 **흔들림과 최악의 경우**를 줄임.
- 선물 계약 크기: 대두·밀 5,000부셸(약 136톤), 옥수수 5,000부셸(127톤), 대두박 100숏톤(90.7톤), 대두유 60,000파운드(27.2톤).
""")
