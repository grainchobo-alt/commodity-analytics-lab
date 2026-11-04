"""원자재 시계열 분석 랩 (Streamlit)

실행:  .venv\\Scripts\\streamlit run app.py
"""
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit.delta_generator import DeltaGenerator as _DG

import analytics as A


def _esc_tilde(x):
    """Streamlit 마크다운은 '~'로 감싼 부분을 취소선으로 그림 → 기간 표기(2007~2016) 등 홑 '~'를 이스케이프 ('~~'·이미 이스케이프된 것은 그대로)."""
    import re
    if isinstance(x, pd.DataFrame):                       # st.table 칸도 마크다운으로 그려짐
        return x.map(_esc_tilde).rename(index=_esc_tilde, columns=_esc_tilde)
    return re.sub(r"(?<![\\~])~(?!~)", r"\\~", x) if isinstance(x, str) else x


def _no_strike(name):
    f = getattr(_DG, name)

    def w(self, body, *a, **k):
        return f(self, _esc_tilde(body), *a, **k)
    setattr(_DG, name, w)
    setattr(st, name, getattr(st._main, name))


if not getattr(_DG, "_tilde_patched", False):
    for _n in ("markdown", "caption", "info", "warning", "success", "error", "expander", "subheader", "table"):
        _no_strike(_n)
    _DG._tilde_patched = True

st.set_page_config(page_title="Commodity Analytics Lab", layout="wide", page_icon="📈")
st.markdown("<style>[data-testid='stMainBlockContainer']{padding-top:4.2rem;}"          # 위쪽 빈 공간 줄이기 (기본 6rem, 헤더 높이 약 3.75rem)
            "[data-testid='stElementContainer']:has(> [data-testid='stMarkdown'] style):not(:has(p, h1, h2, h3, div[data-testid='stCaptionContainer'])){display:none;}</style>",
            unsafe_allow_html=True)                              # CSS만 든 빈 칸은 숨겨 간격(16px)이 생기지 않게

# 탭 상태 유지 + 고른 탭만 계산 (v1.65.0) — 예전 st.tabs는 화면 구성이 바뀌는 재실행 때 첫 탭으로 튀었음.
# 고른 탭만 그리면 다른 탭 위젯 값은 Streamlit이 지우므로, 입력 위젯 값을 _keep에 보관했다가 그 탭으로 돌아올 때 되살림.
# (버튼 · 내려받기 · 업로드 · 표 편집은 값을 넣을 수 없는 위젯이라 기록하지 않음)
_WKEEP = ("selectbox", "multiselect", "radio", "slider", "select_slider", "checkbox", "toggle", "number_input", "text_input",
          "text_area", "date_input", "time_input", "color_picker", "pills", "segmented_control")


def _rec_key(name):
    f = getattr(_DG, name)

    def w(self, *a, **k):
        if isinstance(k.get("key"), str):
            st.session_state.setdefault("_wkeys", set()).add(k["key"])
        return f(self, *a, **k)
    setattr(_DG, name, w)
    setattr(st, name, getattr(st._main, name))


if not getattr(_DG, "_keep_patched", False):
    for _n in _WKEEP:
        if hasattr(_DG, _n):
            _rec_key(_n)
    _DG._keep_patched = True
_KEEP = st.session_state.setdefault("_keep", {})
for _k, _v in list(_KEEP.items()):
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _keep_save():
    ss = st.session_state
    for k in list(st.session_state.get("_wkeys", ())):
        if k in ss:
            _KEEP[k] = ss[k]


def _keep_forget(*keys):
    """위젯 값을 지워 기본값으로 되돌릴 때는 보관본도 함께 지워야 함."""
    for k in keys:
        st.session_state.pop(k, None); _KEEP.pop(k, None)

try:
    DARK = st.context.theme.type == "dark"
except Exception:
    DARK = False
PAL = (["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"] if DARK else
       ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"])
MID = "#383835" if DARK else "#f0efec"
# 히트맵 색: 무지개(보라→파랑→하늘→청록→초록→연두→노랑→주황→빨강→진홍) — 0.1 단위 차이도 색상 자체가 달라 눈으로 구분 (마우스 없이)
# 발산형(DIV): 음수 = 보라·파랑 계열, 0 근처 = 초록, 양수 = 노랑·주황·빨강. 순차형(SEQ): 같은 무지개를 0→최대로.
DIV = [[0.0, "#3b0f70"], [0.1, "#5a2ca0"], [0.2, "#2c5fd6"], [0.3, "#1fa3e0"], [0.4, "#26c6a6"], [0.5, "#5fcf4a"],
       [0.6, "#c9e02f"], [0.7, "#ffd52e"], [0.8, "#ff9a1f"], [0.9, "#e8381f"], [1.0, "#8e0a24"]]
RDBU = [[0.0, "#2166ac"], [0.2, "#4393c3"], [0.4, "#92c5de"], [0.5, "#f7f7f7"], [0.6, "#f4a582"], [0.8, "#d6604d"], [1.0, "#b2182b"]]
SEQ = [[0.0, "#3b0f70"], [0.12, "#2c5fd6"], [0.25, "#1fa3e0"], [0.37, "#26c6a6"], [0.5, "#5fcf4a"], [0.62, "#c9e02f"],
       [0.75, "#ffd52e"], [0.87, "#ff9a1f"], [1.0, "#e8381f"]]

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
def _mm_catalog():
    import predict_lab as PL
    return PL.mm_catalog()


@st.cache_data(show_spinner=False)
def _mm_F():
    """멀티모달 특성 {갈래: {특성: 주간 × 19품목}} — 2026-10-26 기준 고정 (predict_lab.mm_features, 처음 한 번 1–3분)."""
    import predict_lab as PL
    return PL.mm_features()


def _mm_picker(item: str, key: str, cont=st):
    """④ 탭 공용: 멀티모달 설명변수 선택 (갈래 한꺼번에 + 개별). 19개 품목이 아니면 빈 목록."""
    import predict_lab as PL
    try:
        F = _mm_F()
    except Exception as ex:
        cont.caption(f"멀티모달 특성을 읽지 못함: {ex}")
        return []
    opts = PL.mm_options(F, item)
    if not opts:
        cont.caption(f"🧩 멀티모달 설명변수: {short(item)}은(는) 멀티모달 자료가 있는 19개 품목이 아니라 선택할 수 없음")
        return []
    g1, g2 = cont.columns([1, 3])
    fams = g1.multiselect("🧩 멀티모달 갈래 (한꺼번에)", list(F), key=f"{key}_fams",
                          help="포트폴리오 › 멀티모달 서브탭과 같은 자료: 텍스트(FinBERT 감성: WASDE·EIA STEO·베이지북·뉴스) · 리포트 수치(WASDE·EIA 재고·STEO·NASS·LME·ETF) · "
                               "CFTC COT · 기상(엘니뇨·Open-Meteo 주산지) · 매크로·금융·월간 거시(FRED, 품목 민감도 β × 값). 모두 발표 이후 값만.")
    pick = g2.multiselect(f"멀티모달 설명변수 ({short(item)}에 값이 있는 {len(opts)}개)", opts, [o for o in opts if o.split(" · ")[0] in fams],
                          key=f"{key}_pick_{item}_{abs(hash(tuple(fams)))}")
    if pick:
        cont.caption("멀티모달 값은 품목별 과거 자료로만 표준화하고, 자료가 없는 구간(예: LME 재고 2014 이전, 뉴스 2017 이전, 작황 비시즌)은 0(평균)으로 채움 · 주간 값 → 분석 주기로는 그 시점까지 나온 마지막 값.")
    return pick


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
st.sidebar.caption(" · ".join(f"{k}: {a}–{b}" for k, (a, b) in A.CRISES.items() if k in shade_sel))
st.sidebar.divider()
ROLL = st.sidebar.checkbox("연속월물 롤오버 갭 보정", True, key="roll",
                           help="Yahoo =F 연속월물은 만기 교체일에 두 계약의 가격차만큼 가짜 수익률이 생김 (곡물은 매년 같은 달 → 가짜 계절성). "
                                "거래소 만기 규칙으로 롤일을 찾아 검증된 13개 품목의 롤 당일 수익률을 0으로 보정. 상세: ⑥ 데이터·방법론")
st.sidebar.caption("수익률 = 100×로그차분 (금리·스프레드는 단순차분). 휴장 불일치는 최대 5일 ffill 후 공통 구간만 사용. "
                   "연속월물 롤오버 갭 완화를 위해 ±15σ 클리핑.")


def _update_data():
    """증분 수집 → 정제 → 롤 보정 → 품질 점검 → 엑셀 내보내기 (run_daily.bat에서 대시보드 HTML 생성만 뺀 단계). 끝나면 캐시를 비우고 다시 그림."""
    import subprocess, sys, os
    from pathlib import Path
    root = Path(__file__).parent
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    with st.sidebar.status("데이터 업데이트 중… (보통 3–6분)", expanded=True) as stt:
        for script, nm in [("collect.py", "① 수집 (Yahoo·FRED·akshare·COT 등)"), ("clean.py", "② 정제"), ("roll_adjust.py", "③ 롤오버 보정"),
                           ("data_quality.py", "④ 품질 점검"), ("export_excel.py", "⑤ 엑셀 내보내기 (data/excel)")]:
            st.write(nm)
            r = subprocess.run([sys.executable, script], cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
            if r.returncode != 0:
                stt.update(label=f"실패: {script}", state="error")
                st.code((r.stderr or r.stdout)[-1500:])
                return
        stt.update(label="업데이트 완료", state="complete")
    st.cache_data.clear()
    st.session_state["_upd_msg"] = f"데이터 업데이트 완료 ({pd.Timestamp.now():%m-%d %H:%M})"
    st.rerun()


st.sidebar.divider()
_yf_end = pd.to_datetime(cat.loc[cat.id.str.startswith("yf:"), "end"]).max()
st.sidebar.caption(f"데이터 기준일: **{_yf_end:%Y-%m-%d}** (선물 시세 최신일)")
if st.sidebar.button("🔄 데이터 업데이트", key="data_update", use_container_width=True,
                     help="최신 시세를 증분 수집 → 정제 → 롤 보정 후 화면을 다시 계산. 인터넷 연결 필요. ⑥ 실험 결과(AUC 등 근거표)와 ⑤ 포트폴리오의 롱숏·대안 시계열·멀티모달 서브탭(결과 고정)은 갱신되지 않음"):
    _update_data()
if "_upd_msg" in st.session_state:
    st.sidebar.success(st.session_state.pop("_upd_msg"))


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


def range_of(*Ms, nodiag=True, step=0.05):
    """실제 최소~최대 (대각 제외) → 무지개 전체를 실제 값 범위에 펼침 (값이 대부분 양수인 상관행렬에서도 색이 다양해짐)."""
    vals = []
    for M in Ms:
        a = np.asarray(M, float)
        if nodiag and a.ndim == 2 and a.shape[0] == a.shape[1]:
            a = a[~np.eye(a.shape[0], dtype=bool)]
        vals.append(a[np.isfinite(a)])
    v = np.concatenate(vals) if vals else np.array([0.0])
    lo, hi = np.floor(v.min() / step) * step, np.ceil(v.max() / step) * step
    return (lo, hi) if hi > lo else (lo - step, hi + step)


def heat(M: pd.DataFrame, title, zmin=None, zmax=None, fmt=".2f", h=420, scale=None, text_max=30, nodiag=False):
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
                               textfont=dict(size=10 if max(M.shape) <= 12 else 8), hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>"))
    fig.update_layout(title=title, height=h, margin=dict(l=10, r=10, t=40, b=10),
                      yaxis=dict(autorange="reversed", dtick=1), xaxis=dict(dtick=1, tickangle=-45))
    return fig


def heat_row(mats, titles, ranges, scales=None, groups=None, fmt=".2f", h=470, text_max=30, nodiag=True):
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
                                 texttemplate=("%{z:" + fmt + "}") if max(M.shape) <= text_max else None,
                                 textfont=dict(size=9 if max(M.shape) <= 10 else 7.5),
                                 hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>"), 1, i + 1)
    for j, g in enumerate(ng):
        i0 = groups.index(g)
        fig.layout[f"coloraxis{'' if g == 1 else g}"] = dict(colorscale=scales[i0], cmin=ranges[i0][0], cmax=ranges[i0][1],
                                                            colorbar=dict(thickness=12, len=0.9, x=1.0 + 0.09 * j))
    fig.update_yaxes(autorange="reversed", dtick=1)
    fig.update_xaxes(dtick=1, tickangle=-90 if n >= 3 and max(m.shape[1] for m in mats) > 8 else -45, tickfont=dict(size=9 if n >= 3 else 10))
    fig.update_yaxes(tickfont=dict(size=10))
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

st.markdown("<style>[data-testid='stTabPanel'] [data-testid='stTab'] p {font-size:0.8rem !important;}</style>", unsafe_allow_html=True)

# 화면 설정 (⑨ 설정 탭) — 배경색. 위젯 값은 다음 실행 때 맨 위에서 읽어 CSS로 적용, data/ui_settings.json에 저장 (git 제외)
import json
UI_BG = {"기본 (Streamlit 테마 그대로)": None, "흰색": "#ffffff", "아이보리": "#fbf8f1", "따뜻한 회색": "#f4f3ef", "연회색": "#eef0f3",
         "연하늘": "#eaf3fb", "연민트": "#e9f6f1", "연라벤더": "#f1eef9", "연베이지": "#f6efe3", "연분홍": "#fbeff2", "연노랑": "#fdf8e1",
         "짙은 남색 (다크)": "#0f1b2d", "차콜 (다크)": "#1f2125", "짙은 녹색 (다크)": "#10251d", "검정 (다크)": "#0b0b0b", "직접 고르기": "custom"}
UI_FILE = A.DATA / "ui_settings.json"


def _ui_saved() -> dict:
    d = {"bg": "기본 (Streamlit 테마 그대로)", "bg_custom": "#f4f3ef", "sb": "본문보다 조금 진하게", "sb_custom": "#e8e6e1", "chart": True}
    try:
        d.update(json.loads(UI_FILE.read_text(encoding="utf-8")))
    except Exception:
        pass
    return d


def _hex_dim(h: str, k: float) -> str:
    r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
    f_ = (lambda v: int(max(0, min(255, v * (1 - k))))) if k > 0 else (lambda v: int(max(0, min(255, v + (255 - v) * -k))))
    return f"#{f_(r):02x}{f_(g):02x}{f_(b):02x}"


def _lum(h: str) -> float:
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ui_apply():
    sv = _ui_saved()
    ss = st.session_state
    nm = ss.get("ui_bg", sv["bg"]); cu = ss.get("ui_bg_custom", sv["bg_custom"])
    sbn = ss.get("ui_sb", sv["sb"]); sbc = ss.get("ui_sb_custom", sv["sb_custom"]); ch = ss.get("ui_chart", sv["chart"])
    now = {"bg": nm, "bg_custom": cu, "sb": sbn, "sb_custom": sbc, "chart": bool(ch)}
    if now != {k: sv.get(k) for k in now}:
        try:
            UI_FILE.write_text(json.dumps(now, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:
            pass
    bg = UI_BG.get(nm)
    bg = cu if bg == "custom" else bg
    if not bg:
        return
    dark_bg = _lum(bg) < 0.45
    sb = {"본문과 같게": bg, "본문보다 조금 진하게": _hex_dim(bg, 0.06) if not dark_bg else _hex_dim(bg, -0.08), "직접 고르기": sbc}.get(sbn, bg)
    css = (f".stApp, [data-testid='stHeader'], [data-testid='stBottomBlockContainer'] {{background-color:{bg} !important;}}"
           f"[data-testid='stSidebar'], [data-testid='stSidebar'] > div:first-child {{background-color:{sb} !important;}}")
    if dark_bg != DARK:                                                    # 테마와 밝기가 반대인 배경이면 글자색도 뒤집어 읽히게
        txt = "#ece9e2" if dark_bg else "#1f1f1f"
        css += (f".stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, "
                f".stApp [data-testid='stMarkdownContainer'], .stApp [data-testid='stCaptionContainer'], .stApp [data-testid='stWidgetLabel'] {{color:{txt} !important;}}")
    if ch:                                                                 # 차트 바탕을 투명하게 → 고른 배경색이 비침
        css += ".js-plotly-plot .main-svg {background:transparent !important;} .js-plotly-plot .plotly rect.bg {fill:transparent !important;}"
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


_ui_apply()
TAB_LABELS = ["★ 구매 의사결정", "◆ 리스크 (VaR·꼬리)", "① 시장 구조·동조성", "①+ 요인 모형", "② 레짐", "③ 변동성·공분산", "④ 수익률 예측", "⑤ 포트폴리오", "⑥ 실험 결과", "⑦ 모델 도감", "⑧ 데이터·방법론", "⑨ 설정"]
_tdef = st.session_state.get("_test_tab")                                   # 헤드리스 테스트(AppTest)는 탭 상태를 보내지 않음 → 이 값으로 탭을 고름
tabs = st.tabs(TAB_LABELS, key="main_tab", on_change="rerun", default=_tdef if _tdef in TAB_LABELS else None)

# ============================================================================
# ① 동조성
# ============================================================================
def tab_comove():
    st.subheader("원자재 간 동조성 (Co-movement)")
    st.caption("상관 · 꼬리 의존 · 시간에 따른 동조성(평균 상관 · 흡수비율). 주성분 · 스필오버 · TVP-VAR · FAVAR는 '①+ 요인 모형' 탭으로 이동.")
    ids = st.multiselect("분석 품목", ALL_IDS, default=DEF_COMOVE, format_func=lbl, key="cm_ids")
    R = panel_or_stop(ids)
    if R is None:
        return
    m = A.crisis_mask(R.index, shade_sel)

    st.markdown("##### 상관행렬: 평시 vs 위기")
    C_all = R.corr()
    if m.sum() > 20 and (~m).sum() > 20:
        C_n, C_c = R[~m].corr(), R[m].corr()
        rc, vd = range_of(C_n, C_c), vmax_of(C_c - C_n, nodiag=True)
        st.plotly_chart(heat_row([C_n, C_c, C_c - C_n], [f"평시 ({(~m).sum():,}개)", f"위기 구간 ({m.sum():,}개)", "차이 (위기 − 평시)"],
                                 [rc, rc, (-vd, vd)], groups=[1, 1, 2], h=500), key="h123")
        st.caption(f"색 = 무지개 (보라 낮음 → 초록 → 빨강 높음), 범위는 실제 값에 맞춤 (평시·위기 공통 {rc[0]:+.2f}~{rc[1]:+.2f}, 차이 ±{vd:.2f}) · "
                   "대각(자기 자신 = 1)은 비움.")
        off = ~np.eye(len(C_n), dtype=bool)
        st.caption(f"평균 쌍별 상관: 평시 {C_n.values[off].mean():.3f} → 위기 {C_c.values[off].mean():.3f}. "
                   "위기 시 상관 상승은 분산효과 감소를 의미(단, 변동성 상승에 따른 상관 편향 — Forbes-Rigobon(2002) — 에 유의).")
    else:
        cc = st.columns(2)
        cc[0].plotly_chart(heat(C_all, "전체 기간", *range_of(C_all), nodiag=True), key="h0")
        cc[1].info("선택 기간 안에 위기 구간 표본이 부족합니다.")
    if R.shape[1] >= 2:
        import risk as RK
        q_ = st.select_slider("꼬리 의존: 꼬리 분위 q", [0.01, 0.025, 0.05, 0.10], 0.05, key="cm_tdq")
        TL, TU = RK.tail_dependence(R, q_), RK.tail_dependence(R, q_, upper=True)
        rt = range_of(TL, TU)
        st.plotly_chart(heat_row([TL, TU], [f"동반 폭락 P(j 하위 {q_:.0%} | i 하위 {q_:.0%}) — 보유자 위험", f"동반 급등 (상위 {q_:.0%}) — 구매자 위험"],
                                 [rt, rt], scales=[SEQ, SEQ], h=500), key="cm_td")
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


@st.cache_data(show_spinner=False)
def _rv_panel(comm: tuple, macro: tuple, freq: str, roll: bool):
    import regime_mv as RM
    return RM.build_panel(_levels(None, roll), list(comm), list(macro), freq, WB=RM.load_wb() if freq == "M" else None)


@st.cache_data(show_spinner=False)
def _rv_desc(comm: tuple, macro: tuple, freq: str, roll: bool, tgt: str, K: int, cov: str):
    import regime_mv as RM
    return RM.describe(_rv_panel(comm, macro, freq, roll), tgt, K, cov)


@st.cache_data(show_spinner=False)
def _rv_bt(comm: tuple, macro: tuple, freq: str, roll: bool, tgt: str, K: int, cov: str):
    import regime_mv as RM
    return RM.compare_sets(_rv_panel(comm, macro, freq, roll), tgt, list(comm), list(macro), K, cov, freq)


@st.cache_data(show_spinner=False)
def _rv_price(tgt: str, freq: str):
    import regime_mv as RM
    if tgt.startswith("wb:"):
        return RM.load_wb()[tgt].dropna()
    return _levels(None, False)[tgt].dropna().resample("W-FRI" if freq == "W" else "ME").last()


RV_MODELS = ["Hidden Markov (다변량 레짐)", "Markov-switching 회귀", "Kalman TVP", "Threshold 회귀 (TAR)", "STAR (평활 전이)"]
RV_COL = ["#2ea043", "#d29922", "#da3633", "#8250df"]


def tab_regime():
    import regime_mv as RM
    st.subheader("레짐 — 원자재 + 거시를 함께 넣고 지금 국면 찾기")
    st.caption("원자재 수익률과 거시 변수(금리·신용 스프레드·VIX·달러·주가·운임)를 한 패널에 넣고 국면을 추정. "
               "HMM = 모든 변수의 결합 분포가 달라지는 숨은 국면 · MS 회귀 = 국면마다 Y의 베타가 다름 · Kalman TVP = 베타가 연속적으로 변함 · "
               "TAR·STAR = 관측 변수(예: VIX)가 문턱을 넘으면 관계가 바뀜")
    c0, c1, c2 = st.columns([1.1, 2.4, 2.4])
    fq = c0.radio("빈도", ["주간 (선물, 2000–)", "월간 (세계은행, 1990–)"], key="rv_fq",
                  help="주간 = Yahoo 선물 근월물(2000년부터). 월간 = 세계은행 Pink Sheet 월평균 가격(1960–) + 거시 월말 값 — VIX가 1990년부터라 공통 표본은 1990–")
    F = "W" if fq.startswith("주간") else "M"
    if F == "W":
        opts = [i for i in COMMOD_IDS if not i.startswith("etf:")]
        dflt = [i for i in RM.COMM_W_DEFAULT if i in opts]; fmt = lbl
    else:
        wbn = RM.wb_names(); opts = sorted(wbn, key=lambda i: wbn[i])
        dflt = [i for i in RM.COMM_M_DEFAULT if i in opts]; fmt = lambda i: wbn.get(i, i)
    comm = c1.multiselect("원자재 (복수)", opts, dflt, format_func=fmt, key=f"rv_comm_{F}")
    mopts = [i for i in RM.MACRO if i == "spread:10y2y" or i in NAME]
    macro = c2.multiselect("거시 변수 (복수)", mopts, [i for i in RM.MACRO_DEFAULT if i in mopts], format_func=lambda i: RM.MACRO[i][0], key="rv_macro",
                           help="변환: 금리 = 기간 차분(%p), 장단기 금리차·신용 스프레드 = 수준, VIX = 로그 수준, 나머지 = 로그수익률(%)")
    if not comm:
        st.info("원자재를 1개 이상 고르세요."); return
    d1, d2, d3, d4 = st.columns([1.6, 1.8, 1, 1])
    tgt = d1.selectbox("대상 품목 (Y)", comm, format_func=fmt, key=f"rv_tgt_{F}", help="레짐 순서(레짐1 = Y 약세)와 백테스트·회귀 모델의 Y")
    model = d2.selectbox("모델", RV_MODELS, key="rv_model")
    K = d3.radio("레짐 수", [2, 3, 4], index=1, horizontal=True, key="rv_k")
    cov = d4.selectbox("HMM 공분산", ["full", "diag"], key="rv_cov", help="full = 변수 간 상관까지 레짐마다 다름 (모수 많음), diag = 변수별 분산만")
    X = _rv_panel(tuple(comm), tuple(macro), F, ROLL)
    if len(X) < 120:
        st.warning(f"공통 표본이 {len(X)}개뿐입니다 — 늦게 시작하는 변수(예: 원/달러 2003–, 광의 달러 2006–)를 빼세요."); return
    st0 = {c: (_rv_panel((c,), (), F, ROLL) if c in comm else _rv_panel((), (c,), F, ROLL)).index.min() for c in X.columns}
    late = max(st0, key=st0.get)
    why = (f"시작점을 정한 변수: {RM.label(late)}" if st0[late] > min(st0.values()) else "1990년 하한에서 시작")
    st.caption(f"공통 표본 {X.index.min():%Y-%m} – {X.index.max():%Y-%m} · {len(X):,}개 ({'주간' if F == 'W' else '월간'}) · 변수 {X.shape[1]}개 · {why}")
    if model.startswith("Hidden"):
        _rv_hmm(X, tgt, K, cov, F, comm, macro, fmt)
    else:
        _rv_reg(X, tgt, model, K, F)
    with st.expander("전 품목 walk-forward 백테스트 — 거시 변수를 넣으면 국면 예측이 나아지나 (⑥ 실험)"):
        _rv_experiment()
    with st.expander("변동성 레짐 (단일 품목 HMM 3상태 · 반감기 안정성) — 리스크 탭 HMM λ의 근거"):
        vol_regime()


def _rv_hmm_3d(X: pd.DataFrame, G: pd.DataFrame, names: dict, tgt: str):
    """레짐 탭 HMM 3D: ① 변수 3개 공간의 국면 구름 ② 시간 × 국면 × 확률 지형 ③ (3국면) 확률 삼각형 궤적. 새 변수 없이 이미 추정한 결과만 사용."""
    import regime_mv as RM
    st.markdown("##### 3D 보기 — 국면이 변수 공간·시간 위에서 어떻게 나뉘나")
    lab = pd.Series(G.values.argmax(1), index=G.index)
    cols = list(X.columns)
    ink = "#e8e6e1" if DARK else "#0b0b0b"
    pref = [tgt] + [c for c in ("fred:VIXCLS", "fred:BAMLH0A0HYM2", "fred:BAA10Y", "fred:DTWEXBGS") if c in cols] + cols
    dflt = list(dict.fromkeys(pref))[:3]
    tabs = ["국면 구름 (변수 3개 공간)", "국면 확률 지형 (시간 × 국면 × 확률)"] + (["확률 삼각형 궤적 (3국면)"] if G.shape[1] == 3 else [])
    tt = st.tabs(tabs)
    with tt[0]:
        a1, a2, a3 = st.columns(3)
        ax = [a1.selectbox("x축", cols, cols.index(dflt[0]), format_func=RM.label, key="rv3_x"),
              a2.selectbox("y축", cols, cols.index(dflt[1 % len(dflt)]), format_func=RM.label, key="rv3_y"),
              a3.selectbox("z축", cols, cols.index(dflt[2 % len(dflt)]), format_func=RM.label, key="rv3_z")]
        Z = X[ax].reindex(G.index).dropna()
        Z = (Z - Z.mean()) / Z.std()
        f = go.Figure()
        for j, c in enumerate(G.columns):
            m = lab.reindex(Z.index).values == j
            if not m.any():
                continue
            zz = Z[m]
            f.add_trace(go.Scatter3d(x=zz.iloc[:, 0], y=zz.iloc[:, 1], z=zz.iloc[:, 2], mode="markers", name=f"{c}: {names[c]} ({m.sum()}개)",
                                     marker=dict(size=2.6, color=RV_COL[j % 4], opacity=0.55),
                                     text=[f"{d:%Y-%m-%d}" for d in zz.index], hovertemplate="%{text}<extra>" + c + "</extra>"))
            ctr = zz.mean()
            f.add_trace(go.Scatter3d(x=[ctr.iloc[0]], y=[ctr.iloc[1]], z=[ctr.iloc[2]], mode="markers+text", text=[c], textposition="top center",
                                     marker=dict(size=11, color=RV_COL[j % 4], symbol="diamond", line=dict(color=ink, width=1.5)), showlegend=False,
                                     hovertemplate=f"{c} 중심<extra></extra>"))
        last = Z.iloc[-26:]
        f.add_trace(go.Scatter3d(x=last.iloc[:, 0], y=last.iloc[:, 1], z=last.iloc[:, 2], mode="lines+markers", name="최근 26기간 경로",
                                 line=dict(color=ink, width=4), marker=dict(size=3, color=ink)))
        f.update_layout(height=580, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.02, font=dict(size=10)),
                        scene=dict(xaxis_title=RM.label(ax[0]) + " (z)", yaxis_title=RM.label(ax[1]) + " (z)", zaxis_title=RM.label(ax[2]) + " (z)",
                                   camera=dict(eye=dict(x=1.5, y=-1.5, z=0.9))))
        st.plotly_chart(f, key="rv3_cloud")
        st.caption("점 = 각 기간(주·월), 색 = 그 기간에 가장 가능성 높은 국면(전 기간 평활), ◆ = 국면 중심, 검은 선 = 최근 26기간 경로(지금 어느 국면 쪽으로 가는지). "
                   "축은 표준화(z) 값 — 국면 구름이 잘 떨어져 있을수록 그 변수들로 국면이 뚜렷이 구분됨. 마우스로 돌려 보기.")
    with tt[1]:
        f = go.Figure(go.Surface(x=G.index, y=list(range(G.shape[1])), z=G.T.values, colorscale=SEQ, cmin=0, cmax=1, showscale=True,
                                 colorbar=dict(title="확률", thickness=12, len=0.6),
                                 contours=dict(z=dict(show=True, usecolormap=True, project_z=True, highlightcolor=ink)),
                                 hovertemplate="%{x|%Y-%m-%d} · 국면 %{y}<br>확률 %{z:.0%}<extra></extra>"))
        f.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0),
                        scene=dict(xaxis_title="", yaxis=dict(title="국면", tickvals=list(range(G.shape[1])), ticktext=[f"{c}" for c in G.columns]),
                                   zaxis=dict(title="확률", range=[0, 1]), aspectratio=dict(x=2.2, y=0.8, z=0.6), camera=dict(eye=dict(x=0.4, y=-2.0, z=1.0))))
        st.plotly_chart(f, key="rv3_surf")
        st.caption("능선 = 그 국면에 있을 확률이 높았던 시기. 바닥에 비친 등고선이 2D 확률 그래프와 같음 — 능선이 국면 사이를 자주 오가면 국면이 불안정, 길게 이어지면 지속성이 큼.")
    if G.shape[1] == 3:
        with tt[2]:
            pr = G.values
            xs = pr[:, 1] + pr[:, 2] / 2
            ys = pr[:, 2] * np.sqrt(3) / 2
            tnum = np.arange(len(G))
            f = go.Figure()
            for (vx, vy), c, j in zip([(0, 0), (1, 0), (0.5, np.sqrt(3) / 2)], G.columns, range(3)):
                f.add_trace(go.Scatter3d(x=[vx, vx], y=[vy, vy], z=[0, len(G)], mode="lines", line=dict(color=RV_COL[j], width=3, dash="dash"),
                                         name=f"{c}: {names[c]} (꼭짓점 = 확률 100%)"))
            f.add_trace(go.Scatter3d(x=xs, y=ys, z=tnum, mode="lines", line=dict(color=lab.values, colorscale=[[0, RV_COL[0]], [0.5, RV_COL[1]], [1, RV_COL[2]]],
                                                                           cmin=0, cmax=2, width=4), showlegend=False,
                                     text=[f"{d:%Y-%m-%d}" for d in G.index], hovertemplate="%{text}<extra></extra>"))
            f.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.02, font=dict(size=10)),
                            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                                       zaxis=dict(title="시간", tickvals=list(tnum[::max(1, len(tnum) // 6)]),
                                                  ticktext=[f"{G.index[k]:%Y}" for k in tnum[::max(1, len(tnum) // 6)]]),
                                       aspectratio=dict(x=1, y=1, z=1.6), camera=dict(eye=dict(x=1.6, y=-1.6, z=0.6))))
            st.plotly_chart(f, key="rv3_simplex")
            st.caption("바닥 삼각형의 세 꼭짓점 = 각 국면 확률 100%, 가운데 = 셋이 비슷(불확실). 위로 갈수록 최근 — 선이 한 꼭짓점 기둥에 붙어 있으면 그 국면이 확실했던 시기, "
                       "가운데를 가로지르면 국면 전환기.")


def _regime_reg_3d(model: str, o: dict, D, tgt: str):
    """레짐 탭 MS 회귀 · TAR · STAR 3D (이미 추정한 결과만 사용, 새 변수 없음).
    MS = 국면 확률 지형 + 국면별 계수 기둥 · TAR/STAR = 문턱 변수 × 주요 설명변수 × 예측 Y 반응 곡면(TAR는 γ에서 절벽, STAR는 매끄러운 전이)."""
    import regime_mv as RM
    ex = o.get("extra", {}) or {}
    ink = "#e8e6e1" if DARK else "#0b0b0b"
    lab_ = lambda c: RM.label(str(c).split(" (")[0])[:22] if ":" in str(c) else str(c)[:22]  # noqa: E731
    if model.startswith("Markov"):
        G, B = ex.get("probs"), ex.get("state_betas")
        if not isinstance(G, pd.DataFrame):
            return
        st.markdown("##### 3D 보기 — Markov-switching: 국면 확률 지형 · 국면별 계수")
        t1, t2 = st.tabs(["국면 확률 지형 (시간 × 국면 × 확률)", "국면별 계수 기둥 (국면 × 변수 × β)"])
        with t1:
            f = go.Figure(go.Surface(x=G.index, y=list(range(G.shape[1])), z=G.T.values, colorscale=SEQ, cmin=0, cmax=1, colorbar=dict(title="확률", thickness=12, len=0.6),
                                     contours=dict(z=dict(show=True, usecolormap=True, project_z=True)), hovertemplate="%{x|%Y-%m-%d} · 국면 %{y}<br>확률 %{z:.0%}<extra></extra>"))
            f.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0),
                            scene=dict(xaxis_title="", yaxis=dict(title="국면", tickvals=list(range(G.shape[1])), ticktext=[str(c) for c in G.columns]),
                                       zaxis=dict(title="확률", range=[0, 1]), aspectratio=dict(x=2.2, y=0.8, z=0.6), camera=dict(eye=dict(x=0.4, y=-2.0, z=1.0))))
            st.plotly_chart(f, key="rg3_ms_surf")
            st.caption("능선 = 그 국면일 확률이 높았던 시기(필터 확률). 능선이 길게 이어지면 국면 지속성이 큼.")
        with t2:
            if isinstance(B, pd.DataFrame) and B.size:
                f = go.Figure()
                vm = float(np.abs(B.values).max()) or 1.0
                for i_, r_ in enumerate(B.index):
                    for j_, c_ in enumerate(B.columns):
                        v = float(B.loc[r_, c_])
                        f.add_trace(go.Scatter3d(x=[j_, j_], y=[i_, i_], z=[0, v], mode="lines", showlegend=False,
                                                 line=dict(color="#e34948" if v > 0 else "#2a78d6", width=10),
                                                 hovertemplate=f"{r_} · {lab_(c_)}<br>β {v:+.3f}<extra></extra>"))
                        f.add_trace(go.Scatter3d(x=[j_], y=[i_], z=[v], mode="markers", showlegend=False, marker=dict(size=4, color=ink), hoverinfo="skip"))
                f.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0),
                                scene=dict(xaxis=dict(title="", tickvals=list(range(B.shape[1])), ticktext=[lab_(c) for c in B.columns]),
                                           yaxis=dict(title="", tickvals=list(range(B.shape[0])), ticktext=[str(r) for r in B.index]),
                                           zaxis=dict(title="β", range=[-vm * 1.1, vm * 1.1]), aspectratio=dict(x=1.6, y=0.9, z=0.7), camera=dict(eye=dict(x=1.4, y=-1.6, z=0.9))))
                st.plotly_chart(f, key="rg3_ms_beta")
                st.caption("기둥 높이 = 그 국면에서의 계수(빨강 +, 파랑 −). 같은 변수의 기둥이 국면마다 크게 다르면 관계가 국면에 따라 바뀐다는 뜻 "
                           "(설명변수는 주성분으로 줄여 추정하므로 원 변수 계수로 되돌린 값).")
        return
    rc = ex.get("regime_coef"); thr = ex.get("thr_name")
    if not isinstance(rc, pd.DataFrame) or thr not in list(D.names):
        return
    j = list(D.names).index(thr)
    b_lo, b_hi = rc.iloc[:, 0].values, rc.iloc[:, 1].values
    cand = [k for k in range(len(D.names)) if k != j]
    if not cand:
        return
    dflt = max(cand, key=lambda k: abs(b_hi[k + 1] - b_lo[k + 1]))
    st.markdown(f"##### 3D 보기 — {'TAR' if model.startswith('Threshold') else 'STAR'} 반응 곡면: 문턱 변수에 따라 관계가 어떻게 바뀌나")
    kx = st.selectbox("y축 설명변수 (기본 = 두 국면 계수 차이가 가장 큰 변수)", cand, cand.index(dflt), format_func=lambda k: lab_(D.names[k]), key="rg3_kx")
    Xa = np.vstack([D.X_tr, D.X_te]); ya = np.concatenate([D.y_tr, D.y_te])
    qo = Xa[:, j] * D.sd[j] + D.mu[j]
    qg = np.linspace(np.quantile(qo, 0.02), np.quantile(qo, 0.98), 60)
    xg = np.linspace(np.quantile(Xa[:, kx], 0.02), np.quantile(Xa[:, kx], 0.98), 40)
    QQ, XX = np.meshgrid(qg, xg)
    Z = np.zeros((len(xg), len(qg), len(D.names)))
    Z[:, :, j] = (QQ - D.mu[j]) / D.sd[j]; Z[:, :, kx] = XX
    Xc = np.concatenate([np.ones(Z.shape[:2] + (1,)), Z], axis=2)
    if model.startswith("Threshold"):
        gam = float(ex["gamma"])
        Y = np.where(QQ <= gam, Xc @ b_lo, Xc @ b_hi)
        reg_pt = qo > gam
        sub = f"γ = {gam:.3g} (하위 레짐 {ex.get('share_low', np.nan):.0%})"
    else:
        tr = ex.get("transition")
        Gq = np.interp(QQ, tr.index.values, tr.values) if isinstance(tr, pd.Series) else np.full(QQ.shape, 0.5)
        Y = Xc @ b_lo + Gq * (Xc @ (b_hi - b_lo))
        reg_pt = (np.interp(qo, tr.index.values, tr.values) if isinstance(tr, pd.Series) else np.full(len(qo), 0.5)) > 0.5
        sub = f"전이 중심 c = {ex.get('c', np.nan):.3g}, 기울기 γ = {ex.get('gamma', np.nan):.2f}"
    f = go.Figure(go.Surface(x=qg, y=xg, z=Y, colorscale=DIV, cmid=0, opacity=0.9, colorbar=dict(title="예측 Y", thickness=12, len=0.6),
                             hovertemplate=f"{lab_(thr)} %{{x:.3g}}<br>{lab_(D.names[kx])} (z) %{{y:.2f}}<br>예측 Y %{{z:+.3f}}<extra></extra>"))
    for m_, nm_, col_ in [(~reg_pt, "하위 레짐 관측치", "#2a78d6"), (reg_pt, "상위 레짐 관측치", "#e34948")]:
        if m_.any():
            f.add_trace(go.Scatter3d(x=qo[m_], y=Xa[m_, kx], z=ya[m_], mode="markers", name=nm_, marker=dict(size=2, color=col_, opacity=0.35), hoverinfo="skip"))
    zq = np.quantile(ya, [0.02, 0.98])
    f.update_layout(height=580, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.02, font=dict(size=10)),
                    scene=dict(xaxis_title=f"문턱 변수: {lab_(thr)} (원 단위)", yaxis_title=f"{lab_(D.names[kx])} (표준화)", zaxis=dict(title=f"Y: {RM.label(tgt)[:16]}", range=[min(zq[0], Y.min()), max(zq[1], Y.max())]),
                               aspectratio=dict(x=1.5, y=1.2, z=0.8), camera=dict(eye=dict(x=1.5, y=-1.6, z=0.8))))
    st.plotly_chart(f, key="rg3_thr_surf")
    st.caption(f"{sub}. 곡면 = 다른 설명변수를 평균(0)에 두고 문턱 변수(x)와 고른 변수(y)만 바꿨을 때의 모델 예측 Y. "
               + ("TAR는 문턱 γ에서 곡면이 '절벽'처럼 끊기며 기울기가 바뀜 — 절벽 양쪽 기울기 차이가 클수록 국면에 따라 관계가 크게 다름. "
                  if model.startswith("Threshold") else "STAR는 문턱 근처에서 곡면이 매끄럽게 휘어짐 — 휘는 구간이 좁을수록(γ 큼) TAR처럼 급격한 전환. ")
               + "점 = 실제 관측치(파랑 하위·빨강 상위 레짐).")


def _kalman_3d(B: pd.DataFrame, Bf: pd.DataFrame | None, tgt: str):
    """레짐 탭 Kalman TVP 3D: 설명변수 × 시간 × 시변 β 곡면 (이미 추정한 β만 사용)."""
    import regime_mv as RM
    st.markdown("##### 3D 보기 — 시변 계수(β) 지형: 어느 변수의 영향이 언제 커졌나")
    kind = st.radio("β", ["평활 (전 기간 자료, 형태 보기)", "필터 (그 시점까지 자료만, 실시간)"], horizontal=True, key="kf3_kind") if Bf is not None else "평활"
    M = (Bf if (Bf is not None and kind.startswith("필터")) else B).copy()
    M = M[[c for c in M.columns if c.lower() not in ("const", "상수", "intercept")]]
    if M.empty:
        return
    vm = float(np.nanpercentile(np.abs(M.values), 99)) or 1.0
    ylab = [RM.label(c)[:24] for c in M.columns]
    f = go.Figure(go.Surface(x=M.index, y=list(range(M.shape[1])), z=M.T.values, colorscale=DIV, cmin=-vm, cmax=vm,
                             colorbar=dict(title="β", thickness=12, len=0.6),
                             contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
                             hovertemplate="%{x|%Y-%m-%d}<br>β %{z:+.3f}<extra></extra>"))
    f.add_trace(go.Surface(x=M.index, y=list(range(M.shape[1])), z=np.zeros((M.shape[1], len(M))), showscale=False, opacity=0.18,
                           colorscale=[[0, "#898781"], [1, "#898781"]], hoverinfo="skip"))
    f.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0),
                    scene=dict(xaxis_title="", yaxis=dict(title="", tickvals=list(range(M.shape[1])), ticktext=ylab),
                               zaxis_title="β", aspectratio=dict(x=2.2, y=1.1, z=0.7), camera=dict(eye=dict(x=0.6, y=-2.0, z=0.9))))
    st.plotly_chart(f, key="kf3_surf")
    st.caption(f"Y = {RM.label(tgt)}. 위로 솟은 곳 = 그 변수가 Y를 같은 방향으로 크게 움직이던 시기, 아래로 꺼진 곳 = 반대 방향, 회색 판 = β 0. "
               "산맥이 시간에 따라 높아졌다 낮아지면 관계가 국면에 따라 바뀐 것. 설명변수는 표준화돼 있어 높이를 서로 비교 가능.")


def _rv_hmm(X, tgt, K, cov, F, comm, macro, fmt):
    import regime_mv as RM
    with st.spinner("다변량 HMM 추정 중…"):
        try:
            d = _rv_desc(tuple(comm), tuple(macro), F, ROLL, tgt, int(K), cov)
        except Exception as ex:
            st.error(f"추정 실패: {ex} — 레짐 수를 줄이거나 공분산을 diag로"); return
    S, Fp, G = d["summary"], d["filt"], d["smooth"]
    cur = Fp.iloc[-1]; k_now = cur.idxmax(); unit = "주" if F == "W" else "개월"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"지금 국면 ({Fp.index[-1]:%Y-%m-%d})", f"{k_now} · {cur.max():.0%}", d["names"][k_now], delta_color="off")
    m2.metric(f"{k_now}의 {fmt(tgt)} 평균 수익률", f"{S.loc[k_now, 'Y 평균 수익률']:+.2f}%/{unit[0] if F == 'W' else '월'}",
              f"변동성 {S.loc[k_now, 'Y 변동성']:.2f}%", delta_color="off")
    m3.metric("기대 지속", f"{S.loc[k_now, '기대 지속 (기간)']:.0f}{unit}", f"유지확률 {S.loc[k_now, '유지확률']:.0%}", delta_color="off")
    m4.metric("과거 비중", f"{S.loc[k_now, '비중']:.0%}", "전 기간 중 이 국면", delta_color="off")
    # ① 가격 + 국면 음영 + 확률
    px_ = _rv_price(tgt, F).reindex(G.index, method="ffill")
    st_ = G.values.argmax(1)
    f = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.06,
                      subplot_titles=(f"{fmt(tgt)} 가격 — 배경 = 가장 가능성 높은 국면 (전 기간 평활)", "국면 확률 (실선 = 실시간 필터: 그 시점까지 자료만)"))
    f.add_trace(go.Scatter(x=px_.index, y=px_.values, line=dict(color=PAL[0], width=1.5), name="가격", showlegend=False), 1, 1)
    b0 = 0
    for t in range(1, len(st_) + 1):
        if t == len(st_) or st_[t] != st_[b0]:
            f.add_vrect(x0=G.index[b0], x1=G.index[min(t, len(st_) - 1)], fillcolor=RV_COL[st_[b0] % 4], opacity=0.28, line_width=0, layer="below", row=1, col=1)
            b0 = t
    for j, c in enumerate(G.columns):
        f.add_trace(go.Scatter(x=G.index, y=G[c], stackgroup="g", line=dict(width=0.3, color=RV_COL[j % 4]), name=f"{c}: {d['names'][c]}",
                               opacity=0.5), 2, 1)
        f.add_trace(go.Scatter(x=Fp.index, y=Fp[c], line=dict(width=0.8, color=RV_COL[j % 4], dash="dot"), showlegend=False), 2, 1)
    st.plotly_chart(fig_layout(f, 520, hovermode="x"), key="rv_fig_ts")
    _rv_hmm_3d(X, G, d["names"], tgt)
    # ② 국면 프로파일 · ③ 전이행렬
    g1, g2 = st.columns([1.15, 1])
    pv_ = g1.radio("프로파일", ["평균 (z)", "변동성 (전체 대비 배수)"], horizontal=True, key="rv_prof_kind",
                   help="평균 = 국면 안에서 그 변수의 평균이 평소보다 몇 표준편차 높은가. 변동성 = 국면 안의 출렁임이 평소의 몇 배인가 (1 = 평소)")
    isv = pv_.startswith("변동성")
    P = (d["vol"] if isv else d["profile"]).T.copy(); P.index = [RM.label(c) for c in P.index]
    P.columns = [f"{k}<br>{d['names'][k].replace(' · ', '<br>')}" for k in P.columns]
    fp = heat(P, "국면 프로파일 — " + ("변수별 변동성 (1 = 평소, 2 = 평소의 2배)" if isv else "변수별 평균 (표준화 z, + = 평소보다 높음)"),
              *((0, max(2.5, float(P.values.max()))) if isv else (None, None)), fmt=".1f", h=150 + 30 * len(P), text_max=40, scale=SEQ if isv else None)
    fp.update_layout(xaxis=dict(tickangle=0, tickfont=dict(size=11)), margin=dict(t=40, b=70))
    g1.plotly_chart(fp, key="rv_fig_prof")
    g2.plotly_chart(heat(d["trans"], "전이행렬 — 이번 국면(행) → 다음 기간 국면(열)", 0, 1, fmt=".2f", h=330, scale=SEQ), key="rv_fig_tr")
    g2.dataframe(S.style.format({"Y 평균 수익률": "{:+.3f}", "Y 변동성": "{:.2f}", "유지확률": "{:.1%}", "기대 지속 (기간)": "{:.1f}", "비중": "{:.0%}"}))
    # ③-2 국면별 상관 (수익률·변화 변수만 — 수준 변수의 상관은 의미가 약해 제외)
    rc = [c for c in X.columns if c in comm or RM.MACRO.get(c, ("", "ret"))[1] in ("ret", "diff")]
    if len(rc) >= 3:
        lab_ = pd.Series(G.values.argmax(1), index=G.index)
        Cs, ts_ = [], []
        Xr = X[rc].rename(columns=RM.label)
        for j, c in enumerate(G.columns):
            sub = Xr[lab_.values == j]
            if len(sub) >= 20:
                Cs.append(sub.corr()); ts_.append(f"{c} ({len(sub)}개)")
        if len(Cs) >= 2:
            Cs.append(Cs[-1] - Cs[0]); ts_.append(f"차이 ({ts_[-1].split(' ')[0]} − {ts_[0].split(' ')[0]})")
            rr_ = range_of(*Cs[:-1]); vd_ = vmax_of(Cs[-1], nodiag=True)
            hh_ = 160 + 30 * len(rc)
            for k0 in range(0, len(Cs), 2):
                cc_ = st.columns(2)
                for j0 in range(k0, min(k0 + 2, len(Cs))):
                    last_ = j0 == len(Cs) - 1
                    fz = heat(Cs[j0], ts_[j0], *((-vd_, vd_) if last_ else rr_), fmt=".1f", h=hh_ + 40, text_max=30, nodiag=True)
                    fz.update_traces(textfont=dict(size=10)); fz.update_xaxes(tickfont=dict(size=10), tickangle=-90); fz.update_yaxes(tickfont=dict(size=10))
                    cc_[j0 - k0].plotly_chart(fz, key=f"rv_fig_corr_{j0}")
            st.caption("국면별 상관 — 모델이 찾은 국면 안에서 변수들이 얼마나 같이 움직이는가 (① 시장 구조의 '평시 vs 위기'는 달력으로 정한 위기 구간, "
                       "여기는 데이터가 정한 국면). 마지막 칸 = 마지막 레짐(Y 강세) − 레짐1(Y 약세): 빨강이면 Y 강세 국면에서 같이 움직임이 커짐 → 분산 효과 감소.")
    st.caption(f"레짐1 = {fmt(tgt)} 평균 수익률이 가장 낮은 국면 → 마지막 레짐 = 가장 높은 국면. 위 음영·프로파일은 전 기간 추정(설명용, 미래 자료 포함 평활). "
               "백테스트는 아래처럼 매년 그 이전 자료로만 재추정 + 실시간 필터.")
    # ④ walk-forward 백테스트
    st.markdown("##### walk-forward 백테스트 — 국면으로 다음 기간 방향을 맞히나 (매년 재추정 · 실시간 필터)")
    st.caption("예측 = 다음 기간 국면 확률(지금 필터 × 전이행렬) × 국면별 Y 평균 수익률. 같은 표본에서 Y만(단변량) · 원자재만 · Y+거시 · 원자재+거시 비교. "
               "위약 = 포지션을 13기간 블록으로 섞음(롱 비중·지속성 유지).")
    if st.button("▶ 백테스트 실행 (약 30–90초)", key="rv_bt_run"):
        st.session_state["rv_bt"] = (tuple(comm), tuple(macro), F, tgt, int(K), cov)
    if st.session_state.get("rv_bt") == (tuple(comm), tuple(macro), F, tgt, int(K), cov):
        with st.spinner("4개 변수 묶음 × 매년 재추정 중…"):
            T, Ws = _rv_bt(tuple(comm), tuple(macro), F, ROLL, tgt, int(K), cov)
        if len(T):
            st.dataframe(T.set_index(["변수 묶음", "기간"]).style.format(
                {"판단 수": "{:.0f}", "변수 수": "{:.0f}", "롱 비중": "{:.0%}", "롱숏 Sharpe": "{:+.2f}", "롱/현금 Sharpe": "{:+.2f}", "B&H Sharpe": "{:+.2f}",
                 "방향 적중률": "{:.1%}", "OOS R²": "{:+.4f}", "최상−최하 레짐 수익 차": "{:+.2f}", "레짐 차 t": "{:+.2f}", "위약 95%": "{:+.2f}",
                 "위약 p": "{:.2f}", "전환 횟수/년": "{:.1f}"}, na_rep="–"))
            y = X[tgt]; f = go.Figure()
            for j, (nm, W) in enumerate(Ws.items()):
                m_ = W["E"].notna()
                r_ = (np.sign(W.loc[m_, "E"].astype(float)) * y.shift(-1)[m_]).fillna(0).cumsum()
                f.add_trace(go.Scatter(x=r_.index, y=r_.values, name=nm, line=dict(color=PAL[j], width=2 if "거시" in nm else 1.2)))
            any_ = next(iter(Ws.values())); m_ = any_["E"].notna()
            bh = y.shift(-1)[m_].fillna(0).cumsum()
            f.add_trace(go.Scatter(x=bh.index, y=bh.values, name="Buy & Hold", line=dict(color="#898781", dash="dot")))
            st.plotly_chart(shade(fig_layout(f, 340, title="누적 수익 (롱/숏, %) — 변수 묶음별", hovermode="x"), bh.index), key="rv_fig_bt")
            full = T[T["기간"] == "전체 OOS"].set_index("변수 묶음")
            best = full["롱숏 Sharpe"].idxmax()
            ok = (full["롱숏 Sharpe"] > full["위약 95%"]) & (T[T["기간"] != "전체 OOS"].groupby("변수 묶음")["롱숏 Sharpe"].min().reindex(full.index) > 0)
            st.info(f"판정: 전체 OOS 최고 = **{best}** (롱숏 Sharpe {full.loc[best, '롱숏 Sharpe']:+.2f}, 위약 95% {full.loc[best, '위약 95%']:+.2f}). "
                    + ("위약을 넘고 두 하위기간 모두 양수인 묶음: **" + ", ".join(ok[ok].index) + "**" if ok.any() else
                       "위약을 넘으면서 두 하위기간 모두 양수인 묶음은 **없음** → 이 품목에서는 국면이 '지금 상태 설명'에는 쓸 만해도 방향 예측력은 운과 구별되지 않음.")
                    + " 4개 묶음 중 최고를 고르는 것 자체가 유리하므로 위약 95%보다 확실히 높아야 의미 있음.")


def _rv_reg(X, tgt, model, K, F):
    import regime_mv as RM
    others = [c for c in X.columns if c != tgt]
    if not others:
        st.info("회귀 모델은 Y 외에 설명변수가 1개 이상 필요합니다."); return
    thr_def = "fred:VIXCLS" if "fred:VIXCLS" in others else others[0]
    e1, e2 = st.columns(2)
    thr = e1.selectbox("TAR·STAR 문턱 변수", others, others.index(thr_def), format_func=RM.label, key="rv_thr")
    mode = e2.radio("분석 목적", ["예측 (다음 기간)", "설명 (동시점 베타)"], horizontal=True, key="rv_mode",
                    help="예측 = 이번 기간 정보로 다음 기간 Y 수익률 (구매 판단용, 학습 70%·테스트 30%). 설명 = 같은 기간 관계(국면별 베타·헤지 비율)")
    md = "explain" if mode.startswith("설명") else "predict"
    hp_ = dict(thr_var=thr, ms_k=int(min(K, 3)), ms_pc=3, tar_boot=199, hmm_states=int(K), hmm_cov="full", kalman_q=1e-4, mode=md)
    if st.button(f"▶ {model} 실행", type="primary", key="rv_reg_run"):
        D = A.build_design(X, tgt, others, md, 1, 1, True, 0.7)
        with st.spinner("추정 중…"):
            try:
                o = A.run_model(model, D, RM.ANN[F], **hp_)
                st.session_state["rv_reg"] = (D, {model: o}, tgt, md, hp_, model)
            except Exception as ex:
                st.error(f"{model} 실패: {ex}")
    r = st.session_state.get("rv_reg")
    if r and r[5] == model and r[2] == tgt:
        D, out_, tgt_, md_, hp0, _ = r
        met = pd.DataFrame({k: v["metrics"] for k, v in out_.items()}).T
        st.dataframe(met[[c for c in met.columns if c in ("R² (학습)", "R² (테스트)", "RMSE (테스트)", "OOS R² (vs 학습평균)", "방향 적중률",
                                                         "부호전략 Sharpe")]].style.format("{:.4f}", na_rep="–"))
        model_detail(model, out_[model], D, tgt_, out_, hp0, P="rv_")
        ex_ = out_[model].get("extra", {})
        if model.startswith("Kalman") and isinstance(ex_.get("beta_smoothed"), pd.DataFrame):
            _kalman_3d(ex_["beta_smoothed"], ex_.get("beta_filtered"), tgt_)
        elif model.startswith(("Markov", "Threshold", "STAR")):
            _regime_reg_3d(model, out_[model], D, tgt_)
        st.caption("설명 모드 = 같은 시점 관계(국면별 베타·헤지 비율), 예측 모드 = 다음 기간 수익률 (학습 70% / 테스트 30%). "
                   "Markov-switching은 설명변수를 주성분 3개로 줄여 추정 (국면마다 모수가 늘어나기 때문).")


def _rv_experiment():
    df = _exp_header("regime_mv")
    if df is None:
        return
    full = df[df["기간"] == "전체 OOS"]
    pv = full.pivot_table(index=["빈도", "대상", "K"], columns="변수 묶음", values="롱숏 Sharpe")
    pv = pv[[c for c in ["Y만 (단변량)", "원자재만", "Y + 거시", "원자재 + 거시"] if c in pv]]
    st.dataframe(pv.rename(index=lambda i: _rv_lbl(i) if isinstance(i, str) and ":" in i else i).style.format("{:+.2f}", na_rep="–"), height=38 * (len(pv) + 2))
    sub = df[df["기간"] != "전체 OOS"]
    both = sub.groupby(["빈도", "대상", "K", "변수 묶음"])["롱숏 Sharpe"].min()
    pl = full.set_index(["빈도", "대상", "K", "변수 묶음"])["위약 95%"]
    sc = full.set_index(["빈도", "대상", "K", "변수 묶음"])["롱숏 Sharpe"]
    ok = (sc > pl) & (both.reindex(sc.index) > 0)
    g = full.groupby("변수 묶음")[["롱숏 Sharpe", "B&H Sharpe", "방향 적중률", "OOS R²"]].mean().reindex(["Y만 (단변량)", "원자재만", "Y + 거시", "원자재 + 거시"])
    st.dataframe(g.style.format({"롱숏 Sharpe": "{:+.3f}", "B&H Sharpe": "{:+.3f}", "방향 적중률": "{:.1%}", "OOS R²": "{:+.4f}"}))
    st.caption(f"셀 = 전체 OOS 롱숏 Sharpe (주간 OOS 2006–, 월간 OOS 2000–). 위약 95%를 넘고 두 하위기간(분할 2017) 모두 양수인 칸: "
               f"**{int(ok.sum())} / {len(ok)}** ({', '.join(f'{_rv_lbl(i[1])}·K{i[2]}·{i[3]}' for i in ok[ok].index[:8]) or '없음'}). "
               "아래 표 = 변수 묶음별 평균 — 거시를 넣었을 때 평균이 오르는지 확인.")


def _rv_lbl(i):
    import regime_mv as RM
    if i.startswith("wb:"):
        return RM.wb_names().get(i, i)
    return NAME.get(i, i)


# ============================================================================
# ③ 다변량 GARCH (GARCH · DCC · ADCC · BEKK)
# ============================================================================
MG_CRISES = [("2008 금융위기", "2008-09-01", "2009-03-31"), ("2020 코로나", "2020-02-20", "2020-06-30"), ("2022 전쟁·인플레", "2022-02-24", "2022-07-31")]


def _yfrac(idx) -> np.ndarray:
    """날짜 → 연도 소수 (3D 축은 숫자여야 위기 구간 판·곡면을 같은 좌표로 그릴 수 있음)."""
    idx = pd.DatetimeIndex(idx)
    return idx.year + (idx.dayofyear - 1) / 365.25


def _mg_dcc_3d(dc: dict, asym_: bool, when: pd.Timestamp | None = None):
    """변동성-공분산 탭 DCC/ADCC 3D: 시간 × 품목 쌍 × 조건부 상관 지형 + 위기 구간 판."""
    st.markdown("##### 3D 보기 — " + ("ADCC" if asym_ else "DCC") + " 상관 지형: 어느 쌍이 언제 같이 움직였나")
    P = dc["pairs"].copy()
    if len(P) > 700:                                                  # 일간이면 주간으로 줄여 곡면을 가볍게
        P = P.resample("W-FRI").last().dropna(how="all")
    nm = {c: short(c) for c in dc["cols"]}
    c1, c2 = st.columns([2, 1])
    nmax = c1.slider("보여 줄 쌍 수 (평균 상관이 큰 순)", 3, max(3, P.shape[1]), min(15, P.shape[1]), key="mg3_n")
    sort_ = c2.radio("쌍 정렬", ["평균 상관 큰 순", "위기 때 상승폭 큰 순"], key="mg3_sort")
    if sort_.startswith("평균"):
        order = P.mean().sort_values(ascending=False).index[:nmax]
    else:
        msk = np.zeros(len(P), bool)
        for _, a, b in MG_CRISES:
            msk |= (P.index >= a) & (P.index <= b)
        order = (P[msk].mean() - P[~msk].mean()).sort_values(ascending=False).index[:nmax] if msk.any() else P.mean().sort_values(ascending=False).index[:nmax]
    P = P[order]
    xs = _yfrac(P.index)
    ylab = [f"{nm[a]}–{nm[b]}" for a, b in P.columns]
    ink = "#e8e6e1" if DARK else "#0b0b0b"
    f = go.Figure(go.Surface(x=xs, y=list(range(P.shape[1])), z=P.T.values, colorscale=DIV, cmin=-1, cmax=1,
                             colorbar=dict(title="상관", thickness=12, len=0.6), contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
                             customdata=np.array([[f"{d:%Y-%m-%d}" for d in P.index]] * P.shape[1]),
                             text=np.array([[y] * len(P) for y in ylab]),
                             hovertemplate="%{customdata} · %{text}<br>상관 %{z:+.2f}<extra></extra>"))
    ny = P.shape[1] - 1
    for nm_c, a, b in MG_CRISES:
        x0, x1 = _yfrac([pd.Timestamp(a)])[0], _yfrac([pd.Timestamp(b)])[0]
        if x1 < xs.min() or x0 > xs.max():
            continue
        vx = [x0, x1, x1, x0, x0, x1, x1, x0]; vy = [0, 0, ny, ny, 0, 0, ny, ny]; vz = [-1, -1, -1, -1, 1, 1, 1, 1]
        f.add_trace(go.Mesh3d(x=vx, y=vy, z=vz, i=[0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3], j=[1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4], k=[2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7],
                              color="#e34948", opacity=0.10, name=nm_c, hoverinfo="name", showlegend=True))
        f.add_trace(go.Scatter3d(x=[(x0 + x1) / 2], y=[ny], z=[1.05], mode="text", text=[nm_c], showlegend=False, textfont=dict(size=11, color=ink), hoverinfo="skip"))
    f.update_layout(height=600, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.02, font=dict(size=10)),
                    scene=dict(xaxis=dict(title="연도"), yaxis=dict(title="", tickvals=list(range(P.shape[1])), ticktext=ylab),
                               zaxis=dict(title="조건부 상관", range=[-1, 1.1]), aspectratio=dict(x=2.3, y=1.2, z=0.7), camera=dict(eye=dict(x=0.5, y=-2.1, z=1.0))))
    if when is not None:
        k_ = int(np.abs(P.index - when).argmin())
        xw, zw = xs[k_], P.iloc[k_].values
        f.add_trace(go.Mesh3d(x=[xw] * 4, y=[0, ny, ny, 0], z=[-1, -1, 1, 1], i=[0, 0], j=[1, 2], k=[2, 3], color=PAL[3], opacity=0.16,
                              name=f"선택 시점 {P.index[k_]:%Y-%m-%d} (위 히트맵과 같은 날)", hoverinfo="name", showlegend=True))
        f.add_trace(go.Scatter3d(x=[xw] * len(zw), y=list(range(len(zw))), z=zw, mode="lines+markers", showlegend=False,
                                 line=dict(color=ink, width=6), marker=dict(size=4, color=zw, colorscale=DIV, cmin=-1, cmax=1, line=dict(color=ink, width=1)),
                                 text=ylab, hovertemplate=f"{P.index[k_]:%Y-%m-%d} · " + "%{text}<br>상관 %{z:+.2f}<extra></extra>"))
    st.plotly_chart(f, key="mg3_dcc")
    if when is not None:
        st.caption("노란 판·검은 선 = 위 '상관행렬 시점' 슬라이더로 고른 날의 단면 — 슬라이더를 옮기면 2D 히트맵과 이 단면이 함께 바뀜 (3D는 보여 준 쌍만).")
    msk = np.zeros(len(P), bool)
    for _, a, b in MG_CRISES:
        msk |= (P.index >= a) & (P.index <= b)
    if msk.any():
        up = (P[msk].mean() - P[~msk].mean())
        st.caption(f"빨간 판 = 위기 구간(2008-09~2009-03 · 2020-02~06 · 2022-02~07). 위기 때 평균 상관 {P[msk].values.mean():+.2f} vs 평상시 {P[~msk].values.mean():+.2f} "
                   f"(보여 준 {P.shape[1]}쌍 중 {int((up > 0).sum())}쌍이 위기 때 상승, 최대 {ylab[int(np.argmax(up.values))]} {up.max():+.2f}). "
                   "판 안에서 곡면이 전체적으로 솟으면 위기 때 동조화(분산 효과 감소). 마우스로 돌려 보기.")


def _mg_bekk_net(bk: dict, title: str, key: str, nm: dict, h: int = 520):
    """BEKK 전이 네트워크 3D: 아래층 = 충격 전이 A, 위층 = 변동성 전이 B. 선 굵기 = |계수|, 색 = 유의(|t| ≥ 1.96) 여부, 원뿔 = 방향(행 → 열)."""
    cols = bk["cols"]; n = len(cols)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    px_, py_ = np.cos(ang), np.sin(ang)
    ink = "#e8e6e1" if DARK else "#0b0b0b"
    f = go.Figure()
    allv = np.abs(np.concatenate([bk["A"].values[~np.eye(n, dtype=bool)], bk["B"].values[~np.eye(n, dtype=bool)]])) if n > 1 else np.array([1.0])
    vmax = float(allv.max()) or 1.0
    for z, Mx, Mt, lab, col in [(0.0, bk["A"], bk["A_t"], "충격 전이 A (아래층)", PAL[1]), (1.0, bk["B"], bk["B_t"], "변동성 전이 B (위층)", PAL[0])]:
        f.add_trace(go.Scatter3d(x=px_, y=py_, z=[z] * n, mode="markers+text", text=[nm[c] for c in cols], textposition="top center", name=lab,
                                 marker=dict(size=[8 + 30 * abs(Mx.iloc[i, i]) / (abs(np.diag(Mx.values)).max() or 1) for i in range(n)], color=col,
                                             line=dict(color=ink, width=1)), textfont=dict(size=12),
                                 hovertemplate="%{text}<br>자기 계수 " + "<extra></extra>"))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                v, t = float(Mx.iloc[i, j]), float(Mt.iloc[i, j])
                if not np.isfinite(v) or abs(v) < 1e-6:
                    continue
                sig = np.isfinite(t) and abs(t) >= 1.96
                c_ = ("#e34948" if v > 0 else "#2a78d6") if sig else "#898781"
                x0, y0, x1, y1 = px_[i], py_[i], px_[j], py_[j]
                off = 0.06 * (1 if i < j else -1)                       # 양방향 화살표가 겹치지 않게 살짝 비킴
                dx, dy = -(y1 - y0) * off, (x1 - x0) * off
                f.add_trace(go.Scatter3d(x=[x0 + dx, x1 + dx], y=[y0 + dy, y1 + dy], z=[z, z], mode="lines", showlegend=False,
                                         line=dict(color=c_, width=1.5 + 12 * abs(v) / vmax), opacity=1.0 if sig else 0.45,
                                         hovertemplate=f"{nm[cols[i]]} → {nm[cols[j]]}<br>{lab.split(' (')[0]} {v:+.3f} (t {t:.1f})<extra></extra>"))
                f.add_trace(go.Cone(x=[x0 + dx + 0.82 * (x1 - x0)], y=[y0 + dy + 0.82 * (y1 - y0)], z=[z], u=[x1 - x0], v=[y1 - y0], w=[0],
                                    sizemode="absolute", sizeref=0.12, anchor="tip", showscale=False, colorscale=[[0, c_], [1, c_]], hoverinfo="skip"))
    for i in range(n):
        f.add_trace(go.Scatter3d(x=[px_[i], px_[i]], y=[py_[i], py_[i]], z=[0, 1], mode="lines", line=dict(color="#898781", width=1, dash="dot"),
                                 showlegend=False, hoverinfo="skip"))
    f.update_layout(title=dict(text=title, font=dict(size=13)), height=h, margin=dict(l=0, r=0, t=36, b=0), legend=dict(orientation="h", y=-0.02, font=dict(size=10)),
                    scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectratio=dict(x=1, y=1, z=0.8),
                               camera=dict(eye=dict(x=1.3, y=-1.3, z=0.9))))
    st.plotly_chart(f, key=key)


def _mg_bekk_3d(bk: dict, R: pd.DataFrame, bkind: str, ann: int):
    import mvgarch as M
    st.markdown("##### 3D 보기 — BEKK 전이 네트워크: 누가 누구에게 충격·변동성을 옮기나")
    if bkind != "Full":
        st.caption("Diagonal·Scalar BEKK는 품목 사이 전이를 0으로 고정해 네트워크 선이 없음 — 유형을 Full로 추정하면 화살표가 그려짐.")
    nm = {c: short(c) for c in bk["cols"]}
    _mg_bekk_net(bk, "전체 기간", "mg3_bekk", nm)
    st.caption("아래층 = 충격 전이(A, 지난주 뜻밖의 가격 변화가 다른 품목 분산을 키움), 위층 = 변동성 전이(B, 높은 변동성이 이어 옮겨 감). "
               "화살표 = 행 품목 → 열 품목, 굵기 = |계수|, 빨강 = 양(+)·파랑 = 음(−) 이고 유의(|t| ≥ 1.96), 회색 = 유의하지 않음. 점 크기 = 자기 자신에게 남는 정도(대각 계수).")
    if bkind == "Full":
        cut = pd.Timestamp("2017-01-01")
        r1, r2 = R[bk["cols"]].loc[:cut - pd.Timedelta(days=1)].dropna(), R[bk["cols"]].loc[cut:].dropna()
        lab1 = f"{r1.index.min():%Y-%m}–2016" if len(r1) else "–"
        if st.button(f"시기별 비교: {lab1} / 2017– 두 구간으로 다시 추정 (Full, 약 1분)", key="mg3_split_btn"):
            if min(len(r1), len(r2)) < 150:
                st.warning("한 구간의 관측치가 150개 미만이라 추정이 불안정합니다 — 주간 빈도·시작 연도를 확인하세요.")
            else:
                with st.spinner("두 구간 BEKK 추정 중…"):
                    try:
                        st.session_state["mg_b_split"] = (tuple(bk["cols"]), M.bekk_fit(r1, "Full", ann), M.bekk_fit(r2, "Full", ann), lab1)
                    except Exception as ex:
                        st.error(f"구간 추정 실패: {ex}")
        sp_ = st.session_state.get("mg_b_split")
        if sp_ and sp_[0] == tuple(bk["cols"]):
            q1, q2 = st.columns(2)
            with q1:
                _mg_bekk_net(sp_[1], f"{sp_[3]} (수렴 {'✔' if sp_[1]['converged'] else '✖'})", "mg3_bekk_a", nm, 460)
            with q2:
                _mg_bekk_net(sp_[2], f"2017– (수렴 {'✔' if sp_[2]['converged'] else '✖'})", "mg3_bekk_b", nm, 460)
            nA1 = int((np.abs(sp_[1]["A_t"].values[~np.eye(len(nm), dtype=bool)]) >= 1.96).sum())
            nA2 = int((np.abs(sp_[2]["A_t"].values[~np.eye(len(nm), dtype=bool)]) >= 1.96).sum())
            nB1 = int((np.abs(sp_[1]["B_t"].values[~np.eye(len(nm), dtype=bool)]) >= 1.96).sum())
            nB2 = int((np.abs(sp_[2]["B_t"].values[~np.eye(len(nm), dtype=bool)]) >= 1.96).sum())
            st.caption(f"유의한 전이 화살표 수: 충격 A {nA1} → {nA2}, 변동성 B {nB1} → {nB2} (가능한 {len(nm) * (len(nm) - 1)}개 중). "
                       "화살표가 새로 생기거나 사라지면 위험이 옮겨 가는 길이 바뀐 것 — 단 구간이 짧으면 계수가 불안정하니 수렴 표시와 t값을 함께 볼 것.")


def _mg_hmm_overlay(dc: dict):
    """DCC 평균 상관(동조화 지표)과 레짐 탭 다변량 HMM 국면을 같은 시간축에 — 두 방법이 같은 위기 신호를 주는지 교차 확인."""
    import regime_mv as RM
    ss = st.session_state
    opts = [i for i in COMMOD_IDS if not i.startswith("etf:")]
    comm = tuple(ss.get("rv_comm_W") or [i for i in RM.COMM_W_DEFAULT if i in opts])
    mopts = [i for i in RM.MACRO if i == "spread:10y2y" or i in NAME]
    macro = tuple(ss.get("rv_macro") or [i for i in RM.MACRO_DEFAULT if i in mopts])
    tgt = ss.get("rv_tgt_W") or (comm[0] if comm else None)
    K = int(ss.get("rv_k", 3)); cov = ss.get("rv_cov", "full")
    if not comm or tgt not in comm:
        st.info("레짐 탭에서 원자재를 먼저 고르세요."); return
    try:
        d = _rv_desc(comm, macro, "W", ROLL, tgt, K, cov)
    except Exception as ex:
        st.warning(f"HMM 추정 실패: {ex}"); return
    G = d["smooth"]; Fp = d["filt"]
    ac = dc["avg_corr"].copy()
    if len(ac) > 700:
        ac = ac.resample("W-FRI").last()
    lab = pd.Series(G.values.argmax(1), index=G.index)
    f = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.06,
                      subplot_titles=("DCC 평균 조건부 상관 — 배경 = 레짐 탭 HMM의 가장 가능성 높은 국면", "HMM 국면 확률 (실시간 필터)"))
    f.add_trace(go.Scatter(x=ac.index, y=ac.values, line=dict(color=PAL[6], width=2), name="평균 DCC 상관"), 1, 1)
    b0 = 0
    for t in range(1, len(lab) + 1):
        if t == len(lab) or lab.iloc[t] != lab.iloc[b0]:
            f.add_vrect(x0=lab.index[b0], x1=lab.index[min(t, len(lab) - 1)], fillcolor=RV_COL[lab.iloc[b0] % 4], opacity=0.22, line_width=0, layer="below", row=1, col=1)
            b0 = t
    for j, c in enumerate(Fp.columns):
        f.add_trace(go.Scatter(x=Fp.index, y=Fp[c], line=dict(width=1.2, color=RV_COL[j % 4]), name=f"{c}: {d['names'][c]}"), 2, 1)
    st.plotly_chart(fig_layout(f, 480, hovermode="x", legend=dict(orientation="h", y=-0.15, font=dict(size=10))), key="mg_hmm_ov")
    j_ = pd.concat([ac.rename("상관"), lab.reindex(ac.index, method="ffill").rename("국면")], axis=1).dropna()
    if len(j_):
        tb = j_.groupby("국면")["상관"].agg(["mean", "count"])
        txt = " · ".join(f"{G.columns[int(k)]} ({d['names'][G.columns[int(k)]]}) {r['mean']:+.2f} ({int(r['count'])}주)" for k, r in tb.iterrows())
        jump = ac.diff(4)
        top = jump.nlargest(8)
        hit = np.mean([lab.reindex([x], method="ffill").iloc[0] == tb["mean"].idxmax() for x in top.index]) if len(top) else np.nan
        st.caption(f"국면별 평균 DCC 상관: {txt}. 4주간 상관이 가장 크게 뛴 8번 중 {hit:.0%}가 상관이 가장 높은 국면 안 — 높을수록 두 방법이 같은 위기 신호. "
                   f"HMM 설정은 레짐 탭에서 고른 값(원자재 {len(comm)}개 · 거시 {len(macro)}개 · 대상 {short(tgt)} · {K}국면)을 그대로 씀, 주간 빈도.")


def _spill_proxy(R: pd.DataFrame) -> pd.DataFrame:
    """BEKK 조합 고르기용 빠른 근사: i → j 충격 전이 t값 = (표준화 수익률²_j,t) ~ 상수 + 자기 지연² + (i의 지연²) 회귀에서 i 계수의 t.
    BEKK A의 비대각(지난 충격이 다른 품목 분산을 키움)과 같은 방향의 신호 — 추정이 수천 배 빠름."""
    X = ((R - R.mean()) / R.std()).dropna()
    E2 = (X ** 2).values
    n, k = E2.shape
    T = np.full((k, k), np.nan)
    for j in range(k):
        y = E2[1:, j]
        for i in range(k):
            if i == j:
                continue
            Z = np.column_stack([np.ones(n - 1), E2[:-1, j], E2[:-1, i]])
            b, *_ = np.linalg.lstsq(Z, y, rcond=None)
            e = y - Z @ b
            cov = np.linalg.inv(Z.T @ Z) * (e @ e) / (len(y) - 3)
            T[i, j] = b[2] / np.sqrt(cov[2, 2])
    return pd.DataFrame(T, index=R.columns, columns=R.columns)


def _mg_bekk_search(R: pd.DataFrame, ids: list, ann: int):
    """Full BEKK로 추정할 품목 조합(3–5개)을 자동 추천: ① 근사 전이 점수로 모든 조합 순위 → ② 상위 몇 개만 실제 Full BEKK로 확인 → 고른 조합을 아래 품목 칸에 채움."""
    import itertools
    import mvgarch as M
    with st.expander("🔎 전이가 가장 많은 품목 조합 자동 찾기 (Full BEKK, 3–5개)", expanded=False):
        cand = [c for c in ids if c in R]
        if len(cand) < 3:
            st.info("위 '분석 품목'에서 3개 이상 고르세요."); return
        per = st.radio("찾을 기간", ["전체", "2016년까지", "2017년 이후"], horizontal=True, key="mg_bs_per",
                       help="시기별로 따로 찾으면 '전이가 많은 품목 묶음' 자체가 바뀌었는지 볼 수 있음 (각 구간 관측치가 충분해야 안정적)")
        R_all = R
        R = R.loc[:"2016-12-31"] if per == "2016년까지" else (R.loc["2017-01-01":] if per == "2017년 이후" else R)
        if len(R.dropna()) < 150:
            st.warning(f"선택한 기간의 관측치가 {len(R.dropna())}개뿐이라 추정이 불안정합니다."); return
        s1, s2, s3 = st.columns([1, 1, 2])
        kk = s1.radio("조합 크기", [3, 4, 5], horizontal=True, key="mg_bs_k", help="Full BEKK 모수 = 2N² + N(N+1)/2 → 3개 24 · 4개 42 · 5개 65개. 5개는 추정이 느리고 불안정할 수 있음")
        kk = min(kk, len(cand))
        nver = s2.number_input("BEKK로 확인할 상위 조합 수", 1, 6, 3, key="mg_bs_n", help="조합당 3개 ≈ 30초, 4개 ≈ 1–2분, 5개 ≈ 3분 이상 (주간 기준)")
        T = _spill_proxy(R[cand])
        rows = []
        for cb in itertools.combinations(cand, kk):
            sub = T.loc[list(cb), list(cb)].values
            off = sub[~np.eye(kk, dtype=bool)]
            sig = int((np.abs(off) >= 1.96).sum())
            edges = sorted(((abs(T.loc[a, b]), a, b) for a in cb for b in cb if a != b), reverse=True)[:3]
            rows.append({"조합": " · ".join(short(c) for c in cb), "근사 유의 전이 수": sig, "근사 |t| 합": float(np.nansum(np.abs(off))),
                         "주요 전이 (근사 t)": ", ".join(f"{short(a)}→{short(b)} {T.loc[a, b]:+.1f}" for _, a, b in edges), "_ids": cb})
        S = pd.DataFrame(rows).sort_values(["근사 유의 전이 수", "근사 |t| 합"], ascending=False).reset_index(drop=True)
        s3.caption(f"조합 {len(S)}개를 근사 점수로 정렬 — 근사 = 각 품목의 표준화 수익률²를 다른 품목의 지난주 수익률²로 설명하는 회귀 t값(BEKK의 충격 전이 A와 같은 방향). "
                   "근사로 후보를 좁힌 뒤 실제 Full BEKK로 확인.")
        st.dataframe(S.drop(columns="_ids").head(10).style.format({"근사 |t| 합": "{:.1f}"}), hide_index=True, use_container_width=True)
        if st.button(f"상위 {int(nver)}개 조합을 Full BEKK로 확인", key="mg_bs_run"):
            out = []
            prog = st.progress(0.0)
            for q, cb in enumerate(S["_ids"].head(int(nver))):
                try:
                    bk = M.bekk_fit(R[list(cb)].dropna(), "Full", ann)
                    m_ = ~np.eye(len(cb), dtype=bool)
                    nA, nB = int((np.abs(bk["A_t"].values[m_]) >= 1.96).sum()), int((np.abs(bk["B_t"].values[m_]) >= 1.96).sum())
                    out.append({"근사 순위": q + 1, "조합": " · ".join(short(c) for c in cb), "유의 충격 전이 A": nA, "유의 변동성 전이 B": nB, "합계": nA + nB,
                                "가능한 화살표": 2 * len(cb) * (len(cb) - 1), "수렴": "✔" if bk["converged"] else "✖", "BIC": bk["bic"], "_ids": cb})
                except Exception as ex:
                    out.append({"근사 순위": q + 1, "조합": " · ".join(short(c) for c in cb), "유의 충격 전이 A": np.nan, "유의 변동성 전이 B": np.nan, "합계": np.nan,
                                "가능한 화살표": np.nan, "수렴": f"실패: {ex}"[:40], "BIC": np.nan, "_ids": cb})
                prog.progress((q + 1) / int(nver))
            O_ = pd.DataFrame(out).sort_values(["합계"], ascending=False, na_position="last").reset_index(drop=True)
            st.session_state["mg_bs_out"] = O_
            st.session_state["mg_bs_meta"] = (per, kk, f"{R.index.min():%Y-%m}~{R.index.max():%Y-%m}")
            try:
                import runlog as RL
                ok_r = O_[O_["수렴"] == "✔"]
                rid = RL.record("bekk_search", f"BEKK 조합 자동 찾기 ({per}, {kk}개)",
                                dict(기간=per, 시작=f"{R.index.min():%Y-%m-%d}", 끝=f"{R.index.max():%Y-%m-%d}", 조합크기=int(kk), 후보품목=[short(c) for c in cand],
                                     확인수=int(nver), 빈도=str(R.index.freq or pd.infer_freq(R.index[:50]) or "")),
                                summary={"추천 유의 전이 수": float(ok_r["합계"].iloc[0]) if len(ok_r) else float("nan"), "후보 조합 수": len(S)},
                                tables={"근사_순위": S.drop(columns="_ids").head(30), "BEKK_확인": O_.drop(columns="_ids")})
                st.caption(f"실행 기록 저장: `runs/…/{rid}` (⑥ 실험 결과 탭 실행 기록에서 비교)")
            except Exception as ex:
                st.caption(f"실행 기록 저장 실패: {ex}")
        O = st.session_state.get("mg_bs_out")
        if O is not None and len(O):
            st.dataframe(O.drop(columns="_ids").style.format({"BIC": "{:.1f}", "유의 충격 전이 A": "{:.0f}", "유의 변동성 전이 B": "{:.0f}", "합계": "{:.0f}",
                                                             "가능한 화살표": "{:.0f}"}, na_rep="–"), hide_index=True, use_container_width=True)
            ok_ = O[O["수렴"] == "✔"]
            best = (ok_ if len(ok_) else O).iloc[0]
            meta_ = st.session_state.get("mg_bs_meta")
            vv = O.dropna(subset=["합계"])
            if len(vv) >= 3:
                rho = float(vv["근사 순위"].rank().corr(-vv["합계"].rank(), method="spearman"))
                st.caption(f"근사 순위 ↔ 실제 BEKK 유의 전이 수 순위 상관(Spearman) {rho:+.2f} ({len(vv)}개 조합) — +1에 가까울수록 근사를 믿고 후보를 좁혀도 됨, 0 근처면 근사가 실제와 따로 놂 → 확인 조합 수를 늘릴 것."
                           + (f" 기간 {meta_[0]} ({meta_[2]}), {meta_[1]}개 조합." if meta_ else ""))
            elif len(vv):
                st.caption("근사와 실제의 순위 상관은 확인 조합이 3개 이상일 때 계산." + (f" 기간 {meta_[0]} ({meta_[2]})." if meta_ else ""))
            st.caption(f"추천 = **{best['조합']}** (유의 전이 {best['합계']:.0f}개, 수렴한 조합 중 최다). 유의 전이가 많다 = 품목 사이에 위험이 옮겨 가는 길이 많아 네트워크로 볼 거리가 많음 — "
                       "여러 조합 중 고른 것이라 t ≥ 1.96 몇 개는 우연일 수 있음.")

            def _apply(c_=tuple(best["_ids"])):
                st.session_state["mg_bids"] = list(c_)
                st.session_state["mg_bkind"] = "Full"
            st.button(f"추천 조합({best['조합']})을 아래 BEKK 품목에 넣기", key="mg_bs_apply", on_click=_apply)
        st.markdown("---")
        if st.button(f"세 기간 한 번에 찾기 — 전체 · 2016년까지 · 2017년 이후 (기간마다 근사 상위 {int(nver)}개를 Full BEKK로 확인, {kk}개 조합)", key="mg_bs_all_btn"):
            res_ = []
            prog2 = st.progress(0.0)
            pers = ["전체", "2016년까지", "2017년 이후"]
            for pi_, per_ in enumerate(pers):
                Rp = R_all.loc[:"2016-12-31"] if per_ == "2016년까지" else (R_all.loc["2017-01-01":] if per_ == "2017년 이후" else R_all)
                Rp = Rp[cand].dropna()
                row = {"기간": per_, "자료": f"{Rp.index.min():%Y-%m}~{Rp.index.max():%Y-%m} ({len(Rp)}개)" if len(Rp) else "–"}
                if len(Rp) < 150:
                    res_.append({**row, "추천 조합": "관측치 부족"}); continue
                Tp = _spill_proxy(Rp)
                rk = []
                for cb in itertools.combinations(cand, kk):
                    off = Tp.loc[list(cb), list(cb)].values[~np.eye(kk, dtype=bool)]
                    rk.append(((np.abs(off) >= 1.96).sum(), float(np.nansum(np.abs(off))), cb))
                rk.sort(key=lambda x: (x[0], x[1]), reverse=True)
                fits = []
                for _, _, cb in rk[:int(nver)]:
                    try:
                        bk_ = M.bekk_fit(Rp[list(cb)], "Full", ann)
                        m_ = ~np.eye(kk, dtype=bool)
                        fits.append((int((np.abs(bk_["A_t"].values[m_]) >= 1.96).sum()), int((np.abs(bk_["B_t"].values[m_]) >= 1.96).sum()), bool(bk_["converged"]), cb))
                    except Exception:
                        continue
                fits.sort(key=lambda x: (x[2], x[0] + x[1]), reverse=True)
                if fits:
                    a1, b1, cv1, cb1 = fits[0]
                    row.update({"추천 조합": " · ".join(short(c) for c in cb1), "유의 충격 전이 A": a1, "유의 변동성 전이 B": b1, "합계": a1 + b1,
                                "수렴": "✔" if cv1 else "✖", "2위": (" · ".join(short(c) for c in fits[1][3]) + f" ({fits[1][0] + fits[1][1]})") if len(fits) > 1 else "–",
                                "_ids": cb1})
                else:
                    row["추천 조합"] = "추정 실패"
                res_.append(row)
                prog2.progress((pi_ + 1) / len(pers))
            st.session_state["mg_bs_all"] = (kk, pd.DataFrame(res_))
        AL = st.session_state.get("mg_bs_all")
        if AL is not None:
            kk_, A_ = AL
            st.markdown(f"**기간별 추천 조합 ({kk_}개)**")
            st.dataframe(A_.drop(columns=[c for c in ["_ids"] if c in A_]), hide_index=True, use_container_width=True)
            sets_ = [set(x) for x in A_.get("_ids", pd.Series(dtype=object)).dropna()]
            if len(sets_) >= 2:
                common_ = set.intersection(*sets_)
                st.caption("세 기간 공통 품목: " + (", ".join(short(c) for c in common_) if common_ else "없음") +
                           " — 기간마다 추천 묶음이 다르면 위험이 옮겨 다니는 품목 구성이 시기에 따라 바뀐 것. 각 구간은 표본이 짧아 계수가 불안정할 수 있음(수렴 표시 확인).")


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
        st.plotly_chart(heat(Rm, f"DCC 상관행렬 — {pick}  (평균 {dc['avg_corr'].iloc[ti]:.3f})", *range_of(Rm), h=420, nodiag=True), key="mg_dheat")
        _mg_dcc_3d(dc, asym_, pd.Timestamp(pick))
        with st.expander("DCC 평균 상관 × 레짐 탭 HMM 국면 — 같은 위기 신호를 주나 (교차 확인)", expanded=False):
            _mg_hmm_overlay(dc)

    # ---- ③ BEKK ----
    st.markdown("##### ③ BEKK(1,1) — 충격·변동성 전이")
    _mg_bekk_search(R, ids, ANN)
    st.caption("Hₜ = CC' + A'εₜ₋₁εₜ₋₁'A + B'Hₜ₋₁B.  A[i,j] (i≠j): i 시장의 충격 → j 시장 분산, B[i,j]: i 시장의 변동성 → j 시장 변동성. "
               "Full BEKK는 모수가 2N²+N(N+1)/2 개로 많아 2~4개 품목 권장.")
    b1_, b2_, b3_ = st.columns([2, 1, 1])
    if st.session_state.get("mg_bids") and not set(st.session_state["mg_bids"]) <= set(ids):          # 분석 품목이 바뀌어 추천 조합이 목록에 없으면 초기화
        st.session_state["mg_bids"] = [c for c in st.session_state["mg_bids"] if c in ids]
    bids = b1_.multiselect("BEKK 품목 (2~4개 권장)", ids, default=None if "mg_bids" in st.session_state else ids[:3], format_func=short, key="mg_bids")
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
        _mg_bekk_3d(bk, R, bkind, ANN)
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


def _revalidate_ml(cfg: dict, model: str, targets: list[str], progress=None) -> pd.DataFrame:
    """기록된 ④ 설정을 그대로(같은 설명변수·특성·시차·비용·무거래 구간·walk-forward) 다른 품목에 적용 + 기간 반분.
    위약 기준 = 원래 실행에서 시도한 모델 수만큼의 '최댓값' 위약 (그 모델을 여러 개 중에서 골랐으므로)."""
    fq = cfg.get("빈도", FQ); ann = A.ANN[fq]
    Lv = _levels(None, bool(cfg.get("롤보정", True)))
    n_try = max(1, len(cfg.get("모델", [model])))
    hp_ = dict(cfg.get("hp", {})); hp_["mode"] = "predict"
    rows = []
    for j, t in enumerate(targets):
        if progress:
            progress(j / len(targets), f"재검증: {short(t)}")
        feats_ = [f for f in cfg["설명변수"] if f != t and f in Lv]
        fk_ = [k_ for k_ in cfg.get("추가특성", []) if not (k_ == "COT 헤징압력" and t not in A.COT_MAP)
               and not (k_ == "베이시스(캐리)" and t not in A.BASIS_MAP)]
        try:
            Rt = A.make_panel(Lv, [t] + feats_, UNITS, fq, "logret", cfg.get("시작"), cfg.get("끝"))
            F = A.make_features(Lv, t, UNITS, fq, Rt.index, fk_) if fk_ else None
            vsw = {"B": 21, "W-FRI": 13, "ME": 6}[fq] if cfg.get("변동성표준화") else None
            D = A.build_design(Rt, t, feats_, "predict", int(cfg["시차"]), int(cfg["예측기간"]), bool(cfg["자기시차"]), float(cfg["학습비중"]),
                               extra=F, vol_scale=vsw, xlag=int(cfg.get("비동기보정", 0)))
            D.cost = float(cfg["비용bp"]) / 100.0; D.band = float(cfg.get("무거래c", 0.0))
            p_, y_, m_ = A.walk_forward(D, model, ann, int(cfg["재추정주기"]), 0.5, **hp_)
        except Exception as ex:
            rows.append({"품목": short(t), "id": t, "오류": str(ex)[:80]}); continue
        h = D.horizon
        pos = A.band_positions(p_.iloc[::h].values, D.band); yv = y_.iloc[::h].values
        net = pos * yv - np.abs(np.diff(np.r_[0, pos])) * D.cost
        sr = lambda x: float(np.mean(x) / np.std(x) * np.sqrt(ann / h)) if np.std(x) > 0 else np.nan
        half = len(net) // 2
        pl = A.placebo_sharpe(y_.values, ann, h, D.cost, n_models=n_try, long_share=m_["롱 비중"], switch=m_["연간 회전율"] * h / ann / 2, B=500)
        rows.append({"품목": short(t), "id": t, "원래 품목": t == cfg["대상"], "Sharpe (비용 전)": m_["부호전략 Sharpe"], "Sharpe (비용 후)": sr(net),
                     "전반 (비용 후)": sr(net[:half]), "후반 (비용 후)": sr(net[half:]), "적중률": m_["방향 적중률"], "Buy&Hold": m_["Buy&Hold Sharpe"],
                     "위약 단일 95%": pl["single95_net"], f"위약 최댓값 95% ({n_try}개)": pl["max95_net"],
                     "운 이상": sr(net) > pl["max95_net"], "두 기간 +": (sr(net[:half]) > 0) and (sr(net[half:]) > 0),
                     "평가 기간": f"{y_.index[0]:%Y-%m}~{y_.index[-1]:%Y-%m}"})
    if progress:
        progress(1.0, "완료")
    return pd.DataFrame(rows).set_index("품목")


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
    ss["models"] = [m for m in cfg["모델"] if m in A.PREDICT_MODELS]
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
                            "yf:DX-Y.NYB", "fred:VIXCLS", "fred:DGS10", "fred:DTB3", "fred:BAA10Y", "yf:^GSPC"] if i in NAME and i != tgt]
    feats = a2.multiselect("설명변수 (X)", [i for i in ALL_IDS if i != tgt], default=feat_def, format_func=lbl, key="feats")
    _mac = [i for i in feats if GROUP.get(i) in ("Macro", "FX", "FRED daily")]
    a2.caption(f"선택 {len(feats)}개 = 원자재·운임 {len(feats) - len(_mac)} · 거시 {len(_mac)} ({', '.join(short(i) for i in _mac) or '없음'}) — 칩이 많으면 옆으로 밀려 안 보일 수 있음")
    mm_pick = _mm_picker(tgt, "ml_mm")
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
    models = st.multiselect(f"모델 선택 (전체 {len(A.PREDICT_MODELS)}개 · 기본 = 모형군별 대표 {len(A.CORE_MODELS)}개)", A.CORE_MODELS + A.ADV_MODELS,
                            default=A.CORE_MODELS, key="models",
                            help="기본: 기준 OLS · 축소 Ridge · 변수 선택 Lasso · 결합 Elastic Net · 요인 PLS · 부스팅 GB·LightGBM. "
                                 "추가로 벌점 회귀 변형(Adaptive Lasso·SCAD·MCP) · 밀집 요인 · 베이즈 · 트리 · 커널 · LSTM 선택 가능 "
                                 "(레짐·시변 모델은 ② 레짐 탭). 모델을 많이 돌릴수록 '그중 최고'는 운만으로도 높아짐 → 성과 표의 '위약 · 모델 N개 중 최댓값'과 비교")
    import predict_lab as PL
    alt_m = st.multiselect("🔮 대안 시계열 모델 (예측 모드 전용 · 같은 학습/테스트 분할)", list(PL.ALT_SINGLE), default=[], key="alt_models",
                           help="Prophet = 추세 변곡 + 연간 계절 + 설명변수 선형 · neuralforecast PatchTST(자기 시계열만) · iTransformer(설명변수를 토큰으로) · TFT(설명변수 = 과거 변수) · "
                                "Chronos-Bolt·TimesFM = zero-shot(자기 가격만) · Chronos-2 = zero-shot + 설명변수 공변량. 학습 구간으로 한 번 학습/적합하고 테스트 구간 각 시점까지 자료로 예측. "
                                "CPU라 모델당 수십 초~수 분, 테스트 시점이 400개를 넘으면 일정 간격으로만 예측하고 사이는 직전 값 유지 (일간 자료는 주간 권장).")
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
        if not (models or alt_m) or (not feats and not fk and not mm_pick and not (md == "predict" and own)):
            st.warning("모델과 설명변수(또는 추가 특성)를 하나 이상 선택하세요."); return
        F = A.make_features(_levels(extra_bytes, ROLL), tgt, UNITS, FQ, R3.index, fk) if fk else None
        if mm_pick:
            MMF = PL.mm_item_frame(_mm_F(), tgt, mm_pick, R3.index)
            F = MMF if F is None else F.join(MMF)
        vsw = {"B": 21, "W-FRI": 13, "ME": 6}[FQ] if vs else None
        D = A.build_design(R3, tgt, feats, md, int(lags), int(hor), own, trf, extra=F, vol_scale=vsw, xlag=xlag)
        D.cost = cost_bp / 100.0     # bp -> % (수익률이 % 단위)
        D.band = float(band_c)
        out, wf_zs, prog = {}, {}, st.progress(0.0, "모델 추정 중")
        allm = list(models) + list(alt_m)
        for j, mname in enumerate(allm):
            prog.progress(j / len(allm), f"{mname} 추정 중...")
            try:
                if mname in PL.ALT_SINGLE:
                    if md != "predict":
                        st.warning(f"{mname}: 대안 시계열 모델은 '예측' 모드에서만 실행됩니다."); continue
                    sc_ = R3[tgt].rolling(vsw).std() if vsw else None
                    if wf and mname in PL.ZERO_SHOT:     # 학습 없는 모델: walk-forward 구간을 먼저 예측 → 단일 분할은 같은 날짜 값을 캐시에서 가져옴
                        wf_zs[mname] = PL.alt_walk_forward(mname, D, ANN, R3[tgt], sc_)
                    out[mname] = PL.run_alt_single(mname, D, ANN, R3[tgt], sc_)
                else:
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
            wm = [m for m in models if m not in ("LSTM", "분위수 회귀")]          # 학습형 대안 모델·LSTM은 재학습 비용이 커서 제외, zero-shot은 위에서 계산
            prog = st.progress(0.0, "Walk-forward 검증 중")
            for j, mname in enumerate(wm):
                prog.progress(j / max(1, len(wm)), f"Walk-forward: {mname}")
                try:
                    wfres[mname] = A.walk_forward(D, mname, ANN, int(wf_step), 0.5, **hp)
                except Exception as ex:
                    st.error(f"Walk-forward {mname} 실패: {ex}")
            prog.empty()
            wfres.update({k: v for k, v in wf_zs.items() if k in out})
            if "LSTM" in models or set(alt_m) - set(PL.ZERO_SHOT):
                st.info("LSTM·학습형 대안 시계열 모델(Prophet·PatchTST·iTransformer·TFT)은 재학습할 때마다 가중치가 달라져 매번 다시 학습해야 하므로 "
                        "walk-forward에서 제외했습니다. zero-shot 모델(Chronos·TimesFM)은 학습이 없어 재학습 없이 같은 예측을 walk-forward에 그대로 씁니다.")
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
                   모델=list(models) + list(alt_m), 추가특성=list(fk), 멀티모달=list(mm_pick), 변동성표준화=bool(vs), walk_forward=bool(wf and md == "predict"), 재추정주기=int(wf_step),
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
                                       colorscale=DIV, texttemplate="%{z:.2f}" if CM.shape[1] <= 30 else None,
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
    cands = m1.multiselect("후보 모델", [m for m in A.PREDICT_MODELS if m not in ("LSTM", "분위수 회귀")], MS.CANDIDATES, key="ms_models")
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
    sig = c1_.radio("기대수익 신호 μ", ["시계열 모멘텀 (기간 선택)", "과거 평균 (무정보 기준)"], key="pf_sig",
                    help="모멘텀: α = IC × σ × score(선택 기간 수익/변동성, ±2 제한) — 롱숏 전략 탭에서 유일하게 견고했던 신호. "
                         "과거 평균: 확장창 평균 (예측 없이 위험 분산만). Ridge·패널 예측 신호는 ⑥ 검증에서 과거 평균을 못 이겨 v1.14.0에서 제외")
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
def _panel_data(basis: str, uni: tuple, freq: str, target: str, extra_bytes, roll: bool, macro: tuple = (), mm: tuple = ()):
    import panel as PN
    import predict_lab as PL
    return PN.build_panel(_levels(extra_bytes, True), _levels(extra_bytes, False), list(uni), UNITS, freq, basis, target,
                          macro, {m: short(m) for m in macro}, mm=PL.mm_panel(_mm_F(), list(mm)) if mm else None)


@st.cache_data(show_spinner=False)
def _alt_panel_preds() -> dict:
    """여러 품목 탭용 대안 시계열 예측 (19개 품목 · 주간 · 4주 앞, 미리 계산): ⑤ 대안 시계열 탭(2012-07~) + ④ 백테스트 그리드(2007~)."""
    import experiments as EX
    out = {}
    for fn, tag in (("predict_grid_alt_preds.pkl", "2007~"), ("alt_ts_preds.pkl", "2012-07~"), ("alt_ts_macro_preds.pkl", "2012-07~")):
        p = EX.DATA / fn
        if not p.exists():
            continue
        for k, v in pd.read_pickle(p).items():
            nm = (f"{k[0]} · {k[1]}" if isinstance(k, tuple) else str(k)).replace(" · 시드 0", "").replace(" · 시드 1", "")
            if " · 시드 " in nm:
                continue
            out.setdefault(f"{nm} ({tag})", v)
    return out


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
    chars = st.multiselect("특성 (품목별 가격·포지션 정보)", PN.CHARS, default=PN.CHARS, key="pn_chars")
    st.markdown("**🌐 거시 변수 추가** — 품목별 거시 민감도(과거 1년 β) × 이번 기간 거시 변화를 특성 '거시 β×Δ: …'로 넣음")
    mac_opts = [i for i in ALL_IDS if GROUP.get(i) in ("Macro", "FX") or i == "ak:macro_shipping_bdi"]
    pmac = st.multiselect("거시 변수 (복수 선택, 비우면 거시 미반영)", mac_opts,
                           [i for i in ["yf:DX-Y.NYB", "fred:VIXCLS", "fred:DGS10", "fred:BAA10Y"] if i in mac_opts], format_func=lbl, key="pn_macro",
                           help="거시 변수는 모든 품목에 같은 값이라 '어느 품목이 상대적으로 오를까'를 직접 설명하지 못함 → "
                                "품목별 과거 1년 민감도(β) × 이번 기간 거시 변화를 특성으로 사용 (예: 달러가 오른 주에 달러 민감도가 큰 품목은 불리)")
    chars = chars + [PN.macro_char_name(m, short(m)) for m in pmac]
    try:
        import predict_lab as PL
        mm_opts = PL.mm_options(_mm_F())
    except Exception:
        mm_opts = []
    pmm = st.multiselect("🧩 멀티모달 특성 추가 (포트폴리오 › 멀티모달 서브탭 자료, 19개 품목에만 값 — 나머지 품목은 중간 순위)", mm_opts, [], key="pn_mm",
                         help="텍스트 FinBERT 감성 · WASDE·EIA·NASS·LME 리포트 수치 · CFTC COT · 기상 · 매크로·금융(품목 민감도 β × 값). 발표 이후 값만, 매 시점 품목 간 순위로 변환.")
    chars = chars + ["MM · " + o.split(" · ", 1)[1] for o in pmm]
    d1_, d2_, d3_, d4_ = st.columns(4)
    tgt_l = d1_.selectbox("목표", ["변동성 표준화 + 시점별 평균 제거 (상대 순위)", "변동성 표준화", "원 수익률"], key="pn_tgt")
    target = {"변동성 표준화 + 시점별 평균 제거 (상대 순위)": "vol_cs", "변동성 표준화": "vol", "원 수익률": "raw"}[tgt_l]
    refit = d2_.number_input("재학습 주기 (기간)", 1, 500, {"B": 126, "W-FRI": 26, "ME": 6}[FQ], key="pn_refit")
    min_train = d3_.number_input("최소 학습 기간", 20, 3000, {"B": 504, "W-FRI": 104, "ME": 24}[FQ], key="pn_mt")
    q = d4_.select_slider("롱숏 분위", [0.2, 0.25, 1 / 3, 0.5], 1 / 3, format_func=lambda v: f"상·하위 {v:.0%}", key="pn_q")
    models = st.multiselect("모델", PN.PANEL_MODELS, default=["Pooled OLS", "Ridge", "Elastic Net", "Gradient Boosting"], key="pn_models")
    ALTP = _alt_panel_preds()
    alt_ok = FQ == "W-FRI" and basis == "futures"
    alt_pm = st.multiselect("🔮 대안 시계열 모델 (미리 계산된 19개 품목 주간 4주 앞 예측 — 주간 · 연속월물 기준에서만)", list(ALTP) if alt_ok else [], [], key="pn_alt",
                            help="Prophet·PatchTST·iTransformer·TFT·Neural ODE·GNN·Chronos·TimesFM을 19개 품목으로 매년 재학습한 예측(④ 백테스트 그리드·⑤ 대안 시계열 탭). "
                                 "괄호 = 예측 시작. 이 탭의 walk-forward와 달리 위 특성 선택과 무관하고, 4주 앞 예측을 매주 순위 신호로 그대로 평가.")
    if not alt_ok:
        st.caption("대안 시계열 모델은 사이드바 빈도 '주간' + 수익 기준 '연속월물'일 때 선택 가능 (미리 계산된 예측이 그 설정이라서).")
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
        pan = _panel_data(basis, tuple(uni), FQ, target, extra_bytes, ROLL, tuple(pmac), tuple(pmm))
    X = pan["X"]
    st.caption(f"패널: {X.index.get_level_values('date').nunique():,}기간 × {len(pan['ids'])}종목 = {len(X):,}행 "
               f"(품목별 모델이면 품목당 약 {X.index.get_level_values('date').nunique():,}행)")
    La_, Lr_ = _levels(extra_bytes, True), _levels(extra_bytes, False)
    if run and (models or alt_pm) and chars:
        res = {}
        prog = st.progress(0.0)
        for j, m in enumerate(models):
            prog.progress(j / len(models), f"{m} walk-forward 중...")
            wf = PN.walk_forward_panel(pan, m, chars, int(refit), int(min_train), fe=wf_fe, sector=wf_sec)
            if wf["pred"].empty:
                st.warning(f"{m}: 예측 없음 (최소 학습 기간이 너무 김)"); continue
            res[m] = dict(wf=wf, ev=PN.evaluate(pan, wf["pred"], La_, Lr_, q=q))
        for m in alt_pm:                                                  # 미리 계산된 대안 모델 예측 → 패널 (날짜, 품목) 행에 맞춤 (같은 주 = 같은 값)
            pv = ALTP[m].copy(); pv.index = pv.index.to_period("W-FRI")
            ix = pan["X"].index
            vals = pv.stack().rename("p")
            key_ = pd.MultiIndex.from_arrays([ix.get_level_values("date").to_period("W-FRI"), ix.get_level_values("id")])
            pr = pd.Series(vals.reindex(key_).values, index=ix).dropna()
            if pr.empty:
                st.warning(f"{m}: 선택한 유니버스와 겹치는 예측 없음"); continue
            res[m] = dict(wf=dict(pred=pr, importance=None, n_refits=0), ev=PN.evaluate(pan, pr, La_, Lr_, q=q))
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
        c1_, c2_, c4_, c3_ = st.columns([1, 1.5, 1.8, 0.8])
        tid = c1_.selectbox("목표 품목 (다음 달 수익)", ids_c, ids_c.index("yf:ZC=F") if "yf:ZC=F" in ids_c else 0, format_func=lbl, key="md_tid")
        rc_ = c2_.multiselect("원자재 일간 설명변수", COMMOD_IDS, default=[tid], format_func=lbl, key="md_rids", max_selections=2)
        mac_o = [i for i in ALL_IDS if GROUP.get(i) in ("Macro", "FX", "FRED daily")]
        rm_ = c4_.multiselect("🌐 거시 일간 설명변수 (금리·달러·VIX·스프레드)", mac_o, [i for i in ["yf:DX-Y.NYB", "fred:VIXCLS", "fred:DGS10"] if i in mac_o],
                              format_func=lbl, key="md_rmac", max_selections=3,
                              help="MIDAS는 일간(고빈도) 자료의 시차들을 Beta 가중으로 요약해 월간 목표를 예측 — 거시 변수도 매일 관측되므로 월 평균으로 뭉개지 않고 넣음. "
                                   "변화로 변환 (금리·스프레드 = 차분, 나머지 = 로그수익률)")
        rids = list(dict.fromkeys(rc_ + rm_))
        Kr = c3_.number_input("일간 시차 K", 5, 252, 63, key="md_Kr")
        md_mm = _mm_picker(tid, "md_mm")
        if len(md_mm) > 3:
            st.warning("MIDAS는 변수마다 Beta 가중 모수를 따로 추정해 느려지므로 멀티모달 설명변수는 앞의 3개만 사용합니다.")
            md_mm = md_mm[:3]
        import predict_lab as PL
        md_alt = st.multiselect("🔮 대안 시계열 모델 비교 (같은 월말 결정 · 다음 달 수익 예측, 결정일까지 자료만)", list(PL.MIDAS_ALT), [], key="md_alt",
                                help="Prophet = 결정일마다 직전 3년 주간 로그가격에 추세·연간 계절 적합 · Chronos·TimesFM = 직전 512영업일 로그가격 zero-shot. "
                                     "설명변수는 쓰지 않음 (자기 가격만) — MIDAS(설명변수 일간 시차)와 같은 표본에서 외표본 R² 비교.")
        if st.button("▶ MIDAS 수익 예측 (walk-forward)", type="primary", key="md_rrun") and (rids or md_mm):
            with st.spinner("월마다 재추정 중..."):
                R1 = _panel(tuple(dict.fromkeys([tid] + rids)), "B", "logret", start, end, extra_bytes, ROLL)
                regs = {short(i): R1[i] for i in rids}
                if md_mm:
                    MMd_ = PL.mm_item_frame(_mm_F(), tid, md_mm, R1.index)
                    regs.update({c: MMd_[c] for c in MMd_.columns})
                o_ = M.midas_return_wf(R1[tid], regs, int(Kr))
                for a_ in md_alt:
                    try:
                        o_["table"][a_] = PL.midas_alt(a_, R1[tid], o_["table"].index).reindex(o_["table"].index)
                    except Exception as ex:
                        st.error(f"{a_} 실패: {ex}")
                st.session_state["md_r"] = (tid, o_)
        if "md_r" in st.session_state:
            tid_, o = st.session_state["md_r"]
            T_ = o["table"]
            extra_m = [c for c in T_.columns if c not in ("y", "MIDAS", "동일가중", "과거평균")]
            if extra_m:
                rows_ = []
                for c in ["MIDAS", "동일가중"] + extra_m:
                    v = T_[["y", c, "과거평균"]].dropna()
                    pos_ = np.sign(v[c] - v["과거평균"]) if c in ("MIDAS", "동일가중") else np.sign(v[c])
                    ls_ = pos_ * v.y
                    rows_.append({"모델": c, "외표본 R² (vs 과거 평균)": 1 - ((v.y - v[c]) ** 2).sum() / ((v.y - v["과거평균"]) ** 2).sum(),
                                  "방향 적중률": float((np.sign(v[c]) == np.sign(v.y)).mean()), "부호 전략 Sharpe (월간, 비용 전)": float(ls_.mean() / ls_.std() * np.sqrt(12)) if ls_.std() > 0 else np.nan,
                                  "월 수": len(v)})
                st.dataframe(pd.DataFrame(rows_).set_index("모델").style.format({"외표본 R² (vs 과거 평균)": "{:+.2%}", "방향 적중률": "{:.1%}",
                                                                              "부호 전략 Sharpe (월간, 비용 전)": "{:+.2f}"}), use_container_width=True)
                st.caption("대안 모델은 다음 달 수익률 자체를 예측(부호 전략 = 예측 부호), MIDAS·동일가중은 회귀 예측에서 과거 평균을 뺀 부호. 월 수가 다르면 대안 모델의 초기 자료(512영업일·3년) 요건 때문.")
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
    st.subheader("멀티모달 자료 — 포트폴리오 › 멀티모달 서브탭 · ④ 예측 설명변수")
    st.caption("위 카탈로그(가격·거시)와 별도로 수집한 비가격 자료. 시작·끝 = 저장된 원자료의 실제 첫·마지막 날짜. "
               "분석에는 모두 '발표 이후'에만 쓰이도록 시점을 맞춘 뒤, 2026-10-26 기준 품목별 주간 특성(54개)으로 고정해 사용.")
    try:
        import predict_lab as PL
        st.dataframe(_mm_catalog(), hide_index=True, use_container_width=True)
    except Exception as ex:
        st.info(f"멀티모달 자료 목록을 읽지 못함: {ex}")
    st.caption("원자료 저장: API로 받은 값은 `data/mm/*.parquet`, 보고서 원문(WASDE PDF·STEO 엑셀)은 `data/reports/raw/`(1,354개). "
               "엑셀 사본 = `data/excel/멀티모달_원자료.xlsx` (시트별 원자료 + 목록, `python -c \"import predict_lab; predict_lab.export_mm_excel()\"`로 다시 만듦). "
               "텍스트는 FinBERT 점수와 GDELT 헤드라인까지 엑셀에 있고, 베이지북·STEO 본문은 점수만 저장(원문 미보관). "
               "FinBERT 점수 = P(긍정) − P(부정), 문장 평균. Reddit은 과거 자료를 API가 주지 않아 미사용.")
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
    st.subheader("자주 묻는 방법론 질문 (사용자 질문 정리)")
    st.caption("대시보드 작업 중 나온 방법론 질문과 답을 주제별로 모음. 수치는 질문 당시(2026-10) 결과 기준.")
    with st.expander("① 통계 지표 — AUC · Sharpe · 운과 실력 구분"):
        st.markdown("""
**AUC는 ADF 같은 가설검정인가?** → 아님. ADF는 '단위근이 있나'를 묻는 검정(p값)이고, AUC(ROC 곡선 아래 면적)는 **분류 예측의 순위 정확도**.
'실제로 떨어진 주'와 '오른 주'를 하나씩 뽑았을 때 모델이 떨어진 주에 더 높은 확률을 줬을 확률 — 0.5 = 동전, 1.0 = 완벽.
AUC는 Mann-Whitney U 통계량을 표준화한 값과 같아서 'AUC > 0.5'를 Mann-Whitney·DeLong 검정으로 p값을 낼 수 있음 (★ AUC 표의 '시험 p값', h주 목표가 겹치므로 표본을 1/h로 줄여 보수적으로 계산).

**AUC 기준을 0.53 → 0.55로 올리면 '확률이 67% → 22%로 떨어진다'는 뜻은?** → 품목마다 8개 조합(미루는 기간 1–4주 × 계절 특성 유무) 중 학습 AUC가 가장 높은 것을 고르므로,
**예측력이 전혀 없어도** 최댓값은 우연히 0.5보다 큼. 무작위일 때 그 최댓값이 0.53을 넘을 확률 67%, 0.55를 넘을 확률 22% → 기준을 0.55로 둬야 우연 통과가 줄어듦.

**Sharpe 2.2가 나왔는데 왜 '운'이라 하나?** → (1) 여러 전략 중 최고를 고르면 예측력이 없어도 최댓값은 높게 나옴 — 16개 전략 무작위 부호 시뮬레이션의 최댓값 중앙 +3.35, 95% +5.28이라
관측 2.22는 p = 0.86. (2) 관측치 58개(약 4–5년)면 Sharpe 표준오차가 ±1 안팎. (3) 고르는 과정까지 보정한 White Reality Check · Hansen SPA p = 0.2–0.35.
기간 길이 기준: 진짜 Sharpe S를 t = 2로 검증하려면 약 (2/S)²년 — Sharpe 1.0 → 4년, 0.5 → 16년. 19년 표본의 Sharpe 표준오차 ≈ 0.26.

**위약(placebo)이란?** → 같은 롱 비중·전환 빈도·지속성을 유지한 채 포지션을 무작위로 섞어(13주 블록) 만든 가짜 전략의 Sharpe 분포. 실제 전략이 그 95% 분위를 넘어야 '정보가 있다'고 봄.

**두 전략 차이가 의미 있는지는?** → 두 수익률의 차이 시계열로 t값을 봄. 예: 평소 70/30 vs '상관 급등 때 모멘텀 100%' Sharpe 0.72 vs 0.74 →
연 수익 차 +0.39%p, 차이 t = 0.53, 최대낙폭은 −34% → −41%로 오히려 악화 → 개선으로 볼 근거 없음.
""")
    with st.expander("② 학습/시험 · walk-forward · 미래정보 차단"):
        st.markdown("""
**AUC를 왜 2008–2016년까지만 계산했나?** → 그 구간이 **학습 기간**(미루는 기간·계절 특성·활성화 여부를 고르는 데 사용). 2017년 이후는 **시험 기간**으로 고를 때 전혀 보지 않고 성과만 확인.

**walk-forward 구조 (메타 전략·레짐 백테스트)** → ① 처음 20%(또는 2004년까지) = 첫 학습만, 예측 안 함 → ② 이후 매 주기(주간 26주·월간 6개월)마다 **그때까지의 모든 과거 자료로 재학습**(확장창)하며 다음 기간 예측
→ ③ 그중 처음 4년은 메타 선택 기록을 쌓는 예열(성적 제외) → ④ 그 뒤 평가. '4년 학습'이 아니라 '4년은 판단 기록 축적'이며 모델 학습은 계속됨.

**④ 단일 품목 '성과 지표'는?** → 단순 분할: 학습 비중 슬라이더(기본 70%)로 앞부분 학습, 뒷부분 테스트 → R²(학습)·R²(테스트).

**미래정보 차단 장치** → 결정 = 금요일(또는 월말) 종가, 체결·손익 = 다음 거래일부터. 표준화·하이퍼파라미터·HMM 추정은 학습 구간 안에서만, HMM은 평활 확률 대신 **전방 필터**(그 시점까지 자료만).
테스트: 미래 가격을 바꿔도 과거 판단·비중이 그대로인지 교란 검사 (test_lookahead · test_ls_strategies · test_regime_mv).

**가상(더미) 데이터를 1900년부터 만들면 정확도가 오르나?** → 실제 정확도는 안 오름. 가상 데이터는 생성 규칙을 다시 배우는 것일 뿐 새 정보가 없음. 문제는 표본 수보다 **신호 크기**
(주간 R² 1%여도 Sharpe ≈ 연 0.7이 상한에 가까움). 가상 데이터는 '이 표본 길이로 어느 크기의 신호를 찾을 수 있나'(검정력)·파이프라인 검증 용도로만 유용.
대신: 실제 장기 자료(세계은행 월간 1960–), 품목을 묶은 패널, 다른 종류의 정보(재고·작황·포지션), 예측 목표 바꾸기(변동성·순위·'더 쌀 확률').
""")
    with st.expander("③ 예측 모델 · 부호 전략 · 앙상블 · 메타 전략"):
        st.markdown("""
**부호 전략이란?** → 모델이 예측한 '다음 기간 수익률'이 양수면 롱(+1), 음수면 숏(−1). '과거보다 오르면'이 아니라 **지금부터 다음 기간까지 오를 것으로 예측하면 롱**.
개별 모델 행은 그 모델의 예측 부호, '앙상블(단순평균)'은 여러 모델 예측의 평균 부호. 무거래 구간(band) 설정 시 예측이 작으면 직전 포지션 유지.

**Ridge·Lasso가 Buy&Hold와 Sharpe가 같은 이유** → 규제가 계수를 0으로 줄여 예측이 '과거 평균(약간 양수)'으로 고정 → 항상 롱. 변수 선택 후 보유가 아니라, 남은 변수가 없어서 상수 예측이 된 것.

**평균 부호로 정하는 앙상블의 문제** → 모델이 엇갈리면 평균이 0 근처인데도 반드시 롱/숏을 잡음, 항상 롱인 모델이 섞이면 롱 쪽으로 기울어짐. 대안: 다수결 + 기권, 표준화 후 평균, 상수 예측 모델 제외.

**메타 전략(모델을 바꿔 가며 고르기)은?** → 6개 후보 손실이 모두 있어야 고르도록 돼 있어, 한 모델 예측에 빈 구간이 있으면 평가 내내 '균등 앙상블'로만 대체됨(표의 1.100이 모두 같았던 이유).
여러 검증에서 운과 구별되는 성과가 없어 v1.19.0에서 ④ 화면 제거 (⑥ 실험 기록은 유지).

**대안 시계열 모델(Prophet · Chronos · TimesFM · PatchTST · iTransformer · TFT · Neural ODE · GNN)은 원자재 4주 수익률을 맞히나?** → ⑤ 포트폴리오 › 대안 시계열 모델 서브탭.
롱숏 탭과 같은 19개 품목 · 비용 · 목표 변동성으로 2012-07–2026 백테스트: 방향 적중률이 모두 49.7–51.9%(동전 던지기 수준), 방향 매매로 52주 모멘텀(0.45)을 넘은 모델 없음.
순위 롱숏은 TimesFM 0.45(t 1.7)가 가장 높지만 우연과 구별 안 됨. 여러 모델이 2017 이전 음수 · 이후 양수 → 안정적 예측력보다 국면 운.

**정식 패키지와 자체 경량 구현이 왜 다르게 나오나?** → 같은 이름의 모델도 구조 크기·정규화·학습 횟수·초기값이 다르면 예측이 거의 딴판(PatchTST·iTransformer·TFT·GNN 예측 상관 0.07–0.25).
Prophet(0.84)·Neural ODE(0.57)만 비슷. 즉 '어떤 신경망이 좋다'가 아니라 '그 설정이 그 기간에 맞았다'에 가까움 → 시드만 바꿔 다시 학습한 흔들림 표로 운의 폭을 함께 봄.

**시드만 바꾸면 결과가 얼마나 흔들리나?** → 신경망 8개 × 시드 3개. TFT 정식판은 순위 롱숏이 −0.50 ~ +0.18로 부호까지 뒤집힘(한 번 돌린 Sharpe는 운). 세 시드 모두 양수이고 시드 평균 예측(앙상블)이 더 나은 것은 PatchTST 정식판 하나(방향 0.29–0.38 → 평균 예측 0.47, 순위 0.21–0.51 → 0.53, IC 평균 0.032).
다만 시험 기간 전체로 본 사후 결과이고 t ≈ 2 수준이라 관찰 후보로만 둠. 신경망을 쓸 때는 시드 여러 개의 평균 예측이 기본 처방.

**IC 0.02를 넘은 모델을 롱숏 혼합에 넣으면?** → 후보 고르기는 2012-07–2016 IC로만(시험 기간으로 고르면 사후 선택). 6개 모델이 기준 통과 → 'MinVar 50 + 모멘텀 25 + 모델 25'를 기존 혼합 후보 전부와 함께 구간 이동 선택.
2017–20: 학습 1위(PatchTST 경량 혼합)가 시험 11/20위로 미끄러짐, 2021–26: 가치 혼합 선택. 이어붙인 표본 외 0.97 vs 기존 0.88(차이 t 0.67) — 모델 혼합이 가치 혼합을 대신할 근거는 없음.

**모델 단독 Sharpe는 0–0.3인데 혼합 후보 Sharpe는 왜 0.6–1.1로 높나?** → 혼합 = MinVar 롱 50% + 52주 모멘텀 25% + 모델 25%. 혼합 Sharpe의 대부분은 MinVar(단독 약 0.7)와 모멘텀 몫이고 모델은 1/4만 차지.
모델의 기여는 '기존 70/30(같은 기간 0.71) 대비 차이'로 봐야 함: PatchTST 경량 혼합 0.72(+0.05%p, t 0.04) · iTransformer 정식 0.71(0.00) → 사실상 기여 없음. Neural ODE 정식 혼합 0.87(+1.8%p, t 1.36)은 단독 0.34라도 MinVar·모멘텀과 상관이 낮아 분산 효과로 혼합을 끌어올린 경우.

**'선택된 조합'은 무엇인가?** → 구간 이동 선택의 결과. 시험 구간마다 그 전 자료(학습 구간)만으로 후보 20개(기존 70/30 · MinVar 50 + 모멘텀 25 + 대체 전략 25 × 10 · 상한 15%판 3 · 모델 혼합 6) 중 학습 Sharpe 1위를 골라 다음 구간에 그대로 적용.
2017–20: 2012–16 학습 1위 PatchTST 경량 혼합(0.74) → 시험 0.02(20개 중 11위), 2021–26: 2012–20 학습 1위 가치 혼합(0.46) → 시험 1.66(1위). 이어붙이면 0.97 vs 기존 0.88. 사후 최고(Neural ODE 혼합 등)는 고를 수 없었던 결과라 성과로 치지 않음.

**레짐·시변 모델(Kalman TVP·HMM·MS·TAR·STAR)은 어디에?** → v1.12.0부터 ② 레짐 탭에서만 (④ 예측·메타 후보에서 제외).

**패널 모델에 거시 변수를 넣는 방법** → 거시 값은 모든 품목에 같아 시점별 순위 변환에서 사라짐 → **품목별 거시 민감도(과거 1년 β) × 이번 기간 거시 변화**를 특성 '거시 β×Δ'로 넣음 (예: 달러가 오른 주엔 달러 민감 품목이 불리).

**MIDAS** → 일간 자료(원자재·거시)의 시차들을 Beta 가중치로 요약해 월간 목표를 예측 — 월 평균으로 뭉개지 않고 최근 날에 더 큰 비중.

**횡단면 군집(④+ 탭)의 원리와 쓰임** → 같은 시점에 품목마다 특성(★ 구매 탭의 가격 매력 점수·가격 위치·단기 반등·상대가치·계절성 + 13·52주 수익률·고점 대비·범위 내 위치·변동성·추세)을 계산,
특성마다 품목 간 평균 0·표준편차 1로 맞춘 뒤 특성 공간에서 가까운 품목끼리 묶음(K-평균 = 중심까지 거리 최소, 계층 Ward = 합칠 때 분산 증가 최소 순, GMM = 군집별 정규분포). K는 실루엣 점수로 추천.
군집 1 = 가격 매력 점수 평균이 가장 높은 묶음 → '지금 상대적으로 싼 상태의 품목들'이 어디인지, 여러 품목 중 어디부터 살지 판단. 같은 군집은 비슷한 이유로 싸졌으므로 한꺼번에 사면 분산 효과가 작을 수 있음.

**분류 모델(로지스틱·LDA·SVM)은 군집에 어떻게 쓰나?** → 분류는 정답(라벨)이 있어야 해서 그 자체로는 군집을 만들 수 없음 → 군집 결과를 정답으로 두고 **경계를 설명**하는 데 씀:
LDA = 군집이 가장 잘 갈리는 2차원 지도 축, 로지스틱 계수 = 어떤 특성이 품목을 어느 군집으로 보내나, SVM(RBF) = 하나씩 빼고 맞혀 본 정확도(군집이 얼마나 뚜렷한가).
'다음 4주 상대 강세' 분류 예측은 AUC 0.505–0.523으로 운과 구별되지 않아 화면에서 제외.

**'오늘 기준' 상관은 어떻게?** → 상관은 한 시점 값으로 계산할 수 없어, 최근 2년 주간 수익률에 최근일수록 큰 가중(반감기 8주: 8주 전 자료 1/2, 16주 전 1/4)을 준 지수가중 상관으로 오늘 시점을 추정.

**Prophet(메타) 같은 빅테크 시계열 모델은 왜 원자재 가격에 약한가?** → Prophet은 '가격 = 매끄러운 추세(몇 번 꺾이는 직선) + 연간·주간 계절 + 휴일 + 고른 잡음'으로 봄. 웹 방문자 수처럼 요일·연말 패턴이 강하고 큰 충격이 드문 자료에 맞춘 설계.
원자재는 ① **오늘 값이 내일의 출발점**(랜덤워크에 가까움)인데 Prophet에는 '직전 값·직전 충격'을 쓰는 자기회귀 항이 없어, 급등 다음 주에도 매끄러운 추세선 쪽으로 되돌아간다고 예측함.
② **점프**(WASDE 발표 · 가뭄 · 전쟁 · 수출 금지)는 추세 꺾임 몇 개와 잡음으로만 설명 — 꺾임점은 과적합을 막으려 일부러 둔하게(벌점) 잡혀서 점프 뒤 새 수준을 늦게 따라가거나, 반대로 한 번의 점프를 '새 추세'로 착각해 길게 연장함.
③ **변동성이 몰려다님**(위기 땐 출렁임이 몇 배) — Prophet의 잡음은 기간 내내 같은 크기의 정규분포라 위기 때 예측 범위가 너무 좁고, 꼬리(큰 손실)를 과소평가.
④ **계절성은 이미 가격에 들어 있음** — 수확기 패턴은 모두가 알아 선물 가격(만기별 가격 차)에 미리 반영되므로 '계절 그대로 오르내린다'는 예측은 돈이 안 됨.
⑤ 미래 충격 원인(기상 · 정책)은 미리 알 수 없어 '휴일' 같은 정해진 달력 변수로 넣을 수 없음.
우리 검증(대안 시계열 결과표 `alt_ts_compare`, 주간 방향 예측 전체 기간): Prophet 정식 패키지 방향 적중률 50.6% · 방향 전략 Sharpe −0.17, 자체 구현 49.8% · −0.34 → 동전 던지기 수준. 다른 모델도 방향은 우연 수준이라 Prophet만의 실패는 아니지만, 구조상 점프·변동성 군집을 다루는 GARCH·레짐 모형이 위험 예측에는 더 맞음.

**구글 TimesFM · 아마존 Chronos 같은 파운데이션 모델, TFT · PatchTST · iTransformer 같은 딥러닝 모델도 왜 방향을 못 맞히나?** → 먼저 구분: TFT(구글 2019)는 LLM이 아니라 'LSTM + 어텐션' 딥러닝 모델이고,
LLM 구조를 쓴 것은 Chronos(숫자를 토큰으로 바꿔 T5 언어모델 구조로 학습) · TimesFM(GPT처럼 다음 구간을 생성) 쪽 — 둘 다 **뉴스나 문장을 읽는 게 아니라 숫자 패턴만 봄**.
못 맞히는 이유 ① **맞힐 몫이 거의 없음**: 주간 원자재 수익률에서 과거로 설명되는 부분은 1% 안팎(R² ≈ 0), 나머지는 새 뉴스(기상 · 정책 · 수요 충격) — 모델이 정교해도 없는 신호는 못 찾음.
② **사전학습 자료가 다름**: 파운데이션 모델은 전력 사용량 · 교통 · 날씨 · 소매 판매처럼 계절이 뚜렷하고 매끄러운 자료로 주로 배움 → 원자재 가격을 주면 '최근 수준 · 추세를 이어 간다'는 예측으로 기울어, 사실상 단순 모멘텀과 비슷해짐
(우리 결과표 '방향 모멘텀 상관' 0.26–0.41, 52주 모멘텀 기준선 Sharpe 0.45를 넘은 모델 없음).
③ **시장은 패턴을 지움**: 누구나 아는 규칙적 패턴은 거래로 사라지고, 국면이 바뀌면 관계가 뒤집힘 → 학습 기간에 배운 것이 시험 기간에 안 통함(대부분 모델이 ~2017 음수 · 2017~ 양수로 부호가 바뀜).
④ **자료에 비해 모델이 너무 큼**: 품목당 주간 약 1,000개 관측에 수십만~수백만 모수 → 잡음까지 외워 과적합. TFT · iTransformer는 큰 자료(수만~수백만 시계열)에서 힘을 쓰는 설계.
⑤ **오차를 줄이는 목표 ≠ 방향 맞히기**: 손실 함수(평균 제곱 · 분위 오차)를 줄이는 가장 쉬운 답은 '직전 값 근처'라 오차 지표는 좋아 보여도 오를지 내릴지에 대한 정보는 0.
⑥ **점프 · 두꺼운 꼬리**: Chronos는 값을 구간(토큰)으로 잘라 극단값이 잘리고, 대부분 모델이 잔잔한 시기 기준으로 범위를 잡음.
우리 검증(`alt_ts_compare`, 주간): 정식 패키지 기준 방향 적중률 Chronos-Bolt 51.7% · Chronos-2 50.7% · TimesFM 50.7% · TFT 51.6% · PatchTST 51.2% · iTransformer 50.2% · Prophet 50.6% (동전 50%),
주별 순위 상관(IC) 전부 0 근처(−0.03 ~ +0.02), 모델 합의도 52주 모멘텀보다 낮음 → 멀티모달 · 대안 시계열을 혼합에 넣었을 때 학습 최적 비중 0%와 같은 결론. 쓸모가 남는 곳은 방향이 아니라 **범위(분위) · 변동성 예측**과 계절이 뚜렷한 물량 · 수요 예측.
""")
    with st.expander("④ ★ 구매 판단 — 3색 · 점수 · 계절성 · 미루는 기간"):
        st.markdown("""
**3색 구간** → 🟢 적극 매수 = 가격 매력 점수 ≥ 문턱 τ(기본 60). 🔴 매수 보류 = 점수 < τ 이고 '미루는 기간 안에 더 쌀 확률' ≥ p*(기본 55%), 단 학습 AUC > 기준(0.55) 품목만.
🟠 판단 유보·매수 쪽 = 점수 50–59, 🟡 판단 유보·대기 쪽 = 점수 50 미만 (둘 다 확률 < p* 또는 AUC 기준치 미달). 🔴는 점수만이 아니라 확률 조건이라 2차원 지도로 표시.

**품목마다 미루는 기간(1–4주)이 다른 이유** → 8개 조합 중 학습 기간 AUC가 가장 높은 것을 품목별로 고르기 때문 (돈육·생우·휘발유 = 4주·계절, 코코아 = 2주 등).

**계절성 점수 계산** → 월평균 가격이 그해 연평균보다 몇 % 높은지 연도별로 구하고, **올해 이전 연도만**으로 '이 달은 평균적으로 얼마나 쌌나'를 계산(추천 프리셋은 최근 연도 가중, 반감기 5년)
→ 표준화해 0–100 점수. 가격 매력 점수에 15% 가중(추천). '몇 달 기다리라'는 신호가 아니라 이번 주 점수를 조금 올리고 내릴 뿐이며, 미루는 판단은 최대 1–4주.

**'더 쌀 확률' 모델의 설명변수** → 점수·점수 변화, 구성요소(가격 위치·반등·상대가치·계절성)·변동성 분위, 1·4·13주 수익률, 13주 변동성, (선택) 연중 주차 사인·코사인 — 매년 1월 그 이전 자료로만 로지스틱 회귀 재학습.

**롱숏 연구의 신호(가치·추세·칼만·레짐)를 넣으면 AUC가 오르나?** → 19개 품목, 품목별 미루는 기간·계절 특성은 그대로 두고 설명변수만 추가(학습 2008–16 / 시험 2017–).
시험 AUC 평균: 기본 0.544 · +가치 0.553 · +추세 0.545 · +레짐 0.551 · +전체 0.555. 품목마다 학습 AUC 1위 묶음을 고르면 0.556(+0.012, 19개 중 11개 개선, t 2.07) — 레짐 묶음이 8품목에서 선택.
개선은 통계적으로 의미 있는 경계지만 크기가 작고, 시험 AUC > 0.55 품목 수는 7 → 8. ★ 구매 탭 '보완 실험' 펼치기에 품목별 결과.

**상관·멀티모달·대안 시계열 등 다른 탭 방법론을 구매 판단에 넣으면 AUC가 오르나? (보완 실험 2, 2026-10-27)** → 같은 틀로 상관(품목-바스켓 52주 상관·흡수비율), 멀티모달 7갈래 요약(뉴스 제외), COT·리포트만,
④ 그리드의 표본 외 4주 예측(Lasso·EN·패널·MIDAS·Neural ODE·PatchTST·Chronos-Bolt)을 추가. 시험 AUC 평균 기본 0.544 → 레짐 0.551 · 상관 0.550(t 1.86) · 예측 모델 0.548 · 멀티모달 0.548 · 전체 0.544(변수가 많아 오히려 그대로).
품목별 학습 1위 묶음 선택 0.552(9/19 개선, t 1.26). 어느 묶음도 t > 2가 아니라 판단 기준은 그대로 두고 참고 표시만 — ★ 구매 탭 '보완 실험 2'.

**추세 배지** → 52주 수익률 부호 (견고성이 확인된 유일한 방향 신호). 🔴 보류인데 상승 추세면 '보류는 신중'.

**🏷️ 횡단면상 매력 묶음 배지** → '횡단면' 하위 탭과 같은 방식(기본 19개 품목 · 구매 매력도 특성 · K-평균, K = 실루엣 자동)으로 군집했을 때 가격 매력 점수 평균이 가장 높은 군집 1에 속한 품목.
개별 점수(내 품목이 문턱을 넘었나)와 품목 간 비교(다른 품목들보다 상대적으로 싼 상태인가)를 한 카드에서 보기 위함. 하위 탭에서 품목·특성을 바꿔도 배지는 기본 설정 기준.
배지 옆 '같이 사면 분산 효과 큰 묶음 내 품목' = 같은 군집 1 안에서 오늘 기준 상관(지수가중, 반감기 8주)이 가장 낮은 2개.

**배지 검증** → 2006년부터 4주마다 그 시점 자료로 다시 군집, 군집 1 품목의 다음 13주 가격 변화 vs 전체 평균: +0.70%p(t 1.03, 13주 겹침 보정), 2017 전/후 +0.77/+0.62%p, 더 오른 시점 52%.
가격 매력 점수 5개만으로 군집하면 +1.41%p(t 1.67). 방향은 맞으나(싼 묶음이 이후 더 오름 → 그때 산 게 유리) 통계적으로 확실하진 않음 → '참고 신호'.

**실루엣 점수 · 로지스틱 · SVM은 K-평균·GMM과 별개의 군집인가?** → 아님. 군집은 선택한 방법(K-평균·계층·GMM) 하나로만 만듦.
실루엣 = 그 군집 결과의 품질 점수(각 품목이 자기 군집과 얼마나 가깝고 다른 군집과 얼마나 먼가, −1–1) → K를 고를 때 사용. 로지스틱·LDA·SVM = 만들어진 군집을 '정답'으로 두고 **사후 설명**(어떤 특성이 가르나, 경계가 얼마나 뚜렷한가).

**지도 축(LDA · PCA)은 시각화 방법인가?** → 그렇다. 군집 계산은 특성 전체(예: 11차원) 공간에서 하고, 지도는 그것을 2차원으로 눌러 보여주는 것.
PCA = 특성 전체의 퍼짐을 가장 많이 담는 두 방향(군집과 무관), LDA = 군집끼리 가장 잘 갈리는 두 방향(군집 결과를 사용) → LDA 지도에서 군집이 더 또렷이 떨어져 보임. 지도에서 가까워 보여도 원래 공간에선 멀 수 있음.
""")
    with st.expander("⑤ 레짐 · 요인 · 상관"):
        st.markdown("""
**다변량 레짐(HMM)** → 원자재 수익률 + 거시(금리 변화, 장단기 금리차·신용 스프레드 수준, VIX 로그 수준, 달러·주가·운임 수익률)를 한 패널로 넣어 결합 분포가 달라지는 국면을 추정.
'지금 상태 설명'에는 유용(스트레스 국면 = 2008·2020·2022)하나 다음 기간 방향 예측은 96칸 중 4칸만 위약 통과(우연 수준).

**국면 프로파일의 평균이 대부분 0 근처인 이유** → 주간 수익률 평균은 변동성에 비해 매우 작음(옥수수 평균 0.1% vs 표준편차 4% → z ≈ 0.03). 국면은 수준 변수(VIX·스프레드)와 **변동성**으로 갈림 → '변동성 (전체 대비 배수)' 보기.

**① 시장 구조의 '평시 vs 위기' 상관과 ② 레짐 '국면별 상관'의 차이** → 전자는 달력으로 정한 위기 구간(사이드바), 후자는 데이터(HMM)가 정한 국면.

**흡수비율(AR)** → 상관행렬 1주성분의 설명 비중 = 여러 품목이 한 요인에 묶인 정도. 급등 시 분산 효과 감소 → 리스크 탭 경고, ① 시계열, ⑤ 롱숏 ②.

**변동성 레짐 (HMM 3상태 · 반감기)** → 일간 수익률을 저·중·고 변동 국면으로 나누고, 국면별 EWMA 감쇠율 λ와 반감기(충격이 절반으로 줄기까지 걸리는 기간)·3년 블록 간 λ 동일성 검정 → 리스크 탭 'HMM 3상태 λ' VaR의 근거.

**Conformal 예측구간** → 학습 끝 25%에서 모델이 틀린 크기를 모아 '예측 ± 오차의 90% 분위'를 90% 구간으로 씀(분포 가정 없음). 시험 기간에 실제 값이 들어간 비율(커버리지)로 검증, ACI는 자주 빗나가면 구간을 넓힘.
""")
    with st.expander("⑥ 포트폴리오 · 롱숏 — 역변동성 · MinVar · 평균-분산 · 목표 변동성 · 혼합"):
        st.markdown("""
| 방법 | 품목별 비중 | 상관 반영 | 기대수익 μ | 쓰는 곳 |
|---|---|---|---|---|
| 동일가중 | 1/N | ✗ | ✗ | 롱숏 기준 바스켓 |
| 역변동성 | 부호(또는 1) ÷ 품목 변동성 | ✗ | 부호만 | ④ 시계열 모멘텀 |
| 최소분산 (MinVar) | 포트폴리오 분산 최소 (Ledoit-Wolf 공분산, 품목 상한) | ✓ | ✗ | ⑥ · ⑧ |
| 평균-분산 | w'μ − (λ/2)·w'Σw 최대 | ✓ | ✓ | Portfolio Customization 품목 단위(**Newton-Raphson**), ⑧(SLSQP·Σ⁻¹μ) |

**롱숏 탭 ①–⑤는 최적화가 아님** → 정해진 규칙(동일가중·헤지비율·켬/끔·부호÷변동성·상위/하위 1/3).

**'목표 변동성 10%'는 상한이 아니라 목표** → 직전 26주 실현 변동성으로 배율을 정해 포트폴리오 **전체**를 연 10%로 키우거나 줄임(최대 4배, 사후 배율 아님). 같은 위험에서 비교하려는 장치.
롱숏 탭의 ④ 모멘텀·⑥·⑦·⑧에 적용, Portfolio Customization의 품목 단위 평균-분산(Newton) 화면에는 별도 '목표 변동성' 입력칸.

**'추세 방향 × (10% ÷ 품목 변동성) ÷ 품목 수'의 10%** → 품목 하나만 들면 연 변동성 10%가 되는 크기(위험 균등). 이후 전체를 다시 10%로 맞추므로 숫자 자체는 상쇄되고 **1/σ 상대 비율만** 의미.
품목이 많으면 품목당 몫은 작아지고 서로 상쇄돼 전체 변동성이 낮아지지만(배율 전 약 3.9%), 전체 배율로 10%를 맞춤.

**MinVar '상한 15%·10%'** → 변동성이 아니라 **품목당 최대 비중**(집중 제한). 전체 10% 목표와 별개.

**모멘텀 기간** → 롱숏 탭의 '모멘텀'은 **52주** 수익률 부호 (13주판은 비교용, Sharpe ≈ 0으로 탈락). ⑧ 평균-분산의 μ도 52주 부호. Portfolio Customization의 품목 단위 평균-분산(Newton) 화면은 '모멘텀 기간' 선택(기본 12-1개월).

**목표 변동성의 작동 로직** → ① 방법별로 '상대 비중'을 먼저 정함(역변동성·MinVar·혼합) → ② 그 비중으로 운용했을 때의 직전 26주 주간 손익 표준편차(연율) σ̂ₜ를 계산
→ ③ 배율 kₜ = 10% ÷ σ̂ₜ (최대 4배)를 모든 품목 비중에 똑같이 곱해 다음 주에 적용. σ̂ₜ가 15%면 전 품목을 2/3로, 5%면 2배로 — **특정 품목만 줄이는 게 아니라 전체 크기만 조절**, 품목 간 비율은 그대로.
과거 26주를 보므로 급변 시 반응이 늦고, 실현 변동성은 평균적으로 10.7–11.2% 수준. 10%는 분산이 아니라 **변동성(표준편차)**.
Portfolio Customization의 품목 단위 평균-분산(Newton) 화면(Newton)은 기본 꺼짐(목표 0) — 켜면 예측 공분산(Ledoit-Wolf 또는 MIDAS)으로 계산한 예측 변동성 기준, 최대 배율 기본 2배.

**MinVar의 '최소'와 목표 변동성 10%는 다른 개념인가?** → 다름. MinVar는 '원금을 다 투자(Σw = 1)·공매도 없음·품목 상한' 조건에서 분산이 가장 작은 **구성 비율**을 정함 — 결과 변동성은 데이터가 결정(이 19개 품목에선 약 10.7%, 10%와 비슷한 건 우연).
목표 변동성은 그 구성을 **몇 배로 들고 갈지(크기)**만 정함 → 비율이 그대로라 '최소분산 구성'의 성질은 유지. 롱숏 탭에서 10%를 붙인 이유는 ① 여러 전략을 같은 위험에서 비교 ② 혼합 비율(30/70 등)이 '위험 배분'이 되게 하려는 것.
배율 없는 MinVar: Sharpe 0.69 · 변동성 10.7% · 낙폭 −44% / 10% 배율: 0.75 · 10.8% · −41%.

**평균-분산(Newton)은 왜 목표 변동성이 0인가?** → '0 = 사용 안 함'. 평균-분산은 위험회피계수 λ와 제약(롱온리 Σw = 1, 롱숏 Σ|w| ≤ L)이 이미 위험 크기를 정하므로, 목표 변동성까지 켜면 같은 일을 두 번 함 → 기본 꺼짐, 필요할 때만 켬.

**포지션 크기(배율)를 정하는 위험 측정 두 가지** → ① 롱숏 탭: 그 전략이 **실제로 낸 과거 26주 주간 손익**의 표준편차(연율) — 단순·실제 결과 기준이나 비중이 바뀌어도 반응이 늦음.
② Newton 화면(켰을 때): **현시점 T의 비중 w × 품목 공분산 추정치 Σ̂** → √(w′Σ̂w). Ledoit-Wolf는 T까지의 과거 창(기본 104기간) 공분산을 안정화한 '현시점 추정치'(시계열 예측 모형 아님)를 다음 기간 변동성으로 그대로 씀, MIDAS는 최근 일간 변동성에 더 큰 비중을 둔 모형 예측.
두 방식 모두 비율은 1단계(각 방법) 그대로, 크기만 배율로 조정. 배율이 개별 품목 비중을 정하는 것은 아님 — 개별 비중은 역변동성(품목별 26주 변동성)·MinVar(104주 공분산) 등 1단계에서.

**총노출 한도 L과 최대 배율의 차이** → L(Newton 화면, 롱숏일 때): **1단계 최적화 안**에서 롱 + 숏(절댓값) 합 ≤ 원금 × L. 최대 배율: **크기 조정 단계**에서 목표 변동성 ÷ 위험이 커져도 이 배수 이상 키우지 않음(롱숏 탭 4배, 바스켓 2배) — 변동성이 1/5로 떨어져도 5배가 아니라 상한까지만.
Newton 화면 최종 총노출 ≤ L × 최대 배율. '배율'과 '실제 노출'도 다름: 모멘텀은 배율 평균 2.9배지만 배율 전 비중 합이 원금의 약 40%라 실제 총노출 평균 1.26배(최대 2.33배), MinVar는 배율 평균 1.07배(최대 노출 1.84배).

**레버리지 이자 비용을 넣어야 하나?** → 선물은 증거금만 넣고 명목 노출을 가지므로 빌린 돈·이자가 없고, 선물 수익률 자체가 무위험 이자를 뺀 **초과수익**(Sharpe도 이 기준). 오히려 빠진 건 원금을 단기 국채에 넣어 받는 **이자 수입**(2007년 이후 미 3개월 평균 연 1.47%) — 총수익이면 모든 전략에 약 +1.5%p, 순위 불변.
'유동성 비용 배수'는 이자가 아니라 거래할 때의 호가 차이·체결 미끄러짐 비용. ETF·현물을 빌려 사는 방식이라면 차입액 × 금리를 비용으로 넣어야 함.

**IC (모멘텀)란?** → Information Coefficient = 신호와 다음 기간 실제 수익률의 상관계수(0 = 쓸모없음, 실무 0.02–0.10이면 양호). Newton 화면은 Grinold 공식 **기대 초과수익 μ = IC × 품목 변동성 × 신호 점수**(모멘텀 ÷ 변동성, ±2 제한).
예: 옥수수 변동성 25%, 점수 +1.5, IC 0.05 → μ = 연 +1.9%. IC를 키우면 신호 강한 품목에 더 몰고 비용을 감수해 더 자주 거래, 줄이면 분산·비용 절약 우선. IC ÷ λ 비율이 '신호 대 위험' 균형을 정하므로 λ와 많이 겹치나, IC는 거래비용과의 균형도 정해 완전히 같지는 않음.

**26주·104주 창은 왜? (52주가 아닌 이유)** → 26주 ≈ 6개월, 104주 ≈ 2년. 변동성은 몰려서 오고 빨리 변하므로 **반응성**을 위해 짧게(CTA 관례 3–6개월), 공분산은 19개 품목이면 추정할 모수가 190개(분산 19 + 상관 171)라 **안정성**을 위해 길게(표본 ≥ 품목 수 × 5).
검증(롱숏 탭 '크기 조정 방식 · 창 길이 비교'): 모멘텀 품목 변동성 창 13주 0.40 · 26주 0.49 · 52주 0.52, MinVar 공분산 창 52주 0.55 · 104주 0.73 · 156주 0.82 — 결과를 보고 고르면 과적합이라 기본값 유지.

**변동성을 볼 때 다른 품목과의 상관도 보나?** → 방법마다 다름. 역변동성(모멘텀)은 **자기 변동성만**(상관 무시). MinVar·평균-분산은 **공분산 전체**(상관 포함). 크기 조정은 포트폴리오 전체 손익(롱숏 탭) 또는 w′Σ̂w(Newton 화면)를 보므로 상관이 자동 반영.

**Ledoit-Wolf 공분산이란?** → 품목 수에 비해 표본이 짧으면 표본 공분산은 잡음이 커서 상관이 극단적으로 나오고, 최적화가 그 잡음을 '기회'로 착각해 극단적 비중을 만듦.
Ledoit-Wolf(2004)는 **Σ̂ = δ·F + (1−δ)·S** — S = 표본 공분산, F = 단순한 목표 행렬(모든 분산 = 평균 분산, 상관 0), δ = 수축 강도를 '실제 공분산과의 기대 오차가 최소'가 되도록 자료에서 공식으로 계산.
효과: 극단적 상관이 0 쪽으로 당겨지고 행렬이 항상 역행렬을 가짐 → 비중이 안정. 우리 19개 품목 · 104주에서 δ 평균 0.16 (0.09–0.30) = 표본 84% + 목표 16%.

**롱숏 탭의 크기 조정을 현시점 LW 공분산으로 바꾸면?** → 더 나빠짐. MinVar: 과거 실현 26주 0.73 vs LW 104주 0.60, 모멘텀: 0.49 vs 0.41. 긴 창 공분산은 급변에 늦게 반응해 목표 10%에서 더 많이 벗어남(평균 이탈 1.6 → 2.3%p).

**Newton 화면 옵션 중 목표 변동성을 켜면 필요 없는 것은?** → 크기와 겹치는 것: 총노출 한도 L(부분 중복), IC(λ와 대부분 중복). 계속 필요한 것: 유동성 비용·수수료(거래비용), 종목 최대 비중(쏠림), 공분산 창·변동성 예측(Σ), 모멘텀 기간(μ), 거래 규칙(회전율), 최대 배율(배율 상한), 리스크 버짓 τ(4번째 전략 전용).

**혼합 'MinVar 30% + 모멘텀 70%'** → MinVar 최적화 포트폴리오(10%)와 52주 모멘텀 포트폴리오(10%)의 **주간 수익률을 0.3 : 0.7로 섞은 뒤** 다시 10%로 배율 — 두 전략에 위험을 30:70으로 배분.

**Buy&Hold Sharpe 0.32** → 원자재 19개 동일가중 바스켓(2007–) 연 4.8% · 변동성 14.8%. 품목을 섞어 변동성이 낮아져 단일 품목보다 높음.

**국면 전환 오버레이(평소 70/30 → 스트레스·상관 급등 때 전환)가 개선이 아닌 이유** → 0.72 → 0.74 차이는 연 +0.39%p·t 0.53으로 잡음 수준, 같은 빈도 무작위 전환 위약 95% 0.80 미만, 최대낙폭 악화.
②·③이 좋아 보였던 건 변동성 큰 동일가중 바스켓(낙폭 −64%)에 적용했을 때 — 이미 저변동·추세 혼합인 70/30에는 지킬 낙폭이 적고 HMM 신호는 급락 뒤 켜짐.

**데이터 빈도** → Portfolio Customization 품목 단위 평균-분산 = 사이드바 빈도 + 리밸런싱 주기 선택, 롱숏 탭 = 주간 고정.

**롱숏·대안 시계열·멀티모달 서브탭은 데이터 업데이트를 해도 왜 숫자가 그대로인가?** → 계산이 길어(롱숏 1–2분, 대안 시계열 1–2시간, 멀티모달 수십 분) 2026-10-26까지 자료로 한 번 계산해 저장(`data/frozen/`·`data/*_compare.csv`)한 값을 씀.
서브탭 제목 괄호 = 분석 기간. 기준일을 옮기려면 `app.py`의 FROZEN_END를 바꾸고 `data/frozen/`을 지운 뒤 `python experiments.py alt_ts`·`multimodal`을 다시 실행.

**Portfolio Customization '전략 조합'은 어떻게 계산하나?** → 재료 = 각 탭 전략의 주간 수익률(각각 목표 변동성 10%, 편도 5bp). 비중 × 수익률을 더한 뒤 직전 26주 실현 변동성으로 다시 10%에 맞춤(최대 4배).
비중: 직접 입력·동일 = 고정, 역변동성(52주)·최소분산(104주 Ledoit-Wolf, 롱온리)·과거 Sharpe 비례(156주, 음수 0) = 4주마다 **그 전 주까지 자료로만** 다시 계산.
고른 전략이 모두 있는 구간만 쓰므로 대안 시계열을 넣으면 2012-07부터. 여러 조합을 시험해 최고를 고르면 사후 선택 과대평가 — 차이 t값과 하위기간 Sharpe로 확인.

**④ 예측 탭에서 멀티모달 자료를 설명변수로 넣으면 어떻게 처리되나?** → 포트폴리오 › 멀티모달 서브탭과 같은 품목별 주간 특성(54개, 발표 이후 값)을 2026-10-26 기준으로 저장해 씀.
단일 품목 탭: 품목별 과거 자료로만 표준화하고 자료가 없는 구간은 0(평균), 분석 주기가 일간이면 그 날까지 나온 마지막 주간 값. 여러 품목 탭: 매 시점 품목 간 순위로 변환(19개 품목 밖은 중간 순위). MIDAS 탭: 일간 계단식 시계열로 Beta 가중(최대 3개).

**대안 시계열 모델(Prophet·TFT 등)은 ④ 탭에서 어떻게 학습하나?** → 단일 품목: 같은 학습/테스트 분할에서 학습 구간으로 한 번 학습, 테스트 구간 각 시점까지 자료로 예측. walk-forward는 학습형(Prophet·PatchTST·iTransformer·TFT·LSTM)은 제외, zero-shot(Chronos·TimesFM)만 포함 — 학습이 없어 재학습해도 값이 같으므로 날짜별 예측을 한 번 계산해 단일 분할과 공유. Prophet은 목표를 '추세 + 연간 계절 + 설명변수 선형'으로,
TFT는 설명변수를 과거 변수로, iTransformer는 설명변수(상관 상위 15개)를 토큰으로, Chronos-2는 상위 8개를 공변량으로. PatchTST·Chronos-Bolt·TimesFM은 구조상 자기 시계열만. 여러 품목: 19개 품목으로 미리 계산한 예측을 순위 신호로. MIDAS: 월말마다 자기 가격으로 다음 달 예측.

**walk-forward를 해도 결과가 같은 모델이 있나? 그럼 다시 계산할 필요가 없지 않나?** → 같은 모델은 **학습이 없는 zero-shot(Chronos-Bolt·TimesFM·Chronos-2)뿐**. 시점 t의 예측이 t까지의 가격에만 의존하므로 '언제 재학습하나'가 의미가 없음 → 날짜별 예측을 한 번 계산해 저장하고 단일 분할·walk-forward·재실행이 가져다 씀(Chronos-2는 공변량 고르는 구간만 달라 따로 계산). LSTM·Prophet·PatchTST·iTransformer·TFT는 재학습하면 가중치(신경망 숫자)가 새 자료로 바뀌어 예측도 달라짐 → walk-forward = 시점마다 새로 학습해야 해서 비용이 큼(그래서 제외). 이 모델들의 '매년 재학습' 결과는 ⑤ 대안 시계열 탭(19개 품목, 주간·4주 앞)에 미리 계산되어 있음.

**전략 백테스트(모델 × 설명변수)에서 '위약'은?** → '전부' 묶음과 같은 설명변수지만 자기 정보 외 변수의 시간 순서를 13주 블록으로 섞어 정보만 없앤 판. 실제 묶음이 같은 모델의 위약보다 확실히 높지 않으면 그 설명변수는 우연과 구별되지 않음.
전략 수가 많을수록(85개+) 예측력이 없어도 최댓값 Sharpe가 높게 나옴 — 표준오차 0.22 × √(2 ln N) ≈ 0.67까지는 운으로 가능.

**칼만 필터를 쓰는 롱숏 전략은?** → ⑨에 2가지: K1 칼만 추세(로그가격을 '수준 + 기울기' 상태공간 모형으로 보고 필터 기울기 부호로 롱숏 — 52주 모멘텀의 적응형 판, 잡음 비율은 사전 고정),
K2 칼만 동적 헤지 페어(회귀계수 β가 매주 변하는 페어의 예측오차 z로 평균회귀, 관측 잡음은 처음 52주로만 추정). 결과: K1 0.22(2017 이후 음수), MinVar 혼합에 넣으면 0.59–0.64로 기존 70/30(0.69)보다 나쁨, K2 0.14(기존 페어 −0.15보다 나음).
② 레짐의 Kalman TVP는 '베타가 시간에 따라 변하는 회귀'(설명·예측용)이고, 여기의 칼만은 같은 필터를 추세·헤지비율 추정에 쓴 것.

**주식 헤지펀드식 '구글 롱/엔비디아 숏' 같은 전략을 원자재로?** → 같은 테마 안 상대가치·과열을 사고팔고 시장 방향은 상쇄하는 구조로 옮김(⑨): 섹터 내 승자 롱(S1) −0.51, 섹터 내 과열 숏·낙오 롱(S2) +0.41,
가치(V1) +0.42, 왜도(W1) 0.05, 저변동성(L1) −0.03, 저베타(B1) 0.15, COT 헤징 압력(C1) −0.07, COT 쏠림 역발상(C2) −0.29. 원자재에선 '섹터 안 승자 추종'은 손해, '과열 숏·낙오 롱'과 가치가 유효.
가치·따라잡기는 모멘텀과 음의 상관 → MinVar 50 + 모멘텀 25 + 가치 25 = 0.89(낙폭 −28%, 기존 대비 차이 t 2.0). 단 결과를 보고 고른 사후 선택이므로 과대평가 가능, 미리 고르지 않고 대체 전략 전부를 섞으면 0.18.

**가치 혼합은 표본 외(OOS) 검증인가?** → 두 단계로 나눠 봐야 함. ① 각 전략의 매 시점 비중은 그 시점까지의 자료만 사용(5년 평균·순위·변동성 모두 과거 창) → 판단 단계는 표본 외.
② 어떤 전략을 섞을지는 처음엔 전체 기간 결과를 보고 골랐음(사후 선택). 그래서 '사전 선택 검증'을 추가: 후보 11개(기존 70/30 + 대체 전략 10개를 각각 25% 섞은 조합) 중 **2007–2016 학습 Sharpe만으로 고르면 가치 혼합이 1위(0.71)**, 그 조합의 2017년 이후 Sharpe 1.08 vs 기존 70/30 0.88.
2017년 이후는 모든 혼합이 좋았던 시기(기존도 0.88)라 절대 수준보다 '기존 대비 +0.2'가 핵심.

**가치(V1) 계산** → 품목마다 '로그가격 − 직전 260주(5년, 최소 156주) 로그가격 평균'(%) = 5년 평균 대비 얼마나 비싼가. 4주마다 19개 품목을 이 값으로 줄 세워 가장 싼 1/3(약 6개) 롱(합 +0.5, 동일가중), 가장 비싼 1/3(6개) 숏(합 −0.5) → 달러 중립, 이후 전체를 목표 변동성 10%로.
재조정당 평균 교체 비중 0.3. 연도별로 들쭉날쭉(2010·2013·2025 손실, 2016·2020 큰 이익) — 가치는 수년 단위로 회복되는 느린 신호.
주의: 롤 보정 연속월물 가격 기준이라 '5년 평균 대비'에 누적 롤 수익(콘탱고 손실·백워데이션 이익)이 섞임 → 순수한 현물 가치와는 다름.

**'가치 단독 0.42'와 '가치 혼합 학습 0.71 · 시험 1.08'은 왜 다른가?** → 대상이 다름. 0.42 = 가치 롱숏 하나만(전체 기간), 0.71·1.08 = MinVar 50 + 모멘텀 25 + 가치 25 **혼합 전체**.
같은 기간 비교: 학습(~2016) MinVar 0.36 · 모멘텀 0.66 · 가치 0.34 → 혼합 0.69–0.71 / 시험(2017~) 0.92 · 0.34 · 0.51 → 1.08. 혼합이 각 구성보다 높은 이유 = 상관이 낮거나 음수(가치–모멘텀 −0.44, 가치–MinVar −0.06)라 손실이 서로 상쇄돼 같은 10% 위험에서 Sharpe가 올라감.

**MinVar 50 + 모멘텀 25 + 가치 25는 같은 시점에 어떻게 계산되나?** → 매주 ① 세 전략이 각자 품목별 비중을 정하고, 각각 단독으로 연 10% 위험이 되게 크기를 맞춤 ② 품목별로 50:25:25 가중합 = 순 비중(예: 옥수수 MinVar +20%·모멘텀 −30%·가치 +40% → 0.5×20 + 0.25×(−30) + 0.25×40 = +12.5% 롱) — 반대 방향 포지션은 서로 상쇄 ③ 섞은 전체를 직전 26주 변동성으로 다시 연 10%에 맞춤(배율 상한 4배) ④ 다음 주 다시 50:25:25로. 수익으로 보면 '주간 수익 = 0.5×MinVar + 0.25×모멘텀 + 0.25×가치'에 ③의 배율을 곱한 것과 같음.

**2015–2020처럼 한 전략이 잘되는 해에 비중을 바꾸면 더 낫지 않나?** → 해마다 직전 자료만으로 비중을 정해 다음 해에 적용(2012–2026, 비용·전체 배율 제외)한 결과 고정 50/25/25 0.95 · 균등 1/3 1.05 · 역변동성 1.04 · 직전 5년 최소분산 0.95 · 직전 1년 승자 전략 비중↑ 0.95 · 고정과 최적 반반 0.94 · **직전 5년 최고 Sharpe 비중 추종 0.75(가장 나쁨)**. 미래를 아는 그해 최적은 1.67이지만 미리 알 수 없음. 고정 대비 수익 차 t는 모두 |t| < 1 → 유연 조정의 이득은 우연과 구별 안 됨. 단순한 균등·역변동성이 2015–20 구간(0.54 vs 0.27)에서 조금 나았던 정도.

**LSTM은 AR·VAR와 무엇이 다른가?** → AR은 고정 시차의 선형 가중합(기억 길이 = 차수로 고정). LSTM은 셀 스테이트 Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙C̃ₜ에 망각 f·입력 i·출력 o 게이트(0~1)가 붙어, 큰 충격이 오면 옛 기억을 지우고 새로 적는 식으로 시점마다 기억 길이가 달라짐(비선형 장기 의존성). 대가는 가중치 수(은닉 32 → 약 4,400개) — 주간 수익률처럼 표본 적고 신호 약한 자료에선 과적합, 이 앱 실험에서도 Ridge보다 열위 · 시드마다 결과가 달라 자동 선택·앙상블 후보에서 제외.

**시공간 GNN의 '정식 vs 경량 예측 상관 0.09'는 위기에 상관구조가 무너졌다는 뜻인가?** → 아님. 같은 자료·같은 구조(품목별 GRU 요약 → 그래프 합성곱)를 두 방식으로 구현한 모델(경량 = 자체 구현·행 정규화 인접, 정식 = torch_geometric GCNConv·대칭 정규화 + 자기 루프 + 잔차)의 **예측값끼리 상관이 0.09(부호 일치 51%)** 라는 뜻 → 구현 세부만 달라도 예측이 거의 안 겹칠 만큼 불안정. 정적 그래프의 위기 취약성(평시 상관으로 만든 인접 행렬이 위기 때 엉뚱한 이웃을 섞음)은 별개 문제 — 이 앱 그래프는 직전 156주 |상관| 상위 4개로 매년 갱신(그 해 안에선 고정). 두 구현 모두 방향 Sharpe 0.23·0.35로 52주 모멘텀(0.45) 미달.

**시험 기간(2017·2018년 이후) 성과가 객관적 평가인가?** → 그렇다. 고를 때 쓰지 않은 구간이라 학습 성과보다 객관적. 단 세 가지 한계:
① 약 9–10년 한 구간이라 Sharpe 95% 구간이 넓음(가치 혼합 2017~ 1.08 → 0.29–1.87, 기존 70/30 0.88 → 0.14–1.61, 두 구간이 대부분 겹침).
② 그래서 절대 수준보다 '같은 기간 기존 대비 차이'를 봐야 함: 가치 혼합 2017~ 연 +2.2%p(t 1.53), 2018~ +2.8%p(t 1.80), 따라잡기 +1.8%p(t 1.18), 칼만 페어 +0.6%p(t 0.65) — 방향은 일관되나 t < 2.
③ 시험 기간은 저변동 품목(생우·금)이 강했던 시기라 MinVar가 들어간 혼합이 모두 0.9–1.2로 높음 → 이 시기에 유리했던 구조인지, 늘 그런지는 시험 기간 하나로 확정 불가.

**구간 이동 선택 검증 (여러 시험 구간)** → 2013–16 · 2017–20 · 2021–26 각 구간 시작 시점에 그 전 자료(2007–)만으로 11개 혼합 후보 중 학습 Sharpe 1위를 골라 그 구간에 적용.
선택: 2013–16 따라잡기 혼합(시험 0.35 vs 기존 0.29, 11개 중 3위), 2017–20 가치 혼합(0.26 vs −0.04, 2위), 2021–26 가치 혼합(1.66 vs 1.52, 1위) — 세 구간 모두 기존보다 나음.
이어붙인 표본 외 2013–2026: 0.87 vs 기존 0.71, 연 +1.75%p, 차이 t 1.40 → 한 구간에만 의존하지 않고도 일관되게 우위지만 아직 t < 2.

**2021–2026 가치 혼합 Sharpe 1.66은 왜 그렇게 높았나?** → 대부분 시장 환경과 MinVar(롱온리) 덕. 같은 기간 MinVar 단독 1.50 · 모멘텀 0.86 · 가치 0.24 · 기존 70/30 1.52 · 바스켓 B&H 0.83(2007–2020은 0.15).
원자재 강세장(2021–22 인플레이션, 이후 금·축산·코코아 급등)에서 MinVar가 저변동 품목인 생우(평균 비중 22%, 연 +20%, 소 사육두수 감소로 사상 최고가)·금(18%, +13.5%)·대두박·코코아에 실려 있었음 — MinVar 품목별 연 기여 생우 +4.9%p · 코코아 +2.6 · 금 +2.6.
가치는 이 기간 자체 성과는 약했지만(0.24, 2024·2025 손실) 모멘텀과 상관 −0.53이라 혼합의 흔들림을 줄이는 역할 → 기존 대비 +0.14. 바스켓 대비 회귀로 나누면 가치 혼합 연 18.3% = 시장 몫(β 0.59 × 바스켓 11.2%) 6.6% + 알파 11.7%(t 3.7) → '강세장 덕'은 약 1/3이고, 나머지는 바스켓보다 더 오른 품목(생우·금 등)에 실려 있던 품목 선택 효과. 가치 혼합만의 실력분은 기존 대비 차이(+0.14).
연도별 혼합 Sharpe 2021 2.16 · 2022 1.71 · 2023 0.83 · 2024 2.21 · 2025 1.23 · 2026 1.89 (매년 양수). 생우·금이 꺾이면 MinVar 쪽이 크게 흔들릴 수 있는 집중 위험이 있음.

**MinVar 품목 상한을 15%로 낮추면 (생우·금 쏠림 완화)?** → 가치 혼합: 전체 0.89 → 0.79, 2021–26 1.66 → 1.51, 2017–20 0.26 → 0.19. 기존 70/30도 0.69 → 0.60. 쏠림을 줄이면 성과가 조금 줄지만 가치 혼합은 여전히 기존(25%)보다 나음.
구간 이동 선택 검증(후보 14개)에서도 상한 15%판은 한 번도 1위가 아님 → 25% 상한 유지, 대신 쏠림은 품목 기여도 표로 감시.

**강세장 덕 vs 전략 덕은 어떻게 나누나?** → 주간 수익률을 바스켓(19개 동일가중 Buy&Hold)에 회귀: r = α + β·바스켓. β = 원자재 시장을 얼마나 탔나, 시장 몫 = β × 바스켓 연수익, 알파 = 시장을 빼고 남은 연수익(품목 고르기·비중·롱숏).
전체 2007–: 가치 혼합 연 9.7% = 시장 2.5% + 알파 7.2%(t 3.96), 기존 70/30 = 2.6% + 5.0%(t 2.83). 2017–20은 두 혼합 모두 알파가 작음(기존 −1.2%, 가치 +2.2%).

**2년 롤링 알파 추세** → 가치 혼합의 2년 알파(연말 기준): 2014 +15.4 · 2016 +3.3 · 2018 −1.6 · 2020 +7.4 · 2022 +11.5 · 2024 +14.8 · 2026 +7.5%. 음수였던 기간 14%(기존 70/30은 29%), 유의(|t| ≥ 2) 29%.
최근 2년 평균 +11.8% vs 그 전 4년 +11.5% → 우위가 줄어드는 신호는 아직 없음(단 2024 정점 이후 내려오는 중). 기존 70/30은 2016–2019 알파가 음수였고 가치 혼합은 그 시기 손실이 작았음 → 가치의 완충 효과.
연도별 최대 기여 품목(가치 혼합): 생우 2010·2011·2014·2023·2024, 금 2009·2017·2019·2025, 대두박 2007·2020–2022 — 해마다 이끈 품목이 바뀌어 한 품목에만 의존하지는 않음.

**섹터별 기여 (가치 혼합, 2007–)** → 연평균 축산 +2.7 · 소프트 +2.2 · 곡물 +2.0 · 금속 +1.9 · 에너지 +1.3%p. 연도별 1위 섹터: 곡물 6회 · 축산 6회 · 에너지 3 · 금속 3 · 소프트 2 — 섹터도 해마다 돌아가며 이끔.
최근: 2022 곡물 +13.2, 2023·2024 소프트 +9.4·+9.3(코코아·커피), 2025 금속 +9.7(금), 축산은 2023–25 꾸준히 +5–9.

**전략 우위 경고 (리스크 탭 상단)** → 가치 혼합·기존 70/30의 2년 롤링 알파(바스켓 대비)가 0 아래면 ⚠️ 경고, 그 전 4년 평균의 절반 아래면 ℹ️ 주의, 그 외 ✅. 롤링 창 1·2·3년 선택은 롱숏 탭 그림에서(경고는 2년 고정).

**캐리(기간구조)** → Yahoo는 만기 지난 개별 월물이 없어 미국 곡물·금속 과거 캐리 불가 → 미국 에너지(EIA, 2024-04까지)·중국(Sina 2018–)로 검증했으나 견고하지 않아 화면 제외 (`collect_carry.py` 보존).

**BEKK는 공분산(변동성) 모형인데 왜 '충격 전이 · 변동성 전이'가 나오나? VAR 아닌가?** → VAR가 아니라 다변량 GARCH가 맞음. 다만 BEKK는 Hₜ = CC' + A'εₜ₋₁εₜ₋₁'A + B'Hₜ₋₁B 로
**다른 품목의 지난 충격·분산이 이 품목의 분산에 들어가도록** 허용함. A의 비대각 = i 품목의 지난주 뜻밖의 가격 변화(충격)가 j 품목의 이번 주 분산을 키우는 정도('충격 → 분산 전이', ARCH 전이),
B의 비대각 = i 품목의 높은 분산이 j 품목 분산으로 이어지는 정도('분산 → 분산 전이', GARCH 전이). 즉 **수익률(평균)이 옮겨 가는 것(VAR의 영역)이 아니라 위험(2차 모멘트)이 옮겨 가는 것** — 문헌의 'volatility spillover'.

**투자선(3D·2D, Portfolio Customization) 수익률이 목표변동성 서브탭 표의 같은 목표(예: 50%) 수익률보다 훨씬 낮은 이유?** → 점 구름·투자선은 재료 전략을 **섞기만 한 상태**(레버리지를 다시 맞추기 전)이기 때문.
섞으면 손실이 서로 상쇄돼 변동성이 목표보다 낮아지고 수익도 그만큼 낮게 찍힘. 성과표의 가치 혼합은 섞은 뒤 **다시 목표 변동성까지 레버리지를 키운** 결과 → 3D·2D의 ☆ 점이 표와 같은 위치.
또 배율 상한(예: 4배)에 걸린 재료(52주 모멘텀 등)는 목표를 올려도 단독 수익이 덜 늘어 구름 전체가 아래에 머묾.

**효율적 투자선의 점 수천 개(대시보드 4,000 · 발표 3,000 · PPT 5,000)는 진짜 계산인가, 가상인가?** → **비중만 무작위이고, 위험·수익은 실제 자료로 계산한 값.**
재료 전략(MinVar · 모멘텀 · 가치)의 실제 주간 수익률로 평균 μ와 공분산 Σ를 구한 뒤, 롱온리·합 100% 비중 w를 무작위로 뽑아(Dirichlet) 점마다 수익 = w·μ×52, 위험 = √(w'Σw×52)를 계산.
이 값은 '그 비중으로 매주 섞은 포트폴리오'의 비용 전 산술 연수익·연 변동성과 **정확히 같음**(섞은 수익률의 평균·표준편차를 공식으로 바로 구한 것) → 점 하나 = 실제 과거 자료로 돌린 고정 비중 백테스트 1개.
다만 ① '전체 기간' 보기는 같은 기간 자료로 μ·Σ를 구한 **사후(in-sample) 그림**이라 그 투자선을 미리 알 수는 없었음('직전 5년' 시점을 고르면 당시 알 수 있던 투자선 + 다음 해 실제 위치 ◇),
② 거래비용·배율 재조정은 빠짐(앞 질문), ③ 점 개수는 그림을 채우는 용도일 뿐 — 빨간 투자선은 점에서 고른 것이 아니라 최적화(SLSQP)로 따로 구한 경계.
점 수천 개는 근사 분포가 아니라 하나하나 정확히 계산(비중 행렬 × 공분산을 한 번에 곱해 점마다 w'Σw · w·μ를 구함 — 반복문 대신 행렬 계산이라 빠를 뿐 값은 같음).

**연도별 투자선의 최적점이 늘 우리 혼합보다 위에 있는데, 해마다 그 최적 비중으로 갈아타면 낫지 않나?** → 실제로 돌려 보니 **반대로 나빠짐** (`strategy_switch.py`, 2012–2026, 갈아타기 비용 포함, 문턱은 2012–16 학습에서만 선택).
연 1회 직전 5년 최대 Sharpe(접점)로 바꾸면 Sharpe 0.69 vs 고정 50/25/25 0.92(연 −2.6%p), 직전 3년이면 0.65. 성과 트리거(26주 Sharpe 하락 시 재최적화) 0.58–0.71, 투자선 이탈 트리거 0.64–0.80,
전략 모멘텀(최근 1년 1등 전략으로) 학습 1.09 → 시험 0.46, 위기 트리거(흡수비율 급등 → MinVar) 0.84–0.86(낙폭만 −20.4 → −19.2%로 소폭 개선), 결합 0.79. 나은 것은 **아무것도 예측하지 않는 고른 분산**뿐:
동일 비중 1.04 · 역변동성 1.03–1.04(고정 대비 +1.3%p, t 1.3 — 유의하지 않음).
이유: 그해 투자선은 **그해 결과를 알고 나서** 그린 것이라 항상 위에 있음(사후 최적은 늘 이김). 미리 쓸 수 있는 건 직전 5년으로 그린 투자선인데, 전략별 평균 수익률은 5년 자료로도 추정 오차가 커서(Sharpe 표준오차 약 ±0.45)
'직전 5년 1등'이 다음 해에도 1등일 근거가 약함 → 접점 비중은 운 좋았던 전략에 몰리고, 다음 해에 되돌림. 결론: 비중은 자주 바꾸지 말고, 고정 50/25/25도 사후 선택이므로 **동일 비중 · 역변동성 같은 규칙형 분산**이 더 정직한 기준.

**그럼 '다음 해(분기) 사후 최적 축'을 미리 읽는 신호를 여러 층으로 넣으면?** → `strategy_meta.py`(3축을 2001년 자료부터 다시 만들어 학습 2005–2016 · 시험 2017–, 비용 포함).
해마다 사후 최적 비중은 크게 출렁임(예: 2017 MinVar 100% → 2018 가치 100% → 2019 MinVar 67%) · 전년 1등 축이 올해도 1등인 비율 **33% = 무작위**.
레이어 6개(L1 흡수비율 · L2 변동성 국면 · L3 추세 강도 · L4 가치 스프레드 · L5 축 자체 모멘텀 · L6 VIX · 달러 · 금리)로 '다음 13주에 어느 축이 이길지'를 매년 다시 학습해 예측 →
분기 1등 축 적중률 학습 33% · 시험 29%(무작위 33%). 예측 1등 축을 +10%p 기울이면 시험 Sharpe 1.05 vs 고정 1.07, 위약(레이어를 섞은 가짜) 200회 중 26%가 더 좋음 → 효과 없음.
기울기를 키울수록(±20 · 30%p) 나빠지고(0.96 · 0.91, 낙폭 −44 · −49%), 레이어 하나씩만 써도 6개 모두 고정보다 낮음. → 사후 최적은 '미리 읽을 수 있는 패턴'이 아니라 그해 충격의 결과라 지금 자료로는 예측 불가.

**칼만 · HMM · 뉴스 감성 · DCC 동조성으로 3축 비중을 환경에 맞게 유연하게 바꾸면?** → `strategy_adaptive.py`(블랙-리터만 방식: 기본 50/25/25에서 ±25%p 안, 4주마다, 축 공분산은 3축 DCC, 학습 2008–16 · 시험 2017–, 비용 포함).
학습 최고 설정의 시험 Sharpe(고정 1.07): 칼만 평균 추적 0.79(학습 0.83으로 고정 0.61을 크게 이겼지만 시험에서 역전) · HMM 위기 국면 1.08 · 뉴스 감성 1.10(+0.3%p, t 0.26) · DCC 동조성 1.05 · 4개 결합 0.85(위약 p 0.82)
· DCC 위험 균등 1.10 · 동일 비중 1.11. → 비중을 바꿔 나아진 것이 없음.
**위험 신호가 켜지면 현금(이자 0)으로 피신하면?** (`strategy_cash.py`) → 대조군 '늘 70% 투자'도 Sharpe는 그대로(0.86)이고 낙폭만 −24.5 → −17.8%로 줄어듦 — 현금 효과의 대부분은 '덜 투자한 몫'.
신호 9종(HMM · 변동성 · DCC 동조성 · 흡수비율 · VIX · 뉴스 · 칼만 · 낙폭 손절 · 결합) 학습 최고 설정의 시험 Sharpe 0.63–1.12(고정 1.07), 위약 p 0.07–0.94 — 유의한 것 없음.
변동성 · HMM 신호는 오히려 나빠짐(0.63 · 0.92): 혼합이 이미 목표 변동성으로 위험을 줄이고, 위기 때 모멘텀이 벌어 주는 구간을 놓침. 그나마 낙폭 5% 손절(50% 현금)이 최악 1년을 −11.3 → −5.7%로 줄였지만(평균 노출 72%) 늘 70% 투자(−7.9%)와 큰 차이 없음.
단기 금리(3개월 국채) 이자를 넣고, 신호 개수에 따라 목표 변동성을 12 · 10 · 7 · 5%로 바꾸는 판(`strategy_voltgt_dyn.py`)도 시험 초과 Sharpe 0.96–1.02 vs 고정 1.07(위약 p 0.84–0.92) → 미채택.
선물은 증거금만 쓰고 자본 전체가 국채 이자를 받으므로 현금 피신으로 더 받는 이자는 없음(현금 몫만 이자를 받는다고 가정하면 금리가 높았던 2017–에 피신이 약간 유리해 보일 뿐).

**스트레스 테스트는 어떻게 하나? 포트폴리오 전체에 VaR를 적용하는 것인가?** → 둘은 다름. VaR · ES = 최근 변동성 · 상관으로 본 '보통의 나쁜 날'의 손실 분위(포트폴리오 전체 = 품목별 노출 × 공분산, 품목별 기여도 합 = VaR).
스트레스 테스트 = 확률을 묻지 않고 '그 일이 다시 오면 지금 장부는 얼마를 잃나'를 직접 계산: ① 과거 위기 재현(그 구간의 실제 가격 변화를 지금 비중에 적용) ② 가상 충격(곡물 −20%, 전 품목 동반 −15% 등)
③ 상관 스트레스(평소 0.1인 품목 간 상관을 위기처럼 0.6 · 0.9로 올린 VaR) ④ 역스트레스(자본 10%를 잃으려면 무엇이 얼마나 움직여야 하나). `stress_portfolio.py` 결과(2026-10-26 50/25/25 비중):
20일 99% VaR 2.9% · 상관 0.9면 8.6% · 2008 하반기 재현 17.4% · 전 품목 동반 −15% 충격 10.5% · 곡물(대두 · 밀 · 대두박)이 VaR의 약 절반 → 평소 VaR는 작아도 동반 급락 때 3–6배 손실.
대시보드 ◆ 리스크 → '구매·보유 물량 위험'은 같은 함수로 구매 물량에 VaR · 기여도 · 위기 시나리오를 적용.

**상관이 높아질 때 모멘텀 · 강한 추세 품목(2022 곡물 · 원유, 2025 금 · 은)에 집중하면?** (`strategy_crisis_mom.py`) → 학습 최고 규칙 시험 Sharpe 0.91 vs 고정 1.07(위약 p 0.48).
2022년 곡물 상관은 오히려 낮았고(연평균 0.45, 2011–12년 0.7–0.8) 급등은 1분기에 끝나 2분기에 되돌아(밀 −24%, 3분기 원유 −30%) 26 · 52주 추세는 늦게 타고 늦게 내림.
'상승 추세 상위 3개 롱'은 2025–26년 금 · 은을 잡아 +44%였지만 2008–16년엔 손실(학습 Sharpe −0.01) — 사후에 맞는 해가 있어도 미리 고른 규칙으로는 못 이김.
**곡물 합계 비중 상한 25%**(`strategy_graincap.py`): 시험 Sharpe 1.09(기준 1.07) · 지금 20일 VaR 2.87 → 2.25% · 곡물 VaR 기여 62 → 34% — 수익 손실 없이 위험 쏠림만 줄어 위험 한도 규칙으로 검토할 만함.
위험 신호는 비중 조절 대신 **손실 한도 경보**(`risk_alert.py`, 정상 · 주의 · 경고)로 씀 — 과거에 경보 뒤 큰 손실이 몰리지는 않아(하위 10% 손실 비율 0–14%) 예측용이 아니라 현재 위험 확인용. 칼만은 '안 먹히는 축을 줄이는' 적응을 실제로 하지만, 줄인 뒤에 그 축이 살아나 늦게 따라가는 손해가 더 큼(가치가 살아난 2016 이후에야 비중을 35%로 늘림).

**연수익 '산술'과 '복리'의 차이?** → 산술 = 주간 수익률 평균 × 52 (평균적으로 한 해에 버는 몫), 복리(CAGR) = 실제 자본이 불어난 속도 ((최종 자본 / 처음)^(1/년수) − 1).
변동이 있으면 복리가 항상 낮음 — 대략 복리 ≈ 산술 − 변동성²/2. 예: +50% 다음 −50%면 산술 평균 0%지만 자본은 1 → 0.75(복리 연 −13%).
가치 혼합 목표 10%: 산술 9.6% · 복리 9.4% (차이 작음) / 목표 50%: 산술 45% · 복리 36% (변동성 54%라 차이 큼) → 레버리지를 키울수록 '실제로 남는 수익'은 산술보다 덜 늘어남.
""")
    with st.expander("⑦ 선물 이음새(롤오버) · 롤 비용 · 차입·거래비용 — 19개 품목 처리 현황"):
        st.markdown("""
**롤 비용·차입·거래비용은 어떻게 반영했나?** → 차입비용: 선물 숏은 빌리는 자산이 없어 0 (반영 안 하는 게 맞음). 거래비용: 거래 금액 비례 편도 5bp (포트폴리오 탭은 선물 틱 기준).
증거금 이자는 더하지 않음(보수적). 롤 비용: 따로 빼지 않고 **이음새 날 가짜 점프를 지우면** 남은 가격 흐름에 롤 손익(콘탱고 손실·백워데이션 이익)이 자연히 들어감.

**이음새 날 가짜 점프란?** → Yahoo 연속월물(=F)은 만기 때 다음 월물로 그냥 갈아끼움 → 그날 수익률 = ① 두 월물 가격차(가짜) + ② 그날 진짜 움직임.
예: 콘탱고에서 3월물 100 → 5월물 103으로 갈아타면 그날 +3%가 찍히지만 실제 손익은 0, 진짜 롤 비용 −3%는 그 뒤 5월물이 현물로 수렴하며 서서히 생김.
①을 안 지우면 가짜 +3%가 이후 −3%를 상쇄해 롤 비용이 사라짐.

**1.5배 검정** (`roll_adjust.py`) → 거래소 만기 규칙으로 롤 날짜를 정하고, 롤 날 |수익률| 평균이 평상일의 1.5배 이상이면 '가짜 점프 있음' → 그 품목의 롤 날 수익률을 **0**으로 두고 가격을 다시 이음(최신 가격 고정, 과거 조정).
①과 ②를 나눌 자료(월물별 가격)가 없어 통째로 0 → ①은 확실히 제거, ②(평균 0)는 잡음으로 손실 — 편향 없는 근사. 한계: 크기만 보므로 작지만 같은 방향으로 쌓이는 갭(커피·원유)은 놓침.
①을 직접 잴 수 있는 곳은 EIA 근월물·차월물(F1·F2, 에너지 4개, 2024-04까지)뿐 — 원유·천연가스에서 Yahoo 롤 날 수익률과 ln(F2/F1)의 상관 0.70·0.80 (가짜 점프 확인).

**19개 롱숏 품목 처리 현황** (2026-10-28 점검; ✖ 검정 = 원래 `RULES`에 없어 별도 스크립트로 처음 검정)

| 섹터 | 품목 | 롤 월물 (연속월물이 거치는 계약월) | 롤/년 | 롤 시점 (추정 규칙) | 검정 | 1.5배 비율 | 현재 처리 | 이음새 갭 (연) | 권고 |
|---|---|---|---|---|---|---|---|---|---|
| 곡물 | 옥수수 | 3·5·7·9·12월 (H K N U Z) | 5 | 계약월 15일 직전 영업일 만기 → 다음 날 | ✔ | 2.84 | 0 처리 | +4.4% (제거됨) | 유지 |
| 곡물 | 소맥 (CBOT) | 3·5·7·9·12월 (H K N U Z) | 5 | 〃 | ✔ | 2.04 | 0 처리 | +9.5% (제거됨) | 유지 |
| 곡물 | 대두 | 1·3·5·7·8·9·11월 (F H K N Q U X) | 7 | 〃 | ✔ | 2.28 | 0 처리 | −6.7% (제거됨) | 유지 |
| 곡물 | 대두박 | 1·3·5·7·8·9·10·12월 (F H K N Q U V Z) | 8 | 〃 | ✔ | 2.54 | 0 처리 | −14.6% (제거됨) | 유지 |
| 곡물 | 대두유 | 1·3·5·7·8·9·10·12월 (F H K N Q U V Z) | 8 | 〃 | ✔ | 1.29 ✖ | Yahoo 그대로 | +0.3% (남음) | 유지 |
| 소프트 | 설탕 | 3·5·7·10월 (H K N V) | 4 | 계약월 첫 거래일 (전월 말 만기) | ✔ | 2.87 | 0 처리 | +1.7% (제거됨) | 유지 |
| 소프트 | 커피 | 3·5·7·9·12월 (H K N U Z) | 5 | 첫 통지일(인도월 7영업일 전) 직전 | ✖ | 1.27 ✖ | Yahoo 그대로 | **+3.5% (남음, t 3.2)** | 유지 · ⚠ 주의 |
| 소프트 | 코코아 | 3·5·7·9·12월 (H K N U Z) | 5 | 첫 통지일(인도월 10영업일 전) 직전 | ✖ | 1.09 ✖ | Yahoo 그대로 | +2.2% (남음, t 2.1) | 유지 · ⚠ 주의 |
| 소프트 | 면화 | 3·5·7·10·12월 (H K N V Z) | 5 | 첫 통지일(인도월 5영업일 전) 직전 | ✖ | 1.49 ✖ | Yahoo 그대로 | +0.4% (남음) | 유지 |
| 축산 | 생우 | 2·4·6·8·10·12월 (G J M Q V Z) | 6 | 계약월 말 만기 → 다음 달 첫 거래일 | ✔ | 4.12 | 0 처리 | −5.3% (제거됨) | 유지 |
| 축산 | 돈육 | 2·4·5·6·7·8·10·12월 (G J K M N Q V Z) | 8 | 계약월 10번째 거래일 만기 → 11번째 | ✔ | 6.91 | 0 처리 | −1.7% (제거됨) | 유지 |
| 에너지 | 휘발유 | 매월 (12개) | 12 | 계약월 첫 거래일 (전월 말 만기) | ✔ | 2.10 | 0 처리 | −9.2% (제거됨) | 유지 |
| 에너지 | 난방유 | 매월 (12개) | 12 | 〃 | ✔ | 1.55 | 0 처리 | −2.4% (제거됨) | 유지 |
| 에너지 | **천연가스** | 매월 (12개) | 12 | 인도월 첫날 3영업일 전 만기 → 다음 날 | ✖ | **1.92 ✔** | Yahoo 그대로 | **+16.8% (남음; EIA 실측 ≈ +23%)** | **0 처리 대상** (아래 영향 참고) |
| 에너지 | 원유 (WTI) | 매월 (12개) | 12 | 인도 전월 25일의 3영업일 전 만기 → 다음 날 | ✖ | 1.24 ✖ | Yahoo 그대로 | ≈ +2.9% (남음, EIA 실측) | 유지 · ⚠ 주의 |
| 금속 | 금 | 2·4·6·8·10·12월 (G J M Q V Z) — 상장은 매월, 거래는 짝수월 집중 | 6 | 인도월 직전 (전월 말, 첫 통지일 전) | ✖ | 1.13 ✖ | Yahoo 그대로 | +1.1% (금리 수준) | 유지 |
| 금속 | 은 | 3·5·7·9·12월 (H K N U Z) | 5 | 〃 | ✖ | 1.19 ✖ | Yahoo 그대로 | +1.1% (금리 수준) | 유지 |
| 금속 | 구리 | 3·5·7·9·12월 (H K N U Z) — 상장은 매월 | 5 | 〃 | ✖ | 1.05 ✖ | Yahoo 그대로 | +1.3% (금리 수준) | 유지 |
| 금속 | 백금 | 1·4·7·10월 (F J N V) | 4 | 〃 | ✖ | 1.09 ✖ | Yahoo 그대로 | +1.1% (금리 수준) | 유지 |

월물 코드: F 1월 · G 2월 · H 3월 · J 4월 · K 5월 · M 6월 · N 7월 · Q 8월 · U 9월 · V 10월 · X 11월 · Z 12월. 롤 시점은 거래소 만기·첫 통지일 규칙으로 정한 추정 — Yahoo가 실제로 갈아끼운 날은 공개되지 않아 확인 안 됨. 롤 주기가 짧을수록(에너지 매월) 한 번 갭은 작아도 연간 누적이 커짐.
이음새 갭 = 롤 날 평균 수익률 × 연 롤 횟수 (+ = 수익 과대, − = 과소). '제거됨' = 0 처리로 지운 가짜 수익, '남음' = 지금 자료에 남은 가짜 수익.

**천연가스만 0 처리하면 가치 혼합은?** (별도 스크립트, 2007–) → 천연가스 연 수익률 −1.8% → −15.2%로 바뀌지만, 가치 혼합 Sharpe 0.88 → 0.85 (2017~ 1.07 → 1.08, 차이 연 −0.3%p · t −0.72),
기존 70/30 0.68 → 0.69, 52주 모멘텀 0.49 → 0.57(천연가스 숏 이익이 커짐), 가치 단독 0.42 → 0.30. 사전 선택 검증 1위도 그대로 가치 혼합(학습 0.61 · 시험 1.08) → 결론 불변이라 고정 결과는 재계산하지 않음.

**실제 롤을 한 ETF 수익으로 바꾸면?** (ETF 있는 11개 품목, 2014.11–2026.10) → 가치 혼합 연속월물 0.49 → ETF 0.26 (연 −3.1%p, t −1.65), MinVar 롱 0.25 → −0.06, 52주 모멘텀 0.11 → 0.41.
롱 비중이 큰 전략은 연속월물 기준에서 과대평가될 수 있음 — 19개 기준 0.88 중 0.90 → 0.49는 품목 수(생우·커피·코코아 등 제외), 0.49 → 0.26은 롤 비용 차이. 정확한 확인은 월물별 원자료(Refinitiv 등)로 연속 가격을 직접 만들어야 함.

**대표 전략은 천연가스를 얼마나, 롤 날에 얼마나 들고 있나?** (2007–2026, 주간, 목표 변동성 10%, 별도 스크립트로 비중 재구성)

| 전략 | 평균 비중 (순자산 대비) | 총노출 중 천연가스 | 롱 / 숏 주 비율 | 롤 주 평균 비중 | 롤 날 손익 기여 (연) |
|---|---|---|---|---|---|
| MinVar 롱 | +2.6% | 2.6% | 98% / 0% | +2.6% | +0.22%p |
| 52주 모멘텀 | +0.1% | 2.7% | 49% / 51% | +0.3% | −0.58%p |
| 가치 V1 단독 | +3.7% | 5.8% | 56% / 13% | +3.6% | **+1.93%p** |
| 기존 70/30 | +2.4% | 2.3% | 93% / 7% | +2.5% | −0.01%p |
| **가치 혼합 50/25/25** | **+3.8%** | 3.6% | **90% / 10%** | **+3.8%** | **+0.79%p** |

롤 주 평균 비중 = 천연가스 롤 날(2007년 이후 237번)이 낀 주에 실제로 들고 있던 비중 → 모든 전략에서 평소 비중과 거의 같음(롤 날을 노려 비중이 커지는 패턴 없음).
롤 날 손익 기여 = 롤 날 Yahoo 수익률(평균 +1.14%) × 그 주 천연가스 비중의 합계 — 롱이면 가짜 점프 이익, 숏이면 손해.
재구성 비중으로 다시 계산한 가치 혼합 수익(비용 전 연 10.1%)이 실제 백테스트(비용 후 9.6%)와 맞음 → 재구성 검증.

해석: ① 가치 혼합은 천연가스를 90%의 주에 롱(평균 3.8%, 2013년 10%) — 롤 손실로 계속 내려가 5년 평균보다 늘 싸 보이므로 가치 신호와 MinVar가 꾸준히 롱 → 롤 날 가짜 이익 연 약 0.8%p가 수익 9.6%에 섞여 있음.
② 가치 V1 단독이 영향 최대(연 +1.9%p) — 천연가스 0 처리 시 0.42 → 0.30으로 떨어진 이유와 일치. ③ 모멘텀은 하락 추세에 숏이 많아 오히려 −0.6%p(보정하면 이득).
④ 0 처리 테스트(−0.3%p)와 직접 기여(+0.8%p)가 다른 이유 = 천연가스 가격을 보정하면 신호도 바뀜(모멘텀 숏 증가, 가치 롱 확대) → 일부 상쇄.
정리: 천연가스 비중 자체는 작지만 가치 혼합 수익 중 연 0.3–0.8%p(약 3–8%)는 천연가스 롤 갭 덕분 — Sharpe로 0.03–0.07, 결론 불변이라 재계산하지 않음.

**롤 시점 추정은 맞나? (규칙 날짜 검증)** → 각 품목의 규칙 롤 날짜를 앞뒤 ±5거래일 옮겨 가며 '롤 날 |수익률| ÷ 평상일' 비율이 가장 큰 위치를 찾음 — 0이면 규칙 날짜가 Yahoo의 실제 갈아끼우는 날과 맞는다는 뜻.

| 결과 | 품목 |
|---|---|
| 규칙 날짜(0일)에서 비율 최대 = 맞음 | 옥수수 · 소맥 · 대두 · 대두박 · 대두유 · 설탕 · 커피 · 면화 · 생우 · 돈육 · 휘발유 · 난방유 · 천연가스 · 원유 (14개) |
| ±1일 차이 (비율 차이 0.04–0.09, 사실상 맞음) | 금 · 은 · 구리 (+1일) · 백금 (−1일, 1.73이지만 방향 없음 t −0.3) |
| 확인 불가 | 코코아 — 어느 위치에서도 비율 1.09–1.11로 튀는 날이 없어 날짜를 검증할 수 없음 |

보정(0 처리)을 적용한 9개 품목은 모두 규칙 날짜가 정확히 맞음 → 지금의 0 처리가 엉뚱한 날을 지운 것은 아님.

**보정 안 된 10개 품목의 롤 날이 대표 전략에 준 손익 (2007–, 연 %p)** — 롤 날 Yahoo 수익률(①가짜 점프 + ②진짜 움직임) × 그 주 비중

| 품목 | 롤 날 평균 | MinVar 롱 | 52주 모멘텀 | 가치 V1 | 기존 70/30 | 가치 혼합 50/25/25 |
|---|---|---|---|---|---|---|
| 천연가스 | +1.22% | +0.22 | −0.57 | **+1.97** | −0.01 | **+0.81** |
| 코코아 | +0.48% | +0.14 | +0.03 | +0.08 | +0.16 | +0.18 |
| 커피 | +0.54% | +0.10 | −0.08 | +0.19 | +0.04 | +0.10 |
| 원유 | +0.14% | −0.04 | −0.11 | +0.39 | −0.07 | +0.09 |
| 대두유 | −0.06% | +0.18 | −0.11 | +0.26 | +0.15 | +0.22 |
| 금 · 은 · 구리 · 백금 · 면화 | +0.03–0.33% | +0.07 | −0.09 | +0.04 | +0.13 | +0.10 |
| **합계 (10개)** | | **+0.69** | **−0.94** | **+2.94** | **+0.40** | **+1.51** |

읽는 법: 가치 혼합 연 9.6% 중 약 1.5%p가 보정 안 된 품목의 롤 날에서 나옴(절반 이상이 천연가스). 단 롤 날 수익률엔 진짜 움직임(②)도 섞여 있고, 가격을 보정하면 신호도 바뀌어
실제 효과는 더 작을 수 있음(천연가스 0 처리 실험: −0.3%p). 모멘텀은 숏이 많아 오히려 손해(−0.9%p) → 보정하면 이득. 가치 V1 단독이 가장 민감(+2.9%p).
""")
        from pathlib import Path
        _cal = Path(__file__).parent / "docs" / "롤_이음새_달력_19품목.csv"
        if _cal.exists():
            st.download_button("⬇ 19개 품목 롤(이음새) 날짜 목록 내려받기 (CSV, 2000–2026)", _cal.read_bytes(), file_name=_cal.name, mime="text/csv", key="dl_rollcal",
                               help="품목 · 롤 날짜 · 요일 · 롤 규칙 · 롤 당일 Yahoo 수익률 · 평상일 |수익률| 평균 · 현재 처리(0 처리 / 원자료)")


# ============================================================================
# ★ 구매 타이밍 (실물 구매 의사결정)
# ============================================================================
@st.cache_data(show_spinner=False)
def _proc_inputs(i: str, risk: str, relm: str, roll: bool, extra_bytes):
    import procurement as PR
    return PR.inputs(_levels(extra_bytes, True), _levels(extra_bytes, False), i, risk, relm)


@st.cache_data(show_spinner=False)
def _proc_defer(i: str, w_items: tuple, risk: str, lookback: int, season: str, h: int, sea_feat: bool, roll: bool):
    """주간 체결가·점수 프레임 + h주 앞 하락 확률 (walk-forward, 매년 재학습, 계절 특성 선택)."""
    import procurement as PR
    La_, Lr_ = _levels(None, True), _levels(None, False)
    inp = _proc_inputs(i, risk, "price", roll, None)
    sig = PR.signals(La_[i], lookback, inp["rel"], inp["rel_sign"], inp["vol_fc"], inp["crisis"], season)
    W_ = PR.weekly_frame(Lr_[i], sig, PR.score(sig, dict(w_items)))
    return W_, PR.fc_walkforward(W_, h, season=sea_feat)


@st.cache_data(show_spinner=False)
def _defer_ev(tau: float, p_star: float, auc_min: float):
    import experiments as EX
    return EX.defer_evidence(None, auc_min, tau, p_star)


def _place_labels(xs, ys, texts, W=620, H=300):
    """산점도 라벨 겹침 회피: 점마다 주변 후보 위치(가까운 것부터) 중 다른 라벨·점과 겹치지 않는 곳을 고름 → (ax, ay) 픽셀 오프셋."""
    P = [(x / 100 * W, (1 - y / 100) * H) for x, y in zip(xs, ys)]
    tw = lambda t: sum(11.5 if ord(c) > 0x2E80 else 6.8 for c in t) + 4
    boxes, out = [], [None] * len(P)

    def ov(a, b):
        return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    dots = [(x - 8, y - 8, x + 8, y + 8) for x, y in P]
    for k in sorted(range(len(P)), key=lambda k: (P[k][1], P[k][0])):
        x0, y0 = P[k]; w, h = tw(texts[k]), 14
        best = None
        for r in (16, 28, 42, 58, 76, 96):
            for ang in (-90, 90, 0, 180, -35, -145, 35, 145):
                c, s_ = np.cos(np.radians(ang)), np.sin(np.radians(ang))
                cx, cy = x0 + c * (r + w / 2 * abs(c)), y0 + s_ * r
                b = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                cost = sum(ov(b, q) for q in boxes) + sum(ov(b, q) for q in dots) + (500 if b[0] < 0 or b[2] > W or b[1] < 0 or b[3] > H else 0)
                cost += r * 0.5
                if best is None or cost < best[0]:
                    best = (cost, cx - x0, cy - y0, b)
                if cost <= r * 0.5:
                    break
            if best[0] <= r * 0.5:
                break
        boxes.append(best[3]); out[k] = (best[1], best[2])
    return out


MACRO_ROWS = [("금리", "fred:DTB3", "미 3개월 국채", "↑ = 자금·보유 비용↑ (미루는 동안의 비용↑)"),
              ("금리", "fred:DGS2", "미 2년 국채", "연준 정책 기대"),
              ("금리", "fred:DGS10", "미 10년 국채", "장기 자금 비용"),
              ("금리", "spread", "장단기 금리차 (10Y−2Y)", "음수(역전) = 경기 둔화 신호 → 수요 약화"),
              ("금리", "fred:T10YIE", "10년 기대인플레", "↑ = 원자재 전반 상승 압력"),
              ("금리", "fred:BAA10Y", "신용 스프레드 (Baa−10Y)", "↑ = 신용 경색·위험 회피"),
              ("달러·환율", "yf:DX-Y.NYB", "달러 인덱스 (DXY)", "달러↑ = 달러 표시 원자재 약세 압력"),
              ("달러·환율", "fred:DTWEXBGS", "달러 지수 (광의)", "교역 가중 달러 가치"),
              ("달러·환율", "yf:KRW=X", "원/달러", "↑ = 원화 기준 매입가↑"),
              ("달러·환율", "yf:CNY=X", "위안/달러", "↑(위안 약세) = 중국 수입 수요 약화"),
              ("위험·경기", "fred:VIXCLS", "VIX", "↑ = 위험 회피·변동성 확대"),
              ("위험·경기", "yf:^GSPC", "S&P 500", "경기·위험 선호"),
              ("해운", "ak:macro_shipping_bdi", "BDI (건화물 종합)", "↑ = 건화물 운임↑ → 도착 원가↑"),
              ("해운", "ak:macro_shipping_bpi", "BPI (파나막스)", "곡물 파나막스 운임"),
              ("해운", "ak:macro_shipping_bci", "BCI (케이프사이즈)", "철광석·석탄 운임"),
              ("에너지", "fred:DCOILBRENTEU", "브렌트 현물", "↑ = 비료·운송·바이오연료 비용↑")]


@st.cache_data(show_spinner=False)
def _macro_table() -> pd.DataFrame:
    L = _levels(None, False)
    rows = []
    for grp, i, nm, note in MACRO_ROWS:
        if i == "spread":
            if not {"fred:DGS10", "fred:DGS2"} <= set(L.columns):
                continue
            x = (L["fred:DGS10"] - L["fred:DGS2"]).dropna(); rate = True
        else:
            if i not in L:
                continue
            x = L[i].dropna(); rate = A.is_rate(UNITS.get(i, ""))
        if len(x) < 30:
            continue
        t = x.index[-1]; v = float(x.iloc[-1])
        ago = lambda d: float(x.loc[:t - pd.Timedelta(days=d)].iloc[-1]) if len(x.loc[:t - pd.Timedelta(days=d)]) else np.nan
        chg = (lambda a: v - a) if rate else (lambda a: (v / a - 1) * 100)
        x5 = x.loc[t - pd.DateOffset(years=5):]
        rows.append({"구분": grp, "지표": nm, "기준일": t.date(), "최근값": v, "단위": "%" if rate else UNITS.get(i, ""),
                     "1주 변화": chg(ago(7)), "4주 변화": chg(ago(28)), "1년 변화": chg(ago(365)), "변화 단위": "%p" if rate else "%",
                     "5년 분위": float((x5 < v).mean()), "구매 관점": note})
    return pd.DataFrame(rows).set_index(["구분", "지표"])


def _defer_plus_view(dp_):
    """구매 탭 보완 실험 공용: 변수 묶음별 학습·시험 AUC 요약 + 품목별 시험 AUC (학습 1위 묶음 선택 포함)."""
    tr_ = dp_[dp_.기간 == "학습"].pivot_table(index="품목", columns="변수묶음", values="AUC")
    te_ = dp_[dp_.기간 == "시험"].pivot_table(index="품목", columns="변수묶음", values="AUC")
    base_ = "기본 (현재)"
    order_ = [base_] + [c for c in te_.columns if c != base_]
    pick_ = tr_.idxmax(1)
    sel_ = pd.Series([te_.loc[i, pick_[i]] for i in te_.index], index=te_.index)
    summ_ = []
    for c in order_:
        d_ = te_[c] - te_[base_]
        summ_.append({"변수 묶음": c, "학습 AUC 평균": tr_[c].mean(), "시험 AUC 평균": te_[c].mean(), "기본 대비 시험 차": d_.mean(),
                      "개선 품목 수": f"{int((d_ > 0).sum())} / {len(d_)}", "차이 t값": d_.mean() / d_.std() * np.sqrt(len(d_)) if d_.std() > 0 else np.nan})
    d_ = sel_ - te_[base_]
    summ_.append({"변수 묶음": "품목마다 학습 AUC 1위 묶음 선택", "학습 AUC 평균": tr_.max(1).mean(), "시험 AUC 평균": sel_.mean(), "기본 대비 시험 차": d_.mean(),
                  "개선 품목 수": f"{int((d_ > 0).sum())} / {len(d_)}", "차이 t값": d_.mean() / d_.std() * np.sqrt(len(d_))})
    st.dataframe(pd.DataFrame(summ_).set_index("변수 묶음").style.format({"학습 AUC 평균": "{:.3f}", "시험 AUC 평균": "{:.3f}", "기본 대비 시험 차": "{:+.4f}",
                                                                        "차이 t값": "{:+.2f}"}, na_rep="–"))
    show_ = te_[order_].copy(); show_["학습으로 고른 묶음"] = pick_.map(lambda c: c.split(" (")[0])
    st.dataframe(show_.style.format({c: "{:.3f}" for c in order_}), height=min(36 * (len(show_) + 1) + 4, 470))


def tab_procure():
    import procurement as PR
    st.subheader("구매 의사결정 — 지금 살까, 미룰까")
    st.caption("매주 금요일 판단 · 🟢 점수 ≥ τ → 적극 매수 · 🔴 점수 < τ 이고 더 쌀 확률 ≥ p* (AUC 기준 충족 품목) → 보류 · 🟠/🟡 그 외 → 판단 유보 (매수 쪽/대기 쪽)  \n"
               "점수·확률은 그 시점까지 데이터만 사용 · 체결 = 다음 거래일 종가 · 미루는 기간·특성은 학습(2008–2016)으로 고르고 시험(2017–)으로 확인")
    ids_av = [i for i in PR.BUY_IDS if i in NAME]
    c1_, c2_, c3_ = st.columns([2, 1.6, 0.8])
    ids = c1_.multiselect(f"구매 품목 ({len(ids_av)}개 중 선택)", ids_av, default=[i for i in ["yf:ZC=F", "yf:ZW=F", "yf:ZS=F", "yf:ZM=F"] if i in ids_av],
                          format_func=lambda i: PR.BUY_IDS[i] + ("" if i in PR.REL_DEF else " · 상대가치 없음"), key="pc_ids",
                          help="곡물·철강 원료·에너지·귀금속·소프트·축산·DCE 전 품목. '상대가치 없음' = 검증을 통과한 대체재·원료 관계가 없어 "
                               "가격 위치·반등·계절성으로만 판단 (⑥ 실험 결과 '전 품목 구매 타이밍')")
    adv_p = c3_.checkbox("고급 프리셋", False, key="pc_adv",
                         help="보수적·공격적은 단계별 검증에서 '추천'보다 절감이 작았던 비교용 설정 (⑥ 실험 결과 → 구매 타이밍)")
    preset = c2_.selectbox("전략 프리셋", list(PR.PRESETS) if adv_p else PR.BASIC_PRESETS, 0, key="pc_preset",
                           help="추천: 단계별 검증에서 전·후반 모두 균등 매수보다 나았고 꼬리 위험도 가장 작은 조합. "
                                "균형: 싸면서 반등을 시작할 때 (상대가치 데이터 없이 가격만으로). "
                                "고급 — 보수적: 계절성 추가, 공격적: 가격 위치만 2배 (꼬리 위험 큼)")
    P0 = PR.PRESETS[preset]
    with st.expander("신호 · 위험 판단 · 기간 설정 (프리셋 값에서 조정)", expanded=False):
        e = st.columns(5)
        w = {c: e[j].slider(PR.COMP_LBL[c], 0.0, 1.0, float(P0["w"].get(c, 0.0)), 0.05, key=f"pc_w_{c}_{preset}") for j, c in enumerate(PR.COMP_LBL)}
        g = st.columns(5)
        tau = g[0].slider("가격 매력 문턱 τ (점수)", 0.40, 0.90, float(PR.DEFER_TAU), 0.05, key=f"pc_tau_{preset}",
                          help="점수가 이 값 이상이면 '지금 살 만한 가격' → 미루지 않음. 기본값은 ⑥ '미루기 권고' 실험에서 학습 기간(2008–2016)으로 고른 값")
        k = float(P0["k"])
        risk_l = g[1].selectbox("변동성·위험 추정", list(PR.RISK_MODES), list(PR.RISK_MODES.values()).index(P0["risk"]), key=f"pc_risk_{preset}",
                                help="4주 가격 범위·위험 표시에 쓰는 변동성. MIDAS = 일간 실현분산 시차로 예측, HMM = 고변동 레짐 필터 확률 (모두 walk-forward)")
        lb_l = g[2].selectbox("가격 위치 창", ["품목별 자동", "6개월", "1년", "2년"], 0, key=f"pc_lb_{preset}",
                              help="품목별 자동 = 대두박·밀·면화·팔라듐·원당·금 등 9품목은 1년, 나머지는 6개월 (⑥ '벌크 구매' 검증에서 전·후반 모두 나은 쪽)")
        sea_l = g[3].selectbox("계절성 산정", list(PR.SEASON_MODES), list(PR.SEASON_MODES.values()).index(P0["season"]), key=f"pc_sea_{preset}",
                               help="남미 작황 확대 등으로 수확기 패턴이 이동 → 최근 가중이 전체 평균보다 소폭 우수")
        st.caption("12개월 추세는 롱숏 전략에선 유효했지만 구매 시점 선택에서는 '오를 때 서둘러 사기'가 평균 −20–−50bp로 손해 → 기본 0. "
                   "상대가치는 경제적 관계(옥수수↔밀 가격비, 대두 크러시 마진, 대두박·대두유 가치/대두, 철광석 SGX↔DCE 수입 차익, "
                   "점결탄↔철근 고로 마진, 구리↔금)의 1년 분위 — 대두유는 바이오연료 정책으로 관계 자체가 이동해 효과가 약하고, "
                   "구리는 대체재(알루미늄) 관계가 최근 기각돼 구리/금만 참고 수준. 점결탄은 DCE 위안 가격을 그날 환율로 달러 환산(증치세 포함), "
                   "철강 원료 3품목은 데이터가 2010–2013년부터라 검증 기간이 짧음. "
                   "확장 품목(에너지·귀금속·소프트·축산·DCE 등)은 후보 관계 중 두 기간 개선 + 무관한 가격비(위약)·자기 가격 대조군을 모두 넘은 것만 채택"
                   "(가스/원유, 생우/송아지, 알루미늄/구리, 브렌트 경유 크랙 등 10품목), "
                   "통과 관계가 없는 품목은 상대가치 없이 가격 위치·반등·계절로만 판단 → ⑥ 실험 결과 '전 품목 구매 타이밍'.")
    risk = PR.RISK_MODES[risk_l]; season = PR.SEASON_MODES[sea_l]
    with st.expander("판단 기준 — 품목마다 어떤 설명변수로 판단하나", expanded=False):
        wt = lambda c: f"{w.get(c, 0):.0%}" if w.get(c, 0) > 0 else "참고 (0%)"
        st.markdown(f"""
**① 가격 매력 점수 (0–100, 🟢 판단 · 지도 가로축)** — 구성요소 가중 평균 (현재 프리셋 가중치, 값이 없는 요소는 빼고 재배분)

| 구성요소 | 내용 | 가중치 |
|---|---|---|
| 가격 위치 (저점) | 최근 6개월–1년 가격 범위 중 지금이 얼마나 낮은지 | {wt("value")} |
| 단기 반등 | 최근 10일 동안 올랐는지 (저점 통과 확인) | {wt("turn")} |
| 상대가치 | 대체재·원료 대비 싼지 (크러시 마진, 옥수수/밀 비율 등, 관계가 검증된 품목만) | {wt("relval")} |
| 계절성 | 과거 연도들에서 이 달이 평균적으로 쌌는지 (올해 이전 연도만) | {wt("season")} |
| 12개월 추세 | 구매 시점 선택에선 손해 → 참고 | {wt("trend")} |

**② 미루는 기간 안에 더 쌀 확률 (🔴 판단 · 지도 세로축)** — 로지스틱 회귀, 매년 1월 그 이전 자료로만 재학습

| 묶음 | 변수 | 의미 |
|---|---|---|
| 매수 점수 | 점수, 점수 변화(전주 대비) | 가격 매력도 종합과 그 변화 |
| 점수 구성요소 | 가격 위치 · 단기 반등 · 상대가치 · 계절성 점수 | 위 ①과 같은 값 |
| | 변동성 분위 | 지금 변동성이 과거 대비 높은지 |
| 최근 가격 흐름 | 1주·4주·13주 수익률 | 단기 반전, 중기 추세 |
| | 13주 변동성 | 최근 출렁임 크기 |
| (선택) 연중 주차 | 사인·코사인 4개 | 연중 몇째 주인가 — 휘발유 드라이빙 시즌, 축산 사이클 같은 달력 패턴 |

미루는 기간(1–4주)·연중 주차 사용 여부는 품목마다 학습 기간 AUC가 가장 높은 조합 · 펀더멘털(WASDE·COT)·거시 지표는 검증에서 예측력이 재현되지 않아 **참고 표시만**.
""")
    lb_of = (lambda i: PR.item_lookback(i, int(P0["lookback"]))) if lb_l == "품목별 자동" else (lambda i, v={"6개월": 126, "1년": 252, "2년": 504}[lb_l]: v)
    if not ids:
        st.info("품목을 선택하세요."); return
    La_, Lr_ = _levels(extra_bytes, True), _levels(extra_bytes, False)

    def sig_of(i):
        inp = _proc_inputs(i, risk, "price", ROLL, extra_bytes)
        return PR.signals(La_[i], lb_of(i), inp["rel"], inp["rel_sign"], inp["vol_fc"], inp["crisis"], season), inp

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
                     "4주 하단 90%": px * b_lo, "4주 상단 90%": px * b_hi, "가격 위치 창": f"{lb_of(i) // 21}개월"})
    T = pd.DataFrame(rows)
    import experiments as EX
    st.markdown(f"##### 이번 주 판단 — 지금 살까, 미룰까? (기준일 {T['기준일'].max()} · 가격 매력 문턱 {tau:.2f})")
    cq1, cq2 = st.columns(2)
    p_star = cq1.slider("미루기 기준 p* — 미루는 기간 안에 지금보다 쌀 확률 ≥", 0.50, 0.80, float(PR.DEFER_PSTAR), 0.05, key="pc_pstar",
                        help="높일수록 확신이 클 때만 미룸(미루는 횟수 ↓). 기본값 = 학습 기간(2008–2016)에 판단당 절감이 가장 컸던 값")
    auc_min = cq2.slider("미루기 권고 활성화 — 학습 기간 하락 예측 AUC >", 0.50, 0.60, float(EX.DEFER_AUC_MIN), 0.01, key="pc_aucmin",
                         help="품목마다 8개 조합(미루는 기간 1–4주 × 계절 특성 유무) 중 학습 AUC가 가장 높은 것을 고르므로, 예측력이 전혀 없어도 "
                              "최댓값이 0.53을 넘을 확률이 67%(0.55는 22%) → 기본 0.55. 이 기준을 못 넘는 품목은 '설정 AUC 기준치 미달' — 🔴 보류 판단을 하지 않고 🟢/🟡만 표시")
    g_tau = min((0.5, 0.6), key=lambda v: abs(v - tau)); g_p = min((0.5, 0.55, 0.6, 0.65), key=lambda v: abs(v - p_star))
    ev = _defer_ev(g_tau, g_p, float(auc_min))
    if ev.empty:
        st.warning("⑥ 실험 결과 '미루기 권고' 결과가 없습니다 — ⑥에서 재실행하면 품목별 근거(AUC·시험 성과)가 표시됩니다.")
    elif (g_tau, g_p) != (round(tau, 2), round(p_star, 2)):
        st.caption(f"근거표(과거 AUC·시험 성과)는 실험 격자의 가장 가까운 설정(τ {g_tau:.2f}, p* {g_p:.2f}) 기준 — 아래 '과거 성과 검증'은 지금 설정 그대로 다시 계산.")
    tl = tau * 100; yb = max(tl - 10, 0)
    box_ = lambda col, bg, t1, t2, t3: (f'<div style="flex:1 1 170px;border-left:5px solid {col};background:{bg};padding:6px 10px;border-radius:6px;'
                                        f'font-size:0.86rem;line-height:1.45"><b>{t1}</b><br>{t2}<br><span style="opacity:0.75">{t3}</span></div>')
    st.markdown('<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:4px">'
                + box_("#2ea043", "rgba(46,160,67,0.12)", "🟢 적극 매수", f"점수 {tl:.0f}–100", "국소 저점권 → 지금 매수")
                + box_("#f0883e", "rgba(240,136,62,0.14)", "🟠 판단 유보 · 매수 쪽", f"점수 {yb:.0f}–{tl - 1:.0f} · 확률 &lt; {p_star:.0%}", "문턱 근접 → 필요하면 매수")
                + box_("#b07c10", "rgba(210,153,34,0.20)", "🟡 판단 유보 · 대기 쪽", f"점수 0–{yb - 1:.0f} · 확률 &lt; {p_star:.0%}", "비싼 편 → 급하지 않으면 다음 주 재확인")
                + box_("#da3633", "rgba(218,54,51,0.12)", "🔴 매수 보류", f"점수 0–{tl - 1:.0f} · 확률 ≥ {p_star:.0%}", f"AUC &gt; {auc_min:.2f} 품목만 → 1–4주 미루기")
                + "</div>", unsafe_allow_html=True)
    st.caption(f"확률 = 미루는 기간 안에 지금보다 쌀 확률 · 설정 AUC 기준치({auc_min:.2f}) 미달 품목은 확률과 무관하게 🟢/🟡만 (지도의 빈 원) · "
               "🏷️ 횡단면상 매력 묶음 = '횡단면' 하위 탭에서 19개 품목을 군집했을 때 가격 매력 점수 평균이 가장 높은 묶음(군집 1)에 속한 품목")
    try:
        cs1, cs1_sc, cs1_k, cs1_n, cs1_pt = _cs_cluster1(ROLL)
    except Exception:
        cs1, cs1_sc, cs1_pt = set(), np.nan, {}
    ncol = min(len(T), 4)
    plan, cfg, pts = [], {}, []
    for j, r in T.iterrows():
        if j % ncol == 0:
            qc = st.columns(ncol)
        box = qc[j % ncol].container(border=True)
        i = r["id"]; u = "온스" if r["단위"] == "$/oz" else "톤"; s_ = r["매수 점수"]
        e_ = ev.loc[i] if (len(ev) and i in ev.index) else None
        h = int(e_["미룸주"]) if e_ is not None else PR.DEFER_WAIT
        seaf = bool(e_["계절특성"]) if e_ is not None else False
        active = bool(e_ is not None and np.isfinite(e_["학습AUC"]) and e_["학습AUC"] > auc_min)
        cfg[i] = (h, seaf, active)
        qty = box.number_input(f"{r['품목']} 카고 ({u})", 0.0, 1e8, 30000.0 if u == "톤" else 10000.0, 5000.0 if u == "톤" else 1000.0, key=f"pc_q_{i}")
        W_, F = _proc_defer(i, tuple(sorted(w.items())), risk, lb_of(i), season, h, seaf, ROLL)
        pdn = float(F["p_down"].iloc[-1]) if len(F) and np.isfinite(F["p_down"].iloc[-1]) else np.nan
        if s_ >= tl:
            dec = "🟢 적극 매수"
        elif active and np.isfinite(pdn) and pdn >= p_star:
            dec = f"🔴 매수 보류 ({h}주)"
        else:
            dec = "🟠 판단 유보 · 매수 쪽" if s_ >= yb else "🟡 판단 유보 · 대기 쪽"
        box.metric(dec, f"${qty * r['가격'] / 1e6:,.2f}M", f"{qty:,.0f} {u} × ${r['가격']:,.1f}{r['단위'][1:]}", delta_color="off")
        srow = sigs[i].loc[sigs[i].dropna(subset=["value"]).index.max()]
        ew = PR.eff_weights(srow, w)
        down, up = qty * (r["4주 하단 90%"] - r["가격"]), qty * (r["4주 상단 90%"] - r["가격"])
        pr_txt = f"{pdn:.0%}" if np.isfinite(pdn) else "–"
        short_r = PR.reason(srow, w).replace(" (유리)", "↑").replace(" (불리)", "↓").replace(" (보통)", "").replace("단기 반등", "반등").replace("가격 위치", "위치")
        pz = La_[i].dropna(); p0 = pz.asof(pz.index[-1] - pd.Timedelta(days=364))
        r52 = float(np.log(pz.iloc[-1] / p0) * 100) if np.isfinite(p0) and p0 > 0 else np.nan
        trend = ("" if not np.isfinite(r52) else
                 f"📈 **추세 상승 중** (52주 {r52:+.{0 if abs(r52) >= 1 else 1}f}%)" + (" — 상승 추세에서 보류는 신중" if dec.startswith("🔴") else "") if r52 > 0
                 else f"📉 추세 하락 중 (52주 {r52:+.{0 if abs(r52) >= 1 else 1}f}%)")
        lines = [f"점수 **{s_:.0f}**/{tl:.0f} · {h}주 내 쌀 확률 **{pr_txt}**/{p_star:.0%}", f"근거: {short_r}"] + ([trend] if trend else [])
        if i in cs1:
            pt_ = cs1_pt.get(i)
            lines.insert(0, f"🏷️ **횡단면상 매력 묶음** (19개 품목 중 군집 1 · 묶음 평균 {cs1_sc:.0f}점)"
                         + (" · 같이 사면 분산 효과 큰 묶음 내 품목(현재 상관): " + ", ".join(f"{PR.BUY_IDS.get(j, short(j))} {v:+.2f}" for j, v in pt_.items())
                            if pt_ is not None and len(pt_) else ""))
        if e_ is not None:
            lines.append(f"학습 AUC {e_['학습AUC']:.2f} · 시험 AUC {e_['시험AUC']:.2f}" + ("" if active else " · **기준치 미달**"))
            lines.append(f"시험 {e_['시험절감bp']:+.1f}bp (무작위 {e_['시험위약bp']:+.1f}) · {h}주{'·계절' if seaf else ''}")
        else:
            lines.append("학습 AUC – · 시험 AUC – (검증 대상 밖 → 가격 매력도만)")
        lines.append(f"4주 범위 {down / 1e6:+,.2f}M – {up / 1e6:+,.2f}M $" + ("" if "relval" in ew or w.get("relval", 0) == 0 else " · 상대가치 없음 (가중치 재배분)"))
        box.caption("  \n".join(lines))
        plan.append({"품목": r["품목"], "판단": dec, "매수 점수": s_, "문턱": tl, "미루는 기간 (주)": h, "계절 특성": "사용" if seaf else "–",
                     "더 쌀 확률": pdn, "학습 AUC": e_["학습AUC"] if e_ is not None else np.nan, "시험 AUC": e_["시험AUC"] if e_ is not None else np.nan,
                     "시험 절감 (bp)": e_["시험절감bp"] if e_ is not None else np.nan, "시험 위약 (bp)": e_["시험위약bp"] if e_ is not None else np.nan,
                     "보류 판단": "적용" if active else "설정 AUC 기준치 미달", "점수 근거": PR.reason(srow, w)})
        tr = pd.DataFrame({"x": W_["score"] * 100, "y": F["p_down"].reindex(W_.index) * 100}).dropna().tail(5)
        pts.append(dict(품목=r["품목"], 점수=s_, 확률=pdn * 100 if np.isfinite(pdn) else 50.0, 적용=active, 판단=dec, tr=tr))
    if pts:
        Pm = pd.DataFrame(pts)
        show_tr = st.checkbox("지난 4주 경로 표시 (점선 꼬리 = 4주 전 → 지금)", len(Pm) <= 8, key="pc_trail")
        col_of = lambda d: "#2ea043" if d.startswith("🟢") else ("#da3633" if d.startswith("🔴") else ("#f0883e" if d.startswith("🟠") else "#d29922"))
        f = go.Figure()
        f.add_shape(type="rect", x0=tl, x1=100, y0=0, y1=100, fillcolor="rgba(46,160,67,0.16)", line_width=0, layer="below")
        f.add_shape(type="rect", x0=0, x1=tl, y0=p_star * 100, y1=100, fillcolor="rgba(218,54,51,0.16)", line_width=0, layer="below")
        f.add_shape(type="rect", x0=yb, x1=tl, y0=0, y1=p_star * 100, fillcolor="rgba(240,136,62,0.18)", line_width=0, layer="below")
        f.add_shape(type="rect", x0=0, x1=yb, y0=0, y1=p_star * 100, fillcolor="rgba(210,153,34,0.22)", line_width=0, layer="below")
        for x_, y_, t_ in [(tl + (100 - tl) / 2, 96, "🟢 적극 매수"), (tl / 2, 96, "🔴 매수 보류"), (yb / 2, 4, "🟡 대기 쪽"), ((yb + tl) / 2, 4, "🟠 매수 쪽")]:
            f.add_annotation(x=x_, y=y_, text=t_, showarrow=False, font=dict(size=12), opacity=0.8)
        if show_tr:
            for _, q in Pm.iterrows():
                if len(q["tr"]) >= 2:
                    f.add_trace(go.Scatter(x=list(q["tr"]["x"]) + [q["점수"]], y=list(q["tr"]["y"]) + [q["확률"]], mode="lines+markers",
                                           line=dict(color=col_of(q["판단"]), width=1, dash="dot"), marker=dict(size=4), opacity=0.55,
                                           showlegend=False, hoverinfo="skip"))
        for ok_, nm_, sym in [(True, "AUC 기준 충족 (🔴 가능)", "circle"), (False, "AUC 기준치 미달 (🟢/🟡만)", "circle-open")]:
            q = Pm[Pm["적용"] == ok_]
            if len(q):
                f.add_trace(go.Scatter(x=q["점수"], y=q["확률"], mode="markers", text=q["품목"], showlegend=False, customdata=q["판단"],
                                       marker=dict(size=12, symbol=sym, color=[col_of(d) for d in q["판단"]], line=dict(width=2, color=[col_of(d) for d in q["판단"]])),
                                       hovertemplate="%{text} · %{customdata}<br>점수 %{x:.0f} · 더 쌀 확률 %{y:.0f}%<extra></extra>"))
            f.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=nm_, marker=dict(size=11, symbol=sym, color="#898781", line=dict(width=2, color="#898781"))))
        for (dx, dy), (_, q) in zip(_place_labels(Pm["점수"], Pm["확률"], list(Pm["품목"])), Pm.iterrows()):
            f.add_annotation(x=q["점수"], y=q["확률"], text=q["품목"], ax=dx, ay=dy, showarrow=True, arrowwidth=0.8, arrowhead=0,
                             arrowcolor="#898781", standoff=6, font=dict(size=11))
        st.plotly_chart(fig_layout(f, 420, title=f"이번 주 위치 — 가격 매력 점수 × 더 쌀 확률 (τ {tl:.0f}점 · p* {p_star:.0%})",
                                   xaxis=dict(title="가격 매력 점수 (높을수록 쌈)", range=[0, 100]), yaxis=dict(title="더 쌀 확률 (%)", range=[0, 100]),
                                   hovermode="closest"), key="pc_map")
    with st.expander("판단 요약표 (품목별 점수·확률·근거)"):
        st.dataframe(pd.DataFrame(plan).set_index("품목").style.format(
            {"매수 점수": "{:.0f}", "문턱": "{:.0f}", "더 쌀 확률": "{:.0%}", "학습 AUC": "{:.3f}", "시험 AUC": "{:.3f}", "시험 절감 (bp)": "{:+.1f}",
             "시험 위약 (bp)": "{:+.1f}"}, na_rep="–"))
        st.caption("시험 절감 = 2017년 이후 이 규칙대로 했을 때 '바로 산 경우' 대비 판단 1회당 평균 매입가 차이 · 위약 = 같은 비율을 무작위로 미뤘을 때")
    with st.expander(f"품목별 '더 쌀 확률' 예측 정확도 (AUC) — 전체 {len(ev)}품목, 학습 2008–2016 · 시험 2017–"):
        if len(ev):
            A_ = ev.copy()
            A_["보류 판단"] = np.where(A_["학습AUC"] > auc_min, "✅ 적용", "– 설정 AUC 기준치 미달")
            A_["시험 판정"] = np.where((A_["시험AUC"] > 0.53) & (A_["시험절감bp"] > A_["시험위약bp"]), "✅ 시험에서도 통함",
                                      np.where(A_["시험AUC"] < 0.5, "⚠️ 시험에서 동전 던지기 이하", "– 약함"))
            A_["미루는 기간"] = A_["미룸주"].astype(str) + "주" + np.where(A_["계절특성"], " · 계절", "")
            from math import erfc, sqrt
            n_te = max((pd.Timestamp(T["기준일"].max()) - pd.Timestamp("2017-01-01")).days / 7, 10)

            def _p_mw(auc, h_):
                n = n_te / max(int(h_), 1)               # h주 겹치는 목표 → 유효 표본 n/h
                se = sqrt((n + 1) / (12 * (n / 2) ** 2))  # 귀무(AUC 0.5)에서 Mann-Whitney AUC 표준오차
                return 0.5 * erfc((auc - 0.5) / se / sqrt(2)) if np.isfinite(auc) else np.nan
            A_["시험 p값"] = [_p_mw(a, h_) for a, h_ in zip(A_["시험AUC"], A_["미룸주"])]
            A_ = A_.sort_values("학습AUC", ascending=False)[["품목", "군", "미루는 기간", "학습AUC", "시험AUC", "시험절감bp", "시험위약bp", "시험미룸비율",
                                                          "시험적중률", "시험 p값", "보류 판단", "시험 판정"]]
            st.dataframe(A_.rename(columns={"학습AUC": "학습 AUC", "시험AUC": "시험 AUC", "시험절감bp": "시험 판단당 절감 (bp)", "시험위약bp": "무작위 미룸 (bp)",
                                            "시험미룸비율": "시험 미룬 비율", "시험적중률": "미룬 경우 적중률"}).set_index("품목").style.format(
                {"학습 AUC": "{:.3f}", "시험 AUC": "{:.3f}", "시험 판단당 절감 (bp)": "{:+.1f}", "무작위 미룸 (bp)": "{:+.1f}", "시험 미룬 비율": "{:.0%}",
                 "미룬 경우 적중률": "{:.0%}", "시험 p값": "{:.3f}"}, na_rep="–"), height=min(38 * (len(A_) + 1), 980))
            st.caption(f"AUC = 과거에 '더 쌀 확률'이 높게 나온 주에 실제로 더 쌌는지의 순위 정확도 (0.5 = 동전 던지기, 1.0 = 완벽). 미루는 기간·계절 특성은 품목마다 "
                       f"학습 기간 AUC가 가장 높은 조합으로 고르고(시험 기간은 보지 않음), 학습 AUC > {auc_min:.2f}인 품목만 미루기 권고. "
                       f"절감·위약은 τ {g_tau:.2f}, p* {g_p:.2f} 기준. '시험에서도 통함' = 시험 AUC > 0.53이고 무작위 미룸보다 절감이 큼. "
                       "시험 p값 = 'AUC가 0.5(동전)보다 큰가'의 한쪽 Mann-Whitney 근사 검정 (h주 목표가 겹치므로 표본을 1/h로 줄여 보수적으로 계산) — 0.05 미만이면 우연으로 보기 어려움.")
        else:
            st.info("⑥ 실험 결과 '미루기 권고'를 재실행하면 표시됩니다.")
    bl1_, bl2_ = st.columns([4, 1])
    bl1_.caption("📋 **검증 백로그** — 지금 기준은 그대로 유지. 다른 탭의 방법론(가치·추세·칼만·레짐·상관·멀티모달·④ 예측 모델·대안 시계열)을 '더 쌀 확률'에 넣어 "
                 "같은 학습(2008–16)/시험(2017–) 틀로 검증했으나 시험 AUC 개선이 +0.01 안팎(t < 2)이라 채택하지 않음. 시험한 아이디어 전체·결과·판단은 아래 두 펼치기와 엑셀 백로그.")
    from pathlib import Path
    _bl = Path(__file__).parent / "docs" / "구매판단_검증_백로그.xlsx"
    if _bl.exists():
        bl2_.download_button("백로그 엑셀", _bl.read_bytes(), file_name=_bl.name, key="proc_backlog",
                             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    with st.expander("보완 실험 — 롱숏 연구의 가치·추세·칼만·레짐 신호를 '더 쌀 확률'에 넣으면 AUC가 오르나"):
        dp_, _m = EX.load("defer_plus")
        if dp_ is None:
            st.info("⑥ 실험 결과가 없습니다 — experiments.defer_plus() 실행 필요 (약 2분).")
        else:
            _defer_plus_view(dp_)
            st.caption("품목마다 기존 '미루기' 실험이 고른 미루는 기간·계절 특성은 그대로 두고 설명변수만 추가. 가치 = 5년 평균 대비 위치·19개 품목 중 순위·섹터 평균 대비 뒤처짐, "
                       "추세 = 52주 수익률·칼만 기울기 t, 레짐 = 다변량 HMM 스트레스 확률·VIX 3년 분위·흡수비율 급등 z·바스켓 13주 수익(모두 그 주까지 정보). "
                       "결과: 평균 AUC +0.01 안팎(0.544 → 0.555), 품목별로 학습 1위 묶음을 고르면 +0.012(19품목 중 11개 개선, t 2.07) — 개선은 있으나 작음. "
                       "레짐 신호가 가장 자주 선택(8품목). 품목 하나의 AUC 표준오차가 약 0.03이라 개별 품목의 ±0.02 변화는 잡음 범위.")
    with st.expander("보완 실험 2 — 상관·멀티모달(뉴스 제외)·예측 모델(④ 그리드) 신호를 넣으면 AUC가 오르나"):
        dp2_, _m2 = EX.load("defer_plus2")
        if dp2_ is None:
            st.info("⑥ 실험 결과가 없습니다 — experiments.defer_plus2() 실행 필요 (약 15분).")
        else:
            _defer_plus_view(dp2_)
            st.caption("같은 틀(품목별 미룸 주·계절 특성 고정, 학습 2008–16 / 시험 2017–). 상관 = 품목-바스켓 52주 상관·흡수비율 수준, "
                       "멀티모달 = 7갈래 요약(텍스트 FinBERT·리포트·COT·기상·매크로·금융·월간 거시, 품목별 과거 표준화 평균, GDELT 뉴스 제외), "
                       "예측 모델 = ④ 전략 백테스트의 표본 외 4주 예측(Lasso·Elastic Net·패널 Elastic Net·MIDAS·경량 Neural ODE·PatchTST·Chronos-Bolt, 과거 표준화). "
                       "결과: 시험 AUC 평균 기본 0.544 → 레짐 0.551 · 상관 0.550(11/19 개선, t 1.86) · 예측 모델 0.548 · 멀티모달 0.548 · 전체 0.544. "
                       "품목마다 학습 1위 묶음을 고르면 0.552(9/19 개선, t 1.26) — 어느 묶음도 기본보다 확실히 낫지 않아 판단 기준은 그대로 둠.")
    with st.expander("신호 구성 상세표 (구성요소 점수 · 위험 · 4주 범위)"):
        show = T.drop(columns=["id"]).set_index("품목")
        st.dataframe(show.style.format({c: "{:,.1f}" for c in show.columns if show[c].dtype.kind == "f"}))
        st.caption(f"점수 = 구성 요소의 가중 평균 (0–100, 가중치 0인 요소는 참고 표시, 값이 없는 요소는 빼고 재배분). 위험 지표 = {risk_l}. "
                   "'4주 90% 범위' = 현재 변동성 × 과거 '표준화 4주 수익률'의 경험적 5%·95% 분위 → 하단 근처 지정가 주문 참고. 적중률 검증: ◆ 리스크 탭.")

    fund_box = st.expander("펀더멘털 맥락 — 곡물 수급·펀드 포지션 / 거시·금리·달러·해운 (참고 — 점수에는 미반영)", expanded=False)
    fund_box.markdown("**곡물 수급 · 펀드 포지션** (USDA WASDE · CFTC COT)")
    FD = PR.fundamentals([i for i in ["yf:ZC=F", "yf:ZW=F", "yf:KE=F", "yf:ZS=F", "yf:ZM=F", "yf:ZL=F"] if i in NAME]).dropna(how="all")
    fund_box.dataframe(FD.style.format({c: "{:+.2f}" for c in ["전월 대비 (%p)", "USDA 가격 전망 수정 (%)", "펀드 포지션 z (3년)"]}
                                 | {"재고/사용비율 (%)": "{:.1f}", "같은 달 과거 분위": "{:.0%}", "펀드 순포지션 (% OI)": "{:+.1f}"}, na_rep="–"))
    fund_box.caption("WASDE 재고/사용비율이 낮을수록(분위↓) 수급이 빡빡 → 가격 상방 위험. 펀드 포지션 z가 크면 투기적 매수가 몰린 상태 → 되돌림 위험. "
               "검증 결과 이 지표들의 '발표 이후' 가격 예측력은 과거 20년에서 재현되지 않아 점수에 넣지 않고, 위험 맥락으로만 표시 (대두 계열은 미국 수급표 기준).")
    fund_box.markdown("**거시 · 금리 · 달러 · 해운** (FRED · Yahoo · akshare)")
    MT = _macro_table()
    if len(MT):
        fund_box.dataframe(MT.style.format({"최근값": "{:,.2f}", "1주 변화": "{:+.2f}", "4주 변화": "{:+.2f}", "1년 변화": "{:+.2f}", "5년 분위": "{:.0%}"}, na_rep="–"),
                     height=35 * (len(MT) + 1) + 3)
        fund_box.caption("변화 = 금리·스프레드는 %p 차이, 나머지는 % 변화 · 5년 분위 = 최근 5년 중 지금보다 낮았던 날의 비율 (높을수록 5년 고점권) · 점수에는 미반영, 맥락 참고용")

    st.markdown("##### 신호 상세")
    di = st.selectbox("품목 선택", list(T["id"]), format_func=lambda i: PR.BUY_IDS[i], key="pc_detail")
    sig, inp = sig_of(di); sc = PR.score(sig, w) * 100
    pa = PR.usd_t(Lr_, di)
    lo = pa.index.max() - pd.DateOffset(years=3)
    pa3, sc3 = pa[pa.index >= lo], sc[sc.index >= lo]
    f = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.45, 0.3, 0.25], vertical_spacing=0.05,
                      subplot_titles=(f"{PR.BUY_IDS[di]} 가격 ($/t, 근월물) — 🟢 점수 ≥ {tl:.0f} · 🟠 {yb:.0f}–{tl - 1:.0f} · 🟡 {yb:.0f} 미만 (판단 카드와 같은 문턱)",
                                      "매수 점수와 구성 요소", f"상대가치 관계: {PR.REL_DEF.get(di, ('–',))[0]}"))
    f.add_trace(go.Scatter(x=pa3.index, y=pa3.values, line=dict(color=PAL[0], width=1.6), name="가격"), 1, 1)
    # 점은 trace 하나 (색만 점마다 다르게) — 색별 trace 3개로 그리면 마우스오버(x unified)가 다른 색 trace의 '가장 가까운 다른 날' 점까지 함께 보여 줌
    cls_ = [(f"🟢 적극 매수 (점수 ≥ {tl:.0f})", "#2ea043", 5, 1.0), (f"🟠 판단 유보 · 매수 쪽 (점수 {yb:.0f}–{tl - 1:.0f})", "#f0883e", 4, 0.55),
            (f"🟡 판단 유보 · 대기 쪽 (점수 < {yb:.0f})", "#d29922", 3.5, 0.35)]
    ix_ = sc3.dropna().index.intersection(pa3.index)
    s_ix = sc3.reindex(ix_)
    k_ = np.where(s_ix >= tl, 0, np.where(s_ix >= yb, 1, 2))
    f.add_trace(go.Scatter(x=ix_, y=pa3.reindex(ix_), mode="markers", showlegend=False, name="판단",
                           marker=dict(color=[cls_[k][1] for k in k_], size=[cls_[k][2] for k in k_], opacity=[cls_[k][3] for k in k_]),
                           customdata=np.c_[[cls_[k][0] for k in k_], s_ix.values],
                           hovertemplate="%{customdata[0]} · 이번 주 점수 %{customdata[1]:.0f}<extra></extra>"), 1, 1)
    for nm_, col_, sz_, op_ in cls_:                                       # 범례 전용 (데이터 없음)
        f.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color=col_, size=8, opacity=max(op_, 0.6)), name=nm_, hoverinfo="skip"), 1, 1)
    f.add_trace(go.Scatter(x=sc3.index, y=sc3.values, line=dict(color=PAL[0], width=2.2), name="매수 점수"), 2, 1)
    show_comp = st.checkbox("점수 구성 요소도 함께 그리기 (선이 많아짐)", False, key="pc_comp")
    for j, c in enumerate([c for c in PR.COMP_LBL if w.get(c, 0) > 0 and show_comp]):
        s_ = sig[c][sig.index >= lo] * 100
        f.add_trace(go.Scatter(x=s_.index, y=s_.values, line=dict(color=PAL[(j + 3) % 8], width=1, dash="dot"), name=PR.COMP_LBL[c]), 2, 1)
    f.add_hline(y=tl, line_dash="dash", line_color="#2ea043", row=2, col=1); f.add_hline(y=yb, line_dash="dot", line_color="#f0883e", row=2, col=1)
    rel = inp.get("rel")
    if rel is not None:
        r3 = rel[rel.index >= lo]
        f.add_trace(go.Scatter(x=r3.index, y=r3.values, line=dict(color=PAL[6], width=1.4), name="상대가치 관계"), 3, 1)
        q_ = rel.rolling(252, min_periods=126)
        for qq, nm_ in [(0.2, "1년 하위 20%"), (0.8, "1년 상위 20%")]:
            b_ = q_.quantile(qq)[rel.index >= lo]
            f.add_trace(go.Scatter(x=b_.index, y=b_.values, line=dict(color="#898781", width=1, dash="dot"), name=nm_, showlegend=False), 3, 1)
    st.plotly_chart(fig_layout(f, 680, hoverlabel=dict(namelength=-1)), key="pc_detail_fig")
    st.caption(f"점 색 = 그 주 가격 매력 점수 기준 판단 (🟢 ≥ τ {tl:.0f} · 🟠 τ−10 ~ τ · 🟡 그 아래, 진할수록 매수 쪽). "
               "🔴 매수 보류는 점수뿐 아니라 '미루는 기간 안에 더 쌀 확률 ≥ p*'(AUC 기준 충족 품목)까지 봐야 해서 이 과거 차트에는 표시하지 않음 — 이번 주 🔴 여부는 위 판단 카드·지도에서.")
    if risk != "ewma":
        rs_ = sig["risk"][sig.index >= lo] * 100
        f = go.Figure(go.Scatter(x=rs_.index, y=rs_.values, fill="tozeroy", line=dict(color=PAL[1], width=1)))
        st.plotly_chart(fig_layout(f, 200, title=f"위험 지표 ({risk_l}, %) — 높을수록 변동성 큼 (4주 가격 범위 넓음)", yaxis_range=[0, 100]), key="pc_risk_fig")
    m = np.log(La_[di].dropna()).resample("ME").mean()
    rel_m = (m - m.groupby(m.index.year).transform("mean")) * 100
    yy = rel_m.index.year.max()
    f = go.Figure()
    for j, (nm_, sub) in enumerate([("전체 연도", rel_m), ("최근 10년", rel_m[rel_m.index.year >= yy - 10])]):
        prof = sub.groupby(sub.index.month).mean()
        f.add_trace(go.Bar(x=[f"{mm}월" for mm in prof.index], y=prof.values, name=nm_, marker_color=[PAL[0], PAL[2]][j]))
    st.plotly_chart(fig_layout(f, 260, barmode="group", title="계절 패턴: 월평균 가격의 연평균 대비 차이 (%, 롤 보정) — 음수 = 평균적으로 싼 달", hovermode="x"),
                    key="pc_season")

    st.markdown("##### 과거 성과 검증 — '지금 살까, 미룰까' 규칙 (학습 2008–2016 / 시험 2017–)")
    st.caption(f"매주 한 번 구매 판단: 점수 ≥ {tau:.2f}이면 즉시, 아니면 (근거 있는 품목에서) 하락 확률 ≥ {p_star:.0%}일 때 미루고 매주 재평가(최대 품목별 시계). "
               "기준선 = 바로 사기. 위약 = 같은 비율의 주를 무작위로 골라 같은 기간 미룸. 시험 기간 결과가 실제 성과에 가까움 (시계·기준은 학습 기간으로 정함).")
    rows, yrs = [], {}
    for i in ids:
        h, seaf, active = cfg[i]
        if not active:
            continue
        W_, F = _proc_defer(i, tuple(sorted(w.items())), risk, lb_of(i), season, h, seaf, ROLL)
        for per, (a_, b_) in [("학습 2008–16", ("2008-01-01", "2016-12-31")), ("시험 2017~", ("2017-01-01", None))]:
            D = PR.backtest_defer(W_, F, float(tau), float(p_star), h, "adaptive", a_, b_)
            S = PR.defer_summary(D)
            if not S:
                continue
            rows.append({"품목": PR.BUY_IDS[i], "기간": per, "미루는 기간": f"{h}주" + (" · 계절" if seaf else ""), "판단 수": S["판단 수"], "미룸 비율": S["미룸 비율"],
                         "판단당 절감 (bp)": S["판단당 평균 절감 (bp)"], "미룬 경우 평균 (bp)": S["미룬 경우 평균 (bp)"], "미룬 경우 적중률": S["미룬 경우 적중률"],
                         "위약 평균 (bp)": S["위약 평균 (bp)"], "위약 90% (bp)": S["위약 90% (bp)"], "연도 t": S["연도 t"]})
            if per.startswith("시험"):
                yrs[PR.BUY_IDS[i]] = D.groupby(D.index.year)["절감bp"].mean()
    if not rows:
        st.info("선택한 품목 중 미루기 권고가 활성화된 품목이 없습니다 — 모두 '가격 매력도만으로 판단'. (활성화 기준을 낮추면 근거가 약한 품목도 포함)")
    else:
        R_ = pd.DataFrame(rows).set_index(["품목", "기간"])
        st.dataframe(R_.style.format({"판단 수": "{:.0f}", "미룸 비율": "{:.0%}", "판단당 절감 (bp)": "{:+.1f}", "미룬 경우 평균 (bp)": "{:+.1f}",
                                      "미룬 경우 적중률": "{:.0%}", "위약 평균 (bp)": "{:+.1f}", "위약 90% (bp)": "{:+.1f}", "연도 t": "{:+.2f}"}, na_rep="–"))
        Y = pd.DataFrame(yrs)
        if len(Y):
            f = go.Figure([go.Bar(x=Y.index, y=Y[c], name=c) for c in Y.columns])
            f.add_hline(y=0, line_color="#898781")
            st.plotly_chart(fig_layout(f, 300, barmode="group", title="시험 기간 연도별: 바로 사기 대비 판단당 절감 (bp) — 0 위 = 그해 미루기 규칙이 쌌음", hovermode="x"),
                            key="pc_yr")
        te = R_.xs("시험 2017~", level="기간")
        avg = te["판단당 절감 (bp)"].mean(); pl = te["위약 평균 (bp)"].mean()
        st.info(f"해석: 시험 기간 활성 품목 평균 판단당 {avg:+.1f}bp (무작위로 같은 비율 미룸 {pl:+.1f}bp) — 카고 1척 750만 달러(3만 톤 × 250달러)라면 판단 1회당 약 "
                f"{avg / 1e4 * 7.5e6:,.0f}달러. 미루기 자체는 가격이 오르는 국면에서 비용(위약이 음수)이므로, 위약보다 확실히 높아야 예측이 정보를 담고 있다고 볼 수 있음.")
    dfp, _m = EX.load("procure_defer")
    if dfp is not None and "미룸주" in dfp:
        di2 = st.selectbox("미루는 기간별 예측력 보기", list(T["id"]), format_func=lambda i: PR.BUY_IDS[i], key="pc_hz_item")
        q = dfp[(dfp.구분 == "예측") & (dfp.id == di2)]
        if len(q):
            f = go.Figure()
            for j, (per, sea) in enumerate([("학습", False), ("학습", True), ("시험", False), ("시험", True)]):
                v = q[(q.기간 == per) & (q.계절특성 == sea)].sort_values("미룸주")
                f.add_trace(go.Bar(x=[f"{m}주" for m in v.미룸주], y=v.하락AUC, name=f"{per} · {'계절 특성' if sea else '기본'}",
                                   marker_color=PAL[j], opacity=0.55 if per == "학습" else 1.0))
            f.add_hline(y=0.5, line_color="#898781", annotation_text="0.5 = 동전 던지기")
            f.add_hline(y=float(auc_min), line_dash="dot", line_color=PAL[1], annotation_text=f"활성화 기준 {auc_min:.2f}")
            st.plotly_chart(fig_layout(f, 300, barmode="group", title=f"{PR.BUY_IDS[di2]} — 미루는 기간 × 예측 특성별 '더 쌀 확률' AUC (학습 2008–16 · 시험 2017–)",
                                       yaxis_range=[0.35, 0.75], hovermode="x"), key="pc_hz_fig")
            st.caption("품목별 시계는 학습 막대 중 가장 높은 것으로 고름 (시험 막대는 고를 때 보지 않음). 계절 특성 = 연중 주차의 사인·코사인 — 휘발유·생우처럼 연중 패턴이 강한 품목에서 "
                       "긴 시계(3–4주)일수록 AUC를 올림.")


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


@st.cache_data(show_spinner=False)
def _ar_now(roll: bool):
    import ls_strategies as LS
    P = LS.weekly(_levels(None, roll), LS.BASKET, "2001-01-01")
    ar = LS.absorption(LS._ret(P), 52)
    d = ar - ar.shift(4)
    return ar, d / d.rolling(156, min_periods=52).std()


def _ar_alert():
    """흡수비율(원자재 19개, 52주 상관의 1주성분 비중) 급등 경고 — 여러 품목을 함께 사도 분산 효과가 줄어드는 시기."""
    ar, z = _ar_now(ROLL)
    a, zz = ar.dropna(), z.dropna()
    if not len(a) or not len(zz):
        return
    pct = float((a < a.iloc[-1]).mean())
    last_d = min(a.index[-1], _levels(None, ROLL).index.max())
    msg = (f"흡수비율 {a.iloc[-1]:.2f} (과거 {pct:.0%} 분위) · 4주 변화 z {zz.iloc[-1]:+.1f} ({last_d:%Y-%m-%d} 기준) — "
           "원자재 19개가 하나의 요인에 얼마나 묶여 움직이는지 (높을수록 여러 품목을 함께 사도 분산 효과가 작음)")
    if zz.iloc[-1] > 1:
        st.warning("⚠️ **상관 급등 경고** — " + msg + ". 과거 급등 시작 후 13주 바스켓 평균 −1.1% (평소 +1.5%, t −1.5): 동반 하락 위험↑, 품목 분산에 기대지 말 것")
    elif pct > 0.8:
        st.info("ℹ️ 상관 높은 상태 — " + msg)
    else:
        st.caption("✅ 상관 급등 신호 없음 — " + msg)


def _alpha_alert():
    """롱숏 전략 우위 경고: 가치 혼합·기존 70/30의 2년 롤링 알파(바스켓 대비)가 0 아래면 경고, 그 전 4년 평균의 절반 아래면 주의."""
    import ls_extra as LX
    try:
        o = _lx_run(5.0, ROLL)
    except Exception:
        return
    msgs, worst = [], 2
    for nm, k in [("가치 혼합", "혼합 · MinVar 50 + 모멘텀 25 + V1 가치 25"), ("기존 70/30", "혼합 · MinVar 70 + 모멘텀 30 (기존)")]:
        RA = LX.rolling_alpha(o["cands"][k], o["bh"], 104).dropna()
        if not len(RA):
            continue
        a_ = float(RA["알파 (연 %)"].iloc[-1])
        prev = float(RA["알파 (연 %)"].loc[(RA.index < RA.index[-1] - pd.DateOffset(years=2)) & (RA.index >= RA.index[-1] - pd.DateOffset(years=6))].mean())
        lvl = 0 if a_ < 0 else (1 if np.isfinite(prev) and prev > 0 and a_ < prev / 2 else 2)
        worst = min(worst, lvl)
        msgs.append(f"{nm} {a_:+.1f}% (그 전 4년 평균 {prev:+.1f}%)")
    if not msgs:
        return
    txt = "롱숏 전략 2년 롤링 알파(바스켓 대비, 연): " + " · ".join(msgs)
    if worst == 0:
        st.warning("⚠️ **전략 우위 약화** — " + txt + ". 알파가 0 아래 = 최근 2년은 원자재 시장을 탄 것 이상의 성과가 없음 → 비중 축소·원인 점검 (⑤ 롱숏 → 품목 기여도)")
    elif worst == 1:
        st.info("ℹ️ 전략 우위 둔화 — " + txt + ". 그 전 4년 평균의 절반 아래로 내려옴")
    else:
        st.caption("✅ 전략 우위 유지 — " + txt)


def tab_risk():
    import risk as RK
    st.subheader("리스크 — VaR/ES · 꼬리 위험 · 구매·보유 물량 위험")
    _ar_alert()
    _alpha_alert()
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
    c, d, e = subtabs(["VaR · ES 백테스트", "꼬리 위험 (EVT · 스트레스)", "구매·보유 물량 위험"])
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
            from scipy import stats as _ss
            hi = st.selectbox("수익률 분포를 볼 품목", tids, 0, format_func=short, key="rk_hist")
            x_ = Rt[hi].dropna()
            mu_, sd_ = float(x_.mean()), float(x_.std())
            sk_, ku_ = float(_ss.skew(x_)), float(_ss.kurtosis(x_))
            f = go.Figure(go.Histogram(x=x_, nbinsx=100, histnorm="probability density", marker_color=PAL[0], opacity=0.7, name="실제 일간 수익률"))
            gx = np.linspace(x_.quantile(0.0005), x_.quantile(0.9995), 400)
            f.add_trace(go.Scatter(x=gx, y=_ss.norm.pdf(gx, mu_, sd_), line=dict(color=PAL[1], width=2.5), name="같은 평균·표준편차의 정규분포"))
            f.update_xaxes(range=[float(x_.quantile(0.0005)), float(x_.quantile(0.9995))], title="일간 수익률 (%)")
            f.add_annotation(xref="paper", yref="paper", x=0.99, y=0.97, xanchor="right", showarrow=False, align="right", font=dict(size=13),
                             text=f"<b>왜도 {sk_:+.2f}</b> ({'급락 꼬리가 김' if sk_ < -0.1 else '급등 꼬리가 김' if sk_ > 0.1 else '좌우 대칭에 가까움'})<br>"
                                  f"<b>초과첨도 {ku_:.1f}</b> (0 = 정규, 클수록 극단값 잦음)<br>3σ 넘는 날 {(np.abs(x_ - mu_) > 3 * sd_).mean():.2%} (정규 0.27%)")
            st.plotly_chart(fig_layout(f, 380, title=f"{short(hi)} 일간 수익률 분포 vs 정규분포", hovermode="x", bargap=0), key="rk_hist_fig")
            st.caption("막대(실제)가 빨간 곡선(정규)보다 가운데가 뾰족하고 양 끝이 두꺼우면 '팻테일' → 정규분포 가정 VaR가 큰 손실을 과소평가. "
                       "품목별 왜도·첨도 수치는 위 꼬리 통계 표에 있음.")
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
        u1_, u2_, u3_ = st.columns(3)
        alpha_p = u1_.selectbox("신뢰수준", [0.95, 0.99], 1, format_func=lambda x: f"{x:.0%}", key="rk_palpha")
        hp = u2_.selectbox("기간 (영업일)", [5, 10, 20, 60], 2, key="rk_ph")
        vm_p = u3_.selectbox("σ 모형", ["GARCH(1,1)", "EWMA 품목별 λ (확장)"], 0, key="rk_pvm",
                             help="기본 GARCH(1,1) = ◆ 리스크 단일 품목 VaR와 같은 σ (변동성 모형 경주에서 1일 분산 예측 최상). "
                                  "EWMA = 이전 버전(v1.37 이전) 기준 — 잔잔한 시기엔 GARCH보다 σ가 낮게 나옴(장기 평균으로 돌아가는 성질이 없음)")
        Lr_ = _levels(extra_bytes, False)
        qc = st.columns(max(len(pids), 1))
        qty = {i: qc[j].number_input(f"{PR.BUY_IDS[i]} ({'온스' if PR.unit_of(i) == '$/oz' else '톤'})", 0.0, 1e8, 30000.0, 1000.0, key=f"rk_qty_{i}") for j, i in enumerate(pids)}
        if len(pids) >= 1 and sum(qty.values()) > 0:
            px = {i: float(PR.usd_t(Lr_, i).dropna().iloc[-1]) for i in pids}
            expo = pd.Series({i: sg * qty[i] * px[i] for i in pids})
            OO = {i: _risk_vm(i, (vm_p,), ROLL, extra_bytes) for i in pids}
            Rp = pd.DataFrame({i: OO[i]["r"] for i in pids}).dropna()
            Sp = pd.DataFrame({i: OO[i]["S"][vm_p] for i in pids}).reindex(Rp.index).dropna()
            PRk = RK.position_risk(Rp.loc[Sp.index], Sp, expo, alpha_p, int(hp))
            w1_, w2_, w3_, w4_ = st.columns(4)
            w1_.metric("총 노출", f"${expo.abs().sum():,.0f}")
            w2_.metric(f"{hp}일 {alpha_p:.0%} VaR (정규)", f"${PRk['var_normal']:,.0f}", f"ES ${PRk['es_normal']:,.0f}", delta_color="off")
            w3_.metric(f"{hp}일 {alpha_p:.0%} VaR (다변량 FHS)", f"${PRk['var_fhs']:,.0f}", f"ES ${PRk['es_fhs']:,.0f}", delta_color="off")
            w4_.metric("분산 효과", f"${PRk['undiversified'] - PRk['var_normal']:,.0f}", f"단순 합 ${PRk['undiversified']:,.0f}", delta_color="off")
            st.dataframe(PRk["table"].rename(index=lambda i: PR.BUY_IDS[i]).style.format(
                {"노출 ($)": "{:,.0f}", "일 σ (%)": "{:.2f}", f"{hp}일 σ ($)": "{:,.0f}", "VaR 기여 ($, 정규)": "{:,.0f}", "기여 비중": "{:.0%}"}))
            st.caption(f"σ = {vm_p} (walk-forward, 매년 1월 재추정), 상관 = 직전 500일 표준화 잔차. 다변량 FHS = 과거 표준화 잔차 '벡터'(같은 날 동시 충격)를 현재 σ로 "
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
    ("구매", "미루기 권고 (재고 무관 · 1~4주)", "재고를 빼고 '지금 가격이 매력적인가 + 미루는 기간(1~4주) 안에 지금보다 쌀 확률'로 구매를 미루면 바로 사는 것보다 싼가? 계절 특성은 도움이 되나?",
     "곡물 하락 예측 AUC ≈ 0.48~0.51 (예측력 없음). 품목마다 학습 기간으로 시계·특성을 고르고 학습 AUC > 0.55인 9품목(휘발유·생우·돈육·오렌지주스·면화 등)만 쓰면 "
     "시험 2017~ 판단당 +23bp vs 무작위 미룸 −14bp. 계절 특성은 3~4주 시계에서 AUC +0.02. 전 품목 일괄 적용은 학습 +6bp → 시험 −1.5bp (과적합)",
     "★ 구매 타이밍 '지금 살까, 미룰까' (τ 0.60, p* 0.55, 품목별 시계, AUC 0.55 이하는 '설정 AUC 기준치 미달' → 🟢/🟡만)", "⚠️ 일부만 (근거 있는 품목만)"),
    ("구매", "구매 타이밍 (프리셋 · 단계별)", "기준 프리셋에서 신호를 하나씩 바꾸면 매입가가 얼마나 싸지나? 어떤 프리셋이 가장 좋나?",
     "상대가치(크러시 마진·옥수수/밀 비율)만 전·후반 모두 뚜렷. '추세만'은 손해, 공격적 프리셋은 꼬리 손실 큼",
     "★ 구매 타이밍 '추천' 프리셋의 근거", "✅ 채택"),
    ("구매", "철강 원료·구리 상대가치", "곡물에서 통한 상대가치를 철광석·점결탄·구리에도 쓸 수 있나? 어떤 관계로?",
     "철광석 = SGX/DCE 차익, 점결탄 = 고로 마진이 두 기간 모두 +. 구리/알루미늄은 후반에 무너짐, 구리/금은 작음",
     "★ 금속 3품목의 상대가치 정의", "✅ 채택 (구리는 참고)"),
    ("구매", "재고 기반 반복 구매 (학습/시험 · 예측)", "평균 4주마다 1카고·저장 상한 2척일 때 '점수 ≥ τ면 이번 주 1척'이 정기·매주 균등 구매보다 싼가? (학습 2008~16 → 시험 2017~) 1~2주 예측은 도움이 되나?",
     "학습으로 고른 τ=0.45가 시험 기간 25품목 평균 정기 대비 +10bp·균등 대비 +13bp (학습 +37bp에서 크게 줄어듦). 곡물만은 시험 기간 효과 소멸(−10~+2bp), "
     "확장 품목 +13~+21bp. 1~2주 가격 방향 예측 AUC 0.51, '하락 확률 > 0.6이면 1주 미루기'는 시험 +5bp 소폭(신뢰 낮음)",
     "실험 기록 (v1.9.0부터 ★는 재고 무관 '미루기 권고'로 전환)", "⚠️ 일부만"),
    ("구매", "벌크(일괄) 구매 — 문턱 · 가격 위치 창", "카고 전량을 한 번에 살 때, '점수가 문턱 τ 이상인 첫 주에 매수'가 오늘 바로 사기보다 나은가? τ와 가격 위치 창(6개월·1년)은?",
     "곡물은 τ 0.55~0.65에서 바로 사기 대비 +20~42bp(두 하위기간 모두 +). 확장 품목 다수는 상승 추세 탓에 기다릴수록 손해(−30bp). "
     "1년 창은 대두박·밀·면화 등 9품목에서만 우세, 돈육·귀리는 6개월이 훨씬 나음",
     "품목별 가격 위치 창 (★ 점수에 사용) · 문턱 0.60 (★ 가격 매력 문턱 기본값)", "✅ 채택 (곡물)"),
    ("구매", "전 품목 구매 타이밍 (단일 vs 다변량)", "곡물 외 에너지·귀금속·소프트·축산·DCE 품목에도 상대가치가 통하나? 대체재 하나 대신 여러 관계를 묶으면(다변량) 나아지나?",
     "무관한 가격비(위약)도 31%가 '두 기간 개선'을 통과 → 위약·자기 가격 대조군을 넘은 관계만 인정하면 확장 28품목 중 10개(가스/원유·생우/송아지·"
     "알루미늄/구리·브렌트 경유 크랙 등). 다변량은 단일 관계 평균 수준. 기존 곡물 정의도 엄격 기준으론 대두 크러시만 통과 → 재검증 필요",
     "★ 구매 타이밍 전 품목 선택 + 품목별 채택 관계", "⚠️ 일부만"),
    ("구매", "곡물 상대가치 재검증 (2008~)", "'추천' 프리셋의 근거였던 곡물 6품목 상대가치가 원래 기간(2008~, 경계 2017)에서도 위약·자기 가격 대조군을 넘나?",
     "현재 정의는 6품목 모두 엄격 기준 미통과 (크러시 마진도 26주 전반 −1bp). 6품목 평균 개선 +3.9bp ≈ 위약 평균 +2bp·대조군 +5bp → "
     "곡물 상대가치의 효과는 대부분 자기 가격 위치 효과. 통과는 대두 CBOT/DCE(+12bp)뿐 (5개 중 사후 선택·t≈1.8)",
     "프리셋 가중치는 유지(변경 권고만), 상대가치의 근거를 '약함'으로 정정", "⚠️ 재검증: 약함"),
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
    files = {"미루기 권고 (재고 무관 · 1~4주)": "procure_defer", "구매 타이밍 (프리셋 · 단계별)": "procure_steps", "철강 원료·구리 상대가치": "procure_metals", "전 품목 구매 타이밍 (단일 vs 다변량)": "procure_all", "곡물 상대가치 재검증 (2008~)": "procure_grains", "벌크(일괄) 구매 — 문턱 · 가격 위치 창": "procure_lump", "재고 기반 반복 구매 (학습/시험 · 예측)": "procure_inventory", "리스크 모형": "risk", "모델 선택 메타 전략": "metasel",
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


SIG_SEC = "🧪 3축 신호 · 위험 한도 실험 (2026-11)"
SIG_DIR = A.DATA.parent / "docs" / "실험요약"


def ex_signal_tests():
    """3축(MinVar · 모멘텀 · 가치) 비중을 HMM · 칼만 · DCC · 뉴스 · 상관 신호로 바꾸는 실험 묶음 요약 — docs/실험요약 워드 문서와 같은 내용 · 그림."""
    st.markdown("#### 3축 혼합 전략 · 위험 신호 백테스트 요약 (2026-11-04, 자료 기준일 2026-10-26)")
    st.success("**결론** — ① 신호로 비중을 바꾸는 방법(갈아타기 · 메타 예측 · 적응형 비중 · 현금 피신 · 목표 변동성 조절 · 상관↑ 추세 집중 · 짧은 추세)은 모두 시험 기간(2017–26) Sharpe가 "
               "고정 50/25/25(1.07)를 유의하게 넘지 못해 **미채택** ② 쓸 만한 것은 위험 관리: **곡물 합계 비중 상한 25%**(수익 그대로 · VaR 2.87 → 2.25%, ⑤ Portfolio Customization에 반영)와 "
               "**손실 한도 경보**(`risk_alert.py`, 비중은 그대로 두고 담당자에게 알림)")
    doc = next(SIG_DIR.glob("*.docx"), None)
    if doc:
        st.download_button("📄 요약 문서 내려받기 (Word, 5쪽)", doc.read_bytes(), file_name=doc.name, key="ex_sig_doc",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    fig = lambda n, cap: st.image(str(SIG_DIR / "figs" / n), caption=cap, use_container_width=True) if (SIG_DIR / "figs" / n).exists() else None   # noqa: E731
    fig("f1_flow.png", "실험 구조: 3축 재료 × 신호 7종 × 쓰는 방식 A–H → 같은 검증 틀(학습 2008–16 선택 · 시험 2017–26 · 비용 · 위약 · 대조군)")
    st.markdown("""
| 실험 | 방법 | 결과 (시험 Sharpe · 기준 1.07) | 판정 |
|---|---|---|---|
| A 갈아타기 | 매년 직전 3 · 5년 최적 비중 · 성과/위기/투자선 이탈 트리거 | 0.58–0.89 · 직전 5년 최적점은 다음 해 15번 중 3번만 이김 | 미채택 |
| B 메타 레이어 | 6개 신호로 다음 13주 1등 축 예측 → 기울이기 | 1.05 · 1등 축 적중 29%(무작위 33%) | 미채택 |
| C 적응형 비중 | 칼만 · HMM · 뉴스 · DCC 관점으로 ±25%p | 칼만 0.79(학습만 우세) · HMM 1.08 · 결합 0.85 | 미채택 |
| D 현금 피신 | 신호 9종 켜지면 50 · 100% 현금 | 0.63–1.12 · 낙폭 감소는 '덜 투자한 몫' | 미채택 |
| E 목표 변동성 동적 | 신호 개수로 12 · 10 · 7 · 5% + 국채 이자 | 0.96–1.02 | 미채택 |
| F 상관↑ + 추세 집중 | 상관 급등(+ 같이 오름) 때 강한 추세 상위 2–3개 롱 | 0.91–1.07 · 위약 p 0.48 | 미채택 |
| J · K 짧은 추세 · 모멘텀 기간 | 4 · 8 · 13주 추세 · 모멘텀 축 기간 4–39주 | 짧을수록 나쁨(4주 혼합 0.98 · 13주 0.77, 39주 1.09 ≈ 52주) | 미채택 |
| G · I 품목군 상한 | 곡물 · 에너지 · 축산 합계 비중 상한 | 곡물 25%: 1.09 · VaR 2.87 → 2.25% | **채택(⑤)** |
| H 손실 한도 경보 | VaR · 역스트레스 · 상관 · 낙폭 등 10항목 | 경보 뒤 큰 손실 비율 0–14% → 예측용 아님 | 운영 도구 |
""")
    fig("f2_results.png", "실험별 시험 Sharpe(각 실험의 학습 최고 설정) — 점선 = 기준 1.07")
    st.markdown("##### 특수 케이스 — '상관이 높으면서 같이 오르는' 구간 (2022 곡물 · 원유, 2025–26 금 · 은)")
    fig("f3_f_logic.png", "판단 흐름: 매주 ① 상관 급등 ② 같이 오르는 중(직전 13주 수익 > 0)이면 ③ 강한 추세 품목에 집중, 아니면 기본 50/25/25")
    fig("f4_cases.png", "위: 가격(시작 = 100) · 아래: 곡물 5개 26주 상관과 신호 문턱(과거 상위 10%)")
    st.markdown("""
- **2022년은 '상관 급등'이 아니었음** — 곡물 상관은 전쟁 직전(2021년 말 약 0.7)에 높았고 전쟁 중엔 0.4대(연평균 0.45, 2011–12년 0.7–0.8보다 낮음) → 상관 신호가 켜지지 않음.
- **급등이 너무 짧음** — 밀은 2월 24일–3월 8일 약 2주에 끝나고 2분기 −24%, 원유 3분기 −30% → 26 · 52주는 물론 4주 추세도 늦게 타고 늦게 내림(회전율만 연 20배+).
- **금 · 은은 잡았지만 규칙으로는 못 이김** — 상승 추세 집중이 2025–26년 금 · 은으로 +44%(52주 모멘텀 +16%), 같은 규칙이 2008–16년엔 손실(학습 Sharpe −0.01).
- **기준 혼합도 이미 그 구간을 탔음** — 50/25/25가 2022년 1분기 +20%, 2022–23년 +29%(집중판 +22%).
""")
    fig("f5_graincap.png", "곡물 합계 비중 상한 — 수익은 그대로, VaR 중 곡물 몫 62% → 34%")
    st.caption("재현: strategy_switch · strategy_meta · strategy_adaptive · strategy_cash · strategy_voltgt_dyn · strategy_crisis_mom · strategy_short_trend · strategy_mom_lookback · "
               "strategy_graincap · strategy_groupcap · stress_portfolio · risk_alert (*.py). 모든 실험 기록: docs/백테스트_로그.md")


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
    KIND = {"bekk_search": "BEKK 조합 찾기", "payoff": "구매 진입 손익비", "ml": "④ 단일 품목 모델", "panel": "④ 여러 품목", "midas": "④ MIDAS"}
    kinds = sorted(R["kind"].dropna().unique()) if "kind" in R else []
    f0, f1, f2, f3 = st.columns([1.4, 1.2, 1.0, 1.8])
    ksel = f0.multiselect("종류", kinds, kinds, format_func=lambda k: KIND.get(k, k), key="rn_kind",
                          help="BEKK 조합 찾기 = ③ 변동성·공분산 › 다변량 GARCH의 자동 찾기 결과(기간·조합 크기는 제목, 근사 순위·BEKK 확인 표는 아래 실행 상세)")
    vsel = f1.multiselect("판정", list(RL.VERDICTS), list(RL.VERDICTS), format_func=RL.VERDICTS.get, key="rn_v")
    wfo = f2.checkbox("walk-forward 실행만", False, key="rn_wf")
    q = f3.text_input("제목 검색 (품목·모델 등)", "", key="rn_q")
    V = R[R.verdict.isin(vsel)]
    if kinds:
        V = V[V["kind"].isin(ksel)]
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
    if M.get("kind") == "ml" and M["settings"].get("walk_forward") and M["settings"].get("목적") == "predict":
        with st.container(border=True):
            st.markdown("**🔁 운인지 실력인지 — 같은 설정을 다른 품목·기간에 다시 돌려 보기**")
            wf_tab = M["tables"].get("walk_forward")
            mods = [m_ for m_ in (wf_tab.index if wf_tab is not None else []) if not str(m_).startswith("위약")] or M["settings"]["모델"]
            best_m = (M.get("summary") or {}).get("WF 최고 모델") or R.set_index("run_id").loc[rid].get("WF 최고 모델")
            r1, r2, r3 = st.columns([1.3, 3, 0.9])
            mdl = r1.selectbox("모델", list(mods), list(mods).index(best_m) if best_m in mods else 0, key=f"rv_m_{rid}")
            t0_ = M["settings"]["대상"]
            pool = [i for i in COMMOD_IDS if GROUP.get(i) in ("Grains", "Energy", "Metals", "Softs", "Livestock", "Iron Ore / Coal")]
            dflt = list(dict.fromkeys([t0_] + [i for i in ["yf:ZC=F", "yf:ZS=F", "yf:ZW=F", "yf:ZM=F", "yf:KE=F", "yf:ZL=F", "yf:CL=F", "yf:HG=F"] if i in pool]))[:8]
            tg = r2.multiselect("품목 (원래 품목 + 비교 품목)", pool, dflt, format_func=lbl, key=f"rv_t_{rid}")
            if r3.button("재검증 실행", key=f"rv_run_{rid}", type="primary") and tg:
                import runlog as RL
                pg = st.progress(0.0, "재검증 중")
                T_ = _revalidate_ml(M["settings"], mdl, tg, progress=lambda x, m_: pg.progress(min(1.0, x), m_))
                pg.empty()
                ok_ = T_.get("운 이상", pd.Series(dtype=bool)).fillna(False)
                summ = {"재검증 모델": mdl, "품목 수": len(T_), "운 이상 품목": int(ok_.sum()), "두 기간 + 품목": int(T_.get("두 기간 +", pd.Series(dtype=bool)).fillna(False).sum())}
                rid2 = RL.record("revalidate", f"재검증: {M['title']} · {mdl}", dict(원실행=rid, 모델=mdl, 품목=tg, 원설정=M["settings"]), summ,
                                 {"revalidate": T_.drop(columns=["id"], errors="ignore")},
                                 [f"{int(ok_.sum())}/{len(T_)} 품목이 위약 최댓값 초과"])
                st.session_state[f"rv_res_{rid}"] = (T_, rid2)
            if f"rv_res_{rid}" in st.session_state:
                T_, rid2 = st.session_state[f"rv_res_{rid}"]
                if "Sharpe (비용 후)" in T_:
                    pc = [c for c in T_.columns if c.startswith("위약 최댓값")][0]
                    f = go.Figure()
                    f.add_trace(go.Bar(x=T_.index, y=T_["Sharpe (비용 후)"], name="Sharpe (비용 후)",
                                       marker_color=[PAL[2] if v else "#898781" for v in T_["운 이상"].fillna(False)]))
                    f.add_trace(go.Scatter(x=T_.index, y=T_[pc], mode="markers", name=pc, marker=dict(symbol="line-ew-open", size=22, color=PAL[7])))
                    f.add_trace(go.Scatter(x=T_.index, y=T_["전반 (비용 후)"], mode="markers", name="전반", marker=dict(color=PAL[0], size=8)))
                    f.add_trace(go.Scatter(x=T_.index, y=T_["후반 (비용 후)"], mode="markers", name="후반", marker=dict(color=PAL[3], size=8, symbol="diamond")))
                    f.add_hline(y=0, line_color="#898781")
                    st.plotly_chart(fig_layout(f, 320, title="품목별 walk-forward Sharpe (비용 후) — 초록 = 위약 최댓값(빨간 선) 초과", hovermode="x"),
                                    key=f"rv_fig_{rid}")
                    n_ok = int(T_["운 이상"].fillna(False).sum()); n2 = int(T_["두 기간 +"].fillna(False).sum())
                    verdict = ("✅ 여러 품목·기간에서 재현 — 실력일 가능성" if n_ok >= max(2, len(T_) // 2) and n2 >= len(T_) // 2 else
                               "⚠️ 일부 품목에서만 — 품목 특성일 수 있음" if n_ok >= 1 else "❌ 재현 안 됨 — 원래 결과는 운(여러 시도 중 최고)일 가능성이 큼")
                    st.markdown(f"**판정: {verdict}** · 위약 최댓값 초과 {n_ok}/{len(T_)} 품목, 두 하위기간 모두 + {n2}/{len(T_)} 품목")
                st.dataframe(T_.drop(columns=["id"], errors="ignore").style.format(
                    {c: "{:+.2f}" for c in T_.columns if "Sharpe" in c or "위약" in c or "반 (" in c or c == "Buy&Hold"} | {"적중률": "{:.1%}"}, na_rep="–"))
                st.caption(f"재검증 결과도 실행 기록에 저장됨 (`{rid2}`). 같은 설명변수·특성·시차·비용·무거래 구간을 품목만 바꿔 적용 — 대상 품목이 설명변수에 있으면 제외. "
                           "위약 최댓값 = 원래 실행에서 비교한 모델 수만큼 무작위 부호를 시도했을 때의 95% 분위 (그 모델이 '여러 개 중 최고'로 골라졌으므로).")
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
    sec = st.radio("실험", ["📋 한눈에 보기", "📂 내 실행 기록", SIG_SEC,"구매 타이밍 (프리셋 · 단계별)", "철강 원료·구리 상대가치", "전 품목 구매 타이밍 (단일 vs 다변량)", "곡물 상대가치 재검증 (2008~)", "벌크(일괄) 구매 — 문턱 · 가격 위치 창", "미루기 권고 (재고 무관 · 1~4주)", "재고 기반 반복 구매 (학습/시험 · 예측)", "리스크 모형", "모델 선택 메타 전략", "최신 시계열 모델 · 예측 목표", "전략 비교 (리스크 버짓)",
                          "모멘텀 기간", "거래 규칙", "변동성 사이징"],
                   horizontal=True, key="ex_sec")
    pct = lambda cols: {c: "{:.2%}" for c in cols}
    pc_view = None
    if sec == "📋 한눈에 보기":
        ex_overview(); return
    if sec == "📂 내 실행 기록":
        ex_runs(); return
    if sec == SIG_SEC:
        ex_signal_tests(); return
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
    elif sec == "미루기 권고 (재고 무관 · 1~4주)":
        import procurement as PR
        import experiments as EX
        st.markdown("##### 미루기 권고 — '지금 매력적이면 사고, 아니면 미루는 기간 안에 더 쌀 확률을 보고 미룬다' (학습 2008–16 / 시험 2017–)")
        df = _exp_header("procure_defer")
        if df is None:
            return
        if "미룸주" not in df:
            st.info("이전 형식(2주 고정) 결과입니다 — 재실행하면 미루는 기간 1–4주 × 계절 특성 비교가 표시됩니다."); return
        P_ = df[df.구분 == "예측"]
        A_ = P_.pivot_table(index=["군", "미룸주"], columns=["기간", "계절특성"], values="하락AUC")
        f = go.Figure()
        for j, (per, sea) in enumerate([("학습", False), ("학습", True), ("시험", False), ("시험", True)]):
            for k_, grp in enumerate(["곡물", "확장"]):
                v = A_.loc[grp][(per, sea)] if (per, sea) in A_.columns else None
                if v is not None:
                    f.add_trace(go.Scatter(x=[f"{m}주" for m in v.index], y=v.values, mode="lines+markers", name=f"{grp} · {per} · {'계절' if sea else '기본'}",
                                           line=dict(color=PAL[j], dash="solid" if grp == "확장" else "dot", width=2.2 if per == "시험" else 1.2)))
        f.add_hline(y=0.5, line_color="#898781", annotation_text="0.5 = 동전 던지기")
        st.plotly_chart(fig_layout(f, 320, title="미루는 기간별 '지금보다 쌀 확률' 예측 AUC (품목 평균) — 실선 확장 품목, 점선 곡물", hovermode="x"), key="ex_df_auc")
        st.caption("곡물은 어떤 시계·특성에서도 0.5 근처 → 미루기 근거 없음. 확장 품목은 시계가 길수록(3–4주), 계절 특성을 넣을수록 AUC가 오름 — 휘발유·생우·돈육처럼 연중 패턴이 강한 품목 덕분.")
        auc_min = st.slider("활성화 기준 (학습 AUC >)", 0.50, 0.60, float(EX.DEFER_AUC_MIN), 0.01, key="ex_df_auc_min")
        c1_, c2_ = st.columns(2)
        tau_ = c1_.selectbox("가격 매력 문턱 τ", [0.6, 0.5, -1.0], 0, key="ex_df_tau", format_func=lambda v: "게이트 없음" if v < 0 else f"{v:.2f}")
        ps_ = c2_.selectbox("미루기 기준 p*", [0.55, 0.5, 0.6, 0.65], 0, key="ex_df_ps")
        E_ = EX.defer_evidence(df, auc_min, tau_, ps_)
        a_, n_ = E_[E_.활성], E_[~E_.활성]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("활성 품목", f"{len(a_)} / {len(E_)}")
        m2.metric("시험 AUC (활성)", f"{a_.시험AUC.mean():.3f}" if len(a_) else "–")
        m3.metric("시험 판단당 절감", f"{a_.시험절감bp.mean():+.1f}bp" if len(a_) else "–", f"위약 {a_.시험위약bp.mean():+.1f}bp" if len(a_) else None, delta_color="off")
        m4.metric("비활성 품목에 적용했다면", f"{n_.시험절감bp.mean():+.1f}bp" if len(n_) else "–")
        st.dataframe(E_.sort_values(["활성", "학습AUC"], ascending=False).style.format(
            {"학습AUC": "{:.3f}", "시험AUC": "{:.3f}", "학습절감bp": "{:+.1f}", "시험절감bp": "{:+.1f}", "시험위약bp": "{:+.1f}", "시험미룸비율": "{:.0%}", "시험적중률": "{:.0%}"}, na_rep="–"))
        import numpy as _np
        n_obs = 470; se_ = _np.sqrt((n_obs + 1) / (12 * (n_obs / 2) ** 2))
        p_null = float((0.5 + se_ * _np.random.default_rng(0).standard_normal((20000, 8))).max(1).__gt__(auc_min).mean())
        st.caption(f"시계·특성은 품목마다 8개 조합(1–4주 × 계절 유무) 중 학습 AUC 최댓값으로 고름 → 예측력이 전혀 없어도 최댓값이 {auc_min:.2f}를 넘을 확률 ≈ {p_null:.0%}. "
                   "그래서 판단은 '시험 AUC'와 '시험 절감 > 위약'으로 확인. 위약(같은 비율 무작위 미룸)은 가격이 오르는 시험 기간에 음수 — 미루기 자체가 비용임을 보여줌.")
        B_ = df[df.구분 == "백테스트"]
        G_ = B_.groupby(["tau", "p_star", "방식", "기간"])[["판단당 평균 절감 (bp)", "위약 평균 (bp)", "미룸 비율"]].mean().unstack("기간")
        tb = pd.DataFrame({"학습 절감": G_[("판단당 평균 절감 (bp)", "학습")], "시험 절감": G_[("판단당 평균 절감 (bp)", "시험")],
                           "시험 위약": G_[("위약 평균 (bp)", "시험")], "시험 미룸 비율": G_[("미룸 비율", "시험")]}).sort_values("학습 절감", ascending=False).head(10)
        with st.expander("전 품목·전 시계에 같은 설정을 일괄 적용했을 때 (학습 상위 10개 → 시험) — 과적합 확인용"):
            st.dataframe(tb.style.format({"학습 절감": "{:+.1f}", "시험 절감": "{:+.1f}", "시험 위약": "{:+.1f}", "시험 미룸 비율": "{:.0%}"}))
            st.caption("품목을 가리지 않고 같은 규칙을 쓰면 학습 기간 최선이 시험 기간에 거의 0 또는 음수 → 미루기는 '예측이 통한 품목'에서만 의미가 있음.")
    elif sec == "재고 기반 반복 구매 (학습/시험 · 예측)":
        import procurement as PR
        st.markdown("##### 재고 기반 반복 구매 — 평균 G주마다 1카고, '점수 ≥ τ면 이번 주 1카고' 규칙의 학습(2008~16) / 시험(2017~) 성과")
        df = _exp_header("procure_inventory")
        if df is None:
            return
        bt = df[df.구분 == "백테스트"]; fc = df[df.구분 == "예측"]
        a_, b_, c_ = st.columns(3)
        G_ = a_.radio("평균 구매 간격 G", sorted(bt.G.unique()), list(sorted(bt.G.unique())).index(PR.INV_G) if PR.INV_G in bt.G.unique() else 0,
                      horizontal=True, key="ex_iv_g", format_func=lambda x: f"{x}주")
        grp = b_.radio("품목군", ["곡물", "확장", "전체"], horizontal=True, key="ex_iv_grp")
        met = c_.radio("기준", ["vs정기bp", "vs균등bp"], horizontal=True, key="ex_iv_met", format_func=lambda x: {"vs정기bp": "정기 구매 대비", "vs균등bp": "매주 균등 대비"}[x])
        d = bt[(bt.G == G_) & ((bt.군 == grp) if grp != "전체" else True)]
        f = go.Figure()
        for j, (per, pw_) in enumerate([(p_, w_) for p_ in ["학습", "시험"] for w_ in [0.0, 0.6]]):
            x_ = d[(d.기간 == per) & (d.예측보류 == pw_)].groupby("tau")[met].mean()
            f.add_trace(go.Scatter(x=x_.index, y=x_.values, mode="lines+markers", name=f"{per}{' · 예측 미루기 0.6' if pw_ else ''}",
                                   line=dict(color=[PAL[0], PAL[0], PAL[1], PAL[1]][j], dash="dot" if pw_ else None, width=2.4 if per == "시험" else 1.6)))
        f.add_hline(y=0, line_color="#898781"); f.add_vline(x=PR.INV_TAU, line_dash="dot", line_color=PAL[2], annotation_text="기본 τ")
        st.plotly_chart(fig_layout(f, 340, title=f"문턱 τ별 평균 절감 (bp, {grp}, {G_}주마다 1척) — 학습(파랑) vs 시험(주황), 점선 = 예측으로 1주 미루기",
                                   xaxis_title="문턱 τ", hovermode="x unified"), key="ex_iv_line")
        M = d[(d.tau == PR.INV_TAU) & (d.예측보류 == 0)].pivot_table(index="품목", columns="기간", values=met)
        M = M.loc[M["시험"].sort_values(ascending=False).index] if "시험" in M else M
        vm = vmax_of(M.values, step=10)
        f = go.Figure(go.Heatmap(z=M.values, x=list(M.columns), y=list(M.index), zmin=-vm, zmax=vm, colorscale=DIV, texttemplate="%{z:+.0f}", colorbar=dict(thickness=12)))
        f.update_layout(height=140 + 20 * len(M), margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(autorange="reversed", dtick=1),
                        title=f"품목별 (τ = {PR.INV_TAU:.2f}) — 학습 vs 시험, 색 = bp · 학습에서 +인데 시험에서 −면 과적합/국면 변화")
        st.plotly_chart(f, key="ex_iv_heat")
        st.markdown("**1~2주 방향 예측의 실제 정확도 (walk-forward, 매년 재학습)**")
        A_ = fc.groupby(["기간", "예측주", "군"])[["하락AUC", "신호AUC"]].mean().unstack("군")
        st.dataframe(A_.style.format("{:.3f}", na_rep="–"))
        with st.expander("품목별 예측 AUC · 전체 백테스트 표"):
            st.dataframe(fc.pivot_table(index="품목", columns=["기간", "예측주"], values="하락AUC").style.format("{:.3f}", na_rep="–"))
            st.dataframe(bt.set_index(["품목", "기간", "G", "tau", "예측보류"]).drop(columns=["구분", "id", "군"]).style.format(
                {"vs정기bp": "{:+.1f}", "vs균등bp": "{:+.1f}", "신호비율": "{:.0%}", "연도t": "{:+.2f}", "연도양수비율": "{:.0%}"}, na_rep="–"))
        st.caption("읽는 법: 실선(예측 없음)의 '시험' 곡선이 실제로 기대할 수 있는 성과 — 문턱은 학습 곡선만 보고 고름. 하락 AUC 0.5 = 동전 던지기, "
                   "'신호 AUC'가 높은 것은 점수가 주 단위로 천천히 움직여서(자기상관) — 다음 주 신호 충족은 예측되지만 가격 방향은 거의 예측되지 않음.")
    elif sec == "벌크(일괄) 구매 — 문턱 · 가격 위치 창":
        import procurement as PR
        st.markdown("##### 벌크 구매 — '점수 ≥ τ 첫 주에 전량, 없으면 마감 주' 규칙 (36품목, 2008~, 4주 간격 반복)")
        df = _exp_header("procure_lump")
        if df is None:
            return
        a_, b_ = st.columns([1, 1])
        wk = a_.radio("구매 창", sorted(df.구매창주.unique()), 1, horizontal=True, key="ex_lp_wk", format_func=lambda x: f"{x}주")
        met = b_.radio("기준", ["즉시대비bp", "절감bp"], horizontal=True, key="ex_lp_met",
                       format_func=lambda x: {"즉시대비bp": "오늘 바로 사기 대비", "절감bp": "무작위 하루 대비"}[x])
        d = df[df.구매창주 == wk]
        f = go.Figure()
        for j, (grp, lb) in enumerate([(g_, l_) for g_ in ["곡물", "확장"] for l_ in sorted(d.위치창.unique())]):
            x_ = d[(d.군 == grp) & (d.위치창 == lb)].groupby("tau")[[met, "전반bp", "후반bp"]].mean()
            f.add_trace(go.Scatter(x=x_.index, y=x_[met], mode="lines+markers", name=f"{grp} · 위치창 {lb // 21}개월",
                                   line=dict(color=PAL[j % 8], width=2.4 if grp == "곡물" else 1.4, dash=None if grp == "곡물" else "dot")))
        f.add_hline(y=0, line_color="#898781"); f.add_vline(x=PR.LUMP_TAU, line_dash="dot", line_color=PAL[1], annotation_text="기본 τ")
        st.plotly_chart(fig_layout(f, 340, title=f"문턱 τ별 평균 절감 (bp, {wk}주 창) — 곡물 6품목(실선) vs 확장 30품목(점선)", xaxis_title="문턱 τ",
                                   hovermode="x unified"), key="ex_lp_line")
        M = d[(d.tau == PR.LUMP_TAU)].pivot_table(index="품목", columns="위치창", values=met)
        M.columns = [f"{c // 21}개월" for c in M.columns]
        M = M.loc[M.max(axis=1).sort_values(ascending=False).index]
        vm = vmax_of(M.values, step=10)
        f = go.Figure(go.Heatmap(z=M.values, x=list(M.columns), y=list(M.index), zmin=-vm, zmax=vm, colorscale=DIV, texttemplate="%{z:+.0f}",
                                 colorbar=dict(thickness=12)))
        f.update_layout(height=140 + 20 * len(M), margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(autorange="reversed", dtick=1),
                        title=f"품목별 (τ = {PR.LUMP_TAU:.2f}, {wk}주) — 위치창 6개월 vs 1년, 색 = bp (★ 1년 기본 품목: {', '.join(PR.BUY_IDS[i] for i in sorted(PR.LOOKBACK_1Y))})")
        st.plotly_chart(f, key="ex_lp_heat")
        with st.expander("전체 표"):
            st.dataframe(df.set_index(["품목", "위치창", "구매창주", "tau"]).drop(columns=["id"]).style.format(
                {c: "{:+.1f}" for c in df.columns if c.endswith("bp")} | {"평균매수주": "{:.1f}", "신호매수비율": "{:.0%}"}, na_rep="–"))
        st.caption("읽는 법: '오늘 바로 사기 대비'가 실제 의사결정 비교 — 곡물은 τ 0.55~0.65에서 두 하위기간 모두 +, τ가 너무 높으면(0.75) 신호가 드물어 "
                   "마감 주 강제 매수가 늘고 후반에 음수. 확장 품목 평균은 '즉시 대비' 음수 = 상승 추세가 강해 기다릴수록 손해였던 품목이 많음 → "
                   "★ 화면에서 품목별 '규칙 vs 즉시' 부호를 함께 표시. '무작위 하루 대비'가 크게 + 인 것은 '즉시'도 무작위 하루보다 나았던(추세) 효과가 섞인 값.")
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
    elif sec == "곡물 상대가치 재검증 (2008~)":
        st.markdown("##### 곡물 6품목 상대가치 재검증 — '추천' 프리셋을 정한 원래 기간 (2008~, 전반 ~2016 / 후반 2017~)에서 엄격 기준")
        df = _exp_header("procure_grains")
        if df is None:
            return
        import experiments as EX
        G = EX.grains_verdict(df)
        g = G[G.종류 != "위약"].copy()
        g["현재"] = g.상대가치.str.startswith("현재 정의")
        B_ = g[g.종류 != "대조군"].sort_values(["품목", "개선"], ascending=[True, False])
        f = go.Figure()
        f.add_trace(go.Bar(y=[f"{a} · {b}" for a, b in zip(B_.품목, B_.상대가치)], x=B_.개선, orientation="h", name="'상대가치 없음' 대비 개선",
                           marker_color=[PAL[2] if v else (PAL[0] if c else "#898781") for v, c in zip(B_.엄격채택, B_.현재)]))
        f.add_trace(go.Scatter(y=[f"{a} · {b}" for a, b in zip(B_.품목, B_.상대가치)], x=B_.위약90, mode="markers", name="위약 90%",
                               marker=dict(symbol="line-ns-open", size=12, color=PAL[7])))
        f.add_trace(go.Scatter(y=[f"{a} · {b}" for a, b in zip(B_.품목, B_.상대가치)], x=B_.대조군개선, mode="markers", name="대조군 (자기 가격 1년 위치)",
                               marker=dict(symbol="diamond", size=7, color=PAL[3])))
        f.add_vline(x=0, line_color="#898781")
        f.update_layout(height=140 + 20 * len(B_), margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(autorange="reversed", dtick=1),
                        title="개선 (bp, 13·26주 평균) — 파랑 = 현재 대시보드 정의, 초록 = 엄격 기준 통과, 회색 = 기타 후보")
        st.plotly_chart(f, key="ex_gr_bar")
        T = g.set_index(["품목", "상대가치"])[["종류", "개선", "전반", "후반", "두기간통과", "위약90", "대조군개선", "엄격채택"]]
        st.dataframe(T.style.format({"개선": "{:+.1f}", "전반": "{:+.1f}", "후반": "{:+.1f}", "위약90": "{:+.1f}", "대조군개선": "{:+.1f}"}, na_rep="–"))
        pl = df[df.종류 == "위약"].groupby(["id", "상대가치"]).agg(a=("개선_전반", "min"), b=("개선_후반", "min"), m=("개선_평균절감", "mean"))
        cur = df[df.상대가치.str.startswith("현재 정의")]
        c1, c2, c3 = st.columns(3)
        c1.metric("현재 정의 6품목 평균 개선", f"{cur.개선_평균절감.mean():+.1f}bp", f"전반 {cur.개선_전반.mean():+.1f} / 후반 {cur.개선_후반.mean():+.1f}", delta_color="off")
        c2.metric("위약 평균 개선 · 두 기간 통과율", f"{pl.m.mean():+.1f}bp", f"{((pl.a > 0) & (pl.b > 0)).mean():.0%} 우연 통과", delta_color="off")
        c3.metric("대조군 평균 개선 (자기 가격 1년 위치)", f"{df[df.종류 == '대조군'].개선_평균절감.mean():+.1f}bp")
        st.caption("전반·후반은 '전체 기간 평균'이 아니라 13·26주 중 나쁜 쪽(min) — 둘 다 0보다 커야 '두 기간 통과'. "
                   "판정: 현재 정의 6개 모두 미통과 — 옥수수/밀은 13주 후반 −7bp, 밀/옥수수·KC밀/옥수수는 전반 −, 크러시 마진은 26주 전반 −1bp, "
                   "대두박 가치/대두는 두 기간 통과지만 자기 가격 대조군(+16bp)보다 작고, 대두유는 전반 −31bp로 오히려 손해. "
                   "6품목 평균 개선은 전반 −, 후반 + (2017~ +7~12bp)로 v1.2 단계별 검증의 '두 기간 모두 +'가 이 프리셋 구성(6개월 창·MIDAS·계절)에서는 재현되지 않음. "
                   "유일한 통과 = 대두 CBOT/DCE(수입 패리티, +12bp, 위약 30개 90% +9bp도 초과) — 다만 후보 5개 중 사후 선택, t≈1.8, DCE 1호 대두는 비GMO 국산이라 "
                   "수입 패리티 연결이 약하고 DCE 연속월물 롤 보정 없음 → 즉시 채택하지 않고 권고로 기록 (HANDOFF). 프리셋 가중치는 바꾸지 않음.")
    elif sec == "최신 시계열 모델 · 예측 목표":
        st.markdown("##### 최신 시계열 파운데이션 모델 vs 기준 모델 — 목표를 바꾸면 예측력이 생기나 (9개 구매 품목, 2012~ walk-forward, 매주 금요일 결정)")
        st.caption("파운데이션 모델 = 사전학습만 된 zero-shot (Chronos-Bolt-small·Chronos-2: Amazon, TimesFM 2.5: Google Research, TiRex: NX-AI xLSTM — "
                   "TiRex는 CPU에서 느려 4주마다 결정). 기준 = 과거 평균·모멘텀(12-1)·단기 반전·LightGBM 글로벌(9품목 함께 학습, 매년 재학습, Purged K-fold로 "
                   "하이퍼파라미터 선택). 주의: 파운데이션 모델의 사전학습 자료에 평가 기간의 원자재 시계열이 일부 포함됐을 수 있음(낙관 편향 가능).")
        df = _exp_header("forecast_lab")
        if df is None:
            return
        g = lambda k: df[df.구분 == k].dropna(axis=1, how="all")
        t1, t2, t3, t4, t5, t6, t7 = subtabs(["방향 (5일)", "변동성 (20일)", "스프레드 (상대가치)", "품목 순위", "구매: 기다릴까", "과적합 판정", "메타 라벨링"])
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


def tab_factor():
    st.subheader("요인 모형 — 주성분 · 스필오버(연결성) · TVP-VAR · FAVAR")
    st.caption("여러 시장의 움직임을 소수의 공통 요인이나 '충격이 누구에게서 누구로 퍼지나'로 요약. 거시 변수는 변화율(금리·스프레드는 차분, 나머지는 로그수익률)로 함께 넣음.")
    f1_, f2_ = st.columns([3, 2])
    ids = f1_.multiselect("원자재", COMMOD_IDS, default=[i for i in DEF_COMOVE if i in COMMOD_IDS], format_func=lbl, key="fm_ids")
    mac_opts = [i for i in ALL_IDS if GROUP.get(i) in ("Macro", "FX", "FRED daily")]
    macs = f2_.multiselect("거시 변수 (PCA · 스필오버 · TVP-VAR에 함께 반영)", mac_opts, [i for i in ["fred:VIXCLS", "yf:DX-Y.NYB", "fred:DGS10"] if i in mac_opts],
                           format_func=lbl, key="fm_macro", help="레짐 탭과 같은 거시 변수 목록. VAR는 정상 시계열이 필요해 수준 대신 변화로 넣음")
    R = panel_or_stop(ids + [m for m in macs if m not in ids])
    if R is None:
        return
    st.markdown("##### 주성분 분석 (PCA)")
    ev, load, scores = A.pca(R)
    k = min(4, R.shape[1])
    cp = st.columns([1, 2])
    f = go.Figure(go.Bar(x=[f"PC{i+1}" for i in range(len(ev))], y=ev * 100, marker_color=PAL[0], name="설명력"))
    f.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(ev))], y=np.cumsum(ev) * 100, name="누적",
                           line=dict(color=PAL[1], width=2)))
    cp[0].plotly_chart(fig_layout(f, 380, title="Scree (설명분산 %)", hovermode="x"), key="scree")
    cp[1].plotly_chart(heat(load.iloc[:, :k].T, "적재값 (Loadings) — PC1≈공통 원자재 요인", h=380, text_max=30), key="load")
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
        fdy = heat(core, f"스필오버 표 (%) — 총 스필오버 지수 {tot:.1f}% (대각 = 자기 몫, 색에서 제외)", 0, None,
                   ".1f", scale=DIV, text_max=30, nodiag=True, h=max(420, 32 * len(core) + 120))
        fdy.update_traces(textfont=dict(size=9))
        fdy.update_xaxes(tickfont=dict(size=10)); fdy.update_yaxes(tickfont=dict(size=10))
        x1.plotly_chart(fdy, key="dyh_")
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
        netm = cn["net"].resample("YE").mean()
        vmx = float(np.abs(netm.values).max()) or 1
        f = go.Figure(go.Heatmap(z=netm.T.values, x=[str(d.year) for d in netm.index], y=[short(c) for c in netm.columns], zmin=-vmx, zmax=vmx,
                                 colorscale=DIV, texttemplate="%{z:.0f}", textfont=dict(size=9), hovertemplate="%{y} %{x}: %{z:.1f}<extra></extra>"))
        f.update_layout(title="순 전이 (TO − FROM) — 연평균 · 빨강 = 충격 전달자, 보라·파랑 = 수신자", height=140 + 30 * netm.shape[1],
                        margin=dict(l=10, r=10, t=40, b=10), xaxis=dict(dtick=1, tickangle=-45, tickfont=dict(size=10)), yaxis=dict(tickfont=dict(size=10)))
        st.plotly_chart(f, key="tvp_net")


def page_structure():
    tab_comove()


def page_factor():
    a, b = subtabs(["주성분 · 스필오버 · TVP-VAR", "FAVAR (요인 충격의 전파)"])
    with a:
        tab_factor()
    with b:
        tab_favar()


def page_vol():
    a, b, c, dd = subtabs(["단변량 GARCH · 레버리지", "다변량 GARCH (DCC · ADCC · BEKK)", "GARCH-MIDAS (장기·단기 변동성 분해)",
                            "변동성 예측 검증 (QLIKE · DM · MCS · 부호 편향)"])
    with a:
        tab_vol()
    with b:
        tab_mvgarch()
    with c:
        tab_midas(vol_only=True)
    with dd:
        vol_forecast_check()


def _grid_preselect(T: pd.DataFrame, mode: str):
    """사전 선택 검증: 2007–2016 성과로만 조합을 고르고 2017년 이후 성과로 확인 (전략 전체, 위약 제외)."""
    tr, te = f"{mode} 2007–2016", f"{mode} 2017~"
    with st.expander(f"사전 선택 검증 — 2007–2016 성과로 상위 조합을 고르면 2017년 이후에도 좋았나 ({mode})", expanded=True):
        if te not in T:
            st.caption("결과표를 다시 계산하는 중 — 끝나면 표시됩니다.")
            return
        C = T[~T["설명변수"].str.startswith(("⑧", "P⑤"))].dropna(subset=[tr, te]).copy()
        C["시험 순위"] = C[te].rank(ascending=False).astype(int)
        rho = C[tr].corr(C[te], method="spearman")
        top = C.sort_values(tr, ascending=False).head(10)
        mom = T[T["모델"] == "52주 모멘텀"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("학습 1위의 2017~ Sharpe", f"{top[te].iloc[0]:+.2f}", f"시험 {int(top['시험 순위'].iloc[0])}위 / {len(C)}", delta_color="off")
        m2.metric("학습 상위 10개 평균 (2017~)", f"{top[te].mean():+.2f}", f"전체 평균 {C[te].mean():+.2f}", delta_color="off")
        m3.metric("52주 모멘텀 (2017~)", f"{mom[te].iloc[0]:+.2f}" if len(mom) else "–")
        m4.metric("학습·시험 Sharpe 순위 상관", f"{rho:+.2f}", "0 근처 = 과거 순위가 미래를 못 맞힘", delta_color="off")
        st.dataframe(top[["모델군", "모델", "설명변수", tr, te, "시험 순위"]].style.format({tr: "{:+.2f}", te: "{:+.2f}"}), hide_index=True, use_container_width=True)
        f = go.Figure()
        for j, (g, G) in enumerate(C.groupby("모델군")):
            f.add_trace(go.Scatter(x=G[tr], y=G[te], mode="markers", name=g, marker=dict(size=7, color=PAL[j % 8], opacity=0.75),
                                   text=G["모델"] + " · " + G["설명변수"], hovertemplate="%{text}<br>2007–16 %{x:+.2f} → 2017~ %{y:+.2f}<extra></extra>"))
        f.add_hline(y=0, line_color="#898781", line_width=1); f.add_vline(x=0, line_color="#898781", line_width=1)
        st.plotly_chart(fig_layout(f, 340, title=f"전략별 2007–2016 Sharpe(학습) vs 2017~ Sharpe(시험) — {len(C)}개, 위약 제외", xaxis_title="2007–2016", yaxis_title="2017~",
                                   hovermode="closest"), key=f"pg_pre_{mode}")
        st.caption("고르는 과정까지 표본 외로 둔 검증: 시험 기간 성과는 고를 때 보지 않음. 학습 상위가 시험에서도 상위·양수면 그 조합의 우위가 지속적이라는 근거, "
                   "순위 상관이 0 근처이고 상위 10개 평균이 전체 평균과 비슷하면 2007–2016의 순위는 운이 컸다는 뜻. 대안 시계열(neuralforecast)은 2년마다 재학습판.")


def _predict_grid_section():
    import experiments as EX
    if "predict_grid" not in EX.EXPERIMENTS:
        import importlib
        EX = importlib.reload(EX)
    st.subheader("전략 백테스트 — 예측 모델 × 설명변수 조합 (19개 품목)")
    st.caption(f"📌 결과 고정: {PLS_MD} (`experiments.predict_grid` · 대안 모델은 `predict_grid_alt`로 미리 계산). 데이터 업데이트를 해도 다시 계산하지 않습니다.")
    st.caption("공통 틀: 롱숏 탭과 같은 원자재 19개 · 매주 금요일 결정 · 목표 = 4주 뒤 로그수익률 · 품목마다 매년 1월 재학습(그해 이전 자료만, 학습 끝 4주 purge, 2001년부터 확장 창) · "
               "예측 → 두 방식으로 매매: 방향 = 예측 부호 ÷ 품목 변동성(품목마다 롱/숏), 순위 롱숏 = 예측 ÷ 변동성 상위 1/3 롱·하위 1/3 숏(4주 보유, 달러 중립). 편도 5bp · 목표 변동성 10%. "
               "설명변수는 모두 그 시점까지 자료로만 표준화. 표준 모델 = OLS · Ridge · Lasso · Elastic Net · PLS · LightGBM (λ·요인 수는 학습 구간 시계열 교차검증). "
               "대안 시계열 = Prophet · PatchTST · iTransformer · TFT · Neural ODE · GNN · Chronos · TimesFM (설명변수 없음 / 매크로 / 멀티모달). "
               "패널 = 여러 품목 탭 모델 8종 × 특성 묶음 5개, MIDAS = MIDAS 탭 Beta 가중·동일가중 × 일간 설명변수 3가지.")
    T, meta = EX.load("predict_grid")
    fw = EX.DATA / "predict_grid_weekly_compare.csv"
    if T is None or not fw.exists():
        st.info("결과가 없습니다 — `EX.predict_grid()` 실행 필요 (표준 모델 약 10분).")
        return
    s_ = meta.get("settings", {})
    st.caption(f"계산 시각 {meta.get('run_at', '?')} · 표준 모델 {len(s_.get('표준모델', []))}개 · 대안 모델 조합 {s_.get('대안모델수', 0)}개 · 전략 {len(T)}개")
    with st.expander("설명변수 묶음 — 무엇을 넣었나", expanded=False):
        st.markdown("""
| 묶음 | 내용 |
|---|---|
| ① 자기 정보만 | 그 품목의 1·4·13·26·52주 모멘텀, 26주 변동성 (6개) |
| ② + 19개 원자재 | 다른 18개 품목의 4주·52주 수익률 (36개) |
| ③ + 거시 | VIX(로그) · 신용 스프레드 · 달러 4주 변화 · 10년/3개월 금리 4주 변화 · S&P500 4주 · BDI 4주 (1주 늦춤) |
| ④ 일반적 설정 | 자기 + 19개 원자재 + 거시 |
| ⑤-1~7 멀티모달 갈래 | 자기 + 텍스트(FinBERT) / 리포트 수치 / COT / 기상 / 매크로 β / 금융 β / 월간 거시 β 하나씩 |
| ⑥ 멀티모달 전부 | 자기 + 7갈래 54개 |
| ⑦ 전부 | 자기 + 19개 원자재 + 거시 + 멀티모달 |
| ⑧ 위약 | ⑦과 같지만 자기 정보 외 변수의 시간 순서를 13주 블록으로 섞음 → 설명변수가 진짜 정보인지 판별 기준 |
| 대안 모델 '매크로 + 공통 멀티모달' | 매크로 3종 + ONI·SOI·지구 기온 + NFCI·STLFSI·실질금리·5y5y·금리차 (모든 품목 같은 값) |
| 대안 모델 '품목별 멀티모달 7갈래' | 품목마다 7갈래 요약값(갈래 안 특성들의 과거 표준화 평균) — TFT·Prophet 방식만 |
""")
    k1, k2 = st.columns([1, 1.3])
    mode = k1.radio("매매 방식", ["순위 롱숏", "방향"], horizontal=True, key="pg_mode")
    grp = k2.multiselect("모델군", sorted(T["모델군"].unique()), sorted(T["모델군"].unique()), key="pg_grp")
    D = T[T["모델군"].isin(grp)].copy()
    sc = f"{mode} Sharpe"
    D = D.sort_values(sc, ascending=False)
    cols = ["모델군", "모델", "설명변수", "예측 시작", sc, f"{mode} 위약 대비 차이", f"{mode} 위약 대비 차이 t", f"{mode} 2007–2016", f"{mode} 2017~",
            f"{mode} 2017–2021-10", f"{mode} 최근 5년", f"{mode} 공통 2012-07~",
            f"{mode} t값", f"{mode} 최대낙폭 (%)", f"{mode} 52주 모멘텀 대비 차이 t", "IC (주별 순위 상관)", "방향 적중률"]
    cols = [c for c in cols if c in D.columns]
    fmt = {c: "{:+.2f}" for c in cols[4:]}
    fmt.update({"IC (주별 순위 상관)": "{:+.3f}", "방향 적중률": "{:.1%}", f"{mode} 최대낙폭 (%)": "{:.1f}"})
    st.dataframe(D[cols].style.format(fmt, na_rep="–"), hide_index=True, use_container_width=True, height=420)
    n_ = len(T)
    se = 1 / np.sqrt(19.8)
    thr = se * np.sqrt(2 * np.log(max(n_, 2)))
    pl = T[T["설명변수"].str.startswith("⑧")][sc]
    base = T[T["모델"] == "52주 모멘텀"][sc]
    if len(D):
        best = D.iloc[0]
        st.info(f"**최고 = {best['모델']} · {best['설명변수']}** ({mode} Sharpe {best[sc]:+.2f} · 52주 모멘텀 대비 차이 t {best[f'{mode} 52주 모멘텀 대비 차이 t']:+.2f}). "
                f"기준 52주 모멘텀 {base.iloc[0]:+.2f}. 전략을 {n_}개 시험했으므로 예측력이 전혀 없어도 최댓값은 약 +{thr:.2f}까지 나올 수 있음 (Sharpe 표준오차 {se:.2f} × √(2 ln N)) — "
                f"위약(⑧, 설명변수 시간 섞기) 모델 6개 평균 {pl.mean():+.2f} · 최고 {pl.max():+.2f}. 진짜 정보라면 같은 모델의 위약보다 확실히 높고 하위 기간 모두 양수여야 함.")
    if f"{mode} 위약 대비 차이" in T:
        st.caption("위약 대비 차이 = 같은 모델·같은 설명변수 개수인데 설명변수의 시간 순서만 섞은 판(표준 ⑧ · 패널 P⑤)과의 Sharpe 차이, t = 두 전략 주간 수익 차이의 t값. "
                   "|t| ≥ 2여야 설명변수가 '진짜 정보'라고 볼 수 있음. 대안 시계열·MIDAS는 위약판이 없어 빈칸.")
    _grid_preselect(T, mode)
    st.markdown(f"##### 표준 모델 — 모델 × 설명변수 ({mode} Sharpe, 2007–)")
    S = T[T["모델군"] == "표준"].pivot_table(index="설명변수", columns="모델", values=sc)
    if len(S):
        S = S[[m for m in ["OLS", "Ridge", "Lasso", "Elastic Net", "PLS", "LightGBM"] if m in S]]
        f = go.Figure(go.Heatmap(z=S.values, x=list(S.columns), y=list(S.index), zmin=-0.6, zmax=0.6, colorscale=DIV,
                                 text=[[f"{v:+.2f}" for v in r] for r in S.values], texttemplate="%{text}", hovertemplate="%{y} · %{x}: %{z:+.2f}<extra></extra>"))
        f.update_layout(height=120 + 30 * len(S), margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(f, key="pg_heat")
        rowm = S.mean(1).sort_values(ascending=False)
        plr = [i for i in rowm.index if i.startswith("⑧")]
        st.caption("행 평균(모델 6개) 상위: " + " · ".join(f"{k.split(' ', 1)[0]} {v:+.2f}" for k, v in rowm.head(4).items()) +
                   " | 하위: " + " · ".join(f"{k.split(' ', 1)[0]} {v:+.2f}" for k, v in rowm.tail(3).items()) +
                   (f" — 위약(⑧) 행 평균 {rowm[plr[0]]:+.2f}" if plr else "") + ". 열(모델) 평균: " +
                   " · ".join(f"{k} {v:+.2f}" for k, v in S.mean(0).sort_values(ascending=False).items()))
    A_ = T[T["모델군"] == "대안 시계열"]
    if len(A_):
        st.markdown(f"##### 대안 시계열 모델 — 모델 × 설명변수 ({mode} Sharpe)")
        S2 = A_.pivot_table(index="모델", columns="설명변수", values=sc)
        f = go.Figure(go.Heatmap(z=S2.values, x=list(S2.columns), y=list(S2.index), zmin=-0.6, zmax=0.6, colorscale=DIV,
                                 text=[["" if not np.isfinite(v) else f"{v:+.2f}" for v in r] for r in S2.values], texttemplate="%{text}",
                                 hovertemplate="%{y} · %{x}: %{z:+.2f}<extra></extra>"))
        f.update_layout(height=120 + 30 * len(S2), margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(f, key="pg_heat_alt")
        st.caption("빈칸 = 그 조합은 돌리지 않음 (PatchTST·Neural ODE·zero-shot은 외부 변수를 받지 않는 구조, 품목별 7갈래는 TFT·Prophet만). "
                   "경량판은 매년, neuralforecast 정식판(PatchTST·iTransformer·TFT)은 계산 시간 때문에 2년마다 재학습(2년 묶음의 첫 결정일까지 자료로 학습해 그 2년을 예측). "
                   "zero-shot은 학습 없음. 시드 1개 — 같은 모델도 시드에 따라 Sharpe가 ±0.3 정도 흔들림 (⑤ 대안 시계열 탭).")
    else:
        st.caption("대안 시계열 모델 결과는 계산 중 — 끝나면 이 자리에 표시됩니다 (`EX.predict_grid_alt()`).")
    for g_, ttl_, note_ in (("패널", "여러 품목 탭 패널 모델 — 모델 × 특성 묶음",
                             "여러 품목 탭과 같은 엔진: 특성을 매 시점 품목 간 순위로 바꿔 19개 품목을 한 모델로 학습, 목표 = 다음 주 변동성 표준화 상대 수익(시점 평균 제거), "
                             "26주마다 재학습 · 최소 학습 156주. 상대 수익을 예측하므로 '방향' 매매도 사실상 품목 간 상대 베팅. P⑤ = 거시·멀티모달 특성의 날짜를 섞은 위약. "
                             "P③-요약 = 멀티모달 54개 대신 7갈래 요약값 7개만 — P③보다 나으면 54개를 넣어 생긴 과적합, 비슷하게 나쁘면 정보 자체가 없음."),
                            ("MIDAS", "MIDAS 탭 — 가중 방식 × 일간 설명변수",
                             "MIDAS 탭과 같은 엔진: 월말마다 그때까지 자료로 재추정(최소 60개월) → 다음 달 수익 예측, 일간 시차 63개를 Beta 가중(MIDAS) 또는 월 평균(동일가중)으로 요약. "
                             "월 1회 예측을 그 달 매주 같은 값으로 매매. M③ 멀티모달 종합 = 품목별 7갈래 요약값의 평균(주간 → 일간 계단식).")):
        G_ = T[T["모델군"] == g_]
        if not len(G_):
            st.caption(f"{ttl_.split(' — ')[0]} 결과는 계산 중 — 끝나면 이 자리에 표시됩니다 (`EX.predict_grid_panel_midas()`).")
            continue
        st.markdown(f"##### {ttl_} ({mode} Sharpe, 2007–)")
        S3 = G_.pivot_table(index="설명변수", columns="모델", values=sc)
        f = go.Figure(go.Heatmap(z=S3.values, x=list(S3.columns), y=list(S3.index), zmin=-0.6, zmax=0.6, colorscale=DIV,
                                 text=[["" if not np.isfinite(v) else f"{v:+.2f}" for v in r] for r in S3.values], texttemplate="%{text}",
                                 hovertemplate="%{y} · %{x}: %{z:+.2f}<extra></extra>"))
        f.update_layout(height=120 + 34 * len(S3), margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(f, key=f"pg_heat_{g_}")
        st.caption(note_)
    W = pd.read_csv(fw, index_col=0, parse_dates=True, encoding="utf-8-sig")
    lab = " | " + mode
    opts = [c for c in W.columns if c.endswith(lab)]
    top = [f"{r['모델']} · {r['설명변수']}{lab}" for _, r in D.head(4).iterrows()]
    dflt = [c for c in top if c in opts] + [c for c in opts if c.startswith("52주 모멘텀") and c not in top]
    sel = st.multiselect("누적 수익 비교", opts, dflt, key=f"pg_sel_{mode}", format_func=lambda c: c.replace(lab, ""))
    if sel:
        f = go.Figure()
        for j, c in enumerate(sel):
            r = W[c].dropna()
            bse = c.startswith("52주 모멘텀")
            f.add_trace(go.Scatter(x=r.index, y=r.cumsum(), name=c.replace(lab, ""), line=dict(width=2.2 if bse else 1.3, dash="dash" if bse else None,
                                                                                         color=("#e8e6e1" if DARK else "#0b0b0b") if bse else PAL[j % 8])))
        st.plotly_chart(shade(fig_layout(f, 380, title=f"누적 수익 (%, {mode}, 목표 변동성 10%) — 점선 = 52주 모멘텀", hovermode="x",
                                         legend=dict(orientation="h", y=-0.3, font=dict(size=10))), W.index), key="pg_cum")
    st.caption("읽는 법: Sharpe 95% 구간은 2007– 전체 약 ±0.45. 표준 모델은 같은 모델의 ①(자기 정보만)·⑧(위약)과 비교해 설명변수를 넣은 효과를 보고, "
               "대안 모델은 '설명변수 없음' 열과 비교. 순위 롱숏은 품목 간 상대 예측, 방향은 품목별 절대 예측을 평가 — 둘이 엇갈리면 상대 순위만 맞히거나 시장 방향만 맞힌 것.")


def page_predict():
    a, b, c, d = subtabs(["단일 품목 (회귀 · ML · 계량)", "여러 품목 (패널 · 횡단면)", "MIDAS 수익 예측 (혼합 주기)",
                          f"전략 백테스트 — 모델 × 설명변수 ({PLS_S})"])
    with a:
        tab_ml()
    with b:
        tab_panel()
    with c:
        tab_midas(vol_only=False)
    with d:
        _predict_grid_section()


# 롱숏·대안 시계열·멀티모달 결과는 고정 — 데이터 업데이트(캐시 비움)해도 다시 계산하지 않음 (계산이 길어서).
# 기준일을 바꾸려면 FROZEN_END를 고치고 data/frozen/을 지운 뒤 대안 시계열·멀티모달은 experiments.py로 다시 실행.
FROZEN_END = "2026-10-26"
FROZEN_LBL = "2026.10.26"
PERIOD_LS, PERIOD_ALT, PERIOD_MM = f"2007.01~{FROZEN_LBL}", f"2012.07~{FROZEN_LBL}", f"2007.01~{FROZEN_LBL}"
# 캡션(마크다운)에서는 "~" 두 개가 취소선이 되므로 이스케이프한 판을 씀
PLS_MD, PALT_MD, PMM_MD = (p_.replace("~", "\\~") for p_ in (PERIOD_LS, PERIOD_ALT, PERIOD_MM))
# 하위 탭 제목용 짧은 기간 (2007.01~2026.10.26 → '07.01~'26.10.26) — 탭이 한 화면에 들어가도록
PLS_S, PALT_S, PMM_S = (p_.replace("20", "'", 1).replace("~20", "~'") for p_ in (PERIOD_LS, PERIOD_ALT, PERIOD_MM))


def subtabs(labels: list[str]):
    """하위 탭: 제목 앞에 A. B. C. … 를 붙여 순서를 보이게 함."""
    return st.tabs([f"{chr(65 + k)}. {l_}" for k, l_ in enumerate(labels)])


def _frozen(key: str, fn):
    """디스크 고정 캐시: data/frozen/{key}.pkl 이 있으면 읽고, 없으면 FROZEN_END까지 자료로 한 번 계산해 저장."""
    from pathlib import Path
    p = Path(__file__).parent / "data" / "frozen" / f"{key}_{FROZEN_END}.pkl"
    if p.exists():
        return pd.read_pickle(p)
    out = fn()
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, p)
    return out


def _levels_frozen(roll: bool) -> pd.DataFrame:
    return _levels(None, roll).loc[:FROZEN_END]


@st.cache_data(show_spinner=False)
def _ls_run(cost_bp: float, with_hmm: bool, roll: bool):
    import ls_strategies as LS
    return _frozen(f"ls_run_{cost_bp:g}_{with_hmm}_{roll}", lambda: LS.run_all(_levels_frozen(roll), cost_bp, with_hmm=with_hmm))


@st.cache_data(show_spinner=False)
def _alloc_run(cost_bp: float, roll: bool):
    import ls_strategies as LS
    return _frozen(f"alloc_{cost_bp:g}_{roll}", lambda: LS.run_alloc(_levels_frozen(roll), cost_bp))


@st.cache_data(show_spinner=False)
def _alloc_robust(cost_bp: float, roll: bool):
    import ls_strategies as LS
    return _frozen(f"alloc_robust_{cost_bp:g}_{roll}", lambda: LS.run_alloc_robust(_levels_frozen(roll), cost_bp))


@st.cache_data(show_spinner=False)
def _sizing_run(cost_bp: float, roll: bool):
    import ls_strategies as LS

    def run():
        L = _levels_frozen(roll)
        return LS.run_sizing(L, cost_bp)[0], LS.run_windows(L, cost_bp)
    return _frozen(f"sizing_{cost_bp:g}_{roll}", run)


@st.cache_data(show_spinner=False)
def _lx_run(cost_bp: float, roll: bool):
    import ls_extra as LX
    return _frozen(f"lx_{cost_bp:g}_{roll}", lambda: LX.run_extra(_levels_frozen(roll), cost_bp))


@st.cache_data(show_spinner=False)
def _lx_contrib(roll: bool):
    import ls_extra as LX
    return _frozen(f"lx_contrib_{roll}", lambda: LX.item_contrib(_levels_frozen(roll)))


def tab_ls():
    import ls_strategies as LS
    import ls_extra as LX
    import procurement as PR
    st.subheader("롱숏 전략 — 상관 · 레짐 · 추세 (헤지펀드 원자재 전략 묶음)")
    st.caption(f"📌 결과 고정: {PLS_MD} 자료 기준으로 한 번 계산해 저장한 값입니다. 데이터 업데이트를 해도 다시 계산하지 않습니다 (계산 시간이 길어서).")
    st.caption("주간 금요일 종가로 결정 → 다음 주 수익률로 손익 (그 시점까지의 정보만 사용) · 비용 = 비중 변화량 × 편도 비용. "
               "① 페어 평균회귀 = 관계가 강한 두 품목의 가격차가 평소보다 벌어지면 되돌림에 베팅 · ② 상관 급등 = 품목들이 한 방향으로 묶이기 시작하면(흡수비율↑) 동반 하락 대비 · "
               "③ 국면 진입 = 다변량 HMM 스트레스 국면 확률 > 0.5 이면 위험 회피 · ④ 시계열 모멘텀 = 품목별 52주 추세 방향 (CTA의 기본 전략) · ⑤ 횡단면 모멘텀 = 강한 품목 롱 · 약한 품목 숏 · "
               "⑥ 비중 조합 = 최소분산 롱(공분산 최적화)과 ④ 모멘텀을 30/70 · 50/50 · 70/30으로 섞음 (각 10% 변동성 → 섞은 뒤 다시 10%) · "
               "⑦ 국면 전환 = 평소엔 ⑥ 70/30, 스트레스(HMM)·상관 급등(AR) 신호가 켜진 다음 주엔 모멘텀 100% / 현금 / 금 롱·경기민감 숏 혼합으로 바꿈")
    with st.expander("헤지펀드·CTA가 원자재에서 흔히 쓰는 전략과 이 화면의 범위"):
        st.markdown("""
| 전략 | 핵심 | 여기서 | 비고 |
|---|---|---|---|
| **시계열 모멘텀 (추세추종, CTA)** | 품목별 과거 수익률 부호대로 롱/숏, 변동성 역비례 | ④ | 100년 이상 자료에서 확인된 대표 전략 (Hurst-Ooi-Pedersen) |
| **횡단면 모멘텀** | 강한 품목 롱 · 약한 품목 숏 | ⑤ | 원자재에서는 추세보다 약하고 기간 의존 |
| **캐리 (기간구조)** | 백워데이션 품목 롱 · 콘탱고 품목 숏 | – | v1.14.0 검증(에너지 음수·중국 약한 양수)에서 견고하지 않아 화면에서 제외 (`collect_carry.py`·`LS.run_carry`는 남김) |
| **페어 · 스프레드 차익** | 크러시·크랙·곡물 비율 등 관계의 평균회귀 | ① | 비용·관계 붕괴(구조 변화)에 취약 |
| **상관 · 위험 국면 타이밍** | 흡수비율·VIX·신용 스프레드로 위험 축소 | ② ③ | 수익보다 낙폭 축소 목적 |
| **가치 (장기 역추세)** | 5년 평균 대비 크게 싼 품목 롱 | ⑨ V1 | 모멘텀과 음의 상관 → 결합 효과 (⑨) |
| **칼만 필터** | 적응형 추세 · 동적 헤지 페어 | ⑨ K1·K2 | 이 자료에선 52주 모멘텀·MinVar 혼합보다 약함 |
| **섹터 상대가치 · 과열 역발상** | 섹터 안 과열 숏 · 낙오 롱, 저베타, 왜도, 저변동성, COT | ⑨ | 주식의 '대장주 숏·후발주 롱'을 원자재로 |
| **펀더멘털 · 포지션** | 재고·작황·COT 극단에서 역방향 | – | 펀더멘털 표는 참고용 (예측력 미재현) |
| **변동성 · 분산(dispersion)** | 옵션 내재변동성·상관 프리미엄 | ✗ | 옵션 자료 없음 |
""")
    c1, c2, c3 = st.columns([1, 1, 2])
    cost = c1.number_input("편도 비용 (bp)", 0.0, 50.0, 5.0, 1.0, key="ls_cost")
    hmm = c2.checkbox("HMM 스트레스 국면 포함 (+20초)", True, key="ls_hmm")
    with st.spinner("전략 계산 중… (처음 20–40초)"):
        o = _ls_run(float(cost), bool(hmm), ROLL)
    T, D = o["table"].copy(), o["returns"].copy()
    sp = [c for c in T.columns if c.startswith("Sharpe ~") or c.startswith("Sharpe 2")]
    # ⑩ 혼합 후보 (⑨ 사전 선택 검증 상위): 가치 · 섹터 내 따라잡기 · 칼만 페어를 MinVar 50 + 모멘텀 25에 25%
    LXo = _lx_run(float(cost), ROLL)
    for key_, nm_ in [("V1 가치", "가치"), ("S2 섹터 내 따라잡기", "섹터 내 따라잡기"), ("K2 칼만 동적 헤지 페어", "칼만 페어")]:
        ck = f"혼합 · MinVar 50 + 모멘텀 25 + {key_} 25"
        if ck not in LXo["cands"]:
            continue
        r_ = LXo["cands"][ck].reindex(D.index).fillna(0.0)
        rn = f"⑩ 혼합 후보 · MinVar 50 + 모멘텀 25 + {nm_} 25"
        D[rn] = r_
        m_ = LS.metrics(r_)
        T.loc[rn] = {**m_, sp[0]: LS.metrics(r_[r_.index < "2017-01-01"])["Sharpe"], sp[1]: LS.metrics(r_[r_.index >= "2017-01-01"])["Sharpe"],
                     "보유 주 비율": float((r_ != 0).mean()), "바스켓과 상관": float(r_.corr(D.iloc[:, 0])), "t값": m_["Sharpe"] * np.sqrt(len(r_) / 52)}
    st.dataframe(T.style.format({"Sharpe": "{:+.2f}", "연수익 (%)": "{:+.2f}", "연변동성 (%)": "{:.1f}", "최대낙폭 (%)": "{:.1f}", sp[0]: "{:+.2f}", sp[1]: "{:+.2f}",
                                 "보유 주 비율": "{:.0%}", "바스켓과 상관": "{:+.2f}", "t값": "{:+.2f}", "위약 95%": "{:+.2f}"}, na_rep="–"),
                 height=min(38 * (len(T) + 1) + 3, 470))
    st.caption(f"전략 {len(T)}개 — 표 안에서 스크롤. ⑩ 혼합 후보 = ⑨ '사전 선택 검증'에서 2016년까지 학습 Sharpe 상위 조합(가치 1위 · 따라잡기 2위 · 칼만 페어 4위, 3위 왜도는 단독 성과가 약해 제외)")
    sel = st.multiselect("누적 수익 비교", list(D.columns), [c for c in D.columns if c.startswith(("기준", "④ 시계열 모멘텀 52", "⑥ 혼합 · MinVar 70",
                                                                                             "⑩ 혼합 후보 · MinVar 50 + 모멘텀 25 + 가치", "⑩ 혼합 후보 · MinVar 50 + 모멘텀 25 + 섹터"))], key="ls_sel")
    norm_ = st.checkbox("변동성 10%로 맞춰 비교 (사후 배율 — 모양 비교용)", True, key="ls_norm")
    if sel:
        f = go.Figure()
        for j, c in enumerate(sel):
            r = D[c]
            if norm_ and r.std() > 0:
                r = r * (10 / (r.std() * np.sqrt(52)))
            f.add_trace(go.Scatter(x=r.index, y=r.cumsum(), name=c, line=dict(width=2.2 if c.startswith("④ 시계열 모멘텀 52") else 1.3, color=PAL[j % 8])))
        st.plotly_chart(shade(fig_layout(f, 380, title="누적 수익 (%, 로그 합)", hovermode="x"), D.index), key="ls_cum")
    with st.expander("① 페어별 성과 · 스프레드 z 점수"):
        PB = o["pairs"].get("pairs_b")
        import procurement as PR
        kn = lambda c: PR.BUY_IDS.get(f"yf:{c}", NAME.get(f"yf:{c}", c))
        if PB is not None and len(PB):
            PB = PB.copy(); PB["페어"] = PB["페어"].map(lambda x: " / ".join(kn(c) for c in x.split("/")))
            st.dataframe(PB.set_index("페어").style.format({"보유 비율": "{:.0%}", "Sharpe": "{:+.2f}", "연수익 (%)": "{:+.2f}", "연변동성 (%)": "{:.1f}", "최대낙폭 (%)": "{:.1f}"}))
            st.caption("상관 필터(52주 수익률 상관 > 0.5) 버전. 몇 개 페어가 양수여도 12개 중 결과를 보고 고르면 과적합 — 페어 전체 평균(위 표 ①)이 판단 기준.")
        pr = st.selectbox("페어", [f"{a}|{b}" for a, b in LS.PAIRS if a in o["P"] and b in o["P"]], format_func=lambda x: " / ".join(PR.BUY_IDS.get(i, short(i)) for i in x.split("|")), key="ls_pair")
        a, b = pr.split("|")
        W, z = LS.pair_weights(o["P"], a, b, rho_min=0.5)
        f = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.06,
                          subplot_titles=("스프레드 z = (ln A − β ln B − 52주 평균) / 52주 표준편차", "포지션 (+1 = A 롱·B 숏)"))
        f.add_trace(go.Scatter(x=z.index, y=z, line=dict(color=PAL[0], width=1.2), name="z"), 1, 1)
        for yv in (2, -2):
            f.add_hline(y=yv, line_dash="dash", line_color=PAL[1], row=1, col=1)
        pos = np.sign(W[a])
        f.add_trace(go.Scatter(x=pos.index, y=pos, line=dict(color=PAL[2], width=1, shape="hv"), name="포지션"), 2, 1)
        st.plotly_chart(fig_layout(f, 420, hovermode="x"), key="ls_pair_fig")
    with st.expander("② 흡수비율 · ③ 스트레스 국면 확률 — 위험회피 신호가 언제 켜졌나"):
        f = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                          subplot_titles=("흡수비율 AR (52주 상관행렬 1주성분 비중 — 높을수록 품목들이 한 요인에 묶임)", "HMM 스트레스 국면 실시간 확률 (VIX 최고 국면)"))
        ar = o["ar"].dropna()
        f.add_trace(go.Scatter(x=ar.index, y=ar, line=dict(color=PAL[0], width=1.4), name="AR"), 1, 1)
        if o["stress"] is not None:
            ps = o["stress"].dropna()
            f.add_trace(go.Scatter(x=ps.index, y=ps, fill="tozeroy", line=dict(color=PAL[1], width=0.8), name="스트레스 확률"), 2, 1)
        st.plotly_chart(shade(fig_layout(f, 440, hovermode="x"), ar.index), key="ls_sig_fig")
    with st.expander("국면 진입 초기 — 신호가 켜진 직후 1·4·8·13주 동안 무슨 일이 있었나 (이벤트 분석)", expanded=True):
        for nm, E in o["events"].items():
            if len(E):
                st.markdown(f"**{nm}** (직전 8주 꺼져 있다가 켜진 주 = 진입)")
                st.dataframe(E.set_index(["묶음", "h주"]).style.format({"진입 후 평균 (%)": "{:+.2f}", "평소 평균 (%)": "{:+.2f}", "차이 t값": "{:+.2f}", "하락 비율": "{:.0%}"}))
        st.caption("'진입 후 평균'이 '평소 평균'보다 확실히 낮아야(t값 < −2) 국면 진입이 동반 하락을 미리 알려준다고 볼 수 있음. "
                   "HMM 스트레스 확률은 실시간 필터라 급락이 이미 진행된 뒤 켜지는 경우가 많아, 진입 직후엔 오히려 반등이 나오기도 함.")
    st.markdown("##### ⑨ 확장 — 칼만 필터 · 헤지펀드식 상대가치 롱숏 (원자재판)")
    st.caption("주식 헤지펀드의 'A 롱 / B 숏'(예: 구글 롱·엔비디아 숏, 과열 대장주 숏으로 신생 테마주 매수 자금 마련)은 ① 같은 테마 안에서 싼 것·뒤처진 것을 사고 "
               "비싼 것·과열된 것을 팔며 ② 시장 방향(베타)은 상쇄하는 구조 → 원자재 19개에 같은 아이디어를 옮김. 모두 4주 재조정 · 상위/하위 1/3 · 목표 변동성 10% · 편도 5bp, 평가 2007–.")
    with st.expander("전략 목록 — 무엇을 사고 무엇을 파나", expanded=False):
        st.markdown("""
| 코드 | 전략 | 롱 | 숏 | 주식 헤지펀드 유사 사례 · 근거 |
|---|---|---|---|---|
| K1 | 칼만 추세 | 칼만 필터 기울기 > 0 | 기울기 < 0 | 52주 모멘텀의 적응형 판 (상태공간 국소 선형 추세) |
| K2 | 칼만 동적 헤지 페어 | 관계 대비 싼 쪽 | 비싼 쪽 (β가 매주 변함) | 통계적 차익 페어 (Chan 2013) |
| S1 | 섹터 중립 모멘텀 | 섹터 안 26주 승자 | 섹터 안 패자 | '같은 업종 1등 롱 · 꼴찌 숏' |
| S2 | 섹터 내 따라잡기 | 섹터 안 26주 낙오 | 섹터 안 과열 | '과열 대장주(엔비디아) 숏 → 후발주 매수' |
| V1 | 가치 | 5년 평균 대비 가장 싼 | 가장 비싼 | 가치 (Asness-Moskowitz-Pedersen 2013) |
| W1 | 왜도 | 급락 꼬리(음의 왜도) | 급등 꼬리 | 복권형 자산 회피 (Fernandez-Perez 2018) |
| L1 | 저변동성 | 변동성 낮은 | 높은 | 저변동성 이상현상 |
| B1 | 저베타 (BAB) | 바스켓 베타 낮은 | 높은 (베타 중립) | Betting Against Beta (Frazzini-Pedersen 2014) |
| C1 | 헤징 압력 (COT) | 상업 헤저 매도 많은 | 적은 | 헤지 수요 보상 (Basu-Miffre 2013) |
| C2 | 쏠림 역발상 (COT) | 투기 펀드 비중 최저 | 최고 (쏠림) | '모두가 산 승자 숏' |
""")
    LX_ = _lx_run(float(cost), ROLL)
    TX = LX_["table"]
    spx = [c for c in TX.columns if c.startswith("Sharpe ~") or c.startswith("Sharpe 2")]
    st.dataframe(TX.style.format({"Sharpe": "{:+.2f}", "연수익 (%)": "{:+.2f}", "연변동성 (%)": "{:.1f}", "최대낙폭 (%)": "{:.1f}", spx[0]: "{:+.2f}", spx[1]: "{:+.2f}",
                                  "t값": "{:+.2f}", "기존 70/30과 상관": "{:+.2f}", "52주 모멘텀과 상관": "{:+.2f}", "기존 70/30 대비 차이 t": "{:+.2f}"}, na_rep="–"),
                 height=min(36 * (len(TX) + 1) + 4, 470))
    st.caption("※ 같은 '혼합 · MinVar 50 + 모멘텀 25 + 가치 25'가 ⑤ Portfolio Customization에서는 Sharpe 0.89–0.90으로 조금 다르게 나옴 — 계산 주기·재료는 같고, "
               "섞은 뒤 목표 변동성 10%로 맞추는 배율(직전 26주 실현 변동성)의 출발점만 다름. 여기는 2001년부터 이어진 배율을 쓰고, Customization은 고정 재료가 2007년부터라 "
               "2007년 초 약 반년(처음 13주 배율 1, 26주까지 짧은 창)을 새로 계산 → 2007년 중반 이후는 완전히 같음(상관 1.0, 처음 26주를 빼면 둘 다 0.896).")
    st.caption("목표 변동성을 10% 대신 5–50%로 바꾼 결과는 ⑤ 포트폴리오 › '목표변동성에 따른 수익율 변화 Backtest' 서브탭.")
    DX = LX_["returns"]
    selx = st.multiselect("누적 수익 비교 (⑨)", list(DX.columns), [c for c in DX.columns if c.startswith(("혼합 · MinVar 70 + 모멘텀 30", "혼합 · MinVar 70 + 칼만",
                                                                                                     "V1", "S2", "K1 칼만 추세 (느림", "혼합 · MinVar 50 + 모멘텀 25 + 가치"))], key="ls_x_sel")
    if selx:
        f = go.Figure([go.Scatter(x=DX.index, y=DX[c].fillna(0).cumsum(), name=c, line=dict(width=2.2 if c.startswith("혼합 · MinVar 50 + 모멘텀 25 + 가치") else 1.3,
                                                                                         color=PAL[j % 8])) for j, c in enumerate(selx)])
        st.plotly_chart(shade(fig_layout(f, 380, title="누적 수익 (%, 모두 목표 변동성 10%)", hovermode="x"), DX.index), key="ls_x_cum")
    with st.expander("사전 선택 검증 — 2016년까지 자료로만 혼합을 고르고 2017년 이후로 확인", expanded=True):
        SL = LX_["selection"]
        st.dataframe(SL.style.format("{:+.2f}"), height=min(36 * (len(SL) + 1) + 4, 470))
        st.caption(f"후보 = 기존 70/30 + 'MinVar 50 + 모멘텀 25 + 대체 전략 하나 25' 10개 + MinVar 품목 상한 15%판 3개(생우·금 쏠림 완화). 학습 Sharpe 1위({SL.index[0]})를 고르면 시험 기간 {SL.iloc[0, 1]:+.2f} "
                   f"(기존 70/30 시험 {SL.loc['혼합 · MinVar 70 + 모멘텀 30 (기존)'].iloc[1]:+.2f}). 표의 숫자는 '혼합' 전체의 Sharpe — 대체 전략 단독 성과(위 큰 표)와 다름. 시험 기간은 고를 때 보지 않음 — 다만 전략 정의(5년 창·1/3·4주)와 후보 목록은 미리 정한 관례이지 이 자료로 튜닝한 것은 아님.")
    with st.expander("구간 이동 선택 검증 — 2013–16 · 2017–20 · 2021–26 시험 구간마다 그 전 자료로만 혼합 선택", expanded=True):
        WK = LX.walk_select(LX_["cands"], "혼합 · MinVar 70 + 모멘텀 30 (기존)")
        st.dataframe(WK["table"].set_index("시험 구간").style.format({"선택 조합 학습 Sharpe": "{:+.2f}", "선택 조합 시험 Sharpe": "{:+.2f}", "기존 70/30 시험 Sharpe": "{:+.2f}",
                                                                   "기존 대비 연 수익 차 (%p)": "{:+.2f}", "차이 t값": "{:+.2f}"}))
        sm_ = WK["summary"]
        w1, w2, w3 = st.columns(3)
        w1.metric(f"이어붙인 표본 외 Sharpe ({sm_['기간']})", f"{sm_['이어붙인 표본 외 Sharpe (선택 조합)']:+.2f}", f"기존 70/30 {sm_['같은 기간 기존 70/30']:+.2f}", delta_color="off")
        w2.metric("기존 대비 연 수익 차", f"{sm_['연 수익 차 (%p)']:+.2f}%p", f"차이 t {sm_['차이 t값']:+.2f}", delta_color="off")
        w3.metric("세 구간 모두 기존보다 나았나", "예" if (WK["table"]["기존 대비 연 수익 차 (%p)"] > 0).all() else "아니오",
                  f"{int((WK['table']['기존 대비 연 수익 차 (%p)'] > 0).sum())} / {len(WK['table'])} 구간", delta_color="off")
        f = go.Figure([go.Scatter(x=WK["oos"].index, y=WK["oos"].cumsum(), name="매 구간 학습으로 고른 조합 (표본 외)", line=dict(color=PAL[2], width=2.2)),
                       go.Scatter(x=WK["base"].index, y=WK["base"].cumsum(), name="기존 70/30", line=dict(color="#898781", width=1.5))])
        for a_, _b in LX.FOLDS[1:]:
            f.add_vline(x=pd.Timestamp(a_), line_dash="dot", line_color="#898781")
        st.plotly_chart(fig_layout(f, 300, title="이어붙인 표본 외 누적 수익 (%) — 점선 = 다시 고른 시점", hovermode="x"), key="ls_walk")
        st.caption("매 시험 구간 시작 시점에 그 전 자료(2007–)만으로 11개 후보의 Sharpe 1위를 골라 다음 구간에 적용 → 고른 결과만 이어 붙인 것이 진짜 표본 외 성과. "
                   "'학습 순위 → 시험 순위' = 학습 1위가 시험 기간 11개 중 몇 위였나.")
    with st.expander("강세장 덕 vs 전략 덕 — 바스켓 Buy&Hold 대비 베타 · 알파", expanded=True):
        key_s = {"기존 MinVar 70 + 모멘텀 30": "혼합 · MinVar 70 + 모멘텀 30 (기존)", "가치 혼합 (50/25/25)": "혼합 · MinVar 50 + 모멘텀 25 + V1 가치 25",
                 "따라잡기 혼합 (50/25/25)": "혼합 · MinVar 50 + 모멘텀 25 + S2 섹터 내 따라잡기 25",
                 "가치 혼합 · MinVar 상한 15%": "혼합 · MinVar(상한 15%) 50 + 모멘텀 25 + V1 가치 25"}
        AT = LX.alpha_table({k: LX_["cands"][v] for k, v in key_s.items() if v in LX_["cands"]}, LX_["bh"])
        st.dataframe(AT.set_index(["전략", "기간"]).style.format({c: "{:+.2f}" for c in AT.columns if c not in ("전략", "기간")}), height=min(36 * (len(AT) + 1) + 4, 470))
        pick_ = st.selectbox("기간별 분해 그림", list(key_s), 1, key="ls_ab_pick")
        q_ = AT[AT["전략"] == pick_]
        f = go.Figure([go.Bar(x=q_["기간"], y=q_["시장 몫 = β×바스켓 (%)"], name="시장 몫 (β × 바스켓 수익)", marker_color="#898781"),
                       go.Bar(x=q_["기간"], y=q_["알파 = 전략 몫 (연 %)"], name="전략 몫 (알파)", marker_color=PAL[2])])
        st.plotly_chart(fig_layout(f, 300, barmode="relative", title=f"{pick_} — 연수익 = 시장 몫 + 전략 몫 (%)", hovermode="x"), key="ls_ab_fig")
        rw_lbl = st.radio("롤링 창", ["1년", "2년", "3년"], 1, horizontal=True, key="ls_ra_win", help="짧을수록 최근 변화에 민감(흔들림 큼), 길수록 장기 추세")
        rw = {"1년": 52, "2년": 104, "3년": 156}[rw_lbl]
        RA = LX.rolling_alpha(LX_["cands"][key_s[pick_]], LX_["bh"], rw).dropna()
        RB_ = LX.rolling_alpha(LX_["cands"][key_s["기존 MinVar 70 + 모멘텀 30"]], LX_["bh"], rw).dropna()
        f = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.07,
                          subplot_titles=(f"{rw_lbl} 롤링 알파 (연 %, 시장 몫을 뺀 전략 몫) · 주황 점 = 그 {rw_lbl} 알파가 유의(|t| ≥ 2)", f"{rw_lbl} 롤링 베타 (원자재 시장을 탄 정도)"))
        f.add_trace(go.Scatter(x=RA.index, y=RA["알파 (연 %)"], name=pick_, line=dict(color=PAL[2], width=2)), 1, 1)
        if pick_ != "기존 MinVar 70 + 모멘텀 30":
            f.add_trace(go.Scatter(x=RB_.index, y=RB_["알파 (연 %)"], name="기존 70/30", line=dict(color="#898781", width=1.3)), 1, 1)
        sig_ = RA["알파 t값"].abs() >= 2
        f.add_trace(go.Scatter(x=RA.index[sig_], y=RA["알파 (연 %)"][sig_], mode="markers", marker=dict(size=4, color=PAL[1]), name="|t| ≥ 2 (유의)"), 1, 1)
        f.add_hline(y=0, line_color="#898781", row=1, col=1)
        f.add_trace(go.Scatter(x=RA.index, y=RA["베타"], name="베타", line=dict(color=PAL[0], width=1.5), showlegend=False), 2, 1)
        st.plotly_chart(shade(fig_layout(f, 460, hovermode="x"), RA.index), key="ls_ra_fig")
        last_a, last_b = RA["알파 (연 %)"].iloc[-1], RA["베타"].iloc[-1]
        rec = RA["알파 (연 %)"].loc[RA.index >= RA.index[-1] - pd.DateOffset(years=2)].mean()
        prev = RA["알파 (연 %)"].loc[(RA.index < RA.index[-1] - pd.DateOffset(years=2)) & (RA.index >= RA.index[-1] - pd.DateOffset(years=6))].mean()
        st.caption(f"지금 {rw_lbl} 알파 연 {last_a:+.1f}% · 베타 {last_b:.2f} · 최근 2년 평균 알파 {rec:+.1f}% vs 그 전 4년 평균 {prev:+.1f}% → "
                   + ("우위가 커지는 추세." if rec > prev + 1 else "우위가 줄어드는 추세." if rec < prev - 1 else "비슷한 수준 유지.")
                   + f" 롤링 창이 {rw_lbl}이라 큰 사건(예: 2008·2020·2022) 전후로 흔들림 — 창이 짧을수록 더 크게. 추세는 여러 해에 걸친 방향으로 판단.")
        st.caption("바스켓 = 원자재 19개 동일가중 Buy&Hold. 주간 수익률을 바스켓에 회귀(r = α + β·바스켓): β = 원자재 시장을 얼마나 탔나, 시장 몫 = β × 바스켓 연수익(강세장 덕), "
                   "알파 = 시장 움직임을 빼고 남은 연수익(전략 덕 — 품목 고르기·비중·롱숏). 2021–2026 가치 혼합 연 18.3% 중 시장 몫 6.6% · 알파 11.7%(t 3.7) → "
                   "강세장만이 아니라 생우·금 같은 '바스켓보다 더 오른 품목'에 실려 있던 품목 선택 효과가 큼.")
    with st.expander("기간별 품목 기여도 — 어떤 품목 덕분에 성과가 났나"):
        PnL = _lx_contrib(ROLL)
        k1_, k2_ = st.columns(2)
        st_nm = k1_.selectbox("전략", list(PnL), 4 if len(PnL) > 4 else 0, key="ls_ct_strat")
        per_ = k2_.selectbox("기간", list(LX.PERIODS), 3, key="ls_ct_per")
        a_, b_ = LX.PERIODS[per_]
        X_ = PnL[st_nm].loc[a_:b_]
        yrs = max(len(X_) / 52, 1e-9)
        ct = (X_.sum() / yrs).rename(lambda i: PR.BUY_IDS.get(i, short(i))).sort_values()
        f = go.Figure(go.Bar(x=ct.values, y=ct.index, orientation="h", marker_color=[PAL[2] if v > 0 else PAL[1] for v in ct.values],
                             text=[f"{v:+.2f}" for v in ct.values], textposition="outside"))
        st.plotly_chart(fig_layout(f, 120 + 24 * len(ct), title=f"{st_nm} · {per_} — 품목별 연평균 기여 (%p, 합계 {ct.sum():+.1f}%)", hovermode="y"), key="ls_ct_fig")
        st.caption("기여 = 그 품목 비중 × 그 품목 수익률을 기간 동안 더해 연평균으로 (거래비용 제외, 목표 변동성 배율 반영). 혼합은 구성 전략의 품목 손익을 비율대로 더한 뒤 혼합 배율까지 반영.")
        Y_ = PnL[st_nm].loc["2007-01-01":].groupby(PnL[st_nm].loc["2007-01-01":].index.year).sum().T
        Y_.index = [PR.BUY_IDS.get(i, short(i)) for i in Y_.index]
        Y_ = Y_.loc[Y_.abs().sum(1).sort_values(ascending=False).index]
        inv_ = {PR.BUY_IDS.get(i, short(i)): LX.SECTOR.get(i, "기타") for i in PnL[st_nm].columns}
        SEC = Y_.groupby(lambda nm: inv_.get(nm, "기타")).sum()
        SEC = SEC.loc[SEC.sum(1).sort_values(ascending=False).index]
        SEC.index = [f"▣ 섹터 · {i}" for i in SEC.index]
        Y_ = pd.concat([SEC, Y_])
        vmx = float(np.nanquantile(np.abs(Y_.values), 0.97)) or 1.0
        fy = go.Figure(go.Heatmap(z=Y_.values, x=[str(c) for c in Y_.columns], y=list(Y_.index), zmin=-vmx, zmax=vmx, colorscale=DIV,
                                  text=[[f"{v:+.1f}" for v in row] for row in Y_.values], texttemplate="%{text}", textfont=dict(size=9),
                                  hovertemplate="%{y} · %{x}: %{z:+.2f}%p<extra></extra>", colorbar=dict(title="%p", thickness=12)))
        fy.add_hline(y=len(SEC) - 0.5, line_color="#898781", line_width=2)
        fy.update_layout(title=f"{st_nm} — 섹터·품목 × 연도 기여 (%p, 그해 합계) · 위 ▣ = 섹터 합계, 아래 = 품목(기여 큰 순)", height=140 + 26 * len(Y_),
                         margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(autorange="reversed"), xaxis=dict(dtick=1, tickangle=-45, tickfont=dict(size=10)))
        st.plotly_chart(fy, key="ls_ct_heat")
        tot_ = Y_.sum()
        st.caption("빨강 = 그해 이익을 낸 품목, 보라·파랑 = 손실. 연도별 합계: " + " · ".join(f"{y} {v:+.1f}" for y, v in tot_.items()))
    st.caption("읽는 법: 칼만 추세(K1)는 52주 모멘텀보다 약하고(0.22 vs 0.50, 2017 이후 음수) MinVar 혼합에 넣어도 기존 70/30보다 나빠짐(0.59·0.64 vs 0.69). "
               "칼만 페어(K2)는 기존 페어(−0.15)보다 낫지만 0.14. 섹터 내 승자 롱(S1)은 크게 손해, 반대로 과열 숏·낙오 롱(S2)과 가치(V1)는 두 기간 모두 양수(0.41·0.42)이고 "
               "모멘텀과 음의 상관 → 기존 혼합에 가치 25%를 섞으면 0.89(차이 t 2.0)·낙폭 −28%. 단 가치·따라잡기 혼합은 결과를 보고 고른 '사후 선택'이라 과대평가 가능 — "
               "가치와 모멘텀의 음의 상관·결합 효과는 학계에서 미리 알려진 결과(Asness 외 2013)라는 점이 근거. 미리 고르지 않고 대체 전략을 전부 섞으면 0.18(다전략 줄).")
    st.markdown("##### ⑧ 비중 결정 방식 비교 — 동일가중 · 역변동성 · 최소분산(MinVar) · 평균-분산(μ = 52주 모멘텀)")
    st.caption("같은 19개 품목 · 공분산 = 과거 104주 Ledoit-Wolf 수축 · 롱온리는 4주마다, 모멘텀 롱숏은 매주 재조정 · 품목 상한 25% · "
               "모두 사전 목표 변동성 10% (직전 26주 실현 변동성으로 배율, 최대 4배)로 맞춰 같은 위험에서 비교. "
               "μ = IC 0.05 × 품목 변동성 × 52주 추세 부호 (Grinold 공식), 평균-분산 위험회피 λ = 5.")
    AL = _alloc_run(float(cost), ROLL)
    k1, k2 = st.columns([1.25, 1])
    spa = [c for c in AL["table"].columns if c.startswith("Sharpe ~") or c.startswith("Sharpe 2")]
    k1.dataframe(AL["table"].style.format({"Sharpe": "{:+.2f}", "연수익 (%)": "{:+.2f}", "연변동성 (%)": "{:.1f}", "최대낙폭 (%)": "{:.1f}",
                                           spa[0]: "{:+.2f}", spa[1]: "{:+.2f}", "t값": "{:+.2f}"}, na_rep="–"))
    f = go.Figure()
    for j, c in enumerate(AL["returns"].columns):
        f.add_trace(go.Scatter(x=AL["returns"].index, y=AL["returns"][c].cumsum(), name=c, line=dict(color=PAL[j % 8], width=2 if "MinVar" in c else 1.2)))
    k2.plotly_chart(fig_layout(f, 330, title="누적 수익 (%, 목표 변동성 10%)", hovermode="x", legend=dict(orientation="h", y=-0.35, font=dict(size=10))), key="ls_alloc")
    st.caption("최소분산은 변동성이 낮고 다른 품목과 덜 같이 움직이는 생우·금·돈육에 평균 19%·17%·12%를 두게 됨 → 이 기간 그 품목들이 강해 성과가 좋았지만 "
               "소수 품목 집중의 결과일 수 있음. 평균-분산에 모멘텀 μ를 넣으면 추세 품목으로 몰려 오히려 나빠짐(낙폭 확대). 모멘텀은 롱숏 · 역변동성이 가장 안정적.")
    with st.expander("크기 조정(변동성 사이징) 방식 · 창 길이 비교 — 과거 실현 손익 vs 현시점 Ledoit-Wolf 공분산"):
        SZ, WN = _sizing_run(float(cost), ROLL)
        st.markdown("**① 같은 상대 비중에 크기만 다르게** (목표 10%, 최대 4배)")
        spz = [c for c in SZ.columns if c.startswith("Sharpe ~") or c.startswith("Sharpe 2")]
        st.dataframe(SZ.style.format({"Sharpe": "{:+.2f}", "연수익 (%)": "{:+.2f}", "연변동성 (%)": "{:.1f}", "최대낙폭 (%)": "{:.1f}", spz[0]: "{:+.2f}",
                                      spz[1]: "{:+.2f}", "목표 이탈 (평균 |변동성−10|)": "{:.2f}"}, na_rep="–"), height=min(36 * (len(SZ) + 1) + 4, 470))
        st.caption("과거 실현 = 그 전략이 실제로 낸 직전 N주 손익의 표준편차, 현시점 LW = 오늘 비중 × 직전 N주 Ledoit-Wolf 공분산 → √(w′Σ̂w). "
                   "두 전략 모두 '과거 실현 26주'(현재 방식)가 Sharpe·낙폭에서 가장 낫거나 비슷함. 현시점 LW 공분산은 긴 창일수록 급변에 늦게 반응해 목표 이탈·낙폭이 커짐. "
                   "'목표 이탈' = 26주 실현 변동성이 10%에서 평균 몇 %p 벗어났나.")
        st.markdown("**② 창 길이 민감도** (크기 = 과거 실현 26주)")
        spw = [c for c in WN.columns if c.startswith("Sharpe ~") or c.startswith("Sharpe 2")]
        st.dataframe(WN.style.format({"Sharpe": "{:+.2f}", "연수익 (%)": "{:+.2f}", "연변동성 (%)": "{:.1f}", "최대낙폭 (%)": "{:.1f}", spw[0]: "{:+.2f}",
                                      spw[1]: "{:+.2f}"}, na_rep="–"))
        st.caption("모멘텀의 품목 변동성 창은 26주 ≈ 52주(0.49 vs 0.52), MinVar 공분산 창은 길수록 좋음(52주 0.55 → 104주 0.73 → 156주 0.82). "
                   "단, 결과를 보고 창을 고르면 과적합 — 기본값(26·104주)은 실험 전에 정한 관례(변동성은 반응성, 공분산은 19개 품목 × 190개 모수를 안정적으로 추정할 표본)라 그대로 둠.")
    with st.expander("MinVar 강건성 점검 — 특정 품목 덕인가? 집중을 줄이면? 모멘텀과 섞으면?", expanded=True):
        RB, DB = _alloc_robust(float(cost), ROLL)
        spr = [c for c in RB.columns if c.startswith("Sharpe ~") or c.startswith("Sharpe 2")]
        st.dataframe(RB.style.format({"Sharpe": "{:+.2f}", "연수익 (%)": "{:+.2f}", "연변동성 (%)": "{:.1f}", "최대낙폭 (%)": "{:.1f}", spr[0]: "{:+.2f}",
                                      spr[1]: "{:+.2f}", "t값": "{:+.2f}", "MinVar와 상관": "{:+.2f}"}, na_rep="–"), height=min(36 * (len(RB) + 1) + 4, 470))
        f = go.Figure()
        for j, c in enumerate([c for c in DB.columns if c.startswith(("MinVar (상한 25%", "MinVar — 생우·금·돈육", "MinVar (상한 15%", "모멘텀", "혼합"))]):
            f.add_trace(go.Scatter(x=DB.index, y=DB[c].cumsum(), name=c, line=dict(color=PAL[j % 8], width=2.2 if c.startswith("혼합") else 1.3)))
        st.plotly_chart(fig_layout(f, 320, title="누적 수익 (%, 모두 목표 변동성 10%)", hovermode="x"), key="ls_alloc_rb")
        st.caption("생우·금·돈육을 모두 빼면 MinVar Sharpe가 절반으로 떨어짐 → 성과의 상당 부분이 그 세 품목(이 기간 저변동·상승) 덕. "
                   "상한을 낮춰 집중을 줄이면 Sharpe가 단계적으로 내려감. MinVar와 모멘텀은 상관이 낮아(약 0.2) 섞으면 하위기간 간 차이가 줄어듦 — "
                   "혼합 비율 = 두 전략(각 변동성 10%)에 주는 비중, 섞은 뒤 다시 10%로 배율.")
    _sig_bt_section()


SIG_EXP = [("A 연도별 최적 비중 갈아타기", "strategy_switch_compare.csv", "규칙", "Sharpe 학습 2012–16", "Sharpe 시험 2017–", "시험 차이 t"),
           ("B 다음 분기 1등 축 미리 읽기", "strategy_meta_compare.csv", "규칙", "Sharpe 학습 2008–16", "Sharpe 시험 2017–", "시험 차이 t"),
           ("C 칼만 · HMM · 뉴스 · DCC 비중 조정", "strategy_adaptive_compare.csv", "규칙", "Sharpe 학습 2008–16", "Sharpe 시험 2017–", "시험 차이 t"),
           ("D 위험할 때 현금 피신", "strategy_cash_compare.csv", "신호", "Sharpe 학습 2008–16", "Sharpe 시험 2017–", "시험 차이 t"),
           ("E 목표 변동성 동적 + 국채 이자 (초과 Sharpe)", "strategy_voltgt_dyn_compare.csv", "규칙", "초과 Sharpe 학습 2008–16", "초과 Sharpe 시험 2017–", None),
           ("F 상관↑ (+ 같이 오름) 추세 집중", "strategy_crisis_mom_compare.csv", "규칙", "Sharpe 학습 2008–16", "Sharpe 시험 2017–", "시험 차이 t"),
           ("J 짧은 추세 4 · 8주 (단독 성과)", "strategy_short_trend_compare.csv", "판", "Sharpe 학습 2008–16", "Sharpe 시험 2017–", "시험 차이 t"),
           ("K 모멘텀 기간 4–52주 (혼합)", "strategy_mom_lookback_compare.csv", "모멘텀 기간", "혼합 학습 2008–16", "혼합 시험 2017–", "혼합 시험 차이 t"),
           ("G · I 품목군 상한", "strategy_groupcap_compare.csv", "상한", "Sharpe 학습 2008–16", "Sharpe 시험 2017–", "시험 차이 t")]


@st.cache_data(show_spinner=False)
def _sig_rows() -> pd.DataFrame:
    rows = []
    for exp, f, nc, lc, tc, dc in SIG_EXP:
        fp = A.DATA / f
        if not fp.exists():
            continue
        T = pd.read_csv(fp, encoding="utf-8-sig")
        for _, r in T.iterrows():
            nm = str(r[nc]).replace(" ← 학습 최고", "") + "".join(f" · {r[c]}" for c in ("설정", "문턱", "현금 시 노출", "λ·q", "이자 관점") if c in T and pd.notna(r.get(c)) and str(r[c]).strip() not in ("", "nan") and not str(r[c]).startswith("←"))
            best = any(str(r.get(c, "")).startswith("←") for c in ("학습 최고", "설정")) or "← 학습 최고" in str(r[nc])
            rows.append({"실험": exp, "규칙": nm, "학습 최고": "◆" if best else "", "학습 Sharpe": r.get(lc), "시험 Sharpe": r.get(tc),
                         "시험 차이 t": r.get(dc) if dc else np.nan, "최대 낙폭 (%)": r.get("최대 낙폭 (%)", r.get("혼합 최대 낙폭 (%)", np.nan)),
                         "기준": ("(기준" in str(r[nc])) or str(r[nc]).strip() in ("고정 50/25/25", "고정 50/25/25 (사후 선택)", "고정 목표 10% (기준)")})
    return pd.DataFrame(rows)


def _sig_bt_section():
    """⑤ 롱숏 탭 › ⑪ 3축 혼합(50/25/25) 추가 백테스트 — strategy_*.py 결과 표(규칙마다 학습 · 시험 Sharpe)와 3D 보기. ⑥ '3축 신호 · 위험 한도 실험' 요약과 같은 실험."""
    st.markdown("##### ⑪ 3축 혼합(MinVar 50 · 모멘텀 25 · 가치 25) 추가 백테스트 — 신호로 비중을 바꾸면 나아지나")
    D = _sig_rows()
    if D.empty:
        st.info("결과 파일이 없습니다 — strategy_*.py를 먼저 실행하세요."); return
    base = 1.067
    st.caption(f"실험 {D['실험'].nunique()}묶음 · 규칙 {int((~D['기준']).sum())}개. 모두 학습 2008–16에서만 규칙을 고르고 시험 2017–26에서 확인 · 편도 5bp + 비중 변경 비용 · 목표 변동성 10%. "
               f"기준 = 고정 50/25/25 시험 Sharpe {base:.2f}. ◆ = 그 신호(또는 실험)에서 학습 성적이 가장 좋은 규칙 = 미리 고른다면 실제로 쓰게 될 규칙. "
               "시험 차이 t = 기준 대비 주간 수익 차이의 t값(2 이상이어야 우연과 구별) — 모든 실험에서 최대 1.5 미만. 학습 1위 = 실험이 표시한 학습 최고(◆, 여러 개면 그중 최고), 없으면 학습 Sharpe 1위.")
    S = D[~D["기준"]].groupby("실험", sort=False).agg(규칙수=("규칙", "size"), 시험_최소=("시험 Sharpe", "min"), 시험_최대=("시험 Sharpe", "max"),
                                                    학습최고_시험=("시험 Sharpe", lambda x: np.nan), 최대_차이t=("시험 차이 t", "max"))
    for e in S.index:                                                     # 그 실험에서 학습 Sharpe 1위 규칙의 시험 Sharpe
        d_ = D[(D["실험"] == e) & ~D["기준"]].dropna(subset=["학습 Sharpe"])
        d_ = d_[d_["학습 최고"] == "◆"] if (d_["학습 최고"] == "◆").any() else d_       # 실험이 정한 학습 최고(◆)가 있으면 그중, 없으면 학습 Sharpe 1위
        if len(d_):
            S.loc[e, "학습최고_시험"] = d_.loc[d_["학습 Sharpe"].idxmax(), "시험 Sharpe"]
    S.columns = ["규칙 수", "시험 Sharpe 최소", "시험 Sharpe 최대", "학습 1위 규칙의 시험 Sharpe", "시험 차이 t 최대"]
    S["학습 1위 규칙의 시험 Sharpe"] = S["학습 1위 규칙의 시험 Sharpe"].astype(float)
    st.dataframe(S.style.format({"규칙 수": "{:.0f}", "시험 Sharpe 최소": "{:.2f}", "시험 Sharpe 최대": "{:.2f}", "학습 1위 규칙의 시험 Sharpe": "{:.2f}",
                                 "시험 차이 t 최대": "{:+.2f}"}, na_rep="–"), use_container_width=True)
    t3, t2 = st.tabs(["3D 보기 (학습 × 실험 × 시험)", "규칙 전체 표"])
    with t3:
        ex = list(dict.fromkeys(D["실험"])); d_ = D[~D["기준"]].dropna(subset=["학습 Sharpe", "시험 Sharpe"])
        f = go.Figure([go.Surface(x=[d_["학습 Sharpe"].min() - 0.1, d_["학습 Sharpe"].max() + 0.1], y=[-0.5, len(ex) - 0.5], z=[[base, base], [base, base]],
                                  colorscale=[[0, PAL[1]], [1, PAL[1]]], opacity=0.2, showscale=False, hoverinfo="skip", name="기준 1.07")])
        for b_, nm, sz in [(False, "규칙 후보", 2.2), (True, "◆ 학습 최고", 5.0)]:
            q = d_[(d_["학습 최고"] == "◆") == b_]
            f.add_trace(go.Scatter3d(x=q["학습 Sharpe"], y=[ex.index(e) for e in q["실험"]], z=q["시험 Sharpe"], mode="markers", name=nm,
                                     text=q["실험"] + "<br>" + q["규칙"].str[:70], marker=dict(size=sz, symbol="diamond" if b_ else "circle",
                                     color=np.where(q["시험 Sharpe"] > base + 0.02, PAL[2], "#9a9a92"), opacity=1.0 if b_ else 0.55),
                                     hovertemplate="%{text}<br>학습 %{x:.2f} → 시험 %{z:.2f}<extra></extra>"))
        f.update_layout(height=520, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", y=1.02),
                        scene=dict(xaxis_title="학습 Sharpe", yaxis=dict(title="", tickvals=list(range(len(ex))), ticktext=[e[:14] for e in ex]), zaxis_title="시험 Sharpe",
                                   aspectmode="manual", aspectratio=dict(x=1.2, y=1.7, z=0.9), camera=dict(eye=dict(x=1.6, y=-1.4, z=0.7))))
        st.plotly_chart(f, key="ls_sig3d")
        st.caption("빨간 평면 = 기준 1.07. 평면 위로 올라온 점(초록)이 있어도, 학습 성적만 보고 고를 수 있는 ◆는 대부분 평면 근처나 아래 — 평면 위 점을 시험 결과를 보고 고르면 사후 선택.")
    with t2:
        ex_pick = st.multiselect("실험 고르기", list(dict.fromkeys(D["실험"])), default=[], key="ls_sig_ex", placeholder="비우면 전부")
        V = D[D["실험"].isin(ex_pick)] if ex_pick else D
        st.dataframe(V.drop(columns="기준").style.format({"학습 Sharpe": "{:.2f}", "시험 Sharpe": "{:.2f}", "시험 차이 t": "{:+.2f}", "최대 낙폭 (%)": "{:.1f}"}, na_rep="–"),
                     use_container_width=True, height=420, hide_index=True)
    st.caption("판정: 신호 기반 비중 변경 A–F · J · K는 모두 미채택(고정 50/25/25 유지), 곡물 합계 25% 상한(G · I)만 채택 — ⑤ Portfolio Customization '품목군 위험 한도'. "
               "요약 그림 · 사례(2022 곡물, 2025–26 금 · 은)는 ⑥ 실험 결과 → '3축 신호 · 위험 한도 실험', 상세 기록은 docs/백테스트_로그.md.")


SPAN_BASE = ["기준 · MinVar 롱 (⑥)", "기준 · 52주 모멘텀 (④)", "V1 가치 (5년 평균 대비 싼 품목 롱)"]


def _spanning_section(grp: str, kp: str, dflt: list[str], why: str):
    """대안 시계열 · 멀티모달 서브탭: '기존 3종(MinVar·모멘텀·가치)만' vs '기존 3종 + 이 서브탭 전략' 효율적 투자선(평균-분산 스패닝) +
    학습 구간(~2016) 최대 Sharpe 비중을 2017년 이후에 그대로 적용한 실제 점(사후 선택 점검) + Huberman-Kandel 알파 회귀."""
    import statsmodels.api as sm
    st.markdown("---")
    st.markdown(f"#### 효율적 투자선 — 기존 3종에 {grp} 전략을 더하면 투자선이 바깥으로 밀려나나 (스패닝)")
    with st.spinner("전략 재료 읽는 중…"):
        X0, G = _sleeves(ROLL)
    base = [b for b in SPAN_BASE if b in X0]
    cand = [c for c in X0 if G.get(c) == grp]
    if len(base) < 2 or not cand:
        st.info("재료가 없습니다.")
        return
    add = st.multiselect(f"더할 {grp} 전략 (기본 = {why})", cand, [d for d in dflt if d in cand][:6] or cand[:2],
                         format_func=lambda x: x[:60], key=f"{kp}_sp_add")
    if not add:
        st.info("더할 전략을 1개 이상 고르세요.")
        return
    X = X0[base + add].loc["2007-01-01":].dropna()
    tr, te = X.loc[:"2016-12-31"], X.loc["2017-01-01":]
    if len(tr) < 104 or len(te) < 104:
        st.warning(f"함께 있는 기간이 짧습니다 (학습 {len(tr)}주 · 시험 {len(te)}주 — 각 104주 이상 필요).")
        return
    per = st.radio("투자선 기간", [f"전체 {X.index[0]:%Y-%m}~ (사후)", "시험 2017~ (사후) + 학습 비중의 실제 점"], index=1, horizontal=True, key=f"{kp}_sp_per")
    Xp = X if per.startswith("전체") else te
    Fb, Fa = _frontier_calc(Xp[base]), _frontier_calc(Xp)
    Tb, Ta = _frontier_calc(tr[base]), _frontier_calc(tr)
    nb, na_ = "기존 3종", f"기존 3종 + {grp}"
    oos = {}
    for nm_, T_, cols_ in [(nb, Tb, base), (na_, Ta, base + add)]:
        r_ = te[cols_] @ T_["w_tan"]
        oos[nm_] = (float(r_.std() * np.sqrt(52)), float(r_.mean() * 52), float(r_.mean() / r_.std() * np.sqrt(52)))
    ink = "#e8e6e1" if DARK else "#0b0b0b"
    gry = "#898781"
    sb, sa = Fb["ret"] / Fb["vol"], Fa["ret"] / Fa["vol"]
    t3, t2 = st.tabs(["3D 보기 (위험 × 수익 × Sharpe)", "2D 보기"])
    with t3:
        f3 = go.Figure()
        f3.add_trace(go.Scatter3d(x=Fa["vol"], y=Fa["ret"], z=sa, mode="markers", name=f"{na_} (무작위 비중)",
                                  marker=dict(size=2, color=sa, colorscale=SEQ, opacity=0.55, colorbar=dict(title="Sharpe", thickness=12, len=0.6))))
        f3.add_trace(go.Scatter3d(x=Fb["vol"], y=Fb["ret"], z=sb, mode="markers", name=f"{nb}만 (무작위 비중)", marker=dict(size=2, color=gry, opacity=0.6)))
        for F_, nm_, col_ in [(Fb, f"투자선 · {nb}", gry), (Fa, f"투자선 · + {grp}", PAL[1])]:
            fr = F_["front"]
            if len(fr):
                f3.add_trace(go.Scatter3d(x=fr[:, 0], y=fr[:, 1], z=fr[:, 1] / fr[:, 0], mode="lines", name=nm_, line=dict(color=col_, width=7)))
        if not per.startswith("전체"):
            for (nm_, (v_, r_, s_)), col_ in zip(oos.items(), [gry, PAL[1]]):
                f3.add_trace(go.Scatter3d(x=[v_], y=[r_], z=[s_], mode="markers", name=f"◇ {nm_}: 학습 비중의 시험 실제",
                                          marker=dict(size=7, symbol="diamond", color=col_, line=dict(color=ink, width=2))))
        f3.update_layout(height=540, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.02, font=dict(size=10)),
                         scene=dict(xaxis_title="연 변동성 (%)", yaxis_title="연 수익 (%)", zaxis_title="Sharpe", camera=dict(eye=dict(x=1.5, y=-1.5, z=0.9))))
        st.plotly_chart(f3, key=f"{kp}_sp_3d")
    with t2:
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=Fa["vol"], y=Fa["ret"], mode="markers", name=f"+ {grp} (무작위 비중)", marker=dict(size=3, color=PAL[1], opacity=0.18), hoverinfo="skip"))
        f2.add_trace(go.Scatter(x=Fb["vol"], y=Fb["ret"], mode="markers", name=f"{nb} (무작위 비중)", marker=dict(size=3, color=gry, opacity=0.25), hoverinfo="skip"))
        for F_, nm_, col_ in [(Fb, nb, gry), (Fa, f"+ {grp}", PAL[1])]:
            fr = F_["front"]
            if len(fr):
                f2.add_trace(go.Scatter(x=fr[:, 0], y=fr[:, 1], mode="lines", name=f"투자선 · {nm_}", line=dict(color=col_, width=3)))
            f2.add_trace(go.Scatter(x=[F_["tan"][0]], y=[F_["tan"][1]], mode="markers", name=f"★ 최대 Sharpe ({nm_}, 사후)",
                                    marker=dict(size=12, symbol="star", color=col_, line=dict(color=ink, width=1))))
        for j_, n_ in enumerate(Fa["names"]):
            f2.add_trace(go.Scatter(x=[np.sqrt(Fa["S"][j_, j_])], y=[Fa["mu"][j_]], mode="markers+text", text=[n_[:14]], textposition="top center", showlegend=False,
                                    textfont=dict(size=9), marker=dict(size=6, color=gry if n_ in base else PAL[1])))
        if not per.startswith("전체"):
            for (nm_, (v_, r_, s_)), col_ in zip(oos.items(), [gry, PAL[1]]):
                f2.add_trace(go.Scatter(x=[v_], y=[r_], mode="markers", name=f"◇ {nm_}: 학습 비중의 시험 실제 (Sharpe {s_:+.2f})",
                                        marker=dict(size=13, symbol="diamond", color=col_, line=dict(color=ink, width=2))))
        st.plotly_chart(fig_layout(f2, 460, title=f"효율적 투자선 ({per.split(' (')[0]}) — 회색 = {nb}, 주황 = {na_}", xaxis=dict(title="연 변동성 (%)"),
                                   yaxis=dict(title="연 수익 (%)"), legend=dict(orientation="h", y=-0.22, font=dict(size=10))), key=f"{kp}_sp_2d")
    shs = lambda F_: float(F_["tan"][1] / F_["tan"][0])  # noqa: E731
    Fte_b, Fte_a = _frontier_calc(te[base]), _frontier_calc(te)
    S_ = pd.DataFrame({"학습 최대 Sharpe (~2016, 사후)": [shs(Tb), shs(Ta)], "시험 실제 Sharpe (학습 비중 고정, 2017~)": [oos[nb][2], oos[na_][2]],
                       "시험 최대 Sharpe (2017~, 사후)": [shs(Fte_b), shs(Fte_a)]}, index=[nb, na_])
    S_.loc["차이 (더한 효과)"] = S_.iloc[1] - S_.iloc[0]
    st.dataframe(S_.style.format("{:+.2f}"), use_container_width=True)
    wt = pd.Series(Ta["w_tan"], Ta["names"])
    d_in, d_oos = float(S_.iloc[2, 0]), float(S_.iloc[2, 1])
    w_add = float(wt.reindex(add).fillna(0).sum())
    verdict = (f" → 학습 최대 Sharpe 조합이 더한 전략에 비중을 거의 주지 않음({w_add:.0%}) — 기존 3종으로 이미 충분, 투자선이 거의 그대로." if w_add < 0.02 else
               " → 학습에서 본 개선이 시험에서 사라짐: 사후 선택 몫이 큼." if d_oos <= 0 < d_in else
               " → 학습 비중을 고정해도 시험에서 개선이 남음." if d_oos > 0 else ".")
    st.caption(f"학습 {tr.index[0]:%Y-%m}~2016 ({len(tr)}주) · 시험 2017~{te.index[-1]:%Y-%m} ({len(te)}주). 학습 최대 Sharpe 비중(+ {grp}): "
               + " · ".join(f"{k.split(' (')[0][:22]} {v:.0%}" for k, v in wt.sort_values(ascending=False).items() if v >= 0.01) + ". "
               "'학습 최대 Sharpe' 차이는 늘 0 이상(재료가 많을수록 사후 최적은 좋아짐) — 그 개선이 '시험 실제'에서도 남아야 진짜 정보. "
               f"시험 실제 차이 {d_oos:+.2f}{verdict} 시험 Sharpe 95% 구간은 약 ±{1.96 / np.sqrt(len(te) / 52):.2f}라 그보다 작은 차이는 우연과 구별 어려움. "
               "점·투자선은 섞기만 한 상태(목표 변동성으로 다시 맞추기 전).")
    rows = []
    Xb = sm.add_constant(X[base])
    for a_ in add:
        res = sm.OLS(X[a_], Xb).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        rows.append({"전략": a_[:60], "연 알파 (%)": float(res.params["const"] * 52), "알파 t값": float(res.tvalues["const"]),
                     **{f"β {b[:10]}": float(res.params[b]) for b in base}, "R²": float(res.rsquared)})
    st.markdown("**스패닝 회귀 (Huberman-Kandel) — 더한 전략 = α + β × 기존 3종 + 오차**")
    st.dataframe(pd.DataFrame(rows).set_index("전략").style.format("{:+.2f}"), use_container_width=True)
    st.caption(f"전체 {X.index[0]:%Y-%m}~ 주간, Newey-West(4주) 표준오차. α = 기존 3종으로 복제하고 남는 연 수익 — |t| ≥ 2이면 그 전략이 투자선을 바깥으로 밀어낼 정보가 있다는 근거. "
               "α가 0 근처면 기존 3종 조합으로 같은 위험·수익을 이미 만들 수 있음(투자선이 거의 안 움직임). β = 기존 3종에 얼마나 닮았나.")


def _alt_ts_section():
    import experiments as EX
    st.markdown("##### 대안 시계열 모델 — Prophet · Chronos · TimesFM · PatchTST · iTransformer · TFT · Neural ODE · 시공간 GNN")
    st.caption("같은 19개 품목 · 매주 금요일 각 모델이 '4주 뒤 수익률'을 예측 → 두 방식으로 매매: 방향 = 예측 부호 ÷ 품목 변동성(품목마다 롱 또는 숏), "
               "순위 롱숏 = 예측 ÷ 변동성 상위 1/3 롱 · 하위 1/3 숏(4주 보유, 달러 중립). 모두 편도 5bp · 사전 목표 변동성 10% · 평가 2012-07–2026. "
               "파운데이션 모델(Chronos·TimesFM)은 학습 없이 직전 512영업일만 보고 예측(zero-shot), 신경망 5종은 19개 품목 자료로 매년 1월 재학습(그해 자료는 안 씀), "
               "Prophet은 직전 3년 추세 변곡 + 연간 계절을 매주 다시 적합. "
               "정식 패키지 = prophet(Meta) · neuralforecast(Nixtla: PatchTST · iTransformer · TFT) · torchdiffeq(적응 스텝 ODE-RNN) · torch_geometric(GCN + GRU), "
               "자체 경량 구현 = 같은 구조를 줄여 직접 짠 판(비교용). 결과는 미리 계산(`python experiments.py alt_ts`).")
    st.caption(f"📌 결과 고정: {PALT_MD} 자료 기준. 데이터 업데이트를 해도 다시 계산하지 않습니다 (신경망 학습 포함 1–2시간).")
    if "alt_ts" not in EX.EXPERIMENTS:                              # 서버가 예전 experiments 모듈을 들고 있으면 다시 읽음
        import importlib
        EX = importlib.reload(EX)
    T, meta = EX.load("alt_ts")
    fw = EX.DATA / "alt_ts_weekly_compare.csv"
    if T is None or not fw.exists():
        st.info("결과가 없습니다 — `python experiments.py alt_ts` 실행 필요 (신경망 학습 포함 약 1~2시간, CPU).")
        return
    T = T.set_index("모델")
    T.insert(0, "구분", ["기준" if i.startswith(("기준", "모델 합의")) else "파운데이션 (zero-shot)" if "zero-shot" in i else
                       "정식 패키지" if "정식" in i else "자체 경량 구현" for i in T.index])
    T = T.iloc[np.argsort([["정식 패키지", "자체 경량 구현", "파운데이션 (zero-shot)", "기준"].index(g) for g in T["구분"]], kind="stable")]
    num = {c: ("{:.1%}" if c == "방향 적중률" else "{:+.3f}" if c.startswith("IC") else "{:+.2f}") for c in T.columns if c != "구분"}
    st.dataframe(T.style.format(num, na_rep="–"),
                 height=min(36 * (len(T) + 1) + 4, 470))
    D = pd.read_csv(fw, index_col=0, parse_dates=True, encoding="utf-8-sig")
    md = st.radio("매매 방식", ["방향", "순위 롱숏"], horizontal=True, key="ls_alt_mode")
    f = go.Figure()
    for j, c in enumerate([c for c in D.columns if c.endswith(f"| {md}")]):
        nm = c.rsplit(" | ", 1)[0]
        base = nm.startswith("기준")
        f.add_trace(go.Scatter(x=D.index, y=D[c].cumsum(), name=nm, line=dict(color=("#e8e6e1" if DARK else "#0b0b0b") if base else PAL[j % 8], width=2.4 if base or nm.startswith("모델 합의") else 1.2,
                                                                        dash="dash" if base else None)))
    st.plotly_chart(fig_layout(f, 380, title=f"누적 수익 (%, {md}, 목표 변동성 10%) — 굵은 점선 = 52주 모멘텀", hovermode="x",
                               legend=dict(orientation="h", y=-0.3, font=dict(size=10))), key="ls_alt_cum")
    m = T.drop(index=[i for i in T.index if i.startswith(("기준", "모델 합의"))])
    b_ = T.loc[[i for i in T.index if i.startswith("기준")][0]]
    best = m[f"{md} Sharpe"].idxmax()
    st.caption(f"{md} 기준 가장 나은 모델: {best} ({m.loc[best, f'{md} Sharpe']:+.2f}) vs 52주 모멘텀 {b_[f'{md} Sharpe']:+.2f}. "
               f"모델 평균 방향 적중률 {m['방향 적중률'].mean():.1%} · 평균 IC {m['IC (주별 순위 상관)'].mean():+.3f} (0이면 순위 예측력 없음). "
               "Sharpe 95% 구간은 14년 기준 약 ±0.53이라 |Sharpe| 0.5 미만은 우연과 구별하기 어려움. "
               f"계산 시각 {meta.get('run_at', '?')}.")
    _alt_ts_extra(EX)
    ic_f = EX.DATA / "alt_ts_mix_ic_compare.csv"
    pre = list(pd.read_csv(ic_f, encoding="utf-8-sig").query("`후보 포함` == True")["모델"]) if ic_f.exists() else []
    _spanning_section("대안 시계열", "alt", [f"{m_} → 순위 롱숏" for m_ in pre], "학습 구간 IC > 0.02 모델의 순위 롱숏판 — 시험 기간을 보지 않고 고른 후보")


def _alt_ts_extra(EX):
    """보완: ① 정식 vs 경량 짝 ② 시드 흔들림 ③ 롱숏 혼합 후보 (experiments.alt_ts_plus 결과)."""
    rd = lambda n: pd.read_csv(EX.DATA / n, encoding="utf-8-sig") if (EX.DATA / n).exists() else None
    pr, sd = rd("alt_ts_pairs_compare.csv"), rd("alt_ts_seeds_compare.csv")
    ic, mc, wk = rd("alt_ts_mix_ic_compare.csv"), rd("alt_ts_mix_compare.csv"), rd("alt_ts_mix_walk_compare.csv")
    if pr is None:
        st.info("보완 분석(정식 vs 경량 · 시드 흔들림 · 롱숏 혼합 후보) 결과가 없습니다 — `python experiments.py alt_ts_plus` 실행 필요.")
        return
    fmt = lambda df: {c: ("{:.1%}" if ("적중률" in c or "일치율" in c) else "{:+.3f}" if ("IC" in c or "상관" in c) else "{:+.2f}")
                      for c in df.columns if pd.api.types.is_float_dtype(df[c])}
    st.markdown("##### ① 정식 패키지 vs 자체 경량 구현 — 구현을 바꾸면 결과가 달라지나")
    st.dataframe(pr.set_index("모델").style.format(fmt(pr), na_rep="–"))
    st.caption("같은 이름의 모델을 정식 패키지(prophet · neuralforecast · torchdiffeq · torch_geometric)와 직접 줄여 짠 판으로 각각 돌린 결과. "
               "예측 상관이 낮거나 Sharpe 부호가 엇갈리면 '그 모델의 성과'가 아니라 구현·초기값·하이퍼파라미터에 따라 바뀌는 결과라는 뜻.")
    if sd is not None:
        st.markdown("##### ② 시드 흔들림 — 같은 신경망을 초기값(시드)만 바꿔 3번 학습")
        st.dataframe(sd.set_index("모델").style.format(fmt(sd), na_rep="–"))
        st.caption("시드 = 신경망 초기 가중치·표본 순서를 정하는 난수. 최소~최대 폭이 넓으면 한 번 돌린 Sharpe는 운의 몫이 큼. "
                   "'시드 평균 예측' = 세 번 학습한 예측을 평균해 매매(앙상블) — 흔들림을 줄이는 표준 처방. Neural ODE 2종은 학습이 느려 제외.")
    else:
        st.caption("② 시드 흔들림: 신경망 재학습 중 — 끝나면 표시됩니다.")
    if ic is not None and mc is not None:
        st.markdown("##### ③ 롱숏 혼합 후보 — 학습 구간 IC > 0.02 모델을 'MinVar 50 + 모멘텀 25 + 모델 25'로")
        icc = "학습 IC (2012-07–2016)"
        ic_ = ic.assign(**{"후보 포함": ic["후보 포함"].map({True: "✓ (IC > 0.02)", False: "–"})}).set_index("모델")
        st.dataframe(ic_.style.format({icc: "{:+.3f}"}), height=min(36 * (len(ic_) + 1) + 4, 470), use_container_width=True)
        mc_ = mc.assign(**{icc: mc[icc].map(lambda v: "– (기준)" if pd.isna(v) else f"{v:+.3f}")}).set_index("후보")
        st.dataframe(mc_.style.format({c: f for c, f in fmt(mc_).items() if c != icc}, na_rep="–"), use_container_width=True)
        if wk is not None:
            st.dataframe(wk.drop(columns=["요약"]).set_index("시험 구간").style.format(fmt(wk), na_rep="–"))
            sm = wk["요약"].dropna()
            if len(sm) and str(sm.iloc[0]).strip():
                st.caption("이어붙인 표본 외: " + str(sm.iloc[0]))
        st.caption("미래 정보 차단: 후보에 넣을 모델은 2012-07–2016 IC로만 고름(시험 기간 IC로 고르면 사후 선택). 모델 매매는 순위 롱숏(달러 중립)판. "
                   "구간 이동 선택 = 기존 혼합 후보(가치·따라잡기·칼만 등) 전부 + 새 후보 중 학습 Sharpe 1위를 골라 다음 구간에 적용 — "
                   "롱숏 탭과 같은 방식이지만 모델 예측이 2012년부터라 시험 구간은 2017–20 · 2021–26 두 개.")
    mt, mq = rd("alt_ts_macro_compare.csv"), rd("alt_ts_macro_q_compare.csv")
    if mt is None:
        return
    st.markdown("##### ④ 매크로 3종 추가 — VIX · 신용 스프레드(BAA−10년) · 달러 인덱스 4주 변화")
    mt_ = mt.copy()
    mt_["모델"] = [m + "  ·  " + g for m, g in zip(mt_["모델"], mt_["매크로"])]
    st.dataframe(mt_.drop(columns=["매크로"]).set_index("모델").style.format(fmt(mt_.drop(columns=["모델", "매크로"])), na_rep="–"),
                 height=min(36 * (len(mt_) + 1) + 4, 520), use_container_width=True)
    st.caption("HMM 스트레스 모형에 쓰던 거시 중 3종을 그 시점까지 자료로만 표준화하고 1주 늦춰(발표 지연·미래 정보 차단) 입력. "
               "TFT = 과거 변수, iTransformer = 추가 토큰(19 + 3), GNN = 노드 입력에 붙임, Chronos-2 = 과거 공변량(4주마다 결정, 없음판도 같은 날짜). "
               "같은 시드끼리 비교 — 차이가 시드 흔들림(±0.3 안팎)보다 작으면 우연과 구별 안 됨. PatchTST 정식판은 외부 변수를 받지 않아 제외.")
    if mq is not None:
        st.markdown("**Chronos-2 분위(예측 범위) 품질 — 매크로가 변동성을 더 잘 맞히나**")
        st.dataframe(mq.set_index(["매크로", "기간"]).drop(columns=["모델"]).style.format(
            {"10–90% 포함률": "{:.1%}", "평균 폭 (로그, 20일)": "{:.3f}", "폭↔실제 |수익률| 순위상관": "{:+.3f}", "분위 손실 (×100)": "{:.3f}"}), use_container_width=True)
        st.caption("포함률 목표 80%. 순위상관이 클수록 '범위가 넓다고 한 때 실제로 크게 움직였다' = 변동성 예측력. 분위 손실은 작을수록 좋음.")
    else:
        st.caption("Chronos-2 분위 품질 비교는 계산 중 — 끝나면 이 자리에 표시됩니다.")



def _multimodal_section():
    import experiments as EX
    if "multimodal" not in EX.EXPERIMENTS:
        import importlib
        EX = importlib.reload(EX)
    st.markdown("##### 멀티모달 — 뉴스·USDA 원문(FinBERT 감성) · 리포트 수치 · 기상 · 매크로·금융을 수치 전략에 섞으면")
    st.caption(f"📌 결과 고정: {PMM_MD} 자료 기준 (`python experiments.py multimodal`로 미리 계산). 데이터 업데이트를 해도 다시 계산하지 않습니다.")
    st.caption("갈래 7개: ① 텍스트 = USDA WASDE 서술 문단(밀·곡물·유지작물·설탕·축산·면화) · EIA STEO 서술(원유·연료·가스) · 연준 베이지북(에너지·금속·농업 문장) 서술을 FinBERT(금융 문장 감성 모델)로 채점 (GDELT 뉴스는 요청 제한으로 수집이 너무 느려 제외), "
               "② 리포트 수치 = WASDE 기말재고·재고율·생산 수정폭, EIA 주간 재고(원유·휘발유·난방유·천연가스, 5년 평년 대비), EIA STEO 3개월 가격 전망 수정폭, LME 구리 재고·금/은 ETF 보유량, USDA NASS 주간 작황 우량 비율, "
               "③ 포지션 = CFTC COT 19개 품목(투기·상업 순포지션, 쏠림, 4주 변화), "
               "④ 기상 = NOAA 엘니뇨 ONI·남방진동 SOI·지구 기온 편차 + Open-Meteo 주산지 22곳 4주 강수(z)·기온 편차·냉난방 수요, "
               "⑤ 매크로 = VIX·신용 스프레드·달러 4주 변화, ⑥ 금융(FRED 추가) = 실질금리·기대인플레·금리차·환율 4종·금융여건·스트레스·실업수당·연준 자산·원유/금 변동성, "
               "⑦ 월간 거시(FRED) = CPI·실업률·산업생산·M2·소비자심리·PPI·고용 전년비(발표 지연 45일, 최신 수정치라 미래 정보 일부 섞일 수 있음). "
               "공통 값(기상 지수·매크로·금융·거시)은 '품목별 과거 민감도(β) × 현재 값'으로 바꿈. "
               "모든 값은 발표 이후에만 사용. 신호 방향(+/−)은 두 방식: ① 매주 갱신 = 그 시점까지 104주 순위 IC 평균 부호, ② 고정 = 특성이 생긴 뒤 처음 3년 IC로 한 번 정하고 고정(사후 부호 선택 없음). "
               "매매는 순위 롱숏(상위 1/3 롱 · 하위 1/3 숏, 4주 보유), 편도 5bp, 목표 변동성 10%. 평가 2007–(롱숏 탭과 같음, 최근 5년 2021-11~ 따로), 자료는 1990년부터.")
    T, meta = EX.load("multimodal")
    fw = EX.DATA / "multimodal_weekly_compare.csv"
    if T is None or not fw.exists():
        st.info("결과가 없습니다 — 자료 수집(multimodal.py: 기상·WASDE 원문·GDELT) 후 `python experiments.py multimodal` 실행 필요.")
        return
    src = meta.get("settings", {}).get("자료", {})
    if src:
        st.caption("자료 상태: " + " · ".join(f"{k} {'✓' if v else '✕ (없음 — 이 갈래에서 빠짐)'}" for k, v in src.items()) + " · Reddit: 과거 자료를 API가 주지 않아 백테스트 미사용 (collect_reddit.py로 앞으로 수집)")
    T = T.set_index("전략")
    f_ = {c: "{:+.2f}" for c in T.columns if pd.api.types.is_float_dtype(T[c])}
    f_.update({"연변동성 (%)": "{:.1f}", "최대낙폭 (%)": "{:.1f}", "기존 70/30과 상관": "{:+.2f}"})
    st.dataframe(T.style.format(f_, na_rep="–"), height=min(36 * (len(T) + 1) + 4, 560), use_container_width=True)
    if any(i.startswith("대안 시계열 ×") for i in T.index):
        st.caption("'대안 시계열 × …' 행 = ④ 예측 › 전략 백테스트에서 대안 시계열 모델(경량 TFT·iTransformer·GNN, Prophet 방식, neuralforecast TFT 등)에 멀티모달을 입력으로 넣어 "
                   "4주 뒤 수익률을 예측하고 순위 롱숏으로 매매한 결과(매년 재학습, 2007–). '없음 (비교)' = 같은 모델의 설명변수 없는 판. "
                   "'혼합 · MinVar 50 + 모멘텀 25 + 모델·멀티모달 25' = 그 전략을 다른 혼합과 같은 방식으로 25% 섞은 것. 대안 모델 그리드가 끝날 때마다 행이 늘어남.")
    D = pd.read_csv(fw, index_col=0, parse_dates=True, encoding="utf-8-sig")
    grp = st.radio("그래프", ["혼합 전략", "갈래 단독 (부호 매주 갱신)", "갈래 단독 (부호 고정)", "대안 시계열 × 멀티모달"], horizontal=True, key="mm_grp")
    pick = {"혼합 전략": lambda c: c.startswith("혼합"),
            "갈래 단독 (부호 매주 갱신)": lambda c: c.startswith("갈래 단독 ·") or c == "멀티모달 단독 (모든 갈래 평균)" or c.startswith("기준"),
            "갈래 단독 (부호 고정)": lambda c: c.startswith("갈래 단독 (고정 부호)") or c == "멀티모달 단독 (고정 부호)" or c.startswith("기준"),
            "대안 시계열 × 멀티모달": lambda c: c.startswith("대안 시계열 × 멀티모달") or "(기존)" in c}[grp]
    cols = [c for c in D.columns if pick(c)]
    f = go.Figure()
    for j, c in enumerate(cols):
        base = "(기존)" in c
        f.add_trace(go.Scatter(x=D.index, y=D[c].cumsum(), name=c, line=dict(color=("#e8e6e1" if DARK else "#0b0b0b") if base else PAL[j % 8],
                                                                         width=2.4 if base or "멀티모달" in c else 1.3, dash="dash" if base else None)))
    st.plotly_chart(fig_layout(f, 400, title="누적 수익 (%, 목표 변동성 10%) — 굵은 점선 = 기존 MinVar 70 + 모멘텀 30", hovermode="x",
                               legend=dict(orientation="h", y=-0.3, font=dict(size=10))), key="mm_cum")
    b0, m25 = T.loc["혼합 · MinVar 70 + 모멘텀 30 (기존)"], T.loc["혼합 · MinVar 50 + 모멘텀 25 + 멀티모달 25"]
    st.caption(f"멀티모달 25% 혼합 {m25['Sharpe']:+.2f} vs 기존 70/30 {b0['Sharpe']:+.2f} (연 {m25['기존 대비 연 수익 차 (%p)']:+.2f}%p, 차이 t {m25['차이 t값']:+.2f}). "
               f"Sharpe 95% 구간은 2007– 전체 약 ±0.45, 최근 5년만이면 약 ±0.9라 그보다 작은 차이는 우연과 구별하기 어려움. 계산 시각 {meta.get('run_at', '?')}.")
    ic = EX.DATA / "multimodal_ic_compare.csv"
    if ic.exists():
        I = pd.read_csv(ic, encoding="utf-8-sig")
        st.markdown("**특성별 예측력 (평가 기간 주별 순위 IC — 방향 맞춤 전 원래 부호)**")
        st.dataframe(I.set_index(["갈래", "특성"]).style.format({"평가 기간 IC 평균": "{:+.3f}", "IC t값": "{:+.2f}"}, na_rep="–"), use_container_width=True)
        st.caption("IC = 그 주 신호 순위와 이후 4주 실제 수익률 순위의 상관(+면 신호 높은 품목이 더 오름). 4주 겹침을 감안해 t값은 표본 수를 1/4로 계산. "
                   "|t| ≥ 2가 아니면 우연과 구별 어려움. 매매에서는 이 부호를 그때까지 자료로만 정해 곱함.")
    _mm_model_section(EX)
    _spanning_section("멀티모달", "mm", ["멀티모달 단독 (고정 부호)", "멀티모달 단독 (모든 갈래 평균)"], "갈래를 고르지 않은 전체 평균 2종 — 사후 선택 없음")


def _mm_model_section(EX):
    """멀티모달 자료를 '예측 모델의 입력'으로 넣었을 때 (④ 전략 백테스트 그리드 결과에서): 대안 시계열 × 멀티모달 + 모델군별 없음 vs 넣음."""
    st.markdown("##### 예측 모델 입력으로 넣으면 — 대안 시계열 × 멀티모달 (④ 예측 › 전략 백테스트 결과)")
    T, meta = EX.load("predict_grid")
    if T is None:
        st.info("④ 예측 › 전략 백테스트 결과가 없습니다 — `EX.predict_grid()` 실행 필요.")
        return
    st.caption("위 갈래 신호(IC 부호를 곱해 순위 롱숏)와 달리, 멀티모달 자료를 예측 모델의 설명변수로 넣고 4주 뒤 수익률을 예측해 매매한 결과. 같은 19개 품목 · 매년 재학습 · "
               f"편도 5bp · 목표 변동성 10% · 2007-01~2026-10-26. 계산 시각 {meta.get('run_at', '?')}. "
               "'매크로 + 공통 멀티모달' = 매크로 3종 + 엘니뇨 ONI·SOI·지구 기온 + 금융여건 NFCI·STLFSI·실질금리·5y5y·금리차(모든 품목 같은 값), "
               "'품목별 멀티모달 7갈래' = 품목마다 7갈래(텍스트·리포트·COT·기상·매크로·금융·월간 거시) 요약값.")
    mode = st.radio("매매 방식", ["순위 롱숏", "방향"], horizontal=True, key="mm_pg_mode")
    sc = f"{mode} Sharpe"
    A_ = T[T["모델군"] == "대안 시계열"]
    order = ["설명변수 없음", "매크로 3종", "매크로 + 공통 멀티모달 (기상·금융 지수)", "품목별 멀티모달 7갈래"]
    if len(A_):
        S = A_.pivot_table(index="모델", columns="설명변수", values=sc).reindex(columns=[c for c in order if c in A_["설명변수"].unique()])
        f = go.Figure(go.Heatmap(z=S.values, x=list(S.columns), y=list(S.index), zmin=-0.6, zmax=0.6, colorscale=DIV,
                                 text=[["" if not np.isfinite(v) else f"{v:+.2f}" for v in r] for r in S.values], texttemplate="%{text}",
                                 hovertemplate="%{y} · %{x}: %{z:+.2f}<extra></extra>"))
        f.update_layout(height=120 + 34 * len(S), margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(f, key="mm_pg_heat")
        done = int(A_["설명변수"].str.contains("멀티모달").sum())
        st.caption(f"대안 모델 × 멀티모달 조합 {done}개 계산 완료 (전체 예정 7개: 경량 iTransformer·TFT·GNN × 공통 멀티모달, 경량 TFT·정식 TFT·Prophet 방식 × 품목별 7갈래, "
                   "정식 iTransformer·TFT × 공통 멀티모달 포함). 빈칸 = 아직 계산 중이거나 구조상 외부 변수를 받지 않는 모델(PatchTST·Neural ODE·zero-shot). 시드 1개라 ±0.3 흔들림.")
    else:
        st.caption("대안 시계열 결과는 계산 중입니다.")
    pairs = [("표준", "① 자기 정보만", ["⑥ 자기 + 멀티모달 전부"], "⑧"), ("패널", "P①", ["P③"], "P⑤"), ("MIDAS", "M①", ["M③"], None),
             ("대안 시계열", "설명변수 없음", ["매크로 + 공통 멀티모달 (기상·금융 지수)", "품목별 멀티모달 7갈래"], None)]
    rows = []
    for g, b0, mms, pl in pairs:
        G = T[T["모델군"] == g]
        for m in G["모델"].unique():
            Gm = G[G["모델"] == m]
            base = Gm[Gm["설명변수"].str.startswith(b0)]
            for mm in mms:
                x = Gm[Gm["설명변수"].str.startswith(mm)]
                if not len(base) or not len(x):
                    continue
                p_ = Gm[Gm["설명변수"].str.startswith(pl)][sc] if pl else pd.Series(dtype=float)
                rows.append({"모델군": g, "모델": m, "멀티모달 입력": x["설명변수"].iloc[0], "없음 Sharpe": float(base[sc].iloc[0]),
                             "넣음 Sharpe": float(x[sc].iloc[0]), "차이": float(x[sc].iloc[0] - base[sc].iloc[0]),
                             "위약 Sharpe": float(p_.iloc[0]) if len(p_) else np.nan, "넣음 최근 5년": float(x[f"{mode} 최근 5년"].iloc[0])})
    if rows:
        R_ = pd.DataFrame(rows)
        st.markdown(f"**모델군별 — 멀티모달을 넣기 전 vs 넣은 뒤 ({mode} Sharpe)**")
        st.dataframe(R_.style.format({c: "{:+.2f}" for c in ["없음 Sharpe", "넣음 Sharpe", "차이", "위약 Sharpe", "넣음 최근 5년"]}, na_rep="–"),
                     hide_index=True, use_container_width=True, height=min(36 * (len(R_) + 1) + 4, 470))
        up = (R_["차이"] > 0).mean()
        st.caption(f"{len(R_)}개 비교 중 멀티모달을 넣어 좋아진 경우 {up:.0%}, 차이 평균 {R_['차이'].mean():+.2f}. "
                   "없음 = 표준 ①·패널 P①·MIDAS M①·대안 '설명변수 없음', 넣음 = 표준 ⑥·패널 P③·MIDAS M③·대안 멀티모달판. "
                   "위약 = 같은 설명변수의 시간 순서를 섞은 판(표준 ⑧은 ⑦ 기준, 패널 P⑤는 P④ 기준) — 넣음이 위약보다 확실히 높지 않으면 멀티모달 정보의 효과로 보기 어려움. "
                   "Sharpe 차이 ±0.45 안쪽은 우연과 구별 어려움. 전체 비교는 ④ 예측 › 전략 백테스트 서브탭.")


@st.cache_data(show_spinner=False)
def _sleeves(roll: bool) -> tuple[pd.DataFrame, dict]:
    """Portfolio Customization 재료: 전략별 주간 수익률(%, 각 목표 변동성 10%) — 모두 고정 결과(롱숏·대안 시계열·멀티모달)에서 읽음."""
    import experiments as EX
    S, G = {}, {}

    def add(g, nm, r):
        S[nm] = r; G[nm] = g
    DX = _lx_run(5.0, roll)["returns"]
    for c in DX:
        if c.startswith("기준") or c[:2] in ("K1", "K2", "S1", "S2", "V1", "W1", "L1", "B1", "C1", "C2") or c == "혼합 · MinVar 70 + 모멘텀 30 (기존)":
            add("수치 전략", c, DX[c])
    D = _ls_run(5.0, True, roll)["returns"]
    for c in D:
        if c.startswith(("①", "②", "③", "④ 시계열 모멘텀 13", "⑤")):
            add("수치 전략", c, D[c])
    AL = _alloc_run(5.0, roll)["returns"]
    for c in AL:
        add("수치 전략", f"비중 방식 · {c}", AL[c])
    rd = lambda n: pd.read_csv(EX.DATA / n, index_col=0, parse_dates=True, encoding="utf-8-sig") if (EX.DATA / n).exists() else pd.DataFrame()  # noqa: E731
    for c, r in rd("alt_ts_weekly_compare.csv").items():
        if not c.startswith("기준"):
            add("대안 시계열", c.replace(" | ", " → "), r)
    for c, r in rd("multimodal_weekly_compare.csv").items():
        if c.startswith(("갈래 단독", "멀티모달 단독")):
            add("멀티모달", c, r)
    return pd.DataFrame(S).sort_index(), G


def _combine(X: pd.DataFrame, method: str, w_user: pd.Series, reb: int = 4, tgt: float = 10.0) -> tuple[pd.Series, pd.DataFrame]:
    """전략 비중: 직접·동일은 고정, 나머지는 4주마다 그 전 주까지 자료로만 다시 계산(미래 정보 없음) → 합친 뒤 사전 목표 변동성 tgt%.
    레버리지 상한 = 4배 × tgt/10 (재료가 10% 기준이라, 재료를 tgt로 키운 뒤 롱숏 탭처럼 4배 상한을 거는 것과 같음)."""
    from scipy.optimize import minimize
    from sklearn.covariance import LedoitWolf
    k = X.shape[1]
    W = pd.DataFrame(np.nan, index=X.index, columns=X.columns)
    if method == "직접 입력":
        W.loc[:, :] = (w_user / w_user.sum()).reindex(X.columns).values
    elif method == "동일 비중":
        W.loc[:, :] = 1 / k
    else:
        win = {"역변동성 (직전 52주)": 52, "최소분산 (직전 104주 공분산)": 104, "과거 Sharpe 비례 (직전 156주, 음수 = 0)": 156}[method]
        for t in range(min(win, 52), len(X), reb):
            past = X.iloc[max(0, t - win):t]
            if method.startswith("역변동성"):
                w = 1 / past.std().replace(0, np.nan)
            elif method.startswith("최소분산"):
                C = LedoitWolf().fit(past.values).covariance_ if k > 1 else np.atleast_2d(past.var().values)
                res = minimize(lambda x: x @ C @ x, np.full(k, 1 / k), bounds=[(0, 1)] * k,
                               constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}], method="SLSQP")
                w = pd.Series(res.x if res.success else np.full(k, 1 / k), X.columns)
            else:
                w = (past.mean() / past.std().replace(0, np.nan)).clip(lower=0)
                w = w if w.sum() > 0 else pd.Series(1.0, X.columns)
            W.iloc[t] = (w / w.sum()).values
        W = W.ffill()
    m = (W * X).sum(1, min_count=1).dropna()
    pv = m.rolling(26, min_periods=13).std().shift(1) * np.sqrt(52)
    return m * (tgt / pv).clip(upper=4.0 * tgt / 10.0).fillna(tgt / 10.0), W.loc[m.index]


@st.cache_data(show_spinner=False)
def _frontier_calc(X: pd.DataFrame, n_rand: int = 4000) -> dict:
    """고른 전략(재료)들의 효율적 투자선: 연율 평균·공분산(전체 평가 기간, 사후) → 무작위 비중 구름 · 투자선 · 최소분산 · 최대 Sharpe. 롱온리, 합 100%."""
    from scipy.optimize import minimize
    mu = X.mean().values * 52
    S = X.cov().values * 52
    k = len(mu)
    rng = np.random.default_rng(0)
    Wr = np.vstack([rng.dirichlet(np.full(k, 0.6), n_rand), np.eye(k)])
    vol = np.sqrt(np.einsum("ij,jk,ik->i", Wr, S, Wr))
    ret = Wr @ mu
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bnd = [(0, 1)] * k
    w0 = np.full(k, 1 / k)
    gmv = minimize(lambda w: w @ S @ w, w0, bounds=bnd, constraints=cons, method="SLSQP").x
    tan = minimize(lambda w: -(w @ mu) / np.sqrt(w @ S @ w), w0, bounds=bnd, constraints=cons, method="SLSQP").x
    fr = []
    for t in np.linspace(gmv @ mu, mu.max(), 40):
        r_ = minimize(lambda w: w @ S @ w, w0, bounds=bnd, constraints=cons + [{"type": "eq", "fun": lambda w, t=t: w @ mu - t}], method="SLSQP")
        if r_.success:
            fr.append((float(np.sqrt(r_.x @ S @ r_.x)), float(r_.x @ mu)))
    pt = lambda w: (float(np.sqrt(w @ S @ w)), float(w @ mu))  # noqa: E731
    return dict(mu=mu, S=S, vol=vol, ret=ret, front=np.array(fr), gmv=pt(gmv), tan=pt(tan), w_tan=tan, w_gmv=gmv, names=list(X.columns))


def _frontier_section(X: pd.DataFrame, w_eff: pd.Series, combo: pd.Series, tgt: float = 10.0, kp: str = "cu"):
    """Portfolio Customization: 고른 전략으로 만들 수 있는 위험-수익 조합(구름) · 효율적 투자선 · 내 포트폴리오 위치 (3D · 2D)."""
    st.markdown("##### 효율적 투자선 — 고른 전략들로 만들 수 있는 위험·수익 조합과 내 포트폴리오의 위치")
    ends = [d_ for d_ in pd.date_range(X.index[0] + pd.DateOffset(years=5), X.index[-1], freq="YE")]
    opts_ = ["전체 기간 (사후)"] + [f"{d_:%Y-%m} 시점 (직전 5년)" for d_ in ends]
    pick = st.select_slider("투자선을 그릴 시점 — 그 시점 직전 5년 자료로만 계산하면 '당시 알 수 있었던' 투자선 (사후 정보 없음)", opts_, opts_[0], key=f"{kp}_ef_when")
    nxt = None
    if pick != opts_[0]:
        d_ = ends[opts_.index(pick) - 1]
        X = X[(X.index > d_ - pd.DateOffset(years=5)) & (X.index <= d_)]
        nx = combo[(combo.index > d_) & (combo.index <= d_ + pd.DateOffset(years=1))]
        combo = combo[(combo.index > d_ - pd.DateOffset(years=5)) & (combo.index <= d_)]
        if len(nx) >= 26:
            nxt = (float(nx.std() * np.sqrt(52)), float(nx.mean() * 52))
        st.caption(f"{X.index[0]:%Y-%m} ~ {X.index[-1]:%Y-%m} 자료만 사용. ◇ = 내 포트폴리오의 그다음 1년 실제 위치 — 과거 5년 투자선이 다음 해를 얼마나 맞히는지 비교 "
                   "(대개 크게 빗나감: 평균 수익률은 5년 자료로도 추정 오차가 큼).")
    tag = "(사후)" if nxt is None and pick == opts_[0] else "(그 5년 기준)"
    F = _frontier_calc(X)
    w = (w_eff / w_eff.sum()).reindex(F["names"]).fillna(0).values
    u_vol, u_ret = float(np.sqrt(w @ F["S"] @ w)), float(w @ F["mu"])
    c_vol, c_ret = float(combo.std() * np.sqrt(52)), float(combo.mean() * 52)
    sh = F["ret"] / F["vol"]
    smin, smax = float(np.nanpercentile(sh, 1)), float(np.nanmax(sh))
    short = [n[:22] for n in F["names"]]
    ink = "#e8e6e1" if DARK else "#0b0b0b"
    xmax = max(float(F["vol"].max()), c_vol, u_vol, nxt[0] if nxt else 0) * 1.12
    dv, dr = np.sqrt(np.diag(F["S"])), F["mu"]
    t3, t2 = st.tabs(["3D 보기 (위험 × 수익 × Sharpe)", "2D 보기 (평면 · 레버리지선)"])
    with t3:
        f3 = go.Figure()
        f3.add_trace(go.Mesh3d(x=F["vol"], y=F["ret"], z=sh, alphahull=0, opacity=0.13, color=PAL[0], hoverinfo="skip", name="가능 영역 (껍질)",
                               lighting=dict(ambient=0.55, diffuse=0.8, specular=0.25, roughness=0.6)))
        f3.add_trace(go.Scatter3d(x=F["vol"], y=F["ret"], z=sh, mode="markers", name="무작위 비중 4,000개",
                                  marker=dict(size=2.2, color=sh, colorscale=SEQ, cmin=smin, cmax=smax, opacity=0.75,
                                              colorbar=dict(title="Sharpe", thickness=12, len=0.6)),
                                  hovertemplate="변동성 %{x:.1f}% · 수익 %{y:.1f}% · Sharpe %{z:.2f}<extra></extra>"))
        if len(F["front"]):
            fx, fy = F["front"][:, 0], F["front"][:, 1]
            f3.add_trace(go.Scatter3d(x=fx, y=fy, z=fy / fx, mode="lines", name="효율적 투자선", line=dict(color=PAL[1], width=7),
                                      hovertemplate="투자선 · 변동성 %{x:.1f}% · 수익 %{y:.1f}%<extra></extra>"))
            f3.add_trace(go.Scatter3d(x=fx, y=fy, z=np.full(len(fx), smin), mode="lines", name="투자선 그림자", showlegend=False,
                                      line=dict(color=PAL[1], width=3, dash="dot"), opacity=0.5, hoverinfo="skip"))
        for nm, (vx, vy), sym, col in [(f"최대 Sharpe {tag}", F["tan"], "diamond", PAL[3]), ("최소분산", F["gmv"], "circle", PAL[6]),
                                       ("★ 내 포트폴리오 (섞기만)", (u_vol, u_ret), "diamond", PAL[2]),
                                       (f"☆ 목표 변동성 {tgt:g}%로 다시 맞춘 실제", (c_vol, c_ret), "diamond-open", PAL[2])]:
            f3.add_trace(go.Scatter3d(x=[vx], y=[vy], z=[vy / vx], mode="markers+text", text=[nm], textposition="top center", name=nm,
                                      marker=dict(size=9, symbol=sym, color=col, line=dict(color=ink, width=1)), textfont=dict(size=11)))
        f3.add_trace(go.Scatter3d(x=dv, y=dr, z=dr / dv, mode="markers+text", text=short, textposition="bottom center", name="재료 전략 (단독)",
                                  marker=dict(size=5, color="#b4b2a9", symbol="square", line=dict(color=ink, width=0.5)), textfont=dict(size=10)))
        f3.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.02, font=dict(size=10)),
                         scene=dict(xaxis_title="연 변동성 (%)", yaxis_title="연 수익률 (%)", zaxis_title="Sharpe", aspectmode="manual",
                                    aspectratio=dict(x=1.15, y=1, z=0.7), camera=dict(eye=dict(x=1.55, y=-1.55, z=0.85))))
        st.plotly_chart(f3, key=f"{kp}_ef3d")
        st.caption("마우스로 돌려 보기 · 휠로 확대. 높이(z) = Sharpe라 높이 솟은 곳일수록 같은 위험에서 수익이 많은 조합. 주황 선 = 효율적 투자선, 바닥 점선 = 그 그림자. "
                   "점 구름·투자선은 재료를 섞기만 한 조합(레버리지 다시 맞추기 전)이라 변동성·수익이 목표보다 낮게 모여 있음 → ☆ = 섞은 뒤 목표 변동성으로 다시 키운 실제 위치(성과표의 수익률과 같은 점).")
    with t2:
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=F["vol"], y=F["ret"], mode="markers", name="무작위 비중 4,000개",
                                marker=dict(size=6, color=sh, colorscale=SEQ, cmin=smin, cmax=smax, opacity=0.35, line=dict(width=0),
                                            colorbar=dict(title="Sharpe", thickness=12, len=0.7)),
                                hovertemplate="변동성 %{x:.1f}% · 수익 %{y:.1f}%<extra></extra>"))
        if len(F["front"]):
            fx, fy = F["front"][:, 0], F["front"][:, 1]
            f2.add_trace(go.Scatter(x=fx, y=fy, mode="lines", line=dict(color=PAL[1], width=14), opacity=0.18, hoverinfo="skip", showlegend=False))
            f2.add_trace(go.Scatter(x=fx, y=fy, mode="lines", name="효율적 투자선", line=dict(color=PAL[1], width=3)))
        tv, tr = F["tan"]
        f2.add_trace(go.Scatter(x=[0, xmax], y=[0, tr / tv * xmax], mode="lines", name="최대 Sharpe 레버리지선 (자본시장선)",
                                line=dict(color=PAL[3], width=1.6, dash="dot")))
        f2.add_trace(go.Scatter(x=[0, xmax], y=[0, u_ret / u_vol * xmax], mode="lines", name="내 포트폴리오 레버리지선", line=dict(color=PAL[2], width=1.6, dash="dash")))
        f2.add_trace(go.Scatter(x=dv, y=dr, mode="markers+text", text=short, textposition="top center", name="재료 전략 (단독)", textfont=dict(size=10),
                                marker=dict(size=10, symbol="square", color="#b4b2a9", line=dict(color=ink, width=0.6))))
        for nm, (vx, vy), sym, col, sz in [("최소분산", F["gmv"], "circle", PAL[6], 13), (f"최대 Sharpe {tag}", F["tan"], "diamond", PAL[3], 15),
                                           ("★ 내 포트폴리오 (섞기만)", (u_vol, u_ret), "star", PAL[2], 20),
                                           (f"☆ 내 포트폴리오 (목표 변동성 {tgt:g}% 적용 후, 실제)", (c_vol, c_ret), "star-open", PAL[2], 20)]:
            f2.add_trace(go.Scatter(x=[vx], y=[vy], mode="markers", name=nm, marker=dict(size=sz, symbol=sym, color=col, line=dict(color=ink, width=1.2)),
                                    hovertemplate=nm + "<br>변동성 %{x:.1f}% · 수익 %{y:.1f}%<extra></extra>"))
        if nxt:
            f2.add_trace(go.Scatter(x=[nxt[0]], y=[nxt[1]], mode="markers", name="◇ 내 포트폴리오 · 그다음 1년 실제", marker=dict(size=16, symbol="diamond-open", color=PAL[7],
                                    line=dict(width=2.5)), hovertemplate="다음 1년 실제<br>변동성 %{x:.1f}% · 수익 %{y:.1f}%<extra></extra>"))
        f2.add_annotation(x=c_vol, y=c_ret, ax=u_vol, ay=u_ret, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowwidth=1.5,
                          arrowcolor=PAL[2], text="")
        ymin = min(0.0, float(F["ret"].min()), float(dr.min()), nxt[1] if nxt else 0) - 1
        ymax = max(float(F["ret"].max()), c_ret, float(dr.max()), nxt[1] if nxt else 0) * 1.25 + 1
        st.plotly_chart(fig_layout(f2, 520, hovermode="closest", xaxis=dict(title="연 변동성 (%)", range=[0, xmax], zeroline=False),
                                   yaxis=dict(title="연 수익률 (%, 산술 평균)", range=[ymin, ymax]), legend=dict(orientation="h", y=-0.18, font=dict(size=10))),
                        key=f"{kp}_ef2d")
    wt = pd.DataFrame({"최대 Sharpe (사후) 비중": F["w_tan"], "최소분산 비중": F["w_gmv"], "내 비중": w}, index=short) * 100
    with st.expander("그래프 읽는 법 · 최대 Sharpe·최소분산 조합의 비중", expanded=False):
        st.markdown(f"""
- **점 구름** = 고른 전략 {len(short)}개를 무작위 비율(롱만, 합 100%)로 섞은 4,000개 포트폴리오. 색이 밝을수록(노랑·빨강) Sharpe가 높음.
- **효율적 투자선 (주황)** = 같은 변동성에서 가장 높은 수익을 내는 조합들의 경계. 이 선 아래쪽 조합은 같은 위험에 수익이 덜함.
- **최소분산 (●)** = 변동성이 가장 낮은 조합 ({F['gmv'][0]:.1f}% · 수익 {F['gmv'][1]:.1f}%). **최대 Sharpe (◆)** = 위험 1단위당 수익이 가장 큰 조합 ({F['tan'][0]:.1f}% · {F['tan'][1]:.1f}%, Sharpe {F['tan'][1] / F['tan'][0]:.2f}).
- **★ 내 포트폴리오 (섞기만)** = 지금 비중으로 섞은 위치 ({u_vol:.1f}% · {u_ret:.1f}%, Sharpe {u_ret / u_vol:.2f}). 섞으면 손실이 서로 상쇄돼 변동성이 재료(각 10%)보다 낮아짐 →
  실제 운용은 **목표 변동성 {tgt:g}%로 다시 맞춤 (☆, 화살표)** — 레버리지선(초록 점선)을 따라 오른쪽 위로 이동, Sharpe는 거의 그대로 ({c_vol:.1f}% · {c_ret:.1f}%).
- **자본시장선 (노랑 점선)** = 최대 Sharpe 조합에 레버리지를 걸 때의 위험-수익. 내 레버리지선이 이 선에 가까울수록 고른 비중이 효율적.
- ⚠ 평균·공분산을 **평가 기간 전체로 계산한 사후 그림** — 최대 Sharpe 조합은 그때 미리 알 수 없었음. 비중을 고르는 근거로 쓰려면 위 표의 '~2016 / 2017~' 하위 기간 Sharpe로 확인.
""")
        st.dataframe(wt.style.format("{:.0f}%"), use_container_width=True)


def tab_custom():
    import ls_strategies as LS
    st.subheader("Portfolio Customization — 전략·모델·데이터 갈래를 골라 나만의 포트폴리오 만들기")
    mode = st.radio("구성 방식", ["전략 조합 (모멘텀 · 가치 · 대안 시계열 · 멀티모달 …)", "품목 단위 평균-분산 최적화 (Newton, 기존 바스켓 포트폴리오)"],
                    horizontal=True, key="cu_mode")
    if mode.startswith("품목"):
        tab_portfolio()
        return
    st.caption(f"재료 = 롱숏 탭 · 대안 시계열 탭 · 멀티모달 탭의 전략별 주간 수익률(편도 5bp 비용 차감, 각각 목표 변동성 10%). 결과가 고정된 탭에서 읽으므로 기간은 최대 {PLS_MD}, "
               "고른 전략이 모두 있는 구간만 사용(대안 시계열을 넣으면 2012-07~). 합친 뒤 직전 26주 실현 변동성으로 아래에서 고른 목표 변동성(기본 10%)에 맞춤(레버리지 상한 4배 × 목표/10, 롱숏 탭 혼합과 같은 방식).")
    with st.spinner("전략 재료 읽는 중… (처음 한 번만 1–2분, 이후 저장된 결과 사용)"):
        X0, G = _sleeves(ROLL)
    groups = {"수치 전략": "수치 전략 (롱숏 탭: MinVar · 모멘텀 · 가치 · 칼만 · COT …)", "대안 시계열": "대안 시계열 (모델 예측 → 방향 / 순위 롱숏)",
              "멀티모달": "멀티모달 (텍스트·리포트·포지션·기상·매크로 갈래)"}
    defaults = {"수치 전략": ["기준 · MinVar 롱 (⑥)", "기준 · 52주 모멘텀 (④)", "V1 가치 (5년 평균 대비 싼 품목 롱)"], "대안 시계열": [], "멀티모달": []}
    sel = []
    for g, lab in groups.items():
        opts = [c for c in X0.columns if G[c] == g]
        sel += st.multiselect(lab, opts, [d for d in defaults[g] if d in opts], key=f"cu_sel_{g}")
    if not sel:
        st.info("전략을 하나 이상 고르세요.")
        return
    tgt_v = st.selectbox("목표 변동성 (연, %) — 섞은 뒤 이 수준에 맞춰 레버리지 조절", list(range(5, 55, 5)), index=1, key="cu_tgt",
                         help="기본 10% = 롱숏 탭과 같음. 직전 26주 실현 변동성으로 배율을 정함(미래 정보 없음). 레버리지 상한 = 4배 × 목표/10 — 수익은 목표에 거의 비례하지만 낙폭도 같이 커지고, 복리 수익은 변동성²/2만큼 덜 늘어남")
    c1, c2 = st.columns([1, 2])
    method = c1.selectbox("비중 결정", ["직접 입력", "동일 비중", "역변동성 (직전 52주)", "최소분산 (직전 104주 공분산)", "과거 Sharpe 비례 (직전 156주, 음수 = 0)"],
                          key="cu_method", help="직접·동일 = 기간 내내 고정 비중. 나머지 = 4주마다 그 시점 이전 자료로만 다시 계산 (미래 정보 없음)")
    w_user = pd.Series(1.0, sel)
    if method == "직접 입력":
        dflt = {"기준 · MinVar 롱 (⑥)": 50.0, "기준 · 52주 모멘텀 (④)": 25.0, "V1 가치 (5년 평균 대비 싼 품목 롱)": 25.0}
        ed = c2.data_editor(pd.DataFrame({"전략": sel, "비중 (%)": [dflt.get(s_, 100.0 / len(sel)) for s_ in sel]}), hide_index=True, disabled=["전략"],
                            key="cu_w_" + str(abs(hash(tuple(sel)))), use_container_width=True)
        w_user = pd.Series(ed["비중 (%)"].clip(lower=0).values, ed["전략"].values)
        if w_user.sum() <= 0:
            st.warning("비중 합이 0입니다.")
            return
        c2.caption(f"합계 {w_user.sum():.0f}% → 합 100%로 맞춰 사용")
    X = X0[sel].dropna()
    if len(X) < 104:
        st.warning(f"고른 전략이 함께 있는 기간이 {len(X)}주뿐이라 평가하기 어렵습니다.")
        return
    combo, W = _combine(X, method, w_user, tgt=float(tgt_v))
    k_ = float(tgt_v) / 10.0                                  # 비교 대상(기존 70/30·재료, 모두 10% 기준)도 같은 목표 변동성으로 비례 조정
    base = X0["혼합 · MinVar 70 + 모멘텀 30 (기존)"].reindex(combo.index).fillna(0) * k_
    series = {"★ 내 포트폴리오": combo, "기준 · MinVar 70 + 모멘텀 30 (롱숏 탭 기존)": base, **{s_: X[s_].reindex(combo.index) * k_ for s_ in sel}}
    if tgt_v != 10:
        st.caption(f"목표 변동성 {tgt_v}%: 내 포트폴리오는 그 수준으로 다시 계산, 비교 대상(기존 70/30 · 재료 전략)은 10% 결과를 ×{k_:g}로 비례 조정해 같은 위험에서 비교.")
    rows = []
    for nm, r in series.items():
        r = r.fillna(0)
        d = r - base
        rows.append({"전략": nm, **LS.metrics(r), "Sharpe ~2016": LS.metrics(r[r.index < "2017-01-01"])["Sharpe"],
                     "Sharpe 2017~2021-10": LS.metrics(r[(r.index >= "2017-01-01") & (r.index < "2021-11-01")])["Sharpe"],
                     "Sharpe 최근 5년": LS.metrics(r[r.index >= "2021-11-01"])["Sharpe"], "t값": LS.metrics(r)["Sharpe"] * np.sqrt(len(r) / 52),
                     "기존 대비 연 수익 차 (%p)": float(d.mean() * 52), "차이 t값": float(d.mean() / d.std() * np.sqrt(len(d))) if d.std() > 0 else np.nan})
    T = pd.DataFrame(rows).drop_duplicates("전략").set_index("전략")
    f_ = {c: "{:+.2f}" for c in T.columns}
    f_.update({"연변동성 (%)": "{:.1f}", "최대낙폭 (%)": "{:.1f}"})
    st.dataframe(T.style.format(f_, na_rep="–"), use_container_width=True, height=min(36 * (len(T) + 1) + 4, 470))
    mc, bc = T.loc["★ 내 포트폴리오"], T.loc["기준 · MinVar 70 + 모멘텀 30 (롱숏 탭 기존)"]
    st.info(f"평가 {combo.index.min():%Y-%m} ~ {combo.index.max():%Y-%m} ({len(combo) / 52:.1f}년) · 내 포트폴리오 Sharpe {mc['Sharpe']:+.2f} vs 기존 70/30 {bc['Sharpe']:+.2f} "
            f"(연 {mc['기존 대비 연 수익 차 (%p)']:+.2f}%p, 차이 t {mc['차이 t값']:+.2f}). Sharpe 95% 구간은 약 ±{1.96 / np.sqrt(len(combo) / 52):.2f}라 그보다 작은 차이는 우연과 구별 어려움. "
            "여러 조합을 시험해 보고 가장 좋은 것을 고르면 과대평가(사후 선택) — 고르기 전에 이유를 정해 두는 것이 안전.")
    st.caption("※ 기본값(MinVar 50 + 모멘텀 25 + 가치 25)이 롱숏 탭 표의 0.88과 조금 다르게(0.89–0.90) 나오는 이유: 계산 주기·재료는 같고, 섞은 뒤 목표 변동성 10%로 맞추는 배율"
               "(직전 26주 실현 변동성)의 출발점만 다름 — 여기는 고정 재료가 2007년부터라 2007년 초 약 반년(처음 13주 배율 1, 26주까지 짧은 창)을 새로 계산하고, 롱숏 탭은 2001년부터 이어진 배율을 씀. "
               "2007년 중반 이후는 완전히 같음(상관 1.0, 처음 26주를 빼면 둘 다 0.896).")
    f = go.Figure()
    for j, (nm, r) in enumerate(series.items()):
        star, bse = nm.startswith("★"), nm.startswith("기준 · MinVar 70")
        f.add_trace(go.Scatter(x=r.index, y=r.fillna(0).cumsum(), name=nm,
                               line=dict(width=2.6 if star else 2.0 if bse else 1.1, dash="dash" if bse else None,
                                         color=PAL[2] if star else ("#e8e6e1" if DARK else "#0b0b0b") if bse else PAL[(j + 1) % 8])))
    st.plotly_chart(shade(fig_layout(f, 400, title="누적 수익 (%, 로그 합)", hovermode="x", legend=dict(orientation="h", y=-0.3, font=dict(size=10))), combo.index),
                    key="cu_cum")
    k1, k2 = st.columns(2)
    if len(sel) > 1:
        Cm = X.corr()
        Cm.index = Cm.columns = [s_[:28] for s_ in Cm.columns]
        k1.plotly_chart(heat(Cm, "고른 전략끼리 주간 수익률 상관 (낮을수록 섞는 효과 큼)", -1, 1, fmt=".2f", h=160 + 32 * len(Cm), nodiag=True), key="cu_corr")
    if method not in ("직접 입력", "동일 비중"):
        Wm = W.resample("QE").last() * 100
        fw = go.Figure([go.Scatter(x=Wm.index, y=Wm[c], name=c[:28], stackgroup="w", line=dict(width=0.5, color=PAL[j % 8])) for j, c in enumerate(Wm.columns)])
        k2.plotly_chart(fig_layout(fw, 160 + 32 * max(len(sel), 3), title="비중 추이 (%, 분기말)", hovermode="x"), key="cu_w_fig")
    else:
        wv = (w_user / w_user.sum() * 100) if method == "직접 입력" else pd.Series(100 / len(sel), sel)
        fw = go.Figure(go.Bar(x=wv.values, y=[s_[:28] for s_ in wv.index], orientation="h", marker_color=PAL[2], text=[f"{v:.0f}%" for v in wv.values],
                              textposition="outside"))
        k2.plotly_chart(fig_layout(fw, 160 + 32 * max(len(sel), 3), title="비중 (%, 고정)", hovermode="y"), key="cu_w_fig")
    if len(sel) > 1:
        w_eff = (w_user / w_user.sum()) if method == "직접 입력" else (pd.Series(1.0 / len(sel), sel) if method == "동일 비중" else W.mean())
        _frontier_section(X, w_eff, combo, float(tgt_v))
    _groupcap_section(X, W, method, w_user, float(tgt_v))
    st.caption("읽는 법: 각 재료는 이미 목표 변동성 10%로 맞춘 전략이라, 비중은 '위험을 얼마씩 나눠 주나'에 가까움. 기본값(MinVar 50 + 모멘텀 25 + 가치 25)이 롱숏 탭의 0.89 근처면 정상. "
               "대안 시계열·멀티모달 갈래는 단독 성과가 약하거나 기간이 짧아(2012-07~) 섞을 때 효과가 작음 — 상관이 낮으면 Sharpe가 약해도 혼합 낙폭을 줄일 수는 있음.")


GCAP = {"없음": "없음 (기준)", "곡물 25% (권장)": "곡물 25%", "3품목군 25% (곡물 · 에너지 · 축산)": "곡물 25% + 에너지 25% + 축산 25%"}
GCAP_RULE = {"없음": {}, "곡물 25% (권장)": {"곡물": 0.25}, "3품목군 25% (곡물 · 에너지 · 축산)": {"곡물": 0.25, "에너지": 0.25, "축산": 0.25}}


@st.cache_data(show_spinner=False)
def _itemw() -> dict:
    """전략별 품목 비중 (strategy_itemw.py → data/frozen/itemw_{FROZEN_END}.pkl). 없으면 빈 사전."""
    f = A.DATA / "frozen" / f"itemw_{FROZEN_END}.pkl"
    return pd.read_pickle(f) if f.exists() else {}


@st.cache_data(show_spinner=False)
def _gcap_mine(Ws: pd.DataFrame, tgt: float, roll: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """고른 전략 조합(전략 비중 Ws, _combine 결과)의 품목 순비중에 품목군 상한 3가지를 걸고 → 품목 손익(편도 5bp) → 목표 변동성 tgt로 다시 맞춤."""
    import ls_strategies as LS
    import strategy_groupcap as SG
    import strategy_itemw as SI
    Wi = _itemw()
    L = _levels_frozen(roll)
    Wit = SI.combine_items(Wi, Ws)
    R = LS._ret(LS.weekly(L, list(Wit.columns), "2001-01-01")).reindex(columns=Wit.columns)
    rows, wk = {}, {}
    for lab, caps in GCAP_RULE.items():
        W2 = Wit
        for g, c in caps.items():
            W2 = SG.cap_group(W2, SG.GROUPS[g], c)
        raw = LS._book(W2, R, 5.0)
        pv = raw.rolling(26, min_periods=13).std().shift(1) * np.sqrt(52)
        sc = (tgt / pv).clip(upper=4.0 * tgt / 10.0).fillna(tgt / 10.0)
        r = (raw * sc).reindex(Ws.index).dropna()
        r = r[r.index >= r.ne(0).idxmax()]
        cum = (1 + r / 100).cumprod(); g_ = [c for c in SG.GROUPS["곡물"] if c in W2]
        sh = lambda x: float(x.mean() / x.std() * np.sqrt(52)) if len(x) > 26 and x.std() > 0 else np.nan  # noqa: E731
        try:
            vb = SG.var_by_group(W2.iloc[-1], L, float(sc.iloc[-1]))
        except Exception:
            vb = {"지금 20일 99% VaR (%)": np.nan, "VaR 기여 곡물 (%)": np.nan}
        rows[lab] = {"연수익 (%)": r.mean() * 52, "연변동성 (%)": r.std() * np.sqrt(52), "Sharpe 전체": sh(r),
                     "Sharpe 학습 ~2016": sh(r[r.index < "2017-01-01"]), "Sharpe 시험 2017–": sh(r[r.index >= "2017-01-01"]),
                     "최대 낙폭 (%)": float((cum / cum.cummax() - 1).min() * 100), "최악 1년 (%)": float(r.groupby(r.index.year).sum().min()),
                     "지금 20일 99% VaR (%)": vb["지금 20일 99% VaR (%)"], "VaR 중 곡물 (%)": vb["VaR 기여 곡물 (%)"],
                     "곡물 몫 평균 (%)": float((W2[g_].abs().sum(1) / W2.abs().sum(1).replace(0, np.nan)).reindex(r.index).mean() * 100)}
        wk[lab] = r
    return pd.DataFrame(rows).T, pd.DataFrame(wk)


def _groupcap_section(X: pd.DataFrame | None = None, Ws: pd.DataFrame | None = None, method: str = "", w_user=None, tgt: float = 10.0):
    """⑤ Portfolio Customization › 품목군 위험 한도.
    고른 전략이 모두 품목 비중을 가진 수치 전략이면 → 그 조합에 바로 적용(v1.65.0, strategy_itemw.py),
    아니면 → 기본 혼합(MinVar 50 · 모멘텀 25 · 가치 25) 고정 결과(strategy_groupcap.py)."""
    st.markdown("##### 품목군 위험 한도 — 곡물 · 에너지 · 축산 합계 비중 상한")
    pick = st.radio("품목군 상한", list(GCAP), index=1, horizontal=True, key="cu_gcap",
                    help="품목별 순비중에서 품목군 비중(절댓값 합)이 전체 노출의 25%를 넘으면 그 품목군만 비율대로 줄이고, 섞은 뒤 목표 변동성으로 다시 맞춤. "
                         "곡물(옥수수 · 밀 · 대두 · 대두박 · 대두유) · 에너지(WTI · 난방유 · 휘발유 · 천연가스) · 축산(생우 · 돈육)")
    Wi = _itemw()
    miss = [c for c in (X.columns if X is not None else []) if c not in Wi]
    live = X is not None and Ws is not None and len(Wi) > 0 and not miss
    if live:
        with st.spinner("고른 조합의 품목 비중에 상한을 거는 중… (조합마다 처음 한 번 10–30초)"):
            T, Wk = _gcap_mine(Ws, float(tgt), ROLL)
        st.caption(f"✅ **내 포트폴리오에 바로 적용** — 위에서 고른 전략 {X.shape[1]}개 · 비중 방식 '{method}' · 목표 변동성 {tgt:g}%. 전략별 품목 비중(저장본)을 같은 비중으로 합친 품목 순비중에 "
                   "상한을 걸고 품목 단위로 다시 계산. '없음' 행도 품목 단위 재계산이라 위 표의 ★와 조금 다를 수 있음(전략끼리 반대 매매가 상쇄돼 거래비용이 줄어드는 몫).")
    else:
        fw, fc = A.DATA / "strategy_groupcap_weekly.parquet", A.DATA / "strategy_groupcap_compare.csv"
        if not (fw.exists() and fc.exists()):
            st.info("결과가 없습니다 — `python strategy_groupcap.py`를 먼저 실행하세요."); return
        Wk0, C = pd.read_parquet(fw), pd.read_csv(fc, encoding="utf-8-sig").set_index("상한")
        why = (f"품목 비중이 저장되지 않은 전략이 섞여 있음: {', '.join(m[:30] for m in miss[:4])}{' 외' if len(miss) > 4 else ''} "
               "(대안 시계열 · 멀티모달 · 수익률끼리 섞은 혼합 전략은 품목 비중이 없음)") if miss else ("전략별 품목 비중 파일이 없음 — `python strategy_itemw.py` 실행" if not Wi else "")
        st.caption(f"ℹ️ 기본 혼합(MinVar 50 · 모멘텀 25 · 가치 25, 목표 10%) 고정 결과를 보여 줌. {why}")
        rows, Wk = {}, {}
        for lab, col in GCAP.items():
            r = Wk0[col].loc["2008":].fillna(0); cum = (1 + r / 100).cumprod(); c_ = C.loc[col]; Wk[lab] = r
            rows[lab] = {"연수익 (%)": r.mean() * 52, "연변동성 (%)": r.std() * np.sqrt(52), "Sharpe 전체": c_["Sharpe 2008–"],
                         "Sharpe 학습 ~2016": c_["Sharpe 학습 2008–16"], "Sharpe 시험 2017–": c_["Sharpe 시험 2017–"],
                         "최대 낙폭 (%)": float((cum / cum.cummax() - 1).min() * 100), "최악 1년 (%)": float(r.groupby(r.index.year).sum().min()),
                         "지금 20일 99% VaR (%)": c_["지금 20일 99% VaR (%)"], "VaR 중 곡물 (%)": c_["VaR 기여 곡물 (%)"], "곡물 몫 평균 (%)": c_["곡물 몫 평균 (%)"]}
        T, Wk = pd.DataFrame(rows).T, pd.DataFrame(Wk)
    sel_, b_ = T.loc[pick], T.loc["없음"]
    m = st.columns(4)
    for k_, (lab, fmt) in enumerate([("Sharpe 시험 2017–", "{:.2f}"), ("최대 낙폭 (%)", "{:.1f}%"), ("지금 20일 99% VaR (%)", "{:.2f}%"), ("VaR 중 곡물 (%)", "{:.0f}%")]):
        v_ = sel_[lab]
        m[k_].metric(lab, "–" if pd.isna(v_) else fmt.format(v_), None if pick == "없음" or pd.isna(v_) else f"{v_ - b_[lab]:+.2f}",
                     delta_color="inverse" if "VaR" in lab else "normal")
    st.dataframe(T.style.format("{:.2f}", na_rep="–").apply(lambda s_: ["font-weight:600; background-color: rgba(46,125,82,0.12)" if s_.name == pick else "" for _ in s_], axis=1),
                 use_container_width=True)
    f = go.Figure()
    for j, lab in enumerate(GCAP):
        r = Wk[lab].fillna(0)
        f.add_trace(go.Scatter(x=r.index, y=r.cumsum(), name=lab, line=dict(width=2.6 if lab == pick else 1.2, color=[PAL[0], PAL[2], PAL[4]][j])))
    st.plotly_chart(fig_layout(f, 320, title="누적 수익 (%, 로그 합) — 품목군 상한별" + (" · 내 포트폴리오" if live else " · 기본 혼합"), hovermode="x",
                               legend=dict(orientation="h", y=-0.25)), key="cu_gcap_cum")
    st.caption("기본 혼합 고정 결과(2026-10-26 자료, `strategy_groupcap.py`): 곡물 25%는 수익을 거의 그대로 두고(시험 1.07 → 1.09, 차이 t 0.6 — 유의하지 않음) 지금 VaR 2.87 → 2.25%, "
               "VaR 중 곡물 몫 62 → 34% — 스트레스 테스트에서 곡물이 위험의 절반이던 쏠림을 줄이는 위험 한도. 3품목군 25%는 낙폭이 −24.5 → −20.7%로 더 줄지만 시험 Sharpe 1.06. "
               "축산만 15%로 강하게 막으면 오히려 손해(0.97) — 생우 · 돈육이 다른 원자재와 덜 엮여 분산 역할을 하기 때문. 다른 조합에서는 효과가 다를 수 있으니 위 표로 확인. "
               "상세: ⑥ 실험 결과 → '3축 신호 · 위험 한도 실험'.")


def _merged_table(PV: pd.DataFrame, fmt: str = "{:.1f}"):
    """(전략, 지표) 2단 행 표를 전략 이름 칸을 세로로 합친(rowspan) HTML 표로 그림 — st.dataframe은 같은 이름을 줄마다 반복하므로."""
    bd = "rgba(255,255,255,0.12)" if DARK else "rgba(11,11,11,0.10)"
    hd = "rgba(255,255,255,0.05)" if DARK else "rgba(11,11,11,0.035)"
    sty = (PV.style.format(fmt, na_rep="–")
           .set_table_styles([{"selector": "", "props": f"border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums;"},
                              {"selector": "th, td", "props": f"border:1px solid {bd};padding:4px 8px;"},
                              {"selector": "thead th", "props": f"background:{hd};font-weight:500;text-align:center;"},
                              {"selector": "th.row_heading.level0", "props": "font-weight:500;text-align:left;vertical-align:middle;min-width:220px;"},
                              {"selector": "th.row_heading.level1", "props": "font-weight:400;text-align:left;white-space:nowrap;"},
                              {"selector": "td", "props": "text-align:right;"}]))
    st.markdown(f'<div style="overflow-x:auto">{sty.to_html()}</div>', unsafe_allow_html=True)


def _voltgt_section():
    """⑤ 포트폴리오 › 목표변동성에 따른 수익율 변화 Backtest — 롱숏 상위 10개 전략을 목표 변동성 5–50% × 레버리지 상한 2·4·8배·없음으로 다시 계산한 고정 결과."""
    st.subheader("목표변동성에 따른 수익율 변화 Backtest — 같은 전략, 레버리지만 바꾸면?")
    st.caption(f"📌 결과 고정: {PLS_MD} 자료 기준 (`ls_extra.run_extra`에서 목표 변동성·레버리지 상한만 바꿔 미리 계산). 데이터 업데이트를 해도 다시 계산하지 않습니다.")
    f_ = A.DATA / "voltgt_compare.csv"
    if not f_.exists():
        st.info("결과가 없습니다 — 목표 변동성 백테스트를 먼저 실행해야 합니다.")
        return
    VT = pd.read_csv(f_, encoding="utf-8-sig")
    if "레버리지 상한" not in VT:
        VT["레버리지 상한"] = "4배"
    st.caption("공통 틀: 롱숏 탭과 같은 19개 품목 · 주간 결정 · 편도 5bp. 품목 비중에 '목표 ÷ 직전 26주 실현 변동성' 배율을 곱해 목표 수준에 맞추고(배율 상한까지), 혼합은 섞은 뒤 다시 맞춤(혼합 배율 상한 4배). "
               "기본 = 목표 10% · 상한 4배(롱숏 탭 표와 같음). 2007-01 ~ 2026-10 평가.")
    st.caption("무엇이 손익에 들어갔나: ✔ 변동성이 낮을수록 레버리지를 키움(배율 = 목표 ÷ 직전 26주 실현 변동성 — 변동성이 1/3이면 3배, 분산이 1/3이면 √3≈1.7배) · ✔ 레버리지 배율 상한(선택값) · ✔ 거래비용 편도 5bp. ✖ 차입이자 없음(선물은 명목 전액을 빌리지 않음) · ✖ 증거금에 붙는 이자 수익도 더하지 않음(보수적) · ✖ 거래소 증거금률은 손익 계산에 안 들어감 — 그 레버리지에 자본이 얼마나 묶이는지 보여 주는 표시용 가정.")
    k1, k2, k3 = st.columns([1.1, 1.6, 1])
    caps = [c for c in ["2배", "4배", "8배", "없음"] if c in set(VT["레버리지 상한"])]
    cap = k1.radio("레버리지 상한 (변동성 맞춤 배율의 최대치)", caps, index=caps.index("4배") if "4배" in caps else 0, horizontal=True, key="vt_cap",
                   help="원래 비중에 곱하는 배율(목표 ÷ 직전 26주 실현 변동성)의 최대치. 상한이 낮으면 목표를 올려도 실제 변동성이 따라가지 않음(특히 52주 모멘텀). '없음' = 목표를 맞출 때까지 배율을 키움. 실제 명목 노출은 '총 명목 노출' 지표로 확인")
    mets = ["연수익 산술 (%)", "연수익 복리 (%)", "Sharpe", "실현 변동성 (%)", "최대 낙폭 (%)", "최악 1년 (%)", "최악 1주 (%)", "총 명목 노출 평균 (배)", "필요 증거금 (자본 대비 %)", "재료 상한 걸린 주 (%)", "혼합 4배 상한 걸린 주 (%)"]
    mets = [m_ for m_ in mets if m_ in VT or m_.startswith("필요 증거금")]
    met = k2.selectbox("그래프 지표 (아래 선 그래프)", mets, key="vt_met")
    mrate = k3.slider("거래소 증거금률 가정 (명목 대비 %)", 3, 20, 8, key="vt_mr",
                      help="원자재 선물 개시 증거금은 대략 명목의 5–12%(품목·변동성에 따라 다름). 필요 증거금 = 총 명목 노출 × 증거금률")
    D = VT[VT["레버리지 상한"] == cap].copy()
    D["필요 증거금 (자본 대비 %)"] = D["총 명목 노출 평균 (배)"] * mrate
    order = list(D["전략"].drop_duplicates())
    short = {s_: s_.replace(" (사후 선택 주의)", "").replace("혼합 · ", "").replace("기준 · ", "") for s_ in order}
    rows3 = [("연수익 산술 (%)", "① 연수익 (%)"), ("실현 변동성 (%)", "② 연 변동성 (%)"), ("최대 낙폭 (%)", "③ 최대 낙폭 MDD (%)")]
    if "재료 상한 걸린 주 (%)" in D:
        rows3.append(("재료 상한 걸린 주 (%)", "④ 상한 걸린 주 (%)"))
    parts = []
    for col_, lab_ in rows3:
        pv_ = D.pivot(index="전략", columns="목표 변동성 (%)", values=col_).reindex(order)
        pv_.index = pd.MultiIndex.from_tuples([(short[s_], lab_) for s_ in pv_.index], names=["전략", "지표"])
        parts.append(pv_)
    PV = pd.concat(parts)
    PV = PV.reindex(pd.MultiIndex.from_tuples([(short[s_], lab_) for s_ in order for _, lab_ in rows3], names=["전략", "지표"]))
    PV.columns = [f"목표 {c}%" for c in PV.columns]
    st.markdown(f"##### 전략별 연수익 · 연 변동성 · 최대 낙폭 · 상한 걸린 주 (레버리지 배율 상한 {cap})")
    _merged_table(PV, "{:.1f}")
    st.caption("행마다 3줄: ① 산술 연수익(주간 수익률 평균 × 52) · ② 실제로 나온 연 변동성(목표와 다르면 배율 상한에 걸린 것) · ③ 최대 낙폭(고점 대비 최대 손실, 복리 자본 기준). "
               "같은 전략이면 ①·②·③이 목표에 거의 비례해 함께 커짐 — 수익만 커지는 게 아님. "
               "④ = 2007– 주 중 재료 전략의 변동성 맞춤 배율(목표 ÷ 직전 26주 실현 변동성)이 상한에 닿은 주의 비율(혼합은 재료 비중으로 가중). "
               "④가 크면 목표를 올려도 ①·②가 덜 늘고, Portfolio Customization(10% 결과를 '목표 ÷ 10'배로 키움, 재료 상한 없음)과 결과가 갈림 — 예: 52주 모멘텀은 목표 15%에서 이미 63%. "
               "섞은 뒤 다시 맞추는 혼합 배율(4배 상한)은 목표 50%에서도 거의 걸리지 않음(0–3%, 지표 '혼합 4배 상한 걸린 주'). 다전략은 비중을 재구성하지 않아 '–'. "
               "복리 수익·Sharpe·노출·증거금은 아래 '지표'에서 선택.")
    if met.startswith("총 명목") or met.startswith("필요 증거금"):
        st.caption("총 명목 노출 = 롱·숏 비중 절댓값의 합(자본 대비 배수), 2007– 주간 평균. 다전략(K2 페어·COT 포함)은 비중을 재구성하지 않아 '–'. "
                   f"필요 증거금 = 노출 × 증거금률 {mrate}% — 100%를 넘으면 그 계좌로는 운용 불가, 50%를 넘으면 손실 시 마진콜 여유가 거의 없음. 급등락 때 거래소가 증거금률을 올리면 더 필요.")
    f = go.Figure()
    for j, s_ in enumerate(order):
        d_ = D[D["전략"] == s_].sort_values("목표 변동성 (%)")
        f.add_trace(go.Scatter(x=d_["목표 변동성 (%)"], y=d_[met], name=short[s_], mode="lines+markers",
                               line=dict(width=2.6 if "가치 25" in s_ else 1.4, color=PAL[j % 8], dash="dot" if j >= 8 else None), marker=dict(size=5)))
    if met in ("실현 변동성 (%)",):
        f.add_trace(go.Scatter(x=[5, 50], y=[5, 50], name="목표 = 실현 (45°)", mode="lines", line=dict(color="#898781", dash="dash", width=1)))
    st.plotly_chart(fig_layout(f, 400, title=f"{met} — 목표 변동성에 따라 (상한 {cap})", hovermode="x unified", xaxis=dict(title="목표 변동성 (%)", dtick=5),
                               legend=dict(orientation="h", y=-0.25, font=dict(size=10))), key="vt_line")
    fw = A.DATA / "voltgt_weekly.parquet"
    if fw.exists():
        W_ = pd.read_parquet(fw)
        c1, c2, c3 = st.columns([1.6, 1.6, 0.8])
        s_pick = c1.selectbox("누적 수익 — 전략", order, format_func=lambda x: short[x], key="vt_s")
        t_pick = c2.multiselect("목표 변동성", sorted(VT["목표 변동성 (%)"].unique()), [10, 20, 30, 50], key="vt_t")
        logy = c3.checkbox("로그 축", True, key="vt_log", help="변동성이 큰 경우 낙폭 구간을 같은 비율로 비교하기 쉬움")
        capn = {"2배": 2, "4배": 4, "8배": 8, "없음": 1000}[cap]
        f2 = go.Figure()
        for j, t_ in enumerate(t_pick):
            col = f"{s_pick}|{t_}|{capn}"
            if col not in W_:
                continue
            eq = (1 + W_[col].fillna(0) / 100).clip(lower=1e-6).cumprod()
            f2.add_trace(go.Scatter(x=eq.index, y=eq.values, name=f"목표 {t_}%", line=dict(width=2.2 if t_ == 10 else 1.4, color=PAL[j % 8])))
        st.plotly_chart(shade(fig_layout(f2, 400, title=f"자본 1 → 누적 (복리, {short[s_pick]}, 상한 {cap})", hovermode="x",
                                         yaxis=dict(type="log" if logy else "linear", title="자본 (배)")), W_.index), key="vt_cum")
    st.markdown("""
**읽는 법**
- **산술 연수익**은 목표에 거의 비례(레버리지만 키운 것이라 Sharpe는 그대로). 하지만 **복리 연수익**(실제 자본이 불어나는 속도)은 변동성²/2만큼 덜 늘어남 — 가치 혼합은 목표 50%에서 산술 45%가 복리 36% 정도.
- **최대 낙폭·최악 1년**이 목표에 거의 비례해 커짐 → 목표 30% 이상은 −60% 이상 손실 구간을 견뎌야 함.
- **레버리지 상한**에 걸리는 전략(52주 모멘텀 · 단독 전략들)은 목표를 올려도 실현 변동성과 수익이 멈춤 → '실현 변동성' 보기에서 45° 선 아래로 벗어나는 지점이 상한에 걸린 곳.
- **필요 증거금**이 크면 수익률 표가 좋아 보여도 실제 계좌로는 운용 불가. 표는 과거 19년 결과이며 미래 낙폭을 보장하지 않음.
- 모두 연속월물 기준이라 롤 비용 과대평가(ETF 기준 연 약 −3%p, ⑧ FAQ ⑦)도 레버리지만큼 커짐.
""")


def page_portfolio():
    a, b, v, c, d = subtabs(["Portfolio Customization", f"롱숏 전략 (Alpha) Backtest ({PLS_S})", f"목표변동성별 수익율 Backtest ({PLS_S})",
                             f"대안 시계열 모델 ({PALT_S})", f"멀티모달 (뉴스·리포트·기상) ({PMM_S})"])
    with a:
        tab_custom()
    with b:
        tab_ls()
    with v:
        _voltgt_section()
    with c:
        _alt_ts_section()
    with d:
        _multimodal_section()


@st.cache_data(show_spinner=False)
def _cs_fp(ids: tuple, roll: bool):
    import crosssec as CS
    return CS.feature_panel(_levels(None, roll), list(ids))


@st.cache_data(show_spinner=False)
def _cs_bt(ids: tuple, feats: tuple, method: str, roll: bool):
    import crosssec as CS
    BP = CS.buy_panel(_levels(None, True), _levels(None, False), list(ids))
    return CS.backtest_cluster1(_cs_fp(ids, roll), BP, list(feats), method)


def _norm100(v: pd.Series) -> pd.Series:
    lo, hi = float(v.min()), float(v.max())
    return (v - lo) / (hi - lo) * 100 if hi > lo else v * 0 + 50


@st.cache_data(show_spinner=False)
def _cs_buy(ids: tuple, roll: bool) -> pd.DataFrame:
    """★ 구매 탭과 같은 방식(추천 프리셋 가중, 위험 = EWMA)으로 계산한 최신 가격 매력 점수·구성요소 (0–100)."""
    import procurement as PR
    La_ = _levels(None, True)
    P0 = PR.PRESETS["추천 (고도화: 상대가치+계절+MIDAS)"]
    rows = {}
    for i in ids:
        if i not in PR.BUY_IDS or i not in La_:
            continue
        try:
            inp = _proc_inputs(i, "ewma", "price", roll, None)
            sig = PR.signals(La_[i], PR.item_lookback(i, int(P0["lookback"])), inp["rel"], inp["rel_sign"], inp["vol_fc"], inp["crisis"], P0["season"])
            sc = PR.score(sig, P0["w"])
            last = sig.dropna(subset=["value"]).index.max()
            r = {c: float(sig[c].loc[last]) * 100 if c in sig and np.isfinite(sig[c].loc[last]) else np.nan for c in ["value", "turn", "relval", "season"]}
            r["score"] = float(sc.loc[last]) * 100
            rows[i] = r
        except Exception:
            continue
    return pd.DataFrame(rows).T


def tab_cross():
    import crosssec as CS
    import ls_strategies as LS
    st.subheader("횡단면 분석 — 지금 이 시점, 품목들을 나란히 놓고 '비슷한 상태'끼리 묶기")
    st.caption("시계열이 '한 품목의 시간에 따른 변화'라면, 횡단면은 '같은 시점에 품목끼리 비교'. 구매 관점에서는 '지금 가격이 매력적인 상태인 품목 묶음'을 찾는 용도 — "
               "★ 구매 탭의 가격 매력 점수·구성요소와 가격 흐름·위험 특성을 함께 넣어 군집.")
    c1_, c2_ = st.columns([3, 2])
    opts = [i for i in COMMOD_IDS if i.startswith("yf:")]
    ids = c1_.multiselect("품목", opts, [i for i in LS.BASKET if i in opts], format_func=lbl, key="cs_ids")
    fset = c2_.radio("특성 묶음", list(CS.SETS), 0, key="cs_fset",
                     help="구매 매력도 중심 = ★ 구매 탭의 가격 매력 점수·가격 위치·단기 반등·상대가치·계절성 + 13·52주 수익률·고점 대비·범위 내 위치·변동성·추세")
    if len(ids) < 6:
        st.info("품목을 6개 이상 고르세요.")
        return
    fp = _cs_fp(tuple(ids), ROLL)
    X0, dt = CS.snapshot(fp)
    B = _cs_buy(tuple(ids), ROLL)
    X0 = X0.join(B, how="left")
    all_f = list(CS.BUY_FEATS) + list(CS.FEATS)
    feats = [f_ for f_ in (CS.SETS[fset] or all_f) if f_ in X0 and X0[f_].notna().sum() >= 6]
    X = X0[feats]
    X = X.fillna(X.median())                                 # 상대가치가 없는 품목 등은 중앙값(중립)으로
    last_d = min(dt, _levels(None, ROLL)[ids].dropna(how="all").index.max())
    st.caption(f"기준: {last_d:%Y-%m-%d} 주 · 품목 {len(X)}개 · 특성 {len(feats)}개 ({', '.join(CS.label_of(f_) for f_ in feats)}) — "
               "값이 없는 특성(예: 상대가치 관계가 없는 품목)은 품목 중앙값으로 채움")

    st.markdown("##### 군집 — 지금 '비슷한 상태'인 품목끼리 묶기")
    st.caption("원리: 특성마다 품목 간 평균 0·표준편차 1로 맞춘 뒤(단위 차이 제거), 특성 공간에서 서로 가까운 품목끼리 묶음. "
               "군집 1 = 가격 매력 점수 평균이 가장 높은 묶음(지금 상대적으로 싼 상태) → 여러 품목 중 어디부터 살지, 같은 상태라 한꺼번에 움직일 품목은 무엇인지 판단에 사용.")
    g1, g2 = st.columns([1, 1])
    meth = g1.radio("방법", ["K-평균", "계층 (Ward)", "가우시안 혼합 (GMM)"], horizontal=True, key="cs_meth",
                    help="K-평균 = 군집 중심까지 거리 합 최소 · 계층(Ward) = 합칠 때 분산 증가가 가장 작은 쌍부터 차례로 합침 · GMM = 군집마다 정규분포를 가정한 확률적 묶음")
    kk = g2.select_slider("군집 수 K", ["자동 (실루엣 최대)", 2, 3, 4, 5, 6], "자동 (실루엣 최대)", key="cs_k")
    cl = CS.cluster(X, meth, None if isinstance(kk, str) else int(kk))
    labs = cl["labels"]
    ex = CS.explain(X, labs)
    COLS = [PAL[2], PAL[0], PAL[3], PAL[1], PAL[4], PAL[6]]
    proj = st.radio("지도 축", ["LDA (군집이 가장 잘 갈리는 방향)", "주성분 PCA (특성 전체 변동을 요약)"], horizontal=True, key="cs_proj")
    if proj.startswith("LDA") and "lda_xy" in ex:
        xy = ex["lda_xy"]; ax_t = ("LD1", "LD2"); ttl = "LDA 지도 — 선형판별로 군집 간 차이가 가장 크게 보이는 두 축"
    else:
        xy = cl["xy"]; ax_t = (f"PC1 ({cl['evr'][0]:.0%})", f"PC2 ({cl['evr'][1]:.0%})"); ttl = f"주성분 지도 — 특성 {len(feats)}개를 2차원으로 요약"
    nx, ny = _norm100(xy.iloc[:, 0]), _norm100(xy.iloc[:, 1])
    f = go.Figure()
    for k_ in sorted(labs.unique()):
        m_ = labs == k_
        sc_ = X.loc[m_, "score"].mean() if "score" in X else np.nan
        f.add_trace(go.Scatter(x=nx[m_], y=ny[m_], mode="markers", marker=dict(size=15, color=COLS[(k_ - 1) % 6]),
                               name=f"군집 {k_}" + (f" (매력 점수 평균 {sc_:.0f})" if np.isfinite(sc_) else ""),
                               text=[short(i) for i in labs.index[m_]], hovertemplate="%{text}<extra></extra>"))
    for (dx, dy), i in zip(_place_labels(nx, ny, [short(i) for i in xy.index]), xy.index):
        f.add_annotation(x=nx[i], y=ny[i], text=short(i), ax=dx, ay=dy, showarrow=True, arrowwidth=0.8, arrowhead=0, arrowcolor="#898781",
                         standoff=6, font=dict(size=11))
    st.plotly_chart(fig_layout(f, 430, title=f"{ttl} · K = {cl['k']}", xaxis=dict(title=ax_t[0], showticklabels=False),
                               yaxis=dict(title=ax_t[1], showticklabels=False), hovermode="closest"), key="cs_pca")
    prof = cl["profile"].copy()
    prof.columns = [CS.label_of(c) for c in prof.columns]
    prof.insert(0, "품목", [", ".join(short(i) for i in labs.index[labs == k_]) for k_ in prof.index])
    prof.index = [f"군집 {k_}" for k_ in prof.index]
    st.dataframe(prof.style.format({c: "{:.1f}" for c in prof.columns if c != "품목"}))
    if "score" in X:
        best = prof.index[0]
        st.info(f"**{best}**이 가격 매력 점수 평균이 가장 높은 묶음 ({prof.iloc[0]['가격 매력 점수']:.0f}점): {prof.iloc[0]['품목']} — "
                "지금 품목들 가운데 상대적으로 싼 상태. 같은 군집 품목은 비슷한 이유로 싸진 것이므로 한꺼번에 살 때 분산 효과는 아래 '현재 상관'으로 확인.")

    st.markdown("##### 군집을 가르는 특성 — 분류 모델로 경계 설명")
    st.caption("군집 결과를 '정답'으로 두고 분류 모델을 맞춰, 무엇이 품목을 각 군집으로 보내는지 읽음 (예측용이 아님). "
               "로지스틱 계수 = 그 특성이 1표준편차 높을 때 해당 군집에 속할 가능성이 커지는(+) / 작아지는(−) 정도 · "
               "LDA = 위 지도의 축 · SVM(RBF) = 비선형 경계로 하나씩 빼고 맞혀 본 정확도(높을수록 군집이 뚜렷).")
    e1, e2 = st.columns([2, 1])
    if "coef" in ex:
        cf = ex["coef"].copy(); cf.columns = [CS.label_of(c) for c in cf.columns]
        fz = heat(cf, "로지스틱 회귀 계수 (군집 × 특성, + = 그 군집 쪽으로)", fmt=".1f", h=150 + 40 * len(cf))
        fz.update_xaxes(tickangle=-40, tickfont=dict(size=10))
        e1.plotly_chart(fz, key="cs_coef")
    sil = cl["silhouette"]
    e2.metric("SVM (RBF) 경계 정확도 (하나씩 빼고 검증)", f"{ex.get('svm_acc', np.nan):.0%}" if np.isfinite(ex.get("svm_acc", np.nan)) else "–")
    e2.metric(f"실루엣 점수 (K = {cl['k']})", f"{sil.get(cl['k'], np.nan):.2f}" if len(sil) else "–", "0.25↑ 보통 · 0.5↑ 뚜렷", delta_color="off")

    st.markdown("##### 현재 상관 — 지금 같이 움직이는 품목 (지수가중, 반감기 8주)")
    C = CS.ewma_corr(fp["R"][X.index], halflife=8)
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    Dm = np.sqrt(np.clip(2 * (1 - C.values), 0, None)); np.fill_diagonal(Dm, 0)
    order = dendrogram(linkage(squareform(Dm, checks=False), "average"), no_plot=True, labels=list(C.index))["ivl"]
    Co = C.loc[order, order]
    fh = heat(Co, "오늘 기준 상관 (최근 주일수록 큰 가중 · 같이 움직이는 품목끼리 정렬)", *range_of(Co), fmt=".2f", h=140 + 28 * len(Co), nodiag=True)
    fh.update_traces(textfont=dict(size=9))
    st.plotly_chart(fh, key="cs_corr")
    st.caption("상관은 한 시점 값만으로는 계산할 수 없어, 최근 2년 주간 수익률에 '최근일수록 큰 가중'(반감기 8주 → 8주 전 자료는 절반, 16주 전은 1/4)을 줘서 오늘 시점의 상관을 추정. "
               "같은 블록(진한 칸)끼리는 분산 효과가 작음 → 군집 1 품목을 여러 개 살 때 블록이 다른 품목을 섞을수록 위험이 줄어듦.")

    st.markdown("##### 배지 검증 — 과거 매달 '군집 1'을 샀다면 다음 13주 가격은?")
    st.caption("2006년부터 4주마다, 그 시점까지의 자료로 같은 특성·방법으로 다시 군집 → 군집 1(매력 점수 평균 최고) 품목들의 다음 13주 가격 변화를 전체 평균과 비교. "
               "구매자 관점: 군집 1이 이후 더 올랐다면 '그때 싼 상태'라는 판단이 맞았던 것 (지금 사는 게 유리했음). t값은 13주 목표가 4주 간격으로 겹치는 것을 보정.")
    if st.button("▶ 배지 검증 백테스트 (약 1–2분)", key="cs_bt_run"):
        st.session_state["cs_bt"] = (tuple(ids), tuple(feats), meth)
    if st.session_state.get("cs_bt") == (tuple(ids), tuple(feats), meth):
        with st.spinner("과거 시점마다 군집 중…"):
            BT = _cs_bt(tuple(ids), tuple(feats), meth, ROLL)
        if BT:
            st.dataframe(BT["table"].style.format({"군집 1 다음 13주 평균 (%)": "{:+.2f}", "전체 평균 (%)": "{:+.2f}", "차이 평균 (%p)": "{:+.2f}",
                                                   "차이 t값 (겹침 보정)": "{:+.2f}", "군집 1이 더 오른 시점 비율": "{:.0%}"}))
            dd_ = BT["detail"]["차이 (군집1 − 전체)"]
            f = go.Figure(go.Bar(x=dd_.index, y=dd_.values, marker_color=[PAL[2] if v > 0 else PAL[1] for v in dd_.values], name="차이"))
            f.add_trace(go.Scatter(x=dd_.index, y=dd_.rolling(13, min_periods=4).mean(), line=dict(color="#898781", width=2), name="1년 이동평균"))
            st.plotly_chart(shade(fig_layout(f, 300, title="시점별: 군집 1 − 전체 평균, 다음 13주 가격 변화 (%p) — 초록 = 군집 1이 더 오름(그때 산 게 유리)",
                                             hovermode="x"), dd_.index), key="cs_bt_fig")
            tb = BT["table"].loc["전체"]
            tv = tb["차이 t값 (겹침 보정)"]
            st.info(f"평균 {tb['차이 평균 (%p)']:+.2f}%p · t {tv:+.2f} · 더 오른 시점 {tb['군집 1이 더 오른 시점 비율']:.0%} → "
                    + ("방향은 맞지만(군집 1이 이후 더 오르는 경향) 통계적으로 확실하지는 않음 — 배지는 '참고 신호'로." if 0 < tv < 2 else
                       "통계적으로 유의 (t > 2)." if tv >= 2 else "군집 1이 이후 오히려 덜 오름 — 배지의 근거 약함."))


def page_procure():
    a, b = subtabs(["구매 타이밍 — 지금 살까, 미룰까", "횡단면 — 품목 비교 · 매력 묶음 (군집)"])
    with a:
        tab_procure()
    with b:
        tab_cross()


@st.cache_data(show_spinner=False)
def _cs_cluster1(roll: bool):
    """★ 카드 배지용: 기본 19개 품목 · '구매 매력도 중심' 특성 · K-평균(실루엣 자동)으로 군집 → 군집 1(매력 점수 평균 최고) 품목."""
    import crosssec as CS
    import ls_strategies as LS
    ids = tuple(i for i in LS.BASKET if i in NAME)
    X0, _ = CS.snapshot(_cs_fp(ids, roll))
    X0 = X0.join(_cs_buy(ids, roll), how="left")
    feats = [f_ for f_ in CS.SETS["구매 매력도 중심 (권장)"] if f_ in X0 and X0[f_].notna().sum() >= 6]
    X = X0[feats].fillna(X0[feats].median())
    cl = CS.cluster(X, "K-평균")
    lab = cl["labels"]
    g1 = list(lab.index[lab == 1])
    C = CS.ewma_corr(_cs_fp(ids, roll)["R"][g1], halflife=8) if len(g1) >= 2 else pd.DataFrame()
    partners = {i: C[i].drop(i).sort_values().head(2) for i in C.columns}           # 같은 묶음 안에서 상관이 가장 낮은 2개
    return set(g1), float(X.loc[lab == 1, "score"].mean()) if "score" in X else np.nan, int(cl["k"]), len(X), partners


def page_settings():
    st.subheader("⑨ 설정 — 화면 배경색")
    sv = _ui_saved()
    names = list(UI_BG)
    sw = "".join(f'<span title="{n}" style="display:inline-block;width:26px;height:26px;margin:2px;border-radius:5px;border:1px solid rgba(128,128,128,0.45);'
                 f'background:{v}"></span>' for n, v in UI_BG.items() if v and v != "custom")
    st.markdown(f"<div style='margin-bottom:6px'>{sw}</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    nm = c1.selectbox("본문 배경색", names, names.index(sv["bg"]) if sv["bg"] in names else 0, key="ui_bg",
                      help="'기본'이면 Streamlit 테마(라이트·다크) 색을 그대로 씀. 다크 계열을 라이트 테마에서 고르면 글자색을 자동으로 밝게 바꿈")
    c2.color_picker("직접 고르기 색", sv["bg_custom"], key="ui_bg_custom", disabled=nm != "직접 고르기")
    d1, d2 = st.columns([2, 1])
    sbo = ["본문보다 조금 진하게", "본문과 같게", "직접 고르기"]
    sbn = d1.selectbox("사이드바 배경색", sbo, sbo.index(sv["sb"]) if sv["sb"] in sbo else 0, key="ui_sb", disabled=UI_BG.get(nm) is None)
    d2.color_picker("사이드바 직접 고르기 색", sv["sb_custom"], key="ui_sb_custom", disabled=sbn != "직접 고르기" or UI_BG.get(nm) is None)
    st.checkbox("차트 바탕도 같은 배경색으로 (차트를 투명하게)", sv["chart"], key="ui_chart", disabled=UI_BG.get(nm) is None)
    if st.button("기본값으로 되돌리기", key="ui_reset"):
        _keep_forget("ui_bg", "ui_bg_custom", "ui_sb", "ui_sb_custom", "ui_chart")
        try:
            UI_FILE.unlink()
        except Exception:
            pass
        st.rerun()
    st.caption("고르면 바로 적용되고 `data/ui_settings.json`에 저장돼 앱을 다시 켜도 유지됨. 라이트·다크 테마 자체는 오른쪽 위 ⋮ › Settings에서 바꿈. "
               "표(st.dataframe)는 Streamlit이 캔버스로 그려 테마 색을 따르고, 차트의 선·글자 색은 테마(라이트·다크) 기준으로 정해짐 — "
               "다크 배경을 라이트 테마에서 쓰면 일부 차트 글자가 어둡게 보일 수 있으니, 다크 계열은 다크 테마와 함께 쓰는 것을 권장.")


def page_atlas():
    try:
        import model_atlas
        model_atlas.render()
    except ImportError:
        st.info("모델 도감 준비 중")


for _t, _fn in zip(tabs, [page_procure, tab_risk, page_structure, page_factor, tab_regime, page_vol, page_predict, page_portfolio, tab_experiments, page_atlas, tab_data, page_settings]):
    with _t:
        if _t.open is not False:                                            # 고른 탭만 계산 (상태 추적이 꺼진 환경이면 None → 전부)
            try:
                _fn()
            finally:
                _keep_save()
