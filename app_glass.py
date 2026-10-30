"""대시보드 유리 테마판 — app.py를 그대로 실행하고, 그 위에 반투명 유리 + 에디토리얼(세리프 제목) CSS만 얹음.
app.py는 수정하지 않으므로 기능·숫자는 기본판과 같음. 실행: 대시보드_유리_열기.bat (포트 8502, 밝은 테마 고정)."""
import runpy
from pathlib import Path

import streamlit as st

CSS = """
:root { --g:#1E3A2F; --gold:#C8A24A; --glass:rgba(255,255,255,.52); --glass-line:rgba(255,255,255,.8);
  --shadow:0 10px 30px rgba(30,58,47,.12); --serif:"Batang","바탕","HANBatang","한컴바탕","Noto Serif KR","Malgun Gothic",serif; }
/* 배경: 은은한 색 덩어리가 천천히 흐르는 그라데이션 (유리 뒤에 비칠 재료) */
html body .stApp {
  background:
    radial-gradient(40% 45% at 8% 12%, rgba(46,125,82,.20), transparent 70%),
    radial-gradient(45% 50% at 92% 88%, rgba(200,162,74,.26), transparent 70%),
    radial-gradient(35% 40% at 75% 6%, rgba(47,110,143,.16), transparent 70%),
    linear-gradient(135deg, #F2F5F0, #FAF7EE) !important;
  background-size: 140% 140% !important; background-attachment: fixed !important; animation: gDrift 30s ease-in-out infinite alternate;
}
@keyframes gDrift { from { background-position: 0% 0%; } to { background-position: 100% 100%; } }
html body [data-testid="stHeader"] { background: rgba(246,248,244,.6) !important; backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); border-bottom: 1px solid rgba(255,255,255,.7); }
/* 사이드바: 뿌연 유리판 */
html body [data-testid="stSidebar"], html body [data-testid="stSidebar"] > div:first-child {
  background: rgba(255,255,255,.38) !important; backdrop-filter: blur(20px) saturate(150%); -webkit-backdrop-filter: blur(20px) saturate(150%);
  border-right: 1px solid var(--glass-line);
}
/* 탭 줄: 유리 막대 + 선택된 탭은 짙은 녹색 알약 */
html body [data-baseweb="tab-list"] {
  background: var(--glass); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); border: 1px solid var(--glass-line);
  border-radius: 14px; padding: 4px 6px; box-shadow: var(--shadow); gap: 2px;
}
html body [data-baseweb="tab-list"] button[role="tab"] { border-radius: 10px; padding: 4px 10px; transition: background .2s; }
html body [data-baseweb="tab-list"] button[role="tab"]:hover { background: rgba(255,255,255,.6); }
html body [data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] { background: var(--g); }
html body [data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] p { color: #fff !important; }
html body [data-baseweb="tab-highlight"], html body [data-baseweb="tab-border"] { display: none; }
/* 유리 카드: 테두리 있는 묶음 · 펼침 상자 · 지표 · 차트 · 표 · 알림 */
html body [data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) ,
html body [data-testid="stExpander"] details, html body [data-testid="stMetric"],
html body [data-testid="stPlotlyChart"], html body [data-testid="stDataFrame"], html body [data-testid="stTable"], html body [data-testid="stAlert"] {
  background: var(--glass) !important; backdrop-filter: blur(16px) saturate(150%); -webkit-backdrop-filter: blur(16px) saturate(150%);
  border: 1px solid var(--glass-line) !important; border-radius: 16px !important; box-shadow: var(--shadow), inset 0 1px 0 rgba(255,255,255,.6);
}
html body [data-testid="stMetric"] { padding: 10px 14px; }
html body [data-testid="stPlotlyChart"] { padding: 6px; }
html body [data-testid="stExpander"] summary:hover { background: rgba(255,255,255,.45); border-radius: 14px; }
/* 차트 바탕은 투명 → 유리판이 비침 */
html body .js-plotly-plot .main-svg { background: transparent !important; }
html body .js-plotly-plot .plotly rect.bg { fill: rgba(255,255,255,.35) !important; }
/* 입력 칸: 반투명 */
html body [data-baseweb="select"] > div, html body [data-baseweb="input"], html body [data-baseweb="base-input"], html body textarea {
  background: rgba(255,255,255,.62) !important; border-radius: 10px !important;
}
html body .stButton > button, html body .stDownloadButton > button {
  background: rgba(255,255,255,.6); backdrop-filter: blur(10px); border: 1px solid rgba(30,58,47,.25); border-radius: 12px; transition: all .2s;
}
html body .stButton > button:hover, html body .stDownloadButton > button:hover { background: var(--g); color: #fff; border-color: var(--g); }
html body .stButton > button[kind="primary"] { background: var(--g); color: #fff; }
/* ③ 에디토리얼: 세리프 제목 + 얇은 금색 밑줄 */
html body .stApp h1, html body .stApp h2, html body .stApp h3 { font-family: var(--serif) !important; letter-spacing: -0.01em; color: var(--g); }
html body .stApp h1 span, html body .stApp h2 span { box-shadow: inset 0 -0.14em 0 rgba(200,162,74,.4); }
/* 글자색: ⑨ 설정에서 어두운 배경을 골라 두면 app.py가 글자를 밝게 뒤집으므로, 유리판에서는 다시 진하게 고정 */
html body .stApp, html body .stApp p, html body .stApp li, html body .stApp label, html body .stApp span, html body .stApp h4, html body .stApp h5, html body .stApp h6,
html body .stApp [data-testid="stMarkdownContainer"], html body .stApp [data-testid="stCaptionContainer"], html body .stApp [data-testid="stWidgetLabel"] { color: #1f2a24 !important; }
html body .stApp [data-testid="stCaptionContainer"], html body .stApp [data-testid="stCaptionContainer"] p { color: #5f6862 !important; }
html body .stApp h1, html body .stApp h2, html body .stApp h3, html body .stApp h1 span, html body .stApp h2 span, html body .stApp h3 span { color: var(--g) !important; }
html body .stApp [data-tag], html body .stApp [data-tag] *, html body .stApp button[kind="primary"] *, html body .stApp [data-testid="stBaseButton-primary"] *,
html body .stApp .stButton > button:hover *, html body .stApp .stDownloadButton > button:hover * { color: #fff !important; }
html body [data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] p { color: #fff !important; }
/* 오른쪽 위 표시 */
html body .stApp::after { content: "유리 테마판"; position: fixed; right: 150px; top: 14px; z-index: 999990; font: 12px "Malgun Gothic",sans-serif;
  color: var(--g); background: rgba(255,255,255,.6); backdrop-filter: blur(8px); border: 1px solid var(--glass-line); padding: 2px 10px; border-radius: 10px; pointer-events: none; }
@media (prefers-reduced-motion: reduce) { html body .stApp { animation: none; } }
"""

_set_page_config = st.set_page_config


def _with_glass(*a, **k):                                   # app.py의 첫 화면 설정 직후에 유리 CSS를 넣음
    _set_page_config(*a, **k)
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


st.set_page_config = _with_glass
runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
