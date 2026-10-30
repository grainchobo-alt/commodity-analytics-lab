"""발표 v4 → 동적 HTML 한 파일 (Reveal.js + Plotly, 오프라인 실행).

deck_v4.js를 고치지 않고 그대로 실행해(deck_capture.js) 장마다 글상자·도형·그림·표·차트를 같은 위치에 HTML로 다시 그림.
- 모든 요소가 순서대로 날아 들어오는 등장 효과, 큰 숫자는 올라가며 세기
- 5장 품목 × 월 수익률 → 칸마다 숫자가 든 2D 히트맵(칸이 차례로 등장)
- 9·10장 AI 모델 카드 8개 그림 → 같은 자리에서 회전되는 3D (Plotly)
- 15장 효율적 투자선 → 대시보드 ⑤ 'Portfolio Customization' 3D(위험 × 수익 × Sharpe) + 연도별 경계선 재생
- 하단 자료 기준일, 열람일 기준 7일 지나면 빨간 경고 · 발표자 노트 = 10분 대본(키보드 S)

사용 (프로젝트 루트에서):  .venv\\Scripts\\python.exe docs\\발표\\scripts\\html_v4.py
결과: docs\\발표\\원자재_퀀트_MI_발표_v4.html (git 제외). deck_v4.js·notes_v4.js를 고친 뒤 다시 실행하면 반영.
"""
import base64
import html as H
import json
import shutil
import subprocess
import sys
import tempfile
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import analytics as A  # noqa: E402
import ls_strategies as LS  # noqa: E402

NM = HERE / "node_modules"
OUT = HERE.parent / "원자재_퀀트_MI_발표_v4.html"
WEB = HERE.parent / "MI_quant_v4_web"                           # 폴더판 (index.html + assets/). zip 안 이름은 영문(압축 도구별 한글 깨짐 방지)
ZIP = HERE.parent / "원자재_퀀트_MI_발표_v4_web"                 # → 원자재_퀀트_MI_발표_v4_web.zip
MIME = {".png": "image/png", ".mp4": "video/mp4"}
IMGS = {}                                                        # 장에 쓰인 그림 이름 → 경로
PLOTS = {"c47a_lasso3d": "lasso", "c47b_pca3d": "pca", "c47c_rf3d": "rf", "c47d_lstm3d": "lstm",
         "c48a_chronos3d": "chronos", "c48b_finbert3d": "finbert", "c48c_garch3d": "garch", "c48d_hmm3d": "hmm",
         "c49a_frontier_grid": "ef3d", "c49b_frontier_ribbon": "efyear", "c51_ml_overfit3d": "ovf",
         "c41_randomwalk": "rw", "c42_brownian": "mc", "c09_volatility": "vol", "c16_strategies": "strat",
         "c52_split3d": "split", "c53_linfit3d": "linfit", "c54_mlfit3d": "mlfit",
         "c55_mom3d": "mom", "c56_minvar3d": "minvar", "c57_value3d": "value"}   # 마지막 4개 = 선이 그려지는 2D


def r(x, n=2):
    return np.round(np.asarray(x, float), n).tolist()


# ════════════════════════════════════════ 데이터
def data():
    adj = A.load_levels(roll_adjusted=True); asof = adj.index.max()
    D = {"asof": f"{asof:%Y-%m-%d}", "built": f"{date.today():%Y-%m-%d}", "buy_asof": "2026-10-26"}
    rng = np.random.default_rng(3)
    # 8장: 변수 고르기 · 판단 나무 숲 · 기억 비중 (charts_v4와 같은 예시 자료)
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import lasso_path
    VN = ["달러", "금리", "원유", "재고", "작황", "기상", "포지션", "운임"]
    Xv = rng.normal(size=(600, 8)); beta = np.zeros(8); beta[[2, 3, 6]] = [0.9, -0.7, 0.5]; beta[[0, 4]] = [0.25, 0.12]
    yv = Xv @ beta + rng.normal(0, 1, 600); alphas = np.array([0.005, 0.05, 0.12, 0.22, 0.35])
    D["lasso"] = dict(vars=VN, lev=["약", "", "중", "", "강"], z=r(np.abs(lasso_path(Xv, yv, alphas=alphas)[1]), 3))
    x1, x2 = rng.uniform(0, 1, 1500), rng.uniform(0, 1, 1500)
    y = 2.0 * (x1 < 0.25) * (x2 > 0.5) - 1.2 * (x1 > 0.8) + 0.8 * np.sin(3 * x2) * x1 + rng.normal(0, 0.35, 1500)
    g = np.linspace(0, 1, 40); GX, GY = np.meshgrid(g, g)
    D["rf"] = dict(g=r(g), z=r(RandomForestRegressor(n_estimators=300, min_samples_leaf=15, random_state=0, n_jobs=-1).fit(np.c_[x1, x2], y).predict(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)))
    LG, TT = np.meshgrid(np.arange(1, 53), np.arange(60)); calm = 0.5 + 0.5 * np.sin(TT / 60 * 2 * np.pi * 1.3)
    D["lstm"] = dict(z=r(np.exp(-(LG - 1) / (3 + 6 * calm)) + 0.45 * np.exp(-((LG - 52) / 2.5) ** 2) + 0.25 * np.exp(-((LG - 26) / 3) ** 2), 3))
    # 8장: 실제 옥수수·대두·밀 PCA
    KN3 = {"yf:ZC=F": "옥수수", "yf:ZS=F": "대두", "yf:ZW=F": "밀"}
    P3 = LS.weekly(adj, list(KN3), "2007-01-01")[list(KN3)].rename(columns=KN3)
    R3 = np.log(P3).diff().dropna(); Z3 = ((R3 - R3.mean()) / R3.std()).clip(-4, 4)
    ev, evec = np.linalg.eigh(np.cov(Z3.values.T)); pc1 = evec[:, -1] * np.sign(evec[:, -1].sum())
    D["pca"] = dict(x=r(Z3.iloc[:, 0]), y=r(Z3.iloc[:, 1]), z=r(Z3.iloc[:, 2]), d=[f"{d:%Y-%m-%d}" for d in Z3.index], pc1=r(pc1, 3), share=round(float(ev[-1] / ev.sum()), 3))
    # 9장: 사전학습 AI 분포 · WASDE 어조(실제) · GARCH(실제) · HMM(실제)
    px = np.linspace(65, 140, 120); hs = [1, 2, 4, 8, 12]
    D["chronos"] = dict(px=r(px, 1), h=hs, z=[r(np.exp(-(np.log(px) - np.log(100)) ** 2 / (2 * (0.032 * np.sqrt(h)) ** 2)) / (px * 0.032 * np.sqrt(h) * np.sqrt(2 * np.pi)), 4) for h in hs])
    wt = pd.read_parquet(ROOT / "data/mm/wasde_text.parquet"); SEC = {"WHEAT": "밀", "COARSE GRAINS": "옥수수 등", "OILSEEDS": "대두 등"}
    wt = wt[wt.section.isin(SEC)]; wt["yr"] = wt.release.dt.year
    T = wt[wt.yr >= 2012].groupby(["yr", "section"]).tone.mean().unstack()[list(SEC)]
    D["finbert"] = dict(yrs=[int(v) for v in T.index], secs=list(SEC.values()), z=r(T.T.values, 3))
    from arch import arch_model
    from hmmlearn.hmm import GaussianHMM
    KN = {"yf:ZC=F": "옥수수", "yf:ZW=F": "밀", "yf:ZS=F": "대두", "yf:ZM=F": "대두박", "yf:CL=F": "원유", "yf:HG=F": "구리"}
    P = LS.weekly(adj, list(KN), "2005-01-01")[list(KN)].rename(columns=KN); Rw = np.log(P).diff().dropna() * 100
    GV = pd.DataFrame({k: arch_model(Rw[k], vol="GARCH", p=1, q=1, dist="t").fit(disp="off").conditional_volatility * np.sqrt(52) for k in Rw})
    GV = GV.loc["2007":].resample("ME").mean().clip(upper=90)
    D["garch"] = dict(t=[f"{d:%Y-%m}" for d in GV.index], items=list(GV.columns), z=r(GV.T.values, 1))
    ew = Rw.mean(axis=1).loc["2006":]
    hm = GaussianHMM(n_components=2, covariance_type="full", n_iter=300, random_state=0).fit(ew.values[:, None])
    cr = int(np.argmax(hm.covars_.ravel())); pc = pd.Series(hm.predict_proba(ew.values[:, None])[:, cr], index=ew.index).loc["2007":].resample("ME").mean()
    D["hmm"] = dict(t=[f"{d:%Y-%m}" for d in pc.index], p=r(pc.values, 3))
    # 5장: 품목 × 월 수익률 (최근 12개월)
    KX = {"yf:ZC=F": "옥수수", "yf:ZW=F": "밀", "yf:ZS=F": "대두", "yf:ZM=F": "대두박", "yf:ZL=F": "대두유", "yf:CL=F": "원유", "yf:NG=F": "천연가스",
          "yf:GC=F": "금", "yf:HG=F": "구리", "yf:KC=F": "커피", "yf:SB=F": "원당", "yf:CT=F": "면화"}
    PX = LS.weekly(adj, list(KX), "2020-01-01")[list(KX)].rename(columns=KX)
    MX = (np.log(PX).resample("ME").last().diff() * 100).iloc[-12:]
    D["xs"] = dict(months=[f"{d:%y.%m}" for d in MX.index], items=list(MX.columns), z=r(MX.T.values, 1))
    # 리스크 장: 19개 품목 주간 수익률 상관 (charts.py c26_corr와 같은 계산)
    KC = {"yf:ZC=F": "옥수수", "yf:ZW=F": "밀", "yf:ZS=F": "대두", "yf:ZM=F": "대두박", "yf:ZL=F": "대두유", "yf:CL=F": "원유", "yf:RB=F": "휘발유", "yf:HO=F": "난방유",
          "yf:NG=F": "천연가스", "yf:GC=F": "금", "yf:SI=F": "은", "yf:PL=F": "백금", "yf:HG=F": "구리", "yf:KC=F": "커피", "yf:SB=F": "원당", "yf:CC=F": "코코아",
          "yf:CT=F": "면화", "yf:LE=F": "생우", "yf:HE=F": "돈육"}
    PC = LS.weekly(adj, LS.BASKET, "2001-01-01"); PC = PC[[c_ for c_ in KC if c_ in PC]].rename(columns=KC)
    CM = (np.log(PC).diff().dropna(how="all") * 100).loc["2007":].corr()
    D["corr"] = dict(items=list(CM.columns), z=r(CM.values, 2))
    # 14장: 대시보드 ⑤ Portfolio Customization과 같은 계산 (3대 전략 · 롱온리 · 합 100%)
    from scipy.optimize import minimize
    W = pd.read_parquet(ROOT / "data/voltgt_weekly.parquet")
    BASE = {"기준 · MinVar 롱 (⑥)": "최소분산", "기준 · 52주 모멘텀 (④)": "모멘텀", "V1 가치 (5년 평균 대비 싼 품목 롱)": "가치"}
    X3 = pd.DataFrame({v: W[f"{k}|10|4"] for k, v in BASE.items()}).loc["2007":].dropna(); wmix = np.array([0.5, 0.25, 0.25])

    def ef(X, n_rand):
        mu, S = X.mean().values * 52, X.cov().values * 52; k = len(mu)
        Wr = np.vstack([np.random.default_rng(0).dirichlet(np.full(k, 0.6), n_rand), np.eye(k)])
        vol = np.sqrt(np.einsum("ij,jk,ik->i", Wr, S, Wr)); ret = Wr @ mu
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]; bnd = [(0, 1)] * k; w0 = np.full(k, 1 / k)
        gmv = minimize(lambda w: w @ S @ w, w0, bounds=bnd, constraints=cons, method="SLSQP").x
        tan = minimize(lambda w: -(w @ mu) / np.sqrt(w @ S @ w), w0, bounds=bnd, constraints=cons, method="SLSQP").x
        fr = []
        for t in np.linspace(gmv @ mu, mu.max(), 30):
            q = minimize(lambda w: w @ S @ w, w0, bounds=bnd, constraints=cons + [{"type": "eq", "fun": lambda w, t=t: w @ mu - t}], method="SLSQP")
            if q.success:
                fr.append((float(np.sqrt(q.x @ S @ q.x)), float(q.x @ mu)))
        pt = lambda w: r([np.sqrt(w @ S @ w), w @ mu])  # noqa: E731
        fr = np.array(fr)
        return dict(v=r(vol), r=r(ret), fv=r(fr[:, 0]), fr=r(fr[:, 1]), gmv=pt(gmv), tan=pt(tan), tanw=r(tan * 100, 0), mix=pt(wmix),
                    single=dict(v=r(np.sqrt(np.diag(S))), r=r(mu)), names=list(X.columns))
    E = {"전체 기간 2007–2026 (사후)": ef(X3, 3000)}
    for yr in range(2011, X3.index.max().year + 1):
        E[f"{yr} 시점 (직전 5년 {yr - 4}–{yr})"] = ef(X3.loc[f"{yr - 4}":f"{yr}"], 1200)
    D["ef3d"] = E
    full = E["전체 기간 2007–2026 (사후)"]; D["ef_text"] = dict(mix_sh=round(full["mix"][1] / full["mix"][0], 2), tan_sh=round(full["tan"][1] / full["tan"][0], 2), tanw=full["tanw"])
    FR = {}
    Wr = np.random.default_rng(0).dirichlet(np.full(3, 0.5), 6000)
    for yr in range(2011, X3.index.max().year + 1):          # 연도별 2D 경계선 (charts_v4 c49와 같은 계산)
        Xw = X3.loc[f"{yr - 4}":f"{yr}"]; mu, S = Xw.mean().values * 52, Xw.cov().values * 52
        vol = np.sqrt(np.einsum("ij,jk,ik->i", Wr, S, Wr)); ret = Wr @ mu; best = -np.inf; fv, fr = [], []
        for i in np.argsort(vol):
            if ret[i] > best:
                best = ret[i]; fv.append(vol[i]); fr.append(ret[i])
        k = int(np.argmax(ret / vol)); mv, mr = float(np.sqrt(wmix @ S @ wmix)), float(wmix @ mu); fv_, fr_ = np.array(fv), np.array(fr)
        gap = float(fr_[fv_ <= mv].max() - mr) if (fv_ <= mv).any() else 0.0
        FR[yr] = dict(fv=r(fv_), fr=r(fr_), mix=r([mv, mr]), tan=r([vol[k], ret[k]]), gap=round(max(gap, 0), 1))
    D["efyear"] = FR
    # 6장 랜덤워크·몬테카를로 (charts_v2 c41·c42와 같은 난수·같은 순서 → PPT와 같은 그림, 정답 위치 동일)
    adj2 = adj.loc[:"2026-10-26"]
    KN12 = ["yf:ZC=F", "yf:ZW=F", "yf:ZS=F", "yf:ZM=F", "yf:ZL=F", "yf:CL=F", "yf:NG=F", "yf:GC=F", "yf:HG=F", "yf:KC=F", "yf:SB=F", "yf:CT=F"]
    P12 = LS.weekly(adj2, KN12, "2001-01-01")[KN12]
    c = np.log(P12["yf:ZC=F"].dropna()).loc["2016-10":]; rr = c.diff().dropna(); sd = float(rr.std()); g = np.random.default_rng(7)
    real = (c - c.iloc[0]).values; paths, ends = [], [real[-1]]
    while len(paths) < 5:
        p_ = np.r_[0, np.cumsum(g.normal(0, sd, len(rr)))]
        if np.exp(p_).min() > 0.45 and np.exp(p_).max() < 2.1 and min(abs(p_[-1] - e) for e in ends) > 0.12:
            paths.append(p_); ends.append(p_[-1])
    paths.insert(3, real)
    D["rw"] = dict(d=[f"{x:%Y-%m-%d}" for x in c.index], y=[r(100 * np.exp(q), 1) for q in paths])
    Z = g.normal(0, sd, (400, 52)).cumsum(1); wk = np.arange(1, 53)
    D["mc"] = dict(paths=[r(100 * np.exp(Z[k]), 1) for k in range(120)], lo=r(100 * np.exp(-1.645 * sd * np.sqrt(wk)), 1), hi=r(100 * np.exp(1.645 * sd * np.sqrt(wk)), 1))
    # 7장 옥수수 변동성 군집 (charts c09)
    wv = np.log(adj2["yf:ZC=F"].dropna()).resample("W-FRI").last().diff().dropna().loc["2006":] * 100; rv = wv.rolling(26).std() * np.sqrt(52)
    D["vol"] = dict(d=[f"{x:%Y-%m-%d}" for x in wv.index], w=r(wv.values, 2), rv=[None if np.isnan(v) else round(float(v), 1) for v in rv.values])
    # 14장 전략 누적 수익 (charts c16)
    D["strat"] = dict(d=[f"{x:%Y-%m-%d}" for x in X3.index], names=["최소분산", "52주 모멘텀", "가치"], y=[r(X3[k].cumsum().values, 1) for k in X3.columns],
                      mix=r(W["혼합 · MinVar 50 + 모멘텀 25 + 가치 25 (사후 선택 주의)|10|4"].loc["2007":].reindex(X3.index).fillna(0).cumsum().values, 1))
    # 7장 선형 평면 vs 머신러닝 곡면 (charts_v4 c53·c54와 같은 난수)
    from sklearn.kernel_ridge import KernelRidge
    rs = np.random.default_rng(11); x1, x2 = rs.uniform(-2, 2, 120), rs.uniform(-2, 2, 120); yl = 0.8 * x1 + 0.5 * x2 + rs.normal(0, 0.35, 120)
    bl = np.linalg.lstsq(np.c_[np.ones(120), x1, x2], yl, rcond=None)[0]; gx = np.linspace(-2, 2, 15)
    D["linfit"] = dict(x=r(x1), y=r(x2), z=r(yl), g=r(gx), plane=[r(bl[0] + bl[1] * gx + bl[2] * v) for v in gx])
    rs = np.random.default_rng(12); z1, z2 = rs.uniform(-3, 3, 260), rs.uniform(-3, 3, 260)
    yn = 1.5 * np.sin(1.3 * z1) + 0.8 * np.cos(1.1 * z2) + 0.15 * z1 * z2 + rs.normal(0, 0.35, 260)
    kr = KernelRidge(kernel="rbf", alpha=0.3, gamma=0.5).fit(np.c_[z1, z2], yn); gz = np.linspace(-3, 3, 30); ZX, ZY = np.meshgrid(gz, gz)
    bn = np.linalg.lstsq(np.c_[np.ones(260), z1, z2], yn, rcond=None)[0]
    D["mlfit"] = dict(x=r(z1), y=r(z2), z=r(yn), g=r(gz), surf=r(kr.predict(np.c_[ZX.ravel(), ZY.ravel()]).reshape(ZX.shape)),
                      plane=r(bn[0] + bn[1] * ZX + bn[2] * ZY))
    return D


# ════════════════════════════════════════ 캡처 → HTML 요소
U = lambda v: f"calc(var(--u)*{float(v):.4f})"          # noqa: E731  인치 → 화면 (var(--u) = 1인치 픽셀)
PT = lambda v: U(float(v) / 72)                          # noqa: E731
VAL = {"top": "flex-start", "middle": "center", "bottom": "flex-end"}


def color(c, alpha=None):
    return f"#{c}" if c and alpha is None else (f"rgba({int(c[0:2], 16)},{int(c[2:4], 16)},{int(c[4:6], 16)},{alpha})" if c else "transparent")


def box_style(o):
    return f"left:{U(o['x'])};top:{U(o['y'])};width:{U(o['w'])};height:{U(o['h'])};"


def run_style(ro, base):
    st = []
    if ro.get("bold") is not None and ro.get("bold") != base.get("bold"):
        st.append(f"font-weight:{700 if ro['bold'] else 400}")
    if ro.get("color") and ro.get("color") != base.get("color"):
        st.append(f"color:#{ro['color']}")
    if ro.get("fontSize") and ro.get("fontSize") != base.get("fontSize"):
        st.append(f"font-size:{PT(ro['fontSize'])}")
    if ro.get("italic"):
        st.append("font-style:italic")
    return ";".join(st)


def text_html(text, o):
    runs = [{"text": text, "options": {}}] if not isinstance(text, list) else text
    paras, props = [[]], [{}]
    for run in runs:
        ro = run.get("options", {}) or {}
        for k, part in enumerate(str(run.get("text", "")).split("\n")):
            if k:
                paras.append([]); props.append({})
            if not paras[-1]:
                props[-1] = ro
            paras[-1].append((part, ro))
        if ro.get("breakLine"):
            paras.append([]); props.append({})
    if len(paras) > 1 and not paras[-1]:
        paras.pop(); props.pop()
    out = []
    for runs_, pr in zip(paras, props):
        bul = pr.get("bullet", o.get("bullet"))
        psa = pr.get("paraSpaceAfter", o.get("paraSpaceAfter", 0)) or 0
        al = pr.get("align", o.get("align"))
        st = f"margin:0 0 {PT(psa)} 0;" + (f"text-align:{al};" if al else "") + ("padding-left:1.05em;text-indent:-0.85em;" if bul else "")
        body = "".join(f'<span style="{run_style(ro, o)}">{H.escape(t)}</span>' if run_style(ro, o) else H.escape(t) for t, ro in runs_) or "&nbsp;"
        out.append(f'<p style="{st}">' + ("•&nbsp;&nbsp;" if bul else "") + body + "</p>")
    return "".join(out)


def text_item(text, o, cls=""):
    m = o.get("margin")
    pad = "0.05in 0.1in" if m is None else None
    pad = f"{U(0.05)} {U(0.1)}" if m is None else (" ".join(PT(v) for v in m) if isinstance(m, list) else PT(m))
    st = (box_style(o) + f"padding:{pad};font-size:{PT(o.get('fontSize', 18))};color:#{o.get('color', '000000')};"
          f"font-weight:{700 if o.get('bold') else 400};justify-content:{VAL.get(o.get('valign'), 'flex-start')};" + (f"text-align:{o['align']};" if o.get("align") else ""))
    raw = text if isinstance(text, str) else ""
    cnt = ""
    if raw and o.get("fontSize", 0) >= 28 and any(ch.isdigit() for ch in raw) and len(raw) <= 9:
        cnt = f' data-count="{H.escape(raw)}"'
    return f'<div class="el tx {cls}" style="{st}"{cnt}>{text_html(text, o)}</div>'


def shape_item(t, o):
    fill, line = o.get("fill") or {}, o.get("line") or {}
    if t == "line":
        x1, y1, x2, y2 = o["x"], o["y"], o["x"] + o["w"], o["y"] + o["h"]
        L, T_, Wd, Ht = min(x1, x2), min(y1, y2), max(abs(o["w"]), 0.02), max(abs(o["h"]), 0.02)
        c = color(line.get("color", "888888")); lw = line.get("width", 1) / 72 * 100
        mk = f'<defs><marker id="m{id(o)}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker></defs>' if line.get("endArrowType") else ""
        dash = ' stroke-dasharray="8 6"' if line.get("dashType") else ""
        return (f'<svg class="el ln" style="left:{U(L)};top:{U(T_)};width:{U(Wd)};height:{U(Ht)};overflow:visible" viewBox="0 0 {Wd * 100:.1f} {Ht * 100:.1f}" preserveAspectRatio="none">{mk}'
                f'<line x1="{(x1 - L) * 100:.1f}" y1="{(y1 - T_) * 100:.1f}" x2="{(x2 - L) * 100:.1f}" y2="{(y2 - T_) * 100:.1f}" stroke="{c}" stroke-width="{lw:.2f}"{dash}'
                + (f' marker-end="url(#m{id(o)})"' if mk else "") + ' vector-effect="non-scaling-stroke"/></svg>')
    a = None if fill.get("transparency") is None else 1 - fill["transparency"] / 100
    st = box_style(o) + f"background:{color(fill.get('color'), a)};"
    if line.get("color") and line.get("color") != fill.get("color"):
        st += f"border:{PT(line.get('width', 0.75))} {'dashed' if line.get('dashType') else 'solid'} #{line['color']};"
    st += {"ellipse": "border-radius:50%;", "roundRect": f"border-radius:{U(o.get('rectRadius', 0.08))};"}.get(t, "")
    return f'<div class="el sh" style="{st}"></div>'


def table_item(rows, o):
    n = len(rows[0]); cw = o.get("colW") or [o["w"] / n] * n; rh = o.get("rowH"); fs = o.get("fontSize", 12)
    bc = (o.get("border") or {}).get("color", "D6D9D2")
    trs = []
    for i, row in enumerate(rows):
        tds = []
        for j, c in enumerate(row):
            t, co = (c, {}) if not isinstance(c, dict) else (c.get("text", ""), c.get("options", {}) or {})
            st = f"border:1px solid #{bc};padding:{U(0.03)} {U(0.07)};vertical-align:{o.get('valign', 'middle')};"
            if co.get("fill"):
                st += f"background:#{co['fill']['color']};"
            st += f"color:#{co.get('color', o.get('color', '222222'))};font-weight:{700 if co.get('bold') else 400};"
            al = co.get("align", o.get("align"))
            if al:
                st += f"text-align:{al};"
            if co.get("fontSize"):
                st += f"font-size:{PT(co['fontSize'])};"
            tds.append(f'<td style="{st}">' + "<br>".join(H.escape(x) for x in str(t).split("\n")) + "</td>")
        h_ = rh[i] if isinstance(rh, list) else rh
        trs.append(f'<tr style="height:{U(h_)}">' + "".join(tds) + "</tr>" if h_ else "<tr>" + "".join(tds) + "</tr>")
    cols = "".join(f'<col style="width:{U(w)}">' for w in cw)
    return (f'<div class="el tb" style="left:{U(o["x"])};top:{U(o["y"])};width:{U(sum(cw))};font-size:{PT(fs)}">'
            f'<table style="width:100%;border-collapse:collapse;table-layout:fixed"><colgroup>{cols}</colgroup>{"".join(trs)}</table></div>')


def heat_html(o, X):
    """5장: 품목 × 월 수익률 2D 히트맵 — 칸마다 숫자, 옥수수 줄(시계열) 파랑 · 마지막 달 열(횡단면) 금색, 칸이 차례로 등장."""
    def cl(v):
        t = max(-1, min(1, v / 12))
        a, b = ((248, 247, 242), (46, 125, 82)) if t >= 0 else ((248, 247, 242), (205, 74, 50))
        k = abs(t) ** 0.8; rgb = [round(a[i] + (b[i] - a[i]) * k) for i in range(3)]
        return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})", ("#fff" if k > 0.55 else "#2a2a2a")
    nm, ni = len(X["months"]), len(X["items"])
    cells = ['<div class="hx-c hx-h"></div>'] + [f'<div class="hx-c hx-h{" hx-last" if b == nm - 1 else ""}">{m}</div>' for b, m in enumerate(X["months"])]
    for a, it in enumerate(X["items"]):
        cells.append(f'<div class="hx-c hx-r{" hx-ts" if a == 0 else ""}">{it}</div>')
        for b in range(nm):
            v = X["z"][a][b]; bg, fg = cl(v); cls = "hx-v" + (" hx-tsv" if a == 0 else "") + (" hx-csv" if b == nm - 1 else "")
            cells.append(f'<div class="hx-c {cls}" style="background-color:{bg};color:{fg};--k:{a + b}" data-r="{a}" data-c="{b}" data-tip="<b>{it}</b> · 20{X["months"][b].replace(".", "년 ")}월<br>월 수익률 {v:+.1f}%">{"0" if round(v) == 0 else f"{v:+.0f}"}</div>')
    leg = ('<div class="hx-leg"><span>−12%</span><i></i><span>+12%</span><b class="hx-key ts">━ 시계열: 옥수수 12개월 (가로)</b><b class="hx-key cs">┃ 횡단면: 최근 월 품목 비교 (세로)</b></div>')
    return (f'<div class="el hx" style="{box_style(o)}"><div class="hx-g" style="grid-template-columns:{U(0.85)} repeat({nm},1fr);grid-template-rows:{U(0.3)} repeat({ni},1fr)">'
            + "".join(cells) + f"</div>{leg}</div>")


def corr_html(o, X):
    """리스크 장: 19 × 19 상관 격자 — 입체 칸 · 마우스를 대면 그 줄과 열만 강조 · 같이 움직이는 묶음 테두리."""
    def cl(v):                                                     # 파랑(−1) · 흰색(0) · 빨강(+1)
        a, b = ((247, 246, 242), (178, 34, 52)) if v >= 0 else ((247, 246, 242), (33, 102, 172))
        k = min(1, abs(v)) ** 0.75; rgb = [round(a[i] + (b[i] - a[i]) * k) for i in range(3)]
        return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})", ("#fff" if k > 0.5 else "#2a2a2a")
    it, n = X["items"], len(X["items"])
    cells = []
    for a in range(n):
        cells.append(f'<div class="cx-r" style="grid-row:{a + 1};grid-column:1" data-r="{a}">{it[a]}</div>')
        for b in range(n):
            v = X["z"][a][b]; bg, fg = cl(v); diag = a == b
            txt = "" if diag else (f"{v:.1f}" if abs(v) >= 0.4 else "")
            tip = f"<b>{it[a]}</b>" if diag else f"<b>{it[a]} × {it[b]}</b><br>상관 {v:+.2f} · " + ("함께 강하게 움직임" if v >= 0.5 else "어느 정도 함께" if v >= 0.25 else "거의 따로 움직임")
            cells.append(f'<div class="hx-c hx-v cx-v{" cx-d" if diag else ""}" style="grid-row:{a + 1};grid-column:{b + 2};background-color:{bg};color:{fg};--k:{(a + b) * 0.55:.1f}" '
                         f'data-r="{a}" data-c="{b}" data-tip="{tip}">{txt}</div>')
    for b in range(n):
        cells.append(f'<div class="cx-col" style="grid-row:{n + 1};grid-column:{b + 2}" data-c="{b}"><span>{it[b]}</span></div>')
    for a0, a1, nm in ((0, 4, "곡물·유지"), (5, 8, "에너지"), (9, 12, "금속")):   # 빨간 묶음 = 같이 움직이는 품목
        cells.append(f'<div class="cx-grp" style="grid-row:{a0 + 1}/{a1 + 2};grid-column:{a0 + 2}/{a1 + 3}"><i>{nm}</i></div>')
    leg = '<div class="hx-leg cx-leg"><span>−1 반대로</span><i></i><span>+1 함께</span><b class="hx-key">마우스를 대면 그 품목의 줄·열만 강조</b></div>'
    return (f'<div class="el hx cx" style="{box_style(o)}"><div class="cx-t">19개 품목 주간 수익률 상관 (2007–2026)</div>'
            f'<div class="hx-g cx-g" style="grid-template-columns:{U(0.62)} repeat({n},1fr);grid-template-rows:repeat({n},1fr) {U(0.55)}">'
            + "".join(cells) + f"</div>{leg}</div>")


def plot_item(o, key, extra=""):
    return f'<div class="el pl" style="{box_style(o)}"><div class="plot" data-plot="{key}"></div>{extra}</div>'


def anim_class(it):
    o = it.get("opts", {}); k = it["kind"]
    if o.get("y", 9) < 1.0:
        return "a-left"
    if k == "image" or k == "chart":
        return "a-zoom"
    if k == "shape":
        return "a-draw" if it["type"] == "line" else "a-fade"
    return "a-up"


def slide_html(n, s, D):
    ef_slide = any(it["kind"] == "image" and Path(it["opts"]["path"]).stem == "c49a_frontier_grid" for it in s["items"])
    dark = (s.get("background") or {}).get("color") == "1E3A2F"
    els, i = [], 0
    for it in s["items"]:
        o = it.get("opts", {}); k = it["kind"]
        # 14장: 연도별 리본 그림의 설명 글은 빼고(재생 막대 자리), 읽는 법 글은 새 3D 구성에 맞게 바꿈
        if ef_slide and k == "text" and isinstance(it["text"], str) and it["text"].startswith("3D: 3년 간격"):
            continue                                       # 재생 막대가 같은 자리를 씀
        if ef_slide and k == "text" and isinstance(it["text"], list) and str(it["text"][0].get("text", "")).startswith("읽는 법"):
            et = D["ef_text"]
            it = dict(it, text=[{"text": "읽는 법 · 시사점", "options": {"bold": True, "color": "1E3A2F", "breakLine": True, "paraSpaceAfter": 6}},
                                {"text": "왼쪽 3D = 대시보드 ⑤ 투자선 화면과 같은 계산: 점 = 세 전략을 섞는 비중, 높이·색 = Sharpe(위험 1단위당 수익)", "options": {"bullet": True, "breakLine": True, "paraSpaceAfter": 6}},
                                {"text": f"전체 기간: ★ 우리 혼합 Sharpe {et['mix_sh']} vs 사후 최대 {et['tan_sh']} (최소분산 {et['tanw'][0]:.0f} · 모멘텀 {et['tanw'][1]:.0f} · 가치 {et['tanw'][2]:.0f}%) → 고정 혼합이 꼭대기 근처", "options": {"bullet": True, "breakLine": True, "paraSpaceAfter": 6}},
                                {"text": "2015–20: 한 전략 독주 시기 → 그해 기준 ★는 경계선보다 연 3–4%p 아래 (오른쪽 위 재생)", "options": {"bullet": True, "breakLine": True, "paraSpaceAfter": 6}},
                                {"text": "사후 최적 위치는 해마다 크게 이동 → 미리 알 수 없음. 시점 선택으로 그해 3D도 확인", "options": {"bullet": True}}])
        a = anim_class(it); i += 1; style_i = f"--i:{min(i, 34)}"
        if k == "text":
            h_ = text_item(it["text"], o)
        elif k == "shape":
            h_ = shape_item(it["type"], o)
        elif k == "table":
            h_ = table_item(it["rows"], o)
        elif k == "media":                                          # 시연 영상: 한 파일판은 base64, 폴더판은 파일
            IMGS[Path(o["path"]).name] = Path(o["path"])
            h_ = (f'<div class="el vd" style="{box_style(o)}"><video src="@IMG:{Path(o["path"]).name}" poster="{o.get("cover", "")}" controls muted playsinline preload="auto"></video></div>')
        elif k == "chart":
            cid = f"chart{n}_{i}"; D.setdefault("charts", {})[cid] = dict(type=it["type"], data=it["data"], opts=o); h_ = plot_item(o, cid)
        else:
            name = Path(o["path"]).stem
            if name == "c50_xs_ts3d":
                h_ = heat_html(o, D["xs"])
            elif name == "c26_corr":
                h_ = corr_html(o, D["corr"])
            elif name in PLOTS:
                key = PLOTS[name]
                extra = ""
                if key == "ef3d":
                    extra = '<select class="ef-pick"></select>'
                if key == "efyear":
                    extra = '<div class="ey-ctl"><button class="ey-play">▶</button><input class="ey-yr" type="range"><b class="ey-lab"></b><span class="ey-gap"></span></div>'
                h_ = plot_item(o, key, extra)
            else:
                IMGS[Path(o["path"]).name] = Path(o["path"])                # 폴더판은 파일로, 한 파일판은 base64로 (main에서 치환)
                h_ = f'<div class="el im" style="{box_style(o)}"><img src="@IMG:{Path(o["path"]).name}" alt=""></div>'
        els.append(h_.replace('class="el ', f'class="el {a} ', 1).replace('style="', f'style="{style_i};', 1))
    note = f'<aside class="notes">{H.escape(s.get("notes", ""))}</aside>'
    bg = ' data-background-color="#1E3A2F"' if dark else ""
    return f'<section data-slide="{n}"{bg}><div class="cv{" dark" if dark else ""}">' + "".join(els) + f"</div>{note}</section>"


def main():
    sys.stdout.reconfigure(errors="replace")
    print("데이터 계산…", flush=True); D = data()
    print("deck_v4.js 캡처…", flush=True)
    cap = Path(tempfile.mkdtemp()) / "capture.json"
    subprocess.run([shutil.which("node"), "deck_capture.js", "4", str(cap)], cwd=HERE, check=True, capture_output=True)
    S = json.loads(cap.read_text(encoding="utf-8"))["slides"]
    secs = [slide_html(n, s, D) for n, s in enumerate(S, 1)]
    tpl = (HERE / "html_v4_template.html").read_text(encoding="utf-8").replace("<!--@SLIDES-->", "\n".join(secs))
    src = {"REVEAL_CSS": NM / "reveal.js/dist/reveal.css", "REVEAL_JS": NM / "reveal.js/dist/reveal.js", "NOTES_JS": NM / "reveal.js/plugin/notes/notes.js",
           "PLOTLY_JS": NM / "plotly.js-dist-min/plotly.min.js", "APP_JS": HERE / "html_v4_app.js"}
    data_js = "const D = " + json.dumps(D, ensure_ascii=False) + ";"
    # ① 한 파일판: 모두 인라인 (파일 하나로 전달할 때)
    rd = lambda k: src[k].read_text(encoding="utf-8").replace("</script", "<\\/script")     # noqa: E731  인라인 스크립트 안의 </script 차단
    one = tpl
    for k in ("REVEAL_CSS", "REVEAL_JS", "NOTES_JS", "PLOTLY_JS", "APP_JS"):
        one = one.replace(f"/*@{k}*/", rd(k))
    one = one.replace("/*@DATA*/", data_js.replace("</", "<\\/"))
    for nm, p in IMGS.items():
        one = one.replace(f"@IMG:{nm}", f"data:{MIME.get(p.suffix, 'image/png')};base64," + base64.b64encode(p.read_bytes()).decode())
    OUT.write_text(one, encoding="utf-8")
    # ② 폴더판: index.html(작음) + assets/ → zip으로 전달, 받은 쪽은 압축을 푼 뒤 index.html을 염 (file://에서도 동작: fetch 없이 <script src>만 사용)
    if WEB.exists():
        shutil.rmtree(WEB)
    (WEB / "assets" / "img").mkdir(parents=True)
    page = tpl.replace("<style>/*@REVEAL_CSS*/</style>", '<link rel="stylesheet" href="assets/reveal.css">')
    for k, fn in (("REVEAL_JS", "reveal.js"), ("NOTES_JS", "notes.js"), ("PLOTLY_JS", "plotly.min.js")):
        page = page.replace(f"<script>/*@{k}*/</script>", f'<script src="assets/{fn}"></script>')
    page = page.replace("<script>\n/*@DATA*/\n/*@APP_JS*/\n</script>", '<script src="assets/data.js"></script>\n<script src="assets/app.js"></script>')
    assert "/*@" not in page, "템플릿 자리표시자 치환 누락"
    for nm, p in IMGS.items():
        page = page.replace(f"@IMG:{nm}", f"assets/img/{nm}"); shutil.copy(p, WEB / "assets" / "img" / nm)
    (WEB / "index.html").write_text(page, encoding="utf-8")
    for k, fn in (("REVEAL_CSS", "reveal.css"), ("REVEAL_JS", "reveal.js"), ("NOTES_JS", "notes.js"), ("PLOTLY_JS", "plotly.min.js"), ("APP_JS", "app.js")):
        shutil.copy(src[k], WEB / "assets" / fn)
    (WEB / "assets" / "data.js").write_text(data_js, encoding="utf-8")
    (WEB / "README.txt").write_text("원자재 퀀트 MI 발표 v4 (인터랙티브 HTML)\n\n1) zip 압축을 먼저 푸세요 (zip 안에서 바로 열면 그림·차트를 못 찾음)\n2) index.html을 Chrome 또는 Edge로 여세요 (메일·메신저 미리보기, Internet Explorer 불가)\n"
                                                 "3) 방향키·마우스 휠 = 장 넘김, F = 전체 화면, S = 발표자 보기(대본·타이머)\n", encoding="utf-8")
    z = shutil.make_archive(str(ZIP), "zip", root_dir=WEB.parent, base_dir=WEB.name)
    print(f"완료 ① 한 파일: {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")
    print(f"완료 ② 폴더+zip: {z}  ({Path(z).stat().st_size / 1e6:.1f} MB, index.html {(WEB / 'index.html').stat().st_size / 1e6:.2f} MB) · {len(S)}장 · 자료 기준일 {D['asof']}")


if __name__ == "__main__":
    main()
