"""발표 v4 → 인터랙티브 HTML 한 파일 (Reveal.js + Plotly, 오프라인 실행).

구성: 본문·별첨 20장은 pptx를 그대로 그림으로 넣고, 아래 장만 HTML + Plotly로 바꿈
  5장 시계열·횡단면 → 3D 막대(회전) / 2D 표 전환 · 14장 효율적 투자선 → 연도 슬라이더·재생
  15장 구매 판단 → 품목 선택 · 8·9장 아래(↓)에 상세 3D(PCA · GARCH · HMM) — 질의응답용
모든 장 하단에 자료 기준일, 열람일 기준 7일 지나면 빨간 경고(15·16장 숫자 갱신 필요).
발표자 노트 = notes_v4.js 대본(키보드 S → 발표자 보기).

사용 (프로젝트 루트에서):  .venv\\Scripts\\python.exe docs\\발표\\scripts\\html_v4.py
결과: docs\\발표\\원자재_퀀트_MI_발표_v4.html (git 제외)
"""
import base64
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
SOFFICE = Path(r"C:\Program Files\LibreOffice\program\soffice.exe")
NATIVE = {5: "xs", 14: "frontier", 15: "buy"}               # 장 번호 → HTML 장으로 대체
DRILL = {8: ["pca"], 9: ["garch", "hmm"]}                     # 장 번호 → 아래(↓) 상세 장


def r(x, n=2):
    return np.round(np.asarray(x, float), n).tolist()


def data():
    adj = A.load_levels(roll_adjusted=True)
    asof = adj.index.max()
    D = {"asof": f"{asof:%Y-%m-%d}", "built": f"{date.today():%Y-%m-%d}"}
    # 5장: 품목 × 월 수익률 (최근 12개월)
    KX = {"yf:ZC=F": "옥수수", "yf:ZW=F": "밀", "yf:ZS=F": "대두", "yf:ZM=F": "대두박", "yf:ZL=F": "대두유", "yf:CL=F": "원유", "yf:NG=F": "천연가스",
          "yf:GC=F": "금", "yf:HG=F": "구리", "yf:KC=F": "커피", "yf:SB=F": "원당", "yf:CT=F": "면화"}
    PX = LS.weekly(adj, list(KX), "2020-01-01")[list(KX)].rename(columns=KX)
    MX = (np.log(PX).resample("ME").last().diff() * 100).iloc[-12:]
    D["xs"] = dict(months=[f"{d:%y.%m}" for d in MX.index], items=list(MX.columns), z=r(MX.T.values, 1))
    # 14장: 연도별 직전 5년 효율적 투자선 (charts_v4 c49와 같은 계산)
    W = pd.read_parquet(ROOT / "data/voltgt_weekly.parquet")
    BASE = {"기준 · MinVar 롱 (⑥)": "최소분산", "기준 · 52주 모멘텀 (④)": "모멘텀", "V1 가치 (5년 평균 대비 싼 품목 롱)": "가치"}
    X3 = pd.DataFrame({v: W[f"{k}|10|4"] for k, v in BASE.items()}).loc["2007":].dropna()
    Wr = np.random.default_rng(0).dirichlet(np.full(3, 0.5), 6000); wmix = np.array([0.5, 0.25, 0.25]); FR = {}
    for yr in range(2011, X3.index.max().year + 1):
        Xw = X3.loc[f"{yr - 4}":f"{yr}"]; mu, S = Xw.mean().values * 52, Xw.cov().values * 52
        vol = np.sqrt(np.einsum("ij,jk,ik->i", Wr, S, Wr)); ret = Wr @ mu; best = -np.inf; fv, fr, fw = [], [], []
        for i in np.argsort(vol):
            if ret[i] > best:
                best = ret[i]; fv.append(vol[i]); fr.append(ret[i]); fw.append(Wr[i])
        k = int(np.argmax(ret / vol)); mv, mr = float(np.sqrt(wmix @ S @ wmix)), float(wmix @ mu)
        fv_, fr_ = np.array(fv), np.array(fr); gap = float(fr_[fv_ <= mv].max() - mr) if (fv_ <= mv).any() else 0.0
        FR[yr] = dict(cv=r(vol[::12]), cr=r(ret[::12]), fv=r(fv_), fr=r(fr_), mix=r([mv, mr]), tan=r([vol[k], ret[k]]), tanw=r(Wr[k] * 100, 0), gap=round(max(gap, 0), 1))
    D["frontier"] = FR
    # 15장: 구매 판단 4품목 — 최근 3년 주간 가격(원 단위 연속월물) + 판단(deck_v4.js와 같은 값)
    raw = A.load_levels(roll_adjusted=False)
    KB = {"yf:ZC=F": ("옥수수", "센트/부셸"), "yf:ZW=F": ("밀", "센트/부셸"), "yf:ZS=F": ("대두", "센트/부셸"), "yf:ZM=F": ("대두박", "달러/숏톤")}
    PB = LS.weekly(raw, list(KB), f"{asof.year - 3}-{asof:%m-%d}")[list(KB)]
    CARD = {"옥수수": ("적극 매수", "#2E7D3E", 60, "저점권 + 대체재 대비 쌈", True), "밀": ("판단 유보", "#B8901C", 41, "최근 상승, 급하지 않으면 대기", False),
            "대두": ("판단 유보", "#B8901C", 43, "반등 약함, 다음 주 재확인", False), "대두박": ("판단 유보", "#B8901C", 35, "1년 추세 상승 중", True)}
    D["buy"] = {KB[k][0]: dict(unit=KB[k][1], d=[f"{d:%Y-%m-%d}" for d in PB.index], p=r(PB[k].values, 1), lo=float(PB[k].min()), hi=float(PB[k].max()),
                                 pct=round(float((PB[k].iloc[-1] - PB[k].min()) / (PB[k].max() - PB[k].min()) * 100), 0),
                                 act=CARD[KB[k][0]][0], col=CARD[KB[k][0]][1], score=CARD[KB[k][0]][2], why=CARD[KB[k][0]][3], evid=CARD[KB[k][0]][4]) for k in KB}
    D["buy_asof"] = "2026-10-26"
    # 8장 아래: 옥수수·대두·밀 PCA
    KN3 = {"yf:ZC=F": "옥수수", "yf:ZS=F": "대두", "yf:ZW=F": "밀"}
    P3 = LS.weekly(adj, list(KN3), "2007-01-01")[list(KN3)].rename(columns=KN3)
    R3 = np.log(P3).diff().dropna(); Z3 = ((R3 - R3.mean()) / R3.std()).clip(-4, 4)
    ev, evec = np.linalg.eigh(np.cov(Z3.values.T)); pc1 = evec[:, -1] * np.sign(evec[:, -1].sum())
    D["pca"] = dict(x=r(Z3.iloc[:, 0]), y=r(Z3.iloc[:, 1]), z=r(Z3.iloc[:, 2]), d=[f"{d:%Y-%m-%d}" for d in Z3.index], pc1=r(pc1, 3), share=round(float(ev[-1] / ev.sum()), 3))
    # 9장 아래: GARCH 6품목 · HMM 위기 확률
    from arch import arch_model
    from hmmlearn.hmm import GaussianHMM
    KN = {"yf:ZC=F": "옥수수", "yf:ZW=F": "밀", "yf:ZS=F": "대두", "yf:ZM=F": "대두박", "yf:CL=F": "원유", "yf:HG=F": "구리"}
    P = LS.weekly(adj, list(KN), "2005-01-01")[list(KN)].rename(columns=KN); Rw = np.log(P).diff().dropna() * 100
    GV = pd.DataFrame({k: arch_model(Rw[k], vol="GARCH", p=1, q=1, dist="t").fit(disp="off").conditional_volatility * np.sqrt(52) for k in Rw})
    GV = GV.loc["2007":].resample("ME").mean().clip(upper=90)
    D["garch"] = dict(t=[f"{d:%Y-%m}" for d in GV.index], items=list(GV.columns), z=r(GV.T.values, 1))
    ew = Rw.mean(axis=1).loc["2006":]
    hm = GaussianHMM(n_components=2, covariance_type="full", n_iter=300, random_state=0).fit(ew.values[:, None])
    cr = int(np.argmax(hm.covars_.ravel())); pc = pd.Series(hm.predict_proba(ew.values[:, None])[:, cr], index=ew.index).loc["2007":]
    cum = ew.loc["2007":].cumsum()
    D["hmm"] = dict(d=[f"{d:%Y-%m-%d}" for d in pc.index], p=r(pc.values, 3), cum=r(cum.values, 1),
                    top={str(k): round(v, 2) for k, v in pc.groupby(pc.index.year).mean().nlargest(4).items()})
    return D


def slide_pngs(n_expected=20):
    pptx = HERE.parent / "원자재_퀀트_MI_발표_v4.pptx"
    tmp = Path(tempfile.mkdtemp()); src = tmp / "deck.pptx"; shutil.copy(pptx, src)
    subprocess.run([str(SOFFICE), "--headless", "--convert-to", "pdf", "--outdir", str(tmp), str(src)], check=True, capture_output=True)
    # PDF → JPG: pymupdf가 .venv에 없으면 시스템 python으로
    code = ("import pymupdf,sys,pathlib;t=pathlib.Path(sys.argv[1]);d=pymupdf.open(t/'deck.pdf')\n"
            "for i,p in enumerate(d,1): (t/f's{i:02d}.jpg').write_bytes(p.get_pixmap(dpi=144).tobytes('jpeg',jpg_quality=88))")
    try:
        import pymupdf  # noqa: F401
        py = sys.executable
    except ImportError:
        py = shutil.which("python")
    subprocess.run([py, "-c", code, str(tmp)], check=True)
    jp = sorted(tmp.glob("s*.jpg")); assert len(jp) == n_expected, f"pptx 장 수 {len(jp)} ≠ {n_expected}"
    return [base64.b64encode(p.read_bytes()).decode() for p in jp]


def notes():
    js = "const n=require('./notes_v4.js');console.log(JSON.stringify(n.timed))"
    return json.loads(subprocess.run([shutil.which("node"), "-e", js], cwd=HERE, capture_output=True, text=True, encoding="utf-8", check=True).stdout)


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main():
    sys.stdout.reconfigure(errors="replace")
    print("데이터 계산…", flush=True); D = data()
    print("pptx → 장 그림…", flush=True); imgs = slide_pngs()
    N = notes(); tpl = (HERE / "html_v4_template.html").read_text(encoding="utf-8"); native = (HERE / "html_v4_slides.html").read_text(encoding="utf-8")
    blocks = {}
    for part in native.split("<!--@")[1:]:
        key, body = part.split("-->", 1); blocks[key.strip()] = body.strip()
    secs = []
    for i, b64 in enumerate(imgs, 1):
        sec, txt = N[i - 1]; note = f'<aside class="notes">{esc(f"({sec}초) " if sec else "")}{esc(txt)}</aside>'
        main_ = (f'<section data-slide="{i}" data-chart="{NATIVE[i]}">{blocks[NATIVE[i]]}{note}</section>' if i in NATIVE else
                 f'<section data-slide="{i}" class="img-slide"><img class="full" src="data:image/jpeg;base64,{b64}" alt="{i}장">{note}'
                 + (f'<div class="drill-hint">↓ 아래로: 인터랙티브 상세</div>' if i in DRILL else "") + "</section>")
        if i in DRILL:
            subs = "".join(f'<section data-chart="{k}">{blocks[k]}</section>' for k in DRILL[i])
            secs.append(f"<section>{main_}{subs}</section>")
        else:
            secs.append(main_)
    rd = lambda p: (NM / p).read_text(encoding="utf-8").replace("</script", "<\\/script")     # 인라인 스크립트 안의 </script 차단
    html = (tpl.replace("/*@REVEAL_CSS*/", rd("reveal.js/dist/reveal.css")).replace("/*@REVEAL_JS*/", rd("reveal.js/dist/reveal.js"))
            .replace("/*@NOTES_JS*/", rd("reveal.js/plugin/notes/notes.js")).replace("/*@PLOTLY_JS*/", rd("plotly.js-dist-min/plotly.min.js"))
            .replace("/*@DATA*/", "const D = " + json.dumps(D, ensure_ascii=False) + ";").replace("/*@APP_JS*/", (HERE / "html_v4_app.js").read_text(encoding="utf-8"))
            .replace("<!--@SLIDES-->", "\n".join(secs)))
    OUT.write_text(html, encoding="utf-8")
    print(f"완료: {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB, 자료 기준일 {D['asof']})")


if __name__ == "__main__":
    main()
