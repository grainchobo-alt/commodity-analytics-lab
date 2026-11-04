"""별첨 9 · 10 (3축 신호 · 위험 한도 실험) — HTML 발표용 실시간 3D 자료 → img/sig.json

PPT는 docs/실험요약/figs의 정적 그림을 그대로 쓰고, html_v4.py는 같은 자리에 아래 4개를 Plotly 3D로 그림.
  flow : 판단 규칙 지형 — 곡물 26주 상관 분위 × 직전 13주 곡물 수익 × '추세 집중 켜짐' 고원 + 2008–2026 매주 실제 점 (2022 · 2025–26 강조)
  cases: 2021-07 ~ 2026-10 품목 누적 수익 리본 (옥수수 · 밀 · 대두 · WTI · 금 · 은 · 기준 혼합)
  res  : 실험별 규칙 점 (학습 Sharpe × 실험 × 시험 Sharpe) + 기준 1.07 평면
  gcap : 품목군 상한별 누적 수익 리본 (없음 · 곡물 25% · 3품목군 25%)
숫자 원천 = data/strategy_*_compare.csv · strategy_groupcap_weekly.parquet (모두 2026-10-26 고정 결과).
실행 (프로젝트 루트): .venv\\Scripts\\python.exe docs\\발표\\scripts\\charts_sig.py
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import analytics as A  # noqa: E402
import ls_strategies as LS  # noqa: E402
import strategy_crisis_mom as CM  # noqa: E402

FROZEN_END = "2026-10-26"
OUT = HERE / "img" / "sig.json"
r2 = lambda v: None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), 3)  # noqa: E731


def flow(L):
    P = LS.weekly(L, CM.GRAINS, "2001-01-01"); R = LS._ret(P)
    idx = R.index[R.index >= "2008-01-01"]
    pct = CM.grain_corr_pct(R, idx)
    up = R[CM.GRAINS].mean(1).rolling(13).sum().reindex(idx)
    d = pd.DataFrame({"p": pct, "u": up}).dropna()
    tag = np.where((d.index >= "2022-01-01") & (d.index <= "2023-06-30"), 1, np.where(d.index >= "2025-01-01", 2, 0))
    on = ((d.p > 0.9) & (d.u > 0)).astype(int)
    return {"d": [f"{x:%Y-%m-%d}" for x in d.index], "p": [r2(v) for v in d.p], "u": [r2(v) for v in d.u], "on": on.tolist(), "tag": tag.tolist(),
            "thr": 0.9, "on_pct": round(float(on.mean() * 100), 1), "on22": int(on[tag == 1].sum())}


def cases(L):
    ids = {"옥수수": "yf:ZC=F", "밀": "yf:ZW=F", "대두": "yf:ZS=F", "WTI 원유": "yf:CL=F", "금": "yf:GC=F", "은": "yf:SI=F"}
    P = LS.weekly(L, list(ids.values()), "2021-01-01").loc["2021-07-01":FROZEN_END]
    C = (100 * np.log(P / P.iloc[0]))
    base = pd.read_parquet(ROOT / "data" / "strategy_groupcap_weekly.parquet")["없음 (기준)"].reindex(P.index).fillna(0).cumsum()
    base = base - base.iloc[0]
    out = {"d": [f"{x:%Y-%m-%d}" for x in P.index], "names": list(ids) + ["기준 혼합 50/25/25"],
           "z": [[r2(v) for v in C[i]] for i in ids.values()] + [[r2(v) for v in base]]}
    return out


def res():
    D = ROOT / "data"
    rd = lambda n: pd.read_csv(D / n, encoding="utf-8-sig")  # noqa: E731
    spec = [("A 연도별 갈아타기", "strategy_switch_compare.csv", "규칙", "Sharpe 학습 2012–16", "Sharpe 시험 2017–"),
            ("B 다음 분기 1등 축", "strategy_meta_compare.csv", "규칙", "Sharpe 학습 2008–16", "Sharpe 시험 2017–"),
            ("C 칼만·HMM·뉴스·DCC", "strategy_adaptive_compare.csv", "규칙", "Sharpe 학습 2008–16", "Sharpe 시험 2017–"),
            ("D 현금 피신", "strategy_cash_compare.csv", "신호", "Sharpe 학습 2008–16", "Sharpe 시험 2017–"),
            ("F 상관↑ 추세 집중", "strategy_crisis_mom_compare.csv", "규칙", "Sharpe 학습 2008–16", "Sharpe 시험 2017–"),
            ("J 짧은 추세 4·8주", "strategy_short_trend_compare.csv", "판", "Sharpe 학습 2008–16", "Sharpe 시험 2017–"),
            ("K 모멘텀 기간", "strategy_mom_lookback_compare.csv", "모멘텀 기간", "혼합 학습 2008–16", "혼합 시험 2017–"),
            ("G 품목군 상한", "strategy_groupcap_compare.csv", "상한", "Sharpe 학습 2008–16", "Sharpe 시험 2017–")]
    pts = []
    for g, (lab, f, nc, lc, tc) in enumerate(spec):
        if not (D / f).exists():
            continue
        T = rd(f)
        for _, row in T.iterrows():
            nm = str(row[nc]); extra = " · ".join(str(row[c]) for c in ("문턱", "λ·q") if c in T and pd.notna(row.get(c)) and str(row[c]) not in ("", "nan"))
            if "기준" in nm or nm.startswith("고정 50/25/25"):
                continue
            l_, t_ = r2(row[lc]), r2(row[tc])
            if l_ is None or t_ is None:
                continue
            best = int(any(str(row[c]).startswith("←") for c in ("학습 최고", "설정") if c in T and pd.notna(row.get(c))))
            pts.append({"g": g, "x": l_, "z": t_, "b": best, "n": (nm + (f" ({extra})" if extra else ""))[:80].replace("← ", "")})
    bests = [p for p in pts if p["b"]]
    return {"groups": [s[0] for s in spec], "pts": pts, "best_max": r2(max([p["z"] for p in bests], default=np.nan)), "base": 1.067, "n_above": int(sum(p["z"] > 1.067 + 0.02 for p in pts)), "n": len(pts)}


def gcap():
    W = pd.read_parquet(ROOT / "data" / "strategy_groupcap_weekly.parquet").loc["2008":]
    cols = {"없음": "없음 (기준)", "곡물 25%": "곡물 25%", "3품목군 25%": "곡물 25% + 에너지 25% + 축산 25%"}
    C = pd.read_csv(ROOT / "data" / "strategy_groupcap_compare.csv", encoding="utf-8-sig").set_index("상한")
    Wm = W.resample("ME").sum()
    return {"d": [f"{x:%Y-%m}" for x in Wm.index], "names": list(cols), "z": [[r2(v) for v in Wm[c].cumsum()] for c in cols.values()],
            "sh": [r2(C.loc[c, "Sharpe 시험 2017–"]) for c in cols.values()], "var": [r2(C.loc[c, "지금 20일 99% VaR (%)"]) for c in cols.values()],
            "vg": [r2(C.loc[c, "VaR 기여 곡물 (%)"]) for c in cols.values()], "mdd": [r2(C.loc[c, "최대 낙폭 (%)"]) for c in cols.values()]}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    L = A.load_levels(roll_adjusted=True).loc[:FROZEN_END]
    S = {"flow": flow(L), "cases": cases(L), "res": res(), "gcap": gcap()}
    OUT.write_text(json.dumps(S, ensure_ascii=False), encoding="utf-8")
    f, r = S["flow"], S["res"]
    print(f"저장 {OUT} ({OUT.stat().st_size / 1e3:.0f} KB) · 규칙 켜진 주 {f['on_pct']}% · 2022–23 켜진 주 {f['on22']} · 실험 점 {r['n']}개 중 기준 위 {r['n_above']}개")


if __name__ == "__main__":
    main()
