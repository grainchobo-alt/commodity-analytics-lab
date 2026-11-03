"""LLM 도구층 — 대시보드 분석 함수를 '작고 검증 가능한 JSON'으로 돌려주는 도구 모음 (MCP 서버·LangGraph가 공통으로 사용).

원칙 (docs/LLM_설계.md §3):
  · 숫자는 여기서만 만든다 — LLM은 계산하지 않고, 이 결과를 인용만 한다.
  · 모든 결과에 asof(자료 기준일) · method(방법) · source(함수) 를 붙인다 → 답변에 출처·기준일 표시, 숫자 대조 검증.
  · 인터넷 접근 없음 — 로컬 저장소(data/)만 읽는다. 외부 수집은 배치 수집기(collect.py 등)의 일.
  · 결과는 작게 — 소형 로컬 LLM(4B)의 문맥에 들어가도록 표는 최대 MAX_ROWS행, 소수 3자리.
  · 쓰기 도구는 save_expert_view 하나 — data/expert/(git 제외, 사내 정보)에만 추가 기록.
LLM 없이 실행: .venv\\Scripts\\python.exe -m llm.tools  (도구 목록 · 옥수수 예시 출력)
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analytics as A  # noqa: E402
import procurement as PR  # noqa: E402
import risk as RK  # noqa: E402

warnings.filterwarnings("ignore")
DATA = ROOT / "data"
EXPERT = DATA / "expert" / "views.csv"           # 전문가 의견 (사내 정보 → .gitignore)
CHARTS = DATA / "llm_charts"                      # 도구가 만든 그림 (HTML)
MAX_ROWS = 30
RESULT_FILES = {p.stem.removesuffix("_compare"): p for p in sorted(DATA.glob("*_compare.csv"))}   # 실험 결과 표 (읽기 전용)
RESULT_FILES |= {p.stem: p for p in sorted(DATA.glob("causal_*.csv"))}


# ============================================================================
# 공통
# ============================================================================
def _mtime() -> float:
    return max(p.stat().st_mtime for p in DATA.glob("daily*.parquet"))


@lru_cache(maxsize=2)
def _levels(_stamp: float, adjusted: bool) -> pd.DataFrame:
    return A.load_levels(roll_adjusted=adjusted)


def levels(adjusted: bool = True) -> pd.DataFrame:
    """분석용 = 롤 보정(adjusted=True), 가격 수준 표시용 = 정제본. 파일이 바뀌면 캐시도 새로."""
    return _levels(_mtime(), adjusted)


@lru_cache(maxsize=1)
def _catalog(_stamp: float) -> pd.DataFrame:
    return A.load_catalog()


def resolve(item: str) -> str:
    """'옥수수' · 'Corn' · 'yf:ZC=F' 모두 id로. 못 찾으면 비슷한 후보를 담아 ValueError."""
    cat = _catalog(_mtime())
    s = str(item).strip()
    if s in set(cat.id):
        return s
    for i, nm in PR.BUY_IDS.items():
        if s == nm or s == nm.split(" (")[0]:
            return i
    low = s.lower()
    hit = cat[cat.name.str.lower() == low]
    if len(hit):
        return hit.id.iloc[0]
    cand = [f"{i} {nm}" for i, nm in PR.BUY_IDS.items() if s in nm] + [f"{r.id} {r['name']}" for _, r in cat.iterrows() if low in str(r['name']).lower()]
    raise ValueError(f"품목 '{item}'을 찾지 못함" + (f" — 후보: {', '.join(cand[:8])}" if cand else " — list_items로 확인"))


def name_of(i: str) -> str:
    if i in PR.BUY_IDS:
        return PR.BUY_IDS[i]
    cat = _catalog(_mtime())
    return str(cat.set_index("id")["name"].get(i, i))


def _clean(x):
    """JSON으로 바꿀 수 있게 (numpy·Timestamp·NaN 정리, 소수 3자리)."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (pd.Timestamp, datetime)) or hasattr(x, "isoformat"):
        return x.isoformat()[:10]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), 3)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _table(df: pd.DataFrame, n: int = MAX_ROWS) -> list[dict]:
    return _clean(df.head(n).reset_index().to_dict("records"))


def _out(data: dict, asof, method: str, source: str, note: str = "") -> dict:
    o = {"asof": asof, "method": method, "source": source, **data}
    if note:
        o["note"] = note
    return _clean(o)


def _ret(i: str, start: str | None = None) -> pd.Series:
    r = RK.returns(levels()[i])
    return r.loc[start:] if start else r


# ============================================================================
# ① 수치 데이터 — 조회
# ============================================================================
def list_items(group: str | None = None) -> dict:
    """품목 목록 (id · 이름 · 그룹 · 단위 · 기간). group으로 거르기 (예: 'Grains', 'Energy')."""
    cat = _catalog(_mtime())
    if group:
        cat = cat[cat.group.str.contains(group, case=False, na=False)]
    cat = cat.assign(구매판단=cat.id.isin(PR.BUY_IDS))
    return _out({"n": len(cat), "items": _table(cat[["id", "name", "group", "unit", "start", "end", "구매판단"]].set_index("id"), 100)},
                cat.end.max() if len(cat) else None, "자료 목록", "analytics.load_catalog")


def get_prices(item: str, start: str = "2024-01-01", freq: str = "W") -> dict:
    """가격 요약 + 최근 값 (freq: D 일간 · W 주간 · M 월간). 가격 수준은 정제본(롤 보정 아님)."""
    i = resolve(item)
    p = levels(adjusted=False)[i].dropna().loc[start:]
    if not len(p):
        raise ValueError(f"{start} 이후 {i} 가격 없음")
    q = {"D": p, "W": p.resample("W-FRI").last(), "M": p.resample("ME").last()}[freq.upper()[0]].dropna()
    r1 = lambda k: float(p.iloc[-1] / p.loc[:p.index[-1] - pd.DateOffset(**k)].iloc[-1] - 1) * 100 if len(p.loc[:p.index[-1] - pd.DateOffset(**k)]) else None   # noqa: E731
    stats = {"현재": p.iloc[-1], "기간 최저": p.min(), "기간 최고": p.max(), "기간 평균": p.mean(),
             "1주 변화(%)": r1({"weeks": 1}), "1개월 변화(%)": r1({"months": 1}), "1년 변화(%)": r1({"years": 1}),
             "기간 내 분위(%)": float((p <= p.iloc[-1]).mean()) * 100}
    unit = _catalog(_mtime()).set_index("id").unit.get(i, "")
    return _out({"item": name_of(i), "id": i, "unit": unit, "stats": stats,
                 "recent": [{"date": d, "price": v} for d, v in q.tail(MAX_ROWS).items()]},
                p.index[-1], f"정제 가격 ({freq})", "analytics.load_levels(roll_adjusted=False)")


def get_results(name: str | None = None, query: str | None = None) -> dict:
    """백테스트·실험 결과 표 (data/*_compare.csv, causal_*.csv) 읽기. name 없으면 목록. query = 행 필터 (pandas query 문법 아님, 단순 포함 검색)."""
    if not name:
        return _out({"available": sorted(RESULT_FILES)}, None, "실험 결과 목록", "data/*_compare.csv")
    if name not in RESULT_FILES:
        raise ValueError(f"결과 '{name}' 없음 — 후보: {', '.join(k for k in RESULT_FILES if name.split('_')[0] in k) or 'get_results()로 목록'}")
    p = RESULT_FILES[name]
    df = pd.read_csv(p, encoding="utf-8-sig")
    if query:
        df = df[df.astype(str).apply(lambda c: c.str.contains(query, case=False, regex=False)).any(axis=1)]
    return _out({"file": p.name, "rows": len(df), "columns": list(df.columns), "table": _clean(df.head(MAX_ROWS).to_dict("records"))},
                datetime.fromtimestamp(p.stat().st_mtime), "저장된 실험 결과", p.name)


# ============================================================================
# ① 수치 데이터 — 변동성 · 위험 · 레짐 · 요인
# ============================================================================
def vol_forecast(item: str, model: str = "GJR", horizon: int = 20, start: str = "2010-01-01") -> dict:
    """GARCH 계열 변동성 (model: GARCH · GJR · EGARCH, Student-t) + RiskMetrics EWMA 비교. 연율 %.
    지속성(persistence)·반감기·비대칭 계수(gamma: GJR>0 = 하락 충격이 변동성을 더 키움)와 horizon일 앞 예측."""
    i = resolve(item)
    r = _ret(i, start)
    pt, vol, info, res = A.garch_fit(r, kind=model.upper() if model.upper() != "GJR-GARCH" else "GJR", dist="t")
    how = dict(method="simulation", simulations=1000) if model.upper() == "EGARCH" and horizon > 1 else {}   # EGARCH는 다기간 해석해 없음
    f = res.forecast(horizon=horizon, reindex=False, **how).variance.values[-1]
    ew = RK.ewma_var(r).iloc[-1]
    return _out({"item": name_of(i), "id": i, "model": model,
                 "params": _table(pt.round(4)),
                 "persistence": info["persistence"], "half_life_days": info["half_life"], "gamma": info["gamma"], "gamma_p": info["gamma_p"],
                 "vol_now_ann_pct": vol.iloc[-1], "vol_1y_avg_ann_pct": vol.iloc[-252:].mean(), "vol_10y_median_ann_pct": vol.iloc[-2520:].median(),
                 f"vol_next_{horizon}d_avg_ann_pct": float(np.sqrt(f.mean() * 252)),
                 "riskmetrics_ann_pct": float(np.sqrt(ew * 252))},
                r.index[-1], f"{model}(1,1) Student-t, 일간 로그수익률 %, {start}~", "analytics.garch_fit · risk.ewma_var",
                "연율 = 일간 σ × √252. gamma_p < 0.05 이면 비대칭이 통계적으로 유의.")


def var_es(item: str, alpha: float = 0.99, horizon: int = 20, side: str = "long", notional_usd: float | None = None,
           start: str = "2008-01-01") -> dict:
    """h일 VaR·ES (손실 %, 양수) — 정규·Student-t·FHS·EVT·역사적 5방법 + 과거 백테스트(초과율·Kupiec·바젤 신호등).
    side: long = 가격 하락이 손실(보유), short = 가격 상승이 손실(매입 예정 물량). notional_usd 주면 금액도."""
    i = resolve(item)
    r = _ret(i, "2001-01-01")
    s2 = pd.Series(RK.ewma_path(r.values, RK.RM_LAMBDA), index=r.index)
    V = RK.var_forecasts(r, s2, alpha=alpha, side=side, h=horizon, start=start)
    s2_next = RK.RM_LAMBDA * s2.iloc[-1] + (1 - RK.RM_LAMBDA) * r.iloc[-1] ** 2
    now = RK.var_now(V, r, s2_next)
    if notional_usd:
        now["VaR (USD)"] = now["VaR (%)"] / 100 * notional_usd
        now["ES (USD)"] = now["ES (%)"] / 100 * notional_usd
    bt = RK.backtest_table(V)[["평가일수", "초과율", "Kupiec p (적중률)", "조건부 적중 p", "바젤 신호등 (99%)"]]
    return _out({"item": name_of(i), "id": i, "alpha": alpha, "horizon_days": horizon, "side": side,
                 "now": _table(now), "backtest": _table(bt), "expected_exceed_rate": 1 - alpha},
                r.index[-1], f"RiskMetrics σ(λ=0.94) × 표준화 손실 분위, {horizon}일, {side}", "risk.var_forecasts · var_now · backtest_table",
                "초과율이 기대(1−α)에 가깝고 Kupiec p > 0.05면 그 방법이 과거에 맞았음.")


def regime(item: str, k: int = 2, freq: str = "W", start: str = "2008-01-01") -> dict:
    """Markov 레짐 스위칭(Hamilton): 레짐별 평균·변동성·지속기간 + 최근 레짐 확률. 레짐1 = 가장 잔잔한 국면.
    ※ 평활 확률(전체 표본 사용) — 설명용. 실시간 판단용 필터 확률은 대시보드 ③."""
    i = resolve(item)
    p = levels()[i].dropna().loc[start:]
    if freq.upper().startswith("W"):
        p = p.resample("W-FRI").last()
    r = (np.log(p).diff() * 100).dropna()
    prob, summ = A.markov_switching(r, k=k)
    cur = prob.iloc[-1]
    return _out({"item": name_of(i), "id": i, "k": k, "summary": _table(summ),
                 "current": {c: float(v) for c, v in cur.items()}, "current_regime": cur.idxmax(),
                 "recent_probs": _table(prob.tail(12))},
                r.index[-1], f"MarkovRegression(k={k}, 평균·분산 전환), {freq} 로그수익률 %", "analytics.markov_switching",
                "평활 확률은 미래 자료도 써서 구간을 사후에 나눈 것 — 예측 신호로 쓰지 말 것.")


def factors(items: list[str] | None = None, k: int = 3, weeks: int = 260) -> dict:
    """요인 구조: PCA(공통 요인 몫·적재값) + Diebold-Yilmaz 스필오버(총 연계도·순 전이). 기본 = 곡물 6 + 원유·금·구리."""
    ids = [resolve(x) for x in items] if items else PR.GRAIN_IDS + ["yf:CL=F", "yf:GC=F", "yf:HG=F"]
    P = levels()[ids].resample("W-FRI").last().dropna().iloc[-weeks - 1:]
    R = (np.log(P).diff() * 100).dropna()
    ev, load, _ = A.pca(R, k)
    tbl, total = A.dy_spillover(R, p=1, H=10)
    net = tbl.loc["NET (TO-FROM)"].drop("FROM (타 품목→)").rename(index=name_of).sort_values(ascending=False)
    return _out({"items": [name_of(i) for i in ids], "weeks": len(R),
                 "explained_ratio": {f"PC{j+1}": float(ev[j]) for j in range(k)},
                 "loadings": _table(load.rename(index=name_of)),
                 "spillover_total_pct": total, "net_transmitter_pct": {k_: float(v) for k_, v in net.items()}},
                R.index[-1], f"주간 로그수익률 {len(R)}주: 상관행렬 PCA + VAR(1) GFEVD 10주", "analytics.pca · analytics.dy_spillover",
                "PC1 몫이 크면 품목들이 한 요인으로 같이 움직임. 순 전이 + = 다른 품목에 충격을 주는 쪽.")


def correlation(items: list[str], weeks: int = 104) -> dict:
    """주간 수익률 상관 (최근 weeks주 vs 그 이전 같은 길이) — 동조성이 커졌는지."""
    ids = [resolve(x) for x in items]
    P = levels()[ids].resample("W-FRI").last().dropna()
    R = (np.log(P).diff() * 100).dropna().rename(columns=name_of)
    now, before = R.iloc[-weeks:].corr(), R.iloc[-2 * weeks:-weeks].corr()
    return _out({"weeks": weeks, "recent": _table(now.round(3)), "previous": _table(before.round(3)),
                 "avg_pair_recent": float(now.values[np.triu_indices(len(ids), 1)].mean()),
                 "avg_pair_previous": float(before.values[np.triu_indices(len(ids), 1)].mean())},
                R.index[-1], "주간 로그수익률 피어슨 상관", "pandas.corr")


# ============================================================================
# ① → 의사결정 — 구매 판단 · 펀더멘털
# ============================================================================
def buy_signal(items: list[str] | None = None) -> dict:
    """이번 주 '지금 살까' 판단 (대시보드 ★ 탭과 같은 계산): 매수 점수 0~100, 구성요소, 4주 90% 범위, 권장 속도."""
    ids = [resolve(x) for x in items] if items else ["yf:ZC=F", "yf:ZW=F", "yf:ZS=F", "yf:ZM=F"]
    T = PR.today_table(levels(), levels(adjusted=False), ids)
    keep = ["품목", "기준일", "단위", "가격", "매수 점수", "위험 지표", "연 변동성", "4주 하단 90%", "4주 상단 90%", "권장 속도 (균등 대비 배)", "상대가치 지표"]
    keep += [c for c in PR.COMP_LBL.values() if c in T]
    return _out({"table": _clean(T[[c for c in keep if c in T]].to_dict("records"))},
                T["기준일"].max() if len(T) else None, "매수 점수 = 가격 위치·단기 반등·상대가치·계절성 가중 평균", "procurement.today_table",
                "점수 ≥ 60 적극 매수 · 40~60 판단 유보 · < 40 매수 보류(대시보드 기준). 근거 품목 외에는 참고용.")


def fundamentals(items: list[str] | None = None) -> dict:
    """최근 WASDE 재고/사용비율·전월 대비·같은 달 과거 분위 + CFTC 투기 순포지션 (점수에는 안 들어가는 참고 정보)."""
    ids = [resolve(x) for x in items] if items else ["yf:ZC=F", "yf:ZW=F", "yf:ZS=F", "yf:ZM=F"]
    F = PR.fundamentals(ids)
    return _out({"table": _clean(F.to_dict("records"))}, None, "USDA WASDE · CFTC COT (발표일 기준)", "procurement.fundamentals",
                "공개 수급 정보는 발표 즉시 가격에 반영 → 방향 예측력 없음(별첨 5). 맥락 설명용.")


# ============================================================================
# ③ 온톨로지 — 연결 · 역추적
# ============================================================================
def ontology_graph(version: int = 2) -> dict:
    """곡물 2단 온톨로지 노드·연결 + 추정 계수(전체 · 2008–16 · 2017–, 두 기간 같은 부호 여부)."""
    import causal_grain as CG
    roots, pa = CG.GRAPHS[version]
    edges = [{"from": p, "to": c} for c, ps in pa.items() for p in ps]
    est = DATA / "causal_scm_edges.csv"
    if est.exists():
        E = pd.read_csv(est, encoding="utf-8-sig").set_index("연결")
        for e in edges:
            k = f"{e['from']} → {e['to']}"
            if k in E.index:
                e.update(E.loc[k].to_dict())
    return _out({"version": version, "roots": roots, "nodes": list(pa), "edges": edges},
                datetime.fromtimestamp(est.stat().st_mtime) if est.exists() else None, "선형 SCM (주간, 같은 주 부모 + 1주 시차)",
                "causal_grain.GRAPHS · data/causal_scm_edges.csv", "계수 = 원인 1단위(표준화) 변화 시 결과 주간 변동(%). t 절댓값 2 이상이 유의.")


@lru_cache(maxsize=1)
def _scm(_stamp: float):
    import causal_grain as CG
    CG.set_graph(1)
    X = CG.nodes(levels())
    M = CG.fit_scm(X)
    return CG, X, M, CG.decompose(M, X)


def ontology_backtrace(target: str, start: str, end: str) -> dict:
    """결과 → 원인 역추적: target(옥수수·대두·밀·대두박·대두유)의 start~end 누적 주간 변동(%)을 원인 충격·경로별 기여로 분해.
    분해는 정확(합 = 실제). 단, '설명'이지 예측이 아님."""
    CG, X, M, D = _scm(_mtime())
    if target not in CG.PRICES:
        raise ValueError(f"target은 {CG.PRICES} 중 하나")
    rows = CG.backtrace(M, D, X, target, start, end, "질의")
    return _out({"target": target, "period": f"{start} ~ {end}", "contributions": _clean(rows[:12])},
                X.index[-1], "선형 SCM 역사적 분해(1판 온톨로지)", "causal_grain.backtrace",
                "기여 합 = 실제 누적 변동. '설명 안 됨(자기 충격)'이 크면 온톨로지 밖 요인(뉴스·수급 심리 등).")


# ============================================================================
# ② 전문가 의견 — 쓰기 (승인 필요) · 읽기
# ============================================================================
EXPERT_COLS = ["recorded_at", "date", "item", "id", "author", "direction", "confidence", "horizon_weeks", "text", "source"]


def save_expert_view(item: str, direction: float, confidence: float, text: str, author: str,
                     date: str | None = None, horizon_weeks: int = 4, source: str = "") -> dict:
    """[쓰기] 전문가 시황 의견 1건 저장 — direction −1(약세)~+1(강세), confidence 0~1. data/expert/views.csv(사내, git 제외)에 추가만.
    나중에 백테스트(의견 → 이후 horizon_weeks 수익률)로 검증한 뒤에만 판단에 반영."""
    i = resolve(item)
    if not -1 <= float(direction) <= 1 or not 0 <= float(confidence) <= 1:
        raise ValueError("direction은 −1~+1, confidence는 0~1")
    if not str(text).strip() or not str(author).strip():
        raise ValueError("text·author는 비울 수 없음")
    row = dict(recorded_at=datetime.now().isoformat(timespec="seconds"), date=date or datetime.now().date().isoformat(), item=name_of(i), id=i,
               author=author.strip(), direction=float(direction), confidence=float(confidence), horizon_weeks=int(horizon_weeks),
               text=str(text).strip()[:2000], source=source)
    EXPERT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=EXPERT_COLS).to_csv(EXPERT, mode="a", header=not EXPERT.exists(), index=False, encoding="utf-8-sig")
    n = len(pd.read_csv(EXPERT, encoding="utf-8-sig"))
    return _out({"saved": row, "total_views": n}, row["date"], "추가 기록", "data/expert/views.csv")


def expert_views(item: str | None = None, last: int = 20) -> dict:
    """저장된 전문가 의견 조회 (최근 순)."""
    if not EXPERT.exists():
        return _out({"views": [], "total_views": 0}, None, "없음", "data/expert/views.csv")
    V = pd.read_csv(EXPERT, encoding="utf-8-sig")
    if item:
        V = V[V.id == resolve(item)]
    V = V.sort_values("date", ascending=False)
    return _out({"views": _clean(V.head(last).to_dict("records")), "total_views": len(V)},
                V.date.max() if len(V) else None, "저장된 의견", "data/expert/views.csv")


# ============================================================================
# 시각화 — Plotly HTML (로컬 파일). LLM은 경로만 받아 사용자에게 안내
# ============================================================================
def make_chart(kind: str, item: str, start: str = "2018-01-01") -> dict:
    """그림 생성 → data/llm_charts/*.html. kind: price(가격) · vol(GJR 변동성 vs EWMA) · regime(레짐 확률) · var(20일 99% VaR vs 실현손실)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    i = resolve(item)
    nm = name_of(i)
    kind = kind.lower()
    if kind == "price":
        p = levels(adjusted=False)[i].dropna().loc[start:]
        fig = go.Figure(go.Scatter(x=p.index, y=p.values, name=nm, line=dict(width=1.6)))
        title = f"{nm} 가격"
    elif kind == "vol":
        r = _ret(i, "2010-01-01")
        _, vol, _, _ = A.garch_fit(r, "GJR")
        ew = np.sqrt(RK.ewma_var(r) * 252)
        fig = go.Figure([go.Scatter(x=vol.loc[start:].index, y=vol.loc[start:].values, name="GJR-GARCH"),
                         go.Scatter(x=ew.loc[start:].index, y=ew.loc[start:].values, name="EWMA λ=0.94", line=dict(dash="dot"))])
        title = f"{nm} 연 변동성 (%)"
    elif kind == "regime":
        o = regime(i, 2, "W", "2008-01-01")
        p = levels()[i].dropna().loc["2008":].resample("W-FRI").last()
        r = (np.log(p).diff() * 100).dropna()
        prob, _ = A.markov_switching(r, 2)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4])
        fig.add_trace(go.Scatter(x=p.index, y=p.values, name=nm), 1, 1)
        fig.add_trace(go.Scatter(x=prob.index, y=prob["레짐2"], name="고변동 레짐 확률", fill="tozeroy"), 2, 1)
        title = f"{nm} 레짐 (현재: {o['current_regime']})"
    elif kind == "var":
        r = _ret(i, "2001-01-01")
        s2 = pd.Series(RK.ewma_path(r.values, RK.RM_LAMBDA), index=r.index)
        T = RK.var_forecasts(r, s2, 0.99, "long", 20, start=start)["table"]
        fig = go.Figure([go.Bar(x=T.index, y=T["실현손실"], name="실현 20일 손실 (%)", marker_color="#999"),
                         go.Scatter(x=T.index, y=T["VaR FHS (필터링 역사적)"], name="99% VaR (FHS)", line=dict(color="#D9582B"))])
        title = f"{nm} 20일 99% VaR 백테스트"
    else:
        raise ValueError("kind는 price · vol · regime · var")
    fig.update_layout(title=title, template="plotly_white", height=520, legend=dict(orientation="h"))
    CHARTS.mkdir(parents=True, exist_ok=True)
    out = CHARTS / f"{kind}_{i.replace(':', '_').replace('=', '')}_{datetime.now():%Y%m%d_%H%M%S}.html"
    fig.write_html(out, include_plotlyjs="cdn")
    return _out({"path": str(out), "title": title}, None, "Plotly HTML", "plotly.graph_objects")


# ============================================================================
# 등록부 — MCP 서버 · LangGraph가 이 목록만 노출 (읽기/쓰기 구분)
# ============================================================================
READ_TOOLS = [list_items, get_prices, get_results, vol_forecast, var_es, regime, factors, correlation,
              buy_signal, fundamentals, ontology_graph, ontology_backtrace, expert_views, make_chart]
WRITE_TOOLS = [save_expert_view]


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    for f in READ_TOOLS + WRITE_TOOLS:
        print(f"{'[쓰기]' if f in WRITE_TOOLS else '      '} {f.__name__}: {f.__doc__.strip().splitlines()[0]}")
    print(json.dumps(vol_forecast("옥수수"), ensure_ascii=False, indent=1)[:1500])
