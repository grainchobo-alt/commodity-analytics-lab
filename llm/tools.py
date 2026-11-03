"""LLM 도구층 — 대시보드 분석 함수를 '작고 검증 가능한 JSON'으로 돌려주는 도구 모음 (MCP 서버·LangGraph가 공통으로 사용).

원칙 (docs/LLM_설계.md §3):
  · 숫자는 여기서만 만든다 — LLM은 계산하지 않고, 이 결과를 인용만 한다.
  · 모든 결과에 asof(자료 기준일) · method(방법) · source(함수) 를 붙인다 → 답변에 출처·기준일 표시, 숫자 대조 검증.
  · 읽기 도구는 인터넷 접근 없음 — 로컬 저장소(data/)만 읽는다. 외부 수집은 refresh_data(쓰기) 하나뿐이고, confirm=True(사람 승인)일 때만 실행.
  · 대시보드와 같은 숫자 — 도구마다 대응 탭·일치 여부를 DASH에 적고 결과에 dashboard로 붙임. "같음"은 tests/test_llm_parity.py가 검증.
  · 결과는 작게 — 소형 로컬 LLM(4B)의 문맥에 들어가도록 표는 최대 MAX_ROWS행, 소수 3자리.
  · 쓰기 도구 2개 — save_expert_view(data/expert/에 추가, 사내 정보) · refresh_data(외부 수집, 계획 → 승인 → 실행).
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


def _panel(ids: list[str], freq: str = "W-FRI", start: str = "2000-01-01") -> pd.DataFrame:
    """대시보드 사이드바 기본값과 같은 패널: 롤 보정 가격 → 주간(금) 로그수익률 %, 2000-01-01 ~ 오늘, 공통 표본."""
    cat = _catalog(_mtime())
    return A.make_panel(levels(), ids, dict(zip(cat.id, cat.unit)), freq, "logret", start, pd.Timestamp.today().date())


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
def vol_forecast(item: str, model: str = "GJR", horizon: int = 20, dist: str = "t") -> dict:
    """GARCH 계열 변동성 (model: GARCH · GJR · EGARCH) — 대시보드 ③ '변동성 동학'과 같은 계산(주간 로그수익률 2000~, 연율 = √52).
    지속성(persistence)·반감기(주)·비대칭 계수(gamma: GJR>0 = 하락 충격이 변동성을 더 키움)와 horizon 영업일(→ 주 단위로 올림) 앞 평균 예측."""
    i = resolve(item)
    r = _panel([i])[i]
    kind = "GJR" if model.upper() in ("GJR", "GJR-GARCH") else model.upper()
    pt, vol, info, res = A.garch_fit(r, kind, dist, 52)
    hw = max(1, int(np.ceil(horizon / 5)))
    how = dict(method="simulation", simulations=1000) if kind == "EGARCH" and hw > 1 else {}   # EGARCH는 다기간 해석해 없음
    f = res.forecast(horizon=hw, reindex=False, **how).variance.values[-1]
    ew = RK.ewma_var(r).iloc[-1]
    return _out({"item": name_of(i), "id": i, "model": kind, "dist": dist,
                 "params": _table(pt.round(4)),
                 "persistence": info["persistence"], "half_life_weeks": info["half_life"], "gamma": info["gamma"], "gamma_p": info["gamma_p"],
                 "vol_now_ann_pct": vol.iloc[-1], "vol_1y_avg_ann_pct": vol.iloc[-52:].mean(), "vol_10y_median_ann_pct": vol.iloc[-520:].median(),
                 f"vol_next_{hw}w_avg_ann_pct": float(np.sqrt(f.mean() * 52)),
                 "riskmetrics_ann_pct": float(np.sqrt(ew * 52))},
                r.index[-1], f"{kind}(1,1) {dist}분포, 주간(금) 롤 보정 로그수익률 %, 2000~", "analytics.garch_fit · risk.ewma_var",
                "연율 = 주간 σ × √52. gamma_p < 0.05 이면 비대칭이 통계적으로 유의. 기준일 = 마지막 주(금요일 묶음).")


@lru_cache(maxsize=8)
def _vol_models(i: str, _stamp: float) -> dict:
    return RK.vol_models(RK.returns(levels()[i]))


def var_es(item: str, alpha: float = 0.99, horizon: int = 20, side: str = "long", notional_usd: float | None = None,
           sigma: str = "GARCH(1,1)", start: str = "2010-01-01") -> dict:
    """h일 VaR·ES (손실 %, 양수) — 대시보드 ◆ 리스크 'VaR · ES 백테스트'와 같은 계산 (σ 모형 기본 GARCH(1,1) walk-forward, 평가 시작 2010).
    정규·Student-t·FHS·EVT·역사적 5방법 + 과거 백테스트(초과율·Kupiec·바젤 신호등).
    side: long = 가격 하락이 손실(보유), short = 가격 상승이 손실(매입 예정 물량). notional_usd 주면 금액도.
    sigma: GARCH(1,1) · RiskMetrics λ=0.94 등 risk.VOL_MODELS 이름. 처음 한 번 품목당 약 10초(매년 재추정)."""
    i = resolve(item)
    o = _vol_models(i, _mtime())
    r, S = o["r"], o["S"]
    if sigma not in S:
        raise ValueError(f"sigma는 {list(S.columns)} 중 하나")
    V = RK.var_forecasts(r, S[sigma], alpha, side, horizon, start=start)
    now = RK.var_now(V, r, RK.next_var(o, sigma))
    if notional_usd:
        now["VaR (USD)"] = now["VaR (%)"] / 100 * notional_usd
        now["ES (USD)"] = now["ES (%)"] / 100 * notional_usd
    bt = RK.backtest_table(V)[["평가일수", "초과율", "Kupiec p (적중률)", "조건부 적중 p", "바젤 신호등 (99%)"]]
    return _out({"item": name_of(i), "id": i, "alpha": alpha, "horizon_days": horizon, "side": side,
                 "now": _table(now), "backtest": _table(bt), "expected_exceed_rate": 1 - alpha},
                r.index[-1], f"{sigma} σ(walk-forward) × 표준화 손실 분위, {horizon}일, {side}, 평가 {start[:4]}~", "risk.vol_models · var_forecasts · var_now · backtest_table",
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
    L = levels()[ids].dropna()
    P = L.resample("W-FRI").last().dropna().iloc[-weeks - 1:]
    R = (np.log(P).diff() * 100).dropna()
    ev, load, _ = A.pca(R, k)
    tbl, total = A.dy_spillover(R, p=1, H=10)
    net = tbl.loc["NET (TO-FROM)"].drop("FROM (타 품목→)").rename(index=name_of).sort_values(ascending=False)
    return _out({"items": [name_of(i) for i in ids], "weeks": len(R),
                 "explained_ratio": {f"PC{j+1}": float(ev[j]) for j in range(k)},
                 "loadings": _table(load.rename(index=name_of)),
                 "spillover_total_pct": total, "net_transmitter_pct": {k_: float(v) for k_, v in net.items()}},
                L.index[-1], f"주간 로그수익률 {len(R)}주: 상관행렬 PCA + VAR(1) GFEVD 10주", "analytics.pca · analytics.dy_spillover",
                "PC1 몫이 크면 품목들이 한 요인으로 같이 움직임. 순 전이 + = 다른 품목에 충격을 주는 쪽.")


def correlation(items: list[str], weeks: int = 104) -> dict:
    """주간 수익률 상관 (최근 weeks주 vs 그 이전 같은 길이) — 동조성이 커졌는지."""
    ids = [resolve(x) for x in items]
    L = levels()[ids].dropna()
    P = L.resample("W-FRI").last().dropna()
    R = (np.log(P).diff() * 100).dropna().rename(columns=name_of)
    now, before = R.iloc[-weeks:].corr(), R.iloc[-2 * weeks:-weeks].corr()
    return _out({"weeks": weeks, "recent": _table(now.round(3)), "previous": _table(before.round(3)),
                 "avg_pair_recent": float(now.values[np.triu_indices(len(ids), 1)].mean()),
                 "avg_pair_previous": float(before.values[np.triu_indices(len(ids), 1)].mean())},
                L.index[-1], "주간 로그수익률 피어슨 상관 (주 단위 = 금요일 묶음, 기준일 = 마지막 자료일)", "pandas.corr")


# ============================================================================
# ① → 의사결정 — 구매 판단 · 펀더멘털
# ============================================================================
BUY_PRESET = "추천 (고도화: 상대가치+계절+MIDAS)"          # ★ 탭 기본 프리셋


def buy_signal(items: list[str] | None = None) -> dict:
    """이번 주 '지금 살까' 판단 (대시보드 ★ 탭 기본값 = '추천' 프리셋 · 가격 위치 창 품목별 자동과 같은 계산): 매수 점수 0~100, 구성요소, 4주 90% 범위, 권장 속도."""
    ids = [resolve(x) for x in items] if items else ["yf:ZC=F", "yf:ZW=F", "yf:ZS=F", "yf:ZM=F"]
    P0 = PR.PRESETS[BUY_PRESET]
    T = pd.concat([PR.today_table(levels(), levels(adjusted=False), [i], P0["w"], P0["k"], risk=P0["risk"], season=P0["season"],
                                  lookback=PR.item_lookback(i, int(P0["lookback"]))) for i in ids], ignore_index=True)
    keep = ["품목", "기준일", "단위", "가격", "매수 점수", "위험 지표", "연 변동성", "4주 하단 90%", "4주 상단 90%", "권장 속도 (균등 대비 배)", "상대가치 지표"]
    keep += [c for c in PR.COMP_LBL.values() if c in T]
    return _out({"table": _clean(T[[c for c in keep if c in T]].to_dict("records"))},
                T["기준일"].max() if len(T) else None, f"매수 점수 = 가격 위치·단기 반등·상대가치·계절성 가중 평균 ({BUY_PRESET})", "procurement.today_table",
                "점수 ≥ 60 적극 매수 · 40~60 판단 유보 · < 40 매수 보류(대시보드 기준). 근거 품목 외에는 참고용.")


def fundamentals(items: list[str] | None = None) -> dict:
    """최근 WASDE 재고/사용비율·전월 대비·같은 달 과거 분위 + CFTC 투기 순포지션 (점수에는 안 들어가는 참고 정보)."""
    ids = [resolve(x) for x in items] if items else ["yf:ZC=F", "yf:ZW=F", "yf:ZS=F", "yf:ZM=F"]
    F = PR.fundamentals(ids)
    F = F.reset_index() if "품목" not in F.columns else F
    return _out({"table": _clean(F.to_dict("records"))}, None, "USDA WASDE · CFTC COT (발표일 기준)", "procurement.fundamentals",
                "공개 수급 정보는 발표 즉시 가격에 반영 → 방향 예측력 없음(별첨 5). 맥락 설명용.")


# ============================================================================
# ① 수치 데이터 — 다변량 GARCH · 칼만 · 머신러닝 · 포트폴리오 전략 · 텍스트 감성
# ============================================================================
MG_IDS = ["yf:ZC=F", "yf:ZS=F", "yf:ZM=F", "yf:ZW=F", "yf:CL=F", "yf:HG=F", "yf:GC=F", "ak:macro_shipping_bdi"]   # ③ 탭 기본 품목


def _pair_key(a: str, b: str) -> str:
    return f"{name_of(a)}–{name_of(b)}"


def dcc_correlation(items: list[str] | None = None, asymmetric: bool = False, model: str = "GJR", dist: str = "t") -> dict:
    """DCC(비대칭이면 ADCC) 동적 조건부 상관 — 대시보드 ③ '② DCC / ADCC'와 같은 계산(주간 2000~, 1단계 단변량 GJR-t).
    품목 쌍별 지금 상관 · 1년 전 · 전체 평균, 평균 상관의 변화. 기본 품목 = ③ 탭 기본 8개."""
    import mvgarch as MG
    ids = [resolve(x) for x in items] if items else MG_IDS
    R = _panel(ids)
    dc = MG.dcc_fit(R, asymmetric, model, dist, 52)
    P = dc["pairs"]
    rows = [{"쌍": _pair_key(a, b), "현재": s.iloc[-1], "1년 전": s.iloc[-53] if len(s) > 53 else None, "전체 평균": s.mean(),
             "현재 분위(%)": float((s <= s.iloc[-1]).mean()) * 100} for (a, b), s in P.items()]
    rows.sort(key=lambda x: -x["현재"])
    avg = dc["avg_corr"]
    return _out({"items": [name_of(i) for i in ids], "params": _table(dc["params"].round(4)), "persistence": dc["persistence"],
                 "avg_corr_now": avg.iloc[-1], "avg_corr_1y_ago": avg.iloc[-53] if len(avg) > 53 else None, "avg_corr_mean": avg.mean(),
                 "pairs": _clean(rows[:MAX_ROWS])},
                R.index[-1], f"{'ADCC' if asymmetric else 'DCC'}(1,1) · 1단계 {model}-{dist}, 주간 롤 보정 로그수익률 2000~", "mvgarch.dcc_fit",
                "지속성(a+b) 1에 가까우면 상관 변화가 오래 감. 현재 분위 = 그 쌍의 과거 상관 중 지금 위치(높을수록 동조 강함).")


def bekk_spillover(items: list[str] | None = None, kind: str = "Full") -> dict:
    """BEKK 변동성 전이(spillover) — 대시보드 ③ '③ BEKK'와 같은 계산(주간 2000~). 한 품목의 충격(A)·변동성(B)이 다른 품목으로 번지는지, t값.
    기본 = ③ 탭 기본 3개(옥수수·대두·대두박). Full은 5개 이하. 3개 기준 약 40초."""
    import mvgarch as MG
    ids = [resolve(x) for x in items] if items else MG_IDS[:3]
    if not 2 <= len(ids) <= (5 if kind == "Full" else 10):
        raise ValueError("BEKK 품목은 2개 이상 (Full은 5개 이하)")
    R = _panel(ids)
    bk = MG.bekk_fit(R[ids], kind, 52)
    sp = MG.spillover_table(bk, {i: name_of(i) for i in ids})
    return _out({"items": [name_of(i) for i in ids], "kind": kind, "converged": bool(bk["converged"]), "stationarity_root": bk["stationarity"],
                 "loglik": bk["loglik"], "aic": bk["aic"], "spillover": _clean(sp.head(MAX_ROWS).to_dict("records"))},
                R.index[-1], f"{kind} BEKK(1,1), 주간 롤 보정 로그수익률 2000~", "mvgarch.bekk_fit · spillover_table",
                "|t| ≥ 1.96 = 유의한 전이. 수렴 실패·정상성 근 ≥ 1이면 결과를 믿지 말 것.")


def kalman_beta(item: str = "옥수수", drivers: list[str] | None = None, q: float = 1e-4, mode: str = "explain") -> dict:
    """Kalman TVP 회귀 — 시간에 따라 변하는 민감도 β(t). 대시보드 ② 레짐 → 'Kalman TVP'(주간, 원자재 6 + 거시 8 기본 패널)와 같은 계산.
    mode: explain = 같은 주 관계(헤지 비율·민감도), predict = 다음 주 수익률 예측. β = 설명변수 1σ 변화당 Y 주간 변동(%)."""
    import regime_mv as RM
    tgt = resolve(item)
    comm = list(dict.fromkeys([tgt] + RM.COMM_W_DEFAULT))
    macro = RM.MACRO_DEFAULT if drivers is None else [d for d in drivers if d in RM.MACRO]
    if drivers is not None:
        comm = list(dict.fromkeys([tgt] + [resolve(d) for d in drivers if d not in RM.MACRO]))
    X = RM.build_panel(levels(), comm, macro, "W")
    D = A.build_design(X, tgt, [c for c in X.columns if c != tgt], mode, 1, 1, True, 0.7)
    o = A.run_model("Kalman TVP", D, RM.ANN["W"], kalman_q=q, mode=mode)
    Bs = o["extra"]["beta_smoothed"]
    lab = lambda c: RM.label(c.split(" (")[0]) if c.split(" (")[0] in RM.MACRO else name_of(c.split(" (")[0]) + c[len(c.split(" (")[0]):]   # noqa: E731
    rows = [{"변수": lab(c), "현재 β(필터)": o["coef"][c], "1년 전 β(평활)": Bs[c].iloc[-53] if len(Bs) > 53 else None,
             "5년 범위 최저": Bs[c].iloc[-260:].min(), "5년 범위 최고": Bs[c].iloc[-260:].max()} for c in o["coef"].abs().sort_values(ascending=False).index]
    met = {k: o["metrics"].get(k) for k in ("R² (학습)", "R² (테스트)", "OOS R² (vs 학습평균)", "방향 적중률")}
    return _out({"item": name_of(tgt), "mode": mode, "q": f"{q:g}", "betas": _clean(rows[:MAX_ROWS]), "metrics": met},
                Bs.index[-1], f"Kalman TVP(q={q:g}), 주간 패널(regime_mv.build_panel), 표준화 X, 관측잡음 = 학습 70% 구간", "analytics.run_model('Kalman TVP')",
                "현재 β는 그 주까지 자료만 쓴 필터 값, 과거 β는 전체 자료로 다듬은 평활 값. explain은 동시점 관계라 예측력이 아님.")


ML_FEATS = ["yf:ZS=F", "yf:ZM=F", "yf:ZW=F", "yf:CL=F", "yf:HG=F", "yf:GC=F", "ak:macro_shipping_bdi",
            "yf:DX-Y.NYB", "fred:VIXCLS", "fred:DGS10", "fred:DTB3", "fred:BAA10Y", "yf:^GSPC"]                 # ④ 탭 기본 설명변수
ML_HP = dict(l1_ratio=[0.1, 0.5, 0.9], ada_gamma=1.0, scad_gamma=3.7, mcp_gamma=3.0, tau=0.05, kalman_q=1e-4, n_trees=300, max_depth=4,
             seq=20, hidden=32, epochs=40, hmm_states=0, hmm_cov="full", k_max=10, csr_kmax=8, csr_M=2000, bma_iters=20000, hs_iters=2000,
             mn_theta=0.5, ms_k=2, ms_pc=3, tar_boot=199, lgb_leaves=7, lgb_lr=0.03, lgb_l2=5.0, gp_nmax=1500)   # ④ 탭 하이퍼파라미터 기본값


def ml_models(item: str = "옥수수", models: list[str] | None = None, mode: str = "predict", lags: int = 3, horizon: int = 1,
              train_frac: float = 0.7, cost_bp: float = 5.0) -> dict:
    """회귀·머신러닝 모델 비교 — 대시보드 ④ 수익률 예측과 같은 계산(주간 2000~, 기본 설명변수 13개, 학습 70% → 시험 30%).
    models: analytics.MODEL_LIST 27개 중 고름(OLS·Ridge·Lasso·Elastic Net·SCAD·PLS·BMA·Random Forest·Gradient Boosting·LightGBM·LSTM 등).
    기본 = ④ 기본 7개(모형군별 대표). 시험 구간 OOS R²·방향 적중률·부호 전략 Sharpe(비용 후). 지금 시점 예측값은 내지 않음(④도 성과 비교만)."""
    tgt = resolve(item)
    models = models or A.CORE_MODELS
    bad = [m for m in models if m not in A.MODEL_LIST]
    if bad:
        raise ValueError(f"모델 {bad} 없음 — 후보: {', '.join(A.MODEL_LIST)}")
    feats = [f for f in ML_FEATS if f != tgt]
    R = _panel([tgt] + feats)
    D = A.build_design(R, tgt, feats, mode, lags, horizon, True, train_frac)
    D.cost, D.band = cost_bp / 100.0, 0.0
    rows = []
    for m in models:
        try:
            met = A.run_model(m, D, 52, **ML_HP, mode=mode, thr_var="fred:VIXCLS" if "fred:VIXCLS" in feats else tgt)["metrics"]   # ④ TAR·STAR 문턱 기본
        except Exception as e:                                # noqa: BLE001 — 한 모델 실패가 전체를 막지 않게
            rows.append({"모델": m, "오류": f"{type(e).__name__}: {e}"[:200]}); continue
        rows.append({"모델": m, **{k: met.get(k) for k in ("OOS R² (vs 학습평균)", "R² (테스트)", "방향 적중률", "부호전략 Sharpe (비용 후)", "Buy&Hold Sharpe", "연간 회전율")}})
    return _out({"item": name_of(tgt), "mode": mode, "lags": lags, "horizon_weeks": horizon, "train": f"{D.idx_tr[0]:%Y-%m}~{D.idx_tr[-1]:%Y-%m}",
                 "test": f"{D.idx_te[0]:%Y-%m}~{D.idx_te[-1]:%Y-%m}", "n_features": len(D.names), "table": _clean(rows)},
                D.idx_te[-1], f"주간 롤 보정 로그수익률, {'예측: t+' + str(horizon) + '주' if mode == 'predict' else '설명: 같은 주'}, 시차 {lags}, 학습 {train_frac:.0%}",
                "analytics.build_design · run_model",
                "OOS R² ≤ 0 이면 '학습 평균으로 찍기'보다 못함. 여러 모델 중 최고는 운으로도 높아짐 — 방향 예측은 검증상 근거 없음(⑥).")


def strategy_weights(top: int = 10) -> dict:
    """포트폴리오 전략의 이번 주 품목별 비중 — 롱숏 탭 3대 전략(MinVar 롱 · 52주 모멘텀 · 가치)과 평균회귀(섹터 내 따라잡기),
    가치 혼합(MinVar 50 + 모멘텀 25 + 가치 25)의 순 비중. 각 전략은 목표 변동성 10% 배율 반영. 비중 % (+ 롱 · − 숏)."""
    import ls_extra as LX
    import ls_strategies as LS
    P = LS.weekly(levels(), LS.BASKET, "2001-01-01"); P = P[[c for c in LS.BASKET if c in P]]; R = LS._ret(P)
    lp = np.log(P)
    W = {"MinVar 롱": LS._vol_target(LS.alloc_weights_cap(P, 0.25), R, 10.0),
         "52주 모멘텀": LS._vol_target(LS.tsmom_weights(P, 52, port_tgt=None), R, 10.0),
         "가치 (5년 평균 대비 싼 품목 롱·비싼 품목 숏)": LS._vol_target(LX.xs_weights(-(lp - lp.rolling(260, min_periods=156).mean())), R, 10.0),
         "평균회귀 (섹터 내 26주 과열 숏·낙오 롱)": LS._vol_target(LX.xs_weights(-lp.diff(26), sector_neutral=True), R, 10.0)}
    now = pd.DataFrame({k: w.iloc[-1] for k, w in W.items()}) * 100
    now["가치 혼합 순 비중 (50/25/25)"] = 0.5 * now["MinVar 롱"] + 0.25 * now["52주 모멘텀"] + 0.25 * now.iloc[:, 2]
    now = now.rename(index=name_of)
    gross = now.abs().sum()
    out = {k: _clean(now[k][now[k].abs() > 0.05].sort_values(key=abs, ascending=False).head(top).to_dict()) for k in now}
    return _out({"weights_pct": out, "gross_pct": _clean(gross.to_dict()), "n_items": len(now)},
                P.index[-1], "19개 원자재 바스켓, 주간 롤 보정 가격, 전략별 비중 규칙(ls_strategies · ls_extra) + 직전 26주 변동성으로 목표 10% 배율",
                "ls_strategies.alloc_weights_cap · tsmom_weights · ls_extra.xs_weights",
                "롱숏·Portfolio Customization 탭은 고정 기준일 자료로 그린 결과, 이 도구는 최신 자료로 같은 규칙을 다시 계산. 혼합 순 비중은 섞은 뒤 배율 조정 전 값.")


def text_sentiment(last: int = 3) -> dict:
    """문서 감성(FinBERT, −1 부정 ~ +1 긍정) 최근 값 — 멀티모달 갈래: EIA STEO(에너지 품목별) · 연준 베이지북(업종별) · USDA WASDE(분야별).
    문장 수(n)가 적으면 흔들림 큼. 감성은 방향 예측력 검증에서 효과 없음(멀티모달 학습 최적 비중 0%) → 맥락 설명용."""
    MMD = DATA / "mm"
    out = {}
    if (f := MMD / "steo_text.parquet").exists():
        S = pd.read_parquet(f)
        v = sorted(S.vintage.unique())[-last:]
        X = S[S.vintage.isin(v)].assign(item=lambda d: d.id.map(name_of)).pivot_table(index="item", columns="vintage", values="tone")
        out["EIA STEO (에너지 품목, 월간)"] = {"asof": v[-1], "tone": _clean(X.round(3).rename(columns=lambda c: str(c)[:10]).to_dict("index"))}
    if (f := MMD / "beige.parquet").exists():
        B = pd.read_parquet(f)
        m = sorted(B.month.unique())[-last:]
        X = B[B.month.isin(m)].pivot_table(index="group", columns="month", values="tone")
        out["연준 베이지북 (업종, 약 6주마다)"] = {"asof": m[-1], "tone": _clean(X.round(3).rename(columns=lambda c: str(c)[:10]).to_dict("index"))}
    if (f := MMD / "wasde_text.parquet").exists():
        Wt = pd.read_parquet(f)
        r = Wt.release.max()
        out["USDA WASDE (분야, 월간)"] = {"asof": r, "tone": _clean(Wt[Wt.release == r].set_index("section").tone.round(3).to_dict())}
    if not out:
        raise ValueError("감성 파일 없음 — multimodal.py로 먼저 생성")
    asof = max(pd.Timestamp(str(v["asof"])[:10]) for v in out.values())
    return _out({"sources": out}, asof, "FinBERT 문장별 (긍정 − 부정) 평균", "multimodal.steo_text_scores · beige_scores · wasde_text_scores (data/mm/*.parquet)",
                "감성은 발표 이후 날짜에만 사용. 품목 가격 방향을 예측한다는 근거는 없음.")


# ============================================================================
# 자료 수집 — 갱신 필요 여부 조회(읽기) · 갱신 실행(쓰기 · 외부 접속 · 사람 승인 필요)
# ============================================================================
REFRESH = {"prices": dict(label="가격·거시·포지션 (대시보드 사이드바 '데이터 업데이트'와 같은 단계)", minutes="3–6",
                          steps=["collect.py", "clean.py", "roll_adjust.py", "data_quality.py", "export_excel.py"],
                          hosts=["Yahoo Finance (선물 시세)", "fred.stlouisfed.org", "publicreporting.cftc.gov (COT)", "worldbank.org (Pink Sheet)", "akshare (BDI·DCE 중국 사이트)"]),
           "reports": dict(label="USDA WASDE · EIA STEO 리포트 (발표 원문 → 수치)", minutes="1–3", steps=["reports.py"],
                           hosts=["usda.gov (WASDE)", "eia.gov (STEO)"])}
STALE_BDAYS = {"가격 (Yahoo 선물)": 2, "거시 (FRED)": 5, "운임·중국 (akshare)": 5, "CFTC COT": 10, "USDA WASDE": 30, "EIA STEO": 30,
               "문서 감성 (STEO)": 45}


def data_status() -> dict:
    """[읽기] 자료원별 마지막 날짜 · 지난 영업일 수 · 갱신 필요 여부. 인터넷 접속 없음 — 로컬 파일만 봄. 갱신은 refresh_data(승인 필요)."""
    cat = _catalog(_mtime())
    end = pd.to_datetime(cat.end)
    src = cat.id.str.split(":").str[0]
    last = {"가격 (Yahoo 선물)": end[src == "yf"].max(), "거시 (FRED)": end[src == "fred"].max(), "운임·중국 (akshare)": end[src == "ak"].max()}
    for k, f, c in [("CFTC COT", DATA / "cot.parquet", "date"), ("USDA WASDE", DATA / "reports" / "wasde_features.parquet", "release"),
                    ("EIA STEO", DATA / "reports" / "steo_features.parquet", "vintage"), ("문서 감성 (STEO)", DATA / "mm" / "steo_text.parquet", "vintage")]:
        if f.exists():
            last[k] = pd.to_datetime(pd.read_parquet(f, columns=[c])[c]).max()
    last = {k: (pd.Timestamp(d).tz_localize(None) if pd.Timestamp(d).tzinfo else pd.Timestamp(d)).normalize() for k, d in last.items()}
    today = pd.Timestamp.today().normalize()
    rows = []
    for k, d in last.items():
        lag = int(np.busday_count(d.date(), today.date()))
        rows.append({"자료": k, "마지막 날짜": d, "지난 영업일": lag, "허용(영업일)": STALE_BDAYS[k], "갱신 필요": lag > STALE_BDAYS[k]})
    need = [r["자료"] for r in rows if r["갱신 필요"]]
    tgt = sorted({"prices" for r in need if r in ("가격 (Yahoo 선물)", "거시 (FRED)", "운임·중국 (akshare)", "CFTC COT")} |
                 {"reports" for r in need if r in ("USDA WASDE", "EIA STEO")})
    return _out({"today": today, "table": _clean(rows), "stale": need, "suggest_refresh": tgt,
                 "refresh_plans": {k: {x: REFRESH[k][x] for x in ("label", "steps", "hosts", "minutes")} for k in tgt}},
                max(last.values()), "로컬 파일의 마지막 날짜 비교 (주말 제외 영업일)", "analytics.load_catalog · data/*.parquet",
                "WASDE·STEO는 월 1회 발표라 30영업일 이내면 정상. 실제 갱신은 refresh_data(target, confirm=True) — 사람 승인 후에만.")


def refresh_data(target: str = "prices", confirm: bool = False) -> dict:
    """[쓰기 · 외부 접속] 자료 갱신. confirm=False(기본)면 실행하지 않고 계획(실행 단계·접속할 외부 사이트·예상 시간)만 돌려줌.
    confirm=True는 사람이 계획을 보고 승인한 뒤에만 — LangGraph는 approve_refresh(interrupt)에서 사용자 답을 받아야 이 값으로 부름.
    target: prices(가격·거시·COT, 대시보드 사이드바 버튼과 같음) · reports(USDA WASDE · EIA STEO)."""
    import os
    import subprocess
    if target not in REFRESH:
        raise ValueError(f"target은 {list(REFRESH)} 중 하나")
    plan = REFRESH[target]
    if not confirm:
        return _out({"executed": False, "target": target, "plan": plan, "needs_approval": True}, None, "계획만 (실행 안 함)", "llm.tools.REFRESH",
                    "외부 사이트에 접속해 data/를 덮어씀. 사용자 승인 후 confirm=True로 다시 부를 것.")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    log = []
    for s in plan["steps"]:
        t0 = datetime.now()
        r = subprocess.run([sys.executable, s], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
        log.append({"step": s, "ok": r.returncode == 0, "sec": (datetime.now() - t0).total_seconds(), "tail": (r.stdout if r.returncode == 0 else r.stderr or r.stdout)[-400:]})
        if r.returncode != 0:
            break
    _levels.cache_clear(); _catalog.cache_clear(); _vol_models.cache_clear(); _scm.cache_clear()
    ok = all(x["ok"] for x in log) and len(log) == len(plan["steps"])
    return _out({"executed": True, "target": target, "ok": ok, "log": log, "status_after": data_status()["table"] if ok else None},
                datetime.now(), "외부 수집 → 정제 → 보정", " → ".join(plan["steps"]), "" if ok else "실패한 단계 이후는 실행하지 않음 — log의 tail 확인")


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


def ontology_attribution(target: str, start: str, end: str) -> dict:
    """결과 → 원인 역추적: target(옥수수·대두·밀·대두박·대두유)의 start~end 누적 주간 변동(%)을 원인 충격·경로별 기여로 분해.
    분해는 정확(합 = 실제). 단, '설명'이지 예측이 아님."""
    CG, X, M, D = _scm(_mtime())
    if target not in CG.PRICES:
        raise ValueError(f"target은 {CG.PRICES} 중 하나")
    rows = CG.attribution(M, D, X, target, start, end, "질의")
    return _out({"target": target, "period": f"{start} ~ {end}", "contributions": _clean(rows[:12])},
                X.index[-1], "선형 SCM 역사적 분해(1판 온톨로지)", "causal_grain.attribution",
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


def expert_signals(item: str) -> dict:
    """② 의견원별 최근 의견 + 백테스트 판정 (의견 방향 vs 이후 4주 수익률, 채택 규칙: 표본 ≥ 20 · t > 2 · 위약 p < 0.05 · 두 기간 +).
    saved = 사내 전문가 의견(그때그때 계산) · usda = USDA 가격 전망 수정 · cot = CFTC 펀드 순포지션 변화 (llm/expert_bt.py로 미리 계산)."""
    from llm import expert_bt as EB
    i = resolve(item)
    pre = pd.read_csv(EB.OUT, encoding="utf-8-sig") if EB.OUT.exists() else pd.DataFrame(columns=["source", "id"])
    out = {}
    for src in ("saved", "usda", "cot"):
        V = EB.SOURCES[src]()
        V = V[V.id == i].sort_values("avail")
        if src == "saved":
            bt = EB.evaluate(EB.attach_returns(V)) if len(V) else {"n": 0, "verdict": f"표본 부족 (0건 < {EB.MIN_N})"}
        else:
            r = pre[(pre.source == src) & (pre.id == i)]
            bt = r.iloc[0].drop(["source", "item", "id"]).to_dict() if len(r) else {"verdict": "미계산 — python -m llm.expert_bt"}
        last = V.iloc[-1] if len(V) else None
        out[src] = {"latest": None if last is None else {"date": last.avail, "direction": float(last.direction), "confidence": float(last.confidence)},
                    "backtest": {k: bt.get(k) for k in ("n", "hit_rate", "mean_bp", "t", "placebo_p", "early_bp", "late_bp", "verdict") if k in bt}}
    asof = datetime.fromtimestamp(EB.OUT.stat().st_mtime) if EB.OUT.exists() else None
    return _out({"item": name_of(i), "id": i, "sources": out}, asof, "의견 방향 × 이후 4주 롤 보정 수익률, 진입 = 의견을 알 수 있었던 첫 종가",
                "llm.expert_bt", "판정이 '채택'인 의견원만 판단 보조로 쓸 수 있음. 나머지는 참고·설명용.")


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
# 도구 → 대시보드 탭 대조표. "같음" = 같은 함수·같은 기본 설정 → tests/test_llm_parity.py가 대시보드 쪽 호출을 따로 재현해 숫자 일치를 확인.
DASH = {
    "list_items": ("⑧ 데이터·방법론 — 자료 목록", "같음"),
    "get_prices": ("사이드바·각 탭 가격 (정제본, 롤 보정 아님)", "같음"),
    "get_results": ("⑥ 실험 결과 — 저장된 결과 표", "같음"),
    "vol_forecast": ("③ 변동성·공분산 → 변동성 동학 (주간·2000~·GJR-t)", "같음"),
    "var_es": ("◆ 리스크 → VaR·ES 백테스트 (σ GARCH(1,1)·평가 2010~)", "같음"),
    "regime": ("② 레짐", "다름 — 대시보드는 HMM(원자재+거시 다변량), 도구는 품목 하나의 Markov 스위칭(설명용)"),
    "factors": ("①+ 요인 모형", "다름 — 같은 함수(PCA·DY 스필오버)지만 도구는 최근 260주 · 곡물 중심 9품목 · 거시 제외"),
    "correlation": ("① 시장 구조·동조성", "대시보드에 같은 표 없음 — 최근 vs 이전 104주 단순 상관"),
    "buy_signal": ("★ 구매 의사결정 (추천 프리셋 · 가격 위치 창 품목별 자동)", "같음"),
    "fundamentals": ("★ 구매 의사결정 → 펀더멘털 맥락", "같음"),
    "dcc_correlation": ("③ 변동성·공분산 → ② DCC / ADCC", "같음"),
    "bekk_spillover": ("③ 변동성·공분산 → ③ BEKK", "같음"),
    "kalman_beta": ("② 레짐 → Kalman TVP (분석 목적 '설명')", "같음"),
    "ml_models": ("④ 수익률 예측 (분석 목적 '예측', 시차 3)", "같음"),
    "strategy_weights": ("롱숏 · Portfolio Customization (3대 전략 · 가치 혼합)", "같은 규칙 · 최신 자료 (탭은 고정 기준일 자료)"),
    "text_sentiment": ("멀티모달 갈래 — 문서 감성 원자료", "같음 (저장 파일)"),
    "data_status": ("사이드바 '데이터 기준일'", "같음"),
    "refresh_data": ("사이드바 '데이터 업데이트' 버튼", "같은 단계 (reports는 버튼 밖, reports.py)"),
    "ontology_graph": ("발표 별첨 5 · causal_grain (대시보드 탭 없음)", "LLM 계층 전용"),
    "ontology_attribution": ("발표 별첨 5 · causal_grain (대시보드 탭 없음)", "LLM 계층 전용"),
    "expert_views": ("없음 (사내 전문가 의견)", "LLM 계층 전용"),
    "expert_signals": ("없음 (② 전문가 의견 백테스트)", "LLM 계층 전용"),
    "save_expert_view": ("없음 (사내 전문가 의견)", "LLM 계층 전용"),
    "make_chart": ("그림 파일 (대시보드 그림과 별개)", "LLM 계층 전용")}


def _tagged(fn):
    """도구 결과에 dashboard(대응 탭 · 일치 여부)를 붙임 — 답변에서 '대시보드 어느 화면 값인지' 밝히는 근거."""
    import functools

    @functools.wraps(fn)
    def run(*a, **k):
        o = fn(*a, **k)
        if isinstance(o, dict) and fn.__name__ in DASH:
            o["dashboard"] = {"tab": DASH[fn.__name__][0], "match": DASH[fn.__name__][1]}
        return o
    return run


READ_TOOLS = [_tagged(f) for f in [list_items, get_prices, get_results, vol_forecast, var_es, regime, factors, correlation,
                                   buy_signal, fundamentals, dcc_correlation, bekk_spillover, kalman_beta, ml_models, strategy_weights,
                                   text_sentiment, data_status, ontology_graph, ontology_attribution, expert_views, expert_signals, make_chart]]
WRITE_TOOLS = [_tagged(f) for f in [save_expert_view, refresh_data]]
OPEN_WORLD = {"refresh_data"}                                    # 외부 사이트 접속 — MCP openWorldHint, 그래프는 사람 승인 뒤에만


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    for f in READ_TOOLS + WRITE_TOOLS:
        print(f"{'[쓰기]' if f in WRITE_TOOLS else '      '} {f.__name__}: {f.__doc__.strip().splitlines()[0]}")
    print(json.dumps(vol_forecast("옥수수"), ensure_ascii=False, indent=1)[:1500])
