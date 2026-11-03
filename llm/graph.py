"""4계층 LangGraph 흐름 — ① 수치 · ② 전문가 의견 · ③ 온톨로지 · ④ LLM 인터페이스 (설계 docs/LLM_설계.md §6).

흐름
  classify ─┬─ (의견 기록) extract_view → approve_view[사람 확인] → save_view ─────────────┐
            ├─ (자료 갱신) check_data → approve_refresh[사람 확인] → run_refresh(외부 수집) ──┤
            └─ (질문)  numeric(①) → expert(②) → ontology(③) → compose(④) → verify ─┬─ 통과 → finish
                                                                    ↑ 숫자 불일치(최대 2회) ┘    ↓ 2회 실패 → 사실 목록 답변
  · 숫자는 도구(llm/tools.py)만 만든다. LLM은 '사실 목록'을 문장으로 엮기만 하고, verify가 답변 숫자를 사실 목록과 대조한다.
  · 질문 분류·품목 추출은 키워드 규칙(작은 로컬 모델의 오분류 방지). LLM은 답변 작성 · 의견 문장 → JSON 추출에만 쓴다.
  · 로컬 LLM = Ollama(127.0.0.1). 꺼져 있거나 실패하면 사실 목록 답변으로 대체 — 클라우드로 넘어가는 경로 없음.
  · 외부 접속은 자료 갱신 경로 하나 — 갱신 필요 여부를 로컬에서 먼저 보고, 사람이 계획(단계 · 접속 사이트)을 승인해야만 수집기를 실행.
  · 도구 호출: 기본 direct(같은 프로세스에서 import) · --mcp 면 MCP 서버(stdio)를 거침 — 결과는 같음.
  · LangSmith 추적은 강제로 끔(LANGCHAIN_TRACING_V2 등 제거).

실행:  .venv\\Scripts\\python.exe -m llm.graph "옥수수 20일 VaR는? 매입 예정 1천만 달러"   [--mcp] [--no-llm]
그림:  .venv\\Scripts\\python.exe -m llm.graph --mermaid   → docs/LLM_graph.mmd (Mermaid 흐름도, 인터넷 렌더 서비스 안 씀)
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import TypedDict

for _k in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING", "LANGSMITH_TRACING", "LANGCHAIN_API_KEY", "LANGSMITH_API_KEY"):
    os.environ.pop(_k, None)                                     # 대화·도구 결과가 외부 추적 서버로 가지 않게
os.environ["LANGSMITH_TRACING"] = "false"

import requests  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import Command, interrupt  # noqa: E402

from llm import tools as T  # noqa: E402

OLLAMA = "http://127.0.0.1:11434"                                # 로컬만 — 다른 주소는 거부
MODEL = os.environ.get("LLM_MODEL", "qwen3.5:4b")
MAX_FIX = 2
ROOT = Path(__file__).resolve().parents[1]


# ============================================================================
# 상태
# ============================================================================
class S(TypedDict, total=False):
    question: str
    route: str                 # lookup · decision · explain · direction · expert_add · refresh
    items: list[str]           # 품목 id
    intents: list[str]         # prices · vol · var · regime · factors · corr · buy · fund · dcc · spill · kalman · ml · strategy · text · status
    params: dict
    numeric: dict              # ① 도구 결과
    expert: dict               # ② 의견 조회 결과
    ontology: dict             # ③ 기여 분해 · 연결
    facts: list[str]           # 답변에 쓸 수 있는 사실(숫자 포함) — 도구 결과에서만 생성
    draft: str
    issues: list[str]
    attempts: int
    answer: str
    view: dict                 # 저장할 전문가 의견
    approved: bool
    plan: dict                 # 자료 갱신 계획 (대상 · 단계 · 접속 사이트 · 갱신 필요 자료)
    trace: list[str]           # 지나간 노드 (감사·테스트용)


# ============================================================================
# 도구 호출 (direct · MCP)
# ============================================================================
class Tools:
    def __init__(self, backend: str = "direct"):
        self.backend = backend

    def __call__(self, name: str, **args) -> dict:
        if self.backend == "direct":
            fn = {f.__name__: f for f in T.READ_TOOLS + T.WRITE_TOOLS}[name]
            try:
                return fn(**args)
            except Exception as e:                               # noqa: BLE001 — MCP 서버와 같은 형태로
                return {"error": f"{type(e).__name__}: {e}"}
        return asyncio.run(self._mcp(name, args))

    @staticmethod
    async def _mcp(name: str, args: dict) -> dict:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        srv = StdioServerParameters(command=sys.executable, args=["-m", "llm.mcp_server"], cwd=str(ROOT), env={"PYTHONIOENCODING": "utf-8"})
        async with stdio_client(srv) as (rd, wr), ClientSession(rd, wr) as s:
            await s.initialize()
            r = await s.call_tool(name, args)
            return json.loads(r.content[0].text)


# ============================================================================
# 로컬 LLM (Ollama)
# ============================================================================
def llm(messages: list[dict], fmt: dict | str | None = None, timeout: int = 180) -> str | None:
    """Ollama 채팅. 실패하면 None (호출 쪽이 규칙 기반으로 대체)."""
    assert OLLAMA.startswith("http://127.0.0.1"), "로컬 LLM만 허용"
    body = {"model": MODEL, "messages": messages, "stream": False, "think": False, "options": {"temperature": 0.1, "num_ctx": 4096}}
    if fmt:
        body["format"] = fmt
    try:
        r = requests.post(f"{OLLAMA}/api/chat", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception:                                            # noqa: BLE001
        return None


# ============================================================================
# 분류 (규칙)
# ============================================================================
ALIASES = {"옥수수": "yf:ZC=F", "소맥": "yf:ZW=F", "밀": "yf:ZW=F", "대두박": "yf:ZM=F", "대두유": "yf:ZL=F", "대두": "yf:ZS=F", "콩": "yf:ZS=F",
           "원유": "yf:CL=F", "wti": "yf:CL=F", "브렌트": "yf:BZ=F", "금": "yf:GC=F", "은": "yf:SI=F", "구리": "yf:HG=F", "천연가스": "yf:NG=F",
           "철광석": "yf:TIO=F", "커피": "yf:KC=F", "원당": "yf:SB=F", "면화": "yf:CT=F", "corn": "yf:ZC=F", "wheat": "yf:ZW=F", "soybean": "yf:ZS=F"}
KW = {"vol": r"변동성|garch|gjr|출렁", "var": r"(?<![a-z])var(?![a-z])|(?<![a-z])es(?![a-z])|손실|위험 금액|리스크 금액|최대 손실", "regime": r"레짐|국면|위기 확률",
      "factors": r"요인|pca|스필오버|동조", "corr": r"상관", "prices": r"가격|시세|얼마|현재가", "buy": r"살까|사야 할까|사야 하나|사야 해\?|뭐부터 사|매수|구매 판단|미룰|사도|타이밍",
      "fund": r"wasde|재고|수급|cot|포지션", "dcc": r"dcc|동적 상관|상관.{0,4}(추이|변화|높아|낮아|커졌|줄었)", "spill": r"bekk|변동성 전이|충격 전이|전이 효과",
      "kalman": r"칼만|kalman|민감도|베타|β|헤지 비율", "ml": r"머신러닝|ml 모델|lasso|라쏘|ridge|릿지|xgboost|lightgbm|lstm|부스팅|예측 모델|모델 비교|모델 성능",
      "strategy": r"포트폴리오|비중|minvar|최소분산|모멘텀|가치 전략|평균회귀|전략", "text": r"감성|finbert|텍스트|베이지북|steo|문서",
      "status": r"자료 기준일|데이터 기준일|언제 자료|최신 자료|갱신 필요|업데이트 필요"}
REFRESH_Q = r"(자료|데이터|시세|리포트|보고서|wasde|steo).{0,8}(갱신|업데이트|새로 받|다운로드|받아)"   # 실행 요청 (필요 여부만 물으면 status 조회)


def find_items(q: str) -> list[str]:
    ql, out = q.lower(), []
    for nm in sorted(ALIASES, key=len, reverse=True):          # 긴 이름 먼저 ('대두박'이 '대두'로 잡히지 않게)
        if nm in ql:
            i = ALIASES[nm]
            if nm in ("금", "은", "밀", "콩") and not re.search(rf"(^|[\s,·(]){nm}($|[\s,·)은는이가의을를도과와])", ql):
                continue                                         # 한 글자 이름은 단어로 쓰였을 때만 ('지금'의 '금' 제외)
            if i not in out:
                out.append(i)
            ql = ql.replace(nm, " ")
    return out


def parse_params(q: str) -> dict:
    p = {}
    if m := re.search(r"(\d+)\s*(일|영업일)", q):
        p["horizon"] = int(m.group(1))
    if m := re.search(r"(\d+(?:\.\d+)?)\s*%\s*(var|신뢰)", q, re.I):
        p["alpha"] = float(m.group(1)) / 100
    if m := re.search(r"(\d+(?:\.\d+)?)\s*(억|천만|백만|만)?\s*(달러|usd|\$)", q, re.I):
        unit = {"억": 1e8, "천만": 1e7, "백만": 1e6, "만": 1e4}.get(m.group(2) or "", 1)
        p["notional_usd"] = float(m.group(1)) * unit
    if re.search(r"매입 예정|구매 예정|살 예정|사야 할|사야할", q):
        p["side"] = "short"
    if m := re.search(r"작성자\s*[:：=]?\s*([^\s,·)]+)", q):
        p["author"] = m.group(1)
    if m := re.search(r"(20\d\d-\d\d-\d\d)\s*~\s*(20\d\d-\d\d-\d\d)", q):
        p["period"] = (m.group(1), m.group(2))
    return p


def classify(st: S) -> dict:
    q = st["question"]; ql = q.lower()
    items, params = find_items(q), parse_params(q)
    intents = [k for k, pat in KW.items() if re.search(pat, ql)]
    if re.search(r"의견 (저장|기록)|코멘트 (저장|기록)|전문가 의견\s*:", q):
        route = "expert_add"
    elif re.search(REFRESH_Q, ql) and not re.search(r"필요|해야|언제|됐|되었", q):
        route = "refresh"
    elif re.search(r"오를까|내릴까|떨어질까|상승할까|하락할까|오를 것|떨어질 것|방향|목표가|얼마까지", q):
        route, intents = "direction", ["buy", "var"]             # 방향은 단정하지 않고 범위·판단 근거로 대신 답
    elif re.search(r"왜|원인|이유|기여|때문", q):
        route = "explain"
        intents = intents or ["prices"]
    elif "buy" in intents:
        route = "decision"; intents = sorted(set(intents) | {"fund"})
    else:
        route = "lookup"; intents = intents or ["prices"]
    if not items and (route not in ("lookup", "refresh") or set(intents) & {"vol", "var", "regime", "kalman", "ml"}):
        items = ["yf:ZC=F"]                                     # 품목 없으면 대표 품목(옥수수) — 답변에 밝힘
        params["default_item"] = True
    return {"route": route, "items": items, "intents": intents, "params": params, "attempts": 0, "trace": ["classify"]}


# ============================================================================
# ① 수치 · ② 전문가 의견 · ③ 온톨로지 — 도구 결과 → 사실 목록
# ============================================================================
def _f(x, d=1):
    return "–" if x is None else f"{x:,.{d}f}"


def numeric_layer(st: S, tools: Tools) -> dict:
    p, ids, out, facts = st["params"], st["items"], {}, []
    for it in st["intents"]:
        if it == "factors":
            r = tools("factors", items=[T.name_of(i) for i in ids] if len(ids) >= 3 else None)
            out["factors"] = r
            if "error" not in r:
                facts.append(f"요인: PC1이 설명하는 몫 {_f(100 * r['explained_ratio']['PC1'])}%, 스필오버 총 연계도 {_f(r['spillover_total_pct'])}% (기준일 {r['asof']}, {r['method']})")
            continue
        if it == "corr" and len(ids) >= 2:
            r = tools("correlation", items=ids); out["corr"] = r
            if "error" not in r:
                facts.append(f"평균 상관: 최근 {r['weeks']}주 {_f(r['avg_pair_recent'], 2)} vs 그 전 {_f(r['avg_pair_previous'], 2)} (기준일 {r['asof']})")
            continue
        if it == "buy":
            r = tools("buy_signal", items=ids); out["buy"] = r
            for row in r.get("table", []):
                sc = row["매수 점수"]; lab = "적극 매수" if sc >= 60 else ("판단 유보" if sc >= 40 else "매수 보류")
                facts.append(f"{row['품목']} 매수 점수 {_f(sc)}점({lab}), 현재가 {_f(row['가격'])} {row['단위']}, 4주 90% 범위 {_f(row['4주 하단 90%'])}~{_f(row['4주 상단 90%'])} (기준일 {row['기준일']})")
            continue
        if it == "fund":
            r = tools("fundamentals", items=ids); out["fund"] = r
            for row in r.get("table", []):
                if row.get("재고/사용비율 (%)") is not None:
                    facts.append(f"{row['품목']} WASDE 재고/사용비율 {_f(row['재고/사용비율 (%)'])}% (발표 {row.get('WASDE 발표일')}, 전월 대비 {_f(row.get('전월 대비 (%p)'), 2)}%p) — 매수 점수 계산에는 넣지 않는 맥락 정보")
            continue
        if it == "dcc":
            r = tools("dcc_correlation", items=ids if len(ids) >= 2 else None); out["dcc"] = r
            if "error" not in r:
                mine = {T.name_of(i) for i in ids}
                pr = [x for x in r["pairs"] if len(mine) < 2 or set(x["쌍"].split("–")) <= mine][:3]
                facts.append(f"DCC 동적 상관 평균 {_f(r['avg_corr_now'], 2)} (1년 전 {_f(r['avg_corr_1y_ago'], 2)}, 전체 평균 {_f(r['avg_corr_mean'], 2)}), 지속성 {_f(r['persistence'], 3)} (기준일 {r['asof']}, 대시보드 ③)")
                facts += [f"{x['쌍']} 동적 상관 {_f(x['현재'], 2)} (1년 전 {_f(x['1년 전'], 2)}, 과거 분위 {_f(x['현재 분위(%)'], 0)}%)" for x in pr]
            continue
        if it == "spill":
            r = tools("bekk_spillover", items=ids if 2 <= len(ids) <= 5 else None); out["spill"] = r
            if "error" not in r:
                sig = [x for x in r["spillover"] if x.get("유의(5%)")]
                facts.append(f"BEKK 전이({', '.join(r['items'])}, 수렴 {'예' if r['converged'] else '아니오 — 결과 주의'}): "
                             + ("; ".join(f"{x['유형'].split(' (')[0]} {x['from → to']} 계수 {_f(x['계수'], 3)}(t {_f(x['t값'], 1)})" for x in sig) if sig else "5% 유의한 전이 없음")
                             + f" (기준일 {r['asof']}, 대시보드 ③)")
            continue
        if it == "strategy":
            r = tools("strategy_weights"); out["strategy"] = r
            if "error" not in r:
                mix = r["weights_pct"].get("가치 혼합 순 비중 (50/25/25)", {})
                pick = {k: v for k, v in mix.items() if not ids or k in {T.name_of(i) for i in ids}} or mix
                facts.append("가치 혼합(MinVar 50 · 모멘텀 25 · 가치 25) 이번 주 순 비중: " + ", ".join(f"{k} {_f(v)}%" for k, v in list(pick.items())[:6])
                             + f" (+ 롱 · − 숏, 기준일 {r['asof']}, 롱숏 탭 규칙 · 최신 자료)")
            continue
        if it == "text":
            r = tools("text_sentiment"); out["text"] = r
            if "error" not in r:
                for src, d in r["sources"].items():
                    last = {k: (list(v.values())[-1] if isinstance(v, dict) else v) for k, v in d["tone"].items()}
                    facts.append(f"{src} 문서 감성(−1~+1): " + ", ".join(f"{k} {_f(v, 2)}" for k, v in last.items() if v is not None) + f" (기준 {str(d['asof'])[:10]}, 방향 예측력 검증 안 됨)")
            continue
        if it == "status":
            r = tools("data_status"); out["status"] = r
            if "error" not in r:
                facts.append("자료 기준일: " + ", ".join(f"{x['자료']} {x['마지막 날짜']}({x['지난 영업일']}영업일 전)" for x in r["table"])
                             + (f" → 갱신 필요: {', '.join(r['stale'])} ('자료 갱신해줘'로 요청하면 승인 후 실행)" if r["stale"] else " → 모두 최신"))
            continue
        for i in ids:
            nm = T.name_of(i)
            if it == "prices":
                r = tools("get_prices", item=i); out[f"prices:{i}"] = r
                if "error" not in r:
                    s_ = r["stats"]
                    facts.append(f"{nm} 현재가 {_f(s_['현재'], 2)} {r['unit']}, 1개월 {_f(s_['1개월 변화(%)'])}%, 1년 {_f(s_['1년 변화(%)'])}% (기준일 {r['asof']})")
            elif it == "vol":
                r = tools("vol_forecast", item=i, horizon=p.get("horizon", 20)); out[f"vol:{i}"] = r
                if "error" not in r:
                    facts.append(f"{nm} 연 변동성 현재 {_f(r['vol_now_ann_pct'])}% (GJR-GARCH 주간, 대시보드 ③), 1년 평균 {_f(r['vol_1y_avg_ann_pct'])}%, 지속성 {_f(r['persistence'], 3)}, 반감기 {_f(r['half_life_weeks'], 0)}주 (기준일 {r['asof']})")
            elif it == "var":
                kw = {k: p[k] for k in ("alpha", "notional_usd", "side") if k in p}
                r = tools("var_es", item=i, horizon=p.get("horizon", 20), **kw); out[f"var:{i}"] = r
                if "error" not in r:
                    fhs = next((x for x in r["now"] if x["방법"].startswith("FHS")), r["now"][0])
                    bt = next((x for x in r["backtest"] if x["방법"] == fhs["방법"]), {})
                    money = f", 금액 약 ${_f(fhs['VaR (USD)'], 0)}" if "VaR (USD)" in fhs else ""
                    facts.append(f"{nm} {r['horizon_days']}일 {_f(100 * r['alpha'], 0)}% VaR {_f(fhs['VaR (%)'])}% · ES {_f(fhs['ES (%)'])}%{money} "
                                 f"({'매입 예정' if r['side'] == 'short' else '보유'} 기준, FHS, 과거 초과율 {_f(100 * bt.get('초과율', 0), 1)}% vs 기대 {_f(100 * r['expected_exceed_rate'], 1)}%, 기준일 {r['asof']})")
            elif it == "regime":
                r = tools("regime", item=i); out[f"regime:{i}"] = r
                if "error" not in r:
                    hi = r["current"].get("레짐2")
                    facts.append(f"{nm} 현재 {r['current_regime']}(레짐1 = 잔잔한 국면), 고변동 레짐 확률 {_f(100 * hi if hi is not None else None)}% (기준일 {r['asof']}, 사후 평활 확률 — 설명용)")
            elif it == "kalman":
                r = tools("kalman_beta", item=i); out[f"kalman:{i}"] = r
                if "error" not in r:
                    facts.append(f"{nm} 칼만 시변 민감도(같은 주, 설명변수 1σ당 %): " + ", ".join(f"{b['변수']} {_f(b['현재 β(필터)'], 2)}(1년 전 {_f(b['1년 전 β(평활)'], 2)})" for b in r["betas"][:4])
                                 + f" (기준일 {r['asof']}, 대시보드 ② Kalman TVP)")
            elif it == "ml":
                r = tools("ml_models", item=i); out[f"ml:{i}"] = r
                if "error" not in r:
                    ok_ = [x for x in r["table"] if "오류" not in x]
                    best = max(ok_, key=lambda x: x["OOS R² (vs 학습평균)"] or -9) if ok_ else None
                    pos = sum((x["OOS R² (vs 학습평균)"] or 0) > 0 for x in ok_)
                    if best:
                        facts.append(f"{nm} 머신러닝 {len(ok_)}개 모델 시험 구간({r['test']}) 다음 주 수익률 예측: OOS R² 0 초과 {pos}개, 최고 {best['모델']} OOS R² {_f(best['OOS R² (vs 학습평균)'], 3)} · 방향 적중률 {_f(100 * best['방향 적중률'])}% (대시보드 ④, 0 이하면 평균으로 찍기보다 못함)")
    if st["params"].get("default_item"):
        facts.append("질문에 품목이 없어 옥수수 기준으로 답함")
    return {"numeric": out, "facts": facts, "trace": st["trace"] + ["numeric"]}


SRC_NAME = {"saved": "사내 전문가 의견", "usda": "USDA 가격 전망 수정", "cot": "CFTC 펀드 포지션 변화"}


def expert_layer(st: S, tools: Tools) -> dict:
    """② 의견원별 최근 방향 + 백테스트 판정(의견 방향 vs 이후 4주 수익률). '채택'이 아닌 의견은 참고·설명용으로 표시."""
    out, facts = {}, list(st["facts"])
    if st["route"] in ("decision", "direction", "explain"):
        for i in st["items"]:
            nm = T.name_of(i)
            v = tools("expert_views", item=i, last=5).get("views", [])
            if v:
                avg = sum(x["direction"] * x["confidence"] for x in v) / max(sum(x["confidence"] for x in v), 1e-9)
                facts.append(f"{nm} 사내 전문가 의견 최근 {len(v)}건(최근 {v[0]['date']}), 신뢰도 가중 방향 {avg:+.2f} (−1 약세 ~ +1 강세)")
            r = tools("expert_signals", item=i); out[i] = r
            for src, d in r.get("sources", {}).items():
                bt, la = d["backtest"], d["latest"]
                if src == "saved" and not la:
                    facts.append(f"{nm} 사내 전문가 의견 없음 — 검증은 {bt.get('verdict')}")
                    continue
                if not la:
                    continue
                dir_ = "강세(올림)" if la["direction"] > 0 else "약세(내림)"
                perf = (f", 과거 {bt['n']:.0f}건 적중률 {100 * bt['hit_rate']:.0f}% · 의견대로 했을 때 평균 {bt['mean_bp']:+.0f}bp(t {bt['t']:.1f})"
                        if bt.get("hit_rate") is not None else "")
                facts.append(f"{nm} {SRC_NAME[src]}: 최근 {la['date']} {dir_}{perf} → {bt.get('verdict')}")
    return {"expert": out, "facts": facts, "trace": st["trace"] + ["expert"]}


ONT_PRICES = {"yf:ZC=F": "옥수수", "yf:ZS=F": "대두", "yf:ZW=F": "밀", "yf:ZM=F": "대두박", "yf:ZL=F": "대두유"}


def ontology_layer(st: S, tools: Tools) -> dict:
    out, facts = {}, list(st["facts"])
    if st["route"] == "explain":
        g = tools("ontology_graph", version=1)
        last = g.get("asof") or None
        for i in st["items"]:
            if i not in ONT_PRICES:
                facts.append(f"{T.name_of(i)}은(는) 곡물 온톨로지(옥수수·대두·밀·대두박·대두유) 밖이라 원인 분해 없음")
                continue
            a, b = st["params"].get("period") or (None, None)
            if not a:
                import pandas as pd
                end = pd.Timestamp(tools("get_prices", item=i)["asof"]); a, b = f"{end - pd.Timedelta(weeks=3):%Y-%m-%d}", f"{end:%Y-%m-%d}"
            r = tools("ontology_attribution", target=ONT_PRICES[i], start=a, end=b); out[i] = r
            rows = r.get("contributions", [])
            if rows:
                top = ", ".join(f"{x['원인']} {x['기여']:+.1f}%p" for x in rows[:3])
                facts.append(f"{ONT_PRICES[i]} {a}~{b} 누적 {rows[0]['실제_누적']:+.1f}% 중 원인별 기여: {top} (온톨로지 구조 모형 분해 — 설명용, 예측 아님)")
            ed = [e for e in g.get("edges", []) if e["to"] == ONT_PRICES[i] and e.get("t") is not None and abs(e["t"]) >= 2]
            if ed:
                facts.append(f"{ONT_PRICES[i]}에 유의한 연결(|t|≥2): " + ", ".join(f"{e['from']}(계수 {e['계수']:+.2f})" for e in ed[:4]))
        out["_graph_asof"] = last
    return {"ontology": out, "facts": facts, "trace": st["trace"] + ["ontology"]}


# ============================================================================
# ④ LLM 인터페이스 — 작성 · 숫자 대조 · 마무리
# ============================================================================
SYSTEM = ("너는 원자재 구매 담당자를 돕는 분석 비서다. 한국어 개조식으로 답한다.\n"
          "규칙: 1) 숫자는 아래 '사실' 목록에 있는 것만 그대로 쓴다. 새 숫자를 계산하거나 지어내지 않는다.\n"
          "2) 첫 줄에 결론, 이어서 근거(숫자·기준일), 마지막에 한계 1줄.\n"
          "3) 가격이 오를지 내릴지 단정하지 않는다. 방향 질문에는 '방향 예측은 검증상 근거 없음'을 밝히고 범위·판단 점수로 답한다.\n"
          "4) 전문가 의견과 온톨로지 분해는 참고·설명용이라고 밝힌다. 5) 6줄 이내.")
NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def _nums(text: str) -> list[float]:
    out = []
    for m in NUM.findall(text):
        try:
            out.append(float(m.replace(",", "").rstrip(".")))
        except ValueError:
            pass
    return out


def compose(st: S) -> dict:
    facts = st.get("facts") or ["관련 자료를 찾지 못함"]
    fb = "\n".join(f"- {x}" for x in st.get("issues", []))
    user = f"질문: {st['question']}\n\n사실:\n" + "\n".join(f"- {x}" for x in facts)
    if fb:
        user += f"\n\n이전 답변의 문제(고칠 것):\n{fb}"
    if st["route"] == "direction":
        user += "\n\n(이 질문은 가격 방향을 묻는다 — 규칙 3을 지킬 것)"
    draft = None if os.environ.get("LLM_OFF") == "1" else llm([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
    return {"draft": draft or "", "attempts": st.get("attempts", 0) + 1, "trace": st["trace"] + ["compose"]}


def verify(st: S) -> dict:
    """답변 숫자가 사실 목록(또는 질문)에 있는지 — 반올림 허용. 10 미만 정수(번호·개수)는 제외."""
    if not st["draft"]:
        return {"issues": ["LLM 응답 없음"], "trace": st["trace"] + ["verify"]}
    ok_nums = _nums(" ".join(st["facts"]) + " " + st["question"]) + [v for v in st["params"].values() if isinstance(v, (int, float))]
    bad = []
    for x in _nums(st["draft"]):
        if abs(x) < 10 and float(x).is_integer():
            continue
        if not any(abs(abs(x) - abs(y)) <= max(0.051, 0.006 * abs(y)) for y in ok_nums):
            bad.append(x)
    issues = [f"사실 목록에 없는 숫자 {b:g}를 썼음 — 사실 목록의 숫자만 쓸 것" for b in bad[:5]]
    if DIRECTION.search(st["draft"]):                                # 모든 질문에서 방향 단정 금지
        issues.append("가격 방향을 단정했음(예: '하락할 것', '상승 가능성이 높음') — 단정하지 말 것")
    xs = _nums(st["draft"])
    for label, y in required(st):                                    # 질문이 꼭 답해야 할 숫자가 빠졌는지
        if not any(abs(abs(x) - abs(y)) <= max(0.051, 0.006 * abs(y)) for x in xs):
            issues.append(f"핵심 숫자 빠짐: {label} {y:g} — 반드시 포함할 것")
    return {"issues": issues, "trace": st["trace"] + ["verify"]}


DIRECTION = re.compile(r"(오를|상승할|하락할|내릴|떨어질|반등할)\s?것(으로|이다|입니다|임)|(상승|하락|반등|오를|내릴) ?가능성이 (높|크)|소폭 (하락|상승)할?|(강세|약세)가 (예상|전망)")
REQUIRED = [("buy", r"(\S+(?: \([^)]*\))?) 매수 점수 ([\d.]+)점", "매수 점수"), ("var", r"VaR ([\d.]+)%", "VaR %"),
            ("vol", r"연 변동성 현재 ([\d.]+)%", "현재 연 변동성"), ("explain", r"누적 ([-+][\d.]+)%", "기간 누적 변동")]


def required(st: S) -> list[tuple[str, float]]:
    out = []
    for key, pat, label in REQUIRED:
        if key == "explain" and st["route"] != "explain" or key != "explain" and key not in st.get("intents", []):
            continue
        for f in st["facts"]:
            for m in re.findall(pat, f):
                out.append((label, float(m[-1] if isinstance(m, tuple) else m)))
    return out


def after_verify(st: S) -> str:
    if not st["issues"]:
        return "finish"
    return "compose" if st["attempts"] < MAX_FIX and st["draft"] else "fallback"


def finish(st: S) -> dict:
    src = sorted({k.split(":")[0] for k in st.get("numeric", {})} | ({"expert_views"} if st.get("expert") else set()) | ({"ontology_attribution"} if st.get("ontology") else set()))
    facts = "\n".join(f"- {x}" for x in st["facts"])               # 작은 모델은 숫자의 이름표를 잘못 읽을 수 있음(설계 6-3) → 원문을 항상 함께
    return {"answer": st["draft"].strip() + f"\n\n[근거 사실 — 도구 원문, 요약과 다르면 이쪽이 맞음]\n{facts}\n(출처 도구: {', '.join(src)} · 요약의 숫자는 도구 결과와 대조 완료)",
            "trace": st["trace"] + ["finish"]}


def fallback(st: S) -> dict:
    """LLM이 없거나 숫자 검증을 통과하지 못하면 — 사실 목록 그대로 (숫자 정확성 우선)."""
    head = "가격 방향 예측은 검증상 근거가 없어 범위·판단 점수로 답합니다." if st["route"] == "direction" else "도구 결과 요약:"
    return {"answer": head + "\n" + "\n".join(f"- {x}" for x in st["facts"]) + "\n(로컬 LLM 문장화 생략 — 사실 목록 그대로)", "trace": st["trace"] + ["fallback"]}


# ============================================================================
# ② 전문가 의견 기록 경로 — 추출 → 사람 확인 → 저장
# ============================================================================
VIEW_SCHEMA = {"type": "object", "properties": {"item": {"type": "string"}, "direction": {"type": "number"}, "confidence": {"type": "number"},
                                                 "horizon_weeks": {"type": "integer"}, "author": {"type": "string"}},
               "required": ["item", "direction", "confidence"]}


def extract_view(st: S) -> dict:
    q = st["question"]
    text = re.split(r"[:：]", q, 1)[-1].strip()
    raw = None if os.environ.get("LLM_OFF") == "1" else llm([
        {"role": "system", "content": "시황 의견 문장에서 품목, 방향(−1 강한 약세 ~ +1 강한 강세), 신뢰도(0~1), 기간(주), 작성자를 JSON으로만 뽑는다."},
        {"role": "user", "content": text}], fmt=VIEW_SCHEMA)
    v = {}
    if raw:
        try:
            v = json.loads(raw)
        except json.JSONDecodeError:
            v = {}
    ql = text.lower()                                            # 규칙 기반 보정·대체
    rule_dir = 0.5 if re.search(r"강세|상승|오름|상방|매수", ql) else (-0.5 if re.search(r"약세|하락|내림|하방|매도", ql) else 0.0)
    item = st["items"][0] if st["items"] else None
    view = dict(item=T.name_of(item) if item else str(v.get("item", "")),
                direction=max(-1.0, min(1.0, float(v.get("direction", rule_dir)))),
                confidence=max(0.0, min(1.0, float(v.get("confidence", 0.5)))),
                horizon_weeks=int(v.get("horizon_weeks") or 4),
                author=str(st["params"].get("author") or v.get("author") or "미상"), text=text)
    if not item:
        view["item_unresolved"] = True
    return {"view": view, "trace": st["trace"] + ["extract_view"]}


def approve_view(st: S) -> dict:
    ok = interrupt({"확인": "이 의견을 data/expert/views.csv에 저장할까요?", "view": st["view"]})   # 사람이 Command(resume=True/False)로 답함
    return {"approved": bool(ok), "trace": st["trace"] + ["approve_view"]}


def save_view(st: S, tools: Tools) -> dict:
    v = st["view"]
    if not st.get("approved"):
        return {"answer": "저장하지 않았습니다(사용자 취소).", "trace": st["trace"] + ["save_view"]}
    if v.get("item_unresolved"):
        return {"answer": "품목을 알아보지 못해 저장하지 않았습니다 — 품목 이름을 넣어 다시 요청해 주세요.", "trace": st["trace"] + ["save_view"]}
    r = tools("save_expert_view", item=v["item"], direction=v["direction"], confidence=v["confidence"], text=v["text"],
              author=v["author"], horizon_weeks=v["horizon_weeks"])
    msg = r.get("error") or f"저장함: {v['item']} 방향 {v['direction']:+.2f} · 신뢰도 {v['confidence']:.2f} · {v['horizon_weeks']}주 (누적 {r['total_views']}건, 백테스트 전에는 판단에 미반영)"
    return {"answer": msg, "trace": st["trace"] + ["save_view"]}


# ============================================================================
# 자료 갱신 경로 — 로컬 조회(필요 여부 · 계획) → 사람 확인 → 외부 수집 실행
# ============================================================================
def check_data(st: S, tools: Tools) -> dict:
    """인터넷 접속 없이 data_status로 갱신 필요 자료를 보고, 질문에 맞는 대상(가격 · 리포트)과 실행 계획을 만든다."""
    ql = st["question"].lower()
    s_ = tools("data_status")
    if re.search(r"리포트|보고서|wasde|steo|usda|eia", ql):
        tg = ["reports"]
    elif re.search(r"가격|시세|거시|cot", ql):
        tg = ["prices"]
    else:
        tg = s_.get("suggest_refresh") or ["prices"]
    plans = {t: tools("refresh_data", target=t)["plan"] for t in tg}    # confirm 없이 = 계획만 (실행 안 함)
    return {"plan": {"targets": tg, "plans": plans, "stale": s_.get("stale", []), "status": s_.get("table", [])},
            "trace": st["trace"] + ["check_data"]}


def approve_refresh(st: S) -> dict:
    p = st["plan"]
    ok = interrupt({"확인": "아래 계획대로 외부 사이트에 접속해 자료를 갱신할까요? (data/ 파일을 덮어씀)", "plan": p})
    return {"approved": bool(ok), "trace": st["trace"] + ["approve_refresh"]}


def run_refresh(st: S, tools: Tools) -> dict:
    p = st["plan"]
    if not st.get("approved"):
        stale = ", ".join(p["stale"]) or "없음"
        return {"answer": f"갱신하지 않았습니다(사용자 취소). 현재 갱신 필요 자료: {stale}", "trace": st["trace"] + ["run_refresh"]}
    msgs = []
    for t in p["targets"]:
        r = tools("refresh_data", target=t, confirm=True)
        if "error" in r:
            msgs.append(f"{t}: 실패 — {r['error']}"); continue
        fail = next((x for x in r["log"] if not x["ok"]), None)
        msgs.append(f"{t}: {'완료' if r['ok'] else '실패 — ' + fail['step'] + ' 단계'} ({len(r['log'])}단계, {sum(x['sec'] for x in r['log']):.0f}초)")
    return {"answer": "자료 갱신 결과 — " + " · ".join(msgs), "trace": st["trace"] + ["run_refresh"]}


# ============================================================================
# 그래프 조립
# ============================================================================
def build(backend: str = "direct", checkpointer=None):
    tools = Tools(backend)
    g = StateGraph(S)
    g.add_node("classify", classify)
    g.add_node("numeric", lambda s: numeric_layer(s, tools))     # ①
    g.add_node("expert", lambda s: expert_layer(s, tools))       # ②
    g.add_node("ontology", lambda s: ontology_layer(s, tools))   # ③
    g.add_node("compose", compose)                               # ④
    g.add_node("verify", verify)
    g.add_node("finish", finish)
    g.add_node("fallback", fallback)
    g.add_node("extract_view", extract_view)
    g.add_node("approve_view", approve_view)
    g.add_node("save_view", lambda s: save_view(s, tools))
    g.add_node("check_data", lambda s: check_data(s, tools))
    g.add_node("approve_refresh", approve_refresh)
    g.add_node("run_refresh", lambda s: run_refresh(s, tools))
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", lambda s: {"expert_add": "extract_view", "refresh": "check_data"}.get(s["route"], "numeric"),
                            ["numeric", "extract_view", "check_data"])
    g.add_edge("numeric", "expert")
    g.add_edge("expert", "ontology")
    g.add_edge("ontology", "compose")
    g.add_edge("compose", "verify")
    g.add_conditional_edges("verify", after_verify, ["compose", "finish", "fallback"])
    g.add_edge("finish", END)
    g.add_edge("fallback", END)
    g.add_edge("extract_view", "approve_view")
    g.add_edge("approve_view", "save_view")
    g.add_edge("save_view", END)
    g.add_edge("check_data", "approve_refresh")
    g.add_edge("approve_refresh", "run_refresh")
    g.add_edge("run_refresh", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def ask(question: str, backend: str = "direct", thread: str = "cli", approve=None, app=None) -> dict:
    """질문 1개 실행. 사람 확인 지점(의견 저장 · 자료 갱신)에서 approve(내용) → True/False 로 대신 답함(없으면 콘솔에서 묻기).
    내용 = 의견 기록이면 view, 자료 갱신이면 plan(대상 · 단계 · 접속 사이트 · 갱신 필요 자료)."""
    app = app or build(backend)
    cfg = {"configurable": {"thread_id": thread}}
    out = app.invoke({"question": question}, cfg)
    while "__interrupt__" in out:
        req = out["__interrupt__"][0].value
        body = req.get("view", req.get("plan"))
        yes = approve(body) if approve else input(f"{json.dumps(body, ensure_ascii=False, default=str)}\n{req['확인']} [y/N] ").strip().lower() == "y"
        out = app.invoke(Command(resume=yes), cfg)
    return out


def mermaid(path: Path = ROOT / "docs" / "LLM_graph.mmd") -> str:
    m = build().get_graph().draw_mermaid()                       # 로컬 텍스트 생성 (PNG 변환 서비스 mermaid.ink는 인터넷 전송이라 안 씀)
    path.write_text(m, encoding="utf-8")
    return m


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    a = sys.argv[1:]
    if "--mermaid" in a:
        print(mermaid()); sys.exit()
    if "--no-llm" in a:
        os.environ["LLM_OFF"] = "1"
    q = " ".join(x for x in a if not x.startswith("--")) or "옥수수 20일 VaR는? 매입 예정 1천만 달러"
    o = ask(q, "mcp" if "--mcp" in a else "direct")
    print(o["answer"]); print("\n[경로]", " → ".join(o["trace"]))
