"""4계층 LangGraph 흐름 — ① 수치 · ② 전문가 의견 · ③ 온톨로지 · ④ LLM 인터페이스 (설계 docs/LLM_설계.md §6).

흐름
  classify ─┬─ (의견 기록) extract_view → approve_view[사람 확인] → save_view ─────────────┐
            └─ (질문)  numeric(①) → expert(②) → ontology(③) → compose(④) → verify ─┬─ 통과 → finish
                                                                    ↑ 숫자 불일치(최대 2회) ┘    ↓ 2회 실패 → 사실 목록 답변
  · 숫자는 도구(llm/tools.py)만 만든다. LLM은 '사실 목록'을 문장으로 엮기만 하고, verify가 답변 숫자를 사실 목록과 대조한다.
  · 질문 분류·품목 추출은 키워드 규칙(작은 로컬 모델의 오분류 방지). LLM은 답변 작성 · 의견 문장 → JSON 추출에만 쓴다.
  · 로컬 LLM = Ollama(127.0.0.1). 꺼져 있거나 실패하면 사실 목록 답변으로 대체 — 클라우드로 넘어가는 경로 없음.
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
    route: str                 # lookup · decision · explain · direction · expert_add
    items: list[str]           # 품목 id
    intents: list[str]         # prices · vol · var · regime · factors · corr · buy · fund
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
KW = {"vol": r"변동성|garch|gjr|출렁", "var": r"var|es\b|손실|위험 금액|리스크 금액|최대 손실", "regime": r"레짐|국면|위기 확률",
      "factors": r"요인|pca|스필오버|전이|동조", "corr": r"상관", "prices": r"가격|시세|얼마|현재가", "buy": r"살까|사야|매수|구매|미룰|사도|타이밍",
      "fund": r"wasde|재고|수급|cot|포지션"}


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
    elif re.search(r"오를까|내릴까|떨어질까|상승할까|하락할까|오를 것|떨어질 것|방향|목표가|얼마까지", q):
        route, intents = "direction", ["buy", "var"]             # 방향은 단정하지 않고 범위·판단 근거로 대신 답
    elif re.search(r"왜|원인|이유|기여|때문", q):
        route = "explain"
        intents = intents or ["prices"]
    elif "buy" in intents:
        route = "decision"; intents = sorted(set(intents) | {"fund"})
    else:
        route = "lookup"; intents = intents or ["prices"]
    if not items and route != "lookup":
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
                    facts.append(f"{nm} 연 변동성 현재 {_f(r['vol_now_ann_pct'])}% (GJR-GARCH), 1년 평균 {_f(r['vol_1y_avg_ann_pct'])}%, 지속성 {_f(r['persistence'], 3)}, 반감기 {_f(r['half_life_days'], 0)}일 (기준일 {r['asof']})")
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
    if st["params"].get("default_item"):
        facts.append("질문에 품목이 없어 옥수수 기준으로 답함")
    return {"numeric": out, "facts": facts, "trace": st["trace"] + ["numeric"]}


def expert_layer(st: S, tools: Tools) -> dict:
    out, facts = {}, list(st["facts"])
    if st["route"] in ("decision", "direction", "explain"):
        for i in st["items"]:
            r = tools("expert_views", item=i, last=5); out[i] = r
            v = r.get("views", [])
            if v:
                avg = sum(x["direction"] * x["confidence"] for x in v) / max(sum(x["confidence"] for x in v), 1e-9)
                facts.append(f"{T.name_of(i)} 전문가 의견 {len(v)}건(최근 {v[0]['date']}), 신뢰도 가중 방향 {avg:+.2f} (−1 약세 ~ +1 강세) — 미검증, 참고만")
            else:
                facts.append(f"{T.name_of(i)} 저장된 전문가 의견 없음")
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
    if st["route"] == "direction" and re.search(r"오를 것|상승할 것|하락할 것|내릴 것|떨어질 것", st["draft"]):
        issues.append("가격 방향을 단정했음 — 단정하지 말 것")
    return {"issues": issues, "trace": st["trace"] + ["verify"]}


def after_verify(st: S) -> str:
    if not st["issues"]:
        return "finish"
    return "compose" if st["attempts"] < MAX_FIX and st["draft"] else "fallback"


def finish(st: S) -> dict:
    src = sorted({k.split(":")[0] for k in st.get("numeric", {})} | ({"expert_views"} if st.get("expert") else set()) | ({"ontology_attribution"} if st.get("ontology") else set()))
    return {"answer": st["draft"].strip() + f"\n\n(출처 도구: {', '.join(src)} · 숫자는 도구 결과와 대조 완료)", "trace": st["trace"] + ["finish"]}


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
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", lambda s: "extract_view" if s["route"] == "expert_add" else "numeric", ["numeric", "extract_view"])
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
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def ask(question: str, backend: str = "direct", thread: str = "cli", approve=None, app=None) -> dict:
    """질문 1개 실행. 의견 기록이면 approve(view) → True/False 로 사람 확인을 대신(없으면 콘솔에서 묻기)."""
    app = app or build(backend)
    cfg = {"configurable": {"thread_id": thread}}
    out = app.invoke({"question": question}, cfg)
    while "__interrupt__" in out:
        req = out["__interrupt__"][0].value
        yes = approve(req["view"]) if approve else input(f"{json.dumps(req['view'], ensure_ascii=False)}\n저장할까요? [y/N] ").strip().lower() == "y"
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
