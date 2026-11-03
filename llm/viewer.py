"""LangGraph 흐름을 브라우저에서 보는 한 파일 HTML 만들기 → docs/LLM_graph.html (인터넷 없이 열림).

노드 · 연결선은 llm/graph.py의 실제 그래프(build().get_graph())에서 읽으므로 코드를 고치면 다시 실행만 하면 됨.
상자를 누르면 하는 일 · 계층 · LLM 사용 여부 · 코드 위치가 오른쪽에 뜸. 점선 = 조건 분기.
실행: .venv\\Scripts\\python.exe -m llm.viewer
"""
from __future__ import annotations

import html
import inspect
import json

from llm import graph as G

OUT = G.ROOT / "docs" / "LLM_graph.html"
POS = {"__start__": (430, 40), "classify": (430, 120),
       "numeric": (250, 215), "expert": (250, 305), "ontology": (250, 395), "compose": (250, 485), "verify": (250, 575),
       "finish": (140, 670), "fallback": (360, 670),
       "extract_view": (560, 215), "approve_view": (560, 305), "save_view": (560, 395),
       "check_data": (720, 215), "approve_refresh": (720, 305), "run_refresh": (720, 395), "__end__": (430, 760)}
LAYER = {"classify": ("분류", "c0"), "numeric": ("① 수치", "c1"), "expert": ("② 전문가 의견", "c2"), "ontology": ("③ 온톨로지", "c3"),
         "compose": ("④ LLM 인터페이스", "c4"), "verify": ("④ LLM 인터페이스", "c4"), "finish": ("④ LLM 인터페이스", "c4"), "fallback": ("④ LLM 인터페이스", "c4"),
         "extract_view": ("② 의견 기록", "c2"), "approve_view": ("② 의견 기록 · 사람 확인", "c5"), "save_view": ("② 의견 기록", "c2"),
         "check_data": ("① 자료 갱신 (로컬 조회)", "c1"), "approve_refresh": ("① 자료 갱신 · 사람 확인", "c5"), "run_refresh": ("① 자료 갱신 (외부 접속)", "c1")}
INFO = {
    "classify": ("질문 유형(조회 · 판단 · 설명 · 방향 · 의견 기록) · 품목 · 인자(기간 · 신뢰수준 · 금액 · 매입 예정)를 키워드 규칙으로 뽑음. 작은 로컬 모델에 분류를 맡기지 않음.", "없음", "classify"),
    "numeric": ("유형에 맞는 도구 호출(가격 · GJR 변동성 · VaR/ES · 레짐 · 요인 · 상관 · 매수 점수 · WASDE · DCC 상관 · BEKK 전이 · 칼만 β · 머신러닝 27종 성적 · 전략 비중 · 문서 감성 · 자료 기준일) → 숫자 · 기준일 · 방법이 든 '사실 목록'. 숫자는 여기서만 만듦. 도구마다 대시보드 어느 탭과 같은 값인지 결과에 붙음.", "없음", "numeric_layer"),
    "expert": ("판단 · 방향 · 설명 질문에 의견원별 최근 방향 + 백테스트 판정(의견 방향 vs 이후 4주 수익률). 사내 의견 · USDA 가격 전망 수정 · CFTC 펀드 포지션. '채택'이 아니면 참고·설명용으로 표시.", "없음", "expert_layer"),
    "ontology": ("'왜 움직였나' 질문에 곡물 2단 온톨로지로 기간 누적 변동을 원인별 기여로 분해 + 유의한 연결. 온톨로지 밖 품목은 그렇다고 밝힘.", "없음", "ontology_layer"),
    "compose": ("사실 목록을 결론 → 근거 → 한계 문장으로(로컬 Ollama, 127.0.0.1). 숫자는 사실 목록 것만, 가격 방향 단정 금지. 재작성 때는 고칠 점을 함께 받음.", "있음 (문장 작성)", "compose"),
    "verify": ("세 가지 검사: ① 답변 숫자가 사실 목록 · 질문에 있는지(반올림 허용) ② 가격 방향 단정 표현(모든 질문) ③ 질문이 꼭 답해야 할 핵심 숫자(매수 점수 · VaR · 현재 변동성 · 기간 누적 변동) 포함. 어기면 고칠 점을 붙여 compose로(최대 2회).", "없음", "verify"),
    "finish": ("검증 통과 요약 + 근거 사실 원문(도구 결과) + 출처 도구. 작은 모델이 숫자의 이름표를 잘못 읽는 경우가 있어 원문을 항상 함께 보여 줌.", "없음", "finish"),
    "fallback": ("LLM이 꺼졌거나 2회 모두 검증 실패 → 사실 목록 그대로 답(숫자 정확성 우선).", "없음", "fallback"),
    "extract_view": ("'전문가 의견: …' 문장에서 품목 · 방향(−1~+1) · 신뢰도 · 기간 · 작성자 추출(LLM JSON + 규칙 보정).", "있음 (JSON 추출)", "extract_view"),
    "approve_view": ("저장 전 사람 확인(LangGraph interrupt). 흐름이 여기서 멈추고, 승인/거절 답을 받아 이어감.", "없음", "approve_view"),
    "save_view": ("승인됐을 때만 data/expert/views.csv에 추가(사내 정보, git 제외).", "없음", "save_view"),
    "check_data": ("'자료 갱신해줘' 요청 → 인터넷 접속 없이 로컬 파일로 자료원별 마지막 날짜 · 갱신 필요 여부를 보고, 대상(가격 · 리포트)별 실행 계획(단계 · 접속할 외부 사이트 · 예상 시간)을 만듦.", "없음", "check_data"),
    "approve_refresh": ("외부 접속 전 사람 확인(LangGraph interrupt). 계획을 보여 주고 승인/거절을 받음 — LLM이 혼자 외부에 접속하지 못하게 하는 지점.", "없음", "approve_refresh"),
    "run_refresh": ("승인됐을 때만 refresh_data(confirm=True)로 수집기 실행(collect → clean → roll_adjust → data_quality → export_excel, 또는 reports.py). 실패하면 그 단계에서 멈춤.", "없음", "run_refresh"),
    "__start__": ("질문이 들어옴.", "—", None), "__end__": ("답변을 돌려줌.", "—", None),
}


def build_html() -> str:
    g = G.build().get_graph()
    src = inspect.getsource(G).splitlines()
    nodes = []
    for nid in g.nodes:
        x, y = POS[nid]
        lay, cls = LAYER.get(nid, ("시작 · 끝", "c9"))
        d, use, fn = INFO[nid]
        line = next((k + 1 for k, l in enumerate(src) if fn and l.startswith(f"def {fn}(")), None)
        nodes.append(dict(id=nid, x=x, y=y, layer=lay, cls=cls, d=d, llm=use, code=f"llm/graph.py:{line} {fn}()" if line else ""))
    edges = [dict(a=e.source, b=e.target, cond=bool(e.conditional)) for e in g.edges]
    data = json.dumps(dict(nodes=nodes, edges=edges), ensure_ascii=False)
    return TEMPLATE.replace("/*DATA*/", data).replace("@MERMAID", html.escape(g.draw_mermaid()))


TEMPLATE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LangGraph 4계층 흐름</title>
<style>
:root{--bg:#f7f8fa;--card:#fff;--ink:#1d2430;--ink2:#5b6575;--line:#d5dae2;--accent:#e4572e;
 --c0:#64748b;--c1:#2563eb;--c2:#0d9488;--c3:#9333ea;--c4:#e4572e;--c5:#ca8a04;--c9:#94a3b8}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f1318;--card:#161b22;--ink:#e6e9ee;--ink2:#9aa4b2;--line:#2b333d}}
:root[data-theme="dark"]{--bg:#0f1318;--card:#161b22;--ink:#e6e9ee;--ink2:#9aa4b2;--line:#2b333d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 "Malgun Gothic","Segoe UI",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:18px 16px 40px}
h1{font-size:21px;margin:0 0 4px}.sub{color:var(--ink2);margin:0 0 14px}
.grid{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:14px}
@media (max-width:860px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
svg{width:100%;height:auto;display:block}
.node rect{fill:var(--card);stroke-width:2;cursor:pointer}.node text{fill:var(--ink);font-size:13px;pointer-events:none}
.node .lay{font-size:10.5px;fill:var(--ink2)}.node.sel rect{stroke-width:4}
.edge{fill:none;stroke:var(--ink2);stroke-width:1.6}.edge.cond{stroke-dasharray:6 4}.edge.hi{stroke:var(--accent);stroke-width:2.6}
.lg{display:flex;flex-wrap:wrap;gap:6px 14px;margin:8px 0 0;color:var(--ink2);font-size:12.5px}.lg i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.panel h2{font-size:16px;margin:0 0 6px}.panel .k{color:var(--ink2);font-size:12px;margin-top:10px}.panel code{font-size:12.5px}
details{margin-top:14px}pre{white-space:pre-wrap;font-size:12px;background:var(--bg);padding:10px;border-radius:8px;border:1px solid var(--line);overflow:auto}
ul{padding-left:18px;margin:6px 0}
</style></head><body><div class="wrap">
<h1>LangGraph 4계층 흐름 — 수치 · 전문가 의견 · 온톨로지 · LLM 인터페이스</h1>
<p class="sub">llm/graph.py의 실제 그래프에서 그린 그림. 상자를 누르면 설명. 점선 = 조건 분기. 인터넷 없이 열림 · 대시보드/발표에는 미반영(향후 계획).</p>
<div class="grid">
 <div class="card"><svg id="g" viewBox="0 0 820 800" role="img" aria-label="LangGraph 흐름도"></svg>
  <div class="lg"><span><i style="background:var(--c1)"></i>① 수치</span><span><i style="background:var(--c2)"></i>② 전문가 의견</span><span><i style="background:var(--c3)"></i>③ 온톨로지</span><span><i style="background:var(--c4)"></i>④ LLM 인터페이스</span><span><i style="background:var(--c5)"></i>사람 확인</span><span><i style="background:var(--c0)"></i>분류</span></div></div>
 <div class="card panel" id="p"><h2>상자를 눌러 보세요</h2>
  <ul><li>숫자는 ① ② ③ 도구만 만들고, ④ LLM은 문장으로 엮기만 함</li><li>verify가 답변 숫자를 사실 목록과 대조 → 틀리면 다시 쓰게 함(최대 2회)</li>
  <li>의견 기록은 사람 확인(approve_view) 후에만 저장</li><li>로컬 Ollama(127.0.0.1)만 · LangSmith 추적 끔</li></ul>
  <div class="k">실행</div><code>.venv\Scripts\python.exe -m llm.graph "대두박 지금 살까?"</code></div>
</div>
<details><summary>Mermaid 원문 (docs/LLM_graph.mmd와 같음)</summary><pre>@MERMAID</pre></details>
</div>
<script>
const D=/*DATA*/;const NS="http://www.w3.org/2000/svg",svg=document.getElementById("g"),W=150,H=46,N={};
const el=(t,a,p)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);(p||svg).appendChild(e);return e};
const defs=el("defs",{});const m=el("marker",{id:"ar",viewBox:"0 0 10 10",refX:"9",refY:"5",markerWidth:"7",markerHeight:"7",orient:"auto-start-reverse"},defs);el("path",{d:"M0,0L10,5L0,10z",fill:"currentColor"},m);
D.nodes.forEach(n=>N[n.id]=n);const gE=el("g",{}),gN=el("g",{});
function path(a,b){const A=N[a],B=N[b];
 if(a==="verify"&&b==="compose")return `M${A.x-W/2},${A.y} C${A.x-W/2-70},${A.y} ${B.x-W/2-70},${B.y} ${B.x-W/2},${B.y}`;
 const y1=A.y+(B.y>A.y?H/2:-H/2),y2=B.y+(B.y>A.y?-H/2:H/2);const my=(y1+y2)/2;return `M${A.x},${y1} C${A.x},${my} ${B.x},${my} ${B.x},${y2}`}
const E=D.edges.map(e=>({...e,p:el("path",{d:path(e.a,e.b),class:"edge"+(e.cond?" cond":""),"marker-end":"url(#ar)",style:"color:var(--ink2)"},gE)}));
D.nodes.forEach(n=>{const g=el("g",{class:"node",tabindex:"0"},gN);const end=n.id.startsWith("__");
 el("rect",{x:n.x-W/2,y:n.y-H/2,width:W,height:H,rx:end?23:9,style:`stroke:var(--${n.cls})`},g);
 el("text",{x:n.x,y:n.y-2,"text-anchor":"middle"},g).textContent=end?(n.id==="__start__"?"시작 (질문)":"끝 (답변)"):n.id;
 if(!end)el("text",{x:n.x,y:n.y+14,"text-anchor":"middle",class:"lay"},g).textContent=n.layer;
 const pick=()=>{gN.querySelectorAll(".node").forEach(x=>x.classList.remove("sel"));g.classList.add("sel");
  E.forEach(e=>e.p.classList.toggle("hi",e.a===n.id||e.b===n.id));
  const nx=E.filter(e=>e.a===n.id).map(e=>e.b+(e.cond?" (조건)":"")).join(", ")||"—";
  document.getElementById("p").innerHTML=`<h2>${n.id}</h2><div>${n.d}</div><div class="k">계층</div>${n.layer}<div class="k">LLM 사용</div>${n.llm}`+
   (n.code?`<div class="k">코드</div><code>${n.code}</code>`:"")+`<div class="k">다음</div>${nx}`};
 g.addEventListener("click",pick);g.addEventListener("keydown",ev=>{if(ev.key==="Enter")pick()})});
</script></body></html>"""


if __name__ == "__main__":
    OUT.write_text(build_html(), encoding="utf-8")
    print(OUT)
