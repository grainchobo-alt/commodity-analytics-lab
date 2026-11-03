"""4계층 LangGraph 흐름 검증: 분류 규칙 · 계층별 경로 · 숫자 대조(지어낸 숫자 차단) · 사람 확인 후 저장 · MCP 경유 · Mermaid 그림.
LLM 없이(LLM_OFF=1) 결정적으로 돌리고, 로컬 Ollama가 켜져 있으면 마지막에 실제 답변 1건도 확인."""
import os, sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
os.environ["LLM_OFF"] = "1"
from llm import graph as G, tools as T

t0 = time.time()
def ok(m): print(f"[{time.time()-t0:6.1f}s] OK  {m}")

# 1) 분류 규칙
c = lambda q: G.classify({"question": q})                                          # noqa: E731
assert c("대두박 지금 살까?")["route"] == "decision" and c("대두박 지금 살까?")["items"] == ["yf:ZM=F"]
assert c("밀 다음 달에 오를까?")["route"] == "direction"
assert c("대두박 왜 움직였나?")["route"] == "explain"
assert c("전문가 의견: 옥수수 남미 건조로 강세, 작성자 김과장")["route"] == "expert_add"
o = c("옥수수 20일 99% VaR는? 매입 예정 1천만 달러")
assert o["route"] == "lookup" and "var" in o["intents"] and o["params"] == {"horizon": 20, "alpha": 0.99, "notional_usd": 1e7, "side": "short"}
assert G.find_items("대두박과 대두유, 지금 금 시세") == ["yf:ZM=F", "yf:ZL=F", "yf:GC=F"] and G.find_items("지금 시세") == []
assert c("자료 갱신해줘")["route"] == "refresh" and c("WASDE 리포트 새로 받아줘")["route"] == "refresh"
assert c("자료 갱신 필요해?")["route"] == "lookup" and "status" in c("자료 갱신 필요해?")["intents"]          # 필요 여부만 = 조회
assert "var" not in c("MinVar 전략 비중은?")["intents"] and "strategy" in c("MinVar 전략 비중은?")["intents"]   # 'minvar' ≠ VaR
for q_, k_ in [("옥수수 대두 DCC 상관 추이", "dcc"), ("옥수수 대두 변동성 전이 BEKK", "spill"), ("옥수수 칼만 민감도", "kalman"),
               ("옥수수 머신러닝 모델 비교", "ml"), ("WTI 문서 감성", "text")]:
    assert k_ in c(q_)["intents"], (q_, c(q_)["intents"])
assert c("머신러닝 모델 성능은?")["items"] == ["yf:ZC=F"]                                                  # 품목 없으면 옥수수
ok("분류: 판단·방향·설명·의견 기록·자료 갱신·조회, 새 유형(DCC·BEKK·칼만·머신러닝·전략·감성·자료 상태), 품목·인자")

# 2) 숫자 대조 — 지어낸 숫자 · 방향 단정 차단
st = {"question": "옥수수 VaR", "facts": ["옥수수 20일 99% VaR 15.3% · ES 18.6% (기준일 2026-10-26)"], "params": {}, "route": "lookup"}
assert G.verify({**st, "draft": "VaR 15.3%, ES 18.6% (2026-10-26)", "trace": []})["issues"] == []
assert G.verify({**st, "draft": "VaR 15.30%로 3가지 방법 중", "trace": []})["issues"] == []                   # 반올림 · 작은 정수 허용
assert G.verify({**st, "draft": "VaR 17.2%", "trace": []})["issues"]                                       # 없는 숫자
assert G.verify({**st, "route": "direction", "draft": "다음 달 오를 것으로 보임", "trace": []})["issues"]   # 방향 단정
sb = {"question": "대두박 살까", "route": "decision", "intents": ["buy"], "params": {}, "trace": [], "facts": ["대두박 매수 점수 10.9점(매수 보류), 4주 90% 범위 337.0~437.9"]}
assert G.verify({**sb, "draft": "매수 점수 10.9점. 오를 가능성은 검증상 근거 없음"})["issues"] == []          # 부정문은 단정 아님
iss = G.verify({**sb, "draft": "범위 337.0~437.9, 소폭 하락 가능성이 높습니다"})["issues"]
assert any("방향" in x for x in iss) and any("핵심 숫자 빠짐: 매수 점수" in x for x in iss)                # 판단 질문에도 방향 단정 금지 · 핵심 숫자
se = {"question": "왜", "route": "explain", "intents": ["prices"], "params": {}, "trace": [], "facts": ["대두박 2026-10-05~2026-10-26 누적 +3.2% 중 원인별 기여: 대두 -3.4%p"]}
assert any("기간 누적 변동" in x for x in G.verify({**se, "draft": "대두 -3.4%p"})["issues"])
ok("숫자 대조: 사실 목록 숫자·반올림 통과, 지어낸 숫자·방향 단정(모든 질문)·핵심 숫자 누락은 재작성 요청")

# 3) 질문 경로 (LLM 꺼짐 → 사실 목록 답변)
app = G.build()
def run(q, th):
    return app.invoke({"question": q}, {"configurable": {"thread_id": th}})
o = run("옥수수 20일 VaR는? 매입 예정 1천만 달러", "a")
assert o["trace"][:4] == ["classify", "numeric", "expert", "ontology"] and o["trace"][-1] == "fallback"
f = o["facts"][0]; assert "VaR" in f and "$" in f and "매입 예정" in f and "2026-" in f
v = T.var_es("옥수수", 0.99, 20, "short", 1e7)["now"]; fhs = next(x for x in v if x["방법"].startswith("FHS"))
assert f"{fhs['VaR (%)']:.1f}%" in f
ok(f"① 수치: VaR 사실 = 도구 결과 ({fhs['VaR (%)']:.1f}%), 금액·기준일 포함")
o = run("대두박 지금 살까?", "b")
assert any("매수 점수" in x for x in o["facts"]) and any("전문가 의견" in x for x in o["facts"]) and any("WASDE" in x for x in o["facts"])
ok("① + ② 판단: 매수 점수 · WASDE 맥락 · 전문가 의견 상태")
o = run("대두박 왜 움직였나?", "c")
assert any("원인별 기여" in x for x in o["facts"]) and any("유의한 연결" in x and "대두" in x for x in o["facts"])
o2 = run("구리 왜 올랐어?", "c2"); assert any("온톨로지" in x and "밖" in x for x in o2["facts"])
ok("③ 온톨로지: 대두박 최근 4주 원인 기여 · 유의 연결(대두), 온톨로지 밖 품목은 밝힘")
o = run("밀 다음 달에 오를까?", "d")
assert o["answer"].startswith("가격 방향 예측은 검증상 근거가 없어") and any("4주 90% 범위" in x for x in o["facts"])
ok("방향 질문: 단정 대신 범위·매수 점수로")
o = run("옥수수 대두 DCC 상관 추이와 칼만 민감도, 머신러닝 모델 비교", "n1")
assert any("DCC 동적 상관" in x for x in o["facts"]) and any("칼만 시변 민감도" in x for x in o["facts"]) and any("OOS R²" in x for x in o["facts"])
o = run("이번 주 MinVar 포지션 비중, 문서 감성, 자료 기준일", "n2")
assert any("순 비중" in x for x in o["facts"]) and any("문서 감성" in x for x in o["facts"]) and any("자료 기준일" in x for x in o["facts"])
ok("① 새 도구: DCC · 칼만 · 머신러닝 · 전략 비중 · 문서 감성 · 자료 기준일 → 사실 목록")

# 3-1) 자료 갱신 — 로컬 조회 → 사람 확인 → (승인 시에만) 외부 수집. 시험에서는 수집기 대신 무해한 로컬 스크립트로 바꿔 실행
noop = Path(__file__).parent / "_tmp_noop.py"; noop.write_text("print('ok')", encoding="utf-8")
bak_r = T.REFRESH["prices"]; T.REFRESH["prices"] = {**bak_r, "steps": [str(noop)]}
try:
    seen = []
    o = G.ask("가격 자료 갱신해줘", approve=lambda p: seen.append(p) or False, app=app, thread="r1")
    assert "취소" in o["answer"] and seen[0]["targets"] == ["prices"] and seen[0]["plans"]["prices"]["hosts"]
    assert o["trace"] == ["classify", "check_data", "approve_refresh", "run_refresh"]
    o = G.ask("가격 자료 갱신해줘", approve=lambda p: True, app=app, thread="r2")
    assert "완료" in o["answer"], o["answer"]
finally:
    T.REFRESH["prices"] = bak_r; noop.unlink(missing_ok=True)
ok("자료 갱신: 계획(단계 · 접속 사이트)을 보여 주고 거절하면 실행 안 함 · 승인하면 실행")

# 4) 전문가 의견 기록 — 사람 확인(interrupt) 후에만 저장
bak = T.EXPERT; T.EXPERT = Path(__file__).parent / "_tmp_views_graph.csv"
try:
    seen = []
    o = G.ask("전문가 의견: 옥수수 남미 건조로 4주 강세 예상, 작성자 김과장", approve=lambda v: seen.append(v) or False, app=app, thread="e1")
    assert "취소" in o["answer"] and not T.EXPERT.exists() and seen[0]["direction"] > 0 and seen[0]["author"] == "김과장"
    o = G.ask("전문가 의견: 옥수수 남미 건조로 4주 강세 예상, 작성자 김과장", approve=lambda v: True, app=app, thread="e2")
    assert "저장함" in o["answer"] and T.expert_views("옥수수")["total_views"] == 1
    assert o["trace"] == ["classify", "extract_view", "approve_view", "save_view"]
finally:
    T.EXPERT.unlink(missing_ok=True); T.EXPERT = bak
ok("② 의견 기록: 거절 시 저장 안 함 · 승인 시 저장 (작성자·방향 추출)")

# 5) MCP 경유도 같은 사실
o_m = G.build("mcp").invoke({"question": "옥수수 20일 VaR는? 매입 예정 1천만 달러"}, {"configurable": {"thread_id": "m"}})
assert o_m["facts"] == run("옥수수 20일 VaR는? 매입 예정 1천만 달러", "a2")["facts"]
ok("MCP 서버 경유 = 직접 호출 (같은 사실 목록)")

# 6) Mermaid 흐름도
m = G.mermaid(Path(__file__).parent / "_tmp_graph.mmd")
for n in ("classify", "numeric", "expert", "ontology", "compose", "verify", "approve_view", "check_data", "approve_refresh", "run_refresh"):
    assert n in m
(Path(__file__).parent / "_tmp_graph.mmd").unlink()
ok("Mermaid 흐름도 생성 (노드 14개)")

# 7) (선택) 로컬 LLM 실제 답변
os.environ.pop("LLM_OFF")
if G.llm([{"role": "user", "content": "ok"}], timeout=60) is None:
    print("  (Ollama 꺼짐 — LLM 답변 시험 건너뜀)")
else:
    o = G.build().invoke({"question": "대두박 지금 살까?"}, {"configurable": {"thread_id": "l"}})
    assert o["trace"][-1] in ("finish", "fallback") and o["answer"]
    if o["trace"][-1] == "finish":
        assert not G.verify({**o, "trace": []})["issues"]
    ok(f"로컬 LLM 답변: {o['trace'][-1]} (재작성 {o['attempts'] - 1}회)")
print(f"\n모두 통과 ({time.time()-t0:.0f}s)")
