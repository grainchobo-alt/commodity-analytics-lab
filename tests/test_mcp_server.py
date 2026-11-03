"""MCP 서버 기동 검증: 실제 클라이언트처럼 stdio로 서버를 띄워 도구 목록 · 읽기/쓰기 주석 · 도구 호출 · 오류 전달 · 호출 기록 확인."""
import asyncio, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from llm import tools as T

t0 = time.time()
def ok(m): print(f"[{time.time()-t0:6.1f}s] OK  {m}")
LOG = T.DATA / "llm_logs" / "tool_calls.jsonl"


async def main():
    n0 = len(LOG.read_text(encoding="utf-8").splitlines()) if LOG.exists() else 0
    srv = StdioServerParameters(command=sys.executable, args=["-m", "llm.mcp_server"], cwd=str(ROOT), env={"PYTHONIOENCODING": "utf-8"})
    async with stdio_client(srv) as (rd, wr), ClientSession(rd, wr) as s:
        await s.initialize()
        tl = (await s.list_tools()).tools
        names = {t.name for t in tl}
        assert names == {f.__name__ for f in T.READ_TOOLS + T.WRITE_TOOLS}, names
        ann = {t.name: t.annotations for t in tl}
        assert all(ann[f.__name__].read_only_hint for f in T.READ_TOOLS) and ann["save_expert_view"].read_only_hint is False
        sch = {t.name: t.input_schema for t in tl}
        assert all(t.description for t in tl) and {"item", "model", "horizon"} <= set(sch["vol_forecast"]["properties"])   # 감싼 함수의 인자 형식 유지
        ok(f"서버 기동 · 도구 {len(tl)}개 (읽기 {len(T.READ_TOOLS)} · 쓰기 {len(T.WRITE_TOOLS)}, 설명·입력 형식 있음)")

        r = await s.call_tool("vol_forecast", {"item": "옥수수", "model": "GJR"})
        d = json.loads(r.content[0].text)
        assert d["id"] == "yf:ZC=F" and 0.8 < d["persistence"] < 1 and d["asof"]
        ok(f"도구 호출: 옥수수 GJR 연 변동성 {d['vol_now_ann_pct']}% (기준일 {d['asof']})")

        r = await s.call_tool("get_prices", {"item": "없는품목"})
        d = json.loads(r.content[0].text)
        assert "찾지 못함" in d["error"]
        ok("잘못된 입력 → 오류 문장 전달 (LLM이 고쳐 다시 부를 수 있게)")
    L = LOG.read_text(encoding="utf-8").splitlines()[n0:]
    assert [json.loads(x)["tool"] for x in L] == ["vol_forecast", "get_prices"] and json.loads(L[1])["ok"] is False
    ok("호출 기록 data/llm_logs/tool_calls.jsonl (성공·실패 모두)")

asyncio.run(main())
print(f"\n모두 통과 ({time.time()-t0:.0f}s)")
