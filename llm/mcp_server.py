"""commodity-lab MCP 서버 — llm/tools.py의 도구를 MCP로 노출 (로컬 전용, stdio).

  · 전송: stdio — 네트워크 포트를 열지 않음. 클라이언트(LangGraph·Claude Desktop 등)가 이 프로세스를 직접 띄움.
  · 읽기 도구 14개(readOnlyHint) + 쓰기 도구 1개(save_expert_view, 클라이언트가 사용자 승인을 받도록 destructiveHint=False·readOnlyHint=False).
  · 감사 기록: 모든 호출(시각·도구·인자·성공 여부·소요 시간)을 data/llm_logs/tool_calls.jsonl에 추가 (사내 질의 포함 가능 → git 제외).
  · 외부 접근 없음: 도구는 data/의 로컬 파일만 읽음.

실행 (보통은 클라이언트가 띄움):  .venv\\Scripts\\python.exe -m llm.mcp_server
점검 (클라이언트로 띄워 보기):     .venv\\Scripts\\python.exe tests\\test_mcp_server.py
"""
from __future__ import annotations

import functools
import json
import time
from datetime import datetime

from mcp.server.mcpserver import MCPServer          # mcp 2.x (1.x의 FastMCP)
from mcp.types import ToolAnnotations

from llm import tools as T

LOG = T.DATA / "llm_logs" / "tool_calls.jsonl"
mcp = MCPServer("commodity-lab", instructions=(
    "원자재(곡물 중심) 시황 분석 도구. 숫자는 반드시 도구 결과에서 인용하고 asof(자료 기준일)와 method를 함께 밝힐 것. "
    "가격 방향을 단정하지 말 것 — 이 시스템의 검증 결과 방향 예측은 우연 수준이며, 변동성·위험·구매 타이밍 판단만 근거가 있음."))


def _logged(fn):
    """호출 기록 + 오류를 문장으로 돌려줌 (LLM이 고쳐서 다시 부를 수 있게)."""
    @functools.wraps(fn)
    def run(*a, **k):
        t0, ok, err = time.time(), True, ""
        try:
            return fn(*a, **k)
        except Exception as e:                       # noqa: BLE001 — 도구 오류는 LLM에 그대로 전달
            ok, err = False, f"{type(e).__name__}: {e}"
            return {"error": err}
        finally:
            LOG.parent.mkdir(parents=True, exist_ok=True)
            with LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"t": datetime.now().isoformat(timespec="seconds"), "tool": fn.__name__, "args": a, "kwargs": k,
                                    "ok": ok, "error": err, "sec": round(time.time() - t0, 2)}, ensure_ascii=False, default=str) + "\n")
    return run


for f in T.READ_TOOLS:
    mcp.add_tool(_logged(f), annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
for f in T.WRITE_TOOLS:
    mcp.add_tool(_logged(f), annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))


if __name__ == "__main__":
    mcp.run()                                         # 기본 stdio
