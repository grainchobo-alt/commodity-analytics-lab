"""평가 세트 실행 — 모델 크기별 숫자 일치율 · 검증 통과율 · 방향 단정 · 응답 시간 비교.

질문마다 도구(① ② ③)는 한 번만 실행해 같은 사실 목록을 만들고, 모델별로 ④ 작성 → 숫자 대조(최대 2회 재작성)만 바꿔 돌린다.
지표
  분류 정확도   : 규칙 분류의 유형 · 품목이 기대와 같은 비율 (모델과 무관)
  사실 포함률   : 기대 키워드가 사실 목록에 모두 있는 비율 (모델과 무관)
  숫자 일치율   : 첫 답변에 나온 숫자 중 사실 목록 · 질문에 있는 숫자의 비율 (10 미만 정수 제외) — 모델 성능의 핵심
  첫 통과율     : 첫 답변이 숫자 대조 · 방향 규칙을 바로 통과한 비율
  최종 문장 답변 : 재작성까지 포함해 LLM 문장 답변(finish)으로 끝난 비율 (나머지는 사실 목록 대체)
  방향 단정     : 방향 질문에서 오른다/내린다고 단정한 첫 답변 수
  핵심 사실 반영 : 질문 종류별로 꼭 들어가야 할 숫자(KEY_RE — VaR %, 매수 점수, 현재 변동성, 누적 변동 등)가 최종 답변에 있는 비율
  응답 시간     : LLM 작성 + 재작성 시간(초), 도구 시간은 따로
실행: .venv\\Scripts\\python.exe -m llm.eval.run_eval [모델 ...]   (기본: 설치된 qwen3.5:2b-q4_K_M · qwen3.5:4b)
결과: data/llm_eval/results.csv (질문 × 모델) · summary.csv (모델별)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from llm import graph as G

HERE = Path(__file__).resolve().parent
OUT = G.ROOT / "data" / "llm_eval"
DEFAULT_MODELS = ["qwen3.5:2b-q4_K_M", "qwen3.5:4b"]
KEY_RE = {"위험 금액": [r"VaR ([\d.]+)%"], "변동성": [r"현재 ([\d.]+)%"], "레짐": [r"확률 ([\d.]+)%"],
          "요인·상관": [r"몫 ([\d.]+)%", r"최근 \d+주 ([-\d.]+)"], "구매 판단": [r"매수 점수 ([\d.]+)점"], "방향 질문": [r"매수 점수 ([\d.]+)점"],
          "원인 설명": [r"누적 ([-+][\d.]+)%", r"기여: [^,]*? ([-+][\d.]+)%p"]}


def key_coverage(st: dict, answer: str, group: str) -> float:
    """그 질문이 꼭 답해야 할 숫자(사실 목록에서 KEY_RE로 뽑음)가 답변에 있는 비율. 해당 숫자가 없으면 1."""
    import re
    keys = [float(m) for pat in KEY_RE.get(group, []) for f in st["facts"] for m in re.findall(pat, f)]
    if not keys:
        return 1.0
    xs = G._nums(answer)
    return sum(any(abs(abs(x) - abs(y)) <= max(0.051, 0.006 * abs(y)) for x in xs) for y in keys) / len(keys)


def load() -> list[dict]:
    return [json.loads(x) for x in (HERE / "questions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def prepare(e: dict, tools: G.Tools) -> tuple[dict, float]:
    t0 = time.time()
    st = {"question": e["q"]}
    for fn in (G.classify, lambda s: G.numeric_layer(s, tools), lambda s: G.expert_layer(s, tools), lambda s: G.ontology_layer(s, tools)):
        st.update(fn(st))
    return st, time.time() - t0


def number_precision(st: dict, draft: str) -> tuple[int, int]:
    ok = G._nums(" ".join(st["facts"]) + " " + st["question"]) + [v for v in st["params"].values() if isinstance(v, (int, float))]
    xs = [x for x in G._nums(draft) if not (abs(x) < 10 and float(x).is_integer())]
    good = sum(any(abs(abs(x) - abs(y)) <= max(0.051, 0.006 * abs(y)) for y in ok) for x in xs)
    return good, len(xs)


def run_model(st: dict, model: str) -> dict:
    G.MODEL = model
    s = dict(st, attempts=0, issues=[])
    t0 = time.time(); first = None
    for _ in range(G.MAX_FIX):
        s.update(G.compose(s)); s.update(G.verify(s))
        if first is None:
            first = dict(draft=s["draft"], issues=list(s["issues"]))
        if not s["issues"] or not s["draft"]:
            break
    good, n = number_precision(st, first["draft"])
    dir_bad = st["route"] == "direction" and any("방향" in x for x in first["issues"])
    return dict(model=model, llm_sec=time.time() - t0, attempts=s["attempts"], first_pass=not first["issues"] and bool(first["draft"]),
                final="finish" if not s["issues"] else "fallback", nums_ok=good, nums_n=n, dir_assert=dir_bad,
                first_draft=first["draft"], final_draft=s["draft"] if not s["issues"] else "")


def main(models: list[str]):
    OUT.mkdir(parents=True, exist_ok=True)
    tools, rows = G.Tools("direct"), []
    for m in models:                                            # 모델 적재 시간은 응답 시간에서 빼고 따로 기록
        G.MODEL = m; t0 = time.time(); r = G.llm([{"role": "user", "content": "준비"}], timeout=300)
        print(f"[적재] {m}: {time.time() - t0:.1f}s {'OK' if r is not None else '실패'}", flush=True)
    for e in load():
        st, tsec = prepare(e, tools)
        base = dict(id=e["id"], group=e["group"], q=e["q"], route=st["route"], route_ok=st["route"] == e["route"],
                    items_ok=(not e["items"]) or set(st["items"]) == set(e["items"]),
                    facts_ok=all(any(k in f for f in st["facts"]) for k in e["facts"]), tool_sec=tsec, n_facts=len(st["facts"]))
        for m in models:
            r = run_model(st, m)
            r["key_cov"] = key_coverage(st, r["final_draft"] or "\n".join(st["facts"]), e["group"])   # 사실 목록 대체면 전부 포함
            r["facts"] = json.dumps(st["facts"], ensure_ascii=False)
            rows.append({**base, **r})
            print(f"{e['id']:>2} {m:<20} {r['final']:<8} 숫자 {r['nums_ok']}/{r['nums_n']} 첫통과 {r['first_pass']!s:<5} {r['llm_sec']:5.1f}s", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False, encoding="utf-8-sig")   # 중간 저장
    R = pd.DataFrame(rows)
    S = R.groupby("model").apply(lambda g: pd.Series({
        "질문 수": len(g), "분류 정확도(%)": 100 * (g.route_ok & g.items_ok).mean(), "사실 포함률(%)": 100 * g.facts_ok.mean(),
        "숫자 일치율(%)": 100 * g.nums_ok.sum() / max(g.nums_n.sum(), 1), "첫 통과율(%)": 100 * g.first_pass.mean(),
        "최종 문장 답변(%)": 100 * (g.final == "finish").mean(), "핵심 사실 반영(%)": 100 * g.key_cov.mean(), "방향 단정(건)": int(g.dir_assert.sum()),
        "평균 재작성(회)": (g.attempts - 1).mean(), "LLM 시간 중앙값(s)": g.llm_sec.median(), "LLM 시간 90%(s)": g.llm_sec.quantile(0.9),
        "도구 시간 중앙값(s)": g.tool_sec.median()}), include_groups=False).round(1)
    S.to_csv(OUT / "summary.csv", encoding="utf-8-sig")
    print("\n" + S.T.to_string())
    print(f"\n그룹별 최종 문장 답변(%):\n{(100 * R.assign(fin=R.final == 'finish').pivot_table(index='group', columns='model', values='fin')).round(0).to_string()}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main(sys.argv[1:] or DEFAULT_MODELS)
