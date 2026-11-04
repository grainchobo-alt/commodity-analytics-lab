"""백테스트 로그 자동 부분 갱신 — docs/백테스트_로그.md의 <!-- AUTO START --> ~ <!-- AUTO END --> 사이를 다시 씀.

자동 부분
  ① 결과 파일 목록: data/*_compare.csv · causal_*.csv · strategy_*.csv — 행 수 · 마지막 수정일 · 만든 스크립트(그 파일 이름을 쓰는 루트 *.py)
  ② ⑥ 실험 결과 탭 실행 기록: data/experiments_meta.json (실험 이름 · 실행 시각 · 설정)
  ③ 버전별 백테스트 · 실험 이력: versions/CHANGELOG.md에서 결과가 적힌 줄(Sharpe · AUC · t · 위약 · bp · 채택 등)만 버전별로 추림
사람이 쓰는 부분(맨 위 '최근 실험 상세')은 건드리지 않음.
실행: .venv\\Scripts\\python.exe backtest_log.py
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOC = ROOT / "docs" / "백테스트_로그.md"
KEY = re.compile(r"Sharpe|AUC|t값|\bt [−\-]?\d|t≈|위약|bp|채택|미채택|통과|기각|IC|R²|적중|초과율|낙폭|MDD|백테스트|결과|열위|우위|효과 없|개선")


def files_section() -> list[str]:
    import pandas as pd
    py = {p: p.read_text(encoding="utf-8", errors="ignore") for p in ROOT.glob("*.py") if p.name != "backtest_log.py"}
    out = ["### ① 결과 파일 (data/)", "", "| 파일 | 행 | 마지막 수정 | 만든 스크립트 |", "|---|---|---|---|"]
    fs = sorted(set(ROOT.glob("data/*_compare.csv")) | set(ROOT.glob("data/causal_*.csv")) | set(ROOT.glob("data/strategy_*.csv")))
    for f in fs:
        try:
            n = len(pd.read_csv(f, encoding="utf-8-sig"))
        except Exception:
            n = "?"
        stem = f.stem.removesuffix("_compare")
        who = sorted(p.name for p, s in py.items() if f.name in s or f'"{stem}"' in s)
        out.append(f"| `{f.name}` | {n} | {datetime.fromtimestamp(f.stat().st_mtime):%Y-%m-%d} | {', '.join(f'`{w}`' for w in who[:3]) or '–'} |")
    return out + [""]


def meta_section() -> list[str]:
    f = ROOT / "data" / "experiments_meta.json"
    if not f.exists():
        return []
    d = json.loads(f.read_text(encoding="utf-8"))
    out = ["### ② ⑥ 실험 결과 탭 실행 기록 (experiments.py)", "", "| 실험 | 실행 | 설정 |", "|---|---|---|"]
    for k, v in sorted(d.items(), key=lambda kv: kv[1].get("run_at", "")):
        st = "; ".join(f"{a}={b}" for a, b in (v.get("settings") or {}).items())
        out.append(f"| `{k}` | {v.get('run_at', '')} | {st[:160]} |")
    return out + [""]


def changelog_section() -> list[str]:
    lines = (ROOT / "versions" / "CHANGELOG.md").read_text(encoding="utf-8").splitlines()
    out, cur, keep = ["### ③ 버전별 백테스트 · 실험 이력 (CHANGELOG에서 결과 줄만)", ""], None, []

    def flush():
        if cur and keep:
            out.append(f"#### {cur}"); out.extend(keep); out.append("")
    for ln in lines:
        if ln.startswith("## v"):
            flush(); cur, keep = ln[3:].strip(), []
        elif cur and ln.strip().startswith("-") and KEY.search(ln):
            keep.append(ln)
    flush()
    return out


def main():
    auto = ["<!-- AUTO START (backtest_log.py가 다시 씀 — 손으로 고치지 말 것) -->", "",
            f"_자동 생성 {datetime.now():%Y-%m-%d %H:%M}_", ""] + files_section() + meta_section() + changelog_section() + ["<!-- AUTO END -->"]
    s = DOC.read_text(encoding="utf-8") if DOC.exists() else "# 백테스트 로그\n\n<!-- AUTO START -->\n<!-- AUTO END -->\n"
    a, b = s.find("<!-- AUTO START"), s.find("<!-- AUTO END -->")
    s = s[:a] + "\n".join(auto) + s[b + len("<!-- AUTO END -->"):]
    DOC.write_text(s, encoding="utf-8")
    print(f"갱신: {DOC} ({len(auto)}줄 자동 부분)")


if __name__ == "__main__":
    main()
