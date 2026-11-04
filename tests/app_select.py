"""앱 테스트 선택: 바뀐 파일 → tests/test_app.py의 어느 묶음을 돌려야 하나 (push_check.py · test_app.py --auto가 공통 사용).

규칙
  · app.py: 바뀐 줄이 속한 최상위 함수 → 그 함수를 직접·간접으로 부르는 탭 함수 → 묶음. 탭 밖 최상위 코드가 바뀌면 전체.
  · 묶음이 없는 탭 함수(예: ⑧ 데이터·방법론 문구)는 '초기 렌더'만으로 확인(초기 렌더가 모든 탭을 한 번 그림).
  · 분석 모듈: 아래 MODULES 대응표. 표에 없는 루트 *.py · requirements · data/*_compare.csv는 안전하게 전체.
  · tests/ 중 앱 테스트 파일(test_app.py · app_select.py)이 바뀌면 전체, 다른 테스트(LLM 등)는 앱 테스트와 무관.
"""
from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALL = "전체"

# 묶음 → 그 묶음이 누르는 위젯이 있는 앱 함수 (tests/test_app.py GROUPS와 같은 이름)
GROUP_FUNCS = {
    "구매": ["tab_procure", "tab_ls", "_alt_ts_section"],
    "리스크": ["tab_risk", "vol_forecast_check", "vol_regime", "tab_cross", "tab_comove"],
    "실험": ["tab_experiments"],
    "요인·레짐": ["tab_factor", "tab_vol", "tab_regime", "_rv_hmm", "_rv_reg"],
    "다변량GARCH": ["tab_mvgarch"],
    "ML": ["tab_ml", "ex_runs"],
    "포트폴리오": ["tab_portfolio", "tab_custom"],
    "패널": ["tab_panel"],
    "MIDAS": ["tab_midas"],
    "FAVAR·월간": ["tab_favar"],
    "설정": ["page_settings"],
}
FUNC_GROUP = {f: g for g, fs in GROUP_FUNCS.items() for f in fs}
# 분석 모듈 → 묶음 (그 모듈을 쓰는 화면을 누르는 묶음)
MODULES = {
    "procurement.py": ["구매", "실험"], "risk.py": ["리스크", "요인·레짐"], "mvgarch.py": ["다변량GARCH"],
    "ls_strategies.py": ["구매", "포트폴리오", "리스크"], "ls_extra.py": ["구매", "포트폴리오"], "crosssec.py": ["리스크"],
    "experiments.py": ["실험"], "regime.py": ["요인·레짐"], "regime_mv.py": ["요인·레짐"], "panel.py": ["패널"], "midas.py": ["MIDAS", "ML"],
    "portfolio.py": ["포트폴리오"], "runlog.py": ["ML"], "predict_lab.py": ["ML", "구매"], "models_dense.py": ["ML"], "models_ml.py": ["ML"],
    "alt_ts.py": ["구매", "ML"], "multimodal.py": ["구매", "ML"], "metasel.py": ["실험"], "validation.py": ["실험", "ML"],
    "model_atlas.py": [], "export_excel.py": [], "push_check.py": [],
    # 앱이 import하지 않는 배치 · 수집 스크립트 (앱 화면과 무관)
    "collect.py": [], "collect_carry.py": [], "collect_reddit.py": [], "clean.py": [], "roll_adjust.py": [], "data_quality.py": [],
    "build_dashboard.py": [], "setup_check.py": [], "test_sources.py": [], "foreign_test.py": [],
}
APP_TESTS = {"tests/test_app.py", "tests/app_select.py"}


def git(*a) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, encoding="utf-8").stdout


def _top_defs(src: str) -> list[tuple[str, int, int]]:
    """최상위 함수 (이름, 시작 줄, 끝 줄). 데코레이터 줄 포함."""
    out = []
    for n in ast.parse(src).body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = min([n.lineno] + [d.lineno for d in n.decorator_list])
            out.append((n.name, start, n.end_lineno))
    return out


def _changed_lines(base: str, path: str) -> list[int]:
    """base..작업 트리 기준 새 파일의 바뀐 줄 번호 (지운 줄은 그 자리 줄)."""
    lines = []
    for m in re.finditer(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", git("diff", "-U0", base, "--", path), re.M):
        s, n = int(m.group(1)), int(m.group(2) or 1)
        lines += list(range(s, s + max(n, 1)))
    return lines


def app_groups(base: str) -> tuple[set[str] | str, list[str]]:
    """app.py 변경 → 묶음 집합 (또는 ALL), 이유 목록."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    defs = _top_defs(src)
    body = {name: "\n".join(src.splitlines()[s - 1:e]) for name, s, e in defs}
    changed = _changed_lines(base, "app.py")
    if not changed:
        return set(), []
    funcs, why = set(), []
    for ln in changed:
        hit = next((name for name, s, e in defs if s <= ln <= e), None)
        if hit is None:
            return ALL, [f"app.py {ln}줄: 함수 밖 최상위 코드 → 전체"]
        funcs.add(hit)
    groups = set()
    for f in sorted(funcs):
        seen, todo, gs = {f}, [f], set()
        while todo:                                   # 그 함수를 부르는 함수를 따라 올라가 묶음 · 탭 찾기
            g = todo.pop()
            if g in FUNC_GROUP:
                gs.add(FUNC_GROUP[g]); continue
            if g.startswith("tab_") or g.startswith("page_"):
                continue                              # 묶음 없는 탭 = 초기 렌더로 확인
            callers = [n for n, b in body.items() if n != g and re.search(rf"\b{re.escape(g)}\b", b)]
            if not callers and not g.startswith("tab_"):
                return ALL, [f"app.py {f}(): 부르는 곳을 못 찾음 → 전체"]
            for c in callers:
                if c not in seen:
                    seen.add(c); todo.append(c)
        groups |= gs
        why.append(f"app.py {f}() → {', '.join(sorted(gs)) or '초기 렌더만'}")
    return groups, why


def app_imports() -> set[str]:
    """app.py가 직접·간접으로 import하는 루트 모듈 이름."""
    roots = {p.stem for p in ROOT.glob("*.py")}
    seen, todo = set(), ["app"]
    while todo:
        m = todo.pop()
        f = ROOT / f"{m}.py"
        if not f.exists():
            continue
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            names = [a.name.split(".")[0] for a in n.names] if isinstance(n, ast.Import) else \
                    [n.module.split(".")[0]] if isinstance(n, ast.ImportFrom) and n.module else []
            for x in names:
                if x in roots and x not in seen:
                    seen.add(x); todo.append(x)
    return seen


def required(base: str = "origin/main") -> tuple[set[str] | str, list[str]]:
    """base 이후 바뀐 파일(커밋 + 작업 트리) → 돌릴 묶음. ALL = 전체. 빈 집합 = 초기 렌더만(앱 코드가 바뀐 경우) 또는 앱 테스트 불필요."""
    files = sorted({f for f in git("diff", "--name-only", base).splitlines() if f})
    groups, why = set(), []
    for f in files:
        p = Path(f)
        if f in APP_TESTS or p.name.startswith("requirements") or (p.parts[0] == "data" and p.name.endswith("_compare.csv")):
            return ALL, [f"{f} → 전체"]
        if p.parts[0] == "data" and p.name == "experiments_meta.json":
            groups.add("실험"); why.append(f"{f} → 실험"); continue
        if len(p.parts) == 1 and p.suffix in (".py", ".toml", ".bat"):
            if f == "app.py":
                g, w = app_groups(base)
                if g == ALL:
                    return ALL, w
                groups |= g; why += w
            elif f in MODULES:
                groups |= set(MODULES[f]); why.append(f"{f} → {', '.join(MODULES[f]) or '앱 무관'}")
            elif p.suffix == ".py" and p.stem not in app_imports():
                why.append(f"{f} → 앱 무관 (앱이 import하지 않음)")
            else:
                return ALL, [f"{f}: 대응표에 없는 루트 모듈 → 전체"]
    return groups, why


def app_changed(base: str = "origin/main") -> bool:
    """앱 테스트가 필요한 변경이 있나 (앱 코드 · 앱 테스트 · 앱이 쓰는 data)."""
    for f in (x for x in git("diff", "--name-only", base).splitlines() if x):
        p = Path(f)
        if f in APP_TESTS or p.name.startswith("requirements") or (p.parts[0] == "data" and (p.name.endswith("_compare.csv") or p.name == "experiments_meta.json")):
            return True
        if len(p.parts) == 1 and p.suffix in (".py", ".toml", ".bat") and (f == "app.py" or MODULES.get(f, ["?"])):
            return True
    return False
