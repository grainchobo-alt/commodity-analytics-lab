"""push 전 확인: 원격(origin/main)보다 앞선 커밋에서 바뀐 파일을 '앱 코드 / 테스트 / 문서·발표 / 기타'로 나눠
앱 전체 테스트가 필요한지 알려 줌.  사용: .venv\\Scripts\\python.exe push_check.py [기준, 기본 origin/main]

규칙: 루트 *.py(앱·분석 모듈)·data/*_compare.csv·experiments_meta.json·requirements가 바뀌면 → 전체 테스트 필요.
     마지막 전체 테스트(data/last_test.txt)가 그 앱 변경 이후 커밋에서 통과했으면 → 다시 안 해도 됨.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAST = ROOT / "data" / "last_test.txt"
TOOLS = {"push_check.py"}                     # 앱이 import하지 않는 보조 도구 — 바뀌어도 앱 테스트 불필요


def git(*a):
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, encoding="utf-8").stdout.strip()


def kind(f: str) -> str:
    p = Path(f)
    if p.parts[0] == "tests":
        return "테스트"
    if f in TOOLS:
        return "도구"
    if p.parts[0] in ("docs", "versions") or p.suffix in (".md", ".txt") or p.name == ".gitignore":
        return "문서·발표"
    if (len(p.parts) == 1 and p.suffix in (".py", ".bat", ".toml")) or p.name.startswith("requirements") \
            or (p.parts[0] == "data" and (p.name.endswith("_compare.csv") or p.name == "experiments_meta.json")):
        return "앱 코드"
    return "기타"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    base = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    git("fetch", "-q", "origin")
    commits = git("rev-list", "--reverse", f"{base}..HEAD").split()
    if not commits:
        print(f"{base}보다 앞선 커밋 없음 → push할 것 없음"); return
    files = [f for f in git("diff", "--name-only", f"{base}..HEAD").splitlines() if f]
    groups = {}
    for f in files:
        groups.setdefault(kind(f), []).append(f)
    print(f"push 대상: 커밋 {len(commits)}개 ({git('log', '-1', '--format=%h', base)} → {git('log', '-1', '--format=%h')}), 바뀐 파일 {len(files)}개")
    for k in ("앱 코드", "테스트", "문서·발표", "도구", "기타"):
        if k in groups:
            print(f"\n[{k}] {len(groups[k])}개")
            for f in groups[k]:
                print("  ", f)
    tags = [t for t in git("tag", "--contains", commits[0]).split() if not git("ls-remote", "--tags", "origin", t)]
    if tags:
        print("\n새로 올라갈 태그:", " ".join(tags))

    app = groups.get("앱 코드", []) + groups.get("테스트", [])
    print("\n판정:", end=" ")
    if not app:
        print("문서·발표 스크립트·도구만 변경 → 앱 전체 테스트 생략 가능")
        return
    # 앱 코드가 마지막으로 바뀐 커밋 이후(또는 그 커밋)에서 전체 테스트가 통과했나
    last_app = git("log", "-1", "--format=%H", f"{base}..HEAD", "--", *app)
    line = LAST.read_text(encoding="utf-8").strip() if LAST.exists() else ""
    ok = False
    if "APP TEST PASSED" in line and "커밋 " in line:
        tested = line.split("커밋 ")[1].split()[0]
        ok = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", last_app, tested]).returncode == 0
    if ok:
        print(f"앱 코드 변경 있음, 그러나 그 이후 커밋에서 전체 테스트 통과 → 다시 안 해도 됨\n  ({line})")
    else:
        print("앱 코드 변경 있음 → push 전 전체 테스트 필요\n  "
              "PYTHONIOENCODING=utf-8 .venv\\Scripts\\python.exe tests\\test_app.py" + (f"\n  (마지막 기록: {line})" if line else ""))


if __name__ == "__main__":
    main()
