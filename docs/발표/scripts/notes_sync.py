"""발표 대본 양방향 반영 — 원본은 notes_v4.js([초, 대본] 30개).

  ① HTML 발표자 보기(S)에서 고친 대본: 창 위 '내보내기' → notes_edits_v4.json(내려받기 폴더) → 이 스크립트로 원본에 반영
  ② 발표대본_v4.md를 직접 고친 경우: '## N. 제목  ·  20초 (누적 …)' 형식은 그대로 두고 본문 · 초만 고친 뒤 이 스크립트로 반영
반영 후 --build를 주면 PPT 2판 · HTML 2판 · 발표대본_v4.md를 다시 만듦 (HTML에 저장돼 있던 수정본은 원본과 같아지면 자동으로 정리됨).

사용 (프로젝트 루트):
  .venv\\Scripts\\python.exe docs\\발표\\scripts\\notes_sync.py --json "%USERPROFILE%\\Downloads\\notes_edits_v4.json" --build
  .venv\\Scripts\\python.exe docs\\발표\\scripts\\notes_sync.py --md --build
  .venv\\Scripts\\python.exe docs\\발표\\scripts\\notes_sync.py --build       (원본 notes_v4.js 기준으로 전부 다시 만들기)
  (--build 없이 실행하면 원본만 고치고 바뀐 장을 보여 줌. MD가 원본보다 오래되면 --md는 중단 — --force로 강제)
"""
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NOTES = HERE / "notes_v4.js"
MD = HERE.parent / "발표대본_v4.md"
LINE = re.compile(r'^  \[(\d+), (".*")\],?\s*$')


def read_notes():
    lines = NOTES.read_text(encoding="utf-8").split("\n")
    idx = [i for i, l in enumerate(lines) if LINE.match(l)]
    return lines, idx, [(int(LINE.match(lines[i]).group(1)), json.loads(LINE.match(lines[i]).group(2))) for i in idx]


def write_notes(lines, idx, S):
    for i, (sec, txt) in zip(idx, S):
        lines[i] = f"  [{sec}, {json.dumps(txt, ensure_ascii=False)}],"
    NOTES.write_text("\n".join(lines), encoding="utf-8")


def from_json(path: Path, S):
    E = json.loads(path.read_text(encoding="utf-8"))["edits"]
    return {int(k) - 1: (S[int(k) - 1][0], v.strip()) for k, v in E.items() if 0 < int(k) <= len(S)}


def from_md(S):
    t = MD.read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r"^## (\d+)\. [^\n]*?(?:·\s*(\d+)초[^\n]*|·\s*질의응답용)\s*\n(.*?)(?=^## \d+\. |\Z)", t, re.S | re.M):
        k = int(m.group(1)) - 1
        if 0 <= k < len(S):
            out[k] = (int(m.group(2)) if m.group(2) else S[k][0], m.group(3).strip())
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    a = sys.argv[1:]
    lines, idx, S = read_notes()
    if "--json" in a:
        new = from_json(Path(a[a.index("--json") + 1]), S)
    elif "--md" in a:
        if NOTES.stat().st_mtime > MD.stat().st_mtime + 1 and "--force" not in a:    # MD가 원본보다 오래됨 → 가져오면 원본 수정분이 지워짐
            print("중단: notes_v4.js가 발표대본_v4.md보다 나중에 고쳐졌습니다. MD를 가져오면 원본의 최근 수정이 지워집니다. "
                  "원본 기준으로 MD를 다시 만들려면: notes_sync.py --build   ·   그래도 MD로 덮으려면: --md --force")
            return
        new = from_md(S)
    elif "--build" in a:
        new = {}
    else:
        print(__doc__); return
    changed = [k for k, v in new.items() if v != S[k]]
    for k in changed:
        print(f"  {k + 1}장: {S[k][0]}초 → {new[k][0]}초 · {new[k][1][:60]}…")
        S[k] = new[k]
    if not changed:
        print("바뀐 장 없음")
    else:
        write_notes(lines, idx, S)
        print(f"원본 반영: {NOTES.name} ({len(changed)}개 장) · 1–20장 합계 {sum(s for s, _ in S[:20])}초")
    if "--build" in a:
        py = sys.executable
        for args in (["build.py", "v4", str(HERE.parent / "원자재_퀀트_MI_발표_v4.pptx")], ["build.py", "v4", "--glass", str(HERE.parent / "원자재_퀀트_MI_발표_v4_유리.pptx")],
                     ["html_v4.py"], ["html_v4.py", "--glass"]):
            print("다시 만들기:", " ".join(args[:3]), flush=True)
            subprocess.run([py, str(HERE / args[0]), *args[1:]], cwd=ROOT, check=True, capture_output=True)
        print("완료 — PPT 2판 · HTML 2판 · 발표대본_v4.md")


if __name__ == "__main__":
    main()
