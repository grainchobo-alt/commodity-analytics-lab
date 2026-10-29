"""모델 도감 학습 노트 한 번에 다시 만들기: 그림 PNG → content.json(+수식 PNG) → docx.

사용 (프로젝트 루트에서):
    .venv\\Scripts\\python.exe docs\\학습\\scripts\\build.py
결과: docs\\학습\\모델도감_학습노트.docx (*.docx는 git 제외). docx 패키지는 docs\\발표\\scripts\\node_modules 재사용.
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

if __name__ == "__main__":
    for s in ("figs.py", "content.py"):
        print(f"=== {s} ===", flush=True)
        subprocess.run([sys.executable, str(HERE / s)], cwd=ROOT, check=True)
    print("=== build_doc.js ===", flush=True)
    subprocess.run([shutil.which("node"), "build_doc.js"], cwd=HERE, check=True)
