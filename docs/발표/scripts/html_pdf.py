"""HTML 발표를 장마다 캡처(html_shots.js)해 한 PDF로 묶음 — HTML의 Plotly 3D 화면 그대로 담긴 제출용 PDF.
사용 (프로젝트 루트): .venv\\Scripts\\python.exe docs\\발표\\scripts\\html_pdf.py [발표.html]
기본 = 원자재_퀀트_MI_발표_v4_유리.html → 같은 이름의 _HTML캡처.pdf
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent


def main():
    html = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent / "원자재_퀀트_MI_발표_v4_유리.html"
    out = html.with_name(html.stem + "_HTML캡처.pdf")
    tmp = Path(tempfile.mkdtemp())
    try:
        subprocess.run([shutil.which("node"), "html_shots.js", str(html), str(tmp)], cwd=HERE, check=True)
        pages = [Image.open(f).convert("RGB") for f in sorted(tmp.glob("s*.png"))]
        pages[0].save(out, save_all=True, append_images=pages[1:], resolution=150, quality=90)   # 2400×1350 → 16×9인치
        print(f"{len(pages)}쪽 → {out}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
