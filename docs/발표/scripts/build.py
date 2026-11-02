"""발표 PPT 한 번에 다시 만들기: 차트 PNG → data.json → pptx.

사용 (프로젝트 루트에서):
    .venv\\Scripts\\python.exe docs\\발표\\scripts\\build.py [v1|v2|v3|v4 ...] [--glass] [출력.pptx]
--glass(v4만) = 같은 내용에 유리 + 에디토리얼 디자인을 입힌 판(deck_v4_glass.js) → 원자재_퀀트_MI_발표_v4_유리.pptx
버전 = deck_v{N}.js·notes_v{N}.js 묶음(생략하면 가장 최신). 출력을 생략하면 docs\\발표\\원자재_퀀트_MI_발표_v{N}.pptx,
같은 이름이 이미 있으면 _YYYYMMDD-HHMM을 붙여 새로 저장(기존 파일은 덮어쓰지 않음).
"""
import json
from datetime import datetime
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VT_STRAT = "혼합 · MinVar 50 + 모멘텀 25 + 가치 25"      # 38번 장 목표 변동성 표 = 가치 혼합 · 배율 상한 4배


def step(msg):
    print(f"\n=== {msg} ===", flush=True)


def versions() -> list[int]:
    return sorted(int(m.group(1)) for p in HERE.glob("deck_v*.js") if (m := re.fullmatch(r"deck_v(\d+)\.js", p.name)))


def default_out(v: int, suf: str = "") -> Path:
    out = HERE.parent / f"원자재_퀀트_MI_발표_v{v}{suf}.pptx"
    return out if not out.exists() else out.with_name(f"{out.stem}_{datetime.now():%Y%m%d-%H%M}.pptx")


def glass_bg():
    """유리판 배경: 은은한 색 덩어리(녹·금·청) 그라데이션 PNG — 밝은 장용 + 표지(짙은)용. 발표 HTML 유리 테마와 같은 색."""
    import numpy as np
    from PIL import Image
    w, h = 1920, 1080
    y, x = np.mgrid[0:h, 0:w] / np.array([h, w]).reshape(2, 1, 1)

    def blob(cx, cy, rx, ry):
        return np.clip(1 - np.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2) / 0.7, 0, 1) ** 1.6
    for name, a, b, blobs in [
            ("glass_bg", (243, 246, 241), (251, 248, 239),
             [((46, 125, 82), .22, (.12, .18, .38, .46)), ((200, 162, 74), .28, (.88, .82, .42, .5)), ((47, 110, 143), .18, (.72, .08, .34, .4))]),
            ("glass_bg_dark", (16, 37, 27), (18, 40, 58),
             [((83, 160, 116), .45, (.15, .2, .4, .55)), ((200, 162, 74), .35, (.9, .85, .45, .55)), ((47, 110, 143), .45, (.75, .1, .35, .45))])]:
        t = ((x + y) / 2)[..., None]
        img = np.array(a) * (1 - t) + np.array(b) * t
        for col, alpha, (cx, cy, rx, ry) in blobs:
            m = (alpha * blob(cx, cy, rx, ry))[..., None]
            img = img * (1 - m) + np.array(col) * m
        Image.fromarray(img.clip(0, 255).astype("uint8")).save(HERE / "img" / f"{name}.png")
    print("유리 배경: img/glass_bg.png · glass_bg_dark.png")


def build_data():
    nums = json.loads((HERE / "nums.json").read_text(encoding="utf-8"))
    vt = pd.read_csv(ROOT / "data" / "voltgt_compare.csv")
    vt = vt[vt["전략"].str.startswith(VT_STRAT) & (vt["레버리지 상한"] == "4배")].sort_values("목표 변동성 (%)")
    if vt.empty:
        sys.exit(f"voltgt_compare.csv에 '{VT_STRAT}' · 4배 행이 없음 — ⑤ 목표변동성 표를 먼저 갱신")
    nums["voltgt"] = {"tgt": vt["목표 변동성 (%)"].astype(int).tolist(),
                      **{k: vt[c].round(1).tolist() for k, c in [("arith", "연수익 산술 (%)"), ("cagr", "연수익 복리 (%)"),
                                                                  ("mdd", "최대 낙폭 (%)"), ("realvol", "실현 변동성 (%)")]}}
    q16 = ROOT / "data" / "quant16_compare.csv"                      # 퀀트 전략 16개 카드 (quant16.py)
    if q16.exists():
        Q = pd.read_csv(q16, encoding="utf-8-sig")
        nums["q16"] = Q[["행", "번호", "전략", "영문", "한 줄 설명", "결과 1", "결과 2", "판단"]].to_dict("records")
    (HERE / "data.json").write_text(json.dumps(nums, ensure_ascii=False, indent=1), encoding="utf-8")
    print("data.json:", ", ".join(nums))


def script_md(v: int, node: str):
    """발표자 노트에 [초, 대본]이 있으면(v4~) 시간 배분표 + 장별 대본을 docs/발표/발표대본_v{N}.md로 저장."""
    js = f"const n=require('./notes_v{v}.js');console.log(JSON.stringify(n.timed||null))"
    timed = json.loads(subprocess.run([node, "-e", js], cwd=HERE, capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    if not timed:
        return
    deck = (HERE / f"deck_v{v}.js").read_text(encoding="utf-8")
    titles = ["표지"] + re.findall(r'header\(s, "[^"]*", "([^"]*)"', deck)
    lines = [f"# 발표 대본 v{v} — 10분", "", f"`deck_v{v}.js`·`notes_v{v}.js`에서 자동 생성 (`build.py`). PowerPoint에서는 같은 내용이 슬라이드 노트·발표자 보기에 들어 있음.", "",
             "| 장 | 제목 | 시간 | 누적 |", "|---|---|---|---|"]
    acc = 0
    for i, (sec, _) in enumerate(timed):
        acc += sec
        lines.append(f"| {i + 1} | {titles[i] if i < len(titles) else ''} | {sec}초 | {acc // 60}:{acc % 60:02d} |" if sec else f"| {i + 1} | {titles[i] if i < len(titles) else ''} | 질의응답용 | |")
    lines += ["", "---", ""]
    acc = 0
    for i, (sec, txt) in enumerate(timed):
        acc += sec
        head = f"## {i + 1}. {titles[i] if i < len(titles) else ''}"
        lines += [head + (f"  ·  {sec}초 (누적 {acc // 60}:{acc % 60:02d})" if sec else "  ·  질의응답용"), "", txt, ""]
    out = HERE.parent / f"발표대본_v{v}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("대본:", out)


def main():
    sys.stdout.reconfigure(errors="replace")      # Windows 콘솔(cp949)에서 특수문자 출력 오류 방지
    args = sys.argv[1:]
    glass = "--glass" in args
    args = [a for a in args if a != "--glass"]
    v = int(args.pop(0)[1:]) if args and re.fullmatch(r"v\d+", args[0]) else versions()[-1]
    if v not in versions():
        sys.exit(f"deck_v{v}.js 없음 — 있는 버전: {versions()}")
    if glass and v != 4:
        sys.exit("--glass는 v4만 지원")
    out = Path(args[0]).resolve() if args else default_out(v, "_유리" if glass else "")
    print(f"버전 v{v} → {out}")
    py = sys.executable
    step("1/3 차트 PNG (실데이터·개념 예시)")
    subprocess.run([py, str(HERE / "charts.py")], cwd=ROOT, check=True)
    subprocess.run([py, str(HERE / "charts3d.py")], cwd=ROOT, check=True)
    if v >= 2:
        subprocess.run([py, str(HERE / "charts_v2.py")], cwd=ROOT, check=True)
    if v >= 3:
        subprocess.run([py, str(HERE / "charts_v3.py")], cwd=ROOT, check=True)
    if v >= 4:
        subprocess.run([py, str(HERE / "charts_v4.py")], cwd=ROOT, check=True)
    step("2/3 data.json (nums.json + 목표 변동성 표)")
    build_data()
    step("3/3 슬라이드 조립 (pptxgenjs)")
    node = shutil.which("node")
    if not node:
        sys.exit("node가 없음 — Node.js 설치 후 다시 실행")
    if not (HERE / "node_modules" / "pptxgenjs").exists():
        sys.exit(f"pptxgenjs 미설치 — 한 번만: cd \"{HERE}\" && npm install")
    if glass:
        glass_bg()
    subprocess.run([node, "deck_v4_glass.js" if glass else f"deck_v{v}.js", str(out)], cwd=HERE, check=True)
    script_md(v, node)
    print(f"\n완료: {out}")
    print({1: "※ 발표 전 확인: 29번 장(횡단면 군집)·30번 장(이번 주 판단 카드)은 deck_v1.js에 직접 적은 값 -> 앱 화면 기준으로 갱신",
           2: "※ 발표 전 확인: 10번 장(이번 주 판단)·11번 장(물량 위험 금액)·2번 장 요약 숫자는 deck_v2.js에 직접 적은 값 -> 앱 화면 기준으로 갱신",
           3: "※ 발표 전 확인: 14번 장(이번 주 판단)·15번 장(물량 위험 금액)·2번 장 요약 숫자는 deck_v3.js에 직접 적은 값 -> 앱 화면 기준으로 갱신",
           4: "※ 발표 전 확인: 17번 장(이번 주 판단)·18번 장(물량 위험 금액)·2번 장 요약 숫자는 deck_v4.js에 직접 적은 값 -> 앱 화면 기준으로 갱신"}.get(v, ""))


if __name__ == "__main__":
    main()
