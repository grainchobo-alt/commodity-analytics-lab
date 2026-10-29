"""발표 PPT 한 번에 다시 만들기: 차트 PNG → data.json → pptx.

사용 (프로젝트 루트에서):
    .venv\\Scripts\\python.exe docs\\발표\\scripts\\build.py [출력.pptx]
출력을 생략하면 docs\\발표\\원자재_퀀트_MI_발표_v{다음 번호}.pptx (기존 파일은 덮어쓰지 않음).
"""
import json
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


def next_out() -> Path:
    vs = [int(m.group(1)) for p in HERE.parent.glob("원자재_퀀트_MI_발표_v*.pptx") if (m := re.search(r"_v(\d+)\.pptx$", p.name))]
    return HERE.parent / f"원자재_퀀트_MI_발표_v{max(vs, default=0) + 1}.pptx"


def build_data():
    nums = json.loads((HERE / "nums.json").read_text(encoding="utf-8"))
    vt = pd.read_csv(ROOT / "data" / "voltgt_compare.csv")
    vt = vt[vt["전략"].str.startswith(VT_STRAT) & (vt["레버리지 상한"] == "4배")].sort_values("목표 변동성 (%)")
    if vt.empty:
        sys.exit(f"voltgt_compare.csv에 '{VT_STRAT}' · 4배 행이 없음 — ⑤ 목표변동성 표를 먼저 갱신")
    nums["voltgt"] = {"tgt": vt["목표 변동성 (%)"].astype(int).tolist(),
                      **{k: vt[c].round(1).tolist() for k, c in [("arith", "연수익 산술 (%)"), ("cagr", "연수익 복리 (%)"),
                                                                  ("mdd", "최대 낙폭 (%)"), ("realvol", "실현 변동성 (%)")]}}
    (HERE / "data.json").write_text(json.dumps(nums, ensure_ascii=False, indent=1), encoding="utf-8")
    print("data.json:", ", ".join(nums))


def main():
    sys.stdout.reconfigure(errors="replace")      # Windows 콘솔(cp949)에서 특수문자 출력 오류 방지
    out = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else next_out()
    py = sys.executable
    step("1/3 차트 PNG (실데이터·개념 예시)")
    subprocess.run([py, str(HERE / "charts.py")], cwd=ROOT, check=True)
    subprocess.run([py, str(HERE / "charts3d.py")], cwd=ROOT, check=True)
    step("2/3 data.json (nums.json + 목표 변동성 표)")
    build_data()
    step("3/3 슬라이드 조립 (pptxgenjs)")
    node = shutil.which("node")
    if not node:
        sys.exit("node가 없음 — Node.js 설치 후 다시 실행")
    if not (HERE / "node_modules" / "pptxgenjs").exists():
        sys.exit(f"pptxgenjs 미설치 — 한 번만: cd \"{HERE}\" && npm install")
    subprocess.run([node, "deck.js", str(out)], cwd=HERE, check=True)
    print(f"\n완료: {out}")
    print("※ 발표 전 확인: 29번 장(횡단면 군집 품목)·30번 장(★ 이번 주 판단 카드)은 deck.js에 직접 적은 값 -> 앱 화면 기준으로 갱신")


if __name__ == "__main__":
    main()
