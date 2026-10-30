"""새 PC(회사 노트북 등)로 옮긴 뒤 구동 준비 점검 — 파이썬 · 패키지 · 데이터 · (선택) 인터넷 원천 접속.

실행:  .venv\\Scripts\\python.exe setup_check.py          (기본 점검)
       .venv\\Scripts\\python.exe setup_check.py --net    (+ Yahoo·FRED·PyPI·Hugging Face 접속 확인)
결과 마지막 줄의 '필수 실패 0개'면 대시보드를 열 수 있음. 선택 항목 실패는 해당 기능만 안 됨.
"""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.stdout.reconfigure(errors="replace")
fails, warns = [], []


def check(ok: bool, label: str, hint: str = "", required: bool = True):
    print(f"  {'OK ' if ok else ('실패' if required else '주의')}  {label}" + ("" if ok or not hint else f"  → {hint}"))
    if not ok:
        (fails if required else warns).append(label)


print("① 파이썬")
check(sys.version_info[:2] == (3, 12), f"Python {sys.version.split()[0]} (권장 3.12)", "python.org에서 3.12 설치 후 setup.bat 다시 실행", required=sys.version_info[:2] >= (3, 11))
check(Path(sys.executable).parent.parent.name == ".venv", f"가상환경 사용: {sys.executable}", ".venv\\Scripts\\python.exe 로 실행", required=False)

print("② 패키지")
REQ = ["pandas", "numpy", "pyarrow", "plotly", "streamlit", "statsmodels", "sklearn", "scipy", "arch", "yfinance", "openpyxl", "xlsxwriter"]
OPT = {"lightgbm": "나무 모델", "hmmlearn": "HMM 레짐", "linearmodels": "패널 계량", "torch": "LSTM·GNN·대안 시계열", "chronos": "Chronos 파운데이션 모델",
       "timesfm": "TimesFM", "prophet": "Prophet", "neuralforecast": "PatchTST·TFT 등", "torch_geometric": "GNN", "akshare": "중국 선물 수집", "pypdf": "WASDE·STEO PDF 읽기"}
for m in REQ:
    try:
        importlib.import_module(m); check(True, m)
    except Exception as e:
        check(False, m, f"setup.bat 다시 실행 ({type(e).__name__})")
for m, what in OPT.items():
    try:
        importlib.import_module(m); check(True, f"{m} ({what})")
    except Exception:
        check(False, f"{m} ({what})", "이 기능만 안 됨 — 나머지는 정상", required=False)

print("③ 데이터 (GitHub에서 함께 받아짐)")
D = ROOT / "data"
for f, what in [("daily.parquet", "원가격"), ("daily_adj.parquet", "롤 보정 가격"), ("daily_clean.parquet", "정제 가격"), ("cot.parquet", "CFTC 포지션"),
                ("experiments_meta.json", "실험 결과 목록"), ("cleaning_rules.csv", "정제 규칙")]:
    p = D / f
    check(p.exists(), f"data/{f} ({what})", "GitHub에서 다시 받기 (git pull) — data 폴더가 빠졌을 수 있음")
n_cmp = len(list(D.glob("*_compare.csv")))
check(n_cmp >= 30, f"실험 결과 표 {n_cmp}개 (_compare.csv)", "git pull", required=False)
check((D / "fm").exists(), "파운데이션 모델 미리 계산 결과 (data/fm)", "없으면 ⑥ 탭이 새로 계산(가중치 다운로드 필요)", required=False)
try:
    import pandas as pd
    last = pd.read_parquet(D / "daily_adj.parquet", columns=["date"])["date"].max()
    print(f"     자료 마지막 날짜: {pd.Timestamp(last):%Y-%m-%d} (새로 받으려면 run_daily.bat — 인터넷 필요)")
except Exception as e:
    check(False, "가격 자료 읽기", str(e)[:80])
check((ROOT / ".env").exists(), ".env (API 키 — 선택, GitHub에 없음)", "필요하면 집 PC의 .env를 따로 옮기거나 .env.example 복사 후 키 입력. 없어도 대시보드는 열림", required=False)

if "--net" in sys.argv:
    print("④ 인터넷 원천 접속 (회사망 차단 여부)")
    import requests
    for name, url, what in [("Yahoo Finance", "https://query1.finance.yahoo.com/v8/finance/chart/CL=F?range=5d&interval=1d", "가격 갱신·옵션 랩"),
                            ("FRED", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10", "거시 자료"),
                            ("PyPI", "https://pypi.org/simple/pip/", "패키지 설치"),
                            ("Hugging Face", "https://huggingface.co/api/models/amazon/chronos-bolt-small", "파운데이션 모델·FinBERT 가중치")]:
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"}); ok = r.status_code < 400
            check(ok, f"{name} ({what}) — HTTP {r.status_code}", "회사망 차단 가능 — 기존 자료로는 계속 구동됨", required=False)
        except Exception as e:
            check(False, f"{name} ({what})", f"접속 실패 {type(e).__name__} — 회사 프록시·방화벽 확인", required=False)

print(f"\n필수 실패 {len(fails)}개 · 선택 주의 {len(warns)}개" + (f"  ({', '.join(fails)})" if fails else ""))
sys.exit(1 if fails else 0)
