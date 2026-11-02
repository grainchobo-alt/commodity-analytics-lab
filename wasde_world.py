"""WASDE 원문(보유 txt · xls)에서 세계 수급표(옥수수 · 대두 · 밀)의 국가별 값을 발표 시점 그대로 뽑음 — 기존 파일은 고치지 않음.

표 한 줄 = 지역 + 숫자 7개 [기초재고, 생산, 수입, 사료(대두는 압착), 국내 총소비, 수출, 기말재고] (단위 백만 톤).
신년도 전망 칸의 '지난달/이번 달' 두 줄은 이번 달 줄만 씀. 같은 연산연도의 직전 발표 대비 차이 = 그달의 '수정'.
결과: data/reports/wasde_world.parquet (long: release · commodity · region · my · 항목 · value)
실행: .venv\\Scripts\\python.exe wasde_world.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import reports as R

RAW = Path(__file__).resolve().parent / "data" / "reports" / "raw"
OUT = Path(__file__).resolve().parent / "data" / "reports" / "wasde_world.parquet"
TABLES = {"corn": "World Corn Supply and Use", "soybean": "World Soybean Supply and Use", "wheat": "World Wheat Supply and Use"}
COLS = ["beg", "production", "imports", "feed_crush", "domestic", "exports", "ending"]
REGION = {"world": "World", "unitedstates": "United States", "totalforeign": "Total Foreign", "argentina": "Argentina", "brazil": "Brazil",
          "russia": "Russia", "ukraine": "Ukraine", "china": "China", "canada": "Canada", "australia": "Australia", "kazakhstan": "Kazakhstan",
          "eu": "EU", "european": "EU", "paraguay": "Paraguay"}
MONTHS = {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"}
NUM = re.compile(r"-?\d[\d,]*\.?\d*|-?\.\d+")


def _region(label: str) -> str | None:
    k = re.sub(r"[^a-z]", "", re.sub(r"\d+/", "", label.lower()))
    if k.startswith("worldless"):
        return None
    for p, nm in REGION.items():
        if k.startswith(p) and (p != "world" or k == "world"):
            return nm
    return None


def _block(text: str, title: str) -> str:
    i = text.find(title)
    if i < 0:
        return ""
    nxt = [m.start() for m in re.finditer(r"World [A-Za-z ,]+?Supply and Use", text[i + 10:]) if title not in text[i + 10 + m.start():i + 10 + m.start() + len(title) + 2]]
    return text[i: i + 10 + nxt[0]] if nxt else text[i:]


def parse_table(text: str, title: str) -> list[dict]:
    rows, year, pending = [], None, None
    for raw in _block(text, title).splitlines():
        line = raw.replace(":", " ").rstrip()
        ym = re.search(r"(\d{4})/(\d{2,3})", line)
        if ym and not re.search(r"\d+\.\d+.*\d+\.\d+", line):
            year = f"{ym.group(1)}/{(int(ym.group(1)) + 1) % 100:02d}"
            pending = None
            continue
        if year is None:
            continue
        line = re.sub(r"\s\d{1,2}/(?=\s|$)", " ", line)                                     # 각주 표시 'World 3/' → 'World'
        m = re.match(r"^\s*([A-Za-z][A-Za-z .,'()\-]*?(?:\s+\d+/)?)\s{2,}(.*)$", line) or re.match(r"^\s*([A-Za-z][A-Za-z .,'()\-]*\d*/?)\s*$", line)
        if not m:
            continue
        lab = m.group(1).strip()
        vals = [float(v.replace(",", "")) for v in NUM.findall(m.group(2))] if m.lastindex and m.lastindex >= 2 else []
        if lab.lower()[:3] in MONTHS and len(lab) <= 4:                                  # 'Sep  …' / 'Oct  …' (전망 칸)
            if pending and len(vals) == 7:
                rows.append(dict(region=pending, my=year, **dict(zip(COLS, vals)), _m=lab.lower()[:3]))
            continue
        reg = _region(lab)
        if reg is None:
            pending = None
            continue
        if len(vals) == 7:
            rows.append(dict(region=reg, my=year, **dict(zip(COLS, vals)), _m=""))
            pending = None
        elif not vals:
            pending = reg                                                                  # 다음 줄들이 '지난달/이번 달'
    if not rows:
        return []
    D = pd.DataFrame(rows)
    D = D.groupby(["region", "my"]).tail(1)                                                # 전망 칸은 마지막 줄 = 이번 달
    return D.drop(columns="_m").to_dict("records")


def collect() -> pd.DataFrame:
    files = {}
    for f in sorted(RAW.glob("wasde_*.*")):
        m = re.match(r"wasde_(\d{8})\.(txt|xls)$", f.name)
        if m and (m.group(1) not in files or m.group(2) == "txt"):
            files[m.group(1)] = f
    out = []
    for d, f in files.items():
        raw = f.read_bytes()
        try:
            text = R._normalize(raw.decode("latin-1")) if f.suffix == ".txt" else R._xls_to_text(raw)
        except Exception as e:
            print(f"  ! {d} 읽기 실패: {e}")
            continue
        for com, title in TABLES.items():
            for r in parse_table(text, title):
                out.append(dict(release=pd.Timestamp(d), commodity=com, **r))
    D = pd.DataFrame(out)
    L = D.melt(id_vars=["release", "commodity", "region", "my"], value_vars=COLS, var_name="item", value_name="value")
    L.to_parquet(OUT)
    return L


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    L = collect()
    print(f"발간본 {L.release.nunique()}개 · {L.release.min():%Y-%m} ~ {L.release.max():%Y-%m} · 행 {len(L)}")
    print(L.groupby(["commodity", "region"]).release.nunique().unstack(0).to_string())
