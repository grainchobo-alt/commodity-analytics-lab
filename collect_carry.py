"""캐리(기간구조) 수집: 근월물 F1 · 차월물 F2 → data/carry.parquet (date, id, f1, f2, months)

  미국 에너지 = EIA NYMEX 1·2번째 계약 일간 (원유 1983–, 난방유 1994–, 천연가스 1994–, 휘발유 2005–; EIA가 2024-04 이후 갱신 중단)
  중국 = Sina 개별 월물 일간 (만기 지난 계약은 약 2018년 이후만 제공) → 날짜별 '만기가 가장 가까운 두 계약'
  Yahoo는 만기 지난 개별 월물을 제공하지 않아 미국 곡물·금속의 과거 캐리는 만들 수 없음.
"""
from __future__ import annotations

import io
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
OUT = ROOT / "data" / "carry.parquet"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

EIA = {"yf:CL=F": ("pet", "RCLC1", "RCLC2"), "yf:NG=F": ("ng", "RNGC1", "RNGC2"),
       "yf:HO=F": ("pet", "EER_EPD2F_PE1_Y35NY_DPG", "EER_EPD2F_PE2_Y35NY_DPG"),
       "yf:RB=F": ("pet", "EER_EPMRR_PE1_Y35NY_DPG", "EER_EPMRR_PE2_Y35NY_DPG")}
# Sina 품목 코드 → 카탈로그 id (주력 연속물)
SINA = {"M": "ak:M0", "Y": "ak:Y0", "C": "ak:C0", "A": "ak:A0", "I": "ak:I0", "J": "ak:J0", "JM": "ak:JM0", "RB": "ak:RB0"}


def _eia(base: str, s: str) -> pd.Series:
    r = requests.get(f"https://www.eia.gov/dnav/{base}/hist_xls/{s}d.xls", headers=UA, timeout=60)
    d = pd.read_excel(io.BytesIO(r.content), sheet_name=1, skiprows=2)
    return pd.Series(pd.to_numeric(d.iloc[:, 1], errors="coerce").values, index=pd.to_datetime(d.iloc[:, 0])).dropna()


def collect_eia() -> pd.DataFrame:
    rows = []
    for i, (base, a, b) in EIA.items():
        try:
            f1, f2 = _eia(base, a), _eia(base, b)
            d = pd.DataFrame({"f1": f1, "f2": f2}).dropna().rename_axis("date").reset_index()
            d["id"] = i; d["months"] = 1.0
            rows.append(d)
            print(f"EIA {i}: {len(d)}행 {d.date.min().date()}–{d.date.max().date()}", flush=True)
        except Exception as e:
            print(f"EIA {i} 실패: {e}", flush=True)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_sina(y0: int = 2017, y1: int | None = None) -> pd.DataFrame:
    import akshare as ak
    y1 = y1 or pd.Timestamp.today().year + 1
    rows = []
    for code, cid in SINA.items():
        px = {}
        for y in range(y0, y1 + 1):
            for m in range(1, 13):
                sym = f"{code}{y % 100:02d}{m:02d}"
                try:
                    x = ak.futures_zh_daily_sina(symbol=sym)
                except Exception:
                    continue
                if x is None or not len(x):
                    continue
                px[(y, m)] = pd.Series(pd.to_numeric(x["close"], errors="coerce").values, index=pd.to_datetime(x["date"]))
                time.sleep(0.15)
        if not px:
            print(f"Sina {code}: 자료 없음", flush=True); continue
        P = pd.DataFrame({k: v for k, v in px.items()})
        out = []
        for dt, r in P.iterrows():
            live = [(k, v) for k, v in r.dropna().items() if pd.Timestamp(k[0], k[1], 1) + pd.offsets.MonthEnd(0) >= dt]
            live = [kv for kv in live if pd.Timestamp(kv[0][0], kv[0][1], 1) > dt + pd.Timedelta(days=10)]   # 만기 달 진입 직전 계약은 제외
            if len(live) < 2:
                continue
            live.sort(key=lambda kv: kv[0])
            (k1, v1), (k2, v2) = live[0], live[1]
            mo = (k2[0] - k1[0]) * 12 + (k2[1] - k1[1])
            out.append({"date": dt, "f1": v1, "f2": v2, "months": float(mo), "id": cid,
                        "c1": k1[0] * 100 + k1[1], "c2": k2[0] * 100 + k2[1]})   # 계약 식별 (월물 교체 시 수익률 계산용)
        d = pd.DataFrame(out)
        rows.append(d)
        print(f"Sina {code}: 계약 {len(px)}개 · {len(d)}행 {d.date.min().date() if len(d) else '-'}–{d.date.max().date() if len(d) else '-'}", flush=True)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    if "--eia-only" in sys.argv and OUT.exists():            # 기존 파일의 중국 자료는 유지하고 EIA만 갱신
        old = pd.read_parquet(OUT)
        parts = [old[~old.id.isin(list(EIA))], collect_eia()]
    else:
        parts = [collect_eia()]
        if "--no-sina" not in sys.argv:
            parts.append(collect_sina())
    D = pd.concat([p for p in parts if len(p)], ignore_index=True)
    D = D[(D.f1 > 0) & (D.f2 > 0)]
    D.to_parquet(OUT, index=False)
    print(f"저장: {OUT} · {len(D)}행 · 품목 {D.id.nunique()}개")


if __name__ == "__main__":
    main()
