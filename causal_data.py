"""곡물 온톨로지용 추가 자료 수집 (키 불필요) → data/mm/*.parquet

- NOAA STAR 위성 식생건강지수(VHI, 경작지만, 주간 1981–): 미국 콘벨트·남미·흑해 주산지 주(州)별. SMN = 평활 NDVI.
- USDA 주간 수출 판매 (AgTransport `wnn7-29tu`, 1999–): 옥수수·대두·밀 주별 순판매·선적 (서버에서 주·품목 합계).
- USDA 해상 운임 (AgTransport `ehs5-yac3`, 월간 1996–): 걸프·PNW → 일본 ($/톤).
- USDA FAS PSD 세계 수급 (연간, 현재 추정치만 — 과거 발표 시점 값 없음 → 주간 모형에는 미사용, 온톨로지 표·참고용).

실행: .venv\\Scripts\\python.exe causal_data.py [vhi|sales|freight|psd ...]  (생략하면 전부)
"""
import io
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

MM = Path(__file__).resolve().parent / "data" / "mm"
AGT = "https://agtransport.usda.gov/resource"
VHI_URL = "https://www.star.nesdis.noaa.gov/smcd/emb/vci/VH/get_TS_admin.php"
VHI_REG = {  # 묶음: [(국가, 주 번호, 이름)] — 번호는 NOAA STAR 지역 번호(2026-11-02 확인)
    "미국 위성": [("USA", 16, "Iowa"), ("USA", 14, "Illinois"), ("USA", 28, "Nebraska")],
    "남미 위성": [("BRA", 11, "Mato Grosso"), ("BRA", 16, "Paraná"), ("BRA", 21, "Rio Grande do Sul"),
               ("ARG", 1, "Buenos Aires"), ("ARG", 6, "Córdoba"), ("ARG", 21, "Santa Fe")],
    "흑해 위성": [("RUS", 62, "Rostov"), ("RUS", 39, "Krasnodar"), ("RUS", 69, "Stavropol'"), ("RUS", 83, "Volgograd"),
               ("UKR", 9, "Kherson"), ("UKR", 17, "Odessa"), ("UKR", 26, "Zaporizhzhya"), ("UKR", 5, "Dnipropetrovs'k"), ("UKR", 8, "Kharkiv")],
    "미국 밀 위성": [("USA", 17, "Kansas"), ("USA", 35, "North Dakota")],
}
PSD = {"grains": "https://apps.fas.usda.gov/psdonline/downloads/psd_grains_pulses_csv.zip",
       "oilseeds": "https://apps.fas.usda.gov/psdonline/downloads/psd_oilseeds_csv.zip"}
PSD_KEEP = {"Corn", "Wheat", "Oilseed, Soybean", "Meal, Soybean", "Oil, Soybean"}
PSD_ATTR = {"Production", "Exports", "Imports", "Ending Stocks", "Domestic Consumption", "Feed Dom. Consumption", "Crush"}


def vhi_fetch(y1=1981, y2=2026) -> pd.DataFrame:
    rows = []
    for grp, regs in VHI_REG.items():
        for c, pid, nm in regs:
            txt = requests.get(VHI_URL, params=dict(country=c, provinceID=pid, year1=y1, year2=y2, type="Mean"), timeout=60).text
            for ln in txt.replace("<br>", "\n").splitlines():
                p = [x.strip() for x in ln.replace("<tt><pre>", "").split(",")]
                if len(p) >= 7 and p[0].isdigit():
                    rows.append(dict(group=grp, country=c, region=nm, year=int(p[0]), week=int(p[1]),
                                     smn=float(p[2]), smt=float(p[3]), vci=float(p[4]), tci=float(p[5]), vhi=float(p[6])))
            print(f"  VHI {grp} · {nm}: {sum(r['region'] == nm for r in rows)}주", flush=True)
            time.sleep(0.5)
    D = pd.DataFrame(rows)
    D = D[(D.vhi > 0) & (D.vhi <= 100)]                                                    # 결측 = −1
    D.to_parquet(MM / "vhi.parquet")
    return D


def export_sales_fetch() -> pd.DataFrame:
    q = {"$select": "date,commodity,sum(net_sales_total) as net_sales,sum(wkexportscmy) as exports,sum(outstanding_sales_total) as outstanding",
         "$group": "date,commodity", "$order": "date", "$limit": 50000}
    D = pd.DataFrame(requests.get(f"{AGT}/wnn7-29tu.json", params=q, timeout=120).json())
    D["date"] = pd.to_datetime(D.date)
    for c in ("net_sales", "exports", "outstanding"):
        D[c] = pd.to_numeric(D[c], errors="coerce")
    D.to_parquet(MM / "export_sales.parquet")
    print(f"  수출 판매: {len(D)}행 {D.date.min():%Y-%m-%d} ~ {D.date.max():%Y-%m-%d}")
    return D


def freight_fetch() -> pd.DataFrame:
    D = pd.DataFrame(requests.get(f"{AGT}/ehs5-yac3.json", params={"$limit": 5000, "$order": "date"}, timeout=60).json())
    D["date"] = pd.to_datetime(D.date)
    for c in ("gulf_to_japan", "pnw_to_japan", "gulf_pnw_spread"):
        D[c] = pd.to_numeric(D[c], errors="coerce")
    D = D[["date", "gulf_to_japan", "pnw_to_japan", "gulf_pnw_spread"]]
    D.to_parquet(MM / "freight.parquet")
    print(f"  운임: {len(D)}개월 {D.date.min():%Y-%m} ~ {D.date.max():%Y-%m}")
    return D


def psd_fetch() -> pd.DataFrame:
    out = []
    for nm, url in PSD.items():
        z = zipfile.ZipFile(io.BytesIO(requests.get(url, timeout=120).content))
        D = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
        out.append(D[D.Commodity_Description.isin(PSD_KEEP) & D.Attribute_Description.isin(PSD_ATTR)])
    D = pd.concat(out)[["Commodity_Description", "Country_Name", "Market_Year", "Attribute_Description", "Unit_Description", "Value", "Calendar_Year", "Month"]]
    D.columns = ["commodity", "country", "my", "attr", "unit", "value", "cal_year", "month"]
    D.to_parquet(MM / "psd.parquet")
    print(f"  PSD: {len(D)}행, 국가 {D.country.nunique()}개, {D.my.min()}–{D.my.max()}")
    return D


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    todo = sys.argv[1:] or ["vhi", "sales", "freight", "psd"]
    for k in todo:
        print(k, flush=True)
        {"vhi": vhi_fetch, "sales": export_sales_fetch, "freight": freight_fetch, "psd": psd_fetch}[k]()
