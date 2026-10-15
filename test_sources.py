"""소스별 연결 테스트: 어떤 티커/시리즈가 실제로 살아 있는지 확인."""
import io, sys, warnings, traceback
import pandas as pd, requests
warnings.filterwarnings("ignore")
UA = {"User-Agent": "Mozilla/5.0"}
res = []

def rec(src, name, ok, last="", n=0, note=""):
    res.append(dict(source=src, name=name, ok=ok, last=str(last)[:10], rows=n, note=note[:90]))

# 1) Yahoo
import yfinance as yf
Y = {
 "WTI":"CL=F","Brent":"BZ=F","NatGas":"NG=F","HeatingOil":"HO=F","Gasoline":"RB=F",
 "Gold":"GC=F","Silver":"SI=F","Platinum":"PL=F","Palladium":"PA=F","Aluminum":"ALI=F","Copper":"HG=F",
 "Corn":"ZC=F","Soybean":"ZS=F","SoyMeal":"ZM=F","SoyOil":"ZL=F","Wheat_CBOT":"ZW=F","Wheat_KC":"KE=F",
 "Oats":"ZO=F","Rice":"ZR=F","Wheat_MGEX":"MWE=F","Coffee":"KC=F","Sugar":"SB=F","Cocoa":"CC=F","Cotton":"CT=F",
 "OJ":"OJ=F","LiveCattle":"LE=F","FeederCattle":"GF=F","LeanHogs":"HE=F","Lumber":"LBR=F",
 "IronOre_SGX":"TIO=F","Coal_Newc":"MTF=F","BDRY_ETF":"BDRY","DBA":"DBA","DBB":"DBB",
 "Ethanol":"EH=F","Rapeseed":"RS=F","PalmOil":"FCPO=F","Uranium_ETF":"URA","Rotterdam_Coal":"LQ=F",
}
for k, t in Y.items():
    try:
        d = yf.Ticker(t).history(period="1mo", auto_adjust=False)
        if len(d): rec("yahoo", f"{k} ({t})", True, d.index[-1], len(d))
        else: rec("yahoo", f"{k} ({t})", False, note="empty")
    except Exception as e:
        rec("yahoo", f"{k} ({t})", False, note=repr(e))

# 2) FRED (키 불필요 fredgraph.csv)
F = {"WTI_daily":"DCOILWTICO","Brent_daily":"DCOILBRENTEU","HenryHub_daily":"DHHNGSP",
     "Copper_IMF_m":"PCOPPUSDM","Wheat_IMF_m":"PWHEAMTUSDM","Maize_IMF_m":"PMAIZMTUSDM",
     "Soybean_IMF_m":"PSOYBUSDM","SoyMeal_IMF_m":"PSMEAUSDM","Coal_AU_IMF_m":"PCOALAUUSDM",
     "IronOre_IMF_m":"PIORECRUSDM","Aluminum_IMF_m":"PALUMUSDM","GlobalCommodityIdx":"PALLFNFINDEXM",
     "USDKRW":"DEXKOUS","DXY_broad":"DTWEXBGS"}
for k, s in F.items():
    try:
        r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}", headers=UA, timeout=30)
        r.raise_for_status()
        d = pd.read_csv(io.StringIO(r.text)); d = d[pd.to_numeric(d.iloc[:,1], errors="coerce").notna()]
        rec("fred", f"{k} ({s})", True, d.iloc[-1,0], len(d))
    except Exception as e:
        rec("fred", f"{k} ({s})", False, note=repr(e))

# 3) World Bank Pink Sheet (monthly xlsx)
try:
    page = requests.get("https://www.worldbank.org/en/research/commodity-markets", headers=UA, timeout=30).text
    import re
    m = re.findall(r'https://thedocs\.worldbank\.org/[^"\']*CMO-Historical-Data-Monthly\.xlsx', page)
    rec("worldbank", "PinkSheet monthly xlsx link", bool(m), note=m[0] if m else "link not found on page")
except Exception as e:
    rec("worldbank", "PinkSheet", False, note=repr(e))

# 4) IMF PCPS (DBnomics 미러 / IMF 직접)
try:
    r = requests.get("https://api.db.nomics.world/v22/series/IMF/PCPS/M.W00.PCOPP.USD?observations=1", headers=UA, timeout=30)
    r.raise_for_status()
    s = r.json()["series"]["docs"][0]
    rec("imf(dbnomics)", "Copper PCOPP monthly", True, s["period"][-1], len(s["period"]))
except Exception as e:
    rec("imf(dbnomics)", "Copper PCOPP", False, note=repr(e))

# 5) akshare shipping / China futures
import akshare as ak
for fn in ["macro_shipping_bdi","macro_shipping_bci","macro_shipping_bpi","macro_shipping_bcti"]:
    try:
        d = getattr(ak, fn)()
        dc = [c for c in d.columns if "日期" in str(c) or "date" in str(c).lower()]
        rec("akshare", fn, True, d[dc[0]].iloc[-1] if dc else d.iloc[-1,0], len(d), note=str(list(d.columns)))
    except Exception as e:
        rec("akshare", fn, False, note=repr(e))
for sym, nm in [("I0","DCE iron ore"),("M0","DCE soymeal"),("C0","DCE corn"),("A0","DCE soybean#1"),("JM0","DCE coking coal"),("ZC0","ZCE thermal coal")]:
    try:
        d = ak.futures_zh_daily_sina(symbol=sym)
        rec("akshare", f"{nm} ({sym})", True, d["date"].iloc[-1], len(d))
    except Exception as e:
        rec("akshare", f"{nm} ({sym})", False, note=repr(e))

# 6) CFTC COT (socrata, 키 불필요)
try:
    r = requests.get("https://publicreporting.cftc.gov/resource/72hh-3qpy.json?$limit=1&$order=report_date_as_yyyy_mm_dd DESC", headers=UA, timeout=30)
    r.raise_for_status()
    rec("cftc", "disaggregated futures-only", True, r.json()[0]["report_date_as_yyyy_mm_dd"], 1)
except Exception as e:
    rec("cftc", "COT", False, note=repr(e))

# 7) USDA (키 불필요 엔드포인트 확인)
for nm, url in [("AMS MyMarketNews", "https://mymarketnews.ams.usda.gov/public_data"),
                ("FAS ESR api", "https://apps.fas.usda.gov/OpenData/api/esr/commodities"),
                ("NASS quickstats(no key)", "https://quickstats.nass.usda.gov/api/api_GET/?key=TEST&commodity_desc=CORN&year=2025&format=JSON")]:
    try:
        r = requests.get(url, headers=UA, timeout=30)
        rec("usda", nm, r.status_code < 400, note=f"HTTP {r.status_code} {r.text[:60]!r}")
    except Exception as e:
        rec("usda", nm, False, note=repr(e))

df = pd.DataFrame(res)
pd.set_option("display.width", 250, "display.max_colwidth", 95, "display.max_rows", 300)
print(df.to_string(index=False))
df.to_csv("source_test_result.csv", index=False, encoding="utf-8-sig")
