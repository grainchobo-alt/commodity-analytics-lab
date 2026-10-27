"""데이터 수집: Yahoo / FRED / World Bank Pink Sheet / akshare(BDI, DCE) / CFTC COT -> data/*.parquet"""
import io, os, re, sys, time, warnings
from pathlib import Path
import pandas as pd, requests

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent


def _load_env(path: Path = ROOT / ".env"):
    """.env (KEY=VALUE) → 환경변수. 이미 설정된 환경변수가 우선. 형식은 .env.example 참고."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


_load_env()
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
START = pd.Timestamp("1960-01-01")   # 가능한 최대 과거 (소스별 실제 시작일은 다름)
COT_START = "2006-01-01"             # Disaggregated COT 시작
OVERLAP = pd.Timedelta(days=14)      # 증분 수집 시 겹쳐 받는 기간 (지연 수정치 반영)
FULL = "--full" in sys.argv          # --full: 전체 이력 재다운로드
LAST: dict = {}                      # id -> 저장된 마지막 날짜 (증분 수집 기준)

def fresh(key) -> bool:
    """날짜 필터가 없는 소스(akshare: 매번 전체 이력 반환)는 이미 전 영업일까지 있으면 건너뜀."""
    if FULL or key not in LAST:
        return False
    return LAST[key] >= pd.Timestamp.today().normalize() - pd.offsets.BDay(1)


def since(key, default=None):
    """증분 수집 시작일. 저장 이력이 없거나 --full 이면 default(전체)."""
    if FULL or key not in LAST:
        return default
    return LAST[key] - OVERLAP

# (key, 그룹, 이름, 단위)
YAHOO = {
 "CL=F": ("Energy", "WTI", "$/bbl"), "BZ=F": ("Energy", "Brent", "$/bbl"),
 "NG=F": ("Energy", "Henry Hub NatGas", "$/mmbtu"), "HO=F": ("Energy", "Heating Oil", "$/gal"),
 "RB=F": ("Energy", "RBOB Gasoline", "$/gal"),
 "HG=F": ("Metals", "Copper (COMEX)", "$/lb"), "ALI=F": ("Metals", "Aluminum (COMEX)", "$/mt"),
 "GC=F": ("Metals", "Gold", "$/oz"), "SI=F": ("Metals", "Silver", "$/oz"),
 "PL=F": ("Metals", "Platinum", "$/oz"), "PA=F": ("Metals", "Palladium", "$/oz"),
 "TIO=F": ("Iron Ore / Coal", "Iron Ore 62% (SGX)", "$/mt"),
 "ZC=F": ("Grains", "Corn", "c/bu"), "ZS=F": ("Grains", "Soybean", "c/bu"),
 "ZM=F": ("Grains", "Soybean Meal", "$/st"), "ZL=F": ("Grains", "Soybean Oil", "c/lb"),
 "ZW=F": ("Grains", "Wheat (CBOT SRW)", "c/bu"), "KE=F": ("Grains", "Wheat (KC HRW)", "c/bu"),
 "ZO=F": ("Grains", "Oats", "c/bu"), "ZR=F": ("Grains", "Rough Rice", "$/cwt"),
 "KC=F": ("Softs", "Coffee", "c/lb"), "SB=F": ("Softs", "Sugar #11", "c/lb"),
 "CC=F": ("Softs", "Cocoa", "$/mt"), "CT=F": ("Softs", "Cotton", "c/lb"), "OJ=F": ("Softs", "Orange Juice", "c/lb"),
 "LE=F": ("Livestock", "Live Cattle", "c/lb"), "GF=F": ("Livestock", "Feeder Cattle", "c/lb"),
 "HE=F": ("Livestock", "Lean Hogs", "c/lb"),
 "BDRY": ("Freight", "BDRY ETF (BDI futures proxy)", "$"),
 "DX-Y.NYB": ("FX", "US Dollar Index", "idx"), "KRW=X": ("FX", "USD/KRW", "KRW"),
 "CNY=X": ("FX", "USD/CNY", "CNY"),
 "^GSPC": ("Macro", "S&P 500", "idx"),
}
AK_SHIP = {"macro_shipping_bdi": "BDI (Dry)", "macro_shipping_bci": "BCI (Capesize)",
           "macro_shipping_bpi": "BPI (Panamax)", "macro_shipping_bcti": "BCTI (Dirty Tanker)"}
AK_DCE = {"I0": ("DCE Iron Ore", "CNY/mt"), "M0": ("DCE Soybean Meal", "CNY/mt"), "C0": ("DCE Corn", "CNY/mt"),
          "A0": ("DCE Soybean #1", "CNY/mt"), "Y0": ("DCE Soybean Oil", "CNY/mt"), "JM0": ("DCE Coking Coal", "CNY/mt"),
          "J0": ("DCE Coke", "CNY/mt"), "RB0": ("SHFE Rebar", "CNY/mt")}
FRED = {"DCOILWTICO": "WTI spot (EIA)", "DCOILBRENTEU": "Brent spot (EIA)", "DHHNGSP": "Henry Hub spot (EIA)"}
# 회귀/FAVAR 설명변수 및 위기 식별용 거시 변수
FRED_MACRO = {"VIXCLS": ("VIX", "idx"), "DGS10": ("US 10Y yield", "%"), "DGS2": ("US 2Y yield", "%"),
              "T10YIE": ("10Y breakeven inflation", "%"), "DTWEXBGS": ("Broad USD index", "idx"),
              "BAA10Y": ("Baa - 10Y credit spread", "%"), "DTB3": ("US 3M T-bill", "%"),
              # (2026-10-27 추가) 레짐 탭 밖 금융 변수 — 일별·주별
              "DFII10": ("US 10Y real yield (TIPS)", "%"), "T5YIFR": ("5y5y forward inflation", "%"),
              "BAMLH0A0HYM2": ("High-yield OAS", "%"), "BAMLC0A0CM": ("Investment-grade OAS", "%"),
              "OVXCLS": ("Crude oil VIX (OVX)", "idx"), "GVZCLS": ("Gold VIX (GVZ)", "idx"), "DFF": ("Fed funds effective", "%"),
              "T10Y3M": ("10Y - 3M spread", "%"), "DEXBZUS": ("BRL per USD", "BRL"), "DEXUSEU": ("USD per EUR", "USD"),
              "DEXJPUS": ("JPY per USD", "JPY"), "DEXUSAL": ("USD per AUD", "USD"),
              "NFCI": ("Chicago Fed financial conditions (weekly)", "idx"), "STLFSI4": ("St. Louis financial stress (weekly)", "idx"),
              "ICSA": ("Initial jobless claims (weekly)", "persons"), "WALCL": ("Fed total assets (weekly)", "$mn")}
FRED_M = {"PCOPPUSDM": "Copper (IMF)", "PWHEAMTUSDM": "Wheat (IMF)", "PMAIZMTUSDM": "Maize (IMF)",
          "PSOYBUSDM": "Soybeans (IMF)", "PSMEAUSDM": "Soybean Meal (IMF)", "PCOALAUUSDM": "Coal Australia (IMF)",
          "PIORECRUSDM": "Iron Ore (IMF)", "PALUMUSDM": "Aluminum (IMF)", "PALLFNFINDEXM": "Global All Commodities Idx (IMF)",
          # (2026-10-27 추가) 월간 거시 — 최신 수정치라 백테스트엔 발표 지연(45일) 두고 사용
          "CPIAUCSL": "US CPI", "UNRATE": "US unemployment rate", "INDPRO": "US industrial production", "M2SL": "US M2",
          "UMCSENT": "UMich consumer sentiment", "PPIACO": "US PPI all commodities", "PAYEMS": "US nonfarm payrolls"}
COT = {"002602": "Corn", "005602": "Soybeans", "026603": "Soybean Meal", "007601": "Soybean Oil",
       "001602": "Wheat SRW", "001612": "Wheat HRW", "085692": "Copper", "067651": "WTI Crude", "088691": "Gold"}

status = []
def log(src, name, ok, last="", n=0, note=""):
    status.append(dict(source=src, name=name, ok=ok, last=str(last)[:10], rows=n, note=str(note)[:120]))
    print(f"[{'OK ' if ok else 'ERR'}] {src:10s} {name:34s} {str(last)[:10]:10s} {n:6d} {note if not ok else ''}")

def get(url, **kw):
    for i in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=90, **kw); r.raise_for_status(); return r
        except Exception as e:
            err = e; time.sleep(2)
    raise err

# ---- Yahoo ----------------------------------------------------------------
def collect_yahoo():
    import yfinance as yf
    rows = []
    for t, (grp, name, unit) in YAHOO.items():
        try:
            st_ = since(f"yf:{t}")
            for attempt in range(3):   # Yahoo는 간헐적으로 빈 응답을 줌 -> 재시도
                d = (yf.Ticker(t).history(start=st_.strftime("%Y-%m-%d"), auto_adjust=False) if st_ is not None
                     else yf.Ticker(t).history(period="max", auto_adjust=False))
                s = d["Close"].dropna() if len(d) else pd.Series(dtype=float)
                if len(s): break
                time.sleep(3 * (attempt + 1))
            if s.empty: raise ValueError("empty")
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            rows.append(pd.DataFrame({"date": s.index, "id": f"yf:{t}", "name": name, "group": grp,
                                      "unit": unit, "source": "Yahoo", "value": s.values}))
            log("yahoo", f"{name} ({t})", True, s.index[-1], len(s))
        except Exception as e:
            log("yahoo", f"{name} ({t})", False, note=e)
    return rows

# ---- 원자재 ETF (총수익: 분배금 재투자 조정가) ------------------------------------
# 실제 롤 비용이 반영된 '투자 가능한' 수익률 → 포트폴리오 백테스트의 손익 기준.
# 조정가는 분배 때마다 과거 전체가 재조정되므로 증분이 아닌 매번 전체 이력을 받음 (13종목, 수 초).
ETFS = {"USO": ("yf:CL=F", "WTI 원유 ETF", 0.60), "BNO": ("yf:BZ=F", "Brent 원유 ETF", 1.00),
        "UNG": ("yf:NG=F", "천연가스 ETF", 1.06), "UGA": ("yf:RB=F", "휘발유 ETF", 0.97),
        "CORN": ("yf:ZC=F", "옥수수 ETF", 1.00), "SOYB": ("yf:ZS=F", "대두 ETF", 1.00),
        "WEAT": ("yf:ZW=F", "밀 ETF", 1.00), "CANE": ("yf:SB=F", "설탕 ETF", 1.00),
        "CPER": ("yf:HG=F", "구리 ETF", 0.80), "GLD": ("yf:GC=F", "금 ETF", 0.40),
        "SLV": ("yf:SI=F", "은 ETF", 0.50), "PPLT": ("yf:PL=F", "백금 ETF", 0.60), "PALL": ("yf:PA=F", "팔라듐 ETF", 0.60),
        "DBC": (None, "원자재 지수 ETF (DBC)", 0.85), "DBA": (None, "농산물 지수 ETF (DBA)", 0.85)}


def collect_etf():
    import yfinance as yf
    rows = []
    for t, (fut, name, fee) in ETFS.items():
        try:
            for attempt in range(3):
                d = yf.Ticker(t).history(period="max", auto_adjust=True)
                s = d["Close"].dropna() if len(d) else pd.Series(dtype=float)
                if len(s): break
                time.sleep(3 * (attempt + 1))
            if s.empty: raise ValueError("empty")
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            rows.append(pd.DataFrame({"date": s.index, "id": f"etf:{t}", "name": name, "group": "ETF",
                                      "unit": "$", "source": "Yahoo(조정가)", "value": s.values}))
            log("etf", f"{name} ({t})", True, s.index[-1], len(s))
        except Exception as e:
            log("etf", f"{name} ({t})", False, note=e)
    return rows


# ---- akshare --------------------------------------------------------------
def collect_akshare():
    import akshare as ak
    rows = []
    for fn, name in AK_SHIP.items():
        if fresh(f"ak:{fn}"):
            log("akshare", name, True, LAST[f"ak:{fn}"], 0, note="최신 - 건너뜀"); continue
        try:
            d = getattr(ak, fn)()
            s = pd.Series(pd.to_numeric(d["最新值"], errors="coerce").values, index=pd.to_datetime(d["日期"])).dropna()
            s = s[~s.index.duplicated(keep="last")].sort_index()
            s = s[s.index >= START]
            rows.append(pd.DataFrame({"date": s.index, "id": f"ak:{fn}", "name": name, "group": "Freight",
                                      "unit": "pts", "source": "akshare(Baltic)", "value": s.values}))
            log("akshare", name, True, s.index[-1], len(s))
        except Exception as e:
            log("akshare", name, False, note=e)
    for sym, (name, unit) in AK_DCE.items():
        if fresh(f"ak:{sym}"):
            log("akshare", f"{name} ({sym})", True, LAST[f"ak:{sym}"], 0, note="최신 - 건너뜀"); continue
        try:
            d = ak.futures_zh_daily_sina(symbol=sym)
            s = pd.Series(pd.to_numeric(d["close"]).values, index=pd.to_datetime(d["date"])).dropna()
            s = s[s.index >= START]
            rows.append(pd.DataFrame({"date": s.index, "id": f"ak:{sym}", "name": name, "group": "China (DCE)",
                                      "unit": unit, "source": "akshare(Sina)", "value": s.values}))
            log("akshare", f"{name} ({sym})", True, s.index[-1], len(s))
        except Exception as e:
            log("akshare", f"{name} ({sym})", False, note=e)
    return rows

# ---- FRED -----------------------------------------------------------------
def fred_series(sid, start=None):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
    if start is not None:
        url += f"&cosd={start:%Y-%m-%d}"
    d = pd.read_csv(io.StringIO(get(url).text))
    d.columns = ["date", "value"]
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d["date"] = pd.to_datetime(d["date"])
    return d.dropna()

def collect_fred():
    daily, monthly = [], []
    for sid, name in FRED.items():
        try:
            d = fred_series(sid, since(f"fred:{sid}")); d = d[d.date >= START]
            daily.append(d.assign(id=f"fred:{sid}", name=name, group="FRED daily", unit="$", source="FRED"))
            log("fred", f"{name} ({sid})", True, d.date.iloc[-1], len(d))
        except Exception as e:
            log("fred", f"{name} ({sid})", False, note=e)
    for sid, (name, unit) in FRED_MACRO.items():
        try:
            d = fred_series(sid, since(f"fred:{sid}")); d = d[d.date >= START]
            daily.append(d.assign(id=f"fred:{sid}", name=name, group="Macro", unit=unit, source="FRED"))
            log("fred", f"{name} ({sid})", True, d.date.iloc[-1], len(d))
        except Exception as e:
            log("fred", f"{name} ({sid})", False, note=e)
    for sid, name in FRED_M.items():
        try:
            d = fred_series(sid, since(f"fred:{sid}"))
            monthly.append(d.assign(id=f"fred:{sid}", name=name, group="Monthly", unit="", source="FRED/IMF"))
            log("fred", f"{name} ({sid})", True, d.date.iloc[-1], len(d))
        except Exception as e:
            log("fred", f"{name} ({sid})", False, note=e)
    return daily, monthly

# ---- World Bank Pink Sheet --------------------------------------------------
def collect_worldbank():
    try:
        page = get("https://www.worldbank.org/en/research/commodity-markets").text
        link = re.findall(r'https://thedocs\.worldbank\.org/[^"\']*Monthly\.xlsx', page)[0]
        raw = pd.read_excel(io.BytesIO(get(link).content), sheet_name="Monthly Prices", header=None)
        names, units = raw.iloc[4], raw.iloc[5]
        body = raw.iloc[6:].copy()
        body = body[body[0].astype(str).str.match(r"^\d{4}M\d{2}$")]
        dates = pd.to_datetime(body[0].str.replace("M", "-") + "-01")
        out = []
        for c in range(1, raw.shape[1]):
            nm = str(names[c]).strip(" *")
            if nm == "nan": continue
            v = pd.to_numeric(body[c], errors="coerce")
            d = pd.DataFrame({"date": dates.values, "value": v.values}).dropna()
            d = d[d.date >= START]
            out.append(d.assign(id=f"wb:{nm}", name=nm, group="WorldBank", unit=str(units[c]), source="World Bank"))
        log("worldbank", "Pink Sheet monthly", True, dates.iloc[-1], len(out), note=f"{len(out)} series")
        return out
    except Exception as e:
        log("worldbank", "Pink Sheet monthly", False, note=e)
        return []

# ---- CFTC COT (Disaggregated, futures only) -----------------------------------
def collect_cot():
    out = []
    for code, name in COT.items():
        s0 = since(f"cot:{name}")
        since_ = COT_START if s0 is None else (s0 - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        try:
            url = ("https://publicreporting.cftc.gov/resource/72hh-3qpy.json?"
                   f"$where=cftc_contract_market_code='{code}' AND report_date_as_yyyy_mm_dd>='{since_}'"
                   "&$order=report_date_as_yyyy_mm_dd&$limit=5000")
            j = get(url).json()
            d = pd.DataFrame(j)
            for c in ["m_money_positions_long_all", "m_money_positions_short_all", "open_interest_all",
                      "prod_merc_positions_long", "prod_merc_positions_short"]:
                d[c] = pd.to_numeric(d[c], errors="coerce")
            d["date"] = pd.to_datetime(d["report_date_as_yyyy_mm_dd"])
            d["mm_net"] = d["m_money_positions_long_all"] - d["m_money_positions_short_all"]
            d["comm_net"] = d["prod_merc_positions_long"] - d["prod_merc_positions_short"]
            d["name"] = name
            out.append(d[["date", "name", "mm_net", "comm_net", "open_interest_all"]])
            log("cftc", f"{name} ({code})", True, d.date.iloc[-1], len(d))
        except Exception as e:
            log("cftc", f"{name} ({code})", False, note=e)
    return out

# ---- USDA (키 필요: .env 또는 환경변수 NASS_API_KEY 있을 때만) -------------------
def collect_usda():
    import os
    if not os.environ.get("NASS_API_KEY"):
        log("usda", "NASS QuickStats", False, note="NASS_API_KEY 환경변수 없음 (무료 발급 필요) - 건너뜀")
        return []
    key = os.environ["NASS_API_KEY"]; out = []
    try:
        for desc, short in [("CORN", "Corn"), ("SOYBEANS", "Soybeans")]:
            r = get("https://quickstats.nass.usda.gov/api/api_GET/", params=dict(
                key=key, commodity_desc=desc, statisticcat_desc="CONDITION", unit_desc="PCT GOOD",
                agg_level_desc="NATIONAL", year__GE=2021, format="JSON"))
            d = pd.DataFrame(r.json()["data"]); d["Value"] = pd.to_numeric(d["Value"], errors="coerce")
            d["date"] = pd.to_datetime(d["week_ending"]); d["name"] = f"{short} condition % good"
            out.append(d[["date", "name", "Value"]].rename(columns={"Value": "value"}))
            log("usda", f"{short} crop condition", True, d.date.max(), len(d))
    except Exception as e:
        log("usda", "NASS QuickStats", False, note=e)
    return out

def upsert(path: Path, new: list, keys: list) -> pd.DataFrame:
    """기존 parquet + 새 데이터 병합. 같은 키는 새 값으로 덮어씀(수정치 반영), 이번에 실패한 시리즈는 기존 값 유지."""
    new = pd.concat(new) if new else pd.DataFrame()
    if path.exists() and not FULL:
        old = pd.read_parquet(path)
        merged = pd.concat([old, new], ignore_index=True) if len(new) else old
    else:
        merged = new
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
    merged.to_parquet(path)
    return merged


def load_last():
    """증분 기준일 로딩: 시리즈별 마지막 저장일."""
    for f, key in [("daily.parquet", "id"), ("monthly.parquet", "id")]:
        if (DATA / f).exists():
            d = pd.read_parquet(DATA / f, columns=[key, "date"])
            LAST.update(pd.to_datetime(d.groupby(key).date.max()).to_dict())
    if (DATA / "cot.parquet").exists():
        d = pd.read_parquet(DATA / "cot.parquet", columns=["name", "date"])
        LAST.update({f"cot:{k}": pd.Timestamp(v) for k, v in d.groupby("name").date.max().items()})


def main():
    t0 = time.time()
    if not FULL:
        load_last()
    print(f"모드: {'전체 재다운로드' if FULL or not LAST else '증분 업데이트 (저장된 마지막 날짜 - 14일부터)'}")
    daily = collect_yahoo() + collect_akshare()
    etf = collect_etf()
    fd, fm = collect_fred()
    daily += fd
    wb = collect_worldbank()
    cot = collect_cot()
    usda = collect_usda()

    d = upsert(DATA / "daily.parquet", daily, ["id", "date"])
    if etf:   # ETF는 조정가가 과거까지 재계산되므로 기존 행을 통째로 교체
        new_etf = pd.concat(etf)
        d = pd.concat([d[~d.id.isin(new_etf.id.unique())], new_etf], ignore_index=True)
        d["date"] = pd.to_datetime(d["date"])
        d.sort_values(["id", "date"]).reset_index(drop=True).to_parquet(DATA / "daily.parquet")
    m = upsert(DATA / "monthly.parquet", fm + wb, ["id", "date"])
    if cot: upsert(DATA / "cot.parquet", cot, ["name", "date"])
    if usda: upsert(DATA / "usda.parquet", usda, ["name", "date"])
    pd.DataFrame(status).to_csv(DATA / "status.csv", index=False, encoding="utf-8-sig")
    print(f"\n완료: {sum(s['ok'] for s in status)}/{len(status)} 성공 · 일간 {d.id.nunique()}개 시리즈 {len(d):,}행 · "
          f"월간 {m.id.nunique()}개 {len(m):,}행 · {time.time()-t0:.0f}초")

if __name__ == "__main__":
    main()
