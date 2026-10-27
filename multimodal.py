"""멀티모달 신호 백테스트 — 텍스트 · 기관 리포트 수치 · 기상 · 매크로를 기존 수치 전략에 섞으면?

  갈래(모두 '그 시점에 실제로 알 수 있던 값'만, 발표 시각 이후 금요일부터 사용)
    텍스트 ① WASDE 서술 문단 (USDA 공식 PDF 1~7쪽: WHEAT · COARSE GRAINS · OILSEEDS · SUGAR · LIVESTOCK · COTTON) → FinBERT 감성
    텍스트 ② GDELT 뉴스 헤드라인 (품목별 검색, 월별 최대 250건) → FinBERT 감성 (요청 제한으로 못 받으면 빠짐)
    리포트 수치  WASDE 같은 연산연도 전월 대비 수정폭(기말재고·재고율·생산) · EIA STEO 3개월 가격 전망 수정폭
    기상  NOAA ONI(엘니뇨) · SOI(남방진동) · NASA GISTEMP(지구 기온 편차) — 발표 지연 반영, 품목별 과거 민감도(β) × 현재 값
    매크로  VIX · 신용 스프레드 · 달러 4주 변화 (alt_ts.macro_weekly) — 품목별 β × 현재 값
  신호 → 매주 품목 간 표준화 → 방향은 '그때까지 104주 IC 부호'로만 결정(사후 부호 선택 없음) → 갈래별 평균 → 갈래 평균 = 멀티모달
  매매: 순위 롱숏(상위 1/3 롱 · 하위 1/3 숏, 4주 보유) · 편도 5bp · 목표 변동성 10% — 롱숏 탭 ⑨와 같은 틀
  평가: 2007 – 현재 (롱숏 탭과 같음, 최근 5년 2021-11~ 따로), 자료는 1990년부터(소스가 허용하는 만큼). 혼합 = MinVar 50 + 모멘텀 25 + 멀티모달 25 등.
  FinBERT = ProsusAI/finbert (금융 문장 긍정·부정·중립). 감성 = P(긍정) − P(부정).
"""
from __future__ import annotations

import io
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import ls_strategies as LS

ROOT = Path(__file__).parent
MM = ROOT / "data" / "mm"
MM.mkdir(parents=True, exist_ok=True)
RAW = ROOT / "data" / "reports" / "raw"
UA = {"User-Agent": "commodity-analytics-lab/1.0 (research)"}
EVAL = "2007-01-01"                                                   # 롱숏 탭과 같은 평가 시작
RECENT = "2021-11-01"                                                 # 최근 5년 따로 표시
WARM = "1990-01-01"                                                   # 자료 수집 시작 (기존 자료와 맞춤)
ANN = 52

WASDE_SECTIONS = {"WHEAT": ["yf:ZW=F"], "COARSE GRAINS": ["yf:ZC=F"], "OILSEEDS": ["yf:ZS=F", "yf:ZM=F", "yf:ZL=F"],
                  "SUGAR": ["yf:SB=F"], "LIVESTOCK, POULTRY, AND DAIRY": ["yf:LE=F", "yf:HE=F"], "COTTON": ["yf:CT=F"]}
NEWS_Q = {"yf:ZC=F": '("corn prices" OR "corn futures")', "yf:ZW=F": '("wheat prices" OR "wheat futures")',
          "yf:ZS=F": '("soybean prices" OR "soybean futures")', "yf:ZM=F": '("soybean meal" OR soymeal)', "yf:ZL=F": '("soybean oil" OR soyoil)',
          "yf:CL=F": '("crude oil prices" OR "oil prices" OR "WTI crude")', "yf:HO=F": '("heating oil" OR "diesel prices")',
          "yf:RB=F": '("gasoline prices" OR "gasoline futures")', "yf:NG=F": '("natural gas prices" OR "natural gas futures")',
          "yf:HG=F": '("copper prices" OR "copper futures")', "yf:GC=F": '("gold prices" OR "gold futures")', "yf:SI=F": '("silver prices" OR "silver futures")',
          "yf:PL=F": '("platinum prices" OR "platinum futures")', "yf:KC=F": '("coffee prices" OR "coffee futures")', "yf:SB=F": '("sugar prices" OR "sugar futures")',
          "yf:CT=F": '("cotton prices" OR "cotton futures")', "yf:CC=F": '("cocoa prices" OR "cocoa futures")', "yf:LE=F": '("cattle prices" OR "cattle futures")',
          "yf:HE=F": '("hog prices" OR "lean hogs")'}
STEO_IDS = {"WTIPUUS": ["yf:CL=F", "yf:HO=F", "yf:RB=F"], "NGHHUUS": ["yf:NG=F"]}


# ============================================================================
# FinBERT
# ============================================================================
_FB = {}


def finbert(texts: list[str], batch: int = 32) -> np.ndarray:
    """문장 → P(긍정) − P(부정), −1 ~ +1."""
    if not texts:
        return np.zeros(0)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if not _FB:
        torch.set_num_threads(4)
        _FB["tok"] = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        _FB["m"] = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").eval()
    tok, m = _FB["tok"], _FB["m"]
    lab = {v: k for k, v in m.config.id2label.items()}
    out = []
    with torch.no_grad():
        for k in range(0, len(texts), batch):
            x = tok(texts[k:k + batch], return_tensors="pt", padding=True, truncation=True, max_length=96)
            p = torch.softmax(m(**x).logits, -1).numpy()
            out.append(p[:, lab["positive"]] - p[:, lab["negative"]])
    return np.concatenate(out)


# ============================================================================
# 기상 (NOAA CPC · NASA GISS) — 월간, 발표 지연 후 사용
# ============================================================================
def fetch_weather() -> pd.DataFrame:
    """반환: date(값의 기준 월말) · avail(사용 가능일) · var · value. 캐시 data/mm/weather.parquet."""
    def get(u):
        for k in range(3):
            try:
                r = requests.get(u, headers=UA, timeout=45); r.raise_for_status(); return r.text
            except Exception:
                time.sleep(5 * (k + 1))
        return ""
    rows = []
    sea = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
    for ln in get("https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt").splitlines()[1:]:
        p = ln.split()
        if len(p) == 4 and p[0] in sea:
            end_m = sea.index(p[0]) + 2                                               # 3개월 계절의 마지막 달 (DJF → 2월)
            y = int(p[1]) + (1 if end_m > 12 else 0)
            d = pd.Timestamp(y, (end_m - 1) % 12 + 1, 1) + pd.offsets.MonthEnd(0)
            rows.append(dict(date=d, avail=d + pd.Timedelta(days=12), var="oni", value=float(p[3])))
    txt = get("https://www.cpc.ncep.noaa.gov/data/indices/soi")
    blk = txt.split("STANDARDIZED")[0]                                                # 첫 표 = 기압차 편차
    for ln in blk.splitlines():
        p = ln.split()[:1] + re.findall(r"-?\d+\.\d", ln[4:])                  # '-3.3-999.9'처럼 붙은 값도 분리
        if len(p) == 13 and p[0].isdigit() and len(p[0]) == 4:
            for k, v in enumerate(p[1:]):
                try:
                    f = float(v)
                except ValueError:
                    continue
                if f > -99:
                    d = pd.Timestamp(int(p[0]), k + 1, 1) + pd.offsets.MonthEnd(0)
                    rows.append(dict(date=d, avail=d + pd.Timedelta(days=12), var="soi", value=f))
    gt = get("https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv")
    g = pd.read_csv(io.StringIO(gt), skiprows=1) if gt else pd.DataFrame()
    mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for _, r in g.iterrows():
        for k, mth in enumerate(mon):
            v = pd.to_numeric(r[mth], errors="coerce")
            if pd.notna(v):
                d = pd.Timestamp(int(r["Year"]), k + 1, 1) + pd.offsets.MonthEnd(0)
                rows.append(dict(date=d, avail=d + pd.Timedelta(days=20), var="gistemp", value=float(v)))
    nc = get("https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series/globe/land_ocean/1/0/1950-2026/data.csv")
    for ln in nc.splitlines():                                        # NOAA NCEI 지구 육지·해양 기온 편차 (NASA 접속 실패 대비, 같은 개념)
        m = re.match(r"^(\d{4})(\d{2}),(-?[\d.]+)", ln)
        if m:
            d = pd.Timestamp(int(m.group(1)), int(m.group(2)), 1) + pd.offsets.MonthEnd(0)
            rows.append(dict(date=d, avail=d + pd.Timedelta(days=20), var="gtemp_noaa", value=float(m.group(3))))
    df = pd.DataFrame(rows)
    f = MM / "weather.parquet"
    if f.exists():                                                    # 받지 못한 소스는 이전 캐시 유지
        old = pd.read_parquet(f)
        df = pd.concat([old[~old["var"].isin(set(df["var"]) if len(df) else set())], df], ignore_index=True)
    df = df.drop_duplicates(["var", "date"], keep="last")
    df.to_parquet(f, index=False)
    return df


def weather_weekly(idx: pd.DatetimeIndex) -> pd.DataFrame:
    W = pd.read_parquet(MM / "weather.parquet") if (MM / "weather.parquet").exists() else fetch_weather()
    out = {}
    for v, g in W.groupby("var"):
        s = g.sort_values("avail").set_index("avail")["value"]
        out[v] = s.reindex(s.index.union(idx)).ffill().reindex(idx)                  # 금요일에 '이미 발표된' 최신 값
    return pd.DataFrame(out)


# ============================================================================
# WASDE 서술 문단 (PDF) → FinBERT
# ============================================================================
def wasde_pdf_index(pages=range(0, 80), start: str | None = None) -> pd.DataFrame:
    """ESMIS 목록을 최신 → 과거로 넘기며 PDF 링크 수집. start보다 오래된 발간본까지 내려가면 멈춤 (1990년대까지 약 45쪽)."""
    import reports as R
    rows = []
    for p in pages:
        html = R._get(R.WASDE_LIST.format(p=p)).decode("utf-8", "ignore")
        last, n0 = None, len(rows)
        for m in re.finditer(r'datetime="([^"]+)"|(/sites/default/release-files/[^"]+?\.pdf)"', html, flags=re.I):
            if m.group(1):
                last = m.group(1)
            elif last:
                rows.append(dict(release=pd.Timestamp(last), url=R.ESMIS + m.group(2)))
        time.sleep(0.5)
        if len(rows) == n0 or (start and rows and min(r["release"] for r in rows[n0:]).tz_localize(None) < pd.Timestamp(start)):
            break
    df = pd.DataFrame(rows)
    df["orig"] = ~df.url.str.lower().str.rsplit("/", n=1).str[-1].str.match(r"r\d")
    return df.sort_values(["release", "orig"], ascending=[True, False]).drop_duplicates("release").reset_index(drop=True)


def _wasde_sections(raw: bytes) -> dict:
    from pypdf import PdfReader
    rd = PdfReader(io.BytesIO(raw))
    txt = "\n".join((rd.pages[k].extract_text() or "") for k in range(min(7, len(rd.pages))))
    txt = re.sub(r"\s+", " ", txt)
    heads = list(WASDE_SECTIONS) + ["RICE"]
    pat = re.compile(r"\b(" + "|".join(re.escape(h) for h in heads) + r"):")
    hits = [(m.start(), m.group(1)) for m in pat.finditer(txt)]
    out = {}
    for k, (pos, h) in enumerate(hits):
        if h in out or h == "RICE":
            continue
        end = hits[k + 1][0] if k + 1 < len(hits) else min(len(txt), pos + 6000)
        out[h] = txt[pos + len(h) + 1:end].strip()[:6000]
    return out


def wasde_text_scores(start: str = WARM, progress=print) -> pd.DataFrame:
    """발간본별 섹션 FinBERT 평균 감성. 캐시 data/mm/wasde_text.parquet (새 발간본만 추가)."""
    f = MM / "wasde_text.parquet"
    old = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["release", "section", "tone", "n_sent"])
    idx = wasde_pdf_index(start=start)
    idx = idx[idx.release >= pd.Timestamp(start, tz=idx.release.dt.tz)]
    done = set(pd.to_datetime(old.release).astype(str)) if len(old) else set()
    import reports as R
    rows = []
    for _, r in idx.iterrows():
        if str(r.release) in done:
            continue
        path = RAW / r.release.strftime("wasde_%Y%m%d.pdf")
        try:
            sec = _wasde_sections(R._get(r.url, path))
        except Exception as e:
            progress(f"WASDE PDF 실패 {r.release:%Y-%m} {e}")
            continue
        for h, body in sec.items():
            sents = [s.strip() for s in re.split(r"(?<=[.;])\s+", body) if 25 <= len(s.strip()) <= 400]
            if sents:
                sc = finbert(sents)
                rows.append(dict(release=r.release, section=h, tone=float(sc.mean()), n_sent=len(sents)))
        progress(f"WASDE 원문 {r.release:%Y-%m} 섹션 {len(sec)}개")
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if rows else old
    if len(df):
        df["tone"] = pd.to_numeric(df.tone, errors="coerce"); df["n_sent"] = pd.to_numeric(df.n_sent, errors="coerce")
        df.to_parquet(f, index=False)
    return df


# ============================================================================
# GDELT 뉴스 헤드라인 → FinBERT (요청 제한: 5초에 1번)
# ============================================================================
def gdelt_collect(start: str = "2017-01-01", end: str | None = None, ids=None, progress=print, max_fail: int = 12, pace: float = 6.0) -> pd.DataFrame:
    """품목 × 월 artlist(최대 250건) → 헤드라인 캐시 data/mm/gdelt_titles.parquet. 연속 실패 max_fail회면 중단."""
    f = MM / "gdelt_titles.parquet"
    old = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["id", "month", "date", "title", "domain"])
    have = set(zip(old.id, old.month)) if len(old) else set()
    months = pd.period_range(start, end or pd.Timestamp.today(), freq="M")
    rows, fail = [], 0
    for i in (ids or NEWS_Q):
        for mth in months:
            key = str(mth)
            if (i, key) in have:
                continue
            a, b = mth.start_time.strftime("%Y%m%d%H%M%S"), mth.end_time.strftime("%Y%m%d%H%M%S")
            ok = False
            for k in range(4):
                time.sleep(pace + 20 * k)
                try:
                    r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", headers=UA, timeout=60,
                                     params=dict(query=NEWS_Q[i] + " sourcelang:english", mode="artlist", maxrecords=250, format="json",
                                                 startdatetime=a, enddatetime=b, sort="hybridrel"))
                    if r.status_code == 429:
                        continue
                    arts = (r.json() or {}).get("articles", []) if r.text.strip().startswith("{") else []
                    rows += [dict(id=i, month=key, date=pd.to_datetime(x.get("seendate"), format="%Y%m%dT%H%M%SZ", errors="coerce"),
                                  title=x.get("title", ""), domain=x.get("domain", "")) for x in arts]
                    if not arts:
                        rows.append(dict(id=i, month=key, date=pd.NaT, title="", domain=""))   # 0건도 '받음'으로 기록
                    ok = True
                    break
                except Exception:
                    continue
            fail = 0 if ok else fail + 1
            if ok and len(rows) % 500 < 250:
                pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_parquet(f, index=False)
            progress(f"GDELT {i} {key} {'ok' if ok else '실패'}")
            if fail >= max_fail:
                progress("GDELT 연속 실패 — 중단 (요청 제한)")
                break
        if fail >= max_fail:
            break
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if rows else old
    if len(df):
        df.to_parquet(f, index=False)
    return df


def gdelt_scores() -> pd.DataFrame:
    """헤드라인 FinBERT → 주간(W-FRI) 품목별 평균 감성·건수. 캐시 data/mm/gdelt_scored.parquet."""
    f, g = MM / "gdelt_scored.parquet", MM / "gdelt_titles.parquet"
    if not g.exists():
        return pd.DataFrame()
    T = pd.read_parquet(g)
    T = T[(T.title.str.len() > 15) & T.date.notna()].drop_duplicates(["id", "title"])
    S = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["id", "title", "score"])
    new = T.merge(S[["id", "title"]], on=["id", "title"], how="left", indicator=True)
    new = new[new["_merge"] == "left_only"]
    if len(new):
        sc = finbert(new.title.tolist())
        S = pd.concat([S, pd.DataFrame(dict(id=new.id.values, title=new.title.values, score=sc))], ignore_index=True)
        S.to_parquet(f, index=False)
    T = T.merge(S, on=["id", "title"])
    T["wk"] = pd.to_datetime(T.date).dt.to_period("W-FRI").dt.to_timestamp(how="end").dt.normalize()
    return T.groupby(["wk", "id"]).agg(tone=("score", "mean"), n=("score", "size")).reset_index()


# ============================================================================
# Reddit 수집기 (앞으로 쌓기 — 과거 자료는 API가 주지 않아 백테스트에는 미사용)
# ============================================================================
SUBS = ["commodities", "Gold", "Silverbugs", "oil", "agriculture", "farming", "energy", "investing"]


def reddit_collect(limit: int = 100) -> pd.DataFrame:
    """.env의 REDDIT_CLIENT_ID · REDDIT_CLIENT_SECRET · REDDIT_USER_AGENT로 OAuth(앱 전용) → 서브레딧 최신 글 → FinBERT.
    저장 data/mm/reddit_posts.parquet (id 중복 제거, 매일 실행해 누적)."""
    import os
    env = ROOT / ".env"                                               # collect._load_env와 같은 KEY=VALUE 형식, 환경변수 우선
    if env.exists():
        for ln in env.read_text(encoding="utf-8").splitlines():
            if "=" in ln and not ln.lstrip().startswith("#"):
                k, v = ln.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
    cid, sec, ua = os.getenv("REDDIT_CLIENT_ID"), os.getenv("REDDIT_CLIENT_SECRET"), os.getenv("REDDIT_USER_AGENT", "commodity-lab/1.0")
    if not cid or not sec:
        raise RuntimeError("REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET 이 .env에 없음 — reddit.com/prefs/apps 에서 'script' 앱을 만들어 넣어 주세요")
    tok = requests.post("https://www.reddit.com/api/v1/access_token", auth=(cid, sec), data={"grant_type": "client_credentials"},
                        headers={"User-Agent": ua}, timeout=30).json()["access_token"]
    H = {"Authorization": f"bearer {tok}", "User-Agent": ua}
    kw = {i: [w.strip('"()').lower() for w in re.findall(r'"([^"]+)"', q)] for i, q in NEWS_Q.items()}
    rows = []
    for s in SUBS:
        r = requests.get(f"https://oauth.reddit.com/r/{s}/new", headers=H, params={"limit": limit}, timeout=30).json()
        for c in r.get("data", {}).get("children", []):
            d = c["data"]; text = (d.get("title", "") + " " + d.get("selftext", "")[:300]).strip()
            ids = [i for i, ws in kw.items() if any(w.split()[0] in text.lower() for w in ws)]
            for i in ids:
                rows.append(dict(post=d["id"], sub=s, id=i, created=pd.to_datetime(d["created_utc"], unit="s"), text=text[:400], score_up=d.get("score", 0)))
        time.sleep(1)
    df = pd.DataFrame(rows)
    f = MM / "reddit_posts.parquet"
    if len(df):
        df["tone"] = finbert(df.text.tolist())
        old = pd.read_parquet(f) if f.exists() else pd.DataFrame()
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(["post", "id"], keep="first")
        df.to_parquet(f, index=False)
    return df


# ============================================================================
# USDA NASS 주간 작황 등급 (키: .env NASS_API_KEY) — 월요일 16:00 ET 발표, 일요일 기준 주
# ============================================================================
NASS_MAP = {"CORN": ["yf:ZC=F"], "SOYBEANS": ["yf:ZS=F", "yf:ZM=F", "yf:ZL=F"], "WHEAT": ["yf:ZW=F"], "COTTON": ["yf:CT=F"]}


def nass_fetch(start_year: int = 1990) -> pd.DataFrame:
    import os
    env = ROOT / ".env"
    if env.exists():
        for ln in env.read_text(encoding="utf-8").splitlines():
            if "=" in ln and not ln.lstrip().startswith("#"):
                k, v = ln.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
    key = os.environ.get("NASS_API_KEY")
    if not key:
        raise RuntimeError("NASS_API_KEY 가 .env에 없음")
    rows = []
    for desc in NASS_MAP:
        r = requests.get("https://quickstats.nass.usda.gov/api/api_GET/", timeout=90, params=dict(
            key=key, commodity_desc=desc, statisticcat_desc="CONDITION", agg_level_desc="NATIONAL", year__GE=start_year, format="JSON"))
        r.raise_for_status()
        rows.append(pd.DataFrame(r.json()["data"]))
    D = pd.concat(rows, ignore_index=True)
    D.to_parquet(MM / "nass_condition_raw.parquet", index=False)
    return D


def nass_ge() -> pd.DataFrame:
    """작물별 주간 '우량 비율'(GOOD + EXCELLENT, %) — 밀은 겨울밀·봄밀 평균. 반환 long: avail · crop · ge."""
    f = MM / "nass_condition_raw.parquet"
    D = pd.read_parquet(f) if f.exists() else nass_fetch()
    D = D[D.unit_desc.isin(["PCT GOOD", "PCT EXCELLENT"]) & (D.reference_period_desc.str.startswith("WEEK"))]
    D = D[~D.short_desc.str.contains("DURUM|PIMA", regex=True)]
    D["v"] = pd.to_numeric(D.Value.astype(str).str.replace(",", ""), errors="coerce")
    D["we"] = pd.to_datetime(D.week_ending)
    D["variety"] = D.short_desc.str.split(" - ").str[0]                                    # 'WHEAT, WINTER - CONDITION, MEASURED IN PCT GOOD' → 'WHEAT, WINTER'
    g = D.groupby(["commodity_desc", "variety", "we"]).v.sum().reset_index()                # 품종별 우량 = 양호 + 우수
    g = g.groupby(["commodity_desc", "we"]).v.mean().reset_index().rename(columns={"commodity_desc": "crop", "v": "ge"})
    g["avail"] = g.we + pd.Timedelta(days=2)                                               # 일요일 기준 → 월요일 오후 발표 → 화요일부터
    return g


# ============================================================================
# Open-Meteo 과거 날씨 — 품목별 주산지 (ERA5 재분석, 약 5일 지연)
# ============================================================================
REGIONS = {  # 이름: (위도, 경도, 품목들, 종류) — crop = 강수·기온 편차, energy = 냉난방 수요일수(HDD+CDD) 편차
    "미국 아이오와": (41.9, -93.4, ["yf:ZC=F", "yf:ZS=F", "yf:ZM=F", "yf:ZL=F", "yf:HE=F"], "crop"),
    "미국 일리노이": (40.1, -89.2, ["yf:ZC=F", "yf:ZS=F", "yf:ZM=F", "yf:ZL=F"], "crop"),
    "미국 네브래스카": (41.5, -99.8, ["yf:ZC=F", "yf:LE=F"], "crop"),
    "브라질 마투그로수": (-12.6, -55.7, ["yf:ZS=F", "yf:ZM=F", "yf:ZL=F", "yf:ZC=F"], "crop"),
    "아르헨티나 부에노스아이레스": (-36.6, -60.5, ["yf:ZS=F", "yf:ZM=F", "yf:ZL=F", "yf:ZC=F", "yf:ZW=F"], "crop"),
    "미국 캔자스": (38.5, -98.4, ["yf:ZW=F", "yf:LE=F"], "crop"),
    "미국 노스다코타": (47.5, -100.5, ["yf:ZW=F"], "crop"),
    "러시아 로스토프": (47.2, 39.7, ["yf:ZW=F"], "crop"),
    "우크라이나 헤르손": (46.6, 32.6, ["yf:ZW=F"], "crop"),
    "브라질 미나스제라이스": (-21.0, -45.0, ["yf:KC=F"], "crop"),
    "베트남 닥락": (12.7, 108.1, ["yf:KC=F"], "crop"),
    "코트디부아르": (6.9, -5.3, ["yf:CC=F"], "crop"),
    "가나 쿠마시": (6.7, -1.6, ["yf:CC=F"], "crop"),
    "브라질 상파울루": (-21.2, -48.5, ["yf:SB=F"], "crop"),
    "인도 우타르프라데시": (27.0, 80.9, ["yf:SB=F"], "crop"),
    "미국 텍사스 러벅": (33.6, -101.9, ["yf:CT=F", "yf:LE=F"], "crop"),
    "인도 구자라트": (22.3, 71.2, ["yf:CT=F"], "crop"),
    "뉴욕": (40.7, -74.0, ["yf:NG=F", "yf:HO=F"], "energy"),
    "시카고": (41.9, -87.6, ["yf:NG=F", "yf:HO=F"], "energy"),
    "휴스턴": (29.8, -95.4, ["yf:NG=F"], "energy"),
    "애틀랜타": (33.7, -84.4, ["yf:NG=F"], "energy"),
    "보스턴": (42.4, -71.1, ["yf:HO=F", "yf:NG=F"], "energy"),
}


def openmeteo_fetch(start: str = WARM) -> pd.DataFrame:
    """지역별 일 평균기온·강수량 → data/mm/openmeteo.parquet (이미 받은 지역은 마지막 날짜 이후만)."""
    f = MM / "openmeteo.parquet"
    old = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["region", "date", "t", "p"])
    end = (pd.Timestamp.today() - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
    rows = []
    for nm, (lat, lon, _, _) in REGIONS.items():
        o = old[old.region == nm]
        if len(o):                                                    # 앞쪽(start까지)과 뒤쪽(최근) 빈 구간만
            lo, hi = pd.Timestamp(o.date.min()), pd.Timestamp(o.date.max())
            spans = [(start, (lo - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))] if lo > pd.Timestamp(start) else []
            spans += [((hi + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), end)] if (hi + pd.Timedelta(days=1)).strftime("%Y-%m-%d") <= end else []
        else:
            spans = [(start, end)]
        for st_, en_ in spans:
            rows += _om_get(nm, lat, lon, st_, en_)
            time.sleep(3)                                             # 분당 요청 제한 (429) 회피
    df = pd.concat([old] + rows, ignore_index=True) if rows else old
    df = df.drop_duplicates(["region", "date"], keep="last")
    df.to_parquet(f, index=False)
    return df


def _om_get(nm, lat, lon, st_, end) -> list:
    err = None
    for k in range(4):
        try:
            r = requests.get("https://archive-api.open-meteo.com/v1/archive", headers=UA, timeout=90, params=dict(
                latitude=lat, longitude=lon, start_date=st_, end_date=end, daily="temperature_2m_mean,precipitation_sum", timezone="UTC"))
            r.raise_for_status()
            d = r.json()["daily"]
            return [pd.DataFrame(dict(region=nm, date=pd.to_datetime(d["time"]), t=d["temperature_2m_mean"], p=d["precipitation_sum"]))]
        except Exception as e:
            err = e
            time.sleep(20 * (k + 1))                                  # 429면 넉넉히 쉬고 재시도
    print("Open-Meteo 실패", nm, err)
    return []


def openmeteo_features(idx: pd.DatetimeIndex, cols) -> dict:
    """품목별 4주 강수 편차(같은 주차 과거 대비 z) · 4주 기온 편차(°C) · (에너지) 4주 HDD+CDD 편차.
    평년 = 같은 주차의 그해 이전 연도 평균(최소 5년). 금요일에는 6일 전까지의 날씨만 앎(ERA5 지연)."""
    f = MM / "openmeteo.parquet"
    D = pd.read_parquet(f) if f.exists() else openmeteo_fetch()
    D["date"] = pd.to_datetime(D.date)
    out = {"Open-Meteo 4주 강수 편차 (평년 대비 z)": {}, "Open-Meteo 4주 기온 편차 (°C)": {}, "Open-Meteo 4주 냉난방 수요 편차 (HDD+CDD)": {}}
    woy, yr = idx.isocalendar().week.values, idx.year.values
    for nm, g in D.groupby("region"):
        lat, lon, ids, kind = REGIONS[nm]
        g = g.set_index("date").sort_index()
        dd = (18.0 - g.t).clip(lower=0) + (g.t - 18.0).clip(lower=0)                     # 18°C 기준 냉난방 수요일수
        w = pd.DataFrame({"p": g.p.rolling(28, min_periods=20).sum(), "t": g.t.rolling(28, min_periods=20).mean(),
                          "dd": dd.rolling(28, min_periods=20).sum()})
        seen = idx - pd.Timedelta(days=6)
        w = w.reindex(w.index.union(seen)).ffill().reindex(seen)
        w.index = idx
        for c in ("p", "t", "dd"):
            v = w[c].values
            clim, csd = np.full(len(w), np.nan), np.full(len(w), np.nan)
            for j in range(len(w)):
                m = (woy == woy[j]) & (yr < yr[j]) & (yr >= yr[j] - 20)
                if m.sum() >= 5:
                    clim[j], csd[j] = np.nanmean(v[m]), np.nanstd(v[m])
            # 강수는 건기에 평년이 0에 가까워 %가 폭주 → 같은 주차 과거 표준편차로 나눈 z, 기온·냉난방은 차이 그대로
            w[c + "_a"] = (v - clim) / np.maximum(csd, 1.0) if c == "p" else v - clim
        for i in ids:
            if kind == "crop":
                out["Open-Meteo 4주 강수 편차 (평년 대비 z)"].setdefault(i, []).append(w["p_a"])
                out["Open-Meteo 4주 기온 편차 (°C)"].setdefault(i, []).append(w["t_a"])
            else:
                out["Open-Meteo 4주 냉난방 수요 편차 (HDD+CDD)"].setdefault(i, []).append(w["dd_a"])
    res = {}
    for nm, per in out.items():
        T = pd.DataFrame(np.nan, index=idx, columns=cols)
        for i, lst in per.items():
            if i in T:
                T[i] = pd.concat(lst, axis=1).mean(1)
        if T.notna().values.any():
            res[nm] = T
    return res


# ============================================================================
# FRED 추가 금융·거시 (레짐 탭 밖) — fredgraph CSV, 키 불필요
# ============================================================================
FRED_FIN = {"DFII10": ("10년 실질금리", "d4"), "T5YIFR": ("5y5y 기대인플레", "d4"), "BAMLH0A0HYM2": ("하이일드 스프레드", "d4"),
            "BAMLC0A0CM": ("투자등급 스프레드", "d4"), "DFF": ("기준금리", "d4"), "T10Y3M": ("10년−3개월 금리차", "lvl"),
            "DEXBZUS": ("헤알/달러", "r4"), "DEXUSEU": ("달러/유로", "r4"), "DEXJPUS": ("엔/달러", "r4"), "DEXUSAL": ("달러/호주달러", "r4"),
            "NFCI": ("금융여건 NFCI (주간)", "lvl"), "STLFSI4": ("금융스트레스 STLFSI (주간)", "lvl"), "ICSA": ("신규 실업수당 청구 (주간)", "r4"),
            "WALCL": ("연준 총자산 (주간)", "r13")}
FRED_IV = {"OVXCLS": ("원유 변동성지수 OVX", ["yf:CL=F", "yf:HO=F", "yf:RB=F"]), "GVZCLS": ("금 변동성지수 GVZ", ["yf:GC=F", "yf:SI=F", "yf:PL=F"])}
FRED_MON = {"CPIAUCSL": "CPI", "UNRATE": "실업률", "INDPRO": "산업생산", "M2SL": "M2", "UMCSENT": "소비자심리", "PPIACO": "PPI (원자재)", "PAYEMS": "비농업 고용"}


def fred_extra_fetch() -> pd.DataFrame:
    rows = []
    for sid in list(FRED_FIN) + list(FRED_IV) + list(FRED_MON):
        for k in range(3):
            try:
                r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params=dict(id=sid), headers=UA, timeout=60)
                r.raise_for_status()
                d = pd.read_csv(io.StringIO(r.text))
                d.columns = ["date", "value"]
                d["value"] = pd.to_numeric(d.value, errors="coerce")
                d["date"] = pd.to_datetime(d.date)
                rows.append(d.dropna().assign(id=sid))
                break
            except Exception:
                time.sleep(3 * (k + 1))
        time.sleep(0.5)
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(MM / "fred_extra.parquet", index=False)
    return df


def fred_extra_features(idx: pd.DatetimeIndex, cols, R: pd.DataFrame) -> tuple[dict, dict]:
    """(금융 일·주간 특성, 월간 거시 특성). 공통 값은 품목별 β × 값, 변동성지수는 해당 품목에 직접."""
    f = MM / "fred_extra.parquet"
    D = pd.read_parquet(f) if f.exists() else fred_extra_fetch()
    W = {sid: g.set_index("date").value.sort_index() for sid, g in D.groupby("id")}
    wk = lambda s: s.resample("W-FRI").last().reindex(idx).ffill(limit=2)                 # noqa: E731
    z = lambda x: ((x - x.expanding(104).mean()) / x.expanding(104).std()).shift(1)        # noqa: E731 — 과거로만 표준화, 1주 지연
    fin, mon = {}, {}
    for sid, (nm, how) in FRED_FIN.items():
        if sid not in W:
            continue
        s = wk(W[sid])
        x = {"d4": s.diff(4), "r4": 100 * np.log(s).diff(4), "r13": 100 * np.log(s).diff(13), "lvl": s}[how]
        fin[nm + " × 품목 민감도"] = _beta_pred(R, z(x))
    for sid, (nm, ids) in FRED_IV.items():                          # 해당 품목만 같은 값을 주면 품목 간 차이가 없음 → 공통 변수처럼 품목별 β
        if sid in W:
            fin[nm + " 4주 변화 × 품목 민감도"] = _beta_pred(R, z(100 * np.log(wk(W[sid])).diff(4)))
    for sid, nm in FRED_MON.items():
        if sid not in W:
            continue
        m = W[sid]
        yoy = m.diff(12) if sid in ("UNRATE", "UMCSENT") else (m / m.shift(12) - 1) * 100
        yoy.index = yoy.index + pd.offsets.MonthEnd(0) + pd.Timedelta(days=45)            # 발표 지연 보수적 45일 (수정치 문제는 남음)
        x = yoy.reindex(yoy.index.union(idx)).ffill().reindex(idx)
        mon[nm + " 전년비 × 품목 민감도"] = _beta_pred(R, z(x))
    return fin, mon


# ============================================================================
# CFTC COT 19개 품목 (기존형 선물 보고서 — 1986–2016 압축 파일 + 1998~ API)
# ============================================================================
COT19 = {"yf:ZC=F": "002602", "yf:ZW=F": "001602", "yf:ZS=F": "005602", "yf:ZM=F": "026603", "yf:ZL=F": "007601",
         "yf:CL=F": "067651", "yf:HO=F": "022651", "yf:RB=F": "111659", "yf:NG=F": "023651", "yf:HG=F": "085692",
         "yf:GC=F": "088691", "yf:SI=F": "084691", "yf:PL=F": "076651", "yf:KC=F": "083731", "yf:SB=F": "080732",
         "yf:CT=F": "033661", "yf:CC=F": "073732", "yf:LE=F": "057642", "yf:HE=F": "054642"}


def cot_fetch() -> pd.DataFrame:
    """반환 long: id · date(화요일 기준일) · spec_long · spec_short · comm_long · comm_short · oi. 캐시 data/mm/cot19.parquet."""
    codes = {v: k for k, v in COT19.items()}
    rows = []
    zf = MM / "deacot1986_2016.zip"
    if not zf.exists():
        r = requests.get("https://www.cftc.gov/files/dea/history/deacot1986_2016.zip", headers=UA, timeout=180)
        r.raise_for_status(); zf.write_bytes(r.content)
    import zipfile
    with zipfile.ZipFile(zf) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".txt")][0]
        H = pd.read_csv(z.open(name), low_memory=False)
    H.columns = [c.strip() for c in H.columns]
    H["code"] = H["CFTC Contract Market Code"].astype(str).str.strip().str.zfill(6)
    H = H[H.code.isin(codes)]
    rows.append(pd.DataFrame(dict(id=H.code.map(codes), date=pd.to_datetime(H["As of Date in Form YYYY-MM-DD"]),
                                  spec_long=H["Noncommercial Positions-Long (All)"], spec_short=H["Noncommercial Positions-Short (All)"],
                                  comm_long=H["Commercial Positions-Long (All)"], comm_short=H["Commercial Positions-Short (All)"],
                                  oi=H["Open Interest (All)"])))
    for code, i in codes.items():
        r = requests.get("https://publicreporting.cftc.gov/resource/6dca-aqww.json", headers=UA, timeout=90, params={
            "$where": f"cftc_contract_market_code='{code}'", "$order": "report_date_as_yyyy_mm_dd", "$limit": 50000,
            "$select": "report_date_as_yyyy_mm_dd,noncomm_positions_long_all,noncomm_positions_short_all,comm_positions_long_all,comm_positions_short_all,open_interest_all"})
        d = pd.DataFrame(r.json())
        if len(d):
            rows.append(pd.DataFrame(dict(id=i, date=pd.to_datetime(d.report_date_as_yyyy_mm_dd), spec_long=d.noncomm_positions_long_all,
                                          spec_short=d.noncomm_positions_short_all, comm_long=d.comm_positions_long_all,
                                          comm_short=d.comm_positions_short_all, oi=d.open_interest_all)))
        time.sleep(0.5)
    D = pd.concat(rows, ignore_index=True)
    for c in ["spec_long", "spec_short", "comm_long", "comm_short", "oi"]:
        D[c] = pd.to_numeric(D[c], errors="coerce")
    D = D.dropna(subset=["oi"]).drop_duplicates(["id", "date"], keep="last").sort_values(["id", "date"])
    D.to_parquet(MM / "cot19.parquet", index=False)
    return D


def cot_features(idx: pd.DatetimeIndex, cols) -> dict:
    """화요일 기준 → 금요일 15:30 ET 발표(대부분 장 마감 뒤) → 다음 금요일부터(기준일 + 10일).
    투기 순포지션 비율 = (비상업 매수 − 매도)/미결제, 쏠림 = 그 비율의 과거 156주 대비 z, 헤징 압력 = 상업 순포지션 비율."""
    f = MM / "cot19.parquet"
    D = pd.read_parquet(f) if f.exists() else cot_fetch()
    D["spec"] = (D.spec_long - D.spec_short) / D.oi * 100
    D["comm"] = (D.comm_long - D.comm_short) / D.oi * 100
    out = {}
    for c, nm in (("spec", "투기 순포지션 비율 (%)"), ("comm", "상업(헤지) 순포지션 비율 (%)")):
        T = pd.DataFrame(np.nan, index=idx, columns=cols)
        for i, g in D.groupby("id"):
            if i not in T:
                continue
            s = g.set_index("date")[c].sort_index()
            s.index = s.index + pd.Timedelta(days=10)
            T[i] = s.reindex(s.index.union(idx)).ffill().reindex(idx)
        out[nm] = T
    sp = out["투기 순포지션 비율 (%)"]
    out["투기 쏠림 (156주 대비 z)"] = (sp - sp.rolling(156, min_periods=52).mean()) / sp.rolling(156, min_periods=52).std()
    out["투기 순포지션 4주 변화 (%p)"] = sp.diff(4)
    return out


# ============================================================================
# EIA 주간 재고 (키: .env EIA_API_KEY) — 원유·휘발유·증류유 수요일, 천연가스 목요일 발표
# ============================================================================
EIA_SERIES = {"PET.WCESTUS1.W": ("원유 재고 (SPR 제외)", ["yf:CL=F"]), "PET.WGTSTUS1.W": ("휘발유 재고", ["yf:RB=F"]),
              "PET.WDISTUS1.W": ("증류유 재고", ["yf:HO=F"]), "NG.NW2_EPG0_SWO_R48_BCF.W": ("천연가스 저장량", ["yf:NG=F"])}


def _env(key: str) -> str | None:
    import os
    env = ROOT / ".env"
    if env.exists():
        for ln in env.read_text(encoding="utf-8").splitlines():
            if "=" in ln and not ln.lstrip().startswith("#"):
                k, v = ln.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
    return os.environ.get(key)


def eia_fetch() -> pd.DataFrame:
    key = _env("EIA_API_KEY")
    if not key:
        raise RuntimeError("EIA_API_KEY 가 .env에 없음")
    rows = []
    for sid in EIA_SERIES:
        r = requests.get(f"https://api.eia.gov/v2/seriesid/{sid}", params={"api_key": key, "length": 5000}, headers=UA, timeout=90)
        r.raise_for_status()
        d = pd.DataFrame(r.json()["response"]["data"])
        rows.append(pd.DataFrame(dict(sid=sid, date=pd.to_datetime(d.period), value=pd.to_numeric(d.value, errors="coerce"))))
        time.sleep(0.5)
    D = pd.concat(rows, ignore_index=True).dropna()
    D.to_parquet(MM / "eia_weekly.parquet", index=False)
    return D


def _seasonal_dev(s: pd.Series, years: int = 5) -> tuple[pd.Series, pd.Series]:
    """주간 재고 → (주간 변화 − 직전 5년 같은 주 평균 변화, 수준의 직전 5년 같은 주 평균 대비 %). 그해 이전 자료만."""
    s = s.sort_index()
    ch = s.diff()
    woy, yr = s.index.isocalendar().week.values, s.index.year.values
    d1, lv = np.full(len(s), np.nan), np.full(len(s), np.nan)
    for j in range(len(s)):
        m = (woy == woy[j]) & (yr < yr[j]) & (yr >= yr[j] - years)
        if m.sum() >= 3:
            d1[j] = ch.values[j] - np.nanmean(ch.values[m])
            lv[j] = (s.values[j] / np.nanmean(s.values[m]) - 1) * 100
    return pd.Series(d1, index=s.index), pd.Series(lv, index=s.index)


def eia_features(idx: pd.DatetimeIndex, cols) -> dict:
    f = MM / "eia_weekly.parquet"
    D = pd.read_parquet(f) if f.exists() else eia_fetch()
    out = {"EIA 재고 주간 변화 vs 5년 평년 (계절 조정)": pd.DataFrame(np.nan, index=idx, columns=cols),
           "EIA 재고 수준 vs 5년 평년 (%)": pd.DataFrame(np.nan, index=idx, columns=cols)}
    for sid, g in D.groupby("sid"):
        nm, ids = EIA_SERIES[sid]
        d1, lv = _seasonal_dev(g.set_index("date").value)
        for s, key in ((d1, "EIA 재고 주간 변화 vs 5년 평년 (계절 조정)"), (lv, "EIA 재고 수준 vs 5년 평년 (%)")):
            s = s.copy(); s.index = s.index + pd.Timedelta(days=7)            # 금요일 기준 주 → 다음 주 수·목 발표 → 그다음 금요일부터
            v = s.reindex(s.index.union(idx)).ffill().reindex(idx)
            for i in ids:
                if i in cols:
                    out[key][i] = v
    return out


# ============================================================================
# 금속 수치: LME 창고 재고(2014~) · SPDR 금 ETF(2004~) · iShares 은 ETF(2014~) — akshare(jin10)
# ============================================================================
def metals_fetch() -> pd.DataFrame:
    import akshare as ak
    rows = []
    d = ak.macro_euro_lme_stock()
    for c, i in (("铜-库存", "yf:HG=F"),):
        rows.append(pd.DataFrame(dict(id=i, kind="LME 재고", date=pd.to_datetime(d["日期"]), value=pd.to_numeric(d[c], errors="coerce"))))
    for fn, i in (("macro_cons_gold", "yf:GC=F"), ("macro_cons_silver", "yf:SI=F")):
        d = getattr(ak, fn)()
        rows.append(pd.DataFrame(dict(id=i, kind="ETF 보유량", date=pd.to_datetime(d["日期"]), value=pd.to_numeric(d["总库存"], errors="coerce"))))
    D = pd.concat(rows, ignore_index=True).dropna()
    D.to_parquet(MM / "metals.parquet", index=False)
    return D


def metals_features(idx: pd.DatetimeIndex, cols) -> dict:
    f = MM / "metals.parquet"
    D = pd.read_parquet(f) if f.exists() else metals_fetch()
    out = {"금속 재고 4주 변화 (LME 구리, %)": pd.DataFrame(np.nan, index=idx, columns=cols),
           "귀금속 ETF 보유량 4주 변화 (금·은, %)": pd.DataFrame(np.nan, index=idx, columns=cols)}
    for (i, kind), g in D.groupby(["id", "kind"]):
        s = g.set_index("date").value.sort_index()
        s.index = s.index + pd.Timedelta(days=1)                          # 다음 날 공표 → 하루 늦춤
        w = s.resample("W-FRI").last().reindex(idx).ffill(limit=2)
        key = "금속 재고 4주 변화 (LME 구리, %)" if kind == "LME 재고" else "귀금속 ETF 보유량 4주 변화 (금·은, %)"
        if i in cols:
            out[key][i] = 100 * np.log(w).diff(4)
    return out


# ============================================================================
# 텍스트 ③ EIA STEO 서술문(월간 PDF, 1998~) · ④ 연준 베이지북(연 8회, 1990~) → FinBERT
# ============================================================================
STEO_KW = {"yf:CL=F": ["crude oil", "brent", "wti", "oil price", "opec"], "yf:RB=F": ["gasoline"], "yf:HO=F": ["diesel", "distillate", "heating oil"],
           "yf:NG=F": ["natural gas", "henry hub", "lng"]}
BB_KW = {"에너지": (["oil", "natural gas", "energy", "drilling", "gasoline", "fuel"], ["yf:CL=F", "yf:HO=F", "yf:RB=F", "yf:NG=F"]),
         "금속": (["steel", "metal", "copper", "aluminum", "mining", "precious"], ["yf:HG=F", "yf:SI=F", "yf:PL=F", "yf:GC=F"]),
         "농업": (["agricultur", "crop", "farm", "corn", "soybean", "wheat", "harvest", "cattle", "livestock", "dairy", "cotton"],
                ["yf:ZC=F", "yf:ZW=F", "yf:ZS=F", "yf:ZM=F", "yf:ZL=F", "yf:KC=F", "yf:SB=F", "yf:CT=F", "yf:CC=F", "yf:LE=F", "yf:HE=F"])}


def _sentences(txt: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.;])\s+", re.sub(r"\s+", " ", txt)) if 30 <= len(s.strip()) <= 400]


def steo_text_scores(progress=print) -> pd.DataFrame:
    """STEO PDF 보관본(mmmyy.pdf) → 품목 키워드가 든 문장 FinBERT 평균. 사용 가능일 = steo_features의 avail(빈티지별). 캐시 data/mm/steo_text.parquet."""
    from pypdf import PdfReader
    f = MM / "steo_text.parquet"
    old = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["vintage", "id", "tone", "n"])
    done = set(pd.to_datetime(old.vintage).dt.strftime("%Y-%m")) if len(old) else set()
    sf = pd.read_parquet(ROOT / "data" / "reports" / "steo_features.parquet")[["vintage", "avail"]].drop_duplicates("vintage")
    months = pd.period_range("1998-01", pd.Timestamp.today(), freq="M")
    rows = []
    for m in months:
        if str(m) in done:
            continue
        fn = RAW / f"steo_{m.strftime('%b%y').lower()}.pdf"
        try:
            if not fn.exists():
                r = requests.get(f"https://www.eia.gov/outlooks/steo/archives/{m.strftime('%b%y').lower()}.pdf", headers=UA, timeout=90)
                if r.status_code != 200 or not r.content.startswith(b"%PDF"):
                    continue
                fn.write_bytes(r.content); time.sleep(0.5)
            rd = PdfReader(str(fn))
            txt = " ".join((rd.pages[k].extract_text() or "") for k in range(min(8, len(rd.pages))))
        except Exception as e:
            progress(f"STEO {m} 실패 {e}"); continue
        sents = _sentences(txt)
        for i, kws in STEO_KW.items():
            ss = [s for s in sents if any(k in s.lower() for k in kws)][:60]
            if ss:
                rows.append(dict(vintage=m.to_timestamp(), id=i, tone=float(finbert(ss).mean()), n=len(ss)))
        progress(f"STEO 원문 {m}")
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if rows else old
    if len(df):
        df["tone"] = pd.to_numeric(df.tone, errors="coerce")
        df.to_parquet(f, index=False)
    return df


def beige_scores(start_year: int = 1990, progress=print) -> pd.DataFrame:
    """미니애폴리스 연은 보관본 '전국 요약'(…/beige-book-reports/{연}/{연}-{월}-su) → 에너지·금속·농업 문장 FinBERT.
    발표일은 그달 중 → 보수적으로 그달 말 + 2일부터 사용. 캐시 data/mm/beige.parquet."""
    f = MM / "beige.parquet"
    old = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["month", "group", "tone", "n"])
    done = set(pd.to_datetime(old.month).dt.strftime("%Y-%m")) if len(old) else set()
    tried = MM / "beige_tried.txt"
    miss = set(tried.read_text().split()) if tried.exists() else set()
    rows = []
    for y in range(start_year, pd.Timestamp.today().year + 1):
        for mo in range(1, 13):
            key = f"{y}-{mo:02d}"
            if key in done or key in miss or pd.Timestamp(y, mo, 1) > pd.Timestamp.today():
                continue
            try:
                r = requests.get(f"https://www.minneapolisfed.org/beige-book-reports/{y}/{key}-su", headers=UA, timeout=60)
            except Exception:
                continue
            time.sleep(0.4)
            if r.status_code != 200 or "beige" not in r.text.lower():
                miss.add(key); continue
            body = re.sub(r"<script.*?</script>|<style.*?</style>", " ", r.text, flags=re.S | re.I)
            txt = re.sub(r"<[^>]+>", " ", body)
            sents = _sentences(txt)
            if len(sents) < 10:
                miss.add(key); continue
            for g, (kws, _) in BB_KW.items():
                ss = [s for s in sents if any(k in s.lower() for k in kws)][:60]
                if ss:
                    rows.append(dict(month=pd.Timestamp(y, mo, 1), group=g, tone=float(finbert(ss).mean()), n=len(ss)))
            progress(f"베이지북 {key}")
    tried.write_text(" ".join(sorted(miss)))
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if rows else old
    if len(df):
        df["tone"] = pd.to_numeric(df.tone, errors="coerce")
        df.to_parquet(f, index=False)
    return df


def text_more_features(idx: pd.DatetimeIndex, cols) -> dict:
    out = {}
    f = MM / "steo_text.parquet"
    if f.exists():
        S = pd.read_parquet(f)
        sf = pd.read_parquet(ROOT / "data" / "reports" / "steo_features.parquet")[["vintage", "avail"]].drop_duplicates("vintage")
        sf["ym"] = pd.to_datetime(sf.vintage).dt.to_period("M")
        S["ym"] = pd.to_datetime(S.vintage).dt.to_period("M")
        S = S.merge(sf[["ym", "avail"]], on="ym", how="left")
        S["t"] = pd.to_datetime(S.avail).fillna(pd.to_datetime(S.vintage) + pd.Timedelta(days=14))   # 빈티지 자료 없으면 그달 중순 + 여유
        out["EIA STEO 서술 감성"] = _hold(S.rename(columns={"tone": "v"})[["t", "id", "v"]], idx, cols)
        S = S.sort_values("t"); S["v"] = S.groupby("id").tone.diff()
        out["EIA STEO 서술 감성 변화"] = _hold(S.dropna(subset=["v"])[["t", "id", "v"]], idx, cols)
    f = MM / "beige.parquet"
    if f.exists():
        B = pd.read_parquet(f)
        B["t"] = pd.to_datetime(B.month) + pd.offsets.MonthEnd(0) + pd.Timedelta(days=2)
        ev = [dict(t=r.t, id=i, v=r.tone) for _, r in B.iterrows() for i in BB_KW[r.group][1]]
        out["연준 베이지북 감성 (에너지·금속·농업 문장)"] = _hold(pd.DataFrame(ev), idx, cols)
    return out


# ============================================================================
# 신호 만들기
# ============================================================================
def _hold(ev: pd.DataFrame, idx: pd.DatetimeIndex, cols, max_age: int | None = None) -> pd.DataFrame:
    """이벤트(사용 가능 시각, 품목, 값) → 금요일마다 최신 값 (다음 발표까지 유지, max_age일 넘게 묵으면 비움)."""
    out = pd.DataFrame(np.nan, index=idx, columns=cols)
    for i, g in ev.groupby("id"):
        if i not in out:
            continue
        s = g.sort_values("t").drop_duplicates("t", keep="last").set_index("t")["v"]
        v = s.reindex(s.index.union(idx)).ffill().reindex(idx)
        if max_age:
            last = pd.Series(s.index, index=s.index).reindex(s.index.union(idx)).ffill().reindex(idx)
            v[(idx - last).dt.days > max_age] = np.nan
        out[i] = v
    return out


def _beta_pred(R: pd.DataFrame, x: pd.Series, win: int = 260, lag: int = 4) -> pd.DataFrame:
    """품목별 '4주 수익률 ~ 4주 전 x' 과거 win주 회귀 기울기(t까지 실현된 자료만) × 현재 x = 다음 4주 예상."""
    y = R.rolling(4).sum()
    xl = x.shift(lag)
    cov = y.sub(y.rolling(win, min_periods=104).mean()).mul(xl - xl.rolling(win, min_periods=104).mean(), axis=0).rolling(win, min_periods=104).mean()
    var = xl.rolling(win, min_periods=104).var()
    return cov.div(var, axis=0).mul(x, axis=0)


def features(L: pd.DataFrame, P: pd.DataFrame, use_news: bool = False) -> dict:
    """갈래 → {특성 이름: 주간 × 품목 표}."""
    import alt_ts as AT
    idx, cols = P.index, P.columns
    R = LS._ret(P)
    fam = {"텍스트 (FinBERT)": {}, "리포트 수치": {}, "포지션 (CFTC COT)": {}, "기상": {}, "매크로": {}, "금융 (FRED 추가)": {}, "월간 거시 (FRED)": {}}
    for famn, fn in (("텍스트 (FinBERT)", lambda: text_more_features(idx, cols)), ("포지션 (CFTC COT)", lambda: cot_features(idx, cols)),
                     ("리포트 수치", lambda: eia_features(idx, cols)), ("리포트 수치", lambda: metals_features(idx, cols))):
        try:
            fam[famn].update(fn())
        except Exception as e:
            print(f"{famn} 일부 건너뜀:", e)
    # WASDE 서술
    wt = pd.read_parquet(MM / "wasde_text.parquet") if (MM / "wasde_text.parquet").exists() else pd.DataFrame()
    if len(wt):
        wt["tone"] = pd.to_numeric(wt.tone, errors="coerce")
        ev = []
        for _, r in wt.iterrows():
            t = pd.Timestamp(r.release).tz_convert("America/New_York").tz_localize(None).normalize()   # 12:00 ET 발표 → 그날 금요일 종가부터
            for i in WASDE_SECTIONS.get(r.section, []):
                ev.append(dict(t=t, id=i, v=r.tone, sec=r.section))
        ev = pd.DataFrame(ev).sort_values("t")
        fam["텍스트 (FinBERT)"]["WASDE 서술 감성"] = _hold(ev, idx, cols)
        ev["v"] = ev.groupby("id").v.diff()
        fam["텍스트 (FinBERT)"]["WASDE 서술 감성 변화"] = _hold(ev.dropna(), idx, cols)
    if use_news:
        g = gdelt_scores()
        if len(g):
            T = g.pivot(index="wk", columns="id", values="tone").reindex(idx)
            N = g.pivot(index="wk", columns="id", values="n").reindex(idx).fillna(0)
            m4 = (T.fillna(0) * N).rolling(4, min_periods=1).sum() / N.rolling(4, min_periods=1).sum().replace(0, np.nan)
            fam["텍스트 (FinBERT)"]["뉴스 감성 (4주)"] = m4.reindex(columns=cols)
            fam["텍스트 (FinBERT)"]["뉴스 감성 변화 (4주 vs 직전 12주)"] = (m4 - m4.shift(4).rolling(12, min_periods=4).mean()).reindex(columns=cols)
            fam["텍스트 (FinBERT)"]["뉴스 관심도 (건수 4주 vs 52주)"] = (N.rolling(4).mean() / N.rolling(52, min_periods=13).mean()).reindex(columns=cols)
    # WASDE · STEO 수치
    wf = pd.read_parquet(ROOT / "data" / "reports" / "wasde_features.parquet")
    wf = wf[wf["first"] == 0]
    wf["t"] = pd.to_datetime(wf.release).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    for c, nm in (("d_end_stocks", "WASDE 기말재고 수정 (%)"), ("d_stu", "WASDE 재고율 수정 (%p)"), ("d_production", "WASDE 생산 수정 (%)")):
        fam["리포트 수치"][nm] = _hold(wf.rename(columns={c: "v"})[["t", "id", "v"]], idx, cols)
    try:
        g = nass_ge()
        g["woy"] = g.we.dt.isocalendar().week.astype(int)
        g = g.sort_values("we")
        g["d1"] = g.groupby("crop").ge.diff()
        g["yr"] = g.we.dt.year
        clim = []                                                    # 같은 주차의 직전 5년 평균 (그해 이전 자료만)
        for _, r in g.iterrows():
            past = g[(g.crop == r.crop) & (g.woy == r.woy) & (g.yr < r.yr) & (g.yr >= r.yr - 5)]["ge"]
            clim.append(r["ge"] - past.mean() if len(past) >= 3 else np.nan)
        g["vs5"] = clim
        for c, nm in (("d1", "NASS 작황 우량 비율 주간 변화 (%p)"), ("vs5", "NASS 작황 우량 비율 vs 직전 5년 같은 주 (%p)")):
            ev = [dict(t=r.avail, id=i, v=r[c]) for _, r in g.dropna(subset=[c]).iterrows() for i in NASS_MAP.get(r.crop, [])]
            fam["리포트 수치"][nm] = _hold(pd.DataFrame(ev), idx, cols, max_age=21)
    except Exception as e:
        print("NASS 작황 건너뜀:", e)
    sf = pd.read_parquet(ROOT / "data" / "reports" / "steo_features.parquet")
    ev = [dict(t=pd.Timestamp(r.avail), id=i, v=r.d_fc_3m) for _, r in sf.iterrows() for i in STEO_IDS.get(r.code, [])]
    fam["리포트 수치"]["EIA STEO 3개월 전망 수정 (%)"] = _hold(pd.DataFrame(ev), idx, cols)
    # 기상 · 매크로: 품목별 β × 현재 값
    Wx = weather_weekly(idx)
    if "gistemp" not in Wx and "gtemp_noaa" in Wx:
        Wx["gistemp"] = Wx["gtemp_noaa"]
    for v, nm in (("oni", "엘니뇨 ONI"), ("soi", "남방진동 SOI"), ("gistemp", "지구 기온 편차 (NASA·NOAA)")):
        if v in Wx:
            fam["기상"][nm + " × 품목 민감도"] = _beta_pred(R, Wx[v])
    try:
        fam["기상"].update(openmeteo_features(idx, cols))
    except Exception as e:
        print("Open-Meteo 건너뜀:", e)
    try:
        fin, mon = fred_extra_features(idx, cols, R)
        fam["금융 (FRED 추가)"].update(fin)
        fam["월간 거시 (FRED)"].update(mon)
    except Exception as e:
        print("FRED 추가분 건너뜀:", e)
    Mx = AT.macro_weekly(L).reindex(idx)
    for v, nm in AT.MACRO3.items():
        fam["매크로"][nm + " × 품목 민감도"] = _beta_pred(R, Mx[v])
    return {k: v for k, v in fam.items() if v}


def _xz(S: pd.DataFrame) -> pd.DataFrame:
    return S.sub(S.mean(1), axis=0).div(S.std(1).replace(0, np.nan), axis=0)


def signed(S: pd.DataFrame, P: pd.DataFrame, win: int = 104, min_obs: int = 52, mode: str = "rolling") -> tuple[pd.DataFrame, pd.Series]:
    """품목 간 표준화 → 방향(부호)을 곱함. t에는 t−4에 낸 신호의 결과(IC)까지만 앎.
    rolling = 과거 win주 IC 평균 부호를 매주 갱신 · fixed = 특성이 처음 생긴 뒤 156주 IC 평균 부호로 한 번 정하고 고정(그 전 구간은 비움)."""
    lp = np.log(P)
    fwd = lp.shift(-4) - lp
    Z = _xz(S)
    ic = Z.rank(axis=1).corrwith(fwd.rank(axis=1), axis=1)
    known = ic.shift(4)
    if mode == "rolling":                                               # 최근 IC '관측' win개 기준 (작황처럼 시즌에만 있는 신호도 쓰이도록)
        k = known.dropna()
        sgn = np.sign(k.rolling(win, min_periods=min_obs).mean()).reindex(known.index).ffill()
    else:
        k = known.dropna()
        if len(k) < 156:
            return Z * np.nan, ic
        cut = k.index[155]
        sgn = pd.Series(np.where(known.index > cut, np.sign(k.iloc[:156].mean()), np.nan), index=known.index)
    return Z.mul(sgn, axis=0), ic


def build(L: pd.DataFrame, P: pd.DataFrame, use_news: bool = False, F: dict | None = None) -> dict:
    F = F or features(L, P, use_news)
    out = dict(features=F, ic={}, cover={})
    for mode in ("rolling", "fixed"):
        fam_sig = {}
        for fam, feats in F.items():
            parts = []
            for nm, S in feats.items():
                s, ic = signed(S, P, mode=mode)
                parts.append(s)
                out["ic"][(fam, nm)] = ic; out["cover"][(fam, nm)] = int(S.loc[EVAL:].notna().any(axis=0).sum())
            st = pd.concat(parts, keys=range(len(parts))).groupby(level=1).mean()
            fam_sig[fam] = st.reindex(index=P.index, columns=P.columns)
        out[f"family_{mode}"] = fam_sig
        out[f"multimodal_{mode}"] = pd.concat(fam_sig.values(), keys=range(len(fam_sig))).groupby(level=1).mean().reindex(index=P.index, columns=P.columns)
    return out


# ============================================================================
# 백테스트
# ============================================================================
def backtest(L: pd.DataFrame, cost_bp: float = 5.0, use_news: bool = False, progress=print) -> dict:   # GDELT 뉴스는 수집이 너무 느려 제외 (2026-10-27)
    import ls_extra as LX
    P = LS.weekly(L, LS.BASKET, "2001-01-01"); P = P[[c for c in LS.BASKET if c in P]]
    R = LS._ret(P)
    vol = R.rolling(26, min_periods=13).std()
    B = build(L, P, use_news)
    book = lambda S: LS._book(LS._vol_target(LX.xs_weights(S.where(S.notna()), hold=4, inv_vol=1 / vol), R, 10.0), R, cost_bp)   # noqa: E731
    progress("롱숏 기존 전략 계산")
    X = LX.run_extra(L, cost_bp)
    D0 = X["returns"]
    mvb, tsm, v1 = D0["기준 · MinVar 롱 (⑥)"], D0["기준 · 52주 모멘텀 (④)"], D0["V1 가치 (5년 평균 대비 싼 품목 롱)"]

    def mix(parts):
        m = sum(w * s.reindex(mvb.index).fillna(0) for w, s in parts)
        pv = m.rolling(26, min_periods=13).std().shift(1) * np.sqrt(ANN)
        return m * (10.0 / pv).clip(upper=4.0).fillna(1.0)
    S = {}
    mm = book(B["multimodal_rolling"]); mmf = book(B["multimodal_fixed"])
    for fam, sig in B["family_rolling"].items():
        S[f"갈래 단독 · {fam}"] = book(sig)
    for fam, sig in B["family_fixed"].items():
        S[f"갈래 단독 (고정 부호) · {fam}"] = book(sig)
    S["멀티모달 단독 (모든 갈래 평균)"] = mm
    S["멀티모달 단독 (고정 부호)"] = mmf
    S["기준 · 52주 모멘텀"] = tsm
    S["기준 · MinVar 롱"] = mvb
    S["혼합 · MinVar 70 + 모멘텀 30 (기존)"] = mix([(0.7, mvb), (0.3, tsm)])
    S["혼합 · MinVar 50 + 모멘텀 25 + 가치 25"] = mix([(0.5, mvb), (0.25, tsm), (0.25, v1)])
    S["혼합 · MinVar 50 + 모멘텀 25 + 멀티모달 25"] = mix([(0.5, mvb), (0.25, tsm), (0.25, mm)])
    S["혼합 · MinVar 40 + 모멘텀 20 + 가치 20 + 멀티모달 20"] = mix([(0.4, mvb), (0.2, tsm), (0.2, v1), (0.2, mm)])
    S["혼합 · MinVar 50 + 모멘텀 25 + 멀티모달(고정 부호) 25"] = mix([(0.5, mvb), (0.25, tsm), (0.25, mmf)])
    for fam in B["family_rolling"]:
        S[f"혼합 · MinVar 50 + 모멘텀 25 + {fam} 25"] = mix([(0.5, mvb), (0.25, tsm), (0.25, S[f'갈래 단독 · {fam}'])])
    D = pd.DataFrame(S).loc[EVAL:].fillna(0.0)
    base = D["혼합 · MinVar 70 + 모멘텀 30 (기존)"]
    rows = []
    for k in D:
        r = D[k]; m = LS.metrics(r); d = r - base
        row = {"전략": k, **m, "Sharpe 2007–2016": LS.metrics(r[:"2016-12-31"])["Sharpe"], "Sharpe 2017–2021-10": LS.metrics(r["2017-01-01":RECENT])["Sharpe"],
               "Sharpe 최근 5년 (2021-11~)": LS.metrics(r[RECENT:])["Sharpe"],
               "t값": m["Sharpe"] * np.sqrt(len(r) / ANN), "기존 70/30과 상관": float(r.corr(base)),
               "기존 대비 연 수익 차 (%p)": float(d.mean() * ANN), "차이 t값": float(d.mean() / d.std() * np.sqrt(len(d))) if d.std() > 0 else np.nan}
        rows.append(row)
    ic_rows = []
    for (fam, nm), ic in B["ic"].items():
        e = ic.loc[EVAL:].dropna()
        ic_rows.append({"갈래": fam, "특성": nm, "적용 품목 수": B["cover"][(fam, nm)], "평가 기간 IC 평균": float(e.mean()) if len(e) else np.nan,
                        "IC t값": float(e.mean() / e.std() * np.sqrt(len(e) / 4)) if len(e) > 8 and e.std() > 0 else np.nan, "주 수": len(e)})
    return dict(table=pd.DataFrame(rows), returns=D, ic=pd.DataFrame(ic_rows), build=B)
