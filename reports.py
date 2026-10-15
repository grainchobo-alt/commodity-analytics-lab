"""공식 기관 리포트(전망) 수집·파싱 — Time-MMD 원칙: 발행 시각이 확실한 원문만, 기관이 쓴 숫자 그대로, 발표 시점에 고정.

- USDA WASDE (월간, 발표 12:00 ET): 미국 수급표(옥수수·대두·대두박·대두유·밀·면화)의 '이번 달 전망' 값 →
  같은 연산연도 값의 전월 대비 수정폭(%) = 기관 전망 서프라이즈의 대리변수.
- EIA STEO (월간 빈티지 xlsx, 'Forecast date' 명시): WTI·Brent·Henry Hub 향후 12개월 가격 전망, 미국 원유 생산 전망.
  공개는 forecast date 후 화요일 전후 → 보수적으로 forecast date + 5영업일에 이용 가능 처리.

모든 원문은 data/reports/raw/ 에 캐시(재다운로드 없음), 파싱 결과는 data/reports/*.parquet.
"""
from __future__ import annotations
import re, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).parent
RDIR = ROOT / "data" / "reports"; RAW = RDIR / "raw"
RAW.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"}
ESMIS = "https://esmis.nal.usda.gov"
WASDE_LIST = ESMIS + "/publication/world-agricultural-supply-and-demand-estimates?page={p}"


def _get(url: str, path: Path | None = None, tries: int = 3) -> bytes:
    if path is not None and path.exists() and path.stat().st_size > 1000:
        return path.read_bytes()
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 404:                         # 없는 파일은 재시도하지 않음
                raise FileNotFoundError(url)
            r.raise_for_status()
            if path is not None:
                path.write_bytes(r.content)
            return r.content
        except FileNotFoundError:
            raise
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


# ============================================================================
# USDA WASDE
# ============================================================================
# (표 제목 일부, 섹션 머리말 or None, 우리 선물 id)
WASDE_TABLES = {
    "corn":    ("U.S. Feed Grain and Corn Supply and Use", "CORN", "yf:ZC=F"),
    "soybean": ("U.S. Soybeans and Products Supply and Use", "SOYBEANS", "yf:ZS=F"),
    "soyoil":  ("U.S. Soybeans and Products Supply and Use", "SOYBEAN OIL", "yf:ZL=F"),
    "soymeal": ("U.S. Soybeans and Products Supply and Use", "SOYBEAN MEAL", "yf:ZM=F"),
    "wheat":   ("U.S. Wheat Supply and Use", None, "yf:ZW=F"),
    "cotton":  ("U.S. Cotton Supply and Use", None, "yf:CT=F"),
}
SECTIONS = {"FEED GRAINS", "CORN", "SORGHUM", "BARLEY", "OATS", "SOYBEANS", "SOYBEAN OIL", "SOYBEAN MEAL"}
ITEM_KEYS = {  # 정규화 이름 ← 원문 행 머리(소문자, 공백 제거 후 접두 일치) — 1997~2026 표기 모두
    "planted": ("areaplanted",), "harvested": ("areaharvested",), "yield": ("yieldperharvestedacre",),
    "beg_stocks": ("beginningstocks",), "production": ("production",), "imports": ("imports",),
    "supply": ("supply,total",), "exports": ("exports",), "use": ("use,total",), "end_stocks": ("endingstocks",),
    "price": ("avg.", "averageprice", "avgfarmprice"), "crush": ("crushings",), "domestic": ("domestic",),
    "feed": ("feedandresidual",),
}
CONT_LABELS = {"acre": "Yield per harvested", "planted": "Area", "harvested": "Area"}


def wasde_index(pages=range(0, 4)) -> pd.DataFrame:
    """ESMIS 목록 → 발간본별 발표 시각(UTC)·원문 URL. 형식 우선순위: txt > xls (2010-05~2016-10은 xls만 존재).
    같은 발간본에 원본(ORIG)과 정정본(R1)이 있으면 '원본' — 발표 시각에 시장이 실제로 본 문서."""
    rows = []
    for p in pages:
        html = _get(WASDE_LIST.format(p=p)).decode("utf-8", "ignore")
        last = None
        for m in re.finditer(r'datetime="([^"]+)"|(/sites/default/release-files/[^"]+?\.(?:txt|xls))"', html, flags=re.I):
            if m.group(1):
                last = m.group(1)
            elif last:
                rows.append(dict(release=pd.Timestamp(last), url=ESMIS + m.group(2)))
    df = pd.DataFrame(rows).drop_duplicates()

    def rank(u):
        f = u.rsplit("/", 1)[-1].lower()
        return (0 if f.endswith(".txt") else 1) * 10 + (0 if f.startswith("orig") else (2 if re.match(r"r\d", f) else 1))
    df["rk"] = df.url.map(rank)
    df = df.sort_values(["release", "rk"]).drop_duplicates("release").sort_values("release").reset_index(drop=True)
    df["fmt"] = df.url.str.rsplit(".", n=1).str[-1].str.lower()
    df["fname"] = df.release.dt.strftime("wasde_%Y%m%d.") + df.fmt
    return df.drop(columns="rk")


def _xls_to_text(raw: bytes) -> str:
    """WASDE xls(쪽별 시트) → 텍스트판과 같은 '항목  값  값 ...' 줄 형식."""
    import xlrd
    wb = xlrd.open_workbook(file_contents=raw)
    out = []
    for sh in wb.sheets():
        out.append(f"WASDE - sheet {sh.name}")
        for r in range(sh.nrows):
            cells = [sh.cell_value(r, c) for c in range(sh.ncols)]
            strs = [str(c).strip() for c in cells if isinstance(c, str) and str(c).strip()]
            if any(re.search(r"\d{4}/\d{2}", x) for x in strs):            # 연산연도 머리줄: 섹션명 + 연도들
                lab = strs[0] if not re.search(r"\d{4}/\d{2}", strs[0]) else ""
                if lab:
                    out.append(lab)
                out.append("    ".join(x for x in strs if re.search(r"\d{4}/\d{2}", x)))
                continue
            lab, vals = None, []
            for c in cells:
                if isinstance(c, float):
                    vals.append(f"{c:g}")
                elif isinstance(c, str) and c.strip():
                    t = RANGE.sub(_mid, c.strip().rstrip("*").strip())   # 가격 범위 '6.25 - 7.55' → 중간값
                    if re.fullmatch(r"-?[\d,]*\.?\d+", t) or t == "NA":
                        vals.append(t)
                    elif lab is None:
                        lab = c.strip()
            if lab and vals:
                out.append(f"{lab}    " + "    ".join(vals))
            elif lab:
                out.append(lab)
    return "\n".join(out)


# 가격 범위: '60.00-80.00', '6.75- 8.25', '6.25 - 7.55' → 중간값. 음수 '  -0.17'(앞 공백·뒤 붙음)은 제외.
RANGE = re.compile(r"(\d[\d,]*\.\d+)(?:-\s*|\s+-\s+| -)(\d[\d,]*\.\d+)")    # ' -'(공백 1칸): '12.00 -13.50' (열 구분은 2칸 이상)


def _mid(m):
    return f"{(float(m.group(1).replace(',', '')) + float(m.group(2).replace(',', ''))) / 2:g}"


def _normalize(text: str) -> str:
    """텍스트판 정규화: 'label : vals' 콜론 형식(1997~2009), 두 줄 항목명, 가격 범위(→ 중간값), 'U. S.' 표기."""
    text = text.replace("U. S.", "U.S.")
    rng, mid = RANGE, _mid
    lines, out, last_lab = text.splitlines(), [], ""
    k = 0
    while k < len(lines):
        l = lines[k]
        m = re.match(r"^([^:=]{0,40}?)\s*:(.*)$", l)
        if m and not re.search(r"\d{4}/\d{2}", l):
            lab, rest = m.group(1).strip().rstrip(":").strip(), m.group(2)
            rest = re.sub(r"_?\d/|\*+", " ", rest)                      # 각주 '_3/', '*', '**' 제거
            toks = rest.split()
            if toks and toks[-1].endswith("-") and k + 1 < len(lines):    # 두 줄에 걸친 범위 '13.50-' / '16.50'
                m2 = re.match(r"^\s*:(.*)$", lines[k + 1])
                if m2:
                    nxt = m2.group(1).split()
                    j = 0
                    for ti, t in enumerate(toks):
                        if t.endswith("-") and j < len(nxt):
                            try:
                                toks[ti] = f"{(float(t[:-1].replace(',', '')) + float(nxt[j].replace(',', ''))) / 2:g}"
                            except ValueError:
                                pass
                            j += 1
                    k += 1
            rest = rng.sub(mid, " ".join(toks))
            nums = re.findall(r"-?[\d,]*\.?\d+", rest)
            first = lab.lower().split(" ")[0] if lab else ""
            if first in CONT_LABELS and nums and last_lab.lower().startswith(CONT_LABELS[first].lower()):
                lab = f"{last_lab} {lab}"
            if nums and re.fullmatch(r"[\s\-\d,.]*", rest):
                out.append(f"{lab}    " + "    ".join(nums))
            else:
                out.append(lab)
                if lab:
                    last_lab = lab
        else:
            l = rng.sub(mid, l)
            m3 = re.match(r"^\s*(Acre|acre)\b\s+(.*)$", l)                # 'Yield per Harvested' / 'Acre  ...' 두 줄
            if m3 and out and out[-1].strip().lower().startswith("yield per harvested") and not re.search(r"\d", out[-1]):
                out[-1] = f"{out[-1].strip()} Acre    {m3.group(2)}"
            else:
                out.append(l)
        k += 1
    return "\n".join(out)


_NUM = r"(NA|-?[\d,]*\.?\d+)\*{0,3}"


def _parse_row(line: str):
    """'Ending Stocks   325   315   310   315' → ('Ending Stocks', [325, 315, 310, 315])."""
    m = re.match(r"^\s*(.*?\S)\s{2,}((?:" + _NUM + r"\s*)+)$", line.rstrip())
    if not m:
        return None
    vals = re.findall(_NUM, m.group(2))
    out = [np.nan if v == "NA" else float(v.replace(",", "")) for v in vals]
    return m.group(1), out


def parse_wasde_table(text: str, title: str, section: str | None) -> pd.DataFrame | None:
    """한 표(제목 기준)의 섹션 행들 → DataFrame(item × [prev_year, est, proj_prev, proj]) + 연산연도 라벨(오름차순)."""
    i = text.find(title)
    if i < 0:
        return None
    blk = text[i:]
    j = re.search(r"WASDE\s*-\s*(\d|sheet)", blk[10:])                # 다음 쪽 머리말까지
    blk = blk[: j.start() + 10 if j else None]
    lines = blk.splitlines()
    y0 = sorted({int(y) for y in re.findall(r"(\d{4})/\d{2,3}", "\n".join(lines[:12]))})   # '2011/012' 표기도 허용
    years = [f"{y}/{(y + 1) % 100:02d}" for y in y0]
    cur = None if section is None else ""
    rows = {}
    for l in lines:
        s = l.strip().rstrip(":").strip()
        if s.upper() in SECTIONS:
            cur = s.upper(); continue
        if section is not None and cur != section:
            continue
        pr = _parse_row(l)
        if not pr or len(pr[1]) < 3:
            continue
        key = re.sub(r"\s+", "", pr[0].lower())
        for k, prefs in ITEM_KEYS.items():
            if any(key.startswith(pf) for pf in prefs) and k not in rows:
                rows[k] = pr[1][-4:] if len(pr[1]) >= 4 else [np.nan] * (4 - len(pr[1])) + pr[1]
                break
    if not rows:
        return None
    df = pd.DataFrame(rows, index=["prev_year", "est", "proj_prev", "proj"]).T
    df.attrs["years"] = years
    return df


def wasde_text(r) -> str:
    raw = _get(r.url, RAW / r.fname)
    return _normalize(raw.decode("latin-1")) if r.fmt == "txt" else _xls_to_text(raw)


def wasde_collect(pages=range(0, 4), progress=None) -> pd.DataFrame:
    """발간본별 · 품목별 · 항목별 값 (long). 열: release, commodity, item, my(연산연도), value, kind(est/proj), fmt."""
    idx = wasde_index(pages)
    out = []
    for k, r in idx.iterrows():
        try:
            txt = wasde_text(r)
        except Exception as e:
            print(f"\n  ! {r.release:%Y-%m-%d} 읽기 실패: {e}")
            continue
        for com, (title, sec, _) in WASDE_TABLES.items():
            t = parse_wasde_table(txt, title, sec)
            if t is None or len(t.attrs["years"]) < 2:
                continue
            ys = t.attrs["years"]                                       # 오름차순: [..., 금년 Est., 신년 Proj.]
            for item, v in t.iterrows():
                out.append(dict(release=r.release, commodity=com, item=item, my=ys[-2], kind="est", value=v["est"], fmt=r.fmt))
                out.append(dict(release=r.release, commodity=com, item=item, my=ys[-1], kind="proj", value=v["proj"], fmt=r.fmt))
        if progress:
            progress(k + 1, len(idx))
    df = pd.DataFrame(out).dropna(subset=["value"])
    df.to_parquet(RDIR / "wasde_long.parquet")
    return df


def wasde_features(long: pd.DataFrame | None = None) -> pd.DataFrame:
    """발간본·품목별 서프라이즈 특성 (신규 작황 = proj 연산연도 기준).
    d_<item> = (이번 값 / 직전 발간본의 같은 연산연도 값 − 1)·100 [%], stu = 기말재고/총사용, d_stu = 변화(%p).
    직전 발간본에 같은 연산연도가 없으면(5월 신규 작황 첫 전망) NaN → 'first'=1."""
    L = long if long is not None else pd.read_parquet(RDIR / "wasde_long.parquet")
    P = L[L.kind == "proj"].pivot_table(index=["commodity", "release", "my"], columns="item", values="value").reset_index()
    rows = []
    for com, g in P.groupby("commodity"):
        g = g.sort_values("release").reset_index(drop=True)
        for i, r in g.iterrows():
            prev = g[(g.release < r.release) & (g.my == r.my)].tail(1)
            d = dict(release=r.release, commodity=com, my=r.my, first=int(prev.empty))
            for it in ["production", "yield", "end_stocks", "use", "exports", "price"]:
                if it in r and pd.notna(r[it]):
                    d[it] = r[it]
                    pv = prev[it].iloc[0] if (not prev.empty and it in prev) else np.nan
                    d["d_" + it] = (r[it] / pv - 1) * 100 if pd.notna(pv) and pv != 0 else np.nan
            if pd.notna(r.get("end_stocks")) and pd.notna(r.get("use")) and r["use"]:
                d["stu"] = r["end_stocks"] / r["use"] * 100
                if not prev.empty and pd.notna(prev["end_stocks"].iloc[0]) and prev["use"].iloc[0]:
                    d["d_stu"] = d["stu"] - prev["end_stocks"].iloc[0] / prev["use"].iloc[0] * 100
            rows.append(d)
    F = pd.DataFrame(rows)
    F["id"] = F.commodity.map({k: v[2] for k, v in WASDE_TABLES.items()})
    F.to_parquet(RDIR / "wasde_features.parquet")
    return F


# ============================================================================
# EIA STEO (월간 빈티지)
# ============================================================================
STEO_SERIES = {"WTIPUUS": "yf:CL=F", "BREPUUS": "yf:BZ=F", "NGHHUUS": "yf:NG=F", "COPRPUS": None}
STEO_SHEETS = {"WTIPUUS": "2tab", "BREPUUS": "2tab", "NGHHUUS": "2tab", "COPRPUS": "4atab"}
MON = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def _sheet_rows(path: Path, sheet: str) -> list[tuple]:
    """xlsx(2015~)·xls(~2014) 공통 행 목록."""
    if path.suffix == ".xls":
        import xlrd
        sh = xlrd.open_workbook(str(path)).sheet_by_name(sheet)
        return [tuple(sh.cell_value(r, c) for c in range(sh.ncols)) for r in range(sh.nrows)]
    import openpyxl
    return list(openpyxl.load_workbook(path, read_only=True, data_only=True)[sheet].iter_rows(values_only=True))


def _steo_sheet(path: Path, sheet: str, codes: list[str]):
    """→ (전망일 or None, {code: 월별 Series}). 'Forecast date:' 행이 없는 구형(~2014)은 전망일 None."""
    rows = _sheet_rows(path, sheet)
    fdate = None
    yrow = next((r for r in rows if r and r[0] == "Forecast date:"), None)
    if yrow is not None:
        mrow = rows[rows.index(yrow) + 1]
        m = re.search(r"([A-Za-z]{3})[A-Za-z]*\.?\s+(\d{1,2}),?\s+(\d{4})", str(mrow[0]))   # 원문 오타 허용 (예: 'Augutst')
        if m:
            fdate = pd.Timestamp(year=int(m.group(3)), month=MON.index(m.group(1).lower()) + 1, day=int(m.group(2)))
    else:                                                                   # 구형: 3번째 열에 시작 연도가 있는 행
        yrow = next(r for r in rows[:10] if len(r) > 2 and isinstance(r[2], (int, float)) and 1990 < r[2] < 2100)
        mrow = rows[rows.index(yrow) + 1]
    year, cols = None, []
    for j, (y, m) in enumerate(zip(yrow, mrow)):
        if j < 2:
            continue
        if isinstance(y, (int, float)) and y:
            year = int(y)
        if isinstance(m, str) and m[:3].lower() in MON and year:
            cols.append((j, pd.Timestamp(year=year, month=MON.index(m[:3].lower()) + 1, day=1)))
    out = {}
    for r in rows:
        if r and r[0] in codes:
            out[r[0]] = pd.Series({d: r[j] for j, d in cols if isinstance(r[j], (int, float))}, dtype=float)
    return fdate, out


def steo_collect(start="2023-06", end=None, progress=None) -> pd.DataFrame:
    """월간 빈티지별 경로 (long): vintage(전망일), avail(이용 가능일), code, month, value.
    이용 가능일: 전망일이 있으면 + 5영업일, 없는 구형(~2014)은 보수적으로 해당 월 15일 (STEO 발간은 매달 6~13일)."""
    from concurrent.futures import ThreadPoolExecutor
    end = pd.Timestamp(end or pd.Timestamp.today()).to_period("M")
    months = pd.period_range(start, end, freq="M")
    exts = lambda p: (".xls", ".xlsx") if p.year <= 2014 else (".xlsx", ".xls")   # 2014년 이전은 구형 xls

    def prefetch(p):                                                    # EIA 서버가 느려 (파일당 ~10초) 병렬 선다운로드
        stem = f"{MON[p.month - 1]}{p.year % 100:02d}_base"
        for ext in exts(p):
            try:
                _get(f"https://www.eia.gov/outlooks/steo/archives/{stem}{ext}", RAW / f"steo_{p.year}{p.month:02d}{ext}")
                return
            except Exception:
                continue
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(prefetch, months))
    out = []
    for k, p in enumerate(months):
        stem = f"{MON[p.month - 1]}{p.year % 100:02d}_base"
        path = None
        for ext in exts(p):
            if (RAW / f"steo_{p.year}{p.month:02d}{ext}").exists():
                path = RAW / f"steo_{p.year}{p.month:02d}{ext}"; break
        for ext in (() if path else exts(p)):
            cand = RAW / f"steo_{p.year}{p.month:02d}{ext}"
            try:
                _get(f"https://www.eia.gov/outlooks/steo/archives/{stem}{ext}", cand)
                path = cand; break
            except Exception:
                continue
        if path is None and p == end:                                   # 가장 최근 호는 아카이브에 없을 수 있음
            try:
                path = RAW / f"steo_{p.year}{p.month:02d}.xlsx"
                _get("https://www.eia.gov/outlooks/steo/xls/STEO_m.xlsx", path)
            except Exception:
                path = None
        if path is None:
            continue
        try:
            for sheet in set(STEO_SHEETS.values()):
                codes = [c for c, s in STEO_SHEETS.items() if s == sheet]
                fdate, ser = _steo_sheet(path, sheet, codes)
                exact = fdate is not None
                if not exact:
                    fdate = p.to_timestamp()
                avail = fdate + pd.offsets.BDay(5) if exact else (p.to_timestamp() + pd.Timedelta(days=14))
                for c, s in ser.items():
                    for d, v in s.items():
                        out.append(dict(vintage=fdate, avail=avail, exact_date=exact, code=c, month=d, value=v))
        except Exception as e:
            print(f"\n  ! STEO {p} 읽기 실패: {e}")
        if progress:
            progress(k + 1, len(months))
    df = pd.DataFrame(out).drop_duplicates(["vintage", "code", "month"])
    df.to_parquet(RDIR / "steo_long.parquet")
    return df


def steo_features(long: pd.DataFrame | None = None, horizon: int = 12) -> pd.DataFrame:
    """빈티지별: 향후 N개월 평균 전망(fc), 직전 빈티지 대비 수정(d_fc, %), 다음 달·3개월 전망."""
    L = long if long is not None else pd.read_parquet(RDIR / "steo_long.parquet")
    rows = []
    for (v, c), g in L.groupby(["vintage", "code"]):
        s = g.set_index("month").value.sort_index()
        m0 = v.to_period("M").to_timestamp()
        fut = s[s.index > m0]
        rows.append(dict(vintage=v, avail=g.avail.iloc[0], code=c, fc_1m=fut.iloc[:1].mean(), fc_3m=fut.iloc[:3].mean(),
                         fc_12m=fut.iloc[:horizon].mean(), cur_m=s.get(m0, np.nan)))
    F = pd.DataFrame(rows).sort_values(["code", "vintage"])
    for col in ["fc_3m", "fc_12m"]:
        F["d_" + col] = F.groupby("code")[col].pct_change() * 100
    F["id"] = F.code.map(STEO_SERIES)
    F.to_parquet(RDIR / "steo_features.parquet")
    return F


# ============================================================================
# 평가: 기관 숫자가 가격 데이터에 '추가' 정보를 주는가
# ============================================================================
STEO_SPOT = {"WTIPUUS": "fred:DCOILWTICO", "BREPUUS": "fred:DCOILBRENTEU", "NGHHUUS": "fred:DHHNGSP"}


def _ols_cluster(y, X, groups):
    import statsmodels.api as sm
    return sm.OLS(np.asarray(y, float), sm.add_constant(X), missing="drop").fit(cov_type="cluster", cov_kwds={"groups": groups})


def _ols_hac(y, X, lags):
    import statsmodels.api as sm
    return sm.OLS(y, sm.add_constant(X), missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def _et_date(ts) -> pd.Timestamp:
    return pd.Timestamp(pd.Timestamp(ts).tz_convert("America/New_York").date())


def wasde_events(F: pd.DataFrame, levels: pd.DataFrame, drift=(5, 20)) -> pd.DataFrame:
    """발간본 × 품목 이벤트 표. r0 = 발표일 종가/전일 종가 (발표 12:00 ET는 CBOT 장중),
    z0 = r0 / 직전 60일 일간 변동성, drift_k = 발표일 종가 → k거래일 후 (발표 반영 '이후' 구간만)."""
    rows = []
    for _, e in F.iterrows():
        if e.id not in levels:
            continue
        p = levels[e.id].dropna()
        d = _et_date(e.release)
        if d not in p.index:
            continue
        t = p.index.get_loc(d)
        if t < 61:
            continue
        lr = np.log(p.iloc[t - 61:t]).diff()
        row = dict(r0=np.log(p.iloc[t] / p.iloc[t - 1]), sig=lr.std(), date=d, id=e.id, commodity=e.commodity)
        for k in drift:
            row[f"drift_{k}"] = np.log(p.iloc[t + k] / p.iloc[t]) if t + k < len(p) else np.nan
        for c in ["d_end_stocks", "d_stu", "d_production", "d_yield", "d_price", "stu", "first"]:
            row[c] = e.get(c, np.nan)
        rows.append(row)
    E = pd.DataFrame(rows)
    E["z0"] = E.r0 / E.sig
    return E


def event_study(E: pd.DataFrame, x: str = "d_stu") -> dict:
    """발표일 반응(z0)을 수정폭에 회귀 (발간일 군집 SE) + 드리프트 회귀 + 부호 일치율."""
    D = E[(E["first"] == 0)].dropna(subset=[x, "z0"]).copy()
    D["xs"] = D.groupby("commodity")[x].transform(lambda v: v / (v.std() or 1))   # 품목별 수정폭 표준편차 단위
    res = {"n": len(D), "n_release": D.date.nunique()}
    m = _ols_cluster(D["z0"], D[["xs"]], D["date"])
    res["day0"] = dict(beta=m.params["xs"], t=m.tvalues["xs"], r2=m.rsquared)
    nz = D[D[x] != 0]
    res["sign_hit"] = float((np.sign(nz.r0) == -np.sign(nz[x])).mean()) if len(nz) else np.nan
    res["n_nonzero"] = len(nz)
    for k in [c for c in D.columns if c.startswith("drift_")]:
        Dk = D.dropna(subset=[k])
        yk = Dk[k] / (Dk["sig"] * np.sqrt(int(k.split("_")[1])))
        mk = _ols_cluster(yk, Dk[["xs"]], Dk["date"])
        res[k] = dict(beta=mk.params["xs"], t=mk.tvalues["xs"], n=len(Dk))
    by = {}
    for c, g in D.groupby("commodity"):
        if len(g) >= 8 and g["xs"].std() > 0:
            mc = _ols_hac(g["z0"], g[["xs"]], 1)
            by[c] = dict(beta=mc.params["xs"], t=mc.tvalues["xs"], n=len(g), corr=np.corrcoef(g["z0"], g["xs"])[0, 1])
    res["by_commodity"] = pd.DataFrame(by).T
    return res


def steo_accuracy(G: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
    """EIA 1·3개월 전망 vs 무작위보행(발표 시점 직전 5일 현물 평균). 실제 = 해당 월 현물 평균 (완료된 월만)."""
    out = {}
    for code, spot in STEO_SPOT.items():
        if spot not in levels:
            continue
        s = levels[spot].dropna(); mavg = s.resample("MS").mean()
        last_full = s.index.max().to_period("M").to_timestamp() - pd.offsets.MonthBegin(1)
        e = {"f1": [], "r1": [], "f3": [], "r3": []}
        for _, r in G[G.code == code].iterrows():
            rw = s[s.index <= r.vintage].tail(5).mean()
            m1 = r.vintage.to_period("M").to_timestamp() + pd.offsets.MonthBegin(1)
            m3 = [m1 + pd.offsets.MonthBegin(k) for k in range(3)]
            if m1 <= last_full:
                a1 = mavg[m1]; e["f1"].append(np.log(r.fc_1m / a1)); e["r1"].append(np.log(rw / a1))
            if m3[-1] <= last_full:
                a3 = mavg[m3].mean(); e["f3"].append(np.log(r.fc_3m / a3)); e["r3"].append(np.log(rw / a3))
        e = {k: np.array(v) for k, v in e.items()}
        out[code] = {"1개월 RMSE(로그): EIA": np.sqrt(np.mean(e["f1"] ** 2)), "1개월 RMSE: 무작위보행": np.sqrt(np.mean(e["r1"] ** 2)),
                     "3개월 RMSE: EIA": np.sqrt(np.mean(e["f3"] ** 2)), "3개월 RMSE: 무작위보행": np.sqrt(np.mean(e["r3"] ** 2)),
                     "1개월 EIA 승률": float(np.mean(np.abs(e["f1"]) < np.abs(e["r1"]))),
                     "EIA 평균 편향(로그)": float(np.mean(e["f1"])), "평가 빈티지": len(e["f1"])}
    return pd.DataFrame(out).T


def steo_signal(G: pd.DataFrame, levels_adj: pd.DataFrame, levels_raw: pd.DataFrame) -> dict:
    """EIA 전망이 선물 가격보다 높으면 이후 선물이 오르는가?
    x1 = log(1개월 전망 / 진입 시 선물), x3 = 3개월판, d12 = 12개월 평균 전망 수정(%)
    진입 = 이용가능일 '다음' 거래일 종가, y = k거래일 후까지 롤보정 로그수익. 겹침은 HAC."""
    out = {}
    for code, fid in STEO_SERIES.items():
        if fid is None or fid not in levels_adj:
            continue
        pa, pr = levels_adj[fid].dropna(), levels_raw[fid].dropna()
        rows = []
        for _, r in G[G.code == code].iterrows():
            ix = pa.index.searchsorted(r.avail, side="right")
            if ix >= len(pa):
                continue
            t0 = pa.index[ix]; f_raw = pr[pr.index <= t0].iloc[-1]
            row = dict(x1=np.log(r.fc_1m / f_raw), x3=np.log(r.fc_3m / f_raw), d12=r.d_fc_12m)
            for k in (21, 63):
                row[f"y{k}"] = np.log(pa.iloc[ix + k] / pa.iloc[ix]) if ix + k < len(pa) else np.nan
            rows.append(row)
        D = pd.DataFrame(rows)
        res = {}
        for x, k in [("x1", 21), ("x3", 63), ("d12", 21)]:
            Dk = D.dropna(subset=[x, f"y{k}"])
            if len(Dk) > 8:
                m = _ols_hac(Dk[f"y{k}"], Dk[[x]], max(1, k // 21))
                res[f"{x}→{k}일"] = dict(beta=m.params[x], t=m.tvalues[x], corr=np.corrcoef(Dk[x], Dk[f"y{k}"])[0, 1], n=len(Dk))
        out[code] = res
    return out


def monthly_panel(F: pd.DataFrame, levels: pd.DataFrame, min_train: int = 12, lam: float = 1.0) -> dict:
    """WASDE 발간본 간 보유(발표일 종가 → 다음 발표일 종가)의 변동성 표준화 수익을 확장창 OOS로 예측:
    ① 가격 특성만 (모멘텀 1·12개월, 변동성) vs ② 가격 + WASDE 수정폭. Ridge(작은 표본) + Clark-West 검정.
    학습 표본 = 목표 구간이 이미 끝난 발간본 (rel ≤ k−1) → 미래정보 없음."""
    from scipy import stats
    rels = sorted(F.release.unique())
    rows = []
    for _, e in F.iterrows():
        if e.id not in levels:
            continue
        p = levels[e.id].dropna()
        k = rels.index(e.release)
        if k + 1 >= len(rels):
            continue
        d, d1 = _et_date(e.release), _et_date(rels[k + 1])
        if d not in p.index or d1 not in p.index:
            continue
        t, t1 = p.index.get_loc(d), p.index.get_loc(d1)
        if t < 260:
            continue
        lr = np.log(p).diff()
        vol = lr.iloc[t - 60:t + 1].std()
        rows.append(dict(rel=k, commodity=e.commodity, y=np.log(p.iloc[t1] / p.iloc[t]) / (vol * np.sqrt(t1 - t)),
                         mom1=np.log(p.iloc[t] / p.iloc[t - 21]), mom12=np.log(p.iloc[t - 21] / p.iloc[t - 252]), vol=vol,
                         d_stu=e.get("d_stu"), d_prod=e.get("d_production"), d_price=e.get("d_price"), first=e["first"]))
    D = pd.DataFrame(rows)
    for c in ["d_stu", "d_prod", "d_price"]:
        D[c] = D[c].where(D["first"] == 0, 0.0).fillna(0.0)          # 첫 전망(5월 신규 작황) = 수정 없음
    base = ["mom1", "mom12", "vol"]; full = base + ["d_stu", "d_prod", "d_price"]
    P = {"base": [], "full": []}; ys = []
    for k in sorted(D.rel.unique()):
        tr, te = D[D.rel <= k - 1], D[D.rel == k]
        if tr.rel.nunique() < min_train or te.empty:
            continue
        for nm, cols in [("base", base), ("full", full)]:
            mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
            Xtr = np.column_stack([np.ones(len(tr)), ((tr[cols] - mu) / sd).values])
            Pm = np.eye(Xtr.shape[1]) * lam * len(tr) * 0.1; Pm[0, 0] = 0
            b = np.linalg.solve(Xtr.T @ Xtr + Pm, Xtr.T @ tr["y"].values)
            P[nm].extend(np.column_stack([np.ones(len(te)), ((te[cols] - mu) / sd).values]) @ b)
        ys.extend(te["y"].values)
    y = np.array(ys); P = {k: np.array(v) for k, v in P.items()}
    oos = lambda pr: 1 - np.sum((y - pr) ** 2) / np.sum(y ** 2)
    f = (y - P["base"]) ** 2 - ((y - P["full"]) ** 2 - (P["base"] - P["full"]) ** 2)     # Clark-West (2007)
    cw_t = f.mean() / (f.std(ddof=1) / np.sqrt(len(f)))
    sr = lambda pr: np.mean(np.sign(pr) * y) / np.std(np.sign(pr) * y) * np.sqrt(12)
    Z = (D[full] - D[full].mean()) / D[full].std()
    ins = _ols_cluster(D["y"], Z, D["rel"])
    return dict(n_obs=len(D), n_oos=len(y), oos_r2={k: oos(v) for k, v in P.items()},
                clark_west_t=cw_t, clark_west_p=1 - stats.norm.cdf(cw_t),
                sign_sharpe={k: sr(v) for k, v in P.items()},
                insample=pd.DataFrame({"계수(표준화)": ins.params, "t값(발간본 군집)": ins.tvalues}))


if __name__ == "__main__":
    pr = lambda i, n: print(f"\r  {i}/{n}", end="", flush=True)
    import sys
    full = "--full" in sys.argv                                          # 1997~ 전체 (기본: 최근 3년)
    print("WASDE 수집"); L = wasde_collect(range(0, 36) if full else range(0, 4), progress=pr); print(); F = wasde_features(L)
    print(F.groupby("commodity").size().to_string())
    print("STEO 수집"); S = steo_collect("2010-01" if full else "2023-06", progress=pr); print(); G = steo_features(S)
    print(G.groupby("code").size().to_string())
