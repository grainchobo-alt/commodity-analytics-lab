"""옵션 가격 모형 · 내재변동성 · 옵션 시세 불러오기 (옵션 랩 options_lab.py 전용, 기존 대시보드와 독립).

- 블랙숄즈(BS): 현물·ETF 옵션.  Black-76: 선물옵션(기초자산 = 선물가격 F, 보유비용이 선물가격에 이미 반영).
- 시세 원천은 SOURCES에 등록: yfinance(공개 ETF 옵션) 또는 CSV(직접 구한 자료, 표준 열 이름 CSV_COLS).
  새 원자재 추가 = SOURCES에 한 줄 + (CSV면) 표준 열로 파일 저장 → 나머지 계산·그림은 그대로 동작.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
SNAP_DIR = ROOT / "data" / "options"                      # 매일 받은 시세 저장(누적하면 과거 곡선 자료가 됨, git 제외)

# 등록 원천: 이름 → 설정. kind = yfinance | csv,  model = BS(현물·ETF) | B76(선물옵션)
SOURCES = {
    "금 (GLD)": {"kind": "yfinance", "ticker": "GLD", "model": "BS"},
    "은 (SLV)": {"kind": "yfinance", "ticker": "SLV", "model": "BS"},
    "WTI 원유 (USO)": {"kind": "yfinance", "ticker": "USO", "model": "BS"},
    # 예) 자료를 구하면 이렇게 한 줄 추가:
    # "옥수수 선물옵션 (CBOT ZC)": {"kind": "csv", "path": "data/options/corn_zc.csv", "model": "B76"},
}

# CSV 표준 열: 한 행 = 한 옵션(한 날짜·만기·행사가·콜/풋)
CSV_COLS = {"asof": "시세 날짜 (YYYY-MM-DD)", "underlying": "기초자산 가격 (현물·ETF 가격, 선물옵션이면 해당 만기 선물가격)",
            "expiry": "옵션 만기일 (YYYY-MM-DD)", "strike": "행사가", "type": "C = 콜, P = 풋",
            "bid": "매수 호가 (없으면 비워 둠)", "ask": "매도 호가 (없으면 비워 둠)", "last": "최근 체결가 (호가 없을 때 사용)",
            "open_interest": "미결제약정 (선택, 없으면 비워 둠)", "last_time": "최근 체결 시각 (선택 — 호가가 없을 때 오래된 체결가를 빼는 데 씀)"}


# ── 가격 모형
def d1_d2(S, K, T, r, sig, q=0.0):
    """d1 = [ln(S/K) + (r − q + σ²/2)T] / (σ√T),  d2 = d1 − σ√T.  q = 배당·보유수익(원자재 ETF는 0으로 둠)."""
    S, K, T, sig = map(np.asarray, (S, K, T, sig))
    d1 = (np.log(S / K) + (r - q + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    return d1, d1 - sig * np.sqrt(T)


def bs_price(S, K, T, r, sig, call=True, q=0.0):
    """블랙숄즈: 콜 = S·e^(−qT)·N(d1) − K·e^(−rT)·N(d2),  풋 = K·e^(−rT)·N(−d2) − S·e^(−qT)·N(−d1)."""
    d1, d2 = d1_d2(S, K, T, r, sig, q)
    if call:
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def b76_price(F, K, T, r, sig, call=True):
    """Black-76(선물옵션): 블랙숄즈에서 S·e^(−qT) 자리에 F·e^(−rT) — 선물은 사는 데 돈이 들지 않으므로 q = r과 같음."""
    return bs_price(F, K, T, r, sig, call, q=r)


def greeks(S, K, T, r, sig, call=True, q=0.0):
    """민감도: 델타(가격 1 변화당), 감마(델타 변화), 베가(변동성 1%p당), 세타(하루당)."""
    d1, d2 = d1_d2(S, K, T, r, sig, q)
    pdf, eq, er = norm.pdf(d1), np.exp(-q * T), np.exp(-r * T)
    delta = eq * norm.cdf(d1) if call else -eq * norm.cdf(-d1)
    gamma = eq * pdf / (S * sig * np.sqrt(T))
    vega = S * eq * pdf * np.sqrt(T) / 100
    th = -S * eq * pdf * sig / (2 * np.sqrt(T))
    theta = (th - r * K * er * norm.cdf(d2) + q * S * eq * norm.cdf(d1)) if call else (th + r * K * er * norm.cdf(-d2) - q * S * eq * norm.cdf(-d1))
    return {"델타": delta, "감마": gamma, "베가": vega, "세타": theta / 365}


def implied_vol(price, S, K, T, r, call=True, model="BS"):
    """시장 옵션 가격 → 그 가격이 나오게 하는 σ (내재변동성). 내재가치 이하·범위 밖이면 NaN."""
    q = r if model == "B76" else 0.0
    lo = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) if call else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
    if not np.isfinite(price) or price <= lo + 1e-8 or T <= 0:
        return np.nan
    f = lambda s: bs_price(S, K, T, r, s, call, q) - price
    try:
        return brentq(f, 1e-4, 5.0) if f(1e-4) < 0 < f(5.0) else np.nan
    except ValueError:
        return np.nan


# ── 시세 불러오기 → 표준 표 (CSV_COLS 열)
def chain_yfinance(ticker: str, targets=(30, 60, 90, 180, 270, 365)) -> pd.DataFrame:
    """만기는 목표 일수(약 1·2·3·6·9·12개월)에 가장 가까운 것만 — 주간·일간 만기는 거래가 적어 곡선이 흐트러짐."""
    import yfinance as yf
    tk = yf.Ticker(ticker)
    S = float(tk.history(period="5d")["Close"].iloc[-1])
    today = pd.Timestamp.today().normalize()
    out = []
    days = {e: (pd.Timestamp(e) - today).days for e in tk.options if 7 <= (pd.Timestamp(e) - today).days <= 420}
    pick = sorted({min(days, key=lambda e: abs(days[e] - t)) for t in targets} if days else set())
    for e in pick:
        ch = tk.option_chain(e)
        for df, t in ((ch.calls, "C"), (ch.puts, "P")):
            out.append(pd.DataFrame({"asof": today, "underlying": S, "expiry": pd.Timestamp(e), "strike": df.strike, "type": t,
                                     "bid": df.bid, "ask": df.ask, "last": df.lastPrice, "open_interest": df.openInterest,
                                     "last_time": pd.to_datetime(df.lastTradeDate, utc=True).dt.tz_localize(None)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=list(CSV_COLS))


def chain_csv(src) -> pd.DataFrame:
    df = pd.read_csv(src)
    miss = [c for c in ("asof", "underlying", "expiry", "strike", "type") if c not in df.columns]
    if miss:
        raise ValueError(f"CSV에 필수 열 없음: {miss} — 표준 열: {list(CSV_COLS)}")
    for c in ("bid", "ask", "last", "open_interest", "last_time"):
        if c not in df.columns:
            df[c] = np.nan
    df["asof"], df["expiry"], df["last_time"] = pd.to_datetime(df["asof"]), pd.to_datetime(df["expiry"]), pd.to_datetime(df["last_time"])
    df["type"] = df["type"].astype(str).str.upper().str[0]
    return df[(df["asof"] == df["asof"].max())][list(CSV_COLS)].reset_index(drop=True)   # 가장 최근 날짜만


def load_chain(name: str) -> pd.DataFrame:
    s = SOURCES[name]
    return chain_yfinance(s["ticker"]) if s["kind"] == "yfinance" else chain_csv(ROOT / s["path"])


def save_snapshot(name: str, chain: pd.DataFrame):
    """그날 시세를 data/options/<이름>_<날짜>.csv로 저장(이미 있으면 덮어씀)."""
    if chain.empty:
        return None
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    tag = SOURCES.get(name, {}).get("ticker", "upload")
    p = SNAP_DIR / f"{tag}_{pd.Timestamp(chain['asof'].iloc[0]):%Y%m%d}.csv"
    chain.to_csv(p, index=False)
    return p


# ── 표준 표 → 내재변동성 점
def iv_points(chain: pd.DataFrame, r: float = 0.04, model: str = "BS", max_spread: float = 0.25, min_oi: float = 10,
              max_m: float = 0.35, stale_days: int = 2) -> pd.DataFrame:
    """외가격(OTM) 옵션만 사용(콜 K>S, 풋 K<S — 거래가 많고 조기행사 영향이 작음).
    가격 = 호가 중간값, 호가가 없으면 최근 체결가(가장 최근 체결보다 stale_days일 넘게 오래된 것은 제외).
    호가 차이가 넓거나 미결제약정이 적은 행은 제외 — 원천이 미결제약정을 비워 주면(전부 0) 이 기준은 건너뜀.
    결과 .attrs["note"] = 자료 품질 안내."""
    if chain.empty:
        return pd.DataFrame(columns=["expiry", "T", "strike", "m", "iv", "type", "spread", "oi"])
    c = chain.copy()
    c["T"] = (c["expiry"] - c["asof"]).dt.days / 365
    q = (c["bid"] > 0) & (c["ask"] > 0)
    c["price"] = np.where(q, (c["bid"] + c["ask"]) / 2, c["last"])
    c["spread"] = np.where(q, (c["ask"] - c["bid"]) / c["price"], np.nan)
    c["oi"] = c["open_interest"].fillna(0)
    has_oi = bool((c["oi"] > 0).mean() > 0.3)                # 원천이 일부만 채워 주면(장 마감 후) 기준으로 쓰지 않음
    lt = pd.to_datetime(c["last_time"]) if "last_time" in c else pd.Series(pd.NaT, index=c.index)
    fresh = q | lt.isna() | (lt >= lt.max() - pd.Timedelta(days=stale_days)) if lt.notna().any() else pd.Series(True, index=c.index)
    c["m"] = np.log(c["strike"] / c["underlying"])
    otm = ((c["type"] == "C") & (c["m"] > 0)) | ((c["type"] == "P") & (c["m"] <= 0))
    keep = otm & (c["T"] > 0) & (c["price"] > 0) & (c["m"].abs() < max_m) & ((c["spread"] < max_spread) | c["spread"].isna()) & ((c["oi"] >= min_oi) | (not has_oi)) & fresh
    c = c[keep].copy()
    c["iv"] = [implied_vol(p, S, K, T, r, t == "C", model) for p, S, K, T, t in zip(c["price"], c["underlying"], c["strike"], c["T"], c["type"])]
    c = c.dropna(subset=["iv"])
    out = c[(c["iv"] > 0.02) & (c["iv"] < 3)][["expiry", "T", "strike", "m", "iv", "type", "spread", "oi"]].reset_index(drop=True)
    nq = bool(q.mean() > 0.3)
    out.attrs["note"] = ("" if nq and has_oi else
                         "⚠️ 원천이 호가·미결제약정을 비워 줌(미국 장 마감 후 흔함) → 최근 체결가로 계산. 체결 시각이 행사가마다 달라 점이 흩어질 수 있음 — 미국 장중(한국 밤 10:30~새벽 5시) 다시 받으면 정확.")
    return out


def smile_fit(pts: pd.DataFrame):
    """만기별 2차 곡선 σ(m) = a + b·m + c·m² 맞춤 → {만기: (a, b, c, 점 수)}. a = 등가격 변동성, b = 기울기(skew), c = 휘어짐(smile)."""
    out = {}
    for e, g in pts.groupby("expiry"):
        if len(g) >= 5:
            c2, c1, c0 = np.polyfit(g["m"], g["iv"], 2)
            out[e] = (c0, c1, c2, len(g))
    return out
