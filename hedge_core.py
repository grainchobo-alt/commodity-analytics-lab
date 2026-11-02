"""헤지 랩 계산 모듈 (hedge_lab.py 전용, 기존 대시보드와 독립) — 선물 헤지 비율·효과·수단 추천·크러시 묶음 헤지.

모든 추정은 walk-forward: 시점 t의 헤지 비율은 t−1까지의 자료로만 정하고 t의 수익으로 평가 (표본 외 헤지 효과).
헤지 효과(HE) = 1 − Var(헤지 후 손익) / Var(헤지 전 손익). 1이면 위험 전부 제거, 0이면 효과 없음, 음수면 오히려 키움.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import procurement as PR
import risk as RK

# 선물 1계약 크기 (톤) — CBOT 표준: 5,000부셸 · 대두박 100숏톤 · 대두유 60,000파운드
CONTRACT_T = {"yf:ZS=F": 5000 * 0.0272155, "yf:ZW=F": 5000 * 0.0272155, "yf:KE=F": 5000 * 0.0272155, "yf:ZC=F": 5000 * 0.0254012,
              "yf:ZO=F": 5000 * 0.0145150, "yf:ZM=F": 100 * 0.907185, "yf:ZL=F": 60000 * 0.000453592, "yf:ZR=F": 2000 * 0.0453592}
HEDGE_CANDS = ["yf:ZS=F", "yf:ZM=F", "yf:ZL=F", "yf:ZW=F", "yf:KE=F", "yf:ZC=F", "yf:ZO=F", "yf:ZR=F", "yf:CL=F", "yf:HO=F"]
TARGETS = ["ak:M0", "ak:Y0", "ak:A0", "ak:C0", "yf:ZM=F", "yf:ZL=F", "yf:ZS=F", "yf:ZW=F", "yf:KE=F", "yf:ZC=F"]
NAMES = {**PR.BUY_IDS, "yf:CL=F": "WTI 원유", "yf:HO=F": "난방유 (경유)"}
PERIODS = {"2008–16": ("2008-01-01", "2016-12-31"), "2017–21": ("2017-01-01", "2021-12-31"), "2022–": ("2022-01-01", None)}


def name(i: str) -> str:
    return NAMES.get(i, i)


def weekly_usd(La: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    """롤 보정 가격 → 달러/톤 → 주간(금요일 종가). 위안 품목(DCE)은 그날 환율로 달러 환산."""
    out = {}
    for i in ids:
        if i in La:
            s = PR.usd_t(La, i) if (i in PR.TO_USD_T or i in PR.CNY_IDS) else La[i]
            out[i] = s.dropna()[lambda x: x > 0].resample("W-FRI").last()
    return pd.DataFrame(out)


def log_ret(P: pd.DataFrame | pd.Series, h: int = 1):
    """h주 로그 수익 (h > 1이면 겹치는 구간 — 헤지 기간이 h주인 물량)."""
    return np.log(P).diff(h)


def wf_minvar(s: pd.Series, F: pd.DataFrame | pd.Series, window: int = 104, min_obs: int = 52) -> pd.DataFrame:
    """최소분산 헤지 비율 h_t = Cov(s, F)·Var(F)⁻¹ (직전 window주, t−1까지). F가 여러 열이면 다중 회귀(절편 없음).
    반환: 열 = 헤지 비율(수단별) + s(헤지 전) + e(헤지 후)."""
    F = F.to_frame() if isinstance(F, pd.Series) else F
    D = pd.concat([s.rename("_s"), F], axis=1).dropna()
    cols = list(F.columns)
    H = pd.DataFrame(np.nan, index=D.index, columns=cols)
    x = D[cols].values; y = D["_s"].values
    for t in range(min_obs, len(D)):
        a = max(0, t - window)
        X, Y = x[a:t], y[a:t]
        try:
            H.iloc[t] = np.linalg.solve(X.T @ X, X.T @ Y)
        except np.linalg.LinAlgError:
            pass
    e = D["_s"] - (H * D[cols]).sum(axis=1, min_count=len(cols))
    return H.assign(s=D["_s"], e=e.where(H.notna().all(axis=1)))


def kalman_beta(s: pd.Series, f: pd.Series, delta: float = 1e-3, window: int = 52) -> pd.DataFrame:
    """시변 헤지 비율 β_t (랜덤워크, 칼만 필터): s_t = β_t f_t + ε_t,  β_t = β_{t−1} + η_t.
    η 분산 = delta × (직전 window주 s 분산 / f 분산) — β가 한 주에 움직일 수 있는 크기. 반환 β = 관측 전 예측값(t−1까지 정보)."""
    D = pd.concat([s.rename("s"), f.rename("f")], axis=1).dropna()
    vs = D["s"].rolling(window, min_periods=26).var().shift(1); vf = D["f"].rolling(window, min_periods=26).var().shift(1)
    b, P = 1.0, 1.0
    out = np.full(len(D), np.nan)
    for t, (st_, ft) in enumerate(D[["s", "f"]].values):
        R = vs.iloc[t] if np.isfinite(vs.iloc[t]) else np.nan
        if not np.isfinite(R) or not np.isfinite(vf.iloc[t]) or vf.iloc[t] <= 0:
            continue
        P = P + delta * R / vf.iloc[t]
        out[t] = b                                                          # t를 보기 전 예측 → 헤지에 사용
        S_ = ft * P * ft + R; K = P * ft / S_
        b = b + K * (st_ - b * ft); P = (1 - K * ft) * P
    beta = pd.Series(out, index=D.index)
    return pd.DataFrame({"beta": beta, "s": D["s"], "e": D["s"] - beta * D["f"]})


def he(s: pd.Series, e: pd.Series) -> float:
    m = s.notna() & e.notna()
    return float(1 - e[m].var() / s[m].var()) if m.sum() > 20 and s[m].var() > 0 else np.nan


def he_by_period(s, e, start="2008-01-01") -> dict:
    out = {"전체": he(s[s.index >= start], e[e.index >= start])}
    for k, (a, b) in PERIODS.items():
        m = (s.index >= a) & ((s.index <= b) if b else True)
        out[k] = he(s[m], e.reindex(s.index)[m])
    return out


def compare_hedges(s: pd.Series, F: pd.DataFrame, cands: list[str], window: int = 104, start="2008-01-01") -> pd.DataFrame:
    """헤지 수단 추천표: 수단마다 (가치 1:1, 표본 외 최소분산) 헤지 효과 — 전체·기간별, 최근 비율, 상관. 안정성 = 기간별 효과의 최솟값."""
    rows = []
    for c in cands:
        if c not in F or F[c].notna().sum() < 104:
            continue
        D = pd.concat([s, F[c]], axis=1).dropna()
        if len(D) < 104:
            continue
        W = wf_minvar(D.iloc[:, 0], D[[c]], window)
        hb = he_by_period(W["s"], W["e"], start)
        one = he_by_period(D.iloc[:, 0], D.iloc[:, 0] - D[c], start)
        rows.append({"수단": name(c), "id": c, "상관": D.corr().iloc[0, 1], "최근 헤지 비율": W[c].dropna().iloc[-1] if W[c].notna().any() else np.nan,
                     "효과 (최소분산)": hb["전체"], **{f"효과 {k}": hb[k] for k in PERIODS}, "안정성 (기간 최저)": min(hb[k] for k in PERIODS),
                     "효과 (가치 1:1)": one["전체"]})
    T = pd.DataFrame(rows)
    return T.sort_values("효과 (최소분산)", ascending=False).reset_index(drop=True) if len(T) else T


def best_pairs(s: pd.Series, F: pd.DataFrame, top: list[str], window: int = 104, start="2008-01-01") -> pd.DataFrame:
    """상위 수단 중 2개 조합의 표본 외 최소분산 헤지 효과."""
    rows = []
    for a_ in range(len(top)):
        for b_ in range(a_ + 1, len(top)):
            c = [top[a_], top[b_]]
            D = pd.concat([s, F[c]], axis=1).dropna()
            if len(D) < 104:
                continue
            W = wf_minvar(D.iloc[:, 0], D[c], window)
            hb = he_by_period(W["s"], W["e"], start)
            rows.append({"조합": " + ".join(name(x) for x in c), "효과 (최소분산)": hb["전체"], **{f"효과 {k}": hb[k] for k in PERIODS},
                         "안정성 (기간 최저)": min(hb[k] for k in PERIODS), **{f"비율 {name(x)}": W[x].dropna().iloc[-1] for x in c}})
    T = pd.DataFrame(rows)
    return T.sort_values("효과 (최소분산)", ascending=False).reset_index(drop=True) if len(T) else T


def contracts(expo_usd: float, ratio: float, fut_px_usd_t: float, fut: str) -> float:
    """헤지 계약 수 = 헤지 비율 × 노출($) / (선물 가격 $/t × 1계약 톤)."""
    ct = CONTRACT_T.get(fut)
    return float(ratio * expo_usd / (fut_px_usd_t * ct)) if ct and fut_px_usd_t > 0 else np.nan


# ── A1: 헤지 비율 → VaR
def hedge_curve(R: pd.DataFrame, S: pd.DataFrame, expo: pd.Series, alpha: float = 0.99, h: int = 20,
                ratios=np.linspace(0, 1, 11), order: str = "uniform") -> pd.DataFrame:
    """같은 품목 선물로 헤지(베이시스 0 가정): 남는 노출 = 노출 × (1 − 헤지 비율).
    order='uniform': 모든 품목 같은 비율. 'priority': 총 헤지 예산(노출 합 × 비율)을 '노출 1달러당 VaR 기여'가 큰 품목부터 채움."""
    base = RK.position_risk(R, S, expo, alpha, h)
    per_usd = (base["table"]["VaR 기여 ($, 정규)"] / expo.abs()).sort_values(ascending=False)
    rows = []
    for r in ratios:
        if order == "uniform":
            w = pd.Series(r, index=expo.index)
        else:
            budget, w = r * expo.abs().sum(), pd.Series(0.0, index=expo.index)
            for i in per_usd.index:
                take = min(abs(expo[i]), budget); w[i] = take / abs(expo[i]) if expo[i] else 0; budget -= take
        o = RK.position_risk(R, S, expo * (1 - w), alpha, h)
        rows.append({"헤지 비율": r, "VaR (정규)": o["var_normal"], "ES (정규)": o["es_normal"], "VaR (FHS)": o["var_fhs"], "ES (FHS)": o["es_fhs"],
                     **{f"비율 {i}": w[i] for i in expo.index}})
    return pd.DataFrame(rows)


# ── A4: 크러시
def board_crush(Lr: pd.DataFrame) -> pd.Series:
    """보드 크러시 마진 ($/부셸) = 대두박($/숏톤)×0.022 + 대두유(¢/lb)×0.11 − 대두(¢/부셸)/100 — 1부셸 대두 → 대두박 44lb · 대두유 11lb."""
    m, o, s = Lr["yf:ZM=F"], Lr["yf:ZL=F"], Lr["yf:ZS=F"]
    return (m * 0.022 + o * 0.11 - s / 100).dropna()


# ── A1 보강: USDA 보고서가 들어 있는 기간의 VaR (보고서 날 변동 배수 반영)
WASDE_PATH = Path(__file__).resolve().parent / "data" / "reports" / "wasde_long.parquet"
QTR_MONTHS = (3, 6, 9)                     # 3·6·9월 마지막 영업일 = 파종의향·작付면적·곡물재고 (근사)


def wasde_dates() -> pd.DatetimeIndex:
    """과거 WASDE 실제 발표일 (뉴욕 날짜)."""
    lng = pd.read_parquet(WASDE_PATH, columns=["release"])
    d = pd.to_datetime(lng.release).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    return pd.DatetimeIndex(sorted(set(d)))


def quarterly_dates(y0: int, y1: int) -> pd.DatetimeIndex:
    bd = pd.bdate_range(f"{y0}-01-01", f"{y1}-12-31")
    return pd.DatetimeIndex([bd[(bd.year == y) & (bd.month == m)].max() for y in range(y0, y1 + 1) for m in QTR_MONTHS])


def report_calendar(asof: pd.Timestamp, h: int) -> pd.DataFrame:
    """asof 다음 영업일부터 h영업일 안의 보고서일. 미래 WASDE는 '작년 같은 달 발표일과 같은 날(주말이면 다음 영업일)'로 추정 — 실제 일정은 USDA 공지 확인."""
    days = pd.bdate_range(asof + pd.Timedelta(days=1), periods=h)
    past = wasde_dates()
    rows = [(d, "WASDE", "실제") for d in days if d in past]
    for y, m in sorted({(d.year, d.month) for d in days}):
        if ((past.year == y) & (past.month == m)).any():
            continue
        ly = past[(past.year == y - 1) & (past.month == m)]
        if len(ly):
            est = pd.Timestamp(y, m, min(ly[0].day, 28))
            est = est if est.weekday() < 5 else est + pd.offsets.BDay(1)
            if est in days:
                rows.append((est, "WASDE", "추정 (작년 같은 날짜)"))
    rows += [(d, "분기 보고서", "근사 (월말 영업일)") for d in days if d in quarterly_dates(d.year, d.year)]
    return pd.DataFrame(sorted(rows), columns=["날짜", "보고서", "일정"])


def event_multipliers(r: pd.Series, s2: pd.Series, start="2008-01-01") -> dict:
    """보고서 날 변동 배수 = √(보고서 날 z² 평균 / 평소 z² 평균), z = 일간 수익 / 예측 σ (VaR와 같은 σ 모형)."""
    z = (r / np.sqrt(s2)).dropna(); z = z[z.index >= start]
    out = {}
    for nm, dates in (("WASDE", wasde_dates()), ("분기 보고서", quarterly_dates(z.index[0].year, z.index[-1].year))):
        d0 = pd.DatetimeIndex([z.index[z.index.searchsorted(d)] for d in dates if z.index[0] <= d <= z.index[-1]])
        on = z.index.isin(d0)
        out[nm] = float(np.sqrt((z[on] ** 2).mean() / (z[~on] ** 2).mean())) if on.sum() > 5 else 1.0
    return out


def event_scale(S: pd.DataFrame, h: int, k: dict, mult: dict) -> pd.DataFrame:
    """h일 분산 = σ² × [(h − Σk) + Σ k·배수²] → 마지막 행(현재 σ²)에 [ ] / h 를 곱한 S. k = {'WASDE': 일수, '분기 보고서': 일수}, mult = {품목: {종류: 배수}}."""
    S2 = S.copy()
    for i in S2.columns:
        n = sum(k.values())
        f = (h - n + sum(k[t] * mult[i].get(t, 1.0) ** 2 for t in k)) / h
        S2.iloc[-1, S2.columns.get_loc(i)] *= f
    return S2
