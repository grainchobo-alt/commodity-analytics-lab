"""포트폴리오 VaR · 섹터별 위험 기여 · 섹터 비중 조절 권고 — 곡물 상한 25%(백테스트 로그 G · I)를 어떤 포트폴리오 · 섹터에도 쓸 수 있게 일반화.

입력: 품목별 비중(자본 대비 비율, + 롱 / − 숏) · 자본($) · 최근 일간 수익률과 σ
  σ = EWMA λ=0.94(RiskMetrics) — 곡물 상한 분석(strategy_graincap.var20)과 같음. GARCH(1,1) 등 다른 σ 경로는 호출하는 쪽에서 S로 넘김.
출력(모두 risk.position_risk 재사용: 정규 VaR · 다변량 FHS · 오일러 기여도)
  · 포트폴리오 VaR / ES / 연 변동성 / 분산 효과
  · 섹터별 노출 비중 vs VaR 기여 비중 (위험/비중 배수)
  · 최근 가격 추세 · 변동성 상태 표
  · 섹터 상한 시뮬레이션 3방식: 그 섹터만 줄임 / 줄인 만큼 나머지에 재배분(총 노출 유지) / 줄인 뒤 전체 σ를 원래 수준으로(목표 변동성 유지)
  · 자동 권고: 위험 비중이 노출 비중보다 큰 섹터의 상한과 그때 VaR · σ 변화

핵심 메시지(곡물 25% 사례): 목표 변동성을 같게 맞추면 σ는 같은데, 꼬리(FHS) 손실과 섹터 쏠림(기여 비중)은 달라질 수 있다 → σ만 보지 말고 VaR · 기여도를 함께 본다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import risk as RK

MODES = {"cut": "그 섹터만 줄임 (재배분 없음 → 총 노출 감소)", "redistribute": "줄인 만큼 나머지 품목에 비례 재배분 (총 노출 유지)",
         "retarget": "줄인 뒤 전체 크기를 조절해 연 변동성(σ)을 원래 수준으로 유지"}


# ------------------------------------------------------------------ 입력 준비
def ewma_inputs(L: pd.DataFrame, ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """일간 수익률(%)과 EWMA(λ=0.94) 분산(%²) — 모두 같은 날짜로 정렬. L = 롤 보정 가격 수준."""
    R, S = {}, {}
    for i in ids:
        r = RK.returns(L[i]); R[i] = r
        S[i] = pd.Series(RK.ewma_path(r.values, RK.RM_LAMBDA), index=r.index)
    R, S = pd.DataFrame(R).dropna(), pd.DataFrame(S).dropna()
    idx = R.index.intersection(S.index)
    return R.loc[idx], S.loc[idx]


def _clean(w: pd.Series) -> pd.Series:
    w = pd.Series(w, dtype=float); return w[w.abs() > 1e-9]


# ------------------------------------------------------------------ 위험 측정
def measure(R: pd.DataFrame, S: pd.DataFrame, w: pd.Series, capital: float, alpha: float = 0.99, h: int = 20, win: int = 500) -> dict:
    """w = 자본 대비 비중(+롱 −숏). 반환: position_risk 결과 + 자본 대비 % · 연 변동성(%)."""
    w = _clean(w); ids = list(w.index)
    o = RK.position_risk(R[ids], S[ids], w * capital, alpha, h, win)
    cap = float(capital)
    o["gross"] = float(w.abs().sum()); o["net"] = float(w.sum())
    o["ann_vol"] = float(o["sigma"] / np.sqrt(h) * np.sqrt(252) / cap * 100)       # 연 변동성 (% of 자본)
    o["var_normal_pct"] = float(o["var_normal"] / cap * 100); o["var_fhs_pct"] = float(o["var_fhs"] / cap * 100); o["es_fhs_pct"] = float(o["es_fhs"] / cap * 100)
    o["w"] = w
    return o


def sector_table(o: dict, sector_of: dict, w: pd.Series) -> pd.DataFrame:
    """섹터별 노출 비중(절댓값 기준) vs 정규 VaR 기여 비중 · 위험/비중 배수."""
    t = o["table"]; w = o["w"]
    sec = pd.Series({i: sector_of.get(i, "기타") for i in t.index})
    g = w.abs().groupby(sec).sum(); gs = g / g.sum()
    c = t["VaR 기여 ($, 정규)"].groupby(sec).sum(); cs = c / c.sum()
    df = pd.DataFrame({"노출 비중": gs, "순비중 (% 자본)": w.groupby(sec).sum() * 100, "VaR 기여 ($)": c, "VaR 기여 비중": cs})
    df["위험/비중 배수"] = df["VaR 기여 비중"] / df["노출 비중"]
    df["품목 수"] = sec.groupby(sec).size()
    return df.sort_values("VaR 기여 비중", ascending=False)


# ------------------------------------------------------------------ 섹터 비중 조절
def cap_sector(w: pd.Series, sector_of: dict, sector: str, cap_share: float, mode: str = "retarget",
               R: pd.DataFrame | None = None, S: pd.DataFrame | None = None, capital: float = 1e7, alpha: float = 0.99, h: int = 20) -> pd.Series:
    """섹터의 노출 비중(절댓값 합 ÷ 전체)이 cap_share를 넘지 않게 그 섹터 품목만 같은 비율로 줄임 (strategy_graincap.cap_grains와 같은 식)."""
    w = _clean(w); ids = [i for i in w.index if sector_of.get(i) == sector]
    tot = float(w.abs().sum()); ga = float(w[ids].abs().sum()); other = tot - ga
    if not ids or ga <= 0 or ga / tot <= cap_share or other <= 0:
        return w
    f = cap_share * other / ((1 - cap_share) * ga)
    w2 = w.copy(); w2[ids] = w2[ids] * f
    if mode == "cut":
        return w2
    if mode == "redistribute":
        oth = [i for i in w.index if i not in ids]
        w2[oth] = w2[oth] * ((tot - float(w2[ids].abs().sum())) / other)
        return w2
    s0 = measure(R, S, w, capital, alpha, h)["sigma"]; s1 = measure(R, S, w2, capital, alpha, h)["sigma"]       # retarget: σ를 원래 수준으로
    return w2 * (s0 / s1) if s1 > 0 else w2


def compare(R, S, w0: pd.Series, w1: pd.Series, capital: float, alpha: float = 0.99, h: int = 20) -> pd.DataFrame:
    a, b = measure(R, S, w0, capital, alpha, h), measure(R, S, w1, capital, alpha, h)
    rows = {"총 노출 (% 자본)": (a["gross"] * 100, b["gross"] * 100), "연 변동성 (%)": (a["ann_vol"], b["ann_vol"]), f"{h}일 VaR 정규 (% 자본)": (a["var_normal_pct"], b["var_normal_pct"]),
            f"{h}일 VaR FHS (% 자본)": (a["var_fhs_pct"], b["var_fhs_pct"]), f"{h}일 ES FHS (% 자본)": (a["es_fhs_pct"], b["es_fhs_pct"]),
            "분산 효과 (% 자본)": ((a["undiversified"] - a["var_normal"]) / capital * 100, (b["undiversified"] - b["var_normal"]) / capital * 100)}
    df = pd.DataFrame(rows, index=["조절 전", "조절 후"]).T
    df["변화"] = df["조절 후"] - df["조절 전"]; df["변화율 (%)"] = np.where(df["조절 전"].abs() > 1e-12, df["변화"] / df["조절 전"] * 100, np.nan)
    return df


def sweep(R, S, w: pd.Series, sector_of: dict, sector: str, caps, mode: str, capital: float, alpha: float = 0.99, h: int = 20) -> pd.DataFrame:
    """상한을 바꿔 가며 (총 노출 · σ · VaR · 그 섹터의 VaR 기여 비중). 지수 = 조절 전 100."""
    base = measure(R, S, w, capital, alpha, h); rows = []
    for c in [None] + list(caps):
        w2 = w if c is None else cap_sector(w, sector_of, sector, c, mode, R, S, capital, alpha, h)
        m = measure(R, S, w2, capital, alpha, h); st_ = sector_table(m, sector_of, w2)
        rows.append({"상한 (%)": np.nan if c is None else c * 100, "총 노출 (%)": m["gross"] * 100, "연 변동성 (%)": m["ann_vol"], "VaR 정규 (%)": m["var_normal_pct"], "VaR FHS (%)": m["var_fhs_pct"],
                     "ES FHS (%)": m["es_fhs_pct"], "섹터 노출 비중": float(st_["노출 비중"].get(sector, 0.0)), "섹터 VaR 기여 비중": float(st_["VaR 기여 비중"].get(sector, 0.0)),
                     "σ 지수": m["ann_vol"] / base["ann_vol"] * 100, "VaR FHS 지수": m["var_fhs_pct"] / base["var_fhs_pct"] * 100, "VaR 정규 지수": m["var_normal_pct"] / base["var_normal_pct"] * 100})
    return pd.DataFrame(rows)


def recommend(R, S, w: pd.Series, sector_of: dict, capital: float, alpha: float = 0.99, h: int = 20, margin: float = 0.08, min_share: float = 0.10) -> pd.DataFrame:
    """위험 비중이 노출 비중보다 margin(8%p) 넘게 큰 섹터마다, 그 섹터의 VaR 기여 비중이 '원래 노출 비중'까지 내려오는 상한을 이분법으로 찾음(목표 변동성 유지 방식)."""
    base = measure(R, S, w, capital, alpha, h); st_ = sector_table(base, sector_of, w); rows = []
    for sec, r in st_.iterrows():
        if sec == "기타" or r["노출 비중"] < min_share or r["VaR 기여 비중"] - r["노출 비중"] <= margin:
            continue
        lo, hi = 0.03, r["노출 비중"]                                         # 상한 후보 범위 (노출 비중 미만)
        for _ in range(14):
            mid = (lo + hi) / 2
            w2 = cap_sector(w, sector_of, sec, mid, "retarget", R, S, capital, alpha, h)
            sh = float(sector_table(measure(R, S, w2, capital, alpha, h), sector_of, w2)["VaR 기여 비중"].get(sec, 0.0))
            if sh > r["노출 비중"]:
                hi = mid
            else:
                lo = mid
        c = round(lo, 2) if lo > 0.03 else 0.03
        w2 = cap_sector(w, sector_of, sec, c, "retarget", R, S, capital, alpha, h); b = measure(R, S, w2, capital, alpha, h)
        rows.append({"섹터": sec, "현재 노출 비중": r["노출 비중"], "VaR 기여 비중": r["VaR 기여 비중"], "위험/비중 배수": r["위험/비중 배수"], "권고 상한": c,
                     "σ 변화 (%)": (b["ann_vol"] / base["ann_vol"] - 1) * 100, "VaR 정규 변화 (%)": (b["var_normal"] / base["var_normal"] - 1) * 100,
                     "VaR FHS 변화 (%)": (b["var_fhs"] / base["var_fhs"] - 1) * 100, "조정 후 섹터 VaR 기여": float(sector_table(b, sector_of, w2)["VaR 기여 비중"].get(sec, 0.0))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 최근 가격 추세 · 변동성 상태
def trend_table(L: pd.DataFrame, ids: list[str], w: pd.Series | None = None, lam: float = RK.RM_LAMBDA) -> pd.DataFrame:
    rows = {}
    for i in ids:
        p = L[i].dropna(); p = p[p > 0]
        if len(p) < 270:
            continue
        r = RK.returns(p); ch = lambda n: float((p.iloc[-1] / p.iloc[-1 - n] - 1) * 100)    # noqa: E731
        hi, lo = float(p.iloc[-252:].max()), float(p.iloc[-252:].min())
        v20, v252 = float(r.iloc[-20:].std() * np.sqrt(252)), float(r.iloc[-252:].std() * np.sqrt(252))
        up60, up120 = ch(60) > 0, ch(120) > 0
        trend = "상승" if up60 and up120 else "하락" if (not up60 and not up120) else "혼조"
        ratio = v20 / v252 if v252 > 0 else np.nan
        vs = "확대" if ratio > 1.3 else "진정" if ratio < 0.8 else "보통"
        row = {"20일 수익 (%)": ch(20), "60일 수익 (%)": ch(60), "120일 수익 (%)": ch(120), "252일 수익 (%)": ch(252), "52주 고점 대비 (%)": (p.iloc[-1] / hi - 1) * 100,
               "52주 저점 대비 (%)": (p.iloc[-1] / lo - 1) * 100, "20일 변동성 (%, 연)": v20, "1년 변동성 (%, 연)": v252, "변동성 배수": ratio, "추세": trend, "변동성 상태": vs}
        if w is not None and i in w.index:
            pos = "롱" if w[i] > 0 else "숏"
            fit = "순행" if (pos == "롱" and trend == "상승") or (pos == "숏" and trend == "하락") else "역행" if (pos == "롱" and trend == "하락") or (pos == "숏" and trend == "상승") else "중립"
            row.update({"포지션": pos, "추세와 포지션": fit, "주의": "손실 위험 높음" if fit == "역행" and vs == "확대" else ("변동성 확대" if vs == "확대" else "")})
        rows[i] = row
    return pd.DataFrame(rows).T
