"""구매 타이밍 (중간 트레이더의 매입 의사결정) — '정해진 기간 안에 필요한 물량을 언제·얼마나 살까'.

평가 기준은 Sharpe가 아니라 '균등 분할 매수(TWAP) 대비 평균 매입가 절감(bp)' (+ 선택: 재판매 시점까지 보유비용).
매수 점수 (0~1, 높을수록 지금 사기 좋음) = 가중 평균:
  value  : 1 − (최근 lookback일 가격 분위)                  → 쌀수록 ↑
  turn   : Φ(최근 10일 수익 z)                             → 단기 반등 = 국소 저점 통과 확인 (떨어지는 칼날 회피)
  relval : 대체재·가공마진 대비 상대가치 (1년 분위)          → 옥수수↔밀, 대두↔크러시 마진, 대두박·대두유↔대두,
                                                            철광석↔DCE 수입 차익, 점결탄↔철근 고로 마진, 구리↔금
  season : Φ(−과거 연도들의 같은 달 상대가격 / σ)          → 계절 패턴 (올해 이전 연도만; 전체·최근 10년·최근 가중 선택)
  trend  : Φ(12-1개월 모멘텀 z)                            → 정보용 (실험상 구매 타이밍에는 해로워 기본 가중치 0)
속도 규칙: 기본 = 남은 물량/남은 주, 배율 = clip(1 + k·확신·(2·점수 − 1), 0, m_max), 마감 주 전량.
  확신 = 1 − 0.5·위험,  위험 = EWMA 변동성 분위 | MIDAS 변동성 예측 분위 | HMM 위기 레짐 필터 확률 → 위험할수록 균등 매수에 가깝게.
신호는 롤 보정 가격(만기 교체 갭 제거), 경제적 가격 관계(크러시·대체비율)는 실제 근월물 시세, 매입가는 실제 근월물 시세,
체결 = 결정일(금) 다음 거래일 종가. MIDAS·HMM은 walk-forward(그 시점 이전 데이터로만 추정·필터) → 미래정보 없음.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

BUY_IDS = {"yf:ZC=F": "옥수수", "yf:ZW=F": "밀 (CBOT SRW)", "yf:KE=F": "밀 (KC HRW)", "yf:ZS=F": "대두",
           "yf:ZM=F": "대두박", "yf:ZL=F": "대두유",
           "yf:TIO=F": "철광석 (SGX 62%)", "ak:JM0": "점결탄 (DCE)", "yf:HG=F": "구리 (COMEX)"}
# 원 단위 → 미국 달러/메트릭톤 (위안 표시 품목은 CNY_IDS: 그날 달러/위안 환율로 환산)
TO_USD_T = {"yf:ZC=F": 39.368 / 100, "yf:ZW=F": 36.744 / 100, "yf:KE=F": 36.744 / 100, "yf:ZS=F": 36.744 / 100,
            "yf:ZM=F": 1.10231, "yf:ZL=F": 22.0462 / 100, "yf:TIO=F": 1.0, "yf:HG=F": 2204.62}
GRAIN_IDS = ["yf:ZC=F", "yf:ZW=F", "yf:KE=F", "yf:ZS=F", "yf:ZM=F", "yf:ZL=F"]   # 2008~ 검증·프리셋 결정에 쓴 품목
METAL_IDS = ["yf:TIO=F", "ak:JM0", "yf:HG=F"]                                    # 2013~ 별도 검증 (experiments.procurement_metals)
CNY_IDS = {"ak:JM0"}
FX_CNY = "yf:CNY=X"
DEFAULT_W = {"value": 0.6, "turn": 0.4, "relval": 0.0, "season": 0.0, "trend": 0.0}   # 균형 = 국소 저점 (싸면서 반등 시작)
# 프리셋: 가중치·공격성 k·위험 판단·가격 위치 창·계절성 방식. 2008~2026, 6품목 × 13주 구매 기간 단계별 검증 결과:
#  - 추세는 구매자에게 해로움(평균 −20~−50bp) → 0
#  - 상대가치(실제 시세 관계)만 전·후반 모두 + (후반 +12bp, t≈2.1) — 롤보정 상대성과 버전은 효과 없음 → 경제적 관계가 핵심
#  - MIDAS 위험 판단·6개월 창·최근가중 계절은 소폭 개선 + 꼬리 위험 감소
_REC_W = {"value": 0.27, "turn": 0.18, "relval": 0.40, "season": 0.15, "trend": 0.0}
PRESETS = {"추천 (고도화: 상대가치+계절+MIDAS)": dict(w=_REC_W, k=1.0, risk="midas", lookback=126, season="ewm"),
           "균형 (국소 저점 = 위치+반등)": dict(w=DEFAULT_W, k=1.0, risk="ewma", lookback=252, season="all"),
           "보수적 (위치+반등+계절)": dict(w={"value": 0.45, "turn": 0.30, "season": 0.25, "relval": 0.0, "trend": 0.0}, k=1.0,
                                     risk="ewma", lookback=252, season="all"),
           "공격적 (가격 위치만, 2배)": dict(w={"value": 1.0, "turn": 0.0, "season": 0.0, "relval": 0.0, "trend": 0.0}, k=2.0,
                                       risk="ewma", lookback=252, season="all")}
COMP_LBL = {"value": "가격 위치 (쌀수록↑)", "turn": "단기 반등 (저점 확인)", "relval": "상대가치 (대체재 대비 쌀수록↑)",
            "season": "계절성 (저가 시즌↑)", "trend": "12개월 추세 (참고)"}
RISK_MODES = {"EWMA 변동성 분위": "ewma", "MIDAS 변동성 예측 분위": "midas", "HMM 위기 레짐 확률": "hmm"}
SEASON_MODES = {"전체 과거 연도": "all", "최근 10년": "10y", "최근 가중 (반감기 5년)": "ewm"}
# 상대가치 정의: 실제 근월물 시세 기반 경제적 관계. 부호 +1 = 값이 클수록 '내 품목이 상대적으로 쌈'
REL_DEF = {"yf:ZC=F": ("옥수수/밀 가격비 (낮을수록 옥수수 쌈)", -1), "yf:ZW=F": ("밀/옥수수 가격비 (낮을수록 밀 쌈)", -1),
           "yf:KE=F": ("KC밀/옥수수 가격비", -1), "yf:ZS=F": ("크러시 마진 (높을수록 대두 쌈)", +1),
           "yf:ZM=F": ("대두박 가치/대두 (낮을수록 대두박 쌈)", -1), "yf:ZL=F": ("대두유 가치/대두 (낮을수록 대두유 쌈)", -1),
           "yf:TIO=F": ("SGX/DCE 수입 차익 (낮을수록 SGX 철광석 쌈)", -1),
           "ak:JM0": ("철근 고로 마진 (높을수록 점결탄 쌈)", +1),
           "yf:HG=F": ("구리/금 가격비 (낮을수록 구리 쌈, 효과 약함)", -1)}
# 철강 원료 관계 (2013~2026, 13주 구매 · 추천 프리셋, 전반 2013~19.6 / 후반 2019.7~ 모두 + 인 정의만 채택):
#  - 철광석: SGX 달러 가격 vs DCE 위안 가격(환율·증치세 환산) — 중국 항구 수요 대비 해상 가격이 쌀 때 ↑  (+27bp, 전 +30 / 후 +24)
#  - 점결탄: 철근 − 1.6·철광석 − 0.5·코크스 (톤당 고로 마진) — 마진이 높으면 제철소 재고 확충 전 ↑     (+23bp, 전 +31 / 후 +17)
#  - 구리: 대체재 알루미늄 비율은 후반에 음(−) → 기각, 구리/금은 전·후반 + 이나 작음(+5bp, t≈1) — 참고 수준
#  - 탈락: 철광석 고로 마진·철광석/철근·철광석/점결탄, 점결탄 코크스 마진(효과 미미)


def cn_vat(idx: pd.DatetimeIndex) -> np.ndarray:
    """중국 증치세율 (DCE 가격은 세금 포함): 17% → 16% (2018-05-01) → 13% (2019-04-01)."""
    return np.where(idx < "2018-05-01", 1.17, np.where(idx < "2019-04-01", 1.16, 1.13))


def usd_t(Lr: pd.DataFrame, i: str) -> pd.Series:
    """실제 근월물 시세 → 달러/톤 (위안 품목은 그날까지의 달러/위안 환율, 세금 포함 그대로)."""
    s = Lr[i].dropna()
    if i in CNY_IDS:
        if FX_CNY not in Lr:
            return s * np.nan
        return s / Lr[FX_CNY].ffill().reindex(s.index)
    return s * TO_USD_T.get(i, 1.0)


# ============================================================================
# 입력 준비: 상대가치 관계 · MIDAS 변동성 · HMM 위기 확률 (모두 그 시점까지의 정보)
# ============================================================================
def rel_series(Lr: pd.DataFrame, i: str, La: pd.DataFrame | None = None, mode: str = "price") -> pd.Series | None:
    """mode='price': 실제 근월물 시세로 경제적 관계 (크러시 마진·가격비). mode='adj': 롤 보정 가격의 상대 성과(로그 비율)."""
    S = La if (mode == "adj" and La is not None) else Lr
    g = lambda k: S[k].dropna() if k in S else None
    ZC, ZW, ZS, ZM, ZL, KE = (g(k) for k in ["yf:ZC=F", "yf:ZW=F", "yf:ZS=F", "yf:ZM=F", "yf:ZL=F", "yf:KE=F"])
    TIO, I0, J0, JM, RB, HG, GC = (g(k) for k in ["yf:TIO=F", "ak:I0", "ak:J0", "ak:JM0", "ak:RB0", "yf:HG=F", "yf:GC=F"])
    try:
        if i == "yf:ZC=F":
            x = np.log(ZC) - np.log(ZW)
        elif i == "yf:ZW=F":
            x = np.log(ZW) - np.log(ZC)
        elif i == "yf:KE=F":
            x = np.log(KE) - np.log(ZC)
        elif i == "yf:ZS=F":
            x = (ZM * 0.022 * 100 + ZL * 11 - ZS) if mode == "price" else -(np.log(ZS) - np.log(ZM * 0.022 * 100 + ZL * 11))
        elif i == "yf:ZM=F":
            x = np.log(ZM * 0.022 * 100) - np.log(ZS)
        elif i == "yf:ZL=F":
            x = np.log(ZL * 11) - np.log(ZS)
        elif i == "yf:TIO=F":
            if mode == "price":
                fx = Lr[FX_CNY].ffill().reindex(I0.index)
                x = np.log(TIO) - np.log(I0 / fx / cn_vat(I0.index))
            else:
                x = -(np.log(TIO) - np.log(I0))
        elif i == "ak:JM0":
            x = (RB - 1.6 * I0 - 0.5 * J0) if mode == "price" else -(np.log(JM) - np.log(RB))
        elif i == "yf:HG=F":
            x = np.log(HG) - np.log(GC) if mode == "price" else -(np.log(HG) - np.log(GC))
        else:
            return None
    except (TypeError, KeyError):
        return None
    x = x.dropna()
    return x[x.index.dayofweek < 5]


def hmm_crisis_prob(p_adj: pd.Series, min_years: int = 5, seed: int = 0) -> pd.Series:
    """2-레짐 가우시안 HMM (일간 수익률). 매년 초 그 이전 데이터로만 재추정 → 그해 각 날짜의 '필터' 확률
    P(고변동 레짐 | r₁..r_t) (평활 확률은 미래를 쓰므로 사용 안 함)."""
    from hmmlearn.hmm import GaussianHMM
    r = (np.log(p_adj.dropna()).diff() * 100).dropna()
    r = r[r.index.dayofweek < 5]
    x = r.values; idx = r.index; yrs = idx.year.values
    out = np.full(len(r), np.nan)
    for y in sorted(set(yrs)):
        tr = yrs < y
        if tr.sum() < 252 * min_years:
            continue
        m = GaussianHMM(2, "diag", n_iter=100, random_state=seed).fit(x[tr][:, None])
        mu = m.means_.ravel(); sd = np.sqrt(np.array([np.ravel(c)[0] for c in m.covars_]))
        hi = int(np.argmax(sd))
        seg = np.flatnonzero(yrs <= y)
        lb = norm.logpdf(x[seg][:, None], mu[None, :], sd[None, :])
        A = m.transmat_; a = m.startprob_ * np.exp(lb[0] - lb[0].max()); a /= a.sum()
        cur = yrs[seg] == y
        filt = np.empty(len(seg)); filt[0] = a[hi]
        for t in range(1, len(seg)):
            a = (a @ A) * np.exp(lb[t] - lb[t].max()); a /= a.sum(); filt[t] = a[hi]
        out[seg[cur]] = filt[cur]
    return pd.Series(out, index=idx)


def midas_var(p_adj: pd.Series, h: int = 5) -> pd.Series:
    import midas as MD
    r = (np.log(p_adj.dropna()).diff() * 100).dropna()
    r = r[r.index.dayofweek < 5]
    return MD.daily_vol_forecasts(r.rename(p_adj.name), h=h)["MIDAS-RV"]


def inputs(La: pd.DataFrame, Lr: pd.DataFrame, i: str, risk: str = "ewma", rel_mode: str = "price") -> dict:
    return dict(rel=rel_series(Lr, i, La, rel_mode), rel_sign=REL_DEF.get(i, ("", 1))[1] if rel_mode == "price" else 1,
                vol_fc=midas_var(La[i]) if risk == "midas" else None, crisis=hmm_crisis_prob(La[i]) if risk == "hmm" else None)


# ============================================================================
# 신호 · 점수 · 속도
# ============================================================================
def _season(lp: pd.Series, idx: pd.DatetimeIndex, mode: str = "all", hl: float = 5.0) -> np.ndarray:
    m = lp.resample("ME").mean()
    rel = m - m.groupby(m.index.year).transform("mean")
    sea = np.full(len(idx), np.nan)
    years = idx.year.values; months = idx.month.values
    for y in np.unique(years):
        past = rel[rel.index.year < y]
        if mode == "10y":
            past = past[past.index.year >= y - 10]
        if past.index.year.nunique() < 5:
            continue
        if mode == "ewm":
            wt = 0.5 ** ((y - 1 - past.index.year.values) / hl)
            g = pd.Series(past.values * wt).groupby(past.index.month.values).sum() / pd.Series(wt).groupby(past.index.month.values).sum()
        else:
            g = past.groupby(past.index.month).mean()
        sd = past.std() or 1.0
        sel = years == y
        sea[sel] = norm.cdf(-g.reindex(months[sel]).fillna(0).values / sd)
    return sea


def signals(p_adj: pd.Series, lookback: int = 252, rel: pd.Series | None = None, rel_sign: int = 1,
            vol_fc: pd.Series | None = None, crisis: pd.Series | None = None, season: str = "all") -> pd.DataFrame:
    """일간 롤 보정 가격 → 시점별 구성 점수(0~1)·위험 지표. 모든 값은 그 시점까지의 데이터만 사용."""
    p = p_adj.dropna(); p = p[p > 0]
    p = p[p.index.dayofweek < 5]
    lp = np.log(p); r = lp.diff()
    out = pd.DataFrame(index=p.index)
    out["value"] = 1 - lp.rolling(lookback, min_periods=lookback // 2).rank(pct=True)
    vol_d = r.rolling(63, min_periods=20).std()
    mom = lp.shift(21) - lp.shift(252)
    out["trend"] = norm.cdf(mom / (vol_d * np.sqrt(231)))
    out["turn"] = norm.cdf((lp - lp.shift(10)) / (vol_d * np.sqrt(10)))
    out["season"] = _season(lp, p.index, season)
    if rel is not None and len(rel):
        rr = rel.reindex(p.index).ffill(limit=5)
        pct = rr.rolling(252, min_periods=126).rank(pct=True)
        out["relval"] = pct if rel_sign > 0 else 1 - pct
    else:
        out["relval"] = np.nan
    ew = r.ewm(alpha=0.06, adjust=False).std()
    out["vol_ann"] = ew * np.sqrt(252)
    out["vol_pct"] = ew.rolling(756, min_periods=126).rank(pct=True)
    if vol_fc is not None:
        vf = np.sqrt(vol_fc.reindex(p.index).ffill(limit=5))
        out["risk"] = vf.rolling(756, min_periods=126).rank(pct=True)
        out["vol_ann"] = (vf / 100 * np.sqrt(252)).fillna(out["vol_ann"])
    elif crisis is not None:
        out["risk"] = crisis.reindex(p.index).ffill(limit=5)
    else:
        out["risk"] = out["vol_pct"]
    return out


def score(sig: pd.DataFrame, w: dict | None = None) -> pd.Series:
    w = w or DEFAULT_W
    cols = [k for k in w if k in sig and w[k] > 0]
    S = sig[cols]
    ww = pd.Series({k: w[k] for k in cols})
    num = (S * ww).sum(axis=1, min_count=1); den = S.notna().mul(ww).sum(axis=1)
    return (num / den.replace(0, np.nan)).clip(0, 1)


def pace(score_now: float, risk: float, remaining: float, weeks_left: int, k: float = 1.0, m_max: float = 3.0) -> float:
    """이번 주 매수량 (남은 물량 단위). 마감 주는 전량. risk ∈ [0,1] (클수록 보수적)."""
    if weeks_left <= 1:
        return remaining
    base = remaining / weeks_left
    if not np.isfinite(score_now):
        return base
    conf = 1 - 0.5 * (risk if np.isfinite(risk) else 0.5)
    mult = float(np.clip(1 + k * conf * (2 * score_now - 1), 0, m_max))
    return min(remaining, base * mult)


# ============================================================================
# 백테스트
# ============================================================================
def backtest(p_act: pd.Series, p_adj: pd.Series, weeks: int = 13, step: int = 4, start="2008-01-01", k: float = 1.0,
             m_max: float = 3.0, w: dict | None = None, lookback: int = 252, inp: dict | None = None, season: str = "all",
             carry_bp_week: float = 0.0) -> pd.DataFrame:
    """구매 기간(weeks주)을 step주 간격으로 반복 시작. 매주 금요일(결정일) 점수 → 다음 거래일 실제 종가에 매수.
    carry_bp_week: 기간 말(재판매·인도 시점)까지 보유하는 주당 비용(금융+보관, bp) — 일찍 살수록 비용 ↑, 모든 방식에 동일 적용.
    반환: 기간별 실효 평균 매입가 — 규칙 / 균등(TWAP) / 기간 초 일괄 / 가격 위치만 / 사후 최저(이론 상한) + 시작 시점 위험 지표."""
    inp = inp or {}
    sig = signals(p_adj, lookback, inp.get("rel"), inp.get("rel_sign", 1), inp.get("vol_fc"), inp.get("crisis"), season)
    sc = score(sig, w)
    sc_v = score(sig, {"value": 1.0})
    pa = p_act.dropna(); pa = pa[pa > 0]
    days = pa.index.intersection(sig.index)
    fri = pd.Series(days, index=days).groupby(days.to_period("W-FRI")).max()
    fri = pd.DatetimeIndex(fri.values)
    fri = fri[fri >= pd.Timestamp(start)]
    pos = days.get_indexer(fri)
    exe = [days[i + 1] if i + 1 < len(days) else None for i in pos]
    hold = 1 + carry_bp_week / 1e4 * (weeks - 1 - np.arange(weeks))      # 주차별 보유비용 배수
    rows = []
    for s0 in range(0, len(fri) - weeks - 1, step):
        win = range(s0, s0 + weeks)
        if exe[s0 + weeks - 1] is None:
            break
        px = np.array([pa[exe[j]] for j in win]) * hold
        res = {}
        for nm, s_ in [("규칙", sc), ("가격 위치만", sc_v)]:
            rem, q = 1.0, np.zeros(weeks)
            for t, j in enumerate(win):
                d = fri[j]
                q[t] = pace(float(s_.get(d, np.nan)), float(sig["risk"].get(d, np.nan)), rem, weeks - t, k, m_max)
                rem -= q[t]
            res[nm] = float(q @ px)
        rows.append(dict(start=exe[s0], end=exe[s0 + weeks - 1], 규칙=res["규칙"], 가격위치만=res["가격 위치만"],
                         균등=float(px.mean()), 일괄=float(px[0]), 사후최저=float(px.min()),
                         시작위험=float(sig["vol_pct"].get(fri[s0], np.nan))))
    B = pd.DataFrame(rows).set_index("start")
    for c in ["규칙", "가격위치만", "일괄", "사후최저"]:
        B[f"절감_{c}"] = (B["균등"] - B[c]) / B["균등"] * 1e4            # bp, +면 균등보다 싸게 삼
    return B


def summarize(B: pd.DataFrame, weeks: int, step: int, halflife_years: float | None = None) -> dict:
    nov = B.iloc[::max(1, weeks // step)]                                # 겹치지 않는 기간만으로 t값
    out = {}
    if halflife_years:
        age = np.asarray((B.index.max() - B.index).days, float) / 365.25
        wt = pd.Series(0.5 ** (age / halflife_years), index=B.index)
    for c in ["규칙", "가격위치만", "일괄", "사후최저"]:
        x, xn = B[f"절감_{c}"], nov[f"절감_{c}"]
        out[c] = {"평균 절감 (bp)": x.mean(), "중앙값 (bp)": x.median(), "균등보다 싸게 산 비율": (x > 0).mean(),
                  "하위 5% (bp)": x.quantile(0.05), "t값 (비중첩)": xn.mean() / xn.std() * np.sqrt(len(xn)) if xn.std() > 0 else np.nan,
                  "기간 수": len(x)}
        if halflife_years:
            out[c]["최근 가중 평균 (bp)"] = float((x * wt).sum() / wt.sum())
    return out


def band_4w(p_adj: pd.Series, sig: pd.DataFrame, last, h: int = 20) -> tuple[float, float]:
    """향후 h영업일 가격의 90% 범위 배수 (하단, 상단). 정규 ±1.645σ 대신 '표준화 h일 수익률의 경험적 5%·95% 분위'
    (그 시점까지 실현이 끝난 표본만) → 꼬리가 두꺼운 품목(철광석 등)의 범위가 실제 적중률 90%에 가깝게 넓어짐."""
    import risk as RK
    sd = sig["vol_ann"] / np.sqrt(252) * 100
    qlo, qhi = RK.hday_band(RK.returns(p_adj), sd, h, asof=last)
    s = float(sd.loc[last]) * np.sqrt(h) / 100
    return float(np.exp(qlo * s)), float(np.exp(qhi * s))


def today_table(La: pd.DataFrame, Lr: pd.DataFrame, ids: list[str], w: dict | None = None, k: float = 1.0, m_max: float = 3.0,
                weeks_left: int = 13, risk: str = "ewma", season: str = "all", lookback: int = 252, rel_mode: str = "price") -> pd.DataFrame:
    rows = []
    for i in ids:
        if i not in La or i not in Lr:
            continue
        inp = inputs(La, Lr, i, risk, rel_mode)
        sig = signals(La[i], lookback, inp["rel"], inp["rel_sign"], inp["vol_fc"], inp["crisis"], season)
        sc = score(sig, w)
        last = sig.dropna(subset=["value"]).index.max()
        pa = usd_t(Lr, i)
        px = float(pa.loc[:last].iloc[-1])
        p1 = pa.loc[last - pd.DateOffset(years=1):last]
        rk = float(sig["risk"].loc[last])
        mult = pace(float(sc.loc[last]), rk, 1.0, weeks_left, k, m_max) / (1.0 / weeks_left)
        b_lo, b_hi = band_4w(La[i], sig, last)
        va = float(sig["vol_ann"].loc[last])
        rows.append({"품목": BUY_IDS.get(i, i), "id": i, "기준일": last.date(), "가격 ($/t)": px,
                     "1년 최저 ($/t)": p1.min(), "1년 최고 ($/t)": p1.max(),
                     "매수 점수": float(sc.loc[last]) * 100, **{COMP_LBL[c]: float(sig[c].loc[last]) * 100 for c in COMP_LBL},
                     "위험 지표": rk * 100, "변동성 분위": float(sig["vol_pct"].loc[last]) * 100, "연 변동성": va,
                     "4주 하단 90% ($/t)": px * b_lo, "4주 상단 90% ($/t)": px * b_hi,
                     "권장 속도 (균등 대비 배)": mult, "상대가치 지표": REL_DEF.get(i, ("–",))[0]})
    return pd.DataFrame(rows)


# ============================================================================
# 펀더멘털 맥락 (점수에는 넣지 않음 — 기관 리포트·COT는 발표 후 예측력이 재현되지 않았음, 참고 정보)
# ============================================================================
WASDE_MAP = {"yf:ZC=F": "corn", "yf:ZS=F": "soybean", "yf:ZM=F": "soymeal", "yf:ZL=F": "soyoil", "yf:ZW=F": "wheat", "yf:KE=F": "wheat"}
COT_NAME = {"yf:ZC=F": "Corn", "yf:ZS=F": "Soybeans", "yf:ZM=F": "Soybean Meal", "yf:ZL=F": "Soybean Oil", "yf:ZW=F": "Wheat SRW", "yf:KE=F": "Wheat HRW",
            "yf:HG=F": "Copper"}


def fundamentals(ids: list[str]) -> pd.DataFrame:
    from pathlib import Path
    root = Path(__file__).parent / "data"
    W = pd.read_parquet(root / "reports" / "wasde_features.parquet") if (root / "reports" / "wasde_features.parquet").exists() else None
    C = pd.read_parquet(root / "cot.parquet") if (root / "cot.parquet").exists() else None
    rows = []
    for i in ids:
        r = {"품목": BUY_IDS.get(i, i)}
        if W is not None and i in WASDE_MAP:
            g = W[W.commodity == WASDE_MAP[i]].sort_values("release")
            if len(g):
                last = g.iloc[-1]
                hist = g[(g.release.dt.month == last.release.month) & (g.release < last.release)]["stu"]
                r.update({"WASDE 발표일": last.release.date(), "연산연도": last.my, "재고/사용비율 (%)": last.stu,
                          "전월 대비 (%p)": last.get("d_stu", np.nan),
                          "같은 달 과거 분위": float((hist < last.stu).mean()) if len(hist) else np.nan,
                          "USDA 가격 전망 수정 (%)": last.get("d_price", np.nan)})
        if C is not None and i in COT_NAME:
            c = C[C.name == COT_NAME[i]].sort_values("date")
            if len(c):
                mm = c.mm_net / c.open_interest_all * 100
                z = (mm - mm.rolling(156, min_periods=52).mean()) / mm.rolling(156, min_periods=52).std()
                r.update({"COT 기준일": c.date.iloc[-1].date() if hasattr(c.date.iloc[-1], "date") else c.date.iloc[-1],
                          "펀드 순포지션 (% OI)": float(mm.iloc[-1]), "펀드 포지션 z (3년)": float(z.iloc[-1])})
        rows.append(r)
    return pd.DataFrame(rows).set_index("품목")
