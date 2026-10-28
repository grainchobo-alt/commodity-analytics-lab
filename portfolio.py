"""멀티 원자재 바스켓 포트폴리오 백테스트 — Newton-Raphson 최적 비중, 거래비용, 미래정보 차단

타이밍 (미래정보 누수 없음)
  결정일 D_k : 리밸런싱 주기의 마지막 거래일 (일간=매일, 주간=금요일, 월간=월말)
               → 신호·공분산·비용은 D_k 종가까지의 정보만 사용
  실행일 E_k : D_k 다음 거래일 종가 (신호를 본 뒤 실제로 거래 가능한 첫 시점)
  보유수익   : E_k 종가 → E_{k+1} 종가 (롤오버 갭 보정 가격)

최적화 (매 리밸런싱, Newton-Raphson)
  max_w  μ'w − (λ/2) w'Σw − Σ_i c_i·φ(w_i − w_i⁰)      φ(x)=√(x²+ε²)−ε  (|x|의 매끈한 근사 → 2계 미분 가능)
  롱온리   : Σw = 1, w ≥ 0      → 등식제약 Newton(KKT) + 로그배리어 τΣlog w_i (내부점법, τ → 0)
  롱숏     : Σ|w_i| ≤ L (총노출 한도, 정해진 투자금 대비) → 로그배리어 τ·log(L − Σφ(w_i))
  w⁰ = 직전 보유비중(가격 변동으로 표류한 비중) → 거래비용을 고려해 '덜 거래하는' 해를 찾음
  리스크 패리티(벤치마크): min ½y'Σy − Σ b_i log y_i  (Spinu 2013)를 Newton으로 → w = y/Σy

거래비용 (원자재 선물 통상 기준, 편도)
  매매비용 = (유동성 틱 수 × 1틱 가치 + 수수료) / 계약 명목금액(= 가격 × 승수)   — 가격이 바뀌면 비율도 바뀜
  롤 비용  = 연간 롤 횟수 × (1틱 + 수수료×2) / 명목금액 × |보유비중|  (만기 교체 스프레드 거래)
             — 연속월물 기준에만 적용. ETF 기준은 ETF 수익률에 실제 롤 비용이 이미 반영되어 있어 차감하지 않음
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import analytics as A

COMMISSION = 3.0   # 계약당 편도 수수료+거래소 수수료 (USD)

# 승수(계약 단위), scale(가격 → USD/단위), tick(1틱 USD), liq(유동성 틱: 반스프레드+슬리피지), rolls(연간 롤 횟수)
SPECS = {
    "yf:ZC=F": dict(mult=5000, scale=0.01, tick=12.50, liq=1.0, rolls=5),
    "yf:ZS=F": dict(mult=5000, scale=0.01, tick=12.50, liq=1.0, rolls=7),
    "yf:ZM=F": dict(mult=100, scale=1.0, tick=10.00, liq=1.0, rolls=8),
    "yf:ZL=F": dict(mult=60000, scale=0.01, tick=6.00, liq=1.0, rolls=8),
    "yf:ZW=F": dict(mult=5000, scale=0.01, tick=12.50, liq=1.0, rolls=5),
    "yf:KE=F": dict(mult=5000, scale=0.01, tick=12.50, liq=1.5, rolls=5),
    "yf:ZO=F": dict(mult=5000, scale=0.01, tick=12.50, liq=2.0, rolls=5),
    "yf:ZR=F": dict(mult=2000, scale=1.0, tick=10.00, liq=2.0, rolls=6),
    "yf:CL=F": dict(mult=1000, scale=1.0, tick=10.00, liq=1.0, rolls=12),
    "yf:BZ=F": dict(mult=1000, scale=1.0, tick=10.00, liq=1.0, rolls=12),
    "yf:NG=F": dict(mult=10000, scale=1.0, tick=10.00, liq=1.0, rolls=12),
    "yf:HO=F": dict(mult=42000, scale=1.0, tick=4.20, liq=1.0, rolls=12),
    "yf:RB=F": dict(mult=42000, scale=1.0, tick=4.20, liq=1.0, rolls=12),
    "yf:HG=F": dict(mult=25000, scale=1.0, tick=12.50, liq=1.0, rolls=5),
    "yf:GC=F": dict(mult=100, scale=1.0, tick=10.00, liq=1.0, rolls=6),
    "yf:SI=F": dict(mult=5000, scale=1.0, tick=25.00, liq=1.0, rolls=5),
    "yf:PL=F": dict(mult=50, scale=1.0, tick=5.00, liq=1.5, rolls=4),
    "yf:PA=F": dict(mult=100, scale=1.0, tick=50.00, liq=2.0, rolls=4),
    "yf:ALI=F": dict(mult=25, scale=1.0, tick=6.25, liq=2.0, rolls=12),
    "yf:TIO=F": dict(mult=100, scale=1.0, tick=1.00, liq=1.5, rolls=12),
    "yf:KC=F": dict(mult=37500, scale=0.01, tick=18.75, liq=1.0, rolls=5),
    "yf:SB=F": dict(mult=112000, scale=0.01, tick=11.20, liq=1.0, rolls=4),
    "yf:CC=F": dict(mult=10, scale=1.0, tick=10.00, liq=1.0, rolls=5),
    "yf:CT=F": dict(mult=50000, scale=0.01, tick=5.00, liq=1.0, rolls=5),
    "yf:OJ=F": dict(mult=15000, scale=0.01, tick=7.50, liq=2.0, rolls=6),
    "yf:LE=F": dict(mult=40000, scale=0.01, tick=10.00, liq=1.0, rolls=6),
    "yf:GF=F": dict(mult=50000, scale=0.01, tick=12.50, liq=1.5, rolls=8),
    "yf:HE=F": dict(mult=40000, scale=0.01, tick=10.00, liq=1.0, rolls=8),
}
DEFAULT_UNIVERSE = ["yf:ZC=F", "yf:ZS=F", "yf:ZM=F", "yf:ZL=F", "yf:ZW=F", "yf:KE=F", "yf:CL=F", "yf:NG=F",
                    "yf:HO=F", "yf:RB=F", "yf:HG=F", "yf:GC=F", "yf:SI=F", "yf:KC=F", "yf:SB=F", "yf:CC=F",
                    "yf:CT=F", "yf:LE=F", "yf:HE=F"]


# 손익 기준 'ETF': 선물 → (ETF, 연 보수%)
ETF_MAP = {"yf:CL=F": ("etf:USO", 0.60), "yf:BZ=F": ("etf:BNO", 1.00), "yf:NG=F": ("etf:UNG", 1.06),
           "yf:RB=F": ("etf:UGA", 0.97), "yf:ZC=F": ("etf:CORN", 1.00), "yf:ZS=F": ("etf:SOYB", 1.00),
           "yf:ZW=F": ("etf:WEAT", 1.00), "yf:SB=F": ("etf:CANE", 1.00), "yf:HG=F": ("etf:CPER", 0.80),
           "yf:GC=F": ("etf:GLD", 0.40), "yf:SI=F": ("etf:SLV", 0.50), "yf:PL=F": ("etf:PPLT", 0.60), "yf:PA=F": ("etf:PALL", 0.60)}


def excess_index(levels: pd.DataFrame, etf_id: str, add_fee: float = 0.0) -> pd.Series:
    """ETF 총수익 → 초과수익 지수: 일간 ln(ETF) − ln(1 + (단기금리 − 보수)×일수/365).
    ETF 총수익 ≈ 선물 초과수익 + 담보 단기금리 − 보수  ⇒  선물 초과수익 ≈ ETF − 금리 + 보수."""
    p = levels[etf_id].dropna(); p = p[p > 0]
    rf = levels["fred:DTB3"].ffill().reindex(p.index, method="ffill").fillna(0) / 100 if "fred:DTB3" in levels else pd.Series(0.0, index=p.index)
    days = p.index.to_series().diff().dt.days.fillna(0)
    er = np.log(p).diff().fillna(0) - np.log1p((rf - add_fee / 100) * days / 365)
    return 100 * np.exp(er.cumsum())


def etf_excess_levels(levels: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    """선물 id 열 이름으로 'ETF 기반 선물 초과수익 지수'를 반환 (실제 롤 비용 반영)."""
    return pd.DataFrame({i: excess_index(levels, ETF_MAP[i][0], ETF_MAP[i][1]) for i in ids
                         if i in ETF_MAP and ETF_MAP[i][0] in levels})


def cost_frames(prices: pd.DataFrame, liq_mult: float = 1.0, comm: float = COMMISSION):
    """prices: 실제 시세(날짜×id). return (편도 매매비용 비율, 연간 롤비용 비율)."""
    trade, roll = {}, {}
    for c in prices.columns:
        s = SPECS[c]
        notional = prices[c] * s["scale"] * s["mult"]
        trade[c] = (s["liq"] * liq_mult * s["tick"] + comm) / notional
        roll[c] = s["rolls"] * (s["tick"] + 2 * comm) / notional
    return pd.DataFrame(trade), pd.DataFrame(roll)


# ============================================================================
# Newton-Raphson 최적화
# ============================================================================
def _solve(M, b):
    """Newton 방향 계산. 공분산이 거의 특이(예: 일간 창에서 가격 정체 종목)해 헤시안이 특이하면 최소노름 해로 대체."""
    try:
        return np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(M, b, rcond=None)[0]


def newton_mv(mu, S, w0, c, lam, mode="long_only", gross=1.0, eps=1e-4, tol=1e-12, max_iter=60, w_max=1.0):
    """평균-분산-거래비용 최적화. 내부점 Newton(로그배리어, τ: 1e-3 → 1e-12).
    return (w, info{iter, newton_decrement, kkt, converged})."""
    n = len(mu)
    w_max = max(float(w_max), 1.0 / n + 1e-3) if mode == "long_only" else float(w_max)   # 롱온리는 1/N 이상이어야 실현가능
    cap = w_max < 1.0 - 1e-9 if mode == "long_only" else w_max < gross - 1e-9
    mu, w0, c = np.asarray(mu, float), np.asarray(w0, float), np.asarray(c, float)
    S = np.asarray(S, float)
    w = np.full(n, 1.0 / n) if mode == "long_only" else np.zeros(n)
    total, dec = 0, np.nan
    path = [w.copy()]                                   # 반복마다 비중 (도감 3D 경로 표시용)

    def F(w, tau):
        d = w - w0
        val = mu @ w - 0.5 * lam * w @ S @ w - c @ (np.sqrt(d * d + eps * eps) - eps)
        if cap:
            room = np.sqrt(w_max ** 2 + eps * eps) - np.sqrt(w * w + eps * eps)   # 경계가 정확히 |w_i| = w_max
            if np.any(room <= 0): return -np.inf
            val = val + tau * np.log(room).sum()
        if mode == "long_only":
            if np.any(w <= 0): return -np.inf
            return val + tau * np.log(w).sum()
        sl = gross - (np.sqrt(w * w + eps * eps) - eps).sum()
        return val + tau * np.log(sl) if sl > 0 else -np.inf

    tau = 1e-3
    while tau >= 1e-12:      # 배리어법 쌍대간극 ≤ n·τ → 최종 간극 ~1e-11
        for _ in range(max_iter):
            d = w - w0
            r = np.sqrt(d * d + eps * eps)
            g = mu - lam * S @ w - c * d / r
            H = -lam * S - np.diag(c * eps * eps / r ** 3)
            if cap:
                a = np.sqrt(w * w + eps * eps); room = np.sqrt(w_max ** 2 + eps * eps) - a; da = w / a; dda = eps * eps / a ** 3
                g = g - tau * da / room
                H = H - np.diag(tau * (dda / room + da * da / room ** 2))
            if mode == "long_only":
                g = g + tau / w
                H = H - np.diag(tau / w ** 2)
                K = np.zeros((n + 1, n + 1)); K[:n, :n] = H; K[:n, n] = 1; K[n, :n] = 1
                sol = _solve(K, np.r_[-g, 0.0])               # 등식제약 Newton (Σ Δw = 0 → Σw=1 유지)
                dx = sol[:n]
            else:
                a = np.sqrt(w * w + eps * eps); sl = gross - (a - eps).sum()
                da = w / a; dda = eps * eps / a ** 3
                g = g - tau * da / sl
                H = H - tau * (np.diag(dda) / sl + np.outer(da, da) / sl ** 2)
                dx = _solve(H, -g)
            dec = float(g @ dx)                                # Newton decrement² (≥0, 최대화)
            total += 1
            if dec / 2 < tol:
                break
            t, F0 = 1.0, F(w, tau)
            while t > 1e-14:
                wn = w + t * dx
                Fn = F(wn, tau)
                if np.isfinite(Fn) and Fn >= F0 + 0.25 * t * dec:   # Armijo (최대화)
                    break
                t *= 0.5
            w = w + t * dx
            path.append(w.copy())
        tau *= 0.1
    # KKT 잔차 (배리어 제외 원문제 기울기의 실현가능 방향 성분)
    d = w - w0
    grad = mu - lam * S @ w - c * d / np.sqrt(d * d + eps * eps)
    kkt = float(np.abs(grad - grad.mean()).max()) if mode == "long_only" else float(np.abs(grad).max())
    return w, dict(iter=total, decrement=dec, kkt=kkt, converged=bool(dec / 2 < 1e-8 or np.isnan(dec)), path=np.array(path))


def risk_budget_tilt(mu, S, tilt: float = 0.5, mode: str = "long_only", gross: float = 1.0):
    """리스크 패리티 + 신호 기울기 (risk budgeting). 위험 기여 비율 = 예산 b, b = (1−τ)/N + τ·(신호 강도 비례)."""
    mu = np.asarray(mu, float); n = len(mu)
    z = mu / np.sqrt(np.diag(S))
    if mode == "long_only":
        s_, sgn = np.maximum(z, 0), np.ones(n)
    else:
        s_, sgn = np.abs(z), np.where(mu >= 0, 1.0, -1.0)
    base = np.full(n, 1.0 / n)
    b = (1 - tilt) * base + tilt * (s_ / s_.sum() if s_.sum() > 0 else base)
    b = np.maximum(b, 1e-6); b /= b.sum()
    y, _ = newton_risk_parity(S * np.outer(sgn, sgn), b)
    return sgn * y * (gross if mode != "long_only" else 1.0)


def newton_risk_parity(S, b=None, tol=1e-12, max_iter=100):
    """Spinu(2013): min ½y'Σy − Σ b_i log y_i  → 기울기 0이면 각 자산 위험기여 = b_i. Newton + 백트래킹."""
    n = S.shape[0]
    b = np.full(n, 1.0 / n) if b is None else np.asarray(b)
    y = 1.0 / np.sqrt(np.diag(S)); it = 0
    f = lambda y: 0.5 * y @ S @ y - b @ np.log(y)
    for it in range(1, max_iter + 1):
        g = S @ y - b / y
        H = S + np.diag(b / y ** 2)
        dx = -np.linalg.solve(H, g)
        if -g @ dx / 2 < tol:
            break
        t = 1.0
        while np.any(y + t * dx <= 0) or f(y + t * dx) > f(y) + 0.25 * t * (g @ dx):
            t *= 0.5
        y = y + t * dx
    return y / y.sum(), dict(iter=it)


# ============================================================================
# 신호 (기대수익 μ) — 결정일까지의 정보만 사용
# ============================================================================
# 모멘텀 신호 기간 (영업일 기준 → 데이터 주기로 자동 환산). 업계 표준 = 12-1개월 (Moskowitz-Ooi-Pedersen 2012);
# 1개월 이하는 원자재에서 단기 반전이 섞이는 구간. CTA 혼합 = EWMA 교차 3쌍 (Baz et al. 2015).
MOM_SPECS = {"12-1개월 (표준)": (252, 21), "6개월": (126, 0), "3개월": (63, 0), "1개월": (21, 0),
             "3주": (15, 0), "1주": (5, 0), "CTA 혼합 (EWMA 교차 8/24·16/48·32/96일)": "cta"}


def _cta_score(r_hist: pd.DataFrame, ppy: int) -> pd.Series:
    """Baz-Granger-Harvey-Le Roux-Rattray (2015): 로그가격 EWMA(S) − EWMA(L)을 가격 변동성으로 나누고,
    다시 자기 1년 표준편차로 표준화 → 반응함수 φ(y)=y·e^(−y²/4)/0.89 (과도한 추세엔 비중 축소), 3쌍 평균."""
    scale = ppy / 252
    lp = r_hist.fillna(0).cumsum()
    sd = r_hist.rolling(max(5, int(63 * scale)), min_periods=5).std()
    out = []
    for S_, L_ in [(8, 24), (16, 48), (32, 96)]:
        hs, hl = max(1.0, S_ * scale), max(2.0, L_ * scale)
        x = (lp.ewm(halflife=hs).mean() - lp.ewm(halflife=hl).mean()) / (sd * np.sqrt(hl))
        y = x / x.rolling(max(10, int(252 * scale)), min_periods=10).std()
        out.append((y * np.exp(-y ** 2 / 4) / 0.89).iloc[-1])
    return pd.concat(out, axis=1).mean(axis=1)


def momentum_alpha(r_hist: pd.DataFrame, ppy: int, ic: float = 0.05, spec: str = "12-1개월 (표준)") -> pd.Series:
    """시계열 모멘텀을 기대수익으로 환산: α = IC × σ × score (Grinold 'alpha = IC·vol·score').
    score = clip(lookback 누적수익(최근 skip 제외) / (σ·√기간수), −2, 2). CTA 혼합은 EWMA 교차 점수."""
    sig = r_hist.iloc[-max(2, ppy):].std()
    sp = MOM_SPECS.get(spec, MOM_SPECS["12-1개월 (표준)"])
    if sp == "cta":
        score = _cta_score(r_hist, ppy).clip(-2, 2)
    else:
        lb, sk = sp
        k_lb = max(1, round(lb * ppy / 252)); k_sk = round(sk * ppy / 252)
        h = r_hist.iloc[-k_lb:]                                          # 창 = lookback (skip 포함, 기존 12-1과 동일)
        mom = h.iloc[:len(h) - k_sk].sum() if k_sk and len(h) > k_sk else h.sum()
        n = max(1, len(h) - (k_sk if len(h) > k_sk else 0))
        score = (mom / (sig * np.sqrt(n))).clip(-2, 2)
    return (ic * sig * score).fillna(0.0)


def ridge_forecasts(levels_adj: pd.DataFrame, ids: list[str], units: dict, freq: str, dec_index: pd.DatetimeIndex,
                    kinds: list[str], refit: int, min_train: int) -> pd.DataFrame:
    """종목별 walk-forward Ridge 예측 (특성: 모멘텀·변동성·계절성·COT·베이시스 + 자기시차).
    walk_forward는 시점 k의 예측에 [0, k) 구간만으로 학습한 모형을 사용 → 미래정보 없음. 반환: 다음 기간 로그수익률(소수)."""
    key = (tuple(ids), freq, tuple(kinds), refit, min_train, dec_index[0], dec_index[-1], len(dec_index),
           float(np.nansum(levels_adj[ids].iloc[-1].values)))
    if key in _FC_CACHE:              # 제약·λ·비용만 바꿔 재실행할 때 예측 재학습 생략
        return _FC_CACHE[key]
    out = {}
    ppy = A.PPY[freq]
    for i in ids:
        px = levels_adj[i].reindex(dec_index)
        r = A.log_return(px, units.get(i, "")).to_frame(i).dropna()
        if len(r) < min_train + 20:
            continue
        F = A.make_features(levels_adj, i, units, freq, r.index, kinds)
        D = A.build_design(r, i, [], "predict", lags=1, horizon=1, own_lags=True, train_frac=0.5, extra=F)
        if len(D.raw) < min_train + 10:
            continue
        p, _, _ = A.walk_forward(D, "Ridge", ppy, refit, start_frac=min_train / len(D.raw))
        out[i] = p / 100.0
    _FC_CACHE[key] = pd.DataFrame(out).reindex(dec_index)
    return _FC_CACHE[key]


_FC_CACHE: dict = {}


# ============================================================================
# 백테스트
# ============================================================================
REB_DAYS = {"B": 1, "W-FRI": 5, "ME": 21, "QE": 63}      # 주기당 평균 영업일


def decision_dates(idx: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """주기의 마지막 거래일 (B=매일, W-FRI=금요일, ME=월말, QE=분기말)."""
    if freq == "B":
        return idx
    per = {"W-FRI": "W-FRI", "ME": "M", "QE": "Q"}[freq]
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby(s.index.to_period(per)).max().values)


def backtest(levels_adj: pd.DataFrame, levels_act: pd.DataFrame, ids: list[str], units: dict, freq: str,
             start=None, end=None, signal: str = "momentum", mode: str = "long_only", lam: float = 5.0,
             gross: float = 1.0, ic: float = 0.05, cov_window: int | None = None, capital: float = 10_000_000,
             liq_mult: float = 1.0, comm: float = COMMISSION, kinds: list[str] | None = None,
             refit: int | None = None, min_train: int | None = None, progress=None, basis: str = "etf",
             w_max: float = 1.0, rebalance: str | None = None, forecasts: pd.DataFrame | None = None,
             mom_spec: str = "12-1개월 (표준)", trade_rule: str = "fixed", band: float = 0.10, max_hold: int = 21,
             kappa: float = 0.3, vol_model: str = "lw", vol_target: float | None = None, max_lev: float = 2.0,
             rb_tilt: float = 0.5) -> dict:
    """freq      : 신호·공분산을 계산하는 데이터 주기 (B/W-FRI/ME)
       rebalance : 리밸런싱 주기 (B/W-FRI/ME/QE, freq보다 같거나 느려야 함). 기본 = freq
       손익은 리밸런싱 사이에도 '일간'으로 추적 (계약 수 고정 → 가격 변동에 따라 비중 표류) → 통계는 모두 일간 기준(연 252일)
       basis='etf'    : 손익·신호·공분산 = ETF 기반 선물 초과수익 (실제 롤 비용 반영, 권장)
       basis='futures': Yahoo 연속월물(롤 보정) — 롤 비용 과소반영으로 수익 과대 가능
       trade_rule: 'fixed'   = 매 리밸런싱 점검일마다 목표 비중으로 이동 (기존)
                   'band'    = 무거래 구간: Σ|목표−현재| > band 이거나 마지막 거래 후 max_hold 영업일이 지나면 목표로 이동, 아니면 보유
                   'partial' = 부분 조정 (Gârleanu-Pedersen 2013): 현재 + κ·(목표−현재)
       → 'band'/'partial'은 rebalance='B'(매일 점검)와 함께 쓰는 것이 의도된 사용법
       vol_model : 'lw' = 과거 창 Ledoit-Wolf 공분산 그대로 / 'EWMA'·'HAR-RV'·'MIDAS-RV' = 상관은 LW, 품목별 변동성만
                   일간 walk-forward 예측(midas.daily_vol_forecasts, 다음 보유기간 일평균 분산)으로 교체
       vol_target: 연율 목표 변동성(예: 0.10). 예측 포트폴리오 변동성 대비 배율 min(max_lev, 목표/예측)로 전체 비중 조정
                   (모든 전략 동일 적용, 롱온리는 배율 < 1이면 나머지 현금 · > 1이면 레버리지)
       rb_tilt   : '리스크 버짓(신호 기울기)' 전략의 기울기 τ ∈ [0,1]. 위험 예산 bᵢ = (1−τ)/N + τ·sᵢ/Σs,
                   sᵢ = |μᵢ/σᵢ| (롱숏) 또는 max(μᵢ/σᵢ, 0) (롱온리). τ=0 → 리스크 패리티, τ=1 → 위험을 신호 강도에 비례 배분.
                   롱숏은 부호를 반영한 공분산 D·Σ·D (D = diag(sign μ))로 위험 기여를 계산 → 롱·숏 간 헤지 효과까지 반영"""
    rebalance = rebalance or freq
    if REB_DAYS[rebalance] < REB_DAYS[freq]:
        raise ValueError("리밸런싱 주기는 데이터 주기보다 같거나 길어야 합니다")
    ppy = A.PPY[freq]
    if basis == "etf":
        bench_src = levels_adj
        levels_adj = etf_excess_levels(levels_adj, ids)
        kinds = [k for k in (kinds or ["모멘텀", "실현변동성", "계절성", "COT 헤징압력"]) if k != "베이시스(캐리)"]
    else:
        bench_src = levels_adj
    cov_window = cov_window or {"B": 252, "W-FRI": 104, "ME": 60}[freq]
    refit = refit or {"B": 63, "W-FRI": 13, "ME": 6}[freq]
    min_train = min_train or {"B": 756, "W-FRI": 156, "ME": 60}[freq]
    ids = [i for i in ids if i in SPECS and i in levels_adj]
    # 일간 가격 (영업일, 최대 5일 ffill)
    Pa = levels_adj[ids].ffill(limit=5); Pa = Pa[Pa.index.dayofweek < 5]
    Pc = levels_act[ids].reindex(Pa.index).ffill(limit=5)
    if start is not None:
        lo = pd.Timestamp(start) - pd.DateOffset(years=12)
        Pa, Pc = Pa[Pa.index >= lo], Pc[Pc.index >= lo]
    if end is not None:
        Pa, Pc = Pa[Pa.index <= pd.Timestamp(end)], Pc[Pc.index <= pd.Timestamp(end)]
    days = Pa.index
    # 신호·공분산용: 데이터 주기 수익률
    Ddat = decision_dates(days, freq)
    Pdat = Pa.reindex(Ddat)
    Rdat = np.log(Pdat.where(Pdat > 0)).diff()
    # 리밸런싱: 결정일(주기 마지막 거래일) → 실행일(다음 거래일 종가)
    D_all = decision_dates(days, rebalance)
    pos = days.get_indexer(D_all)
    keep = pos + 1 < len(days)
    D_all = D_all[keep]; E_all = days[pos[keep] + 1]
    tcF, rcF = cost_frames(Pc, liq_mult, comm)
    tcF, rcF = tcF.values, rcF.values
    if basis == "etf":
        rcF = np.zeros_like(rcF)      # ETF 수익률에 실제 롤 비용이 이미 반영 → 선물 롤비용 이중 차감 방지
    fc = None
    if signal == "ridge":
        fc = ridge_forecasts(levels_adj, ids, units, freq, Ddat,
                             kinds or ["모멘텀", "실현변동성", "계절성", "COT 헤징압력", "베이시스(캐리)"], refit, min_train)
    elif signal == "panel":       # 패널 모델 walk-forward 예측 (결정일 × 종목, 기간 기대 로그수익)
        fc = forecasts.reindex(columns=ids)
    from sklearn.covariance import LedoitWolf
    VF = None
    if vol_model != "lw":
        import midas as MD
        rd = np.log(Pa.where(Pa > 0)).diff() * 100
        hv = max(5, REB_DAYS[rebalance])
        VF = pd.DataFrame({i: MD.daily_vol_forecasts(rd[i].rename(i), h=hv)[vol_model] for i in ids})   # %²/일
    Pv = Pa.values
    dpos = {d: i for i, d in enumerate(days)}
    N = len(ids); K = len(D_all)
    start_ts = pd.Timestamp(start) if start is not None else D_all[0]
    strategies = {"최적화(Newton)": dict(w=np.zeros(N), since=0), "동일가중 1/N": dict(w=np.zeros(N), since=0),
                  "리스크 패리티(Newton)": dict(w=np.zeros(N), since=0),
                  "리스크 버짓(신호 기울기)": dict(w=np.zeros(N), since=0)}
    rec = {k: dict(ret=[], gross=[], cost=[], turn=[], W=[]) for k in strategies}
    didx, ridx, diag = [], [], []
    started = False
    for k in range(K - 1):
        d = D_all[k]
        if d < start_ts:
            continue
        i0, i1 = dpos[E_all[k]], dpos[E_all[k + 1]]
        hold_days = days[i0 + 1:i1 + 1]
        if len(hold_days) == 0:
            continue
        hist = Rdat[Rdat.index <= d]
        win = hist.iloc[-cov_window:]
        fcrow = None
        if fc is not None:
            fh = fc[fc.index <= d]
            fcrow = fh.iloc[-1] if len(fh) else None

        def has_fc(i):
            return fc is None or (fcrow is not None and i in fcrow.index and np.isfinite(fcrow[i]))
        act = [j for j, i in enumerate(ids) if win[i].notna().sum() >= int(0.9 * cov_window)
               and np.isfinite(Pv[i0, j]) and np.isfinite(tcF[i0, j]) and has_fc(i)]
        cols = [ids[j] for j in act]
        X = win[cols].dropna() if len(act) >= 2 else pd.DataFrame()
        if len(act) < 2 or len(X) < 20:
            if started:   # 운용 시작 후 편입 종목 부족 → 해당 구간 무포지션
                for name in strategies:
                    z = [0.0] * len(hold_days)
                    rec[name]["ret"] += z; rec[name]["gross"] += z; rec[name]["cost"] += z
                didx += list(hold_days)
            continue
        started = True
        Hs = (i1 - i0) / REB_DAYS[freq]                                # 보유기간 (데이터 주기 단위)
        S = LedoitWolf().fit(X.values).covariance_ * Hs               # 보유기간 공분산
        if VF is not None:                                             # 품목 변동성만 예측치로 교체 (상관 유지)
            vf = VF[cols][VF.index <= d]
            if len(vf):
                sd_s = np.sqrt(np.diag(S))
                sd_f = np.sqrt(vf.iloc[-1].values / 1e4 * REB_DAYS[freq] * Hs)
                sd_f = np.where(np.isfinite(sd_f) & (sd_f > 0), sd_f, sd_s)
                S = S / np.outer(sd_s, sd_s) * np.outer(sd_f, sd_f)
        if signal == "momentum":
            mu = momentum_alpha(hist[cols].dropna(how="all"), ppy, ic, mom_spec).values * Hs
        elif signal in ("ridge", "panel"):
            mu = fcrow[cols].values * Hs
        else:  # hist_mean
            mu = hist[cols].mean().values * Hs
        c_now = tcF[i0, act]; roll_d = rcF[i0, act] / 252
        seg = pd.DataFrame(Pv[i0:i1 + 1][:, act]).ffill().values
        rel = seg / seg[0]
        rel = np.where(np.isfinite(rel), rel, 1.0)                    # 실행일 대비 가격비 (일간)
        for name, stt in strategies.items():
            w0 = stt["w"][act]
            if name == "최적화(Newton)":
                w_new, info = newton_mv(mu, S, w0, c_now, lam, mode, gross, w_max=w_max)
                diag.append(dict(date=d, n=len(act), hold_days=i1 - i0, **info))
            elif name == "동일가중 1/N":
                w_new = np.full(len(act), 1.0 / len(act))
                if mode != "long_only":
                    w_new = np.sign(mu) * gross / len(act)             # 롱숏 벤치마크: 신호 부호 × 동일비중
            elif name == "리스크 패리티(Newton)":
                w_new, _ = newton_risk_parity(S)
                if mode != "long_only":
                    w_new = np.sign(mu) * w_new * gross
            else:                                                      # 리스크 버짓: 위험 예산을 신호 강도로 기울임
                w_new = risk_budget_tilt(mu, S, rb_tilt, mode, gross)
            if vol_target:                                             # 변동성 타기팅: 예측 연율 변동성 → 목표
                pv = float(np.sqrt(max(w_new @ S @ w_new, 0) / Hs * ppy))
                if pv > 0:
                    w_new = w_new * min(max_lev, vol_target / pv)
            fresh = not np.any(stt["w"])                                # 첫 편입은 규칙과 무관하게 거래
            if trade_rule == "band" and not fresh:
                stt["since"] += i1 - i0
                if np.abs(w_new - w0).sum() <= band and stt["since"] < max_hold:
                    w_new = w0.copy()                                     # 무거래 구간 안 → 표류 비중 그대로 보유
            elif trade_rule == "partial" and not fresh:
                w_new = w0 + kappa * (w_new - w0)
            full_new = np.zeros(N); full_new[act] = w_new
            turn = np.abs(full_new - stt["w"])
            if turn.sum() > 1e-9:
                stt["since"] = 0
            tcost = float((turn[act] * c_now).sum())
            path = 1.0 + (rel - 1.0) @ w_new                           # 계약 수 고정 시 가치 경로
            dr = path[1:] / path[:-1] - 1.0
            rollc = float(np.abs(w_new) @ roll_d)
            cost = np.full(len(dr), rollc); cost[0] += tcost
            r_ = rec[name]
            r_["gross"] += list(dr); r_["ret"] += list(dr - cost); r_["cost"] += list(cost)
            r_["turn"].append(float(turn.sum())); r_["W"].append(full_new)
            drift = np.zeros(N)
            if path[-1] > 0:
                drift[act] = w_new * rel[-1] / path[-1]                # 다음 결정 시 직전(표류) 비중
            stt["w"] = drift
        didx += list(hold_days); ridx.append(E_all[k])
        if progress and k % max(1, K // 50) == 0:
            progress(k / K)
    idx = pd.DatetimeIndex(didx); ridx = pd.DatetimeIndex(ridx)
    res = {}
    for name, r_ in rec.items():
        ret = pd.Series(r_["ret"], index=idx); gr = pd.Series(r_["gross"], index=idx); cs = pd.Series(r_["cost"], index=idx)
        tn = pd.Series(r_["turn"], index=ridx); W = pd.DataFrame(r_["W"], index=ridx, columns=ids)
        res[name] = dict(ret=ret, nav=capital * (1 + ret).cumprod(), gross=gr, cost=cs, turn=tn, W=W,
                         stats=perf_stats(ret, gr, cs, tn, W, 252))
    # 투자 가능한 원자재 지수 벤치마크: DBC 보유 (총수익 − 단기금리 = 초과수익, 리밸런싱·비용 없음)
    if "etf:DBC" in bench_src and len(idx):
        dbc = excess_index(bench_src, "etf:DBC").reindex(days).ffill()
        pr = (dbc / dbc.shift(1) - 1).reindex(idx).fillna(0)
        z = pd.Series(0.0, index=idx); W1 = pd.DataFrame({"DBC": 1.0}, index=ridx)
        res["DBC 보유 (투자가능 지수)"] = dict(ret=pr, nav=capital * (1 + pr).cumprod(), gross=pr, cost=z,
                                          turn=pd.Series(0.0, index=ridx), W=W1,
                                          stats=perf_stats(pr, pr, z, pd.Series(0.0, index=ridx), W1, 252))
    return dict(results=res, diag=pd.DataFrame(diag), ids=ids, freq=freq, rebalance=rebalance, ppy=252, capital=capital,
                settings=dict(signal=signal, mode=mode, lam=lam, gross=gross, ic=ic, cov_window=cov_window, refit=refit,
                              min_train=min_train, liq_mult=liq_mult, comm=comm, basis=basis, w_max=w_max,
                              rebalance=rebalance, freq=freq, mom_spec=mom_spec, trade_rule=trade_rule, band=band,
                              max_hold=max_hold, kappa=kappa, vol_model=vol_model, vol_target=vol_target, max_lev=max_lev,
                              rb_tilt=rb_tilt))


def perf_stats(ret: pd.Series, gross: pd.Series, cost: pd.Series, turn: pd.Series, W: pd.DataFrame, ppy: int = 252) -> dict:
    """ret·gross·cost = 일간 시계열, turn·W = 리밸런싱 시점 기록."""
    if len(ret) < 3:
        return {}
    nav = (1 + ret).cumprod()
    yrs = len(ret) / ppy
    dd = nav / nav.cummax() - 1
    sr = lambda x: float(x.mean() / x.std() * np.sqrt(ppy)) if x.std() > 0 else np.nan
    down = ret[ret < 0].std()
    cagr = float(nav.iloc[-1] ** (1 / yrs) - 1) if nav.iloc[-1] > 0 else -1.0
    return {"연환산 수익률(CAGR)": cagr, "연환산 변동성": float(ret.std() * np.sqrt(ppy)),
            "Sharpe (비용 후)": sr(ret), "Sharpe (비용 전)": sr(gross),
            "Sortino": float(ret.mean() / down * np.sqrt(ppy)) if down > 0 else np.nan,
            "최대낙폭(MDD)": float(dd.min()), "Calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
            "연간 회전율(편도 합)": float(turn.sum() / yrs), "연간 비용 부담": float(cost.sum() / yrs),
            "연간 리밸런싱 횟수": float((turn > 1e-9).sum() / yrs),
            "평균 총노출": float(W.abs().sum(axis=1).mean()) if len(W) else np.nan,
            "평균 순노출": float(W.sum(axis=1).mean()) if len(W) else np.nan,
            "승률(일간)": float((ret > 0).mean()), "일수": int(len(ret)),
            "시작": ret.index.min().date(), "끝": ret.index.max().date()}


# ============================================================================
# 주어진 목표 비중 실행 (패널 롱숏·팩터 포트폴리오 등 공용)
# ============================================================================
def run_weights(W_dec: pd.DataFrame, levels_adj: pd.DataFrame, levels_act: pd.DataFrame, basis: str = "etf",
                liq_mult: float = 1.0, comm: float = COMMISSION, capital: float = 1.0) -> dict:
    """W_dec: 결정일 × 종목 목표 비중 (결정일 종가까지의 정보로 산출됐다고 가정).
    실행 = 결정일 다음 거래일 종가, 손익 = 일간 추적(계약 수 고정, 비중 표류), 비용 = 틱 기준 매매비용 + 롤비용.
    return dict(ret, gross, cost, turn, stats)"""
    ids = [c for c in W_dec.columns if c in SPECS]
    PX = etf_excess_levels(levels_adj, ids) if basis == "etf" else levels_adj[ids]
    ids = [c for c in ids if c in PX]
    Pa = PX[ids].ffill(limit=5); Pa = Pa[Pa.index.dayofweek < 5]
    Pc = levels_act[ids].reindex(Pa.index).ffill(limit=5)
    days = Pa.index; Pv = Pa.values
    tcF, rcF = cost_frames(Pc, liq_mult, comm); tcF, rcF = np.nan_to_num(tcF.values, nan=1e-3), np.nan_to_num(rcF.values)
    if basis == "etf":
        rcF = np.zeros_like(rcF)      # ETF 수익률에 롤 비용 내재 → 이중 차감 방지
    W_dec = W_dec[ids].fillna(0.0)
    W_dec = W_dec[W_dec.index.isin(days)]
    pos = days.get_indexer(W_dec.index); keep = pos + 1 < len(days)
    W_dec = W_dec[keep]; E = pos[keep] + 1
    w_prev = np.zeros(len(ids)); rets, gross, costs, turns, didx, ridx = [], [], [], [], [], []
    for k in range(len(E) - 1):
        i0, i1 = E[k], E[k + 1]
        if i1 <= i0:
            continue
        w = W_dec.iloc[k].values.astype(float)
        seg = pd.DataFrame(Pv[i0:i1 + 1]).ffill().values
        rel = np.where(np.isfinite(seg / seg[0]), seg / seg[0], 1.0)
        w = np.where(np.isfinite(Pv[i0]), w, 0.0)
        turn = np.abs(w - w_prev); tcost = float(turn @ tcF[i0])
        path = 1.0 + (rel - 1.0) @ w
        dr = path[1:] / path[:-1] - 1.0
        rollc = float(np.abs(w) @ rcF[i0]) / 252
        c = np.full(len(dr), rollc); c[0] += tcost
        rets += list(dr - c); gross += list(dr); costs += list(c); didx += list(days[i0 + 1:i1 + 1])
        turns.append(float(turn.sum())); ridx.append(days[i0])
        w_prev = w * rel[-1] / path[-1] if path[-1] > 0 else np.zeros(len(ids))
    idx = pd.DatetimeIndex(didx)
    ret, gr, cs = pd.Series(rets, index=idx), pd.Series(gross, index=idx), pd.Series(costs, index=idx)
    tn = pd.Series(turns, index=pd.DatetimeIndex(ridx))
    Wx = W_dec.iloc[:len(tn)].set_index(tn.index) if len(tn) else pd.DataFrame()
    return dict(ret=ret, gross=gr, cost=cs, turn=tn, nav=capital * (1 + ret).cumprod(),
                stats=perf_stats(ret, gr, cs, tn, Wx, 252) if len(ret) > 2 else {})
