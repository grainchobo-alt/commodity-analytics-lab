"""롱숏 전략 확장 — 칼만 필터 · 헤지펀드식 상대가치(횡단면) 전략을 원자재에 옮겨 백테스트

  주식 헤지펀드의 'A 롱 / B 숏'(예: 구글 롱·엔비디아 숏, 과열 대형주 숏으로 신생 테마주 매수 자금 마련)은
  ① 같은 테마(섹터) 안에서 상대가치·과열을 사고팔고 ② 시장 방향(베타)은 상쇄하는 구조 → 원자재로 옮기면:
    K1 칼만 추세        : 로그가격의 '수준 + 기울기' 상태공간 모형, 필터 기울기 t값 부호 (52주 모멘텀의 적응형 대안)
    K2 칼만 동적 헤지 페어: 회귀계수 β가 매주 변하는 페어(칼만), 예측오차 z로 평균회귀
    S1 섹터 중립 모멘텀  : 섹터(곡물·에너지·금속·소프트·축산) 안에서 26주 강한 품목 롱 · 약한 품목 숏 (승자 롱/패자 숏)
    S2 섹터 내 따라잡기  : 반대로 섹터 안 과열(26주 최강) 숏 · 낙오(최약) 롱 — '과열 대장주 숏으로 후발주 매수'의 원자재판
    V1 가치 (5년 역추세): 5년 평균 대비 가장 싼 품목 롱 · 비싼 품목 숏 (Asness-Moskowitz-Pedersen 2013)
    W1 왜도            : 52주 왜도가 가장 낮은(급락 꼬리) 품목 롱 · 높은 품목 숏 (Fernandez-Perez et al. 2018)
    L1 저변동성        : 변동성 낮은 품목 롱 · 높은 품목 숏 (변동성 역가중으로 위험 균형)
    B1 저베타 (BAB)    : 바스켓 베타 낮은 품목 롱 · 높은 품목 숏, 베타 중립 (Frazzini-Pedersen 2014)
    C1 헤징 압력 (COT)  : 상업 헤저 순매도(헤지 수요) 큰 품목 롱 · 작은 품목 숏 (Basu-Miffre 2013), 8품목
    C2 쏠림 역발상 (COT): 투기 펀드 순매수 z 최고(쏠림) 숏 · 최저 롱 — '모두가 산 승자 숏'의 원자재판, 8품목
  공통: 4주마다 재조정(횡단면), 상위·하위 1/3 동일가중 롱숏(달러 중립), 사전 목표 변동성 10%(직전 26주 실현, 최대 4배), 편도 비용 5bp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import ls_strategies as LS

SECTOR = {"yf:ZC=F": "곡물", "yf:ZW=F": "곡물", "yf:ZS=F": "곡물", "yf:ZM=F": "곡물", "yf:ZL=F": "곡물",
          "yf:CL=F": "에너지", "yf:HO=F": "에너지", "yf:RB=F": "에너지", "yf:NG=F": "에너지",
          "yf:HG=F": "금속", "yf:GC=F": "금속", "yf:SI=F": "금속", "yf:PL=F": "금속",
          "yf:KC=F": "소프트", "yf:SB=F": "소프트", "yf:CT=F": "소프트", "yf:CC=F": "소프트",
          "yf:LE=F": "축산", "yf:HE=F": "축산"}
ANN = 52


# ============================================================================
# 칼만 필터
# ============================================================================
def kalman_trend(lp: pd.Series, q_level: float = 0.1, q_slope: float = 0.001) -> pd.DataFrame:
    """국소 선형 추세 모형 (주간 로그가격):  y_t = μ_t + ε,  μ_t = μ_{t−1} + ν_{t−1} + η,  ν_t = ν_{t−1} + ζ
    잡음 분산은 관측 분산 R(직전 52주 주간 수익률 분산, 그 시점까지)에 대한 비율 q로 고정 — 사후 조정 없음. 반환 = 필터 기울기·t값."""
    y = lp.values; n = len(y)
    r2 = pd.Series(np.diff(y, prepend=np.nan)).rolling(52, min_periods=26).var().values
    x = np.array([y[0] if np.isfinite(y[0]) else 0.0, 0.0]); P = np.eye(2) * 1.0
    F = np.array([[1.0, 1.0], [0.0, 1.0]]); H = np.array([1.0, 0.0])
    slope, tval = np.full(n, np.nan), np.full(n, np.nan)
    for t in range(n):
        R = r2[t] if np.isfinite(r2[t]) and r2[t] > 0 else 1e-3
        Q = np.diag([q_level * R, q_slope * R])
        x = F @ x; P = F @ P @ F.T + Q
        if np.isfinite(y[t]):
            S = H @ P @ H + R; K = P @ H / S
            x = x + K * (y[t] - H @ x); P = P - np.outer(K, H) @ P
        slope[t] = x[1]; tval[t] = x[1] / np.sqrt(max(P[1, 1], 1e-12))
    return pd.DataFrame({"slope": slope, "t": tval}, index=lp.index)


def kalman_trend_weights(P: pd.DataFrame, q_slope: float = 0.001, vol_win: int = 26) -> pd.DataFrame:
    R = LS._ret(P)
    sig = pd.DataFrame({c: np.sign(kalman_trend(np.log(P[c]).ffill(), q_slope=q_slope)["t"]) for c in P}).where(P.notna())
    vol = R.rolling(vol_win, min_periods=13).std()
    W = sig / vol
    return W.div(W.abs().sum(1).replace(0, np.nan), axis=0).fillna(0.0)


def kalman_pair(P: pd.DataFrame, a: str, b: str, delta: float = 1e-4, z_in: float = 1.5, z_out: float = 0.25, max_hold: int = 13) -> pd.DataFrame:
    """동적 헤지 페어 (Chan 2013): ln A_t = α_t + β_t ln B_t + e_t, [α, β] 랜덤워크(칼만). z = 예측오차 / √예측분산.
    z > z_in → A 비쌈: A 숏·B 롱(β만큼), 부호 반전·|z| < z_out·max_hold 주 경과 시 청산."""
    la, lb = np.log(P[a]), np.log(P[b])
    ok = la.notna() & lb.notna()
    la, lb = la[ok], lb[ok]
    n = len(la); th = np.zeros(2); Pm = np.eye(2); Vw = delta / (1 - delta) * np.eye(2)
    Ve = float(la.diff().iloc[:53].var() or 1e-3)              # 관측 잡음 = 처음 52주로만 추정 (전체 기간 분산은 미래 정보)
    z, beta = np.full(n, np.nan), np.full(n, np.nan)
    for t in range(n):
        h = np.array([1.0, lb.iloc[t]])
        Pm = Pm + Vw
        yhat = h @ th; Q = h @ Pm @ h + Ve
        e = la.iloc[t] - yhat
        if t > 52:
            z[t] = e / np.sqrt(Q); beta[t] = th[1]
        K = Pm @ h / Q; th = th + K * e; Pm = Pm - np.outer(K, h) @ Pm
    pos = np.zeros(n); cur = 0.0; held = 0
    for t in range(n):
        if not np.isfinite(z[t]):
            cur = 0.0
        elif cur != 0:
            held += 1
            if abs(z[t]) < z_out or np.sign(z[t]) == cur or held >= max_hold:
                cur = 0.0; held = 0
        if cur == 0 and np.isfinite(z[t]) and abs(z[t]) > z_in:
            cur = -np.sign(z[t]); held = 0
        pos[t] = cur
    bt = np.clip(np.nan_to_num(beta, nan=1.0), 0.2, 5.0)
    return pd.DataFrame({a: pos / (1 + bt), b: -pos * bt / (1 + bt)}, index=la.index)


# ============================================================================
# 횡단면 롱숏 (상위·하위 1/3)
# ============================================================================
def xs_weights(S: pd.DataFrame, hold: int = 4, frac: float = 1 / 3, sector_neutral: bool = False, inv_vol: pd.DataFrame | None = None) -> pd.DataFrame:
    """신호 S(date×id, 클수록 롱) → hold주마다 상위 frac 롱 · 하위 frac 숏, 롱·숏 각 0.5 (섹터 중립이면 섹터마다 1개씩)."""
    W = pd.DataFrame(0.0, index=S.index, columns=S.columns)
    cur = pd.Series(0.0, index=S.columns); last = None
    for t, dt in enumerate(S.index):
        if last is None or t - last >= hold:
            s = S.loc[dt].dropna()
            w = pd.Series(0.0, index=S.columns)
            if sector_neutral:
                for sec in set(SECTOR.get(i) for i in s.index):
                    ss = s[[i for i in s.index if SECTOR.get(i) == sec]]
                    if len(ss) >= 2:
                        w[ss.idxmax()] += 1.0; w[ss.idxmin()] -= 1.0
            elif len(s) >= 6:
                k = max(1, int(round(len(s) * frac)))
                o = s.sort_values()
                w[o.index[-k:]] = 1.0; w[o.index[:k]] = -1.0
            if inv_vol is not None:
                w = w * inv_vol.loc[dt].reindex(w.index).fillna(0)
            if w.abs().sum() > 0:
                lw, sw = w.clip(lower=0).sum(), -w.clip(upper=0).sum()
                w = w.clip(lower=0) / (2 * lw if lw else 1) + w.clip(upper=0) / (2 * sw if sw else 1)
                cur = w; last = t
        W.loc[dt] = cur
    return W


def cot_signals(L: pd.DataFrame, P: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """COT (화요일 기준, 금요일 발표) → 1주 늦춰 사용. 헤징 압력 = 상업 순포지션/미결제 (음수 클수록 헤지 매도 많음 → 롱),
    쏠림 = 투기 순포지션/미결제의 3년 z (클수록 과열 → 숏)."""
    from pathlib import Path
    import analytics as A
    p = Path(__file__).parent / "data" / "cot.parquet"
    if not p.exists():
        return pd.DataFrame(), pd.DataFrame()
    C = pd.read_parquet(p); C["date"] = pd.to_datetime(C["date"])
    hp, mz = {}, {}
    for i, nm in A.COT_MAP.items():
        if i not in P:
            continue
        c = C[C.name == nm].set_index("date").sort_index()
        if c.empty:
            continue
        comm = (c.comm_net / c.open_interest_all).resample("W-FRI").last().shift(1)
        mm = (c.mm_net / c.open_interest_all).resample("W-FRI").last().shift(1)
        hp[i] = -comm.reindex(P.index).ffill(limit=2)
        mz[i] = -((mm - mm.rolling(156, min_periods=52).mean()) / mm.rolling(156, min_periods=52).std()).reindex(P.index).ffill(limit=2)
    return pd.DataFrame(hp), pd.DataFrame(mz)


# ============================================================================
# 전체 실행
# ============================================================================
def run_extra(L: pd.DataFrame, cost_bp: float = 5.0, start: str = "2001-01-01", split: str = "2017-01-01", tgt: float = 10.0) -> dict:
    P = LS.weekly(L, LS.BASKET, start); P = P[[c for c in LS.BASKET if c in P]]; R = LS._ret(P)
    lp = np.log(P)
    vol26 = R.rolling(26, min_periods=13).std()
    S = {}

    def book(W, Rr=R):
        return LS._book(LS._vol_target(W, Rr, tgt), Rr, cost_bp)

    def rescale(r):                     # 수익률 단계에서 사전 목표 변동성 (직전 26주 실현, 최대 4배)
        pv = r.rolling(26, min_periods=13).std().shift(1) * np.sqrt(ANN)
        return r * (tgt / pv).clip(upper=4.0).fillna(0.0)
    # 기준: 52주 모멘텀 · MinVar 70/30
    tsm = book(LS.tsmom_weights(P, 52, port_tgt=None))
    mvb = book(LS.alloc_weights_cap(P, 0.25))
    S["기준 · 52주 모멘텀 (④)"] = tsm
    S["기준 · MinVar 롱 (⑥)"] = mvb
    # 칼만
    kt_slow = book(kalman_trend_weights(P, 0.001)); kt_fast = book(kalman_trend_weights(P, 0.01))
    S["K1 칼만 추세 (느림, q=0.001)"] = kt_slow
    S["K1 칼만 추세 (빠름, q=0.01)"] = kt_fast
    kp = []
    for a, b in LS.PAIRS:
        if a in P and b in P:
            W = kalman_pair(P, a, b)
            kp.append(LS._book(W, R[[a, b]], cost_bp))
    if kp:
        S["K2 칼만 동적 헤지 페어 (12쌍)"] = rescale(pd.concat(kp, axis=1).fillna(0).mean(1))
    # 횡단면 (섹터·가치·왜도·변동성·베타)
    S["S1 섹터 중립 모멘텀 (섹터 내 승자 롱·패자 숏)"] = book(xs_weights(lp.diff(26), sector_neutral=True))
    S["S2 섹터 내 따라잡기 (과열 숏·낙오 롱)"] = book(xs_weights(-lp.diff(26), sector_neutral=True))
    S["V1 가치 (5년 평균 대비 싼 품목 롱)"] = book(xs_weights(-(lp - lp.rolling(260, min_periods=156).mean())))
    S["W1 왜도 (음의 왜도 롱 · 양의 왜도 숏)"] = book(xs_weights(-R.rolling(52, min_periods=40).skew()))
    S["L1 저변동성 (낮은 변동성 롱 · 높은 변동성 숏)"] = book(xs_weights(-vol26, inv_vol=1 / vol26))
    mkt = R.mean(1)
    beta = R.apply(lambda c: c.rolling(52, min_periods=40).cov(mkt)).div(mkt.rolling(52, min_periods=40).var(), axis=0)
    Wb = xs_weights(-beta)
    bl = (Wb.clip(lower=0) * beta).sum(1); bs = (-Wb.clip(upper=0) * beta).sum(1)
    Wb = Wb.clip(lower=0).mul((bs / bl.replace(0, np.nan)).clip(0.25, 4).fillna(1), axis=0) + Wb.clip(upper=0)   # 롱 쪽을 키워 베타 중립
    S["B1 저베타 BAB (베타 중립)"] = book(Wb)
    hp, mz = cot_signals(L, P)
    if not hp.empty:
        Pc = P[hp.columns]; Rc = R[hp.columns]
        S["C1 헤징 압력 (COT, 8품목)"] = LS._book(LS._vol_target(xs_weights(hp, frac=0.34), Rc, tgt), Rc, cost_bp)
        S["C2 쏠림 역발상 (COT 투기 과열 숏, 8품목)"] = LS._book(LS._vol_target(xs_weights(mz, frac=0.34), Rc, tgt), Rc, cost_bp)
    # MinVar 혼합에 칼만 추세 투입
    def mix(parts):
        m = sum(w * s.reindex(mvb.index).fillna(0) for w, s in parts)
        pv = m.rolling(26, min_periods=13).std().shift(1) * np.sqrt(ANN)
        return m * (tgt / pv).clip(upper=4.0).fillna(1.0)
    S["혼합 · MinVar 70 + 모멘텀 30 (기존)"] = mix([(0.7, mvb), (0.3, tsm)])
    S["혼합 · MinVar 70 + 칼만 추세 30"] = mix([(0.7, mvb), (0.3, kt_slow)])
    S["혼합 · MinVar 70 + 모멘텀 15 + 칼만 15"] = mix([(0.7, mvb), (0.15, tsm), (0.15, kt_slow)])
    # 분산 효과 점검: 위 대체 전략 전부를 같은 위험으로 (사전 선택 없이)
    alt = [k for k in S if k[:2] in ("K1", "K2", "S1", "S2", "V1", "W1", "L1", "B1", "C1", "C2")]
    S["다전략 · 대체 전략 전부 동일 위험 (사전 선택 없음)"] = mix([(1 / len(alt), S[k]) for k in alt])
    S["다전략 · MinVar 50 + 모멘텀 25 + 대체 전략 전부 25"] = mix([(0.5, mvb), (0.25, tsm), (0.25, S["다전략 · 대체 전략 전부 동일 위험 (사전 선택 없음)"])])
    v1, s2 = S["V1 가치 (5년 평균 대비 싼 품목 롱)"], S["S2 섹터 내 따라잡기 (과열 숏·낙오 롱)"]
    S["혼합 · MinVar 50 + 모멘텀 25 + 가치 25 (사후 선택 주의)"] = mix([(0.5, mvb), (0.25, tsm), (0.25, v1)])
    S["혼합 · MinVar 50 + 모멘텀 25 + 따라잡기 25 (사후 선택 주의)"] = mix([(0.5, mvb), (0.25, tsm), (0.25, s2)])
    D = pd.DataFrame(S)
    D = D[D.index >= pd.Timestamp("2007-01-01")]
    base = D["혼합 · MinVar 70 + 모멘텀 30 (기존)"]
    rows = []
    for c in D:
        r = D[c].fillna(0)
        dlt = r - base
        rows.append({"전략": c, **LS.metrics(r), f"Sharpe ~{split[:4]}": LS.metrics(r[r.index < split])["Sharpe"],
                     f"Sharpe {split[:4]}~": LS.metrics(r[r.index >= split])["Sharpe"], "t값": LS.metrics(r)["Sharpe"] * np.sqrt(len(r) / ANN),
                     "기존 70/30과 상관": float(r.corr(base)), "52주 모멘텀과 상관": float(r.corr(D["기준 · 52주 모멘텀 (④)"])),
                     "기존 70/30 대비 차이 t": float(dlt.mean() / dlt.std() * np.sqrt(len(dlt))) if dlt.std() > 0 else np.nan})
    # 사전 선택 검증: 혼합 후보(기존 70/30, MinVar 50 + 모멘텀 25 + 대체 전략 하나 25)를 학습(~split)으로만 고르고 시험(split~) 확인
    sh = lambda r: float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else np.nan
    cands = {"혼합 · MinVar 70 + 모멘텀 30 (기존)": D["혼합 · MinVar 70 + 모멘텀 30 (기존)"]}
    for a in alt:
        cands["혼합 · MinVar 50 + 모멘텀 25 + " + a.split(" (")[0] + " 25"] = mix([(0.5, mvb), (0.25, tsm), (0.25, S[a])]).reindex(D.index)
    sel = pd.DataFrame([{"후보": k, f"학습 Sharpe (~{int(split[:4]) - 1})": sh(v[v.index < split]), f"시험 Sharpe ({split[:4]}~)": sh(v[v.index >= split])}
                        for k, v in cands.items()]).set_index("후보").sort_values(f"학습 Sharpe (~{int(split[:4]) - 1})", ascending=False)
    return dict(table=pd.DataFrame(rows).set_index("전략"), returns=D, selection=sel, cands=cands)
