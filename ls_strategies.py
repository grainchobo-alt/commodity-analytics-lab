"""롱숏 전략 백테스트 — 상관·레짐·추세 (헤지펀드에서 흔히 쓰는 원자재 전략 묶음)

  주간(금 종가) 결정 → 다음 주 수익률로 손익 (그 시점까지의 정보만 사용), 비용 = 비중 변화량 × cost_bp (편도)
  ① 페어 평균회귀 (통계적 차익, Gatev-Goetzmann-Rouwenhorst 2006 · Vidyamurthy 2004)
     스프레드 s = ln A − β ln B (β = 과거 104주 OLS), z = (s − 52주 평균) / 52주 표준편차
     |z| > z_in 이면 비싼 쪽 숏·싼 쪽 롱 → |z| < z_out 또는 max_hold 주 경과 또는 |z| > z_stop 이면 청산
     필터: (b) 진입 시 52주 수익률 상관 > ρ_min (관계가 살아 있을 때만), (c) + 스트레스 국면 아닐 때만 (위기엔 스프레드가 더 벌어짐)
  ② 상관 급등 위험회피 (흡수비율 AR, Kritzman-Li-Page-Rigobon 2011)
     AR = 52주 상관행렬의 1주성분 설명비중. ΔAR(4주) z > 1 → 동반 하락 위험 → 바스켓 롱을 끔(또는 숏)
  ③ 국면 진입 위험회피 (다변량 HMM 스트레스 국면, regime_mv)
     매년 재추정한 HMM의 'VIX 최고 국면' 실시간 확률 > 0.5 → 바스켓 끔 / 금 롱 · 경기민감 바스켓 숏
  ④ 시계열 모멘텀 (TSMOM, Moskowitz-Ooi-Pedersen 2012) — 품목별 sign(52주 수익률) × (목표 변동성 / 26주 변동성)
  ⑤ 횡단면 모멘텀 (Asness-Moskowitz-Pedersen 2013 · Miffre-Rallis 2007) — 26주 수익률 상위 1/3 롱 · 하위 1/3 숏, 4주마다 교체
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASKET = ["yf:ZC=F", "yf:ZW=F", "yf:ZS=F", "yf:ZM=F", "yf:ZL=F", "yf:CL=F", "yf:HO=F", "yf:RB=F", "yf:NG=F",
          "yf:HG=F", "yf:GC=F", "yf:SI=F", "yf:PL=F", "yf:KC=F", "yf:SB=F", "yf:CT=F", "yf:CC=F", "yf:LE=F", "yf:HE=F"]
PAIRS = [("yf:ZC=F", "yf:ZW=F"), ("yf:ZW=F", "yf:KE=F"), ("yf:ZS=F", "yf:ZM=F"), ("yf:ZS=F", "yf:ZL=F"), ("yf:ZM=F", "yf:ZL=F"),
         ("yf:CL=F", "yf:HO=F"), ("yf:CL=F", "yf:RB=F"), ("yf:HO=F", "yf:RB=F"), ("yf:CL=F", "yf:BZ=F"),
         ("yf:GC=F", "yf:SI=F"), ("yf:GC=F", "yf:PL=F"), ("yf:LE=F", "yf:GF=F")]
CYCLICAL = ["yf:CL=F", "yf:HG=F", "yf:ZS=F", "yf:CT=F"]
SAFE = "yf:GC=F"
ANN = 52


def weekly(L: pd.DataFrame, ids: list[str], start="2001-01-01") -> pd.DataFrame:
    P = L[[i for i in ids if i in L]].resample("W-FRI").last()
    return P[P.index >= pd.Timestamp(start)]


def _ret(P: pd.DataFrame) -> pd.DataFrame:
    return 100 * np.log(P).diff()


def _book(W: pd.DataFrame, R: pd.DataFrame, cost_bp: float) -> pd.Series:
    """비중 W_t (t 금요일 결정) × 다음 주 수익률 R_{t+1} − 비용. 결과 index = 손익이 실현되는 주."""
    W = W.reindex(R.index).fillna(0.0)
    gross = (W.shift(1) * R).sum(1)
    tc = W.diff().abs().sum(1).shift(1).fillna(0) * cost_bp / 100
    return (gross - tc).iloc[1:]


# ============================================================================
# ① 페어 평균회귀
# ============================================================================
def pair_weights(P: pd.DataFrame, a: str, b: str, z_in=2.0, z_out=0.5, z_stop=4.0, max_hold=13, beta_win=104, z_win=52,
                 rho_min: float | None = None, stress: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    la, lb = np.log(P[a]), np.log(P[b])
    ok = la.notna() & lb.notna()
    la, lb = la[ok], lb[ok]
    cov = la.rolling(beta_win, min_periods=beta_win // 2).cov(lb); var = lb.rolling(beta_win, min_periods=beta_win // 2).var()
    beta = (cov / var).clip(0.2, 5.0)
    s = la - beta * lb
    z = (s - s.rolling(z_win, min_periods=26).mean()) / s.rolling(z_win, min_periods=26).std()
    rho = la.diff().rolling(52, min_periods=26).corr(lb.diff())
    st = stress.reindex(z.index).fillna(0).values > 0.5 if stress is not None else np.zeros(len(z), bool)
    pos = np.zeros(len(z)); cur = 0.0; held = 0
    zv, rv, bv = z.values, rho.values, beta.values
    for t in range(len(z)):
        if not np.isfinite(zv[t]) or not np.isfinite(bv[t]):
            cur = 0.0; held = 0
        elif cur != 0:
            held += 1
            if abs(zv[t]) < z_out or held >= max_hold or abs(zv[t]) > z_stop or np.sign(zv[t]) == cur:
                cur = 0.0; held = 0
        if cur == 0 and np.isfinite(zv[t]) and abs(zv[t]) > z_in and abs(zv[t]) <= z_stop:
            okr = rho_min is None or (np.isfinite(rv[t]) and rv[t] > rho_min)
            if okr and not st[t]:
                cur = -np.sign(zv[t]); held = 0          # z > 0: A 비쌈 → A 숏 · B 롱
        pos[t] = cur
    pos = pd.Series(pos, index=z.index)
    W = pd.DataFrame({a: pos / (1 + beta), b: -pos * beta / (1 + beta)}).fillna(0.0)
    return W, z


def pairs_book(P: pd.DataFrame, cost_bp=5.0, rho_min=None, stress=None, **kw) -> tuple[pd.Series, pd.DataFrame]:
    R = _ret(P); books, stats = {}, []
    for a, b in PAIRS:
        if a not in P or b not in P:
            continue
        W, _ = pair_weights(P, a, b, rho_min=rho_min, stress=stress, **kw)
        r = _book(W, R[[a, b]], cost_bp)
        books[f"{a.split(':')[1]}/{b.split(':')[1]}"] = r
        act = (W.abs().sum(1) > 0)
        stats.append({"페어": f"{a.split(':')[1]}/{b.split(':')[1]}", "보유 비율": float(act.mean()),
                      "거래 수": int((act.astype(int).diff() == 1).sum()), **metrics(r)})
    B = pd.DataFrame(books)
    return B.mean(1), pd.DataFrame(stats)


# ============================================================================
# ② 흡수비율 · ③ HMM 스트레스 · 바스켓
# ============================================================================
def absorption(R: pd.DataFrame, win=52) -> pd.Series:
    out = pd.Series(np.nan, index=R.index)
    for t in range(win, len(R) + 1):
        X = R.iloc[t - win:t].dropna(axis=1, how="any")
        if X.shape[1] < 4:
            continue
        ev = np.linalg.eigvalsh(np.corrcoef(X.values.T))
        out.iloc[t - 1] = ev[-1] / ev.sum()
    return out


def ar_signal(R: pd.DataFrame, win=52, d=4, zwin=156, thr=1.0) -> tuple[pd.Series, pd.Series]:
    ar = absorption(R, win)
    dz = (ar - ar.shift(d)) / (ar - ar.shift(d)).rolling(zwin, min_periods=52).std()
    return (dz > thr).astype(float).where(dz.notna(), 0.0), ar


def basket_weights(P: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    ids = [i for i in ids if i in P]
    avail = P[ids].notna()
    return avail.div(avail.sum(1).replace(0, np.nan), axis=0).fillna(0.0)


def stress_prob(L: pd.DataFrame, start="2001-01-01", K: int = 3) -> pd.Series:
    """다변량 HMM(원자재 6 + 거시 8)에서 'VIX 평균이 가장 높은 국면'의 실시간 확률 (매년 재추정, 전방 필터)."""
    import regime_mv as RM
    X = RM.build_panel(L, RM.COMM_W_DEFAULT, RM.MACRO_DEFAULT, "W", start)
    W = RM.walk_forward(X, "fred:VIXCLS", K, "full", "W", 2)
    return W[f"p{K}"].astype(float)


# ============================================================================
# ④ 시계열 모멘텀 · ⑤ 횡단면 모멘텀
# ============================================================================
def tsmom_weights(P: pd.DataFrame, look=52, vol_win=26, tgt_vol=10.0, port_tgt: float | None = 10.0, max_lev=4.0) -> pd.DataFrame:
    """품목별 sign(look주 수익률) × (목표 변동성 / 품목 변동성) / 품목 수 → (선택) 포트폴리오 전체를 사전 목표 변동성 port_tgt %로 배율.
    배율 = port_tgt / (직전 26주 실현 포트폴리오 변동성) — t 시점까지의 손익만 사용 (사후 배율 아님)."""
    R = _ret(P)
    sig = np.sign(np.log(P).diff(look))
    vol = R.rolling(vol_win, min_periods=13).std() * np.sqrt(ANN)
    W = sig * (tgt_vol / vol)
    n = W.notna().sum(1).replace(0, np.nan)
    W = W.div(n, axis=0).clip(-2, 2).fillna(0.0)
    if port_tgt:
        rb = (W.shift(1) * R).sum(1)
        pv = rb.rolling(26, min_periods=13).std() * np.sqrt(ANN)
        W = W.mul((port_tgt / pv).clip(upper=max_lev).fillna(0.0), axis=0)
    return W


def xsmom_weights(P: pd.DataFrame, look=26, hold=4, frac=1 / 3) -> pd.DataFrame:
    mom = np.log(P).diff(look)
    W = pd.DataFrame(0.0, index=P.index, columns=P.columns)
    last = None
    for t, dt in enumerate(P.index):
        if last is None or t - last >= hold:
            m = mom.loc[dt].dropna()
            if len(m) >= 6:
                k = max(1, int(len(m) * frac))
                o = m.sort_values()
                w = pd.Series(0.0, index=P.columns)
                w[o.index[-k:]] = 0.5 / k; w[o.index[:k]] = -0.5 / k
                last = t; cur = w
            else:
                cur = pd.Series(0.0, index=P.columns)
        W.loc[dt] = cur
    return W


# ============================================================================
# 평가
# ============================================================================
def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 26 or r.std() == 0:
        return {"Sharpe": np.nan, "연수익 (%)": np.nan, "연변동성 (%)": np.nan, "최대낙폭 (%)": np.nan}
    cum = r.cumsum()
    return {"Sharpe": float(r.mean() / r.std() * np.sqrt(ANN)), "연수익 (%)": float(r.mean() * ANN),
            "연변동성 (%)": float(r.std() * np.sqrt(ANN)), "최대낙폭 (%)": float((cum - cum.cummax()).min())}


def placebo_timing(on: pd.Series, r_on: pd.Series, r_off: pd.Series, n=300, block=13, seed=0) -> float:
    """타이밍 위약: 켬/끔 신호를 13주 블록으로 섞어(빈도·지속성 유지) 같은 전략 Sharpe의 95% 분위."""
    m = on.notna() & r_on.notna() & r_off.notna()
    s, a, b = on[m].values, r_on[m].values, r_off[m].values
    rng = np.random.default_rng(seed); nb = int(np.ceil(len(s) / block)); out = []
    for _ in range(n):
        ss = np.concatenate([s[k * block:(k + 1) * block] for k in rng.permutation(nb)])[:len(s)]
        x = np.where(ss > 0, a, b)
        out.append(x.mean() / x.std() * np.sqrt(ANN) if x.std() > 0 else np.nan)
    return float(np.nanquantile(out, 0.95))


def run_all(L: pd.DataFrame, cost_bp: float = 5.0, start="2001-01-01", split="2017-01-01", with_hmm: bool = True,
            progress=None) -> dict:
    ids = list(dict.fromkeys(BASKET + [x for p in PAIRS for x in p]))
    P = weekly(L, ids, start); R = _ret(P)
    Pb = P[[i for i in BASKET if i in P]]; Rb = R[Pb.columns]
    Wb = basket_weights(Pb, list(Pb.columns))
    base = _book(Wb, Rb, cost_bp)
    S, legs = {"기준: 바스켓 롱 (동일가중)": base}, {}
    # ① 페어
    S["① 페어 평균회귀 (무조건)"], legs["pairs_a"] = pairs_book(P, cost_bp)
    S["① 페어 + 상관 필터 (ρ>0.5)"], legs["pairs_b"] = pairs_book(P, cost_bp, rho_min=0.5)
    if progress: progress(0.3)
    # 스트레스: VIX 3년 분위 > 80% (단순) / HMM
    vix = L["fred:VIXCLS"].resample("W-FRI").last().reindex(P.index).ffill() if "fred:VIXCLS" in L else None
    v_st = (vix.rolling(156, min_periods=52).rank(pct=True) > 0.8).astype(float) if vix is not None else None
    if v_st is not None:
        S["① 페어 + 상관 + 평시만 (VIX)"], legs["pairs_c"] = pairs_book(P, cost_bp, rho_min=0.5, stress=v_st)
    # ② 흡수비율
    ar_on, ar = ar_signal(Rb)
    S["② 상관 급등 시 바스켓 끔 (AR)"] = _book(Wb.mul(1 - ar_on.reindex(Wb.index).fillna(0), axis=0), Rb, cost_bp)
    S["② 상관 급등 시 바스켓 숏 (AR)"] = _book(Wb.mul(1 - 2 * ar_on.reindex(Wb.index).fillna(0), axis=0), Rb, cost_bp)
    if progress: progress(0.5)
    # ③ HMM 스트레스
    ps = None
    if with_hmm:
        try:
            ps = stress_prob(L, start).reindex(P.index)
        except Exception:
            ps = None
    hm_on = (ps > 0.5).astype(float).where(ps.notna(), 0.0) if ps is not None else None
    if hm_on is not None:
        S["③ 스트레스 국면 시 바스켓 끔 (HMM)"] = _book(Wb.mul(1 - hm_on.reindex(Wb.index), axis=0), Rb, cost_bp)
        cy = [i for i in CYCLICAL if i in P]
        Wro = pd.DataFrame(0.0, index=P.index, columns=list(dict.fromkeys(cy + [SAFE])))
        Wro.loc[:, SAFE] = 0.5 * hm_on; Wro.loc[:, cy] = (-0.5 / len(cy)) * hm_on.values[:, None]
        S["③ 스트레스 국면 시 금 롱 · 경기민감 숏"] = _book(Wro, R[Wro.columns], cost_bp)
        if v_st is not None:
            Wv = Wro * 0
            Wv.loc[:, SAFE] = 0.5 * v_st; Wv.loc[:, cy] = (-0.5 / len(cy)) * v_st.values[:, None]
            S["③' VIX 고분위 시 금 롱 · 경기민감 숏"] = _book(Wv, R[Wv.columns], cost_bp)
    if progress: progress(0.8)
    # ④ ⑤ 모멘텀
    S["④ 시계열 모멘텀 52주 (TSMOM, 목표 변동성 10%)"] = _book(tsmom_weights(Pb, 52), Rb, cost_bp)
    S["④ 시계열 모멘텀 13주 (목표 변동성 10%)"] = _book(tsmom_weights(Pb, 13), Rb, cost_bp)
    S["⑤ 횡단면 모멘텀 26주 (상위−하위 1/3)"] = _book(xsmom_weights(Pb, 26), Rb, cost_bp)
    # ⑥ 비중 조합: 최소분산 롱 + 모멘텀 롱숏 (두 책 모두 목표 변동성 10%, 섞은 뒤 다시 10%)
    mvb = _book(_vol_target(alloc_weights_cap(Pb, 0.25), Rb), Rb, cost_bp)
    S["⑥ 최소분산 롱 (MinVar, 목표 변동성 10%)"] = mvb
    tsm = S["④ 시계열 모멘텀 52주 (TSMOM, 목표 변동성 10%)"]
    for a_ in (0.3, 0.5, 0.7):
        mix = (a_ * mvb + (1 - a_) * tsm.reindex(mvb.index).fillna(0)).fillna(0)
        pv = mix.rolling(26, min_periods=13).std().shift(1) * np.sqrt(ANN)
        S[f"⑥ 혼합 · MinVar {a_:.0%} + 모멘텀 {1 - a_:.0%}"] = mix * (10.0 / pv).clip(upper=4.0).fillna(1.0)
    # ⑦ 국면 전환 오버레이: 평소 = 혼합 MinVar 70% + 모멘텀 30%, 신호가 켜진 다음 주에는 다른 구성으로 (신호는 t 결정 → t+1 적용)
    base = S["⑥ 혼합 · MinVar 70% + 모멘텀 30%"]
    alts = {"모멘텀 100%": tsm.reindex(base.index).fillna(0), "현금 (포지션 0)": base * 0}
    cy_ = [i for i in CYCLICAL if i in P]
    Wsp = pd.DataFrame(0.0, index=P.index, columns=list(dict.fromkeys(cy_ + [SAFE])))
    Wsp.loc[:, SAFE] = 0.5; Wsp.loc[:, cy_] = -0.5 / len(cy_)
    spread = _book(_vol_target(Wsp, R[Wsp.columns]), R[Wsp.columns], cost_bp).reindex(base.index).fillna(0)   # 항상 켜진 금/경기민감 책 (10%)
    alts["혼합 50% + 금 롱·경기민감 숏 50%"] = 0.5 * base + 0.5 * spread
    for sig_nm, on_ in [("스트레스 (HMM)", hm_on), ("상관 급등 (AR)", ar_on)]:
        if on_ is None:
            continue
        o1 = on_.reindex(base.index).fillna(0).shift(1).fillna(0) > 0
        for alt_nm, alt in alts.items():
            S[f"⑦ 평소 70/30 → {sig_nm} 때 {alt_nm}"] = pd.Series(np.where(o1, alt, base), index=base.index)
    # 공통 평가 구간
    D = pd.DataFrame(S)
    first = D.apply(lambda c: c.ne(0).idxmax() if c.ne(0).any() else D.index[0]).max()
    D = D[D.index >= max(first, pd.Timestamp(start) + pd.DateOffset(years=5))]
    rows = []
    for c in D.columns:
        r = D[c]
        rows.append({"전략": c, **metrics(r), f"Sharpe ~{split[:4]}": metrics(r[r.index < split])["Sharpe"],
                     f"Sharpe {split[:4]}~": metrics(r[r.index >= split])["Sharpe"], "보유 주 비율": float((r != 0).mean()),
                     "바스켓과 상관": float(r.corr(D.iloc[:, 0]))})
    T = pd.DataFrame(rows).set_index("전략")
    T["t값"] = T["Sharpe"] * np.sqrt(len(D) / ANN)
    # 타이밍 전략 위약
    T["위약 95%"] = np.nan
    rb = D.iloc[:, 0]
    if ar_on is not None:
        T.loc["② 상관 급등 시 바스켓 끔 (AR)", "위약 95%"] = placebo_timing(1 - ar_on.reindex(D.index).shift(1).fillna(0), rb, rb * 0)
    if hm_on is not None:
        T.loc["③ 스트레스 국면 시 바스켓 끔 (HMM)", "위약 95%"] = placebo_timing(1 - hm_on.reindex(D.index).shift(1).fillna(0), rb, rb * 0)
    b70 = D["⑥ 혼합 · MinVar 70% + 모멘텀 30%"]
    for sig_nm, on_ in [("스트레스 (HMM)", hm_on), ("상관 급등 (AR)", ar_on)]:
        for alt_nm, alt in [("모멘텀 100%", D["④ 시계열 모멘텀 52주 (TSMOM, 목표 변동성 10%)"]), ("현금 (포지션 0)", b70 * 0)]:
            k_ = f"⑦ 평소 70/30 → {sig_nm} 때 {alt_nm}"
            if on_ is not None and k_ in T.index:                    # 위약 = 같은 빈도·지속성의 무작위 전환
                T.loc[k_, "위약 95%"] = placebo_timing(1 - on_.reindex(D.index).shift(1).fillna(0), b70, alt)
    grp = {"바스켓 (동일가중)": list(Pb.columns), "경기민감 (원유·구리·대두·면화)": CYCLICAL, "금": [SAFE]}
    ev = {}
    if hm_on is not None:
        ev["HMM 스트레스 국면 진입"] = event_study(hm_on, R, grp)
    ev["상관 급등 (AR) 시작"] = event_study(ar_on, R, grp)
    if v_st is not None:
        ev["VIX 3년 80% 분위 돌파"] = event_study(v_st, R, grp)
    if progress: progress(1.0)
    return dict(table=T, returns=D, pairs=legs, ar=ar, stress=ps, vix_stress=v_st, n_years=len(D) / ANN, events=ev, P=P)


def event_study(on: pd.Series, R: pd.DataFrame, groups: dict, hs=(1, 4, 8, 13), min_gap=8) -> pd.DataFrame:
    """국면 진입 초기: 신호가 0→1로 바뀐 주(직전 min_gap주 동안 꺼져 있던 경우만) 이후 h주 누적 수익률(%) 평균 · t값.
    groups = {이름: [id, ...]} (동일가중)."""
    s = on.reindex(R.index).fillna(0).values
    ent = [t for t in range(min_gap, len(s)) if s[t] > 0 and s[t - min_gap:t].max() == 0]
    rows = []
    for nm, ids in groups.items():
        g = R[[i for i in ids if i in R]].mean(1)
        for h in hs:
            x = np.array([g.iloc[t + 1:t + 1 + h].sum() for t in ent if t + h < len(g)])
            uncond = g.rolling(h).sum().shift(-h).dropna()
            if len(x) >= 3:
                rows.append({"묶음": nm, "h주": h, "진입 횟수": len(x), "진입 후 평균 (%)": x.mean(), "평소 평균 (%)": uncond.mean(),
                             "차이 t값": (x.mean() - uncond.mean()) / (x.std(ddof=1) / np.sqrt(len(x))) if x.std() > 0 else np.nan,
                             "하락 비율": float((x < 0).mean())})
    return pd.DataFrame(rows)


# ============================================================================
# ⑥ 캐리 (기간구조, Koijen-Moskowitz-Pedersen-Vrugt 2018 · Erb-Harvey 2006)
#    캐리 = (ln F1 − ln F2) × 12 / 만기 간격(개월) — 양수 = 백워데이션(근월물이 비쌈, 재고 부족) → 롱
# ============================================================================
def load_carry() -> pd.DataFrame:
    from pathlib import Path
    p = Path(__file__).parent / "data" / "carry.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def carry_panel(C: pd.DataFrame, L: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """주간 캐리(연율 %) · 근월물 보유 수익률(%) 패널. 중국 = 계약 식별로 월물 교체를 반영한 수익률, 미국 에너지 = Yahoo 롤 보정 수익률."""
    K, R = {}, {}
    for i, g in C.groupby("id"):
        g = g.sort_values("date").set_index("date")
        w = g.resample("W-FRI").last().dropna(subset=["f1", "f2"])
        K[i] = 100 * (np.log(w.f1) - np.log(w.f2)) * 12 / w.months.clip(lower=1)
        if "c1" in w and w["c1"].notna().any():
            same = w.c1 == w.c1.shift(1); rolled = w.c1 == w.c2.shift(1)
            r = np.where(same, np.log(w.f1 / w.f1.shift(1)), np.where(rolled, np.log(w.f1 / w.f2.shift(1)), np.nan))
            R[i] = pd.Series(100 * r, index=w.index)
        elif i in L:
            R[i] = 100 * np.log(L[i].resample("W-FRI").last()).diff()
    return pd.DataFrame(K), pd.DataFrame(R)


def carry_books(K: pd.DataFrame, R: pd.DataFrame, cost_bp=5.0, vol_win=26) -> dict:
    R = R.reindex(K.index); K = K.where(R.notna())
    vol = R.rolling(vol_win, min_periods=13).std()
    inv = (1 / vol).where(K.notna())
    ts = np.sign(K) * inv
    ts = ts.div(ts.abs().sum(1).replace(0, np.nan), axis=0)                       # 총노출 1
    rk = K.rank(axis=1); dm = rk.sub(rk.mean(1), axis=0)
    xs = dm.div(dm.abs().sum(1).replace(0, np.nan), axis=0)                        # 순위 편차 비중 (롱=숏)
    lo = (K.notna() * inv).div((K.notna() * inv).sum(1).replace(0, np.nan), axis=0)  # 기준: 위험 균등 롱
    return {"기준 롱 (위험 균등)": _book(lo.fillna(0), R.fillna(0), cost_bp), "캐리 시계열 (백워 롱 · 콘탱고 숏)": _book(ts.fillna(0), R.fillna(0), cost_bp),
            "캐리 횡단면 (상대 캐리 순위)": _book(xs.fillna(0), R.fillna(0), cost_bp)}


def carry_ic(K: pd.DataFrame, R: pd.DataFrame, h: int = 4) -> pd.DataFrame:
    """캐리와 다음 h주 수익률의 관계: 품목별 상관(IC)·백워/콘탱고 구간의 평균 수익률."""
    rows = []
    for i in K:
        k = K[i]; f = R[i].rolling(h).sum().shift(-h).reindex(k.index)
        m = k.notna() & f.notna()
        if m.sum() < 52:
            continue
        rows.append({"id": i, "기간": f"{k[m].index.min():%Y}–{k[m].index.max():%Y}", "주 수": int(m.sum()), "평균 캐리 (연 %)": float(k[m].mean()),
                     "백워데이션 비율": float((k[m] > 0).mean()), f"IC (캐리 vs 다음 {h}주)": float(np.corrcoef(k[m], f[m])[0, 1]),
                     f"백워 시 {h}주 평균 (%)": float(f[m][k[m] > 0].mean()) if (k[m] > 0).any() else np.nan,
                     f"콘탱고 시 {h}주 평균 (%)": float(f[m][k[m] <= 0].mean()) if (k[m] <= 0).any() else np.nan})
    return pd.DataFrame(rows)


def run_carry(L: pd.DataFrame, cost_bp=5.0, split="2017-01-01") -> dict:
    C = load_carry()
    if C.empty:
        return {}
    out = {}
    for nm, ids in [("미국 에너지 4 (EIA, 2001–2024.04)", ["yf:CL=F", "yf:HO=F", "yf:RB=F", "yf:NG=F"]),
                    ("중국 대련·상해 8 (Sina, 2018–)", ["ak:M0", "ak:Y0", "ak:C0", "ak:A0", "ak:I0", "ak:J0", "ak:JM0", "ak:RB0"])]:
        Ci = C[C.id.isin(ids)]
        if Ci.empty:
            continue
        K, R = carry_panel(Ci, L)
        if "yf:CL=F" in ids:
            K = K[K.index >= "2001-01-01"]
        B = pd.DataFrame(carry_books(K, R, cost_bp)).dropna(how="all")
        B = B[B.index >= B.index.min() + pd.DateOffset(weeks=26)]
        rows = []
        for c in B:
            r = B[c].fillna(0)
            rows.append({"전략": c, **metrics(r), "t값": metrics(r)["Sharpe"] * np.sqrt(len(r) / ANN),
                         "바스켓과 상관": float(r.corr(B.iloc[:, 0].fillna(0)))})
        out[nm] = dict(table=pd.DataFrame(rows).set_index("전략"), returns=B, ic=carry_ic(K, R), carry=K)
    return out


# ============================================================================
# ⑦ 비중 결정 방식 비교 — 동일가중 · 역변동성 · 최소분산(MinVar) · 평균-분산(μ = 52주 모멘텀)
#    공분산 = 과거 104주 Ledoit-Wolf 수축, 4주마다 재조정, 모두 사전 목표 변동성 10%로 배율 (같은 위험에서 비교)
# ============================================================================
def _vol_target(W: pd.DataFrame, R: pd.DataFrame, tgt=10.0, max_lev=4.0) -> pd.DataFrame:
    rb = (W.shift(1) * R).sum(1)
    pv = rb.rolling(26, min_periods=13).std() * np.sqrt(ANN)
    return W.mul((tgt / pv).clip(upper=max_lev).fillna(0.0), axis=0)


def alloc_weights(P: pd.DataFrame, method: str, cov_win=104, reb=4, look=52, lam=5.0) -> pd.DataFrame:
    """reb = 재조정 주기(주). 모멘텀 계열은 신호가 매주 바뀌므로 run_alloc에서 1주로 돌림."""
    from scipy.optimize import minimize
    from sklearn.covariance import LedoitWolf
    R = _ret(P); mom = np.sign(np.log(P).diff(look))
    W = pd.DataFrame(np.nan, index=P.index, columns=P.columns)
    for t in range(cov_win, len(P), reb):
        win = R.iloc[t - cov_win + 1:t + 1]
        cols = [c for c in P.columns if win[c].notna().sum() >= cov_win * 0.8 and np.isfinite(mom[c].iloc[t])]
        if len(cols) < 4:
            continue
        X = win[cols].fillna(0.0).values
        S = LedoitWolf().fit(X / 100).covariance_; sd = np.sqrt(np.diag(S)); n = len(cols)   # 소수 단위 (주간)
        sgn = mom[cols].iloc[t].values
        mu = sgn * sd * 0.05                                            # μ = IC(0.05) × σ × 추세 부호 (Grinold 공식)
        if method == "eq":
            w = np.ones(n) / n
        elif method == "iv":
            w = (1 / sd) / (1 / sd).sum()
        elif method in ("minvar", "mv_lo"):
            obj = (lambda w: 1e4 * (w @ S @ w)) if method == "minvar" else (lambda w: 1e4 * (lam / 2 * w @ S @ w - w @ mu))   # 1e4 = 수치 안정
            jac = (lambda w: 1e4 * 2 * S @ w) if method == "minvar" else (lambda w: 1e4 * (lam * S @ w - mu))
            res = minimize(obj, np.ones(n) / n, jac=jac, bounds=[(0, 0.25)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                           method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
            x = np.clip(res.x, 0, 0.25)
            w = x / x.sum() if (abs(res.x.sum() - 1) < 1e-3 and np.isfinite(res.x).all()) else np.ones(n) / n
        elif method == "tsmom":
            w = sgn / sd                                                # 부호 / σ (역변동성)
            w = w / np.abs(w).sum()
        elif method == "mv_ls":
            w = np.linalg.solve(S + 1e-8 * np.eye(n), mu)              # Σ⁻¹μ (상관까지 고려한 모멘텀)
            w = w / np.abs(w).sum()
        else:
            raise ValueError(method)
        W.loc[P.index[t], cols] = w
        W.loc[P.index[t], [c for c in P.columns if c not in cols]] = 0.0
    return W.ffill().fillna(0.0)


ALLOC = {"eq": "동일가중 롱", "iv": "역변동성 롱 (위험 균등 근사)", "minvar": "최소분산 롱 (MinVar)",
         "mv_lo": "최소분산 + μ 모멘텀 (평균-분산 롱온리)", "tsmom": "모멘텀 롱숏 · 역변동성 (④ TSMOM)", "mv_ls": "모멘텀 롱숏 · Σ⁻¹μ (상관 반영)"}


def run_alloc(L: pd.DataFrame, cost_bp=5.0, start="2001-01-01", split="2017-01-01", tgt=10.0) -> dict:
    P = weekly(L, BASKET, start); P = P[[c for c in BASKET if c in P]]; R = _ret(P)
    D = {}
    for k, nm in ALLOC.items():
        Wk = tsmom_weights(P, 52, port_tgt=None) if k == "tsmom" else alloc_weights(P, k, reb=1 if k == "mv_ls" else 4)
        W = _vol_target(Wk, R, tgt) if tgt else Wk
        D[nm] = _book(W, R, cost_bp)
    D = pd.DataFrame(D)
    D = D[D.index >= D.index[0] + pd.DateOffset(years=3)]
    D = D.loc[D.ne(0).all(1).idxmax():]
    rows = []
    for c in D:
        r = D[c]
        rows.append({"방식": c, **metrics(r), f"Sharpe ~{split[:4]}": metrics(r[r.index < split])["Sharpe"],
                     f"Sharpe {split[:4]}~": metrics(r[r.index >= split])["Sharpe"], "t값": metrics(r)["Sharpe"] * np.sqrt(len(r) / ANN)})
    return dict(table=pd.DataFrame(rows).set_index("방식"), returns=D)


def run_alloc_robust(L: pd.DataFrame, cost_bp=5.0, start="2001-01-01", split="2017-01-01", tgt=10.0) -> pd.DataFrame:
    """MinVar 강건성: (a) 생우·금·돈육을 하나씩/모두 빼기 (b) 품목 상한 25% → 15% (c) MinVar 롱 50% + 모멘텀 롱숏 50% 혼합."""
    P0 = weekly(L, BASKET, start); P0 = P0[[c for c in BASKET if c in P0]]
    rows, books = [], {}

    def mv(P, cap=0.25):
        import ls_strategies as _me
        W = _me.alloc_weights_cap(P, cap)
        return _book(_vol_target(W, _ret(P), tgt), _ret(P), cost_bp)
    books["MinVar (상한 25%, 기준)"] = mv(P0)
    for drop, nm in [(["yf:LE=F"], "생우 제외"), (["yf:GC=F"], "금 제외"), (["yf:HE=F"], "돈육 제외"), (["yf:LE=F", "yf:GC=F", "yf:HE=F"], "생우·금·돈육 모두 제외")]:
        books[f"MinVar — {nm}"] = mv(P0.drop(columns=[c for c in drop if c in P0]))
    books["MinVar (상한 15%)"] = mv(P0, 0.15)
    books["MinVar (상한 10%)"] = mv(P0, 0.10)
    R0 = _ret(P0)
    tsm = _book(tsmom_weights(P0, 52), R0, cost_bp)
    books["모멘텀 롱숏 (TSMOM 10%)"] = tsm
    for a_ in (0.3, 0.5, 0.7):                                   # 위험 기여 비율 (두 책 모두 10% 변동성)
        mix = a_ * books["MinVar (상한 25%, 기준)"] + (1 - a_) * tsm
        pv = mix.rolling(26, min_periods=13).std().shift(1) * np.sqrt(ANN)
        books[f"혼합: MinVar {a_:.0%} + 모멘텀 {1 - a_:.0%}"] = mix * (tgt / pv).clip(upper=4.0).fillna(1.0)
    D = pd.DataFrame(books)
    D = D[D.index >= D.index[0] + pd.DateOffset(years=3)]
    D = D.loc[D.ne(0).all(1).idxmax():]
    for c in D:
        r = D[c]
        rows.append({"구성": c, **metrics(r), f"Sharpe ~{split[:4]}": metrics(r[r.index < split])["Sharpe"],
                     f"Sharpe {split[:4]}~": metrics(r[r.index >= split])["Sharpe"], "t값": metrics(r)["Sharpe"] * np.sqrt(len(r) / ANN),
                     "MinVar와 상관": float(r.corr(D["MinVar (상한 25%, 기준)"]))})
    return pd.DataFrame(rows).set_index("구성"), D


def alloc_weights_cap(P: pd.DataFrame, cap: float = 0.25, cov_win=104, reb=4) -> pd.DataFrame:
    """최소분산 롱온리 (품목 상한 cap)."""
    from scipy.optimize import minimize
    from sklearn.covariance import LedoitWolf
    R = _ret(P)
    W = pd.DataFrame(np.nan, index=P.index, columns=P.columns)
    for t in range(cov_win, len(P), reb):
        win = R.iloc[t - cov_win + 1:t + 1]
        cols = [c for c in P.columns if win[c].notna().sum() >= cov_win * 0.8]
        if len(cols) < 4:
            continue
        S = LedoitWolf().fit(win[cols].fillna(0.0).values / 100).covariance_; n = len(cols)
        capn = max(cap, 1.0 / n + 1e-6)
        res = minimize(lambda w: 1e4 * (w @ S @ w), np.ones(n) / n, jac=lambda w: 1e4 * 2 * S @ w, bounds=[(0, capn)] * n,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
        x = np.clip(res.x, 0, capn)
        w = x / x.sum() if (abs(res.x.sum() - 1) < 1e-3 and np.isfinite(res.x).all()) else np.ones(n) / n
        W.loc[P.index[t], cols] = w
        W.loc[P.index[t], [c for c in P.columns if c not in cols]] = 0.0
    return W.ffill().fillna(0.0)
