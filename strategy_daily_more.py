"""나머지 3축 실험(A · B · E · F · J · K · G · I)을 일간 자료로 — 주간 실험과 같은 규칙, 창은 달력 기간이 같게 ×5.
(C 환경 적응형 · D 현금 피신은 strategy_daily_signals.py, H 손실 한도 경보는 운영 도구라 백테스트가 아니므로 제외)

재료: 일간 3축(MinVar · 52주 모멘텀 · 가치, strategy_daily_freq D1). 기준 = 고정 50/25/25(일간 학습 0.61 / 시험 1.32).
A 연도별 최적 비중 갈아타기(최대 Sharpe · 최소분산 · 역변동성 · 축소 × 직전 3 · 5년) · 트리거 T1 성과 하락 · T2 흡수비율 위기 · T3 전략 모멘텀 · T4 투자선 이탈 — 평가 2012~ (학습 2012–16)
B 메타 레이어 6개로 다음 65일(13주) 1등 축 예측(릿지, 매년 재학습) → 기울이기 κ
E 목표 변동성 동적(계단 · 연속, 신호 6종), 이자 관점 B(선물 구조: 자본 전체가 이자를 받아 현금 피신해도 추가 이자 없음)만 — 관점 A는 선물 포트폴리오에 맞지 않아 제외
F 상관 급등기 모멘텀 확대 · 상승 추세 집중 (곡물 상관 · DCC 동조성 · 흡수비율 트리거 × 방향 조건)
J 짧은 추세(20 · 40 · 65 · 130일) 상승 집중 · K 모멘텀 기간(20 ~ 252일 · 여러 기간 평균)
G · I 품목군 상한(곡물 · 에너지 · 축산) — 품목 단위 합산 후 상한, 일간 비중
공통: 설정 선택은 학습(2008–16, A는 2012–16)에서만, 비용 편도 5bp, 일간 Sharpe √252 · 주간 합산 Sharpe(√52) 병기.
실행: .venv\\Scripts\\python.exe strategy_daily_more.py [A B E F JK GI] → data/strategy_daily_more_compare.csv
"""
from __future__ import annotations

import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import analytics as A  # noqa: E402
import ls_extra as LX  # noqa: E402
import ls_strategies as LS  # noqa: E402
import strategy_cash as SC  # noqa: E402
import strategy_daily_freq as DF  # noqa: E402
import strategy_daily_signals as SDS  # noqa: E402
import strategy_graincap as GC  # noqa: E402
import strategy_groupcap as GP  # noqa: E402
import strategy_meta as SM  # noqa: E402
import strategy_switch as SS  # noqa: E402

ANN, COST, GROSS, TGT = 252, 5.0, 1.15, 10.0
VW, START = 130, "2005-01-01"
BASE, AX = np.array([0.5, 0.25, 0.25]), SM.AXES
rw = SDS.run_weights_d


def sc(r, base, lo="2008", e=None):
    return SDS.score(r, base, e, lo)


def fixed(S):
    return SS.const(S, dict(zip(AX, BASE)))


def turn(W, S):
    Wf = W.reindex(S.index).ffill().fillna(0.0)
    return float(Wf.diff().abs().sum(1).loc["2008":].sum() / (len(Wf.loc["2008":]) / ANN))


def book(W, R):
    return LS._book(W, R, COST).loc[START:].fillna(0.0)


# ------------------------------------------------------------------ A 갈아타기
def opt_d(X, kind):
    from scipy.optimize import minimize
    mu, C, k = X.mean().values * ANN, X.cov().values * ANN, X.shape[1]
    w0, b, c = np.full(k, 1 / k), [(0, 1)] * k, [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if kind == "invvol":
        w = 1 / np.sqrt(np.diag(C)); return w / w.sum()
    f = (lambda w: -(w @ mu) / np.sqrt(w @ C @ w)) if kind == "maxsharpe" else (lambda w: w @ C @ w)
    r = minimize(f, w0, bounds=b, constraints=c, method="SLSQP")
    return r.x if r.success else w0


def fgap(X, w):
    from scipy.optimize import minimize
    mu, C, k = X.mean().values * ANN, X.cov().values * ANN, X.shape[1]; v = float(np.sqrt(w @ C @ w))
    r = minimize(lambda x: -(x @ mu), w, bounds=[(0, 1)] * k, constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}, {"type": "ineq", "fun": lambda x: v ** 2 - x @ C @ x}], method="SLSQP")
    return max(0.0, float(-r.fun - w @ mu)) if r.success else 0.0


def yearly_d(S, kind, years, shrink):
    Wt = pd.DataFrame(np.nan, index=S.index, columns=S.columns)
    for y in range(S.index.year.min() + 1, S.index.year.max() + 1):
        d = S.index[S.index < f"{y}-01-01"][-1]; X = S.loc[d - pd.DateOffset(years=years):d]
        if len(X) < ANN * years * 0.9:
            continue
        Wt.loc[d] = (1 - shrink) * opt_d(X, kind) + shrink * BASE
    first = Wt.dropna().index.min(); Wt.loc[:first] = Wt.loc[:first].fillna(pd.Series(BASE, S.columns))
    return Wt


def t1_d(S, thr, years=5):
    Wt = pd.DataFrame(np.nan, index=S.index, columns=S.columns); w = BASE.copy(); last = -999
    for t in range(len(S)):
        if t >= VW and t - last >= 65 and S.index[t] >= S.index[0] + pd.DateOffset(years=years):
            m = S.iloc[t - VW + 1:t + 1].values @ w
            if m.std() > 0 and m.mean() / m.std() * np.sqrt(ANN) < thr:
                w = opt_d(S.loc[S.index[t] - pd.DateOffset(years=years):S.index[t]], "maxsharpe"); last = t
        Wt.iloc[t] = w
    return Wt


def t3_d(S, k, look=260, reb=20):
    Wt = pd.DataFrame(np.nan, index=S.index, columns=S.columns)
    for t in range(look, len(S), reb):
        top = S.iloc[t - look + 1:t + 1].sum().nlargest(k).index
        Wt.iloc[t] = [1 / k if c in top else 0.0 for c in S.columns]
    return Wt.fillna(Wt.bfill().iloc[0])


def t4_d(S, gap, years=5, check=20, min_gap=130):
    Wt = pd.DataFrame(np.nan, index=S.index, columns=S.columns); w = BASE.copy(); last = -999
    for t in range(len(S)):
        d = S.index[t]
        if t % check == 0 and t - last >= min_gap and d >= S.index[0] + pd.DateOffset(years=years):
            X = S.loc[d - pd.DateOffset(years=years):d]
            if fgap(X, w) > gap:
                w = opt_d(X, "maxsharpe"); last = t
        Wt.iloc[t] = w
    return Wt


def run_A(S, sig):
    base = rw(S, fixed(S)); lo = "2012"; rows, store = [], {"고정 50/25/25": base}
    rows.append({"규칙": "고정 50/25/25 (기준)", "묶음": "기준", **sc(base, base, lo)})
    cands = {"고정 동일 비중": ("기준", SS.const(S, {a: 1 / 3 for a in AX}))}
    for years in (3, 5):
        for kind, nm in (("maxsharpe", "최대 Sharpe(접점)"), ("minvar", "최소분산"), ("invvol", "역변동성")):
            cands[f"연 1회 · 직전 {years}년 {nm}"] = (f"연 1회 {years}년", yearly_d(S, kind, years, 0.0))
        cands[f"연 1회 · 직전 {years}년 축소(고정 50% + 접점 50%)"] = (f"연 1회 {years}년", yearly_d(S, "maxsharpe", years, 0.5))
    for thr in (-0.5, 0.0, 0.5):
        cands[f"T1 성과 트리거 · 130일 Sharpe < {thr}"] = ("T1 성과", t1_d(S, thr))
    z = sig["흡수비율"]
    for thr in (1.0, 1.5, 2.0):
        W = fixed(S).copy(); on = (z.reindex(S.index).ffill() > thr); W.loc[on] = [1.0, 0.0, 0.0]
        cands[f"T2 위기 트리거 · 흡수비율 z > {thr}"] = ("T2 위기", W)
    for k in (1, 2):
        cands[f"T3 전략 모멘텀 · 상위 k = {k}"] = ("T3 전략 모멘텀", t3_d(S, k))
    for gap in (1.0, 2.0, 4.0):
        cands[f"T4 투자선 이탈 · 갭 %p > {gap}"] = ("T4 투자선 이탈", t4_d(S, gap))
    res = {}
    for nm, (grp, W) in cands.items():
        r = rw(S, W); res[nm] = (grp, r, sc(r, base, lo), turn(W, S)); store[nm] = r
    best = {}
    for nm, (grp, r, s_, _) in res.items():
        if grp != "기준" and (grp.split(" ")[0] not in best or s_["Sharpe 학습 2008–16"] > res[best[grp.split(" ")[0]]][2]["Sharpe 학습 2008–16"]):
            best[grp.split(" ")[0]] = nm
    for nm, (grp, r, s_, tn) in res.items():
        rows.append({"규칙": nm, "묶음": grp, **s_, "연 비중 변경 (|Δw| 합)": tn, "학습 최고": "←" if best.get(grp.split(" ")[0]) == nm else ""})
    return rows, store


# ------------------------------------------------------------------ B 메타 레이어
def layers_d(L, R, P, S):
    idx = S.index
    ar = LS.absorption(R, 252); d20 = ar - ar.shift(20); dz = d20 / d20.rolling(780, min_periods=260).std()
    bv = R.std(1).rolling(130, min_periods=65).mean(); vreg = bv / bv.rolling(780, min_periods=260).median()
    lp = np.log(P); dev = lp - lp.rolling(1300, min_periods=780).mean()
    Mx = L[["fred:VIXCLS", "yf:DX-Y.NYB", "fred:DGS10"]].reindex(P.index).ffill()
    F = pd.DataFrame({"L1 흡수비율": ar, "L1 흡수비율 20일 z": dz, "L2 변동성 국면": vreg, "L3 추세 강도": lp.diff(252).abs().mean(1),
                      "L4 가치 스프레드": dev.quantile(0.8, axis=1) - dev.quantile(0.2, axis=1), "VIX": Mx["fred:VIXCLS"],
                      "달러 130일": np.log(Mx["yf:DX-Y.NYB"]).diff(130) * 100, "10년 금리 130일": Mx["fred:DGS10"].diff(130)})
    for a in AX:
        F[f"L5 {a} 65일"] = S[a].rolling(65).sum(); F[f"L5 {a} 252일"] = S[a].rolling(252).sum()
    return F.reindex(idx).ffill()


def walk_predict_d(F, Y, alpha=10.0, first=2008):
    from sklearn.linear_model import Ridge
    pred = pd.DataFrame(np.nan, index=F.index, columns=Y.columns)
    for yr in range(first, F.index.max().year + 1):
        cut = pd.Timestamp(f"{yr - 1}-12-31"); tr = F.index[F.index <= cut - pd.Timedelta(days=91)]
        Xtr = F.loc[tr].dropna(); ok = Xtr.index.intersection(Y.dropna().index)
        if len(ok) < 520:
            continue
        mu, sd = Xtr.loc[ok].mean(), Xtr.loc[ok].std().replace(0, 1)
        te = F.index[(F.index > cut) & (F.index <= pd.Timestamp(f"{yr}-12-31"))]; Xte = ((F.loc[te] - mu) / sd).fillna(0)
        for a in Y.columns:
            pred.loc[te, a] = Ridge(alpha=alpha).fit(((Xtr.loc[ok] - mu) / sd).values, Y.loc[ok, a].values).predict(Xte.values)
    return pred


def run_B(L, P, R, S):
    base = rw(S, fixed(S)); rows, store = [{"규칙": "고정 50/25/25 (기준)", "κ": 0.0, **sc(base, base)}], {"고정 50/25/25": base}
    F = layers_d(L, R, P, S); fwd = S[::-1].rolling(65).sum()[::-1].shift(-1); Y = fwd.sub(fwd.mean(1), axis=0)
    pred = walk_predict_d(F, Y)
    hit = {}
    for nm, sl in (("학습 2008–16", slice("2008", "2016")), ("시험 2017–", slice("2017", None))):
        d = pred.loc[sl].dropna().index[::65]; ok = [pred.loc[x].idxmax() == Y.loc[x].idxmax() for x in d if Y.loc[x].notna().all()]
        hit[nm] = float(np.mean(ok))
    for kap in (0.05, 0.1, 0.2, 0.3):
        r = rw(S, SM.tilt(pred, BASE, kap, reb=20)); rows.append({"규칙": f"메타 레이어 기울이기 κ={kap}", "κ": kap, **sc(r, base), "1등 축 적중률 학습": hit["학습 2008–16"], "1등 축 적중률 시험": hit["시험 2017–"]}); store[f"κ={kap}"] = r
    best = max((x for x in rows[1:]), key=lambda x: x["Sharpe 학습 2008–16"]); best["학습 최고"] = "←"
    return rows, store


# ------------------------------------------------------------------ E 목표 변동성 동적
def run_E(S, sig):
    base = rw(S, fixed(S)); idx = S.index; MID = {"HMM": 0.7, "변동성": 1.5, "동조성": 0.9, "흡수비율": 1.5, "VIX": 0.9, "뉴스": 1.0}
    n = sum((sig[k] > v).astype(int) for k, v in MID.items())
    def apply(e):
        e = e.reindex(idx).ffill().fillna(1.0); held = e.shift(1).fillna(1.0)
        return held * base - held.diff().abs().fillna(0) * GROSS * 0.05          # 노출 변경 비용: |Δe| × 1.15 × 5bp (수익률 % 단위)
    rows, store = [{"규칙": "고정 목표 10% (기준)", "설정": "", **sc(base, base)}], {"고정 10%": base}
    cands = {}
    for hi in (10, 12):
        cands[f"계단형 (0개 {hi}% · 1개 10% · 2개 7% · 3개+ 5%)"] = pd.Series(np.select([n == 0, n == 1, n == 2], [hi, 10, 7], 5) / 10.0, idx)
    Z = pd.concat([(s - s.expanding(520).mean()) / s.expanding(520).std() for s in sig.values()], axis=1).mean(1)
    for b in (0.1, 0.2, 0.3, 0.5):
        cands[f"연속형 β={b}"] = (1 - b * Z.fillna(0)).clip(0.5, 1.3)
    for c in (0.7, 0.5):
        cands[f"고정 목표 {int(c * 10)}% (대조군)"] = pd.Series(c, idx)
    res = {nm: (apply(e), e) for nm, e in cands.items()}
    for nm, (r, e) in res.items():
        ctrl = apply(pd.Series(float(e.reindex(base.loc["2008":].index).mean()), idx))
        s_ = sc(r, base, e=e); s_["같은 평균 노출 고정 낙폭 (%)"] = sc(ctrl, base)["최대 낙폭 (%)"]
        rows.append({"규칙": nm, "설정": "", **s_}); store[nm] = r
    sub = [x for x in rows[1:] if not x["규칙"].startswith("고정")]
    max(sub, key=lambda x: x["Sharpe 학습 2008–16"])["학습 최고"] = "←"
    return rows, store


# ------------------------------------------------------------------ F · J · K (축 변형)
def mom_axis(P, R, looks):
    lp = np.log(P); sig = sum(np.sign(lp.diff(l)) for l in looks) / len(looks)
    vol = R.rolling(VW, min_periods=VW // 2).std() * np.sqrt(ANN)
    W = (sig * (TGT / vol)); W = W.div(W.notna().sum(1).replace(0, np.nan), axis=0).clip(-2, 2).fillna(0.0)
    return book(DF.vt(DF.sample(W, 5), R, VW, ANN, 5), R)


def up_trend(P, R, n, n2, k, reb):
    lp = np.log(P); s1, s2 = lp.diff(n), (lp.diff(n2) if n2 else None); v = R.rolling(VW, min_periods=VW // 2).std()
    W = pd.DataFrame(np.nan, index=P.index, columns=P.columns)
    for t in range(0, len(P), reb):
        ok = (s1.iloc[t] > 0) & ((s2.iloc[t] > 0) if s2 is not None else True)
        sc_ = (s1.iloc[t] / v.iloc[t])[ok].dropna(); W.iloc[t] = 0.0
        if len(sc_):
            top = sc_.nlargest(k).index; w = 1 / v.iloc[t][top]; W.loc[P.index[t], top] = (w / w.sum()).values
    return book(DF.vt(W.ffill().fillna(0.0), R, VW, ANN, 5), R)


def run_JK(P, R, S):
    base = rw(S, fixed(S)); rows, store = [], {}
    for nm, looks in [("20일", [20]), ("40일", [40]), ("65일", [65]), ("130일", [130]), ("195일", [195]), ("260일 (기준)", [260]), ("여러 기간 평균 (20·65·260일)", [20, 65, 260])]:
        x = mom_axis(P, R, looks).reindex(S.index).fillna(0.0); X = S.assign(모멘텀=x); r = rw(X, fixed(X))
        rows.append({"실험": "K 모멘텀 기간", "규칙": nm, "단독 학습": SDS.sh(x.loc["2008":"2016"]), "단독 시험": SDS.sh(x.loc["2017":]), **sc(r, base)}); store[f"K {nm}"] = r
    for n in (20, 40, 65, 130):
        for k in (2, 3, 5):
            x = up_trend(P, R, n, None, k, 5).reindex(S.index).fillna(0.0); X = S.assign(단기=x)
            r = rw(X, pd.DataFrame([[0.25, 0.0, 0.25, 0.5]] * len(X), index=X.index, columns=["MinVar", "모멘텀", "가치", "단기"]))
            rows.append({"실험": "J 짧은 추세", "규칙": f"상승 추세 집중 {n}일 · 상위 {k}개 (혼합 25/50/25)", "단독 학습": SDS.sh(x.loc["2008":"2016"]), "단독 시험": SDS.sh(x.loc["2017":]), **sc(r, base)}); store[f"J {n} {k}"] = r
    for g in ("K 모멘텀 기간", "J 짧은 추세"):
        sub = [x for x in rows if x["실험"] == g]; max(sub, key=lambda x: x["Sharpe 학습 2008–16"])["학습 최고"] = "←"
    return rows, store


COLS = ["MinVar", "MinVar 무제한", "모멘텀", "모멘텀 집중", "상승 추세 집중 3", "상승 추세 집중 2", "가치"]


def run_F(L, P, R, S, sig):
    X = pd.DataFrame({"MinVar": S["MinVar"], "MinVar 무제한": book(DF.vt(DF.minvar_w(P, R, 520, 20, cap=1.0), R, VW, ANN, 5), R),
                      "모멘텀": S["모멘텀"], "모멘텀 집중": None, "상승 추세 집중 3": up_trend(P, R, 130, 260, 3, 20), "상승 추세 집중 2": up_trend(P, R, 130, 260, 2, 20), "가치": S["가치"]}).reindex(S.index)
    lp = np.log(P); s = lp.diff(260); v = R.rolling(VW, min_periods=VW // 2).std(); scr = (s / v).abs(); W = pd.DataFrame(np.nan, index=P.index, columns=P.columns)
    for t in range(0, len(P), 20):
        row = scr.iloc[t].dropna()
        if len(row) >= 5:
            top = row.nlargest(5).index; w = np.sign(s.iloc[t][top]) / v.iloc[t][top]; W.iloc[t] = 0.0; W.loc[P.index[t], top] = (w / w.abs().sum()).values
    X["모멘텀 집중"] = book(DF.vt(W.ffill().fillna(0.0), R, VW, ANN, 5), R); X = X.fillna(0.0)
    base = rw(X, pd.DataFrame([[0.5, 0, 0.25, 0, 0, 0, 0.25]] * len(X), index=X.index, columns=COLS)); idx = X.index
    G = R[GC.GRAINS]; gc = pd.concat([G[a].rolling(VW).corr(G[b]) for i, a in enumerate(GC.GRAINS) for b in GC.GRAINS[i + 1:]], axis=1).mean(1)
    gpct = SC.expanding_pct(gc.reindex(idx).ffill(), 520); up = (G.mean(1).rolling(65).sum().reindex(idx) > 0)
    trig = {"곡물 상관 > 0.8 (방향 무관)": gpct > 0.8, "곡물 상관 > 0.9 (방향 무관)": gpct > 0.9, "곡물 상관 > 0.9 · 상승 동조": (gpct > 0.9) & up,
            "DCC 동조성 > 0.9": sig["동조성"] > 0.9, "흡수비율 z > 1.5": sig["흡수비율"] > 1.5}
    ons = {"모멘텀 확대 25/50/25": {"MinVar": .25, "모멘텀": .5, "가치": .25}, "모멘텀 100%": {"모멘텀": 1.0}, "모멘텀 집중 100%": {"모멘텀 집중": 1.0},
           "상승 추세 집중 3 100%": {"상승 추세 집중 3": 1.0}, "상승 추세 집중 2 100%": {"상승 추세 집중 2": 1.0}}
    off = {"MinVar": .5, "모멘텀": .25, "가치": .25}
    def Wof(on, w_on):
        a = np.array([off.get(c, 0.0) for c in COLS]); b = np.array([w_on.get(c, 0.0) for c in COLS])
        return pd.DataFrame(np.where(on.reindex(idx).fillna(False).values[:, None], b, a), index=idx, columns=COLS)
    rows, store = [{"규칙": "고정 50/25/25 (기준)", "트리거": "", "켜지면": "", **sc(base, base)}], {"고정 50/25/25": base}
    for wn, wd in ons.items():
        r = rw(X, pd.DataFrame([[wd.get(c, 0.0) for c in COLS]] * len(X), index=idx, columns=COLS))
        rows.append({"규칙": f"늘 {wn} (대조군)", "트리거": "항상", "켜지면": wn, **sc(r, base)})
    res = {}
    for tn, on in trig.items():
        for wn, wd in ons.items():
            r = rw(X, Wof(on, wd)); res[(tn, wn)] = (r, on); rows.append({"규칙": "트리거", "트리거": tn, "켜지면": wn, **sc(r, base), "켜진 날 (%)": float(on.reindex(idx).fillna(False).mean() * 100)})
    best = max(res, key=lambda k: sc(res[k][0], base)["Sharpe 학습 2008–16"])
    for x in rows:
        if x["트리거"] == best[0] and x["켜지면"] == best[1] and x["규칙"] == "트리거":
            x["학습 최고"] = "←"
            rng = np.random.default_rng(0); r0, on0 = res[best]; real = x["Sharpe 시험 2017–"]; blocks = [idx[i:i + 65] for i in range(0, len(idx), 65)]; pl = []
            for _ in range(200):
                o = rng.permutation(len(blocks)); ep = pd.concat([on0.reindex(blocks[j]).fillna(False).astype(bool) for j in o]).iloc[:len(idx)]; ep.index = idx
                pl.append(sc(rw(X, Wof(ep, ons[best[1]])), base)["Sharpe 시험 2017–"])
            x["위약 p (시험)"] = float((np.array(pl) >= real).mean())
    store["F 학습 최고"] = res[best][0]
    return rows, store


# ------------------------------------------------------------------ G · I 품목군 상한
def run_GI(P, R, S):
    f_ = lambda W: DF.vt(W, R, VW, ANN, 5)                                      # noqa: E731
    lp = np.log(P); vol = R.rolling(VW, min_periods=VW // 2).std() * np.sqrt(ANN)
    Wt = (np.sign(lp.diff(260)) * (TGT / vol)); Wt = Wt.div(Wt.notna().sum(1).replace(0, np.nan), axis=0).clip(-2, 2).fillna(0.0)
    W0 = 0.5 * f_(DF.minvar_w(P, R, 520, 20)) + 0.25 * f_(DF.sample(Wt, 5)) + 0.25 * f_(LX.xs_weights(-(lp - lp.rolling(1300, min_periods=780).mean()), hold=20))
    def run(W):
        raw = LS._book(W, R, COST); pv = raw.rolling(VW, min_periods=VW // 2).std().shift(1) * np.sqrt(ANN)
        return (raw * (TGT / pv).clip(upper=4.0).fillna(1.0)).loc["2005":]
    base = run(W0); rows, store = [], {"상한 없음": base}
    rules = {"없음 (기준)": {}, "곡물 50%": {"곡물": .5}, "곡물 35%": {"곡물": .35}, "곡물 25%": {"곡물": .25}, "에너지 25%": {"에너지": .25}, "에너지 15%": {"에너지": .15},
             "축산 25%": {"축산": .25}, "축산 15%": {"축산": .15}, "곡물 25% + 에너지 25% + 축산 25%": {"곡물": .25, "에너지": .25, "축산": .25},
             "곡물 25% + 에너지 15% + 축산 15%": {"곡물": .25, "에너지": .15, "축산": .15}}
    for nm, rule in rules.items():
        W = W0
        for g in ("곡물", "에너지", "축산"):
            if g in rule:
                W = GP.cap_group(W, [c for c in GP.GROUPS[g] if c in W], rule[g])
        r = run(W); g_ = [c for c in GC.GRAINS if c in W]; share = (W[g_].abs().sum(1) / W.abs().sum(1)).loc["2008":]
        rows.append({"상한": nm, **sc(r, base), "곡물 몫 평균 (%)": float(share.mean() * 100)}); store[nm] = r
    return rows, store


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    which = set(sys.argv[1:]) or {"A", "B", "E", "F", "JK", "GI"}
    t00 = time.time(); L = A.load_levels(roll_adjusted=True).loc[:SDS.FROZEN_END]; ids = [c for c in LS.BASKET if c in L]
    P = L[ids].loc["2001-01-01":]; P = P[P.index.dayofweek < 5].ffill(limit=5); R = LS._ret(P); S = DF.axes(P, R, ANN, 5, 5, 20, 20, 5)
    cache = pickle.load(open(SDS.CACHE, "rb")); sig = SDS.daily_signals(L, S, R, cache)
    print(f"준비 {time.time() - t00:.0f}s · 일간 3축 기준 학습 {SDS.sh(rw(S, fixed(S)).loc['2008':'2016']):.2f} 시험 {SDS.sh(rw(S, fixed(S)).loc['2017':]):.2f}", flush=True)
    allrows, store = [], {}
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 45)
    jobs = {"A": ("A 갈아타기", lambda: run_A(S, sig)), "B": ("B 메타 레이어", lambda: run_B(L, P, R, S)), "E": ("E 목표 변동성 동적", lambda: run_E(S, sig)),
            "F": ("F 상관 급등기 추세 집중", lambda: run_F(L, P, R, S, sig)), "JK": ("J·K 짧은 추세 · 모멘텀 기간", lambda: run_JK(P, R, S)), "GI": ("G·I 품목군 상한", lambda: run_GI(P, R, S))}
    for key, (title, fn) in jobs.items():
        if key not in which:
            continue
        t0 = time.time(); rows, st = fn()
        for r_ in rows:
            allrows.append({"실험": title, **r_})
        store.update({f"{key}|{k}": v for k, v in st.items()})
        T = pd.DataFrame(rows); cols = [c for c in ["규칙", "상한", "트리거", "켜지면", "Sharpe 학습 2008–16", "Sharpe 시험 2017–", "주간 합산 시험", "시험 차이 t", "최대 낙폭 (%)", "학습 최고", "위약 p (시험)"] if c in T]
        show = T[T.get("학습 최고", pd.Series("", index=T.index)).eq("←") | T.index.isin(T.head(1).index)] if key in ("A", "F", "JK") else T
        print(f"== {title} ({time.time() - t0:.0f}s)\n" + show[cols].round(2).to_string(index=False), flush=True)
    suffix = "" if which == {"A", "B", "E", "F", "JK", "GI"} else "_" + "".join(sorted(which))
    pd.DataFrame(allrows).to_csv(A.DATA / f"strategy_daily_more_compare{suffix}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({k: v.resample("W-FRI").sum() for k, v in store.items()}).to_parquet(A.DATA / f"strategy_daily_more_weekly{suffix}.parquet")
    print(f"저장 완료 · 총 {time.time() - t00:.0f}s")


if __name__ == "__main__":
    main()
