"""곡물 2단 온톨로지 + 역방향 인과 추론(결과 → 원인) 실험 — 기존 파일은 고치지 않음.

① 정책 이벤트 반사실 (Causal Impact 방식): 이벤트 전 1년간 '곡물 밖 품목'으로 대상 품목 수익률을 설명하는 회귀를 세우고,
   이벤트 뒤 실제 − 반사실 = 초과 수익. 이벤트가 없는 날 400번 같은 계산(위약) → p값.
② 온톨로지 구조 모형 (선형 SCM, 주간): 기상 → 작황 → WASDE 재고율 수정 → 가격, 대두 → 대두박·대두유(크러시),
   원유 → 옥수수·대두유(바이오연료), 옥수수 → 밀(사료 대체), 달러·헤알. 각 연결을 회귀로 추정하고
   가격 변동을 '어느 원인 충격이 어떤 경로로 왔나'로 정확히 분해(역사적 분해 = 역방향 추적).
③ PCMCI (편상관 검정, tigramite 없이 직접 구현): 자료만으로 시차 인과 연결을 찾아 ②의 온톨로지 연결과 대조.

실행: .venv\\Scripts\\python.exe causal_grain.py  → data/causal_events.csv · causal_scm_edges.csv · causal_attribution.csv · causal_pcmci.csv
"""
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import analytics as A
import ls_strategies as LS
import multimodal as MMD

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
DATA, MM = ROOT / "data", ROOT / "data" / "mm"
START = "2008-01-01"
PX = {"옥수수": "yf:ZC=F", "대두": "yf:ZS=F", "밀": "yf:ZW=F", "대두박": "yf:ZM=F", "대두유": "yf:ZL=F", "원유": "yf:CL=F"}
GRAIN = ["yf:ZC=F", "yf:ZW=F", "yf:ZS=F", "yf:ZM=F", "yf:ZL=F", "yf:KE=F"]


# ============================================================================
# 온톨로지 (2단) — 노드 · 부모(같은 주에 영향을 주는 원인)
# ============================================================================
ROOTS = ["미국 기상", "남미 기상", "흑해 기상", "원유", "달러 강세", "헤알 약세"]          # 바깥에서 오는 충격
PA = {
    "옥수수 작황": ["미국 기상"],
    "대두 작황": ["미국 기상"],
    "옥수수 재고율 수정": ["옥수수 작황"],
    "대두 재고율 수정": ["대두 작황"],
    "밀 재고율 수정": [],
    "옥수수": ["미국 기상", "남미 기상", "옥수수 작황", "옥수수 재고율 수정", "원유", "달러 강세"],
    "대두": ["미국 기상", "남미 기상", "대두 작황", "대두 재고율 수정", "헤알 약세", "달러 강세"],
    "밀": ["흑해 기상", "밀 재고율 수정", "옥수수", "달러 강세"],                       # 2단: 사료 대체
    "대두박": ["대두", "남미 기상"],                                                    # 2단: 크러시 · 아르헨티나 수출
    "대두유": ["대두", "원유"],                                                         # 2단: 크러시 · 바이오디젤
}
ORDER = ROOTS + list(PA)                                                                # 위상 순서 (부모가 항상 앞)
PRICES = ["옥수수", "대두", "밀", "대두박", "대두유"]
# 2판: 위성 식생지수(NOAA VHI) · 주간 수출 판매 · 해상 운임 추가 (causal_data.py로 수집)
ROOTS2 = ROOTS + ["미국 밀 위성", "해상 운임", "옥수수 수출 판매", "대두 수출 판매", "밀 수출 판매"]
PA2 = {"미국 위성": ["미국 기상"], "남미 위성": ["남미 기상"], "흑해 위성": ["흑해 기상"],
       "옥수수 작황": ["미국 기상", "미국 위성"], "대두 작황": ["미국 기상", "미국 위성"],
       "옥수수 재고율 수정": ["옥수수 작황", "옥수수 수출 판매"], "대두 재고율 수정": ["대두 작황", "대두 수출 판매"],
       "밀 재고율 수정": ["미국 밀 위성", "밀 수출 판매"],
       "옥수수": PA["옥수수"] + ["남미 위성", "옥수수 수출 판매", "해상 운임"],
       "대두": PA["대두"] + ["남미 위성", "대두 수출 판매", "해상 운임"],
       "밀": PA["밀"] + ["흑해 위성", "미국 밀 위성", "밀 수출 판매", "해상 운임"],
       "대두박": PA["대두박"] + ["남미 위성"], "대두유": PA["대두유"]}
GRAPHS = {1: (ROOTS, PA), 2: (ROOTS2, PA2)}


def set_graph(v: int):
    """온톨로지 판 바꾸기 (1 = 기상·작황·WASDE, 2 = + 위성·수출 판매·운임)."""
    global ROOTS, PA, ORDER
    ROOTS, PA = GRAPHS[v]
    ORDER = ROOTS + list(PA)
WREG = {"미국 기상": (["미국 아이오와", "미국 일리노이", "미국 네브래스카"], (6, 7, 8)),   # 미국 옥수수·대두 생육기
        "남미 기상": (["브라질 마투그로수", "아르헨티나 부에노스아이레스"], (12, 1, 2, 3)),
        "흑해 기상": (["러시아 로스토프", "우크라이나 헤르손"], (4, 5, 6))}


def _friday(d: pd.Series) -> pd.Series:
    return d.dt.to_period("W-FRI").dt.end_time.dt.normalize()


def weather_stress(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """주산지 '덥고 마름' 지수의 주간 변화 (생육기만, 그 밖은 0). 지수 = 4주 기온 z − 4주 강수 z,
    z = 같은 주차 이전 연도(최소 5년) 대비. 금요일에는 6일 전까지의 날씨만 앎(ERA5 지연)."""
    D = pd.read_parquet(MM / "openmeteo.parquet")
    D["date"] = pd.to_datetime(D.date)
    seen, wk = idx - pd.Timedelta(days=6), idx.isocalendar().week.values.astype(int)
    out = {}
    for nm, (regs, months) in WREG.items():
        zs = []
        for r in regs:
            g = D[D.region == r].set_index("date").sort_index()
            w = pd.DataFrame({"p": g.p.rolling(28, min_periods=20).sum(), "t": g.t.rolling(28, min_periods=20).mean()})
            w = w.reindex(w.index.union(seen)).ffill().reindex(seen)
            w.index = idx
            z = {}
            for c, floor in (("p", 5.0), ("t", 0.3)):
                df = pd.DataFrame({"v": w[c].values, "wk": wk}, index=idx)
                m = df.groupby("wk").v.transform(lambda s: s.shift(1).expanding(5).mean())
                sd = df.groupby("wk").v.transform(lambda s: s.shift(1).expanding(5).std())
                z[c] = (df.v - m) / sd.clip(lower=floor)
            zs.append(z["t"] - z["p"])
        s = pd.concat(zs, axis=1).mean(1)
        on = pd.Series(idx.month.isin(months), index=idx)
        out[nm] = s.diff().where(on & on.shift(1, fill_value=False)).fillna(0.0)
    return pd.DataFrame(out)


def crop_condition(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """NASS 주간 우량 비율(%)의 주간 변화 — 같은 시즌 안에서만 (시즌 첫 주·비시즌 = 0)."""
    g = MMD.nass_ge()
    out = {}
    for crop, nm in (("CORN", "옥수수 작황"), ("SOYBEANS", "대두 작황")):
        s = g[g.crop == crop].sort_values("avail")
        d = s["ge"].diff().where(s.avail.diff() <= pd.Timedelta(days=10))
        out[nm] = pd.Series(d.values, index=_friday(s.avail)).groupby(level=0).last()
    return pd.DataFrame(out).reindex(idx).fillna(0.0)


def wasde_surprise(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """WASDE 발표 주의 미국 기말재고율(재고/사용량) 수정 폭 (%p), 발표가 없는 주 = 0."""
    W = pd.read_parquet(DATA / "reports" / "wasde_features.parquet")
    W["d"] = pd.to_datetime(W.release).dt.tz_convert("America/New_York").dt.tz_localize(None)
    out = {}
    for c, nm in (("corn", "옥수수 재고율 수정"), ("soybean", "대두 재고율 수정"), ("wheat", "밀 재고율 수정")):
        s = W[W.commodity == c]
        out[nm] = pd.Series(s.d_stu.fillna(0.0).values, index=_friday(s.d)).groupby(level=0).sum()
    return pd.DataFrame(out).reindex(idx).fillna(0.0)


def fx(idx: pd.DatetimeIndex) -> pd.DataFrame:
    F = pd.read_parquet(MM / "fred_extra.parquet")
    F["date"] = pd.to_datetime(F.date)
    w = F.pivot_table(index="date", columns="id", values="value").resample("W-FRI").last().reindex(idx)
    return pd.DataFrame({"달러 강세": -100 * np.log(w["DEXUSEU"]).diff(),                 # 달러/유로 ↓ = 달러 강세
                         "헤알 약세": 100 * np.log(w["DEXBZUS"]).diff()}).fillna(0.0)    # 헤알/달러 ↑ = 헤알 약세


VHI_SEASON = {"미국 위성": (6, 7, 8, 9), "남미 위성": (12, 1, 2, 3), "흑해 위성": (4, 5, 6), "미국 밀 위성": (4, 5, 6, 7)}


def satellite(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """NOAA 위성 식생건강지수(VHI, 경작지) 묶음 평균의 주간 변화 — 생육기만. 관측 주 끝 + 7일 뒤 사용(발표 지연)."""
    D = pd.read_parquet(MM / "vhi.parquet")
    D["d"] = pd.to_datetime(D.year.astype(str) + "-01-01") + pd.to_timedelta(7 * D.week - 1, unit="D") + pd.Timedelta(days=7)
    out = {}
    for g, months in VHI_SEASON.items():
        s = D[D.group == g].groupby("d").vhi.mean()
        s = pd.Series(s.values, index=_friday(pd.Series(s.index)).values).groupby(level=0).last()
        s = s.reindex(s.index.union(idx)).ffill().reindex(idx)
        on = pd.Series(idx.month.isin(months), index=idx)
        out[g] = s.diff().where(on & on.shift(1, fill_value=False)).fillna(0.0)
    return pd.DataFrame(out)


def export_sales(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """주간 순판매(현·차기 연도 합)의 '평소 대비' z — 직전 52주 평균·표준편차 기준. 목요일 마감 주 → 다음 주 목요일 발표 → 그 주 금요일."""
    D = pd.read_parquet(MM / "export_sales.parquet")
    out = {}
    for c, nm in (("Corn", "옥수수 수출 판매"), ("Soybeans", "대두 수출 판매"), ("Wheat", "밀 수출 판매")):
        s = D[D.commodity == c].set_index("date").net_sales.sort_index()
        z = (s - s.shift(1).rolling(52, min_periods=26).mean()) / s.shift(1).rolling(52, min_periods=26).std()
        out[nm] = pd.Series(z.values, index=_friday(pd.Series(z.index + pd.Timedelta(days=8))).values).groupby(level=0).last()
    return pd.DataFrame(out).reindex(idx).fillna(0.0)


def freight(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """걸프 → 일본 곡물 운임(월간)의 로그 변화(%) — 다음 달 10일 이후 그 주에 반영, 그 밖은 0."""
    D = pd.read_parquet(MM / "freight.parquet").set_index("date").gulf_to_japan.sort_index()
    ch = 100 * np.log(D).diff()
    av = _friday(pd.Series(ch.index + pd.offsets.MonthEnd(0) + pd.Timedelta(days=10)))
    return pd.DataFrame({"해상 운임": pd.Series(ch.values, index=av.values).groupby(level=0).last()}).reindex(idx).fillna(0.0)


def nodes(L: pd.DataFrame) -> pd.DataFrame:
    R = LS._ret(LS.weekly(L, list(PX.values()), START)).rename(columns={v: k for k, v in PX.items()})
    idx = R.index[1:]
    X = pd.concat([weather_stress(idx), crop_condition(idx), wasde_surprise(idx), fx(idx), satellite(idx), export_sales(idx), freight(idx), R.loc[idx]], axis=1)
    return X[GRAPHS[2][0] + list(GRAPHS[2][1])].dropna()


# ============================================================================
# ② 구조 모형 (선형 SCM) 추정 · 역사적 분해 · 역방향 추적
# ============================================================================
def _ols(y, Z):
    Z1 = np.column_stack([np.ones(len(y)), Z])
    b, *_ = np.linalg.lstsq(Z1, y, rcond=None)
    e = y - Z1 @ b
    s2 = e @ e / max(len(y) - Z1.shape[1], 1)
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(Z1.T @ Z1)))
    return b, se, e


def fit_scm(X: pd.DataFrame) -> dict:
    """노드마다: x_t = 상수 + Σ 부모_t + Σ (자기·부모)_{t−1} + 충격. 바깥 노드(ROOTS)는 그 자체가 충격."""
    M = {}
    for n in ORDER:
        if n in ROOTS:
            M[n] = dict(c=X[n].mean(), b={}, l={}, e=(X[n] - X[n].mean()).values, r2=np.nan)
            continue
        lag = [n] + PA[n]
        Z = np.column_stack([X[p].values[1:] for p in PA[n]] + [X[q].values[:-1] for q in lag]) if PA[n] else np.column_stack([X[q].values[:-1] for q in lag])
        y = X[n].values[1:]
        b, se, e = _ols(y, Z)
        k = len(PA[n])
        e0 = X[n].values[0] - b[0] - sum(bb * X[p].values[0] for p, bb in zip(PA[n], b[1:1 + k]))   # 첫 주(시차 없음)도 실제와 맞게
        M[n] = dict(c=b[0], b=dict(zip(PA[n], b[1:1 + k])), t=dict(zip(PA[n], b[1:1 + k] / se[1:1 + k])),
                    l=dict(zip(lag, b[1 + k:])), e=np.r_[e0, e], r2=1 - e.var() / y.var())
    return M


def oos_r2(m: dict, X: pd.DataFrame, n: str) -> float:
    """학습 계수 그대로 시험 기간 가격 주간 변동을 부모(같은 주 · 직전 주)로 설명한 R²."""
    y = X[n].values[1:]
    f = m["c"] + sum(b * X[p].values[1:] for p, b in m["b"].items()) + sum(b * X[q].values[:-1] for q, b in m["l"].items())
    return 1 - np.mean((y - f) ** 2) / np.var(y)


def simulate(M: dict, T: int, src: str | None) -> pd.DataFrame:
    """src 노드의 충격만 넣고(None = 상수항만) 온톨로지를 따라 전파. 모든 src 합 + 상수 = 실제값(정확)."""
    S = {n: np.zeros(T) for n in ORDER}
    for t in range(T):
        for n in ORDER:
            m = M[n]
            v = (m["c"] if src is None else 0.0) + (m["e"][t] if n == src else 0.0)
            if n not in ROOTS:
                v += sum(bb * S[p][t] for p, bb in m["b"].items())
                if t > 0:
                    v += sum(bb * S[q][t - 1] for q, bb in m["l"].items())
            S[n][t] = v
    return pd.DataFrame(S)


def decompose(M: dict, X: pd.DataFrame) -> dict:
    T = len(X)
    C = {s: simulate(M, T, s).set_index(X.index) for s in ORDER}
    C["상수"] = simulate(M, T, None).set_index(X.index)
    err = max(abs(sum(C[s][p] for s in C) - X[p]).iloc[1:].max() for p in PRICES)
    return dict(C=C, err=err)


def paths(src: str, dst: str) -> list[list[str]]:
    """같은 주 안의 온톨로지 경로 (src → … → dst)."""
    if src == dst:
        return [[dst]]
    return [[src] + p for c, ps in PA.items() if src in ps for p in paths(c, dst)]


def path_effect(M: dict, p: list[str]) -> float:
    return float(np.prod([M[b]["b"][a] for a, b in zip(p[:-1], p[1:])])) if len(p) > 1 else 1.0


def attribution(M: dict, D: dict, X: pd.DataFrame, target: str, a: str, b: str, label: str) -> list[dict]:
    """결과(target의 a~b 누적 변동, %)에서 원인으로 거슬러 올라감: 원인 충격별 기여 + 가장 큰 경로."""
    sl = slice(pd.Timestamp(a), pd.Timestamp(b))
    tot = X.loc[sl, target].sum()
    rows = []
    for s in ORDER + ["상수"]:
        v = D["C"][s].loc[sl, target].sum()
        if s != "상수" and s != target and abs(v) < 0.05:
            continue
        best = ""
        if s not in ("상수", target):
            shock = pd.Series(M[s]["e"], index=X.index).loc[sl].sum()
            ps = sorted(paths(s, target), key=lambda p: -abs(path_effect(M, p) * shock))
            best = " → ".join(ps[0]) if ps else "시차 경로만"
        rows.append(dict(사례=label, 대상=target, 기간=f"{a} ~ {b}", 실제_누적=round(tot, 1),
                         원인=("설명 안 됨 (자기 충격)" if s == target else s), 기여=round(v, 1), 주요_경로=best))
    return sorted(rows, key=lambda r: -abs(r["기여"]))


# ============================================================================
# ③ PCMCI (ParCorr) — PC1 부모 선택 + MCI 검정, 같은 주(시차 0)는 PC 골격
# ============================================================================
def parcorr(x, y, Z=None):
    if Z is not None and Z.shape[1]:
        Z1 = np.column_stack([np.ones(len(x)), Z])
        x = x - Z1 @ np.linalg.lstsq(Z1, x, rcond=None)[0]
        y = y - Z1 @ np.linalg.lstsq(Z1, y, rcond=None)[0]
        k = Z.shape[1]
    else:
        k = 0
    r = np.corrcoef(x, y)[0, 1]
    df = len(x) - 2 - k
    t = r * np.sqrt(df / max(1 - r * r, 1e-12))
    return r, 2 * stats.t.sf(abs(t), df)


def pcmci(X: pd.DataFrame, tau_max=2, a_pc=0.2, a_mci=0.01) -> pd.DataFrame:
    V = (X - X.mean()) / X.std()
    A_, names, T = V.values, list(V.columns), len(V)
    N = len(names)
    lagv = lambda i, tau: A_[tau_max - tau:T - tau, i]                                   # noqa: E731
    cur = lambda j: A_[tau_max:, j]                                                       # noqa: E731
    P = {}
    for j in range(N):                                                                    # PC1: 시차 부모 후보 줄이기
        cand = {(i, tau): 0.0 for i in range(N) for tau in range(1, tau_max + 1)}
        p = 0
        while p < len(cand):
            order = sorted(cand, key=lambda c: -abs(cand[c]))
            for c in list(order):
                cond = [d for d in order if d != c][:p]
                Z = np.column_stack([lagv(*d) for d in cond]) if cond else None
                r, pv = parcorr(cur(j), lagv(*c), Z)
                if pv > a_pc:
                    cand.pop(c)
                else:
                    cand[c] = r
            p += 1
        P[j] = sorted(cand, key=lambda c: -abs(cand[c]))
    rows = []
    for j in range(N):                                                                    # MCI: 시차 연결
        for i in range(N):
            for tau in range(1, tau_max + 1):
                cj = [d for d in P[j] if d != (i, tau)]
                ci = [(k, t2 + tau) for k, t2 in P[i] if t2 + tau <= tau_max]
                Z = np.column_stack([lagv(*d) for d in cj + ci]) if cj + ci else None
                r, pv = parcorr(cur(j), lagv(i, tau), Z)
                if pv < a_mci and i != j:
                    rows.append(dict(원인=names[i], 결과=names[j], 시차=tau, 편상관=round(r, 3), p값=pv))
    adj = {(i, j) for i in range(N) for j in range(N) if i < j}                         # 같은 주: PC 골격 (조건 = 시차 부모 + 이웃 최대 2개)
    for size in range(3):
        for (i, j) in sorted(adj):
            nb = sorted({k for a, b in adj for k in (a, b) if i in (a, b) or j in (a, b)} - {i, j})
            lag_par = [d for d in P[i] + P[j]]
            for S in combinations(nb, size):
                Z = np.column_stack([lagv(*d) for d in lag_par] + [cur(k) for k in S]) if lag_par or S else None
                r, pv = parcorr(cur(i), cur(j), Z)
                if pv > a_mci:
                    adj.discard((i, j))
                    break
    for (i, j) in adj:
        Z = np.column_stack([lagv(*d) for d in P[i] + P[j]]) if P[i] + P[j] else None
        r, pv = parcorr(cur(i), cur(j), Z)
        rows.append(dict(원인=names[i], 결과=names[j], 시차=0, 편상관=round(r, 3), p값=pv))
    return pd.DataFrame(rows)


def compare_ontology(Pc: pd.DataFrame) -> pd.DataFrame:
    same = {frozenset((r.원인, r.결과)) for r in Pc[Pc.시차 == 0].itertuples()}
    lagged = {(r.원인, r.결과) for r in Pc[Pc.시차 > 0].itertuples()}
    rows = [dict(연결=f"{p} → {c}", 온톨로지="있음", PCMCI=("같은 주 확인" if frozenset((p, c)) in same else "시차로 확인" if (p, c) in lagged else "확인 안 됨"))
            for c, ps in PA.items() for p in ps]
    ont = {frozenset((p, c)) for c, ps in PA.items() for p in ps}
    rows += [dict(연결=f"{r.원인} — {r.결과}" if r.시차 == 0 else f"{r.원인} → {r.결과} ({r.시차}주 뒤)", 온톨로지="없음 (후보)", PCMCI=f"편상관 {r.편상관}")
             for r in Pc.itertuples() if frozenset((r.원인, r.결과)) not in ont]
    return pd.DataFrame(rows)


# ============================================================================
# ① 정책 이벤트 반사실 (Causal Impact 방식)
# ============================================================================
EVENTS = [("2018-06-15", "미국산 대두에 중국 보복관세 발표", "대두", -1),
          ("2022-02-24", "러시아 우크라이나 침공", "밀", +1),
          ("2022-05-16", "인도 밀 수출 금지 (5/14 발표, 첫 거래일)", "밀", +1),
          ("2022-07-22", "흑해 곡물 협정 체결", "밀", -1),
          ("2022-09-05", "아르헨티나 '대두 달러' 우대 환율 시행", "대두박", -1),
          ("2023-07-17", "러시아 흑해 곡물 협정 탈퇴", "밀", +1)]


def _car(r: pd.DataFrame, y: str, X: list[str], k: int, pre=(260, 11), post=20):
    if k - pre[0] < 0 or k + post > len(r):
        return None
    tr = r.iloc[k - pre[0]:k - pre[1]].dropna()
    if len(tr) < 150:
        return None
    b, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(tr)), tr[X].values]), tr[y].values, rcond=None)
    ev = r.iloc[k:k + post].fillna(0.0)
    ar = ev[y].values - np.column_stack([np.ones(post), ev[X].values]) @ b
    return ar[:5].sum(), ar.sum(), np.cumsum(ar)


def event_study(L: pd.DataFrame, n_placebo=400, seed=0) -> pd.DataFrame:
    ctrl = [c for c in LS.BASKET if c not in GRAIN]
    r = 100 * np.log(L[list(PX.values()) + ctrl]).diff()
    r = r[r.index >= pd.Timestamp(START)].dropna(how="all")
    rng = np.random.default_rng(seed)
    ev_pos = [r.index.searchsorted(pd.Timestamp(d)) for d, *_ in EVENTS]
    ok = [k for k in range(300, len(r) - 25) if all(abs(k - e) > 60 for e in ev_pos)]
    rows = []
    for (d, nm, tgt, sign), k in zip(EVENTS, ev_pos):
        y = PX[tgt]
        res = _car(r, y, ctrl, k)
        pl = [x for x in (_car(r, y, ctrl, kk) for kk in rng.choice(ok, n_placebo, replace=False)) if x]
        p5 = np.mean([abs(x[0]) >= abs(res[0]) for x in pl])
        p20 = np.mean([abs(x[1]) >= abs(res[1]) for x in pl])
        rows.append(dict(날짜=d, 이벤트=nm, 대상=tgt, 예상_방향="+" if sign > 0 else "−", 첫날_초과=round(res[2][0], 1),
                         초과_5일=round(res[0], 1), p_5일=round(p5, 3), 초과_20일=round(res[1], 1), p_20일=round(p20, 3),
                         판단=("방향 맞고 유의" if np.sign(res[0]) == sign and p5 < 0.05 else "방향 맞음 · 우연 수준" if np.sign(res[0]) == sign else "예상과 반대")))
    return pd.DataFrame(rows)


# ============================================================================
CASES = [("옥수수", "2012-06-08", "2012-07-20", "2012 미국 가뭄"), ("대두", "2012-06-08", "2012-07-20", "2012 미국 가뭄"),
         ("대두박", "2022-12-02", "2023-03-03", "2023 아르헨티나 가뭄"), ("밀", "2022-02-18", "2022-03-11", "2022 러-우 전쟁"),
         ("대두유", "2021-01-01", "2021-05-07", "2021 대두유 급등(바이오연료)")]


def run():
    sys.stdout.reconfigure(errors="replace")
    L = A.load_levels()
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 60)

    print("① 정책 이벤트 반사실 (대조군 = 곡물 밖 13품목, 사전 1년 회귀, 위약 400회)")
    E = event_study(L)
    print(E.to_string(index=False))
    E.to_csv(DATA / "causal_events.csv", index=False, encoding="utf-8-sig")

    X = nodes(L)
    cut = pd.Timestamp("2016-12-31")
    tr, te = X[X.index <= cut], X[X.index > cut]
    print(f"\n② 온톨로지 구조 모형: {X.index[0]:%Y-%m-%d} ~ {X.index[-1]:%Y-%m-%d}, {len(X)}주")
    shares = []
    for v in (1, 2):
        set_graph(v)
        M, Mt = fit_scm(X), fit_scm(tr)
        D = decompose(M, X)
        for p in PRICES:
            shares.append(dict(판=v, 가격=p, 설명_몫_전체=round(1 - np.var(D["C"][p][p].values) / np.var(X[p].values), 3),
                               시험_R2_2017_=round(oos_r2(Mt[p], te, p), 3)))
        print(f"   {v}판 노드 {len(ORDER)}개 · 분해 검산 오차 {D['err']:.1e} (0이면 정확)")
    SH = pd.DataFrame(shares).pivot(index="가격", columns="판", values=["설명_몫_전체", "시험_R2_2017_"])
    print("   가격 주간 변동 중 온톨로지 원인으로 설명되는 몫 (1판 → 2판), 시험 = 2008–16 계수로 2017– 설명:")
    print(SH.to_string())
    SH.columns = [f"{a}_{b}판" for a, b in SH.columns]
    SH.to_csv(DATA / "causal_scm_share.csv", encoding="utf-8-sig")
    M1, M2 = fit_scm(tr), fit_scm(te)
    er = []
    for c, ps in PA.items():
        for p in ps:
            er.append(dict(연결=f"{p} → {c}", 계수=round(M[c]["b"][p], 3), t=round(M[c]["t"][p], 1),
                           계수_2008_16=round(M1[c]["b"][p], 3), t_2008_16=round(M1[c]["t"][p], 1),
                           계수_2017_=round(M2[c]["b"][p], 3), t_2017_=round(M2[c]["t"][p], 1),
                           두기간_같은부호=bool(np.sign(M1[c]["b"][p]) == np.sign(M2[c]["b"][p]))))
    ER = pd.DataFrame(er)
    print(ER.to_string(index=False))
    ER.to_csv(DATA / "causal_scm_edges.csv", index=False, encoding="utf-8-sig")

    print("\n   역방향 추적 (결과 → 원인): 사례별 누적 변동(%)을 원인 충격·경로로 분해")
    last = X.index[-1]
    cases = CASES + [(p, f"{last - pd.Timedelta(weeks=3):%Y-%m-%d}", f"{last:%Y-%m-%d}", "최근 4주") for p in PRICES]
    B = pd.DataFrame(sum((attribution(M, D, X, *c) for c in cases), []))
    print(B.to_string(index=False))
    B.to_csv(DATA / "causal_attribution.csv", index=False, encoding="utf-8-sig")

    print("\n③ PCMCI (ParCorr, 최대 2주 시차, MCI α=0.01) — 온톨로지 연결과 대조")
    Pc = pcmci(X[ORDER])
    Cmp = compare_ontology(Pc)
    print(Cmp.to_string(index=False))
    Cmp.to_csv(DATA / "causal_pcmci.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    run()
