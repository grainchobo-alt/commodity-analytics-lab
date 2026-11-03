"""② 전문가 의견 백테스트 — '의견 방향 vs 이후 h주 수익률'. 채택 규칙을 통과한 의견원만 판단 보조 신호로 쓸 수 있다.

의견원 (source)
  saved : 사내 전문가 의견 data/expert/views.csv (save_expert_view로 쌓임) — 20건 이상부터 판정
  usda  : USDA 분석가의 시즌 평균 가격 전망 수정(WASDE d_price) — 올림 = 강세 의견. 발표 12:00 ET → 그날 종가에 진입 가능
  cot   : CFTC 펀드(managed money) 순포지션 주간 변화 — 늘림 = 강세. 화요일 기준 · 금요일 15:30 ET 공표 → 다음 거래일 종가 진입
평가 (진입 = 의견을 실제로 알 수 있었던 첫 종가, 미래정보 없음)
  적중률(방향 맞힘) · 방향 × 이후 수익(bp, 의견대로 샀을 때 평균 성과) · Newey-West t · 정보계수(IC, 순위 상관)
  위약: 같은 품목 안에서 방향을 무작위로 섞어 500회 → p값 · 하위 기간 2개(–2016 · 2017–) 모두 같은 부호인지
채택 규칙 (온톨로지 실험과 같은 기준): 표본 ≥ 20 · 평균 성과 t > 2 · 위약 p < 0.05 · 두 하위 기간 모두 + → '채택', 아니면 '설명용'
실행: .venv\\Scripts\\python.exe -m llm.expert_bt   → data/expert_proxy_bt.csv (usda · cot 품목별 판정)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from llm import tools as T

OUT = T.DATA / "expert_proxy_bt.csv"
MIN_N = 20
GRAINS = ["yf:ZC=F", "yf:ZS=F", "yf:ZW=F", "yf:ZM=F", "yf:ZL=F"]
COT = {"yf:ZC=F": "Corn", "yf:ZS=F": "Soybeans", "yf:ZW=F": "Wheat SRW", "yf:ZM=F": "Soybean Meal", "yf:ZL=F": "Soybean Oil"}


# ============================================================================
# 의견원 → 표준 형식 (avail = 의견을 알 수 있었던 시각의 날짜, same_close = 그날 종가 진입 가능 여부)
# ============================================================================
def views_saved() -> pd.DataFrame:
    if not T.EXPERT.exists():
        return pd.DataFrame(columns=["avail", "id", "direction", "confidence", "horizon_weeks", "same_close", "source"])
    V = pd.read_csv(T.EXPERT, encoding="utf-8-sig")
    return pd.DataFrame({"avail": pd.to_datetime(V.date), "id": V.id, "direction": V.direction, "confidence": V.confidence,
                         "horizon_weeks": V.horizon_weeks, "same_close": False, "source": "saved"})   # 작성 시각 모름 → 보수적으로 다음 종가


def views_usda() -> pd.DataFrame:
    W = pd.read_parquet(T.DATA / "reports" / "wasde_features.parquet")
    W = W[W.id.isin(GRAINS) & (W["first"] == 0) & W.d_price.notna()]           # 같은 연산연도 안의 전망 수정만 (새 연도 첫 전망 제외)
    W = W[W.d_price != 0]
    return pd.DataFrame({"avail": W.release.dt.tz_localize(None).dt.normalize(), "id": W.id, "direction": np.sign(W.d_price),
                         "confidence": np.clip(W.d_price.abs() / 10, 0, 1), "horizon_weeks": 4, "same_close": True, "source": "usda"})


def views_cot() -> pd.DataFrame:
    C = pd.read_parquet(T.DATA / "cot.parquet")
    rows = []
    for i, nm in COT.items():
        g = C[C.name == nm].sort_values("date")
        d = (g.mm_net.diff() / g.open_interest_all).dropna()                      # 미결제약정 대비 순포지션 변화
        z = (d - d.rolling(156, min_periods=52).mean()) / d.rolling(156, min_periods=52).std()   # 직전 3년 기준 크기 (과거만)
        ok = z.notna() & (d != 0)
        rows.append(pd.DataFrame({"avail": g.date[ok.index[ok]] + pd.Timedelta(days=3), "id": i, "direction": np.sign(d[ok]),
                                  "confidence": np.clip(z[ok].abs() / 3, 0, 1), "horizon_weeks": 4, "same_close": False, "source": "cot"}))
    return pd.concat(rows, ignore_index=True)


SOURCES = {"saved": views_saved, "usda": views_usda, "cot": views_cot}


# ============================================================================
# 평가
# ============================================================================
def attach_returns(V: pd.DataFrame) -> pd.DataFrame:
    """진입 = 의견을 알 수 있었던 첫 종가(같은 날 가능하면 그날, 아니면 다음 거래일), 청산 = 진입 + h주(5·h 거래일). 롤 보정 가격."""
    L = T.levels()
    out = []
    for (i, h), g in V.groupby(["id", "horizon_weeks"]):
        if i not in L:
            continue
        p = L[i].dropna()
        idx = p.index
        side = np.where(g.same_close, "left", "right")
        k = np.array([idx.searchsorted(d, side=s) for d, s in zip(g.avail, side)])
        e = k + 5 * int(h)
        ok = e < len(idx)
        g = g[ok].copy()
        g["entry"], g["exit"] = idx[k[ok]], idx[e[ok]]
        g["ret_pct"] = 100 * np.log(p.values[e[ok]] / p.values[k[ok]])
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else V.assign(entry=pd.NaT, exit=pd.NaT, ret_pct=np.nan).iloc[:0]


def nw_t(x: np.ndarray, L: int = 4) -> float:
    x = np.asarray(x, float); n = len(x)
    if n < 3 or x.std() == 0:
        return np.nan
    e = x - x.mean(); v = e @ e / n
    for l in range(1, min(L, n - 1) + 1):
        v += 2 * (1 - l / (L + 1)) * (e[l:] @ e[:-l]) / n
    return float(x.mean() / np.sqrt(v / n)) if v > 0 else np.nan


def evaluate(R: pd.DataFrame, n_placebo: int = 500, seed: int = 0, split: str = "2017-01-01") -> dict:
    """R: attach_returns 결과. 성과 = 방향 × 이후 수익(%) — 의견대로 샀다/안 샀다를 했을 때 얻은 것."""
    R = R[R.direction != 0].dropna(subset=["ret_pct"]).sort_values("entry")
    n = len(R)
    if n < MIN_N:
        return dict(n=n, verdict=f"표본 부족 ({n}건 < {MIN_N})")
    pnl = (R.direction * R.ret_pct).values
    hit = float((np.sign(R.ret_pct) == R.direction).mean())
    ic = float(pd.Series(R.direction * R.confidence).rank().corr(R.ret_pct.rank()))
    rng = np.random.default_rng(seed)
    dirs = R.direction.values; grp = R.id.values
    base = pnl.mean(); cnt = 0
    for _ in range(n_placebo):
        d = dirs.copy()
        for gid in np.unique(grp):                                              # 품목 안에서만 섞음 (품목별 강세/약세 비중 유지)
            m = grp == gid; d[m] = rng.permutation(d[m])
        cnt += (d * R.ret_pct.values).mean() >= base
    p_pl = (cnt + 1) / (n_placebo + 1)
    early, late = R.entry < pd.Timestamp(split), R.entry >= pd.Timestamp(split)
    sub = {k: (float(pnl[m.values].mean()) if m.sum() >= 5 else np.nan) for k, m in (("early", early), ("late", late))}
    t = nw_t(pnl)
    adopt = (t > 2) and (p_pl < 0.05) and all(np.isfinite(v) and v > 0 for v in sub.values())
    return dict(n=n, start=R.entry.min(), end=R.entry.max(), hit_rate=hit, mean_bp=100 * base, t=t, ic=ic, placebo_p=p_pl,
                early_bp=100 * sub["early"], late_bp=100 * sub["late"], verdict="채택" if adopt else "설명용 (채택 규칙 미통과)")


def backtest(source: str = "usda", item: str | None = None) -> dict:
    V = SOURCES[source]()
    if item:
        V = V[V.id == T.resolve(item)]
    return evaluate(attach_returns(V))


def run(sources=("usda", "cot")) -> pd.DataFrame:
    rows = []
    for s in sources:
        V = attach_returns(SOURCES[s]())
        for i in [None] + GRAINS:
            R = V if i is None else V[V.id == i]
            rows.append(dict(source=s, item="곡물 5종 전체" if i is None else T.name_of(i), id=i or "all", **evaluate(R)))
    D = pd.DataFrame(rows)
    D.to_csv(OUT, index=False, encoding="utf-8-sig")
    return D


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pd.set_option("display.width", 220)
    D = run()
    cols = ["source", "item", "n", "hit_rate", "mean_bp", "t", "ic", "placebo_p", "early_bp", "late_bp", "verdict"]
    print(D[cols].round(3).to_string(index=False))
    print(f"\n→ {OUT}")
