"""상관 급등기에 모멘텀 비중 확대 · 집중 투자 백테스트 — 2022–23 우크라이나 전쟁처럼 곡물 · 원자재가 함께 움직일 때 추세에 더 실으면 나아지나?

재료(2001년 자료로 재구성, 목표 10% · 상한 4배 · 편도 5bp): strategy_meta.sleeves의 3축 + 변형 4개
  상승 추세 집중 k: 26 · 52주 수익이 모두 + 인 품목 중 26주 위험조정 수익 상위 k(3 · 2)개만 롱, 같은 위험 — 2022 곡물 · 원유, 2025 금 · 은 같은 급등에 몰기 (사용자 제안)
  모멘텀 집중  : 52주 수익률 ÷ 26주 변동성의 절댓값 상위 5개 품목만, 추세 방향으로 같은 위험(4주마다) — 품목 수를 줄여 강한 추세에 몰기
  MinVar 무제한: MinVar 롱의 품목 상한 25%를 없앤 판(alloc_weights_cap 상한 100%)
상관 급등 신호(그 주까지 자료만)
  곡물 상관   : 곡물 5개(옥수수 · 밀 · 대두 · 대두박 · 대두유) 26주 평균 쌍 상관의 그때까지 분위 > 문턱
  DCC 동조성  : 원자재 8개 DCC 평균 상관의 그때까지 분위 > 문턱 (strategy_cash.comove_dcc)
  흡수비율    : 원자재 19개 1주성분 몫 4주 변화 z > 문턱
방향 조건: 방향 무관 · 상승 동조(곡물 또는 바스켓 직전 13주 수익 > 0일 때만) · 하락 동조(< 0일 때만) — 함께 폭락하는 동조에는 모멘텀을 늘리지 않는 판을 따로 봄
켜지면 바꿀 비중(MinVar / 모멘텀 / 가치): 25/50/25(모멘텀 확대) · 25/50/25 + 모멘텀 집중 · 0/100/0 모멘텀 집중 · 25/50/25 + 모멘텀 집중 + MinVar 무제한
대조군: 늘 그 비중(신호 없이). 신호 · 문턱 · 비중은 학습 2008–2016 Sharpe로만 고르고 시험 2017–에서 확인. 2022–23 · 2008 구간 성과 따로. 위약: 켜짐 구간을 13주 블록으로 섞기 200회.
실행: .venv\\Scripts\\python.exe strategy_crisis_mom.py → data/strategy_crisis_mom_compare.csv
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import analytics as A  # noqa: E402
import ls_strategies as LS  # noqa: E402
import strategy_cash as SC  # noqa: E402
import strategy_meta as SM  # noqa: E402
import strategy_switch as SS  # noqa: E402

ANN, TGT, COST = 52, 10.0, 5.0
FROZEN_END, SPLIT = "2026-10-26", "2017-01-01"
GRAINS = ["yf:ZC=F", "yf:ZW=F", "yf:ZS=F", "yf:ZM=F", "yf:ZL=F"]
COLS = ["MinVar", "MinVar 무제한", "모멘텀", "모멘텀 집중", "상승 추세 집중 3", "상승 추세 집중 2", "가치"]


def up_conc(P: pd.DataFrame, R: pd.DataFrame, k: int = 3, reb: int = 4) -> tuple[pd.DataFrame, dict]:
    """상승 추세 집중(롱만): 26주 · 52주 수익이 모두 + 인 품목 중 26주 수익 ÷ 26주 변동성 상위 k개를 같은 위험으로 — 2022 곡물 · 원유, 2025 금 · 은 같은 급등에 몰기."""
    lp = np.log(P); s26, s52 = lp.diff(26), lp.diff(52); v = R.rolling(26, min_periods=13).std()
    W = pd.DataFrame(np.nan, index=P.index, columns=P.columns); held = {}
    for t in range(0, len(P), reb):
        ok = (s26.iloc[t] > 0) & (s52.iloc[t] > 0)
        sc = (s26.iloc[t] / v.iloc[t])[ok].dropna()
        W.iloc[t] = 0.0
        if len(sc):
            top = sc.nlargest(k).index
            w = 1 / v.iloc[t][top]; W.loc[P.index[t], top] = (w / w.sum()).values
            held[P.index[t]] = list(top)
    return W.ffill().fillna(0.0), held


def conc_mom(P: pd.DataFrame, R: pd.DataFrame, k: int = 5, reb: int = 4) -> pd.DataFrame:
    s = np.log(P).diff(52); v = R.rolling(26, min_periods=13).std()
    sc = (s / v).abs()
    W = pd.DataFrame(np.nan, index=P.index, columns=P.columns)
    for t in range(0, len(P), reb):
        row = sc.iloc[t].dropna()
        if len(row) < k:
            continue
        top = row.nlargest(k).index
        w = np.sign(s.iloc[t][top]) / v.iloc[t][top]
        W.iloc[t] = 0.0; W.loc[P.index[t], top] = (w / w.abs().sum()).values
    return W.ffill().fillna(0.0)


HELD: dict = {}


def sleeves5(L: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    S, R, P = SM.sleeves(L)
    book = lambda W: LS._book(LS._vol_target(W, R, TGT), R, COST)   # noqa: E731
    W3, HELD[3] = up_conc(P, R, 3); W2, HELD[2] = up_conc(P, R, 2)
    X = pd.DataFrame({"MinVar": S["MinVar"], "MinVar 무제한": book(LS.alloc_weights_cap(P, 1.0)).reindex(S.index),
                      "모멘텀": S["모멘텀"], "모멘텀 집중": book(conc_mom(P, R)).reindex(S.index),
                      "상승 추세 집중 3": book(W3).reindex(S.index), "상승 추세 집중 2": book(W2).reindex(S.index), "가치": S["가치"]}).fillna(0.0)
    return X, R


def grain_corr_pct(R: pd.DataFrame, idx) -> pd.Series:
    G = R[GRAINS]
    c = pd.Series(np.nan, index=G.index)
    for t in range(26, len(G)):
        C = G.iloc[t - 25:t + 1].corr().values
        c.iloc[t] = C[np.triu_indices(len(GRAINS), 1)].mean()
    return SC.expanding_pct(c.reindex(idx).ffill())


def W_of(idx, on: pd.Series, w_on: dict, w_off: dict) -> pd.DataFrame:
    on = on.reindex(idx).fillna(False).astype(bool)
    a = np.array([w_off.get(c, 0.0) for c in COLS]); b = np.array([w_on.get(c, 0.0) for c in COLS])
    return pd.DataFrame(np.where(on.values[:, None], b, a), index=idx, columns=COLS)


def seg_sh(r: pd.Series, a: str, b: str) -> float:
    x = r.loc[a:b]
    return float(x.mean() / x.std() * np.sqrt(ANN)) if len(x) > 10 and x.std() > 0 else np.nan


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    L = A.load_levels(roll_adjusted=True).loc[:FROZEN_END]
    X, R = sleeves5(L); idx = X.index
    print("단독 Sharpe 2008–:", {c: round(seg_sh(X[c], "2008", "2026"), 2) for c in COLS}, flush=True)
    from llm import tools as T_
    for a, b in [("2022-01-01", "2022-12-31"), ("2025-01-01", "2026-10-31")]:     # 상승 추세 집중이 실제로 무엇을 들었나
        hs = [h for d, h in HELD[3].items() if a <= f"{d:%Y-%m-%d}" <= b]
        cnt = pd.Series([i for h in hs for i in h]).value_counts()
        print(f"상승 추세 집중 3 보유 {a[:4]}–{b[:4]}:", ", ".join(f"{T_.name_of(i)} {n}회" for i, n in cnt.head(8).items()))
        print(f"  단독 수익 {a[:4]}–{b[:4]}: 상승 추세 집중 3 {X['상승 추세 집중 3'].loc[a:b].sum():+.1f}% · 모멘텀 {X['모멘텀'].loc[a:b].sum():+.1f}% · 고정 3축 혼합 원료 MinVar {X['MinVar'].loc[a:b].sum():+.1f}%", flush=True)
    ar = LS.absorption(R, 52); dz = ((ar - ar.shift(4)) / (ar - ar.shift(4)).rolling(156, min_periods=52).std()).reindex(idx).ffill()
    sig = {"곡물 상관": grain_corr_pct(R, idx), "DCC 동조성": SC.expanding_pct(SC.comove_dcc(R, idx)), "흡수비율": dz}
    grid = {"곡물 상관": [0.8, 0.9], "DCC 동조성": [0.8, 0.9], "흡수비율": [1.0, 1.5]}
    # 방향 조건: 상관이 급등해도 '함께 오르는 중'(직전 13주 수익 > 0)일 때만 — 함께 폭락하는 동조는 제외 (사용자 지적)
    up_g = (R[GRAINS].mean(1).rolling(13).sum() > 0).reindex(idx).ffill().fillna(False)
    up_b = (R.mean(1).rolling(13).sum() > 0).reindex(idx).ffill().fillna(False)
    DIR = {"방향 무관": pd.Series(True, idx), "상승 동조 (13주 수익 > 0)": None, "하락 동조 (13주 수익 < 0)": None}
    base = {"MinVar": 0.5, "모멘텀": 0.25, "가치": 0.25}
    ons = {"모멘텀 확대 25/50/25": {"MinVar": 0.25, "모멘텀": 0.5, "가치": 0.25},
           "모멘텀 확대 + 집중": {"MinVar": 0.25, "모멘텀 집중": 0.5, "가치": 0.25},
           "모멘텀 집중 100%": {"모멘텀 집중": 1.0},
           "모멘텀 확대 + 집중 + MinVar 한도 해제": {"MinVar 무제한": 0.25, "모멘텀 집중": 0.5, "가치": 0.25},
           "상승 추세 집중 3개 50%": {"MinVar": 0.25, "상승 추세 집중 3": 0.5, "가치": 0.25},
           "상승 추세 집중 2개 50%": {"MinVar": 0.25, "상승 추세 집중 2": 0.5, "가치": 0.25},
           "상승 추세 집중 3개 100%": {"상승 추세 집중 3": 1.0}}
    bench = SS.run_weights(X, W_of(idx, pd.Series(False, idx), base, base))
    rows = []

    def add(nm, setting, r, on=None, extra=None):
        o = SM.score(r, bench)
        o.update({"2022–23 Sharpe": seg_sh(r, "2022-01-01", "2023-12-31"), "2022–23 수익 (%)": float(r.loc["2022":"2023"].sum()),
                  "2008 Sharpe": seg_sh(r, "2008-01-01", "2008-12-31")})
        if on is not None:
            o["켜진 주 (%)"] = float(on.loc["2008":].mean() * 100)
            o["2022–23 켜진 주 (%)"] = float(on.loc["2022":"2023"].mean() * 100)
        rows.append({"규칙": nm, "설정": setting, **o, **(extra or {})})
    add("고정 50/25/25 (기준)", "", bench)
    for nm, w in ons.items():                                      # 대조군: 늘 그 비중
        add(f"늘 {nm} (대조군)", "", SS.run_weights(X, W_of(idx, pd.Series(True, idx), w, base)))
    cand = {}
    for sn, s in sig.items():
        up = up_g if sn == "곡물 상관" else up_b
        for dn in DIR:
            dmask = DIR[dn] if dn == "방향 무관" else (up if dn.startswith("상승") else ~up)
            for thr in grid[sn]:
                on = (s > thr) & dmask
                for wn, w in ons.items():
                    cand[(sn, dn, thr, wn)] = (SS.run_weights(X, W_of(idx, on, w, base)), on)
    sc = {k: SM.score(r, bench)["Sharpe 학습 2008–16"] for k, (r, _) in cand.items()}
    best = max(sc, key=sc.get)
    best_dir = {dn: max((k for k in sc if k[1] == dn), key=sc.get) for dn in DIR}   # 방향 조건별 학습 최고도 표시
    for k, (r, on) in cand.items():
        tag = "← 학습 최고" if k == best else ("← 이 방향 조건의 학습 최고" if k in best_dir.values() else "")
        add(f"트리거 · {k[0]} > {k[2]} · {k[1]} → {k[3]}", tag, r, on.astype(float))
    r0, on0 = cand[best]
    best = (best[0], best[2], best[3])
    rng = np.random.default_rng(0); blocks = [idx[i:i + 13] for i in range(0, len(idx), 13)]; pl = []
    for _ in range(200):
        o = rng.permutation(len(blocks)); op = pd.concat([on0.reindex(blocks[j]) for j in o]).iloc[:len(idx)]; op.index = idx
        pl.append(SM.score(SS.run_weights(X, W_of(idx, op, ons[best[2]], base)), bench)["Sharpe 시험 2017–"])
    real = SM.score(r0, bench)["Sharpe 시험 2017–"]
    for row in rows:
        if row["설정"] == "← 학습 최고":
            row["위약 p (시험)"] = float((np.array(pl) >= real).mean())
    T = pd.DataFrame(rows)
    T.to_csv(A.DATA / "strategy_crisis_mom_compare.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 260)
    cols = ["규칙", "설정", "Sharpe 학습 2008–16", "Sharpe 시험 2017–", "Sharpe 2008–", "최대 낙폭 (%)", "2022–23 Sharpe", "2022–23 수익 (%)", "2008 Sharpe",
            "켜진 주 (%)", "2022–23 켜진 주 (%)", "시험 차이 t", "위약 p (시험)"]
    top = T.sort_values("Sharpe 학습 2008–16", ascending=False)
    show = pd.concat([T[~T["규칙"].str.startswith("트리거")], T[T["설정"].str.startswith("←")], top[top["규칙"].str.startswith("트리거")].head(5)]).drop_duplicates("규칙")
    print(show[[c for c in cols if c in show]].round(2).to_string(index=False))
    print(f"위약 200회(학습 최고): 시험 Sharpe 실제 {real:.2f} vs 위약 중앙 {np.median(pl):.2f} · 95% {np.quantile(pl, 0.95):.2f}")


if __name__ == "__main__":
    main()
