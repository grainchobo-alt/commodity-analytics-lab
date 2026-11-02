"""해외 수급 수정(WASDE 세계 수급표, 발표 시점 값) 추가 실험 — 기존 파일은 고치지 않음.

노드 9개 = 발표 주의 '직전 발표 대비 수정 폭(%)' (발표 없는 주 = 0). 같은 연산연도끼리만 비교(새 연도 추가는 수정 아님).
해외 재고는 중국 제외(2018-11 중국 재고 대폭 개정 등), 수정 폭은 ±15%로 자름(EU 범위 변경 등 정의 변경 대비).
A. 온톨로지 3판(2판 + 해외 노드) 구조 모형: 설명 몫 · 시험 R²(전체 · WASDE 발표 주만) · 새 연결의 두 기간 계수
B. 수정 뒤 방향: 발표 다음 주부터 4주 수익률을 수정 폭으로 회귀 (HAC t, 두 기간) — 지연 반응(+)·과잉 반응(−) 있나
C. 판단 개선: ④ 구간 확률(M2 3판 vs M1) · ⑤ 구매 미루기(기본 vs +해외, 2·4주)
실행: .venv\\Scripts\\python.exe foreign_test.py → data/foreign_*.csv
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import analytics as A
import causal_grain as CG
import experiments as EX
import procurement as PR
import range_bt as RB

warnings.filterwarnings("ignore")
W = pd.read_parquet(CG.DATA / "reports" / "wasde_world.parquet")
CLIP = 15.0          # 통계 정의 변경(중국 재고 개정 · EU 범위 변경)으로 생기는 극단 수정 자르기 — 돌리기 전 고정
NODES = {  # 이름: (품목, 지역들, 항목)
    "남미 옥수수 생산 수정": ("corn", ["Brazil", "Argentina"], "production"),
    "흑해 옥수수 생산 수정": ("corn", ["Ukraine", "Russia"], "production"),
    "해외 옥수수 재고 수정": ("corn", ["Total Foreign", "-China"], "ending"),
    "남미 대두 생산 수정": ("soybean", ["Brazil", "Argentina", "Paraguay"], "production"),
    "중국 대두 수입 수정": ("soybean", ["China"], "imports"),
    "해외 대두 재고 수정": ("soybean", ["Total Foreign", "-China"], "ending"),
    "흑해 밀 생산 수정": ("wheat", ["Russia", "Ukraine", "Kazakhstan"], "production"),
    "기타 수출국 밀 생산 수정": ("wheat", ["EU", "Canada", "Australia", "Argentina"], "production"),
    "해외 밀 재고 수정": ("wheat", ["Total Foreign", "-China"], "ending"),
}
NEW_PA = {"옥수수": ["남미 옥수수 생산 수정", "흑해 옥수수 생산 수정", "해외 옥수수 재고 수정"],
          "대두": ["남미 대두 생산 수정", "중국 대두 수입 수정", "해외 대두 재고 수정"],
          "밀": ["흑해 밀 생산 수정", "기타 수출국 밀 생산 수정", "해외 밀 재고 수정"],
          "대두박": ["남미 대두 생산 수정"], "대두유": ["남미 대두 생산 수정"]}
CG.GRAPHS[3] = (CG.ROOTS2 + list(NODES), {k: v + NEW_PA.get(k, []) for k, v in CG.PA2.items()})


def foreign(idx: pd.DatetimeIndex) -> pd.DataFrame:
    rel = sorted(W.release.unique())
    out = {k: {} for k in NODES}
    for p, r in zip(rel[:-1], rel[1:]):
        a, b = W[W.release == p], W[W.release == r]
        for nm, (c, regs, item) in NODES.items():
            sel = lambda d: d[(d.commodity == c) & (d.item == item) & d.region.isin([g.lstrip("-") for g in regs])].set_index(["region", "my"]).value
            x, y = sel(a), sel(b)
            com = x.index.intersection(y.index)
            if not len(com):
                continue
            sg = pd.Series([-1.0 if ("-" + rg) in regs else 1.0 for rg, _ in com], index=com)   # '-China' = 빼기 (해외 − 중국)
            if (sg < 0).any() and not (sg > 0).any():
                continue
            base = (x[com] * sg).sum()
            out[nm][r] = float(np.clip(100 * ((y[com] - x[com]) * sg).sum() / max(abs(base), 1e-6), -CLIP, CLIP))   # 비교 가능한 연도 합 대비 %
    F = pd.DataFrame({k: pd.Series(v) for k, v in out.items()})
    F.index = CG._friday(pd.Series(pd.to_datetime(F.index))).values
    return F.groupby(level=0).sum().reindex(idx).fillna(0.0)


def nw(y, X, lag):
    X1 = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(X1, y, rcond=None)[0]
    e = y - X1 @ b
    XtXi = np.linalg.pinv(X1.T @ X1)
    S = (X1 * e[:, None]).T @ (X1 * e[:, None])
    for k in range(1, lag + 1):
        G = (X1[k:] * e[k:, None]).T @ (X1[:-k] * e[:-k, None])
        S += (1 - k / (lag + 1)) * (G + G.T)
    se = np.sqrt(np.diag(XtXi @ S @ XtXi))
    return b[1:], b[1:] / se[1:]


def run():
    sys.stdout.reconfigure(errors="replace")
    pd.set_option("display.width", 220)
    La, Lr, _, _ = EX._data()
    X2 = CG.nodes(La)
    X = pd.concat([X2, foreign(X2.index)], axis=1)
    rw = X[list(NODES)].abs().sum(1) > 0                                                   # WASDE 발표 주
    cut = pd.Timestamp("2016-12-31")
    tr, te = X[X.index <= cut], X[X.index > cut]

    print("A. 구조 모형 2판 → 3판 (해외 수급 수정 추가)")
    rows = []
    for v in (2, 3):
        CG.set_graph(v)
        M, Mt = CG.fit_scm(X), CG.fit_scm(tr)
        D = CG.decompose(M, X)
        for p in CG.PRICES:
            y = te[p].values[1:]
            f = Mt[p]["c"] + sum(b * te[q].values[1:] for q, b in Mt[p]["b"].items()) + sum(b * te[q].values[:-1] for q, b in Mt[p]["l"].items())
            m = rw.loc[te.index].values[1:]
            rows.append(dict(판=v, 가격=p, 설명몫=round(1 - np.var(D["C"][p][p].values) / np.var(X[p].values), 3),
                             시험R2=round(1 - np.mean((y - f) ** 2) / np.var(y), 3),
                             시험R2_발표주=round(1 - np.mean((y[m] - f[m]) ** 2) / np.var(y[m]), 3)))
    SA = pd.DataFrame(rows).pivot(index="가격", columns="판")
    print(SA.to_string())
    M1, M2 = CG.fit_scm(tr), CG.fit_scm(te)
    er = [dict(연결=f"{q} → {p}", 계수_08_16=round(M1[p]["b"][q], 3), t_08_16=round(M1[p]["t"][q], 1), 계수_17_=round(M2[p]["b"][q], 3), t_17_=round(M2[p]["t"][q], 1))
          for p, qs in NEW_PA.items() for q in qs]
    EA = pd.DataFrame(er)
    print(EA.to_string(index=False))

    print("\nB. 수정 발표 뒤 4주 수익률 (발표 다음 주부터) — 발표 주 반응과 같은 부호면 지연 반응(계속), 반대면 되돌림")
    rows = []
    for p, qs in NEW_PA.items():
        fwd = X[p].shift(-1).rolling(4).sum().shift(-3)                                    # t+1..t+4
        for q in qs:
            for per, m in (("2008–16", X.index <= cut), ("2017–", X.index > cut)):
                s = rw & m & fwd.notna() & (X[q] != 0)
                if s.sum() < 20:
                    continue
                b, t = nw(fwd[s].values, X.loc[s, [q]].values, 3)
                b0, t0 = nw(X.loc[s, p].values, X.loc[s, [q]].values, 0)
                rows.append(dict(노드=q, 가격=p, 기간=per, 발표수=int(s.sum()), 발표주_반응=round(b0[0], 2), 발표주_t=round(t0[0], 1),
                                 이후4주_계수=round(b[0], 2), 이후4주_t=round(t[0], 1)))
    B = pd.DataFrame(rows)
    print(B.to_string(index=False))

    print("\nC-1. 구간 확률 (RPS, 시험 2017–): M2 3판 부모 vs M1 변동성")
    P = LS_weekly(La, X.index)
    rows = []
    for h in (4, 13):
        for p in CG.PRICES:
            R = RB.run_one(P[p], X, {"M2 3판": [p] + CG.GRAPHS[3][1][p], "M2 2판": [p] + CG.PA2[p]}, h)
            Wd = R.pivot(index="t", columns="모형", values="RPS")
            for k in ("M2 2판", "M2 3판"):
                d = (Wd["M1 변동성"] - Wd[k]).values
                rows.append(dict(기간=f"{h}주", 가격=p, 모형=k, RPS_개선pct=round(100 * d.mean() / Wd["M1 변동성"].mean(), 1), t=round(RB.nw_t(d, h - 1), 2)))
    C1 = pd.DataFrame(rows).pivot_table(index=["기간", "가격"], columns="모형", values=["RPS_개선pct", "t"])
    print(C1.to_string())

    print("\nC-2. 구매 미루기 (대시보드 규칙, 시험 AUC · 판단당 절감 bp)")
    P_ = PR.PRESETS["추천 (고도화: 상대가치+계절+MIDAS)"]
    rows = []
    for i, nm in {"yf:ZC=F": "옥수수", "yf:ZS=F": "대두", "yf:ZW=F": "밀", "yf:ZM=F": "대두박", "yf:ZL=F": "대두유", "yf:KE=F": "KC 밀"}.items():
        inp = PR.inputs(La, Lr, i, "midas", "price")
        sig = PR.signals(La[i], PR.item_lookback(i, P_["lookback"]), inp["rel"], inp["rel_sign"], inp["vol_fc"], inp["crisis"], P_["season"])
        Wf = PR.weekly_frame(Lr[i], sig, PR.score(sig, P_["w"]), start="2008-01-01")
        par = NEW_PA["밀" if "밀" in nm else nm]
        for h in (2, 4):
            for k, ex in {"기본": None, "+해외": X[par].rolling(4).sum().add_suffix("_4주")}.items():
                F = PR.fc_walkforward(Wf, h, extra=ex, first_test_year=2013)
                for per, (a_, b_) in {"2013–16": ("2013-01-01", "2016-12-31"), "2017–": ("2017-01-01", None)}.items():
                    m_ = (F.index >= pd.Timestamp(a_)) & ((F.index <= pd.Timestamp(b_)) if b_ else True) & F.p_down.notna() & F.y_down.notna()
                    S = PR.defer_summary(PR.backtest_defer(Wf, F, PR.DEFER_TAU, PR.DEFER_PSTAR, h, "adaptive", a_, b_))
                    rows.append(dict(품목=nm, 미룸주=h, 정보=k, 기간=per, AUC=round(roc_auc_score(F.y_down[m_], F.p_down[m_]), 3),
                                     절감bp=round(S["판단당 평균 절감 (bp)"], 1)))
    C2 = pd.DataFrame(rows)
    g = C2.pivot_table(index=["미룸주", "품목", "기간"], columns="정보", values=["AUC", "절감bp"])
    d = pd.DataFrame({"AUC_기본": g["AUC"]["기본"], "AUC_해외": g["AUC"]["+해외"], "절감Δbp": g["절감bp"]["+해외"] - g["절감bp"]["기본"]}).unstack("기간")
    d["통과"] = (d[("절감Δbp", "2013–16")] > 0) & (d[("절감Δbp", "2017–")] > 0) & (d[("AUC_해외", "2017–")] > d[("AUC_기본", "2017–")])
    print(d.round(3).to_string())

    SA.to_csv("data/foreign_scm_share.csv", encoding="utf-8-sig"); EA.to_csv("data/foreign_scm_edges.csv", index=False, encoding="utf-8-sig")
    B.to_csv("data/foreign_drift.csv", index=False, encoding="utf-8-sig"); C1.to_csv("data/foreign_range.csv", encoding="utf-8-sig")
    C2.to_csv("data/foreign_defer.csv", index=False, encoding="utf-8-sig")


def LS_weekly(La, idx):
    import ls_strategies as LS
    return LS.weekly(La, list(CG.PX.values()), CG.START).rename(columns={v: k for k, v in CG.PX.items()}).reindex(idx)


if __name__ == "__main__":
    run()
