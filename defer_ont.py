"""구매 판단 '지금 살까, 미룰까' + 온톨로지 백테스트 — 기존 파일은 고치지 않음.

규칙은 대시보드와 같음 (procurement.backtest_defer): 점수 ≥ τ 0.6 → 즉시, 아니고 P(h주 뒤 더 쌈) ≥ p* 0.55 → 미룸(최대 h주, 매주 재평가).
바꾸는 것은 하락 확률 모형(walk-forward 로지스틱, 매년 재학습)에 넣는 정보만:
  기본      : 대시보드 특성 (점수·구성요소·최근 수익률·변동성)
  +원인     : 기본 + 온톨로지 부모 노드(기상·작황·WASDE·위성·수출 판매·운임·환율·원유·연결 품목)의 최근 4주 합
  +분해     : 기본 + 최근 4주 가격 변동을 '온톨로지로 설명되는 몫'과 '설명 안 되는 몫(자기 충격)'으로 나눈 두 값
              (구조 모형은 그해 1월 이전 자료로만 다시 추정 → 미래정보 없음). 가설: 설명 안 되는 움직임은 되돌림이 더 큼.
채택 기준(돌리기 전 고정): 기본 대비 판단당 절감이 2013–16(OOS 예측)과 2017– 모두 + 이고 시험 AUC도 + .
실행: .venv\\Scripts\\python.exe defer_ont.py → data/defer_ont_compare.csv
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import causal_grain as CG
import experiments as EX
import procurement as PR

warnings.filterwarnings("ignore")
ITEMS = {"yf:ZC=F": "옥수수", "yf:ZS=F": "대두", "yf:ZW=F": "밀", "yf:ZM=F": "대두박", "yf:ZL=F": "대두유"}
TAU, PSTAR, MODE, SPLIT = PR.DEFER_TAU, PR.DEFER_PSTAR, "adaptive", "2017-01-01"


def decomposition(X: pd.DataFrame, p: str) -> pd.DataFrame:
    """해마다 그 이전 자료로 구조 모형을 추정해 그해 주간 변동 = 설명 몫 + 자기 충격으로 나눔 (4주 합)."""
    CG.set_graph(2)
    out = []
    for yr in range(2010, X.index[-1].year + 1):
        tr = X[X.index < pd.Timestamp(f"{yr}-01-01")]
        m = CG.fit_scm(tr)[p]
        Y = X[X.index.year == yr]
        prev = X.shift(1).loc[Y.index]
        fit = m["c"] + sum(b * Y[q] for q, b in m["b"].items()) + sum(b * prev[q] for q, b in m["l"].items())
        out.append(pd.DataFrame({"설명몫": fit - m["c"], "자기충격": Y[p] - fit}))
    D = pd.concat(out)
    return D.rolling(4).sum().add_suffix("_4주")


def run():
    sys.stdout.reconfigure(errors="replace")
    pd.set_option("display.width", 220)
    La, Lr, _, _ = EX._data()
    X = CG.nodes(La)
    P_ = PR.PRESETS["추천 (고도화: 상대가치+계절+MIDAS)"]
    rows = []
    for i, nm in ITEMS.items():
        inp = PR.inputs(La, Lr, i, "midas", "price")
        sig = PR.signals(La[i], PR.item_lookback(i, P_["lookback"]), inp["rel"], inp["rel_sign"], inp["vol_fc"], inp["crisis"], P_["season"])
        W_ = PR.weekly_frame(Lr[i], sig, PR.score(sig, P_["w"]), start="2008-01-01")   # 온톨로지 자료가 2008~ → 세 모형 같은 학습 구간
        par = [q for q in CG.PA2[nm] if q != nm]
        extras = {"기본": None,
                  "+원인": X[par].rolling(4).sum().add_suffix("_4주"),
                  "+분해": decomposition(X, nm)}
        for h in (2, 4):
            for k, ex in extras.items():
                F = PR.fc_walkforward(W_, h, extra=ex, first_test_year=2013)
                for per, (a_, b_) in {"2013–16": ("2013-01-01", "2016-12-31"), "2017–": (SPLIT, None)}.items():
                    m_ = (F.index >= pd.Timestamp(a_)) & ((F.index <= pd.Timestamp(b_)) if b_ else True) & F.p_down.notna() & F.y_down.notna()
                    auc = roc_auc_score(F.y_down[m_], F.p_down[m_]) if m_.sum() > 50 else np.nan
                    S = PR.defer_summary(PR.backtest_defer(W_, F, TAU, PSTAR, h, MODE, a_, b_))
                    rows.append(dict(품목=nm, 미룸주=h, 정보=k, 기간=per, AUC=round(auc, 3), 판단당_절감bp=round(S["판단당 평균 절감 (bp)"], 1),
                                     미룸비율=round(S["미룸 비율"], 2), 미룬경우_적중률=round(S["미룬 경우 적중률"], 2), 위약bp=round(S["위약 평균 (bp)"], 1),
                                     위약초과확률=round(S["위약 초과 확률"], 2)))
        print(f"  {nm} 완료", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv("data/defer_ont_compare.csv", index=False, encoding="utf-8-sig")
    piv = R.pivot_table(index=["미룸주", "품목"], columns=["정보", "기간"], values=["AUC", "판단당_절감bp"])
    print(piv.round(3).to_string())
    for k in ("+원인", "+분해"):
        g = R.pivot_table(index=["미룸주", "품목", "기간"], columns="정보", values=["판단당_절감bp", "AUC"])
        d = (g["판단당_절감bp"][k] - g["판단당_절감bp"]["기본"]).unstack("기간")
        a = (g["AUC"][k] - g["AUC"]["기본"]).unstack("기간")
        ok = (d["2013–16"] > 0) & (d["2017–"] > 0) & (a["2017–"] > 0)
        print(f"\n{k} vs 기본: 절감 차이(bp) · AUC 차이 · 채택 기준 통과")
        print(pd.concat([d.round(1).add_prefix("절감Δ_"), a.round(3).add_prefix("AUCΔ_"), ok.rename("통과")], axis=1).to_string())


if __name__ == "__main__":
    run()
