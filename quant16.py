"""퀀트 전략 16개(옵션 2개 제외) 한눈에 — 같은 조건 백테스트 + 기존 실험 결과 요약 → data/quant16_compare.csv.
발표 5장(4×4 카드)·구매 판단 검증 백로그 '퀀트 전략 16' 시트가 이 파일을 읽음.

롱숏 전략: 대시보드 ⑤ 롱숏 틀 그대로 (19품목 주간 · 비용 5bp · 목표 변동성 10% · 2007– 평가, 2007–16 / 2017– 분할).
혼합 시험(사전 고정): 기준 혼합(MinVar 50 + 모멘텀 25 + 가치 25)을 45 / 22.5 / 22.5로 줄이고 그 전략 10%를 더해 차이 t.
채택 규칙(사전 고정): 혼합 차이 t ≥ 2 이고 두 기간 모두 차이 > 0 일 때만 '채택 후보'.
실행: .venv\\Scripts\\python.exe quant16.py   (약 1~2분)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import analytics as A
import hedge_core as H
import ls_extra as LX
import ls_strategies as LS
import risk as RK

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = DATA / "quant16_compare.csv"
TGT, COST, SPLIT, ANN = 10.0, 5.0, "2017-01-01", LS.ANN
ADOPT, PART, NO, NA = "● 채택", "◐ 부분 적용", "✕ 시험 후 미채택", "— 해당 없음"


def _setup():
    L = A.load_levels()
    P = LS.weekly(L, LS.BASKET, "2001-01-01"); P = P[[c for c in LS.BASKET if c in P]]
    return L, P, LS._ret(P), np.log(P)


def book(W, R):
    return LS._book(LS._vol_target(W, R, TGT), R, COST)


def rescale(r):
    pv = r.rolling(26, min_periods=13).std().shift(1) * np.sqrt(ANN)
    return r * (TGT / pv).clip(upper=4.0).fillna(0.0)


def mix(parts, idx):
    m = sum(w * s.reindex(idx).fillna(0) for w, s in parts)
    pv = m.rolling(26, min_periods=13).std().shift(1) * np.sqrt(ANN)
    return m * (TGT / pv).clip(upper=4.0).fillna(1.0)


def donchian_weights(P, R, n_in=55, n_out=20):
    """돌파(터틀식): n_in주 최고가 돌파 → 롱, 최저가 붕괴 → 숏, 반대쪽 n_out주 채널에서 청산. 품목 변동성 역가중."""
    hi_in, lo_in = P.rolling(n_in, min_periods=n_in).max().shift(1), P.rolling(n_in, min_periods=n_in).min().shift(1)
    hi_out, lo_out = P.rolling(n_out, min_periods=n_out).max().shift(1), P.rolling(n_out, min_periods=n_out).min().shift(1)
    S = pd.DataFrame(0.0, index=P.index, columns=P.columns)
    for c in P:
        pos, out = 0.0, []
        for p, hi, lo, ho, lw in zip(P[c].values, hi_in[c].values, lo_in[c].values, hi_out[c].values, lo_out[c].values):
            if not (np.isfinite(p) and np.isfinite(hi)):
                out.append(0.0); continue
            if pos == 0:
                pos = 1.0 if p > hi else (-1.0 if p < lo else 0.0)
            elif pos > 0 and p < lw:
                pos = -1.0 if p < lo else 0.0
            elif pos < 0 and p > ho:
                pos = 1.0 if p > hi else 0.0
            out.append(pos)
        S[c] = out
    vol = R.rolling(26, min_periods=13).std() * np.sqrt(ANN)
    W = S * (TGT / vol)
    return W.div(W.notna().sum(1).replace(0, np.nan), axis=0).clip(-2, 2).fillna(0.0)


def pca_resid_weights(R, k=3, win=104, sw=26, enter=1.25, exit_=0.5):
    """PCA 잔차 통계적 차익 (Avellaneda & Lee 2010 주간판): 주성분 k개를 뺀 잔차의 최근 sw주 누적 이탈 s-점수 → 역방향, 시장 중립."""
    Rd = R / 100
    Sg = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
    for t in range(win, len(R)):
        X = Rd.iloc[t - win + 1:t + 1]
        cols = [c for c in X if X[c].notna().all()]
        if len(cols) < 10:
            continue
        X = X[cols].values; Z = (X - X.mean(0)) / X.std(0)
        Vt = np.linalg.svd(Z, full_matrices=False)[2]
        Fc = np.column_stack([np.ones(win), Z @ Vt[:k].T])
        E = X - Fc @ np.linalg.lstsq(Fc, X, rcond=None)[0]
        C = np.cumsum(E, axis=0)[-sw:]
        Sg.iloc[t, [R.columns.get_loc(c) for c in cols]] = (C[-1] - C.mean(0)) / C.std(0)
    St = pd.DataFrame(0.0, index=Sg.index, columns=Sg.columns)
    for c in Sg:
        pos, o = 0.0, []
        for v in Sg[c].values:
            if not np.isfinite(v):
                pos = 0.0
            elif pos == 0 and abs(v) > enter:
                pos = -np.sign(v)
            elif pos != 0 and abs(v) < exit_:
                pos = 0.0
            o.append(pos)
        St[c] = o
    W = St / (R.rolling(26, min_periods=13).std() * np.sqrt(ANN))
    W = W.sub(W.mean(1), axis=0)
    return W.div(W.abs().sum(1).replace(0, np.nan), axis=0).fillna(0.0)


def ls_metrics(r, base=None, tsm=None):
    r = r[r.index >= "2007-01-01"].fillna(0)
    m = LS.metrics(r)
    d = {"Sharpe": m["Sharpe"], "Sharpe 2007–16": LS.metrics(r[r.index < SPLIT])["Sharpe"], "Sharpe 2017–": LS.metrics(r[r.index >= SPLIT])["Sharpe"],
         "최대낙폭 (%)": m["최대낙폭 (%)"], "52주 모멘텀과 상관": float(r.corr(tsm.reindex(r.index).fillna(0))) if tsm is not None else np.nan}
    return d


def blend_test(x, mvb, tsm, v1, base, idx):
    m = mix([(0.45, mvb), (0.225, tsm), (0.225, v1), (0.10, x)], idx)
    m = m[m.index >= "2007-01-01"]; dl = m - base.reindex(m.index).fillna(0)
    pre, post = dl[dl.index < SPLIT], dl[dl.index >= SPLIT]
    return {"혼합 Sharpe (10% 추가)": LS.metrics(m)["Sharpe"], "혼합 차이 t": float(dl.mean() / dl.std() * np.sqrt(len(dl))),
            "혼합 차이 2007–16 (연 %p)": float(pre.mean() * ANN), "혼합 차이 2017– (연 %p)": float(post.mean() * ANN)}


def run():
    L, P, R, lp = _setup()
    tsm = book(LS.tsmom_weights(P, 52, port_tgt=None), R)
    mvb = book(LS.alloc_weights_cap(P, 0.25), R)
    v1 = book(LX.xs_weights(-(lp - lp.rolling(260, min_periods=156).mean())), R)
    idx = mvb.index
    base = mix([(0.5, mvb), (0.25, tsm), (0.25, v1)], idx); base = base[base.index >= "2007-01-01"]
    S = {
        1: tsm, 2: v1,
        5: book(LX.kalman_trend_weights(P, 0.001), R),
        14: book(donchian_weights(P, R, 55, 20), R),
        13: book(LX.xs_weights(-R, hold=1), R),
        4: rescale(LS.pairs_book(P, COST)[0]),
        3: book(pca_resid_weights(R), R),
        15: book(LS.alloc_weights(P, "iv"), R),
        8: book(LX.xs_weights(lp.diff(26).rank(axis=1, pct=True) + (-(lp - lp.rolling(260, min_periods=156).mean())).rank(axis=1, pct=True)), R),
        12: book(LS.xsmom_weights(P, 26, 4), R),
    }
    M = {k: ls_metrics(v, tsm=tsm) for k, v in S.items()}
    for k in (5, 14, 13, 4, 3, 15, 8, 12):
        M[k].update(blend_test(S[k], mvb, tsm, v1, base, idx))
    M["base"] = ls_metrics(base, tsm=tsm)
    # 캐리: 기존 ⑥ 캐리 실험 (선물 기간구조 자료가 있는 미국 에너지 4 · 중국 8)
    car = LS.run_carry(L, COST)
    carry_txt = "; ".join(f"{nm.split(' (')[0]} 롱숏 Sharpe {v['table']['Sharpe'].iloc[-1]:.2f}" for nm, v in car.items()) if car else "자료 없음"
    # 기존 실험 결과 파일
    pg = pd.read_csv(DATA / "predict_grid_compare.csv", encoding="utf-8-sig")
    ml_dir, ml_rank = int((pg["방향 위약 대비 차이 t"] > 2).sum()), int((pg["순위 롱숏 위약 대비 차이 t"] > 2).sum())
    best_ml = pg.sort_values("순위 롱숏 Sharpe", ascending=False).iloc[0]
    rg = pd.read_csv(DATA / "regime_mv_compare.csv", encoding="utf-8-sig"); rgo = rg[rg["기간"] == "전체 OOS"]
    rg_pass = int((rgo["위약 p"] < 0.05).sum())
    ev = {}
    for i, nm in {"yf:ZC=F": "옥수수", "yf:ZS=F": "대두", "yf:ZM=F": "대두박", "yf:ZW=F": "밀"}.items():
        r = RK.returns(L[i]); o = RK.vol_models(r, ["GARCH(1,1)"]); ev[nm] = H.event_multipliers(o["r"], o["S"]["GARCH(1,1)"])
    pd.DataFrame(ev).T.to_csv(DATA / "quant16_events.csv", encoding="utf-8-sig")

    f2 = lambda k, c="Sharpe": f"{M[k][c]:.2f}"
    def ls_line(k):
        return f"Sharpe {f2(k)} (07–16 {f2(k, 'Sharpe 2007–16')} · 17– {f2(k, 'Sharpe 2017–')})"
    def bl_line(k):
        return f"혼합에 10% 넣으면 {M['base']['Sharpe']:.2f}→{M[k]['혼합 Sharpe (10% 추가)']:.2f} (t {M[k]['혼합 차이 t']:+.1f})"
    def verdict(k, yes=PART):
        ok = M[k]["혼합 차이 t"] >= 2 and M[k]["혼합 차이 2007–16 (연 %p)"] > 0 and M[k]["혼합 차이 2017– (연 %p)"] > 0
        return yes if ok else NO
    corn_q = ev["옥수수"]["분기 보고서"]
    rows = [
        # (행, 번호, 전략, 영문, 한 줄 설명, 대시보드 위치, 결과 1줄, 결과 2줄, 판단)
        ("추세", 1, "모멘텀", "Momentum", "지난 1년 오른 품목 롱, 내린 품목 숏", "⑤ 포트폴리오 · 3대 전략",
         ls_line(1), "기준 혼합의 한 축 (혼합 Sharpe 0.88)", ADOPT),
        ("추세", 5, "추세 추종 (칼만)", "Trend Following", "잡음을 걸러낸 추세 기울기 방향으로", "⑤ 롱숏 ⑨ 확장",
         ls_line(5), bl_line(5), verdict(5)),
        ("추세", 14, "돌파 매매", "Breakout", "55주 고점 돌파 매수, 저점 붕괴 매도", "없음 (2026-11 시험)",
         ls_line(14), f"모멘텀과 상관 {M[14]['52주 모멘텀과 상관']:.2f} — 같은 신호", verdict(14)),
        ("추세", 13, "단기 반전", "Short-Term Reversal", "지난주 급등 품목 숏, 급락 품목 롱", "패널 특성 (REV 1P)",
         ls_line(13), bl_line(13), verdict(13)),
        ("되돌림", 2, "평균회귀 (가치)", "Mean Reversion", "5년 평균보다 싼 품목 롱, 비싼 품목 숏", "⑤ 포트폴리오 · 3대 전략",
         ls_line(2), "기준 혼합의 한 축", ADOPT),
        ("되돌림", 4, "페어 트레이딩", "Pairs Trading", "같이 움직이는 두 품목 간격이 벌어지면 역으로", "⑤ 롱숏 ①",
         ls_line(4), bl_line(4), verdict(4)),
        ("되돌림", 3, "통계적 차익 (PCA)", "Statistical Arbitrage", "공통 요인을 뺀 품목 고유 이탈의 되돌림", "없음 (2026-11 시험)",
         ls_line(3), "이탈이 되돌아오지 않음 (원자재 고유 충격 지속)", verdict(3)),
        ("되돌림", 9, "캐리", "Carry", "백워데이션(근월물 비쌈) 롱, 콘탱고 숏", "⑤ 롱숏 ⑥ 캐리",
         carry_txt, "기간·시장에 따라 부호가 바뀜 — 견고하지 않음", NO),
        ("비중·위험", 15, "리스크 패리티", "Risk Parity", "금액이 아니라 위험을 품목마다 같게", "⑤ 비중 결정 방식 비교",
         ls_line(15) + " · 역변동성 근사", "최소분산(MinVar)이 더 나아 혼합엔 MinVar 사용", PART),
        ("비중·위험", 16, "국면 전환", "Regime Switching", "위기·평시를 판별해 위험을 줄이거나 늘림", "② 레짐 · ◆ 리스크",
         f"방향 예측: 위약 통과 {rg_pass}/{len(rgo)}칸 (우연 수준)", "위기 확률은 위험 관리·구매 위험 판단에 사용", PART),
        ("비중·위험", 8, "팩터 투자", "Factor Investing", "가치 + 모멘텀 순위를 합친 다중 팩터", "패널 모델 (13개 특성)",
         ls_line(8), bl_line(8), verdict(8)),
        ("비중·위험", 12, "횡단면 알파", "Cross-Sectional Alpha", "매달 품목 점수 순위 → 상위 롱 · 하위 숏", "④ 패널 · 순위 롱숏",
         ls_line(12), bl_line(12), verdict(12)),
        ("데이터·실행", 17, "머신러닝 알파", "Machine Learning Alpha", "모델이 자료에서 수익 패턴을 찾음", "④ 예측 그리드 · 대안 시계열",
         f"위약 대비 t>2: 방향 {ml_dir}/{len(pg)} · 순위 {ml_rank}/{len(pg)}", f"최고 순위 롱숏 Sharpe {best_ml['순위 롱숏 Sharpe']:.2f} ({best_ml['모델']})", NO),
        ("데이터·실행", 7, "이벤트 드리븐", "Event-Driven", "USDA 보고서 전후 움직임 (헤지 시점용)", "헤지 랩 A1 (2026-11)",
         f"분기 보고서 날 변동 옥수수 {corn_q:.1f}배", "방향 예측력은 재현 안 됨 → VaR 보정에만 사용", PART),
        ("데이터·실행", 18, "집행 (분할 매수)", "Execution", "점수·위험에 따라 매수 속도를 조절", "★ 구매 의사결정",
         "균등 분할 대비 +13bp (t 2.35)", "07–16 +14 · 17– +12bp — 추천 프리셋 운영 중", ADOPT),   # 검증 백로그 v1.1.0 (experiments.BACKLOG)
        ("데이터·실행", 6, "시장조성", "Market Making", "매수·매도 호가를 동시에 내고 차이를 챙김", "없음",
         "호가창·초단타 체계 필요", "구매·위험 관리 목적과 무관", NA),
    ]
    T = pd.DataFrame(rows, columns=["행", "번호", "전략", "영문", "한 줄 설명", "대시보드 위치", "결과 1", "결과 2", "판단"])
    num = pd.DataFrame({k: M[k] for k in M if k != "base"}).T
    T = T.merge(num, left_on="번호", right_index=True, how="left")
    T["시험일"] = "2026-11-02"; T["시험 조건"] = "19품목 주간 · 비용 5bp · 목표 변동성 10% · 2007– (롱숏), 그 외 기존 실험 결과"
    T.to_csv(OUT, index=False, encoding="utf-8-sig")
    return T


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    T = run()
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print(T[["번호", "전략", "결과 1", "결과 2", "판단"]].to_string(index=False))
