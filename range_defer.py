"""범위(하락 꼬리 확률) 기반 '일부 물량 미루기' 백테스트 — 기존 파일은 고치지 않음.

매주 1단위 구매. P(4주 뒤 −5% 아래) ≥ q* 이면 그 주 물량의 50%를 미룸:
  고정형 : 미룬 물량을 정확히 4주 뒤 매수
  지정가형: 4주 안에 결정가 대비 −5% 이하 주간 종가가 나오면 그 주에 매수, 아니면 4주 뒤 매수
확률 = range_bt의 M1 변동성 모형(EWMA λ 0.94 × √4, 표준화 잔차의 과거 분포, 13주마다 재학습, 정답 확정 구간만 학습).
q* 는 2010–16(학습)에서 판단당 절감이 가장 큰 값을 고르고 2017– 시험에서 확인. 위약 = 같은 비율의 주를 무작위로 같은 방식 미룸.
참고: 가격이 마팅게일이면 어떤 미루기 규칙도 기대 원가는 같음(선택적 정지 정리) → 이득이 나오면 그건 '큰 변동 뒤 되돌림' 같은 실제 패턴 덕분.
실행: .venv\\Scripts\\python.exe range_defer.py → data/range_defer_compare.csv
"""
import sys
import warnings

import numpy as np
import pandas as pd

import experiments as EX
import procurement as PR

warnings.filterwarnings("ignore")
ITEMS = {"yf:ZC=F": "옥수수", "yf:ZS=F": "대두", "yf:ZW=F": "밀", "yf:ZM=F": "대두박", "yf:ZL=F": "대두유", "yf:KE=F": "KC 밀(확인용)"}
H, DROP, FRAC, LAM, REFIT = 4, -5.0, 0.5, 0.94, 13
QS = (0.10, 0.15, 0.20, 0.25, 0.30)
PER = {"2010–16": ("2010-01-01", "2016-12-31"), "2017–": ("2017-01-01", None)}


def p_drop(P: pd.Series) -> pd.Series:
    """P(H주 뒤 로그 수익률 < DROP%) — 변동성 모형, 그 시점까지 확정된 정답으로만 학습."""
    r1 = 100 * np.log(P).diff()
    fwd = 100 * np.log(P.shift(-H) / P)
    sig = np.sqrt(H * r1.pow(2).ewm(alpha=1 - LAM, adjust=False).mean())
    out, fit, last = pd.Series(np.nan, index=P.index), None, None
    for k, t in enumerate(P.index):
        if k < 156 + H or not np.isfinite(sig.iloc[k]):
            continue
        if last is None or k - last >= REFIT:
            tr = slice(52, k - H + 1)                                                      # t−H까지: 정답 확정
            y, s = fwd.iloc[tr].values, sig.iloc[tr].values
            ok = np.isfinite(y) & np.isfinite(s)
            mu = y[ok].mean()
            fit = (mu, np.sort((y[ok] - mu) / s[ok]))
            last = k
        mu, zs = fit
        out.iloc[k] = np.searchsorted(zs, (DROP - mu) / sig.iloc[k], side="right") / len(zs)
    return out


def simulate(px: np.ndarray, defer: np.ndarray, mode: str) -> np.ndarray:
    """판단 주별 절감 bp (+ = 즉시 매수보다 쌈). 미룬 50%만 차이를 만듦."""
    n, sv = len(px), np.zeros(len(px))
    for t in np.flatnonzero(defer):
        if t + H >= n:
            continue
        e = px[t + H]
        if mode == "지정가":
            hit = [k for k in range(1, H + 1) if 100 * np.log(px[t + k] / px[t]) <= DROP]
            e = px[t + hit[0]] if hit else px[t + H]
        sv[t] = FRAC * (px[t] - e) / px[t] * 1e4
    return sv


def run():
    sys.stdout.reconfigure(errors="replace")
    pd.set_option("display.width", 220)
    La, Lr, _, _ = EX._data()
    rows = []
    rng = np.random.default_rng(0)
    for i, nm in ITEMS.items():
        pw = La[i].resample("W-FRI").last().dropna()
        pr = p_drop(pw)
        ex = Lr[i].resample("W-FRI").last().reindex(pw.index).values                      # 체결은 실제 가격(대시보드와 같음)
        ex_adj = pw.values                                                                 # 롤 이음새 없는 가격으로도 확인
        for mode in ("고정", "지정가"):
            best_q, best = None, -np.inf
            for q in QS:                                                                   # q* 는 학습 기간에서만 고름
                a, b = PER["2010–16"]
                m = (pw.index >= a) & (pw.index <= b)
                d = (pr.values >= q) & m
                s = simulate(ex_adj, d, mode)[m].mean()
                if s > best:
                    best, best_q = s, q
            for per, (a, b) in PER.items():
                m = (pw.index >= a) & ((pw.index <= b) if b else True) & np.isfinite(pr.values)
                m[-H:] = False
                d = (pr.values >= best_q) & m
                for px_nm, px in (("실제가", ex), ("롤조정가", ex_adj)):
                    sv = simulate(px, d, mode)
                    k, n = int(d.sum()), int(m.sum())
                    idx_m = np.flatnonzero(m)
                    pl = [simulate(px, np.isin(np.arange(len(px)), rng.choice(idx_m, k, replace=False)), mode)[m].mean() for _ in range(300)] if k else [0]
                    rows.append(dict(품목=nm, 방식=mode, q별=best_q, 기간=per, 가격=px_nm, 판단수=n, 미룸비율=round(k / max(n, 1), 2),
                                     판단당_절감bp=round(sv[m].mean(), 1), 미룬경우_평균bp=round(sv[d].mean(), 1) if k else np.nan,
                                     미룬경우_적중률=round((sv[d] > 0).mean(), 2) if k else np.nan,
                                     위약_평균bp=round(np.mean(pl), 1), 위약_초과확률=round(np.mean(np.array(pl) < sv[m].mean()), 2)))
        print(f"  {nm} 완료 (q* 고정 {best_q})", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv("data/range_defer_compare.csv", index=False, encoding="utf-8-sig")
    print(R[R.가격 == "롤조정가"].drop(columns="가격").to_string(index=False))
    print("\n실제가 기준 (대시보드 체결 가격):")
    print(R[R.가격 == "실제가"].pivot_table(index=["품목", "방식"], columns="기간", values="판단당_절감bp").to_string())


if __name__ == "__main__":
    run()
