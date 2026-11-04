"""3축 혼합 추가 백테스트(⑤ 롱숏 탭 ⑪)의 규칙별 주간 수익률 저장 — 결과표(strategy_*_compare.csv) 한 줄 = 주간 수익률 한 열.

각 실험 스크립트의 main()을 그대로 다시 돌리면서 점수 함수(strategy_meta.score, strategy_voltgt_dyn.stats)에 들어간 수익률을 엿듣고,
결과표 각 줄의 Sharpe(2008– · 학습 2008–16 · 시험 2017–)와 값이 같은 기록을 찾아 묶음. (규칙 · 숫자 자체는 바꾸지 않음 — 결과표도 그대로 다시 써짐)
  · J 짧은 추세: 결과표 기본 숫자는 단독 성과, 여기 저장하는 것은 혼합 25/50/25(단기 50%) 수익률 (단독 줄은 혼합 Sharpe로 짝지음)
  · K 모멘텀 기간: 혼합(모멘텀 축만 바꾼 50/25/25) 수익률
  · E 목표 변동성: 이자 포함 총수익률(결과표 Sharpe는 초과 수익 기준)
  · A 갈아타기: strategy_switch_weekly.parquet(이미 저장)을 이름으로 짝지음
결과: data/strategy_rules_weekly.parquet (열 = '실험 파일::줄 번호') + MinVar 축 주간 수익률(상관 계산용) 열 'MinVar 축'
실행: .venv\\Scripts\\python.exe strategy_rules_weekly.py [모듈 이름 …]   (약 20–40분, 위약 포함)
"""
from __future__ import annotations

import importlib
import io
import sys
import time
import warnings
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import analytics as A  # noqa: E402
import strategy_meta as SM  # noqa: E402

OUT = A.DATA / "strategy_rules_weekly.parquet"


class _Sink(io.StringIO):
    def reconfigure(self, *a, **k):                       # 실험 스크립트가 sys.stdout.reconfigure를 부름
        pass
K3 = ("Sharpe 2008–", "Sharpe 학습 2008–16", "Sharpe 시험 2017–")
# 모듈 → (점수 함수를 가진 모듈, 함수 이름, [(결과표 열들, 점수 dict 열들), …] 순서대로 시도)
SPEC = {
    "strategy_adaptive": ("strategy_meta", "score", [(K3, K3)]),
    "strategy_cash": ("strategy_meta", "score", [(K3, K3)]),
    "strategy_crisis_mom": ("strategy_meta", "score", [(K3, K3)]),
    "strategy_groupcap": ("strategy_meta", "score", [(K3, K3)]),
    "strategy_meta": ("strategy_meta", "score", [(K3, K3)]),
    "strategy_mom_lookback": ("strategy_meta", "score", [(("혼합 2008–", "혼합 학습 2008–16", "혼합 시험 2017–"), K3)]),
    "strategy_short_trend": ("strategy_meta", "score", [(K3, K3), (("혼합 25/50/25 학습", "혼합 25/50/25 시험"), K3[1:])]),
    "strategy_voltgt_dyn": ("strategy_voltgt_dyn", "stats", [(("초과 Sharpe 2008–", "초과 Sharpe 학습 2008–16", "초과 Sharpe 시험 2017–"),) * 2]),
}


def run_one(name: str) -> dict:
    mod_nm, fn_nm, pairs = SPEC[name]
    host = importlib.import_module(mod_nm)
    orig = getattr(host, fn_nm)
    cap = []

    def spy(r, *a, **k):
        o = orig(r, *a, **k)
        cap.append((r, o))
        return o
    setattr(host, fn_nm, spy)
    try:
        m = importlib.import_module(name)
        t = time.time()
        with redirect_stdout(_Sink()):
            m.main()
    finally:
        setattr(host, fn_nm, orig)
    T = pd.read_csv(A.DATA / f"{name}_compare.csv", encoding="utf-8-sig")
    out, miss = {}, 0
    for i, row in T.iterrows():
        hit = None
        for cc, kk in pairs:
            if not all(c in T for c in cc) or row[list(cc)].isna().any():
                continue
            want = row[list(cc)].astype(float).values
            for r, o in cap:
                if all(k in o for k in kk) and np.allclose([o[k] for k in kk], want, atol=1e-9, rtol=0):
                    hit = r
                    break
            if hit is not None:
                break
        if hit is None:
            miss += 1
        else:
            out[f"{name}::{i}"] = hit
    print(f"{name}: {len(T)}줄 중 {len(out)}줄 짝지음 (못 찾음 {miss}) · {time.time() - t:.0f}초", flush=True)
    return out


def switch_rows() -> dict:
    W = pd.read_parquet(A.DATA / "strategy_switch_weekly.parquet")
    T = pd.read_csv(A.DATA / "strategy_switch_compare.csv", encoding="utf-8-sig")
    out = {f"strategy_switch::{i}": W[f"{r['재료']} | {r['규칙']}"] for i, r in T.iterrows() if f"{r['재료']} | {r['규칙']}" in W}
    print(f"strategy_switch: {len(T)}줄 중 {len(out)}줄 짝지음", flush=True)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pick = sys.argv[1:] or list(SPEC)
    old = pd.read_parquet(OUT) if OUT.exists() and sys.argv[1:] else pd.DataFrame()
    cols = {}
    for nm in pick:
        cols.update(run_one(nm))
    cols.update(switch_rows())
    L = A.load_levels(roll_adjusted=True).loc[:"2026-10-26"]
    cols["MinVar 축"] = SM.sleeves(L)[0]["MinVar"]
    D = pd.DataFrame(cols).astype("float32")
    if not old.empty:
        D = old.drop(columns=[c for c in D.columns if c in old], errors="ignore").join(D, how="outer")
    D.to_parquet(OUT)
    print(f"저장 {OUT} · 열 {D.shape[1]}개 · {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
