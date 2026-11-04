"""전략별 품목 비중 저장 — ⑤ Portfolio Customization에서 '사용자가 고른 전략 조합'에 품목군 상한을 바로 걸기 위한 재료.

롱숏 탭 재료(ls_strategies.run_all · run_alloc, ls_extra.run_extra)를 같은 고정 자료(FROZEN_END)로 다시 돌리면서
LS._book(W, R, cost) 호출을 엿들어 비중 W를 기록하고, 결과 수익률 표의 열과 값이 같은 기록을 찾아 전략 이름 → 품목 비중으로 묶음.
(혼합 · 국면 전환 오버레이처럼 수익률끼리 섞어 만든 전략은 _book을 직접 거치지 않아 비중이 없음 → 화면에서 '품목 비중 없음'으로 안내)
대안 시계열 · 멀티모달 재료는 실험 결과 수익률만 저장돼 있어 이 파일에 없음.

실행: .venv\\Scripts\\python.exe strategy_itemw.py → data/frozen/itemw_{FROZEN_END}.pkl  (약 3–6분, HMM 포함)
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import analytics as A  # noqa: E402
import ls_extra as LX  # noqa: E402
import ls_strategies as LS  # noqa: E402

FROZEN_END = "2026-10-26"                    # app.py FROZEN_END와 같아야 함
OUT = A.DATA / "frozen" / f"itemw_{FROZEN_END}.pkl"


def _match(D: pd.DataFrame, rec: list, prefix: str = "") -> dict:
    out = {}
    for c in D.columns:
        y = D[c].fillna(0.0).values
        for r, W in rec:
            x = r.reindex(D.index).fillna(0.0).values
            if np.allclose(x, y, atol=1e-9, rtol=0) and np.abs(y).sum() > 0:
                out[prefix + c] = W.reindex(r.index.union(W.index)).fillna(0.0).astype("float32")
                break
    return out


def build(roll: bool = True, cost_bp: float = 5.0) -> dict:
    L = A.load_levels(roll_adjusted=roll).loc[:FROZEN_END]
    rec: list = []
    orig = LS._book

    def spy(W, R, cost, *a, **k):
        r = orig(W, R, cost, *a, **k)
        rec.append((r, W.reindex(R.index).fillna(0.0)))
        return r
    LS._book = spy
    try:
        out = {}
        t = time.time(); rec.clear()
        out.update(_match(LX.run_extra(L, cost_bp)["returns"], list(rec))); print(f"run_extra {time.time() - t:.0f}s", flush=True)
        t = time.time(); rec.clear()
        out.update(_match(LS.run_all(L, cost_bp, with_hmm=True)["returns"], list(rec))); print(f"run_all {time.time() - t:.0f}s", flush=True)
        t = time.time(); rec.clear()
        out.update(_match(LS.run_alloc(L, cost_bp)["returns"], list(rec), prefix="비중 방식 · ")); print(f"run_alloc {time.time() - t:.0f}s", flush=True)
    finally:
        LS._book = orig
    return out


def combine_items(Wi: dict, Ws: pd.DataFrame) -> pd.DataFrame:
    """전략 비중 Ws(주 × 전략, t주 수익에 적용) × 전략별 품목 비중(t−1 결정) → 품목 순비중(t−1 결정, LS._book 규칙)."""
    cols = sorted(set().union(*[Wi[s].columns for s in Ws.columns]))
    idx = Ws.index
    tot = pd.DataFrame(0.0, index=idx, columns=cols)
    for s in Ws.columns:
        w_next = Ws[s].shift(-1).reindex(idx).ffill().fillna(0.0)          # t−1에 정한 품목 비중은 t주 전략 비중으로 섞임
        tot = tot.add(Wi[s].reindex(idx).fillna(0.0).reindex(columns=cols, fill_value=0.0).mul(w_next, axis=0), fill_value=0.0)
    return tot


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = build(True)
    pd.to_pickle(out, OUT)
    print(f"전략 {len(out)}개 품목 비중 저장 → {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    for k in out:
        print(" ·", k, out[k].shape)


if __name__ == "__main__":
    main()
