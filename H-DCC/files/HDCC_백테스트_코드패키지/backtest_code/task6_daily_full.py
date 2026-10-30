"""Daily H-DCC full-period check: eval from earliest feasible (1996) through 2025,
walk-forward expanding lambda (causal). Sub-period breakdown incl. 1996-1998."""
import pandas as pd, numpy as np
from daily_engine import build_paths, daily_backtest, sharpe_d

dates, Yv, SIG, Z = build_paths('hdcc', lam_mode='exp')
rp = daily_backtest(dates, Yv, SIG, Z, '1996-01-01')
print(f"일간 H-DCC 워크포워드, 평가 1996-01~2025-12: n={len(rp)}")
print(f"  전체 Sharpe(일간 컨벤션) = {sharpe_d(rp):.3f}")
cum = (1+rp).cumprod(); peak = cum.cummax()
print(f"  MaxDD = {((cum-peak)/peak).min()*100:.1f}%,  AnnRet = {rp.mean()*252*100:.1f}%,  AnnVol = {rp.std()*np.sqrt(252)*100:.1f}%")

for lbl, a, b in [('1996~1998 (초기/IMF 포함)','1996-01-01','1998-12-31'),
                  ('1999~2009','1999-01-01','2009-12-31'),
                  ('2010~2025 (OOS)','2010-01-01','2025-12-31'),
                  ('1999~2025','1999-01-01','2025-12-31')]:
    seg = rp[(rp.index>=a)&(rp.index<=b)]
    print(f"  {lbl:26s}: SR={sharpe_d(seg):.3f} (n={len(seg)})")

# Monthly-aggregated Sharpe for convention comparison
mr = rp.resample('ME').sum()
print(f"\n  [참고] 동일 수익을 월간 합산 후 Sharpe(월간 컨벤션): {mr.mean()*12/(mr.std(ddof=1)*np.sqrt(12)):.3f}")
