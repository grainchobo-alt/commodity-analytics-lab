"""옵션 랩 계산 점검: 블랙숄즈 알려진 값 · 풋-콜 패리티 · 내재변동성 역산 · Black-76 · CSV → 스마일 복원."""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import options_bs as O

# 교과서 값 (Hull): S=42, K=40, r=10%, σ=20%, T=0.5 → 콜 4.76, 풋 0.81
assert abs(O.bs_price(42, 40, 0.5, 0.1, 0.2, True) - 4.76) < 0.01
assert abs(O.bs_price(42, 40, 0.5, 0.1, 0.2, False) - 0.81) < 0.01
# 풋-콜 패리티
c, p = O.bs_price(100, 110, 1, 0.04, 0.3, True), O.bs_price(100, 110, 1, 0.04, 0.3, False)
assert abs(c - p - (100 - 110 * np.exp(-0.04))) < 1e-9
# 내재변동성 역산 (BS · B76)
for sig in (0.1, 0.35, 0.8):
    assert abs(O.implied_vol(O.bs_price(100, 95, 0.3, 0.04, sig, False), 100, 95, 0.3, 0.04, False) - sig) < 1e-6
    assert abs(O.implied_vol(O.b76_price(450, 480, 0.25, 0.04, sig, True), 450, 480, 0.25, 0.04, True, "B76") - sig) < 1e-6
assert np.isnan(O.implied_vol(0.0, 100, 90, 0.5, 0.04, True))               # 내재가치 이하 → NaN
# 델타 = 수치 미분
g = O.greeks(100, 100, 0.5, 0.04, 0.25, True)
num = (O.bs_price(100.01, 100, 0.5, 0.04, 0.25) - O.bs_price(99.99, 100, 0.5, 0.04, 0.25)) / 0.02
assert abs(g["델타"] - num) < 1e-4
# CSV → 스마일 복원 (가상 선물옵션: 등가격 24%, 기울기 +0.05, 휘어짐 0.6)
asof, F, rows = pd.Timestamp("2026-10-30"), 450.0, []
for dte in (30, 90):
    for K in range(360, 561, 10):
        m = np.log(K / F); s = 0.24 + 0.05 * m + 0.6 * m * m
        for t in "CP":
            rows.append({"asof": asof, "underlying": F, "expiry": asof + pd.Timedelta(days=dte), "strike": K, "type": t,
                         "last": O.b76_price(F, K, dte / 365, 0.04, s, t == "C")})
buf = io.StringIO(); pd.DataFrame(rows).to_csv(buf, index=False); buf.seek(0)
ch = O.chain_csv(buf)
fits = O.smile_fit(O.iv_points(ch, 0.04, "B76", max_spread=1, min_oi=0))
assert len(fits) == 2
for a, b, c2, n in fits.values():
    assert abs(a - 0.24) < 1e-3 and abs(b - 0.05) < 1e-2 and abs(c2 - 0.6) < 2e-2, (a, b, c2)
print("test_options_bs OK")
