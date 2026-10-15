"""기관 리포트 파서 검증: 캐시된 원문에서 알려진 값이 정확히 나오는가, 수정폭·시점 처리가 맞는가."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import reports as R

txt = (R.RAW / "wasde_202610.txt").read_text(encoding="latin-1")
t = R.parse_wasde_table(txt, *R.WASDE_TABLES["corn"][:2])
assert t.attrs["years"][-1] == "2026/27"
assert (t.loc["end_stocks", "proj_prev"], t.loc["end_stocks", "proj"]) == (1567.0, 1849.0), t
assert (t.loc["yield", "proj_prev"], t.loc["yield", "proj"]) == (178.5, 181.2)
s = R.parse_wasde_table(txt, *R.WASDE_TABLES["soymeal"][:2])
assert s.loc["price", "proj"] == 340.0 and s.loc["end_stocks", "proj"] == 450.0       # 섹션 구분 (대두박 ≠ 대두)
print("OK  WASDE 2026-10 옥수수 기말재고 1567→1849, 단수 178.5→181.2, 대두박 섹션 분리")

F = pd.read_parquet(R.RDIR / "wasde_features.parquet")
c = F[(F.commodity == "corn")].set_index("release")
oct26 = c.loc[pd.Timestamp("2026-10-09 12:00", tz="UTC")]
assert abs(oct26.d_end_stocks - (1849 / 1567 - 1) * 100) < 1e-9          # 직전 발간본(9월)의 같은 연산연도 대비
may26 = c.loc[pd.Timestamp("2026-05-12 12:00", tz="UTC")]
assert may26["first"] == 1 and np.isnan(may26.d_end_stocks)                # 신규 작황 첫 전망은 수정폭 없음
assert not (F.release.dt.strftime("%Y-%m") == "2025-10").any()            # 2025-10 셧다운 미발간
print("OK  수정폭 = 직전 발간본 대비, 5월 첫 전망 NaN, 2025-10 미발간 반영")

G = pd.read_parquet(R.RDIR / "steo_features.parquet")
assert (G.avail > G.vintage).all() and (G.avail - G.vintage).dt.days.min() >= 5
print("OK  STEO 이용가능일 = forecast date + 5영업일 (공개 전 사용 금지)")
print("REPORTS TEST PASSED")
