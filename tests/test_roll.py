"""롤오버 갭 보정 검증: 최신가 = 실제 시세, 롤일 수익률 = 0, 그 외 수익률 불변, 미적용 품목 불변."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import roll_adjust as RA

D = ROOT / "data"
cln = pd.read_parquet(D / "daily_clean.parquet"); adj = pd.read_parquet(D / "daily_adj.parquet")
for x in (cln, adj): x["date"] = pd.to_datetime(x.date)
summ = pd.read_csv(D / "roll_summary.csv"); cal = pd.read_csv(D / "roll_calendar.csv", parse_dates=["date"])
s = lambda df, i: df[df.id == i].set_index("date").value.sort_index()

applied = summ.loc[summ["보정적용"] == "✔", "id"].tolist()
assert "yf:ZC=F" in applied and "yf:ZL=F" not in applied
for i in applied:
    a, c = s(adj, i), s(cln, i)
    assert np.isclose(a.iloc[-1], c.iloc[-1]), f"{i} 최신가 불일치"
    ra, rc = np.log(a).diff(), np.log(c.where(c > 0)).diff()
    rd = pd.DatetimeIndex(cal[(cal.id == i)].date)
    assert np.allclose(ra.reindex(rd).dropna(), 0), f"{i} 롤일 수익률 ≠ 0"
    other = ra.index.difference(rd)
    assert np.allclose(ra.reindex(other).dropna(), rc.reindex(other).reindex(ra.reindex(other).dropna().index), atol=1e-10), f"{i} 평상일 수익률 변경됨"
print(f"OK 보정 적용 {len(applied)}개: 최신가 = 실제 시세, 롤일 수익률 0, 평상일 수익률 불변")
for i in ["yf:CL=F", "yf:ZL=F", "yf:GC=F"]:
    assert np.allclose(s(adj, i), s(cln, i)), f"{i} 미적용 품목이 변경됨"
print("OK 미적용 품목(WTI·대두유·금) 불변")
# 롤 규칙: 옥수수 2021년 7월 롤일 = 7/15(목) (7/14 만기 다음 거래일)
rd = RA.roll_dates(s(cln, "yf:ZC=F").index, [7], "pre15")
assert pd.Timestamp("2021-07-15") in rd, rd[rd.year == 2021]
r = np.log(s(cln, "yf:ZC=F")).diff()
print(f"OK 옥수수 2021-07-15 롤: 원자료 수익률 {100*r['2021-07-15']:+.2f}% (올드크롭 7월물→뉴크롭 9월물 역전 갭) → 보정 후 0")
print("ROLL TEST PASSED")
