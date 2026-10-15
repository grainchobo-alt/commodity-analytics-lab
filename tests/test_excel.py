"""엑셀 출력 정합성 검증: 원본 parquet 값과 엑셀 값 대조."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd

X = ROOT / "data" / "excel"
raw = pd.read_parquet(ROOT / "data" / "daily.parquet"); raw["date"] = pd.to_datetime(raw.date)
cln = pd.read_parquet(ROOT / "data" / "daily_clean.parquet"); cln["date"] = pd.to_datetime(cln.date)
def s(df, i): return df[df.id == i].set_index("date").value.sort_index()

d_raw = pd.read_excel(X / "원자재_일별_원본.xlsx", sheet_name="일별_원본", index_col=0)
meta = pd.read_excel(X / "원자재_일별_원본.xlsx", sheet_name="메타")
col = dict(zip(meta.id, meta["열이름"]))
# 1) 원본: 모든 시리즈의 관측수·마지막 값 일치
bad = []
for i in meta.id:
    a, b = s(raw, i), d_raw[col[i]].dropna()
    if len(a) != len(b) or not np.isclose(a.iloc[-1], b.iloc[-1]) or not np.isclose(a.sum(), b.sum(), rtol=1e-9):
        bad.append(i)
assert not bad, f"원본 불일치: {bad}"
print(f"OK 일별 원본: {len(meta)}개 시리즈 관측수·합계·최종값 일치, 기간 {d_raw.index.min():%Y-%m-%d}~{d_raw.index.max():%Y-%m-%d}")
# 2) 정제: 규칙 적용 값이 빠졌는지
d_cln = pd.read_excel(X / "원자재_일별_정제.xlsx", sheet_name="일별_정제", index_col=0)
assert pd.isna(d_cln.loc["2015-10-08", col["ak:I0"]]) and not pd.isna(d_raw.loc["2015-10-08", col["ak:I0"]])
assert np.isclose(d_cln.loc["2020-04-20", col["yf:CL=F"]], -37.63, atol=0.01), "WTI 마이너스 유가는 유지돼야 함"
print("OK 일별 정제: 오류값 제거(DCE 철광석 2015-10-08), 실제 이벤트 유지(WTI 2020-04-20 -37.63)")
# 3) 월별: 월말 종가 = 그 달 마지막 정제 관측치, 수익률 = 월말/전월말-1
m_last = pd.read_excel(X / "원자재_월별.xlsx", sheet_name="종가(기말)", index_col=0)
m_ret = pd.read_excel(X / "원자재_월별.xlsx", sheet_name="수익률(%)·금리는 %p", index_col=0)
m_avg = pd.read_excel(X / "원자재_월별.xlsx", sheet_name="평균", index_col=0)
c = s(cln, "yf:ZC=F")
adj = pd.read_parquet(ROOT / "data" / "daily_adj.parquet"); adj["date"] = pd.to_datetime(adj.date); ca = s(adj, "yf:ZC=F")
assert np.isclose(m_last.loc["2026-09", col["yf:ZC=F"]], c["2026-09"].iloc[-1]), "종가는 실제 시세"
assert np.isclose(m_avg.loc["2026-09", col["yf:ZC=F"]], c["2026-09"].mean())
assert np.isclose(m_ret.loc["2026-09", col["yf:ZC=F"]], (ca["2026-09"].iloc[-1] / ca["2026-08"].iloc[-1] - 1) * 100), "수익률은 롤 보정 가격 기준"
t = s(cln, "fred:DGS10")
assert np.isclose(m_ret.loc["2026-09", col["fred:DGS10"]], t["2026-09"].iloc[-1] - t["2026-08"].iloc[-1]), "금리는 %p 차분"
print("OK 월별: 옥수수 2026-09 월말종가·월평균·수익률, 10년물 %p 변화 일치")
# 4) 연도별
y_last = pd.read_excel(X / "원자재_연도별.xlsx", sheet_name="종가(기말)", index_col=0)
y_ret = pd.read_excel(X / "원자재_연도별.xlsx", sheet_name="수익률(%)·금리는 %p", index_col=0)
y_vol = pd.read_excel(X / "원자재_연도별.xlsx", sheet_name="변동성(연율, %)", index_col=0)
w = s(adj, "yf:CL=F")   # WTI는 롤 보정 미적용 품목 → 보정본 = 정제본
assert np.isclose(y_last.loc[2008, col["yf:CL=F"]], w["2008"].iloc[-1])
assert np.isclose(y_ret.loc[2008, col["yf:CL=F"]], (w["2008"].iloc[-1] / w["2007"].iloc[-1] - 1) * 100)
lr = 100 * np.log(w[w.index.year <= 2019]).diff()   # 2019년 첫 수익률은 2018년 말 대비
v2019 = lr[lr.index.year == 2019].std() * np.sqrt(252)
assert np.isclose(y_vol.loc[2019, col["yf:CL=F"]], v2019, rtol=1e-6)
print(f"OK 연도별: WTI 2008 연말종가 {y_last.loc[2008, col['yf:CL=F']]:.2f}, 수익률 {y_ret.loc[2008, col['yf:CL=F']]:.1f}%, 2019 변동성 {v2019:.1f}%")
print(f"   WTI 2020 연간 변동성 {y_vol.loc[2020, col['yf:CL=F']]:.1f}% (음수가격 당일 수익률 제외)")
wb = pd.read_excel(X / "원자재_월별.xlsx", sheet_name="WorldBank_월평균(원천)", index_col=0)
print(f"OK World Bank 월평균: {wb.shape[1]}개 품목, {wb.index.min()}~{wb.index.max()}")
cot = pd.read_excel(X / "COT_주별.xlsx", sheet_name="MM_순포지션", index_col=0)
print(f"OK COT: {cot.shape[1]}개 계약, {cot.index.min():%Y-%m-%d}~{cot.index.max():%Y-%m-%d}")
print("EXCEL TEST PASSED")
