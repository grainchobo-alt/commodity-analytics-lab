"""저장된 이력 -> 엑셀 (data/excel/)

  원자재_일별_원본.xlsx   : 수집 원본 그대로 (날짜 × 품목, 휴장일은 빈칸)
  원자재_일별_정제.xlsx   : clean.py 정제본 + 정제 내역
  원자재_월별.xlsx        : 월말종가 · 월평균 · 월고가 · 월저가 · 월간수익률 · 관측일수 + World Bank / IMF 원천 월간
  원자재_연도별.xlsx      : 연말종가 · 연평균 · 연고가 · 연저가 · 연간수익률 · 연간변동성 · 관측일수 + World Bank 연평균
  COT_주별.xlsx           : CFTC 포지션 (Managed Money 순포지션 · 상업 순포지션 · 미결제약정)
월별·연도별 집계는 정제본 기준. 진행 중인 달·연도는 '해당 기간 마지막 관측일까지'의 값임.
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd

import analytics as A

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = DATA / "excel"
ORDER = ["Grains", "Energy", "Metals", "Iron Ore / Coal", "Freight", "China (DCE)", "Softs", "Livestock",
         "FRED daily", "Macro", "FX"]


def meta_table(d: pd.DataFrame) -> pd.DataFrame:
    m = d.groupby("id").agg(품목=("name", "first"), 그룹=("group", "first"), 단위=("unit", "first"),
                            출처=("source", "first"), 시작=("date", "min"), 끝=("date", "max"), 관측수=("value", "size"))
    m["정렬"] = m["그룹"].map(lambda g: ORDER.index(g) if g in ORDER else 99)
    m = m.sort_values(["정렬", "품목"]).drop(columns="정렬")
    m["열이름"] = m["품목"] + " (" + m["단위"].fillna("") + ")"
    m["수익률 계산"] = np.where(m["단위"] == "%", "차분(%p)", "로그/단순 수익률(%)")
    return m.reset_index()


def wide(d: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    w = d.pivot_table(index="date", columns="id", values="value", aggfunc="last").sort_index()
    w = w[[i for i in meta["id"] if i in w.columns]]
    w.columns = meta.set_index("id").loc[w.columns, "열이름"].values
    w.index = pd.to_datetime(w.index)
    w.index.name = "날짜"
    return w


def write(path: Path, sheets: dict, notes: list[str]):
    with pd.ExcelWriter(path, engine="xlsxwriter", datetime_format="yyyy-mm-dd", date_format="yyyy-mm-dd") as xw:
        wb = xw.book
        num = wb.add_format({"num_format": "#,##0.0000"})
        pd.DataFrame({"설명": notes}).to_excel(xw, sheet_name="설명", index=False)
        xw.sheets["설명"].set_column(0, 0, 120)
        for name, df in sheets.items():
            idx = not isinstance(df.index, pd.RangeIndex)
            df.to_excel(xw, sheet_name=name[:31], index=idx)
            ws = xw.sheets[name[:31]]
            ws.freeze_panes(1, 1 if idx else 0)
            if idx and df.shape[1] > 0 and all(dt.kind == "f" for dt in df.dtypes):
                ws.set_column(0, 0, 12)
                ws.set_column(1, df.shape[1], 14, num)
            else:
                ws.autofit()
    print(f"  {path.name:28s} {path.stat().st_size / 1e6:6.1f} MB  시트: {', '.join(sheets)}")


def period_tables(w: pd.DataFrame, rule: str, meta: pd.DataFrame, ann: int, w_adj: pd.DataFrame | None = None) -> dict:
    """w: 날짜 × 품목 (정제 레벨, 실제 시세). w_adj: 롤 보정 레벨 — 수익률·변동성은 이것으로 계산. rule: 'ME' 또는 'YE'."""
    g = w.resample(rule)
    last, mean, hi, lo, cnt = g.last(), g.mean(), g.max(), g.min(), g.count()
    last_r = (w_adj if w_adj is not None else w).resample(rule).last().reindex(columns=last.columns)
    is_rate = meta.set_index("열이름")["단위"].eq("%")
    ret = pd.DataFrame(index=last.index)
    for c in last.columns:
        if is_rate.get(c, False):
            ret[c] = last_r[c].diff()                                 # %p 변화
        else:
            p = last_r[c].where(last_r[c] > 0)
            ret[c] = (p / p.shift(1) - 1) * 100                       # 단순수익률 % (롤 보정)
    keep = cnt.sum(axis=1) > 0
    out = {"종가(기말)": last, "평균": mean, "고가": hi, "저가": lo, "수익률(%)·금리는 %p": ret, "관측일수": cnt.astype(float)}
    if rule == "YE":
        vol = pd.DataFrame(index=last.index)
        units = meta.set_index("열이름")["단위"]
        wv = w_adj if w_adj is not None else w
        for c in w.columns:
            r = A.log_return(wv[c].dropna(), units.get(c, ""))
            vol[c] = r.groupby(r.index.year).std().reindex(last.index.year).values * np.sqrt(ann)
        out["변동성(연율, %)"] = vol
    lbl = "%Y-%m" if rule == "ME" else "%Y"
    for k in out:
        out[k] = out[k][keep].copy()
        out[k].index = out[k].index.strftime(lbl)
        out[k].index.name = "월" if rule == "ME" else "연도"
    return out


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(DATA / "daily.parquet"); raw["date"] = pd.to_datetime(raw["date"])
    cln_path = DATA / "daily_clean.parquet"
    cln = pd.read_parquet(cln_path) if cln_path.exists() else raw
    cln["date"] = pd.to_datetime(cln["date"])
    meta = meta_table(raw)
    asof = f"{pd.Timestamp.now():%Y-%m-%d %H:%M} 생성"

    # 1) 일별 원본
    write(OUT / "원자재_일별_원본.xlsx", {"일별_원본": wide(raw, meta), "메타": meta},
          [asof, "수집 원본 그대로입니다. 휴장일·미발표일은 빈칸이며 값을 채우지 않았습니다.",
           "열 이름 = 품목 (단위). 출처·시작일·관측수는 '메타' 시트 참조.",
           "Yahoo '=F' 시리즈는 연속월물(근월물 연결) — 롤오버 시점에 가격 갭이 있을 수 있습니다."])
    # 2) 일별 정제
    log = pd.read_csv(DATA / "cleaning_log.csv") if (DATA / "cleaning_log.csv").exists() else pd.DataFrame()
    write(OUT / "원자재_일별_정제.xlsx", {"일별_정제": wide(cln, meta), "정제내역": log, "메타": meta},
          [asof, "원본에서 ① 주말 날짜 관측치 ② 중복 ③ data/cleaning_rules.csv 의 오류 의심 값을 제거한 버전입니다.",
           "제거한 값의 목록과 사유는 '정제내역' 시트에 있습니다. 결측을 보간하지 않았습니다(빈칸 유지).",
           "WTI 2020-04-20 마이너스 유가, BCI 2020년 음수 등 실제 이벤트는 유지했습니다."])
    wc = wide(cln, meta)
    # 2-b) 롤 보정
    adj_path = DATA / "daily_adj.parquet"
    wa = None
    if adj_path.exists():
        adj = pd.read_parquet(adj_path); adj["date"] = pd.to_datetime(adj["date"])
        wa = wide(adj, meta)
        rs = pd.read_csv(DATA / "roll_summary.csv") if (DATA / "roll_summary.csv").exists() else pd.DataFrame()
        rc = pd.read_csv(DATA / "roll_calendar.csv") if (DATA / "roll_calendar.csv").exists() else pd.DataFrame()
        adj_ids = rs.loc[rs["보정적용"] == "✔", "id"].tolist() if len(rs) else []
        cols = [meta.set_index("id").loc[i, "열이름"] for i in adj_ids if i in set(meta.id)]
        write(OUT / "원자재_일별_롤보정.xlsx", {"일별_롤보정": wa[cols], "롤요약": rs, "롤일정": rc},
              [asof, "Yahoo 연속월물(=F)은 만기 교체일에 두 계약의 가격차만큼 가짜 수익률이 생깁니다(롤오버 갭).",
               "거래소 만기 규칙으로 롤일을 추정하고, 롤일 |수익률|이 평상시의 1.5배 이상으로 검증된 품목만 롤 당일 수익률을 0으로 두고 가격을 재누적했습니다.",
               "최신 가격은 실제 시세와 같고 과거 가격이 조정됩니다(비율 백어저스트 근사). 가격 '수준' 비교에는 정제본, 수익률·변동성·계절성 분석에는 이 파일을 쓰세요.",
               "'롤요약' 시트: 품목별 롤일 평균수익률·연간 누적 롤갭(= 보정하지 않으면 수익률에 섞이는 가짜 수익)."])
    # 3) 월별
    mt = period_tables(wc, "ME", meta, 12, wa)
    mon = pd.read_parquet(DATA / "monthly.parquet"); mon["date"] = pd.to_datetime(mon["date"])
    for src, nm in [("World Bank", "WorldBank_월평균(원천)"), ("FRED/IMF", "IMF_월평균(원천)")]:
        sub = mon[mon.source == src]
        if len(sub):
            ww = sub.pivot_table(index="date", columns="name", values="value", aggfunc="last").sort_index()
            units = sub.groupby("name").unit.first()
            ww.columns = [f"{c} {units[c]}".strip() for c in ww.columns]
            ww.index = ww.index.strftime("%Y-%m"); ww.index.name = "월"
            mt[nm] = ww
    write(OUT / "원자재_월별.xlsx", mt,
          [asof, "일별 정제본을 월 단위로 집계: 종가(월말 마지막 관측치), 평균, 고가, 저가, 수익률(전월말 대비), 관측일수.",
           "수익률: 가격은 단순수익률(%), 금리·스프레드(단위 %)는 %p 변화. 음수 가격이 포함된 달은 수익률 빈칸.", "종가·평균·고가·저가는 실제 시세, 수익률은 연속월물 롤오버 갭을 보정한 가격 기준입니다(원자재_일별_롤보정.xlsx 참고).",
           "WorldBank_월평균 / IMF_월평균 시트는 원천 기관이 발표한 월평균 가격 그대로입니다(석탄·철광석 등 일별 무료 소스가 없는 품목 포함).",
           "진행 중인 이번 달 값은 마지막 관측일까지의 값입니다."])
    # 4) 연도별
    yt = period_tables(wc, "YE", meta, 252, wa)
    wbm = mon[mon.source == "World Bank"]
    if len(wbm):
        wy = wbm.assign(y=wbm.date.dt.year).pivot_table(index="y", columns="name", values="value", aggfunc="mean")
        units = wbm.groupby("name").unit.first()
        wy.columns = [f"{c} {units[c]}".strip() for c in wy.columns]
        wy.index = wy.index.astype(str); wy.index.name = "연도"
        yt["WorldBank_연평균"] = wy
    write(OUT / "원자재_연도별.xlsx", yt,
          [asof, "일별 정제본을 연 단위로 집계: 종가(연말), 평균, 고가, 저가, 수익률(전년말 대비), 변동성(일간 로그수익률 표준편차×√252), 관측일수.", "가격 수준은 실제 시세, 수익률·변동성은 롤오버 갭 보정 가격 기준입니다.",
           f"올해({pd.Timestamp.now().year})는 마지막 관측일까지의 연초 대비(YTD) 값입니다.",
           "WorldBank_연평균은 World Bank 월평균의 연 평균입니다."])
    # 5) COT
    if (DATA / "cot.parquet").exists():
        cot = pd.read_parquet(DATA / "cot.parquet"); cot["date"] = pd.to_datetime(cot["date"])
        sh = {}
        for col, nm in [("mm_net", "MM_순포지션"), ("comm_net", "상업_순포지션"), ("open_interest_all", "미결제약정")]:
            ww = cot.pivot_table(index="date", columns="name", values=col, aggfunc="last").sort_index().astype(float)
            ww.index.name = "보고일(화)"; sh[nm] = ww
        write(OUT / "COT_주별.xlsx", sh,
              [asof, "CFTC Disaggregated Futures-Only 보고서 (매주 화요일 기준, 금요일 발표). 단위: 계약 수.",
               "MM_순포지션 = Managed Money 매수 - 매도 / 상업_순포지션 = Producer/Merchant 매수 - 매도."])
    print(f"엑셀 저장 완료 ({time.time() - t0:.0f}초) -> {OUT}")


if __name__ == "__main__":
    main()
