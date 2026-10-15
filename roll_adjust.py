"""연속월물(=F) 롤오버 갭 보정 -> data/daily_adj.parquet, data/roll_calendar.csv

Yahoo '=F' 연속월물은 근월물 만기 시 다음 월물로 '그냥 갈아끼운' 가격이라 롤 당일에 두 계약의 가격차(스프레드)만큼
가짜 수익률이 생긴다. 곡물은 롤 시점이 매년 같은 달이라 가짜 계절성까지 만든다
(예: 옥수수 7월 롤 = 올드크롭 7월물 → 뉴크롭 9월물, 역전 시 큰 음(-)의 점프).

방법: 거래소 만기 규칙으로 롤 추정일을 만든 뒤 → 그날 |수익률|이 평상시보다 뚜렷이 큰지(비율 ≥ RATIO_MIN) 검증 →
검증된 품목만 롤 당일 로그수익률을 0으로 두고 가격을 다시 누적(최신 가격 = 실제 시세로 고정, 과거만 조정 = 비율 백어저스트 근사).
한계: 롤 당일의 '진짜' 가격 변화도 함께 0이 됨(평균 0의 잡음, 편향 없음). 정확한 보정은 개별 만기 계약 이력이 필요.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
RATIO_MIN = 1.5

# (계약월, 롤 규칙)  rule: "pre15" = 계약월 15일 직전 영업일 만기 → 다음 거래일 롤
#                         "nth:k" = 계약월 k번째 거래일 (만기 다음 날)
#                         "first" = 계약월 첫 거래일 (전월 말 만기)
#                         "first_next" = 계약월 다음 달 첫 거래일 (계약월 말 만기)
RULES = {
    "yf:ZC=F": ([3, 5, 7, 9, 12], "pre15"), "yf:ZW=F": ([3, 5, 7, 9, 12], "pre15"),
    "yf:KE=F": ([3, 5, 7, 9, 12], "pre15"), "yf:ZO=F": ([3, 5, 7, 9, 12], "pre15"),
    "yf:ZS=F": ([1, 3, 5, 7, 8, 9, 11], "pre15"), "yf:ZM=F": ([1, 3, 5, 7, 8, 9, 10, 12], "pre15"),
    "yf:ZL=F": ([1, 3, 5, 7, 8, 9, 10, 12], "pre15"), "yf:ZR=F": ([1, 3, 5, 7, 9, 11], "pre15"),
    "yf:HE=F": ([2, 4, 5, 6, 7, 8, 10, 12], "nth:11"),
    "yf:LE=F": ([2, 4, 6, 8, 10, 12], "first_next"),
    "yf:TIO=F": (list(range(1, 13)), "first"),
    "yf:RB=F": (list(range(1, 13)), "first"), "yf:HO=F": (list(range(1, 13)), "first"),
    "yf:SB=F": ([3, 5, 7, 10], "first"),
}


def roll_dates(idx: pd.DatetimeIndex, months: list[int], rule: str) -> pd.DatetimeIndex:
    out = []
    for y in range(idx.min().year, idx.max().year + 1):
        for m in months:
            mm = idx[(idx.year == y) & (idx.month == m)]
            if rule == "pre15":
                exp = pd.Timestamp(y, m, 15) - pd.offsets.BDay(1)
                nxt = idx[idx > exp]
                if len(nxt) and nxt[0].month == m:
                    out.append(nxt[0])
            elif rule.startswith("nth:"):
                k = int(rule.split(":")[1])
                if len(mm) >= k:
                    out.append(mm[k - 1])
            elif rule == "first":
                if len(mm):
                    out.append(mm[0])
            elif rule == "first_next":
                ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
                nn = idx[(idx.year == ny) & (idx.month == nm)]
                if len(nn):
                    out.append(nn[0])
    return pd.DatetimeIndex(out)


def main():
    src = DATA / "daily_clean.parquet"
    d = pd.read_parquet(src if src.exists() else DATA / "daily.parquet")
    d["date"] = pd.to_datetime(d["date"])
    cal, summ, adj_parts = [], [], []
    for sid, (months, rule) in RULES.items():
        g = d[d.id == sid].sort_values("date")
        if g.empty:
            continue
        s = g.set_index("date")["value"]
        pos = s.where(s > 0)
        r = np.log(pos).diff()
        rd = roll_dates(r.dropna().index, months, rule)
        rr = r.reindex(rd).dropna()
        normal = r.drop(rd, errors="ignore").dropna()
        ratio = rr.abs().mean() / normal.abs().mean()
        t = (rr.mean() - normal.mean()) / (rr.std() / np.sqrt(len(rr)))
        apply = ratio >= RATIO_MIN
        summ.append(dict(id=sid, name=g["name"].iloc[0], 규칙=rule, 계약월=",".join(map(str, months)), 롤횟수=len(rr),
                         롤일_평균수익률=round(100 * rr.mean(), 3), 롤일_평균절대=round(100 * rr.abs().mean(), 3),
                         평상일_평균절대=round(100 * normal.abs().mean(), 3), 비율=round(ratio, 2), t값=round(t, 2),
                         연간_롤갭_누적=round(100 * rr.mean() * len(months), 2), 보정적용="✔" if apply else "✖ (증거 부족)"))
        for dt, v in rr.items():
            cal.append(dict(id=sid, name=g["name"].iloc[0], date=dt.date(), roll_return_pct=round(100 * v, 3), applied=apply))
        if apply:
            r2 = r.copy()
            r2.loc[r2.index.isin(rd)] = 0.0
            # 최신 가격을 실제 시세로 고정하고 과거를 조정 (비율 백어저스트)
            cum = r2.fillna(0).cumsum()
            adj = pos.iloc[-1] * np.exp(cum - cum.iloc[-1])
            adj = adj.where(pos.notna())               # 원래 결측·음수 시점은 결측 유지
            adj_parts.append(g.assign(value=adj.reindex(g.date).values))
    S = pd.DataFrame(summ)
    keep = d[~d.id.isin([p.id.iloc[0] for p in adj_parts])]
    out = pd.concat([keep] + adj_parts, ignore_index=True)
    out.to_parquet(DATA / "daily_adj.parquet")
    pd.DataFrame(cal).to_csv(DATA / "roll_calendar.csv", index=False, encoding="utf-8-sig")
    S.to_csv(DATA / "roll_summary.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(S.drop(columns=["id", "계약월"]).to_string(index=False))
    print(f"\n롤 보정 완료: {sum(S['보정적용'] == '✔')}/{len(S)}개 품목 적용 -> daily_adj.parquet")


if __name__ == "__main__":
    main()
