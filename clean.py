"""원본(daily.parquet) -> 정제본(daily_clean.parquet)

원본은 절대 수정하지 않고, 아래 규칙을 적용한 사본을 만든다. 적용 내역은 data/cleaning_log.csv.
  1. 주말 날짜 관측치 제거 (Baltic·DCE 일부 레코드가 토/일로 찍혀 있음 — 거래일 아님)
  2. 같은 (id, 날짜) 중복 제거
  3. data/cleaning_rules.csv 의 수동 검토 목록 적용 (action=nan -> 해당 값 제거)

실제 이벤트로 판단해 '유지'한 대표 사례 (규칙에 넣지 않음):
  - WTI 2020-04-20 -37.63$ (선물 만기 마이너스 유가), EIA WTI/Brent·RBOB 2020-04-20~21
  - BCI 2020-01-31~03 음수 (케이프사이즈 지수 실제 마이너스 진입)
  - Henry Hub 현물 한파 급등 (2021-02 Uri, 2024-01, 2025-01), NG 선물 2022-01-27 만기 스퀴즈
  - VIX 2011-08, 2020-03 급등
음수·0 가격의 수익률 처리는 analytics.make_panel 에서 해당 시점만 결측 처리한다(시리즈 전체를 차분으로 바꾸지 않음).
"""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).parent / "data"


def main():
    raw = pd.read_parquet(DATA / "daily.parquet")
    raw["date"] = pd.to_datetime(raw["date"])
    log = []
    wk = raw.date.dt.dayofweek >= 5
    for _, r in raw[wk].iterrows():
        log.append(dict(id=r.id, name=r["name"], date=r.date.date(), value=r.value, action="삭제", reason="주말 날짜 관측치"))
    d = raw[~wk]
    dup = d.duplicated(["id", "date"], keep="last")
    if dup.any():
        log += [dict(id=r.id, name=r["name"], date=r.date.date(), value=r.value, action="삭제", reason="중복 날짜") for _, r in d[dup].iterrows()]
    d = d[~dup].copy()
    rules_path = DATA / "cleaning_rules.csv"
    if rules_path.exists():
        rules = pd.read_csv(rules_path, parse_dates=["date"], comment="#")
        for _, r in rules.iterrows():
            m = (d.id == r.id) & (d.date == r.date)
            if not m.any():
                log.append(dict(id=r.id, name="", date=r.date.date(), value=None, action="규칙 미적용", reason="해당 관측치 없음: " + str(r.reason)))
                continue
            if str(r.action).strip().lower() == "nan":
                for _, x in d[m].iterrows():
                    log.append(dict(id=x.id, name=x["name"], date=x.date.date(), value=x.value, action="결측 처리", reason=r.reason))
                d = d[~m]
    d.to_parquet(DATA / "daily_clean.parquet")
    lg = pd.DataFrame(log)
    lg.to_csv(DATA / "cleaning_log.csv", index=False, encoding="utf-8-sig")
    print(f"정제 완료: 원본 {len(raw):,}행 -> 정제 {len(d):,}행 (제거 {len(raw) - len(d)}건)")
    if len(lg):
        print(lg.groupby(["action", "reason"]).size().rename("건수").reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
