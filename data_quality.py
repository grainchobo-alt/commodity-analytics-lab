"""데이터 품질 점검: 결측(공백 구간) · 이상치 · 정체값 · 중복 · 소스 간 교차검증 -> data/excel/data_quality_report.xlsx

이상치 판정 (일간 로그수익률 기준)
  - 강건 z = r / (1.4826 × 롤링 MAD, 중심 125일창) 의 절댓값 > Z_TH  또는  |r| > ABS_TH%
  - 분류
      데이터 오류 의심 : 급등락 직후 반대 방향으로 되돌아와 전전일 수준에 복귀 (spike-reversal)
      위기 구간 이벤트 : analytics.CRISES 구간 내
      롤오버 의심      : Yahoo 연속월물(=F) — 만기 교체 시 가격 갭
      검토 필요        : 그 외
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import analytics as A

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = DATA / "excel"
Z_TH, ABS_TH = 10.0, 25.0      # 강건 z 임계, 절대수익률(%) 임계
GAP_BD = 5                      # 이 영업일 수를 넘는 공백은 '공백 구간'으로 기록 (연휴는 보통 ≤4)
STALE_RUN = 5                   # 동일 값이 연속 N회 이상이면 '정체값'
# 검토 결과 실제 시장 이벤트로 확인되어 유지하는 관측치 (정제하지 않음)
KNOWN_REAL = {
    ("yf:CL=F", "2020-04-20"): "WTI 5월물 만기 마이너스 유가",
    ("fred:DCOILWTICO", "2020-04-20"): "WTI 현물 마이너스 유가",
    ("fred:DCOILBRENTEU", "2020-04-21"): "마이너스 유가 다음날 Brent 현물 급락",
    ("yf:RB=F", "2020-04-21"): "마이너스 유가 여파",
    ("yf:NG=F", "2022-01-27"): "NG 2월물 만기일 숏스퀴즈 (+46% 장중)",
    ("fred:DHHNGSP", "2003-02-25"): "2003년 2월 한파 현물 급등",
    ("fred:DHHNGSP", "2018-01-16"): "2018년 1월 한파 현물 급등",
    ("fred:DHHNGSP", "2021-02-17"): "텍사스 한파(Uri) 현물 급등",
    ("fred:DHHNGSP", "2024-01-12"): "2024년 1월 북극한파 현물 급등",
    ("fred:DHHNGSP", "2025-01-17"): "2025년 1월 한파 현물 급등",
    ("fred:VIXCLS", "2011-08-08"): "미국 신용등급 강등 직후",
    ("fred:VIXCLS", "2011-08-09"): "미국 신용등급 강등 직후 반락",
    ("fred:VIXCLS", "2020-03-12"): "코로나 폭락",
    ("fred:VIXCLS", "2014-07-17"): "MH17 격추 당일 급등",
    ("yf:CL=F", "2020-03-18"): "코로나·사우디 증산전쟁 (선물·현물 동시 확인)",
    ("fred:DCOILWTICO", "2020-03-18"): "코로나·사우디 증산전쟁 (선물·현물 동시 확인)",
    ("fred:DCOILWTICO", "2020-03-20"): "코로나·사우디 증산전쟁 (선물·현물 동시 확인)",
    ("fred:DHHNGSP", "2019-03-04"): "3월 초 한파 현물 급등",
    ("fred:DHHNGSP", "2020-09-04"): "연휴 직전 현물 유동성 급감",
}


def series_dict(daily: pd.DataFrame) -> dict[str, pd.Series]:
    out = {}
    for sid, g in daily.groupby("id"):
        s = g.sort_values("date").set_index("date")["value"]
        out[sid] = s
    return out


def returns(s: pd.Series, unit: str) -> pd.Series:
    return A.log_return(s, unit)


def check_series(sid, s_raw, meta, crisis_names):
    name, unit, src = meta["name"], meta["unit"], meta["source"]
    dup = int(s_raw.index.duplicated().sum())
    s = s_raw[~s_raw.index.duplicated(keep="last")].dropna()
    bdays = pd.bdate_range(s.index.min(), s.index.max())
    obs_bd = s.index[s.index.dayofweek < 5]
    weekend = int((s.index.dayofweek >= 5).sum())
    miss_bd = len(bdays.difference(obs_bd))
    # 공백 구간
    pos = np.searchsorted(bdays, obs_bd)
    gaps = []
    for a, b, ia, ib in zip(obs_bd[:-1], obs_bd[1:], pos[:-1], pos[1:]):
        if ib - ia - 1 > GAP_BD:
            gaps.append(dict(id=sid, 품목=name, 공백시작=a + pd.offsets.BDay(1), 공백끝=b - pd.offsets.BDay(1), 누락영업일=int(ib - ia - 1)))
    # 정체값 (동일 값 연속)
    runs = []
    grp = (s != s.shift()).cumsum()
    for _, g in s.groupby(grp):
        if len(g) >= STALE_RUN:
            runs.append(dict(id=sid, 품목=name, 시작=g.index[0], 끝=g.index[-1], 연속횟수=len(g), 값=g.iloc[0]))
    # 이상치
    r = returns(s, unit).dropna()
    mad = (r - r.rolling(125, center=True, min_periods=30).median()).abs().rolling(125, center=True, min_periods=30).median()
    z = r / (1.4826 * mad.replace(0, np.nan))
    is_pct = not A.is_rate(unit) and (s > 0).mean() >= 0.9
    flag = (z.abs() > Z_TH) | ((r.abs() > ABS_TH) if is_pct else False)
    outs = []
    cm = A.crisis_mask(r.index, crisis_names)
    for d in r.index[flag.fillna(False)]:
        i = s.index.get_loc(d)
        prev2 = s.iloc[i - 1] if i >= 1 else np.nan
        nxt = s.iloc[i + 1] if i + 1 < len(s) else np.nan
        rt, rnext = r.loc[d], (r.shift(-1).loc[d] if d in r.index else np.nan)
        reverted = (np.sign(rnext) == -np.sign(rt)) and abs(rnext) > 0.6 * abs(rt) and \
            (abs(nxt / prev2 - 1) * 100 < 0.35 * abs(rt) if is_pct and prev2 > 0 else abs(nxt - prev2) < 0.35 * abs(rt))
        if reverted:
            kind = "데이터 오류 의심 (급등락 후 즉시 복귀)"
        elif cm.loc[d]:
            kind = "위기 구간 이벤트"
        elif sid.startswith("yf:") and sid.endswith("=F"):
            kind = "롤오버 의심 (연속월물)"
        else:
            kind = "검토 필요"
        outs.append(dict(id=sid, 품목=name, 날짜=d, 전일=prev2, 당일=s.loc[d], 익일=nxt,
                         수익률=round(float(rt), 3), 단위="%" if is_pct else "차분", 강건z=round(float(z.loc[d]), 1), 분류=kind))
    summ = dict(id=sid, 품목=name, 그룹=meta["group"], 단위=unit, 소스=src, 시작=s.index.min().date(), 끝=s.index.max().date(),
                관측수=len(s), 영업일수=len(bdays), 누락영업일=miss_bd, 누락률=round(miss_bd / len(bdays) * 100, 2),
                공백구간수=len(gaps), 최장공백=max([g["누락영업일"] for g in gaps], default=0), 주말데이터=weekend,
                중복날짜=dup, 음수_0값=int((s <= 0).sum()), 정체구간수=len(runs), 이상치수=len(outs),
                오류의심=sum(o["분류"].startswith("데이터 오류") for o in outs),
                최대일간수익률=round(float(r.max()), 2), 최소일간수익률=round(float(r.min()), 2),
                첨도=round(float(r.kurt()), 1))
    return summ, gaps, runs, outs


def cross_check(S: dict) -> pd.DataFrame:
    """동일 자산 다른 소스 비교 (선물 근월물 vs 현물 — 베이시스가 있으므로 큰 괴리만 의미 있음)."""
    pairs = [("yf:CL=F", "fred:DCOILWTICO", "WTI 선물(Yahoo) vs WTI 현물(EIA)"),
             ("yf:BZ=F", "fred:DCOILBRENTEU", "Brent 선물(Yahoo) vs Brent 현물(EIA)"),
             ("yf:NG=F", "fred:DHHNGSP", "HH 선물(Yahoo) vs HH 현물(EIA)")]
    rows = []
    for a, b, lbl in pairs:
        if a not in S or b not in S:
            continue
        j = pd.concat([S[a], S[b]], axis=1, keys=["a", "b"]).dropna()
        j = j[(j.a > 0) & (j.b > 0)]
        d = (j.a / j.b - 1) * 100
        for dt, v in d[d.abs() > 10].items():
            rows.append(dict(비교=lbl, 날짜=dt, 소스A=j.a[dt], 소스B=j.b[dt], 괴리율=round(v, 2)))
        rows.append(dict(비교=lbl + " [요약]", 날짜=None, 소스A=len(j), 소스B=None,
                         괴리율=f"중앙값 {d.median():.2f}% / 95%구간 {d.quantile(.025):.1f}~{d.quantile(.975):.1f}% / |괴리|>10%: {int((d.abs()>10).sum())}일"))
    return pd.DataFrame(rows)


def run(crisis_names=None, write=True):
    crisis_names = crisis_names or list(A.CRISES)
    daily = pd.read_parquet(DATA / "daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    meta = daily.groupby("id")[["name", "group", "unit", "source"]].first()
    S = series_dict(daily)
    summ, gaps, runs, outs = [], [], [], []
    for sid, s in S.items():
        a, b, c, d = check_series(sid, s, meta.loc[sid], crisis_names)
        summ.append(a); gaps += b; runs += c; outs += d
    summ = pd.DataFrame(summ).sort_values(["그룹", "품목"])
    gaps, runs, outs = pd.DataFrame(gaps), pd.DataFrame(runs), pd.DataFrame(outs)
    # 처리 상태: 정제 규칙 적용 / 실제 이벤트로 유지 / 미처리
    rules = pd.read_csv(DATA / "cleaning_rules.csv", parse_dates=["date"]) if (DATA / "cleaning_rules.csv").exists() else pd.DataFrame(columns=["id", "date", "reason"])
    rk = {(r.id, r.date.strftime("%Y-%m-%d")): r.reason for r in rules.itertuples()}
    rc = pd.read_csv(DATA / "roll_calendar.csv") if (DATA / "roll_calendar.csv").exists() else pd.DataFrame(columns=["id", "date", "applied"])
    rolls = {(r.id, str(r.date)[:10]) for r in rc.itertuples() if bool(r.applied)}
    def status(r):
        k = (r["id"], pd.Timestamp(r["날짜"]).strftime("%Y-%m-%d"))
        if k in rk: return "정제: 결측 처리 — " + rk[k]
        if k in rolls: return "롤 보정: 연속월물 만기 교체일 — 분석에서는 롤오버 갭 제거"
        if k in KNOWN_REAL: return "유지: 실제 이벤트 — " + KNOWN_REAL[k]
        return "미처리 (분석 시 ±15σ 클리핑만 적용)"
    if len(outs):
        outs["처리 상태"] = outs.apply(status, axis=1)
    xc = cross_check(S)
    # 월간 시리즈 결측
    mon = pd.read_parquet(DATA / "monthly.parquet")
    msum = []
    for sid, g in mon.groupby("id"):
        g = g.sort_values("date"); rng = pd.date_range(g.date.min(), g.date.max(), freq="MS")
        msum.append(dict(id=sid, 품목=g.name.iloc[0], 소스=g.source.iloc[0], 단위=g.unit.iloc[0], 시작=g.date.min().date(),
                         끝=g.date.max().date(), 관측수=len(g), 누락월=len(rng) - g.date.nunique()))
    msum = pd.DataFrame(msum)
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(OUT / "data_quality_report.xlsx", engine="xlsxwriter", datetime_format="yyyy-mm-dd") as w:
            pd.DataFrame({"항목": ["기준", "이상치 기준", "공백 구간", "정체값", "분류 방법"],
                          "설명": [f"생성 {pd.Timestamp.now():%Y-%m-%d %H:%M}, 일간 {len(S)}개 / 월간 {len(msum)}개 시리즈",
                                 f"강건 z(롤링 MAD 125일) |z|>{Z_TH} 또는 |일간수익률|>{ABS_TH}%",
                                 f"{GAP_BD}영업일 초과 연속 누락 (주말·일반 연휴 제외 목적)",
                                 f"동일 값 {STALE_RUN}회 이상 연속",
                                 "급등락 후 즉시 복귀=데이터 오류 의심 / 위기구간=실제 이벤트 / =F=롤오버 의심 / 기타=검토 필요"]}
                         ).to_excel(w, sheet_name="설명", index=False)
            for df, sh in [(summ, "요약_일간"), (msum, "요약_월간"), (outs, "이상치"), (gaps, "공백구간"),
                           (runs, "정체값"), (xc, "소스교차검증")]:
                (df if len(df) else pd.DataFrame({"결과": ["해당 없음"]})).to_excel(w, sheet_name=sh, index=False)
                w.sheets[sh].autofit(); w.sheets[sh].freeze_panes(1, 0)
    return summ, msum, gaps, runs, outs, xc


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
    summ, msum, gaps, runs, outs, xc = run()
    cols = ["품목", "시작", "관측수", "누락률", "공백구간수", "최장공백", "주말데이터", "중복날짜", "음수_0값", "정체구간수", "이상치수", "오류의심", "최대일간수익률", "최소일간수익률"]
    print(summ[cols].to_string(index=False))
    print("\n[이상치 분류]\n", outs["분류"].value_counts().to_string() if len(outs) else "없음")
    print("\n[처리 상태]\n", outs["처리 상태"].str.split(" — ").str[0].value_counts().to_string() if len(outs) else "")
    left = outs[outs["분류"].str.startswith("데이터") & outs["처리 상태"].str.startswith("미처리")]
    print("\n[오류 의심이지만 미처리]\n", left[["품목", "날짜", "전일", "당일", "익일", "수익률"]].to_string(index=False) if len(left) else "없음")
    print("\n[교차검증 요약]\n", xc[xc["비교"].str.contains("요약")][["비교", "괴리율"]].to_string(index=False))
    print("\n[월간 누락]\n", msum[msum["누락월"] > 0].to_string(index=False))
